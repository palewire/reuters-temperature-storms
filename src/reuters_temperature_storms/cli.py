"""Run temperature-wave tracking from daily cells or footprints."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import click
import fsspec

from .anomaly_run import AnomalyFileSource, AnomalyRun
from .cell_run import CellRun
from .cell_scores import WaveCellScoreSource
from .cell_state import WaveCellStateSource
from .cells import WaveCellSource
from .footprint_run import FootprintRun
from .footprints import WaveFootprintSource
from .runner import run_status, run_waves
from .score_run import ScoreRun
from .series import WaveSeries
from .state_run import StateRun

if TYPE_CHECKING:
    from .source import WaveSource


@click.group(
    help="How to calculate the supersized heat waves tracked by the Reuters Climate Monitor"
)
@click.version_option(package_name="reuters-temperature-storms")
def main() -> None:
    """Expose the daily wave runner and local progress commands.

    Steps:
        1. Expose the registered Click subcommands.
        2. Let Click dispatch the selected command or display help and version information.

    Args:
        None.

    Returns:
        None.

    Examples:
        rts --help
    """


@main.command(help="Write daily wave records, lineage and a catalog.")
@click.option("--wave-footprints", help="Prepared footprint directory or S3 URL.")
@click.option("--wave-cells", help="Mature wave-cell directory or S3 URL.")
@click.option("--profile", help="Caller-selected AWS profile for S3 access.")
@click.option(
    "--series",
    default="tasmax/warm",
    show_default=True,
    type=click.Choice(WaveSeries.choices()),
)
@click.option("--start-date", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end-date", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option(
    "--output", required=True, type=click.Path(path_type=Path, file_okay=False)
)
@click.option(
    "--empty-through",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Explicitly declare an empty lead-in through this date.",
)
@click.option(
    "--stop-after-days",
    type=click.IntRange(min=1),
    help="Save this many new days and stop; rerun the same request to resume.",
)
@click.option("--upload-to", help="Optional S3 directory for finished results.")
@click.option(
    "--extend-from",
    type=click.Path(path_type=Path, file_okay=False),
    help="Completed local output to extend into a new empty output.",
)
def run(
    wave_footprints: str | None,
    wave_cells: str | None,
    profile: str | None,
    series: str,
    start_date: datetime,
    end_date: datetime,
    output: Path,
    empty_through: datetime | None,
    stop_after_days: int | None,
    upload_to: str | None,
    extend_from: Path | None,
) -> None:
    """Track one series chronologically and optionally upload finished files.

    Steps:
        1. Validate input selection, dates and any declared empty lead-in.
        2. Configure caller-selected access and run chronological tracking with saved state or extension.
        3. Upload only finished products when requested and report the result location.

    Args:
        wave_footprints: Directory or URL containing prepared footprint Parquets.
        wave_cells: Directory or URL containing mature wave-cell Parquets.
        profile: Caller-selected AWS profile.
        series: Maximum/minimum and warm/cold series to process.
        start_date: First included input date.
        end_date: Last included input date.
        output: Local result directory with a sibling restart-state directory.
        empty_through: Optional inclusive end of a known-empty lead-in.
        stop_after_days: Optional number of new days before a clean stop.
        upload_to: Optional S3 result directory used only after completion.
        extend_from: Completed output whose saved tracker history is continued.

    Returns:
        None. Prints daily progress and the result location.

    Raises:
        click.ClickException: If inputs, tracking or writing fails.

    Examples:
        rts run --wave-footprints ./footprints --start-date 1960-01-05 --end-date 1960-01-06 --output ./waves
    """
    if (wave_footprints is None) == (wave_cells is None):
        raise click.UsageError(
            "supply exactly one of --wave-footprints or --wave-cells"
        )
    start, end = start_date.date(), end_date.date()
    empty_dates: tuple[date, ...] = ()
    if empty_through is not None:
        last = empty_through.date()
        if not start <= last <= end:
            raise click.ClickException(
                "--empty-through must lie inside the requested dates"
            )
        empty_dates = tuple(
            start + timedelta(days=i) for i in range((last - start).days + 1)
        )
    options: dict[str, object] = {"profile": profile} if profile else {}
    try:
        selected = WaveSeries.parse(series)
        source: WaveSource
        if wave_cells is not None:
            source = WaveCellSource(wave_cells, options, known_empty_dates=empty_dates)
        else:
            assert wave_footprints is not None
            source = WaveFootprintSource(
                wave_footprints, options, known_empty_dates=empty_dates
            )
        result = run_waves(
            source,
            start,
            end,
            output,
            progress=lambda day, waves, storms: _progress(day, waves, storms, selected),
            catalog_progress=lambda message: click.echo(
                f"{selected.temperature}/{selected.direction} {message}", err=True
            ),
            stop_after_days=stop_after_days,
            extend_from=extend_from,
            series=selected,
        )
        if result is not None and upload_to:
            _upload_results(output, upload_to, options)
    except (ValueError, OSError, ImportError) as error:
        raise click.ClickException(str(error)) from error
    if result is None:
        click.echo(f"Stopped with saved tracking state: {output}-state")
    else:
        click.echo(f"Completed wave records: {output}")
        catalog_root = upload_to.rstrip("/") if upload_to else str(output)
        click.echo(f"Named-storm catalog: {catalog_root}/all_storms.parquet")
        click.echo(f"Full wave catalog: {catalog_root}/all_waves.parquet")


@main.command(help="Build complete daily temperature anomalies on independent dates.")
@click.option(
    "--temperatures", required=True, help="Daily temperature directory or URL."
)
@click.option(
    "--climatology", required=True, help="Ordinary calendar-day mean directory or URL."
)
@click.option(
    "--seasonal-distribution",
    required=True,
    help="Adjusted calendar-day distribution directory or URL.",
)
@click.option(
    "--output", required=True, help="Separate daily anomaly directory or URL."
)
@click.option("--profile", help="Caller-selected AWS profile for remote files.")
@click.option("--start-date", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end-date", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option(
    "--max-workers",
    type=click.IntRange(min=1),
    help="Worker processes, defaulting to available CPUs.",
)
@click.option("--force-update", is_flag=True, help="Replace accepted existing outputs.")
@click.option(
    "--check-only", is_flag=True, help="Validate output coverage without input reads."
)
def anomalies(
    temperatures: str,
    climatology: str,
    seasonal_distribution: str,
    output: str,
    profile: str | None,
    start_date: datetime,
    end_date: datetime,
    max_workers: int | None,
    force_update: bool,
    check_only: bool,
) -> None:
    """Save four unthresholded daily scores for subsequent weighted scores.

    Steps:
        1. Configure caller-owned daily and calendar baseline roots.
        2. Validate saved files or calculate independent dates in workers.
        3. Print completed-file statuses and the output location.

    Args:
        temperatures: Complete daily minimum and maximum temperatures.
        climatology: Ordinary calendar means in MM-DD.parquet files.
        seasonal_distribution: Adjusted distributions in MM-DD.parquet files.
        output: Separate root for YYYY-MM-DD.parquet anomalies.
        profile: Caller-selected AWS profile for remote files.
        start_date: First included output date.
        end_date: Last included output date.
        max_workers: Positive process count or available CPUs.
        force_update: Replace existing outputs.
        check_only: Check output dates without reading source tables.

    Returns:
        None. Prints file statuses and completed location.

    Raises:
        click.ClickException: If settings, source grids or output files are invalid.

    Examples:
        rts anomalies --temperatures ./daily --climatology ./normal --seasonal-distribution ./adjusted --output ./anomalies --start-date 2024-07-01 --end-date 2024-07-31
    """
    options: dict[str, object] = {"profile": profile} if profile else {}
    calendar = "{month:02d}-{day:02d}.parquet"
    try:
        job = AnomalyRun(
            AnomalyFileSource(temperatures, options),
            AnomalyFileSource(climatology, options, filename_template=calendar),
            AnomalyFileSource(
                seasonal_distribution, options, filename_template=calendar
            ),
            AnomalyFileSource(output, options),
            start_date.date(),
            end_date.date(),
            max_workers,
            force_update,
        )
        if check_only:
            if force_update:
                raise ValueError("--check-only cannot be combined with --force-update")
            missing = job.check()
            if missing:
                raise ValueError(
                    "missing daily anomaly dates: "
                    + ", ".join(day.isoformat() for day in missing)
                )
            click.echo(
                f"Verified complete daily anomaly dates: {start_date.date()} through {end_date.date()}"
            )
        else:
            results = job.run(
                progress=lambda result: click.echo(
                    f"{result.day}: {result.rows} cells, {result.bytes} bytes, {result.elapsed_seconds:.2f}s ({'reused' if result.skipped else 'written'})"
                )
            )
            click.echo(f"Completed {len(results)} daily anomaly files: {output}")
    except (ValueError, OSError, ImportError) as error:
        raise click.ClickException(str(error)) from error


@main.command(help="Build five-day weighted scores in parallel date blocks.")
@click.option("--daily-scores", required=True, help="Daily score directory or URL.")
@click.option("--output", required=True, help="Weighted score directory or URL.")
@click.option("--profile", help="Caller-selected AWS profile for input and output.")
@click.option(
    "--series",
    default="tasmax/warm",
    show_default=True,
    type=click.Choice(WaveSeries.choices()),
)
@click.option("--start-date", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end-date", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option(
    "--max-workers",
    type=click.IntRange(min=1),
    help="Worker processes; defaults to available CPUs, capped by date blocks.",
)
@click.option(
    "--block-days",
    type=click.IntRange(min=1),
    default=30,
    show_default=True,
    help="Output dates per worker block; adjacent dates share accepted inputs.",
)
@click.option(
    "--force-update",
    is_flag=True,
    help="Replace outputs instead of checking and reusing existing files.",
)
@click.option(
    "--check-only",
    is_flag=True,
    help="Check saved scores and date coverage without reading daily inputs.",
)
def scores(
    daily_scores: str,
    output: str,
    profile: str | None,
    series: str,
    start_date: datetime,
    end_date: datetime,
    max_workers: int | None,
    block_days: int,
    force_update: bool,
    check_only: bool,
) -> None:
    """Save weighted score grids for a later per-cell history stage.

    Steps:
        1. Select one score series, output dates and file-access settings.
        2. Check saved outputs when requested, or assign date blocks to workers.
        3. Require five complete daily inputs ending on each output date. Workers
           retain accepted inputs as their windows move forward by one day.
        4. Print per-date results and the completed weighted-score location.

    Args:
        daily_scores: Root containing flat daily score Parquets.
        output: Root for flat daily five-day weighted score Parquets.
        profile: Caller-selected AWS profile for input and output.
        series: Maximum/minimum and warm/cold score selection.
        start_date: First included output date; requires four earlier inputs.
        end_date: Last included output date.
        max_workers: Worker processes, or available CPUs when omitted.
        block_days: Number of adjacent output dates assigned to each worker.
        force_update: Replace saved scores rather than check and reuse them.
        check_only: Check saved scores and missing dates without daily inputs.

    Returns:
        None. Prints dated file statuses and the completed output location.

    Raises:
        click.ClickException: If settings or dated input/output scores are invalid.

    Examples:
        rts scores --daily-scores ./daily-scores --output ./scores --start-date 2024-07-01 --end-date 2024-07-31 --max-workers 8
    """
    start, end = start_date.date(), end_date.date()
    options: dict[str, object] = {"profile": profile} if profile else {}
    try:
        job = ScoreRun(
            WaveCellScoreSource(daily_scores, options),
            WaveCellScoreSource(output, options),
            start,
            end,
            WaveSeries.parse(series),
            max_workers=max_workers,
            block_days=block_days,
            force_update=force_update,
        )
        if check_only:
            if force_update:
                raise ValueError("--check-only cannot be combined with --force-update")
            missing = job.check()
            if missing:
                raise ValueError(
                    "missing weighted-score dates: "
                    + ", ".join(day.isoformat() for day in missing)
                )
            click.echo(f"Verified complete weighted-score dates: {start} through {end}")
        else:
            results = job.run(
                progress=lambda result: click.echo(
                    f"{result.day}: {result.rows} cell scores, {result.bytes} bytes, {result.elapsed_seconds:.2f}s ({'reused' if result.skipped else 'written'})"
                )
            )
            click.echo(f"Completed {len(results)} daily weighted-score files: {output}")
    except (ValueError, OSError, ImportError) as error:
        raise click.ClickException(str(error)) from error


@main.command(help="Build daily wave-cell histories chronologically from scores.")
@click.option("--wave-scores", required=True, help="Calculated-score directory or URL.")
@click.option("--output", required=True, help="Daily state directory or URL.")
@click.option("--profile", help="Caller-selected AWS profile for input and output.")
@click.option(
    "--series",
    default="tasmax/warm",
    show_default=True,
    type=click.Choice(WaveSeries.choices()),
)
@click.option("--start-date", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end-date", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option(
    "--seed",
    is_flag=True,
    help="Explicitly initialize the first requested date without earlier history.",
)
@click.option(
    "--seed-lead-in",
    is_flag=True,
    help="With --seed, count four earlier weighted-score dates; save from start-date.",
)
@click.option(
    "--force-update",
    is_flag=True,
    help="Rebuild the whole requested sequence instead of reusing a saved prefix.",
)
@click.option(
    "--check-only",
    is_flag=True,
    help="Check saved state tables and date coverage without reading scores.",
)
def state(
    wave_scores: str,
    output: str,
    profile: str | None,
    series: str,
    start_date: datetime,
    end_date: datetime,
    seed: bool,
    force_update: bool,
    check_only: bool,
    seed_lead_in: bool,
) -> None:
    """Save one continuous sequence of per-cell qualifying-day histories.

    Steps:
        1. Select the calculated-score root, output root, dates and series.
        2. Check saved state only when requested, or reuse a continuous saved
           prefix and then advance one complete date at a time.
        3. Require the exact preceding state for continuation unless the first
           requested date was explicitly selected as a fresh seed.
        4. With seed_lead_in, count the four preceding weighted-score dates in
           memory and save only the requested dates. For January 1, prepare
           weighted scores from December 28, requiring daily scores from
           December 24. Start footprints and tracking on January 1 to keep
           storm identities and naming inside the retained archive.
        5. Print each saved date and the completed state-file location.

    Args:
        wave_scores: Root containing flat daily calculated-score Parquets.
        output: Root for daily state files beneath field/direction directories.
        profile: Caller-selected AWS profile for input and output.
        series: Maximum/minimum and warm/cold selection.
        start_date: First included calendar date.
        end_date: Last included calendar date.
        seed: Explicitly initialize the first date without earlier history.
        force_update: Rebuild every requested date rather than reuse a prefix.
        seed_lead_in: Count four earlier score dates without saving earlier outputs.
        check_only: Check saved state and missing dates without opening scores.

    Returns:
        None. Prints dated statuses and the completed output location.

    Raises:
        click.ClickException: If settings, score inputs or saved states are invalid.

    Examples:
        rts state --wave-scores ./scores --output ./state --start-date 2024-07-01 --end-date 2024-07-31 --seed
        rts state --wave-scores ./scores --output ./state --start-date 1960-01-01 --end-date 1960-01-31 --seed --seed-lead-in
    """
    start, end = start_date.date(), end_date.date()
    options: dict[str, object] = {"profile": profile} if profile else {}
    try:
        job = StateRun(
            WaveCellScoreSource(wave_scores, options),
            WaveCellStateSource(output, options),
            start,
            end,
            WaveSeries.parse(series),
            seed,
            force_update,
            seed_lead_in=seed_lead_in,
        )
        if check_only:
            if force_update or seed or seed_lead_in:
                raise ValueError(
                    "--check-only cannot be combined with --force-update, --seed or --seed-lead-in"
                )
            missing = job.check()
            if missing:
                raise ValueError(
                    "missing wave-state dates: "
                    + ", ".join(day.isoformat() for day in missing)
                )
            click.echo(f"Verified complete wave-state dates: {start} through {end}")
        else:
            results = job.run(
                progress=lambda result: click.echo(
                    f"{result.day}: {result.rows} cell histories, {result.bytes} bytes, {result.elapsed_seconds:.2f}s ({'reused' if result.skipped else 'written'})"
                )
            )
            click.echo(f"Completed {len(results)} daily wave-state files: {output}")
    except (ValueError, OSError, ImportError) as error:
        raise click.ClickException(str(error)) from error


@main.command(help="Build reusable daily mature wave cells in parallel.")
@click.option(
    "--wave-state", required=True, help="Per-cell wave-state directory or URL."
)
@click.option("--output", required=True, help="Daily cell directory or URL.")
@click.option("--profile", help="Caller-selected AWS profile for input and output.")
@click.option(
    "--series",
    default="tasmax/warm",
    show_default=True,
    type=click.Choice(WaveSeries.choices()),
)
@click.option("--start-date", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end-date", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option(
    "--max-workers",
    type=click.IntRange(min=1),
    help="Worker processes; defaults to available CPUs, capped by dates.",
)
@click.option(
    "--closing-radius",
    type=click.IntRange(min=0),
    default=3,
    show_default=True,
    help="Grid-cell radius for filling small spatial breaks; zero disables filling.",
)
@click.option(
    "--force-update",
    is_flag=True,
    help="Replace outputs instead of checking and reusing existing files.",
)
@click.option(
    "--check-only",
    is_flag=True,
    help="Check saved cells and date coverage without reading wave state or building.",
)
def cells(
    wave_state: str,
    output: str,
    profile: str | None,
    series: str,
    start_date: datetime,
    end_date: datetime,
    max_workers: int | None,
    closing_radius: int,
    force_update: bool,
    check_only: bool,
) -> None:
    """Prepare independent daily cells for a later footprint-generation stage.

    Steps:
        1. Select the date range, series and caller's file-access settings.
        2. Build each complete global date in a separate worker, or check existing
           cell files without reading state inputs when check-only is requested.
        3. Print small per-date file results and the final output location.
           Missing state files remain errors, including early immature dates.

    Args:
        wave_state: Root of daily per-cell wave-state Parquets.
        output: Local or remote root for daily cell GeoParquets.
        profile: Caller-selected AWS access for input and output.
        series: Maximum/minimum and warm/cold selection.
        start_date: First included calendar date.
        end_date: Last included calendar date.
        max_workers: Processes, or available CPUs when omitted.
        closing_radius: Grid-cell radius for filling small spatial breaks.
        force_update: Replace existing files rather than check and reuse.
        check_only: Check saved cell tables and report absent dates without building.

    Returns:
        None. Prints daily results and the completed output directory.

    Raises:
        click.ClickException: If a setting or dated input/output is invalid.

    Examples:
        rts cells --wave-state ./state --output ./cells --start-date 2024-07-01 --end-date 2024-07-31 --max-workers 8
    """
    start, end = start_date.date(), end_date.date()
    options: dict[str, object] = {"profile": profile} if profile else {}
    try:
        job = CellRun(
            WaveCellStateSource(wave_state, options),
            WaveCellSource(output, options),
            start,
            end,
            WaveSeries.parse(series),
            max_workers,
            force_update,
            closing_radius,
        )
        if check_only:
            if force_update:
                raise ValueError("--check-only cannot be combined with --force-update")
            missing = job.check()
            if missing:
                raise ValueError(
                    "missing wave-cell dates: "
                    + ", ".join(day.isoformat() for day in missing)
                )
            click.echo(f"Verified complete wave-cell dates: {start} through {end}")
        else:
            results = job.run(
                progress=lambda result: click.echo(
                    f"{result.day}: {result.rows} wave cells, {result.bytes} bytes, {result.elapsed_seconds:.2f}s ({'reused' if result.skipped else 'written'})"
                )
            )
            click.echo(f"Completed {len(results)} daily wave-cell files: {output}")
    except (ValueError, OSError, ImportError) as error:
        raise click.ClickException(str(error)) from error


@main.command(help="Build reusable daily wave footprints in parallel before tracking.")
@click.option("--wave-cells", required=True, help="Mature wave-cell directory or URL.")
@click.option(
    "--output",
    required=True,
    help="Footprint directory or URL; shared by fleet date ranges.",
)
@click.option("--profile", help="Caller-selected AWS profile for input and output.")
@click.option(
    "--series",
    default="tasmax/warm",
    show_default=True,
    type=click.Choice(WaveSeries.choices()),
)
@click.option("--start-date", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end-date", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option(
    "--max-workers",
    type=click.IntRange(min=1),
    help="Worker processes; defaults to available CPUs, capped by dates.",
)
@click.option(
    "--force-update",
    is_flag=True,
    help="Replace outputs instead of checking and reusing existing files.",
)
@click.option(
    "--empty-through",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Explicit empty lead-in; writes an empty file for every date.",
)
@click.option(
    "--check-only",
    is_flag=True,
    help="Check complete output coverage without reading wave cells or building.",
)
def footprints(
    wave_cells: str,
    output: str,
    profile: str | None,
    series: str,
    start_date: datetime,
    end_date: datetime,
    max_workers: int | None,
    force_update: bool,
    empty_through: datetime | None,
    check_only: bool,
) -> None:
    """Build independent dates into saved files for a later chronological run.

    Steps:
        1. Validate dates and configure cell inputs and footprint outputs.
        2. Check existing products when requested, or build independent dates in workers.
        3. Report missing or failed dates through Click and print completed output status.

    Args:
        wave_cells: Root of finished mature wave-cell inputs.
        output: Local or remote root for daily footprint GeoParquets.
        profile: Caller-selected AWS access for both roots.
        series: Maximum/minimum and warm/cold selection.
        start_date: First included date.
        end_date: Last included date.
        max_workers: Worker processes or available CPUs when omitted.
        force_update: Whether to replace existing files.
        empty_through: Inclusive end of explicitly empty leading dates.
        check_only: Check saved footprints and report missing dates without building.

    Returns:
        None. Prints per-date file status and the final output location.

    Raises:
        click.ClickException: If a date fails or expected output is missing or invalid.

    Examples:
        rts footprints --wave-cells ./cells --output ./footprints --start-date 2024-07-01 --end-date 2024-07-31 --max-workers 8
    """
    start, end = start_date.date(), end_date.date()
    empty_dates: tuple[date, ...] = ()
    if empty_through is not None:
        last = empty_through.date()
        if not start <= last <= end:
            raise click.ClickException(
                "--empty-through must lie inside the requested dates"
            )
        empty_dates = tuple(
            start + timedelta(days=i) for i in range((last - start).days + 1)
        )
    options: dict[str, object] = {"profile": profile} if profile else {}
    try:
        job = FootprintRun(
            WaveCellSource(wave_cells, options, known_empty_dates=empty_dates),
            WaveFootprintSource(output, options),
            start,
            end,
            WaveSeries.parse(series),
            max_workers,
            force_update,
        )
        if check_only:
            if force_update:
                raise ValueError("--check-only cannot be combined with --force-update")
            missing = job.check()
            if missing:
                raise ValueError(
                    "missing footprint dates: "
                    + ", ".join(day.isoformat() for day in missing)
                )
            click.echo(f"Verified complete footprint dates: {start} through {end}")
        else:
            results = job.run(
                progress=lambda result: click.echo(
                    f"{result.day}: {result.rows} wave footprints, {result.bytes} bytes, {result.elapsed_seconds:.2f}s ({'reused' if result.skipped else 'written'})"
                )
            )
            click.echo(f"Completed {len(results)} daily footprint files: {output}")
    except (ValueError, OSError, ImportError) as error:
        raise click.ClickException(str(error)) from error


@main.command(help="Show local checkpoint progress and disk use.")
@click.option(
    "--output", required=True, type=click.Path(path_type=Path, file_okay=False)
)
def status(output: Path) -> None:
    """Print progress for a saved local run.

    Steps:
        1. Read progress from the saved local run.
        2. Print JSON status or translate validation and read failures to Click errors.

    Args:
        output: Result directory supplied to the run command.

    Returns:
        None. Prints local progress as JSON.

    Examples:
        rts status --output ./waves
    """
    try:
        click.echo(json.dumps(run_status(output), indent=2))
    except (ValueError, OSError) as error:
        raise click.ClickException(str(error)) from error


def _upload_results(output: Path, destination: str, options: dict[str, object]) -> None:
    """Upload finished files with the completion marker last.

    Steps:
        1. Require an S3 destination and configure caller-selected filesystem access.
        2. Upload every finished file except the completion marker.
        3. Upload the completion marker last and report the destination.

    Args:
        output: Completed local result directory.
        destination: Caller-selected S3 result URL.
        options: Caller-selected fsspec access options.

    Returns:
        None. Writes files beneath the destination.

    Raises:
        ValueError: If the destination is not an S3 URL.

    Examples:
        _upload_results(Path("./waves"), "s3://bucket/results", {})
    """
    if not destination.startswith("s3://"):
        raise ValueError("--upload-to must be an S3 URL")
    fs, root = fsspec.core.url_to_fs(destination, **options)
    marker = output / "_COMPLETE.json"
    for path in sorted(output.rglob("*")):
        if path.is_file() and path != marker:
            fs.put_file(
                str(path), f"{root.rstrip('/')}/{path.relative_to(output).as_posix()}"
            )
    fs.put_file(str(marker), f"{root.rstrip('/')}/_COMPLETE.json")
    click.echo(f"Uploaded finished records: {destination}")


def _progress(day: date, waves: int, storms: int, series: WaveSeries) -> None:
    """Print one saved day's unnamed waves and named storms.

    Steps:
        1. Choose heat or cold wording from the series direction.
        2. Print the completed date and separate unnamed-wave and named-storm counts.

    Args:
        day: Completed date.
        waves: Active unnamed wave count.
        storms: Active named storm count.
        series: Tracked series, whose direction selects heat or cold wording.

    Returns:
        None.

    Examples:
        _progress(date(1960, 1, 5), 78, 2, WaveSeries("tasmax", "warm"))
    """
    term = "heat" if series.direction == "warm" else "cold"
    click.echo(f"{day}: {waves} active {term} waves, {storms} named {term} storms")
