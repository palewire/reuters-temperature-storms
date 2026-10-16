"""Verify independent parallel footprint jobs before chronological tracking."""

from concurrent.futures.process import BrokenProcessPool
from datetime import date, datetime, timedelta
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from click.testing import CliRunner

import reuters_temperature_storms.footprint_run as footprint_run_module
from reuters_temperature_storms import (
    FootprintRun,
    WaveCells,
    WaveCellSource,
    WaveFootprints,
    WaveFootprintSource,
    WaveSeries,
)
from reuters_temperature_storms.cli import main
from reuters_temperature_storms.footprint_run import FootprintRunError
from reuters_temperature_storms.records import open_completed_generation, scan_wave_days
from tests.test_wave_cells import _frame

DAY = date(2024, 7, 1)
SERIES = WaveSeries("tasmax", "warm")


def _write_inputs(root, *, count=3, series=SERIES):
    """Write independent tiny cell files including a deliberately empty date.

    Args:
        root: Input directory root.
        count: Number of consecutive daily input files.
        series: Temperature and direction represented by the inputs.

    Returns:
        Configured cell source ready for serial or process-based jobs.

    Examples:
        source = _write_inputs(tmp_path / "cells")

    Steps:
        1. Write independent tiny cell files including a deliberately empty date.
    """
    source = WaveCellSource(str(root))
    for offset in range(count):
        day = DAY + timedelta(days=offset)
        positions = (
            ((179.75, 0, "qualified"), (180, 0, "grace"), (-179.75, 0, "fill"))
            if offset == 0
            else ((0, 0, "qualified"), (0.25, 0, "fill"), (0.5, 0, "grace"))
            if offset == 1
            else ()
        )
        WaveCells(_frame(positions, day=day)).to_parquet(source.path_for(day, series))
    return source


def _scientific_frame(path):
    """Read daily output and retain exact geometries and physical scalar values.

    Args:
        path: Written footprint file.

    Returns:
        Frame with WKB columns substituted for shapes for exact comparison.

    Examples:
        frame = _scientific_frame(path)

    Steps:
        1. Read daily output and retain exact geometries and physical scalar values.
    """
    frame = pd.DataFrame(WaveFootprints.read_parquet(path).frame.copy())
    for column in ("geometry", "grace_geometry", "buffer_geometry"):
        frame[column] = gpd.GeoSeries(frame[column], crs=4326).to_wkb()
    return pd.DataFrame(frame)


@pytest.mark.parametrize(
    "series",
    [
        WaveSeries(temperature, direction)
        for temperature in ("tasmax", "tasmin")
        for direction in ("warm", "cold")
    ],
)
def test_real_processes_match_serial_footprints_for_every_series(tmp_path, series):
    """Write the same independent daily files with one and two real processes.

    Args:
        tmp_path: Input and two output roots.
        series: Each supported scientific series.

    Returns:
        None. Files retain identical IDs, WKB and scalar measurements.

    Examples:
        pytest tests/test_footprint_stage.py -k real_processes

    Steps:
        1. Write the same independent daily files with one and two real processes.
        2. Check that files retain identical IDs, WKB and scalar measurements.
    """
    source = _write_inputs(tmp_path / "cells", series=series)
    serial = WaveFootprintSource(str(tmp_path / "serial"))
    parallel = WaveFootprintSource(str(tmp_path / "parallel"))
    end = DAY + timedelta(days=2)
    serial_results = FootprintRun(source, serial, DAY, end, series, max_workers=1).run()
    progress = []
    parallel_results = FootprintRun(
        source, parallel, DAY, end, series, max_workers=2
    ).run(progress=progress.append)
    assert [result.day for result in serial_results] == [
        DAY + timedelta(days=offset) for offset in range(3)
    ]
    assert [result.day for result in parallel_results] == [
        result.day for result in serial_results
    ]
    assert len(progress) == 3
    assert sorted(result.day for result in progress) == [
        result.day for result in serial_results
    ]
    for expected, actual in zip(serial_results, parallel_results, strict=True):
        assert actual.rows == expected.rows
        assert actual.bytes > 0
        assert actual.elapsed_seconds >= 0
        assert not actual.skipped
        assert _scientific_frame(serial.path_for(actual.day, series)).equals(
            _scientific_frame(parallel.path_for(actual.day, series))
        )
        assert serial.read(actual.day, series) == parallel.read(actual.day, series)
    assert parallel_results[-1].rows == 0
    assert not (tmp_path / "parallel" / "_COMPLETE.json").exists()
    assert not (tmp_path / "parallel-state").exists()
    assert not list((tmp_path / "parallel").rglob("all_storms.parquet"))


def test_failed_day_does_not_prevent_other_days_and_retry_skips_valid_files(tmp_path):
    """Retain independent successes and recover only the missing input date.

    Args:
        tmp_path: Independent local stage roots.

    Returns:
        None. Failed dates are named and valid completed files remain untouched.

    Examples:
        pytest tests/test_footprint_stage.py -k failed_day

    Steps:
        1. Retain independent successes and recover only the missing input date.
        2. Check that failed dates are named and valid completed files remain untouched.
    """
    source = _write_inputs(tmp_path / "cells")
    missing = DAY + timedelta(days=1)
    source._filesystem.rm(source.path_for(missing, SERIES))
    output = WaveFootprintSource(str(tmp_path / "output"))
    stage = FootprintRun(
        source, output, DAY, DAY + timedelta(days=2), SERIES, max_workers=2
    )
    with pytest.raises(FootprintRunError, match=str(missing)):
        stage.run()
    assert output._filesystem.exists(output.path_for(DAY, SERIES))
    assert output._filesystem.exists(output.path_for(DAY + timedelta(days=2), SERIES))
    first_bytes = output._filesystem.cat(output.path_for(DAY, SERIES))
    WaveCells(_frame(((0, 0, "qualified"),), day=missing)).to_parquet(
        source.path_for(missing, SERIES)
    )
    results = stage.run()
    assert [result.skipped for result in results] == [True, False, True]
    assert output._filesystem.cat(output.path_for(DAY, SERIES)) == first_bytes
    assert len(output.read(missing, SERIES)) == 1


def test_existing_valid_output_skips_without_reopening_source(tmp_path):
    """Reuse a checked footprint file when its source is temporarily unavailable.

    Args:
        tmp_path: Independent source and output roots.

    Returns:
        None. A valid existing date is checked and skipped without a cell read.

    Examples:
        pytest tests/test_footprint_stage.py -k reopening_source

    Steps:
        1. Reuse a checked footprint file when its source is temporarily unavailable.
        2. Check that a valid existing date is checked and skipped without a cell read.
    """
    source = _write_inputs(tmp_path / "cells", count=1)
    output = WaveFootprintSource(str(tmp_path / "output"))
    stage = FootprintRun(source, output, DAY, DAY, SERIES, max_workers=1)
    (first,) = stage.run()
    source._filesystem.rm(source.path_for(DAY, SERIES))
    (second,) = stage.run()
    assert not first.skipped
    assert second.skipped
    assert second.rows == first.rows
    assert second.bytes == first.bytes


@pytest.mark.parametrize("mutation", ["corrupt", "identity", "date"])
def test_invalid_existing_file_is_refused_until_forced(tmp_path, mutation):
    """Check existing output data before treating a date as finished.

    Args:
        tmp_path: Source and output roots.
        mutation: Corruption, wrong identity or wrong represented date.

    Returns:
        None. Invalid files raise a dated error and explicit force replaces them.

    Examples:
        pytest tests/test_footprint_stage.py -k invalid_existing

    Steps:
        1. Check existing output data before treating a date as finished.
        2. Check that invalid files raise a dated error and explicit force replaces them.
    """
    source = _write_inputs(tmp_path / "cells", count=1)
    output = WaveFootprintSource(str(tmp_path / "output"))
    stage = FootprintRun(source, output, DAY, DAY, SERIES, max_workers=1)
    stage.run()
    path = output.path_for(DAY, SERIES)
    if mutation == "corrupt":
        with output._filesystem.open(path, "wb") as stream:
            stream.write(b"broken parquet")
    else:
        frame = gpd.read_parquet(path)
        if mutation == "identity":
            frame["storm_id"] = "0" * 32
        else:
            frame["date"] = pd.Timestamp(DAY + timedelta(days=1))
        frame.to_parquet(path, index=False)
    corrupted_bytes = output._filesystem.cat(path)
    with pytest.raises(FootprintRunError, match=str(DAY)):
        stage.run()
    assert output._filesystem.cat(path) == corrupted_bytes
    (result,) = FootprintRun(
        source, output, DAY, DAY, SERIES, max_workers=1, force_update=True
    ).run()
    assert not result.skipped
    assert len(output.read(DAY, SERIES)) == 1
    assert output._filesystem.cat(path) != corrupted_bytes


def test_declared_missing_empty_day_creates_explicit_empty_parquet(tmp_path):
    """Turn an explicitly declared absent input day into a real empty output file.

    Args:
        tmp_path: Independent empty source and output roots.

    Returns:
        None. The output includes a readable typed empty daily GeoParquet.

    Examples:
        pytest tests/test_footprint_stage.py -k declared_missing

    Steps:
        1. Turn an explicitly declared absent input day into a real empty output file.
        2. Check that the output includes a readable typed empty daily GeoParquet.
    """
    source = WaveCellSource(str(tmp_path / "cells"), known_empty_dates=(DAY,))
    output = WaveFootprintSource(str(tmp_path / "output"))
    (result,) = FootprintRun(source, output, DAY, DAY, SERIES, max_workers=1).run()
    assert result.rows == 0
    assert result.bytes > 0
    assert output.read(DAY, SERIES) == ()
    assert output._filesystem.exists(output.path_for(DAY, SERIES))


@pytest.mark.parametrize(
    "mutation",
    [
        "source_equals_destination",
        "source_destination_url_alias",
        "source_model",
        "output_model",
        "start_datetime",
        "date_order",
        "series_type",
        "series_value",
        "workers_zero",
        "workers_negative",
        "workers_bool",
        "force_string",
    ],
)
def test_stage_rejects_invalid_configuration_before_writing(tmp_path, mutation):
    """Reject unsafe source selection and unsupported job settings at setup.

    Args:
        tmp_path: Independent local paths.
        mutation: Invalid typed model, date, series or worker setting.

    Returns:
        None. No output is written for invalid stage configuration.

    Examples:
        pytest tests/test_footprint_stage.py -k invalid_configuration

    Steps:
        1. Reject unsafe source selection and unsupported job settings at setup.
        2. Check that no output is written for invalid stage configuration.
    """
    source = _write_inputs(tmp_path / "cells", count=1)
    output = WaveFootprintSource(str(tmp_path / "output"))
    start, end, series, workers, force = DAY, DAY, SERIES, 1, False
    if mutation == "source_equals_destination":
        output = WaveFootprintSource(source.location)
    elif mutation == "source_destination_url_alias":
        output = WaveFootprintSource((tmp_path / "cells").as_uri())
    elif mutation == "source_model":
        source = WaveFootprintSource(source.location)
    elif mutation == "output_model":
        output = WaveCellSource(output.location)
    elif mutation == "start_datetime":
        start = datetime.combine(DAY, datetime.min.time())
    elif mutation == "date_order":
        end = DAY - timedelta(days=1)
    elif mutation == "series_type":
        series = "tasmax/warm"
    elif mutation == "series_value":
        series = WaveSeries("tasmean", "warm")
    elif mutation == "workers_zero":
        workers = 0
    elif mutation == "workers_negative":
        workers = -1
    elif mutation == "workers_bool":
        workers = True
    else:
        force = "yes"
    with pytest.raises(FootprintRunError):
        FootprintRun(
            source, output, start, end, series, max_workers=workers, force_update=force
        ).run()
    assert not (tmp_path / "output").exists()


def test_check_reports_missing_dates_and_validates_existing_files(tmp_path):
    """Check date coverage before one continuous chronological tracking run.

    Args:
        tmp_path: Independent source and destination roots.

    Returns:
        None. Missing dates are reported, complete inputs pass, invalid files fail.

    Examples:
        pytest tests/test_footprint_stage.py -k check_reports

    Steps:
        1. Check date coverage before one continuous chronological tracking run.
        2. Check that missing dates are reported, complete inputs pass, invalid files fail.
    """
    source = _write_inputs(tmp_path / "cells")
    output = WaveFootprintSource(str(tmp_path / "output"))
    end = DAY + timedelta(days=2)
    stage = FootprintRun(source, output, DAY, end, SERIES, max_workers=1)
    assert stage.check() == (DAY, DAY + timedelta(days=1), end)
    FootprintRun(source, output, DAY, DAY, SERIES, max_workers=1).run()
    assert stage.check() == (DAY + timedelta(days=1), end)
    stage.run()
    assert stage.check() == ()
    with output._filesystem.open(output.path_for(end, SERIES), "wb") as stream:
        stream.write(b"not a readable parquet")
    with pytest.raises(FootprintRunError, match=str(end)):
        stage.check()


def test_fsspec_file_urls_work_as_separate_process_output(tmp_path):
    """Read and write caller-selected filesystem URLs in spawned workers.

    Args:
        tmp_path: Local roots represented by explicit file URLs.

    Returns:
        None. Worker jobs produce valid files through the configured filesystem.

    Examples:
        pytest tests/test_footprint_stage.py -k fsspec_file_urls

    Steps:
        1. Read and write caller-selected filesystem URLs in spawned workers.
        2. Check that worker jobs produce valid files through the configured filesystem.
    """
    original = _write_inputs(tmp_path / "cells", count=2)
    source = WaveCellSource((tmp_path / "cells").as_uri())
    output = WaveFootprintSource((tmp_path / "output").as_uri())
    stage = FootprintRun(
        source, output, DAY, DAY + timedelta(days=1), SERIES, max_workers=2
    )
    results = stage.run()
    assert len(results) == 2
    assert stage.check() == ()
    assert output.read(DAY, SERIES) == original.read(DAY, SERIES)


def test_build_check_and_track_cli_pipeline(tmp_path):
    """Build daily files, verify readiness and track them through the real CLI.

    Args:
        tmp_path: Independent source, footprint and tracking output directories.

    Returns:
        None. Only tracking writes catalogs and saved chronological state.

    Examples:
        pytest tests/test_footprint_stage.py -k build_check_and_track

    Steps:
        1. Build daily files, verify readiness and track them through the real CLI.
        2. Check that only tracking writes catalogs and saved chronological state.
    """
    source = _write_inputs(tmp_path / "cells")
    footprint_root, result_root = tmp_path / "footprints", tmp_path / "results"
    runner = CliRunner()
    stage_arguments = [
        "footprints",
        "--wave-cells",
        source.location,
        "--output",
        str(footprint_root),
        "--start-date",
        str(DAY),
        "--end-date",
        str(DAY + timedelta(days=2)),
        "--series",
        "tasmax/warm",
        "--max-workers",
        "1",
    ]
    missing = runner.invoke(main, [*stage_arguments, "--check-only"])
    assert missing.exit_code == 1
    assert "missing footprint dates" in missing.output
    assert not footprint_root.exists()
    built = runner.invoke(main, stage_arguments)
    assert built.exit_code == 0, built.output
    assert "Completed 3 daily footprint files" in built.output
    assert len(list(footprint_root.rglob("*.parquet"))) == 3
    assert not (footprint_root / "all_storms.parquet").exists()
    assert not (footprint_root / "all_waves.parquet").exists()
    assert not (footprint_root / "_COMPLETE.json").exists()
    assert not footprint_root.with_name("footprints-state").exists()
    checked = runner.invoke(main, [*stage_arguments, "--check-only"])
    assert checked.exit_code == 0, checked.output
    assert "Verified complete footprint dates" in checked.output
    tracked = runner.invoke(
        main,
        [
            "run",
            "--wave-footprints",
            str(footprint_root),
            "--output",
            str(result_root),
            "--start-date",
            str(DAY),
            "--end-date",
            str(DAY + timedelta(days=2)),
            "--series",
            "tasmax/warm",
        ],
    )
    assert tracked.exit_code == 0, tracked.output
    assert "Completed wave records" in tracked.output
    assert (result_root / "_COMPLETE.json").exists()
    assert result_root.with_name("results-state").joinpath("checkpoint.json").exists()
    assert gpd.read_parquet(result_root / "all_storms.parquet").empty
    assert len(gpd.read_parquet(result_root / "all_waves.parquet")) == 2
    assert len(scan_wave_days(open_completed_generation(result_root))) == 2


def test_local_replacement_failure_preserves_prior_file_and_cleans_temporary(
    tmp_path, monkeypatch
):
    """Preserve an existing footprint when final local file replacement fails.

    Args:
        tmp_path: Local output root.
        monkeypatch: Fixture simulating interruption at the final file replacement.

    Returns:
        None. Existing bytes remain intact and no hidden temporary file remains.

    Examples:
        pytest tests/test_footprint_stage.py -k replacement_failure

    Steps:
        1. Preserve an existing footprint when final local file replacement fails.
        2. Check that existing bytes remain intact and no hidden temporary file remains.
    """
    target = tmp_path / "footprints.parquet"
    original = WaveFootprints.from_cells(
        WaveCells(_frame(((0, 0, "qualified"),))), DAY, SERIES
    )
    replacement = WaveFootprints.from_cells(
        WaveCells(_frame(((10, 0, "qualified"),))), DAY, SERIES
    )
    original.to_parquet(target)
    previous_bytes = target.read_bytes()

    def fail_replace(self, destination):
        """Raise after confirming the temporary output has been fully written.

        Args:
            self: Temporary file path awaiting replacement.
            destination: Existing completed footprint path.

        Returns:
            Never returns; raises the simulated local write failure.

        Examples:
            Invoked by WaveFootprints.to_parquet at the final replacement.

        Steps:
            1. Raise after confirming the temporary output has been fully written.
        """
        assert self.exists()
        assert Path(destination) == target
        assert WaveFootprints.read_parquet(self).frame.equals(replacement.frame)
        raise OSError("simulated replacement interruption")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replacement interruption"):
        replacement.to_parquet(target)
    assert target.read_bytes() == previous_bytes
    assert WaveFootprints.read_parquet(target).frame.equals(original.frame)
    assert list(tmp_path.iterdir()) == [target]


def test_unexpected_worker_exit_reports_retry_and_preserves_completed_files(
    tmp_path, monkeypatch
):
    """Report recovery guidance if a native worker process exits unexpectedly.

    Args:
        tmp_path: Independent source and destination directories.
        monkeypatch: Fixture simulating a process pool that fails on submission.

    Returns:
        None. Existing outputs survive and the error explains safe retry.

    Examples:
        pytest tests/test_footprint_stage.py -k unexpected_worker_exit

    Steps:
        1. Report recovery guidance if a native worker process exits unexpectedly.
        2. Check that existing outputs survive and the error explains safe retry.
    """
    source = _write_inputs(tmp_path / "cells")
    output = WaveFootprintSource(str(tmp_path / "output"))
    FootprintRun(source, output, DAY, DAY, SERIES, max_workers=1).run()
    prior = output._filesystem.cat(output.path_for(DAY, SERIES))

    class BrokenPool:
        """Context manager imitating a worker pool that exits before submission."""

        def __init__(self, *args, **kwargs):
            """Accept executor configuration without creating processes.

            Args:
                *args: Existing executor arguments.
                **kwargs: Worker, spawn and initialization options.

            Returns:
                None.

            Examples:
                Used in place of ProcessPoolExecutor.

            Steps:
                1. Accept executor configuration without creating processes.
            """

        def __enter__(self):
            """Return this simulated executor.

            Args:
                None.

            Returns:
                The BrokenPool context.

            Examples:
                Called by the footprint run's with statement.

            Steps:
                1. Return this simulated executor.
            """
            return self

        def __exit__(self, *args):
            """Leave the exception visible to the stage recovery handler.

            Args:
                *args: Context exception information.

            Returns:
                False so the simulated process exception propagates.

            Examples:
                Called after failed submission.

            Steps:
                1. Leave the exception visible to the stage recovery handler.
            """
            return False

        def submit(self, *args, **kwargs):
            """Simulate a process exiting before it accepts its task.

            Args:
                *args: Worker callable and optional positional arguments.
                **kwargs: Optional worker keyword arguments.

            Returns:
                Never returns; raises BrokenProcessPool.

            Examples:
                Called when the stage sends its first daily request.

            Steps:
                1. Simulate a process exiting before it accepts its task.
            """
            raise BrokenProcessPool("simulated native worker exit")

    monkeypatch.setattr(footprint_run_module, "ProcessPoolExecutor", BrokenPool)
    with pytest.raises(FootprintRunError, match="stopped unexpectedly") as caught:
        FootprintRun(
            source, output, DAY, DAY + timedelta(days=2), SERIES, max_workers=2
        ).run()
    assert "completed files are retained" in str(caught.value)
    assert "Rerun to finish missing dates" in str(caught.value)
    assert "fewer workers" in str(caught.value)
    assert output._filesystem.cat(output.path_for(DAY, SERIES)) == prior
