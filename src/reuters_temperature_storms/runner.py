"""Run daily footprints through tracking and catalog writing."""

from __future__ import annotations

import gzip
import json
import shutil
from datetime import date, datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

from .footprints import WaveFootprintSource
from .records import (
    CompletionContext,
    CompletionEvidence,
    write_generation,
)
from .run_state import CompletedWaveRun, WaveRunRequest
from .series import WaveSeries
from .tracker import (
    DailyComponentBatch,
    WaveTracker,
    WaveTrackerCheckpoint,
    WaveTrackerError,
    WaveTrackingBatch,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from .source import WaveSource

_DEFAULT_SERIES = WaveSeries("tasmax", "warm")


class WaveRunnerError(ValueError):
    """Raised when a footprint run cannot start or resume safely.

    Steps:
        1. Describe a rejected input, configuration or stage result.
        2. Propagate the message through callers without changing data.

    Args:
        *args: Exception message and any additional exception arguments.

    Examples:
        raise WaveRunnerError("invalid input")
    """


def _input_identity(source: WaveSource) -> str:
    """Describe all configured inputs in the existing saved request field.

    Args:
        source: Daily cell source or prepared shapes with a cell companion.

    Returns:
        The original source location for cell-only input, or both explicit
        locations for prepared shapes. Changing either source then prevents
        resuming state calculated from a different complete daily input.

    Examples:
        ``_input_identity(WaveCellSource("./cells"))`` returns ``"./cells"``.
    """
    if not isinstance(source, WaveFootprintSource) or source.cell_source is None:
        return source.location
    return json.dumps(
        {"footprints": source.location, "cells": source.cell_source.location},
        sort_keys=True,
        separators=(",", ":"),
    )


def run_waves(
    source: WaveSource,
    start_date: date,
    end_date: date,
    destination: str | Path,
    progress: Callable[[date, int, int], None] | None = None,
    stop_after_days: int | None = None,
    series: WaveSeries = _DEFAULT_SERIES,
    catalog_progress: Callable[[str], None] | None = None,
    extend_from: str | Path | None = None,
) -> CompletionEvidence | None:
    """Construct daily records, lineage and a catalog from finished inputs.

    Rerunning an interrupted request resumes after the last complete tracked
    date. Inputs already tracked must remain unchanged. A standalone date range
    starts a new history; historical names require tracking from the start.

    Steps:
        1. Check the requested dates, series, output folder and stopping option.
        2. Save a new request or require an existing request to match exactly.
        3. Read saved tracking state and continue after its last completed date.
        4. Read each next day's inputs and advance the tracker in date order.
        5. Save each daily batch before its restart checkpoint and report progress.
        6. Stop with saved state when requested, or write the finished catalogs
           after reaching the last date.

    Args:
        source: Local or remote wave-cell or footprint source for one series.
        start_date: First included date, or the original start when resuming.
        end_date: Last included date, or the original end when resuming.
        extend_from: Completed local output whose saved history is copied into a new
            empty destination and continued to a later end date.
        destination: New local output directory. Its sibling with suffix
            ``-state`` retains tracked daily batches and the restart checkpoint.
        progress: Optional callback receiving each committed date, unnamed wave
            count and named storm count.
        catalog_progress: Optional callback receiving catalog stage and counts.
        stop_after_days: Optional positive count of new days to commit before
            returning. The original full date range remains available for resume.
        series: One maximum/minimum and warm/cold series to track.

    Returns:
        Completion inventory after reaching the requested end, or None
        after stopping early with the committed tracker state retained.

    Raises:
        WaveRunnerError: If dates, output or restart settings disagree.
        FileNotFoundError: If a requested input file is missing.
        ValueError: If input footprints or scientific records are invalid.

    Examples:
        source = WaveFootprintSource(
            "./footprints", cell_source=WaveCellSource("./cells")
        )
        result = run_waves(source, date(1960, 1, 1), date(1960, 1, 6), "./wave-results")
    """
    for value in (start_date, end_date):
        if not isinstance(value, date) or isinstance(value, datetime):
            raise WaveRunnerError("start_date and end_date must be calendar dates")
    if start_date > end_date:
        raise WaveRunnerError("start_date must not follow end_date")
    if stop_after_days is not None and (
        type(stop_after_days) is not int or stop_after_days < 1
    ):
        raise WaveRunnerError("stop_after_days must be a positive integer")
    if (
        not isinstance(series, WaveSeries)
        or series.temperature not in WaveSeries.temperatures
        or series.direction not in WaveSeries.directions
    ):
        raise WaveRunnerError("series must be one supported Reuters series")
    destination = Path(destination).absolute()
    incomplete = destination / "_INCOMPLETE.json"
    if destination.exists() and any(destination.iterdir()) and not incomplete.is_file():
        raise WaveRunnerError(
            "output must be empty; completed output cannot be overwritten"
        )
    state = destination.with_name(destination.name + "-state")
    state.mkdir(parents=True, exist_ok=True)
    batches = state / "batches"
    batches.mkdir(exist_ok=True)
    source_identity = _input_identity(source)
    typed_request = WaveRunRequest(
        source_identity,
        start_date,
        end_date,
        series,
        source.known_empty_dates,
        source.kind,
    )
    if extend_from is not None:
        try:
            CompletedWaveRun.load(Path(extend_from)).copy_for_extension(
                state, typed_request
            )
        except ValueError as error:
            raise WaveRunnerError(str(error)) from error
    request_path = state / "request.json"
    if request_path.exists():
        if WaveRunRequest.load(request_path) != typed_request:
            raise WaveRunnerError(
                "restart input kind, source, date range, series and empty dates must match the original request"
            )
    else:
        _write_state(request_path, typed_request.to_bytes())
    checkpoint_path = state / "checkpoint.json"
    checkpoint = (
        WaveTrackerCheckpoint.from_bytes(checkpoint_path.read_bytes())
        if checkpoint_path.exists()
        else None
    )
    if checkpoint is not None and (
        checkpoint.requested_start != start_date
        or checkpoint.through_date > end_date
        or checkpoint.source_identity != source_identity
        or checkpoint.series != series
    ):
        raise WaveRunnerError("restart checkpoint does not match the original request")
    build_id = typed_request.generation_id
    if incomplete.exists():
        marker = json.loads(incomplete.read_bytes())
        if (
            checkpoint is None
            or checkpoint.through_date != end_date
            or marker != {"generation_id": build_id, "status": "writing"}
        ):
            raise WaveRunnerError("incomplete output does not match this saved run")
        complete = destination / "_COMPLETE.json"
        if complete.exists():
            try:
                complete_marker = json.loads(complete.read_bytes())
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
            else:
                if (
                    not isinstance(complete_marker, dict)
                    or complete_marker.get("generation_id") != build_id
                ):
                    raise WaveRunnerError(
                        "completion marker does not match this saved run"
                    )
        shutil.rmtree(destination)
    next_day = start_date if checkpoint is None else checkpoint.next_date
    tracker = WaveTracker(
        requested_start=next_day,
        source_identity=source_identity,
        prior_checkpoint=checkpoint,
        series=series,
    )
    try:
        current = next_day
        processed = 0
        while current <= end_date:
            batch = tracker.advance(
                DailyComponentBatch(
                    date=current, components=source.read(current, series), series=series
                )
            )
            _write_state(
                batches / f"{current.isoformat()}.json.gz",
                gzip.compress(batch.to_bytes(), compresslevel=1, mtime=0),
            )
            _write_state(checkpoint_path, tracker.checkpoint_bytes())
            checkpoint = batch.checkpoint
            if progress is not None:
                named_count = sum(record.name is not None for record in batch.current)
                progress(current, len(batch.current) - named_count, named_count)
            current += timedelta(days=1)
            processed += 1
            if stop_after_days is not None and processed >= stop_after_days:
                break
    finally:
        tracker.close()
    if checkpoint is None:
        raise WaveRunnerError("no input dates were processed")
    if checkpoint.through_date < end_date:
        return None
    completion = CompletionContext(
        start_date=start_date,
        end_date=end_date,
        package_version=version("reuters-temperature-storms"),
        series=series,
    )
    return write_generation(
        _stored_batches(batches, start_date, end_date, checkpoint),
        destination=destination,
        generation_id=build_id,
        completion=completion,
        progress=catalog_progress,
    )


def run_status(destination: str | Path) -> dict[str, object]:
    """Read local progress and disk use without opening any input data.

    Steps:
        1. Read the saved request, checkpoint and local file sizes.
        2. Calculate processed dates, active counts and completion status without input reads.

    Args:
        destination: Local output directory supplied to run_waves.

    Returns:
        Original date range, checkpoint date, next date, completed-day and batch
        counts, separate active unnamed-wave and named-storm counts, local state
        sizes and output status.

    Raises:
        WaveRunnerError: If the saved run request is missing or malformed.

    Examples:
        status = run_status("./wave-results")
    """
    destination = Path(destination).absolute()
    state = destination.with_name(destination.name + "-state")
    request_path = state / "request.json"
    if not request_path.is_file():
        raise WaveRunnerError("no saved run request exists for this output")
    request = json.loads(request_path.read_bytes())
    if not isinstance(request, dict):
        raise WaveRunnerError("saved run request must be an object")
    start = _document_date(request.get("start_date"), "request start_date")
    end = _document_date(request.get("end_date"), "request end_date")
    if start > end:
        raise WaveRunnerError("saved run start_date must not follow end_date")
    selected = request.get("series")
    if not isinstance(selected, dict):
        raise WaveRunnerError("saved run request must include one series")
    series = _series_from_document(selected)
    checkpoint_path = state / "checkpoint.json"
    try:
        checkpoint_contents = checkpoint_path.read_bytes()
    except FileNotFoundError:
        checkpoint_contents = None
    checkpoint = (
        WaveTrackerCheckpoint.from_bytes(checkpoint_contents)
        if checkpoint_contents is not None
        else None
    )
    if checkpoint is not None and (
        checkpoint.requested_start != start
        or checkpoint.through_date > end
        or checkpoint.source_identity != request.get("source")
        or checkpoint.series != series
    ):
        raise WaveRunnerError("saved checkpoint does not match the original request")
    sizes: dict[Path, int] = {}
    for path in state.rglob("*"):
        if path.name.endswith(".tmp"):
            continue
        try:
            size = path.stat()
        except FileNotFoundError:
            continue
        if path.is_file():
            sizes[path] = size.st_size
    batch_files = tuple(
        path
        for path in sizes
        if path.parent == state / "batches" and path.name.endswith(".json.gz")
    )
    through = None if checkpoint is None else checkpoint.through_date
    next_date = start if checkpoint is None else checkpoint.next_date
    active = () if checkpoint is None else checkpoint.active
    named_count = sum(record.name is not None for record in active)
    return {
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "series": selected,
        "input_kind": request["input_kind"],
        "known_empty_dates": request.get("known_empty_dates", []),
        "through_date": None if through is None else through.isoformat(),
        "next_date": next_date.isoformat() if next_date <= end else None,
        "processed_days": 0 if through is None else (through - start).days + 1,
        "active_waves": len(active) - named_count,
        "active_named_storms": named_count,
        "batch_files": len(batch_files),
        "batch_bytes": sum(sizes[path] for path in batch_files),
        "checkpoint_bytes": 0
        if checkpoint_contents is None
        else len(checkpoint_contents),
        "state_bytes": sum(sizes.values()),
        "output_status": "incomplete"
        if (destination / "_INCOMPLETE.json").is_file()
        else "complete"
        if (destination / "_COMPLETE.json").is_file()
        else "ready_to_write"
        if through == end
        else "tracking",
    }


def _stored_batches(
    root: Path, start: date, end: date, checkpoint: WaveTrackerCheckpoint
) -> Iterator[WaveTrackingBatch]:
    """Read saved tracking results one day at a time.

    Steps:
        1. Visit each requested date in chronological order.
        2. Read and decompress its saved batch, then validate it against the final checkpoint.

    Args:
        root: Local directory of committed daily batch documents.
        start: First included day.
        end: Last included day.
        checkpoint: Final validated tracker state.

    Yields:
        One typed daily batch for the existing catalog writer.

    Examples:
        batches = _stored_batches(path, start, end, checkpoint)
    """
    current = start
    while current <= end:
        try:
            yield WaveTrackingBatch.from_bytes(
                gzip.decompress((root / f"{current.isoformat()}.json.gz").read_bytes()),
                checkpoint,
            )
        except WaveTrackerError as error:
            raise WaveRunnerError(str(error)) from error
        current += timedelta(days=1)


def _write_state(path: Path, contents: bytes) -> None:
    """Replace one complete restart file after its contents have been written.

    Steps:
        1. Write the complete contents to a temporary sibling file.
        2. Replace the destination only after that write succeeds.

    Args:
        path: Destination file owned by the current run.
        contents: Complete encoded file contents.

    Returns:
        None.

    Examples:
        _write_state(path, checkpoint.to_bytes())
    """
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(contents)
    temporary.replace(path)


def _json_bytes(value: object) -> bytes:
    """Encode a deterministic JSON document.

    Steps:
        1. Serialize sorted JSON keys without extra whitespace.
        2. Encode the serialized document as UTF-8 bytes.

    Args:
        value: JSON-compatible document.

    Returns:
        Sorted compact UTF-8 JSON bytes.

    Examples:
        contents = _json_bytes({"source": "./footprints"})
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _document_date(value: object, label: str) -> date:
    """Decode a strict ISO calendar day from saved JSON.

    Steps:
        1. Require ISO date text and parse its calendar date.
        2. Reject alternative spellings and report the supplied field label on errors.

    Args:
        value: Candidate ISO date text.
        label: Field label used in validation errors.

    Returns:
        Parsed calendar date.

    Raises:
        WaveRunnerError: If text is not a strict ISO calendar date.

    Examples:
        day = _document_date("1960-01-01", "batch date")
    """
    if not isinstance(value, str):
        raise WaveRunnerError(f"{label} must be an ISO calendar date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise WaveRunnerError(f"{label} must be an ISO calendar date") from error
    if parsed.isoformat() != value:
        raise WaveRunnerError(f"{label} must be an ISO calendar date")
    return parsed


def _series_from_document(value: Mapping[str, object]) -> WaveSeries:
    """Build one supported series from serialized temperature and direction fields.

    Steps:
        1. Read the serialized temperature and direction fields.
        2. Validate their supported choices and translate failures to WaveRunnerError.

    Args:
        value: Mapping with temperature and direction fields.

    Returns:
        The matching typed temperature series.

    Raises:
        WaveRunnerError: If fields do not name one of four supported series.

    Examples:
        series = _series_from_document({"temperature": "tasmax", "direction": "warm"})
    """
    try:
        return WaveSeries.from_fields(value.get("temperature"), value.get("direction"))
    except ValueError as error:
        raise WaveRunnerError(
            "saved daily record does not name a Reuters branch"
        ) from error
