"""Tests for streamed scientific wave records and lightweight dataset readers."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, timedelta
from threading import get_ident
from typing import TYPE_CHECKING, cast, overload

import duckdb
import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from shapely.geometry import MultiPolygon, Polygon, box

from reuters_temperature_storms import records
from reuters_temperature_storms._wave_tracking import (
    EndedWave,
    LineageEdge,
    TrackedWaveComponent,
    WaveSeries,
)
from reuters_temperature_storms.components import (
    ComponentMembership,
    DailyWaveComponentGeometry,
)
from reuters_temperature_storms.records import (
    CompletedGeneration,
    CompletionContext,
    Lineage,
    NamedStorms,
    WaveDays,
    WaveRecordStorageError,
    Waves,
    open_completed_generation,
    scan_lineage,
    scan_wave_days,
    scan_waves,
    write_generation,
)
from reuters_temperature_storms.schemas import (
    LineageSchema,
    NamedStormsSchema,
    WaveCatalogSchema,
    WaveDayFileSchema,
    WaveDaySchema,
)
from reuters_temperature_storms.tracker import (
    WaveTrackerCheckpoint,
    WaveTrackingBatch,
)

if TYPE_CHECKING:
    from pathlib import Path
    from typing import BinaryIO


@overload
def _values(frame: pd.DataFrame) -> list[dict[str, object]]: ...


@overload
def _values(frame: pd.Series) -> list[object]: ...


def _values(frame: pd.DataFrame | pd.Series) -> list[dict[str, object]] | list[object]:
    """Normalize pandas date/null types for existing scientific expectations.

    Args:
        frame: Selected output DataFrame or Series.

    Returns:
        Row dictionaries or scalar values with Python dates and absent dates as None.

    Examples:
        dates = _values(days["date"])

    Steps:
        1. Convert timestamps to Python dates and missing timestamps to None.
        2. Apply that conversion to each table row or series value.
    """

    def scalar(value: object) -> object:
        """Express a pandas scalar in its existing Python representation.

        Args:
            value: Output scalar.

        Returns:
            Python date, None for NaT, or the unchanged scientific value.

        Examples:
            scalar(pd.Timestamp("2024-01-01"))

        Steps:
            1. Return None for a missing timestamp.
            2. Convert a timestamp to its Python date, leaving other scalar values unchanged.
        """
        if value is pd.NaT:
            return None
        return value.date() if isinstance(value, pd.Timestamp) else value

    if isinstance(frame, pd.DataFrame):
        return [
            {key: scalar(value) for key, value in row.items()}
            for row in frame.to_dict("records")
        ]
    return [scalar(value) for value in frame.tolist()]


_IDS = tuple(f"20240101-{number:032x}" for number in range(1, 20))
_SERIES = WaveSeries("tasmax", "warm")
_CROSS_YEAR_IDS = tuple(
    f"{('20250101' if number == 12 else '20241231')}-{number:032x}"
    for number in range(1, 20)
)


@pytest.mark.component
def test_readers_distinguish_keys_with_the_same_short_prefix(tmp_path: Path) -> None:
    """Select different full digests even when their first eight digits match.

    Args:
        tmp_path: Private generation directory supplied by pytest.

    Returns:
        None. Each filter returns only its selected permanent wave.

    Examples:
        Run pytest tests/test_records.py -k same_short_prefix.

    Steps:
        1. Write two distinct keys sharing the formerly shortened suffix.
        2. Select each complete key through the public catalog reader.
        3. Reject an old shortened key as an invalid selection.
    """
    first, second = _CROSS_YEAR_IDS[:2]
    assert first[:17] == second[:17]
    write_generation(
        _batches(),
        destination=tmp_path / "generation",
        generation_id="full-digest-selection",
        completion=_completion("full-digest-selection"),
    )
    generation = open_completed_generation(tmp_path / "generation")
    for key in (first, second):
        assert scan_waves(generation, wave_ids=[key])["wave_id"].tolist() == [key]
    with pytest.raises(WaveRecordStorageError, match="full-digest"):
        scan_waves(generation, wave_ids=[first[:17]])


@pytest.mark.component
def test_readers_close_files_before_returning_dataframes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read through the caller filesystem and return data with closed file handles.

    Args:
        tmp_path: Private generation directory.
        monkeypatch: Fixture recording which thread opens each input file.

    Returns:
        None.

    Examples:
        pytest tests/test_records.py -k readers_close_files

    Steps:
        1. Read through the caller filesystem and return data with closed file handles.
    """
    write_generation(
        _batches(),
        destination=tmp_path / "generation",
        generation_id="synchronous-scan",
        completion=_completion("synchronous-scan"),
        row_group_rows=2,
        file_rows=3,
    )
    generation = open_completed_generation(tmp_path / "generation")
    caller = get_ident()
    opened: list[BinaryIO] = []
    original_open = generation.filesystem.open

    def open_on_caller_thread(
        path: str, mode: str = "rb", **kwargs: object
    ) -> BinaryIO:
        """Require Python filesystem callbacks to run on the consuming thread.

        Args:
            path: Selected manifest object path.
            mode: Read mode.
            **kwargs: Remaining filesystem options.

        Returns:
            Tracked binary file handle.

        Examples:
            The record reader invokes this for each selected row group.

        Steps:
            1. Require Python filesystem callbacks to run on the consuming thread.
        """
        assert get_ident() == caller
        handle = cast("BinaryIO", original_open(path, mode, **kwargs))
        opened.append(handle)
        return handle

    monkeypatch.setattr(generation.filesystem, "open", open_on_caller_thread)
    for scan, count in ((scan_wave_days, 7), (scan_waves, 6), (scan_lineage, 1)):
        result = scan(generation)
        assert len(result) == count
        assert opened
        assert all(handle.closed for handle in opened)
        assert isinstance(result, pd.DataFrame)
    filtered = scan_wave_days(
        generation,
        temperature="tasmax",
        direction="warm",
        start=date(2025, 1, 1),
        end=date(2025, 1, 1),
        wave_ids=[_CROSS_YEAR_IDS[0]],
    )
    assert len(filtered) == 1
    assert _values(filtered["wave_id"]) == [_CROSS_YEAR_IDS[0]]
    assert _values(filtered.loc[:, ["date"]]) == [{"date": date(2025, 1, 1)}]
    assert len(scan_waves(generation, active=True)) == 2
    assert len(scan_waves(generation, active=False)) == 4
    for empty in (
        scan_waves(generation, wave_ids=[]),
        scan_waves(generation, wave_ids=["20300101-000000000000000000000000ffffffff"]),
    ):
        assert len(empty) == 0
        assert isinstance(empty, gpd.GeoDataFrame)
        assert empty.geometry.name == "lifetime_geometry"
        assert empty.crs is not None
        assert empty.crs.to_epsg() == 4326
    for ids in (
        [_CROSS_YEAR_IDS[9]],
        [_CROSS_YEAR_IDS[11]],
        [_CROSS_YEAR_IDS[9], _CROSS_YEAR_IDS[11]],
    ):
        assert len(scan_lineage(generation, wave_ids=ids)) == 1
    for empty in (
        scan_wave_days(generation, wave_ids=[]),
        scan_wave_days(generation, start=date(2030, 1, 1)),
    ):
        assert len(empty) == 0
        assert isinstance(empty, gpd.GeoDataFrame)
        assert empty.geometry.name == "full_geometry"
    assert all(handle.closed for handle in opened)


def _completion(
    generation_id: str, start: date = date(2024, 12, 31), end: date = date(2025, 1, 1)
) -> CompletionContext:
    """Return a requested series and date range for storage tests.

    Args:
        generation_id: Scenario label passed to the storage fixture.
        start: Requested first observation date.
        end: Requested last observation date.

    Returns:
        Requested series and observation dates for writing the fixture generation.

    Steps:
        1. Construct the storage request using the requested dates and fixed test series.

    Examples:
        Used by ``pytest tests/test_records.py`` for fixture construction and output checks.
    """
    return CompletionContext(
        start_date=start, end_date=end, series=_SERIES, package_version="test"
    )


def _record(
    series: WaveSeries,
    day: date,
    wave_id: str,
    component_id: str,
    *,
    start: date | None = None,
    geometry: Polygon | MultiPolygon | None = None,
    qualified: Polygon | MultiPolygon | None = None,
    grace: Polygon | MultiPolygon | None = None,
    parents: tuple[str, ...] = (),
    null_qualified: bool = False,
) -> TrackedWaveComponent:
    """Create a compact tracker record suitable for storage-boundary tests.

    Args:
        series: Temperature and direction combination under test.
        day: Observation date for the synthetic tracking input.
        wave_id: Persistent wave identifier to store or select.
        component_id: Daily geometry identifier normalized into a 32-character hash.
        start: Requested first observation date.
        geometry: Geographic polygon whose equal-area measurement is requested.
        qualified: Optional qualified polygon; defaults to the full polygon.
        grace: Optional grace-period polygon.
        parents: Wave identifiers recorded as parents of this component.
        null_qualified: Whether to omit the qualified polygon from the record.

    Returns:
        Tracked component with the requested identity, geometry roles, and dates.

    Steps:
        1. Normalize the daily component identifier to a 32-character hexadecimal hash.
        2. Construct qualified, grace, and full geometry roles for the requested date.
        3. Return the tracked wave with its requested identity, start date, and parents.

    Examples:
        Used by ``pytest tests/test_records.py`` for fixture construction and output checks.
    """
    component_id = f"{int(component_id.split('-')[-1], 16):032x}"
    full = geometry or box(170, -1, 180, 1)
    daily = DailyWaveComponentGeometry(
        time=np.datetime64(day, "D"),
        label=1,
        geometry=None
        if null_qualified
        else qualified
        if qualified is not None
        else full,
        grace_geometry=grace,
        full_geometry=full,
        area_qualified=2.0,
        area_grace=0.0 if grace is None else 1.0,
        area_gap_filled=0.0,
        qualified_fraction=1.0,
    )
    membership = ComponentMembership(component_id=component_id, geometry=daily)
    return TrackedWaveComponent(
        series=series,
        component=membership,
        component_id=component_id,
        wave_id=wave_id,
        start_date=np.datetime64(start or day, "D"),
        peak_qualified_area=2.0,
        peak_date=np.datetime64(day, "D"),
        continues=start is not None,
        overlap_percent=80.0 if start is not None else 0.0,
        parent_wave_ids=parents,
    )


def _batch(
    day: date,
    current: tuple[TrackedWaveComponent, ...],
    *,
    ended: tuple[EndedWave, ...] = (),
    lineage: tuple[LineageEdge, ...] = (),
) -> WaveTrackingBatch:
    """Create a storage-ready batch without testing tracker checkpoint behavior.

    Args:
        day: Observation date for the synthetic tracking input.
        current: Active wave records for this observation date.
        ended: Wave endings recorded by this observation date.
        lineage: Split and merge relationships recorded on this date.

    Returns:
        Daily active records, endings, and lineage ready for storage.

    Steps:
        1. Collect the active records, endings, and lineage for the requested date.
        2. Construct the corresponding checkpoint and daily tracking batch.

    Examples:
        Used by ``pytest tests/test_records.py`` for fixture construction and output checks.
    """
    return WaveTrackingBatch(
        date=day,
        current=current,
        ended=ended,
        lineage=lineage,
        checkpoint=cast("WaveTrackerCheckpoint", None),
    )


def _batches() -> tuple[WaveTrackingBatch, ...]:
    """Return one series with year crossing, wave endings and merge lineage.

    Returns:
        Chronological batches containing the fixture movements and lineage events.

    Steps:
        1. Construct wave records crossing the year boundary.
        2. Build consecutive batches containing continuations, endings, and merge relationships.

    Examples:
        Used by ``pytest tests/test_records.py`` for fixture construction and output checks.
    """
    first_day = date(2024, 12, 31)
    second_day = date(2025, 1, 1)
    records = tuple(
        _record(_SERIES, first_day, _CROSS_YEAR_IDS[index], _CROSS_YEAR_IDS[index + 4])
        for index in range(4)
    )
    source = _record(_SERIES, first_day, _CROSS_YEAR_IDS[9], _CROSS_YEAR_IDS[10])
    continued = _record(
        _SERIES,
        second_day,
        _CROSS_YEAR_IDS[0],
        _CROSS_YEAR_IDS[8],
        start=first_day,
        geometry=MultiPolygon([box(170, -1, 180, 1), box(-180, -1, -170, 1)]),
        null_qualified=True,
    )
    child = _record(
        _SERIES,
        second_day,
        _CROSS_YEAR_IDS[11],
        _CROSS_YEAR_IDS[12],
        parents=(_CROSS_YEAR_IDS[9],),
        grace=box(175, -1, 176, 1),
    )
    merged = EndedWave(
        source, np.datetime64(first_day, "D"), "merged", _CROSS_YEAR_IDS[11]
    )
    ended = tuple(
        EndedWave(record, np.datetime64(first_day, "D"), "died", None)
        for record in records[1:]
    )
    edge = LineageEdge(
        _SERIES,
        _CROSS_YEAR_IDS[9],
        _CROSS_YEAR_IDS[11],
        np.datetime64(second_day, "D"),
        "merge",
    )
    return (
        _batch(first_day, records + (source,)),
        _batch(second_day, (continued, child), ended=(merged, *ended), lineage=(edge,)),
    )


_LINEAGE_BRANCH = _SERIES
_LINEAGE_DAY_1 = date(2025, 1, 5)
_LINEAGE_DAY_2 = date(2025, 1, 6)
_LINEAGE_DAY_3 = date(2025, 1, 7)
_PARENT = "20250105-00000000000000000000000000000002"
_CHILD_A = "20250107-00000000000000000000000000000003"
_CHILD_B = "20250107-00000000000000000000000000000004"
_ABSORBED = "20250105-00000000000000000000000000000005"
_KEEPER = "20250105-00000000000000000000000000000006"


def _lineage_completion(generation_id: str) -> CompletionContext:
    """Return the requested series and dates for the lineage scenario.

    Args:
        generation_id: Scenario label passed to the storage fixture.

    Returns:
        Requested series and observation dates for writing the fixture generation.

    Steps:
        1. Construct the completion request spanning the three lineage fixture dates.

    Examples:
        Used by ``pytest tests/test_records.py`` for fixture construction and output checks.
    """
    return _completion(generation_id, _LINEAGE_DAY_1, _LINEAGE_DAY_3)


def _lineage_batches() -> tuple[WaveTrackingBatch, ...]:
    """Return three days holding one real split and one real merge.

    The split parent runs on the first two days and then breaks apart into two
    new waves, so it has no wave-day row on the split event date. The merge
    source is absorbed on the second day by a wave that was already running,
    so it has no wave-day row on the merge event date either. Both shapes come
    straight from the tracker, which reads the previous day's waves and writes
    them as lineage sources against the event day's targets.

    Returns:
        Chronological batches containing the fixture movements and lineage events.

    Steps:
        1. Create the initial split parent, absorbed wave, and continuing merge target.
        2. Merge the absorbed wave on the second date and split the parent on the third.
        3. Return all three chronological batches with their ending records and lineage edges.

    Examples:
        Used by ``pytest tests/test_records.py`` for fixture construction and output checks.
    """
    parent_day_one = _record(
        _LINEAGE_BRANCH, _LINEAGE_DAY_1, _PARENT, _IDS[8], geometry=box(0, 0, 5, 1)
    )
    absorbed = _record(
        _LINEAGE_BRANCH, _LINEAGE_DAY_1, _ABSORBED, _IDS[9], geometry=box(10, 0, 12, 1)
    )
    keeper_day_one = _record(
        _LINEAGE_BRANCH, _LINEAGE_DAY_1, _KEEPER, _IDS[10], geometry=box(12, 0, 15, 1)
    )
    parent_day_two = _record(
        _LINEAGE_BRANCH,
        _LINEAGE_DAY_2,
        _PARENT,
        _IDS[11],
        start=_LINEAGE_DAY_1,
        geometry=box(0, 0, 5, 1),
    )
    keeper_day_two = _record(
        _LINEAGE_BRANCH,
        _LINEAGE_DAY_2,
        _KEEPER,
        _IDS[12],
        start=_LINEAGE_DAY_1,
        geometry=box(10, 0, 15, 1),
        parents=(_ABSORBED,),
    )
    keeper_day_three = _record(
        _LINEAGE_BRANCH,
        _LINEAGE_DAY_3,
        _KEEPER,
        _IDS[15],
        start=_LINEAGE_DAY_1,
        geometry=box(10, 0, 15, 1),
    )
    children = tuple(
        (
            _record(
                _LINEAGE_BRANCH,
                _LINEAGE_DAY_3,
                wave_id,
                component_id,
                geometry=geometry,
                parents=(_PARENT,),
            )
            for wave_id, component_id, geometry in (
                (_CHILD_A, _IDS[13], box(0, 0, 2, 1)),
                (_CHILD_B, _IDS[14], box(3, 0, 5, 1)),
            )
        )
    )
    return (
        _batch(_LINEAGE_DAY_1, (parent_day_one, absorbed, keeper_day_one)),
        _batch(
            _LINEAGE_DAY_2,
            (parent_day_two, keeper_day_two),
            ended=(
                EndedWave(
                    absorbed, np.datetime64(_LINEAGE_DAY_1, "D"), "merged", _KEEPER
                ),
            ),
            lineage=(
                LineageEdge(
                    _LINEAGE_BRANCH,
                    _ABSORBED,
                    _KEEPER,
                    np.datetime64(_LINEAGE_DAY_2, "D"),
                    "merge",
                ),
            ),
        ),
        _batch(
            _LINEAGE_DAY_3,
            (*children, keeper_day_three),
            ended=(
                EndedWave(
                    parent_day_two, np.datetime64(_LINEAGE_DAY_2, "D"), "died", None
                ),
            ),
            lineage=tuple(
                LineageEdge(
                    _LINEAGE_BRANCH,
                    _PARENT,
                    child,
                    np.datetime64(_LINEAGE_DAY_3, "D"),
                    "split",
                )
                for child in (_CHILD_A, _CHILD_B)
            ),
        ),
    )


def _observed_dates(generation: CompletedGeneration, wave_id: str) -> list[date]:
    """Return the sorted wave-day dates recorded for one persisted wave.

    Args:
        generation: Completed output whose persisted daily records are queried.
        wave_id: Persistent wave identifier to store or select.

    Returns:
        Sorted observation dates for the selected persistent wave.

    Steps:
        1. Read daily records for the selected wave from the completed generation.
        2. Convert the dates to Python values and return them in sorted order.

    Examples:
        Used by ``pytest tests/test_records.py`` for fixture construction and output checks.
    """
    days = scan_wave_days(generation, wave_ids=(wave_id,))
    return sorted(cast("list[date]", _values(days["date"])))


def _numpy_record(record: TrackedWaveComponent) -> TrackedWaveComponent:
    """Return one tracked record whose dates use the tracker's NumPy days.

    Args:
        record: Tracked component whose date fields are converted to NumPy days.

    Returns:
        Tracked component with the requested identity, geometry roles, and dates.

    Steps:
        1. Replace the tracked start date, peak date, and component date with NumPy day values.
        2. Keep all other record fields unchanged.

    Examples:
        Used by ``pytest tests/test_records.py`` for fixture construction and output checks.
    """
    return replace(
        record,
        start_date=np.datetime64(record.start_date, "D"),
        peak_date=np.datetime64(record.peak_date, "D"),
        component=replace(
            record.component,
            geometry=replace(
                record.component.geometry,
                time=np.datetime64(record.component.geometry.time, "D"),
            ),
        ),
    )


def _numpy_dated(batch: WaveTrackingBatch) -> WaveTrackingBatch:
    """Return one batch shaped exactly like live tracker output.

    The staged month fragments a resumed run replays are written from tracker
    output, which carries NumPy days rather than the calendar dates these
    storage tests build directly. Converting first lets a test push a scenario
    through the internal month encoder and decoder the resume path uses.

    Args:
        batch: Daily tracking result whose records and relationships are converted.

    Returns:
        Daily active records, endings, and lineage ready for storage.

    Steps:
        1. Convert the batch date and all active and ended record dates to NumPy days.
        2. Convert lineage event dates while preserving wave identities and relationships.

    Examples:
        Used by ``pytest tests/test_records.py`` for fixture construction and output checks.
    """
    return replace(
        batch,
        current=tuple(_numpy_record(record) for record in batch.current),
        ended=tuple(
            replace(
                ending,
                final_record=_numpy_record(ending.final_record),
                end_date=np.datetime64(ending.end_date, "D"),
            )
            for ending in batch.ended
        ),
        lineage=tuple(
            replace(edge, event_date=np.datetime64(edge.event_date, "D"))
            for edge in batch.lineage
        ),
    )


@pytest.mark.unit
def test_schemas_define_file_and_partition_columns() -> None:
    """Pandera makes physical file fields and restored partition fields explicit.

    Args:
        None.

    Returns:
        None.

    Examples:
        pytest tests/test_records.py -k schemas_define

    Steps:
        1. Pandera makes physical file fields and restored partition fields explicit.
    """
    daily = WaveDaySchema.to_schema().columns
    physical = WaveDayFileSchema.to_schema().columns
    assert set(daily) - set(physical) == {
        "temperature",
        "direction",
        "observation_year",
    }
    assert WaveCatalogSchema.to_schema().columns["end_date"].nullable
    assert not LineageSchema.to_schema().columns["relationship"].nullable


@pytest.mark.unit
def test_write_read_filter_and_compaction_are_deterministic(tmp_path) -> None:
    """Two incoming current-record orders produce matching compact Parquet files.

    Args:
        tmp_path: Pytest-owned temporary directory for input and output files.

    Returns:
        None.

    Steps:
        1. Two incoming current-record orders produce matching compact Parquet files.

    Examples:
        Run ``pytest tests/test_records.py -k write_read_filter_and_compaction_are_deterministic``.
    """
    first = tmp_path / "one"
    second = tmp_path / "two"
    inventory = write_generation(
        _batches(),
        destination=first,
        generation_id="one",
        completion=_completion("one"),
        row_group_rows=1,
        file_rows=2,
    )
    reordered = (
        _batch(
            _batches()[0].date,
            tuple(reversed(_batches()[0].current)),
            ended=_batches()[0].ended,
        ),
        _batches()[1],
    )
    second_inventory = write_generation(
        reordered,
        destination=second,
        generation_id="two",
        completion=_completion("two"),
        row_group_rows=1,
        file_rows=2,
    )
    assert inventory.inventory.rows == second_inventory.inventory.rows == 14
    assert (
        inventory.inventory.datasets["wave_days"].objects
        < inventory.inventory.datasets["wave_days"].rows
    )
    first_files = sorted(first.rglob("*.parquet"))
    second_files = sorted(second.rglob("*.parquet"))
    assert [path.relative_to(first) for path in first_files] == [
        path.relative_to(second) for path in second_files
    ]
    assert [path.read_bytes() for path in first_files] == [
        path.read_bytes() for path in second_files
    ]
    generation = open_completed_generation(first)
    days = scan_wave_days(
        generation, temperature="tasmax", direction="warm", start=date(2025, 1, 1)
    )
    assert isinstance(days, gpd.GeoDataFrame)
    assert days.geometry.name == "full_geometry"
    assert _values(days["wave_id"]) == [_CROSS_YEAR_IDS[0], _CROSS_YEAR_IDS[11]]
    assert _values(days["qualified_geometry"])[0] is None
    assert _values(scan_waves(generation, active=False)["wave_id"]) == [
        _CROSS_YEAR_IDS[1],
        _CROSS_YEAR_IDS[2],
        _CROSS_YEAR_IDS[3],
        _CROSS_YEAR_IDS[9],
    ]
    assert len(scan_waves(generation, wave_ids=(_CROSS_YEAR_IDS[0],))) == 1
    assert len(scan_lineage(generation, wave_ids=(_CROSS_YEAR_IDS[9],))) == 1
    row_count = duckdb.sql(
        "SELECT count(*) FROM read_parquet(?, hive_partitioning = true)",
        params=[str(first / "wave_days" / "**" / "*.parquet")],
    ).fetchone()
    assert row_count == (7,)


@pytest.mark.unit
def test_generation_assembly_streams_once_with_row_bounded_buffers(
    tmp_path, monkeypatch
) -> None:
    """Assembly consumes a lazy source once and never flushes above file_rows.

    Args:
        tmp_path: Pytest-owned temporary directory for input and output files.
        monkeypatch: Pytest fixture replacing dependencies during the check.

    Yields:
        Prepared batches in observation order.

    Steps:
        1. Assembly consumes a lazy source once and never flushes above file_rows.

    Examples:
        Run ``pytest tests/test_records.py -k generation_assembly_streams_once_with_row_bounded_buffers``.
    """
    day = date(2024, 1, 1)
    batches = tuple(
        _batch(
            day + timedelta(days=offset),
            (
                _record(
                    _SERIES,
                    day + timedelta(days=offset),
                    _IDS[0],
                    _IDS[offset + 4],
                    start=day if offset else None,
                ),
            ),
        )
        for offset in range(3)
    )
    yielded: list[date] = []

    def source():
        """Yield each batch exactly once without exposing a replayable sequence.

        Yields:
            Prepared batches in observation order.

        Steps:
            1. Yield the prepared tracking batches once, preserving their observation order.

        Examples:
            Used by ``pytest tests/test_records.py`` for fixture construction and output checks.
        """
        for batch in batches:
            yielded.append(batch.date)
            yield batch

    original_flush = records._PartitionWriter._flush

    def bounded_flush(writer, key):
        """Assert file buffers remain capped before their Parquet conversion.

        Args:
            writer: Streaming record writer being observed by the surrounding test.
            key: Partition key whose buffer is checked before flushing.

        Returns:
            None.

        Steps:
            1. Check the selected partition buffer does not exceed its configured row limit.
            2. Delegate the flush to the original writer implementation.

        Examples:
            Used by ``pytest tests/test_records.py`` for fixture construction and output checks.
        """
        assert len(writer.buffers[key]) <= 1
        original_flush(writer, key)

    monkeypatch.setattr(records._PartitionWriter, "_flush", bounded_flush)
    write_generation(
        source(),
        destination=tmp_path / "bounded",
        generation_id="bounded",
        completion=_completion("bounded", day, day + timedelta(days=2)),
        row_group_rows=1,
        file_rows=1,
    )
    assert yielded == [batch.date for batch in batches]
    generation = open_completed_generation(tmp_path / "bounded")
    assert len(scan_wave_days(generation)) == 3


@pytest.mark.unit
def test_chronological_assembly_flushes_underfull_historical_partitions(
    tmp_path, monkeypatch
) -> None:
    """Historical branch-years do not accumulate partial partition buffers.

    Args:
        tmp_path: Pytest-owned temporary directory for input and output files.
        monkeypatch: Pytest fixture replacing dependencies during the check.

    Yields:
        Prepared batches in observation order.

    Steps:
        1. Historical branch-years do not accumulate partial partition buffers.

    Examples:
        Run ``pytest tests/test_records.py -k chronological_assembly_flushes_underfull_historical_partitions``.
    """
    start = date(2020, 1, 1)
    end = date(2024, 12, 31)
    original_append = records._PartitionWriter.append
    active_rows: list[int] = []

    def monitored_append(writer, table, row):
        """Capture the aggregate buffer size after each streamed append.

        Args:
            writer: Streaming record writer being observed by the surrounding test.
            table: Record table receiving the appended row.
            row: Scientific record appended to the streaming writer.

        Returns:
            None.

        Steps:
            1. Append the scientific row through the original writer implementation.
            2. Record the combined size of the writer's remaining partition buffers.

        Examples:
            Used by ``pytest tests/test_records.py`` for fixture construction and output checks.
        """
        original_append(writer, table, row)
        active_rows.append(writer.active_rows)
        assert writer.active_rows <= writer.max_active_rows

    def source():
        """Yield a continuous stream with one underfull wave partition per year.

        Yields:
            Prepared batches in observation order.

        Steps:
            1. Yield the prepared tracking batches once, preserving their observation order.

        Examples:
            Used by ``pytest tests/test_records.py`` for fixture construction and output checks.
        """
        current = start
        while current <= end:
            if current.month == 1 and current.day == 1:
                year_id = f"{current.year}0101-{3000 - current.year:032x}"
                yield _batch(
                    current,
                    (_record(_SERIES, current, year_id, f"{current.year + 10:064x}"),),
                )
            elif current.month == 1 and current.day == 2:
                previous = current - timedelta(days=1)
                yield _batch(
                    current,
                    (),
                    ended=(
                        EndedWave(
                            _record(
                                _SERIES,
                                previous,
                                f"{previous.year}0101-{3000 - previous.year:032x}",
                                f"{previous.year + 10:064x}",
                            ),
                            np.datetime64(previous, "D"),
                            "died",
                            None,
                        ),
                    ),
                )
            else:
                yield _batch(current, ())
            current += timedelta(days=1)

    monkeypatch.setattr(records._PartitionWriter, "append", monitored_append)
    write_generation(
        source(),
        destination=tmp_path / "historical",
        generation_id="historical",
        completion=_completion("historical", start, end),
        row_group_rows=10,
        file_rows=10,
    )
    assert max(active_rows) == 1
    generation = open_completed_generation(tmp_path / "historical")
    assert len(scan_wave_days(generation)) == 5
    assert len(scan_waves(generation)) == 5


@pytest.mark.unit
def test_split_and_merge_sources_are_kept_on_the_day_before_the_event(tmp_path) -> None:
    """A parent that breaks apart is a valid split source without an event-day row.

    Args:
        tmp_path: Pytest-owned temporary directory for input and output files.

    Returns:
        None.

    Steps:
        1. A parent that breaks apart is a valid split source without an event-day row.

    Examples:
        Run ``pytest tests/test_records.py -k split_and_merge_sources_are_kept_on_the_day_before_the_event``.
    """
    destination = tmp_path / "lineage"
    write_generation(
        _lineage_batches(),
        destination=destination,
        generation_id="lineage",
        completion=_lineage_completion("lineage"),
    )
    generation = open_completed_generation(destination)
    edges = sorted(
        (
            edge["relationship"],
            edge["source_wave_id"],
            edge["target_wave_id"],
            edge["event_date"],
        )
        for edge in _values(scan_lineage(generation))
    )
    assert edges == [
        ("merge", _ABSORBED, _KEEPER, _LINEAGE_DAY_2),
        ("split", _PARENT, _CHILD_A, _LINEAGE_DAY_3),
        ("split", _PARENT, _CHILD_B, _LINEAGE_DAY_3),
    ]
    assert _observed_dates(generation, _PARENT) == [_LINEAGE_DAY_1, _LINEAGE_DAY_2]
    assert _observed_dates(generation, _CHILD_A) == [_LINEAGE_DAY_3]
    assert _observed_dates(generation, _CHILD_B) == [_LINEAGE_DAY_3]
    assert _observed_dates(generation, _ABSORBED) == [_LINEAGE_DAY_1]
    assert _observed_dates(generation, _KEEPER) == [
        _LINEAGE_DAY_1,
        _LINEAGE_DAY_2,
        _LINEAGE_DAY_3,
    ]
    catalog = {row["wave_id"]: row for row in _values(scan_waves(generation))}
    assert catalog[_ABSORBED]["end_date"] == _LINEAGE_DAY_1
    assert catalog[_ABSORBED]["end_reason"] == "merged"
    assert catalog[_ABSORBED]["merged_into_wave_id"] == _KEEPER
    assert catalog[_PARENT]["end_date"] == _LINEAGE_DAY_2
    assert catalog[_PARENT]["duration_days"] == 2
    assert catalog[_KEEPER]["parent_keys"] == []
    assert catalog[_KEEPER]["duration_days"] == 3
    for wave_id, expected in (
        (_PARENT, box(0, 0, 5, 1)),
        (_CHILD_A, box(0, 0, 2, 1)),
        (_CHILD_B, box(3, 0, 5, 1)),
        (_ABSORBED, box(10, 0, 12, 1)),
        (_KEEPER, box(10, 0, 15, 1)),
    ):
        assert catalog[wave_id]["lifetime_geometry"].equals(expected)
    merger_day = _values(
        scan_wave_days(
            generation, wave_ids=(_KEEPER,), start=_LINEAGE_DAY_2, end=_LINEAGE_DAY_2
        )
    )[0]
    assert merger_day["parent_keys"] == [_ABSORBED, _KEEPER]
    for child_key in (_CHILD_A, _CHILD_B):
        assert catalog[child_key]["parent_keys"] == [_PARENT]
        assert catalog[child_key]["start_date"] == _LINEAGE_DAY_3


@pytest.mark.unit
def test_resumed_and_compacted_lineage_output_stays_valid(tmp_path) -> None:
    """Restored batches retain identical split and merge rows across several files.

    Args:
        tmp_path: Pytest-owned temporary directory for input and output files.

    Returns:
        None.

    Steps:
        1. Restored batches retain identical split and merge rows across several files.

    Examples:
        Run ``pytest tests/test_records.py -k resumed_and_compacted_lineage_output_stays_valid``.
    """
    checkpoint = WaveTrackerCheckpoint(
        requested_start=_LINEAGE_DAY_1,
        through_date=_LINEAGE_DAY_3,
        source_identity="synthetic-footprints",
        series=_SERIES,
        active=(),
    )
    restored = tuple(
        WaveTrackingBatch.from_bytes(_numpy_dated(batch).to_bytes(), checkpoint)
        for batch in _lineage_batches()
    )
    destination = tmp_path / "resumed"
    inventory = write_generation(
        restored,
        destination=destination,
        generation_id="resumed",
        completion=_lineage_completion("resumed"),
        row_group_rows=1,
        file_rows=1,
    )
    assert inventory.inventory.datasets["lineage"].objects == 3
    generation = open_completed_generation(destination)
    assert sorted(
        (edge["source_wave_id"], edge["target_wave_id"], edge["event_date"])
        for edge in _values(scan_lineage(generation, wave_ids=(_PARENT,)))
    ) == [(_PARENT, _CHILD_A, _LINEAGE_DAY_3), (_PARENT, _CHILD_B, _LINEAGE_DAY_3)]


@pytest.mark.unit
def test_split_source_that_keeps_running_is_still_accepted(tmp_path) -> None:
    """A parent observed on both days remains a valid split source.

    Args:
        tmp_path: Pytest-owned temporary directory for input and output files.

    Returns:
        None.

    Steps:
        1. A parent observed on both days remains a valid split source.

    Examples:
        Run ``pytest tests/test_records.py -k split_source_that_keeps_running_is_still_accepted``.
    """
    child_key = "20250106-00000000000000000000000000000003"
    parent_day_one = _record(
        _LINEAGE_BRANCH, _LINEAGE_DAY_1, _PARENT, _IDS[8], geometry=box(0, 0, 5, 1)
    )
    parent_day_two = _record(
        _LINEAGE_BRANCH,
        _LINEAGE_DAY_2,
        _PARENT,
        _IDS[9],
        start=_LINEAGE_DAY_1,
        geometry=box(0, 0, 3, 1),
    )
    child = _record(
        _LINEAGE_BRANCH,
        _LINEAGE_DAY_2,
        child_key,
        _IDS[10],
        geometry=box(4, 0, 5, 1),
        parents=(_PARENT,),
    )
    destination = tmp_path / "continuing"
    write_generation(
        (
            _batch(_LINEAGE_DAY_1, (parent_day_one,)),
            _batch(
                _LINEAGE_DAY_2,
                (parent_day_two, child),
                lineage=(
                    LineageEdge(
                        _LINEAGE_BRANCH,
                        _PARENT,
                        child_key,
                        np.datetime64(_LINEAGE_DAY_2, "D"),
                        "split",
                    ),
                ),
            ),
        ),
        destination=destination,
        generation_id="continuing",
        completion=_completion("continuing", _LINEAGE_DAY_1, _LINEAGE_DAY_2),
    )
    generation = open_completed_generation(destination)
    assert _observed_dates(generation, _PARENT) == [_LINEAGE_DAY_1, _LINEAGE_DAY_2]
    assert _observed_dates(generation, child_key) == [_LINEAGE_DAY_2]


@pytest.mark.unit
def test_merge_target_may_be_new_on_the_event_date(tmp_path) -> None:
    """A merge into a brand-new wave keeps its source on the preceding day.

    Args:
        tmp_path: Pytest-owned temporary directory for input and output files.

    Returns:
        None.

    Steps:
        1. A merge into a brand-new wave keeps its source on the preceding day.

    Examples:
        Run ``pytest tests/test_records.py -k merge_target_may_be_new_on_the_event_date``.
    """
    target_key = "20250106-00000000000000000000000000000006"
    absorbed = _record(
        _LINEAGE_BRANCH, _LINEAGE_DAY_1, _ABSORBED, _IDS[8], geometry=box(0, 0, 5, 1)
    )
    target = _record(
        _LINEAGE_BRANCH,
        _LINEAGE_DAY_2,
        target_key,
        _IDS[9],
        geometry=box(0, 0, 8, 1),
        parents=(_ABSORBED,),
    )
    destination = tmp_path / "new-target"
    write_generation(
        (
            _batch(_LINEAGE_DAY_1, (absorbed,)),
            _batch(
                _LINEAGE_DAY_2,
                (target,),
                ended=(
                    EndedWave(
                        absorbed,
                        np.datetime64(_LINEAGE_DAY_1, "D"),
                        "merged",
                        target_key,
                    ),
                ),
                lineage=(
                    LineageEdge(
                        _LINEAGE_BRANCH,
                        _ABSORBED,
                        target_key,
                        np.datetime64(_LINEAGE_DAY_2, "D"),
                        "merge",
                    ),
                ),
            ),
        ),
        destination=destination,
        generation_id="new-target",
        completion=_completion("new-target", _LINEAGE_DAY_1, _LINEAGE_DAY_2),
    )
    generation = open_completed_generation(destination)
    assert _observed_dates(generation, _ABSORBED) == [_LINEAGE_DAY_1]
    assert _observed_dates(generation, target_key) == [_LINEAGE_DAY_2]


@pytest.mark.component
@pytest.mark.parametrize("ended", [False, True])
def test_catalog_lifetime_covers_pre_name_qualified_visits_once(
    tmp_path: Path, ended: bool
) -> None:
    """Store the complete qualified union without repeated or grace-only coverage.

    Args:
        tmp_path: Independent output directory supplied by pytest.
        ended: Whether the wave ends or remains active at the requested cutoff.

    Returns:
        None. The area and geographic lifetime shape survive GeoParquet reading.

    Examples:
        Run ``pytest tests/test_records.py -k lifetime_covers``.

    Steps:
        1. Store the complete qualified union without repeated or grace-only coverage.
        2. Check that the area and geographic lifetime shape survive GeoParquet reading.
    """
    first = date(2024, 1, 1)
    first_visit, next_visit = box(0, 0, 2, 1), box(1, 0, 3, 1)
    grace_only = box(30, 0, 31, 1)
    rows: list[TrackedWaveComponent] = []
    for offset, geometry in enumerate(
        (first_visit, next_visit, first_visit, grace_only)
    ):
        day = first + timedelta(days=offset)
        row = _record(
            _SERIES,
            day,
            _IDS[0],
            _IDS[offset + 4],
            start=first if offset else None,
            geometry=geometry,
        )
        if offset == 3:
            row = replace(
                row,
                component=replace(
                    row.component,
                    geometry=replace(
                        row.component.geometry,
                        geometry=None,
                        grace_geometry=grace_only,
                        area_qualified=0.0,
                        area_grace=2.0,
                        qualified_fraction=0.0,
                    ),
                ),
            )
        rows.append(replace(row, name="Example 2024" if offset else None))
    batches = [
        _batch(first + timedelta(days=offset), (row,))
        for offset, row in enumerate(rows)
    ]
    cutoff = first + timedelta(days=3)
    if ended:
        event = cutoff + timedelta(days=1)
        batches.append(
            _batch(
                event,
                (),
                ended=(EndedWave(rows[-1], np.datetime64(cutoff), "died", None),),
            )
        )
        cutoff = event
    destination = tmp_path / "lifetime"
    write_generation(
        batches,
        destination=destination,
        generation_id="lifetime",
        completion=_completion("lifetime", first, cutoff),
    )
    generation = open_completed_generation(destination)
    catalog = scan_waves(generation)
    expected = box(0, 0, 3, 1)
    expected_area = float(
        gpd.GeoSeries([expected], crs="EPSG:4326").to_crs("EPSG:6933").area.iloc[0]
        / 1_000_000
    )
    assert isinstance(catalog, gpd.GeoDataFrame)
    assert catalog.geometry.name == "lifetime_geometry"
    assert catalog.crs is not None
    assert catalog.crs.to_epsg() == 4326
    wave = catalog.iloc[0]
    assert wave["lifetime_geometry"].equals(expected)
    assert wave["total_area_visited_km2"] == pytest.approx(expected_area, rel=1e-12)
    assert wave["name"] == "Example 2024"
    assert wave["duration_days"] == wave["observation_count"] == 4
    assert bool(wave["active"]) is not ended
    assert wave["lifetime_geometry"].disjoint(grace_only)
    entry = next(
        item for item in generation.manifest["files"] if item["table"] == "waves"
    )
    stored = gpd.read_parquet(destination / entry["path"])
    assert stored.geometry.name == "lifetime_geometry"
    assert stored.crs is not None
    assert stored.crs.to_epsg() == 4326
    assert stored.geometry.iloc[0].equals(expected)


@pytest.mark.component
def test_catalog_merge_lifetime_does_not_import_source_history(tmp_path: Path) -> None:
    """A surviving identity counts its own observations rather than ancestor shapes.

    Args:
        tmp_path: Independent output directory supplied by pytest.

    Returns:
        None. Each catalog geometry contains only its recorded qualified history.

    Examples:
        Run ``pytest tests/test_records.py -k merge_lifetime``.

    Steps:
        1. A surviving identity counts its own observations rather than ancestor shapes.
        2. Check that each catalog geometry contains only its recorded qualified history.
    """
    first = date(2024, 1, 1)
    second = first + timedelta(days=1)
    source_old = box(0, 0, 1, 1)
    source_latest = box(10, 0, 11, 1)
    survivor_original = box(11, 0, 12, 1)
    source = _record(_SERIES, first, _IDS[0], _IDS[4], geometry=source_old)
    survivor = _record(_SERIES, first, _IDS[1], _IDS[5], geometry=survivor_original)
    source_last = _record(
        _SERIES, second, _IDS[0], _IDS[6], start=first, geometry=source_latest
    )
    survivor_second = _record(
        _SERIES, second, _IDS[1], _IDS[7], start=first, geometry=survivor_original
    )
    merge_day = second + timedelta(days=1)
    merged_current = _record(
        _SERIES,
        merge_day,
        _IDS[1],
        _IDS[8],
        start=first,
        geometry=box(10, 0, 12, 1),
        parents=(_IDS[0],),
    )
    destination = tmp_path / "merged-lifetime"
    write_generation(
        (
            _batch(first, (source, survivor)),
            _batch(second, (source_last, survivor_second)),
            _batch(
                merge_day,
                (merged_current,),
                ended=(
                    EndedWave(source_last, np.datetime64(second), "merged", _IDS[1]),
                ),
                lineage=(
                    LineageEdge(
                        _SERIES, _IDS[0], _IDS[1], np.datetime64(merge_day), "merge"
                    ),
                ),
            ),
        ),
        destination=destination,
        generation_id="merged-lifetime",
        completion=_completion("merged-lifetime", first, merge_day),
    )
    catalog = scan_waves(open_completed_generation(destination)).set_index("wave_id")
    assert catalog.loc[_IDS[0], "lifetime_geometry"].equals(
        source_old.union(source_latest)
    )
    assert catalog.loc[_IDS[1], "lifetime_geometry"].equals(box(10, 0, 12, 1))
    assert catalog.loc[_IDS[1], "lifetime_geometry"].disjoint(source_old)


@pytest.mark.component
def test_catalog_lifetime_preserves_dateline_parts_and_holes(tmp_path: Path) -> None:
    """Export distinct dateline pieces and retain areas never visited within holes.

    Args:
        tmp_path: Independent output directory supplied by pytest.

    Returns:
        None. Geographic geometry preserves the separated components and hole.

    Examples:
        Run ``pytest tests/test_records.py -k dateline_parts``.

    Steps:
        1. Export distinct dateline pieces and retain areas never visited within holes.
        2. Check that geographic geometry preserves the separated components and hole.
    """
    first = date(2024, 1, 1)
    eastern = Polygon(
        box(178, 0, 180, 2).exterior,
        holes=[box(178.5, 0.5, 179.5, 1.5).exterior],
    )
    western = box(-180, 0, -179, 2)
    shapes = (eastern, western)
    destination = tmp_path / "dateline"
    write_generation(
        tuple(
            _batch(
                first + timedelta(days=offset),
                (
                    _record(
                        _SERIES,
                        first + timedelta(days=offset),
                        _IDS[0],
                        _IDS[offset + 4],
                        start=first if offset else None,
                        geometry=shape,
                    ),
                ),
            )
            for offset, shape in enumerate(shapes)
        ),
        destination=destination,
        generation_id="dateline",
        completion=_completion("dateline", first, first + timedelta(days=1)),
    )
    wave = scan_waves(open_completed_generation(destination)).iloc[0]
    expected = MultiPolygon(shapes)
    actual = wave["lifetime_geometry"]
    assert actual.equals(expected)
    assert isinstance(actual, MultiPolygon)
    assert len(actual.geoms) == 2
    assert sum(len(part.interiors) for part in actual.geoms) == 1
    expected_area = float(
        gpd.GeoSeries([expected], crs="EPSG:4326").to_crs("EPSG:6933").area.iloc[0]
        / 1_000_000
    )
    assert wave["total_area_visited_km2"] == pytest.approx(expected_area, rel=1e-12)


@pytest.mark.unit
def test_catalog_omits_zero_peak_and_retains_daily_records(tmp_path) -> None:
    """Preserve provisional daily identities while only exporting qualified waves.

    Args:
        tmp_path: Independent generation and scratch locations.

    Returns:
        None. Active and ended provisional candidates are absent from catalog.

    Examples:
        ``pytest tests/test_records.py -k omits_zero_peak``

    Steps:
        1. Preserve provisional daily identities while only exporting qualified waves.
        2. Check that active and ended provisional candidates are absent from catalog.
    """
    first, second = (date(2024, 1, 1), date(2024, 1, 2))
    provisional = _record(_SERIES, first, _IDS[0], _IDS[4])
    provisional = replace(
        provisional,
        peak_qualified_area=0.0,
        component=replace(
            provisional.component,
            geometry=replace(
                provisional.component.geometry,
                geometry=None,
                grace_geometry=provisional.component.geometry.full_geometry,
                area_qualified=0.0,
                area_grace=2.0,
                qualified_fraction=0.0,
            ),
        ),
    )
    born_later = _record(
        _SERIES, second, "20240102-00000000000000000000000000000002", _IDS[5]
    )
    born_later = replace(born_later, name="Test 2024")
    active_provisional = replace(
        provisional,
        wave_id="20240102-00000000000000000000000000000003",
        start_date=second,
        peak_date=second,
        component_id="00000000000000000000000000000007",
        component=replace(
            provisional.component,
            component_id="00000000000000000000000000000007",
            geometry=replace(provisional.component.geometry, time=second),
        ),
    )
    write_generation(
        (
            _batch(first, (provisional,)),
            _batch(
                second,
                (born_later, active_provisional),
                ended=(
                    EndedWave(provisional, np.datetime64(first, "D"), "died", None),
                ),
            ),
        ),
        destination=tmp_path / "generation",
        generation_id="qualified-only",
        completion=_completion("qualified-only", first, second),
    )
    generation = open_completed_generation(tmp_path / "generation")
    daily = _values(scan_wave_days(generation))
    assert len(daily) == 3
    assert {row["wave_id"] for row in daily} == {
        provisional.wave_id,
        born_later.wave_id,
        active_provisional.wave_id,
    }
    catalog = _values(scan_waves(generation))
    assert len(catalog) == 1
    assert catalog[0]["wave_id"] == born_later.wave_id
    assert catalog[0]["name"] == "Test 2024"
    assert catalog[0]["duration_days"] == 1
    assert catalog[0]["active"] is True


@pytest.mark.unit
def test_provisional_requalification_retains_key_and_qualified_grace_survives(
    tmp_path,
) -> None:
    """Retain the provisional start and eligible peak through a grace-only cutoff.

    Args:
        tmp_path: Independent generation and scratch locations.

    Returns:
        None. All daily rows survive; one positive catalog row remains active.

    Examples:
        ``pytest tests/test_records.py -k provisional_requalification``

    Steps:
        1. Retain the provisional start and eligible peak through a grace-only cutoff.
        2. Check that all daily rows survive; one positive catalog row remains active.
    """
    first = date(2024, 1, 1)
    second, third = (date(2024, 1, 2), date(2024, 1, 3))
    initial = _record(_SERIES, first, _IDS[0], _IDS[4])
    zero_geometry = replace(
        initial.component.geometry,
        geometry=None,
        grace_geometry=initial.component.geometry.full_geometry,
        area_qualified=0.0,
        area_grace=2.0,
        qualified_fraction=0.0,
    )
    initial = replace(
        initial,
        peak_qualified_area=0.0,
        component=replace(initial.component, geometry=zero_geometry),
    )
    qualified = _record(_SERIES, second, _IDS[0], _IDS[5], start=first)
    grace = _record(_SERIES, third, _IDS[0], _IDS[6], start=first)
    grace = replace(
        grace,
        peak_date=second,
        component=replace(grace.component, geometry=replace(zero_geometry, time=third)),
    )
    write_generation(
        tuple(
            (
                _batch(day, (record,))
                for day, record in (
                    (first, initial),
                    (second, qualified),
                    (third, grace),
                )
            )
        ),
        destination=tmp_path / "generation",
        generation_id="requalified",
        completion=_completion("requalified", first, third),
    )
    generation = open_completed_generation(tmp_path / "generation")
    daily = _values(scan_wave_days(generation))
    catalog = _values(scan_waves(generation))
    assert [row["wave_id"] for row in daily] == [initial.wave_id] * 3
    assert len(catalog) == 1
    assert catalog[0]["start_date"] == first
    assert catalog[0]["peak_date"] == second
    assert catalog[0]["duration_days"] == 3
    assert catalog[0]["peak_qualified_area_km2"] == 2.0
    assert catalog[0]["active"] is True
    assert catalog[0]["end_date"] is None


@pytest.mark.unit
def test_selected_series_metadata_and_rows_reject_other_series(tmp_path: Path) -> None:
    """Keep only the selected series and reject rows from another temperature.

    Args:
        tmp_path: Isolated output directories supplied by pytest.

    Returns:
        None.

    Examples:
        Run ``pytest tests/test_records.py -k selected_series``.

    Steps:
        1. Keep only the selected series and reject rows from another temperature.
    """
    path = tmp_path / "selected-series"
    write_generation(
        _batches(),
        destination=path,
        generation_id="selected-series",
        completion=_completion("selected-series"),
    )
    generation = open_completed_generation(path)
    assert generation.manifest["series"] == {
        "temperature": "tasmax",
        "direction": "warm",
    }
    assert all(
        entry["partition"]["temperature"] == "tasmax"
        and entry["partition"]["direction"] == "warm"
        for entry in generation.manifest["files"]
    )
    first, second = _batches()
    wrong_series = replace(first.current[0], series=WaveSeries("tasmin", "warm"))
    with pytest.raises(WaveRecordStorageError, match="outside the completed selection"):
        write_generation(
            (replace(first, current=(wrong_series,)), second),
            destination=tmp_path / "wrong-series",
            generation_id="wrong-series",
            completion=_completion("wrong-series"),
        )


@pytest.mark.unit
def test_catalog_preserves_exact_peak_date_name_and_duration(tmp_path: Path) -> None:
    """Keep the first equal peak and all days through a grace-only ending.

    Args:
        tmp_path: Independent generation and temporary state directories.

    Returns:
        None. The catalog contains the exact dates, peak, name and ending.

    Examples:
        Run ``pytest tests/test_records.py -k exact_peak``.

    Steps:
        1. Keep the first equal peak and all days through a grace-only ending.
        2. Check that the catalog contains the exact dates, peak, name and ending.
    """
    first = date(2024, 1, 1)
    last = first + timedelta(days=3)
    rows: list[TrackedWaveComponent] = []
    for offset, area in enumerate((2.0, 3.0, 3.0, 0.0)):
        day = first + timedelta(days=offset)
        row = _record(
            _SERIES, day, _IDS[0], _IDS[offset + 4], start=first if offset else None
        )
        rows.append(
            replace(
                row,
                name=None if offset == 0 else "Vera 2024",
                peak_qualified_area=2.0 if offset == 0 else 3.0,
                peak_date=first if offset == 0 else first + timedelta(days=1),
                component=replace(
                    row.component,
                    geometry=replace(
                        row.component.geometry,
                        geometry=None if area == 0 else row.component.geometry.geometry,
                        grace_geometry=row.component.geometry.full_geometry
                        if area == 0
                        else None,
                        area_qualified=area,
                        area_grace=2.0 if area == 0 else 0.0,
                        qualified_fraction=0.0 if area == 0 else 1.0,
                    ),
                ),
            )
        )
    end_event = last + timedelta(days=1)
    batches = [
        _batch(first + timedelta(days=offset), (row,))
        for offset, row in enumerate(rows)
    ]
    batches.append(
        _batch(
            end_event,
            (),
            ended=(EndedWave(rows[-1], np.datetime64(last, "D"), "died", None),),
        )
    )
    write_generation(
        batches,
        destination=tmp_path / "generation",
        generation_id="exact-peak",
        completion=_completion("exact-peak", first, end_event),
    )
    generation = open_completed_generation(tmp_path / "generation")
    catalog = _values(scan_waves(generation))
    assert len(catalog) == 1
    wave = catalog[0]
    assert wave["wave_id"] == rows[0].wave_id
    assert wave["start_date"] == first
    assert wave["last_observed_date"] == wave["end_date"] == last
    assert wave["duration_days"] == wave["observation_count"] == 4
    assert wave["peak_qualified_area_km2"] == 3.0
    assert wave["peak_date"] == first + timedelta(days=1)
    assert wave["name"] == "Vera 2024"
    assert wave["active"] is False
    assert wave["end_reason"] == "died"
    daily = scan_wave_days(generation)
    assert _values(daily["qualified_area_km2"]) == [2.0, 3.0, 3.0, 0.0]


@pytest.mark.unit
def test_open_completed_generation_reads_only_completion_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Open a generation without listing or reopening scientific data files.

    Args:
        tmp_path: Independent generation and temporary state directories.
        monkeypatch: Fixture restricting filesystem reads during opening.

    Returns:
        None. Opening accesses only the manifest and completion marker.

    Examples:
        Run ``pytest tests/test_records.py -k completion_metadata``.

    Steps:
        1. Open a generation without listing or reopening scientific data files.
        2. Check that opening accesses only the manifest and completion marker.
    """
    destination = tmp_path / "generation"
    evidence = write_generation(
        _batches(),
        destination=destination,
        generation_id="metadata-only",
        completion=_completion("metadata-only"),
    )
    assert not hasattr(evidence, "validation")
    assert not (destination / "validation").exists()
    assert not list(tmp_path.rglob("*.sqlite*"))
    filesystem = fsspec.filesystem("file")
    original_open = filesystem.open
    opened: list[str] = []

    def metadata_open(path: str, mode: str = "rb", **kwargs: object) -> BinaryIO:
        """Permit completion metadata and reject all other file reads.

        Args:
            path: Requested object path.
            mode: File open mode.
            **kwargs: Filesystem open options.

        Returns:
            Binary metadata file handle.

        Examples:
            Called by ``open_completed_generation``.

        Steps:
            1. Permit completion metadata and reject all other file reads.
        """
        assert path.endswith(("manifest.json", "_COMPLETE.json"))
        opened.append(path)
        return cast("BinaryIO", original_open(path, mode, **kwargs))

    def forbid_listing(*args: object, **kwargs: object) -> None:
        """Reject a recursive dataset listing while opening completed output.

        Args:
            *args: Filesystem listing arguments.
            **kwargs: Filesystem listing options.

        Returns:
            Never returns.

        Examples:
            A repeated whole-output inspection fails this assertion.

        Steps:
            1. Reject a recursive dataset listing while opening completed output.
        """
        pytest.fail("opening output must not list scientific datasets")

    monkeypatch.setattr(filesystem, "open", metadata_open)
    monkeypatch.setattr(filesystem, "find", forbid_listing)
    monkeypatch.setattr(pq, "read_table", forbid_listing)
    generation = open_completed_generation(destination, filesystem=filesystem)
    assert generation.inventory == evidence.inventory
    assert opened
    assert {entry["table"] for entry in generation.manifest["files"]} == {
        "wave_days",
        "waves",
        "lineage",
    }
    assert set(generation.manifest["tables"]) == {"wave_days", "waves", "lineage"}
    assert set(generation.manifest) == {
        "generation_id",
        "series",
        "package",
        "start_date",
        "end_date",
        "files",
        "tables",
        "all_storms",
        "all_waves",
    }
    assert set(json.loads((destination / "_COMPLETE.json").read_bytes())) == {
        "generation_id",
        "completed_at",
    }


@pytest.mark.component
def test_pipeline_writes_both_self_contained_catalogs(tmp_path: Path) -> None:
    """Export active and ended names across years without unqualified histories.

    Args:
        tmp_path: Independent completed-generation directory.

    Returns:
        None. Direct GeoPandas reading equals the full catalog's named selection.

    Examples:
        Run ``pytest tests/test_records.py -k self_contained``.

    Steps:
        1. Export active and ended names across years without unqualified histories.
        2. Check that direct GeoPandas reading equals the full catalog's named selection.
    """
    batches = tuple(
        replace(
            batch,
            current=tuple(
                replace(record, name=f"Example {record.wave_id}")
                if record.wave_id
                in {_CROSS_YEAR_IDS[0], _CROSS_YEAR_IDS[1], _CROSS_YEAR_IDS[11]}
                else record
                for record in batch.current
            ),
        )
        for batch in _batches()
    )
    destination = tmp_path / "named"
    write_generation(
        batches,
        destination=destination,
        generation_id="named",
        completion=_completion("named"),
        row_group_rows=1,
        file_rows=1,
    )
    generation = open_completed_generation(destination)
    download = generation.manifest["all_storms"]
    assert download["path"] == "all_storms.parquet"
    path = destination / download["path"]
    direct = gpd.read_parquet(path)
    assert "storm_key" not in direct.columns
    daily = scan_wave_days(generation)
    assert "storm_key" not in daily.columns
    assert {"wave_id", "component_id"} <= set(daily.columns)
    direct["parent_keys"] = direct.parent_keys.map(list)
    expected = scan_waves(generation)
    full = Waves.read_parquet(destination / "all_waves.parquet").frame
    pd.testing.assert_frame_equal(
        full[expected.columns], expected, check_exact=True, check_dtype=False
    )
    assert full.name.isna().any()
    assert full.name.notna().any()
    assert set(full.active) == {True, False}
    assert set(full.start_year) == {2024, 2025}
    assert len(full) == generation.manifest["all_waves"]["rows"] == 6
    assert (
        generation.manifest["all_waves"]["bytes"]
        == (destination / "all_waves.parquet").stat().st_size
    )
    assert not (destination / "named_storms.parquet").exists()
    expected = expected[expected.name.notna()].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        direct[expected.columns], expected, check_exact=True, check_dtype=False
    )
    WaveCatalogSchema.validate(direct)
    assert direct.geometry.name == "lifetime_geometry"
    assert direct.crs.to_epsg() == 4326
    assert download["rows"] == len(direct) == 3
    assert download["bytes"] == path.stat().st_size
    assert set(direct.active) == {True, False}
    assert set(direct.start_year) == {2024, 2025}
    assert direct.name.notna().all()
    assert direct.total_area_visited_km2.gt(0).all()
    table = NamedStorms.read_parquet(path)
    pd.testing.assert_frame_equal(table.frame, direct)
    supplied = table.frame.copy()
    constructed = NamedStorms(supplied)
    assert constructed.frame is supplied
    pd.testing.assert_frame_equal(supplied, direct)
    copied = tmp_path / "roundtrip.parquet"
    saved = table.to_parquet(copied, row_group_rows=1)
    pd.testing.assert_frame_equal(NamedStorms.read_parquet(copied).frame, table.frame)
    assert saved.rows == 3
    assert saved.objects == 1
    assert saved.bytes == copied.stat().st_size

    unnamed = table.frame.copy()
    unnamed.loc[0, "name"] = None
    with pytest.raises(WaveRecordStorageError, match="invalid NamedStormsSchema table"):
        NamedStorms(unnamed)
    invalid = table.frame.copy()
    invalid.loc[0, "total_area_visited_km2"] = 0.0
    with pytest.raises(WaveRecordStorageError, match="invalid NamedStormsSchema table"):
        NamedStorms(invalid)
    memory = fsspec.filesystem("memory")
    table.to_parquet("/named-catalog/roundtrip.parquet", filesystem=memory)
    pd.testing.assert_frame_equal(
        NamedStorms.read_parquet(
            "/named-catalog/roundtrip.parquet", filesystem=memory
        ).frame,
        table.frame,
    )
    with pytest.raises(WaveRecordStorageError, match="cannot read NamedStormsSchema"):
        NamedStorms.read_parquet(tmp_path / "missing.parquet")
    with pytest.raises(WaveRecordStorageError, match="positive integer"):
        table.to_parquet(tmp_path / "invalid.parquet", row_group_rows=0)
    table.frame.loc[0, "name"] = None
    with pytest.raises(WaveRecordStorageError, match="invalid NamedStormsSchema table"):
        table.to_parquet(tmp_path / "mutated.parquet")
    assert not (tmp_path / "mutated.parquet").exists()


@pytest.mark.component
def test_full_wave_table_preserves_unnamed_histories_and_shared_io(
    tmp_path: Path,
) -> None:
    """Read and write all catalog waves through the shared class implementation.

    Args:
        tmp_path: Independent output directory.

    Returns:
        None. Full active/ended, named/unnamed histories survive GeoParquet I/O.

    Examples:
        Run ``pytest tests/test_records.py -k full_wave_table``.

    Steps:
        1. Read and write all catalog waves through the shared class implementation.
        2. Check that full active/ended, named/unnamed histories survive GeoParquet I/O.
    """
    first, second = _batches()
    named = replace(first.current[0], name="Example 2024")
    continued = replace(second.current[0], name="Example 2024")
    destination = tmp_path / "generation"
    write_generation(
        (
            replace(first, current=(named, *first.current[1:])),
            replace(second, current=(continued, *second.current[1:])),
        ),
        destination=destination,
        generation_id="full",
        completion=_completion("full"),
    )
    catalog = scan_waves(open_completed_generation(destination))
    for column in ("name", "end_reason", "merged_into_wave_id"):
        catalog[column] = catalog[column].astype("str")
    table = Waves(catalog)
    assert table.frame is catalog
    assert table.schema is WaveCatalogSchema
    assert catalog.name.notna().sum() == 1
    assert catalog.name.isna().sum() == 5
    assert set(catalog.active) == {True, False}
    assert len(table.frame) == 6
    built = Waves.from_rows(table.frame.to_dict("records"))
    assert type(built) is Waves
    pd.testing.assert_frame_equal(
        built.frame[catalog.columns], catalog, check_dtype=False
    )
    path = tmp_path / "waves.parquet"
    inventory = table.to_parquet(path)
    restored = Waves.read_parquet(path)
    assert type(restored) is Waves
    pd.testing.assert_frame_equal(
        restored.frame, catalog, check_dtype=False, check_exact=True
    )
    assert inventory.rows == 6
    assert inventory.objects == 1
    assert inventory.bytes == path.stat().st_size
    assert restored.frame.crs.to_epsg() == 4326

    named_table = NamedStorms(catalog[catalog.name.notna()].reset_index(drop=True))
    assert named_table.schema is NamedStormsSchema
    assert isinstance(named_table, Waves)
    with pytest.raises(WaveRecordStorageError, match="NamedStormsSchema"):
        NamedStorms(catalog)
    invalid = catalog.copy()
    invalid.loc[0, "total_area_visited_km2"] = -1.0
    with pytest.raises(WaveRecordStorageError, match="WaveCatalogSchema"):
        Waves(invalid)


@pytest.mark.component
def test_full_wave_table_empty_roundtrip(tmp_path: Path) -> None:
    """Preserve an empty full catalog's schema and CRS through file I/O.

    Args:
        tmp_path: Independent output directory.

    Returns:
        None. Empty full-catalog fields and EPSG:4326 remain readable.

    Examples:
        Run ``pytest tests/test_records.py -k full_wave_table_empty``.

    Steps:
        1. Preserve an empty full catalog's schema and CRS through file I/O.
        2. Check that empty full-catalog fields and EPSG:4326 remain readable.
    """
    table = Waves.from_rows([])
    assert table.frame.empty
    assert set(table.frame.columns) == set(WaveCatalogSchema.to_schema().columns)
    inventory = table.to_parquet(tmp_path / "empty-waves.parquet")
    restored = Waves.read_parquet(tmp_path / "empty-waves.parquet")
    pd.testing.assert_frame_equal(restored.frame, table.frame)
    assert restored.frame.crs.to_epsg() == 4326
    assert inventory.rows == 0
    assert inventory.objects == 1


@pytest.mark.component
def test_pipeline_writes_typed_empty_named_catalog(tmp_path: Path) -> None:
    """Complete a run with no names and a directly readable empty GeoParquet.

    Args:
        tmp_path: Independent completed-generation directory.

    Returns:
        None. All catalog columns and EPSG:4326 remain available in an empty file.

    Examples:
        Run ``pytest tests/test_records.py -k empty_named``.

    Steps:
        1. Complete a run with no names and a directly readable empty GeoParquet.
        2. Check that all catalog columns and EPSG:4326 remain available in an empty file.
    """
    destination = tmp_path / "unnamed"
    write_generation(
        _batches(),
        destination=destination,
        generation_id="unnamed",
        completion=_completion("unnamed"),
    )
    generation = open_completed_generation(destination)
    direct = gpd.read_parquet(destination / "all_storms.parquet")
    assert direct.empty
    assert direct.crs.to_epsg() == 4326
    assert set(direct.columns) == set(WaveCatalogSchema.to_schema().columns)
    WaveCatalogSchema.validate(direct)
    assert generation.manifest["all_storms"]["rows"] == 0
    NamedStormsSchema.validate(direct)
    assert NamedStorms.read_parquet(destination / "all_storms.parquet").frame.empty
    full = Waves.read_parquet(destination / "all_waves.parquet").frame
    assert len(full) == generation.manifest["all_waves"]["rows"] == 6
    assert full.name.isna().all()


@pytest.mark.component
def test_quiet_run_writes_both_typed_empty_downloads(tmp_path: Path) -> None:
    """Keep both catalog files directly readable when no waves are present.

    Args:
        tmp_path: Independent completed-generation directory.

    Returns:
        None. Both empty files retain the catalog columns and geographic metadata.

    Examples:
        Run ``pytest tests/test_records.py -k quiet_run``.

    Steps:
        1. Keep both catalog files directly readable when no waves are present.
        2. Check that both empty files retain the catalog columns and geographic metadata.
    """
    destination = tmp_path / "quiet"
    write_generation(
        tuple(_batch(batch.date, ()) for batch in _batches()),
        destination=destination,
        generation_id="quiet",
        completion=_completion("quiet"),
    )
    generation = open_completed_generation(destination)
    for name, table in (("all_storms", NamedStorms), ("all_waves", Waves)):
        frame = table.read_parquet(destination / f"{name}.parquet").frame
        assert frame.empty
        assert set(frame.columns) == set(WaveCatalogSchema.to_schema().columns)
        assert frame.crs.to_epsg() == 4326
        assert generation.manifest[name]["rows"] == 0
        assert (
            generation.manifest[name]["bytes"]
            == (destination / f"{name}.parquet").stat().st_size
        )


@pytest.mark.unit
@pytest.mark.parametrize("name", ["all_storms", "all_waves"])
@pytest.mark.parametrize(
    ("field", "value"), [("path", "wrong.parquet"), ("rows", -1), ("bytes", 0)]
)
def test_manifest_checks_both_download_entries(
    tmp_path: Path, name: str, field: str, value: object
) -> None:
    """Reject malformed download metadata without opening the catalog files.

    Args:
        tmp_path: Independent completed-generation directory.
        name: Either required catalog download entry.
        field: Inventory field to corrupt.
        value: Invalid path, row count or byte count.

    Returns:
        None. Opening rejects the malformed entry for either catalog.

    Examples:
        Run ``pytest tests/test_records.py -k both_download_entries``.

    Steps:
        1. Reject malformed download metadata without opening the catalog files.
        2. Check that opening rejects the malformed entry for either catalog.
    """
    destination = tmp_path / "catalogs"
    write_generation(
        _batches(),
        destination=destination,
        generation_id="catalogs",
        completion=_completion("catalogs"),
    )
    path = destination / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest[name][field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(WaveRecordStorageError, match=f"malformed {name} download"):
        open_completed_generation(destination)


@pytest.mark.component
@pytest.mark.parametrize("filename", ["all_storms.parquet", "all_waves.parquet"])
def test_catalog_download_failure_prevents_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
) -> None:
    """Leave either failed download incomplete so users never receive partial output.

    Args:
        tmp_path: Independent output directory.
        monkeypatch: Fixture simulating failure of either catalog upload.
        filename: Either required self-contained catalog whose write will fail.

    Returns:
        None. The writer raises and no completed marker or manifest is emitted.

    Examples:
        Run ``pytest tests/test_records.py -k catalog_download_failure``.

    Steps:
        1. Leave either failed download incomplete so users never receive partial output.
        2. Check that the writer raises and no completed marker or manifest is emitted.
    """
    filesystem = fsspec.filesystem("file", skip_instance_cache=True)
    original_open = filesystem.open

    def fail_named(path, mode="rb", **kwargs):
        """Reject the selected download and retain ordinary filesystem writes.

        Args:
            path: Destination file path.
            mode: File mode used by the writer.
            **kwargs: Writer options including filesystem and row-group size.

        Returns:
            File handle for ordinary paths; raises for the selected catalog.

        Raises:
            OSError: For the simulated named-file failure.

        Examples:
            fail_named("output/all_storms.parquet", "wb")

        Steps:
            1. Reject the selected download and retain ordinary filesystem writes.
        """
        if str(path).endswith(filename) and mode == "wb":
            raise OSError("simulated named download failure")
        return original_open(path, mode, **kwargs)

    monkeypatch.setattr(filesystem, "open", fail_named)
    destination = tmp_path / "failed"
    with pytest.raises(OSError, match="simulated named download failure"):
        write_generation(
            _batches(),
            destination=destination,
            generation_id="failed",
            completion=_completion("failed"),
            filesystem=filesystem,
        )
    assert (destination / "_INCOMPLETE.json").exists()
    assert not (destination / "_COMPLETE.json").exists()
    assert not (destination / "manifest.json").exists()


@pytest.mark.unit
def test_missing_completion_marker_and_mismatched_generation_reject(
    tmp_path: Path,
) -> None:
    """Reject incomplete or mismatched metadata without inspecting Parquets.

    Args:
        tmp_path: Independent generation and temporary state directories.

    Returns:
        None. Both incomplete and altered metadata cannot be opened.

    Examples:
        Run ``pytest tests/test_records.py -k mismatched_generation``.

    Steps:
        1. Reject incomplete or mismatched metadata without inspecting Parquets.
        2. Check that both incomplete and altered metadata cannot be opened.
    """
    destination = tmp_path / "generation"
    write_generation(
        _batches(),
        destination=destination,
        generation_id="metadata",
        completion=_completion("metadata"),
    )
    marker = destination / "_COMPLETE.json"
    original_marker = marker.read_bytes()
    marker.unlink()
    with pytest.raises(WaveRecordStorageError):
        open_completed_generation(destination)
    marker.write_bytes(original_marker)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["generation_id"] = "different-generation"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )
    with pytest.raises(WaveRecordStorageError, match="manifest"):
        open_completed_generation(destination)


@pytest.mark.unit
def test_unrequested_broken_parquet_does_not_block_opening(tmp_path: Path) -> None:
    """Delay scientific data reads until callers request the affected dataset.

    Args:
        tmp_path: Independent generation and temporary state directories.

    Returns:
        None. The untouched catalog remains readable beside a broken daily file.

    Examples:
        Run ``pytest tests/test_records.py -k unrequested_broken``.

    Steps:
        1. Delay scientific data reads until callers request the affected dataset.
        2. Check that the untouched catalog remains readable beside a broken daily file.
    """
    destination = tmp_path / "generation"
    write_generation(
        _batches(),
        destination=destination,
        generation_id="read-on-demand",
        completion=_completion("read-on-demand"),
    )
    manifest = json.loads((destination / "manifest.json").read_bytes())
    daily_file = next(
        entry for entry in manifest["files"] if entry["table"] == "wave_days"
    )
    (destination / daily_file["path"]).write_bytes(b"broken parquet")
    generation = open_completed_generation(destination)
    assert len(scan_waves(generation)) == 6
    with pytest.raises((WaveRecordStorageError, pa.ArrowInvalid, OSError)):
        scan_wave_days(generation)


@pytest.mark.unit
@pytest.mark.parametrize("bad_area", [-1.0, float("inf")])
def test_writer_checks_outgoing_scientific_schema(
    tmp_path: Path, bad_area: float
) -> None:
    """Reject invalid measurements before publishing any daily Parquet file.

    Args:
        tmp_path: New output directory.
        bad_area: Invalid negative or infinite qualified area.

    Returns:
        None. The incomplete marker remains and no completion is published.

    Examples:
        pytest tests/test_records.py -k outgoing_scientific_schema

    Steps:
        1. Reject invalid measurements before publishing any daily Parquet file.
        2. Check that the incomplete marker remains and no completion is published.
    """
    batches = _batches()
    record = batches[0].current[0]
    geometry = replace(record.component.geometry, area_qualified=bad_area)
    invalid = replace(record, component=replace(record.component, geometry=geometry))
    first = replace(batches[0], current=(invalid, *batches[0].current[1:]))
    destination = tmp_path / "invalid"
    with pytest.raises(WaveRecordStorageError, match="invalid WaveDaySchema table"):
        write_generation(
            (first, batches[1]),
            destination=destination,
            generation_id="invalid",
            completion=_completion("invalid"),
        )
    assert (destination / "_INCOMPLETE.json").exists()
    assert not (destination / "_COMPLETE.json").exists()
    assert not list(destination.rglob("*.parquet"))


@pytest.mark.unit
@pytest.mark.parametrize(
    "change",
    ["missing", "extra", "invalid", "date_string", "date_integer", "parent_string"],
)
def test_reader_checks_selected_file_schema(tmp_path: Path, change: str) -> None:
    """Validate selected catalog columns and measurements once after reading.

    Args:
        tmp_path: Completed output directory.
        change: Invalid columns, measurements, physical date type or parent-list type.

    Returns:
        None. Opening metadata succeeds, while reading the edited file fails.

    Examples:
        pytest tests/test_records.py -k selected_file_schema

    Steps:
        1. Validate selected catalog columns and measurements once after reading.
        2. Check that opening metadata succeeds, while reading the edited file fails.
    """
    destination = tmp_path / "generation"
    write_generation(
        _batches(),
        destination=destination,
        generation_id="schema",
        completion=_completion("schema"),
    )
    generation = open_completed_generation(destination)
    entry = next(
        item for item in generation.manifest["files"] if item["table"] == "waves"
    )
    path = destination / entry["path"]
    frame = gpd.read_parquet(path)
    if change == "missing":
        frame = frame.drop(columns="start_date")
    elif change == "extra":
        frame["unexpected"] = "extra"
    elif change == "invalid":
        frame.loc[0, "peak_qualified_area_km2"] = -1.0
    elif change == "date_string":
        frame["start_date"] = frame["start_date"].dt.strftime("%Y-%m-%d")
    elif change == "date_integer":
        frame["start_date"] = frame["start_date"].astype("int64")
    else:
        frame["parent_keys"] = "not a list"
    frame.to_parquet(path, index=False)
    generation = open_completed_generation(destination)
    with pytest.raises(WaveRecordStorageError, match="cannot read valid waves"):
        scan_waves(generation)


@pytest.mark.unit
@pytest.mark.parametrize("table", ["wave_days", "waves", "lineage"])
def test_reader_checks_restored_partition_years(tmp_path: Path, table: str) -> None:
    """Reject a selected file whose restored year disagrees with its dates.

    Args:
        tmp_path: Completed output directory.
        table: Scientific dataset whose manifest partition is edited.

    Returns:
        None. One full schema validation catches the mismatch before row filtering.

    Examples:
        pytest tests/test_records.py -k restored_partition_years

    Steps:
        1. Reject a selected file whose restored year disagrees with its dates.
        2. Check that one full schema validation catches the mismatch before row filtering.
    """
    destination = tmp_path / "generation"
    write_generation(
        _batches(),
        destination=destination,
        generation_id="partition",
        completion=_completion("partition"),
    )
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = next(item for item in manifest["files"] if item["table"] == table)
    field = {
        "wave_days": "observation_year",
        "waves": "start_year",
        "lineage": "event_year",
    }[table]
    original = entry["partition"][field]
    edited = original + 1
    path = destination / entry["path"]
    entry["partition"][field] = edited
    entry["path"] = (
        entry["path"]
        .replace(f"{field}={original}", f"{field}={edited}")
        .replace("part-00000.parquet", "part-99999.parquet")
    )
    new_path = destination / entry["path"]
    new_path.parent.mkdir(parents=True, exist_ok=True)
    path.rename(new_path)
    manifest["files"].sort(key=lambda item: item["path"])
    manifest_path.write_text(json.dumps(manifest))
    generation = open_completed_generation(destination)
    if table == "waves":
        with pytest.raises(WaveRecordStorageError, match="cannot read valid waves"):
            scan_waves(generation)
    else:
        scan = scan_wave_days if table == "wave_days" else scan_lineage
        with pytest.raises(WaveRecordStorageError, match=f"cannot read valid {table}"):
            scan(generation, start=date(edited, 1, 1), end=date(edited, 12, 31))


@pytest.mark.unit
@pytest.mark.parametrize("table", ["wave_days", "waves", "lineage"])
def test_reader_does_not_overwrite_stored_partition_columns(
    tmp_path: Path, table: str
) -> None:
    """Reject unexpected physical partition columns before restoring metadata.

    Args:
        tmp_path: Completed output directory.
        table: Scientific dataset whose physical columns are edited.

    Returns:
        None. The reader does not silently replace the stored temperature column.

    Examples:
        pytest tests/test_records.py -k stored_partition_columns

    Steps:
        1. Reject unexpected physical partition columns before restoring metadata.
        2. Check that the reader does not silently replace the stored temperature column.
    """
    destination = tmp_path / "generation"
    write_generation(
        _batches(),
        destination=destination,
        generation_id="columns",
        completion=_completion("columns"),
    )
    generation = open_completed_generation(destination)
    entry = next(
        item for item in generation.manifest["files"] if item["table"] == table
    )
    path = destination / entry["path"]
    frame = (
        gpd.read_parquet(path)
        if table in {"wave_days", "waves"}
        else pd.read_parquet(path)
    )
    frame["temperature"] = "tasmin"
    frame.to_parquet(path, index=False)
    scan = {
        "wave_days": scan_wave_days,
        "waves": scan_waves,
        "lineage": scan_lineage,
    }[table]
    with pytest.raises(WaveRecordStorageError, match=f"cannot read valid {table}"):
        scan(generation)


@pytest.mark.component
@pytest.mark.parametrize("table_class", [WaveDays, Lineage])
def test_table_classes_share_validated_io_without_copying(tmp_path, table_class):
    """Keep daily and lineage frames intact through local and configured I/O.

    Args:
        tmp_path: Independent output directory.
        table_class: Daily geographic or lineage table class.

    Returns:
        None. Construction retains the frame, roundtrips preserve every field,
        and writing catches caller edits before creating an invalid file.

    Examples:
        Run ``pytest tests/test_records.py -k table_classes_share``.

    Steps:
        1. Keep daily and lineage frames intact through local and configured I/O.
        2. Check that construction retains the frame, roundtrips preserve every field, and writing catches caller edits before creating an invalid file.
    """
    destination = tmp_path / "generation"
    write_generation(
        _batches(),
        destination=destination,
        generation_id="classes",
        completion=_completion("classes"),
    )
    generation = open_completed_generation(destination)
    scan = scan_wave_days if table_class is WaveDays else scan_lineage
    frame = scan(generation)
    table = table_class(frame)
    assert table.frame is frame
    original = frame.copy()
    built = table_class.from_rows(frame.to_dict("records"))
    pd.testing.assert_frame_equal(built.frame[frame.columns], frame, check_exact=True)
    filesystem = fsspec.filesystem("memory")
    inventory = table.to_parquet("/tables/rows.parquet", filesystem=filesystem)
    restored = table_class.read_parquet("/tables/rows.parquet", filesystem=filesystem)
    pd.testing.assert_frame_equal(restored.frame, original, check_exact=True)
    assert inventory.rows == len(frame)
    empty = table_class.from_rows([])
    empty.to_parquet(tmp_path / "empty.parquet")
    pd.testing.assert_frame_equal(
        table_class.read_parquet(tmp_path / "empty.parquet").frame, empty.frame
    )
    if table_class is WaveDays:
        table.frame.loc[0, "qualified_area_km2"] = -1.0
    else:
        table.frame.loc[0, "relationship"] = "invalid"
    with pytest.raises(WaveRecordStorageError, match="invalid"):
        table.to_parquet(tmp_path / "invalid.parquet")
    assert not (tmp_path / "invalid.parquet").exists()


@pytest.mark.component
def test_pipeline_keeps_one_schema_check_per_file_buffer(tmp_path, monkeypatch):
    """Prevent extra schema passes when table classes replace the loose helpers.

    Args:
        tmp_path: Independent completed-generation directory.
        monkeypatch: Fixture counting existing Pandera schema calls.

    Returns:
        None. Each table buffer is validated once during writing and each
        selected file once during reading; filtering adds no schema checks.

    Examples:
        Run ``pytest tests/test_records.py -k one_schema_check``.

    Steps:
        1. Prevent extra schema passes when table classes replace the loose helpers.
        2. Check that each table buffer is validated once during writing and each selected file once during reading; filtering adds no schema checks.
    """
    models = [WaveDaySchema, WaveCatalogSchema, LineageSchema, NamedStormsSchema]
    for model in models:
        model.to_schema()
    counts = dict.fromkeys(models, 0)
    originals = {model: model.validate for model in models}
    for model in models:
        original = originals[model]

        def counted(cls, frame, *args, model=model, original=original, **kwargs):
            """Count a schema call while preserving the original validation.

            Args:
                cls: Schema class receiving the call.
                frame: Scientific rows supplied to the schema.
                *args: Existing positional validation arguments.
                model: Schema being counted.
                original: Original bound Pandera validation method.
                **kwargs: Existing validation options.

            Returns:
                The original validation result.

            Examples:
                Called while writing and reading the test generation.

            Steps:
                1. Count a schema call while preserving the original validation.
            """
            counts[model] += 1
            return original(frame, *args, **kwargs)

        monkeypatch.setattr(model, "validate", classmethod(counted))
    destination = tmp_path / "generation"
    write_generation(
        _batches(),
        destination=destination,
        generation_id="passes",
        completion=_completion("passes"),
    )
    generation = open_completed_generation(destination)
    for name, model in [
        ("wave_days", WaveDaySchema),
        ("waves", WaveCatalogSchema),
        ("lineage", LineageSchema),
    ]:
        assert counts[model] == sum(
            entry["table"] == name for entry in generation.manifest["files"]
        )
    assert (
        counts[NamedStormsSchema] == 0
    )  # Downloads reuse already checked catalog parts.
    counts.update(dict.fromkeys(models, 0))
    scan_wave_days(generation, wave_ids=(_CROSS_YEAR_IDS[0],))
    scan_waves(generation, active=True)
    scan_lineage(generation)
    for name, model in [
        ("wave_days", WaveDaySchema),
        ("waves", WaveCatalogSchema),
        ("lineage", LineageSchema),
    ]:
        assert counts[model] == sum(
            entry["table"] == name for entry in generation.manifest["files"]
        )
    assert counts[NamedStormsSchema] == 0
