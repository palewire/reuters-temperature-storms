"""Focused tracking, naming and restart checks for the footprint beta."""

from __future__ import annotations

import json
from dataclasses import fields, replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform

import reuters_temperature_storms._wave_tracking as tracking_module
import reuters_temperature_storms.tracker as tracker_module
from reuters_temperature_storms._wave_tracking import WaveSeries, track_reuters_waves
from reuters_temperature_storms.components import (
    ComponentMembership,
    DailyWaveComponentGeometry,
)
from reuters_temperature_storms.identity import component_identity
from reuters_temperature_storms.naming import name_at_ordinal
from reuters_temperature_storms.tracker import (
    DailyComponentBatch,
    WaveTracker,
    WaveTrackerCheckpoint,
    WaveTrackerError,
    WaveTrackingBatch,
    track_reuters_component_batches,
)

_START = date(2023, 12, 31)
_SERIES = WaveSeries("tasmax", "warm")
_AREA_TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)


def _component(
    day: date,
    label: int,
    *,
    west: float = -20,
    east: float = 20,
    series: WaveSeries = _SERIES,
) -> ComponentMembership:
    """Create a large qualifying footprint eligible for a Reuters name.

    Args:
        day: Date of the component.
        label: Daily component label.
        west: Western boundary in degrees.
        east: Eastern boundary in degrees.
        series: Temperature and direction used for component identity.

    Returns:
        A valid named-size component with measured equal-area geometry.

    Examples:
        ``_component(_START, 1)`` creates the opening test footprint.

    Steps:
        1. Choose the supplied qualified and grace polygons and measure their areas.
        2. Build the daily geometry record and compute the full polygon's component identifier.
        3. Return the identified component membership.
    """
    geometry = box(west, -20, east, 20)
    area = float(transform(_AREA_TRANSFORMER.transform, geometry).area / 1_000_000)
    day_value = np.datetime64(day, "D")
    return ComponentMembership(
        component_id=component_identity(
            temperature=series.temperature,
            direction=series.direction,
            date=day_value,
            geometry=geometry,
        ),
        geometry=DailyWaveComponentGeometry(
            time=day_value,
            label=label,
            geometry=geometry,
            grace_geometry=None,
            full_geometry=geometry,
            area_qualified=area,
            area_grace=0.0,
            area_gap_filled=0.0,
            qualified_fraction=1.0,
        ),
    )


def _batch(
    day: date,
    components: tuple[ComponentMembership, ...],
    *,
    series: WaveSeries = _SERIES,
) -> DailyComponentBatch:
    """Create one footprint batch for the selected series.

    Args:
        day: Date to track.
        components: Finished components for the date.
        series: Selected temperature and direction for the batch.

    Returns:
        Batch containing only the selected series.

    Examples:
        ``_batch(_START, ())`` represents an empty opening date.

    Steps:
        1. Collect the active records, endings, and lineage for the requested date.
        2. Construct the corresponding checkpoint and daily tracking batch.
    """
    return DailyComponentBatch(date=day, series=series, components=components)


@pytest.mark.unit
def test_batch_and_checkpoint_require_no_upstream_baseline_or_cell_state() -> None:
    """Verify batch and checkpoint require no upstream baseline or cell state.

    Returns:
        None.

    Steps:
        1. Advance an empty footprint batch through the tracker.
        2. Check the batch and serialized checkpoint contain only tracking fields, without upstream baseline or cell state.

    Examples:
        Run ``pytest tests/test_tracker.py -k batch_and_checkpoint_require_no_upstream_baseline_or_cell_state``.
    """
    batch = _batch(_START, ())
    assert {field.name for field in fields(batch)} == {"date", "series", "components"}
    tracker = WaveTracker(
        series=WaveSeries("tasmax", "warm"),
        requested_start=_START,
        source_identity="footprint-files",
    )
    output = tracker.advance(batch)
    document = json.loads(output.checkpoint.to_bytes())
    assert set(document) == {
        "requested_start",
        "through_date",
        "source_identity",
        "temperature",
        "direction",
        "next_name_ordinal",
        "active",
    }
    assert output.checkpoint.series == WaveSeries("tasmax", "warm")
    assert output.checkpoint.active == ()
    assert output.checkpoint.next_name_ordinal == 0


@pytest.mark.unit
def test_resume_preserves_names_and_ordinal_across_year_boundary() -> None:
    """Verify resume preserves names and ordinal across year boundary.

    Returns:
        None.

    Steps:
        1. Track the three synthetic dates without interruption.
        2. Restore the first checkpoint and track the remaining dates.
        3. Compare checkpoint bytes, assigned names, and retained ending names across the year boundary.

    Examples:
        Run ``pytest tests/test_tracker.py -k resume_preserves_names_and_ordinal_across_year_boundary``.
    """
    first = _batch(_START, (_component(_START, 1),))
    next_day = _START + timedelta(days=1)
    second = _batch(next_day, (_component(next_day, 1),))
    third_day = next_day + timedelta(days=1)
    third = _batch(third_day, (_component(third_day, 2, west=40, east=80),))
    uninterrupted = tuple(
        track_reuters_component_batches(
            (first, second, third),
            series=WaveSeries("tasmax", "warm"),
            requested_start=_START,
            source_identity="footprint-files",
        )
    )
    restored = WaveTrackerCheckpoint.from_bytes(uninterrupted[0].checkpoint.to_bytes())
    resumed = tuple(
        track_reuters_component_batches(
            (second, third),
            series=WaveSeries("tasmax", "warm"),
            requested_start=next_day,
            source_identity="footprint-files",
            prior_checkpoint=restored,
        )
    )
    assert [batch.checkpoint.to_bytes() for batch in resumed] == [
        batch.checkpoint.to_bytes() for batch in uninterrupted[1:]
    ]
    assert resumed[0].current[0].name == f"{name_at_ordinal(0)} 2023"
    assert resumed[1].current[0].name == f"{name_at_ordinal(1)} 2024"
    assert len(resumed[1].ended) == 1
    assert resumed[1].ended[0].final_record.name == resumed[0].current[0].name


@pytest.mark.unit
def test_tracker_preserves_split_merge_lineage_after_restart() -> None:
    """Verify tracker preserves split merge lineage after restart.

    Returns:
        None.

    Steps:
        1. Track an initial component, its split, and its subsequent merge.
        2. Restore the split-day checkpoint and replay the merge date.
        3. Compare lineage, ending records, and checkpoint bytes with uninterrupted tracking.

    Examples:
        Run ``pytest tests/test_tracker.py -k tracker_preserves_split_merge_lineage_after_restart``.
    """
    day2 = _START + timedelta(days=1)
    day3 = day2 + timedelta(days=1)
    first = _batch(_START, (_component(_START, 1),))
    split = _batch(
        day2,
        (_component(day2, 1, west=-20, east=5), _component(day2, 2, west=5, east=20)),
    )
    merged = _batch(day3, (_component(day3, 1),))
    outputs = tuple(
        track_reuters_component_batches(
            (first, split, merged),
            series=WaveSeries("tasmax", "warm"),
            requested_start=_START,
            source_identity="footprint-files",
        )
    )
    assert [edge.relationship for edge in outputs[1].lineage] == ["split"]
    assert [edge.relationship for edge in outputs[2].lineage] == ["merge"]
    restored = WaveTrackerCheckpoint.from_bytes(outputs[1].checkpoint.to_bytes())
    resumed = next(
        track_reuters_component_batches(
            (merged,),
            series=WaveSeries("tasmax", "warm"),
            requested_start=day3,
            source_identity="footprint-files",
            prior_checkpoint=restored,
        )
    )
    assert resumed.lineage == outputs[2].lineage
    assert resumed.ended == outputs[2].ended
    assert resumed.checkpoint.to_bytes() == outputs[2].checkpoint.to_bytes()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("next_name_ordinal", -1, "nonnegative name ordinal"),
        ("next_name_ordinal", 0, "active named storms"),
        ("through_date", "2023-12-30", "must not precede"),
        ("direction", "unknown", "temperature/direction pair"),
    ],
)
def test_checkpoint_rejects_invalid_state(
    field: str, value: object, message: str
) -> None:
    """Reject malformed persisted state before restarting.

    Args:
        field: Checkpoint field to change.
        value: Invalid value to store.
        message: Expected validation error.

    Returns:
        None.

    Examples:
        pytest tests/test_tracker.py -k invalid_state

    Steps:
        1. Reject malformed persisted state before restarting.
    """
    tracker = WaveTracker(
        series=WaveSeries("tasmax", "warm"),
        requested_start=_START,
        source_identity="footprints",
    )
    output = tracker.advance(_batch(_START, (_component(_START, 1),)))
    document = json.loads(output.checkpoint.to_bytes())
    document[field] = value
    with pytest.raises(WaveTrackerError, match=message):
        WaveTrackerCheckpoint.from_bytes(json.dumps(document).encode())


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("name", None, "peak threshold"),
        ("name", "Vera 2024", "start year"),
        ("overlap_percent", 101, "between 0 and 100"),
    ],
)
def test_checkpoint_rejects_invalid_active_record(
    field: str, value: object, message: str
) -> None:
    """Validate names and scientific scalars when restoring active records.

    Args:
        field: Active-record field to change.
        value: Invalid value to store.
        message: Expected validation error.

    Returns:
        None.

    Examples:
        pytest tests/test_tracker.py -k invalid_active_record

    Steps:
        1. Validate names and scientific scalars when restoring active records.
    """
    tracker = WaveTracker(
        series=WaveSeries("tasmax", "warm"),
        requested_start=_START,
        source_identity="footprints",
    )
    output = tracker.advance(_batch(_START, (_component(_START, 1),)))
    document = json.loads(output.checkpoint.to_bytes())
    assert "cells" not in document["active"][0]["component"]
    document["active"][0][field] = value
    with pytest.raises(WaveTrackerError, match=message):
        WaveTrackerCheckpoint.from_bytes(json.dumps(document).encode())


@pytest.mark.unit
def test_tracker_rejects_skipped_dates_and_changed_source_on_restart() -> None:
    """Verify tracker rejects skipped dates and changed source on restart.

    Returns:
        None.

    Steps:
        1. Create a tracker checkpoint from the opening date.
        2. Try restarting after a skipped date or with a different source identity.
        3. Check that both restart requests are rejected.

    Examples:
        Run ``pytest tests/test_tracker.py -k tracker_rejects_skipped_dates_and_changed_source_on_restart``.
    """
    tracker = WaveTracker(
        series=WaveSeries("tasmax", "warm"),
        requested_start=_START,
        source_identity="footprint-files",
    )
    first = tracker.advance(_batch(_START, ()))
    with pytest.raises(WaveTrackerError, match="source_identity"):
        WaveTracker(
            series=WaveSeries("tasmax", "warm"),
            requested_start=_START + timedelta(days=1),
            source_identity="different-footprint-files",
            prior_checkpoint=first.checkpoint,
        )
    with pytest.raises(WaveTrackerError, match="selected series"):
        WaveTracker(
            series=WaveSeries("tasmin", "cold"),
            requested_start=_START + timedelta(days=1),
            source_identity="footprint-files",
            prior_checkpoint=first.checkpoint,
        )
    with pytest.raises(WaveTrackerError, match="exactly the next"):
        tracker.advance(_batch(_START + timedelta(days=2), ()))
    with pytest.raises(WaveTrackerError, match="does not touch"):
        WaveTracker(
            series=WaveSeries("tasmax", "warm"),
            requested_start=_START + timedelta(days=2),
            source_identity="footprint-files",
            prior_checkpoint=first.checkpoint,
        )


@pytest.mark.unit
def test_tracker_checks_each_geometry_once_and_keeps_exact_checkpoint_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reuse only immutable geometry checks across prior state and persistence.

    Args:
        monkeypatch: Fixture counting original geometry validation calls.

    Returns:
        None.

    Examples:
        ``pytest tests/test_tracker.py -k each_geometry`` runs this check.

    Steps:
        1. Reuse only immutable geometry checks across prior state and persistence.
    """
    checked: list[DailyWaveComponentGeometry] = []
    original = tracking_module._validate_component_geometry

    def validate(component: ComponentMembership) -> None:
        """Record and perform each real geometry check.

        Args:
            component: Daily membership to validate.

        Returns:
            None after the original check.

        Examples:
            Cached tracker saves must not invoke this again for the same object.

        Steps:
            1. Record and perform each real geometry check.
        """
        checked.append(component.geometry)
        original(component)

    monkeypatch.setattr(tracking_module, "_validate_component_geometry", validate)
    tracker = WaveTracker(
        series=WaveSeries("tasmax", "warm"),
        requested_start=_START,
        source_identity="footprint-files",
    )
    first_component = _component(_START, 1)
    first = tracker.advance(_batch(_START, (first_component,)))
    cached_bytes = tracker.checkpoint_bytes()
    tracker.checkpoint_bytes()
    assert len(checked) == 1
    assert checked[0] is first_component.geometry
    assert cached_bytes == first.checkpoint.to_bytes()
    checked.clear()
    day2 = _START + timedelta(days=1)
    second_component = _component(day2, 1)
    second = tracker.advance(_batch(day2, (second_component,)))
    second_bytes = tracker.checkpoint_bytes()
    assert len(checked) == 1
    assert checked[0] is second_component.geometry
    cache = tracker._geometry_validator._checked
    assert tuple(cache.values()) == (second_component.geometry,)
    assert second_bytes == second.checkpoint.to_bytes()
    checked.clear()
    restored = WaveTrackerCheckpoint.from_bytes(second_bytes)
    assert len(checked) == 1
    assert restored.to_bytes() == second_bytes
    tracker.close()
    assert tracker._geometry_validator._checked == {}


@pytest.mark.unit
def test_cached_geometry_still_checks_new_geometry() -> None:
    """Reject replacement invalid geometry after the preceding shapes were checked.

    Args:
        None.

    Returns:
        None.

    Examples:
        ``pytest tests/test_tracker.py -k new_geometry`` runs this check.

    Steps:
        1. Reject replacement invalid geometry after the preceding shapes were checked.
    """
    component = _component(_START, 1)
    tracker = WaveTracker(
        series=WaveSeries("tasmax", "warm"),
        requested_start=_START,
        source_identity="footprint-files",
    )
    tracker.advance(_batch(_START, (component,)))
    tracker.checkpoint_bytes()
    day2 = _START + timedelta(days=1)
    valid = _component(day2, 1)
    broken = replace(valid, geometry=replace(valid.geometry, area_grace=100.0))
    with pytest.raises(WaveTrackerError, match="area_grace"):
        tracker.advance(_batch(day2, (broken,)))


@pytest.mark.unit
def test_public_tracking_always_rechecks_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep full geometry checks for direct public tracking calls.

    Args:
        monkeypatch: Fixture counting public geometry checks.

    Returns:
        None.

    Examples:
        ``pytest tests/test_tracker.py -k public_tracking`` runs this check.

    Steps:
        1. Keep full geometry checks for direct public tracking calls.
    """
    checked = []
    original = tracking_module._validate_component_geometry

    def validate(component: ComponentMembership) -> None:
        """Record a public validation call and preserve its behavior.

        Args:
            component: Requested daily component.

        Returns:
            None after full geometry validation.

        Examples:
            Calling the public method twice performs two checks.

        Steps:
            1. Record a public validation call and preserve its behavior.
        """
        checked.append(component.geometry)
        original(component)

    monkeypatch.setattr(tracking_module, "_validate_component_geometry", validate)
    component = _component(_START, 1)
    for _ in range(2):
        track_reuters_waves(
            date=np.datetime64(_START, "D"),
            series=WaveSeries("tasmax", "warm"),
            components=(component,),
            seed=True,
        )
    assert len(checked) == 2


@pytest.mark.unit
@pytest.mark.parametrize(
    "series",
    [
        WaveSeries("tasmax", "warm"),
        WaveSeries("tasmax", "cold"),
        WaveSeries("tasmin", "warm"),
        WaveSeries("tasmin", "cold"),
    ],
)
def test_each_selected_series_starts_its_own_name_sequence(series: WaveSeries) -> None:
    """Track any selected series through empty opening dates and restart.

    Args:
        series: Selected temperature and direction.

    Returns:
        None.

    Examples:
        pytest tests/test_tracker.py -k selected_series

    Steps:
        1. Track any selected series through empty opening dates and restart.
    """
    tracker = WaveTracker(
        series=series, requested_start=_START, source_identity="footprint-files"
    )
    empty = tracker.advance(_batch(_START, (), series=series))
    assert empty.current == ()
    assert empty.checkpoint.next_name_ordinal == 0
    next_day = _START + timedelta(days=1)
    first = tracker.advance(
        _batch(next_day, (_component(next_day, 1, series=series),), series=series)
    )
    assert first.current[0].series == series
    assert first.current[0].name == f"{name_at_ordinal(0)} 2024"
    assert first.checkpoint.next_name_ordinal == 1
    restored = WaveTrackerCheckpoint.from_bytes(first.checkpoint.to_bytes())
    final_day = next_day + timedelta(days=1)
    resumed = WaveTracker(
        series=series,
        requested_start=final_day,
        source_identity="footprint-files",
        prior_checkpoint=restored,
    ).advance(
        _batch(final_day, (_component(final_day, 1, series=series),), series=series)
    )
    assert resumed.current[0].wave_id == first.current[0].wave_id
    assert resumed.current[0].name == first.current[0].name
    assert resumed.checkpoint.next_name_ordinal == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    "other_series",
    [
        WaveSeries("tasmax", "cold"),
        WaveSeries("tasmin", "warm"),
        WaveSeries("tasmin", "cold"),
    ],
)
@pytest.mark.parametrize("has_previous", [False, True])
def test_wrong_series_batch_cannot_change_tracking_or_names(
    other_series: WaveSeries, has_previous: bool
) -> None:
    """Reject incorrectly selected footprints before touching chronological state.

    Args:
        other_series: A valid series different from the selected maximum-warm one.
        has_previous: Whether a preceding named storm has already been tracked.

    Returns:
        None. Rejection retains checkpoint bytes and geometry cache, and the
        correct replacement batch yields exactly the untouched control output.

    Examples:
        Run ``pytest tests/test_tracker.py -k wrong_series_batch``.

    Steps:
        1. Reject incorrectly selected footprints before touching chronological state.
        2. Check that rejection retains checkpoint bytes and geometry cache, and the correct replacement batch yields exactly the untouched control output.
    """
    selected = WaveTracker(
        series=_SERIES, requested_start=_START, source_identity="footprint-files"
    )
    control = WaveTracker(
        series=_SERIES, requested_start=_START, source_identity="footprint-files"
    )
    day = _START
    if has_previous:
        opening = _batch(day, (_component(day, 1),))
        selected.advance(opening)
        control.advance(opening)
        day += timedelta(days=1)
    before = selected.checkpoint
    checkpoint_bytes = None if before is None else selected.checkpoint_bytes()
    cached = selected._geometry_validator._checked.copy()
    foreign = _batch(
        day,
        (_component(day, 1, west=40, east=80, series=other_series),),
        series=other_series,
    )

    with pytest.raises(
        WaveTrackerError,
        match=r"batch series.*selected tracker series",
    ):
        selected.advance(foreign)

    assert selected.checkpoint is before
    assert selected._geometry_validator._checked == cached
    if checkpoint_bytes is not None:
        assert selected.checkpoint_bytes() == checkpoint_bytes
    correct = _batch(day, (_component(day, 1, west=40, east=80),))
    assert selected.advance(correct) == control.advance(correct)


@pytest.mark.parametrize("choice", WaveSeries.choices())
def test_saved_batch_encoding_stays_exact_without_extra_validation(choice, monkeypatch):
    """Preserve saved JSON and geometry bytes across split, merge and ending.

    Args:
        choice: Each supported temperature/direction series.
        monkeypatch: Fixture detecting redundant checkpoint validation.

    Returns:
        None. Round trips every daily batch without repeating checkpoint checks.

    Examples:
        Run pytest tests/test_tracker.py -k saved_batch_encoding.

    Steps:
        1. Preserve saved JSON and geometry bytes across split, merge and ending.
        2. Check that round trips every daily batch without repeating checkpoint checks.
    """
    series = WaveSeries.parse(choice)
    days = tuple(_START + timedelta(days=offset) for offset in range(4))
    inputs = (
        _batch(days[0], (_component(days[0], 1, series=series),), series=series),
        _batch(
            days[1],
            (
                _component(days[1], 1, west=-20, east=-1, series=series),
                _component(days[1], 2, west=1, east=20, series=series),
            ),
            series=series,
        ),
        _batch(days[2], (_component(days[2], 1, series=series),), series=series),
        _batch(days[3], (), series=series),
    )
    batches = tuple(
        track_reuters_component_batches(
            inputs,
            requested_start=_START,
            source_identity="footprints",
            series=series,
        )
    )
    assert [edge.relationship for edge in batches[1].lineage] == ["split", "split"]
    assert [edge.relationship for edge in batches[2].lineage] == ["merge"]
    assert batches[3].ended
    checkpoint = batches[-1].checkpoint
    calls = []
    original = tracker_module._validate_checkpoint

    def validate_checkpoint(value):
        """Count redundant validation of the supplied final checkpoint.

        Args:
            value: State supplied to the existing tracker check.

        Returns:
            None. Retains the original validation behavior.

        Examples:
            This function should not run when encoding or decoding batches.

        Steps:
            1. Append the supplied checkpoint to the validation-call list.
            2. Delegate to the original checkpoint validator.
        """
        calls.append(value)
        return original(value)

    monkeypatch.setattr(tracker_module, "_validate_checkpoint", validate_checkpoint)
    encoded = [batch.to_bytes() for batch in batches]
    for saved in encoded:
        restored = WaveTrackingBatch.from_bytes(saved, checkpoint)
        assert restored.to_bytes() == saved
        assert restored.checkpoint is checkpoint
    assert calls == []


def test_existing_saved_batches_keep_exact_bytes():
    """Read pre-refactor saved records and reproduce their JSON and WKB bytes.

    Args:
        None.

    Returns:
        None. Verifies named records, split edges, mergers and explicit endings.

    Examples:
        Run pytest tests/test_tracker.py -k existing_saved_batches.

    Steps:
        1. Load the saved fixture batch documents and construct the matching checkpoint.
        2. Decode and re-encode each document.
        3. Compare the resulting JSON bytes with the original compact representation.
    """
    documents = json.loads(
        (Path(__file__).parent / "data" / "saved-wave-batches.json").read_text()
    )
    checkpoint = WaveTrackerCheckpoint(
        requested_start=_START,
        through_date=_START + timedelta(days=3),
        source_identity="footprints",
        series=WaveSeries("tasmax", "warm"),
        active=(),
    )
    for document in documents:
        original = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
        restored = WaveTrackingBatch.from_bytes(original, checkpoint)
        assert restored.to_bytes() == original


@pytest.mark.parametrize(
    "saved",
    [
        b"bad json",
        b"\xff",
        b"[]",
        b"{}",
        b'{"date":"20240101","current":[],"ended":[],"lineage":[]}',
    ],
)
def test_saved_batch_rejects_invalid_document(saved):
    """Reject unreadable JSON, wrong fields and non-calendar date strings.

    Args:
        saved: Invalid saved batch bytes.

    Returns:
        None. Confirms decoding fails before a batch is returned.

    Examples:
        Run pytest tests/test_tracker.py -k invalid_document.

    Steps:
        1. Construct an empty checkpoint for the supplied malformed saved bytes.
        2. Decode the document and check that it raises the saved-batch validation error.
    """
    checkpoint = WaveTrackerCheckpoint(
        requested_start=_START,
        through_date=_START,
        source_identity="footprints",
        series=_SERIES,
        active=(),
    )
    with pytest.raises(WaveTrackerError, match="saved daily batch"):
        WaveTrackingBatch.from_bytes(saved, checkpoint)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("ordinal", "expected"),
    [
        (2196, "Thabiso"),
        (2217, "Lijun"),
        (2235, "Vera"),
        (2236, "Yakubu"),
        (2300, "Stacy"),
        (2307, "Jia"),
        (2323, "Mahesh"),
        (2480, "Raimundo"),
    ],
)
def test_checkpoint_restores_target_declaration_names(
    ordinal: int, expected: str
) -> None:
    """Name target declarations identically after checkpoint serialization.

    Args:
        ordinal: Next declaration position in the deterministic naming sequence.
        expected: Given name expected at that position.

    Returns:
        None. Confirms assignment and continuation after a tracker restart.

    Examples:
        Run ``pytest tests/test_tracker.py -k target_declaration``.

    Steps:
        1. Serialize and restore a checkpoint at the parametrized next-name ordinal.
        2. Advance direct and restored trackers with the same next-day component.
        3. Compare assigned names and continued tracking state.
    """
    tracker = WaveTracker(
        series=_SERIES, requested_start=_START, source_identity="footprint-files"
    )
    opening = tracker.advance(_batch(_START, ()))
    checkpoint = replace(opening.checkpoint, next_name_ordinal=ordinal)
    restored = WaveTrackerCheckpoint.from_bytes(checkpoint.to_bytes())
    next_day = _START + timedelta(days=1)
    component = _component(next_day, 1)
    batch = _batch(next_day, (component,))
    direct = WaveTracker(
        series=_SERIES,
        requested_start=next_day,
        source_identity="footprint-files",
        prior_checkpoint=checkpoint,
    ).advance(batch)
    resumed = WaveTracker(
        series=_SERIES,
        requested_start=next_day,
        source_identity="footprint-files",
        prior_checkpoint=restored,
    ).advance(batch)
    assert resumed.checkpoint.to_bytes() == direct.checkpoint.to_bytes()
    assert resumed.current[0].name == f"{expected} 2024"
    assert resumed.checkpoint.next_name_ordinal == ordinal + 1
    following_day = next_day + timedelta(days=1)
    continued = WaveTracker(
        series=_SERIES,
        requested_start=following_day,
        source_identity="footprint-files",
        prior_checkpoint=WaveTrackerCheckpoint.from_bytes(
            resumed.checkpoint.to_bytes()
        ),
    ).advance(_batch(following_day, (_component(following_day, 1),)))
    assert continued.current[0].name == resumed.current[0].name
    assert continued.checkpoint.next_name_ordinal == ordinal + 1


def test_january_archive_seed_names_from_zero_and_resumes_exactly():
    """Keep storm births and names at January even when cells have earlier history.

    Args:
        None.

    Returns:
        None. First January footprint consumes ordinal zero and resumes identically.

    Examples:
        pytest tests/test_tracker.py -k january_archive

    Steps:
        1. Keep storm births and names at January even when cells have earlier history.
        2. Check that first January footprint consumes ordinal zero and resumes identically.
    """
    start = date(1960, 1, 1)
    first = _batch(start, (_component(start, 1),))
    tracker = WaveTracker(series=_SERIES, requested_start=start, source_identity="jan")
    opening = tracker.advance(first)
    assert opening.current[0].start_date == np.datetime64(start, "D")
    assert opening.current[0].name == f"{name_at_ordinal(0)} 1960"
    assert opening.checkpoint.next_name_ordinal == 1
    tomorrow = start + timedelta(days=1)
    batch = _batch(tomorrow, (_component(tomorrow, 2),))
    restored = WaveTracker(
        series=_SERIES,
        requested_start=tomorrow,
        source_identity="jan",
        prior_checkpoint=WaveTrackerCheckpoint.from_bytes(tracker.checkpoint_bytes()),
    )
    assert restored.advance(batch).to_bytes() == tracker.advance(batch).to_bytes()
