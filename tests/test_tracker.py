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
    order_digest: str | None = None,
) -> ComponentMembership:
    """Create a large qualifying footprint eligible for a Reuters name.

    Args:
        day: Date of the component.
        label: Daily component label.
        west: Western boundary in degrees.
        east: Eastern boundary in degrees.
        series: Temperature and direction used for component identity.
        order_digest: Explicit synthetic ordering token. The component ID is
            the default token because these test polygons are not native cells.

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
    component_id = component_identity(
        temperature=series.temperature,
        direction=series.direction,
        date=day_value,
        geometry=geometry,
    )
    return ComponentMembership(
        component_id=component_id,
        order_digest=component_id if order_digest is None else order_digest,
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


def _unnamed_component(
    day: date, label: int, *, west: float, east: float, order_digest: str
) -> ComponentMembership:
    """Supply a large active footprint with a small qualifying core.

    Args:
        day: Calendar date of the synthetic component.
        label: Unique positive component label for this date.
        west: Western boundary of the complete footprint.
        east: Eastern boundary of the complete footprint.
        order_digest: Explicit synthetic ordering digest for this date.

    Returns:
        A component below the naming threshold whose full coverage can continue
        on later dates before its qualifying core grows.

    Examples:
        component = _unnamed_component(_START, 1, west=-80, east=-40, order_digest="0" * 32)
    """
    component = _component(day, label, west=west, east=east, order_digest=order_digest)
    core = box(west, -20, west + 2, 20)
    grace = component.geometry.full_geometry.difference(core)
    qualified_area = float(
        transform(_AREA_TRANSFORMER.transform, core).area / 1_000_000
    )
    grace_area = float(transform(_AREA_TRANSFORMER.transform, grace).area / 1_000_000)
    return replace(
        component,
        geometry=replace(
            component.geometry,
            geometry=core,
            grace_geometry=grace,
            area_qualified=qualified_area,
            area_grace=grace_area,
            qualified_fraction=qualified_area / (qualified_area + grace_area),
        ),
    )


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


def test_saved_batches_keep_exact_bytes():
    """Read saved full-digest records and reproduce their JSON and WKB bytes.

    Args:
        None.

    Returns:
        None. Verifies named records, split edges, mergers and explicit endings.

    Examples:
        Run pytest tests/test_tracker.py -k saved_batches_keep_exact_bytes.

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


@pytest.mark.unit
def test_input_labels_do_not_change_saved_state_or_resumed_endings() -> None:
    """Resume scientific state after temporary input-row labels are omitted.

    Args:
        None.

    Returns:
        None. Direct and resumed records have equal scientific bytes, including
        names and explicit endings, while today's input labels remain available.

    Examples:
        Run pytest tests/test_tracker.py -k input_labels_do_not_change.

    Steps:
        1. Track labelled input and restore its checkpoint without those labels.
        2. Continue both trackers with the next labelled component.
        3. Restore the continued state and compare direct and resumed endings.
    """
    first = _batch(_START, (replace(_component(_START, 1), input_id="1" * 32),))
    direct = WaveTracker(
        series=_SERIES, requested_start=_START, source_identity="footprints"
    )
    opening = direct.advance(first)
    saved = opening.checkpoint.to_bytes()
    assert "input_id" not in json.loads(saved)["active"][0]["component"]
    restored = WaveTrackerCheckpoint.from_bytes(saved)
    assert restored.active[0].component.input_id is None
    resumed = WaveTracker(
        series=_SERIES,
        requested_start=_START + timedelta(days=1),
        source_identity="footprints",
        prior_checkpoint=restored,
    )
    next_day = _START + timedelta(days=1)
    second = _batch(next_day, (replace(_component(next_day, 1), input_id="2" * 32),))
    continued = direct.advance(second)
    resumed_continuation = resumed.advance(second)
    assert resumed_continuation.to_bytes() == continued.to_bytes()
    assert resumed_continuation.current[0].component.input_id == "2" * 32
    assert continued.current[0].wave_id == opening.current[0].wave_id
    assert continued.current[0].name == opening.current[0].name
    ending_tracker = WaveTracker(
        series=_SERIES,
        requested_start=_START + timedelta(days=2),
        source_identity="footprints",
        prior_checkpoint=WaveTrackerCheckpoint.from_bytes(
            continued.checkpoint.to_bytes()
        ),
    )
    empty = _batch(_START + timedelta(days=2), ())
    direct_ending = direct.advance(empty)
    resumed_ending = ending_tracker.advance(empty)
    assert resumed_ending.to_bytes() == direct_ending.to_bytes()
    assert direct_ending.ended[0].final_record.component.input_id == "2" * 32
    assert resumed_ending.ended[0].final_record.component.input_id is None


@pytest.mark.unit
def test_checkpoint_rejects_shortened_permanent_identity() -> None:
    """Reject an old shortened key before it can continue under the shared rule.

    Args:
        None.

    Returns:
        None. Restoring the shortened active key raises the checkpoint error.

    Examples:
        Run pytest tests/test_tracker.py -k shortened_permanent_identity.

    Steps:
        1. Create valid full-digest tracker state.
        2. Replace its permanent key with the former eight-character suffix.
        3. Require checkpoint restoration to reject the changed state.
    """
    tracker = WaveTracker(
        series=_SERIES, requested_start=_START, source_identity="footprints"
    )
    tracker.advance(_batch(_START, (_component(_START, 1),)))
    document = json.loads(tracker.checkpoint_bytes())
    document["active"][0]["wave_id"] = document["active"][0]["wave_id"][:17]
    with pytest.raises(WaveTrackerError, match="previous wave IDs"):
        WaveTrackerCheckpoint.from_bytes(json.dumps(document).encode())


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


@pytest.mark.unit
def test_delayed_names_keep_birth_order_after_daily_reorder_and_resume() -> None:
    """Keep original naming order when active footprints qualify on a later day.

    Args:
        None.

    Returns:
        None. Quiet dates consume no names, and a restart retains birth order,
        names, years and the next unused sequence position.

    Examples:
        pytest tests/test_tracker.py -k delayed_names_keep_birth_order
    """
    second_day = _START + timedelta(days=1)
    naming_day = _START + timedelta(days=2)
    batches = (
        _batch(
            _START,
            (
                _unnamed_component(
                    _START, 1, west=-80, east=-40, order_digest="f" * 32
                ),
                _unnamed_component(_START, 2, west=40, east=80, order_digest="0" * 32),
            ),
        ),
        _batch(
            second_day,
            (
                _unnamed_component(
                    second_day, 1, west=-80, east=-40, order_digest="a" * 32
                ),
                _unnamed_component(
                    second_day, 2, west=40, east=80, order_digest="b" * 32
                ),
            ),
        ),
        _batch(
            naming_day,
            (
                _component(naming_day, 1, west=-80, east=-40, order_digest="0" * 32),
                _component(naming_day, 2, west=40, east=80, order_digest="f" * 32),
            ),
        ),
    )
    direct = tuple(
        track_reuters_component_batches(
            batches, series=_SERIES, requested_start=_START, source_identity="synthetic"
        )
    )
    assert [output.checkpoint.next_name_ordinal for output in direct] == [0, 0, 2]
    assert all(
        record.name is None for output in direct[:2] for record in output.current
    )
    restored = WaveTrackerCheckpoint.from_bytes(direct[1].checkpoint.to_bytes())
    resumed = WaveTracker(
        series=_SERIES,
        requested_start=naming_day,
        source_identity="synthetic",
        prior_checkpoint=restored,
    ).advance(batches[2])
    assert resumed.to_bytes() == direct[2].to_bytes()
    assert resumed.checkpoint.to_bytes() == direct[2].checkpoint.to_bytes()
    records = {
        record.component.geometry.full_geometry.bounds[0]: record
        for record in resumed.current
    }
    assert records[-80].name == f"{name_at_ordinal(1)} 2023"
    assert records[40].name == f"{name_at_ordinal(0)} 2023"
    assert records[-80].birth_order_digest == "f" * 32
    assert records[-80].component.order_digest == "0" * 32
    assert records[40].birth_order_digest == "0" * 32
    final_day = naming_day + timedelta(days=1)
    ended = WaveTracker(
        series=_SERIES,
        requested_start=final_day,
        source_identity="synthetic",
        prior_checkpoint=WaveTrackerCheckpoint.from_bytes(
            resumed.checkpoint.to_bytes()
        ),
    ).advance(_batch(final_day, ()))
    assert ended.checkpoint.active == ()
    assert ended.checkpoint.next_name_ordinal == 2
    assert {item.final_record.name for item in ended.ended} == {
        record.name for record in resumed.current
    }
    assert {item.final_record.birth_order_digest for item in ended.ended} == {
        "0" * 32,
        "f" * 32,
    }


@pytest.mark.unit
def test_naming_uses_birth_date_before_birth_digest() -> None:
    """Give an older qualifying wave its name before a lower-digest new birth.

    Args:
        None.

    Returns:
        None. Name years and sequence positions follow birth chronology.

    Examples:
        pytest tests/test_tracker.py -k naming_uses_birth_date
    """
    tracker = WaveTracker(
        series=_SERIES, requested_start=_START, source_identity="synthetic"
    )
    tracker.advance(
        _batch(
            _START,
            (_unnamed_component(_START, 1, west=-80, east=-40, order_digest="f" * 32),),
        )
    )
    next_day = _START + timedelta(days=1)
    output = tracker.advance(
        _batch(
            next_day,
            (
                _component(next_day, 1, west=-80, east=-40, order_digest="a" * 32),
                _component(next_day, 2, west=40, east=80, order_digest="0" * 32),
            ),
        )
    )
    records = sorted(output.current, key=lambda record: record.start_date)
    assert records[0].name == f"{name_at_ordinal(0)} 2023"
    assert records[1].name == f"{name_at_ordinal(1)} 2024"
    assert output.checkpoint.next_name_ordinal == 2


@pytest.mark.unit
def test_full_birth_digest_orders_names_independently_of_permanent_ids() -> None:
    """Use the full suffix for a tie in the original eight-character prefix.

    Args:
        None.

    Returns:
        None. Opposite cell-ID and birth-digest orders assign names by the latter.

    Examples:
        pytest tests/test_tracker.py -k full_birth_digest_orders_names
    """
    first, second = sorted(
        (
            _component(_START, 1, west=-80, east=-40),
            _component(_START, 2, west=40, east=80),
        ),
        key=lambda component: component.component_id,
    )
    first = replace(first, order_digest="01234567" + "f" * 24)
    second = replace(second, order_digest="01234567" + "0" * 24)
    output = WaveTracker(
        series=_SERIES, requested_start=_START, source_identity="synthetic"
    ).advance(_batch(_START, (first, second)))
    records = {record.component_id: record for record in output.current}
    assert records[first.component_id].name == f"{name_at_ordinal(1)} 2023"
    assert records[second.component_id].name == f"{name_at_ordinal(0)} 2023"
    assert output.checkpoint.next_name_ordinal == 2


@pytest.mark.unit
def test_split_and_merge_keep_each_wave_birth_order_and_name() -> None:
    """Keep survivor naming facts while a split child gets its own birth order.

    Args:
        None.

    Returns:
        None. A merge retains the survivor name and ending child state without
        consuming another name or transferring the child's name.

    Examples:
        pytest tests/test_tracker.py -k split_and_merge_keep_each_wave_birth_order
    """
    tracker = WaveTracker(
        series=_SERIES, requested_start=_START, source_identity="synthetic"
    )
    opening = tracker.advance(
        _batch(_START, (_component(_START, 1, order_digest="f" * 32),))
    )
    second_day = _START + timedelta(days=1)
    split = tracker.advance(
        _batch(
            second_day,
            (
                _component(second_day, 1, west=-20, east=5, order_digest="b" * 32),
                _component(second_day, 2, west=5, east=20, order_digest="0" * 32),
            ),
        )
    )
    survivor = next(record for record in split.current if record.continues)
    child = next(record for record in split.current if not record.continues)
    assert survivor.birth_order_digest == "f" * 32
    assert survivor.name == opening.current[0].name
    assert child.birth_order_digest == "0" * 32
    assert child.name == f"{name_at_ordinal(1)} 2024"
    assert child.name != survivor.name
    third_day = second_day + timedelta(days=1)
    merged = WaveTracker(
        series=_SERIES,
        requested_start=third_day,
        source_identity="synthetic",
        prior_checkpoint=WaveTrackerCheckpoint.from_bytes(split.checkpoint.to_bytes()),
    ).advance(_batch(third_day, (_component(third_day, 1, order_digest="a" * 32),)))
    assert merged.current[0].birth_order_digest == survivor.birth_order_digest
    assert merged.current[0].name == survivor.name
    assert merged.ended[0].final_record.birth_order_digest == child.birth_order_digest
    assert merged.ended[0].final_record.name == child.name
    assert merged.checkpoint.next_name_ordinal == 2


@pytest.mark.unit
@pytest.mark.parametrize("field", ["order_digest", "birth_order_digest"])
@pytest.mark.parametrize(
    "mutation", ["missing", "null", "short", "uppercase", "number"]
)
def test_checkpoint_requires_complete_daily_and_birth_order_digests(
    field: str, mutation: str
) -> None:
    """Reject missing ordering state instead of deriving it from a later shape.

    Args:
        field: Required daily or retained birth ordering field.
        mutation: Missing value or malformed value to place in the saved record.

    Returns:
        None. Resume reports the specific rejected field.

    Examples:
        pytest tests/test_tracker.py -k requires_complete_daily_and_birth_order
    """
    output = WaveTracker(
        series=_SERIES, requested_start=_START, source_identity="synthetic"
    ).advance(_batch(_START, (_component(_START, 1),)))
    document = json.loads(output.checkpoint.to_bytes())
    if mutation == "missing":
        del document["active"][0][field]
    else:
        document["active"][0][field] = {
            "null": None,
            "short": "a" * 31,
            "uppercase": "A" * 32,
            "number": 7,
        }[mutation]
    with pytest.raises(WaveTrackerError, match=field):
        WaveTrackerCheckpoint.from_bytes(json.dumps(document).encode())


@pytest.mark.unit
def test_checkpoint_rejects_birth_order_that_disagrees_with_birth_component() -> None:
    """Require a newborn wave to retain its actual first-day ordering digest.

    Args:
        None.

    Returns:
        None. A structurally valid replacement digest still cannot alter birth state.

    Examples:
        pytest tests/test_tracker.py -k disagrees_with_birth_component
    """
    output = WaveTracker(
        series=_SERIES, requested_start=_START, source_identity="synthetic"
    ).advance(_batch(_START, (_component(_START, 1),)))
    document = json.loads(output.checkpoint.to_bytes())
    document["active"][0]["birth_order_digest"] = "e" * 32
    with pytest.raises(WaveTrackerError, match="birth component"):
        WaveTrackerCheckpoint.from_bytes(json.dumps(document).encode())


@pytest.mark.unit
def test_checkpoint_rejects_duplicate_full_birth_ordering_keys() -> None:
    """Reject two active waves claiming the same birth date and full digest.

    Args:
        None.

    Returns:
        None. Valid but duplicated retained ordering state prevents resume.

    Examples:
        pytest tests/test_tracker.py -k duplicate_full_birth_ordering_keys
    """
    tracker = WaveTracker(
        series=_SERIES, requested_start=_START, source_identity="synthetic"
    )
    tracker.advance(
        _batch(
            _START,
            (
                _component(_START, 1, west=-80, east=-40, order_digest="0" * 32),
                _component(_START, 2, west=40, east=80, order_digest="f" * 32),
            ),
        )
    )
    next_day = _START + timedelta(days=1)
    output = tracker.advance(
        _batch(
            next_day,
            (
                _component(next_day, 1, west=-80, east=-40, order_digest="a" * 32),
                _component(next_day, 2, west=40, east=80, order_digest="b" * 32),
            ),
        )
    )
    document = json.loads(output.checkpoint.to_bytes())
    document["active"][1]["birth_order_digest"] = document["active"][0][
        "birth_order_digest"
    ]
    with pytest.raises(WaveTrackerError, match="unique birth ordering keys"):
        WaveTrackerCheckpoint.from_bytes(json.dumps(document).encode())
