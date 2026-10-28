"""Protect issued labels and stable future naming across independent callers."""

from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pytest
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform

from reuters_temperature_storms import (
    PublishedStormNames,
    StormNames,
    load_published_storm_names,
)
from reuters_temperature_storms.components import (
    ComponentMembership,
    DailyWaveComponentGeometry,
)
from reuters_temperature_storms.identity import CellIdentity, FootprintIdentity
from reuters_temperature_storms.naming import StormNamingError, name_at_ordinal
from reuters_temperature_storms.series import WaveSeries
from reuters_temperature_storms.tracker import (
    DailyComponentBatch,
    WaveTracker,
    WaveTrackerCheckpoint,
)

_SERIES = WaveSeries("tasmax", "warm")
_CUTOFF = date(2026, 9, 30)


def _assign(names, *, day, cell_id, naming_day=None, ordinal=2527, series=_SERIES):
    """Request a name using one complete set of native birth facts.

    Args:
        names: Shared RTS naming policy under test.
        day: Physical storm birth day.
        cell_id: Original occupied native cell address.
        naming_day: First qualifying day, or None to use the birth day.
        ordinal: Next position in the Reuters sequence.
        series: Tracked temperature and direction.

    Returns:
        The assigned full Reuters label.

    Examples:
        ``_assign(names, day=day, cell_id=519119)`` names that physical birth.
    """
    return names.assign(
        series=series,
        wave_id=CellIdentity(np.datetime64(day, "D"), cell_id).wave_id,
        birth_date=day,
        cell_id=cell_id,
        naming_date=day if naming_day is None else naming_day,
        ordinal=ordinal,
    )


def test_package_contains_all_issued_labels_and_exact_known_births():
    """Load every issued label and check independently verified story-era births.

    Args:
        None.

    Returns:
        None.

    Examples:
        Run ``pytest tests/test_published_names.py -k all_issued``.
    """
    published = load_published_storm_names()
    assert published.through_date == _CUTOFF
    assert len(published.assignments) == 2527
    assert len({row.wave_id for row in published.assignments}) == 2527
    assert len({row.name for row in published.assignments}) == 2527
    by_name = {row.name: row for row in published.assignments}
    assert by_name["Uriel 1960"].wave_id == "19601018-b2322edc221021ce1ee5f6e01cf8f051"
    assert by_name["Wendy 1987"].cell_id == 262501
    assert by_name["Yiyi 1987"].cell_id == 350294
    assert load_published_storm_names() is published


def test_published_names_survive_reversed_digest_priority():
    """Retain Wendy and Yiyi even though the new hashes reverse their old order.

    Args:
        None.

    Returns:
        None.

    Examples:
        Run ``pytest tests/test_published_names.py -k reversed_digest``.
    """
    day = date(1987, 3, 24)
    assert (
        CellIdentity(np.datetime64(day), 262501).digest
        > CellIdentity(np.datetime64(day), 350294).digest
    )
    for names in (StormNames(), StormNames()):
        assert _assign(names, day=day, cell_id=350294, ordinal=0) == "Yiyi 1987"
        assert _assign(names, day=day, cell_id=262501, ordinal=1) == "Wendy 1987"


def test_all_issued_assignments_ignore_new_sequence_positions():
    """Check every packaged historical label through the actual assignment API.

    Args:
        None.

    Returns:
        None.

    Examples:
        Run ``pytest tests/test_published_names.py -k ignore_new_sequence``.
    """
    names = StormNames()
    for assignment in reversed(load_published_storm_names().assignments):
        assert (
            _assign(
                names,
                day=assignment.birth_date,
                cell_id=assignment.cell_id,
                naming_day=_CUTOFF,
                ordinal=0,
            )
            == assignment.name
        )


def test_missing_historical_match_stops_without_spending_a_name():
    """Reject unknown historical eligibility while leaving future lookup unchanged.

    Args:
        None.

    Returns:
        None.

    Examples:
        Run ``pytest tests/test_published_names.py -k missing_historical``.
    """
    names = StormNames()
    with pytest.raises(StormNamingError, match="missing published name"):
        _assign(names, day=_CUTOFF, cell_id=0)
    future = _CUTOFF + timedelta(days=1)
    assert _assign(names, day=future, cell_id=0) == f"{name_at_ordinal(2527)} 2026"


def test_older_unnamed_storm_can_first_qualify_after_the_cutoff():
    """Allow future first qualification without inventing a historical label.

    Args:
        None.

    Returns:
        None.

    Examples:
        Run ``pytest tests/test_published_names.py -k older_unnamed``.
    """
    assert (
        _assign(
            StormNames(),
            day=_CUTOFF,
            cell_id=0,
            naming_day=_CUTOFF + timedelta(days=1),
        )
        == f"{name_at_ordinal(2527)} 2026"
    )


def test_other_series_do_not_claim_published_warm_high_names():
    """Keep independent temperature and direction series outside this name table.

    Args:
        None.

    Returns:
        None.

    Examples:
        Run ``pytest tests/test_published_names.py -k other_series``.
    """
    for series in (WaveSeries("tasmin", "warm"), WaveSeries("tasmax", "cold")):
        assert (
            _assign(
                StormNames(),
                day=date(1960, 1, 1),
                cell_id=0,
                ordinal=0,
                series=series,
            )
            == f"{name_at_ordinal(0)} 1960"
        )


def test_duplicate_and_wrong_birth_assignments_are_rejected():
    """Reject conflicting labels and a key that does not identify its native birth.

    Args:
        None.

    Returns:
        None.

    Examples:
        Run ``pytest tests/test_published_names.py -k duplicate_and_wrong``.
    """
    assignment = load_published_storm_names().assignments[0]
    with pytest.raises(StormNamingError, match="duplicate"):
        PublishedStormNames((assignment, assignment))
    with pytest.raises(StormNamingError, match="does not match"):
        replace(assignment, cell_id=assignment.cell_id + 1)
    with pytest.raises(StormNamingError, match="later than"):
        PublishedStormNames((assignment,), through_date=date(1959, 12, 31))


def test_saved_published_name_cannot_be_swapped_or_missing():
    """Reject corrupt historical resume labels, including an unlisted birth key.

    Args:
        None.

    Returns:
        None.

    Examples:
        Run ``pytest tests/test_published_names.py -k saved_published``.
    """
    names = StormNames()
    kwargs = {
        "series": _SERIES,
        "wave_id": "19870324-ed909ccd8f5f0433953dfe60175b30b7",
        "as_of_date": _CUTOFF,
    }
    names.validate_existing(**kwargs, name="Wendy 1987")
    with pytest.raises(StormNamingError, match="differs"):
        names.validate_existing(**kwargs, name="Yiyi 1987")
    with pytest.raises(StormNamingError, match="no published assignment"):
        names.validate_existing(
            series=_SERIES,
            wave_id=CellIdentity(np.datetime64(_CUTOFF), 0).wave_id,
            name="Uriel 2026",
            as_of_date=_CUTOFF,
        )


def test_assignment_requires_correct_retained_birth_facts():
    """Reject a name request whose stored original cell differs from its key.

    Args:
        None.

    Returns:
        None.

    Examples:
        Run ``pytest tests/test_published_names.py -k correct_retained``.
    """
    with pytest.raises(StormNamingError, match="disagrees"):
        StormNames().assign(
            series=_SERIES,
            wave_id="19870324-ed909ccd8f5f0433953dfe60175b30b7",
            birth_date=date(1987, 3, 24),
            cell_id=350294,
            naming_date=date(1987, 3, 24),
            ordinal=0,
        )


def _moving_component(day, label, full, *, first_day=False):
    """Create a native-cell footprint that qualifies after its opening day.

    Args:
        day: Observation date of the invented footprint.
        label: Input label used to follow the footprint between days.
        full: Complete native-cell coverage of the footprint.
        first_day: Whether just one original cell is qualifying.

    Returns:
        Measured RTS component with its native daily identity.

    Examples:
        ``_moving_component(day, 1, full, first_day=True)`` starts unnamed.
    """
    projection = Transformer.from_crs(4326, 6933, always_xy=True)
    if first_day:
        west, south, _, _ = full.bounds
        qualified = box(west, south, west + 0.25, south + 0.25)
    else:
        qualified = full
    full_area = transform(projection.transform, full).area / 1_000_000
    area = transform(projection.transform, qualified).area / 1_000_000
    day_value = np.datetime64(day, "D")
    return ComponentMembership(
        component_id=FootprintIdentity(day_value, full).digest,
        geometry=DailyWaveComponentGeometry(
            time=day_value,
            label=label,
            geometry=qualified,
            grace_geometry=None,
            full_geometry=full,
            area_qualified=area,
            area_grace=0.0,
            area_gap_filled=full_area - area,
            qualified_fraction=area / full_area,
        ),
    )


def test_delayed_future_names_use_original_cells_after_movement_and_restart():
    """Preserve birth priority when both present-day cells and hashes reverse it.

    Args:
        None.

    Returns:
        None. Direct and resumed complete batches agree byte for byte.

    Examples:
        Run ``pytest tests/test_published_names.py -k delayed_future``.
    """
    start = date(2027, 1, 1)
    series = WaveSeries("tasmax", "warm")
    opening = (
        _moving_component(
            start, 1, box(-100.125, -20.125, -49.875, 0.125), first_day=True
        ),
        _moving_component(
            start, 2, box(19.875, -19.125, 70.125, 1.125), first_day=True
        ),
    )
    assert opening[0].component_id > opening[1].component_id
    tracker = WaveTracker(
        series=series, requested_start=start, source_identity="moving-native-cells"
    )
    first = tracker.advance(DailyComponentBatch(start, series, opening))
    by_label = {row.component.geometry.label: row for row in first.current}
    assert all(row.name is None for row in first.current)
    assert by_label[1].birth_cell_id == 403519
    assert by_label[2].birth_cell_id == 409759
    saved = first.checkpoint.to_bytes()
    restored = WaveTrackerCheckpoint.from_bytes(saved)
    assert restored.to_bytes() == saved
    resumed = WaveTracker(
        series=series,
        requested_start=start + timedelta(days=1),
        source_identity="moving-native-cells",
        prior_checkpoint=restored,
    )
    following_day = start + timedelta(days=1)
    following = (
        _moving_component(following_day, 2, box(19.875, -19.125, 70.125, 1.125)),
        _moving_component(following_day, 1, box(-100.125, -18.125, -49.875, 2.125)),
    )
    assert (
        FootprintIdentity(
            np.datetime64(following_day), following[0].geometry.full_geometry
        ).cell_id
        < FootprintIdentity(
            np.datetime64(following_day), following[1].geometry.full_geometry
        ).cell_id
    )
    batch = DailyComponentBatch(following_day, series, following)
    direct = tracker.advance(batch)
    restarted = resumed.advance(batch)
    assert restarted.to_bytes() == direct.to_bytes()
    named = {row.wave_id: row for row in direct.current}
    assert named[by_label[1].wave_id].name == f"{name_at_ordinal(0)} 2027"
    assert named[by_label[2].wave_id].name == f"{name_at_ordinal(1)} 2027"
    assert named[by_label[1].wave_id].birth_cell_id == 403519
    assert named[by_label[2].wave_id].birth_cell_id == 409759
    assert direct.checkpoint.next_name_ordinal == 2
