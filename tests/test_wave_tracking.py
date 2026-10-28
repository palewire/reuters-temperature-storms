"""Tests for geometry-key Reuters wave tracking."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pytest
import shapely
from pyproj import Transformer
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.ops import transform

import reuters_temperature_storms._wave_tracking as wave_tracking
from reuters_temperature_storms._wave_tracking import (
    WaveSeries,
    WaveTrackingError,
    _candidate_geometry_indices,
    _full_area,
    _Overlap,
    _overlap_percent,
    _overlaps,
    _spatial_index,
    _validate_output,
    track_reuters_waves,
)
from reuters_temperature_storms.components import (
    ComponentMembership,
    DailyWaveComponentGeometry,
)
from reuters_temperature_storms.identity import FootprintIdentity, component_identity

_AREA_TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
_SERIES = WaveSeries("tasmax", "warm")
_DAY_1 = np.datetime64("2024-01-01")
_DAY_2 = np.datetime64("2024-01-02")
_DAY_3 = np.datetime64("2024-01-03")


def _area(geometry):
    """Measure a geographic polygon in square kilometers using equal-area projection.

    Args:
        geometry: Geographic polygon whose equal-area measurement is requested.

    Returns:
        Polygon area in square kilometers.

    Steps:
        1. Project the geographic polygon into the equal-area coordinate system.
        2. Divide its square-meter area by one million.

    Examples:
        Used by ``pytest tests/test_wave_tracking.py`` for fixture construction and output checks.
    """
    return float(transform(_AREA_TRANSFORMER.transform, geometry).area / 1_000_000)


def _component(
    day: np.datetime64,
    label: int,
    full,
    *,
    qualified=None,
    grace=None,
    order_digest: str | None = None,
) -> ComponentMembership:
    """Build a synthetic daily component with measured role areas and a geometry key.

    Args:
        day: Observation date for the synthetic tracking input.
        label: Component label in the synthetic daily input.
        full: Complete polygon including all component roles.
        qualified: Optional qualified polygon; defaults to the full polygon.
        grace: Optional grace-period polygon.
        order_digest: Explicit synthetic ordering token. The component ID is
            the default token because these test polygons are not native cells.

    Returns:
        Synthetic component with geometry identity and measured role areas.

    Steps:
        1. Choose the supplied qualified and grace polygons and measure their areas.
        2. Build the daily geometry record and compute the full polygon's component identifier.
        3. Return the identified component membership.

    Examples:
        Used by ``pytest tests/test_wave_tracking.py`` for fixture construction and output checks.
    """
    core = full if qualified is None else qualified
    qualified_area = _area(core)
    grace_area = _area(grace) if grace is not None else 0.0
    geometry = DailyWaveComponentGeometry(
        time=day,
        label=label,
        geometry=core,
        grace_geometry=grace,
        full_geometry=full,
        area_qualified=qualified_area,
        area_grace=grace_area,
        area_gap_filled=_area(full) - qualified_area - grace_area,
        qualified_fraction=qualified_area / _area(full),
    )
    component_id = component_identity(
        temperature=_SERIES.temperature,
        direction=_SERIES.direction,
        date=day,
        geometry=full,
    )
    return ComponentMembership(
        component_id,
        geometry,
        component_id if order_digest is None else order_digest,
    )


def _seed(day: np.datetime64, *components: ComponentMembership):
    """Initialize tracking from synthetic components on the requested date.

    Args:
        day: Observation date for the synthetic tracking input.
        components: Synthetic components supplied to the initial tracker call.

    Returns:
        Tracking result for the first observation date.

    Steps:
        1. Pass the supplied components to the tracker with initial seeding enabled.

    Examples:
        Used by ``pytest tests/test_wave_tracking.py`` for fixture construction and output checks.
    """
    return track_reuters_waves(
        date=day, series=_SERIES, components=components, seed=True
    )


@pytest.mark.unit
def test_seed_ids_use_geometry_keys_and_ignore_discovery_order() -> None:
    """Compare tracked records and check identities derive from geometry rather than discovery labels.

    Returns:
        None.

    Steps:
        1. Seed the same two polygons in forward and reverse order.
        2. Compare tracked records and check identities derive from geometry rather than discovery labels.

    Examples:
        Run ``pytest tests/test_wave_tracking.py -k seed_ids_use_geometry_keys_and_ignore_discovery_order``.
    """
    first = _component(_DAY_1, 1, box(0, 0, 4, 1))
    second = _component(_DAY_1, 2, box(5, 0, 9, 1))

    forward = _seed(_DAY_1, first, second)
    reverse = _seed(_DAY_1, second, first)

    assert forward.current == reverse.current
    assert all(
        len(record.component_id) == 32 and len(record.wave_id) == 41
        for record in forward.current
    )
    # The fixed vectors hash date plus the known first cells 519119 and 519139.
    assert {record.wave_id for record in forward.current} == {
        "20240101-32c596b38c0f962c549c745f65580edd",
        "20240101-0f6cc26746ca7b71935a7cf774bfc3d8",
    }
    assert all(record.parent_wave_ids == () for record in forward.current)


@pytest.mark.unit
def test_continuation_retains_wave_id_and_replaces_component_id() -> None:
    """Check that the persistent wave identifier stays fixed while the dated component identifier changes.

    Returns:
        None.

    Steps:
        1. Seed a polygon and move it east on the following day.
        2. Check that its permanent identity and birth address stay fixed while today's component identity changes.

    Examples:
        Run ``pytest tests/test_wave_tracking.py -k continuation_retains_wave_id_and_replaces_component_id``.
    """
    first = _component(_DAY_1, 1, box(0, 0, 10, 1))
    previous = _seed(_DAY_1, first).current
    current = _component(_DAY_2, 1, box(0.25, 0, 10.25, 1))

    update = track_reuters_waves(
        date=_DAY_2, series=_SERIES, components=(current,), previous=previous
    )

    assert update.current[0].wave_id == previous[0].wave_id
    assert update.current[0].component_id != previous[0].component_id
    assert update.current[0].birth_cell_id == previous[0].birth_cell_id
    assert (
        update.current[0].birth_cell_id
        != FootprintIdentity(_DAY_2, current.geometry.full_geometry).cell_id
    )
    assert update.current[0].continues is True
    assert update.lineage == ()


@pytest.mark.unit
def test_split_retains_winner_and_emits_one_explicit_edge() -> None:
    """Check that the winning child retains its parent identity and the other child receives one split relationship.

    Returns:
        None.

    Steps:
        1. Track a single parent polygon followed by two separated children.
        2. Check that the winning child retains its parent identity and the other child receives one split relationship.

    Examples:
        Run ``pytest tests/test_wave_tracking.py -k split_retains_winner_and_emits_one_explicit_edge``.
    """
    previous = _seed(_DAY_1, _component(_DAY_1, 1, box(0, 0, 10, 1))).current
    large = _component(_DAY_2, 1, box(0, 0, 6, 1))
    small = _component(_DAY_2, 2, box(6, 0, 10, 1))

    update = track_reuters_waves(
        date=_DAY_2, series=_SERIES, components=(small, large), previous=previous
    )
    winner = next(record for record in update.current if record.continues)
    child = next(record for record in update.current if not record.continues)

    assert winner.wave_id == previous[0].wave_id
    assert winner.birth_cell_id == previous[0].birth_cell_id
    assert (
        child.birth_cell_id
        == FootprintIdentity(_DAY_2, small.geometry.full_geometry).cell_id
    )
    assert child.parent_wave_ids == (previous[0].wave_id,)
    assert [
        (edge.source_wave_id, edge.target_wave_id, edge.relationship)
        for edge in update.lineage
    ] == [(previous[0].wave_id, child.wave_id, "split")]


@pytest.mark.unit
def test_merge_retains_winner_and_emits_loser_edge() -> None:
    """Check that the winning wave retains identity while the losing wave ends with a merge relationship.

    Returns:
        None.

    Steps:
        1. Seed two separate polygons and track their combined footprint.
        2. Check that the winning wave retains identity while the losing wave ends with a merge relationship.

    Examples:
        Run ``pytest tests/test_wave_tracking.py -k merge_retains_winner_and_emits_loser_edge``.
    """
    first = _component(_DAY_1, 1, box(0, 0, 6, 1))
    second = _component(_DAY_1, 2, box(6, 0, 10, 1))
    previous = _seed(_DAY_1, first, second).current
    update = track_reuters_waves(
        date=_DAY_2,
        series=_SERIES,
        components=(_component(_DAY_2, 1, box(0, 0, 10, 1)),),
        previous=previous,
    )

    winner = update.current[0]
    predecessor = next(
        record for record in previous if record.wave_id == winner.wave_id
    )
    loser = next(record for record in previous if record.wave_id != winner.wave_id)
    assert winner.parent_wave_ids == (loser.wave_id,)
    assert winner.birth_cell_id == predecessor.birth_cell_id
    assert update.ended[0].final_record.birth_cell_id == loser.birth_cell_id
    assert winner.parent_keys == tuple(sorted((winner.wave_id, loser.wave_id)))
    assert update.ended[0].merged_into_wave_id == winner.wave_id
    assert [
        (edge.source_wave_id, edge.target_wave_id, edge.relationship)
        for edge in update.lineage
    ] == [(loser.wave_id, winner.wave_id, "merge")]


@pytest.mark.unit
@pytest.mark.parametrize("cell_id", [-1, 1440 * 721, True, 519120])
def test_previous_birth_cell_must_reproduce_its_permanent_key(cell_id) -> None:
    """Corrupted birth state stops before any continuation decisions.

    Args:
        cell_id: Missing, invalid or inconsistent saved birth address.

    Returns:
        None after the damaged predecessor is rejected.

    Examples:
        Run ``pytest tests/test_wave_tracking.py -k previous_birth_cell``.
    """
    previous = _seed(_DAY_1, _component(_DAY_1, 1, box(0, 0, 10, 1))).current[0]
    with pytest.raises(WaveTrackingError, match="birth_cell_id"):
        track_reuters_waves(
            date=_DAY_2,
            series=_SERIES,
            components=(_component(_DAY_2, 1, box(0, 0, 10, 1)),),
            previous=(replace(previous, birth_cell_id=cell_id),),
        )


@pytest.mark.unit
def test_output_birth_cell_must_reproduce_its_permanent_key() -> None:
    """An invalid birth address cannot escape the daily output check.

    Args:
        None.

    Returns:
        None after the damaged current record is rejected.

    Examples:
        Run ``pytest tests/test_wave_tracking.py -k output_birth_cell``.
    """
    record = _seed(_DAY_1, _component(_DAY_1, 1, box(0, 0, 10, 1))).current[0]
    with pytest.raises(WaveTrackingError, match="birth_cell_id"):
        _validate_output((), (replace(record, birth_cell_id=519120),), (), ())


@pytest.mark.unit
def test_split_rejoin_has_no_self_edges() -> None:
    """Check that the rejoined lineage contains a merge and no self-referential edges.

    Returns:
        None.

    Steps:
        1. Track a polygon through splitting and rejoining on consecutive dates.
        2. Check that the rejoined lineage contains a merge and no self-referential edges.

    Examples:
        Run ``pytest tests/test_wave_tracking.py -k split_rejoin_has_no_self_edges``.
    """
    previous = _seed(_DAY_1, _component(_DAY_1, 1, box(0, 0, 10, 1))).current
    split = track_reuters_waves(
        date=_DAY_2,
        series=_SERIES,
        components=(
            _component(_DAY_2, 1, box(0, 0, 6, 1)),
            _component(_DAY_2, 2, box(6, 0, 10, 1)),
        ),
        previous=previous,
    )
    rejoined = track_reuters_waves(
        date=_DAY_3,
        series=_SERIES,
        components=(_component(_DAY_3, 1, box(0, 0, 10, 1)),),
        previous=split.current,
    )

    assert all(
        edge.source_wave_id != edge.target_wave_id
        for edge in (*split.lineage, *rejoined.lineage)
    )
    assert {edge.relationship for edge in rejoined.lineage} == {"merge"}


@pytest.mark.unit
def test_invalid_component_identity_fails_closed() -> None:
    """Check that seeding rejects the mismatched identity.

    Returns:
        None.

    Steps:
        1. Replace the component identifier with a hash that does not match its geometry.
        2. Check that seeding rejects the mismatched identity.

    Examples:
        Run ``pytest tests/test_wave_tracking.py -k invalid_component_identity_fails_closed``.
    """
    component = _component(_DAY_1, 1, box(0, 0, 5, 1))
    with pytest.raises(WaveTrackingError, match="must match"):
        _seed(_DAY_1, replace(component, component_id="0" * 32))


@pytest.mark.unit
def test_supplied_footprints_retain_identity_and_area_validation() -> None:
    """Validate supplied footprints without reconstructed cell membership.

    Args:
        None.

    Returns:
        None. Bad IDs, inconsistent areas and overlapping shapes still fail.

    Examples:
        pytest tests/test_wave_tracking.py -k supplied_footprints

    Steps:
        1. Validate supplied footprints without reconstructed cell membership.
        2. Check that bad IDs, inconsistent areas and overlapping shapes still fail.
    """
    component = _component(_DAY_1, 1, box(0, 0, 5, 1))
    assert component == replace(component)
    assert component == _component(_DAY_1, 1, box(0, 0, 5, 1))
    with pytest.raises(WaveTrackingError, match="must match"):
        _seed(_DAY_1, replace(component, component_id="0" * 32))
    with pytest.raises(WaveTrackingError, match="area_qualified must match"):
        _seed(
            _DAY_1,
            replace(
                component, geometry=replace(component.geometry, area_qualified=1.0)
            ),
        )
    overlapping = _component(_DAY_1, 2, box(4, 0, 7, 1))
    with pytest.raises(WaveTrackingError, match="must not overlap"):
        _seed(_DAY_1, component, overlapping)


@pytest.mark.unit
def test_duplicate_component_id_fails_closed() -> None:
    """Check that the tracker rejects the duplicate before producing new state.

    Returns:
        None.

    Steps:
        1. Supply two components with the same daily identifier.
        2. Check that the tracker rejects the duplicate before producing new state.

    Examples:
        Run ``pytest tests/test_wave_tracking.py -k duplicate_component_id_fails_closed``.
    """
    component = _component(_DAY_1, 1, box(0, 0, 5, 1))
    duplicate_geometry = replace(
        component.geometry,
        label=2,
        full_geometry=box(6, 0, 10, 1),
        geometry=box(6, 0, 10, 1),
    )
    duplicate = ComponentMembership(
        component.component_id, duplicate_geometry, component.order_digest
    )

    with pytest.raises(WaveTrackingError, match="component IDs"):
        _seed(_DAY_1, component, duplicate)


@pytest.mark.unit
def test_invalid_previous_parent_ids_fail_closed() -> None:
    """Check that the next tracking date rejects the malformed prior state.

    Returns:
        None.

    Steps:
        1. Alter the parent identifiers in a previously tracked record.
        2. Check that the next tracking date rejects the malformed prior state.

    Examples:
        Run ``pytest tests/test_wave_tracking.py -k invalid_previous_parent_ids_fail_closed``.
    """
    previous = _seed(_DAY_1, _component(_DAY_1, 1, box(0, 0, 5, 1))).current[0]
    with pytest.raises(WaveTrackingError, match="parent_wave_ids"):
        track_reuters_waves(
            date=_DAY_2,
            series=_SERIES,
            components=(_component(_DAY_2, 1, box(0, 0, 5, 1)),),
            previous=(replace(previous, parent_wave_ids=(previous.wave_id,)),),
        )


@pytest.mark.unit
def test_quiet_period_requires_no_silent_identity_reset() -> None:
    """Check that subsequent tracking obeys the explicit previous-state requirement.

    Returns:
        None.

    Steps:
        1. Track the existing wave through the supplied quiet-date scenario.
        2. Check that subsequent tracking obeys the explicit previous-state requirement.

    Examples:
        Run ``pytest tests/test_wave_tracking.py -k quiet_period_requires_no_silent_identity_reset``.
    """
    previous = _seed(_DAY_1, _component(_DAY_1, 1, box(0, 0, 5, 1))).current
    update = track_reuters_waves(
        date=_DAY_2, series=_SERIES, components=(), previous=previous
    )

    assert update.current == ()
    assert update.ended[0].end_reason == "died"


@pytest.mark.unit
def test_full_cover_measured_just_over_one_hundred_reads_as_one_hundred() -> None:
    """A cover reading inside the role-area agreement snaps to exactly 100.0.

    The divisor is the sum of three separately measured role areas, so a fully
    covered component can measure a shade larger than the areas stated for it.

    Returns:
        None.

    Steps:
        1. A cover reading inside the role-area agreement snaps to exactly 100.0.

    Examples:
        Run ``pytest tests/test_wave_tracking.py -k full_cover_measured_just_over_one_hundred_reads_as_one_hundred``.
    """
    covered = _component(_DAY_1, 1, box(0, 0, 10, 1))
    tracked = _seed(_DAY_1, covered).current[0]
    stated = _full_area(covered.geometry)

    reading = _overlap_percent(
        _Overlap(predecessor_index=0, area=stated * (1 + 5e-7)), (tracked,)
    )

    assert reading == 100.0


@pytest.mark.unit
def test_overlap_larger_than_its_component_is_rejected_immediately() -> None:
    """An intersection bigger than the shape it was cut from cannot be stored.

    Returns:
        None.

    Steps:
        1. An intersection bigger than the shape it was cut from cannot be stored.

    Examples:
        Run ``pytest tests/test_wave_tracking.py -k overlap_larger_than_its_component_is_rejected_immediately``.
    """
    covered = _component(_DAY_1, 1, box(0, 0, 10, 1))
    tracked = _seed(_DAY_1, covered).current[0]
    stated = _full_area(covered.geometry)

    with pytest.raises(WaveTrackingError, match="finite share"):
        _overlap_percent(_Overlap(predecessor_index=0, area=stated * 2), (tracked,))


@pytest.mark.unit
def test_out_of_range_overlap_fails_on_the_date_it_is_produced() -> None:
    """A bad overlap is caught the same day, so it never reaches a checkpoint.

    Returns:
        None.

    Steps:
        1. A bad overlap is caught the same day, so it never reaches a checkpoint.

    Examples:
        Run ``pytest tests/test_wave_tracking.py -k out_of_range_overlap_fails_on_the_date_it_is_produced``.
    """
    tracked = _seed(_DAY_1, _component(_DAY_1, 1, box(0, 0, 10, 1))).current[0]
    damaged = replace(tracked, overlap_percent=101.0, continues=True)

    with pytest.raises(WaveTrackingError, match="current overlap_percent"):
        _validate_output((), (damaged,), (), ())


@pytest.mark.unit
def test_rejected_preceding_state_names_the_wave_it_came_from() -> None:
    """Replayed state that fails a rule reports which wave carried it.

    Returns:
        None.

    Steps:
        1. Replayed state that fails a rule reports which wave carried it.

    Examples:
        Run ``pytest tests/test_wave_tracking.py -k rejected_preceding_state_names_the_wave_it_came_from``.
    """
    tracked = _seed(_DAY_1, _component(_DAY_1, 1, box(0, 0, 10, 1))).current[0]
    damaged = replace(tracked, overlap_percent=101.0)

    with pytest.raises(WaveTrackingError) as failure:
        track_reuters_waves(
            date=_DAY_2,
            series=_SERIES,
            components=(_component(_DAY_2, 1, box(0, 0, 10, 1)),),
            previous=(damaged,),
        )

    message = str(failure.value)
    assert "previous overlap_percent must be between 0 and 100" in message
    assert f"for wave {tracked.wave_id}" in message


def _reference_extents(components):
    """Prepare complete role extents independently with GeoPandas.

    Args:
        components: Daily geometries in their original tracking order.

    Returns:
        An EPSG:6933 GeoSeries after sequential role union and polygon repair.

    Examples:
        extents = _reference_extents([component.geometry])

    Steps:
        1. Prepare complete role extents independently with GeoPandas.
    """
    rows = gpd.GeoDataFrame(
        {
            "geometry": [item.geometry for item in components],
            "grace_geometry": [item.grace_geometry for item in components],
            "buffer_geometry": [item.full_geometry for item in components],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    extents = []
    for row in rows.itertuples(index=False):
        present = [
            item
            for item in (row.geometry, row.grace_geometry, row.buffer_geometry)
            if item is not None and not item.is_empty
        ]
        extent = present[0]
        for role in present[1:]:
            extent = extent.union(role)
        extents.append(extent)
    return gpd.GeoSeries(extents, index=rows.index, crs=rows.crs).to_crs(6933).buffer(0)


def _exhaustive_overlaps(components, previous):
    """Measure all pairs using complete projected role extents.

    Args:
        components: Current daily test components in canonical order.
        previous: Preceding tracked test waves in canonical order.

    Returns:
        Exact positive overlaps grouped by current component index.

    Examples:
        ``_exhaustive_overlaps((component,), previous)`` measures the one
        current component against every predecessor.

    Steps:
        1. Measure all pairs using complete projected role extents.
    """
    current_extents = _reference_extents([item.geometry for item in components])
    previous_extents = _reference_extents(
        [item.component.geometry for item in previous]
    )
    return {
        current_index: tuple(
            _Overlap(predecessor_index, area)
            for predecessor_index, predecessor in enumerate(previous_extents)
            if (area := extent.intersection(predecessor).buffer(0).area / 1_000_000)
            > 0.0
        )
        for current_index, extent in enumerate(current_extents)
    }


def _track_exhaustively(*, components, previous, date=_DAY_2):
    """Run the tracker with its original exhaustive current/predecessor pairs.

    Args:
        components: Components to track on the selected test date.
        previous: Active components from the preceding test date.
        date: Date represented by the component batch.

    Returns:
        The public tracking update produced with all-pairs overlaps.

    Examples:
        ``_track_exhaustively(components=components, previous=previous)``
        provides the reference update for the indexed tracker.

    Steps:
        1. Run the tracker with its original exhaustive current/predecessor pairs.
    """
    with patch.object(wave_tracking, "_overlaps", _exhaustive_overlaps):
        return track_reuters_waves(
            date=date,
            series=_SERIES,
            components=components,
            previous=previous,
        )


def _components_for_geometries(day, geometries):
    """Create deterministic valid test components from full geometries.

    Args:
        day: Shared date for every generated component.
        geometries: Full geometries assigned sequential positive labels.

    Returns:
        Valid supplied daily footprint components.

    Examples:
        ``_components_for_geometries(_DAY_1, (box(0, 0, 1, 1),))`` returns
        one component for a seed update.

    Steps:
        1. Create deterministic valid test components from full geometries.
    """
    return tuple(
        _component(day, index, geometry)
        for index, geometry in enumerate(geometries, start=1)
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("previous_geometries", "current_geometries"),
    [
        pytest.param(
            (
                Polygon(((0, 0), (4, 0), (0, 4), (0, 0))),
                Polygon(((4, 0), (4, 4), (0, 4), (4, 0))),
            ),
            (box(0, 0, 4, 4),),
            id="duplicate-coverage-envelopes",
        ),
        pytest.param(
            (MultiPolygon((box(175, -5, 180, 5), box(-180, -5, -175, 5))),),
            (MultiPolygon((box(175, -5, 180, 5), box(-180, -5, -175, 5))),),
            id="negative-positive-dateline",
        ),
        pytest.param(
            (
                Polygon(
                    ((0, 0), (8, 0), (8, 8), (0, 8), (0, 0)),
                    (((2, 2), (6, 2), (6, 6), (2, 6), (2, 2)),),
                ),
            ),
            (
                Polygon(
                    ((0, 0), (8, 0), (8, 8), (0, 8), (0, 0)),
                    (((2, 2), (6, 2), (6, 6), (2, 6), (2, 2)),),
                ),
            ),
            id="holes",
        ),
        pytest.param(
            (MultiPolygon((box(0, 0, 2, 2), box(6, 0, 8, 2))),),
            (MultiPolygon((box(0, 0, 2, 2), box(6, 0, 8, 2))),),
            id="multipart",
        ),
        pytest.param(
            (box(-180, -45, 180, 45),),
            (box(-180, -45, 180, 45),),
            id="world-spanning",
        ),
        pytest.param(
            (box(0, 0, 2, 2),),
            (box(2, 0, 4, 2),),
            id="boundary-touch-only",
        ),
    ],
)
def test_spatial_overlap_index_matches_original_exhaustive_path(
    previous_geometries, current_geometries
) -> None:
    """Keep edge geometry outputs identical to all-pairs tracking.

    Args:
        previous_geometries: Parameterized predecessor geometries.
        current_geometries: Parameterized current-day geometries.

    Returns:
        None. Exact areas and all output records match the reference path.

    Examples:
        The dateline parameter verifies a wrapped multipart component against
        the independently implemented exhaustive overlap path.

    Steps:
        1. Keep edge geometry outputs identical to all-pairs tracking.
        2. Check that exact areas and all output records match the reference path.
    """
    previous_components = _components_for_geometries(_DAY_1, previous_geometries)
    previous = _seed(_DAY_1, *previous_components).current
    components = _components_for_geometries(_DAY_2, current_geometries)

    assert _overlaps(components, previous) == _exhaustive_overlaps(components, previous)
    assert track_reuters_waves(
        date=_DAY_2,
        series=_SERIES,
        components=components,
        previous=previous,
    ) == _track_exhaustively(components=components, previous=previous)


@pytest.mark.unit
def test_spatial_overlap_index_matches_original_split_merge_tie_records() -> None:
    """Keep split/merge tie records identical to all-pairs tracking.

    Args:
        None.

    Returns:
        None. Tie decisions, updates, and lineage records match the reference.

    Examples:
        Equal halves of one predecessor verify deterministic split selection;
        equal predecessors joining one current component verify merge selection.

    Steps:
        1. Keep split/merge tie records identical to all-pairs tracking.
        2. Check that tie decisions, updates, and lineage records match the reference.
    """
    split_previous = _seed(_DAY_1, _component(_DAY_1, 1, box(0, 0, 8, 2))).current
    split_components = _components_for_geometries(
        _DAY_2, (box(0, 0, 4, 2), box(4, 0, 8, 2))
    )
    merge_previous = _seed(
        _DAY_1,
        _component(_DAY_1, 1, box(0, 0, 4, 2)),
        _component(_DAY_1, 2, box(4, 0, 8, 2)),
    ).current
    merge_components = _components_for_geometries(_DAY_2, (box(0, 0, 8, 2),))

    for components, previous in (
        (split_components, split_previous),
        (merge_components, merge_previous),
    ):
        assert _overlaps(components, previous) == _exhaustive_overlaps(
            components, previous
        )
        assert track_reuters_waves(
            date=_DAY_2,
            series=_SERIES,
            components=components,
            previous=previous,
        ) == _track_exhaustively(components=components, previous=previous)


@pytest.mark.unit
@pytest.mark.parametrize("right_edge", [5.999999999999, 6.000000000001])
def test_spatial_overlap_index_matches_exhaustive_near_sixty_percent(
    right_edge,
) -> None:
    """Keep either side of 60-percent coverage identical to all-pairs.

    Args:
        right_edge: Parameterized current-component boundary near 60 percent.

    Returns:
        None. Exact overlap and resulting tracking update match the reference.

    Examples:
        A right edge of ``5.999999999999`` compares the local just-below-60
        result without changing numerical threshold behavior.

    Steps:
        1. Keep either side of 60-percent coverage identical to all-pairs.
        2. Check that exact overlap and resulting tracking update match the reference.
    """
    previous = _seed(_DAY_1, _component(_DAY_1, 1, box(0, 0, 10, 1))).current
    components = (_component(_DAY_2, 1, box(0, 0, right_edge, 1)),)
    indexed = _overlaps(components, previous)
    exhaustive = _exhaustive_overlaps(components, previous)

    assert indexed == exhaustive
    assert abs(_overlap_percent(indexed[0][0], previous) - 60.0) < 1e-9
    assert track_reuters_waves(
        date=_DAY_2,
        series=_SERIES,
        components=components,
        previous=previous,
    ) == _track_exhaustively(components=components, previous=previous)


def _generated_scene_components(day, *, x_offset=0.0):
    """Generate a deterministic separated scene without extra libraries.

    Args:
        day: Shared date for every generated component.
        x_offset: Uniform longitude shift applied to all component boxes.

    Returns:
        Eighteen non-overlapping components in a reproducible arrangement.

    Examples:
        ``_generated_scene_components(_DAY_2, x_offset=0.5)`` produces the
        successor scene used by the equivalence test.

    Steps:
        1. Generate a deterministic separated scene without extra libraries.
    """
    return tuple(
        _component(
            day,
            index + 1,
            box(
                (index % 6) * 20 + x_offset,
                (index // 6) * 12,
                (index % 6) * 20 + 3 + x_offset,
                (index // 6) * 12 + 3,
            ),
        )
        for index in range(18)
    )


@pytest.mark.unit
def test_spatial_overlap_index_matches_generated_exhaustive_scene() -> None:
    """Keep a generated-scene output equivalent to all-pairs tracking.

    Args:
        None.

    Returns:
        None. Exact overlaps and the full update equal the reference results.

    Examples:
        Eighteen shifted, separated boxes exercise many index queries in one
        deterministic scene.

    Steps:
        1. Keep a generated-scene output equivalent to all-pairs tracking.
        2. Check that exact overlaps and the full update equal the reference results.
    """
    previous = _seed(_DAY_1, *_generated_scene_components(_DAY_1)).current
    components = _generated_scene_components(_DAY_2, x_offset=0.5)

    assert _overlaps(components, previous) == _exhaustive_overlaps(components, previous)
    assert track_reuters_waves(
        date=_DAY_2,
        series=_SERIES,
        components=components,
        previous=previous,
    ) == _track_exhaustively(components=components, previous=previous)


@pytest.mark.unit
def test_spatial_overlap_index_handles_empty_current_or_predecessors() -> None:
    """Keep empty current or predecessor maps equivalent to all-pairs.

    Args:
        None.

    Returns:
        None. Empty overlap maps and the all-ended update equal the reference.

    Examples:
        A current empty sequence ends a preceding wave without an index query.

    Steps:
        1. Keep empty current or predecessor maps equivalent to all-pairs.
        2. Check that empty overlap maps and the all-ended update equal the reference.
    """
    component = _component(_DAY_2, 1, box(0, 0, 2, 2))
    previous = _seed(_DAY_1, _component(_DAY_1, 1, box(0, 0, 2, 2))).current

    assert _overlaps((component,), ()) == _exhaustive_overlaps((component,), ())
    assert _overlaps((), previous) == _exhaustive_overlaps((), previous)
    assert track_reuters_waves(
        date=_DAY_2,
        series=_SERIES,
        components=(),
        previous=previous,
    ) == _track_exhaustively(components=(), previous=previous)


@pytest.mark.unit
def test_spatial_index_falls_back_for_an_unindexable_envelope() -> None:
    """Use every candidate when one supplied envelope is unsafe.

    Args:
        None.

    Returns:
        None. An empty geometry prevents index construction and returns every
        candidate position for exhaustive processing.

    Examples:
        An empty ``GeometryCollection`` beside a polygon yields indices
        ``(0, 1)`` even for a disjoint query polygon.

    Steps:
        1. Use every candidate when one supplied envelope is unsafe.
        2. Check that an empty geometry prevents index construction and returns every candidate position for exhaustive processing.
    """
    predecessors = (GeometryCollection(), box(4, 0, 6, 2))
    spatial_index = _spatial_index(predecessors)

    assert spatial_index is None
    assert _candidate_geometry_indices(
        box(0, 0, 2, 2), predecessors, spatial_index
    ) == (0, 1)


@pytest.mark.unit
def test_same_day_spatial_index_accepts_boundary_touching_components() -> None:
    """Allow same-day full geometries that meet only at a boundary.

    Args:
        None.

    Returns:
        None. Two seeded components remain valid and active.

    Examples:
        Boxes ending and beginning at longitude 2 have zero exact overlap.

    Steps:
        1. Allow same-day full geometries that meet only at a boundary.
        2. Check that two seeded components remain valid and active.
    """
    update = _seed(
        _DAY_1,
        _component(_DAY_1, 1, box(0, 0, 2, 2)),
        _component(_DAY_1, 2, box(2, 0, 4, 2)),
    )

    assert len(update.current) == 2


@pytest.mark.unit
def test_same_day_spatial_index_rejects_positive_area_overlap() -> None:
    """Reject same-day full geometries with positive exact overlap.

    Args:
        None.

    Returns:
        None. The expected ``WaveTrackingError`` is raised.

    Examples:
        Boxes spanning longitudes 0..2 and 1..3 share positive area and fail.

    Steps:
        1. Reject same-day full geometries with positive exact overlap.
        2. Check that the expected ``WaveTrackingError`` is raised.
    """
    with pytest.raises(
        WaveTrackingError, match="daily component full geometries must not overlap"
    ):
        _seed(
            _DAY_1,
            _component(_DAY_1, 1, box(0, 0, 2, 2)),
            _component(_DAY_1, 2, box(1, 0, 3, 2)),
        )


@pytest.mark.unit
def test_spatial_overlap_index_reduces_intersection_calls_for_separated_components() -> (
    None
):
    """Shortlist cross-day separated components before exact intersections.

    Args:
        None.

    Returns:
        None. The indexed call count equals one per current component and is
        less than the exhaustive all-pairs count.

    Examples:
        Eighteen shifted boxes make 18 indexed calls instead of 324 calls.

    Steps:
        1. Shortlist cross-day separated components before exact intersections.
        2. Check that the indexed call count equals one per current component and is less than the exhaustive all-pairs count.
    """
    previous = _seed(_DAY_1, *_generated_scene_components(_DAY_1)).current
    components = _generated_scene_components(_DAY_2, x_offset=0.5)
    original_intersection = shapely.intersection

    with patch("shapely.intersection", wraps=original_intersection) as indexed:
        indexed_overlaps = _overlaps(components, previous)
    with patch("shapely.intersection", wraps=original_intersection) as exhaustive:
        exhaustive_overlaps = _exhaustive_overlaps(components, previous)

    assert indexed_overlaps == exhaustive_overlaps
    assert indexed.call_count == len(components)
    assert exhaustive.call_count == len(components) * len(previous)


@pytest.mark.unit
def test_same_day_spatial_index_reduces_intersection_calls_for_separated_components() -> (
    None
):
    """Shortlist same-day separated components before exact validation.

    Args:
        None.

    Returns:
        None. The indexed same-day validation avoids exact intersections while
        the forced exhaustive path evaluates every unique pair.

    Examples:
        Eighteen separated boxes make zero indexed calls instead of 153 exact
        all-pairs calls.

    Steps:
        1. Shortlist same-day separated components before exact validation.
        2. Check that the indexed same-day validation avoids exact intersections while the forced exhaustive path evaluates every unique pair.
    """
    components = _generated_scene_components(_DAY_1)
    original_intersection = shapely.intersection

    with patch("shapely.intersection", wraps=original_intersection) as indexed:
        _seed(_DAY_1, *components)
    with (
        patch.object(
            wave_tracking,
            "_candidate_geometry_indices",
            return_value=tuple(range(len(components))),
        ),
        patch("shapely.intersection", wraps=original_intersection) as exhaustive,
    ):
        _seed(_DAY_1, *components)

    assert indexed.call_count == 0
    assert exhaustive.call_count == len(components) * (len(components) - 1) // 2


@pytest.mark.unit
def test_merge_tie_uses_previous_daily_footprint_digest() -> None:
    """Break an equal-overlap merge using ordering separate from permanent IDs.

    Args:
        None.

    Returns:
        None. The survivor matches the final tie rule.

    Examples:
        ``pytest tests/test_wave_tracking.py -k merge_tie_uses``

    Steps:
        1. Continue two equal footprints with daily ordering opposite their cell-ID order.
        2. Merge them and check that the daily footprint digest selects the survivor.
    """
    left = _component(_DAY_1, 1, box(0, 0, 4, 1))
    right = _component(_DAY_1, 2, box(4, 0, 8, 1))
    previous = _seed(_DAY_1, left, right).current
    lower_id, higher_id = sorted(
        (
            _component(_DAY_2, 1, box(0, 0, 4, 1)),
            _component(_DAY_2, 2, box(4, 0, 8, 1)),
        ),
        key=lambda component: component.component_id,
    )
    previous = track_reuters_waves(
        date=_DAY_2,
        series=_SERIES,
        components=(
            replace(lower_id, order_digest="f" * 32),
            replace(higher_id, order_digest="0" * 32),
        ),
        previous=previous,
    ).current
    ordered = sorted(previous, key=lambda record: record.component.order_digest)
    assert ordered[0].component_id == higher_id.component_id
    merged = _component(_DAY_3, 1, box(0, 0, 8, 1))
    update = track_reuters_waves(
        date=_DAY_3, series=_SERIES, components=(merged,), previous=previous
    )
    assert update.current[0].wave_id == ordered[0].wave_id
    assert update.current[0].birth_order_digest == ordered[0].birth_order_digest


@pytest.mark.unit
def test_duplicate_full_daily_ordering_digests_are_rejected() -> None:
    """Reject distinct footprints claiming an identical complete daily order.

    Args:
        None.

    Returns:
        None. A full digest collision cannot depend on supplied row order.

    Examples:
        pytest tests/test_wave_tracking.py -k duplicate_full_daily_ordering
    """
    first = _component(_DAY_1, 1, box(0, 0, 4, 1), order_digest="a" * 32)
    second = _component(_DAY_1, 2, box(4, 0, 8, 1), order_digest="a" * 32)
    with pytest.raises(WaveTrackingError, match="ordering digests must be unique"):
        _seed(_DAY_1, first, second)


@pytest.mark.unit
@pytest.mark.parametrize("digest", [None, "abc", "A" * 32, 7])
def test_daily_ordering_requires_a_complete_lowercase_digest(digest) -> None:
    """Reject malformed supplied ordering fields before using them to sort.

    Args:
        digest: Missing, short, uppercase or nontext ordering field.

    Returns:
        None. The tracker reports the field rather than computing a replacement.

    Examples:
        pytest tests/test_wave_tracking.py -k daily_ordering_requires
    """
    component = _component(_DAY_1, 1, box(0, 0, 4, 1))
    with pytest.raises(WaveTrackingError, match="order_digest"):
        _seed(_DAY_1, replace(component, order_digest=digest))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value"), [("birth_order_digest", "e" * 32), ("name", "Changed 2024")]
)
def test_continued_output_cannot_change_birth_order_or_assigned_name(
    field, value
) -> None:
    """Check inheritance independently of the code that constructs a continuation.

    Args:
        field: Retained birth ordering or name field to alter.
        value: A replacement that must not overwrite preceding state.

    Returns:
        None. Output validation rejects changed continuation facts.

    Examples:
        pytest tests/test_wave_tracking.py -k continued_output_cannot_change
    """
    previous = _seed(_DAY_1, _component(_DAY_1, 1, box(0, 0, 4, 1))).current
    update = track_reuters_waves(
        date=_DAY_2,
        series=_SERIES,
        components=(_component(_DAY_2, 1, box(0, 0, 4, 1)),),
        previous=previous,
    )
    altered = replace(update.current[0], **{field: value})
    with pytest.raises(WaveTrackingError, match="retain their birth ordering digest"):
        _validate_output(previous, (altered,), update.ended, update.lineage)


@pytest.mark.unit
def test_equal_disjoint_half_parents_do_not_create_split_lineage() -> None:
    """Reject tied split parents when each covers only half of the new extent.

    Args:
        None.

    Returns:
        None. No eligible parent exists and both absorbed predecessors merge.

    Examples:
        ``pytest tests/test_wave_tracking.py -k equal_disjoint_half``

    Steps:
        1. Reject tied split parents when each covers only half of the new extent.
        2. Check that no eligible parent exists and both absorbed predecessors merge.
    """
    previous = _seed(
        _DAY_1,
        _component(_DAY_1, 1, box(0, 0, 5, 1)),
        _component(_DAY_1, 2, box(5, 0, 10, 1)),
    ).current
    current = _component(_DAY_2, 1, box(0, 0, 10, 1))
    matches = _overlaps((current,), previous)[0]
    assert len(matches) == 2
    assert matches[0].area == pytest.approx(matches[1].area)
    assert wave_tracking._best_parent(matches, current, previous) is None
    update = track_reuters_waves(
        date=_DAY_2, series=_SERIES, components=(current,), previous=previous
    )
    assert update.current[0].continues is True
    assert all(edge.relationship == "merge" for edge in update.lineage)
    assert len(update.lineage) == 1


@pytest.mark.unit
def test_exact_sixty_percent_ratio_continues_despite_display_rounding(monkeypatch):
    """Use the direct quotient at a grid-aligned sixty-percent boundary.

    Args:
        monkeypatch: Fixture controlling the display-only percentage rounding.

    Returns:
        None. The same physical wave survives with its unrounded display percent.

    Examples:
        ``pytest tests/test_wave_tracking.py -k display_rounding``

    Steps:
        1. Use the direct quotient at a grid-aligned sixty-percent boundary.
        2. Check that the same physical wave survives with its unrounded display percent.
    """
    previous = _seed(_DAY_1, _component(_DAY_1, 1, box(-10, -80, 0, -79.75))).current
    current = _component(_DAY_2, 1, box(-10, -80, -4, -79.75))
    overlap = _overlaps((current,), previous)[0][0]
    ratio = overlap.area / wave_tracking._full_area(previous[0].component.geometry)
    assert ratio == pytest.approx(0.6, rel=0, abs=1e-14)
    assert ratio >= 0.6
    assert _overlap_percent(overlap, previous) == pytest.approx(60, rel=0, abs=1e-12)
    # Projection rounding varies by platform. Force the display-only rounding
    # case so comparing that percentage still fails this regression everywhere.
    monkeypatch.setattr(
        wave_tracking, "_overlap_percent", lambda *_: float(np.nextafter(60.0, 0.0))
    )
    update = track_reuters_waves(
        date=_DAY_2, series=_SERIES, components=(current,), previous=previous
    )
    assert update.current[0].wave_id == previous[0].wave_id
    assert update.current[0].continues
    assert not update.ended
    again = track_reuters_waves(
        date=_DAY_3,
        series=_SERIES,
        components=(_component(_DAY_3, 1, current.geometry.full_geometry),),
        previous=update.current,
    )
    assert again.current[0].wave_id == previous[0].wave_id


@pytest.mark.unit
@pytest.mark.parametrize(
    "key",
    [
        "20250101-00000000000000000000000012345678",
        "20241399-00000000000000000000000012345678",
    ],
)
def test_previous_key_requires_actual_birth_date(key):
    """Reject impossible prior permanent keys before overlap or naming decisions.

    Args:
        key: A wrong or impossible birth-date prefix.

    Returns:
        None. Prior state is rejected against its actual start date.

    Examples:
        ``pytest tests/test_wave_tracking.py -k actual_birth_date``

    Steps:
        1. Reject impossible prior permanent keys before overlap or naming decisions.
        2. Check that prior state is rejected against its actual start date.
    """
    previous = _seed(_DAY_1, _component(_DAY_1, 1, box(0, 0, 1, 1))).current
    with pytest.raises(WaveTrackingError, match="birth-date prefix"):
        track_reuters_waves(
            date=_DAY_2,
            series=_SERIES,
            components=(_component(_DAY_2, 1, box(0, 0, 1, 1)),),
            previous=(replace(previous[0], wave_id=key),),
        )


@pytest.mark.unit
@pytest.mark.parametrize("roles", ["qualified_and_grace", "grace_only", "full_only"])
def test_extent_projection_preserves_role_union_order(roles: str) -> None:
    """Retain the role union even when the full extent already contains each role.

    Args:
        roles: Optional role combination represented by the supplied footprint.

    Returns:
        None. The tracking extent equals the independent GeoPandas reference,
        including its coordinate ordering after projection and repair.

    Examples:
        Run ``pytest tests/test_wave_tracking.py -k role_union_order``.

    Steps:
        1. Retain the role union even when the full extent already contains each role.
        2. Check that the tracking extent equals the independent GeoPandas reference, including its coordinate ordering after projection and repair.
    """
    full = box(0, 0, 4, 2)
    qualified = box(0, 0, 2, 2) if roles == "qualified_and_grace" else None
    grace = box(2, 0, 3, 2) if roles != "full_only" else None
    component = DailyWaveComponentGeometry(
        time=_DAY_1,
        label=1,
        geometry=qualified,
        grace_geometry=grace,
        full_geometry=full,
        area_qualified=0.0 if qualified is None else _area(qualified),
        area_grace=0.0 if grace is None else _area(grace),
        area_gap_filled=0.0,
        qualified_fraction=0.0,
    )

    expected = _reference_extents([component]).iloc[0]
    actual = wave_tracking._project_extents((component,))[0]

    assert actual.wkb == expected.wkb
    if roles != "full_only":
        direct = gpd.GeoSeries([full], crs=4326).to_crs(6933).buffer(0).iloc[0]
        assert actual.equals(direct)
        assert actual.wkb != direct.wkb


@pytest.mark.unit
def test_role_union_overlap_preserves_merge_history_into_following_day() -> None:
    """Use the exact prepared overlap to select merge winners and future lineage.

    Args:
        None.

    Returns:
        None. All tracking state equals the independently prepared reference,
        including the next day's ending and merge into another wave.

    Examples:
        Run ``pytest tests/test_wave_tracking.py -k role_union_overlap``.

    Steps:
        1. Use the exact prepared overlap to select merge winners and future lineage.
        2. Check that all tracking state equals the independently prepared reference, including the next day's ending and merge into another wave.
    """
    left = _component(
        _DAY_1,
        1,
        box(-80, -60, -78, -59.75),
        qualified=box(-80, -60, -79.6, -59.75),
    )
    right = _component(
        _DAY_1,
        2,
        box(-78, -60, -76, -59.75),
        qualified=box(-78, -60, -77.6, -59.75),
    )
    previous = _seed(_DAY_1, left, right).current
    merged = _component(
        _DAY_2,
        1,
        box(-80, -60, -76, -59.75),
        qualified=box(-80, -60, -79.2, -59.75),
    )
    overlaps = _exhaustive_overlaps((merged,), previous)[0]
    expected_winner = min(
        overlaps,
        key=lambda value: (
            -value.area,
            previous[value.predecessor_index].start_date,
            previous[value.predecessor_index].component_id,
        ),
    ).predecessor_index
    update = track_reuters_waves(
        date=_DAY_2, series=_SERIES, components=(merged,), previous=previous
    )
    reference = _track_exhaustively(components=(merged,), previous=previous)

    assert update == reference
    assert update.current[0].wave_id == previous[expected_winner].wave_id
    loser = previous[1 - expected_winner]
    assert update.ended[0].final_record.wave_id == loser.wave_id
    assert update.lineage[0].source_wave_id == loser.wave_id
    assert update.lineage[0].target_wave_id == previous[expected_winner].wave_id

    competitor = _seed(
        _DAY_2, _component(_DAY_2, 2, box(-90, -60, -82, -59.75))
    ).current
    following = _component(
        _DAY_3,
        1,
        box(-90, -60, -76, -59.75),
        qualified=box(-90, -60, -89, -59.75),
    )
    next_update = track_reuters_waves(
        date=_DAY_3,
        series=_SERIES,
        components=(following,),
        previous=(*update.current, *competitor),
    )
    next_reference = _track_exhaustively(
        components=(following,),
        previous=(*reference.current, *competitor),
        date=_DAY_3,
    )

    assert next_update == next_reference
    assert next_update.current[0].wave_id != update.current[0].wave_id
    assert next_update.ended[0].final_record.wave_id == update.current[0].wave_id
    assert next_update.lineage[0].source_wave_id == update.current[0].wave_id
    assert next_update.lineage[0].target_wave_id == next_update.current[0].wave_id
