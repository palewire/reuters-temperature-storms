"""Sequential wave tracking with deterministic persistent identities."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Literal, cast

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from shapely import Geometry, STRtree
from shapely.ops import transform

from .components import (
    _ROLE_AREA_ABSOLUTE_TOLERANCE_KM2,
    _ROLE_AREA_RELATIVE_TOLERANCE,
    ComponentMembership,
    DailyWaveComponentGeometry,
    _relative_difference,
)
from .identity import WAVE_ID_PATTERN, birth_wave_identity, component_identity
from .series import WaveSeries

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

_AREA_TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
_CONTINUATION_COVERAGE = 0.60
_OVERLAP_BOUNDARY_TOLERANCE = 1e-12
_COMPONENT_AREA_RELATIVE_TOLERANCE = 1e-9
_COMPONENT_AREA_ABSOLUTE_TOLERANCE_KM2 = 1e-6
# An overlap percent divides one measured intersection area by the sum of a
# component's three stored role areas.  Those role areas only have to agree with
# the component's measured full geometry within _ROLE_AREA_RELATIVE_TOLERANCE,
# so the quotient for a fully covered component can land a little outside 0-100
# without describing impossible geometry.  Anything further out is a real error.
_OVERLAP_MEASUREMENT_TOLERANCE = 100.0 * _ROLE_AREA_RELATIVE_TOLERANCE


class WaveTrackingError(ValueError):
    """Raised when a wave-tracking step cannot preserve valid lineage."""


@dataclass(frozen=True, slots=True)
class TrackedWaveComponent:
    """A daily component carrying its persistent wave identity.

    ``component_id`` identifies the footprint on this date. ``wave_id`` stays
    with the wave throughout its history and is used by catalog and lineage.
    """

    series: WaveSeries
    component: ComponentMembership
    component_id: str
    wave_id: str
    start_date: np.datetime64
    peak_qualified_area: float
    peak_date: np.datetime64
    continues: bool
    overlap_percent: float
    parent_wave_ids: tuple[str, ...]
    name: str | None = None

    @property
    def parent_keys(self) -> tuple[str, ...]:
        """Return stored parents, including a surviving merger identity.

        Steps:
            1. Start with the saved parent wave IDs.
            2. If this is a continued merger, include the surviving wave's own ID.
            3. Return the IDs in sorted order for the stored parent list.

        Args:
            None.

        Returns:
            Sorted parent keys. A continued merger also includes its survivor;
            non-self internal lineage remains in ``parent_wave_ids``.

        Examples:
            ``record.parent_keys`` includes both identities swallowed by a merger.
        """
        parents = self.parent_wave_ids
        if self.continues and parents:
            return tuple(sorted((*parents, self.wave_id)))
        return parents


@dataclass(frozen=True, slots=True)
class EndedWave:
    """An explicit end for a wave that was active on the preceding date."""

    final_record: TrackedWaveComponent
    end_date: np.datetime64
    end_reason: Literal["died", "merged"]
    merged_into_wave_id: str | None


@dataclass(frozen=True, slots=True)
class LineageEdge:
    """One split or merge relationship between full wave identities."""

    series: WaveSeries
    source_wave_id: str
    target_wave_id: str
    event_date: np.datetime64
    relationship: Literal["split", "merge"]


@dataclass(frozen=True, slots=True)
class WaveTrackingUpdate:
    """The active, ended, and lineage records produced for one date."""

    date: np.datetime64
    series: WaveSeries
    current: tuple[TrackedWaveComponent, ...]
    ended: tuple[EndedWave, ...]
    lineage: tuple[LineageEdge, ...]


@dataclass(frozen=True, slots=True)
class _Overlap:
    """One current/predecessor intersection measured in equal-area km2."""

    predecessor_index: int
    area: float


class _ComponentGeometryValidator:
    """Remember checked immutable geometry objects for one tracker branch."""

    def __init__(self) -> None:
        """Create an empty geometry-validation cache.

        Steps:
            1. Initialize an empty geometry-validation cache.

        Args:
            None.

        Returns:
            None.

        Examples:
            validator = _ComponentGeometryValidator()
        """
        self._checked: dict[int, DailyWaveComponentGeometry] = {}

    def __call__(self, component: ComponentMembership) -> None:
        """Measure a component only if its immutable geometry is new.

        Steps:
            1. Look up whether this exact, unchanging geometry object has already been checked.
            2. If it is new, check its geometry and area measurements and remember the object.

        Args:
            component: Membership carrying the geometry to check.

        Returns:
            None after successful geometry validation.

        Raises:
            WaveTrackingError: If a new geometry or its measured areas is invalid.

        Examples:
            validator(component)
        """
        geometry = component.geometry
        if self._checked.get(id(geometry)) is geometry:
            return
        _validate_component_geometry(component)
        self._checked[id(geometry)] = geometry

    def retain(self, components: Iterable[ComponentMembership]) -> None:
        """Keep only the active date's geometry objects after branch work joins.

        Steps:
            1. Keep checks only for geometry objects that are still used by the current waves.
            2. Release references to older geometry so completed dates do not accumulate in memory.

        Args:
            components: Active component memberships for the completed date.

        Returns:
            None. Older geometry references are released.

        Examples:
            validator.retain(record.component for record in active)
        """
        active = {
            id(component.geometry): component.geometry for component in components
        }
        self._checked = {
            key: geometry
            for key, geometry in self._checked.items()
            if active.get(key) is geometry
        }


def track_reuters_waves(
    *,
    date: np.datetime64,
    series: WaveSeries,
    components: Sequence[ComponentMembership],
    previous: Sequence[TrackedWaveComponent] = (),
    seed: bool = False,
) -> WaveTrackingUpdate:
    """Track one consecutive daily component set for one temperature series.

    Steps:
        1. Validate the consecutive daily inputs.
        2. Assign continuations, births, endings, names, and lineage for this series.

    Args:
        date: Current tracking date.
        series: Independent temperature and direction series.
        components: Current daily component memberships.
        previous: Active records from the preceding date.
        seed: Whether this date initializes the series.

    Returns:
        Current wave records, explicit endings and split/merge lineage.

    Examples:
        update = track_reuters_waves(
            date=day, series=series, components=components, seed=True
        )
    """
    return _track_reuters_waves(
        date=date, series=series, components=components, previous=previous, seed=seed
    )


def _track_reuters_waves(
    *,
    date: np.datetime64,
    series: WaveSeries,
    components: Sequence[ComponentMembership],
    previous: Sequence[TrackedWaveComponent] = (),
    seed: bool = False,
    geometry_validator: Callable[[ComponentMembership], None] | None = None,
) -> WaveTrackingUpdate:
    """Track one date with optional tracker-owned immutable geometry reuse.

    Steps:
        1. Check the date, series, current footprints and preceding wave records.
        2. On the first date, assign a permanent ID to each footprint.
        3. Otherwise measure overlaps and select the preceding wave and successor
           for each continuing footprint using the fixed threshold and tie rules.
        4. Keep IDs for continuations and assign new IDs to other footprints.
        5. Record splits, absorbed waves and waves that disappeared.
        6. Check the completed relationships and return records in fixed ID order.

    Args:
        date: Current tracking date.
        series: Independent temperature and direction series.
        components: Current daily component memberships.
        previous: Active records from the preceding date.
        seed: Whether this date initializes a branch.
        geometry_validator: Private branch-scoped geometry checker, or None for
            full checks. Membership, dates and identity are always checked.

    Returns:
        Identical scientific tracking records, endings and lineage.

    Examples:
        update = _track_reuters_waves(
            date=day, series=series, components=components, seed=True
        )
    """
    day = _as_day(date, "date")
    _validate_series(series)
    ordered_components = _validate_components(
        day, series, components, geometry_validator=geometry_validator
    )
    ordered_previous = _validate_previous(
        day, series, previous, geometry_validator=geometry_validator
    )

    if seed:
        if ordered_previous:
            msg = "seed=True requires no previous tracked waves"
            raise WaveTrackingError(msg)
        if not ordered_components:
            msg = "seed=True requires at least one daily component"
            raise WaveTrackingError(msg)
        current = tuple(
            _new_record(component, series, day, 0.0) for component in ordered_components
        )
        return WaveTrackingUpdate(day, series, current, (), ())
    if not ordered_previous:
        msg = "no previous tracked waves; start the series with seed=True"
        raise WaveTrackingError(msg)

    overlaps = _overlaps(ordered_components, ordered_previous)
    best = {
        index: _best_overlap(matches, ordered_previous)
        for index, matches in overlaps.items()
    }
    successors = _successors(overlaps, ordered_components, ordered_previous)
    current: list[TrackedWaveComponent] = []
    parent_ids: list[set[str]] = [set() for _ in ordered_components]
    relationships: list[dict[str, Literal["split", "merge"]]] = [
        {} for _ in ordered_components
    ]

    for current_index, component in enumerate(ordered_components):
        matched = best[current_index]
        if (
            matched is not None
            and successors.get(matched.predecessor_index) == current_index
        ):
            record = _continued_record(
                ordered_previous[matched.predecessor_index],
                component,
                _overlap_percent(matched, ordered_previous),
            )
            current.append(record)
            for predecessor_index in _absorbed_predecessors(current_index, successors):
                if predecessor_index != matched.predecessor_index:
                    predecessor = ordered_previous[predecessor_index]
                    parent_ids[current_index].add(predecessor.wave_id)
                    relationships[current_index][predecessor.wave_id] = "merge"
            continue

        record = _new_record(component, series, day, 0.0)
        current.append(record)
        for predecessor_index in _absorbed_predecessors(current_index, successors):
            predecessor = ordered_previous[predecessor_index]
            parent_ids[current_index].add(predecessor.wave_id)
            relationships[current_index][predecessor.wave_id] = "merge"
        parent = _best_parent(overlaps[current_index], component, ordered_previous)
        if parent is not None:
            predecessor = ordered_previous[parent.predecessor_index]
            parent_ids[current_index].add(predecessor.wave_id)
            relationships[current_index].setdefault(predecessor.wave_id, "split")

    ended: list[EndedWave] = []
    for predecessor_index, predecessor in enumerate(ordered_previous):
        successor = successors.get(predecessor_index)
        if successor is not None:
            matched = best[successor]
            if matched is not None and matched.predecessor_index == predecessor_index:
                continue
            parent_ids[successor].add(predecessor.wave_id)
            relationships[successor][predecessor.wave_id] = "merge"
            ended.append(
                EndedWave(
                    predecessor,
                    day - np.timedelta64(1, "D"),
                    "merged",
                    current[successor].wave_id,
                )
            )
        else:
            ended.append(
                EndedWave(predecessor, day - np.timedelta64(1, "D"), "died", None)
            )

    resolved_current = tuple(
        TrackedWaveComponent(
            series=record.series,
            component=record.component,
            component_id=record.component_id,
            wave_id=record.wave_id,
            start_date=record.start_date,
            peak_qualified_area=record.peak_qualified_area,
            peak_date=record.peak_date,
            continues=record.continues,
            overlap_percent=record.overlap_percent,
            parent_wave_ids=tuple(sorted(parent_ids[index])),
            name=record.name,
        )
        for index, record in enumerate(current)
    )
    lineage = tuple(
        LineageEdge(series, source, record.wave_id, day, relationship)
        for index, record in enumerate(resolved_current)
        for source, relationship in relationships[index].items()
        if source != record.wave_id
    )
    _validate_output(ordered_previous, resolved_current, ended, lineage)
    return WaveTrackingUpdate(
        day,
        series,
        tuple(sorted(resolved_current, key=lambda record: record.wave_id)),
        tuple(sorted(ended, key=lambda record: record.final_record.wave_id)),
        tuple(
            sorted(
                lineage,
                key=lambda edge: (
                    edge.series.temperature,
                    edge.series.direction,
                    edge.event_date,
                    edge.relationship,
                    edge.source_wave_id,
                    edge.target_wave_id,
                ),
            )
        ),
    )


def _validate_series(series: WaveSeries) -> None:
    """Validate the shared temperature and direction model.

    Steps:
        1. Check the shared temperature/direction model.
        2. Report unsupported pairs through the module validation error.

    Args:
        series: Candidate temperature and direction series.

    Returns:
        None when the series is supported.

    Examples:
        _validate_series(WaveSeries("tasmax", "warm"))
    """
    if not isinstance(series, WaveSeries):
        msg = "series must be WaveSeries"
        raise TypeError(msg)
    try:
        series.validate()
    except ValueError as error:
        raise WaveTrackingError(
            "series must use one of the four Reuters branches"
        ) from error


def _validate_components(
    day: np.datetime64,
    series: WaveSeries,
    components: Sequence[ComponentMembership],
    *,
    geometry_validator: Callable[[ComponentMembership], None] | None = None,
) -> tuple[ComponentMembership, ...]:
    """Validate and canonically order one day's components before tracking.

    Steps:
        1. Check each component's day, membership, and geometry.
        2. Reject repeated component IDs and return canonical component order.

    Args:
        day: The tracking day that every component must represent.
        series: Temperature and direction series for the components.
        components: Unordered daily component memberships to validate.
        geometry_validator: Optional geometry checker for immutable geometries.

    Returns:
        Components in ascending component-ID order after all membership,
        geometry, role-area, and positive-overlap checks pass.

    Raises:
        TypeError: If ``components`` contains a non-membership value.
        WaveTrackingError: If a component is invalid or full geometries have
            a positive-area overlap.

    Examples:
        Two boundary-touching valid components are returned in canonical order,
        while components with a positive-area full-geometry intersection raise
        ``WaveTrackingError``.
    """
    supplied = tuple(components)
    if any(not isinstance(component, ComponentMembership) for component in supplied):
        msg = "components must contain ComponentMembership records"
        raise TypeError(msg)
    ordered = tuple(sorted(supplied, key=lambda component: component.component_id))
    component_ids: set[str] = set()
    labels: set[int] = set()
    for component in ordered:
        if component.component_id in component_ids:
            msg = "component IDs must be unique within one branch and date"
            raise WaveTrackingError(msg)
        component_ids.add(component.component_id)
        if _as_day(component.geometry.time, "component time") != day:
            msg = "all components must belong to the tracking date"
            raise WaveTrackingError(msg)
        if component.geometry.label <= 0 or component.geometry.label in labels:
            msg = "component labels must be unique positive daily-local identifiers"
            raise WaveTrackingError(msg)
        labels.add(component.geometry.label)
        _validate_component_membership(component, series)
        (geometry_validator or _validate_component_geometry)(component)
    full_geometries = tuple(component.geometry.full_geometry for component in ordered)
    spatial_index = _spatial_index(full_geometries)
    for index, component in enumerate(ordered):
        for other_index in _candidate_geometry_indices(
            component.geometry.full_geometry,
            full_geometries,
            spatial_index,
        ):
            if other_index <= index:
                continue
            if (
                _area_km2(
                    component.geometry.full_geometry.intersection(
                        full_geometries[other_index]
                    )
                )
                > 0
            ):
                msg = "daily component full geometries must not overlap"
                raise WaveTrackingError(msg)
    return ordered


def _validate_component_membership(
    component: ComponentMembership, series: WaveSeries
) -> None:
    """Validate identity for a supplied footprint.

    Steps:
        1. Check the complete daily component ID and membership fields.
        2. Require the ID to agree with the date and first occupied grid cell.

    Args:
        component: Supplied daily footprint.
        series: Temperature and direction being tracked.

    Returns:
        None when the footprint passes the identity check.

    Raises:
        WaveTrackingError: If the footprint identity is invalid.

    Examples:
        _validate_component_membership(component, WaveSeries("tasmax", "warm"))
    """
    if len(component.component_id) != 32 or any(
        character not in "0123456789abcdef" for character in component.component_id
    ):
        msg = "component_id must be a 32-character lowercase date-and-cell digest"
        raise WaveTrackingError(msg)
    expected = component_identity(
        temperature=series.temperature,
        direction=series.direction,
        date=component.geometry.time,
        geometry=component.geometry.full_geometry,
    )
    if component.component_id != expected:
        msg = "component_id must match its date and extent geometry"
        raise WaveTrackingError(msg)


def _validate_component_geometry(membership: ComponentMembership) -> None:
    """Check one component's geometries and stored areas against each other.

    Steps:
        1. Check role shapes and coordinate bounds.
        2. Measure role areas and compare their union and stored totals within tolerance.

    Args:
        membership: The component whose geometry is being checked, carrying the
            component ID that names it in any failure message.

    Returns:
        None. The geometry is accepted when every rule holds.

    Raises:
        WaveTrackingError: If the geometries are unusable or the stored areas
            disagree with them. Area failures quote both measurements and how
            far apart they sit, so a log identifies the problem without the
            original run.

    Examples:
        A component whose qualifying and grace geometries sit inside its full
        geometry without overlapping, and whose three role areas add up to the
        full geometry's equal-area measurement, is accepted.
    """
    component = membership.geometry
    for name, geometry in (
        ("geometry", component.geometry),
        ("grace_geometry", component.grace_geometry),
        ("full_geometry", component.full_geometry),
    ):
        if geometry is not None and (
            not isinstance(geometry, Geometry)
            or geometry.is_empty
            or not geometry.is_valid
        ):
            msg = f"component {name} must be a non-empty valid geometry when present"
            raise WaveTrackingError(msg)
    if component.geometry is not None and not component.full_geometry.covers(
        component.geometry
    ):
        msg = "component full_geometry must contain geometry"
        raise WaveTrackingError(msg)
    if component.grace_geometry is not None and not component.full_geometry.covers(
        component.grace_geometry
    ):
        msg = "component full_geometry must contain grace_geometry"
        raise WaveTrackingError(msg)
    if (
        component.geometry is not None
        and component.grace_geometry is not None
        and _area_km2(component.geometry.intersection(component.grace_geometry))
        > _COMPONENT_AREA_ABSOLUTE_TOLERANCE_KM2
    ):
        msg = "component qualifying and grace geometries must not overlap"
        raise WaveTrackingError(msg)
    for name, value in (
        ("area_qualified", component.area_qualified),
        ("area_grace", component.area_grace),
        ("area_gap_filled", component.area_gap_filled),
    ):
        if not isinstance(value, Real) or not np.isfinite(value) or value < 0:
            msg = f"component {name} must be a finite non-negative area"
            raise WaveTrackingError(msg)
    expected_qualified = _area_km2(component.geometry)
    expected_grace = _area_km2(component.grace_geometry)
    expected_full = _area_km2(component.full_geometry)
    stated_full = _full_area(component)
    if expected_full <= 0.0 or (
        component.geometry is None and component.grace_geometry is None
    ):
        msg = "component must contain qualifying or grace geometry with positive full area"
        raise WaveTrackingError(msg)
    if not np.isclose(
        component.area_qualified,
        expected_qualified,
        rtol=_COMPONENT_AREA_RELATIVE_TOLERANCE,
        atol=_COMPONENT_AREA_ABSOLUTE_TOLERANCE_KM2,
    ):
        msg = "component area_qualified must match geometry in EPSG:6933 km2"
        raise WaveTrackingError(msg)
    if not np.isclose(
        component.area_grace,
        expected_grace,
        rtol=_COMPONENT_AREA_RELATIVE_TOLERANCE,
        atol=_COMPONENT_AREA_ABSOLUTE_TOLERANCE_KM2,
    ):
        msg = "component area_grace must match grace_geometry in EPSG:6933 km2"
        raise WaveTrackingError(msg)
    if not np.isclose(
        stated_full,
        expected_full,
        rtol=_ROLE_AREA_RELATIVE_TOLERANCE,
        atol=_ROLE_AREA_ABSOLUTE_TOLERANCE_KM2,
    ):
        msg = (
            "component role areas must match full_geometry in EPSG:6933 km2: "
            f"component {membership.component_id} label {component.label} on "
            f"{component.time.astype('datetime64[D]')} states "
            f"{stated_full} km2 (qualified {component.area_qualified}, grace "
            f"{component.area_grace}, gap {component.area_gap_filled}) against "
            f"a measured {expected_full} km2, "
            f"{_relative_difference(stated_full, expected_full):.3e} apart"
        )
        raise WaveTrackingError(msg)
    if not np.isclose(
        component.qualified_fraction,
        component.area_qualified / stated_full,
        rtol=1e-9,
        atol=1e-9,
    ):
        msg = "component qualified_fraction must match its role areas"
        raise WaveTrackingError(msg)


def _validate_previous(
    day: np.datetime64,
    series: WaveSeries,
    previous: Sequence[TrackedWaveComponent],
    *,
    geometry_validator: Callable[[ComponentMembership], None] | None = None,
) -> tuple[TrackedWaveComponent, ...]:
    """Check preceding records while preserving all identity and date rules.

    Steps:
        1. Require one preceding record per wave ID.
        2. Validate each record against the current day and series.

    Args:
        day: Incoming tracking date.
        series: Independent temperature and direction series.
        previous: Records active on the preceding date.
        geometry_validator: Private reusable checker or None for full geometry checks.

    Returns:
        Verified preceding records sorted by wave ID.

    Examples:
        ordered = _validate_previous(day, series, previous)
    """
    supplied = tuple(previous)
    if any(not isinstance(record, TrackedWaveComponent) for record in supplied):
        msg = "previous must contain TrackedWaveComponent records"
        raise TypeError(msg)
    ordered = tuple(sorted(supplied, key=lambda record: record.wave_id))
    seen: set[str] = set()
    for record in ordered:
        if not _is_wave_id(record.wave_id) or record.wave_id in seen:
            msg = "previous wave IDs must be unique permanent wave IDs"
            raise WaveTrackingError(msg)
        seen.add(record.wave_id)
        try:
            _validate_previous_record(
                day, series, record, geometry_validator=geometry_validator
            )
        except WaveTrackingError as error:
            msg = f"{error} for wave {record.wave_id}"
            raise WaveTrackingError(msg) from error
    return ordered


def _validate_previous_record(
    day: np.datetime64,
    series: WaveSeries,
    record: TrackedWaveComponent,
    *,
    geometry_validator: Callable[[ComponentMembership], None] | None = None,
) -> None:
    """Check one replayed wave against the date it is being carried into.

    Steps:
        1. Validate preceding identity, dates, peak, name, and overlap fields.
        2. Check component geometry and require the immediately preceding calendar day.

    Args:
        day: Date of the incoming daily components.
        series: Temperature and direction branch being advanced.
        record: One wave carried forward from the preceding date.
        geometry_validator: Optional geometry checker for immutable geometries.

    Returns:
        None. The record is accepted when every rule holds.

    Raises:
        WaveTrackingError: If the record cannot be carried into ``day``.

    Examples:
        A wave whose component is dated one day before ``day``, whose peak
        falls between its start and its component date, and whose overlap sits
        between 0 and 100 is accepted.
    """
    if record.series != series:
        msg = "previous waves must belong to the same temperature and direction series"
        raise WaveTrackingError(msg)
    if record.component_id != record.component.component_id:
        msg = "previous component_id must match component membership"
        raise WaveTrackingError(msg)
    _validate_component_membership(record.component, series)
    (geometry_validator or _validate_component_geometry)(record.component)
    previous_day = _as_day(record.component.geometry.time, "previous component time")
    if previous_day + np.timedelta64(1, "D") != day:
        msg = "previous waves must be from the consecutive preceding date"
        raise WaveTrackingError(msg)
    start_date = _as_day(record.start_date, "start_date")
    if record.wave_id[:8] != np.datetime_as_string(start_date).replace("-", ""):
        msg = "previous wave key birth-date prefix must match start_date"
        raise WaveTrackingError(msg)
    peak_date = _as_day(record.peak_date, "peak_date")
    if start_date > previous_day or not start_date <= peak_date <= previous_day:
        msg = "previous wave dates must run from start_date through its component date"
        raise WaveTrackingError(msg)
    if (
        not isinstance(record.peak_qualified_area, Real)
        or not np.isfinite(record.peak_qualified_area)
        or record.peak_qualified_area < record.component.geometry.area_qualified
    ):
        msg = "previous peak_qualified_area must be finite and include its component"
        raise WaveTrackingError(msg)
    if not 0.0 <= record.overlap_percent <= 100.0:
        msg = "previous overlap_percent must be between 0 and 100"
        raise WaveTrackingError(msg)
    if (
        len(set(record.parent_wave_ids)) != len(record.parent_wave_ids)
        or record.wave_id in record.parent_wave_ids
        or any(not _is_wave_id(parent) for parent in record.parent_wave_ids)
    ):
        msg = "previous parent_wave_ids must be unique full IDs and exclude itself"
        raise WaveTrackingError(msg)


def _overlaps(
    components: Sequence[ComponentMembership], previous: Sequence[TrackedWaveComponent]
) -> dict[int, tuple[_Overlap, ...]]:
    """Measure exact overlaps after conservatively shortlisting by envelope.

    Steps:
        1. Project preceding and current extents.
        2. Use envelope candidates when a spatial index is available.
        3. Measure positive intersections for the shortlisted pairs.

    The spatial index only eliminates predecessor geometries whose bounding
    boxes do not meet a current component's bounding box. Every shortlisted
    pair still uses the established exact intersection and equal-area
    measurement, preserving tracking decisions and their ordering.

    Args:
        components: Current daily components in their already canonical order.
        previous: Preceding tracked waves in their already canonical order.

    Returns:
        Exact positive-area overlaps for each current-component index, with
        predecessor indices in ascending order.

    Examples:
        A current component whose bounding box is disjoint from every preceding
        wave has an empty overlap tuple without measuring any intersections.
    """
    predecessor_geometries = _project_extents(
        tuple(predecessor.component.geometry for predecessor in previous)
    )
    current_geometries = _project_extents(
        tuple(component.geometry for component in components)
    )
    spatial_index = _spatial_index(predecessor_geometries)
    return {
        current_index: tuple(
            _Overlap(predecessor_index, area)
            for predecessor_index in _candidate_geometry_indices(
                current_geometries[current_index],
                predecessor_geometries,
                spatial_index,
            )
            if (
                area := current_geometries[current_index]
                .intersection(predecessor_geometries[predecessor_index])
                .buffer(0)
                .area
                / 1_000_000
            )
            > 0.0
        )
        for current_index, component in enumerate(components)
    }


def _spatial_index(
    geometries: Sequence[Geometry],
) -> STRtree | None:
    """Build an envelope index when every geometry can be safely indexed.

    Steps:
        1. Check whether every geometry has usable bounds.
        2. Build an STRtree or return None for full-scan fallback.

    Args:
        geometries: Candidate geometries in their original positional order.

    Returns:
        A Shapely STRtree retaining the supplied positions, or None when any
        geometry has an unsafe envelope and callers must use every pair.

    Examples:
        An ordinary pair of finite polygons produces an STRtree; a geometry
        with non-finite bounds makes the result None.
    """
    if not geometries or any(
        not _has_indexable_bounds(geometry) for geometry in geometries
    ):
        return None
    return STRtree(geometries)


def _candidate_geometry_indices(
    geometry: Geometry,
    candidates: Sequence[Geometry],
    spatial_index: STRtree | None,
) -> tuple[int, ...]:
    """Return candidate positions whose envelopes could overlap a geometry.

    Steps:
        1. Query the envelope index when available.
        2. Otherwise return every geometry position for exact intersection checks.

    Args:
        geometry: One current component's full geometry.
        candidates: Geometries in the positional order used by the caller.
        spatial_index: An index built from ``candidates``, when safe.

    Returns:
        Ascending candidate indices whose bounding boxes meet ``geometry``;
        every candidate when either side cannot safely use the index.

    Examples:
        A component separated from all indexed candidates returns ``()``; an
        unsafe geometry returns every candidate index for exhaustive
        intersection measurement.
    """
    if spatial_index is None or not _has_indexable_bounds(geometry):
        return tuple(range(len(candidates)))
    return tuple(sorted(int(index) for index in spatial_index.query(geometry)))


def _has_indexable_bounds(geometry: Geometry) -> bool:
    """Return whether a geometry has a finite, non-empty valid STRtree envelope.

    Steps:
        1. Inspect geometry type, validity, emptiness, and bounds.
        2. Require finite bounds before indexing.

    Args:
        geometry: Geometry considered for an envelope-only spatial index.

    Returns:
        True if the geometry can safely participate in an STRtree shortlist.

    Examples:
        A valid finite polygon returns True, while an empty geometry returns
        False and requires exhaustive handling by the caller.
    """
    bounds = geometry.bounds
    return bool(
        not geometry.is_empty
        and geometry.is_valid
        and len(bounds) == 4
        and np.all(np.isfinite(bounds))
    )


def _best_overlap(
    matches: Sequence[_Overlap], previous: Sequence[TrackedWaveComponent]
) -> _Overlap | None:
    """Select the strongest preceding overlap with deterministic tie breaks.

    Steps:
        1. Order intersections by descending area.
        2. Break ties by predecessor birth date and component ID.

    Args:
        matches: Measured predecessor intersections.
        previous: Preceding records indexed by each overlap.

    Returns:
        The selected overlap, or None when there are no matches.

    Examples:
        best = _best_overlap(matches, previous)
    """
    return (
        min(
            matches,
            key=lambda match: (
                -match.area,
                _as_day(previous[match.predecessor_index].start_date, "start_date"),
                previous[match.predecessor_index].component_id,
            ),
        )
        if matches
        else None
    )


def _successors(
    overlaps: dict[int, tuple[_Overlap, ...]],
    components: Sequence[ComponentMembership],
    previous: Sequence[TrackedWaveComponent],
) -> dict[int, int]:
    """Choose each predecessor's qualifying continuation component.

    Steps:
        1. Validate each predecessor overlap share.
        2. Keep candidates covering the continuation threshold.
        3. Choose each predecessor's largest overlap, breaking ties by component ID.

    Args:
        overlaps: Current component indices mapped to predecessor intersections.
        components: Current components used to break equal-area ties.
        previous: Preceding tracked records and their stored areas.

    Returns:
        Mapping from predecessor indices to their selected current indices.

    Examples:
        successors = _successors(overlaps, components, previous)
    """
    eligible: dict[int, list[tuple[int, _Overlap]]] = {}
    for current_index, matches in overlaps.items():
        for match in matches:
            _overlap_percent(match, previous)
            if (
                match.area
                / _full_area(previous[match.predecessor_index].component.geometry)
                >= _CONTINUATION_COVERAGE
            ):
                eligible.setdefault(match.predecessor_index, []).append(
                    (current_index, match)
                )
    return {
        predecessor_index: min(
            successors,
            key=lambda successor: (
                -successor[1].area,
                components[successor[0]].component_id,
            ),
        )[0]
        for predecessor_index, successors in eligible.items()
    }


def _absorbed_predecessors(
    current_index: int, successors: dict[int, int]
) -> tuple[int, ...]:
    """Return predecessors continuing into one current component.

    Steps:
        1. Select predecessors whose successor is the current component.
        2. Return their indices in mapping order.

    Args:
        current_index: Index of the current component.
        successors: Chosen current index for each predecessor.

    Returns:
        Predecessor indices assigned to the current component.

    Examples:
        assert _absorbed_predecessors(2, {0: 2, 1: 3}) == (0,)
    """
    return tuple(
        predecessor
        for predecessor, successor in successors.items()
        if successor == current_index
    )


def _best_parent(
    matches: Sequence[_Overlap],
    component: ComponentMembership,
    previous: Sequence[TrackedWaveComponent],
) -> _Overlap | None:
    """Choose a qualifying parent for a newly born component.

    Steps:
        1. Keep overlaps meeting the current component coverage threshold.
        2. Choose the largest overlap, breaking ties by birth date and ID.

    Args:
        matches: Measured predecessor intersections.
        component: Current component whose coverage is tested.
        previous: Preceding records used to break ties.

    Returns:
        The strongest qualifying overlap, or None.

    Examples:
        parent = _best_parent(matches, component, previous)
    """
    candidates = [
        match
        for match in matches
        if match.area / _full_area(component.geometry) >= _CONTINUATION_COVERAGE
    ]
    return (
        min(
            candidates,
            key=lambda match: (
                -match.area,
                _as_day(previous[match.predecessor_index].start_date, "start_date"),
                previous[match.predecessor_index].component_id,
            ),
        )
        if candidates
        else None
    )


def _overlap_percent(
    match: _Overlap, previous: Sequence[TrackedWaveComponent]
) -> float:
    """Return how much of a preceding component the overlap covers.

    Steps:
        1. Divide measured intersection area by predecessor stored area.
        2. Reject nonfinite or excessive shares and clamp tolerated endpoint differences.

    The intersection is part of the preceding component, so a true share is
    always between 0 and 100.  The reading is a quotient of two separately
    measured areas: the intersection is measured once, while the divisor is the
    sum of the preceding component's three stored role areas.  Those role areas
    only have to agree with their measured outline within
    ``_ROLE_AREA_RELATIVE_TOLERANCE``, so that same relative disagreement
    carries straight through the division.  Scaled to a percent it becomes
    ``_OVERLAP_MEASUREMENT_TOLERANCE``, which is how far outside 0 to 100 a
    reading may sit and still describe real geometry.  Readings inside that
    range are snapped to the end they belong to.  Readings beyond it describe
    an intersection larger than the shape it was cut from, so they fail here
    rather than being carried into tracker state.

    Args:
        match: One measured intersection with a preceding component.
        previous: Preceding tracked waves, indexed by ``predecessor_index``.

    Returns:
        The covered share of the preceding component, from 0.0 to 100.0.

    Raises:
        WaveTrackingError: If the reading is too far outside 0 to 100 to be
            measurement disagreement, including a missing or infinite reading.

    Examples:
        A component fully covered by its successor reads as exactly 100.0, even
        when the measured areas disagree in their final digits.
    """
    percent = (
        100.0
        * match.area
        / _full_area(previous[match.predecessor_index].component.geometry)
    )
    if not (
        -_OVERLAP_MEASUREMENT_TOLERANCE
        <= percent
        <= 100.0 + _OVERLAP_MEASUREMENT_TOLERANCE
    ):
        msg = "overlap must be a finite share of the preceding component area"
        raise WaveTrackingError(msg)
    if percent <= _OVERLAP_BOUNDARY_TOLERANCE:
        return 0.0
    if percent >= 100.0 - _OVERLAP_BOUNDARY_TOLERANCE:
        return 100.0
    return percent


def _continued_record(
    predecessor: TrackedWaveComponent,
    component: ComponentMembership,
    overlap_percent: float,
) -> TrackedWaveComponent:
    """Carry a preceding wave into its selected daily component.

    Steps:
        1. Compare today's qualified area with the previous peak.
        2. Carry forward the permanent ID, birth date, and name into today's record.

    Args:
        predecessor: Preceding record whose identity and name continue.
        component: Selected current footprint.
        overlap_percent: Share of the predecessor covered by the current component.

    Returns:
        Current tracked record with preserved identity and updated peak.

    Examples:
        current = _continued_record(previous, component, 90.0)
    """
    geometry = component.geometry
    if geometry.area_qualified > predecessor.peak_qualified_area:
        peak_qualified_area, peak_date = (
            geometry.area_qualified,
            _as_day(geometry.time, "component time"),
        )
    else:
        peak_qualified_area, peak_date = (
            predecessor.peak_qualified_area,
            _as_day(predecessor.peak_date, "peak_date"),
        )
    return TrackedWaveComponent(
        predecessor.series,
        component,
        component.component_id,
        predecessor.wave_id,
        _as_day(predecessor.start_date, "start_date"),
        peak_qualified_area,
        peak_date,
        True,
        overlap_percent,
        (),
        predecessor.name,
    )


def _new_record(
    component: ComponentMembership,
    series: WaveSeries,
    day: np.datetime64,
    overlap_percent: float,
) -> TrackedWaveComponent:
    """Initialize a tracked record for a newly born daily component.

    Steps:
        1. Derive a permanent ID from the birth date and component hash.
        2. Initialize peak area and peak date from today's component.

    Args:
        component: Birth footprint and its daily hash.
        series: Temperature and direction of the new wave.
        day: Birth calendar day.
        overlap_percent: Measured parent overlap share stored on the record.

    Returns:
        New record with a birth ID and initial peak, without a name.

    Examples:
        record = _new_record(component, series, day, 0.0)
    """
    wave_id = birth_wave_identity(
        temperature=series.temperature,
        direction=series.direction,
        date=day,
        component_id=component.component_id,
    )
    return TrackedWaveComponent(
        series,
        component,
        component.component_id,
        wave_id,
        day,
        component.geometry.area_qualified,
        day,
        False,
        overlap_percent,
        (),
    )


def _project_extents(
    components: Sequence[DailyWaveComponentGeometry],
) -> tuple[Geometry, ...]:
    """Combine footprint layers, then project and repair them for tracking.

    The shapes are joined in this order: qualified, grace, then full geometry.
    Even when the full outline contains the other layers, omitting these
    unions can change coordinate order and the last digits of an overlap.

    Steps:
        1. Collect qualified, grace and complete shapes in that order for each component.
        2. Join each component's available shapes in that same order.
        3. Project the combined regions to EPSG:6933, repair polygon boundaries,
           and return the shapes used to measure daily overlaps.

    Args:
        components: Daily footprint geometries in their tracking order.

    Returns:
        Combined, repaired EPSG:6933 geometries in the same order.

    Examples:
        projected = _project_extents((component.geometry,))
    """
    extents: list[Geometry] = []
    for component in components:
        present = [
            geometry
            for geometry in (
                component.geometry,
                component.grace_geometry,
                component.full_geometry,
            )
            if geometry is not None and not geometry.is_empty
        ]
        extent = present[0]
        for geometry in present[1:]:
            extent = extent.union(geometry)
        extents.append(extent)
    return tuple(gpd.GeoSeries(extents, crs="EPSG:4326").to_crs("EPSG:6933").buffer(0))


def _area_km2(geometry: Geometry | None) -> float:
    """Measure an optional geographic geometry in square kilometers.

    Steps:
        1. Treat an absent geometry as zero area.
        2. Project a present shape to the equal-area CRS and convert square meters to square kilometers.

    Args:
        geometry: EPSG:4326 shape, or None.

    Returns:
        Equal-area size in square kilometers, with zero for None.

    Examples:
        assert _area_km2(None) == 0.0
    """
    return (
        0.0
        if geometry is None
        else float(transform(_AREA_TRANSFORMER.transform, geometry).area / 1_000_000)
    )


def _full_area(component: DailyWaveComponentGeometry) -> float:
    """Sum a component's qualified, grace, and gap-filled areas.

    Steps:
        1. Read qualified, grace, and gap-filled stored areas.
        2. Return their sum.

    Args:
        component: Daily geometry with stored role areas.

    Returns:
        Complete footprint area in square kilometers.

    Examples:
        area = _full_area(component.geometry)
    """
    return component.area_qualified + component.area_grace + component.area_gap_filled


def _as_day(value: np.datetime64, label: str) -> np.datetime64:
    """Validate a NumPy datetime and reduce it to a calendar day.

    Steps:
        1. Reject missing or non-NumPy datetime values.
        2. Convert valid values to calendar-day precision.

    Args:
        value: Candidate NumPy datetime.
        label: Field name used in validation errors.

    Returns:
        NumPy datetime at day precision.

    Examples:
        day = _as_day(np.datetime64("2024-01-01T12:00"), "date")
    """
    if not isinstance(value, np.datetime64) or np.isnat(value):
        msg = f"{label} must be a valid numpy datetime64"
        raise WaveTrackingError(msg)
    return cast("np.datetime64", value.astype("datetime64[D]"))


def _is_wave_id(value: str) -> bool:
    """Check the permanent wave ID format.

    Steps:
        1. Check the compact birth-date prefix and separator.
        2. Require the complete 32-character lowercase date-and-cell digest.

    Args:
        value: Candidate wave ID.

    Returns:
        Whether the ID has birth-date and complete daily-ID form.

    Examples:
        _is_wave_id("19600105-94691d760123456789abcdef01234567")
    """
    return isinstance(value, str) and WAVE_ID_PATTERN.fullmatch(value) is not None


def _validate_output(
    previous: Sequence[TrackedWaveComponent],
    current: Sequence[TrackedWaveComponent],
    ended: Sequence[EndedWave],
    lineage: Sequence[LineageEdge],
) -> None:
    """Validate identity, ancestry, and endings in daily tracking output.

    Steps:
        1. Check active and ended IDs are exclusive and unique.
        2. Validate parent references, overlap percentages, and merge targets.
        3. Require lineage edges to reference distinct known endpoints on the current date and series.

    Args:
        previous: Preceding active records.
        current: Current active records.
        ended: Waves ended by death or merge today.
        lineage: Split and merge relationships emitted today.

    Returns:
        None when output references and overlap fields are valid.

    Examples:
        _validate_output(previous, current, ended, lineage)
    """
    current_ids = {record.wave_id for record in current}
    ended_ids = {record.final_record.wave_id for record in ended}
    if len(current_ids) != len(current) or current_ids & ended_ids:
        msg = "tracking output must have one active or ended result for every wave ID"
        raise WaveTrackingError(msg)
    previous_ids = {record.wave_id for record in previous}
    for record in current:
        if (
            len(set(record.parent_wave_ids)) != len(record.parent_wave_ids)
            or record.wave_id in record.parent_wave_ids
            or set(record.parent_wave_ids) - previous_ids
        ):
            msg = "current parent_wave_ids must be unique preceding IDs and exclude itself"
            raise WaveTrackingError(msg)
        if not 0.0 <= record.overlap_percent <= 100.0:
            msg = "current overlap_percent must be between 0 and 100"
            raise WaveTrackingError(msg)
    for record in ended:
        if (
            record.end_reason == "merged"
            and record.merged_into_wave_id not in current_ids
        ):
            msg = "merged waves must point to a current wave ID"
            raise WaveTrackingError(msg)
        if record.end_reason == "died" and record.merged_into_wave_id is not None:
            msg = "ordinary ended waves must not have a merge target"
            raise WaveTrackingError(msg)
    for edge in lineage:
        if (
            edge.series != current[0].series
            or edge.source_wave_id not in previous_ids
            or edge.target_wave_id not in current_ids
            or edge.source_wave_id == edge.target_wave_id
            or edge.event_date
            != current[0].component.geometry.time.astype("datetime64[D]")
        ):
            msg = (
                "lineage edges must have known same-branch, same-day, non-self "
                "endpoints"
            )
            raise WaveTrackingError(msg)
