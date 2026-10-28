"""Check historical name order from complete cells without relying on input bytes."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import MultiPolygon, Polygon, box

from reuters_temperature_storms.cells import WaveCells
from reuters_temperature_storms.components import DailyWaveComponentGeometry
from reuters_temperature_storms.footprint_order import (
    FootprintOrderError,
    HistoricalFootprintOrder,
    _HistoricalComparisonSort,
    _NativeRoleCells,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from shapely import Geometry

pytestmark = pytest.mark.unit
DAY = np.datetime64("2024-01-01")


def _cells(
    specification: Sequence[tuple[float, float, str]] = (),
    *,
    day: np.datetime64 = DAY,
) -> WaveCells:
    """Create exact synthetic native rectangles with their qualifying roles.

    Args:
        specification: Longitude, latitude and qualified/grace/gap role triples.
        day: Calendar date stored at nanosecond precision in every row.

    Returns:
        A validated complete native cell table. A +180 cell has both display
        halves, and pole cells stop at the pole.

    Examples:
        cells = _cells(((0.0, 0.0, "qualified"), (0.25, 0.0, "grace")))
    """
    rows = []
    for longitude, latitude, role in specification:
        bottom = max(-90.0, latitude - 0.125)
        top = min(90.0, latitude + 0.125)
        parts = (
            (
                box(179.875, bottom, 180.0, top),
                box(-180.0, bottom, -179.875, top),
            )
            if longitude == 180.0
            else (box(longitude - 0.125, bottom, longitude + 0.125, top),)
        )
        rows.extend(
            (
                float(longitude),
                float(latitude),
                np.nan if role == "gap" else 5.0,
                np.nan if role == "gap" else float(role == "grace"),
                role == "gap",
                part,
            )
            for part in parts
        )
    frame = gpd.GeoDataFrame(
        {
            "longitude": pd.Series([row[0] for row in rows], dtype="float64"),
            "latitude": pd.Series([row[1] for row in rows], dtype="float64"),
            "days": pd.Series([row[2] for row in rows], dtype="float64"),
            "grace_days": pd.Series([row[3] for row in rows], dtype="float64"),
            "gap_filled": pd.Series([row[4] for row in rows], dtype="bool"),
            "geometry": gpd.GeoSeries([row[5] for row in rows], crs=4326),
            "date": pd.Series([day] * len(rows), dtype="datetime64[ns]"),
        },
        crs=4326,
    )
    return WaveCells(frame)


def _role(
    qualified: Geometry | None,
    grace: Geometry | None = None,
    gap: Geometry | None = None,
    *,
    day: np.datetime64 = DAY,
    label: int = 1,
) -> DailyWaveComponentGeometry:
    """Retain supplied role coverage with deliberately precise scalar inputs.

    Args:
        qualified: Complete qualifying native coverage, or None when absent.
        grace: Complete grace native coverage, or None when absent.
        gap: Complete gap-fill coverage, or None when absent.
        day: Represented calendar day.
        label: Input label, which must have no effect on historical ordering.

    Returns:
        One role row whose full shape is the supplied coverage union. Scalar
        values are placeholders because this calculation must not measure area.

    Examples:
        role = _role(box(-0.125, -0.125, 0.125, 0.125))
    """
    shapes = [shape for shape in (qualified, grace, gap) if shape is not None]
    full = shapely.union_all(shapes)
    return DailyWaveComponentGeometry(
        time=day,
        label=label,
        geometry=qualified,
        grace_geometry=grace,
        full_geometry=full,
        area_qualified=1.234567890123,
        area_grace=2.345678901234,
        area_gap_filled=3.456789012345,
        qualified_fraction=0.123456789012,
    )


def _turn_ring(shape: Polygon) -> Polygon:
    """Change a ring's starting point and direction while retaining exact coverage.

    Args:
        shape: Single native rectangle to encode differently.

    Returns:
        An equivalent polygon with different stored coordinate bytes.

    Examples:
        turned = _turn_ring(box(-0.125, -0.125, 0.125, 0.125))
    """
    ring = list(shape.exterior.coords)[:-1]
    shifted = ring[2:] + ring[:2]
    return Polygon(list(reversed(shifted)))


def test_literal_single_cell_historical_digest() -> None:
    """Reproduce the original date-and-shape text vector for one native cell.

    Args:
        None.

    Returns:
        None after the complete digest agrees with the fixed historical value.

    Examples:
        Run ``pytest tests/test_footprint_order.py -k literal_single_cell``.
    """
    cells = _cells(((0.0, 0.0, "qualified"),))
    role = _role(box(-0.125, -0.125, 0.125, 0.125))
    assert HistoricalFootprintOrder.for_day(DAY, (role,), cells) == (
        "e974c8427e300f1c02f671230093f7b9",
    )


def test_polygon_encoding_labels_and_measurements_do_not_change_order() -> None:
    """Ignore ring bytes, row labels and scalar measurements when rebuilding order.

    Args:
        None.

    Returns:
        None after differently encoded source cells and roles give the same digest.
        The precise source geometry bytes and scalars remain unchanged.

    Examples:
        Run ``pytest tests/test_footprint_order.py -k polygon_encoding``.
    """
    cells = _cells(((0.0, 0.0, "qualified"), (0.25, 0.0, "grace")))
    role = _role(cells.frame.geometry.iloc[0], cells.frame.geometry.iloc[1])
    expected = HistoricalFootprintOrder.for_day(DAY, (role,), cells)
    frame = cells.frame.iloc[::-1].reset_index(drop=True).copy()
    frame.geometry = [_turn_ring(shape) for shape in frame.geometry]
    changed_cells = WaveCells(frame)
    changed = replace(
        role,
        label=867,
        geometry=shapely.reverse(role.geometry),
        grace_geometry=shapely.reverse(role.grace_geometry),
        full_geometry=shapely.reverse(role.full_geometry),
        area_qualified=9876543.123456789,
        area_grace=1234567.987654321,
    )
    cell_bytes = tuple(frame.geometry.to_wkb())
    role_bytes = tuple(
        shape.wkb
        for shape in (
            changed.geometry,
            changed.grace_geometry,
            changed.full_geometry,
        )
    )
    assert HistoricalFootprintOrder.for_day(DAY, (changed,), changed_cells) == expected
    assert tuple(frame.geometry.to_wkb()) == cell_bytes
    assert (
        tuple(
            shape.wkb
            for shape in (
                changed.geometry,
                changed.grace_geometry,
                changed.full_geometry,
            )
        )
        == role_bytes
    )
    assert changed.area_qualified == 9876543.123456789
    assert changed.area_grace == 1234567.987654321


def test_results_follow_supplied_role_order() -> None:
    """Return each digest beside the same input footprint after row permutations.

    Args:
        None.

    Returns:
        None after reversing both source and role rows preserves their associations.

    Examples:
        Run ``pytest tests/test_footprint_order.py -k supplied_role_order``.
    """
    cells = _cells(((0.0, 0.0, "qualified"), (5.0, 0.0, "qualified")))
    roles = tuple(
        _role(shape, label=index + 5)
        for index, shape in enumerate(cells.frame.geometry)
    )
    expected = HistoricalFootprintOrder.for_day(DAY, roles, cells)
    shuffled_cells = WaveCells(cells.frame.iloc[::-1].reset_index(drop=True))
    assert (
        HistoricalFootprintOrder.for_day(DAY, roles[::-1], shuffled_cells)
        == expected[::-1]
    )
    assert expected[0] != expected[1]


@pytest.mark.parametrize("latitude", [-90.0, 0.0, 90.0])
def test_dateline_halves_have_one_unchanged_order_digest(latitude: float) -> None:
    """Retain both native display halves, including the clipped pole cell shapes.

    Args:
        latitude: Native latitude at the equator or either pole.

    Returns:
        None after reversing part order and polygon direction preserves the digest.

    Examples:
        Run ``pytest tests/test_footprint_order.py -k dateline_halves``.
    """
    cells = _cells(((180.0, latitude, "qualified"),))
    parts = cells.frame.geometry.tolist()
    role = _role(MultiPolygon(parts))
    expected = HistoricalFootprintOrder.for_day(DAY, (role,), cells)
    changed_role = _role(MultiPolygon([_turn_ring(part) for part in reversed(parts)]))
    changed_cells = WaveCells(cells.frame.iloc[::-1].reset_index(drop=True))
    assert (
        HistoricalFootprintOrder.for_day(DAY, (changed_role,), changed_cells)
        == expected
    )
    assert len(expected[0]) == 32


def test_missing_dateline_half_in_finished_coverage_is_rejected() -> None:
    """Require finished footprint coverage to include both accepted native halves.

    Args:
        None.

    Returns:
        None after an otherwise valid half-only polygon is rejected.

    Examples:
        Run ``pytest tests/test_footprint_order.py -k missing_dateline_half``.
    """
    cells = _cells(((180.0, 0.0, "qualified"),))
    with pytest.raises(FootprintOrderError, match="match one native-cell component"):
        HistoricalFootprintOrder.for_day(
            DAY, (_role(cells.frame.geometry.iloc[0]),), cells
        )


def test_partial_qualified_role_is_rejected() -> None:
    """Reject partial role coverage even when the full outline matches exactly.

    Args:
        None.

    Returns:
        None after the changed qualifying role fails the complete-cell check.

    Examples:
        Run ``pytest tests/test_footprint_order.py -k partial_qualified_role``.
    """
    cells = _cells(((0.0, 0.0, "qualified"), (0.25, 0.0, "grace")))
    correct = _role(cells.frame.geometry.iloc[0], cells.frame.geometry.iloc[1])
    partial = replace(correct, geometry=box(-0.125, -0.125, 0.0, 0.125))
    with pytest.raises(
        FootprintOrderError, match="qualified must contain complete native cells"
    ):
        HistoricalFootprintOrder.for_day(DAY, (partial,), cells)


def test_missing_finished_footprint_is_rejected() -> None:
    """Reject a partial footprint table instead of calculating isolated birth order.

    Args:
        None.

    Returns:
        None after supplying only one of two real-cell components fails.

    Examples:
        Run ``pytest tests/test_footprint_order.py -k missing_finished_footprint``.
    """
    cells = _cells(((0.0, 0.0, "qualified"), (5.0, 0.0, "qualified")))
    with pytest.raises(FootprintOrderError, match="every finished real-cell footprint"):
        HistoricalFootprintOrder.for_day(
            DAY, (_role(cells.frame.geometry.iloc[0]),), cells
        )


def test_repeated_finished_footprint_is_rejected() -> None:
    """Require a unique native component for each supplied role row.

    Args:
        None.

    Returns:
        None after duplicate footprint rows fail despite a matching row count.

    Examples:
        Run ``pytest tests/test_footprint_order.py -k repeated_finished_footprint``.
    """
    cells = _cells(((0.0, 0.0, "qualified"), (5.0, 0.0, "qualified")))
    role = _role(cells.frame.geometry.iloc[0])
    with pytest.raises(FootprintOrderError, match="match one native-cell component"):
        HistoricalFootprintOrder.for_day(DAY, (role, role), cells)


@pytest.mark.parametrize("day", [None, "2024-01-01", np.datetime64("NaT", "D")])
def test_invalid_order_date_is_rejected(day: object) -> None:
    """Require a present NumPy day before computing historical allocation order.

    Args:
        day: Missing, text or absent NumPy date candidate.

    Returns:
        None after the invalid date is rejected.

    Examples:
        Run ``pytest tests/test_footprint_order.py -k invalid_order_date``.
    """
    with pytest.raises(FootprintOrderError, match="present NumPy calendar day"):
        HistoricalFootprintOrder.for_day(day, (), _cells())


def test_non_midnight_date_is_rejected() -> None:
    """Keep historical digest dates independent of an unintended time-of-day value.

    Args:
        None.

    Returns:
        None after a midday request is rejected.

    Examples:
        Run ``pytest tests/test_footprint_order.py -k non_midnight_date``.
    """
    with pytest.raises(FootprintOrderError, match="midnight"):
        HistoricalFootprintOrder.for_day(
            np.datetime64("2024-01-01T12:00"), (), _cells()
        )


def test_native_cell_dates_must_match() -> None:
    """Reject a complete native cell table from another day.

    Args:
        None.

    Returns:
        None after the date mismatch fails before shape reconstruction.

    Examples:
        Run ``pytest tests/test_footprint_order.py -k native_cell_dates``.
    """
    cells = _cells(((0.0, 0.0, "qualified"),), day=np.datetime64("2024-01-02"))
    with pytest.raises(FootprintOrderError, match="all native cell dates"):
        HistoricalFootprintOrder.for_day(
            DAY, (_role(cells.frame.geometry.iloc[0]),), cells
        )


def test_finished_role_dates_must_match() -> None:
    """Reject an otherwise complete finished footprint from another day.

    Args:
        None.

    Returns:
        None after the footprint date mismatch fails before reconstruction.

    Examples:
        Run ``pytest tests/test_footprint_order.py -k finished_role_dates``.
    """
    cells = _cells(((0.0, 0.0, "qualified"),))
    wrong = _role(cells.frame.geometry.iloc[0], day=np.datetime64("2024-01-02"))
    with pytest.raises(FootprintOrderError, match="all finished footprint dates"):
        HistoricalFootprintOrder.for_day(DAY, (wrong,), cells)


def test_complete_validated_native_cells_are_mandatory() -> None:
    """Require native context even for an empty footprint date.

    Args:
        None.

    Returns:
        None after missing or loose frame inputs fail the required cell-model check.

    Examples:
        Run ``pytest tests/test_footprint_order.py -k native_cells_are_mandatory``.
    """
    for absent in (None, _cells().frame):
        with pytest.raises(FootprintOrderError, match="validated complete WaveCells"):
            HistoricalFootprintOrder.for_day(DAY, (), absent)


def test_empty_and_gap_only_days_have_no_finished_digests() -> None:
    """Accept quiet dates without inventing real storms from gap-only native cells.

    Args:
        None.

    Returns:
        None after both quiet input forms give an empty retained digest tuple.

    Examples:
        Run ``pytest tests/test_footprint_order.py -k empty_and_gap_only_days``.
    """
    assert HistoricalFootprintOrder.for_day(DAY, (), _cells()) == ()
    assert HistoricalFootprintOrder.for_day(DAY, (), _cells(((0.0, 0.0, "gap"),))) == ()


def test_unsupported_overlay_runtime_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop allocation before an unverified GEOS release can change shape order.

    Args:
        monkeypatch: Pytest helper restoring the simulated runtime version.

    Returns:
        None after the unsupported runtime fails clearly.

    Examples:
        Run ``pytest tests/test_footprint_order.py -k unsupported_overlay_runtime``.
    """
    monkeypatch.setattr(shapely, "geos_version", (3, 14, 0))
    with pytest.raises(FootprintOrderError, match=r"requires GEOS 3\.13\.1"):
        HistoricalFootprintOrder.for_day(DAY, (), _cells())


def test_reading_digest_tuple_does_not_repeat_cell_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calculate once and retain the same tuple across later reads.

    Args:
        monkeypatch: Pytest helper replacing reconstruction after the initial result.

    Returns:
        None after cached access succeeds with reconstruction disabled.

    Examples:
        Run ``pytest tests/test_footprint_order.py -k reading_digest_tuple``.
    """
    cells = _cells(((0.0, 0.0, "qualified"),))
    order = HistoricalFootprintOrder(DAY, (_role(cells.frame.geometry.iloc[0]),), cells)
    expected = order.digests

    def unexpected_rebuild(*_args: object, **_kwargs: object) -> None:
        """Reject a repeated reconstruction after the order was already calculated.

        Args:
            *_args: Unexpected positional reconstruction arguments.
            **_kwargs: Unexpected named reconstruction arguments.

        Returns:
            None. Always raises so a repeated calculation fails this test.

        Raises:
            AssertionError: If digest access repeats native cell processing.

        Examples:
            unexpected_rebuild(cells)
        """
        raise AssertionError("Reading order digests repeated cell processing.")

    monkeypatch.setattr(_NativeRoleCells, "from_cells", unexpected_rebuild)
    assert order.digests is expected
    assert order.digests is expected


@pytest.mark.parametrize(
    ("values", "start", "end", "heap", "expected"),
    [
        (
            [1.0] * 17,
            0,
            17,
            False,
            [8, 16, 15, 14, 13, 12, 11, 10, 9, 0, 7, 6, 5, 4, 3, 2, 1],
        ),
        (
            [1.0] * 20,
            2,
            19,
            False,
            [0, 1, 10, 18, 17, 16, 15, 14, 13, 12, 11, 2, 9, 8, 7, 6, 5, 4, 3, 19],
        ),
        (
            [1.0] * 12,
            0,
            12,
            True,
            [8, 10, 4, 7, 11, 3, 9, 1, 5, 0, 2, 6],
        ),
        (
            [float(index % 4) for index in range(25)],
            0,
            25,
            False,
            [
                12,
                24,
                20,
                16,
                0,
                8,
                4,
                1,
                5,
                9,
                13,
                17,
                21,
                10,
                14,
                6,
                22,
                18,
                2,
                15,
                23,
                19,
                3,
                11,
                7,
            ],
        ),
    ],
)
def test_fixed_gnu_equal_value_order(
    values: list[float], start: int, end: int, heap: bool, expected: list[int]
) -> None:
    """Retain the original GNU equal-value moves verified by an independent control.

    Args:
        values: Values attached to labelled positions in the input list.
        start: First included position in the selected range.
        end: First excluded position in the selected range.
        heap: Whether to directly exercise the sorting depth safeguard.
        expected: Literal position sequence from the original GNU control.

    Returns:
        None after whole, selected and safeguarded ranges retain the exact ties.

    Examples:
        Run ``pytest tests/test_footprint_order.py -k fixed_gnu_equal_value_order``.
    """
    sorter = _HistoricalComparisonSort(list(range(len(values))), values)
    if heap:
        sorter._heap_sort(start, end)
    else:
        sorter.sort(start, end)
    assert sorter.order == expected


def test_changed_native_polygon_is_rejected_after_validation() -> None:
    """Recheck complete cell coverage if a caller later edits the retained frame.

    Args:
        None.

    Returns:
        None after a shortened cell polygon fails rather than issuing a digest.

    Examples:
        Run ``pytest tests/test_footprint_order.py -k changed_native_polygon``.
    """
    cells = _cells(((0.0, 0.0, "qualified"),))
    role = _role(cells.frame.geometry.iloc[0])
    cells.frame.loc[0, "geometry"] = box(-0.125, -0.125, 0.0, 0.125)
    with pytest.raises(FootprintOrderError, match="complete native rectangle"):
        HistoricalFootprintOrder.for_day(DAY, (role,), cells)


def test_repeated_native_cell_is_rejected_after_validation() -> None:
    """Reject cell rows repeated by a caller after the original table validation.

    Args:
        None.

    Returns:
        None after duplicating a cell prevents historical order calculation.

    Examples:
        Run ``pytest tests/test_footprint_order.py -k repeated_native_cell``.
    """
    cells = _cells(((0.0, 0.0, "qualified"), (5.0, 0.0, "qualified")))
    roles = tuple(_role(shape) for shape in cells.frame.geometry)
    cells.frame.loc[1, "longitude"] = 0.0
    cells.frame.loc[1, "geometry"] = cells.frame.geometry.iloc[0]
    with pytest.raises(FootprintOrderError, match="must not repeat a native cell half"):
        HistoricalFootprintOrder.for_day(DAY, roles, cells)
