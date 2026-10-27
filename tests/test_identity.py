"""Date-and-cell identities independent of polygon serialization."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Polygon, box

from reuters_temperature_storms.identity import (
    FootprintIdentity,
    IdentityError,
    birth_wave_identity,
    component_identity,
)


@pytest.mark.unit
@pytest.mark.parametrize("temperature", ["tasmax", "tasmin"])
@pytest.mark.parametrize("direction", ["warm", "cold"])
def test_date_and_cell_identity(temperature, direction) -> None:
    """Use the date and first occupied cell in each independent series.

    Args:
        temperature: Temperature field to validate.
        direction: Warm or cold direction to validate.

    Returns:
        None.

    Steps:
        1. Check the cell address and complete permanent ID.

    Examples:
        Run ``pytest tests/test_identity.py -k date_and_cell_identity``.
    """
    geometry = box(-0.125, -0.125, 0.125, 0.125)
    identity = FootprintIdentity(np.datetime64("2024-01-01"), geometry)
    assert identity.cell_id == 519119
    expected = identity.digest
    daily_id = component_identity(
        temperature=temperature,
        direction=direction,
        date=np.datetime64("2024-01-01T12:30"),
        geometry=geometry,
    )
    assert daily_id == expected
    assert (
        birth_wave_identity(
            temperature=temperature,
            direction=direction,
            date=np.datetime64("2024-01-01"),
            component_id=daily_id,
        )
        == f"20240101-{expected}"
    )


@pytest.mark.unit
def test_ring_order_is_ignored() -> None:
    """Equivalent rotated rings must produce the same cell-based identity.

    Returns:
        None.

    Steps:
        1. Equivalent rotated rings must select the same occupied cell.

    Examples:
        Run ``pytest tests/test_identity.py -k ring_order_is_ignored``.
    """
    geometry = box(-0.125, -0.125, 0.125, 0.125)
    ring = list(geometry.exterior.coords)[:-1]
    rotated = Polygon(ring[1:] + ring[:1])
    assert rotated.equals(geometry)
    assert component_identity(
        temperature="tasmax",
        direction="warm",
        date=np.datetime64("2024-01-01"),
        geometry=geometry,
    ) == component_identity(
        temperature="tasmax",
        direction="warm",
        date=np.datetime64("2024-01-01"),
        geometry=rotated,
    )


@pytest.mark.unit
def test_day_changes_daily_hash() -> None:
    """Identical footprints on consecutive dates have distinct daily hashes.

    Returns:
        None.

    Steps:
        1. Identical footprints on consecutive dates have distinct daily hashes.

    Examples:
        Run ``pytest tests/test_identity.py -k day_changes_daily_hash``.
    """
    geometry = box(0, 0, 0.25, 0.25)
    hashes = {
        component_identity(
            temperature="tasmin",
            direction="cold",
            date=np.datetime64(date),
            geometry=geometry,
        )
        for date in ["1960-01-05", "1960-01-06"]
    }
    assert len(hashes) == 2


@pytest.mark.unit
@pytest.mark.parametrize(
    "geometry", [Polygon(), None, Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])]
)
def test_invalid_footprint_rejected(geometry) -> None:
    """Reject invalid geometry instead of deriving a usable wave key.

    Args:
        geometry: Geographic polygon whose equal-area measurement is requested.

    Returns:
        None.

    Steps:
        1. Reject invalid geometry instead of deriving a usable wave key.

    Examples:
        Run ``pytest tests/test_identity.py -k invalid_footprint_rejected``.
    """
    with pytest.raises(IdentityError, match="footprint"):
        component_identity(
            temperature="tasmax",
            direction="warm",
            date=np.datetime64("2024-01-01"),
            geometry=geometry,
        )


@pytest.mark.unit
@pytest.mark.parametrize("digest", ["ABC", "a" * 64, "A" * 32, "g" * 32])
def test_invalid_component_digest_is_rejected(digest: str) -> None:
    """Only complete lowercase daily IDs can become birth keys.

    Args:
        digest: Candidate component hash to validate before constructing a wave key.

    Returns:
        None.

    Steps:
        1. Reject shortened or malformed daily IDs.

    Examples:
        Run ``pytest tests/test_identity.py -k invalid_component_digest_is_rejected``.
    """
    with pytest.raises(IdentityError, match="lowercase"):
        birth_wave_identity(
            temperature="tasmax",
            direction="warm",
            date=np.datetime64("2024-01-01"),
            component_id=digest,
        )


@pytest.mark.unit
def test_shape_changes_outside_first_cell_do_not_change_identity() -> None:
    """Ignore the rest of the shape when its first occupied cell stays fixed.

    Args:
        None.

    Returns:
        None.

    Examples:
        Run ``pytest tests/test_identity.py -k shape_changes_outside_first_cell``.
    """
    day = np.datetime64("2024-01-01")
    first = FootprintIdentity(day, box(-0.125, -0.125, 0.125, 0.125))
    larger = FootprintIdentity(day, box(-0.125, -0.125, 0.875, 0.625))
    assert not first.geometry.equals(larger.geometry)
    assert first.cell_id == larger.cell_id
    assert first.digest == larger.digest


@pytest.mark.unit
def test_cell_order_is_latitude_then_longitude() -> None:
    """Select an occupied cell rather than an empty bounding-box corner.

    Args:
        None.

    Returns:
        None.

    Examples:
        Run ``pytest tests/test_identity.py -k cell_order``.
    """
    eastern = box(1.875, -0.125, 2.125, 0.125)
    northern = box(-0.125, 0.125, 0.125, 0.375)
    day = np.datetime64("2024-01-01")
    assert FootprintIdentity(day, eastern).cell_id == 519127
    assert FootprintIdentity(day, MultiPolygon([northern, eastern])).cell_id == 519127
    assert FootprintIdentity(day, MultiPolygon([eastern, northern])).cell_id == 519127


@pytest.mark.unit
def test_dateline_halves_have_one_address() -> None:
    """Count the paired +180 cell once in the native grid ordering.

    Args:
        None.

    Returns:
        None.

    Examples:
        Run ``pytest tests/test_identity.py -k dateline_halves``.
    """
    eastern = box(179.875, -0.125, 180, 0.125)
    western = box(-180, -0.125, -179.875, 0.125)
    day = np.datetime64("2024-01-01")
    identity = FootprintIdentity(day, MultiPolygon([eastern, western]))
    assert identity.cell_id == 519839
    assert (
        identity.digest
        == FootprintIdentity(day, MultiPolygon([western, eastern])).digest
    )
    for half in (eastern, western):
        with pytest.raises(IdentityError, match="both dateline cell halves"):
            _ = FootprintIdentity(day, half).digest


@pytest.mark.unit
@pytest.mark.parametrize(
    ("geometry", "expected"),
    [
        (box(-179.875, -90, -179.625, -89.875), 0),
        (box(-0.125, 89.875, 0.125, 90), 1037519),
    ],
)
def test_clipped_polar_cells(geometry, expected) -> None:
    """Include native cell centres on the geographic polar boundaries.

    Args:
        geometry: Complete clipped quarter-degree cell.
        expected: Its address on the native Reuters grid.

    Returns:
        None.

    Examples:
        Run ``pytest tests/test_identity.py -k clipped_polar_cells``.
    """
    assert FootprintIdentity(np.datetime64("2024-01-01"), geometry).cell_id == expected


@pytest.mark.unit
@pytest.mark.parametrize("geometry", [box(0.01, 0.01, 0.02, 0.02), box(180, 0, 181, 1)])
def test_unsupported_grid_coverage_rejected(geometry) -> None:
    """Reject footprints lacking a grid centre or outside geographic bounds.

    Args:
        geometry: Candidate polygon with unsupported grid coverage.

    Returns:
        None.

    Examples:
        Run ``pytest tests/test_identity.py -k unsupported_grid_coverage``.
    """
    with pytest.raises(IdentityError, match="footprint"):
        _ = FootprintIdentity(np.datetime64("2024-01-01"), geometry).digest
