"""Footprint-hash and wave-key tests."""

from __future__ import annotations

from hashlib import md5

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from reuters_temperature_storms.identity import (
    IdentityError,
    birth_wave_identity,
    component_identity,
)


@pytest.mark.unit
@pytest.mark.parametrize("temperature", ["tasmax", "tasmin"])
@pytest.mark.parametrize("direction", ["warm", "cold"])
def test_date_wkt_hash_and_birth_key(temperature, direction) -> None:
    """Use the date/WKT bytes in each independent series.

    Args:
        temperature: Temperature field to validate.
        direction: Warm or cold direction to validate.

    Returns:
        None.

    Steps:
        1. Use the date/WKT bytes in each independent series.

    Examples:
        Run ``pytest tests/test_identity.py -k date_wkt_hash_and_birth_key``.
    """
    geometry = box(-0.125, -0.125, 0.125, 0.125)
    expected = md5(
        f"2024-01-01 {geometry.wkt}".encode(), usedforsecurity=False
    ).hexdigest()
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
        == f"20240101-{expected[:8]}"
    )


@pytest.mark.unit
def test_raw_ring_order_remains_significant() -> None:
    """Equivalent rotated rings must keep their distinct geometry hashes.

    Returns:
        None.

    Steps:
        1. Equivalent rotated rings must keep their distinct geometry hashes.

    Examples:
        Run ``pytest tests/test_identity.py -k raw_ring_order_remains_significant``.
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
    ) != component_identity(
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
    """Only lowercase MD5 geometry hashes can become birth keys.

    Args:
        digest: Candidate component hash to validate before constructing a wave key.

    Returns:
        None.

    Steps:
        1. Only lowercase MD5 geometry hashes can become birth keys.

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
