"""Daily footprint hashes and permanent wave IDs."""

from __future__ import annotations

from hashlib import md5

import numpy as np
from shapely import Geometry

from .series import Direction, Temperature, WaveSeries


class IdentityError(ValueError):
    """Raised when canonical wave identity input is invalid."""


def component_identity(
    *,
    temperature: Temperature,
    direction: Direction,
    date: np.datetime64,
    geometry: Geometry,
) -> str:
    """Return the daily footprint hash from the date and unchanged shape text.

    The hash covers the complete qualified, grace, and gap-filled extent.
    Ring starts and multipart order are significant; geometry normalization
    would change the daily ID. Series are validated independently
    and are not part of the hash input.

    Steps:
        1. Check the series, calendar date and nonempty polygon.
        2. Write the date followed by the complete shape's unchanged coordinate text.
        3. Hash those bytes into the 32-character daily ID; coordinate order is part of the input.

    Args:
        temperature: ``"tasmax"`` or ``"tasmin"``.
        direction: ``"warm"`` or ``"cold"``.
        date: The footprint's calendar day.
        geometry: Complete footprint with its original coordinate order in EPSG:4326.

    Returns:
        The 32-character lowercase MD5 daily footprint hash.

    Raises:
        IdentityError: If the series, date, or geometry is invalid.

    Examples:
        from shapely.geometry import box

        daily_id = component_identity(
            temperature="tasmax",
            direction="warm",
            date=np.datetime64("2024-01-01"),
            geometry=box(-0.125, -0.125, 0.125, 0.125),
        )
    """
    _validate_series(temperature, direction)
    _days_since_epoch(date, "date")
    if not isinstance(geometry, Geometry) or geometry.is_empty:
        msg = "geometry must be a nonempty Shapely footprint"
        raise IdentityError(msg)
    if geometry.geom_type not in {"Polygon", "MultiPolygon"} or not geometry.is_valid:
        msg = "geometry must be a valid polygonal footprint"
        raise IdentityError(msg)
    day = np.datetime_as_string(date.astype("datetime64[D]"), unit="D")
    return md5(f"{day} {geometry.wkt}".encode(), usedforsecurity=False).hexdigest()


def birth_wave_identity(
    *,
    temperature: Temperature,
    direction: Direction,
    date: np.datetime64,
    component_id: str,
) -> str:
    """Return the permanent ID for a wave born from a daily footprint.

    Steps:
        1. Check the series, birth date and 32-character daily footprint hash.
        2. Remove separators from the birth date and append the first eight characters of that hash.

    Args:
        temperature: ``"tasmax"`` or ``"tasmin"``.
        direction: ``"warm"`` or ``"cold"``.
        date: The wave birth day.
        component_id: A 32-character lowercase daily footprint MD5 hash.

    Returns:
        The permanent ID in ``YYYYMMDD-first8hash`` form. Series remain
        separate even when they have identical geometry and dates.

    Raises:
        IdentityError: If the series, date, or footprint hash is invalid.

    Examples:
        >>> birth_wave_identity(
        ...     temperature="tasmax",
        ...     direction="warm",
        ...     date=np.datetime64("2024-01-01"),
        ...     component_id="0123456789abcdef0123456789abcdef",
        ... )
        '20240101-01234567'
    """
    _validate_series(temperature, direction)
    _days_since_epoch(date, "date")
    if (
        not isinstance(component_id, str)
        or len(component_id) != 32
        or any(character not in "0123456789abcdef" for character in component_id)
    ):
        msg = "component_id must be a 32-character lowercase hexadecimal MD5 hash"
        raise IdentityError(msg)
    day = np.datetime_as_string(date.astype("datetime64[D]"), unit="D")
    return f"{day.replace('-', '')}-{component_id[:8]}"


def _validate_series(temperature: Temperature, direction: Direction) -> None:
    """Check identity's series fields against the shared model.

    Steps:
        1. Check the shared temperature/direction model.
        2. Report unsupported pairs through the module validation error.

    Args:
        temperature: Candidate maximum/minimum temperature field.
        direction: Candidate warm/cold direction.

    Returns:
        None.

    Raises:
        IdentityError: If the pair is unsupported.

    Examples:
        _validate_series("tasmax", "warm")
    """
    try:
        WaveSeries(temperature, direction).validate()
    except ValueError as error:
        raise IdentityError(
            "temperature and direction must be one of the four Reuters branches"
        ) from error


def _days_since_epoch(value: np.datetime64, label: str) -> int:
    """Convert a validated NumPy date to an epoch day count.

    Steps:
        1. Reject non-datetime and missing values.
        2. Convert to whole days and check the signed 32-bit range.

    Args:
        value: Candidate NumPy calendar date.
        label: Field name used in validation errors.

    Returns:
        Signed integer count of days since 1970-01-01.

    Examples:
        _days_since_epoch(np.datetime64("1970-01-02"), "date")
    """
    if not isinstance(value, np.datetime64) or np.isnat(value):
        msg = f"{label} must be a valid numpy datetime64"
        raise IdentityError(msg)
    days = int(value.astype("datetime64[D]").astype("int64"))
    if not -(2**31) <= days < 2**31:
        msg = f"{label} must fit in a signed 32-bit day count"
        raise IdentityError(msg)
    return days
