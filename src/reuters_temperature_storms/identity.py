"""Daily footprint identities from dates and occupied quarter-degree cells."""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from math import ceil, floor

import numpy as np
import shapely
from shapely import Geometry

from .series import Direction, Temperature, WaveSeries

WAVE_ID_PATTERN = re.compile(r"\d{8}-[0-9a-f]{32}")
_LONGITUDE_COUNT = 1440
_LATITUDE_COUNT = 721


class IdentityError(ValueError):
    """Raised when canonical wave identity input is invalid."""


@dataclass(frozen=True, slots=True)
class FootprintIdentity:
    """Identify a daily component by its first occupied quarter-degree cell.

    Finished footprints contain complete cells on the Reuters grid. Longitude
    runs from -179.75 to +180 and latitude from -90 to +90. The two display
    halves of a +180 cell have one address. Cells are ordered by latitude, then
    longitude; separate components on the same day cannot own the same cell.

    Args:
        date: Calendar date represented by the footprint.
        geometry: Complete two-dimensional EPSG:4326 grid-cell coverage, including
            qualified, grace and gap-filled cells.

    Examples:
        identity = FootprintIdentity(np.datetime64("2024-01-01"), geometry)
        daily_id = identity.digest
    """

    #: Calendar day represented by the complete component.
    date: np.datetime64
    #: Finished coverage of the component's physical quarter-degree cells.
    geometry: Geometry

    @property
    def cell_id(self) -> int:
        """Return the first occupied cell address without encoding the polygon.

        Args:
            None.

        Returns:
            Latitude-row times 1,440 plus longitude-column, starting at zero.
            Longitude columns start at -179.75; latitude rows start at -90.

        Raises:
            IdentityError: If geometry is invalid, outside geographic bounds,
                lacks a grid centre, or has unmatched dateline cell halves.

        Examples:
            assert FootprintIdentity(day, box(-0.125, -0.125, 0.125, 0.125)).cell_id == 519119
        """
        # A missing or broken shape cannot tell us which cells belong to the
        # footprint. Check the shape before looking for an occupied cell.
        geometry = self.geometry
        if not isinstance(geometry, Geometry) or geometry.is_empty:
            raise IdentityError("geometry must be a nonempty Shapely footprint")
        if (
            geometry.geom_type not in {"Polygon", "MultiPolygon"}
            or not geometry.is_valid
        ):
            raise IdentityError("geometry must be a valid polygonal footprint")

        # This grid uses longitude and latitude only. Reject height values and
        # missing or infinite numbers, which would make the lookup unreliable.
        if (
            shapely.get_coordinate_dimension(geometry) != 2
            or not np.isfinite(shapely.get_coordinates(geometry)).all()
        ):
            raise IdentityError("geometry must use finite two-dimensional coordinates")

        # Read the shape's outer limits to narrow the search. Those limits must
        # stay inside the world's longitude and latitude ranges.
        min_x, min_y, max_x, max_y = geometry.bounds
        if min_x < -180 or max_x > 180 or min_y < -90 or max_y > 90:
            raise IdentityError("footprint must stay inside longitude/latitude bounds")

        # Grid centres are a quarter degree apart, starting at latitude -90 and
        # longitude -179.75. Multiplying each offset by four gives its row or
        # column number. Round inward to skip centres outside the shape's outer
        # limits, and keep the resulting numbers inside the stored grid.
        first_row = max(0, ceil((min_y + 90) * 4))
        last_row = min(_LATITUDE_COUNT - 1, floor((max_y + 90) * 4))
        first_column = max(0, ceil((min_x + 179.75) * 4))
        last_column = min(_LONGITUDE_COUNT - 1, floor((max_x + 179.75) * 4))

        # The cell centred on +180 is drawn in two halves, one at each edge of
        # the map. Check both edges at the same latitudes: every occupied half
        # must have its partner so they still represent one complete cell.
        if min_x == -180 or max_x == 180:
            latitudes = np.arange(first_row, last_row + 1) / 4 - 90
            western = shapely.intersects_xy(geometry, -180, latitudes)
            eastern = shapely.intersects_xy(geometry, 180, latitudes)
            if not np.array_equal(western, eastern):
                raise IdentityError("footprint must retain both dateline cell halves")

        # Build the possible longitudes once. Every latitude row uses the same
        # columns, so the search can reuse them as it moves north.
        columns = np.arange(first_column, last_column + 1)
        longitudes = columns / 4 - 179.75

        # Search from south to north, then west to east. Testing cell centres
        # makes the first choice independent of the polygon's coordinate order.
        # Separate finished footprints cannot own the same occupied grid cell.
        for row in range(first_row, last_row + 1):
            occupied = np.flatnonzero(
                shapely.intersects_xy(geometry, longitudes, row / 4 - 90)
            )
            if occupied.size:
                # Give each row 1,440 addresses, then add the column number.
                # This turns the chosen row and column into one integer ID.
                return row * _LONGITUDE_COUNT + int(columns[occupied[0]])

        # A finished footprint must contain at least one grid cell centre.
        # Reaching the end means this shape cannot provide a usable address.
        raise IdentityError("footprint must contain a quarter-degree grid cell centre")

    @property
    def digest(self) -> str:
        """Hash the calendar date and the first occupied cell address.

        Args:
            None.

        Returns:
            The first 128 bits of SHA-256 as 32 lowercase hexadecimal characters.
            Polygon coordinates, bytes, ring order and measurements are absent
            from the hash input.

        Examples:
            assert len(FootprintIdentity(day, geometry).digest) == 32
        """
        _days_since_epoch(self.date, "date")
        day = np.datetime_as_string(self.date.astype("datetime64[D]"), unit="D")
        return sha256(f"{day} {self.cell_id}".encode("ascii")).hexdigest()[:32]


def component_identity(
    *,
    temperature: Temperature,
    direction: Direction,
    date: np.datetime64,
    geometry: Geometry,
) -> str:
    """Return a daily identity from the date and first occupied grid cell.

    This uses a cell address rather than a shape fingerprint. Boundary encoding
    and the rest of the shape do not affect the ID when the first cell is the
    same. Series remain independent and are not part of the hash input.

    Steps:
        1. Check the series, calendar date and finished grid-cell footprint.
        2. Find its first occupied cell in latitude/longitude order.
        3. Hash only the date and integer cell address into a 32-character ID.

    Args:
        temperature: ``"tasmax"`` or ``"tasmin"``.
        direction: ``"warm"`` or ``"cold"``.
        date: The footprint's calendar day.
        geometry: Finished two-dimensional quarter-degree coverage in EPSG:4326.

    Returns:
        The 32-character lowercase date-and-cell digest.

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
    return FootprintIdentity(date, geometry).digest


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
        2. Remove separators from the birth date and append its complete digest.

    Args:
        temperature: ``"tasmax"`` or ``"tasmin"``.
        direction: ``"warm"`` or ``"cold"``.
        date: The wave birth day.
        component_id: A 32-character lowercase daily footprint digest.

    Returns:
        The permanent ID in ``YYYYMMDD-digest`` form. Series remain
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
        '20240101-0123456789abcdef0123456789abcdef'
    """
    _validate_series(temperature, direction)
    _days_since_epoch(date, "date")
    if (
        not isinstance(component_id, str)
        or len(component_id) != 32
        or any(character not in "0123456789abcdef" for character in component_id)
    ):
        msg = "component_id must be a 32-character lowercase hexadecimal digest"
        raise IdentityError(msg)
    day = np.datetime_as_string(date.astype("datetime64[D]"), unit="D")
    return f"{day.replace('-', '')}-{component_id}"


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
