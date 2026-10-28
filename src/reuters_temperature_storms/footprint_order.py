"""Recreate the published allocation order from finished native cell coverage.

Names published in the Reuters story depend on the original Linux ordering of
birth shapes. This module rebuilds that exact ordering input from the supplied
qualified, grace and complete coverage. It does not read a name list, catalog,
external file, input label or permanent wave ID.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import md5
from math import ceil, floor, sqrt
from typing import TYPE_CHECKING, cast

import numpy as np
import shapely
from shapely import Geometry
from shapely.affinity import translate

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from numpy.typing import NDArray

    from .cells import WaveCells
    from .components import DailyWaveComponentGeometry


class FootprintOrderError(ValueError):
    """Raised when finished coverage cannot reproduce the historical order."""


@dataclass(frozen=True, slots=True)
class HistoricalFootprintOrder:
    """Rebuild one whole day's published naming order from its exact native cells.

    All native cells matter, including gap-only groups later removed from the
    footprint table. Their position in the original polygon-join sequence can
    affect a retained dateline wave. Finished roles alone cannot supply that
    complete calculation context.

    Args:
        date: Calendar day as a NumPy datetime value at midnight.
        roles: Every finished footprint's qualified, grace and full coverage,
            in the order for which output digests are needed. Labels and scalar
            measurements do not enter the calculation.
        native_cells: Validated complete native cell table for the same day,
            including both dateline halves and any gap-only groups.

    Examples:
        order = HistoricalFootprintOrder(day, tuple(roles), cells)
        assert len(order.digests) == len(roles)
    """

    date: np.datetime64
    roles: tuple[DailyWaveComponentGeometry, ...]
    native_cells: WaveCells
    _digests: tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Calculate the whole day's digests once without changing any input.

        Args:
            None.

        Returns:
            None. Retains one immutable digest per supplied footprint.

        Raises:
            FootprintOrderError: If the runtime, date, cells or finished role
                coverage cannot reproduce the complete daily calculation.

        Examples:
            order = HistoricalFootprintOrder(day, tuple(roles), cells)
        """
        # The tested GEOS release determines the overlay's written coordinate
        # order. A different release must be checked before it can issue names.
        if shapely.geos_version != (3, 13, 1):
            raise FootprintOrderError("historical naming order requires GEOS 3.13.1")
        if not isinstance(self.date, np.datetime64) or np.isnat(self.date):
            raise FootprintOrderError("date must be a present NumPy calendar day")
        if self.date != self.date.astype("datetime64[D]"):
            raise FootprintOrderError(
                "date must represent midnight of its calendar day"
            )
        if not isinstance(self.roles, tuple):
            raise FootprintOrderError("roles must be a tuple containing the whole day")

        # Import the cell model only when calculation begins. Its input reader
        # also imports footprints, so a module-level import would create a loop.
        from .cells import WaveCells

        if not isinstance(self.native_cells, WaveCells):
            raise FootprintOrderError(
                "native_cells must be a validated complete WaveCells table"
            )
        dates = self.native_cells.frame["date"].to_numpy(dtype="datetime64[ns]")
        if not np.all(dates == self.date.astype("datetime64[ns]")):
            raise FootprintOrderError(
                "all native cell dates must match the requested day"
            )
        for role in self.roles:
            if not isinstance(role.time, np.datetime64) or role.time != self.date:
                raise FootprintOrderError(
                    "all finished footprint dates must match the requested day"
                )
            self._check_geometry("full", role.full_geometry, required=True)
            self._check_geometry("qualified", role.geometry, required=False)
            self._check_geometry("grace", role.grace_geometry, required=False)

        # Use exact daily cell membership rather than recovering only retained
        # footprints. The original global extent included gap-only cells too.
        # Rebuild their rectangle text so input bytes and row order cannot alter
        # the original sorting or the historical date-and-shape digest.
        cells = _NativeRoleCells.from_cells(self.native_cells)
        groups = (
            _RoleShapes.join_dateline(cells.role_groups()) if len(cells.shapes) else []
        )
        retained = [group for group in groups if group.has_real_cells]
        if len(retained) != len(self.roles):
            raise FootprintOrderError(
                "roles must contain every finished real-cell footprint"
            )
        full_shapes = [group.full for group in retained]
        tree = shapely.STRtree(full_shapes)
        day = str(np.datetime_as_string(self.date.astype("datetime64[D]"), unit="D"))
        digests = []
        used = set()
        for role in self.roles:
            candidates = [
                int(index)
                for index in tree.query(role.full_geometry)
                if shapely.equals(full_shapes[index], role.full_geometry)
            ]
            if len(candidates) != 1 or candidates[0] in used:
                raise FootprintOrderError(
                    "each supplied footprint must match one native-cell component"
                )
            selected = candidates[0]
            used.add(selected)
            rebuilt = retained[selected]
            # Whole-role equality checks reject partial cells, incomplete seam
            # halves and inconsistent companion data. This never normalizes or
            # replaces the source's geometry or physical measurements.
            self._require_same_coverage(
                "full", role.full_geometry, full_shapes[selected]
            )
            self._require_same_coverage("qualified", role.geometry, rebuilt.qualified)
            self._require_same_coverage("grace", role.grace_geometry, rebuilt.grace)
            value = md5(
                f"{day} {full_shapes[selected].wkt}".encode("ascii"),
                usedforsecurity=False,
            ).hexdigest()
            digests.append(value)
        object.__setattr__(self, "_digests", tuple(digests))

    @classmethod
    def for_day(
        cls,
        date: np.datetime64,
        roles: Sequence[DailyWaveComponentGeometry],
        native_cells: WaveCells,
    ) -> tuple[str, ...]:
        """Calculate one retained ordering digest for each complete daily role row.

        Args:
            date: Calendar day as a NumPy datetime value at midnight.
            roles: Every finished footprint for that day, in desired output order.
            native_cells: Complete validated native cell table, including gap-only
                cells absent from the finished footprint table.

        Returns:
            Full lowercase historical MD5 digests in the same order as roles.
            No input label, name, catalog or file lookup enters the calculation.

        Raises:
            FootprintOrderError: If the supplied daily calculation is incomplete
                or the supported geometry runtime is unavailable.

        Examples:
            digests = HistoricalFootprintOrder.for_day(day, roles, cells)
        """
        return cls(date, tuple(roles), native_cells).digests

    @property
    def digests(self) -> tuple[str, ...]:
        """Return the immutable original date-and-shape digests for the whole day.

        Args:
            None.

        Returns:
            One 32-character lowercase MD5 value per input footprint. Reading
            this tuple repeats no cell processing, polygon joins or file access.

        Examples:
            assert order.digests == order.digests
        """
        return self._digests

    @staticmethod
    def _check_geometry(
        label: str, geometry: Geometry | None, *, required: bool
    ) -> None:
        """Require finite geographic polygon coverage before comparing cell roles.

        Args:
            label: Role name included in a rejection message.
            geometry: Supplied shape, possibly missing for an optional role.
            required: Whether this role must have present coverage.

        Returns:
            None after the geographic and polygon checks pass.

        Raises:
            FootprintOrderError: If coverage is missing, broken or outside the grid.

        Examples:
            HistoricalFootprintOrder._check_geometry("full", full, required=True)
        """
        if geometry is None or (isinstance(geometry, Geometry) and geometry.is_empty):
            if required:
                raise FootprintOrderError(f"{label} must contain native cell coverage")
            return
        if not isinstance(geometry, Geometry):
            raise FootprintOrderError(f"{label} must be a Shapely polygon shape")
        if (
            geometry.geom_type not in {"Polygon", "MultiPolygon"}
            or not geometry.is_valid
        ):
            raise FootprintOrderError(f"{label} must contain valid polygon coverage")
        if (
            shapely.get_coordinate_dimension(geometry) != 2
            or not np.isfinite(shapely.get_coordinates(geometry)).all()
        ):
            raise FootprintOrderError(
                f"{label} must use finite two-dimensional coordinates"
            )
        min_x, min_y, max_x, max_y = geometry.bounds
        if min_x < -180 or max_x > 180 or min_y < -90 or max_y > 90:
            raise FootprintOrderError(
                f"{label} must stay within longitude/latitude bounds"
            )

    @staticmethod
    def _require_same_coverage(
        label: str, supplied: Geometry | None, rebuilt: Geometry | None
    ) -> None:
        """Require exact covered ground after reconstructing the complete daily cells.

        Args:
            label: Role name included in a rejection message.
            supplied: Original finished coverage, or a missing optional role.
            rebuilt: Coverage assembled from exact native cell membership.

        Returns:
            None when the original and rebuilt roles cover exactly the same ground.

        Raises:
            FootprintOrderError: If a role contains partial cells or different coverage.

        Examples:
            HistoricalFootprintOrder._require_same_coverage("grace", old, rebuilt)
        """
        supplied_present = supplied is not None and not supplied.is_empty
        rebuilt_present = rebuilt is not None and not rebuilt.is_empty
        if supplied_present != rebuilt_present or (
            supplied_present and not shapely.equals(supplied, rebuilt)
        ):
            raise FootprintOrderError(
                f"{label} must contain complete native cells with unchanged coverage"
            )


@dataclass(slots=True)
class _HistoricalComparisonSort:
    """Sort integer positions with the historical GNU treatment of equal values.

    Args:
        order: Mutable positions to rearrange. Each points into values.
        values: Numeric comparison values, such as each polygon's X centre.

    Examples:
        sorter = _HistoricalComparisonSort([0, 1, 2], [3.0, 1.0, 2.0])
        sorter.sort()
        assert sorter.order == [1, 2, 0]
    """

    order: list[int]
    values: list[float]

    def sort(self, start: int = 0, end: int | None = None) -> None:
        """Rearrange a selected range without adding an equal-value tie breaker.

        Args:
            start: First included position in order.
            end: First excluded position, or None for the complete remaining list.

        Returns:
            None. Updates order in place.

        Examples:
            _HistoricalComparisonSort([0, 1], [2.0, 1.0]).sort(0, 2)
        """
        stop = len(self.order) if end is None else end
        if not 0 <= start <= stop <= len(self.order):
            raise ValueError("The requested sorting range is outside the list.")
        size = stop - start
        if size < 2:
            return
        self._divide(start, stop, 2 * (size.bit_length() - 1))
        if size <= 16:
            self._insert_range(start, stop)
        else:
            # Finishing the first sixteen positions establishes the smallest
            # value at the start. Later insertions can then stop on that value.
            self._insert_range(start, start + 16)
            for position in range(start + 16, stop):
                self._insert_one(position)

    def _place_pivot(self, start: int, end: int) -> None:
        """Move the historical three-position choice into the first position.

        Args:
            start: First included position.
            end: First excluded position.

        Returns:
            None. Exchanges the chosen position with start.

        Examples:
            sorter._place_pivot(0, len(sorter.order))
        """
        left = start + 1
        middle = start + (end - start) // 2
        right = end - 1
        a, b, c = (self.values[self.order[i]] for i in (left, middle, right))
        if a < b:
            chosen = middle if b < c else (right if a < c else left)
        elif a < c:
            chosen = left
        elif b < c:
            chosen = right
        else:
            chosen = middle
        self.order[start], self.order[chosen] = self.order[chosen], self.order[start]

    def _split(self, start: int, end: int) -> int:
        """Exchange opposite out-of-place values and return the dividing position.

        Args:
            start: First included position, where the chosen value is stored.
            end: First excluded position.

        Returns:
            First position of the right-hand group.

        Examples:
            dividing_position = sorter._split(0, len(sorter.order))
        """
        self._place_pivot(start, end)
        order = self.order
        values = self.values
        pivot = values[order[start]]
        left, right = start + 1, end
        while True:
            while values[order[left]] < pivot:
                left += 1
            right -= 1
            while pivot < values[order[right]]:
                right -= 1
            if left >= right:
                return left
            # Equal values are exchanged too. Keeping this step is what gives
            # the old Linux build its particular ordering of equal-centre cells.
            order[left], order[right] = order[right], order[left]
            left += 1

    def _divide(self, start: int, end: int, remaining_depth: int) -> None:
        """Work on the right group first, then continue with the left group.

        Args:
            start: First included position.
            end: First excluded position.
            remaining_depth: Remaining allowed rounds before the heap safeguard.

        Returns:
            None. Leaves groups of at most sixteen for the final insertion pass.

        Examples:
            sorter._divide(0, len(sorter.order), 20)
        """
        while end - start > 16:
            if remaining_depth == 0:
                self._heap_sort(start, end)
                return
            remaining_depth -= 1
            cut = self._split(start, end)
            self._divide(cut, end, remaining_depth)
            end = cut

    def _insert_one(self, position: int) -> None:
        """Move one value left until the preceding value is no larger.

        Args:
            position: Position of the value being inserted into a prepared prefix.

        Returns:
            None. Shifts larger preceding values one place right.

        Examples:
            sorter._insert_one(2)
        """
        order = self.order
        values = self.values
        item = order[position]
        item_value = values[item]
        previous = position - 1
        while item_value < values[order[previous]]:
            order[position] = order[previous]
            position = previous
            previous -= 1
            if previous < 0:
                raise ValueError("The prepared prefix has no stopping value.")
        order[position] = item

    def _insert_range(self, start: int, end: int) -> None:
        """Finish a short range with the historical insertion procedure.

        Args:
            start: First included position.
            end: First excluded position.

        Returns:
            None. Sorts the selected range in place.

        Examples:
            sorter._insert_range(0, min(16, len(sorter.order)))
        """
        for position in range(start + 1, end):
            item = self.order[position]
            if self.values[item] < self.values[self.order[start]]:
                self.order[start + 1 : position + 1] = self.order[start:position]
                self.order[start] = item
            else:
                self._insert_one(position)

    def _adjust_heap(self, start: int, hole: int, length: int, item: int) -> None:
        """Fill a heap hole with larger children, choosing the right child on ties.

        Args:
            start: First position of the heap within order.
            hole: Relative position that needs a value.
            length: Number of positions in the heap.
            item: Original value to insert after lowering the hole.

        Returns:
            None. Restores the historical maximum-heap ordering.

        Examples:
            sorter._adjust_heap(0, 0, len(sorter.order), sorter.order[0])
        """
        order = self.order
        values = self.values
        top = hole
        child = hole
        while child < (length - 1) // 2:
            child = 2 * (child + 1)
            if values[order[start + child]] < values[order[start + child - 1]]:
                child -= 1
            order[start + hole] = order[start + child]
            hole = child
        if length % 2 == 0 and child == (length - 2) // 2:
            child = 2 * (child + 1)
            order[start + hole] = order[start + child - 1]
            hole = child - 1
        parent = (hole - 1) // 2
        while hole > top and values[order[start + parent]] < values[item]:
            order[start + hole] = order[start + parent]
            hole = parent
            parent = (hole - 1) // 2
        order[start + hole] = item

    def _heap_sort(self, start: int, end: int) -> None:
        """Apply the historical heap safeguard without changing its tie choices.

        Args:
            start: First included position.
            end: First excluded position.

        Returns:
            None. Sorts the range in place if repeated splitting reaches its limit.

        Examples:
            sorter._heap_sort(0, len(sorter.order))
        """
        length = end - start
        if length < 2:
            return
        for parent in range((length - 2) // 2, -1, -1):
            self._adjust_heap(start, parent, length, self.order[start + parent])
        while length > 1:
            length -= 1
            item = self.order[start + length]
            self.order[start + length] = self.order[start]
            self._adjust_heap(start, 0, length, item)


class _HistoricalPolygonUnion:
    """Join polygons with the original Linux leaf sorting and complete overlays.

    Args:
        None. Each join keeps its own polygon list and sorting positions.

    Examples:
        combined = _HistoricalPolygonUnion.join([west, east])
    """

    @classmethod
    def join(cls, geometries: Iterable[Geometry | None]) -> Geometry | None:
        """Join present polygon parts using the fixed historical calculation.

        Args:
            geometries: Polygon shapes in their supplied order; missing roles
                and empty shapes add no polygon parts.

        Returns:
            Their polygonal union, or None when the role has no present parts.

        Examples:
            qualified = _HistoricalPolygonUnion.join(qualified_cells)
        """
        polygons = [
            part
            for geometry in geometries
            if geometry is not None
            for part in cls.parts(geometry)
        ]
        if not polygons:
            return None
        order = list(range(len(polygons)))
        if len(polygons) > 1:
            bounds = shapely.bounds(polygons)
            horizontal = cast("list[float]", (bounds[:, 0] + bounds[:, 2]).tolist())
            vertical = cast("list[float]", (bounds[:, 1] + bounds[:, 3]).tolist())
            # The original tree grouped ten children per branch. Only its leaf
            # ordering enters the union: sort X centres, divide vertical strips,
            # then sort Y centres inside each strip with the fixed tie moves.
            strips = ceil(sqrt(ceil(len(polygons) / 10)))
            per_strip = ceil(len(polygons) / strips)
            _HistoricalComparisonSort(order, horizontal).sort()
            sorter = _HistoricalComparisonSort(order, vertical)
            for start in range(0, len(polygons), per_strip):
                sorter.sort(start, min(start + per_strip, len(polygons)))
        return cls._combine(polygons, order, 0, len(order))

    @classmethod
    def _combine(
        cls, polygons: list[Geometry], order: list[int], start: int, end: int
    ) -> Geometry:
        """Join a consecutive leaf range by splitting it at its midpoint.

        Args:
            polygons: Original polygon parts.
            order: Sorted positions into polygons.
            start: First included position in order.
            end: First excluded position in order.

        Returns:
            The joined polygonal coverage with the original operand sequence.

        Examples:
            shape = _HistoricalPolygonUnion._combine(polygons, order, 0, len(order))
        """
        if end - start == 1:
            return polygons[order[start]]
        if end - start == 2:
            left, right = polygons[order[start]], polygons[order[start + 1]]
        else:
            middle = (start + end) // 2
            left = cls._combine(polygons, order, start, middle)
            right = cls._combine(polygons, order, middle, end)
        # Ordinary binary union can copy separated pieces without computing an
        # overlay. The historical cascaded union always performed the complete
        # overlay. Explicit zero grid size selects that same operation while
        # preserving floating-point coordinates; it does not snap cells to a grid.
        return shapely.union(left, right, grid_size=0)

    @classmethod
    def parts(cls, geometry: Geometry) -> list[Geometry]:
        """Collect polygon pieces in stored order after union or world-map clipping.

        Args:
            geometry: Present polygon, multipart shape or clipping result.

        Returns:
            Polygon parts in their stored order. Edge-only lines and points add
            no coverage and are omitted.

        Examples:
            pieces = _HistoricalPolygonUnion.parts(clipped)
        """
        if geometry.is_empty:
            return []
        if geometry.geom_type == "Polygon":
            return [geometry]
        if geometry.geom_type in {"MultiPolygon", "GeometryCollection"}:
            return [part for member in geometry.geoms for part in cls.parts(member)]
        return []


@dataclass(frozen=True, slots=True)
class _NativeRoleCells:
    """Retain original rectangles and roles from the complete accepted cell table.

    Args:
        shapes: Canonical counterclockwise native rectangles in original input order.
        qualified: Whether each rectangle contains today's qualifying real cells.
        grace: Whether each rectangle contains real cells in their grace day.

    Examples:
        cells = _NativeRoleCells.from_cells(accepted_cells)
    """

    shapes: NDArray[np.object_]
    qualified: NDArray[np.bool_]
    grace: NDArray[np.bool_]

    @classmethod
    def from_cells(cls, native_cells: WaveCells) -> _NativeRoleCells:
        """Rebuild original rectangle bytes while retaining every accepted cell role.

        Args:
            native_cells: Complete, validated daily cell table with exact coordinates,
                original role counts and both display halves of a dateline cell.

        Returns:
            Rebuilt native rectangles sorted by longitude, latitude and rectangle
            bytes. All real, grace and gap-only membership stays unchanged.

        Raises:
            FootprintOrderError: If cell geometry or native coordinates changed
                after the caller's table was validated.

        Examples:
            cells = _NativeRoleCells.from_cells(accepted_cells)
        """
        frame = native_cells.frame
        longitude = frame["longitude"].to_numpy(dtype=float)
        latitude = frame["latitude"].to_numpy(dtype=float)
        if not (
            np.isfinite(longitude).all()
            and np.isfinite(latitude).all()
            and np.all((longitude >= -179.75) & (longitude <= 180))
            and np.all((latitude >= -90) & (latitude <= 90))
            and np.all(longitude * 4 == np.rint(longitude * 4))
            and np.all(latitude * 4 == np.rint(latitude * 4))
        ):
            raise FootprintOrderError(
                "cell centres must belong to the native quarter-degree grid"
            )
        supplied = frame.geometry.to_numpy()
        bounds = shapely.bounds(supplied)
        western_half = (longitude == 180) & (bounds[:, 0] == -180)
        minimum_x = longitude - 0.125
        maximum_x = np.minimum(longitude + 0.125, 180)
        minimum_x[western_half] = -180
        maximum_x[western_half] = -179.875
        # Box construction gives the original counterclockwise start and ring
        # order, independent of the accepted file's polygon encoding. Clipped
        # pole cells and both display halves retain their exact covered ground.
        shapes = shapely.box(
            minimum_x,
            np.maximum(latitude - 0.125, -90),
            maximum_x,
            np.minimum(latitude + 0.125, 90),
        )
        if not (
            np.all(shapely.get_coordinate_dimension(supplied) == 2)
            and np.all(shapely.is_valid(supplied))
            and np.all(shapely.equals(shapes, supplied))
        ):
            raise FootprintOrderError(
                "every cell must retain its complete native rectangle"
            )
        gap = frame["gap_filled"].to_numpy(dtype=bool)
        on_grace = (~gap) & frame["grace_days"].fillna(0).to_numpy(dtype=float).astype(
            bool
        )
        qualified = (~gap) & ~on_grace
        # Longitude and latitude place all daily cells in the original order.
        # Rectangle bytes resolve only the two halves with a shared +180 centre.
        rectangle_bytes = shapely.to_wkb(shapes)
        order = np.lexsort((rectangle_bytes, latitude, longitude))
        ordered_bytes = rectangle_bytes[order]
        if np.any(ordered_bytes[1:] == ordered_bytes[:-1]):
            raise FootprintOrderError(
                "the complete day must not repeat a native cell half"
            )
        return cls(shapes[order], qualified[order], on_grace[order])

    def role_groups(self) -> list[_RoleShapes]:
        """Combine each role separately within the original pre-dateline groups.

        Args:
            None.

        Returns:
            Qualified, grace and gap-fill shapes for each connected group, in
            the original union's part order.

        Raises:
            FootprintOrderError: If cell membership is missing or ambiguous.

        Examples:
            groups = cells.role_groups()
        """
        extent = _HistoricalPolygonUnion.join(self.shapes)
        if extent is None:
            raise FootprintOrderError("native cells must provide present coverage")
        parts = _HistoricalPolygonUnion.parts(extent)
        bounds = shapely.bounds(self.shapes)
        # A rectangle's inside midpoint has exactly one containing connected
        # group. Using inside points avoids choosing between shared boundaries.
        points = shapely.points(
            (bounds[:, 0] + bounds[:, 2]) / 2, (bounds[:, 1] + bounds[:, 3]) / 2
        )
        matches = shapely.STRtree(parts).query(points, predicate="within")
        if matches.shape[1] != len(self.shapes) or len(np.unique(matches[0])) != len(
            self.shapes
        ):
            raise FootprintOrderError(
                "every native cell must belong to one connected group"
            )
        groups = np.empty(len(self.shapes), dtype=np.int64)
        groups[matches[0]] = matches[1]
        gap = ~(self.qualified | self.grace)
        result = []
        for group in range(len(parts)):
            selected = groups == group
            result.append(
                _RoleShapes(
                    _HistoricalPolygonUnion.join(
                        self.shapes[selected & self.qualified]
                    ),
                    _HistoricalPolygonUnion.join(self.shapes[selected & self.grace]),
                    _HistoricalPolygonUnion.join(self.shapes[selected & gap]),
                )
            )
        return result


@dataclass(frozen=True, slots=True)
class _RoleShapes:
    """Retain one pre-dateline group's separately combined physical roles.

    Args:
        qualified: Qualified coverage, or None when absent.
        grace: Grace coverage, or None when absent.
        gap: Morphology-added gap-fill coverage, or None when absent.

    Examples:
        group = _RoleShapes(core, grace, gap)
    """

    qualified: Geometry | None
    grace: Geometry | None
    gap: Geometry | None

    @property
    def full(self) -> Geometry:
        """Combine qualified, then grace, then gap fill in the original order.

        Args:
            None.

        Returns:
            Full group coverage using the original binary extent operation.

        Raises:
            FootprintOrderError: If all roles are absent.

        Examples:
            complete = group.full
        """
        geometry = self._extend(self._extend(self.qualified, self.grace), self.gap)
        if geometry is None:
            raise FootprintOrderError("a native group must contain at least one role")
        return geometry

    @property
    def has_real_cells(self) -> bool:
        """Report whether qualified or grace coverage survives in this group.

        Args:
            None.

        Returns:
            True when the group contains real cells rather than only gap fill.

        Examples:
            assert _RoleShapes(core, None, None).has_real_cells
        """
        return any(
            shape is not None and not shape.is_empty
            for shape in (self.qualified, self.grace)
        )

    @staticmethod
    def _extend(first: Geometry | None, second: Geometry | None) -> Geometry | None:
        """Retain absent roles and combine present extents in their given order.

        Args:
            first: Earlier role or previously combined extent.
            second: Later role to add, possibly absent.

        Returns:
            Their ordinary binary union, the one present shape, or None.

        Examples:
            full = _RoleShapes._extend(qualified, grace)
        """
        if first is None or first.is_empty:
            return second if second is not None and not second.is_empty else first
        if second is None or second.is_empty:
            return first
        # Role combination used ordinary binary union in the original builder.
        # Its separated-piece shortcut must stay here, even though the role
        # dissolves above deliberately use complete overlays.
        return first.union(second)

    @staticmethod
    def _leader(parents: dict[int, int], component: int) -> int:
        """Follow a dateline group's links to its shared first representative.

        Args:
            parents: Mutable links between seam component indexes.
            component: Group index whose representative is needed.

        Returns:
            Representative index. Shortens visited links for later lookups.

        Examples:
            representative = _RoleShapes._leader({0: 1, 1: 1}, 0)
        """
        while parents[component] != component:
            parents[component] = parents[parents[component]]
            component = parents[component]
        return component

    @classmethod
    def join_dateline(cls, roles: list[_RoleShapes]) -> list[_RoleShapes]:
        """Join groups sharing real edges across the world-map cut.

        Args:
            roles: Separately combined roles in original pre-seam group order.

        Returns:
            Joined role groups in first-member order; unjoined groups pass through.

        Examples:
            joined = _RoleShapes.join_dateline(groups)
        """
        full = [group.full for group in roles]
        seam = [
            index
            for index, shape in enumerate(full)
            if shape.bounds[0] <= -180 or shape.bounds[2] >= 180
        ]
        parents = {index: index for index in seam}
        for position, left in enumerate(seam):
            for right in seam[position + 1 :]:
                # Opposite map edges are neighbours one world-width apart.
                # Require a shared edge; a corner touch does not join a wave.
                shared_edge = max(
                    full[left].intersection(translate(full[right], xoff=shift)).length
                    for shift in (-360.0, 360.0)
                )
                if shared_edge > 1e-6:
                    parents[cls._leader(parents, left)] = cls._leader(parents, right)
        connected: dict[int, list[int]] = {}
        for component in seam:
            connected.setdefault(cls._leader(parents, component), []).append(component)
        by_first = {
            members[0]: members for members in connected.values() if len(members) > 1
        }
        absorbed = {member for members in by_first.values() for member in members[1:]}
        groups = [
            by_first.get(component, [component])
            for component in range(len(roles))
            if component not in absorbed
        ]
        result = []
        for members in groups:
            if len(members) == 1:
                result.append(roles[members[0]])
                continue
            # Shift only a western-only component. A component already spanning
            # both display edges must not move its eastern coverage as well.
            shifts = [
                360.0
                if full[member].bounds[0] <= -180 and full[member].bounds[2] < 180
                else 0.0
                for member in members
            ]
            result.append(
                cls(
                    cls._stitch(
                        [roles[member].qualified for member in members], shifts
                    ),
                    cls._stitch([roles[member].grace for member in members], shifts),
                    cls._stitch([roles[member].gap for member in members], shifts),
                )
            )
        return result

    @classmethod
    def _stitch(
        cls, geometries: list[Geometry | None], shifts: list[float]
    ) -> Geometry | None:
        """Move each present role beside its dateline neighbours and join it.

        Args:
            geometries: One role for each joined seam member.
            shifts: Matching longitude offsets selected from the full extents.

        Returns:
            Joined role coverage folded into [-180, 180], or None when absent.

        Examples:
            stitched = _RoleShapes._stitch([west, east], [360.0, 0.0])
        """
        unfolded = [
            translate(shape, xoff=shift)
            for shape, shift in zip(geometries, shifts, strict=True)
            if shape is not None and not shape.is_empty
        ]
        combined = _HistoricalPolygonUnion.join(unfolded)
        return None if combined is None else cls._wrap(combined)

    @staticmethod
    def _wrap(geometry: Geometry) -> Geometry:
        """Clip occupied world-width strips and fold their polygons onto the map.

        Args:
            geometry: Joined role, potentially extending beyond +180 longitude.

        Returns:
            In-range polygonal coverage joined in its original strip order.

        Examples:
            wrapped = _RoleShapes._wrap(unfolded)
        """
        minimum, _, maximum, _ = geometry.bounds
        if minimum >= -180 and maximum <= 180:
            return geometry
        first = floor((minimum + 180) / 360)
        last = floor((maximum + 180) / 360)
        polygons = []
        for strip in range(first, last + 1):
            clipped = geometry.intersection(
                shapely.box(-180 + 360 * strip, -90, 180 + 360 * strip, 90)
            )
            polygons.extend(
                translate(part, xoff=-360 * strip)
                for part in _HistoricalPolygonUnion.parts(clipped)
            )
        result = _HistoricalPolygonUnion.join(polygons)
        return geometry if result is None else result
