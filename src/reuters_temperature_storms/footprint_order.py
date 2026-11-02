"""Calculate fixed footprint ordering codes from complete grid-cell coverage.

This module constructs grid-cell rectangles, sorts and joins their coverage in
an explicit sequence, and hashes each date and resulting shape description.
The codes order tracking and naming decisions independently of permanent wave
IDs. Qualified, grace and gap-filled cells all contribute to the calculation.
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
    """Raised when coverage or runtime settings cannot produce footprint ordering codes."""


@dataclass(frozen=True, slots=True)
class HistoricalFootprintOrder:
    """Calculate one whole day's footprint ordering codes from its grid cells.

    A footprint is the ground covered by one connected group of grid cells.
    Qualified cells meet the selection rules today. Grace cells remain after
    a brief break in those conditions. Gap-filled cells connect nearby coverage.
    All three kinds affect how the combined shape is written and ordered.

    Groups containing only gap-filled cells still take part in the joining
    calculation, even though they are omitted from the final footprint list.
    Shapes crossing the left and right edges of the map also need both halves.

    Args:
        date: Calendar day as a NumPy datetime value at midnight.
        roles: Every finished footprint's qualified, grace and full coverage,
            in the order for which output codes are needed. Labels and numeric
            measurements do not enter the calculation.
        native_cells: Validated complete grid-cell table for the same day,
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
        """Calculate the whole day's ordering codes once without changing any input.

        Args:
            None.

        Returns:
            None. Saves one unchanging ordering code per supplied footprint.

        Raises:
            FootprintOrderError: If the runtime, date, cells or finished role
                coverage cannot reproduce the complete daily calculation.

        Examples:
            order = HistoricalFootprintOrder(day, tuple(roles), cells)
        """
        # Joining shapes can list their corners differently across software
        # releases. Require the supported version because that written order
        # affects the code we calculate below, even for equal covered ground.
        if shapely.geos_version != (3, 13, 1):
            raise FootprintOrderError("historical naming order requires GEOS 3.13.1")

        # One calculation represents one whole calendar day. Reject missing
        # dates and times within a day so the date text has only one meaning.
        if not isinstance(self.date, np.datetime64) or np.isnat(self.date):
            raise FootprintOrderError("date must be a present NumPy calendar day")
        if self.date != self.date.astype("datetime64[D]"):
            raise FootprintOrderError(
                "date must represent midnight of its calendar day"
            )
        if not isinstance(self.roles, tuple):
            raise FootprintOrderError("roles must be a tuple containing the whole day")

        # Load the cell class here because its own imports lead back to this
        # module. Loading it at the top would leave both modules half loaded.
        from .cells import WaveCells

        # The cells and supplied footprints must describe the same day. Mixing
        # days could produce a plausible shape with an incorrect ordering code.
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

        # Rebuild from every grid cell, including groups made only of gap fill.
        # Keeping those groups through the joining step gives it the complete
        # input. Only afterward remove groups that cannot become footprints.
        cells = _NativeRoleCells.from_cells(self.native_cells)
        groups = (
            _RoleShapes.join_dateline(cells.role_groups()) if len(cells.shapes) else []
        )
        retained = [group for group in groups if group.has_real_cells]
        # Require the complete footprint list, rather than assigning codes to
        # a partial day and silently overlooking an unmatched rebuilt group.
        if len(retained) != len(self.roles):
            raise FootprintOrderError(
                "roles must contain every finished real-cell footprint"
            )
        # Build a searchable list of rebuilt shapes. Nearby shapes are only
        # candidates: the loop still checks that they cover exactly the same
        # ground. Format the date once so every code uses the same date text.
        full_shapes = [group.full for group in retained]
        tree = shapely.STRtree(full_shapes)
        day = str(np.datetime_as_string(self.date.astype("datetime64[D]"), unit="D"))
        digests = []
        used = set()
        for role in self.roles:
            # Match by covered ground, not file labels or corner-list order.
            # Each supplied footprint must match exactly one unused group.
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
            # Check the kinds of coverage separately too: equal total outlines
            # could still hide a qualified cell incorrectly marked as grace.
            # These checks do not replace the supplied shapes or measurements.
            self._require_same_coverage(
                "full", role.full_geometry, full_shapes[selected]
            )
            self._require_same_coverage("qualified", role.geometry, rebuilt.qualified)
            self._require_same_coverage("grace", role.grace_geometry, rebuilt.grace)
            # Turn the date and fixed written shape into a 32-character code.
            # Append in the caller's footprint order so each returned code
            # belongs to the corresponding supplied footprint. MD5 is used
            # for repeatable ordering here, not for protecting secret data.
            value = md5(
                f"{day} {full_shapes[selected].wkt}".encode("ascii"),
                usedforsecurity=False,
            ).hexdigest()
            digests.append(value)
        # Save the completed codes once. The class prevents later assignment;
        # this initialization writes its result without changing any inputs.
        object.__setattr__(self, "_digests", tuple(digests))

    @classmethod
    def for_day(
        cls,
        date: np.datetime64,
        roles: Sequence[DailyWaveComponentGeometry],
        native_cells: WaveCells,
    ) -> tuple[str, ...]:
        """Calculate one ordering code for each supplied daily footprint.

        Args:
            date: Calendar day as a NumPy datetime value at midnight.
            roles: Every finished footprint for that day, in desired output order.
            native_cells: Complete validated grid-cell table, including gap-only
                cells absent from the finished footprint table.

        Returns:
            32-character lowercase codes calculated with MD5 from dates and
            shape descriptions, in the same order as roles.

        Raises:
            FootprintOrderError: If the supplied daily calculation is incomplete
                or the supported geometry runtime is unavailable.

        Examples:
            digests = HistoricalFootprintOrder.for_day(day, roles, cells)
        """
        # Accept any sequence of footprints, but keep its order in a tuple.
        # Construction performs all checks and calculation before returning.
        return cls(date, tuple(roles), native_cells).digests

    @property
    def digests(self) -> tuple[str, ...]:
        """Return the saved date-and-shape ordering codes for the whole day.

        Args:
            None.

        Returns:
            One 32-character lowercase MD5 value per input footprint. Reading
            this tuple repeats no cell processing, polygon joins or file access.

        Examples:
            assert order.digests == order.digests
        """
        # Return the saved result; repeated reads must not rebuild the shapes.
        return self._digests

    @staticmethod
    def _check_geometry(
        label: str, geometry: Geometry | None, *, required: bool
    ) -> None:
        """Check that supplied shapes describe valid ground within the world map.

        Args:
            label: Role name included in a rejection message.
            geometry: Supplied shape, possibly missing for an optional role.
            required: Whether this role must have present coverage.

        Returns:
            None after checking the shape, its coordinates and its map bounds.

        Raises:
            FootprintOrderError: If coverage is missing, broken or outside the grid.

        Examples:
            HistoricalFootprintOrder._check_geometry("full", full, required=True)
        """
        # Qualified or grace coverage can be absent. Full coverage cannot:
        # there would be no footprint to match or write into an ordering code.
        if geometry is None or (isinstance(geometry, Geometry) and geometry.is_empty):
            if required:
                raise FootprintOrderError(f"{label} must contain native cell coverage")
            return
        # Lines, points and broken outlines do not describe covered ground.
        if not isinstance(geometry, Geometry):
            raise FootprintOrderError(f"{label} must be a Shapely polygon shape")
        if (
            geometry.geom_type not in {"Polygon", "MultiPolygon"}
            or not geometry.is_valid
        ):
            raise FootprintOrderError(f"{label} must contain valid polygon coverage")
        # Only ordinary longitude/latitude pairs are supported. Missing,
        # infinite or extra coordinate values make the calculation unsuitable.
        if (
            shapely.get_coordinate_dimension(geometry) != 2
            or not np.isfinite(shapely.get_coordinates(geometry)).all()
        ):
            raise FootprintOrderError(
                f"{label} must use finite two-dimensional coordinates"
            )
        # Catch projected coordinates and shapes extending beyond the map.
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
            rebuilt: Coverage assembled from the supplied complete grid cells.

        Returns:
            None when the original and rebuilt roles cover exactly the same ground.

        Raises:
            FootprintOrderError: If a role contains partial cells or different coverage.

        Examples:
            HistoricalFootprintOrder._require_same_coverage("grace", old, rebuilt)
        """
        # Treat None and an empty shape as the same absence of coverage.
        # Otherwise compare covered ground, allowing a different corner-list
        # order but rejecting missing cells or extra ground.
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
    """Sort integer positions with explicit comparison and equal-value rules.

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
        """Sort a selected range using fixed choices for equal values.

        Args:
            start: First included position in order.
            end: First excluded position, or None for the complete remaining list.

        Returns:
            None. Updates order in place.

        Examples:
            _HistoricalComparisonSort([0, 1], [2.0, 1.0]).sort(0, 2)
        """
        # Work only inside the requested part of the list. Zero or one item
        # already has an order and needs no comparisons.
        stop = len(self.order) if end is None else end
        if not 0 <= start <= stop <= len(self.order):
            raise ValueError("The requested sorting range is outside the list.")
        size = stop - start
        if size < 2:
            return
        # Split large ranges into smaller groups first. Limit the number of
        # splits so awkward input cannot keep this step running indefinitely.
        self._divide(start, stop, 2 * (size.bit_length() - 1))
        # Finish by moving each item into its place among preceding items.
        if size <= 16:
            self._insert_range(start, stop)
        else:
            # Finishing the first sixteen positions establishes the smallest
            # value at the start. Later insertions can then stop on that value.
            self._insert_range(start, start + 16)
            for position in range(start + 16, stop):
                self._insert_one(position)

    def _place_pivot(self, start: int, end: int) -> None:
        """Choose a comparison value from three positions and place it first.

        Args:
            start: First included position.
            end: First excluded position.

        Returns:
            None. Exchanges the chosen position with start.

        Examples:
            sorter._place_pivot(0, len(sorter.order))
        """
        # Choose the middle value among three samples. It makes the next split
        # less likely to put nearly every item on the same side. The exact
        # comparisons also fix which sample is selected when values are equal.
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
        # Put the chosen value at the start so the splitting method can find it.
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
        # Use the selected value to separate smaller and larger values.
        self._place_pivot(start, end)
        order = self.order
        values = self.values
        pivot = values[order[start]]
        left, right = start + 1, end
        # Walk inward from both ends. Swap values found on the wrong side;
        # when the walks meet, the returned position separates the groups.
        while True:
            while values[order[left]] < pivot:
                left += 1
            right -= 1
            while pivot < values[order[right]]:
                right -= 1
            if left >= right:
                return left
            # Equal values are exchanged too. Skipping that swap would change
            # the order of shapes whose bounding boxes have equal centres.
            order[left], order[right] = order[right], order[left]
            left += 1

    def _divide(self, start: int, end: int, remaining_depth: int) -> None:
        """Work on the right group first, then continue with the left group.

        Args:
            start: First included position.
            end: First excluded position.
            remaining_depth: Allowed splitting rounds before switching to largest-first sorting.

        Returns:
            None. Leaves groups of at most sixteen for the final insertion pass.

        Examples:
            sorter._divide(0, len(sorter.order), 20)
        """
        # Stop splitting small groups: the final insertion pass handles them.
        # If too many splits were needed, use the alternative sorting method
        # that repeatedly moves the largest value to the end of the range.
        while end - start > 16:
            if remaining_depth == 0:
                self._heap_sort(start, end)
                return
            remaining_depth -= 1
            cut = self._split(start, end)
            # Process the right group with a new call, then reuse this call
            # for the left group. Keep that processing order fixed on ties.
            self._divide(cut, end, remaining_depth)
            end = cut

    def _insert_one(self, position: int) -> None:
        """Move one value left until the preceding value is no larger.

        Args:
            position: Position of the value being inserted into a sorted preceding range.

        Returns:
            None. Shifts larger preceding values one place right.

        Examples:
            sorter._insert_one(2)
        """
        # Save the item before shifting larger preceding values to the right.
        # Equal values stay in place during this insertion step.
        order = self.order
        values = self.values
        item = order[position]
        item_value = values[item]
        previous = position - 1
        while item_value < values[order[previous]]:
            order[position] = order[previous]
            position = previous
            previous -= 1
            # The caller prepares a smaller stopping value at the front.
            # Reaching past it means that preparation was not satisfied.
            if previous < 0:
                raise ValueError("The prepared prefix has no stopping value.")
        # The shifts leave one open position for the saved item.
        order[position] = item

    def _insert_range(self, start: int, end: int) -> None:
        """Sort a short range by placing each item among its preceding items.

        Args:
            start: First included position.
            end: First excluded position.

        Returns:
            None. Sorts the selected range in place.

        Examples:
            sorter._insert_range(0, min(16, len(sorter.order)))
        """
        # Grow a sorted range one item at a time. A new smallest value needs
        # special handling because there is no earlier value to stop its walk.
        for position in range(start + 1, end):
            item = self.order[position]
            if self.values[item] < self.values[self.order[start]]:
                self.order[start + 1 : position + 1] = self.order[start:position]
                self.order[start] = item
            else:
                self._insert_one(position)

    def _adjust_heap(self, start: int, hole: int, length: int, item: int) -> None:
        """Fill an open tree position with larger children, choosing right on ties.

        Args:
            start: First position of the largest-at-top tree within order.
            hole: Relative position that needs a value.
            length: Number of positions in that tree.
            item: Original value to insert after lowering the hole.

        Returns:
            None. Restores the tree with larger values above their children.

        Examples:
            sorter._adjust_heap(0, 0, len(sorter.order), sorter.order[0])
        """
        # Here the list is treated as a branching tree: each position has up
        # to two children, and larger values belong above smaller children.
        # Move the larger child up into each open position, choosing the right
        # child when both values are equal so those choices stay repeatable.
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
        # An even-sized tree can finish with only a left child. Move it up too.
        if length % 2 == 0 and child == (length - 2) // 2:
            child = 2 * (child + 1)
            order[start + hole] = order[start + child - 1]
            hole = child - 1
        # Place the saved item in the hole, moving it upward while it is larger
        # than the position above it. This restores the largest-at-top layout.
        parent = (hole - 1) // 2
        while hole > top and values[order[start + parent]] < values[item]:
            order[start + hole] = order[start + parent]
            hole = parent
            parent = (hole - 1) // 2
        order[start + hole] = item

    def _heap_sort(self, start: int, end: int) -> None:
        """Sort a range by repeatedly moving its largest remaining value to the end.

        Args:
            start: First included position.
            end: First excluded position.

        Returns:
            None. Sorts the range in place if repeated splitting reaches its limit.

        Examples:
            sorter._heap_sort(0, len(sorter.order))
        """
        # Arrange the range as a tree with its largest value at the front.
        # Start at the lowest parents so their children are already arranged.
        length = end - start
        if length < 2:
            return
        for parent in range((length - 2) // 2, -1, -1):
            self._adjust_heap(start, parent, length, self.order[start + parent])
        # Move that largest value to the end, shorten the working range, and
        # restore its largest-at-front layout. Repeating leaves ascending order.
        while length > 1:
            length -= 1
            item = self.order[start + length]
            self.order[start + length] = self.order[start]
            self._adjust_heap(start, 0, length, item)


class _HistoricalPolygonUnion:
    """Combine covered ground using a fixed shape order and joining sequence.

    Equal ground can have different written corner orders. Sort the supplied
    pieces and join them in a defined sequence so their written result can be
    used consistently in footprint ordering codes.

    Args:
        None. Each join keeps its own polygon list and sorting positions.

    Examples:
        combined = _HistoricalPolygonUnion.join([west, east])
    """

    @classmethod
    def join(cls, geometries: Iterable[Geometry | None]) -> Geometry | None:
        """Join present polygon parts using fixed sorting and midpoint grouping.

        Args:
            geometries: Polygon shapes in their supplied order; missing roles
                and empty shapes add no polygon parts.

        Returns:
            All ground covered by the pieces, or None when no area is present.

        Examples:
            qualified = _HistoricalPolygonUnion.join(qualified_cells)
        """
        # Remove missing, empty and non-area pieces first. Keep the supplied
        # order of all remaining pieces until the explicit sorting step below.
        polygons = cls._collect_parts(geometries)
        if not polygons:
            return None
        order = list(range(len(polygons)))
        if len(polygons) > 1:
            bounds = shapely.bounds(polygons)
            # Twice each bounding-box centre is enough to compare positions;
            # dividing by two would not change their relative order.
            horizontal = cast("list[float]", (bounds[:, 0] + bounds[:, 2]).tolist())
            vertical = cast("list[float]", (bounds[:, 1] + bounds[:, 3]).tolist())
            # Sort west to east, divide into vertical strips, then sort south
            # to north within each strip. The number of strips uses groups of
            # ten to keep this layout fixed for each input size. Many squares
            # share centre coordinates, so use the explicit equal-value rules.
            strips = ceil(sqrt(ceil(len(polygons) / 10)))
            per_strip = ceil(len(polygons) / strips)
            _HistoricalComparisonSort(order, horizontal).sort()
            sorter = _HistoricalComparisonSort(order, vertical)
            for start in range(0, len(polygons), per_strip):
                sorter.sort(start, min(start + per_strip, len(polygons)))
        # Join the ordered pieces by repeatedly dividing the list in half.
        # Both the piece order and the joining sequence affect the written shape.
        return cls._combine(polygons, order, 0, len(order))

    @classmethod
    def _collect_parts(cls, geometries: Iterable[Geometry | None]) -> list[Geometry]:
        """Collect polygon parts together without changing their order or bytes.

        Args:
            geometries: Shapes in their supplied order, possibly including
                missing roles, empty shapes and nested collections.

        Returns:
            Present polygon parts in original input and stored part order.
            Arrays of simple polygons retain the exact supplied shape objects.
            Nonpolygon pieces add no coverage and are omitted.

        Examples:
            polygons = _HistoricalPolygonUnion._collect_parts([west, None, east])
        """
        # Accept lists and other collections through one array. Keep existing
        # shape objects so collecting pieces does not rewrite their coordinates.
        values = np.asarray(
            geometries if isinstance(geometries, np.ndarray) else list(geometries),
            dtype=object,
        )
        type_ids = shapely.get_type_id(values)
        polygon_type = shapely.GeometryType.POLYGON
        # Grid cells normally contain one rectangle each. Handle that common
        # case together instead of visiting every rectangle in a Python loop.
        if np.all((type_ids == polygon_type) | (type_ids == -1)):
            present = (type_ids == polygon_type) & ~shapely.is_empty(values)
            return cast("list[Geometry]", values[present].tolist())

        # Split shapes containing several pieces without reordering those
        # pieces. If each resulting piece is simple, filtering is enough.
        pieces = shapely.get_parts(values)
        type_ids = shapely.get_type_id(pieces)
        present = ~shapely.is_empty(pieces)
        nested = (type_ids == shapely.GeometryType.MULTIPOLYGON) | (
            type_ids == shapely.GeometryType.GEOMETRYCOLLECTION
        )
        if not np.any(nested & present):
            return cast(
                "list[Geometry]", pieces[(type_ids == polygon_type) & present].tolist()
            )
        # Some collections contain further collections. Open those in their
        # existing position so later pieces do not jump ahead of their contents.
        polygons = []
        for position in np.flatnonzero(present & ((type_ids == polygon_type) | nested)):
            piece = pieces[position]
            if type_ids[position] == polygon_type:
                polygons.append(piece)
            else:
                polygons.extend(cls.parts(piece))
        return polygons

    @classmethod
    def _combine(
        cls, polygons: list[Geometry], order: list[int], start: int, end: int
    ) -> Geometry:
        """Join a consecutive range of pieces by splitting it at its midpoint.

        Args:
            polygons: Original polygon parts.
            order: Sorted positions into polygons.
            start: First included position in order.
            end: First excluded position in order.

        Returns:
            Combined covered ground using the specified piece order.

        Examples:
            shape = _HistoricalPolygonUnion._combine(polygons, order, 0, len(order))
        """
        # A single piece needs no join. For two pieces, keep left before right.
        if end - start == 1:
            return polygons[order[start]]
        if end - start == 2:
            left, right = polygons[order[start]], polygons[order[start + 1]]
        else:
            # Build each half before joining the halves. This fixes the sequence
            # of joins rather than combining everything in an unspecified order.
            middle = (start + end) // 2
            left = cls._combine(polygons, order, start, middle)
            right = cls._combine(polygons, order, middle, end)
        # Ask the geometry library to calculate the combined ground even when
        # pieces are separated. Copying them directly could list corners in a
        # different order. A zero grid size keeps coordinates at full precision.
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
        # Missing area contributes nothing; one polygon is already a piece.
        if geometry.is_empty:
            return []
        if geometry.geom_type == "Polygon":
            return [geometry]
        # Walk through nested collections in stored order. Lines or points
        # left by clipping an edge contain no area and are left out.
        if geometry.geom_type in {"MultiPolygon", "GeometryCollection"}:
            return [part for member in geometry.geoms for part in cls.parts(member)]
        return []


@dataclass(frozen=True, slots=True)
class _NativeRoleCells:
    """Keep grid-cell rectangles and their three kinds of coverage together.

    Args:
        shapes: Grid-cell rectangles with a fixed corner order, sorted by location.
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
        """Rebuild fixed grid-cell rectangles while keeping coverage kinds unchanged.

        Args:
            native_cells: Complete, validated daily cell table with exact coordinates,
                original role counts and both display halves of a dateline cell.

        Returns:
            Rebuilt grid-cell rectangles sorted by longitude, latitude and rectangle
            bytes. All real, grace and gap-only membership stays unchanged.

        Raises:
            FootprintOrderError: If cell shapes or grid coordinates changed
                after the caller's table was validated.

        Examples:
            cells = _NativeRoleCells.from_cells(accepted_cells)
        """
        # Quarter-degree centres are exact multiples of 0.25 degrees. Reject
        # off-grid, missing or out-of-range positions before building rectangles.
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
        # A square is 0.25 degrees wide, extending 0.125 from its centre.
        # The square centred at +180 has two displayed halves. Recognize its
        # western half by the supplied bounds and rebuild it at the left edge.
        supplied = frame.geometry.to_numpy()
        bounds = shapely.bounds(supplied)
        western_half = (longitude == 180) & (bounds[:, 0] == -180)
        minimum_x = longitude - 0.125
        maximum_x = np.minimum(longitude + 0.125, 180)
        minimum_x[western_half] = -180
        maximum_x[western_half] = -179.875
        # Constructing rectangles ourselves fixes where the corner list starts
        # and which way it travels, regardless of how the input file lists it.
        # Clip at the poles and map edges without changing the covered ground.
        shapes = shapely.box(
            minimum_x,
            np.maximum(latitude - 0.125, -90),
            maximum_x,
            np.minimum(latitude + 0.125, 90),
        )
        # A file must describe complete squares, not partial or altered shapes.
        # Equal covered ground is sufficient even if its corners are reordered.
        if not (
            np.all(shapely.get_coordinate_dimension(supplied) == 2)
            and np.all(shapely.is_valid(supplied))
            and np.all(shapely.equals(shapes, supplied))
        ):
            raise FootprintOrderError(
                "every cell must retain its complete native rectangle"
            )
        # Each square has one role: gap fill, grace, or qualified coverage.
        # Keep those flags aligned with the rectangles when sorting them below.
        gap = frame["gap_filled"].to_numpy(dtype=bool)
        on_grace = (~gap) & frame["grace_days"].fillna(0).to_numpy(dtype=float).astype(
            bool
        )
        qualified = (~gap) & ~on_grace
        # Sort by longitude, then latitude. At +180, the two displayed halves
        # share a centre, so their fixed rectangle bytes choose their order.
        # Repeated identical rectangles would count the same half twice.
        rectangle_bytes = shapely.to_wkb(shapes)
        order = np.lexsort((rectangle_bytes, latitude, longitude))
        ordered_bytes = rectangle_bytes[order]
        if np.any(ordered_bytes[1:] == ordered_bytes[:-1]):
            raise FootprintOrderError(
                "the complete day must not repeat a native cell half"
            )
        return cls(shapes[order], qualified[order], on_grace[order])

    def role_groups(self) -> list[_RoleShapes]:
        """Group touching cells and combine each kind of coverage separately.

        Args:
            None.

        Returns:
            Qualified, grace and gap-fill shapes for each connected group, in
            the combined shape's stored part order, before joining map edges.

        Raises:
            FootprintOrderError: If cell membership is missing or ambiguous.

        Examples:
            groups = cells.role_groups()
        """
        # Join all cells, including gap fill, to find the connected groups.
        # Keeping every cell here matters even for groups removed later.
        extent = _HistoricalPolygonUnion.join(self.shapes)
        if extent is None:
            raise FootprintOrderError("native cells must provide present coverage")
        parts = _HistoricalPolygonUnion.parts(extent)
        bounds = shapely.bounds(self.shapes)
        # Match each square to a group using a point at its centre. Centres
        # are inside their squares, so shared edges do not make the match unclear.
        points = shapely.points(
            (bounds[:, 0] + bounds[:, 2]) / 2, (bounds[:, 1] + bounds[:, 3]) / 2
        )
        matches = shapely.STRtree(parts).query(points, predicate="within")
        # Require one match per square: neither unmatched squares nor squares
        # assigned to several groups can supply a valid complete footprint.
        if matches.shape[1] != len(self.shapes) or len(np.unique(matches[0])) != len(
            self.shapes
        ):
            raise FootprintOrderError(
                "every native cell must belong to one connected group"
            )
        # Store each cell's group number in cell order, then combine qualified,
        # grace and gap-filled cells separately within that group. Keeping the
        # kinds separate lets callers check each against the supplied footprint.
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
    """Keep three kinds of coverage for one group before joining map edges.

    Args:
        qualified: Qualified coverage, or None when absent.
        grace: Grace coverage, or None when absent.
        gap: Coverage added to connect nearby cells, or None when absent.

    Examples:
        group = _RoleShapes(core, grace, gap)
    """

    qualified: Geometry | None
    grace: Geometry | None
    gap: Geometry | None

    @property
    def full(self) -> Geometry:
        """Combine qualified, then grace, then gap-filled coverage in that order.

        Args:
            None.

        Returns:
            All ground covered by the three shapes, joined two at a time.

        Raises:
            FootprintOrderError: If all roles are absent.

        Examples:
            complete = group.full
        """
        # Fix which coverage is joined first; changing the grouping can change
        # the written corner order even when the same ground is covered.
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
        # Gap fill connects coverage but cannot make a group a footprint by
        # itself. At least qualified or grace coverage must be present.
        return any(
            shape is not None and not shape.is_empty
            for shape in (self.qualified, self.grace)
        )

    @staticmethod
    def _extend(first: Geometry | None, second: Geometry | None) -> Geometry | None:
        """Skip absent coverage and combine present shapes in their given order.

        Args:
            first: Earlier coverage kind or previously combined shape.
            second: Later role to add, possibly absent.

        Returns:
            Their combined ground, the one present shape, or no coverage.

        Examples:
            full = _RoleShapes._extend(qualified, grace)
        """
        # Avoid a join when only one shape contains ground. Keep that shape's
        # existing coordinates and piece order instead of rebuilding it.
        if first is None or first.is_empty:
            return second if second is not None and not second.is_empty else first
        if second is None or second.is_empty:
            return first
        # This operation combines the already built coverage kinds. Its handling
        # of separated pieces differs from the cell-joining calculation above;
        # keep this operation so the resulting written shape follows fixed rules.
        return first.union(second)

    @staticmethod
    def _leader(parents: dict[int, int], component: int) -> int:
        """Follow linked group numbers to the number identifying their shared group.

        Args:
            parents: Changeable links between group numbers at the map edges.
            component: Group index whose representative is needed.

        Returns:
            Shared group number. Shortens visited links for later lookups.

        Examples:
            representative = _RoleShapes._leader({0: 1, 1: 1}, 0)
        """
        # A group points to itself when it is the final answer. Otherwise skip
        # one link on each visit, shortening the path for subsequent lookups.
        while parents[component] != component:
            parents[component] = parents[parents[component]]
            component = parents[component]
        return component

    @classmethod
    def join_dateline(cls, roles: list[_RoleShapes]) -> list[_RoleShapes]:
        """Join groups sharing real edges across the world-map cut.

        Args:
            roles: The three coverage kinds in each group, in supplied group order.

        Returns:
            Joined role groups in first-member order; unjoined groups pass through.

        Examples:
            joined = _RoleShapes.join_dateline(groups)
        """
        # The two map edges are adjacent on the globe. Only groups reaching an
        # edge can connect across that cut, so leave interior groups out of this
        # search. Initially each edge group belongs to itself.
        full = [group.full for group in roles]
        seam = [
            index
            for index, shape in enumerate(full)
            if shape.bounds[0] <= -180 or shape.bounds[2] >= 180
        ]
        parents = {index: index for index in seam}
        for position, left in enumerate(seam):
            for right in seam[position + 1 :]:
                # Move one group by a whole world width to test the other side
                # of the map. Join only when they share an edge longer than the
                # small comparison threshold; touching at a corner is not enough.
                shared_edge = max(
                    full[left].intersection(translate(full[right], xoff=shift)).length
                    for shift in (-360.0, 360.0)
                )
                if shared_edge > 1e-6:
                    parents[cls._leader(parents, left)] = cls._leader(parents, right)
        # Collect connected groups and retain their earliest list position.
        # Later members disappear as separate entries, while unrelated groups
        # keep their positions. This prevents the join from reordering the day.
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
        # Rebuild only groups that actually joined. Process each coverage kind
        # separately so qualified, grace and gap fill retain their meanings.
        result = []
        for members in groups:
            if len(members) == 1:
                result.append(roles[members[0]])
                continue
            # Move a group entirely at the western edge beside its eastern
            # neighbours. A group already spanning both edges must stay put:
            # moving all of it would also move its eastern coverage incorrectly.
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
            geometries: One coverage kind for each joined map-edge group.
            shifts: Matching longitude shifts selected from the complete footprints.

        Returns:
            Joined role coverage folded into [-180, 180], or None when absent.

        Examples:
            stitched = _RoleShapes._stitch([west, east], [360.0, 0.0])
        """
        # Move present shapes beside their neighbours before joining them.
        # The offsets come from complete footprints so each kind of coverage
        # follows the same placement, even if its own outline reaches no edge.
        unfolded = [
            translate(shape, xoff=shift)
            for shape, shift in zip(geometries, shifts, strict=True)
            if shape is not None and not shape.is_empty
        ]
        # Return the joined shape to the normal map range. A coverage kind
        # absent from every member remains absent rather than gaining a shape.
        combined = _HistoricalPolygonUnion.join(unfolded)
        return None if combined is None else cls._wrap(combined)

    @staticmethod
    def _wrap(geometry: Geometry) -> Geometry:
        """Clip occupied world-width strips and fold their polygons onto the map.

        Args:
            geometry: Joined role, potentially extending beyond +180 longitude.

        Returns:
            Covered ground inside the map bounds, joined in strip order.

        Examples:
            wrapped = _RoleShapes._wrap(unfolded)
        """
        # Most shapes already fit the map; avoid rewriting their coordinates.
        minimum, _, maximum, _ = geometry.bounds
        if minimum >= -180 and maximum <= 180:
            return geometry
        # Find every 360-degree-wide copy of the map touched by this shape.
        # Negative strip numbers cover ground to the west of the usual map.
        first = floor((minimum + 180) / 360)
        last = floor((maximum + 180) / 360)
        polygons = []
        # Cut the shape at each copy's edges, then move its area pieces into
        # [-180, 180]. Keep strip and piece order fixed for the final join.
        for strip in range(first, last + 1):
            clipped = geometry.intersection(
                shapely.box(-180 + 360 * strip, -90, 180 + 360 * strip, 90)
            )
            polygons.extend(
                translate(part, xoff=-360 * strip)
                for part in _HistoricalPolygonUnion.parts(clipped)
            )
        # Combine the folded pieces. If clipping produced no area pieces,
        # leave the supplied shape intact rather than inventing coverage.
        result = _HistoricalPolygonUnion.join(polygons)
        return geometry if result is None else result
