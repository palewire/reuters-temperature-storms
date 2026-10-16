"""Build connected daily wave footprints from accepted cell shapes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, cast

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import Geometry
from shapely.affinity import translate
from shapely.geometry import box
from shapely.ops import unary_union

from .identity import component_identity

if TYPE_CHECKING:
    from datetime import date

    from .series import WaveSeries


@dataclass(frozen=True, slots=True)
class WaveFootprintBuilder:
    """Combine an accepted daily wave-cell table into footprint role shapes.

    Combining shapes and joining dateline edges follow a fixed calculation order.
    That order keeps the shape text and resulting daily hashes stable.
    Validation belongs to the input and output table models.

    Steps:
        1. Combine accepted cell shapes while keeping their qualifying, grace or gap-fill role.
        2. Join matching edges at the dateline and measure each role separately.
        3. Use a fixed calculation order so input row order does not change the daily shape text or ID.

    Args:
        cells: Accepted wave-cell rows in EPSG:4326 for one day.
        day: Calendar date represented by the cells.
        series: Temperature and direction represented by the cells.

    Examples:
        frame = WaveFootprintBuilder(cells.frame, day, series).build()
    """

    cells: gpd.GeoDataFrame
    day: date
    series: WaveSeries
    columns: ClassVar[tuple[str, ...]] = (
        "storm_id",
        "date",
        "geometry",
        "grace_geometry",
        "buffer_geometry",
        "area_qualified",
        "area_grace",
        "area_gap_filled",
        "qualified_fraction",
    )

    def build(self) -> gpd.GeoDataFrame:
        """Build shapes, role areas, fractions and daily footprint hashes.

        Steps:
            1. Return the typed empty table when there are no cells.
            2. Sort cells by longitude, latitude and polygon bytes before combining shapes.
            3. Combine all roles to find connected groups and assign each cell to a group using a point inside its polygon.
            4. Combine qualifying, grace and gap-fill shapes separately within each group.
            5. Join groups sharing a real edge across the dateline and measure each role area.
            6. Remove groups made only of gap-fill cells and calculate the qualifying share of the remaining area.
            7. Combine qualifying with grace first, then add gap fill to obtain the complete shape.
            8. Calculate the daily ID from that shape text and date, and return the footprint columns.

        Args:
            None.

        Returns:
            One row per connected footprint containing real cells. Gap-fill-only
            groups are discarded; grace-only groups remain available to tracking.
            The buffer geometry is the complete qualified/grace/gap-fill extent.

        Examples:
            footprints = WaveFootprintBuilder(cells.frame, day, series).build()
        """
        # A quiet day still needs the usual columns and types. This lets later
        # stages read an empty file rather than guess whether the date is missing.
        if self.cells.empty:
            return self.empty_frame()

        # Work on a copy so temporary grouping columns never change the input.
        # Coordinate order affects the resulting shape text and daily ID. The
        # polygon bytes break the tie between the two halves of a dateline cell.
        cells = self.cells.copy()
        cells["_geometry_wkb_sort"] = cells.geometry.to_wkb()
        cells = cells.sort_values(
            ["longitude", "latitude", "_geometry_wkb_sort"],
            kind="stable",
            ignore_index=True,
        ).drop(columns="_geometry_wkb_sort")
        # A real cell is either qualifying today or on its one-day grace period.
        # Gap-fill cells connect nearby regions but never count as real cells.
        cells["gap_filled"] = cells["gap_filled"].astype(bool)
        cells["on_grace"] = (~cells["gap_filled"]) & cells["grace_days"].fillna(0).gt(0)

        # Combine all cell roles to find connected regions. Gap-fill cells may
        # bridge real cells; corner-only contact leaves separate polygon parts.
        # These regions determine membership, not the qualifying area.
        extent = (
            gpd.GeoSeries([cells.geometry.union_all()], crs=4326)
            .explode(index_parts=False)
            .reset_index(drop=True)
        )
        groups = gpd.GeoDataFrame(
            {"footprint_id": range(len(extent))}, geometry=extent, crs=4326
        )
        # Give each cell to the region containing a point inside that cell.
        # Using an inside point avoids ambiguous matches along shared edges.
        # Keep the original polygon so we can measure its own role afterward.
        cells["cell_geometry"] = cells.geometry
        points = cells.set_geometry(cells.geometry.representative_point())
        joined = gpd.sjoin(
            points[["geometry", "gap_filled", "on_grace", "cell_geometry"]],
            groups,
            predicate="within",
            how="inner",
        ).set_geometry("cell_geometry")
        joined = joined.drop(columns="geometry")
        # Rebuild each role separately within those shared groups. Reindex all
        # three results to the same group numbers: an absent role stays missing,
        # rather than shifting the next group's shape into the wrong row.
        qualified = self.dissolve(
            joined[~joined["gap_filled"] & ~joined["on_grace"]]
        ).reindex(range(len(groups)))
        grace = self.dissolve(joined[joined["on_grace"]]).reindex(range(len(groups)))
        gap_fill = self.dissolve(joined[joined["gap_filled"]]).reindex(
            range(len(groups))
        )
        # At this point buffer_geometry holds ONLY gap-fill area. It becomes the
        # full outline later, after that separate area has been measured.
        footprints = gpd.GeoDataFrame(
            {
                "grace_geometry": grace.to_numpy(),
                "buffer_geometry": gap_fill.to_numpy(),
            },
            geometry=gpd.GeoSeries(qualified.to_numpy(), crs=4326),
            crs=4326,
        )
        for column in ("grace_geometry", "buffer_geometry"):
            footprints[column] = gpd.GeoSeries(footprints[column], crs=4326)
        # The flat world map splits connected regions at +/-180 degrees. Join
        # matching edges before measuring, while retaining each role's shape.
        footprints = self.merge_dateline(footprints)
        footprints["area_qualified"] = self.area_km2(footprints.geometry)
        footprints["area_grace"] = self.area_km2(footprints["grace_geometry"])
        footprints["area_gap_filled"] = self.area_km2(footprints["buffer_geometry"])
        # A bridge without any real cells is not a wave. Keep grace-only groups:
        # they can carry a wave through a day without fresh qualifying cells.
        real_area = footprints["area_qualified"] + footprints["area_grace"]
        footprints = footprints[real_area > 0].reset_index(drop=True)
        # The qualifying share uses the whole connected region as its denominator,
        # including grace and gap fill. Their separate areas remain in the output.
        total_area = (
            footprints["area_qualified"]
            + footprints["area_grace"]
            + footprints["area_gap_filled"]
        )
        footprints["qualified_fraction"] = footprints["area_qualified"] / total_area
        # Build the complete outline in a fixed role order, then hash its text
        # together with the date. This ID describes this day's footprint; the
        # tracker assigns the permanent wave ID afterward.
        extents = self.extents(footprints)
        footprints["storm_id"] = pd.Series(
            [
                component_identity(
                    temperature=self.series.temperature,
                    direction=self.series.direction,
                    date=np.datetime64(self.day),
                    geometry=geometry,
                )
                for geometry in extents
            ],
            index=footprints.index,
            dtype="str",
        )
        # Replace the temporary gap-fill shape with the full outline expected by
        # tracking. Keep the measured gap-fill area, stamp the day and select only
        # the published table columns, dropping all temporary grouping details.
        footprints["buffer_geometry"] = gpd.GeoSeries(
            extents, index=footprints.index, crs=4326
        )
        footprints["date"] = pd.Series(
            pd.Timestamp(self.day), index=footprints.index, dtype="datetime64[ns]"
        )
        return cast("gpd.GeoDataFrame", footprints[list(self.columns)])

    @classmethod
    def empty_frame(cls) -> gpd.GeoDataFrame:
        """Create the same typed footprint columns for a day without cells.

        Steps:
            1. Create empty ID and date columns with their expected types.
            2. Create empty geographic columns in EPSG:4326 and floating-point area fields.
            3. Return the columns in the same order as a populated footprint table.

        Args:
            None.

        Returns:
            Empty EPSG:4326 table with floating measurements and calendar dates.

        Examples:
            frame = WaveFootprintBuilder.empty_frame()
        """
        # Explicit types preserve a readable file even when there are no rows.
        # Every shape column still records longitude/latitude coordinates.
        frame = gpd.GeoDataFrame(
            {
                "storm_id": pd.Series([], dtype="str"),
                "date": pd.Series([], dtype="datetime64[ns]"),
                "geometry": gpd.GeoSeries([], crs=4326),
                "grace_geometry": gpd.GeoSeries([], crs=4326),
                "buffer_geometry": gpd.GeoSeries([], crs=4326),
                **{name: pd.Series([], dtype="float64") for name in cls.columns[5:]},
            },
            geometry="geometry",
            crs=4326,
        )
        return cast("gpd.GeoDataFrame", frame[list(cls.columns)])

    @staticmethod
    def dissolve(cells: gpd.GeoDataFrame) -> gpd.GeoSeries:
        """Combine one role's polygons within their original footprint groups.

        Steps:
            1. Return an empty geographic series when no cells have this role.
            2. Group the remaining cells by footprint_id.
            3. Combine each group's polygons and return one shape per group.

        Args:
            cells: Selected role rows carrying their footprint_id and geometry.

        Returns:
            Geometry indexed by footprint_id, or an empty geographic series.

        Examples:
            qualified = WaveFootprintBuilder.dissolve(qualified_cells)
        """
        # A group can lack this role entirely. Return a geographic series so it
        # can be lined up with the other roles without inventing a polygon.
        if cells.empty:
            return gpd.GeoSeries([], crs=4326)
        # Joining only within footprint_id preserves the membership found from
        # all roles, even when this role's own cells form several separate pieces.
        return cells.dissolve("footprint_id").geometry

    @staticmethod
    def area_km2(geometries: gpd.GeoSeries) -> pd.Series:
        """Measure role shapes in square kilometres without latitude distortion.

        Steps:
            1. Describe the input shapes as longitude and latitude in EPSG:4326.
            2. Convert them to EPSG:6933, which preserves physical area in square metres.
            3. Apply a zero-width buffer to remove small polygon defects before measuring.
            4. Convert square metres to square kilometres and represent missing shapes as zero.

        Args:
            geometries: Qualified, grace or gap-fill shapes in EPSG:4326.

        Returns:
            Square-kilometre areas; missing shapes contribute zero.

        Examples:
            areas = WaveFootprintBuilder.area_km2(footprints.geometry)
        """
        # Longitude/latitude degrees are not an area unit. EPSG:6933 preserves
        # physical area; a zero-width buffer repairs polygon boundaries before
        # measuring. Missing role shapes contribute zero, not a missing number.
        return cast(
            "pd.Series",
            (
                gpd.GeoSeries(geometries, crs=4326).to_crs(6933).buffer(0).area / 1e6
            ).fillna(0.0),
        )

    @staticmethod
    def extent(qualified: Geometry | None, other: Geometry | None) -> Geometry | None:
        """Combine two present role shapes without changing union order.

        Steps:
            1. Return the other present shape when the first shape is missing or empty.
            2. Return the first shape when the second shape is missing or empty.
            3. Otherwise combine the first shape with the second in that order.

        Args:
            qualified: Earlier role shape, which may be missing or empty.
            other: Later role shape, which may be missing or empty.

        Returns:
            Their union, the present shape, or the first missing/empty value.

        Examples:
            full = WaveFootprintBuilder.extent(qualified, grace)
        """
        # Some groups have no qualifying cells or no grace cells. Keep a present
        # shape directly so an unnecessary join cannot reorder its coordinates.
        if not isinstance(qualified, Geometry) or qualified.is_empty:
            return (
                other
                if isinstance(other, Geometry) and not other.is_empty
                else qualified
            )
        if not isinstance(other, Geometry) or other.is_empty:
            return qualified
        # When both exist, retain this operand order: changing it can change the
        # shape's coordinate text even when the covered ground is identical.
        return qualified.union(other)

    @classmethod
    def extents(cls, footprints: gpd.GeoDataFrame) -> list[Geometry]:
        """Combine each row's qualified, grace and gap-fill shapes in order.

        Steps:
            1. Read each row's qualifying, grace and gap-fill shapes in table order.
            2. Combine qualifying with grace, then combine that result with gap fill.
            3. Return complete shapes in the original row order.

        Args:
            footprints: Daily geometry rows whose buffer is still gap fill.

        Returns:
            The complete connected extent for each row, preserving row order.

        Examples:
            extents = WaveFootprintBuilder.extents(footprints)
        """
        # Keep the role order explicit. Here buffer_geometry still means gap
        # fill; joining qualifying, then grace, then gap fill makes the full shape.
        return [
            cast("Geometry", cls.extent(cls.extent(qualified, grace), gap_fill))
            for qualified, grace, gap_fill in zip(
                footprints.geometry,
                footprints["grace_geometry"],
                footprints["buffer_geometry"],
                strict=False,
            )
        ]

    @staticmethod
    def find_parent(parents: dict[int, int], component: int) -> int:
        """Find a dateline group's root and shorten the traversed parent chain.

        Steps:
            1. Follow the saved parent indexes until reaching an index that names itself.
            2. Shorten the visited parent links so later lookups do less work.
            3. Return the connected group's root index.

        Args:
            parents: Mutable component-to-parent mapping for seam components.
            component: Component whose connected root is needed.

        Returns:
            The root component index. Updates the mapping along the lookup path.

        Examples:
            root = WaveFootprintBuilder.find_parent({0: 1, 1: 1}, 0)
        """
        # Follow the group links to their shared leader. Point each visited link
        # farther up the chain so repeated seam comparisons do less work.
        while parents[component] != component:
            parents[component] = parents[parents[component]]
            component = parents[component]
        return component

    @classmethod
    def merge_dateline(cls, footprints: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Join matching seam edges and stitch each role independently.

        A translated neighbouring world copy must share an edge longer than
        1e-6 degrees. Corner contact and unrelated seam latitudes stay separate.

        Steps:
            1. Find complete shapes reaching either -180 or +180 degrees longitude.
            2. Compare neighbouring world copies shifted by -360 and +360 degrees.
            3. Group shapes only when a shifted copy shares an edge longer than 1e-6 degrees; corner contact does not join them.
            4. Keep unjoined rows in their original order and leave their shape bytes unchanged.
            5. Shift a western-only member by +360 degrees, then combine and wrap each role separately for joined groups.
            6. Return the joined role columns in the first-member order of their groups.

        Args:
            footprints: Combined role shapes before world-edge joining.

        Returns:
            Rows in their original group order with joined role geometries.
            Unjoined footprints keep their original geometry bytes.

        Examples:
            stitched = WaveFootprintBuilder.merge_dateline(footprints)
        """
        # Only groups reaching the world-map edge can connect across it. Compare
        # their full outlines so a grace or gap-fill cell can also make the link.
        extents = cls.extents(footprints)
        seam = [
            index
            for index, geometry in enumerate(extents)
            if geometry.bounds[0] <= -180 or geometry.bounds[2] >= 180
        ]
        # Start each edge group separately. Moving another group one world-width
        # left or right reveals neighbours hidden by the map cut. Require shared
        # edge length, because a single touching corner must not join two waves.
        parents = {index: index for index in seam}
        for offset, left in enumerate(seam):
            for right in seam[offset + 1 :]:
                edge_length = max(
                    extents[left]
                    .intersection(translate(extents[right], xoff=shift))
                    .length
                    for shift in (-360.0, 360.0)
                )
                if edge_length > 1e-6:
                    parents[cls.find_parent(parents, left)] = cls.find_parent(
                        parents, right
                    )
        # Gather all linked members, including connections through a third group.
        # Keep the first member's row position and remove its absorbed rows so
        # joined groups do not alter the ordering of unrelated footprints.
        merged: dict[int, list[int]] = {}
        for component in seam:
            merged.setdefault(cls.find_parent(parents, component), []).append(component)
        by_leader = {
            members[0]: members for members in merged.values() if len(members) > 1
        }
        absorbed = {
            component for members in by_leader.values() for component in members[1:]
        }
        groups = [
            by_leader.get(component, [component])
            for component in range(len(footprints))
            if component not in absorbed
        ]
        # With no joins, return the original table and leave its shape bytes alone.
        if all(len(group) == 1 for group in groups):
            return footprints

        # Move western-edge pieces just past +180, beside eastern-edge pieces.
        # A piece already spanning both edges stays put; shifting that whole
        # piece would move ground that is already on the intended side.
        shifts = {
            component: (
                360.0
                if extents[component].bounds[0] <= -180
                and extents[component].bounds[2] < 180
                else 0.0
            )
            for group in groups
            for component in group
        }
        # Join each role separately using the same member offsets. This keeps
        # qualifying measurements free of grace and gap-fill area. Single-member
        # groups pass through without changing their coordinates.
        rows = []
        for members in groups:
            if len(members) > 1:
                member_shifts = [shifts[member] for member in members]
                rows.append(
                    {
                        column: cls.stitch(
                            [footprints[column].iloc[member] for member in members],
                            member_shifts,
                        )
                        for column in ("geometry", "grace_geometry", "buffer_geometry")
                    }
                )
            else:
                rows.append(
                    {
                        column: footprints[column].iloc[members[0]]
                        for column in ("geometry", "grace_geometry", "buffer_geometry")
                    }
                )
        # Restore geographic types on every role column, including missing roles,
        # so the caller can measure and write them consistently.
        result = gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)
        for column in ("grace_geometry", "buffer_geometry"):
            result[column] = gpd.GeoSeries(result[column], crs=4326)
        return result

    @classmethod
    def stitch(
        cls, geometries: list[Geometry | None], shifts: list[float]
    ) -> Geometry | None:
        """Unfold present role geometries, combine them and wrap them back.

        Steps:
            1. Ignore missing and empty shapes for the selected role.
            2. Shift each remaining shape by its corresponding longitude offset.
            3. Combine the shifted shapes and fold the result back onto the world map.
            4. Return None when the role has no shapes.

        Args:
            geometries: One role shape for each member of a joined seam group.
            shifts: Longitude offset for each corresponding member.

        Returns:
            The joined role geometry in [-180, 180], or None for an absent role.

        Examples:
            geometry = WaveFootprintBuilder.stitch([western, eastern], [360.0, 0.0])
        """
        # Use the offsets chosen from each member's full outline for this role
        # too. Missing role shapes add nothing; their placeholders are skipped.
        unfolded = [
            translate(geometry, xoff=shift)
            for geometry, shift in zip(geometries, shifts, strict=False)
            if isinstance(geometry, Geometry) and not geometry.is_empty
        ]
        # No members have this role. Otherwise combine in the unfolded space,
        # where neighbours touch, before cutting the result back onto the map.
        if not unfolded:
            return None
        return cls.wrap(unary_union(unfolded))

    @classmethod
    def wrap(cls, geometry: Geometry) -> Geometry:
        """Fold every occupied longitude strip onto the world map.

        Steps:
            1. Leave empty shapes and shapes already between -180 and +180 unchanged.
            2. Find every 360-degree longitude strip touched by the input shape.
            3. Clip the shape to each strip and keep only polygon pieces.
            4. Shift each piece back onto the world map and combine the pieces in strip order.

        Args:
            geometry: Combined role shape, possibly extending beyond +/-180.

        Returns:
            Polygonal shape within [-180, 180], combined in longitude-strip order.
            Already in-range shapes pass through unchanged.

        Examples:
            wrapped = WaveFootprintBuilder.wrap(unfolded_geometry)
        """
        # Already in-range shapes need no clipping. Returning them unchanged
        # avoids both extra work and needless changes to coordinate ordering.
        if geometry.is_empty:
            return geometry
        minimum, _, maximum, _ = geometry.bounds
        if minimum >= -180 and maximum <= 180:
            return geometry
        # Each numbered strip is another 360-degree copy of the world. The shape
        # may span more than one, so include every strip touched by its bounds.
        first_strip = math.floor((minimum + 180) / 360)
        last_strip = math.floor((maximum + 180) / 360)
        polygons = []
        # Cut to one world copy at a time, then shift its polygon pieces into
        # [-180, +180]. Edge-only lines and points have no footprint area.
        for strip in range(first_strip, last_strip + 1):
            part = geometry.intersection(
                box(-180 + 360 * strip, -90, 180 + 360 * strip, 90)
            )
            polygons.extend(
                translate(polygon, xoff=-360 * strip)
                for polygon in cls.polygon_parts(part)
            )
        # Combine pieces in strip order to retain a fixed calculation order.
        # If clipping yielded no polygons, keep the original shape for the table
        # checks to handle rather than inventing a replacement.
        return unary_union(polygons) if polygons else geometry

    @classmethod
    def polygon_parts(cls, geometry: Geometry) -> list[Geometry]:
        """Extract polygon parts after clipping a longitude strip.

        Steps:
            1. Return no pieces for an empty shape.
            2. Keep a polygon as one piece.
            3. Walk through multipart or mixed shapes to collect their polygon pieces in order.
            4. Drop lines and points that can occur when clipping only touches an edge.

        Args:
            geometry: Polygon, multipart geometry or mixed clipping result.

        Returns:
            Polygon parts in stored order, excluding edge-only lines and points.

        Examples:
            polygons = WaveFootprintBuilder.polygon_parts(clipped_geometry)
        """
        # Clipping can produce empty shapes, polygons, or mixed collections of
        # polygons and boundary lines. Retain only pieces that cover ground.
        if geometry.is_empty:
            return []
        if geometry.geom_type == "Polygon":
            return [geometry]
        # Walk nested collections in their stored order, preserving polygon
        # coordinates while discarding lines and points from edge-only contact.
        if geometry.geom_type in {"MultiPolygon", "GeometryCollection"}:
            return [
                polygon
                for part in geometry.geoms
                for polygon in cls.polygon_parts(part)
            ]
        return []
