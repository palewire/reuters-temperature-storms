"""Turn accepted daily histories into mature, grace and gap-filled grid cells."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing
from shapely import box

from .cell_state import WaveCellState
from .cells import WaveCellError, WaveCells


@dataclass(frozen=True, slots=True)
class WaveCellBuilder:
    """Build exact cell polygons without allocating a global geometry table.

    Steps:
        1. Retain the supplied fields for this record.
        2. Expose the values to the corresponding processing and validation methods.

    Args:
        state: Accepted histories for one observation date.
        closing_radius: Circular gap-filling radius measured in grid cells.

    Examples:
        cells = WaveCellBuilder(state, closing_radius=3).build()
    """

    state: WaveCellState
    closing_radius: int = 3

    def __post_init__(self) -> None:
        """Require accepted state and a nonnegative whole-number radius.

        Steps:
            1. Require an already validated WaveCellState input.
            2. Require the closing radius to be a nonnegative integer.

        Args:
            None.

        Returns:
            None. Leaves already validated state unchanged.

        Raises:
            WaveCellError: If state or the radius has an unsupported type.

        Examples:
            builder = WaveCellBuilder(state, 3)
        """
        if not isinstance(self.state, WaveCellState):
            raise WaveCellError("state must be a validated WaveCellState table")
        if type(self.closing_radius) is not int or self.closing_radius < 0:
            raise WaveCellError("closing_radius must be a nonnegative integer")

    def mature_cells(self) -> pd.DataFrame:
        """Keep active histories with at least five qualifying dates.

        Steps:
            1. Drop ended histories because they no longer connect today's waves.
            2. Keep cells with five or more qualifying days, including cells in
               their one allowed grace day. Preserve both supplied day counts.

        Args:
            None.

        Returns:
            Coordinates and counts of mature active cells, without geometry.

        Examples:
            selected = builder.mature_cells()
        """
        frame = self.state.frame
        return cast(
            "pd.DataFrame",
            frame.loc[
                frame["end_date"].isna() & frame["days"].ge(5),
                ["longitude", "latitude", "days", "grace_days"],
            ],
        )

    def fill_gaps(self, mature: pd.DataFrame) -> pd.DataFrame:
        """Expand then shrink the selected grid to close small spatial holes.

        Steps:
            1. Place selected coordinates in a small boolean global grid. This
               records membership only, without constructing a million polygons.
            2. Build a circular neighbourhood with distances strictly smaller
               than the radius. Zero and one leave the selection unchanged.
            3. Pad longitude by wrapping across the dateline. Do not wrap latitude:
               positions beyond either pole remain absent during the calculation.
            4. Expand and shrink once, then recover surviving coordinates and
               their original counts. Added cells have absent counts and a gap flag.

        Args:
            mature: Selected coordinate and count columns from mature_cells.

        Returns:
            Surviving and added coordinates with counts and gap-fill flags.
            The first two polar rows can be removed by the default calculation.

        Examples:
            coordinates = builder.fill_gaps(builder.mature_cells())
        """
        radius = self.closing_radius
        if mature.empty or radius <= 1:
            return mature.assign(gap_filled=False).reset_index(drop=True)

        # Each array entry represents one exact quarter-degree centre.
        selected = np.zeros((721, 1440), dtype=bool)
        rows = np.rint((mature["latitude"].to_numpy() + 90) * 4).astype(int)
        columns = np.rint((mature["longitude"].to_numpy() + 179.75) * 4).astype(int)
        selected[rows, columns] = True

        # Circular neighbours connect short gaps while preserving longitude wrapping.
        offsets = np.arange(-radius, radius + 1)
        x, y = np.meshgrid(offsets, offsets)
        neighbourhood = x * x + y * y < radius * radius
        padding = radius * 2
        padded = np.pad(selected, ((0, 0), (padding, padding)), mode="wrap")
        closed = binary_closing(padded, structure=neighbourhood, iterations=1)
        rows, columns = np.nonzero(closed[:, padding:-padding])

        # Attach original counts only where an input cell actually survives.
        coordinates = pd.DataFrame(
            {
                "longitude": columns / 4 - 179.75,
                "latitude": rows / 4 - 90,
            }
        )
        result = coordinates.merge(mature, on=["longitude", "latitude"], how="left")
        result["gap_filled"] = result["days"].isna()
        return result

    def build(self) -> WaveCells:
        """Construct polygons only for the cells left after small-gap filling.

        Steps:
            1. Select mature active histories and run the spatial gap calculation.
            2. Sort their coordinates so file order does not depend on input order.
            3. Create quarter-degree rectangles, clipping at the poles and +180.
            4. Add the western half of every +180 cell with matching counts and
               flags. Both halves retain the same +180 cell coordinate.
            5. Attach the observation date and validate the seven-column output
               once. It can now be stored or passed to footprint construction.

        Args:
            None.

        Returns:
            Validated daily cells with EPSG:4326 geometry and original real counts.

        Examples:
            cells = builder.build()
            cells.to_parquet("wave-cells/2024-07-01.parquet")
        """
        coordinates = (
            self.fill_gaps(self.mature_cells())
            .sort_values(["longitude", "latitude"], kind="stable")
            .reset_index(drop=True)
        )
        longitude = coordinates["longitude"].to_numpy(dtype=float)
        latitude = coordinates["latitude"].to_numpy(dtype=float)
        geometry = box(
            longitude - 0.125,
            np.maximum(latitude - 0.125, -90),
            np.minimum(longitude + 0.125, 180),
            np.minimum(latitude + 0.125, 90),
        )
        frame = gpd.GeoDataFrame(coordinates, geometry=geometry, crs=4326)

        # The dateline cell is split for ordinary geographic storage and display.
        western = frame.loc[frame["longitude"].eq(180)].copy()
        western_latitude = western["latitude"].to_numpy(dtype=float)
        western["geometry"] = box(
            -180,
            np.maximum(western_latitude - 0.125, -90),
            -179.875,
            np.minimum(western_latitude + 0.125, 90),
        )
        frame = gpd.GeoDataFrame(
            pd.concat([frame, western], ignore_index=True), crs=4326
        )
        frame["date"] = pd.Series(
            pd.Timestamp(self.state.day), index=frame.index, dtype="datetime64[ns]"
        )
        return WaveCells(frame.loc[:, list(WaveCells.columns)])
