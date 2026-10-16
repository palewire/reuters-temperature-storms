"""Synthetic geographic files for CLI tracking regression tests."""

from __future__ import annotations

from datetime import date, timedelta
from hashlib import md5
from typing import TYPE_CHECKING

import geopandas as gpd
import pandas as pd
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform

from reuters_temperature_storms import WaveCells

if TYPE_CHECKING:
    from pathlib import Path

    from shapely.geometry import Polygon

START_DATE = date(2020, 1, 1)
END_DATE = date(2020, 1, 5)
_AREA_TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)


def create_footprints(directory: Path) -> tuple[Path, ...]:
    """Write invented footprints showing movement, a split, a merge and an end.

    These boxes are synthetic tracking fixtures. They do not describe
    observed temperatures or real waves. Every box is entirely qualifying;
    grace and gap-filled areas are zero. The last date has an empty input file.

    Args:
        directory: Root directory in which to create daily footprint files.

    Returns:
        Paths to five GeoParquet files in chronological order.

    Examples:
        files = create_footprints(Path("./test-footprints"))
        assert len(files) == 5

    Steps:
        1. Construct five daily polygon sets showing movement, splitting, merging, and an empty ending day.
        2. Write dated GeoParquet files beneath the maximum-temperature warm-series directory.
        3. Return the five file paths in chronological order.
    """
    days = (
        (box(-25, -15, 15, 15),),
        (box(-23, -15, 17, 15),),
        (box(-23, -15, -4, 15), box(-2, -15, 17, 15)),
        (box(-21, -15, 19, 15),),
        (),
    )
    target = directory / "t2m_max" / "positive"
    target.mkdir(parents=True, exist_ok=True)
    files = []
    for offset, shapes in enumerate(days):
        day = START_DATE + timedelta(days=offset)
        path = target / f"{day.isoformat()}.parquet"
        _footprint_frame(day, shapes).to_parquet(path, index=False)
        files.append(path)
    return tuple(files)


def _footprint_frame(day: date, shapes: tuple[Polygon, ...]) -> gpd.GeoDataFrame:
    """Build a readable footprint table with geographic geometry columns.

    Args:
        day: Calendar date shared by every row.
        shapes: Invented qualifying polygons in longitude and latitude.

    Returns:
        GeoDataFrame with measured square-kilometer areas, including the same
        columns and types when no footprints remain.

    Examples:
        frame = _footprint_frame(START_DATE, (box(-25, -15, 15, 15),))
        assert len(frame) == 1

    Steps:
        1. Measure each polygon with the equal-area projection and derive its dated geometry hash.
        2. Construct a GeoDataFrame with explicit role areas, geographic geometry columns, and stable empty-column types.
    """
    areas = [
        float(transform(_AREA_TRANSFORMER.transform, shape).area / 1_000_000)
        for shape in shapes
    ]
    # Daily footprint IDs use the date and complete WKT with its original order.
    ids = [
        md5(f"{day} {shape.wkt}".encode(), usedforsecurity=False).hexdigest()
        for shape in shapes
    ]
    return gpd.GeoDataFrame(
        {
            "storm_id": pd.Series(ids, dtype="str"),
            "date": pd.Series([day] * len(shapes), dtype="datetime64[ns]"),
            "geometry": gpd.GeoSeries(shapes, crs="EPSG:4326"),
            "grace_geometry": gpd.GeoSeries([None] * len(shapes), crs="EPSG:4326"),
            "buffer_geometry": gpd.GeoSeries(shapes, crs="EPSG:4326"),
            "area_qualified": pd.Series(areas, dtype="float64"),
            "area_grace": pd.Series([0.0] * len(shapes), dtype="float64"),
            "area_gap_filled": pd.Series([0.0] * len(shapes), dtype="float64"),
            "qualified_fraction": pd.Series([1.0] * len(shapes), dtype="float64"),
        },
        geometry="geometry",
        crs="EPSG:4326",
    )


def create_wave_cells(directory: Path) -> tuple[Path, ...]:
    """Write five days of invented mature cells with movement, roles and an end.

    These synthetic cell states cover the input format. They do not
    represent observed temperatures. Qualification, grace and spatial gap
    closing have already happened before this package reads the cell files.

    Args:
        directory: Root directory for the daily synthetic GeoParquets.

    Returns:
        Paths to five daily files in chronological order, with the last empty.

    Examples:
        files = create_wave_cells(Path("./test-wave-cells"))
        assert len(files) == 5

    Steps:
        1. Construct five daily cell groups showing movement, role changes, and an empty ending day.
        2. Write the dated geographic cell files and return their paths in chronological order.
    """
    target = directory / "t2m_max" / "positive"
    target.mkdir(parents=True, exist_ok=True)
    files = []
    for offset in range(5):
        day = START_DATE + timedelta(days=offset)
        positions = (
            [
                (longitude / 4, latitude / 4)
                for longitude in range(-4 + offset, 5 + offset)
                for latitude in range(-3, 4)
            ]
            if offset < 4
            else []
        )
        gap_filled = [
            longitude == offset / 4 and latitude == 0
            for longitude, latitude in positions
        ]
        grace_days = [
            None if filled else int(longitude == (4 + offset) / 4)
            for (longitude, _), filled in zip(positions, gap_filled, strict=True)
        ]
        frame = gpd.GeoDataFrame(
            {
                "longitude": pd.Series(
                    [longitude for longitude, _ in positions], dtype="float64"
                ),
                "latitude": pd.Series(
                    [latitude for _, latitude in positions], dtype="float64"
                ),
                "days": pd.Series(
                    [None if filled else 5 + offset for filled in gap_filled],
                    dtype="float64",
                ),
                "grace_days": pd.Series(grace_days, dtype="float64"),
                "gap_filled": pd.Series(gap_filled, dtype="bool"),
                "geometry": gpd.GeoSeries(
                    [
                        box(
                            longitude - 0.125,
                            latitude - 0.125,
                            longitude + 0.125,
                            latitude + 0.125,
                        )
                        for longitude, latitude in positions
                    ],
                    crs=4326,
                ),
                "date": pd.Series([day] * len(positions), dtype="datetime64[ns]"),
            },
            geometry="geometry",
            crs=4326,
        )
        path = target / f"{day}.parquet"
        WaveCells(frame).to_parquet(path)
        files.append(path)
    return tuple(files)
