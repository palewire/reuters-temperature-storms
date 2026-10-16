"""Readable schemas for the footprint, wave and editorial-name files."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime  # noqa: TC003 - Pandera resolves schema annotations.
from decimal import Decimal, InvalidOperation
from math import isfinite
from typing import Any, ClassVar, cast

import geopandas as gpd  # noqa: TC002 - Pandera resolves runtime annotations.
import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.engines.geopandas_engine import Geometry  # noqa: TC002 - schema dtype.
from pandera.typing import Series  # noqa: TC002 - schema annotations.
from pyproj import CRS
from shapely import bounds, box, equals

from .series import WaveSeries

_WAVE_ID = r"^[0-9]{8}-[0-9a-f]{8}$"
_FOOTPRINT_ID = r"^[0-9a-f]{32}$"


class FileSchema(pa.DataFrameModel):
    """Require declared columns and types without changing incoming values.

    Steps:
        1. Declare the accepted columns and validation rules.
        2. Apply those rules when validate is called, preserving incoming values.

    Args:
        None. Use class-level validation methods rather than constructing records.

    Examples:
        FileSchema.to_schema()
    """

    class Config:
        """Shared file validation settings.

        Steps:
            1. Declare validation settings used by the surrounding schema.
            2. Apply these settings when the schema validates a table.

        Args:
            None. Use class-level validation methods rather than constructing records.

        Examples:
            settings = FileSchema.Config
        """

        strict: ClassVar[bool] = True
        coerce: ClassVar[bool] = False
        unique_column_names: ClassVar[bool] = True

    @pa.check(
        r"^(date|start_date|peak_date|last_observed_date|end_date|event_date)$",
        regex=True,
        name="calendar_dates",
    )
    @classmethod
    def calendar_dates(cls, values: pd.Series) -> pd.Series:
        """Require timestamps at midnight, preserving their timezone-free dtype.

        Steps:
            1. Read each accepted timestamp without changing its stored type.
            2. Compare it with midnight on the same date and mark timestamps with a
               time of day as invalid.

        Args:
            values: Nonmissing values from one declared date column.

        Returns:
            Whether each timestamp represents an exact calendar date.

        Examples:
            FileSchema.calendar_dates(pd.Series(pd.to_datetime(["2020-01-01"])))
        """
        return values.eq(values.dt.normalize())

    @pa.check(
        r"^(qualified_area_km2|grace_area_km2|gap_filled_area_km2|peak_qualified_area_km2|total_area_visited_km2|overlap_percent|qualified_fraction|area_qualified|area_grace|area_gap_filled)$",
        regex=True,
        name="finite_numbers",
    )
    @classmethod
    def finite_numbers(cls, values: pd.Series) -> pd.Series:
        """Reject infinite scientific measurements without rounding them.

        Steps:
            1. Visit each nonmissing measurement.
            2. Reject infinite or otherwise nonfinite values without rounding the
               stored measurements.

        Args:
            values: Nonmissing floating-point measurements.

        Returns:
            Whether each value is finite.

        Examples:
            FileSchema.finite_numbers(pd.Series([1.0, float("inf")]))
        """
        return values.map(isfinite)

    @pa.check(
        r"^(geometry|grace_geometry|buffer_geometry|qualified_geometry|full_geometry|lifetime_geometry)$",
        regex=True,
        name="geographic_polygons",
    )
    @classmethod
    def geographic_polygons(cls, values: gpd.GeoSeries) -> pd.Series:
        """Require unchanged valid two-dimensional geographic polygons.

        Steps:
            1. Require the longitude and latitude coordinate system EPSG:4326.
            2. For each shape, require a nonempty valid polygon or group of polygons
               with neither height nor another measured coordinate.
            3. Require finite bounds inside -180 to +180 longitude and -90 to +90
               latitude, then return one result per shape.

        Args:
            values: Nonmissing shapes with their GeoPandas geometry dtype.

        Returns:
            Whether each shape is valid, nonempty, polygonal and inside the
            longitude/latitude range, with a geographic CRS.

        Examples:
            FileSchema.geographic_polygons(gpd.GeoSeries([], crs="EPSG:4326"))
        """
        crs = getattr(values, "crs", None)
        if crs is None or not CRS(crs).equals(CRS("EPSG:4326"), ignore_axis_order=True):
            return pd.Series(False, index=values.index)
        return values.map(
            lambda shape: (
                not shape.is_empty
                and shape.is_valid
                and shape.geom_type in {"Polygon", "MultiPolygon"}
                and not shape.has_z
                and not shape.has_m
                and all(isfinite(value) for value in shape.bounds)
                and -180 <= shape.bounds[0] <= shape.bounds[2] <= 180
                and -90 <= shape.bounds[1] <= shape.bounds[3] <= 90
            )
        )

    @pa.check(
        r"^(parent_wave_ids|parent_keys)$",
        regex=True,
        name="wave_id_lists",
    )
    @classmethod
    def wave_id_lists(cls, values: pd.Series) -> pd.Series:
        """Require lists of permanent wave IDs, including empty lists.

        Steps:
            1. Check each stored list of related wave identifiers independently.
            2. Reject scalar strings, bytes and mappings in place of a list.
            3. Require distinct identifiers in YYYYMMDD-xxxxxxxx form, retaining the
               supplied list order and values.

        Args:
            values: Object-valued parent or merging-wave list column.

        Returns:
            Whether each list contains only well-formed distinct wave IDs.

        Examples:
            FileSchema.wave_id_lists(pd.Series([[], ["20200101-01234567"]]))
        """

        def valid(value: object) -> bool:
            """Check one list without changing its representation or order.

            Steps:
                1. Require an iterable value that is neither text, bytes nor a mapping.
                2. Read its entries and check eight date digits, one hyphen and eight
                   lowercase hexadecimal characters in each identifier.
                3. Require no repeated identifiers and return the result without changing
                   the original value.

            Args:
                value: One list-like value read from a Parquet list column.

            Returns:
                Whether its entries are distinct permanent wave IDs.

            Examples:
                valid(["20200101-01234567"])
            """
            if not isinstance(value, Iterable) or isinstance(value, str | bytes | dict):
                return False
            keys = list(value)
            return all(
                isinstance(key, str)
                and len(key) == 17
                and key[:8].isdigit()
                and key[8] == "-"
                and all(letter in "0123456789abcdef" for letter in key[9:])
                for key in keys
            ) and len(set(keys)) == len(keys)

        return values.map(valid)


class WaveFootprintSchema(FileSchema):
    """Daily GeoParquet inputs: exact role shapes and supplied physical areas.

    Steps:
        1. Declare the accepted columns and validation rules.
        2. Apply those rules when validate is called, preserving incoming values.

    Args:
        None. Use class-level validation methods rather than constructing records.

    Examples:
        WaveFootprintSchema.to_schema()
    """

    storm_id: Series[str] = pa.Field(str_matches=_FOOTPRINT_ID, unique=True)
    date: Series[datetime]
    geometry: Series[Geometry] = pa.Field(nullable=True)
    buffer_geometry: Series[Geometry]
    grace_geometry: Series[Geometry] = pa.Field(nullable=True)
    area_qualified: Series[Any] = pa.Field(ge=0)
    area_grace: Series[Any] = pa.Field(ge=0)
    area_gap_filled: Series[Any] = pa.Field(ge=0)
    qualified_fraction: Series[Any] = pa.Field(ge=0, le=1)

    @pa.check("area_qualified", "area_grace", "area_gap_filled", "qualified_fraction")
    @classmethod
    def floating_measurements(cls, values: pd.Series) -> bool:
        """Require floating-point inputs while accepting their stored precision.

        Steps:
            1. Inspect the column type rather than converting its values.
            2. Accept a floating-point column at its stored precision; reject other
               measurement types.

        Args:
            values: Supplied footprint measurements in a pandas Series.

        Returns:
            Whether the stored dtype is floating point, without casting it.

        Examples:
            WaveFootprintSchema.floating_measurements(pd.Series([1.0]))
        """
        return pd.api.types.is_float_dtype(values.dtype)

    @pa.dataframe_check
    @classmethod
    def qualified_fraction_and_role_areas(cls, frame: pd.DataFrame) -> pd.Series:
        """Require positive totals, matching fractions and role-shape presence.

        Steps:
            1. Add qualified, grace and gap-fill areas for each daily footprint.
            2. Require a positive finite total and compare the supplied qualified
               fraction with qualified area divided by that total, allowing
               only the stated floating-point tolerance.
            3. Require a missing qualified or grace shape exactly when its role area
               is zero. Require at least one of those two shapes to be present.
            4. Combine those checks into one result per footprint without replacing
               its stored shapes or measurements.

        Args:
            frame: Validly typed daily footprint rows.

        Returns:
            Whether each row's roles, areas and fraction agree.

        Examples:
            WaveFootprintSchema.qualified_fraction_and_role_areas(footprints)
        """
        areas = frame[["area_qualified", "area_grace", "area_gap_filled"]].astype(float)
        total = areas.sum(axis=1)
        fractions = pd.Series(
            [
                abs(fraction - area / size) <= 1e-9 + 1e-9 * abs(area / size)
                if size > 0 and isfinite(size)
                else False
                for fraction, area, size in zip(
                    frame["qualified_fraction"],
                    areas["area_qualified"],
                    total,
                    strict=True,
                )
            ],
            index=frame.index,
            dtype=bool,
        )
        return cast(
            "pd.Series",
            (
                total.gt(0)
                & fractions
                & frame["geometry"].isna().eq(frame["area_qualified"].eq(0))
                & frame["grace_geometry"].isna().eq(frame["area_grace"].eq(0))
                & ~(frame["geometry"].isna() & frame["grace_geometry"].isna())
            ),
        )


class WaveDayFileSchema(FileSchema):
    """Daily wave GeoParquet columns before dataset partitions.

    ``wave_id`` follows the wave across dates; ``component_id`` identifies
    only the observed footprint on this date.

    Steps:
        1. Declare the accepted columns and validation rules.
        2. Apply those rules when validate is called, preserving incoming values.

    Args:
        None. Use class-level validation methods rather than constructing records.

    Examples:
        WaveDayFileSchema.to_schema()
    """

    wave_id: Series[str] = pa.Field(str_matches=_WAVE_ID)
    component_id: Series[str] = pa.Field(str_matches=_FOOTPRINT_ID)
    name: Series[str] = pa.Field(nullable=True, str_length={"min_value": 1})
    date: Series[datetime]
    start_date: Series[datetime]
    component_label: Series[int] = pa.Field(ge=1)
    continues: Series[bool]
    overlap_percent: Series[float] = pa.Field(ge=0, le=100)
    parent_wave_ids: Series[object]
    parent_keys: Series[object]
    peak_qualified_area_km2: Series[float] = pa.Field(ge=0)
    peak_date: Series[datetime]
    qualified_area_km2: Series[float] = pa.Field(ge=0)
    grace_area_km2: Series[float] = pa.Field(ge=0)
    gap_filled_area_km2: Series[float] = pa.Field(ge=0)
    qualified_fraction: Series[float] = pa.Field(ge=0, le=1)
    qualified_geometry: Series[Geometry] = pa.Field(nullable=True)
    grace_geometry: Series[Geometry] = pa.Field(nullable=True)
    full_geometry: Series[Geometry]

    @pa.dataframe_check
    @classmethod
    def wave_dates(cls, frame: pd.DataFrame) -> pd.Series:
        """Require recorded dates inside the wave's history.

        Steps:
            1. Compare the wave start date with its peak date.
            2. Require the peak date to be no later than this observation date, so
               both belong to the history recorded by this row.

        Args:
            frame: One validated daily wave file.

        Returns:
            Whether each recorded date is inside the wave history.

        Examples:
            WaveDayFileSchema.wave_dates(wave_days)
        """
        return cast(
            "pd.Series",
            frame["start_date"].le(frame["peak_date"])
            & frame["peak_date"].le(frame["date"]),
        )


class WaveDaySchema(WaveDayFileSchema):
    """Daily wave DataFrame with the selected series and year partitions.

    Steps:
        1. Declare the accepted columns and validation rules.
        2. Apply those rules when validate is called, preserving incoming values.

    Args:
        None. Use class-level validation methods rather than constructing records.

    Examples:
        WaveDaySchema.to_schema()
    """

    temperature: Series[str] = pa.Field(isin=WaveSeries.temperatures)
    direction: Series[str] = pa.Field(isin=WaveSeries.directions)
    observation_year: Series[int] = pa.Field(ge=1, le=9999)

    @pa.dataframe_check
    @classmethod
    def observation_partition(cls, frame: pd.DataFrame) -> pd.Series:
        """Match each observation-year partition to its scientific date.

        Steps:
            1. Read the calendar year from each observation date.
            2. Require the year encoded in the observation directory to match it.

        Args:
            frame: Daily wave rows including partitions.

        Returns:
            Whether the recorded partition year matches each date.

        Examples:
            WaveDaySchema.observation_partition(wave_days)
        """
        return cast("pd.Series", frame["observation_year"].eq(frame["date"].dt.year))


class WaveCatalogFileSchema(FileSchema):
    """One row per permanent wave ID with qualified lifetime footprint and area.

    Steps:
        1. Declare the accepted columns and validation rules.
        2. Apply those rules when validate is called, preserving incoming values.

    Args:
        None. Use class-level validation methods rather than constructing records.

    Examples:
        WaveCatalogFileSchema.to_schema()
    """

    wave_id: Series[str] = pa.Field(str_matches=_WAVE_ID, unique=True)
    start_date: Series[datetime]
    last_observed_date: Series[datetime]
    end_date: Series[datetime] = pa.Field(nullable=True)
    end_reason: Series[str] = pa.Field(nullable=True, isin=["died", "merged"])
    merged_into_wave_id: Series[str] = pa.Field(nullable=True, str_matches=_WAVE_ID)
    active: Series[bool]
    peak_qualified_area_km2: Series[float] = pa.Field(gt=0)
    peak_date: Series[datetime]
    observation_count: Series[int] = pa.Field(ge=1)
    name: Series[str] = pa.Field(nullable=True, str_length={"min_value": 1})
    duration_days: Series[int] = pa.Field(ge=1)
    parent_keys: Series[object]
    total_area_visited_km2: Series[float] = pa.Field(gt=0)
    lifetime_geometry: Series[Geometry]

    @pa.dataframe_check
    @classmethod
    def catalog_history(cls, frame: pd.DataFrame) -> pd.Series:
        """Require coherent dates, inclusive duration and ending fields.

        Steps:
            1. Count calendar days from the start through the last observation,
               including both dates.
            2. Require the peak date to fall within that span, duration to equal
               the span, and the observation count not to exceed it.
            3. Require active waves to have no ending date or reason. Require ended
               waves to have both, with the end date matching the last observation.
            4. Require a destination wave identifier exactly when the ending reason
               is a merge, then return one combined result per catalog row.

        Args:
            frame: Catalog rows describing active and ended waves.

        Returns:
            Whether each row's stored history and ending state agree.

        Examples:
            WaveCatalogFileSchema.catalog_history(wave_catalog)
        """
        span = (frame["last_observed_date"] - frame["start_date"]).dt.days + 1
        ended = ~frame["active"]
        return cast(
            "pd.Series",
            (
                frame["start_date"].le(frame["peak_date"])
                & frame["peak_date"].le(frame["last_observed_date"])
                & frame["duration_days"].eq(span)
                & frame["observation_count"].le(span)
                & frame["end_date"].notna().eq(ended)
                & frame["end_reason"].notna().eq(ended)
                & (
                    frame["end_date"].isna()
                    | frame["end_date"].eq(frame["last_observed_date"])
                )
                & frame["merged_into_wave_id"]
                .notna()
                .eq(frame["end_reason"].eq("merged"))
            ),
        )


class WaveCatalogSchema(WaveCatalogFileSchema):
    """Per-wave catalog DataFrame with series and start-year partitions.

    Steps:
        1. Declare the accepted columns and validation rules.
        2. Apply those rules when validate is called, preserving incoming values.

    Args:
        None. Use class-level validation methods rather than constructing records.

    Examples:
        WaveCatalogSchema.to_schema()
    """

    temperature: Series[str] = pa.Field(isin=WaveSeries.temperatures)
    direction: Series[str] = pa.Field(isin=WaveSeries.directions)
    start_year: Series[int] = pa.Field(ge=1, le=9999)

    @pa.dataframe_check
    @classmethod
    def start_partition(cls, frame: pd.DataFrame) -> pd.Series:
        """Match each start-year partition to its birth date.

        Steps:
            1. Read the calendar year from the wave start date.
            2. Require the year encoded in the start directory to match it.

        Args:
            frame: Wave catalog rows including partitions.

        Returns:
            Whether each partition year matches the birth date.

        Examples:
            WaveCatalogSchema.start_partition(wave_catalog)
        """
        return cast("pd.Series", frame["start_year"].eq(frame["start_date"].dt.year))


class NamedStormsSchema(WaveCatalogSchema):
    """Self-contained named-storm catalog, retaining all physical catalog checks.

    Steps:
        1. Declare the accepted columns and validation rules.
        2. Apply those rules when validate is called, preserving incoming values.

    Args:
        None. Use class-level validation methods rather than constructing records.

    Examples:
        NamedStormsSchema.to_schema()
    """

    name: Series[str] = pa.Field(str_length={"min_value": 1})


class LineageFileSchema(FileSchema):
    """Split and merge lineage Parquet columns, before dataset partitions.

    Steps:
        1. Declare the accepted columns and validation rules.
        2. Apply those rules when validate is called, preserving incoming values.

    Args:
        None. Use class-level validation methods rather than constructing records.

    Examples:
        LineageFileSchema.to_schema()
    """

    source_wave_id: Series[str] = pa.Field(str_matches=_WAVE_ID)
    target_wave_id: Series[str] = pa.Field(str_matches=_WAVE_ID)
    event_date: Series[datetime]
    relationship: Series[str] = pa.Field(isin=["split", "merge"])

    @pa.dataframe_check
    @classmethod
    def distinct_waves(cls, frame: pd.DataFrame) -> pd.Series:
        """Reject relationships from a wave to itself.

        Steps:
            1. Read the source and destination wave identifiers in each relationship.
            2. Reject a relationship whose two identifiers refer to the same wave.

        Args:
            frame: Split and merge rows.

        Returns:
            Whether every row connects two different waves.

        Examples:
            LineageFileSchema.distinct_waves(lineage)
        """
        return cast("pd.Series", frame["source_wave_id"].ne(frame["target_wave_id"]))


class LineageSchema(LineageFileSchema):
    """Split and merge DataFrame with series and event-year partitions.

    Steps:
        1. Declare the accepted columns and validation rules.
        2. Apply those rules when validate is called, preserving incoming values.

    Args:
        None. Use class-level validation methods rather than constructing records.

    Examples:
        LineageSchema.to_schema()
    """

    temperature: Series[str] = pa.Field(isin=WaveSeries.temperatures)
    direction: Series[str] = pa.Field(isin=WaveSeries.directions)
    event_year: Series[int] = pa.Field(ge=1, le=9999)

    @pa.dataframe_check
    @classmethod
    def event_partition(cls, frame: pd.DataFrame) -> pd.Series:
        """Match each event-year partition to its lineage date.

        Steps:
            1. Read the calendar year from the split or merge date.
            2. Require the year encoded in the event directory to match it.

        Args:
            frame: Lineage rows including partitions.

        Returns:
            Whether the partition year matches the event date.

        Examples:
            LineageSchema.event_partition(lineage)
        """
        return cast("pd.Series", frame["event_year"].eq(frame["event_date"].dt.year))


class NameLibrarySchema(FileSchema):
    """Ordered UTF-8 CSV fields; group decimals retain their source spelling.

    Steps:
        1. Declare the accepted columns and validation rules.
        2. Apply those rules when validate is called, preserving incoming values.

    Args:
        None. Use class-level validation methods rather than constructing records.

    Examples:
        NameLibrarySchema.to_schema()
    """

    geography: Series[str] = pa.Field(str_matches=r".*\S.*")
    gender: Series[str] = pa.Field(isin=["F", "M"])
    name: Series[str] = pa.Field(str_matches=r".*\S.*")
    group: Series[str]

    class Config(FileSchema.Config):
        """Require the declared CSV header order.

        Steps:
            1. Declare validation settings used by the surrounding schema.
            2. Apply these settings when the schema validates a table.

        Args:
            None. Use class-level validation methods rather than constructing records.

        Examples:
            settings = NameLibrarySchema.Config
        """

        ordered: ClassVar[bool] = True

    @pa.check("group")
    @classmethod
    def integral_groups(cls, values: pd.Series) -> pd.Series:
        """Accept finite positive integral decimals without changing text.

        Steps:
            1. Read each group field as a decimal while preserving the CSV text.
            2. Require a finite whole number from 1 through 13 and return one result
               per supplied field.

        Args:
            values: Raw CSV group strings, such as ``"1.0"``.

        Returns:
            Whether each group is an integer in the supported 1-13 range.

        Examples:
            NameLibrarySchema.integral_groups(pd.Series(["1.0", "1.5"]))
        """

        def valid(value: str) -> bool:
            """Check one raw decimal group value.

            Steps:
                1. Parse the supplied field as a decimal; reject text that cannot be read
                   as a number.
                2. Require a finite whole number from 1 through 13 without changing the
                   original field.

            Args:
                value: CSV field exactly as stored.

            Returns:
                Whether the field names a supported integral group.

            Examples:
                valid("1.0")
            """
            try:
                number = Decimal(value)
            except InvalidOperation:
                return False
            return (
                number.is_finite()
                and 1 <= number <= 13
                and number == number.to_integral_value()
            )

        return values.map(valid)

    @pa.dataframe_check
    @classmethod
    def nonempty_library(cls, frame: pd.DataFrame) -> bool:
        """Reject a CSV with no editorial names.

        Steps:
            1. Inspect whether the parsed name table contains any rows.
            2. Reject an empty table so naming cannot begin without a name library.

        Args:
            frame: Parsed name CSV.

        Returns:
            Whether at least one name is present.

        Examples:
            NameLibrarySchema.nonempty_library(pd.DataFrame())
        """
        return not frame.empty


class GridCellSchema(FileSchema):
    """Unique finite quarter-degree cell centres shared by scores and state.

    Steps:
        1. Declare the accepted columns and validation rules.
        2. Apply those rules when validate is called, preserving incoming values.

    Args:
        None. Use class-level validation methods rather than constructing records.

    Examples:
        GridCellSchema.to_schema()
    """

    longitude: Series[float] = pa.Field(gt=-180, le=180)
    latitude: Series[float] = pa.Field(ge=-90, le=90)

    class Config(FileSchema.Config):
        """Require exactly one history row at each quarter-degree coordinate.

        Steps:
            1. Declare validation settings used by the surrounding schema.
            2. Apply these settings when the schema validates a table.

        Args:
            None. Use class-level validation methods rather than constructing records.

        Examples:
            settings = GridCellSchema.Config
        """

        unique: ClassVar[list[str]] = ["longitude", "latitude"]

    @pa.check("longitude", "latitude")
    @classmethod
    def grid_coordinates(cls, values: pd.Series) -> pd.Series:
        """Require finite centres on the exact quarter-degree grid.

        Steps:
            1. Reject infinite coordinates.
            2. Require four times each coordinate to be a whole number.

        Args:
            values: Longitude or latitude column.

        Returns:
            Whether each coordinate falls on the supported grid.

        Examples:
            GridCellSchema.grid_coordinates(pd.Series([0.0, 0.25]))
        """
        return pd.Series(
            np.isfinite(values) & (values * 4).eq(np.floor(values * 4)),
            index=values.index,
        )


class WaveCellScoreSchema(GridCellSchema):
    """One finite calculated score for every cell in the complete global grid.

    Steps:
        1. Declare the accepted columns and validation rules.
        2. Apply those rules when validate is called, preserving incoming values.

    Args:
        None. Use class-level validation methods rather than constructing records.

    Examples:
        WaveCellScoreSchema.to_schema()
    """

    score: Series[Any]

    @pa.check("score")
    @classmethod
    def finite_scores(cls, values: pd.Series) -> pd.Series | bool:
        """Accept finite native floating scores without rounding the threshold.

        Steps:
            1. Accept 32-bit or 64-bit floating score storage.
            2. Reject missing and infinite scores. Keep accepted values unchanged
               so values close to one are tested at their supplied precision.

        Args:
            values: Calculated warm or cold score column.

        Returns:
            Whether the column type and each score are supported.

        Examples:
            WaveCellScoreSchema.finite_scores(pd.Series([0.99, 1.0]))
        """
        if values.dtype not in (np.dtype("float32"), np.dtype("float64")):
            return False
        return pd.Series(np.isfinite(values), index=values.index)

    @pa.dataframe_check
    @classmethod
    def complete_grid(cls, frame: pd.DataFrame) -> bool:
        """Require every quarter-degree cell once without rebuilding the grid.

        Steps:
            1. The coordinate fields restrict positions to the supported grid,
               and their shared uniqueness rule rejects repeated cell pairs.
            2. Require 1440 longitude positions times 721 latitude positions.
               Those rules together establish complete global coverage.

        Args:
            frame: Three-column score table.

        Returns:
            Whether the table contains exactly the global number of cells.

        Examples:
            WaveCellScoreSchema.complete_grid(scores.frame)
        """
        return len(frame) == 1440 * 721


class WaveCellStateSchema(GridCellSchema):
    """One history per grid cell before mature-cell selection and gap filling.

    Steps:
        1. Declare the accepted columns and validation rules.
        2. Apply those rules when validate is called, preserving incoming values.

    Args:
        None. Use class-level validation methods rather than constructing records.

    Examples:
        WaveCellStateSchema.to_schema()
    """

    start_date: Series[datetime]
    end_date: Series[datetime] = pa.Field(nullable=True)
    days: Series[int] = pa.Field(ge=1)
    grace_days: Series[int] = pa.Field(ge=0, le=1)

    @pa.dataframe_check
    @classmethod
    def ended_cell_state(cls, frame: pd.DataFrame) -> pd.Series:
        """Require ended histories to have no current grace count.

        Steps:
            1. Identify rows carrying an end date.
            2. Require those rows to have zero grace days. Active rows may have
               zero or one, as checked by the grace-days field.

        Args:
            frame: Typed cell-history table.

        Returns:
            Whether each row has an allowed ended or active state.

        Examples:
            WaveCellStateSchema.ended_cell_state(state.frame)
        """
        return cast("pd.Series", frame["end_date"].isna() | frame["grace_days"].eq(0))


class WaveCellSchema(FileSchema):
    """Daily mature wave cells with exact grid polygons and supplied roles.

    Steps:
        1. Declare the accepted columns and validation rules.
        2. Apply those rules when validate is called, preserving incoming values.

    Args:
        None. Use class-level validation methods rather than constructing records.

    Examples:
        WaveCellSchema.to_schema()
    """

    longitude: Series[float] = pa.Field(gt=-180, le=180)
    latitude: Series[float] = pa.Field(ge=-90, le=90)
    days: Series[Any] = pa.Field(nullable=True)
    grace_days: Series[Any] = pa.Field(nullable=True)
    gap_filled: Series[bool]
    geometry: Series[Geometry]
    date: Series[datetime]

    @pa.check("geometry", name="geographic_polygons")
    @classmethod
    def geographic_polygons(cls, values: gpd.GeoSeries) -> pd.Series | bool:
        """Check geographic cell shape metadata with vectorized operations.

        Steps:
            1. Require EPSG:4326 even when the cell table has no rows.
            2. Check the cells together for valid nonempty two-dimensional shapes.
            3. Leave polygon size and location to the exact-cell check, avoiding
               a second calculation of those same bounds here.

        Args:
            values: Nonmissing cell geometries with their GeoPandas CRS.

        Returns:
            Whether each cell is valid, nonempty and two-dimensional in
            EPSG:4326. The exact-cell check separately establishes polygon type
            and geographic extent without repeating those calculations here.

        Examples:
            WaveCellSchema.geographic_polygons(gpd.GeoSeries([], crs=4326))
        """
        crs = getattr(values, "crs", None)
        if crs is None or not CRS(crs).equals(CRS("EPSG:4326"), ignore_axis_order=True):
            return False
        return cast(
            "pd.Series",
            values.is_valid & ~values.is_empty & ~values.has_z & ~values.has_m,
        )

    @pa.check("longitude", "latitude")
    @classmethod
    def grid_coordinates(cls, values: pd.Series) -> pd.Series:
        """Require finite cell centres at exact quarter-degree positions.

        Steps:
            1. Require each coordinate to be finite.
            2. Multiply each coordinate by four and require a whole number, so
               its position lies exactly on the 0.25-degree grid.

        Args:
            values: Longitude or latitude coordinates.

        Returns:
            Whether each coordinate belongs to the 0.25-degree grid.

        Examples:
            WaveCellSchema.grid_coordinates(pd.Series([0.0, 0.25, 0.3]))
        """
        return pd.Series(
            np.isfinite(values) & (values * 4).eq(np.floor(values * 4)),
            index=values.index,
        )

    @pa.check("days", "grace_days")
    @classmethod
    def numeric_counts(cls, values: pd.Series) -> bool:
        """Accept numeric count storage without casting scientific values.

        Steps:
            1. Inspect the stored count type without changing any values.
            2. Accept integer or floating-point columns and reject booleans.
               Floating-point columns can hold missing counts for gap-fill cells.

        Args:
            values: Qualifying or grace counts, nullable for gap-fill rows.

        Returns:
            Whether the column has an integer or floating dtype, excluding booleans.

        Examples:
            WaveCellSchema.numeric_counts(pd.Series([5.0, float("nan")]))
        """
        return (
            pd.api.types.is_integer_dtype(values.dtype)
            or pd.api.types.is_float_dtype(values.dtype)
        ) and not pd.api.types.is_bool_dtype(values.dtype)

    @pa.dataframe_check
    @classmethod
    def mature_cell_state(cls, frame: pd.DataFrame) -> pd.Series:
        """Require mature real cells and null counts on added gap-fill cells.

        Steps:
            1. Read qualifying and grace counts into temporary numeric arrays,
               leaving the stored table unchanged.
            2. For each real cell, require a finite whole qualifying-day count
               of at least five and a finite grace-day count of zero or one.
            3. For each added gap-fill cell, require both counts to be absent.
            4. Use the supplied gap-fill flag to choose the relevant checks for
               each row. Do not calculate qualification from a score.

        Args:
            frame: Typed daily cell table.

        Returns:
            Whether each real cell has integral qualifying counts of at least five
            and zero or one grace day; added cells must have both counts absent.

        Examples:
            WaveCellSchema.mature_cell_state(cells)
        """
        days = pd.to_numeric(frame["days"], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        grace = pd.to_numeric(frame["grace_days"], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        valid_real = (
            np.isfinite(days)
            & (days >= 5)
            & (days == np.floor(days))
            & np.isfinite(grace)
            & (grace >= 0)
            & (grace <= 1)
            & (grace == np.floor(grace))
        )
        valid_fill = frame["days"].isna() & frame["grace_days"].isna()
        return pd.Series(
            np.where(frame["gap_filled"], valid_fill, valid_real), index=frame.index
        )

    @pa.dataframe_check
    @classmethod
    def exact_cell_polygons(cls, frame: pd.DataFrame) -> pd.Series:
        """Require the complete clipped cell box at each supplied coordinate.

        Steps:
            1. Read the supplied cell centres and the stored polygon bounds.
            2. Build each expected box with edges 0.125 degrees from its centre,
               clipping latitude at the poles.
            3. For a +180-degree cell, accept its eastern box ending at +180 or
               its western box beginning at -180. The paired-row check requires
               both pieces for that latitude.
            4. Compare covered shapes with the expected boxes without replacing
               the stored polygons or changing their coordinate order.

        Args:
            frame: Daily cells with EPSG:4326 polygons.

        Returns:
            Whether each polygon equals its quarter-degree cell box, clipped at
            the poles. A +180 cell may hold either exact dateline half here; the
            paired-row check requires both halves.

        Examples:
            WaveCellSchema.exact_cell_polygons(cells)
        """
        longitude = frame["longitude"].to_numpy(dtype=float)
        latitude = frame["latitude"].to_numpy(dtype=float)
        seam = longitude == 180
        actual_bounds = bounds(frame["geometry"].array)
        west = seam & (actual_bounds[:, 0] == -180)
        minimum_x = np.where(west, -180, longitude - 0.125)
        maximum_x = np.where(west, -179.875, np.minimum(longitude + 0.125, 180))
        expected = box(
            minimum_x,
            np.maximum(latitude - 0.125, -90),
            maximum_x,
            np.minimum(latitude + 0.125, 90),
        )
        return pd.Series(equals(frame["geometry"].array, expected), index=frame.index)

    @pa.dataframe_check
    @classmethod
    def coordinate_rows(cls, frame: pd.DataFrame) -> bool:
        """Allow duplicate coordinates only for one complete matching seam pair.

        Steps:
            1. Require each ordinary longitude/latitude pair to occur once.
            2. Collect rows labelled +180 by latitude and require exactly two rows
               for each latitude: one eastern half and one western half.
            3. Require both halves to agree on qualifying days, grace days, the
               gap-fill flag and represented date, including missing counts.
            4. Return whether all row relationships agree, retaining both halves
               of each accepted dateline cell.

        Args:
            frame: Daily cell rows with validated box geometries.

        Returns:
            Whether ordinary coordinates occur once and each +180 latitude has
            exactly one east and one west half with matching state and date.

        Examples:
            WaveCellSchema.coordinate_rows(cells)
        """
        ordinary = frame.loc[frame["longitude"].ne(180)]
        if ordinary.duplicated(["longitude", "latitude"]).any():
            return False
        seam = frame.loc[frame["longitude"].eq(180)].copy()
        if seam.empty:
            return True
        seam["western_half"] = bounds(seam["geometry"].array)[:, 0] == -180
        grouped = seam.groupby("latitude", sort=False)
        sizes = grouped.size()
        western = grouped["western_half"].sum()
        matching = grouped[["days", "grace_days", "gap_filled", "date"]].nunique(
            dropna=False
        )
        return bool(
            sizes.eq(2).all() and western.eq(1).all() and matching.eq(1).all().all()
        )
