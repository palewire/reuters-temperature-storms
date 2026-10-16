"""Read daily mature wave cells and construct their wave footprints."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Self
from uuid import uuid4

import fsspec
import geopandas as gpd
import pandas as pd
from pandera.errors import SchemaError, SchemaErrors
from pyproj import CRS
from shapely.errors import GEOSException

from .footprints import WaveFootprints
from .schemas import WaveCellSchema
from .source import DailyFileSource

if TYPE_CHECKING:
    from datetime import date
    from os import PathLike

    from fsspec.spec import AbstractFileSystem

    from .components import ComponentMembership
    from .series import WaveSeries


class WaveCellError(ValueError):
    """Raised when daily cell rows cannot safely produce wave footprints.

    Steps:
        1. Describe a rejected input, configuration or stage result.
        2. Propagate the message through callers without changing data.

    Args:
        *args: Exception message and any additional exception arguments.

    Examples:
        raise WaveCellError("invalid input")
    """


@dataclass(frozen=True, slots=True)
class WaveCells:
    """One validated daily table of mature, grace and gap-fill wave cells.

    Steps:
        1. Retain the supplied fields for this record.
        2. Expose the values to the corresponding processing and validation methods.

    Args:
        frame: Seven-column GeoDataFrame with exact EPSG:4326 grid polygons.

    Examples:
        cells = WaveCells.read_parquet("2024-07-01.parquet")
        footprints = WaveFootprints.from_cells(cells, day, series)
    """

    frame: gpd.GeoDataFrame
    schema: ClassVar[type[WaveCellSchema]] = WaveCellSchema
    columns: ClassVar[tuple[str, ...]] = (
        "longitude",
        "latitude",
        "days",
        "grace_days",
        "gap_filled",
        "geometry",
        "date",
    )

    def __post_init__(self) -> None:
        """Validate cell roles, grid geometry and types once without changing them.

        Steps:
            1. Check that the seven required columns have the expected types.
            2. Check quarter-degree coordinates and their exact cell polygons,
               including clipped poles and matching halves at the dateline.
            3. Require at least five qualifying days and zero or one grace day
               on real cells; gap-fill cells must have both counts absent.
            4. Keep the supplied values and geometry bytes unchanged, or report
               which cell-table rule failed.

        Args:
            None.

        Returns:
            None. The table retains the caller's values, order and polygon bytes.

        Raises:
            WaveCellError: If the supplied rows do not follow the cell schema.

        Examples:
            cells = WaveCells(frame)
        """
        try:
            self.schema.validate(self.frame)
        except (SchemaError, SchemaErrors) as error:
            raise WaveCellError(f"invalid wave-cell table: {error}") from error

    @classmethod
    def read_parquet(
        cls,
        path: str | PathLike[str],
        *,
        filesystem: AbstractFileSystem | None = None,
    ) -> Self:
        """Read only the daily cell fields needed to build wave footprints.

        Steps:
            1. Select the supplied filesystem, or choose one from the file URL.
            2. Read longitude, latitude, days, grace_days, gap_filled, geometry
               and date. Leave other stored columns unread.
            3. Prepare the date and empty flag types explicitly.
            4. Construct WaveCells to check the table once. A missing file stays
               a missing-file error; malformed contents get a cell-input error.

        Args:
            path: Local file or URL on the caller's configured filesystem.
            filesystem: Optional filesystem with access to the supplied path.

        Returns:
            Once-validated cells, preserving geometry and supplied role counts.
            Additional stored measurements are omitted while reading the file.

        Raises:
            FileNotFoundError: If the file does not exist.
            WaveCellError: If required fields, geometry metadata or values are invalid.

        Examples:
            cells = WaveCells.read_parquet("2024-07-01.parquet")
        """
        if filesystem is None:
            filesystem, location = fsspec.core.url_to_fs(str(path))
        else:
            location = str(path)
        stream = filesystem.open(location, "rb")
        try:
            with stream:
                frame = gpd.read_parquet(
                    stream,
                    columns=list(cls.columns),
                    to_pandas_kwargs={"date_as_object": False},
                )
            return cls(cls.prepare_frame(frame))
        except (ValueError, TypeError, AttributeError, GEOSException) as error:
            raise WaveCellError(
                f"{path}: invalid GeoParquet wave-cell input: {error}"
            ) from error

    @staticmethod
    def prepare_frame(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Prepare decoded calendar dates and the established empty flag dtype.

        Steps:
            1. Require a geometry column using longitude and latitude coordinates
               in EPSG:4326. Do not transform or replace the polygons.
            2. Reject dates with a timezone or a non-date type, then store accepted
               dates at nanosecond precision. The schema checks midnight values.
            3. If there are no rows, change an empty floating gap-fill column to
               boolean. Never turn nonempty numeric flags into booleans.
            4. Return the same frame without changing cell counts or geometry.

        Args:
            frame: Cell fields read by GeoPandas from one daily GeoParquet file.

        Returns:
            The supplied frame with nanosecond calendar dates. An empty
            floating gap-fill column becomes boolean; nonempty flags are
            never converted. Geometry and scientific counts stay unchanged.

        Raises:
            WaveCellError: If geometry metadata or the date dtype is invalid.

        Examples:
            frame = WaveCells.prepare_frame(gpd.read_parquet("2024-07-01.parquet"))
        """
        geometry = frame.get("geometry")
        if not isinstance(geometry, gpd.GeoSeries):
            raise WaveCellError("geometry must be a GeoParquet geometry column")
        if geometry.crs is None or not geometry.crs.equals(
            CRS.from_epsg(4326), ignore_axis_order=True
        ):
            raise WaveCellError("geometry must use the EPSG:4326 coordinate system")
        if "date" in frame.columns:
            dates = frame["date"]
            if isinstance(dates.dtype, pd.DatetimeTZDtype):
                raise WaveCellError("date must be a timezone-free timestamp column")
            if not pd.api.types.is_datetime64_dtype(dates.dtype):
                raise WaveCellError(
                    "date must be a date or timezone-free timestamp column"
                )
            frame["date"] = pd.to_datetime(dates).astype("datetime64[ns]")
        if (
            frame.empty
            and "gap_filled" in frame.columns
            and pd.api.types.is_float_dtype(frame["gap_filled"].dtype)
        ):
            frame["gap_filled"] = frame["gap_filled"].astype(bool)
        return frame

    def to_parquet(
        self,
        path: str | PathLike[str],
        *,
        filesystem: AbstractFileSystem | None = None,
    ) -> None:
        """Write the already validated daily cells as a standalone GeoParquet.

        Steps:
            1. Select the supplied filesystem, or choose one from the file URL.
            2. Create the destination's parent directory if it is needed.
            3. Encode the accepted cell table with its geographic metadata and
               without storing its row index as another column.
            4. For a local file, write a temporary sibling and replace the target
               only after the write succeeds. For a remote file, upload complete
               encoded bytes through the supplied filesystem.

        Args:
            path: Local destination or URL on the caller's filesystem.
            filesystem: Optional configured filesystem for the destination.

        Returns:
            None. Writes the cell fields and EPSG:4326 geometry metadata.

        Raises:
            OSError: If the destination cannot be written.

        Examples:
            cells.to_parquet("wave-cells/2024-07-01.parquet")
        """
        if filesystem is None:
            filesystem, location = fsspec.core.url_to_fs(str(path))
        else:
            location = str(path)
        parent = location.rpartition("/")[0]
        if parent:
            filesystem.makedirs(parent, exist_ok=True)
        # Encode before touching the destination so a failed write can be retried.
        encoded = io.BytesIO()
        self.frame.to_parquet(encoded, index=False)
        payload = encoded.getvalue()
        protocol = filesystem.protocol
        local = protocol == "file" or (
            isinstance(protocol, tuple) and "file" in protocol
        )
        if local:
            target = Path(location)
            temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
            try:
                temporary.write_bytes(payload)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            filesystem.pipe_file(location, payload)


class WaveCellSource(DailyFileSource):
    """Read finished daily wave cells through the shared source file layout.

    Steps:
        1. Configure the caller-selected file location and access options.
        2. Resolve and validate daily files through the source's read methods.

    Args:
        location: Local directory or fsspec URL for the daily files.
        storage_options: Caller-selected filesystem access settings.
        known_empty_dates: Explicit dates permitted to have missing empty files.

    Examples:
        components = WaveCellSource("./cells").read(day, series)
    """

    __slots__ = ()
    kind: ClassVar[str] = "wave-cells"
    error_type: ClassVar[type[ValueError]] = WaveCellError

    def read(self, day: date, series: WaveSeries) -> tuple[ComponentMembership, ...]:
        """Build tracker components from one existing mature-cell file.

        Steps:
            1. Check the requested date and series and select their daily path.
            2. Read and check the cell table. Allow a missing file only when that
               date was explicitly declared empty.
            3. Reject nonempty files on dates declared empty.
            4. Build the wave footprints, check their date and geometry-based
               identifiers, and return the daily shapes needed by the tracker.

        Args:
            day: Requested plain calendar date.
            series: Maximum/minimum and warm/cold scientific series.

        Returns:
            Daily components produced by the same wave-footprint builder. A
            declared missing empty date returns an empty tuple.

        Raises:
            FileNotFoundError: If an undeclared daily cell file is missing.
            WaveCellError: If an input or declared empty date contains invalid rows.
            FootprintError: If footprint construction or identity fails.

        Examples:
            source = WaveCellSource("/data/wave-cells/era5")
            components = source.read(date(2024, 7, 1), series)
        """
        path = self.path_for(day, series)
        try:
            cells = WaveCells.read_parquet(path, filesystem=self._filesystem)
        except FileNotFoundError:
            if day in self.known_empty_dates:
                return ()
            raise
        if day in self.known_empty_dates and not cells.frame.empty:
            raise WaveCellError(f"{path}: declared empty date contains wave-cell rows")
        return WaveFootprints.from_cells(cells, day, series).to_components(day, series)
