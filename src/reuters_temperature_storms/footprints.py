"""Read supplied daily footprint shapes."""

from __future__ import annotations

import io
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, NamedTuple, Self, cast
from uuid import uuid4

import fsspec
import geopandas as gpd
import pandas as pd
from pandera.errors import SchemaError, SchemaErrors
from pyproj import CRS
from shapely.errors import GEOSException

from .components import (
    ComponentMembership,
    DailyWaveComponentGeometry,
)
from .footprint_geometry import WaveFootprintBuilder
from .footprint_order import FootprintOrderError, HistoricalFootprintOrder
from .identity import component_identity
from .schemas import WaveFootprintSchema
from .series import WaveSeries
from .source import DailyFileSource

if TYPE_CHECKING:
    from os import PathLike

    from fsspec.spec import AbstractFileSystem
    from shapely import Geometry

    from .cells import WaveCells, WaveCellSource


class _FootprintRow(NamedTuple):
    """Typed row attributes after the footprint schema accepts a frame.

    Steps:
        1. Retain the supplied fields for this record.
        2. Expose the values to the corresponding processing and validation methods.

    Args:
        storm_id: Identifier of the daily footprint.
        date: Observation date.
        geometry: Qualified mature core geometry, or None when absent.
        grace_geometry: Active mature geometry currently in its grace period, or None.
        buffer_geometry: Full daily geometry including gap-filled cells.
        area_qualified: Qualified-core area in square kilometres.
        area_grace: Grace-only area in square kilometres.
        area_gap_filled: Gap-filled area in square kilometres.
        qualified_fraction: Qualified area divided by full area.

    Examples:
        row = _FootprintRow(identifier, day, core, grace, full, 10.0, 2.0, 1.0, 10.0 / 13.0)
    """

    storm_id: str
    date: pd.Timestamp
    geometry: Geometry | None
    grace_geometry: Geometry | None
    buffer_geometry: Geometry
    area_qualified: float
    area_grace: float
    area_gap_filled: float
    qualified_fraction: float


class FootprintError(ValueError):
    """Raised when stored daily footprints cannot be used safely.

    Steps:
        1. Describe a rejected input, configuration or stage result.
        2. Propagate the message through callers without changing data.

    Args:
        *args: Exception message and any additional exception arguments.

    Examples:
        raise FootprintError("invalid input")
    """


@dataclass(frozen=True, slots=True)
class WaveFootprints:
    """One validated daily footprint input table with its existing schema.

    The input column ``storm_id`` identifies the stored row and any associated
    cell records. Tracking derives its own ``component_id`` from the physical
    coverage and retains that input label separately. Tracked outputs use the
    persistent ``wave_id``, which stays the same when a wave becomes named.

    Steps:
        1. Keep the accepted daily table on the model without copying it.
        2. Check its fields and role measurements once when the model is created.
        3. Read, write or convert the same shapes through the methods on this class.

    Args:
        frame: GeoDataFrame with the required footprint columns and types.

    Examples:
        footprints = WaveFootprints.read_parquet("2026-08-31.parquet")
        components = footprints.to_components(date(2026, 8, 31), series, native_cells=cells)
    """

    frame: gpd.GeoDataFrame
    schema: ClassVar[type[WaveFootprintSchema]] = WaveFootprintSchema

    def __post_init__(self) -> None:
        """Apply the existing footprint schema once without altering the frame.

        Steps:
            1. Check field types, required values and relationships between shapes and areas.
            2. Report a table error if a check fails, retaining the original frame otherwise.

        Args:
            None.

        Returns:
            None. Retains the caller's frame without copies or conversions.

        Raises:
            FootprintError: If the supplied frame violates the footprint schema.

        Examples:
            footprints = WaveFootprints(frame)
        """
        try:
            self.schema.validate(self.frame)
        except (SchemaError, SchemaErrors) as error:
            raise FootprintError(f"invalid footprint table: {error}") from error

    @classmethod
    def read_parquet(
        cls,
        path: str | PathLike[str],
        *,
        filesystem: AbstractFileSystem | None = None,
    ) -> Self:
        """Decode a footprint file and validate its prepared table once.

        Steps:
            1. Use the supplied filesystem, or choose one from the input path.
            2. Read the GeoParquet file and close its stream before returning.
            3. Prepare geography and dates, then check the resulting table once.
            4. Include the source path when reporting a file or table error.

        Args:
            path: Local file path or caller-selected filesystem URL.
            filesystem: Optional configured filesystem for the source file.

        Returns:
            Validated footprint table preserving exact geometry and measurements.

        Raises:
            FileNotFoundError: If the input file is missing.
            FootprintError: If geographic metadata, dates or columns are invalid.

        Examples:
            footprints = WaveFootprints.read_parquet("2026-08-31.parquet")
        """
        if filesystem is None:
            filesystem, location = fsspec.core.url_to_fs(str(path))
        else:
            location = str(path)
        stream = filesystem.open(location, "rb")
        try:
            with stream:
                frame = gpd.read_parquet(
                    stream, to_pandas_kwargs={"date_as_object": False}
                )
            return cls(cls.prepare_frame(frame))
        except (ValueError, TypeError, AttributeError, GEOSException) as error:
            raise FootprintError(
                f"{path}: invalid GeoParquet footprint input: {error}"
            ) from error

    @staticmethod
    def prepare_frame(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Prepare geographic columns and calendar dates for the input schema.

        Steps:
            1. Represent an absent grace-shape field as a column of missing shapes.
            2. Require every present shape column to use longitude and latitude in EPSG:4326.
            3. Require dates without a timezone and express them as nanosecond timestamps.
            4. Keep the supplied shapes and measurements unchanged.

        Args:
            frame: Daily GeoParquet footprint rows decoded by GeoPandas.

        Returns:
            The same rows and shapes with nanosecond calendar dates and an optional
            missing grace column represented by a null GeoSeries.

        Raises:
            FootprintError: If the geographic coordinate system or date type is wrong.

        Examples:
            frame = WaveFootprints.prepare_frame(gpd.read_parquet("2026-08-31.parquet"))
        """
        if "grace_geometry" not in frame.columns:
            frame["grace_geometry"] = gpd.GeoSeries(None, index=frame.index, crs=4326)
        for name in ("geometry", "grace_geometry", "buffer_geometry"):
            if name not in frame.columns:
                continue  # The named schema reports missing columns.
            column = frame[name]
            if not isinstance(column, gpd.GeoSeries):
                msg = f"{name} must be a GeoParquet geometry column"
                raise FootprintError(msg)
            if column.crs is None or not column.crs.equals(
                CRS.from_epsg(4326), ignore_axis_order=True
            ):
                msg = f"{name} must use the EPSG:4326 coordinate system"
                raise FootprintError(msg)
        if "date" in frame.columns:
            dates = frame["date"]
            if isinstance(dates.dtype, pd.DatetimeTZDtype):
                msg = "date must be a timezone-free timestamp column"
                raise FootprintError(msg)
            if not pd.api.types.is_datetime64_dtype(dates.dtype):
                msg = "date must be a date or timezone-free timestamp column"
                raise FootprintError(msg)
            frame["date"] = pd.to_datetime(dates).astype("datetime64[ns]")
        return frame

    @classmethod
    def from_cells(cls, cells: WaveCells, day: date, series: WaveSeries) -> Self:
        """Build daily footprints from accepted mature wave cells.

        Steps:
            1. Require an accepted WaveCells table, a calendar date and a supported series.
            2. Check that every cell row represents the requested date.
            3. Combine the cells into daily shapes and calculate the three role areas.
            4. Check the resulting footprint table once before returning its model.

        Args:
            cells: Validated qualifying, grace and gap-fill cell table.
            day: Calendar day represented by every input row.
            series: Maximum/minimum and warm/cold selection.

        Returns:
            Validated daily footprint shapes, role areas and identity hashes.

        Raises:
            FootprintError: If the date or series is invalid or rows disagree.

        Examples:
            footprints = WaveFootprints.from_cells(cells, date(2024, 7, 1), series)
        """
        if not isinstance(day, date) or isinstance(day, datetime):
            raise FootprintError("day must be a plain calendar date")
        from .cells import WaveCells

        if not isinstance(cells, WaveCells):
            raise FootprintError("cells must be a validated WaveCells table")
        if not isinstance(series, WaveSeries):
            raise FootprintError("series must use tasmax/tasmin and warm/cold")
        try:
            series.validate()
        except ValueError as error:
            raise FootprintError(str(error)) from error
        if not cells.frame["date"].eq(pd.Timestamp(day)).all():
            raise FootprintError(f"wave-cell dates must all equal midnight {day}")
        return cls(WaveFootprintBuilder(cells.frame, day, series).build())

    def to_parquet(
        self, path: str | PathLike[str], *, filesystem: AbstractFileSystem | None = None
    ) -> None:
        """Write accepted footprint rows without another validation pass.

        Steps:
            1. Choose the filesystem and create the output directory if needed.
            2. Encode the accepted rows as GeoParquet in memory without their table index.
            3. For a local file, write a temporary sibling file and replace the target only after the complete write succeeds.
            4. For a remote file, upload the complete encoded bytes through the supplied filesystem.

        Args:
            path: Local path or filesystem URL.
            filesystem: Optional caller-configured filesystem.

        Returns:
            None. Writes a directly readable GeoParquet file.

        Examples:
            footprints.to_parquet("2024-07-01.parquet")
        """
        if filesystem is None:
            filesystem, location = fsspec.core.url_to_fs(str(path))
        else:
            location = str(path)
        parent = location.rpartition("/")[0]
        if parent:
            filesystem.makedirs(parent, exist_ok=True)
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

    def validate_day(self, day: date, series: WaveSeries) -> None:
        """Check a stored footprint date without reconstructing naming order.

        Args:
            day: Expected plain calendar date.
            series: Maximum/minimum and warm/cold scientific series.

        Returns:
            None. The accepted footprint schema, date and physical identities
            remain usable for a preparation-stage check without opening cells.

        Raises:
            FootprintError: If dates or physical identities disagree.

        Examples:
            footprints.validate_day(day, series)
        """
        self._prepare_components(day, series)

    def _prepare_components(
        self, day: date, series: WaveSeries
    ) -> tuple[tuple[str, _FootprintRow, DailyWaveComponentGeometry], ...]:
        """Retain shapes and measurements while deriving daily physical IDs.

        Args:
            day: Expected plain calendar date.
            series: Maximum/minimum and warm/cold scientific series.

        Returns:
            Input-ordered physical IDs, original rows and role records. This
            preparation does not compute naming order or open cell files.

        Raises:
            FootprintError: If a row has a wrong date or physical IDs repeat.

        Examples:
            prepared = footprints._prepare_components(day, series)
        """
        if not isinstance(day, date) or isinstance(day, datetime):
            raise FootprintError("day must be a plain calendar date")
        if not isinstance(series, WaveSeries):
            raise FootprintError("series must use tasmax/tasmin and warm/cold")
        try:
            series.validate()
        except ValueError as error:
            raise FootprintError(str(error)) from error
        frame = self.frame
        if not frame["date"].eq(pd.Timestamp(day)).all():
            msg = f"footprint dates must all equal midnight {day.isoformat()}"
            raise FootprintError(msg)
        prepared: list[tuple[str, _FootprintRow, DailyWaveComponentGeometry]] = []
        for label, values in enumerate(frame.itertuples(index=False), 1):
            row = cast("_FootprintRow", values)
            identity = component_identity(
                temperature=series.temperature,
                direction=series.direction,
                date=row.date.to_datetime64(),
                geometry=row.buffer_geometry,
            )
            roles = DailyWaveComponentGeometry(
                time=row.date.to_datetime64(),
                label=label,
                geometry=row.geometry,
                grace_geometry=row.grace_geometry,
                full_geometry=row.buffer_geometry,
                area_qualified=float(row.area_qualified),
                area_grace=float(row.area_grace),
                area_gap_filled=float(row.area_gap_filled),
                qualified_fraction=float(row.qualified_fraction),
            )
            prepared.append((identity, row, roles))
        if len({identity for identity, _, _ in prepared}) != len(prepared):
            raise FootprintError("footprints must have unique physical coverage")
        return tuple(prepared)

    def to_components(
        self, day: date, series: WaveSeries, *, native_cells: WaveCells
    ) -> tuple[ComponentMembership, ...]:
        """Create tracker components from footprints and complete native cells.

        Steps:
            1. Require every footprint row to represent the requested date.
            2. Derive date-and-first-cell identities from the complete footprints.
            3. Rebuild all historical ordering digests together from the day's
               complete native cell roles, retaining the original labels separately.
            4. Sort the ordering digests and assign daily component numbers.
               Keep the permanent identities, shapes and measurements unchanged.

        Args:
            day: Expected plain calendar date.
            series: Scientific series represented by the file.
            native_cells: Complete accepted native cells for the whole date,
                including gap-filled cells outside retained footprints.

        Returns:
            Components sorted by historical footprint digest. Geometry bytes and scalar
            values retain the supplied precision. Geometry area measurements and
            containment checks remain the tracker's responsibility.

        Raises:
            FootprintError: If a date is wrong, identities or ordering digests
                repeat, or supplied shapes cannot represent complete native cells.

        Examples:
            components = footprints.to_components(day, series, native_cells=cells)
        """
        prepared = self._prepare_components(day, series)
        try:
            # The original day-wide grouping affects how equal-size pieces are
            # ordered at the date line. Rebuild the complete input date once;
            # a storm's isolated shape does not contain that surrounding context.
            order_digests = HistoricalFootprintOrder.for_day(
                pd.Timestamp(day).to_datetime64(),
                tuple(roles for _, _, roles in prepared),
                native_cells=native_cells,
            )
        except FootprintOrderError as error:
            raise FootprintError(
                f"cannot compute historical ordering for {day.isoformat()}: {error}"
            ) from error
        components = [
            ComponentMembership(
                component_id=identity,
                order_digest=order_digest,
                input_id=row.storm_id,
                geometry=roles,
            )
            for (identity, row, roles), order_digest in zip(
                prepared, order_digests, strict=True
            )
        ]
        if len({component.order_digest for component in components}) != len(components):
            raise FootprintError(
                "footprints must have unique historical ordering digests"
            )
        return tuple(
            replace(component, geometry=replace(component.geometry, label=label))
            for label, component in enumerate(
                sorted(components, key=lambda component: component.order_digest), 1
            )
        )


class WaveFootprintSource(DailyFileSource):
    """Read prepared footprints with their explicit complete native-cell source.

    Steps:
        1. Configure the caller-selected file location and access options.
        2. Resolve and validate daily files through the source's read methods.

    Args:
        location: Local directory or fsspec URL for the daily files.
        storage_options: Caller-selected filesystem access settings.
        known_empty_dates: Explicit dates permitted to have missing empty files.
        cell_source: Complete native-cell source required for tracking reads.
            None permits listing, writing and direct footprint schema checks.

    Examples:
        source = WaveFootprintSource("./footprints", cell_source=WaveCellSource("./cells"))
        components = source.read(day, series)
    """

    __slots__ = ("cell_source",)
    error_type = FootprintError

    def __init__(
        self,
        location: str,
        storage_options: dict[str, object] | None = None,
        *,
        known_empty_dates: tuple[date, ...] = (),
        cell_source: WaveCellSource | None = None,
    ) -> None:
        """Keep prepared shapes and complete native cells at explicit locations.

        Args:
            location: Prepared footprint directory or filesystem URL.
            storage_options: Caller-selected access options for footprint files.
            known_empty_dates: Explicit dates allowed to lack empty footprint files.
            cell_source: Native-cell reader used when producing tracker components.
                None allows listing and writing without starting tracking.

        Returns:
            None. Stores the configured readers without opening daily files.

        Raises:
            FootprintError: If dates, locations or the companion reader are invalid.

        Examples:
            source = WaveFootprintSource("./footprints", cell_source=WaveCellSource("./cells"))
        """
        from .cells import WaveCellSource

        if cell_source is not None and not isinstance(cell_source, WaveCellSource):
            raise FootprintError("cell_source must be a WaveCellSource")
        super().__init__(location, storage_options, known_empty_dates=known_empty_dates)
        self.cell_source = cell_source

    def read(self, day: date, series: WaveSeries) -> tuple[ComponentMembership, ...]:
        """Read one date and series while retaining the exact stored shapes.

        Steps:
            1. Choose the file for the requested day and temperature/direction series.
            2. Read and check its footprint table through WaveFootprints.
            3. Allow a missing file only when the caller explicitly declared that day empty.
            4. Reject rows on a declared empty date and read the complete native
               cells from the configured companion source.
            5. Compute all ordering digests together, preserving stored shapes.

        Args:
            day: Date of the input file.
            series: Maximum/minimum and warm/cold wave series.

        Returns:
            Daily components with stored role shapes, areas and ordering digests.
            A declared missing empty date returns an empty tuple only after its
            native-cell context agrees.

        Raises:
            FileNotFoundError: If an undeclared daily input is missing.
            FootprintError: If the companion source is absent, footprint fields
                are invalid, or complete native cells disagree with stored roles.
            WaveCellError: If the companion cell data is invalid.

        Examples:
            components = source.read(date(2026, 8, 31), series)
        """
        path = self.path_for(day, series)
        if self.cell_source is None:
            raise FootprintError(
                "prepared footprint tracking requires a complete native cell_source"
            )
        try:
            footprints = WaveFootprints.read_parquet(path, filesystem=self._filesystem)
        except FileNotFoundError:
            if day in self.known_empty_dates:
                native_cells = self.cell_source.read_cells(day, series)
                try:
                    HistoricalFootprintOrder.for_day(
                        pd.Timestamp(day).to_datetime64(), (), native_cells=native_cells
                    )
                except FootprintOrderError as error:
                    raise FootprintError(
                        f"declared empty footprint date {day} has inconsistent native cells: {error}"
                    ) from error
                return ()
            raise
        if day in self.known_empty_dates and not footprints.frame.empty:
            msg = f"{path}: declared empty date contains footprint rows"
            raise FootprintError(msg)
        return footprints.to_components(
            day, series, native_cells=self.cell_source.read_cells(day, series)
        )
