"""Write scientific wave datasets and read selected Parquet partitions.

Daily records stream to files while a rollup builds lifetime catalog footprints.
Opening completed output reads its metadata only.
"""

from __future__ import annotations

import json
import posixpath
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from io import BytesIO
from typing import TYPE_CHECKING, ClassVar, Final, Literal, Self, TypedDict, cast

import fsspec
import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandera.errors import SchemaError, SchemaErrors

from ._catalog import CatalogRollup
from ._wave_tracking import EndedWave, LineageEdge
from .schemas import (
    FileSchema,
    LineageFileSchema,
    LineageSchema,
    NamedStormsSchema,
    WaveCatalogFileSchema,
    WaveCatalogSchema,
    WaveDayFileSchema,
    WaveDaySchema,
)
from .series import WaveSeries
from .tracker import WaveTrackingBatch

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from os import PathLike
    from types import TracebackType

    from fsspec.spec import AbstractFileSystem

RecordTable = Literal["wave_days", "waves", "lineage"]


class SeriesEntry(TypedDict):
    """The selected temperature and direction."""

    temperature: str
    direction: str


class FileEntry(TypedDict):
    """One scientific Parquet file and its partition and size."""

    path: str
    table: str
    partition: dict[str, str | int]
    rows: int
    row_groups: int
    bytes: int


class DatasetSummary(TypedDict):
    """Row, file and byte counts for one scientific dataset."""

    rows: int
    files: int
    bytes: int


class CatalogEntry(TypedDict):
    """One self-contained catalog GeoParquet download."""

    path: str
    rows: int
    bytes: int


class Manifest(TypedDict):
    """Selected series, requested dates and completed output inventory."""

    generation_id: str
    series: SeriesEntry
    package: dict[str, str]
    start_date: str
    end_date: str
    files: list[FileEntry]
    tables: dict[str, DatasetSummary]
    all_storms: CatalogEntry
    all_waves: CatalogEntry


@dataclass(frozen=True, slots=True)
class CompletionContext:
    """Requested date range and series for the completed output.

    Args:
        start_date: First requested date, inclusive.
        end_date: Last requested date, inclusive.
        series: Temperature and direction tracked in this run.
        package_version: Producing distribution version.

    Examples:
        context = CompletionContext(date(2020, 1, 1), date(2020, 1, 5),
                                    WaveSeries("tasmax", "warm"), "0.1.0")
    """

    start_date: date
    end_date: date
    series: WaveSeries
    package_version: str


RecordRow = dict[str, object]
_TABLES: Final[tuple[RecordTable, ...]] = ("wave_days", "waves", "lineage")
_MAX_FILE_BYTES: Final = 512 * 1024 * 1024
_HEX = frozenset("0123456789abcdef")


class WaveRecordStorageError(ValueError):
    """Raised when a wave-record generation cannot be safely read or written."""


@dataclass(frozen=True, slots=True)
class DatasetInventory:
    """Counts and size information for one completed scientific dataset.

    Args:
        rows: Number of records across the dataset.
        objects: Number of Parquet objects across the dataset.
        bytes: Total encoded Parquet bytes.
    """

    rows: int
    objects: int
    bytes: int


@dataclass(frozen=True, slots=True)
class GenerationInventory:
    """Inventory returned after a generation is written.

    Args:
        generation_id: Identifier shared by the manifest and completion marker.
        datasets: Inventory for each scientific table.
        rows: Total scientific rows.
        objects: Total scientific objects.
        bytes: Total scientific bytes.
    """

    generation_id: str
    datasets: dict[RecordTable, DatasetInventory]
    rows: int
    objects: int
    bytes: int


@dataclass(frozen=True, slots=True)
class WaveTable:
    """A schema-backed wave table with shared Parquet construction and I/O.

    Args:
        frame: Table with the concrete class's schema columns.

    Examples:
        table = WaveDays.from_rows(daily_rows)
        table.to_parquet("wave_days.parquet")
    """

    frame: pd.DataFrame
    schema: ClassVar[type[FileSchema]]
    file_schema: ClassVar[type[FileSchema]]
    date_columns: ClassVar[tuple[str, ...]]
    geometry_columns: ClassVar[tuple[str, ...]] = ()
    geometry_crs: ClassVar[str] = "EPSG:4326"
    partition_field: ClassVar[str]

    def __post_init__(self) -> None:
        """Validate once without copying or converting the supplied frame.

        Steps:
            1. Validate the supplied frame with its declared schema.
            2. Translate schema failures into a storage error without copying the frame.

        Args:
            None.

        Returns:
            None. Retains the caller's frame and checks its existing schema.

        Raises:
            WaveRecordStorageError: If the table violates its schema.

        Examples:
            table = WaveDays(frame)
        """
        try:
            self.schema.validate(self.frame)
        except (SchemaError, SchemaErrors) as error:
            raise WaveRecordStorageError(
                f"invalid {self.schema.__name__} table: {error}"
            ) from error

    @classmethod
    def empty_frame(cls) -> pd.DataFrame:
        """Create a typed empty frame without scanning or validating any rows.

        Steps:
            1. Read the column names and types from the table schema.
            2. Create empty columns of those types, including geographic shape columns when required.
            3. Return a rowless table that can still be written and read.

        Args:
            None.

        Returns:
            Empty frame with the declared scalar and geographic columns.

        Examples:
            empty = WaveDays.empty_frame()
        """
        frame = pd.DataFrame(
            {
                name: pd.Series(dtype=str(column.dtype))
                for name, column in cls.schema.to_schema().columns.items()
                if name not in cls.geometry_columns
            }
        )
        if cls.geometry_columns:
            for name in cls.geometry_columns:
                frame[name] = gpd.GeoSeries([], crs=cls.geometry_crs)
            frame = gpd.GeoDataFrame(frame, geometry=cls.geometry_columns[-1])
        return frame

    @classmethod
    def prepare_frame(
        cls, frame: pd.DataFrame, *, writing: bool = False
    ) -> pd.DataFrame:
        """Prepare newly built or decoded columns before the single schema check.

        Steps:
            1. Prepare newly built dates or decoded date precision for the table's calendar fields.
            2. When reading, turn decoded parent arrays into ordinary lists without accepting unrelated types.
            3. When writing geographic rows, attach the table's coordinate system to each shape column.
            4. Return the prepared table for its single schema check.

        Args:
            frame: Newly created or decoded table owned by this I/O operation.
            writing: Whether Python dates and shapes come from tracker records.

        Returns:
            Frame with existing date/list decoding and geographic types applied.
            Wrong incoming types remain available for the schema to reject.

        Examples:
            frame = WaveDays.prepare_frame(decoded)
        """
        for name in cls.date_columns:
            if name in frame:
                if writing:
                    frame[name] = pd.to_datetime(frame[name]).astype("datetime64[ns]")
                elif (
                    pd.api.types.is_datetime64_dtype(frame[name].dtype)
                    and frame[name].dtype != "datetime64[ns]"
                ):
                    frame[name] = frame[name].astype("datetime64[ns]")
        if not writing:
            for name in ("parent_wave_ids", "parent_keys"):
                if name in frame:
                    frame[name] = frame[name].map(
                        lambda value: (
                            list(value)
                            if pd.api.types.is_list_like(value)
                            and not isinstance(value, dict | set)
                            else value
                        )
                    )
        if cls.geometry_columns and writing:
            for name in cls.geometry_columns:
                frame[name] = gpd.GeoSeries(
                    frame[name], index=frame.index, crs=cls.geometry_crs
                )
            return gpd.GeoDataFrame(frame, geometry=cls.geometry_columns[-1])
        return frame

    @classmethod
    def from_rows(cls, rows: list[RecordRow]) -> Self:
        """Build one file buffer or standalone table and validate it once.

        Steps:
            1. Return a correctly typed empty table when there are no records.
            2. Build columns from the supplied records and prepare date and shape types.
            3. Construct the table model, which checks the schema once.

        Args:
            rows: Tracker-derived rows including series and year partitions.

        Returns:
            Validated table, including a typed empty table for quiet output.

        Raises:
            WaveRecordStorageError: If rows violate the existing table schema.

        Examples:
            table = WaveDays.from_rows(daily_rows)
        """
        if not rows:
            return cls(cls.empty_frame())
        frame = pd.DataFrame.from_records(rows)
        return cls(cls.prepare_frame(frame, writing=True))

    @property
    def file_frame(self) -> pd.DataFrame:
        """Select stored columns from an already validated partitioned table.

        Steps:
            1. Select the schema's stored columns.
            2. Retain the existing validated scalar and geometry values.

        Args:
            None.

        Returns:
            Frame without the fields represented by directory partitions.
            Selection performs no additional validation.

        Examples:
            stored = table.file_frame
        """
        return cast(
            "pd.DataFrame",
            self.frame.loc[:, list(self.file_schema.to_schema().columns)],
        )

    @classmethod
    def read_parquet(
        cls,
        path: str | PathLike[str],
        *,
        filesystem: AbstractFileSystem | None = None,
        partition: Mapping[str, str | int] | None = None,
    ) -> Self:
        """Read standalone or partitioned Parquet with one restored-table check.

        Steps:
            1. Open the requested local or remote file using the caller's access settings.
            2. Read geographic columns when this table has shapes; otherwise read ordinary columns.
            3. Prepare decoded dates and lists.
            4. For a file inside partition folders, reject unexpected fields and restore the series and year from its folder.
            5. Construct the table model and report file errors with its location.

        Args:
            path: Local file or caller-selected filesystem URL.
            filesystem: Optional configured filesystem, including S3.
            partition: Directory fields to restore before validation, when reading
                a physical file from the pipeline's partitioned output.

        Returns:
            Validated table with decoded dates, lists and exact stored shapes.

        Raises:
            WaveRecordStorageError: If the file cannot be read or validated.

        Examples:
            table = WaveDays.read_parquet("wave_days.parquet")
        """
        fs, location = _resolve_location(path, filesystem)
        try:
            with fs.open(location, "rb") as source:
                frame = (
                    gpd.read_parquet(source, to_pandas_kwargs={"date_as_object": False})
                    if cls.geometry_columns
                    else pd.read_parquet(
                        source, to_pandas_kwargs={"date_as_object": False}
                    )
                )
            frame = cls.prepare_frame(frame)
            if partition is not None:
                unexpected = set(frame.columns) - set(
                    cls.file_schema.to_schema().columns
                )
                if unexpected:
                    raise WaveRecordStorageError(
                        f"physical file contains unexpected columns: {sorted(unexpected)}"
                    )
                if set(partition) != {"temperature", "direction", cls.partition_field}:
                    raise WaveRecordStorageError(
                        "partition fields must match the table Hive partitioning"
                    )
                for name, value in partition.items():
                    frame[name] = value
            return cls(frame)
        except (OSError, ValueError, TypeError) as error:
            raise WaveRecordStorageError(
                f"cannot read {cls.schema.__name__} table {path}"
            ) from error

    def to_parquet(
        self,
        path: str | PathLike[str],
        *,
        filesystem: AbstractFileSystem | None = None,
        row_group_rows: int = 16_384,
    ) -> DatasetInventory:
        """Write a standalone table using the existing check for caller edits.

        Steps:
            1. Check the requested number of rows per stored group.
            2. Apply the existing check for changes to the table since it was constructed.
            3. Encode the table and create the destination folder when needed.
            4. Write the bytes and return the row, file and byte counts.

        Args:
            path: Local path or caller-selected filesystem URL.
            filesystem: Optional configured filesystem, including S3.
            row_group_rows: Positive maximum rows in each Parquet row group.

        Returns:
            Written row count, one object and encoded byte count.

        Raises:
            WaveRecordStorageError: If the edited table or row-group size is invalid.
            OSError: If the destination cannot be written.

        Examples:
            inventory = table.to_parquet("waves.parquet")
        """
        if (
            not isinstance(row_group_rows, int)
            or isinstance(row_group_rows, bool)
            or row_group_rows < 1
        ):
            raise WaveRecordStorageError("row_group_rows must be a positive integer")
        frame = type(self)(self.frame).frame
        content = _encode_parquet(frame, row_group_rows)
        fs, location = _resolve_location(path, filesystem)
        parent = location.rpartition("/")[0]
        if parent:
            fs.makedirs(parent, exist_ok=True)
        _write_bytes(fs, location, content)
        return DatasetInventory(rows=len(self.frame), objects=1, bytes=len(content))


@dataclass(frozen=True, slots=True)
class Waves(WaveTable):
    """A validated full wave catalog with qualified lifetime geometry.

    Includes unnamed heat/cold waves and named heat/cold storms. Each keeps
    its ``wave_id`` throughout its history; ``name`` distinguishes named storms.

    Args:
        frame: GeoDataFrame with all WaveCatalogSchema columns.

    Examples:
        waves = Waves.read_parquet("waves.parquet")
        waves.frame.sort_values("total_area_visited_km2", ascending=False)
    """

    frame: gpd.GeoDataFrame
    schema: ClassVar[type[FileSchema]] = WaveCatalogSchema
    file_schema: ClassVar[type[FileSchema]] = WaveCatalogFileSchema
    date_columns: ClassVar[tuple[str, ...]] = (
        "start_date",
        "last_observed_date",
        "end_date",
        "peak_date",
    )
    geometry_columns: ClassVar[tuple[str, ...]] = ("lifetime_geometry",)
    partition_field: ClassVar[str] = "start_year"


@dataclass(frozen=True, slots=True)
class WaveDays(WaveTable):
    """Daily wave observations with exact qualified, grace and full geometry.

    ``name`` is absent before naming and retained on later named-storm days.
    Earlier days are not retroactively named, and ``wave_id`` does not change.

    Args:
        frame: GeoDataFrame with all WaveDaySchema columns.

    Examples:
        days = WaveDays.read_parquet("wave_days.parquet")
    """

    frame: gpd.GeoDataFrame
    schema: ClassVar[type[FileSchema]] = WaveDaySchema
    file_schema: ClassVar[type[FileSchema]] = WaveDayFileSchema
    date_columns: ClassVar[tuple[str, ...]] = ("date", "start_date", "peak_date")
    geometry_columns: ClassVar[tuple[str, ...]] = (
        "qualified_geometry",
        "grace_geometry",
        "full_geometry",
    )
    geometry_crs: ClassVar[str] = "OGC:CRS84"
    partition_field: ClassVar[str] = "observation_year"

    @staticmethod
    def record_row(record: object, batch_date: date) -> RecordRow:
        """Retain one tracker's scalar facts and supplied polygon objects.

        Steps:
            1. Require a tracked wave record whose observation date matches the daily batch.
            2. Check the series and require sorted, distinct parent IDs.
            3. Copy the permanent wave ID, daily footprint ID, dates, peak and parent facts.
            4. Retain the supplied shapes and role areas for the table writer to check and save.

        Args:
            record: Current tracked wave component.
            batch_date: Date emitted by the chronological tracker.

        Returns:
            Daily scientific row including selected series and year partition.
            The file writer subsequently validates the complete DataFrame.

        Raises:
            WaveRecordStorageError: If the component date or parent ordering disagrees.
            TypeError: If the current record is not a tracked wave component.

        Examples:
            row = WaveDays.record_row(record, date(2024, 1, 1))
        """
        from ._wave_tracking import TrackedWaveComponent

        if not isinstance(record, TrackedWaveComponent):
            msg = "tracking batch current records must be TrackedWaveComponent values"
            raise TypeError(msg)
        component = record.component.geometry
        record_date = _as_date(component.time, "component date")
        if record_date != batch_date:
            msg = "component date must match tracking batch date"
            raise WaveRecordStorageError(msg)
        _validate_branch(record.series.temperature, record.series.direction)
        parent_ids = tuple(record.parent_wave_ids)
        if parent_ids != tuple(sorted(set(parent_ids))):
            msg = "wave-day parent_wave_ids must be sorted and unique"
            raise WaveRecordStorageError(msg)
        return {
            "temperature": record.series.temperature,
            "direction": record.series.direction,
            "observation_year": record_date.year,
            "wave_id": record.wave_id,
            "component_id": record.component_id,
            "name": record.name,
            "date": record_date,
            "start_date": _as_date(record.start_date, "start_date"),
            "component_label": component.label,
            "continues": record.continues,
            "overlap_percent": float(record.overlap_percent),
            "parent_wave_ids": list(parent_ids),
            "parent_keys": list(record.parent_keys),
            "peak_qualified_area_km2": float(record.peak_qualified_area),
            "peak_date": _as_date(record.peak_date, "peak_date"),
            "qualified_area_km2": float(component.area_qualified),
            "grace_area_km2": float(component.area_grace),
            "gap_filled_area_km2": float(component.area_gap_filled),
            "qualified_fraction": float(component.qualified_fraction),
            "qualified_geometry": component.geometry,
            "grace_geometry": component.grace_geometry,
            "full_geometry": component.full_geometry,
        }


@dataclass(frozen=True, slots=True)
class Lineage(WaveTable):
    """Split and merge relationships between permanent wave IDs.

    Args:
        frame: DataFrame with all LineageSchema columns.

    Examples:
        lineage = Lineage.read_parquet("lineage.parquet")
    """

    schema: ClassVar[type[FileSchema]] = LineageSchema
    file_schema: ClassVar[type[FileSchema]] = LineageFileSchema
    date_columns: ClassVar[tuple[str, ...]] = ("event_date",)
    partition_field: ClassVar[str] = "event_year"

    @staticmethod
    def edge_row(edge: LineageEdge, batch_date: date) -> RecordRow:
        """Convert a tracker relationship to its scientific output row.

        Steps:
            1. Require a split or merge relationship emitted on this batch's date.
            2. Reject a relationship from a wave to itself or an unknown relationship type.
            3. Copy both permanent wave IDs, the series and the event date into one output row.

        Args:
            edge: Split or merge emitted by the tracker.
            batch_date: Current daily batch date.

        Returns:
            Lineage row with its year partition fields.

        Raises:
            WaveRecordStorageError: If dates, keys or relationship are invalid.

        Examples:
            row = Lineage.edge_row(edge, date(2020, 1, 3))
        """
        if not isinstance(edge, LineageEdge):
            raise TypeError("tracking batch lineage must contain LineageEdge values")
        event_date = _as_date(edge.event_date, "lineage event date")
        if event_date != batch_date:
            raise WaveRecordStorageError(
                "lineage event date must match its tracking batch date"
            )
        if edge.source_wave_id == edge.target_wave_id or edge.relationship not in {
            "split",
            "merge",
        }:
            raise WaveRecordStorageError(
                "lineage requires distinct waves and a split or merge"
            )
        return {
            "temperature": edge.series.temperature,
            "direction": edge.series.direction,
            "event_year": event_date.year,
            "source_wave_id": edge.source_wave_id,
            "target_wave_id": edge.target_wave_id,
            "event_date": event_date,
            "relationship": edge.relationship,
        }


@dataclass(frozen=True, slots=True)
class NamedStorms(Waves):
    """A named-storm catalog using the full catalog's construction and I/O.

    Args:
        frame: Self-contained GeoDataFrame with all NamedStormsSchema columns.

    Examples:
        storms = NamedStorms.read_parquet("data/all_storms.parquet")
        storms.to_parquet("all_storms.parquet")
    """

    schema: ClassVar[type[FileSchema]] = NamedStormsSchema


_TABLE_CLASSES: Final[dict[RecordTable, type[WaveTable]]] = {
    "wave_days": WaveDays,
    "waves": Waves,
    "lineage": Lineage,
}


@dataclass(frozen=True, slots=True)
class CompletionEvidence:
    """File inventory returned by :func:`write_generation`.

    Args:
        inventory: Manifest-backed scientific object inventory.
    """

    inventory: GenerationInventory


@dataclass(frozen=True, slots=True)
class CompletedGeneration:
    """Completed output metadata and filesystem access for selected scans.

    Args:
        location: Caller-owned generation location.
        filesystem: Filesystem used to access the generation.
        manifest: Completed-output manifest.
        inventory: Scientific file counts and sizes.
    """

    location: str
    filesystem: AbstractFileSystem
    manifest: Manifest
    inventory: GenerationInventory


def write_generation(
    batches: Iterable[WaveTrackingBatch],
    *,
    destination: str | PathLike[str],
    generation_id: str,
    completion: CompletionContext,
    filesystem: AbstractFileSystem | None = None,
    row_group_rows: int = 16_384,
    file_rows: int = 65_536,
    progress: Callable[[str], None] | None = None,
) -> CompletionEvidence:
    """Write one new, compact, immutable wave-record generation.

    Steps:
        1. Validate settings and reserve an empty destination.
        2. Stream daily records and assemble lifetime catalogs.
        3. Write the manifest and completion marker after all products succeed.

    Args:
        batches: Ordered, complete tracker output batches.
        destination: New, empty caller-selected generation directory or URL.
        generation_id: Non-empty identifier retained in manifest and markers.
        completion: Requested dates, series and package version.
        filesystem: Optional generic fsspec filesystem for the destination.
        row_group_rows: Positive maximum records in a Parquet row group.
        file_rows: Positive maximum records in one Parquet object.
        progress: Optional callback receiving catalog stage and count messages.

    Returns:
        Completion inventory for the three scientific datasets.

    Raises:
        WaveRecordStorageError: If inputs are invalid, output already exists,
            tracker records disagree, or an object cannot be encoded.

    Examples:
        evidence = write_generation(batches, destination="/tmp/new-waves",
            generation_id="example", completion=completion)
    """
    _validate_writer_settings(generation_id, row_group_rows, file_rows)
    _validate_completion_context(completion)
    fs, destination_path = _resolve_location(destination, filesystem)
    _require_nonempty_location(destination_path, "destination")
    _require_new_destination(fs, destination_path)
    _write_json_exclusive(
        fs,
        _join(destination_path, "_INCOMPLETE.json"),
        {"generation_id": generation_id, "status": "writing"},
    )
    report = progress if progress is not None else lambda message: None
    report("Catalog: reading saved daily batches")
    with CatalogDownloads(fs, destination_path, row_group_rows) as downloads:
        entries = _assemble_generation(
            batches,
            fs,
            destination_path,
            completion,
            row_group_rows,
            file_rows,
            downloads,
            report,
        )
        report("Catalog: finalizing all_storms.parquet and all_waves.parquet")
        all_storms, all_waves = downloads.finish()
    report(
        f"Catalog: exports saved; {all_storms['rows']} named storms, {all_waves['rows']} waves"
    )
    manifest = _manifest(generation_id, entries, completion, all_storms, all_waves)
    manifest_bytes = _canonical_json(manifest)
    _write_bytes(fs, _join(destination_path, "manifest.json"), manifest_bytes)
    _write_json(
        fs,
        _join(destination_path, "_COMPLETE.json"),
        {
            "generation_id": generation_id,
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )
    fs.rm(_join(destination_path, "_INCOMPLETE.json"))
    inventory = _inventory_from_manifest(manifest)
    report("Catalog: completion marker saved")
    return CompletionEvidence(inventory=inventory)


class CatalogDownloads:
    """Stream the already checked catalog into full and named-only downloads.

    Uses the partition writer's encoded WKB and scalar columns. Each catalog
    part is decoded in memory, given its partition columns, and written to both
    downloads without new geometry calculations or schema validation passes.

    Args:
        filesystem: Destination filesystem.
        root: Completed-generation directory.
        row_group_rows: Maximum records per download row group.

    Examples:
        with CatalogDownloads(fs, "output", 16384) as downloads:
            downloads.append(encoded_catalog_part, partition)
            storms, waves = downloads.finish()
    """

    def __init__(
        self, filesystem: AbstractFileSystem, root: str, row_group_rows: int
    ) -> None:
        """Retain output settings and prepare lazily opened file writers.

        Steps:
            1. Retain the filesystem, destination and rows per stored group.
            2. Prepare a shared file-closing context and zero row counts for both
               catalogs, opening no files yet.

        Args:
            filesystem: Destination filesystem.
            root: Completed-generation directory.
            row_group_rows: Maximum records per download row group.

        Returns:
            None. Files open when the first catalog part is supplied.

        Examples:
            downloads = CatalogDownloads(fs, "output", 16384)
        """
        self.filesystem = filesystem
        self.root = root
        self.row_group_rows = row_group_rows
        self._resources = ExitStack()
        self._writers: dict[str, pq.ParquetWriter] = {}
        self._rows = {"all_storms": 0, "all_waves": 0}

    def __enter__(self) -> Self:
        """Start the download writer context.

        Steps:
            1. Return the initialized writer for use in the context block.

        Args:
            None.

        Returns:
            This writer, whose open files close when the context ends.

        Examples:
            with CatalogDownloads(fs, "output", 16384) as downloads:
                downloads.finish()
        """
        self._resources.__enter__()
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        """Close Parquet writers and file handles, including on failure.

        Steps:
            1. Close both catalog writers and their file handles.
            2. Allow any context exception to propagate.

        Args:
            kind: Raised exception type, if any.
            error: Raised exception, if any.
            traceback: Exception traceback, if any.

        Returns:
            The resource stack's exception handling result.

        Examples:
            The context manager calls this after writing or a failed upload.
        """
        return self._resources.__exit__(kind, error, traceback)

    def append(self, encoded: bytes, partition: dict[str, str | int]) -> None:
        """Append one validated catalog part to both self-contained files.

        Steps:
            1. Read an already checked catalog part and restore its series and birth year as ordinary columns.
            2. Use consistent list and text types so later parts can join the same files.
            3. Remove shape limits that describe only this part rather than the whole catalog.
            4. Open the two catalog writers on the first part.
            5. Append every row to all_waves and only rows with names to all_storms, retaining exact stored shapes.

        Args:
            encoded: Existing partition Parquet bytes from the scientific writer.
            partition: Temperature, direction and start year for those records.

        Returns:
            None. Reuses exact geometry bytes and filters named rows by name.

        Examples:
            downloads.append(encoded, {"temperature": "tasmax", "direction": "warm", "start_year": 2024})
        """
        table = pq.read_table(BytesIO(encoded))
        for name, value in partition.items():
            table = table.append_column(name, pa.array([value] * len(table)))
        parents = table.schema.get_field_index("parent_keys")
        table = table.set_column(
            parents, "parent_keys", table.column(parents).cast(pa.list_(pa.string()))
        )
        for name in ("name", "end_reason", "merged_into_wave_id"):
            index = table.schema.get_field_index(name)
            table = table.set_column(
                index, name, table.column(index).cast(pa.large_string())
            )
        metadata = dict(table.schema.metadata or {})
        geo = json.loads(metadata[b"geo"])
        for column in geo["columns"].values():
            column.pop("bbox", None)
            # GeoParquet permits unknown types; a part cannot describe the
            # geometry types or extent of the whole streamed catalog.
            column["geometry_types"] = []
        metadata[b"geo"] = _canonical_json(geo)
        # Partition columns are ordinary columns in these self-contained files.
        metadata.pop(b"pandas", None)
        table = table.replace_schema_metadata(metadata)
        if not self._writers:
            for name in self._rows:
                sink = self._resources.enter_context(
                    self.filesystem.open(_join(self.root, f"{name}.parquet"), "wb")
                )
                self._writers[name] = self._resources.enter_context(
                    pq.ParquetWriter(sink, table.schema, compression="zstd")
                )
        selections = {
            "all_waves": table,
            "all_storms": table.filter(table["name"].is_valid()),
        }
        for name, selection in selections.items():
            if len(selection):
                self._writers[name].write_table(
                    selection, row_group_size=self.row_group_rows
                )
            self._rows[name] += len(selection)

    def finish(self) -> tuple[CatalogEntry, CatalogEntry]:
        """Finish both catalogs, creating typed empty files when necessary.

        Steps:
            1. Create correctly typed empty catalogs if no rows have been received.
            2. Close both files so their final Parquet information is saved.
            3. Return each file's name, row count and byte count.

        Args:
            None.

        Returns:
            Named-only and full catalog inventory entries, in that order.

        Examples:
            storms, waves = downloads.finish()
        """
        if not self._writers:
            self.append(_encode_parquet(Waves.empty_frame(), self.row_group_rows), {})
        self._resources.close()
        entries: dict[str, CatalogEntry] = {
            name: {
                "path": f"{name}.parquet",
                "rows": count,
                "bytes": self.filesystem.size(_join(self.root, f"{name}.parquet")),
            }
            for name, count in self._rows.items()
        }
        return entries["all_storms"], entries["all_waves"]


class _PartitionWriter:
    """Limit waiting rows and write files into table, series and year folders."""

    def __init__(
        self,
        fs: AbstractFileSystem,
        root: str,
        row_group_rows: int,
        file_rows: int,
        downloads: CatalogDownloads,
    ) -> None:
        """Prepare empty file buffers with the requested row limits.

        Steps:
            1. Retain the destination, row limits and standalone catalog writer.
            2. Start empty row buffers, file counters and output-file records.

        Args:
            fs: Destination fsspec filesystem.
            root: Destination generation root.
            row_group_rows: Maximum Parquet row-group size.
            file_rows: Maximum buffered rows before a file is flushed.
            downloads: Reuse encoded catalog parts for the self-contained files.

        Returns:
            None. No files are opened until rows are ready to write.

        Examples:
            writer = _PartitionWriter(fs, root, 16384, 65536, downloads)
        """
        self.fs = fs
        self.root = root
        self.row_group_rows = row_group_rows
        self.file_rows = file_rows
        self.buffers: dict[tuple[RecordTable, str, str, int], list[RecordRow]] = {}
        self.indices: dict[tuple[RecordTable, str, str, int], int] = {}
        self.entries: list[FileEntry] = []
        self.downloads = downloads

    def append(self, table: RecordTable, row: RecordRow) -> None:
        """Append one sorted row, flushing exactly at the configured row cap.

        Steps:
            1. Find the row's table, temperature, direction and year.
            2. Add it to that folder's waiting rows.
            3. Write and clear the buffer when it reaches the file-row limit.

        Args:
            table: Scientific record table containing the row.
            row: Schema-ready row with its Hive partition fields.

        Returns:
            None. Rows remain buffered or have been written to a file.

        Examples:
            writer.append("wave_days", row)
        """
        key = (
            table,
            str(row["temperature"]),
            str(row["direction"]),
            cast("int", row[_TABLE_CLASSES[table].partition_field]),
        )
        buffer = self.buffers.setdefault(key, [])
        buffer.append(row)
        if len(buffer) == self.file_rows:
            self._flush(key)
        if self.active_rows > self.max_active_rows:
            msg = "partition writer exceeded its aggregate active-row bound"
            raise WaveRecordStorageError(msg)

    @property
    def active_rows(self) -> int:
        """Return the aggregate rows retained across active partition buffers.

        Steps:
            1. Count rows in every active partition buffer.
            2. Return their total.

        Args:
            None.

        Returns:
            None after validation or writing succeeds.

        Examples:
            rows = writer.active_rows
        """
        return sum(len(buffer) for buffer in self.buffers.values())

    @property
    def max_active_rows(self) -> int:
        """Return the strict aggregate cap for active partition buffers.

        Steps:
            1. Multiply the table count by the maximum unflushed rows per buffer.

        Args:
            None.

        Returns:
            None after validation or writing succeeds.

        Examples:
            limit = writer.max_active_rows
        """
        return len(_TABLES) * (self.file_rows - 1)

    def flush_before_year(self, year: int, tables: tuple[RecordTable, ...]) -> None:
        """Flush completed chronological years for the specified source tables.

        Steps:
            1. Select nonempty buffers for the requested tables before the year.
            2. Flush selected buffers in sorted key order.

        Args:
            year: First year whose buffers should remain active.
            tables: Tables whose earlier buffers should be written.

        Returns:
            None after validation or writing succeeds.

        Examples:
            writer.flush_before_year(2025, ("wave_days", "lineage"))
        """
        for key in sorted(self.buffers):
            if key[0] in tables and key[3] < year and self.buffers[key]:
                self._flush(key)

    def flush_tables(self, tables: tuple[RecordTable, ...]) -> None:
        """Flush all remaining buffers belonging to the specified tables.

        Steps:
            1. Select nonempty buffers for the requested tables.
            2. Flush selected buffers in sorted key order.

        Args:
            tables: Tables whose remaining buffers should be written.

        Returns:
            None after validation or writing succeeds.

        Examples:
            writer.flush_tables(("wave_days",))
        """
        for key in sorted(self.buffers):
            if key[0] in tables and self.buffers[key]:
                self._flush(key)

    def finish(self) -> list[FileEntry]:
        """Flush remaining buffers and return path-sorted manifest entries.

        Steps:
            1. Flush remaining partition buffers in sorted key order.
            2. Return file entries sorted by path.

        Args:
            None.

        Returns:
            Scientific manifest entries in canonical path order.

        Examples:
            entries = writer.finish()
        """
        for key in sorted(self.buffers):
            if self.buffers[key]:
                self._flush(key)
        return sorted(self.entries, key=lambda entry: str(entry["path"]))

    def _flush(self, key: tuple[RecordTable, str, str, int]) -> None:
        """Encode and write one folder's waiting rows.

        Steps:
            1. Build and check a table from the rows waiting for this series and year.
            2. Encode it into parts that fit the configured file size.
            3. Write each part under its table, series and year folders.
            4. For catalog rows, also append the same bytes to the two standalone catalogs.
            5. Save file counts and sizes, then clear the written rows from memory.

        Args:
            key: Table and Hive partition addressed by the in-memory buffer.

        Returns:
            None. Written rows are removed and the file inventory is updated.

        Examples:
            writer._flush(("wave_days", "tasmax", "warm", 2024))
        """
        table, temperature, direction, year = key
        rows = self.buffers[key]
        frame = _TABLE_CLASSES[table].from_rows(rows).file_frame
        for part, encoded in _compact_parquet_parts(frame, self.row_group_rows):
            index = self.indices.get(key, 0)
            self.indices[key] = index + 1
            partition_path = _join(
                self.root,
                table,
                f"temperature={temperature}",
                f"direction={direction}",
                f"{_TABLE_CLASSES[table].partition_field}={year}",
            )
            path = _join(partition_path, f"part-{index:05d}.parquet")
            _write_bytes(self.fs, path, encoded)
            partition: dict[str, str | int] = {
                "temperature": temperature,
                "direction": direction,
                _TABLE_CLASSES[table].partition_field: year,
            }
            if table == "waves":
                self.downloads.append(encoded, partition)
            relative = path.removeprefix(self.root.rstrip("/") + "/")
            self.entries.append(
                {
                    "path": relative,
                    "table": table,
                    "partition": partition,
                    "rows": len(part),
                    "row_groups": (len(part) + self.row_group_rows - 1)
                    // self.row_group_rows,
                    "bytes": len(encoded),
                }
            )
        self.buffers[key] = []


def _assemble_generation(
    batches: Iterable[WaveTrackingBatch],
    fs: AbstractFileSystem,
    root: str,
    completion: CompletionContext,
    row_group_rows: int,
    file_rows: int,
    downloads: CatalogDownloads,
    progress: Callable[[str], None],
) -> list[FileEntry]:
    """Write chronological daily records and roll up each wave's lifetime catalog.

    Steps:
        1. Read batches in chronological order and append daily and lineage rows.
        2. Accumulate lifetime geometry and final records for each wave.
        3. Flush partition buffers and return written file entries.

    Args:
        batches: Consecutive complete tracker batches, including quiet dates.
        fs: Destination filesystem.
        root: Result directory.
        completion: Requested output range and series.
        row_group_rows: Maximum rows per Parquet row group.
        file_rows: Maximum buffered rows per file.
        downloads: Stream both catalogs from already checked partition bytes.
        progress: Callback receiving stage and completed-count messages.

    Returns:
        Path-sorted scientific inventory, with both catalog downloads appended.

    Raises:
        WaveRecordStorageError: If dates, identities or selected series disagree.

    Examples:
        entries = _assemble_generation(batches, fs, "waves", context, 1024, 4096, downloads)
    """
    writer = _PartitionWriter(fs, root, row_group_rows, file_rows, downloads)
    catalog = CatalogRollup()
    expected = completion.start_date
    total_days = (completion.end_date - completion.start_date).days + 1
    for completed_days, batch in enumerate(batches, start=1):
        if not isinstance(batch, WaveTrackingBatch):
            raise TypeError("batches must contain WaveTrackingBatch values")
        batch_date = _as_date(batch.date, "tracking batch date")
        if batch_date != expected or batch_date > completion.end_date:
            raise WaveRecordStorageError(
                f"expected tracking batch {expected}, got {batch_date}"
            )
        writer.flush_before_year(batch_date.year, ("wave_days", "lineage"))
        rows = sorted(
            (WaveDays.record_row(record, batch_date) for record in batch.current),
            key=_row_sort_key("wave_days"),
        )
        wave_ids: set[tuple[str, str, str]] = set()
        component_keys: set[tuple[str, str, str]] = set()
        for row in rows:
            temperature, direction = str(row["temperature"]), str(row["direction"])
            _require_selected_series(temperature, direction, completion)
            wave_identity = (temperature, direction, str(row["wave_id"]))
            component_key = (temperature, direction, str(row["component_id"]))
            if wave_identity in wave_ids or component_key in component_keys:
                raise WaveRecordStorageError(
                    "wave_days contains duplicate daily wave or component IDs"
                )
            wave_ids.add(wave_identity)
            component_keys.add(component_key)
            try:
                catalog.add_wave_day(row)
            except ValueError as error:
                raise WaveRecordStorageError(str(error)) from error
            writer.append("wave_days", row)
        for ended in batch.ended:
            if not isinstance(ended, EndedWave):
                raise TypeError("tracking batch endings must be EndedWave values")
            _require_selected_series(
                ended.final_record.series.temperature,
                ended.final_record.series.direction,
                completion,
            )
            try:
                catalog.add_ending(ended, batch_date)
            except ValueError as error:
                raise WaveRecordStorageError(str(error)) from error
        edge_rows = sorted(
            (Lineage.edge_row(edge, batch_date) for edge in batch.lineage),
            key=_row_sort_key("lineage"),
        )
        edge_keys: set[tuple[object, ...]] = set()
        for row in edge_rows:
            _require_selected_series(
                str(row["temperature"]), str(row["direction"]), completion
            )
            key = _lineage_key(row)
            if key in edge_keys:
                raise WaveRecordStorageError("lineage contains duplicate daily edges")
            edge_keys.add(key)
            writer.append("lineage", row)
        expected += timedelta(days=1)
        if completed_days % 365 == 0 or completed_days == total_days:
            progress(
                f"Catalog: daily batches {completed_days}/{total_days} through {batch_date}"
            )
    if expected != completion.end_date + timedelta(days=1):
        raise WaveRecordStorageError(f"missing tracking batch {expected}")
    writer.flush_tables(("wave_days", "lineage"))
    progress("Catalog: writing wave lifetime geometry and catalog rows")
    catalog_rows = 0
    for row in catalog.rows():
        writer.flush_before_year(cast("int", row["start_year"]), ("waves",))
        writer.append("waves", row)
        catalog_rows += 1
        if catalog_rows % 4096 == 0:
            progress(f"Catalog: {catalog_rows} wave rows written")
    progress(f"Catalog: flushing partitions after {catalog_rows} wave rows")
    return writer.finish()


def open_completed_generation(
    location: str | PathLike[str],
    *,
    filesystem: AbstractFileSystem | None = None,
) -> CompletedGeneration:
    """Open a completed generation for catalog, daily-record and lineage scans.

    Steps:
        1. Resolve the supplied filesystem location.
        2. Read the completion marker and manifest and validate the generation.

    Args:
        location: Caller-owned completed-generation directory or fsspec URL.
        filesystem: Optional generic fsspec filesystem.

    Returns:
        A completed-generation handle for the typed scan functions.

    Raises:
        WaveRecordStorageError: If the manifest or completion marker is invalid.

    Examples:
        generation = open_completed_generation("outputs/waves")
    """
    fs, path = _resolve_location(location, filesystem)
    return _open_complete_generation(
        fs, path, _read_bytes(fs, _join(path, "manifest.json"))
    )


def _open_complete_generation(
    fs: AbstractFileSystem,
    path: str,
    manifest_bytes: bytes,
) -> CompletedGeneration:
    """Open an internal completed-generation layout after its manifest is read.

    Steps:
        1. Decode and validate the generation manifest.
        2. Require matching completion metadata and construct the completed generation object.

    Args:
        fs: Generic fsspec filesystem that holds the generation.
        path: Resolved generation directory within ``fs``.
        manifest_bytes: Exact internal generation ``manifest.json`` content.

    Returns:
        A completed-output handle for selected scans.

    Raises:
        WaveRecordStorageError: If completion metadata or the internal
            generation layout is invalid.

    Example:
        generation = _open_complete_generation(filesystem, "generation", content)

    Examples:
        generation = _open_complete_generation(fs, root, manifest_bytes)
    """
    marker = _read_json(fs, _join(path, "_COMPLETE.json"))
    manifest = _load_manifest(manifest_bytes)
    if set(marker) != {"generation_id", "completed_at"}:
        msg = "completion marker has an unsupported schema"
        raise WaveRecordStorageError(msg)
    if not isinstance(marker.get("completed_at"), str):
        msg = "completion marker completed_at must be an ISO timestamp"
        raise WaveRecordStorageError(msg)
    try:
        completed_at = datetime.fromisoformat(cast("str", marker["completed_at"]))
    except ValueError as error:
        msg = "completion marker completed_at must be an ISO timestamp"
        raise WaveRecordStorageError(msg) from error
    if completed_at.tzinfo is None:
        msg = "completion marker completed_at must be timezone-aware"
        raise WaveRecordStorageError(msg)
    if marker.get("generation_id") != manifest.get("generation_id"):
        msg = "completion marker generation_id does not match manifest"
        raise WaveRecordStorageError(msg)
    if fs.exists(_join(path, "_INCOMPLETE.json")):
        raise WaveRecordStorageError(
            "completed generation still has an incomplete marker"
        )
    return CompletedGeneration(
        location=path,
        filesystem=fs,
        manifest=manifest,
        inventory=_inventory_from_manifest(manifest),
    )


def scan_wave_days(
    generation: CompletedGeneration,
    *,
    temperature: str | None = None,
    direction: str | None = None,
    start: date | None = None,
    end: date | None = None,
    wave_ids: Iterable[str] | None = None,
) -> gpd.GeoDataFrame:
    """Scan typed daily GeoParquet rows using branch, date, and ID filters.

    Steps:
        1. Select files matching series and date filters.
        2. Read the selected daily rows and apply wave ID filters.

    Args:
        generation: Verified generation returned by ``open_completed_generation``.
        temperature: Optional ``tasmax`` or ``tasmin`` branch filter.
        direction: Optional ``warm`` or ``cold`` branch filter.
        start: Optional inclusive observation date.
        end: Optional inclusive observation date.
        wave_ids: Optional persistent wave IDs to retain.

    Returns:
        A GeoDataFrame with the three wave geometry columns and partition fields.

    Examples:
        daily = scan_wave_days(generation, temperature="tasmax", direction="warm")
    """
    return cast(
        "gpd.GeoDataFrame",
        _scan(
            generation,
            "wave_days",
            temperature=temperature,
            direction=direction,
            start=start,
            end=end,
            active=None,
            wave_ids=wave_ids,
        ),
    )


def scan_waves(
    generation: CompletedGeneration,
    *,
    temperature: str | None = None,
    direction: str | None = None,
    active: bool | None = None,
    wave_ids: Iterable[str] | None = None,
) -> gpd.GeoDataFrame:
    """Scan typed catalog GeoParquet rows using branch, state, and ID filters.

    Steps:
        1. Select catalog files matching series filters.
        2. Read the selected rows and apply active-state and wave ID filters.

    Args:
        generation: Verified generation returned by ``open_completed_generation``.
        temperature: Optional ``tasmax`` or ``tasmin`` branch filter.
        direction: Optional ``warm`` or ``cold`` branch filter.
        active: Optional active-state filter.
        wave_ids: Optional persistent wave IDs to retain.

    Returns:
        A GeoDataFrame with lifetime footprints, catalog fields and partitions.

    Examples:
        ``scan_waves(generation, active=True)`` selects active lifetime footprints.
    """
    return cast(
        "gpd.GeoDataFrame",
        _scan(
            generation,
            "waves",
            temperature=temperature,
            direction=direction,
            start=None,
            end=None,
            active=active,
            wave_ids=wave_ids,
        ),
    )


def scan_lineage(
    generation: CompletedGeneration,
    *,
    temperature: str | None = None,
    direction: str | None = None,
    start: date | None = None,
    end: date | None = None,
    wave_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Scan typed lineage rows using branch, date, and endpoint filters.

    Steps:
        1. Select files matching series and event dates.
        2. Read the selected rows and filter their wave endpoints.

    Args:
        generation: Verified generation returned by ``open_completed_generation``.
        temperature: Optional ``tasmax`` or ``tasmin`` branch filter.
        direction: Optional ``warm`` or ``cold`` branch filter.
        start: Optional inclusive event date.
        end: Optional inclusive event date.
        wave_ids: Optional source or target wave IDs to retain.

    Returns:
        A DataFrame with lineage fields and their partition columns.

    Examples:
        edges = scan_lineage(generation, wave_ids=["20240101-01234567"])
    """
    return _scan(
        generation,
        "lineage",
        temperature=temperature,
        direction=direction,
        start=start,
        end=end,
        active=None,
        wave_ids=wave_ids,
    )


def _row_sort_key(
    table: Literal["wave_days", "lineage"],
) -> Callable[[dict[str, object]], tuple[object, ...]]:
    """Return the canonical row ordering key for a daily table.

    Steps:
        1. Select daily-record or lineage sort fields.
        2. Return a callable extracting those fields in canonical order.

    Args:
        table: Daily wave or lineage table name.

    Returns:
        Callable extracting ordered scalar fields from a row.

    Examples:
        key = _row_sort_key("wave_days")
    """
    if table == "wave_days":
        return lambda row: (
            row["temperature"],
            row["direction"],
            row["date"],
            row["wave_id"],
        )
    return lambda row: (
        row["temperature"],
        row["direction"],
        row["event_date"],
        row["relationship"],
        row["source_wave_id"],
        row["target_wave_id"],
    )


def _lineage_key(row: Mapping[str, object]) -> tuple[object, ...]:
    """Identify one split or merge within its daily batch.

    Steps:
        1. Read the date, relationship, and endpoint IDs.
        2. Return those fields as the deduplication key.

    Args:
        row: Schema-ready lineage row.

    Returns:
        Date, relationship and both wave identifiers used to reject duplicates.

    Examples:
        key = _lineage_key({"event_date": date(2020, 1, 2), "relationship": "split",
                            "source_wave_id": "a", "target_wave_id": "b"})
    """
    return (
        row["event_date"],
        row["relationship"],
        row["source_wave_id"],
        row["target_wave_id"],
    )


def _compact_parquet_parts(
    frame: pd.DataFrame, row_group_rows: int
) -> list[tuple[pd.DataFrame, bytes]]:
    """Encode one file buffer, splitting files that exceed the byte cap.

    Steps:
        1. Encode the row buffer to Parquet.
        2. Split and encode smaller buffers if the file exceeds the byte limit.

    Args:
        frame: Validated sorted rows for one selected series and year.
        row_group_rows: Maximum records per Parquet row group.

    Returns:
        Ordered DataFrame slices paired with encoded Parquet bytes.

    Raises:
        WaveRecordStorageError: If one record exceeds the file size limit.

    Examples:
        parts = _compact_parquet_parts(frame, 16384)
    """
    parts: list[tuple[pd.DataFrame, bytes]] = []
    pending = [frame]
    while pending:
        candidate = pending.pop(0)
        encoded = _encode_parquet(candidate, row_group_rows)
        if len(encoded) > _MAX_FILE_BYTES and len(candidate) > 1:
            midpoint = len(candidate) // 2
            pending[0:0] = [candidate.iloc[:midpoint], candidate.iloc[midpoint:]]
            continue
        if len(encoded) > _MAX_FILE_BYTES:
            raise WaveRecordStorageError(
                "one record exceeds the 512 MiB Parquet file limit"
            )
        parts.append((candidate, encoded))
    return parts


def _manifest(
    generation_id: str,
    entries: list[FileEntry],
    completion: CompletionContext,
    all_storms: CatalogEntry,
    all_waves: CatalogEntry,
) -> Manifest:
    """Describe the completed output without reading or hashing its files.

    Steps:
        1. Summarize partition files by table.
        2. Combine the summaries, completion context, and catalog entries into a manifest.

    Args:
        generation_id: Identifier shared by the manifest and completion marker.
        entries: Sorted inventory produced while writing scientific files.
        completion: Requested dates, selected series and package version.
        all_storms: Single directly readable named-storm catalog download.
        all_waves: Single directly readable full catalog download.

    Returns:
        Compact manifest with per-table totals.

    Examples:
        manifest = _manifest("run", entries, context, all_storms, all_waves)
    """
    tables: dict[str, DatasetSummary] = {}
    for name in _TABLES:
        selected = [entry for entry in entries if entry["table"] == name]
        tables[name] = {
            "rows": sum(entry["rows"] for entry in selected),
            "files": len(selected),
            "bytes": sum(entry["bytes"] for entry in selected),
        }
    return {
        "generation_id": generation_id,
        "series": {
            "temperature": completion.series.temperature,
            "direction": completion.series.direction,
        },
        "package": {
            "name": "reuters-temperature-storms",
            "version": completion.package_version,
        },
        "start_date": completion.start_date.isoformat(),
        "end_date": completion.end_date.isoformat(),
        "files": entries,
        "tables": tables,
        "all_storms": all_storms,
        "all_waves": all_waves,
    }


def _scan(
    generation: CompletedGeneration,
    table: RecordTable,
    *,
    temperature: str | None,
    direction: str | None,
    start: date | None,
    end: date | None,
    active: bool | None,
    wave_ids: Iterable[str] | None,
) -> pd.DataFrame:
    """Read selected files and filter scientific rows with pandas masks.

    Steps:
        1. Select manifest entries by partitions and requested dates.
        2. Read their Parquet frames and apply row-level filters.
        3. Return a typed empty frame when no rows match.

    Args:
        generation: Completed-output metadata and caller filesystem.
        table: Scientific dataset to read.
        temperature: Optional temperature filter.
        direction: Optional warm or cold filter.
        start: Optional inclusive first observation or event date.
        end: Optional inclusive last observation or event date.
        active: Optional catalog active-state filter.
        wave_ids: Optional IDs, matching either lineage endpoint.

    Returns:
        Deterministically ordered DataFrame, or GeoDataFrame for geometric records.
        Only selected partition files are opened. Each restored DataFrame is
        validated once, including its selected series and date/year agreement.

    Raises:
        WaveRecordStorageError: If filters are invalid or a file violates its schema.

    Examples:
        days = scan_wave_days(generation, start=date(2024, 1, 1))
    """
    if not isinstance(generation, CompletedGeneration):
        raise TypeError("generation must be opened with open_completed_generation")
    _validate_optional_branch(temperature, direction)
    if start is not None and end is not None and start > end:
        raise WaveRecordStorageError("start must not be after end")
    ids = None if wave_ids is None else tuple(wave_ids)
    if ids is not None:
        for wave_id in ids:
            _validate_wave_id(wave_id, "wave_ids value")
    year_field = _TABLE_CLASSES[table].partition_field
    table_class = _TABLE_CLASSES[table]
    geometry_columns = table_class.geometry_columns
    frames: list[pd.DataFrame] = []
    for entry in generation.manifest["files"]:
        partition = entry["partition"]
        if (
            entry["table"] != table
            or (temperature is not None and partition["temperature"] != temperature)
            or (direction is not None and partition["direction"] != direction)
            or (start is not None and int(partition[year_field]) < start.year)
            or (end is not None and int(partition[year_field]) > end.year)
        ):
            continue
        path = _join(generation.location, entry["path"])
        try:
            frame = table_class.read_parquet(
                path, filesystem=generation.filesystem, partition=partition
            ).frame
        except (OSError, ValueError, TypeError, SchemaError, SchemaErrors) as error:
            raise WaveRecordStorageError(
                f"cannot read valid {table} Parquet file {entry['path']}"
            ) from error
        date_field = "date" if table == "wave_days" else "event_date"
        mask = pd.Series(True, index=frame.index)
        if start is not None:
            mask &= frame[date_field] >= pd.Timestamp(start)
        if end is not None:
            mask &= frame[date_field] <= pd.Timestamp(end)
        if active is not None:
            mask &= frame["active"] == active
        if ids is not None:
            mask &= (
                frame["source_wave_id"].isin(ids) | frame["target_wave_id"].isin(ids)
                if table == "lineage"
                else frame["wave_id"].isin(ids)
            )
        frames.append(frame.loc[mask])
    columns = [
        "temperature",
        "direction",
        year_field,
        *list(table_class.file_schema.to_schema().columns),
    ]
    if frames:
        result = cast(
            "pd.DataFrame", pd.concat(frames, ignore_index=True).loc[:, columns]
        )
        return (
            gpd.GeoDataFrame(result, geometry=geometry_columns[-1])
            if geometry_columns
            else result
        )
    return cast("pd.DataFrame", table_class.empty_frame().loc[:, columns])


def _inventory_from_manifest(manifest: Manifest) -> GenerationInventory:
    """Build aggregate file and row counts from a checked manifest.

    Steps:
        1. Read each table's row, file, and byte totals.
        2. Sum those totals into the generation inventory.

    Args:
        manifest: Validated generation manifest.

    Returns:
        Generation inventory with per-table and total counts.

    Examples:
        inventory = _inventory_from_manifest(manifest)
    """
    datasets = {
        name: DatasetInventory(
            rows=int(manifest["tables"][name]["rows"]),
            objects=int(manifest["tables"][name]["files"]),
            bytes=int(manifest["tables"][name]["bytes"]),
        )
        for name in _TABLES
    }
    return GenerationInventory(
        generation_id=str(manifest["generation_id"]),
        datasets=datasets,
        rows=sum(item.rows for item in datasets.values()),
        objects=sum(item.objects for item in datasets.values()),
        bytes=sum(item.bytes for item in datasets.values()),
    )


def _validate_entry_path(entry: FileEntry) -> None:
    """Validate one manifest object path against its Hive partition fields.

    Steps:
        1. Validate the table-specific partition keys and types.
        2. Require the Parquet path prefix to match its series and year.

    Args:
        entry: Manifest file entry with its table and partition fields.

    Returns:
        None after validation or writing succeeds.

    Examples:
        _validate_entry_path(entry)
    """
    table = entry["table"]
    partition = entry["partition"]
    year_field = _TABLE_CLASSES[cast("RecordTable", table)].partition_field
    if set(partition) != {"temperature", "direction", year_field}:
        msg = "manifest partition fields must match the table Hive partitioning"
        raise WaveRecordStorageError(msg)
    temperature = partition.get("temperature")
    direction = partition.get("direction")
    year = partition.get(year_field)
    if (
        not isinstance(temperature, str)
        or not isinstance(direction, str)
        or isinstance(year, bool)
        or not isinstance(year, int)
    ):
        msg = "manifest partition fields have invalid types"
        raise WaveRecordStorageError(msg)
    _validate_branch(temperature, direction)
    expected_prefix = "/".join(
        (
            table,
            f"temperature={temperature}",
            f"direction={direction}",
            f"{year_field}={year}",
            "part-",
        )
    )
    if not entry["path"].startswith(expected_prefix) or not entry["path"].endswith(
        ".parquet"
    ):
        msg = "manifest path does not match its table or Hive branch/year partition"
        raise WaveRecordStorageError(msg)


def _validate_writer_settings(
    generation_id: str, row_group_rows: int, file_rows: int
) -> None:
    """Validate the generation label and Parquet row limits.

    Steps:
        1. Check the generation label uses permitted characters.
        2. Require positive row limits and a file limit at least as large as the row-group limit.

    Args:
        generation_id: Caller-selected generation label.
        row_group_rows: Maximum rows per Parquet row group.
        file_rows: Maximum rows per Parquet file.

    Returns:
        None when the label and row limits are supported.

    Examples:
        _validate_writer_settings("example", 16384, 65536)
    """
    if (
        not isinstance(generation_id, str)
        or not generation_id
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"  # pragma: allowlist secret
            for character in generation_id
        )
    ):
        msg = (
            "generation_id must contain only letters, numbers, hyphens, and underscores"
        )
        raise WaveRecordStorageError(msg)
    for name, value in (("row_group_rows", row_group_rows), ("file_rows", file_rows)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            msg = f"{name} must be a positive integer"
            raise WaveRecordStorageError(msg)
    if file_rows < row_group_rows:
        msg = "file_rows must be at least row_group_rows"
        raise WaveRecordStorageError(msg)


def _require_selected_series(
    temperature: str, direction: str, completion: CompletionContext
) -> None:
    """Reject scientific rows from a series outside the requested generation.

    Steps:
        1. Compare row temperature and direction with the requested series.
        2. Reject rows from an unselected series.

    Args:
        temperature: Row's temperature field.
        direction: Row's warm/cold direction.
        completion: Validated generation completion facts.

    Returns:
        None.

    Raises:
        WaveRecordStorageError: If the row's series was not selected.

    Examples:
        _require_selected_series("tasmax", "warm", completion)
    """
    if (temperature, direction) != (
        completion.series.temperature,
        completion.series.direction,
    ):
        msg = f"row series {temperature}/{direction} is outside the completed selection"
        raise WaveRecordStorageError(msg)


def _validate_completion_context(completion: CompletionContext) -> None:
    """Require an ordered requested range, supported series and package version.

    Steps:
        1. Validate requested dates and their ordering.
        2. Require supported series and a nonempty package version.

    Args:
        completion: Requested output facts supplied by the runner.

    Returns:
        None.

    Raises:
        TypeError: If the context or series has an unsupported type.
        WaveRecordStorageError: If dates, series or version are invalid.

    Examples:
        _validate_completion_context(context)
    """
    if not isinstance(completion, CompletionContext):
        raise TypeError("completion must be a CompletionContext")
    if not isinstance(completion.series, WaveSeries):
        raise TypeError("completion series must be a WaveSeries")
    _validate_branch(completion.series.temperature, completion.series.direction)
    _as_date(completion.start_date, "start_date")
    _as_date(completion.end_date, "end_date")
    if completion.start_date > completion.end_date:
        raise WaveRecordStorageError("start_date must not be after end_date")
    if (
        not isinstance(completion.package_version, str)
        or not completion.package_version
    ):
        raise WaveRecordStorageError("completion package version must be non-empty")


def _resolve_location(
    location: str | PathLike[str], filesystem: AbstractFileSystem | None
) -> tuple[AbstractFileSystem, str]:
    """Resolve a filesystem and its root from a caller-selected location.

    Steps:
        1. Convert the location to text.
        2. Resolve its protocol with fsspec or use the supplied filesystem.
        3. Return the filesystem and normalized root path.

    Args:
        location: Local path or filesystem URL.
        filesystem: Optional existing filesystem used to resolve the path.

    Returns:
        Resolved filesystem and root path.

    Examples:
        fs, root = _resolve_location("/tmp/waves", None)
    """
    value = _display_location(location)
    if filesystem is None:
        fs, path = fsspec.core.url_to_fs(value)
        return cast("AbstractFileSystem", fs), cast("str", path).rstrip("/")
    protocol, _ = fsspec.core.split_protocol(value)
    path = fsspec.core.strip_protocol(value) if protocol is not None else value
    return filesystem, _normalise_path(cast("str", path))


def _require_nonempty_location(path: str, name: str) -> None:
    """Reject a filesystem root before it can receive generation objects.

    Steps:
        1. Check the root path is not empty or the current directory.
        2. Reject locations that do not identify a child directory.

    Args:
        path: Resolved destination root path.
        name: Location label used in validation errors.

    Returns:
        None after validation or writing succeeds.

    Examples:
        _require_nonempty_location("example/waves", "destination")
    """
    if path in {"", "."}:
        msg = f"{name} must name a directory below the filesystem root"
        raise WaveRecordStorageError(msg)


def _require_new_destination(fs: AbstractFileSystem, root: str) -> None:
    """Require an output destination without existing generation files.

    Steps:
        1. Check whether the destination contains files.
        2. Reject either completed or incomplete existing output.

    Args:
        fs: Filesystem containing the destination.
        root: Generation root path.

    Returns:
        None when the destination is absent or empty.

    Examples:
        _require_new_destination(fs, "new-generation")
    """
    if not fs.exists(root):
        return
    files = fs.find(root)
    if files:
        if fs.exists(_join(root, "_COMPLETE.json")):
            msg = "destination is an accepted complete generation and is immutable"
        else:
            msg = "destination contains incomplete output; choose a new destination"
        raise WaveRecordStorageError(msg)


def _validate_branch(temperature: str, direction: str) -> None:
    """Validate a temperature and direction combination.

    Steps:
        1. Check temperature and direction against the shared choices.
        2. Reject unsupported combinations.

    Args:
        temperature: Candidate maximum/minimum temperature field.
        direction: Candidate warm/cold direction.

    Returns:
        None when both fields are supported.

    Examples:
        _validate_branch("tasmax", "warm")
    """
    if (
        temperature not in WaveSeries.temperatures
        or direction not in WaveSeries.directions
    ):
        msg = "records must use one of the four Reuters branches"
        raise WaveRecordStorageError(msg)


def _validate_optional_branch(temperature: str | None, direction: str | None) -> None:
    """Validate provided series filters without requiring both fields.

    Steps:
        1. Check each provided series filter independently.
        2. Reject unsupported temperature or direction values.

    Args:
        temperature: Optional temperature filter.
        direction: Optional direction filter.

    Returns:
        None when provided filters are supported.

    Examples:
        _validate_optional_branch(None, "warm")
    """
    if temperature is not None and temperature not in WaveSeries.temperatures:
        msg = "temperature must be tasmax or tasmin"
        raise WaveRecordStorageError(msg)
    if direction is not None and direction not in WaveSeries.directions:
        msg = "direction must be warm or cold"
        raise WaveRecordStorageError(msg)


def _validate_wave_id(value: object, name: str) -> None:
    """Validate one permanent wave ID.

    Steps:
        1. Check the compact date and hash-prefix layout.
        2. Reject IDs with invalid length, separator, or characters.

    Args:
        value: Candidate permanent wave ID.
        name: Field name included in any error.

    Returns:
        None when the value has birth-date/hash form.

    Examples:
        _validate_wave_id("19600105-94691d76", "wave_id")
    """
    if (
        not isinstance(value, str)
        or len(value) != 17
        or not value[:8].isdigit()
        or value[8] != "-"
        or any(letter not in _HEX for letter in value[9:])
    ):
        msg = f"{name} must be a YYYYMMDD-first8hash ID"
        raise WaveRecordStorageError(msg)


def _as_date(value: object, name: str) -> date:
    """Convert a tracker calendar date to a Python date through pandas.

    Steps:
        1. Convert the supplied value through pandas Timestamp.
        2. Reject missing dates and return its Python calendar date.

    Args:
        value: Tracker date value, including numpy calendar dates.
        name: Field name for an invalid value error.

    Returns:
        Corresponding calendar date without timezone or coordinate changes.

    Raises:
        WaveRecordStorageError: If the value is absent or cannot be a date.

    Examples:
        result = _as_date(date(2024, 1, 1), "date")
    """
    try:
        converted = pd.Timestamp(cast("date", value))
    except (TypeError, ValueError) as error:
        raise WaveRecordStorageError(f"{name} must be a calendar date") from error
    if pd.isna(converted):
        raise WaveRecordStorageError(f"{name} cannot be NaT")
    result = converted.date()
    if not isinstance(result, date):
        raise WaveRecordStorageError(f"{name} must be a calendar date")
    return result


def _encode_parquet(frame: pd.DataFrame, row_group_rows: int) -> bytes:
    """Serialize pandas or GeoPandas rows using their public Parquet writer.

    Steps:
        1. Write the frame to an in-memory Parquet stream.
        2. Return compressed bytes with the requested row-group size.

    Args:
        frame: Validated scientific DataFrame for one file.
        row_group_rows: Maximum records per row group.

    Returns:
        Encoded Parquet bytes with geometry metadata when applicable.

    Examples:
        content = _encode_parquet(frame, 16384)
    """
    sink = BytesIO()
    frame.to_parquet(
        sink, index=False, compression="zstd", row_group_size=row_group_rows
    )
    return sink.getvalue()


def _read_bytes(fs: AbstractFileSystem, path: str) -> bytes:
    """Read a required generation object as binary data.

    Steps:
        1. Open the object in binary read mode.
        2. Require bytes and translate filesystem read errors.

    Args:
        fs: Filesystem containing the object.
        path: Object path within the filesystem.

    Returns:
        Binary object content.

    Examples:
        content = _read_bytes(fs, "manifest.json")
    """
    try:
        with fs.open(path, "rb") as source:
            content = source.read()
            if not isinstance(content, bytes):
                msg = f"generation object {path} did not return bytes"
                raise WaveRecordStorageError(msg)
            return content
    except OSError as error:
        msg = f"cannot read required generation object {path}"
        raise WaveRecordStorageError(msg) from error


def _write_bytes(fs: AbstractFileSystem, path: str, content: bytes) -> None:
    """Write binary content to a generation object.

    Steps:
        1. Create the parent directory when needed.
        2. Write binary content and translate filesystem errors.

    Args:
        fs: Destination filesystem.
        path: Object path within the filesystem.
        content: Bytes to write.

    Returns:
        None after content is written.

    Examples:
        _write_bytes(fs, "example/data.bin", b"data")
    """
    parent = path.rpartition("/")[0]
    try:
        if parent:
            fs.makedirs(parent, exist_ok=True)
        with fs.open(path, "wb") as sink:
            sink.write(content)
    except OSError as error:
        msg = f"cannot write generation object {path}"
        raise WaveRecordStorageError(msg) from error


def _write_json(fs: AbstractFileSystem, path: str, value: object) -> None:
    """Write a JSON value with canonical serialization.

    Steps:
        1. Serialize the value using canonical JSON.
        2. Write the resulting bytes to the filesystem object.

    Args:
        fs: Destination filesystem.
        path: JSON object path.
        value: JSON-compatible content.

    Returns:
        None after the JSON object is written.

    Examples:
        _write_json(fs, "status.json", {"status": "ready"})
    """
    _write_bytes(fs, path, _canonical_json(value))


def _write_json_exclusive(fs: AbstractFileSystem, path: str, value: object) -> None:
    """Write a new JSON object without replacing a competing writer's marker.

    Steps:
        1. Create the parent directory when needed.
        2. Open the new object exclusively and write canonical JSON.
        3. Report an existing reservation without replacing it.

    Args:
        fs: Destination filesystem.
        path: New object path to reserve exclusively.
        value: JSON-compatible object content.

    Returns:
        None after validation or writing succeeds.

    Examples:
        _write_json_exclusive(fs, "_COMPLETE.json", {"generation_id": "example"})
    """
    parent = path.rpartition("/")[0]
    if parent:
        fs.makedirs(parent, exist_ok=True)
    try:
        with fs.open(path, "xb") as sink:
            sink.write(_canonical_json(value))
    except FileExistsError as error:
        msg = "destination was reserved by another generation writer"
        raise WaveRecordStorageError(msg) from error
    except OSError as error:
        msg = f"cannot reserve generation object {path}"
        raise WaveRecordStorageError(msg) from error


def _read_json(fs: AbstractFileSystem, path: str) -> dict[str, object]:
    """Read a generation object as a JSON mapping.

    Steps:
        1. Read the object bytes.
        2. Decode and require a JSON mapping.

    Args:
        fs: Filesystem containing the object.
        path: JSON object path.

    Returns:
        Decoded top-level JSON mapping.

    Examples:
        document = _read_json(fs, "manifest.json")
    """
    return _load_json(_read_bytes(fs, path), path)


def _load_json(content: bytes, name: str) -> dict[str, object]:
    """Decode JSON bytes and require an object at the top level.

    Steps:
        1. Decode UTF-8 JSON bytes.
        2. Reject malformed JSON or a non-object top-level value.

    Args:
        content: UTF-8 JSON bytes.
        name: Object name used in decoding errors.

    Returns:
        Decoded JSON mapping.

    Examples:
        assert _load_json(b'{"rows": 1}', "example") == {"rows": 1}
    """
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        msg = f"{name} must be valid UTF-8 JSON"
        raise WaveRecordStorageError(msg) from error
    if not isinstance(value, dict):
        msg = f"{name} must be a JSON object"
        raise WaveRecordStorageError(msg)
    return cast("dict[str, object]", value)


def _load_manifest(content: bytes) -> Manifest:
    """Read the compact inventory and check metadata needed for file selection.

    Steps:
        1. Decode JSON and validate required manifest fields.
        2. Check table inventories, partition paths, and completion metadata.

    Args:
        content: UTF-8 JSON manifest bytes.

    Returns:
        Typed manifest describing one completed generation.

    Raises:
        WaveRecordStorageError: If metadata cannot safely select its files.

    Examples:
        manifest = _load_manifest(Path("manifest.json").read_bytes())
    """
    value = _load_json(content, "manifest")
    if set(value) != {
        "generation_id",
        "series",
        "package",
        "start_date",
        "end_date",
        "files",
        "tables",
        "all_storms",
        "all_waves",
    }:
        raise WaveRecordStorageError("manifest has an unsupported schema")
    raw_files, raw_tables = value["files"], value["tables"]
    if not isinstance(raw_files, list) or not isinstance(raw_tables, dict):
        raise WaveRecordStorageError("manifest has malformed file or table inventory")
    if not all(
        isinstance(entry, dict)
        and set(entry) == {"path", "table", "partition", "rows", "row_groups", "bytes"}
        and isinstance(entry["path"], str)
        and entry["table"] in _TABLES
        and isinstance(entry["partition"], dict)
        and all(
            isinstance(entry[field], int)
            and not isinstance(entry[field], bool)
            and entry[field] > 0
            for field in ("rows", "row_groups", "bytes")
        )
        for entry in raw_files
    ):
        raise WaveRecordStorageError("manifest has malformed file inventory entries")
    inventory = cast("list[FileEntry]", raw_files)
    paths = [entry["path"] for entry in inventory]
    if paths != sorted(set(paths)) or not all(
        _is_safe_relative_path(path) for path in paths
    ):
        raise WaveRecordStorageError(
            "manifest file inventory must have safe unique sorted paths"
        )
    if set(raw_tables) != set(_TABLES) or not all(
        _is_dataset_summary(raw_tables[name]) for name in _TABLES
    ):
        raise WaveRecordStorageError("manifest has malformed table inventory")
    manifest = cast("Manifest", value)
    for name in ("all_storms", "all_waves"):
        catalog = value[name]
        if (
            not isinstance(catalog, dict)
            or set(catalog) != {"path", "rows", "bytes"}
            or catalog["path"] != f"{name}.parquet"
            or any(
                not isinstance(catalog[field], int)
                or isinstance(catalog[field], bool)
                or catalog[field] < (1 if field == "bytes" else 0)
                for field in ("rows", "bytes")
            )
        ):
            raise WaveRecordStorageError(f"manifest has malformed {name} download")
    try:
        context = _completion_from_manifest(manifest)
        _validate_writer_settings(manifest["generation_id"], 1, 1)
        for entry in inventory:
            _validate_entry_path(entry)
            _require_selected_series(
                str(entry["partition"]["temperature"]),
                str(entry["partition"]["direction"]),
                context,
            )
    except (TypeError, KeyError, ValueError) as error:
        raise WaveRecordStorageError(
            "manifest has invalid completion metadata"
        ) from error
    return manifest


def _is_safe_relative_path(path: str) -> bool:
    """Return whether one generation-relative object path is safe.

    Steps:
        1. Reject absolute paths and empty or traversal segments.
        2. Require a normalized generation-relative path.

    Args:
        path: Relative POSIX generation path.

    Returns:
        Whether ``path`` is nonempty, relative, normalized, and excludes parent
        traversal.

    Example:
        assert _is_safe_relative_path("waves/part-00000.parquet")

    Examples:
        assert _is_safe_relative_path("waves/part-000.parquet")
    """
    return (
        bool(path)
        and not path.startswith("/")
        and "\\" not in path
        and path != "."
        and _normalise_path(path) == path
        and all(part not in {"", ".", ".."} for part in path.split("/"))
    )


def _completion_from_manifest(manifest: Manifest) -> CompletionContext:
    """Restore the requested range and selected series from completion metadata.

    Steps:
        1. Read request dates and series from the manifest.
        2. Validate the restored completion context.

    Args:
        manifest: Decoded completed-output manifest.

    Returns:
        Validated completion context.

    Raises:
        WaveRecordStorageError: If dates, package or selected series are invalid.

    Examples:
        context = _completion_from_manifest(manifest)
    """
    package, series = manifest["package"], manifest["series"]
    if (
        not isinstance(package, dict)
        or set(package) != {"name", "version"}
        or package["name"] != "reuters-temperature-storms"
        or not isinstance(series, dict)
        or set(series) != {"temperature", "direction"}
        or not all(isinstance(value, str) for value in series.values())
    ):
        raise WaveRecordStorageError("manifest package or series is malformed")
    context = CompletionContext(
        start_date=_manifest_date(manifest, "start_date"),
        end_date=_manifest_date(manifest, "end_date"),
        series=WaveSeries(
            cast("Literal['tasmax', 'tasmin']", series["temperature"]),
            cast("Literal['warm', 'cold']", series["direction"]),
        ),
        package_version=package["version"],
    )
    _validate_completion_context(context)
    return context


def _manifest_date(mapping: Mapping[str, object], name: str) -> date:
    """Return one required ISO calendar date from a decoded manifest object.

    Steps:
        1. Read the required string field.
        2. Parse it as an ISO calendar date or raise a storage error.

    Args:
        mapping: Decoded manifest mapping.
        name: Required date field name.

    Returns:
        None after validation or writing succeeds.

    Examples:
        assert _manifest_date({"start_date": "2024-01-01"}, "start_date") == date(2024, 1, 1)
    """
    value = mapping.get(name)
    if not isinstance(value, str):
        msg = f"manifest {name} must be an ISO date"
        raise WaveRecordStorageError(msg)
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        msg = f"manifest {name} must be an ISO date"
        raise WaveRecordStorageError(msg) from error


def _is_dataset_summary(value: object) -> bool:
    """Return whether a decoded value holds nonnegative row, file and byte totals.

    Steps:
        1. Require exactly the row, file, and byte totals.
        2. Check that each is a nonnegative integer rather than Boolean.

    Args:
        value: Decoded table inventory value.

    Returns:
        Whether the value has the three required nonnegative integer totals.

    Examples:
        assert _is_dataset_summary({"rows": 1, "files": 1, "bytes": 200})
    """
    return (
        isinstance(value, dict)
        and set(value) == {"rows", "files", "bytes"}
        and all(
            isinstance(value[field], int)
            and not isinstance(value[field], bool)
            and value[field] >= 0
            for field in ("rows", "files", "bytes")
        )
    )


def _canonical_json(value: object) -> bytes:
    """Serialize compact sorted JSON with a terminal newline.

    Steps:
        1. Sort JSON keys and use compact separators.
        2. Reject nonfinite values and append a newline before UTF-8 encoding.

    Args:
        value: JSON-compatible content with finite numbers.

    Returns:
        UTF-8 JSON bytes ending with a newline.

    Examples:
        content = _canonical_json({"rows": 1})
    """
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _join(*parts: str) -> str:
    """Join filesystem path fragments without doubled boundary separators.

    Steps:
        1. Remove empty path fragments.
        2. Join remaining fragments with a single separator at each boundary.

    Args:
        *parts: Path fragments, with empty fragments ignored.

    Returns:
        Joined path, or an empty string when all fragments are empty.

    Examples:
        assert _join("root/", "/waves", "") == "root/waves"
    """
    nonempty = [part for part in parts if part != ""]
    if not nonempty:
        return ""
    return nonempty[0].rstrip("/") + "".join(
        "/" + part.strip("/") for part in nonempty[1:]
    )


def _display_location(location: str | PathLike[str]) -> str:
    """Convert a path-like location to its text representation.

    Steps:
        1. Convert the path-like location to text.

    Args:
        location: Filesystem URL or path-like location.

    Returns:
        Location text.

    Examples:
        assert _display_location("/tmp/waves") == "/tmp/waves"
    """
    return str(location)


def _normalise_path(path: str) -> str:
    """Normalize fsspec path segments before safety-sensitive comparisons.

    Steps:
        1. Normalize POSIX path segments.
        2. Remove trailing separators for path comparisons.

    Args:
        path: Filesystem path to normalize.

    Returns:
        None after validation or writing succeeds.

    Examples:
        assert _normalise_path("waves/./data/") == "waves/data"
    """
    return posixpath.normpath(path).rstrip("/")
