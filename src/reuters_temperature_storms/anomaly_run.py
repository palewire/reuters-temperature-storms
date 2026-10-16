"""Read and build independent daily anomaly files with caller-selected access."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, ClassVar, TypeVar

import pandas as pd
import pyarrow.parquet as pq

from .anomalies import (
    AnomalyError,
    DailyAnomalies,
    DailyAnomalyBuilder,
    DailyClimatology,
    DailyTemperatures,
    MeasurementGrid,
    SeasonalDistribution,
)
from .daily_run import DailyRunExecutor
from .source import DailyFileSource

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from .series import WaveSeries


Grid = TypeVar("Grid", bound=MeasurementGrid)


class AnomalyFileSource(DailyFileSource):
    """Flat daily or calendar-day Parquets with configurable filename templates.

    Steps:
        1. Configure the caller-selected file location and access options.
        2. Resolve and validate daily files through the source's read methods.

    Args:
        location: Local directory or fsspec URL for input files.
        storage_options: Caller-selected filesystem access settings.
        filename_template: Flat Parquet filename with date, month or day placeholders.

    Examples:
        source = AnomalyFileSource("./daily")
    """

    __slots__ = ("filename_template",)
    error_type: ClassVar[type[ValueError]] = AnomalyError
    kind: ClassVar[str] = "daily-anomalies"

    def __init__(
        self,
        location: str,
        storage_options: dict[str, object] | None = None,
        *,
        filename_template: str = "{date}.parquet",
    ) -> None:
        """Configure a caller-owned root and date-formatted filename.

        Steps:
            1. Configure the caller-selected root and filesystem access.
            2. Retain the filename template and validate its calendar-day formatting.

        Args:
            location: Local or fsspec root directory.
            storage_options: Caller-selected filesystem settings.
            filename_template: Filename using date, month and day placeholders.

        Returns:
            None. Retains the filesystem and filename settings.

        Examples:
            source = AnomalyFileSource("normal", filename_template="{month:02d}-{day:02d}.parquet")
        """
        super().__init__(location, storage_options)
        self.filename_template = filename_template
        self.path_for(date(2000, 2, 29))

    def path_for(self, day: date, series: WaveSeries | None = None) -> str:
        """Select one safe filename using the requested calendar date.

        Steps:
            1. Require a plain calendar date and format its date, month and day placeholders.
            2. Require a flat Parquet filename and place it beneath the configured root.

        Args:
            day: Plain requested date.
            series: Unused shared-source selection; anomalies contain every metric.

        Returns:
            Filename beneath the configured root.

        Examples:
            path = source.path_for(date(2024, 7, 1))
        """
        if not isinstance(day, date) or isinstance(day, datetime):
            raise AnomalyError("day must be a plain calendar date")
        try:
            filename = self.filename_template.format(
                date=day.isoformat(), month=day.month, day=day.day
            )
        except (ValueError, KeyError, AttributeError) as error:
            raise AnomalyError(f"invalid filename template: {error}") from error
        if "/" in filename or "\\" in filename or not filename.endswith(".parquet"):
            raise AnomalyError("filename template must select a flat Parquet filename")
        return f"{self._path}/{filename}"

    def read_grid(self, day: date, grid_type: type[Grid]) -> Grid:
        """Select needed columns, expand to float64 and validate the input grid.

        Steps:
            1. Select declared scientific fields and the existing date/calendar labels.
            2. Read only these columns; expand floating inputs before calculation.
            3. For raw temperatures only, require timezone-free timestamps on the
               requested day and remove their time of day.
            4. Validate the complete global grid and requested date.

        Args:
            day: Requested calendar date.
            grid_type: Accepted temperature, baseline or anomaly dataclass.

        Returns:
            Validated float64 input grid.

        Examples:
            daily = source.read_grid(day, DailyTemperatures)
        """
        columns = list(grid_type.schema.to_schema().columns)
        with self._filesystem.open(self.path_for(day), "rb") as stream:
            frame = pq.read_table(stream, columns=columns).to_pandas()
        if grid_type is DailyTemperatures:
            dates = frame["date"]
            if not pd.api.types.is_datetime64_dtype(dates.dtype):
                raise AnomalyError(
                    "temperature dates must be timezone-free datetime values"
                )
            calendar_dates = dates.dt.normalize()
            if not calendar_dates.eq(pd.Timestamp(day)).all():
                raise AnomalyError(f"temperature calendar dates differ from {day}")
            frame["date"] = calendar_dates
        for column in columns:
            if column not in ("date", "month", "day"):
                if not pd.api.types.is_float_dtype(frame[column].dtype):
                    raise AnomalyError(f"{column} must be a floating measurement")
                if grid_type is DailyAnomalies and frame[column].dtype != "float64":
                    raise AnomalyError(f"stored anomaly {column} must be float64")
                frame[column] = frame[column].astype("float64")
        return grid_type(frame, day)


@dataclass(frozen=True, slots=True)
class AnomalyResult:
    """Small completed-file status retained outside the worker.

    Steps:
        1. Retain the supplied fields for this record.
        2. Expose the values to the corresponding processing and validation methods.

    Args:
        day: Finished calendar date.
        rows: Complete global row count.
        bytes: Stored file size.
        elapsed_seconds: Worker duration.
        skipped: Whether an existing accepted file was reused.

    Examples:
        result = AnomalyResult(day, 1038240, 4096, 0.2, False)
    """

    day: date
    rows: int
    bytes: int
    elapsed_seconds: float
    skipped: bool


@dataclass(frozen=True, slots=True)
class AnomalyTask:
    """One independent date and serialized filesystem settings.

    Steps:
        1. Retain the supplied fields for this record.
        2. Expose the values to the corresponding processing and validation methods.

    Args:
        day: Output date.
        sources: Temperature, ordinary, adjusted and output root settings.
        force_update: Whether to replace existing output.

    Examples:
        result = task.run()
    """

    day: date
    sources: tuple[tuple[str, dict[str, object], str], ...]
    force_update: bool

    def run(self) -> AnomalyResult:
        """Read, calculate and save one date, or validate its existing output.

        Steps:
            1. Configure file sources for the worker and select the daily output.
            2. Validate an existing output when reuse is allowed; otherwise read inputs, calculate anomalies and save them.
            3. Return row count, byte size, elapsed time and reuse status.

        Args:
            None.

        Returns:
            Small file status without transferring global tables.

        Examples:
            result = task.run()
        """
        started = time.perf_counter()
        temperature, ordinary, adjusted, destination = tuple(
            AnomalyFileSource(location, options, filename_template=template)
            for location, options, template in self.sources
        )
        path = destination.path_for(self.day)
        skipped = not self.force_update and destination._filesystem.exists(path)
        if skipped:
            anomalies = destination.read_grid(self.day, DailyAnomalies)
        else:
            daily = temperature.read_grid(self.day, DailyTemperatures)
            normal = ordinary.read_grid(self.day, DailyClimatology)
            seasonal = adjusted.read_grid(self.day, SeasonalDistribution)
            anomalies = DailyAnomalyBuilder().build(daily, normal, seasonal)
            destination._filesystem.makedirs(destination._path, exist_ok=True)
            with destination._filesystem.open(path, "wb") as stream:
                anomalies.frame.to_parquet(stream, index=False)
        return AnomalyResult(
            self.day,
            len(anomalies.frame),
            destination._filesystem.size(path),
            time.perf_counter() - started,
            skipped,
        )


@dataclass(frozen=True, slots=True)
class AnomalyRun:
    """Build complete daily anomalies before five-day weighted scores.

    Steps:
        1. Retain the supplied fields for this record.
        2. Expose the values to the corresponding processing and validation methods.

    Args:
        temperature: Daily temperature root.
        climatology: Ordinary calendar mean root.
        seasonal: Adjusted calendar distribution root.
        destination: Separate dated output root.
        start_date: First included date.
        end_date: Last included date.
        max_workers: Process count or available CPUs.
        force_update: Replace existing outputs.

    Examples:
        results = AnomalyRun(daily, normal, adjusted, output, start, end, 8).run()
    """

    temperature: AnomalyFileSource
    climatology: AnomalyFileSource
    seasonal: AnomalyFileSource
    destination: AnomalyFileSource
    start_date: date
    end_date: date
    max_workers: int | None = None
    force_update: bool = False

    def validate(self) -> None:
        """Check ordered dates, worker count and separate writable output roots.

        Steps:
            1. Validate source types, ordered dates, worker count and replacement flag.
            2. Require full dated output filenames and output roots separate from every input.

        Args:
            None.

        Returns:
            None. Raises AnomalyError for invalid settings.

        Examples:
            job.validate()
        """
        if not all(
            isinstance(source, AnomalyFileSource)
            for source in (
                self.temperature,
                self.climatology,
                self.seasonal,
                self.destination,
            )
        ):
            raise AnomalyError(
                "all inputs and output must be AnomalyFileSource objects"
            )
        if not isinstance(self.force_update, bool):
            raise AnomalyError("force_update must be boolean")
        for day in (self.start_date, self.end_date):
            if not isinstance(day, date) or isinstance(day, datetime):
                raise AnomalyError("run dates must be plain calendar dates")
        if self.start_date > self.end_date:
            raise AnomalyError("start_date must not follow end_date")
        if self.max_workers is not None and (
            isinstance(self.max_workers, bool)
            or not isinstance(self.max_workers, int)
            or self.max_workers < 1
        ):
            raise AnomalyError("max_workers must be a positive integer")
        if self.destination.filename_template != "{date}.parquet":
            raise AnomalyError("output must use complete dated filenames")
        for source in (self.temperature, self.climatology, self.seasonal):
            if source._filesystem.protocol == self.destination._filesystem.protocol:
                source_root, output_root = (
                    source._path.rstrip("/"),
                    self.destination._path.rstrip("/"),
                )
                if (
                    source_root == output_root
                    or source_root.startswith(output_root + "/")
                    or output_root.startswith(source_root + "/")
                ):
                    raise AnomalyError("output root must be separate from all inputs")

    def dates(self) -> Iterator[date]:
        """Yield every included date once in calendar order.

        Steps:
            1. Yield the first requested date.
            2. Advance one day at a time until the inclusive end date.

        Args:
            None.

        Yields:
            Each plain calendar date in the requested inclusive range.

        Examples:
            dates = tuple(job.dates())
        """
        current = self.start_date
        while True:
            yield current
            if current == self.end_date:
                break
            current += timedelta(days=1)

    def check(self) -> tuple[date, ...]:
        """Validate existing outputs and report missing dates without source reads.

        Steps:
            1. Validate run settings and visit each requested date.
            2. Validate existing anomaly files and collect missing dates without reading source inputs.

        Args:
            None.

        Returns:
            Missing dates in calendar order.

        Examples:
            missing = job.check()
        """
        self.validate()
        missing = []
        for day in self.dates():
            if self.destination._filesystem.exists(self.destination.path_for(day)):
                self.destination.read_grid(day, DailyAnomalies)
            else:
                missing.append(day)
        return tuple(missing)

    def run(
        self, progress: Callable[[AnomalyResult], None] | None = None
    ) -> tuple[AnomalyResult, ...]:
        """Build independent dates in workers while retaining completed files.

        Steps:
            1. Validate settings and serialize file-source configuration for each task.
            2. Choose workers within the requested date count.
            3. Execute independent dates and return ordered statuses after all succeed.

        Args:
            progress: Optional callback after each completed or reused output.

        Returns:
            Date-ordered file statuses after every requested date succeeds.

        Examples:
            results = job.run(progress=lambda result: print(result.day))
        """
        self.validate()
        sources = tuple(
            (source.location, source.storage_options, source.filename_template)
            for source in (
                self.temperature,
                self.climatology,
                self.seasonal,
                self.destination,
            )
        )
        tasks = (AnomalyTask(day, sources, self.force_update) for day in self.dates())
        workers = min(
            self.max_workers or os.cpu_count() or 1,
            (self.end_date - self.start_date).days + 1,
        )
        return DailyRunExecutor(
            tasks,
            workers,
            DailyRunExecutor.configure_arrow,
            "daily anomaly",
            AnomalyError,
        ).run(progress)
