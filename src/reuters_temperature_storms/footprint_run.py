"""Create reusable daily wave footprints in independent worker processes."""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

from .cells import WaveCells, WaveCellSource
from .daily_run import DailyRunExecutor
from .footprint_geometry import WaveFootprintBuilder
from .footprints import WaveFootprints, WaveFootprintSource
from .series import WaveSeries

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


class FootprintRunError(ValueError):
    """Raised when daily footprint preparation cannot complete the requested dates.

    Steps:
        1. Describe a rejected input, configuration or stage result.
        2. Propagate the message through callers without changing data.

    Args:
        *args: Exception message and any additional exception arguments.

    Examples:
        raise FootprintRunError("invalid input")
    """


@dataclass(frozen=True, slots=True)
class FootprintResult:
    """Small worker result without transporting cell or footprint geometry.

    Steps:
        1. Retain the supplied fields for this record.
        2. Expose the values to the corresponding processing and validation methods.

    Args:
        day: Completed calendar date.
        rows: Number of daily footprints in the file.
        bytes: Stored file size.
        elapsed_seconds: Worker read, calculation and write duration.
        skipped: Whether an accepted existing output was reused.

    Examples:
        result = FootprintResult(date(2024, 7, 1), 2, 4096, 0.2, False)
    """

    day: date
    rows: int
    bytes: int
    elapsed_seconds: float
    skipped: bool


@dataclass(frozen=True, slots=True)
class FootprintTask:
    """Picklable file-access settings for one independent date.

    Steps:
        1. Retain the supplied fields for this record.
        2. Expose the values to the corresponding processing and validation methods.

    Args:
        source_location: Root of daily mature wave cells.
        destination_location: Root for reusable daily footprint files.
        source_options: Caller-selected cell filesystem access settings.
        destination_options: Caller-selected output filesystem access settings.
        empty_dates: Explicit source dates permitted to be missing.
        day: Date this worker owns.
        series: Maximum/minimum and warm/cold selection.
        force_update: Whether to replace an existing output.

    Examples:
        result = task.run()
    """

    source_location: str
    destination_location: str
    source_options: dict[str, object]
    destination_options: dict[str, object]
    empty_dates: tuple[date, ...]
    day: date
    series: WaveSeries
    force_update: bool

    @staticmethod
    def configure_worker() -> None:
        """Keep Arrow's internal thread pools from multiplying across processes.

        Steps:
            1. Load Arrow inside the worker process.
            2. Limit its CPU pool to one thread and its file-reading pool to one
               thread, so each worker does not start extra pools of threads.

        Args:
            None.

        Returns:
            None. Each process uses one Arrow CPU and I/O thread.

        Examples:
            FootprintTask.configure_worker()
        """
        DailyRunExecutor.configure_arrow()

    def run(self) -> FootprintResult:
        """Read, build and publish one date entirely inside its worker.

        Steps:
            1. Open the configured cell source and footprint destination and
               select this date's output path.
            2. Unless replacement was requested, check any existing footprint
               file, including its date and geometry-based identifiers, and reuse
               it when accepted.
            3. Otherwise read and check this date's cells. A missing date can make
               an empty footprint file only if it was declared empty; a declared
               empty date cannot contain cell rows.
            4. Build the daily wave footprints and write them to the output path.
            5. Return only the date, row count, file size and elapsed time. Keep
               the large cell and footprint tables inside this worker.

        Args:
            None.

        Returns:
            File counts, size and duration. Cell and footprint tables remain here.

        Raises:
            ValueError: If input or an existing output is invalid.
            OSError: If reading or writing fails.

        Examples:
            result = task.run()
        """
        started = time.perf_counter()
        source = WaveCellSource(
            self.source_location,
            self.source_options,
            known_empty_dates=self.empty_dates,
        )
        destination = WaveFootprintSource(
            self.destination_location, self.destination_options
        )
        output_path = destination.path_for(self.day, self.series)
        if not self.force_update and destination._filesystem.exists(output_path):
            existing = WaveFootprints.read_parquet(
                output_path, filesystem=destination._filesystem
            )
            existing.validate_day(self.day, self.series)
            return FootprintResult(
                self.day,
                len(existing.frame),
                destination._filesystem.size(output_path),
                time.perf_counter() - started,
                True,
            )
        input_path = source.path_for(self.day, self.series)
        try:
            cells = WaveCells.read_parquet(input_path, filesystem=source._filesystem)
        except FileNotFoundError:
            if self.day not in self.empty_dates:
                raise
            footprints = WaveFootprints(WaveFootprintBuilder.empty_frame())
        else:
            if self.day in self.empty_dates and not cells.frame.empty:
                raise FootprintRunError(
                    f"{input_path}: declared empty date contains wave cells"
                )
            footprints = WaveFootprints.from_cells(cells, self.day, self.series)
        footprints.to_parquet(output_path, filesystem=destination._filesystem)
        return FootprintResult(
            self.day,
            len(footprints.frame),
            destination._filesystem.size(output_path),
            time.perf_counter() - started,
            False,
        )


@dataclass(frozen=True, slots=True)
class FootprintRun:
    """Prepare a date range before the chronological tracker starts.

    Independent instances may process nonoverlapping ranges into one destination.
    Each date is read, calculated and written by a worker, so large geometry
    results never accumulate in the parent process.

    Steps:
        1. Retain the supplied fields for this record.
        2. Expose the values to the corresponding processing and validation methods.

    Args:
        source: Mature wave-cell files with caller-selected access.
        destination: Root for footprint files with caller-selected access.
        start_date: First included date.
        end_date: Last included date.
        series: One maximum/minimum and warm/cold series.
        max_workers: Worker processes, or available CPUs when omitted.
        force_update: Replace existing output files instead of checking and reusing.

    Examples:
        results = FootprintRun(source, destination, start, end, series, 8).run()
    """

    source: WaveCellSource
    destination: WaveFootprintSource
    start_date: date
    end_date: date
    series: WaveSeries
    max_workers: int | None = None
    force_update: bool = False

    def validate(self) -> None:
        """Check the requested range and access settings before starting workers.

        Steps:
            1. Require a wave-cell source and wave-footprint destination.
            2. Require plain calendar dates in order and one supported series.
            3. Require a positive worker count when supplied and a boolean choice
               for replacing existing output files.
            4. Reject a destination that is the same directory on the same
               filesystem as the input, so writing cannot replace cell files.

        Args:
            None.

        Returns:
            None when every request setting is supported.

        Raises:
            FootprintRunError: If dates, models, workers or destinations disagree.

        Examples:
            run.validate()
        """
        if not isinstance(self.source, WaveCellSource) or not isinstance(
            self.destination, WaveFootprintSource
        ):
            raise FootprintRunError(
                "source and destination must be wave-cell and wave-footprint sources"
            )
        if (
            any(
                not isinstance(value, date) or isinstance(value, datetime)
                for value in (self.start_date, self.end_date)
            )
            or self.start_date > self.end_date
        ):
            raise FootprintRunError(
                "start_date and end_date must be ordered calendar dates"
            )
        if not isinstance(self.series, WaveSeries):
            raise FootprintRunError("series must use tasmax/tasmin and warm/cold")
        try:
            self.series.validate()
        except ValueError as error:
            raise FootprintRunError(str(error)) from error
        if self.max_workers is not None and (
            type(self.max_workers) is not int or self.max_workers < 1
        ):
            raise FootprintRunError("max_workers must be a positive integer")
        if type(self.force_update) is not bool:
            raise FootprintRunError("force_update must be boolean")
        if (
            self.source._filesystem.protocol == self.destination._filesystem.protocol
            and self.source._path == self.destination._path
        ):
            raise FootprintRunError(
                "wave-cell input and footprint output must be different directories"
            )

    def tasks(self) -> Iterator[FootprintTask]:
        """Describe each date without loading any input table.

        Steps:
            1. Start at the first included date.
            2. Create one task with that date, the series, file locations, access
               settings, declared empty dates and replacement choice.
            3. Advance one calendar day and repeat through the last included date.
               Describe file access only; do not read any cells in this method.

        Args:
            None.

        Returns:
            Date-ordered tasks containing only file settings and the request.

        Examples:
            tasks = list(run.tasks())
        """
        current = self.start_date
        while current <= self.end_date:
            yield FootprintTask(
                self.source.location,
                self.destination.location,
                self.source.storage_options,
                self.destination.storage_options,
                self.source.known_empty_dates,
                current,
                self.series,
                self.force_update,
            )
            current += timedelta(days=1)

    def run(
        self, progress: Callable[[FootprintResult], None] | None = None
    ) -> tuple[FootprintResult, ...]:
        """Build all dates in parallel, retaining successes if another date fails.

        Steps:
            1. Check the request and choose a worker count no greater than the
               number of dates. Use the requested count or available CPUs.
            2. With one worker, process dates directly. With several workers,
               start separate processes and keep at most twice the worker count
               submitted at a time.
            3. Each task reads, builds and writes its own date. Retain only its
               small result in this process and call progress after a success.
            4. Continue other dates after a date fails, retaining successful files
               and collecting the failing dates and their error messages.
            5. Report all dated failures, or return successful results in date
               order. If a worker process stops unexpectedly, report that as well.
               Rerunning checks and reuses accepted files, then retries missing
               dates. Use fewer workers if available memory caused the failure.

        Args:
            progress: Optional callback for each finished or reused date.

        Returns:
            Date-ordered file results after all dates succeed.

        Raises:
            FootprintRunError: If any date fails; the message identifies each date.

        Examples:
            results = run.run(progress=lambda result: print(result.day))
        """
        self.validate()
        count = (self.end_date - self.start_date).days + 1
        cpu_count = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else os.cpu_count() or 1
        )
        workers = min(self.max_workers or cpu_count, count)
        return DailyRunExecutor[FootprintResult](
            self.tasks(),
            workers,
            FootprintTask.configure_worker,
            "footprint",
            FootprintRunError,
            ProcessPoolExecutor,
        ).run(progress)

    def check(self) -> tuple[date, ...]:
        """Check every output before starting the tracker over a completed stage.

        Steps:
            1. Check the requested range and source/destination settings.
            2. Visit every date's footprint path and read any file that exists.
            3. Check the table, represented date and geometry-based identifiers,
               including files that represent an empty day.
            4. Collect missing dates separately from invalid files. Report invalid
               files together, or return missing dates for the next run to build.

        Args:
            None.

        Returns:
            Missing calendar dates. Present files must satisfy the footprint schema,
            represented date and raw geometry hash, including empty-date files.

        Raises:
            FootprintRunError: If an existing output is invalid.

        Examples:
            missing = run.check()
        """
        self.validate()
        missing = []
        errors = []
        for task in self.tasks():
            path = self.destination.path_for(task.day, self.series)
            try:
                footprints = WaveFootprints.read_parquet(
                    path, filesystem=self.destination._filesystem
                )
                footprints.validate_day(task.day, self.series)
            except FileNotFoundError:
                missing.append(task.day)
            except (ValueError, OSError) as error:
                errors.append(f"{task.day}: {error}")
        if errors:
            raise FootprintRunError("invalid footprint dates: " + "; ".join(errors))
        return tuple(missing)
