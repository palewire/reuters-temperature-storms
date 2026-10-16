"""Create reusable daily wave cells from independent cell-state files."""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd

from .cell_state import WaveCellStateSource
from .cells import WaveCells, WaveCellSource
from .daily_run import DailyRunExecutor
from .series import WaveSeries

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


class CellRunError(ValueError):
    """Raised when daily cell preparation cannot complete the requested dates.

    Steps:
        1. Describe a rejected input, configuration or stage result.
        2. Propagate the message through callers without changing data.

    Args:
        *args: Exception message and any additional exception arguments.

    Examples:
        raise CellRunError("invalid input")
    """


@dataclass(frozen=True, slots=True)
class CellResult:
    """Small worker status without transferring cell tables or geometry.

    Steps:
        1. Retain the supplied fields for this record.
        2. Expose the values to the corresponding processing and validation methods.

    Args:
        day: Completed calendar date.
        rows: Number of daily cells, including added gap-fill cells.
        bytes: Stored file size.
        elapsed_seconds: Worker read, calculation and write duration.
        skipped: Whether an accepted existing output was reused.

    Examples:
        result = CellResult(date(2024, 7, 1), 200, 4096, 0.2, False)
    """

    day: date
    rows: int
    bytes: int
    elapsed_seconds: float
    skipped: bool


@dataclass(frozen=True, slots=True)
class CellTask:
    """File-access and calculation settings for one independent date.

    Steps:
        1. Retain the supplied fields for this record.
        2. Expose the values to the corresponding processing and validation methods.

    Args:
        source_location: Root of daily per-cell wave-state files.
        destination_location: Root for daily mature-cell GeoParquets.
        source_options: Caller-selected input filesystem access settings.
        destination_options: Caller-selected output filesystem access settings.
        empty_dates: Explicitly declared source dates allowed to be missing.
        day: Calendar date this worker owns.
        series: Maximum/minimum and warm/cold selection.
        force_update: Whether to replace an existing output.
        closing_radius: Grid-cell radius for filling small spatial breaks.

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
    closing_radius: int

    @staticmethod
    def read_existing(
        destination: WaveCellSource, day: date, series: WaveSeries
    ) -> WaveCells:
        """Read and check an existing cell file without calculating footprints.

        Steps:
            1. Read the seven cell columns and check the table once.
            2. Require each represented observation date to equal the requested
               filename date. An empty table has no conflicting dates.
            3. Return the checked cells without measuring or combining geometry.

        Args:
            destination: Cell-file root with the caller's filesystem access.
            day: Expected calendar date.
            series: Requested maximum/minimum and warm/cold selection.

        Returns:
            Once-validated cells whose rows represent the requested date.

        Raises:
            FileNotFoundError: If the output does not exist.
            ValueError: If the table or represented date is invalid.

        Examples:
            cells = CellTask.read_existing(output, day, series)
        """
        path = destination.path_for(day, series)
        cells = WaveCells.read_parquet(path, filesystem=destination._filesystem)
        if not cells.frame["date"].eq(pd.Timestamp(day)).all():
            raise CellRunError(f"{path}: wave-cell rows must represent {day}")
        return cells

    def run(self) -> CellResult:
        """Read, build and save one date entirely inside its worker.

        Steps:
            1. Open the configured state source and cell destination.
            2. Unless replacement was requested, read and check any existing cell
               output, including its observation date, and reuse accepted files.
            3. Otherwise read this date's state and select mature active cells.
               Fill small spatial breaks and attach exact grid-cell polygons.
            4. Write the cell table, including an explicitly empty table when no
               mature cells are present. Missing state inputs fail unless their
               dates were explicitly declared empty on the source.
            5. Return only the date, row count, byte size and elapsed time.

        Args:
            None.

        Returns:
            Small completed-file status. Large tables stay inside this worker.

        Raises:
            ValueError: If state or an existing output is invalid.
            OSError: If reading or writing fails.

        Examples:
            result = task.run()
        """
        started = time.perf_counter()
        source = WaveCellStateSource(
            self.source_location,
            self.source_options,
            known_empty_dates=self.empty_dates,
        )
        destination = WaveCellSource(
            self.destination_location, self.destination_options
        )
        output_path = destination.path_for(self.day, self.series)
        # Reuse is a cell-schema/date check and never creates wave footprints.
        if not self.force_update and destination._filesystem.exists(output_path):
            existing = self.read_existing(destination, self.day, self.series)
            return CellResult(
                self.day,
                len(existing.frame),
                destination._filesystem.size(output_path),
                time.perf_counter() - started,
                True,
            )
        # Each global date is independent, including the dateline connections.
        state = source.read_state(self.day, self.series)
        cells = state.to_cells(closing_radius=self.closing_radius)
        cells.to_parquet(output_path, filesystem=destination._filesystem)
        return CellResult(
            self.day,
            len(cells.frame),
            destination._filesystem.size(output_path),
            time.perf_counter() - started,
            False,
        )


@dataclass(frozen=True, slots=True)
class CellRun:
    """Prepare independent daily cell files before generating wave footprints.

    Steps:
        1. Retain the supplied fields for this record.
        2. Expose the values to the corresponding processing and validation methods.

    Args:
        source: Daily wave-state source with caller-selected access.
        destination: Root for finished daily wave-cell files.
        start_date: First included calendar date.
        end_date: Last included calendar date.
        series: Maximum/minimum and warm/cold selection.
        max_workers: Worker processes, or available CPUs when omitted.
        force_update: Replace outputs instead of checking and reusing.
        closing_radius: Grid-cell radius for filling small spatial breaks.

    Examples:
        results = CellRun(source, output, start, end, series, 8).run()
    """

    source: WaveCellStateSource
    destination: WaveCellSource
    start_date: date
    end_date: date
    series: WaveSeries
    max_workers: int | None = None
    force_update: bool = False
    closing_radius: int = 3

    def validate(self) -> None:
        """Check the requested dates, workers and file roots before writing.

        Steps:
            1. Require state and cell file sources and ordered plain dates.
            2. Require a supported series, a positive worker count when supplied,
               a boolean replacement choice and a nonnegative integer radius.
            3. Reject the same input and output directory on the same filesystem
               so the stage cannot replace its state inputs.

        Args:
            None.

        Returns:
            None when each request setting is supported.

        Raises:
            CellRunError: If a request setting or output root is unsafe.

        Examples:
            run.validate()
        """
        if not isinstance(self.source, WaveCellStateSource) or not isinstance(
            self.destination, WaveCellSource
        ):
            raise CellRunError(
                "source and destination must be wave-state and wave-cell sources"
            )
        if (
            any(
                not isinstance(value, date) or isinstance(value, datetime)
                for value in (self.start_date, self.end_date)
            )
            or self.start_date > self.end_date
        ):
            raise CellRunError("start_date and end_date must be ordered calendar dates")
        if not isinstance(self.series, WaveSeries):
            raise CellRunError("series must use tasmax/tasmin and warm/cold")
        try:
            self.series.validate()
        except ValueError as error:
            raise CellRunError(str(error)) from error
        if self.max_workers is not None and (
            type(self.max_workers) is not int or self.max_workers < 1
        ):
            raise CellRunError("max_workers must be a positive integer")
        if type(self.force_update) is not bool:
            raise CellRunError("force_update must be boolean")
        if type(self.closing_radius) is not int or self.closing_radius < 0:
            raise CellRunError("closing_radius must be a nonnegative integer")
        if (
            self.source._filesystem.protocol == self.destination._filesystem.protocol
            and self.source._path == self.destination._path
        ):
            raise CellRunError(
                "wave-state input and wave-cell output must be different directories"
            )

    def tasks(self) -> Iterator[CellTask]:
        """Describe one daily file job without reading any table.

        Steps:
            1. Start at the first included date.
            2. Yield its source/output access, series and calculation settings.
            3. Advance one day through the final included date.

        Args:
            None.

        Returns:
            Date-ordered requests holding only small file-access settings.

        Examples:
            tasks = list(run.tasks())
        """
        current = self.start_date
        while current <= self.end_date:
            yield CellTask(
                self.source.location,
                self.destination.location,
                self.source.storage_options,
                self.destination.storage_options,
                self.source.known_empty_dates,
                current,
                self.series,
                self.force_update,
                self.closing_radius,
            )
            current += timedelta(days=1)

    def run(
        self, progress: Callable[[CellResult], None] | None = None
    ) -> tuple[CellResult, ...]:
        """Build and save all dates with independent process workers.

        Steps:
            1. Check the request and cap available or requested workers by dates.
            2. Run each complete global date in a worker, retaining only small
               file results and keeping at most twice the worker count waiting.
            3. Continue other dates after ordinary failures and retain completed
               outputs so rerunning can check and reuse them.
            4. Return results in date order or report the failed dates together.

        Args:
            progress: Optional callback for each finished or reused file.

        Returns:
            Date-ordered results after all requested cell files succeed.

        Raises:
            CellRunError: If a request is invalid or any daily job fails.

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
        return DailyRunExecutor[CellResult](
            self.tasks(),
            workers,
            DailyRunExecutor.configure_arrow,
            "wave-cell",
            CellRunError,
            ProcessPoolExecutor,
        ).run(progress)

    def check(self) -> tuple[date, ...]:
        """Check saved cells and their dates without reading state or footprints.

        Steps:
            1. Check the range and source/destination settings.
            2. Read each existing output and check its cell schema and date.
            3. Collect absent files and report invalid files with their dates.

        Args:
            None.

        Returns:
            Missing calendar dates; accepted empty cell files count as present.

        Raises:
            CellRunError: If any existing cell output is invalid.

        Examples:
            missing = run.check()
        """
        self.validate()
        missing: list[date] = []
        errors: list[str] = []
        for task in self.tasks():
            try:
                CellTask.read_existing(self.destination, task.day, self.series)
            except FileNotFoundError:
                missing.append(task.day)
            except (ValueError, OSError) as error:
                errors.append(f"{task.day}: {error}")
        if errors:
            raise CellRunError("invalid wave-cell dates: " + "; ".join(errors))
        return tuple(missing)
