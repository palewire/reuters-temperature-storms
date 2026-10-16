"""Build daily weighted-score files in independent blocks of calendar dates."""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

from .cell_scores import WaveCellScores, WaveCellScoreSource
from .daily_run import DailyRunExecutor
from .series import WaveSeries
from .weighted_scores import WeightedScoreBuilder

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


class ScoreRunError(ValueError):
    """Raised when weighted-score preparation cannot complete requested dates.

    Steps:
        1. Describe a rejected input, configuration or stage result.
        2. Propagate the message through callers without changing data.

    Args:
        *args: Exception message and any additional exception arguments.

    Examples:
        raise ScoreRunError("invalid input")
    """


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """Small completed-file status without transferring global score tables.

    Steps:
        1. Retain the supplied fields for this record.
        2. Expose the values to the corresponding processing and validation methods.

    Args:
        day: Completed calendar date.
        rows: Number of global grid cells in the output.
        bytes: Stored file size.
        elapsed_seconds: Worker read, calculation and write duration for this date.
        skipped: Whether an accepted existing output was reused.

    Examples:
        result = ScoreResult(date(2024, 7, 1), 1038240, 4096, 0.2, False)
    """

    day: date
    rows: int
    bytes: int
    elapsed_seconds: float
    skipped: bool


@dataclass(frozen=True, slots=True)
class ScoreBlockResult:
    """Small results and dated failures from one independent date block.

    Steps:
        1. Retain the supplied fields for this record.
        2. Expose the values to the corresponding processing and validation methods.

    Args:
        day: First calendar date owned by this block.
        results: Finished or accepted output files.
        errors: Failed output dates with their error messages.

    Examples:
        block = ScoreBlockResult(day, (result,), ())
    """

    day: date
    results: tuple[ScoreResult, ...]
    errors: tuple[tuple[date, str], ...]


@dataclass(frozen=True, slots=True)
class ScoreBlockTask:
    """File access and date settings for one independent weighted-score block.

    Steps:
        1. Retain the supplied fields for this record.
        2. Expose the values to the corresponding processing and validation methods.

    Args:
        source_location: Root of complete daily calculated-score Parquets.
        destination_location: Separate root for daily weighted-score Parquets.
        source_options: Caller-selected input filesystem access settings.
        destination_options: Caller-selected output filesystem access settings.
        day: First included output date.
        end_date: Last included output date.
        series: Maximum/minimum and warm/cold score selection.
        force_update: Whether to replace existing outputs.

    Examples:
        result = task.run()
    """

    source_location: str
    destination_location: str
    source_options: dict[str, object]
    destination_options: dict[str, object]
    day: date
    end_date: date
    series: WaveSeries
    force_update: bool

    def run(self) -> ScoreBlockResult:
        """Build consecutive outputs while retaining only five daily inputs.

        Steps:
            1. Open this block's source and separate destination inside the worker.
            2. Check and reuse an existing output unless replacement was requested.
               Reuse requires no daily input reads.
            3. For a new output, read its five complete days, including four days
               before the block starts. Check each input once and arrange its
               coordinates once so overlapping windows reuse the same table.
            4. Calculate the weighted score and save it under the selected score
               field name. Keep only inputs needed by the next five-day window.
            5. Retain finished files after a failed date and continue other dates.
               Return small statuses and dated failures without transferring tables.

        Args:
            None.

        Returns:
            Completed-file statuses and failures for this block.

        Examples:
            block_result = task.run()
        """
        source = WaveCellScoreSource(self.source_location, self.source_options)
        destination = WaveCellScoreSource(
            self.destination_location, self.destination_options
        )
        builder = WeightedScoreBuilder()
        cache: dict[date, WaveCellScores] = {}
        input_errors: dict[date, str] = {}
        results: list[ScoreResult] = []
        errors: list[tuple[date, str]] = []
        current = self.day
        while current <= self.end_date:
            started = time.perf_counter()
            output_path = destination.path_for(current, self.series)
            # A complete saved output is independent of today's source availability.
            try:
                skipped = not self.force_update and destination._filesystem.exists(
                    output_path
                )
                if skipped:
                    scores = destination.read_scores(current, self.series)
                else:
                    days = tuple(current - timedelta(days=age) for age in range(5))
                    for input_day in days:
                        if input_day in input_errors:
                            raise ScoreRunError(input_errors[input_day])
                        if input_day not in cache:
                            try:
                                accepted = source.read_scores(input_day, self.series)
                            except (ValueError, OSError) as error:
                                message = f"daily score input {input_day}: {error}"
                                input_errors[input_day] = message
                                raise ScoreRunError(message) from error
                            # These freshly read frames belong only to this worker.
                            # Arrange each once before overlapping windows reuse it.
                            accepted.frame.sort_values(
                                ["longitude", "latitude"], inplace=True
                            )
                            accepted.frame.reset_index(drop=True, inplace=True)
                            cache[input_day] = accepted
                    scores = builder.build(
                        tuple(cache[input_day] for input_day in days)
                    )
                    scores.to_parquet(
                        output_path, self.series, filesystem=destination._filesystem
                    )
                results.append(
                    ScoreResult(
                        current,
                        len(scores.frame),
                        destination._filesystem.size(output_path),
                        time.perf_counter() - started,
                        skipped,
                    )
                )
            except (ValueError, OSError) as error:
                errors.append((current, str(error)))
            # Tomorrow needs today's inputs through D-3; discard D-4 now.
            earliest_next = current - timedelta(days=3)
            cache = {
                day: scores for day, scores in cache.items() if day >= earliest_next
            }
            input_errors = {
                day: message
                for day, message in input_errors.items()
                if day >= earliest_next
            }
            if current == self.end_date:
                break
            current += timedelta(days=1)
        return ScoreBlockResult(self.day, tuple(results), tuple(errors))


@dataclass(frozen=True, slots=True)
class ScoreRun:
    """Prepare five-day weighted scores before building daily wave-cell state.

    Steps:
        1. Retain the supplied fields for this record.
        2. Expose the values to the corresponding processing and validation methods.

    Args:
        source: Complete daily calculated-score source with caller-selected access.
        destination: Separate root for finished daily weighted-score files.
        start_date: First included output date; four preceding inputs are required.
        end_date: Last included output date.
        series: Maximum/minimum and warm/cold selection.
        max_workers: Worker processes, or available CPUs when omitted.
        force_update: Replace outputs instead of checking and reusing them.
        block_days: Consecutive output dates per worker job.

    Examples:
        results = ScoreRun(source, output, start, end, series, 8).run()
    """

    source: WaveCellScoreSource
    destination: WaveCellScoreSource
    start_date: date
    end_date: date
    series: WaveSeries
    max_workers: int | None = None
    force_update: bool = False
    block_days: int = 30

    def validate(self) -> None:
        """Check the requested dates, workers and separate roots before writing.

        Steps:
            1. Require score sources and ordered plain dates with four prior days.
            2. Require a supported series, positive worker and block counts, and
               a boolean replacement choice.
            3. Reject the same input and output root on the same filesystem so
               weighted outputs cannot replace their daily inputs.

        Args:
            None.

        Returns:
            None when the request settings are supported.

        Raises:
            ScoreRunError: If a request setting or output root is unsafe.

        Examples:
            run.validate()
        """
        if not isinstance(self.source, WaveCellScoreSource) or not isinstance(
            self.destination, WaveCellScoreSource
        ):
            raise ScoreRunError(
                "source and destination must be wave-cell score sources"
            )
        if (
            any(
                not isinstance(value, date) or isinstance(value, datetime)
                for value in (self.start_date, self.end_date)
            )
            or self.start_date > self.end_date
        ):
            raise ScoreRunError(
                "start_date and end_date must be ordered calendar dates"
            )
        if self.start_date < date.min + timedelta(days=4):
            raise ScoreRunError("start_date must allow four preceding input dates")
        if not isinstance(self.series, WaveSeries):
            raise ScoreRunError("series must use tasmax/tasmin and warm/cold")
        try:
            self.series.validate()
        except ValueError as error:
            raise ScoreRunError(str(error)) from error
        if self.max_workers is not None and (
            type(self.max_workers) is not int or self.max_workers < 1
        ):
            raise ScoreRunError("max_workers must be a positive integer")
        if type(self.block_days) is not int or self.block_days < 1:
            raise ScoreRunError("block_days must be a positive integer")
        if type(self.force_update) is not bool:
            raise ScoreRunError("force_update must be boolean")
        if (
            self.source._filesystem.protocol == self.destination._filesystem.protocol
            and self.source._path == self.destination._path
        ):
            raise ScoreRunError(
                "daily-score input and weighted-score output must be different directories"
            )

    def tasks(self) -> Iterator[ScoreBlockTask]:
        """Describe independent date blocks without reading score tables.

        Steps:
            1. Start at the first included output date.
            2. Yield its block dates, filesystem access and calculation settings.
            3. Start the next block after the current block's final output date.

        Args:
            None.

        Returns:
            Date-ordered block requests holding only small settings.

        Examples:
            tasks = list(run.tasks())
        """
        current = self.start_date
        while current <= self.end_date:
            remaining = (self.end_date - current).days
            end = current + timedelta(days=min(self.block_days - 1, remaining))
            yield ScoreBlockTask(
                self.source.location,
                self.destination.location,
                self.source.storage_options,
                self.destination.storage_options,
                current,
                end,
                self.series,
                self.force_update,
            )
            if end == self.end_date:
                break
            current = end + timedelta(days=1)

    def run(
        self, progress: Callable[[ScoreResult], None] | None = None
    ) -> tuple[ScoreResult, ...]:
        """Finish weighted-score blocks in independent process workers.

        Steps:
            1. Check the request and cap available or requested workers by blocks.
            2. Run each block inside one worker, retaining its five input tables
               there and returning only small statuses and error messages.
            3. Report finished files when their block returns, continuing other
               blocks after ordinary date failures and retaining saved outputs.
            4. Return file results in date order or report all failed dates.

        Args:
            progress: Optional callback for each finished or reused file.

        Returns:
            Date-ordered results after all requested score files succeed.

        Raises:
            ScoreRunError: If the request is invalid or a score date fails.

        Examples:
            results = run.run(progress=lambda result: print(result.day))
        """
        self.validate()
        count = (self.end_date - self.start_date).days + 1
        block_count = (count + self.block_days - 1) // self.block_days
        cpu_count = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else os.cpu_count() or 1
        )
        workers = min(self.max_workers or cpu_count, block_count)

        def report(block: ScoreBlockResult) -> None:
            """Forward each finished file after its worker block completes.

            Steps:
                1. Skip reporting when no progress callback was supplied.
                2. Otherwise pass each completed file result to the callback in block order.

            Args:
                block: Small completed-file statuses from one block.

            Returns:
                None. Calls the user's progress callback for successful files.

            Examples:
                report(block_result)
            """
            if progress is not None:
                for result in block.results:
                    progress(result)

        blocks = DailyRunExecutor[ScoreBlockResult](
            self.tasks(),
            workers,
            DailyRunExecutor.configure_arrow,
            "weighted-score",
            ScoreRunError,
            ProcessPoolExecutor,
        ).run(report)
        errors = sorted(error for block in blocks for error in block.errors)
        if errors:
            details = "; ".join(f"{day}: {message}" for day, message in errors)
            raise ScoreRunError(
                f"weighted-score creation failed for {len(errors)} date(s): {details}"
            )
        return tuple(result for block in blocks for result in block.results)

    def check(self) -> tuple[date, ...]:
        """Check saved weighted scores without reading any daily input files.

        Steps:
            1. Check the range and source/destination settings.
            2. Read and validate each requested output once.
            3. Collect absent files and report invalid files with their dates.

        Args:
            None.

        Returns:
            Missing calendar dates; complete accepted score grids count as present.

        Raises:
            ScoreRunError: If an existing weighted-score output is invalid.

        Examples:
            missing = run.check()
        """
        self.validate()
        missing: list[date] = []
        errors: list[str] = []
        for offset in range((self.end_date - self.start_date).days + 1):
            day = self.start_date + timedelta(days=offset)
            try:
                self.destination.read_scores(day, self.series)
            except FileNotFoundError:
                missing.append(day)
            except (ValueError, OSError) as error:
                errors.append(f"{day}: {error}")
        if errors:
            raise ScoreRunError("invalid weighted-score dates: " + "; ".join(errors))
        return tuple(missing)
