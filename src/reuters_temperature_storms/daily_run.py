"""Run independent daily file jobs without transferring large tables."""

from __future__ import annotations

import multiprocessing
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from concurrent.futures import Future
    from datetime import date


class DailyResult(Protocol):
    """Small daily result used to order successful file jobs.

    Steps:
        1. Declare the attributes and methods required by the processing interface.
        2. Let concrete implementations supply daily access or processing.

    Args:
        None.

    Examples:
        day = result.day
    """

    @property
    def day(self) -> date:
        """Return the date completed by a worker.

        Steps:
            1. Expose the calendar date represented by this result.
            2. Return it without loading the completed output file.

        Args:
            None.

        Returns:
            Plain calendar date.

        Examples:
            day = result.day
        """
        ...


Result = TypeVar("Result", bound=DailyResult)
Result_co = TypeVar("Result_co", bound=DailyResult, covariant=True)


class DailyTask(Protocol[Result_co]):
    """A daily request whose worker retains all large input and output tables.

    Steps:
        1. Declare the attributes and methods required by the processing interface.
        2. Let concrete implementations supply daily access or processing.

    Args:
        None.

    Examples:
        result = task.run()
    """

    @property
    def day(self) -> date:
        """Return the date owned by this task.

        Steps:
            1. Expose the calendar date assigned to this task.
            2. Return it for ordering results and reporting failures.

        Args:
            None.

        Returns:
            Plain calendar date.

        Examples:
            day = task.day
        """
        ...

    def run(self) -> Result_co:
        """Read, calculate and save this date inside the worker.

        Steps:
            1. Read the task's configured inputs and calculate its daily product.
            2. Save or reuse the output and return its small status record.

        Args:
            None.

        Returns:
            Small completed-file result.

        Examples:
            result = task.run()
        """
        ...


@dataclass(frozen=True, slots=True)
class DailyRunExecutor(Generic[Result]):
    """Shared process loop for independent daily cell and footprint files.

    Steps:
        1. Retain the supplied fields for this record.
        2. Expose the values to the corresponding processing and validation methods.

    Args:
        tasks: Daily requests containing only file access and calculation settings.
        workers: Positive process count already capped by the number of dates.
        configure_worker: Initializer limiting each worker's internal thread pools.
        product: Plain-language product name used in failure messages.
        error_type: Stage error class used for dated failures.
        executor_factory: Process pool constructor.

    Examples:
        results = executor.run(progress=lambda result: print(result.day))
    """

    tasks: Iterable[DailyTask[Result]]
    workers: int
    configure_worker: Callable[[], None]
    product: str
    error_type: type[ValueError]
    executor_factory: Callable[..., ProcessPoolExecutor] = ProcessPoolExecutor

    @staticmethod
    def configure_arrow() -> None:
        """Keep each process's Arrow work within its own worker thread.

        Steps:
            1. Import Arrow inside the worker.
            2. Limit Arrow CPU and file-reading thread counts to one each.

        Args:
            None.

        Returns:
            None. Arrow uses one CPU and one file-reading thread in this worker.

        Examples:
            DailyRunExecutor.configure_arrow()
        """
        import pyarrow as pa

        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)

    def run(
        self, progress: Callable[[Result], None] | None = None
    ) -> tuple[Result, ...]:
        """Finish independent dates and report failures after retaining successes.

        Steps:
            1. With one worker, run each date directly. With several workers,
               start separate processes with at most twice the worker count
               waiting at once.
            2. Collect only small file results and call progress after success.
               Each worker keeps its large tables inside its own process.
            3. Continue other dates after an ordinary date failure. Keep completed
               files available for a later run to check and reuse.
            4. Return results in date order, or report all failed dates together.
               If a process exits unexpectedly, report how to finish missing dates.

        Args:
            progress: Optional callback after each finished or reused file.

        Returns:
            Date-ordered completed-file results when all dates succeed.

        Raises:
            ValueError: Stage-specific error listing failed dates or stopped workers.

        Examples:
            results = executor.run()
        """
        tasks = iter(self.tasks)
        results: list[Result] = []
        errors: list[tuple[date, str]] = []

        def collect(
            task: DailyTask[Result], result: Result | None, error: Exception | None
        ) -> None:
            """Retain a small success result or the failed date and message.

            Steps:
                1. Retain the task's date and message when it failed.
                2. Otherwise retain its result and invoke the optional progress callback.

            Args:
                task: Daily request that just finished.
                result: Completed-file status when successful.
                error: Exception when the daily request failed.

            Returns:
                None. Retains the status and calls progress after success.

            Examples:
                collect(task, result, None)
            """
            if error is not None:
                errors.append((task.day, str(error)))
            elif result is not None:
                results.append(result)
                if progress is not None:
                    progress(result)

        # Direct execution avoids process startup for a single date or worker.
        if self.workers == 1:
            for task in tasks:
                try:
                    result = task.run()
                except Exception as error:  # noqa: BLE001 - collect dated failures and continue independent dates.
                    collect(task, None, error)
                else:
                    collect(task, result, None)
        else:
            # Spawn isolates file access and native libraries between workers.
            try:
                with self.executor_factory(
                    max_workers=self.workers,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=self.configure_worker,
                ) as pool:
                    pending: dict[Future[Result], DailyTask[Result]] = {}
                    for _ in range(self.workers * 2):
                        task = next(tasks, None)
                        if task is None:
                            break
                        pending[pool.submit(task.run)] = task
                    # Replace each completed task to keep the queue small.
                    while pending:
                        done, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            task = pending.pop(future)
                            try:
                                result = future.result()
                            except BrokenProcessPool:
                                raise
                            except Exception as error:  # noqa: BLE001 - collect dated failures while remaining tasks finish.
                                collect(task, None, error)
                            else:
                                collect(task, result, None)
                            next_task = next(tasks, None)
                            if next_task is not None:
                                pending[pool.submit(next_task.run)] = next_task
            except BrokenProcessPool as error:
                raise self.error_type(
                    f"a {self.product} worker process stopped unexpectedly; completed "
                    "files are retained. Rerun to finish missing dates, using fewer "
                    "workers if the machine ran out of memory"
                ) from error
        # An ordinary bad date must not hide which other dates also failed.
        if errors:
            details = "; ".join(f"{day}: {message}" for day, message in sorted(errors))
            raise self.error_type(
                f"{self.product} creation failed for {len(errors)} date(s): {details}"
            )
        return tuple(sorted(results, key=lambda result: result.day))
