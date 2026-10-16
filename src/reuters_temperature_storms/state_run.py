"""Advance daily wave-cell histories chronologically from calculated scores."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

from .cell_scores import WaveCellScoreSource
from .cell_state import WaveCellState, WaveCellStateSource
from .series import WaveSeries
from .state_builder import WaveCellStateBuilder

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


class StateRunError(ValueError):
    """Raised when daily cell histories cannot continue the requested sequence.

    Steps:
        1. Describe a rejected input, configuration or stage result.
        2. Propagate the message through callers without changing data.

    Args:
        *args: Exception message and any additional exception arguments.

    Examples:
        raise StateRunError("invalid input")
    """


@dataclass(frozen=True, slots=True)
class StateResult:
    """Small status for one saved or reused cell-history file.

    Steps:
        1. Retain the supplied fields for this record.
        2. Expose the values to the corresponding processing and validation methods.

    Args:
        day: Completed calendar date.
        rows: Number of cell histories stored for that date.
        bytes: Stored file size.
        elapsed_seconds: Read, calculation and write duration.
        skipped: Whether an accepted existing output was reused.

    Examples:
        result = StateResult(date(2024, 7, 1), 100, 4096, 0.2, False)
    """

    day: date
    rows: int
    bytes: int
    elapsed_seconds: float
    skipped: bool


@dataclass(frozen=True, slots=True)
class StateRun:
    """Save one continuous daily sequence, carrying yesterday's state in memory.

    Steps:
        1. Retain the supplied fields for this record.
        2. Expose the values to the corresponding processing and validation methods.

    Args:
        source: Daily calculated-score files with caller-selected access.
        destination: Root for daily cell-state files.
        start_date: First included observation date.
        end_date: Last included observation date.
        series: Maximum/minimum and warm/cold selection.
        seed: Explicitly start the first requested date without earlier history.
        force_update: Rebuild every requested date instead of reusing a prefix.
        seed_lead_in: Count four preceding score dates before the first saved date.

    Examples:
        results = StateRun(source, output, start, end, series, seed=True).run()
    """

    source: WaveCellScoreSource
    destination: WaveCellStateSource
    start_date: date
    end_date: date
    series: WaveSeries
    seed: bool = False
    force_update: bool = False
    seed_lead_in: bool = False

    def validate(self) -> None:
        """Check the chronological request and prevent replacement of score inputs.

        Steps:
            1. Require score and state sources and ordered plain calendar dates.
            2. Require a supported series and explicit boolean seed/rebuild choices.
            3. Reject the same input and output root on the same filesystem.

        Args:
            None.

        Returns:
            None when the request settings are supported.

        Raises:
            StateRunError: If dates, source types or file settings are invalid.

        Examples:
            run.validate()
        """
        if not isinstance(self.source, WaveCellScoreSource) or not isinstance(
            self.destination, WaveCellStateSource
        ):
            raise StateRunError(
                "source and destination must be wave-score and wave-state sources"
            )
        if (
            any(
                not isinstance(value, date) or isinstance(value, datetime)
                for value in (self.start_date, self.end_date)
            )
            or self.start_date > self.end_date
        ):
            raise StateRunError(
                "start_date and end_date must be ordered calendar dates"
            )
        if not isinstance(self.series, WaveSeries):
            raise StateRunError("series must use tasmax/tasmin and warm/cold")
        try:
            self.series.validate()
        except ValueError as error:
            raise StateRunError(str(error)) from error
        if type(self.seed) is not bool or type(self.force_update) is not bool:
            raise StateRunError("seed and force_update must be boolean")
        if type(self.seed_lead_in) is not bool:
            raise StateRunError("seed_lead_in must be boolean")
        if self.seed_lead_in and not self.seed:
            raise StateRunError("seed lead-in requires --seed")
        if self.seed_lead_in and self.start_date < date.min + timedelta(days=4):
            raise StateRunError("seed lead-in requires four preceding calendar dates")
        if (
            self.source._filesystem.protocol == self.destination._filesystem.protocol
            and self.source._path == self.destination._path
        ):
            raise StateRunError(
                "wave-score input and wave-state output must be different directories"
            )

    def dates(self) -> Iterator[date]:
        """Visit the requested dates in chronological order without reading tables.

        Steps:
            1. Start at the requested first date.
            2. Yield each day through the inclusive end date in order.

        Args:
            None.

        Yields:
            Every included plain calendar date from start through end.

        Examples:
            requested = list(run.dates())
        """
        current = self.start_date
        while current <= self.end_date:
            yield current
            current += timedelta(days=1)

    def read_state(self, day: date) -> WaveCellState:
        """Read an actual saved state without substituting an empty missing file.

        Steps:
            1. Select the destination filename for the requested date and series.
            2. Read and validate that exact saved state without substituting an empty table.

        Args:
            day: Exact observation date needed for reuse or continuation.

        Returns:
            Once-validated state read from the corresponding saved file.

        Raises:
            FileNotFoundError: If this exact state file is missing.
            ValueError: If the saved table is invalid for the date.

        Examples:
            previous = run.read_state(start - timedelta(days=1))
        """
        path = self.destination.path_for(day, self.series)
        return WaveCellState.read_parquet(
            path, day, filesystem=self.destination._filesystem
        )

    def existing_dates(self) -> set[date]:
        """List existing requested state dates and reject a stale saved suffix.

        Steps:
            1. List daily state filenames without opening their tables.
            2. Select the requested dates from that list.
            3. When reusing outputs, require a continuous saved prefix. Existing
               dates after a missing date need a full rebuild so an old future
               state cannot be joined to newly calculated history.

        Args:
            None.

        Returns:
            Requested dates with actual existing output files.

        Raises:
            StateRunError: If reuse would cross a missing date into a saved suffix.

        Examples:
            existing = run.existing_dates()
        """
        names = self.destination.list_files(self.series)
        existing = {
            day for day in self.dates() if f"{day.isoformat()}.parquet" in names
        }
        if not self.force_update:
            missing_seen = False
            for day in self.dates():
                if day not in existing:
                    missing_seen = True
                elif missing_seen:
                    raise StateRunError(
                        f"{day}: existing state follows a missing date; use --force-update "
                        "to rebuild the complete requested sequence"
                    )
        return existing

    def initial_state(self) -> WaveCellState | None:
        """Build four preceding days in memory without saving earlier outputs.

        Steps:
            1. Return no predecessor when seed lead-in is disabled.
            2. Otherwise build the four preceding dates in memory, seeding only the earliest.
            3. Return the final predecessor without saving earlier outputs.

        Args:
            None.

        Returns:
            State on the day before start, or None when lead-in is disabled.

        Raises:
            StateRunError: If any preceding score date is missing or invalid.

        Examples:
            previous = run.initial_state()
        """
        if not self.seed_lead_in:
            return None
        previous: WaveCellState | None = None
        for offset in range(4, 0, -1):
            day = self.start_date - timedelta(days=offset)
            try:
                scores = self.source.read_scores(day, self.series)
                previous = WaveCellStateBuilder(
                    scores, predecessor=previous, seed=previous is None
                ).build()
            except (ValueError, OSError, ImportError) as error:
                raise StateRunError(f"seed lead-in {day}: {error}") from error
        return previous

    def run(
        self, progress: Callable[[StateResult], None] | None = None
    ) -> tuple[StateResult, ...]:
        """Read scores and advance one daily history at a time.

        Steps:
            1. Check the request and saved filenames before writing. Reuse only a
               continuous accepted prefix, without opening its score inputs.
            2. Before the first new date, retain the accepted preceding date or
               read the exact day-before-start output. A fresh first date requires
               the explicit seed choice when no preceding state is available.
            3. Read each new date's calculated scores and advance yesterday's state
               using the fixed one-day grace rule. Keep the resulting table in
               memory for the next date rather than opening it again.
            4. Save each completed date and report a small result. Stop at a failed
               date, preserving earlier completed files for a later continuation.

        Args:
            progress: Optional callback after each saved or reused file.

        Returns:
            Date-ordered small results after the full request succeeds.

        Raises:
            StateRunError: If inputs fail or the exact preceding date is unavailable.

        Examples:
            results = run.run(progress=lambda result: print(result.day))
        """
        self.validate()
        existing = self.existing_dates()
        previous = self.initial_state()
        results = []
        for day in self.dates():
            started = time.perf_counter()
            path = self.destination.path_for(day, self.series)
            reuse = not self.force_update and day in existing
            try:
                # Ordinary reuse avoids scores; lead-in verifies the first saved state.
                if reuse:
                    current = self.read_state(day)
                    if self.seed_lead_in and day == self.start_date:
                        expected = WaveCellStateBuilder(
                            self.source.read_scores(day, self.series),
                            predecessor=previous,
                        ).build()
                        coordinates = ["longitude", "latitude"]
                        saved = current.frame.set_index(coordinates).sort_index()
                        rebuilt = expected.frame.set_index(coordinates).sort_index()
                        if not saved.equals(rebuilt):
                            raise StateRunError(
                                "saved first state differs from seed lead-in; "
                                "use --force-update or a separate output directory"
                            )
                else:
                    fresh = (
                        self.seed and not self.seed_lead_in and day == self.start_date
                    )
                    if previous is None and not fresh:
                        preceding_day = day - timedelta(days=1)
                        try:
                            previous = self.read_state(preceding_day)
                        except FileNotFoundError as error:
                            raise StateRunError(
                                f"exact preceding state {preceding_day} is missing; "
                                "supply --seed only to explicitly initialize the first requested date"
                            ) from error
                    # The accepted D-1 table stays in memory throughout the chain.
                    scores = self.source.read_scores(day, self.series)
                    current = WaveCellStateBuilder(
                        scores, predecessor=previous, seed=fresh
                    ).build()
                    current.to_parquet(path, filesystem=self.destination._filesystem)
                previous = current
                result = StateResult(
                    day,
                    len(current.frame),
                    self.destination._filesystem.size(path),
                    time.perf_counter() - started,
                    reuse,
                )
            except (ValueError, OSError, ImportError) as error:
                raise StateRunError(f"{day}: {error}") from error
            results.append(result)
            if progress is not None:
                progress(result)
        return tuple(results)

    def check(self) -> tuple[date, ...]:
        """Check saved state tables and date coverage without opening score inputs.

        Steps:
            1. Check the request settings and list actual saved filenames.
            2. Read and validate each existing state for its expected date.
            3. Return missing dates or report invalid saved tables together.

        Args:
            None.

        Returns:
            Missing observation dates. Existing files must satisfy the state schema.

        Raises:
            StateRunError: If a request or existing saved state is invalid.

        Examples:
            missing = run.check()
        """
        self.validate()
        names = self.destination.list_files(self.series)
        missing = []
        errors = []
        for day in self.dates():
            if f"{day.isoformat()}.parquet" not in names:
                missing.append(day)
                continue
            try:
                self.read_state(day)
            except (ValueError, OSError) as error:
                errors.append(f"{day}: {error}")
        if errors:
            raise StateRunError("invalid wave-state dates: " + "; ".join(errors))
        return tuple(missing)
