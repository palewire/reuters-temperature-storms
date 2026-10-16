"""Build fixed five-day scores from complete daily score grids."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import ClassVar

import numpy as np
import pandas as pd

from .cell_scores import WaveCellScores
from .cells import WaveCellError


@dataclass(frozen=True, slots=True)
class WeightedScoreBuilder:
    """Combine five daily grids with greater weight on more recent days.

    Steps:
        1. Configure the supplied inputs and processing settings.
        2. Expose methods that validate and process the requested data.

    Args:
        None. The five-day window and daily weight decay of 0.9 are fixed.

    Examples:
        weighted = WeightedScoreBuilder().build((today, yesterday, day2, day3, day4))
    """

    window_days: ClassVar[int] = 5
    decay: ClassVar[float] = 0.9

    def build(self, scores: tuple[WaveCellScores, ...]) -> WaveCellScores:
        """Calculate one dated score per cell without applying a threshold.

        Steps:
            1. Require five accepted complete grids in newest-to-oldest date
               order. Missing days must never become a shorter average.
            2. Match cells by longitude and latitude. Reuse matching row order;
               otherwise reorder scores to the newest grid's coordinates.
            3. Multiply each daily score by 0.9 raised to its age in days. Add
               from oldest to newest using float64, then divide by the sum of
               the five weights. Keep the original input scores unchanged.
            4. Check the finished table once and attach the newest date. The
               later cell-history stage decides whether a score reaches one.

        Args:
            scores: Five once-validated daily score tables, newest first.

        Returns:
            Complete float64 weighted grid in the newest input's row order.

        Raises:
            WaveCellError: If inputs are not five consecutive accepted grids,
                or the calculated score table is invalid.

        Examples:
            weighted = WeightedScoreBuilder().build((today, yesterday, day2, day3, day4))
        """
        # Dates are checked here; accepted input schemas are not run again.
        if (
            not isinstance(scores, tuple)
            or len(scores) != self.window_days
            or any(not isinstance(item, WaveCellScores) for item in scores)
        ):
            raise WaveCellError("weighted scores require five accepted daily grids")
        newest = scores[0]
        if newest.day < date.min + timedelta(days=self.window_days - 1):
            raise WaveCellError("score dates must allow four preceding days")
        if any(
            item.day != newest.day - timedelta(days=age)
            for age, item in enumerate(scores)
        ):
            raise WaveCellError("score dates must be consecutive, newest first")

        # The full-grid input schema guarantees the same unique cell set.
        coordinates = newest.frame[["longitude", "latitude"]]
        positions: pd.MultiIndex | None = None
        weights = np.power(self.decay, np.arange(self.window_days, dtype=float))
        total = np.zeros(len(coordinates), dtype=np.float64)
        for age in reversed(range(self.window_days)):
            frame = scores[age].frame
            values = frame["score"].to_numpy(dtype=np.float64, copy=False)
            if not (
                np.array_equal(frame["longitude"], coordinates["longitude"])
                and np.array_equal(frame["latitude"], coordinates["latitude"])
            ):
                if positions is None:
                    positions = pd.MultiIndex.from_frame(coordinates)
                order = pd.MultiIndex.from_frame(
                    frame[["longitude", "latitude"]]
                ).get_indexer(positions)
                values = values[order]
            total += values * weights[age]

        # Keep division separate from multiplication to use one fixed average.
        frame = coordinates.reset_index(drop=True).copy()
        frame["score"] = total / weights.sum()
        return WaveCellScores(frame, newest.day)
