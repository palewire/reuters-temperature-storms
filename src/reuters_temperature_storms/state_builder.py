"""Advance cell histories from complete calculated-score grids, one day at a time."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import cast

import numpy as np
import pandas as pd

from .cell_scores import WaveCellScores
from .cell_state import WaveCellState
from .cells import WaveCellError


@dataclass(frozen=True, slots=True)
class WaveCellStateBuilder:
    """Start or advance cell histories using the inclusive score threshold of one.

    Steps:
        1. Retain the supplied fields for this record.
        2. Expose the values to the corresponding processing and validation methods.

    Args:
        scores: Accepted complete grid for the new observation date.
        predecessor: Accepted state from exactly one calendar day earlier.
        seed: Explicitly start a new history without a predecessor.

    Examples:
        state = WaveCellStateBuilder(scores, yesterday).build()
    """

    scores: WaveCellScores
    predecessor: WaveCellState | None = None
    seed: bool = False

    def __post_init__(self) -> None:
        """Require an explicit start or the exact previous day's accepted state.

        Steps:
            1. Require accepted scores and a boolean start choice.
            2. Starting a history excludes a predecessor. Continuing a history
               requires one, with no missing dates between it and today's scores.
            3. Keep already validated input tables unchanged without rechecking
               their full schemas here.

        Args:
            None.

        Returns:
            None. Retains the accepted input models.

        Raises:
            WaveCellError: If start/predecessor choices or dates disagree.

        Examples:
            builder = WaveCellStateBuilder(scores, seed=True)
        """
        if not isinstance(self.scores, WaveCellScores):
            raise WaveCellError("scores must be a validated WaveCellScores table")
        if type(self.seed) is not bool:
            raise WaveCellError("seed must be boolean")
        if self.seed:
            if self.predecessor is not None:
                raise WaveCellError("a new cell history cannot have a predecessor")
        elif not isinstance(self.predecessor, WaveCellState):
            raise WaveCellError("cell history requires the exact previous day's state")
        elif self.predecessor.day != self.scores.day - timedelta(days=1):
            raise WaveCellError(
                "cell history predecessor must represent exactly the previous day"
            )

    @staticmethod
    def cell_positions(frame: pd.DataFrame) -> np.ndarray:
        """Map accepted coordinates to small integer array positions.

        Steps:
            1. Convert exact quarter-degree latitude and longitude to row/column.
            2. Combine them into one position in the 721-by-1440 global grid.
               These positions are temporary calculation addresses, not stored IDs.

        Args:
            frame: Accepted score or state coordinates on the global grid.

        Returns:
            One integer position per supplied coordinate, in supplied row order.

        Examples:
            positions = WaveCellStateBuilder.cell_positions(scores.frame)
        """
        rows = ((frame["latitude"].to_numpy() + 90) * 4).astype(np.intp)
        columns = ((frame["longitude"].to_numpy() + 179.75) * 4).astype(np.intp)
        return rows * 1440 + columns

    def new_cells(self, coordinates: pd.DataFrame) -> pd.DataFrame:
        """Create day-one histories for qualifying cells without an active history.

        Steps:
            1. Copy the selected coordinates and set today's date as their start.
            2. Leave end dates absent, set qualifying days to one and grace to zero.
            3. Give all columns explicit types even when no cells qualify.

        Args:
            coordinates: Selected longitude and latitude rows from today's scores.

        Returns:
            Six-column day-one state rows, preserving coordinate order.

        Examples:
            rows = builder.new_cells(scores.frame.loc[:, ["longitude", "latitude"]])
        """
        frame = coordinates.copy().reset_index(drop=True)
        frame["start_date"] = pd.Series(
            pd.Timestamp(self.scores.day), index=frame.index, dtype="datetime64[ns]"
        )
        frame["end_date"] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
        frame["days"] = pd.Series(1, index=frame.index, dtype="int64")
        frame["grace_days"] = pd.Series(0, index=frame.index, dtype="int64")
        return cast("pd.DataFrame", frame.loc[:, list(WaveCellState.columns)])

    def build(self) -> WaveCellState:
        """Apply continuing, grace, ending and new cell histories for one day.

        Steps:
            1. Select scores at least one. A cold score uses the same positive
               threshold because its upstream calculation already selects cold.
            2. For an explicit start, create day-one rows and finish. Otherwise
               discard histories that were already ended yesterday.
            3. Use small boolean arrays to compare today's qualifying positions
               with yesterday's active cells, avoiding per-cell Python loops.
            4. Continuing and recovering cells gain one qualifying day and clear
               grace. A first missed day keeps the count and sets grace to one.
            5. A second missed day ends the history at its last qualifying date,
               two dates before today, and clears its current grace count.
            6. Start new qualifying cells, preserve the continuing/grace/ending/new
               group order, and validate the six-column result once.

        Args:
            None.

        Returns:
            Accepted state ready to write or use as tomorrow's predecessor.

        Examples:
            state = builder.build()
        """
        frame = self.scores.frame
        qualified = frame["score"].ge(1)
        coordinates = frame.loc[qualified, ["longitude", "latitude"]]
        if self.seed:
            return WaveCellState(self.new_cells(coordinates), self.scores.day)
        assert self.predecessor is not None
        previous = self.predecessor.frame
        active = previous.loc[previous["end_date"].isna()].copy()

        # Array positions allow all cells to be compared in one batch.
        today_positions = self.cell_positions(coordinates)
        previous_positions = self.cell_positions(active)
        today = np.zeros(721 * 1440, dtype=bool)
        yesterday = np.zeros_like(today)
        today[today_positions] = True
        yesterday[previous_positions] = True
        qualifies_again = today[previous_positions]

        # Qualifying again also recovers a cell from its allowed grace day.
        continuing = active.loc[qualifies_again].copy()
        continuing["days"] = continuing["days"] + 1
        continuing["grace_days"] = 0
        absent = active.loc[~qualifies_again].copy()
        absent["grace_days"] = absent["grace_days"] + 1
        within_grace = absent["grace_days"].le(1)
        grace = absent.loc[within_grace].copy()
        ending = absent.loc[~within_grace].copy()
        ending["end_date"] = pd.Timestamp(self.scores.day) - pd.to_timedelta(
            ending["grace_days"], unit="D"
        )
        ending["grace_days"] = 0

        # Ended rows from yesterday are excluded, so qualifying again starts fresh.
        new = self.new_cells(coordinates.loc[~yesterday[today_positions]])
        parts = [part for part in (continuing, grace, ending, new) if not part.empty]
        if not parts:
            return WaveCellState(self.new_cells(coordinates), self.scores.day)
        result = pd.concat(parts, ignore_index=True).loc[:, list(WaveCellState.columns)]
        return WaveCellState(result, self.scores.day)
