"""Read daily cell histories and select the mature cells for spatial processing."""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Self
from uuid import uuid4

import fsspec
import pandas as pd
from pandera.errors import SchemaError, SchemaErrors

from .cells import WaveCellError
from .schemas import WaveCellStateSchema
from .source import DailyFileSource

if TYPE_CHECKING:
    from os import PathLike

    from fsspec.spec import AbstractFileSystem

    from .cells import WaveCells
    from .series import WaveSeries


@dataclass(frozen=True, slots=True)
class WaveCellState:
    """The qualifying-day history of each cell on one observation date.

    Steps:
        1. Retain the supplied fields for this record.
        2. Expose the values to the corresponding processing and validation methods.

    Args:
        frame: Six-column table of coordinates, start/end dates and day counts.
        day: Plain calendar date represented by this table.

    Examples:
        state = WaveCellState.read_parquet("2024-07-01.parquet", date(2024, 7, 1))
        cells = state.to_cells()
    """

    frame: pd.DataFrame
    day: date
    schema: ClassVar[type[WaveCellStateSchema]] = WaveCellStateSchema
    columns: ClassVar[tuple[str, ...]] = (
        "longitude",
        "latitude",
        "start_date",
        "end_date",
        "days",
        "grace_days",
    )

    def __post_init__(self) -> None:
        """Check types and history dates once without changing supplied values.

        Steps:
            1. Require a plain observation date and validate the six-column schema.
            2. Require every history to begin by that date. An ended history must
               end between its start and the day before observation.
            3. Count the available dates from start through observation or end.
               Qualifying days plus the current grace count cannot exceed that
               span. Earlier recovered grace days may make the count smaller.

        Args:
            None.

        Returns:
            None. Retains the original values and row order.

        Raises:
            WaveCellError: If types, coordinates, counts or dates are invalid.

        Examples:
            state = WaveCellState(frame, date(2024, 7, 1))
        """
        if not isinstance(self.day, date) or isinstance(self.day, datetime):
            raise WaveCellError("day must be a plain calendar date")
        try:
            self.schema.validate(self.frame)
        except (SchemaError, SchemaErrors) as error:
            raise WaveCellError(f"invalid wave-cell state: {error}") from error
        observation = pd.Timestamp(self.day)
        starts = self.frame["start_date"]
        ends = self.frame["end_date"]
        ended = ends.notna()
        if (
            starts.gt(observation).any()
            or (ended & (ends.lt(starts) | ends.ge(observation))).any()
        ):
            raise WaveCellError("cell history dates disagree with the observation date")
        last_dates = ends.fillna(observation)
        span = (last_dates - starts).dt.days + 1
        if self.frame["days"].gt(span - self.frame["grace_days"]).any():
            raise WaveCellError("cell counts exceed their available date span")

    @classmethod
    def empty(cls, day: date) -> Self:
        """Create a typed empty state for an explicitly empty date.

        Steps:
            1. Create empty coordinate, date and counter columns with their required dtypes.
            2. Validate the resulting six-column state for the requested date.

        Args:
            day: Observation date with no cell histories.

        Returns:
            Once-validated empty six-column state.

        Examples:
            state = WaveCellState.empty(date(1960, 1, 1))
        """
        return cls(
            pd.DataFrame(
                {
                    "longitude": pd.Series(dtype="float64"),
                    "latitude": pd.Series(dtype="float64"),
                    "start_date": pd.Series(dtype="datetime64[ns]"),
                    "end_date": pd.Series(dtype="datetime64[ns]"),
                    "days": pd.Series(dtype="int64"),
                    "grace_days": pd.Series(dtype="int64"),
                }
            ),
            day,
        )

    @classmethod
    def read_parquet(
        cls,
        path: str | PathLike[str],
        day: date,
        *,
        filesystem: AbstractFileSystem | None = None,
    ) -> Self:
        """Read the six state fields without opening other stored measurements.

        Steps:
            1. Choose the filesystem and read only the six required columns.
            2. Prepare stored dates as timezone-free nanosecond timestamps. A
               wholly absent end-date column contains no dates to interpret.
            3. Validate the resulting state once against the observation date.

        Args:
            path: Local file or URL containing daily cell state.
            day: Observation date supplied by the filename or caller.
            filesystem: Optional configured filesystem for the file.

        Returns:
            Validated daily state with unchanged coordinates and counts.

        Raises:
            FileNotFoundError: If the file is absent.
            WaveCellError: If required columns or stored values are invalid.

        Examples:
            state = WaveCellState.read_parquet("2024-07-01.parquet", date(2024, 7, 1))
        """
        if filesystem is None:
            filesystem, location = fsspec.core.url_to_fs(str(path))
        else:
            location = str(path)
        stream = filesystem.open(location, "rb")
        try:
            with stream:
                frame = pd.read_parquet(
                    stream,
                    columns=list(cls.columns),
                    to_pandas_kwargs={"date_as_object": False},
                )
            for column in ("start_date", "end_date"):
                values = frame[column]
                if isinstance(values.dtype, pd.DatetimeTZDtype):
                    raise WaveCellError(f"{column} must be stored timezone-free dates")
                if (
                    column == "end_date"
                    and values.dtype == object
                    and values.isna().all()
                ):
                    frame[column] = pd.Series(
                        pd.NaT, index=frame.index, dtype="datetime64[ns]"
                    )
                elif not pd.api.types.is_datetime64_dtype(values.dtype):
                    raise WaveCellError(f"{column} must be stored timezone-free dates")
                else:
                    frame[column] = values.astype("datetime64[ns]")
            return cls(frame, day)
        except (ValueError, TypeError, AttributeError) as error:
            raise WaveCellError(
                f"{path}: invalid wave-cell state input: {error}"
            ) from error

    def to_parquet(
        self,
        path: str | PathLike[str],
        *,
        filesystem: AbstractFileSystem | None = None,
    ) -> None:
        """Write the accepted six-column state without another validation pass.

        Steps:
            1. Choose the filesystem and create the destination directory.
            2. Encode the accepted state without its row index.
            3. For a local file, write a temporary sibling and replace the target
               only after the complete write succeeds. For a remote file, upload
               the complete encoded bytes through the supplied filesystem.

        Args:
            path: Local destination or caller-selected URL.
            filesystem: Optional configured filesystem for writing.

        Returns:
            None. Stores coordinates, dates and counts in one Parquet file.

        Raises:
            OSError: If the destination cannot be written.

        Examples:
            state.to_parquet("wave-state/2024-07-01.parquet")
        """
        if filesystem is None:
            filesystem, location = fsspec.core.url_to_fs(str(path))
        else:
            location = str(path)
        parent = location.rpartition("/")[0]
        if parent:
            filesystem.makedirs(parent, exist_ok=True)
        # Encode first so a failed write never removes an accepted previous state.
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

    def to_cells(self, *, closing_radius: int = 3) -> WaveCells:
        """Select mature cells, fill small spatial gaps and build their polygons.

        Steps:
            1. Configure a WaveCellBuilder with this state and closing radius.
            2. Build validated mature cells, filled gaps and polygons.

        Args:
            closing_radius: Circular neighbourhood radius in quarter-degree cells.
                Zero disables gap filling; the default is three.

        Returns:
            Once-validated daily WaveCells ready for the footprint stage.

        Examples:
            cells = state.to_cells(closing_radius=3)
        """
        from .cell_geometry import WaveCellBuilder

        return WaveCellBuilder(self, closing_radius).build()


class WaveCellStateSource(DailyFileSource):
    """Read daily cell histories using the shared field/direction directories.

    Steps:
        1. Configure the caller-selected file location and access options.
        2. Resolve and validate daily files through the source's read methods.

    Args:
        location: Local directory or fsspec URL for the daily files.
        storage_options: Caller-selected filesystem access settings.
        known_empty_dates: Explicit dates permitted to have missing empty files.

    Examples:
        state = WaveCellStateSource("./state").read_state(day, series)
    """

    __slots__ = ()
    kind: ClassVar[str] = "wave-cell-state"
    error_type: ClassVar[type[ValueError]] = WaveCellError

    def read_state(self, day: date, series: WaveSeries) -> WaveCellState:
        """Read one complete date, allowing only explicitly declared missing dates.

        Steps:
            1. Select the requested series and dated state file.
            2. Read and validate its state. A declared missing empty date creates
               typed empty state; any other missing date raises an error.
            3. Reject existing nonempty state on a date declared empty.

        Args:
            day: Plain observation date.
            series: Maximum/minimum and warm/cold selection.

        Returns:
            Validated six-column state for the requested date.

        Raises:
            FileNotFoundError: If an undeclared file is absent.
            WaveCellError: If its state or empty declaration is invalid.

        Examples:
            state = source.read_state(date(2024, 7, 1), series)
        """
        path = self.path_for(day, series)
        try:
            state = WaveCellState.read_parquet(path, day, filesystem=self._filesystem)
        except FileNotFoundError:
            if day not in self.known_empty_dates:
                raise
            return WaveCellState.empty(day)
        if day in self.known_empty_dates and not state.frame.empty:
            raise WaveCellError(f"{day}: declared empty date contains cell histories")
        return state
