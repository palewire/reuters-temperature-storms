"""Read complete daily calculated-score grids for cell-history processing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING, ClassVar, Self

import fsspec
import pandas as pd
import pyarrow.parquet as pq
from pandera.errors import SchemaError, SchemaErrors

from .cells import WaveCellError
from .schemas import WaveCellScoreSchema
from .series import WaveSeries
from .source import DailyFileSource

if TYPE_CHECKING:
    from os import PathLike

    from fsspec.spec import AbstractFileSystem


@dataclass(frozen=True, slots=True)
class WaveCellScores:
    """One calculated warm or cold score at every cell on one calendar date.

    Steps:
        1. Retain the supplied fields for this record.
        2. Expose the values to the corresponding processing and validation methods.

    Args:
        frame: Complete three-column longitude, latitude and score table.
        day: Observation date taken from the daily filename or caller.

    Examples:
        scores = WaveCellScores.read_parquet("2024-07-01.parquet", day, series)
    """

    frame: pd.DataFrame
    day: date
    schema: ClassVar[type[WaveCellScoreSchema]] = WaveCellScoreSchema
    columns: ClassVar[tuple[str, ...]] = ("longitude", "latitude", "score")

    def __post_init__(self) -> None:
        """Check the complete score grid once without changing its values.

        Steps:
            1. Require a plain observation date.
            2. Check finite scores and exact, unique quarter-degree coordinates.
            3. Require all global cells before any threshold is applied, so a
               missing cell cannot be interpreted as failing today's threshold.

        Args:
            None.

        Returns:
            None. Preserves score precision and input row order.

        Raises:
            WaveCellError: If the date, scores or global coverage are invalid.

        Examples:
            scores = WaveCellScores(frame, date(2024, 7, 1))
        """
        if not isinstance(self.day, date) or isinstance(self.day, datetime):
            raise WaveCellError("day must be a plain calendar date")
        try:
            self.schema.validate(self.frame)
        except (SchemaError, SchemaErrors) as error:
            raise WaveCellError(f"invalid wave-cell scores: {error}") from error

    @staticmethod
    def score_column(series: WaveSeries) -> str:
        """Select the stored score field for the requested scientific series.

        Steps:
            1. Validate the requested temperature and direction.
            2. Map warm to p90 and cold to p10, then construct the stored score name.

        Args:
            series: Maximum/minimum and warm/cold selection.

        Returns:
            t2m_max/min with the matching p90 warm or p10 cold score suffix.

        Raises:
            WaveCellError: If the series is unsupported.

        Examples:
            WaveCellScores.score_column(WaveSeries("tasmax", "warm"))
        """
        if not isinstance(series, WaveSeries):
            raise WaveCellError("series must use tasmax/tasmin and warm/cold")
        try:
            series.validate()
        except ValueError as error:
            raise WaveCellError(str(error)) from error
        field = "t2m_max" if series.temperature == "tasmax" else "t2m_min"
        percentile = "p90" if series.direction == "warm" else "p10"
        return f"{field}_{percentile}_score"

    @classmethod
    def read_parquet(
        cls,
        path: str | PathLike[str],
        day: date,
        series: WaveSeries,
        *,
        filesystem: AbstractFileSystem | None = None,
    ) -> Self:
        """Read coordinates and one selected score from a daily calculated file.

        Steps:
            1. Select the requested warm/cold maximum/minimum score field.
            2. Read longitude, latitude, that field and a stored date when present.
               Require stored dates to match the requested date. Other measurements
               remain outside this stage and are not decoded or validated here.
            3. Expand float32 coordinates to exact float64 quarter-degree values,
               rename the selected field to score and check the table once.
               Preserve its native floating precision, including near one.

        Args:
            path: Daily calculated-score Parquet file or URL.
            day: Observation date supplied by its filename or caller.
            series: Requested maximum/minimum and warm/cold selection.
            filesystem: Optional configured filesystem for the file.

        Returns:
            Validated complete three-column score grid.

        Raises:
            FileNotFoundError: If the daily file is absent.
            WaveCellError: If the selected field or score grid is invalid.

        Examples:
            scores = WaveCellScores.read_parquet("2024-07-01.parquet", day, series)
        """
        column = cls.score_column(series)
        if filesystem is None:
            filesystem, location = fsspec.core.url_to_fs(str(path))
        else:
            location = str(path)
        stream = filesystem.open(location, "rb")
        try:
            with stream:
                parquet = pq.ParquetFile(stream)
                columns = ["longitude", "latitude", column]
                if "date" in parquet.schema_arrow.names:
                    columns.append("date")
                frame = parquet.read(columns=columns).to_pandas()
            if "date" in frame:
                dates = pd.to_datetime(frame.pop("date"))
                if not dates.eq(pd.Timestamp(day)).all():
                    raise WaveCellError(f"{path}: stored dates differ from {day}")
            for coordinate in ("longitude", "latitude"):
                if frame[coordinate].dtype == "float32":
                    frame[coordinate] = frame[coordinate].astype("float64")
            return cls(frame.rename(columns={column: "score"}), day)
        except (ValueError, TypeError, AttributeError, KeyError) as error:
            raise WaveCellError(
                f"{path}: invalid wave-cell score input: {error}"
            ) from error

    def to_parquet(
        self,
        path: str | PathLike[str],
        series: WaveSeries,
        *,
        filesystem: AbstractFileSystem | None = None,
    ) -> None:
        """Store the accepted score under its selected scientific field name.

        Steps:
            1. Choose the stored field name and filesystem.
            2. Create the destination directory and write coordinates and scores,
               omitting the pandas index and preserving native score precision.

        Args:
            path: Local destination or caller-selected URL.
            series: Scientific field name used for the stored score.
            filesystem: Optional configured filesystem for writing.

        Returns:
            None. Writes one calculated-score Parquet for this series.

        Raises:
            OSError: If writing the destination fails.

        Examples:
            scores.to_parquet("scores/2024-07-01.parquet", series)
        """
        column = self.score_column(series)
        if filesystem is None:
            filesystem, location = fsspec.core.url_to_fs(str(path))
        else:
            location = str(path)
        parent = location.rpartition("/")[0]
        if parent:
            filesystem.makedirs(parent, exist_ok=True)
        with filesystem.open(location, "wb") as stream:
            self.frame.rename(columns={"score": column}).to_parquet(stream, index=False)


class WaveCellScoreSource(DailyFileSource):
    """Read calculated-score files directly beneath a daily source directory.

    Steps:
        1. Configure the caller-selected file location and access options.
        2. Resolve and validate daily files through the source's read methods.

    Args:
        location: Local directory or fsspec URL for the daily files.
        storage_options: Caller-selected filesystem access settings.
        known_empty_dates: Explicit dates permitted to have missing empty files.

    Examples:
        scores = WaveCellScoreSource("./scores").read_scores(day, series)
    """

    __slots__ = ()
    kind: ClassVar[str] = "wave-cell-scores"
    error_type: ClassVar[type[ValueError]] = WaveCellError

    def path_for(self, day: date, series: WaveSeries) -> str:
        """Choose a flat dated score file after checking date and series.

        Steps:
            1. Require a plain calendar date and validate the score selection.
            2. Append the ISO date and Parquet suffix directly beneath the source root.

        Args:
            day: Plain observation date.
            series: Scientific score selection, checked before reading.

        Returns:
            ROOT/YYYY-MM-DD.parquet; one file may contain all four score fields.

        Raises:
            WaveCellError: If the date or series is unsupported.

        Examples:
            path = source.path_for(date(2024, 7, 1), series)
        """
        if not isinstance(day, date) or isinstance(day, datetime):
            raise WaveCellError("day must be a plain calendar date")
        WaveCellScores.score_column(series)
        return f"{self._path.rstrip('/')}/{day.isoformat()}.parquet"

    def read_scores(self, day: date, series: WaveSeries) -> WaveCellScores:
        """Read and validate the requested score grid, requiring the daily file.

        Steps:
            1. Select the exact daily score path for the requested series.
            2. Read and validate the selected field using the configured filesystem.

        Args:
            day: Plain observation date.
            series: Maximum/minimum and warm/cold score selection.

        Returns:
            Once-validated complete daily score grid.

        Raises:
            FileNotFoundError: If the daily file is absent.
            WaveCellError: If the selected score grid is invalid.

        Examples:
            scores = source.read_scores(date(2024, 7, 1), series)
        """
        return WaveCellScores.read_parquet(
            self.path_for(day, series), day, series, filesystem=self._filesystem
        )
