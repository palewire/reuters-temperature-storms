"""Complete daily warm and cold scores used by wave tracking."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import ClassVar

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors
from pandera.typing import Series  # noqa: TC002 - Pandera resolves schema annotations.

from .schemas import GridCellSchema

METRICS = ("t2m_min", "t2m_max")
ANOMALY_FIELDS = tuple(
    f"{metric}_{suffix}" for metric in METRICS for suffix in ("p90_score", "p10_score")
)


class AnomalyError(ValueError):
    """Raised when daily temperature or baseline inputs cannot produce anomalies.

    Steps:
        1. Describe a rejected input, configuration or stage result.
        2. Propagate the message through callers without changing data.

    Args:
        *args: Exception message and any additional exception arguments.

    Examples:
        raise AnomalyError("invalid input")
    """


class CompleteMeasurementSchema(GridCellSchema):
    """Finite complete global measurements before any qualification threshold.

    Steps:
        1. Declare the accepted columns and validation rules.
        2. Apply those rules when validate is called, preserving incoming values.

    Args:
        None. Use class-level validation methods rather than constructing records.

    Examples:
        CompleteMeasurementSchema.to_schema()
    """

    @pa.check(r"^t2m_.*$", regex=True)
    @classmethod
    def finite_measurements(cls, values: pd.Series) -> pd.Series:
        """Check each numeric measurement without changing its stored precision.

        Steps:
            1. Evaluate whether each supplied measurement is finite.
            2. Return the checks with the original row index.

        Args:
            values: One measurement column.

        Returns:
            Whether each measurement is finite.

        Examples:
            CompleteMeasurementSchema.finite_measurements(pd.Series([1.0]))
        """
        return pd.Series(np.isfinite(values), index=values.index)

    @pa.dataframe_check
    @classmethod
    def complete_grid(cls, frame: pd.DataFrame) -> bool:
        """Require all unique quarter-degree cells inside the supported bounds.

        Steps:
            1. Use the coordinate checks inherited from the grid schema.
            2. Require the row count of the complete 1440 by 721 global grid.

        Args:
            frame: Measurement table with checked coordinates.

        Returns:
            Whether all 1440 by 721 global cells are present.

        Examples:
            CompleteMeasurementSchema.complete_grid(frame)
        """
        return len(frame) == 1440 * 721


class TemperatureSchema(CompleteMeasurementSchema):
    """Checked measurement columns for temperature tables.

    Steps:
        1. Declare the accepted columns and validation rules.
        2. Apply those rules when validate is called, preserving incoming values.

    Args:
        None. Use class-level validation methods rather than constructing records.

    Examples:
        TemperatureSchema.to_schema()
    """

    date: Series[datetime]
    t2m_min: Series[float]
    t2m_max: Series[float]


class ClimatologySchema(CompleteMeasurementSchema):
    """Checked measurement columns for climatology tables.

    Steps:
        1. Declare the accepted columns and validation rules.
        2. Apply those rules when validate is called, preserving incoming values.

    Args:
        None. Use class-level validation methods rather than constructing records.

    Examples:
        ClimatologySchema.to_schema()
    """

    month: Series[int] = pa.Field(ge=1, le=12)
    day: Series[int] = pa.Field(ge=1, le=31)
    t2m_min: Series[float]
    t2m_max: Series[float]


class SeasonalSchema(CompleteMeasurementSchema):
    """Checked measurement columns for seasonal tables.

    Steps:
        1. Declare the accepted columns and validation rules.
        2. Apply those rules when validate is called, preserving incoming values.

    Args:
        None. Use class-level validation methods rather than constructing records.

    Examples:
        SeasonalSchema.to_schema()
    """

    month: Series[int] = pa.Field(ge=1, le=12)
    day: Series[int] = pa.Field(ge=1, le=31)
    t2m_min: Series[float]
    t2m_min_p10: Series[float]
    t2m_min_p90: Series[float]
    t2m_max: Series[float]
    t2m_max_p10: Series[float]
    t2m_max_p90: Series[float]


class DailyAnomalySchema(CompleteMeasurementSchema):
    """Checked measurement columns for daily anomaly tables.

    Steps:
        1. Declare the accepted columns and validation rules.
        2. Apply those rules when validate is called, preserving incoming values.

    Args:
        None. Use class-level validation methods rather than constructing records.

    Examples:
        DailyAnomalySchema.to_schema()
    """

    date: Series[datetime]
    t2m_min_p90_score: Series[float]
    t2m_min_p10_score: Series[float]
    t2m_max_p90_score: Series[float]
    t2m_max_p10_score: Series[float]


@dataclass(frozen=True, slots=True)
class MeasurementGrid:
    """One date's accepted global measurement table.

    Steps:
        1. Retain the supplied fields for this record.
        2. Expose the values to the corresponding processing and validation methods.

    Args:
        frame: Finite float64 measurements and exact global coordinates.
        day: Plain date represented by the rows or calendar baseline labels.

    Examples:
        daily = DailyTemperatures(frame, date(2024, 7, 1))
    """

    frame: pd.DataFrame
    day: date
    schema: ClassVar[type[CompleteMeasurementSchema]]

    def __post_init__(self) -> None:
        """Require a complete grid and the exact requested calendar labels.

        Steps:
            1. Validate the date, schema, finite values and full global grid.
            2. Require every stored date or baseline calendar label to match.

        Args:
            None.

        Returns:
            None. Keeps the accepted frame unchanged.

        Examples:
            daily = DailyTemperatures(frame, day)
        """
        if not isinstance(self.day, date) or isinstance(self.day, datetime):
            raise AnomalyError("day must be a plain calendar date")
        try:
            self.schema.validate(self.frame)
        except (SchemaError, SchemaErrors) as error:
            raise AnomalyError(f"invalid {type(self).__name__}: {error}") from error
        if "date" in self.frame:
            if not self.frame["date"].eq(pd.Timestamp(self.day)).all():
                raise AnomalyError(f"stored dates differ from {self.day}")
        elif not (
            self.frame["month"].eq(self.day.month) & self.frame["day"].eq(self.day.day)
        ).all():
            raise AnomalyError(f"baseline calendar labels differ from {self.day}")


class DailyTemperatures(MeasurementGrid):
    """Daily minimum and maximum temperatures on the global grid.

    Steps:
        1. Retain the supplied table and requested calendar date.
        2. Validate the table with this measurement schema and require matching calendar labels.

    Args:
        frame: Complete global measurement table with the declared schema.
        day: Plain calendar date represented by its date or baseline calendar labels.

    Examples:
        daily = DailyTemperatures(frame, day)
    """

    schema = TemperatureSchema


class DailyClimatology(MeasurementGrid):
    """Ordinary calendar-day means used for the temperature differences.

    Steps:
        1. Retain the supplied table and requested calendar date.
        2. Validate the table with this measurement schema and require matching calendar labels.

    Args:
        frame: Complete global measurement table with the declared schema.
        day: Plain calendar date represented by its date or baseline calendar labels.

    Examples:
        normal = DailyClimatology(frame, day)
    """

    schema = ClimatologySchema


class SeasonalDistribution(MeasurementGrid):
    """Adjusted means and signed percentile scales used for percentile scores.

    Steps:
        1. Retain the supplied table and requested calendar date.
        2. Validate the table with this measurement schema and require matching calendar labels.

    Args:
        frame: Complete global measurement table with the declared schema.
        day: Plain calendar date represented by its date or baseline calendar labels.

    Examples:
        seasonal = SeasonalDistribution(frame, day)
    """

    schema = SeasonalSchema


class DailyAnomalies(MeasurementGrid):
    """Four float64 warm and cold scores on the global grid.

    Steps:
        1. Retain the supplied table and requested calendar date.
        2. Validate the table with this measurement schema and require matching calendar labels.

    Args:
        frame: Complete global measurement table with the declared schema.
        day: Plain calendar date represented by its date or baseline calendar labels.

    Examples:
        scores = DailyAnomalies(frame, day)
    """

    schema = DailyAnomalySchema


@dataclass(frozen=True, slots=True)
class DailyAnomalyBuilder:
    """Calculate differences against ordinary means using adjusted score scales.

    Steps:
        1. Read accepted daily, ordinary and seasonal measurement grids.
        2. Calculate the four warm and cold scores using seasonal percentile scales.

    Args:
        None.

    Examples:
        anomalies = DailyAnomalyBuilder().build(daily, ordinary, adjusted)
    """

    def build(
        self,
        daily: DailyTemperatures,
        climatology: DailyClimatology,
        seasonal: SeasonalDistribution,
    ) -> DailyAnomalies:
        """Calculate all four unthresholded scores after aligning physical cells.

        Steps:
            1. Require one date and align all tables by longitude and latitude.
            2. Subtract ordinary means from daily temperatures in float64.
            3. Divide these differences by adjusted p90/p10 minus adjusted mean. Cold scales are negative.
            4. Return the complete dated grid without qualification or weighting.

        Args:
            daily: Complete daily minimum and maximum temperatures.
            climatology: Matching ordinary calendar-day means.
            seasonal: Matching adjusted means and percentiles.

        Returns:
            Complete float64 daily anomaly grid, readable by ScoreRun.

        Raises:
            AnomalyError: If dates, coordinates, ordering or scales are invalid.

        Examples:
            output = DailyAnomalyBuilder().build(daily, normal, seasonal)
        """
        if not (daily.day == climatology.day == seasonal.day):
            raise AnomalyError("daily and baseline dates must match")
        keys = ["longitude", "latitude"]
        temperature = daily.frame.set_index(keys)
        normal = climatology.frame.set_index(keys).reindex(temperature.index)
        adjusted = seasonal.frame.set_index(keys).reindex(temperature.index)
        if normal.isna().to_numpy().any() or adjusted.isna().to_numpy().any():
            raise AnomalyError("daily and baseline coordinates must match")
        if not temperature[METRICS[0]].le(temperature[METRICS[1]]).all():
            raise AnomalyError("daily temperatures must be ordered minimum <= maximum")
        if not normal[METRICS[0]].le(normal[METRICS[1]]).all():
            raise AnomalyError("ordinary means must be ordered minimum <= maximum")
        result = pd.DataFrame(index=temperature.index)
        result["date"] = pd.Timestamp(daily.day)
        for metric in METRICS:
            delta = temperature[metric].astype("float64") - normal[metric].astype(
                "float64"
            )
            warm = adjusted[f"{metric}_p90"].astype("float64") - adjusted[
                metric
            ].astype("float64")
            cold = adjusted[f"{metric}_p10"].astype("float64") - adjusted[
                metric
            ].astype("float64")
            if not (warm.gt(0) & cold.lt(0)).all():
                raise AnomalyError(
                    f"{metric} requires positive p90 and negative p10 denominators"
                )
            result[f"{metric}_p90_score"] = delta / warm
            result[f"{metric}_p10_score"] = delta / cold
        return DailyAnomalies(result.reset_index(), daily.day)
