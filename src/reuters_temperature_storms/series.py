"""The supported temperature and direction choices for wave tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal, Self, cast

Temperature = Literal["tasmax", "tasmin"]
Direction = Literal["warm", "cold"]


@dataclass(frozen=True, slots=True)
class WaveSeries:
    """One independent temperature-grid and direction tracking series.

    Warm events are heat waves until named, then heat storms. Cold events are
    cold waves until named, then cold storms. Naming retains their identity.

    Steps:
        1. Retain the supplied fields for this record.
        2. Expose the values to the corresponding processing and validation methods.

    Args:
        temperature: Daily maximum or minimum temperature field.
        direction: Warm or cold events relative to the baseline.

    Examples:
        series = WaveSeries.parse("tasmax/warm")
    """

    temperature: Temperature
    direction: Direction
    temperatures: ClassVar[tuple[Temperature, ...]] = ("tasmax", "tasmin")
    directions: ClassVar[tuple[Direction, ...]] = ("warm", "cold")

    def validate(self) -> None:
        """Check the series at the existing input boundary.

        Steps:
            1. Check temperature against the supported maximum and minimum fields.
            2. Check direction against the supported warm and cold selections.

        Args:
            None.

        Returns:
            None. Performs no table or geometry validation.

        Raises:
            ValueError: If either field is unsupported.

        Examples:
            WaveSeries("tasmax", "warm").validate()
        """
        if (
            self.temperature not in self.temperatures
            or self.direction not in self.directions
        ):
            raise ValueError("series must use tasmax/tasmin and warm/cold")

    @classmethod
    def from_fields(cls, temperature: object, direction: object) -> Self:
        """Parse saved fields using the same choices as the CLI and schemas.

        Steps:
            1. Require temperature and direction to be strings.
            2. Construct and validate the selected series.

        Args:
            temperature: Candidate temperature field.
            direction: Candidate event direction.

        Returns:
            The matching supported series.

        Raises:
            ValueError: If fields do not name a supported series.

        Examples:
            WaveSeries.from_fields("tasmin", "cold")
        """
        if not isinstance(temperature, str) or not isinstance(direction, str):
            raise ValueError("series fields must be strings")
        series = cls(cast("Temperature", temperature), cast("Direction", direction))
        series.validate()
        return series

    @classmethod
    def parse(cls, value: str) -> Self:
        """Parse the CLI spelling of a series.

        Steps:
            1. Split the input at the slash and require two fields.
            2. Validate the resulting temperature and direction selection.

        Args:
            value: Temperature and direction separated by a slash.

        Returns:
            The matching supported series.

        Raises:
            ValueError: If the spelling does not identify one supported series.

        Examples:
            WaveSeries.parse("tasmax/warm")
        """
        fields = value.split("/")
        if len(fields) != 2:
            raise ValueError("series must be temperature/direction")
        return cls.from_fields(*fields)

    @classmethod
    def choices(cls) -> tuple[str, ...]:
        """Return CLI choices in the existing temperature and direction order.

        Steps:
            1. Visit supported temperatures in their declared order.
            2. Pair each with every supported direction to form CLI choices.

        Args:
            None.

        Returns:
            The four supported temperature/direction strings.

        Examples:
            WaveSeries.choices()[0] == "tasmax/warm"
        """
        return tuple(
            f"{temperature}/{direction}"
            for temperature in cls.temperatures
            for direction in cls.directions
        )
