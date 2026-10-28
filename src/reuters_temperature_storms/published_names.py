"""Keep issued Reuters storm names and order future names by birth facts."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from importlib.resources import files
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from .identity import CellIdentity
from .naming import (
    DEFAULT_NAMES_SEED,
    StormNamingError,
    _NameSequence,
    load_name_library,
)
from .series import WaveSeries

if TYPE_CHECKING:
    from collections.abc import Mapping

PUBLISHED_NAMES_THROUGH = date(2026, 9, 30)
_PUBLISHED_SERIES = WaveSeries("tasmax", "warm")
_CSV_COLUMNS = ("temperature", "direction", "birth_date", "cell_id", "wave_id", "name")


@dataclass(frozen=True, slots=True)
class PublishedStormName:
    """One issued name tied to a verified storm's original occupied cell.

    Args:
        series: Maximum-temperature warm events covered by the published table.
        birth_date: First day of the physical storm's history.
        cell_id: First complete occupied cell on its birth day.
        wave_id: Full permanent ID calculated from those birth facts.
        name: Exact issued name, including its birth-year suffix.

    Examples:
        ``assignment.name`` returns the issued label without changing it.
    """

    series: WaveSeries
    birth_date: date
    cell_id: int
    wave_id: str
    name: str

    def __post_init__(self) -> None:
        """Reject an assignment that cannot identify the same physical birth.

        Args:
            None.

        Returns:
            None. The issued text and birth facts remain unchanged.

        Raises:
            StormNamingError: If a series, birth fact, ID or name is invalid.

        Examples:
            A key for a different cell is rejected before it can name a storm.
        """
        if self.series != _PUBLISHED_SERIES or type(self.birth_date) is not date:
            raise StormNamingError(
                "published names require tasmax/warm and a birth date"
            )
        try:
            identity = CellIdentity(np.datetime64(self.birth_date, "D"), self.cell_id)
            expected = identity.wave_id
        except (TypeError, ValueError) as error:
            raise StormNamingError(
                "published name has invalid birth-cell facts"
            ) from error
        if self.wave_id != expected:
            raise StormNamingError(
                "published name key does not match its birth date and cell"
            )
        suffix = f" {self.birth_date.year}"
        if (
            not isinstance(self.name, str)
            or not self.name.endswith(suffix)
            or not self.name.removesuffix(suffix).strip()
            or self.name != self.name.strip()
        ):
            raise StormNamingError(
                "published name must retain its exact name and birth year"
            )


@dataclass(frozen=True, slots=True)
class PublishedStormNames:
    """The complete issued labels through one published observation date.

    This table supplies names only. It contains no weather, anomaly, footprint
    or predecessor data and cannot start a scientific tracking calculation.

    Args:
        assignments: Verified issued labels and their new birth identities.
        through_date: Last observation date covered by the published labels.

    Examples:
        names = load_published_storm_names()
        assignment = names.lookup(series, wave_id)
    """

    assignments: tuple[PublishedStormName, ...]
    through_date: date = PUBLISHED_NAMES_THROUGH
    _by_wave_id: Mapping[str, PublishedStormName] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Require one unambiguous issued label for each physical storm.

        Args:
            None.

        Returns:
            None. Creates a read-only lookup for the validated assignments.

        Raises:
            StormNamingError: If dates, assignments, names or keys conflict.

        Examples:
            Two rows claiming the same birth key stop package loading.
        """
        if type(self.through_date) is not date or not isinstance(
            self.assignments, tuple
        ):
            raise StormNamingError(
                "published names require a cutoff date and a tuple of assignments"
            )
        indexed: dict[str, PublishedStormName] = {}
        issued: set[str] = set()
        library_names = {entry.name for entry in load_name_library().entries}
        for assignment in self.assignments:
            if not isinstance(assignment, PublishedStormName):
                raise StormNamingError(
                    "published name rows must be PublishedStormName records"
                )
            if assignment.birth_date > self.through_date:
                raise StormNamingError(
                    "published name birth is later than the published cutoff"
                )
            if assignment.wave_id in indexed or assignment.name in issued:
                raise StormNamingError(
                    "published names contain duplicate keys or issued labels"
                )
            base_name = assignment.name.rsplit(" ", 1)[0]
            if base_name not in library_names:
                raise StormNamingError(
                    "published name is absent from the Reuters name library"
                )
            indexed[assignment.wave_id] = assignment
            issued.add(assignment.name)
        object.__setattr__(self, "_by_wave_id", MappingProxyType(indexed))

    def lookup(self, series: WaveSeries, wave_id: str) -> PublishedStormName | None:
        """Look up the exact issued label within the published storm series.

        Args:
            series: Temperature and direction whose history is being tracked.
            wave_id: Full permanent storm identity to find.

        Returns:
            The issued assignment, or None for a different series or absent key.

        Examples:
            ``names.lookup(series, wave_id).name`` supplies an issued label.
        """
        series.validate()
        if series != _PUBLISHED_SERIES:
            return None
        return self._by_wave_id.get(wave_id)


@lru_cache(maxsize=1)
def load_published_storm_names() -> PublishedStormNames:
    """Load issued labels from the installed RTS package once.

    Args:
        None.

    Returns:
        Validated immutable assignments with a read-only name lookup.

    Raises:
        StormNamingError: If packaged rows are missing, malformed or ambiguous.

    Examples:
        names = load_published_storm_names()
        assert len(names.assignments) == 2527
    """
    resource = files("reuters_temperature_storms").joinpath(
        "data", "published_storm_names.csv"
    )
    assignments: list[PublishedStormName] = []
    try:
        with resource.open("r", encoding="utf-8") as stream:
            reader = csv.DictReader(stream, strict=True)
            if tuple(reader.fieldnames or ()) != _CSV_COLUMNS:
                raise StormNamingError(
                    "packaged published names have unexpected CSV columns"
                )
            for row in reader:
                if set(row) != set(_CSV_COLUMNS) or any(
                    value is None for value in row.values()
                ):
                    raise StormNamingError(
                        "packaged published names contain an incomplete CSV row"
                    )
                birth_date = date.fromisoformat(row["birth_date"])
                cell_id = int(row["cell_id"])
                if (
                    birth_date.isoformat() != row["birth_date"]
                    or str(cell_id) != row["cell_id"]
                ):
                    raise StormNamingError(
                        "packaged birth dates and cells must use their exact spellings"
                    )
                assignments.append(
                    PublishedStormName(
                        series=WaveSeries.from_fields(
                            row["temperature"], row["direction"]
                        ),
                        birth_date=birth_date,
                        cell_id=cell_id,
                        wave_id=row["wave_id"],
                        name=row["name"],
                    )
                )
    except (OSError, UnicodeError, csv.Error, TypeError, ValueError) as error:
        raise StormNamingError(
            f"cannot load packaged published storm names: {error}"
        ) from error
    if len(assignments) != 2527:
        raise StormNamingError(
            "packaged published names must include all 2,527 issued storms"
        )
    return PublishedStormNames(tuple(assignments))


class StormNames:
    """Preserve issued names and select future names using the Reuters sequence.

    Future qualifying storms are visited in birth-date and original-cell order.
    This separates name priority from the spelling or encoding of a storm ID.

    Args:
        published: Issued labels, or None to use the installed complete table.

    Examples:
        names = StormNames()
    """

    __slots__ = ("_published", "_sequence")

    def __init__(self, published: PublishedStormNames | None = None) -> None:
        """Prepare the shared issued-name lookup and future name sequence.

        Args:
            published: Validated issued labels, or None for packaged assignments.

        Returns:
            None.

        Examples:
            Both RTS and its Climate Monitor caller use ``StormNames()``.
        """
        self._published = (
            load_published_storm_names() if published is None else published
        )
        self._sequence = _NameSequence(load_name_library(), DEFAULT_NAMES_SEED)

    def assign(
        self,
        *,
        series: WaveSeries,
        wave_id: str,
        birth_date: date,
        cell_id: int,
        naming_date: date,
        ordinal: int,
    ) -> str:
        """Return an issued label or the next future label at first qualification.

        Args:
            series: Tracked temperature and direction.
            wave_id: Permanent identity of the qualifying physical storm.
            birth_date: Retained first day of that storm's history.
            cell_id: Retained first occupied cell from its birth footprint.
            naming_date: Day on which its qualified area first earns a name.
            ordinal: Next unused position in this series' Reuters name sequence.

        Returns:
            Exact issued label, or the future given name plus the birth year.

        Raises:
            StormNamingError: If birth facts disagree or a historical name is missing.

        Examples:
            A historical eligible storm cannot silently receive a future label.
        """
        if (
            type(birth_date) is not date
            or type(naming_date) is not date
            or naming_date < birth_date
        ):
            raise StormNamingError("naming date must follow a valid birth date")
        if type(ordinal) is not int or ordinal < 0:
            raise StormNamingError("name ordinal must be a nonnegative integer")
        try:
            identity = CellIdentity(np.datetime64(birth_date, "D"), cell_id)
            expected = identity.wave_id
        except (TypeError, ValueError) as error:
            raise StormNamingError(
                "storm naming requires valid birth-cell facts"
            ) from error
        if expected != wave_id:
            raise StormNamingError(
                "storm naming key disagrees with the retained birth cell"
            )
        # Names already issued in the story and published table remain labels
        # of those physical storms, even when replay or ID encoding changes.
        assignment = self._published.lookup(series, wave_id)
        if assignment is not None:
            return assignment.name
        # A missing historical match signals different tracking or bad data.
        # Stop here so that an unexpected storm cannot shift the name sequence.
        if series == _PUBLISHED_SERIES and naming_date <= self._published.through_date:
            raise StormNamingError(
                f"missing published name for {wave_id} on {naming_date}; "
                "verify the physical history before continuing"
            )
        # Only future qualifications use the next sequence position. An older
        # unnamed wave may first qualify after the published observation date.
        return f"{self._sequence.name_at_ordinal(ordinal)} {birth_date.year}"

    def validate_existing(
        self, *, series: WaveSeries, wave_id: str, name: str | None, as_of_date: date
    ) -> None:
        """Reject resumed state that has changed an already issued label.

        Args:
            series: Temperature and direction represented by the saved state.
            wave_id: Retained permanent storm identity.
            name: Saved label, or None before the storm first qualifies.
            as_of_date: Last complete observation day of the saved state.

        Returns:
            None. Existing valid labels are carried forward without reassignment.

        Raises:
            StormNamingError: If a saved historical name is missing or differs.

        Examples:
            A swapped historical name stops resume before another day is written.
        """
        if name is None:
            return
        assignment = self._published.lookup(series, wave_id)
        if assignment is not None:
            if name != assignment.name:
                raise StormNamingError(
                    f"saved name for {wave_id} differs from its published label"
                )
        elif series == _PUBLISHED_SERIES and as_of_date <= self._published.through_date:
            raise StormNamingError(
                f"saved historical name for {wave_id} has no published assignment"
            )
