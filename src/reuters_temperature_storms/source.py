"""Shared daily file access for wave cells and footprints."""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, ClassVar, Protocol

import fsspec

from .series import WaveSeries

if TYPE_CHECKING:
    from .components import ComponentMembership


class WaveSource(Protocol):
    """Daily components and source settings used by the chronological runner.

    Steps:
        1. Retain the supplied fields for this record.
        2. Expose the values to the corresponding processing and validation methods.

    Args:
        location: Supplied location.
        known_empty_dates: Supplied known empty dates.

    Examples:
        components = source.read(day, WaveSeries("tasmax", "warm"))
    """

    location: str
    known_empty_dates: tuple[date, ...]
    kind: ClassVar[str]

    def read(self, day: date, series: WaveSeries) -> tuple[ComponentMembership, ...]:
        """Read one date into tracker components.

        Steps:
            1. Select the requested date and maximum/minimum, warm/cold series.
            2. Read the appropriate cell or footprint file and check its contents.
            3. Return daily shapes for tracking, or an empty tuple for an accepted
               empty day. Each concrete source owns its reading method.

        Args:
            day: Requested calendar day.
            series: Temperature and direction.

        Returns:
            Daily components, including an empty tuple for an empty day.

        Examples:
            components = source.read(date(2024, 7, 1), WaveSeries("tasmax", "warm"))
        """
        ...


class DailyFileSource:
    """File access shared by the two daily input stages.

    Steps:
        1. Configure the caller-selected file location and access options.
        2. Resolve and validate daily files through the source's read methods.

    Args:
        location: Local directory or fsspec URL for the daily files.
        storage_options: Caller-selected filesystem access settings.
        known_empty_dates: Explicit dates permitted to have missing empty files.

    Examples:
        source = DailyFileSource("./inputs", storage_options={})
    """

    __slots__ = (
        "_filesystem",
        "_path",
        "known_empty_dates",
        "location",
        "storage_options",
    )

    kind: ClassVar[str] = "wave-footprints"
    error_type: ClassVar[type[ValueError]] = ValueError

    def __init__(
        self,
        location: str,
        storage_options: dict[str, object] | None = None,
        *,
        known_empty_dates: tuple[date, ...] = (),
    ) -> None:
        """Configure a root containing field and direction directories.

        Steps:
            1. Require a nonempty root location and plain calendar dates for the
               explicitly allowed empty days.
            2. Retain the caller's file-access settings and choose the filesystem
               from the root URL.
            3. Store the allowed empty dates once each in date order. Reading a
               file later still checks whether a declared empty day has rows.

        Args:
            location: Local directory or fsspec URL ending before the field.
            storage_options: Options passed to fsspec, such as an S3 profile.
            known_empty_dates: Explicit dates permitted to have missing files.
                An existing file on a declared date must contain no rows.

        Returns:
            None. The source retains access settings and declared empty dates.

        Examples:
            source = WaveFootprintSource("/data/footprints/era5")
        """
        if not isinstance(known_empty_dates, tuple) or any(
            not isinstance(day, date) or isinstance(day, datetime)
            for day in known_empty_dates
        ):
            msg = "known_empty_dates must be a tuple of plain calendar dates"
            raise self.error_type(msg)
        if not isinstance(location, str) or not location.strip():
            msg = "daily source location must be a non-empty directory or URL"
            raise self.error_type(msg)
        self.location = location.rstrip("/")
        self.storage_options = dict(storage_options or {})
        self._filesystem, self._path = fsspec.core.url_to_fs(
            self.location, **self.storage_options
        )
        self.known_empty_dates = tuple(sorted(set(known_empty_dates)))

    def path_for(self, day: date, series: WaveSeries) -> str:
        """Select a daily filename after checking the request.

        Steps:
            1. Require a plain date and a supported WaveSeries selection.
            2. Map tasmax/tasmin to t2m_max/t2m_min, and warm/cold to
               positive/negative directory names.
            3. Append YYYY-MM-DD.parquet beneath those directories. For example,
               tasmax/warm selects root/t2m_max/positive/2024-07-01.parquet.

        Args:
            day: Plain calendar date.
            series: Maximum/minimum and warm/cold selection.

        Returns:
            Filesystem path beneath the configured source root.

        Raises:
            ValueError: If the date or series is unsupported.

        Examples:
            path = source.path_for(date(2024, 7, 1), WaveSeries("tasmax", "warm"))
        """
        if not isinstance(day, date) or isinstance(day, datetime):
            raise self.error_type("day must be a datetime.date without a time")
        if not isinstance(series, WaveSeries):
            raise self.error_type("series must use tasmax/tasmin and warm/cold")
        try:
            series.validate()
        except ValueError as error:
            raise self.error_type(
                "series must use tasmax/tasmin and warm/cold"
            ) from error
        field = "t2m_max" if series.temperature == "tasmax" else "t2m_min"
        direction = "positive" if series.direction == "warm" else "negative"
        return f"{self._path.rstrip('/')}/{field}/{direction}/{day.isoformat()}.parquet"

    def list_files(self, series: WaveSeries) -> dict[str, int]:
        """List daily filename sizes for one series without opening file contents.

        Steps:
            1. Check the series and select its field and direction directory.
            2. List file details without opening the daily tables. Treat a missing
               directory as an empty result.
            3. Ignore subdirectories and names without the .parquet suffix.
            4. Require a nonnegative integer size for each selected file, then
               return the filenames and sizes for callers to inspect.

        Args:
            series: Maximum/minimum and warm/cold wave series.

        Returns:
            Daily Parquet filenames and their byte sizes. A missing directory
            returns an empty mapping. Subdirectories are excluded.

        Raises:
            ValueError: If a listed Parquet file lacks a valid byte size.

        Examples:
            sizes = source.list_files(WaveSeries("tasmax", "warm"))
        """
        path = self.path_for(date(1960, 1, 1), series).rsplit("/", 1)[0]
        try:
            entries = self._filesystem.ls(path, detail=True)
        except FileNotFoundError:
            return {}
        result: dict[str, int] = {}
        for entry in entries:
            name = str(entry["name"]).rsplit("/", 1)[-1]
            if entry["type"] != "file" or not name.endswith(".parquet"):
                continue
            size = entry.get("size")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                msg = f"listed daily file has an invalid byte size: {entry['name']}"
                raise self.error_type(msg)
            result[name] = size
        return result
