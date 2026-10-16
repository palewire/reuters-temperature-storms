"""Accumulate catalog facts and qualified lifetime coverage per wave."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TYPE_CHECKING, cast

import geopandas as gpd
from shapely import union_all

if TYPE_CHECKING:
    from collections.abc import Iterator

    from shapely import Geometry

    from ._wave_tracking import EndedWave


@dataclass(slots=True)
class _WaveSummary:
    """Catalog facts and temporary qualified shapes for one wave history.

    Steps:
        1. Retain the supplied fields for this record.
        2. Expose the values to the corresponding processing and validation methods.

    Args:
        start_date: First observed date.
        last_observed_date: Most recent observed date.
        peak_area: Largest observed qualified area in square kilometres.
        peak_date: Date of the first occurrence of the peak area.
        observation_count: Number of recorded daily observations.
        name: Assigned storm name, or None for an unnamed wave.
        parent_keys: Permanent identifiers of predecessor waves.
        qualified_shapes: Qualified daily geometries awaiting lifetime combination.
        lifetime_geometry: Combined qualified lifetime geometry, when available.
        end_date: Last active date, or None while active.
        end_reason: Reason the wave ended, when available.
        merged_into_wave_id: Successor identifier when this wave ended in a merger.

    Examples:
        summary = _WaveSummary(day, day, 10.0, day, 1, None, ())
    """

    start_date: date
    last_observed_date: date
    peak_area: float
    peak_date: date
    observation_count: int
    name: str | None
    parent_keys: tuple[str, ...]
    qualified_shapes: list[Geometry] = field(default_factory=list)
    lifetime_geometry: Geometry | None = None
    end_date: date | None = None
    end_reason: str | None = None
    merged_into_wave_id: str | None = None


class CatalogRollup:
    """Build catalog rows, combining only each wave's qualified footprints.

    Steps:
        1. Accumulate observed wave dates, peaks and qualified shapes.
        2. Combine qualified lifetime coverage and emit one row per wave.

    Args:
        None.

    Examples:
        catalog = CatalogRollup()
    """

    __slots__ = ("_waves",)

    def __init__(self) -> None:
        """Create an empty collection of wave summaries and qualified shapes.

        Steps:
            1. Create an empty mapping keyed by series and wave identifier.
            2. Retain summaries there as daily records arrive.

        Args:
            None.

        Returns:
            None.

        Examples:
            ``catalog = CatalogRollup()`` starts an empty catalog.
        """
        self._waves: dict[tuple[str, str, str], _WaveSummary] = {}

    def add_wave_day(self, row: dict[str, object]) -> bool:
        """Update a wave from one validated, chronologically ordered daily row.

        Steps:
            1. Find the wave using its series and permanent ID.
            2. For a new wave, save its first date, peak and qualified shape.
            3. For an existing wave, reject an observation after its ending; update the date and count.
            4. Replace the peak only when the new qualified area is larger, and retain its qualified shape.

        Args:
            row: Wave-day facts and exact qualified geometry. The caller checks
                daily record uniqueness; grace and full geometries are ignored.

        Returns:
            Whether this is the wave's first observation.

        Raises:
            ValueError: If an ended wave appears again.

        Examples:
            ``catalog.add_wave_day(row)`` records one daily observation.
            Equal areas retain the earlier peak date.
        """
        key = (str(row["temperature"]), str(row["direction"]), str(row["wave_id"]))
        row_date = cast("date", row["date"])
        area = float(cast("float | int", row["qualified_area_km2"]))
        name = cast("str | None", row.get("name"))
        parents = tuple(cast("list[str]", row["parent_keys"]))
        wave = self._waves.get(key)
        if wave is None:
            wave = _WaveSummary(
                start_date=cast("date", row["start_date"]),
                last_observed_date=row_date,
                peak_area=area,
                peak_date=row_date,
                observation_count=1,
                name=name,
                parent_keys=parents,
            )
            self._waves[key] = wave
            shape = cast("Geometry | None", row.get("qualified_geometry"))
            if shape is not None:
                wave.qualified_shapes.append(shape)
            return True
        if wave.end_date is not None:
            msg = "an ended wave cannot appear in a later tracking batch"
            raise ValueError(msg)
        wave.last_observed_date = row_date
        wave.observation_count += 1
        if area > wave.peak_area:
            wave.peak_area = area
            wave.peak_date = row_date
        wave.name = name
        wave.parent_keys = parents
        shape = cast("Geometry | None", row.get("qualified_geometry"))
        if shape is not None:
            wave.qualified_shapes.append(shape)
        return False

    def add_ending(self, ended: EndedWave, batch_date: date) -> date:
        """Mark a previously observed wave as ended.

        Steps:
            1. Find the earlier observations for the wave that is ending.
            2. Check that the ending is reported the day after its last observation and has not already been saved.
            3. Save the ending reason and any wave that absorbed it.
            4. Combine its qualified daily shapes and release the separate shapes.

        Args:
            ended: Tracker ending with the final tracked record and end reason.
            batch_date: Date of the batch emitting this ending.

        Returns:
            The wave's last observation date.

        Raises:
            ValueError: If the wave is missing, has already ended, or its
                ending dates disagree with the last observation and batch.

        Examples:
            ``catalog.add_ending(ended, date(2024, 1, 3))`` ends a wave
            last observed on January 2.
        """
        record = ended.final_record
        key = (record.series.temperature, record.series.direction, record.wave_id)
        wave = self._waves.get(key)
        if wave is None:
            msg = "an ending must refer to an earlier wave-day record"
            raise ValueError(msg)
        end_date = (
            ended.end_date
            if isinstance(ended.end_date, date)
            else date.fromisoformat(str(ended.end_date.astype("datetime64[D]")))
        )
        if wave.end_date is not None:
            msg = "a wave can have only one ending"
            raise ValueError(msg)
        if end_date != wave.last_observed_date:
            msg = "ending date must equal the wave's last observed date"
            raise ValueError(msg)
        if batch_date != end_date + timedelta(days=1):
            msg = "an ending must be emitted on the date after its last observation"
            raise ValueError(msg)
        wave.end_date = end_date
        wave.end_reason = ended.end_reason
        wave.merged_into_wave_id = ended.merged_into_wave_id
        self._combine_coverage(wave)
        return end_date

    @staticmethod
    def _combine_coverage(wave: _WaveSummary) -> None:
        """Combine qualified visits once and release individual daily shapes.

        Steps:
            1. Keep a single qualified shape as it is, or join multiple qualified shapes into one covered region.
            2. Save that combined region and clear the list of separate daily shapes.

        Args:
            wave: Summary holding this identity's own qualified observations.

        Returns:
            None. The geographic union replaces its temporary daily shapes.

        Examples:
            CatalogRollup._combine_coverage(wave)
        """
        if wave.qualified_shapes:
            wave.lifetime_geometry = (
                wave.qualified_shapes[0]
                if len(wave.qualified_shapes) == 1
                else union_all(wave.qualified_shapes)
            )
            wave.qualified_shapes.clear()

    def rows(self) -> Iterator[dict[str, object]]:
        """Consume summaries, yielding catalog rows in canonical partition order.

        Steps:
            1. Sort waves by series, birth year and permanent ID.
            2. Take up to 4,096 summaries at a time and omit waves with no qualified peak.
            3. Combine each wave's qualified daily shapes, counting repeatedly visited ground once.
            4. Measure the combined region in square kilometres using EPSG:6933.
            5. Yield dates, peak, duration, ending and covered region, releasing summaries as they are written.

        Args:
            None.

        Yields:
            Catalog rows ordered by temperature, direction, start year
            and wave ID. Duration counts both the first and last dates;
            observation count counts daily records, including grace days.
            Each row carries the union of this identity's qualified visits,
            measured in equal-area EPSG:6933. Written summaries are released.

        Examples:
            ``list(catalog.rows())`` returns catalog rows ready for a DataFrame.
        """
        ordered = sorted(
            self._waves,
            key=lambda key: (
                key[0],
                key[1],
                self._waves[key].start_date.year,
                key[2],
            ),
        )
        for offset in range(0, len(ordered), 4096):
            group = [
                (key, self._waves.pop(key)) for key in ordered[offset : offset + 4096]
            ]
            group = [item for item in group if item[1].peak_area > 0]
            for _, wave in group:
                self._combine_coverage(wave)
            geometries = gpd.GeoSeries(
                [wave.lifetime_geometry for _, wave in group], crs="EPSG:4326"
            )
            areas = geometries.to_crs("EPSG:6933").area / 1_000_000.0
            for ((temperature, direction, wave_id), wave), area in zip(
                group, areas, strict=True
            ):
                yield {
                    "temperature": temperature,
                    "direction": direction,
                    "start_year": wave.start_date.year,
                    "wave_id": wave_id,
                    "start_date": wave.start_date,
                    "last_observed_date": wave.last_observed_date,
                    "end_date": wave.end_date,
                    "end_reason": wave.end_reason,
                    "merged_into_wave_id": wave.merged_into_wave_id,
                    "active": wave.end_date is None,
                    "peak_qualified_area_km2": wave.peak_area,
                    "peak_date": wave.peak_date,
                    "observation_count": wave.observation_count,
                    "name": wave.name,
                    "parent_keys": list(wave.parent_keys),
                    "duration_days": (wave.last_observed_date - wave.start_date).days
                    + 1,
                    "total_area_visited_km2": float(area),
                    "lifetime_geometry": wave.lifetime_geometry,
                }
