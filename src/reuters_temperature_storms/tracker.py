"""Checkpointable chronological tracking for one temperature wave series."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from shapely import from_wkb, to_wkb
from shapely.errors import ShapelyError

from ._wave_tracking import (
    EndedWave,
    LineageEdge,
    TrackedWaveComponent,
    WaveTrackingError,
    WaveTrackingUpdate,
    _ComponentGeometryValidator,
    _track_reuters_waves,
    _validate_previous,
)
from .components import (
    ComponentMembership,
    DailyWaveComponentGeometry,
    _is_order_digest,
)
from .naming import (
    NAME_AREA_THRESHOLD_KM2,
    NameSequence,
    StormNamingError,
)
from .series import WaveSeries

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

    from shapely import Geometry


class WaveTrackerError(ValueError):
    """Raised when chronological tracker state cannot be safely resumed."""


@dataclass(frozen=True, slots=True)
class DailyComponentBatch:
    """Finished footprints for one date in the tracker's selected series.

    Args:
        date: Date represented by every component.
        series: Temperature and direction represented by the components.
        components: Finished component memberships for the selected series.

    Examples:
        batch = DailyComponentBatch(
            date=date(2026, 8, 31),
            series=WaveSeries("tasmax", "warm"),
            components=(),
        )
    """

    date: date
    series: WaveSeries
    components: tuple[ComponentMembership, ...]


@dataclass(frozen=True, slots=True)
class WaveTrackerCheckpoint:
    """Storage-neutral state needed to resume one wave series.

    Args:
        requested_start: First tracked date in this run.
        through_date: Final fully tracked date.
        source_identity: Caller-supplied identity for the input location.
        series: Selected temperature and direction.
        active: Active records from the completed date, sorted by wave ID.
        next_name_ordinal: First unused name position for this series.

    Examples:
        restored = WaveTrackerCheckpoint.from_bytes(saved_bytes)
    """

    requested_start: date
    through_date: date
    source_identity: str
    series: WaveSeries
    active: tuple[TrackedWaveComponent, ...]
    next_name_ordinal: int = 0

    @property
    def next_date(self) -> date:
        """Return the only date that can follow this checkpoint.

        Steps:
            1. Add one calendar day to the checkpoint through date.

        Args:
            None.

        Returns:
            Calendar day immediately after ``through_date``.

        Examples:
            next_day = checkpoint.next_date
        """
        return self.through_date + timedelta(days=1)

    def to_bytes(self) -> bytes:
        """Encode validated state as deterministic JSON.

        Steps:
            1. Check the date range, name position and saved active wave records.
            2. Convert the accepted state and shape bytes into JSON-ready fields.
            3. Encode sorted field names as compact UTF-8 JSON.

        Args:
            None.

        Returns:
            Compact UTF-8 JSON containing the selected series and active state.

        Raises:
            WaveTrackerError: If tracked state is malformed.

        Examples:
            saved = checkpoint.to_bytes()
        """
        _validate_checkpoint(self)
        return _json_bytes(_checkpoint_payload(self))

    @classmethod
    def from_bytes(cls, value: bytes) -> WaveTrackerCheckpoint:
        """Restore and validate persisted state.

        Steps:
            1. Require bytes and decode them as UTF-8 JSON.
            2. Require a JSON object and restore its dates, selected series and active records.
            3. Check the restored state before returning a checkpoint that can be resumed.

        Args:
            value: UTF-8 JSON previously returned by ``to_bytes``.

        Returns:
            Restored typed checkpoint.

        Raises:
            TypeError: If ``value`` is not bytes.
            WaveTrackerError: If JSON or tracked state is malformed.

        Examples:
            restored = WaveTrackerCheckpoint.from_bytes(saved)
        """
        if not isinstance(value, bytes):
            raise TypeError("tracker checkpoint bytes must be bytes")
        try:
            document = json.loads(value.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise WaveTrackerError(
                "tracker checkpoint bytes must contain valid UTF-8 JSON"
            ) from error
        if not isinstance(document, dict):
            raise WaveTrackerError("tracker checkpoint document must be an object")
        checkpoint = _checkpoint_from_payload(document)
        _validate_checkpoint(checkpoint)
        return checkpoint


@dataclass(frozen=True, slots=True)
class WaveTrackingBatch:
    """Deterministic tracking output for one date in the selected series.

    Args:
        date: Tracked component date.
        current: Tracked wave-day records ordered by persistent wave ID.
        ended: Explicit endings ordered by persistent wave ID.
        lineage: Split and merge edges ordered by relationship and wave IDs.
        checkpoint: State after this date, ready for the next date.

    Examples:
        tracked = tracker.advance(component_batch)
    """

    date: date
    current: tuple[TrackedWaveComponent, ...]
    ended: tuple[EndedWave, ...]
    lineage: tuple[LineageEdge, ...]
    checkpoint: WaveTrackerCheckpoint

    def to_bytes(self) -> bytes:
        """Encode this daily batch using the existing compact saved format.

        Steps:
            1. Encode current wave records in their stored order, including exact shape bytes.
            2. Encode each ending with its final record, date, reason and optional merge destination.
            3. Encode split and merge relationships with their source, destination and event date.
            4. Return compact UTF-8 JSON with sorted field names without repeating tracker validation.

        Args:
            None.

        Returns:
            Sorted UTF-8 JSON bytes retaining exact record and geometry values.
            Performs no extra tracker or table validation.

        Examples:
            saved = batch.to_bytes()
        """
        document = {
            "date": self.date.isoformat(),
            "current": [_record_payload(record) for record in self.current],
            "ended": [
                {
                    "record": _record_payload(ended.final_record),
                    "end_date": _day_string(ended.end_date),
                    "end_reason": ended.end_reason,
                    "merged_into_wave_id": ended.merged_into_wave_id,
                }
                for ended in self.ended
            ],
            "lineage": [
                {
                    "source_wave_id": edge.source_wave_id,
                    "target_wave_id": edge.target_wave_id,
                    "event_date": _day_string(edge.event_date),
                    "relationship": edge.relationship,
                }
                for edge in self.lineage
            ],
        }

        return _json_bytes(document)

    @classmethod
    def from_bytes(
        cls, value: bytes, checkpoint: WaveTrackerCheckpoint
    ) -> WaveTrackingBatch:
        """Decode saved daily records using an already validated checkpoint.

        Steps:
            1. Require bytes containing valid JSON with exactly the daily batch fields.
            2. Restore current records, endings and split/merge relationships for the checkpoint's series.
            3. Parse the batch date and attach the already accepted checkpoint.
            4. Return records in saved order without checking the checkpoint again for every day.

        Args:
            value: UTF-8 JSON returned by to_bytes, after caller decompression.
            checkpoint: Validated final tracker state for the selected series.

        Returns:
            Restored batch preserving record order and exact stored shapes.
            The checkpoint is not revalidated for each daily batch.

        Raises:
            WaveTrackerError: If the JSON or saved batch fields are malformed.
            TypeError: If value is not bytes.

        Examples:
            restored = WaveTrackingBatch.from_bytes(saved, checkpoint)
        """
        if not isinstance(value, bytes):
            raise TypeError("saved daily batch bytes must be bytes")
        try:
            document = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise WaveTrackerError(
                "saved daily batch must contain valid UTF-8 JSON"
            ) from error
        value = document
        if not isinstance(value, dict) or set(value) != {
            "date",
            "current",
            "ended",
            "lineage",
        }:
            msg = "saved daily batch has an unsupported schema"
            raise WaveTrackerError(msg)
        current = cls._records_from_document(value.get("current"), checkpoint.series)
        ended = cls._endings_from_document(value.get("ended"), checkpoint.series)
        lineage = cls._lineage_from_document(value.get("lineage"), checkpoint.series)
        try:
            return cls(
                date=_calendar_date(value["date"], "staged batch date"),
                current=current,
                ended=ended,
                lineage=lineage,
                checkpoint=checkpoint,
            )
        except (TypeError, ValueError) as error:
            msg = "saved daily batch cannot be restored"
            raise WaveTrackerError(msg) from error

    @staticmethod
    def _records_from_document(
        value: object, series: WaveSeries
    ) -> tuple[TrackedWaveComponent, ...]:
        """Decode daily records for the run's selected series.

        Steps:
            1. Require a list of saved daily records.
            2. Restore each record's fields and exact shapes for the selected series.
            3. Return the records in the same order, or report which kind of saved record failed.

        Args:
            value: Saved list of record payloads.
            series: The single series belonging to this run.

        Returns:
            Records in their saved order.

        Raises:
            WaveTrackerError: If a payload cannot be restored.

        Examples:
            records = WaveTrackingBatch._records_from_document(document["current"], checkpoint.series)
        """
        if not isinstance(value, list):
            raise WaveTrackerError("saved daily current records must be a list")
        try:
            return tuple(_record_from_payload(item, series) for item in value)
        except (TypeError, ValueError) as error:
            raise WaveTrackerError("saved daily record cannot be restored") from error

    @staticmethod
    def _endings_from_document(
        value: object, series: WaveSeries
    ) -> tuple[EndedWave, ...]:
        """Decode explicit wave endings from one staged batch.

        Steps:
            1. Require a list whose entries contain exactly the four ending fields.
            2. Restore each final wave record for the selected series.
            3. Require a died or merged reason, a valid end date and an optional text merge destination.
            4. Create typed ending records in saved order.

        Args:
            value: Candidate list of serialized wave endings.
            series: The selected series for this run.

        Returns:
            Typed explicit ending records in their stored order.

        Raises:
            WaveTrackerError: If ending fields or final records are invalid.

        Examples:
            endings = WaveTrackingBatch._endings_from_document(value, series)
        """
        if not isinstance(value, list):
            msg = "saved daily batch ended must be a list"
            raise WaveTrackerError(msg)
        results: list[EndedWave] = []
        for item in value:
            if not isinstance(item, dict) or set(item) != {
                "record",
                "end_date",
                "end_reason",
                "merged_into_wave_id",
            }:
                msg = "saved daily batch ending is malformed"
                raise WaveTrackerError(msg)
            try:
                record = _record_from_payload(item["record"], series)
                end_reason = item["end_reason"]
                if end_reason not in {"died", "merged"}:
                    raise ValueError("end_reason")
                target = item["merged_into_wave_id"]
                if target is not None and not isinstance(target, str):
                    raise ValueError("merged_into_wave_id")
                results.append(
                    EndedWave(
                        final_record=record,
                        end_date=np.datetime64(
                            _calendar_date(item["end_date"], "staged ending end_date"),
                            "D",
                        ),
                        end_reason=cast("Literal['died', 'merged']", end_reason),
                        merged_into_wave_id=target,
                    )
                )
            except (TypeError, ValueError) as error:
                msg = "saved daily batch ending cannot be restored"
                raise WaveTrackerError(msg) from error
        return tuple(results)

    @staticmethod
    def _lineage_from_document(
        value: object, series: WaveSeries
    ) -> tuple[LineageEdge, ...]:
        """Decode lineage edges from one staged batch.

        Steps:
            1. Require a list whose entries contain exactly the four relationship fields.
            2. Require text source and destination IDs and either a split or merge relationship.
            3. Parse the event date and attach the selected series.
            4. Return typed relationships in saved order.

        Args:
            value: Candidate list of serialized lineage edges.
            series: The selected series for this run.

        Returns:
            Typed lineage edges in their stored order.

        Raises:
            WaveTrackerError: If lineage fields are malformed.

        Examples:
            edges = WaveTrackingBatch._lineage_from_document(value, series)
        """
        if not isinstance(value, list):
            msg = "saved daily batch lineage must be a list"
            raise WaveTrackerError(msg)
        results: list[LineageEdge] = []
        for item in value:
            if not isinstance(item, dict) or set(item) != {
                "source_wave_id",
                "target_wave_id",
                "event_date",
                "relationship",
            }:
                msg = "saved daily batch lineage edge is malformed"
                raise WaveTrackerError(msg)
            source = item["source_wave_id"]
            target = item["target_wave_id"]
            relationship = item["relationship"]
            if (
                not isinstance(source, str)
                or not isinstance(target, str)
                or relationship not in {"split", "merge"}
            ):
                msg = "saved daily batch lineage edge fields are invalid"
                raise WaveTrackerError(msg)
            results.append(
                LineageEdge(
                    series=series,
                    source_wave_id=source,
                    target_wave_id=target,
                    event_date=np.datetime64(
                        _calendar_date(item["event_date"], "staged lineage event_date"),
                        "D",
                    ),
                    relationship=cast("Literal['split', 'merge']", relationship),
                )
            )
        return tuple(results)


class WaveTracker:
    """Advance one selected wave series chronologically and checkpoint each day.

    Args:
        series: Selected temperature and direction.
        requested_start: First component date this tracker will consume.
        source_identity: Caller-supplied input location identity.
        prior_checkpoint: Optional state ending the preceding date.

    Examples:
        tracker = WaveTracker(
            series=WaveSeries("tasmax", "warm"),
            requested_start=date(2026, 8, 31),
            source_identity="footprint-files",
        )
    """

    def __init__(
        self,
        *,
        series: WaveSeries,
        requested_start: date,
        source_identity: str,
        prior_checkpoint: WaveTrackerCheckpoint | None = None,
    ) -> None:
        """Initialize fresh or resumed chronological state.

        Steps:
            1. Check the selected series, first expected date and input location identity.
            2. If resuming, check the saved state and require its next date, series and source to match.
            3. Keep the original run start and accepted checkpoint, or start with empty state.
            4. Prepare reusable shape checks and the independent packaged name sequence.

        Args:
            series: Temperature and direction to track.
            requested_start: First batch date expected by this tracker.
            source_identity: Caller-supplied identity of the input location.
            prior_checkpoint: Complete state immediately before requested start.

        Returns:
            None.

        Raises:
            WaveTrackerError: If series, dates or saved state are incompatible.

        Examples:
            tracker = WaveTracker(
                series=WaveSeries("tasmax", "warm"),
                requested_start=date(2026, 8, 31),
                source_identity="footprints",
            )
        """
        _validate_date(requested_start, "requested_start")
        _validate_identity(source_identity, "source_identity")
        _validate_series(series)
        if prior_checkpoint is not None:
            _validate_checkpoint(prior_checkpoint)
            if prior_checkpoint.next_date != requested_start:
                raise WaveTrackerError(
                    "prior checkpoint through_date does not touch requested_start"
                )
            if prior_checkpoint.source_identity != source_identity:
                raise WaveTrackerError(
                    "prior checkpoint source_identity does not match the source"
                )
            if prior_checkpoint.series != series:
                raise WaveTrackerError(
                    "prior checkpoint series does not match the selected series"
                )
        self._requested_start = (
            requested_start
            if prior_checkpoint is None
            else prior_checkpoint.requested_start
        )
        self._first_expected_date = requested_start
        self._source_identity = source_identity
        self._series = series
        self._checkpoint = prior_checkpoint
        self._geometry_validator = _ComponentGeometryValidator()
        self._name_sequence = NameSequence()

    @property
    def checkpoint(self) -> WaveTrackerCheckpoint | None:
        """Return the latest complete tracker state.

        Steps:
            1. Return the most recently accepted complete state.

        Args:
            None.

        Returns:
            Accepted checkpoint, or ``None`` before the first batch.

        Examples:
            saved_state = tracker.checkpoint
        """
        return self._checkpoint

    def checkpoint_bytes(self) -> bytes:
        """Encode state while reusing checks of this tracker's immutable shapes.

        Steps:
            1. Require a checkpoint from a fully completed day.
            2. Check its current state while reusing earlier checks for unchanged shapes.
            3. Encode the same compact JSON fields as the checkpoint model.

        Args:
            None.

        Returns:
            Bytes identical to ``checkpoint.to_bytes()`` for valid current state.

        Raises:
            WaveTrackerError: If no date is complete or current state is invalid.

        Examples:
            saved = tracker.checkpoint_bytes()
        """
        checkpoint = self._checkpoint
        if checkpoint is None:
            raise WaveTrackerError("no complete tracker checkpoint is available")
        _validate_checkpoint(checkpoint, geometry_validator=self._geometry_validator)
        return _json_bytes(_checkpoint_payload(checkpoint))

    def close(self) -> None:
        """Release cached geometry checks when the stream stops.

        Steps:
            1. Release the geometry validation cache held by the tracker.

        Args:
            None.

        Returns:
            None.

        Examples:
            tracker.close()
        """
        self._geometry_validator.retain(())

    def advance(self, batch: DailyComponentBatch) -> WaveTrackingBatch:
        """Track one consecutive date and save its named active state.

        Steps:
            1. Require the next consecutive date and the tracker's selected series.
            2. Take active wave records and the next unused name position from the preceding checkpoint.
            3. Match today's footprints to the preceding records and calculate continuations, splits and merges.
            4. Keep existing names and assign newly qualifying storms in birth-date and full birth-footprint-digest order.
            5. Save active records and the updated name position as today's checkpoint.
            6. Release checks for shapes no longer active and return sorted records, endings and relationships.

        Args:
            batch: Finished components for the next calendar date.

        Returns:
            Ordered records, endings, lineage and the next checkpoint.

        Raises:
            TypeError: If ``batch`` is not a ``DailyComponentBatch``.
            WaveTrackerError: If components or dates cannot be tracked.

        Examples:
            tracked = tracker.advance(component_batch)
        """
        _validate_batch(
            batch,
            expected_date=(
                self._first_expected_date
                if self._checkpoint is None
                else self._checkpoint.next_date
            ),
            expected_series=self._series,
        )
        previous = () if self._checkpoint is None else self._checkpoint.active
        ordinal = 0 if self._checkpoint is None else self._checkpoint.next_name_ordinal
        try:
            update = self._advance_components(batch, previous)
            prior_names = {record.wave_id: record.name for record in previous}
            named = {}
            for record in sorted(
                update.current,
                key=lambda record: (record.start_date, record.birth_order_digest),
            ):
                name = prior_names.get(record.wave_id)
                if (
                    name is None
                    and record.peak_qualified_area >= NAME_AREA_THRESHOLD_KM2
                ):
                    # Historical keys used the first eight digest characters.
                    # The complete digest keeps every distinct-prefix order and
                    # supplies a fixed suffix order for a future prefix tie.
                    name = (
                        f"{self._name_sequence.name_at_ordinal(ordinal)} "
                        f"{str(record.start_date)[:4]}"
                    )
                    ordinal += 1
                named[record.wave_id] = replace(record, name=name)
        except (WaveTrackingError, StormNamingError) as error:
            raise WaveTrackerError(
                "component batch cannot produce valid tracker state for "
                f"{batch.date.isoformat()} {self._series.temperature}/"
                f"{self._series.direction}: {error}"
            ) from error
        current = tuple(sorted(named.values(), key=lambda record: record.wave_id))
        checkpoint = WaveTrackerCheckpoint(
            requested_start=self._requested_start,
            through_date=batch.date,
            source_identity=self._source_identity,
            series=self._series,
            active=current,
            next_name_ordinal=ordinal,
        )
        self._checkpoint = checkpoint
        self._geometry_validator.retain(record.component for record in current)
        return WaveTrackingBatch(
            date=batch.date,
            current=current,
            ended=tuple(
                sorted(update.ended, key=lambda ending: ending.final_record.wave_id)
            ),
            lineage=tuple(
                sorted(
                    update.lineage,
                    key=lambda edge: (
                        edge.relationship,
                        edge.source_wave_id,
                        edge.target_wave_id,
                    ),
                )
            ),
            checkpoint=checkpoint,
        )

    def _advance_components(
        self,
        batch: DailyComponentBatch,
        previous: tuple[TrackedWaveComponent, ...],
    ) -> WaveTrackingUpdate:
        """Apply the scientific tracker to one day's components.

        Steps:
            1. Represent the batch date as a calendar day for the geometry tracker.
            2. Return an empty update when both preceding and current components are empty.
            3. Otherwise track today's components using previous records and reusable shape checks.

        Args:
            batch: Validated next-day components.
            previous: Active records from the preceding day.

        Returns:
            Unnamed update preserving exact tracking, ending and lineage rules.

        Examples:
            update = tracker._advance_components(batch, previous)
        """
        numpy_day = np.datetime64(batch.date, "D")
        if not previous and not batch.components:
            return WaveTrackingUpdate(numpy_day, self._series, (), (), ())
        return _track_reuters_waves(
            date=numpy_day,
            series=self._series,
            components=batch.components,
            previous=previous,
            seed=not previous,
            geometry_validator=self._geometry_validator,
        )


def track_reuters_component_batches(
    batches: Iterable[DailyComponentBatch],
    *,
    series: WaveSeries,
    requested_start: date,
    source_identity: str,
    prior_checkpoint: WaveTrackerCheckpoint | None = None,
) -> Iterator[WaveTrackingBatch]:
    """Track one ordered series from fresh or restored state.

    Steps:
        1. Create a fresh or restored tracker.
        2. Advance it through the ordered component batches and yield daily output.
        3. Release the tracker cache when iteration finishes.

    Args:
        batches: Consecutive dates containing finished components.
        series: Temperature and direction to track.
        requested_start: First date expected in ``batches``.
        source_identity: Caller-supplied input location identity.
        prior_checkpoint: Complete tracker state immediately before the start.

    Yields:
        Ordered tracking output and checkpoint for each component batch.

    Raises:
        WaveTrackerError: If batches or saved state are incompatible.

    Examples:
        tracked = track_reuters_component_batches(
            batches,
            series=WaveSeries("tasmax", "warm"),
            requested_start=date(2026, 8, 31),
            source_identity="footprint-files",
        )
    """
    tracker = WaveTracker(
        series=series,
        requested_start=requested_start,
        source_identity=source_identity,
        prior_checkpoint=prior_checkpoint,
    )
    try:
        yield from (tracker.advance(batch) for batch in batches)
    finally:
        tracker.close()


def _validate_batch(
    batch: DailyComponentBatch, *, expected_date: date, expected_series: WaveSeries
) -> None:
    """Validate one component batch at the chronological boundary.

    Steps:
        1. Require a daily batch for the selected series and exactly the next date.
        2. Require its components to be a tuple of component membership records.

    Args:
        batch: Candidate finished component batch.
        expected_date: Only acceptable next date.
        expected_series: Selected temperature and direction owned by the tracker.

    Returns:
        None.

    Raises:
        TypeError: If batch has the wrong type.
        WaveTrackerError: If the date, series or components are incompatible.

    Examples:
        _validate_batch(batch, expected_date=day, expected_series=series)
    """
    if not isinstance(batch, DailyComponentBatch):
        raise TypeError("batch must be a DailyComponentBatch")
    _validate_series(batch.series)
    if batch.series != expected_series:
        raise WaveTrackerError(
            "component batch series does not match the selected tracker series"
        )
    _validate_date(batch.date, "component batch date")
    if batch.date != expected_date:
        raise WaveTrackerError(
            "component batch date must be exactly the next tracker date"
        )
    if not isinstance(batch.components, tuple) or any(
        not isinstance(component, ComponentMembership) for component in batch.components
    ):
        raise WaveTrackerError(
            "component batch must contain a component membership tuple"
        )


def _validate_series(series: WaveSeries) -> None:
    """Require one supported temperature and direction.

    Steps:
        1. Check the shared temperature/direction model.
        2. Report unsupported pairs through the module validation error.

    Args:
        series: Candidate wave series.

    Returns:
        None.

    Raises:
        WaveTrackerError: If the series is unsupported.

    Examples:
        _validate_series(WaveSeries("tasmax", "warm"))
    """
    if not isinstance(series, WaveSeries):
        raise WaveTrackerError("series must be a WaveSeries")
    try:
        series.validate()
    except ValueError as error:
        raise WaveTrackerError(
            "tracker series is not a Reuters temperature/direction pair"
        ) from error


def _validate_checkpoint(
    checkpoint: WaveTrackerCheckpoint,
    *,
    geometry_validator: _ComponentGeometryValidator | None = None,
) -> None:
    """Validate checkpoint chronology, names and active scientific state.

    Steps:
        1. Check the saved date range, source identity, selected series and active record types.
        2. Require a nonnegative next-name position that includes already named active storms.
        3. Check that names match the naming threshold and retain each wave's starting year.
        4. Require active records in wave-ID order and check their suitability for the next day.

    Args:
        checkpoint: Candidate state before restoring or writing.
        geometry_validator: Cached immutable-shape checker, or None for full checks.

    Returns:
        None.

    Raises:
        TypeError: If checkpoint has the wrong type.
        WaveTrackerError: If state is invalid.

    Examples:
        _validate_checkpoint(checkpoint)
    """
    if not isinstance(checkpoint, WaveTrackerCheckpoint):
        raise TypeError("prior_checkpoint must be a WaveTrackerCheckpoint")
    _validate_date(checkpoint.requested_start, "checkpoint requested_start")
    _validate_date(checkpoint.through_date, "checkpoint through_date")
    if checkpoint.through_date < checkpoint.requested_start:
        raise WaveTrackerError(
            "checkpoint through_date must not precede requested_start"
        )
    _validate_identity(checkpoint.source_identity, "checkpoint source_identity")
    _validate_series(checkpoint.series)
    if not isinstance(checkpoint.active, tuple) or any(
        not isinstance(record, TrackedWaveComponent) for record in checkpoint.active
    ):
        raise WaveTrackerError("checkpoint active must be a tracked component tuple")
    if (
        type(checkpoint.next_name_ordinal) is not int
        or checkpoint.next_name_ordinal < 0
    ):
        raise WaveTrackerError("checkpoint must retain a nonnegative name ordinal")
    named_count = sum(record.name is not None for record in checkpoint.active)
    if checkpoint.next_name_ordinal < named_count:
        raise WaveTrackerError(
            "checkpoint name ordinal cannot precede its active named storms"
        )
    for record in checkpoint.active:
        if (record.name is not None) != (
            record.peak_qualified_area >= NAME_AREA_THRESHOLD_KM2
        ):
            raise WaveTrackerError(
                "checkpoint names must match the inclusive peak threshold for naming"
            )
        if record.name is not None and (
            not record.name.endswith(f" {str(record.start_date)[:4]}")
            or not record.name.removesuffix(f" {str(record.start_date)[:4]}")
        ):
            raise WaveTrackerError("checkpoint name must retain the wave start year")
    if (
        tuple(sorted(checkpoint.active, key=lambda record: record.wave_id))
        != checkpoint.active
    ):
        raise WaveTrackerError(
            "checkpoint active records must be sorted by persistent wave ID"
        )
    try:
        _validate_previous(
            np.datetime64(checkpoint.next_date, "D"),
            checkpoint.series,
            checkpoint.active,
            geometry_validator=geometry_validator,
        )
    except (WaveTrackingError, TypeError) as error:
        raise WaveTrackerError(
            "checkpoint active state is invalid for "
            f"{checkpoint.series.temperature}/{checkpoint.series.direction}: {error}"
        ) from error


def _checkpoint_payload(checkpoint: WaveTrackerCheckpoint) -> dict[str, object]:
    """Build JSON-compatible state for the selected series.

    Steps:
        1. Encode the run dates, source identity, selected series and next-name position.
        2. Encode active records in their saved order, including their exact shape bytes.

    Args:
        checkpoint: State to encode.

    Returns:
        Dates, selected series, active records and next name position.

    Examples:
        payload = _checkpoint_payload(checkpoint)
    """
    return {
        "requested_start": checkpoint.requested_start.isoformat(),
        "through_date": checkpoint.through_date.isoformat(),
        "source_identity": checkpoint.source_identity,
        "temperature": checkpoint.series.temperature,
        "direction": checkpoint.series.direction,
        "next_name_ordinal": checkpoint.next_name_ordinal,
        "active": [_record_payload(record) for record in checkpoint.active],
    }


def _checkpoint_from_payload(payload: dict[object, object]) -> WaveTrackerCheckpoint:
    """Decode typed checkpoint state from its JSON object.

    Steps:
        1. Read the selected temperature/direction series and require an active-record list.
        2. Restore dates, input identity, active records and the next unused name position.
        3. Return the checkpoint for its caller to check before use.

    Args:
        payload: JSON-compatible checkpoint fields.

    Returns:
        Decoded checkpoint before full scientific validation.

    Raises:
        WaveTrackerError: If field types or selected series are unsupported.

    Examples:
        checkpoint = _checkpoint_from_payload(payload)
    """
    series = _series_from_fields(
        _string_field(payload, "temperature"), _string_field(payload, "direction")
    )
    active = payload.get("active")
    if not isinstance(active, list):
        raise WaveTrackerError("tracker checkpoint active must be a list")
    return WaveTrackerCheckpoint(
        requested_start=_date_field(payload, "requested_start"),
        through_date=_date_field(payload, "through_date"),
        source_identity=_string_field(payload, "source_identity"),
        series=series,
        active=tuple(_record_from_payload(record, series) for record in active),
        next_name_ordinal=_integer_field(payload, "next_name_ordinal"),
    )


def _record_payload(record: TrackedWaveComponent) -> dict[str, object]:
    """Encode every active record field needed for exact tracker resume.

    Steps:
        1. Read the wave's current component geometry.
        2. Encode identity, start, peak, continuation, parents and name fields.
        3. Encode component shapes as binary-geometry text with their stored dates and areas.

    Args:
        record: Active tracked wave record.

    Returns:
        JSON-compatible active-record data with the stored WKB shapes.

    Examples:
        payload = _record_payload(active_record)
    """
    geometry = record.component.geometry
    return {
        "component_id": record.component_id,
        "order_digest": record.component.order_digest,
        "wave_id": record.wave_id,
        "start_date": _day_string(record.start_date),
        "birth_cell_id": record.birth_cell_id,
        "birth_order_digest": record.birth_order_digest,
        "peak_qualified_area": record.peak_qualified_area,
        "peak_date": _day_string(record.peak_date),
        "continues": record.continues,
        "overlap_percent": record.overlap_percent,
        "parent_wave_ids": list(record.parent_wave_ids),
        "name": record.name,
        "component": {
            "time": _day_string(geometry.time),
            "label": geometry.label,
            "geometry": _geometry_payload(geometry.geometry),
            "grace_geometry": _geometry_payload(geometry.grace_geometry),
            "full_geometry": _geometry_payload(geometry.full_geometry),
            "area_qualified": geometry.area_qualified,
            "area_grace": geometry.area_grace,
            "area_gap_filled": geometry.area_gap_filled,
            "qualified_fraction": geometry.qualified_fraction,
        },
    }


def _record_from_payload(value: object, series: WaveSeries) -> TrackedWaveComponent:
    """Decode one complete record from a checkpoint or saved daily batch.

    Steps:
        1. Require a record object with component fields and a text list of parent IDs.
        2. Restore component dates, labels, exact shapes and area measurements.
        3. Require the saved name field, accepting either nonblank text or no name.
        4. Check that a first-day record retains its daily ordering as birth ordering.
        5. Restore the wave identity, start, peak and continuation fields without changing stored order.

    Args:
        value: Candidate active-record object.
        series: Selected series that owns the active record.

    Returns:
        A tracked record with exact component membership and geometry state.

    Raises:
        WaveTrackerError: If required fields are missing or malformed.

    Examples:
        record = _record_from_payload(saved_record, series)
    """
    if not isinstance(value, dict):
        msg = "tracker checkpoint active record must be an object"
        raise WaveTrackerError(msg)
    component_data = value.get("component")
    if not isinstance(component_data, dict):
        msg = "tracker checkpoint active record must contain component data"
        raise WaveTrackerError(msg)
    parent_ids = value.get("parent_wave_ids")
    if not isinstance(parent_ids, list) or any(
        not isinstance(parent, str) for parent in parent_ids
    ):
        msg = "tracker checkpoint parent_wave_ids must be a string list"
        raise WaveTrackerError(msg)
    component = ComponentMembership(
        component_id=_string_field(value, "component_id"),
        order_digest=_order_digest_field(value, "order_digest"),
        geometry=DailyWaveComponentGeometry(
            time=np.datetime64(_string_field(component_data, "time"), "D"),
            label=_integer_field(component_data, "label"),
            geometry=_geometry_from_payload(component_data.get("geometry")),
            grace_geometry=_geometry_from_payload(component_data.get("grace_geometry")),
            full_geometry=_required_geometry(
                _geometry_from_payload(component_data.get("full_geometry")),
                "full_geometry",
            ),
            area_qualified=_float_field(component_data, "area_qualified"),
            area_grace=_float_field(component_data, "area_grace"),
            area_gap_filled=_float_field(component_data, "area_gap_filled"),
            qualified_fraction=_float_field(component_data, "qualified_fraction"),
        ),
    )
    if "name" not in value:
        msg = "tracker checkpoint must retain its name field"
        raise WaveTrackerError(msg)
    name = value.get("name")
    if name is not None and (not isinstance(name, str) or not name.strip()):
        msg = "tracker checkpoint name must be nonblank text or None"
        raise WaveTrackerError(msg)
    start_date = np.datetime64(_string_field(value, "start_date"), "D")
    birth_order_digest = _order_digest_field(value, "birth_order_digest")
    # The first day's two ordering fields describe the same footprint. Later
    # days can differ because the wave moved, so keep their saved birth order.
    if (
        start_date == component.geometry.time
        and birth_order_digest != component.order_digest
    ):
        raise WaveTrackerError(
            "saved birth_order_digest must match the birth component order_digest"
        )
    return TrackedWaveComponent(
        series=series,
        component=component,
        component_id=component.component_id,
        wave_id=_string_field(value, "wave_id"),
        start_date=start_date,
        birth_cell_id=_integer_field(value, "birth_cell_id"),
        birth_order_digest=birth_order_digest,
        peak_qualified_area=_float_field(value, "peak_qualified_area"),
        peak_date=np.datetime64(_string_field(value, "peak_date"), "D"),
        continues=_boolean_field(value, "continues"),
        overlap_percent=_float_field(value, "overlap_percent"),
        parent_wave_ids=tuple(parent_ids),
        name=name,
    )


def _order_digest_field(payload: dict[object, object], name: str) -> str:
    """Require one complete saved ordering digest without rebuilding geometry.

    Args:
        payload: Saved active-record fields.
        name: Required daily or birth ordering field.

    Returns:
        The unchanged 32-character lowercase hexadecimal digest.

    Raises:
        WaveTrackerError: If the field is absent or malformed.

    Examples:
        digest = _order_digest_field(saved_record, "birth_order_digest")
    """
    value = _string_field(payload, name)
    if not _is_order_digest(value):
        raise WaveTrackerError(
            f"tracker checkpoint {name} must be a 32-character lowercase footprint digest"
        )
    return value


def _geometry_payload(value: Geometry | None) -> str | None:
    """Encode optional geometry as stable binary WKB text.

    Steps:
        1. Keep an absent shape as None.
        2. Encode a present shape as binary geometry with a fixed byte order and then as Base64 text.

    Args:
        value: Optional geographic geometry.

    Returns:
        Base64 WKB, or ``None`` when geometry is absent.

    Examples:
        saved_shape = _geometry_payload(shape)
    """
    return None if value is None else _binary(to_wkb(value, byte_order=1))


def _geometry_from_payload(value: object) -> Geometry | None:
    """Decode optional WKB geometry from a checkpoint field.

    Steps:
        1. Keep a missing shape field as None.
        2. Require text, decode its Base64 bytes and restore the binary geometry.
        3. Report malformed shape data as a tracker checkpoint error.

    Args:
        value: ``None`` or Base64 WKB text.

    Returns:
        The decoded geometry, or ``None``.

    Raises:
        WaveTrackerError: If WKB data is malformed.

    Examples:
        shape = _geometry_from_payload(saved_shape)
    """
    if value is None:
        return None
    if not isinstance(value, str):
        msg = "tracker checkpoint geometry must be Base64 WKB or null"
        raise WaveTrackerError(msg)
    try:
        return from_wkb(_unbinary(value, "geometry"))
    except (ShapelyError, TypeError, ValueError) as error:
        msg = "tracker checkpoint geometry WKB is invalid"
        raise WaveTrackerError(msg) from error


def _required_geometry(value: Geometry | None, label: str) -> Geometry:
    """Require a non-null geometry where the tracker schema needs one.

    Steps:
        1. Check whether the decoded shape is absent.
        2. Return the shape or raise a field-specific error.

    Args:
        value: Decoded optional geometry.
        label: Name of the schema field.

    Returns:
        The non-null geometry.

    Raises:
        WaveTrackerError: If ``value`` is absent.

    Examples:
        shape = _required_geometry(geometry, "full_geometry")
    """
    if value is None:
        msg = f"tracker checkpoint {label} must not be null"
        raise WaveTrackerError(msg)
    return value


def _binary(value: bytes) -> str:
    """Encode bytes as ASCII Base64.

    Steps:
        1. Encode bytes as Base64.
        2. Decode the result as ASCII text.

    Args:
        value: Binary data to encode.

    Returns:
        ASCII Base64 text.

    Examples:
        assert _binary(b"hello") == "aGVsbG8="
    """
    return base64.b64encode(value).decode("ascii")


def _unbinary(value: object, label: str) -> bytes:
    """Decode strict ASCII Base64 checkpoint data.

    Steps:
        1. Require a text value.
        2. Decode strict ASCII Base64 and report invalid input.

    Args:
        value: Candidate Base64 text.
        label: Name of the encoded field.

    Returns:
        Decoded bytes.

    Raises:
        WaveTrackerError: If text is not valid Base64.

    Examples:
        assert _unbinary("aGVsbG8=", "data") == b"hello"
    """
    if not isinstance(value, str):
        msg = f"tracker checkpoint {label} must be Base64 text"
        raise WaveTrackerError(msg)
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as error:
        msg = f"tracker checkpoint {label} is not valid Base64"
        raise WaveTrackerError(msg) from error


def _date_field(value: Mapping[object, object], name: str) -> date:
    """Read one ISO calendar-date field from a checkpoint object.

    Steps:
        1. Read a required nonempty string field.
        2. Parse its ISO calendar date.

    Args:
        value: JSON object containing the field.
        name: Required field name.

    Returns:
        Parsed calendar date.

    Raises:
        WaveTrackerError: If the field is missing or not an ISO date.

    Examples:
        assert _date_field({"date": "2024-01-01"}, "date") == date(2024, 1, 1)
    """
    field = _string_field(value, name)
    try:
        return date.fromisoformat(field)
    except ValueError as error:
        msg = f"tracker checkpoint {name} must be an ISO calendar date"
        raise WaveTrackerError(msg) from error


def _string_field(value: Mapping[object, object], name: str) -> str:
    """Read one required string field from a checkpoint object.

    Steps:
        1. Read the named field.
        2. Require a nonempty string.

    Args:
        value: JSON object containing the field.
        name: Required field name.

    Returns:
        The non-empty string field.

    Raises:
        WaveTrackerError: If the field is missing or not a non-empty string.

    Examples:
        assert _string_field({"name": "Vera"}, "name") == "Vera"
    """
    field = value.get(name)
    if not isinstance(field, str) or not field:
        msg = f"tracker checkpoint {name} must be a non-empty string"
        raise WaveTrackerError(msg)
    return field


def _integer_field(value: Mapping[object, object], name: str) -> int:
    """Read one required integer field from a checkpoint object.

    Steps:
        1. Read the named field.
        2. Require an integer while rejecting Boolean values.

    Args:
        value: JSON object containing the field.
        name: Required field name.

    Returns:
        The integer field.

    Raises:
        WaveTrackerError: If the field is missing or not an integer.

    Examples:
        assert _integer_field({"count": 1}, "count") == 1
    """
    field = value.get(name)
    if isinstance(field, bool) or not isinstance(field, int):
        msg = f"tracker checkpoint {name} must be an integer"
        raise WaveTrackerError(msg)
    return field


def _float_field(value: Mapping[object, object], name: str) -> float:
    """Read one required finite floating-point field from a checkpoint object.

    Steps:
        1. Read the named numeric field.
        2. Reject Booleans and nonfinite values and convert to float.

    Args:
        value: JSON object containing the field.
        name: Required field name.

    Returns:
        The field converted to float.

    Raises:
        WaveTrackerError: If the field is missing or non-finite.

    Examples:
        assert _float_field({"area": 2}, "area") == 2.0
    """
    field = value.get(name)
    if isinstance(field, bool) or not isinstance(field, (int, float)):
        msg = f"tracker checkpoint {name} must be a finite number"
        raise WaveTrackerError(msg)
    converted = float(field)
    if not np.isfinite(converted):
        msg = f"tracker checkpoint {name} must be a finite number"
        raise WaveTrackerError(msg)
    return converted


def _boolean_field(value: Mapping[object, object], name: str) -> bool:
    """Read one required Boolean field from a checkpoint object.

    Steps:
        1. Read the named field.
        2. Require a Boolean value.

    Args:
        value: JSON object containing the field.
        name: Required field name.

    Returns:
        The Boolean field.

    Raises:
        WaveTrackerError: If the field is missing or not Boolean.

    Examples:
        assert _boolean_field({"active": True}, "active") is True
    """
    field = value.get(name)
    if not isinstance(field, bool):
        msg = f"tracker checkpoint {name} must be a Boolean"
        raise WaveTrackerError(msg)
    return field


def _validate_date(value: date, label: str) -> None:
    """Validate a calendar date without accepting datetime subclasses.

    Steps:
        1. Require a Python date.
        2. Reject datetime subclasses.

    Args:
        value: Candidate date.
        label: Name used in a validation error.

    Returns:
        None after validation or writing succeeds.

    Raises:
        WaveTrackerError: If value is not a calendar date.

    Examples:
        _validate_date(date(2024, 1, 1), "date")
    """
    if not isinstance(value, date) or isinstance(value, datetime):
        msg = f"{label} must be a datetime.date"
        raise WaveTrackerError(msg)


def _validate_identity(value: str, label: str) -> None:
    """Validate a caller-supplied immutable identity label.

    Steps:
        1. Require a nonempty text identity.
        2. Reject blank or non-string values.

    Args:
        value: Candidate identity string.
        label: Name used in a validation error.

    Returns:
        None after validation or writing succeeds.

    Raises:
        WaveTrackerError: If identity is blank or not a string.

    Examples:
        _validate_identity("example", "source_identity")
    """
    if not isinstance(value, str) or not value.strip():
        msg = f"{label} must be a non-empty caller-supplied identity"
        raise WaveTrackerError(msg)


def _series_from_fields(temperature: str, direction: str) -> WaveSeries:
    """Build a typed series from validated checkpoint strings.

    Steps:
        1. Build the shared series model from its fields.
        2. Translate unsupported pairs into a checkpoint validation error.

    Args:
        temperature: Validated temperature field.
        direction: Validated direction field.

    Returns:
        The matching temperature wave series.

    Raises:
        WaveTrackerError: If fields do not identify a supported series.

    Examples:
        series = _series_from_fields("tasmax", "warm")
    """
    try:
        return WaveSeries.from_fields(temperature, direction)
    except ValueError as error:
        msg = "tracker checkpoint series is not a Reuters temperature/direction pair"
        raise WaveTrackerError(msg) from error


def _day_string(value: np.datetime64) -> str:
    """Convert a NumPy date to its ISO calendar-day representation.

    Steps:
        1. Convert the NumPy value to calendar-day precision.
        2. Format an ISO date string.

    Args:
        value: NumPy date value.

    Returns:
        ISO calendar-day text.

    Examples:
        assert _day_string(np.datetime64("2024-01-01")) == "2024-01-01"
    """
    result = np.datetime_as_string(value.astype("datetime64[D]"), unit="D")
    assert isinstance(result, str)
    return result


def _json_bytes(value: object) -> bytes:
    """Encode JSON deterministically for checkpoint persistence.

    Steps:
        1. Sort JSON field names and remove optional whitespace.
        2. Reject nonfinite numbers, then encode the compact JSON as UTF-8 bytes.

    Args:
        value: JSON-compatible value.

    Returns:
        Compact sorted UTF-8 JSON bytes.

    Examples:
        saved = _json_bytes({"date": "2024-07-01"})
    """
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _calendar_date(value: object, label: str) -> date:
    """Decode a strict ISO calendar day from saved JSON.

    Steps:
        1. Require text that parses as a calendar date.
        2. Require its spelling to match YYYY-MM-DD exactly before returning the date.

    Args:
        value: Candidate ISO date text.
        label: Field label used in validation errors.

    Returns:
        Parsed calendar date.

    Raises:
        WaveTrackerError: If text is not a strict ISO calendar date.

    Examples:
        day = _calendar_date("1960-01-01", "batch date")
    """
    if not isinstance(value, str):
        raise WaveTrackerError(f"{label} must be an ISO calendar date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise WaveTrackerError(f"{label} must be an ISO calendar date") from error
    if parsed.isoformat() != value:
        raise WaveTrackerError(f"{label} must be an ISO calendar date")
    return parsed
