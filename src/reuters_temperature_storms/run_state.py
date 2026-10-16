"""Typed loading and validation of saved tracking requests and completed runs."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Self

from .records import CompletedGeneration, open_completed_generation
from .series import WaveSeries
from .tracker import WaveTrackerCheckpoint, WaveTrackingBatch

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class WaveRunRequest:
    """The existing saved request, decoded into calendar dates and a series.

    Steps:
        1. Retain the supplied fields for this record.
        2. Expose the values to the corresponding processing and validation methods.

    Args:
        source: Original input location.
        start_date: First tracked day.
        end_date: Last requested day.
        series: Independent temperature and direction.
        known_empty_dates: Explicitly permitted empty input days.
        input_kind: Footprint or cell input spelling.

    Examples:
        request = WaveRunRequest.load(Path("storms-state/request.json"))
    """

    source: str
    start_date: date
    end_date: date
    series: WaveSeries
    known_empty_dates: tuple[date, ...]
    input_kind: str

    def __post_init__(self) -> None:
        """Validate decoded request fields without changing their values.

        Steps:
            1. Require nonempty source text and plain, ordered calendar dates.
            2. Validate the series, input kind and sorted unique empty dates.

        Args:
            None.

        Returns:
            None.

        Raises:
            ValueError: If fields do not describe an accepted request.

        Examples:
            request = WaveRunRequest("inputs", date(2020, 1, 1),
                date(2020, 1, 2), WaveSeries("tasmax", "warm"), (), "wave-footprints")
        """
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("saved request source must be nonempty text")
        days = (self.start_date, self.end_date, *self.known_empty_dates)
        if any(not isinstance(day, date) or isinstance(day, datetime) for day in days):
            raise ValueError("saved request dates must be calendar dates")
        if self.start_date > self.end_date:
            raise ValueError("saved request start must not follow end")
        self.series.validate()
        if self.input_kind not in ("wave-footprints", "wave-cells"):
            raise ValueError("saved request input kind is unsupported")
        if tuple(sorted(set(self.known_empty_dates))) != self.known_empty_dates:
            raise ValueError("saved request empty dates must be sorted and unique")

    @staticmethod
    def _date(value: object) -> date:
        """Decode one strict ISO calendar day.

        Steps:
            1. Require text and parse an ISO calendar date.
            2. Reject spellings that differ from the parsed date's ISO form.

        Args:
            value: JSON date value.

        Returns:
            Decoded calendar day.

        Raises:
            ValueError: If text is not an ISO calendar day.

        Examples:
            day = WaveRunRequest._date("2026-08-31")
        """
        if not isinstance(value, str):
            raise ValueError("saved request date must be ISO calendar text")
        day = date.fromisoformat(value)
        if day.isoformat() != value:
            raise ValueError("saved request date must be ISO calendar text")
        return day

    @classmethod
    def from_bytes(cls, contents: bytes) -> Self:
        """Load the existing JSON shape with explicit field validation.

        Steps:
            1. Decode UTF-8 JSON and require exactly the saved request fields.
            2. Parse the series and each date, then validate the typed request.

        Args:
            contents: UTF-8 saved request JSON.

        Returns:
            Validated typed request.

        Raises:
            ValueError: If JSON or its fields are invalid.

        Examples:
            restored = WaveRunRequest.from_bytes(request.to_bytes())
        """
        try:
            payload = json.loads(contents)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("saved request must contain valid UTF-8 JSON") from error
        if not isinstance(payload, dict) or set(payload) != {
            "source",
            "start_date",
            "end_date",
            "series",
            "known_empty_dates",
            "input_kind",
        }:
            raise ValueError("saved request has an unsupported schema")
        series = payload["series"]
        empty = payload["known_empty_dates"]
        if not isinstance(series, dict) or set(series) != {"temperature", "direction"}:
            raise ValueError("saved request series must have temperature and direction")
        if not isinstance(empty, list):
            raise ValueError("saved request empty dates must be a list")
        return cls(
            source=payload["source"],
            start_date=cls._date(payload["start_date"]),
            end_date=cls._date(payload["end_date"]),
            series=WaveSeries.from_fields(series["temperature"], series["direction"]),
            known_empty_dates=tuple(cls._date(day) for day in empty),
            input_kind=payload["input_kind"],
        )

    @classmethod
    def load(cls, path: Path) -> Self:
        """Read and validate a local saved request.

        Steps:
            1. Read the saved request file as bytes.
            2. Decode and validate the request through from_bytes.

        Args:
            path: Existing request file.

        Returns:
            Validated request.

        Raises:
            OSError: If the file cannot be read.
            ValueError: If its content is invalid.

        Examples:
            request = WaveRunRequest.load(Path("storms-state/request.json"))
        """
        return cls.from_bytes(path.read_bytes())

    def to_bytes(self) -> bytes:
        """Encode the unchanged saved JSON shape deterministically.

        Steps:
            1. Convert dates and the series to their saved JSON fields.
            2. Encode sorted keys without extra whitespace as UTF-8 bytes.

        Args:
            None.

        Returns:
            Sorted compact UTF-8 JSON bytes.

        Examples:
            path.write_bytes(request.to_bytes())
        """
        return json.dumps(
            {
                "source": self.source,
                "start_date": self.start_date.isoformat(),
                "end_date": self.end_date.isoformat(),
                "series": {
                    "temperature": self.series.temperature,
                    "direction": self.series.direction,
                },
                "known_empty_dates": [
                    day.isoformat() for day in self.known_empty_dates
                ],
                "input_kind": self.input_kind,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    @property
    def generation_id(self) -> str:
        """Return the existing deterministic output identifier.

        Steps:
            1. Encode the request deterministically.
            2. Hash those bytes and select the first 24 hexadecimal characters.

        Args:
            None.

        Returns:
            First 24 hexadecimal characters of the request SHA256.

        Examples:
            build_id = request.generation_id
        """
        return hashlib.sha256(self.to_bytes()).hexdigest()[:24]


@dataclass(frozen=True, slots=True)
class CompletedWaveRun:
    """A completed generation and validated saved tracking state.

    Steps:
        1. Retain the supplied fields for this record.
        2. Expose the values to the corresponding processing and validation methods.

    Args:
        location: Original output directory.
        generation: Validated completion metadata.
        request: Typed original request.
        checkpoint: Validated final tracker checkpoint.
        batches: Complete chronological sequence of saved batch paths.

    Examples:
        completed = CompletedWaveRun.load(Path("storms"))
    """

    location: Path
    generation: CompletedGeneration
    request: WaveRunRequest
    checkpoint: WaveTrackerCheckpoint
    batches: tuple[Path, ...]

    @classmethod
    def load(cls, location: Path) -> Self:
        """Load completion, request and checkpoint and check their agreement.

        Steps:
            1. Load the completion metadata, sibling request and checkpoint.
            2. Require matching dates, series, source and generation identifier.
            3. Require every historical daily batch and compare the final batch with the checkpoint.

        Args:
            location: Completed output with its sibling state directory.

        Returns:
            Validated completed run ready for extension.

        Raises:
            ValueError: If metadata, checkpoint or required batches disagree.
            OSError: If required saved files cannot be read.

        Examples:
            completed = CompletedWaveRun.load(Path("old-storms"))
        """
        location = location.absolute()
        state = location.with_name(location.name + "-state")
        if not (location / "_COMPLETE.json").is_file():
            raise ValueError("extension requires a completed source output")
        generation = open_completed_generation(location)
        request = WaveRunRequest.load(state / "request.json")
        manifest = generation.manifest
        if (
            manifest["generation_id"] != request.generation_id
            or manifest["start_date"] != request.start_date.isoformat()
            or manifest["end_date"] != request.end_date.isoformat()
            or manifest["series"]
            != {
                "temperature": request.series.temperature,
                "direction": request.series.direction,
            }
        ):
            raise ValueError("extension completion does not match saved request")
        checkpoint = WaveTrackerCheckpoint.from_bytes(
            (state / "checkpoint.json").read_bytes()
        )
        if (
            checkpoint.through_date != request.end_date
            or checkpoint.requested_start != request.start_date
            or checkpoint.source_identity != request.source
            or checkpoint.series != request.series
        ):
            raise ValueError("extension checkpoint does not match completed request")
        batches = tuple(
            state / "batches" / f"{request.start_date + timedelta(days=i)}.json.gz"
            for i in range((request.end_date - request.start_date).days + 1)
        )
        if not all(path.is_file() for path in batches):
            raise ValueError("extension checkpoint lacks a saved daily batch")
        final = WaveTrackingBatch.from_bytes(
            gzip.decompress(batches[-1].read_bytes()), checkpoint
        )
        if final.date != request.end_date or final.current != checkpoint.active:
            raise ValueError("extension final batch differs from checkpoint")
        return cls(location, generation, request, checkpoint, batches)

    def copy_for_extension(self, state: Path, request: WaveRunRequest) -> None:
        """Copy accepted history into separate state for a later end date.

        Steps:
            1. Require separate state and an unchanged request except for a later end date.
            2. Reuse an already prepared matching request, or require an empty destination.
            3. Copy batches and checkpoint, then save the extended request last; remove partial state if copying fails.

        Args:
            state: New tracker state directory containing an empty batches folder.
            request: Extended request with all other settings unchanged.

        Returns:
            None. Saves copied history and the extended request last.

        Raises:
            ValueError: If requests differ or destination is occupied.
            OSError: If copying fails; partial new state is removed.

        Examples:
            completed.copy_for_extension(Path("new-state"), new_request)
        """
        prior_state = self.location.with_name(self.location.name + "-state")
        if state == self.location or state == prior_state:
            raise ValueError("extension requires a separate destination")
        if (
            replace(self.request, end_date=request.end_date) != request
            or request.end_date <= self.request.end_date
        ):
            raise ValueError(
                "extension must preserve the original request and increase end date"
            )
        saved = state / "request.json"
        if saved.is_file() and WaveRunRequest.load(saved) == request:
            return
        if any(path.is_file() for path in state.rglob("*")):
            raise ValueError("extension requires empty new tracker state")
        try:
            for path in self.batches:
                shutil.copyfile(path, state / "batches" / path.name)
            shutil.copyfile(prior_state / "checkpoint.json", state / "checkpoint.json")
            temporary = saved.with_suffix(".json.tmp")
            temporary.write_bytes(request.to_bytes())
            temporary.replace(saved)
        except OSError:
            shutil.rmtree(state)
            raise
