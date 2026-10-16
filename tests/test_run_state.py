"""Saved-request schema validation and unchanged encoded format."""

from __future__ import annotations

import json
from datetime import date

import pytest

from reuters_temperature_storms.run_state import WaveRunRequest
from reuters_temperature_storms.series import WaveSeries


def test_request_roundtrip_preserves_existing_document() -> None:
    """Retain the existing JSON shape and deterministic identifier.

    Args:
        None.

    Returns:
        None. Typed decoding retains every accepted request value.

    Examples:
        Run ``pytest tests/test_run_state.py``.

    Steps:
        1. Retain the existing JSON shape and deterministic identifier.
        2. Check that typed decoding retains every accepted request value.
    """
    request = WaveRunRequest(
        "inputs",
        date(2026, 8, 1),
        date(2026, 8, 31),
        WaveSeries("tasmax", "warm"),
        (date(2026, 8, 1),),
        "wave-footprints",
    )
    assert WaveRunRequest.from_bytes(request.to_bytes()) == request
    assert len(request.generation_id) == 24
    assert json.loads(request.to_bytes())["known_empty_dates"] == ["2026-08-01"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source", ""),
        ("source", 42),
        ("start_date", "20260801"),
        ("start_date", "2026-09-01"),
        ("end_date", None),
        ("series", {"temperature": "tasmax", "direction": "invalid"}),
        ("input_kind", "other"),
        ("known_empty_dates", "2026-08-01"),
        ("known_empty_dates", ["2026-08-02", "2026-08-01"]),
        ("known_empty_dates", ["2026-08-01", "2026-08-01"]),
    ],
)
def test_request_rejects_invalid_fields(field: str, value: object) -> None:
    """Reject malformed saved fields through the object loader.

    Args:
        field: Saved request field to replace.
        value: Invalid replacement value.

    Returns:
        None. Invalid documents cannot become typed requests.

    Examples:
        Run ``pytest tests/test_run_state.py -k rejects``.

    Steps:
        1. Reject malformed saved fields through the object loader.
        2. Check that invalid documents cannot become typed requests.
    """
    payload: dict[str, object] = {
        "source": "inputs",
        "start_date": "2026-08-01",
        "end_date": "2026-08-31",
        "series": {"temperature": "tasmax", "direction": "warm"},
        "input_kind": "wave-footprints",
        "known_empty_dates": [],
    }
    payload[field] = value
    with pytest.raises(ValueError, match=r"saved request|series"):
        WaveRunRequest.from_bytes(json.dumps(payload).encode())


@pytest.mark.parametrize("contents", [b"not-json", b"[]", b"{}", b"\xff"])
def test_request_rejects_invalid_documents(contents: bytes) -> None:
    """Reject invalid JSON and unsupported document shapes.

    Args:
        contents: Invalid saved document bytes.

    Returns:
        None. Loader gives a readable validation error.

    Examples:
        Run ``pytest tests/test_run_state.py``.

    Steps:
        1. Reject invalid JSON and unsupported document shapes.
        2. Check that loader gives a readable validation error.
    """
    with pytest.raises(ValueError, match="saved request"):
        WaveRunRequest.from_bytes(contents)
