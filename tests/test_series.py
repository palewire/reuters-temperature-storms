"""Shared series parsing, choices and caller validation."""

import pytest

from reuters_temperature_storms import WaveSeries


def test_shared_choices_keep_cli_order():
    """Check that the combination order retains the expected warm and cold series.

    Returns:
        None.

    Steps:
        1. Compare the supported temperature and direction choices with their expected CLI order.
        2. Check that the combination order retains the expected warm and cold series.

    Examples:
        Run ``pytest tests/test_series.py -k shared_choices_keep_cli_order``.
    """
    assert WaveSeries.choices() == (
        "tasmax/warm",
        "tasmax/cold",
        "tasmin/warm",
        "tasmin/cold",
    )
    for choice in WaveSeries.choices():
        series = WaveSeries.parse(choice)
        assert f"{series.temperature}/{series.direction}" == choice
        assert WaveSeries.from_fields(series.temperature, series.direction) == series


@pytest.mark.parametrize(
    ("temperature", "direction"),
    [("t2m_max", "warm"), ("tasmax", "positive"), (None, "cold"), ([], "warm")],
)
def test_saved_series_fields_reject_unknown_choices(temperature, direction):
    """Check that series validation rejects the unknown temperature or direction.

    Args:
        temperature: Temperature field to validate.
        direction: Warm or cold direction to validate.

    Returns:
        None.

    Steps:
        1. Construct a series using the parametrized unsupported saved field.
        2. Check that series validation rejects the unknown temperature or direction.

    Examples:
        Run ``pytest tests/test_series.py -k saved_series_fields_reject_unknown_choices``.
    """
    with pytest.raises(ValueError, match="series"):
        WaveSeries.from_fields(temperature, direction)


@pytest.mark.parametrize("value", ["tasmax", "tasmax/warm/extra", "tasmax/cool"])
def test_cli_spelling_rejects_unknown_choices(value):
    """Check that parsing rejects the unknown series.

    Args:
        value: Candidate scalar value selected by parametrization.

    Returns:
        None.

    Steps:
        1. Parse the parametrized unsupported CLI spelling.
        2. Check that parsing rejects the unknown series.

    Examples:
        Run ``pytest tests/test_series.py -k cli_spelling_rejects_unknown_choices``.
    """
    with pytest.raises(ValueError, match="series"):
        WaveSeries.parse(value)
