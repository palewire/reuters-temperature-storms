"""Check fixed five-day score arithmetic and accepted-table boundaries."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from reuters_temperature_storms.cell_scores import WaveCellScores
from reuters_temperature_storms.cells import WaveCellError
from reuters_temperature_storms.weighted_scores import WeightedScoreBuilder

DAY = date(2024, 7, 1)
LONGITUDE, LATITUDE = np.meshgrid(
    np.arange(-179.75, 180.25, 0.25), np.arange(-90, 90.25, 0.25)
)


def score_frame(value, *, dtype="float64"):
    """Make a complete grid with distinct values at a few physical cells.

    Args:
        value: Asymmetric score at longitude zero on the equator.
        dtype: Native score storage precision.

    Returns:
        Complete coordinate table with score values ready for model acceptance.

    Examples:
        frame = score_frame(2.0, dtype="float32")

    Steps:
        1. Make a complete grid with distinct values at a few physical cells.
    """
    native = np.dtype(dtype)
    values = np.zeros(LONGITUDE.shape, dtype=native)
    equator = 90 * 4
    zero = round(179.75 * 4)
    values[equator, zero] = value
    values[equator, zero + 1] = -2 * value
    values[equator, zero + 2] = np.nextafter(native.type(1), native.type(0))
    values[equator, zero + 3] = 1
    values[equator, zero + 4] = 1.25
    return pd.DataFrame(
        {
            "longitude": LONGITUDE.ravel(),
            "latitude": LATITUDE.ravel(),
            "score": values.ravel(),
        }
    )


@pytest.fixture(scope="module")
def daily_scores():
    """Accept six consecutive grids once for window and arithmetic tests.

    Args:
        None.

    Returns:
        Six accepted score models, newest first, with unequal daily values.

    Examples:
        window = daily_scores[:5]

    Steps:
        1. Accept six consecutive grids once for window and arithmetic tests.
    """
    return tuple(
        WaveCellScores(score_frame(2 ** (age + 1)), DAY - timedelta(days=age))
        for age in range(6)
    )


def value_at(scores, longitude):
    """Read one equatorial physical cell without depending on row position.

    Args:
        scores: Accepted daily or weighted score table.
        longitude: Longitude of the equatorial cell to inspect.

    Returns:
        The exact stored floating score at that coordinate.

    Examples:
        value = value_at(weighted, 0.25)

    Steps:
        1. Read one equatorial physical cell without depending on row position.
    """
    frame = scores.frame
    return frame.loc[
        frame.longitude.eq(longitude) & frame.latitude.eq(0), "score"
    ].iloc[0]


def test_fixed_weights_favor_recent_days_and_return_complete_float64(daily_scores):
    """Distinguish weighting and date direction with unequal daily scores.

    Args:
        daily_scores: Accepted consecutive daily grids.

    Returns:
        None. All cells remain and the explicit five-term weighted mean matches.

    Examples:
        pytest tests/test_weighted_scores.py -k fixed_weights

    Steps:
        1. Distinguish weighting and date direction with unequal daily scores.
        2. Check that all cells remain and the explicit five-term weighted mean matches.
    """
    weighted = WeightedScoreBuilder().build(daily_scores[:5])
    expected = (2 + 4 * 0.9 + 8 * 0.81 + 16 * 0.729 + 32 * 0.6561) / 4.0951
    assert weighted.day == DAY
    assert len(weighted.frame) == 1_038_240
    assert weighted.frame.score.dtype == np.dtype("float64")
    assert value_at(weighted, 0) == pytest.approx(expected, rel=1e-15)
    assert value_at(weighted, 0.25) == pytest.approx(-2 * expected, rel=1e-15)
    assert weighted.frame.score.eq(0).sum() == len(weighted.frame) - 5


def test_coordinate_alignment_keeps_newest_order_without_mutating_inputs(daily_scores):
    """Match physical cells when input tables and indices have different orders.

    Args:
        daily_scores: Accepted consecutive daily grids.

    Returns:
        None. The newest order survives and each coordinate has the same mean.

    Examples:
        pytest tests/test_weighted_scores.py -k coordinate_alignment

    Steps:
        1. Match physical cells when input tables and indices have different orders.
        2. Check that the newest order survives and each coordinate has the same mean.
    """
    ordered = WeightedScoreBuilder().build(daily_scores[:5])
    newest = WaveCellScores(daily_scores[0].frame.iloc[::-1].copy(), DAY)
    older = WaveCellScores(
        daily_scores[2].frame.sample(frac=1, random_state=17), daily_scores[2].day
    )
    originals = newest.frame.copy(), older.frame.copy()
    window = (newest, daily_scores[1], older, *daily_scores[3:5])
    weighted = WeightedScoreBuilder().build(window)
    pd.testing.assert_frame_equal(
        weighted.frame[["longitude", "latitude"]],
        newest.frame[["longitude", "latitude"]].reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        weighted.frame.sort_values(["longitude", "latitude"]).reset_index(drop=True),
        ordered.frame.sort_values(["longitude", "latitude"]).reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(newest.frame, originals[0])
    pd.testing.assert_frame_equal(older.frame, originals[1])


def test_mixed_native_precision_is_promoted_for_arithmetic_only(daily_scores):
    """Retain float32 input values while calculating every product in float64.

    Args:
        daily_scores: Accepted float64 grids used for the remaining dates.

    Returns:
        None. Native float32 inputs stay unchanged and use their exact values.

    Examples:
        pytest tests/test_weighted_scores.py -k mixed_native_precision

    Steps:
        1. Retain float32 input values while calculating every product in float64.
        2. Check that native float32 inputs stay unchanged and use their exact values.
    """
    native_value = np.nextafter(np.float32(1), np.float32(2))
    previous = WaveCellScores(
        score_frame(native_value, dtype="float32"), DAY - timedelta(days=1)
    )
    original = previous.frame.copy()
    window = (daily_scores[0], previous, *daily_scores[2:5])
    weighted = WeightedScoreBuilder().build(window)
    expected = (
        2 + float(native_value) * 0.9 + 8 * 0.81 + 16 * 0.729 + 32 * 0.6561
    ) / 4.0951
    assert value_at(weighted, 0) == pytest.approx(expected, rel=1e-15)
    assert previous.frame.score.dtype == np.dtype("float32")
    pd.testing.assert_frame_equal(previous.frame, original)


def test_scores_are_not_clipped_rounded_or_thresholded(daily_scores):
    """Keep scores on both sides of one, at one, and below zero.

    Args:
        daily_scores: Accepted daily grids containing fixed threshold neighbors.

    Returns:
        None. The full score grid remains available to the next stage.

    Examples:
        pytest tests/test_weighted_scores.py -k not_clipped

    Steps:
        1. Keep scores on both sides of one, at one, and below zero.
        2. Check that the full score grid remains available to the next stage.
    """
    weighted = WeightedScoreBuilder().build(daily_scores[:5])
    assert 0 < value_at(weighted, 0.5) < 1
    assert value_at(weighted, 0.75) == 1
    assert value_at(weighted, 1) == 1.25
    assert value_at(weighted, 0.25) < 0


@pytest.mark.parametrize(
    "kind", ["list", "empty", "short", "long", "not-model", "none"]
)
def test_builder_requires_exactly_five_accepted_models(daily_scores, kind):
    """Reject malformed windows before any weighted calculation starts.

    Args:
        daily_scores: Accepted score grids for otherwise valid inputs.
        kind: Invalid window container, length or member.

    Returns:
        None. Invalid windows raise a clear wave-cell error.

    Examples:
        pytest tests/test_weighted_scores.py -k exactly_five

    Steps:
        1. Reject malformed windows before any weighted calculation starts.
        2. Check that invalid windows raise a clear wave-cell error.
    """
    windows = {
        "list": list(daily_scores[:5]),
        "empty": (),
        "short": daily_scores[:4],
        "long": daily_scores,
        "not-model": (daily_scores[0].frame, *daily_scores[1:5]),
        "none": None,
    }
    with pytest.raises(WaveCellError, match=r"five.*daily grids"):
        WeightedScoreBuilder().build(windows[kind])


@pytest.mark.parametrize("kind", ["repeated", "missing", "oldest-first", "swapped"])
def test_dates_must_be_consecutive_and_newest_first(daily_scores, kind):
    """Reject missing or misordered dates even when every grid is accepted.

    Args:
        daily_scores: Six accepted consecutive grids.
        kind: Repeated day, skipped day or wrong chronological order.

    Returns:
        None. Every malformed five-date window raises before writing output.

    Examples:
        pytest tests/test_weighted_scores.py -k dates_must

    Steps:
        1. Reject missing or misordered dates even when every grid is accepted.
        2. Check that every malformed five-date window raises before writing output.
    """
    windows = {
        "repeated": (*daily_scores[:4], daily_scores[3]),
        "missing": (*daily_scores[:4], daily_scores[5]),
        "oldest-first": tuple(reversed(daily_scores[:5])),
        "swapped": (
            daily_scores[0],
            daily_scores[2],
            daily_scores[1],
            *daily_scores[3:5],
        ),
    }
    with pytest.raises(WaveCellError, match=r"consecutive.*newest first"):
        WeightedScoreBuilder().build(windows[kind])


def test_builder_validates_only_the_finished_table(daily_scores, monkeypatch):
    """Avoid validating accepted daily grids again while checking output once.

    Args:
        daily_scores: Already accepted complete daily score models.
        monkeypatch: Fixture recording calls to the existing table schema.

    Returns:
        None. The only schema validation belongs to the newly calculated output.

    Examples:
        pytest tests/test_weighted_scores.py -k finished_table

    Steps:
        1. Avoid validating accepted daily grids again while checking output once.
        2. Check that the only schema validation belongs to the newly calculated output.
    """
    schema = WaveCellScores.schema
    original = schema.validate
    calls = []

    def counted(cls, frame, *args, **kwargs):
        """Record one schema boundary and preserve its actual validation.

        Args:
            cls: Score-table schema supplied by the class method.
            frame: Score frame being validated.
            *args: Existing positional validation options.
            **kwargs: Existing keyword validation options.

        Returns:
            The existing schema's accepted score frame.

        Examples:
            counted(schema, weighted_frame)

        Steps:
            1. Record one schema boundary and preserve its actual validation.
        """
        calls.append(frame)
        return original(frame, *args, **kwargs)

    monkeypatch.setattr(schema, "validate", classmethod(counted))
    weighted = WeightedScoreBuilder().build(daily_scores[:5])
    assert len(calls) == 1
    assert calls[0] is weighted.frame
    assert all(calls[0] is not item.frame for item in daily_scores)
