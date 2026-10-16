"""Check chronological qualifying, grace, recovery and ending transitions."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from reuters_temperature_storms.cell_scores import WaveCellScores
from reuters_temperature_storms.cell_state import WaveCellState
from reuters_temperature_storms.cells import WaveCellError
from reuters_temperature_storms.state_builder import WaveCellStateBuilder
from tests.test_cell_scores import score_frame

DAY = date(2024, 7, 1)


def predecessor(rows=(), *, day=DAY - timedelta(days=1)):
    """Construct valid yesterday histories from explicitly supplied row values.

    Args:
        rows: Longitude, latitude, start date, end date, qualifying count and grace.
        day: Date represented by the supplied histories.

    Returns:
        Once-validated six-column cell state.

    Examples:
        state = predecessor(((0, 0, DAY - timedelta(days=6), None, 5, 0),))

    Steps:
        1. Construct valid yesterday histories from explicitly supplied row values.
    """
    return WaveCellState(
        pd.DataFrame(
            {
                "longitude": pd.Series([row[0] for row in rows], dtype="float64"),
                "latitude": pd.Series([row[1] for row in rows], dtype="float64"),
                "start_date": pd.Series(
                    [row[2] for row in rows], dtype="datetime64[ns]"
                ),
                "end_date": pd.Series([row[3] for row in rows], dtype="datetime64[ns]"),
                "days": pd.Series([row[4] for row in rows], dtype="int64"),
                "grace_days": pd.Series([row[5] for row in rows], dtype="int64"),
            }
        ),
        day,
    )


def keyed(state):
    """Index state coordinates for order-independent expected-value assertions.

    Args:
        state: Validated daily state returned by a builder.

    Returns:
        A sorted DataFrame with coordinates as its index.

    Examples:
        rows = keyed(state)

    Steps:
        1. Index state coordinates for order-independent expected-value assertions.
    """
    return state.frame.set_index(["longitude", "latitude"]).sort_index()


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_explicit_seed_uses_the_inclusive_native_score_threshold(dtype):
    """Qualify exact one and above while preserving a value immediately below one.

    Args:
        dtype: Native floating-point score precision.

    Returns:
        None. Only scores at or above one start one-day histories.

    Examples:
        pytest tests/test_state_builder.py -k inclusive_native

    Steps:
        1. Qualify exact one and above while preserving a value immediately below one.
        2. Check that only scores at or above one start one-day histories.
    """
    native = np.dtype(dtype)
    below = np.nextafter(native.type(1), native.type(0))
    above = np.nextafter(native.type(1), native.type(2))
    frame = score_frame(((0, 0, below), (0.25, 0, 1), (0.5, 0, above)), dtype=dtype)
    scores = WaveCellScores(frame, DAY)
    state = WaveCellStateBuilder(scores, seed=True).build()
    assert isinstance(state, WaveCellState)
    assert state.day == DAY
    rows = keyed(state)
    assert list(rows.index) == [(0.25, 0), (0.5, 0)]
    assert rows.days.eq(1).all()
    assert rows.grace_days.eq(0).all()
    assert rows.start_date.eq(pd.Timestamp(DAY)).all()
    assert rows.end_date.isna().all()
    assert scores.frame.score.dtype == native
    assert scores.frame is frame


def test_all_nonqualifying_seed_and_empty_predecessor_create_typed_empty_state():
    """A complete quiet score grid yields empty state without pretending missing input.

    Args:
        None.

    Returns:
        None. Both explicit seeding and continuation retain the typed empty table.

    Examples:
        pytest tests/test_state_builder.py -k typed_empty_state

    Steps:
        1. A complete quiet score grid yields empty state without pretending missing input.
        2. Check that both explicit seeding and continuation retain the typed empty table.
    """
    scores = WaveCellScores(score_frame(), DAY)
    seeded = WaveCellStateBuilder(scores, seed=True).build()
    continued = WaveCellStateBuilder(scores, predecessor=predecessor()).build()
    for state in (seeded, continued):
        assert state.frame.empty
        pd.testing.assert_frame_equal(state.frame, WaveCellState.empty(DAY).frame)


def test_one_update_handles_continuing_recovery_grace_ending_rebirth_and_new_rows():
    """Apply every transition without changing prior state or carrying older endings.

    Args:
        None.

    Returns:
        None. Each coordinate has its independently expected history and counters.

    Examples:
        pytest tests/test_state_builder.py -k every_transition

    Steps:
        1. Apply every transition without changing prior state or carrying older endings.
        2. Check that each coordinate has its independently expected history and counters.
    """
    start = DAY - timedelta(days=7)
    yesterday = predecessor(
        (
            (0, 0, start, None, 5, 0),
            (0.25, 0, start, None, 5, 1),
            (0.5, 0, start, None, 5, 0),
            (0.75, 0, start, None, 5, 1),
            (1, 0, start, DAY - timedelta(days=3), 5, 0),
            (1.25, 0, start, DAY - timedelta(days=3), 5, 0),
        )
    )
    original = yesterday.frame.copy(deep=True)
    scores = WaveCellScores(
        score_frame(((0, 0, 1), (0.25, 0, 1.25), (1, 0, 1), (1.5, 0, 2))), DAY
    )
    today = keyed(WaveCellStateBuilder(scores, predecessor=yesterday).build())
    assert list(today.index) == [
        (0, 0),
        (0.25, 0),
        (0.5, 0),
        (0.75, 0),
        (1, 0),
        (1.5, 0),
    ]
    for longitude in (0, 0.25):
        row = today.loc[(longitude, 0)]
        assert row.days == 6
        assert row.grace_days == 0
        assert row.start_date == pd.Timestamp(start)
        assert pd.isna(row.end_date)
    grace = today.loc[(0.5, 0)]
    assert grace.days == 5
    assert grace.grace_days == 1
    assert pd.isna(grace.end_date)
    ending = today.loc[(0.75, 0)]
    assert ending.days == 5
    assert ending.grace_days == 0
    assert ending.end_date == pd.Timestamp(DAY - timedelta(days=2))
    for longitude in (1, 1.5):
        row = today.loc[(longitude, 0)]
        assert row.days == 1
        assert row.grace_days == 0
        assert row.start_date == pd.Timestamp(DAY)
        assert pd.isna(row.end_date)
    pd.testing.assert_frame_equal(yesterday.frame, original)


def test_second_missed_day_ends_then_later_qualification_begins_a_fresh_history():
    """Keep one grace day, end on the next miss, and restart after the ending.

    Args:
        None.

    Returns:
        None. End date is backdated and a returning cell has a new one-day start.

    Examples:
        pytest tests/test_state_builder.py -k second_missed

    Steps:
        1. Keep one grace day, end on the next miss, and restart after the ending.
        2. Check that end date is backdated and a returning cell has a new one-day start.
    """
    active = predecessor(((180, 0, DAY - timedelta(days=6), None, 5, 0),))
    quiet = score_frame()
    first = WaveCellStateBuilder(WaveCellScores(quiet, DAY), predecessor=active).build()
    assert first.frame.iloc[0].grace_days == 1
    second_day = DAY + timedelta(days=1)
    second = WaveCellStateBuilder(
        WaveCellScores(quiet, second_day), predecessor=first
    ).build()
    assert second.frame.iloc[0].end_date == pd.Timestamp(DAY - timedelta(days=1))
    assert second.frame.iloc[0].grace_days == 0
    third_day = DAY + timedelta(days=2)
    third = WaveCellStateBuilder(
        WaveCellScores(score_frame(((180, 0, 1),)), third_day), predecessor=second
    ).build()
    row = third.frame.iloc[0]
    assert row.longitude == 180
    assert row.days == 1
    assert row.start_date == pd.Timestamp(third_day)
    assert pd.isna(row.end_date)


def test_recovering_after_grace_counts_only_qualifying_days():
    """Recovery retains the original start while counting no qualifying day for grace.

    Args:
        None.

    Returns:
        None. Seven calendar days can contain six qualifying days after recovery.

    Examples:
        pytest tests/test_state_builder.py -k recovering_after

    Steps:
        1. Recovery retains the original start while counting no qualifying day for grace.
        2. Check that seven calendar days can contain six qualifying days after recovery.
    """
    start = DAY - timedelta(days=6)
    active = predecessor(((0, 0, start, None, 5, 1),))
    today = WaveCellStateBuilder(
        WaveCellScores(score_frame(((0, 0, 1),)), DAY), predecessor=active
    ).build()
    row = today.frame.iloc[0]
    assert row.days == 6
    assert row.start_date == pd.Timestamp(start)
    assert row.grace_days == 0


def test_missing_predecessor_requires_an_explicit_seed_and_seed_rejects_predecessor():
    """Avoid silently starting a second chain when yesterday's state is absent.

    Args:
        None.

    Returns:
        None. Ambiguous starts and simultaneous seed/continuation are rejected.

    Examples:
        pytest tests/test_state_builder.py -k missing_predecessor

    Steps:
        1. Avoid silently starting a second chain when yesterday's state is absent.
        2. Check that ambiguous starts and simultaneous seed/continuation are rejected.
    """
    scores = WaveCellScores(score_frame(), DAY)
    with pytest.raises(WaveCellError):
        WaveCellStateBuilder(scores).build()
    with pytest.raises(WaveCellError):
        WaveCellStateBuilder(scores, predecessor=predecessor(), seed=True).build()


@pytest.mark.parametrize("offset", [-2, 0, 1])
def test_predecessor_must_be_the_exact_previous_calendar_date(offset):
    """Reject older, same-day and future state rather than jumping dates.

    Args:
        offset: Invalid predecessor offset relative to the score observation.

    Returns:
        None. The builder rejects state that is not for yesterday.

    Examples:
        pytest tests/test_state_builder.py -k previous_calendar_date

    Steps:
        1. Reject older, same-day and future state rather than jumping dates.
        2. Check that the builder rejects state that is not for yesterday.
    """
    scores = WaveCellScores(score_frame(), DAY)
    with pytest.raises(WaveCellError):
        WaveCellStateBuilder(
            scores, predecessor=predecessor(day=DAY + timedelta(days=offset))
        ).build()


def test_input_row_order_does_not_change_physical_state_histories():
    """Compare history values after reversing both scores and predecessor rows.

    Args:
        None.

    Returns:
        None. Physical cell histories and counters match after unique coordinates.

    Examples:
        pytest tests/test_state_builder.py -k input_row_order

    Steps:
        1. Compare history values after reversing both scores and predecessor rows.
        2. Check that physical cell histories and counters match after unique coordinates.
    """
    start = DAY - timedelta(days=7)
    old = predecessor(((180, 0, start, None, 5, 0), (0, 0, start, None, 5, 1)))
    frame = score_frame(((180, 0, 1), (0, 0, 1), (-90, 90, 1)))
    forward = WaveCellStateBuilder(WaveCellScores(frame, DAY), predecessor=old).build()
    reverse = WaveCellStateBuilder(
        WaveCellScores(frame.iloc[::-1], DAY),
        predecessor=WaveCellState(old.frame.iloc[::-1], old.day),
    ).build()
    pd.testing.assert_frame_equal(keyed(forward), keyed(reverse))


@pytest.mark.parametrize("seed", [0, 1, None, "yes"])
def test_seed_selection_requires_a_boolean(seed):
    """Only an explicit boolean selects whether a new chain starts.

    Args:
        seed: A false-like or true-like nonboolean value.

    Returns:
        None. Ambiguous seed settings cannot discard predecessor state.

    Examples:
        pytest tests/test_state_builder.py -k requires_a_boolean

    Steps:
        1. Only an explicit boolean selects whether a new chain starts.
        2. Check that ambiguous seed settings cannot discard predecessor state.
    """
    scores = WaveCellScores(score_frame(), DAY)
    with pytest.raises(WaveCellError):
        WaveCellStateBuilder(scores, seed=seed)


def test_continuing_grace_ending_and_new_groups_keep_their_established_order():
    """Retain predecessor order within existing groups and score order for births.

    Args:
        None.

    Returns:
        None. The four transition groups appear in their explicit calculation order.

    Examples:
        pytest tests/test_state_builder.py -k groups_keep

    Steps:
        1. Retain predecessor order within existing groups and score order for births.
        2. Check that the four transition groups appear in their explicit calculation order.
    """
    start = DAY - timedelta(days=7)
    old = predecessor(
        (
            (1, 0, start, None, 5, 1),
            (2, 0, start, None, 5, 0),
            (3, 0, start, None, 5, 0),
            (4, 0, start, None, 5, 1),
            (5, 0, start, None, 5, 0),
        )
    )
    frame = score_frame(((3, 0, 1), (4, 0, 1), (6, 0, 1)))
    state = WaveCellStateBuilder(WaveCellScores(frame, DAY), predecessor=old).build()
    assert list(state.frame.longitude) == [3, 4, 2, 5, 1, 6]
    assert list(state.frame.grace_days) == [0, 0, 1, 1, 0, 0]
    assert state.frame.end_date.notna().tolist() == [
        False,
        False,
        False,
        False,
        True,
        False,
    ]
