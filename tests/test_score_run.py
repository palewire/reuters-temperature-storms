"""Check reusable weighted-score outputs and rolling date-block input reads."""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reuters_temperature_storms.cell_scores import WaveCellScores, WaveCellScoreSource
from reuters_temperature_storms.score_run import ScoreRun, ScoreRunError
from reuters_temperature_storms.series import WaveSeries

DAY = date(2024, 7, 1)
SERIES = WaveSeries("tasmax", "warm")


@pytest.fixture
def daily_scores(tmp_path):
    """Save six complete grids covering two output dates and their lead-in.

    Args:
        tmp_path: Independent temporary daily-input root.

    Returns:
        Complete daily-score source with reversed order on alternating dates.

    Examples:
        Used by the two-day weighted-score run tests.

    Steps:
        1. Save six complete grids covering two output dates and their lead-in.
    """
    source = WaveCellScoreSource(str(tmp_path / "daily"))
    longitude = np.tile(np.arange(-179.75, 180.25, 0.25), 721)
    latitude = np.repeat(np.arange(-90.0, 90.25, 0.25), 1440)
    for offset in range(-4, 2):
        day = DAY + timedelta(days=offset)
        frame = pd.DataFrame(
            {
                "longitude": longitude,
                "latitude": latitude,
                "score": np.full(len(longitude), offset + 5.0),
            }
        )
        if offset % 2:
            frame = frame.iloc[::-1].reset_index(drop=True)
        WaveCellScores(frame, day).to_parquet(source.path_for(day, SERIES), SERIES)
    return source


def test_overlapping_windows_read_each_input_once_and_reuse_outputs(
    tmp_path, monkeypatch, daily_scores
):
    """Read six input dates for two outputs, then reuse without input access.

    Args:
        tmp_path: Independent weighted-score output root.
        monkeypatch: Fixture observing input reads and guarding later reuse.
        daily_scores: Six complete input grids with differing row order.

    Returns:
        None. Outputs feed the score reader, and check/reuse avoid input files.

    Examples:
        pytest tests/test_score_run.py -k overlapping

    Steps:
        1. Read six input dates for two outputs, then reuse without input access.
        2. Check that outputs feed the score reader, and check/reuse avoid input files.
    """
    output = WaveCellScoreSource(str(tmp_path / "weighted"))
    tomorrow = DAY + timedelta(days=1)
    run = ScoreRun(daily_scores, output, DAY, tomorrow, SERIES, max_workers=1)
    observed = []
    original = WaveCellScoreSource.read_scores

    def observe(self, day, series):
        """Record input reads while allowing the normal complete-grid reader.

        Args:
            self: Configured score source.
            day: Requested calendar date.
            series: Requested warm/cold maximum/minimum selection.

        Returns:
            Accepted complete score grid.

        Examples:
            Installed while two overlapping windows are built.

        Steps:
            1. Record input reads while allowing the normal complete-grid reader.
        """
        if self.location == daily_scores.location:
            observed.append(day)
        return original(self, day, series)

    monkeypatch.setattr(WaveCellScoreSource, "read_scores", observe)
    progress = []
    results = run.run(progress=progress.append)
    assert [result.day for result in results] == [DAY, tomorrow]
    assert progress == list(results)
    assert len(observed) == len(set(observed)) == 6
    weights = np.power(0.9, np.arange(5))
    expected = np.dot(np.arange(5.0, 0.0, -1.0), weights) / weights.sum()
    actual = output.read_scores(DAY, SERIES)
    np.testing.assert_allclose(actual.frame["score"], expected, rtol=1e-15)
    assert all(result.rows == 1038240 and result.bytes > 0 for result in results)
    observed.clear()
    assert run.check() == ()
    assert all(result.skipped for result in run.run())
    assert observed == []


def test_missing_leadin_retains_later_independent_output(tmp_path, daily_scores):
    """Fail a missing fifth input while keeping a later complete window.

    Args:
        tmp_path: Independent weighted-score output root.
        daily_scores: Six complete inputs before removing the oldest date.

    Returns:
        None. No partial-window average is saved and the next output survives.

    Examples:
        pytest tests/test_score_run.py -k missing_leadin

    Steps:
        1. Fail a missing fifth input while keeping a later complete window.
        2. Check that no partial-window average is saved and the next output survives.
    """
    missing = DAY - timedelta(days=4)
    Path(daily_scores.path_for(missing, SERIES)).unlink()
    output = WaveCellScoreSource(str(tmp_path / "weighted"))
    tomorrow = DAY + timedelta(days=1)
    run = ScoreRun(daily_scores, output, DAY, tomorrow, SERIES, max_workers=1)
    progress = []
    with pytest.raises(ScoreRunError, match=str(missing)):
        run.run(progress=progress.append)
    assert [result.day for result in progress] == [tomorrow]
    assert not Path(output.path_for(DAY, SERIES)).exists()
    assert Path(output.path_for(tomorrow, SERIES)).exists()
    assert run.check() == (DAY,)


def test_independent_blocks_complete_in_real_process_workers(tmp_path, daily_scores):
    """Build distinct output dates in separate worker processes.

    Args:
        tmp_path: Independent weighted-score output root.
        daily_scores: Complete inputs covering two one-day worker blocks.

    Returns:
        None. Both spawned workers save readable complete score files.

    Examples:
        pytest tests/test_score_run.py -k real_process

    Steps:
        1. Build distinct output dates in separate worker processes.
        2. Check that both spawned workers save readable complete score files.
    """
    output = WaveCellScoreSource(str(tmp_path / "weighted"))
    tomorrow = DAY + timedelta(days=1)
    run = ScoreRun(
        daily_scores, output, DAY, tomorrow, SERIES, max_workers=2, block_days=1
    )
    results = run.run()
    assert [result.day for result in results] == [DAY, tomorrow]
    assert all(not result.skipped for result in results)
    assert run.check() == ()


def test_corrupt_output_is_preserved_until_force_update(tmp_path, daily_scores):
    """Reject unreadable existing output and require an explicit replacement.

    Args:
        tmp_path: Independent weighted-score output root.
        daily_scores: Complete daily score input source.

    Returns:
        None. Ordinary build/check preserve corrupt bytes; force rebuilds them.

    Examples:
        pytest tests/test_score_run.py -k corrupt_output

    Steps:
        1. Reject unreadable existing output and require an explicit replacement.
        2. Check that ordinary build/check preserve corrupt bytes; force rebuilds them.
    """
    output = WaveCellScoreSource(str(tmp_path / "weighted"))
    path = Path(output.path_for(DAY, SERIES))
    path.parent.mkdir(parents=True)
    path.write_bytes(b"bad score file")
    run = ScoreRun(daily_scores, output, DAY, DAY, SERIES, max_workers=1)
    with pytest.raises(ScoreRunError, match=str(DAY)):
        run.run()
    with pytest.raises(ScoreRunError, match="invalid weighted-score dates"):
        run.check()
    assert path.read_bytes() == b"bad score file"
    results = ScoreRun(
        daily_scores, output, DAY, DAY, SERIES, max_workers=1, force_update=True
    ).run()
    assert not results[0].skipped
    assert run.check() == ()


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"start_date": datetime(2024, 7, 1, tzinfo=UTC)}, "ordered calendar dates"),
        ({"end_date": DAY - timedelta(days=1)}, "ordered calendar dates"),
        ({"start_date": date.min}, "four preceding"),
        ({"max_workers": True}, "positive integer"),
        ({"max_workers": 0}, "positive integer"),
        ({"block_days": 0}, "positive integer"),
        ({"block_days": True}, "positive integer"),
        ({"force_update": 1}, "boolean"),
        ({"series": WaveSeries("tasmax", "invalid")}, "warm/cold"),
    ],
)
def test_invalid_requests_fail_without_output_reads(tmp_path, settings, message):
    """Reject unsafe run settings before any daily source file is opened.

    Args:
        tmp_path: Independent absent source/output roots.
        settings: Invalid request fields overriding an otherwise valid request.
        message: Expected plain-language validation detail.

    Returns:
        None. Every invalid request fails before creating an output directory.

    Examples:
        pytest tests/test_score_run.py -k invalid_requests

    Steps:
        1. Reject unsafe run settings before any daily source file is opened.
        2. Check that every invalid request fails before creating an output directory.
    """
    request = {
        "source": WaveCellScoreSource(str(tmp_path / "daily")),
        "destination": WaveCellScoreSource(str(tmp_path / "weighted")),
        "start_date": DAY,
        "end_date": DAY,
        "series": SERIES,
    }
    with pytest.raises(ScoreRunError, match=message):
        ScoreRun(**(request | settings)).run()
    assert not (tmp_path / "weighted").exists()


def test_block_requests_have_four_day_overlap_and_separate_output_roots(tmp_path):
    """Describe output blocks without opening files and reject root overwrite.

    Args:
        tmp_path: Independent absent source/output roots.

    Returns:
        None. Output dates never overlap and each block reads its own lead-in.

    Examples:
        pytest tests/test_score_run.py -k block_requests

    Steps:
        1. Describe output blocks without opening files and reject root overwrite.
        2. Check that output dates never overlap and each block reads its own lead-in.
    """
    source = WaveCellScoreSource(str(tmp_path / "daily"))
    output = WaveCellScoreSource(str(tmp_path / "weighted"))
    end = DAY + timedelta(days=4)
    run = ScoreRun(source, output, DAY, end, SERIES, block_days=2)
    tasks = list(run.tasks())
    assert [(task.day, task.end_date) for task in tasks] == [
        (DAY, DAY + timedelta(days=1)),
        (DAY + timedelta(days=2), DAY + timedelta(days=3)),
        (end, end),
    ]
    with pytest.raises(ScoreRunError, match="different directories"):
        ScoreRun(source, source, DAY, end, SERIES).validate()
