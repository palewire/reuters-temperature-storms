"""Check explicit initialization and exact daily continuation of state files."""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from reuters_temperature_storms.cell_scores import WaveCellScores, WaveCellScoreSource
from reuters_temperature_storms.cell_state import WaveCellState, WaveCellStateSource
from reuters_temperature_storms.cli import main
from reuters_temperature_storms.series import WaveSeries
from reuters_temperature_storms.state_run import StateRun, StateRunError

DAY = date(2024, 7, 1)
SERIES = WaveSeries("tasmax", "warm")


def _write_scores(root, count=3, start=DAY):
    """Write full global score grids with one continuing qualified cell.

    Args:
        root: Independent score directory.
        count: Number of consecutive daily files.
        start: First score date.

    Returns:
        Configured score source for the synthetic daily grids.

    Examples:
        source = _write_scores(tmp_path / "scores")

    Steps:
        1. Write full global score grids with one continuing qualified cell.
    """
    source = WaveCellScoreSource(str(root))
    longitude = np.tile(np.arange(-179.75, 180.25, 0.25), 721)
    latitude = np.repeat(np.arange(-90.0, 90.25, 0.25), 1440)
    score = np.zeros(len(longitude), dtype=float)
    score[5000] = 1.0
    frame = pd.DataFrame({"longitude": longitude, "latitude": latitude, "score": score})
    for offset in range(count):
        day = start + timedelta(days=offset)
        WaveCellScores(frame, day).to_parquet(source.path_for(day, SERIES), SERIES)
    return source


def test_seed_resume_and_continuation_keep_exact_preceding_state(tmp_path, monkeypatch):
    """Reuse a saved prefix and then continue from its final accepted state.

    Args:
        tmp_path: Independent score and state file roots.
        monkeypatch: Fixture observing score reads during reuse.

    Returns:
        None. Existing states avoid score reads and new days gain one day each.

    Examples:
        pytest tests/test_state_run.py -k seed_resume

    Steps:
        1. Reuse a saved prefix and then continue from its final accepted state.
        2. Check that existing states avoid score reads and new days gain one day each.
    """
    source = _write_scores(tmp_path / "scores")
    output = WaveCellStateSource(str(tmp_path / "state"))
    end = DAY + timedelta(days=2)
    with pytest.raises(StateRunError, match="exact preceding state"):
        StateRun(source, output, DAY, end, SERIES).run()
    assert not (tmp_path / "state").exists()
    first = StateRun(source, output, DAY, DAY, SERIES, seed=True).run()
    assert first[0].rows == 1
    assert not first[0].skipped
    observed = []
    original = WaveCellScoreSource.read_scores

    def observe(self, day, series):
        """Record each score date before invoking the real reader.

        Args:
            self: Configured score source.
            day: Requested date.
            series: Selected score field.

        Returns:
            Accepted complete score grid.

        Examples:
            Used while the runner resumes the saved prefix.

        Steps:
            1. Record each score date before invoking the real reader.
        """
        observed.append(day)
        return original(self, day, series)

    monkeypatch.setattr(WaveCellScoreSource, "read_scores", observe)
    results = StateRun(source, output, DAY, end, SERIES, seed=True).run()
    assert [result.skipped for result in results] == [True, False, False]
    assert observed == [DAY + timedelta(days=1), end]
    assert output.read_state(end, SERIES).frame["days"].tolist() == [3]
    assert StateRun(source, output, DAY, end, SERIES).check() == ()
    # A request starting later can continue by reading its exact D-1 output.
    source = _write_scores(tmp_path / "scores", count=4)
    tomorrow = end + timedelta(days=1)
    StateRun(source, output, tomorrow, tomorrow, SERIES).run()
    assert output.read_state(tomorrow, SERIES).frame["days"].tolist() == [4]


def test_missing_middle_state_requires_force_before_any_write(tmp_path):
    """Refuse an old saved suffix following a missing intermediate state.

    Args:
        tmp_path: Independent score and state roots.

    Returns:
        None. Ordinary resume preserves all remaining bytes; force rebuilds chain.

    Examples:
        pytest tests/test_state_run.py -k missing_middle

    Steps:
        1. Refuse an old saved suffix following a missing intermediate state.
        2. Check that ordinary resume preserves all remaining bytes; force rebuilds chain.
    """
    source = _write_scores(tmp_path / "scores")
    output = WaveCellStateSource(str(tmp_path / "state"))
    end = DAY + timedelta(days=2)
    StateRun(source, output, DAY, end, SERIES, seed=True).run()
    middle_path = Path(output.path_for(DAY + timedelta(days=1), SERIES))
    middle_path.unlink()
    first_path = Path(output.path_for(DAY, SERIES))
    final_path = Path(output.path_for(end, SERIES))
    prior = (first_path.read_bytes(), final_path.read_bytes())
    with pytest.raises(StateRunError, match="existing state follows a missing date"):
        StateRun(source, output, DAY, end, SERIES, seed=True).run()
    assert (first_path.read_bytes(), final_path.read_bytes()) == prior
    rebuilt = StateRun(
        source, output, DAY, end, SERIES, seed=True, force_update=True
    ).run()
    assert all(not result.skipped for result in rebuilt)
    assert output.read_state(end, SERIES).frame["days"].tolist() == [3]


def test_missing_scores_stop_without_skipping_dates_and_resume_retains_prefix(tmp_path):
    """Stop the dependent chain at a missing score file and keep its saved prefix.

    Args:
        tmp_path: Independent score and state roots.

    Returns:
        None. The missing date fails and no state is created for later dates.

    Examples:
        pytest tests/test_state_run.py -k missing_scores

    Steps:
        1. Stop the dependent chain at a missing score file and keep its saved prefix.
        2. Check that the missing date fails and no state is created for later dates.
    """
    source = _write_scores(tmp_path / "scores")
    middle = DAY + timedelta(days=1)
    Path(source.path_for(middle, SERIES)).unlink()
    output = WaveCellStateSource(str(tmp_path / "state"))
    end = DAY + timedelta(days=2)
    with pytest.raises(StateRunError, match=str(middle)):
        StateRun(source, output, DAY, end, SERIES, seed=True).run()
    assert Path(output.path_for(DAY, SERIES)).exists()
    assert not Path(output.path_for(middle, SERIES)).exists()
    assert not Path(output.path_for(end, SERIES)).exists()
    assert StateRun(source, output, DAY, end, SERIES).check() == (middle, end)


def test_corrupt_prefix_and_missing_predecessor_cannot_start_new_history(tmp_path):
    """Keep corrupted existing output and reject an undeclared fresh start.

    Args:
        tmp_path: Independent score and state roots.

    Returns:
        None. Invalid existing bytes remain unchanged and D-1 is required.

    Examples:
        pytest tests/test_state_run.py -k corrupt_prefix

    Steps:
        1. Keep corrupted existing output and reject an undeclared fresh start.
        2. Check that invalid existing bytes remain unchanged and D-1 is required.
    """
    source = _write_scores(tmp_path / "scores", count=1)
    output = WaveCellStateSource(str(tmp_path / "state"))
    StateRun(source, output, DAY, DAY, SERIES, seed=True).run()
    path = Path(output.path_for(DAY, SERIES))
    path.write_bytes(b"bad state")
    with pytest.raises(StateRunError, match=str(DAY)):
        StateRun(source, output, DAY, DAY, SERIES, seed=True).run()
    with pytest.raises(StateRunError, match="invalid wave-state dates"):
        StateRun(source, output, DAY, DAY, SERIES).check()
    assert path.read_bytes() == b"bad state"
    with pytest.raises(StateRunError, match="exact preceding state"):
        StateRun(source, output, DAY, DAY, SERIES, force_update=True).run()
    assert path.read_bytes() == b"bad state"


def test_check_and_reuse_do_not_read_calculated_scores(tmp_path, monkeypatch):
    """Check and reuse accepted complete states without reopening score files.

    Args:
        tmp_path: Independent score and state roots.
        monkeypatch: Fixture preventing any calculated-score read.

    Returns:
        None. All existing files can be checked and reused without source access.

    Examples:
        pytest tests/test_state_run.py -k do_not_read

    Steps:
        1. Check and reuse accepted complete states without reopening score files.
        2. Check that all existing files can be checked and reused without source access.
    """
    source = _write_scores(tmp_path / "scores", count=1)
    output = WaveCellStateSource(str(tmp_path / "state"))
    StateRun(source, output, DAY, DAY, SERIES, seed=True).run()

    def reject_read(*args, **kwargs):
        """Fail on any unexpected score read.

        Args:
            *args: Unexpected positional reader inputs.
            **kwargs: Unexpected reader options.

        Returns:
            Never returns; raises AssertionError.

        Examples:
            Installed while only accepted state files are needed.

        Steps:
            1. Fail on any unexpected score read.
        """
        raise AssertionError("scores must remain unread during saved-state reuse")

    monkeypatch.setattr(WaveCellScoreSource, "read_scores", reject_read)
    run = StateRun(source, output, DAY, DAY, SERIES)
    assert run.check() == ()
    assert run.run()[0].skipped


def test_cli_explicit_seed_build_and_check(tmp_path):
    """Expose initialization and existing-state checking through the real CLI.

    Args:
        tmp_path: Independent score and state roots.

    Returns:
        None. The CLI creates only daily state outputs and rejects mixed options.

    Examples:
        pytest tests/test_state_run.py -k cli_explicit

    Steps:
        1. Expose initialization and existing-state checking through the real CLI.
        2. Check that the CLI creates only daily state outputs and rejects mixed options.
    """
    source = _write_scores(tmp_path / "scores", count=1)
    output = tmp_path / "state"
    args = [
        "state",
        "--wave-scores",
        source.location,
        "--output",
        str(output),
        "--start-date",
        str(DAY),
        "--end-date",
        str(DAY),
    ]
    runner = CliRunner()
    absent = runner.invoke(main, args)
    assert absent.exit_code == 1
    assert "exact preceding state" in absent.output
    built = runner.invoke(main, [*args, "--seed"])
    assert built.exit_code == 0, built.output
    assert "Completed 1 daily wave-state files" in built.output
    checked = runner.invoke(main, [*args, "--check-only"])
    assert checked.exit_code == 0, checked.output
    invalid = runner.invoke(main, [*args, "--check-only", "--seed"])
    assert invalid.exit_code == 1
    assert "cannot be combined" in invalid.output
    duplicate = runner.invoke(main, [*args, "--output", source.location, "--seed"])
    assert duplicate.exit_code == 1
    assert "different directories" in duplicate.output
    assert not (output / "_COMPLETE.json").exists()
    assert len(list(output.rglob("*.parquet"))) == 1


def test_state_replacement_failure_preserves_previous_file(tmp_path, monkeypatch):
    """Keep accepted state bytes when replacement of a local file fails.

    Args:
        tmp_path: Independent output root.
        monkeypatch: Fixture interrupting final replacement.

    Returns:
        None. Prior state survives and the temporary sibling is removed.

    Examples:
        pytest tests/test_state_run.py -k replacement_failure

    Steps:
        1. Keep accepted state bytes when replacement of a local file fails.
        2. Check that prior state survives and the temporary sibling is removed.
    """
    state = WaveCellState.empty(DAY)
    target = tmp_path / "state.parquet"
    state.to_parquet(target)
    prior = target.read_bytes()

    def reject_replace(self, destination):
        """Raise after a complete temporary state file has been written.

        Args:
            self: Temporary sibling path.
            destination: Existing accepted state path.

        Returns:
            Never returns; raises OSError.

        Examples:
            Used for a failed state replacement.

        Steps:
            1. Raise after a complete temporary state file has been written.
        """
        assert Path(destination) == target
        assert WaveCellState.read_parquet(self, DAY).frame.empty
        raise OSError("state replacement interrupted")

    monkeypatch.setattr(Path, "replace", reject_replace)
    with pytest.raises(OSError, match="interrupted"):
        state.to_parquet(target)
    assert target.read_bytes() == prior
    assert list(tmp_path.glob(".*.tmp")) == []


@pytest.mark.parametrize(
    "mutation",
    [
        "date_order",
        "seed_type",
        "force_type",
        "source_type",
        "output_type",
        "root_alias",
    ],
)
def test_invalid_request_settings_fail_before_any_write(tmp_path, mutation):
    """Reject unsafe typed settings and equivalent input/output roots early.

    Args:
        tmp_path: Independent roots used without needing score files.
        mutation: Invalid request setting under test.

    Returns:
        None. The runner writes nothing when its request settings are invalid.

    Examples:
        pytest tests/test_state_run.py -k invalid_request

    Steps:
        1. Reject unsafe typed settings and equivalent input/output roots early.
        2. Check that the runner writes nothing when its request settings are invalid.
    """
    source = WaveCellScoreSource(str(tmp_path / "scores"))
    output = WaveCellStateSource(str(tmp_path / "state"))
    start, end, seed, force = DAY, DAY, True, False
    if mutation == "date_order":
        end = DAY - timedelta(days=1)
    elif mutation == "seed_type":
        seed = "yes"
    elif mutation == "force_type":
        force = "yes"
    elif mutation == "source_type":
        source = WaveCellStateSource(source.location)
    elif mutation == "output_type":
        output = WaveCellScoreSource(output.location)
    else:
        output = WaveCellStateSource((tmp_path / "scores").as_uri())
    with pytest.raises(StateRunError):
        StateRun(source, output, start, end, SERIES, seed, force).run()
    assert not (tmp_path / "state").exists()


def test_four_day_seed_lead_in_matches_explicit_history_and_resume(tmp_path):
    """Count preceding days but save and resume only the requested archive.

    Args:
        tmp_path: Separate synthetic score and state roots.

    Returns:
        None. January state equals explicit December processing and reaches maturity.

    Examples:
        pytest tests/test_state_run.py -k four_day_seed

    Steps:
        1. Count preceding days but save and resume only the requested archive.
        2. Check that january state equals explicit December processing and reaches maturity.
    """
    start = date(1960, 1, 1)
    earlier = start - timedelta(days=4)
    source = _write_scores(tmp_path / "scores", count=6, start=earlier)
    output = WaveCellStateSource(str(tmp_path / "retained"))
    explicit = WaveCellStateSource(str(tmp_path / "explicit"))
    StateRun(source, explicit, earlier, start, SERIES, seed=True).run()
    first = StateRun(
        source, output, start, start, SERIES, seed=True, seed_lead_in=True
    ).run()
    assert len(first) == 1
    assert set(output.list_files(SERIES)) == {"1960-01-01.parquet"}
    initial = output.read_state(start, SERIES)
    pd.testing.assert_frame_equal(
        initial.frame, explicit.read_state(start, SERIES).frame
    )
    assert initial.frame["days"].tolist() == [5]
    assert initial.frame["start_date"].tolist() == [pd.Timestamp(earlier)]
    assert not initial.to_cells(closing_radius=0).frame.empty
    end = start + timedelta(days=1)
    results = StateRun(
        source, output, start, end, SERIES, seed=True, seed_lead_in=True
    ).run()
    assert [result.skipped for result in results] == [True, False]
    assert output.read_state(end, SERIES).frame["days"].tolist() == [6]


def test_seed_lead_in_rejects_missing_day_and_old_seed_before_writes(tmp_path):
    """Require all preceding inputs and reject reuse of a different initial state.

    Args:
        tmp_path: Separate synthetic score and state roots.

    Returns:
        None. Missing inputs and stale saved history cannot silently change the start.

    Examples:
        pytest tests/test_state_run.py -k rejects_missing_day

    Steps:
        1. Require all preceding inputs and reject reuse of a different initial state.
        2. Check that missing inputs and stale saved history cannot silently change the start.
    """
    source = _write_scores(tmp_path / "scores", count=5, start=DAY - timedelta(days=4))
    output = WaveCellStateSource(str(tmp_path / "state"))
    StateRun(source, output, DAY, DAY, SERIES, seed=True).run()
    path = Path(output.path_for(DAY, SERIES))
    before = path.read_bytes()
    with pytest.raises(StateRunError, match="saved first state differs"):
        StateRun(source, output, DAY, DAY, SERIES, seed=True, seed_lead_in=True).run()
    assert path.read_bytes() == before
    StateRun(
        source,
        output,
        DAY,
        DAY,
        SERIES,
        seed=True,
        seed_lead_in=True,
        force_update=True,
    ).run()
    assert output.read_state(DAY, SERIES).frame["days"].tolist() == [5]
    missing = DAY - timedelta(days=2)
    Path(source.path_for(missing, SERIES)).unlink()
    empty = WaveCellStateSource(str(tmp_path / "empty"))
    with pytest.raises(StateRunError, match=f"seed lead-in {missing}"):
        StateRun(source, empty, DAY, DAY, SERIES, seed=True, seed_lead_in=True).run()
    assert not (tmp_path / "empty").exists()


@pytest.mark.parametrize(
    ("start", "seed", "lead_in", "message"),
    [
        (DAY, False, True, "requires --seed"),
        (date.min, True, True, "four preceding"),
        (DAY, True, 4, "must be boolean"),
    ],
)
def test_seed_lead_in_invalid_requests(tmp_path, start, seed, lead_in, message):
    """Reject unsupported initialization choices before reading scores.

    Args:
        tmp_path: Separate input and output roots.
        start: Requested start date.
        seed: Explicit initialization choice.
        lead_in: Lead-in setting to validate.
        message: Expected diagnostic text.

    Returns:
        None. Invalid settings raise StateRunError.

    Examples:
        pytest tests/test_state_run.py -k invalid_requests

    Steps:
        1. Reject unsupported initialization choices before reading scores.
        2. Check that invalid settings raise StateRunError.
    """
    run = StateRun(
        WaveCellScoreSource(str(tmp_path / "scores")),
        WaveCellStateSource(str(tmp_path / "state")),
        start,
        DAY,
        SERIES,
        seed=seed,
        seed_lead_in=lead_in,
    )
    with pytest.raises(StateRunError, match=message):
        run.validate()


def test_cli_seed_lead_in_preserves_requested_start(tmp_path):
    """Expose the narrow lead-in while retaining January output dates.

    Args:
        tmp_path: Separate synthetic score and state roots.

    Returns:
        None. CLI initializes five qualifying days and rejects unseeded lead-in.

    Examples:
        pytest tests/test_state_run.py -k cli_seed_lead

    Steps:
        1. Expose the narrow lead-in while retaining January output dates.
        2. Check that cLI initializes five qualifying days and rejects unseeded lead-in.
    """
    start = date(1960, 1, 1)
    source = _write_scores(tmp_path / "scores", count=5, start=date(1959, 12, 28))
    args = [
        "state",
        "--wave-scores",
        source.location,
        "--output",
        str(tmp_path / "state"),
        "--start-date",
        str(start),
        "--end-date",
        str(start),
        "--seed-lead-in",
    ]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 1
    assert "requires --seed" in result.output
    result = CliRunner().invoke(main, [*args, "--seed"])
    assert result.exit_code == 0, result.output
    output = WaveCellStateSource(str(tmp_path / "state"))
    assert set(output.list_files(SERIES)) == {"1960-01-01.parquet"}
    assert output.read_state(start, SERIES).frame["days"].tolist() == [5]
