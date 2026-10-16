"""Check weighted-score command settings, progress and saved-output checks."""

from datetime import date
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from reuters_temperature_storms import cli
from reuters_temperature_storms.cli import main


def score_arguments(input_root, output_root):
    """Select one output date and independent daily-input and score roots.

    Args:
        input_root: Daily score input location.
        output_root: Five-day weighted score output location.

    Returns:
        Command arguments requesting one complete output date.

    Examples:
        args = score_arguments("daily", "weighted")

    Steps:
        1. Select one output date and independent daily-input and score roots.
    """
    return [
        "scores",
        "--daily-scores",
        str(input_root),
        "--output",
        str(output_root),
        "--start-date",
        "2024-07-05",
        "--end-date",
        "2024-07-05",
    ]


@pytest.mark.parametrize(
    "series", ["tasmax/warm", "tasmax/cold", "tasmin/warm", "tasmin/cold"]
)
def test_scores_cli_passes_selected_series_and_worker_settings(
    tmp_path, monkeypatch, series
):
    """Pass command settings unchanged and print one completed-file status.

    Args:
        tmp_path: Independent input and output roots.
        monkeypatch: Fixture substituting a small runner that records settings.
        series: Each supported maximum/minimum and warm/cold selection.

    Returns:
        None. The selected series, date block and access profile reach the runner.

    Examples:
        pytest tests/test_score_cli.py -k worker_settings

    Steps:
        1. Pass command settings unchanged and print one completed-file status.
        2. Check that the selected series, date block and access profile reach the runner.
    """
    observed = []
    result = SimpleNamespace(
        day=date(2024, 7, 5),
        rows=1_038_240,
        bytes=4096,
        elapsed_seconds=0.25,
        skipped=False,
    )

    def run(progress):
        """Print the same small status returned by a completed worker.

        Args:
            progress: Command callback receiving one completed date.

        Returns:
            Tuple containing the completed-file status.

        Examples:
            results = job.run(progress)

        Steps:
            1. Append forwarded CLI arguments and options to the calls list.
        """
        progress(result)
        return (result,)

    def select_run(source, destination, start, end, selected, **settings):
        """Record CLI settings without opening either file root.

        Args:
            source: Caller-selected daily score source.
            destination: Caller-selected weighted score destination.
            start: First requested output date.
            end: Last requested output date.
            selected: Maximum/minimum and warm/cold series.
            **settings: Worker count, date block size and replacement choice.

        Returns:
            Small runner exposing the progress method.

        Examples:
            job = select_run(source, destination, start, end, series)

        Steps:
            1. Record CLI settings without opening either file root.
        """
        observed.append((source, destination, start, end, selected, settings))
        return SimpleNamespace(run=run)

    monkeypatch.setattr(cli, "ScoreRun", select_run)
    arguments = score_arguments(tmp_path / "daily", tmp_path / "weighted")
    completed = CliRunner().invoke(
        main,
        [
            *arguments,
            "--series",
            series,
            "--profile",
            "selected-profile",
            "--max-workers",
            "8",
            "--block-days",
            "14",
            "--force-update",
        ],
    )
    assert completed.exit_code == 0, completed.output
    source, destination, start, end, selected, settings = observed[0]
    assert source.location == str(tmp_path / "daily")
    assert destination.location == str(tmp_path / "weighted")
    assert (
        source.storage_options
        == destination.storage_options
        == {"profile": "selected-profile"}
    )
    assert f"{selected.temperature}/{selected.direction}" == series
    assert start == end == date(2024, 7, 5)
    assert settings == {"max_workers": 8, "block_days": 14, "force_update": True}
    assert (
        "2024-07-05: 1038240 cell scores, 4096 bytes, 0.25s (written)"
        in completed.output
    )
    assert "Completed 1 daily weighted-score files" in completed.output


def test_scores_cli_check_only_reports_missing_outputs_without_daily_inputs(tmp_path):
    """Check an absent output date without requiring its five daily inputs.

    Args:
        tmp_path: Independent absent daily-input and output roots.

    Returns:
        None. Check-only reports the dated output and incompatible options clearly.

    Examples:
        pytest tests/test_score_cli.py -k missing_outputs

    Steps:
        1. Check an absent output date without requiring its five daily inputs.
        2. Check that check-only reports the dated output and incompatible options clearly.
    """
    arguments = score_arguments(tmp_path / "absent-daily", tmp_path / "weighted")
    runner = CliRunner()
    missing = runner.invoke(main, [*arguments, "--check-only"])
    assert missing.exit_code == 1
    assert "missing weighted-score dates: 2024-07-05" in missing.output
    assert not (tmp_path / "absent-daily").exists()
    invalid = runner.invoke(main, [*arguments, "--check-only", "--force-update"])
    assert invalid.exit_code == 1
    assert "cannot be combined" in invalid.output


@pytest.mark.parametrize("option", ["--max-workers", "--block-days"])
def test_scores_cli_rejects_nonpositive_parallel_settings(tmp_path, option):
    """Reject invalid worker or block sizes before starting any calculation.

    Args:
        tmp_path: Independent absent input and output roots.
        option: Worker-count or date-block-size flag.

    Returns:
        None. Click explains the positive integer requirement.

    Examples:
        pytest tests/test_score_cli.py -k nonpositive

    Steps:
        1. Reject invalid worker or block sizes before starting any calculation.
        2. Check that click explains the positive integer requirement.
    """
    result = CliRunner().invoke(
        main,
        [*score_arguments(tmp_path / "daily", tmp_path / "weighted"), option, "0"],
    )
    assert result.exit_code == 2
    assert option in result.output
    assert "not in the range" in result.output


def test_scores_cli_wraps_dated_input_failure(tmp_path, monkeypatch):
    """Keep a missing lead-in date readable as a command-line error.

    Args:
        tmp_path: Independent daily-input and output roots.
        monkeypatch: Fixture substituting a runner with a dated read failure.

    Returns:
        None. The error preserves the failed date without a traceback.

    Examples:
        pytest tests/test_score_cli.py -k input_failure

    Steps:
        1. Keep a missing lead-in date readable as a command-line error.
        2. Check that the error preserves the failed date without a traceback.
    """

    def fail(progress):
        """Raise a missing daily-input error for the first required lead-in date.

        Args:
            progress: Unused completed-date callback.

        Returns:
            None.

        Raises:
            FileNotFoundError: The complete five-day input window is unavailable.

        Examples:
            job.run(progress)

        Steps:
            1. Raise a missing daily-input error for the first required lead-in date.
        """
        raise FileNotFoundError("missing daily score input: 2024-07-01.parquet")

    monkeypatch.setattr(
        cli, "ScoreRun", lambda *args, **kwargs: SimpleNamespace(run=fail)
    )
    result = CliRunner().invoke(
        main,
        score_arguments(tmp_path / "daily", tmp_path / "weighted"),
    )
    assert result.exit_code == 1
    assert "Error: missing daily score input: 2024-07-01.parquet" in result.output
