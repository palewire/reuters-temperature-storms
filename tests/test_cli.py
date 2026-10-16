"""CLI tests for daily footprint creation and chronological tracking."""

from datetime import date
from pathlib import Path

import pytest
from click.testing import CliRunner

from reuters_temperature_storms import cli
from reuters_temperature_storms.cli import main


def test_pipeline_commands_are_exposed():
    """Verify pipeline commands are exposed.

    Returns:
        None.

    Steps:
        1. Invoke root CLI help.
        2. Compare exposed command names with the expected pipeline stages.

    Examples:
        Run ``pytest tests/test_cli.py -k pipeline_commands_are_exposed``.
    """
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert set(main.commands) == {
        "run",
        "status",
        "footprints",
        "cells",
        "state",
        "scores",
        "anomalies",
    }


def test_run_selects_heat_and_explicit_empty_dates(tmp_path, monkeypatch):
    """Verify run selects heat and explicit empty dates.

    Args:
        tmp_path: Pytest-owned temporary directory for input and output files.
        monkeypatch: Pytest fixture replacing dependencies during the check.

    Returns:
        None.

    Steps:
        1. Replace tracking with an argument-recording runner.
        2. Invoke the run command with its date range, explicit empty dates, and stop limit.
        3. Check the forwarded source, default warm maximum-temperature series, and stop limit.

    Examples:
        Run ``pytest tests/test_cli.py -k run_selects_heat_and_explicit_empty_dates``.
    """
    calls = []

    def run(source, start, end, output, **options):
        """Record runner arguments forwarded by the CLI.

        Args:
            source: Daily footprint source passed by the CLI.
            start: Requested first observation date.
            end: Requested last observation date.
            output: Requested output directory.
            options: Additional runner options forwarded by the CLI.

        Returns:
            None.

        Steps:
            1. Append forwarded CLI arguments and options to the calls list.

        Examples:
            Used by ``pytest tests/test_cli.py`` for fixture construction and output checks.
        """
        calls.append((source, start, end, output, options))
        return

    monkeypatch.setattr(cli, "run_waves", run)
    result = CliRunner().invoke(
        main,
        [
            "run",
            "--wave-footprints",
            str(tmp_path),
            "--start-date",
            "1960-01-01",
            "--end-date",
            "1960-01-06",
            "--empty-through",
            "1960-01-04",
            "--output",
            str(tmp_path / "waves"),
            "--stop-after-days",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    source, _start, _end, _output, options = calls[0]
    assert source.known_empty_dates == tuple(date(1960, 1, i) for i in range(1, 5))
    assert options["series"].temperature == "tasmax"
    assert options["series"].direction == "warm"
    assert options["stop_after_days"] == 2
    assert "Stopped" in result.output


def test_empty_declaration_must_be_inside_range(tmp_path):
    """Verify empty declaration must be inside range.

    Args:
        tmp_path: Pytest-owned temporary directory for input and output files.

    Returns:
        None.

    Steps:
        1. Invoke tracking with an empty-through date preceding the requested start.
        2. Check that the CLI rejects the declaration.

    Examples:
        Run ``pytest tests/test_cli.py -k empty_declaration_must_be_inside_range``.
    """
    result = CliRunner().invoke(
        main,
        [
            "run",
            "--wave-footprints",
            str(tmp_path),
            "--start-date",
            "1960-01-05",
            "--end-date",
            "1960-01-06",
            "--empty-through",
            "1960-01-04",
            "--output",
            str(tmp_path / "waves"),
        ],
    )
    assert result.exit_code == 1
    assert "inside" in result.output


def test_upload_completion_marker_is_last(tmp_path, monkeypatch):
    """Verify upload completion marker is last.

    Args:
        tmp_path: Pytest-owned temporary directory for input and output files.
        monkeypatch: Pytest fixture replacing dependencies during the check.

    Returns:
        None.

    Steps:
        1. Create synthetic catalog files and replace the remote store with an upload recorder.
        2. Upload the result directory and check that the completion marker follows every catalog file.

    Examples:
        Run ``pytest tests/test_cli.py -k upload_completion_marker_is_last``.
    """
    calls = []

    class Store:
        def put_file(self, source, destination):
            """Record uploaded filenames for completion-order assertions.

            Args:
                source: Daily footprint source passed by the CLI.
                destination: Requested destination path recorded by the test store.

            Returns:
                None.

            Steps:
                1. Append the source filename to the upload-order list.

            Examples:
                Used by ``pytest tests/test_cli.py`` for fixture construction and output checks.
            """
            calls.append(Path(source).name)

    monkeypatch.setattr(
        cli.fsspec.core,
        "url_to_fs",
        lambda *args, **kwargs: (Store(), "bucket/results"),
    )
    (tmp_path / "_COMPLETE.json").write_text("{}")
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "all_storms.parquet").write_bytes(b"data")
    (tmp_path / "all_waves.parquet").write_bytes(b"data")
    cli._upload_results(tmp_path, "s3://bucket/results", {})
    assert calls[-1] == "_COMPLETE.json"
    assert set(calls) == {
        "manifest.json",
        "all_storms.parquet",
        "all_waves.parquet",
        "_COMPLETE.json",
    }


def test_completed_cli_points_to_named_catalog(tmp_path, monkeypatch):
    """Show the pipeline's directly readable catalog after completion.

    Args:
        tmp_path: Independent output directory.
        monkeypatch: Fixture replacing tracking with a completed result.

    Returns:
        None. The final CLI output identifies the named GeoParquet file.

    Examples:
        Run ``pytest tests/test_cli.py -k points_to_named``.

    Steps:
        1. Show the pipeline's directly readable catalog after completion.
        2. Check that the final CLI output identifies the named GeoParquet file.
    """
    monkeypatch.setattr(cli, "run_waves", lambda *args, **kwargs: object())
    destination = tmp_path / "waves"
    result = CliRunner().invoke(
        main,
        [
            "run",
            "--wave-footprints",
            str(tmp_path),
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-01-02",
            "--output",
            str(destination),
        ],
    )
    assert result.exit_code == 0
    assert f"Named-storm catalog: {destination}/all_storms.parquet" in result.output
    assert f"Full wave catalog: {destination}/all_waves.parquet" in result.output


@pytest.mark.parametrize(
    "series", ["tasmax/warm", "tasmin/warm", "tasmax/cold", "tasmin/cold"]
)
def test_progress_calls_only_named_events_storms(tmp_path, monkeypatch, series):
    """Use heat/cold wave and storm wording for the selected tracking direction.

    Args:
        tmp_path: Private CLI output path.
        monkeypatch: Fixture supplying representative progress counts.
        series: Each supported maximum/minimum and warm/cold selection.

    Returns:
        None. Unnamed wave counts and named storm counts have separate labels.

    Examples:
        Run ``pytest tests/test_cli.py -k only_named``.

    Steps:
        1. Use heat/cold wave and storm wording for the selected tracking direction.
        2. Check that unnamed wave counts and named storm counts have separate labels.
    """

    def run(source, start, end, output, **options):
        """Emit counts before and after an event receives its name.

        Args:
            source: CLI-selected source.
            start: First included date.
            end: Last included date.
            output: CLI-selected result directory.
            **options: Selected series and progress callback.

        Returns:
            None, representing a cleanly stopped run.

        Examples:
            The CLI supplies the progress callback.

        Steps:
            1. Append forwarded CLI arguments and options to the calls list.
        """
        options["progress"](start, 2, 0)
        options["progress"](end, 1, 1)

    monkeypatch.setattr(cli, "run_waves", run)
    result = CliRunner().invoke(
        main,
        [
            "run",
            "--wave-footprints",
            str(tmp_path),
            "--series",
            series,
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-01-02",
            "--output",
            str(tmp_path / "waves"),
        ],
    )
    assert result.exit_code == 0, result.output
    term = "heat" if series.endswith("warm") else "cold"
    assert f"2024-01-01: 2 active {term} waves, 0 named {term} storms" in result.output
    assert f"2024-01-02: 1 active {term} waves, 1 named {term} storms" in result.output


@pytest.mark.parametrize("option", ["--wave-footprints", "--wave-cells"])
def test_run_selects_one_input_stage(tmp_path, monkeypatch, option):
    """Select the requested daily source while sharing one runner.

    Args:
        tmp_path: Independent source and destination paths.
        monkeypatch: Fixture replacing tracking with a stopped run.
        option: Selected footprint or cell input option.

    Returns:
        None. Exactly the selected source reaches the runner.

    Examples:
        Run ``pytest tests/test_cli.py -k input_stage``.

    Steps:
        1. Select the requested daily source while sharing one runner.
        2. Check that exactly the selected source reaches the runner.
    """
    calls = []
    monkeypatch.setattr(
        cli, "run_waves", lambda source, *args, **kwargs: calls.append(source)
    )
    result = CliRunner().invoke(
        main,
        [
            "run",
            option,
            str(tmp_path),
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-01-02",
            "--output",
            str(tmp_path / "waves"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls[0].kind == option.removeprefix("--")


@pytest.mark.parametrize(
    "options", [[], ["--wave-footprints", "one", "--wave-cells", "two"]]
)
def test_run_rejects_missing_or_ambiguous_input(options):
    """Require exactly one input stage before opening files.

    Args:
        options: Missing input options or both conflicting options.

    Returns:
        None. The CLI identifies the required exclusive choice.

    Examples:
        Run ``pytest tests/test_cli.py -k ambiguous``.

    Steps:
        1. Require exactly one input stage before opening files.
        2. Check that the CLI identifies the required exclusive choice.
    """
    result = CliRunner().invoke(
        main,
        [
            "run",
            *options,
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-01-02",
            "--output",
            "waves",
        ],
    )
    assert result.exit_code == 2
    assert "exactly one" in result.output
