"""Exercise synthetic footprint histories through the real CLI."""

import json

from click.testing import CliRunner

from reuters_temperature_storms import (
    open_completed_generation,
    scan_lineage,
    scan_wave_days,
    scan_waves,
)
from reuters_temperature_storms.cli import main
from reuters_temperature_storms.naming import name_at_ordinal
from tests.fixtures import create_footprints


def test_fresh_and_resumed_cli_results_are_identical(tmp_path):
    """Keep names, geometry, history and lineage equal after a CLI restart.

    Args:
        tmp_path: Temporary directory for invented inputs and both result sets.

    Returns:
        None. Checks complete scientific rows and exact final checkpoint bytes.

    Examples:
        pytest tests/test_cli_tracking.py

    Steps:
        1. Keep names, geometry, history and lineage equal after a CLI restart.
        2. Verify complete scientific rows and exact final checkpoint bytes.
    """
    footprints = tmp_path / "footprints"
    files = create_footprints(footprints)
    assert len(files) == 5
    assert len(list(footprints.rglob("*.parquet"))) == 5
    runner = CliRunner()
    arguments = [
        "run",
        "--wave-footprints",
        str(footprints),
        "--start-date",
        "2020-01-01",
        "--end-date",
        "2020-01-05",
    ]
    fresh = tmp_path / "fresh"
    result = runner.invoke(main, [*arguments, "--output", str(fresh)])
    assert result.exit_code == 0, result.output
    assert "Completed" in result.output

    resumed = tmp_path / "resumed"
    result = runner.invoke(
        main,
        [*arguments, "--output", str(resumed), "--stop-after-days", "2"],
    )
    assert result.exit_code == 0, result.output
    assert "Stopped" in result.output
    assert not (resumed / "_COMPLETE.json").exists()
    status = runner.invoke(main, ["status", "--output", str(resumed)])
    assert status.exit_code == 0, status.output
    progress = json.loads(status.output)
    assert progress["processed_days"] == 2
    assert progress["through_date"] == "2020-01-02"
    assert progress["next_date"] == "2020-01-03"

    result = runner.invoke(main, [*arguments, "--output", str(resumed)])
    assert result.exit_code == 0, result.output
    assert "2020-01-01:" not in result.output
    assert "2020-01-02:" not in result.output
    assert "2020-01-03:" in result.output
    assert "Completed" in result.output
    fresh_generation = open_completed_generation(str(fresh))
    resumed_generation = open_completed_generation(str(resumed))
    for scan in (scan_wave_days, scan_waves, scan_lineage):
        expected = scan(fresh_generation)
        actual = scan(resumed_generation)
        assert actual.equals(expected)
    assert (tmp_path / "fresh-state" / "checkpoint.json").read_bytes() == (
        tmp_path / "resumed-state" / "checkpoint.json"
    ).read_bytes()

    catalog = scan_waves(fresh_generation).to_dict("records")
    assert catalog
    assert all(not wave["active"] for wave in catalog)
    assert f"{name_at_ordinal(0)} 2020" in {wave["name"] for wave in catalog}
    lineage = scan_lineage(fresh_generation).to_dict("records")
    assert {edge["relationship"] for edge in lineage} == {"split", "merge"}
