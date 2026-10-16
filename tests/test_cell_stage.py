"""Exercise independent daily state-to-cell files and their CLI."""

from datetime import date, timedelta
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from click.testing import CliRunner

from reuters_temperature_storms.cell_run import CellRun, CellRunError
from reuters_temperature_storms.cell_state import WaveCellStateSource
from reuters_temperature_storms.cells import WaveCells, WaveCellSource
from reuters_temperature_storms.cli import main
from reuters_temperature_storms.series import WaveSeries

DAY = date(2024, 7, 1)
SERIES = WaveSeries("tasmax", "warm")


def _write_state(root: Path, count: int = 3) -> WaveCellStateSource:
    """Write small dated states covering qualified, grace and immature cells.

    Args:
        root: Independent temporary state directory.
        count: Number of consecutive dates to write.

    Returns:
        Daily state source whose files can be read by spawned workers.

    Examples:
        source = _write_state(tmp_path / "state", 2)

    Steps:
        1. Write small dated states covering qualified, grace and immature cells.
    """
    source = WaveCellStateSource(str(root))
    for offset in range(count):
        day = DAY + timedelta(days=offset)
        path = Path(source.path_for(day, SERIES))
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "longitude": [0.0, 0.25, 10.0],
                "latitude": [0.0, 0.0, 0.0],
                "start_date": pd.to_datetime([day - timedelta(days=5)] * 3),
                "end_date": pd.to_datetime([None] * 3),
                "days": [6, 5, 1],
                "grace_days": [0, 1, 0],
            }
        ).to_parquet(path, index=False)
    return source


def test_serial_and_spawned_dates_have_identical_cell_rows(tmp_path):
    """Keep full daily cell calculations identical across worker counts.

    Args:
        tmp_path: Independent input and output roots.

    Returns:
        None. Spawned and direct workers save equal valid cell tables.

    Examples:
        pytest tests/test_cell_stage.py -k serial_and_spawned

    Steps:
        1. Keep full daily cell calculations identical across worker counts.
        2. Check that spawned and direct workers save equal valid cell tables.
    """
    source = _write_state(tmp_path / "state")
    direct = WaveCellSource(str(tmp_path / "direct"))
    parallel = WaveCellSource(str(tmp_path / "parallel"))
    end = DAY + timedelta(days=2)
    direct_run = CellRun(source, direct, DAY, end, SERIES, max_workers=1)
    parallel_run = CellRun(source, parallel, DAY, end, SERIES, max_workers=2)
    serial = direct_run.run()
    spawned = parallel_run.run()
    assert [r.day for r in serial] == [r.day for r in spawned]
    assert all(r.rows >= 2 and r.bytes > 0 and not r.skipped for r in spawned)
    assert parallel_run.check() == ()
    for offset in range(3):
        day = DAY + timedelta(days=offset)
        expected = WaveCells.read_parquet(direct.path_for(day, SERIES)).frame
        actual = WaveCells.read_parquet(parallel.path_for(day, SERIES)).frame
        assert actual.equals(expected)


def test_reuse_and_check_need_no_input_and_never_build_footprints(
    tmp_path, monkeypatch
):
    """Reuse accepted cell files without reading state or calculating footprints.

    Args:
        tmp_path: Independent file roots.
        monkeypatch: Fixture preventing all state and footprint calculation calls.

    Returns:
        None. Reuse/check succeed while forced rebuilding needs the missing state.

    Examples:
        pytest tests/test_cell_stage.py -k reuse_and_check

    Steps:
        1. Reuse accepted cell files without reading state or calculating footprints.
        2. Check that reuse/check succeed while forced rebuilding needs the missing state.
    """
    source = _write_state(tmp_path / "state", 1)
    output = WaveCellSource(str(tmp_path / "cells"))
    run = CellRun(source, output, DAY, DAY, SERIES, max_workers=1)
    run.run()
    Path(source.path_for(DAY, SERIES)).unlink()

    def reject_read(*args, **kwargs):
        """Fail if a check or reuse touches another stage.

        Args:
            *args: Unexpected stage inputs.
            **kwargs: Unexpected stage options.

        Returns:
            Never returns; raises AssertionError.

        Examples:
            Used to replace state reading and footprint calculation.

        Steps:
            1. Fail if a check or reuse touches another stage.
        """
        raise AssertionError("reuse must only check saved cells")

    from reuters_temperature_storms.footprints import WaveFootprints

    monkeypatch.setattr(WaveCellStateSource, "read_state", reject_read)
    monkeypatch.setattr(WaveFootprints, "from_cells", reject_read)
    assert run.check() == ()
    assert run.run()[0].skipped
    with pytest.raises(CellRunError, match="only check saved cells"):
        CellRun(
            source, output, DAY, DAY, SERIES, max_workers=1, force_update=True
        ).run()


def test_failures_retain_other_dates_and_invalid_output_is_not_overwritten(tmp_path):
    """Continue independent dates and retain invalid existing bytes for diagnosis.

    Args:
        tmp_path: Independent dated state and cell outputs.

    Returns:
        None. Missing inputs are dated errors and completed files remain reusable.

    Examples:
        pytest tests/test_cell_stage.py -k failures_retain

    Steps:
        1. Continue independent dates and retain invalid existing bytes for diagnosis.
        2. Check that missing inputs are dated errors and completed files remain reusable.
    """
    source = _write_state(tmp_path / "state")
    middle = DAY + timedelta(days=1)
    Path(source.path_for(middle, SERIES)).unlink()
    output = WaveCellSource(str(tmp_path / "cells"))
    run = CellRun(source, output, DAY, DAY + timedelta(days=2), SERIES, max_workers=1)
    with pytest.raises(CellRunError, match=str(middle)):
        run.run()
    assert run.check() == (middle,)
    first_path = Path(output.path_for(DAY, SERIES))
    first_path.write_bytes(b"invalid")
    with pytest.raises(CellRunError, match=str(DAY)):
        run.run()
    assert first_path.read_bytes() == b"invalid"


def test_wrong_observation_date_is_rejected_by_check_and_reuse(tmp_path):
    """Reject a valid cell table saved beneath the wrong date filename.

    Args:
        tmp_path: Independent file roots.

    Returns:
        None. Checking and reusing identify the conflicting observation date.

    Examples:
        pytest tests/test_cell_stage.py -k wrong_observation_date

    Steps:
        1. Reject a valid cell table saved beneath the wrong date filename.
        2. Check that checking and reusing identify the conflicting observation date.
    """
    source = _write_state(tmp_path / "state", 1)
    output = WaveCellSource(str(tmp_path / "cells"))
    run = CellRun(source, output, DAY, DAY, SERIES, max_workers=1)
    run.run()
    path = output.path_for(DAY, SERIES)
    frame = gpd.read_parquet(path)
    frame["date"] = pd.Timestamp(DAY + timedelta(days=1))
    WaveCells(frame).to_parquet(path)
    with pytest.raises(CellRunError, match="must represent"):
        run.check()
    with pytest.raises(CellRunError, match="must represent"):
        run.run()


def test_cli_build_check_and_missing_state_paths(tmp_path):
    """Expose the reusable cell stage without running later pipeline stages.

    Args:
        tmp_path: Independent state and output roots.

    Returns:
        None. CLI writes only dated cell files and supports check-only coverage.

    Examples:
        pytest tests/test_cell_stage.py -k cli_build

    Steps:
        1. Expose the reusable cell stage without running later pipeline stages.
        2. Check that cLI writes only dated cell files and supports check-only coverage.
    """
    source = _write_state(tmp_path / "state", 1)
    output = tmp_path / "cells"
    args = [
        "cells",
        "--wave-state",
        source.location,
        "--output",
        str(output),
        "--start-date",
        str(DAY),
        "--end-date",
        str(DAY),
        "--max-workers",
        "1",
    ]
    runner = CliRunner()
    missing = runner.invoke(main, [*args, "--check-only"])
    assert missing.exit_code == 1
    assert "missing wave-cell dates" in missing.output
    built = runner.invoke(main, args)
    assert built.exit_code == 0, built.output
    assert "Completed 1 daily wave-cell files" in built.output
    assert len(list(output.rglob("*.parquet"))) == 1
    assert not (output / "_COMPLETE.json").exists()
    assert not (output / "all_waves.parquet").exists()
    checked = runner.invoke(main, [*args, "--check-only"])
    assert checked.exit_code == 0, checked.output
    invalid = runner.invoke(main, [*args, "--check-only", "--force-update"])
    assert invalid.exit_code == 1
    assert "cannot be combined" in invalid.output
    overwrite = runner.invoke(main, [*args, "--output", source.location])
    assert overwrite.exit_code == 1
    assert "different directories" in overwrite.output


def test_immature_state_writes_empty_cells_and_missing_state_is_not_empty(tmp_path):
    """Represent an immature date explicitly and keep absent state as a failure.

    Args:
        tmp_path: Independent state and cell file roots.

    Returns:
        None. Existing immature state writes a typed empty cell GeoParquet.

    Examples:
        pytest tests/test_cell_stage.py -k immature_state

    Steps:
        1. Represent an immature date explicitly and keep absent state as a failure.
        2. Check that existing immature state writes a typed empty cell GeoParquet.
    """
    source = _write_state(tmp_path / "state", 1)
    path = source.path_for(DAY, SERIES)
    frame = pd.read_parquet(path)
    frame["days"] = 1
    frame["grace_days"] = 0
    frame.to_parquet(path, index=False)
    output = WaveCellSource(str(tmp_path / "cells"))
    run = CellRun(source, output, DAY, DAY, SERIES, max_workers=1)
    (result,) = run.run()
    assert result.rows == 0
    assert result.bytes > 0
    assert WaveCells.read_parquet(output.path_for(DAY, SERIES)).frame.empty
    assert run.check() == ()
    tomorrow = DAY + timedelta(days=1)
    with pytest.raises(CellRunError, match=str(tomorrow)):
        CellRun(source, output, tomorrow, tomorrow, SERIES, max_workers=1).run()


def test_cell_local_replacement_failure_preserves_prior_bytes(tmp_path, monkeypatch):
    """Retain a completed local cell file when replacement fails midway.

    Args:
        tmp_path: Independent state and output paths.
        monkeypatch: Fixture interrupting replacement after the temporary write.

    Returns:
        None. Prior file bytes remain readable and temporary output is removed.

    Examples:
        pytest tests/test_cell_stage.py -k local_replacement

    Steps:
        1. Retain a completed local cell file when replacement fails midway.
        2. Check that prior file bytes remain readable and temporary output is removed.
    """
    source = _write_state(tmp_path / "state", 1)
    cells = source.read_state(DAY, SERIES).to_cells()
    target = tmp_path / "cells.parquet"
    cells.to_parquet(target)
    before = target.read_bytes()

    def reject_replace(self, destination):
        """Raise after the full temporary cell file has been encoded.

        Args:
            self: Temporary sibling containing completed cell bytes.
            destination: Existing final cell path.

        Returns:
            Never returns; raises OSError to simulate interruption.

        Examples:
            Called at the final local replacement step.

        Steps:
            1. Raise after the full temporary cell file has been encoded.
        """
        assert Path(destination) == target
        assert WaveCells.read_parquet(self).frame.equals(cells.frame)
        raise OSError("simulated cell write interruption")

    monkeypatch.setattr(Path, "replace", reject_replace)
    with pytest.raises(OSError, match="write interruption"):
        cells.to_parquet(target)
    assert target.read_bytes() == before
    assert list(tmp_path.glob(".*.tmp")) == []


def test_explicit_empty_source_settings_reach_spawned_workers(tmp_path):
    """Honor the source's explicit empty dates when creating independent tasks.

    Args:
        tmp_path: Independent source and output directories.

    Returns:
        None. Declared missing dates become saved typed empty cell files.

    Examples:
        pytest tests/test_cell_stage.py -k explicit_empty_source

    Steps:
        1. Honor the source's explicit empty dates when creating independent tasks.
        2. Check that declared missing dates become saved typed empty cell files.
    """
    tomorrow = DAY + timedelta(days=1)
    source = WaveCellStateSource(
        str(tmp_path / "state"), known_empty_dates=(DAY, tomorrow)
    )
    output = WaveCellSource(str(tmp_path / "cells"))
    run = CellRun(source, output, DAY, tomorrow, SERIES, max_workers=2)
    results = run.run()
    assert [result.rows for result in results] == [0, 0]
    assert run.check() == ()
