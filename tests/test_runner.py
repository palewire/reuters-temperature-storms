"""Focused complete-output and restart tests for the footprint runner."""

from __future__ import annotations

import gzip
import json
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import pytest
from fsspec.implementations.local import LocalFileSystem
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform

import reuters_temperature_storms.records as records_module
import reuters_temperature_storms.runner as runner_module
from reuters_temperature_storms._wave_tracking import WaveSeries
from reuters_temperature_storms.components import (
    ComponentMembership,
    DailyWaveComponentGeometry,
)
from reuters_temperature_storms.footprints import WaveFootprintSource
from reuters_temperature_storms.identity import component_identity
from reuters_temperature_storms.records import (
    open_completed_generation,
    scan_lineage,
    scan_wave_days,
    scan_waves,
)
from reuters_temperature_storms.runner import WaveRunnerError, run_status, run_waves

if TYPE_CHECKING:
    from fsspec.spec import AbstractFileSystem

_START = date(2023, 12, 31)
_SERIES = (
    WaveSeries("tasmax", "warm"),
    WaveSeries("tasmax", "cold"),
    WaveSeries("tasmin", "warm"),
    WaveSeries("tasmin", "cold"),
)
_AREA_TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)


def _component(day: date, series: WaveSeries) -> ComponentMembership:
    """Build a measured footprint large enough to receive a Reuters name.

    Args:
        day: Date of the footprint.
        series: Independent temperature and direction owning the footprint.

    Returns:
        A valid compact membership and all-qualified daily geometry.

    Examples:
        ``_component(_START, _SERIES[0])`` builds the opening warm wave.

    Steps:
        1. Choose the supplied qualified and grace polygons and measure their areas.
        2. Build the daily geometry record and compute the full polygon's component identifier.
        3. Return the identified component membership.
    """
    shape = box(-20, -20, 20, 20)
    area = float(transform(_AREA_TRANSFORMER.transform, shape).area / 1_000_000)
    day_value = np.datetime64(day, "D")
    return ComponentMembership(
        component_id=component_identity(
            temperature=series.temperature,
            direction=series.direction,
            date=day_value,
            geometry=shape,
        ),
        geometry=DailyWaveComponentGeometry(
            time=day_value,
            label=1,
            geometry=shape,
            grace_geometry=None,
            full_geometry=shape,
            area_qualified=area,
            area_grace=0.0,
            area_gap_filled=0.0,
            qualified_fraction=1.0,
        ),
    )


@pytest.mark.component
@pytest.mark.parametrize("series", _SERIES)
def test_wave_becomes_named_storm_without_changing_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, series: WaveSeries
) -> None:
    """Keep wave identity and dated naming state while reporting distinct counts.

    Args:
        tmp_path: Private restart and output directory.
        monkeypatch: Fixture supplying finished components instead of cloud data.
        series: Each supported maximum/minimum and warm/cold branch.

    Returns:
        None. Naming, restart, status and output retain the same wave identity.

    Examples:
        Run ``pytest tests/test_runner.py -k becomes_named``.

    Steps:
        1. Keep wave identity and dated naming state while reporting distinct counts.
        2. Check that naming, restart, status and output retain the same wave identity.
    """

    def read(
        self: WaveFootprintSource, day: date, selected: WaveSeries
    ) -> tuple[ComponentMembership, ...]:
        """Supply one growing wave and one separate, persistently unnamed wave.

        Args:
            self: Footprint source used by the runner.
            day: Requested date.
            selected: Requested tracking branch.

        Returns:
            Two valid daily footprints, with the first named only on day two.

        Examples:
            The runner reads these sequentially and saves each date.

        Steps:
            1. Supply one growing wave and one separate, persistently unnamed wave.
        """
        widths = 20 if day == _START + timedelta(days=1) else 2
        shapes = (box(-widths, -20, widths, 20), box(50, 0, 51, 1))
        full_shapes = (box(-20, -20, 20, 20), shapes[1])
        template = _component(day, selected)
        return tuple(
            replace(
                template,
                component_id=component_identity(
                    temperature=selected.temperature,
                    direction=selected.direction,
                    date=np.datetime64(day, "D"),
                    geometry=full_shape,
                ),
                geometry=replace(
                    template.geometry,
                    label=index,
                    geometry=shape,
                    full_geometry=full_shape,
                    grace_geometry=full_shape.difference(shape)
                    if widths == 2 and index == 1
                    else None,
                    area_qualified=float(
                        transform(_AREA_TRANSFORMER.transform, shape).area / 1_000_000
                    ),
                    area_grace=float(
                        transform(
                            _AREA_TRANSFORMER.transform, full_shape.difference(shape)
                        ).area
                        / 1_000_000
                    ),
                    qualified_fraction=float(
                        transform(_AREA_TRANSFORMER.transform, shape).area
                        / transform(_AREA_TRANSFORMER.transform, full_shape).area
                    ),
                ),
            )
            for index, (shape, full_shape) in enumerate(
                zip(shapes, full_shapes, strict=True), 1
            )
        )

    monkeypatch.setattr(WaveFootprintSource, "read", read)
    source = WaveFootprintSource(str(tmp_path / "inputs"))
    destination = tmp_path / "waves"
    end = _START + timedelta(days=2)
    progress: list[tuple[date, int, int]] = []
    for offset, counts in enumerate(((2, 0), (1, 1), (1, 1))):
        run_waves(
            source,
            _START,
            end,
            destination,
            series=series,
            stop_after_days=1,
            progress=lambda day, waves, storms: progress.append((day, waves, storms)),
        )
        status = run_status(destination)
        assert (status["active_waves"], status["active_named_storms"]) == counts
        assert "active_storms" not in status
        assert progress[-1] == (_START + timedelta(days=offset), *counts)
    generation = open_completed_generation(destination)
    daily = scan_wave_days(generation).sort_values(["wave_id", "date"])
    named_id = daily.loc[daily.name.notna(), "wave_id"].unique()
    assert len(named_id) == 1
    history = daily[daily.wave_id == named_id[0]].reset_index(drop=True)
    assert len(history) == 3
    assert pd.isna(history.loc[0, "name"])
    assert history.loc[1, "name"] == history.loc[2, "name"]
    assert history.loc[2, "qualified_area_km2"] < history.loc[1, "qualified_area_km2"]
    assert "storm_id" not in daily
    assert "parent_wave_ids" in daily
    checkpoint = json.loads(
        destination.with_name("waves-state").joinpath("checkpoint.json").read_bytes()
    )
    assert all(
        "wave_id" in record and "storm_id" not in record
        for record in checkpoint["active"]
    )


def _scientific_rows(destination: Path) -> tuple[list[dict[str, object]], ...]:
    """Read scientific tables from a completed output for exact comparison.

    Args:
        destination: Completed generation directory.

    Returns:
        Daily, catalog and lineage rows in writer order.

    Examples:
        ``_scientific_rows(destination)`` verifies ordinary public readback.

    Steps:
        1. Read scientific tables from a completed output for exact comparison.
    """
    generation = open_completed_generation(destination)
    results = []
    for scanner in (scan_wave_days, scan_waves, scan_lineage):
        frame = scanner(generation)
        for column in frame:
            if pd.api.types.is_datetime64_any_dtype(frame[column]):
                frame[column] = frame[column].dt.date.where(frame[column].notna(), None)
        results.append(frame.to_dict("records"))
    return tuple(results)


@pytest.mark.component
def test_footprint_runner_writes_named_catalog_and_empty_dates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Produce and read back a named storm ending on an empty input day.

    Args:
        tmp_path: Private source and destination directories.
        monkeypatch: Fixture replacing cloud reads with finished components.

    Returns:
        None.

    Examples:
        ``pytest tests/test_runner.py -k writes_named`` runs this check.

    Steps:
        1. Produce and read back a named storm ending on an empty input day.
    """
    read_dates: list[tuple[date, WaveSeries]] = []

    def read(
        self: WaveFootprintSource, day: date, series: WaveSeries
    ) -> tuple[ComponentMembership, ...]:
        """Supply a two-day warm wave followed by an empty date.

        Args:
            self: Source configured by the test.
            day: Requested input date.
            series: Requested independent series.

        Returns:
            A component on the first two maximum-warm dates, otherwise empty.

        Examples:
            The runner calls this once per date and series.

        Steps:
            1. Supply a two-day warm wave followed by an empty date.
        """
        read_dates.append((day, series))
        if series == _SERIES[0] and day < _START + timedelta(days=2):
            return (_component(day, series),)
        return ()

    monkeypatch.setattr(WaveFootprintSource, "read", read)
    source = WaveFootprintSource(str(tmp_path / "source"))
    destination = tmp_path / "beta"
    progress: list[tuple[date, int, int]] = []
    catalog_progress: list[str] = []
    evidence = run_waves(
        source,
        _START,
        _START + timedelta(days=2),
        destination,
        progress=lambda day, waves, storms: progress.append((day, waves, storms)),
        catalog_progress=catalog_progress.append,
    )
    assert evidence is not None
    assert catalog_progress[0] == "Catalog: reading saved daily batches"
    assert any("daily batches 3/3" in message for message in catalog_progress)
    assert any("1 named storms, 1 waves" in message for message in catalog_progress)
    assert catalog_progress[-1] == "Catalog: completion marker saved"
    assert evidence.inventory.datasets["wave_days"].rows == 2
    assert evidence.inventory.datasets["waves"].rows == 1
    assert evidence.inventory.datasets["lineage"].rows == 0
    assert len(read_dates) == 3
    assert [day for day, _, _ in progress] == [
        _START + timedelta(days=offset) for offset in range(3)
    ]
    daily, catalog, lineage = _scientific_rows(destination)
    assert len(daily) == 2
    assert len(catalog) == 1
    assert lineage == []
    assert daily[0]["wave_id"] == daily[1]["wave_id"]
    assert daily[0]["name"] == daily[1]["name"]
    name = daily[0]["name"]
    assert isinstance(name, str)
    assert name.endswith(" 2023")
    assert catalog[0]["name"] == daily[0]["name"]
    assert catalog[0]["active"] is False
    assert catalog[0]["duration_days"] == 2
    assert catalog[0]["last_observed_date"] == _START + timedelta(days=1)
    assert catalog[0]["end_date"] == _START + timedelta(days=1)
    state = tmp_path / "beta-state"
    assert (state / "checkpoint.json").exists()
    assert len(list((state / "batches").glob("*.json.gz"))) == 3
    assert (
        json.loads((destination / "manifest.json").read_text())["end_date"]
        == (_START + timedelta(days=2)).isoformat()
    )


@pytest.mark.component
def test_footprint_runner_resumes_without_reading_committed_dates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume interrupted input reads and match an uninterrupted generation.

    Args:
        tmp_path: Separate completed and interrupted destinations.
        monkeypatch: Fixture injecting one failure in a footprint read.

    Returns:
        None.

    Examples:
        ``pytest tests/test_runner.py -k resumes`` runs this check.

    Steps:
        1. Resume interrupted input reads and match an uninterrupted generation.
    """
    end = _START + timedelta(days=2)
    read_dates: list[tuple[date, WaveSeries]] = []
    fail_day: date | None = _START + timedelta(days=1)

    def read(
        self: WaveFootprintSource, day: date, series: WaveSeries
    ) -> tuple[ComponentMembership, ...]:
        """Return real tracking inputs while injecting a selected failed date.

        Args:
            self: Source configured by the test.
            day: Requested date.
            series: Requested series.

        Returns:
            One maximum-warm component per date and empty other series.

        Raises:
            OSError: When the test's failure date is requested.

        Examples:
            Clearing ``fail_day`` permits the interrupted run to resume.

        Steps:
            1. Return real tracking inputs while injecting a selected failed date.
        """
        read_dates.append((day, series))
        if day == fail_day:
            msg = "interrupted footprint read"
            raise OSError(msg)
        return (_component(day, series),) if series == _SERIES[0] else ()

    monkeypatch.setattr(WaveFootprintSource, "read", read)
    source = WaveFootprintSource(str(tmp_path / "source"))
    interrupted = tmp_path / "interrupted"
    with pytest.raises(OSError, match="interrupted footprint read"):
        run_waves(source, _START, end, interrupted)
    assert (tmp_path / "interrupted-state" / "checkpoint.json").exists()
    assert not (interrupted / "_COMPLETE.json").exists()
    fail_day = None
    read_dates.clear()
    resumed = run_waves(source, _START, end, interrupted)
    assert resumed is not None
    assert resumed.inventory.datasets["wave_days"].rows == 3
    assert resumed.inventory.datasets["waves"].rows == 1
    assert (interrupted / "_COMPLETE.json").exists()
    assert len(read_dates) == 2
    assert all(day > _START for day, _ in read_dates)
    uninterrupted = tmp_path / "uninterrupted"
    run_waves(source, _START, end, uninterrupted)
    assert _scientific_rows(interrupted) == _scientific_rows(uninterrupted)


@pytest.mark.component
@pytest.mark.parametrize("changed", ["source", "start", "end"])
def test_footprint_runner_rejects_changed_request_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    """Reject a resumed request that changes the saved source or date range.

    Args:
        tmp_path: Private interrupted destination.
        monkeypatch: Fixture injecting a stop after the first committed date.
        changed: Request field altered on the resume attempt.

    Returns:
        None.

    Examples:
        ``pytest tests/test_runner.py -k changed_request`` checks all fields.

    Steps:
        1. Reject a resumed request that changes the saved source or date range.
    """
    end = _START + timedelta(days=2)
    reads: list[date] = []

    def read(
        self: WaveFootprintSource, day: date, series: WaveSeries
    ) -> tuple[ComponentMembership, ...]:
        """Stop on the second date after the first date is committed.

        Args:
            self: Source configured by the test.
            day: Requested date.
            series: Requested independent wave series.

        Returns:
            An empty opening date.

        Raises:
            OSError: On later requested dates.

        Examples:
            The checkpoint remains available for testing rejected requests.

        Steps:
            1. Stop on the second date after the first date is committed.
        """
        reads.append(day)
        if day > _START:
            msg = "stop after opening date"
            raise OSError(msg)
        return ()

    monkeypatch.setattr(WaveFootprintSource, "read", read)
    source = WaveFootprintSource(str(tmp_path / "source"))
    destination = tmp_path / "beta"
    with pytest.raises(OSError, match="stop after opening date"):
        run_waves(source, _START, end, destination)
    reads.clear()
    changed_source = (
        WaveFootprintSource(str(tmp_path / "other-source"))
        if changed == "source"
        else source
    )
    changed_start = _START + timedelta(days=1) if changed == "start" else _START
    changed_end = end + timedelta(days=1) if changed == "end" else end
    with pytest.raises(WaveRunnerError, match=r"request|source|range|match"):
        run_waves(changed_source, changed_start, changed_end, destination)
    assert reads == []


@pytest.mark.component
def test_footprint_runner_resumes_interrupted_writer_without_rereading_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recover matching partial output using the completed tracker batches.

    Args:
        tmp_path: Private source, output and restart directories.
        monkeypatch: Fixture injecting a real writer interruption after setup.

    Returns:
        None.

    Examples:
        ``pytest tests/test_runner.py -k interrupted_writer`` runs this check.

    Steps:
        1. Recover matching partial output using the completed tracker batches.
    """
    reads: list[date] = []
    destination = tmp_path / "beta"
    end = _START + timedelta(days=1)

    def read(
        self: WaveFootprintSource, day: date, series: WaveSeries
    ) -> tuple[ComponentMembership, ...]:
        """Supply a named maximum-warm wave while recording every input read.

        Args:
            self: Source configured by the test.
            day: Requested date.
            series: Requested independent wave series.

        Returns:
            One maximum-warm footprint and no other series footprints.

        Examples:
            The successful resume must not invoke this function again.

        Steps:
            1. Supply a named maximum-warm wave while recording every input read.
        """
        reads.append(day)
        return (_component(day, series),) if series == _SERIES[0] else ()

    def interrupted_batches(*args: object, **kwargs: object):
        """Raise after the real writer has created its incomplete marker.

        Args:
            *args: Saved-batch reader arguments supplied by the runner.
            **kwargs: Any saved-batch reader keyword arguments.

        Yields:
            No rows; the injected error occurs on first iteration.

        Raises:
            OSError: Always, representing interrupted output assembly.

        Examples:
            The writer leaves its matching incomplete output marker on failure.

        Steps:
            1. Raise after the real writer has created its incomplete marker.
        """
        msg = "interrupted catalog write"
        raise OSError(msg)
        yield

    monkeypatch.setattr(WaveFootprintSource, "read", read)
    source = WaveFootprintSource(str(tmp_path / "source"))
    stored_batches = runner_module._stored_batches
    monkeypatch.setattr(runner_module, "_stored_batches", interrupted_batches)
    with pytest.raises(OSError, match="interrupted catalog write"):
        run_waves(source, _START, end, destination)
    assert len(reads) == 2
    checkpoint = tmp_path / "beta-state" / "checkpoint.json"
    assert checkpoint.exists()
    assert (destination / "_INCOMPLETE.json").exists()
    assert not (destination / "_COMPLETE.json").exists()
    monkeypatch.setattr(runner_module, "_stored_batches", stored_batches)
    reads.clear()
    evidence = run_waves(source, _START, end, destination)
    assert evidence is not None
    assert evidence.inventory.datasets["wave_days"].rows == 2
    assert evidence.inventory.datasets["waves"].rows == 1
    assert reads == []
    daily, catalog, _ = _scientific_rows(destination)
    assert len(daily) == 2
    assert len(catalog) == 1
    assert catalog[0]["active"] is True
    assert catalog[0]["duration_days"] == 2
    assert catalog[0]["name"] == daily[0]["name"]
    assert not (destination / "_INCOMPLETE.json").exists()
    assert (destination / "_COMPLETE.json").exists()


@pytest.mark.component
@pytest.mark.parametrize("interruption", ["complete_write", "incomplete_cleanup"])
def test_footprint_runner_recovers_interrupted_completion_without_input_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interruption: str
) -> None:
    """Recover both sides of the completion marker write using saved batches.

    Args:
        tmp_path: Private source, output and restart directories.
        monkeypatch: Fixture injecting failure at the selected completion step.
        interruption: Completion marker write or incomplete marker cleanup.

    Returns:
        None.

    Examples:
        ``pytest tests/test_runner.py -k interrupted_completion`` runs both.

    Steps:
        1. Recover both sides of the completion marker write using saved batches.
    """
    reads: list[date] = []
    destination = tmp_path / "beta"
    source = WaveFootprintSource(str(tmp_path / "source"))

    def read(
        self: WaveFootprintSource, day: date, series: WaveSeries
    ) -> tuple[ComponentMembership, ...]:
        """Record reads and supply one named storm for the completion test.

        Args:
            self: Source configured by the test.
            day: Requested date.
            series: Requested independent wave series.

        Returns:
            One maximum-warm footprint and empty other series.

        Examples:
            The recovery attempt must not read any input again.

        Steps:
            1. Record reads and supply one named storm for the completion test.
        """
        reads.append(day)
        return (_component(day, series),) if series == _SERIES[0] else ()

    original_write = records_module._write_json
    original_rm = LocalFileSystem.rm

    def interrupted_write(
        filesystem: AbstractFileSystem, path: str, value: object
    ) -> None:
        """Fail only when the writer attempts to publish its completion marker.

        Args:
            filesystem: Filesystem used by the original writer.
            path: JSON destination path.
            value: JSON contents.

        Returns:
            None for unaffected writes.

        Raises:
            OSError: At the selected completion-marker boundary.

        Examples:
            ``_INCOMPLETE.json`` must still exist when this fails.

        Steps:
            1. Fail only when the writer attempts to publish its completion marker.
        """
        if path.endswith("/_COMPLETE.json"):
            with filesystem.open(path, "wb") as stream:
                stream.write(b'{"generation_id":')
            raise OSError("interrupted completion")
        original_write(filesystem, path, value)

    def interrupted_rm(
        self: LocalFileSystem,
        path: str,
        recursive: bool = False,
        maxdepth: int | None = None,
    ) -> None:
        """Fail on incomplete cleanup after the completion marker is durable.

        Args:
            self: Original local filesystem.
            path: Path selected for deletion.
            recursive: Whether descendants are included.
            maxdepth: Optional deletion depth.

        Returns:
            None for unaffected deletions.

        Raises:
            OSError: When incomplete-marker cleanup is attempted.

        Examples:
            Both output markers remain after this injected interruption.

        Steps:
            1. Fail on incomplete cleanup after the completion marker is durable.
        """
        if path.endswith("/_INCOMPLETE.json"):
            raise OSError("interrupted completion")
        original_rm(self, path, recursive=recursive, maxdepth=maxdepth)

    monkeypatch.setattr(WaveFootprintSource, "read", read)
    if interruption == "complete_write":
        monkeypatch.setattr(records_module, "_write_json", interrupted_write)
    else:
        monkeypatch.setattr(LocalFileSystem, "rm", interrupted_rm)
    with pytest.raises(OSError, match="interrupted completion"):
        run_waves(source, _START, _START, destination)
    assert len(reads) == 1
    incomplete = destination / "_INCOMPLETE.json"
    complete = destination / "_COMPLETE.json"
    assert incomplete.exists()
    assert complete.exists()
    if interruption == "complete_write":
        assert complete.read_bytes() == b'{"generation_id":'
    else:
        original_marker = complete.read_bytes()
        complete.write_text(json.dumps({"generation_id": "another-run"}))
        with pytest.raises(WaveRunnerError, match="completion marker"):
            run_waves(source, _START, _START, destination)
        assert incomplete.exists()
        assert complete.exists()
        complete.write_bytes(original_marker)
    monkeypatch.setattr(records_module, "_write_json", original_write)
    monkeypatch.setattr(LocalFileSystem, "rm", original_rm)
    reads.clear()
    evidence = run_waves(source, _START, _START, destination)
    assert evidence is not None
    assert evidence.inventory.datasets["wave_days"].rows == 1
    assert evidence.inventory.datasets["waves"].rows == 1
    assert reads == []
    daily, catalog, _ = _scientific_rows(destination)
    assert len(daily) == 1
    assert len(catalog) == 1
    assert catalog[0]["name"] == daily[0]["name"]
    assert not incomplete.exists()
    assert complete.exists()


@pytest.mark.component
def test_clean_stop_keeps_full_request_and_compressed_batches_for_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stop before writing output, inspect local state, then complete the range.

    Args:
        tmp_path: Private output and source directories.
        monkeypatch: Fixture providing finished footprints without network reads.

    Returns:
        None.

    Examples:
        ``pytest tests/test_runner.py -k clean_stop`` runs this check.

    Steps:
        1. Stop before writing output, inspect local state, then complete the range.
    """
    reads: list[date] = []
    end = _START + timedelta(days=2)
    destination = tmp_path / "beta"

    def read(
        self: WaveFootprintSource, day: date, series: WaveSeries
    ) -> tuple[ComponentMembership, ...]:
        """Supply a continuing named storm and record dates read.

        Args:
            self: Configured source.
            day: Requested date.
            series: Requested wave series.

        Returns:
            A maximum-warm component, with other series empty.

        Examples:
            Resume must only request dates after the saved checkpoint.

        Steps:
            1. Supply a continuing named storm and record dates read.
        """
        reads.append(day)
        return (_component(day, series),) if series == _SERIES[0] else ()

    monkeypatch.setattr(WaveFootprintSource, "read", read)
    source = WaveFootprintSource(str(tmp_path / "source"))
    result = run_waves(source, _START, end, destination, stop_after_days=1)
    assert result is None
    assert not destination.exists()
    assert reads == [_START]
    state = tmp_path / "beta-state"
    request = json.loads((state / "request.json").read_bytes())
    assert request["end_date"] == end.isoformat()
    batch_path = state / "batches" / f"{_START}.json.gz"
    compressed = batch_path.read_bytes()
    restored = gzip.decompress(compressed)
    assert json.loads(restored)["date"] == _START.isoformat()
    assert len(compressed) < len(restored)
    reads.clear()
    status = run_status(destination)
    assert reads == []
    assert status["through_date"] == _START.isoformat()
    assert status["next_date"] == (_START + timedelta(days=1)).isoformat()
    assert status["processed_days"] == 1
    assert status["batch_files"] == 1
    assert status["batch_bytes"] == len(compressed)
    assert status["checkpoint_bytes"] == (state / "checkpoint.json").stat().st_size
    assert status["state_bytes"] == sum(
        path.stat().st_size for path in state.rglob("*") if path.is_file()
    )
    evidence = run_waves(source, _START, end, destination, stop_after_days=2)
    assert evidence is not None
    assert evidence.inventory.datasets["wave_days"].rows == 3
    assert evidence.inventory.datasets["waves"].rows == 1
    assert all(day > _START for day in reads)
    finished = run_status(destination)
    assert finished["next_date"] is None
    assert finished["processed_days"] == 3
    assert finished["output_status"] == "complete"
    assert len(_scientific_rows(destination)[0]) == 3


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_stop_after_days_rejects_invalid_limits(tmp_path: Path, limit: object) -> None:
    """Reject invalid day limits before creating output state.

    Args:
        tmp_path: Private unused output directory.
        limit: Invalid early-stop count.

    Returns:
        None.

    Examples:
        ``pytest tests/test_runner.py -k invalid_limits`` runs this check.

    Steps:
        1. Reject invalid day limits before creating output state.
    """
    with pytest.raises(WaveRunnerError, match="positive integer"):
        run_waves(
            WaveFootprintSource(str(tmp_path / "footprints")),
            _START,
            _START,
            tmp_path / "output",
            stop_after_days=cast("int", limit),
        )


@pytest.mark.component
@pytest.mark.parametrize(
    "change", ["series", "source", "start_date", "end_date", "reversed_dates"]
)
def test_status_rejects_request_that_disagrees_with_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """Reject saved request changes instead of reporting mixed run facts.

    Args:
        tmp_path: Temporary run state and output directory.
        monkeypatch: Fixture supplying an empty footprint day.
        change: Request field changed after its checkpoint was written.

    Returns:
        None. Every mismatched request raises a runner error.

    Examples:
        ``pytest tests/test_runner.py -k disagrees_with_checkpoint`` runs this check.

    Steps:
        1. Reject saved request changes instead of reporting mixed run facts.
        2. Check that every mismatched request raises a runner error.
    """
    monkeypatch.setattr(WaveFootprintSource, "read", lambda self, day, series: ())
    output = tmp_path / "waves"
    run_waves(
        WaveFootprintSource(str(tmp_path / "footprints")),
        _START,
        _START + timedelta(days=2),
        output,
        stop_after_days=2,
    )
    request_path = tmp_path / "waves-state" / "request.json"
    request = json.loads(request_path.read_bytes())
    if change == "series":
        request["series"] = {"temperature": "tasmin", "direction": "cold"}
    elif change == "source":
        request["source"] = str(tmp_path / "other-footprints")
    elif change == "start_date":
        request["start_date"] = (_START - timedelta(days=1)).isoformat()
    elif change == "end_date":
        request["end_date"] = _START.isoformat()
    else:
        request["start_date"] = (_START + timedelta(days=3)).isoformat()
    request_path.write_text(json.dumps(request))
    before = request_path.read_bytes()
    with pytest.raises(WaveRunnerError, match=r"does not match|must not follow"):
        run_status(output)
    assert request_path.read_bytes() == before


@pytest.mark.component
def test_status_ignores_temporary_files_and_handles_disappearing_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Report saved progress while temporary checkpoint files are renamed.

    Args:
        tmp_path: Private checkpoint and output directories.
        monkeypatch: Fixture simulating replacement during the status inventory.

    Returns:
        None.

    Examples:
        ``pytest tests/test_runner.py -k disappearing_state`` runs this check.

    Steps:
        1. Report saved progress while temporary checkpoint files are renamed.
    """
    monkeypatch.setattr(WaveFootprintSource, "read", lambda self, day, series: ())
    output = tmp_path / "waves"
    run_waves(
        WaveFootprintSource(str(tmp_path / "footprints")),
        _START,
        _START + timedelta(days=1),
        output,
        stop_after_days=1,
    )
    state = tmp_path / "waves-state"
    checkpoint = state / "checkpoint.json"
    temporary = state / "checkpoint.json.tmp"
    temporary.write_bytes(checkpoint.read_bytes())
    transient = state / "writer-marker.json"
    transient.write_text("{}")
    original_stat = Path.stat
    disappeared = []

    def stat(path: Path, *, follow_symlinks: bool = True):
        """Replace the checkpoint and remove another transient inventory file.

        Args:
            path: Local path inspected by the status command.
            follow_symlinks: Whether to inspect the target of a symbolic link.

        Returns:
            Original stat result for stable paths.

        Raises:
            AssertionError: If status inspects temporary-file size.
            FileNotFoundError: If the transient marker vanishes mid-inventory.

        Examples:
            Stable checkpoint bytes must remain readable during replacement.

        Steps:
            1. Replace the checkpoint and remove another transient inventory file.
        """
        assert path != temporary
        if path == transient and not disappeared:
            temporary.replace(checkpoint)
            transient.unlink()
            disappeared.append(path)
            raise FileNotFoundError(path)
        return original_stat(path, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", stat)
    status = run_status(output)
    assert disappeared == [transient]
    assert status["through_date"] == _START.isoformat()
    assert status["next_date"] == (_START + timedelta(days=1)).isoformat()
    assert status["batch_files"] == 1
    assert status["checkpoint_bytes"] == len(checkpoint.read_bytes())
    assert status["state_bytes"] == sum(
        path.stat().st_size for path in state.rglob("*") if path.is_file()
    )


@pytest.mark.component
@pytest.mark.parametrize("series", _SERIES)
def test_each_series_runs_independently_and_resumes(tmp_path, monkeypatch, series):
    """Keep names and scientific rows identical after a single-series restart.

    Args:
        tmp_path: Separate fresh and resumed output directories.
        monkeypatch: Fixture supplying named synthetic footprints.
        series: The maximum/minimum and warm/cold series under test.

    Returns:
        None. Checks exact output rows, selected reads and final state bytes.

    Examples:
        pytest tests/test_runner.py -k each_series

    Steps:
        1. Supply the selected series with two named-storm dates followed by an empty ending date.
        2. Run uninterrupted and stopped-then-resumed outputs.
        3. Compare scientific rows, input selections, and final checkpoint bytes.
    """
    reads = []

    def read(self, day, selected):
        """Return one named storm for two days, then an empty ending day.

        Args:
            self: Configured footprint source.
            day: Requested date.
            selected: The run's single selected series.

        Returns:
            One synthetic component on the first two dates, otherwise empty.

        Examples:
            The runner calls this once per date.

        Steps:
            1. Return one named storm for two days, then an empty ending day.
        """
        reads.append((day, selected))
        return (_component(day, selected),) if day < _START + timedelta(days=2) else ()

    monkeypatch.setattr(WaveFootprintSource, "read", read)
    source = WaveFootprintSource(str(tmp_path / "source"))
    end = _START + timedelta(days=2)
    fresh = tmp_path / "fresh"
    run_waves(source, _START, end, fresh, series=series)
    assert len(reads) == 3
    assert all(selected == series for _, selected in reads)
    expected = _scientific_rows(fresh)
    assert all(
        row["temperature"] == series.temperature
        and row["direction"] == series.direction
        for table in expected
        for row in table
    )
    resumed = tmp_path / "resumed"
    run_waves(source, _START, end, resumed, stop_after_days=1, series=series)
    reads.clear()
    other = next(candidate for candidate in _SERIES if candidate != series)
    with pytest.raises(WaveRunnerError, match="series"):
        run_waves(source, _START, end, resumed, series=other)
    assert reads == []
    manifest = json.loads((fresh / "manifest.json").read_text())
    assert run_status(resumed)["series"] == manifest["series"]
    run_waves(source, _START, end, resumed, series=series)
    assert len(reads) == 2
    assert _scientific_rows(resumed) == expected
    assert (tmp_path / "fresh-state" / "checkpoint.json").read_bytes() == (
        tmp_path / "resumed-state" / "checkpoint.json"
    ).read_bytes()
    batch = json.loads(
        gzip.decompress(
            (tmp_path / "resumed-state" / "batches" / f"{_START}.json.gz").read_bytes()
        )
    )
    assert set(batch) == {"date", "current", "ended", "lineage"}


@pytest.mark.parametrize("series", [None, (), (_SERIES[0],), "tasmax/warm"])
def test_runner_rejects_invalid_series_selection(tmp_path, series):
    """Reject invalid selections before creating saved run state.

    Args:
        tmp_path: Temporary output directory.
        series: Value that is not a single WaveSeries.

    Returns:
        None.

    Examples:
        pytest tests/test_runner.py -k invalid_series

    Steps:
        1. Reject invalid selections before creating saved run state.
    """
    with pytest.raises(WaveRunnerError, match="series"):
        run_waves(
            WaveFootprintSource(str(tmp_path / "footprints")),
            _START,
            _START,
            tmp_path / "output",
            series=series,
        )
    assert not (tmp_path / "output-state").exists()


@pytest.mark.component
@pytest.mark.parametrize("series", _SERIES)
def test_completed_run_extension_matches_fresh_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, series: WaveSeries
) -> None:
    """Extend without rereading old inputs or changing names and lineage.

    Args:
        tmp_path: Separate original, extended and uninterrupted outputs.
        monkeypatch: Fixture providing finished daily components.
        series: Independent warm or cold maximum or minimum series.

    Returns:
        None. Every scientific row matches and original files remain unchanged.

    Examples:
        Run ``pytest tests/test_runner.py -k completed_run_extension``.

    Steps:
        1. Extend without rereading old inputs or changing names and lineage.
        2. Check that every scientific row matches and original files remain unchanged.
    """
    read_dates: list[date] = []

    def read(
        self: WaveFootprintSource, day: date, selected: WaveSeries
    ) -> tuple[ComponentMembership, ...]:
        """Supply a named wave, two split children, and a named successor.

        Args:
            self: Source used by each run.
            day: Requested date.
            selected: Requested independent series.

        Returns:
            Two split children on day three and no components on day four.

        Examples:
            The extension retains the named parent and records both children.

        Steps:
            1. Supply a named wave, two split children, and a named successor.
        """
        read_dates.append(day)
        template = _component(day, selected)
        if day == _START + timedelta(days=3):
            return ()
        if day != _START + timedelta(days=2):
            return (template,)
        shapes = (box(-20, -20, -1, 20), box(1, -20, 20, 20))
        return tuple(
            replace(
                template,
                component_id=component_identity(
                    temperature=selected.temperature,
                    direction=selected.direction,
                    date=np.datetime64(day, "D"),
                    geometry=shape,
                ),
                geometry=replace(
                    template.geometry,
                    label=index,
                    geometry=shape,
                    full_geometry=shape,
                    area_qualified=float(
                        transform(_AREA_TRANSFORMER.transform, shape).area / 1_000_000
                    ),
                ),
            )
            for index, shape in enumerate(shapes, 1)
        )

    monkeypatch.setattr(WaveFootprintSource, "read", read)
    source = WaveFootprintSource(str(tmp_path / "inputs"))
    original, extended, fresh = (tmp_path / name for name in ("old", "new", "fresh"))
    old_end, end = _START + timedelta(days=1), _START + timedelta(days=4)
    run_waves(source, _START, old_end, original, series=series)
    original_files = {
        path: path.read_bytes()
        for root in (original, tmp_path / "old-state")
        for path in root.rglob("*")
        if path.is_file()
    }
    read_dates.clear()
    assert (
        run_waves(
            source,
            _START,
            end,
            extended,
            series=series,
            extend_from=original,
            stop_after_days=1,
        )
        is None
    )
    assert read_dates == [old_end + timedelta(days=1)]
    run_waves(source, _START, end, extended, series=series, extend_from=original)
    assert read_dates == [old_end + timedelta(days=i) for i in range(1, 4)]
    run_waves(source, _START, end, fresh, series=series)
    rows = _scientific_rows(extended)
    assert rows == _scientific_rows(fresh)
    assert rows[2]  # The extension records nonempty split lineage.
    assert all(
        path.read_bytes() == contents for path, contents in original_files.items()
    )
    assert (tmp_path / "new-state" / "checkpoint.json").read_bytes() == (
        tmp_path / "fresh-state" / "checkpoint.json"
    ).read_bytes()


@pytest.mark.component
@pytest.mark.parametrize(
    "damage", ["missing_batch", "marker", "source", "end", "final_records"]
)
def test_extension_rejects_invalid_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    """Reject incomplete history and altered requests before continuation.

    Args:
        tmp_path: Separate original and proposed extension directories.
        monkeypatch: Fixture supplying a valid finished component.
        damage: Missing batch, inconsistent marker, changed source or end date.

    Returns:
        None. Invalid extension raises without committing a new checkpoint.

    Examples:
        Run ``pytest tests/test_runner.py -k extension_rejects``.

    Steps:
        1. Reject incomplete history and altered requests before continuation.
        2. Check that invalid extension raises without committing a new checkpoint.
    """
    monkeypatch.setattr(
        WaveFootprintSource,
        "read",
        lambda self, day, selected: (_component(day, selected),),
    )
    source = WaveFootprintSource(str(tmp_path / "inputs"))
    original = tmp_path / "old"
    run_waves(source, _START, _START, original)
    if damage == "missing_batch":
        (tmp_path / "old-state" / "batches" / f"{_START}.json.gz").unlink()
    elif damage == "final_records":
        path = tmp_path / "old-state" / "batches" / f"{_START}.json.gz"
        payload = json.loads(gzip.decompress(path.read_bytes()))
        payload["current"] = []
        path.write_bytes(gzip.compress(json.dumps(payload).encode()))
    elif damage == "marker":
        marker = original / "_COMPLETE.json"
        payload = json.loads(marker.read_bytes())
        payload["generation_id"] = "wrong"
        marker.write_text(json.dumps(payload))
    elif damage == "source":
        source = WaveFootprintSource(str(tmp_path / "other-inputs"))
    end = _START if damage == "end" else _START + timedelta(days=1)
    with pytest.raises(ValueError, match=r"extension|generation_id"):
        run_waves(source, _START, end, tmp_path / "new", extend_from=original)
    assert not (tmp_path / "new-state" / "checkpoint.json").exists()
