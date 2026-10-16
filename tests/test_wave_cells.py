"""Exercise wave-cell inputs, exact daily geometry and shared runner outputs."""

import json
from datetime import date, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from pyproj import Transformer
from shapely.geometry import Point, Polygon, box
from shapely.ops import transform

from reuters_temperature_storms.cells import WaveCellError, WaveCells, WaveCellSource
from reuters_temperature_storms.cli import main
from reuters_temperature_storms.footprints import (
    FootprintError,
    WaveFootprints,
    WaveFootprintSource,
)
from reuters_temperature_storms.records import (
    open_completed_generation,
    scan_lineage,
    scan_wave_days,
    scan_waves,
)
from reuters_temperature_storms.runner import WaveRunnerError, run_waves
from reuters_temperature_storms.series import WaveSeries
from tests.fixtures import create_wave_cells

DAY = date(2024, 7, 1)
SERIES = WaveSeries("tasmax", "warm")
AREA_TRANSFORMER = Transformer.from_crs(4326, 6933, always_xy=True)


def _frame(cells=(), *, day=DAY):
    """Create exact cell polygons and canonical state columns.

    Args:
        cells: Sequence of longitude, latitude and qualified/grace/fill roles.
        day: Represented calendar date.

    Returns:
        Typed seven-column geographic frame, including paired dateline pieces.

    Examples:
        frame = _frame(((0, 0, "qualified"), (0.25, 0, "grace")))

    Steps:
        1. Create exact cell polygons and canonical state columns.
    """
    rows = []
    shapes = []
    for longitude, latitude, role in cells:
        bottom, top = max(-90, latitude - 0.125), min(90, latitude + 0.125)
        pieces = (
            (box(179.875, bottom, 180, top), box(-180, bottom, -179.875, top))
            if longitude == 180
            else (box(longitude - 0.125, bottom, longitude + 0.125, top),)
        )
        for shape in pieces:
            rows.append(
                {
                    "longitude": longitude,
                    "latitude": latitude,
                    "days": np.nan if role == "fill" else 5,
                    "grace_days": np.nan if role == "fill" else int(role == "grace"),
                    "gap_filled": role == "fill",
                }
            )
            shapes.append(shape)
    return gpd.GeoDataFrame(
        {
            "longitude": pd.Series([row["longitude"] for row in rows], dtype="float64"),
            "latitude": pd.Series([row["latitude"] for row in rows], dtype="float64"),
            "days": pd.Series([row["days"] for row in rows], dtype="float64"),
            "grace_days": pd.Series(
                [row["grace_days"] for row in rows], dtype="float64"
            ),
            "gap_filled": pd.Series([row["gap_filled"] for row in rows], dtype="bool"),
            "geometry": gpd.GeoSeries(shapes, crs=4326),
            "date": pd.Series([day] * len(rows), dtype="datetime64[ns]"),
        },
        geometry="geometry",
        crs=4326,
    )


def _area(shape):
    """Measure a supplied independent shape with the EPSG:6933 projection.

    Args:
        shape: Geographic shape whose expected area is needed.

    Returns:
        Square-kilometre area.

    Examples:
        area = _area(box(-0.125, -0.125, 0.125, 0.125))

    Steps:
        1. Project the geographic polygon into the equal-area coordinate system.
        2. Divide its square-meter area by one million.
    """
    return transform(AREA_TRANSFORMER.transform, shape).area / 1_000_000


def _write(root, frame, *, day=DAY, series=SERIES):
    """Write cells under the selected field and direction directories.

    Args:
        root: Independent input root.
        frame: Accepted cells for the requested day.
        day: Filename date.
        series: Requested temperature and direction.

    Returns:
        Daily input path.

    Examples:
        path = _write(tmp_path, _frame(((0, 0, "qualified"),)))

    Steps:
        1. Write cells under the selected field and direction directories.
    """
    field = "t2m_max" if series.temperature == "tasmax" else "t2m_min"
    direction = "positive" if series.direction == "warm" else "negative"
    path = root / field / direction / f"{day}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    WaveCells(frame).to_parquet(path)
    return path


def test_roles_shapes_and_area_are_preserved():
    """Keep real and gap-fill roles separate while connecting the full extent.

    Args:
        None.

    Returns:
        None. Role shapes, measured areas and fraction match independent cells.

    Examples:
        pytest tests/test_wave_cells.py -k roles_shapes

    Steps:
        1. Keep real and gap-fill roles separate while connecting the full extent.
        2. Check that role shapes, measured areas and fraction match independent cells.
    """
    frame = _frame(((0, 0, "qualified"), (0.25, 0, "grace"), (0.5, 0, "fill")))
    row = WaveFootprints.from_cells(WaveCells(frame), DAY, SERIES).frame.iloc[0]
    assert row.geometry.equals(frame.geometry.iloc[0])
    assert row.grace_geometry.equals(frame.geometry.iloc[1])
    assert row.buffer_geometry.equals(box(-0.125, -0.125, 0.625, 0.125))
    for column, shape in zip(
        ("area_qualified", "area_grace", "area_gap_filled"),
        frame.geometry,
        strict=True,
    ):
        assert row[column] == pytest.approx(_area(shape), rel=1e-12)
    assert row.qualified_fraction == pytest.approx(1 / 3)


def test_fill_bridge_connects_real_cells_but_fill_only_is_dropped():
    """Use fill for connectivity without creating waves from fill-only groups.

    Args:
        None.

    Returns:
        None. One connected real footprint remains.

    Examples:
        pytest tests/test_wave_cells.py -k fill_bridge

    Steps:
        1. Use fill for connectivity without creating waves from fill-only groups.
        2. Check that one connected real footprint remains.
    """
    cells = WaveCells(
        _frame(
            (
                (0, 0, "qualified"),
                (0.25, 0, "fill"),
                (0.5, 0, "qualified"),
                (5, 0, "fill"),
            )
        )
    )
    frame = WaveFootprints.from_cells(cells, DAY, SERIES).frame
    assert len(frame) == 1
    assert frame.buffer_geometry.iloc[0].equals(box(-0.125, -0.125, 0.625, 0.125))
    assert frame.area_gap_filled.iloc[0] > 0
    assert len(frame.geometry.iloc[0].geoms) == 2


def test_grace_only_is_retained_and_corner_contact_is_separate():
    """Preserve grace-only waves and require edge contact for connectivity.

    Args:
        None.

    Returns:
        None. Both corner-touching waves remain distinct.

    Examples:
        pytest tests/test_wave_cells.py -k grace_only

    Steps:
        1. Preserve grace-only waves and require edge contact for connectivity.
        2. Check that both corner-touching waves remain distinct.
    """
    cells = WaveCells(_frame(((0, 0, "grace"), (0.25, 0.25, "qualified"))))
    frame = WaveFootprints.from_cells(cells, DAY, SERIES).frame
    assert len(frame) == 2
    grace = frame[frame.area_grace > 0].iloc[0]
    assert grace.geometry is None
    assert grace.qualified_fraction == 0
    assert grace.grace_geometry.equals(box(-0.125, -0.125, 0.125, 0.125))


@pytest.mark.parametrize("latitude", [0, -90, 90])
def test_paired_dateline_cell_has_exact_extent_and_area(latitude):
    """Keep both halves of a dateline cell, including clipped pole cells.

    Args:
        latitude: Equatorial or pole cell centre.

    Returns:
        None. One wave contains both halves and conserves their combined area.

    Examples:
        pytest tests/test_wave_cells.py -k paired_dateline

    Steps:
        1. Keep both halves of a dateline cell, including clipped pole cells.
        2. Check that one wave contains both halves and conserves their combined area.
    """
    cells = WaveCells(_frame(((180, latitude, "qualified"),)))
    frame = WaveFootprints.from_cells(cells, DAY, SERIES).frame
    assert len(frame) == 1
    assert frame.buffer_geometry.iloc[0].equals(cells.frame.geometry.union_all())
    assert frame.area_qualified.iloc[0] == pytest.approx(
        sum(_area(shape) for shape in cells.frame.geometry), rel=1e-12
    )


@pytest.mark.parametrize("offset", [0.25, 0.5])
def test_dateline_corner_or_distinct_latitude_does_not_join(offset):
    """Keep waves separate when opposite dateline edges do not overlap.

    Args:
        offset: Latitude offset producing corner contact or no contact.

    Returns:
        None. Two independent waves remain.

    Examples:
        pytest tests/test_wave_cells.py -k dateline_corner

    Steps:
        1. Keep waves separate when opposite dateline edges do not overlap.
        2. Check that two independent waves remain.
    """
    cells = WaveCells(_frame(((179.75, 0, "qualified"), (180, offset, "qualified"))))
    frame = WaveFootprints.from_cells(cells, DAY, SERIES).frame
    assert len(frame) == 2


def test_dateline_neighbors_join_and_role_area_is_conserved():
    """Join neighbors through paired edge pieces without changing role areas.

    Args:
        None.

    Returns:
        None. One three-cell wave retains qualified and grace measures.

    Examples:
        pytest tests/test_wave_cells.py -k dateline_neighbors

    Steps:
        1. Join neighbors through paired edge pieces without changing role areas.
        2. Check that one three-cell wave retains qualified and grace measures.
    """
    cells = WaveCells(
        _frame(((179.75, 0, "qualified"), (180, 0, "grace"), (-179.75, 0, "qualified")))
    )
    row = WaveFootprints.from_cells(cells, DAY, SERIES).frame.iloc[0]
    assert row.buffer_geometry.equals(cells.frame.geometry.union_all())
    assert row.area_qualified == pytest.approx(
        sum(
            _area(shape)
            for shape in cells.frame.loc[~cells.frame.grace_days.gt(0), "geometry"]
        ),
        rel=1e-12,
    )
    assert row.area_grace == pytest.approx(
        sum(
            _area(shape)
            for shape in cells.frame.loc[cells.frame.grace_days.gt(0), "geometry"]
        ),
        rel=1e-12,
    )


def test_full_longitude_band_and_shuffle_preserve_wkb_and_hash():
    """Keep world-width geometry and hashes stable when source rows are shuffled.

    Args:
        None.

    Returns:
        None. Exact stored geometries and IDs match for every footprint role.

    Examples:
        pytest tests/test_wave_cells.py -k full_longitude

    Steps:
        1. Keep world-width geometry and hashes stable when source rows are shuffled.
        2. Check that exact stored geometries and IDs match for every footprint role.
    """
    frame = _frame(
        tuple(
            (float(longitude), 0, "grace" if index % 3 == 0 else "qualified")
            for index, longitude in enumerate(np.arange(-179.75, 180.25, 0.25))
        )
    )
    baseline = WaveFootprints.from_cells(WaveCells(frame), DAY, SERIES).frame
    shuffled = WaveFootprints.from_cells(
        WaveCells(frame.sample(frac=1, random_state=42).reset_index(drop=True)),
        DAY,
        SERIES,
    ).frame
    assert len(baseline) == 1
    assert baseline.buffer_geometry.iloc[0].equals(box(-180, -0.125, 180, 0.125))
    assert shuffled.storm_id.tolist() == baseline.storm_id.tolist()
    for column in ("geometry", "grace_geometry", "buffer_geometry"):
        assert shuffled[column].to_wkb().tolist() == baseline[column].to_wkb().tolist()
    for column in (
        "area_qualified",
        "area_grace",
        "area_gap_filled",
        "qualified_fraction",
    ):
        assert shuffled[column].tolist() == baseline[column].tolist()


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("longitude", 0.1),
        ("longitude", -180.0),
        ("longitude", np.nan),
        ("latitude", 90.25),
        ("latitude", np.inf),
        ("days", 4.0),
        ("days", 5.5),
        ("days", np.nan),
        ("grace_days", 2.0),
        ("grace_days", -1.0),
        ("grace_days", np.nan),
        ("gap_filled", 1.0),
        ("gap_filled", None),
        ("geometry", Point(0, 0)),
        ("geometry", box(-0.125, -0.125, 0.126, 0.125)),
        ("geometry", Polygon([(0, 0), (1, 1), (0, 1), (1, 0), (0, 0)])),
        ("date", pd.Timestamp(DAY) + pd.Timedelta(nanoseconds=1)),
    ],
)
def test_invalid_cell_values_are_rejected(column, value):
    """Reject malformed grid coordinates, role state, dates and geometry.

    Args:
        column: Input column to alter.
        value: Invalid replacement value.

    Returns:
        None when the table rejects the malformed input.

    Examples:
        pytest tests/test_wave_cells.py -k invalid_cell_values

    Steps:
        1. Reject malformed grid coordinates, role state, dates and geometry.
    """
    frame = _frame(((0, 0, "qualified"),))
    frame[column] = [value]
    with pytest.raises(WaveCellError):
        WaveCells(frame)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "wrong_crs",
        "missing_column",
        "extra_column",
        "missing_seam_piece",
        "seam_state",
        "filled_counts",
        "integer_flag",
        "bool_count",
        "string_date",
    ],
)
def test_invalid_table_relationships_are_rejected(mutation):
    """Reject incomplete seam cells and conflicting table structure or state.

    Args:
        mutation: Structural input error to introduce.

    Returns:
        None when the strict table boundary rejects the input.

    Examples:
        pytest tests/test_wave_cells.py -k invalid_table_relationships

    Steps:
        1. Reject incomplete seam cells and conflicting table structure or state.
    """
    frame = _frame(((180, 0, "qualified"),))
    if mutation == "duplicate":
        frame = pd.concat([frame, frame], ignore_index=True)
    elif mutation == "wrong_crs":
        frame = frame.set_crs(3857, allow_override=True)
    elif mutation == "missing_column":
        frame = frame.drop(columns="days")
    elif mutation == "extra_column":
        frame["score"] = 3.0
    elif mutation == "missing_seam_piece":
        frame = frame.iloc[:1]
    elif mutation == "seam_state":
        frame.loc[1, "grace_days"] = 1
    elif mutation == "filled_counts":
        frame["gap_filled"] = True
    elif mutation == "integer_flag":
        frame["gap_filled"] = frame.gap_filled.astype("int64")
    elif mutation == "bool_count":
        frame["days"] = True
    else:
        frame["date"] = str(DAY)
    with pytest.raises(WaveCellError):
        WaveCells(frame)


def test_requested_day_must_match_input():
    """Reject a correctly typed cell file represented under the wrong date.

    Args:
        None.

    Returns:
        None when footprint construction rejects the date mismatch.

    Examples:
        pytest tests/test_wave_cells.py -k requested_day

    Steps:
        1. Reject a correctly typed cell file represented under the wrong date.
    """
    with pytest.raises(FootprintError, match="dates must all equal"):
        WaveFootprints.from_cells(
            WaveCells(_frame(((0, 0, "qualified"),))), DAY + timedelta(days=1), SERIES
        )


def test_reader_selects_required_columns_and_roundtrips_exact_geometry(tmp_path):
    """Accept optional extras at file input and preserve canonical geometry.

    Args:
        tmp_path: Local input directory.

    Returns:
        None. Extra unused scores are ignored and table I/O preserves cells.

    Examples:
        pytest tests/test_wave_cells.py -k selects_required

    Steps:
        1. Accept optional extras at file input and preserve canonical geometry.
        2. Check that extra unused scores are ignored and table I/O preserves cells.
    """
    frame = _frame(((0, 0, "qualified"), (180, 0, "grace")))
    stored = frame.assign(score=4.5, cell_area=123.0, qualification=True)
    path = tmp_path / "extended.parquet"
    stored.to_parquet(path, index=False)
    cells = WaveCells.read_parquet(path)
    assert cells.frame.equals(frame)
    cells.to_parquet(tmp_path / "roundtrip.parquet")
    assert WaveCells.read_parquet(tmp_path / "roundtrip.parquet").frame.equals(frame)


def test_empty_stored_flag_is_prepared_only_by_reader(tmp_path):
    """Accept a stored empty float flag without relaxing nonempty inputs.

    Args:
        tmp_path: Local input directory.

    Returns:
        None. The reader prepares the empty dtype; direct tables remain strict.

    Examples:
        pytest tests/test_wave_cells.py -k empty_stored

    Steps:
        1. Accept a stored empty float flag without relaxing nonempty inputs.
        2. Check that the reader prepares the empty dtype; direct tables remain strict.
    """
    frame = _frame()
    frame["gap_filled"] = pd.Series([], dtype="float64")
    with pytest.raises(WaveCellError):
        WaveCells(frame)
    path = tmp_path / "empty.parquet"
    frame.to_parquet(path, index=False)
    cells = WaveCells.read_parquet(path)
    assert cells.frame.gap_filled.dtype == np.dtype(bool)
    assert WaveFootprints.from_cells(cells, DAY, SERIES).frame.empty
    frame = _frame(((0, 0, "qualified"),))
    frame["gap_filled"] = frame.gap_filled.astype("float64")
    frame.to_parquet(path, index=False)
    with pytest.raises(WaveCellError):
        WaveCells.read_parquet(path)


@pytest.mark.parametrize(
    "series",
    [
        WaveSeries(temperature, direction)
        for temperature in ("tasmax", "tasmin")
        for direction in ("warm", "cold")
    ],
)
def test_source_selects_all_series_and_missing_dates_are_explicit(tmp_path, series):
    """Read only the selected series and distinguish missing dates from empty.

    Args:
        tmp_path: Source root for synthetic cells.
        series: Each supported temperature and direction pair.

    Returns:
        None. Exact components match table construction and missing dates fail.

    Examples:
        pytest tests/test_wave_cells.py -k source_selects

    Steps:
        1. Read only the selected series and distinguish missing dates from empty.
        2. Check that exact components match table construction and missing dates fail.
    """
    frame = _frame(((0, 0, "qualified"),))
    path = _write(tmp_path, frame, series=series)
    source = WaveCellSource(str(tmp_path))
    assert source.read(DAY, series) == WaveFootprints.from_cells(
        WaveCells(frame), DAY, series
    ).to_components(DAY, series)
    assert source.list_files(series) == {path.name: path.stat().st_size}
    missing_day = DAY + timedelta(days=1)
    with pytest.raises(FileNotFoundError):
        source.read(missing_day, series)
    assert (
        WaveCellSource(str(tmp_path), known_empty_dates=(missing_day,)).read(
            missing_day, series
        )
        == ()
    )
    with pytest.raises(WaveCellError, match="declared empty"):
        WaveCellSource(str(tmp_path), known_empty_dates=(DAY,)).read(DAY, series)
    _write(tmp_path, _frame((), day=missing_day), day=missing_day, series=series)
    assert source.read(missing_day, series) == ()


def test_cell_schema_runs_once_per_read_and_not_again_in_construction(
    tmp_path, monkeypatch
):
    """Avoid a duplicate cell validation pass between reading and construction.

    Args:
        tmp_path: Input directory.
        monkeypatch: Fixture counting calls to the table's Pandera schema.

    Returns:
        None. Each explicit cell-table boundary validates exactly once.

    Examples:
        pytest tests/test_wave_cells.py -k schema_runs_once

    Steps:
        1. Avoid a duplicate cell validation pass between reading and construction.
        2. Check that each explicit cell-table boundary validates exactly once.
    """
    frame = _frame(((0, 0, "qualified"),))
    path = _write(tmp_path, frame)
    schema = WaveCells.schema
    schema.to_schema()
    original = schema.validate
    calls = []

    def counted(cls, supplied, *args, **kwargs):
        """Count cell schema validation while retaining its behavior.

        Args:
            cls: Table schema.
            supplied: Cell frame.
            *args: Existing positional options.
            **kwargs: Existing keyword options.

        Returns:
            Original schema result.

        Examples:
            Called by WaveCells.read_parquet.

        Steps:
            1. Count cell schema validation while retaining its behavior.
        """
        calls.append(supplied)
        return original(supplied, *args, **kwargs)

    monkeypatch.setattr(schema, "validate", classmethod(counted))
    cells = WaveCells.read_parquet(path)
    assert len(calls) == 1
    footprints = WaveFootprints.from_cells(cells, DAY, SERIES)
    footprints.to_components(DAY, SERIES)
    assert len(calls) == 1
    assert WaveCells(frame).frame is frame
    assert len(calls) == 2


def test_named_histories_match_footprint_input_and_cell_restart(tmp_path):
    """Run a named split/merge wave identically from both inputs and on restart.

    Args:
        tmp_path: Independent input, output and restart directories.

    Returns:
        None. Names, daily histories, lineage, catalogs and checkpoints match.

    Examples:
        pytest tests/test_wave_cells.py -k named_histories

    Steps:
        1. Run a named split/merge wave identically from both inputs and on restart.
        2. Check that names, daily histories, lineage, catalogs and checkpoints match.
    """
    cell_root, footprint_root = tmp_path / "cells", tmp_path / "footprints"
    for offset in range(4):
        day = DAY + timedelta(days=offset)
        selected = tuple(
            (longitude / 4, latitude / 4, "qualified")
            for longitude in range(-48, 48)
            for latitude in range(-40, 40)
            if offset < 3 and (offset != 1 or abs(longitude) > 1)
        )
        frame = _frame(selected, day=day)
        _write(cell_root, frame, day=day)
        footprint_path = footprint_root / "t2m_max" / "positive" / f"{day}.parquet"
        footprint_path.parent.mkdir(parents=True, exist_ok=True)
        WaveFootprints.from_cells(WaveCells(frame), day, SERIES).to_parquet(
            footprint_path
        )
    end = DAY + timedelta(days=3)
    cells = WaveCellSource(str(cell_root))
    destinations = [
        tmp_path / name for name in ("from-cells", "from-footprints", "resumed")
    ]
    run_waves(cells, DAY, end, destinations[0], series=SERIES)
    run_waves(
        WaveFootprintSource(str(footprint_root)),
        DAY,
        end,
        destinations[1],
        series=SERIES,
    )
    assert (
        run_waves(cells, DAY, end, destinations[2], series=SERIES, stop_after_days=2)
        is None
    )
    with pytest.raises(WaveRunnerError, match="restart input kind"):
        run_waves(
            WaveFootprintSource(str(cell_root)),
            DAY,
            end,
            destinations[2],
            series=SERIES,
        )
    run_waves(cells, DAY, end, destinations[2], series=SERIES)
    expected = open_completed_generation(destinations[0])
    catalog = scan_waves(expected)
    assert catalog.name.notna().any()
    assert {"split", "merge"} <= set(scan_lineage(expected).relationship)
    for destination in destinations[1:]:
        generation = open_completed_generation(destination)
        for scan in (scan_wave_days, scan_waves, scan_lineage):
            assert scan(generation).equals(scan(expected))
        for filename in ("all_storms.parquet", "all_waves.parquet"):
            assert gpd.read_parquet(destination / filename).equals(
                gpd.read_parquet(destinations[0] / filename)
            )
        actual_checkpoint = json.loads(
            destination.with_name(destination.name + "-state")
            .joinpath("checkpoint.json")
            .read_bytes()
        )
        expected_checkpoint = json.loads(
            destinations[0]
            .with_name(destinations[0].name + "-state")
            .joinpath("checkpoint.json")
            .read_bytes()
        )
        # Different source roots intentionally retain different request identity.
        if destination == destinations[1]:
            actual_checkpoint.pop("source_identity")
            expected_checkpoint.pop("source_identity")
        assert actual_checkpoint == expected_checkpoint
        if destination == destinations[2]:
            assert (
                destination.with_name(destination.name + "-state")
                .joinpath("checkpoint.json")
                .read_bytes()
                == destinations[0]
                .with_name(destinations[0].name + "-state")
                .joinpath("checkpoint.json")
                .read_bytes()
            )


def test_synthetic_wave_cells_run_through_cli(tmp_path):
    """Create synthetic daily files and process them with the public CLI.

    Args:
        tmp_path: Independent synthetic input and result directories.

    Returns:
        None. The fixture creates five files and CLI emits complete wave catalogs.

    Examples:
        pytest tests/test_wave_cells.py -k synthetic_wave_cells

    Steps:
        1. Create synthetic daily files and process them with the public CLI.
        2. Check that the fixture creates five files and CLI emits complete wave catalogs.
    """
    inputs, output = tmp_path / "inputs", tmp_path / "results"
    files = create_wave_cells(inputs)
    assert len(files) == 5
    assert len(list(inputs.rglob("*.parquet"))) == 5
    result = CliRunner().invoke(
        main,
        [
            "run",
            "--wave-cells",
            str(inputs),
            "--series",
            "tasmax/warm",
            "--start-date",
            "2020-01-01",
            "--end-date",
            "2020-01-05",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Completed" in result.output
    assert (output / "_COMPLETE.json").exists()
    assert gpd.read_parquet(output / "all_storms.parquet").empty
    assert len(gpd.read_parquet(output / "all_waves.parquet")) == 1
    assert len(scan_wave_days(open_completed_generation(output))) == 4


def test_empty_cells_require_geographic_crs():
    """Reject an empty table that would conceal an invalid input CRS.

    Args:
        None.

    Returns:
        None. The constructor rejects wrong geographic metadata even without rows.

    Examples:
        Run ``pytest tests/test_wave_cells.py -k empty_cells_require``.

    Steps:
        1. Reject an empty table that would conceal an invalid input CRS.
        2. Check that the constructor rejects wrong geographic metadata even without rows.
    """
    with pytest.raises(WaveCellError):
        WaveCells(_frame().set_crs(3857, allow_override=True))


@pytest.mark.parametrize(
    "mutation", ["cells", "series_type", "series_value", "day_string", "day_timestamp"]
)
def test_from_cells_rejects_invalid_request_even_for_empty_input(mutation):
    """Require typed input models and plain dates before empty-day processing.

    Args:
        mutation: Invalid cell, series or day argument to introduce.

    Returns:
        None. Each invalid public call raises a descriptive footprint error.

    Examples:
        pytest tests/test_wave_cells.py -k invalid_request

    Steps:
        1. Require typed input models and plain dates before empty-day processing.
        2. Check that each invalid public call raises a descriptive footprint error.
    """
    cells, day, series = WaveCells(_frame()), DAY, SERIES
    if mutation == "cells":
        cells = _frame()
    elif mutation == "series_type":
        series = "tasmax/warm"
    elif mutation == "series_value":
        series = WaveSeries("tasmean", "warm")
    elif mutation == "day_string":
        day = str(DAY)
    else:
        day = pd.Timestamp(DAY)
    with pytest.raises(FootprintError):
        WaveFootprints.from_cells(cells, day, series)
