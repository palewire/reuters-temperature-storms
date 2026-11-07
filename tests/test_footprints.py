"""Tests for exact stored footprint inputs and explicitly empty dates."""

import json
from datetime import UTC, date, datetime

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyproj import CRS
from shapely.geometry import Point, Polygon, box

from reuters_temperature_storms._wave_tracking import WaveSeries, _area_km2
from reuters_temperature_storms.cells import WaveCells, WaveCellSource
from reuters_temperature_storms.footprints import (
    FootprintError,
    WaveFootprints,
    WaveFootprintSource,
)
from reuters_temperature_storms.identity import component_identity
from tests.fixtures import _native_footprint_cells

DAY = date(2026, 8, 31)
SERIES = WaveSeries("tasmax", "warm")


def _table(qualified=None, grace=None):
    """Build one valid stored summary without invoking cell construction.

    Args:
        qualified: Optional qualifying shape; the default is a simple box.
        grace: Optional grace-only shape.

    Returns:
        One-row Arrow table with all required summary columns and geo metadata.

    Examples:
        table = _table()

    Steps:
        1. Build one valid stored summary without invoking cell construction.
    """
    if qualified is None and grace is None:
        qualified = box(-0.125, -0.125, 0.875, 0.875)
    full = (
        qualified
        if grace is None
        else grace
        if qualified is None
        else qualified.union(grace)
    )
    areas = (_area_km2(qualified), _area_km2(grace), 0.0)
    table = pa.table(
        {
            "storm_id": [
                component_identity(
                    temperature=SERIES.temperature,
                    direction=SERIES.direction,
                    date=np.datetime64(DAY),
                    geometry=full,
                )
            ],
            "date": pa.array([DAY], type=pa.date32()),
            "geometry": pa.array(
                [None if qualified is None else qualified.wkb], type=pa.binary()
            ),
            "grace_geometry": pa.array(
                [None if grace is None else grace.wkb], type=pa.binary()
            ),
            "buffer_geometry": [full.wkb],
            "area_qualified": [areas[0]],
            "area_grace": [areas[1]],
            "area_gap_filled": [areas[2]],
            "qualified_fraction": [areas[0] / sum(areas)],
        }
    )
    metadata = {
        "version": "1.0.0",
        "primary_column": "geometry",
        "columns": {
            name: {"encoding": "WKB", "crs": CRS.from_epsg(4326).to_json_dict()}
            for name in ("geometry", "grace_geometry", "buffer_geometry")
        },
    }
    return table.replace_schema_metadata({b"geo": json.dumps(metadata).encode()})


def _cells(*, day=DAY, offsets=(0,), grace=False):
    """Construct explicit native companions for the test's simple rectangles.

    Args:
        day: Calendar date represented by every cell.
        offsets: Western cell centres of the separated one-degree rectangles.
        grace: Whether every real cell is in its grace period.

    Returns:
        Complete native cells for the stated synthetic rectangles and roles.

    Examples:
        cells = _cells(offsets=(0, 2, 4))
    """
    frame = _native_footprint_cells(
        day,
        tuple(box(offset - 0.125, -0.125, offset + 0.875, 0.875) for offset in offsets),
    )
    if grace:
        frame["grace_days"] = 1.0
    return WaveCells(frame)


def _source(tmp_path, *, known_empty_dates=()):
    """Pair the synthetic footprint root with its explicit native cell directory.

    Args:
        tmp_path: Root containing the synthetic footprint and native-cell files.
        known_empty_dates: Plain dates explicitly permitted to lack empty files.

    Returns:
        Prepared-footprint reader with a complete native-cell companion.

    Examples:
        components = _source(tmp_path).read(DAY, SERIES)
    """
    return WaveFootprintSource(
        str(tmp_path),
        known_empty_dates=known_empty_dates,
        cell_source=WaveCellSource(
            str(tmp_path / "native-cells"), known_empty_dates=known_empty_dates
        ),
    )


def _write(
    tmp_path,
    table,
    *,
    day=DAY,
    field="t2m_max",
    direction="positive",
    native_cells=None,
):
    """Write one synthetic daily footprint file in the field/direction layout.

    Args:
        tmp_path: Local test root.
        table: Arrow summary table.
        day: Date used in the filename.
        field: Temperature field directory.
        direction: Direction directory.
        native_cells: Explicit cell companion, or the default qualifying rectangle.

    Returns:
        Path to the newly written Parquet file.

    Examples:
        path = _write(tmp_path, _table())

    Steps:
        1. Write one synthetic daily footprint file in the field/direction layout.
    """
    path = tmp_path / field / direction / f"{day.isoformat()}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    cells_path = tmp_path / "native-cells" / field / direction / path.name
    (_cells(day=day) if native_cells is None else native_cells).to_parquet(cells_path)
    return path


def _replace(table, name, values, data_type=None):
    """Replace one stored column while preserving the GeoParquet metadata.

    Args:
        table: Original Arrow table.
        name: Column to replace.
        values: Replacement values.
        data_type: Optional explicit replacement Arrow type.

    Returns:
        Table with that column replaced.

    Examples:
        invalid = _replace(table, "area_grace", [-1.0])

    Steps:
        1. Replace one stored column while preserving the GeoParquet metadata.
    """
    index = table.schema.get_field_index(name)
    return table.set_column(index, name, pa.array(values, type=data_type))


def test_reader_preserves_shapes_ids_and_scalars(tmp_path):
    """Verify reader preserves shapes ids and scalars.

    Args:
        tmp_path: Pytest-owned temporary directory for input and output files.

    Returns:
        None.

    Steps:
        1. Check that reader preserves shapes ids and scalars.

    Examples:
        Run ``pytest tests/test_footprints.py -k reader_preserves_shapes_ids_and_scalars``.
    """
    table = _table()
    path = _write(tmp_path, table)
    source = _source(tmp_path)
    (component,) = source.read(DAY, SERIES)
    assert component.component_id == table["storm_id"][0].as_py()
    assert component.geometry.full_geometry.wkb == table["buffer_geometry"][0].as_py()
    assert component.geometry.geometry.wkb == table["geometry"][0].as_py()
    assert component.geometry.area_qualified == table["area_qualified"][0].as_py()
    assert source.list_files(SERIES) == {path.name: path.stat().st_size}


def test_grace_only_footprint_is_preserved(tmp_path):
    """Verify grace only footprint is preserved.

    Args:
        tmp_path: Pytest-owned temporary directory for input and output files.

    Returns:
        None.

    Steps:
        1. Check that grace only footprint is preserved.

    Examples:
        Run ``pytest tests/test_footprints.py -k grace_only_footprint_is_preserved``.
    """
    _write(
        tmp_path,
        _table(grace=box(-0.125, -0.125, 0.875, 0.875)),
        native_cells=_cells(grace=True),
    )
    (component,) = _source(tmp_path).read(DAY, SERIES)
    assert component.geometry.geometry is None
    assert component.geometry.grace_geometry is not None
    assert component.geometry.qualified_fraction == 0


def test_reader_accepts_omitted_zero_grace_column(tmp_path):
    """Verify reader accepts omitted zero grace column.

    Args:
        tmp_path: Pytest-owned temporary directory for input and output files.

    Returns:
        None.

    Steps:
        1. Check that reader accepts omitted zero grace column.

    Examples:
        Run ``pytest tests/test_footprints.py -k reader_accepts_omitted_zero_grace_column``.
    """
    _write(tmp_path, _table().drop_columns("grace_geometry"))
    (component,) = _source(tmp_path).read(DAY, SERIES)
    assert component.geometry.grace_geometry is None


def test_missing_grace_shape_with_positive_area_is_rejected(tmp_path):
    """Verify missing grace shape with positive area is rejected.

    Args:
        tmp_path: Pytest-owned temporary directory for input and output files.

    Returns:
        None.

    Steps:
        1. Check that missing grace shape with positive area is rejected.

    Examples:
        Run ``pytest tests/test_footprints.py -k missing_grace_shape_with_positive_area_is_rejected``.
    """
    _write(
        tmp_path,
        _table(grace=box(-0.125, -0.125, 0.875, 0.875)).drop_columns("grace_geometry"),
    )
    with pytest.raises(FootprintError, match="grace"):
        _source(tmp_path).read(DAY, SERIES)


def test_missing_day_requires_explicit_empty_declaration(tmp_path):
    """Verify missing day requires explicit empty declaration.

    Args:
        tmp_path: Pytest-owned temporary directory for input and output files.

    Returns:
        None.

    Steps:
        1. Check that missing day requires explicit empty declaration.

    Examples:
        Run ``pytest tests/test_footprints.py -k missing_day_requires_explicit_empty_declaration``.
    """
    with pytest.raises(FileNotFoundError):
        _source(tmp_path).read(DAY, SERIES)
    source = _source(tmp_path, known_empty_dates=(DAY,))
    assert source.read(DAY, SERIES) == ()
    assert source.list_files(SERIES) == {}
    with pytest.raises(FileNotFoundError):
        source.read(date(2026, 8, 30), SERIES)


def test_declared_empty_date_rejects_existing_waves(tmp_path):
    """Verify declared empty date rejects existing waves.

    Args:
        tmp_path: Pytest-owned temporary directory for input and output files.

    Returns:
        None.

    Steps:
        1. Check that declared empty date rejects existing waves.

    Examples:
        Run ``pytest tests/test_footprints.py -k declared_empty_date_rejects_existing_waves``.
    """
    _write(tmp_path, _table())
    with pytest.raises(FootprintError, match="declared empty date"):
        _source(tmp_path, known_empty_dates=(DAY,)).read(DAY, SERIES)


def test_typed_empty_file_is_accepted(tmp_path):
    """Verify typed empty file is accepted.

    Args:
        tmp_path: Pytest-owned temporary directory for input and output files.

    Returns:
        None.

    Steps:
        1. Check that typed empty file is accepted.

    Examples:
        Run ``pytest tests/test_footprints.py -k typed_empty_file_is_accepted``.
    """
    _write(tmp_path, _table().slice(0, 0), native_cells=WaveCells.empty(DAY))
    assert _source(tmp_path).read(DAY, SERIES) == ()
    assert _source(tmp_path, known_empty_dates=(DAY,)).read(DAY, SERIES) == ()


@pytest.mark.parametrize(
    "dates", [[DAY], ("2026-08-31",), (datetime(2026, 8, 31, tzinfo=UTC),)]
)
def test_empty_dates_must_be_explicit_plain_date_tuple(tmp_path, dates):
    """Verify empty dates must be explicit plain date tuple.

    Args:
        tmp_path: Pytest-owned temporary directory for input and output files.
        dates: Explicit empty-date declaration to validate.

    Returns:
        None.

    Steps:
        1. Check that empty dates must be explicit plain date tuple.

    Examples:
        Run ``pytest tests/test_footprints.py -k empty_dates_must_be_explicit_plain_date_tuple``.
    """
    with pytest.raises(FootprintError, match="tuple of plain calendar dates"):
        WaveFootprintSource(str(tmp_path), known_empty_dates=dates)


@pytest.mark.parametrize(
    ("series", "field", "direction"),
    [
        (WaveSeries("tasmax", "cold"), "t2m_max", "negative"),
        (WaveSeries("tasmin", "warm"), "t2m_min", "positive"),
        (WaveSeries("tasmin", "cold"), "t2m_min", "negative"),
    ],
)
def test_reader_selects_requested_series(tmp_path, series, field, direction):
    """Verify reader selects requested series.

    Args:
        tmp_path: Pytest-owned temporary directory for input and output files.
        series: Temperature and direction combination under test.
        field: Stored temperature field corresponding to the requested wave series.
        direction: Warm or cold direction to validate.

    Returns:
        None.

    Steps:
        1. Check that reader selects requested series.

    Examples:
        Run ``pytest tests/test_footprints.py -k reader_selects_requested_series``.
    """
    _write(tmp_path, _table(), field=field, direction=direction)
    assert len(_source(tmp_path).read(DAY, series)) == 1


def test_duplicate_ids_are_rejected(tmp_path):
    """Verify duplicate ids are rejected.

    Args:
        tmp_path: Pytest-owned temporary directory for input and output files.

    Returns:
        None.

    Steps:
        1. Check that duplicate ids are rejected.

    Examples:
        Run ``pytest tests/test_footprints.py -k duplicate_ids_are_rejected``.
    """
    table = _table()
    _write(tmp_path, pa.concat_tables([table, table]))
    with pytest.raises(FootprintError, match="storm_id"):
        _source(tmp_path).read(DAY, SERIES)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("storm_id", "0" * 31, "storm_id"),
        ("storm_id", "Z" * 32, "storm_id"),
        ("date", date(2026, 8, 30), "dates must all equal"),
        ("geometry", b"broken", "WKB"),
        ("geometry", Point(0, 0).wkb, "geometry"),
        (
            "geometry",
            Polygon([(0, 0), (1, 1), (0, 1), (1, 0), (0, 0)]).wkb,
            "geometry",
        ),
        ("buffer_geometry", None, "buffer_geometry"),
        ("area_qualified", -1.0, "area_"),
        ("area_qualified", float("inf"), "area_"),
        ("qualified_fraction", float("nan"), "qualified_fraction"),
        ("qualified_fraction", 0.5, "role_areas"),
    ],
)
def test_bad_values_are_rejected(tmp_path, name, value, message):
    """Verify bad values are rejected.

    Args:
        tmp_path: Pytest-owned temporary directory for input and output files.
        name: Input column to alter.
        value: Candidate scalar value selected by parametrization.
        message: Expected validation-error text.

    Returns:
        None.

    Steps:
        1. Check that bad values are rejected.

    Examples:
        Run ``pytest tests/test_footprints.py -k bad_values_are_rejected``.
    """
    table = _table()
    _write(tmp_path, _replace(table, name, [value], table.schema.field(name).type))
    with pytest.raises(FootprintError, match=message):
        _source(tmp_path).read(DAY, SERIES)


@pytest.mark.parametrize(
    "mutation", ["missing", "encoding", "crs", "null_crs", "column"]
)
def test_invalid_geo_metadata_is_rejected(tmp_path, mutation):
    """Verify invalid geo metadata is rejected.

    Args:
        tmp_path: Pytest-owned temporary directory for input and output files.
        mutation: Input mutation applied before validation.

    Returns:
        None.

    Steps:
        1. Check that invalid geo metadata is rejected.

    Examples:
        Run ``pytest tests/test_footprints.py -k invalid_geo_metadata_is_rejected``.
    """
    table = _table()
    geo = json.loads(table.schema.metadata[b"geo"])
    if mutation == "missing":
        table = table.replace_schema_metadata(None)
    else:
        if mutation == "encoding":
            geo["columns"]["geometry"]["encoding"] = "point"
        elif mutation == "crs":
            geo["columns"]["buffer_geometry"]["crs"] = "EPSG:3857"
        elif mutation == "null_crs":
            geo["columns"]["geometry"]["crs"] = None
        else:
            del geo["columns"]["buffer_geometry"]
        table = table.replace_schema_metadata({b"geo": json.dumps(geo).encode()})
    _write(tmp_path, table)
    with pytest.raises(FootprintError, match="GeoParquet"):
        _source(tmp_path).read(DAY, SERIES)


def test_default_crs84_geo_metadata_is_accepted(tmp_path):
    """Verify default crs84 geo metadata is accepted.

    Args:
        tmp_path: Pytest-owned temporary directory for input and output files.

    Returns:
        None.

    Steps:
        1. Check that default crs84 geo metadata is accepted.

    Examples:
        Run ``pytest tests/test_footprints.py -k default_crs84_geo_metadata_is_accepted``.
    """
    table = _table()
    geo = json.loads(table.schema.metadata[b"geo"])
    for column in geo["columns"].values():
        del column["crs"]
    _write(tmp_path, table.replace_schema_metadata({b"geo": json.dumps(geo).encode()}))
    assert len(_source(tmp_path).read(DAY, SERIES)) == 1


@pytest.mark.parametrize("mutation", ["missing", "extra", "type", "date_type"])
def test_invalid_columns_are_rejected_even_when_empty(tmp_path, mutation):
    """Verify invalid columns are rejected even when empty.

    Args:
        tmp_path: Pytest-owned temporary directory for input and output files.
        mutation: Input mutation applied before validation.

    Returns:
        None.

    Steps:
        1. Check that invalid columns are rejected even when empty.

    Examples:
        Run ``pytest tests/test_footprints.py -k invalid_columns_are_rejected_even_when_empty``.
    """
    table = _table().slice(0, 0)
    if mutation == "missing":
        table = table.drop_columns("buffer_geometry")
    elif mutation == "extra":
        table = table.append_column("unused", pa.array([], type=pa.int64()))
    elif mutation == "type":
        table = _replace(table, "area_qualified", [], pa.int64())
    else:
        table = _replace(table, "date", [], pa.string())
    _write(tmp_path, table)
    with pytest.raises(FootprintError, match="column"):
        _source(tmp_path).read(DAY, SERIES)


@pytest.mark.parametrize(
    "data_type",
    [
        pa.date32(),
        pa.date64(),
        pa.timestamp("s"),
        pa.timestamp("ms"),
        pa.timestamp("us"),
        pa.timestamp("ns"),
    ],
)
def test_date_types_preserve_calendar_midnight(tmp_path, data_type):
    """Accept equivalent stored midnight dates through the DataFrame reader.

    Args:
        tmp_path: Directory for a synthetic footprint file.
        data_type: Arrow date or timezone-free timestamp type.

    Returns:
        None. Every supported date representation produces the same component.

    Examples:
        pytest tests/test_footprints.py -k date_types

    Steps:
        1. Accept equivalent stored midnight dates through the DataFrame reader.
        2. Check that every supported date representation produces the same component.
    """
    table = _table()
    baseline_path = tmp_path / "baseline"
    _write(baseline_path, table)
    baseline = _source(baseline_path).read(DAY, SERIES)
    _write(tmp_path, table.set_column(1, "date", table["date"].cast(data_type)))
    assert _source(tmp_path).read(DAY, SERIES) == baseline


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("storm_id", "storm_id"),
        ("date", "date"),
        ("area_qualified", "area_"),
        ("area_grace", "area_"),
        ("area_gap_filled", "area_"),
        ("qualified_fraction", "qualified_fraction"),
    ],
)
def test_required_nulls_are_rejected(tmp_path, name, message):
    """Reject missing required values while allowing absent optional role shapes.

    Args:
        tmp_path: Directory for a synthetic footprint file.
        name: Required scalar column containing one null.
        message: Expected input error description.

    Returns:
        None when the reader rejects the missing required value.

    Examples:
        pytest tests/test_footprints.py -k required_nulls

    Steps:
        1. Reject missing required values while allowing absent optional role shapes.
    """
    table = _table()
    _write(tmp_path, _replace(table, name, [None], table.schema.field(name).type))
    with pytest.raises(FootprintError, match=message):
        _source(tmp_path).read(DAY, SERIES)


@pytest.mark.parametrize("offset", [1, 1_000, 3_600_000_000_000])
def test_timestamp_rejects_any_nonmidnight_precision(tmp_path, offset):
    """Check midnight at nanosecond precision without rounding dates.

    Args:
        tmp_path: Directory for a synthetic footprint file.
        offset: Positive nanoseconds added to the expected midnight.

    Returns:
        None when a nonmidnight value is rejected, including submicrosecond time.

    Examples:
        pytest tests/test_footprints.py -k nonmidnight

    Steps:
        1. Check midnight at nanosecond precision without rounding dates.
    """
    table = _table()
    midnight = pa.scalar(DAY, type=pa.date32()).cast(pa.timestamp("ns")).value
    values = pa.array([midnight + offset], type=pa.timestamp("ns"))
    _write(tmp_path, table.set_column(1, "date", values))
    with pytest.raises(FootprintError, match="date"):
        _source(tmp_path).read(DAY, SERIES)


def test_timezone_timestamp_is_rejected(tmp_path):
    """Keep stored calendar dates independent of timezone conversion.

    Args:
        tmp_path: Directory for a synthetic footprint file.

    Returns:
        None when even a UTC timestamp is rejected by the input schema rule.

    Examples:
        pytest tests/test_footprints.py -k timezone_timestamp

    Steps:
        1. Keep stored calendar dates independent of timezone conversion.
    """
    table = _table()
    _write(
        tmp_path,
        _replace(
            table,
            "date",
            [datetime(2026, 8, 31, tzinfo=UTC)],
            pa.timestamp("ns", "UTC"),
        ),
    )
    with pytest.raises(FootprintError, match="timezone-free"):
        _source(tmp_path).read(DAY, SERIES)


def test_row_groups_preserve_scalars_and_sorted_components(tmp_path):
    """Read unsorted row groups with optional nulls and original scalar precision.

    Args:
        tmp_path: Directory for a synthetic footprint file.

    Returns:
        None. The reader retains WKB, input labels and float values while
        assigning component numbers in historical footprint-digest order.

    Examples:
        pytest tests/test_footprints.py -k chunked_rows

    Steps:
        1. Read unsorted row groups with optional nulls and original scalar precision.
        2. Replace the input labels independently of the cell identities and ordering digests.
        3. Check stable component order, original labels and exact physical values.
    """
    tables = [
        _table(box(2 * i - 0.125, -0.125, 2 * i + 0.875, 0.875)) for i in range(3)
    ]
    table = pa.concat_tables(tables)
    daily_ids = table["storm_id"].to_pylist()
    table = _replace(table, "storm_id", [f"{index:032x}" for index in range(3)])
    for name in ("storm_id",):
        index = table.schema.get_field_index(name)
        table = table.set_column(index, name, table[name].cast(pa.large_string()))
    for name in ("geometry", "grace_geometry", "buffer_geometry"):
        index = table.schema.get_field_index(name)
        table = table.set_column(index, name, table[name].cast(pa.large_binary()))
    shuffled = table.take(pa.array([2, 1, 0]))
    path = _write(tmp_path, shuffled, native_cells=_cells(offsets=(0, 2, 4)))
    pq.write_table(shuffled, path, row_group_size=1)
    components = _source(tmp_path).read(DAY, SERIES)
    assert [component.order_digest for component in components] == sorted(
        component.order_digest for component in components
    )
    for label, component in enumerate(components, 1):
        index = daily_ids.index(component.component_id)
        assert component.component_id == daily_ids[index]
        assert component.input_id == table["storm_id"][index].as_py()
        geometry = component.geometry
        assert geometry.label == label
        assert geometry.geometry.wkb == table["geometry"][index].as_py()
        assert geometry.full_geometry.wkb == table["buffer_geometry"][index].as_py()
        assert geometry.grace_geometry is None
        for name in (
            "area_qualified",
            "area_grace",
            "area_gap_filled",
            "qualified_fraction",
        ):
            assert getattr(geometry, name) == table[name][index].as_py()


def test_arbitrary_polygon_is_rejected_when_computing_historical_order(tmp_path):
    """Keep shape reading available while requiring native coverage for ordering.

    Args:
        tmp_path: Temporary directory for a valid but nonnative stored rectangle.

    Returns:
        None. Ordering reports unsupported coverage without altering stored shapes.

    Examples:
        pytest tests/test_footprints.py -k arbitrary_polygon_is_rejected
    """
    path = _write(tmp_path, _table(box(0, 0, 1, 1)))
    footprints = WaveFootprints.read_parquet(path)
    saved_shape = footprints.frame.buffer_geometry.iloc[0].wkb
    with pytest.raises(FootprintError, match="cannot compute historical ordering"):
        footprints.to_components(DAY, SERIES, native_cells=_cells())
    assert footprints.frame.buffer_geometry.iloc[0].wkb == saved_shape


def test_prepared_tracking_requires_an_explicit_native_cell_source(tmp_path):
    """Read and list stored shapes without guessing their complete cell context.

    Args:
        tmp_path: Root containing a valid footprint and its explicit cell companion.

    Returns:
        None. The tracking interface rejects absent configuration while ordinary
        schema checks and file listings still work.

    Examples:
        pytest tests/test_footprints.py -k explicit_native_cell_source
    """
    path = _write(tmp_path, _table())
    source = WaveFootprintSource(str(tmp_path))
    assert source.list_files(SERIES) == {path.name: path.stat().st_size}
    WaveFootprints.read_parquet(path).validate_day(DAY, SERIES)
    with pytest.raises(FootprintError, match="cell_source"):
        source.read(DAY, SERIES)


@pytest.mark.parametrize("data_type", [pa.float32(), pa.float64()])
def test_numeric_columns_preserve_supplied_float_values(tmp_path, data_type):
    """Preserve representable float values while validating the DataFrame schema.

    Args:
        tmp_path: Directory for a synthetic footprint file.
        data_type: Stored floating-point precision.

    Returns:
        None. Each output scalar equals its supplied stored value without rounding.

    Examples:
        pytest tests/test_footprints.py -k numeric_columns

    Steps:
        1. Preserve representable float values while validating the DataFrame schema.
        2. Check that each output scalar equals its supplied stored value without rounding.
    """
    table = _table()
    area = np.nextafter(table["area_qualified"][0].as_py(), float("inf"))
    fraction = np.nextafter(1.0, 0.0)
    table = _replace(table, "area_qualified", [area], data_type)
    table = _replace(table, "qualified_fraction", [fraction], data_type)
    _write(tmp_path, table)
    (component,) = _source(tmp_path).read(DAY, SERIES)
    assert component.geometry.area_qualified == table["area_qualified"][0].as_py()
    assert (
        component.geometry.qualified_fraction == table["qualified_fraction"][0].as_py()
    )


def test_footprint_table_validates_once_and_preserves_caller_frame(
    tmp_path, monkeypatch
):
    """Retain input geometry bytes and avoid a second schema pass in conversion.

    Args:
        tmp_path: Independent daily footprint directory.
        monkeypatch: Fixture counting the existing schema validation.

    Returns:
        None. Reading checks the schema once, component conversion adds no
        schema pass, and direct construction keeps the supplied frame.

    Examples:
        Run ``pytest tests/test_footprints.py -k validates_once``.

    Steps:
        1. Retain input geometry bytes and avoid a second schema pass in conversion.
        2. Check that reading checks the schema once, component conversion adds no schema pass, and direct construction keeps the supplied frame.
    """
    path = _write(tmp_path, _table())
    schema = WaveFootprints.schema
    schema.to_schema()
    original = schema.validate
    calls = []

    def counted(cls, frame, *args, **kwargs):
        """Record a schema call and run the original validation.

        Args:
            cls: Schema receiving the call.
            frame: Supplied footprint rows.
            *args: Existing validation arguments.
            **kwargs: Existing validation options.

        Returns:
            The original Pandera validation result.

        Examples:
            Called by ``WaveFootprints.read_parquet``.

        Steps:
            1. Record a schema call and run the original validation.
        """
        calls.append(frame)
        return original(frame, *args, **kwargs)

    monkeypatch.setattr(schema, "validate", classmethod(counted))
    table = WaveFootprints.read_parquet(path)
    assert len(calls) == 1
    (component,) = table.to_components(DAY, SERIES, native_cells=_cells())
    assert len(calls) == 1
    assert (
        component.geometry.full_geometry.wkb == table.frame.buffer_geometry.iloc[0].wkb
    )
    supplied = table.frame
    assert WaveFootprints(supplied).frame is supplied
    assert len(calls) == 2
