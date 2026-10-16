"""Check daily wave state, small-gap filling and exact cell polygons."""

from datetime import date, timedelta

import fsspec
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from reuters_temperature_storms.cell_geometry import WaveCellBuilder
from reuters_temperature_storms.cell_state import WaveCellState, WaveCellStateSource
from reuters_temperature_storms.cells import WaveCellError, WaveCells
from reuters_temperature_storms.series import WaveSeries

DAY = date(2024, 7, 1)


def _state_frame(rows=()):
    """Make typed state rows with independently supplied cell histories.

    Args:
        rows: Longitude, latitude, qualifying count, grace count and optional
        end date. Every row started seven days before the observation.

    Returns:
        A six-column DataFrame suitable for constructing daily state.

    Examples:
        frame = _state_frame(((0, 0, 5, 0, None),))

    Steps:
        1. Make typed state rows with independently supplied cell histories.
    """
    return pd.DataFrame(
        {
            "longitude": pd.Series([row[0] for row in rows], dtype="float64"),
            "latitude": pd.Series([row[1] for row in rows], dtype="float64"),
            "start_date": pd.Series(
                [DAY - timedelta(days=7)] * len(rows), dtype="datetime64[ns]"
            ),
            "end_date": pd.Series([row[4] for row in rows], dtype="datetime64[ns]"),
            "days": pd.Series([row[2] for row in rows], dtype="int64"),
            "grace_days": pd.Series([row[3] for row in rows], dtype="int64"),
        }
    )


def _patch(*, hole=True, seam=False):
    """Create a seven-by-seven square of mature cells with an optional hole.

    Args:
        hole: Leave out the centre cell when true.
        seam: Centre the square on the longitude seam when true.

    Returns:
        Daily state with real counts varied by latitude.

    Examples:
        state = WaveCellState(_patch(seam=True), DAY)

    Steps:
        1. Create a seven-by-seven square of mature cells with an optional hole.
    """
    rows = []
    for x in range(-3, 4):
        for y in range(-3, 4):
            if hole and x == y == 0:
                continue
            longitude = (180 + x / 4 + 180) % 360 - 180 if seam else x / 4
            if longitude == -180:
                longitude = 180
            rows.append((longitude, y / 4, 5 + int(y > 0), int(y == -1), None))
    return _state_frame(rows)


def test_mature_active_and_grace_rows_keep_their_original_counts():
    """Keep mature active and grace cells while dropping young and ended rows.

    Args:
        None.

    Returns:
        None. Only the two mature active cells retain their exact state counts.

    Examples:
        pytest tests/test_cell_state.py -k mature_active

    Steps:
        1. Keep mature active and grace cells while dropping young and ended rows.
        2. Check that only the two mature active cells retain their exact state counts.
    """
    frame = _state_frame(
        (
            (0, 0, 5, 0, None),
            (0.25, 0, 6, 1, None),
            (0.5, 0, 4, 0, None),
            (0.75, 0, 5, 0, DAY - timedelta(days=1)),
        )
    )
    cells = WaveCellState(frame, DAY).to_cells(closing_radius=0)
    assert isinstance(cells, WaveCells)
    assert list(cells.frame.columns) == list(WaveCells.columns)
    assert list(cells.frame.longitude) == [0, 0.25]
    assert list(cells.frame.days) == [5, 6]
    assert list(cells.frame.grace_days) == [0, 1]
    assert not cells.frame.gap_filled.any()
    assert cells.frame.date.eq(pd.Timestamp(DAY)).all()


@pytest.mark.parametrize("radius", [0, 1])
def test_zero_and_one_radius_leave_a_missing_centre_open(radius):
    """A zero radius and a radius containing only the centre leave gaps alone.

    Args:
        radius: Disabled closing or a neighbourhood with only its centre.

    Returns:
        None. The missing centre stays absent and every input count survives.

    Examples:
        pytest tests/test_cell_state.py -k zero_and_one

    Steps:
        1. A zero radius and a radius containing only the centre leave gaps alone.
        2. Check that the missing centre stays absent and every input count survives.
    """
    frame = WaveCellBuilder(WaveCellState(_patch(), DAY), radius).build().frame
    assert len(frame) == 48
    assert not (frame.longitude.eq(0) & frame.latitude.eq(0)).any()
    assert not frame.gap_filled.any()


@pytest.mark.parametrize("seam", [False, True])
def test_radius_three_fills_a_hole_with_uncounted_cells(seam):
    """Fill one enclosed hole, including a hole crossing the longitude seam.

    Args:
        seam: Put the centre on the seam rather than at zero longitude.

    Returns:
        None. All real counts survive and the added cell has no history counts.

    Examples:
        pytest tests/test_cell_state.py -k fills_a_hole

    Steps:
        1. Fill one enclosed hole, including a hole crossing the longitude seam.
        2. Check that all real counts survive and the added cell has no history counts.
    """
    source = _patch(seam=seam)
    frame = WaveCellState(source, DAY).to_cells().frame
    coordinates = frame.drop_duplicates(["longitude", "latitude"])
    assert len(coordinates) == 49
    centre = coordinates[
        coordinates.longitude.eq(180 if seam else 0) & coordinates.latitude.eq(0)
    ]
    assert len(centre) == 1
    assert centre.gap_filled.all()
    assert centre[["days", "grace_days"]].isna().all().all()
    real = coordinates[~coordinates.gap_filled].set_index(["longitude", "latitude"])
    expected = source.set_index(["longitude", "latitude"])
    pd.testing.assert_frame_equal(
        real[["days", "grace_days"]].sort_index(),
        expected[["days", "grace_days"]].astype(float).sort_index(),
        check_dtype=False,
    )
    assert not frame[["longitude", "latitude"]].isna().any().any()
    assert frame.crs.to_epsg() == 4326


@pytest.mark.parametrize("latitude", [-90, -89.75, 89.75, 90])
def test_radius_three_erases_cells_in_the_two_polar_edge_rows(latitude):
    """Preserve the specified false-border behavior beside either pole.

    Args:
        latitude: A cell in one of the two rows nearest either pole.

    Returns:
        None. Closing removes the polar cell but radius zero retains it.

    Examples:
        pytest tests/test_cell_state.py -k polar_edge

    Steps:
        1. Preserve the specified false-border behavior beside either pole.
        2. Check that closing removes the polar cell but radius zero retains it.
    """
    state = WaveCellState(_state_frame(((0, latitude, 5, 0, None),)), DAY)
    assert state.to_cells().frame.empty
    assert len(state.to_cells(closing_radius=0).frame) == 1


@pytest.mark.parametrize("latitude", [-89.5, 0, 89.5])
def test_radius_three_preserves_single_cells_outside_the_polar_edge(latitude):
    """Retain isolated mature cells outside the two polar edge rows.

    Args:
        latitude: First unaffected latitude or an equatorial cell.

    Returns:
        None. The original cell survives without introducing any gap cells.

    Examples:
        pytest tests/test_cell_state.py -k outside_the_polar

    Steps:
        1. Retain isolated mature cells outside the two polar edge rows.
        2. Check that the original cell survives without introducing any gap cells.
    """
    frame = (
        WaveCellState(_state_frame(((0, latitude, 5, 1, None),)), DAY).to_cells().frame
    )
    assert len(frame) == 1
    assert frame.iloc[0].grace_days == 1
    assert not frame.gap_filled.any()


@pytest.mark.parametrize(
    ("longitude", "latitude"),
    [(0, 0), (-179.75, 0), (0, -90), (0, 90), (180, 0), (180, -90), (180, 90)],
)
def test_grid_polygons_are_exact_and_the_dateline_cell_has_two_halves(
    longitude, latitude
):
    """Generate quarter-degree polygons with exact polar and seam boundaries.

    Args:
        longitude: An interior, western-edge or seam centre.
        latitude: Equatorial or polar cell centre.

    Returns:
        None. Every geometry matches its separately specified rectangle.

    Examples:
        pytest tests/test_cell_state.py -k grid_polygons

    Steps:
        1. Generate quarter-degree polygons with exact polar and seam boundaries.
        2. Check that every geometry matches its separately specified rectangle.
    """
    frame = (
        WaveCellState(_state_frame(((longitude, latitude, 5, 0, None),)), DAY)
        .to_cells(closing_radius=0)
        .frame
    )
    bottom = max(-90, latitude - 0.125)
    top = min(90, latitude + 0.125)
    shapes = (
        [box(179.875, bottom, 180, top), box(-180, bottom, -179.875, top)]
        if longitude == 180
        else [box(longitude - 0.125, bottom, longitude + 0.125, top)]
    )
    assert len(frame) == len(shapes)
    assert list(frame.longitude) == [longitude] * len(shapes)
    assert list(frame.latitude) == [latitude] * len(shapes)
    for actual, expected in zip(frame.geometry, shapes, strict=True):
        assert actual.equals_exact(expected, tolerance=0, normalize=True)
        assert actual.bounds == expected.bounds


def test_empty_state_yields_an_empty_typed_cell_table():
    """Allow explicit empty dates to continue into the footprint stage.

    Args:
        None.

    Returns:
        None. The empty output retains all seven typed cell columns.

    Examples:
        pytest tests/test_cell_state.py -k empty_state

    Steps:
        1. Allow explicit empty dates to continue into the footprint stage.
        2. Check that the empty output retains all seven typed cell columns.
    """
    cells = WaveCellState(_state_frame(), DAY).to_cells()
    assert cells.frame.empty
    assert list(cells.frame.columns) == list(WaveCells.columns)
    assert cells.frame.crs.to_epsg() == 4326
    assert cells.frame.gap_filled.dtype == np.dtype("bool")


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("longitude", -180.0),
        ("longitude", 0.1),
        ("longitude", np.inf),
        ("latitude", 90.25),
        ("latitude", 0.1),
        ("days", 0),
        ("days", -1),
        ("days", 9),
        ("days", 5.5),
        ("days", np.nan),
        ("days", "5"),
        ("grace_days", np.nan),
        ("grace_days", 0.5),
        ("grace_days", -1),
        ("grace_days", 2),
        ("start_date", pd.Timestamp(DAY + timedelta(days=1))),
        ("start_date", pd.Timestamp(DAY) + pd.Timedelta(hours=1)),
        ("end_date", pd.Timestamp(DAY + timedelta(days=1))),
        ("end_date", pd.Timestamp(DAY)),
        ("start_date", pd.NaT),
        ("end_date", pd.Timestamp(DAY - timedelta(days=8))),
    ],
)
def test_invalid_state_values_are_rejected_before_building(column, value):
    """Reject invalid coordinates, counts and calendar relationships.

    Args:
        column: Required state field to change.
        value: Invalid value for an otherwise valid row.

    Returns:
        None. Construction raises an input error before creating geometry.

    Examples:
        pytest tests/test_cell_state.py -k invalid_state_values

    Steps:
        1. Reject invalid coordinates, counts and calendar relationships.
        2. Check that construction raises an input error before creating geometry.
    """
    frame = _state_frame(((0, 0, 5, 0, None),))
    frame[column] = value
    with pytest.raises(WaveCellError):
        WaveCellState(frame, DAY)


def test_duplicate_coordinate_rows_and_extra_or_missing_columns_are_rejected():
    """Require one history per coordinate and the stated six-column input.

    Args:
        None.

    Returns:
        None. Duplicate histories and malformed table shapes are rejected.

    Examples:
        pytest tests/test_cell_state.py -k duplicate_coordinate

    Steps:
        1. Require one history per coordinate and the stated six-column input.
        2. Check that duplicate histories and malformed table shapes are rejected.
    """
    original = _state_frame(((0, 0, 5, 0, None),))
    frames = [
        pd.concat([original, original], ignore_index=True),
        original.drop(columns="grace_days"),
        original.assign(unexpected=1),
    ]
    for frame in frames:
        with pytest.raises(WaveCellError):
            WaveCellState(frame, DAY)


def test_local_parquet_round_trip_reads_only_state_columns(tmp_path):
    """Project the six state columns and normalize stored calendar dates.

    Args:
        tmp_path: Temporary directory supplied by pytest.

    Returns:
        None. Additional stored data is unread and state survives a round trip.

    Examples:
        pytest tests/test_cell_state.py -k local_parquet

    Steps:
        1. Project the six state columns and normalize stored calendar dates.
        2. Check that additional stored data is unread and state survives a round trip.
    """
    original = _state_frame(((0, 0, 5, 0, None), (0.25, 0, 5, 1, None)))
    path = tmp_path / "state.parquet"
    original.assign(unused_measurement=999).to_parquet(path)
    state = WaveCellState.read_parquet(path, DAY)
    pd.testing.assert_frame_equal(state.frame, original)
    destination = tmp_path / "round-trip.parquet"
    state.to_parquet(destination)
    restored = WaveCellState.read_parquet(destination, DAY)
    pd.testing.assert_frame_equal(restored.frame, original)


def test_memory_filesystem_round_trip_preserves_state_and_missing_errors():
    """Keep local and supplied filesystem I/O behavior consistent.

    Args:
        None.

    Returns:
        None. Counts and dates survive writes; absent files remain absent errors.

    Examples:
        pytest tests/test_cell_state.py -k memory_filesystem

    Steps:
        1. Keep local and supplied filesystem I/O behavior consistent.
        2. Check that counts and dates survive writes; absent files remain absent errors.
    """
    filesystem = fsspec.filesystem("memory")
    path = "/cell-state-test/2024-07-01.parquet"
    state = WaveCellState(_state_frame(((0, 0, 5, 0, None),)), DAY)
    state.to_parquet(path, filesystem=filesystem)
    restored = WaveCellState.read_parquet(path, DAY, filesystem=filesystem)
    pd.testing.assert_frame_equal(restored.frame, state.frame)
    with pytest.raises(FileNotFoundError):
        WaveCellState.read_parquet(
            "/cell-state-test/missing.parquet", DAY, filesystem=filesystem
        )


@pytest.mark.parametrize("column", ["start_date", "end_date"])
def test_parquet_reader_rejects_text_dates_instead_of_interpreting_them(
    tmp_path, column
):
    """Require stored calendar dates rather than silently parsing arbitrary text.

    Args:
        tmp_path: Temporary directory supplied by pytest.
        column: Date field to replace with text.

    Returns:
        None. Reading malformed stored dates raises an input error.

    Examples:
        pytest tests/test_cell_state.py -k text_dates

    Steps:
        1. Require stored calendar dates rather than silently parsing arbitrary text.
        2. Check that reading malformed stored dates raises an input error.
    """
    frame = _state_frame(((0, 0, 5, 0, None),))
    frame[column] = "2024-06-24"
    path = tmp_path / "bad-date.parquet"
    frame.to_parquet(path)
    with pytest.raises(WaveCellError):
        WaveCellState.read_parquet(path, DAY)


@pytest.mark.parametrize("radius", [-1, 1.5, True])
def test_invalid_closing_radius_is_rejected(radius):
    """Require an integer nonnegative spatial-gap radius.

    Args:
        radius: A negative, fractional or boolean value.

    Returns:
        None. The builder rejects invalid configuration before calculation.

    Examples:
        pytest tests/test_cell_state.py -k invalid_closing

    Steps:
        1. Require an integer nonnegative spatial-gap radius.
        2. Check that the builder rejects invalid configuration before calculation.
    """
    state = WaveCellState(_state_frame(((0, 0, 5, 0, None),)), DAY)
    with pytest.raises(WaveCellError):
        WaveCellBuilder(state, radius)


def test_ended_state_cannot_keep_a_grace_count():
    """An ended cell must reset its consecutive grace count.

    Args:
        None.

    Returns:
        None. A stored ending with a retained grace count is rejected.

    Examples:
        pytest tests/test_cell_state.py -k ended_state

    Steps:
        1. An ended cell must reset its consecutive grace count.
        2. Check that a stored ending with a retained grace count is rejected.
    """
    frame = _state_frame(((0, 0, 5, 1, DAY - timedelta(days=1)),))
    with pytest.raises(WaveCellError):
        WaveCellState(frame, DAY)


def test_reader_normalizes_arrow_calendar_dates_and_an_all_null_end(tmp_path):
    """Accept native Parquet dates and an all-active empty end-date column.

    Args:
        tmp_path: Temporary directory supplied by pytest.

    Returns:
        None. Calendar dates and empty endings become canonical date columns.

    Examples:
        pytest tests/test_cell_state.py -k arrow_calendar

    Steps:
        1. Accept native Parquet dates and an all-active empty end-date column.
        2. Check that calendar dates and empty endings become canonical date columns.
    """
    frame = _state_frame(((0, 0, 5, 0, None),))
    frame["start_date"] = frame.start_date.dt.date
    frame["end_date"] = None
    path = tmp_path / "calendar-dates.parquet"
    frame.to_parquet(path)
    state = WaveCellState.read_parquet(path, DAY)
    pd.testing.assert_frame_equal(state.frame, _state_frame(((0, 0, 5, 0, None),)))


def test_state_counts_cannot_exceed_the_elapsed_calendar_span():
    """Counted qualifying and current grace days must fit inside the history.

    Args:
        None.

    Returns:
        None. A short active or ended history cannot hold excessive counts.

    Examples:
        pytest tests/test_cell_state.py -k elapsed_calendar

    Steps:
        1. Counted qualifying and current grace days must fit inside the history.
        2. Check that a short active or ended history cannot hold excessive counts.
    """
    frame = _state_frame(((0, 0, 5, 1, None),))
    frame["start_date"] = pd.Timestamp(DAY - timedelta(days=4))
    with pytest.raises(WaveCellError):
        WaveCellState(frame, DAY)
    ended = _state_frame(((0, 0, 5, 0, DAY - timedelta(days=4)),))
    ended["start_date"] = pd.Timestamp(DAY - timedelta(days=5))
    with pytest.raises(WaveCellError):
        WaveCellState(ended, DAY)


def test_radius_three_excludes_cells_exactly_three_steps_from_the_centre():
    """Keep two six-step-separated strips apart under the strict radius rule.

    Args:
        None.

    Returns:
        None. Two strips stay separate where an inclusive radius would bridge.

    Examples:
        pytest tests/test_cell_state.py -k excludes_cells_exactly

    Steps:
        1. Keep two six-step-separated strips apart under the strict radius rule.
        2. Check that two strips stay separate where an inclusive radius would bridge.
    """
    rows = [
        (longitude, y / 4, 5, 0, None) for longitude in (0, 1.5) for y in range(-8, 9)
    ]
    frame = WaveCellState(_state_frame(rows), DAY).to_cells().frame
    middle = frame[frame.latitude.eq(0)]
    assert set(middle.longitude) == {0, 1.5}
    assert not frame.gap_filled.any()


@pytest.mark.parametrize(
    ("temperature", "direction", "field", "folder"),
    [
        ("tasmax", "warm", "t2m_max", "positive"),
        ("tasmax", "cold", "t2m_max", "negative"),
        ("tasmin", "warm", "t2m_min", "positive"),
        ("tasmin", "cold", "t2m_min", "negative"),
    ],
)
def test_state_source_selects_the_requested_series(
    tmp_path, temperature, direction, field, folder
):
    """Read the chosen temperature and direction from their daily folders.

    Args:
        tmp_path: Temporary directory supplied by pytest.
        temperature: Maximum or minimum temperature selection.
        direction: Warm or cold direction.
        field: Expected field directory.
        folder: Expected direction directory.

    Returns:
        None. Each series reads only its requested dated six-column file.

    Examples:
        pytest tests/test_cell_state.py -k selects_the_requested

    Steps:
        1. Read the chosen temperature and direction from their daily folders.
        2. Check that each series reads only its requested dated six-column file.
    """
    frame = _state_frame(((0, 0, 5, 0, None),))
    path = tmp_path / field / folder / f"{DAY}.parquet"
    WaveCellState(frame, DAY).to_parquet(path)
    source = WaveCellStateSource(str(tmp_path))
    restored = source.read_state(DAY, WaveSeries(temperature, direction))
    pd.testing.assert_frame_equal(restored.frame, frame)


def test_declared_empty_state_may_be_missing_or_typed_empty(tmp_path):
    """Allow only declared empty dates to omit their input files.

    Args:
        tmp_path: Temporary directory supplied by pytest.

    Returns:
        None. Missing and typed-empty declared dates return empty state.

    Examples:
        pytest tests/test_cell_state.py -k declared_empty_state

    Steps:
        1. Allow only declared empty dates to omit their input files.
        2. Check that missing and typed-empty declared dates return empty state.
    """
    source = WaveCellStateSource(str(tmp_path), known_empty_dates=(DAY,))
    series = WaveSeries("tasmax", "warm")
    assert source.read_state(DAY, series).frame.empty
    path = tmp_path / "t2m_max" / "positive" / f"{DAY}.parquet"
    WaveCellState.empty(DAY).to_parquet(path)
    assert source.read_state(DAY, series).frame.empty
    with pytest.raises(FileNotFoundError):
        source.read_state(DAY + timedelta(days=1), series)


def test_declared_empty_date_rejects_existing_nonempty_or_malformed_state(tmp_path):
    """An empty-date declaration must never conceal existing bad input.

    Args:
        tmp_path: Temporary directory supplied by pytest.

    Returns:
        None. Nonempty and malformed existing files remain input errors.

    Examples:
        pytest tests/test_cell_state.py -k declaration_rejects

    Steps:
        1. An empty-date declaration must never conceal existing bad input.
        2. Check that nonempty and malformed existing files remain input errors.
    """
    source = WaveCellStateSource(str(tmp_path), known_empty_dates=(DAY,))
    series = WaveSeries("tasmax", "warm")
    path = tmp_path / "t2m_max" / "positive" / f"{DAY}.parquet"
    WaveCellState(_state_frame(((0, 0, 5, 0, None),)), DAY).to_parquet(path)
    with pytest.raises(WaveCellError):
        source.read_state(DAY, series)
    _state_frame().drop(columns="days").to_parquet(path)
    with pytest.raises(WaveCellError):
        source.read_state(DAY, series)


@pytest.mark.parametrize("column", ["start_date", "end_date"])
def test_state_rejects_timezone_dates_including_an_all_null_end(column):
    """Reject timezone-bearing date columns before comparing calendar histories.

    Args:
        column: Date field converted to a timezone-bearing type.

    Returns:
        None. Construction raises a cell error even if all endings are absent.

    Examples:
        pytest tests/test_cell_state.py -k state_rejects_timezone

    Steps:
        1. Reject timezone-bearing date columns before comparing calendar histories.
        2. Check that construction raises a cell error even if all endings are absent.
    """
    frame = _state_frame(((0, 0, 5, 0, None),))
    frame[column] = frame[column].dt.tz_localize("UTC")
    with pytest.raises(WaveCellError):
        WaveCellState(frame, DAY)


def test_reader_rejects_a_timezone_bearing_all_null_end(tmp_path):
    """Do not silently strip stored timezone metadata from absent end dates.

    Args:
        tmp_path: Temporary directory supplied by pytest.

    Returns:
        None. A timezone-bearing all-null date column remains invalid input.

    Examples:
        pytest tests/test_cell_state.py -k timezone_bearing_all_null

    Steps:
        1. Do not silently strip stored timezone metadata from absent end dates.
        2. Check that a timezone-bearing all-null date column remains invalid input.
    """
    frame = _state_frame(((0, 0, 5, 0, None),))
    frame["end_date"] = frame.end_date.dt.tz_localize("UTC")
    path = tmp_path / "timezone-empty-end.parquet"
    frame.to_parquet(path)
    with pytest.raises(WaveCellError):
        WaveCellState.read_parquet(path, DAY)


@pytest.mark.parametrize("ended", [False, True])
def test_filled_young_or_ended_coordinate_does_not_inherit_history_counts(ended):
    """Count a closed hole as a gap even when younger or ended history is there.

    Args:
        ended: Give the omitted centre an ended rather than young history.

    Returns:
        None. The generated centre has no qualifying or grace-day counts.

    Examples:
        pytest tests/test_cell_state.py -k does_not_inherit

    Steps:
        1. Count a closed hole as a gap even when younger or ended history is there.
        2. Check that the generated centre has no qualifying or grace-day counts.
    """
    centre = _state_frame(
        (
            (
                0,
                0,
                5 if ended else 4,
                0 if ended else 1,
                DAY - timedelta(days=1) if ended else None,
            ),
        )
    )
    source = pd.concat([_patch(), centre], ignore_index=True)
    frame = WaveCellState(source, DAY).to_cells().frame
    filled = frame[frame.longitude.eq(0) & frame.latitude.eq(0)]
    assert len(filled) == 1
    assert filled.gap_filled.all()
    assert filled[["days", "grace_days"]].isna().all().all()


@pytest.mark.parametrize("radius", [0, 3])
def test_cell_output_order_and_geometry_are_independent_of_state_row_order(radius):
    """Produce the same ordered cells from differently ordered state tables.

    Args:
        radius: Disabled or default spatial closing.

    Returns:
        None. Cell rows, flags, counts and polygon bytes are exactly equal.

    Examples:
        pytest tests/test_cell_state.py -k independent_of_state_row_order

    Steps:
        1. Produce the same ordered cells from differently ordered state tables.
        2. Check that cell rows, flags, counts and polygon bytes are exactly equal.
    """
    source = _patch(seam=True)
    forward = WaveCellState(source, DAY).to_cells(closing_radius=radius).frame
    reversed_rows = source.iloc[::-1].copy()
    reverse = WaveCellState(reversed_rows, DAY).to_cells(closing_radius=radius).frame
    pd.testing.assert_frame_equal(forward, reverse)
    assert forward.geometry.to_wkb().equals(reverse.geometry.to_wkb())
