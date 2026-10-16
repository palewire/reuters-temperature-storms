"""Check complete daily score inputs without changing score precision."""

from datetime import date

import fsspec
import numpy as np
import pandas as pd
import pytest

from reuters_temperature_storms.cell_scores import WaveCellScores, WaveCellScoreSource
from reuters_temperature_storms.cells import WaveCellError
from reuters_temperature_storms.series import WaveSeries

DAY = date(2024, 7, 1)
LONGITUDE, LATITUDE = np.meshgrid(
    np.arange(-179.75, 180.25, 0.25), np.arange(-90, 90.25, 0.25)
)


def test_reader_expands_coordinates_and_checks_embedded_dates(tmp_path):
    """Read exact float32 coordinates while preserving native score precision.

    Args:
        tmp_path: Isolated directory for a dated score Parquet.

    Returns:
        None. Coordinates expand exactly and wrong stored dates are rejected.

    Examples:
        pytest tests/test_cell_scores.py -k embedded_dates

    Steps:
        1. Read exact float32 coordinates while preserving native score precision.
        2. Check that coordinates expand exactly and wrong stored dates are rejected.
    """
    frame = score_frame(dtype="float32")
    stored = frame.rename(columns={"score": "t2m_max_p90_score"}).assign(
        latitude=frame.latitude.astype("float32"), date=pd.Timestamp(DAY)
    )
    path = tmp_path / "scores.parquet"
    stored.to_parquet(path, index=False)
    accepted = WaveCellScores.read_parquet(path, DAY, WaveSeries("tasmax", "warm"))
    pd.testing.assert_frame_equal(accepted.frame, frame)
    stored.loc[0, "date"] = pd.Timestamp("2024-07-02")
    stored.to_parquet(path, index=False)
    with pytest.raises(WaveCellError, match="stored dates differ"):
        WaveCellScores.read_parquet(path, DAY, WaveSeries("tasmax", "warm"))


def score_frame(points=(), *, dtype="float64"):
    """Create a complete grid with selected score values and zero elsewhere.

    Args:
        points: Longitude, latitude and score triplets to place in the grid.
        dtype: Native floating-point type of the score column.

    Returns:
        Complete three-column grid with independently assigned scores.

    Examples:
        frame = score_frame(((0, 0, 1.0),), dtype="float32")

    Steps:
        1. Create a complete grid with selected score values and zero elsewhere.
    """
    scores = np.zeros(LONGITUDE.shape, dtype=dtype)
    for longitude, latitude, value in points:
        scores[round((latitude + 90) * 4), round((longitude + 179.75) * 4)] = value
    return pd.DataFrame(
        {
            "longitude": LONGITUDE.ravel(),
            "latitude": LATITUDE.ravel(),
            "score": scores.ravel(),
        }
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_full_scores_preserve_the_native_precision_and_caller_frame(dtype):
    """Accept complete scores without converting values or copying their table.

    Args:
        dtype: Native stored score precision.

    Returns:
        None. Construction retains the original table and exact score values.

    Examples:
        pytest tests/test_cell_scores.py -k native_precision

    Steps:
        1. Accept complete scores without converting values or copying their table.
        2. Check that construction retains the original table and exact score values.
    """
    native = np.dtype(dtype)
    below = np.nextafter(native.type(1), native.type(0))
    frame = score_frame(((0, 0, below), (0.25, 0, 1), (0.5, 0, 1.25)), dtype=dtype)
    scores = WaveCellScores(frame, DAY)
    assert scores.frame is frame
    assert scores.frame.score.dtype == native
    assert (
        scores.frame.loc[
            scores.frame.longitude.eq(0) & scores.frame.latitude.eq(0), "score"
        ].iloc[0]
        == below
    )


@pytest.mark.parametrize(
    "kind", ["missing", "duplicate", "off-grid", "wrong-longitude"]
)
def test_incomplete_duplicate_or_shifted_grid_is_rejected(kind):
    """Require exactly one score for every supported global coordinate.

    Args:
        kind: How an otherwise complete grid is corrupted.

    Returns:
        None. Invalid coverage raises before any cell history is updated.

    Examples:
        pytest tests/test_cell_scores.py -k shifted_grid

    Steps:
        1. Require exactly one score for every supported global coordinate.
        2. Check that invalid coverage raises before any cell history is updated.
    """
    frame = score_frame()
    if kind == "missing":
        frame = frame.iloc[:-1]
    elif kind == "duplicate":
        frame.iloc[-1] = frame.iloc[0]
    elif kind == "off-grid":
        frame.loc[0, "latitude"] = -89.9
    else:
        frame.loc[0, "longitude"] = -180
    with pytest.raises(WaveCellError):
        WaveCellScores(frame, DAY)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_scores_are_rejected(value):
    """Missing or infinite scores cannot silently end an existing history.

    Args:
        value: Invalid score substituted for one supported coordinate.

    Returns:
        None. Construction rejects the invalid measurement.

    Examples:
        pytest tests/test_cell_scores.py -k nonfinite

    Steps:
        1. Missing or infinite scores cannot silently end an existing history.
        2. Check that construction rejects the invalid measurement.
    """
    frame = score_frame()
    frame.loc[0, "score"] = value
    with pytest.raises(WaveCellError):
        WaveCellScores(frame, DAY)


@pytest.mark.parametrize("dtype", ["int64", "bool", "str"])
def test_score_columns_must_use_floating_point_storage(dtype):
    """Reject integer, boolean and textual score storage without conversion.

    Args:
        dtype: Unsupported type to apply to the score column.

    Returns:
        None. Construction raises instead of coercing the measurements.

    Examples:
        pytest tests/test_cell_scores.py -k floating_point_storage

    Steps:
        1. Reject integer, boolean and textual score storage without conversion.
        2. Check that construction raises instead of coercing the measurements.
    """
    frame = score_frame()
    frame["score"] = frame.score.astype(dtype)
    with pytest.raises(WaveCellError):
        WaveCellScores(frame, DAY)


@pytest.mark.parametrize(
    ("temperature", "direction", "column"),
    [
        ("tasmax", "warm", "t2m_max_p90_score"),
        ("tasmax", "cold", "t2m_max_p10_score"),
        ("tasmin", "warm", "t2m_min_p90_score"),
        ("tasmin", "cold", "t2m_min_p10_score"),
    ],
)
def test_reader_selects_only_the_requested_stored_score_and_preserves_float32(
    tmp_path, temperature, direction, column
):
    """Read one native score series from a file containing several series.

    Args:
        tmp_path: Temporary directory supplied by pytest.
        temperature: Maximum or minimum temperature selection.
        direction: Warm or cold direction.
        column: Stored field corresponding to the requested series.

    Returns:
        None. Only the selected score is returned under its three-column schema.

    Examples:
        pytest tests/test_cell_scores.py -k requested_stored_score

    Steps:
        1. Read one native score series from a file containing several series.
        2. Check that only the selected score is returned under its three-column schema.
    """
    frame = score_frame(((0, 0, 1.125),), dtype="float32")
    stored = frame.rename(columns={"score": column}).assign(unused_score=-999)
    path = tmp_path / f"{DAY}.parquet"
    stored.to_parquet(path, index=False)
    scores = WaveCellScores.read_parquet(path, DAY, WaveSeries(temperature, direction))
    pd.testing.assert_frame_equal(scores.frame, frame)
    source = WaveCellScoreSource(str(tmp_path))
    pd.testing.assert_frame_equal(
        source.read_scores(DAY, WaveSeries(temperature, direction)).frame, frame
    )


def test_score_source_uses_flat_dated_files_and_missing_input_stays_an_error(tmp_path):
    """Look up score files by date directly beneath the configured input root.

    Args:
        tmp_path: Temporary directory supplied by pytest.

    Returns:
        None. Missing files cannot stand in for complete nonqualifying score grids.

    Examples:
        pytest tests/test_cell_scores.py -k flat_dated

    Steps:
        1. Look up score files by date directly beneath the configured input root.
        2. Check that missing files cannot stand in for complete nonqualifying score grids.
    """
    source = WaveCellScoreSource(str(tmp_path))
    series = WaveSeries("tasmax", "warm")
    assert source.path_for(DAY, series) == str(tmp_path / f"{DAY}.parquet")
    with pytest.raises(FileNotFoundError):
        source.read_scores(DAY, series)


def test_reader_on_a_supplied_filesystem_keeps_scores_and_missing_column_errors():
    """Honor the supplied filesystem while requiring the selected score column.

    Args:
        None.

    Returns:
        None. Scores survive filesystem I/O and missing measurements raise.

    Examples:
        pytest tests/test_cell_scores.py -k supplied_filesystem

    Steps:
        1. Honor the supplied filesystem while requiring the selected score column.
        2. Check that scores survive filesystem I/O and missing measurements raise.
    """
    filesystem = fsspec.filesystem("memory")
    path = "/score-input-test/2024-07-01.parquet"
    frame = score_frame(((180, 0, 1),))
    stored = frame.rename(columns={"score": "t2m_max_p90_score"})
    filesystem.makedirs("/score-input-test", exist_ok=True)
    with filesystem.open(path, "wb") as stream:
        stored.to_parquet(stream, index=False)
    restored = WaveCellScores.read_parquet(
        path, DAY, WaveSeries("tasmax", "warm"), filesystem=filesystem
    )
    pd.testing.assert_frame_equal(restored.frame, frame)
    with pytest.raises(WaveCellError):
        WaveCellScores.read_parquet(
            path, DAY, WaveSeries("tasmin", "cold"), filesystem=filesystem
        )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_writer_round_trip_preserves_the_exact_native_values(tmp_path, dtype):
    """Store selected scientific field names without rounding native scores.

    Args:
        tmp_path: Temporary directory supplied by pytest.
        dtype: Native score precision to retain while writing.

    Returns:
        None. Field projection, precision and exact values survive the round trip.

    Examples:
        pytest tests/test_cell_scores.py -k writer_round_trip

    Steps:
        1. Store selected scientific field names without rounding native scores.
        2. Check that field projection, precision and exact values survive the round trip.
    """
    native = np.dtype(dtype)
    below = np.nextafter(native.type(1), native.type(0))
    frame = score_frame(((0, 0, below), (0.25, 0, 1)), dtype=dtype)
    scores = WaveCellScores(frame, DAY)
    series = WaveSeries("tasmin", "cold")
    path = tmp_path / "nested" / f"{DAY}.parquet"
    scores.to_parquet(path, series)
    stored = pd.read_parquet(path)
    assert list(stored.columns) == ["longitude", "latitude", "t2m_min_p10_score"]
    assert stored.t2m_min_p10_score.dtype == native
    restored = WaveCellScores.read_parquet(path, DAY, series)
    pd.testing.assert_frame_equal(restored.frame, frame)
