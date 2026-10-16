"""Verify daily anomaly science and reusable independent output dates."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from reuters_temperature_storms.anomalies import (
    ANOMALY_FIELDS,
    METRICS,
    AnomalyError,
    DailyAnomalies,
    DailyAnomalyBuilder,
    DailyClimatology,
    DailyTemperatures,
    SeasonalDistribution,
)
from reuters_temperature_storms.anomaly_run import AnomalyFileSource, AnomalyRun
from reuters_temperature_storms.cell_scores import WaveCellScoreSource
from reuters_temperature_storms.cli import main
from reuters_temperature_storms.series import WaveSeries

DAY = date(2024, 2, 29)


@pytest.fixture(scope="module")
def inputs():
    """Build full global inputs with distinct ordinary and adjusted means.

    Args:
        None.

    Returns:
        Checked daily, ordinary and adjusted input models.

    Examples:
        daily, ordinary, seasonal = inputs

    Steps:
        1. Build full global inputs with distinct ordinary and adjusted means.
    """
    coordinates = {
        "longitude": np.tile(np.arange(-179.75, 180.25, 0.25), 721),
        "latitude": np.repeat(np.arange(-90.0, 90.25, 0.25), 1440),
    }
    daily = pd.DataFrame(coordinates)
    normal = pd.DataFrame(coordinates)
    seasonal = pd.DataFrame(coordinates)
    daily["date"] = pd.Timestamp(DAY)
    for frame in (normal, seasonal):
        frame["month"], frame["day"] = DAY.month, DAY.day
    for index, metric in enumerate(METRICS):
        daily[metric] = 270.0 + index * 10
        normal[metric] = 268.0 + index * 10
        seasonal[metric] = 275.0 + index * 10
        seasonal[f"{metric}_p90"] = 277.0 + index * 10
        seasonal[f"{metric}_p10"] = 274.0 + index * 10
    # One cold cell distinguishes signed cold division from absolute denominators.
    daily.loc[0, list(METRICS)] -= 4
    return (
        DailyTemperatures(daily, DAY),
        DailyClimatology(normal.iloc[::-1].reset_index(drop=True), DAY),
        SeasonalDistribution(seasonal, DAY),
    )


def test_ordinary_delta_adjusted_scale_alignment_and_complete_output(inputs):
    """Distinguish the Reuters formula from adjusted-mean subtraction.

    Args:
        inputs: Complete global inputs with reversed ordinary row order.

    Returns:
        None. All four scores match their independently stated values.

    Examples:
        pytest tests/test_anomalies.py -k ordinary_delta

    Steps:
        1. Distinguish the Reuters formula from adjusted-mean subtraction.
        2. Check that all four scores match their independently stated values.
    """
    result = DailyAnomalyBuilder().build(*inputs)
    assert len(result.frame) == 1038240
    assert tuple(result.frame.columns) == (
        "longitude",
        "latitude",
        "date",
        *ANOMALY_FIELDS,
    )
    for metric in METRICS:
        np.testing.assert_array_equal(
            result.frame[f"{metric}_p90_score"].iloc[:2], [-1.0, 1.0]
        )
        np.testing.assert_array_equal(
            result.frame[f"{metric}_p10_score"].iloc[:2], [2.0, -2.0]
        )
    assert all(result.frame[column].dtype == "float64" for column in ANOMALY_FIELDS)


@pytest.mark.parametrize(
    "damage", ["missing", "duplicate", "date", "infinite", "off_grid"]
)
def test_inputs_reject_incomplete_or_invalid_cells(inputs, damage):
    """Reject damaged global inputs before they can become nonqualifying cells.

    Args:
        inputs: Valid complete global input models.
        damage: Independently invalid coverage, date or measurement condition.

    Returns:
        None. Model construction rejects each condition.

    Examples:
        pytest tests/test_anomalies.py -k incomplete

    Steps:
        1. Reject damaged global inputs before they can become nonqualifying cells.
        2. Check that model construction rejects each condition.
    """
    frame = inputs[0].frame.copy()
    if damage == "missing":
        frame = frame.iloc[:-1]
    elif damage == "duplicate":
        frame.loc[1, ["longitude", "latitude"]] = frame.loc[
            0, ["longitude", "latitude"]
        ]
    elif damage == "date":
        frame.loc[0, "date"] = pd.Timestamp("2024-03-01")
    elif damage == "infinite":
        frame.loc[0, "t2m_max"] = np.inf
    else:
        frame.loc[0, "longitude"] += 0.125
    with pytest.raises(AnomalyError):
        DailyTemperatures(frame, DAY)


@pytest.mark.parametrize(
    ("column", "value"),
    [("t2m_max_p90", 285.0), ("t2m_max_p10", 300.0)],
)
def test_builder_rejects_invalid_signed_scales(inputs, column, value):
    """Reject zero and wrong-tail denominators before division.

    Args:
        inputs: Valid complete input models.
        column: Adjusted distribution field to damage.
        value: Invalid denominator-producing measurement.

    Returns:
        None. Builder rejects the invalid scale.

    Examples:
        pytest tests/test_anomalies.py -k signed_scales

    Steps:
        1. Reject zero and wrong-tail denominators before division.
        2. Check that builder rejects the invalid scale.
    """
    frame = inputs[2].frame.copy()
    frame.loc[0, column] = value
    seasonal = SeasonalDistribution(frame, DAY)
    with pytest.raises(AnomalyError, match="denominators"):
        DailyAnomalyBuilder().build(inputs[0], inputs[1], seasonal)


def test_run_reuse_cli_and_score_stage_compatibility(inputs, tmp_path):
    """Generate real Parquets, consume all four series and reuse without inputs.

    Args:
        inputs: Checked complete daily and calendar baseline grids.
        tmp_path: Independent input and output directories.

    Returns:
        None. New output, selective score reading, reuse and checks succeed.

    Examples:
        pytest tests/test_anomalies.py -k run_reuse

    Steps:
        1. Generate real Parquets, consume all four series and reuse without inputs.
        2. Check that new output, selective score reading, reuse and checks succeed.
    """
    roots = [
        AnomalyFileSource(
            str(tmp_path / name),
            filename_template="{date}.parquet"
            if name in ("daily", "output")
            else "{month:02d}-{day:02d}.parquet",
        )
        for name in ("daily", "normal", "seasonal", "output")
    ]
    for source, grid in zip(roots, inputs, strict=False):
        source._filesystem.makedirs(source._path, exist_ok=True)
        # Demonstrate archive float32 is expanded before subtraction.
        frame = grid.frame.copy()
        for metric in METRICS:
            frame[metric] = frame[metric].astype("float32")
        frame["t2m_mean"] = "unused daily mean"
        frame["t2m_max_std"] = "unused standard deviation"
        frame.to_parquet(source.path_for(DAY), index=False)
    job = AnomalyRun(*roots, DAY, DAY, max_workers=1)
    first = job.run()
    assert len(first) == 1
    assert not first[0].skipped
    for selection in WaveSeries.choices():
        scores = WaveCellScoreSource(roots[-1].location).read_scores(
            DAY, WaveSeries.parse(selection)
        )
        assert len(scores.frame) == 1038240
    for source in roots[:3]:
        source._filesystem.rm(source._path, recursive=True)
    assert job.run()[0].skipped
    assert job.check() == ()
    tomorrow = DAY + timedelta(days=1)
    output_frame = pd.read_parquet(roots[-1].path_for(DAY))
    output_frame["date"] = pd.Timestamp(tomorrow)
    output_frame.to_parquet(roots[-1].path_for(tomorrow), index=False)
    parallel = AnomalyRun(*roots, DAY, tomorrow, max_workers=2)
    completed = parallel.run()
    assert tuple(result.day for result in completed) == (DAY, tomorrow)
    assert all(result.skipped for result in completed)
    with pytest.raises(AnomalyError, match="failed for 1 date"):
        AnomalyRun(*roots, DAY, DAY, max_workers=1, force_update=True).run()
    assert job.check() == ()
    result = CliRunner().invoke(
        main,
        [
            "anomalies",
            "--temperatures",
            roots[0].location,
            "--climatology",
            roots[1].location,
            "--seasonal-distribution",
            roots[2].location,
            "--output",
            roots[3].location,
            "--start-date",
            str(DAY),
            "--end-date",
            str(DAY),
            "--check-only",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Verified complete daily anomaly dates" in result.output


def test_float32_archive_is_expanded_before_subtraction(inputs, tmp_path):
    """Preserve the exact promoted difference rather than float32 subtraction.

    Args:
        inputs: Complete valid temperature and baseline models.
        tmp_path: Independent float32 source file directory.

    Returns:
        None. Stored float32 inputs produce float64 differences and scales.

    Examples:
        pytest tests/test_anomalies.py -k float32_archive

    Steps:
        1. Preserve the exact promoted difference rather than float32 subtraction.
        2. Check that stored float32 inputs produce float64 differences and scales.
    """
    ordinary_frame = inputs[1].frame.copy()
    seasonal_frame = inputs[2].frame.copy()
    daily_frame = inputs[0].frame.copy()
    for frame in (ordinary_frame, seasonal_frame, daily_frame):
        for column in frame:
            if column.startswith("t2m_"):
                frame[column] = frame[column].astype("float32")
    # Baseline value with vastly different magnitude makes float32 subtraction
    # measurably distinct from expansion followed by subtraction.
    ordinary_frame.loc[:, "t2m_min"] = np.float32(0.001)
    path = tmp_path / f"{DAY}.parquet"
    daily_frame.to_parquet(path, index=False)
    source = AnomalyFileSource(str(tmp_path))
    daily = source.read_grid(DAY, DailyTemperatures)
    for frame in (ordinary_frame, seasonal_frame):
        for column in frame:
            if column.startswith("t2m_"):
                frame[column] = frame[column].astype("float64")
    result = DailyAnomalyBuilder().build(
        daily,
        DailyClimatology(ordinary_frame, DAY),
        SeasonalDistribution(seasonal_frame, DAY),
    )
    expected = float(np.float32(266.0)) - float(np.float32(0.001))
    assert result.frame.loc[0, "t2m_min_p90_score"] == expected / 2
    assert (
        result.frame.loc[0, "t2m_min_p90_score"]
        != float(np.float32(266.0) - np.float32(0.001)) / 2
    )


def test_output_root_and_check_only_force_are_rejected(tmp_path):
    """Prevent source replacement and mutually exclusive CLI modes.

    Args:
        tmp_path: Independent root with no input reads required.

    Returns:
        None. Both unsafe settings fail before processing dates.

    Examples:
        pytest tests/test_anomalies.py -k output_root

    Steps:
        1. Prevent source replacement and mutually exclusive CLI modes.
        2. Check that both unsafe settings fail before processing dates.
    """
    daily = AnomalyFileSource(str(tmp_path / "daily"))
    normal = AnomalyFileSource(
        str(tmp_path / "normal"), filename_template="{month:02d}-{day:02d}.parquet"
    )
    seasonal = AnomalyFileSource(
        str(tmp_path / "seasonal"), filename_template="{month:02d}-{day:02d}.parquet"
    )
    with pytest.raises(AnomalyError, match="separate"):
        AnomalyRun(daily, normal, seasonal, daily, DAY, DAY).validate()
    result = CliRunner().invoke(
        main,
        [
            "anomalies",
            "--temperatures",
            daily.location,
            "--climatology",
            normal.location,
            "--seasonal-distribution",
            seasonal.location,
            "--output",
            str(tmp_path / "output"),
            "--start-date",
            str(DAY),
            "--end-date",
            str(DAY),
            "--check-only",
            "--force-update",
        ],
    )
    assert result.exit_code != 0
    assert "cannot be combined" in result.output


@pytest.mark.parametrize("clock", ["10:04:56.147662", "23:59:59.999999"])
def test_temperature_reader_accepts_same_day_times(inputs, tmp_path, clock):
    """Remove raw temperature times without changing measurements or scores.

    Args:
        inputs: Valid complete daily and baseline grids.
        tmp_path: Isolated local input directory.
        clock: Non-midnight time within the requested calendar date.

    Returns:
        None. Loaded values and all four calculated scores match the control.

    Examples:
        Run pytest tests/test_anomalies.py -k same_day_times.

    Steps:
        1. Remove raw temperature times without changing measurements or scores.
        2. Check that loaded values and all four calculated scores match the control.
    """
    source = AnomalyFileSource(str(tmp_path))
    frame = inputs[0].frame.copy()
    frame["date"] = pd.Timestamp(f"{DAY} {clock}")
    # Different times across cells still refer to the same daily aggregate.
    frame.loc[0, "date"] = pd.Timestamp(DAY)
    frame.to_parquet(source.path_for(DAY), index=False)
    loaded = source.read_grid(DAY, DailyTemperatures)
    expected_input = inputs[0].frame.copy()
    expected_input["date"] = expected_input["date"].astype(loaded.frame["date"].dtype)
    pd.testing.assert_frame_equal(loaded.frame, expected_input)
    actual = DailyAnomalyBuilder().build(loaded, *inputs[1:])
    expected = DailyAnomalyBuilder().build(*inputs)
    expected.frame["date"] = expected.frame["date"].astype(actual.frame["date"].dtype)
    pd.testing.assert_frame_equal(actual.frame, expected.frame)
    stored = pd.read_parquet(source.path_for(DAY))
    pd.testing.assert_frame_equal(stored, frame)


@pytest.mark.parametrize(
    "damage", ["wrong_day", "mixed_day", "missing", "timezone", "text"]
)
def test_temperature_reader_rejects_unsafe_dates(inputs, tmp_path, damage):
    """Keep wrong, missing and ambiguous raw temperature dates rejected.

    Args:
        inputs: Valid complete daily and baseline grids.
        tmp_path: Isolated local input directory.
        damage: Invalid date condition to store in the daily file.

    Returns:
        None. The reader rejects the file before building daily scores.

    Examples:
        Run pytest tests/test_anomalies.py -k unsafe_dates.

    Steps:
        1. Keep wrong, missing and ambiguous raw temperature dates rejected.
        2. Check that the reader rejects the file before building daily scores.
    """
    source = AnomalyFileSource(str(tmp_path))
    frame = inputs[0].frame.copy()
    if damage == "wrong_day":
        frame["date"] = pd.Timestamp(DAY + timedelta(days=1))
    elif damage == "mixed_day":
        frame.loc[0, "date"] = pd.Timestamp(DAY + timedelta(days=1))
    elif damage == "missing":
        frame.loc[0, "date"] = pd.NaT
    elif damage == "timezone":
        frame["date"] = frame["date"].dt.tz_localize("UTC")
    else:
        frame["date"] = frame["date"].astype(str)
    frame.to_parquet(source.path_for(DAY), index=False)
    with pytest.raises(AnomalyError, match=r"temperature.*dates"):
        source.read_grid(DAY, DailyTemperatures)


def test_saved_anomaly_dates_remain_strict(inputs, tmp_path):
    """Reject a saved score file with times instead of midnight dates.

    Args:
        inputs: Valid complete daily and baseline grids.
        tmp_path: Isolated local output directory.

    Returns:
        None. Stored anomaly validation does not normalize malformed output.

    Examples:
        Run pytest tests/test_anomalies.py -k remain_strict.

    Steps:
        1. Reject a saved score file with times instead of midnight dates.
        2. Check that stored anomaly validation does not normalize malformed output.
    """
    source = AnomalyFileSource(str(tmp_path))
    frame = DailyAnomalyBuilder().build(*inputs).frame
    frame["date"] += pd.Timedelta(hours=10)
    frame.to_parquet(source.path_for(DAY), index=False)
    with pytest.raises(AnomalyError, match="calendar_dates"):
        source.read_grid(DAY, DailyAnomalies)
