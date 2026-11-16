"""Verify opt-in regional anomaly runs without loosening the global default."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from reuters_temperature_storms.anomalies import (
    METRICS,
    AnomalyError,
    DailyAnomalyBuilder,
    DailyClimatology,
    DailyTemperatures,
    GridRegion,
    SeasonalDistribution,
    TemperatureSchema,
)
from reuters_temperature_storms.anomaly_run import AnomalyFileSource, AnomalyRun
from reuters_temperature_storms.cli import main

DAY = date(2024, 2, 29)
REGION = GridRegion(-97.75, 30.0, -97.25, 30.5)


def regional_frames(day=DAY, region=REGION):
    """Build daily, ordinary and adjusted tables for every cell of a region.

    Args:
        day: Date stored in the daily table and calendar labels.
        region: Rectangle whose cells the tables hold.

    Returns:
        Daily, ordinary and adjusted frames with cell-specific temperatures.

    Examples:
        daily, normal, seasonal = regional_frames()

    Steps:
        1. Build every cell of the region once.
        2. Give each cell distinct values so misaligned joins change scores.
    """
    longitudes = np.arange(region.west, region.east + 0.125, 0.25)
    latitudes = np.arange(region.south, region.north + 0.125, 0.25)
    coordinates = {
        "longitude": np.tile(longitudes, len(latitudes)),
        "latitude": np.repeat(latitudes, len(longitudes)),
    }
    offsets = np.arange(len(coordinates["longitude"]), dtype="float64")
    daily = pd.DataFrame(coordinates)
    normal = pd.DataFrame(coordinates)
    seasonal = pd.DataFrame(coordinates)
    daily["date"] = pd.Timestamp(day)
    for frame in (normal, seasonal):
        frame["month"], frame["day"] = day.month, day.day
    for index, metric in enumerate(METRICS):
        daily[metric] = 300.0 + index * 10 + offsets
        normal[metric] = 298.0 + index * 10 + offsets
        seasonal[metric] = 299.0 + index * 10 + offsets
        seasonal[f"{metric}_p90"] = 303.0 + index * 10 + offsets
        seasonal[f"{metric}_p10"] = 297.0 + index * 10 + offsets
    return daily, normal.iloc[::-1].reset_index(drop=True), seasonal


def test_regional_scores_follow_the_reuters_formula():
    """Score every regional cell with the unchanged anomaly formula.

    Args:
        None.

    Returns:
        None. The nine regional cells carry the hand-calculated scores.

    Examples:
        Run pytest tests/test_anomaly_regions.py -k formula.

    Steps:
        1. Build regional inputs whose ordinary rows are reversed.
        2. Check row count, region retention and all four scores.
    """
    daily, normal, seasonal = regional_frames()
    result = DailyAnomalyBuilder().build(
        DailyTemperatures(daily, DAY, REGION),
        DailyClimatology(normal, DAY, REGION),
        SeasonalDistribution(seasonal, DAY, REGION),
    )
    assert REGION.cells == len(result.frame) == 9
    assert result.region == REGION
    for metric in METRICS:
        # Daily minus ordinary is 2; adjusted p90 and p10 sit 4 and -2 from the mean.
        np.testing.assert_array_equal(result.frame[f"{metric}_p90_score"], 0.5)
        np.testing.assert_array_equal(result.frame[f"{metric}_p10_score"], -1.0)


def test_global_default_still_rejects_a_regional_table():
    """Keep the complete global grid as the requirement when no region is given.

    Args:
        None.

    Returns:
        None. A regional table without a region is refused.

    Examples:
        Run pytest tests/test_anomaly_regions.py -k global_default.

    Steps:
        1. Construct a temperature grid from regional rows without a region.
        2. Check that the global coverage rule rejects it.
    """
    daily, _, _ = regional_frames()
    with pytest.raises(AnomalyError, match="complete_grid"):
        DailyTemperatures(daily, DAY)


def test_regional_validation_leaves_the_cached_global_schema_intact():
    """Keep complete_grid on the shared schema after a regional validation.

    Args:
        None.

    Returns:
        None. The cached global schema still requires the complete grid.

    Examples:
        Run pytest tests/test_anomaly_regions.py -k cached.

    Steps:
        1. Validate a regional temperature table.
        2. Check that the cached global schema still carries complete_grid.
    """
    daily, _, _ = regional_frames()
    DailyTemperatures(daily, DAY, REGION)
    names = [check.name for check in TemperatureSchema.to_schema().checks]
    assert "complete_grid" in names


@pytest.mark.parametrize(
    "damage", ["missing", "outside", "duplicate", "off_grid", "infinite"]
)
def test_regional_tables_reject_incomplete_or_invalid_cells(damage):
    """Apply every other schema rule and exact regional coverage.

    Args:
        damage: Coverage, coordinate or measurement defect.

    Returns:
        None. Each damaged regional table is refused.

    Examples:
        Run pytest tests/test_anomaly_regions.py -k incomplete.

    Steps:
        1. Damage one valid regional temperature table.
        2. Check that regional validation rejects the defect.
    """
    frame, _, _ = regional_frames()
    if damage == "missing":
        frame = frame.iloc[:-1]
    elif damage == "outside":
        extra = frame.iloc[[0]].assign(longitude=-98.0)
        frame = pd.concat([frame.iloc[1:], extra], ignore_index=True)
    elif damage == "duplicate":
        frame.loc[1, ["longitude", "latitude"]] = frame.loc[
            0, ["longitude", "latitude"]
        ]
    elif damage == "off_grid":
        frame.loc[0, "longitude"] += 0.125
    else:
        frame.loc[0, "t2m_max"] = np.inf
    with pytest.raises(AnomalyError):
        DailyTemperatures(frame, DAY, REGION)


@pytest.mark.parametrize(
    "text",
    [
        "-97.75,30,-97.25",
        "west,30,-97.25,30.5",
        "-97.8,30,-97.25,30.5",
        "-97.25,30,-97.75,30.5",
        "-97.75,30.5,-97.25,30",
        "-180,30,-97.25,30.5",
        "-97.75,-90.25,-97.25,30.5",
        "nan,30,-97.25,30.5",
    ],
)
def test_region_bounds_reject_invalid_rectangles(text):
    """Refuse bounds that cannot name an ordered quarter-degree rectangle.

    Args:
        text: Option value with a count, number, grid, order or range defect.

    Returns:
        None. Parsing raises AnomalyError.

    Examples:
        Run pytest tests/test_anomaly_regions.py -k bounds.

    Steps:
        1. Parse one invalid option value.
        2. Check that the region is refused.
    """
    with pytest.raises(AnomalyError):
        GridRegion.parse(text)


def test_region_bounds_reject_oversized_integers():
    """Report bounds too large for floating point as invalid regions.

    Args:
        None.

    Returns:
        None. Construction raises AnomalyError rather than a numpy error.

    Examples:
        Run pytest tests/test_anomaly_regions.py -k oversized.

    Steps:
        1. Construct a region with an integer beyond the float range.
        2. Check that the region is refused with AnomalyError.
    """
    with pytest.raises(AnomalyError):
        GridRegion(10**400, 0, 0, 0)


def test_builder_rejects_mismatched_regions():
    """Refuse daily and baseline grids validated against different regions.

    Args:
        None.

    Returns:
        None. The builder raises before calculating scores.

    Examples:
        Run pytest tests/test_anomaly_regions.py -k mismatched.

    Steps:
        1. Validate a one-cell region three times, once with an equal but separate region.
        2. Replace one region with a different rectangle and check the refusal.
    """
    single = GridRegion(-97.75, 30.25, -97.75, 30.25)
    daily, normal, seasonal = regional_frames(region=single)
    other = GridRegion(-97.5, 30.25, -97.5, 30.25)
    shifted = normal.assign(longitude=-97.5)
    builder = DailyAnomalyBuilder()
    builder.build(
        DailyTemperatures(daily, DAY, single),
        DailyClimatology(normal, DAY, GridRegion.parse("-97.75,30.25,-97.75,30.25")),
        SeasonalDistribution(seasonal, DAY, single),
    )
    with pytest.raises(AnomalyError, match="regions must match"):
        builder.build(
            DailyTemperatures(daily, DAY, single),
            DailyClimatology(shifted, DAY, other),
            SeasonalDistribution(seasonal, DAY, single),
        )


def test_cli_region_runs_workers_and_checks_outputs(tmp_path):
    """Run two regional dates in worker processes through the command line.

    Args:
        tmp_path: Independent input and output directories.

    Returns:
        None. Regional outputs are written and verified only with the region.

    Examples:
        Run pytest tests/test_anomaly_regions.py -k cli_region.

    Steps:
        1. Write regional inputs for two dates and run rts anomalies with --region.
        2. Check regional output, regional check-only success and global refusal.
    """
    roots = {name: tmp_path / name for name in ("daily", "normal", "seasonal", "out")}
    for root in roots.values():
        root.mkdir()
    days = (DAY, DAY + timedelta(days=1))
    for day in days:
        daily, normal, seasonal = regional_frames(day)
        daily.to_parquet(roots["daily"] / f"{day}.parquet", index=False)
        calendar = f"{day.month:02d}-{day.day:02d}.parquet"
        normal.to_parquet(roots["normal"] / calendar, index=False)
        seasonal.to_parquet(roots["seasonal"] / calendar, index=False)
    common = [
        "anomalies",
        "--temperatures",
        str(roots["daily"]),
        "--climatology",
        str(roots["normal"]),
        "--seasonal-distribution",
        str(roots["seasonal"]),
        "--output",
        str(roots["out"]),
        "--start-date",
        str(days[0]),
        "--end-date",
        str(days[1]),
    ]
    region = ["--region", "-97.75,30,-97.25,30.5"]
    result = CliRunner().invoke(main, [*common, *region, "--max-workers", "2"])
    assert result.exit_code == 0, result.output
    assert result.output.count(": 9 cells") == 2
    for day in days:
        output = pd.read_parquet(roots["out"] / f"{day}.parquet")
        assert len(output) == 9
        assert output["date"].eq(pd.Timestamp(day)).all()
    checked = CliRunner().invoke(main, [*common, *region, "--check-only"])
    assert checked.exit_code == 0, checked.output
    unscoped = CliRunner().invoke(main, [*common, "--check-only"])
    assert unscoped.exit_code != 0
    assert "complete_grid" in unscoped.output
    job = AnomalyRun(
        AnomalyFileSource(str(roots["daily"])),
        AnomalyFileSource(
            str(roots["normal"]), filename_template="{month:02d}-{day:02d}.parquet"
        ),
        AnomalyFileSource(
            str(roots["seasonal"]), filename_template="{month:02d}-{day:02d}.parquet"
        ),
        AnomalyFileSource(str(roots["out"])),
        days[0],
        days[1],
        region="-97.75,30,-97.25,30.5",
    )
    with pytest.raises(AnomalyError, match="GridRegion"):
        job.check()
