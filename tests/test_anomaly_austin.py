"""Regress the Austin ERA5 sample without a private repository dependency.

Values are from the 2023-08-17 daily, climatology and seasonal Parquets in
saulbanov/heat-storms-sample-inputs (Git blob SHAs 1036edb36dd45dca5e7e4bba90de6bd9c5c967e1,
518c4795bf8bf798957d20bf62f45a39a3c7916c, and
24884f386b80fa8ec92af9432a557fbb4917a74b).
Weather data by Open-Meteo.com, CC BY 4.0. Contains modified Copernicus Climate
Change Service information: hourly ERA5 reduced to UTC daily extremes and
1991-2020 calendar-day means and percentiles. Independent sample, not Reuters data.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from reuters_temperature_storms.anomalies import (
    METRICS,
    AnomalyError,
    DailyAnomalies,
    DailyAnomalyBuilder,
    DailyClimatology,
    DailyTemperatures,
    GridRegion,
    SeasonalDistribution,
)
from reuters_temperature_storms.anomaly_run import AnomalyFileSource, AnomalyRun

DAY = date(2023, 8, 17)
REGION = GridRegion(-97.75, 30.25, -97.75, 30.25)


@pytest.fixture
def austin():
    """Return the exact published one-cell sample values for one hot day."""
    coordinates = {"longitude": [-97.75], "latitude": [30.25]}
    daily = pd.DataFrame(
        {
            **coordinates,
            "date": [pd.Timestamp(DAY)],
            "t2m_min": [25.0],
            "t2m_max": [41.7],
        }
    )
    normal = pd.DataFrame(
        {
            **coordinates,
            "month": [8],
            "day": [17],
            "t2m_min": [24.08511111111111],
            "t2m_max": [35.55177777777778],
        }
    )
    seasonal = normal.assign(
        t2m_min_p10=22.090000000000003,
        t2m_min_p90=25.9,
        t2m_max_p10=31.790000000000003,
        t2m_max_p90=38.6,
    )
    output = (
        DailyAnomalyBuilder()
        .build(
            DailyTemperatures(daily, DAY, REGION),
            DailyClimatology(normal, DAY, REGION),
            SeasonalDistribution(seasonal, DAY, REGION),
        )
        .frame
    )
    return daily, normal, seasonal, output


MODELS = (DailyTemperatures, DailyClimatology, SeasonalDistribution, DailyAnomalies)


@pytest.mark.parametrize(("index", "model"), tuple(enumerate(MODELS)))
@pytest.mark.parametrize(
    "damage",
    [
        "default",
        "empty",
        "duplicate",
        "off_grid",
        "out_of_range",
        "nonfinite",
        "wrong_cell",
    ],
)
def test_austin_rejects_invalid_grids(austin, index, model, damage):
    """Retain global defaults and coordinate rules for every measurement model."""
    frame = austin[index].copy()
    region = REGION
    if damage == "default":
        region = None
    elif damage == "empty":
        frame = frame.iloc[:0]
    elif damage == "duplicate":
        frame = pd.concat([frame, frame], ignore_index=True)
    elif damage == "off_grid":
        frame["latitude"] = 30.3
    elif damage == "out_of_range":
        frame["latitude"] = 90.25
    elif damage == "nonfinite":
        frame["longitude"] = np.inf
    else:
        frame["longitude"] = -97.5
    with pytest.raises(AnomalyError, match=f"invalid {model.__name__}:"):
        model(frame, DAY, region)
    # A regional check must never remove complete_grid from the cached schema.
    with pytest.raises(AnomalyError, match="complete_grid"):
        model(austin[index], DAY)


def test_austin_worker_run_reuse_and_independent_scores(austin, tmp_path):
    """Run real sample values in workers, then verify reuse and all four scores."""
    roots = [tmp_path / name for name in ("daily", "normal", "seasonal", "output")]
    for root, frame, filename in zip(
        roots[:3],
        austin[:3],
        (f"{DAY}.parquet", "08-17.parquet", "08-17.parquet"),
        strict=True,
    ):
        root.mkdir()
        frame.to_parquet(root / filename, index=False)
    sources = [
        AnomalyFileSource(
            str(root),
            filename_template="{month:02d}-{day:02d}.parquet"
            if index in (1, 2)
            else "{date}.parquet",
        )
        for index, root in enumerate(roots)
    ]
    job = AnomalyRun(*sources, DAY, DAY, max_workers=1, region=REGION)
    assert job.check() == (DAY,)
    assert job.run()[0].rows == 1
    assert job.check() == ()
    assert job.run()[0].skipped
    output = pd.read_parquet(roots[-1] / f"{DAY}.parquet").iloc[0]
    daily, normal, seasonal = (frame.iloc[0] for frame in austin[:3])
    for metric in METRICS:
        for percentile in ("p90", "p10"):
            expected = (daily[metric] - normal[metric]) / (
                seasonal[f"{metric}_{percentile}"] - seasonal[metric]
            )
            assert output[f"{metric}_{percentile}_score"] == expected
    assert output["t2m_max_p90_score"] == pytest.approx(2.02, abs=0.005)
