"""File-schema checks reject malformed inputs without changing science values."""

from datetime import datetime

import geopandas as gpd
import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors
from shapely.geometry import LineString, Polygon, box

from reuters_temperature_storms.schemas import (
    LineageFileSchema,
    LineageSchema,
    NameLibrarySchema,
    WaveCatalogFileSchema,
    WaveCatalogSchema,
    WaveDayFileSchema,
    WaveDaySchema,
    WaveFootprintSchema,
)

WAVE = "20200101-01234567"
OTHER_WAVE = "20200102-89abcdef"
COMPONENT = "0123456789abcdef0123456789abcdef"


def _dates(values):
    """Construct nanosecond timestamps from date-like values.

    Args:
        values: Date-like values to convert to nanosecond timestamps.

    Returns:
        Nanosecond-resolution pandas datetime values.

    Steps:
        1. Convert the supplied date-like values into pandas timestamps.
        2. Use nanosecond datetime storage for the result.

    Examples:
        _dates(["2024-01-01"])
    """
    return pd.to_datetime(values).astype("datetime64[ns]")


def _footprints():
    """Build one valid geographic footprint for schema checks.

    Returns:
        One-row geographic footprint table.

    Steps:
        1. Construct a single qualified box with geographic geometry columns.
        2. Populate the required dated identity and measured role fields.

    Examples:
        Used by ``pytest tests/test_schemas.py`` for fixture construction and output checks.
    """
    shape = box(0, 0, 1, 1)
    return gpd.GeoDataFrame(
        {
            "storm_id": [COMPONENT],
            "date": _dates(["2020-01-01"]),
            "geometry": gpd.GeoSeries([shape], crs="EPSG:4326"),
            "buffer_geometry": gpd.GeoSeries([shape], crs="EPSG:4326"),
            "grace_geometry": gpd.GeoSeries([None], crs="EPSG:4326"),
            "area_qualified": [1.0],
            "area_grace": [0.0],
            "area_gap_filled": [0.0],
            "qualified_fraction": [1.0],
        },
        geometry="buffer_geometry",
        crs="EPSG:4326",
    )


def _wave_days():
    """Build one valid daily wave table for schema checks.

    Returns:
        One-row geographic daily wave table.

    Steps:
        1. Use the synthetic footprint geometry for a continuing two-day wave.
        2. Populate daily identity, area, naming, and parent fields with valid typed values.

    Examples:
        Used by ``pytest tests/test_schemas.py`` for fixture construction and output checks.
    """
    footprint = _footprints()
    return gpd.GeoDataFrame(
        {
            "wave_id": [WAVE],
            "component_id": [COMPONENT],
            "name": pd.Series([None], dtype="str"),
            "date": _dates(["2020-01-02"]),
            "start_date": _dates(["2020-01-01"]),
            "component_label": [1],
            "continues": [True],
            "overlap_percent": [100.0],
            "parent_wave_ids": [[]],
            "parent_keys": [[]],
            "peak_qualified_area_km2": [1.0],
            "peak_date": _dates(["2020-01-01"]),
            "qualified_area_km2": [1.0],
            "grace_area_km2": [0.0],
            "gap_filled_area_km2": [0.0],
            "qualified_fraction": [1.0],
            "qualified_geometry": footprint["geometry"],
            "grace_geometry": footprint["grace_geometry"],
            "full_geometry": footprint["buffer_geometry"],
        },
        geometry="full_geometry",
        crs="EPSG:4326",
    )


def _catalog():
    """Build one active two-day wave catalog row for schema checks.

    Returns:
        One-row geographic catalog table.

    Steps:
        1. Construct an active two-day catalog entry with matching duration and observation count.
        2. Attach a geographic lifetime polygon and typed nullable ending fields.

    Examples:
        Used by ``pytest tests/test_schemas.py`` for fixture construction and output checks.
    """
    return gpd.GeoDataFrame(
        {
            "wave_id": [WAVE],
            "start_date": _dates(["2020-01-01"]),
            "last_observed_date": _dates(["2020-01-02"]),
            "end_date": _dates([None]),
            "end_reason": pd.Series([None], dtype="str"),
            "merged_into_wave_id": pd.Series([None], dtype="str"),
            "active": [True],
            "peak_qualified_area_km2": [1.0],
            "peak_date": _dates(["2020-01-01"]),
            "observation_count": [2],
            "name": pd.Series([None], dtype="str"),
            "duration_days": [2],
            "parent_keys": [[]],
            "total_area_visited_km2": [1.0],
            "lifetime_geometry": gpd.GeoSeries([box(0, 0, 1, 1)], crs="EPSG:4326"),
        },
        geometry="lifetime_geometry",
        crs="EPSG:4326",
    )


def _lineage():
    """Build a split relationship between two distinct wave identifiers.

    Returns:
        One-row lineage table.

    Steps:
        1. Construct a split edge between distinct persistent wave identifiers.
        2. Assign the edge its observation date.

    Examples:
        Used by ``pytest tests/test_schemas.py`` for fixture construction and output checks.
    """
    return pd.DataFrame(
        {
            "source_wave_id": [WAVE],
            "target_wave_id": [OTHER_WAVE],
            "event_date": _dates(["2020-01-02"]),
            "relationship": ["split"],
        }
    )


def test_footprint_validation_preserves_measurements_and_polygon_bytes():
    """Validate the table and compare stored precision and polygon bytes with the original inputs.

    Returns:
        None.

    Steps:
        1. Convert the synthetic footprint measurements to float32.
        2. Validate the table and compare stored precision and polygon bytes with the original inputs.

    Examples:
        Run ``pytest tests/test_schemas.py -k footprint_validation_preserves_measurements_and_polygon_bytes``.
    """
    frame = _footprints()
    shape_bytes = frame.geometry.iloc[0].wkb
    for field in (
        "area_qualified",
        "area_grace",
        "area_gap_filled",
        "qualified_fraction",
    ):
        frame[field] = frame[field].astype("float32")
    validated = WaveFootprintSchema.validate(frame)
    assert validated.geometry.iloc[0].wkb == shape_bytes
    pd.testing.assert_frame_equal(validated, frame)
    assert validated["area_qualified"].dtype == "float32"


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("area_qualified", -1.0),
        ("area_qualified", float("inf")),
        ("area_qualified", None),
        ("qualified_fraction", 0.5),
        ("qualified_fraction", float("nan")),
        ("storm_id", "broken-key"),
        ("date", datetime(2020, 1, 1, 0, 0, 0, 1)),  # noqa: DTZ001 - timezone-free file date.
        ("geometry", None),
    ],
)
def test_footprint_schema_rejects_invalid_values(column, value):
    """Check that schema validation rejects the changed table.

    Args:
        column: Input column to alter.
        value: Candidate scalar value selected by parametrization.

    Returns:
        None.

    Steps:
        1. Replace the selected footprint column with its parametrized invalid value.
        2. Check that schema validation rejects the changed table.

    Examples:
        Run ``pytest tests/test_schemas.py -k footprint_schema_rejects_invalid_values``.
    """
    frame = _footprints()
    frame.loc[0, column] = value
    with pytest.raises(SchemaError):
        WaveFootprintSchema.validate(frame)


def test_footprint_schema_rejects_integer_measurement_without_coercion():
    """Check that validation rejects the field rather than coercing its values.

    Returns:
        None.

    Steps:
        1. Convert a measured footprint field to integer storage.
        2. Check that validation rejects the field rather than coercing its values.

    Examples:
        Run ``pytest tests/test_schemas.py -k footprint_schema_rejects_integer_measurement_without_coercion``.
    """
    frame = _footprints()
    frame["area_qualified"] = pd.Series([1], dtype="int64")
    with pytest.raises(SchemaError, match="floating_measurements"):
        WaveFootprintSchema.validate(frame)
    assert frame["area_qualified"].dtype == "int64"


@pytest.mark.parametrize(
    "shape",
    [
        Polygon(),
        LineString([(0, 0), (1, 1)]),
        Polygon([(0, 0), (1, 1), (0, 1), (1, 0)]),
        Polygon([(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 0, 1)]),
        box(180, 0, 181, 1),
        box(0, 90, 1, 91),
    ],
)
def test_geometry_schema_rejects_invalid_or_non_geographic_polygons(shape):
    """Check that validation rejects its geometry type or shape.

    Args:
        shape: Candidate geometry selected by parametrization.

    Returns:
        None.

    Steps:
        1. Place the parametrized invalid geometry into the synthetic footprint.
        2. Check that validation rejects its geometry type or shape.

    Examples:
        Run ``pytest tests/test_schemas.py -k geometry_schema_rejects_invalid_or_non_geographic_polygons``.
    """
    frame = _footprints()
    frame.loc[0, "buffer_geometry"] = shape
    with pytest.raises(SchemaError, match="geographic_polygons"):
        WaveFootprintSchema.validate(frame)


def test_geometry_schema_rejects_wrong_crs_and_accepts_crs84():
    """Assign a projected CRS and check that validation rejects it.

    Returns:
        None.

    Steps:
        1. Validate the footprint under the supported geographic CRS84 identifier.
        2. Assign a projected CRS and check that validation rejects it.

    Examples:
        Run ``pytest tests/test_schemas.py -k geometry_schema_rejects_wrong_crs_and_accepts_crs84``.
    """
    frame = _footprints()
    frame["geometry"] = frame["geometry"].set_crs("EPSG:3857", allow_override=True)
    with pytest.raises(SchemaError, match="geographic_polygons"):
        WaveFootprintSchema.validate(frame)
    frame["geometry"] = frame["geometry"].set_crs("OGC:CRS84", allow_override=True)
    WaveFootprintSchema.validate(frame)


def test_footprint_schema_rejects_duplicate_ids_and_unknown_columns():
    """Check that the schema rejects both malformed table variants.

    Returns:
        None.

    Steps:
        1. Add a duplicate component identifier or an unsupported column to the synthetic table.
        2. Check that the schema rejects both malformed table variants.

    Examples:
        Run ``pytest tests/test_schemas.py -k footprint_schema_rejects_duplicate_ids_and_unknown_columns``.
    """
    frame = pd.concat([_footprints(), _footprints()], ignore_index=True)
    with pytest.raises(SchemaError, match="duplicate"):
        WaveFootprintSchema.validate(frame)
    frame = _footprints().assign(unexpected=True)
    with pytest.raises((SchemaError, SchemaErrors), match="unexpected"):
        WaveFootprintSchema.validate(frame)


@pytest.mark.parametrize(
    ("schema", "factory", "year_field"),
    [
        (WaveDaySchema, _wave_days, "observation_year"),
        (WaveCatalogSchema, _catalog, "start_year"),
        (LineageSchema, _lineage, "event_year"),
    ],
)
def test_partitioned_schema_rejects_series_and_year_mismatches(
    schema, factory, year_field
):
    """Alter its series and year metadata and check that validation rejects the mismatches.

    Args:
        schema: Schema class used to validate the synthetic table.
        factory: Callable constructing the synthetic input table.
        year_field: Partition year column to alter.

    Returns:
        None.

    Steps:
        1. Create the parametrized partitioned scientific table.
        2. Alter its series and year metadata and check that validation rejects the mismatches.

    Examples:
        Run ``pytest tests/test_schemas.py -k partitioned_schema_rejects_series_and_year_mismatches``.
    """
    frame = factory().assign(
        temperature="tasmax", direction="warm", **{year_field: 2020}
    )
    schema.validate(frame)
    frame[year_field] = 2019
    with pytest.raises(SchemaError):
        schema.validate(frame)
    frame[year_field] = 2020
    frame["direction"] = "positive"
    with pytest.raises(SchemaError, match="direction"):
        schema.validate(frame)


def test_wave_day_schema_rejects_invalid_dates_and_parent_list():
    """Check that validation rejects each inconsistent history or parent list.

    Returns:
        None.

    Steps:
        1. Change the daily-wave date fields and parent identifiers in separate table variants.
        2. Check that validation rejects each inconsistent history or parent list.

    Examples:
        Run ``pytest tests/test_schemas.py -k wave_day_schema_rejects_invalid_dates_and_parent_list``.
    """
    frame = _wave_days()
    WaveDayFileSchema.validate(frame)
    frame["peak_date"] = _dates(["2020-01-03"])
    with pytest.raises(SchemaError, match="wave_dates"):
        WaveDayFileSchema.validate(frame)
    frame["peak_date"] = _dates(["2020-01-01"])
    frame.at[0, "parent_keys"] = ["invalid-key"]
    with pytest.raises(SchemaError, match="wave_id_lists"):
        WaveDayFileSchema.validate(frame)


@pytest.mark.parametrize(
    ("schema", "factory"),
    [(WaveDayFileSchema, _wave_days), (WaveCatalogFileSchema, _catalog)],
)
@pytest.mark.parametrize("removed_id", ["storm_key", "storm_id"])
def test_output_schemas_use_one_permanent_wave_id(schema, factory, removed_id):
    """Validate outputs with one permanent ID and reject the removed duplicate.

    Args:
        schema: Daily or catalog file schema.
        factory: Constructor for a valid daily or catalog frame.
        removed_id: Previous identifier spelling rejected by the new output schema.

    Returns:
        None. The schema preserves the ID and rejects the extra column.

    Examples:
        Run ``pytest tests/test_schemas.py -k one_permanent_wave_id``.

    Steps:
        1. Validate outputs with one permanent ID and reject the removed duplicate.
        2. Check that the schema preserves the ID and rejects the extra column.
    """
    frame = factory()
    validated = schema.validate(frame)
    assert validated["wave_id"].tolist() == [WAVE]
    assert removed_id not in validated.columns
    frame[removed_id] = WAVE
    with pytest.raises(SchemaErrors, match=removed_id):
        schema.validate(frame)


def test_catalog_schema_requires_inclusive_duration_and_consistent_endings():
    """Check that validation rejects inconsistent duration or ending combinations.

    Returns:
        None.

    Steps:
        1. Alter the active catalog row's duration and ending fields.
        2. Check that validation rejects inconsistent duration or ending combinations.

    Examples:
        Run ``pytest tests/test_schemas.py -k catalog_schema_requires_inclusive_duration_and_consistent_endings``.
    """
    frame = _catalog()
    WaveCatalogFileSchema.validate(frame)
    frame["duration_days"] = 1
    with pytest.raises(SchemaError, match="catalog_history"):
        WaveCatalogFileSchema.validate(frame)
    frame["duration_days"] = 2
    frame["active"] = False
    with pytest.raises(SchemaError, match="catalog_history"):
        WaveCatalogFileSchema.validate(frame)
    frame["end_date"] = _dates(["2020-01-02"])
    frame["end_reason"] = "merged"
    frame["merged_into_wave_id"] = OTHER_WAVE
    WaveCatalogFileSchema.validate(frame)


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan"), None])
def test_catalog_schema_requires_finite_positive_visited_area(value):
    """Check that catalog validation rejects nonfinite or nonpositive measurements.

    Args:
        value: Candidate scalar value selected by parametrization.

    Returns:
        None.

    Steps:
        1. Replace lifetime visited area with the parametrized invalid value.
        2. Check that catalog validation rejects nonfinite or nonpositive measurements.

    Examples:
        Run ``pytest tests/test_schemas.py -k catalog_schema_requires_finite_positive_visited_area``.
    """
    frame = _catalog()
    frame.loc[0, "total_area_visited_km2"] = value
    with pytest.raises(SchemaError):
        WaveCatalogFileSchema.validate(frame)


@pytest.mark.parametrize(
    "shape",
    [
        None,
        Polygon(),
        LineString([(0, 0), (1, 1)]),
        Polygon([(0, 0), (1, 1), (0, 1), (1, 0)]),
        box(180, 0, 181, 1),
        box(0, 90, 1, 91),
    ],
)
def test_catalog_schema_requires_valid_geographic_lifetime_polygon(shape):
    """Check that catalog validation rejects invalid geographic polygon coverage.

    Args:
        shape: Candidate geometry selected by parametrization.

    Returns:
        None.

    Steps:
        1. Replace lifetime geometry with the parametrized unsupported shape.
        2. Check that catalog validation rejects invalid geographic polygon coverage.

    Examples:
        Run ``pytest tests/test_schemas.py -k catalog_schema_requires_valid_geographic_lifetime_polygon``.
    """
    frame = _catalog()
    frame.loc[0, "lifetime_geometry"] = shape
    with pytest.raises(SchemaError):
        WaveCatalogFileSchema.validate(frame)


def test_catalog_schema_preserves_lifetime_geometry_and_rejects_projected_crs():
    """Reproject the catalog and check that geographic schema validation rejects it.

    Returns:
        None.

    Steps:
        1. Validate the catalog and compare the lifetime polygon bytes with its input.
        2. Reproject the catalog and check that geographic schema validation rejects it.

    Examples:
        Run ``pytest tests/test_schemas.py -k catalog_schema_preserves_lifetime_geometry_and_rejects_projected_crs``.
    """
    frame = _catalog()
    validated = WaveCatalogFileSchema.validate(frame)
    assert validated.geometry.iloc[0].wkb == frame.geometry.iloc[0].wkb
    assert validated.crs == "EPSG:4326"
    frame = frame.set_crs("EPSG:6933", allow_override=True)
    with pytest.raises(SchemaError, match="geographic_polygons"):
        WaveCatalogFileSchema.validate(frame)


@pytest.mark.parametrize("end_reason", ["died", "merged"])
def test_catalog_schema_requires_end_date_to_equal_last_observation(end_reason):
    """Check that catalog validation rejects the date mismatch.

    Args:
        end_reason: Ending classification to validate.

    Returns:
        None.

    Steps:
        1. Assign the parametrized ending reason with inconsistent end and observation dates.
        2. Check that catalog validation rejects the date mismatch.

    Examples:
        Run ``pytest tests/test_schemas.py -k catalog_schema_requires_end_date_to_equal_last_observation``.
    """
    frame = _catalog()
    frame["active"] = False
    frame["end_date"] = frame["last_observed_date"]
    frame["end_reason"] = end_reason
    if end_reason == "merged":
        frame["merged_into_wave_id"] = OTHER_WAVE
    WaveCatalogFileSchema.validate(frame)

    frame["end_date"] = _dates(["2020-01-03"])
    with pytest.raises(SchemaError, match="catalog_history"):
        WaveCatalogFileSchema.validate(frame)


def test_lineage_schema_requires_two_distinct_waves_and_known_relationship():
    """Check that the schema rejects identical endpoints and unsupported relationships.

    Returns:
        None.

    Steps:
        1. Change lineage identifiers and relationship values in separate variants.
        2. Check that the schema rejects identical endpoints and unsupported relationships.

    Examples:
        Run ``pytest tests/test_schemas.py -k lineage_schema_requires_two_distinct_waves_and_known_relationship``.
    """
    frame = _lineage()
    LineageFileSchema.validate(frame)
    frame["target_wave_id"] = WAVE
    with pytest.raises(SchemaError, match="distinct_waves"):
        LineageFileSchema.validate(frame)
    frame["target_wave_id"] = OTHER_WAVE
    frame["relationship"] = "nearby"
    with pytest.raises(SchemaError, match="relationship"):
        LineageFileSchema.validate(frame)


def test_name_schema_preserves_decimal_group_spelling_and_source_order():
    """Validate the rows and compare group spelling and order with the inputs.

    Returns:
        None.

    Steps:
        1. Build name-library rows with decimal group values and deliberate source ordering.
        2. Validate the rows and compare group spelling and order with the inputs.

    Examples:
        Run ``pytest tests/test_schemas.py -k name_schema_preserves_decimal_group_spelling_and_source_order``.
    """
    frame = pd.DataFrame(
        {
            "geography": ["Europe", "Asia"],
            "gender": ["F", "M"],
            "name": ["Vera", "Abdul"],
            "group": ["1.0", "13"],
        }
    )
    pd.testing.assert_frame_equal(NameLibrarySchema.validate(frame), frame)
    with pytest.raises((SchemaError, SchemaErrors), match="order"):
        NameLibrarySchema.validate(frame[["name", "geography", "gender", "group"]])


@pytest.mark.parametrize("group", ["0", "14", "1.5", "NaN", "Infinity", "", "word"])
def test_name_schema_rejects_invalid_group(group):
    """Check that name-library validation rejects the group.

    Args:
        group: Candidate name-library group identifier.

    Returns:
        None.

    Steps:
        1. Replace the category group with the parametrized invalid value.
        2. Check that name-library validation rejects the group.

    Examples:
        Run ``pytest tests/test_schemas.py -k name_schema_rejects_invalid_group``.
    """
    frame = pd.DataFrame(
        {"geography": ["Europe"], "gender": ["F"], "name": ["Vera"], "group": [group]}
    )
    with pytest.raises(SchemaError, match="integral_groups"):
        NameLibrarySchema.validate(frame)


def test_name_schema_rejects_empty_library_and_blank_name():
    """Check that name-library validation rejects both cases.

    Returns:
        None.

    Steps:
        1. Create empty-library and blank-name variants.
        2. Check that name-library validation rejects both cases.

    Examples:
        Run ``pytest tests/test_schemas.py -k name_schema_rejects_empty_library_and_blank_name``.
    """
    frame = pd.DataFrame(
        {
            field: pd.Series([], dtype="str")
            for field in ("geography", "gender", "name", "group")
        }
    )
    with pytest.raises(SchemaError, match="nonempty_library"):
        NameLibrarySchema.validate(frame)
    frame = pd.DataFrame(
        {"geography": ["Europe"], "gender": ["F"], "name": ["  "], "group": ["1.0"]}
    )
    with pytest.raises(SchemaError, match="name"):
        NameLibrarySchema.validate(frame)
