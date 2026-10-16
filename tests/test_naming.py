"""Tests for deterministic editorial wave naming."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from reuters_temperature_storms import naming as naming_module
from reuters_temperature_storms.naming import (
    DEFAULT_NAMES_SEED,
    DEFAULT_NAMES_SEED_SHA256,
    NAME_AREA_THRESHOLD_KM2,
    NAME_LIBRARY_SHA256,
    NameLibrary,
    NameLibraryEntry,
    StormNamingError,
    load_name_library,
    name_at_ordinal,
    seed_sha256,
)

_GROUPS = tuple(range(1, 14))
_GEOGRAPHIES = (
    "Africa",
    "Asia",
    "Europe",
    "North America",
    "South America",
)
_GENDERS = ("F", "M")


def test_library_constructor_checks_schema_and_buckets_once(monkeypatch):
    """Keep the CSV schema and naming checks to one call each.

    Args:
        monkeypatch: Fixture counting the existing schema and model checks.

    Returns:
        None. Confirms one pass per check and a reproducible 5,000-name sequence.

    Examples:
        Run pytest tests/test_naming.py -k constructor_checks.

    Steps:
        1. Count CSV schema and model validation calls while loading the packaged library.
        2. Draw 5,000 names and compare their digest with the expected sequence hash.
        3. Check that each validation layer ran exactly once.
    """
    calls = {"schema": 0, "library": 0}
    schema_validate = NameLibrary.schema.validate
    library_validate = NameLibrary.validate

    def check_schema(cls, frame, *args, **kwargs):
        """Count the existing CSV schema check.

        Args:
            cls: Schema class receiving the call.
            frame: CSV rows passed to validation.
            *args: Existing positional validation options.
            **kwargs: Existing keyword validation options.

        Returns:
            The original validation result.

        Examples:
            Invoked by NameLibrary.read_csv during this test.

        Steps:
            1. Increment the CSV-schema validation counter.
            2. Delegate to the original schema validator with unchanged options.
        """
        calls["schema"] += 1
        return schema_validate(frame, *args, **kwargs)

    def check_library(self):
        """Count the model check covering entries and all naming buckets.

        Args:
            self: Constructed name library.

        Returns:
            None. Delegates to the original model check.

        Examples:
            Invoked once when the CSV constructor builds the library.

        Steps:
            1. Increment the model-validation counter.
            2. Delegate to the original library validation method.
        """
        calls["library"] += 1
        return library_validate(self)

    monkeypatch.setattr(NameLibrary.schema, "validate", classmethod(check_schema))
    monkeypatch.setattr(NameLibrary, "validate", check_library)
    library = NameLibrary.read_csv()
    assert calls == {"schema": 1, "library": 1}
    sequence = naming_module._NameSequence(library, DEFAULT_NAMES_SEED)
    names = [sequence.next_name() for _ in range(5000)]
    digest = hashlib.sha256(
        json.dumps(names, separators=(",", ":")).encode()
    ).hexdigest()
    assert digest == "43e1cc2970360af81f59509beabe35ffdf7207aad09d54a8a824b7e1a646b401"


def _small_library() -> NameLibrary:
    """Construct a compact name library spanning all category buckets.

    Returns:
        Synthetic category-complete name library.

    Steps:
        1. Enumerate every group, geography, and gender combination.
        2. Construct one deterministic name entry per bucket and assemble the complete library.

    Examples:
        Used by ``pytest tests/test_naming.py`` for fixture construction and output checks.
    """
    entries = tuple(
        NameLibraryEntry(
            geography=geography,
            gender=gender,
            name=f"{group}-{geography}-{gender}",
            group=group,
        )
        for group in _GROUPS
        for geography in _GEOGRAPHIES
        for gender in _GENDERS
    )
    return NameLibrary(entries, _GROUPS, _GEOGRAPHIES, _GENDERS)


def _write_library(path: Path, *, omit: tuple[int, str, str] | None = None) -> None:
    """Write the synthetic category-complete name library to disk.

    Args:
        path: Destination path for the synthetic name library.
        omit: Category bucket to exclude, or None to write every bucket.

    Returns:
        None.

    Steps:
        1. Enumerate every category bucket except the optional omitted bucket.
        2. Write the CSV header and rows to the requested path.

    Examples:
        Used by ``pytest tests/test_naming.py`` for fixture construction and output checks.
    """
    rows = ["geography,gender,name,group\n"]
    for group in _GROUPS:
        for geography in _GEOGRAPHIES:
            for gender in _GENDERS:
                if (group, geography, gender) != omit:
                    rows.append(
                        f"{geography},{gender},{group}-{geography}-{gender},{group}\n"
                    )
    path.write_text("".join(rows), encoding="utf-8", newline="")


@pytest.mark.unit
def test_packaged_library_matches_reviewed_artifact() -> None:
    """Compare their checksum and row content with the reviewed fixture expectations.

    Returns:
        None.

    Steps:
        1. Read the packaged name-library bytes.
        2. Compare their checksum and row content with the reviewed fixture expectations.

    Examples:
        Run ``pytest tests/test_naming.py -k packaged_library_matches_reviewed_artifact``.
    """
    path = (
        Path(__file__).parents[1]
        / "src"
        / "reuters_temperature_storms"
        / "data"
        / "names_library.csv"
    )
    raw = path.read_bytes()
    library = load_name_library()

    assert len(raw) == 52_127
    assert (
        hashlib.sha256(raw).hexdigest()
        == "9a08859fe923c5cefbf2eadd88d7074eec865a2ad9c428c5cc086930496991ee"
    )
    assert (
        NAME_LIBRARY_SHA256
        == "9a08859fe923c5cefbf2eadd88d7074eec865a2ad9c428c5cc086930496991ee"
    )
    assert raw.startswith(b"geography,gender,name,group\n")
    assert b"\r\n" not in raw
    assert len(library.entries) == 2_380
    assert library.groups == _GROUPS
    assert library.geographies == _GEOGRAPHIES
    assert library.genders == _GENDERS


@pytest.mark.unit
def test_reviewed_library_sequence_matches_checkpoints_across_cycles() -> None:
    """Compare the selected positions with the expected checkpoint names.

    Returns:
        None.

    Steps:
        1. Advance the deterministic name sequence across category cycles.
        2. Compare the selected positions with the expected checkpoint names.

    Examples:
        Run ``pytest tests/test_naming.py -k reviewed_library_sequence_matches_checkpoints_across_cycles``.
    """
    library = load_name_library()

    expected = {
        0: "Uriel",
        64: "Degu",
        129: "Tian",
        130: "Trigeñito",
        259: "Vijay",
        260: "Terry",
        999: "Laxman",
        4_999: "Ivone",
    }

    assert {
        ordinal: name_at_ordinal(ordinal, library=library) for ordinal in expected
    } == expected
    assert seed_sha256(DEFAULT_NAMES_SEED) == DEFAULT_NAMES_SEED_SHA256
    assert (
        DEFAULT_NAMES_SEED_SHA256
        == "0ea3c4caa98b5c9715a22435b1341ece86ea60ffa297c02ed5f11c2e55ded364"
    )
    assert NAME_AREA_THRESHOLD_KM2 == 5_100_720.0


@pytest.mark.unit
def test_ordinal_lookup_cycles_each_bucket_without_advancing() -> None:
    """Compare direct lookup with the sequential picker and check lookup does not advance its position.

    Returns:
        None.

    Steps:
        1. Look up names at selected ordinals spanning bucket repetitions.
        2. Compare direct lookup with the sequential picker and check lookup does not advance its position.

    Examples:
        Run ``pytest tests/test_naming.py -k ordinal_lookup_cycles_each_bucket_without_advancing``.
    """
    entries = _small_library().entries + (NameLibraryEntry("Africa", "F", "second", 1),)
    library = NameLibrary(entries, _GROUPS, _GEOGRAPHIES, _GENDERS)

    sequence = naming_module._NameSequence(library, DEFAULT_NAMES_SEED)
    ordinal = next(
        index
        for index in range(130)
        if sequence.name_at_ordinal(index) in {"1-Africa-F", "second"}
    )
    first = sequence.name_at_ordinal(ordinal)

    assert sequence.name_at_ordinal(ordinal + 130) != first
    assert sequence.name_at_ordinal(ordinal + 260) == first
    assert sequence.name_at_ordinal(ordinal) == first


@pytest.mark.unit
def test_directly_constructed_invalid_library_fails_clearly() -> None:
    """Check that each invalid library raises a clear naming error.

    Returns:
        None.

    Steps:
        1. Construct name libraries with invalid entries or category dimensions.
        2. Check that each invalid library raises a clear naming error.

    Examples:
        Run ``pytest tests/test_naming.py -k directly_constructed_invalid_library_fails_clearly``.
    """
    with pytest.raises(StormNamingError, match="groups do not match"):
        NameLibrary(
            entries=_small_library().entries,
            groups=(),
            geographies=_GEOGRAPHIES,
            genders=_GENDERS,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("name,gender,geography,group\nAlex,M,Europe,1\n", "column"),
        ("geography,gender,name,group\nEurope,X,Alex,1\n", "gender"),
        (
            "geography,gender,name,group\nEurope,M,Alex,1.5\n",
            "group",
        ),
        ("geography,gender,name,group\n ,M,Alex,1\n", "geography"),
        ("geography,gender,name,group\nEurope,M,Alex,1,extra\n", "CSV"),
        ("geography,gender,name,group\n", "CSV"),
        ("geography,gender,name,name\nEurope,M,Alex,1\n", "column"),
    ],
)
def test_malformed_library_fails_clearly(
    tmp_path: Path, contents: str, message: str
) -> None:
    """Check that reading the library raises the expected naming error.

    Args:
        tmp_path: Pytest-owned temporary directory for input and output files.
        contents: Malformed name-library CSV text.
        message: Expected validation-error text.

    Returns:
        None.

    Steps:
        1. Write the parametrized malformed CSV to the temporary directory.
        2. Check that reading the library raises the expected naming error.

    Examples:
        Run ``pytest tests/test_naming.py -k malformed_library_fails_clearly``.
    """
    path = tmp_path / "names.csv"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(StormNamingError, match=message):
        load_name_library(path)


@pytest.mark.unit
def test_invalid_utf8_library_fails_with_naming_error(tmp_path: Path) -> None:
    """Check that reading the file wraps the decoding failure as a naming error.

    Args:
        tmp_path: Pytest-owned temporary directory for input and output files.

    Returns:
        None.

    Steps:
        1. Write invalid UTF-8 bytes as the name-library file.
        2. Check that reading the file wraps the decoding failure as a naming error.

    Examples:
        Run ``pytest tests/test_naming.py -k invalid_utf8_library_fails_with_naming_error``.
    """
    path = tmp_path / "names.csv"
    path.write_bytes(b"geography,gender,name,group\nEurope,F,An\xffa,1\n")

    with pytest.raises(StormNamingError, match="cannot read UTF-8 CSV"):
        load_name_library(path)


@pytest.mark.unit
def test_missing_bucket_fails_clearly(tmp_path: Path) -> None:
    """Check that the library reader reports its missing bucket.

    Args:
        tmp_path: Pytest-owned temporary directory for input and output files.

    Returns:
        None.

    Steps:
        1. Write the synthetic library with one category bucket omitted.
        2. Check that the library reader reports its missing bucket.

    Examples:
        Run ``pytest tests/test_naming.py -k missing_bucket_fails_clearly``.
    """
    path = tmp_path / "names.csv"
    _write_library(path, omit=(1, "Africa", "F"))

    with pytest.raises(StormNamingError, match="missing 1 required"):
        load_name_library(path)


@pytest.mark.unit
def test_wrong_dimension_counts_fail_clearly(tmp_path: Path) -> None:
    """Check that the reader rejects the incomplete category dimensions.

    Args:
        tmp_path: Pytest-owned temporary directory for input and output files.

    Returns:
        None.

    Steps:
        1. Write a library with invalid category counts.
        2. Check that the reader rejects the incomplete category dimensions.

    Examples:
        Run ``pytest tests/test_naming.py -k wrong_dimension_counts_fail_clearly``.
    """
    path = tmp_path / "names.csv"
    path.write_text(
        "geography,gender,name,group\n"
        "Africa,F,a,1\n"
        "Africa,M,b,1\n"
        "Asia,F,c,1\n"
        "Asia,M,d,1\n"
        "Africa,F,e,2\n"
        "Africa,M,f,2\n"
        "Asia,F,g,2\n"
        "Asia,M,h,2\n",
        encoding="utf-8",
    )

    with pytest.raises(StormNamingError, match="expected group, geography, and gender"):
        load_name_library(path)


@pytest.mark.unit
def test_wrong_dimension_values_fail_clearly(tmp_path: Path) -> None:
    """Check that the reader rejects its category dimensions.

    Args:
        tmp_path: Pytest-owned temporary directory for input and output files.

    Returns:
        None.

    Steps:
        1. Write a library containing unsupported category values.
        2. Check that the reader rejects its category dimensions.

    Examples:
        Run ``pytest tests/test_naming.py -k wrong_dimension_values_fail_clearly``.
    """
    path = tmp_path / "names.csv"
    _write_library(path)
    contents = path.read_text(encoding="utf-8").replace("South America", "Oceania")
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(StormNamingError, match="reviewed Reuters values"):
        load_name_library(path)


@pytest.mark.unit
def test_stateful_sequence_matches_ordinal_lookup_across_complete_cycles() -> None:
    """The chronological sequence preserves every reviewed ordinal name.

    Args:
        None.

    Returns:
        None.

    Examples:
        Run ``pytest tests/test_naming.py -k stateful_sequence``.

    Steps:
        1. The chronological sequence preserves every reviewed ordinal name.
    """
    from reuters_temperature_storms.naming import _NameSequence

    library = load_name_library()
    sequence = _NameSequence(library, DEFAULT_NAMES_SEED)
    expected = _NameSequence(library, DEFAULT_NAMES_SEED)
    for ordinal in range(5_000):
        assert sequence.next_name() == expected.name_at_ordinal(ordinal)
    assert sequence.name_at_ordinal(0) == "Uriel"


@pytest.mark.unit
@pytest.mark.parametrize("ordinal", [-1, True, 1.5, "1"])
def test_invalid_ordinal_fails_clearly(ordinal: object) -> None:
    """Invalid positions cannot silently change the naming sequence.

    Args:
        ordinal: Invalid sequence position supplied by pytest.

    Returns:
        None.

    Examples:
        Run ``pytest tests/test_naming.py -k invalid_ordinal``.

    Steps:
        1. Invalid positions cannot silently change the naming sequence.
    """
    with pytest.raises(StormNamingError, match="nonnegative integer"):
        name_at_ordinal(ordinal)


@pytest.mark.unit
@pytest.mark.parametrize("seed", [True, 1.5, "30", -1])
def test_invalid_seed_fails_clearly(seed: object) -> None:
    """Invalid seeds are rejected before a different name order is returned.

    Args:
        seed: Invalid shuffle seed supplied by pytest.

    Returns:
        None.

    Examples:
        Run ``pytest tests/test_naming.py -k invalid_seed``.

    Steps:
        1. Invalid seeds are rejected before a different name order is returned.
    """
    with pytest.raises(StormNamingError, match="seed must be a nonnegative integer"):
        name_at_ordinal(0, seed=seed)


@pytest.mark.unit
def test_csv_reader_preserves_names_strings_and_source_order(tmp_path: Path) -> None:
    """CSV parsing preserves literal names and reviewed row order.

    Args:
        tmp_path: Directory for a complete synthetic library.

    Returns:
        None. Values commonly treated as missing remain exact names, and
        decimal integral group notation leaves bucket assignment unchanged.

    Examples:
        Run ``pytest tests/test_naming.py -k preserves_names_strings``.

    Steps:
        1. CSV parsing preserves literal names and reviewed row order.
        2. Check that values commonly treated as missing remain exact names, and decimal integral group notation leaves bucket assignment unchanged.
    """
    path = tmp_path / "names.csv"
    _write_library(path)
    contents = path.read_text(encoding="utf-8")
    contents = contents.replace("1-Africa-F,1\n", "NA,1.0\n")
    contents = contents.replace("1-Africa-M,1\n", " NULL ,1\n")
    path.write_text(contents, encoding="utf-8")

    library = load_name_library(path)

    assert library.entries[0] == NameLibraryEntry("Africa", "F", "NA", 1)
    assert library.entries[1] == NameLibraryEntry("Africa", "M", " NULL ", 1)
    assert library.entries[2].name == "1-Asia-F"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("ordinal", "expected"),
    [
        (2196, "Thabiso"),
        (2217, "Lijun"),
        (2235, "Vera"),
        (2236, "Yakubu"),
        (2300, "Stacy"),
        (2302, "Ankit"),
        (2303, "Conchi"),
        (2307, "Jia"),
        (2308, "Leszek"),
        (2323, "Mahesh"),
        (2480, "Raimundo"),
    ],
)
def test_rebuilt_target_storm_names(ordinal: int, expected: str) -> None:
    """Keep the eleven editorial target names at measured revised declarations.

    Args:
        ordinal: Zero-based declaration position in the deterministic naming sequence.
        expected: Name expected for the matching physical storm.

    Returns:
        None. Checks stateless lookup, chronological draws and a fresh sequence.

    Examples:
        Run ``pytest tests/test_naming.py -k rebuilt_target``.

    Steps:
        1. Advance the seeded picker to the parametrized declaration ordinal.
        2. Compare the next draw, stateless lookup, and a fresh picker lookup with the expected name.
    """
    library = load_name_library()
    sequence = naming_module._NameSequence(library, DEFAULT_NAMES_SEED)
    for _ in range(ordinal):
        sequence.next_name()
    assert sequence.next_name() == expected
    assert name_at_ordinal(ordinal, library=library) == expected
    restarted = naming_module._NameSequence(library, DEFAULT_NAMES_SEED)
    assert restarted.name_at_ordinal(ordinal) == expected


@pytest.mark.unit
@pytest.mark.parametrize("seed", [0, 30, DEFAULT_NAMES_SEED, 9])
def test_seeded_rotations_preserve_category_balance(seed: int) -> None:
    """Check category balance in every complete cycle for arbitrary seeds.

    Args:
        seed: Arbitrary seed selecting the rotations and bucket order.

    Returns:
        None. Verifies complete geography, gender and group coverage per cycle.

    Examples:
        Run ``pytest tests/test_naming.py -k category_balance``.

    Steps:
        1. Generate four complete category cycles using the parametrized seed.
        2. Check each 130-entry cycle contains every category combination and the expected group, geography, and gender counts.
    """
    sequence = naming_module._NameSequence(_small_library(), seed)
    categories = [sequence.next_name().split("-") for _ in range(520)]
    for start in range(0, 520, 130):
        cycle = categories[start : start + 130]
        assert len({tuple(row) for row in cycle}) == 130
        assert all(
            sum(row[0] == str(group) for row in cycle) == 10 for group in _GROUPS
        )
        assert all(
            sum(row[1] == geography for row in cycle) == 26
            for geography in _GEOGRAPHIES
        )
        assert all(sum(row[2] == gender for row in cycle) == 65 for gender in _GENDERS)
