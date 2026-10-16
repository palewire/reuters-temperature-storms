"""Deterministic editorial names that turn qualifying waves into storms."""

from __future__ import annotations

import hashlib
import math
import warnings
from dataclasses import dataclass
from decimal import Decimal
from importlib.resources import files
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Final,
    Literal,
    NamedTuple,
    Self,
    TypeVar,
    cast,
)

import pandas as pd
from pandera.errors import SchemaError, SchemaErrors

from .schemas import NameLibrarySchema

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable
    from os import PathLike

NAME_AREA_THRESHOLD_KM2: Final = 5_100_720.0
DEFAULT_NAMES_SEED: Final = int(
    "5147a92d00d79b69696347aeb6d192f6e49f0bdd57dc644f1bedf9e1ed083dd035338d45f83d"
    "29436785af88fbf8dc94d3c856a2972f8369a0cbcb6ae7dab84e3a1bf20118adc1228fa03ba0"
    "b6372a21815dbcab8b1473a8a8b4226ebdee680261b267bdabfb685979e612d7da93314b3334"
    "349f5fc749038805fc15825a969811bbedb0eafbd76152624c347975b915472549e787d7fa59"
    "d09cb61788282983e40deef8e8f14c740cde0451e4118fdbd8d2844ff5dd555df2b527fe3b03"
    "6283f536a401218e084eb71ac917c17da7e9e3ddd2fce48056991f2913e749cf112ac2a4ab42"
    "3968d4a8457e6d335b0540b3fe7ef4280cf0d2ad6d2890b57bc188cd57a564e7935607ddcede"
    "8710d55f7209cee9db2c2d32acda4584818ee6b98d69c4878cf6295b55a8459c4e4008336179"
    "14e00edce39e951598f0a12cb51041567914f0863e9378ec6105df4b34eada5757a544222263"
    "e4e6f968b7a1ada6c0acc8213702774e8e320115417be54a37a61416400e561848b434400456"
    "4442692ad7594f65461ca37efb5e9c02238b39b9d14979324fc9f42e2b43d621a6d055f57ea6"
    "d2434c107053070069aebb6db8c4462e67292dd6762f9f4c1975841b675162ad714450b805fb"
    "080b2beea7f2147b2191383db28b4f27ac240f58e73331481000828eb3e9d0a41ceca5d5b204"
    "e4fc52a66534a12400120f7ecf63728b5bfae003f69c858adbbc943755dd3d213ee5a134b813"
    "d7ad5acb99ee1777cf2f7f30519026c044c37e30d9002ead6c37fee1687da7ab566f021a2172"
    "fc95b16ab6673f5db4b53e9561b08c9602679b05b2d2d0ed66c1357856c691c7b88a7071ca9d"
    "c97f3ace781f6552b98336cbc25d84fed8679a24ebb30a418f8de026064deb6067ba711ed2c5"
    "ca05dfd6b636505015902eb5e0f55e7ac7eb272e96aee0dcde96e39557604c88bf56ce2ca10f"
    "b05f3e8df949757e64cf08d825d099c3e97200a67e5ccb10959de355f1c493162c5f727c933f"
    "17a3807612d137c96b1bfa94439e9be2f0cbf227ac3baf516570d18d3f945dfd96101ea8b2ea"
    "7f7a79c6ddef3be93ffecc0d6f80020dc22e11ba1e75bc772222c6ab98977c414aee4980c218"
    "fa9251ece18089168392f6058ae36bbe3c6a37c83fa7b4a3c1f1a6754f5947192942a7d61ff5"
    "a9913403bc511445ce5ab896d8605f42942af28cea080b84d6e3b3f09039f47b9f14b505b1cf"
    "43b233bd3fa24f86144ce4fb1905dcd86c3e4a9078ec12da398813f2a47bd4db3bdb7fafb920"
    "b3eeb2e646ceadb48b7d441717d864b318689c47316cedfe7a9cdb20fb0e77ea3129ec773323"
    "5d3a6e4dcb549e06e666e9f0cc37c22d2818bd4e9d8c46bff1a9ff752a9dc31de9845e039286"
    "a6d33219b153458d092b2cf4184846648096bc9e7e11d0b80ca398",
    16,
)
DEFAULT_NAMES_SEED_SHA256: Final = (
    "0ea3c4caa98b5c9715a22435b1341ece86ea60ffa297c02ed5f11c2e55ded364"
)
NAME_LIBRARY_COLUMNS: Final = ("geography", "gender", "name", "group")
NAME_LIBRARY_SHA256: Final = "9a08859fe923c5cefbf2eadd88d7074eec865a2ad9c428c5cc086930496991ee"  # pragma: allowlist secret
_REQUIRED_DIMENSIONS: Final = (13, 5, 2)
_REQUIRED_GROUPS: Final = tuple(range(1, 14))
_REQUIRED_GEOGRAPHIES: Final = (
    "Africa",
    "Asia",
    "Europe",
    "North America",
    "South America",
)
_REQUIRED_GENDERS: Final = ("F", "M")

Gender = Literal["F", "M"]
BucketKey = tuple[int, str, Gender]
_PermutationItem = TypeVar("_PermutationItem")


class _NameRow(NamedTuple):
    """Typed attributes supplied by the validated names DataFrame.

    Steps:
        1. Retain the supplied fields for this record.
        2. Expose the values to the corresponding processing and validation methods.

    Args:
        geography: Name-library geography category.
        gender: Name-library gender category.
        name: Base given name from the library, without a year suffix.
        group: Name-library selection group.

    Examples:
        row = _NameRow("region", "F", "Example", "1")
    """

    geography: str
    gender: str
    name: str
    group: str


class StormNamingError(ValueError):
    """Raised when a name library, seed or sequence position is invalid.

    Steps:
        1. Describe a rejected input, configuration or stage result.
        2. Propagate the message through callers without changing data.

    Args:
        *args: Exception message and any additional exception arguments.

    Examples:
        raise StormNamingError("invalid input")
    """


@dataclass(frozen=True, slots=True)
class NameLibraryEntry:
    """One editorial name and its selection bucket.

    Steps:
        1. Retain the supplied fields for this record.
        2. Expose the values to the corresponding processing and validation methods.

    Args:
        geography: Name-library geography category.
        gender: Name-library gender category.
        name: Base given name from the library, without a year suffix.
        group: Name-library selection group.

    Examples:
        entry = NameLibraryEntry("region", "F", "Example", 1)
    """

    geography: str
    gender: Gender
    name: str
    group: int


@dataclass(frozen=True, slots=True)
class NameLibrary:
    """Editorial names with CSV loading and selection validation.

    Steps:
        1. Keep names in their source order along with their group, geography and gender selections.
        2. Check that every selection combination has names before creating a sequence.

    Args:
        entries: Names in source order with their selection buckets.
        groups: Sorted group numbers represented by the entries.
        geographies: Sorted geography values represented by the entries.
        genders: Sorted gender values represented by the entries.

    Examples:
        library = NameLibrary.read_csv()
    """

    entries: tuple[NameLibraryEntry, ...]
    groups: tuple[int, ...]
    geographies: tuple[str, ...]
    genders: tuple[Gender, ...]
    schema: ClassVar[type[NameLibrarySchema]] = NameLibrarySchema

    def __post_init__(self) -> None:
        """Reject directly constructed libraries that bypass the CSV loader.

        Steps:
            1. Run the same entry and selection checks used after CSV loading.
            2. Keep the supplied values unchanged when all checks pass.

        Args:
            None.

        Returns:
            None.

        Raises:
            StormNamingError: If entries or declared dimensions are invalid.

        Examples:
            ``NameLibrary(entries, groups, geographies, genders)`` validates
            itself before it can be used by a naming sequence.
        """
        self.validate()

    @classmethod
    def read_csv(cls, path: str | PathLike[str] | None = None) -> Self:
        """Read an editorial CSV and apply its existing schema and bucket checks.

        Steps:
            1. Select the packaged name library or the caller's CSV file.
            2. Read UTF-8 text with all values kept as text and reject parsing warnings.
            3. Check the CSV fields, then create typed name entries in source order.
            4. Collect the sorted groups, geographies and genders and check the complete library once.

        Args:
            path: CSV path, or None for the bundled library.

        Returns:
            Immutable names in source order with validated selection dimensions.

        Raises:
            StormNamingError: If the CSV or its naming buckets are invalid.

        Examples:
            library = NameLibrary.read_csv()
        """
        source: Traversable
        if path is None:
            source = files("reuters_temperature_storms").joinpath(
                "data", "names_library.csv"
            )
        else:
            source = Path(path)
        try:
            with (
                source.open("r", encoding="utf-8") as stream,
                warnings.catch_warnings(),
            ):
                warnings.simplefilter("error", pd.errors.ParserWarning)
                frame = pd.read_csv(
                    stream, dtype=str, keep_default_na=False, index_col=False
                )
        except (
            OSError,
            UnicodeError,
            pd.errors.ParserError,
            pd.errors.ParserWarning,
            pd.errors.EmptyDataError,
        ) as error:
            msg = f"{source}: cannot read UTF-8 CSV"
            raise StormNamingError(msg) from error
        try:
            cls.schema.validate(frame)
        except (SchemaError, SchemaErrors) as error:
            msg = f"{source}: invalid name library CSV: {error}"
            raise StormNamingError(msg) from error
        entries = []
        for values in frame.itertuples(index=False):
            row = cast("_NameRow", values)
            entries.append(
                NameLibraryEntry(
                    geography=row.geography,
                    gender=cast("Gender", row.gender),
                    name=row.name,
                    group=int(Decimal(row.group)),
                )
            )
        entries = tuple(entries)
        try:
            return cls(
                entries=entries,
                groups=tuple(sorted({entry.group for entry in entries})),
                geographies=tuple(sorted({entry.geography for entry in entries})),
                genders=tuple(sorted({entry.gender for entry in entries})),
            )
        except StormNamingError as error:
            raise StormNamingError(f"{source}: {error}") from error

    def validate(self) -> None:
        """Check directly constructed entries and their complete naming cycle.

        Steps:
            1. Require a nonempty tuple of entries with valid names, geography, gender and positive group numbers.
            2. Check that the declared selection values match those present in the entries.
            3. Require at least one name in every group/geography/gender combination.
            4. Require the supported group numbers, geography labels and genders.
            5. Check that advancing all three selections together visits every combination before repeating.

        Args:
            None.

        Returns:
            None. Checks each declared bucket once without changing names.

        Raises:
            StormNamingError: If entries, dimensions or buckets are invalid.

        Examples:
            library.validate()
        """
        if not isinstance(self.entries, tuple) or not self.entries:
            msg = "entries must be a nonempty tuple"
            raise StormNamingError(msg)
        for index, entry in enumerate(self.entries):
            if not isinstance(entry, NameLibraryEntry):
                msg = f"entry {index} must be a NameLibraryEntry"
                raise StormNamingError(msg)
            if not isinstance(entry.geography, str) or not entry.geography.strip():
                msg = f"entry {index} geography must be a nonblank string"
                raise StormNamingError(msg)
            if not isinstance(entry.gender, str) or entry.gender not in {"F", "M"}:
                msg = f"entry {index} gender must be M or F"
                raise StormNamingError(msg)
            if not isinstance(entry.name, str) or not entry.name.strip():
                msg = f"entry {index} name must be a nonblank string"
                raise StormNamingError(msg)
            if (
                isinstance(entry.group, bool)
                or not isinstance(entry.group, int)
                or entry.group <= 0
            ):
                msg = f"entry {index} group must be a positive integer"
                raise StormNamingError(msg)
        expected_groups = tuple(sorted({entry.group for entry in self.entries}))
        expected_geographies = tuple(
            sorted({entry.geography for entry in self.entries})
        )
        expected_genders = tuple(sorted({entry.gender for entry in self.entries}))
        if self.groups != expected_groups:
            msg = "groups do not match entry values"
            raise StormNamingError(msg)
        if self.geographies != expected_geographies:
            msg = "geographies do not match entry values"
            raise StormNamingError(msg)
        if self.genders != expected_genders:
            msg = "genders do not match entry values"
            raise StormNamingError(msg)
        entries = self.entries
        groups, geographies, genders = self.groups, self.geographies, self.genders
        missing_genders = sorted({"F", "M"} - set(genders))
        if missing_genders:
            msg = f"no names available for genders {missing_genders}"
            raise StormNamingError(msg)
        actual = {(entry.group, entry.geography, entry.gender) for entry in entries}
        expected = {
            (group, geography, gender)
            for group in groups
            for geography in geographies
            for gender in genders
        }
        missing = sorted(expected - actual)
        if missing:
            msg = (
                f"missing {len(missing)} required "
                f"(group, geography, gender) buckets; for example {missing[:3]}"
            )
            raise StormNamingError(msg)
        dimensions = (len(groups), len(geographies), len(genders))
        if dimensions != _REQUIRED_DIMENSIONS:
            msg = (
                f"expected group, geography, and gender counts "
                f"{_REQUIRED_DIMENSIONS}, got {dimensions}"
            )
            raise StormNamingError(msg)
        if (
            groups != _REQUIRED_GROUPS
            or geographies != _REQUIRED_GEOGRAPHIES
            or genders != _REQUIRED_GENDERS
        ):
            msg = "library dimensions do not match the reviewed Reuters values"
            raise StormNamingError(msg)
        cycle_length = math.lcm(*dimensions)
        bucket_count = math.prod(dimensions)
        if cycle_length != bucket_count:
            msg = (
                f"dimension counts {dimensions} produce a "
                f"{cycle_length}-step shared cycle, not all {bucket_count} buckets"
            )
            raise StormNamingError(msg)


@dataclass(slots=True)
class _PermutationDecoder:
    """Consume permutation ranks from a nonnegative integer seed.

    Steps:
        1. Retain the supplied fields for this record.
        2. Expose the values to the corresponding processing and validation methods.

    Args:
        remaining: Unused nonnegative permutation seed.

    Examples:
        decoder = _PermutationDecoder(3)
    """

    remaining: int

    def take(self, items: list[_PermutationItem]) -> list[_PermutationItem]:
        """Decode the next permutation in canonical source order.

        Steps:
            1. Separate the next factorial permutation rank from the remaining seed.
            2. Select each item from canonical source order using that rank.
            3. Retain the unused seed portion for the next permutation.

        Args:
            items: Entries in their canonical order.

        Returns:
            All entries reordered by the next factorial rank in the seed.

        Examples:
            ``_PermutationDecoder(3).take(["a", "b", "c"])`` returns an ordering.
        """
        self.remaining, rank = divmod(self.remaining, math.factorial(len(items)))
        available = items.copy()
        result: list[_PermutationItem] = []
        for size in range(len(items), 0, -1):
            position, rank = divmod(rank, math.factorial(size - 1))
            result.append(available.pop(position))
        return result


def seed_sha256(seed: int) -> str:
    """Hash the seed's minimal unsigned byte encoding, most significant first.

    Steps:
        1. Validate that the seed is a nonnegative integer.
        2. Encode its minimal unsigned bytes and hash them with SHA-256.

    Args:
        seed: Nonnegative integer encoding the naming permutations.

    Returns:
        Lowercase SHA-256 hexadecimal digest.

    Raises:
        StormNamingError: If the seed is not a nonnegative integer.

    Examples:
        ``seed_sha256(DEFAULT_NAMES_SEED)`` returns its deterministic digest.
    """
    _validate_seed(seed)
    data = seed.to_bytes(max(1, (seed.bit_length() + 7) // 8), "big")
    return hashlib.sha256(data).hexdigest()


class _NameSequence:
    """Stateful deterministic traversal of one validated name library.

    Steps:
        1. Build category and name permutations from the supplied seed.
        2. Advance through the deterministic sequence as names are requested.

    Args:
        library: Validated name library.
        seed: Nonnegative integer encoding category and name permutations.

    Examples:
        sequence = _NameSequence(library, DEFAULT_NAMES_SEED)
    """

    def __init__(self, library: NameLibrary, seed: int) -> None:
        """Build one independently advancing name sequence.

        Steps:
            1. Place names into their group/geography/gender combinations in source order.
            2. Decode one permutation of all category combinations from the seed.
            3. Decode each source-order name pool in sorted combination order.
            4. Start at sequence position zero without changing the name library.

        Args:
            library: Validated names and bucket dimensions.
            seed: Nonnegative integer encoding category and name permutations.

        Returns:
            None.

        Examples:
            ``_NameSequence(library, DEFAULT_NAMES_SEED)`` starts the naming sequence.
        """
        _validate_seed(seed)
        unshuffled: dict[BucketKey, list[str]] = {
            (group, geography, gender): []
            for group in library.groups
            for geography in library.geographies
            for gender in library.genders
        }
        for entry in library.entries:
            unshuffled[(entry.group, entry.geography, entry.gender)].append(entry.name)
        decoder = _PermutationDecoder(seed)
        keys = sorted(unshuffled)
        self._category_order = decoder.take(keys)
        self._buckets = {key: decoder.take(unshuffled[key]) for key in keys}
        self._cycle_length = len(self._category_order)
        self._ordinal = 0

    def name_at_ordinal(self, ordinal: int) -> str:
        """Return one deterministic name without advancing the sequence.

        Steps:
            1. Require a nonnegative whole-number sequence position.
            2. Use that position to choose a group, geography and gender.
            3. Choose a name within that combination using the number of completed selection cycles.
            4. Return the name without advancing the saved sequence position.

        Args:
            ordinal: Zero-based position in the complete naming sequence.

        Returns:
            The given name assigned at that position.

        Raises:
            StormNamingError: If ``ordinal`` is not a nonnegative integer.

        Examples:
            ``sequence.name_at_ordinal(0)`` returns the first name.
        """
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            msg = "name ordinal must be a nonnegative integer"
            raise StormNamingError(msg)
        key = self._category_order[ordinal % self._cycle_length]
        bucket = self._buckets[key]
        return bucket[(ordinal // self._cycle_length) % len(bucket)]

    def next_name(self) -> str:
        """Return the next deterministic name and advance the sequence.

        Steps:
            1. Look up the name at the current saved sequence position.
            2. Advance that position by one and return the selected name.

        Args:
            None.

        Returns:
            The next given name without a year suffix.

        Examples:
            ``sequence.next_name()`` returns one name from the current bucket.
        """
        name = self.name_at_ordinal(self._ordinal)
        self._ordinal += 1
        return name


def load_name_library(path: str | PathLike[str] | None = None) -> NameLibrary:
    """Load and validate a UTF-8 editorial name library.

    Steps:
        1. Read the requested CSV or bundled name library.
        2. Return its validated entries and category dimensions.

    Args:
        path: Optional CSV path. ``None`` loads the bundled library.

    Returns:
        The validated immutable library.

    Raises:
        StormNamingError: If the CSV schema, values, bucket coverage, or shared
            bucket cycle is invalid.

    Examples:
        >>> library = load_name_library()
        >>> len(library.entries)
        2380
    """
    return NameLibrary.read_csv(path)


def name_at_ordinal(
    ordinal: int,
    *,
    library: NameLibrary | None = None,
    seed: int = DEFAULT_NAMES_SEED,
) -> str:
    """Return the deterministic given name at one zero-based position.

    Steps:
        1. Check the shuffle seed and load the default library when none is supplied.
        2. Create a naming sequence and look up the requested position without advancing it.

    Args:
        ordinal: Zero-based position in one branch's naming sequence.
        library: Optional validated library. The packaged library is the default.
        seed: Nonnegative integer encoding category and name permutations.

    Returns:
        The given name at ``ordinal`` without a year suffix.

    Raises:
        StormNamingError: If the seed or ordinal is invalid.

    Examples:
        ``name_at_ordinal(0)`` returns the first name in the sequence.
    """
    _validate_seed(seed)
    selected_library = load_name_library() if library is None else library
    return _NameSequence(selected_library, seed).name_at_ordinal(ordinal)


def _validate_seed(seed: int) -> None:
    """Require a nonnegative integer permutation seed.

    Steps:
        1. Reject booleans, non-integers and negative values.
        2. Accept nonnegative integer seeds without changing them.

    Args:
        seed: Seed supplied to the naming sequence.

    Returns:
        None.

    Raises:
        StormNamingError: If the seed is not a nonnegative integer.

    Examples:
        ``_validate_seed(DEFAULT_NAMES_SEED)`` accepts the default seed.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        msg = "name seed must be a nonnegative integer"
        raise StormNamingError(msg)
