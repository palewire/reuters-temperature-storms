"""Verify release checks reject incomplete and changed assets."""

import io
import tarfile
import zipfile
from pathlib import Path

import click
import pytest

from scripts.check_release_artifacts import check_artifacts


@pytest.fixture
def artifacts(tmp_path):
    """Create a wheel and sdist with the approved names and typing marker.

    Args:
        tmp_path: Pytest-owned temporary directory.

    Returns:
        Directory and a callable that writes the requested archive contents.

    Examples:
        Tests request ``artifacts`` through pytest fixture injection.

    Steps:
        1. Read the packaged name-library CSV and prepare wheel and source-distribution member mappings.
        2. Write both synthetic archives and return their directory, writer, and original member mappings.
    """
    csv_path = "reuters_temperature_storms/data/names_library.csv"
    csv_bytes = (Path(__file__).parents[1] / "src" / csv_path).read_bytes()
    wheel_members = {csv_path: csv_bytes, "reuters_temperature_storms/py.typed": b""}
    sdist_members = {
        f"release/src/{csv_path}": csv_bytes,
        "release/src/reuters_temperature_storms/py.typed": b"",
    }

    def write(*, wheel=None, sdist=None):
        """Write supplied archive members into the fixture directory.

        Args:
            wheel: Optional mapping overriding wheel members.
            sdist: Optional mapping overriding source-distribution members.

        Returns:
            None.

        Examples:
            write(wheel={"asset": b"changed"})

        Steps:
            1. Write the selected wheel members into a ZIP archive.
            2. Write the selected source-distribution members into a compressed tar archive.
        """
        with zipfile.ZipFile(tmp_path / "release.whl", "w") as archive:
            for name, contents in (
                wheel if wheel is not None else wheel_members
            ).items():
                archive.writestr(name, contents)
        with tarfile.open(tmp_path / "release.tar.gz", "w:gz") as archive:
            for name, contents in (
                sdist if sdist is not None else sdist_members
            ).items():
                member = tarfile.TarInfo(name)
                member.size = len(contents)
                archive.addfile(member, io.BytesIO(contents))

    write()
    return tmp_path, write, wheel_members, sdist_members


def test_valid_assets_pass(artifacts):
    """Validate the synthetic wheel and source distribution containing the required data and typing marker.

    Args:
        artifacts: Fixture containing the archive directory, writer, and original members.

    Returns:
        None.

    Steps:
        1. Validate the synthetic wheel and source distribution containing the required data and typing marker.

    Examples:
        Run ``pytest tests/test_release_artifacts.py -k valid_assets_pass``.
    """
    check_artifacts(artifacts[0])


@pytest.mark.parametrize("artifact", ["release.whl", "release.tar.gz"])
def test_missing_artifact_rejected(artifacts, artifact):
    """Check that release validation reports the missing wheel or source distribution.

    Args:
        artifacts: Fixture containing the archive directory, writer, and original members.
        artifact: Archive filename to remove before validation.

    Returns:
        None.

    Steps:
        1. Remove one of the two required archives.
        2. Check that release validation reports the missing wheel or source distribution.

    Examples:
        Run ``pytest tests/test_release_artifacts.py -k missing_artifact_rejected``.
    """
    directory = artifacts[0]
    (directory / artifact).unlink()
    with pytest.raises(click.ClickException, match="exactly one"):
        check_artifacts(directory)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_changed_names_rejected(artifacts, kind):
    """Check that release validation reports the data mismatch.

    Args:
        artifacts: Fixture containing the archive directory, writer, and original members.
        kind: Archive or malformed-input variant selected by parametrization.

    Returns:
        None.

    Steps:
        1. Rewrite the selected archive with altered name-library bytes.
        2. Check that release validation reports the data mismatch.

    Examples:
        Run ``pytest tests/test_release_artifacts.py -k changed_names_rejected``.
    """
    directory, write, wheel, sdist = artifacts
    changed = dict(wheel if kind == "wheel" else sdist)
    csv_member = next(name for name in changed if name.endswith(".csv"))
    changed[csv_member] = changed[csv_member].replace(b"Abdo", b"Changed", 1)
    write(**{kind: changed})
    with pytest.raises(click.ClickException, match="CSV differs"):
        check_artifacts(directory)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_missing_typing_marker_rejected(artifacts, kind):
    """Check that release validation reports the missing marker.

    Args:
        artifacts: Fixture containing the archive directory, writer, and original members.
        kind: Archive or malformed-input variant selected by parametrization.

    Returns:
        None.

    Steps:
        1. Rewrite the selected archive without its typing marker.
        2. Check that release validation reports the missing marker.

    Examples:
        Run ``pytest tests/test_release_artifacts.py -k missing_typing_marker_rejected``.
    """
    directory, write, wheel, sdist = artifacts
    changed = {
        name: data
        for name, data in (wheel if kind == "wheel" else sdist).items()
        if not name.endswith("py.typed")
    }
    write(**{kind: changed})
    with pytest.raises(click.ClickException, match="missing the typing marker"):
        check_artifacts(directory)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_missing_names_rejected(artifacts, kind):
    """Check that release validation reports the missing data file.

    Args:
        artifacts: Fixture containing the archive directory, writer, and original members.
        kind: Archive or malformed-input variant selected by parametrization.

    Returns:
        None.

    Steps:
        1. Rewrite the selected archive without its name-library CSV.
        2. Check that release validation reports the missing data file.

    Examples:
        Run ``pytest tests/test_release_artifacts.py -k missing_names_rejected``.
    """
    directory, write, wheel, sdist = artifacts
    changed = {
        name: data
        for name, data in (wheel if kind == "wheel" else sdist).items()
        if not name.endswith(".csv")
    }
    write(**{kind: changed})
    with pytest.raises(click.ClickException, match="missing the Reuters names CSV"):
        check_artifacts(directory)
