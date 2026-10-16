"""Check the Reuters names and typing marker in built release distributions."""

import csv
import io
import tarfile
import zipfile
from pathlib import Path

import click


def check_artifacts(directory: Path) -> None:
    """Verify package assets match the source in one wheel and one sdist.

    Steps:
        1. Read the bundled source name library and require one wheel and one source distribution.
        2. Compare the packaged CSV bytes with source and require the typing marker in both archives.
        3. Report success only when every required package asset matches.

    Args:
        directory: Directory containing the built wheel and source distribution.

    Returns:
        None. Raises ClickException when assets are missing or differ.

    Examples:
        >>> check_artifacts(Path("dist"))  # doctest: +SKIP
    """
    source = Path(__file__).resolve().parents[1] / "src/reuters_temperature_storms"
    names_path = source / "data/names_library.csv"
    expected = names_path.read_bytes()
    rows = list(csv.reader(io.StringIO(expected.decode("utf-8"))))
    if len(rows) != 2381:
        raise click.ClickException("Expected a CSV header and 2,380 Reuters names.")
    wheels = list(directory.glob("*.whl"))
    sdists = list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise click.ClickException(
            "Expected exactly one wheel and one source distribution."
        )
    csv_member = f"reuters_temperature_storms/data/{names_path.name}"
    with zipfile.ZipFile(wheels[0]) as wheel:
        if csv_member not in wheel.namelist():
            raise click.ClickException("Wheel is missing the Reuters names CSV.")
        if wheel.read(csv_member) != expected:
            raise click.ClickException("Wheel names CSV differs from the source.")
        if "reuters_temperature_storms/py.typed" not in wheel.namelist():
            raise click.ClickException("Wheel is missing the typing marker.")
    with tarfile.open(sdists[0]) as sdist:
        prefixes = {member.name.split("/", 1)[0] for member in sdist.getmembers()}
        if len(prefixes) != 1:
            raise click.ClickException(
                "Expected one source-distribution root directory."
            )
        prefix = prefixes.pop()
        if f"{prefix}/src/{csv_member}" not in sdist.getnames():
            raise click.ClickException(
                "Source distribution is missing the Reuters names CSV."
            )
        csv_file = sdist.extractfile(f"{prefix}/src/{csv_member}")
        if csv_file is None or csv_file.read() != expected:
            raise click.ClickException(
                "Source-distribution names CSV differs from the source."
            )
        if f"{prefix}/src/reuters_temperature_storms/py.typed" not in sdist.getnames():
            raise click.ClickException(
                "Source distribution is missing the typing marker."
            )
    click.echo("Wheel and sdist preserve the exact Reuters CSV and typing marker.")


@click.command()
@click.argument("directory", type=click.Path(path_type=Path, exists=True))
def main(directory: Path) -> None:
    """Verify a built distribution directory.

    Steps:
        1. Accept an existing distribution directory through Click.
        2. Run the wheel and source-distribution asset checks and report the result.

    Args:
        directory: Directory containing the wheel and source distribution.

    Returns:
        None. Prints success or exits with an error through Click.

    Examples:
        $ uv run python scripts/check_release_artifacts.py dist
    """
    check_artifacts(directory)


if __name__ == "__main__":
    main()
