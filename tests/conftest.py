"""Explicit naming setup for tests that invent historical physical storms."""

from datetime import date

import pytest

from reuters_temperature_storms.published_names import PublishedStormNames


@pytest.fixture
def synthetic_name_history(monkeypatch):
    """Use an empty earlier issued history for invented storm fixtures only.

    Actual package and issued-name tests do not request this fixture. Invented
    historical shapes cannot claim names from the real Reuters story table.

    Args:
        monkeypatch: Pytest helper that restores the loader after each test.

    Returns:
        None. Only requesting tests replace the packaged issued-name loader.

    Examples:
        ``pytestmark = pytest.mark.usefixtures("synthetic_name_history")``
        opts an invented-history test module into its isolated naming setup.
    """
    monkeypatch.setattr(
        "reuters_temperature_storms.published_names.load_published_storm_names",
        lambda: PublishedStormNames((), through_date=date(1959, 12, 31)),
    )
