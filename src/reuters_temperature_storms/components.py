"""Daily footprint models and shared scientific area tolerances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from shapely import Geometry

_ROLE_AREA_RELATIVE_TOLERANCE = 1e-6
_ROLE_AREA_ABSOLUTE_TOLERANCE_KM2 = 1e-6


@dataclass(frozen=True, slots=True)
class DailyWaveComponentGeometry:
    """One daily footprint and its exact supplied role geometry products.

    ``geometry`` is today's qualifying mature core, ``grace_geometry`` is the
    mature active grace-only portion, and ``full_geometry`` includes those parts
    plus morphology-added gap-filled cells. Areas use EPSG:6933 and km².

    Steps:
        1. Retain the supplied fields for this record.
        2. Expose the values to the corresponding processing and validation methods.

    Args:
        time: Calendar date as a NumPy datetime value.
        label: Numeric label for the daily component.
        geometry: Qualified mature core geometry, or None when absent.
        grace_geometry: Active mature geometry currently in its grace period, or None.
        full_geometry: Qualified, grace and gap-filled geometry combined.
        area_qualified: Qualified-core area in square kilometres.
        area_grace: Grace-only area in square kilometres.
        area_gap_filled: Gap-filled area in square kilometres.
        qualified_fraction: Qualified area divided by full area.

    Examples:
        component = DailyWaveComponentGeometry(day, 1, core, grace, full, 10.0, 2.0, 1.0, 10.0 / 13.0)
    """

    time: np.datetime64
    label: int
    geometry: Geometry | None
    grace_geometry: Geometry | None
    full_geometry: Geometry
    area_qualified: float
    area_grace: float
    area_gap_filled: float
    qualified_fraction: float


@dataclass(frozen=True, slots=True)
class ComponentMembership:
    """One supplied daily footprint without reconstructed cell membership.

    Steps:
        1. Retain the supplied fields for this record.
        2. Expose the values to the corresponding processing and validation methods.

    Args:
        component_id: Identifier of the supplied daily component.
        geometry: Daily geometry and role-area record for the component.
        order_digest: Full historical footprint digest used to order daily
            tracking decisions. It is separate from the permanent cell ID.
        input_id: Original file-row identifier, when the component came from a
            prepared input file. It does not participate in tracking decisions.

    Examples:
        component = ComponentMembership("daily-component", geometry, order_digest)
    """

    #: Daily identity derived from the calendar date and first occupied cell.
    component_id: str
    #: Finished role shapes and physical measurements for this component.
    geometry: DailyWaveComponentGeometry
    #: Daily footprint order, computed once from its complete native cell roles.
    order_digest: str
    #: Original prepared-file label retained for associated cell joins.
    input_id: str | None = None


def _is_order_digest(value: object) -> bool:
    """Check the saved full footprint-order digest without rebuilding shapes.

    Args:
        value: Candidate ordering field from a component or saved tracker state.

    Returns:
        True for exactly 32 lowercase hexadecimal characters, otherwise False.

    Examples:
        ``_is_order_digest("0123456789abcdef0123456789abcdef")`` returns True.
    """
    return (
        isinstance(value, str)
        and len(value) == 32
        and all(character in "0123456789abcdef" for character in value)
    )


def _relative_difference(stated: float, measured: float) -> float:
    """Return how far two areas sit apart relative to the larger one.

    Steps:
        1. Select the larger absolute magnitude as the scale.
        2. Return the scaled absolute difference, or zero when both values are zero.

    Args:
        stated: Area in square kilometres reported by adding role areas.
        measured: Area in square kilometres measured from the full extent.

    Returns:
        The absolute difference divided by the larger magnitude, or 0.0 when
        both areas are zero.

    Examples:
        ``_relative_difference(100.0, 100.0)`` returns 0.0, and
        ``_relative_difference(101.0, 100.0)`` returns about 0.0099.
    """
    scale = max(abs(stated), abs(measured))
    return 0.0 if scale == 0.0 else abs(stated - measured) / scale
