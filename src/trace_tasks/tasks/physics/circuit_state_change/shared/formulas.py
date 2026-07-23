"""Formula helpers for circuit state-change bulb brightness."""

from __future__ import annotations

from typing import Dict, Mapping


EPS = 1e-9


def powers_for_state(resistances: Mapping[str, int], *, switch_closed: bool) -> Dict[str, float]:
    """Return relative bulb powers for one switch state."""

    r_series = float(resistances["series_bulb"])
    r_main = float(resistances["main_branch_bulb"])
    r_switched = float(resistances["switched_branch_bulb"])
    r_reference_1 = float(resistances["reference_branch_bulb_1"])
    r_reference_2 = float(resistances["reference_branch_bulb_2"])
    reference_current = 1.0 / (r_reference_1 + r_reference_2)
    reference_powers = {
        "reference_branch_bulb_1": float(reference_current * reference_current * r_reference_1),
        "reference_branch_bulb_2": float(reference_current * reference_current * r_reference_2),
    }
    if not bool(switch_closed):
        current = 1.0 / (r_series + r_main)
        return {
            "series_bulb": float(current * current * r_series),
            "main_branch_bulb": float(current * current * r_main),
            "switched_branch_bulb": 0.0,
            **reference_powers,
        }
    parallel = (r_main * r_switched) / (r_main + r_switched)
    total_current = 1.0 / (r_series + parallel)
    branch_voltage = total_current * parallel
    return {
        "series_bulb": float(total_current * total_current * r_series),
        "main_branch_bulb": float(branch_voltage * branch_voltage / r_main),
        "switched_branch_bulb": float(branch_voltage * branch_voltage / r_switched),
        **reference_powers,
    }


def change_class(before: float, after: float) -> str:
    """Classify the visible brightness change between two switch states."""

    if float(before) <= EPS and float(after) > EPS:
        return "turns_on"
    if float(before) > EPS and float(after) <= EPS:
        return "turns_off"
    if float(before) > EPS and float(after) > float(before) + EPS:
        return "brightens"
    if float(after) > EPS and float(after) + EPS < float(before):
        return "dims"
    return "unchanged"


__all__ = ["change_class", "powers_for_state"]
