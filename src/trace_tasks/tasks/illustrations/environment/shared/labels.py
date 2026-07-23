"""Prompt-facing labels for environment illustration features."""

from __future__ import annotations

from typing import Dict


CROSSING_NAMES: Dict[str, str] = {"bridge": "bridges", "crosswalk": "crosswalks"}
CROSSED_FEATURE_NAMES: Dict[str, str] = {"bridge": "river", "crosswalk": "road"}


def feature_name(feature_type: str | None) -> str:
    """Return the prompt-facing name for a road/river feature type."""

    return "road" if str(feature_type) == "road" else "river"


def on_feature_phrase(feature_type: str | None) -> str:
    """Return the prompt-facing phrase for object placement on a feature."""

    return "on the road" if str(feature_type) == "road" else "in or on the river"


def feature_relation_phrase(feature_type: str | None, relation: str | None) -> str:
    """Return a prompt-facing feature relation phrase."""

    if str(relation) == "on":
        return on_feature_phrase(feature_type)
    return f"{relation} the {feature_name(feature_type)}"


__all__ = [
    "CROSSED_FEATURE_NAMES",
    "CROSSING_NAMES",
    "feature_relation_phrase",
    "feature_name",
    "on_feature_phrase",
]
