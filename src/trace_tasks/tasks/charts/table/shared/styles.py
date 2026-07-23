"""Visual-default helpers for styled table chart tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from trace_tasks.tasks.shared.font_assets import font_asset_version, sample_font_family
from trace_tasks.tasks.shared.visual_defaults import (
    default_noise_fallback,
    load_scene_background_defaults,
    load_scene_noise_defaults,
)


def solid_light_background_fallback() -> Dict[str, Any]:
    """Return the canonical low-structure table background fallback."""

    return {
        "enabled": True,
        "styles": {
            "solid_light": {
                "kind": "solid",
                "color": [248, 248, 248],
            }
        },
        "weights": {"solid_light": 1.0},
    }


def load_table_background_defaults(*, scene_id: str = "table") -> Dict[str, Any]:
    """Load table-scene background config with the canonical fallback."""

    return load_scene_background_defaults(
        domain="charts",
        scene_id=str(scene_id),
        fallback=solid_light_background_fallback(),
        merge_with_fallback=True,
    )


def load_table_noise_defaults(*, scene_id: str = "table", apply_prob: float) -> Dict[str, Any]:
    """Load table-scene post-image noise config with the canonical fallback."""

    return load_scene_noise_defaults(
        domain="charts",
        scene_id=str(scene_id),
        fallback=default_noise_fallback(apply_prob=float(apply_prob)),
        merge_with_fallback=False,
    )


def sample_table_font_family(
    *,
    instance_seed: int,
    namespace: str,
    params: Mapping[str, Any],
    exclude_tags: Sequence[str] = ("display",),
) -> str:
    """Sample one table text font family from the shared vendored font pool."""

    return str(
        sample_font_family(
            role="readout",
            instance_seed=int(instance_seed),
            namespace=str(namespace),
            params=params,
            exclude_tags=tuple(str(tag) for tag in exclude_tags),
            explicit_key="table_font_family",
            weights_key="table_font_family_weights",
        )
    )


def table_font_asset_metadata(table_font_family: str) -> Dict[str, str]:
    """Return trace metadata for the sampled table text font."""

    return {
        "font_asset_version": str(font_asset_version()),
        "table_font_family": str(table_font_family),
    }


__all__ = [
    "load_table_background_defaults",
    "load_table_noise_defaults",
    "sample_table_font_family",
    "solid_light_background_fallback",
    "table_font_asset_metadata",
]
