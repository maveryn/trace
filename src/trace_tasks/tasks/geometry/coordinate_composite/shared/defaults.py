"""Default loading helpers for coordinate-composite scene packages."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.geometry.shared.background_defaults import load_geometry_background_defaults
from trace_tasks.tasks.geometry.shared.noise_defaults import load_geometry_noise_defaults
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults


def load_coordinate_composite_default_sets(*, domain: str, scene_id: str) -> tuple[Any, Mapping[str, Any], Mapping[str, Any]]:
    """Load scene, background, and noise defaults for this coordinate scene."""

    return (
        get_scene_defaults(str(domain), str(scene_id)),
        load_geometry_background_defaults(scene_id=str(scene_id)),
        load_geometry_noise_defaults(scene_id=str(scene_id)),
    )


def split_coordinate_composite_defaults(scene_defaults: Any, *, public_identifier: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Split defaults for one public task without storing public identity in shared code."""

    return split_scene_generation_rendering_prompt_defaults(
        scene_defaults if isinstance(scene_defaults, Mapping) else {},
        task_id=str(public_identifier),
    )


__all__ = ["load_coordinate_composite_default_sets", "split_coordinate_composite_defaults"]
