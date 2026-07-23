"""Shared source-panel helpers for indoor-room visual reconstruction tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from .....core.seed import spawn_rng
from ...shared.canvas_profiles import resolve_reconstruction_source_profile
from ...shared.task_support import bounds as _shared_bounds
from ...shared.task_support import sample_count as _shared_sample_count
from ..shared.output import render_fallback_from_defaults
from ..shared.rendering import render_indoor_scene_from_specs
from ..shared.sampling import support_choice, theme_support, typed_support
from ..shared.state import (
    INDOOR_CONTAINER_TYPES,
    INDOOR_OBJECT_TYPES,
    INDOOR_SURFACE_TYPES,
    IndoorObjectSpec,
)


@dataclass(frozen=True)
class IndoorSourceSceneSpec:
    """Dense room source scene shared by cutout-style visual tasks."""

    theme_id: str
    source_object_count: int
    source_size: Tuple[int, int]
    source_profile_trace: Dict[str, Any]
    specs: Tuple[IndoorObjectSpec, ...]
    theme_probabilities: Dict[str, float]
    source_object_count_probabilities: Dict[str, float]


def sample_indoor_source_scene_spec(
    *,
    seed_namespace: str,
    instance_seed: int,
    params: Mapping[str, Any],
    attempt_index: int,
    generation_defaults: Mapping[str, Any],
    source_object_count_min: int,
    source_object_count_max: int,
    source_width: int,
    source_height: int,
) -> IndoorSourceSceneSpec:
    """Sample one dense room source panel without task-specific answer operands."""

    theme_id, theme_probabilities = support_choice(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{seed_namespace}:theme",
        support=theme_support(params, generation_defaults),
        explicit_key="theme_id",
    )
    object_support = typed_support(
        params,
        generation_defaults,
        param_key="object_type_support",
        default_key="indoor_object_type_support",
        fallback=INDOOR_OBJECT_TYPES,
        error_name="object_type_support",
    )
    source_profile = resolve_reconstruction_source_profile(
        params=params,
        defaults=generation_defaults,
        fallback_source_width=int(source_width),
        fallback_source_height=int(source_height),
        instance_seed=int(instance_seed),
        namespace=f"{seed_namespace}:source_profile",
    )
    object_min, object_max = _shared_bounds(
        params,
        generation_defaults,
        "source_object_count_min",
        "source_object_count_max",
        int(source_object_count_min),
        int(source_object_count_max),
    )
    source_object_count, object_count_probabilities = _shared_sample_count(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{seed_namespace}:source_object_count",
        low=int(object_min),
        high=int(object_max),
        explicit_key="source_object_count",
    )
    rng = spawn_rng(int(instance_seed), f"{seed_namespace}:source_scene_spec", int(attempt_index))
    base_targets: list[Tuple[str, str]] = [("surface", str(surface_type)) for surface_type in INDOOR_SURFACE_TYPES]
    base_targets.extend(("container", str(container_type)) for container_type in INDOOR_CONTAINER_TYPES)
    base_targets.extend(
        (
            ("region", "table:left"),
            ("region", "table:right"),
            ("region", "table:above"),
            ("region", "table:below"),
            ("region", "sofa:above"),
            ("region", "sofa:below"),
            ("region", "cabinet:above"),
            ("region", "cabinet:below"),
        )
    )
    rng.shuffle(base_targets)
    specs: list[IndoorObjectSpec] = []
    for index in range(int(source_object_count)):
        placement_kind, target_type = base_targets[index] if index < len(base_targets) else rng.choice(base_targets)
        specs.append(
            IndoorObjectSpec(
                object_type=str(rng.choice(object_support)),
                placement_kind=str(placement_kind),
                target_type=str(target_type),
                role="source",
            )
        )
    rng.shuffle(specs)
    return IndoorSourceSceneSpec(
        theme_id=str(theme_id),
        source_object_count=int(source_object_count),
        source_size=(int(source_profile.width), int(source_profile.height)),
        source_profile_trace=dict(source_profile.trace()),
        specs=tuple(specs),
        theme_probabilities=dict(theme_probabilities),
        source_object_count_probabilities=dict(object_count_probabilities),
    )


def render_indoor_source_scene(
    *,
    render_namespace: str,
    instance_seed: int,
    attempt_index: int,
    source: IndoorSourceSceneSpec,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    fallback_defaults: Any,
) -> Any:
    """Render one sampled source room through the scene renderer."""

    render_params = {
        **dict(params),
        "canvas_width": int(source.source_size[0]),
        "canvas_height": int(source.source_size[1]),
    }
    return render_indoor_scene_from_specs(
        render_namespace=str(render_namespace),
        instance_seed=int(instance_seed),
        attempt_index=int(attempt_index),
        specs=source.specs,
        theme_id=str(source.theme_id),
        params=render_params,
        render_defaults=render_defaults,
        fallback=render_fallback_from_defaults(fallback_defaults),
    )


__all__ = ["IndoorSourceSceneSpec", "render_indoor_source_scene", "sample_indoor_source_scene_spec"]
