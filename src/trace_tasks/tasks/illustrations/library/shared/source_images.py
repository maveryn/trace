"""Shared source-panel helpers for library visual reconstruction tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from .....core.seed import spawn_rng
from ...shared.canvas_profiles import resolve_reconstruction_source_profile
from ..shared.rendering import render_library_scene
from ..shared.sampling import (
    bounds,
    color_support,
    make_library_section_specs,
    random_book_specs,
    render_params,
    sample_count,
    section_support,
    setting_weights,
    spawned_task_rng,
    style_weights,
)


@dataclass(frozen=True)
class LibrarySourceSceneSpec:
    """Dense library source scene shared by cutout-style visual tasks."""

    section_count: int
    section_specs: Tuple[Any, ...]
    section_keys: Tuple[str, ...]
    source_size: Tuple[int, int]
    source_profile_trace: Dict[str, Any]
    section_count_probabilities: Dict[str, float]
    section_book_counts_by_section: Dict[str, int]


def _sample_section_keys(*, rng: Any, support: Sequence[str], section_count: int) -> Tuple[str, ...]:
    values = tuple(str(value) for value in support if str(value))
    if int(section_count) > len(values):
        raise ValueError("section_count exceeds available unique library section support")
    selected = list(rng.sample(list(values), k=int(section_count)))
    rng.shuffle(selected)
    return tuple(str(value) for value in selected)


def sample_library_source_scene_spec(
    *,
    seed_namespace: str,
    instance_seed: int,
    params: Mapping[str, Any],
    attempt_index: int,
    generation_defaults: Mapping[str, Any],
    section_count_min: int,
    section_count_max: int,
    section_book_count_min: int,
    section_book_count_max: int,
    source_width: int,
    source_height: int,
) -> LibrarySourceSceneSpec:
    """Sample one dense library source panel without task-specific answer operands."""

    section_min, section_max = bounds(
        params,
        generation_defaults,
        "section_count_min",
        "section_count_max",
        int(section_count_min),
        int(section_count_max),
    )
    section_values = section_support(params, generation_defaults)
    section_count, section_count_probabilities = sample_count(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{seed_namespace}:section_count",
        low=int(section_min),
        high=min(int(section_max), len(section_values)),
        explicit_key="section_count",
    )
    rng = spawned_task_rng(int(instance_seed), f"{seed_namespace}:source_scene_spec", int(attempt_index))
    source_profile = resolve_reconstruction_source_profile(
        params=params,
        defaults=generation_defaults,
        fallback_source_width=int(source_width),
        fallback_source_height=int(source_height),
        instance_seed=int(instance_seed),
        namespace=f"{seed_namespace}:source_profile",
    )
    section_keys = _sample_section_keys(rng=rng, support=section_values, section_count=int(section_count))
    colors = color_support(params, generation_defaults)
    section_book_min, section_book_max = bounds(
        params,
        generation_defaults,
        "section_book_count_min",
        "section_book_count_max",
        int(section_book_count_min),
        int(section_book_count_max),
    )
    specs_by_section = {}
    book_counts_by_section: Dict[str, int] = {}
    for section_key in section_keys:
        count = int(rng.randint(int(section_book_min), int(section_book_max)))
        book_counts_by_section[str(section_key)] = int(count)
        specs_by_section[str(section_key)] = random_book_specs(
            rng=rng,
            section_key=str(section_key),
            count=int(count),
            colors=colors,
            role="source",
        )
    return LibrarySourceSceneSpec(
        section_count=int(section_count),
        section_specs=make_library_section_specs(section_keys=section_keys, specs_by_section=specs_by_section),
        section_keys=tuple(section_keys),
        source_size=(int(source_profile.width), int(source_profile.height)),
        source_profile_trace=dict(source_profile.trace()),
        section_count_probabilities=dict(section_count_probabilities),
        section_book_counts_by_section=dict(book_counts_by_section),
    )


def render_library_source_scene(
    *,
    seed_namespace: str,
    instance_seed: int,
    attempt_index: int,
    source: LibrarySourceSceneSpec,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    fallback: Mapping[str, Any],
) -> Any:
    """Render one sampled source library through the scene renderer."""

    scene_rng = spawn_rng(int(instance_seed), f"{seed_namespace}:scene", int(attempt_index))
    render_param_overrides = {
        **dict(params),
        "canvas_width": int(source.source_size[0]),
        "canvas_height": int(source.source_size[1]),
    }
    rp = render_params(
        render_param_overrides,
        render_defaults,
        fallback_width=int(fallback["canvas_width"]),
        fallback_height=int(fallback["canvas_height"]),
        fallback_scale=int(fallback["render_scale"]),
        instance_seed=int(instance_seed),
        namespace=f"{seed_namespace}:canvas_profile",
    )
    return render_library_scene(
        rng=scene_rng,
        section_specs=source.section_specs,
        canvas_width=int(rp["canvas_width"]),
        canvas_height=int(rp["canvas_height"]),
        render_scale=int(rp["render_scale"]),
        setting_weights=setting_weights(render_param_overrides, render_defaults),
        style_weights=style_weights(render_param_overrides, render_defaults),
        instance_seed=int(instance_seed),
        font_params=render_param_overrides,
    )


__all__ = ["LibrarySourceSceneSpec", "render_library_source_scene", "sample_library_source_scene_spec"]
