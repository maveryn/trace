"""Neutral Life automaton scene plumbing shared by public task files."""

from __future__ import annotations

from typing import Any, Mapping

from .shared.layout import fit_life_render_params
from .shared.rendering import render_life_scene_bundle
from .shared.rules import SCENE_ID
from .shared.state import LifeRenderBundle, LifeSceneSpec
from .shared.styles import resolve_render_params


def render_life_task_scene(
    *,
    scene: LifeSceneSpec,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    scene_variant: str,
    instance_seed: int,
    sampling_scope: str,
) -> LifeRenderBundle:
    """Resolve render params, fit the scene, and render it."""

    render_params = resolve_render_params(params, render_defaults, instance_seed=int(instance_seed))
    render_params = fit_life_render_params(scene=scene, render_params=render_params)
    return render_life_scene_bundle(
        scene=scene,
        params=params,
        gen_defaults=gen_defaults,
        render_params=render_params,
        scene_variant=str(scene_variant),
        instance_seed=int(instance_seed),
        sampling_scope=str(sampling_scope),
    )


def life_query_params(
    *,
    query_id: str,
    query_probabilities: Mapping[str, float],
    question_format: str,
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
    render_bundle: LifeRenderBundle,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return scene/query metadata common to Life tasks."""

    payload = {
        "query_id": str(query_id),
        "query_id_probabilities": {str(key): float(value) for key, value in query_probabilities.items()},
        "question_format": str(question_format),
        "scene_id": SCENE_ID,
        "scene_variant": str(scene_variant),
        "scene_variant_probabilities": {str(key): float(value) for key, value in scene_variant_probabilities.items()},
        "life_board_style": str(render_bundle.board_style),
        "life_board_style_probabilities": dict(render_bundle.board_style_probabilities),
        "life_cell_palette_id": str(render_bundle.cell_palette_id),
        "life_cell_palette_probabilities": dict(render_bundle.cell_palette_probabilities),
    }
    if extra:
        payload.update({str(key): value for key, value in extra.items()})
    return payload
