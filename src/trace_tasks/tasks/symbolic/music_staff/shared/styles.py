"""Rendering style resolution for symbolic music-staff scenes."""

from __future__ import annotations

from typing import Any, Mapping

from ...shared.scene_style import SymbolicSceneStyle, make_symbolic_scene_background, resolve_symbolic_scene_style

from .components import MusicRenderParams, resolve_music_render_params


def resolve_render_params(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    instance_seed: int,
) -> MusicRenderParams:
    """Resolve music renderer dimensions from scene defaults."""

    return resolve_music_render_params(params, defaults, instance_seed=int(instance_seed))


def resolve_background(*, instance_seed: int, canvas_width: int, canvas_height: int) -> tuple[object, dict[str, Any], SymbolicSceneStyle, dict[str, Any]]:
    """Resolve the symbolic scene style and background image."""

    scene_style, scene_style_meta = resolve_symbolic_scene_style(
        instance_seed=int(instance_seed),
        namespace="music_staff.background",
    )
    background, background_meta = make_symbolic_scene_background(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        style=scene_style,
    )
    return background, dict(background_meta), scene_style, dict(scene_style_meta)
