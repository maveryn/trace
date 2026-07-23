"""Scene-style resolution for symbolic Turing tape rendering."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from ...shared.scene_style import SYMBOLIC_SCENE_TREATMENTS, SymbolicSceneStyle, resolve_panel_chrome_mode, resolve_symbolic_scene_style

from .defaults import style_meta_with_font
from .state import TuringRenderParams


def resolve_turing_style(
    *,
    scene_variant: str,
    render_params: TuringRenderParams,
) -> Tuple[SymbolicSceneStyle, Dict[str, Any]]:
    """Resolve one non-semantic scene-level style pack."""

    style, metadata = resolve_symbolic_scene_style(
        instance_seed=int(render_params.layout_seed),
        namespace=f"turing_tape.{scene_variant}",
        treatments=tuple(SYMBOLIC_SCENE_TREATMENTS),
    )
    chrome_mode, chrome_metadata = resolve_panel_chrome_mode(
        instance_seed=int(render_params.layout_seed),
        namespace=f"turing_tape.{scene_variant}",
    )
    return style, style_meta_with_font(
        {
            **dict(metadata),
            "scene_variant": str(scene_variant),
            "panel_chrome": dict(chrome_metadata),
            "panel_chrome_mode": str(chrome_mode),
        },
        render_params,
    )
