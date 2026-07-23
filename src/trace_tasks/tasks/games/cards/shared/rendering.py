"""Rendering wrappers for cards scene tasks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Sequence

from PIL import Image

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.games.shared.layout import resolve_games_layout_jitter
from trace_tasks.tasks.games.shared.scene_style import make_panel_scene_background, resolve_game_panel_scene_style
from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults
from trace_tasks.tasks.shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.font_assets import sample_font_family

from .components import CardRenderParams, RenderedCardHandScene, render_cards_hand_scene
from .defaults import RENDER_FALLBACKS, SCENE_ID
from .state import CardInstance

_GEN_DEFAULTS_UNUSED, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
)
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)


@dataclass(frozen=True)
class RenderedCardsTaskContext:
    """Rendered cards scene plus post-image and style metadata."""

    image: Image.Image
    rendered_scene: RenderedCardHandScene
    render_params: CardRenderParams
    panel_style_meta: Dict[str, Any]
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]


def resolve_cards_render_params(params: Mapping[str, Any], *, instance_seed: int) -> CardRenderParams:
    """Resolve card-scene rendering parameters from config/defaults."""

    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace="games.cards.font",
        params=params,
    )
    defaults = RENDER_FALLBACKS
    return CardRenderParams(
        canvas_width=int(params.get("canvas_width", group_default(_RENDER_DEFAULTS, "canvas_width", defaults.canvas_width))),
        canvas_height=int(params.get("canvas_height", group_default(_RENDER_DEFAULTS, "canvas_height", defaults.canvas_height))),
        card_width_px=int(params.get("card_width_px", group_default(_RENDER_DEFAULTS, "card_width_px", defaults.card_width_px))),
        card_height_px=int(params.get("card_height_px", group_default(_RENDER_DEFAULTS, "card_height_px", defaults.card_height_px))),
        panel_margin_px=int(params.get("panel_margin_px", group_default(_RENDER_DEFAULTS, "panel_margin_px", defaults.panel_margin_px))),
        card_gap_px=int(params.get("card_gap_px", group_default(_RENDER_DEFAULTS, "card_gap_px", defaults.card_gap_px))),
        row_gap_px=int(params.get("row_gap_px", group_default(_RENDER_DEFAULTS, "row_gap_px", defaults.row_gap_px))),
        card_corner_radius_px=int(params.get("card_corner_radius_px", group_default(_RENDER_DEFAULTS, "card_corner_radius_px", defaults.card_corner_radius_px))),
        rank_font_size_px=int(params.get("rank_font_size_px", group_default(_RENDER_DEFAULTS, "rank_font_size_px", defaults.rank_font_size_px))),
        center_symbol_font_size_px=int(params.get("center_symbol_font_size_px", group_default(_RENDER_DEFAULTS, "center_symbol_font_size_px", defaults.center_symbol_font_size_px))),
        reference_banner_height_px=int(params.get("reference_banner_height_px", group_default(_RENDER_DEFAULTS, "reference_banner_height_px", defaults.reference_banner_height_px))),
        reference_font_size_px=int(params.get("reference_font_size_px", group_default(_RENDER_DEFAULTS, "reference_font_size_px", defaults.reference_font_size_px))),
        continuation_font_size_px=int(params.get("continuation_font_size_px", group_default(_RENDER_DEFAULTS, "continuation_font_size_px", defaults.continuation_font_size_px))),
        continuation_gap_px=int(params.get("continuation_gap_px", group_default(_RENDER_DEFAULTS, "continuation_gap_px", defaults.continuation_gap_px))),
        max_cards_per_row=int(params.get("max_cards_per_row", group_default(_RENDER_DEFAULTS, "max_cards_per_row", defaults.max_cards_per_row))),
        center_label_mode=str(params.get("center_label_mode", group_default(_RENDER_DEFAULTS, "center_label_mode", "suit_symbol"))),
        layout_jitter_meta=resolve_games_layout_jitter(
            params,
            _RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            namespace="games.cards.layout",
        ),
        group_label_font_size_px=int(params.get("group_label_font_size_px", group_default(_RENDER_DEFAULTS, "group_label_font_size_px", defaults.group_label_font_size_px))),
        font_family=str(font_family),
    )


def apply_cards_render_overrides(
    render_params: CardRenderParams,
    overrides: Mapping[str, int],
    *,
    center_label_mode: str | None = None,
    max_cards_per_row: int | None = None,
) -> CardRenderParams:
    """Return render params with sample-level card layout overrides applied."""

    return replace(
        render_params,
        canvas_width=int(overrides.get("canvas_width", render_params.canvas_width)),
        canvas_height=int(overrides.get("canvas_height", render_params.canvas_height)),
        card_width_px=int(overrides.get("card_width_px", render_params.card_width_px)),
        card_height_px=int(overrides.get("card_height_px", render_params.card_height_px)),
        card_gap_px=int(overrides.get("card_gap_px", render_params.card_gap_px)),
        row_gap_px=int(overrides.get("row_gap_px", render_params.row_gap_px)),
        rank_font_size_px=int(overrides.get("rank_font_size_px", render_params.rank_font_size_px)),
        center_symbol_font_size_px=int(overrides.get("center_symbol_font_size_px", render_params.center_symbol_font_size_px)),
        reference_banner_height_px=int(overrides.get("reference_banner_height_px", render_params.reference_banner_height_px)),
        reference_font_size_px=int(overrides.get("reference_font_size_px", render_params.reference_font_size_px)),
        max_cards_per_row=int(overrides.get("max_cards_per_row", max_cards_per_row if max_cards_per_row is not None else render_params.max_cards_per_row)),
        center_label_mode=str(center_label_mode if center_label_mode is not None else render_params.center_label_mode),
    )


def render_cards_task_scene(
    *,
    cards: Sequence[CardInstance],
    scene_variant: str,
    style_variant: str,
    params: Mapping[str, Any],
    instance_seed: int,
    render_params: CardRenderParams | None = None,
    show_continuation_cue: bool = False,
    row_card_counts: Sequence[int] | None = None,
) -> RenderedCardsTaskContext:
    """Render cards on a background and apply post-image noise."""

    resolved_params = render_params or resolve_cards_render_params(params, instance_seed=int(instance_seed))
    allowed_panel_treatments_raw = params.get(
        "panel_scene_treatments",
        group_default(_RENDER_DEFAULTS, "panel_scene_treatments", None),
    )
    if isinstance(allowed_panel_treatments_raw, str):
        allowed_panel_treatments = (str(allowed_panel_treatments_raw),)
    elif allowed_panel_treatments_raw is None:
        allowed_panel_treatments = None
    else:
        allowed_panel_treatments = tuple(str(item) for item in allowed_panel_treatments_raw)
    panel_style, panel_style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace="games.cards.panel_scene_style",
        treatments=allowed_panel_treatments,
        treatment_weights=params.get(
            "panel_scene_treatment_weights",
            group_default(_RENDER_DEFAULTS, "panel_scene_treatment_weights", None),
        ),
        palette_weights=params.get(
            "panel_scene_palette_weights",
            group_default(_RENDER_DEFAULTS, "panel_scene_palette_weights", None),
        ),
    )
    background, background_meta = make_panel_scene_background(
        canvas_width=int(resolved_params.canvas_width),
        canvas_height=int(resolved_params.canvas_height),
        style=panel_style,
    )
    rendered_scene = render_cards_hand_scene(
        cards=list(cards),
        background=background,
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        params=resolved_params,
        panel_style=panel_style,
        show_continuation_cue=bool(show_continuation_cue),
        row_card_counts=tuple(int(value) for value in row_card_counts) if row_card_counts else None,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return RenderedCardsTaskContext(
        image=image,
        rendered_scene=rendered_scene,
        render_params=resolved_params,
        panel_style_meta=dict(panel_style_meta),
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
    )


__all__ = [
    "RenderedCardsTaskContext",
    "apply_cards_render_overrides",
    "render_cards_task_scene",
    "resolve_cards_render_params",
]
