"""Identity-free semantic sampling primitives for Go scene tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.tasks.games.shared.layout import (
    attach_games_unit_size_jitter,
    resolve_games_layout_jitter,
    resolve_games_unit_size_scale,
    scale_games_px,
)
from trace_tasks.tasks.games.shared.sampling import resolve_games_integer_axis, resolve_games_named_axis
from trace_tasks.tasks.games.shared.style import SUPPORTED_GO_STYLE_VARIANTS
from trace_tasks.tasks.shared.config_defaults import group_default

from .rendering import GoRenderParams
from .state import (
    DEFAULTS,
    GO_NAMESPACE,
    SUPPORTED_GO_PLAYER_COLORS,
    SUPPORTED_GO_SCENE_VARIANTS,
    GoIntegerAxis,
    GoPlayerColorAxis,
    GoSceneAxes,
)


def _resolve_named_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace: str,
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    supported: Sequence[str],
) -> Tuple[str, Dict[str, float]]:
    """Resolve one named semantic or visual axis without public-task identity."""

    return resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=f"{GO_NAMESPACE}.{str(namespace)}",
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        balance_flag_key=str(balance_flag_key),
        supported_variants=[str(item) for item in supported],
    )


def resolve_go_scene_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> GoSceneAxes:
    """Resolve scene and style axes shared by Go tasks."""

    scene_variant, scene_variant_probabilities = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace="scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported=SUPPORTED_GO_SCENE_VARIANTS,
    )
    style_variant, style_variant_probabilities = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace="style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported=SUPPORTED_GO_STYLE_VARIANTS,
    )
    return GoSceneAxes(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        style_variant_probabilities=dict(style_variant_probabilities),
    )


def resolve_go_player_color_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> GoPlayerColorAxis:
    """Resolve the marked-group player color for local Go group tasks."""

    player_color, probabilities = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace="player_color",
        explicit_key="player_color",
        weights_key="player_color_weights",
        balance_flag_key="balanced_player_color_sampling",
        supported=SUPPORTED_GO_PLAYER_COLORS,
    )
    return GoPlayerColorAxis(player_color=str(player_color), probabilities=dict(probabilities))


def resolve_go_integer_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[int],
    namespace: str,
    balanced_flag_key: str,
) -> GoIntegerAxis:
    """Resolve one task-owned integer axis."""

    value, support, probabilities = resolve_games_integer_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(support_key),
        explicit_key=str(explicit_key),
        fallback_support=tuple(int(value) for value in fallback_support),
        namespace=f"{GO_NAMESPACE}.{str(namespace)}",
        balanced_flag_key=str(balanced_flag_key),
    )
    return GoIntegerAxis(
        value=int(value),
        support=tuple(int(item) for item in support),
        probabilities=dict(probabilities),
    )


def resolve_go_target_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    fallback_support: Sequence[int],
    namespace: str,
) -> GoIntegerAxis:
    """Resolve one Go count answer target."""

    return resolve_go_integer_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(support_key),
        explicit_key="target_answer",
        fallback_support=tuple(int(value) for value in fallback_support),
        namespace=str(namespace),
        balanced_flag_key="balanced_target_answer_sampling",
    )


def resolve_go_board_size_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> GoIntegerAxis:
    """Resolve Go board size."""

    return resolve_go_integer_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key="board_size_support",
        explicit_key="board_size",
        fallback_support=DEFAULTS.board_size_support,
        namespace="board_size",
        balanced_flag_key="balanced_board_size_sampling",
    )


def resolve_go_render_params(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    instance_seed: int,
) -> GoRenderParams:
    """Resolve Go rendering parameters from config/defaults."""

    unit_scale, unit_scale_meta = resolve_games_unit_size_scale(
        params,
        render_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{GO_NAMESPACE}.unit_size",
    )
    layout_jitter = attach_games_unit_size_jitter(
        resolve_games_layout_jitter(
            params,
            render_defaults,
            instance_seed=int(instance_seed),
            namespace=f"{GO_NAMESPACE}.layout",
        ),
        unit_scale_meta,
    )
    base_canvas_width = int(params.get("canvas_width", group_default(render_defaults, "canvas_width", DEFAULTS.canvas_width)))
    base_canvas_height = int(params.get("canvas_height", group_default(render_defaults, "canvas_height", DEFAULTS.canvas_height)))
    max_board_size_px = scale_games_px(
        params.get("max_board_size_px", group_default(render_defaults, "max_board_size_px", DEFAULTS.max_board_size_px)),
        unit_scale,
        min_px=380,
    )
    dynamic_canvas_enabled = bool(
        params.get(
            "dynamic_canvas_size_enabled",
            group_default(render_defaults, "dynamic_canvas_size_enabled", DEFAULTS.dynamic_canvas_size_enabled),
        )
    )
    canvas_width = int(base_canvas_width)
    canvas_height = int(base_canvas_height)
    if dynamic_canvas_enabled and params.get("canvas_width") is None:
        canvas_width = min(
            int(base_canvas_width),
            max(
                int(params.get("canvas_min_width_px", group_default(render_defaults, "canvas_min_width_px", DEFAULTS.canvas_min_width_px))),
                int(round(float(max_board_size_px) + (2.0 * float(params.get("canvas_side_padding_px", group_default(render_defaults, "canvas_side_padding_px", DEFAULTS.canvas_side_padding_px)))))),
            ),
        )
    if dynamic_canvas_enabled and params.get("canvas_height") is None:
        canvas_height = min(
            int(base_canvas_height),
            max(
                int(params.get("canvas_min_height_px", group_default(render_defaults, "canvas_min_height_px", DEFAULTS.canvas_min_height_px))),
                int(round(float(max_board_size_px) + (2.0 * float(params.get("canvas_vertical_padding_px", group_default(render_defaults, "canvas_vertical_padding_px", DEFAULTS.canvas_vertical_padding_px)))))),
            ),
        )
    return GoRenderParams(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        panel_margin_px=int(params.get("panel_margin_px", group_default(render_defaults, "panel_margin_px", DEFAULTS.panel_margin_px))),
        max_board_size_px=int(max_board_size_px),
        board_padding_px=scale_games_px(
            params.get("board_padding_px", group_default(render_defaults, "board_padding_px", DEFAULTS.board_padding_px)),
            unit_scale,
            min_px=38,
        ),
        board_corner_radius_px=scale_games_px(
            params.get("board_corner_radius_px", group_default(render_defaults, "board_corner_radius_px", DEFAULTS.board_corner_radius_px)),
            unit_scale,
            min_px=10,
        ),
        board_frame_width_px=scale_games_px(
            params.get("board_frame_width_px", group_default(render_defaults, "board_frame_width_px", DEFAULTS.board_frame_width_px)),
            unit_scale,
            min_px=6,
        ),
        line_width_px=scale_games_px(
            params.get("line_width_px", group_default(render_defaults, "line_width_px", DEFAULTS.line_width_px)),
            unit_scale,
            min_px=2,
        ),
        point_radius_px=scale_games_px(
            params.get("point_radius_px", group_default(render_defaults, "point_radius_px", DEFAULTS.point_radius_px)),
            unit_scale,
            min_px=2,
        ),
        stone_radius_fraction=float(params.get("stone_radius_fraction", group_default(render_defaults, "stone_radius_fraction", DEFAULTS.stone_radius_fraction))),
        highlight_outline_width_px=scale_games_px(
            params.get("highlight_outline_width_px", group_default(render_defaults, "highlight_outline_width_px", DEFAULTS.highlight_outline_width_px)),
            unit_scale,
            min_px=5,
        ),
        liberty_bbox_fraction=float(params.get("liberty_bbox_fraction", group_default(render_defaults, "liberty_bbox_fraction", DEFAULTS.liberty_bbox_fraction))),
        layout_jitter_meta=layout_jitter,
        instance_seed=int(instance_seed),
    )


__all__ = [
    "resolve_go_board_size_axis",
    "resolve_go_integer_axis",
    "resolve_go_player_color_axis",
    "resolve_go_render_params",
    "resolve_go_scene_axes",
    "resolve_go_target_axis",
]
