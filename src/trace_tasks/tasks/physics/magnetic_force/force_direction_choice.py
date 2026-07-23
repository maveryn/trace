"""Choose the magnetic-force direction for a moving charge."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....core.seed import spawn_rng
from ....core.query_ids import SINGLE_QUERY_ID
from ....core.types import TypedValue
from ....core.visual.noise import apply_post_image_noise
from ...base import TaskOutput
from ...registry import register_task
from ...shared.bbox_projection import bbox_union_many as _bbox_union
from ...shared.config_defaults import (
    group_default,
    load_scene_generation_rendering_prompt_defaults,
    required_group_defaults,
)
from ...shared.drawing import draw_centered_text, draw_rounded_rect
from ...shared.fixed_query import select_task_query_id
from ...shared.font_assets import font_asset_version, get_font_family_record, sample_font_family
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_json_example import build_prompt_json_examples
from ...shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_query_spec,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)
from ...shared.render_variation import resolve_layout_jitter, resolve_render_int
from ...shared.text_rendering import load_font, resolve_text_stroke_fill
from ...shared.variant_sampling import (
    apply_balanced_variant_sampling,
    is_uniform_probability_map,
    resolve_variant,
)
from ..shared.diagram_style import prepare_physics_diagram_style_and_background
from ..shared.style import SUPPORTED_PHYSICS_COLOR_NAMES, build_physics_magnetism_theme
from ..shared.vector_arrows import (
    SEMANTIC_DIRECTION_VECTORS,
    centered_arrow_endpoints,
    direction_endpoint,
    draw_arrow_with_bbox,
)
from ..shared.visual_defaults import load_physics_noise_defaults


TASK_ID = "task_physics__magnetic_force__force_direction_choice"
TASK_NAMESPACE = "physics_magnetic_force_force_direction_choice"
SCENE_ID = "magnetic_force"
SCENE_PROMPT_KEY = "magnetic_force_field"
TASK_PROMPT_KEY = "force_direction_choice_query"
PROMPT_QUERY_KEY = "single"
INTERNAL_QUERY_ID = "force_direction_choice"
SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = (
    "clean_panel",
    "field_grid",
    "lab_card",
)
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (
    SINGLE_QUERY_ID,
)
SUPPORTED_FIELD_ORIENTATIONS: Tuple[str, ...] = ("out_of_page", "into_page")
SUPPORTED_DIRECTIONS: Tuple[str, ...] = (
    "east",
    "northeast",
    "north",
    "northwest",
    "west",
    "southwest",
    "south",
    "southeast",
)
DIRECTION_VECTORS: Dict[str, Tuple[int, int]] = dict(SEMANTIC_DIRECTION_VECTORS)
OPTION_LETTERS: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "H")


@dataclass(frozen=True)
class _TaskDefaults:
    """Stable fallback defaults for magnetic-force scenes."""

    canvas_width: int = 1180
    canvas_height: int = 760
    panel_left_px: int = 58
    panel_top_px: int = 58
    panel_width_px: int = 760
    panel_height_px: int = 500
    side_left_px: int = 850
    side_top_px: int = 82
    side_width_px: int = 270
    field_symbol_spacing_px: int = 92
    particle_radius_px: int = 36
    arrow_length_px: int = 148
    arrow_width_px: int = 9
    arrow_head_length_px: int = 24
    arrow_head_width_px: int = 22
    option_cell_width_px: int = 126
    option_cell_height_px: int = 104
    option_cell_gap_x_px: int = 16
    option_cell_gap_y_px: int = 20
    option_arrow_length_px: int = 64
    option_arrow_width_px: int = 8
    option_arrow_head_length_px: int = 20
    option_arrow_head_width_px: int = 18
    label_font_size_px: int = 22
    symbol_font_size_px: int = 30
    particle_font_size_px: int = 31
    option_font_size_px: int = 23
    equation_font_size_px: int = 25


@dataclass(frozen=True)
class _ResolvedAxes:
    """Resolved scene/query axes for one instance."""

    scene_variant: str
    query_id: str
    field_orientation: str
    velocity_direction: str | None
    charge_sign: int
    correct_option_letter: str | None
    accent_color_name: str
    scene_variant_probabilities: Dict[str, float]
    query_id_probabilities: Dict[str, float]
    field_orientation_probabilities: Dict[str, float]
    velocity_direction_probabilities: Dict[str, float]
    charge_sign_probabilities: Dict[str, float]
    correct_option_letter_probabilities: Dict[str, float]
    accent_color_name_probabilities: Dict[str, float]


@dataclass(frozen=True)
class _DirectionScenario:
    """One symbolic magnetic-force direction setup."""

    charge_sign: int
    field_orientation: str
    velocity_direction: str
    force_direction: str
    option_directions: Dict[str, str]


@dataclass(frozen=True)
class _SceneSpec:
    """Resolved symbolic magnetism scene."""

    scene_variant: str
    query_id: str
    field_orientation: str
    direction_scenario: _DirectionScenario | None
    correct_option_letter: str | None
    target_answer: int | str
    annotation_entity_ids: Tuple[str, ...]


@dataclass(frozen=True)
class _RenderedScene:
    """Rendered magnetism scene plus prompt-facing annotation metadata."""

    image: Image.Image
    annotation_bbox_map: Dict[str, List[float]]
    annotation_entity_ids: List[str]
    scene_entities: List[Dict[str, Any]]
    render_map: Dict[str, Any]


_DEFAULTS = _TaskDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "physics",
    SCENE_ID,
    task_id=TASK_ID,
)
POST_IMAGE_NOISE_DEFAULTS = load_physics_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def _with_sampling_divisor(params: Mapping[str, Any], *, divisor: int, explicit_keys: Sequence[str]) -> Mapping[str, Any]:
    """No-op hook for axis-decoupling call sites."""

    _ = int(divisor), explicit_keys
    return params


def _charge_sign_probability_map(selected: int | None = None) -> Dict[str, float]:
    """Return a probability map over charge signs."""

    if selected is not None:
        return {str(int(selected)): 1.0}
    return {"-1": 0.5, "1": 0.5}


def _resolve_scene_variant(instance_seed: int, *, params: Mapping[str, Any]) -> Tuple[str, Dict[str, float]]:
    """Resolve the scene style axis."""

    selected, probabilities = resolve_variant(
        spawn_rng(int(instance_seed), f"{TASK_NAMESPACE}.scene_variant"),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        supported_variants=SUPPORTED_SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=SUPPORTED_SCENE_VARIANTS,
        balance_flag_key="balanced_scene_variant_sampling",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        sampling_namespace=f"{TASK_NAMESPACE}.scene_variant",
    )
    return str(selected), {str(key): float(value) for key, value in sorted(probabilities.items())}


def _resolve_field_orientation(instance_seed: int, *, params: Mapping[str, Any]) -> Tuple[str, Dict[str, float]]:
    """Resolve whether B points into or out of the page."""

    selected, probabilities = resolve_variant(
        spawn_rng(int(instance_seed), f"{TASK_ID}.field_orientation"),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        supported_variants=SUPPORTED_FIELD_ORIENTATIONS,
        explicit_key="field_orientation",
        weights_key="field_orientation_weights",
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=SUPPORTED_FIELD_ORIENTATIONS,
        balance_flag_key="balanced_field_orientation_sampling",
        explicit_key="field_orientation",
        weights_key="field_orientation_weights",
        sampling_namespace=f"{TASK_ID}.field_orientation",
    )
    return str(selected), {str(key): float(value) for key, value in sorted(probabilities.items())}


def _resolve_velocity_direction(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    internal_query_id: str,
) -> Tuple[str | None, Dict[str, float]]:
    """Resolve the velocity direction for force-direction scenes."""

    if str(internal_query_id) != INTERNAL_QUERY_ID:
        return None, {}
    adjusted_params = _with_sampling_divisor(params, divisor=len(SUPPORTED_QUERY_IDS), explicit_keys=("velocity_direction",))
    selected, probabilities = resolve_variant(
        spawn_rng(int(instance_seed), f"{TASK_ID}.velocity_direction"),
        params=adjusted_params,
        gen_defaults=_GEN_DEFAULTS,
        supported_variants=SUPPORTED_DIRECTIONS,
        explicit_key="velocity_direction",
        weights_key="velocity_direction_weights",
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=adjusted_params,
        gen_defaults=_GEN_DEFAULTS,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=SUPPORTED_DIRECTIONS,
        balance_flag_key="balanced_velocity_direction_sampling",
        explicit_key="velocity_direction",
        weights_key="velocity_direction_weights",
        sampling_namespace=f"{TASK_ID}.velocity_direction",
    )
    return str(selected), {str(key): float(value) for key, value in sorted(probabilities.items())}


def _resolve_charge_sign(instance_seed: int, *, params: Mapping[str, Any]) -> Tuple[int, Dict[str, float]]:
    """Resolve the sign of the visible charge."""

    explicit = params.get("charge_sign")
    if explicit is not None:
        selected = int(explicit)
        if int(selected) not in {-1, 1}:
            raise ValueError(f"unsupported charge_sign: {explicit}")
        return int(selected), _charge_sign_probability_map(int(selected))
    balanced_enabled = bool(params.get("balanced_charge_sign_sampling", group_default(_GEN_DEFAULTS, "balanced_charge_sign_sampling", True)))
    if bool(balanced_enabled):
        selected = 1 if int(spawn_rng(int(instance_seed), f"{TASK_ID}.charge_sign").randrange(2)) == 0 else -1
    else:
        selected = 1 if int(spawn_rng(int(instance_seed), f"{TASK_ID}.charge_sign").randrange(2)) == 0 else -1
    return int(selected), _charge_sign_probability_map()


def _resolve_correct_option_letter(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    internal_query_id: str,
) -> Tuple[str | None, Dict[str, float]]:
    """Resolve the correct option letter for force-direction scenes."""

    if str(internal_query_id) != INTERNAL_QUERY_ID:
        return None, {}
    adjusted_params = dict(_with_sampling_divisor(params, divisor=len(SUPPORTED_QUERY_IDS), explicit_keys=("correct_option_letter",)))
    if adjusted_params.get("correct_option_letter") is None and adjusted_params.get("target_answer") is not None:
        adjusted_params["correct_option_letter"] = str(adjusted_params["target_answer"]).strip().upper()
    selected, probabilities = resolve_variant(
        spawn_rng(int(instance_seed), f"{TASK_ID}.correct_option_letter"),
        params=adjusted_params,
        gen_defaults=_GEN_DEFAULTS,
        supported_variants=OPTION_LETTERS,
        explicit_key="correct_option_letter",
        weights_key="direction_option_letter_weights",
    )
    balanced_enabled = bool(adjusted_params.get("balanced_direction_option_letter_sampling", group_default(_GEN_DEFAULTS, "balanced_direction_option_letter_sampling", True)))
    has_override = any(adjusted_params.get(str(key)) is not None for key in ("correct_option_letter", "direction_option_letter_weights"))
    if bool(balanced_enabled) and not bool(has_override) and is_uniform_probability_map(probabilities):
        selected = apply_balanced_variant_sampling(
            instance_seed=int(instance_seed),
            params=adjusted_params,
            gen_defaults=_GEN_DEFAULTS,
            selected_variant=str(selected),
            variant_probabilities=probabilities,
            supported_variants=OPTION_LETTERS,
            balance_flag_key="balanced_direction_option_letter_sampling",
            explicit_key="correct_option_letter",
            weights_key="direction_option_letter_weights",
            sampling_namespace=f"{TASK_ID}.correct_option_letter",
        )
    return str(selected), {str(key): float(value) for key, value in sorted(probabilities.items())}


def _resolve_axes(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    selected_query_id: str,
    query_probabilities: Mapping[str, float],
) -> _ResolvedAxes:
    """Resolve scene, query, and internal sampling axes."""

    scene_variant, scene_probs = _resolve_scene_variant(int(instance_seed), params=params)
    field_orientation, field_probs = _resolve_field_orientation(int(instance_seed), params=params)
    velocity_direction, velocity_probs = _resolve_velocity_direction(
        int(instance_seed),
        params=params,
        internal_query_id=INTERNAL_QUERY_ID,
    )
    charge_sign, charge_probs = _resolve_charge_sign(int(instance_seed), params=params)
    correct_option_letter, option_probs = _resolve_correct_option_letter(
        int(instance_seed),
        params=params,
        internal_query_id=INTERNAL_QUERY_ID,
    )

    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.accent")
    accent_name, accent_probs = resolve_variant(
        rng,
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        supported_variants=SUPPORTED_PHYSICS_COLOR_NAMES,
        explicit_key="accent_color_name",
        weights_key="accent_color_name_weights",
    )
    accent_name = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        selected_variant=str(accent_name),
        variant_probabilities=accent_probs,
        supported_variants=SUPPORTED_PHYSICS_COLOR_NAMES,
        balance_flag_key="balanced_accent_color_name_sampling",
        explicit_key="accent_color_name",
        weights_key="accent_color_name_weights",
        sampling_namespace=f"{TASK_ID}.accent_color_name",
    )
    return _ResolvedAxes(
        scene_variant=str(scene_variant),
        query_id=str(selected_query_id),
        field_orientation=str(field_orientation),
        velocity_direction=velocity_direction,
        charge_sign=int(charge_sign),
        correct_option_letter=correct_option_letter,
        accent_color_name=str(accent_name),
        scene_variant_probabilities={str(key): float(value) for key, value in sorted(scene_probs.items())},
        query_id_probabilities={str(key): float(value) for key, value in sorted(dict(query_probabilities).items())},
        field_orientation_probabilities=dict(field_probs),
        velocity_direction_probabilities=dict(velocity_probs),
        charge_sign_probabilities=dict(charge_probs),
        correct_option_letter_probabilities=dict(option_probs),
        accent_color_name_probabilities={str(key): float(value) for key, value in sorted(accent_probs.items())},
    )


def _direction_from_vector(dx: int, dy: int) -> str:
    """Return the named compass direction for one nonzero integer vector."""

    sx = 0 if int(dx) == 0 else 1 if int(dx) > 0 else -1
    sy = 0 if int(dy) == 0 else 1 if int(dy) > 0 else -1
    for name, vector in DIRECTION_VECTORS.items():
        if vector == (sx, sy):
            return str(name)
    raise ValueError(f"unsupported direction vector: {(dx, dy)}")


def _force_direction(*, velocity_direction: str, field_orientation: str, charge_sign: int) -> str:
    """Return magnetic-force direction using F = q v x B."""

    vx, vy = DIRECTION_VECTORS[str(velocity_direction)]
    bz = 1 if str(field_orientation) == "out_of_page" else -1
    fx = int(charge_sign) * int(vy) * int(bz)
    fy = int(charge_sign) * -int(vx) * int(bz)
    return _direction_from_vector(int(fx), int(fy))


def _direction_options(rng, *, force_direction: str, correct_option_letter: str) -> Dict[str, str]:
    """Assign every option letter to a unique compass direction."""

    remaining = [direction for direction in SUPPORTED_DIRECTIONS if str(direction) != str(force_direction)]
    rng.shuffle(remaining)
    option_directions: Dict[str, str] = {}
    for letter in OPTION_LETTERS:
        if str(letter) == str(correct_option_letter):
            option_directions[str(letter)] = str(force_direction)
        else:
            option_directions[str(letter)] = str(remaining.pop())
    return option_directions


def _sample_scene_spec(rng, *, axes: _ResolvedAxes, params: Mapping[str, Any], instance_seed: int) -> _SceneSpec:
    """Sample one symbolic magnetism scene."""

    if axes.velocity_direction is None or axes.correct_option_letter is None:
        raise ValueError("force_direction_choice requires velocity direction and correct option")
    force_direction = _force_direction(
        velocity_direction=str(axes.velocity_direction),
        field_orientation=str(axes.field_orientation),
        charge_sign=int(axes.charge_sign),
    )
    scenario = _DirectionScenario(
        charge_sign=int(axes.charge_sign),
        field_orientation=str(axes.field_orientation),
        velocity_direction=str(axes.velocity_direction),
        force_direction=str(force_direction),
        option_directions=_direction_options(
            rng,
            force_direction=str(force_direction),
            correct_option_letter=str(axes.correct_option_letter),
        ),
    )
    return _SceneSpec(
        scene_variant=str(axes.scene_variant),
        query_id=str(axes.query_id),
        field_orientation=str(axes.field_orientation),
        direction_scenario=scenario,
        correct_option_letter=str(axes.correct_option_letter),
        target_answer=str(axes.correct_option_letter),
        annotation_entity_ids=("field_orientation_label", "particle", "velocity_vector"),
    )


def _draw_text_tag(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    center: Tuple[float, float],
    font,
    fill_rgb: Tuple[int, int, int],
    outline_rgb: Tuple[int, int, int],
    text_rgb: Tuple[int, int, int],
    stroke_width_px: int = 2,
) -> List[float]:
    """Draw one rounded text tag and return its bbox."""

    text_bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=max(0, int(stroke_width_px)))
    text_width = float(text_bbox[2] - text_bbox[0])
    text_height = float(text_bbox[3] - text_bbox[1])
    pad_x = 12.0
    pad_y = 8.0
    cx, cy = float(center[0]), float(center[1])
    bbox = [
        round(cx - 0.5 * text_width - pad_x, 3),
        round(cy - 0.5 * text_height - pad_y, 3),
        round(cx + 0.5 * text_width + pad_x, 3),
        round(cy + 0.5 * text_height + pad_y, 3),
    ]
    draw_rounded_rect(draw, tuple(float(value) for value in bbox), radius=9, fill=fill_rgb, outline=outline_rgb, width=max(1, int(stroke_width_px)))
    text_draw_bbox = draw_centered_text(
        draw,
        text=str(text),
        center=(cx, cy),
        font=font,
        fill=text_rgb,
        stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(text_rgb)),
        stroke_width=1,
    )
    return _bbox_union(bbox, text_draw_bbox)


def _draw_panel(draw: ImageDraw.ImageDraw, *, bbox: Sequence[float], scene_variant: str, theme, render_defaults: Mapping[str, Any]) -> None:
    """Draw the main magnetism panel."""

    fill_rgb = theme.panel_alt_fill_rgb if str(scene_variant) == "lab_card" else theme.panel_fill_rgb
    draw_rounded_rect(draw, tuple(float(value) for value in bbox), radius=0, fill=fill_rgb, outline=theme.panel_outline_rgb, width=3)
    left, top, right, bottom = [float(value) for value in bbox[:4]]
    if str(scene_variant) == "field_grid":
        spacing = float(render_defaults["field_symbol_spacing_px"])
        x = left + spacing
        while x < right:
            draw.line([(x, top), (x, bottom)], fill=theme.grid_rgb, width=1)
            x += spacing
        y = top + spacing
        while y < bottom:
            draw.line([(left, y), (right, y)], fill=theme.grid_rgb, width=1)
            y += spacing


def _draw_field_symbols(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    field_orientation: str,
    theme,
    render_defaults: Mapping[str, Any],
    scene_entities: List[Dict[str, Any]],
) -> List[List[float]]:
    """Draw repeated B-field symbols and return their bboxes."""

    left, top, right, bottom = [float(value) for value in bbox[:4]]
    spacing = float(render_defaults["field_symbol_spacing_px"])
    radius = 13.0
    bboxes: List[List[float]] = []
    row = 0
    y = top + 0.75 * spacing
    while y < bottom - 0.35 * spacing:
        x = left + (0.70 * spacing if row % 2 == 0 else 1.12 * spacing)
        while x < right - 0.45 * spacing:
            symbol_bbox = [round(x - radius, 3), round(y - radius, 3), round(x + radius, 3), round(y + radius, 3)]
            draw.ellipse(tuple(symbol_bbox), outline=theme.field_symbol_rgb, width=2)
            if str(field_orientation) == "out_of_page":
                draw.ellipse((x - 3.0, y - 3.0, x + 3.0, y + 3.0), fill=theme.field_symbol_rgb)
            else:
                draw.line([(x - 7.0, y - 7.0), (x + 7.0, y + 7.0)], fill=theme.field_symbol_rgb, width=2)
                draw.line([(x - 7.0, y + 7.0), (x + 7.0, y - 7.0)], fill=theme.field_symbol_rgb, width=2)
            scene_entities.append(
                {
                    "entity_id": f"field_symbol_{len(bboxes)}",
                    "entity_type": "magnetic_field_symbol",
                    "bbox": list(symbol_bbox),
                    "meta": {"field_orientation": str(field_orientation)},
                }
            )
            bboxes.append(symbol_bbox)
            x += spacing
        row += 1
        y += spacing
    return bboxes


def _draw_particle(
    draw: ImageDraw.ImageDraw,
    *,
    center: Tuple[float, float],
    charge_sign: int,
    theme,
    render_defaults: Mapping[str, Any],
    particle_font,
    scene_entities: List[Dict[str, Any]],
) -> List[float]:
    """Draw the charged particle and return its bbox."""

    radius = float(render_defaults["particle_radius_px"])
    cx, cy = float(center[0]), float(center[1])
    bbox = [round(cx - radius, 3), round(cy - radius, 3), round(cx + radius, 3), round(cy + radius, 3)]
    draw.ellipse(tuple(float(value) for value in bbox), fill=theme.particle_fill_rgb, outline=theme.particle_outline_rgb, width=3)
    sign_text = "q=+" if int(charge_sign) > 0 else "q=-"
    draw_centered_text(
        draw,
        text=sign_text,
        center=(cx, cy),
        font=particle_font,
        fill=theme.particle_text_rgb,
        stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(theme.particle_text_rgb)),
        stroke_width=1,
    )
    scene_entities.append(
        {
            "entity_id": "particle",
            "entity_type": "charged_particle",
            "bbox": list(bbox),
            "meta": {"charge_sign": int(charge_sign)},
        }
    )
    return bbox


def _draw_vector_arrow(
    draw: ImageDraw.ImageDraw,
    *,
    center: Tuple[float, float],
    direction: str,
    length_px: float,
    color_rgb: Tuple[int, int, int],
    render_defaults: Mapping[str, Any],
    label: str | None,
    label_font,
    label_offset: float = 24.0,
) -> List[float]:
    """Draw one vector arrow from a center point."""

    end = direction_endpoint(center, direction=str(direction), length_px=float(length_px), direction_vectors=DIRECTION_VECTORS)
    bbox = draw_arrow_with_bbox(
        draw,
        start=(float(center[0]), float(center[1])),
        end=end,
        fill=color_rgb,
        width=max(1, int(render_defaults["arrow_width_px"])),
        head_length_px=float(render_defaults["arrow_head_length_px"]),
        head_width_px=float(render_defaults["arrow_head_width_px"]),
        padding_px=18.0,
    )
    if label:
        label_center = direction_endpoint(
            center,
            direction=str(direction),
            length_px=float(length_px) + float(label_offset),
            direction_vectors=DIRECTION_VECTORS,
        )
        text_bbox = draw_centered_text(
            draw,
            text=str(label),
            center=label_center,
            font=label_font,
            fill=color_rgb,
            stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(color_rgb)),
            stroke_width=2,
        )
        bbox = _bbox_union(bbox, text_bbox)
    return bbox


def _draw_option_arrows(
    draw: ImageDraw.ImageDraw,
    *,
    option_directions: Mapping[str, str],
    correct_option_letter: str,
    render_defaults: Mapping[str, Any],
    theme,
    option_font,
    scene_entities: List[Dict[str, Any]],
) -> Dict[str, Dict[str, List[float]]]:
    """Draw the A-H candidate force arrows."""

    left = float(render_defaults["side_left_px"]) + float(render_defaults.get("layout_offset_x_px", 0))
    top = float(render_defaults["side_top_px"]) + float(render_defaults.get("layout_offset_y_px", 0))
    cell_w = float(render_defaults["option_cell_width_px"])
    cell_h = float(render_defaults["option_cell_height_px"])
    gap_x = float(render_defaults["option_cell_gap_x_px"])
    gap_y = float(render_defaults["option_cell_gap_y_px"])
    option_bboxes: Dict[str, List[float]] = {}
    option_cell_bboxes: Dict[str, List[float]] = {}
    option_arrow_bboxes: Dict[str, List[float]] = {}
    for index, letter in enumerate(OPTION_LETTERS):
        col = index % 2
        row = index // 2
        cell_left = left + col * (cell_w + gap_x)
        cell_top = top + row * (cell_h + gap_y)
        cell_bbox = [cell_left, cell_top, cell_left + cell_w, cell_top + cell_h]
        option_cell_bboxes[str(letter)] = [round(float(value), 3) for value in cell_bbox]
        draw_rounded_rect(draw, tuple(cell_bbox), radius=10, fill=(255, 255, 255), outline=theme.option_outline_rgb, width=2)
        label_bbox = draw_centered_text(
            draw,
            text=str(letter),
            center=(cell_left + 21.0, cell_top + 22.0),
            font=option_font,
            fill=theme.label_text_rgb,
            stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(theme.label_text_rgb)),
            stroke_width=1,
        )
        center = (cell_left + 0.56 * cell_w, cell_top + 0.58 * cell_h)
        direction = str(option_directions[str(letter)])
        start, end = centered_arrow_endpoints(
            center,
            direction=direction,
            length_px=float(render_defaults["option_arrow_length_px"]),
            direction_vectors=DIRECTION_VECTORS,
            half_fraction=0.42,
        )
        arrow_bbox = draw_arrow_with_bbox(
            draw,
            start=start,
            end=end,
            fill=theme.option_arrow_rgb,
            width=max(1, int(render_defaults["option_arrow_width_px"])),
            head_length_px=float(render_defaults["option_arrow_head_length_px"]),
            head_width_px=float(render_defaults["option_arrow_head_width_px"]),
            padding_px=14.0,
        )
        option_arrow_bboxes[str(letter)] = list(arrow_bbox)
        option_bbox = _bbox_union(label_bbox, arrow_bbox)
        option_bboxes[str(letter)] = option_bbox
        scene_entities.append(
            {
                "entity_id": f"option_{str(letter)}",
                "entity_type": "candidate_force_arrow",
                "bbox": list(option_bbox),
                "meta": {"option_letter": str(letter), "direction": direction, "is_correct": str(letter) == str(correct_option_letter)},
            }
        )
    return {
        "option_bboxes_px": option_bboxes,
        "option_cell_bboxes_px": option_cell_bboxes,
        "option_arrow_bboxes_px": option_arrow_bboxes,
    }


def _magnetism_content_bbox(*, render_defaults: Mapping[str, Any]) -> List[float]:
    """Return a conservative bbox for the whole magnetic-force diagram before placement."""

    panel_left = float(render_defaults["panel_left_px"])
    panel_top = float(render_defaults["panel_top_px"])
    panel_right = panel_left + float(render_defaults["panel_width_px"])
    panel_bottom = panel_top + float(render_defaults["panel_height_px"])
    option_left = float(render_defaults["side_left_px"])
    option_top = float(render_defaults["side_top_px"])
    option_right = option_left + (2.0 * float(render_defaults["option_cell_width_px"])) + float(render_defaults["option_cell_gap_x_px"])
    option_bottom = option_top + (4.0 * float(render_defaults["option_cell_height_px"])) + (3.0 * float(render_defaults["option_cell_gap_y_px"]))
    return [
        round(float(min(panel_left, option_left)), 3),
        round(float(min(panel_top, option_top)), 3),
        round(float(max(panel_right, option_right)), 3),
        round(float(max(panel_bottom, option_bottom)), 3),
    ]


def _resolve_magnetism_layout_placement(
    *,
    render_defaults: Mapping[str, int],
    params: Mapping[str, Any],
    instance_seed: int,
    canvas_width: int,
    canvas_height: int,
) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """Resolve whole-diagram placement before rendering and annotation projection."""

    content_bbox = _magnetism_content_bbox(render_defaults=render_defaults)
    content_left, content_top, content_right, content_bottom = [float(value) for value in content_bbox]
    jitter = resolve_layout_jitter(
        params,
        _RENDER_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.magnetic_force_layout",
    )
    min_margin = int(jitter.get("min_margin_px", 18))
    requested_dx = int(jitter.get("requested_dx_px", 0))
    requested_dy = int(jitter.get("requested_dy_px", 0))
    min_dx = int(math.ceil(float(min_margin) - float(content_left)))
    max_dx = int(math.floor(float(canvas_width) - float(min_margin) - float(content_right)))
    min_dy = int(math.ceil(float(min_margin) - float(content_top)))
    max_dy = int(math.floor(float(canvas_height) - float(min_margin) - float(content_bottom)))
    if int(min_dx) > int(max_dx):
        min_dx = 0
        max_dx = 0
    if int(min_dy) > int(max_dy):
        min_dy = 0
        max_dy = 0
    if not bool(jitter.get("enabled", False)):
        requested_dx = 0
        requested_dy = 0
    dx = max(int(min_dx), min(int(max_dx), int(requested_dx)))
    dy = max(int(min_dy), min(int(max_dy), int(requested_dy)))
    adjusted = dict(render_defaults)
    adjusted["layout_offset_x_px"] = int(dx)
    adjusted["layout_offset_y_px"] = int(dy)

    content_width = round(float(content_right) - float(content_left), 3)
    content_height = round(float(content_bottom) - float(content_top), 3)
    final_bbox = [
        round(float(content_left) + float(dx), 3),
        round(float(content_top) + float(dy), 3),
        round(float(content_right) + float(dx), 3),
        round(float(content_bottom) + float(dy), 3),
    ]
    placement = dict(jitter)
    placement.update(
        {
            "mode": "whole_magnetic_force_diagram_offset",
            "content_bbox_px": list(content_bbox),
            "content_size_px": [float(content_width), float(content_height)],
            "final_content_bbox_px": list(final_bbox),
            "canvas_size_px": [int(canvas_width), int(canvas_height)],
            "free_space_px": [
                round(float(canvas_width) - float(content_width), 3),
                round(float(canvas_height) - float(content_height), 3),
            ],
            "available_offset_x_px": [int(min_dx), int(max_dx)],
            "available_offset_y_px": [int(min_dy), int(max_dy)],
            "sampled_offset_px": [int(requested_dx), int(requested_dy)],
            "final_offset_px": [int(dx), int(dy)],
            "default_origin_px": [round(float(content_left), 3), round(float(content_top), 3)],
            "final_origin_px": [round(float(content_left) + float(dx), 3), round(float(content_top) + float(dy), 3)],
            "dx_px": int(dx),
            "dy_px": int(dy),
        }
    )
    return adjusted, placement


def _render_scene(
    *,
    background: Image.Image,
    render_defaults: Mapping[str, Any],
    accent_color_name: str,
    scene_spec: _SceneSpec,
    font_family: str,
    diagram_style: Any | None = None,
) -> _RenderedScene:
    """Render the magnetic-force diagram and project annotation boxes."""

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    theme = build_physics_magnetism_theme(str(accent_color_name), diagram_style=diagram_style)
    label_font = load_font(int(render_defaults["label_font_size_px"]), bold=False, font_family=str(font_family))
    symbol_font = load_font(int(render_defaults["symbol_font_size_px"]), bold=True, font_family=str(font_family))
    particle_font = load_font(int(render_defaults["particle_font_size_px"]), bold=True, font_family=str(font_family))
    option_font = load_font(int(render_defaults["option_font_size_px"]), bold=True, font_family=str(font_family))
    dx = float(render_defaults.get("layout_offset_x_px", 0))
    dy = float(render_defaults.get("layout_offset_y_px", 0))
    panel = [
        float(render_defaults["panel_left_px"]) + dx,
        float(render_defaults["panel_top_px"]) + dy,
        float(render_defaults["panel_left_px"]) + dx + float(render_defaults["panel_width_px"]),
        float(render_defaults["panel_top_px"]) + dy + float(render_defaults["panel_height_px"]),
    ]
    _draw_panel(draw, bbox=panel, scene_variant=str(scene_spec.scene_variant), theme=theme, render_defaults=render_defaults)
    scene_entities: List[Dict[str, Any]] = []
    render_map: Dict[str, Any] = {
        "technical_diagram_frame_mode": str(getattr(diagram_style, "frame_mode", "none")),
        "panel_bbox_px": list(panel),
    }
    _draw_field_symbols(
        draw,
        bbox=panel,
        field_orientation=str(scene_spec.field_orientation),
        theme=theme,
        render_defaults=render_defaults,
        scene_entities=scene_entities,
    )
    field_tag = _draw_text_tag(
        draw,
        text="B out of page" if str(scene_spec.field_orientation) == "out_of_page" else "B into page",
        center=(panel[0] + 142.0, panel[1] + 34.0),
        font=label_font,
        fill_rgb=theme.label_fill_rgb,
        outline_rgb=theme.label_outline_rgb,
        text_rgb=theme.label_text_rgb,
    )
    scene_entities.append({"entity_id": "field_orientation_label", "entity_type": "field_label", "bbox": list(field_tag), "meta": {"field_orientation": str(scene_spec.field_orientation)}})

    annotation_ids: List[str] = [str(entity_id) for entity_id in scene_spec.annotation_entity_ids]

    if scene_spec.direction_scenario is None:
        raise ValueError("force direction render requires a direction scenario")
    scenario = scene_spec.direction_scenario
    center = ((panel[0] + panel[2]) / 2.0, (panel[1] + panel[3]) / 2.0)
    velocity_bbox = _draw_vector_arrow(
        draw,
        center=center,
        direction=str(scenario.velocity_direction),
        length_px=float(render_defaults["arrow_length_px"]),
        color_rgb=theme.velocity_arrow_rgb,
        render_defaults=render_defaults,
        label="v",
        label_font=symbol_font,
    )
    scene_entities.append({"entity_id": "velocity_vector", "entity_type": "velocity_arrow", "bbox": list(velocity_bbox), "meta": {"direction": str(scenario.velocity_direction)}})
    particle_bbox = _draw_particle(
        draw,
        center=center,
        charge_sign=int(scenario.charge_sign),
        theme=theme,
        render_defaults=render_defaults,
        particle_font=particle_font,
        scene_entities=scene_entities,
    )
    option_maps = _draw_option_arrows(
        draw,
        option_directions=scenario.option_directions,
        correct_option_letter=str(scene_spec.correct_option_letter),
        render_defaults=render_defaults,
        theme=theme,
        option_font=option_font,
        scene_entities=scene_entities,
    )
    option_bboxes = dict(option_maps["option_bboxes_px"])
    annotation_bbox_map = {
        "field_orientation": list(field_tag),
        "charge": list(particle_bbox),
        "velocity": list(velocity_bbox),
    }
    render_map.update(
        {
            "field_orientation_label_bbox_px": list(field_tag),
            "particle_bbox_px": list(particle_bbox),
            "velocity_vector_bbox_px": list(velocity_bbox),
            "option_bboxes_px": dict(option_bboxes),
            "option_cell_bboxes_px": dict(option_maps["option_cell_bboxes_px"]),
            "option_arrow_bboxes_px": dict(option_maps["option_arrow_bboxes_px"]),
            "correct_option_bbox_px": list(option_bboxes[str(scene_spec.correct_option_letter)]),
            "annotation_bbox_map_px": {str(key): list(bbox) for key, bbox in annotation_bbox_map.items()},
            "annotation_entity_ids": list(annotation_ids),
        }
    )

    return _RenderedScene(
        image=image,
        annotation_bbox_map={str(key): list(bbox) for key, bbox in annotation_bbox_map.items()},
        annotation_entity_ids=list(annotation_ids),
        scene_entities=[dict(entity) for entity in scene_entities],
        render_map=dict(render_map),
    )


def _answer_type(query_id: str) -> str:
    """Return answer type for one query."""

    if str(query_id) != "force_direction_choice":
        raise ValueError(f"unsupported magnetism query id: {query_id}")
    return "option_letter"


def _build_prompt_examples(query_id: str) -> Tuple[str, str]:
    """Build deterministic JSON examples for one query."""

    if str(query_id) != "force_direction_choice":
        raise ValueError(f"unsupported magnetism query id: {query_id}")
    return build_prompt_json_examples(
        annotation_value={
            "field_orientation": [86, 72, 216, 110],
            "charge": [390, 270, 462, 342],
            "velocity": [426, 220, 560, 334],
        },
        answer_type="option_letter",
    )


@register_task
class PhysicsMagneticForceForceDirectionChoiceTask:
    """Choose the magnetic force direction for a moving charged particle."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "physics"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Own sampling, rendering, prompt binding, answer, annotation, and output trace."""

        query_id, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=dict(params or {}),
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SINGLE_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_NAMESPACE}.query",
        )
        axes = _resolve_axes(
            int(instance_seed),
            params=task_params,
            selected_query_id=str(query_id),
            query_probabilities=query_probabilities,
        )
        rendered_scene: _RenderedScene | None = None
        scene_spec: _SceneSpec | None = None

        for attempt_index in range(max(1, int(max_attempts))):
            attempt_rng = spawn_rng(int(instance_seed), f"{TASK_ID}.attempt.{int(attempt_index)}")
            try:
                scene_spec = _sample_scene_spec(attempt_rng, axes=axes, params=task_params, instance_seed=int(instance_seed))
            except ValueError:
                continue
            background, background_meta, diagram_style, diagram_style_meta = prepare_physics_diagram_style_and_background(
                scene_id=SCENE_ID,
                canvas_width=int(task_params.get("canvas_width", group_default(_RENDER_DEFAULTS, "canvas_width", _DEFAULTS.canvas_width))),
                canvas_height=int(task_params.get("canvas_height", group_default(_RENDER_DEFAULTS, "canvas_height", _DEFAULTS.canvas_height))),
                instance_seed=int(instance_seed),
                params=task_params,
            )
            font_family = sample_font_family(
                role="readout",
                instance_seed=int(instance_seed),
                namespace=f"{TASK_ID}.render.font",
                params=task_params,
            )
            font_record = get_font_family_record(str(font_family))
            render_defaults = {
                key: resolve_render_int(
                    task_params,
                    _RENDER_DEFAULTS,
                    key,
                    int(getattr(_DEFAULTS, key)),
                    instance_seed=int(instance_seed),
                    namespace=TASK_ID,
                )
                for key in (
                    "canvas_width",
                    "canvas_height",
                    "panel_left_px",
                    "panel_top_px",
                    "panel_width_px",
                    "panel_height_px",
                    "side_left_px",
                    "side_top_px",
                    "side_width_px",
                    "field_symbol_spacing_px",
                    "particle_radius_px",
                    "arrow_length_px",
                    "arrow_width_px",
                    "arrow_head_length_px",
                    "arrow_head_width_px",
                    "option_cell_width_px",
                    "option_cell_height_px",
                    "option_cell_gap_x_px",
                    "option_cell_gap_y_px",
                    "option_arrow_length_px",
                    "option_arrow_width_px",
                    "option_arrow_head_length_px",
                    "option_arrow_head_width_px",
                    "label_font_size_px",
                    "symbol_font_size_px",
                    "particle_font_size_px",
                    "option_font_size_px",
                    "equation_font_size_px",
                )
            }
            render_defaults, layout_placement_meta = _resolve_magnetism_layout_placement(
                render_defaults=render_defaults,
                params=task_params,
                instance_seed=int(instance_seed),
                canvas_width=int(render_defaults["canvas_width"]),
                canvas_height=int(render_defaults["canvas_height"]),
            )
            rendered_scene = _render_scene(
                background=background,
                render_defaults=render_defaults,
                accent_color_name=str(axes.accent_color_name),
                scene_spec=scene_spec,
                font_family=str(font_family),
                diagram_style=diagram_style,
            )
            image, post_noise_meta = apply_post_image_noise(
                rendered_scene.image,
                instance_seed=int(instance_seed),
                params=task_params,
                default_config=POST_IMAGE_NOISE_DEFAULTS,
            )

            prompt_defaults = required_group_defaults(
                _PROMPT_DEFAULTS,
                (
                    "bundle_id",
                    "task_key",
                ),
                context=f"prompt defaults for {self.task_id}",
            )
            prompt_selection = render_scene_prompt_variants(
                domain=self.domain,
                scene_id=SCENE_ID,
                bundle_id=str(prompt_defaults["bundle_id"]),
                scene_key=SCENE_PROMPT_KEY,
                task_key=str(prompt_defaults["task_key"]),
                query_key=PROMPT_QUERY_KEY,
                answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
                dynamic_slots={},
                instance_seed=int(instance_seed),
            )
            prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)

            answer_type = _answer_type(INTERNAL_QUERY_ID)
            answer_value: int | str = scene_spec.target_answer
            answer_gt = TypedValue(type=str(answer_type), value=answer_value)
            annotation_gt = TypedValue(
                type="bbox_map",
                value={str(key): list(bbox) for key, bbox in rendered_scene.annotation_bbox_map.items()},
            )

            direction_payload: Dict[str, Any] = {}
            if scene_spec.direction_scenario is not None:
                scenario = scene_spec.direction_scenario
                direction_payload = {
                    "charge_sign": int(scenario.charge_sign),
                    "field_orientation": str(scenario.field_orientation),
                    "velocity_direction": str(scenario.velocity_direction),
                    "force_direction": str(scenario.force_direction),
                    "option_directions": dict(scenario.option_directions),
                }

            query_spec = build_prompt_query_spec(
                prompt_artifacts=prompt_artifacts,
                query_id=str(query_id),
                params={
                    "scene_variant": str(axes.scene_variant),
                    "internal_query_id": INTERNAL_QUERY_ID,
                    "field_orientation": str(axes.field_orientation),
                    "velocity_direction": axes.velocity_direction,
                    "charge_sign": int(axes.charge_sign),
                    "correct_option_letter": axes.correct_option_letter,
                    "accent_color_name": str(axes.accent_color_name),
                    "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
                    "query_id_probabilities": dict(axes.query_id_probabilities),
                    "field_orientation_probabilities": dict(axes.field_orientation_probabilities),
                    "velocity_direction_probabilities": dict(axes.velocity_direction_probabilities),
                    "charge_sign_probabilities": dict(axes.charge_sign_probabilities),
                    "correct_option_letter_probabilities": dict(axes.correct_option_letter_probabilities),
                    "accent_color_name_probabilities": dict(axes.accent_color_name_probabilities),
                    "target_answer": answer_value,
                },
            )
            trace_payload = {
                "scene_ir": {
                    "scene_kind": f"physics_magnetic_force_{str(axes.scene_variant)}",
                    "entities": [dict(entity) for entity in rendered_scene.scene_entities],
                    "relations": {
                        "scene_variant": str(axes.scene_variant),
                        "query_id": str(query_id),
                        "internal_query_id": INTERNAL_QUERY_ID,
                        "field_orientation": str(axes.field_orientation),
                        "charge_sign": int(axes.charge_sign),
                        "velocity_direction": axes.velocity_direction,
                        "accent_color_name": str(axes.accent_color_name),
                        "target_answer": answer_value,
                        "answer_type": str(answer_type),
                        "direction_scenario": dict(direction_payload),
                        "annotation_entity_ids": list(rendered_scene.annotation_entity_ids),
                    },
                },
                "query_spec": query_spec,
                "render_spec": {
                    "scene_variant": str(axes.scene_variant),
                    "canvas_width": int(image.size[0]),
                    "canvas_height": int(image.size[1]),
                    "accent_color_name": str(axes.accent_color_name),
                    "font": {
                        "font_family": str(font_family),
                        "font_asset_version": font_asset_version(),
                        "font_asset": font_record.to_trace(),
                        "scope": "magnetic_force_diagram",
                        "selection_policy": {
                            "pool": "global_approved_font_pool",
                            "include_tags": [],
                            "exclude_tags": [],
                            "exclusion_reason": "",
                        },
                    },
                    "technical_diagram_style": dict(diagram_style_meta),
                    "background_style": background_meta,
                    "layout_placement": dict(layout_placement_meta),
                    "post_image_noise": post_noise_meta,
                },
                "render_map": dict(rendered_scene.render_map),
                "execution_trace": {
                    "scene_variant": str(axes.scene_variant),
                    "query_id": str(query_id),
                    "internal_query_id": INTERNAL_QUERY_ID,
                    "field_orientation": str(axes.field_orientation),
                    "velocity_direction": axes.velocity_direction,
                    "charge_sign": int(axes.charge_sign),
                    "accent_color_name": str(axes.accent_color_name),
                    "target_answer": answer_value,
                    "answer_type": str(answer_type),
                    "option_letters": list(OPTION_LETTERS),
                    "direction_scenario": dict(direction_payload),
                    "correct_option_letter": scene_spec.correct_option_letter,
                    "annotation_entity_ids": list(rendered_scene.annotation_entity_ids),
                },
                "witness_symbolic": {
                    "type": "object_map",
                    "ids": [str(item) for item in rendered_scene.annotation_entity_ids],
                    "key_to_entity_id": {
                        "field_orientation": "field_orientation_label",
                        "charge": "particle",
                        "velocity": "velocity_vector",
                    },
                },
                "projected_annotation": {
                    "type": "bbox_map",
                    "bbox_map": {str(key): list(bbox) for key, bbox in rendered_scene.annotation_bbox_map.items()},
                    "pixel_bbox_map": {
                        str(key): list(bbox) for key, bbox in rendered_scene.annotation_bbox_map.items()
                    },
                },
                "background": background_meta,
                "post_image_noise": post_noise_meta,
            }
            return TaskOutput(
                prompt=str(prompt_artifacts.prompt),
                prompt_variants=dict(prompt_artifacts.prompt_variants),
                answer_gt=answer_gt,
                annotation_gt=annotation_gt,
                image=image,
                image_id="img0",
                trace_payload=trace_payload,
                task_versions=default_task_versions(),
                scene_id=SCENE_ID,
                query_id=str(query_id),
            )

        raise RuntimeError(f"{self.task_id} failed to generate a valid scene after {max_attempts} attempts")
__all__ = [
    "PhysicsMagneticForceForceDirectionChoiceTask",
]
