"""Compute pressure difference from a U-tube manometer."""

from __future__ import annotations

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
from ...shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults, required_group_defaults
from ...shared.deterministic_sampling import uniform_probability_map
from ...shared.drawing import draw_arrow, draw_centered_text, draw_dashed_line
from ...shared.fixed_query import select_task_query_id
from ...shared.font_assets import font_asset_version, get_font_family_record, sample_font_family
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_query_spec,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)
from ...shared.support_sampling import resolve_integer_choice, resolve_integer_support
from ...shared.text_rendering import load_font, resolve_text_stroke_fill
from ..shared.diagram_style import prepare_physics_diagram_style_and_background
from ..shared.visual_defaults import load_physics_noise_defaults


TASK_ID = "task_physics__manometer__pressure_difference_value"
TASK_NAMESPACE = "physics_manometer_pressure_difference_value"
SCENE_ID = "manometer"
SCENE_PROMPT_KEY = "manometer_diagram"
TASK_PROMPT_KEY = "pressure_difference_value_query"
INTERNAL_QUERY_ID = "u_tube_pressure_difference"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)
SUPPORTED_HIGHER_PRESSURE_SIDES: Tuple[str, ...] = ("A", "B")

POST_IMAGE_NOISE_DEFAULTS = load_physics_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "physics",
    SCENE_ID,
    task_id=TASK_ID,
)


@dataclass(frozen=True)
class _TaskDefaults:
    """Stable fallback defaults for U-tube manometer scenes."""

    canvas_width: int = 1120
    canvas_height: int = 720
    height_cm_support: Tuple[int, ...] = tuple(range(2, 13))
    kpa_per_cm_support: Tuple[int, ...] = (1, 2, 3, 4, 5)
    px_per_cm: int = 17
    panel_margin_x_px: int = 54
    panel_margin_top_px: int = 52
    panel_margin_bottom_px: int = 58
    tube_top_px: int = 188
    tube_bottom_px: int = 560
    tube_width_px: int = 86
    tube_gap_px: int = 290
    high_level_y_px: int = 258
    label_font_size_px: int = 28
    small_font_size_px: int = 22
    title_font_size_px: int = 30
    label_stroke_width_px: int = 2


@dataclass(frozen=True)
class _ManometerScenario:
    """Symbolic manometer state."""

    height_cm: int
    kpa_per_cm: int
    pressure_difference_kpa: int
    higher_pressure_side: str
    height_probabilities: Dict[str, float]
    kpa_per_cm_probabilities: Dict[str, float]
    target_answer_probabilities: Dict[str, float]
    higher_pressure_side_probabilities: Dict[str, float]


@dataclass(frozen=True)
class _RenderedScene:
    """Rendered manometer scene plus prompt-facing annotation metadata."""

    image: Image.Image
    annotation_bbox_map: Dict[str, List[float]]
    scene_entities: List[Dict[str, Any]]
    render_map: Dict[str, Any]


_DEFAULTS = _TaskDefaults()


def _bbox(values: Sequence[float]) -> List[float]:
    return [round(float(value), 3) for value in values]


def _integer_support(params: Mapping[str, Any], key: str, fallback: Sequence[int]) -> Tuple[int, ...]:
    return resolve_integer_support(
        params,
        gen_defaults=_GEN_DEFAULTS,
        key=str(key),
        fallback=fallback,
    )


def _feasible_scenarios(params: Mapping[str, Any]) -> Tuple[Tuple[int, int, int], ...]:
    heights = _integer_support(params, "height_cm_support", _DEFAULTS.height_cm_support)
    conversions = _integer_support(params, "kpa_per_cm_support", _DEFAULTS.kpa_per_cm_support)
    return tuple((int(height), int(conversion), int(height * conversion)) for height in heights for conversion in conversions)


def _resolve_explicit_int(
    params: Mapping[str, Any],
    *,
    key: str,
    support: Sequence[int],
) -> int | None:
    explicit = params.get(str(key))
    if explicit is None:
        return None
    selected = int(explicit)
    if selected not in set(int(value) for value in support):
        raise ValueError(f"unsupported {key}: {selected}")
    return int(selected)


def _resolve_higher_pressure_side(instance_seed: int, params: Mapping[str, Any]) -> Tuple[str, Dict[str, float]]:
    explicit = params.get("higher_pressure_side")
    if explicit is not None:
        selected = str(explicit).upper()
        if selected not in SUPPORTED_HIGHER_PRESSURE_SIDES:
            raise ValueError(f"unsupported higher_pressure_side: {explicit}")
        return selected, {side: (1.0 if side == selected else 0.0) for side in SUPPORTED_HIGHER_PRESSURE_SIDES}
    selected = str(spawn_rng(int(instance_seed), f"{TASK_NAMESPACE}.higher_pressure_side").choice(SUPPORTED_HIGHER_PRESSURE_SIDES))
    probability = 1.0 / float(len(SUPPORTED_HIGHER_PRESSURE_SIDES))
    return str(selected), {side: float(probability) for side in SUPPORTED_HIGHER_PRESSURE_SIDES}


def _resolve_scenario(instance_seed: int, params: Mapping[str, Any]) -> _ManometerScenario:
    """Resolve operands while preserving answer-balanced manometer sampling invariants."""

    height_support = _integer_support(params, "height_cm_support", _DEFAULTS.height_cm_support)
    conversion_support = _integer_support(params, "kpa_per_cm_support", _DEFAULTS.kpa_per_cm_support)
    explicit_height = _resolve_explicit_int(params, key="height_cm", support=height_support)
    explicit_conversion = _resolve_explicit_int(params, key="kpa_per_cm", support=conversion_support)
    explicit_answer = params.get("target_answer", params.get("pressure_difference_kpa"))

    if explicit_height is not None and explicit_conversion is not None:
        height_cm = int(explicit_height)
        kpa_per_cm = int(explicit_conversion)
        if explicit_answer is not None and int(explicit_answer) != int(height_cm * kpa_per_cm):
            raise ValueError(
                f"target_answer {int(explicit_answer)} does not match height_cm={height_cm} "
                f"and kpa_per_cm={kpa_per_cm}"
            )
    elif explicit_answer is not None:
        target_answer = int(explicit_answer)
        candidates = [
            (height, conversion)
            for height, conversion, answer in _feasible_scenarios(params)
            if int(answer) == target_answer
            and (explicit_height is None or int(height) == int(explicit_height))
            and (explicit_conversion is None or int(conversion) == int(explicit_conversion))
        ]
        if not candidates:
            raise ValueError(f"target_answer {target_answer} is not feasible for manometer supports")
        rng = spawn_rng(int(instance_seed), f"{TASK_NAMESPACE}.target_answer_pair")
        height_cm, kpa_per_cm = rng.choice(candidates)
    elif explicit_height is not None:
        height_cm = int(explicit_height)
        kpa_per_cm, _ = resolve_integer_choice(
            instance_seed=int(instance_seed),
            params=params,
            gen_defaults=_GEN_DEFAULTS,
            support_key="kpa_per_cm_support",
            explicit_key="kpa_per_cm",
            fallback_support=_DEFAULTS.kpa_per_cm_support,
            namespace=f"{TASK_NAMESPACE}.kpa_per_cm",
            balanced_flag_key="balanced_kpa_per_cm_sampling",
        )
    elif explicit_conversion is not None:
        kpa_per_cm = int(explicit_conversion)
        height_cm, _ = resolve_integer_choice(
            instance_seed=int(instance_seed),
            params=params,
            gen_defaults=_GEN_DEFAULTS,
            support_key="height_cm_support",
            explicit_key="height_cm",
            fallback_support=_DEFAULTS.height_cm_support,
            namespace=f"{TASK_NAMESPACE}.height_cm",
            balanced_flag_key="balanced_height_sampling",
        )
    else:
        answer_support = sorted({int(answer) for _, _, answer in _feasible_scenarios(params)})
        if bool(params.get("balanced_target_answer_sampling", group_default(_GEN_DEFAULTS, "balanced_target_answer_sampling", True))):
            rng = spawn_rng(int(instance_seed), f"{TASK_NAMESPACE}.target_answer")
            target_answer = int(rng.choice(answer_support))
        else:
            rng = spawn_rng(int(instance_seed), f"{TASK_NAMESPACE}.target_answer")
            target_answer = int(answer_support[int(rng.randrange(len(answer_support)))])
        candidates = [(height, conversion) for height, conversion, answer in _feasible_scenarios(params) if int(answer) == target_answer]
        rng = spawn_rng(int(instance_seed), f"{TASK_NAMESPACE}.target_answer_pair")
        height_cm, kpa_per_cm = rng.choice(candidates)

    pressure_difference_kpa = int(height_cm) * int(kpa_per_cm)
    answer_support = sorted({int(answer) for _, _, answer in _feasible_scenarios(params)})
    higher_pressure_side, side_probabilities = _resolve_higher_pressure_side(int(instance_seed), params)
    return _ManometerScenario(
        height_cm=int(height_cm),
        kpa_per_cm=int(kpa_per_cm),
        pressure_difference_kpa=int(pressure_difference_kpa),
        higher_pressure_side=str(higher_pressure_side),
        height_probabilities=uniform_probability_map(height_support, selected=explicit_height),
        kpa_per_cm_probabilities=uniform_probability_map(conversion_support, selected=explicit_conversion),
        target_answer_probabilities=uniform_probability_map(answer_support, selected=int(explicit_answer) if explicit_answer is not None else None),
        higher_pressure_side_probabilities=side_probabilities,
    )


def _draw_pressure_chamber(
    draw: ImageDraw.ImageDraw,
    *,
    center_x: float,
    center_y: float,
    label: str,
    font: Any,
    style: Any,
) -> List[float]:
    chamber = (
        float(center_x - 76),
        float(center_y - 30),
        float(center_x + 76),
        float(center_y + 30),
    )
    draw.rounded_rectangle(
        chamber,
        radius=16,
        fill=tuple(int(v) for v in style.panel_alt_fill_rgb),
        outline=tuple(int(v) for v in style.panel_border_rgb),
        width=3,
    )
    draw.ellipse(
        (float(center_x - 19), float(center_y - 19), float(center_x + 19), float(center_y + 19)),
        fill=tuple(int(v) for v in style.label_fill_rgb),
        outline=tuple(int(v) for v in style.stroke_rgb),
        width=3,
    )
    label_bbox = draw_centered_text(
        draw,
        text=str(label),
        center=(float(center_x), float(center_y)),
        font=font,
        fill=tuple(int(v) for v in style.label_rgb),
        stroke_fill=resolve_text_stroke_fill(tuple(int(v) for v in style.label_rgb)),
        stroke_width=1,
    )
    return _bbox(_bbox_union(_bbox(chamber), label_bbox))


def _draw_tube_outline(
    draw: ImageDraw.ImageDraw,
    *,
    left_x: float,
    right_x: float,
    tube_top: float,
    tube_bottom: float,
    tube_width: float,
    style: Any,
    fill_glass: bool = True,
) -> None:
    """Draw a continuous U-tube instead of three disconnected rounded boxes."""

    outline = tuple(int(v) for v in style.stroke_rgb)
    glass = (238, 248, 251)
    left_center_x = float(left_x + tube_width * 0.5)
    right_center_x = float(right_x + tube_width * 0.5)
    bottom_center_y = float(tube_bottom - tube_width * 0.45)
    centerline = [
        (left_center_x, float(tube_top)),
        (left_center_x, bottom_center_y),
        (right_center_x, bottom_center_y),
        (right_center_x, float(tube_top)),
    ]
    outer_width = max(1, int(round(float(tube_width) + 10.0)))
    inner_width = max(1, int(round(float(tube_width))))
    draw.line(centerline, fill=outline, width=outer_width, joint="curve")
    if bool(fill_glass):
        draw.line(centerline, fill=glass, width=inner_width, joint="curve")


def _render_manometer(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    scenario: _ManometerScenario,
) -> _RenderedScene:
    """Render the U-tube diagram and project keyed annotation boxes."""

    canvas_width = int(_RENDER_DEFAULTS.get("canvas_width", _DEFAULTS.canvas_width))
    canvas_height = int(_RENDER_DEFAULTS.get("canvas_height", _DEFAULTS.canvas_height))
    background, background_meta, diagram_style, diagram_style_meta = prepare_physics_diagram_style_and_background(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=SCENE_ID,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        require_grid=True,
    )
    draw = ImageDraw.Draw(background)
    rng = spawn_rng(int(instance_seed), f"{TASK_NAMESPACE}.render")
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{TASK_NAMESPACE}.font",
        params=params,
    )

    label_font = load_font(int(_RENDER_DEFAULTS.get("label_font_size_px", _DEFAULTS.label_font_size_px)), bold=True, font_family=str(font_family))
    small_font = load_font(int(_RENDER_DEFAULTS.get("small_font_size_px", _DEFAULTS.small_font_size_px)), bold=True, font_family=str(font_family))
    stroke_rgb = tuple(int(v) for v in diagram_style.stroke_rgb)
    guide_rgb = tuple(int(v) for v in diagram_style.guide_rgb)
    label_rgb = tuple(int(v) for v in diagram_style.label_rgb)

    panel = (
        float(_RENDER_DEFAULTS.get("panel_margin_x_px", _DEFAULTS.panel_margin_x_px)),
        float(_RENDER_DEFAULTS.get("panel_margin_top_px", _DEFAULTS.panel_margin_top_px)),
        float(canvas_width - _RENDER_DEFAULTS.get("panel_margin_x_px", _DEFAULTS.panel_margin_x_px)),
        float(canvas_height - _RENDER_DEFAULTS.get("panel_margin_bottom_px", _DEFAULTS.panel_margin_bottom_px)),
    )
    draw.rounded_rectangle(
        panel,
        radius=18,
        fill=tuple(int(v) for v in diagram_style.panel_fill_rgb),
        outline=tuple(int(v) for v in diagram_style.panel_border_rgb),
        width=3,
    )
    tube_width = float(_RENDER_DEFAULTS.get("tube_width_px", _DEFAULTS.tube_width_px))
    tube_gap = float(_RENDER_DEFAULTS.get("tube_gap_px", _DEFAULTS.tube_gap_px))
    tube_top = float(_RENDER_DEFAULTS.get("tube_top_px", _DEFAULTS.tube_top_px)) + float(rng.randint(-8, 8))
    tube_bottom = float(_RENDER_DEFAULTS.get("tube_bottom_px", _DEFAULTS.tube_bottom_px)) + float(rng.randint(-6, 6))
    left_x = float(canvas_width * 0.5 - tube_gap * 0.5 - tube_width) + float(rng.randint(-14, 14))
    right_x = float(canvas_width * 0.5 + tube_gap * 0.5) + float(rng.randint(-14, 14))
    px_per_cm = float(_RENDER_DEFAULTS.get("px_per_cm", _DEFAULTS.px_per_cm))
    high_level_y = float(_RENDER_DEFAULTS.get("high_level_y_px", _DEFAULTS.high_level_y_px)) + float(rng.randint(-10, 10))
    diff_px = float(int(scenario.height_cm) * px_per_cm)
    low_level_y = float(high_level_y + diff_px)
    if str(scenario.higher_pressure_side) == "A":
        left_level_y = float(low_level_y)
        right_level_y = float(high_level_y)
    else:
        left_level_y = float(high_level_y)
        right_level_y = float(low_level_y)

    fluid_palette = (
        (78, 154, 214),
        (63, 174, 145),
        (126, 149, 217),
        (211, 142, 66),
        (154, 120, 190),
    )
    fluid_rgb = spawn_rng(int(instance_seed), f"{TASK_NAMESPACE}.fluid_color").choice(fluid_palette)
    _draw_tube_outline(
        draw,
        left_x=left_x,
        right_x=right_x,
        tube_top=tube_top,
        tube_bottom=tube_bottom,
        tube_width=tube_width,
        style=diagram_style,
    )
    inner_margin = 12.0
    fluid_width = max(1, int(round(float(tube_width - 2.0 * inner_margin))))
    left_center_x = float(left_x + tube_width * 0.5)
    right_center_x = float(right_x + tube_width * 0.5)
    tube_bottom_center_y = float(tube_bottom - tube_width * 0.45)
    fluid_path = [
        (left_center_x, float(left_level_y)),
        (left_center_x, tube_bottom_center_y),
        (right_center_x, tube_bottom_center_y),
        (right_center_x, float(right_level_y)),
    ]
    draw.line(fluid_path, fill=tuple(int(v) for v in fluid_rgb), width=fluid_width, joint="curve")
    meniscus_rgb = tuple(max(0, int(v) - 45) for v in fluid_rgb)
    for x, level_y in ((left_x, left_level_y), (right_x, right_level_y)):
        center_x = float(x + tube_width * 0.5)
        meniscus = (
            float(center_x - fluid_width * 0.5),
            float(level_y - 8),
            float(center_x + fluid_width * 0.5),
            float(level_y + 8),
        )
        draw.arc(meniscus, 0, 180, fill=meniscus_rgb, width=4)
        draw.line(
            (float(center_x - fluid_width * 0.5 + 2), float(level_y), float(center_x + fluid_width * 0.5 - 2), float(level_y)),
            fill=meniscus_rgb,
            width=4,
        )

    chamber_y = float(tube_top - 72)
    draw.line((left_center_x, chamber_y + 30, left_center_x, tube_top + 2), fill=stroke_rgb, width=5)
    draw.line((right_center_x, chamber_y + 30, right_center_x, tube_top + 2), fill=stroke_rgb, width=5)
    left_pressure_bbox = _draw_pressure_chamber(
        draw,
        center_x=left_center_x,
        center_y=chamber_y,
        label="A",
        font=label_font,
        style=diagram_style,
    )
    right_pressure_bbox = _draw_pressure_chamber(
        draw,
        center_x=right_center_x,
        center_y=chamber_y,
        label="B",
        font=label_font,
        style=diagram_style,
    )

    top_level_y = float(min(left_level_y, right_level_y))
    bottom_level_y = float(max(left_level_y, right_level_y))
    bracket_x = float((left_center_x + right_center_x) * 0.5)
    draw_dashed_line(
        draw,
        start=(float(left_center_x + 42), float(left_level_y)),
        end=(float(bracket_x - 18), float(left_level_y)),
        fill=guide_rgb,
        width=2,
        dash_px=8,
        gap_px=6,
    )
    draw_dashed_line(
        draw,
        start=(float(right_center_x - 42), float(right_level_y)),
        end=(float(bracket_x + 18), float(right_level_y)),
        fill=guide_rgb,
        width=2,
        dash_px=8,
        gap_px=6,
    )
    midpoint_y = float((top_level_y + bottom_level_y) * 0.5)
    draw_arrow(
        draw,
        start=(bracket_x, midpoint_y),
        end=(bracket_x, top_level_y),
        fill=stroke_rgb,
        width=4,
        head_length_px=14,
        head_width_px=16,
    )
    draw_arrow(
        draw,
        start=(bracket_x, midpoint_y),
        end=(bracket_x, bottom_level_y),
        fill=stroke_rgb,
        width=4,
        head_length_px=14,
        head_width_px=16,
    )
    height_label_bbox = draw_centered_text(
        draw,
        text=f"h = {int(scenario.height_cm)} cm",
        center=(float(bracket_x + 72), float(midpoint_y)),
        font=small_font,
        fill=label_rgb,
        stroke_fill=resolve_text_stroke_fill(label_rgb),
        stroke_width=1,
    )
    height_bbox = _bbox(
        _bbox_union(
            _bbox((bracket_x - 28, top_level_y - 18, bracket_x + 28, bottom_level_y + 18)),
            height_label_bbox,
        )
    )

    conversion_text = f"rho g = {int(scenario.kpa_per_cm)} kPa/cm"
    conversion_box = (float(canvas_width * 0.5 - 160), float(panel[3] - 72), float(canvas_width * 0.5 + 160), float(panel[3] - 28))
    draw.rounded_rectangle(
        conversion_box,
        radius=12,
        fill=tuple(int(v) for v in diagram_style.label_fill_rgb),
        outline=tuple(int(v) for v in diagram_style.label_border_rgb),
        width=3,
    )
    conversion_text_bbox = draw_centered_text(
        draw,
        text=conversion_text,
        center=(float(canvas_width * 0.5), float((conversion_box[1] + conversion_box[3]) * 0.5)),
        font=small_font,
        fill=label_rgb,
        stroke_fill=resolve_text_stroke_fill(label_rgb),
        stroke_width=1,
    )
    conversion_bbox = _bbox(_bbox_union(_bbox(conversion_box), conversion_text_bbox))

    image, post_noise_meta = apply_post_image_noise(
        background,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    annotation_map = {
        "height_difference": height_bbox,
        "fluid_density_label": conversion_bbox,
    }
    scene_entities = [
        {"id": "left_pressure_point", "label": "A", "bbox_px": left_pressure_bbox},
        {"id": "right_pressure_point", "label": "B", "bbox_px": right_pressure_bbox},
        {"id": "height_difference", "height_cm": int(scenario.height_cm), "bbox_px": height_bbox},
        {"id": "fluid_density_label", "kpa_per_cm": int(scenario.kpa_per_cm), "bbox_px": conversion_bbox},
    ]
    render_map = {
        "height_cm": int(scenario.height_cm),
        "kpa_per_cm": int(scenario.kpa_per_cm),
        "pressure_difference_kpa": int(scenario.pressure_difference_kpa),
        "higher_pressure_side": str(scenario.higher_pressure_side),
        "left_level_y_px": round(float(left_level_y), 3),
        "right_level_y_px": round(float(right_level_y), 3),
        "fluid_rgb": list(int(v) for v in fluid_rgb),
        "technical_diagram_style": dict(diagram_style_meta),
        "background_style": background_meta,
        "post_image_noise": post_noise_meta,
        "annotation_bbox_map_px": {str(key): list(value) for key, value in annotation_map.items()},
    }
    return _RenderedScene(
        image=image,
        annotation_bbox_map=annotation_map,
        scene_entities=scene_entities,
        render_map=render_map,
    )


@register_task
class PhysicsManometerPressureDifferenceValueTask:
    """Compute the absolute pressure difference from a U-tube manometer."""

    domain = "physics"
    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Own sampling, rendering, prompt binding, answer, annotation, and output trace."""

        _ = int(max_attempts)
        query_id, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=dict(params or {}),
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SINGLE_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_NAMESPACE}.query",
        )

        scenario = _resolve_scenario(int(instance_seed), task_params)
        rendered = _render_manometer(
            instance_seed=int(instance_seed),
            params=task_params,
            scenario=scenario,
        )
        prompt_defaults = required_group_defaults(
            _PROMPT_DEFAULTS,
            (
                "bundle_id",
                "task_key",
            ),
            context=f"prompt defaults for {TASK_ID}",
        )
        answer_gt = TypedValue(type="integer", value=int(scenario.pressure_difference_kpa))
        annotation_gt = TypedValue(type="bbox_map", value={str(k): list(v) for k, v in rendered.annotation_bbox_map.items()})
        prompt_selection = render_scene_prompt_variants(
            domain=self.domain,
            scene_id=SCENE_ID,
            bundle_id=str(prompt_defaults["bundle_id"]),
            scene_key=SCENE_PROMPT_KEY,
            task_key=str(prompt_defaults["task_key"]),
            query_key=str(query_id),
            dynamic_slots={},
            instance_seed=int(instance_seed),
            answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        )
        prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)
        font_family = sample_font_family(
            role="readout",
            instance_seed=int(instance_seed),
            namespace=f"{TASK_NAMESPACE}.font",
            params=task_params,
        )
        font_record = get_font_family_record(str(font_family))
        trace_payload = {
            "scene_ir": {
                "scene_kind": "physics_manometer_u_tube",
                "entities": list(rendered.scene_entities),
                "relations": {
                    "query_id": str(query_id),
                    "internal_query_id": INTERNAL_QUERY_ID,
                    "height_cm": int(scenario.height_cm),
                    "kpa_per_cm": int(scenario.kpa_per_cm),
                    "higher_pressure_side": str(scenario.higher_pressure_side),
                    "pressure_difference_kpa": int(scenario.pressure_difference_kpa),
                },
            },
            "query_spec": build_prompt_query_spec(
                prompt_artifacts=prompt_artifacts,
                query_id=str(query_id),
                params={
                    "internal_query_id": INTERNAL_QUERY_ID,
                    "height_cm": int(scenario.height_cm),
                    "kpa_per_cm": int(scenario.kpa_per_cm),
                    "higher_pressure_side": str(scenario.higher_pressure_side),
                    "target_answer": int(scenario.pressure_difference_kpa),
                    "query_id_probabilities": dict(query_probabilities),
                    "height_cm_probabilities": dict(scenario.height_probabilities),
                    "kpa_per_cm_probabilities": dict(scenario.kpa_per_cm_probabilities),
                    "target_answer_probabilities": dict(scenario.target_answer_probabilities),
                    "higher_pressure_side_probabilities": dict(scenario.higher_pressure_side_probabilities),
                },
            ),
            "render_spec": {
                "canvas_width": int(rendered.image.size[0]),
                "canvas_height": int(rendered.image.size[1]),
                "font": {
                    "font_family": str(font_family),
                    "font_asset_version": font_asset_version(),
                    "font_asset": font_record.to_trace(),
                    "scope": "manometer_diagram",
                },
                "technical_diagram_style": dict(rendered.render_map["technical_diagram_style"]),
                "background_style": dict(rendered.render_map["background_style"]),
                "post_image_noise": dict(rendered.render_map["post_image_noise"]),
            },
            "render_map": dict(rendered.render_map),
            "execution_trace": {
                "query_id": str(query_id),
                "internal_query_id": INTERNAL_QUERY_ID,
                "height_cm": int(scenario.height_cm),
                "kpa_per_cm": int(scenario.kpa_per_cm),
                "higher_pressure_side": str(scenario.higher_pressure_side),
                "target_answer": int(scenario.pressure_difference_kpa),
                "annotation_entity_ids": sorted(annotation_gt.value.keys()),
            },
            "witness_symbolic": {
                "type": "object_map",
                "ids": sorted(annotation_gt.value.keys()),
            },
            "projected_annotation": {
                "type": "bbox_map",
                "bbox_map": dict(annotation_gt.value),
                "pixel_bbox_map": dict(annotation_gt.value),
            },
            "background": dict(rendered.render_map["background_style"]),
            "post_image_noise": dict(rendered.render_map["post_image_noise"]),
        }
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=rendered.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(query_id),
        )


__all__ = [
    "PhysicsManometerPressureDifferenceValueTask",
]
