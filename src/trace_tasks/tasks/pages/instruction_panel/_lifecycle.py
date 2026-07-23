"""Scene-private lifecycle for instruction-panel page tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ....core.scene_config import get_scene_defaults
from ....core.types import TypedValue
from ....core.visual.noise import apply_post_image_noise
from ...base import TaskOutput
from ...shared.config_defaults import group_default, required_group_defaults, split_generation_rendering_prompt_defaults
from ...shared.deterministic_sampling import resolve_selection_index
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_query_spec,
    build_prompt_trace_artifacts,
    render_task_prompt_variants,
)
from ...shared.text_legibility import draw_text_traced
from ...shared.text_rendering import fit_font_to_box, load_font
from ...shared.visual_style.information_scene import InformationSceneStyle, make_information_scene_background
from ..shared.information_style import resolve_pages_information_style
from ..shared.page_text_resources import page_text_resource_metadata, sample_page_context_batch, sample_page_label_batch
from ..shared.sampling import (
    resolve_int_support as resolve_pages_int_support,
    resolve_named_axis as resolve_pages_named_axis,
    resolve_supported_int as resolve_pages_supported_int,
)
from ..shared.visual_defaults import load_pages_noise_defaults


SCENE = "instruction_panel"
TASK_NAMESPACE = "pages.instruction_panel"
PROMPT_BUNDLE = "pages_instruction_panel_v1"
PROMPT_SCENE_KEY = "instruction_panel"
PROMPT_TASK_KEY = "instruction_panel_query"


def _prompt_key(*parts: str) -> str:
    return "_".join(str(part) for part in parts)


_SHARED_CONTROL_PROMPT_KEY = _prompt_key("shared", "control", "for", "step", "set", "label")
_CONTROL_PAIR_PROMPT_KEY = _prompt_key("step", "for", "control", "pair", "label")
SUPPORTED_PROMPT_QUERY_KEYS: Tuple[str, ...] = (
    _SHARED_CONTROL_PROMPT_KEY,
    _CONTROL_PAIR_PROMPT_KEY,
)
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)
SCENE_VARIANTS: Tuple[str, ...] = ("manual_cards", "checklist_table", "side_legend_sheet")

_CONTROL_DEFINITIONS: Tuple[Tuple[str, str], ...] = (
    ("save", "Save"),
    ("print", "Print"),
    ("upload", "Upload"),
    ("sync", "Sync"),
    ("search", "Search"),
    ("share", "Share"),
    ("settings", "Settings"),
    ("alert", "Alert"),
    ("download", "Download"),
    ("lock", "Lock"),
    ("filter", "Filter"),
    ("review", "Review"),
)
_CONTROL_ACCENTS: Tuple[Tuple[int, int, int], ...] = (
    (50, 111, 171),
    (37, 137, 112),
    (188, 88, 74),
    (123, 101, 181),
    (194, 136, 53),
    (71, 132, 151),
    (160, 87, 130),
    (83, 126, 91),
    (47, 96, 143),
    (141, 103, 72),
    (68, 129, 97),
    (162, 92, 73),
)
_SCENE_LOAD_BY_VARIANT: Dict[str, float] = {
    "manual_cards": 0.45,
    "checklist_table": 0.54,
    "side_legend_sheet": 0.60,
}


@dataclass(frozen=True)
class _RenderParams:
    canvas_width: int
    canvas_height: int
    outer_margin_px: int
    header_height_px: int
    gap_px: int
    corner_radius_px: int
    outline_width_px: int
    title_font_size_px: int
    subtitle_font_size_px: int
    step_title_font_size_px: int
    step_detail_font_size_px: int
    control_font_size_px: int
    number_badge_size_px: int
    control_chip_height_px: int


@dataclass(frozen=True)
class _Control:
    control_id: str
    label: str
    accent_rgb: Tuple[int, int, int]


@dataclass(frozen=True)
class _Step:
    step_id: str
    order_index: int
    step_number: int
    title: str
    detail: str
    controls: Tuple[_Control, ...]


@dataclass(frozen=True)
class _InstructionPanelSpec:
    title: str
    subtitle: str
    controls: Tuple[_Control, ...]
    steps: Tuple[_Step, ...]
    text_resource_metadata: Dict[str, Any]


@dataclass(frozen=True)
class _RenderedInstructionPanel:
    image: Image.Image
    entities: List[Dict[str, Any]]
    panel_bbox_px: List[float]
    title_bbox_px: List[float]
    step_card_bboxes_px: Dict[str, List[float]]
    step_number_bboxes_px: Dict[str, List[float]]
    step_title_bboxes_px: Dict[str, List[float]]
    control_chip_bboxes_px: Dict[str, Dict[str, List[float]]]
    control_label_bboxes_px: Dict[str, Dict[str, List[float]]]
    legend_control_bboxes_px: Dict[str, List[float]]
    layout_meta: Dict[str, Any]


_SCENE_DEFAULTS = get_scene_defaults("pages", SCENE)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
)
POST_IMAGE_NOISE_DEFAULTS = load_pages_noise_defaults(scene_id=SCENE, apply_prob=0.0)


def _resolve_named_variant(
    *,
    task_id: str,
    gen_defaults: Mapping[str, Any],
    params: Mapping[str, Any],
    instance_seed: int,
    supported: Sequence[str],
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    return resolve_pages_named_axis(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        namespace_root=task_id,
        supported=supported,
        explicit_key=explicit_key,
        weights_key=weights_key,
        balance_flag_key=balance_flag_key,
        namespace=namespace,
    )


def _resolve_int_support(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    key: str,
    fallback: Sequence[int],
) -> Tuple[int, ...]:
    return resolve_pages_int_support(params=params, gen_defaults=gen_defaults, key=key, fallback=fallback)


def _resolve_supported_int(
    *,
    task_id: str,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    explicit_key: str,
    support_key: str,
    fallback: Sequence[int],
    instance_seed: int,
    namespace: str,
) -> Tuple[int, Tuple[int, ...], Dict[str, float]]:
    return resolve_pages_supported_int(
        params=params,
        gen_defaults=gen_defaults,
        namespace_root=task_id,
        explicit_key=explicit_key,
        support_key=support_key,
        fallback=fallback,
        instance_seed=int(instance_seed),
        namespace=namespace,
    )


def _resolve_render_params(params: Mapping[str, Any], render_defaults: Mapping[str, Any]) -> _RenderParams:
    def _int_value(key: str, fallback: int, *, minimum: int = 1) -> int:
        return max(int(minimum), int(params.get(key, group_default(render_defaults, key, fallback))))

    return _RenderParams(
        canvas_width=_int_value("canvas_width", 1100, minimum=360),
        canvas_height=_int_value("canvas_height", 880, minimum=360),
        outer_margin_px=_int_value("outer_margin_px", 34, minimum=0),
        header_height_px=_int_value("header_height_px", 92, minimum=54),
        gap_px=_int_value("gap_px", 14, minimum=4),
        corner_radius_px=_int_value("corner_radius_px", 12, minimum=0),
        outline_width_px=_int_value("outline_width_px", 2, minimum=1),
        title_font_size_px=_int_value("title_font_size_px", 30, minimum=14),
        subtitle_font_size_px=_int_value("subtitle_font_size_px", 16, minimum=10),
        step_title_font_size_px=_int_value("step_title_font_size_px", 20, minimum=12),
        step_detail_font_size_px=_int_value("step_detail_font_size_px", 13, minimum=9),
        control_font_size_px=_int_value("control_font_size_px", 14, minimum=9),
        number_badge_size_px=_int_value("number_badge_size_px", 34, minimum=22),
        control_chip_height_px=_int_value("control_chip_height_px", 30, minimum=22),
    )


def _text_bbox(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float],
    text: str,
    font: Any,
    *,
    stroke_width: int = 0,
) -> List[float]:
    try:
        bbox = draw.textbbox((float(xy[0]), float(xy[1])), str(text), font=font, stroke_width=int(stroke_width))
        return [float(value) for value in bbox]
    except Exception:
        width, height = draw.textsize(str(text), font=font)
        return [float(xy[0]), float(xy[1]), float(xy[0]) + float(width), float(xy[1]) + float(height)]


def _draw_text(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float],
    text: str,
    font: Any,
    fill: Tuple[int, int, int],
    *,
    role: str = "readout",
) -> List[float]:
    draw_text_traced(
        draw,
        (float(xy[0]), float(xy[1])),
        str(text),
        fill=fill,
        font=font,
        role=str(role),
        required=False,
    )
    return _text_bbox(draw, xy, str(text), font)


def _blend_rgb(color_a: Sequence[int], color_b: Sequence[int], weight_b: float) -> Tuple[int, int, int]:
    weight = max(0.0, min(1.0, float(weight_b)))
    return tuple(
        int(round((float(color_a[index]) * (1.0 - weight)) + (float(color_b[index]) * weight)))
        for index in range(3)
    )


def _control_pool(*, count: int, instance_seed: int) -> Tuple[_Control, ...]:
    if int(count) > len(_CONTROL_DEFINITIONS):
        raise ValueError(f"control_count cannot exceed {len(_CONTROL_DEFINITIONS)}")
    rng = spawn_rng(int(instance_seed), "pages.instruction_panel.control_pool")
    definitions = list(_CONTROL_DEFINITIONS)
    rng.shuffle(definitions)
    color_offset = int(rng.randrange(len(_CONTROL_ACCENTS)))
    controls: List[_Control] = []
    for index, (control_id, label) in enumerate(definitions[: int(count)]):
        accent = _CONTROL_ACCENTS[(int(index) + int(color_offset)) % len(_CONTROL_ACCENTS)]
        controls.append(
            _Control(
                control_id=str(control_id),
                label=str(label),
                accent_rgb=tuple(int(channel) for channel in accent),
            )
        )
    return tuple(controls)


def _control_by_label(controls: Sequence[_Control], label: str) -> _Control:
    for control in controls:
        if str(control.label).lower() == str(label).strip().lower():
            return control
    raise ValueError(f"unknown control label: {label}")


def _pick_control(
    *,
    task_id: str,
    controls: Sequence[_Control],
    params: Mapping[str, Any],
    explicit_label_key: str,
    explicit_index_key: str,
    instance_seed: int,
    namespace: str,
) -> _Control:
    explicit_label = params.get(str(explicit_label_key))
    if explicit_label is not None:
        return _control_by_label(controls, str(explicit_label))
    explicit_index = params.get(str(explicit_index_key))
    if explicit_index is not None:
        index = int(explicit_index)
        if index < 0 or index >= len(controls):
            raise ValueError(f"{explicit_index_key} must be in 0..{len(controls) - 1}")
        return controls[int(index)]
    index = resolve_selection_index(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{task_id}.{namespace}",
    )
    return controls[int(index) % len(controls)]


def _pick_step_indices(
    *,
    task_id: str,
    params: Mapping[str, Any],
    step_count: int,
    step_set_size: int,
    instance_seed: int,
) -> Tuple[int, ...]:
    explicit = params.get("target_step_indices")
    if explicit is not None:
        values = tuple(int(value) for value in explicit)
        if len(values) != int(step_set_size):
            raise ValueError("target_step_indices length must match step_set_size")
        if len(set(values)) != len(values):
            raise ValueError("target_step_indices must be unique")
        if any(value < 0 or value >= int(step_count) for value in values):
            raise ValueError(f"target_step_indices must be in 0..{int(step_count) - 1}")
        return tuple(sorted(values))
    rng = spawn_rng(int(instance_seed), f"{task_id}.target_step_indices.{step_count}.{step_set_size}")
    return tuple(sorted(rng.sample(range(int(step_count)), k=int(step_set_size))))


def _make_control_assignment(
    *,
    controls: Sequence[_Control],
    count: int,
    rng: Any,
    avoid_pair: Tuple[str, str] | None = None,
) -> Tuple[_Control, ...]:
    selected = list(rng.sample(list(controls), k=int(count)))
    if avoid_pair is not None:
        pair_ids = {str(value) for value in avoid_pair}
        if pair_ids.issubset({str(control.control_id) for control in selected}):
            remove_id = sorted(pair_ids)[0]
            selected = [control for control in selected if str(control.control_id) != remove_id]
            replacements = [
                control
                for control in controls
                if str(control.control_id) not in {str(item.control_id) for item in selected}
                and str(control.control_id) not in pair_ids
            ]
            if not replacements:
                replacements = [
                    control
                    for control in controls
                    if str(control.control_id) not in {str(item.control_id) for item in selected}
                ]
            selected.append(replacements[int(rng.randrange(len(replacements)))])
    rng.shuffle(selected)
    return tuple(selected)


def _build_base_step_text(*, step_count: int, instance_seed: int) -> Tuple[List[str], List[str], str, str, Dict[str, Any]]:
    rng = spawn_rng(int(instance_seed), "pages.instruction_panel.step_text")
    title_batch = sample_page_context_batch(
        rng,
        role="instruction_panel_title",
        count=1,
        manifest_names=("phrases/headlines.txt",),
    )
    subtitle_batch = sample_page_context_batch(
        rng,
        role="instruction_panel_subtitle",
        count=1,
        manifest_names=("phrases/captions.txt", "phrases/legend_notes.txt"),
    )
    step_title_batch = sample_page_label_batch(
        rng,
        role="instruction_panel_step_title",
        count=int(step_count),
        manifest_name="panel_titles/technical_topics.txt",
        min_chars=3,
        max_chars=16,
        allow_spaces=True,
        allow_punctuation=False,
    )
    detail_batch = sample_page_context_batch(
        rng,
        role="instruction_panel_step_detail",
        count=int(step_count),
        manifest_names=("sentences/context_template_sentences.txt", "phrases/callout_phrases.txt"),
    )
    return (
        list(step_title_batch.values),
        list(detail_batch.values),
        str(title_batch.values[0]),
        str(subtitle_batch.values[0]),
        page_text_resource_metadata(title_batch, subtitle_batch, step_title_batch, detail_batch),
    )


def _build_shared_control_spec(
    *,
    task_id: str,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    step_count: int,
    controls_per_step: int,
    step_set_size: int,
    control_count: int,
) -> Tuple[_InstructionPanelSpec, Tuple[int, ...], _Control]:
    """Build steps so exactly one control is common to the referenced step set."""

    controls = _control_pool(count=int(control_count), instance_seed=int(instance_seed))
    target_control = _pick_control(
        task_id=task_id,
        controls=controls,
        params=params,
        explicit_label_key="target_control_label",
        explicit_index_key="target_control_index",
        instance_seed=int(instance_seed),
        namespace="target_control",
    )
    target_indices = _pick_step_indices(
        task_id=task_id,
        params=params,
        step_count=int(step_count),
        step_set_size=int(step_set_size),
        instance_seed=int(instance_seed),
    )
    titles, details, panel_title, panel_subtitle, page_text_resources = _build_base_step_text(
        step_count=int(step_count),
        instance_seed=int(instance_seed),
    )
    rng = spawn_rng(int(instance_seed), f"{task_id}.control_assignment")
    filler_pool = [control for control in controls if str(control.control_id) != str(target_control.control_id)]
    if len(filler_pool) < int(step_set_size) * max(1, int(controls_per_step) - 1):
        raise ValueError("control pool is too small for unique shared-control fillers")
    shuffled_fillers = list(filler_pool)
    rng.shuffle(shuffled_fillers)

    steps: List[_Step] = []
    for index in range(int(step_count)):
        if int(index) in set(target_indices):
            selected_position = list(target_indices).index(int(index))
            start = int(selected_position) * max(1, int(controls_per_step) - 1)
            fillers = shuffled_fillers[start : start + max(1, int(controls_per_step) - 1)]
            step_controls = [target_control, *fillers[: max(0, int(controls_per_step) - 1)]]
            rng.shuffle(step_controls)
        else:
            step_controls = list(
                _make_control_assignment(
                    controls=controls,
                    count=int(controls_per_step),
                    rng=rng,
                )
            )
        steps.append(
            _Step(
                step_id=f"step_{index + 1}",
                order_index=int(index),
                step_number=int(index) + 1,
                title=str(titles[int(index)]),
                detail=str(details[int(index)]),
                controls=tuple(step_controls),
            )
        )
    selected_sets = [
        {str(control.control_id) for control in steps[int(index)].controls}
        for index in target_indices
    ]
    if set.intersection(*selected_sets) != {str(target_control.control_id)}:
        raise RuntimeError("shared-control construction failed to produce a unique common control")
    spec = _InstructionPanelSpec(
        title=str(panel_title),
        subtitle=str(panel_subtitle),
        controls=tuple(controls),
        steps=tuple(steps),
        text_resource_metadata=dict(page_text_resources),
    )
    return spec, tuple(target_indices), target_control


def _build_control_pair_spec(
    *,
    task_id: str,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    step_count: int,
    controls_per_step: int,
    control_count: int,
) -> Tuple[_InstructionPanelSpec, int, Tuple[_Control, _Control]]:
    """Build steps so exactly one step contains the requested control pair."""

    controls = _control_pool(count=int(control_count), instance_seed=int(instance_seed))
    explicit_pair = params.get("target_pair_control_labels")
    if explicit_pair is not None:
        values = tuple(str(value) for value in explicit_pair)
        if len(values) != 2 or values[0].lower() == values[1].lower():
            raise ValueError("target_pair_control_labels must contain two distinct labels")
        pair = (_control_by_label(controls, values[0]), _control_by_label(controls, values[1]))
    else:
        rng_pair = spawn_rng(int(instance_seed), f"{task_id}.target_pair")
        pair = tuple(rng_pair.sample(list(controls), k=2))  # type: ignore[assignment]
    explicit_step = params.get("target_step_index")
    if explicit_step is not None:
        target_index = int(explicit_step)
        if target_index < 0 or target_index >= int(step_count):
            raise ValueError(f"target_step_index must be in 0..{int(step_count) - 1}")
    else:
        target_index = int(
            resolve_selection_index(
                params=params,
                instance_seed=int(instance_seed),
                namespace=f"{task_id}.target_step_index.{step_count}",
            )
            % int(step_count)
        )

    titles, details, panel_title, panel_subtitle, page_text_resources = _build_base_step_text(
        step_count=int(step_count),
        instance_seed=int(instance_seed),
    )
    rng = spawn_rng(int(instance_seed), f"{task_id}.control_assignment")
    pair_ids = (str(pair[0].control_id), str(pair[1].control_id))
    steps: List[_Step] = []
    for index in range(int(step_count)):
        if int(index) == int(target_index):
            remaining = [control for control in controls if str(control.control_id) not in set(pair_ids)]
            fillers = rng.sample(remaining, k=max(0, int(controls_per_step) - 2))
            step_controls = [pair[0], pair[1], *fillers]
            rng.shuffle(step_controls)
        else:
            step_controls = list(
                _make_control_assignment(
                    controls=controls,
                    count=int(controls_per_step),
                    rng=rng,
                    avoid_pair=pair_ids,
                )
            )
        steps.append(
            _Step(
                step_id=f"step_{index + 1}",
                order_index=int(index),
                step_number=int(index) + 1,
                title=str(titles[int(index)]),
                detail=str(details[int(index)]),
                controls=tuple(step_controls),
            )
        )
    matching = [
        int(index)
        for index, step in enumerate(steps)
        if set(pair_ids).issubset({str(control.control_id) for control in step.controls})
    ]
    if matching != [int(target_index)]:
        raise RuntimeError("control-pair construction failed to produce a unique matching step")
    spec = _InstructionPanelSpec(
        title=str(panel_title),
        subtitle=str(panel_subtitle),
        controls=tuple(controls),
        steps=tuple(steps),
        text_resource_metadata=dict(page_text_resources),
    )
    return spec, int(target_index), (pair[0], pair[1])


def _layout_step_bboxes(
    *,
    scene_variant: str,
    step_count: int,
    render_params: _RenderParams,
    content_bbox: Sequence[float],
) -> Tuple[List[List[float]], Dict[str, Any], List[float] | None]:
    """Resolve visible step-card geometry for each supported panel layout."""

    x0, y0, x1, y1 = [float(value) for value in content_bbox]
    gap = float(render_params.gap_px)
    if str(scene_variant) == "checklist_table":
        row_h = (y1 - y0 - float(step_count - 1) * 2.0) / float(step_count)
        rows = []
        for index in range(int(step_count)):
            top = y0 + float(index) * (row_h + 2.0)
            rows.append([x0, top, x1, top + row_h])
        return rows, {"layout_columns": 1, "layout_rows": int(step_count), "layout_kind": "checklist_table"}, None

    if str(scene_variant) == "side_legend_sheet":
        legend_w = min(250.0, max(210.0, (x1 - x0) * 0.26))
        legend_bbox = [x1 - legend_w, y0, x1, y1]
        left_x1 = legend_bbox[0] - gap
        row_h = (y1 - y0 - float(step_count - 1) * float(gap)) / float(step_count)
        rows = []
        for index in range(int(step_count)):
            top = y0 + float(index) * (row_h + gap)
            rows.append([x0, top, left_x1, top + row_h])
        return (
            rows,
            {
                "layout_columns": 1,
                "layout_rows": int(step_count),
                "layout_kind": "side_legend_sheet",
                "legend_bbox_px": [float(value) for value in legend_bbox],
            },
            legend_bbox,
        )

    columns = 2 if int(step_count) > 5 else 1
    rows = int(math.ceil(float(step_count) / float(columns)))
    card_w = (x1 - x0 - float(columns - 1) * gap) / float(columns)
    card_h = (y1 - y0 - float(rows - 1) * gap) / float(rows)
    bboxes: List[List[float]] = []
    for index in range(int(step_count)):
        row = int(index) // int(columns)
        col = int(index) % int(columns)
        left = x0 + float(col) * (card_w + gap)
        top = y0 + float(row) * (card_h + gap)
        bboxes.append([left, top, left + card_w, top + card_h])
    return (
        bboxes,
        {"layout_columns": int(columns), "layout_rows": int(rows), "layout_kind": "manual_cards"},
        None,
    )


def _draw_control_icon(
    draw: ImageDraw.ImageDraw,
    *,
    control_id: str,
    bbox: Sequence[float],
    fill: Tuple[int, int, int],
) -> None:
    """Draw compact glyphs that keep control chips visually distinguishable."""

    x0, y0, x1, y1 = [float(value) for value in bbox]
    w = x1 - x0
    h = y1 - y0
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    line_w = max(1, int(round(min(w, h) / 8.0)))
    control = str(control_id)
    if control == "save":
        draw.rectangle((x0 + 2, y0 + 2, x1 - 2, y1 - 2), outline=fill, width=line_w)
        draw.rectangle((x0 + 5, y0 + 5, x1 - 6, y0 + h * 0.42), outline=fill, width=line_w)
        draw.line((x0 + 6, y1 - 6, x1 - 6, y1 - 6), fill=fill, width=line_w)
    elif control == "print":
        draw.rectangle((x0 + 4, y0 + 2, x1 - 4, y0 + h * 0.40), outline=fill, width=line_w)
        draw.rectangle((x0 + 2, y0 + h * 0.38, x1 - 2, y1 - 4), outline=fill, width=line_w)
        draw.line((x0 + 6, y1 - 8, x1 - 6, y1 - 8), fill=fill, width=line_w)
    elif control == "upload":
        draw.line((cx, y0 + 3, cx, y1 - 5), fill=fill, width=line_w)
        draw.polygon([(cx, y0 + 2), (x0 + 5, y0 + h * 0.40), (x1 - 5, y0 + h * 0.40)], fill=fill)
        draw.line((x0 + 4, y1 - 5, x1 - 4, y1 - 5), fill=fill, width=line_w)
    elif control == "download":
        draw.line((cx, y0 + 4, cx, y1 - 3), fill=fill, width=line_w)
        draw.polygon([(cx, y1 - 2), (x0 + 5, y1 - h * 0.40), (x1 - 5, y1 - h * 0.40)], fill=fill)
        draw.line((x0 + 4, y1 - 4, x1 - 4, y1 - 4), fill=fill, width=line_w)
    elif control == "sync":
        draw.arc((x0 + 2, y0 + 2, x1 - 2, y1 - 2), 25, 205, fill=fill, width=line_w)
        draw.arc((x0 + 2, y0 + 2, x1 - 2, y1 - 2), 205, 25, fill=fill, width=line_w)
        draw.polygon([(x0 + 5, cy), (x0 + 5, cy - 6), (x0, cy - 2)], fill=fill)
        draw.polygon([(x1 - 5, cy), (x1 - 5, cy + 6), (x1, cy + 2)], fill=fill)
    elif control == "search":
        draw.ellipse((x0 + 3, y0 + 3, x1 - 7, y1 - 7), outline=fill, width=line_w)
        draw.line((cx + 4, cy + 4, x1 - 2, y1 - 2), fill=fill, width=line_w)
    elif control == "share":
        points = [(x0 + 5, cy), (cx + 2, y0 + 5), (x1 - 5, y1 - 5)]
        draw.line((points[0][0], points[0][1], points[1][0], points[1][1]), fill=fill, width=line_w)
        draw.line((points[1][0], points[1][1], points[2][0], points[2][1]), fill=fill, width=line_w)
        for px, py in points:
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=fill)
    elif control == "settings":
        draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), outline=fill, width=line_w)
        for angle in range(0, 360, 45):
            radians = math.radians(float(angle))
            draw.line(
                (
                    cx + math.cos(radians) * 8.0,
                    cy + math.sin(radians) * 8.0,
                    cx + math.cos(radians) * 12.0,
                    cy + math.sin(radians) * 12.0,
                ),
                fill=fill,
                width=line_w,
            )
    elif control == "alert":
        draw.polygon([(cx, y0 + 2), (x0 + 2, y1 - 3), (x1 - 2, y1 - 3)], outline=fill)
        draw.line((cx, y0 + 8, cx, y1 - 9), fill=fill, width=line_w)
        draw.ellipse((cx - 1.5, y1 - 6, cx + 1.5, y1 - 3), fill=fill)
    elif control == "lock":
        draw.rectangle((x0 + 4, cy - 1, x1 - 4, y1 - 3), outline=fill, width=line_w)
        draw.arc((x0 + 6, y0 + 2, x1 - 6, cy + 6), 180, 360, fill=fill, width=line_w)
    elif control == "filter":
        draw.polygon([(x0 + 2, y0 + 4), (x1 - 2, y0 + 4), (cx + 3, cy), (cx + 3, y1 - 4), (cx - 3, y1 - 4), (cx - 3, cy)], outline=fill)
    elif control == "review":
        draw.line((x0 + 4, cy, cx - 1, y1 - 5, x1 - 3, y0 + 5), fill=fill, width=line_w + 1)
    else:
        draw.ellipse((x0 + 4, y0 + 4, x1 - 4, y1 - 4), outline=fill, width=line_w)


def _draw_control_chip(
    draw: ImageDraw.ImageDraw,
    *,
    control: _Control,
    bbox: Sequence[float],
    style: InformationSceneStyle,
    font_size: int,
) -> Tuple[List[float], List[float]]:
    """Draw one control chip and return chip plus label witness boxes."""

    x0, y0, x1, y1 = [float(value) for value in bbox]
    accent = tuple(int(channel) for channel in control.accent_rgb)
    fill = _blend_rgb(style.callout_fill_rgb, accent, 0.08)
    border = _blend_rgb(accent, style.panel_border_rgb, 0.22)
    text_rgb = tuple(int(channel) for channel in style.text_rgb)
    radius = max(4, int(round((y1 - y0) / 2.8)))
    draw.rounded_rectangle((x0, y0, x1, y1), radius=radius, fill=fill, outline=border, width=2)
    icon_box = [x0 + 8.0, y0 + 6.0, x0 + 24.0, y1 - 6.0]
    _draw_control_icon(draw, control_id=str(control.control_id), bbox=icon_box, fill=accent)
    label_left = x0 + 32.0
    label_right = x1 - 8.0
    font = fit_font_to_box(
        draw,
        text=str(control.label),
        max_width=max(24.0, label_right - label_left),
        max_height=max(12.0, (y1 - y0) - 8.0),
        bold=True,
        min_size_px=9,
        max_size_px=int(font_size),
        fill_ratio=0.96,
    )
    label_measure = _text_bbox(draw, (0.0, 0.0), str(control.label), font)
    label_h = label_measure[3] - label_measure[1]
    label_y = y0 + ((y1 - y0 - label_h) * 0.48) - 1.0
    label_bbox = _draw_text(
        draw,
        (label_left, label_y),
        str(control.label),
        font,
        text_rgb,
        role="control_label",
    )
    return [float(value) for value in bbox], [float(value) for value in label_bbox]


def _draw_step(
    draw: ImageDraw.ImageDraw,
    *,
    step: _Step,
    bbox: Sequence[float],
    scene_variant: str,
    style: InformationSceneStyle,
    render_params: _RenderParams,
    row_index: int,
) -> Tuple[List[float], List[float], Dict[str, List[float]], Dict[str, List[float]], Dict[str, Any]]:
    """Draw one numbered step while preserving control-chip bbox metadata."""

    x0, y0, x1, y1 = [float(value) for value in bbox]
    height = y1 - y0
    is_table = str(scene_variant) == "checklist_table"
    compact = bool(height < 106.0 or str(scene_variant) == "side_legend_sheet")
    accent = tuple(int(channel) for channel in step.controls[0].accent_rgb)
    fill_base = style.panel_fill_rgb if int(row_index) % 2 == 0 else style.surface_alt_rgb
    card_fill = _blend_rgb(fill_base, accent, 0.025 if is_table else 0.045)
    border = tuple(int(channel) for channel in style.panel_border_rgb)
    text_rgb = tuple(int(channel) for channel in style.text_rgb)
    muted_rgb = tuple(int(channel) for channel in style.muted_text_rgb)
    if is_table:
        draw.rectangle((x0, y0, x1, y1), fill=card_fill, outline=border, width=1)
    else:
        draw.rounded_rectangle(
            (x0, y0, x1, y1),
            radius=int(render_params.corner_radius_px),
            fill=card_fill,
            outline=border,
            width=int(render_params.outline_width_px),
        )
        draw.rounded_rectangle(
            (x0, y0, x1, y0 + 6.0),
            radius=int(render_params.corner_radius_px),
            fill=accent,
        )

    badge_size = float(render_params.number_badge_size_px)
    badge_x0 = x0 + 14.0
    badge_y0 = y0 + max(8.0, (height - badge_size) * 0.5 if is_table else 16.0)
    badge_bbox = [badge_x0, badge_y0, badge_x0 + badge_size, badge_y0 + badge_size]
    draw.ellipse(tuple(badge_bbox), fill=accent, outline=_blend_rgb(accent, (0, 0, 0), 0.18), width=2)
    number_font = fit_font_to_box(
        draw,
        text=str(step.step_number),
        max_width=badge_size - 7.0,
        max_height=badge_size - 7.0,
        bold=True,
        min_size_px=11,
        max_size_px=21,
        fill_ratio=0.94,
    )
    number_measure = _text_bbox(draw, (0.0, 0.0), str(step.step_number), number_font)
    number_xy = (
        badge_x0 + (badge_size - (number_measure[2] - number_measure[0])) * 0.5,
        badge_y0 + (badge_size - (number_measure[3] - number_measure[1])) * 0.46 - 1.0,
    )
    _draw_text(draw, number_xy, str(step.step_number), number_font, (255, 255, 255), role="step_number")

    text_left = badge_x0 + badge_size + 14.0
    title_font = fit_font_to_box(
        draw,
        text=str(step.title),
        max_width=max(70.0, (x1 - x0) * (0.34 if is_table else 0.68)),
        max_height=28.0,
        bold=True,
        min_size_px=11,
        max_size_px=int(render_params.step_title_font_size_px),
        fill_ratio=0.95,
    )
    title_top = y0 + (12.0 if is_table or compact else 24.0)
    title_bbox = _draw_text(draw, (text_left, title_top), str(step.title), title_font, text_rgb, role="step_title")
    if not is_table:
        detail_font = fit_font_to_box(
            draw,
            text=str(step.detail),
            max_width=max(70.0, x1 - text_left - 16.0),
            max_height=20.0,
            bold=False,
            min_size_px=9,
            max_size_px=int(render_params.step_detail_font_size_px),
            fill_ratio=0.95,
        )
        if not compact:
            _draw_text(draw, (text_left, title_top + 30.0), str(step.detail), detail_font, muted_rgb, role="step_detail")

    chip_gap = 8.0
    chip_h = float(render_params.control_chip_height_px)
    chip_count = len(step.controls)
    if is_table:
        chips_left = max(text_left + 230.0, x1 - (float(chip_count) * 118.0 + float(chip_count - 1) * chip_gap) - 16.0)
        chips_top = y0 + (height - chip_h) * 0.5
        available = x1 - chips_left - 16.0
    else:
        chips_left = text_left
        chips_top = y0 + (48.0 if compact else 76.0)
        available = max(80.0, x1 - chips_left - 16.0)
    chip_w = max(86.0, min(132.0, (available - float(chip_count - 1) * chip_gap) / float(chip_count)))
    control_chip_bboxes: Dict[str, List[float]] = {}
    control_label_bboxes: Dict[str, List[float]] = {}
    for control_index, control in enumerate(step.controls):
        cx0 = chips_left + float(control_index) * (chip_w + chip_gap)
        cy0 = min(chips_top, y1 - chip_h - 6.0)
        chip_bbox = [cx0, cy0, cx0 + chip_w, cy0 + chip_h]
        drawn_bbox, label_bbox = _draw_control_chip(
            draw,
            control=control,
            bbox=chip_bbox,
            style=style,
            font_size=int(render_params.control_font_size_px),
        )
        control_chip_bboxes[str(control.control_id)] = [float(value) for value in drawn_bbox]
        control_label_bboxes[str(control.control_id)] = [float(value) for value in label_bbox]

    entity = {
        "id": str(step.step_id),
        "type": "instruction_panel_step",
        "bbox_px": [float(value) for value in bbox],
        "attrs": {
            "order_index": int(step.order_index),
            "step_number": int(step.step_number),
            "title": str(step.title),
            "control_labels": [str(control.label) for control in step.controls],
            "control_ids": [str(control.control_id) for control in step.controls],
        },
    }
    return (
        [float(value) for value in badge_bbox],
        [float(value) for value in title_bbox],
        control_chip_bboxes,
        control_label_bboxes,
        entity,
    )


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    *,
    controls: Sequence[_Control],
    legend_bbox: Sequence[float],
    style: InformationSceneStyle,
    render_params: _RenderParams,
) -> Dict[str, List[float]]:
    x0, y0, x1, y1 = [float(value) for value in legend_bbox]
    draw.rounded_rectangle(
        (x0, y0, x1, y1),
        radius=int(render_params.corner_radius_px),
        fill=_blend_rgb(style.surface_alt_rgb, style.accent_rgb, 0.025),
        outline=tuple(int(channel) for channel in style.panel_border_rgb),
        width=int(render_params.outline_width_px),
    )
    title_font = load_font(max(11, int(render_params.subtitle_font_size_px)), bold=True)
    _draw_text(draw, (x0 + 14.0, y0 + 12.0), "Control legend", title_font, tuple(style.text_rgb), role="legend_title")
    entry_top = y0 + 44.0
    entry_gap = 8.0
    chip_h = float(render_params.control_chip_height_px)
    bboxes: Dict[str, List[float]] = {}
    for index, control in enumerate(controls):
        cy0 = entry_top + float(index) * (chip_h + entry_gap)
        if cy0 + chip_h > y1 - 12.0:
            break
        chip_bbox = [x0 + 12.0, cy0, x1 - 12.0, cy0 + chip_h]
        drawn_bbox, _label_bbox = _draw_control_chip(
            draw,
            control=control,
            bbox=chip_bbox,
            style=style,
            font_size=int(render_params.control_font_size_px),
        )
        bboxes[str(control.control_id)] = [float(value) for value in drawn_bbox]
    return bboxes


def _render_instruction_panel(
    background: Image.Image,
    *,
    spec: _InstructionPanelSpec,
    scene_variant: str,
    style: InformationSceneStyle,
    render_params: _RenderParams,
) -> _RenderedInstructionPanel:
    """Render the full instruction panel and collect all projection metadata."""

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    width = int(render_params.canvas_width)
    height = int(render_params.canvas_height)
    margin = int(render_params.outer_margin_px)
    panel_bbox = [float(margin), float(margin), float(width - margin), float(height - margin)]
    shadow_offset = max(0, min(8, int(style.shadow_offset_px)))
    if shadow_offset:
        draw.rounded_rectangle(
            (
                panel_bbox[0] + shadow_offset,
                panel_bbox[1] + shadow_offset,
                panel_bbox[2] + shadow_offset,
                panel_bbox[3] + shadow_offset,
            ),
            radius=max(4, int(style.corner_radius_px) + 8),
            fill=tuple(int(channel) for channel in style.shadow_rgb),
        )
    draw.rounded_rectangle(
        tuple(panel_bbox),
        radius=max(4, int(style.corner_radius_px) + 8),
        fill=tuple(int(channel) for channel in style.surface_rgb),
        outline=tuple(int(channel) for channel in style.panel_border_rgb),
        width=2,
    )
    header_bbox = [panel_bbox[0], panel_bbox[1], panel_bbox[2], panel_bbox[1] + float(render_params.header_height_px)]
    draw.rounded_rectangle(
        tuple(header_bbox),
        radius=max(4, int(style.corner_radius_px) + 8),
        fill=tuple(int(channel) for channel in style.header_rgb),
    )
    draw.rectangle(
        (header_bbox[0], header_bbox[3] - 14.0, header_bbox[2], header_bbox[3]),
        fill=tuple(int(channel) for channel in style.header_rgb),
    )
    title_font = load_font(int(render_params.title_font_size_px), bold=True)
    subtitle_font = load_font(int(render_params.subtitle_font_size_px), bold=False)
    title_xy = (panel_bbox[0] + 24.0, panel_bbox[1] + 16.0)
    title_bbox = _draw_text(draw, title_xy, str(spec.title), title_font, tuple(style.header_text_rgb), role="page_title")
    _draw_text(
        draw,
        (title_xy[0], title_xy[1] + 38.0),
        str(spec.subtitle),
        subtitle_font,
        tuple(style.header_text_rgb),
        role="page_subtitle",
    )

    content_bbox = [
        panel_bbox[0] + 22.0,
        header_bbox[3] + 18.0,
        panel_bbox[2] - 22.0,
        panel_bbox[3] - 22.0,
    ]
    step_bboxes, layout_meta, legend_bbox = _layout_step_bboxes(
        scene_variant=str(scene_variant),
        step_count=len(spec.steps),
        render_params=render_params,
        content_bbox=content_bbox,
    )
    legend_bboxes: Dict[str, List[float]] = {}
    if legend_bbox is not None:
        legend_bboxes = _draw_legend(
            draw,
            controls=spec.controls,
            legend_bbox=legend_bbox,
            style=style,
            render_params=render_params,
        )

    entities: List[Dict[str, Any]] = []
    step_card_bboxes: Dict[str, List[float]] = {}
    step_number_bboxes: Dict[str, List[float]] = {}
    step_title_bboxes: Dict[str, List[float]] = {}
    control_chip_bboxes: Dict[str, Dict[str, List[float]]] = {}
    control_label_bboxes: Dict[str, Dict[str, List[float]]] = {}
    for row_index, (step, bbox) in enumerate(zip(spec.steps, step_bboxes)):
        number_bbox, title_step_bbox, chip_bboxes, label_bboxes, entity = _draw_step(
            draw,
            step=step,
            bbox=bbox,
            scene_variant=str(scene_variant),
            style=style,
            render_params=render_params,
            row_index=int(row_index),
        )
        step_card_bboxes[str(step.step_id)] = [float(value) for value in bbox]
        step_number_bboxes[str(step.step_id)] = [float(value) for value in number_bbox]
        step_title_bboxes[str(step.step_id)] = [float(value) for value in title_step_bbox]
        control_chip_bboxes[str(step.step_id)] = dict(chip_bboxes)
        control_label_bboxes[str(step.step_id)] = dict(label_bboxes)
        entities.append(dict(entity))
        for control in step.controls:
            entities.append(
                {
                    "id": f"{step.step_id}__{control.control_id}",
                    "type": "instruction_panel_control_chip",
                    "bbox_px": [float(value) for value in chip_bboxes[str(control.control_id)]],
                    "attrs": {
                        "step_id": str(step.step_id),
                        "step_number": int(step.step_number),
                        "control_id": str(control.control_id),
                        "control_label": str(control.label),
                    },
                }
            )
    return _RenderedInstructionPanel(
        image=image,
        entities=entities,
        panel_bbox_px=list(panel_bbox),
        title_bbox_px=list(title_bbox),
        step_card_bboxes_px=dict(step_card_bboxes),
        step_number_bboxes_px=dict(step_number_bboxes),
        step_title_bboxes_px=dict(step_title_bboxes),
        control_chip_bboxes_px=dict(control_chip_bboxes),
        control_label_bboxes_px=dict(control_label_bboxes),
        legend_control_bboxes_px=dict(legend_bboxes),
        layout_meta={
            "scene_variant": str(scene_variant),
            "step_count": int(len(spec.steps)),
            "control_count": int(len(spec.controls)),
            "controls_per_step": int(len(spec.steps[0].controls)) if spec.steps else 0,
            **dict(layout_meta),
        },
    )


def _format_step_reference_list(step_numbers: Sequence[int]) -> str:
    values = [str(int(value)) for value in step_numbers]
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def _make_step_payload(
    *,
    spec: _InstructionPanelSpec,
    rendered: _RenderedInstructionPanel,
) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for step in spec.steps:
        step_id = str(step.step_id)
        payload.append(
            {
                "step_id": step_id,
                "order_index": int(step.order_index),
                "step_number": int(step.step_number),
                "title": str(step.title),
                "detail": str(step.detail),
                "step_bbox_px": [float(value) for value in rendered.step_card_bboxes_px[step_id]],
                "number_bbox_px": [float(value) for value in rendered.step_number_bboxes_px[step_id]],
                "title_bbox_px": [float(value) for value in rendered.step_title_bboxes_px[step_id]],
                "controls": [
                    {
                        "control_id": str(control.control_id),
                        "label": str(control.label),
                        "chip_bbox_px": [
                            float(value)
                            for value in rendered.control_chip_bboxes_px[step_id][str(control.control_id)]
                        ],
                        "label_bbox_px": [
                            float(value)
                            for value in rendered.control_label_bboxes_px[step_id][str(control.control_id)]
                        ],
                        "accent_rgb": [int(channel) for channel in control.accent_rgb],
                    }
                    for control in step.controls
                ],
            }
        )
    return payload


def _generate_instruction_panel_output(
    *,
    task_namespace: str,
    prompt_key: str,
    question_format: str,
    instance_seed: int,
    params: Dict[str, Any],
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    prompt_defaults_raw: Mapping[str, Any],
) -> TaskOutput:
    """Generate one task instance from scene metadata, prompt, and annotation."""

    scene_variant, scene_variant_probabilities = _resolve_named_variant(
        task_id=task_namespace,
        gen_defaults=gen_defaults,
        params=params,
        instance_seed=int(instance_seed),
        supported=SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        namespace="scene_variant",
    )
    step_count, step_count_support, step_count_probabilities = _resolve_supported_int(
        task_id=task_namespace,
        params=params,
        gen_defaults=gen_defaults,
        explicit_key="step_count",
        support_key="step_count_support",
        fallback=(5, 6, 7, 8),
        instance_seed=int(instance_seed),
        namespace="step_count",
    )
    controls_per_step, controls_per_step_support, controls_per_step_probabilities = _resolve_supported_int(
        task_id=task_namespace,
        params=params,
        gen_defaults=gen_defaults,
        explicit_key="controls_per_step",
        support_key="controls_per_step_support",
        fallback=(2, 3),
        instance_seed=int(instance_seed),
        namespace="controls_per_step",
    )
    control_count, control_count_support, control_count_probabilities = _resolve_supported_int(
        task_id=task_namespace,
        params=params,
        gen_defaults=gen_defaults,
        explicit_key="control_count",
        support_key="control_count_support",
        fallback=(9, 10, 11, 12),
        instance_seed=int(instance_seed),
        namespace="control_count",
    )
    if int(controls_per_step) < 2:
        raise ValueError("controls_per_step must be at least 2")
    if int(control_count) < max(6, int(controls_per_step) * 3):
        raise ValueError("control_count is too small for instruction-panel uniqueness constraints")

    if str(prompt_key) == _SHARED_CONTROL_PROMPT_KEY:
        step_set_size, step_set_size_support, step_set_size_probabilities = _resolve_supported_int(
            task_id=task_namespace,
            params=params,
            gen_defaults=gen_defaults,
            explicit_key="step_set_size",
            support_key="step_set_size_support",
            fallback=(2, 3),
            instance_seed=int(instance_seed),
            namespace="step_set_size",
        )
        spec, target_step_indices, target_control = _build_shared_control_spec(
            task_id=task_namespace,
            params=params,
            gen_defaults=gen_defaults,
            instance_seed=int(instance_seed),
            step_count=int(step_count),
            controls_per_step=int(controls_per_step),
            step_set_size=int(step_set_size),
            control_count=int(control_count),
        )
        target_step_numbers = [int(spec.steps[int(index)].step_number) for index in target_step_indices]
        target_payload: Dict[str, Any] = {
            "target_step_indices": [int(index) for index in target_step_indices],
            "target_step_numbers": [int(value) for value in target_step_numbers],
            "target_control_id": str(target_control.control_id),
            "target_control_label": str(target_control.label),
            "step_reference_list": _format_step_reference_list(target_step_numbers),
            "step_set_size": int(step_set_size),
            "step_set_size_support": [int(value) for value in step_set_size_support],
            "step_set_size_probabilities": dict(step_set_size_probabilities),
        }
    else:
        spec, target_step_index, control_pair = _build_control_pair_spec(
            task_id=task_namespace,
            params=params,
            gen_defaults=gen_defaults,
            instance_seed=int(instance_seed),
            step_count=int(step_count),
            controls_per_step=int(controls_per_step),
            control_count=int(control_count),
        )
        target_step = spec.steps[int(target_step_index)]
        target_payload = {
            "target_step_index": int(target_step_index),
            "target_step_number": int(target_step.step_number),
            "first_control_id": str(control_pair[0].control_id),
            "first_control_label": str(control_pair[0].label),
            "second_control_id": str(control_pair[1].control_id),
            "second_control_label": str(control_pair[1].label),
        }

    render_params = _resolve_render_params(params, render_defaults)
    style_params = {**dict(render_defaults), **dict(params)}
    style, style_meta = resolve_pages_information_style(
        instance_seed=int(instance_seed),
        params=style_params,
        scene_id=SCENE,
    )
    background, background_meta = make_information_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=style,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_NAMESPACE}.information_scene_background",
    )
    rendered = _render_instruction_panel(
        background,
        spec=spec,
        scene_variant=str(scene_variant),
        style=style,
        render_params=render_params,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )

    if str(prompt_key) == _SHARED_CONTROL_PROMPT_KEY:
        answer_value: str | int = str(target_payload["target_control_label"])
        annotation_group_value = {
            "step_numbers": [
                [float(value) for value in rendered.step_number_bboxes_px[str(spec.steps[int(index)].step_id)]]
                for index in target_payload["target_step_indices"]
            ],
            "shared_control_chips": [
                [
                    float(value)
                    for value in rendered.control_chip_bboxes_px[str(spec.steps[int(index)].step_id)][
                        str(target_payload["target_control_id"])
                    ]
                ]
                for index in target_payload["target_step_indices"]
            ],
        }
        annotation_value = [list(bbox) for bbox in annotation_group_value["shared_control_chips"]]
        answer_type = "string"
        annotation_type = "bbox_set"
    else:
        target_step_id = str(spec.steps[int(target_payload["target_step_index"])].step_id)
        answer_value = int(target_payload["target_step_number"])
        annotation_keyed_value = {
            "first_control": [
                float(value)
                for value in rendered.control_chip_bboxes_px[target_step_id][str(target_payload["first_control_id"])]
            ],
            "second_control": [
                float(value)
                for value in rendered.control_chip_bboxes_px[target_step_id][str(target_payload["second_control_id"])]
            ],
            "target_step_number": [float(value) for value in rendered.step_number_bboxes_px[target_step_id]],
        }
        annotation_value = [list(bbox) for bbox in annotation_keyed_value.values()]
        answer_type = "integer"
        annotation_type = "bbox_set"
    prompt_defaults = required_group_defaults(
        prompt_defaults_raw,
        (
            "bundle_id",
        ),
        context=f"prompt defaults for {task_namespace}",
    )
    dynamic_slots = {
        "step_reference_list": str(target_payload.get("step_reference_list", "")),
        "first_control_label": f'"{str(target_payload.get("first_control_label", ""))}"',
        "second_control_label": f'"{str(target_payload.get("second_control_label", ""))}"',
    }
    prompt_selection = render_task_prompt_variants(
        domain="pages",
        scene_id=SCENE,
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=PROMPT_SCENE_KEY,
        task_key=PROMPT_TASK_KEY,
        query_key=str(prompt_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=dynamic_slots,
        instance_seed=int(instance_seed),
    )
    prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)
    query_params = {
        "query_id": SINGLE_QUERY_ID,
        "prompt_query_key": str(prompt_key),
        "source_query_id": str(prompt_key),
        "scene_variant": str(scene_variant),
        "target": dict(target_payload),
        "target_answer": answer_value,
        "step_count": int(step_count),
        "controls_per_step": int(controls_per_step),
        "control_count": int(control_count),
        "query_id_probabilities": {SINGLE_QUERY_ID: 1.0},
        "prompt_query_key_probabilities": {str(prompt_key): 1.0},
        "scene_variant_probabilities": dict(scene_variant_probabilities),
        "step_count_probabilities": dict(step_count_probabilities),
        "controls_per_step_probabilities": dict(controls_per_step_probabilities),
        "control_count_probabilities": dict(control_count_probabilities),
    }
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=SINGLE_QUERY_ID,
        params=dict(query_params),
    )
    query_spec["scene_id"] = SCENE
    steps_payload = _make_step_payload(spec=spec, rendered=rendered)
    trace_payload = {
        "scene_ir": {
            "scene_id": SCENE,
            "scene_kind": "pages_instruction_panel",
            "entities": [dict(entity) for entity in rendered.entities],
            "relations": {
                "query_id": SINGLE_QUERY_ID,
                "prompt_query_key": str(prompt_key),
                "source_query_id": str(prompt_key),
                "scene_variant": str(scene_variant),
                "target": dict(target_payload),
                "answer_value": answer_value,
            },
        },
        "query_spec": query_spec,
        "render_spec": {
            "canvas_width": int(render_params.canvas_width),
            "canvas_height": int(render_params.canvas_height),
            "coord_space": "pixel",
            "scene_id": SCENE,
            "query_id": SINGLE_QUERY_ID,
            "prompt_query_key": str(prompt_key),
            "scene_variant": str(scene_variant),
            "background_style": dict(background_meta),
            "information_scene_style": dict(style_meta),
            "post_image_noise": dict(post_noise_meta),
            "panel_bbox_px": list(rendered.panel_bbox_px),
            "layout": dict(rendered.layout_meta),
            "text_style": {
                "title_font_size_px": int(render_params.title_font_size_px),
                "step_title_font_size_px": int(render_params.step_title_font_size_px),
                "control_font_size_px": int(render_params.control_font_size_px),
            },
            "context_text_layer": {"enabled": False},
            "page_text_resources": dict(spec.text_resource_metadata),
        },
        "render_map": {
            "image_id": "img0",
            "panel_bbox_px": list(rendered.panel_bbox_px),
            "document_title_bbox_px": list(rendered.title_bbox_px),
            "step_card_bboxes_px": dict(rendered.step_card_bboxes_px),
            "step_number_bboxes_px": dict(rendered.step_number_bboxes_px),
            "step_title_bboxes_px": dict(rendered.step_title_bboxes_px),
            "control_chip_bboxes_px": dict(rendered.control_chip_bboxes_px),
            "control_label_bboxes_px": dict(rendered.control_label_bboxes_px),
            "legend_control_bboxes_px": dict(rendered.legend_control_bboxes_px),
            "context_text_bboxes_px": {},
        },
        "execution_trace": {
            **dict(query_params),
            "scene_variant": str(scene_variant),
            "question_format": str(question_format),
            "step_count": int(step_count),
            "controls_per_step": int(controls_per_step),
            "control_count": int(control_count),
            "step_count_support": [int(value) for value in step_count_support],
            "controls_per_step_support": [int(value) for value in controls_per_step_support],
            "control_count_support": [int(value) for value in control_count_support],
            "target": dict(target_payload),
            "answer_value": answer_value,
            "controls": [
                {
                    "control_id": str(control.control_id),
                    "label": str(control.label),
                    "accent_rgb": [int(channel) for channel in control.accent_rgb],
                }
                for control in spec.controls
            ],
            "steps": list(steps_payload),
            "page_text_resources": dict(spec.text_resource_metadata),
        },
        "witness_symbolic": {
            "type": str(question_format),
            "target": dict(target_payload),
            "answer_value": answer_value,
        },
    }
    if str(prompt_key) == _SHARED_CONTROL_PROMPT_KEY:
        trace_payload["projected_annotation"] = {
            "type": "bbox_set",
            "bbox_set": list(annotation_value),
            "pixel_bbox_set": list(annotation_value),
            "bbox_set_map": dict(annotation_group_value),
            "pixel_bbox_set_map": dict(annotation_group_value),
        }
    else:
        trace_payload["projected_annotation"] = {
            "type": "bbox_set",
            "bbox_set": list(annotation_value),
            "pixel_bbox_set": list(annotation_value),
            "bbox_map": dict(annotation_keyed_value),
            "pixel_bbox_map": dict(annotation_keyed_value),
        }

    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=TypedValue(type=str(answer_type), value=answer_value),
        annotation_gt=TypedValue(type=str(annotation_type), value=annotation_value),
        image=image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE,
        query_id=SINGLE_QUERY_ID,
        prompt_variants=dict(prompt_artifacts.prompt_variants),
    )


def build_instruction_panel_response(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    task_namespace: str,
    prompt_query_key: str,
    question_format: str,
) -> TaskOutput:
    lifecycle_params = dict(params)
    return _generate_instruction_panel_output(
        task_namespace=str(task_namespace),
        prompt_key=str(prompt_query_key),
        question_format=str(question_format),
        instance_seed=int(instance_seed),
        params=lifecycle_params,
        gen_defaults=_GEN_DEFAULTS,
        render_defaults=_RENDER_DEFAULTS,
        prompt_defaults_raw=_PROMPT_DEFAULTS,
    )


__all__ = [
    "SCENE",
    "SCENE_VARIANTS",
    "SUPPORTED_PROMPT_QUERY_KEYS",
    "SUPPORTED_QUERY_IDS",
    "build_instruction_panel_response",
]
