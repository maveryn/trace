"""Scene-private lifecycle for numbered step-list page tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ....core.scene_config import get_scene_defaults
from ....core.types import TypedValue
from ....core.visual.background import make_background_canvas
from ....core.visual.noise import apply_post_image_noise
from ...base import TaskOutput
from ...shared.config_defaults import group_default, split_generation_rendering_prompt_defaults
from ...shared.deterministic_sampling import resolve_selection_index
from ...shared.fixed_query import select_task_query_id
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_query_spec,
    build_prompt_trace_artifacts,
    render_task_prompt_variants,
)
from ...shared.text_rendering import draw_text_centered, fit_font_to_box, load_font
from ...shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant
from ...shared.text_legibility import draw_text_traced
from ..shared.page_text_resources import page_text_resource_metadata, sample_page_context_batch, sample_page_label_batch
from ..shared.visual_defaults import load_pages_background_defaults, load_pages_noise_defaults


DOMAIN = "pages"
SCENE = "step_list"
TASK_NAMESPACE = "pages.step_list"
PROMPT_BUNDLE = "pages_step_list_v1"
PROMPT_SCENE_KEY = "step_list"
PROMPT_TASK_KEY = "step_lookup_query"
ORDINAL_TITLE_MODE = "ordinal_title"
ORDINAL_DETAIL_MODE = "ordinal_detail"
AFTER_NAMED_TITLE_MODE = "after_named_title"
DETAIL_TO_TITLE_MODE = "detail_to_title"
DETAIL_TO_NUMBER_MODE = "detail_to_number"
OFFSET_AFTER_TITLE_MODE = "offset_after_title"
OFFSET_BEFORE_TITLE_MODE = "offset_before_title"
BETWEEN_NAMED_STEPS_COUNT_MODE = "boundary_title_gap_count"
SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = (
    "vertical_cards",
    "horizontal_cards",
    "two_column_cards",
)
SUPPORTED_ORDINAL_REFERENCES: Tuple[str, ...] = ("first", "interior", "final")

_CARD_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (61, 119, 175),
    (45, 142, 113),
    (193, 94, 77),
    (130, 105, 184),
    (198, 145, 61),
    (77, 132, 95),
    (59, 107, 145),
    (172, 88, 132),
)
@dataclass(frozen=True)
class _RenderParams:
    canvas_width: int
    canvas_height: int
    outer_margin_px: int
    header_height_px: int
    card_gap_px: int
    card_corner_radius_px: int
    card_outline_width_px: int
    number_badge_size_px: int
    title_font_size_px: int
    subtitle_font_size_px: int
    step_title_font_size_px: int
    step_detail_font_size_px: int
    step_meta_font_size_px: int


@dataclass(frozen=True)
class _StepSpec:
    step_id: str
    order_index: int
    step_number: int
    title: str
    detail: str
    owner: str
    status: str
    due_date: str
    tag: str
    accent_rgb: Tuple[int, int, int]


@dataclass(frozen=True)
class _RenderedStepList:
    image: Image.Image
    entities: List[Dict[str, Any]]
    card_traces: List[Dict[str, Any]]
    panel_bbox_px: List[float]
    title_bbox_px: List[float]
    layout_meta: Dict[str, Any]


_TASK_GROUP_DEFAULTS = get_scene_defaults(DOMAIN, SCENE)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_generation_rendering_prompt_defaults(
    _TASK_GROUP_DEFAULTS if isinstance(_TASK_GROUP_DEFAULTS, Mapping) else {},
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_pages_background_defaults(scene_id=SCENE)
POST_IMAGE_NOISE_DEFAULTS = load_pages_noise_defaults(scene_id=SCENE, apply_prob=0.0)


def _resolve_named_variant(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    supported: Sequence[str],
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    rng = spawn_rng(int(instance_seed), f"{TASK_NAMESPACE}.{namespace}")
    selected, probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        supported_variants=[str(value) for value in supported],
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
    )
    balanced = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=[str(value) for value in supported],
        balance_flag_key=str(balance_flag_key),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        sampling_namespace=f"{TASK_NAMESPACE}.{namespace}",
    )
    if str(balanced) != str(selected) and params.get(str(explicit_key)) is not None:
        return str(balanced), {str(key): (1.0 if str(key) == str(balanced) else 0.0) for key in supported}
    return str(balanced), dict(probabilities)


def _resolve_int_support(params: Mapping[str, Any], key: str, fallback: Sequence[int]) -> Tuple[int, ...]:
    raw_values = params.get(str(key), group_default(_GEN_DEFAULTS, str(key), fallback))
    values: List[int] = []
    for raw_value in raw_values:
        value = int(raw_value)
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError(f"{key} must not be empty for {TASK_NAMESPACE}")
    return tuple(int(value) for value in values)


def _resolve_step_count(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
) -> Tuple[int, Tuple[int, ...], Dict[str, float]]:
    support = _resolve_int_support(params, "step_count_support", (10, 11, 12, 13, 14, 15, 16))
    explicit = params.get("step_count")
    if explicit is not None:
        selected = int(explicit)
        if int(selected) not in set(support):
            raise ValueError(f"step_count must be in {support}")
        return int(selected), tuple(support), {str(int(selected)): 1.0}
    index = resolve_selection_index(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_NAMESPACE}.step_count",
    )
    selected = int(support[int(index) % len(support)])
    probability = 1.0 / float(len(support))
    return int(selected), tuple(support), {str(value): float(probability) for value in support}


def _ordinal_label(index: int, *, final_index: int) -> str:
    if int(index) == 0:
        return "first"
    if int(index) == int(final_index):
        return "final"
    number = int(index) + 1
    if 10 <= (number % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def _resolve_target_index(
    *,
    lookup_mode: str,
    params: Mapping[str, Any],
    step_count: int,
    instance_seed: int,
) -> Tuple[int, int | None, str, Dict[str, float], Dict[str, Any]]:
    """Select the rendered step operand for one neutral lookup mode.

    Public task files own query-id semantics. This helper only chooses the
    target step and optional source step needed by the scene-local program.
    """

    if str(lookup_mode) == AFTER_NAMED_TITLE_MODE:
        max_source = int(step_count) - 2
        if max_source < 0:
            raise ValueError("after-named lookups require at least two steps")
        explicit_source = params.get("source_step_index")
        if explicit_source is not None:
            source_index = int(explicit_source)
            if source_index < 0 or source_index > int(max_source):
                raise ValueError(f"source_step_index must be in 0..{max_source}")
        else:
            source_index = int(
                resolve_selection_index(
                    params=params,
                    instance_seed=int(instance_seed),
                    namespace=f"{TASK_NAMESPACE}.source_step_index.{step_count}",
                )
                % (int(max_source) + 1)
            )
        return int(source_index) + 1, int(source_index), "after_named", {"after_named": 1.0}, {}

    if str(lookup_mode) in {OFFSET_AFTER_TITLE_MODE, OFFSET_BEFORE_TITLE_MODE}:
        support = tuple(
            int(value)
            for value in _resolve_int_support(params, "relative_offset_support", (2, 3))
            if 1 <= int(value) < int(step_count)
        )
        if not support:
            raise ValueError("relative_offset_support must contain at least one value below step_count")
        explicit_offset = params.get("relative_offset")
        if explicit_offset is not None:
            relative_offset = int(explicit_offset)
            if int(relative_offset) not in set(support):
                raise ValueError(f"relative_offset must be in {support}")
            offset_probabilities = {str(int(relative_offset)): 1.0}
        else:
            offset_index = int(
                resolve_selection_index(
                    params=params,
                    instance_seed=int(instance_seed),
                    namespace=f"{TASK_NAMESPACE}.relative_offset.{lookup_mode}.{step_count}",
                )
                % len(support)
            )
            relative_offset = int(support[offset_index])
            offset_probabilities = {str(value): 1.0 / float(len(support)) for value in support}

        if str(lookup_mode) == OFFSET_AFTER_TITLE_MODE:
            max_source = int(step_count) - int(relative_offset) - 1
            if max_source < 0:
                raise ValueError("offset-after queries require source room after applying offset")
            explicit_source = params.get("source_step_index")
            if explicit_source is not None:
                source_index = int(explicit_source)
                if source_index < 0 or source_index > int(max_source):
                    raise ValueError(f"source_step_index must be in 0..{max_source}")
            else:
                source_index = int(
                    resolve_selection_index(
                        params=params,
                        instance_seed=int(instance_seed),
                        namespace=f"{TASK_NAMESPACE}.offset_source_index.after.{step_count}.{relative_offset}",
                    )
                    % (int(max_source) + 1)
                )
            target_index = int(source_index) + int(relative_offset)
            relation = "after"
        else:
            min_source = int(relative_offset)
            max_source = int(step_count) - 1
            explicit_source = params.get("source_step_index")
            if explicit_source is not None:
                source_index = int(explicit_source)
                if source_index < int(min_source) or source_index > int(max_source):
                    raise ValueError(f"source_step_index must be in {min_source}..{max_source}")
            else:
                source_index = int(min_source) + int(
                    resolve_selection_index(
                        params=params,
                        instance_seed=int(instance_seed),
                        namespace=f"{TASK_NAMESPACE}.offset_source_index.before.{step_count}.{relative_offset}",
                    )
                    % (int(max_source) - int(min_source) + 1)
                )
            target_index = int(source_index) - int(relative_offset)
            relation = "before"
        phrase_unit = "step" if int(relative_offset) == 1 else "steps"
        return (
            int(target_index),
            int(source_index),
            f"{int(relative_offset)} {phrase_unit} {relation}",
            dict(offset_probabilities),
            {
                "relative_offset": int(relative_offset),
                "relative_offset_support": [int(value) for value in support],
                "offset_relation": str(relation),
            },
        )

    if str(lookup_mode) == BETWEEN_NAMED_STEPS_COUNT_MODE:
        support = tuple(
            int(value)
            for value in _resolve_int_support(params, "between_count_support", (2, 3, 4, 5, 6, 7, 8))
            if 0 <= int(value) <= int(step_count) - 2
        )
        if not support:
            raise ValueError("between_count_support must contain at least one feasible value for step_count")
        explicit_between_count = params.get("between_count")
        if explicit_between_count is not None:
            between_count = int(explicit_between_count)
            if int(between_count) not in set(support):
                raise ValueError(f"between_count must be in {support}")
            count_probabilities = {str(int(between_count)): 1.0}
        else:
            count_index = int(
                resolve_selection_index(
                    params=params,
                    instance_seed=int(instance_seed),
                    namespace=f"{TASK_NAMESPACE}.between_count.{step_count}",
                )
                % len(support)
            )
            between_count = int(support[count_index])
            count_probabilities = {str(value): 1.0 / float(len(support)) for value in support}
        max_source = int(step_count) - int(between_count) - 2
        explicit_source = params.get("source_step_index")
        if explicit_source is not None:
            source_index = int(explicit_source)
            if source_index < 0 or source_index > int(max_source):
                raise ValueError(f"source_step_index must be in 0..{max_source}")
        else:
            source_index = int(
                resolve_selection_index(
                    params=params,
                    instance_seed=int(instance_seed),
                    namespace=f"{TASK_NAMESPACE}.between_source_index.{step_count}.{between_count}",
                )
                % (int(max_source) + 1)
            )
        target_index = int(source_index) + int(between_count) + 1
        return (
            int(target_index),
            int(source_index),
            f"{int(between_count)} between",
            dict(count_probabilities),
            {
                "between_count": int(between_count),
                "between_count_support": [int(value) for value in support],
            },
        )

    explicit_index = params.get("target_step_index")
    if explicit_index is not None:
        target_index = int(explicit_index)
        if target_index < 0 or target_index >= int(step_count):
            raise ValueError(f"target_step_index must be in 0..{int(step_count) - 1}")
        reference = _ordinal_label(int(target_index), final_index=int(step_count) - 1)
        return int(target_index), None, str(reference), {str(reference): 1.0}, {}

    if str(lookup_mode) in {DETAIL_TO_TITLE_MODE, DETAIL_TO_NUMBER_MODE}:
        target_index = int(
            resolve_selection_index(
                params=params,
                instance_seed=int(instance_seed),
                namespace=f"{TASK_NAMESPACE}.detail_lookup_step_index.{lookup_mode}.{step_count}",
            )
            % int(step_count)
        )
        probabilities = {str(index + 1): 1.0 / float(step_count) for index in range(int(step_count))}
        return (
            int(target_index),
            None,
            _ordinal_label(int(target_index), final_index=int(step_count) - 1),
            dict(probabilities),
            {},
        )

    ordinal_reference, ordinal_probabilities = _resolve_named_variant(
        params=params,
        instance_seed=int(instance_seed),
        supported=SUPPORTED_ORDINAL_REFERENCES,
        explicit_key="ordinal_reference",
        weights_key="ordinal_reference_weights",
        balance_flag_key="balanced_ordinal_reference_sampling",
        namespace=f"ordinal_reference.{lookup_mode}",
    )
    if str(ordinal_reference) == "first":
        target_index = 0
    elif str(ordinal_reference) == "final":
        target_index = int(step_count) - 1
    else:
        interior_count = max(1, int(step_count) - 2)
        target_index = 1 + int(
            resolve_selection_index(
                params=params,
                instance_seed=int(instance_seed),
                namespace=f"{TASK_NAMESPACE}.interior_step_index.{lookup_mode}.{step_count}",
            )
            % int(interior_count)
        )
    return (
        int(target_index),
        None,
        _ordinal_label(int(target_index), final_index=int(step_count) - 1),
        dict(ordinal_probabilities),
        {},
    )


def _resolve_render_params(params: Mapping[str, Any]) -> _RenderParams:
    def _int_value(key: str, fallback: int, *, minimum: int = 1) -> int:
        return max(int(minimum), int(params.get(key, group_default(_RENDER_DEFAULTS, key, fallback))))

    return _RenderParams(
        canvas_width=_int_value("canvas_width", 1120, minimum=320),
        canvas_height=_int_value("canvas_height", 980, minimum=320),
        outer_margin_px=_int_value("outer_margin_px", 30, minimum=0),
        header_height_px=_int_value("header_height_px", 74, minimum=40),
        card_gap_px=_int_value("card_gap_px", 8, minimum=4),
        card_corner_radius_px=_int_value("card_corner_radius_px", 10, minimum=0),
        card_outline_width_px=_int_value("card_outline_width_px", 2, minimum=1),
        number_badge_size_px=_int_value("number_badge_size_px", 30, minimum=20),
        title_font_size_px=_int_value("title_font_size_px", 28, minimum=14),
        subtitle_font_size_px=_int_value("subtitle_font_size_px", 15, minimum=10),
        step_title_font_size_px=_int_value("step_title_font_size_px", 18, minimum=12),
        step_detail_font_size_px=_int_value("step_detail_font_size_px", 14, minimum=10),
        step_meta_font_size_px=_int_value("step_meta_font_size_px", 11, minimum=8),
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
    trace: bool = True,
) -> List[float]:
    if bool(trace):
        draw_text_traced(draw,(float(xy[0]), float(xy[1])), str(text), fill=fill, font=font, role="readout", required=False)
    else:
        draw.text((float(xy[0]), float(xy[1])), str(text), fill=fill, font=font)
    return _text_bbox(draw, xy, str(text), font)


def _draw_inline_field(
    draw: ImageDraw.ImageDraw,
    *,
    label: str,
    value: str,
    x: float,
    y: float,
    width: float,
    label_font: Any,
    value_font: Any,
    label_fill: Tuple[int, int, int],
    value_fill: Tuple[int, int, int],
    trace_label: bool = False,
    trace_value: bool = True,
) -> Tuple[List[float], List[float]]:
    label_text = f"{str(label)}:"
    label_bbox = _draw_text(draw, (float(x), float(y)), label_text, label_font, label_fill, trace=bool(trace_label))
    label_w = max(0.0, float(label_bbox[2]) - float(label_bbox[0]))
    value_x = float(x) + label_w + 5.0
    value_bbox = _draw_text(draw, (value_x, float(y)), str(value), value_font, value_fill, trace=bool(trace_value))
    if float(value_bbox[2]) > float(x) + float(width):
        value_bbox[2] = float(x) + float(width)
    return label_bbox, value_bbox


def _blend_rgb(color_a: Sequence[int], color_b: Sequence[int], weight_b: float) -> Tuple[int, int, int]:
    weight = max(0.0, min(1.0, float(weight_b)))
    return tuple(
        int(round((float(color_a[index]) * (1.0 - weight)) + (float(color_b[index]) * weight)))
        for index in range(3)
    )


def _layout_card_bboxes(
    *,
    scene_variant: str,
    step_count: int,
    render_params: _RenderParams,
) -> Tuple[List[List[float]], Dict[str, Any]]:
    width = int(render_params.canvas_width)
    height = int(render_params.canvas_height)
    margin = int(render_params.outer_margin_px)
    gap = int(render_params.card_gap_px)
    top = float(margin + render_params.header_height_px)
    left = float(margin)
    right = float(width - margin)
    bottom = float(height - margin)
    inner_w = max(1.0, right - left)
    inner_h = max(1.0, bottom - top)
    bboxes: List[List[float]] = []

    if str(scene_variant) == "vertical_cards":
        card_h = (inner_h - (float(step_count - 1) * float(gap))) / float(step_count)
        for index in range(int(step_count)):
            y0 = top + (float(index) * (card_h + float(gap)))
            bboxes.append([left, y0, right, y0 + card_h])
        return bboxes, {"layout_columns": 1, "layout_rows": int(step_count)}

    if str(scene_variant) == "two_column_cards":
        columns = 2
        rows = int((int(step_count) + 1) // 2)
    else:
        columns = 3 if int(step_count) <= 12 else 4
        rows = int((int(step_count) + int(columns) - 1) // int(columns))

    card_w = (inner_w - (float(columns - 1) * float(gap))) / float(columns)
    card_h = (inner_h - (float(rows - 1) * float(gap))) / float(rows)
    for index in range(int(step_count)):
        row = int(index) // int(columns)
        col = int(index) % int(columns)
        x0 = left + (float(col) * (card_w + float(gap)))
        y0 = top + (float(row) * (card_h + float(gap)))
        bboxes.append([x0, y0, x0 + card_w, y0 + card_h])
    return bboxes, {"layout_columns": int(columns), "layout_rows": int(rows)}


def _sample_due_dates(rng: Any, *, count: int) -> Tuple[List[str], Dict[str, Any]]:
    months = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
    candidates = [f"{month} {day:02d}" for month in months for day in range(2, 29)]
    indices = list(range(len(candidates)))
    rng.shuffle(indices)
    selected = [str(candidates[index]) for index in indices[: int(count)]]
    return selected, {
        "role": "step_list_due_date",
        "source_kind": "synthetic_calendar_date",
        "candidate_count": len(candidates),
        "values": list(selected),
    }


def _build_steps(*, step_count: int, instance_seed: int) -> Tuple[List[_StepSpec], str, str, Dict[str, Any]]:
    """Sample visible step records used by both rendering and answers.

    The sampled text resource batches are the symbolic source of truth. Later
    task binding reads title/detail/number values from these records and uses
    the rendered bboxes for annotation witnesses.
    """

    rng = spawn_rng(int(instance_seed), f"{TASK_NAMESPACE}.steps")
    title_batch = sample_page_context_batch(
        rng,
        role="step_list_panel_title",
        count=1,
        manifest_names=("phrases/headlines.txt",),
        max_chars=28,
    )
    subtitle_batch = sample_page_context_batch(
        rng,
        role="step_list_panel_subtitle",
        count=1,
        manifest_names=("phrases/captions.txt", "phrases/legend_notes.txt"),
        max_chars=40,
    )
    step_title_batch = sample_page_label_batch(
        rng,
        role="step_list_step_title",
        count=int(step_count),
        manifest_name="panel_titles/technical_topics.txt",
        min_chars=3,
        max_chars=10,
        allow_spaces=True,
        allow_punctuation=False,
    )
    detail_batch = sample_page_context_batch(
        rng,
        role="step_list_step_detail",
        count=int(step_count),
        manifest_names=("phrases/callout_phrases.txt",),
        max_chars=16,
    )
    owner_batch = sample_page_label_batch(
        rng,
        role="step_list_owner",
        count=int(step_count),
        manifest_name="people/first_names_ssa.txt",
        min_chars=3,
        max_chars=9,
        allow_spaces=False,
        allow_punctuation=False,
    )
    status_batch = sample_page_label_batch(
        rng,
        role="step_list_status",
        count=int(step_count),
        manifest_name="categories/status_labels.txt",
        min_chars=4,
        max_chars=11,
        allow_spaces=True,
        allow_punctuation=False,
    )
    tag_batch = sample_page_label_batch(
        rng,
        role="step_list_tag",
        count=int(step_count),
        manifest_name="categories/priority_labels.txt",
        min_chars=3,
        max_chars=9,
        allow_spaces=True,
        allow_punctuation=False,
    )
    due_dates, due_date_meta = _sample_due_dates(rng, count=int(step_count))
    titles = list(step_title_batch.values)
    details = list(detail_batch.values)
    owners = list(owner_batch.values)
    statuses = list(status_batch.values)
    tags = list(tag_batch.values)
    panel_title = str(title_batch.values[0])
    panel_subtitle = str(subtitle_batch.values[0])
    steps: List[_StepSpec] = []
    color_offset = int(rng.randrange(len(_CARD_PALETTE)))
    for index in range(int(step_count)):
        accent = _CARD_PALETTE[(int(index) + int(color_offset)) % len(_CARD_PALETTE)]
        steps.append(
            _StepSpec(
                step_id=f"step_{index + 1}",
                order_index=int(index),
                step_number=int(index) + 1,
                title=str(titles[int(index)]),
                detail=str(details[int(index)]),
                owner=str(owners[int(index)]),
                status=str(statuses[int(index)]),
                due_date=str(due_dates[int(index)]),
                tag=str(tags[int(index)]),
                accent_rgb=tuple(int(channel) for channel in accent),
            )
        )
    return (
        steps,
        panel_title,
        panel_subtitle,
        {
            **page_text_resource_metadata(
                title_batch,
                subtitle_batch,
                step_title_batch,
                detail_batch,
                owner_batch,
                status_batch,
                tag_batch,
            ),
            "synthetic_due_dates": dict(due_date_meta),
        },
    )


def _render_step_list(
    background: Image.Image,
    *,
    steps: Sequence[_StepSpec],
    panel_title: str,
    panel_subtitle: str,
    scene_variant: str,
    render_params: _RenderParams,
    trace_title_step_ids: Sequence[str] = (),
    trace_detail_step_ids: Sequence[str] = (),
    trace_number_step_ids: Sequence[str] = (),
) -> _RenderedStepList:
    """Draw the shared step-list page and collect final visual witnesses.

    Every public task uses this single renderer. Card, number, title, and
    detail boxes are measured after final font fitting so annotations bind to
    exactly the displayed text geometry.
    """

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    width = int(render_params.canvas_width)
    height = int(render_params.canvas_height)
    margin = int(render_params.outer_margin_px)
    panel_bbox = [float(margin), float(margin), float(width - margin), float(height - margin)]
    card_fill = (255, 255, 255)
    panel_fill = (249, 250, 250)
    panel_outline = (205, 212, 220)
    text_rgb = (35, 42, 50)
    muted_rgb = (91, 101, 113)

    draw.rounded_rectangle(
        tuple(panel_bbox),
        radius=18,
        fill=panel_fill,
        outline=panel_outline,
        width=2,
    )
    title_font = load_font(int(render_params.title_font_size_px), bold=True)
    subtitle_font = load_font(int(render_params.subtitle_font_size_px), bold=False)
    title_xy = (float(margin + 24), float(margin + 18))
    title_bbox = _draw_text(draw, title_xy, str(panel_title), title_font, text_rgb, trace=False)
    _draw_text(draw, (title_xy[0], title_xy[1] + 38.0), str(panel_subtitle), subtitle_font, muted_rgb, trace=False)

    card_bboxes, layout_meta = _layout_card_bboxes(
        scene_variant=str(scene_variant),
        step_count=len(steps),
        render_params=render_params,
    )
    traced_title_ids = {str(value) for value in trace_title_step_ids}
    traced_detail_ids = {str(value) for value in trace_detail_step_ids}
    traced_number_ids = {str(value) for value in trace_number_step_ids}
    card_traces: List[Dict[str, Any]] = []
    entities: List[Dict[str, Any]] = []
    for step, card_bbox in zip(steps, card_bboxes):
        trace_title = str(step.step_id) in traced_title_ids
        trace_detail = str(step.step_id) in traced_detail_ids
        trace_number = str(step.step_id) in traced_number_ids
        x0, y0, x1, y1 = [float(value) for value in card_bbox]
        accent = tuple(int(channel) for channel in step.accent_rgb)
        local_fill = _blend_rgb(card_fill, accent, 0.035)
        draw.rounded_rectangle(
            (x0, y0, x1, y1),
            radius=int(render_params.card_corner_radius_px),
            fill=local_fill,
            outline=panel_outline,
            width=int(render_params.card_outline_width_px),
        )
        draw.rounded_rectangle(
            (x0, y0, x1, y0 + 7.0),
            radius=int(render_params.card_corner_radius_px),
            fill=accent,
        )

        card_h = max(1.0, float(y1) - float(y0))
        compact_row = str(scene_variant) == "vertical_cards" or float(card_h) < 112.0
        badge_size = float(render_params.number_badge_size_px if not compact_row else max(24, render_params.number_badge_size_px - 4))
        badge_x0 = x0 + 16.0
        badge_y0 = y0 + max(7.0, (float(card_h) - float(badge_size)) * 0.5)
        badge_bbox = [badge_x0, badge_y0, badge_x0 + badge_size, badge_y0 + badge_size]
        draw.ellipse(tuple(badge_bbox), fill=accent, outline=_blend_rgb(accent, (0, 0, 0), 0.18), width=2)
        number_font = fit_font_to_box(
            draw,
            text=str(step.step_number),
            max_width=badge_size - 8.0,
            max_height=badge_size - 8.0,
            bold=True,
            min_size_px=12,
            max_size_px=22,
            fill_ratio=0.95,
        )
        number_bbox = list(
            draw_text_centered(
                draw,
                text=str(step.step_number),
                center=(badge_x0 + (badge_size * 0.5), badge_y0 + (badge_size * 0.5)),
                font=number_font,
                fill=(255, 255, 255),
                stroke_width=0,
                role="readout",
                required=False,
                trace=trace_number,
            )
        )

        text_left = badge_x0 + badge_size + 16.0
        text_right = x1 - 18.0
        field_label_font = load_font(int(render_params.step_meta_font_size_px), bold=True)
        meta_value_font = load_font(int(render_params.step_meta_font_size_px), bold=False)
        field_label_rgb = (77, 88, 101)
        meta_rgb = (67, 78, 91)

        if compact_row:
            available_w = max(120.0, float(text_right) - float(text_left))
            if str(scene_variant) == "vertical_cards":
                title_w = available_w * 0.22
                detail_w = available_w * 0.24
                owner_w = available_w * 0.13
                status_w = available_w * 0.14
                due_w = available_w * 0.11
                tag_w = max(58.0, available_w - title_w - detail_w - owner_w - status_w - due_w)
                line_y = y0 + max(7.0, (float(card_h) - 15.0) * 0.5)
                title_font = fit_font_to_box(
                    draw,
                    text=str(step.title),
                    max_width=max(35.0, title_w - 42.0),
                    max_height=18.0,
                    bold=True,
                    min_size_px=10,
                    max_size_px=max(11, int(render_params.step_title_font_size_px) - 2),
                    fill_ratio=0.95,
                )
                detail_font = fit_font_to_box(
                    draw,
                    text=str(step.detail),
                    max_width=max(35.0, detail_w - 46.0),
                    max_height=16.0,
                    bold=False,
                    min_size_px=9,
                    max_size_px=max(10, int(render_params.step_detail_font_size_px) - 1),
                    fill_ratio=0.95,
                )
                _, title_bbox_step = _draw_inline_field(
                    draw,
                    label="Title",
                    value=str(step.title),
                    x=text_left,
                    y=line_y,
                    width=title_w,
                    label_font=field_label_font,
                    value_font=title_font,
                    label_fill=field_label_rgb,
                    value_fill=text_rgb,
                    trace_value=trace_title,
                )
                _, detail_bbox = _draw_inline_field(
                    draw,
                    label="Detail",
                    value=str(step.detail),
                    x=text_left + title_w,
                    y=line_y,
                    width=detail_w,
                    label_font=field_label_font,
                    value_font=detail_font,
                    label_fill=field_label_rgb,
                    value_fill=muted_rgb,
                    trace_value=trace_detail,
                )
                _, owner_bbox = _draw_inline_field(
                    draw,
                    label="Owner",
                    value=str(step.owner),
                    x=text_left + title_w + detail_w,
                    y=line_y,
                    width=owner_w,
                    label_font=field_label_font,
                    value_font=meta_value_font,
                    label_fill=field_label_rgb,
                    value_fill=meta_rgb,
                    trace_value=False,
                )
                _, status_bbox = _draw_inline_field(
                    draw,
                    label="Status",
                    value=str(step.status),
                    x=text_left + title_w + detail_w + owner_w,
                    y=line_y,
                    width=status_w,
                    label_font=field_label_font,
                    value_font=meta_value_font,
                    label_fill=field_label_rgb,
                    value_fill=meta_rgb,
                    trace_value=False,
                )
                _, due_bbox = _draw_inline_field(
                    draw,
                    label="Due",
                    value=str(step.due_date),
                    x=text_left + title_w + detail_w + owner_w + status_w,
                    y=line_y,
                    width=due_w,
                    label_font=field_label_font,
                    value_font=meta_value_font,
                    label_fill=field_label_rgb,
                    value_fill=meta_rgb,
                    trace_value=False,
                )
                _, tag_bbox = _draw_inline_field(
                    draw,
                    label="Tag",
                    value=str(step.tag),
                    x=text_left + title_w + detail_w + owner_w + status_w + due_w,
                    y=line_y,
                    width=tag_w,
                    label_font=field_label_font,
                    value_font=meta_value_font,
                    label_fill=field_label_rgb,
                    value_fill=meta_rgb,
                    trace_value=False,
                )
            else:
                title_w = available_w * 0.48
                detail_w = available_w - title_w
                meta_w = available_w * 0.25
                line_y = y0 + max(18.0, (float(card_h) - 32.0) * 0.5)
                meta_y = line_y + 20.0
                title_font = fit_font_to_box(
                    draw,
                    text=str(step.title),
                    max_width=max(35.0, title_w - 42.0),
                    max_height=18.0,
                    bold=True,
                    min_size_px=10,
                    max_size_px=max(11, int(render_params.step_title_font_size_px) - 2),
                    fill_ratio=0.95,
                )
                detail_font = fit_font_to_box(
                    draw,
                    text=str(step.detail),
                    max_width=max(35.0, detail_w - 46.0),
                    max_height=16.0,
                    bold=False,
                    min_size_px=9,
                    max_size_px=max(10, int(render_params.step_detail_font_size_px) - 1),
                    fill_ratio=0.95,
                )
                _, title_bbox_step = _draw_inline_field(
                    draw,
                    label="Title",
                    value=str(step.title),
                    x=text_left,
                    y=line_y,
                    width=title_w,
                    label_font=field_label_font,
                    value_font=title_font,
                    label_fill=field_label_rgb,
                    value_fill=text_rgb,
                    trace_value=trace_title,
                )
                _, detail_bbox = _draw_inline_field(
                    draw,
                    label="Detail",
                    value=str(step.detail),
                    x=text_left + title_w,
                    y=line_y,
                    width=detail_w,
                    label_font=field_label_font,
                    value_font=detail_font,
                    label_fill=field_label_rgb,
                    value_fill=muted_rgb,
                    trace_value=trace_detail,
                )
                _, owner_bbox = _draw_inline_field(
                    draw,
                    label="Owner",
                    value=str(step.owner),
                    x=text_left,
                    y=meta_y,
                    width=meta_w,
                    label_font=field_label_font,
                    value_font=meta_value_font,
                    label_fill=field_label_rgb,
                    value_fill=meta_rgb,
                    trace_value=False,
                )
                _, status_bbox = _draw_inline_field(
                    draw,
                    label="Status",
                    value=str(step.status),
                    x=text_left + meta_w,
                    y=meta_y,
                    width=meta_w,
                    label_font=field_label_font,
                    value_font=meta_value_font,
                    label_fill=field_label_rgb,
                    value_fill=meta_rgb,
                    trace_value=False,
                )
                _, due_bbox = _draw_inline_field(
                    draw,
                    label="Due",
                    value=str(step.due_date),
                    x=text_left + (2.0 * meta_w),
                    y=meta_y,
                    width=meta_w,
                    label_font=field_label_font,
                    value_font=meta_value_font,
                    label_fill=field_label_rgb,
                    value_fill=meta_rgb,
                    trace_value=False,
                )
                _, tag_bbox = _draw_inline_field(
                    draw,
                    label="Tag",
                    value=str(step.tag),
                    x=text_left + (3.0 * meta_w),
                    y=meta_y,
                    width=meta_w,
                    label_font=field_label_font,
                    value_font=meta_value_font,
                    label_fill=field_label_rgb,
                    value_fill=meta_rgb,
                    trace_value=False,
                )
        else:
            available_w = max(80.0, float(text_right) - float(text_left))
            narrow_card = available_w < 300.0
            title_font = fit_font_to_box(
                draw,
                text=str(step.title),
                max_width=max(30.0, available_w - 44.0),
                max_height=25.0,
                bold=True,
                min_size_px=11,
                max_size_px=int(render_params.step_title_font_size_px),
                fill_ratio=0.97,
            )
            detail_font = fit_font_to_box(
                draw,
                text=str(step.detail),
                max_width=max(30.0, available_w - 50.0),
                max_height=22.0,
                bold=False,
                min_size_px=9,
                max_size_px=int(render_params.step_detail_font_size_px),
                fill_ratio=0.97,
            )
            text_top = y0 + 17.0
            if float(card_h) > 150.0:
                text_top = y0 + 24.0
            _, title_bbox_step = _draw_inline_field(
                draw,
                label="Title",
                value=str(step.title),
                x=text_left,
                y=text_top,
                width=available_w,
                label_font=field_label_font,
                value_font=title_font,
                label_fill=field_label_rgb,
                value_fill=text_rgb,
                trace_value=trace_title,
            )
            _, detail_bbox = _draw_inline_field(
                draw,
                label="Detail",
                value=str(step.detail),
                x=text_left,
                y=text_top + 24.0,
                width=available_w,
                label_font=field_label_font,
                value_font=detail_font,
                label_fill=field_label_rgb,
                value_fill=muted_rgb,
                trace_value=trace_detail,
            )
            meta_y = text_top + 50.0
            if narrow_card:
                _, owner_bbox = _draw_inline_field(
                    draw,
                    label="Owner",
                    value=str(step.owner),
                    x=text_left,
                    y=meta_y,
                    width=available_w,
                    label_font=field_label_font,
                    value_font=meta_value_font,
                    label_fill=field_label_rgb,
                    value_fill=meta_rgb,
                    trace_value=False,
                )
                _, status_bbox = _draw_inline_field(
                    draw,
                    label="Status",
                    value=str(step.status),
                    x=text_left,
                    y=meta_y + 18.0,
                    width=available_w,
                    label_font=field_label_font,
                    value_font=meta_value_font,
                    label_fill=field_label_rgb,
                    value_fill=meta_rgb,
                    trace_value=False,
                )
                _, due_bbox = _draw_inline_field(
                    draw,
                    label="Due",
                    value=str(step.due_date),
                    x=text_left,
                    y=meta_y + 36.0,
                    width=available_w,
                    label_font=field_label_font,
                    value_font=meta_value_font,
                    label_fill=field_label_rgb,
                    value_fill=meta_rgb,
                    trace_value=False,
                )
                _, tag_bbox = _draw_inline_field(
                    draw,
                    label="Tag",
                    value=str(step.tag),
                    x=text_left,
                    y=meta_y + 54.0,
                    width=available_w,
                    label_font=field_label_font,
                    value_font=meta_value_font,
                    label_fill=field_label_rgb,
                    value_fill=meta_rgb,
                    trace_value=False,
                )
            else:
                half_w = available_w * 0.5
                _, owner_bbox = _draw_inline_field(
                    draw,
                    label="Owner",
                    value=str(step.owner),
                    x=text_left,
                    y=meta_y,
                    width=half_w - 8.0,
                    label_font=field_label_font,
                    value_font=meta_value_font,
                    label_fill=field_label_rgb,
                    value_fill=meta_rgb,
                    trace_value=False,
                )
                _, status_bbox = _draw_inline_field(
                    draw,
                    label="Status",
                    value=str(step.status),
                    x=text_left + half_w,
                    y=meta_y,
                    width=half_w,
                    label_font=field_label_font,
                    value_font=meta_value_font,
                    label_fill=field_label_rgb,
                    value_fill=meta_rgb,
                    trace_value=False,
                )
                _, due_bbox = _draw_inline_field(
                    draw,
                    label="Due",
                    value=str(step.due_date),
                    x=text_left,
                    y=meta_y + 20.0,
                    width=half_w - 8.0,
                    label_font=field_label_font,
                    value_font=meta_value_font,
                    label_fill=field_label_rgb,
                    value_fill=meta_rgb,
                    trace_value=False,
                )
                _, tag_bbox = _draw_inline_field(
                    draw,
                    label="Tag",
                    value=str(step.tag),
                    x=text_left + half_w,
                    y=meta_y + 20.0,
                    width=half_w,
                    label_font=field_label_font,
                    value_font=meta_value_font,
                    label_fill=field_label_rgb,
                    value_fill=meta_rgb,
                    trace_value=False,
                )

        trace = {
            "step_id": str(step.step_id),
            "order_index": int(step.order_index),
            "step_number": int(step.step_number),
            "title": str(step.title),
            "detail": str(step.detail),
            "owner": str(step.owner),
            "status": str(step.status),
            "due_date": str(step.due_date),
            "tag": str(step.tag),
            "card_bbox_px": [float(value) for value in card_bbox],
            "number_badge_bbox_px": [float(value) for value in badge_bbox],
            "number_bbox_px": [float(value) for value in number_bbox],
            "title_bbox_px": [float(value) for value in title_bbox_step],
            "detail_bbox_px": [float(value) for value in detail_bbox],
            "owner_bbox_px": [float(value) for value in owner_bbox],
            "status_bbox_px": [float(value) for value in status_bbox],
            "due_date_bbox_px": [float(value) for value in due_bbox],
            "tag_bbox_px": [float(value) for value in tag_bbox],
            "accent_rgb": [int(channel) for channel in accent],
        }
        entity = {
            "id": str(step.step_id),
            "type": "step_list_card",
            "bbox_px": [float(value) for value in card_bbox],
            "attrs": {
                "order_index": int(step.order_index),
                "step_number": int(step.step_number),
                "title": str(step.title),
                "detail": str(step.detail),
                "owner": str(step.owner),
                "status": str(step.status),
                "due_date": str(step.due_date),
                "tag": str(step.tag),
            },
        }
        card_traces.append(trace)
        entities.append(entity)

    return _RenderedStepList(
        image=image,
        entities=entities,
        card_traces=card_traces,
        panel_bbox_px=list(panel_bbox),
        title_bbox_px=list(title_bbox),
        layout_meta={
            "scene_variant": str(scene_variant),
            "card_count": int(len(steps)),
            **dict(layout_meta),
        },
    )


def _bbox_maps(
    card_traces: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]]]:
    card_map = {str(card["step_id"]): [float(value) for value in card["card_bbox_px"]] for card in card_traces}
    number_map = {str(card["step_id"]): [float(value) for value in card["number_bbox_px"]] for card in card_traces}
    number_badge_map = {
        str(card["step_id"]): [float(value) for value in card["number_badge_bbox_px"]] for card in card_traces
    }
    title_map = {str(card["step_id"]): [float(value) for value in card["title_bbox_px"]] for card in card_traces}
    detail_map = {str(card["step_id"]): [float(value) for value in card["detail_bbox_px"]] for card in card_traces}
    return card_map, number_map, number_badge_map, title_map, detail_map


def _target_annotation_role(lookup_mode: str) -> str:
    if str(lookup_mode) in {
        ORDINAL_TITLE_MODE,
        AFTER_NAMED_TITLE_MODE,
        DETAIL_TO_TITLE_MODE,
        OFFSET_AFTER_TITLE_MODE,
        OFFSET_BEFORE_TITLE_MODE,
    }:
        return "target_title"
    if str(lookup_mode) == ORDINAL_DETAIL_MODE:
        return "target_detail"
    if str(lookup_mode) == DETAIL_TO_NUMBER_MODE:
        return "target_number"
    if str(lookup_mode) == BETWEEN_NAMED_STEPS_COUNT_MODE:
        return "boundary_titles"
    raise ValueError(f"unsupported lookup_mode: {lookup_mode}")


def _target_annotation_bbox(
    *,
    lookup_mode: str,
    target_step_id: str,
    number_badge_bbox_map: Mapping[str, Sequence[float]],
    title_bbox_map: Mapping[str, Sequence[float]],
    detail_bbox_map: Mapping[str, Sequence[float]],
) -> List[float]:
    if str(lookup_mode) in {
        ORDINAL_TITLE_MODE,
        AFTER_NAMED_TITLE_MODE,
        DETAIL_TO_TITLE_MODE,
        OFFSET_AFTER_TITLE_MODE,
        OFFSET_BEFORE_TITLE_MODE,
    }:
        return [float(value) for value in title_bbox_map[str(target_step_id)]]
    if str(lookup_mode) == ORDINAL_DETAIL_MODE:
        return [float(value) for value in detail_bbox_map[str(target_step_id)]]
    if str(lookup_mode) == DETAIL_TO_NUMBER_MODE:
        return [float(value) for value in number_badge_bbox_map[str(target_step_id)]]
    raise ValueError(f"unsupported lookup_mode: {lookup_mode}")


def _annotation_bbox_map(
    *,
    lookup_mode: str,
    target_step_id: str,
    source_step_id: str | None,
    number_badge_bbox_map: Mapping[str, Sequence[float]],
    title_bbox_map: Mapping[str, Sequence[float]],
    detail_bbox_map: Mapping[str, Sequence[float]],
) -> Dict[str, List[float]]:
    """Bind visual title/detail/number witnesses for each neutral lookup mode.

    The private lifecycle branches only on scene-internal modes; public task and
    query identities stay in the wrapper files that select those modes.
    """

    if str(lookup_mode) == ORDINAL_TITLE_MODE:
        return {"target_title": [float(value) for value in title_bbox_map[str(target_step_id)]]}
    if str(lookup_mode) == ORDINAL_DETAIL_MODE:
        return {"target_detail": [float(value) for value in detail_bbox_map[str(target_step_id)]]}
    if str(lookup_mode) == AFTER_NAMED_TITLE_MODE:
        if source_step_id is None:
            raise ValueError("after-named lookups require source_step_id")
        return {
            "source_title": [float(value) for value in title_bbox_map[str(source_step_id)]],
            "target_title": [float(value) for value in title_bbox_map[str(target_step_id)]],
        }
    if str(lookup_mode) in {OFFSET_AFTER_TITLE_MODE, OFFSET_BEFORE_TITLE_MODE}:
        if source_step_id is None:
            raise ValueError("offset lookups require source_step_id")
        return {
            "source_title": [float(value) for value in title_bbox_map[str(source_step_id)]],
            "target_title": [float(value) for value in title_bbox_map[str(target_step_id)]],
        }
    if str(lookup_mode) == BETWEEN_NAMED_STEPS_COUNT_MODE:
        if source_step_id is None:
            raise ValueError("between-count lookups require source_step_id")
        return {
            "first_named_title": [float(value) for value in title_bbox_map[str(source_step_id)]],
            "second_named_title": [float(value) for value in title_bbox_map[str(target_step_id)]],
        }
    if str(lookup_mode) == DETAIL_TO_TITLE_MODE:
        return {
            "source_detail": [float(value) for value in detail_bbox_map[str(target_step_id)]],
            "target_title": [float(value) for value in title_bbox_map[str(target_step_id)]],
        }
    if str(lookup_mode) == DETAIL_TO_NUMBER_MODE:
        return {
            "source_detail": [float(value) for value in detail_bbox_map[str(target_step_id)]],
            "target_number": [float(value) for value in number_badge_bbox_map[str(target_step_id)]],
        }
    raise ValueError(f"unsupported lookup_mode: {lookup_mode}")


def select_public_branch(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    supported: tuple[str, ...],
    default: str,
    public_task: str,
) -> tuple[str, Dict[str, float], Dict[str, Any]]:
    """Resolve the caller-selected public branch through the shared policy."""

    branch, probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in supported),
        default_query_id=str(default),
        task_id=str(public_task),
    )
    return str(branch), dict(probabilities), dict(task_params)


def _answer_value_for_mode(lookup_mode: str, target_step: _StepSpec, source_step: _StepSpec | None = None) -> str | int:
    if str(lookup_mode) in {
        ORDINAL_TITLE_MODE,
        AFTER_NAMED_TITLE_MODE,
        DETAIL_TO_TITLE_MODE,
        OFFSET_AFTER_TITLE_MODE,
        OFFSET_BEFORE_TITLE_MODE,
    }:
        return str(target_step.title)
    if str(lookup_mode) == DETAIL_TO_NUMBER_MODE:
        return str(target_step.step_number)
    if str(lookup_mode) == ORDINAL_DETAIL_MODE:
        return str(target_step.detail)
    if str(lookup_mode) == BETWEEN_NAMED_STEPS_COUNT_MODE:
        if source_step is None:
            raise ValueError("between-count answers require source_step")
        return abs(int(target_step.order_index) - int(source_step.order_index)) - 1
    raise ValueError(f"unsupported lookup_mode: {lookup_mode}")


def _step_payload(step: _StepSpec) -> Dict[str, Any]:
    return {
        "step_id": str(step.step_id),
        "order_index": int(step.order_index),
        "step_number": int(step.step_number),
        "title": str(step.title),
        "detail": str(step.detail),
        "owner": str(step.owner),
        "status": str(step.status),
        "due_date": str(step.due_date),
        "tag": str(step.tag),
    }


def build_step_list_response(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    lookup_mode: str,
    source_query_id: str,
    prompt_query_key: str,
    question_format: str,
) -> TaskOutput:
    """Build one complete public step-list response."""

    source_query = str(source_query_id)

    scene_variant, scene_variant_probabilities = _resolve_named_variant(
        params=params,
        instance_seed=int(instance_seed),
        supported=SUPPORTED_SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        namespace="scene_variant",
    )
    step_count, step_count_support, step_count_probabilities = _resolve_step_count(
        params,
        instance_seed=int(instance_seed),
    )
    target_index, source_index, step_reference, operand_probabilities, operand_params = _resolve_target_index(
        lookup_mode=str(lookup_mode),
        params=params,
        step_count=int(step_count),
        instance_seed=int(instance_seed),
    )
    steps, panel_title, panel_subtitle, page_text_resources = _build_steps(
        step_count=int(step_count),
        instance_seed=int(instance_seed),
    )
    target_step = steps[int(target_index)]
    source_step = steps[int(source_index)] if source_index is not None else None
    answer_value = _answer_value_for_mode(str(lookup_mode), target_step, source_step)
    trace_title_step_ids: List[str] = []
    trace_detail_step_ids: List[str] = []
    trace_number_step_ids: List[str] = []
    if str(lookup_mode) == ORDINAL_TITLE_MODE:
        trace_title_step_ids.append(str(target_step.step_id))
    elif str(lookup_mode) == ORDINAL_DETAIL_MODE:
        trace_detail_step_ids.append(str(target_step.step_id))
    elif str(lookup_mode) == AFTER_NAMED_TITLE_MODE:
        if source_step is not None:
            trace_title_step_ids.append(str(source_step.step_id))
        trace_title_step_ids.append(str(target_step.step_id))
    elif str(lookup_mode) in {OFFSET_AFTER_TITLE_MODE, OFFSET_BEFORE_TITLE_MODE, BETWEEN_NAMED_STEPS_COUNT_MODE}:
        if source_step is not None:
            trace_title_step_ids.append(str(source_step.step_id))
        trace_title_step_ids.append(str(target_step.step_id))
    elif str(lookup_mode) == DETAIL_TO_TITLE_MODE:
        trace_detail_step_ids.append(str(target_step.step_id))
        trace_title_step_ids.append(str(target_step.step_id))
    elif str(lookup_mode) == DETAIL_TO_NUMBER_MODE:
        trace_detail_step_ids.append(str(target_step.step_id))
        trace_number_step_ids.append(str(target_step.step_id))

    render_params = _resolve_render_params(params)
    background, background_meta = make_background_canvas(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_BACKGROUND_DEFAULTS,
    )
    rendered = _render_step_list(
        background,
        steps=steps,
        panel_title=str(panel_title),
        panel_subtitle=str(panel_subtitle),
        scene_variant=str(scene_variant),
        render_params=render_params,
        trace_title_step_ids=tuple(trace_title_step_ids),
        trace_detail_step_ids=tuple(trace_detail_step_ids),
        trace_number_step_ids=tuple(trace_number_step_ids),
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    card_bbox_map, number_bbox_map, number_badge_bbox_map, title_bbox_map, detail_bbox_map = _bbox_maps(rendered.card_traces)
    reasoning_bbox_map = _annotation_bbox_map(
        lookup_mode=str(lookup_mode),
        target_step_id=str(target_step.step_id),
        source_step_id=str(source_step.step_id) if source_step is not None else None,
        number_badge_bbox_map=number_badge_bbox_map,
        title_bbox_map=title_bbox_map,
        detail_bbox_map=detail_bbox_map,
    )
    annotation_role = _target_annotation_role(str(lookup_mode))
    scalar_annotation_bbox: List[float] | None = None
    if str(lookup_mode) != BETWEEN_NAMED_STEPS_COUNT_MODE:
        scalar_annotation_bbox = _target_annotation_bbox(
            lookup_mode=str(lookup_mode),
            target_step_id=str(target_step.step_id),
            number_badge_bbox_map=number_badge_bbox_map,
            title_bbox_map=title_bbox_map,
            detail_bbox_map=detail_bbox_map,
        )

    dynamic_slots = {
        "source_step_title": f'"{str(source_step.title)}"' if source_step is not None else "",
        "target_step_title": f'"{str(target_step.title)}"',
        "first_step_title": f'"{str(source_step.title)}"' if source_step is not None else "",
        "second_step_title": f'"{str(target_step.title)}"',
        "relative_offset_phrase": str(step_reference),
    }
    prompt_selection = render_task_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE,
        bundle_id=PROMPT_BUNDLE,
        scene_key=PROMPT_SCENE_KEY,
        task_key=PROMPT_TASK_KEY,
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=dynamic_slots,
        instance_seed=int(instance_seed),
    )
    prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)

    if str(lookup_mode) == BETWEEN_NAMED_STEPS_COUNT_MODE:
        answer_gt = TypedValue(type="integer", value=int(answer_value))
        annotation_gt = TypedValue(type="bbox_map", value=dict(reasoning_bbox_map))
        projected_annotation: Dict[str, Any] = {
            "type": "bbox_map",
            "bbox_map": dict(reasoning_bbox_map),
            "pixel_bbox_map": dict(reasoning_bbox_map),
            "target_step_id": str(target_step.step_id),
            "source_step_id": str(source_step.step_id) if source_step is not None else "",
        }
        witness_value: Any = dict(reasoning_bbox_map)
    else:
        if scalar_annotation_bbox is None:
            raise ValueError("scalar step-list annotation bbox missing")
        answer_gt = TypedValue(type="string", value=str(answer_value))
        annotation_gt = TypedValue(type="bbox", value=list(scalar_annotation_bbox))
        projected_annotation = {
            "type": "bbox",
            "bbox": list(scalar_annotation_bbox),
            "pixel_bbox": list(scalar_annotation_bbox),
            "target_step_id": str(target_step.step_id),
            "source_step_id": str(source_step.step_id) if source_step is not None else "",
        }
        witness_value = list(scalar_annotation_bbox)
    source_step_payload = _step_payload(source_step) if source_step is not None else None
    target_step_payload = _step_payload(target_step)
    probabilities = {str(key): float(value) for key, value in branch_probabilities.items()}
    common_params = {
        "query_id": str(selected_branch),
        "prompt_query_key": str(prompt_query_key),
        "source_query_id": source_query,
        "lookup_mode": str(lookup_mode),
        "scene_variant": str(scene_variant),
        "step_count": int(step_count),
        "step_reference": str(step_reference),
        "source_step_detail": str(target_step.detail),
        "target_step_index": int(target_index),
        "source_step_index": int(source_index) if source_index is not None else None,
        "target_answer": int(answer_value) if str(lookup_mode) == BETWEEN_NAMED_STEPS_COUNT_MODE else str(answer_value),
        "query_id_probabilities": dict(probabilities),
        "scene_variant_probabilities": dict(scene_variant_probabilities),
        "operand_probabilities": dict(operand_probabilities),
        "step_count_probabilities": dict(step_count_probabilities),
        **dict(operand_params),
    }
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_branch),
        params=common_params,
    )
    query_spec["scene_id"] = SCENE

    trace_payload = {
        "scene_ir": {
            "scene_id": SCENE,
            "scene_kind": "pages_step_list_cards",
            "entities": [dict(entity) for entity in rendered.entities],
            "relations": {
                "query_id": str(selected_branch),
                "prompt_query_key": str(prompt_query_key),
                "source_query_id": source_query,
                "lookup_mode": str(lookup_mode),
                "scene_variant": str(scene_variant),
                "step_count": int(step_count),
                "target_step": dict(target_step_payload),
                "source_step": dict(source_step_payload) if source_step_payload is not None else None,
                "answer_value": int(answer_value)
                if str(lookup_mode) == BETWEEN_NAMED_STEPS_COUNT_MODE
                else str(answer_value),
            },
        },
        "query_spec": query_spec,
        "render_spec": {
            "canvas_width": int(render_params.canvas_width),
            "canvas_height": int(render_params.canvas_height),
            "coord_space": "pixel",
            "scene_id": SCENE,
            "scene_variant": str(scene_variant),
            "background_style": dict(background_meta),
            "post_image_noise": dict(post_noise_meta),
            "panel_bbox_px": list(rendered.panel_bbox_px),
            "layout": dict(rendered.layout_meta),
            "text_style": {
                "title_font_size_px": int(render_params.title_font_size_px),
                "step_title_font_size_px": int(render_params.step_title_font_size_px),
                "step_detail_font_size_px": int(render_params.step_detail_font_size_px),
                "step_meta_font_size_px": int(render_params.step_meta_font_size_px),
            },
            "traced_text_fields": {
                "title_step_ids": [str(value) for value in trace_title_step_ids],
                "detail_step_ids": [str(value) for value in trace_detail_step_ids],
                "number_step_ids": [str(value) for value in trace_number_step_ids],
            },
            "page_text_resources": dict(page_text_resources),
        },
        "render_map": {
            "image_id": "img0",
            "panel_bbox_px": list(rendered.panel_bbox_px),
            "document_title_bbox_px": list(rendered.title_bbox_px),
            "card_bboxes_px": dict(card_bbox_map),
            "number_bboxes_px": dict(number_bbox_map),
            "number_badge_bboxes_px": dict(number_badge_bbox_map),
            "title_bboxes_px": dict(title_bbox_map),
            "detail_bboxes_px": dict(detail_bbox_map),
            "reasoning_bboxes_px": dict(reasoning_bbox_map),
            "target_annotation_role": str(annotation_role),
            "target_annotation_bbox_px": list(scalar_annotation_bbox) if scalar_annotation_bbox is not None else [],
            "target_annotation_bbox_map_px": dict(reasoning_bbox_map)
            if str(lookup_mode) == BETWEEN_NAMED_STEPS_COUNT_MODE
            else {},
        },
        "execution_trace": {
            **dict(common_params),
            "question_format": str(question_format),
            "step_count_support": [int(value) for value in step_count_support],
            "target_step": dict(target_step_payload),
            "source_step": dict(source_step_payload) if source_step_payload is not None else None,
            "answer_value": int(answer_value)
            if str(lookup_mode) == BETWEEN_NAMED_STEPS_COUNT_MODE
            else str(answer_value),
            "steps": [dict(card) for card in rendered.card_traces],
            "page_text_resources": dict(page_text_resources),
            "reasoning_bbox_roles": sorted(str(key) for key in reasoning_bbox_map.keys()),
            "reasoning_bboxes_px": dict(reasoning_bbox_map),
            "target_annotation_role": str(annotation_role),
            "target_annotation_bbox_px": list(scalar_annotation_bbox) if scalar_annotation_bbox is not None else [],
            "target_annotation_bbox_map_px": dict(reasoning_bbox_map)
            if str(lookup_mode) == BETWEEN_NAMED_STEPS_COUNT_MODE
            else {},
        },
        "witness_symbolic": {
            "type": "bbox_map" if str(lookup_mode) == BETWEEN_NAMED_STEPS_COUNT_MODE else "bbox",
            "target_step_id": str(target_step.step_id),
            "source_step_id": str(source_step.step_id) if source_step is not None else "",
            "answer_value": int(answer_value) if str(lookup_mode) == BETWEEN_NAMED_STEPS_COUNT_MODE else str(answer_value),
            "annotation_role": str(annotation_role),
            "reasoning_roles": sorted(str(key) for key in reasoning_bbox_map.keys()),
            "value": witness_value,
        },
        "projected_annotation": dict(projected_annotation),
    }
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=answer_gt,
        annotation_gt=annotation_gt,
        image=image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        query_id=str(selected_branch),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
    )


__all__ = [
    "AFTER_NAMED_TITLE_MODE",
    "BETWEEN_NAMED_STEPS_COUNT_MODE",
    "DETAIL_TO_NUMBER_MODE",
    "DETAIL_TO_TITLE_MODE",
    "DOMAIN",
    "OFFSET_AFTER_TITLE_MODE",
    "OFFSET_BEFORE_TITLE_MODE",
    "ORDINAL_DETAIL_MODE",
    "ORDINAL_TITLE_MODE",
    "SCENE",
    "SUPPORTED_SCENE_VARIANTS",
    "build_step_list_response",
    "select_public_branch",
]
