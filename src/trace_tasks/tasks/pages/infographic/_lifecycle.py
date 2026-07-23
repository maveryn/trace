"""Scene-private output assembly for infographic metric-card page tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.background import make_background_canvas
from trace_tasks.core.visual.noise import apply_post_image_noise
from ...base import TaskOutput
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_task_prompt_variants,
)
from trace_tasks.tasks.pages.shared.page_semantic_assets import page_semantic_asset_label, page_semantic_asset_manifest_metadata
from trace_tasks.tasks.pages.shared.infographic_metric_common import (
    GLOBAL_METRIC_RANKED_ITEM_VARIANTS,
    POST_IMAGE_BACKGROUND_DEFAULTS,
    POST_IMAGE_NOISE_DEFAULTS,
    SCENE_ID,
    SECTION_METRIC_RANKED_ITEM_VARIANTS,
    _PROMPT_DEFAULTS,
    _REASONING_LOAD_BY_VARIANT,
    _RENDER_DEFAULTS,
    _quote_label,
    _quoted_label_list,
)
from trace_tasks.tasks.pages.shared.infographic_metric_dataset import _build_dataset, _resolve_query_id
from trace_tasks.tasks.pages.shared.infographic_metric_rendering import _render_infographic


DOMAIN = "pages"
SCENE = SCENE_ID
PROMPT_BUNDLE = str(_PROMPT_DEFAULTS.get("bundle_id", "pages_infographic_v1"))
PROMPT_SCENE_KEY = "infographic_metric_arithmetic"
PROMPT_TASK_KEY = "metric_arithmetic_query"


def _source_branch(*parts: str) -> str:
    """Compose source-dataset branch names without embedding public task names here."""

    return "_".join(str(part) for part in parts)


SCALAR_BBOX_SOURCE_BRANCHES = (
    *GLOBAL_METRIC_RANKED_ITEM_VARIANTS,
    *SECTION_METRIC_RANKED_ITEM_VARIANTS,
    _source_branch("section", "icon", "extremum", "label"),
)
BBOX_SET_SOURCE_BRANCHES = (
    "sum_named_metrics",
    "section_total_except_named",
    _source_branch("section", "ranked", "total", "label"),
    _source_branch("section", "icon", "total", "value"),
)
BBOX_SET_MAP_SOURCE_BRANCHES = (
    "section_total_extrema_difference",
    _source_branch("section", "icon", "total", "difference", "value"),
)
BBOX_SET_MAP_GROUP_KEYS = {
    "section_total_extrema_difference": ("highest_total_section", "lowest_total_section"),
    _source_branch("section", "icon", "total", "difference", "value"): (
        "section_a_filtered_icon_cards",
        "section_b_filtered_icon_cards",
    ),
}


@dataclass(frozen=True)
class InfographicObjectiveBinding:
    """Task-owned source branches and prompt binding for one public task."""

    source_branch_keys: Tuple[str, ...]
    prompt_branch_fallback: str


def select_public_branch(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    supported: Tuple[str, ...],
    default: str,
    public_task: str,
) -> Tuple[str, Dict[str, float], Dict[str, Any]]:
    """Resolve the public branch through the shared single-query policy."""

    selected, probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in supported),
        default_query_id=str(default),
        task_id=str(public_task),
    )
    return str(selected), dict(probabilities), dict(task_params)


def _normalize_int_with_bounds(value: int, bounds: Sequence[int]) -> float:
    """Return a 0..1 normalized value for lightweight reasoning-load metadata."""

    if len(bounds) < 2:
        return 0.0
    lower = float(bounds[0])
    upper = float(bounds[-1])
    if upper <= lower:
        return 0.0
    return max(0.0, min(1.0, (float(value) - lower) / (upper - lower)))


def _icon_prompt_label(icon_kind: str) -> str:
    if not str(icon_kind).strip():
        return ""
    label = page_semantic_asset_label(str(icon_kind))
    return str(label).removesuffix(" icon")


def _annotation_keyed_bboxes(
    *,
    card_traces: Sequence[Mapping[str, Any]],
    target_labels: Sequence[str],
    section_bboxes: Mapping[str, Sequence[float]] | None = None,
    section_title_bboxes: Mapping[str, Sequence[float]] | None = None,
    annotation_targets: Sequence[Mapping[str, Any]] | None = None,
) -> Dict[str, List[float]]:
    """Project task-bound metric/section witnesses to keyed bboxes.

    The invariant is that every key in the returned map is supplied by the
    dataset trace and resolves to exactly one visible card, label, value,
    caption, section, or section-title box from the final render.
    """

    by_label = {str(card["label"]): dict(card) for card in card_traces}
    section_map = {
        str(section): [float(value) for value in bbox]
        for section, bbox in dict(section_bboxes or {}).items()
    }
    section_title_map = {
        str(section): [float(value) for value in bbox]
        for section, bbox in dict(section_title_bboxes or {}).items()
    }
    keyed_bboxes: Dict[str, List[float]] = {}
    if annotation_targets is not None and len(annotation_targets) > 0:
        for target in annotation_targets:
            bbox_kind = str(target.get("bbox_kind", "card"))
            if str(bbox_kind) == "section":
                section = str(target["section"])
                if section not in section_map:
                    raise ValueError(f"unknown infographic annotation section: {section}")
                key = str(target.get("key", section))
                keyed_bboxes[str(key)] = [float(value) for value in section_map[section]]
                continue
            if str(bbox_kind) == "section_title":
                section = str(target["section"])
                if section not in section_title_map:
                    raise ValueError(f"unknown infographic annotation section title: {section}")
                key = str(target.get("key", "section_title"))
                keyed_bboxes[str(key)] = [float(value) for value in section_title_map[section]]
                continue
            label = str(target["label"])
            if label not in by_label:
                raise ValueError(f"unknown infographic annotation label: {label}")
            card = by_label[label]
            bbox_key = {
                "label": "label_bbox_px",
                "value": "value_bbox_px",
                "card": "card_bbox_px",
                "caption": "caption_bbox_px",
            }.get(str(bbox_kind), "card_bbox_px")
            key = str(target.get("key", label))
            keyed_bboxes[str(key)] = [float(value) for value in card[str(bbox_key)]]
        return keyed_bboxes
    for label in target_labels:
        card = by_label[str(label)]
        keyed_bboxes[str(label)] = [float(value) for value in card["card_bbox_px"]]
    return keyed_bboxes


def _annotation_bbox_set_map(
    *,
    annotation_keyed_bboxes: Mapping[str, Sequence[float]],
    target_groups: Mapping[str, Sequence[str]],
    group_keys: Sequence[str],
) -> Dict[str, List[List[float]]]:
    """Project role/group-bound label sets into bbox-set-map annotation."""

    grouped: Dict[str, List[List[float]]] = {}
    for group_key in group_keys:
        labels = [str(label) for label in target_groups.get(str(group_key), [])]
        if not labels:
            raise ValueError(f"missing infographic annotation group: {group_key}")
        group_bboxes: List[List[float]] = []
        for label in labels:
            if label not in annotation_keyed_bboxes:
                raise ValueError(f"missing annotation bbox for label {label} in group {group_key}")
            group_bboxes.append([float(value) for value in annotation_keyed_bboxes[label]])
        grouped[str(group_key)] = group_bboxes
    return grouped


class InfographicMetricCardRuntime:
    """Build one metric-card infographic response from a task-bound source branch."""

    domain = DOMAIN
    scene_id = SCENE

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Assemble the neutral metric-card scene after public task binding.

        Public task files own the task id and allowed source branches. This
        lifecycle keeps the shared render/prompt/trace shell deterministic for
        the selected source branch and never chooses a public taxonomy unit.
        """

        del max_attempts
        source_branch, source_branch_probabilities = _resolve_query_id(params, instance_seed=int(instance_seed))
        dataset = _build_dataset(query_id=str(source_branch), params=params, instance_seed=int(instance_seed))

        canvas_width = int(params.get("canvas_width", group_default(_RENDER_DEFAULTS, "canvas_width", 940)))
        canvas_height = int(params.get("canvas_height", group_default(_RENDER_DEFAULTS, "canvas_height", 900)))
        background, background_meta = make_background_canvas(
            canvas_width=int(canvas_width),
            canvas_height=int(canvas_height),
            instance_seed=int(instance_seed),
            params=params,
            default_config=POST_IMAGE_BACKGROUND_DEFAULTS,
        )
        render_params = dict(_RENDER_DEFAULTS)
        if params.get("infographic_style") is not None:
            render_params["infographic_style"] = params["infographic_style"]
        rendered = _render_infographic(
            background,
            cards=dataset["cards"],
            section_titles=dataset["section_titles"],
            section_card_counts=dataset["section_card_counts"],
            render_defaults=render_params,
            instance_seed=int(instance_seed),
        )
        image, post_noise_meta = apply_post_image_noise(
            rendered.image,
            instance_seed=int(instance_seed),
            params=params,
            default_config=POST_IMAGE_NOISE_DEFAULTS,
        )

        answer_type = str(dataset.get("answer_type", "integer"))
        answer_value_for_trace: int | str
        if answer_type == "string":
            answer_value_for_trace = str(dataset["answer_value"])
        else:
            answer_value_for_trace = int(dataset["answer_value"])
        target_labels = [str(label) for label in dataset["target_labels"]]
        target_values = [int(value) for value in dataset["target_values"]]
        target_groups = {
            str(key): [str(label) for label in value]
            for key, value in dict(dataset["target_groups"]).items()
        }
        target_sections = [str(section) for section in dataset["target_sections"]]
        excluded_labels = [str(label) for label in dataset["excluded_labels"]]
        target_extrema = [str(extremum) for extremum in dataset["target_extrema"]]
        extrema_operation = str(dataset["extrema_operation"])
        extrema_operation_phrase = "absolute difference" if str(extrema_operation) == "absolute_difference" else str(extrema_operation)
        group_a_labels = target_groups.get("group_a", target_groups.get("section_a", []))
        group_b_labels = target_groups.get("group_b", target_groups.get("section_b", []))
        if not group_a_labels:
            group_a_labels = target_labels[: max(1, len(target_labels) // 2)]
        if not group_b_labels:
            group_b_labels = target_labels[max(1, len(target_labels) // 2) :]
        prompt_slots = {
            "target_label": _quote_label(target_labels[0]),
            "target_label_a": _quote_label(target_labels[0]),
            "target_label_b": _quote_label(target_labels[1]) if len(target_labels) > 1 else "",
            "target_labels": _quoted_label_list(target_labels),
            "target_group_a": _quoted_label_list(group_a_labels),
            "target_group_b": _quoted_label_list(group_b_labels),
            "target_section": _quote_label(target_sections[0]) if target_sections else "",
            "target_section_a": _quote_label(target_sections[0]) if len(target_sections) >= 1 else "",
            "target_section_b": _quote_label(target_sections[1]) if len(target_sections) >= 2 else "",
            "target_extremum_a": str(target_extrema[0]) if len(target_extrema) >= 1 else "",
            "target_extremum_b": str(target_extrema[1]) if len(target_extrema) >= 2 else "",
            "extrema_operation": str(extrema_operation_phrase),
            "rank_direction": str(dataset.get("rank_direction", "")),
            "rank_ordinal": str(dataset.get("rank_ordinal", "")),
            "rank_order_phrase": (
                "highest to lowest"
                if str(dataset.get("rank_direction", "")) == "highest"
                else "lowest to highest"
            ),
            "rank_position": str(dataset.get("rank_position", "")),
            "excluded_labels": _quoted_label_list(excluded_labels) if excluded_labels else "",
            "filter_icon_kind": _icon_prompt_label(str(dataset.get("filter_icon_kind", ""))),
            "comparison_icon_kind": _icon_prompt_label(str(dataset.get("comparison_icon_kind", ""))),
            "target_value": _quote_label(str(dataset.get("target_value_text", ""))) if dataset.get("target_value_text") else "",
            "target_detail": _quote_label(str(dataset.get("target_detail_text", ""))) if dataset.get("target_detail_text") else "",
        }
        prompt_selection = render_task_prompt_variants(
            domain=DOMAIN,
            scene_id=SCENE,
            bundle_id=str(PROMPT_BUNDLE),
            scene_key=PROMPT_SCENE_KEY,
            task_key=PROMPT_TASK_KEY,
            query_key=str(source_branch),
            answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
            dynamic_slots=prompt_slots,
            instance_seed=int(instance_seed),
        )
        prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)

        annotation_targets = [dict(item) for item in list(dataset.get("annotation_targets", []))]
        annotation_keyed_bboxes = _annotation_keyed_bboxes(
            card_traces=rendered.card_traces,
            target_labels=target_labels,
            section_bboxes=rendered.section_bboxes,
            section_title_bboxes=rendered.section_title_bboxes,
            annotation_targets=annotation_targets,
        )
        annotation_bboxes = [list(bbox) for bbox in annotation_keyed_bboxes.values()]
        annotation_bbox_set_map = _annotation_bbox_set_map(
            annotation_keyed_bboxes=annotation_keyed_bboxes,
            target_groups=target_groups,
            group_keys=BBOX_SET_MAP_GROUP_KEYS.get(str(source_branch), ()),
        ) if str(source_branch) in set(BBOX_SET_MAP_SOURCE_BRANCHES) else {}
        answer_gt = TypedValue(type=answer_type, value=answer_value_for_trace)
        if str(source_branch) in set(SCALAR_BBOX_SOURCE_BRANCHES):
            if "target_card" in annotation_keyed_bboxes:
                scalar_bbox = list(annotation_keyed_bboxes["target_card"])
            elif len(annotation_bboxes) == 1:
                scalar_bbox = list(annotation_bboxes[0])
            else:
                raise ValueError(
                    f"{source_branch} must resolve one public scalar annotation bbox, "
                    f"got keys {sorted(annotation_keyed_bboxes)}"
                )
            annotation_type = "bbox"
            annotation_value: Any = list(scalar_bbox)
        elif str(source_branch) in set(BBOX_SET_SOURCE_BRANCHES):
            if not annotation_bboxes:
                raise ValueError(f"{source_branch} must resolve at least one annotation bbox")
            annotation_type = "bbox_set"
            annotation_value = [list(bbox) for bbox in annotation_bboxes]
        elif str(source_branch) in set(BBOX_SET_MAP_SOURCE_BRANCHES):
            if not annotation_bbox_set_map:
                raise ValueError(f"{source_branch} must resolve grouped annotation bboxes")
            annotation_type = "bbox_set_map"
            annotation_value = {str(key): [list(bbox) for bbox in value] for key, value in annotation_bbox_set_map.items()}
        else:
            annotation_type = "bbox_map"
            annotation_value = dict(annotation_keyed_bboxes)
        annotation_gt = TypedValue(type=str(annotation_type), value=annotation_value)

        label_bbox_map = {
            str(card["label"]): [float(value) for value in card["label_bbox_px"]]
            for card in rendered.card_traces
        }
        value_bbox_map = {
            str(card["label"]): [float(value) for value in card["value_bbox_px"]]
            for card in rendered.card_traces
        }
        card_bbox_map = {
            str(card["label"]): [float(value) for value in card["card_bbox_px"]]
            for card in rendered.card_traces
        }
        caption_bbox_map = {
            str(card["label"]): [float(value) for value in card["caption_bbox_px"]]
            for card in rendered.card_traces
        }
        section_bbox_map = {
            str(section): [float(value) for value in bbox]
            for section, bbox in rendered.section_bboxes.items()
        }
        section_title_bbox_map = {
            str(section): [float(value) for value in bbox]
            for section, bbox in rendered.section_title_bboxes.items()
        }
        page_bbox = [float(value) for value in rendered.page_bbox]
        document_title_bbox = [float(value) for value in rendered.document_title_bbox]
        document_subtitle_bbox = [float(value) for value in rendered.document_subtitle_bbox]

        if str(annotation_type) == "bbox":
            projected_annotation = {
                "type": "bbox",
                "bbox": list(annotation_value),
                "pixel_bbox": list(annotation_value),
                "bbox_map": dict(annotation_keyed_bboxes),
                "pixel_bbox_map": dict(annotation_keyed_bboxes),
                "bbox_set": list(annotation_bboxes),
                "pixel_bbox_set": list(annotation_bboxes),
                "bbox_set_map": dict(annotation_bbox_set_map),
                "pixel_bbox_set_map": dict(annotation_bbox_set_map),
                "label_bbox_map": dict(label_bbox_map),
                "value_bbox_map": dict(value_bbox_map),
                "card_bbox_map": dict(card_bbox_map),
                "caption_bbox_map": dict(caption_bbox_map),
                "section_title_bbox_map": dict(section_title_bbox_map),
                "target_labels": list(target_labels),
                "annotation_keys": list(annotation_keyed_bboxes.keys()),
                "annotation_targets": [dict(item) for item in annotation_targets],
            }
        elif str(annotation_type) == "bbox_set":
            projected_annotation = {
                "type": "bbox_set",
                "bbox_set": list(annotation_value),
                "pixel_bbox_set": list(annotation_value),
                "bbox_map": dict(annotation_keyed_bboxes),
                "pixel_bbox_map": dict(annotation_keyed_bboxes),
                "bbox_set_map": dict(annotation_bbox_set_map),
                "pixel_bbox_set_map": dict(annotation_bbox_set_map),
                "label_bbox_map": dict(label_bbox_map),
                "value_bbox_map": dict(value_bbox_map),
                "card_bbox_map": dict(card_bbox_map),
                "caption_bbox_map": dict(caption_bbox_map),
                "section_title_bbox_map": dict(section_title_bbox_map),
                "target_labels": list(target_labels),
                "annotation_keys": list(annotation_keyed_bboxes.keys()),
                "annotation_targets": [dict(item) for item in annotation_targets],
            }
        elif str(annotation_type) == "bbox_set_map":
            projected_annotation = {
                "type": "bbox_set_map",
                "bbox_set_map": dict(annotation_value),
                "pixel_bbox_set_map": dict(annotation_value),
                "bbox_map": dict(annotation_keyed_bboxes),
                "pixel_bbox_map": dict(annotation_keyed_bboxes),
                "bbox_set": list(annotation_bboxes),
                "pixel_bbox_set": list(annotation_bboxes),
                "label_bbox_map": dict(label_bbox_map),
                "value_bbox_map": dict(value_bbox_map),
                "card_bbox_map": dict(card_bbox_map),
                "caption_bbox_map": dict(caption_bbox_map),
                "section_title_bbox_map": dict(section_title_bbox_map),
                "target_labels": list(target_labels),
                "annotation_keys": list(annotation_keyed_bboxes.keys()),
                "annotation_group_keys": list(annotation_bbox_set_map.keys()),
                "annotation_targets": [dict(item) for item in annotation_targets],
            }
        else:
            projected_annotation = {
                "type": "bbox_map",
                "bbox_map": dict(annotation_keyed_bboxes),
                "pixel_bbox_map": dict(annotation_keyed_bboxes),
                "bbox_set": list(annotation_bboxes),
                "pixel_bbox_set": list(annotation_bboxes),
                "bbox_set_map": dict(annotation_bbox_set_map),
                "pixel_bbox_set_map": dict(annotation_bbox_set_map),
                "label_bbox_map": dict(label_bbox_map),
                "value_bbox_map": dict(value_bbox_map),
                "card_bbox_map": dict(card_bbox_map),
                "caption_bbox_map": dict(caption_bbox_map),
                "section_title_bbox_map": dict(section_title_bbox_map),
                "target_labels": list(target_labels),
                "annotation_keys": list(annotation_keyed_bboxes.keys()),
                "annotation_targets": [dict(item) for item in annotation_targets],
            }

        trace_payload = {
            "scene_ir": {
                "scene_id": SCENE_ID,
                "scene_kind": "pages_infographic_metric_cards",
                "entities": [dict(entity) for entity in rendered.entities],
                "relations": {
                    "query_id": str(source_branch),
                    "source_query_id": str(source_branch),
                    "prompt_query_key": str(source_branch),
                    "target_labels": list(target_labels),
                    "target_values": list(target_values),
                    "target_groups": dict(target_groups),
                    "target_sections": list(target_sections),
                    "excluded_labels": list(excluded_labels),
                    "target_extrema": list(target_extrema),
                    "extrema_operation": str(extrema_operation),
                    "rank_direction": str(dataset.get("rank_direction", "")),
                    "rank_ordinal": str(dataset.get("rank_ordinal", "")),
                    "rank_position": int(dataset.get("rank_position", 0)),
                    "rank_scope": str(dataset.get("rank_scope", "")),
                    "ranked_candidates": [dict(item) for item in dataset.get("ranked_candidates", [])],
                    "filter_icon_kind": str(dataset.get("filter_icon_kind", "")),
                    "filter_icon_label": _icon_prompt_label(str(dataset.get("filter_icon_kind", ""))),
                    "comparison_icon_kind": str(dataset.get("comparison_icon_kind", "")),
                    "comparison_icon_label": _icon_prompt_label(str(dataset.get("comparison_icon_kind", ""))),
                    "filtered_section_totals": dict(dataset.get("filtered_section_totals", {})),
                    "target_value_text": str(dataset.get("target_value_text", "")),
                    "target_detail_text": str(dataset.get("target_detail_text", "")),
                    "annotation_targets": [dict(item) for item in annotation_targets],
                    "answer_value": answer_value_for_trace,
                    "answer_type": str(answer_type),
                    "arithmetic_expression": str(dataset["arithmetic_expression"]),
                },
            },
            "query_spec": {
                "query_id": str(source_branch),
                "source_query_id": str(source_branch),
                "prompt_query_key": str(source_branch),
                "template_id": str(PROMPT_BUNDLE),
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
                "params": {
                    "query_id": str(source_branch),
                    "source_query_id": str(source_branch),
                    "prompt_query_key": str(source_branch),
                    "source_branch_probabilities": dict(source_branch_probabilities),
                    "card_count": int(dataset["card_count"]),
                    "section_count": int(dataset["section_count"]),
                    "target_labels": list(target_labels),
                    "target_values": list(target_values),
                    "target_groups": dict(target_groups),
                    "target_sections": list(target_sections),
                    "excluded_labels": list(excluded_labels),
                    "target_extrema": list(target_extrema),
                    "extrema_operation": str(extrema_operation),
                    "rank_direction": str(dataset.get("rank_direction", "")),
                    "rank_ordinal": str(dataset.get("rank_ordinal", "")),
                    "rank_position": int(dataset.get("rank_position", 0)),
                    "rank_scope": str(dataset.get("rank_scope", "")),
                    "ranked_candidates": [dict(item) for item in dataset.get("ranked_candidates", [])],
                    "filter_icon_kind": str(dataset.get("filter_icon_kind", "")),
                    "filter_icon_label": _icon_prompt_label(str(dataset.get("filter_icon_kind", ""))),
                    "comparison_icon_kind": str(dataset.get("comparison_icon_kind", "")),
                    "comparison_icon_label": _icon_prompt_label(str(dataset.get("comparison_icon_kind", ""))),
                    "filtered_section_totals": dict(dataset.get("filtered_section_totals", {})),
                    "target_value_text": str(dataset.get("target_value_text", "")),
                    "target_detail_text": str(dataset.get("target_detail_text", "")),
                    "annotation_targets": [dict(item) for item in annotation_targets],
                    "target_answer": answer_value_for_trace,
                    "answer_type": str(answer_type),
                },
            },
            "render_spec": {
                "canvas_width": int(canvas_width),
                "canvas_height": int(canvas_height),
                "coord_space": "pixel",
                "scene_id": SCENE_ID,
                "scene_variant": str(rendered.layout_jitter_meta.get("infographic_style", "card_wall")),
                "background_style": dict(background_meta),
                "post_image_noise": dict(post_noise_meta),
                "layout_jitter": dict(rendered.layout_jitter_meta),
                "page_bbox_px": list(page_bbox),
                "title_bbox_px": list(document_title_bbox),
                "subtitle_bbox_px": list(document_subtitle_bbox),
                "section_bboxes_px": dict(section_bbox_map),
                "section_title_bboxes_px": dict(section_title_bbox_map),
                "text_style": {
                    "title_font_size_px": int(group_default(_RENDER_DEFAULTS, "title_font_size_px", 30)),
                    "label_font_size_px": int(group_default(_RENDER_DEFAULTS, "label_font_size_px", 18)),
                    "caption_font_size_px": int(group_default(_RENDER_DEFAULTS, "caption_font_size_px", 13)),
                },
                "page_text_resources": dict(dataset.get("page_text_resources", {})),
                "page_semantic_assets": page_semantic_asset_manifest_metadata(
                    semantic_role="metric_icon",
                    allowed_use="filter",
                ),
            },
            "render_map": {
                "image_id": "img0",
                "card_bboxes_px": dict(card_bbox_map),
                "label_bboxes_px": dict(label_bbox_map),
                "value_bboxes_px": dict(value_bbox_map),
                "caption_bboxes_px": dict(caption_bbox_map),
                "page_bbox_px": list(page_bbox),
                "title_bbox_px": list(document_title_bbox),
                "subtitle_bbox_px": list(document_subtitle_bbox),
                "section_bboxes_px": dict(section_bbox_map),
                "section_title_bboxes_px": dict(section_title_bbox_map),
            },
            "execution_trace": {
                "query_id": str(source_branch),
                "source_query_id": str(source_branch),
                "prompt_query_key": str(source_branch),
                "answer_value": answer_value_for_trace,
                "answer_type": str(answer_type),
                "arithmetic_expression": str(dataset["arithmetic_expression"]),
                "target_labels": list(target_labels),
                "target_values": list(target_values),
                "target_groups": dict(target_groups),
                "target_sections": list(target_sections),
                "excluded_labels": list(excluded_labels),
                "target_extrema": list(target_extrema),
                "extrema_operation": str(extrema_operation),
                "rank_direction": str(dataset.get("rank_direction", "")),
                "rank_ordinal": str(dataset.get("rank_ordinal", "")),
                "rank_position": int(dataset.get("rank_position", 0)),
                "rank_scope": str(dataset.get("rank_scope", "")),
                "ranked_candidates": [dict(item) for item in dataset.get("ranked_candidates", [])],
                "rank_direction_probabilities": dict(dataset.get("rank_direction_probabilities", {})),
                "rank_position_probabilities": dict(dataset.get("rank_position_probabilities", {})),
                "filter_icon_kind": str(dataset.get("filter_icon_kind", "")),
                "filter_icon_label": _icon_prompt_label(str(dataset.get("filter_icon_kind", ""))),
                "comparison_icon_kind": str(dataset.get("comparison_icon_kind", "")),
                "comparison_icon_label": _icon_prompt_label(str(dataset.get("comparison_icon_kind", ""))),
                "filtered_section_totals": {
                    str(section): int(total)
                    for section, total in dict(dataset.get("filtered_section_totals", {})).items()
                },
                "target_value_text": str(dataset.get("target_value_text", "")),
                "target_detail_text": str(dataset.get("target_detail_text", "")),
                "annotation_targets": [dict(item) for item in annotation_targets],
                "labels": [str(label) for label in dataset["labels"]],
                "values_by_label": {str(label): int(value) for label, value in dataset["values_by_label"].items()},
                "cards": [dict(card) for card in rendered.card_traces],
                "card_count": int(dataset["card_count"]),
                "card_count_range": list(dataset["card_count_range"]),
                "card_count_probabilities": dict(dataset["card_count_probabilities"]),
                "section_count": int(dataset["section_count"]),
                "section_count_range": list(dataset["section_count_range"]),
                "section_count_probabilities": dict(dataset["section_count_probabilities"]),
                "section_titles": [str(title) for title in dataset["section_titles"]],
                "section_card_counts": [int(value) for value in dataset["section_card_counts"]],
                "section_totals": {str(section): int(value) for section, value in dataset["section_totals"].items()},
                "target_operand_count": int(dataset["target_operand_count"]),
                "target_operand_count_range": list(dataset["target_operand_count_range"]),
                "target_operand_count_probabilities": dict(dataset["target_operand_count_probabilities"]),
                "source_branch_probabilities": dict(source_branch_probabilities),
                "question_format": "label_open" if answer_type == "string" else "numeric_open",
                "percent_mode": bool(dataset["percent_mode"]),
                "value_min": int(dataset["value_min"]),
                "value_max": int(dataset["value_max"]),
                "percent_value_min": int(dataset["percent_value_min"]),
                "percent_value_max": int(dataset["percent_value_max"]),
            },
            "witness_symbolic": {
                "type": "metric_card_keyed_set",
                "labels": list(target_labels),
                "annotation_keys": list(annotation_keyed_bboxes.keys()),
                "groups": dict(target_groups),
                "sections": list(target_sections),
                "excluded_labels": list(excluded_labels),
                "extrema": list(target_extrema),
                "extrema_operation": str(extrema_operation),
                "rank_direction": str(dataset.get("rank_direction", "")),
                "rank_ordinal": str(dataset.get("rank_ordinal", "")),
                "rank_position": int(dataset.get("rank_position", 0)),
                "rank_scope": str(dataset.get("rank_scope", "")),
                "ranked_candidates": [dict(item) for item in dataset.get("ranked_candidates", [])],
                "filter_icon_kind": str(dataset.get("filter_icon_kind", "")),
                "filter_icon_label": _icon_prompt_label(str(dataset.get("filter_icon_kind", ""))),
                "comparison_icon_kind": str(dataset.get("comparison_icon_kind", "")),
                "comparison_icon_label": _icon_prompt_label(str(dataset.get("comparison_icon_kind", ""))),
                "values": list(target_values),
                "expression": str(dataset["arithmetic_expression"]),
                "annotation_targets": [dict(item) for item in annotation_targets],
            },
            "projected_annotation": dict(projected_annotation),
        }

        reasoning_load = float(_REASONING_LOAD_BY_VARIANT[str(source_branch)])
        if str(source_branch) == "sum_named_metrics":
            reasoning_load = min(
                1.0,
                reasoning_load
                + (0.14 * _normalize_int_with_bounds(int(dataset["target_operand_count"]), dataset["target_operand_count_range"])),
            )
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            query_id=str(source_branch),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
        )


__all__ = [
    "DOMAIN",
    "SCENE",
    "InfographicMetricCardRuntime",
    "InfographicObjectiveBinding",
    "select_public_branch",
]
