"""Select the patch option that restores an RPG house source panel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    group_default,
    required_group_defaults,
    split_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.illustrations.shared.canvas_profiles import MAX_RECONSTRUCTION_OUTPUT_PIXELS
from trace_tasks.tasks.illustrations.shared.cutouts import (
    DEFAULT_OPTION_LABELS,
    FRAMELESS_ILLUSTRATION_PATCH_STYLE,
    PATCH_MODE_PLAIN,
    compose_patch_options,
    downscale_patch_option_artifacts,
    style_trace,
)
from trace_tasks.tasks.illustrations.shared.missing_patch_sizing import sample_missing_patch_size
from trace_tasks.tasks.illustrations.shared.option_rendering import sample_visual_label_font_trace

from .shared.output import room_bbox_map, rpg_house_scene_ir
from .shared.prompts import build_rpg_house_prompt_artifacts
from .shared.rendering import MAX_ROOM_COUNT, MIN_ROOM_COUNT, SCENE_ID
from .shared.source_images import (
    candidate_crop_boxes_from_scene,
    option_count_support,
    render_rpg_house_source_scene,
    rpg_house_source_style_trace,
    sample_rpg_house_source_scene_spec,
    sample_support_index,
)


TASK_ID = "task_illustrations__rpg_house__missing_patch_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "missing_patch_label"


@dataclass(frozen=True)
class _Defaults:
    source_room_count_min: int = 6
    source_room_count_max: int = 8
    option_count_support: Tuple[int, ...] = (4, 6)
    patch_width_ratio_min: float = 0.15
    patch_width_ratio_max: float = 0.30
    patch_height_ratio_min: float = 0.15
    patch_height_ratio_max: float = 0.26
    patch_area_ratio_max: float = 0.065
    crop_margin_px: int = 34
    source_width: int = 820
    source_height: int = 560


@dataclass(frozen=True)
class _SampleSpec:
    query_id: str
    patch_size: Tuple[int, int]
    patch_size_trace: Dict[str, Any]
    crop_margin_px: int
    option_count: int
    correct_index: int
    source_room_count: int
    source_size: Tuple[int, int]
    source_profile_trace: Dict[str, Any]
    query_probabilities: Dict[str, float]
    option_count_probabilities: Dict[str, float]
    correct_index_probabilities: Dict[str, float]
    source_room_count_probabilities: Dict[str, float]


_DEFAULTS = _Defaults()
_SCENE_DEFAULTS = get_scene_defaults("illustrations", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)


def _bbox_map(value: Mapping[str, Sequence[float]]) -> Dict[str, list[float]]:
    return {
        str(key): [round(float(coord), 3) for coord in bbox[:4]]
        for key, bbox in value.items()
    }


def _source_bboxes(scene: Any) -> Dict[str, Any]:
    return {
        "rooms": room_bbox_map(scene),
        "doors": {str(door.door_id): [round(float(value), 3) for value in door.bbox_xyxy] for door in scene.doors},
        "entities": {str(entity.entity_id): [round(float(value), 3) for value in entity.bbox_xyxy] for entity in scene.entities},
    }


def _sample_spec(*, instance_seed: int, params: Mapping[str, Any], attempt_index: int) -> _SampleSpec:
    """Sample source-panel and option parameters without selecting pixels."""

    query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=SUPPORTED_QUERY_IDS,
        default_query_id=SINGLE_QUERY_ID,
        task_id=TASK_ID,
        namespace=f"{TASK_ID}:query",
    )
    option_support = option_count_support(
        params=task_params,
        defaults=_GEN_DEFAULTS,
        fallback=_DEFAULTS.option_count_support,
    )
    option_count, option_probabilities = sample_support_index(
        seed_namespace=f"{TASK_ID}:option_count",
        instance_seed=int(instance_seed),
        params=task_params,
        support=option_support,
        explicit_key="option_count",
    )
    correct_index, correct_probabilities = sample_support_index(
        seed_namespace=f"{TASK_ID}:correct_index",
        instance_seed=int(instance_seed),
        params=task_params,
        support=tuple(range(int(option_count))),
        explicit_key="correct_index",
    )
    source = sample_rpg_house_source_scene_spec(
        seed_namespace=TASK_ID,
        instance_seed=int(instance_seed),
        params=task_params,
        generation_defaults=_GEN_DEFAULTS,
        source_room_count_min=_DEFAULTS.source_room_count_min,
        source_room_count_max=_DEFAULTS.source_room_count_max,
        source_width=_DEFAULTS.source_width,
        source_height=_DEFAULTS.source_height,
    )
    patch_sample = sample_missing_patch_size(
        rng=spawn_rng(int(instance_seed), f"{TASK_ID}:patch_spec", int(attempt_index)),
        params=task_params,
        defaults=_GEN_DEFAULTS,
        source_size=source.source_size,
        fallback_width_ratio_min=_DEFAULTS.patch_width_ratio_min,
        fallback_width_ratio_max=_DEFAULTS.patch_width_ratio_max,
        fallback_height_ratio_min=_DEFAULTS.patch_height_ratio_min,
        fallback_height_ratio_max=_DEFAULTS.patch_height_ratio_max,
        fallback_area_ratio_max=_DEFAULTS.patch_area_ratio_max,
    )
    return _SampleSpec(
        query_id=str(query_id),
        patch_size=tuple(int(value) for value in patch_sample.patch_size),
        patch_size_trace=dict(patch_sample.trace()),
        crop_margin_px=int(task_params.get("crop_margin_px", group_default(_GEN_DEFAULTS, "crop_margin_px", _DEFAULTS.crop_margin_px))),
        option_count=int(option_count),
        correct_index=int(correct_index),
        source_room_count=int(source.source_room_count),
        source_size=tuple(source.source_size),
        source_profile_trace=dict(source.source_profile_trace),
        query_probabilities=dict(query_probabilities),
        option_count_probabilities=dict(option_probabilities),
        correct_index_probabilities=dict(correct_probabilities),
        source_room_count_probabilities=dict(source.source_room_count_probabilities),
    )


@register_task
class IllustrationsRpgHouseMissingPatchLabelTask:
    """Select the patch option that restores a missing RPG house region."""

    task_id = TASK_ID
    reasoning_operations = ('transformation', 'matching')
    domain = "illustrations"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Render one source-with-hole panel and lettered patch options."""

        last_error: Exception | None = None
        sample: _SampleSpec | None = None
        source_scene = None
        artifacts = None
        candidate_crop_boxes: Tuple[Tuple[int, int, int, int], ...] = tuple()
        label_font_trace: Dict[str, Any] | None = None
        frame_style = {"style_id": "frameless_illustration", **dict(FRAMELESS_ILLUSTRATION_PATCH_STYLE)}

        for attempt in range(max(1, int(max_attempts))):
            try:
                sample = _sample_spec(instance_seed=int(instance_seed), params=params, attempt_index=int(attempt))
                source_spec = sample_rpg_house_source_scene_spec(
                    seed_namespace=TASK_ID,
                    instance_seed=int(instance_seed),
                    params=params,
                    generation_defaults=_GEN_DEFAULTS,
                    source_room_count_min=_DEFAULTS.source_room_count_min,
                    source_room_count_max=_DEFAULTS.source_room_count_max,
                    source_width=_DEFAULTS.source_width,
                    source_height=_DEFAULTS.source_height,
                )
                source_scene = render_rpg_house_source_scene(
                    seed_namespace=TASK_ID,
                    instance_seed=int(instance_seed),
                    attempt_index=int(attempt),
                    source=source_spec,
                    params={**dict(params), "source_room_count": int(sample.source_room_count)},
                    render_defaults=_RENDER_DEFAULTS,
                )
                candidate_crop_boxes = candidate_crop_boxes_from_scene(
                    scene=source_scene,
                    patch_size=sample.patch_size,
                    margin_px=int(sample.crop_margin_px),
                )
                label_font_trace = sample_visual_label_font_trace(
                    namespace_prefix=TASK_ID,
                    instance_seed=int(instance_seed),
                    params={**dict(_RENDER_DEFAULTS), **dict(params)},
                    namespace_suffix="patch_option_labels",
                    explicit_key="patch_label_font_family",
                    weights_key="patch_label_font_weights",
                )
                artifacts = compose_patch_options(
                    source_image=source_scene.image.convert("RGB"),
                    rng=spawn_rng(int(instance_seed), f"{TASK_ID}:patch_options", int(attempt)),
                    patch_mode=PATCH_MODE_PLAIN,
                    correct_index=int(sample.correct_index),
                    option_count=int(sample.option_count),
                    patch_size=sample.patch_size,
                    crop_margin_px=int(sample.crop_margin_px),
                    frame_style=frame_style,
                    label_font_family=str(label_font_trace["font_family"]),
                    candidate_crop_boxes=candidate_crop_boxes,
                    render_margin=0,
                    draw_source_outline=False,
                    draw_option_outlines=True,
                )
                artifacts = downscale_patch_option_artifacts(
                    artifacts,
                    max_pixels=MAX_RECONSTRUCTION_OUTPUT_PIXELS,
                )
                break
            except Exception as exc:  # pragma: no cover - retry surface is seed/layout dependent.
                last_error = exc
                sample = None
                source_scene = None
                artifacts = None
                candidate_crop_boxes = tuple()
                label_font_trace = None

        if sample is None or source_scene is None or artifacts is None or label_font_trace is None:
            raise RuntimeError(f"could not generate {TASK_ID}: {last_error}") from last_error

        answer_label = str(artifacts.selected_label)
        annotation_value = _bbox_map(
            {
                "missing_region": artifacts.missing_region_bbox,
                "selected_option": artifacts.selected_option_bbox,
            }
        )
        prompt_defaults = required_group_defaults(
            _PROMPT_DEFAULTS,
            [
                "bundle_id",
                "scene_key",
                "task_key",
                "json_output_contract",
                "json_output_contract_answer_only",
                "answer_hint_rpg_house_missing_patch_label",
                "annotation_hint_rpg_house_missing_patch_label",
                "json_example_rpg_house_missing_patch_label",
                "json_example_answer_only_rpg_house_missing_patch_label",
            ],
            context=f"prompt defaults for {TASK_ID}",
        )
        prompt_artifacts = build_rpg_house_prompt_artifacts(
            domain=self.domain,
            scene_id=SCENE_ID,
            prompt_defaults=prompt_defaults,
            prompt_query_key=PROMPT_QUERY_KEY,
            slots={
                "json_output_contract": str(prompt_defaults["json_output_contract"]),
                "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
                "answer_hint": str(prompt_defaults["answer_hint_rpg_house_missing_patch_label"]),
                "annotation_hint": str(prompt_defaults["annotation_hint_rpg_house_missing_patch_label"]),
                "json_example": str(prompt_defaults["json_example_rpg_house_missing_patch_label"]),
                "json_example_answer_only": str(prompt_defaults["json_example_answer_only_rpg_house_missing_patch_label"]),
            },
            instance_seed=int(instance_seed),
        )
        trace_payload = {
            "scene_ir": rpg_house_scene_ir(
                domain=self.domain,
                scene_id=SCENE_ID,
                scene=source_scene,
                relations={
                    "query_id": str(sample.query_id),
                    "prompt_query_key": PROMPT_QUERY_KEY,
                    "answer_label": answer_label,
                    "patch_mode": PATCH_MODE_PLAIN,
                },
            ),
            "query_spec": {
                "task_id": TASK_ID,
                "query_id": str(sample.query_id),
                "prompt_query_key": PROMPT_QUERY_KEY,
                "prompt_variant_active_key": prompt_artifacts.prompt_variant_active_key,
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
                "params": {
                    "query_id": str(sample.query_id),
                    "query_id_probabilities": dict(sample.query_probabilities),
                    "prompt_query_key": PROMPT_QUERY_KEY,
                    "patch_mode": PATCH_MODE_PLAIN,
                    "source_room_count": int(sample.source_room_count),
                    "source_room_count_probabilities": dict(sample.source_room_count_probabilities),
                    "option_count": int(sample.option_count),
                    "option_count_support": list(option_count_support(params=params, defaults=_GEN_DEFAULTS, fallback=_DEFAULTS.option_count_support)),
                    "option_count_probabilities": dict(sample.option_count_probabilities),
                    "option_labels": list(DEFAULT_OPTION_LABELS[: int(sample.option_count)]),
                    "correct_index": int(sample.correct_index),
                    "correct_index_probabilities": dict(sample.correct_index_probabilities),
                    "answer_label": answer_label,
                    "patch_size": [int(sample.patch_size[0]), int(sample.patch_size[1])],
                    "patch_size_ratio": dict(sample.patch_size_trace),
                    "crop_margin_px": int(sample.crop_margin_px),
                    "source_size": [int(sample.source_size[0]), int(sample.source_size[1])],
                    **dict(sample.source_profile_trace),
                },
            },
            "render_spec": {
                "canvas_size": [int(artifacts.image.width), int(artifacts.image.height)],
                "coord_space": "pixel",
                "scene_id": SCENE_ID,
                "source_scene_canvas_size": [int(source_scene.image.width), int(source_scene.image.height)],
                "source_profile": dict(sample.source_profile_trace),
                "style": {
                    **rpg_house_source_style_trace(source_scene),
                    "patch_frame_style": style_trace(frame_style),
                    "patch_label_font": dict(label_font_trace),
                },
            },
            "render_map": {
                "image_id": "img0",
                "source_bboxes_px": _source_bboxes(source_scene),
                "missing_region_bbox_px": list(artifacts.missing_region_bbox),
                "option_bboxes_px_by_label": {str(key): list(value) for key, value in artifacts.option_bboxes.items()},
                "selected_option_bbox_px": list(artifacts.selected_option_bbox),
                "source_crop_box_px": [int(value) for value in artifacts.source_crop_box],
                "option_source_crop_boxes_px": [[int(coord) for coord in box] for box in artifacts.option_source_crop_boxes],
                "selected_transform": str(artifacts.selected_transform),
                "option_grid_shape": [int(artifacts.option_grid_shape[0]), int(artifacts.option_grid_shape[1])],
                "candidate_crop_count": int(len(candidate_crop_boxes)),
                "source_size": [int(sample.source_size[0]), int(sample.source_size[1])],
                "source_scene_canvas_size": [int(source_scene.image.width), int(source_scene.image.height)],
                "pre_downscale_canvas_size": [int(value) for value in artifacts.pre_downscale_canvas_size],
                "output_scale_xy": [float(value) for value in artifacts.output_scale_xy],
            },
            "execution_trace": {
                "query_id": str(sample.query_id),
                "prompt_query_key": PROMPT_QUERY_KEY,
                "scene_id": SCENE_ID,
                "answer": answer_label,
                "answer_label": answer_label,
                "correct_index": int(sample.correct_index),
                "selected_transform": str(artifacts.selected_transform),
                "source_crop_box_px": [int(value) for value in artifacts.source_crop_box],
                "option_labels": list(DEFAULT_OPTION_LABELS[: int(sample.option_count)]),
                "source_scene": dict(source_scene.trace),
            },
            "witness_symbolic": {
                "answer_label": answer_label,
                "missing_region_bbox": list(artifacts.missing_region_bbox),
                "selected_option_bbox": list(artifacts.selected_option_bbox),
            },
            "projected_annotation": {
                "type": "bbox_map",
                "bbox_map": dict(annotation_value),
                "pixel_bbox_map": dict(annotation_value),
            },
        }
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants={str(key): str(value) for key, value in prompt_artifacts.prompt_variants.items()},
            answer_gt=TypedValue(type="option_letter", value=answer_label),
            annotation_gt=TypedValue(type="bbox_map", value=dict(annotation_value)),
            image=artifacts.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(sample.query_id),
        )


__all__ = [
    "IllustrationsRpgHouseMissingPatchLabelTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
