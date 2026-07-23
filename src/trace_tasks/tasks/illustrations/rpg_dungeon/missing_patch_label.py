"""Select the patch option that restores an RPG dungeon source panel."""

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
from trace_tasks.tasks.illustrations.shared.rpg_tile_profiles import resolve_rpg_tile_render_params

from .shared.output import (
    blocker_bbox_map,
    entity_bbox_map,
    rpg_dungeon_render_spec,
    rpg_dungeon_scene_ir,
)
from .shared.prompts import build_rpg_dungeon_prompt_artifacts
from .shared.rendering import (
    DEFAULT_TILE_PX,
    MAX_MONSTER_CHAMBER_COUNT,
    MAX_TOTAL_CHEST_COUNT,
    MIN_TOTAL_CHEST_COUNT,
    SCENE_ID,
    render_rpg_dungeon_profile_scene,
)
from .shared.sampling import select_count_from_support
from .shared.state import RpgDungeonScene


TASK_ID = "task_illustrations__rpg_dungeon__missing_patch_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "missing_patch_label"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class _Defaults:
    source_chest_count_min: int = 5
    source_chest_count_max: int = 6
    option_count_support: Tuple[int, ...] = (4, 6)
    patch_width_ratio_min: float = 0.15
    patch_width_ratio_max: float = 0.30
    patch_height_ratio_min: float = 0.15
    patch_height_ratio_max: float = 0.26
    patch_area_ratio_max: float = 0.065
    crop_margin_px: int = 36


@dataclass(frozen=True)
class _SampleSpec:
    query_id: str
    source_chest_count: int
    source_reachable_chest_count: int
    source_monster_count: int
    option_count: int
    correct_index: int
    patch_size: Tuple[int, int]
    patch_size_trace: Dict[str, Any]
    crop_margin_px: int
    render_params: Dict[str, Any]
    query_probabilities: Dict[str, float]
    source_chest_count_probabilities: Dict[str, float]
    source_reachable_chest_count_probabilities: Dict[str, float]
    source_monster_count_probabilities: Dict[str, float]
    option_count_probabilities: Dict[str, float]
    correct_index_probabilities: Dict[str, float]


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


def _option_count_support(params: Mapping[str, Any]) -> Tuple[int, ...]:
    raw = params.get("option_count_support", group_default(_GEN_DEFAULTS, "option_count_support", _DEFAULTS.option_count_support))
    raw_values = (raw,) if isinstance(raw, int) else tuple(raw if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) else ())
    support = tuple(dict.fromkeys(int(value) for value in raw_values if int(value) in set(_DEFAULTS.option_count_support)))
    if not support:
        raise ValueError("option_count_support must include 4 or 6")
    return support


def _sample_spec(*, instance_seed: int, params: Mapping[str, Any], attempt_index: int) -> _SampleSpec:
    """Sample source-scene density and patch-option parameters before rendering pixels."""

    query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=SUPPORTED_QUERY_IDS,
        default_query_id=SINGLE_QUERY_ID,
        task_id=TASK_ID,
        namespace=f"{TASK_ID}:query",
    )
    source_chest_count, chest_probabilities = select_count_from_support(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="source_chest_count_support",
        explicit_key="source_chest_count",
        fallback_support=tuple(range(_DEFAULTS.source_chest_count_min, _DEFAULTS.source_chest_count_max + 1)),
        namespace=f"{TASK_ID}:source_chest_count",
        max_value=MAX_TOTAL_CHEST_COUNT,
    )
    source_reachable_count, reachable_probabilities = select_count_from_support(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="source_reachable_chest_count_support",
        explicit_key="source_reachable_chest_count",
        fallback_support=tuple(range(1, int(source_chest_count))),
        namespace=f"{TASK_ID}:source_reachable_chest_count",
        max_value=int(source_chest_count) - 1,
    )
    source_monster_count, monster_probabilities = select_count_from_support(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="source_monster_count_support",
        explicit_key="source_monster_count",
        fallback_support=tuple(range(1, min(MAX_MONSTER_CHAMBER_COUNT, int(source_chest_count)) + 1)),
        namespace=f"{TASK_ID}:source_monster_count",
        max_value=min(MAX_MONSTER_CHAMBER_COUNT, int(source_chest_count)),
    )
    option_count, option_probabilities = select_count_from_support(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="option_count_support",
        explicit_key="option_count",
        fallback_support=_option_count_support(task_params),
        namespace=f"{TASK_ID}:option_count",
    )
    correct_index, correct_probabilities = select_count_from_support(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="correct_index_support",
        explicit_key="correct_index",
        fallback_support=tuple(range(int(option_count))),
        namespace=f"{TASK_ID}:correct_index",
        max_value=int(option_count) - 1,
    )
    render_params = resolve_rpg_tile_render_params(
        task_params,
        _RENDER_DEFAULTS,
        tile_px_key="rpg_dungeon_tile_px",
        fallback_tile_px=DEFAULT_TILE_PX,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:source_profile",
    )
    source_size = (int(render_params["canvas_width"]), int(render_params["canvas_height"]))
    patch_sample = sample_missing_patch_size(
        rng=spawn_rng(int(instance_seed), f"{TASK_ID}:patch_spec", int(attempt_index)),
        params=task_params,
        defaults=_GEN_DEFAULTS,
        source_size=source_size,
        fallback_width_ratio_min=_DEFAULTS.patch_width_ratio_min,
        fallback_width_ratio_max=_DEFAULTS.patch_width_ratio_max,
        fallback_height_ratio_min=_DEFAULTS.patch_height_ratio_min,
        fallback_height_ratio_max=_DEFAULTS.patch_height_ratio_max,
        fallback_area_ratio_max=_DEFAULTS.patch_area_ratio_max,
    )
    return _SampleSpec(
        query_id=str(query_id),
        source_chest_count=int(source_chest_count),
        source_reachable_chest_count=int(source_reachable_count),
        source_monster_count=int(source_monster_count),
        option_count=int(option_count),
        correct_index=int(correct_index),
        patch_size=tuple(int(value) for value in patch_sample.patch_size),
        patch_size_trace=dict(patch_sample.trace()),
        crop_margin_px=int(task_params.get("crop_margin_px", group_default(_GEN_DEFAULTS, "crop_margin_px", _DEFAULTS.crop_margin_px))),
        render_params=dict(render_params),
        query_probabilities=dict(query_probabilities),
        source_chest_count_probabilities=dict(chest_probabilities),
        source_reachable_chest_count_probabilities=dict(reachable_probabilities),
        source_monster_count_probabilities=dict(monster_probabilities),
        option_count_probabilities=dict(option_probabilities),
        correct_index_probabilities=dict(correct_probabilities),
    )


def _candidate_crop_boxes_from_scene(
    *,
    scene: RpgDungeonScene,
    patch_size: Sequence[int],
    margin_px: int,
) -> Tuple[Tuple[int, int, int, int], ...]:
    """Build patch crops centered on visible chambers, entities, and boulders."""

    width, height = scene.image.size
    patch_w, patch_h = int(patch_size[0]), int(patch_size[1])
    margin = int(margin_px)
    max_x0 = int(width) - patch_w - margin
    max_y0 = int(height) - patch_h - margin
    if max_x0 < margin or max_y0 < margin:
        return tuple()

    def crop_for_box(box: Sequence[float]) -> Tuple[int, int, int, int]:
        cx = int(round((float(box[0]) + float(box[2])) * 0.5))
        cy = int(round((float(box[1]) + float(box[3])) * 0.5))
        x0 = min(max(margin, cx - patch_w // 2), max_x0)
        y0 = min(max(margin, cy - patch_h // 2), max_y0)
        return (int(x0), int(y0), int(x0 + patch_w), int(y0 + patch_h))

    boxes: list[Tuple[int, int, int, int]] = []
    for entity in scene.entities:
        boxes.append(crop_for_box(entity.bbox_xyxy))
    for blocker in scene.blockers:
        boxes.append(crop_for_box(blocker.bbox_xyxy))
    for chamber in scene.chambers:
        if str(chamber.chamber_id) != "start":
            boxes.append(crop_for_box(chamber.bbox_xyxy))
    return tuple(dict.fromkeys(boxes))


def _source_bboxes(scene: RpgDungeonScene) -> Dict[str, Any]:
    return {
        "chambers": {
            str(chamber.chamber_id): [round(float(value), 3) for value in chamber.bbox_xyxy]
            for chamber in scene.chambers
        },
        "entities": entity_bbox_map(scene),
        "blockers": blocker_bbox_map(scene),
    }


@register_task
class IllustrationsRpgDungeonMissingPatchLabelTask:
    """Select the patch option that restores a missing RPG dungeon region."""

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
                source_scene = render_rpg_dungeon_profile_scene(
                    int(instance_seed),
                    render_params=sample.render_params,
                    tile_px=int(sample.render_params["tile_px"]),
                    reachable_chest_count=int(sample.source_reachable_chest_count),
                    total_chest_count=int(sample.source_chest_count),
                    monster_chamber_count=int(sample.source_monster_count),
                )
                candidate_crop_boxes = _candidate_crop_boxes_from_scene(
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
                    candidate_crop_boxes=candidate_crop_boxes or None,
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
                "answer_hint_rpg_dungeon_missing_patch_label",
                "annotation_hint_rpg_dungeon_missing_patch_label",
                "json_example_rpg_dungeon_missing_patch_label",
                "json_example_answer_only_rpg_dungeon_missing_patch_label",
            ],
            context=f"prompt defaults for {TASK_ID}",
        )
        prompt_artifacts = build_rpg_dungeon_prompt_artifacts(
            domain=self.domain,
            scene_id=SCENE_ID,
            prompt_defaults=prompt_defaults,
            prompt_query_key=PROMPT_QUERY_KEY,
            slots={
                "json_output_contract": str(prompt_defaults["json_output_contract"]),
                "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
                "answer_hint": str(prompt_defaults["answer_hint_rpg_dungeon_missing_patch_label"]),
                "annotation_hint": str(prompt_defaults["annotation_hint_rpg_dungeon_missing_patch_label"]),
                "json_example": str(prompt_defaults["json_example_rpg_dungeon_missing_patch_label"]),
                "json_example_answer_only": str(prompt_defaults["json_example_answer_only_rpg_dungeon_missing_patch_label"]),
            },
            instance_seed=int(instance_seed),
        )
        render_spec = rpg_dungeon_render_spec(source_scene, scene_id=SCENE_ID)
        trace_payload = {
            "scene_ir": _json_safe(
                rpg_dungeon_scene_ir(
                    domain=self.domain,
                    scene_id=SCENE_ID,
                    scene=source_scene,
                    relations={
                        "query_id": str(sample.query_id),
                        "prompt_query_key": PROMPT_QUERY_KEY,
                        "answer_label": answer_label,
                        "patch_mode": PATCH_MODE_PLAIN,
                    },
                )
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
                    "source_chest_count": int(sample.source_chest_count),
                    "source_chest_count_probabilities": dict(sample.source_chest_count_probabilities),
                    "source_reachable_chest_count": int(sample.source_reachable_chest_count),
                    "source_reachable_chest_count_probabilities": dict(sample.source_reachable_chest_count_probabilities),
                    "source_monster_count": int(sample.source_monster_count),
                    "source_monster_count_probabilities": dict(sample.source_monster_count_probabilities),
                    "option_count": int(sample.option_count),
                    "option_count_support": list(_option_count_support(params)),
                    "option_count_probabilities": dict(sample.option_count_probabilities),
                    "option_labels": list(DEFAULT_OPTION_LABELS[: int(sample.option_count)]),
                    "correct_index": int(sample.correct_index),
                    "correct_index_probabilities": dict(sample.correct_index_probabilities),
                    "answer_label": answer_label,
                    "patch_size": [int(sample.patch_size[0]), int(sample.patch_size[1])],
                    "patch_size_ratio": dict(sample.patch_size_trace),
                    "crop_margin_px": int(sample.crop_margin_px),
                    "source_size": [int(sample.render_params["canvas_width"]), int(sample.render_params["canvas_height"])],
                    "canvas_profile": str(sample.render_params.get("canvas_profile", "")),
                    "canvas_profile_probabilities": dict(sample.render_params.get("canvas_profile_probabilities", {})),
                    "rpg_tile_profile": _json_safe(dict(sample.render_params.get("rpg_tile_profile", {}))),
                },
            },
            "render_spec": {
                **render_spec,
                "canvas_size": [int(artifacts.image.width), int(artifacts.image.height)],
                "source_scene_canvas_size": [int(source_scene.image.width), int(source_scene.image.height)],
                "source_profile": {
                    "canvas_profile": str(sample.render_params.get("canvas_profile", "")),
                    "canvas_profile_size": list(sample.render_params.get("canvas_profile_size", [])),
                    "canvas_profile_probabilities": dict(sample.render_params.get("canvas_profile_probabilities", {})),
                    "rpg_tile_profile": _json_safe(dict(sample.render_params.get("rpg_tile_profile", {}))),
                },
                "style": {
                    **_json_safe(dict(render_spec["style"])),
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
                "source_size": [int(sample.render_params["canvas_width"]), int(sample.render_params["canvas_height"])],
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
                "source_scene": _json_safe(dict(source_scene.trace)),
            },
            "witness_symbolic": {
                "answer_label": answer_label,
                "missing_region_bbox": list(artifacts.missing_region_bbox),
                "selected_option_bbox": list(artifacts.selected_option_bbox),
            },
            "projected_annotation": {
                "type": "bbox_map",
                "bbox_map": _json_safe(dict(annotation_value)),
                "pixel_bbox_map": _json_safe(dict(annotation_value)),
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
    "IllustrationsRpgDungeonMissingPatchLabelTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
