"""Count icons by set relation between two icon panels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from ....core.seed import hash64, spawn_rng
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import (
    group_default,
    load_scene_generation_rendering_prompt_defaults,
)
from ...shared.fixed_query import select_task_query_id
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import build_prompt_query_spec
from ..shared.annotation import icon_bbox_set_annotation
from ..shared.icon_task_rendering import resolve_icon_render_params

from .shared.annotations import bboxes_from_icon_indices
from .shared.defaults import SCENE_ID, PairedCanvasDefaults
from .shared.output import build_paired_canvas_trace_payload
from .shared.prompts import build_paired_prompt, required_paired_prompt_defaults
from .shared.rendering import render_paired_canvas
from .shared.sampling import (
    load_icon_pool_from_params,
    make_icon_spec,
    resolve_paired_counts,
    sample_base_attributes,
    sample_palette,
    sample_positions,
)


DOMAIN = "icons"
TASK_ID = "task_icons__paired_canvas__panel_set_relation_count"
QUERY_IDS: Tuple[str, ...] = (
    "added_in_right_count",
    "missing_from_right_count",
)


@dataclass(frozen=True)
class _SetRelationScene:
    """Task-owned symbolic payload for a set-relation count instance."""

    image: Any
    panel_geometry: Dict[str, Any]
    left_icons: Tuple[Dict[str, Any], ...]
    right_icons: Tuple[Dict[str, Any], ...]
    matching_right_indices: Tuple[int, ...]
    matching_left_indices: Tuple[int, ...]
    target_count: int
    object_count: int
    distractor_count: int
    sampled_palette_rgb: Tuple[Tuple[int, int, int], ...]
    object_count_probabilities: Dict[str, float]
    target_count_probabilities: Dict[str, float]
    distractor_count_probabilities: Dict[str, float]
    question_format: str
    trace_relation: Dict[str, Any]


_DEFAULTS = PairedCanvasDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
    task_id=TASK_ID,
)


def _select_query(instance_seed: int, params: Mapping[str, Any]) -> Tuple[str, Dict[str, float], Dict[str, Any]]:
    """Select and validate one semantic set-relation query branch."""

    return select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=QUERY_IDS,
        default_query_id=QUERY_IDS[0],
        task_id=TASK_ID,
        namespace=f"{TASK_ID}.query",
    )


def _rotation_candidates(params: Mapping[str, Any]) -> Tuple[int, ...]:
    raw = params.get(
        "rotation_candidates_degrees",
        group_default(_GEN_DEFAULTS, "rotation_candidates_degrees", _DEFAULTS.rotation_candidates_degrees),
    )
    return tuple(int(value) for value in raw)


def _min_gap(params: Mapping[str, Any]) -> float:
    return float(
        params.get(
            "min_center_gap_frac",
            group_default(_RENDER_DEFAULTS, "min_center_gap_frac", _DEFAULTS.min_center_gap_frac),
        )
    )


def _make_added_removed_scene(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_params: Mapping[str, Any],
    query_id: str,
) -> _SetRelationScene:
    """Render a scene where counted icons are added to or missing from Right."""

    rng = spawn_rng(int(instance_seed), "scene")
    (
        diff_object_count,
        object_count_probabilities,
        target_count,
        target_count_probabilities,
        distractor_count,
        distractor_count_probabilities,
    ) = resolve_paired_counts(
        rng,
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        defaults=_DEFAULTS,
    )
    common_min = int(params.get("common_count_min", group_default(_GEN_DEFAULTS, "common_count_min", 3)))
    common_max = int(params.get("common_count_max", group_default(_GEN_DEFAULTS, "common_count_max", 6)))
    common_count = int(rng.randint(common_min, max(common_min, common_max)))
    added_count = int(target_count if query_id == "added_in_right_count" else distractor_count)
    removed_count = int(target_count if query_id == "missing_from_right_count" else distractor_count)
    total_unique = int(common_count) + int(added_count) + int(removed_count)

    pool = list(load_icon_pool_from_params(params=params, gen_defaults=_GEN_DEFAULTS, defaults=_DEFAULTS))
    rng.shuffle(pool)
    if len(pool) < total_unique:
        raise ValueError("icon pool is too small for paired added/removed scene")
    palette = sample_palette(rng, render_params=render_params)
    attrs = sample_base_attributes(
        rng,
        pool=pool,
        palette=palette,
        count=total_unique,
        render_params=render_params,
        rotation_candidates=_rotation_candidates(params),
    )
    common_attrs = attrs[:common_count]
    added_attrs = attrs[common_count : common_count + added_count]
    removed_attrs = attrs[common_count + added_count :]
    left_attrs = list(common_attrs) + list(removed_attrs)
    right_attrs = list(common_attrs) + list(added_attrs)
    rng.shuffle(left_attrs)
    rng.shuffle(right_attrs)

    left_positions = sample_positions(rng, count=len(left_attrs), min_gap_frac=_min_gap(params))
    right_positions = sample_positions(rng, count=len(right_attrs), min_gap_frac=_min_gap(params))
    left_specs = [
        make_icon_spec(
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}:left:{index}",
            render_params=render_params,
            instance_id=f"left_{index}",
            identity_id=str(attr["identity_id"]),
            icon_id=str(attr["icon_id"]),
            panel="left",
            position=pos,
            tint_rgb=tuple(attr["tint_rgb"]),
            size_px=int(attr["size_px"]),
            rotation_degrees=int(attr["rotation_degrees"]),
        )
        for index, (attr, pos) in enumerate(zip(left_attrs, left_positions))
    ]
    right_specs = [
        make_icon_spec(
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}:right:{index}",
            render_params=render_params,
            instance_id=f"right_{index}",
            identity_id=str(attr["identity_id"]),
            icon_id=str(attr["icon_id"]),
            panel="right",
            position=pos,
            tint_rgb=tuple(attr["tint_rgb"]),
            size_px=int(attr["size_px"]),
            rotation_degrees=int(attr["rotation_degrees"]),
        )
        for index, (attr, pos) in enumerate(zip(right_attrs, right_positions))
    ]
    rendered = render_paired_canvas(left_icons=left_specs, right_icons=right_specs, render_params=render_params)
    added_ids = {str(attr["identity_id"]) for attr in added_attrs}
    removed_ids = {str(attr["identity_id"]) for attr in removed_attrs}
    return _SetRelationScene(
        image=rendered.image,
        panel_geometry=dict(rendered.panel_geometry),
        left_icons=tuple(dict(item) for item in rendered.left_icons),
        right_icons=tuple(dict(item) for item in rendered.right_icons),
        matching_right_indices=tuple(
            index
            for index, icon in enumerate(rendered.right_icons)
            if str(icon["identity_id"]) in added_ids
        ),
        matching_left_indices=tuple(
            index
            for index, icon in enumerate(rendered.left_icons)
            if str(icon["identity_id"]) in removed_ids
        ),
        target_count=int(target_count),
        object_count=int(diff_object_count),
        distractor_count=int(distractor_count),
        sampled_palette_rgb=tuple(palette),
        object_count_probabilities=dict(object_count_probabilities),
        target_count_probabilities=dict(target_count_probabilities),
        distractor_count_probabilities=dict(distractor_count_probabilities),
        question_format="count_icons_added_to_or_missing_from_right_panel",
        trace_relation={
            "counting_target": str(query_id),
            "common_count": int(common_count),
            "added_count": int(added_count),
            "removed_count": int(removed_count),
            "left_count": len(rendered.left_icons),
            "right_count": len(rendered.right_icons),
        },
    )


def _make_scene(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_params: Mapping[str, Any],
    query_id: str,
) -> _SetRelationScene:
    return _make_added_removed_scene(
        instance_seed=int(instance_seed),
        params=params,
        render_params=render_params,
        query_id=str(query_id),
    )


def _question_text(prompt_defaults: Mapping[str, Any], *, query_id: str) -> str:
    key = f"question_text_{query_id}"
    return str(prompt_defaults.get(key, prompt_defaults.get("question_text", "")))


def _selected_annotation(scene: _SetRelationScene, *, query_id: str) -> tuple[str, list[list[int]]]:
    annotation_panel = "left" if str(query_id) == "missing_from_right_count" else "right"
    return annotation_panel, bboxes_from_icon_indices(
        panel_icons=scene.left_icons if annotation_panel == "left" else scene.right_icons,
        indices=scene.matching_left_indices if annotation_panel == "left" else scene.matching_right_indices,
    )


@register_task
class IconsCountingPanelSetRelationCountTask:
    """Count additions or removals across paired icon panels."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'logical_composition')
    domain = DOMAIN
    supported_query_ids = QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one deterministic paired-panel set-relation count instance."""

        query_id, query_probabilities, task_params = _select_query(int(instance_seed), params)
        render_params = resolve_icon_render_params(
            params=task_params,
            render_defaults=_RENDER_DEFAULTS,
            fallback_defaults=_DEFAULTS,
            instance_seed=int(instance_seed),
        )
        scene = None
        last_error: Exception | None = None
        for attempt_index in range(max(1, int(max_attempts))):
            try:
                attempt_seed = int(hash64(int(instance_seed), TASK_ID, int(attempt_index)))
                scene = _make_scene(
                    instance_seed=attempt_seed,
                    params=task_params,
                    render_params=render_params,
                    query_id=str(query_id),
                )
                break
            except Exception as exc:
                last_error = exc
                continue
        if scene is None:
            raise RuntimeError(f"failed to generate {TASK_ID} instance") from last_error

        prompt_defaults = required_paired_prompt_defaults(_PROMPT_DEFAULTS, run_namespace=TASK_ID)
        prompt_artifacts = build_paired_prompt(
            domain=self.domain,
            prompt_defaults=prompt_defaults,
            question_text=_question_text(prompt_defaults, query_id=str(query_id)),
            instance_seed=int(instance_seed),
        )
        annotation_panel, annotation_bboxes = _selected_annotation(scene, query_id=str(query_id))
        annotation_artifacts = icon_bbox_set_annotation(annotation_bboxes)
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(query_id),
            params={
                "query_id_probabilities": dict(query_probabilities),
                "object_count": int(scene.object_count),
                "object_count_probabilities": dict(scene.object_count_probabilities),
                "target_count": int(scene.target_count),
                "target_count_probabilities": dict(scene.target_count_probabilities),
                "distractor_count": int(scene.distractor_count),
                "distractor_count_probabilities": dict(scene.distractor_count_probabilities),
                "annotation_panel": str(annotation_panel),
            },
        )
        execution_trace = {
            "scene_variant": SCENE_ID,
            "query_id": str(query_id),
            "query_id_probabilities": dict(query_probabilities),
            "question_format": str(scene.question_format),
            "object_count": int(scene.object_count),
            "object_count_probabilities": dict(scene.object_count_probabilities),
            "target_count": int(scene.target_count),
            "target_count_probabilities": dict(scene.target_count_probabilities),
            "distractor_count": int(scene.distractor_count),
            "distractor_count_probabilities": dict(scene.distractor_count_probabilities),
            "matching_right_indices": list(scene.matching_right_indices),
            "matching_left_indices": list(scene.matching_left_indices),
            "annotation_panel": str(annotation_panel),
            **dict(scene.trace_relation),
        }
        trace_payload = build_paired_canvas_trace_payload(
            scene_kind=f"icons_{SCENE_ID}",
            panel_geometry=scene.panel_geometry,
            left_icons=scene.left_icons,
            right_icons=scene.right_icons,
            relations=scene.trace_relation,
            query_spec=query_spec,
            render_params=render_params,
            sampled_palette_rgb=scene.sampled_palette_rgb,
            render_map_extra=None,
            execution_trace=execution_trace,
            witness_symbolic={
                "query_id": str(query_id),
                "matching_right_indices": list(scene.matching_right_indices),
                "matching_left_indices": list(scene.matching_left_indices),
                "annotation_panel": str(annotation_panel),
            },
            annotation_payload=annotation_artifacts,
        )
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            answer_gt=TypedValue(type="integer", value=int(scene.target_count)),
            annotation_gt=TypedValue(
                type=str(annotation_artifacts["annotation_type"]),
                value=[list(bbox) for bbox in annotation_artifacts["annotation_value"]],
            ),
            image=scene.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(query_id),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
        )


__all__ = ["IconsCountingPanelSetRelationCountTask", "QUERY_IDS", "TASK_ID"]
