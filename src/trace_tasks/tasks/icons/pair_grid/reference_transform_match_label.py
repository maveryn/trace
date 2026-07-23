"""Select the Scene cell with the same icon-pair transform as Reference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import (
    group_default,
    load_scene_generation_rendering_prompt_defaults,
    required_group_defaults,
)
from ...shared.fixed_query import select_task_query_id
from ...shared.labeling import LABEL_POOL_A_L
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_query_spec,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)
from ..shared.annotation import bbox_annotation
from ..shared.icon_assets import icon_transform_signature, resolve_icon_pool
from ..shared.icon_style import sample_single_icon_tint
from ..shared.icon_task_rendering import sample_icon_instance_noise
from ..shared.icon_transform import IDENTITY_TRANSFORM_ID

from .shared.defaults import PairGridTaskDefaults
from .shared.output import render_pair_grid_payload, selected_scene_cell_bbox
from .shared.state import IconPairSpec
from .shared.styles import pair_grid_style_trace, resolve_pair_grid_render_params


@dataclass(frozen=True)
class _ScenePayload:
    """Trace-ready payload for one transform-match selection scene."""

    option_count: int
    reference_transform_id: str
    reference_icon_id: str
    answer_label: str
    cell_labels: Tuple[str, ...]
    cell_icon_ids: Tuple[str, ...]
    cell_transform_ids: Tuple[str, ...]
    sampled_palette_rgb: Tuple[Tuple[int, int, int], ...]
    panel_geometry: Dict[str, Any]
    reference_pair: Dict[str, Any]
    scene_cells: Tuple[Dict[str, Any], ...]


_SUPPORTED_TRANSFORM_IDS: Tuple[str, ...] = ("rot90", "rot180", "rot270", "flip_h", "flip_v")


_DEFAULTS = PairGridTaskDefaults(
    option_count=4,
    pool_manifest="non_symmetry.txt",
    palette_size_min=1,
    palette_size_max=1,
    min_color_distance=40.0,
    transform_ids=_SUPPORTED_TRANSFORM_IDS,
)
TASK_ID = "task_icons__pair_grid__reference_transform_match_label"
DOMAIN = "icons"
SCENE_ID = "pair_grid"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = ("single",)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
    task_id=TASK_ID,
)


def _select_query(instance_seed: int, params: Mapping[str, Any]) -> Tuple[str, Dict[str, float], Dict[str, Any]]:
    """Select and validate the single public query contract."""

    return select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=SUPPORTED_QUERY_IDS,
        default_query_id="single",
        task_id=TASK_ID,
        namespace=f"{TASK_ID}.query",
    )


def _resolve_option_count(params: Mapping[str, Any]) -> Tuple[int, Dict[str, float]]:
    """Resolve the fixed number of visible labeled options."""

    option_count = int(params.get("option_count", group_default(_GEN_DEFAULTS, "option_count", _DEFAULTS.option_count)))
    if not 2 <= int(option_count) <= len(LABEL_POOL_A_L):
        raise ValueError(f"option_count must be in [2, {len(LABEL_POOL_A_L)}]")
    return int(option_count), {str(option_count): 1.0}


def _resolve_answer_label(rng, *, params: Mapping[str, Any], labels: Sequence[str]) -> Tuple[str, Dict[str, float]]:
    """Resolve the unique correct option label."""

    label_set = {str(label) for label in labels}
    explicit = params.get("answer_label")
    if explicit is not None:
        selected = str(explicit).strip().upper()
        if selected not in label_set:
            raise ValueError(f"answer_label must be one of {sorted(label_set)}")
    else:
        selected = str(rng.choice(tuple(labels)))
    probability = 1.0 / float(len(labels))
    return selected, {str(label): probability for label in labels}


def _resolve_transform_ids(params: Mapping[str, Any]) -> Tuple[str, ...]:
    """Resolve supported non-identity transform ids."""

    raw = params.get("transform_ids", group_default(_GEN_DEFAULTS, "transform_ids", list(_DEFAULTS.transform_ids)))
    if not isinstance(raw, (list, tuple)):
        raise ValueError("transform_ids must be a sequence")
    transform_ids = tuple(str(value).strip() for value in raw if str(value).strip())
    if not transform_ids:
        raise ValueError("transform_ids resolved no transforms")
    unsupported = [value for value in transform_ids if value not in set(_SUPPORTED_TRANSFORM_IDS)]
    if unsupported:
        raise ValueError(f"unsupported transform_ids: {unsupported}")
    return transform_ids


def _distinct_distractor_transforms(
    icon_id: str,
    *,
    reference_transform_id: str,
    check_size_px: int,
    transform_ids: Sequence[str],
) -> Tuple[str, ...]:
    """Return non-identity transforms visually distinct from the reference transform."""

    identity_signature = icon_transform_signature(str(icon_id), int(check_size_px), IDENTITY_TRANSFORM_ID)
    reference_signature = icon_transform_signature(str(icon_id), int(check_size_px), str(reference_transform_id))
    if reference_signature == identity_signature:
        return ()
    distractors = [
        str(transform_id)
        for transform_id in transform_ids
        if str(transform_id) != str(reference_transform_id)
        and icon_transform_signature(str(icon_id), int(check_size_px), str(transform_id)) != identity_signature
        and icon_transform_signature(str(icon_id), int(check_size_px), str(transform_id)) != reference_signature
    ]
    return tuple(str(value) for value in distractors)


def _sample_scene(
    rng,
    *,
    instance_seed: int,
    option_count: int,
    answer_label: str,
    pool_manifest: str,
    transform_ids: Sequence[str],
    transform_check_size_px: int,
    render_params: Mapping[str, Any],
) -> Tuple[_ScenePayload, Any]:
    """Sample and render one reference-pair transform-match label scene."""

    pool = list(resolve_icon_pool(str(pool_manifest)))
    if len(pool) < int(option_count) + 1:
        raise ValueError("icon pool is too small for requested transform-match scene")
    reference_transform_id = str(rng.choice(list(transform_ids)))

    candidate_records: List[Tuple[str, Tuple[str, ...]]] = []
    shuffled_pool = list(pool)
    rng.shuffle(shuffled_pool)
    for icon_id in shuffled_pool:
        distractors = _distinct_distractor_transforms(
            str(icon_id),
            reference_transform_id=str(reference_transform_id),
            check_size_px=int(transform_check_size_px),
            transform_ids=transform_ids,
        )
        if distractors:
            candidate_records.append((str(icon_id), tuple(str(value) for value in distractors)))
        if len(candidate_records) >= int(option_count) + 1:
            break
    if len(candidate_records) < int(option_count) + 1:
        raise ValueError("insufficient transform-distinct icons for transform-match scene")

    reference_icon_id, _ = candidate_records[0]
    scene_records = list(candidate_records[1 : 1 + int(option_count)])
    labels = tuple(str(value) for value in LABEL_POOL_A_L[: int(option_count)])
    answer_index = labels.index(str(answer_label))
    tint_rgb, sampled_palette_rgb = sample_single_icon_tint(
        rng,
        channel_min=int(render_params["color_channel_min"]),
        channel_max=int(render_params["color_channel_max"]),
        anchor_colors=(
            tuple(int(v) for v in render_params["background_color_rgb"]),
            tuple(int(v) for v in render_params["panel_fill_rgb"]),
            tuple(int(v) for v in render_params["panel_border_rgb"]),
            tuple(int(v) for v in render_params["header_text_rgb"]),
        ),
        min_color_distance=float(render_params["min_color_distance"]),
        distance_space=str(render_params["color_distance_space"]),
    )

    reference_pair = _make_pair_spec(
        icon_id=str(reference_icon_id),
        transform_id=str(reference_transform_id),
        tint_rgb=tuple(int(v) for v in tint_rgb),
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:reference",
        render_params=render_params,
    )

    scene_pairs: List[IconPairSpec] = []
    scene_icon_ids: List[str] = []
    scene_transform_ids: List[str] = []
    for index, (label, record) in enumerate(zip(labels, scene_records)):
        icon_id, distractor_options = record
        transform_id = str(reference_transform_id) if int(index) == int(answer_index) else str(rng.choice(list(distractor_options)))
        scene_icon_ids.append(str(icon_id))
        scene_transform_ids.append(str(transform_id))
        scene_pairs.append(
            _make_pair_spec(
                icon_id=str(icon_id),
                transform_id=str(transform_id),
                tint_rgb=tuple(int(v) for v in tint_rgb),
                instance_seed=int(instance_seed),
                namespace=f"{TASK_ID}:scene_{str(label)}",
                render_params=render_params,
            )
        )

    rendered_payload = render_pair_grid_payload(
        reference_pair=reference_pair,
        scene_pairs=scene_pairs,
        scene_labels=labels,
        render_params=render_params,
        matching_labels=(str(answer_label),),
        reference_extra={"target_relation": "geometric_transform_match"},
        cell_extra_by_label={
            str(label): {
                "target_relation": "geometric_transform_match",
                "is_correct_option": bool(str(label) == str(answer_label)),
            }
            for label in labels
        },
    )
    return _ScenePayload(
        option_count=int(option_count),
        reference_transform_id=str(reference_transform_id),
        reference_icon_id=str(reference_icon_id),
        answer_label=str(answer_label),
        cell_labels=tuple(str(value) for value in labels),
        cell_icon_ids=tuple(str(value) for value in scene_icon_ids),
        cell_transform_ids=tuple(str(value) for value in scene_transform_ids),
        sampled_palette_rgb=tuple(sampled_palette_rgb),
        panel_geometry=dict(rendered_payload.panel_geometry),
        reference_pair=dict(rendered_payload.reference_pair),
        scene_cells=tuple(dict(item) for item in rendered_payload.scene_cells),
    ), rendered_payload.image


def _make_pair_spec(
    *,
    icon_id: str,
    transform_id: str,
    tint_rgb: Tuple[int, int, int],
    instance_seed: int,
    namespace: str,
    render_params: Mapping[str, Any],
) -> IconPairSpec:
    """Build one before/after pair spec with deterministic visual noise."""

    left_noise_edits, left_noise_seed = sample_icon_instance_noise(
        instance_seed=int(instance_seed),
        namespace=f"{namespace}:left",
        render_params=render_params,
    )
    right_noise_edits, right_noise_seed = sample_icon_instance_noise(
        instance_seed=int(instance_seed),
        namespace=f"{namespace}:right",
        render_params=render_params,
    )
    return IconPairSpec(
        icon_id=str(icon_id),
        transform_id=str(transform_id),
        tint_rgb=tuple(int(v) for v in tint_rgb),
        left_noise_edits=tuple(left_noise_edits),
        left_noise_seed=int(left_noise_seed),
        right_noise_edits=tuple(right_noise_edits),
        right_noise_seed=int(right_noise_seed),
    )


@register_task
class IconsPairGridReferenceTransformMatchLabelTask:
    """Select the labeled cell that matches a Reference pair transform."""

    task_id = TASK_ID
    reasoning_operations = ('transformation', 'matching')
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one deterministic transform-match label instance."""

        query_id, query_probabilities, task_params = _select_query(int(instance_seed), params)
        scene_rng = spawn_rng(int(instance_seed), "scene")
        option_count, option_count_probabilities = _resolve_option_count(task_params)
        labels = tuple(str(value) for value in LABEL_POOL_A_L[: int(option_count)])
        answer_label, answer_label_probabilities = _resolve_answer_label(scene_rng, params=task_params, labels=labels)
        render_params = resolve_pair_grid_render_params(
            params=task_params,
            render_defaults=_RENDER_DEFAULTS,
            fallback_defaults=_DEFAULTS,
            instance_seed=int(instance_seed),
        )
        pool_manifest = str(task_params.get("pool_manifest", group_default(_GEN_DEFAULTS, "pool_manifest", _DEFAULTS.pool_manifest)))
        transform_ids = _resolve_transform_ids(task_params)
        transform_check_size_px = int(
            task_params.get(
                "transform_check_size_px",
                group_default(_GEN_DEFAULTS, "transform_check_size_px", _DEFAULTS.transform_check_size_px),
            )
        )

        scene_payload = None
        image = None
        last_error: Exception | None = None
        for _ in range(max(1, int(max_attempts))):
            try:
                scene_payload, image = _sample_scene(
                    scene_rng,
                    instance_seed=int(instance_seed),
                    option_count=int(option_count),
                    answer_label=str(answer_label),
                    pool_manifest=str(pool_manifest),
                    transform_ids=transform_ids,
                    transform_check_size_px=int(transform_check_size_px),
                    render_params=render_params,
                )
                break
            except Exception as exc:
                last_error = exc
                continue
        if scene_payload is None or image is None:
            raise RuntimeError(f"failed to generate {self.task_id} instance") from last_error

        prompt_defaults = required_group_defaults(
            _PROMPT_DEFAULTS,
            (
                "bundle_id",
                "scene_key",
                "task_key",
                "json_output_contract",
                "json_output_contract_answer_only",
                "object_description",
                "question_text",
                "annotation_hint",
                "answer_hint",
                "json_example",
                "json_example_answer_only",
            ),
            context=f"prompt defaults for {self.task_id}",
        )
        prompt_selection = render_scene_prompt_variants(
            domain=self.domain,
            scene_id=SCENE_ID,
            bundle_id=str(prompt_defaults["bundle_id"]),
            scene_key=str(prompt_defaults["scene_key"]),
            task_key=str(prompt_defaults["task_key"]),
            answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
            dynamic_slots={
                "object_description": str(prompt_defaults["object_description"]),
                "question_text": str(prompt_defaults["question_text"]),
                "json_output_contract": str(prompt_defaults["json_output_contract"]),
                "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
                "annotation_hint": str(prompt_defaults["annotation_hint"]),
                "answer_hint": str(prompt_defaults["answer_hint"]),
                "json_example": str(prompt_defaults["json_example"]),
                "json_example_answer_only": str(prompt_defaults["json_example_answer_only"]),
            },
            instance_seed=int(instance_seed),
        )
        prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)

        selected_bbox = selected_scene_cell_bbox(
            scene_cells=scene_payload.scene_cells,
            selected_label=str(scene_payload.answer_label),
        )
        annotation_artifacts = bbox_annotation(selected_bbox)
        answer_gt = TypedValue(type="option_letter", value=str(scene_payload.answer_label))
        annotation_gt = TypedValue(
            type=str(annotation_artifacts["annotation_type"]),
            value=list(annotation_artifacts["annotation_value"]),
        )
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(query_id),
            params={
                "task_id": str(self.task_id),
                "scene_id": SCENE_ID,
                "query_id_probabilities": dict(query_probabilities),
                "option_count": int(option_count),
                "option_count_probabilities": dict(option_count_probabilities),
                "answer_label_probabilities": dict(answer_label_probabilities),
                "pool_manifest": str(pool_manifest),
                "transform_ids": list(transform_ids),
                "transform_check_size_px": int(transform_check_size_px),
            },
        )

        trace_payload = {
            "scene_ir": {
                "scene_kind": "icons_reference_pair_transform_match_label",
                "task_id": str(self.task_id),
                "scene_id": SCENE_ID,
                "query_id": str(query_id),
                "entities": [dict(scene_payload.reference_pair), *[dict(item) for item in scene_payload.scene_cells]],
                "relations": {
                    "selection_target": "same_geometric_transform_as_reference",
                    "reference_transform_id": str(scene_payload.reference_transform_id),
                    "answer_label": str(scene_payload.answer_label),
                },
                "frames": {
                    "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
                    "panels": dict(scene_payload.panel_geometry),
                },
            },
            "query_spec": dict(query_spec),
            "render_spec": {
                "task_id": str(self.task_id),
                "scene_id": SCENE_ID,
                "query_id": str(query_id),
                "canvas_size": [int(render_params["canvas_width"]), int(render_params["canvas_height"])],
                "coord_space": "pixel",
                "panel_geometry": dict(scene_payload.panel_geometry),
                "style": pair_grid_style_trace(
                    render_params=render_params,
                    sampled_palette_rgb=scene_payload.sampled_palette_rgb,
                ),
            },
            "render_map": {
                "image_id": "img0",
                "anchors": {
                    "reference_pair": dict(scene_payload.reference_pair),
                    "answer_label": str(scene_payload.answer_label),
                    "answer_cell_bbox_xyxy": list(selected_bbox),
                    "scene_cells": [dict(item) for item in scene_payload.scene_cells],
                },
            },
            "execution_trace": {
                "scene_variant": "reference_pair_grid",
                "task_id": str(self.task_id),
                "scene_id": SCENE_ID,
                "query_id": str(query_id),
                "query_id_probabilities": dict(query_probabilities),
                "option_count": int(option_count),
                "option_count_probabilities": dict(option_count_probabilities),
                "answer_label": str(scene_payload.answer_label),
                "answer_label_probabilities": dict(answer_label_probabilities),
                "reference_transform_id": str(scene_payload.reference_transform_id),
                "reference_icon_id": str(scene_payload.reference_icon_id),
                "cell_labels": list(scene_payload.cell_labels),
                "cell_icon_ids": list(scene_payload.cell_icon_ids),
                "cell_transform_ids": list(scene_payload.cell_transform_ids),
                "question_format": "select_scene_cell_matching_reference_transform",
            },
            "witness_symbolic": {
                "answer_label": str(scene_payload.answer_label),
                "reference_transform_id": str(scene_payload.reference_transform_id),
            },
            "projected_annotation": dict(annotation_artifacts["projected_annotation"]),
        }
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            query_id=str(query_id),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
        )


__all__ = ["IconsPairGridReferenceTransformMatchLabelTask"]
