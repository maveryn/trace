"""Scene-private lifecycle helpers for cube-net puzzle rendering."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from PIL import Image

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.puzzles.shared.visual_defaults import load_puzzle_noise_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.font_assets import font_asset_version, sample_font_family
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.shared.text_rendering import temporary_default_font_family
from trace_tasks.tasks.shared.variant_sampling import resolve_variant

from .shared.annotations import (
    bbox_typed_value,
    projected_bbox,
    round_annotation_bbox,
)
from .shared.output import (
    build_cube_net_trace_payload,
    json_ready,
)
from .shared.prompts import build_cube_net_prompt_artifacts
from .shared.rendering import (
    render_equivalent_net_scene,
    render_face_relation_scene,
)
from .shared.sampling import (
    equivalent_net_option_specs,
    face_option_specs,
    sample_equivalent_net_dataset,
    sample_face_relation_dataset,
)
from .shared.state import DOMAIN, NET_COORDS, SCENE_ID, SCENE_VARIANTS


def select_cube_net_scene_variant(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    namespace: str,
) -> tuple[str, Dict[str, float]]:
    """Select nonsemantic cube-net panel chrome from scene variant support."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.scene_variant")
    return resolve_variant(
        rng,
        params=params,
        gen_defaults=generation_defaults,
        supported_variants=SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
    )


def sample_cube_net_font(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    namespace: str,
) -> str:
    """Sample one global font family for all cube-net labels in the image."""

    return sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.label_font",
        params={**dict(rendering_defaults), **dict(params)},
    )


def font_trace_record(font_family: str) -> Dict[str, Any]:
    """Build trace metadata for the sampled cube-net label font."""

    return {
        "source": "global_font_pool",
        "font_family": str(font_family),
        "font_asset_version": font_asset_version(),
        "scope": "cube_net_panel_face_option_labels",
    }


def apply_cube_net_post_noise(
    image: Image.Image,
    *,
    instance_seed: int,
    params: Mapping[str, Any],
) -> tuple[Image.Image, Dict[str, Any]]:
    """Apply scene-configured post-image noise after semantic rendering is done."""

    visual_defaults = {
        **load_puzzle_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5),
        "apply_prob": 0.5,
        "edit_types": ["blur", "downsample", "jpeg", "noise"],
        "edit_count_range": [1, 1],
        "value_ranges": {
            "blur": {"radius": [0.08, 0.24]},
            "downsample": {"scale": [0.94, 0.98]},
            "jpeg": {"quality": [86.0, 95.0]},
            "noise": {"alpha": [0.006, 0.022]},
        },
    }
    return apply_post_image_noise(
        image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=visual_defaults,
    )


def _select_public_query(
    *,
    task_identity: str,
    supported_queries: Sequence[str],
    instance_seed: int,
    params: Mapping[str, Any],
) -> tuple[str, Dict[str, float], Dict[str, Any]]:
    """Delegate query selection while keeping supported ids task-owned."""

    supported = tuple(str(value) for value in supported_queries)
    return select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=supported,
        default_query_id=str(supported[0]),
        task_id=str(task_identity),
        namespace=f"{task_identity}.query",
    )


def run_face_relation_lifecycle(
    *,
    task_identity: str,
    relation_kind: str,
    prompt_task_key: str,
    prompt_query_key: str,
    marked_cue_key: str,
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Run one fixed face-relation objective after public files bind semantics."""

    query_id, query_probabilities, task_params = _select_public_query(
        task_identity=str(task_identity),
        supported_queries=(SINGLE_QUERY_ID,),
        instance_seed=int(instance_seed),
        params=dict(params),
    )
    scene_variant, scene_probabilities = select_cube_net_scene_variant(
        instance_seed=int(instance_seed),
        params=task_params,
        generation_defaults=generation_defaults,
        namespace=f"cube_net.{relation_kind}",
    )
    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        attempt_seed = int(instance_seed) + int(attempt_index)
        try:
            dataset = sample_face_relation_dataset(
                relation_kind=str(relation_kind),
                params=task_params,
                generation_defaults=generation_defaults,
                instance_seed=attempt_seed,
                namespace=f"cube_net.{relation_kind}",
            )
            font_family = sample_cube_net_font(
                instance_seed=attempt_seed,
                params=task_params,
                rendering_defaults=rendering_defaults,
                namespace=f"cube_net.{relation_kind}",
            )
            with temporary_default_font_family(str(font_family)):
                image, render_meta = render_face_relation_scene(
                    dataset=dataset,
                    params=task_params,
                    rendering_defaults=rendering_defaults,
                    instance_seed=attempt_seed,
                    scene_variant=str(scene_variant),
                )
            break
        except ValueError as exc:
            last_error = exc
    else:
        raise RuntimeError(f"{task_identity} failed to construct a sample") from last_error

    image, post_noise_meta = apply_cube_net_post_noise(
        image,
        instance_seed=int(instance_seed),
        params=task_params,
    )
    prompt_defaults, prompt_artifacts = build_cube_net_prompt_artifacts(
        prompt_task_key=str(prompt_task_key),
        prompt_query_key=str(prompt_query_key),
        dynamic_slots={},
        instance_seed=int(instance_seed),
    )
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(query_id),
        params={
            "scene_id": SCENE_ID,
            "query_id_probabilities": dict(query_probabilities),
            "prompt_query_key": str(prompt_query_key),
            "scene_variant": str(scene_variant),
            "scene_variant_probabilities": dict(scene_probabilities),
            "option_labels": [str(option.option_label) for option in dataset.options],
            "answer_support": [str(option.option_label) for option in dataset.options],
            "relation_kind": str(dataset.relation_kind),
        },
    )
    relation_bboxes = render_meta.get("relation_bboxes_px", {})
    if str(marked_cue_key) not in relation_bboxes:
        raise ValueError(f"missing relation cue bbox: {marked_cue_key}")
    option_bbox = render_meta["option_panel_bboxes_px"][
        f"option_{dataset.correct_option_label}"
    ]
    annotation_bbox = round_annotation_bbox(option_bbox)
    answer_gt = TypedValue(type="option_letter", value=str(dataset.correct_option_label))
    annotation_gt = bbox_typed_value(annotation_bbox)
    render_spec = {
        "canvas_width": int(image.width),
        "canvas_height": int(image.height),
        "coord_space": "pixel",
        "scene_id": SCENE_ID,
        "query_id": str(query_id),
        "scene_variant": str(scene_variant),
        "post_image_noise": dict(post_noise_meta),
        "label_style": {"font": font_trace_record(str(font_family))},
        **dict(render_meta),
    }
    trace_payload = build_cube_net_trace_payload(
        scene_ir={
            "scene_kind": "puzzle_cube_net",
            "scene_id": SCENE_ID,
            "task_id": str(task_identity),
            "entities": [
                {
                    "entity_id": f"face_{face}",
                    "kind": "cube_net_face",
                    "face_id": str(face),
                    "face_label": str(label),
                }
                for face, label in sorted(dataset.face_labels.items())
            ],
            "relations": {
                "query_id": str(query_id),
                "internal_query_id": str(prompt_query_key),
                "scene_variant": str(scene_variant),
                "relation_kind": str(dataset.relation_kind),
                "reference_face": str(dataset.reference_face),
                "marked_side": dataset.marked_side,
                "marked_cue_key": str(marked_cue_key),
                "correct_face": str(dataset.correct_face),
                "correct_option_label": str(dataset.correct_option_label),
            },
        },
        query_spec=query_spec,
        render_spec=render_spec,
        render_map={
            "image_id": "img0",
            "face_bboxes_px": dict(render_meta["face_bboxes_px"]),
            "relation_bboxes_px": dict(render_meta["relation_bboxes_px"]),
            "option_panel_bboxes_px": dict(render_meta["option_panel_bboxes_px"]),
            "annotation_source": "option_panel_bboxes_px",
        },
        execution_trace={
            "scene_id": SCENE_ID,
            "query_id": str(query_id),
            "internal_query_id": str(prompt_query_key),
            "scene_variant": str(scene_variant),
            "relation_kind": str(dataset.relation_kind),
            "face_labels": dict(dataset.face_labels),
            "net_coords": {
                str(face): [int(coord[0]), int(coord[1])]
                for face, coord in NET_COORDS.items()
            },
            "reference_face": str(dataset.reference_face),
            "marked_side": dataset.marked_side,
            "marked_cue_key": str(marked_cue_key),
            "net_rotation_degrees": int(render_meta.get("net_rotation_degrees", 0)),
            "correct_face": str(dataset.correct_face),
            "option_specs": face_option_specs(dataset.options),
            "answer_value": str(dataset.correct_option_label),
        },
        witness_symbolic={
            "type": "cube_face_relation",
            "value": {
                "reference_face": str(dataset.reference_face),
                "marked_side": dataset.marked_side,
                "marked_cue_key": str(marked_cue_key),
                "correct_face": str(dataset.correct_face),
                "correct_option_label": str(dataset.correct_option_label),
            },
        },
        projected_annotation=projected_bbox(annotation_bbox),
        answer_gt=answer_gt.to_dict(),
        annotation_gt=annotation_gt.to_dict(),
        prompt_defaults=prompt_defaults,
        prompt_artifacts=prompt_artifacts,
    )
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=answer_gt,
        annotation_gt=annotation_gt,
        image=image,
        image_id="img0",
        trace_payload=json_ready(trace_payload),
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(query_id),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
    )


def run_equivalent_net_lifecycle(
    *,
    task_identity: str,
    prompt_task_key: str,
    prompt_query_key: str,
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Run one colored-net equivalence objective after task-owned binding."""

    query_id, query_probabilities, task_params = _select_public_query(
        task_identity=str(task_identity),
        supported_queries=(SINGLE_QUERY_ID,),
        instance_seed=int(instance_seed),
        params=dict(params),
    )
    scene_variant, scene_probabilities = select_cube_net_scene_variant(
        instance_seed=int(instance_seed),
        params=task_params,
        generation_defaults=generation_defaults,
        namespace="cube_net.equivalent_net",
    )
    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        attempt_seed = int(instance_seed) + int(attempt_index)
        try:
            dataset = sample_equivalent_net_dataset(
                params=task_params,
                generation_defaults=generation_defaults,
                instance_seed=attempt_seed,
                namespace="cube_net.equivalent_net",
            )
            font_family = sample_cube_net_font(
                instance_seed=attempt_seed,
                params=task_params,
                rendering_defaults=rendering_defaults,
                namespace="cube_net.equivalent_net",
            )
            with temporary_default_font_family(str(font_family)):
                image, render_meta = render_equivalent_net_scene(
                    dataset=dataset,
                    params=task_params,
                    rendering_defaults=rendering_defaults,
                    instance_seed=attempt_seed,
                    scene_variant=str(scene_variant),
                )
            break
        except ValueError as exc:
            last_error = exc
    else:
        raise RuntimeError(f"{task_identity} failed to construct a sample") from last_error

    image, post_noise_meta = apply_cube_net_post_noise(
        image,
        instance_seed=int(instance_seed),
        params=task_params,
    )
    prompt_defaults, prompt_artifacts = build_cube_net_prompt_artifacts(
        prompt_task_key=str(prompt_task_key),
        prompt_query_key=str(prompt_query_key),
        dynamic_slots={},
        instance_seed=int(instance_seed),
    )
    option_specs = equivalent_net_option_specs(dataset.options)
    answer_value = str(dataset.correct_option_label)
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(query_id),
        params={
            "scene_id": SCENE_ID,
            "query_id_probabilities": dict(query_probabilities),
            "prompt_query_key": str(prompt_query_key),
            "scene_variant": str(scene_variant),
            "scene_variant_probabilities": dict(scene_probabilities),
            "option_labels": [str(option["option_label"]) for option in option_specs],
            "answer_support": [str(option["option_label"]) for option in option_specs],
            "equivalence_rule": "same folded colored cube up to whole-cube rotation",
        },
    )
    option_bbox = render_meta["option_panel_bboxes_px"][f"option_{answer_value}"]
    annotation_bbox = round_annotation_bbox(option_bbox)
    answer_gt = TypedValue(type="option_letter", value=answer_value)
    annotation_gt = bbox_typed_value(annotation_bbox)
    render_spec = {
        "canvas_width": int(image.width),
        "canvas_height": int(image.height),
        "coord_space": "pixel",
        "scene_id": SCENE_ID,
        "query_id": str(query_id),
        "scene_variant": str(scene_variant),
        "post_image_noise": dict(post_noise_meta),
        "label_style": {"font": font_trace_record(str(font_family))},
        **dict(render_meta),
    }
    trace_payload = build_cube_net_trace_payload(
        scene_ir={
            "scene_kind": "puzzle_cube_net",
            "scene_id": SCENE_ID,
            "task_id": str(task_identity),
            "entities": [
                {
                    "entity_id": f"reference_face_{face}",
                    "kind": "colored_cube_net_face",
                    "face_id": str(face),
                    "color_name": str(color_name),
                }
                for face, color_name in sorted(dataset.reference_face_color_names.items())
            ],
            "relations": {
                "query_id": str(query_id),
                "internal_query_id": str(prompt_query_key),
                "scene_variant": str(scene_variant),
                "reference_signature": list(dataset.reference_signature),
                "correct_option_label": answer_value,
            },
        },
        query_spec=query_spec,
        render_spec=render_spec,
        render_map={
            "image_id": "img0",
            "reference_panel_bbox_px": list(render_meta["reference_panel_bbox_px"]),
            "reference_face_bboxes_px": dict(render_meta["reference_face_bboxes_px"]),
            "option_panel_bboxes_px": dict(render_meta["option_panel_bboxes_px"]),
            "option_face_bboxes_px": dict(render_meta["option_face_bboxes_px"]),
            "annotation_source": "option_panel_bboxes_px",
        },
        execution_trace={
            "scene_id": SCENE_ID,
            "query_id": str(query_id),
            "internal_query_id": str(prompt_query_key),
            "scene_variant": str(scene_variant),
            "reference_face_color_names": dict(dataset.reference_face_color_names),
            "reference_signature": list(dataset.reference_signature),
            "option_specs": list(option_specs),
            "answer_value": answer_value,
        },
        witness_symbolic={
            "type": "cube_net_equivalence",
            "value": {
                "reference_signature": list(dataset.reference_signature),
                "correct_option_label": answer_value,
            },
        },
        projected_annotation=projected_bbox(annotation_bbox),
        answer_gt=answer_gt.to_dict(),
        annotation_gt=annotation_gt.to_dict(),
        prompt_defaults=prompt_defaults,
        prompt_artifacts=prompt_artifacts,
    )
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=answer_gt,
        annotation_gt=annotation_gt,
        image=image,
        image_id="img0",
        trace_payload=json_ready(trace_payload),
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(query_id),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
    )


__all__ = [
    "apply_cube_net_post_noise",
    "font_trace_record",
    "run_equivalent_net_lifecycle",
    "run_face_relation_lifecycle",
    "sample_cube_net_font",
    "select_cube_net_scene_variant",
]
