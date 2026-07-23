"""Select the option cell whose mirror symmetry matches a Reference cell."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from ...shared.fixed_query import resolve_task_query_id_param, strip_query_id_params
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import (
    build_prompt_query_spec,
)
from ..shared.annotation import bbox_map_annotation

from .shared.defaults import MirrorGridDefaults
from .shared.output import mirror_grid_render_spec
from .shared.prompts import render_mirror_grid_prompt_artifacts, required_mirror_grid_prompt_defaults
from .shared.rendering import sample_and_render_mirror_grid_scene
from .shared.sampling import (
    fixed_grid_labels,
    probability_map,
    requested_answer_label,
    resolve_answer_label,
    resolve_option_count,
    resolve_pool_manifest,
)
from .shared.state import MirrorGridScenePayload
from .shared.styles import resolve_mirror_grid_render_params


TASK_ID = "task_icons__mirror_grid__mirror_symmetry_match_label"
DOMAIN = "icons"
SCENE_ID = "mirror_grid"
QUERY_ID = SINGLE_QUERY_ID
MIRROR_VERTICAL_SIGNATURE = "mirror_vertical"
MIRROR_HORIZONTAL_SIGNATURE = "mirror_horizontal"
MIRROR_DIAGONAL_MAIN_SIGNATURE = "mirror_diagonal_main"
MIRROR_DIAGONAL_ANTI_SIGNATURE = "mirror_diagonal_anti"
MIRROR_BOTH_AXES_SIGNATURE = "mirror_both_axes"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (
    QUERY_ID,
)
MIRROR_SIGNATURES: Tuple[str, ...] = (
    MIRROR_VERTICAL_SIGNATURE,
    MIRROR_HORIZONTAL_SIGNATURE,
    MIRROR_DIAGONAL_MAIN_SIGNATURE,
    MIRROR_DIAGONAL_ANTI_SIGNATURE,
    MIRROR_BOTH_AXES_SIGNATURE,
)
NOISE_NAMESPACE = "mirror_grid_symmetry_match_label"

_SIGNATURE_TO_SYMMETRY_KIND: Dict[str, str] = {
    MIRROR_VERTICAL_SIGNATURE: "vertical",
    MIRROR_HORIZONTAL_SIGNATURE: "horizontal",
    MIRROR_DIAGONAL_MAIN_SIGNATURE: "diagonal_main",
    MIRROR_DIAGONAL_ANTI_SIGNATURE: "diagonal_anti",
    MIRROR_BOTH_AXES_SIGNATURE: "both_axes",
}
_SYMMETRY_KIND_TO_SIGNATURE: Dict[str, str] = {
    str(kind): str(query) for query, kind in _SIGNATURE_TO_SYMMETRY_KIND.items()
}


_DEFAULTS = MirrorGridDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
    task_id=TASK_ID,
)


def _normalize_public_query(
    *,
    params: Mapping[str, Any],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Validate the single public query id and remove query selector aliases."""

    resolve_task_query_id_param(
        params=params,
        supported_query_ids=SUPPORTED_QUERY_IDS,
        default_query_id=QUERY_ID,
        task_id=TASK_ID,
    )
    return {QUERY_ID: 1.0}, strip_query_id_params(params)


def _select_mirror_signature(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
) -> Tuple[str, Dict[str, float], Dict[str, Any]]:
    """Select the internal mirror-signature axis for this single public query."""

    task_params = dict(params)
    requested_signature: str | None = None
    requested_key: str | None = None
    for key in ("mirror_signature", "target_mirror_signature", "reference_symmetry_id"):
        raw_value = task_params.pop(str(key), None)
        if raw_value is None:
            continue
        value = str(raw_value)
        if requested_signature is not None and value != requested_signature:
            raise ValueError(f"{requested_key} conflicts with {key}")
        requested_signature = value
        requested_key = str(key)

    if requested_signature is not None:
        if requested_signature not in MIRROR_SIGNATURES:
            raise ValueError(
                f"unsupported mirror_signature for {TASK_ID}: {requested_signature}; "
                f"supported: {MIRROR_SIGNATURES}"
            )
        return (
            str(requested_signature),
            probability_map(MIRROR_SIGNATURES, selected=str(requested_signature)),
            task_params,
        )

    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.mirror_signature")
    selected = str(rng.choice(MIRROR_SIGNATURES))
    return str(selected), probability_map(MIRROR_SIGNATURES), task_params


def _symmetry_kind_for_signature(mirror_signature: str) -> str:
    """Translate an internal mirror signature into the neutral scene symmetry kind."""

    try:
        return str(_SIGNATURE_TO_SYMMETRY_KIND[str(mirror_signature)])
    except KeyError as exc:
        raise ValueError(f"unsupported mirror_signature for {TASK_ID}: {mirror_signature}") from exc


def _public_symmetry_id(symmetry_kind: str) -> str:
    """Translate a neutral scene symmetry kind into public trace vocabulary."""

    if str(symmetry_kind) == "none":
        return "none"
    try:
        return str(_SYMMETRY_KIND_TO_SIGNATURE[str(symmetry_kind)])
    except KeyError as exc:
        raise ValueError(f"unsupported symmetry kind in {TASK_ID}: {symmetry_kind}") from exc


def _public_cell_payload(cell: Mapping[str, Any]) -> Dict[str, Any]:
    """Return one trace cell payload with public and neutral symmetry fields."""

    payload = dict(cell)
    symmetry_kind = str(payload.get("symmetry_kind", ""))
    payload["symmetry_id"] = _public_symmetry_id(symmetry_kind)
    return payload


def _public_scene_payload(scene_payload: MirrorGridScenePayload) -> Tuple[Dict[str, Any], Tuple[Dict[str, Any], ...]]:
    """Return public trace entities for the reference cell and option cells."""

    reference_cell = _public_cell_payload(scene_payload.reference_cell)
    scene_cells = tuple(_public_cell_payload(cell) for cell in scene_payload.scene_cells)
    return dict(reference_cell), tuple(dict(cell) for cell in scene_cells)


def _scene_cell_public_symmetry_ids(scene_payload: MirrorGridScenePayload) -> Tuple[str, ...]:
    """Return public symmetry ids for every visible option cell."""

    return tuple(_public_symmetry_id(str(kind)) for kind in scene_payload.scene_cell_symmetry_kinds)


def _cell_by_label(scene_cells: Sequence[Mapping[str, Any]], label: str) -> Dict[str, Any]:
    """Return one option cell payload by visible label."""

    for cell in scene_cells:
        if str(cell.get("label")) == str(label):
            return dict(cell)
    raise RuntimeError(f"missing option cell for answer label {label!r}")


@register_task
class IconsMirrorGridMirrorSymmetryMatchLabelTask:
    """Select the labeled option cell matching the Reference cell's mirror symmetry."""

    task_id = TASK_ID
    reasoning_operations = ('transformation', 'matching')
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one deterministic mirror-grid option-match instance."""

        query_probabilities, normalized_params = _normalize_public_query(
            params=params,
        )
        mirror_signature, mirror_signature_probabilities, task_params = _select_mirror_signature(
            instance_seed=int(instance_seed),
            params=normalized_params,
        )
        reference_symmetry_kind = _symmetry_kind_for_signature(str(mirror_signature))
        scene_rng = spawn_rng(int(instance_seed), "scene")
        explicit_answer_label = requested_answer_label(task_params)
        option_count, option_count_probabilities = resolve_option_count(
            scene_rng,
            params=task_params,
            generation_defaults=_GEN_DEFAULTS,
            fallback_choices=_DEFAULTS.option_count_choices,
            explicit_answer_label=str(explicit_answer_label),
            context=TASK_ID,
        )
        option_labels = fixed_grid_labels(int(option_count))
        answer_label, answer_index, answer_label_probabilities = resolve_answer_label(
            scene_rng,
            params=task_params,
            option_labels=option_labels,
            context=TASK_ID,
        )
        distractor_count = int(option_count) - 1
        distractor_count_probabilities = {str(distractor_count): 1.0}
        render_params = resolve_mirror_grid_render_params(
            task_params,
            render_defaults=_RENDER_DEFAULTS,
            defaults=_DEFAULTS,
            instance_seed=int(instance_seed),
        )
        pool_manifest = resolve_pool_manifest(
            task_params,
            generation_defaults=_GEN_DEFAULTS,
            fallback=_DEFAULTS.pool_manifest,
        )

        scene_payload = None
        image = None
        last_error: Exception | None = None
        for _ in range(max(1, int(max_attempts))):
            try:
                scene_payload, image = sample_and_render_mirror_grid_scene(
                    scene_rng,
                    instance_seed=int(instance_seed),
                    reference_symmetry_kind=str(reference_symmetry_kind),
                    object_count=int(option_count),
                    target_count=1,
                    matching_indices=(int(answer_index),),
                    render_params=render_params,
                    pool_manifest=str(pool_manifest),
                    noise_namespace=f"{NOISE_NAMESPACE}:{mirror_signature}:{answer_label}",
                )
                break
            except Exception as exc:  # pragma: no cover - exercised through retry loop
                last_error = exc
                continue
        if scene_payload is None or image is None:
            raise RuntimeError(f"failed to generate {TASK_ID} instance") from last_error

        prompt_defaults = required_mirror_grid_prompt_defaults(
            _PROMPT_DEFAULTS,
            context=f"prompt defaults for {self.task_id}",
        )
        prompt_artifacts = render_mirror_grid_prompt_artifacts(
            domain=DOMAIN,
            scene_id=SCENE_ID,
            instance_seed=int(instance_seed),
            prompt_defaults=prompt_defaults,
        )

        reference_cell, scene_cells = _public_scene_payload(scene_payload)
        matching_labels = tuple(str(label) for label in scene_payload.matching_labels)
        if matching_labels != (str(answer_label),):
            raise RuntimeError(
                f"mirror-grid answer mismatch: expected {answer_label!r}, got {matching_labels!r}"
            )
        matching_cell = _cell_by_label(scene_cells, str(answer_label))
        annotation_artifacts = bbox_map_annotation(
            {
                "reference_cell": reference_cell["cell_bbox_xyxy"],
                "matching_option_cell": matching_cell["cell_bbox_xyxy"],
            }
        )
        answer_gt = TypedValue(type="option_letter", value=str(answer_label))
        annotation_gt = TypedValue(
            type=str(annotation_artifacts["annotation_type"]),
            value=dict(annotation_artifacts["annotation_value"]),
        )
        scene_cell_symmetry_ids = _scene_cell_public_symmetry_ids(scene_payload)
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=QUERY_ID,
            params={
                "task_id": str(self.task_id),
                "scene_id": SCENE_ID,
                "query_id_probabilities": dict(query_probabilities),
                "internal_query_id": str(mirror_signature),
                "option_count": int(option_count),
                "option_count_probabilities": dict(option_count_probabilities),
                "option_labels": list(option_labels),
                "answer_label": str(answer_label),
                "answer_label_probabilities": dict(answer_label_probabilities),
                "distractor_count": int(distractor_count),
                "distractor_count_probabilities": dict(distractor_count_probabilities),
                "pool_manifest": str(pool_manifest),
                "mirror_signature": str(mirror_signature),
                "mirror_signature_probabilities": dict(mirror_signature_probabilities),
                "reference_symmetry_kind": str(scene_payload.reference_symmetry_kind),
            },
        )
        common_ids = {
            "task_id": str(self.task_id),
            "scene_id": SCENE_ID,
            "query_id": QUERY_ID,
        }
        trace_payload = {
            "scene_ir": {
                **common_ids,
                "scene_kind": "icons_reference_grid_mirror_symmetry_match_label",
                "entities": [dict(reference_cell), *[dict(item) for item in scene_cells]],
                "relations": {
                    "target": "option_cell_matching_reference_mirror_symmetry",
                    "reference_symmetry_id": str(mirror_signature),
                    "reference_symmetry_kind": str(scene_payload.reference_symmetry_kind),
                    "mirror_signature": str(mirror_signature),
                    "matching_cell_label": str(answer_label),
                    "answer_label": str(answer_label),
                },
                "frames": {
                    "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
                    "panels": dict(scene_payload.panel_geometry),
                },
            },
            "query_spec": query_spec,
            "render_spec": mirror_grid_render_spec(
                common_ids=common_ids,
                render_params=render_params,
                sampled_palette_rgb=scene_payload.sampled_palette_rgb,
                panel_geometry=scene_payload.panel_geometry,
            ),
            "render_map": {
                "image_id": "img0",
                "anchors": {
                    "reference_cell": dict(reference_cell),
                    "matching_option_cell": dict(matching_cell),
                    "answer_label": str(answer_label),
                    "option_cells": [dict(item) for item in scene_cells],
                },
            },
            "execution_trace": {
                **common_ids,
                "scene_variant": "reference_with_labeled_option_cells",
                "query_id_probabilities": dict(query_probabilities),
                "internal_query_id": str(mirror_signature),
                "mirror_signature": str(mirror_signature),
                "mirror_signature_probabilities": dict(mirror_signature_probabilities),
                "reference_symmetry_kind": str(scene_payload.reference_symmetry_kind),
                "option_count": int(scene_payload.object_count),
                "option_count_probabilities": dict(option_count_probabilities),
                "option_labels": list(scene_payload.cell_labels),
                "answer_label": str(answer_label),
                "matching_cell_label": str(answer_label),
                "distractor_count": int(scene_payload.distractor_count),
                "distractor_count_probabilities": dict(distractor_count_probabilities),
                "scene_cell_symmetry_ids": list(scene_cell_symmetry_ids),
                "scene_cell_symmetry_kinds": list(scene_payload.scene_cell_symmetry_kinds),
                "question_format": "select_option_cell_matching_reference_mirror_symmetry",
                "annotation_roles": ["reference_cell", "matching_option_cell"],
            },
            "witness_symbolic": {
                "answer_label": str(answer_label),
                "reference_symmetry_id": str(mirror_signature),
                "reference_symmetry_kind": str(scene_payload.reference_symmetry_kind),
                "mirror_signature": str(mirror_signature),
                "reference_cell_bbox": list(reference_cell["cell_bbox_xyxy"]),
                "matching_option_cell_bbox": list(matching_cell["cell_bbox_xyxy"]),
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
            scene_id=SCENE_ID,
            query_id=QUERY_ID,
            prompt_variants=dict(prompt_artifacts.prompt_variants),
        )


__all__ = ["IconsMirrorGridMirrorSymmetryMatchLabelTask", "TASK_ID"]
