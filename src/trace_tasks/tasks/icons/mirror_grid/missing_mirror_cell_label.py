"""Select the option icon that completes a mirror-symmetric grid."""

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
from ..shared.annotation import bbox_annotation

from .shared.defaults import MirrorGridDefaults
from .shared.output import mirror_grid_render_spec
from .shared.prompts import render_mirror_grid_prompt_artifacts, required_mirror_grid_prompt_defaults
from .shared.rendering import sample_and_render_missing_mirror_cell_scene
from .shared.sampling import (
    fixed_grid_labels,
    probability_map,
    requested_answer_label,
    resolve_answer_label,
    resolve_option_count,
    resolve_pool_manifest,
)
from .shared.styles import resolve_mirror_grid_render_params


TASK_ID = "task_icons__mirror_grid__missing_mirror_cell_label"
DOMAIN = "icons"
SCENE_ID = "mirror_grid"
QUERY_ID = SINGLE_QUERY_ID
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (QUERY_ID,)
MIRROR_AXES: Tuple[str, ...] = ("vertical", "horizontal")
NOISE_NAMESPACE = "mirror_grid_missing_mirror_cell_label"


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


def _select_mirror_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
) -> Tuple[str, Dict[str, float], Dict[str, Any]]:
    """Select the internal mirror-axis generation axis."""

    task_params = dict(params)
    requested_axis = task_params.pop("mirror_axis", None)
    if requested_axis is not None:
        axis = str(requested_axis).strip().lower()
        if axis not in set(MIRROR_AXES):
            raise ValueError(f"unsupported mirror_axis for {TASK_ID}: {axis}; supported: {MIRROR_AXES}")
        return axis, probability_map(MIRROR_AXES, selected=axis), task_params

    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.mirror_axis")
    axis = str(rng.choice(MIRROR_AXES))
    return axis, probability_map(MIRROR_AXES), task_params


def _option_cell_by_label(option_cells: Sequence[Mapping[str, Any]], label: str) -> Dict[str, Any]:
    """Return one rendered option cell payload by visible label."""

    for cell in option_cells:
        if str(cell.get("label")) == str(label):
            return dict(cell)
    raise RuntimeError(f"missing option cell for answer label {label!r}")


def _completion_relation_payload(scene_payload, *, answer_label: str) -> Dict[str, Any]:
    """Return task-specific relation metadata for the missing-cell completion."""

    return {
        "target": "option_icon_completes_missing_mirror_cell",
        "mirror_axis": str(scene_payload.mirror_axis),
        "missing_cell": {
            "row": int(scene_payload.missing_row),
            "col": int(scene_payload.missing_col),
        },
        "counterpart_cell": {
            "row": int(scene_payload.counterpart_row),
            "col": int(scene_payload.counterpart_col),
        },
        "answer_label": str(answer_label),
    }


@register_task
class IconsMirrorGridMissingMirrorCellLabelTask:
    """Select the option icon that completes a mirror-symmetric grid."""

    task_id = TASK_ID
    reasoning_operations = ('transformation', 'matching')
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one deterministic missing-cell mirror-grid instance."""

        query_probabilities, normalized_params = _normalize_public_query(
            params=params,
        )
        mirror_axis, mirror_axis_probabilities, task_params = _select_mirror_axis(
            instance_seed=int(instance_seed),
            params=normalized_params,
        )
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
                scene_payload, image = sample_and_render_missing_mirror_cell_scene(
                    scene_rng,
                    instance_seed=int(instance_seed),
                    mirror_axis=str(mirror_axis),
                    option_count=int(option_count),
                    answer_index=int(answer_index),
                    render_params=render_params,
                    pool_manifest=str(pool_manifest),
                    noise_namespace=f"{NOISE_NAMESPACE}:{mirror_axis}:{answer_label}",
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

        selected_option = _option_cell_by_label(scene_payload.option_cells, str(answer_label))
        if not bool(selected_option.get("is_answer")):
            raise RuntimeError(f"mirror-grid completion answer mismatch for label {answer_label!r}")
        annotation_artifacts = bbox_annotation(selected_option["cell_bbox_xyxy"])
        answer_gt = TypedValue(type="option_letter", value=str(answer_label))
        annotation_gt = TypedValue(
            type=str(annotation_artifacts["annotation_type"]),
            value=list(annotation_artifacts["annotation_value"]),
        )
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=QUERY_ID,
            params={
                "task_id": str(self.task_id),
                "scene_id": SCENE_ID,
                "query_id_probabilities": dict(query_probabilities),
                "mirror_axis": str(mirror_axis),
                "mirror_axis_probabilities": dict(mirror_axis_probabilities),
                "option_count": int(option_count),
                "option_count_probabilities": dict(option_count_probabilities),
                "option_labels": list(option_labels),
                "answer_label": str(answer_label),
                "answer_label_probabilities": dict(answer_label_probabilities),
                "pool_manifest": str(pool_manifest),
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
                "scene_kind": "icons_missing_mirror_cell_label",
                "entities": [
                    dict(scene_payload.grid_panel),
                    *[dict(item) for item in scene_payload.grid_cells],
                    *[dict(item) for item in scene_payload.option_cells],
                ],
                "relations": _completion_relation_payload(scene_payload, answer_label=str(answer_label)),
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
                    "answer_label": str(answer_label),
                    "selected_option": dict(selected_option),
                    "missing_cell": {
                        "row": int(scene_payload.missing_row),
                        "col": int(scene_payload.missing_col),
                    },
                    "mirror_grid_cells": [dict(item) for item in scene_payload.grid_cells],
                    "option_cells": [dict(item) for item in scene_payload.option_cells],
                },
            },
            "execution_trace": {
                **common_ids,
                "scene_variant": "missing_icon_cell_with_labeled_options",
                "question_format": "select_option_icon_completing_mirror_symmetry",
                "query_id_probabilities": dict(query_probabilities),
                "mirror_axis": str(scene_payload.mirror_axis),
                "mirror_axis_probabilities": dict(mirror_axis_probabilities),
                "option_count": int(scene_payload.option_count),
                "option_count_probabilities": dict(option_count_probabilities),
                "option_labels": list(scene_payload.option_labels),
                "answer_label": str(answer_label),
                "answer_label_probabilities": dict(answer_label_probabilities),
                "missing_row": int(scene_payload.missing_row),
                "missing_col": int(scene_payload.missing_col),
                "counterpart_row": int(scene_payload.counterpart_row),
                "counterpart_col": int(scene_payload.counterpart_col),
                "annotation_roles": ["selected_option"],
            },
            "witness_symbolic": {
                "answer_label": str(answer_label),
                "mirror_axis": str(scene_payload.mirror_axis),
                "selected_option_bbox": list(selected_option["cell_bbox_xyxy"]),
                "missing_cell": {
                    "row": int(scene_payload.missing_row),
                    "col": int(scene_payload.missing_col),
                },
                "counterpart_cell": {
                    "row": int(scene_payload.counterpart_row),
                    "col": int(scene_payload.counterpart_col),
                },
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


__all__ = ["IconsMirrorGridMissingMirrorCellLabelTask", "TASK_ID"]
