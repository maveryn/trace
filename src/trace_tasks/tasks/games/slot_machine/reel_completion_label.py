"""Third-reel completion option task for slot-machine games."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID, select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import option_bbox_annotation
from .shared.defaults import SCENE_ID, SCENE_NAMESPACE
from .shared.output import build_slot_machine_common_trace_params
from .shared.prompts import build_slot_machine_prompt_artifacts, slot_label_bbox_json_examples, slot_output_slots
from .shared.rendering import RenderedSlotMachineScene, render_slot_completion_scene, resolve_slot_machine_render_params
from .shared.sampling import resolve_slot_machine_axes, sample_slot_reel_completion_scene
from .shared.state import SlotCompletionScene, completion_option_grid


TASK_ID = "task_games__slot_machine__reel_completion_label"
PROMPT_QUERY_KEY = "reel_completion_label"
SUPPORTED_QUERY_IDS = (DEFAULT_QUERY_ID,)
OPTION_LABELS = ("A", "B", "C", "D")

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _prompt_dynamic_slots() -> dict[str, Any]:
    """Return task-owned prompt slots for the option-label completion task."""

    json_example, json_example_answer_only = slot_label_bbox_json_examples(answer_label="B")
    return slot_output_slots(
        prompt_query_key=PROMPT_QUERY_KEY,
        json_example=json_example,
        json_example_answer_only=json_example_answer_only,
        extra_slots={
            "object_description": "a toy slot machine with two visible reels and four labeled options for the third reel",
        },
    )


def _completion_trace_payload(
    *,
    axes: Any,
    scene: SlotCompletionScene,
    rendered_scene: RenderedSlotMachineScene,
    annotation_artifacts: AnnotationArtifacts,
    query_spec: Mapping[str, Any],
    prompt_bundle_id: str,
) -> dict[str, Any]:
    """Assemble trace payload for a labeled third-reel completion instance."""

    option_records = []
    for option in scene.options:
        grid = completion_option_grid(scene.base_cells, option.cells)
        option_records.append(
            {
                "label": str(option.label),
                "completed_payline_ids": [str(payline_id) for payline_id in option.completed_payline_ids],
                "completed_grid": [list(row) for row in grid],
            }
        )
    base_rows = [
        [
            str(next(cell.symbol_key for cell in scene.base_cells if int(cell.row) == row and int(cell.col) == col))
            for col in range(2)
        ]
        for row in range(3)
    ]
    return {
        "scene_ir": {
            "scene_kind": f"games_slot_machine_{str(axes.scene_variant)}_reel_completion",
            "entities": [dict(entity) for entity in rendered_scene.scene_entities],
            "relations": {
                "scene_variant": str(axes.scene_variant),
                "style_variant": str(axes.style_variant),
                "payline_scope": "rows_plus_long_diagonals",
                "option_labels": [str(label) for label in OPTION_LABELS],
                "answer_label": str(scene.answer_label),
                "answer_completed_payline_ids": [str(payline_id) for payline_id in scene.answer_completed_payline_ids],
            },
        },
        "query_spec": dict(query_spec),
        "render_spec": {
            "scene_variant": str(axes.scene_variant),
            "style_variant": str(axes.style_variant),
            "canvas_width": int(rendered_scene.image.size[0]),
            "canvas_height": int(rendered_scene.image.size[1]),
            "panel_scene_style": dict(rendered_scene.panel_style_meta),
        },
        "render_map": dict(rendered_scene.render_map),
        "execution_trace": {
            "prompt_query_key": PROMPT_QUERY_KEY,
            "prompt_bundle_id": str(prompt_bundle_id),
            "scene_variant": str(axes.scene_variant),
            "style_variant": str(axes.style_variant),
            "base_reel_rows": base_rows,
            "options": option_records,
            "answer": str(scene.answer_label),
            "answer_completed_payline_ids": [str(payline_id) for payline_id in scene.answer_completed_payline_ids],
            "annotation_option_label": str(scene.answer_label),
        },
        "witness_symbolic": {
            "type": "slot_completion_option_bbox",
            "option_label": str(scene.answer_label),
            "completed_payline_ids": [str(payline_id) for payline_id in scene.answer_completed_payline_ids],
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "background": dict(rendered_scene.background_meta),
        "post_image_noise": dict(rendered_scene.post_noise_meta),
    }


@register_task
class GamesSlotMachineReelCompletionLabelTask:
    """Choose the third-reel option that creates one row or diagonal payline."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'formula_evaluation', 'matching')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int) -> TaskOutput:
        """Build one option-label instance while keeping answer and bbox from the same sampled scene."""

        selected_query, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=dict(params or {}),
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=DEFAULT_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{SCENE_NAMESPACE}.{TASK_ID}.query",
        )
        axes = resolve_slot_machine_axes(
            instance_seed=int(instance_seed),
            params=task_params,
            gen_defaults=_GEN_DEFAULTS,
        )
        render_params = resolve_slot_machine_render_params(task_params, _RENDER_DEFAULTS)
        prompt_defaults, prompt_artifacts = build_slot_machine_prompt_artifacts(
            domain=self.domain,
            prompt_query_key=PROMPT_QUERY_KEY,
            dynamic_slots=_prompt_dynamic_slots(),
            instance_seed=int(instance_seed),
        )
        query_params = build_slot_machine_common_trace_params(
            axes=axes,
            extra_params={
                "option_labels": [str(label) for label in OPTION_LABELS],
                "payline_scope": "rows_plus_long_diagonals",
                "query_id_probabilities": dict(query_probabilities),
            },
        )
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(selected_query),
            params=query_params,
        )
        last_error: Exception | None = None
        for attempt in range(max(1, int(max_attempts))):
            rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.reel_completion_label.attempt.{attempt}")
            try:
                scene = sample_slot_reel_completion_scene(
                    rng=rng,
                    axes=axes,
                    option_labels=OPTION_LABELS,
                )
                rendered_scene = render_slot_completion_scene(
                    scene=scene,
                    render_params=render_params,
                    instance_seed=int(instance_seed),
                )
                annotation_artifacts = option_bbox_annotation(rendered_scene, scene.answer_label)
                answer_gt = TypedValue(type="option_letter", value=str(scene.answer_label))
                return TaskOutput(
                    prompt=str(prompt_artifacts.prompt),
                    prompt_variants=dict(prompt_artifacts.prompt_variants),
                    answer_gt=answer_gt,
                    annotation_gt=annotation_artifacts.annotation_gt,
                    image=rendered_scene.image,
                    image_id=f"{TASK_ID}_{int(instance_seed)}",
                    trace_payload=_completion_trace_payload(
                        axes=axes,
                        scene=scene,
                        rendered_scene=rendered_scene,
                        annotation_artifacts=annotation_artifacts,
                        query_spec=query_spec,
                        prompt_bundle_id=str(prompt_defaults["bundle_id"]),
                    ),
                    task_versions=default_task_versions(),
                    scene_id=SCENE_ID,
                    query_id=str(selected_query),
                )
            except ValueError as exc:
                last_error = exc
                continue
        raise RuntimeError(f"failed to generate {TASK_ID} after {max_attempts} attempts") from last_error


__all__ = ["GamesSlotMachineReelCompletionLabelTask", "TASK_ID"]
