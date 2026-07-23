"""Scene-private rendering and output lifecycle for public Tents tasks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.puzzles.shared.scene_style import (
    make_puzzle_scene_background,
    resolve_puzzle_scene_style,
)
from trace_tasks.tasks.puzzles.shared.unit_size_jitter import with_puzzle_unit_size_jitter
from trace_tasks.tasks.puzzles.shared.visual_defaults import load_puzzle_noise_defaults
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.defaults import (
    resolve_palette_variant,
    resolve_render_params,
    resolve_scene_variant,
)
from .shared.output import build_trace_payload, json_ready
from .shared.prompts import (
    object_description_for_scene_variant,
    render_tents_prompt_artifacts,
)
from .shared.rendering import render_tents_scene
from .shared.state import DOMAIN, SCENE_ID, TentsSample

_NOISE_DEFAULTS = load_puzzle_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)
_AttemptBuilder = Callable[[int, Mapping[str, Any], int], TaskOutput]
_SampleBuilder = Callable[[Mapping[str, Any], int, Mapping[str, Any], Any], TentsSample]
_OutputBinder = Callable[
    [TentsSample, Mapping[str, Any], str, Mapping[str, float]], "TentsObjectiveBinding"
]


@dataclass(frozen=True)
class TentsObjectiveBinding:
    """Task-owned answer, annotation, and trace fields for a sampled Tents case."""

    answer_gt: TypedValue
    annotation_gt: TypedValue
    projected_annotation: Mapping[str, Any]
    witness_symbolic: Mapping[str, Any]
    semantic_params: Mapping[str, Any]
    execution_fields: Mapping[str, Any]


class TentsSceneTask:
    """Neutral public-task lifecycle; subclasses provide objective callbacks."""

    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = (SINGLE_QUERY_ID,)
    prompt_task_key: str
    prompt_query_key: str
    namespace: str
    sample_builder: _SampleBuilder
    output_binder: _OutputBinder

    def generate(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any],
        max_attempts: int,
    ) -> TaskOutput:
        """Generate one Tents instance through the task-owned callbacks."""

        return retry_tents_generation(
            build_case=self._build_case,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )

    def _build_case(
        self,
        instance_seed: int,
        params: Mapping[str, Any],
        max_attempts: int,
    ) -> TaskOutput:
        """Run common scene plumbing after the subclass selects callbacks."""

        _ = int(max_attempts)
        gen_defaults, render_defaults, prompt_defaults = load_tents_task_defaults(
            str(self.task_id)
        )
        return run_tents_fixed_query_case(
            task_id=str(self.task_id),
            supported_query_ids=tuple(str(item) for item in self.supported_query_ids),
            namespace=str(self.namespace),
            prompt_task_key=str(self.prompt_task_key),
            prompt_query_key=str(self.prompt_query_key),
            instance_seed=int(instance_seed),
            params=params,
            generation_defaults=gen_defaults,
            rendering_defaults=render_defaults,
            prompt_defaults=prompt_defaults,
            sample_builder=self.sample_builder,
            output_binder=self.output_binder,
        )


def load_tents_task_defaults(
    task_id: str,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load generation, rendering, and prompt defaults for one Tents task."""

    return load_scene_generation_rendering_prompt_defaults(
        DOMAIN,
        SCENE_ID,
        task_id=str(task_id),
    )


def retry_tents_generation(
    *,
    build_case: _AttemptBuilder,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Retry task-local construction when semantic constraints reject a seed."""

    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        try:
            return build_case(
                int(instance_seed) + int(attempt_index),
                params,
                int(max_attempts),
            )
        except (RuntimeError, ValueError) as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("Tents generation failed without a captured error")


def prepare_tents_visual_case(
    *,
    sample: TentsSample,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    prompt_task_key: str,
    prompt_query_key: str,
    namespace: str,
    instance_seed: int,
) -> Dict[str, Any]:
    """Resolve visual axes, render the board, and build prompt artifacts."""

    axis_rng = spawn_rng(int(instance_seed), f"{namespace}.scene_axes")
    scene_variant, scene_variant_probabilities = resolve_scene_variant(
        params=params,
        generation_defaults=generation_defaults,
        rng=axis_rng,
    )
    palette_variant, palette_variant_probabilities = resolve_palette_variant(
        params=params,
        generation_defaults=generation_defaults,
        rng=axis_rng,
    )
    render_params = resolve_render_params(
        params,
        rendering_defaults,
        instance_seed=int(instance_seed),
        grid_rows=int(sample.rows),
        grid_cols=int(sample.cols),
    )
    scene_style, scene_style_meta = resolve_puzzle_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.background",
    )
    render_params = replace(
        render_params,
        text_color_rgb=tuple(int(value) for value in scene_style.text_rgb),
        text_stroke_rgb=tuple(int(value) for value in scene_style.text_stroke_rgb),
        style_overrides={
            "panel_fill": tuple(int(value) for value in scene_style.panel_fill_rgb),
            "grid_fill": tuple(int(value) for value in scene_style.option_fill_rgb),
            "cell_a": tuple(int(value) for value in scene_style.option_fill_rgb),
            "cell_b": tuple(int(value) for value in scene_style.step_fill_rgb),
            "grid_line": tuple(int(value) for value in scene_style.grid_rgb),
            "heavy_line": tuple(int(value) for value in scene_style.panel_border_rgb),
            "clue_fill": tuple(int(value) for value in scene_style.panel_fill_rgb),
            "candidate_fill": tuple(int(value) for value in scene_style.step_fill_rgb),
            "candidate_outline": tuple(int(value) for value in scene_style.mark_rgb),
            "candidate_label_fill": tuple(
                int(value) for value in scene_style.option_fill_rgb
            ),
            "tree_fill": tuple(int(value) for value in scene_style.agent_rgb),
            "tree_outline": tuple(int(value) for value in scene_style.panel_border_rgb),
            "tent_fill": tuple(int(value) for value in scene_style.mark_rgb),
            "tent_shadow": tuple(int(value) for value in scene_style.panel_border_rgb),
            "tent_flap_fill": tuple(
                int(value) for value in scene_style.agent_inner_rgb
            ),
            "accent": tuple(int(value) for value in scene_style.mark_rgb),
        },
    )
    background, background_meta = make_puzzle_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=scene_style,
    )
    rendered_scene = render_tents_scene(
        background,
        scene_variant=str(scene_variant),
        palette_variant=str(palette_variant),
        rows=int(sample.rows),
        cols=int(sample.cols),
        row_clues=list(sample.row_clues),
        col_clues=list(sample.col_clues),
        marked_tree=(
            tuple(sample.marked_tree) if sample.marked_tree is not None else None
        ),
        tree_cells=list(sample.tree_cells),
        visible_tents=list(sample.visible_tents),
        candidate_specs=list(sample.candidate_specs),
        labeled_tent_specs=list(sample.labeled_tent_specs),
        render_params=render_params,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=_NOISE_DEFAULTS,
    )
    object_description = object_description_for_scene_variant(
        prompt_defaults,
        str(scene_variant),
    )
    prompt_artifacts = render_tents_prompt_artifacts(
        prompt_defaults=prompt_defaults,
        prompt_task_key=str(prompt_task_key),
        prompt_query_key=str(prompt_query_key),
        object_description=str(object_description),
        instance_seed=int(instance_seed),
    )
    return {
        "image": image,
        "rendered_scene": rendered_scene,
        "render_params": render_params,
        "scene_variant": str(scene_variant),
        "scene_variant_probabilities": dict(scene_variant_probabilities),
        "palette_variant": str(palette_variant),
        "palette_variant_probabilities": dict(palette_variant_probabilities),
        "background_meta": dict(background_meta),
        "scene_style_meta": dict(scene_style_meta),
        "post_noise_meta": dict(post_noise_meta),
        "prompt_defaults": dict(prompt_defaults),
        "prompt_artifacts": prompt_artifacts,
    }


def run_tents_fixed_query_case(
    *,
    task_id: str,
    supported_query_ids: tuple[str, ...],
    namespace: str,
    prompt_task_key: str,
    prompt_query_key: str,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    sample_builder: _SampleBuilder,
    output_binder: _OutputBinder,
) -> TaskOutput:
    """Run common single-branch Tents plumbing around task-owned callbacks."""

    selected_query, branch_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=supported_query_ids,
        task_id=str(task_id),
        namespace=f"{namespace}.branch",
    )
    rng = spawn_rng(int(instance_seed), f"{namespace}.sample")
    sample = sample_builder(
        task_params,
        int(instance_seed),
        generation_defaults,
        rng,
    )
    visual = prepare_tents_visual_case(
        sample=sample,
        params=task_params,
        generation_defaults=generation_defaults,
        rendering_defaults=rendering_defaults,
        prompt_defaults=prompt_defaults,
        prompt_task_key=str(prompt_task_key),
        prompt_query_key=str(prompt_query_key),
        namespace=str(namespace),
        instance_seed=int(instance_seed),
    )
    binding = output_binder(
        sample,
        visual,
        str(selected_query),
        dict(branch_probabilities),
    )
    return build_tents_task_output(
        sample=sample,
        visual=visual,
        public_query_id=str(selected_query),
        branch_probabilities=dict(branch_probabilities),
        answer_gt=binding.answer_gt,
        annotation_gt=binding.annotation_gt,
        projected_annotation=binding.projected_annotation,
        witness_symbolic=binding.witness_symbolic,
        prompt_query_key=str(prompt_query_key),
        semantic_params=binding.semantic_params,
        execution_fields=binding.execution_fields,
    )


def build_tents_task_output(
    *,
    sample: TentsSample,
    visual: Mapping[str, Any],
    public_query_id: str,
    branch_probabilities: Mapping[str, float],
    answer_gt: TypedValue,
    annotation_gt: TypedValue,
    projected_annotation: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    prompt_query_key: str,
    semantic_params: Mapping[str, Any],
    execution_fields: Mapping[str, Any],
) -> TaskOutput:
    """Assemble TaskOutput after the public task binds answer and annotation."""

    rendered_scene = visual["rendered_scene"]
    render_params = visual["render_params"]
    prompt_artifacts = visual["prompt_artifacts"]
    query_params = {
        "query_id": str(public_query_id),
        "query_id_probabilities": dict(branch_probabilities),
        "prompt_query_key": str(prompt_query_key),
        "scene_id": SCENE_ID,
        "scene_variant": str(visual["scene_variant"]),
        "scene_variant_probabilities": dict(visual["scene_variant_probabilities"]),
        "palette_variant": str(visual["palette_variant"]),
        "palette_variant_probabilities": dict(visual["palette_variant_probabilities"]),
        "grid_rows": int(sample.rows),
        "grid_cols": int(sample.cols),
        "grid_rows_range": list(sample.grid_rows_range),
        "grid_cols_range": list(sample.grid_cols_range),
        "option_count": int(sample.option_count),
        "target_answer_support": list(sample.target_answer_support),
        **dict(semantic_params),
    }
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(public_query_id),
        params=query_params,
    )
    trace_payload = build_trace_payload(
        scene_ir={
            "scene_kind": "puzzle_tents_grid",
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": {
                "selected_branch": str(public_query_id),
                "scene_id": SCENE_ID,
                "scene_variant": str(visual["scene_variant"]),
                "palette_variant": str(visual["palette_variant"]),
                "construction_mode": str(sample.construction_mode),
                "answer_value": answer_gt.value,
            },
        },
        query_spec=query_spec,
        render_spec={
            "scene_id": SCENE_ID,
            "canvas_width": int(render_params.canvas_width),
            "canvas_height": int(render_params.canvas_height),
            "coord_space": "pixel",
            "scene_variant": str(visual["scene_variant"]),
            "palette_variant": str(visual["palette_variant"]),
            "scene_style": dict(visual["scene_style_meta"]),
            "background_style": dict(visual["background_meta"]),
            "post_image_noise": dict(visual["post_noise_meta"]),
            "scene_bbox_px": list(rendered_scene.scene_bbox_px),
            "text_style": {
                "clue_font_size_px": int(render_params.clue_font_size_px),
                "candidate_font_size_px": int(render_params.candidate_font_size_px),
            },
            "unit_size_jitter": dict(render_params.unit_size_jitter),
        },
        render_map=with_puzzle_unit_size_jitter(
            {
                "image_id": "img0",
                "scene_bbox_px": list(rendered_scene.scene_bbox_px),
                "cell_bboxes_px": {
                    str(key): list(value)
                    for key, value in rendered_scene.cell_bbox_map.items()
                },
                "clue_bboxes_px": {
                    str(key): list(value)
                    for key, value in rendered_scene.clue_bbox_map.items()
                },
                "option_panel_bboxes_px": {
                    str(key): list(value)
                    for key, value in rendered_scene.option_panel_bbox_map.items()
                },
                "item_bboxes_px": {
                    str(key): list(value)
                    for key, value in rendered_scene.item_bbox_map.items()
                },
                "annotation_source": "item_bboxes_px",
            },
            render_params.unit_size_jitter,
        ),
        execution_trace={
            **dict(query_params),
            "marked_tree": (
                [int(value) for value in sample.marked_tree]
                if sample.marked_tree is not None
                else None
            ),
            "row_clues": [int(value) for value in sample.row_clues],
            "col_clues": [int(value) for value in sample.col_clues],
            "visible_tents": [
                [int(row), int(col)] for row, col in sample.visible_tents
            ],
            "tree_cells": [[int(row), int(col)] for row, col in sample.tree_cells],
            "candidate_specs": [
                json_ready(candidate) for candidate in sample.candidate_specs
            ],
            "labeled_tent_specs": [
                json_ready(tent) for tent in sample.labeled_tent_specs
            ],
            "legal_candidate_cells": [
                [int(row), int(col)] for row, col in sample.legal_candidate_cells
            ],
            "answer_value": answer_gt.value,
            "question_format": str(prompt_query_key),
            **dict(execution_fields),
        },
        witness_symbolic=witness_symbolic,
        projected_annotation=projected_annotation,
        answer_gt=answer_gt.to_dict(),
        annotation_gt=annotation_gt.to_dict(),
        prompt_defaults=visual["prompt_defaults"],
        prompt_artifacts=prompt_artifacts,
    )
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=answer_gt,
        annotation_gt=annotation_gt,
        image=visual["image"],
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(public_query_id),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
    )


__all__ = [
    "TentsSceneTask",
    "TentsObjectiveBinding",
    "build_tents_task_output",
    "load_tents_task_defaults",
    "prepare_tents_visual_case",
    "retry_tents_generation",
    "run_tents_fixed_query_case",
]
