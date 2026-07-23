"""Public pipe-flow task for selecting the tile that must be rotated."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.puzzles.shared.visual_defaults import load_puzzle_noise_defaults
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions

from .shared.annotations import pipe_flow_misrotated_annotation
from .shared.defaults import (
    resolve_answer_label,
    resolve_candidate_count,
    resolve_grid_size_variant,
    resolve_render_params,
    resolve_scene_variant,
)
from .shared.output import build_misrotated_trace_payload
from .shared.prompts import build_prompt
from .shared.rendering import (
    render_pipe_flow_misrotated_scene,
    resolve_pipe_flow_visual_context,
)
from .shared.rules import connected_to_destination, normalize_openings
from .shared.sampling import sample_pipe_flow_misrotated_dataset
from .shared.state import SCENE_ID

TASK_ID = "task_puzzles__pipe_flow__misrotated_tile_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "pipe_flow_misrotated_tile_label"
_NAMESPACE_BASE = "puzzles.pipe_flow.misrotated_tile_label"
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = (
    load_scene_generation_rendering_prompt_defaults("puzzles", SCENE_ID, task_id=TASK_ID)
)
_NOISE_DEFAULTS = load_puzzle_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)


@dataclass(frozen=True)
class _MisrotatedState:
    """Resolved semantic state for one misrotated-tile instance."""

    selected_query_id: str
    task_params: Mapping[str, Any]
    probability_maps: Mapping[str, Mapping[str, float]]
    dataset: Any


@register_task
class PuzzlesPipeFlowMisrotatedTileLabelTask:
    """Choose which labeled pipe tile must be rotated to restore flow."""

    task_id = TASK_ID
    reasoning_operations = ('topology', 'transformation')
    domain = "puzzles"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any],
        max_attempts: int,
    ) -> TaskOutput:
        """Generate one misrotated-tile task."""

        error: Exception | None = None
        for attempt_offset in range(max(1, int(max_attempts))):
            try:
                return self._attempt(
                    seed=int(instance_seed) + int(attempt_offset),
                    params=params,
                    max_attempts=int(max_attempts),
                )
            except (RuntimeError, ValueError) as exc:
                error = exc
        if error is not None:
            raise error
        raise RuntimeError("pipe-flow misrotated-tile generation failed")

    def _attempt(
        self,
        *,
        seed: int,
        params: Mapping[str, Any],
        max_attempts: int,
    ) -> TaskOutput:
        """Render one sampled pipe-flow state and bind answer/annotation together.

        The invariant is that the misrotated tile selected by the sampler is
        the same labeled tile highlighted in the final annotation payload.
        """

        state = self._sample_state(seed=int(seed), params=params)
        render_params = resolve_render_params(
            state.task_params,
            _RENDER_DEFAULTS,
            instance_seed=int(seed),
        )
        visual_context = resolve_pipe_flow_visual_context(
            render_params=render_params,
            instance_seed=int(seed),
            namespace=f"{_NAMESPACE_BASE}.background",
        )
        rendered_scene = render_pipe_flow_misrotated_scene(
            background=visual_context.background,
            dataset=state.dataset,
            render_params=visual_context.render_params,
        )
        image, post_noise_meta = apply_post_image_noise(
            rendered_scene.image,
            instance_seed=int(seed),
            params=state.task_params,
            default_config=_NOISE_DEFAULTS,
        )
        prompt, prompt_variants, prompt_meta = build_prompt(
            _PROMPT_DEFAULTS,
            scene_variant=str(state.dataset.scene_variant),
            prompt_query_key=PROMPT_QUERY_KEY,
            instance_seed=int(seed),
        )
        annotation = pipe_flow_misrotated_annotation(
            dataset=state.dataset,
            rendered_scene=rendered_scene,
        )
        fields = self._task_fields(state=state, max_attempts=int(max_attempts))
        trace_payload = build_misrotated_trace_payload(
            dataset=state.dataset,
            rendered_scene=rendered_scene,
            render_params=visual_context.render_params,
            prompt_meta=prompt_meta,
            task_fields=fields,
            background_meta=visual_context.background_meta,
            scene_style_meta=visual_context.scene_style_meta,
            post_noise_meta=post_noise_meta,
            projected_annotation=annotation.projected_annotation,
            question_format=PROMPT_QUERY_KEY,
        )
        trace_payload["scene_ir"]["relations"]["query_id"] = str(state.selected_query_id)
        trace_payload["query_spec"] = {
            "query_id": str(state.selected_query_id),
            "template_id": str(prompt_meta["bundle_id"]),
            "prompt_variant": dict(prompt_meta["prompt_variant"]),
            "prompt_variant_active_key": str(prompt_meta["prompt_variant_active_key"]),
            "prompt_variants": dict(prompt_meta["prompt_variants_for_trace"]),
            "params": dict(fields),
        }
        return TaskOutput(
            prompt=prompt,
            prompt_variants=prompt_variants,
            answer_gt=TypedValue(type="option_letter", value=str(state.dataset.answer_label)),
            annotation_gt=annotation.annotation_gt,
            image=image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(state.selected_query_id),
        )

    def _sample_state(
        self,
        *,
        seed: int,
        params: Mapping[str, Any],
    ) -> _MisrotatedState:
        """Resolve query, visual axes, and a valid misrotated pipe dataset.

        This keeps all stochastic generation choices explicit before rendering
        so retries cannot drift answer labels away from the sampled board.
        """

        query_id, query_probs, task_params = select_task_query_id(
            instance_seed=int(seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SINGLE_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{_NAMESPACE_BASE}.query",
        )
        axes_rng = spawn_rng(int(seed), f"{_NAMESPACE_BASE}.axes")
        grid_variant, grid_probs = resolve_grid_size_variant(
            axes_rng,
            params=task_params,
            gen_defaults=_GEN_DEFAULTS,
        )
        scene_variant, scene_probs = resolve_scene_variant(
            axes_rng,
            params=task_params,
            gen_defaults=_GEN_DEFAULTS,
        )
        candidate_count, candidate_probs = resolve_candidate_count(
            axes_rng,
            params=task_params,
            gen_defaults=_GEN_DEFAULTS,
        )
        answer_label, answer_probs = resolve_answer_label(
            params=task_params,
            instance_seed=int(seed),
            candidate_count=int(candidate_count),
            namespace=_NAMESPACE_BASE,
        )
        dataset = sample_pipe_flow_misrotated_dataset(
            task_params,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(seed),
            grid_size_variant=str(grid_variant),
            scene_variant=str(scene_variant),
            candidate_count=int(candidate_count),
            answer_label=str(answer_label),
        )
        self._validate_dataset(dataset)
        return _MisrotatedState(
            selected_query_id=str(query_id),
            task_params=dict(task_params),
            probability_maps={
                "query_id": dict(query_probs),
                "grid_size_variant": dict(grid_probs),
                "scene_variant": dict(scene_probs),
                "candidate_count": dict(candidate_probs),
                "answer_label": dict(answer_probs),
            },
            dataset=dataset,
        )

    def _task_fields(
        self,
        *,
        state: _MisrotatedState,
        max_attempts: int,
    ) -> dict[str, Any]:
        dataset = state.dataset
        return {
            "query_id": str(state.selected_query_id),
            "query_id_probabilities": dict(state.probability_maps["query_id"]),
            "internal_question_format": "pipe_flow_misrotated_tile_label",
            "scene_id": SCENE_ID,
            "scene_variant": str(dataset.scene_variant),
            "scene_variant_probabilities": dict(state.probability_maps["scene_variant"]),
            "grid_size_variant": str(dataset.grid_size_variant),
            "grid_size_variant_probabilities": dict(
                state.probability_maps["grid_size_variant"]
            ),
            "candidate_count": int(dataset.candidate_count),
            "candidate_count_probabilities": dict(
                state.probability_maps["candidate_count"]
            ),
            "answer_label": str(dataset.answer_label),
            "answer_label_probabilities": dict(state.probability_maps["answer_label"]),
            "rows": int(dataset.rows),
            "cols": int(dataset.cols),
            "path_length": int(len(dataset.path_cells)),
            "branch_cell_count": int(len(dataset.branch_cells)),
            "branch_terminal_count": int(len(dataset.branch_terminal_cells)),
            "branching_allowed": False,
            "rotation_rule": "rotate_exactly_one_labeled_tile",
            "distractor_policy": "labeled_path_tiles_not_repairable_by_rotation",
            "max_attempts": int(max_attempts),
        }

    def _validate_dataset(self, dataset: Any) -> None:
        candidates = list(dataset.candidates)
        correct = [candidate for candidate in candidates if candidate.is_correct]
        if len(correct) != 1:
            raise ValueError("pipe-flow misrotated task requires exactly one answer")
        if str(correct[0].label) != str(dataset.answer_label):
            raise ValueError("pipe-flow misrotated answer label drifted")
        if str(correct[0].tile_id) != str(dataset.misrotated_tile_id):
            raise ValueError("pipe-flow misrotated tile id drifted")
        if dataset.branch_cells or dataset.branch_terminal_cells:
            raise ValueError("pipe-flow misrotated task must not include branches")
        current_map = {
            (int(tile.row), int(tile.col)): normalize_openings(tile.current_openings)
            for tile in dataset.tiles
        }
        if connected_to_destination(
            current_map,
            rows=int(dataset.rows),
            cols=int(dataset.cols),
            start_cell=tuple(dataset.start_cell),
            destination_cell=tuple(dataset.destination_cell),
        ):
            raise ValueError("pipe-flow misrotated board must start disconnected")
        repairing = [candidate for candidate in candidates if candidate.connects_after_rotation]
        if len(repairing) != 1 or str(repairing[0].label) != str(dataset.answer_label):
            raise ValueError("pipe-flow misrotated task must have one repairable label")
        labels = [str(candidate.label) for candidate in candidates]
        if labels != list("ABCD")[: len(labels)]:
            raise ValueError("pipe-flow misrotated labels must be ordered A-D")


__all__ = [
    "PuzzlesPipeFlowMisrotatedTileLabelTask",
    "PROMPT_QUERY_KEY",
    "SCENE_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
