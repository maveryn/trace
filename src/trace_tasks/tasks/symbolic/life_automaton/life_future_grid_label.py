"""Choose the future grid of a symbolic Life automaton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from ....core.seed import spawn_rng
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from ...shared.fixed_query import select_task_query_id
from ...shared.output_metadata import default_task_versions
from ..shared.common import get_int_param as _get_int
from ..shared.common import get_int_range as _get_range

from .shared.annotations import annotation_trace_payload, keyed_bboxes
from ._lifecycle import life_query_params, render_life_task_scene
from .shared.output import build_life_trace_payload, life_grid_execution_fields
from .shared.prompts import build_life_prompt
from .shared.rules import SCENE_ID, choose_life_options, sample_life_grid, sample_square_grid_size, simulate_life
from .shared.state import LifeOptionSpec, LifeSceneSpec
from .shared.styles import resolve_scene_variant


DOMAIN = "symbolic"
TASK_ID = "task_symbolic__life_automaton__life_future_grid_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = ("one_step_future_grid", "two_step_future_grid")
QUESTION_FORMAT = "life_future_grid_label"
TASK_PROMPT_KEY = "life_future_grid_query"
_STEPS_BY_QUERY_ID = {
    "one_step_future_grid": 1,
    "two_step_future_grid": 2,
}

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
    task_id=TASK_ID,
)


@register_task
class SymbolicLifeAutomatonFutureGridLabelTask:
    """Choose the option showing the grid after one or two Life updates."""

    task_id = TASK_ID
    reasoning_operations = ('state_update',)
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one Life future-grid option task."""

        query_id, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SUPPORTED_QUERY_IDS[0],
            task_id=TASK_ID,
            namespace=f"{QUESTION_FORMAT}.query",
        )
        scene_variant, scene_variant_probabilities = resolve_scene_variant(
            params=task_params,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
            sampling_scope=QUESTION_FORMAT,
        )
        last_error: Exception | None = None
        dataset: _FutureGridDataset | None = None
        for attempt_index in range(max(1, int(max_attempts))):
            try:
                dataset = _build_future_grid_dataset(
                    params=task_params,
                    instance_seed=int(instance_seed) + int(attempt_index),
                    query_id=str(query_id),
                    scene_variant=str(scene_variant),
                    scene_variant_probabilities=scene_variant_probabilities,
                )
                break
            except Exception as exc:  # pragma: no cover - exercised by smoke/review generation.
                last_error = exc
                dataset = None
        if dataset is None:
            raise RuntimeError(f"could not generate {TASK_ID}: {last_error}") from last_error

        scene = LifeSceneSpec(
            rows=int(dataset.rows),
            cols=int(dataset.cols),
            initial_grid=dataset.initial_grid,
            future_grid=dataset.future_grid,
            option_specs=dataset.option_specs,
            source_marker_label="START",
        )
        render_bundle = render_life_task_scene(
            scene=scene,
            params=task_params,
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            scene_variant=str(dataset.scene_variant),
            instance_seed=int(instance_seed),
            sampling_scope=QUESTION_FORMAT,
        )
        prompt, prompt_variants, prompt_meta, prompt_artifacts = build_life_prompt(
            domain=DOMAIN,
            prompt_defaults=_PROMPT_DEFAULTS,
            scene_variant=str(dataset.scene_variant),
            query_key=str(query_id),
            steps=int(dataset.steps),
            instance_seed=int(instance_seed),
            task_key=TASK_PROMPT_KEY,
            query_instruction_key=f"query_instruction_{query_id}",
            annotation_hint_key="annotation_hint_life_future_grid_label",
            answer_hint_key="answer_hint_life_future_grid_label",
            json_example_key="json_example_life_future_grid_label",
            json_example_answer_only_key="json_example_answer_only_life_future_grid_label",
        )
        correct_option = next(option for option in dataset.option_specs if option.is_correct)
        annotation_role_item_ids = {
            "source_grid": "source_grid",
            "selected_option": str(correct_option.option_id),
        }
        annotation_value = keyed_bboxes(render_bundle.rendered.item_bboxes, annotation_role_item_ids)
        answer_gt = TypedValue(type="option_letter", value=str(dataset.answer_label))
        annotation_gt = TypedValue(type="bbox_map", value=dict(annotation_value))
        query_params = life_query_params(
            query_id=str(query_id),
            query_probabilities=query_probabilities,
            question_format=QUESTION_FORMAT,
            scene_variant=str(dataset.scene_variant),
            scene_variant_probabilities=dataset.scene_variant_probabilities,
            render_bundle=render_bundle,
        )
        witness_symbolic, projected_annotation = annotation_trace_payload(
            annotation_type="bbox_map",
            annotation_value=annotation_value,
        )
        execution_trace = {
            **dict(query_params),
            "task_id": TASK_ID,
            "answer_value": str(dataset.answer_label),
            **life_grid_execution_fields(
                rows=dataset.rows,
                cols=dataset.cols,
                steps=dataset.steps,
                initial_grid=dataset.initial_grid,
                future_grid=dataset.future_grid,
            ),
            "option_specs": [
                {
                    "option_id": str(option.option_id),
                    "label": str(option.label),
                    "grid": [list(row) for row in option.grid],
                    "is_correct": bool(option.is_correct),
                }
                for option in dataset.option_specs
            ],
            "supporting_item_ids": list(annotation_role_item_ids.values()),
            "supporting_item_ids_by_role": dict(annotation_role_item_ids),
        }
        trace_payload = build_life_trace_payload(
            scene_name=SCENE_ID,
            prompt_artifacts=prompt_artifacts,
            branch_name=str(query_id),
            params_payload=query_params,
            render_bundle=render_bundle,
            execution_record=execution_trace,
            witness_symbolic=witness_symbolic,
            projected_annotation=projected_annotation,
            render_map_extra={"annotation_role_item_ids": dict(annotation_role_item_ids)},
        )
        return TaskOutput(
            prompt=prompt,
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=render_bundle.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(query_id),
            prompt_variants=prompt_variants,
        )


@dataclass(frozen=True)
class _FutureGridDataset:
    scene_variant: str
    scene_variant_probabilities: dict[str, float]
    rows: int
    cols: int
    steps: int
    initial_grid: Tuple[Tuple[int, ...], ...]
    future_grid: Tuple[Tuple[int, ...], ...]
    option_specs: Tuple[LifeOptionSpec, ...]
    answer_label: str


def _build_future_grid_dataset(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    query_id: str,
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
) -> _FutureGridDataset:
    """Sample one source grid and option set for a selected Life step count."""

    rng = spawn_rng(int(instance_seed), f"{QUESTION_FORMAT}.life")
    grid_size_min, grid_size_max = _get_range(
        params,
        _GEN_DEFAULTS,
        min_key="life_grid_size_min",
        max_key="life_grid_size_max",
        fallback_min=3,
        fallback_max=5,
    )
    rows_min, rows_max = _get_range(params, _GEN_DEFAULTS, min_key="life_rows_min", max_key="life_rows_max", fallback_min=grid_size_min, fallback_max=grid_size_max)
    cols_min, cols_max = _get_range(params, _GEN_DEFAULTS, min_key="life_cols_min", max_key="life_cols_max", fallback_min=grid_size_min, fallback_max=grid_size_max)
    grid_size = sample_square_grid_size(
        rng,
        grid_size_min=grid_size_min,
        grid_size_max=grid_size_max,
        rows_min=rows_min,
        rows_max=rows_max,
        cols_min=cols_min,
        cols_max=cols_max,
    )
    rows = int(grid_size)
    cols = int(grid_size)
    density = float(params.get("live_density", group_default(_GEN_DEFAULTS, "live_density", 0.36)))
    initial_grid = sample_life_grid(rng, rows=rows, cols=cols, live_prob=density)
    steps = int(_STEPS_BY_QUERY_ID[str(query_id)])
    future_grid = simulate_life(initial_grid, steps=steps)
    option_count = _get_int(params, _GEN_DEFAULTS, "grid_option_count", 4)
    option_specs, answer_label = choose_life_options(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{QUESTION_FORMAT}.{query_id}.answer_option",
        future_grid=future_grid,
        option_count=option_count,
        rng=rng,
    )
    return _FutureGridDataset(
        scene_variant=str(scene_variant),
        scene_variant_probabilities={str(key): float(value) for key, value in scene_variant_probabilities.items()},
        rows=int(rows),
        cols=int(cols),
        steps=int(steps),
        initial_grid=tuple(initial_grid),
        future_grid=tuple(future_grid),
        option_specs=tuple(option_specs),
        answer_label=str(answer_label),
    )
