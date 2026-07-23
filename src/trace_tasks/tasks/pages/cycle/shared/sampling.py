"""Sampling helpers for directed cycle page diagrams."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.pages.shared.diagram.common import sample_diagram_short_names
from trace_tasks.tasks.shared.config_defaults import resolve_required_int_bounds
from trace_tasks.tasks.shared.deterministic_sampling import resolve_selection_index
from trace_tasks.tasks.shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant

from .defaults import (
    CYCLE_DIRECTIONS,
    DEFAULTS,
    GENERATION_DEFAULTS,
    NAMESPACE_ROOT,
    SCENE_VARIANTS,
    TITLE_OPTIONS,
)
from .state import CycleCase


def _resolve_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    supported: Tuple[str, ...],
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    rng = spawn_rng(int(instance_seed), f"{NAMESPACE_ROOT}.{namespace}")
    selected, probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=GENERATION_DEFAULTS,
        supported_variants=supported,
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
    )
    balanced = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GENERATION_DEFAULTS,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=supported,
        balance_flag_key=str(balance_flag_key),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        sampling_namespace=f"{NAMESPACE_ROOT}:{namespace}",
    )
    return str(balanced), {str(key): float(value) for key, value in sorted(probabilities.items())}


def _select_index(params: Mapping[str, Any], *, instance_seed: int, namespace: str) -> int:
    return int(
        resolve_selection_index(
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{NAMESPACE_ROOT}:{namespace}",
        )
    )


def _sample_title(*, instance_seed: int) -> str:
    rng = spawn_rng(int(instance_seed), f"{NAMESPACE_ROOT}.title")
    return str(TITLE_OPTIONS[int(rng.randrange(len(TITLE_OPTIONS)))])


def build_cycle_case(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    query_relationship: str,
    query_relationship_probabilities: Mapping[str, float] | None = None,
) -> CycleCase:
    """Build one deterministic cycle offset-stage case."""

    scene_variant, scene_variant_probabilities = _resolve_axis(
        instance_seed=int(instance_seed),
        params=params,
        supported=SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        namespace="scene_variant",
    )
    relationship = str(query_relationship)
    if relationship not in {"after", "before"}:
        raise ValueError(f"unsupported query_relationship: {query_relationship}")
    relationship_probabilities = {
        str(key): float(value)
        for key, value in (query_relationship_probabilities or {relationship: 1.0}).items()
    }
    cycle_direction, cycle_direction_probabilities = _resolve_axis(
        instance_seed=int(instance_seed),
        params=params,
        supported=CYCLE_DIRECTIONS,
        explicit_key="cycle_direction",
        weights_key="cycle_direction_weights",
        balance_flag_key="balanced_cycle_direction_sampling",
        namespace="cycle_direction",
    )

    stage_count_min, stage_count_max = resolve_required_int_bounds(
        params,
        GENERATION_DEFAULTS,
        min_key="stage_count_min",
        max_key="stage_count_max",
        fallback_min=int(DEFAULTS.stage_count_min),
        fallback_max=int(DEFAULTS.stage_count_max),
        context="cycle stage count",
    )
    max_step_count = max(1, int(stage_count_max) - 1)
    step_count_min, step_count_max = resolve_required_int_bounds(
        params,
        GENERATION_DEFAULTS,
        min_key="step_count_min",
        max_key="step_count_max",
        fallback_min=int(DEFAULTS.step_count_min),
        fallback_max=int(max_step_count),
        context="cycle step count",
    )
    stage_count = int(
        int(stage_count_min)
        + (_select_index(params, instance_seed=int(instance_seed), namespace="stage_count") % (int(stage_count_max) - int(stage_count_min) + 1))
    )
    step_count_max_for_stage = min(int(step_count_max), max(1, int(stage_count) - 1))
    step_count_min_for_stage = min(int(step_count_min), int(step_count_max_for_stage))
    step_count = int(
        int(step_count_min_for_stage)
        + (
            _select_index(params, instance_seed=int(instance_seed), namespace="step_count")
            % max(1, int(step_count_max_for_stage) - int(step_count_min_for_stage) + 1)
        )
    )
    rng = spawn_rng(int(instance_seed), f"{NAMESPACE_ROOT}.labels")
    stage_labels = sample_diagram_short_names(count=int(stage_count), rng=rng)
    query_index = int(_select_index(params, instance_seed=int(instance_seed), namespace="query_stage_index") % int(stage_count))

    direction_delta = 1 if str(cycle_direction) == "clockwise" else -1
    if relationship == "after":
        answer_index = int((query_index + (direction_delta * step_count)) % int(stage_count))
    elif relationship == "before":
        answer_index = int((query_index - (direction_delta * step_count)) % int(stage_count))
    else:
        raise ValueError(f"unsupported query_relationship: {query_relationship}")

    stage_specs = []
    edge_specs = []
    for index, label in enumerate(stage_labels):
        stage_specs.append(
            {
                "stage_id": f"stage_{index}",
                "stage_bbox_id": f"stage_bbox_{index}",
                "stage_label_bbox_id": f"stage_label_bbox_{index}",
                "stage_label": str(label),
                "order_index": int(index),
            }
        )
        edge_specs.append(
            {
                "edge_id": f"edge_{index}",
                "source_stage_id": f"stage_{index}",
                "target_stage_id": f"stage_{(index + direction_delta) % int(stage_count)}",
                "direction": str(cycle_direction),
            }
        )

    return CycleCase(
        scene_title=_sample_title(instance_seed=int(instance_seed)),
        scene_variant=str(scene_variant),
        query_relationship=str(relationship),
        cycle_direction=str(cycle_direction),
        stage_count=int(stage_count),
        step_count=int(step_count),
        query_stage_index=int(query_index),
        query_stage_id=f"stage_{query_index}",
        query_stage_label=str(stage_labels[query_index]),
        answer_stage_index=int(answer_index),
        answer_stage_id=f"stage_{answer_index}",
        answer_stage_label=str(stage_labels[answer_index]),
        answer_stage_bbox_id=f"stage_bbox_{answer_index}",
        stage_specs=tuple(dict(spec) for spec in stage_specs),
        edge_specs=tuple(dict(spec) for spec in edge_specs),
        prompt_slots={
            "step_count": int(step_count),
            "step_noun": "step" if int(step_count) == 1 else "steps",
            "query_relationship": str(relationship),
            "query_stage_label": str(stage_labels[query_index]),
        },
        scene_variant_probabilities=dict(scene_variant_probabilities),
        query_relationship_probabilities=dict(relationship_probabilities),
        cycle_direction_probabilities=dict(cycle_direction_probabilities),
    )
