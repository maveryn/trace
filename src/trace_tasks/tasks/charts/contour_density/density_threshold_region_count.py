"""Public task for `task_charts__contour_density__density_threshold_region_count`."""

from __future__ import annotations

from typing import Any, List, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.contour_density._lifecycle import ContourTaskPlan, contour_task_output_fields, run_contour_public_task
from trace_tasks.tasks.charts.contour_density.shared.defaults import DOMAIN, GENERATION_DEFAULTS, SCENE_NAMESPACE
from trace_tasks.tasks.charts.contour_density.shared.prompts import build_prompt_artifacts
from trace_tasks.tasks.charts.contour_density.shared.sampling import balanced_choice, build_regions, density_from_level, region_count, region_labels, scene_variant
from trace_tasks.tasks.charts.contour_density.shared.state import ContourDataset, DensityThresholdGuide, QuerySelection
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import group_default, resolve_required_int_bounds
from trace_tasks.tasks.shared.fixed_query import select_task_query_id


AT_LEAST_QUERY_ID = "density_at_least_threshold_region_count"
BELOW_QUERY_ID = "density_below_threshold_region_count"
QUERY_DIRECTIONS = {
    AT_LEAST_QUERY_ID: "at_least",
    BELOW_QUERY_ID: "below",
}


def _build_task_output(materialized):
    return TaskOutput(**contour_task_output_fields(materialized))


def _threshold_count_support(params: Mapping[str, Any], *, current_region_count: int) -> Tuple[int, ...]:
    raw_support = params.get(
        "threshold_count_answer_support",
        group_default(GENERATION_DEFAULTS, "threshold_count_answer_support", list(range(1, 6))),
    )
    support: List[int] = []
    if isinstance(raw_support, Sequence) and not isinstance(raw_support, (str, bytes)):
        for value in raw_support:
            try:
                support.append(int(value))
            except Exception:
                continue
    if not support:
        support = list(range(1, 6))
    filtered = sorted({int(value) for value in support if 1 <= int(value) < int(current_region_count)})
    if not filtered:
        filtered = list(range(1, max(2, min(5, int(current_region_count)))))
    return tuple(filtered)


@register_task
class ChartsContourDensityDensityThresholdRegionCountTask:
    """Count regions whose visible density level satisfies a threshold."""

    task_id = "task_charts__contour_density__density_threshold_region_count"
    reasoning_operations = ('filtering', 'counting', 'comparison')
    domain = DOMAIN
    objective_contract = "density_threshold_region_count"
    supported_query_ids = (AT_LEAST_QUERY_ID, BELOW_QUERY_ID)
    default_dataset_enabled = True

    def _build_density_threshold_plan(self, instance_seed: int, params: Mapping[str, Any], selected_query_id: str) -> ContourTaskPlan:
        """Construct density levels to hit the requested count before binding region annotations."""

        scene_name, scene_probabilities = scene_variant(params, instance_seed=int(instance_seed))
        try:
            direction = QUERY_DIRECTIONS[str(selected_query_id)]
        except KeyError as exc:
            raise ValueError(f"unsupported contour-density threshold query: {selected_query_id}") from exc
        threshold_min, threshold_max = resolve_required_int_bounds(
            params,
            GENERATION_DEFAULTS,
            min_key="density_threshold_level_min",
            max_key="density_threshold_level_max",
            fallback_min=2,
            fallback_max=5,
            context=f"density threshold levels for {SCENE_NAMESPACE}",
        )
        level_support = list(range(max(2, int(threshold_min)), min(5, int(threshold_max)) + 1)) or [3, 4]
        threshold_level = int(
            balanced_choice(
                level_support,
                params,
                instance_seed=int(instance_seed),
                namespace=f"{SCENE_NAMESPACE}.density_threshold.level",
            )
        )
        count = region_count(params, instance_seed=int(instance_seed))
        target_count = int(
            balanced_choice(
                _threshold_count_support(params, current_region_count=int(count)),
                params,
                instance_seed=int(instance_seed),
                namespace=f"{SCENE_NAMESPACE}.density_threshold.answer_count",
            )
        )
        labels = region_labels(int(count), instance_seed=int(instance_seed))
        rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.density_threshold.levels")
        candidate_indices = list(range(int(count)))
        rng.shuffle(candidate_indices)
        matching_indices = set(candidate_indices[: int(target_count)])
        density_levels: List[int] = []
        for index in range(int(count)):
            if str(direction) == "at_least":
                level_choices = list(range(int(threshold_level), 6)) if index in matching_indices else list(range(1, int(threshold_level)))
            else:
                level_choices = list(range(1, int(threshold_level))) if index in matching_indices else list(range(int(threshold_level), 6))
            if not level_choices:
                raise RuntimeError("invalid threshold-level construction")
            density_levels.append(int(level_choices[rng.randrange(len(level_choices))]))
        densities = [density_from_level(level) for level in density_levels]
        regions = build_regions(
            count=int(count),
            labels=labels,
            option_labels=(),
            densities=densities,
            density_levels=density_levels,
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.density_threshold.regions",
        )
        if str(direction) == "at_least":
            annotation_regions = tuple(region for region in regions if int(region.density_level) >= int(threshold_level))
            phrase = "at least"
            operator = ">="
        else:
            annotation_regions = tuple(region for region in regions if int(region.density_level) < int(threshold_level))
            phrase = "below"
            operator = "<"
        if len(annotation_regions) != int(target_count):
            raise RuntimeError("density-threshold construction lost target count")
        guide = DensityThresholdGuide(label=f"Density level {operator} {threshold_level}", level=int(threshold_level), operator=str(operator))
        selection = QuerySelection(
            prompt_key=str(selected_query_id),
            answer=int(target_count),
            answer_type="integer",
            annotation_type="bbox_set",
            annotation_roles={},
            annotation_region_ids=tuple(str(region.region_id) for region in annotation_regions),
            trace={
                "density_threshold_direction": str(direction),
                "density_threshold_phrase": str(phrase),
                "density_threshold_operator": str(operator),
                "density_threshold_level": int(threshold_level),
                "density_level_by_region_label": {str(region.label): int(region.density_level) for region in regions},
                "matching_region_labels": [str(region.label) for region in annotation_regions],
                "scene_variant_probabilities": dict(scene_probabilities),
            },
        )
        dataset = ContourDataset(scene_variant=str(scene_name), regions=tuple(regions), query=selection, reference=None, threshold_guide=guide)
        prompt_artifacts = build_prompt_artifacts(
            prompt_query_key=str(selected_query_id),
            dynamic_slots={
                "density_threshold_phrase": str(selection.trace["density_threshold_phrase"]),
                "density_threshold_level": str(selection.trace["density_threshold_level"]),
            },
            instance_seed=int(instance_seed),
        )
        return ContourTaskPlan(
            dataset=dataset,
            prompt_artifacts=prompt_artifacts,
            relations={"scene_variant": str(scene_name), "region_count": int(len(dataset.regions)), **dict(selection.trace)},
        )

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        selected_query_id, _probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=self.supported_query_ids,
            default_query_id=AT_LEAST_QUERY_ID,
            task_id=self.task_id,
        )
        return run_contour_public_task(
            instance_seed=int(instance_seed),
            params=task_params,
            max_attempts=int(max_attempts),
            selected_query_id=str(selected_query_id),
            failure_label=self.task_id,
            build_plan=self._build_density_threshold_plan,
            build_output=_build_task_output,
        )


__all__ = ["ChartsContourDensityDensityThresholdRegionCountTask"]
