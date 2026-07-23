"""Public task for `task_charts__contour_density__density_extremum_region_label`."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.contour_density._lifecycle import ContourTaskPlan, contour_task_output_fields, run_contour_public_task
from trace_tasks.tasks.charts.contour_density.shared.defaults import DOMAIN, SCENE_NAMESPACE
from trace_tasks.tasks.charts.contour_density.shared.prompts import build_prompt_artifacts
from trace_tasks.tasks.charts.contour_density.shared.sampling import (
    balanced_choice,
    build_regions,
    density_from_level,
    region_count,
    region_labels,
    scene_variant,
)
from trace_tasks.tasks.charts.contour_density.shared.state import ContourDataset, QuerySelection
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id


HIGHEST_QUERY_ID = "highest_density_region_label"
LOWEST_QUERY_ID = "lowest_density_region_label"
QUERY_DENSITY_EXTREMA = {
    HIGHEST_QUERY_ID: "highest",
    LOWEST_QUERY_ID: "lowest",
}


def _build_task_output(materialized):
    return TaskOutput(**contour_task_output_fields(materialized))


@register_task
class ChartsContourDensityDensityExtremumRegionLabelTask:
    """Return the region label with the highest or lowest density."""

    task_id = "task_charts__contour_density__density_extremum_region_label"
    reasoning_operations = ('ranking',)
    domain = DOMAIN
    objective_contract = "density_extremum_region_label"
    supported_query_ids = (HIGHEST_QUERY_ID, LOWEST_QUERY_ID)
    default_dataset_enabled = True

    def _build_density_extremum_plan(self, instance_seed: int, params: Mapping[str, Any], selected_query_id: str) -> ContourTaskPlan:
        """Bind the unique density extremum before neutral rendering projects the selected region."""

        scene_name, scene_probabilities = scene_variant(params, instance_seed=int(instance_seed))
        try:
            extremum = QUERY_DENSITY_EXTREMA[str(selected_query_id)]
        except KeyError as exc:
            raise ValueError(f"unsupported contour-density extremum query: {selected_query_id}") from exc
        count = region_count(params, instance_seed=int(instance_seed))
        labels = region_labels(int(count), instance_seed=int(instance_seed))
        answer_index = int(
            balanced_choice(
                list(range(int(count))),
                params,
                instance_seed=int(instance_seed),
                namespace=f"{SCENE_NAMESPACE}.density_extremum.answer",
            )
        )
        density_levels = [2 + (index % 2) for index in range(int(count))]
        if str(extremum) == "highest":
            density_levels[int(answer_index)] = 5
            for index in range(int(count)):
                if int(index) != int(answer_index):
                    density_levels[int(index)] = 2 + (int(index) % 2)
        else:
            density_levels[int(answer_index)] = 1
            for index in range(int(count)):
                if int(index) != int(answer_index):
                    density_levels[int(index)] = 3 + (int(index) % 2)
        densities = [density_from_level(int(level)) for level in density_levels]
        regions = build_regions(
            count=int(count),
            labels=labels,
            option_labels=(),
            densities=densities,
            density_levels=density_levels,
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.density_extremum.regions",
        )
        density_by_label = {str(region.label): float(region.density) for region in regions}
        answer_label = max(density_by_label, key=lambda label: (density_by_label[label], label)) if str(extremum) == "highest" else min(density_by_label, key=lambda label: (density_by_label[label], label))
        answer_region = next(region for region in regions if str(region.label) == str(answer_label))
        selection = QuerySelection(
            prompt_key=str(selected_query_id),
            answer=str(answer_label),
            answer_type="string",
            annotation_type="bbox",
            annotation_roles={},
            annotation_region_ids=(str(answer_region.region_id),),
            trace={
                "density_extremum": str(extremum),
                "density_extremum_phrase": "highest" if str(extremum) == "highest" else "lowest",
                "density_by_region_label": {key: round(float(value), 3) for key, value in density_by_label.items()},
                "scene_variant_probabilities": dict(scene_probabilities),
            },
        )
        dataset = ContourDataset(scene_variant=str(scene_name), regions=tuple(regions), query=selection, reference=None)
        prompt_artifacts = build_prompt_artifacts(
            prompt_query_key=str(selected_query_id),
            dynamic_slots={"density_extremum_phrase": str(selection.trace["density_extremum_phrase"])},
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
            default_query_id=HIGHEST_QUERY_ID,
            task_id=self.task_id,
        )
        return run_contour_public_task(
            instance_seed=int(instance_seed),
            params=task_params,
            max_attempts=int(max_attempts),
            selected_query_id=str(selected_query_id),
            failure_label=self.task_id,
            build_plan=self._build_density_extremum_plan,
            build_output=_build_task_output,
        )

__all__ = ["ChartsContourDensityDensityExtremumRegionLabelTask"]
