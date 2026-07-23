"""Public task for `task_charts__contour_density__spread_extremum_region_label`."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, List, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.contour_density._lifecycle import ContourTaskPlan, contour_task_output_fields, run_contour_public_task
from trace_tasks.tasks.charts.contour_density.shared.defaults import DOMAIN, SCENE_NAMESPACE
from trace_tasks.tasks.charts.contour_density.shared.prompts import build_prompt_artifacts
from trace_tasks.tasks.charts.contour_density.shared.sampling import balanced_choice, build_regions, density_from_level, fit_regions_within_unit_bounds, region_count, region_labels, scene_variant
from trace_tasks.tasks.charts.contour_density.shared.state import ContourDataset, QuerySelection, Region
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id


WIDEST_QUERY_ID = "widest_spread_region_label"
NARROWEST_QUERY_ID = "narrowest_spread_region_label"
QUERY_SPREAD_EXTREMA = {
    WIDEST_QUERY_ID: "widest",
    NARROWEST_QUERY_ID: "narrowest",
}


def _build_task_output(materialized):
    return TaskOutput(**contour_task_output_fields(materialized))


def _spread_density_profile(count: int, *, rng: Any) -> Tuple[List[int], List[float]]:
    density_levels = [1 + ((index + rng.randrange(5)) % 5) for index in range(int(count))]
    return density_levels, [density_from_level(level) for level in density_levels]


def _spread_radius_pair(*, rng: Any, is_answer: bool, spread_extremum: str) -> Tuple[float, float]:
    if str(spread_extremum) == "widest":
        return (
            rng.uniform(13.5, 15.0) if bool(is_answer) else rng.uniform(6.0, 9.5),
            rng.uniform(12.0, 13.8) if bool(is_answer) else rng.uniform(5.4, 8.4),
        )
    return (
        rng.uniform(5.1, 6.2) if bool(is_answer) else rng.uniform(8.4, 12.0),
        rng.uniform(4.8, 5.8) if bool(is_answer) else rng.uniform(7.2, 11.2),
    )


def _construct_spread_regions(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    labels: Sequence[str],
    answer_index: int,
    spread_extremum: str,
) -> Tuple[Region, ...]:
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.spread_extremum.regions")
    density_levels, densities = _spread_density_profile(len(labels), rng=rng)
    base_regions = build_regions(
        count=len(labels),
        labels=labels,
        option_labels=(),
        densities=densities,
        density_levels=density_levels,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.spread_extremum.base_regions",
    )
    regions: List[Region] = []
    for index, region in enumerate(base_regions):
        radius_x, radius_y = _spread_radius_pair(
            rng=rng,
            is_answer=int(index) == int(answer_index),
            spread_extremum=str(spread_extremum),
        )
        regions.append(replace(region, radius_x=float(radius_x), radius_y=float(radius_y)))
    return fit_regions_within_unit_bounds(regions)


def _footprint_area_by_label(regions: Sequence[Region]) -> Mapping[str, float]:
    return {str(region.label): float(region.radius_x) * float(region.radius_y) for region in regions}


def _spread_answer_label(spread_by_label: Mapping[str, float], *, spread_extremum: str) -> str:
    if str(spread_extremum) == "widest":
        return str(max(spread_by_label, key=lambda label: (spread_by_label[label], label)))
    return str(min(spread_by_label, key=lambda label: (spread_by_label[label], label)))


@register_task
class ChartsContourDensitySpreadExtremumRegionLabelTask:
    """Return the region label with the widest or narrowest visible footprint."""

    task_id = "task_charts__contour_density__spread_extremum_region_label"
    reasoning_operations = ('ranking', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = "spread_extremum_region_label"
    supported_query_ids = (WIDEST_QUERY_ID, NARROWEST_QUERY_ID)
    default_dataset_enabled = True

    def _build_spread_extremum_plan(self, instance_seed: int, params: Mapping[str, Any], selected_query_id: str) -> ContourTaskPlan:
        """Resize the sampled answer region to create a unique visible-footprint extremum."""

        scene_name, scene_probabilities = scene_variant(params, instance_seed=int(instance_seed))
        try:
            spread_extremum = QUERY_SPREAD_EXTREMA[str(selected_query_id)]
        except KeyError as exc:
            raise ValueError(f"unsupported contour-density spread query: {selected_query_id}") from exc
        count = region_count(params, instance_seed=int(instance_seed))
        labels = region_labels(int(count), instance_seed=int(instance_seed))
        answer_index = int(
            balanced_choice(
                list(range(int(count))),
                params,
                instance_seed=int(instance_seed),
                namespace=f"{SCENE_NAMESPACE}.spread_extremum.answer",
            )
        )
        regions = _construct_spread_regions(
            params=params,
            instance_seed=int(instance_seed),
            labels=labels,
            answer_index=int(answer_index),
            spread_extremum=str(spread_extremum),
        )
        spread_by_label = _footprint_area_by_label(regions)
        answer_label = _spread_answer_label(spread_by_label, spread_extremum=str(spread_extremum))
        phrase = "widest" if str(spread_extremum) == "widest" else "narrowest"
        answer_region = next(region for region in regions if str(region.label) == str(answer_label))
        if str(answer_region.region_id) != f"region_{int(answer_index)}":
            raise RuntimeError("spread-extremum construction lost unique answer")
        selection = QuerySelection(
            prompt_key=str(selected_query_id),
            answer=str(answer_label),
            answer_type="string",
            annotation_type="bbox",
            annotation_roles={},
            annotation_region_ids=(str(answer_region.region_id),),
            trace={
                "spread_extremum": str(spread_extremum),
                "spread_extremum_phrase": str(phrase),
                "footprint_area_by_region_label": {key: round(float(value), 3) for key, value in spread_by_label.items()},
                "scene_variant_probabilities": dict(scene_probabilities),
            },
        )
        dataset = ContourDataset(scene_variant=str(scene_name), regions=tuple(regions), query=selection, reference=None)
        prompt_artifacts = build_prompt_artifacts(
            prompt_query_key=str(selected_query_id),
            dynamic_slots={"spread_extremum_phrase": str(selection.trace["spread_extremum_phrase"])},
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
            default_query_id=WIDEST_QUERY_ID,
            task_id=self.task_id,
        )
        return run_contour_public_task(
            instance_seed=int(instance_seed),
            params=task_params,
            max_attempts=int(max_attempts),
            selected_query_id=str(selected_query_id),
            failure_label=self.task_id,
            build_plan=self._build_spread_extremum_plan,
            build_output=_build_task_output,
        )


__all__ = ["ChartsContourDensitySpreadExtremumRegionLabelTask"]
