"""Public task for `task_charts__contour_density__reference_distance_extremum_label`."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.contour_density._lifecycle import ContourTaskPlan, contour_task_output_fields, run_contour_public_task
from trace_tasks.tasks.charts.contour_density.shared.defaults import DOMAIN, SCENE_NAMESPACE
from trace_tasks.tasks.charts.contour_density.shared.prompts import build_prompt_artifacts
from trace_tasks.tasks.charts.contour_density.shared.sampling import balanced_choice, build_regions, distance_to_reference, density_from_level, region_count, region_labels, scene_variant
from trace_tasks.tasks.charts.contour_density.shared.state import ContourDataset, QuerySelection, Reference, Region
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id


POINT_NEAREST_QUERY_ID = "point_nearest_region_label"
POINT_FARTHEST_QUERY_ID = "point_farthest_region_label"
VERTICAL_NEAREST_QUERY_ID = "vertical_line_nearest_region_label"
VERTICAL_FARTHEST_QUERY_ID = "vertical_line_farthest_region_label"
HORIZONTAL_NEAREST_QUERY_ID = "horizontal_line_nearest_region_label"
HORIZONTAL_FARTHEST_QUERY_ID = "horizontal_line_farthest_region_label"
QUERY_REFERENCE_DISTANCE = {
    POINT_NEAREST_QUERY_ID: ("point", "nearest"),
    POINT_FARTHEST_QUERY_ID: ("point", "farthest"),
    VERTICAL_NEAREST_QUERY_ID: ("vertical_line", "nearest"),
    VERTICAL_FARTHEST_QUERY_ID: ("vertical_line", "farthest"),
    HORIZONTAL_NEAREST_QUERY_ID: ("horizontal_line", "nearest"),
    HORIZONTAL_FARTHEST_QUERY_ID: ("horizontal_line", "farthest"),
}


def _build_task_output(materialized):
    return TaskOutput(**contour_task_output_fields(materialized))


def _reference_for_answer(region: Region, *, kind: str, distance_extremum: str) -> Reference:
    if str(kind) == "point":
        if str(distance_extremum) == "nearest":
            return Reference(kind="point", x_value=max(4.0, min(96.0, region.center_x + 2.0)), y_value=max(4.0, min(96.0, region.center_y + 2.0)))
        return Reference(kind="point", x_value=100.0 - float(region.center_x), y_value=100.0 - float(region.center_y))
    if str(kind) == "vertical_line":
        if str(distance_extremum) == "nearest":
            return Reference(kind="vertical_line", x_value=float(region.center_x) + 1.5, y_value=50.0)
        return Reference(kind="vertical_line", x_value=5.0 if float(region.center_x) > 50.0 else 95.0, y_value=50.0)
    if str(kind) == "horizontal_line":
        if str(distance_extremum) == "nearest":
            return Reference(kind="horizontal_line", x_value=50.0, y_value=float(region.center_y) + 1.5)
        return Reference(kind="horizontal_line", x_value=50.0, y_value=5.0 if float(region.center_y) > 50.0 else 95.0)
    raise ValueError(f"unsupported reference kind: {kind}")


@register_task
class ChartsContourDensityReferenceDistanceExtremumLabelTask:
    """Return the region label nearest to or farthest from a reference mark."""

    task_id = "task_charts__contour_density__reference_distance_extremum_label"
    reasoning_operations = ('ranking', 'spatial_relations')
    domain = DOMAIN
    objective_contract = "reference_distance_extremum_label"
    supported_query_ids = (
        POINT_NEAREST_QUERY_ID,
        POINT_FARTHEST_QUERY_ID,
        VERTICAL_NEAREST_QUERY_ID,
        VERTICAL_FARTHEST_QUERY_ID,
        HORIZONTAL_NEAREST_QUERY_ID,
        HORIZONTAL_FARTHEST_QUERY_ID,
    )
    default_dataset_enabled = True

    def _build_reference_distance_plan(self, instance_seed: int, params: Mapping[str, Any], selected_query_id: str) -> ContourTaskPlan:
        """Choose a reference mark from the answer region so nearest/farthest ranking is unique."""

        scene_name, scene_probabilities = scene_variant(params, instance_seed=int(instance_seed))
        try:
            reference_kind, distance_extremum = QUERY_REFERENCE_DISTANCE[str(selected_query_id)]
        except KeyError as exc:
            raise ValueError(f"unsupported contour-density reference-distance query: {selected_query_id}") from exc
        count = region_count(params, instance_seed=int(instance_seed))
        labels = region_labels(int(count), instance_seed=int(instance_seed))
        answer_index = int(
            balanced_choice(
                list(range(int(count))),
                params,
                instance_seed=int(instance_seed),
                namespace=f"{SCENE_NAMESPACE}.reference_distance.answer",
            )
        )
        density_levels = [1 + (index % 5) for index in range(int(count))]
        densities = [density_from_level(level) for level in density_levels]
        regions = list(
            build_regions(
                count=int(count),
                labels=labels,
                option_labels=(),
                densities=densities,
                density_levels=density_levels,
                params=params,
                instance_seed=int(instance_seed),
                namespace=f"{SCENE_NAMESPACE}.reference_distance.regions",
            )
        )
        answer_region = regions[int(answer_index)]
        reference = _reference_for_answer(answer_region, kind=str(reference_kind), distance_extremum=str(distance_extremum))
        distances = {str(region.label): distance_to_reference(region, reference) for region in regions}
        answer_label = min(distances, key=lambda label: (distances[label], label)) if str(distance_extremum) == "nearest" else max(distances, key=lambda label: (distances[label], label))
        if str(answer_label) != str(answer_region.label):
            raise RuntimeError("reference-distance construction lost unique answer")
        selection = QuerySelection(
            prompt_key=str(selected_query_id),
            answer=str(answer_label),
            answer_type="string",
            annotation_type="bbox",
            annotation_roles={},
            annotation_region_ids=(str(answer_region.region_id),),
            trace={
                "distance_extremum": str(distance_extremum),
                "distance_extremum_phrase": "nearest to" if str(distance_extremum) == "nearest" else "farthest from",
                "reference_kind": str(reference_kind),
                "reference_kind_phrase": {
                    "point": "reference point labeled \"R\"",
                    "vertical_line": "vertical reference line",
                    "horizontal_line": "horizontal reference line",
                }[str(reference_kind)],
                "reference_value": {"x": round(float(reference.x_value), 3), "y": round(float(reference.y_value), 3)},
                "distances_by_region_label": {key: round(float(value), 3) for key, value in distances.items()},
                "scene_variant_probabilities": dict(scene_probabilities),
            },
        )
        dataset = ContourDataset(scene_variant=str(scene_name), regions=tuple(regions), query=selection, reference=reference)
        prompt_artifacts = build_prompt_artifacts(
            prompt_query_key=str(selected_query_id),
            dynamic_slots={
                "distance_extremum_phrase": str(selection.trace["distance_extremum_phrase"]),
                "reference_kind_phrase": str(selection.trace["reference_kind_phrase"]),
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
            default_query_id=POINT_NEAREST_QUERY_ID,
            task_id=self.task_id,
        )
        return run_contour_public_task(
            instance_seed=int(instance_seed),
            params=task_params,
            max_attempts=int(max_attempts),
            selected_query_id=str(selected_query_id),
            failure_label=self.task_id,
            build_plan=self._build_reference_distance_plan,
            build_output=_build_task_output,
        )


__all__ = ["ChartsContourDensityReferenceDistanceExtremumLabelTask"]
