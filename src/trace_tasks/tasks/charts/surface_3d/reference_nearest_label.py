"""Public task for `task_charts__surface_3d__reference_nearest_label`."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.charts.surface_3d._lifecycle import Surface3DTaskPlan, run_surface_3d_lifecycle
from trace_tasks.tasks.charts.surface_3d.shared.annotations import point_for_single_witness
from trace_tasks.tasks.charts.surface_3d.shared.defaults import DOMAIN, SCATTER_VARIANT
from trace_tasks.tasks.charts.surface_3d.shared.sampling import (
    PALETTE,
    balanced_choice,
    balanced_int,
    configured_count,
    sample_entity_labels,
)
from trace_tasks.tasks.charts.surface_3d.shared.state import Point3D, Surface3DDataset
from trace_tasks.tasks.registry import register_task


TASK_ID = "task_charts__surface_3d__reference_nearest_label"
OBJECTIVE_CONTRACT = "reference_nearest_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
DEFAULT_QUERY_ID = SINGLE_QUERY_ID
PROMPT_QUERY_KEY = "reference_nearest_label"


def _build_reference_dataset(params, instance_seed):
    """Sample one 3D scatter cloud where exactly one point is nearest to the requested y-axis value."""

    category_count = balanced_int(
        low=configured_count(params, "category_count_min", 5),
        high=configured_count(params, "category_count_max", 8),
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.category_count",
    )
    target_value = balanced_int(
        low=25,
        high=75,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.target_value",
    )
    labels = sample_entity_labels(int(category_count), instance_seed=int(instance_seed), namespace="reference")
    answer_label = str(
        balanced_choice(
            labels,
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}.answer_label",
        )
    )
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.points")
    points: list[Point3D] = []
    for index, label in enumerate(labels):
        if str(label) == answer_label:
            y_value = float(target_value + rng.choice([-2, -1, 1, 2]))
        else:
            offset = int(rng.choice([-1, 1])) * int(rng.randint(18, 38))
            y_value = float(max(5, min(95, int(target_value) + int(offset))))
            if abs(y_value - float(target_value)) <= 16:
                y_value = float(max(5, min(95, int(target_value) + (18 if offset >= 0 else -18))))
        points.append(
            Point3D(
                point_id=f"point_{label}",
                label=str(label),
                x_value=float(rng.randint(8, 92)),
                y_value=float(y_value),
                z_value=float(rng.randint(8, 92)),
                color_rgb=PALETTE[int(index) % len(PALETTE)],
            )
        )
    distances = {str(point.label): round(abs(float(point.y_value) - float(target_value)), 3) for point in points}
    dataset = Surface3DDataset(
        scene_variant=SCATTER_VARIANT,
        points=tuple(points),
        surface_cells=(),
        panels=(),
        x_axis_label="x-axis",
        y_axis_label="y-axis",
        z_axis_label="z-axis",
        x_range=(0.0, 100.0),
        y_range=(0.0, 100.0),
        z_range=(0.0, 100.0),
        x_labels=(),
        y_labels=(),
        title="3D Scatter Chart",
        reference_y_value=float(target_value),
    )
    return dataset, answer_label, int(target_value), distances


def _build_plan(params, instance_seed, selected_branch, query_probabilities):
    """Bind nearest-reference semantics for the sampled 3D scatter chart."""

    if str(selected_branch) != SINGLE_QUERY_ID:
        raise ValueError(f"unsupported query_id for {TASK_ID}: {selected_branch}")
    dataset, answer_label, target_value, distances = _build_reference_dataset(params, int(instance_seed))
    answer_point_id = f"point_{answer_label}"

    def _bind_annotation(rendered):
        return point_for_single_witness(rendered.point_bboxes_px[str(answer_point_id)])

    return Surface3DTaskPlan(
        dataset=dataset,
        answer_gt=TypedValue(type="string", value=str(answer_label)),
        annotation_builder=_bind_annotation,
        prompt_query_key=PROMPT_QUERY_KEY,
        dynamic_slots={
            "target_axis_label": "y-axis",
            "target_axis_value": int(target_value),
        },
        branch_params={
            "target_axis": "y",
            "target_axis_label": "y-axis",
            "target_axis_value": int(target_value),
            "answer_point_id": str(answer_point_id),
            "answer_label": str(answer_label),
            "distances_from_target": dict(distances),
            "category_count": len(dataset.points),
        },
        relations={
            "answer_point_id": str(answer_point_id),
            "answer_label": str(answer_label),
            "query_id_probabilities": dict(query_probabilities),
        },
        witness_symbolic={
            "type": "surface_3d_reference_nearest_witness",
            "point_id": str(answer_point_id),
            "answer": str(answer_label),
        },
        question_format="surface_3d_reference_nearest_label",
    )


class ChartsThreeDReferenceNearestLabelTask:
    task_id = TASK_ID
    reasoning_operations = ('ranking', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = OBJECTIVE_CONTRACT
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True
    default_query_id = DEFAULT_QUERY_ID
    build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed, *, params, max_attempts):
        return run_surface_3d_lifecycle(
            task=self,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            default_query_id=DEFAULT_QUERY_ID,
            build_plan=_build_plan,
        )


register_task(ChartsThreeDReferenceNearestLabelTask)


__all__ = ["ChartsThreeDReferenceNearestLabelTask"]
