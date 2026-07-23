"""Public task for `task_charts__surface_3d__series_trend_label`."""

from __future__ import annotations

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.charts.surface_3d._lifecycle import Surface3DTaskPlan, run_surface_3d_lifecycle
from trace_tasks.tasks.charts.surface_3d.shared.annotations import segment_between_bboxes
from trace_tasks.tasks.charts.surface_3d.shared.defaults import DOMAIN, SCATTER_VARIANT
from trace_tasks.tasks.charts.surface_3d.shared.sampling import (
    PALETTE,
    TIME_POOL,
    balanced_choice,
    balanced_int,
    configured_count,
    sample_entity_labels,
)
from trace_tasks.tasks.charts.surface_3d.shared.state import Point3D, Surface3DDataset
from trace_tasks.tasks.registry import register_task


TASK_ID = "task_charts__surface_3d__series_trend_label"
OBJECTIVE_CONTRACT = "series_trend_label"
INCREASE_QUERY_ID = "increase"
DECREASE_QUERY_ID = "decrease"
SUPPORTED_QUERY_IDS = (INCREASE_QUERY_ID, DECREASE_QUERY_ID)
DEFAULT_QUERY_ID = INCREASE_QUERY_ID


def _target_delta(rng, selected_branch):
    if str(selected_branch) == INCREASE_QUERY_ID:
        return int(rng.randint(38, 55))
    if str(selected_branch) == DECREASE_QUERY_ID:
        return -int(rng.randint(38, 55))
    raise ValueError(f"unsupported query_id for {TASK_ID}: {selected_branch}")


def _distractor_delta(rng, selected_branch):
    delta = int(rng.randint(-24, 28))
    if str(selected_branch) == INCREASE_QUERY_ID:
        return min(int(delta), 24)
    if str(selected_branch) == DECREASE_QUERY_ID:
        return max(int(delta), -24)
    raise ValueError(f"unsupported query_id for {TASK_ID}: {selected_branch}")


def _build_series_dataset(params, instance_seed, selected_branch):
    """Sample connected 3D series with exactly one strongest first-to-last trend."""

    series_count = balanced_int(
        low=configured_count(params, "series_count_min", 4),
        high=configured_count(params, "series_count_max", 6),
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.series_count",
    )
    time_count = balanced_int(
        low=configured_count(params, "time_count_min", 5),
        high=configured_count(params, "time_count_max", 7),
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.time_count",
    )
    labels = sample_entity_labels(int(series_count), instance_seed=int(instance_seed), namespace="trend_series")
    answer_label = str(
        balanced_choice(
            labels,
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}.answer.{selected_branch}",
        )
    )
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.values")
    points: list[Point3D] = []
    deltas: dict[str, int] = {}
    for series_index, label in enumerate(labels):
        delta = _target_delta(rng, str(selected_branch)) if str(label) == answer_label else _distractor_delta(rng, str(selected_branch))
        start_low = 18 if delta >= 0 else 58
        start_high = 38 if delta >= 0 else 82
        start = int(rng.randint(start_low, start_high))
        end = max(5, min(96, int(start) + int(delta)))
        deltas[str(label)] = int(end) - int(start)
        for time_index in range(int(time_count)):
            t = float(time_index) / float(max(1, int(time_count) - 1))
            value = float(start) + (float(end - start) * t) + rng.uniform(-3.0, 3.0)
            points.append(
                Point3D(
                    point_id=f"series_{label}_{time_index}",
                    label=str(label),
                    x_value=float(time_index),
                    y_value=float(series_index),
                    z_value=max(0.0, min(100.0, float(value))),
                    color_rgb=PALETTE[int(series_index) % len(PALETTE)],
                )
            )
    if str(selected_branch) == INCREASE_QUERY_ID:
        winner = max(deltas, key=lambda label: (int(deltas[label]), str(label)))
    else:
        winner = min(deltas, key=lambda label: (int(deltas[label]), str(label)))
    if str(winner) != str(answer_label):
        raise ValueError("series trend answer is not unique")
    dataset = Surface3DDataset(
        scene_variant=SCATTER_VARIANT,
        points=tuple(points),
        surface_cells=(),
        panels=(),
        x_axis_label="x-axis",
        y_axis_label="y-axis",
        z_axis_label="z-axis",
        x_range=(0.0, float(max(1, int(time_count) - 1))),
        y_range=(0.0, float(max(1, int(series_count) - 1))),
        z_range=(0.0, 100.0),
        x_labels=tuple(str(value) for value in TIME_POOL[: int(time_count)]),
        y_labels=tuple(labels),
        title="3D Series Trend Chart",
        connect_points_by_label=True,
    )
    return dataset, answer_label, deltas


def _build_plan(params, instance_seed, selected_branch, query_probabilities):
    """Bind endpoint trend semantics for one connected 3D series chart."""

    if str(selected_branch) not in SUPPORTED_QUERY_IDS:
        raise ValueError(f"unsupported query_id for {TASK_ID}: {selected_branch}")
    dataset, answer_label, deltas = _build_series_dataset(params, int(instance_seed), str(selected_branch))
    time_count = len(dataset.x_labels)
    start_point_id = f"series_{answer_label}_0"
    end_point_id = f"series_{answer_label}_{int(time_count) - 1}"

    def _bind_annotation(rendered):
        return segment_between_bboxes(
            rendered.point_bboxes_px[str(start_point_id)],
            rendered.point_bboxes_px[str(end_point_id)],
        )

    return Surface3DTaskPlan(
        dataset=dataset,
        answer_gt=TypedValue(type="string", value=str(answer_label)),
        annotation_builder=_bind_annotation,
        prompt_query_key=str(selected_branch),
        dynamic_slots={},
        branch_params={
            "answer_series_label": str(answer_label),
            "start_point_id": str(start_point_id),
            "end_point_id": str(end_point_id),
            "series_count": len(dataset.y_labels),
            "time_count": int(time_count),
            "deltas_by_series": dict(deltas),
        },
        relations={
            "answer_series_label": str(answer_label),
            "start_point_id": str(start_point_id),
            "end_point_id": str(end_point_id),
            "query_id_probabilities": dict(query_probabilities),
        },
        witness_symbolic={
            "type": "surface_3d_series_trend_witness",
            "start_point_id": str(start_point_id),
            "end_point_id": str(end_point_id),
            "answer": str(answer_label),
        },
        question_format="surface_3d_series_trend_label",
    )


class ChartsThreeDSeriesTrendLabelTask:
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


register_task(ChartsThreeDSeriesTrendLabelTask)


__all__ = ["ChartsThreeDSeriesTrendLabelTask"]
