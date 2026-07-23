"""Public task for `task_charts__surface_3d__panel_variation_label`."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.charts.surface_3d._lifecycle import Surface3DTaskPlan, run_surface_3d_lifecycle
from trace_tasks.tasks.charts.surface_3d.shared.annotations import bbox_for_single_witness
from trace_tasks.tasks.charts.surface_3d.shared.defaults import DOMAIN, PANEL_VARIANT
from trace_tasks.tasks.charts.surface_3d.shared.sampling import (
    PALETTE,
    balanced_choice,
    balanced_int,
    configured_count,
    sample_panel_labels,
)
from trace_tasks.tasks.charts.surface_3d.shared.state import Panel3D, Surface3DDataset
from trace_tasks.tasks.registry import register_task


TASK_ID = "task_charts__surface_3d__panel_variation_label"
OBJECTIVE_CONTRACT = "panel_variation_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
DEFAULT_QUERY_ID = SINGLE_QUERY_ID
PROMPT_QUERY_KEY = "panel_variation_label"


def _build_panel_dataset(params, instance_seed):
    """Sample small-multiple 3D panels with one panel having the unique largest vertical range."""

    min_panel_count = configured_count(params, "panel_count_min", 4)
    max_panel_count = configured_count(params, "panel_count_max", 6)
    panel_count_support = tuple(count for count in (4, 6) if int(min_panel_count) <= count <= int(max_panel_count))
    if not panel_count_support:
        raise ValueError("panel variation panel-count support must include 4 or 6")
    panel_count = int(
        balanced_choice(
            panel_count_support,
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}.panel_count",
        )
    )
    time_count = balanced_int(
        low=5,
        high=7,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.time_count",
    )
    labels, label_meta = sample_panel_labels(
        int(panel_count),
        params=params,
        instance_seed=int(instance_seed),
        reserved_labels=("x-axis", "y-axis", "z-axis"),
    )
    answer_label = str(
        balanced_choice(
            labels,
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}.answer_panel",
        )
    )
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.panel_values")
    panels: list[Panel3D] = []
    ranges: dict[str, int] = {}
    for index, label in enumerate(labels):
        if str(label) == answer_label:
            low = int(rng.randint(8, 20))
            high = int(rng.randint(80, 96))
        else:
            low = int(rng.randint(22, 42))
            high = int(rng.randint(55, 74))
        values = [
            int(round(low + (high - low) * (step / max(1, int(time_count) - 1)) + rng.uniform(-5, 5)))
            for step in range(int(time_count))
        ]
        if str(label) == answer_label:
            values[0] = low
            values[-1] = high
        values = [max(0, min(100, int(value))) for value in values]
        ranges[str(label)] = int(max(values) - min(values))
        panels.append(
            Panel3D(
                panel_label=str(label),
                values=tuple(values),
                color_rgb=PALETTE[int(index) % len(PALETTE)],
            )
        )
    winner = max(ranges, key=lambda label: (int(ranges[label]), str(label)))
    if str(winner) != str(answer_label):
        raise ValueError("panel variation answer is not unique")
    dataset = Surface3DDataset(
        scene_variant=PANEL_VARIANT,
        points=(),
        surface_cells=(),
        panels=tuple(panels),
        x_axis_label="x-axis",
        y_axis_label="y-axis",
        z_axis_label="z-axis",
        x_range=(0.0, float(max(1, int(time_count) - 1))),
        y_range=(0.0, 1.0),
        z_range=(0.0, 100.0),
        x_labels=(),
        y_labels=(),
        title="3D Panel Comparison",
    )
    return dataset, answer_label, ranges, label_meta


def _build_plan(params, instance_seed, selected_branch, query_probabilities):
    """Bind panel-range semantics for the small-multiple 3D chart."""

    if str(selected_branch) != SINGLE_QUERY_ID:
        raise ValueError(f"unsupported query_id for {TASK_ID}: {selected_branch}")
    dataset, answer_label, ranges, label_meta = _build_panel_dataset(params, int(instance_seed))

    def _bind_annotation(rendered):
        return bbox_for_single_witness(rendered.panel_bboxes_px[str(answer_label)])

    return Surface3DTaskPlan(
        dataset=dataset,
        answer_gt=TypedValue(type="string", value=str(answer_label)),
        annotation_builder=_bind_annotation,
        prompt_query_key=PROMPT_QUERY_KEY,
        dynamic_slots={},
        branch_params={
            "answer_panel_label": str(answer_label),
            "ranges_by_panel": dict(ranges),
            "panel_count": len(dataset.panels),
            "time_count": len(dataset.panels[0].values) if dataset.panels else 0,
            **dict(label_meta),
        },
        relations={
            "answer_panel_label": str(answer_label),
            "query_id_probabilities": dict(query_probabilities),
        },
        witness_symbolic={
            "type": "surface_3d_panel_variation_witness",
            "panel_label": str(answer_label),
            "answer": str(answer_label),
        },
        question_format="surface_3d_panel_variation_label",
    )


class ChartsThreeDPanelVariationLabelTask:
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


register_task(ChartsThreeDPanelVariationLabelTask)


__all__ = ["ChartsThreeDPanelVariationLabelTask"]
