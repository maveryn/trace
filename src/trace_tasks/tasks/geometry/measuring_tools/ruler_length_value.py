"""Read a marked length using a visible ruler."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.registry import register_task

from ._lifecycle import run_measuring_public_entry
from .shared.rendering import render_length_measurement
from .shared.sampling import build_ruler_length_plan, select_index
from .shared.state import LengthMeasurementPlan

TASK_ID = "task_geometry__measuring_tools__ruler_length_value"
SUPPORTED_QUERY_IDS: tuple[str, ...] = ("single",)
MEASUREMENT_KIND = "ruler_length_reading"
PROMPT_TASK_KEY = "ruler_length_value"
OBJECT_DESCRIPTION = "a geometric figure with a ruler placed alongside the marked length"
ANNOTATION_TYPE = "segment"
ANNOTATION_KEYS = ("measure_start", "measure_end")
SHAPE_OPTIONS = ("circle", "triangle", "parallelogram", "trapezoid")


def _select_shape_kind(instance_seed: int, params: Mapping[str, Any]) -> str:
    """Select the internal carrier shape for the ruler readout."""

    explicit_shape = params.get("shape_kind")
    if explicit_shape is not None:
        shape_kind = str(explicit_shape)
        if shape_kind not in SHAPE_OPTIONS:
            raise ValueError(f"shape_kind={shape_kind!r} is not supported")
        return shape_kind
    index = select_index(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.shape_kind",
        count=len(SHAPE_OPTIONS),
    )
    return SHAPE_OPTIONS[int(index)]


def _build_plan(
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> LengthMeasurementPlan:
    """Bind merged ruler-length semantics to one carrier shape."""

    return build_ruler_length_plan(
        params=params,
        instance_seed=int(instance_seed),
        gen_defaults=gen_defaults,
        measurement_kind=MEASUREMENT_KIND,
        shape_kind=_select_shape_kind(int(instance_seed), params),
        answer_namespace=f"{TASK_ID}.target_length",
        offset_namespace=f"{TASK_ID}.ruler_start_cm",
    )


@register_task
class GeometryMeasuringToolsRulerLengthValueTask:
    """Task-owned ruler length readout objective."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "geometry"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    build_plan = staticmethod(_build_plan)
    render_measurement = staticmethod(render_length_measurement)
    prompt_task_key = PROMPT_TASK_KEY
    object_description = OBJECT_DESCRIPTION
    annotation_type = ANNOTATION_TYPE
    annotation_keys = ANNOTATION_KEYS

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        return run_measuring_public_entry(
            self,
            int(instance_seed),
            params=params,
            max_attempts=max_attempts,
        )


__all__ = ["GeometryMeasuringToolsRulerLengthValueTask"]
