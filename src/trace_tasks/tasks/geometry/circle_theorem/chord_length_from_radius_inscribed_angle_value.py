"""Chord length from radius and inscribed angle."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    split_scene_generation_rendering_prompt_defaults,
)

from ._lifecycle import run_label_keyed_number_circle_theorem_task
from .shared.measurements import (
    DEFAULT_INSCRIBED_ANGLE_SUPPORT,
    bind_chord_length_inputs,
    build_chord_length_payload,
    chord_length_query_params,
)

TASK_ID = "task_geometry__circle_theorem__chord_length_from_radius_inscribed_angle_value"
QUERY_ID = "chord_length_from_radius_and_inscribed_angle"

_SCENE_DEFAULTS = get_scene_defaults("geometry", "circle_theorem")
_GEN_DEFAULTS, _RENDER_DEFAULTS, _ = (
    split_scene_generation_rendering_prompt_defaults(_SCENE_DEFAULTS, task_id=TASK_ID)
)


def _validate_query_param(params: Mapping[str, Any]) -> None:
    requested_query = params.get("query_id")
    if requested_query is not None and str(requested_query) != QUERY_ID:
        raise ValueError(f"unsupported query_id for {TASK_ID}: {requested_query!r}")


def _central_angle_from_inscribed_angle(angle_degrees: int) -> int:
    """The chord's central angle is twice the visible inscribed angle."""

    return int(2 * int(angle_degrees))


def _resolve_binding(instance_seed: int, *, params: Mapping[str, Any]):
    """Bind radius and inscribed angle while preserving O/A/B/C annotations."""

    _validate_query_param(params)
    return bind_chord_length_inputs(
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.query",
        params=params,
        defaults=_GEN_DEFAULTS,
        angle_support_key="inscribed_angle_support",
        angle_support_fallback=DEFAULT_INSCRIBED_ANGLE_SUPPORT,
        central_angle_from_visible_angle=_central_angle_from_inscribed_angle,
        uses_inscribed_angle=True,
    )


@register_task
class GeometryCircleChordLengthFromRadiusInscribedAngleValueTask:
    """Compute a chord length from a visible radius and inscribed angle."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "geometry"
    default_dataset_enabled = True
    supported_query_ids = (QUERY_ID,)

    def generate(
        self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int
    ) -> TaskOutput:
        """Render the inscribed-angle theorem with O/A/B/C annotation keys."""

        binding = _resolve_binding(int(instance_seed), params=params)
        return run_label_keyed_number_circle_theorem_task(
            task_id=TASK_ID,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            query_id=QUERY_ID,
            answer_value=float(binding.spec.answer_value),
            query_params=chord_length_query_params(
                query_probabilities={QUERY_ID: 1.0},
                binding=binding,
            ),
            build_scene_payload=lambda rng: build_chord_length_payload(
                rng,
                spec=binding.spec,
            ),
            render_defaults=_RENDER_DEFAULTS,
            scene_kind="geometry_circle_theorem_chord_length_inscribed_angle",
            witness_type="circle_theorem_chord_length_inscribed_angle_formula",
            object_description_key="object_description_chord_length",
            answer_hint_key="answer_hint_number",
            annotation_hint_key="annotation_hint_chord_length_points",
        )
