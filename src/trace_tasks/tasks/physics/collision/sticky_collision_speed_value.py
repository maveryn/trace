from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.fixed_query import force_query_id_params

from .shared.mechanics import rounded_speed_value
from .shared.output import (
    build_sticky_output_kwargs,
    build_sticky_support_payload,
)
from .shared.prompts import sticky_speed_prompt_slots
from .shared.rendering import prepare_resolved_sticky_scene_or_raise
from .shared.rendering import sticky_input_motion_arrow_segments
from .shared.sampling import (
    resolve_sticky_speed_axes,
)
from .shared.state import (
    INPUT_MOTION_ARROW_ENTITY_IDS,
    SCENE_ID,
    STICKY_SPEED_OPERATION,
    StickyRenderDefaults,
)


TASK_ID = "task_physics__collision__sticky_collision_speed_value"
TASK_NAMESPACE = "physics_collision_sticky_speed"
TASK_PROMPT_KEY = "sticky_collision_speed_value_query"
PROMPT_QUERY_KEY = STICKY_SPEED_OPERATION
_DEFAULTS = StickyRenderDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "physics",
    SCENE_ID,
    task_id=TASK_ID,
)


def _speed_task_params(raw_params):
    """Normalize the fixed public query branch before speed sampling."""

    task_params = force_query_id_params(raw_params or {}, query_id=SINGLE_QUERY_ID)
    task_params.pop("query_id", None)
    return task_params


def _speed_axes(instance_seed: int, task_params):
    """Resolve task-specific semantic axes for the rounded speed target."""

    return resolve_sticky_speed_axes(
        instance_seed=int(instance_seed),
        params=task_params,
        defaults=_GEN_DEFAULTS,
        namespace=TASK_NAMESPACE,
    )


def _speed_scene(instance_seed: int, max_attempts: int, task_params, axes):
    """Sample and render a sticky collision constrained by final speed."""

    return prepare_resolved_sticky_scene_or_raise(
        instance_seed=int(instance_seed),
        max_attempts=int(max_attempts),
        axes=axes,
        params=task_params,
        generation_defaults=_GEN_DEFAULTS,
        rendering_defaults=_RENDER_DEFAULTS,
        fallback=_DEFAULTS,
        namespace=TASK_NAMESPACE,
    )


def _speed_answer(scene_spec) -> TypedValue:
    """Bind the rounded final speed from the sampled execution trace."""

    return TypedValue(
        "number",
        float(
            rounded_speed_value(
                scene_spec.scenario.final_vx,
                scene_spec.scenario.final_vy,
            )
        ),
    )


def _speed_annotation(rendered) -> TypedValue:
    """Bind both input velocity vectors as speed witnesses."""

    return TypedValue("segment_set", sticky_input_motion_arrow_segments(rendered))


def _speed_support_payload(task_params, axes):
    """Attach task-specific speed support metadata to the trace."""

    payload = build_sticky_support_payload(task_params, _GEN_DEFAULTS)
    payload["annotation_key_by_entity_id"] = {}
    payload["target_speed_tenths"] = int(axes.target_speed_tenths or 0)
    payload["answer_rounding"] = "one_decimal"
    return payload


def _speed_query_params(axes):
    """Expose the sampled rounded-speed support without making it a query id."""

    return {
        "target_speed_tenths": int(axes.target_speed_tenths or 0),
        "target_answer_probabilities": dict(axes.target_answer_probabilities),
    }


@register_task
class PhysicsCollisionStickyCollisionSpeedValueTask:
    domain = "physics"
    task_id = TASK_ID
    reasoning_operations = ('aggregation', 'formula_evaluation')
    supported_query_ids = (SINGLE_QUERY_ID,)
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        task_params = _speed_task_params(params)
        axes = _speed_axes(int(instance_seed), task_params)
        scene_spec, rendered = _speed_scene(
            int(instance_seed),
            int(max_attempts),
            task_params,
            axes,
        )
        answer_gt = _speed_answer(scene_spec)
        annotation_gt = _speed_annotation(rendered)
        output_kwargs = build_sticky_output_kwargs(
            self.domain,
            _PROMPT_DEFAULTS,
            TASK_PROMPT_KEY,
            PROMPT_QUERY_KEY,
            int(instance_seed),
            SINGLE_QUERY_ID,
            {SINGLE_QUERY_ID: 1.0},
            STICKY_SPEED_OPERATION,
            axes,
            rendered,
            scene_spec,
            answer_gt,
            annotation_gt,
            "number",
            _speed_support_payload(task_params, axes),
            SCENE_ID,
            prompt_slots=sticky_speed_prompt_slots(str(scene_spec.scene_variant)),
            extra_query_params=_speed_query_params(axes),
            annotation_entity_ids=INPUT_MOTION_ARROW_ENTITY_IDS,
            witness_symbolic={
                "type": "segment_set",
                "ids": list(INPUT_MOTION_ARROW_ENTITY_IDS),
            },
        )
        return TaskOutput(**output_kwargs, query_id=SINGLE_QUERY_ID, scene_id=SCENE_ID)
