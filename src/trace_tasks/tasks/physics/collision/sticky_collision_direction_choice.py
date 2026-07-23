from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.fixed_query import force_query_id_params

from .shared.output import (
    build_sticky_output_kwargs,
    build_sticky_support_payload,
)
from .shared.prompts import sticky_direction_prompt_slots
from .shared.rendering import prepare_resolved_sticky_scene_or_raise
from .shared.rendering import sticky_input_motion_arrow_segments
from .shared.sampling import (
    resolve_sticky_direction_axes,
)
from .shared.state import (
    INPUT_MOTION_ARROW_ENTITY_IDS,
    SCENE_ID,
    STICKY_DIRECTION_OPERATION,
    StickyRenderDefaults,
)


TASK_ID = "task_physics__collision__sticky_collision_direction_choice"
TASK_NAMESPACE = "physics_collision_sticky_direction"
TASK_PROMPT_KEY = "sticky_collision_direction_choice_query"
PROMPT_QUERY_KEY = STICKY_DIRECTION_OPERATION
_DEFAULTS = StickyRenderDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "physics",
    SCENE_ID,
    task_id=TASK_ID,
)
@register_task
class PhysicsCollisionStickyCollisionDirectionChoiceTask:
    domain = "physics"
    task_id = TASK_ID
    reasoning_operations = ('aggregation', 'formula_evaluation')
    supported_query_ids = (SINGLE_QUERY_ID,)
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        # Direction uses both visible input velocity arrows as the grounding witness.
        task_params = force_query_id_params(params or {}, query_id=SINGLE_QUERY_ID)
        task_params.pop("query_id", None)
        axes = resolve_sticky_direction_axes(
            instance_seed=int(instance_seed),
            params=task_params,
            defaults=_GEN_DEFAULTS,
            namespace=TASK_NAMESPACE,
        )
        scene_spec, rendered = prepare_resolved_sticky_scene_or_raise(
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            axes=axes,
            params=task_params,
            generation_defaults=_GEN_DEFAULTS,
            rendering_defaults=_RENDER_DEFAULTS,
            fallback=_DEFAULTS,
            namespace=TASK_NAMESPACE,
        )
        answer_gt = TypedValue("option_letter", str(scene_spec.correct_option_letter))
        annotation_gt = TypedValue("segment_set", sticky_input_motion_arrow_segments(rendered))
        output_kwargs = build_sticky_output_kwargs(
            self.domain,
            _PROMPT_DEFAULTS,
            TASK_PROMPT_KEY,
            PROMPT_QUERY_KEY,
            int(instance_seed),
            SINGLE_QUERY_ID,
            {SINGLE_QUERY_ID: 1.0},
            STICKY_DIRECTION_OPERATION,
            axes,
            rendered,
            scene_spec,
            answer_gt,
            annotation_gt,
            "option_letter",
            {
                **build_sticky_support_payload(task_params, _GEN_DEFAULTS),
                "annotation_key_by_entity_id": {},
            },
            SCENE_ID,
            prompt_slots=sticky_direction_prompt_slots(str(scene_spec.scene_variant)),
            annotation_entity_ids=INPUT_MOTION_ARROW_ENTITY_IDS,
            witness_symbolic={
                "type": "segment_set",
                "ids": list(INPUT_MOTION_ARROW_ENTITY_IDS),
            },
        )
        return TaskOutput(**output_kwargs, query_id=SINGLE_QUERY_ID, scene_id=SCENE_ID)
