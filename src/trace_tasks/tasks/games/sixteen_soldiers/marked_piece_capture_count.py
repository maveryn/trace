from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import ObjectiveSixteenSoldiersPlan, run_sixteen_soldiers_lifecycle
from .shared.annotations import sixteen_soldiers_capture_piece_annotation
from .shared.rules import piece_to_entity_id, validate_capture_sample
from .shared.sampling import resolve_sixteen_soldiers_target_axis, sample_marked_capture_scene
from .shared.state import SCENE_ID, SCENE_NAMESPACE


TASK_ID = "task_games__sixteen_soldiers__marked_piece_capture_count"
PROMPT_QUERY_KEY = "marked_piece_capture_count"
SUPPORTED_QUERY_IDS = (DEFAULT_QUERY_ID,)
TARGET_SUPPORT = (0, 1, 2, 3, 4)

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _prepare_capture_objective(
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> ObjectiveSixteenSoldiersPlan:
    target_axis = resolve_sixteen_soldiers_target_axis(
        int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key="marked_piece_capture_count_support",
        fallback_support=TARGET_SUPPORT,
        namespace=f"{SCENE_NAMESPACE}.capture.target_answer",
    )
    return ObjectiveSixteenSoldiersPlan(
        attempt_namespace=f"{SCENE_NAMESPACE}.capture",
        prompt_query_key=PROMPT_QUERY_KEY,
        rule_slot_name="capture_rule_text",
        annotation_kind="capturable_opponent_piece",
        target_axis=target_axis,
        construct_attempt=lambda rng, axes: sample_marked_capture_scene(
            rng=rng,
            axes=axes,
            target_axis=target_axis,
        ),
        annotation_projector=sixteen_soldiers_capture_piece_annotation,
        annotation_entity_ids=lambda sample: tuple(piece_to_entity_id(point_id) for point_id in sample.annotation_point_ids),
        validate_sample=validate_capture_sample,
    )


@register_task
class GamesSixteenSoldiersMarkedPieceCaptureCountTask:
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'topology', 'state_update')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict[str, Any] | None = None, max_attempts: int = 100) -> TaskOutput:
        return run_sixteen_soldiers_lifecycle(
            task_id=TASK_ID,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=DEFAULT_QUERY_ID,
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            prompt_defaults=_PROMPT_DEFAULTS,
            instance_seed=int(instance_seed),
            params=dict(params or {}),
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_capture_objective,
            namespace=f"{SCENE_NAMESPACE}.capture",
        )
