"""Line-result count task for rule-override board scenes."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import ObjectiveRuleOverridePlan, run_rule_override_lifecycle
from .shared.prompts import rule_text_from_prompt_defaults
from .shared.sampling import resolve_line_axes, sample_line_result_scene
from .shared.state import LOSS_RESULT, SCENE_ID, SCENE_NAMESPACE, WIN_RESULT


TASK_ID = "task_games__rule_override_board__line_result_count"
LINE_WIN_QUERY_ID = "line_override_win_count"
LINE_LOSS_QUERY_ID = "line_override_loss_count"
SUPPORTED_QUERY_IDS = (LINE_WIN_QUERY_ID, LINE_LOSS_QUERY_ID)
RULE_TEXT_KEY = "line_rule_text"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _target_result_for_query(query_id: str) -> str:
    """Map line-count query branches to the counted board result."""

    if str(query_id) == LINE_WIN_QUERY_ID:
        return WIN_RESULT
    if str(query_id) == LINE_LOSS_QUERY_ID:
        return LOSS_RESULT
    raise ValueError(f"unsupported rule-override line query: {query_id}")


def _prepare_line_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    _query_probabilities: Mapping[str, float],
    selected_query_id: str,
) -> ObjectiveRuleOverridePlan:
    """Bind the selected line query to its result target and sampler."""

    axes = resolve_line_axes(instance_seed=int(instance_seed), params=task_params, gen_defaults=_GEN_DEFAULTS)
    rule_text = rule_text_from_prompt_defaults(prompt_defaults=_PROMPT_DEFAULTS, rule_text_key=RULE_TEXT_KEY)
    target_result = _target_result_for_query(str(selected_query_id))
    return ObjectiveRuleOverridePlan(
        axes=axes,
        attempt_namespace=f"{SCENE_NAMESPACE}.line_result_count",
        prompt_query_key=str(selected_query_id),
        query_params={"target_result": str(target_result)},
        construct_attempt=lambda rng: sample_line_result_scene(
            rng=rng,
            axes=axes,
            target_result=str(target_result),
            rule_text=str(rule_text),
        ),
    )


@register_task
class GamesRuleOverrideLineResultCountTask:
    """Count wins or losses on anti-line mini-boards."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict[str, Any] | None = None, max_attempts: int = 100):
        """Generate one line-rule scene and bind counted mini-board bboxes."""

        return run_rule_override_lifecycle(
            task_id=TASK_ID,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=LINE_WIN_QUERY_ID,
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            prompt_defaults=_PROMPT_DEFAULTS,
            instance_seed=int(instance_seed),
            params=dict(params or {}),
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_line_objective,
        )
