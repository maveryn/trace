from __future__ import annotations

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task

from ._lifecycle import COUNT_JSON_EXAMPLES, bind_ultimate_payload, run_ultimate_lifecycle, sample_with_retry
from .shared.sampling import sample_status_board_count, status_support
from .shared.state import STATUS_DRAWN, STATUS_NEITHER_WON, STATUS_O_WON, STATUS_X_WON


TASK_ID = "task_games__ultimate_tictactoe__small_board_status_count"
NAMESPACE = "ultimate_tictactoe.small_board_status_count"
QUERY_X_WON_COUNT = "x_won_board_count"
QUERY_O_WON_COUNT = "o_won_board_count"
QUERY_NEITHER_WON_COUNT = "neither_won_board_count"
QUERY_DRAWN_COUNT = "drawn_board_count"
SUPPORTED_QUERY_IDS = (
    QUERY_X_WON_COUNT,
    QUERY_O_WON_COUNT,
    QUERY_NEITHER_WON_COUNT,
    QUERY_DRAWN_COUNT,
)
_TARGET_STATUS_BY_BRANCH = {
    QUERY_X_WON_COUNT: STATUS_X_WON,
    QUERY_O_WON_COUNT: STATUS_O_WON,
    QUERY_NEITHER_WON_COUNT: STATUS_NEITHER_WON,
    QUERY_DRAWN_COUNT: STATUS_DRAWN,
}


def _prepare_status_payload(
    instance_seed,
    params,
    branch_key,
    branch_probabilities,
    style_variant,
    style_variant_probabilities,
    max_attempts,
):
    target_status = _TARGET_STATUS_BY_BRANCH[str(branch_key)]
    support_key, fallback_support = status_support(str(target_status))
    sample = sample_with_retry(
        public_id=TASK_ID,
        namespace=NAMESPACE,
        instance_seed=int(instance_seed),
        max_attempts=int(max_attempts),
        build_attempt=lambda rng: sample_status_board_count(
            rng,
                instance_seed=int(instance_seed),
                params=params,
                target_status=str(target_status),
                support_key=str(support_key),
                fallback_support=tuple(int(value) for value in fallback_support),
                namespace=f"{NAMESPACE}.{str(branch_key)}",
                branch_count=len(SUPPORTED_QUERY_IDS),
        ),
    )
    return bind_ultimate_payload(
        sample=sample,
        answer_gt=TypedValue(type="integer", value=int(sample.answer)),
        prompt_key=str(branch_key),
        branch_probabilities=dict(branch_probabilities),
        style_variant=str(style_variant),
        style_variant_probabilities=dict(style_variant_probabilities),
        examples=COUNT_JSON_EXAMPLES,
        semantic_params={"target_status": str(target_status)},
    )


@register_task
class GamesUltimateTicTacToeSmallBoardStatusCountTask:
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params, max_attempts):
        return run_ultimate_lifecycle(
            public_id=TASK_ID,
            supported_branches=SUPPORTED_QUERY_IDS,
            default_branch=QUERY_X_WON_COUNT,
            namespace=NAMESPACE,
            instance_seed=int(instance_seed),
            params=dict(params),
            max_attempts=int(max_attempts),
            prepare_payload=_prepare_status_payload,
        )
