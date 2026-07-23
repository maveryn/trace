"""Find the exact roll needed for one Ludo token to finish."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import LudoObjectivePlan, LudoSingleQueryTaskBase, build_ludo_attempt_result, run_ludo_registered_task
from .shared.prompts import make_ludo_prompt_slots_from_keys
from .shared.annotations import point_ludo_render_map_annotation
from .shared.rendering import make_ludo_render_state
from .shared.rules import roll_sequence_for_total
from .shared.sampling import resolve_ludo_integer_axis, sample_other_token_coords
from .shared.state import DEFAULTS, HOME_LANES, PLAYER_COLORS, SCENE_NAMESPACE, Coord, LudoSceneAxes


TASK_ID = "task_games__ludo_board__winning_roll_value"
_PROMPT_SLOTS = make_ludo_prompt_slots_from_keys(
    keys=("winning_roll_value", "exact_finish_rule_text", "answer_hint_winning_roll_value", "annotation_hint_winning_roll_value"),
    example_annotation=[115, 215],
    example_answer=4,
)


@dataclass(frozen=True)
class _WinningRollSample:
    """Task-owned symbolic sample for exact-finish Ludo roll reasoning."""

    token_coords: Mapping[str, Coord]
    query_color: str
    winning_roll: int
    answer: int
    construction_mode: str


def _construct_winning_roll_sample(*, rng: Any, axes: LudoSceneAxes, winning_roll: int) -> _WinningRollSample:
    """Place the named token in its home lane so exactly one roll reaches finish."""

    roll = int(winning_roll)
    if roll < 1 or roll > 5:
        raise ValueError("winning roll must be in 1..5")
    lane = HOME_LANES[str(axes.query_color)]
    query_coord = tuple(lane[5 - roll])
    occupied = {query_coord}
    other_colors = [color for color in PLAYER_COLORS if color != str(axes.query_color)]
    token_coords = {str(axes.query_color): query_coord}
    token_coords.update(sample_other_token_coords(rng=rng, occupied=occupied, colors=other_colors))
    return _WinningRollSample(
        token_coords=dict(token_coords),
        query_color=str(axes.query_color),
        winning_roll=roll,
        answer=roll,
        construction_mode="target_conditioned_exact_finish_roll",
    )


def _prepare_winning_roll_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    selected_query_id: str,
    _branch_probabilities: Mapping[str, float],
    _axes: LudoSceneAxes,
    gen_defaults: Mapping[str, Any],
) -> LudoObjectivePlan:
    """Resolve the exact-finish target roll and bind winning-roll construction."""

    roll_axis = resolve_ludo_integer_axis(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
        support_key="winning_roll_support",
        explicit_key="winning_roll",
        fallback_support=DEFAULTS.winning_roll_support,
        namespace=f"{SCENE_NAMESPACE}.winning_roll.target",
        balanced_flag_key="balanced_winning_roll_sampling",
    )

    def construct_attempt(rng: Any, axes: LudoSceneAxes):
        """Bind an exact-finish token position to its roll answer and witnesses."""

        sample = _construct_winning_roll_sample(
            rng=rng,
            axes=axes,
            winning_roll=int(roll_axis.value),
        )
        return build_ludo_attempt_result(
            answer_type="integer",
            answer_value=int(sample.answer),
            render_state=make_ludo_render_state(
                style_variant=str(axes.style_variant),
                token_coords=sample.token_coords,
                query_color=str(sample.query_color),
            ),
            build_annotation=lambda rendered: point_ludo_render_map_annotation(
                rendered=rendered,
                point_source=("token_centers_px", f"token_{sample.query_color}"),
                point_entity_id=f"token_{sample.query_color}",
            ),
            selected_query_id=str(selected_query_id),
            axes=axes,
            construction_mode=str(sample.construction_mode),
            token_coords=sample.token_coords,
            query_color=str(sample.query_color),
            target_color=None,
            extra_execution_trace={
                "winning_roll": int(sample.winning_roll),
                "roll_sequence": list(roll_sequence_for_total(int(sample.winning_roll))),
                "answer": int(sample.answer),
            },
            extra_query_params={
                "winning_roll": int(roll_axis.value),
                "winning_roll_support": [int(value) for value in roll_axis.support],
                "winning_roll_probabilities": dict(roll_axis.probabilities),
            },
            relations_extra={"target_color": None},
        )

    return LudoObjectivePlan(
        prompt_slots=_PROMPT_SLOTS,
        attempt_namespace=f"{SCENE_NAMESPACE}.winning_roll",
        construct_attempt=construct_attempt,
    )


@register_task
class GamesLudoBoardWinningRollValueTask(LudoSingleQueryTaskBase):
    """Find the exact roll needed for one Ludo token to finish."""

    task_id = TASK_ID
    reasoning_operations = ('topology', 'state_update', 'formula_evaluation')
    _namespace = f"{SCENE_NAMESPACE}.winning_roll"
    _prepare_objective = staticmethod(_prepare_winning_roll_objective)

    def generate(self, instance_seed: int, *, params: dict[str, Any] | None = None, max_attempts: int = 100) -> TaskOutput:
        return run_ludo_registered_task(self, instance_seed, params=params, max_attempts=max_attempts)


__all__ = ["GamesLudoBoardWinningRollValueTask"]
