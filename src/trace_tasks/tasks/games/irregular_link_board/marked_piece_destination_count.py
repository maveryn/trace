"""Count empty one-step destinations for an X-marked irregular-link-board piece."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import run_irregular_link_board_lifecycle
from .shared.defaults import DEFAULTS
from .shared.sampling import sample_destination_scene
from .shared.state import SCENE_ID, SCENE_NAMESPACE, IrregularLinkBoardAxes, IrregularLinkBoardSample


TASK_ID = "task_games__irregular_link_board__marked_piece_destination_count"
QUERY_ID = "single"
PROMPT_QUERY_KEY = "marked_piece_destination_count"
SUPPORTED_QUERY_IDS = (QUERY_ID,)

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _build_destination_sample(
    rng: Any,
    axes: IrregularLinkBoardAxes,
    gen_defaults: Mapping[str, Any],
) -> IrregularLinkBoardSample:
    """Construct a board whose legal one-step destination count is fixed."""

    return sample_destination_scene(rng=rng, axes=axes, gen_defaults=gen_defaults)


@register_task
class GamesIrregularLinkBoardMarkedPieceDestinationCountTask:
    """Count empty linked destinations for one X-marked piece."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'topology', 'state_update')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any] | None = None, max_attempts: int = 100) -> TaskOutput:
        return run_irregular_link_board_lifecycle(
            task_id=TASK_ID,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            prompt_defaults=_PROMPT_DEFAULTS,
            instance_seed=int(instance_seed),
            params=dict(params or {}),
            max_attempts=int(max_attempts),
            prompt_query_key=PROMPT_QUERY_KEY,
            rule_slot_name="movement_rule_text",
            annotation_trace_key="legal_destinations",
            board_size_support_key="board_size_support",
            fallback_board_size_support=DEFAULTS.board_size_support,
            sample_builder=_build_destination_sample,
            namespace=f"{SCENE_NAMESPACE}.destination",
        )


__all__ = ["GamesIrregularLinkBoardMarkedPieceDestinationCountTask"]
