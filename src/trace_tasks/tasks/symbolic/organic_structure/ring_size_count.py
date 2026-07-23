"""Count pentagonal or hexagonal rings in an organic structure."""

from __future__ import annotations

from functools import partial
from typing import Any, Dict

from ....core.query_ids import SINGLE_QUERY_ID
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import OrganicLifecycleSpec, prepare_ring_size_count_plan, run_organic_count_lifecycle
from .shared.state import SCENE_ID, SUPPORTED_ORGANIC_RING_SIZES


DOMAIN = "symbolic"
TASK_ID = "task_symbolic__organic_structure__ring_size_count"
TASK_PROMPT_KEY = "organic_structure_ring_size_count"
INTERNAL_QUERY_KEY = "ring_size_count"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
TARGET_RING_SIZES = SUPPORTED_ORGANIC_RING_SIZES

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
    task_id=TASK_ID,
)
_LIFECYCLE = OrganicLifecycleSpec(
    TASK_ID,
    DOMAIN,
    INTERNAL_QUERY_KEY,
    SUPPORTED_QUERY_IDS,
    _GEN_DEFAULTS,
    _RENDER_DEFAULTS,
    _PROMPT_DEFAULTS,
    TASK_PROMPT_KEY,
    INTERNAL_QUERY_KEY,
)
_BUILD_PLAN = partial(
    prepare_ring_size_count_plan,
    owner_key=TASK_ID,
    gen_defaults=_GEN_DEFAULTS,
    target_sizes=TARGET_RING_SIZES,
)


def _prepare_ring_size_plan(**kwargs: Any) -> Any:
    return _BUILD_PLAN(**kwargs)


@register_task
class SymbolicRingSizeCountTask:
    """Count pentagonal or hexagonal rings in an organic-structure notation panel."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'topology')
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        return run_organic_count_lifecycle(
            _LIFECYCLE,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            build_plan=_prepare_ring_size_plan,
        )


RING_SIZE_COUNT_TASK_ID = TASK_ID
