"""Pages instruction-panel scene-package tasks."""

from .shared_control_for_step_set_label import (
    PROMPT_QUERY_KEY as SHARED_CONTROL_PROMPT_QUERY_KEY,
)
from .shared_control_for_step_set_label import (
    SUPPORTED_QUERY_IDS as SHARED_CONTROL_SUPPORTED_QUERY_IDS,
)
from .shared_control_for_step_set_label import TASK_ID as SHARED_CONTROL_TASK_ID
from .shared_control_for_step_set_label import PagesInstructionPanelSharedControlForStepSetLabelTask
from .step_for_control_pair_label import (
    PROMPT_QUERY_KEY as CONTROL_PAIR_PROMPT_QUERY_KEY,
)
from .step_for_control_pair_label import (
    SUPPORTED_QUERY_IDS as CONTROL_PAIR_SUPPORTED_QUERY_IDS,
)
from .step_for_control_pair_label import TASK_ID as CONTROL_PAIR_TASK_ID
from .step_for_control_pair_label import PagesInstructionPanelStepForControlPairLabelTask


SCENE_VARIANTS = _lifecycle.SCENE_VARIANTS


__all__ = [
    "CONTROL_PAIR_PROMPT_QUERY_KEY",
    "CONTROL_PAIR_SUPPORTED_QUERY_IDS",
    "CONTROL_PAIR_TASK_ID",
    "SCENE_VARIANTS",
    "SHARED_CONTROL_PROMPT_QUERY_KEY",
    "SHARED_CONTROL_SUPPORTED_QUERY_IDS",
    "SHARED_CONTROL_TASK_ID",
    "PagesInstructionPanelSharedControlForStepSetLabelTask",
    "PagesInstructionPanelStepForControlPairLabelTask",
]
