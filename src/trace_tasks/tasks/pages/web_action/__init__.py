"""Pages web-action scene-package tasks."""

from .action_target_label import (
    DEFAULT_QUERY_ID as ACTION_TARGET_DEFAULT_QUERY_ID,
)
from .action_target_label import (
    SUPPORTED_QUERY_IDS as ACTION_TARGET_SUPPORTED_QUERY_IDS,
)
from .action_target_label import TASK_ID as ACTION_TARGET_TASK_ID
from .action_target_label import PagesWebActionActionTargetLabelTask
from .guide_code_target_count import (
    DEFAULT_QUERY_ID as GUIDE_CODE_COUNT_DEFAULT_QUERY_ID,
)
from .guide_code_target_count import (
    SUPPORTED_QUERY_IDS as GUIDE_CODE_COUNT_SUPPORTED_QUERY_IDS,
)
from .guide_code_target_count import TASK_ID as GUIDE_CODE_COUNT_TASK_ID
from .guide_code_target_count import PagesWebActionGuideCodeTargetCountTask


__all__ = [
    "ACTION_TARGET_DEFAULT_QUERY_ID",
    "ACTION_TARGET_SUPPORTED_QUERY_IDS",
    "ACTION_TARGET_TASK_ID",
    "GUIDE_CODE_COUNT_DEFAULT_QUERY_ID",
    "GUIDE_CODE_COUNT_SUPPORTED_QUERY_IDS",
    "GUIDE_CODE_COUNT_TASK_ID",
    "PagesWebActionActionTargetLabelTask",
    "PagesWebActionGuideCodeTargetCountTask",
]
