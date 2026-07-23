"""Stable scene constants for record-table pages."""

from __future__ import annotations


DOMAIN = "pages"
SCENE = "record_table"
PROMPT_BUNDLE = "pages_record_table_v1"
PROMPT_SCENE_KEY = "record_table"
PROMPT_TASK_KEY = "record_table_count_query"

SELECTED_STATUS_FILTER = "selected_status_filter"
ENABLED_TYPE_ACTION_FILTER = "enabled_type_action_filter"
SECTION_SIZE_THRESHOLD_FILTER = "section_size_threshold_filter"

DEFAULT_ROW_COUNT_SUPPORT = (9, 10, 11, 12, 13, 14, 15)
DEFAULT_ANSWER_COUNT_SUPPORT = (2, 3, 4, 5, 6, 7)
ENABLED_ACTION_ROW_COUNT_SUPPORT = (9, 10, 11, 12)
ENABLED_ACTION_ANSWER_COUNT_SUPPORT = (2, 3, 4, 5, 6)
