"""Scene-package tasks for mixed infographic pages."""

from .module_condition_item_count import (
    QUERY_ID as CONDITION_COUNT_QUERY_ID,
    TASK_ID as MIXED_INFOGRAPHIC_CONDITION_COUNT_TASK_ID,
    PagesMixedInfographicModuleConditionItemCountTask,
)
from .module_field_ranked_item_label import (
    QUERY_ID as FIELD_RANKED_QUERY_ID,
    TASK_ID as MIXED_INFOGRAPHIC_FIELD_RANKED_TASK_ID,
    PagesMixedInfographicModuleFieldRankedItemLabelTask,
)
from .module_field_total_value import (
    QUERY_ID as FIELD_TOTAL_QUERY_ID,
    TASK_ID as MIXED_INFOGRAPHIC_FIELD_TOTAL_TASK_ID,
    PagesMixedInfographicModuleFieldTotalValueTask,
)
from .module_field_value_label import (
    QUERY_ID,
    TASK_ID as MIXED_INFOGRAPHIC_MODULE_FIELD_VALUE_TASK_ID,
    TASK_ID as MIXED_INFOGRAPHIC_TASK_ID,
    PagesMixedInfographicModuleFieldValueLabelTask,
)
from .module_two_field_condition_item_label import (
    QUERY_ID as TWO_FIELD_CONDITION_QUERY_ID,
    TASK_ID as MIXED_INFOGRAPHIC_TWO_FIELD_CONDITION_TASK_ID,
    PagesMixedInfographicModuleTwoFieldConditionItemLabelTask,
)
from .page_field_extremum_module_label import (
    QUERY_ID as PAGE_FIELD_EXTREMUM_QUERY_ID,
    TASK_ID as MIXED_INFOGRAPHIC_PAGE_FIELD_EXTREMUM_TASK_ID,
    PagesMixedInfographicPageFieldExtremumModuleLabelTask,
)
from .shared.defaults import SCENE_VARIANTS
from .shared.layout import NATIVE_LAYOUT_MODES
from .two_module_field_total_comparison_module_label import (
    QUERY_ID as TWO_MODULE_TOTAL_COMPARISON_QUERY_ID,
    TASK_ID as MIXED_INFOGRAPHIC_TWO_MODULE_TOTAL_COMPARISON_TASK_ID,
    PagesMixedInfographicTwoModuleFieldTotalComparisonModuleLabelTask,
)


__all__ = [
    "CONDITION_COUNT_QUERY_ID",
    "FIELD_RANKED_QUERY_ID",
    "FIELD_TOTAL_QUERY_ID",
    "MIXED_INFOGRAPHIC_CONDITION_COUNT_TASK_ID",
    "MIXED_INFOGRAPHIC_FIELD_RANKED_TASK_ID",
    "MIXED_INFOGRAPHIC_FIELD_TOTAL_TASK_ID",
    "MIXED_INFOGRAPHIC_MODULE_FIELD_VALUE_TASK_ID",
    "MIXED_INFOGRAPHIC_PAGE_FIELD_EXTREMUM_TASK_ID",
    "MIXED_INFOGRAPHIC_TASK_ID",
    "MIXED_INFOGRAPHIC_TWO_FIELD_CONDITION_TASK_ID",
    "MIXED_INFOGRAPHIC_TWO_MODULE_TOTAL_COMPARISON_TASK_ID",
    "NATIVE_LAYOUT_MODES",
    "PAGE_FIELD_EXTREMUM_QUERY_ID",
    "QUERY_ID",
    "SCENE_VARIANTS",
    "TWO_FIELD_CONDITION_QUERY_ID",
    "TWO_MODULE_TOTAL_COMPARISON_QUERY_ID",
    "PagesMixedInfographicModuleConditionItemCountTask",
    "PagesMixedInfographicModuleFieldRankedItemLabelTask",
    "PagesMixedInfographicModuleFieldTotalValueTask",
    "PagesMixedInfographicModuleFieldValueLabelTask",
    "PagesMixedInfographicModuleTwoFieldConditionItemLabelTask",
    "PagesMixedInfographicPageFieldExtremumModuleLabelTask",
    "PagesMixedInfographicTwoModuleFieldTotalComparisonModuleLabelTask",
]
