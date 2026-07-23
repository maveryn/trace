"""Chart scene package tasks."""

from .category_panel_condition_count import ChartsDashboardCategoryPanelConditionCountTask
from .category_extremum_panel_label import ChartsDashboardCategoryExtremumPanelLabelTask
from .category_total_extremum_label import ChartsDashboardCategoryTotalExtremumLabelTask
from .global_value_extremum_category_label import ChartsDashboardGlobalValueExtremumCategoryLabelTask
from .panel_value_range_extremum_label import ChartsDashboardPanelValueRangeExtremumLabelTask
from .panel_value_range_value import ChartsDashboardPanelValueRangeValueTask
from .panel_total_extremum_label import ChartsDashboardPanelTotalExtremumLabelTask
from .source_rank_target_value import ChartsDashboardSourceRankTargetValueTask
from .statement_option_selection_label import ChartsDashboardStatementOptionSelectionLabelTask

__all__ = [
    "ChartsDashboardCategoryExtremumPanelLabelTask",
    "ChartsDashboardCategoryPanelConditionCountTask",
    "ChartsDashboardCategoryTotalExtremumLabelTask",
    "ChartsDashboardGlobalValueExtremumCategoryLabelTask",
    "ChartsDashboardPanelValueRangeExtremumLabelTask",
    "ChartsDashboardPanelValueRangeValueTask",
    "ChartsDashboardPanelTotalExtremumLabelTask",
    "ChartsDashboardSourceRankTargetValueTask",
    "ChartsDashboardStatementOptionSelectionLabelTask",
]
