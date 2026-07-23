"""Chart scene package tasks."""

from .cross_panel_delta_extremum_label import (
    ChartsScientificCrossPanelDeltaExtremumLabelTask,
)
from .cross_panel_threshold_earliest_label import (
    ChartsScientificCrossPanelThresholdEarliestLabelTask,
)
from .curve_at_x_extremum_label import ChartsScientificCurveAtXExtremumLabelTask
from .curve_intersection_count import ChartsScientificCurveIntersectionCountTask
from .endpoint_rank_panel_label import ChartsScientificEndpointRankPanelLabelTask
from .earliest_maximum_panel_label import ChartsScientificEarliestMaximumPanelLabelTask
from .global_value_extremum_panel_label import (
    ChartsScientificGlobalValueExtremumPanelLabelTask,
)
from .panel_curve_threshold_crossing_count import (
    ChartsScientificPanelCurveThresholdCrossingCountTask,
)
from .panel_spread_extremum_label import ChartsScientificPanelSpreadExtremumLabelTask
from .threshold_series_count import ChartsScientificThresholdSeriesCountTask

__all__ = [
    "ChartsScientificCrossPanelDeltaExtremumLabelTask",
    "ChartsScientificCrossPanelThresholdEarliestLabelTask",
    "ChartsScientificCurveAtXExtremumLabelTask",
    "ChartsScientificCurveIntersectionCountTask",
    "ChartsScientificEndpointRankPanelLabelTask",
    "ChartsScientificEarliestMaximumPanelLabelTask",
    "ChartsScientificGlobalValueExtremumPanelLabelTask",
    "ChartsScientificPanelCurveThresholdCrossingCountTask",
    "ChartsScientificPanelSpreadExtremumLabelTask",
    "ChartsScientificThresholdSeriesCountTask",
]
