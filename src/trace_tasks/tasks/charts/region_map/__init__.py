"""Chart scene package tasks."""

from .adjacent_category_count import ChartsMapAdjacentCategoryCountTask
from .adjacent_numeric_threshold_count import ChartsMapAdjacentNumericThresholdCountTask
from .adjacent_same_category_count import ChartsMapAdjacentSameCategoryCountTask
from .categorical_region_count import ChartsMapCategoricalRegionCountTask
from .group_category_region_count import ChartsMapGroupCategoryRegionCountTask
from .marker_region_extremum_label import ChartsRegionMapMarkerRegionExtremumLabelTask
from .marker_region_threshold_count import ChartsRegionMapMarkerRegionThresholdCountTask
from .named_region_set_total_value import ChartsMapNamedRegionSetTotalValueTask
from .numeric_interval_region_count import ChartsMapNumericIntervalRegionCountTask
from .numeric_threshold_region_count import ChartsMapNumericThresholdRegionCountTask

__all__ = [
    "ChartsMapAdjacentCategoryCountTask",
    "ChartsMapAdjacentNumericThresholdCountTask",
    "ChartsMapAdjacentSameCategoryCountTask",
    "ChartsMapCategoricalRegionCountTask",
    "ChartsMapGroupCategoryRegionCountTask",
    "ChartsRegionMapMarkerRegionExtremumLabelTask",
    "ChartsRegionMapMarkerRegionThresholdCountTask",
    "ChartsMapNamedRegionSetTotalValueTask",
    "ChartsMapNumericIntervalRegionCountTask",
    "ChartsMapNumericThresholdRegionCountTask",
]
