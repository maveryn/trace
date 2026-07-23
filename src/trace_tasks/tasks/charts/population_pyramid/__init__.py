"""Chart scene package tasks."""

from .age_group_threshold_count import ChartsPopulationPyramidAgeGroupThresholdCountTask
from .dominant_side_count import ChartsPopulationPyramidDominantSideCountTask
from .side_gap_extremum_label import ChartsPopulationPyramidSideGapExtremumLabelTask
from .side_value_extremum_label import ChartsPopulationPyramidSideValueExtremumLabelTask

__all__ = [
    "ChartsPopulationPyramidAgeGroupThresholdCountTask",
    "ChartsPopulationPyramidDominantSideCountTask",
    "ChartsPopulationPyramidSideGapExtremumLabelTask",
    "ChartsPopulationPyramidSideValueExtremumLabelTask",
]
