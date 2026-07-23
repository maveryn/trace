"""Chart scene package tasks."""

from .category_total_extremum_label import ChartsMultiseriesCategoryTotalExtremumLabelTask
from .pair_equality_label import ChartsMultiseriesPairEqualityLabelTask
from .ranked_change_extremum_label import ChartsMultiseriesRankedChangeExtremumTask
from .ranked_pair_ratio_extremum_label import ChartsMultiseriesRankedPairRatioExtremumTask
from .ranked_series_share_extremum_label import ChartsMultiseriesRankedSeriesShareExtremumTask
from .series_rank_at_category_label import ChartsMultiseriesSeriesRankAtCategoryLabelTask

__all__ = [
    "ChartsMultiseriesCategoryTotalExtremumLabelTask",
    "ChartsMultiseriesPairEqualityLabelTask",
    "ChartsMultiseriesRankedChangeExtremumTask",
    "ChartsMultiseriesRankedPairRatioExtremumTask",
    "ChartsMultiseriesRankedSeriesShareExtremumTask",
    "ChartsMultiseriesSeriesRankAtCategoryLabelTask",
]
