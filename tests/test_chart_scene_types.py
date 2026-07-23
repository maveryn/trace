from __future__ import annotations

import importlib.util

from trace_tasks.tasks.charts.shared import chart_scene_types


def test_chart_scene_facade_is_retired() -> None:
    retired_module = ".".join(("trace_tasks", "tasks", "charts", "shared", "chart_scene"))
    assert importlib.util.find_spec(retired_module) is None


def test_chart_scene_types_public_surface() -> None:
    assert set(chart_scene_types.__all__) == {
        "BoxPlotSpec",
        "ChartColor",
        "ChartMarkSpec",
        "ChartRenderParams",
        "HistogramBinSpec",
        "MultiSeriesChartMarkSpec",
        "RenderedChartScene",
        "SUPPORTED_CHART_SCENE_VARIANTS",
        "SUPPORTED_COMPOSITION_CHART_SCENE_VARIANTS",
        "SUPPORTED_DISTRIBUTION_CHART_SCENE_VARIANTS",
        "SUPPORTED_MULTISERIES_CHART_SCENE_VARIANTS",
        "ViolinPlotSpec",
    }


def test_chart_scene_variant_constants_are_tuples() -> None:
    assert isinstance(chart_scene_types.SUPPORTED_CHART_SCENE_VARIANTS, tuple)
    assert isinstance(chart_scene_types.SUPPORTED_MULTISERIES_CHART_SCENE_VARIANTS, tuple)
    assert isinstance(chart_scene_types.SUPPORTED_COMPOSITION_CHART_SCENE_VARIANTS, tuple)
    assert isinstance(chart_scene_types.SUPPORTED_DISTRIBUTION_CHART_SCENE_VARIANTS, tuple)
