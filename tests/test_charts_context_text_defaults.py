"""Chart-domain context-text wrapper tests."""

from __future__ import annotations

import pytest

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks import TASK_REGISTRY
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults


REPORT_PARAGRAPH_SCENE_TASKS = {
    "annotated_series": "task_charts__annotated_series__callout_endpoint_change_value",
    "area": "task_charts__area__interval_area_value",
    "candlestick": "task_charts__candlestick__range_extremum_label",
    "combo_mark": "task_charts__combo_mark__cross_mark_difference_value",
    "dashboard": "task_charts__dashboard__category_panel_condition_count",
    "dumbbell": "task_charts__dumbbell__absolute_gap_threshold_count",
    "heatmap": "task_charts__heatmap__colorbar_threshold_cell_count",
    "hexbin_density": "task_charts__hexbin_density__threshold_bin_count",
}

DENSE_CLEAN_MINIMAL_SCENE_TASKS = {
    "bar_3d": "task_charts__bar_3d__category_total_value",
    "boxplot": "task_charts__boxplot__iqr_extremum_label",
    "contour_density": "task_charts__contour_density__density_threshold_region_count",
    "curve_panels": "task_charts__curve_panels__threshold_series_count",
    "density_curve": "task_charts__density_curve__density_at_x_extremum_label",
    "error_interval": "task_charts__error_interval__reference_containment_count",
    "errorbar_series": "task_charts__errorbar_series__bound_extremum_x_label",
}


def _output(task_id: str, *, seed: int, params: dict):
    output = TASK_REGISTRY[task_id]().generate(
        int(seed),
        params=dict(params),
        max_attempts=120,
    )
    return output


def _context_layer(task_id: str, *, seed: int, params: dict) -> dict:
    output = _output(task_id, seed=seed, params=params)
    return output.trace_payload["render_spec"]["context_text_layer"]


def _rendering_defaults(scene_id: str) -> dict:
    cfg = get_scene_defaults("charts", str(scene_id))
    _generation, rendering, _prompt = split_scene_generation_rendering_prompt_defaults(
        cfg,
        task_id=f"charts.{scene_id}",
    )
    return dict(rendering)


def _overlap(left: list[float], right: list[float]) -> bool:
    return not (
        float(left[2]) <= float(right[0])
        or float(left[0]) >= float(right[2])
        or float(left[3]) <= float(right[1])
        or float(left[1]) >= float(right[3])
    )


def test_chart_context_clean_mode_records_empty_layer() -> None:
    layer = _context_layer(
        "task_charts__area__interval_area_value",
        seed=12345,
        params={"chart_context_mode": "clean"},
    )

    assert layer["enabled"] is True
    assert layer["layout_mode"] == "chart_context:clean"
    assert layer["element_count"] == 0
    assert layer["elements"] == []


def test_source_layout_chart_context_profiles_are_explicit() -> None:
    for scene_id in REPORT_PARAGRAPH_SCENE_TASKS:
        rendering = _rendering_defaults(str(scene_id))
        assert rendering["chart_context_profile"] == "report_paragraph"

    for scene_id in DENSE_CLEAN_MINIMAL_SCENE_TASKS:
        rendering = _rendering_defaults(str(scene_id))
        assert rendering["chart_context_profile"] == "dense_clean_minimal"


def test_chart_context_minimal_and_paragraph_modes_draw_context() -> None:
    minimal = _context_layer(
        "task_charts__area__interval_area_value",
        seed=12345,
        params={"chart_context_mode": "minimal"},
    )
    paragraph = _context_layer(
        "task_charts__area__interval_area_value",
        seed=12001,
        params={"chart_context_mode": "paragraph_box"},
    )

    assert minimal["layout_mode"] == "chart_context:minimal"
    assert minimal["element_count"] >= 1
    assert paragraph["layout_mode"] == "chart_context:paragraph_box"
    assert any(str(element["role"]).startswith("paragraph_box_") for element in paragraph["elements"])


@pytest.mark.parametrize("scene_id,task_id", sorted(REPORT_PARAGRAPH_SCENE_TASKS.items()))
def test_report_paragraph_profile_forced_paragraph_mode_draws_real_box(scene_id: str, task_id: str) -> None:
    layer = _context_layer(
        str(task_id),
        seed=12001,
        params={"chart_context_mode": "paragraph_box"},
    )
    roles = {str(element["role"]) for element in layer["elements"]}

    assert layer["layout_mode"] == "chart_context:paragraph_box"
    assert layer["layout_spec"].get("context_profile") == "report_paragraph"
    assert roles & {
        "paragraph_box_heading",
        "paragraph_box_body",
        "context_box_heading",
        "context_box_body",
    }, scene_id


@pytest.mark.parametrize("scene_id,task_id", sorted(DENSE_CLEAN_MINIMAL_SCENE_TASKS.items()))
def test_dense_clean_minimal_profile_default_never_uses_paragraph(scene_id: str, task_id: str) -> None:
    layer = _context_layer(str(task_id), seed=12345, params={})

    assert layer["layout_spec"].get("context_profile") == "dense_clean_minimal"
    assert layer["layout_mode"] in {"chart_context:clean", "chart_context:minimal"}
    assert layer["layout_mode"] != "chart_context:paragraph_box"


def test_dense_clean_minimal_profile_rejects_explicit_paragraph_mode() -> None:
    with pytest.raises(ValueError, match="not supported by profile"):
        _context_layer(
            "task_charts__bar_3d__category_total_value",
            seed=12345,
            params={"chart_context_mode": "paragraph_box"},
        )


def test_chart_context_wrapper_preserves_scene_specific_clean_layer() -> None:
    output = TASK_REGISTRY["task_charts__dashboard__category_panel_condition_count"]().generate(
        12345,
        params={"chart_context_mode": "clean"},
        max_attempts=120,
    )
    render_spec = output.trace_payload["render_spec"]

    assert render_spec["context_text_layer"]["layout_mode"] == "chart_context:clean"
    assert render_spec["context_text_layer"]["element_count"] == 0
    assert render_spec["context_text_layer"]["elements"] == []
    assert render_spec["chart_context_text_policy"]["domain_wrapper"] == "skipped_existing_context_text_layer"


def test_chart_context_records_and_avoids_protected_bboxes() -> None:
    output = _output(
        "task_charts__combo_mark__cross_mark_difference_value",
        seed=153456684567519,
        params={"query_id": "line_minus_primary_at_label", "chart_context_mode": "paragraph_box"},
    )
    render_map = output.trace_payload["render_map"]
    context_boxes = list(render_map["context_text_bboxes_px"].values())
    protected_boxes = list(render_map["context_protected_bboxes_px"].values())

    assert render_map["legend_bbox_px"] == render_map["context_protected_bboxes_px"]["legend"]
    assert context_boxes
    assert protected_boxes
    assert not any(_overlap(context_box, protected_box) for context_box in context_boxes for protected_box in protected_boxes)
