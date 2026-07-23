"""Behavior tests for radial Sankey chart tasks."""

from __future__ import annotations

import pytest

from tests.helpers import extract_prompt_json_example
from trace_tasks.tasks.charts.radial_sankey.dominant_endpoint_label import (
    SUPPORTED_QUERY_IDS as DOMINANT_QUERY_IDS,
    ChartsRadialSankeyDominantEndpointLabelTask,
)
from trace_tasks.tasks.charts.radial_sankey.shared.state import SUPPORTED_SCENE_VARIANTS
from trace_tasks.tasks.charts.radial_sankey.transfer_total_value import (
    SUPPORTED_QUERY_IDS as TRANSFER_QUERY_IDS,
    ChartsRadialSankeyTransferTotalValueTask,
)
from trace_tasks.tasks.registry import list_default_task_ids


def _assert_bbox_inside_canvas(bbox: list[float], *, width: int, height: int) -> None:
    assert len(bbox) == 4
    x0, y0, x1, y1 = [float(value) for value in bbox]
    assert 0 <= x0 < x1 <= width
    assert 0 <= y0 < y1 <= height


def _expected_transfer_total(execution: dict) -> int:
    links_by_id = {str(key): dict(value) for key, value in execution["links_by_id"].items()}
    return sum(int(links_by_id[str(ref)]["value"]) for ref in execution["query_link_ids"])


def _expected_dominant_label(execution: dict) -> str:
    link_details = [dict(link) for link in execution["link_details"]]
    winner = max(link_details, key=lambda item: int(item["value"]))
    if str(execution["query_id"]) == "largest_target_for_source":
        return str(winner["target_label"])
    if str(execution["query_id"]) == "largest_source_for_target":
        return str(winner["source_label"])
    raise AssertionError(f"unsupported radial Sankey query: {execution['query_id']}")


@pytest.mark.parametrize("query_id", TRANSFER_QUERY_IDS)
def test_charts_radial_sankey_transfer_total_matches_contract(query_id: str) -> None:
    task = ChartsRadialSankeyTransferTotalValueTask()
    out = task.generate(136000 + len(query_id), params={"query_id": query_id}, max_attempts=100)
    trace = out.trace_payload
    execution = trace["execution_trace"]
    render = trace["render_spec"]
    render_map = trace["render_map"]
    assert task.task_id in list_default_task_ids()
    assert out.scene_id == "radial_sankey"
    assert out.query_id == query_id
    assert str(execution["query_id"]) == query_id
    assert str(execution["scene_variant"]) in SUPPORTED_SCENE_VARIANTS
    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "bbox_set"
    assert out.image.size == (int(render["canvas_width"]), int(render["canvas_height"]))
    assert 4 <= int(execution["source_count"]) <= 4
    assert 4 <= int(execution["target_count"]) <= 4
    assert 5 <= int(execution["link_count"]) <= 7
    assert out.answer_gt.value == _expected_transfer_total(execution)
    assert trace["projected_annotation"]["type"] == "bbox_set"
    assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
    refs = [str(value) for value in execution["annotation_link_ids"]]
    expected_boxes = [render_map["link_label_bboxes_px"][ref] for ref in refs]
    assert out.annotation_gt.value == expected_boxes
    for bbox in out.annotation_gt.value:
        _assert_bbox_inside_canvas([float(value) for value in bbox], width=int(render["canvas_width"]), height=int(render["canvas_height"]))
    assert str(render["font_assets"]["font_asset_version"])
    assert str(render["font_assets"]["chart_font_family"])


@pytest.mark.parametrize("query_id", DOMINANT_QUERY_IDS)
def test_charts_radial_sankey_dominant_endpoint_matches_contract(query_id: str) -> None:
    task = ChartsRadialSankeyDominantEndpointLabelTask()
    out = task.generate(136300 + len(query_id), params={"query_id": query_id}, max_attempts=100)
    trace = out.trace_payload
    execution = trace["execution_trace"]
    render = trace["render_spec"]
    render_map = trace["render_map"]
    assert task.task_id in list_default_task_ids()
    assert out.scene_id == "radial_sankey"
    assert out.query_id == query_id
    assert str(execution["query_id"]) == query_id
    assert str(execution["scene_variant"]) in SUPPORTED_SCENE_VARIANTS
    assert out.answer_gt.type == "string"
    assert out.annotation_gt.type == "bbox"
    assert out.image.size == (int(render["canvas_width"]), int(render["canvas_height"]))
    assert out.answer_gt.value == _expected_dominant_label(execution)
    assert trace["projected_annotation"]["type"] == "bbox"
    assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert not execution["annotation_link_ids"]
    node_refs = [str(value) for value in execution["annotation_node_ids"]]
    assert len(node_refs) == 1
    expected_node_box = render_map["node_bboxes_px"][node_refs[0]]
    assert out.annotation_gt.value == expected_node_box
    _assert_bbox_inside_canvas([float(value) for value in expected_node_box], width=int(render["canvas_width"]), height=int(render["canvas_height"]))
    assert str(render["font_assets"]["font_asset_version"])
    assert str(render["font_assets"]["chart_font_family"])


def test_charts_radial_sankey_prompt_examples_match_contract() -> None:
    transfer = ChartsRadialSankeyTransferTotalValueTask().generate(136800, params={}, max_attempts=100)
    transfer_json = extract_prompt_json_example(transfer.prompt_variants["answer_and_annotation"])
    assert isinstance(transfer_json["answer"], int)
    assert isinstance(transfer_json["annotation"], list)
    assert transfer_json["annotation"] and all(len(box) == 4 for box in transfer_json["annotation"])

    dominant = ChartsRadialSankeyDominantEndpointLabelTask().generate(136900, params={}, max_attempts=100)
    dominant_json = extract_prompt_json_example(dominant.prompt_variants["answer_and_annotation"])
    assert isinstance(dominant_json["answer"], str)
    assert isinstance(dominant_json["annotation"], list)
    assert len(dominant_json["annotation"]) == 4


def test_charts_radial_sankey_light_theme_uses_subtle_node_fills() -> None:
    params = {
        "query_id": "largest_source_for_target",
        "information_scene_treatments": ["poster_explainer"],
        "information_scene_palettes": ["metro_bright"],
        "information_scene_chrome_modes": ["none"],
    }
    out = ChartsRadialSankeyDominantEndpointLabelTask().generate(506287278820451, params=params, max_attempts=100)
    render = out.trace_payload["render_spec"]
    roles = render["information_scene_style"]["roles_rgb"]

    assert render["source_node_fill_rgb"] == roles["surface_alt"]
    assert render["target_node_fill_rgb"] == roles["panel_fill"]
    assert render["source_node_fill_rgb"] != roles["header"]


def test_charts_radial_sankey_is_deterministic() -> None:
    params = {"query_id": "largest_target_for_source"}
    first = ChartsRadialSankeyDominantEndpointLabelTask().generate(137000, params=params, max_attempts=100)
    second = ChartsRadialSankeyDominantEndpointLabelTask().generate(137000, params=params, max_attempts=100)
    assert first.prompt == second.prompt
    assert first.answer_gt.to_dict() == second.answer_gt.to_dict()
    assert first.annotation_gt.to_dict() == second.annotation_gt.to_dict()
    assert first.trace_payload["execution_trace"] == second.trace_payload["execution_trace"]
