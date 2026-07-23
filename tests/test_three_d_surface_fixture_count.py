"""Tests for synthetic 3D surface-fixture repeated-element counts."""

from __future__ import annotations

import inspect
from pathlib import Path
from statistics import mean

import pytest
import trace_tasks.tasks  # noqa: F401 - registers tasks.
from trace_tasks.core.source_layout_policy import parse_public_task_id
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks import create_task
from trace_tasks.tasks.registry import TASK_REGISTRY, ensure_scene_tasks_registered
from trace_tasks.tasks.shared.named_colors import available_named_colors
from trace_tasks.tasks.three_d.shared.object_resources import object_profiles
from trace_tasks.tasks.three_d.surface_fixture.color_count_after_operations_value import TASK_ID as COLOR_OPERATIONS_TASK_ID
from trace_tasks.tasks.three_d.surface_fixture.color_frequency_option_label import (
    ABSENT_QUERY_ID as COLOR_FREQUENCY_ABSENT_QUERY_ID,
    MOST_QUERY_ID as COLOR_FREQUENCY_MOST_QUERY_ID,
    TASK_ID as COLOR_FREQUENCY_TASK_ID,
)
from trace_tasks.tasks.three_d.surface_fixture.colored_element_count import TASK_ID as COLORED_TASK_ID
from trace_tasks.tasks.three_d.surface_fixture.element_count_extremum_label import (
    HIGHEST_QUERY_ID as EXTREMUM_HIGHEST_QUERY_ID,
    LOWEST_QUERY_ID as EXTREMUM_LOWEST_QUERY_ID,
    TASK_ID as EXTREMUM_TASK_ID,
)
from trace_tasks.tasks.three_d.surface_fixture.recolor_board_match_label import TASK_ID as RECOLOR_MATCH_TASK_ID
from trace_tasks.tasks.three_d.surface_fixture.repeated_element_count import TASK_ID
from trace_tasks.tasks.three_d.surface_fixture.repeated_element_count import TASK_ID as REPEATED_TASK_ID
from trace_tasks.tasks.three_d.surface_fixture.scoped_colored_element_count import (
    COLUMN_QUERY_ID as SCOPED_COLORED_COLUMN_QUERY_ID,
    ROW_QUERY_ID as SCOPED_COLORED_ROW_QUERY_ID,
    TASK_ID as SCOPED_COLORED_TASK_ID,
)
from trace_tasks.tasks.three_d.surface_fixture.shared.metrics import COLOR_CONFUSION_EXCLUSIONS
from trace_tasks.tasks.three_d.surface_fixture.shared.rendering import bbox_from_points, quad_cell
from trace_tasks.tasks.three_d.surface_fixture.shared.state import (
    COLOR_READOUT_SCENE_VARIANTS,
    ELEMENT_TYPE_BY_SCENE_VARIANT,
    SEMANTIC_COLOR_RGB,
    SEMANTIC_COLOR_SUPPORT,
)
from trace_tasks.tasks.three_d.surface_fixture.shared.layout import VALID_LAYOUT_FAMILIES, VALID_LAYOUT_STYLES
from tests.three_d_canvas_helpers import assert_three_d_canvas_contract


SURFACE_FIXTURE_TASK_IDS = (
    REPEATED_TASK_ID,
    EXTREMUM_TASK_ID,
    COLOR_FREQUENCY_TASK_ID,
    COLORED_TASK_ID,
    COLOR_OPERATIONS_TASK_ID,
    RECOLOR_MATCH_TASK_ID,
    SCOPED_COLORED_TASK_ID,
)

COLOR_READOUT_TASK_CASES = (
    (COLORED_TASK_ID, {"query_id": "single"}),
    (COLOR_OPERATIONS_TASK_ID, {"query_id": "single"}),
    (RECOLOR_MATCH_TASK_ID, {"query_id": "single"}),
    (SCOPED_COLORED_TASK_ID, {"query_id": SCOPED_COLORED_ROW_QUERY_ID}),
    (SCOPED_COLORED_TASK_ID, {"query_id": SCOPED_COLORED_COLUMN_QUERY_ID}),
    (COLOR_FREQUENCY_TASK_ID, {"query_id": COLOR_FREQUENCY_MOST_QUERY_ID}),
    (COLOR_FREQUENCY_TASK_ID, {"query_id": COLOR_FREQUENCY_ABSENT_QUERY_ID}),
)


def _confusable_colors(color_name: str) -> set[str]:
    return set(COLOR_CONFUSION_EXCLUSIONS.get(str(color_name), ()))


def _assert_no_color_conflicts(colors, *, anchor: str | None = None) -> None:
    color_list = [str(color) for color in colors]
    if anchor is not None:
        assert not (_confusable_colors(str(anchor)) & set(color_list))
        return
    for left_index, left in enumerate(color_list):
        for right in color_list[left_index + 1 :]:
            assert right not in _confusable_colors(left)
            assert left not in _confusable_colors(right)


def _mean_rgb_for_bbox(image, bbox):
    x0, y0, x1, y1 = [int(round(float(value))) for value in bbox]
    width = int(x1 - x0)
    height = int(y1 - y0)
    crop = image.convert("RGB").crop(
        (
            x0 + width // 4,
            y0 + height // 4,
            x1 - width // 4,
            y1 - height // 4,
        )
    )
    pixels = list(crop.getdata())
    return tuple(float(mean(pixel[channel] for pixel in pixels)) for channel in range(3))


def _bbox_min_side(bbox) -> float:
    return min(float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1]))


def _assert_composite_canvas_expanded(output) -> None:
    render_spec = output.trace_payload["render_spec"]
    source_pixels = int(render_spec["scene_canvas_width"]) * int(render_spec["scene_canvas_height"])
    final_pixels = int(render_spec["final_canvas_pixels"])

    assert final_pixels > source_pixels
    assert int(output.image.width) == int(render_spec["final_canvas_width"])
    assert int(output.image.height) == int(render_spec["final_canvas_height"])


def _assert_present_cells_have_canonical_colors(cells) -> None:
    present_cells = [dict(cell) for cell in cells if bool(cell.get("present", True))]
    assert present_cells
    colors = [str(cell.get("color_name", "")) for cell in present_cells]

    assert all(color in set(SEMANTIC_COLOR_SUPPORT) for color in colors)
    assert len(set(colors)) >= 2
    for cell, color in zip(present_cells, colors):
        assert list(cell.get("fill_rgb", [])) == list(SEMANTIC_COLOR_RGB[color])


def test_surface_fixture_semantic_colors_use_canonical_palette() -> None:
    canonical = {
        str(name): (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        for name, rgb in available_named_colors()
    }

    assert dict(SEMANTIC_COLOR_RGB) == canonical
    assert tuple(SEMANTIC_COLOR_SUPPORT) == tuple(canonical)


def test_surface_fixture_color_tasks_use_readout_scene_variants() -> None:
    readout_scenes = set(COLOR_READOUT_SCENE_VARIANTS)
    excluded_scene = "pipe_rack"

    assert excluded_scene not in readout_scenes
    assert {"server_rack", "solar_panel_array", "socket_bank", "indicator_light_panel"}.issubset(readout_scenes)

    for offset, (task_id, params) in enumerate(COLOR_READOUT_TASK_CASES):
        task = create_task(task_id)
        for sample_index in range(5):
            output = task.generate(
                20261120 + offset * 100 + sample_index,
                params={**params, "post_image_noise_apply_prob": 0.0},
                max_attempts=120,
            )
            assert output.trace_payload["execution_trace"]["scene_variant"] in readout_scenes

        with pytest.raises(ValueError):
            task.generate(
                20261220 + offset,
                params={**params, "scene_variant": excluded_scene, "post_image_noise_apply_prob": 0.0},
                max_attempts=10,
            )


def test_surface_fixture_generated_color_readout_avoids_confusable_colors() -> None:
    colored = create_task(COLORED_TASK_ID).generate(
        20261240,
        params={
            "query_id": "single",
            "scene_variant": "wall_tile_panel",
            "target_color_name": "blue",
            "target_count": 4,
            "distractor_count": 8,
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=20,
    )
    colored_trace = colored.trace_payload["execution_trace"]
    colored_colors = {
        str(cell["color_name"])
        for cell in colored_trace["surface_cells"]
        if bool(cell.get("present", True)) and str(cell["color_name"]) != "blue"
    }
    _assert_no_color_conflicts(colored_colors, anchor="blue")
    assert all(bool(cell.get("semantic_color", False)) for cell in colored_trace["surface_cells"] if bool(cell.get("present", True)))

    operations = create_task(COLOR_OPERATIONS_TASK_ID).generate(
        20261241,
        params={
            "query_id": "single",
            "scene_variant": "control_panel",
            "target_color_name": "red",
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=60,
    )
    operation_colors = tuple(operations.trace_payload["execution_trace"]["initial_color_counts"].keys())
    assert "red" in set(operation_colors)
    _assert_no_color_conflicts(operation_colors)

    recolor = create_task(RECOLOR_MATCH_TASK_ID).generate(
        20261242,
        params={"query_id": "single", "scene_variant": "socket_bank", "post_image_noise_apply_prob": 0.0},
        max_attempts=80,
    )
    recolor_colors = recolor.trace_payload["execution_trace"]["active_color_names"]
    _assert_no_color_conflicts(recolor_colors)

    color_frequency = create_task(COLOR_FREQUENCY_TASK_ID).generate(
        20261243,
        params={
            "query_id": COLOR_FREQUENCY_MOST_QUERY_ID,
            "scene_variant": "control_panel",
            "answer_color_name": "yellow",
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=80,
    )
    frequency_trace = color_frequency.trace_payload["execution_trace"]
    option_colors = set(str(color) for color in frequency_trace["option_color_names"])
    assert "yellow" in option_colors
    _assert_no_color_conflicts(option_colors - {"yellow"}, anchor="yellow")


def test_surface_fixture_repeated_element_count_variants() -> None:
    task = create_task(TASK_ID)

    for index, (scene_variant, element_type) in enumerate(ELEMENT_TYPE_BY_SCENE_VARIANT.items()):
        count = 6 + (index % 10)
        output = task.generate(
            20260604 + index,
            params={
                "query_id": "single",
                "scene_variant": scene_variant,
                "target_count": count,
                "post_image_noise_apply_prob": 0.0,
            },
            max_attempts=10,
        )

        trace = output.trace_payload["execution_trace"]
        render_map = output.trace_payload["render_map"]
        target_element_ids = [str(element_id) for element_id in trace["target_element_ids"]]

        assert output.scene_id == "surface_fixture"
        assert output.query_id == "single"
        assert output.answer_gt.type == "integer"
        assert output.answer_gt.value == count
        assert output.annotation_gt.type == "bbox_set"
        assert len(output.annotation_gt.value) == count
        assert trace["scene_variant"] == scene_variant
        assert trace["target_element_type"] == element_type
        assert trace["layout_family"] in set(VALID_LAYOUT_FAMILIES)
        assert trace["layout_style"] in set(VALID_LAYOUT_STYLES)
        if scene_variant == "perforated_panel":
            assert trace["layout_family"] == "tiled_staggered"
            assert trace["layout_style"] != "panel_scatter"
        _assert_present_cells_have_canonical_colors(trace["surface_cells"])
        assert trace["solver_trace"]["color_role"] == "non_semantic_visual_variation"
        assert sum(int(count) for count in trace["solver_trace"]["visual_color_counts"].values()) == count
        expected_annotation_bboxes = [render_map["target_element_bboxes_px"][element_id] for element_id in target_element_ids]
        expected_raw_bboxes = [render_map["element_bboxes_px"][element_id] for element_id in target_element_ids]
        assert output.annotation_gt.value == expected_annotation_bboxes
        assert [render_map["target_element_raw_bboxes_px"][element_id] for element_id in target_element_ids] == expected_raw_bboxes
        assert all(_bbox_min_side(bbox) >= 24.0 for bbox in output.annotation_gt.value)
        assert output.trace_payload["projected_annotation"]["bbox_set"] == output.annotation_gt.value
        assert output.trace_payload["projected_annotation"]["pixel_bbox_set"] == output.annotation_gt.value
        assert output.trace_payload["query_spec"]["params"]["target_element_type"] == element_type
        assert output.trace_payload["query_spec"]["prompt_variant"]["scene_key"] == "surface_fixture"
        assert any(entity["entity_id"] == "surface_fixture_panel" for entity in output.trace_payload["scene_ir"]["entities"])
        assert "{target_" not in output.prompt
        assert "repeated" not in output.prompt.lower()
        assert_three_d_canvas_contract(output)


def test_surface_fixture_predicate_count_tasks() -> None:
    cases = [
        (
            COLORED_TASK_ID,
            {
                "query_id": "single",
                "scene_variant": "locker_bank",
                "target_count": 4,
                "distractor_count": 6,
                "target_color_name": "red",
                "post_image_noise_apply_prob": 0.0,
            },
        ),
        (
            SCOPED_COLORED_TASK_ID,
            {
                "query_id": SCOPED_COLORED_ROW_QUERY_ID,
                "scene_variant": "solar_panel_array",
                "target_count": 3,
                "target_color_name": "blue",
                "scope_axis": "row",
                "scope_index": 1,
                "layout_rows": 4,
                "layout_columns": 5,
                "post_image_noise_apply_prob": 0.0,
            },
        ),
    ]

    for index, (task_id, params) in enumerate(cases):
        output = create_task(task_id).generate(20260630 + index, params=params, max_attempts=20)
        trace = output.trace_payload["execution_trace"]
        target_element_ids = [str(element_id) for element_id in trace["target_element_ids"]]

        assert output.scene_id == "surface_fixture"
        assert output.answer_gt.type == "integer"
        assert output.answer_gt.value == len(target_element_ids)
        assert output.annotation_gt.type == "bbox_set"
        assert len(output.annotation_gt.value) == output.answer_gt.value
        render_map = output.trace_payload["render_map"]
        assert output.annotation_gt.value == [
            render_map["target_element_bboxes_px"][element_id] for element_id in target_element_ids
        ]
        assert [render_map["target_element_raw_bboxes_px"][element_id] for element_id in target_element_ids] == [
            render_map["element_bboxes_px"][element_id] for element_id in target_element_ids
        ]
        assert all(_bbox_min_side(bbox) >= 24.0 for bbox in output.annotation_gt.value)
        assert "{target_" not in output.prompt
        assert "{scope_" not in output.prompt


def test_surface_fixture_element_count_extremum_label_queries() -> None:
    cases = (
        (EXTREMUM_HIGHEST_QUERY_ID, max),
        (EXTREMUM_LOWEST_QUERY_ID, min),
    )

    for offset, (query_id, extremum_fn) in enumerate(cases):
        output = create_task(EXTREMUM_TASK_ID).generate(
            20260780 + offset,
            params={
                "query_id": query_id,
                "scene_variant": "socket_bank",
                "answer_label": "C",
                "option_counts": [5, 8, 13, 10] if query_id == EXTREMUM_HIGHEST_QUERY_ID else [8, 10, 4, 13],
                "post_image_noise_apply_prob": 0.0,
            },
            max_attempts=20,
        )

        trace = output.trace_payload["execution_trace"]
        render_map = output.trace_payload["render_map"]
        counts_by_label = {str(key): int(value) for key, value in trace["option_counts_by_label"].items()}
        expected_count = int(extremum_fn(counts_by_label.values()))

        assert output.scene_id == "surface_fixture"
        assert output.query_id == query_id
        assert output.answer_gt.type == "option_letter"
        assert output.answer_gt.value == "C"
        assert trace["answer_label"] == "C"
        assert trace["answer_count"] == expected_count
        assert counts_by_label["C"] == expected_count
        assert set(counts_by_label) == {"A", "B", "C", "D"}
        assert len(set(counts_by_label.values())) == 4
        for option_dataset in trace["surface_option_datasets"].values():
            _assert_present_cells_have_canonical_colors(option_dataset["surface_cells"])
            assert option_dataset["color_role"] == "non_semantic_visual_variation"
            assert sum(int(count) for count in option_dataset["visual_color_counts"].values()) == len(
                [cell for cell in option_dataset["surface_cells"] if bool(cell.get("present", True))]
            )
        assert output.annotation_gt.type == "bbox"
        assert output.annotation_gt.value == render_map["option_panel_bboxes_px"]["C"]
        assert output.trace_payload["projected_annotation"]["bbox"] == output.annotation_gt.value
        assert "{target_" not in output.prompt
        assert "{object_" not in output.prompt
        assert_three_d_canvas_contract(output)
        _assert_composite_canvas_expanded(output)


def test_surface_fixture_color_frequency_option_label_queries() -> None:
    cases = (
        (
            COLOR_FREQUENCY_MOST_QUERY_ID,
            {
                "query_id": COLOR_FREQUENCY_MOST_QUERY_ID,
                "scene_variant": "control_panel",
                "answer_label": "D",
                "option_color_names": ["red", "blue", "green", "yellow", "magenta", "cyan"],
                "color_counts_by_name": {
                    "red": 2,
                    "blue": 3,
                    "green": 1,
                    "yellow": 6,
                    "magenta": 4,
                    "cyan": 2,
                },
                "post_image_noise_apply_prob": 0.0,
            },
        ),
        (
            COLOR_FREQUENCY_ABSENT_QUERY_ID,
            {
                "query_id": COLOR_FREQUENCY_ABSENT_QUERY_ID,
                "scene_variant": "socket_bank",
                "answer_label": "B",
                "option_color_names": ["red", "blue", "green", "yellow", "magenta", "cyan"],
                "color_counts_by_name": {
                    "red": 2,
                    "blue": 0,
                    "green": 3,
                    "yellow": 4,
                    "magenta": 2,
                    "cyan": 1,
                },
                "post_image_noise_apply_prob": 0.0,
            },
        ),
    )

    for offset, (query_id, params) in enumerate(cases):
        output = create_task(COLOR_FREQUENCY_TASK_ID).generate(
            20260930 + offset,
            params=params,
            max_attempts=20,
        )

        trace = output.trace_payload["execution_trace"]
        render_map = output.trace_payload["render_map"]
        counts = {str(color): int(count) for color, count in trace["option_color_counts"].items()}
        records = {str(record["label"]): dict(record) for record in trace["option_records"]}
        answer_label = str(params["answer_label"])
        answer_color = str(records[answer_label]["color_name"])

        assert output.scene_id == "surface_fixture"
        assert output.query_id == query_id
        assert output.answer_gt.type == "option_letter"
        assert output.answer_gt.value == answer_label
        assert trace["answer_label"] == answer_label
        assert trace["answer_color_name"] == answer_color
        assert trace["answer_color_count"] == counts[answer_color]
        assert set(records) == {"A", "B", "C", "D", "E", "F"}
        assert all(str(record["color_name"]) in set(SEMANTIC_COLOR_SUPPORT) for record in records.values())
        assert set(render_map["option_text_bboxes_px"]) == {"A", "B", "C", "D", "E", "F"}
        assert all("option_text_bbox_px" in record for record in records.values())
        assert output.annotation_gt.type == "bbox"
        assert output.annotation_gt.value == render_map["option_bboxes_px"][answer_label]
        assert output.trace_payload["projected_annotation"]["bbox"] == output.annotation_gt.value
        if query_id == COLOR_FREQUENCY_MOST_QUERY_ID:
            max_count = max(counts.values())
            assert counts[answer_color] == max_count
            assert sum(1 for count in counts.values() if count == max_count) == 1
            assert all(count > 0 for count in counts.values())
            assert len(trace["target_element_ids"]) == max_count
        else:
            assert counts[answer_color] == 0
            assert sum(1 for count in counts.values() if count == 0) == 1
            assert len(trace["target_element_ids"]) == 0
        assert "{target_" not in output.prompt
        assert "{scope_" not in output.prompt
        assert "color option" not in output.prompt.lower()
        assert "swatch" not in output.prompt.lower()
        assert_three_d_canvas_contract(output)


def test_surface_fixture_scoped_color_query_ids_bind_scope_axis() -> None:
    cases = (
        (SCOPED_COLORED_ROW_QUERY_ID, "row", 1),
        (SCOPED_COLORED_COLUMN_QUERY_ID, "column", 2),
    )

    for offset, (query_id, expected_axis, scope_index) in enumerate(cases):
        output = create_task(SCOPED_COLORED_TASK_ID).generate(
            20260740 + offset,
            params={
                "query_id": query_id,
                "scene_variant": "wall_tile_panel",
                "target_count": 2,
                "target_color_name": "green",
                "scope_index": scope_index,
                "layout_rows": 4,
                "layout_columns": 5,
                "post_image_noise_apply_prob": 0.0,
            },
            max_attempts=20,
        )

        trace = output.trace_payload["execution_trace"]
        assert output.query_id == query_id
        assert trace["query_id"] == query_id
        assert trace["scope_axis"] == expected_axis
        assert str(trace["scope_phrase"]).startswith(expected_axis)
        assert output.answer_gt.value == len(trace["target_element_ids"])


def test_surface_fixture_color_count_after_operations_tracks_final_count() -> None:
    output = create_task(COLOR_OPERATIONS_TASK_ID).generate(
        20260811,
        params={
            "query_id": "single",
            "scene_variant": "control_panel",
            "target_color_name": "red",
            "active_color_names": ["red", "yellow", "green"],
            "initial_color_counts": {
                "red": 3,
                "yellow": 4,
                "green": 3,
            },
            "operations": [
                {"action": "add", "color_name": "red", "count": 2},
                {"action": "remove", "color_name": "yellow", "count": 1},
                {"action": "add", "color_name": "green", "count": 1},
            ],
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=20,
    )

    trace = output.trace_payload["execution_trace"]
    target_element_ids = [str(element_id) for element_id in trace["target_element_ids"]]

    assert output.scene_id == "surface_fixture"
    assert output.query_id == "single"
    assert output.answer_gt.type == "integer"
    assert output.answer_gt.value == 5
    assert trace["initial_target_count"] == 3
    assert trace["initial_color_counts"]["red"] == 3
    assert trace["final_color_counts"]["red"] == 5
    assert len(trace["operations"]) == 3
    assert len(target_element_ids) == 3
    assert output.annotation_gt.type == "bbox_set"
    assert len(output.annotation_gt.value) == 3
    render_map = output.trace_payload["render_map"]
    assert output.annotation_gt.value == [
        render_map["target_element_bboxes_px"][element_id] for element_id in target_element_ids
    ]
    assert [render_map["target_element_raw_bboxes_px"][element_id] for element_id in target_element_ids] == [
        render_map["element_bboxes_px"][element_id] for element_id in target_element_ids
    ]
    assert all(_bbox_min_side(bbox) >= 24.0 for bbox in output.annotation_gt.value)
    assert "{operation_" not in output.prompt
    assert "{target_" not in output.prompt
    assert "after" in output.prompt.lower()


def test_surface_fixture_recolor_board_match_selects_matching_option() -> None:
    output = create_task(RECOLOR_MATCH_TASK_ID).generate(
        20260841,
        params={
            "query_id": "single",
            "scene_variant": "control_panel",
            "answer_label": "C",
            "active_color_names": ["red", "blue", "green"],
            "source_color_name": "red",
            "destination_color_name": "blue",
            "initial_color_counts": {
                "red": 3,
                "blue": 2,
                "green": 4,
            },
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=20,
    )

    trace = output.trace_payload["execution_trace"]
    render_map = output.trace_payload["render_map"]
    option_counts = {
        str(label): {str(color): int(count) for color, count in counts.items()}
        for label, counts in trace["option_color_counts_by_label"].items()
    }

    assert output.scene_id == "surface_fixture"
    assert output.query_id == "single"
    assert output.answer_gt.type == "option_letter"
    assert output.answer_gt.value == "C"
    assert trace["source_color_name"] == "red"
    assert trace["destination_color_name"] == "blue"
    assert trace["initial_color_counts"] == {"red": 3, "blue": 2, "green": 4}
    assert trace["final_color_counts"] == {"red": 0, "blue": 5, "green": 4}
    assert option_counts["C"] == trace["final_color_counts"]
    assert trace["option_color_by_flat_index_by_label"]["C"] == trace["final_color_by_flat_index"]
    assert len(
        {
            tuple(sorted(color_by_index.items()))
            for color_by_index in trace["option_color_by_flat_index_by_label"].values()
        }
    ) == 4
    original_cells = {
        int(cell["flat_index"]): cell
        for cell in trace["surface_original_dataset"]["surface_cells"]
        if cell["present"]
    }
    for option_dataset in trace["surface_option_datasets"].values():
        option_cells = {
            int(cell["flat_index"]): cell
            for cell in option_dataset["surface_cells"]
            if cell["present"]
        }
        assert set(option_cells) == set(original_cells)
        for index, original_cell in original_cells.items():
            option_cell = option_cells[index]
            assert option_cell["u0"] == original_cell["u0"]
            assert option_cell["u1"] == original_cell["u1"]
            assert option_cell["v0"] == original_cell["v0"]
            assert option_cell["v1"] == original_cell["v1"]
    assert output.annotation_gt.type == "bbox"
    assert output.annotation_gt.value == render_map["option_panel_bboxes_px"]["C"]
    assert output.trace_payload["projected_annotation"]["bbox"] == output.annotation_gt.value
    assert_three_d_canvas_contract(output)
    _assert_composite_canvas_expanded(output)
    assert "{recolor_" not in output.prompt
    assert "{source_" not in output.prompt
    assert "{destination_" not in output.prompt
    assert "rearranged" not in output.prompt.lower()
    assert "positions may change" not in output.prompt.lower()
    assert "color counts" not in output.prompt.lower()


def test_surface_fixture_recolor_drive_bay_uses_visible_fill_rgb() -> None:
    output = create_task(RECOLOR_MATCH_TASK_ID).generate(
        158141830484107,
        params={
            "query_id": "single",
            "scene_variant": "server_rack",
            "active_color_names": ["brown", "green", "blue"],
            "source_color_name": "brown",
            "destination_color_name": "green",
            "initial_color_counts": {
                "brown": 3,
                "green": 2,
                "blue": 4,
            },
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=80,
    )

    trace = output.trace_payload["execution_trace"]
    render_map = output.trace_payload["render_map"]
    answer_label = str(trace["answer_label"])

    assert trace["scene_variant"] == "server_rack"
    assert trace["target_element_type"] == "drive_bay"
    assert trace["source_color_name"] == "brown"
    assert trace["destination_color_name"] == "green"
    assert not any("state" in cell for cell in trace["surface_original_dataset"]["surface_cells"])
    assert not any(
        "state" in cell
        for dataset in trace["surface_option_datasets"].values()
        for cell in dataset["surface_cells"]
    )

    source_cell = next(
        cell
        for cell in trace["surface_original_dataset"]["surface_cells"]
        if cell["present"] and cell["color_name"] == "brown"
    )
    element_id = str(source_cell["element_id"])
    original_brown = _mean_rgb_for_bbox(output.image, render_map["element_bboxes_px"][f"original:{element_id}"])
    recolored_green = _mean_rgb_for_bbox(output.image, render_map["element_bboxes_px"][f"{answer_label}:{element_id}"])

    assert original_brown[0] > original_brown[2] + 30.0
    assert original_brown[1] > original_brown[2] + 20.0
    assert recolored_green[1] > recolored_green[0] + 50.0
    assert recolored_green[1] > recolored_green[2] + 50.0


def test_surface_fixture_drawer_pull_bbox_covers_visible_face() -> None:
    output = create_task(REPEATED_TASK_ID).generate(
        20260699,
        params={
            "query_id": "single",
            "scene_variant": "drawer_pull_panel",
            "target_count": 8,
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=20,
    )

    trace = output.trace_payload["execution_trace"]
    render_map = output.trace_payload["render_map"]
    fixture_entity = next(
        entity
        for entity in output.trace_payload["scene_ir"]["entities"]
        if entity["entity_id"] == "surface_fixture_panel"
    )
    quad = fixture_entity["attrs"]["surface_screen_corners_px"]

    assert trace["scene_variant"] == "drawer_pull_panel"
    assert trace["target_element_type"] == "drawer_pull"
    assert output.annotation_gt.type == "bbox_set"

    for cell in trace["surface_cells"]:
        if not cell["present"]:
            continue
        element_id = str(cell["element_id"])
        cell_bbox = bbox_from_points(
            quad_cell(
                quad,
                float(cell["u0"]),
                float(cell["v0"]),
                float(cell["u1"]),
                float(cell["v1"]),
            )
        )
        element_bbox = render_map["element_bboxes_px"][element_id]
        cell_height = float(cell_bbox[3]) - float(cell_bbox[1])
        element_height = float(element_bbox[3]) - float(element_bbox[1])

        assert element_height >= cell_height * 0.45


def test_surface_fixture_task_registered_in_three_d_taxonomy() -> None:
    ensure_scene_tasks_registered("three_d", "surface_fixture")

    for task_id in SURFACE_FIXTURE_TASK_IDS:
        task = create_task(task_id)
        taxonomy = resolve_task_taxonomy(task_id)
        parts = parse_public_task_id(task_id)
        expected_source = (
            Path(__file__).resolve().parents[1]
            / "trace"
            / "tasks"
            / parts.domain
            / parts.scene_id
            / f"{parts.objective_contract}.py"
        ).resolve()

        assert task_id in TASK_REGISTRY
        assert not hasattr(task, "scene_id")
        assert Path(inspect.getsourcefile(task.__class__) or "").resolve() == expected_source
        assert taxonomy.domain == "three_d"
        assert taxonomy.scene_id == "surface_fixture"
        assert not taxonomy.source_scene_id


def test_surface_fixture_resource_profiles_match_scene_variants() -> None:
    profile_variants = {profile.object_type for profile in object_profiles(source_scene="surface_fixture")}

    assert profile_variants == set(ELEMENT_TYPE_BY_SCENE_VARIANT)
