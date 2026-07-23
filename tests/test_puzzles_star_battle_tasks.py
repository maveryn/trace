"""Contracts for Star Battle source-layout puzzle tasks."""

from __future__ import annotations

from typing import Any, Mapping

from PIL import Image
import pytest

from trace_tasks.core.prompts import load_prompt_bundle
from trace_tasks.core.prompts.schema import REQUIRED_PROMPT_VARIANTS
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.puzzles.star_battle.remaining_valid_cell_count import (
    SUPPORTED_QUERY_IDS as REMAINING_COUNT_QUERY_IDS,
    PuzzlesStarBattleRemainingValidCellCountTask,
)
from trace_tasks.tasks.puzzles.star_battle.shared.rendering import render_star_battle_scene
from trace_tasks.tasks.puzzles.star_battle.shared.rules import cell_key
from trace_tasks.tasks.puzzles.star_battle.shared.state import SCENE_ID, StarBattleDataset, StarBattleRenderParams
from trace_tasks.tasks.puzzles.star_battle.valid_cell_anywhere_label import (
    SUPPORTED_QUERY_IDS as VALID_CELL_ANYWHERE_QUERY_IDS,
    PuzzlesStarBattleValidCellAnywhereLabelTask,
)
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


LABEL_TASK_CLASSES = (
    PuzzlesStarBattleValidCellAnywhereLabelTask,
)
TASK_CLASSES = (
    *LABEL_TASK_CLASSES,
    PuzzlesStarBattleRemainingValidCellCountTask,
)


def test_star_battle_scene_defaults_loaded() -> None:
    cfg = get_scene_defaults("puzzles", "star_battle")
    prompt_shared = cfg["prompt"]["shared"]
    assert str(prompt_shared["bundle_id"]).strip() == "puzzles_star_battle_v1"

    generation_defaults, rendering_defaults, prompt_defaults = (
        split_generation_rendering_prompt_defaults(
            cfg,
            task_id="task_puzzles__star_battle__remaining_valid_cell_count",
        )
    )
    assert str(prompt_defaults["scene_key"]).strip() == "star_battle"
    assert str(prompt_defaults["task_key"]).strip() == "star_battle_remaining_count_query"
    assert "query_id_weights" not in generation_defaults
    assert int(generation_defaults["target_count_min"]) == 1
    assert int(generation_defaults["target_count_max"]) == 6
    assert int(rendering_defaults["canvas_width"]) == 1080


def test_star_battle_prompt_bundle_supports_scene_package_variants() -> None:
    bundle = load_prompt_bundle("puzzles", "star_battle", "puzzles_star_battle_v1")
    assert bundle.schema_version == "v1"
    assert len(bundle.task_templates["star_battle_valid_cell_query"]) == REQUIRED_PROMPT_VARIANTS
    assert len(bundle.task_templates["star_battle_remaining_count_query"]) == REQUIRED_PROMPT_VARIANTS
    assert len(bundle.query_templates["valid_cell_anywhere_label"]) == REQUIRED_PROMPT_VARIANTS
    assert (
        len(bundle.query_templates["remaining_valid_cells_in_marked_row_count"])
        == REQUIRED_PROMPT_VARIANTS
    )
    assert (
        len(bundle.query_templates["remaining_valid_cells_in_marked_column_count"])
        == REQUIRED_PROMPT_VARIANTS
    )
    assert list(bundle.required_slots_by_key["scene:star_battle"]) == ["object_description"]


def _internal_query_id(output: Any) -> str:
    payload = output.trace_payload
    assert isinstance(payload, Mapping)
    execution = payload["execution_trace"]
    if "internal_query_id" in execution:
        return str(execution["internal_query_id"])
    return str(execution["query_id"])


@pytest.mark.parametrize("task_cls", LABEL_TASK_CLASSES)
def test_star_battle_label_tasks_emit_scalar_bbox_contract(task_cls) -> None:
    task = task_cls()
    out = task.generate(83101, params={}, max_attempts=80)

    assert out.scene_id == SCENE_ID
    assert out.answer_gt.type == "option_letter"
    assert out.annotation_gt.type == "bbox"
    assert len(out.annotation_gt.value) == 4
    assert "annotation" in out.prompt_variants["answer_and_annotation"].lower()
    assert '"answer"' in out.prompt_variants["answer_only"]

    trace = out.trace_payload
    assert trace["query_spec"]["params"]["scene_id"] == SCENE_ID
    assert trace["projected_annotation"]["type"] == "bbox"
    assert trace["annotation_gt"]["type"] == "bbox"


def test_star_battle_remaining_count_emits_bbox_set_contract() -> None:
    task = PuzzlesStarBattleRemainingValidCellCountTask()
    out = task.generate(83102, params={}, max_attempts=80)

    assert out.scene_id == SCENE_ID
    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "bbox_set"
    assert len(out.annotation_gt.value) == int(out.answer_gt.value)
    assert out.trace_payload["projected_annotation"]["type"] == "bbox_set"


@pytest.mark.parametrize(
    ("query_id", "expected_text"),
    [
        ("remaining_valid_cells_in_marked_row_count", "from the top"),
        ("remaining_valid_cells_in_marked_column_count", "from the left"),
    ],
)
def test_star_battle_remaining_count_prompt_names_highlighted_axis(query_id: str, expected_text: str) -> None:
    task = PuzzlesStarBattleRemainingValidCellCountTask()
    out = task.generate(83103, params={"query_id": query_id}, max_attempts=80)

    prompt = out.prompt.lower()
    assert "highlighted" in prompt
    assert expected_text in prompt
    if query_id.endswith("_row_count"):
        row_index = int(out.trace_payload["execution_trace"]["marked_row_index"]) + 1
        assert f"row {row_index}" in prompt
    else:
        col_index = int(out.trace_payload["execution_trace"]["marked_col_index"]) + 1
        assert f"column {col_index}" in prompt


@pytest.mark.parametrize(
    "mark_kwargs",
    [
        {"marked_region_index": 0},
        {"marked_row_index": 0},
        {"marked_col_index": 0},
    ],
)
def test_star_battle_marked_scope_preserves_region_fill(mark_kwargs: Mapping[str, int]) -> None:
    base_fill = (246, 217, 215)
    dataset = StarBattleDataset(
        size=2,
        grid_size_range=(2, 2),
        solution_stars=(),
        visible_stars=(),
        region_grid=((0, 0), (1, 1)),
        regions={"0": ((0, 0), (0, 1)), "1": ((1, 0), (1, 1))},
        legal_cells=(),
        scope_cells=((0, 0), (0, 1)),
        scoped_legal_cells=(),
        candidate_specs=(),
        answer_value=0,
        answer_type="integer",
        option_count=0,
        target_answer_support=(),
        **mark_kwargs,
    )
    render_params = StarBattleRenderParams(
        canvas_width=220,
        canvas_height=220,
        cell_size_px=48,
        panel_padding_px=12,
        panel_corner_radius_px=8,
        grid_line_width_px=1,
        heavy_line_width_px=3,
        clue_size_px=24,
        candidate_font_size_px=16,
        clue_font_size_px=14,
        text_color_rgb=(28, 32, 38),
        text_stroke_rgb=(255, 255, 255),
        style_overrides={
            "region_palette": (base_fill, (219, 234, 249)),
            "highlight_fill": (255, 241, 142),
            "accent": (45, 91, 176),
            "accent_backdrop": (255, 255, 255),
        },
        unit_size_jitter={},
    )
    rendered = render_star_battle_scene(
        Image.new("RGB", (220, 220), (255, 255, 255)),
        dataset=dataset,
        scene_variant="star_battle_classic",
        render_params=render_params,
    )
    x0, y0, x1, y1 = rendered.cell_bbox_map[cell_key((0, 0))]
    center_pixel = rendered.image.getpixel((int((x0 + x1) / 2), int((y0 + y1) / 2)))

    assert center_pixel == base_fill


@pytest.mark.parametrize(
    ("task_cls", "query_id"),
    [
        (PuzzlesStarBattleValidCellAnywhereLabelTask, VALID_CELL_ANYWHERE_QUERY_IDS[0]),
    ],
)
def test_star_battle_valid_cell_task_has_one_correct_labeled_cell(task_cls, query_id: str) -> None:
    task = task_cls()
    params = {}
    out = task.generate(83111, params=params, max_attempts=80)
    trace = out.trace_payload["execution_trace"]

    assert _internal_query_id(out) == query_id
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value in {"A", "B", "C", "D", "E", "F", "G", "H"}
    correct = [spec for spec in trace["candidate_specs"] if spec["is_correct"]]
    legal = [spec for spec in trace["candidate_specs"] if spec["is_legal"]]
    assert len(correct) == 1
    assert len(legal) == 1
    assert correct[0]["label"] == out.answer_gt.value


@pytest.mark.parametrize("query_id", REMAINING_COUNT_QUERY_IDS)
def test_star_battle_remaining_count_matches_scoped_legal_cells(query_id: str) -> None:
    task = PuzzlesStarBattleRemainingValidCellCountTask()
    out = task.generate(83121, params={"query_id": query_id}, max_attempts=80)
    trace = out.trace_payload["execution_trace"]

    assert _internal_query_id(out) == query_id
    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == len(trace["scoped_legal_cells"])
    assert int(out.answer_gt.value) == len(out.annotation_gt.value)
    target_min, target_max = trace["target_count_range"]
    assert int(target_min) <= int(out.answer_gt.value) <= int(target_max)


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_star_battle_tasks_are_deterministic(task_cls) -> None:
    task = task_cls()
    params = {}
    out_a = task.generate(83131, params=params, max_attempts=80)
    out_b = task.generate(83131, params=params, max_attempts=80)

    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt == out_b.answer_gt
    assert out_a.annotation_gt == out_b.annotation_gt
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
