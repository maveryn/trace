"""Behavior tests for the wallpaper reference-match task."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.tasks.icons.wallpaper_panels.same_pattern_as_reference_label import (
    IconsWallpaperPanelsSamePatternAsReferenceLabelTask,
    TASK_ID,
)
from trace_tasks.tasks.icons.wallpaper_panels.shared.rendering import (
    WALLPAPER_CANVAS_TREATMENTS,
    WALLPAPER_PANEL_CHROME_POLICY,
)
from trace_tasks.tasks.icons.shared.icon_assets import resolve_icon_pool
from tests.helpers import read_jsonl


def _extract_prompt_json_example(prompt: str) -> dict:
    marker = "Example JSON:\n"
    assert marker in str(prompt)
    return json.loads(str(prompt).split(marker, 1)[1].strip())


def _assert_sixteen_motif_icons_per_panel(trace: dict) -> None:
    motif_icons = [
        entity
        for entity in trace["scene_ir"]["entities"]
        if str(entity.get("entity_kind")) == "wallpaper_motif_icon"
    ]
    counts = Counter(str(entity["panel_label"]) for entity in motif_icons)
    panel_labels = [
        str(entity["label"])
        for entity in trace["scene_ir"]["entities"]
        if str(entity.get("entity_kind")) == "wallpaper_panel"
    ]
    assert counts == Counter({label: 16 for label in panel_labels})


def test_icons_wallpaper_reference_match_contract_matches_scene() -> None:
    task = IconsWallpaperPanelsSamePatternAsReferenceLabelTask()
    out = task.generate(
        2026060901,
        params={"answer_label": "D", "reference_wallpaper_group_id": "p1"},
        max_attempts=300,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    scene_panels = [
        entity
        for entity in trace["scene_ir"]["entities"]
        if str(entity.get("entity_kind")) == "wallpaper_panel"
    ]

    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "D"
    assert out.annotation_gt.type == "bbox_map"
    assert sorted(out.annotation_gt.value.keys()) == ["reference_panel", "selected_panel"]
    assert out.scene_id == "wallpaper_panels"
    assert out.query_id == "single"
    assert trace["scene_ir"]["scene_kind"] == "icons_wallpaper_panels_reference_match"
    assert execution["question_format"] == "select_candidate_panel_matching_reference_wallpaper_pattern"
    assert int(execution["option_count"]) == 4
    assert execution["option_labels"] == list("ABCD")
    assert str(execution["reference_wallpaper_group_id"]) == "p1"
    assert execution["visible_internal_grid"] is False
    assert trace["render_spec"]["style"]["visible_internal_grid"] is False
    assert trace["render_spec"]["style"]["wallpaper_panel_chrome_policy"] == WALLPAPER_PANEL_CHROME_POLICY
    assert trace["render_spec"]["style"]["available_canvas_treatments"] == list(WALLPAPER_CANVAS_TREATMENTS)
    assert trace["render_spec"]["style"]["icon_canvas_style"]["treatment"] in WALLPAPER_CANVAS_TREATMENTS
    assert trace["render_spec"]["panel_geometry"]["motif_lattice"] == {"rows": 4, "cols": 4, "visible_grid": False}
    assert trace["render_spec"]["panel_geometry"]["reference_panel_position"] == "above_candidate_grid"
    canvas_width, canvas_height = trace["render_spec"]["canvas_size"]
    assert int(canvas_width) * int(canvas_height) < 1_200_000

    reference_panels = [panel for panel in scene_panels if bool(panel["is_reference"])]
    candidate_panels = [panel for panel in scene_panels if str(panel["panel_role"]) == "candidate"]
    assert len(reference_panels) == 1
    assert len(candidate_panels) == 4
    assert [str(panel["label"]) for panel in candidate_panels] == list("ABCD")

    reference = reference_panels[0]
    answer = next(panel for panel in candidate_panels if str(panel["label"]) == "D")
    distractors = [panel for panel in candidate_panels if str(panel["label"]) != "D"]
    ref_bbox = [int(value) for value in reference["panel_bbox_xyxy"]]
    candidate_bboxes = [[int(value) for value in panel["panel_bbox_xyxy"]] for panel in candidate_panels]
    ref_size = (int(ref_bbox[2] - ref_bbox[0]), int(ref_bbox[3] - ref_bbox[1]))
    assert all((int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])) == ref_size for bbox in candidate_bboxes)
    assert all(int(ref_bbox[3]) < int(bbox[1]) for bbox in candidate_bboxes)
    assert str(reference["wallpaper_group_id"]) == "p1"
    assert str(answer["wallpaper_group_id"]) == "p1"
    assert all(str(panel["wallpaper_group_id"]) != "p1" for panel in distractors)
    assert len({str(panel["wallpaper_group_id"]) for panel in distractors}) == 3
    assert len({str(panel["icon_id"]) for panel in scene_panels}) == 5

    non_symmetry_pool = set(resolve_icon_pool("non_symmetry.txt"))
    assert set(execution["icon_ids_by_label"].keys()) == {"Reference", *set("ABCD")}
    assert len(set(execution["icon_ids_by_label"].values())) == 5
    assert set(execution["icon_ids_by_label"].values()).issubset(non_symmetry_pool)
    assert out.annotation_gt.value == {
        "reference_panel": list(reference["panel_bbox_xyxy"]),
        "selected_panel": list(answer["panel_bbox_xyxy"]),
    }
    assert trace["projected_annotation"]["type"] == "bbox_map"
    assert trace["projected_annotation"]["bbox_map"] == out.annotation_gt.value
    assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]
    assert "Reference" in out.prompt
    assert "wallpaper" in out.prompt
    assert "pattern" in out.prompt
    _assert_sixteen_motif_icons_per_panel(trace)


def test_icons_wallpaper_reference_match_prompt_example_matches_contract() -> None:
    task = IconsWallpaperPanelsSamePatternAsReferenceLabelTask()
    out = task.generate(2026060902, params={"answer_label": "C"}, max_attempts=300)
    answer_only = _extract_prompt_json_example(out.prompt_variants["answer_only"])
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
    assert answer_only == {"answer": "C"}
    assert list(answer_and_annotation.keys()) == ["annotation", "answer"]
    assert sorted(answer_and_annotation["annotation"].keys()) == ["reference_panel", "selected_panel"]
    assert answer_and_annotation["answer"] == "C"


def test_icons_wallpaper_reference_match_rejects_unsafe_canvas_treatment() -> None:
    task = IconsWallpaperPanelsSamePatternAsReferenceLabelTask()
    with pytest.raises(ValueError, match="shared icon canvas treatments"):
        task.generate(
            2026060912,
            params={"icon_canvas_treatments": ["plain_sheet", "unsupported_canvas"]},
            max_attempts=20,
        )


def test_icons_wallpaper_reference_match_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / TASK_ID
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name=f"build_smoke_{TASK_ID}",
        instance_version="v0",
        image_format="png",
        tasks=[BuildTaskConfig(task_id=TASK_ID, count=3, params={})],
        strict_repro=False,
        max_attempts_per_instance=300,
        sampling_seed=37,
    )
    final_path = build_dataset(config, code_hash="icons-wallpaper-reference-smoke")
    assert final_path.exists()
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 3
    assert all(record["domain"] == "icons" for record in train_records)
    assert all(record["scene_id"] == "wallpaper_panels" for record in train_records)

    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"][TASK_ID]) == 3
    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0
