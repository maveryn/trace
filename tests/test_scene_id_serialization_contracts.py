"""Public task-layout and scene-id ABI regression tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from trace_tasks.core.identity import compute_instance_id
from trace_tasks.core.reward_contracts import resolve_reward_contract
from trace_tasks.core.rlvr_export import build_rlvr_row
from trace_tasks.core.source_layout_policy import uses_current_source_layout
from trace_tasks.core.types import (
    CurriculumIndex,
    ImageRecord,
    TraceRef,
    TrainInstance,
    TypedValue,
)
from trace_tasks.core.validation import _validate_schema


TASK_ID = "task_geometry__graph_paper__polygon_area_value"


def _train_record() -> dict[str, object]:
    return TrainInstance(
        instance_version="v0",
        instance_id="dummy_instance",
        instance_seed=123,
        domain="geometry",
        task=TASK_ID,
        scene_id="graph_paper",
        query_id="single",
        prompt="What is the polygon's area?",
        images=[
            ImageRecord(
                image_id="dummy_image",
                format="png",
                image_hash="abc",
                path="images/dummy.png",
            )
        ],
        answer_gt=TypedValue(type="integer", value=1),
        annotation_gt=TypedValue(type="point_set", value=[[8.0, 8.0]]),
        reward_contract=resolve_reward_contract(
            answer_type="integer", annotation_type="point_set"
        ),
        trace_ref=TraceRef(
            shard_id="trace_shard_0001.jsonl.zst",
            line_index=0,
            trace_record_hash="abc",
        ),
        versions={"renderer_version": "v0"},
    ).to_dict()


def test_source_layout_detection_uses_real_public_modules() -> None:
    assert uses_current_source_layout(TASK_ID, domain="geometry")
    assert not uses_current_source_layout(
        "task_dummy__missing_scene__missing_task", domain="dummy"
    )
    assert not uses_current_source_layout(TASK_ID, domain="wrong")


def test_train_and_curriculum_records_always_emit_scene_id() -> None:
    train_record = _train_record()
    curriculum_record = CurriculumIndex(
        instance_id="dummy_instance",
        domain="geometry",
        task=TASK_ID,
        scene_id="graph_paper",
        query_id="single",
    ).to_dict()

    assert train_record["scene_id"] == "graph_paper"
    assert curriculum_record["scene_id"] == "graph_paper"


def test_scene_id_is_required_and_non_empty() -> None:
    missing = _train_record()
    missing.pop("scene_id")
    missing_errors = _validate_schema(missing)
    assert any(
        error.error_code == "schema_missing_field"
        and error.context.get("field_path") == "scene_id"
        for error in missing_errors
    )

    empty = _train_record()
    empty["scene_id"] = ""
    empty_errors = _validate_schema(empty)
    assert any(
        error.error_code == "schema_invalid_value"
        and error.context.get("field_path") == "scene_id"
        for error in empty_errors
    )


def test_scene_id_does_not_change_v0_instance_identity() -> None:
    record = _train_record()
    without_scene = dict(record)
    without_scene.pop("scene_id")
    other_scene = dict(record)
    other_scene["scene_id"] = "renamed_scene"

    assert compute_instance_id(record) == compute_instance_id(without_scene)
    assert compute_instance_id(record) == compute_instance_id(other_scene)


def test_rlvr_export_rejects_records_without_scene_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires scene_id"):
        build_rlvr_row(
            {"instance_id": "missing_scene"},
            dataset_root=tmp_path,
            output_parent=tmp_path,
        )
