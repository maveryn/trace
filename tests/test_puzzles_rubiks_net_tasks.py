"""Contract tests for Rubik-style cube-net puzzle tasks."""

from __future__ import annotations

import json

from trace_tasks.tasks import TASK_REGISTRY
from trace_tasks.tasks.puzzles.rubiks_net.post_move_face_color_count_label import (
    PuzzlesRubiksNetPostMoveFaceColorCountLabelTask,
)
from trace_tasks.tasks.puzzles.rubiks_net.post_move_sticker_color_label import (
    PuzzlesRubiksNetPostMoveStickerColorLabelTask,
)
from trace_tasks.tasks.puzzles.rubiks_net.rubiks_move_result_label import (
    DIRECT_QUERY_ID,
    INVERSE_QUERY_ID,
    PuzzlesRubiksNetMoveResultLabelTask,
)

TASKS = (
    (
        "task_puzzles__rubiks_net__post_move_sticker_color_label",
        PuzzlesRubiksNetPostMoveStickerColorLabelTask,
        {"single"},
    ),
    (
        "task_puzzles__rubiks_net__post_move_face_color_count_label",
        PuzzlesRubiksNetPostMoveFaceColorCountLabelTask,
        {"single"},
    ),
    (
        "task_puzzles__rubiks_net__rubiks_move_result_label",
        PuzzlesRubiksNetMoveResultLabelTask,
        {DIRECT_QUERY_ID, INVERSE_QUERY_ID},
    ),
)


def test_rubiks_tasks_are_registered() -> None:
    for task_id, task_cls, _queries in TASKS:
        assert TASK_REGISTRY[task_id] is task_cls
        task = task_cls()
        assert task.domain == "puzzles"
        assert not hasattr(task, "scene_id")


def test_rubiks_tasks_emit_public_contracts() -> None:
    for task_index, (_task_id, task_cls, queries) in enumerate(TASKS):
        for query_index, query_id in enumerate(sorted(queries)):
            out = task_cls().generate(
                2026052200 + (task_index * 20) + query_index,
                params={"query_id": query_id},
                max_attempts=30,
            )
            trace = out.trace_payload
            execution = trace["execution_trace"]

            json.dumps(trace)
            assert out.scene_id == "rubiks_net"
            assert out.query_id == query_id
            assert execution["query_id"] == query_id
            assert trace["query_spec"]["query_id"] == query_id
            assert trace["render_spec"]["scene_id"] == "rubiks_net"
            assert sorted(out.prompt_variants.keys()) == [
                "answer_and_annotation",
                "answer_only",
            ]
            assert out.answer_gt.type == "option_letter"
            assert out.annotation_gt.type == "bbox"
            assert str(out.answer_gt.value) == str(execution["answer_option_label"])
            assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
            assert out.image.size == (
                int(trace["render_spec"]["canvas_width"]),
                int(trace["render_spec"]["canvas_height"]),
            )
            assert execution["option_count"] == 4
            assert len(execution["option_specs"]) == 4
            assert execution["start_state"]
            assert execution["final_state"]

            bbox = out.annotation_gt.value
            assert len(bbox) == 4
            assert 0 <= float(bbox[0]) < float(bbox[2]) <= out.image.size[0]
            assert 0 <= float(bbox[1]) < float(bbox[3]) <= out.image.size[1]


def test_rubiks_generation_is_deterministic() -> None:
    task = PuzzlesRubiksNetMoveResultLabelTask()
    params = {
        "query_id": DIRECT_QUERY_ID,
        "scene_variant": "paper_net",
    }
    out_a = task.generate(2026052299, params=params, max_attempts=30)
    out_b = task.generate(2026052299, params=params, max_attempts=30)

    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert (
        out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    )
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_rubiks_scene_variants_affect_rendering() -> None:
    """Configured Rubik scene variants should be visible render treatments."""

    task = PuzzlesRubiksNetMoveResultLabelTask()
    rendered_images: list[bytes] = []
    for scene_variant in ("classic_net", "paper_net", "cool_net"):
        out = task.generate(
            2026052399,
            params={
                "query_id": DIRECT_QUERY_ID,
                "scene_variant": scene_variant,
            },
            max_attempts=30,
        )
        assert out.trace_payload["execution_trace"]["scene_variant"] == scene_variant
        rendered_images.append(out.image.tobytes())

    assert len(set(rendered_images)) == 3
