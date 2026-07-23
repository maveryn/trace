"""Contract tests for cube surface/net puzzle tasks."""

from __future__ import annotations

from trace_tasks.tasks import TASK_REGISTRY
from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.puzzles.cube_net.equivalent_net_label import (
    TASK_ID as EQUIVALENT_NET_TASK_ID,
    PuzzlesCubeNetEquivalentNetLabelTask,
)
from trace_tasks.tasks.puzzles.cube_net.marked_edge_neighbor_face_label import (
    TASK_ID as MARKED_EDGE_NEIGHBOR_TASK_ID,
    PuzzlesCubeNetMarkedEdgeNeighborFaceLabelTask,
)
from trace_tasks.tasks.puzzles.cube_net.opposite_face_label import (
    TASK_ID as OPPOSITE_FACE_TASK_ID,
    PuzzlesCubeNetOppositeFaceLabelTask,
)
from trace_tasks.tasks.puzzles.cube_net.shared.state import NET_COORDS, SCENE_ID, SIDE_OFFSETS


def _assert_bbox_in_image(bbox: list[float], image_size: tuple[int, int]) -> None:
    assert len(bbox) == 4
    assert 0 <= float(bbox[0]) < float(bbox[2]) <= image_size[0]
    assert 0 <= float(bbox[1]) < float(bbox[3]) <= image_size[1]


def test_cube_surface_tasks_are_registered() -> None:
    assert TASK_REGISTRY[OPPOSITE_FACE_TASK_ID] is PuzzlesCubeNetOppositeFaceLabelTask
    assert (
        TASK_REGISTRY[MARKED_EDGE_NEIGHBOR_TASK_ID]
        is PuzzlesCubeNetMarkedEdgeNeighborFaceLabelTask
    )
    assert TASK_REGISTRY[EQUIVALENT_NET_TASK_ID] is PuzzlesCubeNetEquivalentNetLabelTask

    for task_cls in (
        PuzzlesCubeNetOppositeFaceLabelTask,
        PuzzlesCubeNetMarkedEdgeNeighborFaceLabelTask,
        PuzzlesCubeNetEquivalentNetLabelTask,
    ):
        task = task_cls()
        assert task.domain == "puzzles"
        assert not hasattr(task, "scene_id")


def test_cube_net_face_relation_contracts() -> None:
    cases = (
        (
            PuzzlesCubeNetOppositeFaceLabelTask(),
            "opposite",
        ),
        (
            PuzzlesCubeNetMarkedEdgeNeighborFaceLabelTask(),
            "edge_neighbor",
        ),
    )
    for index, (task, relation_kind) in enumerate(cases):
        out = task.generate(2026052800 + index, params={}, max_attempts=50)
        trace = out.trace_payload
        execution = trace["execution_trace"]

        assert out.scene_id == SCENE_ID
        assert out.query_id == SINGLE_QUERY_ID
        assert out.answer_gt.type == "option_letter"
        assert out.annotation_gt.type == "bbox"
        assert trace["query_spec"]["params"]["query_id"] == SINGLE_QUERY_ID
        assert trace["render_spec"]["scene_id"] == SCENE_ID
        assert trace["render_spec"]["label_style"]["font"]["source"] == "global_font_pool"
        assert (
            trace["render_spec"]["scene_variant_style"]["semantic_policy"]
            == "non_semantic_chrome_only_no_layout_or_answer_change"
        )
        assert trace["render_spec"]["post_image_noise"]["apply_prob"] == 0.5
        assert trace["render_spec"]["net_rotation_degrees"] in {0, 90, 180, 270}
        assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]

        assert str(out.answer_gt.value) == str(execution["answer_value"])
        assert str(execution["relation_kind"]) == relation_kind
        assert trace["projected_annotation"]["type"] == "bbox"
        assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
        assert trace["projected_annotation"]["pixel_bbox"] == out.annotation_gt.value
        selected_option_bbox = trace["render_map"]["option_panel_bboxes_px"][
            f"option_{out.answer_gt.value}"
        ]
        assert out.annotation_gt.value == [
            round(float(value), 3) for value in selected_option_bbox
        ]
        option_labels = {str(option["option_label"]) for option in execution["option_specs"]}
        assert option_labels == {"A", "B", "C", "D"}
        assert str(out.answer_gt.value) in option_labels
        option_face_ids = {str(option["face_id"]) for option in execution["option_specs"]}
        assert str(execution["reference_face"]) not in option_face_ids
        assert str(execution["correct_face"]) in option_face_ids
        if relation_kind == "edge_neighbor":
            ref_x, ref_y = NET_COORDS[str(execution["reference_face"])]
            side_dx, side_dy = SIDE_OFFSETS[str(execution["marked_side"])]
            flat_neighbor_coord = (int(ref_x + side_dx), int(ref_y + side_dy))
            assert flat_neighbor_coord not in {tuple(coord) for coord in NET_COORDS.values()}
        _assert_bbox_in_image(out.annotation_gt.value, out.image.size)


def test_cube_net_equivalent_net_contracts() -> None:
    task = PuzzlesCubeNetEquivalentNetLabelTask()
    out = task.generate(2026052850, params={}, max_attempts=50)
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.scene_id == SCENE_ID
    assert out.query_id == SINGLE_QUERY_ID
    assert out.answer_gt.type == "option_letter"
    assert out.annotation_gt.type == "bbox"
    assert trace["query_spec"]["params"]["query_id"] == SINGLE_QUERY_ID
    assert trace["render_spec"]["scene_id"] == SCENE_ID
    assert trace["render_spec"]["label_style"]["font"]["source"] == "global_font_pool"
    assert (
        trace["render_spec"]["scene_variant_style"]["semantic_policy"]
        == "non_semantic_chrome_only_no_layout_or_answer_change"
    )
    assert trace["render_spec"]["post_image_noise"]["apply_prob"] == 0.5
    assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]

    assert str(out.answer_gt.value) == str(execution["answer_value"])
    assert trace["projected_annotation"]["type"] == "bbox"
    selected_option_bbox = trace["render_map"]["option_panel_bboxes_px"][
        f"option_{out.answer_gt.value}"
    ]
    assert out.annotation_gt.value == [
        round(float(value), 3) for value in selected_option_bbox
    ]
    option_specs = execution["option_specs"]
    assert {str(option["option_label"]) for option in option_specs} == {"A", "B", "C", "D"}
    reference_signature = tuple(execution["reference_signature"])
    equivalent_options = [
        str(option["option_label"])
        for option in option_specs
        if tuple(option["canonical_signature"]) == reference_signature
    ]
    assert equivalent_options == [str(out.answer_gt.value)]
    _assert_bbox_in_image(out.annotation_gt.value, out.image.size)


def test_cube_surface_scene_variants_are_visible() -> None:
    task = PuzzlesCubeNetOppositeFaceLabelTask()
    clean = task.generate(2026053010, params={"scene_variant": "clean_net"}, max_attempts=50)
    paper = task.generate(2026053010, params={"scene_variant": "paper_model"}, max_attempts=50)
    mat = task.generate(2026053010, params={"scene_variant": "game_mat"}, max_attempts=50)

    assert clean.trace_payload["render_spec"]["scene_variant_style"]["scene_variant"] == "clean_net"
    assert (
        paper.trace_payload["render_spec"]["scene_variant_style"]["scene_variant"]
        == "paper_model"
    )
    assert mat.trace_payload["render_spec"]["scene_variant_style"]["scene_variant"] == "game_mat"
    assert clean.image.tobytes() != paper.image.tobytes()
    assert clean.image.tobytes() != mat.image.tobytes()


def test_cube_surface_generation_is_deterministic() -> None:
    task = PuzzlesCubeNetEquivalentNetLabelTask()
    params = {"scene_variant": "paper_model"}
    out_a = task.generate(2026052999, params=params, max_attempts=50)
    out_b = task.generate(2026052999, params=params, max_attempts=50)

    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()
