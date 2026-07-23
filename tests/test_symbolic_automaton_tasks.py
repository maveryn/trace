"""Contract tests for automaton puzzle tasks."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks import TASK_REGISTRY
from trace_tasks.tasks.symbolic.agent_automaton.future_grid_label import (
    SCENE_ID as AGENT_SCENE_ID,
    SUPPORTED_QUERY_IDS as AGENT_FUTURE_GRID_QUERY_IDS,
    TASK_ID as AGENT_FUTURE_GRID_TASK_ID,
    SymbolicAutomatonAgentFutureGridLabelTask,
)
from trace_tasks.tasks.symbolic.agent_automaton.agent_final_pose_label import (
    SUPPORTED_QUERY_IDS as AGENT_FINAL_QUERY_IDS,
    TASK_ID as AGENT_FINAL_TASK_ID,
    SymbolicAutomatonAgentFinalPoseLabelTask,
)
from trace_tasks.tasks.symbolic.life_automaton.life_future_grid_label import (
    SUPPORTED_QUERY_IDS as LIFE_GRID_QUERY_IDS,
    TASK_ID as LIFE_GRID_TASK_ID,
    SymbolicLifeAutomatonFutureGridLabelTask,
)
from trace_tasks.tasks.symbolic.life_automaton.one_step_cell_state_count import (
    SUPPORTED_QUERY_IDS as LIFE_STATE_COUNT_QUERY_IDS,
    TASK_ID as LIFE_STATE_COUNT_TASK_ID,
    SymbolicLifeAutomatonOneStepCellStateCountTask,
)
from trace_tasks.tasks.symbolic.life_automaton.shared.rules import SCENE_ID as LIFE_SCENE_ID
from trace_tasks.tasks.symbolic.turing_tape.final_head_position_value import (
    SUPPORTED_QUERY_IDS as TURING_HEAD_QUERY_IDS,
    TASK_ID as TURING_HEAD_TASK_ID,
    SymbolicTuringTapeFinalHeadPositionValueTask,
)
from trace_tasks.tasks.symbolic.turing_tape.shared.state import SCENE_ID as TURING_SCENE_ID
from trace_tasks.tasks.symbolic.turing_tape.turing_written_symbol_count import (
    SUPPORTED_QUERY_IDS as TURING_QUERY_IDS,
    TASK_ID as TURING_SYMBOL_COUNT_TASK_ID,
    SymbolicTuringTapeWrittenSymbolCountTask,
)


TASKS = (
    (AGENT_FINAL_TASK_ID, SymbolicAutomatonAgentFinalPoseLabelTask, AGENT_SCENE_ID, set(AGENT_FINAL_QUERY_IDS), "option_letter", "bbox_map"),
    (AGENT_FUTURE_GRID_TASK_ID, SymbolicAutomatonAgentFutureGridLabelTask, AGENT_SCENE_ID, set(AGENT_FUTURE_GRID_QUERY_IDS), "option_letter", "bbox_map"),
    (LIFE_GRID_TASK_ID, SymbolicLifeAutomatonFutureGridLabelTask, LIFE_SCENE_ID, set(LIFE_GRID_QUERY_IDS), "option_letter", "bbox_map"),
    (LIFE_STATE_COUNT_TASK_ID, SymbolicLifeAutomatonOneStepCellStateCountTask, LIFE_SCENE_ID, set(LIFE_STATE_COUNT_QUERY_IDS), "integer", "bbox_set"),
    (TURING_HEAD_TASK_ID, SymbolicTuringTapeFinalHeadPositionValueTask, TURING_SCENE_ID, set(TURING_HEAD_QUERY_IDS), "integer", "bbox_map"),
    (TURING_SYMBOL_COUNT_TASK_ID, SymbolicTuringTapeWrittenSymbolCountTask, TURING_SCENE_ID, set(TURING_QUERY_IDS), "integer", "bbox_map"),
)


def test_automaton_tasks_are_registered() -> None:
    for task_id, task_cls, _scene_id, _queries, _answer_type, _annotation_type in TASKS:
        assert TASK_REGISTRY[task_id] is task_cls
        task = task_cls()
        assert task.domain == "symbolic"
        if _scene_id in {AGENT_SCENE_ID, LIFE_SCENE_ID, TURING_SCENE_ID}:
            assert not hasattr(task, "scene_id")


def test_automaton_tasks_emit_contracts() -> None:
    for index, (_task_id, task_cls, scene_id, queries, answer_type, annotation_type) in enumerate(TASKS):
        out = task_cls().generate(2026052200 + index, params={}, max_attempts=30)
        trace = out.trace_payload
        execution = trace["execution_trace"]

        assert out.scene_id == scene_id
        assert out.query_id in queries
        assert out.answer_gt.type == answer_type
        assert out.annotation_gt.type == annotation_type
        assert trace["query_spec"]["params"]["query_id"] == out.query_id
        assert trace["render_spec"]["scene_id"] == scene_id
        assert trace["render_map"]["annotation_source"] == "item_bboxes_px"
        assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]
        assert trace["projected_annotation"][annotation_type] == out.annotation_gt.value
        assert out.image.size == (
            int(trace["render_spec"]["canvas_width"]),
            int(trace["render_spec"]["canvas_height"]),
        )
        assert execution["query_id"] == out.query_id
        assert execution["scene_id"] == scene_id
        assert "supporting_item_ids" in execution
        if _task_id != LIFE_STATE_COUNT_TASK_ID:
            assert execution["supporting_item_ids"]
        if annotation_type == "bbox":
            assert len(execution["supporting_item_ids"]) == 1
        else:
            assert len(out.annotation_gt.value) == len(execution["supporting_item_ids"])
        if _task_id == AGENT_FINAL_TASK_ID:
            assert execution["supporting_item_ids_by_role"]["start_marker"] == "initial_agent"
            assert execution["supporting_item_ids_by_role"]["selected_option"].startswith("option_")
            assert set(out.annotation_gt.value) == {"start_marker", "selected_option"}
            assert 3 <= int(execution["steps"]) <= 6
            assert len(execution["option_specs"]) == 4
            assert 3 <= int(execution["grid_rows"]) <= 5
            assert 3 <= int(execution["grid_cols"]) <= 5
        if _task_id == AGENT_FUTURE_GRID_TASK_ID:
            assert execution["supporting_item_ids_by_role"]["source_grid"] == "source_grid"
            assert execution["supporting_item_ids_by_role"]["selected_option"].startswith("option_")
            assert set(out.annotation_gt.value) == {"source_grid", "selected_option"}
            assert 3 <= int(execution["steps"]) <= 6
            assert 3 <= int(execution["grid_rows"]) <= 5
            assert 3 <= int(execution["grid_cols"]) <= 5
            assert len(execution["option_specs"]) == 4
            correct_options = [option for option in execution["option_specs"] if bool(option["is_correct"])]
            assert len(correct_options) == 1
            assert str(correct_options[0]["label"]) == str(out.answer_gt.value)
            assert correct_options[0]["grid"] == execution["final_grid"]
            render_params = trace["render_spec"]["render_params"]
            assert int(render_params["option_grid_cell_px"]) == int(render_params["cell_size_px"])
        if _task_id == LIFE_GRID_TASK_ID:
            assert execution["supporting_item_ids_by_role"]["source_grid"] == "source_grid"
            assert execution["supporting_item_ids_by_role"]["selected_option"].startswith("option_")
            assert set(out.annotation_gt.value) == {"source_grid", "selected_option"}
            assert len(execution["option_specs"]) == 4
            assert int(execution["grid_rows"]) == int(execution["grid_cols"])
            assert 3 <= int(execution["grid_rows"]) <= 5
            render_params = trace["render_spec"]["render_params"]
            assert int(render_params["option_grid_cell_px"]) == int(render_params["cell_size_px"])
        if _task_id == LIFE_STATE_COUNT_TASK_ID:
            assert out.query_id in {"one_step_alive_cell_count", "one_step_dead_cell_count"}
            assert execution["steps"] == 1
            assert int(execution["grid_rows"]) == int(execution["grid_cols"])
            assert 3 <= int(execution["grid_rows"]) <= 5
            assert execution["target_state_name"] in {"alive", "dead"}
            assert execution["target_state_value"] in {0, 1}
            assert int(out.answer_gt.value) == len(execution["target_cells_after_update"])
            assert len(out.annotation_gt.value) == int(out.answer_gt.value)
            for row, col in execution["target_cells_after_update"]:
                assert int(execution["future_grid"][int(row)][int(col)]) == int(execution["target_state_value"])
            assert all(str(item_id).startswith("source_cell_") for item_id in execution["supporting_item_ids"])
        if _task_id == TURING_SYMBOL_COUNT_TASK_ID:
            assert out.query_id == SINGLE_QUERY_ID
            assert execution["prompt_query_key"] == "written_symbol_count"
            assert execution["supporting_item_ids_by_role"] == {
                "machine_panel": "machine_panel",
                "transition_table": "transition_table",
            }
            assert set(out.annotation_gt.value) == {"machine_panel", "transition_table"}
        if _task_id == TURING_HEAD_TASK_ID:
            assert out.query_id == SINGLE_QUERY_ID
            assert execution["prompt_query_key"] == "final_head_position_value"
            assert execution["supporting_item_ids_by_role"] == {
                "machine_panel": "machine_panel",
                "transition_table": "transition_table",
            }
            assert set(out.annotation_gt.value) == {"machine_panel", "transition_table"}
            head = int(execution["start_head"])
            max_index = int(execution["tape_length"]) - 1
            for step_trace in execution["step_trace"]:
                move_delta = -1 if str(step_trace["move"]) == "L" else 1
                head = max(0, min(max_index, int(step_trace["head_position"]) + move_delta))
            assert int(execution["final_head_position_zero_based"]) == head
            assert int(execution["final_head_cell_number"]) == head + 1
            assert int(out.answer_gt.value) == head + 1
        if scene_id in {AGENT_SCENE_ID, LIFE_SCENE_ID, TURING_SCENE_ID}:
            assert trace["render_spec"]["scene_style"]["font"]["source"] == "global_font_pool"
            assert trace["render_spec"]["scene_style"]["font"]["font_family"]
        if scene_id == AGENT_SCENE_ID:
            agent_board = trace["render_spec"]["scene_style"]["agent_board"]
            assert agent_board["board_style"] in {
                "classic_grid",
                "rounded_tiles",
                "inset_cells",
                "lab_matrix",
                "notebook_cells",
            }
            assert agent_board["semantic_color_policy"]["state_colors_preserved_from_scene_style"] is True
        if scene_id == LIFE_SCENE_ID:
            assert trace["render_spec"]["layout_jitter"]["enabled"] is True
            life_board = trace["render_spec"]["scene_style"]["life_board"]
            assert life_board["board_style"] in {
                "classic_grid",
                "rounded_tiles",
                "inset_tiles",
                "lab_matrix",
                "notebook_cells",
                "terminal_cells",
            }
            assert life_board["cell_palette_id"]
            assert life_board["semantic_color_policy"]["alive_cells_remain_dark"] is True
            assert life_board["semantic_color_policy"]["empty_cells_remain_light"] is True
            assert life_board["contrast_checks"]["alive_dead_pass"] is True
            assert life_board["contrast_checks"]["mark_pass"] is True
        if annotation_type == "bbox_map":
            bboxes = out.annotation_gt.value.values()
        elif annotation_type == "bbox":
            bboxes = [out.annotation_gt.value]
        else:
            bboxes = out.annotation_gt.value
        for bbox in bboxes:
            assert len(bbox) == 4
            assert 0 <= float(bbox[0]) < float(bbox[2]) <= out.image.size[0]
            assert 0 <= float(bbox[1]) < float(bbox[3]) <= out.image.size[1]


def test_automaton_generation_is_deterministic() -> None:
    task = SymbolicAutomatonAgentFutureGridLabelTask()
    params = {
        "scene_variant": "lab_panel",
        "query_id": "single",
        "rule_variant": "three_state_rule",
    }
    out_a = task.generate(2026052299, params=params, max_attempts=30)
    out_b = task.generate(2026052299, params=params, max_attempts=30)

    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()
