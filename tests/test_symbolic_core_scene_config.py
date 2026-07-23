"""Regression tests for symbolic scene default config loading."""

from __future__ import annotations
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import (
    required_group_defaults,
    split_generation_rendering_prompt_defaults,
)


def test_symbolic_abacus_defaults_loaded() -> None:
    cfg = get_scene_defaults("symbolic", "abacus")
    generation_defaults, rendering_defaults, prompt_defaults = (
        split_generation_rendering_prompt_defaults(
            cfg, task_id="task_symbolic__abacus__displayed_value_readout"
        )
    )
    assert sorted(generation_defaults["scene_variant_weights"].keys()) == [
        "clean_card",
        "wood_frame",
        "worksheet",
    ]
    assert bool(generation_defaults["balanced_scene_variant_sampling"]) is True
    assert int(generation_defaults["target_answer_min"]) == 0
    assert int(generation_defaults["target_answer_max"]) == 999
    assert int(rendering_defaults["canvas_width"]) == 980
    assert int(rendering_defaults["canvas_height"]) == 760
    assert int(rendering_defaults["panel_width_px"]) == 800
    assert int(rendering_defaults["panel_height_px"]) == 540
    assert int(rendering_defaults["bead_width_px"]) > int(
        rendering_defaults["bead_height_px"]
    )
    assert str(prompt_defaults["bundle_id"]).strip() == "symbolic_abacus_v1"
    assert str(prompt_defaults["scene_key"]).strip() == "abacus"
    assert str(prompt_defaults["task_key"]).strip() == "abacus_displayed_value_query"
    resolved_prompt_values = required_group_defaults(
        prompt_defaults,
        (
            "object_description_displayed_value_readout_clean_card",
            "annotation_hint",
            "answer_hint",
        ),
        context="symbolic abacus readout prompt defaults",
    )
    assert str(
        resolved_prompt_values["object_description_displayed_value_readout_clean_card"]
    ).strip()
    assert str(resolved_prompt_values["annotation_hint"]).strip()
    assert str(resolved_prompt_values["answer_hint"]).strip()


def test_symbolic_abacus_place_digit_defaults_loaded() -> None:
    cfg = get_scene_defaults("symbolic", "abacus")
    generation_defaults, rendering_defaults, prompt_defaults = (
        split_generation_rendering_prompt_defaults(
            cfg, task_id="task_symbolic__abacus__place_digit_readout"
        )
    )
    assert sorted(generation_defaults["scene_variant_weights"].keys()) == [
        "clean_card",
        "wood_frame",
        "worksheet",
    ]
    assert bool(generation_defaults["balanced_scene_variant_sampling"]) is True
    assert int(generation_defaults["target_value_min"]) == 0
    assert int(generation_defaults["target_value_max"]) == 999
    assert sorted(generation_defaults["target_column_role_weights"].keys()) == [
        "hundreds",
        "ones",
        "tens",
    ]
    assert bool(generation_defaults["balanced_target_column_role_sampling"]) is True
    assert int(rendering_defaults["canvas_width"]) == 980
    assert int(rendering_defaults["canvas_height"]) == 760
    assert int(rendering_defaults["panel_width_px"]) == 800
    assert int(rendering_defaults["panel_height_px"]) == 540
    assert str(prompt_defaults["bundle_id"]).strip() == "symbolic_abacus_v1"
    assert str(prompt_defaults["scene_key"]).strip() == "abacus"
    assert str(prompt_defaults["task_key"]).strip() == "abacus_place_digit_query"
    resolved_prompt_values = required_group_defaults(
        prompt_defaults,
        (
            "object_description_place_digit_readout_clean_card",
            "annotation_hint_place_digit_readout",
            "answer_hint_place_digit_readout",
        ),
        context="symbolic abacus place-digit prompt defaults",
    )
    assert str(
        resolved_prompt_values["object_description_place_digit_readout_clean_card"]
    ).strip()
    assert str(resolved_prompt_values["annotation_hint_place_digit_readout"]).strip()
    assert str(resolved_prompt_values["answer_hint_place_digit_readout"]).strip()


def test_symbolic_abacus_option_panel_defaults_loaded() -> None:
    cfg = get_scene_defaults("symbolic", "abacus")
    generation_defaults, rendering_defaults, prompt_defaults = (
        split_generation_rendering_prompt_defaults(
            cfg, task_id="task_symbolic__abacus__target_value_match_label"
        )
    )
    assert sorted(generation_defaults["scene_variant_weights"].keys()) == [
        "clean_card",
        "wood_frame",
        "worksheet",
    ]
    assert list(generation_defaults["option_label_support"]) == [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
    ]
    assert int(generation_defaults["option_count"]) == 6
    assert bool(generation_defaults["balanced_correct_option_label_sampling"]) is True
    assert sorted(generation_defaults["correct_option_label_weights"].keys()) == [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
    ]
    assert int(generation_defaults["target_value_min"]) == 0
    assert int(generation_defaults["target_value_max"]) == 999
    assert int(rendering_defaults["canvas_width"]) == 1200
    assert int(rendering_defaults["canvas_height"]) == 760
    assert int(rendering_defaults["option_card_width_px"]) == 340
    assert int(rendering_defaults["option_card_height_px"]) == 280
    assert int(rendering_defaults["option_bead_width_px"]) > int(
        rendering_defaults["option_bead_height_px"]
    )
    assert str(prompt_defaults["bundle_id"]).strip() == "symbolic_abacus_v1"
    assert str(prompt_defaults["scene_key"]).strip() == "abacus"
    assert str(prompt_defaults["task_key"]).strip() == "abacus_target_value_match_query"


def test_symbolic_agent_automaton_defaults_loaded() -> None:
    cfg = get_scene_defaults("symbolic", "agent_automaton")
    for task_id in (
        "task_symbolic__agent_automaton__agent_final_pose_label",
        "task_symbolic__agent_automaton__future_grid_label",
    ):
        generation_defaults, rendering_defaults, prompt_defaults = (
            split_generation_rendering_prompt_defaults(cfg, task_id=task_id)
        )
        assert sorted(generation_defaults["scene_variant_weights"].keys()) == [
            "clean_grid",
            "lab_panel",
            "notebook_grid",
        ]
        assert sorted(generation_defaults["rule_variant_weights"].keys()) == [
            "binary_rule",
            "three_state_rule",
        ]
        assert sorted(generation_defaults["agent_board_style_weights"].keys()) == [
            "classic_grid",
            "inset_cells",
            "lab_matrix",
            "notebook_cells",
            "rounded_tiles",
        ]
        assert bool(generation_defaults["balanced_scene_variant_sampling"]) is True
        assert bool(generation_defaults["balanced_rule_variant_sampling"]) is True
        assert bool(generation_defaults["balanced_agent_board_style_sampling"]) is True
        assert int(generation_defaults["agent_rows_min"]) == 3
        assert int(generation_defaults["agent_rows_max"]) == 5
        assert int(generation_defaults["agent_cols_min"]) == 3
        assert int(generation_defaults["agent_cols_max"]) == 5
        assert int(rendering_defaults["canvas_width"]) == 1040
        assert int(rendering_defaults["canvas_height"]) == 880
        assert int(rendering_defaults["cell_size_px"]) == 56
        assert int(rendering_defaults["option_card_width_px"]) == 170
        assert int(rendering_defaults["option_card_height_px"]) == 170
        assert int(rendering_defaults["option_grid_cell_px"]) == 24
        assert (
            str(prompt_defaults["bundle_id"]).strip() == "symbolic_agent_automaton_v1"
        )
        assert str(prompt_defaults["scene_key"]).strip() == "agent_automaton"


def test_symbolic_life_automaton_defaults_loaded() -> None:
    cfg = get_scene_defaults("symbolic", "life_automaton")
    future_generation, future_rendering, future_prompt = (
        split_generation_rendering_prompt_defaults(
            cfg,
            task_id="task_symbolic__life_automaton__life_future_grid_label",
        )
    )
    assert sorted(future_generation["scene_variant_weights"].keys()) == [
        "clean_grid",
        "lab_panel",
        "notebook_grid",
    ]
    assert sorted(future_generation["life_board_style_weights"].keys()) == [
        "classic_grid",
        "inset_tiles",
        "lab_matrix",
        "notebook_cells",
        "rounded_tiles",
        "terminal_cells",
    ]
    assert bool(future_generation["balanced_scene_variant_sampling"]) is True
    assert "query_id_weights" not in future_generation
    assert int(future_generation["life_grid_size_min"]) == 3
    assert int(future_generation["life_grid_size_max"]) == 5
    assert int(future_generation["life_rows_min"]) == 3
    assert int(future_generation["life_rows_max"]) == 5
    assert int(future_generation["life_cols_min"]) == 3
    assert int(future_generation["life_cols_max"]) == 5
    assert int(future_generation["grid_option_count"]) == 4
    assert int(future_rendering["canvas_width"]) == 1040
    assert int(future_rendering["canvas_height"]) == 880
    assert str(future_prompt["bundle_id"]).strip() == "symbolic_life_automaton_v1"
    assert str(future_prompt["scene_key"]).strip() == "life_automaton"
    assert str(future_prompt["task_key"]).strip() == "life_future_grid_query"
    count_generation, count_rendering, count_prompt = (
        split_generation_rendering_prompt_defaults(
            cfg,
            task_id="task_symbolic__life_automaton__one_step_cell_state_count",
        )
    )
    assert sorted(count_generation["scene_variant_weights"].keys()) == [
        "clean_grid",
        "lab_panel",
        "notebook_grid",
    ]
    assert bool(count_generation["balanced_scene_variant_sampling"]) is True
    assert "query_id_weights" not in count_generation
    assert int(count_generation["life_grid_size_min"]) == 3
    assert int(count_generation["life_grid_size_max"]) == 5
    assert int(count_generation["life_rows_min"]) == 3
    assert int(count_generation["life_rows_max"]) == 5
    assert int(count_generation["life_cols_min"]) == 3
    assert int(count_generation["life_cols_max"]) == 5
    assert int(count_rendering["canvas_width"]) == 1040
    assert int(count_rendering["canvas_height"]) == 880
    assert str(count_prompt["bundle_id"]).strip() == "symbolic_life_automaton_v1"
    assert str(count_prompt["scene_key"]).strip() == "life_automaton"
    assert (
        str(count_prompt["task_key"]).strip() == "life_one_step_cell_state_count_query"
    )


def test_symbolic_logic_gate_circuit_defaults_loaded() -> None:
    cfg = get_scene_defaults("symbolic", "logic_gate_circuit")
    label_generation, label_rendering, label_prompt = (
        split_generation_rendering_prompt_defaults(
            cfg,
            task_id="task_symbolic__logic_gate_circuit__output_value_label",
        )
    )
    assert sorted(label_generation["scene_variant_weights"].keys()) == [
        "clean_worksheet",
        "exam_scan",
        "notebook_problem",
    ]
    assert bool(label_generation["balanced_scene_variant_sampling"]) is True
    assert "query_id_weights" not in label_generation
    assert int(label_generation["option_count"]) == 4
    assert int(label_generation["input_count_min"]) == 2
    assert int(label_generation["input_count_max"]) == 2
    assert int(label_rendering["logic_canvas_width"]) == 1180
    assert int(label_rendering["logic_canvas_height"]) == 820
    assert str(label_prompt["bundle_id"]).strip() == "symbolic_logic_gate_circuit_v1"
    assert str(label_prompt["scene_key"]).strip() == "logic_gate_circuit"
    assert str(label_prompt["task_key"]).strip() == "logic_gate_output_value_label"

    assignment_generation, _assignment_rendering, assignment_prompt = (
        split_generation_rendering_prompt_defaults(
            cfg,
            task_id="task_symbolic__logic_gate_circuit__satisfying_assignment_label",
        )
    )
    assert "query_id_weights" not in assignment_generation
    assert int(assignment_generation["option_count"]) == 4
    assert (
        str(assignment_prompt["task_key"]).strip()
        == "logic_gate_satisfying_assignment_label"
    )
    resolved_prompt_values = required_group_defaults(
        assignment_prompt,
        (
            "annotation_hint_assignment_outputs_one_label",
            "answer_hint_assignment_outputs_zero_label",
        ),
        context="symbolic logic-gate prompt defaults",
    )
    assert str(
        resolved_prompt_values["annotation_hint_assignment_outputs_one_label"]
    ).strip()
    assert str(
        resolved_prompt_values["answer_hint_assignment_outputs_zero_label"]
    ).strip()

    gate_generation, _gate_rendering, gate_prompt = (
        split_generation_rendering_prompt_defaults(
            cfg,
            task_id="task_symbolic__logic_gate_circuit__gate_type_count",
        )
    )
    assert "query_id_weights" not in gate_generation
    assert int(gate_generation["gate_count"]) == 4
    assert int(gate_generation["target_answer_min"]) == 0
    assert int(gate_generation["target_answer_max"]) == 4
    assert list(gate_generation["target_gate_type_support"]) == [
        "AND",
        "OR",
        "NOT",
        "XOR",
        "NAND",
        "NOR",
    ]
    assert str(gate_prompt["task_key"]).strip() == "logic_gate_gate_type_count"
    resolved_gate_prompt_values = required_group_defaults(
        gate_prompt,
        ("annotation_hint_gate_type_count", "answer_hint_gate_type_count"),
        context="symbolic logic-gate count prompt defaults",
    )
    assert str(resolved_gate_prompt_values["annotation_hint_gate_type_count"]).strip()
    assert str(resolved_gate_prompt_values["answer_hint_gate_type_count"]).strip()

    internal_generation, _internal_rendering, internal_prompt = (
        split_generation_rendering_prompt_defaults(
            cfg,
            task_id="task_symbolic__logic_gate_circuit__internal_output_count",
        )
    )
    assert "query_id_weights" not in internal_generation
    assert int(internal_generation["gate_count"]) == 4
    assert int(internal_generation["target_answer_min"]) == 0
    assert int(internal_generation["target_answer_max"]) == 4
    assert (
        str(internal_prompt["task_key"]).strip() == "logic_gate_internal_output_count"
    )
    resolved_internal_prompt_values = required_group_defaults(
        internal_prompt,
        (
            "annotation_hint_internal_output_one_count",
            "answer_hint_internal_output_zero_count",
        ),
        context="symbolic logic-gate internal-output prompt defaults",
    )
    assert str(
        resolved_internal_prompt_values["annotation_hint_internal_output_one_count"]
    ).strip()
    assert str(
        resolved_internal_prompt_values["answer_hint_internal_output_zero_count"]
    ).strip()


def test_symbolic_braille_cell_defaults_loaded() -> None:
    cfg = get_scene_defaults("symbolic", "braille_cell")
    for task_id in (
        "task_symbolic__braille_cell__matching_pattern_label",
        "task_symbolic__braille_cell__braille_word_read_label",
        "task_symbolic__braille_cell__word_braille_match_label",
    ):
        generation_defaults, rendering_defaults, prompt_defaults = (
            split_generation_rendering_prompt_defaults(cfg, task_id=task_id)
        )
        assert sorted(generation_defaults["scene_variant_weights"].keys()) == [
            "clean_card",
            "exam_scan",
            "notebook_card",
        ]
        assert bool(generation_defaults["balanced_scene_variant_sampling"]) is True
        assert int(generation_defaults["option_count"]) == 6
        assert int(generation_defaults["reference_raised_dot_count_min"]) == 1
        assert int(generation_defaults["reference_raised_dot_count_max"]) == 6
        assert int(generation_defaults["word_length_min"]) == 3
        assert int(generation_defaults["word_length_max"]) == 5
        assert int(generation_defaults["word_option_count"]) == 4
        assert dict(
            generation_defaults["word_option_shared_prefix_length_weights"]
        ) == {1: 1.0, 2: 1.0, 3: 0.5}
        assert int(rendering_defaults["canvas_width"]) == 980
        assert int(rendering_defaults["canvas_height"]) == 680
        assert int(rendering_defaults["cell_width_px"]) == 132
        assert int(rendering_defaults["cell_height_px"]) == 184
        assert int(rendering_defaults["dot_radius_px"]) > int(
            rendering_defaults["empty_dot_radius_px"]
        )
        assert str(prompt_defaults["bundle_id"]).strip() == "symbolic_braille_cell_v1"
        assert str(prompt_defaults["scene_key"]).strip() == "braille_cell"
        resolved_prompt_values = required_group_defaults(
            prompt_defaults,
            (
                "annotation_hint_matching_pattern_label",
                "annotation_hint_braille_word_read_label",
                "annotation_hint_word_braille_match_label",
            ),
            context="symbolic braille prompt defaults",
        )
        assert str(
            resolved_prompt_values["annotation_hint_matching_pattern_label"]
        ).strip()
        assert str(
            resolved_prompt_values["annotation_hint_braille_word_read_label"]
        ).strip()
        assert str(
            resolved_prompt_values["annotation_hint_word_braille_match_label"]
        ).strip()


def test_symbolic_morse_code_defaults_loaded() -> None:
    cfg = get_scene_defaults("symbolic", "morse_code")
    for task_id in (
        "task_symbolic__morse_code__morse_word_read_label",
        "task_symbolic__morse_code__word_morse_match_label",
    ):
        generation_defaults, rendering_defaults, prompt_defaults = (
            split_generation_rendering_prompt_defaults(cfg, task_id=task_id)
        )
        assert sorted(generation_defaults["scene_variant_weights"].keys()) == [
            "clean_card",
            "exam_scan",
            "notebook_card",
        ]
        assert bool(generation_defaults["balanced_scene_variant_sampling"]) is True
        assert int(generation_defaults["word_length_min"]) == 3
        assert int(generation_defaults["word_length_max"]) == 5
        assert int(generation_defaults["word_option_count"]) == 4
        assert dict(
            generation_defaults["word_option_shared_prefix_length_weights"]
        ) == {1: 1.0, 2: 1.0, 3: 0.5}
        assert int(rendering_defaults["canvas_width"]) == 980
        assert int(rendering_defaults["canvas_height"]) == 680
        assert int(rendering_defaults["code_symbol_dot_radius_px"]) > int(
            rendering_defaults["option_symbol_dot_radius_px"]
        )
        assert int(rendering_defaults["code_symbol_dash_width_px"]) > int(
            rendering_defaults["option_symbol_dash_width_px"]
        )
        assert str(prompt_defaults["bundle_id"]).strip() == "symbolic_morse_code_v1"
        assert str(prompt_defaults["scene_key"]).strip() == "morse_code"
        resolved_prompt_values = required_group_defaults(
            prompt_defaults,
            (
                "annotation_hint_morse_word_read_label",
                "annotation_hint_word_morse_match_label",
            ),
            context="symbolic Morse-code prompt defaults",
        )
        assert str(
            resolved_prompt_values["annotation_hint_morse_word_read_label"]
        ).strip()
        assert str(
            resolved_prompt_values["annotation_hint_word_morse_match_label"]
        ).strip()


def test_symbolic_truth_table_defaults_loaded() -> None:
    cfg = get_scene_defaults("symbolic", "truth_table")
    for task_id, task_key in (
        ("task_symbolic__truth_table__satisfying_row_count", "satisfying_row_count"),
        ("task_symbolic__truth_table__truth_pattern_label", "truth_pattern_label"),
        (
            "task_symbolic__truth_table__expression_from_rows_label",
            "expression_from_rows_label",
        ),
    ):
        generation_defaults, rendering_defaults, prompt_defaults = (
            split_generation_rendering_prompt_defaults(cfg, task_id=task_id)
        )
        assert sorted(generation_defaults["scene_variant_weights"].keys()) == [
            "clean_table",
            "exam_scan",
            "notebook_table",
        ]
        assert bool(generation_defaults["balanced_scene_variant_sampling"]) is True
        assert list(generation_defaults["option_label_support"]) == [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
        ]
        assert int(rendering_defaults["canvas_width"]) == 1120
        assert int(rendering_defaults["canvas_height"]) == 780
        assert int(rendering_defaults["row_height_px"]) == 52
        assert str(prompt_defaults["bundle_id"]).strip() == "symbolic_truth_table_v1"
        assert str(prompt_defaults["scene_key"]).strip() == "truth_table"
        assert str(prompt_defaults["task_key"]).strip() == task_key
    count_generation, _count_rendering, count_prompt = (
        split_generation_rendering_prompt_defaults(
            cfg,
            task_id="task_symbolic__truth_table__satisfying_row_count",
        )
    )
    assert list(count_generation["target_count_support"]) == [1, 2, 3, 4, 5, 6, 7]
    expression_generation, _expression_rendering, _expression_prompt = (
        split_generation_rendering_prompt_defaults(
            cfg,
            task_id="task_symbolic__truth_table__expression_from_rows_label",
        )
    )
    assert list(expression_generation["expression_option_label_support"]) == [
        "W",
        "X",
        "Y",
        "Z",
    ]
    resolved_prompt_values = required_group_defaults(
        count_prompt,
        ("annotation_hint_satisfying_row_count", "answer_hint_satisfying_row_count"),
        context="symbolic truth-table prompt defaults",
    )
    assert str(resolved_prompt_values["annotation_hint_satisfying_row_count"]).strip()
    assert str(resolved_prompt_values["answer_hint_satisfying_row_count"]).strip()


def test_symbolic_radial_code_wheel_defaults_loaded() -> None:
    cfg = get_scene_defaults("symbolic", "radial_code_wheel")
    for task_id in (
        "task_symbolic__radial_code_wheel__code_output_label",
        "task_symbolic__radial_code_wheel__output_code_match_label",
    ):
        generation_defaults, rendering_defaults, prompt_defaults = (
            split_generation_rendering_prompt_defaults(cfg, task_id=task_id)
        )
        assert sorted(generation_defaults["scene_variant_weights"].keys()) == [
            "clean_wheel",
            "exam_scan",
            "notebook_wheel",
        ]
        assert bool(generation_defaults["balanced_scene_variant_sampling"]) is True
        assert int(generation_defaults["option_count"]) == 6
        assert int(rendering_defaults["canvas_width"]) == 1080
        assert int(rendering_defaults["canvas_height"]) == 860
        assert int(rendering_defaults["ring_width_px"]) == 80
        assert int(rendering_defaults["terminal_label_radius_px"]) > int(
            rendering_defaults["wheel_inner_radius_px"]
        )
        assert (
            str(prompt_defaults["bundle_id"]).strip() == "symbolic_radial_code_wheel_v1"
        )
        assert str(prompt_defaults["scene_key"]).strip() == "radial_code_wheel"
        resolved_prompt_values = required_group_defaults(
            prompt_defaults,
            (
                "annotation_hint_code_output_label",
                "annotation_hint_output_code_match_label",
            ),
            context="symbolic radial code-wheel prompt defaults",
        )
        assert str(resolved_prompt_values["annotation_hint_code_output_label"]).strip()
        assert str(
            resolved_prompt_values["annotation_hint_output_code_match_label"]
        ).strip()
    missing_generation, missing_rendering, missing_prompt = (
        split_generation_rendering_prompt_defaults(
            cfg,
            task_id="task_symbolic__radial_code_wheel__missing_code_symbol_label",
        )
    )
    assert list(missing_generation["missing_position_support"]) == [
        "inner",
        "middle",
        "outer",
    ]
    assert sorted(missing_generation["answer_symbol_weights"].keys()) == [
        "A",
        "B",
        "C",
        "D",
    ]
    assert bool(missing_generation["balanced_answer_symbol_sampling"]) is True
    assert int(missing_rendering["missing_code_card_top_px"]) == 184
    assert "symbol_option_grid_top_px" not in missing_rendering
    assert str(missing_prompt["bundle_id"]).strip() == "symbolic_radial_code_wheel_v1"
    assert str(missing_prompt["scene_key"]).strip() == "radial_code_wheel"
    resolved_missing_prompt_values = required_group_defaults(
        missing_prompt,
        (
            "annotation_hint_missing_code_symbol_label",
            "answer_hint_missing_code_symbol_label",
        ),
        context="symbolic radial code-wheel missing-symbol prompt defaults",
    )
    assert str(
        resolved_missing_prompt_values["annotation_hint_missing_code_symbol_label"]
    ).strip()
    assert str(
        resolved_missing_prompt_values["answer_hint_missing_code_symbol_label"]
    ).strip()


def test_symbolic_organic_structure_defaults_loaded() -> None:
    cfg = get_scene_defaults("symbolic", "organic_structure")
    bond_generation, bond_rendering, bond_prompt = (
        split_generation_rendering_prompt_defaults(
            cfg,
            task_id="task_symbolic__organic_structure__bond_order_count",
        )
    )
    assert sorted(bond_generation["scene_variant_weights"].keys()) == [
        "clean_worksheet",
        "exam_scan",
        "notebook_problem",
    ]
    assert bool(bond_generation["balanced_scene_variant_sampling"]) is True
    assert "query_id_weights" not in bond_generation
    assert sorted(bond_generation["target_bond_order_weights"].keys()) == [
        "double",
        "triple",
    ]
    assert int(bond_generation["target_answer_min"]) == 1
    assert int(bond_generation["target_answer_max"]) == 5
    assert int(bond_rendering["canvas_width"]) == 1180
    assert int(bond_rendering["canvas_height"]) == 820
    assert str(bond_prompt["bundle_id"]).strip() == "symbolic_organic_structure_v1"
    assert str(bond_prompt["scene_key"]).strip() == "organic_structure"
    assert str(bond_prompt["task_key"]).strip() == "organic_structure_bond_order_count"

    ring_generation, _ring_rendering, ring_prompt = (
        split_generation_rendering_prompt_defaults(
            cfg,
            task_id="task_symbolic__organic_structure__ring_size_count",
        )
    )
    assert "query_id_weights" not in ring_generation
    assert sorted(ring_generation["target_ring_size_weights"].keys()) == ["5", "6"]
    assert int(ring_generation["target_answer_min"]) == 1
    assert int(ring_generation["target_answer_max"]) == 5
    assert str(ring_prompt["task_key"]).strip() == "organic_structure_ring_size_count"
    resolved_prompt_values = required_group_defaults(
        ring_prompt,
        ("annotation_hint_ring_size_count", "answer_hint_ring_size_count"),
        context="symbolic organic prompt defaults",
    )
    assert str(resolved_prompt_values["annotation_hint_ring_size_count"]).strip()
    assert str(resolved_prompt_values["answer_hint_ring_size_count"]).strip()


def test_symbolic_chemical_equation_defaults_loaded() -> None:
    cfg = get_scene_defaults("symbolic", "chemical_equation")
    for task_id, task_key in (
        (
            "task_symbolic__chemical_equation__missing_coefficient_value",
            "missing_coefficient_value",
        ),
        (
            "task_symbolic__chemical_equation__balanced_option_label",
            "balanced_option_label",
        ),
    ):
        generation_defaults, rendering_defaults, prompt_defaults = (
            split_generation_rendering_prompt_defaults(cfg, task_id=task_id)
        )
        assert sorted(generation_defaults["scene_variant_weights"].keys()) == [
            "clean_lab",
            "notebook_scan",
            "worksheet",
        ]
        assert bool(generation_defaults["balanced_scene_variant_sampling"]) is True
        assert list(generation_defaults["coefficient_answer_support"]) == [
            1,
            2,
            3,
            4,
            5,
        ]
        assert int(rendering_defaults["canvas_width"]) == 1240
        assert int(rendering_defaults["canvas_height"]) == 820
        assert int(rendering_defaults["molecule_card_width_px"]) == 160
        assert str(prompt_defaults["bundle_id"]).strip() == "symbolic_chemical_equation_v1"
        assert str(prompt_defaults["scene_key"]).strip() == "chemical_equation"
        assert str(prompt_defaults["task_key"]).strip() == task_key

    option_generation, _option_rendering, option_prompt = (
        split_generation_rendering_prompt_defaults(
            cfg,
            task_id="task_symbolic__chemical_equation__balanced_option_label",
        )
    )
    assert list(option_generation["option_label_support"]) == ["A", "B", "C", "D"]
    resolved_prompt_values = required_group_defaults(
        option_prompt,
        (
            "annotation_hint_balanced_option_label",
            "answer_hint_balanced_option_label",
        ),
        context="symbolic chemical-equation prompt defaults",
    )
    assert str(resolved_prompt_values["annotation_hint_balanced_option_label"]).strip()
    assert str(resolved_prompt_values["answer_hint_balanced_option_label"]).strip()


def test_symbolic_clock_defaults_loaded() -> None:
    cfg = get_scene_defaults("symbolic", "clock")
    generation_shared = cfg["generation"]["shared"]
    assert sorted(generation_shared["scene_variant_weights"].keys()) == [
        "classic",
        "minimal",
        "outline",
    ]
    assert sorted(generation_shared["style_variant_weights"].keys()) == [
        "accented",
        "marker",
        "studio",
    ]
    assert sorted(generation_shared["accent_color_name_weights"].keys()) == [
        "blue",
        "brown",
        "cyan",
        "green",
        "magenta",
        "maroon",
        "orange",
        "purple",
        "red",
        "yellow",
    ]
    assert bool(generation_shared["balanced_scene_variant_sampling"]) is True
    assert bool(generation_shared["balanced_style_variant_sampling"]) is True
    assert bool(generation_shared["balanced_accent_color_name_sampling"]) is True
    assert int(generation_shared["hour_min"]) == 1
    assert int(generation_shared["hour_max"]) == 12
    assert int(generation_shared["minute_step"]) == 5
    assert int(generation_shared["second_step"]) == 5
    assert float(generation_shared["min_hand_angle_gap_deg"]) == 10.0
    render_shared = cfg["rendering"]["shared"]
    assert int(render_shared["canvas_width"]) == 640
    assert int(render_shared["canvas_height"]) == 640
    assert int(render_shared["face_radius_px"]) > 0
    assert int(render_shared["hour_hand_width_px"]) > int(
        render_shared["minute_hand_width_px"]
    )
    assert int(render_shared["minute_hand_width_px"]) > int(
        render_shared["second_hand_width_px"]
    )
    assert int(render_shared["minor_tick_dot_radius_px"]) >= 2
    assert int(render_shared["inner_ring_inset_px"]) > 0
    assert int(render_shared["inner_ring_width_px"]) > 0
    prompt_shared = cfg["prompt"]["shared"]
    assert str(prompt_shared["bundle_id"]).strip() == "symbolic_clock_v1"
    assert str(prompt_shared["scene_key"]).strip() == "clock"


def test_symbolic_clock_offset_readout_defaults_loaded() -> None:
    cfg = get_scene_defaults("symbolic", "clock")
    generation_defaults, rendering_defaults, prompt_defaults = (
        split_generation_rendering_prompt_defaults(
            cfg, task_id="task_symbolic__clock__offset_readout"
        )
    )
    assert dict(generation_defaults["delta_minutes_support"]) == {
        "min": 5,
        "max": 600,
        "step": 5,
    }
    assert int(rendering_defaults["canvas_width"]) == 640
    assert int(rendering_defaults["canvas_height"]) == 640
    assert str(prompt_defaults["bundle_id"]).strip() == "symbolic_clock_v1"
    assert str(prompt_defaults["scene_key"]).strip() == "clock"
    assert str(prompt_defaults["task_key"]).strip() == "clock_offset_readout_query"
    resolved_prompt_values = required_group_defaults(
        prompt_defaults,
        (
            "object_description_offset_readout_classic",
            "annotation_hint_offset_readout",
            "answer_hint_offset_readout",
        ),
        context="symbolic clock offset prompt defaults",
    )
    assert str(
        resolved_prompt_values["object_description_offset_readout_classic"]
    ).strip()
    assert str(resolved_prompt_values["annotation_hint_offset_readout"]).strip()
    assert str(resolved_prompt_values["answer_hint_offset_readout"]).strip()


def test_symbolic_clock_hand_angle_defaults_loaded() -> None:
    cfg = get_scene_defaults("symbolic", "clock")
    generation_defaults, rendering_defaults, prompt_defaults = (
        split_generation_rendering_prompt_defaults(
            cfg, task_id="task_symbolic__clock__hand_angle_value"
        )
    )
    assert int(generation_defaults["minute_step"]) == 10
    assert float(generation_defaults["min_hand_angle_gap_deg"]) == 15.0
    assert int(generation_defaults["hand_angle_answer_min"]) == 15
    assert int(generation_defaults["hand_angle_answer_max"]) == 180
    assert int(generation_defaults["hand_angle_answer_step"]) == 5
    assert int(rendering_defaults["canvas_width"]) == 640
    assert int(rendering_defaults["canvas_height"]) == 640
    assert str(prompt_defaults["bundle_id"]).strip() == "symbolic_clock_v1"
    assert str(prompt_defaults["scene_key"]).strip() == "clock"
    assert str(prompt_defaults["task_key"]).strip() == "clock_hand_angle_value_query"
    resolved_prompt_values = required_group_defaults(
        prompt_defaults,
        (
            "object_description_hand_angle_value_classic",
            "annotation_hint_hand_angle_value",
            "answer_hint_hand_angle_value",
        ),
        context="symbolic clock hand-angle prompt defaults",
    )
    assert str(
        resolved_prompt_values["object_description_hand_angle_value_classic"]
    ).strip()
    assert str(resolved_prompt_values["annotation_hint_hand_angle_value"]).strip()
    assert str(resolved_prompt_values["answer_hint_hand_angle_value"]).strip()


def test_symbolic_clock_full_time_readout_defaults_loaded() -> None:
    cfg = get_scene_defaults("symbolic", "clock")
    generation_defaults, rendering_defaults, prompt_defaults = (
        split_generation_rendering_prompt_defaults(
            cfg, task_id="task_symbolic__clock__full_time_readout"
        )
    )
    assert int(generation_defaults["minute_step"]) == 5
    assert int(generation_defaults["second_step"]) == 5
    assert float(generation_defaults["min_hand_angle_gap_deg"]) == 15.0
    assert int(rendering_defaults["canvas_width"]) == 640
    assert int(rendering_defaults["canvas_height"]) == 640
    assert int(rendering_defaults["second_hand_width_px"]) > 0
    assert str(prompt_defaults["bundle_id"]).strip() == "symbolic_clock_v1"
    assert str(prompt_defaults["scene_key"]).strip() == "clock"
    assert str(prompt_defaults["task_key"]).strip() == "clock_full_time_readout_query"
    resolved_prompt_values = required_group_defaults(
        prompt_defaults,
        (
            "object_description_full_time_readout_classic",
            "annotation_hint_full_time_readout",
            "answer_hint_full_time_readout",
        ),
        context="symbolic clock full-time prompt defaults",
    )
    assert str(
        resolved_prompt_values["object_description_full_time_readout_classic"]
    ).strip()
    assert str(resolved_prompt_values["annotation_hint_full_time_readout"]).strip()
    assert str(resolved_prompt_values["answer_hint_full_time_readout"]).strip()


def test_symbolic_clock_alarm_wait_time_defaults_loaded() -> None:
    cfg = get_scene_defaults("symbolic", "clock")
    generation_defaults, rendering_defaults, prompt_defaults = (
        split_generation_rendering_prompt_defaults(
            cfg, task_id="task_symbolic__clock__alarm_wait_time_value"
        )
    )
    assert int(generation_defaults["alarm_hour_min"]) == 1
    assert int(generation_defaults["alarm_hour_max"]) == 12
    assert float(generation_defaults["min_alarm_hand_gap_deg"]) == 20.0
    assert float(generation_defaults["accent_color_name_weights"]["red"]) == 0.0
    assert float(generation_defaults["accent_color_name_weights"]["maroon"]) == 0.0
    assert float(generation_defaults["accent_color_name_weights"]["orange"]) == 0.0
    assert float(generation_defaults["accent_color_name_weights"]["blue"]) == 1.0
    assert int(rendering_defaults["canvas_width"]) == 640
    assert int(rendering_defaults["canvas_height"]) == 640
    assert str(prompt_defaults["bundle_id"]).strip() == "symbolic_clock_v1"
    assert str(prompt_defaults["scene_key"]).strip() == "clock"
    assert (
        str(prompt_defaults["task_key"]).strip() == "clock_alarm_wait_time_value_query"
    )
    resolved_prompt_values = required_group_defaults(
        prompt_defaults,
        (
            "object_description_alarm_wait_time_value_classic",
            "annotation_hint_alarm_wait_time_value",
            "answer_hint_alarm_wait_time_value",
        ),
        context="symbolic clock alarm wait-time prompt defaults",
    )
    assert str(
        resolved_prompt_values["object_description_alarm_wait_time_value_classic"]
    ).strip()
    assert str(resolved_prompt_values["annotation_hint_alarm_wait_time_value"]).strip()
    assert str(resolved_prompt_values["answer_hint_alarm_wait_time_value"]).strip()


def test_symbolic_clock_compare_defaults_loaded() -> None:
    cfg = get_scene_defaults("symbolic", "clock")
    generation_defaults, rendering_defaults, prompt_defaults = (
        split_generation_rendering_prompt_defaults(
            cfg, task_id="task_symbolic__clock__time_extremum_label"
        )
    )
    assert list(generation_defaults["clock_label_support"]) == [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
    ]
    assert list(generation_defaults["clock_count_support"]) == [6, 7, 8, 9, 10, 11, 12]
    assert int(generation_defaults["min_compare_gap_minutes"]) == 15
    assert int(rendering_defaults["canvas_width"]) == 960
    assert int(rendering_defaults["canvas_height"]) == 760
    assert int(rendering_defaults["face_radius_px"]) == 84
    assert int(rendering_defaults["label_font_size_px"]) == 28
    assert str(prompt_defaults["bundle_id"]).strip() == "symbolic_clock_v1"
    assert str(prompt_defaults["scene_key"]).strip() == "clock"
    assert str(prompt_defaults["task_key"]).strip() == "clock_time_extremum_label_query"
    assert str(
        required_group_defaults(
            prompt_defaults,
            ("object_description_time_extremum_label_classic",),
            context="symbolic clock compare prompt defaults",
        )["object_description_time_extremum_label_classic"]
    ).strip()
    assert str(
        required_group_defaults(
            prompt_defaults,
            ("annotation_hint_time_extremum_label",),
            context="symbolic clock compare prompt defaults",
        )["annotation_hint_time_extremum_label"]
    ).strip()
    assert str(
        required_group_defaults(
            prompt_defaults,
            ("answer_hint_time_extremum_label",),
            context="symbolic clock compare prompt defaults",
        )["answer_hint_time_extremum_label"]
    ).strip()


def test_symbolic_clock_time_order_defaults_loaded() -> None:
    cfg = get_scene_defaults("symbolic", "clock")
    generation_defaults, rendering_defaults, prompt_defaults = (
        split_generation_rendering_prompt_defaults(
            cfg, task_id="task_symbolic__clock__time_order_label"
        )
    )
    assert list(generation_defaults["clock_label_support"]) == ["A", "B", "C", "D"]
    assert list(generation_defaults["option_label_support"]) == [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
    ]
    assert int(generation_defaults["option_count"]) == 6
    assert int(generation_defaults["min_compare_gap_minutes"]) == 20
    assert int(rendering_defaults["canvas_width"]) == 980
    assert int(rendering_defaults["canvas_height"]) == 760
    assert int(rendering_defaults["face_radius_px"]) == 76
    assert int(rendering_defaults["option_card_width_px"]) == 280
    assert int(rendering_defaults["option_card_height_px"]) == 88
    assert str(prompt_defaults["bundle_id"]).strip() == "symbolic_clock_v1"
    assert str(prompt_defaults["scene_key"]).strip() == "clock"
    assert str(prompt_defaults["task_key"]).strip() == "clock_time_order_label_query"
    assert str(
        required_group_defaults(
            prompt_defaults,
            ("object_description_time_order_label_classic",),
            context="symbolic clock time-order prompt defaults",
        )["object_description_time_order_label_classic"]
    ).strip()
    assert str(
        required_group_defaults(
            prompt_defaults,
            ("annotation_hint_time_order_label",),
            context="symbolic clock time-order prompt defaults",
        )["annotation_hint_time_order_label"]
    ).strip()
    assert str(
        required_group_defaults(
            prompt_defaults,
            ("answer_hint_time_order_label",),
            context="symbolic clock time-order prompt defaults",
        )["answer_hint_time_order_label"]
    ).strip()


def test_symbolic_clock_equivalent_time_defaults_loaded() -> None:
    cfg = get_scene_defaults("symbolic", "clock")
    generation_defaults, rendering_defaults, prompt_defaults = (
        split_generation_rendering_prompt_defaults(
            cfg, task_id="task_symbolic__clock__equivalent_time_label"
        )
    )
    assert sorted(generation_defaults["digital_display_palette_weights"].keys()) == [
        "blue_lcd",
        "charcoal_mint",
        "cream_ink",
        "forest_lime",
        "graphite_amber",
        "navy_cyan",
        "plum_ice",
        "wine_rose",
    ]
    assert (
        bool(generation_defaults["balanced_digital_display_palette_sampling"]) is True
    )
    assert list(generation_defaults["option_label_support"]) == [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
    ]
    assert int(generation_defaults["option_count"]) == 6
    assert int(generation_defaults["min_option_gap_minutes"]) == 10
    assert int(rendering_defaults["canvas_width"]) == 980
    assert int(rendering_defaults["canvas_height"]) == 760
    assert int(rendering_defaults["reference_clock_radius_px"]) == 136
    assert int(rendering_defaults["option_clock_radius_px"]) == 76
    assert int(rendering_defaults["digital_font_size_px"]) == 58
    assert int(rendering_defaults["option_digital_font_size_px"]) == 42
    assert str(prompt_defaults["bundle_id"]).strip() == "symbolic_clock_v1"
    assert str(prompt_defaults["scene_key"]).strip() == "clock"
    assert (
        str(prompt_defaults["task_key"]).strip() == "clock_equivalent_time_label_query"
    )
    assert str(
        required_group_defaults(
            prompt_defaults,
            (
                "object_description_equivalent_time_label_analog_reference_digital_options",
            ),
            context="symbolic clock match prompt defaults",
        )["object_description_equivalent_time_label_analog_reference_digital_options"]
    ).strip()
    assert str(
        required_group_defaults(
            prompt_defaults,
            (
                "object_description_equivalent_time_label_digital_reference_analog_options",
            ),
            context="symbolic clock match prompt defaults",
        )["object_description_equivalent_time_label_digital_reference_analog_options"]
    ).strip()
    assert str(
        required_group_defaults(
            prompt_defaults,
            ("annotation_hint_equivalent_time_label",),
            context="symbolic clock match prompt defaults",
        )["annotation_hint_equivalent_time_label"]
    ).strip()
    assert str(
        required_group_defaults(
            prompt_defaults,
            ("answer_hint_equivalent_time_label",),
            context="symbolic clock match prompt defaults",
        )["answer_hint_equivalent_time_label"]
    ).strip()
