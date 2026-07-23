"""Regression tests for scene default config loading."""

from __future__ import annotations

import pytest
from trace_tasks.core.scene_config import (
    get_domain_defaults,
    get_scene_defaults,
    resolve_scene_section_defaults,
)
from trace_tasks.tasks.shared.config_defaults import (
    required_group_default,
    required_group_defaults,
    resolve_optional_int_bounds,
    resolve_required_float_bounds,
    resolve_required_int_bounds,
    split_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.graph.shared.graph_sample_types import SUPPORTED_LAYOUT_VARIANTS

FULL_NODE_LINK_LAYOUT_VARIANTS = set(SUPPORTED_LAYOUT_VARIANTS)


def test_puzzles_star_battle_defaults_loaded() -> None:
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
    assert "query_id_weights" not in generation_defaults
    assert int(generation_defaults["target_count_min"]) == 1
    assert int(generation_defaults["target_count_max"]) == 6
    assert int(rendering_defaults["canvas_width"]) == 1080


def test_puzzles_raven_matrix_scene_defaults_loaded() -> None:
    cfg = get_scene_defaults("puzzles", "raven_matrix")
    task_id = "task_puzzles__raven_matrix__raven_count_progression_label"
    raven_generation_defaults, raven_rendering_defaults, raven_prompt_defaults = (
        split_generation_rendering_prompt_defaults(cfg, task_id=task_id)
    )
    assert str(raven_prompt_defaults["scene_key"]).strip() == "raven_matrix"
    assert str(raven_prompt_defaults["bundle_id"]).strip() == "puzzles_raven_matrix_v1"
    assert (
        str(raven_prompt_defaults["task_key"]).strip()
        == "raven_count_progression_label_query"
    )
    assert "query_id_weights" not in raven_generation_defaults
    assert sorted(raven_generation_defaults["scene_variant_weights"].keys()) == [
        "raven_card",
        "raven_outline",
        "raven_strip",
    ]
    assert int(raven_generation_defaults["option_count"]) == 4
    assert int(raven_generation_defaults["count_min"]) == 1
    assert int(raven_generation_defaults["count_max"]) == 8
    assert int(raven_rendering_defaults["canvas_width"]) > 0
    assert int(raven_rendering_defaults["cell_size_px"]) > 0


def test_puzzles_rubiks_net_scene_defaults_loaded() -> None:
    cfg = get_scene_defaults("puzzles", "rubiks_net")
    task_id = "task_puzzles__rubiks_net__rubiks_move_result_label"
    generation_defaults, rendering_defaults, prompt_defaults = (
        split_generation_rendering_prompt_defaults(cfg, task_id=task_id)
    )
    assert str(prompt_defaults["scene_key"]).strip() == "rubiks_net"
    assert str(prompt_defaults["bundle_id"]).strip() == "puzzles_rubiks_net_v1"
    assert str(prompt_defaults["task_key"]).strip() == "rubiks_move_result_label_query"
    assert "query_id_weights" not in generation_defaults
    assert sorted(generation_defaults["scene_variant_weights"].keys()) == [
        "classic_net",
        "cool_net",
        "paper_net",
    ]
    assert int(generation_defaults["option_count"]) == 4
    assert int(generation_defaults["post_move_sequence_count_min"]) == 1
    assert int(generation_defaults["post_move_sequence_count_max"]) == 3
    assert int(rendering_defaults["canvas_width"]) == 1480
    assert int(rendering_defaults["canvas_height"]) == 780
    assert int(rendering_defaults["main_cell_size_px"]) > 0


def test_puzzles_voxel_cube_defaults_loaded() -> None:
    cfg = get_scene_defaults("puzzles", "voxel_cube")
    prompt_shared = cfg["prompt"]["shared"]
    assert str(prompt_shared["bundle_id"]).strip() == "puzzles_voxel_cube_v1"
    assert str(prompt_shared["scene_key"]).strip() == "voxel_cube"


def test_puzzles_color_gradient_scene_defaults_loaded() -> None:
    cfg = get_scene_defaults("puzzles", "color_gradient")
    generation_defaults, rendering_defaults, prompt_defaults = (
        split_generation_rendering_prompt_defaults(
            cfg,
            task_id="task_puzzles__color_gradient__color_gradient_violation_cell_label",
        )
    )
    assert str(prompt_defaults["scene_key"]).strip() == "color_gradient"
    assert str(prompt_defaults["bundle_id"]).strip() == "puzzles_color_gradient_v1"
    assert "query_id_weights" not in generation_defaults
    assert sorted(generation_defaults["grid_size_variant_weights"].keys()) == [
        "3x3",
        "4x4",
    ]
    assert sorted(generation_defaults["rule_variant_weights"].keys()) == [
        "column_hue_row_lightness",
        "column_hue_row_saturation",
        "row_hue_column_lightness",
    ]
    assert int(rendering_defaults["swatch_size_px"]) > 0

    completion_generation, _, completion_prompt = (
        split_generation_rendering_prompt_defaults(
            cfg,
            task_id="task_puzzles__color_gradient__color_gradient_completion_label",
        )
    )
    assert str(completion_prompt["bundle_id"]).strip() == "puzzles_color_gradient_v1"
    assert sorted(completion_generation["sequence_length_variant_weights"].keys()) == [
        "5_cell",
        "6_cell",
        "7_cell",
    ]
    assert sorted(completion_generation["option_count_variant_weights"].keys()) == [
        "4_options",
        "5_options",
        "6_options",
    ]


def test_puzzles_word_search_scene_defaults_loaded() -> None:
    cfg = get_scene_defaults("puzzles", "word_search")
    generation_defaults, rendering_defaults, prompt_defaults = (
        split_generation_rendering_prompt_defaults(
            cfg,
            task_id="task_puzzles__word_search__present_word_option_label",
        )
    )
    assert str(prompt_defaults["scene_key"]).strip() == "word_search"
    assert str(prompt_defaults["bundle_id"]).strip() == "puzzles_word_search_v1"
    assert "query_id_weights" not in generation_defaults
    assert int(generation_defaults["grid_size_min"]) == 4
    assert int(generation_defaults["grid_size_max"]) == 6
    assert int(generation_defaults["option_count_min"]) == 4
    assert int(generation_defaults["option_count_max"]) == 4
    assert int(rendering_defaults["cell_size_px"]) > 0


def test_puzzles_sheet_transform_scene_defaults_loaded() -> None:
    cfg = get_scene_defaults("puzzles", "sheet_transform")
    generation_defaults, rendering_defaults, prompt_defaults = (
        split_generation_rendering_prompt_defaults(
            cfg,
            task_id="task_puzzles__sheet_transform__fold_projection_result_label",
        )
    )
    assert str(prompt_defaults["scene_key"]).strip() == "sheet_transform"
    assert str(prompt_defaults["bundle_id"]).strip() == "puzzles_sheet_transform_v1"
    assert "query_id_weights" not in generation_defaults
    assert sorted(generation_defaults["fold_axis_weights"].keys()) == [
        "horizontal",
        "vertical",
    ]
    assert sorted(str(key) for key in generation_defaults["fold_count_weights"]) == [
        "1",
        "2",
    ]
    assert int(generation_defaults["option_count_min"]) == 4
    assert int(generation_defaults["option_count_max"]) == 4
    assert int(generation_defaults["mark_count_min"]) == 3
    assert int(generation_defaults["mark_count_max"]) == 5
    assert int(generation_defaults["cut_count_min"]) == 1
    assert int(generation_defaults["cut_count_max"]) == 2
    assert int(generation_defaults["grid_size_min"]) == 4
    assert int(generation_defaults["grid_size_max"]) == 5
    assert int(rendering_defaults["canvas_width"]) > 0
    assert int(rendering_defaults["canvas_height"]) == 780
    assert int(rendering_defaults["reference_panel_height_px"]) > 0
    assert sorted(
        (str(item) for item in rendering_defaults["cut_hole_shape_options"])
    ) == ["circle", "diamond", "rounded_square", "square"]
    assert sorted((str(item) for item in rendering_defaults["mark_shape_options"])) == [
        "circle",
        "diamond",
        "rounded_square",
        "square",
    ]


def test_puzzles_pipe_flow_scene_defaults_loaded() -> None:
    cfg = get_scene_defaults("puzzles", "pipe_flow")
    generation_defaults, rendering_defaults, prompt_defaults = (
        split_generation_rendering_prompt_defaults(
            cfg,
            task_id="task_puzzles__pipe_flow__pipe_flow_repair_tile_label",
        )
    )
    assert str(prompt_defaults["scene_key"]).strip() == "pipe_flow"
    assert str(prompt_defaults["bundle_id"]).strip() == "puzzles_pipe_flow_v1"
    assert "query_id_weights" not in generation_defaults
    assert sorted(generation_defaults["grid_size_variant_weights"].keys()) == [
        "5x5",
        "6x6",
        "7x7",
    ]
    assert sorted(generation_defaults["scene_variant_weights"].keys()) == [
        "circuit_trace",
        "industrial_conduit",
        "water_pipe",
    ]
    assert int(generation_defaults["candidate_count_min"]) == 4
    assert int(generation_defaults["candidate_count_max"]) == 4
    assert int(rendering_defaults["canvas_width"]) > 0
    assert int(rendering_defaults["canvas_width"]) == 760
    assert int(rendering_defaults["canvas_height"]) == 720

    _misrotated_generation, _misrotated_rendering, misrotated_prompt_defaults = (
        split_generation_rendering_prompt_defaults(
            cfg,
            task_id="task_puzzles__pipe_flow__misrotated_tile_label",
        )
    )
    assert (
        str(misrotated_prompt_defaults["task_key"]).strip()
        == "pipe_flow_misrotated_tile_label_query"
    )


def test_puzzles_polyomino_assembly_scene_defaults_loaded() -> None:
    cfg = get_scene_defaults("puzzles", "polyomino_assembly")
    generation_defaults, rendering_defaults, prompt_defaults = (
        split_generation_rendering_prompt_defaults(
            cfg,
            task_id="task_puzzles__polyomino_assembly__decomposition_pair_label",
        )
    )
    assert str(prompt_defaults["scene_key"]).strip() == "polyomino_assembly"
    assert (
        str(prompt_defaults["bundle_id"]).strip()
        == "puzzles_polyomino_assembly_v1"
    )
    assert str(prompt_defaults["task_key"]).strip() == "decomposition_pair_label_query"
    assert "query_id_weights" not in generation_defaults
    assert list(generation_defaults["scene_variants"]) == [
        "clean_table",
        "workbench_card",
        "outline_panel",
    ]
    assert int(generation_defaults["total_cell_count_min"]) == 6
    assert int(generation_defaults["total_cell_count_max"]) == 10
    assert int(rendering_defaults["canvas_width"]) > 0
    assert int(rendering_defaults["canvas_height"]) > 0

    composition_generation_defaults, _, composition_prompt_defaults = (
        split_generation_rendering_prompt_defaults(
            cfg,
            task_id="task_puzzles__polyomino_assembly__composition_result_label",
        )
    )
    assert (
        str(composition_prompt_defaults["bundle_id"]).strip()
        == "puzzles_polyomino_assembly_v1"
    )
    assert (
        str(composition_prompt_defaults["task_key"]).strip()
        == "composition_result_label_query"
    )
    assert "query_id_weights" not in composition_generation_defaults


def test_puzzles_voxel_cube_task_defaults_loaded() -> None:
    cfg = get_scene_defaults("puzzles", "voxel_cube")
    generation_defaults, rendering_defaults, prompt_defaults = (
        split_generation_rendering_prompt_defaults(
            cfg,
            task_id="task_puzzles__voxel_cube__cube_count",
        )
    )
    assert str(prompt_defaults["scene_key"]).strip() == "voxel_cube"
    assert str(prompt_defaults["task_key"]).strip() == "cube_count_query"
    assert "query_id_weights" not in generation_defaults
    assert int(generation_defaults["answer_min"]) == 4
    assert int(generation_defaults["answer_max"]) == 14
    assert int(rendering_defaults["canvas_width"]) == 760
    assert int(rendering_defaults["canvas_height"]) == 520
    assert int(rendering_defaults["cube_size_px"]) == 52


def test_puzzles_cyclic_order_scene_defaults_loaded() -> None:
    cfg = get_scene_defaults("puzzles", "cyclic_order")
    generation_defaults, rendering_defaults, prompt_defaults = (
        split_generation_rendering_prompt_defaults(
            cfg,
            task_id="task_puzzles__cyclic_order__cyclic_order_equivalent_label",
        )
    )
    assert str(prompt_defaults["scene_key"]).strip() == "cyclic_order"
    assert str(prompt_defaults["bundle_id"]).strip() == "puzzles_cyclic_order_v1"
    assert "query_id_weights" not in generation_defaults
    assert sorted(generation_defaults["token_render_style_weights"].keys()) == [
        "colored_beads",
        "colored_shape_tokens",
        "outline_shape_tokens",
        "shape_tokens",
        "symbol_badges",
    ]
    assert sorted(generation_defaults["scene_variant_weights"].keys()) == [
        "charm_card_grid",
        "necklace_board",
        "route_loop_diagram",
        "token_ring_outline",
    ]
    assert sorted(generation_defaults["loop_path_style_weights"].keys()) == [
        "beaded_string",
        "ellipse",
        "polygon_loop",
        "rounded_rect",
        "wavy_loop",
    ]
    assert sorted(generation_defaults["answer_option_label_weights"].keys()) == [
        "A",
        "B",
        "C",
        "D",
    ]
    assert int(generation_defaults["option_count_min"]) == 4
    assert int(generation_defaults["option_count_max"]) == 4
    assert int(generation_defaults["bead_count_min"]) == 4
    assert int(generation_defaults["bead_count_max"]) == 5
    assert float(generation_defaults["min_color_distance"]) == 50.0
    assert str(generation_defaults["color_distance_space"]) == "lab"
    assert int(rendering_defaults["canvas_width"]) > 0
    assert int(rendering_defaults["reference_panel_height_px"]) > 0
    assert int(rendering_defaults["option_image_width_px"]) > 0
    assert int(rendering_defaults["shape_bead_inset_px"]) == 2


def test_puzzles_maze_scene_defaults_loaded() -> None:
    cfg = get_scene_defaults("puzzles", "maze")
    generation_defaults, rendering_defaults, prompt_defaults = (
        split_generation_rendering_prompt_defaults(
            cfg,
            task_id="task_puzzles__maze__exit_reachability_label",
        )
    )
    assert str(prompt_defaults["scene_key"]).strip() == "maze"
    assert str(prompt_defaults["bundle_id"]).strip() == "puzzles_maze_v1"
    assert "query_id_weights" not in generation_defaults
    assert sorted(generation_defaults["scene_variant_weights"].keys()) == [
        "block_wall_maze",
        "classic_wall_maze",
        "paper_labyrinth_maze",
    ]
    assert int(generation_defaults["maze_rows_min"]) == 6
    assert int(generation_defaults["maze_rows_max"]) == 8
    assert int(generation_defaults["maze_cols_min"]) == 7
    assert int(generation_defaults["maze_cols_max"]) == 10
    assert int(generation_defaults["exit_count_min"]) == 4
    assert int(generation_defaults["exit_count_max"]) == 4
    assert int(generation_defaults["nearest_exit_min_gap_edges"]) == 2
    assert int(rendering_defaults["canvas_width"]) == 1200
    assert int(rendering_defaults["canvas_height"]) == 900
    assert int(rendering_defaults["exit_marker_radius_px"]) == 27


def test_puzzles_nonogram_scene_defaults_loaded() -> None:
    cfg = get_scene_defaults("puzzles", "nonogram")
    generation_defaults, rendering_defaults, prompt_defaults = (
        split_generation_rendering_prompt_defaults(
            cfg,
            task_id="task_puzzles__nonogram__line_completion_label",
        )
    )
    assert str(prompt_defaults["scene_key"]).strip() == "nonogram"
    assert str(prompt_defaults["bundle_id"]).strip() == "puzzles_nonogram_v1"
    assert "query_id_weights" not in generation_defaults
    assert list(generation_defaults["option_count_choices"]) == [4]
    assert sorted(generation_defaults["scene_variant_weights"].keys()) == [
        "nonogram_blueprint",
        "nonogram_card",
        "nonogram_classic",
    ]
    assert int(generation_defaults["grid_rows_min"]) == 3
    assert int(generation_defaults["grid_rows_max"]) == 5
    assert int(generation_defaults["grid_cols_min"]) == 3
    assert int(generation_defaults["grid_cols_max"]) == 5
    assert int(rendering_defaults["canvas_width"]) == 980
    assert int(rendering_defaults["canvas_height"]) == 600


def test_puzzles_cell_board_scene_defaults_loaded() -> None:
    cfg = get_scene_defaults("puzzles", "cell_board")
    task_overrides = cfg["generation"]["task_overrides"]
    assert sorted(task_overrides) == [
        "task_puzzles__cell_board__largest_component_size",
        "task_puzzles__cell_board__reachable_region_size",
        "task_puzzles__cell_board__shortest_path_length_value",
        "task_puzzles__cell_board__symmetry_violation_count",
    ]
    generation_shared = cfg["generation"]["shared"]
    assert int(generation_shared["rows_min"]) == 5
    assert int(generation_shared["rows_max"]) == 8
    assert int(generation_shared["cols_min"]) == 5
    assert int(generation_shared["cols_max"]) == 8
    rendering_shared = cfg["rendering"]["shared"]
    assert int(rendering_shared["short_side_px_min"]) >= 28
    assert int(rendering_shared["short_side_px_max"]) >= int(
        rendering_shared["short_side_px_min"]
    )
    assert float(rendering_shared["aspect_ratio_min"]) == pytest.approx(1.0)
    assert float(rendering_shared["aspect_ratio_max"]) == pytest.approx(1.25)
    assert str(cfg["prompt"]["shared"]["bundle_id"]).strip() == "puzzles_cell_board_v1"
    assert str(cfg["prompt"]["shared"]["scene_key"]).strip() == "cell_board"

    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_puzzles__cell_board__reachable_region_size",
    )
    assert int(generation["answer_min"]) == 1
    assert int(generation["answer_max"]) == 8
    assert float(rendering["outer_padding_fraction_min"]) > 0.0
    assert str(prompt["bundle_id"]).strip() == "puzzles_cell_board_v1"
    assert str(prompt["scene_key"]).strip() == "cell_board"
