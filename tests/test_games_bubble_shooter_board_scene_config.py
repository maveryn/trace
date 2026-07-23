"""Config regression tests for games Bubble-shooter defaults."""

from __future__ import annotations

from trace_tasks.core.scene_config import get_scene_defaults, resolve_scene_section_defaults


def test_games_bubble_shooter_defaults_expose_scene_query_answer_and_style_axes() -> (
    None
):
    cfg = get_scene_defaults("games", "bubble_shooter")
    pop_generation = resolve_scene_section_defaults(
        cfg,
        "generation",
        task_id="task_games__bubble_shooter__pop_count",
    )
    drop_generation = resolve_scene_section_defaults(
        cfg,
        "generation",
        task_id="task_games__bubble_shooter__drop_count",
    )
    color_generation = resolve_scene_section_defaults(
        cfg,
        "generation",
        task_id="task_games__bubble_shooter__pop_color_label",
    )
    target_generation = resolve_scene_section_defaults(
        cfg,
        "generation",
        task_id="task_games__bubble_shooter__pop_target_label",
    )
    rendering = resolve_scene_section_defaults(
        cfg,
        "rendering",
        task_id="task_games__bubble_shooter__pop_count",
    )
    prompt = resolve_scene_section_defaults(
        cfg,
        "prompt",
        task_id="task_games__bubble_shooter__pop_count",
    )

    assert "query_id_weights" not in pop_generation
    assert "balanced_query_id_sampling" not in pop_generation
    assert set(pop_generation["scene_variant_weights"].keys()) == {
        "open_pack",
        "dense_pack",
    }
    assert set(pop_generation["style_variant_weights"].keys()) == {
        "classic",
        "pastel",
        "neon",
        "paper",
        "arcade",
    }
    assert pop_generation["row_count_support"] == [7]
    assert pop_generation["col_count_support"] == [8, 9, 10]
    assert pop_generation["pop_count_support"] == [0, 2, 3, 4, 5]
    assert drop_generation["row_count_support"] == [7]
    assert drop_generation["drop_count_support"] == [0, 1, 2, 3, 4]
    assert color_generation["row_count_support"] == [7, 8, 9]
    assert color_generation["option_count_support"] == [4, 5, 6]
    assert target_generation["row_count_support"] == [7, 8, 9]
    assert target_generation["positive_pop_count_support"] == [2, 3, 4, 5]
    assert target_generation["pop_target_label_support"] == ["A", "B", "C", "D"]
    assert rendering["canvas_width"] == 980
    assert rendering["canvas_height"] == 820
    assert rendering["dynamic_canvas_size_enabled"] is True
    assert str(prompt["bundle_id"]) == "games_bubble_shooter_v1"
    assert {"bundle_id", "scene_key", "task_key"}.issubset(set(prompt))
    assert not any(
        "answer_hint" in key or "annotation_hint" in key or "json_example" in key
        for key in prompt
    )
