"""Contract tests for games solitaire-tableau tasks."""

from __future__ import annotations

import json
from pathlib import Path

import trace_tasks.tasks  # noqa: F401
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.registry import create_task
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


def _card_color(suit_name: str) -> str:
    return "red" if str(suit_name) in {"hearts", "diamonds"} else "black"


def _is_legal_tableau(source: dict, target: dict) -> bool:
    return int(target["rank_value"]) == int(source["rank_value"]) + 1 and _card_color(str(target["suit_name"])) != _card_color(str(source["suit_name"]))


def _is_legal_foundation(source: dict, foundation: dict) -> bool:
    return str(source["suit_name"]) == str(foundation["suit_name"]) and int(source["rank_value"]) == int(foundation["top_rank_value"]) + 1


def test_games_solitaire_defaults_expose_scene_axes_and_prompt_bundle() -> None:
    cfg = get_scene_defaults("games", "solitaire")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__solitaire__move_legality_label",
    )

    assert set(generation["scene_variant_weights"].keys()) == {"klondike_tableau", "freecell_tableau"}
    assert set(generation["style_variant_weights"].keys()) == {
        "classic_cards",
        "ivory_table",
        "casino_felt",
        "slate_cards",
        "paper_tableau",
    }
    assert list(generation["move_option_count_support"]) == [4]
    assert list(generation["card_option_count_support"]) == [4, 6]
    assert list(generation["cascade_depth_support"]) == [1, 2, 3, 4, 5, 6]
    assert list(generation["foundation_ready_target_answer_support"]) == [0, 1, 2, 3, 4]
    assert list(generation["column_card_count_target_answer_support"]) == [1, 2, 3, 4, 5, 6]
    assert list(generation["tableau_movable_card_count_target_answer_support"]) == [0, 1, 2, 3, 4]
    assert int(rendering["canvas_width"]) == 1060
    assert int(rendering["card_width_px"]) > 0
    assert str(prompt["bundle_id"]) == "games_solitaire_v1"
    bundle = json.loads(Path("src/trace_tasks/resources/prompts/games/solitaire/games_solitaire_v1.json").read_text(encoding="utf-8"))
    assert "foundation" in str(bundle["code_prompt_defaults"]["foundation_rule_text"])


def test_games_solitaire_prompt_bundle_has_five_queries() -> None:
    bundle = json.loads(Path("src/trace_tasks/resources/prompts/games/solitaire/games_solitaire_v1.json").read_text(encoding="utf-8"))
    assert bundle["schema_version"] == "v1"
    assert set(bundle["templates"]["query"].keys()) == {
        "move_legality_label",
        "foundation_ready_count",
        "tableau_movable_card_count",
        "column_card_count",
        "cascade_card_at_depth_label",
    }
    assert bundle["required_slots_by_key"]["query:move_legality_label"] == [
        "tableau_rule_text",
        "foundation_rule_text",
    ]


def test_games_solitaire_move_legality_has_one_legal_option() -> None:
    out = create_task("task_games__solitaire__move_legality_label").generate(81331, params={}, max_attempts=200)
    execution = out.trace_payload["execution_trace"]
    cards = {str(spec["card_id"]): dict(spec) for spec in execution["card_specs"]}
    foundations = {str(spec["foundation_id"]): dict(spec) for spec in execution["foundation_specs"]}

    legal_labels = []
    for option in execution["move_options"]:
        source = cards[str(option["source_card_id"])]
        target_id = str(option["target_id"])
        if target_id in cards:
            legal = _is_legal_tableau(source, cards[target_id])
        else:
            legal = _is_legal_foundation(source, foundations[target_id])
        if legal:
            legal_labels.append(str(option["label"]))

    assert out.answer_gt.type == "option_letter"
    assert len(execution["move_options"]) == 4
    assert all("Col " in str(option["source_label"]) for option in execution["move_options"])
    assert all(" -> " in str(option["move"]) for option in execution["move_options"])
    assert legal_labels == [str(out.answer_gt.value)]
    assert execution["answer_option_label"] == out.answer_gt.value
    assert out.annotation_gt.type == "bbox_map"
    assert set(out.annotation_gt.value) == {"source_card", "target"}
    assert set(execution["annotation_entity_ids"]) == {
        execution["legal_source_id"],
        execution["legal_target_id"],
    }
    assert out.scene_id == "solitaire"
    assert out.query_id == "single"
    assert execution["prompt_query_key"] == "move_legality_label"
    assert out.trace_payload["projected_annotation"]["type"] == "bbox_map"
    assert out.trace_payload["render_spec"]["solitaire_tableau_style"]["style_variant"]
    assert out.trace_payload["render_spec"]["text_style"]["font_family"]


def test_games_solitaire_foundation_ready_count_matches_trace() -> None:
    out = create_task("task_games__solitaire__foundation_ready_count").generate(
        81341,
        params={"target_answer": 3},
        max_attempts=200,
    )
    execution = out.trace_payload["execution_trace"]

    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == 3
    assert len(execution["ready_card_ids"]) == 3
    assert int(execution["target_answer"]) == 3
    assert out.query_id == "single"
    assert execution["prompt_query_key"] == "foundation_ready_count"
    assert len(out.annotation_gt.value) == 3
    assert execution["annotation_entity_ids"] == execution["ready_card_ids"]
    assert all(card["badge_text"] is None for card in execution["card_specs"])


def test_games_solitaire_tableau_movable_card_count_matches_trace() -> None:
    out = create_task("task_games__solitaire__tableau_movable_card_count_value").generate(
        81346,
        params={"target_answer": 4},
        max_attempts=300,
    )
    execution = out.trace_payload["execution_trace"]
    cards = {str(spec["card_id"]): dict(spec) for spec in execution["card_specs"]}
    exposed_cards = [card for card in cards.values() if bool(card["is_exposed"])]
    movable_ids = [
        str(source["card_id"])
        for source in exposed_cards
        if any(
            str(target["card_id"]) != str(source["card_id"]) and _is_legal_tableau(source, target)
            for target in exposed_cards
        )
    ]

    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == 4
    assert out.query_id == "single"
    assert execution["prompt_query_key"] == "tableau_movable_card_count"
    assert execution["movable_card_ids"] == movable_ids
    assert execution["annotation_entity_ids"] == movable_ids
    assert out.annotation_gt.type == "bbox_set"
    assert len(out.annotation_gt.value) == 4
    assert out.trace_payload["render_map"]["marked_card_id"] is None
    assert out.trace_payload["render_map"]["foundation_bboxes_px"] == {}
    assert out.trace_payload["projected_annotation"]["type"] == "bbox_set"


def test_games_solitaire_column_card_count_matches_trace() -> None:
    out = create_task("task_games__solitaire__column_card_count_value").generate(
        81351,
        params={"target_answer": 4, "target_column": 3},
        max_attempts=200,
    )
    execution = out.trace_payload["execution_trace"]
    target_column_index = int(execution["target_column_index"])
    card_ids = [
        str(card["card_id"])
        for card in execution["card_specs"]
        if int(card["column_index"]) == target_column_index
    ]

    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == 4
    assert out.query_id == "single"
    assert execution["prompt_query_key"] == "column_card_count"
    assert int(execution["target_column_number"]) == 3
    assert execution["target_column_card_ids"] == card_ids
    assert execution["annotation_entity_ids"] == card_ids
    assert all(card["badge_text"] is None for card in execution["card_specs"])
    assert out.annotation_gt.type == "point_set"
    assert len(out.annotation_gt.value) == 4
    assert out.trace_payload["projected_annotation"]["type"] == "point_set"
    assert out.trace_payload["projected_annotation"]["point_set"] == out.annotation_gt.value


def test_games_solitaire_cascade_card_at_depth_label_matches_trace() -> None:
    out = create_task("task_games__solitaire__cascade_card_at_depth_label").generate(
        81361,
        params={"target_column": 2, "target_depth": 4, "option_count": 6, "answer_option_label": "D"},
        max_attempts=200,
    )
    execution = out.trace_payload["execution_trace"]
    target_card_id = str(execution["target_card_id"])
    target_card = next(card for card in execution["card_specs"] if str(card["card_id"]) == target_card_id)
    answer_option = next(option for option in execution["card_options"] if bool(option["is_answer"]))

    assert out.answer_gt.type == "option_letter"
    assert str(out.answer_gt.value) == "D"
    assert out.query_id == "single"
    assert execution["prompt_query_key"] == "cascade_card_at_depth_label"
    assert int(execution["target_column_number"]) == 2
    assert int(execution["target_depth"]) == 4
    assert str(execution["target_depth_ordinal"]) == "4th"
    assert int(target_card["column_index"]) == 1
    assert int(target_card["row_index"]) == 3
    assert str(answer_option["label"]) == "D"
    assert str(answer_option["card_label"]) == str(execution["target_card_label"])
    assert len(execution["card_options"]) == 6
    assert out.annotation_gt.type == "point"
    assert out.annotation_gt.value == out.trace_payload["render_map"]["entity_points_px"][target_card_id]
    assert out.trace_payload["projected_annotation"]["type"] == "point"
    assert out.trace_payload["projected_annotation"]["point"] == out.annotation_gt.value
