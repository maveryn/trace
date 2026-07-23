"""Contracts for migrated balance-scale puzzle tasks."""

from __future__ import annotations

from trace_tasks.tasks import TASK_REGISTRY
from trace_tasks.tasks.puzzles.balance_scale.equivalent_object_count_value import (
    QUERY_ID as EQUIVALENT_QUERY_ID,
)
from trace_tasks.tasks.puzzles.balance_scale.equivalent_object_count_value import (
    TASK_ID as EQUIVALENT_OBJECT_COUNT_TASK_ID,
)
from trace_tasks.tasks.puzzles.balance_scale.equivalent_object_count_value import (
    PuzzlesBalanceScaleEquivalentObjectCountValueTask,
)
from trace_tasks.tasks.puzzles.balance_scale.missing_object_weight_value import (
    QUERY_ID as MISSING_QUERY_ID,
)
from trace_tasks.tasks.puzzles.balance_scale.missing_object_weight_value import (
    TASK_ID as MISSING_OBJECT_WEIGHT_TASK_ID,
)
from trace_tasks.tasks.puzzles.balance_scale.missing_object_weight_value import (
    PuzzlesBalanceScaleMissingObjectWeightValueTask,
)
from trace_tasks.tasks.puzzles.balance_scale.query_side_relation_label import (
    OPTION_LABELS as RELATION_OPTION_LABELS,
)
from trace_tasks.tasks.puzzles.balance_scale.query_side_relation_label import (
    QUERY_ID as RELATION_QUERY_ID,
)
from trace_tasks.tasks.puzzles.balance_scale.query_side_relation_label import (
    RELATION_BY_OPTION,
)
from trace_tasks.tasks.puzzles.balance_scale.query_side_relation_label import (
    TASK_ID as QUERY_SIDE_RELATION_TASK_ID,
)
from trace_tasks.tasks.puzzles.balance_scale.query_side_relation_label import (
    NEUTRAL_FREE_LABEL_FAMILIES,
    TARGET_RELATIONS,
    PuzzlesBalanceScaleQuerySideRelationLabelTask,
)
from trace_tasks.tasks.puzzles.balance_scale.shared.constraints import UNKNOWN_LABELS
from trace_tasks.tasks.puzzles.balance_scale.shared.rules import (
    expanded_item_count,
    expressions_match,
    unique_equivalent_counts,
    unique_order_signatures,
    unique_target_values,
)
from trace_tasks.tasks.puzzles.balance_scale.shared.state import SCENE_ID
from trace_tasks.tasks.puzzles.balance_scale.weight_order_label import (
    OPTION_LABELS,
    QUERY_ID as WEIGHT_QUERY_ID,
)
from trace_tasks.tasks.puzzles.balance_scale.weight_order_label import (
    TASK_ID as WEIGHT_ORDER_TASK_ID,
)
from trace_tasks.tasks.puzzles.balance_scale.weight_order_label import (
    WEIGHT_SUPPORT,
    PuzzlesBalanceScaleWeightOrderLabelTask,
)

TASK_ID = MISSING_OBJECT_WEIGHT_TASK_ID
QUERY_ID = MISSING_QUERY_ID


def _assert_bbox_inside_image(output, bbox) -> None:
    """Check one bbox is valid in the generated image."""

    assert len(bbox) == 4
    assert 0 <= float(bbox[0]) < float(bbox[2]) <= output.image.size[0]
    assert 0 <= float(bbox[1]) < float(bbox[3]) <= output.image.size[1]


def _assert_bbox_set_inside_image(output) -> None:
    """Check every bbox in a bbox-set annotation is valid."""

    for bbox in output.annotation_gt.value:
        _assert_bbox_inside_image(output, bbox)


def _entity_bbox(output, entity_id: str) -> list[float]:
    """Return one rendered entity bbox by id."""

    entities = output.trace_payload["scene_ir"]["entities"]
    for entity in entities:
        if str(entity.get("entity_id")) == str(entity_id):
            return list(entity["bbox_px"])
    raise AssertionError(f"missing rendered entity {entity_id!r}")


def _assert_three_balance_panels(execution) -> None:
    """Every redesigned balance task uses three readable scale panels."""

    assert execution["object_labels"] == list(UNKNOWN_LABELS)
    assert len(execution["panels"]) == 3
    for panel in execution["panels"]:
        assert 1 <= expanded_item_count(panel["left_terms"]) <= 4
        assert 1 <= expanded_item_count(panel["right_terms"]) <= 4
        assert not expressions_match(panel["left_terms"], panel["right_terms"])


def _direct_single_unknown_labels(execution) -> set[str]:
    """Return labels with directly readable one-object numeric scales."""

    labels: set[str] = set()
    for panel in execution["panels"]:
        for object_side, numeric_side in (
            (panel["left_terms"], panel["right_terms"]),
            (panel["right_terms"], panel["left_terms"]),
        ):
            object_terms = [
                term for term in object_side if str(term["kind"]) == "object"
            ]
            object_side_numbers = [
                term for term in object_side if str(term["kind"]) == "numeric"
            ]
            other_side_objects = [
                term for term in numeric_side if str(term["kind"]) == "object"
            ]
            other_side_numbers = [
                term for term in numeric_side if str(term["kind"]) == "numeric"
            ]
            if (
                len(object_terms) == 1
                and int(object_terms[0]["count"]) == 1
                and object_side_numbers
                and not other_side_objects
                and other_side_numbers
            ):
                labels.add(str(object_terms[0]["object_label"]))
    return labels


def _numeric_values(terms) -> set[int]:
    """Return visible numeric token values on one side."""

    return {int(term["value"]) for term in terms if str(term["kind"]) == "numeric"}


def test_balance_scale_task_is_registered() -> None:
    assert (
        TASK_REGISTRY[MISSING_OBJECT_WEIGHT_TASK_ID]
        is PuzzlesBalanceScaleMissingObjectWeightValueTask
    )
    assert (
        TASK_REGISTRY[EQUIVALENT_OBJECT_COUNT_TASK_ID]
        is PuzzlesBalanceScaleEquivalentObjectCountValueTask
    )
    assert (
        TASK_REGISTRY[WEIGHT_ORDER_TASK_ID] is PuzzlesBalanceScaleWeightOrderLabelTask
    )
    assert (
        TASK_REGISTRY[QUERY_SIDE_RELATION_TASK_ID]
        is PuzzlesBalanceScaleQuerySideRelationLabelTask
    )


def test_balance_scale_task_emits_public_contract() -> None:
    task = PuzzlesBalanceScaleMissingObjectWeightValueTask()
    out = task.generate(2026060401, params={}, max_attempts=120)
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.scene_id == SCENE_ID
    assert out.query_id == QUERY_ID
    assert out.answer_gt.type == "integer"
    assert 1 <= int(out.answer_gt.value) <= 20
    assert out.annotation_gt.type == "bbox"
    assert sorted(out.prompt_variants) == ["answer_and_annotation", "answer_only"]
    assert trace["query_spec"]["params"]["query_id"] == QUERY_ID
    assert (
        trace["query_spec"]["params"]["prompt_query_key"]
        == "missing_object_weight_value"
    )
    assert trace["render_spec"]["scene_id"] == SCENE_ID
    assert trace["render_spec"]["text_style"]["font"]["font_family"]
    assert trace["render_map"]["annotation_source"] == "annotation_bbox_px"
    assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert execution["scene_id"] == SCENE_ID
    assert execution["query_id"] == QUERY_ID
    assert execution["supporting_role_item_ids"] == {"target_box": "missing_value_box"}
    _assert_three_balance_panels(execution)
    _assert_bbox_inside_image(out, out.annotation_gt.value)


def test_equivalent_object_count_task_emits_public_contract() -> None:
    task = PuzzlesBalanceScaleEquivalentObjectCountValueTask()
    out = task.generate(2026060501, params={}, max_attempts=120)
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.scene_id == SCENE_ID
    assert out.query_id == EQUIVALENT_QUERY_ID
    assert out.answer_gt.type == "integer"
    assert 2 <= int(out.answer_gt.value) <= 8
    assert out.annotation_gt.type == "bbox"
    assert trace["query_spec"]["params"]["query_id"] == EQUIVALENT_QUERY_ID
    assert (
        trace["query_spec"]["params"]["prompt_query_key"]
        == "equivalent_object_count_value"
    )
    assert trace["render_spec"]["scene_id"] == SCENE_ID
    assert trace["render_map"]["annotation_source"] == "annotation_bbox_px"
    assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert execution["scene_id"] == SCENE_ID
    assert execution["query_id"] == EQUIVALENT_QUERY_ID
    assert execution["supporting_role_item_ids"] == {"target_box": "missing_count_box"}
    _assert_three_balance_panels(execution)
    _assert_bbox_inside_image(out, out.annotation_gt.value)


def test_weight_order_task_emits_public_contract() -> None:
    task = PuzzlesBalanceScaleWeightOrderLabelTask()
    out = task.generate(
        2026060601,
        params={"query_id": WEIGHT_QUERY_ID},
        max_attempts=120,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.scene_id == SCENE_ID
    assert out.query_id == WEIGHT_QUERY_ID
    assert out.answer_gt.type == "string"
    assert str(out.answer_gt.value) in OPTION_LABELS
    assert out.annotation_gt.type == "bbox"
    assert trace["query_spec"]["params"]["query_id"] == WEIGHT_QUERY_ID
    assert trace["query_spec"]["params"]["answer_type"] == "string"
    assert trace["query_spec"]["params"]["target_cue_mode"] == "query_row_only"
    assert trace["render_spec"]["scene_id"] == SCENE_ID
    assert trace["render_map"]["annotation_source"] == "annotation_bbox_px"
    assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert execution["supporting_role_item_ids"] == {
        "selected_option": f"option_{out.answer_gt.value}"
    }
    assert len(execution["order_options"]) == 4
    selected = [
        option
        for option in execution["order_options"]
        if option["option_label"] == out.answer_gt.value
    ][0]
    assert selected["order_text"] == execution["correct_order"]
    assert out.annotation_gt.value == _entity_bbox(
        out,
        f"option_{out.answer_gt.value}",
    )
    _assert_three_balance_panels(execution)
    _assert_bbox_inside_image(out, out.annotation_gt.value)


def test_query_side_relation_task_emits_public_contract() -> None:
    task = PuzzlesBalanceScaleQuerySideRelationLabelTask()
    out = task.generate(2026060701, params={}, max_attempts=120)
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.scene_id == SCENE_ID
    assert out.query_id == RELATION_QUERY_ID
    assert out.answer_gt.type == "string"
    assert str(out.answer_gt.value) in RELATION_OPTION_LABELS
    assert out.annotation_gt.type == "bbox"
    assert trace["query_spec"]["params"]["query_id"] == RELATION_QUERY_ID
    assert trace["query_spec"]["params"]["answer_type"] == "string"
    assert (
        trace["query_spec"]["params"]["prompt_query_key"]
        == "query_side_relation_label"
    )
    assert trace["render_spec"]["scene_id"] == SCENE_ID
    assert trace["render_map"]["annotation_source"] == "annotation_bbox_px"
    assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert execution["scene_id"] == SCENE_ID
    assert execution["query_id"] == RELATION_QUERY_ID
    assert execution["supporting_role_item_ids"] == {
        "selected_option": f"option_{out.answer_gt.value}"
    }
    assert len(execution["relation_options"]) == 4
    assert str(execution["target_relation"]) == RELATION_BY_OPTION[out.answer_gt.value]
    _assert_three_balance_panels(execution)
    _assert_bbox_inside_image(out, out.annotation_gt.value)


def test_balance_scale_equations_are_balanced_and_unique() -> None:
    task = PuzzlesBalanceScaleMissingObjectWeightValueTask()
    out = task.generate(2026060410, params={}, max_attempts=120)
    execution = out.trace_payload["execution_trace"]
    support = [int(value) for value in execution["target_answer_support"]]

    assert int(out.answer_gt.value) == int(execution["answer_value"])
    _assert_three_balance_panels(execution)
    assert str(execution["target_label"]) in _direct_single_unknown_labels(execution)
    for panel in execution["panels"]:
        assert int(panel["left_total"]) == int(panel["right_total"])

    unique_values = unique_target_values(
        equations=execution["equations"],
        labels=execution["object_labels"],
        target_label=str(execution["target_label"]),
        support=support,
    )
    assert unique_values == [int(out.answer_gt.value)]


def test_equivalent_object_count_equations_are_balanced_and_unique() -> None:
    task = PuzzlesBalanceScaleEquivalentObjectCountValueTask()
    out = task.generate(2026060510, params={}, max_attempts=120)
    execution = out.trace_payload["execution_trace"]
    support = [int(value) for value in execution["target_answer_support"]]
    weight_support = [int(value) for value in execution["object_weight_support"]]

    assert int(out.answer_gt.value) == int(execution["answer_value"])
    _assert_three_balance_panels(execution)
    direct_labels = _direct_single_unknown_labels(execution)
    assert str(execution["source_label"]) in direct_labels
    assert str(execution["repeated_label"]) in direct_labels
    for panel in execution["panels"]:
        assert int(panel["left_total"]) == int(panel["right_total"])

    unique_counts = unique_equivalent_counts(
        equations=execution["equations"],
        labels=execution["object_labels"],
        source_label=str(execution["source_label"]),
        repeated_label=str(execution["repeated_label"]),
        count_support=support,
        weight_support=weight_support,
    )
    assert unique_counts == [int(out.answer_gt.value)]


def test_weight_order_comparisons_imply_one_option() -> None:
    task = PuzzlesBalanceScaleWeightOrderLabelTask()
    out = task.generate(2026060610, params={}, max_attempts=120)
    execution = out.trace_payload["execution_trace"]

    _assert_three_balance_panels(execution)
    unique_orders = unique_order_signatures(
        comparisons=execution["comparisons"],
        labels=execution["object_labels"],
        support=WEIGHT_SUPPORT,
    )
    assert unique_orders == [execution["correct_order"]]
    assert len({option["order_text"] for option in execution["order_options"]}) == 4
    for comparison in execution["comparisons"]:
        left_numbers = _numeric_values(comparison["left_terms"])
        right_numbers = _numeric_values(comparison["right_terms"])
        assert not left_numbers.intersection(right_numbers)
        assert str(comparison["comparison_family"]) in {
            "aggregate_shared_context",
            "direct_equality",
            "offset_balance",
            "offset_inequality",
            "shared_object_context",
        }


def test_query_side_relation_targets_are_valid_and_roughly_balanced() -> None:
    task = PuzzlesBalanceScaleQuerySideRelationLabelTask()
    counts = {relation: 0 for relation in TARGET_RELATIONS}

    for offset in range(400):
        out = task.generate(2026060700 + offset, params={}, max_attempts=120)
        execution = out.trace_payload["execution_trace"]
        target_relation = str(execution["target_relation"])
        relation_outcomes = list(execution["query_relation_outcomes"])

        counts[target_relation] += 1
        assert target_relation == RELATION_BY_OPTION[out.answer_gt.value]
        assert len(execution["relation_options"]) == 4
        if target_relation == "not_determined":
            assert len(relation_outcomes) > 1
        else:
            assert relation_outcomes == [target_relation]

    assert set(counts) == set(TARGET_RELATIONS)
    assert all(70 <= count <= 130 for count in counts.values())


def test_query_side_relation_ambiguous_references_are_varied() -> None:
    task = PuzzlesBalanceScaleQuerySideRelationLabelTask()
    families: set[str] = set()
    free_labels: set[str] = set()
    side_signatures: set[tuple[str, str]] = set()

    for offset in range(400):
        seed = 2026060700 + offset
        out = task.generate(seed, params={}, max_attempts=120)
        execution = out.trace_payload["execution_trace"]
        if str(execution["target_relation"]) != "not_determined":
            continue
        neutral_equation = execution["equations"][2]
        family = str(neutral_equation["equation_family"]).replace(
            "neutral_free_label_",
            "",
        )
        free_label = str(execution["query_left_terms"][0]["object_label"])
        families.add(family)
        free_labels.add(free_label)
        side_signatures.add(
            (
                str(neutral_equation["left_terms"]),
                str(neutral_equation["right_terms"]),
            )
        )

    assert families == set(NEUTRAL_FREE_LABEL_FAMILIES)
    assert free_labels == set(UNKNOWN_LABELS)
    assert len(side_signatures) >= len(NEUTRAL_FREE_LABEL_FAMILIES)


def test_balance_scale_task_is_deterministic() -> None:
    task = PuzzlesBalanceScaleMissingObjectWeightValueTask()
    params = {
        "scene_variant": "balance_card",
        "target_cue_mode": "query_row_and_highlight",
    }
    out_a = task.generate(2026060499, params=params, max_attempts=120)
    out_b = task.generate(2026060499, params=params, max_attempts=120)

    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt == out_b.answer_gt
    assert out_a.annotation_gt == out_b.annotation_gt
    assert (
        out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    )
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_equivalent_object_count_task_is_deterministic() -> None:
    task = PuzzlesBalanceScaleEquivalentObjectCountValueTask()
    params = {
        "scene_variant": "balance_card",
        "target_cue_mode": "query_row_and_highlight",
    }
    out_a = task.generate(2026060599, params=params, max_attempts=120)
    out_b = task.generate(2026060599, params=params, max_attempts=120)

    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt == out_b.answer_gt
    assert out_a.annotation_gt == out_b.annotation_gt
    assert (
        out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    )
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_weight_order_task_is_deterministic() -> None:
    task = PuzzlesBalanceScaleWeightOrderLabelTask()
    params = {"query_id": WEIGHT_QUERY_ID, "scene_variant": "balance_card"}
    out_a = task.generate(2026060699, params=params, max_attempts=120)
    out_b = task.generate(2026060699, params=params, max_attempts=120)

    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt == out_b.answer_gt
    assert out_a.annotation_gt == out_b.annotation_gt
    assert (
        out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    )
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_query_side_relation_task_is_deterministic() -> None:
    task = PuzzlesBalanceScaleQuerySideRelationLabelTask()
    params = {"query_id": RELATION_QUERY_ID, "scene_variant": "balance_card"}
    out_a = task.generate(2026060799, params=params, max_attempts=120)
    out_b = task.generate(2026060799, params=params, max_attempts=120)

    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt == out_b.answer_gt
    assert out_a.annotation_gt == out_b.annotation_gt
    assert (
        out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    )
    assert out_a.image.tobytes() == out_b.image.tobytes()
