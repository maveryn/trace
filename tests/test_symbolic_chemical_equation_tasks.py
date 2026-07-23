"""Contract tests for symbolic chemical-equation tasks."""

from __future__ import annotations

from io import BytesIO

from trace_tasks.core.prompts import load_prompt_bundle
from trace_tasks.core.prompts.schema import REQUIRED_PROMPT_VARIANTS
from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks.registry import TASK_REGISTRY
from trace_tasks.tasks.symbolic.chemical_equation.balanced_option_label import (
    INTERNAL_QUERY_ID as OPTION_INTERNAL_QUERY_ID,
)
from trace_tasks.tasks.symbolic.chemical_equation.balanced_option_label import (
    TASK_ID as OPTION_TASK_ID,
)
from trace_tasks.tasks.symbolic.chemical_equation.balanced_option_label import (
    SymbolicChemicalEquationBalancedOptionTask,
)
from trace_tasks.tasks.symbolic.chemical_equation.missing_coefficient_value import (
    INTERNAL_QUERY_ID as MISSING_INTERNAL_QUERY_ID,
)
from trace_tasks.tasks.symbolic.chemical_equation.missing_coefficient_value import (
    TASK_ID as MISSING_TASK_ID,
)
from trace_tasks.tasks.symbolic.chemical_equation.missing_coefficient_value import (
    SymbolicChemicalEquationMissingCoefficientTask,
)
from trace_tasks.tasks.symbolic.chemical_equation.shared.rules import (
    is_balanced_coefficients,
    parse_formula,
    reaction_by_id,
)


def _png_bytes(output) -> bytes:
    buffer = BytesIO()
    output.image.save(buffer, format="PNG")
    return buffer.getvalue()


def _assert_bbox_in_image(bbox: list[float], *, width: int, height: int) -> None:
    assert len(bbox) == 4
    assert 0 <= float(bbox[0]) < float(bbox[2]) <= width
    assert 0 <= float(bbox[1]) < float(bbox[3]) <= height


def test_chemical_equation_tasks_are_registered_and_taxonomized() -> None:
    assert TASK_REGISTRY[MISSING_TASK_ID] is SymbolicChemicalEquationMissingCoefficientTask
    assert TASK_REGISTRY[OPTION_TASK_ID] is SymbolicChemicalEquationBalancedOptionTask
    for task_id in (MISSING_TASK_ID, OPTION_TASK_ID):
        task = TASK_REGISTRY[task_id]()
        taxonomy = resolve_task_taxonomy(task_id)
        assert task.domain == "symbolic"
        assert not hasattr(task, "scene_id")
        assert tuple(task.supported_query_ids) == (SINGLE_QUERY_ID,)
        assert taxonomy.domain == "symbolic"
        assert taxonomy.scene_id == "chemical_equation"
        assert taxonomy.source_scene_id == ""


def test_formula_parser_and_reaction_bank_balance() -> None:
    term = parse_formula("Fe2O3")
    assert term.atom_counts == {"Fe": 2, "O": 3}
    assert term.element_order == ("Fe", "O")
    for reaction_id in (
        "water_synthesis",
        "ammonia_synthesis",
        "propane_combustion",
        "phosphorus_oxide",
    ):
        reaction = reaction_by_id(reaction_id)
        assert is_balanced_coefficients(reaction, reaction.coefficients)


def test_missing_coefficient_contract() -> None:
    out = SymbolicChemicalEquationMissingCoefficientTask().generate(
        2026070301,
        params={
            "reaction_id": "phosphorus_oxide",
            "hidden_slot_index": 1,
            "scene_variant": "clean_lab",
        },
        max_attempts=8,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.scene_id == "chemical_equation"
    assert out.query_id == SINGLE_QUERY_ID
    assert trace["query_spec"]["internal_query_id"] == MISSING_INTERNAL_QUERY_ID
    assert execution["internal_query_id"] == MISSING_INTERNAL_QUERY_ID
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == 5
    assert execution["answer_value"] == 5
    assert out.annotation_gt.type == "bbox_set"
    assert execution["annotation_item_ids"] == [
        "coefficient_slot_2",
        "molecule_card_1",
        "molecule_card_2",
        "molecule_card_3",
    ]
    assert out.annotation_gt.value == [
        trace["render_map"]["item_bboxes_px"][item_id]
        for item_id in execution["annotation_item_ids"]
    ]
    assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
    metadata = execution["chemical_equation_metadata"]
    assert metadata["hidden_slot_number"] == 2
    assert metadata["reaction"]["balanced_coefficients"] == [1, 5, 2]

    width, height = out.image.size
    for bbox in out.annotation_gt.value:
        _assert_bbox_in_image(bbox, width=width, height=height)


def test_missing_coefficient_support_covers_one_through_five() -> None:
    answers = set()
    for answer in range(1, 6):
        out = SymbolicChemicalEquationMissingCoefficientTask().generate(
            2026070310 + answer,
            params={"answer_value": answer},
            max_attempts=20,
        )
        answers.add(int(out.answer_gt.value))
        assert int(out.answer_gt.value) == answer
        assert answer in out.trace_payload["query_spec"]["params"]["target_answer_support"]
    assert answers == {1, 2, 3, 4, 5}


def test_balanced_option_contract() -> None:
    out = SymbolicChemicalEquationBalancedOptionTask().generate(
        2026070321,
        params={
            "reaction_id": "propane_combustion",
            "correct_label": "C",
            "scene_variant": "worksheet",
        },
        max_attempts=8,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    metadata = execution["chemical_equation_metadata"]

    assert out.query_id == SINGLE_QUERY_ID
    assert trace["query_spec"]["internal_query_id"] == OPTION_INTERNAL_QUERY_ID
    assert execution["internal_query_id"] == OPTION_INTERNAL_QUERY_ID
    assert out.answer_gt.type == "string"
    assert out.answer_gt.value == "C"
    assert out.annotation_gt.type == "bbox"
    assert execution["annotation_item_ids"] == ["option_C"]
    assert out.annotation_gt.value == trace["render_map"]["option_bboxes_px"]["option_C"]
    assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert metadata["reaction"]["balanced_coefficients"] == [1, 5, 3, 4]

    option_records = metadata["options"]
    assert sorted(option_records.keys()) == ["A", "B", "C", "D"]
    assert option_records["C"]["balances_equation"] is True
    assert option_records["C"]["coefficients"] == [1, 5, 3, 4]
    assert [
        label
        for label, record in option_records.items()
        if bool(record["balances_equation"])
    ] == ["C"]
    assert len({tuple(record["coefficients"]) for record in option_records.values()}) == 4

    width, height = out.image.size
    _assert_bbox_in_image(out.annotation_gt.value, width=width, height=height)


def test_chemical_equation_generation_is_deterministic() -> None:
    params = {"reaction_id": "ammonia_synthesis", "hidden_slot_index": 1}
    first = SymbolicChemicalEquationMissingCoefficientTask().generate(
        2026070399,
        params=params,
        max_attempts=8,
    )
    second = SymbolicChemicalEquationMissingCoefficientTask().generate(
        2026070399,
        params=params,
        max_attempts=8,
    )
    assert first.prompt == second.prompt
    assert first.answer_gt.to_dict() == second.answer_gt.to_dict()
    assert first.annotation_gt.to_dict() == second.annotation_gt.to_dict()
    assert first.trace_payload["execution_trace"] == second.trace_payload["execution_trace"]
    assert _png_bytes(first) == _png_bytes(second)


def test_chemical_equation_prompt_bundle_supports_tasks() -> None:
    bundle = load_prompt_bundle(
        "symbolic",
        "chemical_equation",
        "symbolic_chemical_equation_v1",
    )
    assert "chemical_equation" in bundle.scene_templates
    assert (
        len(bundle.task_templates["missing_coefficient_value"])
        == REQUIRED_PROMPT_VARIANTS
    )
    assert (
        len(bundle.task_templates["balanced_option_label"])
        == REQUIRED_PROMPT_VARIANTS
    )
    assert not bundle.query_templates
    assert list(bundle.required_slots_by_key["scene:chemical_equation"]) == [
        "object_description"
    ]
    assert list(bundle.required_slots_by_key["task:missing_coefficient_value"]) == []
    assert list(bundle.required_slots_by_key["task:balanced_option_label"]) == []
