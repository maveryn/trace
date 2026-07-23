from __future__ import annotations

from io import BytesIO
from random import Random

import pytest

from trace_tasks.core.prompts import load_prompt_bundle
from trace_tasks.core.prompts.schema import REQUIRED_PROMPT_VARIANTS
from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks.symbolic.organic_structure.shared.rules import (
    ORGANIC_STRUCTURE_MAX_BOND_ORDER_COUNT,
    ORGANIC_STRUCTURE_MAX_RING_SIZE_COUNT,
    build_constrained_organic_ring_size_structure,
    build_constrained_organic_structure,
    organic_ring_item_ids,
    validate_organic_structure,
)
from trace_tasks.tasks.symbolic.organic_structure.bond_order_count import (
    TARGET_BOND_ORDERS,
    TASK_ID,
    SymbolicBondOrderCountTask,
)
from trace_tasks.tasks.symbolic.organic_structure.ring_size_count import (
    RING_SIZE_COUNT_TASK_ID,
    TARGET_RING_SIZES,
    SymbolicRingSizeCountTask,
)
from trace_tasks.tasks.registry import TASK_REGISTRY


def _png_bytes(output) -> bytes:
    buffer = BytesIO()
    output.image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_organic_structure_bond_order_task_is_registered_and_taxonomized() -> None:
    assert TASK_ID in TASK_REGISTRY
    taxonomy = resolve_task_taxonomy(TASK_ID)
    assert taxonomy.domain == "symbolic"
    assert taxonomy.scene_id == "organic_structure"
    assert taxonomy.source_scene_id == ""


def test_organic_structure_ring_size_task_is_registered_and_taxonomized() -> None:
    assert RING_SIZE_COUNT_TASK_ID in TASK_REGISTRY
    taxonomy = resolve_task_taxonomy(RING_SIZE_COUNT_TASK_ID)
    assert taxonomy.domain == "symbolic"
    assert taxonomy.scene_id == "organic_structure"
    assert taxonomy.source_scene_id == ""


def test_organic_structure_double_bond_count_contract() -> None:
    out = SymbolicBondOrderCountTask().generate(
        102,
        params={"target_bond_order": "double"},
        max_attempts=3,
    )
    trace = out.trace_payload
    bonds = trace["execution_trace"]["bonds"]
    expected_ids = [bond["item_id"] for bond in bonds if bond["bond_order"] == "double"]
    assert out.query_id == SINGLE_QUERY_ID
    assert trace["query_spec"]["internal_query_id"] == "bond_order_count"
    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "segment_set"
    assert out.answer_gt.value == len(expected_ids)
    assert len(out.annotation_gt.value) == len(expected_ids)
    assert trace["execution_trace"]["annotation_item_ids"] == expected_ids
    assert trace["projected_annotation"]["type"] == "segment_set"
    assert trace["projected_annotation"]["segment_set"] == out.annotation_gt.value
    for bond_id, point_pair in zip(expected_ids, out.annotation_gt.value):
        assert point_pair == trace["render_map"]["bond_segments_px"][str(bond_id)]
        assert len(point_pair) == 2
        assert all(len(point) == 2 for point in point_pair)
    assert trace["execution_trace"]["organic_metadata"]["text_label_policy"]
    assert trace["execution_trace"]["organic_metadata"]["constraint_report"]["max_valence"] <= 4
    assert "double" in out.prompt


def test_organic_structure_triple_bond_count_contract() -> None:
    out = SymbolicBondOrderCountTask().generate(
        209,
        params={"target_bond_order": "triple"},
        max_attempts=3,
    )
    trace = out.trace_payload
    bonds = trace["execution_trace"]["bonds"]
    expected_ids = [bond["item_id"] for bond in bonds if bond["bond_order"] == "triple"]
    assert out.query_id == SINGLE_QUERY_ID
    assert trace["query_spec"]["internal_query_id"] == "bond_order_count"
    assert out.annotation_gt.type == "segment_set"
    assert out.answer_gt.value == len(expected_ids)
    assert len(out.annotation_gt.value) == len(expected_ids)
    assert trace["execution_trace"]["annotation_item_ids"] == expected_ids
    assert trace["projected_annotation"]["type"] == "segment_set"
    assert trace["projected_annotation"]["segment_set"] == out.annotation_gt.value
    assert set(trace["execution_trace"]["organic_metadata"]["target_bond_order_support"]) == set(TARGET_BOND_ORDERS)
    assert "not enforced" not in trace["execution_trace"]["organic_metadata"]["chemical_validity_policy"]
    assert "triple" in out.prompt


def test_organic_structure_ring_size_count_contract() -> None:
    out = SymbolicRingSizeCountTask().generate(
        811,
        params={"target_ring_size": 6, "answer_value": 2},
        max_attempts=3,
    )
    trace = out.trace_payload
    rings = trace["execution_trace"]["rings"]
    expected_ids = [ring["item_id"] for ring in rings if ring["ring_size"] == 6]
    assert out.query_id == SINGLE_QUERY_ID
    assert trace["query_spec"]["internal_query_id"] == "ring_size_count"
    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "bbox_set"
    assert out.answer_gt.value == len(expected_ids) == 2
    assert len(out.annotation_gt.value) == len(expected_ids)
    assert trace["execution_trace"]["annotation_item_ids"] == expected_ids
    assert trace["execution_trace"]["matching_ring_item_ids"] == expected_ids
    assert trace["projected_annotation"]["type"] == "bbox_set"
    assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
    for ring_id, bbox in zip(expected_ids, out.annotation_gt.value):
        assert bbox == trace["render_map"]["ring_bboxes_px"][str(ring_id)]
        assert len(bbox) == 4
    assert trace["execution_trace"]["target_ring_size"] == 6
    assert trace["execution_trace"]["organic_metadata"]["ring_layout_policy"]
    assert "hexagonal" in out.prompt


def test_organic_structure_ring_size_max_count_contract() -> None:
    out = SymbolicRingSizeCountTask().generate(
        910,
        params={"target_ring_size": 5, "answer_value": 5},
        max_attempts=3,
    )
    expected_ids = [
        ring["item_id"]
        for ring in out.trace_payload["execution_trace"]["rings"]
        if ring["ring_size"] == 5
    ]
    assert out.query_id == SINGLE_QUERY_ID
    assert out.answer_gt.value == 5
    assert out.annotation_gt.type == "bbox_set"
    assert len(out.annotation_gt.value) == 5
    assert out.trace_payload["execution_trace"]["annotation_item_ids"] == expected_ids
    assert len(expected_ids) == 5


def test_organic_structure_generation_is_deterministic() -> None:
    task = SymbolicBondOrderCountTask()
    params = {"target_bond_order": "triple", "scene_variant": "notebook_problem"}
    first = task.generate(311, params=params, max_attempts=3)
    second = task.generate(311, params=params, max_attempts=3)
    assert first.prompt == second.prompt
    assert first.answer_gt.to_dict() == second.answer_gt.to_dict()
    assert first.annotation_gt.to_dict() == second.annotation_gt.to_dict()
    assert first.trace_payload["execution_trace"] == second.trace_payload["execution_trace"]
    assert _png_bytes(first) == _png_bytes(second)

    ring_params = {"target_ring_size": 5, "answer_value": 3, "scene_variant": "notebook_problem"}
    ring_first = SymbolicRingSizeCountTask().generate(812, params=ring_params, max_attempts=3)
    ring_second = SymbolicRingSizeCountTask().generate(812, params=ring_params, max_attempts=3)
    assert ring_first.prompt == ring_second.prompt
    assert ring_first.answer_gt.to_dict() == ring_second.answer_gt.to_dict()
    assert ring_first.annotation_gt.to_dict() == ring_second.annotation_gt.to_dict()
    assert ring_first.trace_payload["execution_trace"] == ring_second.trace_payload["execution_trace"]
    assert _png_bytes(ring_first) == _png_bytes(ring_second)


def test_organic_structure_prompt_bundle_supports_tasks() -> None:
    bundle = load_prompt_bundle("symbolic", "organic_structure", "symbolic_organic_structure_v1")
    assert "organic_structure" in bundle.scene_templates
    assert len(bundle.task_templates["organic_structure_bond_order_count"]) == REQUIRED_PROMPT_VARIANTS
    assert len(bundle.task_templates["organic_structure_ring_size_count"]) == REQUIRED_PROMPT_VARIANTS
    assert not bundle.query_templates
    assert list(bundle.required_slots_by_key["scene:organic_structure"]) == ["object_description"]
    assert list(bundle.required_slots_by_key["task:organic_structure_bond_order_count"]) == ["target_bond_order"]
    assert list(bundle.required_slots_by_key["task:organic_structure_ring_size_count"]) == ["target_ring_name"]


def test_organic_structure_scaffolds_respect_v1_constraints() -> None:
    for target_bond_order in TARGET_BOND_ORDERS:
        for answer_count in range(1, ORGANIC_STRUCTURE_MAX_BOND_ORDER_COUNT + 1):
            spec = build_constrained_organic_structure(
                Random(1000 + answer_count),
                target_bond_order=target_bond_order,
                answer_count=answer_count,
            )
            report = validate_organic_structure(spec)
            matching_bonds = [bond for bond in spec.bonds if bond.order == target_bond_order]
            assert len(matching_bonds) == answer_count
            assert report.max_valence <= 4
            assert report.crossing_count == 0

    for target_ring_size in TARGET_RING_SIZES:
        for answer_count in range(1, ORGANIC_STRUCTURE_MAX_RING_SIZE_COUNT + 1):
            spec = build_constrained_organic_ring_size_structure(
                Random(3000 + target_ring_size * 10 + answer_count),
                target_ring_size=target_ring_size,
                answer_count=answer_count,
            )
            report = validate_organic_structure(spec)
            matching_ring_ids = organic_ring_item_ids(spec, target_ring_size)
            assert len(matching_ring_ids) == answer_count
            assert all(len(ring) in TARGET_RING_SIZES for ring in spec.ring_atom_sets)
            assert report.max_valence <= 4
            assert report.crossing_count == 0


def test_organic_structure_default_scaffold_families_are_rich() -> None:
    double_families = {
        build_constrained_organic_structure(
            Random(4100 + seed),
            target_bond_order="double",
            answer_count=(seed % ORGANIC_STRUCTURE_MAX_BOND_ORDER_COUNT) + 1,
        ).scaffold_family
        for seed in range(80)
    }
    triple_families = {
        build_constrained_organic_structure(
            Random(4200 + seed),
            target_bond_order="triple",
            answer_count=(seed % ORGANIC_STRUCTURE_MAX_BOND_ORDER_COUNT) + 1,
        ).scaffold_family
        for seed in range(120)
    }
    ring_families = {
        build_constrained_organic_ring_size_structure(
            Random(4400 + seed),
            target_ring_size=5 if seed % 2 else 6,
            answer_count=(seed % ORGANIC_STRUCTURE_MAX_RING_SIZE_COUNT) + 1,
        ).scaffold_family
        for seed in range(80)
    }

    assert double_families == {"fused_aromatic"}
    assert triple_families == {"aryl_polyyne"}
    assert ring_families == {"rich_ring_cluster"}


def test_organic_structure_rich_scene_labels_are_trace_entities() -> None:
    for seed in range(6000, 6100):
        out = SymbolicRingSizeCountTask().generate(
            seed,
            params={"target_ring_size": 6},
            max_attempts=20,
        )
        trace = out.trace_payload["execution_trace"]
        if trace["organic_metadata"]["scaffold_family"] != "rich_ring_cluster":
            continue
        assert "text_labels" in trace
        assert len(trace["text_labels"]) >= 1
        assert any(
            entity["entity_type"] == "organic_text_label"
            for entity in out.trace_payload["scene_ir"]["entities"]
        )
        return
    raise AssertionError("expected a rich ring-cluster sample in the deterministic seed window")


def test_organic_structure_rejects_answer_count_outside_v1_support() -> None:
    with pytest.raises(ValueError):
        build_constrained_organic_structure(
            Random(41),
            target_bond_order="double",
            answer_count=0,
        )
    with pytest.raises(ValueError):
        build_constrained_organic_structure(
            Random(42),
            target_bond_order="double",
            answer_count=ORGANIC_STRUCTURE_MAX_BOND_ORDER_COUNT + 1,
        )
    with pytest.raises(ValueError):
        build_constrained_organic_ring_size_structure(
            Random(43),
            target_ring_size=6,
            answer_count=0,
        )
    with pytest.raises(ValueError):
        build_constrained_organic_ring_size_structure(
            Random(44),
            target_ring_size=6,
            answer_count=ORGANIC_STRUCTURE_MAX_RING_SIZE_COUNT + 1,
        )
