"""Regression tests for graph-domain phylogeny tree tasks."""

from __future__ import annotations

from io import BytesIO

import trace_tasks.tasks  # noqa: F401
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks.graph.phylogeny_tree.clade_leaf_count import TASK_ID as CLADE_COUNT_TASK_ID
from trace_tasks.tasks.graph.phylogeny_tree.clade_leaf_count import _prepare_clade_case
from trace_tasks.tasks.graph.phylogeny_tree.mrca_clade_membership_count import MRCA_TASK_ID
from trace_tasks.tasks.graph.phylogeny_tree.mrca_clade_membership_count import _prepare_mrca_case
from trace_tasks.tasks.graph.phylogeny_tree.sister_leaf_label import SISTER_TASK_ID
from trace_tasks.tasks.graph.phylogeny_tree.topology_outlier_label import TOPOLOGY_TASK_ID
from trace_tasks.tasks.graph.phylogeny_tree.shared.sampling import sample_topology_outlier_options
from trace_tasks.tasks.registry import TASK_REGISTRY, create_task


PHYLOGENY_TASK_IDS = (
    CLADE_COUNT_TASK_ID,
    SISTER_TASK_ID,
    MRCA_TASK_ID,
    TOPOLOGY_TASK_ID,
)


def _png_bytes(output) -> bytes:
    buffer = BytesIO()
    output.image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_phylogeny_tasks_are_registered_and_taxonomized() -> None:
    for task_id in PHYLOGENY_TASK_IDS:
        assert task_id in TASK_REGISTRY
        taxonomy = resolve_task_taxonomy(task_id)
        assert taxonomy.domain == "graph"
        assert taxonomy.scene_id == "phylogeny_tree"
        assert taxonomy.source_domain == "graph"
        assert taxonomy.source_scene_id == ""
        assert not hasattr(TASK_REGISTRY[task_id], "scene_id")


def test_phylogeny_clade_leaf_count_contract() -> None:
    out = create_task(CLADE_COUNT_TASK_ID).generate(3101, params={}, max_attempts=200)
    trace = out.trace_payload
    expected_labels = trace["execution_trace"]["target_leaf_labels"]
    assert out.scene_id == "phylogeny_tree"
    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "point_set"
    assert out.answer_gt.value == len(expected_labels)
    assert len(out.annotation_gt.value) == len(expected_labels)
    assert trace["projected_annotation"]["point_set"] == out.annotation_gt.value
    assert trace["scene_ir"]["relations"]["marked_clade_leaf_labels"] == expected_labels


def test_phylogeny_sister_leaf_label_contract() -> None:
    out = create_task(SISTER_TASK_ID).generate(4102, params={}, max_attempts=200)
    trace = out.trace_payload
    assert out.query_id == "single"
    assert out.answer_gt.type == "string"
    assert out.annotation_gt.type == "point"
    assert out.answer_gt.value == trace["execution_trace"]["sister_leaf_label"]
    assert trace["projected_annotation"]["point"] == out.annotation_gt.value
    assert trace["projected_annotation"]["point_map"]["sister_leaf"] == out.annotation_gt.value


def test_phylogeny_mrca_leaf_count_contract() -> None:
    out = create_task(MRCA_TASK_ID).generate(5103, params={}, max_attempts=240)
    trace = out.trace_payload
    expected_labels = trace["execution_trace"]["mrca_descendant_leaf_labels"]
    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "point_set"
    assert out.answer_gt.value == len(expected_labels)
    assert len(out.annotation_gt.value) == len(expected_labels)
    assert trace["projected_annotation"]["point_set"] == out.annotation_gt.value
    assert trace["scene_ir"]["relations"]["mrca_descendant_leaf_labels"] == expected_labels


def test_phylogeny_mrca_case_does_not_request_visual_clade_marker() -> None:
    mrca_case = _prepare_mrca_case(5104, {}, 240)
    assert mrca_case.marked_node_id is None
    assert str(mrca_case.semantic_payload["mrca_node_id"])

    clade_case = _prepare_clade_case(3102, {}, 200)
    assert clade_case.marked_node_id == str(clade_case.semantic_payload["target_node_id"])


def test_phylogeny_leaf_spacing_has_terminal_line_headroom() -> None:
    out = create_task(MRCA_TASK_ID).generate(5002, params={}, max_attempts=240)
    canvas_width, canvas_height = out.trace_payload["render_spec"]["canvas_size"]
    leaf_ys = sorted(
        float(entity["center_px"][1])
        for entity in out.trace_payload["scene_ir"]["entities"]
        if entity.get("entity_kind") == "phylogeny_leaf"
    )
    assert canvas_width == 920
    assert canvas_height >= 800
    assert out.trace_payload["query_spec"]["params"]["leaf_count"] == 12
    assert len(leaf_ys) == 12
    assert min(right - left for left, right in zip(leaf_ys, leaf_ys[1:])) >= 42.0


def test_phylogeny_rectangular_edges_do_not_route_vertically_on_leaf_rail() -> None:
    out = create_task(MRCA_TASK_ID).generate(
        667839716113664,
        params={
            "scene_variant": "rectangular_cladogram",
            "target_mrca_leaf_count": 4,
            "node_color_name": "maroon",
        },
        max_attempts=240,
    )
    entities = out.trace_payload["scene_ir"]["entities"]
    leaf_node_ids = {
        str(entity["node_id"])
        for entity in entities
        if entity.get("entity_kind") == "phylogeny_leaf"
    }
    leaf_edge_paths = [
        entity["path_px"]
        for entity in entities
        if entity.get("entity_kind") == "phylogeny_branch"
        and str(entity.get("child_id")) in leaf_node_ids
    ]
    assert out.trace_payload["query_spec"]["params"]["scene_variant"] == "rectangular_cladogram"
    assert leaf_edge_paths
    for path in leaf_edge_paths:
        assert len(path) == 3
        assert path[0][0] == path[1][0]
        assert path[1][1] == path[2][1]
        assert path[1][0] != path[2][0]


def test_phylogeny_topology_outlier_contract() -> None:
    out = create_task(TOPOLOGY_TASK_ID).generate(6104, params={}, max_attempts=240)
    trace = out.trace_payload
    assert out.query_id == "single"
    assert out.answer_gt.type == "string"
    assert out.annotation_gt.type == "bbox"
    assert len(out.annotation_gt.value) == 4
    assert out.answer_gt.value in {"A", "B", "C", "D"}
    option_records = trace["execution_trace"]["option_records"]
    assert len(option_records) == 4
    assert [record["option_label"] for record in option_records] == list("ABCD")
    assert [record["role"] for record in option_records].count("outlier") == 1
    selected = [record for record in option_records if record["option_label"] == out.answer_gt.value][0]
    assert selected["role"] == "outlier"
    assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value


def test_phylogeny_generation_is_deterministic_for_same_seed() -> None:
    task = create_task(CLADE_COUNT_TASK_ID)
    out_a = task.generate(7105, params={}, max_attempts=200)
    out_b = task.generate(7105, params={}, max_attempts=200)
    assert out_a.answer_gt == out_b.answer_gt
    assert out_a.annotation_gt == out_b.annotation_gt
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert _png_bytes(out_a) == _png_bytes(out_b)


def test_phylogeny_topology_option_signatures_have_one_outlier() -> None:
    dataset = sample_topology_outlier_options(
        8106,
        leaf_count_min=6,
        leaf_count_max=8,
        option_count=4,
        max_attempts=200,
    )
    base_signature = tuple(dataset["base_sample"].canonical_signature)
    outlier_signature = tuple(dataset["outlier_sample"].canonical_signature)
    assert base_signature != outlier_signature
    roles = {str(spec["option_label"]): str(spec["role"]) for spec in dataset["option_specs"]}
    assert set(roles) == set("ABCD")
    assert list(roles.values()).count("outlier") == 1
    for spec in dataset["option_specs"]:
        signature = tuple(spec["canonical_signature"])
        if str(spec["role"]) == "outlier":
            assert signature == outlier_signature
        else:
            assert signature == base_signature
