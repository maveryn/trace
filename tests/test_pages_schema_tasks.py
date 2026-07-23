"""Behavior tests for pages database-schema diagram tasks."""

from __future__ import annotations

from collections import Counter
from collections import deque

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.seed import hash64
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks.pages.schema.field_role_count import (
    TASK_ID as FIELD_ROLE_COUNT_TASK_ID,
    PagesSchemaFieldRoleCountTask,
)
from trace_tasks.tasks.pages.schema.join_path_length_value import (
    TASK_ID as JOIN_PATH_LENGTH_TASK_ID,
    PagesSchemaJoinPathLengthValueTask,
)
from trace_tasks.tasks.pages.schema.relationship_cardinality_label import (
    TASK_ID as RELATIONSHIP_CARDINALITY_TASK_ID,
    PagesSchemaRelationshipCardinalityLabelTask,
)
from trace_tasks.tasks.pages.schema.relationship_count import (
    TASK_ID as RELATIONSHIP_COUNT_TASK_ID,
    PagesSchemaRelationshipCountTask,
)
from trace_tasks.tasks.pages.schema.relationship_endpoint_label import (
    TASK_ID as RELATIONSHIP_ENDPOINT_TASK_ID,
    PagesSchemaRelationshipEndpointLabelTask,
)


def _assert_bboxes_inside_image(out) -> None:
    width, height = out.image.size
    annotation_value = out.annotation_gt.value
    if out.annotation_gt.type == "bbox":
        bboxes = [annotation_value]
    elif isinstance(annotation_value, dict):
        bboxes = annotation_value.values()
    else:
        bboxes = annotation_value
    for bbox in bboxes:
        x0, y0, x1, y1 = [float(value) for value in bbox]
        assert 0.0 <= x0 <= x1 <= float(width)
        assert 0.0 <= y0 <= y1 <= float(height)


def _assert_point_pairs_inside_image(out) -> None:
    width, height = out.image.size
    for point_pair in out.annotation_gt.value:
        assert len(point_pair) == 2
        for point in point_pair:
            x, y = [float(value) for value in point]
            assert 0.0 <= x <= float(width)
            assert 0.0 <= y <= float(height)


def test_pages_schema_tasks_are_registered_in_public_taxonomy() -> None:
    for task_id in [
        FIELD_ROLE_COUNT_TASK_ID,
        JOIN_PATH_LENGTH_TASK_ID,
        RELATIONSHIP_COUNT_TASK_ID,
        RELATIONSHIP_ENDPOINT_TASK_ID,
        RELATIONSHIP_CARDINALITY_TASK_ID,
    ]:
        taxonomy = resolve_task_taxonomy(task_id)
        assert taxonomy.domain == "pages"
        assert taxonomy.scene_id == "schema"
        assert taxonomy.source_scene_id == ""


def _shortest_path_count(relationships, source_id: str, target_id: str) -> tuple[int, int]:
    adjacency: dict[str, list[tuple[str, str]]] = {}
    for relationship in relationships:
        source = str(relationship["source_table_id"])
        target = str(relationship["target_table_id"])
        rid = str(relationship["relationship_id"])
        adjacency.setdefault(source, []).append((target, rid))
        adjacency.setdefault(target, []).append((source, rid))
    queue = deque([(str(source_id), tuple(), frozenset({str(source_id)}))])
    best_length: int | None = None
    path_count = 0
    while queue:
        table_id, edges, seen = queue.popleft()
        if best_length is not None and len(edges) >= best_length:
            continue
        for next_table, relationship_id in adjacency.get(str(table_id), []):
            if next_table in seen:
                continue
            next_edges = tuple([*edges, str(relationship_id)])
            if str(next_table) == str(target_id):
                if best_length is None or len(next_edges) < best_length:
                    best_length = len(next_edges)
                    path_count = 1
                elif len(next_edges) == best_length:
                    path_count += 1
            else:
                queue.append((str(next_table), next_edges, frozenset({*seen, str(next_table)})))
    if best_length is None:
        raise AssertionError("no path found between selected schema tables")
    return int(best_length), int(path_count)


def test_pages_schema_field_role_count_contract() -> None:
    task = PagesSchemaFieldRoleCountTask()
    for query_id in ("all_field_count", "attribute_field_count"):
        out = task.generate(94200, params={"query_id": query_id, "layout_variant": "grid"}, max_attempts=10)
        trace = out.trace_payload
        query = trace["execution_trace"]["query"]
        expected = [
            trace["render_map"]["field_bboxes_px"][str(field_id)]
            for field_id in query["annotation_field_ids"]
        ]

        assert out.scene_id == "schema"
        assert out.query_id == query_id
        assert out.answer_gt.type == "integer"
        assert out.annotation_gt.type == "bbox_set"
        assert int(out.answer_gt.value) == int(query["answer"])
        assert len(out.annotation_gt.value) == int(out.answer_gt.value)
        assert out.annotation_gt.value == expected
        assert sorted(out.prompt_variants) == ["answer_and_annotation", "answer_only"]
        _assert_bboxes_inside_image(out)


def test_pages_schema_relationship_count_contract() -> None:
    task = PagesSchemaRelationshipCountTask()
    for query_id in (SINGLE_QUERY_ID,):
        out = task.generate(94240, params={"query_id": query_id, "layout_variant": "radial"}, max_attempts=10)
        trace = out.trace_payload
        query = trace["execution_trace"]["query"]
        expected = [
            trace["render_map"]["relationship_point_pairs_px"][str(relationship_id)]
            for relationship_id in query["annotation_relationship_ids"]
        ]

        assert out.scene_id == "schema"
        assert out.query_id == query_id
        assert out.answer_gt.type == "integer"
        assert out.annotation_gt.type == "segment_set"
        assert int(out.answer_gt.value) == int(query["answer"])
        assert len(out.annotation_gt.value) == int(out.answer_gt.value)
        assert out.annotation_gt.value == expected
        _assert_point_pairs_inside_image(out)


def test_pages_schema_join_path_length_contract() -> None:
    task = PagesSchemaJoinPathLengthValueTask()
    for expected_length in (1, 2, 3, 4):
        out = task.generate(
            94300 + expected_length,
            params={"query_id": SINGLE_QUERY_ID, "join_path_length": expected_length, "layout_variant": "grid"},
            max_attempts=10,
        )
        trace = out.trace_payload
        query = trace["execution_trace"]["query"]
        relationships = trace["execution_trace"]["relationships"]
        path_relationship_ids = [str(value) for value in query["path_relationship_ids"]]
        expected = [
            trace["render_map"]["relationship_point_pairs_px"][relationship_id]
            for relationship_id in path_relationship_ids
        ]
        shortest_length, shortest_count = _shortest_path_count(
            relationships,
            str(query["source_table_id"]),
            str(query["target_table_id"]),
        )

        assert out.scene_id == "schema"
        assert out.query_id == SINGLE_QUERY_ID
        assert trace["execution_trace"]["source_query_id"] == "join_path_length_value"
        assert out.answer_gt.type == "integer"
        assert out.annotation_gt.type == "segment_set"
        assert int(out.answer_gt.value) == expected_length
        assert int(out.answer_gt.value) == int(query["answer"])
        assert int(query["path_length"]) == expected_length
        assert shortest_length == expected_length
        assert shortest_count == 1
        assert len(path_relationship_ids) == expected_length
        assert out.annotation_gt.value == expected
        assert trace["projected_annotation"]["type"] == "segment_set"
        assert trace["projected_annotation"]["segment_set"] == expected
        assert trace["execution_trace"]["supporting_segment_ids"] == [
            f"relationship_segment:{relationship_id}"
            for relationship_id in path_relationship_ids
        ]
        assert trace["query_spec"]["params"]["context_bbox_ids"] == {
            "source_table": f"table:{query['source_table_id']}",
            "target_table": f"table:{query['target_table_id']}",
        }
        _assert_point_pairs_inside_image(out)


def test_pages_schema_relationship_endpoint_label_contract() -> None:
    task = PagesSchemaRelationshipEndpointLabelTask()
    out = task.generate(
        94260,
        params={"query_id": SINGLE_QUERY_ID, "layout_variant": "grid"},
        max_attempts=10,
    )
    trace = out.trace_payload
    query = trace["execution_trace"]["query"]
    relationships = trace["execution_trace"]["relationships"]
    matches = [
        relationship
        for relationship in relationships
        if str(relationship["source_label"]) == str(query["source_table_label"])
        and str(relationship["label"]) == str(query["relationship_label"])
    ]
    expected = trace["render_map"]["table_bboxes_px"][str(query["target_table_id"])]

    assert out.scene_id == "schema"
    assert out.query_id == SINGLE_QUERY_ID
    assert trace["execution_trace"]["source_query_id"] == "target_table_for_relationship_label"
    assert out.answer_gt.type == "string"
    assert out.annotation_gt.type == "bbox"
    assert str(out.answer_gt.value) == str(query["target_table_label"])
    assert str(out.answer_gt.value) == str(query["answer"])
    assert len(matches) == 1
    assert str(matches[0]["target_label"]) == str(out.answer_gt.value)
    assert out.annotation_gt.value == expected
    assert trace["projected_annotation"]["type"] == "bbox"
    assert trace["projected_annotation"]["bbox"] == expected
    assert trace["witness_symbolic"]["type"] == "bbox_id"
    assert trace["execution_trace"]["supporting_bbox_ids"] == [f"table:{query['target_table_id']}"]
    assert trace["query_spec"]["params"]["context_bbox_ids"] == {
        "source_table": f"table:{query['source_table_id']}",
        "relationship_label": f"relationship_label:{query['relationship_id']}",
    }
    _assert_bboxes_inside_image(out)


def test_pages_schema_relationship_cardinality_label_contract() -> None:
    task = PagesSchemaRelationshipCardinalityLabelTask()
    out = task.generate(
        94270,
        params={"query_id": SINGLE_QUERY_ID, "layout_variant": "grid"},
        max_attempts=10,
    )
    trace = out.trace_payload
    query = trace["execution_trace"]["query"]
    relationships = trace["execution_trace"]["relationships"]
    relationship_id = str(query["relationship_id"])
    matches = [
        relationship
        for relationship in relationships
        if str(relationship["relationship_id"]) == relationship_id
    ]
    expected = {
        "source_cardinality_marker": trace["render_map"]["cardinality_marker_bboxes_px"][f"{relationship_id}:source"],
        "target_cardinality_marker": trace["render_map"]["cardinality_marker_bboxes_px"][f"{relationship_id}:target"],
    }

    assert out.scene_id == "schema"
    assert out.query_id == SINGLE_QUERY_ID
    assert trace["execution_trace"]["source_query_id"] == "relationship_cardinality_between_tables"
    assert out.answer_gt.type == "string"
    assert out.annotation_gt.type == "bbox_map"
    assert str(out.answer_gt.value) == str(query["cardinality_kind"])
    assert str(out.answer_gt.value) == str(query["answer"])
    assert str(out.answer_gt.value) in {"one_to_one", "one_to_many", "optional_many", "many_to_many"}
    assert trace["query_spec"]["params"]["answer_support"] == [
        "one_to_many",
        "optional_many",
        "one_to_one",
        "many_to_many",
    ]
    assert len(matches) == 1
    assert str(matches[0]["cardinality_kind"]) == str(out.answer_gt.value)
    assert str(matches[0]["source_marker"]) == str(query["source_cardinality_marker"])
    assert str(matches[0]["target_marker"]) == str(query["target_cardinality_marker"])
    assert list(out.annotation_gt.value.keys()) == [
        "source_cardinality_marker",
        "target_cardinality_marker",
    ]
    assert out.annotation_gt.value == expected
    assert trace["projected_annotation"]["type"] == "bbox_map"
    assert trace["projected_annotation"]["bbox_map"] == expected
    assert trace["witness_symbolic"]["type"] == "keyed_bbox_id_map"
    assert trace["execution_trace"]["supporting_bbox_ids"] == [
        f"cardinality_marker:{relationship_id}:source",
        f"cardinality_marker:{relationship_id}:target",
    ]
    assert trace["query_spec"]["params"]["context_bbox_ids"] == {
        "source_table": f"table:{query['source_table_id']}",
        "target_table": f"table:{query['target_table_id']}",
    }
    _assert_bboxes_inside_image(out)


def test_pages_schema_generation_is_deterministic() -> None:
    task = PagesSchemaRelationshipEndpointLabelTask()
    params = {
        "query_id": SINGLE_QUERY_ID,
        "layout_variant": "grid",
        "style_variant": "green_erd",
    }
    out_a = task.generate(94280, params=params, max_attempts=10)
    out_b = task.generate(94280, params=params, max_attempts=10)

    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_pages_schema_relationship_cardinality_generation_is_deterministic() -> None:
    task = PagesSchemaRelationshipCardinalityLabelTask()
    params = {
        "query_id": SINGLE_QUERY_ID,
        "layout_variant": "radial",
        "style_variant": "amber_blueprint",
    }
    out_a = task.generate(94290, params=params, max_attempts=10)
    out_b = task.generate(94290, params=params, max_attempts=10)

    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_pages_schema_sampling_covers_visual_and_text_axes() -> None:
    task = PagesSchemaRelationshipEndpointLabelTask()
    layouts: Counter[str] = Counter()
    styles: Counter[str] = Counter()
    contexts: Counter[str] = Counter()
    queries: Counter[str] = Counter()

    for index in range(24):
        out = task.generate(hash64(94320, "schema_axes", index), params={}, max_attempts=10)
        execution = out.trace_payload["execution_trace"]
        layouts[str(execution["layout_variant"])] += 1
        styles[str(execution["style_variant"])] += 1
        contexts[str(execution["context_id"])] += 1
        queries[str(execution["query_id"])] += 1

    assert set(layouts) == {"grid", "layered", "radial"}
    assert set(styles) == {"green_erd", "violet_cards", "monochrome_sql", "amber_blueprint"}
    assert len(contexts) >= 5
    assert set(queries) == {SINGLE_QUERY_ID}
