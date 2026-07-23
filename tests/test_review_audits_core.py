from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from trace_tasks.core.types import TypedValue
from trace_tasks.review import (
    ResourceProvenance,
    ReviewProvenance,
    RuntimeProvenance,
    SourceProvenance,
    audit_generated_output,
    audit_materialized,
    audit_recipe,
    capture_recipe,
    materialize_recipe,
)

TASK_ID = "task_geometry__audit_fixture__answer_value"
OTHER_TASK_ID = "task_icons__audit_fixture__answer_value"
HASH = "sha256:" + ("2" * 64)


class Output:
    def __init__(self, seed: int, params: dict) -> None:
        query_id = str(params["query_id"])
        self.prompt = "Read the fixture and answer."
        self.prompt_variants = {"answer_only": "Answer."}
        self.answer_gt = TypedValue(type="integer", value=seed % 5)
        self.annotation_gt = TypedValue(type="bbox", value=[0, 0, 3, 3])
        self.image = Image.new("RGB", (4, 4), (10, 20, 30))
        self.scene_id = "audit_fixture"
        self.query_id = query_id
        self.task_versions = {"fixture": "v1"}
        self.trace_payload = {
            "scene_ir": {"seed": seed},
            "query_spec": {"query_id": query_id, "params": dict(params)},
            "render_spec": {},
            "render_map": {},
            "execution_trace": {"answer": seed % 5},
            "witness_symbolic": {},
            "projected_annotation": {"type": "bbox", "bbox": [0, 0, 3, 3]},
        }


def _generator(task_id: str, seed: int, params: dict, max_attempts: int) -> Output:
    assert task_id in {TASK_ID, OTHER_TASK_ID}
    return Output(seed, params)


def _provenance(*, dirty: bool = False) -> ReviewProvenance:
    return ReviewProvenance(
        source=SourceProvenance(
            repository="maveryn/trace",
            revision="b" * 40,
            dirty=dirty,
            source_tree_hash=HASH,
            generator_tree_hash=HASH,
            constraints_hash=HASH,
        ),
        resources=ResourceProvenance(HASH, HASH, HASH),
        runtime=RuntimeProvenance(
            "3.12.0", "CPython", "Linux", "x86_64", {"Pillow": "12.0.0"}
        ),
    )


def test_recipe_and_materialized_audits_pass_for_canonical_artifacts(
    tmp_path: Path,
) -> None:
    recipe_root = tmp_path / "recipe"
    output_root = tmp_path / "review"
    capture_recipe(
        (TASK_ID,),
        recipe_root,
        max_attempts=5,
        query_ids_by_task={TASK_ID: ("first", "second")},
        generator=_generator,
        provenance=_provenance(),
    )
    materialize_recipe(recipe_root, output_root, generator=_generator)

    recipe_report = audit_recipe(recipe_root)
    artifact_report = audit_materialized(recipe_root, output_root)

    assert recipe_report.ok, recipe_report.to_dict()
    assert recipe_report.checked == 25
    assert artifact_report.ok, artifact_report.to_dict()
    assert artifact_report.checked == 25


def test_recipe_audit_rejects_dirty_source_provenance(tmp_path: Path) -> None:
    recipe_root = tmp_path / "recipe"
    capture_recipe(
        (TASK_ID,),
        recipe_root,
        max_attempts=5,
        query_ids_by_task={TASK_ID: ("only",)},
        generator=_generator,
        provenance=_provenance(dirty=True),
    )

    report = audit_recipe(recipe_root)

    assert not report.ok
    assert {issue.code for issue in report.errors} == {"dirty_source_capture"}


def test_generated_output_audit_reports_portable_contract_failures() -> None:
    output = Output(3, {"query_id": "only", "_sample_cursor": 0})
    assert audit_generated_output(output, task_id=TASK_ID).ok

    output.prompt = ""
    output.answer_gt = TypedValue(type="", value=None)
    output.trace_payload.pop("execution_trace")
    output.image = None

    report = audit_generated_output(output, task_id=TASK_ID)

    assert not report.ok
    assert {
        "empty_prompt",
        "invalid_answer_gt_type",
        "missing_answer_gt_value",
        "missing_trace_fields",
        "invalid_image",
    }.issubset({issue.code for issue in report.errors})


def test_materialized_audit_applies_all_taxonomy_filters(tmp_path: Path) -> None:
    recipe_root = tmp_path / "recipe"
    output_root = tmp_path / "review"
    capture_recipe(
        (TASK_ID, OTHER_TASK_ID),
        recipe_root,
        max_attempts=5,
        query_ids_by_task={
            TASK_ID: ("first", "second"),
            OTHER_TASK_ID: ("first", "second"),
        },
        generator=_generator,
        provenance=_provenance(),
    )
    materialize_recipe(recipe_root, output_root, generator=_generator)
    other_data = sorted(
        (output_root / "icons" / "audit_fixture" / OTHER_TASK_ID / "data").rglob(
            "*.json"
        )
    )[0]
    tampered = json.loads(other_data.read_text(encoding="utf-8"))
    tampered["prompt"] = "tampered unrelated task"
    other_data.write_text(json.dumps(tampered), encoding="utf-8")

    assert audit_materialized(recipe_root, output_root, task_ids=(TASK_ID,)).ok
    assert audit_materialized(recipe_root, output_root, domains=("geometry",)).ok
    assert audit_materialized(
        recipe_root,
        output_root,
        scene_ids=("geometry/audit_fixture",),
    ).ok
    query_report = audit_materialized(
        recipe_root,
        output_root,
        task_ids=(TASK_ID,),
        query_ids=("first",),
    )
    assert query_report.ok
    assert query_report.checked == 13
    assert not audit_materialized(
        recipe_root, output_root, task_ids=(OTHER_TASK_ID,)
    ).ok
