from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from PIL import Image
import pytest

from trace_tasks.core.source_layout_policy import parse_public_task_id
from trace_tasks.core.types import TypedValue
from trace_tasks.review import (
    REQUESTS_PER_TASK,
    MaterializationError,
    NonDeterministicGenerationError,
    ResourceProvenance,
    ReviewProvenance,
    RuntimeProvenance,
    SourceProvenance,
    capture_recipe,
    load_recipe,
    materialize_recipe,
    verify_materialized,
    verify_recipe,
)
from trace_tasks.review.provenance import (
    collect_review_provenance,
    runtime_provenance,
    sha256_file,
)

TASK_ID = "task_geometry__review_fixture__answer_value"
OTHER_TASK_ID = "task_icons__review_fixture__answer_value"
REAL_TASK_ID = "task_geometry__coordinate_plane__same_quadrant_point_count"
HASH = "sha256:" + ("1" * 64)


@dataclass
class FakeOutput:
    prompt: str
    answer_gt: TypedValue
    annotation_gt: TypedValue
    image: Image.Image
    trace_payload: dict
    task_versions: dict
    scene_id: str
    query_id: str
    prompt_variants: dict


def _provenance() -> ReviewProvenance:
    return ReviewProvenance(
        source=SourceProvenance(
            repository="maveryn/trace",
            revision="a" * 40,
            dirty=False,
            source_tree_hash=HASH,
            generator_tree_hash=HASH,
            constraints_hash=HASH,
        ),
        resources=ResourceProvenance(
            resource_tree_hash=HASH,
            prompt_bundle_tree_hash=HASH,
            task_catalog_hash=HASH,
        ),
        runtime=RuntimeProvenance(
            python_version="3.12.0",
            python_implementation="CPython",
            platform="Linux",
            machine="x86_64",
            dependencies={"Pillow": "12.0.0"},
        ),
    )


def _output(
    task_id: str,
    seed: int,
    params: dict,
    *,
    color: tuple[int, int, int] = (20, 40, 60),
    semantic_suffix: str = "",
) -> FakeOutput:
    parts = parse_public_task_id(task_id)
    query_id = str(params["query_id"])
    trace = {
        "scene_ir": {"seed": seed, "semantic_suffix": semantic_suffix},
        "query_spec": {"query_id": query_id, "params": dict(params)},
        "render_spec": {"width": 8, "height": 6},
        "render_map": {"fixture": [0, 0, 8, 6]},
        "execution_trace": {"answer": seed % 11},
        "witness_symbolic": {"type": "fixture"},
        "projected_annotation": {"type": "bbox", "bbox": [0, 0, 4, 4]},
    }
    return FakeOutput(
        prompt=f"What is the fixture value for {seed}?{semantic_suffix}",
        answer_gt=TypedValue(type="integer", value=seed % 11),
        annotation_gt=TypedValue(type="bbox", value=[0, 0, 4, 4]),
        image=Image.new("RGB", (8, 6), color),
        trace_payload=trace,
        task_versions={"fixture": "v1"},
        scene_id=parts.scene_id,
        query_id=query_id,
        prompt_variants={
            "answer_only": "Answer.",
            "answer_and_annotation": "Answer and annotate.",
        },
    )


def _generator(task_id: str, seed: int, params: dict, max_attempts: int) -> FakeOutput:
    assert max_attempts == 7
    return _output(task_id, seed, dict(params))


def _capture(tmp_path: Path, *, task_ids=(TASK_ID,)) -> Path:
    root = tmp_path / "recipe"
    capture_recipe(
        task_ids,
        root,
        max_attempts=7,
        query_ids_by_task={task_id: ("alpha", "beta", "gamma") for task_id in task_ids},
        generator=_generator,
        provenance=_provenance(),
    )
    return root


def test_capture_writes_versioned_domain_shards_and_exact_query_stratification(
    tmp_path: Path,
) -> None:
    recipe_root = _capture(tmp_path, task_ids=(TASK_ID, OTHER_TASK_ID))

    manifest, requests = load_recipe(recipe_root)

    assert manifest.recipe_id == "trace-review-recipe-v1"
    assert manifest.requests_per_task == REQUESTS_PER_TASK
    assert manifest.task_count == 2
    assert manifest.request_count == 2 * REQUESTS_PER_TASK
    assert [shard.domain for shard in manifest.shards] == ["geometry", "icons"]
    assert (recipe_root / "requests" / "geometry.jsonl").is_file()
    task_requests = [row for row in requests if row.task_id == TASK_ID]
    assert len(task_requests) == REQUESTS_PER_TASK
    assert [row.ordinal for row in task_requests] == list(range(REQUESTS_PER_TASK))
    assert [row.query_id for row in task_requests[:7]] == [
        "alpha",
        "beta",
        "gamma",
        "alpha",
        "beta",
        "gamma",
        "alpha",
    ]
    assert all(row.sample_cursor == row.ordinal for row in task_requests)
    assert all(row.params["_sample_cursor"] == row.ordinal for row in task_requests)
    assert all(row.params["query_id"] == row.query_id for row in task_requests)


def test_capture_rejects_noncanonical_request_count(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="exactly 25"):
        capture_recipe(
            (TASK_ID,),
            tmp_path / "recipe",
            requests_per_task=24,
            query_ids_by_task={TASK_ID: ("alpha",)},
            generator=_generator,
            provenance=_provenance(),
        )


def test_runtime_provenance_records_portable_native_rendering_versions() -> None:
    runtime = runtime_provenance()

    assert runtime.native_libraries["libcairo"]
    assert runtime.native_libraries["pillow:zlib"]
    assert runtime.native_libraries["pillow:freetype2"]
    assert runtime.native_libraries["libc"]
    assert all("/" not in value for value in runtime.native_libraries.values())


def test_provenance_freshness_ignores_review_app_and_non_generation_pins(
    tmp_path: Path,
) -> None:
    package = tmp_path / "src" / "trace_tasks"
    for relative, content in {
        "core/generator.py": "CORE = 1\n",
        "tasks/task.py": "TASK = 1\n",
        "review/models.py": "MODELS = 1\n",
        "review/provenance.py": "PROVENANCE = 1\n",
        "review/recipe.py": "RECIPE = 1\n",
        "review/export.py": "EXPORT = 1\n",
        "review/app/static/app.css": "body {}\n",
        "configs/type_registry_v0.json": "{}\n",
        "resources/prompts/fixture.json": "{}\n",
        "resources/assets/fixture.txt": "asset\n",
    }.items():
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    constraints = tmp_path / "constraints" / "release.txt"
    constraints.parent.mkdir(parents=True)
    constraints.write_text(
        "Pillow==12.2.0\nFastAPI==0.136.1\npytest==9.1.1\n",
        encoding="utf-8",
    )

    def collect():
        return collect_review_provenance(
            repo_root=tmp_path,
            task_query_ids={TASK_ID: ("single",)},
            require_clean_source=False,
        )

    original = collect()
    (package / "review/app/static/app.css").write_text(
        "body { color: red; }\n", encoding="utf-8"
    )
    app_changed = collect()
    assert app_changed.source.source_tree_hash != original.source.source_tree_hash
    assert app_changed.source.generator_tree_hash == original.source.generator_tree_hash
    assert app_changed.source.constraints_hash == original.source.constraints_hash

    (package / "review/export.py").write_text("EXPORT = 2\n", encoding="utf-8")
    export_changed = collect()
    assert (
        export_changed.source.generator_tree_hash == original.source.generator_tree_hash
    )

    (package / "configs/type_registry_v0.json").write_text(
        '{"changed": true}\n', encoding="utf-8"
    )
    config_changed = collect()
    assert (
        config_changed.source.generator_tree_hash != original.source.generator_tree_hash
    )

    constraints.write_text(
        "Pillow==12.2.0\nFastAPI==0.139.2\npytest==9.2.0\n",
        encoding="utf-8",
    )
    app_pins_changed = collect()
    assert app_pins_changed.source.constraints_hash == original.source.constraints_hash

    (package / "review/recipe.py").write_text("RECIPE = 2\n", encoding="utf-8")
    recipe_changed = collect()
    assert (
        recipe_changed.source.generator_tree_hash != original.source.generator_tree_hash
    )

    constraints.write_text(
        "Pillow==12.3.0\nFastAPI==0.139.2\npytest==9.2.0\n",
        encoding="utf-8",
    )
    generation_pin_changed = collect()
    assert (
        generation_pin_changed.source.constraints_hash
        != original.source.constraints_hash
    )


def test_capture_double_generates_and_rejects_nondeterminism(tmp_path: Path) -> None:
    calls = 0

    def unstable(
        task_id: str, seed: int, params: dict, max_attempts: int
    ) -> FakeOutput:
        nonlocal calls
        calls += 1
        color = (255, 0, 0) if calls % 2 else (0, 0, 255)
        return _output(task_id, seed, dict(params), color=color)

    with pytest.raises(NonDeterministicGenerationError, match="identical generations"):
        capture_recipe(
            (TASK_ID,),
            tmp_path / "recipe",
            max_attempts=7,
            query_ids_by_task={TASK_ID: ("alpha",)},
            generator=unstable,
            provenance=_provenance(),
        )
    assert calls == 2
    assert not (tmp_path / "recipe").exists()


def test_parallel_capture_preserves_canonical_order_and_digest(tmp_path: Path) -> None:
    sequential_root = tmp_path / "sequential"
    parallel_root = tmp_path / "parallel"
    kwargs = {
        "task_ids": (TASK_ID, OTHER_TASK_ID),
        "max_attempts": 7,
        "query_ids_by_task": {
            TASK_ID: ("alpha", "beta", "gamma"),
            OTHER_TASK_ID: ("alpha", "beta", "gamma"),
        },
        "generator": _generator,
        "provenance": _provenance(),
    }
    sequential = capture_recipe(
        recipe_root=sequential_root,
        workers=1,
        **kwargs,
    )
    parallel = capture_recipe(
        recipe_root=parallel_root,
        workers=4,
        **kwargs,
    )

    assert parallel.recipe_digest == sequential.recipe_digest
    assert (parallel_root / "manifest.json").read_bytes() == (
        sequential_root / "manifest.json"
    ).read_bytes()
    for domain in ("geometry", "icons"):
        assert (parallel_root / "requests" / f"{domain}.jsonl").read_bytes() == (
            sequential_root / "requests" / f"{domain}.jsonl"
        ).read_bytes()
    _, sequential_requests = load_recipe(sequential_root)
    _, parallel_requests = load_recipe(parallel_root)
    assert parallel_requests == sequential_requests


def test_capture_rejects_unbounded_worker_values(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="workers must be at least 1"):
        capture_recipe(
            (TASK_ID,),
            tmp_path / "recipe",
            workers=0,
            query_ids_by_task={TASK_ID: ("alpha",)},
            generator=_generator,
            provenance=_provenance(),
        )


def test_default_generator_process_capture_matches_sequential(tmp_path: Path) -> None:
    kwargs = {
        "task_ids": (REAL_TASK_ID,),
        "max_attempts": 10,
        "query_ids_by_task": {REAL_TASK_ID: ("single",)},
        "provenance": _provenance(),
    }

    sequential = capture_recipe(
        recipe_root=tmp_path / "real-sequential",
        workers=1,
        **kwargs,
    )
    parallel = capture_recipe(
        recipe_root=tmp_path / "real-parallel",
        workers=2,
        **kwargs,
    )

    assert parallel.recipe_digest == sequential.recipe_digest
    assert (
        tmp_path / "real-parallel" / "requests" / "geometry.jsonl"
    ).read_bytes() == (
        tmp_path / "real-sequential" / "requests" / "geometry.jsonl"
    ).read_bytes()


def test_load_rejects_modified_shard(tmp_path: Path) -> None:
    recipe_root = _capture(tmp_path)
    shard = recipe_root / "requests" / "geometry.jsonl"
    shard.write_bytes(shard.read_bytes() + b"\n")

    with pytest.raises(Exception, match="shard digest mismatch"):
        load_recipe(recipe_root)


def test_materialize_is_atomic_resumable_and_refuses_mismatches(tmp_path: Path) -> None:
    recipe_root = _capture(tmp_path)
    output_root = tmp_path / "review"
    calls = 0

    def counting(
        task_id: str, seed: int, params: dict, max_attempts: int
    ) -> FakeOutput:
        nonlocal calls
        calls += 1
        return _generator(task_id, seed, params, max_attempts)

    report = materialize_recipe(recipe_root, output_root, generator=counting)
    assert report.tasks_materialized == (TASK_ID,)
    assert report.tasks_resumed == ()
    assert report.selected_requests == REQUESTS_PER_TASK
    assert calls == REQUESTS_PER_TASK
    task_root = output_root / "geometry" / "review_fixture" / TASK_ID
    assert (task_root / "manifest.json").is_file()
    assert len(list((task_root / "data").rglob("*.json"))) == REQUESTS_PER_TASK
    assert len(list((task_root / "images").rglob("*.png"))) == REQUESTS_PER_TASK
    assert not (output_root / ".staging").exists()
    first_payload = json.loads(
        sorted((task_root / "data").rglob("*.json"))[0].read_text(encoding="utf-8")
    )
    assert first_payload["reward_contract"]["answer"] == {
        "id": "answer_exact_match_v0",
        "type": "integer",
    }
    assert first_payload["reward_contract"]["annotation"] == {
        "id": "bbox_soft_iou_v0",
        "type": "bbox",
    }
    expected_taxonomy = {
        "domain": "geometry",
        "scene_id": "review_fixture",
        "task_id": TASK_ID,
        "query_id": first_payload["query_id"],
    }
    assert first_payload["taxonomy"] == expected_taxonomy
    assert first_payload["trace_payload"]["taxonomy"]["public"] == expected_taxonomy
    task_manifest = json.loads(
        (task_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert task_manifest["source_provenance"]["revision"] == "a" * 40
    assert task_manifest["recipe_provenance"]["producer_revision"] == "a" * 40
    assert first_payload["recipe_provenance"] == task_manifest["recipe_provenance"]
    assert set(first_payload["recipe_provenance"]) >= {
        "source_tree_hash",
        "generator_tree_hash",
        "constraints_hash",
        "resource_tree_hash",
        "prompt_bundle_tree_hash",
        "task_catalog_hash",
    }

    calls = 0
    resumed = materialize_recipe(recipe_root, output_root, generator=counting)
    assert resumed.tasks_resumed == (TASK_ID,)
    assert resumed.tasks_materialized == ()
    assert calls == 0
    assert verify_materialized(recipe_root, output_root).ok

    data_path = sorted((task_root / "data").rglob("*.json"))[0]
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    payload["prompt"] = "changed"
    data_path.write_text(json.dumps(payload), encoding="utf-8")
    task_manifest_path = task_root / "manifest.json"
    task_manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
    data_relative = data_path.relative_to(output_root).as_posix()
    matching_entry = next(
        entry
        for entry in task_manifest["entries"]
        if entry["data_path"] == data_relative
    )
    # Simulate a coordinated tamper that updates the mutable task-level file
    # hash. The semantic duplicate binding must still detect it.
    matching_entry["data_sha256"] = sha256_file(data_path)
    task_manifest_path.write_text(json.dumps(task_manifest), encoding="utf-8")
    tampered = verify_materialized(recipe_root, output_root)
    assert not tampered.ok
    assert "review_field_mismatch" in {issue.code for issue in tampered.errors}
    with pytest.raises(MaterializationError, match="refusing to overwrite"):
        materialize_recipe(recipe_root, output_root, generator=counting)


def test_rendering_drift_warns_by_default_and_fails_in_strict_mode(
    tmp_path: Path,
) -> None:
    recipe_root = _capture(tmp_path)

    def drifted(task_id: str, seed: int, params: dict, max_attempts: int) -> FakeOutput:
        return _output(task_id, seed, dict(params), color=(90, 100, 110))

    report = verify_recipe(recipe_root, generator=drifted)
    assert report.ok
    assert len(report.warnings) == REQUESTS_PER_TASK * 2
    assert {issue.code for issue in report.warnings} == {
        "raw_pixel_hash_mismatch",
        "png_hash_mismatch",
    }

    strict = verify_recipe(recipe_root, generator=drifted, strict_rendering=True)
    assert not strict.ok
    assert len(strict.errors) == REQUESTS_PER_TASK * 2

    materialized = materialize_recipe(
        recipe_root,
        tmp_path / "drifted-review",
        generator=drifted,
    )
    assert len(materialized.warnings) == REQUESTS_PER_TASK * 2


def test_semantic_drift_is_always_a_hard_failure(tmp_path: Path) -> None:
    recipe_root = _capture(tmp_path)

    def changed(task_id: str, seed: int, params: dict, max_attempts: int) -> FakeOutput:
        return _output(task_id, seed, dict(params), semantic_suffix=" changed")

    report = verify_recipe(recipe_root, generator=changed)
    assert not report.ok
    assert {issue.code for issue in report.errors} == {"semantic_hash_mismatch"}
    with pytest.raises(MaterializationError, match="semantic hash mismatch"):
        materialize_recipe(recipe_root, tmp_path / "review", generator=changed)


def test_materialize_rejects_query_filters_to_preserve_atomic_tasks(
    tmp_path: Path,
) -> None:
    recipe_root = _capture(tmp_path)
    with pytest.raises(MaterializationError, match="query filtering is not supported"):
        materialize_recipe(
            recipe_root,
            tmp_path / "review",
            query_ids=("beta",),
            generator=_generator,
        )
    assert not (tmp_path / "review").exists()


def test_verify_reports_empty_filter_selection_as_error(tmp_path: Path) -> None:
    recipe_root = _capture(tmp_path)

    replay = verify_recipe(
        recipe_root,
        task_ids=("task_geometry__missing_scene__missing_value",),
        generator=_generator,
    )
    materialized = verify_materialized(
        recipe_root,
        tmp_path / "missing-review",
        query_ids=("not-a-recipe-query",),
    )

    assert not replay.ok and replay.checked_requests == 0
    assert not materialized.ok and materialized.checked_requests == 0
    assert {issue.code for issue in replay.errors} == {"empty_selection"}
    assert {issue.code for issue in materialized.errors} == {"empty_selection"}


def test_materialize_accepts_qualified_and_bare_scene_filters(tmp_path: Path) -> None:
    first_recipe = _capture(tmp_path / "qualified")
    qualified = materialize_recipe(
        first_recipe,
        tmp_path / "qualified-review",
        scene_ids=("geometry/review_fixture",),
        generator=_generator,
    )
    assert qualified.selected_requests == REQUESTS_PER_TASK

    second_recipe = _capture(tmp_path / "bare")
    bare = materialize_recipe(
        second_recipe,
        tmp_path / "bare-review",
        scene_ids=("review_fixture",),
        generator=_generator,
    )
    assert bare.selected_requests == REQUESTS_PER_TASK
