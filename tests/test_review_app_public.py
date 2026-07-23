from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from trace_tasks.core.canonical import canonical_json_bytes
from trace_tasks.core.taxonomy import TASK_TAXONOMY
from trace_tasks.core.types import TypedValue
from trace_tasks.review import (
    ResourceProvenance,
    ReviewProvenance,
    RuntimeProvenance,
    SourceProvenance,
    capture_recipe,
    materialize_recipe,
)
from trace_tasks.review.app import create_review_app, validate_review_bind
from trace_tasks.review.app.assets import ASSET_PREVIEW_SCHEMA, build_asset_index
from trace_tasks.review.app.index import build_review_index
from trace_tasks.review.app.store import AUDIT_FIELDS, ReviewStore
from trace_tasks.review.models import (
    ARTIFACT_SCHEMA_VERSION,
    MATERIALIZATION_SCHEMA_VERSION,
    RECIPE_ID,
    REQUESTS_PER_TASK,
)
from trace_tasks.review.provenance import sha256_bytes, sha256_file
from trace_tasks.review.recipe import _raw_pixel_hash

_DIGEST_A = f"sha256:{'a' * 64}"
_DIGEST_B = f"sha256:{'b' * 64}"


@dataclass
class _IntegrationOutput:
    prompt: str
    answer_gt: TypedValue
    annotation_gt: TypedValue
    image: Image.Image
    trace_payload: dict[str, object]
    task_versions: dict[str, str]
    scene_id: str
    query_id: str
    prompt_variants: dict[str, str]


def _source_provenance() -> dict[str, object]:
    return {
        "repository": "https://github.com/maveryn/trace",
        "revision": "0123456789abcdef",
        "dirty": False,
        "source_tree_hash": _DIGEST_A,
        "generator_tree_hash": _DIGEST_B,
        "constraints_hash": _DIGEST_A,
    }


def _recipe_provenance() -> dict[str, object]:
    source = _source_provenance()
    return {
        "producer_repository": source["repository"],
        "producer_revision": source["revision"],
        "source_dirty": source["dirty"],
        "source_tree_hash": source["source_tree_hash"],
        "generator_tree_hash": source["generator_tree_hash"],
        "constraints_hash": source["constraints_hash"],
        "resource_tree_hash": _DIGEST_B,
        "prompt_bundle_tree_hash": _DIGEST_A,
        "task_catalog_hash": _DIGEST_B,
    }


def _integration_provenance() -> ReviewProvenance:
    return ReviewProvenance(
        source=SourceProvenance(**_source_provenance()),
        resources=ResourceProvenance(
            resource_tree_hash=_DIGEST_A,
            prompt_bundle_tree_hash=_DIGEST_B,
            task_catalog_hash=_DIGEST_A,
        ),
        runtime=RuntimeProvenance(
            python_version="3.12.0",
            python_implementation="CPython",
            platform="Linux",
            machine="x86_64",
            dependencies={"Pillow": "test"},
        ),
    )


def _materialize_one(
    root: Path,
    *,
    recipe_digest: str = _DIGEST_A,
    prompt: str = "Count the marked items.",
    task_offset: int = 0,
) -> tuple[str, str, str]:
    task_id, taxonomy = sorted(TASK_TAXONOMY.items())[task_offset]
    domain, scene_id = taxonomy.domain, taxonomy.scene_id
    task_root = root / domain / scene_id / task_id
    image = Image.new("RGB", (64, 48), (240, 240, 240))
    raw_hash = _raw_pixel_hash(image)
    entries: list[dict[str, object]] = []
    for ordinal in range(REQUESTS_PER_TASK):
        filename = f"{ordinal:04d}"
        data_path = task_root / "data" / "default" / f"{filename}.json"
        image_path = task_root / "images" / "default" / f"{filename}.png"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path, format="PNG", optimize=False, compress_level=9)
        semantic = {
            "scene_id": scene_id,
            "query_id": "default",
            "taxonomy": {
                "domain": domain,
                "scene_id": scene_id,
                "task_id": task_id,
                "query_id": "default",
            },
            "prompt": f"{prompt} Review sample {ordinal}.",
            "prompt_variants": {
                "answer_only": 'Count the marked items. Return {"answer": 3}.',
                "answer_and_annotation": "Count and localize the marked items.",
            },
            "answer_gt": {"type": "integer", "value": 3},
            "annotation_gt": {"type": "bbox", "value": [5, 6, 30, 28]},
            "reward_contract": {
                "reward_contract_version": "v0",
                "answer": {"id": "answer_exact_match_v0", "type": "integer"},
                "annotation": {"id": "bbox_soft_iou_v0", "type": "bbox"},
            },
            "trace_payload": {"execution_trace": {"count": 3, "ordinal": ordinal}},
            "versions": {"task": "test-v1"},
        }
        hashes = {
            "semantic_hash": sha256_bytes(canonical_json_bytes(semantic)),
            "raw_pixel_hash": raw_hash,
            "png_hash": sha256_file(image_path),
        }
        request = {
            "task_id": task_id,
            "domain": domain,
            "scene_id": scene_id,
            "query_id": "default",
            "ordinal": ordinal,
            "seed": 7 + ordinal,
            "sample_cursor": ordinal,
            "params": {"_sample_cursor": ordinal, "query_id": "default"},
            "max_attempts": 1,
            "retry": 0,
            "hashes": hashes,
        }
        payload = {
            "schema": ARTIFACT_SCHEMA_VERSION,
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "recipe_id": RECIPE_ID,
            "recipe_digest": recipe_digest,
            "recipe_provenance": _recipe_provenance(),
            "request": request,
            "semantic_payload": semantic,
            "task_id": task_id,
            "domain": domain,
            "scene_id": scene_id,
            "query_id": "default",
            "instance_seed": 7 + ordinal,
            **{
                key: semantic[key]
                for key in (
                    "prompt",
                    "prompt_variants",
                    "answer_gt",
                    "annotation_gt",
                    "reward_contract",
                    "taxonomy",
                    "trace_payload",
                    "versions",
                )
            },
            "image": {
                "path": f"{domain}/{scene_id}/{task_id}/images/default/{filename}.png",
                "format": "png",
                "mode": "RGB",
                "width": 64,
                "height": 48,
            },
            "expected_hashes": hashes,
            "observed_hashes": hashes,
            "rendering_warnings": [],
        }
        data_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        entries.append(
            {
                "ordinal": ordinal,
                "query_id": "default",
                "data_path": f"{domain}/{scene_id}/{task_id}/data/default/{filename}.json",
                "image_path": f"{domain}/{scene_id}/{task_id}/images/default/{filename}.png",
                "data_sha256": sha256_file(data_path),
                "expected_hashes": hashes,
                "observed_hashes": hashes,
            }
        )
    (task_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": MATERIALIZATION_SCHEMA_VERSION,
                "recipe_id": RECIPE_ID,
                "recipe_digest": recipe_digest,
                "source_provenance": _source_provenance(),
                "recipe_provenance": _recipe_provenance(),
                "task_id": task_id,
                "domain": domain,
                "scene_id": scene_id,
                "request_count": REQUESTS_PER_TASK,
                "query_ids": ["default"],
                "entries": entries,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return task_id, domain, scene_id


def _calibration_summary(root: Path, task_id: str) -> None:
    calibration = root.parent / "calibration"
    calibration.mkdir(parents=True, exist_ok=True)
    (calibration / "local.json").write_text(
        json.dumps(
            {
                "schema": "trace-review-calibration-v1",
                "created_at": "2026-07-21T00:00:00+00:00",
                "model": "contributor/model",
                "endpoint": "https://private-host.invalid/v1/chat/completions",
                "authorization": "must-not-render",
                "selection": {"task_ids": [task_id], "sample_count": 1},
                "summary": {
                    "rollout_count": 1,
                    "successful_count": 1,
                    "error_count": 0,
                    "mean_answer_reward": 1.0,
                },
                "results": [{"task_id": task_id, "response": "private raw response"}],
            }
        ),
        encoding="utf-8",
    )


def test_app_browses_samples_overlays_audits_and_issue_threads(tmp_path: Path) -> None:
    root = tmp_path / "review" / "task-reviews"
    task_id, domain, scene_id = _materialize_one(root)
    _calibration_summary(root, task_id)
    app = create_review_app(
        review_root=root,
        database_path=tmp_path / "review" / "feedback" / "state.sqlite",
    )

    with TestClient(app) as client:
        index_payload = client.get("/api/index").json()
        assert index_payload["sample_count"] == REQUESTS_PER_TASK
        index = app.state.trace_review.index
        sample = next(iter(index.samples.values()))

        task_response = client.get(
            f"/domains/{domain}/scenes/{scene_id}/tasks/{task_id}"
        )
        assert task_response.status_code == 200
        assert "materialized" in task_response.text
        assert "contributor/model" in task_response.text
        assert "mean_answer_reward" in task_response.text
        assert "must-not-render" not in task_response.text
        assert "private raw response" not in task_response.text
        query_response = client.get(
            f"/domains/{domain}/scenes/{scene_id}/tasks/{task_id}/queries/default"
        )
        assert query_response.status_code == 200
        assert sample.uid in query_response.text
        sample_response = client.get(f"/samples/{sample.uid}")
        assert sample_response.status_code == 200
        assert "Count the marked items" in sample_response.text
        assert "Sanitized trace" in sample_response.text
        overlay = client.get(f"/overlays/{sample.uid}.png")
        assert overlay.status_code == 200
        assert overlay.headers["content-type"] == "image/png"

        audit_response = client.patch(
            f"/api/tasks/{task_id}/audit",
            json={
                "values": {field: True for field in AUDIT_FIELDS},
                "notes": "reviewed",
            },
        )
        assert audit_response.status_code == 200
        assert all(audit_response.json()[field] for field in AUDIT_FIELDS)

        issue_response = client.post(
            f"/samples/{sample.uid}/issues",
            data={
                "title": "Marker is hard to see",
                "body": "Increase contrast.",
                "category": "rendering",
                "severity": "issue",
                "author": "reviewer",
            },
            follow_redirects=False,
        )
        assert issue_response.status_code == 303
        issue_path = issue_response.headers["location"]
        assert client.get(issue_path).status_code == 200
        assert (
            client.post(
                f"{issue_path}/comments",
                data={"body": "Confirmed on a second display.", "author": "maintainer"},
                follow_redirects=False,
            ).status_code
            == 303
        )
        assert "Confirmed on a second display" in client.get(issue_path).text
        assert client.post("/reload", follow_redirects=False).status_code == 303


def test_missing_materialization_shows_exact_repo_relative_command(
    tmp_path: Path,
) -> None:
    app = create_review_app(
        review_root=tmp_path / "review" / "task-reviews",
        database_path=tmp_path / "review.sqlite",
    )
    task_id, taxonomy = next(iter(sorted(TASK_TAXONOMY.items())))
    with TestClient(app) as client:
        response = client.get(
            f"/domains/{taxonomy.domain}/scenes/{taxonomy.scene_id}/tasks/{task_id}"
        )
    assert response.status_code == 200
    assert "not materialized" in response.text
    assert "trace-review materialize" in response.text
    assert "docs/review/recipes/trace-review-recipe-v1" in response.text
    assert ("/" + "home/") not in response.text


def test_non_loopback_bind_requires_environment_token() -> None:
    assert validate_review_bind("127.0.0.1", environ={}) is None
    assert validate_review_bind("::1", environ={}) is None
    with pytest.raises(ValueError, match="non-loopback"):
        validate_review_bind("0.0.0.0", environ={})
    assert (
        validate_review_bind("0.0.0.0", environ={"TRACE_REVIEW_APP_TOKEN": "secret"})
        == "secret"
    )


def test_session_login_protects_media_and_ignores_query_tokens(tmp_path: Path) -> None:
    root = tmp_path / "task-reviews"
    _materialize_one(root)
    app = create_review_app(
        review_root=root,
        database_path=tmp_path / "state.sqlite",
        auth_token="correct horse",
    )
    sample = next(iter(app.state.trace_review.index.samples.values()))
    with TestClient(app) as client:
        unauthenticated = client.get(
            f"/media/{sample.media_id}?token=correct%20horse", follow_redirects=False
        )
        assert unauthenticated.status_code == 303
        assert unauthenticated.headers["location"].startswith("/login")
        assert (
            client.post("/login", data={"token": "wrong", "next": "/"}).status_code
            == 200
        )
        authenticated = client.post(
            "/login",
            data={"token": "correct horse", "next": f"/media/{sample.media_id}"},
            follow_redirects=False,
        )
        assert authenticated.status_code == 303
        assert client.get(f"/media/{sample.media_id}").status_code == 200


def test_symlink_media_escape_is_not_indexed(tmp_path: Path) -> None:
    root = tmp_path / "task-reviews"
    task_id, domain, scene_id = _materialize_one(root)
    image_path = root / domain / scene_id / task_id / "images" / "default" / "0000.png"
    outside = tmp_path / "outside.png"
    image_path.replace(outside)
    image_path.symlink_to(outside)
    index = build_review_index(root)
    assert index.samples == {}
    assert not index.tasks[task_id].materialized
    assert index.errors


def test_store_uses_public_audit_schema_and_validates_choices(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path / "state.sqlite")
    assert set(AUDIT_FIELDS) == {
        "prompt",
        "rendering",
        "answer",
        "annotation",
        "verifier_trace",
        "distribution",
        "code_docs",
        "taxonomy",
    }
    with pytest.raises(ValueError, match="category"):
        store.create_issue(title="x", body="y", category="internal-supervision")
    with pytest.raises(ValueError, match="recipe digest"):
        store.create_issue(title="x", body="y", task_id="task_x")
    store.set_asset_review("chair", kind="illustrations", decision="approve")
    store.set_asset_review("chair", kind="three_d", decision="improve")
    assert len(store.asset_reviews()) == 2


def test_index_accepts_only_manifest_declared_identity_bound_samples(
    tmp_path: Path,
) -> None:
    root = tmp_path / "task-reviews"
    task_id, domain, scene_id = _materialize_one(root)
    task_root = root / domain / scene_id / task_id
    rogue = task_root / "data" / "rogue" / "0001.json"
    rogue.parent.mkdir(parents=True)
    rogue.write_text("{}", encoding="utf-8")

    index = build_review_index(root)
    assert index.sample_count == REQUESTS_PER_TASK
    task = index.tasks[task_id]
    assert task.recipe_id == RECIPE_ID
    assert task.recipe_digest == _DIGEST_A
    assert task.source_provenance["revision"] == "0123456789abcdef"

    data_path = task_root / "data" / "default" / "0000.json"
    artifact = json.loads(data_path.read_text(encoding="utf-8"))
    artifact["prompt"] = "Tampered review-facing prompt"
    data_path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
    manifest_path = task_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entries"][0]["data_sha256"] = sha256_file(data_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    rejected = build_review_index(root)
    assert rejected.sample_count == 0
    assert not rejected.tasks[task_id].materialized
    assert any("semantic_payload" in error for error in rejected.errors)


def test_index_requires_complete_canonical_task_materialization(tmp_path: Path) -> None:
    root = tmp_path / "task-reviews"
    task_id, domain, scene_id = _materialize_one(root)
    manifest_path = root / domain / scene_id / task_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["request_count"] = REQUESTS_PER_TASK - 1
    manifest["entries"] = manifest["entries"][:-1]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    index = build_review_index(root)
    assert index.sample_count == 0
    assert not index.tasks[task_id].materialized
    assert any("request_count" in error for error in index.errors)


def test_capture_materialize_and_index_accepts_one_real_complete_task(
    tmp_path: Path,
) -> None:
    task_id, taxonomy = sorted(TASK_TAXONOMY.items())[0]

    def generator(
        requested_task: str, seed: int, params: dict, max_attempts: int
    ) -> _IntegrationOutput:
        assert requested_task == task_id
        assert max_attempts == 2
        query_id = str(params["query_id"])
        value = int(seed) % 7
        return _IntegrationOutput(
            prompt=f"Return the fixture value {value}.",
            answer_gt=TypedValue(type="integer", value=value),
            annotation_gt=TypedValue(type="bbox", value=[1, 1, 7, 6]),
            image=Image.new("RGB", (12, 9), (value * 20, 40, 80)),
            trace_payload={
                "scene_ir": {"seed": seed},
                "query_spec": {"query_id": query_id, "params": dict(params)},
                "render_spec": {"width": 12, "height": 9},
                "render_map": {"fixture": [1, 1, 7, 6]},
                "execution_trace": {"answer": value},
                "witness_symbolic": {"type": "fixture"},
                "projected_annotation": {"type": "bbox", "bbox": [1, 1, 7, 6]},
            },
            task_versions={"fixture": "v1"},
            scene_id=str(taxonomy.scene_id),
            query_id=query_id,
            prompt_variants={
                "answer_only": "Return only the answer JSON.",
                "answer_and_annotation": "Return answer and annotation JSON.",
            },
        )

    recipe_root = tmp_path / "recipe"
    output_root = tmp_path / "task-reviews"
    capture_recipe(
        (task_id,),
        recipe_root,
        max_attempts=2,
        query_ids_by_task={task_id: ("default",)},
        generator=generator,
        provenance=_integration_provenance(),
    )
    materialize_recipe(recipe_root, output_root, generator=generator)

    index = build_review_index(output_root)
    assert index.tasks[task_id].materialized
    assert index.tasks[task_id].sample_count == REQUESTS_PER_TASK
    assert [index.samples[uid].ordinal for uid in index.tasks[task_id].samples] == list(
        range(REQUESTS_PER_TASK)
    )


def test_index_rejects_coordinated_png_hash_tamper_with_stale_pixels(
    tmp_path: Path,
) -> None:
    root = tmp_path / "task-reviews"
    task_id, domain, scene_id = _materialize_one(root)
    task_root = root / domain / scene_id / task_id
    image_path = task_root / "images" / "default" / "0000.png"
    Image.new("RGB", (64, 48), (220, 20, 30)).save(image_path, format="PNG")
    changed_png_hash = sha256_file(image_path)

    data_path = task_root / "data" / "default" / "0000.json"
    artifact = json.loads(data_path.read_text(encoding="utf-8"))
    artifact["observed_hashes"]["png_hash"] = changed_png_hash
    data_path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
    manifest_path = task_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entries"][0]["observed_hashes"]["png_hash"] = changed_png_hash
    manifest["entries"][0]["data_sha256"] = sha256_file(data_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    index = build_review_index(root)
    assert index.sample_count == 0
    assert not index.tasks[task_id].materialized
    assert any("raw pixel hash" in error for error in index.errors)


def test_sample_identity_changes_with_recipe_identity(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _materialize_one(first, recipe_digest=_DIGEST_A)
    _materialize_one(second, recipe_digest=_DIGEST_B)
    first_uid = next(iter(build_review_index(first).samples))
    second_uid = next(iter(build_review_index(second).samples))
    assert first_uid != second_uid


def test_host_and_same_origin_checks_protect_local_state(tmp_path: Path) -> None:
    root = tmp_path / "task-reviews"
    task_id, _, _ = _materialize_one(root)
    app = create_review_app(
        review_root=root,
        database_path=tmp_path / "state.sqlite",
    )
    with TestClient(app) as client:
        assert client.get("/", headers={"Host": "attacker.invalid"}).status_code == 400
        hostile = client.patch(
            f"/api/tasks/{task_id}/audit",
            headers={"Origin": "https://attacker.invalid"},
            json={"values": {"prompt": True}},
        )
        assert hostile.status_code == 403
        accepted = client.patch(
            f"/api/tasks/{task_id}/audit",
            headers={"Origin": "http://testserver"},
            json={"values": {"prompt": True}},
        )
        assert accepted.status_code == 200


def test_explicit_trusted_host_supports_non_loopback_access(tmp_path: Path) -> None:
    app = create_review_app(
        review_root=tmp_path / "task-reviews",
        database_path=tmp_path / "state.sqlite",
        trusted_hosts=("reviewbox.lan",),
    )
    with TestClient(app, base_url="http://reviewbox.lan") as client:
        assert client.get("/healthz").status_code == 200
    for wildcard in ("*", "*.example.org"):
        with pytest.raises(ValueError, match="wildcard"):
            create_review_app(
                review_root=tmp_path / "other-task-reviews",
                database_path=tmp_path / "other.sqlite",
                trusted_hosts=(wildcard,),
            )


def test_old_review_database_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "old.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE task_audits (task_id TEXT PRIMARY KEY)")
    with pytest.raises(RuntimeError, match="predates recipe-bound"):
        ReviewStore(path)


def test_asset_preview_manifest_binds_catalog_id_and_contained_image(
    tmp_path: Path,
) -> None:
    root = tmp_path / "task-reviews"
    preview_root = root / "assets" / "illustrations"
    preview_root.mkdir(parents=True)
    image_path = preview_root / "contributor-preview.png"
    Image.new("RGB", (32, 24), "white").save(image_path)
    (preview_root / "contributor-preview.json").write_text(
        json.dumps(
            {
                "schema": ASSET_PREVIEW_SCHEMA,
                "kind": "illustrations",
                "asset_id": "apple",
                "image": {
                    "path": "assets/illustrations/contributor-preview.png",
                    "format": "png",
                    "mode": "RGB",
                    "width": 32,
                    "height": 24,
                    "sha256": sha256_file(image_path),
                },
            }
        ),
        encoding="utf-8",
    )
    assets = build_asset_index(root)
    apple = next(
        record for record in assets.records["illustrations"] if record.id == "apple"
    )
    assert apple.media_id in assets.media
    assert apple.rel_path == "assets/illustrations/contributor-preview.png"


def test_asset_preview_ignores_loose_image_without_manifest(tmp_path: Path) -> None:
    root = tmp_path / "task-reviews"
    preview_root = root / "assets" / "illustrations"
    preview_root.mkdir(parents=True)
    Image.new("RGB", (32, 24), "white").save(preview_root / "apple.png")

    assets = build_asset_index(root)
    apple = next(
        record for record in assets.records["illustrations"] if record.id == "apple"
    )
    assert apple.media_id == ""
    assert assets.media == {}


def test_asset_preview_rejects_svg_bytes_disguised_as_png(tmp_path: Path) -> None:
    root = tmp_path / "task-reviews"
    preview_root = root / "assets" / "illustrations"
    preview_root.mkdir(parents=True)
    image_path = preview_root / "apple-preview.png"
    image_path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="24"></svg>',
        encoding="utf-8",
    )
    (preview_root / "apple-preview.json").write_text(
        json.dumps(
            {
                "schema": ASSET_PREVIEW_SCHEMA,
                "kind": "illustrations",
                "asset_id": "apple",
                "image": {
                    "path": "assets/illustrations/apple-preview.png",
                    "format": "png",
                    "mode": "RGB",
                    "width": 32,
                    "height": 24,
                    "sha256": sha256_file(image_path),
                },
            }
        ),
        encoding="utf-8",
    )

    assets = build_asset_index(root)
    apple = next(
        record for record in assets.records["illustrations"] if record.id == "apple"
    )
    assert apple.media_id == ""
    assert assets.media == {}


def test_asset_preview_rejects_nonexact_manifest_shape(tmp_path: Path) -> None:
    root = tmp_path / "task-reviews"
    preview_root = root / "assets" / "illustrations"
    preview_root.mkdir(parents=True)
    image_path = preview_root / "apple-preview.png"
    Image.new("RGB", (32, 24), "white").save(image_path)
    (preview_root / "apple-preview.json").write_text(
        json.dumps(
            {
                "schema": ASSET_PREVIEW_SCHEMA,
                "kind": "illustrations",
                "asset_id": "apple",
                "renderer": "unversioned-extra-field",
                "image": {
                    "path": "assets/illustrations/apple-preview.png",
                    "format": "png",
                    "mode": "RGB",
                    "width": 32,
                    "height": 24,
                    "sha256": sha256_file(image_path),
                },
            }
        ),
        encoding="utf-8",
    )

    assets = build_asset_index(root)
    apple = next(
        record for record in assets.records["illustrations"] if record.id == "apple"
    )
    assert apple.media_id == ""
