from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlvr.evaluation.scripts import prepare_trace_eval_models as models

SOURCE_REVISION = "a" * 40
SLUG = "fixture-validation-model"
REPO_ID = "fixture/model"


def _write_model(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text('{"model_type":"fixture"}\n', encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"small-offline-weight-fixture")


def _content_revision(path: Path) -> str:
    hashes = models._relative_hashes(path, models._validation_snapshot_files(path))
    return models._content_revision(hashes)


def _write_suite(
    path: Path,
    *,
    inference_revision: str,
    slug: str = SLUG,
    repo_id: str = REPO_ID,
    source_revision: str = SOURCE_REVISION,
) -> Path:
    suite = {
        "schema_version": "trace-validation-suite-v1",
        "models": [
            {
                "slug": slug,
                "label": "Fixture",
                "repo_id": repo_id,
                "revision": source_revision,
                **(
                    {"runtime_view_revision": inference_revision}
                    if inference_revision != source_revision
                    else {}
                ),
            }
        ],
    }
    path.write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")
    return path


def test_canonical_validation_registry_comes_from_suite() -> None:
    registry = models.load_validation_models()

    assert [model.slug for model in registry] == [
        "qwen2.5-vl-3b-base",
        "trace-qwen2.5-vl-3b",
        "qwen2.5-vl-7b-base",
        "trace-qwen2.5-vl-7b",
        "game-rl-qwen2.5-vl-7b",
        "sphinx-qwen2.5-vl-7b",
        "pcgrpo-qwen2.5-vl-7b",
        "vero-qwen2.5-vl-7b",
    ]
    trace_3b = next(model for model in registry if model.slug == "trace-qwen2.5-vl-3b")
    assert trace_3b.repo_id == "maveryn/trace-qwen2.5-vl-3b"
    assert trace_3b.source_revision == "2ec2374d5c219e6b12e26bda93d3b3adeb1e30c5"
    assert trace_3b.inference_revision == (
        "sha256set:fd7d9ef4dd828eb950ce29c8ccde0432ccd31420529d4f023300ede928d070a1"
    )


def test_register_validation_local_writes_and_verifies_exact_inventory(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model"
    _write_model(model_path)
    inference_revision = _content_revision(model_path)
    suite_path = _write_suite(
        tmp_path / "suite.json", inference_revision=inference_revision
    )

    assert (
        models.register_validation_local(SLUG, model_path, suite_path)
        == inference_revision
    )
    marker = json.loads((model_path / models.MARKER_NAME).read_text(encoding="utf-8"))
    assert marker["slug"] == SLUG
    assert marker["source"] == REPO_ID
    assert marker["source_revision"] == SOURCE_REVISION
    assert marker["resolved_commit"] == SOURCE_REVISION
    assert marker["immutable_revision"] == inference_revision
    assert marker["inference_revision"] == inference_revision
    assert marker["content_revision"] == inference_revision
    assert marker["file_count"] == 2
    assert sorted(marker["file_sha256"]) == ["config.json", "model.safetensors"]

    models.verify_validation([(SLUG, model_path)], suite_path)

    (model_path / "unexpected.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="file set mismatch"):
        models.verify_validation([(SLUG, model_path)], suite_path)


def test_deep_verifier_rejects_rehashed_content_outside_suite_identity(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model"
    _write_model(model_path)
    inference_revision = _content_revision(model_path)
    suite_path = _write_suite(
        tmp_path / "suite.json", inference_revision=inference_revision
    )
    models.register_validation_local(SLUG, model_path, suite_path)

    (model_path / "model.safetensors").write_bytes(b"different-arbitrary-weights")
    marker_path = model_path / models.MARKER_NAME
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    hashes = models._relative_hashes(
        model_path, models._validation_snapshot_files(model_path)
    )
    marker["file_sha256"] = hashes
    marker["file_count"] = len(hashes)
    marker["content_revision"] = models._content_revision(hashes)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(RuntimeError, match="suite inference revision"):
        models.verify_model_directory(
            model_path,
            SLUG,
            REPO_ID,
            SOURCE_REVISION,
            inference_revision,
        )


def test_deep_verifier_rejects_unsafe_marker_paths(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    _write_model(model_path)
    inference_revision = _content_revision(model_path)
    suite_path = _write_suite(
        tmp_path / "suite.json", inference_revision=inference_revision
    )
    models.register_validation_local(SLUG, model_path, suite_path)

    marker_path = model_path / models.MARKER_NAME
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["file_sha256"]["../escape"] = marker["file_sha256"].pop("config.json")
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(RuntimeError, match="unsafe model content path"):
        models.verify_model_directory(
            model_path,
            SLUG,
            REPO_ID,
            SOURCE_REVISION,
            inference_revision,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source", "attacker/model", "source mismatch"),
        ("resolved_commit", "b" * 40, "resolved commit mismatch"),
    ],
)
def test_deep_verifier_rejects_false_source_identity(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    model_path = tmp_path / "model"
    _write_model(model_path)
    inference_revision = _content_revision(model_path)
    suite_path = _write_suite(
        tmp_path / "suite.json", inference_revision=inference_revision
    )
    models.register_validation_local(SLUG, model_path, suite_path)

    marker_path = model_path / models.MARKER_NAME
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker[field] = value
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        models.verify_model_directory(
            model_path,
            SLUG,
            REPO_ID,
            SOURCE_REVISION,
            inference_revision,
        )


def test_download_validation_resolves_commit_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = models.ValidationModel(
        slug=SLUG,
        repo_id=REPO_ID,
        source_revision=SOURCE_REVISION,
        inference_revision=SOURCE_REVISION,
    )
    info = SimpleNamespace(
        sha=SOURCE_REVISION,
        siblings=[
            SimpleNamespace(rfilename=models.MARKER_NAME),
            SimpleNamespace(rfilename="config.json"),
            SimpleNamespace(rfilename="model.safetensors"),
        ],
    )

    class FakeApi:
        def __init__(self, *, token: str | None) -> None:
            assert token is None

        def model_info(
            self, repo_id: str, *, revision: str, token: str | None
        ) -> object:
            assert (repo_id, revision, token) == (REPO_ID, SOURCE_REVISION, None)
            return info

    def fake_snapshot_download(
        *, repo_id: str, revision: str, local_dir: Path, token: str | None
    ) -> None:
        assert (repo_id, revision, token) == (REPO_ID, SOURCE_REVISION, None)
        _write_model(Path(local_dir))
        (Path(local_dir) / models.MARKER_NAME).write_text(
            '{"source":"pre-existing-repository-marker"}\n', encoding="utf-8"
        )

    monkeypatch.delenv("TRACE_TEST_HF_TOKEN", raising=False)
    monkeypatch.setattr(models, "HfApi", FakeApi)
    monkeypatch.setattr(models, "snapshot_download", fake_snapshot_download)

    model_root = tmp_path / "models"
    models.download_validation([model], model_root, "TRACE_TEST_HF_TOKEN")
    marker = models.verify_model_directory(
        model_root / SLUG,
        SLUG,
        REPO_ID,
        SOURCE_REVISION,
        SOURCE_REVISION,
    )
    assert marker["model_origin"] == "public_download"
    assert marker["resolved_commit"] == SOURCE_REVISION
    assert marker["content_revision"].startswith("sha256set:")
    assert marker["file_count"] == 2


def test_game_rl_download_compatibility_view_is_exact(tmp_path: Path) -> None:
    model_path = tmp_path / "game-rl"
    model_path.mkdir()
    source_config = {
        "do_convert_rgb": True,
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_processor_type": "Qwen2_5_VLImageProcessor",
        "image_std": [0.26862954, 0.26130258, 0.27577711],
        "max_pixels": 12845056,
        "merge_size": 2,
        "min_pixels": 3136,
        "patch_size": 14,
        "processor_class": "Qwen2_5_VLProcessor",
        "resample": 3,
        "rescale_factor": 0.00392156862745098,
        "size": {"longest_edge": 12845056, "shortest_edge": 3136},
        "temporal_patch_size": 2,
    }
    config_path = model_path / "preprocessor_config.json"
    config_path.write_text(json.dumps(source_config, indent=2) + "\n", encoding="utf-8")
    assert models._sha256_file(config_path) == models.GAME_RL_PREPROCESSOR_SOURCE_SHA256
    model = models.ValidationModel(
        slug=models.GAME_RL_VALIDATION_SLUG,
        repo_id=models.GAME_RL_REPO_ID,
        source_revision=models.GAME_RL_SOURCE_REVISION,
        inference_revision=models.GAME_RL_INFERENCE_REVISION,
    )

    assert (
        models._apply_validation_compatibility_view(model, model_path)
        == models.GAME_RL_COMPATIBILITY_VIEW
    )
    assert (
        models._sha256_file(config_path) == models.GAME_RL_PREPROCESSOR_RUNTIME_SHA256
    )
    assert (
        json.loads(config_path.read_text(encoding="utf-8"))["image_processor_type"]
        == "Qwen2VLImageProcessor"
    )
    assert (
        models._apply_validation_compatibility_view(model, model_path)
        == models.GAME_RL_COMPATIBILITY_VIEW
    )


def test_register_validation_local_rejects_commit_addressed_model(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model"
    _write_model(model_path)
    suite_path = _write_suite(
        tmp_path / "suite.json", inference_revision=SOURCE_REVISION
    )

    with pytest.raises(RuntimeError, match="use download-validation"):
        models.register_validation_local(SLUG, model_path, suite_path)


def test_registration_rejects_symbolic_links(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    _write_model(model_path)
    external = tmp_path / "external.json"
    external.write_text("{}\n", encoding="utf-8")
    (model_path / "processor_config.json").symlink_to(external)

    with pytest.raises(RuntimeError, match="symbolic link"):
        models._validation_snapshot_files(model_path)
