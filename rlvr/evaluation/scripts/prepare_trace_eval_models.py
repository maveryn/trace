#!/usr/bin/env python3
"""Prepare content-bound models for external and TRACE validation evaluation."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from huggingface_hub import HfApi, snapshot_download

MARKER_NAME = ".trace_model_revision.json"
VALIDATION_SUITE_PATH = (
    Path(__file__).resolve().parents[1] / "trace_validation" / "suite.v1.json"
)
HEX_REVISION_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
SHA256SET_REVISION_RE = re.compile(r"^sha256set:[0-9a-f]{64}$")
SAFE_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
GAME_RL_VALIDATION_SLUG = "game-rl-qwen2.5-vl-7b"
GAME_RL_REPO_ID = "OpenMOSS-Team/Game-RL-Qwen2.5-VL-7B"
GAME_RL_SOURCE_REVISION = "205b5934ce70504cfd6ae26b16f705d0b98b9306"
GAME_RL_INFERENCE_REVISION = (
    "sha256set:2a805cbedc07225555712644c3569019da15a30108bb50b2dbe60d9562d24b2f"
)
GAME_RL_PREPROCESSOR_SOURCE_SHA256 = (
    "59689b97e9256755ad0e19fd5211ac6678c0d800b138f060065da09b29ea0b96"
)
GAME_RL_PREPROCESSOR_RUNTIME_SHA256 = (
    "549c158011407dfb750d9ec578047cf76f5bfe365cd0aa069a50137d3f98d9dd"
)
GAME_RL_COMPATIBILITY_VIEW = "game-rl-qwen2vl-image-processor-alias-v1"


@dataclass(frozen=True)
class PublicModel:
    slug: str
    repo_id: str
    revision: str
    target_name: str


@dataclass(frozen=True)
class ValidationModel:
    """One model pin resolved from the public TRACE validation suite."""

    slug: str
    repo_id: str
    source_revision: str
    inference_revision: str

    @property
    def target_name(self) -> str:
        return self.slug


PUBLIC_MODELS = (
    PublicModel(
        "qwen25vl3b-base",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "66285546d2b821cf421d4f5eb2576359d3770cd3",
        "qwen25vl3b-base",
    ),
    PublicModel(
        "qwen25vl7b-base",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "cc594898137f460bfe9f0759e9844b3ce807cfb5",
        "qwen25vl7b-base",
    ),
    PublicModel(
        "game-rl-qwen25vl7b",
        "OpenMOSS-Team/Game-RL-Qwen2.5-VL-7B",
        "205b5934ce70504cfd6ae26b16f705d0b98b9306",
        "game-rl-qwen25vl7b",
    ),
    PublicModel(
        "sphinx-qwen7b-500",
        "xashru/sphinx_qwen7b_500",
        "6ffefb03d5cb0767683bfb42a084ea86b707ef9a",
        "sphinx-qwen7b-500",
    ),
    PublicModel(
        "pcgrpo-qwen25vl7b-jigsaw-care",
        "armenjeddi/PCGRPO-Qwen2.5-VL-7B-Jigsaw-with-curriculum-with-grpo-care",
        "921bbced4176f5d362e98c843a57656c5d78dad7",
        "pcgrpo-qwen25vl7b-jigsaw-care",
    ),
    PublicModel(
        "vero-qwen25-7b",
        "zlab-princeton/Vero-Qwen25-7B",
        "180e84be5acb2aa887cf51015b84b6a6e453ee90",
        "vero-qwen25-7b",
    ),
    PublicModel(
        "qwen3-32b-judge",
        "Qwen/Qwen3-32B",
        "9216db5781bf21249d130ec9da846c4624c16137",
        "qwen3-32b-judge",
    ),
)


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not load JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def load_validation_models(
    suite_path: Path = VALIDATION_SUITE_PATH,
) -> tuple[ValidationModel, ...]:
    """Resolve the canonical model registry from ``suite.v1.json``."""

    suite = _load_json_object(suite_path)
    if suite.get("schema_version") != "trace-validation-suite-v1":
        raise RuntimeError(f"unexpected TRACE validation suite schema: {suite_path}")
    rows = suite.get("models")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(
            f"TRACE validation suite has no model registry: {suite_path}"
        )

    models: list[ValidationModel] = []
    seen_slugs: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"TRACE validation model {index} is not an object")
        slug = str(row.get("slug") or "")
        repo_id = str(row.get("repo_id") or "")
        source_revision = str(row.get("revision") or "")
        inference_revision = str(row.get("runtime_view_revision") or source_revision)
        if not SAFE_SLUG_RE.fullmatch(slug):
            raise RuntimeError(f"unsafe TRACE validation model slug: {slug!r}")
        if slug in seen_slugs:
            raise RuntimeError(f"duplicate TRACE validation model slug: {slug}")
        if (
            not repo_id
            or "@" in repo_id
            or repo_id.startswith("/")
            or repo_id.endswith("/")
            or repo_id.count("/") != 1
        ):
            raise RuntimeError(
                f"invalid TRACE validation model repository for {slug}: {repo_id!r}"
            )
        if not HEX_REVISION_RE.fullmatch(source_revision):
            raise RuntimeError(
                f"invalid TRACE validation source revision for {slug}: "
                f"{source_revision!r}"
            )
        if not (
            HEX_REVISION_RE.fullmatch(inference_revision)
            or SHA256SET_REVISION_RE.fullmatch(inference_revision)
        ):
            raise RuntimeError(
                f"invalid TRACE validation inference revision for {slug}: "
                f"{inference_revision!r}"
            )
        seen_slugs.add(slug)
        models.append(
            ValidationModel(
                slug=slug,
                repo_id=repo_id,
                source_revision=source_revision,
                inference_revision=inference_revision,
            )
        )
    return tuple(models)


def _validation_model(
    slug: str, suite_path: Path = VALIDATION_SUITE_PATH
) -> ValidationModel:
    matches = [
        model for model in load_validation_models(suite_path) if model.slug == slug
    ]
    if not matches:
        raise RuntimeError(f"unknown TRACE validation model slug: {slug}")
    return matches[0]


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_revision(file_sha256: Mapping[str, str]) -> str:
    fingerprint = hashlib.sha256(
        _canonical_json(dict(file_sha256)).encode("utf-8")
    ).hexdigest()
    return f"sha256set:{fingerprint}"


def _safe_relative_file_name(value: object) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"model content path is not a string: {value!r}")
    relative = Path(value)
    if (
        not value
        or "\\" in value
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != value
        or value == "."
    ):
        raise RuntimeError(f"unsafe model content path in marker: {value!r}")
    return value


def _snapshot_files(path: Path) -> list[Path]:
    files: list[Path] = []
    for candidate in path.rglob("*"):
        relative = candidate.relative_to(path)
        if (
            candidate.name == MARKER_NAME
            or candidate.name.startswith(".trace_model_revision.")
            or ".cache" in relative.parts
        ):
            continue
        if candidate.is_file():
            files.append(candidate)
    return sorted(files, key=lambda item: item.relative_to(path).as_posix())


def _validation_snapshot_files(path: Path) -> list[Path]:
    files: list[Path] = []
    for candidate in path.rglob("*"):
        relative = candidate.relative_to(path)
        if candidate.name == MARKER_NAME or ".cache" in relative.parts:
            continue
        _safe_relative_file_name(relative.as_posix())
        if candidate.is_symlink():
            raise RuntimeError(f"model snapshot contains a symbolic link: {candidate}")
        if candidate.is_file():
            files.append(candidate)
    return sorted(files, key=lambda item: item.relative_to(path).as_posix())


def _relative_hashes(path: Path, files: Iterable[Path]) -> dict[str, str]:
    return {item.relative_to(path).as_posix(): _sha256_file(item) for item in files}


def _recorded_hashes(marker: Mapping[str, object], slug: str) -> dict[str, str]:
    raw_hashes = marker.get("file_sha256")
    if not isinstance(raw_hashes, dict) or not raw_hashes:
        raise RuntimeError(
            f"deep verification requires immutable per-file hashes for {slug}"
        )
    hashes: dict[str, str] = {}
    for raw_relative, raw_digest in raw_hashes.items():
        relative = _safe_relative_file_name(raw_relative)
        if not isinstance(raw_digest, str) or not re.fullmatch(
            r"[0-9a-f]{64}", raw_digest
        ):
            raise RuntimeError(f"invalid model content hash in marker: {relative!r}")
        hashes[relative] = raw_digest
    if len(hashes) != len(raw_hashes):
        raise RuntimeError(f"duplicate normalized model content paths for {slug}")
    return hashes


def _repository_file_names(info: object, repo_id: str) -> set[str]:
    siblings = getattr(info, "siblings", None)
    if not isinstance(siblings, (list, tuple)) or not siblings:
        raise RuntimeError(f"pinned repository returned no file inventory: {repo_id}")
    result: set[str] = set()
    for sibling in siblings:
        name = str(getattr(sibling, "rfilename", "") or "")
        relative = Path(name)
        if (
            not name
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != name
        ):
            raise RuntimeError(
                f"unsafe file path in pinned repository {repo_id}: {name!r}"
            )
        result.add(name)
    if len(result) != len(siblings):
        raise RuntimeError(
            f"duplicate file paths in pinned repository inventory: {repo_id}"
        )
    return result


def _validate_model_shape(path: Path) -> None:
    if not (path / "config.json").is_file():
        raise RuntimeError(f"missing model config: {path / 'config.json'}")
    weights = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
    if not weights or any(item.stat().st_size == 0 for item in weights):
        raise RuntimeError(f"missing or empty model weights under {path}")


def _write_marker(path: Path, marker: dict[str, object]) -> None:
    target = path / MARKER_NAME
    temporary = target.with_suffix(f".tmp.{os.getpid()}")
    temporary.write_text(_canonical_json(marker) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def _read_marker(path: Path) -> dict[str, object]:
    marker_path = path / MARKER_NAME
    if not marker_path.is_file():
        raise RuntimeError(
            f"missing model provenance marker: {marker_path}; download or register the model first"
        )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if not isinstance(marker, dict):
        raise RuntimeError(f"invalid model provenance marker: {marker_path}")
    return marker


def verify_model_directory(
    path: Path,
    slug: str,
    repo_id: str,
    source_revision: str,
    inference_revision: str,
) -> dict[str, object]:
    """Deeply verify a suite-bound model directory and return its marker.

    Validation is offline: the download or registration step records the exact
    file inventory after resolving the suite's immutable source commit, and
    this function checks that receipt against the current directory.
    """

    _validate_model_shape(path)
    marker = _read_marker(path)
    if marker.get("schema_version") != "trace-model-revision-v1":
        raise RuntimeError(f"invalid model provenance marker schema for {slug}: {path}")
    if marker.get("slug") != slug:
        raise RuntimeError(
            f"model marker slug mismatch for {path}: {marker.get('slug')} != {slug}"
        )
    if str(marker.get("immutable_revision") or "") != inference_revision:
        raise RuntimeError(f"model inference revision mismatch for {slug}")
    if str(marker.get("inference_revision") or "") != inference_revision:
        raise RuntimeError(
            f"model marker does not bind the inference revision for {slug}"
        )

    source = str(marker.get("source") or "")
    if source not in {repo_id, f"{repo_id}@{source_revision}"}:
        raise RuntimeError(
            f"model marker source mismatch for {slug}: expected "
            f"{repo_id}@{source_revision}"
        )
    if str(marker.get("source_revision") or "") != source_revision:
        raise RuntimeError(f"model marker source revision mismatch for {slug}")
    if str(marker.get("resolved_commit") or "") != source_revision:
        raise RuntimeError(f"model resolved commit mismatch for {slug}")
    if marker.get("model_origin") not in {
        "public_download",
        "public_download_runtime_view",
        "local_registration",
    }:
        raise RuntimeError(f"invalid model marker origin for {slug}")
    model_origin = marker.get("model_origin")
    game_rl_identity = (
        slug,
        repo_id,
        source_revision,
        inference_revision,
    ) == (
        GAME_RL_VALIDATION_SLUG,
        GAME_RL_REPO_ID,
        GAME_RL_SOURCE_REVISION,
        GAME_RL_INFERENCE_REVISION,
    )
    if model_origin == "public_download_runtime_view":
        if (
            not game_rl_identity
            or marker.get("compatibility_view") != GAME_RL_COMPATIBILITY_VIEW
        ):
            raise RuntimeError(f"invalid compatibility view marker for {slug}")
    elif game_rl_identity and model_origin == "public_download":
        raise RuntimeError(f"missing Game-RL compatibility view marker for {slug}")

    recorded_hashes = _recorded_hashes(marker, slug)
    file_count = marker.get("file_count")
    if (
        isinstance(file_count, bool)
        or not isinstance(file_count, int)
        or file_count != len(recorded_hashes)
    ):
        raise RuntimeError(f"model snapshot file count mismatch for {slug}")

    current_files = {
        item.relative_to(path).as_posix(): item
        for item in _validation_snapshot_files(path)
    }
    if set(current_files) != set(recorded_hashes):
        missing = sorted(set(recorded_hashes) - set(current_files))
        added = sorted(set(current_files) - set(recorded_hashes))
        raise RuntimeError(
            f"model snapshot file set mismatch for {slug}: "
            f"missing={missing} added={added}"
        )
    current_hashes: dict[str, str] = {}
    for relative, expected_hash in recorded_hashes.items():
        candidate = current_files[relative]
        actual_hash = _sha256_file(candidate)
        if actual_hash != expected_hash:
            raise RuntimeError(f"model content hash mismatch: {candidate}")
        current_hashes[relative] = actual_hash

    content_revision = _content_revision(current_hashes)
    if str(marker.get("content_revision") or "") != content_revision:
        raise RuntimeError(f"model marker content revision mismatch for {slug}")
    if SHA256SET_REVISION_RE.fullmatch(inference_revision):
        if content_revision != inference_revision:
            raise RuntimeError(
                f"model content does not match the suite inference revision for {slug}: "
                f"{content_revision} != {inference_revision}"
            )
    elif inference_revision != source_revision:
        raise RuntimeError(
            f"commit-addressed inference revision differs from source revision for {slug}"
        )
    return marker


def _token(token_env: str) -> str | None:
    return os.environ.get(token_env, "").strip() or None


def _apply_validation_compatibility_view(
    model: ValidationModel, path: Path
) -> str | None:
    """Materialize a reviewed runtime-only compatibility view when required."""

    if model.slug != GAME_RL_VALIDATION_SLUG:
        return None
    expected_identity = (
        GAME_RL_REPO_ID,
        GAME_RL_SOURCE_REVISION,
        GAME_RL_INFERENCE_REVISION,
    )
    actual_identity = (
        model.repo_id,
        model.source_revision,
        model.inference_revision,
    )
    if actual_identity != expected_identity:
        raise RuntimeError(
            "the Game-RL compatibility transform is stale relative to the "
            "validation suite"
        )

    config_path = path / "preprocessor_config.json"
    if not config_path.is_file():
        raise RuntimeError(f"missing Game-RL preprocessor config: {config_path}")
    current_sha256 = _sha256_file(config_path)
    if current_sha256 == GAME_RL_PREPROCESSOR_RUNTIME_SHA256:
        return GAME_RL_COMPATIBILITY_VIEW
    if current_sha256 != GAME_RL_PREPROCESSOR_SOURCE_SHA256:
        raise RuntimeError(
            f"unexpected Game-RL preprocessor config SHA-256: {current_sha256}"
        )
    config = _load_json_object(config_path)
    if config.get("image_processor_type") != "Qwen2_5_VLImageProcessor":
        raise RuntimeError("unexpected Game-RL image processor type")
    config["image_processor_type"] = "Qwen2VLImageProcessor"
    temporary = config_path.with_suffix(f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, config_path)
    transformed_sha256 = _sha256_file(config_path)
    if transformed_sha256 != GAME_RL_PREPROCESSOR_RUNTIME_SHA256:
        raise RuntimeError(
            "Game-RL compatibility transform produced an unexpected config hash"
        )
    return GAME_RL_COMPATIBILITY_VIEW


def _write_validation_marker(
    path: Path,
    model: ValidationModel,
    hashes: Mapping[str, str],
    *,
    model_origin: str,
    compatibility_view: str | None = None,
) -> None:
    content_revision = _content_revision(hashes)
    marker: dict[str, object] = {
        "schema_version": "trace-model-revision-v1",
        "slug": model.slug,
        "source": model.repo_id,
        "source_revision": model.source_revision,
        "resolved_commit": model.source_revision,
        "immutable_revision": model.inference_revision,
        "inference_revision": model.inference_revision,
        "content_revision": content_revision,
        "model_origin": model_origin,
        "file_count": len(hashes),
        "file_sha256": dict(hashes),
        "registered_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    if compatibility_view is not None:
        marker["compatibility_view"] = compatibility_view
    _write_marker(path, marker)


def download_validation(
    models: Iterable[ValidationModel], model_root: Path, token_env: str
) -> None:
    """Download and content-bind suite models at their immutable source commits."""

    token = _token(token_env)
    api = HfApi(token=token)
    model_root.mkdir(parents=True, exist_ok=True)
    for model in models:
        info = api.model_info(
            model.repo_id, revision=model.source_revision, token=token
        )
        if str(info.sha) != model.source_revision:
            raise RuntimeError(
                f"{model.repo_id}@{model.source_revision} resolved to unexpected "
                f"commit {info.sha}"
            )
        expected_files = _repository_file_names(info, model.repo_id)
        expected_files.discard(MARKER_NAME)
        target = model_root / model.target_name
        print(
            f"[validation-model:download] "
            f"{model.repo_id}@{model.source_revision} -> {target}"
        )
        snapshot_download(
            repo_id=model.repo_id,
            revision=model.source_revision,
            local_dir=target,
            token=token,
        )
        compatibility_view = _apply_validation_compatibility_view(model, target)
        _validate_model_shape(target)
        current_files = {
            item.relative_to(target).as_posix(): item
            for item in _validation_snapshot_files(target)
        }
        if set(current_files) != expected_files:
            missing = sorted(expected_files - set(current_files))
            stale = sorted(set(current_files) - expected_files)
            raise RuntimeError(
                f"downloaded model file set differs from pinned repository "
                f"{model.repo_id}: missing={missing} stale={stale}"
            )
        hashes = _relative_hashes(target, current_files.values())
        if not hashes:
            raise RuntimeError(f"downloaded model snapshot contains no files: {target}")
        content_revision = _content_revision(hashes)
        if SHA256SET_REVISION_RE.fullmatch(model.inference_revision):
            if content_revision != model.inference_revision:
                raise RuntimeError(
                    f"downloaded snapshot for {model.slug} does not match its suite "
                    f"runtime view: {content_revision} != {model.inference_revision}; "
                    "prepare the compatibility view and use register-validation-local"
                )
        _write_validation_marker(
            target,
            model,
            hashes,
            model_origin=(
                "public_download_runtime_view"
                if compatibility_view is not None
                else "public_download"
            ),
            compatibility_view=compatibility_view,
        )
        verify_model_directory(
            target,
            model.slug,
            model.repo_id,
            model.source_revision,
            model.inference_revision,
        )


def register_validation_local(
    slug: str,
    path: Path,
    suite_path: Path = VALIDATION_SUITE_PATH,
) -> str:
    """Register a suite-pinned local runtime view by exact content identity."""

    model = _validation_model(slug, suite_path)
    if not SHA256SET_REVISION_RE.fullmatch(model.inference_revision):
        raise RuntimeError(
            f"{slug} is commit-addressed in the validation suite; use "
            "download-validation so its repository commit is resolved"
        )
    _validate_model_shape(path)
    hashes = _relative_hashes(path, _validation_snapshot_files(path))
    if not hashes:
        raise RuntimeError(f"local model snapshot contains no files: {path}")
    content_revision = _content_revision(hashes)
    if content_revision != model.inference_revision:
        raise RuntimeError(
            f"local model content does not match the suite inference revision for "
            f"{slug}: {content_revision} != {model.inference_revision}"
        )
    _write_validation_marker(path, model, hashes, model_origin="local_registration")
    verify_model_directory(
        path,
        model.slug,
        model.repo_id,
        model.source_revision,
        model.inference_revision,
    )
    print(
        f"[validation-model:registered] slug={slug} "
        f"revision={model.inference_revision} path={path}"
    )
    return model.inference_revision


def verify_validation(
    entries: Iterable[tuple[str, Path]],
    suite_path: Path = VALIDATION_SUITE_PATH,
) -> None:
    """Verify suite model entries without trusting caller-provided revisions."""

    registry = {model.slug: model for model in load_validation_models(suite_path)}
    for slug, path in entries:
        try:
            model = registry[slug]
        except KeyError as error:
            raise RuntimeError(
                f"unknown TRACE validation model slug: {slug}"
            ) from error
        verify_model_directory(
            path,
            model.slug,
            model.repo_id,
            model.source_revision,
            model.inference_revision,
        )
        print(
            f"[validation-model:ok] slug={slug} "
            f"revision={model.inference_revision} path={path}"
        )


def download_public(
    models: Iterable[PublicModel], model_root: Path, token_env: str
) -> None:
    token = _token(token_env)
    api = HfApi(token=token)
    model_root.mkdir(parents=True, exist_ok=True)
    for model in models:
        info = api.model_info(model.repo_id, revision=model.revision, token=token)
        if str(info.sha) != model.revision:
            raise RuntimeError(
                f"{model.repo_id}@{model.revision} resolved to unexpected commit {info.sha}"
            )
        expected_files = _repository_file_names(info, model.repo_id)
        target = model_root / model.target_name
        print(f"[model:download] {model.repo_id}@{model.revision} -> {target}")
        snapshot_download(
            repo_id=model.repo_id,
            revision=model.revision,
            local_dir=target,
            token=token,
        )
        _validate_model_shape(target)
        snapshot_files = _snapshot_files(target)
        current_files = {
            item.relative_to(target).as_posix(): item for item in snapshot_files
        }
        if set(current_files) != expected_files:
            missing = sorted(expected_files - set(current_files))
            stale = sorted(set(current_files) - expected_files)
            raise RuntimeError(
                f"downloaded model file set differs from pinned repository {model.repo_id}: "
                f"missing={missing} stale={stale}"
            )
        hashes = _relative_hashes(target, current_files.values())
        if not hashes:
            raise RuntimeError(f"downloaded model snapshot contains no files: {target}")
        _write_marker(
            target,
            {
                "schema_version": "trace-model-revision-v1",
                "slug": model.slug,
                "source": model.repo_id,
                "immutable_revision": model.revision,
                "resolved_commit": str(info.sha),
                "model_origin": "public_download",
                "file_count": len(hashes),
                "file_sha256": hashes,
                "registered_at": dt.datetime.now(dt.timezone.utc).isoformat(
                    timespec="seconds"
                ),
            },
        )


def register_local(slug: str, path: Path, source: str) -> str:
    _validate_model_shape(path)
    hashes = _relative_hashes(path, _snapshot_files(path))
    if not hashes:
        raise RuntimeError(f"local model snapshot contains no files: {path}")
    fingerprint = hashlib.sha256(_canonical_json(hashes).encode("utf-8")).hexdigest()
    revision = f"sha256set:{fingerprint}"
    _write_marker(
        path,
        {
            "schema_version": "trace-model-revision-v1",
            "slug": slug,
            "source": source,
            "immutable_revision": revision,
            "model_origin": "local_registration",
            "file_count": len(hashes),
            "file_sha256": hashes,
            "registered_at": dt.datetime.now(dt.timezone.utc).isoformat(
                timespec="seconds"
            ),
        },
    )
    print(f"[model:registered] slug={slug} revision={revision} path={path}")
    return revision


def parse_entry(value: str) -> tuple[str, Path, str]:
    parts = value.split("=", 2)
    if len(parts) != 3 or not all(part.strip() for part in parts):
        raise argparse.ArgumentTypeError("--entry must be slug=path=immutable_revision")
    return parts[0].strip(), Path(parts[1]).expanduser(), parts[2].strip()


def parse_validation_entry(value: str) -> tuple[str, Path]:
    parts = value.split("=", 1)
    if len(parts) != 2 or not all(part.strip() for part in parts):
        raise argparse.ArgumentTypeError("--entry must be slug=path")
    return parts[0].strip(), Path(parts[1]).expanduser()


def verify(entries: Iterable[tuple[str, Path, str]], deep: bool) -> None:
    for slug, path, expected_revision in entries:
        _validate_model_shape(path)
        marker = _read_marker(path)
        if marker.get("slug") != slug:
            raise RuntimeError(
                f"model marker slug mismatch for {path}: {marker.get('slug')} != {slug}"
            )
        actual_revision = str(marker.get("immutable_revision") or "")
        if actual_revision != expected_revision:
            raise RuntimeError(
                f"model revision mismatch for {slug}: {actual_revision} != {expected_revision}"
            )
        if deep:
            recorded_hashes = marker.get("file_sha256")
            if not isinstance(recorded_hashes, dict) or not recorded_hashes:
                raise RuntimeError(
                    f"deep verification requires immutable per-file hashes for {slug}: {path}"
                )
            normalized_hashes: dict[str, str] = {}
            for relative, expected_hash in recorded_hashes.items():
                relative_text = str(relative)
                relative_path = Path(relative_text)
                if (
                    not relative_text
                    or relative_path.is_absolute()
                    or ".." in relative_path.parts
                    or relative_path.as_posix() != relative_text
                ):
                    raise RuntimeError(
                        f"unsafe model content path in marker: {relative_text!r}"
                    )
                expected_hash_text = str(expected_hash)
                if len(expected_hash_text) != 64 or any(
                    character not in "0123456789abcdef"
                    for character in expected_hash_text
                ):
                    raise RuntimeError(
                        f"invalid model content hash in marker: {relative_text!r}"
                    )
                normalized_hashes[relative_text] = expected_hash_text

            if marker.get("model_origin") == "public_download" or bool(
                marker.get("resolved_commit")
            ):
                if str(marker.get("resolved_commit") or "") != expected_revision:
                    raise RuntimeError(f"model resolved commit mismatch for {slug}")
            current_names = {
                item.relative_to(path).as_posix(): item
                for item in _snapshot_files(path)
            }
            if set(current_names) != set(normalized_hashes):
                missing = sorted(set(normalized_hashes) - set(current_names))
                added = sorted(set(current_names) - set(normalized_hashes))
                raise RuntimeError(
                    f"model snapshot file set mismatch for {slug}: missing={missing} added={added}"
                )
            if marker.get("file_count") is not None and int(
                marker["file_count"]
            ) != len(normalized_hashes):
                raise RuntimeError(f"model snapshot file count mismatch for {slug}")

            for relative, expected_hash in normalized_hashes.items():
                candidate = current_names[relative]
                if not candidate.is_file() or _sha256_file(candidate) != expected_hash:
                    raise RuntimeError(f"model content hash mismatch: {candidate}")
        print(f"[model:ok] slug={slug} revision={actual_revision} path={path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download-public")
    download.add_argument("--model-root", type=Path, required=True)
    download.add_argument("--token-env", default="HF_TOKEN")
    download.add_argument("--only", action="append", default=[])
    register = commands.add_parser("register-local")
    register.add_argument("--slug", required=True)
    register.add_argument("--path", type=Path, required=True)
    register.add_argument("--source", required=True)
    check = commands.add_parser("verify")
    check.add_argument("--entry", action="append", type=parse_entry, required=True)
    check.add_argument("--deep", action="store_true")
    validation_download = commands.add_parser("download-validation")
    validation_download.add_argument("--model-root", type=Path, required=True)
    validation_download.add_argument(
        "--suite", type=Path, default=VALIDATION_SUITE_PATH
    )
    validation_download.add_argument("--token-env", default="HF_TOKEN")
    validation_download.add_argument("--only", action="append", default=[])
    validation_register = commands.add_parser("register-validation-local")
    validation_register.add_argument("--slug", required=True)
    validation_register.add_argument("--path", type=Path, required=True)
    validation_register.add_argument(
        "--suite", type=Path, default=VALIDATION_SUITE_PATH
    )
    validation_check = commands.add_parser("verify-validation")
    validation_check.add_argument(
        "--entry", action="append", type=parse_validation_entry, required=True
    )
    validation_check.add_argument("--suite", type=Path, default=VALIDATION_SUITE_PATH)
    args = parser.parse_args()

    if args.command == "download-public":
        selected = set(args.only)
        models = [
            model for model in PUBLIC_MODELS if not selected or model.slug in selected
        ]
        unknown = selected - {model.slug for model in PUBLIC_MODELS}
        if unknown:
            raise SystemExit(f"unknown public model slugs: {sorted(unknown)}")
        download_public(models, args.model_root, args.token_env)
    elif args.command == "register-local":
        register_local(args.slug, args.path, args.source)
    elif args.command == "verify":
        verify(args.entry, args.deep)
    elif args.command == "download-validation":
        registry = load_validation_models(args.suite)
        selected = set(args.only)
        models = [model for model in registry if not selected or model.slug in selected]
        unknown = selected - {model.slug for model in registry}
        if unknown:
            raise SystemExit(f"unknown TRACE validation model slugs: {sorted(unknown)}")
        download_validation(models, args.model_root, args.token_env)
    elif args.command == "register-validation-local":
        register_validation_local(args.slug, args.path, args.suite)
    else:
        verify_validation(args.entry, args.suite)


if __name__ == "__main__":
    main()
