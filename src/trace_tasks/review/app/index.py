"""Lightweight index for locally materialized Trace review artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from PIL import Image

from trace_tasks.core.canonical import canonical_json_bytes
from trace_tasks.core.taxonomy import ACTIVE_DOMAINS, TASK_TAXONOMY
from trace_tasks.review.models import (
    ARTIFACT_SCHEMA_VERSION,
    MATERIALIZATION_SCHEMA_VERSION,
    RECIPE_ID,
    REQUESTS_PER_TASK,
    ArtifactHashes,
    ReviewRequest,
    SourceProvenance,
)
from trace_tasks.review.provenance import sha256_bytes, sha256_file
from trace_tasks.review.recipe import _raw_pixel_hash


@dataclass(frozen=True)
class ReviewSample:
    """One materialized sample reference.

    Large JSON sidecars are intentionally loaded only when a sample is opened.
    """

    uid: str
    domain: str
    scene_id: str
    task_id: str
    query_id: str
    ordinal: int
    data_rel_path: str
    image_rel_path: str
    media_id: str
    image_exists: bool
    recipe_id: str
    recipe_digest: str
    semantic_hash: str
    data_sha256: str
    png_hash: str


@dataclass
class ReviewTask:
    """One public task and its local materialization status."""

    domain: str
    scene_id: str
    task_id: str
    manifest_rel_path: str = ""
    materialized: bool = False
    samples: list[str] = field(default_factory=list)
    query_counts: dict[str, int] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    distribution: dict[str, Any] = field(default_factory=dict)
    recipe_id: str = ""
    recipe_digest: str = ""
    source_provenance: dict[str, Any] = field(default_factory=dict)
    recipe_provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def materialize_command(self) -> str:
        return (
            "trace-review materialize "
            "--recipe docs/review/recipes/trace-review-recipe-v1 "
            f"--task {self.task_id} --output review/task-reviews"
        )


@dataclass
class ReviewScene:
    domain: str
    scene_id: str
    tasks: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.domain}/{self.scene_id}"


@dataclass
class ReviewDomain:
    domain: str
    scenes: list[str] = field(default_factory=list)


@dataclass
class ReviewIndex:
    root: Path
    built_at: str
    domains: dict[str, ReviewDomain] = field(default_factory=dict)
    scenes: dict[str, ReviewScene] = field(default_factory=dict)
    tasks: dict[str, ReviewTask] = field(default_factory=dict)
    samples: dict[str, ReviewSample] = field(default_factory=dict)
    media: dict[str, Path] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def materialized_task_count(self) -> int:
        return sum(1 for task in self.tasks.values() if task.materialized)

    @property
    def sample_count(self) -> int:
        return len(self.samples)


_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_RECIPE_PROVENANCE_FIELDS = frozenset(
    {
        "producer_repository",
        "producer_revision",
        "source_dirty",
        "source_tree_hash",
        "generator_tree_hash",
        "constraints_hash",
        "resource_tree_hash",
        "prompt_bundle_tree_hash",
        "task_catalog_hash",
    }
)
_REVIEW_DUPLICATE_FIELDS = (
    "prompt",
    "prompt_variants",
    "answer_gt",
    "annotation_gt",
    "reward_contract",
    "taxonomy",
    "trace_payload",
    "versions",
)


def build_review_index(review_root: Path | str) -> ReviewIndex:
    """Index public taxonomy and any artifacts materialized under ``review_root``."""

    root = Path(review_root).expanduser().resolve()
    index = ReviewIndex(
        root=root,
        built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    for domain in ACTIVE_DOMAINS:
        index.domains[domain] = ReviewDomain(domain=domain)

    for task_id, taxonomy in sorted(TASK_TAXONOMY.items()):
        domain = str(taxonomy.domain)
        scene_id = str(taxonomy.scene_id)
        scene_key = f"{domain}/{scene_id}"
        scene = index.scenes.setdefault(
            scene_key,
            ReviewScene(domain=domain, scene_id=scene_id),
        )
        if (
            scene_id
            not in index.domains.setdefault(domain, ReviewDomain(domain)).scenes
        ):
            index.domains[domain].scenes.append(scene_id)
        scene.tasks.append(task_id)
        index.tasks[task_id] = ReviewTask(
            domain=domain,
            scene_id=scene_id,
            task_id=task_id,
        )

    if not root.exists():
        return index

    for task in index.tasks.values():
        _index_task_materialization(index, task)

    for domain in index.domains.values():
        domain.scenes.sort()
    for scene in index.scenes.values():
        scene.tasks.sort()
    return index


def _index_task_materialization(index: ReviewIndex, task: ReviewTask) -> None:
    """Accept one task only when its manifest and every declared row agree."""

    root = index.root
    task_dir = root / task.domain / task.scene_id / task.task_id
    if not task_dir.is_dir():
        return
    manifest_path = task_dir / "manifest.json"
    if not manifest_path.exists():
        return
    prefix = f"{task.task_id}: invalid materialization"
    try:
        if not _contained_regular_file(root, manifest_path):
            raise ValueError("manifest is not a contained regular file")
        manifest = _load_json_object(manifest_path)
        recipe_id, recipe_digest, source, recipe_provenance = _manifest_identity(
            manifest,
            task=task,
        )
        request_count = manifest.get("request_count")
        if request_count != REQUESTS_PER_TASK:
            raise ValueError(f"manifest request_count must equal {REQUESTS_PER_TASK}")
        entries = manifest.get("entries")
        if not isinstance(entries, list) or len(entries) != REQUESTS_PER_TASK:
            raise ValueError(
                f"manifest must contain exactly {REQUESTS_PER_TASK} entries"
            )
        query_ids = manifest.get("query_ids")
        if (
            not isinstance(query_ids, list)
            or any(not isinstance(value, str) or not value for value in query_ids)
            or query_ids != sorted(set(query_ids))
        ):
            raise ValueError("manifest query_ids must be sorted unique strings")

        pending_samples: list[ReviewSample] = []
        pending_media: dict[str, Path] = {}
        observed_ordinals: list[int] = []
        observed_queries: set[str] = set()
        for entry in entries:
            sample, media_path = _validate_manifest_entry(
                root,
                task,
                entry,
                recipe_id=recipe_id,
                recipe_digest=recipe_digest,
                recipe_provenance=recipe_provenance,
            )
            pending_samples.append(sample)
            pending_media[sample.media_id] = media_path
            observed_ordinals.append(sample.ordinal)
            observed_queries.add(sample.query_id)
        if observed_ordinals != list(range(REQUESTS_PER_TASK)):
            raise ValueError(
                f"manifest entry ordinals must be exactly 0..{REQUESTS_PER_TASK - 1}"
            )
        if sorted(observed_queries) != query_ids:
            raise ValueError("manifest query_ids do not match its entries")
        if len({sample.uid for sample in pending_samples}) != len(pending_samples):
            raise ValueError("manifest entries do not have unique sample identities")

        distribution = _load_distribution(task_dir, root, manifest)
    except (OSError, ValueError) as exc:
        index.errors.append(f"{prefix}: {_portable_exception(exc)}")
        return

    task.manifest_rel_path = _relative(manifest_path, root)
    task.manifest = manifest
    task.distribution = distribution
    task.recipe_id = recipe_id
    task.recipe_digest = recipe_digest
    task.source_provenance = source
    task.recipe_provenance = recipe_provenance
    for sample in pending_samples:
        index.samples[sample.uid] = sample
        task.samples.append(sample.uid)
        task.query_counts[sample.query_id] = (
            task.query_counts.get(sample.query_id, 0) + 1
        )
    index.media.update(pending_media)
    task.samples.sort(
        key=lambda uid: (index.samples[uid].query_id, index.samples[uid].ordinal)
    )
    task.materialized = True


def _manifest_identity(
    manifest: Mapping[str, Any], *, task: ReviewTask
) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    if manifest.get("schema_version") != MATERIALIZATION_SCHEMA_VERSION:
        raise ValueError("unsupported manifest schema_version")
    recipe_id = manifest.get("recipe_id")
    if recipe_id != RECIPE_ID:
        raise ValueError("unsupported manifest recipe_id")
    recipe_digest = _sha256_digest(
        manifest.get("recipe_digest"), label="manifest recipe_digest"
    )
    expected_taxonomy = {
        "task_id": task.task_id,
        "domain": task.domain,
        "scene_id": task.scene_id,
    }
    for key, expected in expected_taxonomy.items():
        if manifest.get(key) != expected:
            raise ValueError(f"manifest {key} does not match the public taxonomy")

    raw_source = manifest.get("source_provenance")
    if not isinstance(raw_source, Mapping):
        raise ValueError("manifest source_provenance must be an object")
    source = SourceProvenance.from_dict(raw_source).to_dict()
    if dict(raw_source) != source:
        raise ValueError("manifest source_provenance has unsupported fields")
    recipe_provenance = _validate_recipe_provenance(manifest.get("recipe_provenance"))
    expected_source = {
        "producer_repository": source["repository"],
        "producer_revision": source["revision"],
        "source_dirty": source["dirty"],
        "source_tree_hash": source["source_tree_hash"],
        "generator_tree_hash": source["generator_tree_hash"],
        "constraints_hash": source["constraints_hash"],
    }
    for key, expected in expected_source.items():
        if recipe_provenance[key] != expected:
            raise ValueError("manifest source and compact recipe provenance disagree")
    return str(recipe_id), recipe_digest, source, recipe_provenance


def _validate_recipe_provenance(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _RECIPE_PROVENANCE_FIELDS:
        raise ValueError("recipe_provenance has an unsupported shape")
    result = dict(value)
    for field in ("producer_repository", "producer_revision"):
        if not isinstance(result[field], str) or not result[field].strip():
            raise ValueError(f"recipe_provenance {field} must be a non-empty string")
    if not isinstance(result["source_dirty"], bool):
        raise ValueError("recipe_provenance source_dirty must be a boolean")
    for field in _RECIPE_PROVENANCE_FIELDS - {
        "producer_repository",
        "producer_revision",
        "source_dirty",
    }:
        _sha256_digest(result[field], label=f"recipe_provenance {field}")
    return result


def _validate_manifest_entry(
    root: Path,
    task: ReviewTask,
    raw_entry: Any,
    *,
    recipe_id: str,
    recipe_digest: str,
    recipe_provenance: Mapping[str, Any],
) -> tuple[ReviewSample, Path]:
    if not isinstance(raw_entry, Mapping):
        raise ValueError("manifest entry must be an object")
    entry = dict(raw_entry)
    ordinal = entry.get("ordinal")
    if (
        isinstance(ordinal, bool)
        or not isinstance(ordinal, int)
        or ordinal < 0
        or ordinal >= REQUESTS_PER_TASK
    ):
        raise ValueError(
            "manifest entry ordinal must be an integer in "
            f"[0, {REQUESTS_PER_TASK - 1}]"
        )
    query_id = entry.get("query_id")
    if not isinstance(query_id, str) or not query_id:
        raise ValueError("manifest entry query_id must be a non-empty string")
    prefix = f"{task.domain}/{task.scene_id}/{task.task_id}"
    expected_data = f"{prefix}/data/{query_id}/{ordinal:04d}.json"
    expected_image = f"{prefix}/images/{query_id}/{ordinal:04d}.png"
    if (
        entry.get("data_path") != expected_data
        or entry.get("image_path") != expected_image
    ):
        raise ValueError(f"manifest entry {ordinal} uses a noncanonical path")
    data_path = root / expected_data
    image_path = root / expected_image
    if not _contained_regular_file(root, data_path):
        raise ValueError(f"entry {ordinal} data is not a contained regular file")
    if not _contained_regular_file(root, image_path):
        raise ValueError(f"entry {ordinal} image is not a contained regular file")

    data_sha256 = _sha256_digest(
        entry.get("data_sha256"), label=f"entry {ordinal} data_sha256"
    )
    if sha256_file(data_path) != data_sha256:
        raise ValueError(f"entry {ordinal} data_sha256 does not match the file")
    expected = _artifact_hashes(
        entry.get("expected_hashes"), label=f"entry {ordinal} expected_hashes"
    )
    observed = _artifact_hashes(
        entry.get("observed_hashes"), label=f"entry {ordinal} observed_hashes"
    )
    if expected.semantic_hash != observed.semantic_hash:
        raise ValueError(f"entry {ordinal} semantic hashes disagree")
    if sha256_file(image_path) != observed.png_hash:
        raise ValueError(f"entry {ordinal} PNG hash does not match the file")

    artifact = _load_json_object(data_path)
    expected_header = {
        "schema": ARTIFACT_SCHEMA_VERSION,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "recipe_id": recipe_id,
        "recipe_digest": recipe_digest,
        "recipe_provenance": dict(recipe_provenance),
        "task_id": task.task_id,
        "domain": task.domain,
        "scene_id": task.scene_id,
        "query_id": query_id,
    }
    for key, value in expected_header.items():
        if artifact.get(key) != value:
            raise ValueError(f"artifact {ordinal} {key} does not match its manifest")
    request_raw = artifact.get("request")
    if not isinstance(request_raw, Mapping):
        raise ValueError(f"artifact {ordinal} request must be an object")
    request = ReviewRequest.from_dict(request_raw)
    if request.to_dict() != dict(request_raw):
        raise ValueError(f"artifact {ordinal} request has unsupported fields")
    if (
        request.task_id != task.task_id
        or request.domain != task.domain
        or request.scene_id != task.scene_id
        or request.query_id != query_id
        or request.ordinal != ordinal
        or artifact.get("instance_seed") != request.seed
    ):
        raise ValueError(f"artifact {ordinal} request identity is inconsistent")
    if request.hashes != expected:
        raise ValueError(f"artifact {ordinal} request hashes differ from its manifest")
    artifact_expected = _artifact_hashes(
        artifact.get("expected_hashes"), label=f"artifact {ordinal} expected_hashes"
    )
    artifact_observed = _artifact_hashes(
        artifact.get("observed_hashes"), label=f"artifact {ordinal} observed_hashes"
    )
    if artifact_expected != expected or artifact_observed != observed:
        raise ValueError(f"artifact {ordinal} hashes differ from its manifest")

    semantic = artifact.get("semantic_payload")
    if not isinstance(semantic, Mapping):
        raise ValueError(f"artifact {ordinal} semantic_payload must be an object")
    semantic_hash = _semantic_hash(semantic)
    if semantic_hash != expected.semantic_hash:
        raise ValueError(f"artifact {ordinal} semantic payload hash is invalid")
    _validate_review_duplicates(artifact, semantic)
    expected_taxonomy = {
        "domain": task.domain,
        "scene_id": task.scene_id,
        "task_id": task.task_id,
        "query_id": query_id,
    }
    if semantic.get("taxonomy") != expected_taxonomy:
        raise ValueError(f"artifact {ordinal} semantic taxonomy is invalid")
    if (
        semantic.get("scene_id") != task.scene_id
        or semantic.get("query_id") != query_id
    ):
        raise ValueError(f"artifact {ordinal} semantic header is invalid")
    image = artifact.get("image")
    if not isinstance(image, Mapping):
        raise ValueError(f"artifact {ordinal} image metadata must be an object")
    if image.get("path") != expected_image or image.get("format") != "png":
        raise ValueError(f"artifact {ordinal} image metadata path is invalid")
    if not isinstance(image.get("mode"), str) or not image.get("mode"):
        raise ValueError(f"artifact {ordinal} image mode is invalid")
    for field in ("width", "height"):
        value = image.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"artifact {ordinal} image {field} is invalid")
    try:
        with Image.open(image_path) as decoded:
            if decoded.format != "PNG":
                raise ValueError("image bytes are not PNG encoded")
            decoded.load()
            actual_image = {
                "mode": str(decoded.mode),
                "width": int(decoded.width),
                "height": int(decoded.height),
            }
            actual_raw_hash = _raw_pixel_hash(decoded)
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError(f"artifact {ordinal} image is not a valid PNG") from exc
    for field, actual in actual_image.items():
        if image.get(field) != actual:
            raise ValueError(
                f"artifact {ordinal} image {field} does not match decoded PNG"
            )
    if actual_raw_hash != observed.raw_pixel_hash:
        raise ValueError(
            f"artifact {ordinal} raw pixel hash does not match decoded PNG"
        )
    warnings = artifact.get("rendering_warnings")
    if not isinstance(warnings, list) or any(
        not isinstance(item, Mapping) for item in warnings
    ):
        raise ValueError(f"artifact {ordinal} rendering_warnings must be object rows")

    uid = _stable_id(
        "sample",
        "\0".join((recipe_digest, semantic_hash, task.task_id, query_id, str(ordinal))),
    )
    media_id = _stable_id(
        "media", "\0".join((recipe_digest, observed.png_hash, expected_image))
    )
    return (
        ReviewSample(
            uid=uid,
            domain=task.domain,
            scene_id=task.scene_id,
            task_id=task.task_id,
            query_id=query_id,
            ordinal=ordinal,
            data_rel_path=expected_data,
            image_rel_path=expected_image,
            media_id=media_id,
            image_exists=True,
            recipe_id=recipe_id,
            recipe_digest=recipe_digest,
            semantic_hash=semantic_hash,
            data_sha256=data_sha256,
            png_hash=observed.png_hash,
        ),
        image_path.resolve(),
    )


def _load_distribution(
    task_dir: Path, root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    for name in ("distribution.json", "answer_distribution.json"):
        path = task_dir / name
        if not path.exists():
            continue
        if not _contained_regular_file(root, path):
            raise ValueError(f"{name} is not a contained regular file")
        return _load_json_object(path)
    value = manifest.get("distribution_summary", manifest.get("distribution"))
    return dict(value) if isinstance(value, Mapping) else {}


def _validate_review_duplicates(
    artifact: Mapping[str, Any], semantic: Mapping[str, Any]
) -> None:
    for field in _REVIEW_DUPLICATE_FIELDS:
        if artifact.get(field) != semantic.get(field):
            raise ValueError(f"top-level {field} is not bound to semantic_payload")


def _semantic_hash(semantic: Mapping[str, Any]) -> str:
    try:
        return sha256_bytes(canonical_json_bytes(dict(semantic)))
    except Exception as exc:
        raise ValueError("semantic_payload cannot be canonically hashed") from exc


def _artifact_hashes(value: Any, *, label: str) -> ArtifactHashes:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    hashes = ArtifactHashes.from_dict(value)
    if hashes.to_dict() != dict(value):
        raise ValueError(f"{label} has unsupported fields")
    return hashes


def _sha256_digest(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a prefixed SHA-256 digest")
    return value


def _contained_regular_file(root: Path, path: Path) -> bool:
    return (
        path.is_file()
        and is_path_within(root, path)
        and not _has_symlink_component(root, path)
    )


def _has_symlink_component(root: Path, path: Path) -> bool:
    try:
        relative = path.absolute().relative_to(root.absolute())
    except ValueError:
        return True
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _portable_exception(exc: BaseException) -> str:
    if isinstance(exc, OSError):
        return "I/O error while reading local review artifacts"
    text = str(exc).strip()
    if not text:
        return type(exc).__name__
    # Exceptions from JSON and schema helpers should already be path-free. Keep
    # that property even if a platform exception embeds an absolute filename.
    if "/" in text or "\\" in text:
        return "invalid review artifact structure"
    return text


def load_sample_payload(index: ReviewIndex, sample: ReviewSample) -> dict[str, Any]:
    """Load one indexed sample while enforcing root containment."""

    path = resolve_contained_path(index.root, sample.data_rel_path, must_exist=True)
    if _has_symlink_component(index.root, path):
        raise ValueError("sample path contains a symlink")
    if sha256_file(path) != sample.data_sha256:
        raise ValueError("sample data changed after indexing")
    payload = _load_json_object(path)
    if (
        payload.get("recipe_id") != sample.recipe_id
        or payload.get("recipe_digest") != sample.recipe_digest
    ):
        raise ValueError("sample recipe identity changed after indexing")
    semantic = payload.get("semantic_payload")
    if (
        not isinstance(semantic, Mapping)
        or _semantic_hash(semantic) != sample.semantic_hash
    ):
        raise ValueError("sample semantic identity changed after indexing")
    _validate_review_duplicates(payload, semantic)
    return payload


def resolve_contained_path(
    root: Path | str,
    relative_path: Path | str,
    *,
    must_exist: bool = True,
) -> Path:
    """Resolve a relative path and reject traversal and symlink escapes."""

    base = Path(root).expanduser().resolve()
    candidate_value = Path(relative_path)
    if candidate_value.is_absolute():
        raise ValueError("absolute paths are not accepted")
    try:
        candidate = (base / candidate_value).resolve(strict=must_exist)
    except (OSError, RuntimeError) as exc:
        raise ValueError("path cannot be resolved safely") from exc
    if not is_path_within(base, candidate):
        raise ValueError("path escapes the review root")
    return candidate


def is_path_within(root: Path | str, path: Path | str) -> bool:
    """Return whether the fully resolved path remains under the root."""

    try:
        base = Path(root).expanduser().resolve()
        candidate = Path(path).expanduser().resolve()
    except (OSError, RuntimeError):
        return False
    try:
        candidate.relative_to(base)
    except ValueError:
        return False
    return True


def sample_view(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the stable public fields rendered by the sample page."""

    prompt_variants = payload.get("prompt_variants")
    if not isinstance(prompt_variants, Mapping):
        prompt_variants = {}
    answer_gt = payload.get("answer_gt")
    annotation_gt = payload.get("annotation_gt")
    trace_payload = payload.get("trace_payload")
    return {
        "schema": str(payload.get("schema_version", payload.get("schema", ""))),
        "recipe_id": str(payload.get("recipe_id", "")),
        "recipe_digest": str(payload.get("recipe_digest", "")),
        "request": (
            dict(payload.get("request"))
            if isinstance(payload.get("request"), Mapping)
            else {}
        ),
        "instance_seed": payload.get(
            "instance_seed",
            (
                payload.get("request", {}).get("instance_seed")
                if isinstance(payload.get("request"), Mapping)
                else None
            ),
        ),
        "prompt": str(payload.get("prompt", "")),
        "prompt_answer_only": str(prompt_variants.get("answer_only", "")),
        "prompt_answer_and_annotation": str(
            prompt_variants.get("answer_and_annotation", "")
        ),
        "answer_gt": dict(answer_gt) if isinstance(answer_gt, Mapping) else {},
        "annotation_gt": (
            dict(annotation_gt) if isinstance(annotation_gt, Mapping) else {}
        ),
        "trace_payload": (
            dict(trace_payload) if isinstance(trace_payload, Mapping) else {}
        ),
        "versions": (
            dict(payload.get("versions"))
            if isinstance(payload.get("versions"), Mapping)
            else {}
        ),
        "expected_hashes": (
            dict(payload.get("expected_hashes"))
            if isinstance(payload.get("expected_hashes"), Mapping)
            else {}
        ),
        "observed_hashes": (
            dict(payload.get("observed_hashes"))
            if isinstance(payload.get("observed_hashes"), Mapping)
            else {}
        ),
        "rendering_warnings": (
            list(payload.get("rendering_warnings"))
            if isinstance(payload.get("rendering_warnings"), list)
            else []
        ),
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object")
    return payload


def _stable_id(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}\0{value}".encode("utf-8")).hexdigest()[:24]


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


__all__ = [
    "ReviewDomain",
    "ReviewIndex",
    "ReviewSample",
    "ReviewScene",
    "ReviewTask",
    "build_review_index",
    "is_path_within",
    "load_sample_payload",
    "resolve_contained_path",
    "sample_view",
]
