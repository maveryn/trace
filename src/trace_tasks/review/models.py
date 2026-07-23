"""Versioned data contracts for deterministic contributor review artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping, Sequence

RECIPE_SCHEMA_VERSION = "trace-review-recipe-v1"
RECIPE_ID = RECIPE_SCHEMA_VERSION
MATERIALIZATION_SCHEMA_VERSION = "trace-review-materialization-v1"
ARTIFACT_SCHEMA_VERSION = "trace-review-artifact-v1"
REQUESTS_PER_TASK = 25

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_TOKEN_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


class RecipeValidationError(ValueError):
    """Raised when a recipe or materialization violates its public contract."""


class RecipeCaptureError(RuntimeError):
    """Raised when a canonical request cannot be captured."""


class NonDeterministicGenerationError(RecipeCaptureError):
    """Raised when the same request does not produce the same output twice."""


class MaterializationError(RuntimeError):
    """Raised when recipe replay cannot be published safely."""


def _object(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RecipeValidationError(f"{field_name} must be an object")
    return {str(key): item for key, item in value.items()}


def _string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecipeValidationError(f"{field_name} must be a non-empty string")
    return value.strip()


def _integer(value: Any, *, field_name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RecipeValidationError(
            f"{field_name} must be an integer greater than or equal to {minimum}"
        )
    return int(value)


def _safe_token(value: Any, *, field_name: str) -> str:
    text = _string(value, field_name=field_name)
    if _SAFE_TOKEN_RE.fullmatch(text) is None:
        raise RecipeValidationError(
            f"{field_name} must contain only lowercase letters, digits, '.', '_', or '-'"
        )
    return text


def _sha256(value: Any, *, field_name: str) -> str:
    text = _string(value, field_name=field_name)
    if _SHA256_RE.fullmatch(text) is None:
        raise RecipeValidationError(f"{field_name} must be a prefixed SHA-256 digest")
    return text


@dataclass(frozen=True)
class ArtifactHashes:
    """Semantic and rendering identities for one successful request."""

    semantic_hash: str
    raw_pixel_hash: str
    png_hash: str

    def __post_init__(self) -> None:
        _sha256(self.semantic_hash, field_name="hashes.semantic_hash")
        _sha256(self.raw_pixel_hash, field_name="hashes.raw_pixel_hash")
        _sha256(self.png_hash, field_name="hashes.png_hash")

    def to_dict(self) -> dict[str, str]:
        return {
            "semantic_hash": self.semantic_hash,
            "raw_pixel_hash": self.raw_pixel_hash,
            "png_hash": self.png_hash,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ArtifactHashes":
        value = _object(raw, field_name="hashes")
        return cls(
            semantic_hash=_sha256(
                value.get("semantic_hash"), field_name="hashes.semantic_hash"
            ),
            raw_pixel_hash=_sha256(
                value.get("raw_pixel_hash"), field_name="hashes.raw_pixel_hash"
            ),
            png_hash=_sha256(value.get("png_hash"), field_name="hashes.png_hash"),
        )


@dataclass(frozen=True)
class ReviewRequest:
    """One fully specified, successfully double-generated review request."""

    task_id: str
    domain: str
    scene_id: str
    query_id: str
    ordinal: int
    seed: int
    sample_cursor: int
    params: Mapping[str, Any]
    max_attempts: int
    retry: int
    hashes: ArtifactHashes

    def __post_init__(self) -> None:
        from trace_tasks.core.source_layout_policy import parse_public_task_id

        task_id = _string(self.task_id, field_name="request.task_id")
        parts = parse_public_task_id(task_id)
        domain = _safe_token(self.domain, field_name="request.domain")
        scene_id = _safe_token(self.scene_id, field_name="request.scene_id")
        _safe_token(self.query_id, field_name="request.query_id")
        if domain != parts.domain or scene_id != parts.scene_id:
            raise RecipeValidationError(
                "request taxonomy must match the canonical public task id"
            )
        ordinal = _integer(self.ordinal, field_name="request.ordinal")
        if ordinal >= REQUESTS_PER_TASK:
            raise RecipeValidationError(
                f"request.ordinal must be less than {REQUESTS_PER_TASK}"
            )
        _integer(self.seed, field_name="request.seed")
        cursor = _integer(self.sample_cursor, field_name="request.sample_cursor")
        if cursor != ordinal:
            raise RecipeValidationError(
                "trace-review-recipe-v1 requires sample_cursor to equal ordinal"
            )
        _integer(self.max_attempts, field_name="request.max_attempts", minimum=1)
        _integer(self.retry, field_name="request.retry")
        params = _object(self.params, field_name="request.params")
        if any(not isinstance(key, str) for key in self.params):
            raise RecipeValidationError("request.params keys must be strings")
        if params.get("_sample_cursor") != cursor:
            raise RecipeValidationError(
                "request.params._sample_cursor must equal request.sample_cursor"
            )
        if params.get("query_id") != self.query_id:
            raise RecipeValidationError(
                "request.params.query_id must equal request.query_id"
            )

    @property
    def task(self) -> str:
        """Compatibility alias for callers that display a task field."""

        return self.task_id

    @property
    def scene(self) -> str:
        """Compatibility alias for callers that display a scene field."""

        return self.scene_id

    @property
    def query(self) -> str:
        """Compatibility alias for callers that display a query field."""

        return self.query_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "scene_id": self.scene_id,
            "query_id": self.query_id,
            "ordinal": int(self.ordinal),
            "seed": int(self.seed),
            "sample_cursor": int(self.sample_cursor),
            "params": dict(self.params),
            "max_attempts": int(self.max_attempts),
            "retry": int(self.retry),
            "hashes": self.hashes.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ReviewRequest":
        value = _object(raw, field_name="request")
        return cls(
            task_id=_string(value.get("task_id"), field_name="request.task_id"),
            domain=_safe_token(value.get("domain"), field_name="request.domain"),
            scene_id=_safe_token(value.get("scene_id"), field_name="request.scene_id"),
            query_id=_safe_token(value.get("query_id"), field_name="request.query_id"),
            ordinal=_integer(value.get("ordinal"), field_name="request.ordinal"),
            seed=_integer(value.get("seed"), field_name="request.seed"),
            sample_cursor=_integer(
                value.get("sample_cursor"), field_name="request.sample_cursor"
            ),
            params=_object(value.get("params"), field_name="request.params"),
            max_attempts=_integer(
                value.get("max_attempts"),
                field_name="request.max_attempts",
                minimum=1,
            ),
            retry=_integer(value.get("retry"), field_name="request.retry"),
            hashes=ArtifactHashes.from_dict(
                _object(value.get("hashes"), field_name="request.hashes")
            ),
        )


@dataclass(frozen=True)
class RuntimeProvenance:
    """Portable runtime facts that can explain host-native rendering drift."""

    python_version: str
    python_implementation: str
    platform: str
    machine: str
    dependencies: Mapping[str, str] = field(default_factory=dict)
    native_libraries: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _string(
            self.python_version,
            field_name="provenance.runtime.python_version",
        )
        _string(
            self.python_implementation,
            field_name="provenance.runtime.python_implementation",
        )
        _string(self.platform, field_name="provenance.runtime.platform")
        _string(self.machine, field_name="provenance.runtime.machine")
        for field_name, values in (
            ("dependencies", self.dependencies),
            ("native_libraries", self.native_libraries),
        ):
            if not isinstance(values, Mapping):
                raise RecipeValidationError(
                    f"provenance.runtime.{field_name} must be an object"
                )
            for key, value in values.items():
                _string(key, field_name=f"provenance.runtime.{field_name} key")
                _string(value, field_name=f"provenance.runtime.{field_name}.{key}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "python_implementation": self.python_implementation,
            "platform": self.platform,
            "machine": self.machine,
            "dependencies": dict(sorted(self.dependencies.items())),
            "native_libraries": dict(sorted(self.native_libraries.items())),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RuntimeProvenance":
        value = _object(raw, field_name="provenance.runtime")
        dependencies = _object(
            value.get("dependencies", {}),
            field_name="provenance.runtime.dependencies",
        )
        native_libraries = _object(
            value.get("native_libraries", {}),
            field_name="provenance.runtime.native_libraries",
        )
        return cls(
            python_version=_string(
                value.get("python_version"),
                field_name="provenance.runtime.python_version",
            ),
            python_implementation=_string(
                value.get("python_implementation"),
                field_name="provenance.runtime.python_implementation",
            ),
            platform=_string(
                value.get("platform"), field_name="provenance.runtime.platform"
            ),
            machine=_string(
                value.get("machine"), field_name="provenance.runtime.machine"
            ),
            dependencies={str(key): str(item) for key, item in dependencies.items()},
            native_libraries={
                str(key): str(item) for key, item in native_libraries.items()
            },
        )


@dataclass(frozen=True)
class SourceProvenance:
    """Frozen public source identities used to capture a recipe."""

    repository: str
    revision: str
    dirty: bool
    source_tree_hash: str
    generator_tree_hash: str
    constraints_hash: str

    def __post_init__(self) -> None:
        _string(self.repository, field_name="provenance.source.repository")
        _string(self.revision, field_name="provenance.source.revision")
        if not isinstance(self.dirty, bool):
            raise RecipeValidationError("provenance.source.dirty must be a boolean")
        _sha256(
            self.source_tree_hash,
            field_name="provenance.source.source_tree_hash",
        )
        _sha256(
            self.generator_tree_hash,
            field_name="provenance.source.generator_tree_hash",
        )
        _sha256(
            self.constraints_hash,
            field_name="provenance.source.constraints_hash",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository": self.repository,
            "revision": self.revision,
            "dirty": bool(self.dirty),
            "source_tree_hash": self.source_tree_hash,
            "generator_tree_hash": self.generator_tree_hash,
            "constraints_hash": self.constraints_hash,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SourceProvenance":
        value = _object(raw, field_name="provenance.source")
        dirty = value.get("dirty")
        if not isinstance(dirty, bool):
            raise RecipeValidationError("provenance.source.dirty must be a boolean")
        return cls(
            repository=_string(
                value.get("repository"), field_name="provenance.source.repository"
            ),
            revision=_string(
                value.get("revision"), field_name="provenance.source.revision"
            ),
            dirty=dirty,
            source_tree_hash=_sha256(
                value.get("source_tree_hash"),
                field_name="provenance.source.source_tree_hash",
            ),
            generator_tree_hash=_sha256(
                value.get("generator_tree_hash"),
                field_name="provenance.source.generator_tree_hash",
            ),
            constraints_hash=_sha256(
                value.get("constraints_hash"),
                field_name="provenance.source.constraints_hash",
            ),
        )


@dataclass(frozen=True)
class ResourceProvenance:
    """Frozen resource and task-catalog identities used for generation."""

    resource_tree_hash: str
    prompt_bundle_tree_hash: str
    task_catalog_hash: str

    def __post_init__(self) -> None:
        _sha256(
            self.resource_tree_hash,
            field_name="provenance.resources.resource_tree_hash",
        )
        _sha256(
            self.prompt_bundle_tree_hash,
            field_name="provenance.resources.prompt_bundle_tree_hash",
        )
        _sha256(
            self.task_catalog_hash,
            field_name="provenance.resources.task_catalog_hash",
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "resource_tree_hash": self.resource_tree_hash,
            "prompt_bundle_tree_hash": self.prompt_bundle_tree_hash,
            "task_catalog_hash": self.task_catalog_hash,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ResourceProvenance":
        value = _object(raw, field_name="provenance.resources")
        return cls(
            resource_tree_hash=_sha256(
                value.get("resource_tree_hash"),
                field_name="provenance.resources.resource_tree_hash",
            ),
            prompt_bundle_tree_hash=_sha256(
                value.get("prompt_bundle_tree_hash"),
                field_name="provenance.resources.prompt_bundle_tree_hash",
            ),
            task_catalog_hash=_sha256(
                value.get("task_catalog_hash"),
                field_name="provenance.resources.task_catalog_hash",
            ),
        )


@dataclass(frozen=True)
class ReviewProvenance:
    """Complete source, resource, and runtime provenance for one recipe."""

    source: SourceProvenance
    resources: ResourceProvenance
    runtime: RuntimeProvenance

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "resources": self.resources.to_dict(),
            "runtime": self.runtime.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ReviewProvenance":
        value = _object(raw, field_name="provenance")
        return cls(
            source=SourceProvenance.from_dict(
                _object(value.get("source"), field_name="provenance.source")
            ),
            resources=ResourceProvenance.from_dict(
                _object(value.get("resources"), field_name="provenance.resources")
            ),
            runtime=RuntimeProvenance.from_dict(
                _object(value.get("runtime"), field_name="provenance.runtime")
            ),
        )


@dataclass(frozen=True)
class RecipeShard:
    """One domain-sharded JSONL file declared by a recipe manifest."""

    domain: str
    path: str
    request_count: int
    sha256: str

    def __post_init__(self) -> None:
        domain = _safe_token(self.domain, field_name="shard.domain")
        if self.path != f"requests/{domain}.jsonl":
            raise RecipeValidationError(
                "shard.path must be the canonical requests/<domain>.jsonl path"
            )
        _integer(self.request_count, field_name="shard.request_count", minimum=1)
        _sha256(self.sha256, field_name="shard.sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "path": self.path,
            "request_count": int(self.request_count),
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RecipeShard":
        value = _object(raw, field_name="shard")
        return cls(
            domain=_safe_token(value.get("domain"), field_name="shard.domain"),
            path=_string(value.get("path"), field_name="shard.path"),
            request_count=_integer(
                value.get("request_count"),
                field_name="shard.request_count",
                minimum=1,
            ),
            sha256=_sha256(value.get("sha256"), field_name="shard.sha256"),
        )


@dataclass(frozen=True)
class RecipeManifest:
    """Top-level manifest for a canonical public task-review recipe."""

    provenance: ReviewProvenance
    shards: Sequence[RecipeShard]
    task_count: int
    request_count: int
    recipe_digest: str
    schema_version: str = RECIPE_SCHEMA_VERSION
    recipe_id: str = RECIPE_ID
    requests_per_task: int = REQUESTS_PER_TASK

    def __post_init__(self) -> None:
        if self.schema_version != RECIPE_SCHEMA_VERSION:
            raise RecipeValidationError(
                f"unsupported recipe schema version: {self.schema_version}"
            )
        if self.recipe_id != RECIPE_ID:
            raise RecipeValidationError(f"unsupported recipe id: {self.recipe_id}")
        if self.requests_per_task != REQUESTS_PER_TASK:
            raise RecipeValidationError(
                f"{RECIPE_ID} requires exactly {REQUESTS_PER_TASK} requests per task"
            )
        task_count = _integer(
            self.task_count, field_name="manifest.task_count", minimum=1
        )
        request_count = _integer(
            self.request_count, field_name="manifest.request_count", minimum=1
        )
        if request_count != task_count * REQUESTS_PER_TASK:
            raise RecipeValidationError(
                "manifest.request_count must equal task_count * requests_per_task"
            )
        if not self.shards:
            raise RecipeValidationError("manifest.shards must not be empty")
        if sum(int(shard.request_count) for shard in self.shards) != request_count:
            raise RecipeValidationError(
                "manifest shard counts must sum to manifest.request_count"
            )
        domains = [shard.domain for shard in self.shards]
        if domains != sorted(set(domains)):
            raise RecipeValidationError(
                "manifest shards must have unique domains in sorted order"
            )
        _sha256(self.recipe_digest, field_name="manifest.recipe_digest")

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "recipe_id": self.recipe_id,
            "requests_per_task": int(self.requests_per_task),
            "task_count": int(self.task_count),
            "request_count": int(self.request_count),
            "provenance": self.provenance.to_dict(),
            "shards": [shard.to_dict() for shard in self.shards],
        }
        if include_digest:
            value["recipe_digest"] = self.recipe_digest
        return value

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RecipeManifest":
        value = _object(raw, field_name="manifest")
        raw_shards = value.get("shards")
        if not isinstance(raw_shards, Sequence) or isinstance(raw_shards, (str, bytes)):
            raise RecipeValidationError("manifest.shards must be an array")
        return cls(
            schema_version=_string(
                value.get("schema_version"), field_name="manifest.schema_version"
            ),
            recipe_id=_string(value.get("recipe_id"), field_name="manifest.recipe_id"),
            requests_per_task=_integer(
                value.get("requests_per_task"),
                field_name="manifest.requests_per_task",
                minimum=1,
            ),
            task_count=_integer(
                value.get("task_count"), field_name="manifest.task_count", minimum=1
            ),
            request_count=_integer(
                value.get("request_count"),
                field_name="manifest.request_count",
                minimum=1,
            ),
            provenance=ReviewProvenance.from_dict(
                _object(value.get("provenance"), field_name="manifest.provenance")
            ),
            shards=tuple(
                RecipeShard.from_dict(_object(item, field_name="manifest.shards[]"))
                for item in raw_shards
            ),
            recipe_digest=_sha256(
                value.get("recipe_digest"), field_name="manifest.recipe_digest"
            ),
        )


@dataclass(frozen=True)
class VerificationIssue:
    """One semantic error or rendering warning found during replay."""

    severity: str
    code: str
    message: str
    task_id: str = ""
    ordinal: int | None = None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }
        if self.task_id:
            value["task_id"] = self.task_id
        if self.ordinal is not None:
            value["ordinal"] = int(self.ordinal)
        return value


@dataclass(frozen=True)
class VerificationReport:
    """Structured result of regenerating or checking materialized artifacts."""

    checked_requests: int
    issues: Sequence[VerificationIssue] = field(default_factory=tuple)

    @property
    def errors(self) -> tuple[VerificationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[VerificationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checked_requests": int(self.checked_requests),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class MaterializationReport:
    """Summary of an atomic recipe materialization run."""

    selected_requests: int
    tasks_materialized: Sequence[str] = field(default_factory=tuple)
    tasks_resumed: Sequence[str] = field(default_factory=tuple)
    warnings: Sequence[VerificationIssue] = field(default_factory=tuple)

    @property
    def task_count(self) -> int:
        return len(self.tasks_materialized) + len(self.tasks_resumed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_requests": int(self.selected_requests),
            "task_count": int(self.task_count),
            "tasks_materialized": list(self.tasks_materialized),
            "tasks_resumed": list(self.tasks_resumed),
            "warning_count": len(self.warnings),
            "warnings": [warning.to_dict() for warning in self.warnings],
        }


@dataclass(frozen=True)
class AuditIssue:
    """One contributor-facing audit finding."""

    severity: str
    category: str
    code: str
    message: str
    task_id: str = ""
    path: str = ""

    def to_dict(self) -> dict[str, str]:
        value = {
            "severity": self.severity,
            "category": self.category,
            "code": self.code,
            "message": self.message,
        }
        if self.task_id:
            value["task_id"] = self.task_id
        if self.path:
            value["path"] = self.path
        return value


@dataclass(frozen=True)
class AuditReport:
    """Portable review audit result independent of historical campaigns."""

    checked: int
    issues: Sequence[AuditIssue] = field(default_factory=tuple)

    @property
    def errors(self) -> tuple[AuditIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[AuditIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checked": int(self.checked),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [issue.to_dict() for issue in self.issues],
        }


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "MATERIALIZATION_SCHEMA_VERSION",
    "RECIPE_ID",
    "RECIPE_SCHEMA_VERSION",
    "REQUESTS_PER_TASK",
    "ArtifactHashes",
    "AuditIssue",
    "AuditReport",
    "MaterializationError",
    "MaterializationReport",
    "NonDeterministicGenerationError",
    "RecipeCaptureError",
    "RecipeManifest",
    "RecipeShard",
    "RecipeValidationError",
    "ResourceProvenance",
    "ReviewProvenance",
    "ReviewRequest",
    "RuntimeProvenance",
    "SourceProvenance",
    "VerificationIssue",
    "VerificationReport",
]
