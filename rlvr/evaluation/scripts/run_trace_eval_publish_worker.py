#!/usr/bin/env python3
"""Publish one completed trace_eval_v1 campaign without blocking evaluation."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import stat
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from huggingface_hub import HfApi


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from trace_eval_hf_archive_lib import (  # noqa: E402
    ArchiveDaemon,
    ArchiveError,
    load_descriptor,
    redact_secret,
    verify_expected_slice_coverage,
)
from trace_eval_code_provenance import trace_eval_code_sha256  # noqa: E402
from trace_eval_evaluator_provenance import (  # noqa: E402
    DEFAULT_VLMEVAL_ROOT,
    evaluator_provenance_sha256,
)
from trace_eval_public_export import (  # noqa: E402
    EXPORT_PLAN_VERSION,
    ExportPlan,
    PublicExportError,
    build_private_export_plan,
    export_public_artifacts,
    load_and_verify_public_export,
)
from trace_eval_suite import TraceEvalSuite, load_trace_eval_suite  # noqa: E402
from verify_trace_eval import verify_campaign  # noqa: E402


STATUS_SCHEMA = "trace-eval-publish-worker-status-v1"
STAGES = ("generation", "extraction", "score")
DEFAULT_PAPER_REPO = "maveryn/trace-eval-runs"


class PublishWorkerError(RuntimeError):
    """Raised when publication cannot continue safely."""


def _read_token(environment_variable: str) -> str:
    token = os.environ.get(environment_variable, "").strip()
    if not token:
        raise PublishWorkerError(
            f"{environment_variable} must contain a Hugging Face access token"
        )
    return token


def _validate_destination_repo(repo_id: str) -> None:
    owner, separator, name = repo_id.partition("/")
    if (
        separator != "/"
        or not owner
        or not name
        or any(char.isspace() for char in repo_id)
        or "internal" in name.lower()
        or "private" in name.lower()
    ):
        raise PublishWorkerError(
            "destination must be a neutral owner/repository dataset id"
        )


@dataclass(frozen=True)
class ModelConfig:
    source_model_id: str
    local_model_path: str
    source_revision: str
    model_id: str
    model_revision: str
    display_name: str
    repository_id: str
    repository_revision: str

    def export_mapping(self) -> dict[str, str]:
        return {
            "source_model_id": self.source_model_id,
            "source_revision": self.source_revision,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "display_name": self.display_name,
            "repository_id": self.repository_id,
            "repository_revision": self.repository_revision,
        }


@dataclass(frozen=True)
class WorkerConfig:
    campaign_root: Path
    score_root: Path
    archive_spool_root: Path
    dataset_manifest: Path
    vlmeval_root: Path
    work_root: Path
    lock_root: Path
    source_run_id: str
    public_run_id: str
    seeds: tuple[int, ...]
    models: tuple[ModelConfig, ...]
    judge: Mapping[str, str]
    token_env: str
    paper_repo: str
    revision: str
    poll_seconds: float
    settle_seconds: float
    timeout_seconds: float
    verification_grace_seconds: float
    upload_attempts: int
    retry_base_seconds: float
    retry_cap_seconds: float
    batch_size: int
    allow_upload: bool
    confirmation: str | None

    @property
    def expected_slices(self) -> int:
        return len(STAGES) * 24 * len(self.models) * len(self.seeds)

    @property
    def expected_public_artifacts(self) -> int:
        return self.expected_slices

    @property
    def plan_path(self) -> Path:
        return self.work_root / "private-export-plan.json"

    @property
    def export_root(self) -> Path:
        return self.work_root / "sanitized-export"

    @property
    def status_path(self) -> Path:
        return self.work_root / "status.json"

    @property
    def worker_lock_path(self) -> Path:
        return self.work_root / "worker.lock"

    @property
    def campaign_lock_path(self) -> Path:
        digest = hashlib.sha256(
            (
                f"{self.archive_spool_root}\0{self.source_run_id}"
            ).encode("utf-8")
        ).hexdigest()
        return self.lock_root / f"campaign-{digest}.lock"

    @property
    def repository_lock_path(self) -> Path:
        digest = hashlib.sha256(
            f"{self.paper_repo}@{self.revision}".encode("utf-8")
        ).hexdigest()
        return self.lock_root / f"repository-{digest}.lock"

    def digest_document(self) -> dict[str, Any]:
        return {
            "archive_spool_root": str(self.archive_spool_root),
            "campaign_root": str(self.campaign_root),
            "dataset_manifest": str(self.dataset_manifest),
            "judge": dict(self.judge),
            "models": [
                {
                    **model.export_mapping(),
                    "local_model_path": model.local_model_path,
                }
                for model in self.models
            ],
            "paper_repo": self.paper_repo,
            "public_run_id": self.public_run_id,
            "revision": self.revision,
            "score_root": str(self.score_root),
            "seeds": list(self.seeds),
            "source_run_id": self.source_run_id,
            "suite_id": "trace_eval_v1",
            "vlmeval_root": str(self.vlmeval_root),
        }

    @property
    def config_sha256(self) -> str:
        encoded = json.dumps(
            self.digest_document(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Coverage:
    ready: int
    expected: int
    missing: tuple[tuple[str, str, int, str], ...]
    fingerprint: str
    campaign_config_hash: str | None
    dataset_revision: str | None
    code_hash: str | None
    descriptor_paths: tuple[Path, ...]

    @property
    def complete(self) -> bool:
        return self.ready == self.expected and not self.missing


class HeldLock:
    def __init__(self, path: Path, *, blocking: bool):
        self.path = path
        self.blocking = blocking
        self.descriptor = -1

    def __enter__(self) -> "HeldLock":
        _ensure_private_directory(self.path.parent)
        flags = os.O_CREAT | os.O_RDWR
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except OSError as error:
            raise PublishWorkerError(
                f"cannot open lock {self.path}: {error}"
            ) from error
        try:
            mode = os.fstat(descriptor).st_mode
            if not stat.S_ISREG(mode):
                raise PublishWorkerError(
                    f"lock path is not a regular file: {self.path}"
                )
            os.fchmod(descriptor, 0o600)
            operation = fcntl.LOCK_EX
            if not self.blocking:
                operation |= fcntl.LOCK_NB
            try:
                fcntl.flock(descriptor, operation)
            except BlockingIOError as error:
                raise PublishWorkerError(
                    f"another publication worker holds {self.path}"
                ) from error
        except Exception:
            os.close(descriptor)
            raise
        self.descriptor = descriptor
        return self

    def __exit__(self, *_args: object) -> None:
        if self.descriptor >= 0:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
            os.close(self.descriptor)
            self.descriptor = -1


def _absolute(path: Path) -> Path:
    return path.expanduser().absolute()


def _reject_symlink_components(path: Path) -> None:
    requested = _absolute(path)
    for component in (requested, *requested.parents):
        if component.is_symlink():
            raise PublishWorkerError(f"refusing path through symlink: {component}")


def _ensure_private_directory(path: Path) -> Path:
    requested = _absolute(path)
    _reject_symlink_components(requested)
    requested.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(requested)
    if not requested.is_dir():
        raise PublishWorkerError(f"private state path is not a directory: {requested}")
    os.chmod(requested, 0o700)
    return requested


def _atomic_status(path: Path, document: Mapping[str, Any]) -> None:
    _reject_symlink_components(path)
    parent = _ensure_private_directory(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        payload = (
            json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _utc_now() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).isoformat()


class StatusWriter:
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.started_at = _utc_now()

    def write(self, phase: str, **details: Any) -> None:
        document = {
            "schema_version": STATUS_SCHEMA,
            "config_sha256": self.config.config_sha256,
            "source_run_id": self.config.source_run_id,
            "public_run_id": self.config.public_run_id,
            "suite_id": "trace_eval_v1",
            "models": [model.source_model_id for model in self.config.models],
            "seeds": list(self.config.seeds),
            "expected_slices": self.config.expected_slices,
            "phase": phase,
            "pid": os.getpid(),
            "started_at": self.started_at,
            "updated_at": _utc_now(),
            **details,
        }
        _atomic_status(self.config.status_path, document)
        progress = ""
        if "ready_slices" in details:
            progress = f" ready={details['ready_slices']}/{self.config.expected_slices}"
        print(f"[trace-eval-publish] phase={phase}{progress}", flush=True)


def _expected_identities(
    config: WorkerConfig, suite: TraceEvalSuite
) -> set[tuple[str, str, int, str]]:
    return {
        (stage, model.source_model_id, seed, benchmark)
        for stage in STAGES
        for model in config.models
        for seed in config.seeds
        for benchmark in suite.benchmark_keys
    }


def inspect_ready_coverage(
    config: WorkerConfig, suite: TraceEvalSuite
) -> Coverage:
    expected = _expected_identities(config, suite)
    model_map = {model.source_model_id: model for model in config.models}
    found: dict[tuple[str, str, int, str], list[tuple[Path, Mapping[str, Any]]]] = {}
    campaign_hashes: set[str] = set()
    dataset_revisions: set[str] = set()
    code_hashes: set[str] = set()
    ready_root = config.archive_spool_root / "ready"
    for path in sorted(ready_root.glob("*.ready.json")):
        descriptor = load_descriptor(path)
        identity = descriptor.get("identity") or {}
        if identity.get("run_id") != config.source_run_id:
            continue
        key = (
            str(identity.get("stage") or ""),
            str(identity.get("model_slug") or ""),
            int(identity.get("seed", -1)),
            str(identity.get("benchmark") or ""),
        )
        if key not in expected:
            raise PublishWorkerError(
                "source run contains an unexpected ready slice: "
                f"stage={key[0]} model={key[1]} seed={key[2]} benchmark={key[3]}"
            )
        model = model_map[key[1]]
        if identity.get("model_revision") != model.source_revision:
            raise PublishWorkerError(
                f"source revision mismatch for ready slice {key}: "
                f"{identity.get('model_revision')!r} != {model.source_revision!r}"
            )
        repository_identity = f"{model.repository_id}@{model.repository_revision}"
        if identity.get("model") != repository_identity:
            raise PublishWorkerError(
                f"source repository mismatch for ready slice {key}: "
                f"{identity.get('model')!r} != {repository_identity!r}"
            )
        found.setdefault(key, []).append((path, descriptor))
        provenance = descriptor.get("provenance") or {}
        campaign_hashes.add(str(provenance.get("campaign_config_hash") or ""))
        dataset_revisions.add(str(identity.get("dataset_revision") or ""))
        code_hashes.add(str(provenance.get("final25_code_hash") or ""))

    duplicates = [key for key, values in found.items() if len(values) != 1]
    if duplicates:
        raise PublishWorkerError(
            f"source run contains {len(duplicates)} duplicate ready identities; "
            f"first={duplicates[:3]}"
        )
    for label, values in (
        ("campaign config hash", campaign_hashes),
        ("dataset revision", dataset_revisions),
        ("code hash", code_hashes),
    ):
        if "" in values:
            raise PublishWorkerError(f"ready descriptors contain an empty {label}")
        if len(values) > 1:
            raise PublishWorkerError(f"ready descriptors disagree on {label}")

    missing = tuple(sorted(expected - set(found)))
    selected = tuple(found[key][0][0] for key in sorted(found))
    fingerprint_material = [
        {
            "descriptor_id": found[key][0][1]["descriptor_id"],
            "identity": key,
            "path": found[key][0][0].name,
        }
        for key in sorted(found)
    ]
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_material,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return Coverage(
        ready=len(found),
        expected=len(expected),
        missing=missing,
        fingerprint=fingerprint,
        campaign_config_hash=next(iter(campaign_hashes), None),
        dataset_revision=next(iter(dataset_revisions), None),
        code_hash=next(iter(code_hashes), None),
        descriptor_paths=selected,
    )


def build_new_descriptors(daemon: ArchiveDaemon, paths: Sequence[Path]) -> int:
    selected = {str(path.absolute()) for path in paths}
    known = {
        str(Path(row["descriptor_path"]).absolute()) for row in daemon.ledger.rows()
    }
    built = 0
    for path in paths:
        absolute = str(path.absolute())
        if absolute in known:
            continue
        try:
            descriptor = load_descriptor(path, spool_root=daemon.spool_root)
            descriptor_id = str(descriptor["descriptor_id"])
            daemon.ledger.register(descriptor_id, path.absolute())
        except Exception as error:
            raise PublishWorkerError(
                f"cannot register archive descriptor {path}: {error}"
            ) from error
    for row in daemon.ledger.rows(("ready",)):
        descriptor_id = str(row["descriptor_id"])
        descriptor_path = Path(row["descriptor_path"])
        if str(descriptor_path.absolute()) not in selected:
            continue
        try:
            daemon.build_descriptor(descriptor_path)
            built += 1
        except Exception as error:
            daemon.ledger.mark_error(descriptor_id, str(error), permanent=True)
            raise PublishWorkerError(
                f"cannot build archive descriptor {descriptor_path}: {error}"
            ) from error
    failed = [
        row
        for row in daemon.ledger.rows(("failed",))
        if str(Path(row["descriptor_path"]).absolute()) in selected
    ]
    if failed:
        first = failed[0]
        raise PublishWorkerError(
            f"archive ledger contains {len(failed)} failed slices; "
            f"first={first['descriptor_path']}: {first['last_error']}"
        )
    return built


def _verification_args(
    config: WorkerConfig,
    suite: TraceEvalSuite,
    *,
    phase: str,
    code_hash: str,
    evaluator_hash: str,
) -> argparse.Namespace:
    return argparse.Namespace(
        campaign_root=config.campaign_root,
        score_root=config.score_root,
        phase=phase,
        model_slugs=[],
        model_entry=[
            f"{model.source_model_id}={model.local_model_path}={model.source_revision}"
            for model in config.models
        ],
        seeds=config.seeds,
        dataset_manifest=config.dataset_manifest,
        suite_manifest=suite.path,
        dataset_revision=None,
        dataset_snapshot_sha256=None,
        code_hash=code_hash if phase == "generation" else None,
        evaluator_provenance_sha256=(
            evaluator_hash if phase == "generation" else None
        ),
        vlmeval_root=config.vlmeval_root,
    )


def verify_local_campaign(
    config: WorkerConfig,
    suite: TraceEvalSuite,
    *,
    coverage: Coverage,
) -> dict[str, Any]:
    evaluator_hash = evaluator_provenance_sha256(
        repo_root=REPO_ROOT, vlmeval_root=config.vlmeval_root
    )
    code_hash = trace_eval_code_sha256(
        repo_root=REPO_ROOT,
        vlmeval_root=config.vlmeval_root,
        evaluator_sha256=evaluator_hash,
    )
    if coverage.code_hash != code_hash:
        raise PublishWorkerError(
            "archive code provenance does not match the current evaluation worktree"
        )
    reports: dict[str, Any] = {}
    for phase in ("generation", "score"):
        report = verify_campaign(
            _verification_args(
                config,
                suite,
                phase=phase,
                code_hash=code_hash,
                evaluator_hash=evaluator_hash,
            ),
            suite,
        )
        reports[phase] = report
    return reports


def _verification_summary(reports: Mapping[str, Any]) -> dict[str, Any]:
    return {
        phase: {
            "complete": bool(report.get("complete")),
            "completed_slices": int(report.get("completed_slices", 0)),
            "expected_slices": int(report.get("expected_slices", 0)),
            "incomplete": list(report.get("incomplete") or [])[:10],
        }
        for phase, report in reports.items()
    }


def _all_verification_complete(reports: Mapping[str, Any]) -> bool:
    return all(bool(report.get("complete")) for report in reports.values())


def _sleep_with_timeout(
    seconds: float,
    *,
    started: float,
    timeout_seconds: float,
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
) -> None:
    if timeout_seconds > 0 and monotonic() - started + seconds > timeout_seconds:
        raise PublishWorkerError(
            f"publication worker timed out after {timeout_seconds:.0f} seconds"
        )
    sleep(max(0.0, seconds))


def _prevalidate(config: WorkerConfig, suite: TraceEvalSuite) -> None:
    if not config.allow_upload:
        raise PublishWorkerError(
            "pass --allow-paper-run-upload to authorize publication"
        )
    expected_confirmation = f"UPLOAD {config.paper_repo}/{config.public_run_id}"
    if config.confirmation != expected_confirmation:
        raise PublishWorkerError(
            f"confirmation must be exactly {expected_confirmation!r}"
        )
    _validate_destination_repo(config.paper_repo)
    if not config.revision.strip():
        raise PublishWorkerError("destination revision must be nonempty")
    if not config.seeds or len(set(config.seeds)) != len(config.seeds):
        raise PublishWorkerError("seeds must be nonempty and unique")
    if any(seed < 0 for seed in config.seeds):
        raise PublishWorkerError("seeds must be nonnegative")
    for model in config.models:
        if (
            not Path(model.local_model_path).is_absolute()
            or "=" in model.local_model_path
        ):
            raise PublishWorkerError(
                "local model path must be the exact absolute launcher path: "
                f"{model.local_model_path!r}"
            )
    ExportPlan.from_mapping(
        {
            "schema_version": EXPORT_PLAN_VERSION,
            "source": {
                "run_id": config.source_run_id,
                "selection_sha256": "0" * 64,
                "slice_set_sha256": "0" * 64,
            },
            "public": {
                "run_id": config.public_run_id,
                "suite_id": suite.suite_id,
                "benchmarks": list(suite.benchmark_keys),
                "categories": {
                    name: list(members) for name, members in suite.categories.items()
                },
                "seeds": list(config.seeds),
            },
            "models": [model.export_mapping() for model in config.models],
            "judge": dict(config.judge),
        }
    )
    _read_token(config.token_env)


def _load_prior_status(config: WorkerConfig) -> bool:
    if not config.status_path.exists():
        return False
    _reject_symlink_components(config.status_path)
    try:
        document = json.loads(config.status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PublishWorkerError(f"cannot read prior worker status: {error}") from error
    if document.get("schema_version") != STATUS_SCHEMA:
        raise PublishWorkerError("prior worker status has an unknown schema")
    if document.get("config_sha256") != config.config_sha256:
        raise PublishWorkerError("work root belongs to a different publication config")
    if document.get("phase") != "complete":
        return False
    report = document.get("upload_report")
    if not isinstance(report, Mapping) or not report:
        raise PublishWorkerError("complete worker status has no verified upload report")
    manifest_sha256 = document.get("public_export_manifest_sha256")
    if (
        not isinstance(manifest_sha256, str)
        or len(manifest_sha256) != 64
        or report.get("public_export_manifest_sha256") != manifest_sha256
    ):
        raise PublishWorkerError(
            "complete worker status has no bound export verification digest"
        )
    print("[trace-eval-publish] phase=complete already_verified=true", flush=True)
    return True


def _wait_for_complete_source(
    config: WorkerConfig,
    suite: TraceEvalSuite,
    daemon: ArchiveDaemon,
    status: StatusWriter,
    *,
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
) -> Coverage:
    started = monotonic()
    exact_since: float | None = None
    while True:
        if (
            config.timeout_seconds > 0
            and monotonic() - started > config.timeout_seconds
        ):
            raise PublishWorkerError(
                "publication worker timed out after "
                f"{config.timeout_seconds:.0f} seconds"
            )
        coverage = inspect_ready_coverage(config, suite)
        built = build_new_descriptors(daemon, coverage.descriptor_paths)
        if not coverage.complete:
            exact_since = None
            status.write(
                "waiting",
                ready_slices=coverage.ready,
                missing_slices=len(coverage.missing),
                newly_built_slices=built,
                missing_preview=[list(item) for item in coverage.missing[:10]],
            )
            _sleep_with_timeout(
                config.poll_seconds,
                started=started,
                timeout_seconds=config.timeout_seconds,
                sleep=sleep,
                monotonic=monotonic,
            )
            continue

        verify_expected_slice_coverage(
            config.archive_spool_root,
            run_id=config.source_run_id,
            model_slugs=[model.source_model_id for model in config.models],
            seeds=config.seeds,
            benchmarks=suite.benchmark_keys,
            stages=STAGES,
            campaign_config_hash=coverage.campaign_config_hash,
            dataset_revision=coverage.dataset_revision,
        )
        reports = verify_local_campaign(config, suite, coverage=coverage)
        if not _all_verification_complete(reports):
            if exact_since is None:
                exact_since = monotonic()
            if monotonic() - exact_since > config.verification_grace_seconds:
                raise PublishWorkerError(
                    "campaign verification remained incomplete after exact "
                    "archive coverage: "
                    + json.dumps(_verification_summary(reports), sort_keys=True)
                )
            status.write(
                "verifying_local",
                ready_slices=coverage.ready,
                missing_slices=0,
                verification=_verification_summary(reports),
            )
            _sleep_with_timeout(
                config.poll_seconds,
                started=started,
                timeout_seconds=config.timeout_seconds,
                sleep=sleep,
                monotonic=monotonic,
            )
            continue

        status.write(
            "settling",
            ready_slices=coverage.ready,
            missing_slices=0,
            verification=_verification_summary(reports),
            settle_seconds=config.settle_seconds,
        )
        _sleep_with_timeout(
            config.settle_seconds,
            started=started,
            timeout_seconds=config.timeout_seconds,
            sleep=sleep,
            monotonic=monotonic,
        )
        settled = inspect_ready_coverage(config, suite)
        build_new_descriptors(daemon, settled.descriptor_paths)
        if not settled.complete or settled.fingerprint != coverage.fingerprint:
            exact_since = None
            continue
        verify_expected_slice_coverage(
            config.archive_spool_root,
            run_id=config.source_run_id,
            model_slugs=[model.source_model_id for model in config.models],
            seeds=config.seeds,
            benchmarks=suite.benchmark_keys,
            stages=STAGES,
            campaign_config_hash=settled.campaign_config_hash,
            dataset_revision=settled.dataset_revision,
        )
        settled_reports = verify_local_campaign(config, suite, coverage=settled)
        if not _all_verification_complete(settled_reports):
            exact_since = exact_since or monotonic()
            continue
        return settled


def _upload_with_retries(
    config: WorkerConfig,
    status: StatusWriter,
    *,
    api_factory: Callable[..., Any],
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    token = _read_token(config.token_env)
    try:
        api = api_factory(token=token)
        for attempt in range(1, config.upload_attempts + 1):
            status.write("uploading", upload_attempt=attempt)
            try:
                api.create_repo(
                    repo_id=config.paper_repo,
                    repo_type="dataset",
                    exist_ok=True,
                )
                commit = api.upload_folder(
                    folder_path=str(config.export_root),
                    path_in_repo=f"runs/{config.public_run_id}",
                    repo_id=config.paper_repo,
                    repo_type="dataset",
                    revision=config.revision,
                    commit_message=f"Publish sanitized {config.public_run_id}",
                )
                return {
                    "repo_id": config.paper_repo,
                    "revision": config.revision,
                    "path_in_repo": f"runs/{config.public_run_id}",
                    "commit_oid": str(getattr(commit, "oid", "")),
                }
            except PublicExportError:
                raise
            except Exception as error:
                if attempt == config.upload_attempts:
                    raise
                delay = min(
                    config.retry_cap_seconds,
                    config.retry_base_seconds * (2 ** (attempt - 1)),
                )
                status.write(
                    "upload_retry",
                    upload_attempt=attempt,
                    retry_seconds=delay,
                    error=redact_secret(error, token),
                )
                sleep(delay)
    finally:
        token = ""
    raise AssertionError("unreachable")


def run_worker(
    config: WorkerConfig,
    *,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    api_factory: Callable[..., Any] = HfApi,
) -> int:
    suite = load_trace_eval_suite()
    _prevalidate(config, suite)
    _ensure_private_directory(config.work_root)
    _ensure_private_directory(config.lock_root)
    status = StatusWriter(config)
    token_for_redaction: str | None = None
    with (
        HeldLock(config.worker_lock_path, blocking=False),
        HeldLock(config.campaign_lock_path, blocking=False),
    ):
        if _load_prior_status(config):
            return 0
        try:
            status.write("starting", ready_slices=0)
            daemon = ArchiveDaemon(
                spool_root=config.archive_spool_root,
                api=object(),
                token=None,
            )
            coverage = _wait_for_complete_source(
                config,
                suite,
                daemon,
                status,
                sleep=sleep,
                monotonic=monotonic,
            )
            status.write("planning", ready_slices=coverage.ready)
            build_private_export_plan(
                config.archive_spool_root,
                source_run_id=config.source_run_id,
                public_run_id=config.public_run_id,
                seeds=config.seeds,
                models=[model.export_mapping() for model in config.models],
                judge=config.judge,
                output=config.plan_path,
            )
            status.write("exporting", ready_slices=coverage.ready)
            export_public_artifacts(
                config.archive_spool_root,
                config.plan_path,
                config.export_root,
            )
            status.write("verifying_export", ready_slices=coverage.ready)
            verified = load_and_verify_public_export(
                config.export_root,
                expected_artifacts=config.expected_public_artifacts,
                dynamic_markers=(config.source_run_id,),
            )

            final_coverage = inspect_ready_coverage(config, suite)
            if (
                not final_coverage.complete
                or final_coverage.fingerprint != coverage.fingerprint
            ):
                raise PublishWorkerError(
                    "source descriptor set changed after the export was sealed"
                )
            final_reports = verify_local_campaign(
                config, suite, coverage=final_coverage
            )
            if not _all_verification_complete(final_reports):
                raise PublishWorkerError(
                    "campaign verification changed after the export was sealed"
                )
            build_private_export_plan(
                config.archive_spool_root,
                source_run_id=config.source_run_id,
                public_run_id=config.public_run_id,
                seeds=config.seeds,
                models=[model.export_mapping() for model in config.models],
                judge=config.judge,
                output=config.plan_path,
            )
            token_for_redaction = _read_token(config.token_env)
            status.write("waiting_upload_lock", ready_slices=coverage.ready)
            with HeldLock(config.repository_lock_path, blocking=True):
                locked_coverage = inspect_ready_coverage(config, suite)
                if (
                    not locked_coverage.complete
                    or locked_coverage.fingerprint != coverage.fingerprint
                ):
                    raise PublishWorkerError(
                        "source descriptor set changed while waiting for the "
                        "upload lock"
                    )
                upload_report = _upload_with_retries(
                    config, status, api_factory=api_factory, sleep=sleep
                )
            status.write(
                "complete",
                ready_slices=coverage.ready,
                public_export_manifest_sha256=verified.manifest_sha256,
                upload_report=upload_report,
                completed_at=_utc_now(),
            )
            return 0
        except Exception as error:
            safe_error = redact_secret(error, token_for_redaction)
            status.write(
                "error",
                error_type=type(error).__name__,
                error=safe_error,
                failed_at=_utc_now(),
            )
            print(
                f"[trace-eval-publish:error] {type(error).__name__}: {safe_error}",
                file=sys.stderr,
                flush=True,
            )
            return 2
        finally:
            token_for_redaction = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path)
    parser.add_argument("--archive-spool-root", type=Path)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--vlmeval-root", type=Path, default=DEFAULT_VLMEVAL_ROOT)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument(
        "--lock-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / ".work" / "publisher-locks",
    )
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--public-run-id", required=True)
    parser.add_argument("--seed", action="append", type=int, required=True)
    parser.add_argument(
        "--model",
        action="append",
        nargs=8,
        required=True,
        metavar=(
            "SOURCE_ID",
            "LOCAL_PATH",
            "SOURCE_REV",
            "MODEL_ID",
            "MODEL_REV",
            "DISPLAY_NAME",
            "REPOSITORY_ID",
            "REPOSITORY_REV",
        ),
    )
    parser.add_argument(
        "--judge",
        nargs=3,
        required=True,
        metavar=("SOURCE_ID", "REPOSITORY_ID", "REPOSITORY_REV"),
    )
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable containing the Hugging Face token.",
    )
    parser.add_argument("--paper-repo", default=DEFAULT_PAPER_REPO)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--settle-seconds", type=float, default=10.0)
    parser.add_argument("--timeout-seconds", type=float, default=86_400.0)
    parser.add_argument("--verification-grace-seconds", type=float, default=300.0)
    parser.add_argument("--upload-attempts", type=int, default=6)
    parser.add_argument("--retry-base-seconds", type=float, default=5.0)
    parser.add_argument("--retry-cap-seconds", type=float, default=300.0)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--allow-paper-run-upload", action="store_true")
    parser.add_argument("--confirm-paper-run")
    return parser


def _config_from_args(args: argparse.Namespace) -> WorkerConfig:
    campaign_root = _absolute(args.campaign_root)
    model_keys = (
        "source_model_id",
        "local_model_path",
        "source_revision",
        "model_id",
        "model_revision",
        "display_name",
        "repository_id",
        "repository_revision",
    )
    if min(
        args.poll_seconds,
        args.settle_seconds,
        args.timeout_seconds,
        args.verification_grace_seconds,
        args.retry_base_seconds,
        args.retry_cap_seconds,
    ) < 0:
        raise PublishWorkerError("timing arguments must be nonnegative")
    if args.upload_attempts < 1 or args.batch_size < 1:
        raise PublishWorkerError("upload attempts and batch size must be positive")
    return WorkerConfig(
        campaign_root=campaign_root,
        score_root=_absolute(args.score_root or campaign_root / "scoring"),
        archive_spool_root=_absolute(
            args.archive_spool_root or campaign_root / "hf_archive"
        ),
        dataset_manifest=_absolute(args.dataset_manifest),
        vlmeval_root=_absolute(args.vlmeval_root),
        work_root=_absolute(args.work_root),
        lock_root=_absolute(args.lock_root),
        source_run_id=args.source_run_id,
        public_run_id=args.public_run_id,
        seeds=tuple(args.seed),
        models=tuple(
            ModelConfig(**dict(zip(model_keys, values))) for values in args.model
        ),
        judge={
            "source_model_id": args.judge[0],
            "model_id": args.judge[1],
            "model_revision": args.judge[2],
        },
        token_env=args.token_env,
        paper_repo=args.paper_repo,
        revision=args.revision,
        poll_seconds=args.poll_seconds,
        settle_seconds=args.settle_seconds,
        timeout_seconds=args.timeout_seconds,
        verification_grace_seconds=args.verification_grace_seconds,
        upload_attempts=args.upload_attempts,
        retry_base_seconds=args.retry_base_seconds,
        retry_cap_seconds=args.retry_cap_seconds,
        batch_size=args.batch_size,
        allow_upload=args.allow_paper_run_upload,
        confirmation=args.confirm_paper_run,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    try:
        config = _config_from_args(parser.parse_args(argv))
        return run_worker(config)
    except (
        OSError,
        ValueError,
        TypeError,
        ArchiveError,
        PublicExportError,
        PublishWorkerError,
    ) as error:
        print(f"[trace-eval-publish:fatal] {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
