"""Dataset build pipeline for Trace."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import shutil
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping

from PIL import Image
from tqdm.auto import tqdm

from ..tasks import create_task
from ..tasks.base import TaskOutput
from . import error_codes
from .canonical import canonical_json_bytes
from .config import BuildConfig, BuildTaskConfig
from .annotation_sanitization import sanitize_trace_payload_for_public_annotation
from .hash_utils import blake3_file, blake3_hex
from .identity import compute_instance_id
from .json_io import write_json_file
from .reward_contracts import resolve_reward_contract
from .sampling import normalize_positive_weights, weighted_choice
from .source_layout_policy import uses_current_source_layout
from .seed import SEED_DERIVATION_VERSION, hash64
from .strict_repro import compare_staging_dirs
from .taxonomy import (
    inject_taxonomy_metadata,
    resolve_task_query_id,
    resolve_task_taxonomy,
)
from .trace_store import TraceShardWriter
from .type_registry import DEFAULT_REGISTRY_PATH, TypeRegistry, load_type_registry
from .types import CurriculumIndex, ImageRecord, TraceInstance, TrainInstance
from .validation import validate_candidate_instance, validate_dataset


class BuildError(RuntimeError):
    """Raised when a dataset build fails."""


class CandidateValidationError(BuildError):
    """Raised when a generated candidate fails pre-acceptance validation."""

    def __init__(self, validation_report: Mapping[str, Any]) -> None:
        self.validation_report = dict(validation_report)
        error_counts = self.validation_report.get("error_counts_by_code")
        if isinstance(error_counts, Mapping) and error_counts:
            summary = ",".join(str(code) for code in sorted(error_counts))
        else:
            summary = "unknown"
        super().__init__(f"candidate validation failed: {summary}")


def _candidate_validation_rejection_reason(report: Mapping[str, Any]) -> str:
    """Return one stable rejection reason key for a failed candidate."""

    counts = report.get("error_counts_by_code")
    if isinstance(counts, Mapping) and counts:
        codes = sorted(str(code) for code in counts)
        if len(codes) == 1:
            return f"candidate_validation:{codes[0]}"
        return "candidate_validation:multiple"
    return "candidate_validation:unknown"


@dataclass
class _BuildStageResult:
    """In-memory summary from one staging build pass."""

    instances: List[Dict[str, Any]]
    curriculum: List[Dict[str, Any]]
    target_counts_by_task: Dict[str, int]
    task_sampling_probabilities: Dict[str, float]
    sampler_mode: str
    accepted_by_task: Dict[str, int]
    rejected_by_task: Dict[str, int]
    rejected_reason_by_task: Dict[str, Dict[str, int]]
    annotation_format_map: Dict[str, str]
    domain_sampling_probabilities: Dict[str, float]
    scene_sampling_probabilities: Dict[str, float]
    warnings: List[str]


@dataclass(frozen=True)
class BuildPaths:
    """Filesystem locations derived from a build config."""

    dataset_id: str
    output_root: Path
    temp_root: Path
    repro_root: Path
    final_root: Path
    failure_root: Path


@dataclass(frozen=True)
class _TaskAttemptSpec:
    """Deterministic generation request for one task attempt."""

    task_id: str
    instance_seed: int
    params: Dict[str, Any]
    max_attempts: int
    seed_index: int


@dataclass
class _TaskAttemptOutcome:
    """Worker result for one task attempt."""

    status: str
    spec: _TaskAttemptSpec
    generated: TaskOutput | None = None
    error_reason: str | None = None


_REQUIRED_TRACE_PAYLOAD_KEYS = (
    "scene_ir",
    "query_spec",
    "render_spec",
    "render_map",
    "execution_trace",
    "witness_symbolic",
    "projected_annotation",
)
_PROGRESS_DISABLED_VALUES = {"0", "false", "no", "off"}
_BUILD_PROGRESS_ENABLED = (
    os.environ.get("TRACE_BUILD_PROGRESS", "1").strip().lower()
    not in _PROGRESS_DISABLED_VALUES
)


def _registered_scene_id(task_id: str, task: Any) -> str | None:
    """Return a class-level scene only for tasks outside the current layout."""

    if uses_current_source_layout(str(task_id), domain=str(getattr(task, "domain", ""))):
        return None
    return str(getattr(task, "scene_id", ""))


def _to_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    """Write JSONL records with deterministic key ordering per row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True)
            )
            handle.write("\n")


def _serialize_task_config(task: BuildTaskConfig) -> Dict[str, Any]:
    """Serialize task config into dataset-id/report-friendly mapping."""
    return {
        "task_id": task.task_id,
        "count": task.count,
        "weight": task.weight,
        "params": dict(task.params),
    }


def _dataset_id_from_config(
    config: BuildConfig, type_registry: TypeRegistry, type_registry_hash: str
) -> str:
    """Compute deterministic dataset id from build-critical configuration."""
    payload = {
        "dataset_name": config.dataset_name,
        "instance_version": config.instance_version,
        "image_format": config.image_format,
        "strict_repro": config.strict_repro,
        "max_attempts_per_instance": config.max_attempts_per_instance,
        "sampling_seed": config.sampling_seed,
        "num_instances": config.num_instances,
        "tasks": [_serialize_task_config(task) for task in config.tasks],
        "type_registry_version": type_registry.version,
        "type_registry_hash": type_registry_hash,
    }
    return blake3_hex(canonical_json_bytes(payload))


def _save_image(image: Image.Image, path: Path, image_format: str) -> None:
    """Persist an image artifact in the requested build output format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format=image_format.upper())


def _validate_trace_payload_keys(
    *, task_id: str, instance_seed: int, trace_payload: Mapping[str, Any]
) -> None:
    """Fail fast when a task omits mandatory sidecar-trace fields."""

    missing = [key for key in _REQUIRED_TRACE_PAYLOAD_KEYS if key not in trace_payload]
    if missing:
        raise BuildError(
            f"{task_id} instance_seed={instance_seed} missing required trace payload keys: "
            + ", ".join(missing)
        )


def _serialize_config(config: BuildConfig) -> Dict[str, Any]:
    """Serialize build config for failure bundles and diagnostics."""
    return {
        "output_root": config.output_root,
        "dataset_name": config.dataset_name,
        "instance_version": config.instance_version,
        "image_format": config.image_format,
        "num_instances": config.num_instances,
        "strict_repro": config.strict_repro,
        "max_attempts_per_instance": config.max_attempts_per_instance,
        "sampling_seed": config.sampling_seed,
        "workers": config.workers,
        "max_in_flight": config.max_in_flight,
        "tasks": [_serialize_task_config(task) for task in config.tasks],
    }


def _resolve_parallelism(config: BuildConfig, *, task_target: int) -> tuple[int, int]:
    """Resolve worker count and queue depth for one task build."""

    raw_workers = int(config.workers)
    if raw_workers < 0:
        raise BuildError("workers must be >= 0")
    if raw_workers == 0:
        workers = max(1, int(os.cpu_count() or 1))
    else:
        workers = raw_workers

    # Small task targets do not benefit from an oversized process pool.
    if int(task_target) > 0:
        workers = min(workers, int(task_target))
    workers = max(1, workers)

    raw_max_in_flight = int(config.max_in_flight)
    if raw_max_in_flight < 0:
        raise BuildError("max_in_flight must be >= 0")
    if raw_max_in_flight == 0:
        max_in_flight = max(1, workers * 2)
    else:
        max_in_flight = raw_max_in_flight
    return workers, max_in_flight


def _build_task_attempt_spec(
    *,
    task_cfg: BuildTaskConfig,
    config: BuildConfig,
    seed_index: int,
) -> _TaskAttemptSpec:
    """Build one deterministic task-attempt request."""

    instance_seed = hash64(
        config.sampling_seed, f"{task_cfg.task_id}:instance_seed", seed_index
    )
    params = dict(task_cfg.params)
    forbidden = [
        key
        for key in params
        if str(key).endswith("sampling_index") or str(key).endswith("sample_cursor")
    ]
    if forbidden:
        raise BuildError(
            f"build params for {task_cfg.task_id} use manual sampler controls: {sorted(str(key) for key in forbidden)}"
        )
    return _TaskAttemptSpec(
        task_id=str(task_cfg.task_id),
        instance_seed=int(instance_seed),
        params=params,
        max_attempts=int(config.max_attempts_per_instance),
        seed_index=int(seed_index),
    )


def _worker_generate_attempt(spec: _TaskAttemptSpec) -> _TaskAttemptOutcome:
    """Generate one task attempt inside a worker process."""

    task = create_task(spec.task_id)
    try:
        generated = task.generate(
            int(spec.instance_seed),
            params=dict(spec.params),
            max_attempts=int(spec.max_attempts),
        )
    except Exception as exc:
        return _TaskAttemptOutcome(
            status="error",
            spec=spec,
            error_reason=type(exc).__name__,
        )
    return _TaskAttemptOutcome(status="ok", spec=spec, generated=generated)


def _resolve_process_pool_context() -> mp.context.BaseContext | None:
    """Prefer a safe process start method for worker pools."""

    for method in ("forkserver", "spawn"):
        try:
            return mp.get_context(method)
        except ValueError:
            continue
    return None


def _write_failure_bundle(
    *,
    failure_root: Path,
    validation_report: Dict[str, Any] | None,
    resolved_build_config: Dict[str, Any],
    warning_messages: List[str],
) -> None:
    """Persist failure diagnostics under `failed_builds/<dataset_id>/`."""
    failure_root.mkdir(parents=True, exist_ok=True)
    if validation_report is not None:
        write_json_file(failure_root / "validation_report.json", validation_report)
    write_json_file(failure_root / "resolved_build_config.json", resolved_build_config)
    write_json_file(failure_root / "log_reference.json", {"warnings": warning_messages})


def _ensure_unique_task_ids(tasks: List[BuildTaskConfig]) -> None:
    """Fail fast when build config contains duplicate task identifiers."""
    seen: set[str] = set()
    for task in tasks:
        if task.task_id in seen:
            raise BuildError(f"duplicate task_id in config: {task.task_id}")
        seen.add(task.task_id)


def _resolve_task_targets(
    config: BuildConfig,
) -> tuple[Dict[str, int], Dict[str, float], str]:
    """Resolve per-task target counts and global task probabilities."""
    _ensure_unique_task_ids(config.tasks)
    has_explicit_counts = all(task.count is not None for task in config.tasks)

    if has_explicit_counts:
        target_counts: Dict[str, int] = {}
        for task in config.tasks:
            count = int(task.count or 0)
            if count < 0:
                raise BuildError(f"task count must be non-negative for {task.task_id}")
            target_counts[task.task_id] = count
        total = sum(target_counts.values())
        if total <= 0:
            raise BuildError("at least one task must request a positive count")
        task_probabilities = {
            task_id: float(count) / float(total)
            for task_id, count in sorted(target_counts.items())
        }
        return target_counts, task_probabilities, "explicit_counts"

    if config.num_instances is None or int(config.num_instances) <= 0:
        raise BuildError(
            "num_instances must be positive when explicit per-task counts are not provided"
        )

    configured_weights = {
        task.task_id: (float(task.weight) if task.weight is not None else 1.0)
        for task in config.tasks
    }
    try:
        task_probabilities = normalize_positive_weights(configured_weights)
    except ValueError as exc:
        raise BuildError(str(exc)) from exc
    target_counts = {task.task_id: 0 for task in config.tasks}

    sampler_rng = random.Random(hash64(config.sampling_seed, "global_task_sampler", 0))
    for _ in range(int(config.num_instances)):
        sampled_task = weighted_choice(sampler_rng, task_probabilities)
        target_counts[sampled_task] += 1

    return target_counts, task_probabilities, "weighted_task_sampler"


def _aggregate_sampling_probabilities(
    task_probabilities: Mapping[str, float],
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Aggregate public domain and scene probabilities from task weights."""
    domain_probs: Dict[str, float] = {}
    scene_probs: Dict[str, float] = {}
    for task_id, probability in task_probabilities.items():
        task = create_task(task_id)
        registered_scene_id = _registered_scene_id(str(task_id), task)
        taxonomy = resolve_task_taxonomy(
            str(task_id),
            source_domain=str(getattr(task, "domain", "")),
            source_scene_id=str(registered_scene_id or ""),
        )
        domain = str(taxonomy.domain)
        scene_id = str(taxonomy.scene_id)
        domain_probs[domain] = domain_probs.get(domain, 0.0) + float(probability)
        scene_probs[scene_id] = scene_probs.get(scene_id, 0.0) + float(probability)
    return dict(sorted(domain_probs.items())), dict(sorted(scene_probs.items()))


def _finalize_generated_output(
    *,
    task: Any,
    generated: TaskOutput,
    accepted_index: int,
    instance_seed: int,
    config: BuildConfig,
    stage_root: Path,
    code_hash: str,
    type_registry: TypeRegistry,
    trace_writer: TraceShardWriter,
) -> tuple[Dict[str, Any], Dict[str, Any], str]:
    """Finalize one generated task output into train/trace records."""

    query_id_used = str(getattr(generated, "query_id", "") or "")
    scene_id = _registered_scene_id(str(task.task_id), task)
    taxonomy = resolve_task_taxonomy(
        str(task.task_id),
        source_domain=str(getattr(task, "domain", "")),
        source_scene_id=str(scene_id or ""),
    )
    canonical_domain = str(taxonomy.domain)
    scene_id = str(getattr(generated, "scene_id", "") or taxonomy.scene_id)
    if scene_id != taxonomy.scene_id:
        taxonomy = resolve_task_taxonomy(
            str(task.task_id),
            source_domain=str(getattr(task, "domain", "")),
            source_scene_id=str(scene_id or ""),
        )
        taxonomy = replace(taxonomy, scene_id=scene_id)
    query_id = str(
        getattr(generated, "query_id", "")
        or resolve_task_query_id(
            query_id=query_id_used, trace_payload=generated.trace_payload
        )
    )
    if not type_registry.validate_answer_type(generated.answer_gt.type):
        raise BuildError(f"unregistered answer type: {generated.answer_gt.type}")
    if not type_registry.validate_annotation_type(generated.annotation_gt.type):
        raise BuildError(
            f"unregistered annotation type: {generated.annotation_gt.type}"
        )
    try:
        reward_contract = resolve_reward_contract(
            answer_type=generated.answer_gt.type,
            annotation_type=generated.annotation_gt.type,
        )
    except ValueError as exc:
        raise BuildError(str(exc)) from exc

    image_rel_path = (
        Path("images")
        / canonical_domain
        / task.task_id
        / f"{accepted_index:06d}.{config.image_format}"
    )
    image_abs_path = stage_root / image_rel_path
    _save_image(generated.image, image_abs_path, config.image_format)
    image_hash = blake3_file(image_abs_path)

    image_record = ImageRecord(
        image_id=generated.image_id,
        format=config.image_format,
        image_hash=image_hash,
        path=str(image_rel_path.as_posix()),
    )

    partial_record = {
        "instance_version": config.instance_version,
        "instance_seed": int(instance_seed),
        "domain": canonical_domain,
        "task": task.task_id,
        "query_id": query_id,
        "prompt": generated.prompt,
        "prompt_variants": dict(getattr(generated, "prompt_variants", {}) or {}),
        "images": [image_record.to_dict()],
        "answer_gt": generated.answer_gt.to_dict(),
        "annotation_gt": generated.annotation_gt.to_dict(),
        "reward_contract": reward_contract.to_dict(),
        "versions": {
            "seed_derivation_version": SEED_DERIVATION_VERSION,
            **generated.task_versions,
            "code_hash": code_hash,
        },
    }
    partial_record["scene_id"] = scene_id
    instance_id = compute_instance_id(partial_record)

    trace_payload = sanitize_trace_payload_for_public_annotation(
        generated.trace_payload,
        annotation_gt=generated.annotation_gt,
    )
    _validate_trace_payload_keys(
        task_id=str(task.task_id),
        instance_seed=int(instance_seed),
        trace_payload=trace_payload,
    )
    trace_payload = inject_taxonomy_metadata(
        trace_payload,
        task_id=str(task.task_id),
        taxonomy=taxonomy,
        query_id=query_id,
        registered_domain=str(getattr(task, "domain", "")),
        registered_scene_id=scene_id,
    )

    trace_instance = TraceInstance(
        instance_id=instance_id,
        scene_ir=trace_payload["scene_ir"],
        query_spec=trace_payload["query_spec"],
        render_spec=trace_payload["render_spec"],
        render_map=trace_payload["render_map"],
        execution_trace=trace_payload["execution_trace"],
        witness_symbolic=trace_payload["witness_symbolic"],
        projected_annotation=trace_payload["projected_annotation"],
        taxonomy=(
            trace_payload.get("taxonomy")
            if isinstance(trace_payload.get("taxonomy"), dict)
            else None
        ),
        answer_gt=generated.answer_gt,
        annotation_gt=generated.annotation_gt,
        reward_contract=reward_contract,
    )
    trace_record = trace_instance.to_dict()
    trace_ref = trace_writer.preview_ref(trace_record)

    train = TrainInstance(
        instance_version=config.instance_version,
        instance_id=instance_id,
        instance_seed=int(instance_seed),
        domain=canonical_domain,
        task=task.task_id,
        scene_id=scene_id,
        query_id=query_id,
        prompt=generated.prompt,
        prompt_variants=dict(getattr(generated, "prompt_variants", {}) or {}),
        images=[image_record],
        answer_gt=generated.answer_gt,
        annotation_gt=generated.annotation_gt,
        reward_contract=reward_contract,
        trace_ref=trace_ref,
        versions={
            "seed_derivation_version": SEED_DERIVATION_VERSION,
            **generated.task_versions,
            "code_hash": code_hash,
        },
    )
    train_record = train.to_dict()
    candidate_report = validate_candidate_instance(
        train_record,
        trace_record,
        staging_root=stage_root,
        expected_instance_version=config.instance_version,
        dataset_id="candidate",
    )
    if int(candidate_report.get("total_errors", 0)) > 0:
        try:
            image_abs_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise CandidateValidationError(candidate_report)
    appended_ref = trace_writer.append(trace_record)
    if appended_ref.to_dict() != trace_ref.to_dict():
        raise BuildError("trace_ref preview did not match appended trace_ref")

    curriculum_record = CurriculumIndex(
        instance_id=instance_id,
        domain=canonical_domain,
        task=task.task_id,
        scene_id=scene_id,
        query_id=query_id,
    ).to_dict()
    return train_record, curriculum_record, str(generated.annotation_gt.type)


def _build_task_serial(
    *,
    task_cfg: BuildTaskConfig,
    task: Any,
    task_target: int,
    config: BuildConfig,
    stage_root: Path,
    code_hash: str,
    type_registry: TypeRegistry,
    trace_writer: TraceShardWriter,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> tuple[
    list[Dict[str, Any]], list[Dict[str, Any]], int, int, Dict[str, int], str | None
]:
    """Generate all accepted instances for one task in-process."""

    accepted = 0
    rejected = 0
    rejection_reasons: Dict[str, int] = {}
    seed_index = 0
    max_candidates = max(1, task_target * 20)
    train_records: list[Dict[str, Any]] = []
    curriculum_records: list[Dict[str, Any]] = []
    annotation_type: str | None = None

    while accepted < task_target and seed_index < max_candidates:
        attempt = _build_task_attempt_spec(
            task_cfg=task_cfg, config=config, seed_index=seed_index
        )
        seed_index += 1
        try:
            generated = task.generate(
                int(attempt.instance_seed),
                params=dict(attempt.params),
                max_attempts=int(attempt.max_attempts),
            )
        except Exception as exc:
            rejected += 1
            reason = type(exc).__name__
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            if progress_callback is not None:
                progress_callback(accepted, rejected, accepted + rejected)
            continue

        try:
            train_record, curriculum_record, annotation_type = _finalize_generated_output(
                task=task,
                generated=generated,
                accepted_index=accepted,
                instance_seed=int(attempt.instance_seed),
                config=config,
                stage_root=stage_root,
                code_hash=code_hash,
                type_registry=type_registry,
                trace_writer=trace_writer,
            )
        except CandidateValidationError as exc:
            rejected += 1
            reason = _candidate_validation_rejection_reason(exc.validation_report)
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            if progress_callback is not None:
                progress_callback(accepted, rejected, accepted + rejected)
            continue
        train_records.append(train_record)
        curriculum_records.append(curriculum_record)
        accepted += 1
        if progress_callback is not None:
            progress_callback(accepted, rejected, accepted + rejected)

    return (
        train_records,
        curriculum_records,
        accepted,
        rejected,
        dict(sorted(rejection_reasons.items())),
        annotation_type,
    )


def _build_task_parallel(
    *,
    task_cfg: BuildTaskConfig,
    task: Any,
    task_target: int,
    config: BuildConfig,
    stage_root: Path,
    code_hash: str,
    type_registry: TypeRegistry,
    trace_writer: TraceShardWriter,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> tuple[
    list[Dict[str, Any]], list[Dict[str, Any]], int, int, Dict[str, int], str | None
]:
    """Generate all accepted instances for one task with a deterministic worker pool."""

    workers, max_in_flight = _resolve_parallelism(config, task_target=task_target)
    if workers <= 1:
        return _build_task_serial(
            task_cfg=task_cfg,
            task=task,
            task_target=task_target,
            config=config,
            stage_root=stage_root,
            code_hash=code_hash,
            type_registry=type_registry,
            trace_writer=trace_writer,
            progress_callback=progress_callback,
        )

    accepted = 0
    rejected = 0
    rejection_reasons: Dict[str, int] = {}
    seed_index = 0
    next_finalize = 0
    max_candidates = max(1, task_target * 20)
    pending_results: Dict[int, _TaskAttemptOutcome] = {}
    futures: Dict[Any, int] = {}
    train_records: list[Dict[str, Any]] = []
    curriculum_records: list[Dict[str, Any]] = []
    annotation_type: str | None = None

    executor_kwargs: Dict[str, Any] = {"max_workers": workers}
    process_pool_context = _resolve_process_pool_context()
    if process_pool_context is not None:
        executor_kwargs["mp_context"] = process_pool_context

    with ProcessPoolExecutor(**executor_kwargs) as executor:
        while futures or (seed_index < max_candidates and accepted < task_target):
            while (
                accepted < task_target
                and seed_index < max_candidates
                and len(futures) < max_in_flight
            ):
                attempt = _build_task_attempt_spec(
                    task_cfg=task_cfg, config=config, seed_index=seed_index
                )
                future = executor.submit(_worker_generate_attempt, attempt)
                futures[future] = int(attempt.seed_index)
                seed_index += 1

            if not futures:
                break

            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                result_seed_index = futures.pop(future)
                try:
                    outcome = future.result()
                except Exception as exc:
                    attempt = _build_task_attempt_spec(
                        task_cfg=task_cfg, config=config, seed_index=result_seed_index
                    )
                    outcome = _TaskAttemptOutcome(
                        status="error",
                        spec=attempt,
                        error_reason=type(exc).__name__,
                    )
                pending_results[result_seed_index] = outcome

            while next_finalize in pending_results and accepted < task_target:
                outcome = pending_results.pop(next_finalize)
                if outcome.status != "ok" or outcome.generated is None:
                    rejected += 1
                    reason = str(outcome.error_reason or "TaskGenerationError")
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                else:
                    try:
                        train_record, curriculum_record, annotation_type = _finalize_generated_output(
                            task=task,
                            generated=outcome.generated,
                            accepted_index=accepted,
                            instance_seed=int(outcome.spec.instance_seed),
                            config=config,
                            stage_root=stage_root,
                            code_hash=code_hash,
                            type_registry=type_registry,
                            trace_writer=trace_writer,
                        )
                    except CandidateValidationError as exc:
                        rejected += 1
                        reason = _candidate_validation_rejection_reason(exc.validation_report)
                        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                    else:
                        train_records.append(train_record)
                        curriculum_records.append(curriculum_record)
                        accepted += 1
                if progress_callback is not None:
                    progress_callback(accepted, rejected, accepted + rejected)
                next_finalize += 1

            if accepted >= task_target:
                for future in futures:
                    future.cancel()
                break

    return (
        train_records,
        curriculum_records,
        accepted,
        rejected,
        dict(sorted(rejection_reasons.items())),
        annotation_type,
    )


def _build_staging(
    config: BuildConfig,
    *,
    stage_root: Path,
    code_hash: str,
    type_registry: TypeRegistry,
) -> _BuildStageResult:
    """Generate one staging dataset pass and return in-memory build summary."""
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True, exist_ok=True)

    target_counts_by_task, task_probabilities, sampler_mode = _resolve_task_targets(
        config
    )
    domain_probs, scene_probs = _aggregate_sampling_probabilities(task_probabilities)

    warning_messages: List[str] = []
    instances: List[Dict[str, Any]] = []
    curriculum: List[Dict[str, Any]] = []

    accepted_by_task: Dict[str, int] = {}
    rejected_by_task: Dict[str, int] = {}
    rejected_reason_by_task: Dict[str, Dict[str, int]] = {}
    annotation_format_map: Dict[str, str] = {}
    progress_enabled = _BUILD_PROGRESS_ENABLED
    total_target = sum(target_counts_by_task.values())
    overall_bar = tqdm(
        total=total_target,
        desc="Trace build",
        unit="inst",
        dynamic_ncols=True,
        disable=not progress_enabled,
    )

    try:
        with TraceShardWriter(stage_root) as trace_writer:
            for task_index, task_cfg in enumerate(config.tasks, start=1):
                task = create_task(task_cfg.task_id)
                task_target = int(target_counts_by_task.get(task_cfg.task_id, 0))
                current_task_id = str(task_cfg.task_id)
                task_bar = tqdm(
                    total=task_target,
                    desc=f"[{task_index}/{len(config.tasks)}] {current_task_id}",
                    unit="inst",
                    dynamic_ncols=True,
                    leave=False,
                    disable=not progress_enabled,
                )
                progress_state = {"accepted": 0, "rejected": 0, "finalized": 0}

                def _on_task_progress(
                    accepted: int,
                    rejected: int,
                    finalized: int,
                    *,
                    _task_id: str = current_task_id,
                ) -> None:
                    accepted_delta = accepted - int(progress_state["accepted"])
                    if accepted_delta > 0:
                        task_bar.update(accepted_delta)
                        overall_bar.update(accepted_delta)
                    progress_state["accepted"] = int(accepted)
                    progress_state["rejected"] = int(rejected)
                    progress_state["finalized"] = int(finalized)

                    if not progress_enabled:
                        return
                    if (
                        accepted_delta > 0
                        or rejected == 0
                        or rejected % 25 == 0
                        or accepted >= task_target
                    ):
                        task_bar.set_postfix(
                            rejected=rejected,
                            attempts=finalized,
                            refresh=accepted_delta == 0,
                        )
                        overall_bar.set_postfix(task=_task_id, refresh=False)

                try:
                    (
                        task_instances,
                        task_curriculum,
                        accepted,
                        rejected,
                        rejection_reasons,
                        annotation_type,
                    ) = _build_task_parallel(
                        task_cfg=task_cfg,
                        task=task,
                        task_target=task_target,
                        config=config,
                        stage_root=stage_root,
                        code_hash=code_hash,
                        type_registry=type_registry,
                        trace_writer=trace_writer,
                        progress_callback=_on_task_progress,
                    )
                finally:
                    if progress_enabled:
                        task_bar.set_postfix(
                            rejected=int(progress_state["rejected"]),
                            attempts=int(progress_state["finalized"]),
                            refresh=False,
                        )
                    task_bar.close()

                instances.extend(task_instances)
                curriculum.extend(task_curriculum)
                if annotation_type is not None:
                    annotation_format_map[task.task_id] = annotation_type

                accepted_by_task[task_cfg.task_id] = accepted
                rejected_by_task[task_cfg.task_id] = rejected
                rejected_reason_by_task[task_cfg.task_id] = dict(rejection_reasons)

                if accepted < task_target:
                    warning_messages.append(
                        f"task {task_cfg.task_id} shortfall: expected {task_target}, accepted {accepted}"
                    )
    finally:
        overall_bar.close()

    _to_jsonl(stage_root / "train_instances.jsonl", instances)
    _to_jsonl(stage_root / "curriculum_index.jsonl", curriculum)

    return _BuildStageResult(
        instances=instances,
        curriculum=curriculum,
        target_counts_by_task=dict(sorted(target_counts_by_task.items())),
        task_sampling_probabilities=dict(sorted(task_probabilities.items())),
        sampler_mode=sampler_mode,
        accepted_by_task=dict(sorted(accepted_by_task.items())),
        rejected_by_task=dict(sorted(rejected_by_task.items())),
        rejected_reason_by_task=dict(sorted(rejected_reason_by_task.items())),
        annotation_format_map=dict(sorted(annotation_format_map.items())),
        domain_sampling_probabilities=domain_probs,
        scene_sampling_probabilities=scene_probs,
        warnings=warning_messages,
    )


def resolve_build_paths(config: BuildConfig) -> BuildPaths:
    """Resolve dataset-id-scoped build artifact paths for one config."""

    output_root = Path(config.output_root)
    type_registry_path = DEFAULT_REGISTRY_PATH
    type_registry = load_type_registry(type_registry_path)
    type_registry_hash = blake3_file(type_registry_path)
    dataset_id = _dataset_id_from_config(config, type_registry, type_registry_hash)
    return BuildPaths(
        dataset_id=dataset_id,
        output_root=output_root,
        temp_root=output_root / "tmp" / dataset_id,
        repro_root=output_root / "tmp" / f"{dataset_id}__strict_repro",
        final_root=output_root / "datasets" / dataset_id,
        failure_root=output_root / "failed_builds" / dataset_id,
    )


def build_dataset(config: BuildConfig, *, code_hash: str = "local") -> Path:
    """Build a dataset into output_root and return finalized dataset path."""
    build_paths = resolve_build_paths(config)
    output_root = build_paths.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    type_registry_path = DEFAULT_REGISTRY_PATH
    type_registry = load_type_registry(type_registry_path)
    type_registry_hash = blake3_file(type_registry_path)

    dataset_id = build_paths.dataset_id
    temp_root = build_paths.temp_root
    repro_root = build_paths.repro_root
    final_root = build_paths.final_root
    failure_root = build_paths.failure_root

    if final_root.exists():
        raise BuildError(f"final dataset path already exists: {final_root}")

    validation_report: Dict[str, Any] | None = None
    warning_messages: List[str] = []

    try:
        primary = _build_staging(
            config,
            stage_root=temp_root,
            code_hash=code_hash,
            type_registry=type_registry,
        )
        warning_messages.extend(primary.warnings)

        if config.strict_repro:
            _build_staging(
                config,
                stage_root=repro_root,
                code_hash=code_hash,
                type_registry=type_registry,
            )
            repro_mismatch = compare_staging_dirs(temp_root, repro_root)
            if repro_mismatch is not None:
                summary = json.dumps(repro_mismatch, ensure_ascii=False, sort_keys=True)
                raise BuildError(f"strict_repro_mismatch: {summary}")
            shutil.rmtree(repro_root)

        validation_report = validate_dataset(
            primary.instances,
            staging_root=temp_root,
            expected_task_counts=primary.target_counts_by_task,
            dataset_id=dataset_id,
            expected_instance_version=config.instance_version,
        )
        write_json_file(temp_root / "validation_report.json", validation_report)

        total_accepted = sum(primary.accepted_by_task.values())
        total_rejected = sum(primary.rejected_by_task.values())
        rejection_rate = (
            float(total_rejected) / float(total_accepted + total_rejected)
            if (total_accepted + total_rejected)
            else 0.0
        )

        sampler_report = {
            "mode": primary.sampler_mode,
            "task_sampling_probabilities": primary.task_sampling_probabilities,
            "domain_sampling_probabilities": primary.domain_sampling_probabilities,
            "scene_sampling_probabilities": primary.scene_sampling_probabilities,
        }
        if primary.scene_sampling_probabilities:
            sampler_report["scene_sampling_probabilities"] = (
                primary.scene_sampling_probabilities
            )

        build_report = {
            "build_report_schema_version": "v0",
            "dataset_id": dataset_id,
            "sampler": sampler_report,
            "accepted_counts_by_task": primary.accepted_by_task,
            "rejected_counts_by_task": primary.rejected_by_task,
            "rejection_reason_breakdown_by_task": primary.rejected_reason_by_task,
            "final_rejection_rate": rejection_rate,
            "resolved_annotation_format_map": primary.annotation_format_map,
            "trace_shard_manifest": {
                "shards": [
                    {
                        "shard_id": "trace_shard_0001.jsonl.zst",
                        "path": "traces/trace_shard_0001.jsonl.zst",
                        "record_count": len(primary.instances),
                    }
                ],
            },
            "trace_hash_policy": {
                "algorithm": "blake3",
                "canonicalization": "RFC8785 JCS",
                "seed_derivation_version": SEED_DERIVATION_VERSION,
            },
            "type_registry": {
                "type_registry_version": type_registry.version,
                "path": str(type_registry_path),
                "hash": type_registry_hash,
            },
            "code_provenance": {
                "code_hash": code_hash,
                "identity_input": True,
            },
            "split_metadata": None,
            "image_encoding": {
                "image_format": config.image_format,
                "compression": None,
                "color_mode": "RGB",
                "resolution_policy": "task_defined",
            },
            "warnings": warning_messages,
        }
        write_json_file(temp_root / "build_report.json", build_report)

        if int(validation_report.get("total_errors", 0)) > 0:
            raise BuildError(
                f"validation failed with {validation_report['total_errors']} errors"
            )

        final_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(temp_root), str(final_root))
        return final_root

    except Exception as exc:
        warning_messages.append(f"build_error: {exc}")
        try:
            if failure_root.exists():
                shutil.rmtree(failure_root)
            _write_failure_bundle(
                failure_root=failure_root,
                validation_report=validation_report,
                resolved_build_config=_serialize_config(config),
                warning_messages=warning_messages,
            )
        except Exception as bundle_exc:
            warning_messages.append(
                f"{error_codes.IO_FAILURE_BUNDLE_WRITE_FAILED}: {bundle_exc}"
            )
        raise
