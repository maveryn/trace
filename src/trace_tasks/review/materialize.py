"""Atomic, resumable materialization and verification of review recipes."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image

from trace_tasks.core.canonical import canonical_json_bytes

from .models import (
    ARTIFACT_SCHEMA_VERSION,
    MATERIALIZATION_SCHEMA_VERSION,
    ArtifactHashes,
    MaterializationError,
    MaterializationReport,
    RecipeManifest,
    ReviewRequest,
    VerificationIssue,
    VerificationReport,
)
from .provenance import sha256_bytes, sha256_file
from .recipe import (
    GeneratorFn,
    _default_generator,
    _raw_pixel_hash,
    filter_requests,
    load_recipe,
    prepare_output,
    resolve_output_query_id,
)


def _relative_artifact_paths(request: ReviewRequest) -> tuple[Path, Path]:
    task_root = Path(request.domain) / request.scene_id / request.task_id
    filename = f"{request.ordinal:04d}"
    data_path = task_root / "data" / request.query_id / f"{filename}.json"
    image_path = task_root / "images" / request.query_id / f"{filename}.png"
    return data_path, image_path


def _compact_recipe_provenance(manifest: RecipeManifest) -> dict[str, Any]:
    """Return path-free producer identities safe to repeat in review artifacts."""

    source = manifest.provenance.source
    resources = manifest.provenance.resources
    return {
        "producer_repository": source.repository,
        "producer_revision": source.revision,
        "source_dirty": bool(source.dirty),
        "source_tree_hash": source.source_tree_hash,
        "generator_tree_hash": source.generator_tree_hash,
        "constraints_hash": source.constraints_hash,
        "resource_tree_hash": resources.resource_tree_hash,
        "prompt_bundle_tree_hash": resources.prompt_bundle_tree_hash,
        "task_catalog_hash": resources.task_catalog_hash,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"invalid materialized JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise MaterializationError(f"expected a JSON object: {path}")
    return {str(key): item for key, item in value.items()}


def _rendering_issues(
    *,
    request: ReviewRequest,
    observed: ArtifactHashes,
    strict_rendering: bool,
) -> list[VerificationIssue]:
    severity = "error" if strict_rendering else "warning"
    issues: list[VerificationIssue] = []
    if observed.raw_pixel_hash != request.hashes.raw_pixel_hash:
        issues.append(
            VerificationIssue(
                severity=severity,
                code="raw_pixel_hash_mismatch",
                message=(
                    "host-native raw pixels differ from the capture runtime "
                    f"({request.hashes.raw_pixel_hash} != {observed.raw_pixel_hash})"
                ),
                task_id=request.task_id,
                ordinal=request.ordinal,
            )
        )
    if observed.png_hash != request.hashes.png_hash:
        issues.append(
            VerificationIssue(
                severity=severity,
                code="png_hash_mismatch",
                message=(
                    "host-native PNG encoding differs from the capture runtime "
                    f"({request.hashes.png_hash} != {observed.png_hash})"
                ),
                task_id=request.task_id,
                ordinal=request.ordinal,
            )
        )
    return issues


def _replay_request(
    request: ReviewRequest,
    *,
    generator: GeneratorFn,
    strict_rendering: bool,
) -> tuple[Any, list[VerificationIssue]]:
    output = generator(
        request.task_id,
        request.seed,
        dict(request.params),
        request.max_attempts,
    )
    observed_query = resolve_output_query_id(output)
    if observed_query != request.query_id:
        raise MaterializationError(
            f"{request.task_id} ordinal={request.ordinal} produced query "
            f"{observed_query!r}, expected {request.query_id!r}"
        )
    prepared = prepare_output(output, task_id=request.task_id)
    if prepared.hashes.semantic_hash != request.hashes.semantic_hash:
        raise MaterializationError(
            f"{request.task_id} ordinal={request.ordinal} semantic hash mismatch: "
            f"expected {request.hashes.semantic_hash}, observed "
            f"{prepared.hashes.semantic_hash}"
        )
    rendering = _rendering_issues(
        request=request,
        observed=prepared.hashes,
        strict_rendering=strict_rendering,
    )
    strict_errors = [issue for issue in rendering if issue.severity == "error"]
    if strict_errors:
        raise MaterializationError(strict_errors[0].message)
    return prepared, rendering


def _artifact_payload(
    *,
    request: ReviewRequest,
    prepared: Any,
    recipe_digest: str,
    recipe_provenance: Mapping[str, Any],
    image_path: Path,
    rendering_issues: Sequence[VerificationIssue],
) -> dict[str, Any]:
    semantic = dict(prepared.semantic_payload)
    return {
        "schema": ARTIFACT_SCHEMA_VERSION,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "recipe_id": "trace-review-recipe-v1",
        "recipe_digest": recipe_digest,
        "recipe_provenance": dict(recipe_provenance),
        "request": request.to_dict(),
        "semantic_payload": semantic,
        # Keep review-facing fields at the top level while retaining the exact
        # semantic payload used for hashing above.
        "task_id": request.task_id,
        "domain": request.domain,
        "scene_id": request.scene_id,
        "query_id": request.query_id,
        "instance_seed": int(request.seed),
        "prompt": semantic["prompt"],
        "prompt_variants": semantic["prompt_variants"],
        "answer_gt": semantic["answer_gt"],
        "annotation_gt": semantic["annotation_gt"],
        "reward_contract": semantic["reward_contract"],
        "taxonomy": semantic["taxonomy"],
        "trace_payload": semantic["trace_payload"],
        "versions": semantic["versions"],
        "expected_hashes": request.hashes.to_dict(),
        "observed_hashes": prepared.hashes.to_dict(),
        "image": {
            "path": image_path.as_posix(),
            "format": "png",
            "mode": prepared.image_mode,
            "width": int(prepared.image_size[0]),
            "height": int(prepared.image_size[1]),
        },
        "rendering_warnings": [
            issue.to_dict() for issue in rendering_issues if issue.severity == "warning"
        ],
    }


def _build_task_stage(
    stage_task: Path,
    *,
    output_root: Path,
    requests: Sequence[ReviewRequest],
    recipe_digest: str,
    source_provenance: Mapping[str, Any],
    recipe_provenance: Mapping[str, Any],
    generator: GeneratorFn,
    strict_rendering: bool,
) -> tuple[dict[str, Any], list[VerificationIssue]]:
    request0 = requests[0]
    entries: list[dict[str, Any]] = []
    warnings: list[VerificationIssue] = []
    for request in requests:
        prepared, request_issues = _replay_request(
            request,
            generator=generator,
            strict_rendering=strict_rendering,
        )
        warnings.extend(request_issues)
        relative_data, relative_image = _relative_artifact_paths(request)
        task_relative_data = relative_data.relative_to(
            Path(request.domain) / request.scene_id / request.task_id
        )
        task_relative_image = relative_image.relative_to(
            Path(request.domain) / request.scene_id / request.task_id
        )
        staged_image = stage_task / task_relative_image
        staged_image.parent.mkdir(parents=True, exist_ok=True)
        staged_image.write_bytes(prepared.png_bytes)
        data_payload = _artifact_payload(
            request=request,
            prepared=prepared,
            recipe_digest=recipe_digest,
            recipe_provenance=recipe_provenance,
            image_path=relative_image,
            rendering_issues=request_issues,
        )
        staged_data = stage_task / task_relative_data
        _write_json(staged_data, data_payload)
        entries.append(
            {
                "ordinal": int(request.ordinal),
                "query_id": request.query_id,
                "data_path": relative_data.as_posix(),
                "image_path": relative_image.as_posix(),
                "data_sha256": sha256_file(staged_data),
                "expected_hashes": request.hashes.to_dict(),
                "observed_hashes": prepared.hashes.to_dict(),
            }
        )
    manifest = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "recipe_id": "trace-review-recipe-v1",
        "recipe_digest": recipe_digest,
        "source_provenance": dict(source_provenance),
        "recipe_provenance": dict(recipe_provenance),
        "task_id": request0.task_id,
        "domain": request0.domain,
        "scene_id": request0.scene_id,
        "request_count": len(requests),
        "query_ids": sorted({request.query_id for request in requests}),
        "entries": entries,
    }
    _write_json(stage_task / "manifest.json", manifest)
    return manifest, warnings


def _resolved_child(root: Path, relative: str, *, require_file: bool = True) -> Path:
    candidate = root / relative
    try:
        resolved = candidate.resolve(strict=require_file)
    except FileNotFoundError as exc:
        raise MaterializationError(
            f"missing materialized artifact: {relative}"
        ) from exc
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise MaterializationError(f"materialized path escapes output root: {relative}")
    return resolved


def _validate_existing_task(
    task_dir: Path,
    *,
    output_root: Path,
    requests: Sequence[ReviewRequest],
    manifest_requests: Sequence[ReviewRequest] | None = None,
    recipe_digest: str,
    source_provenance: Mapping[str, Any],
    recipe_provenance: Mapping[str, Any],
) -> list[VerificationIssue]:
    """Bind a materialized task and every review-facing duplicate to its recipe."""

    manifest_rows = tuple(manifest_requests or requests)
    request0 = manifest_rows[0]
    issues: list[VerificationIssue] = []

    def error(code: str, message: str, *, ordinal: int | None = None) -> None:
        issues.append(
            VerificationIssue(
                severity="error",
                code=code,
                message=message,
                task_id=request0.task_id,
                ordinal=ordinal,
            )
        )

    try:
        manifest = _load_json(task_dir / "manifest.json")
    except MaterializationError as exc:
        error("invalid_task_manifest", str(exc))
        return issues
    expected_header = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "recipe_id": "trace-review-recipe-v1",
        "recipe_digest": recipe_digest,
        "source_provenance": dict(source_provenance),
        "recipe_provenance": dict(recipe_provenance),
        "task_id": request0.task_id,
        "domain": request0.domain,
        "scene_id": request0.scene_id,
        "request_count": len(manifest_rows),
    }
    for key, expected in expected_header.items():
        if manifest.get(key) != expected:
            error(
                "task_manifest_mismatch",
                f"{task_dir}/manifest.json {key} does not match the selected recipe",
            )
    expected_query_ids = sorted({request.query_id for request in manifest_rows})
    if manifest.get("query_ids") != expected_query_ids:
        error(
            "task_manifest_query_ids_mismatch",
            f"{task_dir}/manifest.json query_ids do not match the selected recipe",
        )
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != len(manifest_rows):
        error(
            "task_manifest_entries_mismatch",
            f"{task_dir}/manifest.json does not contain the selected request entries",
        )
        return issues
    expected_ordinals = [request.ordinal for request in manifest_rows]
    observed_ordinals = [
        entry.get("ordinal") if isinstance(entry, Mapping) else None
        for entry in entries
    ]
    if observed_ordinals != expected_ordinals:
        error(
            "task_manifest_entry_order_mismatch",
            f"{task_dir}/manifest.json entries are not in canonical recipe order",
        )
    entries_by_ordinal = {
        entry.get("ordinal"): entry for entry in entries if isinstance(entry, Mapping)
    }
    for request in requests:
        entry = entries_by_ordinal.get(request.ordinal)
        if not isinstance(entry, Mapping):
            error(
                "missing_materialized_request",
                f"missing task manifest entry for ordinal {request.ordinal}",
                ordinal=request.ordinal,
            )
            continue
        relative_data, relative_image = _relative_artifact_paths(request)
        expected_entry_header = {
            "ordinal": request.ordinal,
            "query_id": request.query_id,
            "data_path": relative_data.as_posix(),
            "image_path": relative_image.as_posix(),
            "expected_hashes": request.hashes.to_dict(),
        }
        for key, expected in expected_entry_header.items():
            if entry.get(key) != expected:
                error(
                    "task_manifest_entry_mismatch",
                    f"task manifest entry {request.ordinal} {key} does not match the recipe",
                    ordinal=request.ordinal,
                )
        if (
            entry.get("data_path") != relative_data.as_posix()
            or entry.get("image_path") != relative_image.as_posix()
        ):
            error(
                "artifact_path_mismatch",
                f"artifact paths differ for ordinal {request.ordinal}",
                ordinal=request.ordinal,
            )
            continue
        try:
            data_path = _resolved_child(output_root, relative_data.as_posix())
            image_path = _resolved_child(output_root, relative_image.as_posix())
            data = _load_json(data_path)
        except MaterializationError as exc:
            error("invalid_materialized_artifact", str(exc), ordinal=request.ordinal)
            continue
        if entry.get("data_sha256") != sha256_file(data_path):
            error(
                "data_hash_mismatch",
                f"materialized JSON changed for ordinal {request.ordinal}",
                ordinal=request.ordinal,
            )
        if data.get("request") != request.to_dict():
            error(
                "request_mismatch",
                f"materialized request differs for ordinal {request.ordinal}",
                ordinal=request.ordinal,
            )
        expected_artifact_header = {
            "schema": ARTIFACT_SCHEMA_VERSION,
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "recipe_id": "trace-review-recipe-v1",
            "recipe_digest": recipe_digest,
            "recipe_provenance": dict(recipe_provenance),
            "task_id": request.task_id,
            "domain": request.domain,
            "scene_id": request.scene_id,
            "query_id": request.query_id,
            "instance_seed": request.seed,
        }
        for key, expected in expected_artifact_header.items():
            if data.get(key) != expected:
                error(
                    "artifact_header_mismatch",
                    f"materialized artifact {request.ordinal} {key} does not match the recipe",
                    ordinal=request.ordinal,
                )
        if data.get("expected_hashes") != request.hashes.to_dict():
            error(
                "expected_hashes_mismatch",
                f"materialized expected hashes differ for ordinal {request.ordinal}",
                ordinal=request.ordinal,
            )
        try:
            observed = ArtifactHashes.from_dict(data.get("observed_hashes", {}))
        except Exception as exc:
            error(
                "invalid_observed_hashes",
                f"invalid observed hashes for ordinal {request.ordinal}: {exc}",
                ordinal=request.ordinal,
            )
            continue
        if entry.get("observed_hashes") != observed.to_dict():
            error(
                "task_manifest_observed_hashes_mismatch",
                f"task manifest observed hashes differ for ordinal {request.ordinal}",
                ordinal=request.ordinal,
            )
        semantic = data.get("semantic_payload")
        semantic_hash = ""
        if not isinstance(semantic, Mapping):
            error(
                "invalid_semantic_payload",
                f"materialized semantic payload is not an object for ordinal {request.ordinal}",
                ordinal=request.ordinal,
            )
            semantic = {}
        else:
            try:
                semantic_hash = sha256_bytes(canonical_json_bytes(semantic))
            except Exception as exc:
                error(
                    "invalid_semantic_payload",
                    f"cannot hash semantic payload for ordinal {request.ordinal}: {exc}",
                    ordinal=request.ordinal,
                )
        if semantic_hash != request.hashes.semantic_hash:
            error(
                "semantic_recipe_hash_mismatch",
                f"materialized semantics do not match the recipe for ordinal {request.ordinal}",
                ordinal=request.ordinal,
            )
        if observed.semantic_hash != request.hashes.semantic_hash:
            error(
                "observed_semantic_hash_mismatch",
                f"observed semantic hash does not match the recipe for ordinal {request.ordinal}",
                ordinal=request.ordinal,
            )
        if semantic_hash != observed.semantic_hash:
            error(
                "semantic_artifact_hash_mismatch",
                f"materialized semantic payload changed for ordinal {request.ordinal}",
                ordinal=request.ordinal,
            )
        duplicate_fields = (
            "prompt",
            "prompt_variants",
            "answer_gt",
            "annotation_gt",
            "reward_contract",
            "taxonomy",
            "trace_payload",
            "versions",
        )
        for key in duplicate_fields:
            if data.get(key) != semantic.get(key):
                error(
                    "review_field_mismatch",
                    f"top-level {key} is not bound to semantic_payload for ordinal {request.ordinal}",
                    ordinal=request.ordinal,
                )
        expected_taxonomy = {
            "domain": request.domain,
            "scene_id": request.scene_id,
            "task_id": request.task_id,
            "query_id": request.query_id,
        }
        if semantic.get("taxonomy") != expected_taxonomy:
            error(
                "semantic_taxonomy_mismatch",
                f"semantic taxonomy does not match the recipe for ordinal {request.ordinal}",
                ordinal=request.ordinal,
            )
        if (
            semantic.get("scene_id") != request.scene_id
            or semantic.get("query_id") != request.query_id
        ):
            error(
                "semantic_header_mismatch",
                f"semantic scene/query do not match the recipe for ordinal {request.ordinal}",
                ordinal=request.ordinal,
            )
        expected_warnings = [
            issue.to_dict()
            for issue in _rendering_issues(
                request=request,
                observed=observed,
                strict_rendering=False,
            )
            if issue.severity == "warning"
        ]
        if data.get("rendering_warnings") != expected_warnings:
            error(
                "rendering_warnings_mismatch",
                f"rendering warnings do not match observed hashes for ordinal {request.ordinal}",
                ordinal=request.ordinal,
            )
        if sha256_file(image_path) != observed.png_hash:
            error(
                "png_artifact_hash_mismatch",
                f"materialized PNG changed for ordinal {request.ordinal}",
                ordinal=request.ordinal,
            )
        image_metadata = data.get("image")
        expected_image_header = {
            "path": relative_image.as_posix(),
            "format": "png",
        }
        if not isinstance(image_metadata, Mapping):
            error(
                "invalid_image_metadata",
                f"image metadata is not an object for ordinal {request.ordinal}",
                ordinal=request.ordinal,
            )
            image_metadata = {}
        for key, expected in expected_image_header.items():
            if image_metadata.get(key) != expected:
                error(
                    "image_metadata_mismatch",
                    f"image metadata {key} differs for ordinal {request.ordinal}",
                    ordinal=request.ordinal,
                )
        try:
            with Image.open(image_path) as image:
                image.load()
                raw_hash = _raw_pixel_hash(image)
                actual_image_metadata = {
                    "mode": str(image.mode),
                    "width": int(image.width),
                    "height": int(image.height),
                }
        except Exception as exc:
            error(
                "invalid_png_artifact",
                f"cannot decode PNG for ordinal {request.ordinal}: {exc}",
                ordinal=request.ordinal,
            )
        else:
            for key, expected in actual_image_metadata.items():
                if image_metadata.get(key) != expected:
                    error(
                        "image_metadata_mismatch",
                        f"image metadata {key} differs for ordinal {request.ordinal}",
                        ordinal=request.ordinal,
                    )
            if raw_hash != observed.raw_pixel_hash:
                error(
                    "raw_pixel_artifact_hash_mismatch",
                    f"materialized pixels changed for ordinal {request.ordinal}",
                    ordinal=request.ordinal,
                )
    return issues


def _publish_task(
    *,
    output_root: Path,
    final_task: Path,
    requests: Sequence[ReviewRequest],
    recipe_digest: str,
    source_provenance: Mapping[str, Any],
    recipe_provenance: Mapping[str, Any],
    generator: GeneratorFn,
    strict_rendering: bool,
) -> list[VerificationIssue]:
    staging_root = output_root / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    stage_task = Path(
        tempfile.mkdtemp(prefix=f"{requests[0].task_id}-", dir=str(staging_root))
    )
    try:
        _, warnings = _build_task_stage(
            stage_task,
            output_root=output_root,
            requests=requests,
            recipe_digest=recipe_digest,
            source_provenance=source_provenance,
            recipe_provenance=recipe_provenance,
            generator=generator,
            strict_rendering=strict_rendering,
        )
        final_task.parent.mkdir(parents=True, exist_ok=True)
        if final_task.exists():
            raise MaterializationError(
                f"refusing to overwrite concurrently created task directory: {final_task}"
            )
        stage_task.rename(final_task)
        return warnings
    except Exception:
        shutil.rmtree(stage_task, ignore_errors=True)
        raise
    finally:
        try:
            staging_root.rmdir()
        except OSError:
            pass


def materialize_recipe(
    recipe_root: Path | str,
    output_root: Path | str,
    *,
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
    scene_ids: Sequence[str] | None = None,
    query_ids: Sequence[str] | None = None,
    generator: GeneratorFn | None = None,
    strict_rendering: bool = False,
) -> MaterializationReport:
    """Replay selected requests, staging and publishing each task atomically."""

    if query_ids:
        raise MaterializationError(
            "query filtering is not supported for materialization; select whole tasks "
            "and use query filters only with verify or audit"
        )
    manifest, all_requests = load_recipe(recipe_root)
    requests = filter_requests(
        all_requests,
        task_ids=task_ids,
        domains=domains,
        scene_ids=scene_ids,
        query_ids=None,
    )
    if not requests:
        raise MaterializationError("recipe filters selected no requests")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    generate = generator or _default_generator
    by_task: dict[str, list[ReviewRequest]] = defaultdict(list)
    for request in requests:
        by_task[request.task_id].append(request)
    all_by_task: dict[str, list[ReviewRequest]] = defaultdict(list)
    for request in all_requests:
        all_by_task[request.task_id].append(request)
    materialized: list[str] = []
    resumed: list[str] = []
    warnings: list[VerificationIssue] = []
    for task_id in sorted(by_task):
        task_requests = sorted(by_task[task_id], key=lambda row: row.ordinal)
        first = task_requests[0]
        final_task = root / first.domain / first.scene_id / first.task_id
        if final_task.exists():
            existing_issues = _validate_existing_task(
                final_task,
                output_root=root,
                requests=task_requests,
                manifest_requests=sorted(
                    all_by_task[task_id], key=lambda row: row.ordinal
                ),
                recipe_digest=manifest.recipe_digest,
                source_provenance=manifest.provenance.source.to_dict(),
                recipe_provenance=_compact_recipe_provenance(manifest),
            )
            if existing_issues:
                raise MaterializationError(
                    f"refusing to overwrite non-matching materialization for {task_id}: "
                    f"{existing_issues[0].message}"
                )
            resumed.append(task_id)
            continue
        warnings.extend(
            _publish_task(
                output_root=root,
                final_task=final_task,
                requests=task_requests,
                recipe_digest=manifest.recipe_digest,
                source_provenance=manifest.provenance.source.to_dict(),
                recipe_provenance=_compact_recipe_provenance(manifest),
                generator=generate,
                strict_rendering=strict_rendering,
            )
        )
        materialized.append(task_id)
    return MaterializationReport(
        selected_requests=len(requests),
        tasks_materialized=tuple(materialized),
        tasks_resumed=tuple(resumed),
        warnings=tuple(warnings),
    )


def verify_recipe(
    recipe_root: Path | str,
    *,
    output_root: Path | str | None = None,
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
    scene_ids: Sequence[str] | None = None,
    query_ids: Sequence[str] | None = None,
    generator: GeneratorFn | None = None,
    strict_rendering: bool = False,
) -> VerificationReport:
    """Regenerate selected requests; semantics are hard, rendering is host-native."""

    _, all_requests = load_recipe(recipe_root)
    requests = filter_requests(
        all_requests,
        task_ids=task_ids,
        domains=domains,
        scene_ids=scene_ids,
        query_ids=query_ids,
    )
    if not requests:
        return VerificationReport(
            checked_requests=0,
            issues=(
                VerificationIssue(
                    severity="error",
                    code="empty_selection",
                    message="recipe filters selected no requests",
                ),
            ),
        )
    generate = generator or _default_generator
    issues: list[VerificationIssue] = []
    for request in requests:
        try:
            output = generate(
                request.task_id,
                request.seed,
                dict(request.params),
                request.max_attempts,
            )
            observed_query = resolve_output_query_id(output)
            prepared = prepare_output(output, task_id=request.task_id)
        except Exception as exc:
            issues.append(
                VerificationIssue(
                    severity="error",
                    code="generation_failed",
                    message=f"generation failed: {type(exc).__name__}: {exc}",
                    task_id=request.task_id,
                    ordinal=request.ordinal,
                )
            )
            continue
        if observed_query != request.query_id:
            issues.append(
                VerificationIssue(
                    severity="error",
                    code="query_id_mismatch",
                    message=(
                        f"observed query {observed_query!r}, expected "
                        f"{request.query_id!r}"
                    ),
                    task_id=request.task_id,
                    ordinal=request.ordinal,
                )
            )
        if prepared.hashes.semantic_hash != request.hashes.semantic_hash:
            issues.append(
                VerificationIssue(
                    severity="error",
                    code="semantic_hash_mismatch",
                    message=(
                        f"expected {request.hashes.semantic_hash}, observed "
                        f"{prepared.hashes.semantic_hash}"
                    ),
                    task_id=request.task_id,
                    ordinal=request.ordinal,
                )
            )
        issues.extend(
            _rendering_issues(
                request=request,
                observed=prepared.hashes,
                strict_rendering=strict_rendering,
            )
        )
    checked = len(requests)
    if output_root is not None:
        materialized = verify_materialized(
            recipe_root,
            output_root,
            task_ids=task_ids,
            domains=domains,
            scene_ids=scene_ids,
            query_ids=query_ids,
        )
        checked += materialized.checked_requests
        issues.extend(materialized.issues)
    return VerificationReport(checked_requests=checked, issues=tuple(issues))


def verify_materialized(
    recipe_root: Path | str,
    output_root: Path | str,
    *,
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
    scene_ids: Sequence[str] | None = None,
    query_ids: Sequence[str] | None = None,
) -> VerificationReport:
    """Verify stored JSON/PNG integrity without regenerating task outputs."""

    manifest, all_requests = load_recipe(recipe_root)
    requests = filter_requests(
        all_requests,
        task_ids=task_ids,
        domains=domains,
        scene_ids=scene_ids,
        query_ids=query_ids,
    )
    if not requests:
        return VerificationReport(
            checked_requests=0,
            issues=(
                VerificationIssue(
                    severity="error",
                    code="empty_selection",
                    message="recipe filters selected no requests",
                ),
            ),
        )
    root = Path(output_root)
    by_task: dict[str, list[ReviewRequest]] = defaultdict(list)
    for request in requests:
        by_task[request.task_id].append(request)
    issues: list[VerificationIssue] = []
    all_by_task: dict[str, list[ReviewRequest]] = defaultdict(list)
    for request in all_requests:
        all_by_task[request.task_id].append(request)
    for task_id in sorted(by_task):
        task_requests = sorted(by_task[task_id], key=lambda row: row.ordinal)
        first = task_requests[0]
        task_dir = root / first.domain / first.scene_id / first.task_id
        if not task_dir.is_dir():
            issues.append(
                VerificationIssue(
                    severity="error",
                    code="missing_task_materialization",
                    message=f"missing materialized task directory: {task_dir}",
                    task_id=task_id,
                )
            )
            continue
        issues.extend(
            _validate_existing_task(
                task_dir,
                output_root=root,
                requests=task_requests,
                manifest_requests=sorted(
                    all_by_task[task_id], key=lambda row: row.ordinal
                ),
                recipe_digest=manifest.recipe_digest,
                source_provenance=manifest.provenance.source.to_dict(),
                recipe_provenance=_compact_recipe_provenance(manifest),
            )
        )
    return VerificationReport(checked_requests=len(requests), issues=tuple(issues))


def iter_materialized_records(output_root: Path | str) -> Iterable[dict[str, Any]]:
    """Yield artifact JSON objects from the canonical materialized layout."""

    root = Path(output_root)
    if not root.is_dir():
        return
    for path in sorted(root.glob("*/*/task_*/data/*/*.json")):
        try:
            resolved = _resolved_child(root, path.relative_to(root).as_posix())
            yield _load_json(resolved)
        except MaterializationError:
            continue


__all__ = [
    "iter_materialized_records",
    "materialize_recipe",
    "verify_materialized",
    "verify_recipe",
]
