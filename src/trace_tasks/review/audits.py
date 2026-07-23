"""Focused offline audits for public contributor-review artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from trace_tasks.core.reward_contracts import (
    resolve_reward_contract,
    validate_reward_contract_payload,
)

from .materialize import iter_materialized_records, verify_materialized
from .models import (
    ARTIFACT_SCHEMA_VERSION,
    REQUESTS_PER_TASK,
    AuditIssue,
    AuditReport,
)
from .recipe import load_recipe, resolve_output_query_id

_REQUIRED_TRACE_KEYS = (
    "scene_ir",
    "query_spec",
    "render_spec",
    "render_map",
    "execution_trace",
    "witness_symbolic",
    "projected_annotation",
)


def _issue(
    severity: str,
    category: str,
    code: str,
    message: str,
    *,
    task_id: str = "",
    path: str = "",
) -> AuditIssue:
    return AuditIssue(
        severity=severity,
        category=category,
        code=code,
        message=message,
        task_id=task_id,
        path=path,
    )


def audit_recipe(recipe_root: Path | str) -> AuditReport:
    """Audit canonical count, stratification, provenance, and request controls."""

    try:
        manifest, requests = load_recipe(recipe_root)
    except Exception as exc:
        return AuditReport(
            checked=0,
            issues=(
                _issue(
                    "error",
                    "recipe",
                    "invalid_recipe",
                    f"{type(exc).__name__}: {exc}",
                    path=str(Path(recipe_root) / "manifest.json"),
                ),
            ),
        )
    issues: list[AuditIssue] = []
    if manifest.provenance.source.dirty:
        issues.append(
            _issue(
                "error",
                "provenance",
                "dirty_source_capture",
                "canonical recipes must be captured from a clean source revision",
            )
        )
    if manifest.provenance.source.revision == "unknown":
        issues.append(
            _issue(
                "error",
                "provenance",
                "unknown_source_revision",
                "canonical recipes must name their producer revision",
            )
        )
    by_task = defaultdict(list)
    for request in requests:
        by_task[request.task_id].append(request)
        if request.params.get("query_id") != request.query_id:
            issues.append(
                _issue(
                    "error",
                    "recipe",
                    "query_param_mismatch",
                    "stored caller params do not target the declared query id",
                    task_id=request.task_id,
                )
            )
        if request.params.get("_sample_cursor") != request.sample_cursor:
            issues.append(
                _issue(
                    "error",
                    "recipe",
                    "sample_cursor_param_mismatch",
                    "stored caller params do not contain the declared sample cursor",
                    task_id=request.task_id,
                )
            )
    for task_id, task_requests in sorted(by_task.items()):
        if len(task_requests) != REQUESTS_PER_TASK:
            issues.append(
                _issue(
                    "error",
                    "coverage",
                    "request_count_mismatch",
                    f"expected {REQUESTS_PER_TASK} requests, observed {len(task_requests)}",
                    task_id=task_id,
                )
            )
        query_counts = Counter(request.query_id for request in task_requests)
        if not query_counts:
            issues.append(
                _issue(
                    "error",
                    "coverage",
                    "missing_query_coverage",
                    "task has no query-stratified requests",
                    task_id=task_id,
                )
            )
        elif max(query_counts.values()) - min(query_counts.values()) > 1:
            issues.append(
                _issue(
                    "error",
                    "coverage",
                    "unbalanced_query_stratification",
                    "query request counts must differ by at most one",
                    task_id=task_id,
                )
            )
        ordered_queries = [
            request.query_id
            for request in sorted(task_requests, key=lambda row: row.ordinal)
        ]
        unique_queries = sorted(query_counts)
        expected = [
            unique_queries[index % len(unique_queries)]
            for index in range(len(ordered_queries))
        ]
        if ordered_queries != expected:
            issues.append(
                _issue(
                    "error",
                    "coverage",
                    "noncanonical_query_order",
                    "query ids must use sorted round-robin order",
                    task_id=task_id,
                )
            )
    return AuditReport(checked=len(requests), issues=tuple(issues))


def audit_generated_output(output: Any, *, task_id: str = "") -> AuditReport:
    """Run portable prompt, answer, annotation, trace, and rendering checks."""

    issues: list[AuditIssue] = []
    prompt = getattr(output, "prompt", None)
    if not isinstance(prompt, str) or not prompt.strip():
        issues.append(
            _issue(
                "error",
                "prompt",
                "empty_prompt",
                "generated prompt must be a non-empty string",
                task_id=task_id,
            )
        )
    prompt_variants = getattr(output, "prompt_variants", {})
    if not isinstance(prompt_variants, Mapping):
        issues.append(
            _issue(
                "error",
                "prompt",
                "invalid_prompt_variants",
                "prompt variants must be an object",
                task_id=task_id,
            )
        )
    for name, category in (("answer_gt", "answer"), ("annotation_gt", "annotation")):
        typed = getattr(output, name, None)
        if (
            not isinstance(getattr(typed, "type", None), str)
            or not str(getattr(typed, "type", "")).strip()
        ):
            issues.append(
                _issue(
                    "error",
                    category,
                    f"invalid_{name}_type",
                    f"{name} must declare a non-empty public type",
                    task_id=task_id,
                )
            )
        if getattr(typed, "value", None) is None:
            issues.append(
                _issue(
                    "error",
                    category,
                    f"missing_{name}_value",
                    f"{name} must contain a value",
                    task_id=task_id,
                )
            )
    try:
        resolve_reward_contract(
            answer_type=str(getattr(getattr(output, "answer_gt", None), "type", "")),
            annotation_type=str(
                getattr(getattr(output, "annotation_gt", None), "type", "")
            ),
        )
    except ValueError as exc:
        issues.append(
            _issue(
                "error",
                "verifier_trace",
                "invalid_reward_contract_types",
                str(exc),
                task_id=task_id,
            )
        )
    trace = getattr(output, "trace_payload", None)
    if not isinstance(trace, Mapping):
        issues.append(
            _issue(
                "error",
                "verifier_trace",
                "invalid_trace_payload",
                "trace payload must be an object",
                task_id=task_id,
            )
        )
    else:
        missing = [key for key in _REQUIRED_TRACE_KEYS if key not in trace]
        if missing:
            issues.append(
                _issue(
                    "error",
                    "verifier_trace",
                    "missing_trace_fields",
                    "trace payload is missing: " + ", ".join(missing),
                    task_id=task_id,
                )
            )
    image = getattr(output, "image", None)
    if not isinstance(image, Image.Image) or image.width < 1 or image.height < 1:
        issues.append(
            _issue(
                "error",
                "rendering",
                "invalid_image",
                "generated output must contain a non-empty PIL image",
                task_id=task_id,
            )
        )
    if not resolve_output_query_id(output):
        issues.append(
            _issue(
                "error",
                "taxonomy",
                "missing_query_id",
                "generated output must resolve to a public query id",
                task_id=task_id,
            )
        )
    return AuditReport(checked=1, issues=tuple(issues))


def _audit_materialized_record(record: Mapping[str, Any]) -> list[AuditIssue]:
    task_id = str(record.get("task_id", ""))
    issues: list[AuditIssue] = []
    if record.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        issues.append(
            _issue(
                "error",
                "artifact",
                "artifact_schema_mismatch",
                f"expected {ARTIFACT_SCHEMA_VERSION}",
                task_id=task_id,
            )
        )
    if (
        not isinstance(record.get("prompt"), str)
        or not str(record.get("prompt", "")).strip()
    ):
        issues.append(
            _issue(
                "error",
                "prompt",
                "empty_materialized_prompt",
                "materialized prompt must be non-empty",
                task_id=task_id,
            )
        )
    for name, category in (("answer_gt", "answer"), ("annotation_gt", "annotation")):
        typed = record.get(name)
        if (
            not isinstance(typed, Mapping)
            or not str(typed.get("type", "")).strip()
            or "value" not in typed
        ):
            issues.append(
                _issue(
                    "error",
                    category,
                    f"invalid_materialized_{name}",
                    f"materialized {name} must be a typed value",
                    task_id=task_id,
                )
            )
    answer_gt = record.get("answer_gt")
    annotation_gt = record.get("annotation_gt")
    reward_error = validate_reward_contract_payload(
        record.get("reward_contract", {}),
        answer_type=(
            str(answer_gt.get("type", "")) if isinstance(answer_gt, Mapping) else None
        ),
        annotation_type=(
            str(annotation_gt.get("type", ""))
            if isinstance(annotation_gt, Mapping)
            else None
        ),
    )
    if reward_error:
        issues.append(
            _issue(
                "error",
                "verifier_trace",
                "invalid_materialized_reward_contract",
                reward_error,
                task_id=task_id,
            )
        )
    expected_taxonomy = {
        "domain": str(record.get("domain", "")),
        "scene_id": str(record.get("scene_id", "")),
        "task_id": task_id,
        "query_id": str(record.get("query_id", "")),
    }
    if record.get("taxonomy") != expected_taxonomy:
        issues.append(
            _issue(
                "error",
                "taxonomy",
                "materialized_taxonomy_mismatch",
                "artifact taxonomy must match its domain/scene_id/task_id/query_id fields",
                task_id=task_id,
            )
        )
    trace = record.get("trace_payload")
    if not isinstance(trace, Mapping):
        issues.append(
            _issue(
                "error",
                "verifier_trace",
                "invalid_materialized_trace",
                "materialized trace payload must be an object",
                task_id=task_id,
            )
        )
    else:
        missing = [key for key in _REQUIRED_TRACE_KEYS if key not in trace]
        if missing:
            issues.append(
                _issue(
                    "error",
                    "verifier_trace",
                    "missing_materialized_trace_fields",
                    "materialized trace is missing: " + ", ".join(missing),
                    task_id=task_id,
                )
            )
        trace_taxonomy = trace.get("taxonomy")
        public_taxonomy = (
            trace_taxonomy.get("public")
            if isinstance(trace_taxonomy, Mapping)
            else None
        )
        if public_taxonomy != expected_taxonomy:
            issues.append(
                _issue(
                    "error",
                    "taxonomy",
                    "trace_taxonomy_mismatch",
                    "trace public taxonomy must match the artifact taxonomy",
                    task_id=task_id,
                )
            )
    return issues


def audit_materialized(
    recipe_root: Path | str,
    output_root: Path | str,
    *,
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
    scene_ids: Sequence[str] | None = None,
    query_ids: Sequence[str] | None = None,
) -> AuditReport:
    """Audit file integrity plus contributor-facing artifact contracts."""

    verification = verify_materialized(
        recipe_root,
        output_root,
        task_ids=task_ids,
        domains=domains,
        scene_ids=scene_ids,
        query_ids=query_ids,
    )
    issues = [
        _issue(
            item.severity,
            "artifact",
            item.code,
            item.message,
            task_id=item.task_id,
        )
        for item in verification.issues
    ]
    selected_tasks = {str(value) for value in task_ids or ()}
    selected_domains = {str(value) for value in domains or ()}
    selected_queries = {str(value) for value in query_ids or ()}
    bare_scenes = {str(value) for value in scene_ids or () if "/" not in str(value)}
    qualified_scenes = {
        tuple(str(value).split("/", 1))
        for value in scene_ids or ()
        if "/" in str(value)
    }
    checked_records = 0
    for record in iter_materialized_records(output_root):
        if selected_tasks and str(record.get("task_id", "")) not in selected_tasks:
            continue
        if selected_domains and str(record.get("domain", "")) not in selected_domains:
            continue
        record_scene = str(record.get("scene_id", ""))
        record_domain = str(record.get("domain", ""))
        if (
            (bare_scenes or qualified_scenes)
            and record_scene not in bare_scenes
            and (
                record_domain,
                record_scene,
            )
            not in qualified_scenes
        ):
            continue
        if selected_queries and str(record.get("query_id", "")) not in selected_queries:
            continue
        checked_records += 1
        issues.extend(_audit_materialized_record(record))
    return AuditReport(
        checked=max(verification.checked_requests, checked_records),
        issues=tuple(issues),
    )


__all__ = [
    "audit_generated_output",
    "audit_materialized",
    "audit_recipe",
]
