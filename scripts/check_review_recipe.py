#!/usr/bin/env python3
"""Validate the checked-in canonical contributor-review recipe."""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tokenize
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
DEFAULT_RECIPE_ROOT = (
    REPO_ROOT / "docs" / "review" / "recipes" / "trace-review-recipe-v1"
)
EXPECTED_REPOSITORY = "maveryn/trace"
EXPECTED_REQUESTS_PER_TASK = 25
_REVISION_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
# Mirror the generator-hash roots in review/provenance.py. That module is part
# of the hashed surface, so a future scope change fails this fallback closed.
_GENERATOR_DIRECTORY_PATHS = (
    Path("src/trace_tasks/configs"),
    Path("src/trace_tasks/core"),
    Path("src/trace_tasks/tasks"),
)
_GENERATOR_FILE_PATHS = (
    Path("src/trace_tasks/review/models.py"),
    Path("src/trace_tasks/review/provenance.py"),
    Path("src/trace_tasks/review/recipe.py"),
)
_GENERATOR_PATHS = _GENERATOR_DIRECTORY_PATHS + _GENERATOR_FILE_PATHS
_IGNORED_PARTS = {"__pycache__", ".git", ".mypy_cache", ".pytest_cache"}
_IGNORED_SUFFIXES = {".pyc", ".pyo"}


class ReviewRecipeCheckError(RuntimeError):
    """Raised when the canonical review recipe is stale or incomplete."""


class _DocstringStripper(ast.NodeTransformer):
    """Remove Python docstrings while preserving executable string expressions."""

    @staticmethod
    def _strip_leading_docstring(node: ast.AST) -> ast.AST:
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            return node
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            del body[0]
        return node

    def visit_Module(self, node: ast.Module) -> ast.AST:  # noqa: N802
        self.generic_visit(node)
        return self._strip_leading_docstring(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:  # noqa: N802
        self.generic_visit(node)
        return self._strip_leading_docstring(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:  # noqa: N802
        self.generic_visit(node)
        return self._strip_leading_docstring(node)

    def visit_AsyncFunctionDef(
        self, node: ast.AsyncFunctionDef
    ) -> ast.AST:  # noqa: N802
        self.generic_visit(node)
        return self._strip_leading_docstring(node)


def _git(
    repo_root: Path,
    *args: str,
    timeout: int = 30,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=False,
            capture_output=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise ReviewRecipeCheckError(
            f"unable to inspect Git history for review-recipe freshness: {exc}"
        ) from exc


def _include_generator_path(path: Path) -> bool:
    return not any(part in _IGNORED_PARTS for part in path.parts) and (
        path.suffix not in _IGNORED_SUFFIXES
    )


def _current_generator_paths(repo_root: Path) -> set[str]:
    paths: set[str] = set()
    for relative_root in _GENERATOR_DIRECTORY_PATHS:
        root = repo_root / relative_root
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            relative = path.relative_to(repo_root)
            if not _include_generator_path(relative):
                continue
            if path.is_file() or path.is_symlink():
                paths.add(relative.as_posix())
    for relative in _GENERATOR_FILE_PATHS:
        path = repo_root / relative
        if _include_generator_path(relative) and (path.is_file() or path.is_symlink()):
            paths.add(relative.as_posix())
    return paths


def _recorded_generator_paths(repo_root: Path, revision: str) -> set[str]:
    completed = _git(
        repo_root,
        "ls-tree",
        "-r",
        "--name-only",
        "-z",
        revision,
        "--",
        *(path.as_posix() for path in _GENERATOR_PATHS),
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ReviewRecipeCheckError(
            "canonical recipe producer revision is unavailable; fetch the full "
            f"Git history before running the check ({detail or revision})"
        )
    return {
        text
        for raw in completed.stdout.split(b"\0")
        if raw
        for text in [raw.decode("utf-8")]
        if _include_generator_path(Path(text))
    }


def _current_path_bytes(path: Path) -> bytes:
    if path.is_symlink():
        return os.fsencode(os.readlink(path))
    return path.read_bytes()


def _recorded_path_bytes(repo_root: Path, revision: str, relative: str) -> bytes:
    completed = _git(repo_root, "show", f"{revision}:{relative}")
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ReviewRecipeCheckError(
            f"unable to read {relative!r} from recipe producer revision: "
            f"{detail or revision}"
        )
    return completed.stdout


def _recorded_path_mode(repo_root: Path, revision: str, relative: str) -> str:
    completed = _git(repo_root, "ls-tree", revision, "--", relative)
    if completed.returncode != 0 or not completed.stdout:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ReviewRecipeCheckError(
            f"unable to inspect {relative!r} in the recipe producer revision: "
            f"{detail or revision}"
        )
    return completed.stdout.split(maxsplit=1)[0].decode("ascii", errors="strict")


def _changed_generator_paths(repo_root: Path, revision: str) -> tuple[str, ...]:
    completed = _git(
        repo_root,
        "diff",
        "--name-only",
        "--no-ext-diff",
        "--no-renames",
        "-z",
        revision,
        "--",
        *(path.as_posix() for path in _GENERATOR_PATHS),
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ReviewRecipeCheckError(
            "unable to compare current generator files with the recipe producer "
            f"revision: {detail or revision}"
        )
    return tuple(raw.decode("utf-8") for raw in completed.stdout.split(b"\0") if raw)


def _python_executable_ast(source: bytes, *, filename: str) -> str | None:
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(source).readline)
        tree = ast.parse(source.decode(encoding), filename=filename)
    except (LookupError, SyntaxError, UnicodeDecodeError):
        return None
    stripped = _DocstringStripper().visit(tree)
    return ast.dump(stripped, annotate_fields=True, include_attributes=False)


def _non_executable_generator_drift_paths(
    repo_root: Path,
    *,
    producer_revision: str,
) -> tuple[str, ...] | None:
    """Return changed paths only when drift is Python presentation text only."""

    revision_check = _git(
        repo_root, "cat-file", "-e", f"{producer_revision}^{{commit}}"
    )
    if revision_check.returncode != 0:
        raise ReviewRecipeCheckError(
            "canonical recipe producer revision is unavailable; fetch the full "
            f"Git history before running the check ({producer_revision})"
        )

    recorded_paths = _recorded_generator_paths(repo_root, producer_revision)
    current_paths = _current_generator_paths(repo_root)
    if recorded_paths != current_paths:
        return None

    changed: list[str] = []
    for relative in _changed_generator_paths(repo_root, producer_revision):
        current_path = repo_root / relative
        if (
            current_path.is_symlink()
            or _recorded_path_mode(repo_root, producer_revision, relative) == "120000"
        ):
            return None
        recorded = _recorded_path_bytes(repo_root, producer_revision, relative)
        current = _current_path_bytes(current_path)
        changed.append(relative)
        if Path(relative).suffix != ".py":
            return None
        recorded_ast = _python_executable_ast(recorded, filename=relative)
        current_ast = _python_executable_ast(current, filename=relative)
        if recorded_ast is None or recorded_ast != current_ast:
            return None
    return tuple(changed) or None


def _add_source_tree_to_path() -> None:
    source = str(SOURCE_ROOT)
    if source not in sys.path:
        sys.path.insert(0, source)


def _query_ids_by_task(task_ids: Sequence[str]) -> dict[str, tuple[str, ...]]:
    from trace_tasks.core.query_ids import SINGLE_QUERY_ID
    from trace_tasks.tasks.registry import create_task

    result: dict[str, tuple[str, ...]] = {}
    for task_id in task_ids:
        task = create_task(task_id)
        raw = getattr(task, "supported_query_ids", ()) or ()
        if isinstance(raw, str):
            raw = (raw,)
        query_ids = tuple(sorted({str(value) for value in raw if str(value).strip()}))
        result[task_id] = query_ids or (SINGLE_QUERY_ID,)
    return result


def check_review_recipe(recipe_root: Path = DEFAULT_RECIPE_ROOT) -> dict[str, object]:
    """Validate coverage, provenance, hashes, and current source compatibility."""

    _add_source_tree_to_path()
    from trace_tasks.review import audit_recipe, load_recipe
    from trace_tasks.review.provenance import collect_review_provenance
    from trace_tasks.tasks.registry import list_default_task_ids

    root = Path(recipe_root).resolve()
    manifest, requests = load_recipe(root)
    audit = audit_recipe(root)
    if not audit.ok:
        details = "; ".join(
            f"{issue.code}: {issue.message}" for issue in audit.errors[:10]
        )
        raise ReviewRecipeCheckError(f"recipe audit failed: {details}")

    expected_tasks = list_default_task_ids()
    expected_task_set = set(expected_tasks)
    observed_task_set = {request.task_id for request in requests}
    missing = sorted(expected_task_set - observed_task_set)
    extra = sorted(observed_task_set - expected_task_set)
    if missing or extra:
        raise ReviewRecipeCheckError(
            "recipe task coverage differs from the active public registry: "
            f"missing={missing[:10]!r}, extra={extra[:10]!r}"
        )
    expected_request_count = len(expected_tasks) * EXPECTED_REQUESTS_PER_TASK
    if manifest.task_count != len(expected_tasks):
        raise ReviewRecipeCheckError(
            f"expected {len(expected_tasks)} tasks, found {manifest.task_count}"
        )
    if manifest.request_count != expected_request_count:
        raise ReviewRecipeCheckError(
            f"expected {expected_request_count} requests, found {manifest.request_count}"
        )

    expected_queries = _query_ids_by_task(expected_tasks)
    observed_queries: dict[str, set[str]] = defaultdict(set)
    for request in requests:
        observed_queries[request.task_id].add(request.query_id)
    mismatched_queries = [
        task_id
        for task_id in expected_tasks
        if observed_queries[task_id] != set(expected_queries[task_id])
    ]
    if mismatched_queries:
        task_id = mismatched_queries[0]
        raise ReviewRecipeCheckError(
            f"{task_id} query coverage is stale: expected "
            f"{sorted(expected_queries[task_id])!r}, observed "
            f"{sorted(observed_queries[task_id])!r}"
        )

    source = manifest.provenance.source
    if source.repository != EXPECTED_REPOSITORY:
        raise ReviewRecipeCheckError(
            f"expected source repository {EXPECTED_REPOSITORY!r}, found "
            f"{source.repository!r}"
        )
    if source.dirty:
        raise ReviewRecipeCheckError("canonical recipe records a dirty source checkout")
    if _REVISION_RE.fullmatch(source.revision) is None:
        raise ReviewRecipeCheckError(
            f"canonical recipe has an invalid producer revision: {source.revision!r}"
        )

    current = collect_review_provenance(
        repo_root=REPO_ROOT,
        task_query_ids=expected_queries,
        source_repository=EXPECTED_REPOSITORY,
        require_clean_source=False,
    )
    non_executable_generator_drift: tuple[str, ...] = ()
    if source.generator_tree_hash != current.source.generator_tree_hash:
        accepted_paths = _non_executable_generator_drift_paths(
            REPO_ROOT,
            producer_revision=source.revision,
        )
        if accepted_paths is not None:
            non_executable_generator_drift = accepted_paths
    comparisons = {
        "generator_tree_hash": (
            source.generator_tree_hash,
            current.source.generator_tree_hash,
        ),
        "constraints_hash": (source.constraints_hash, current.source.constraints_hash),
        "resource_tree_hash": (
            manifest.provenance.resources.resource_tree_hash,
            current.resources.resource_tree_hash,
        ),
        "prompt_bundle_tree_hash": (
            manifest.provenance.resources.prompt_bundle_tree_hash,
            current.resources.prompt_bundle_tree_hash,
        ),
        "task_catalog_hash": (
            manifest.provenance.resources.task_catalog_hash,
            current.resources.task_catalog_hash,
        ),
    }
    stale = [
        name
        for name, (recorded, observed) in comparisons.items()
        if recorded != observed
        and not (name == "generator_tree_hash" and non_executable_generator_drift)
    ]
    if stale:
        raise ReviewRecipeCheckError(
            "canonical recipe is stale relative to the current checkout: "
            + ", ".join(stale)
        )

    required_native_versions = {
        "libc",
        "libcairo",
        "pillow:freetype2",
        "pillow:zlib",
    }
    native_versions = manifest.provenance.runtime.native_libraries
    missing_native_versions = sorted(required_native_versions - set(native_versions))
    if missing_native_versions:
        raise ReviewRecipeCheckError(
            "recipe runtime provenance is missing native version fields: "
            + ", ".join(missing_native_versions)
        )

    return {
        "recipe_id": manifest.recipe_id,
        "recipe_digest": manifest.recipe_digest,
        "producer_revision": source.revision,
        "task_count": manifest.task_count,
        "request_count": manifest.request_count,
        "shard_count": len(manifest.shards),
        "non_executable_generator_drift": list(non_executable_generator_drift),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE_ROOT)
    parser.add_argument("--json", action="store_true", help="Print a JSON summary")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        summary = check_review_recipe(args.recipe)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"review recipe check failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        drift = summary["non_executable_generator_drift"]
        drift_note = (
            f", accepted non-executable drift in {len(drift)} Python files"
            if isinstance(drift, list) and drift
            else ""
        )
        print(
            "review recipe check passed: "
            f"{summary['task_count']} tasks, {summary['request_count']} requests, "
            f"{summary['recipe_digest']}{drift_note}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
