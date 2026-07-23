"""Portable source, resource, and runtime provenance for review recipes."""

from __future__ import annotations

import hashlib
from importlib import metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Iterable, Mapping, Sequence

from trace_tasks.core.canonical import canonical_json_bytes

from .models import (
    RecipeCaptureError,
    ResourceProvenance,
    ReviewProvenance,
    RuntimeProvenance,
    SourceProvenance,
)

_DEPENDENCY_DISTRIBUTIONS = (
    "Pillow",
    "CairoSVG",
    "numpy",
    "scipy",
    "networkx",
    "PyYAML",
    "rfc8785",
    "blake3",
)
_IGNORED_PARTS = {"__pycache__", ".git", ".mypy_cache", ".pytest_cache"}
_IGNORED_SUFFIXES = {".pyc", ".pyo"}
_PILLOW_NATIVE_FEATURES = (
    "freetype2",
    "littlecms2",
    "libjpeg_turbo",
    "jpg_2000",
    "zlib",
    "libtiff",
    "webp",
    "raqm",
)
_GENERATION_CONSTRAINT_DISTRIBUTIONS = frozenset(
    {
        "pillow",
        "cairosvg",
        "pyyaml",
        "numpy",
        "scipy",
        "networkx",
        "rfc8785",
        "blake3",
        "cairocffi",
        "cffi",
        "pycparser",
        "cssselect2",
        "tinycss2",
        "webencodings",
        "defusedxml",
    }
)


def sha256_bytes(data: bytes) -> str:
    """Return a self-describing SHA-256 digest."""

    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def sha256_file(path: Path | str) -> str:
    """Hash one file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _tree_entries(
    root: Path, *, suffixes: set[str] | None = None
) -> Iterable[tuple[str, str]]:
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root)
        if any(part in _IGNORED_PARTS for part in relative.parts):
            continue
        if path.is_dir():
            continue
        if path.suffix in _IGNORED_SUFFIXES:
            continue
        if suffixes is not None and path.suffix not in suffixes:
            continue
        relative_text = relative.as_posix()
        if path.is_symlink():
            yield relative_text, sha256_bytes(os.readlink(path).encode("utf-8"))
        elif path.is_file():
            yield relative_text, sha256_file(path)


def tree_sha256(
    roots: Sequence[tuple[str, Path]],
    *,
    suffixes: set[str] | None = None,
) -> str:
    """Hash named directory trees using relative paths and content identities."""

    entries: list[dict[str, str]] = []
    for label, root in sorted(roots, key=lambda item: item[0]):
        if root.is_file() or root.is_symlink():
            if suffixes is not None and root.suffix not in suffixes:
                continue
            digest = (
                sha256_bytes(os.readlink(root).encode("utf-8"))
                if root.is_symlink()
                else sha256_file(root)
            )
            entries.append({"path": label, "sha256": digest})
            continue
        entries.extend(
            {"path": f"{label}/{relative}", "sha256": digest}
            for relative, digest in _tree_entries(root, suffixes=suffixes)
        )
    return sha256_bytes(canonical_json_bytes(entries))


def generation_constraints_sha256(path: Path | str) -> str:
    """Hash only release pins that can affect task generation or rendering."""

    constraint_path = Path(path)
    pins: list[str] = []
    if constraint_path.is_file():
        for raw_line in constraint_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line or line.startswith(("-", "--")):
                continue
            name_chars: list[str] = []
            for character in line:
                if character.isalnum() or character in "-_.":
                    name_chars.append(character)
                else:
                    break
            normalized = "".join(name_chars).lower().replace("_", "-")
            if normalized in _GENERATION_CONSTRAINT_DISTRIBUTIONS:
                pins.append(line)
    return sha256_bytes(canonical_json_bytes(sorted(set(pins))))


def _git_output(repo_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return ""
    return completed.stdout.strip()


def source_revision(repo_root: Path | str) -> str:
    """Return the checked-out Git object id, or ``unknown`` outside a checkout."""

    return _git_output(Path(repo_root), "rev-parse", "HEAD") or "unknown"


def source_is_dirty(repo_root: Path | str) -> bool:
    """Return whether tracked or untracked checkout state differs from HEAD."""

    status = _git_output(
        Path(repo_root),
        "status",
        "--porcelain=v1",
        "--untracked-files=normal",
    )
    return bool(status)


def runtime_provenance() -> RuntimeProvenance:
    """Collect portable runtime versions without machine paths or host names."""

    dependencies: dict[str, str] = {}
    for distribution in _DEPENDENCY_DISTRIBUTIONS:
        try:
            dependencies[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            dependencies[distribution] = "not-installed"
    native_libraries: dict[str, str] = {}
    try:
        import cairocffi

        native_libraries["libcairo"] = str(cairocffi.cairo_version_string())
    except Exception:
        native_libraries["libcairo"] = "unavailable"
    try:
        from PIL import features

        for feature in _PILLOW_NATIVE_FEATURES:
            try:
                version = features.version(feature)
            except Exception:
                version = None
            native_libraries[f"pillow:{feature}"] = str(version or "unavailable")
    except Exception:
        for feature in _PILLOW_NATIVE_FEATURES:
            native_libraries[f"pillow:{feature}"] = "unavailable"
    libc_name, libc_version = platform.libc_ver()
    native_libraries["libc"] = (
        f"{libc_name}:{libc_version}" if libc_name or libc_version else "unavailable"
    )
    return RuntimeProvenance(
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        platform=platform.system() or sys.platform,
        machine=platform.machine() or "unknown",
        dependencies=dependencies,
        native_libraries=native_libraries,
    )


def collect_review_provenance(
    *,
    repo_root: Path | str,
    task_query_ids: Mapping[str, Sequence[str]],
    source_repository: str = "maveryn/trace",
    require_clean_source: bool = True,
) -> ReviewProvenance:
    """Collect the frozen provenance needed to reproduce a review recipe."""

    root = Path(repo_root).resolve()
    package_root = root / "src" / "trace_tasks"
    if not package_root.is_dir():
        raise RecipeCaptureError(
            "repo_root must contain the public src/trace_tasks package"
        )
    revision = source_revision(root)
    dirty = source_is_dirty(root)
    if require_clean_source and revision == "unknown":
        raise RecipeCaptureError("canonical capture requires a Git source revision")
    if require_clean_source and dirty:
        raise RecipeCaptureError(
            "canonical capture requires a clean source checkout; commit or remove local changes"
        )

    resources_root = package_root / "resources"
    prompts_root = resources_root / "prompts"
    release_constraints = root / "constraints" / "release.txt"
    source_hash = tree_sha256((("src/trace_tasks", package_root),))
    generator_hash = tree_sha256(
        (
            ("src/trace_tasks/configs", package_root / "configs"),
            ("src/trace_tasks/core", package_root / "core"),
            ("src/trace_tasks/tasks", package_root / "tasks"),
            (
                "src/trace_tasks/review/models.py",
                package_root / "review" / "models.py",
            ),
            (
                "src/trace_tasks/review/provenance.py",
                package_root / "review" / "provenance.py",
            ),
            (
                "src/trace_tasks/review/recipe.py",
                package_root / "review" / "recipe.py",
            ),
        )
    )
    constraints_hash = generation_constraints_sha256(release_constraints)
    resource_hash = tree_sha256((("resources", resources_root),))
    prompt_hash = tree_sha256((("prompts", prompts_root),))
    catalog = [
        {
            "task_id": str(task_id),
            "query_ids": sorted({str(query_id) for query_id in query_ids}),
        }
        for task_id, query_ids in sorted(task_query_ids.items())
    ]
    catalog_hash = sha256_bytes(canonical_json_bytes(catalog))
    return ReviewProvenance(
        source=SourceProvenance(
            repository=str(source_repository),
            revision=revision,
            dirty=dirty,
            source_tree_hash=source_hash,
            generator_tree_hash=generator_hash,
            constraints_hash=constraints_hash,
        ),
        resources=ResourceProvenance(
            resource_tree_hash=resource_hash,
            prompt_bundle_tree_hash=prompt_hash,
            task_catalog_hash=catalog_hash,
        ),
        runtime=runtime_provenance(),
    )


def default_repo_root() -> Path:
    """Return the checkout root for a source install of this package."""

    return Path(__file__).resolve().parents[3]


__all__ = [
    "collect_review_provenance",
    "default_repo_root",
    "generation_constraints_sha256",
    "runtime_provenance",
    "sha256_bytes",
    "sha256_file",
    "source_is_dirty",
    "source_revision",
    "tree_sha256",
]
