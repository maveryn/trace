#!/usr/bin/env python3
"""Run the portable, public Trace release checks.

The default mode verifies the source release contracts, builds an sdist and a
wheel, installs the wheel into a temporary virtual environment, and exercises
the installed command-line interfaces.  ``--source-only`` skips packaging and
installation so canonical source contracts can run independently of package
compatibility jobs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import tempfile
from typing import Iterable, Mapping, Sequence
import venv
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
# These values are the public 0.1 registry contract. A deliberate registry
# change must update both the generated inventory and this release assertion.
EXPECTED_DEFAULT_TASK_COUNT = 1_000
EXPECTED_REGISTERED_TASK_COUNT = 1_000
EXPECTED_SCENE_COUNT = 277
EXPECTED_RELEASE_TASK_IDS_SHA256 = (
    "1ef0419e23368309a961dd596d0cbc7212cdee64625d98d78cf1f6e654e47a98"
)
HISTORICAL_DATASET_REVISION = "e317b746b258630682367cc6a9d87dedd195113c"
EXPECTED_RELEASE_VIEWER_COLUMNS = [
    "images",
    "image_sizes",
    "prompt_answer",
    "prompt_answer_and_annotation",
    "answer_gt",
    "annotation_gt",
    "reward_contract",
    "instance_id",
    "domain",
    "task",
    "scene_id",
    "query_id",
    "scene_variant",
    "trace_ref",
]
EXPECTED_RELEASE_DROPPED_COLUMNS = [
    "uid",
    "prompt",
    "prompt_active",
    "prompt_answer_only",
    "prompt_mode",
    "difficulty_bin",
    "bucket_id_str",
    "image_sizes_original",
    "image_sizes_exported",
]
EXPECTED_RELEASE_SIDECARS = [
    "build_report.json",
    "curriculum_index.jsonl.zst",
    "export_provenance.jsonl.zst",
    "train_instances.jsonl.zst",
    "validation_report.json",
]
EXPECTED_DOMAIN_TASK_COUNTS = {
    "charts": 180,
    "games": 170,
    "geometry": 170,
    "graph": 60,
    "icons": 50,
    "illustrations": 60,
    "pages": 80,
    "physics": 50,
    "puzzles": 60,
    "symbolic": 60,
    "three_d": 60,
}

# One intentionally small renderer from every public domain. Running each
# twice catches accidental use of ambient randomness in prompts, answers,
# annotations, traces, or pixels.
REPRESENTATIVE_TASKS = {
    "charts": "task_charts__annotated_series__callout_endpoint_change_value",
    "games": "task_games__2048__max_tile_value",
    "geometry": "task_geometry__graph_paper__polygon_area_value",
    "graph": "task_graph__adjacency__undirected_component_count",
    "icons": "task_icons__named_grid__line_adjacency_pair_count",
    "illustrations": "task_illustrations__environment__lit_window_count",
    "pages": "task_pages__calendar__date_weekday_label",
    "physics": "task_physics__analog_meter__meter_readout_value",
    "puzzles": "task_puzzles__arithmetic_panel__number_wall_value",
    "symbolic": "task_symbolic__abacus__displayed_value_readout",
    "three_d": "task_three_d__carousel__belt_total_object_count",
}

REQUIRED_MANIFEST_LINES = {
    "include LICENSE",
    "include NOTICE",
    "include THIRD_PARTY_NOTICES.md",
    "include README.md",
    "include AGENTS.md",
    "include CONTRIBUTING.md",
    "include mkdocs.yml",
    "include CITATION.cff",
    "include constraints/compat-py314.txt",
    "include constraints/release.txt",
    "graft .agents",
    "graft docs",
    "graft examples",
    "graft scripts",
    "recursive-include src/trace_tasks/configs *.json",
    "graft src/trace_tasks/resources",
    "graft src/trace_tasks/review/app/templates",
    "graft src/trace_tasks/review/app/static",
}

FORBIDDEN_TRACKED_SUFFIXES = {
    ".key",
    ".parquet",
    ".pem",
    ".pypirc",
    ".pyc",
    ".sqlite",
    ".sqlite3",
    ".xlsx",
}
FORBIDDEN_TRACKED_PARTS = {
    ".eggs",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "checkpoints",
    "dist",
    "external",
    "logs",
    "runs",
}

SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private key",
        re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    ),
    (
        "GitHub token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    ),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("OpenAI token", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("AWS access key", re.compile(r"\bAKIA[A-Z0-9]{16}\b")),
    (
        "Weights & Biases API key",
        re.compile(
            r"(?ix)"
            r"(?:"
            r"\bWANDB_API_KEY\b\s*[:=]"
            r"|\bwandb\.login\s*\([^\n)]*\bkey\s*="
            r"|['\"]?api[_-]?key['\"]?\s*[:=]"
            r")"
            r"\s*['\"]?[0-9a-f]{40}\b"
        ),
    ),
)

# Require a delimiter before Unix paths so URL components such as
# ``.../src/home/...`` are not mistaken for a developer checkout.
MACHINE_PATH_PATTERN = re.compile(
    r"(?:^|[\s'\"`=:(])"
    r"(?P<path>/(?:home|Users)/[A-Za-z0-9._-]+(?:/[^\s'\"`)]*)?"
    r"|/(?:dev/shm|workspace|root|mnt|data|scratch|opt|content|kaggle/working)"
    r"(?:/[^\s'\"`)]*)?"
    r"|[A-Za-z]:\\Users\\[^\s'\"`)]*)"
)


class ReleaseCheckError(RuntimeError):
    """Raised when a public release contract is not satisfied."""


def _add_source_tree_to_path() -> None:
    source = str(SOURCE_ROOT)
    if source not in sys.path:
        sys.path.insert(0, source)


def _git_tracked_files(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z"],
        check=True,
        stdout=subprocess.PIPE,
    )
    return sorted(path.decode("utf-8") for path in result.stdout.split(b"\0") if path)


def _raise_for_findings(title: str, findings: Iterable[str]) -> None:
    values = sorted(set(findings))
    if values:
        rendered = "\n".join(f"  - {value}" for value in values)
        raise ReleaseCheckError(f"{title}:\n{rendered}")


def find_forbidden_tracked_paths(paths: Iterable[str]) -> list[str]:
    """Return tracked paths that look like local artifacts or credentials."""

    findings: list[str] = []
    for raw_path in paths:
        path = PurePosixPath(raw_path)
        lower_parts = {part.lower() for part in path.parts}
        lower_name = path.name.lower()
        if lower_parts & FORBIDDEN_TRACKED_PARTS:
            findings.append(raw_path)
            continue
        if path.suffix.lower() in FORBIDDEN_TRACKED_SUFFIXES:
            findings.append(raw_path)
            continue
        if lower_name == ".env" or (
            "token" in lower_name and lower_name.endswith(".txt")
        ):
            findings.append(raw_path)
    return findings


def scan_text_for_sensitive_values(text: str) -> list[tuple[int, str]]:
    """Find secret-shaped values and machine paths in decoded text."""

    findings: list[tuple[int, str]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                findings.append((line_number, label))
        match = MACHINE_PATH_PATTERN.search(line)
        if match:
            findings.append(
                (line_number, f"absolute machine path {match.group('path')!r}")
            )
    return findings


def scan_tracked_text(repo_root: Path, paths: Iterable[str]) -> list[str]:
    """Scan UTF-8 tracked files without decoding binary resources."""

    findings: list[str] = []
    for relative_path in paths:
        path = repo_root / relative_path
        if not path.is_file():
            findings.append(f"{relative_path}: tracked file is missing")
            continue
        with path.open("rb") as handle:
            prefix = handle.read(8_192)
            if b"\0" in prefix:
                continue
            remainder = handle.read()
        try:
            text = (prefix + remainder).decode("utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, label in scan_text_for_sensitive_values(text):
            findings.append(f"{relative_path}:{line_number}: {label}")
    return findings


def find_historical_dataset_revision_in_markdown(
    repo_root: Path, paths: Iterable[str]
) -> list[str]:
    """Find stale historical dataset references in human-facing instructions."""

    findings: list[str] = []
    for relative_path in paths:
        if PurePosixPath(relative_path).suffix.lower() != ".md":
            continue
        path = repo_root / relative_path
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, start=1):
            if HISTORICAL_DATASET_REVISION in line:
                findings.append(f"{relative_path}:{line_number}")
    return findings


def check_repository_hygiene(repo_root: Path, tracked_files: Sequence[str]) -> None:
    _raise_for_findings(
        "forbidden generated or credential-shaped tracked paths",
        find_forbidden_tracked_paths(tracked_files),
    )
    _raise_for_findings(
        "tracked-file secret/path scan failed",
        scan_tracked_text(repo_root, tracked_files),
    )
    _raise_for_findings(
        "historical dataset revision appears in public Markdown; use dataset-v1",
        find_historical_dataset_revision_in_markdown(repo_root, tracked_files),
    )


def check_registry() -> None:
    _add_source_tree_to_path()
    from trace_tasks.core.source_layout_policy import parse_public_task_id
    from trace_tasks.tasks.registry import list_default_task_ids, list_task_ids

    default_ids = list_default_task_ids()
    registered_ids = list_task_ids()
    if len(default_ids) != EXPECTED_DEFAULT_TASK_COUNT:
        raise ReleaseCheckError(
            f"expected {EXPECTED_DEFAULT_TASK_COUNT} default tasks, found {len(default_ids)}"
        )
    if len(registered_ids) != EXPECTED_REGISTERED_TASK_COUNT:
        raise ReleaseCheckError(
            "expected "
            f"{EXPECTED_REGISTERED_TASK_COUNT} registered tasks, found {len(registered_ids)}"
        )
    if default_ids != sorted(default_ids) or len(default_ids) != len(set(default_ids)):
        raise ReleaseCheckError("default task ids must be sorted and unique")
    if registered_ids != sorted(registered_ids) or len(registered_ids) != len(
        set(registered_ids)
    ):
        raise ReleaseCheckError("registered task ids must be sorted and unique")
    if default_ids != registered_ids:
        raise ReleaseCheckError("public registered and default task surfaces differ")

    parsed = [parse_public_task_id(task_id) for task_id in default_ids]
    domain_counts = Counter(parts.domain for parts in parsed)
    if dict(sorted(domain_counts.items())) != EXPECTED_DOMAIN_TASK_COUNTS:
        raise ReleaseCheckError(
            "public domain counts changed: "
            f"expected {EXPECTED_DOMAIN_TASK_COUNTS}, found {dict(sorted(domain_counts.items()))}"
        )
    scenes = {(parts.domain, parts.scene_id) for parts in parsed}
    if len(scenes) != EXPECTED_SCENE_COUNT:
        raise ReleaseCheckError(
            f"expected {EXPECTED_SCENE_COUNT} public scenes, found {len(scenes)}"
        )


def _output_fingerprint(output: object) -> str:
    image = getattr(output, "image")
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")
    payload = {
        "prompt": getattr(output, "prompt"),
        "answer_gt": getattr(output, "answer_gt").to_dict(),
        "annotation_gt": getattr(output, "annotation_gt").to_dict(),
        "trace_payload": getattr(output, "trace_payload"),
        "image_mode": image.mode,
        "image_size": list(image.size),
        "image_sha256": sha256(image_buffer.getvalue()).hexdigest(),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def check_representative_generation(seed: int = 42) -> None:
    _add_source_tree_to_path()
    from trace_tasks import generate_task

    for domain, task_id in REPRESENTATIVE_TASKS.items():
        first = generate_task(task_id, seed=seed, max_attempts=100)
        second = generate_task(task_id, seed=seed, max_attempts=100)
        if _output_fingerprint(first) != _output_fingerprint(second):
            raise ReleaseCheckError(
                f"representative generation is nondeterministic for {domain}: {task_id}"
            )


def _expected_release_recipe_contract() -> dict[str, object]:
    """Return the reviewed projection of the immutable dataset recipe."""

    expected_splits = [
        {
            "dataset_name": "trace_rlvr_train_64000_all1000_seed42",
            "generation_seed": 42,
            "required_sidecars": EXPECTED_RELEASE_SIDECARS,
            "role": "train",
            "rows": 64_000,
            "rows_per_shard": 4_000,
            "samples_per_task": 64,
            "shard_count": 16,
            "shard_paths": [
                "data/train/trace_rlvr_train_64000_all1000_seed42-"
                f"{index:05d}-of-00016.parquet"
                for index in range(16)
            ],
            "task_count": EXPECTED_DEFAULT_TASK_COUNT,
        },
        {
            "dataset_name": "trace_rlvr_validation_iid_2000_all1000_seed1042",
            "generation_seed": 1_042,
            "required_sidecars": EXPECTED_RELEASE_SIDECARS,
            "role": "validation_iid",
            "rows": 2_000,
            "rows_per_shard": 2_000,
            "samples_per_task": 2,
            "shard_count": 1,
            "shard_paths": [
                "data/validation/"
                "trace_rlvr_validation_iid_2000_all1000_seed1042.parquet"
            ],
            "task_count": EXPECTED_DEFAULT_TASK_COUNT,
        },
    ]
    expected_recipe_fields = {
        "automatic_resume": False,
        "dropped_export_columns": EXPECTED_RELEASE_DROPPED_COLUMNS,
        "expected_task_ids_sha256": EXPECTED_RELEASE_TASK_IDS_SHA256,
        "image_format": "png",
        "image_storage_mode": "embedded_bytes",
        "instance_version": "v0",
        "max_attempts_per_instance": 100,
        "max_embedded_image_pixels": 1_280_000,
        "prompt_export_variant": "answer_and_annotation",
        "rebuild_contract": {
            "byte_identity_expected": False,
            "historical_paper_training_input": {
                "repository_id": "maveryn/trace",
                "revision": "e317b746b258630682367cc6a9d87dedd195113c",
            },
            "instance_id_identity_expected": False,
            "kind": "fresh_public_semantic_rebuild",
            "relation_to_historical_artifact": (
                "fresh_public_semantic_rebuild_of_frozen_recipe; "
                "not_byte_or_instance_id_identical_to_historical_paper_training_artifact"
            ),
        },
        "recipe": "trace_rlvr_all1000_iid_v1",
        "row_order": "deterministic_shuffle",
        "row_order_seed": 20_260_711,
        "schema_version": "trace-release-dataset-recipe-v1",
        "strict_repro": False,
        "task_count": EXPECTED_DEFAULT_TASK_COUNT,
        "task_ids_sha256": EXPECTED_RELEASE_TASK_IDS_SHA256,
        "task_sampling_policy": "equal_exact_count_per_task",
        "task_selection": "all public default tasks in sorted registry order",
        "viewer_columns": EXPECTED_RELEASE_VIEWER_COLUMNS,
        "viewer_schema_profile": "trace_rlvr_viewer_v1",
    }
    return {
        "fields": expected_recipe_fields,
        "source_input_paths": [
            "src/trace_tasks",
            "scripts/build_release_dataset.py",
            "docs/task_catalog/catalog.v1.json",
            "pyproject.toml",
            "constraints/release.txt",
        ],
        "splits": expected_splits,
    }


def _check_release_dataset_recipe(recipe: object) -> None:
    """Fail when a dry-run plan drifts from any reviewed release input."""

    if not isinstance(recipe, dict):
        raise ReleaseCheckError("canonical release dataset dry-run must emit an object")
    contract = _expected_release_recipe_contract()
    expected_recipe_fields = contract["fields"]
    expected_splits = contract["splits"]
    assert isinstance(expected_recipe_fields, dict)
    assert isinstance(expected_splits, list)
    actual_splits = recipe.get("splits")
    if not isinstance(actual_splits, list) or len(actual_splits) != len(
        expected_splits
    ):
        raise ReleaseCheckError("canonical release dataset recipe has the wrong splits")
    normalized_splits = [
        {key: split.get(key) for key in expected} if isinstance(split, dict) else None
        for split, expected in zip(actual_splits, expected_splits)
    ]
    actual_recipe_fields = {key: recipe.get(key) for key in expected_recipe_fields}
    source = recipe.get("source")
    if (
        actual_recipe_fields != expected_recipe_fields
        or normalized_splits != expected_splits
        or not isinstance(source, dict)
        or source.get("source_input_paths") != contract["source_input_paths"]
    ):
        raise ReleaseCheckError("canonical release dataset recipe changed unexpectedly")


def check_inventory_and_source_manifest(repo_root: Path) -> None:
    inventory_env = dict(os.environ)
    existing_pythonpath = inventory_env.get("PYTHONPATH")
    inventory_env["PYTHONPATH"] = str(repo_root / "src") + (
        f"{os.pathsep}{existing_pythonpath}" if existing_pythonpath else ""
    )
    _run(
        [
            sys.executable,
            repo_root / "scripts" / "generate_active_task_inventory.py",
            "--check",
        ],
        cwd=repo_root,
        env=inventory_env,
    )

    for script_name in (
        "generate_task_catalog.py",
        "generate_release_gallery.py",
        "generate_paper_domain_montage.py",
    ):
        _run(
            [
                sys.executable,
                repo_root / "scripts" / script_name,
                "--check",
            ],
            cwd=repo_root,
            env=inventory_env,
        )

    recipe_result = _run(
        [
            sys.executable,
            repo_root / "scripts" / "build_release_dataset.py",
            "--dry-run",
        ],
        cwd=repo_root,
        env=inventory_env,
    )
    try:
        recipe = json.loads(recipe_result.stdout)
    except json.JSONDecodeError as exc:
        raise ReleaseCheckError(
            "canonical release dataset dry-run did not emit valid JSON"
        ) from exc
    _check_release_dataset_recipe(recipe)

    _run(
        [
            sys.executable,
            repo_root / "scripts" / "check_review_recipe.py",
        ],
        cwd=repo_root,
        env=inventory_env,
    )

    manifest_path = repo_root / "MANIFEST.in"
    manifest_lines = {
        line.strip()
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = REQUIRED_MANIFEST_LINES - manifest_lines
    _raise_for_findings("MANIFEST.in is missing release declarations", missing)


def _source_package_files(repo_root: Path) -> set[str]:
    package_root = repo_root / "src" / "trace_tasks"
    files: set[str] = set()
    for path in package_root.rglob("*"):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        files.add(path.relative_to(repo_root / "src").as_posix())
    return files


def check_package_sources_are_tracked(
    repo_root: Path,
    tracked_files: Sequence[str],
) -> set[str]:
    filesystem_files = _source_package_files(repo_root)
    tracked_package_files = {
        PurePosixPath(path).relative_to("src").as_posix()
        for path in tracked_files
        if path.startswith("src/trace_tasks/")
    }
    _raise_for_findings(
        "untracked files would enter the public package",
        filesystem_files - tracked_package_files,
    )
    _raise_for_findings(
        "tracked package files are missing from the checkout",
        tracked_package_files - filesystem_files,
    )
    return filesystem_files


def _run(
    command: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    rendered = " ".join(str(part) for part in command)
    print(f"    $ {rendered}", flush=True)
    result = subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        env=dict(env) if env is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        details = "\n".join(
            value.strip() for value in (result.stdout, result.stderr) if value.strip()
        )
        raise ReleaseCheckError(
            f"command failed with exit code {result.returncode}: {rendered}"
            + (f"\n{details}" if details else "")
        )
    return result


def _single_artifact(dist_dir: Path, pattern: str, label: str) -> Path:
    artifacts = sorted(dist_dir.glob(pattern))
    if len(artifacts) != 1:
        raise ReleaseCheckError(
            f"expected one {label} matching {pattern!r}, found "
            f"{[path.name for path in artifacts]}"
        )
    return artifacts[0]


def _normalize_sdist_members(names: Iterable[str]) -> set[str]:
    members = {PurePosixPath(name).as_posix().rstrip("/") for name in names if name}
    roots = {name.split("/", 1)[0] for name in members}
    if len(roots) != 1:
        raise ReleaseCheckError(
            f"sdist must have one archive root, found {sorted(roots)}"
        )
    root = next(iter(roots))
    return {
        name[len(root) + 1 :]
        for name in members
        if name.startswith(f"{root}/") and name != root
    }


def check_built_artifact_contents(
    wheel_path: Path,
    sdist_path: Path,
    expected_package_files: set[str],
    expected_checkout_files: set[str] | None = None,
) -> None:
    with zipfile.ZipFile(wheel_path) as archive:
        wheel_members = set(archive.namelist())
        entry_points = [
            name
            for name in wheel_members
            if name.endswith(".dist-info/entry_points.txt")
        ]
        if len(entry_points) != 1:
            raise ReleaseCheckError("wheel must contain exactly one entry_points.txt")
        entry_point_text = archive.read(entry_points[0]).decode("utf-8")
        for command in (
            "trace-list",
            "trace-generate",
            "trace-validate",
            "trace-export",
            "trace-review",
        ):
            if f"{command} =" not in entry_point_text:
                raise ReleaseCheckError(
                    f"wheel is missing the {command} console script"
                )
    _raise_for_findings(
        "wheel is missing tracked package files",
        expected_package_files - wheel_members,
    )

    with tarfile.open(sdist_path, mode="r:gz") as archive:
        archive_members = archive.getmembers()
        sdist_members = _normalize_sdist_members(
            member.name for member in archive_members
        )
        archive_roots = {
            member.name.rstrip("/").split("/", 1)[0]
            for member in archive_members
            if member.name.rstrip("/")
        }
        if len(archive_roots) != 1:
            raise ReleaseCheckError(
                f"sdist must have one archive root, found {sorted(archive_roots)}"
            )
        archive_root = next(iter(archive_roots))
        actual_checkout_files: set[str] = set()
        unexpected_checkout_member_types: list[str] = []
        for member in archive_members:
            prefix = f"{archive_root}/"
            if not member.name.startswith(prefix):
                continue
            relative = PurePosixPath(member.name[len(prefix) :]).as_posix().rstrip("/")
            if not relative.startswith((".agents/", "docs/", "examples/", "scripts/")):
                continue
            if member.isfile():
                actual_checkout_files.add(relative)
            elif not member.isdir():
                unexpected_checkout_member_types.append(f"{relative} ({member.type!r})")
    required_sdist_members = {
        "AGENTS.md",
        "CITATION.cff",
        "CONTRIBUTING.md",
        "LICENSE",
        "MANIFEST.in",
        "NOTICE",
        "README.md",
        "THIRD_PARTY_NOTICES.md",
        "constraints/compat-py314.txt",
        "constraints/release.txt",
        "mkdocs.yml",
        "pyproject.toml",
    } | {f"src/{path}" for path in expected_package_files}
    required_sdist_members.update(expected_checkout_files or set())
    _raise_for_findings(
        "sdist is missing required release files",
        required_sdist_members - sdist_members,
    )
    if expected_checkout_files is not None:
        _raise_for_findings(
            "sdist contains untracked files from grafted release directories",
            actual_checkout_files - expected_checkout_files,
        )
        _raise_for_findings(
            "sdist is missing tracked files from grafted release directories",
            expected_checkout_files - actual_checkout_files,
        )
        _raise_for_findings(
            "sdist contains non-regular members in grafted release directories",
            unexpected_checkout_member_types,
        )


def build_artifacts(repo_root: Path, workspace: Path) -> tuple[Path, Path]:
    dist_dir = workspace / "dist"
    dist_dir.mkdir()
    _run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--sdist",
            "--wheel",
            "--outdir",
            dist_dir,
            repo_root,
        ],
        cwd=workspace,
    )
    return (
        _single_artifact(dist_dir, "*.whl", "wheel"),
        _single_artifact(dist_dir, "*.tar.gz", "sdist"),
    )


def _parse_json_output(
    result: subprocess.CompletedProcess[str], command: str
) -> object:
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ReleaseCheckError(
            f"{command} did not emit valid JSON: {result.stdout!r}"
        ) from exc


def _wheel_install_requirement(wheel_path: Path) -> str:
    """Return a local direct reference that also installs review dependencies."""

    return f"trace-tasks[review] @ {wheel_path.resolve().as_uri()}"


def _check_installed_review_app(
    python: Path,
    smoke_root: Path,
    env: Mapping[str, str],
) -> None:
    review_root = smoke_root / "review" / "task-reviews"
    review_root.mkdir(parents=True)
    smoke_code = "\n".join(
        (
            "from pathlib import Path",
            "import sys",
            "import trace_tasks",
            "from jinja2 import Environment, FileSystemLoader",
            "from trace_tasks.review.app import create_review_app",
            "review_root = Path(sys.argv[1])",
            "repo_root = Path(sys.argv[2])",
            "package_root = Path(trace_tasks.__file__).resolve().parent",
            "template_root = package_root / 'review' / 'app' / 'templates'",
            "static_root = package_root / 'review' / 'app' / 'static'",
            "app = create_review_app(",
            "    review_root=review_root,",
            "    database_path=review_root.parent / 'feedback' / "
            "'review_feedback.sqlite',",
            "    repo_root=repo_root,",
            "    auth_token='',",
            "    trusted_hosts=('localhost',),",
            ")",
            "route_paths = {getattr(route, 'path', '') for route in app.routes}",
            "missing = {'/', '/healthz', '/static'} - route_paths",
            "assert not missing, f'missing review routes: {sorted(missing)}'",
            "Environment(loader=FileSystemLoader(template_root)).get_template("
            "'base.html')",
            "assert (static_root / 'app.css').is_file(), "
            "'installed review CSS is missing'",
            "print('installed review app smoke passed')",
        )
    )
    _run(
        [python, "-c", smoke_code, review_root, smoke_root],
        cwd=smoke_root,
        env=env,
    )


def check_installed_cli(
    wheel_path: Path,
    workspace: Path,
    constraints_path: Path | None,
    repo_root: Path,
) -> None:
    venv_root = workspace / "venv"
    venv.EnvBuilder(with_pip=True, clear=True).create(venv_root)
    bin_dir = venv_root / "bin"
    pip = bin_dir / "python"
    install_command: list[str | os.PathLike[str]] = [
        pip,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-cache-dir",
    ]
    if constraints_path is not None:
        install_command.extend(["--constraint", constraints_path])
    install_command.append(_wheel_install_requirement(wheel_path))

    smoke_root = workspace / "installed-smoke"
    smoke_root.mkdir()
    clean_env = dict(os.environ)
    clean_env.pop("PYTHONPATH", None)
    clean_env.pop("PYTHONHOME", None)
    clean_env["PYTHONNOUSERSITE"] = "1"
    clean_env["VIRTUAL_ENV"] = str(venv_root)
    clean_env["PATH"] = f"{bin_dir}{os.pathsep}{clean_env.get('PATH', '')}"

    _run(install_command, cwd=smoke_root, env=clean_env)

    _run(
        [bin_dir / "trace-review", "--help"],
        cwd=smoke_root,
        env=clean_env,
    )
    _check_installed_review_app(pip, smoke_root, clean_env)

    listed = _parse_json_output(
        _run([bin_dir / "trace-list", "--json"], cwd=smoke_root, env=clean_env),
        "trace-list",
    )
    if (
        not isinstance(listed, dict)
        or listed.get("count") != EXPECTED_DEFAULT_TASK_COUNT
    ):
        raise ReleaseCheckError(
            "installed trace-list returned the wrong registry count"
        )

    generated = _parse_json_output(
        _run(
            [
                bin_dir / "trace-generate",
                "--task",
                REPRESENTATIVE_TASKS["geometry"],
                "--samples-per-task",
                "1",
                "--seed",
                "42",
                "--output",
                smoke_root / "generated",
            ],
            cwd=smoke_root,
            env=clean_env,
        ),
        "trace-generate",
    )
    if not isinstance(generated, dict) or not generated.get("dataset_root"):
        raise ReleaseCheckError("installed trace-generate returned no dataset root")
    dataset_root = Path(str(generated["dataset_root"]))
    if not dataset_root.is_dir():
        raise ReleaseCheckError(
            f"installed trace-generate did not create {dataset_root}"
        )

    validation = _parse_json_output(
        _run(
            [bin_dir / "trace-validate", dataset_root, "--json"],
            cwd=smoke_root,
            env=clean_env,
        ),
        "trace-validate",
    )
    if not isinstance(validation, dict) or validation.get("total_errors") != 0:
        raise ReleaseCheckError("installed trace-validate reported errors")

    export_path = smoke_root / "trace-train.jsonl"
    exported = _parse_json_output(
        _run(
            [
                bin_dir / "trace-export",
                dataset_root,
                "--output",
                export_path,
                "--format",
                "jsonl",
                "--prompt-variant",
                "answer",
            ],
            cwd=smoke_root,
            env=clean_env,
        ),
        "trace-export",
    )
    if not isinstance(exported, dict) or exported.get("row_count") != 1:
        raise ReleaseCheckError("installed trace-export returned the wrong row count")
    if len(export_path.read_text(encoding="utf-8").splitlines()) != 1:
        raise ReleaseCheckError("installed trace-export did not write one JSONL row")

    example_output_root = smoke_root / "python-example"
    example_build = _parse_json_output(
        _run(
            [
                pip,
                repo_root / "examples" / "generate_and_validate.py",
                "--output",
                example_output_root,
                "--count",
                "1",
            ],
            cwd=smoke_root,
            env=clean_env,
        ),
        "examples/generate_and_validate.py",
    )
    if (
        not isinstance(example_build, dict)
        or example_build.get("instance_count") != 1
        or example_build.get("total_errors") != 0
    ):
        raise ReleaseCheckError(
            "installed-package generation example returned bad output"
        )
    example_dataset_root = Path(str(example_build.get("dataset_root", "")))
    if not example_dataset_root.is_dir():
        raise ReleaseCheckError(
            "installed-package generation example did not create its dataset"
        )

    replayed = _parse_json_output(
        _run(
            [
                pip,
                repo_root / "examples" / "replay_and_score.py",
                example_dataset_root,
            ],
            cwd=smoke_root,
            env=clean_env,
        ),
        "examples/replay_and_score.py",
    )
    replay_scores = replayed.get("scores") if isinstance(replayed, dict) else None
    if not isinstance(replay_scores, dict) or replay_scores.get("answer_reward") != 1.0:
        raise ReleaseCheckError(
            "installed-package replay example did not earn full reward"
        )

    example_export_path = smoke_root / "python-example.jsonl"
    example_export = _parse_json_output(
        _run(
            [
                pip,
                repo_root / "examples" / "export_dataset.py",
                example_dataset_root,
                "--output",
                example_export_path,
            ],
            cwd=smoke_root,
            env=clean_env,
        ),
        "examples/export_dataset.py",
    )
    if (
        not isinstance(example_export, dict)
        or example_export.get("row_count") != 1
        or not example_export_path.is_file()
    ):
        raise ReleaseCheckError("installed-package export example returned bad output")


def _resolve_constraints(repo_root: Path, value: Path | None) -> Path:
    if value is not None:
        resolved = value.expanduser().resolve()
        if not resolved.is_file():
            raise ReleaseCheckError(f"constraints file does not exist: {resolved}")
        return resolved
    default = repo_root / "constraints" / "release.txt"
    if not default.is_file():
        raise ReleaseCheckError(f"default constraints file does not exist: {default}")
    return default


def run_release_checks(
    *,
    repo_root: Path,
    source_only: bool,
    constraints_path: Path | None,
    package_only: bool = False,
) -> None:
    tracked_files = _git_tracked_files(repo_root)

    if not package_only:
        print("[1/5] Checking active inventory and source manifest", flush=True)
        check_inventory_and_source_manifest(repo_root)
    else:
        print("[1/5] Skipping frozen source artifacts (--package-only)", flush=True)
    print("[2/5] Scanning tracked files and package sources", flush=True)
    check_repository_hygiene(repo_root, tracked_files)
    expected_package_files = check_package_sources_are_tracked(repo_root, tracked_files)

    if source_only:
        print(
            "[3/5] Skipping build and installed CLI smoke (--source-only)", flush=True
        )
    else:
        print("[3/5] Building artifacts and checking the installed CLI", flush=True)
        resolved_constraints = _resolve_constraints(repo_root, constraints_path)
        with tempfile.TemporaryDirectory(prefix="trace-public-release-") as temporary:
            workspace = Path(temporary)
            wheel_path, sdist_path = build_artifacts(repo_root, workspace)
            check_built_artifact_contents(
                wheel_path,
                sdist_path,
                expected_package_files,
                {
                    path
                    for path in tracked_files
                    if path.startswith((".agents/", "docs/", "examples/", "scripts/"))
                },
            )
            check_installed_cli(
                wheel_path,
                workspace,
                resolved_constraints,
                repo_root,
            )

    if package_only:
        print("[4/5] Skipping source registry contract (--package-only)", flush=True)
        print(
            "[5/5] Skipping source deterministic generation (--package-only)",
            flush=True,
        )
        return

    # Registry imports and renderer caches are intentionally last: the source
    # checks remain lightweight, and the full check does not retain them while
    # building and installing the 184 MB resource package.
    print("[4/5] Checking registry contract", flush=True)
    check_registry()
    print("[5/5] Checking representative deterministic generation", flush=True)
    check_representative_generation()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--source-only",
        action="store_true",
        help="Skip artifact build, wheel installation, and installed CLI smoke.",
    )
    mode.add_argument(
        "--package-only",
        action="store_true",
        help=(
            "Build and test the installed package without checking frozen "
            "source-generated artifacts."
        ),
    )
    parser.add_argument(
        "--constraints",
        type=Path,
        help="Dependency constraints for the temporary wheel installation.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        run_release_checks(
            repo_root=REPO_ROOT,
            source_only=bool(args.source_only),
            constraints_path=args.constraints,
            package_only=bool(args.package_only),
        )
    except (OSError, subprocess.SubprocessError, ReleaseCheckError) as exc:
        print(f"public release check failed: {exc}", file=sys.stderr)
        return 1
    print("public release check passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
