#!/usr/bin/env python3
"""Fingerprint the exact VLMEvalKit evaluator worktree and TRACE extensions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VLMEVAL_ROOT = REPO_ROOT / "external" / "VLMEvalKit"
SCHEMA_VERSION = "trace-eval-evaluator-provenance-v1"
LOCAL_SOURCE_PATHS = (
    "rlvr/evaluation/vlmevalkit_extensions",
    "rlvr/evaluation/scripts/apply_vlmevalkit_trace_extensions.py",
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"cannot inspect git worktree {root}: {error}") from error


def _git_paths(root: Path, pathspecs: Iterable[str] = ()) -> tuple[str, ...]:
    command = ["ls-files", "-z", "--cached", "--others", "--exclude-standard"]
    selected = tuple(pathspecs)
    if selected:
        command.extend(("--", *selected))
    encoded = _git(root, *command)
    paths = {
        item.decode("utf-8", errors="surrogateescape")
        for item in encoded.split(b"\0")
        if item
    }
    return tuple(sorted(paths))


def _path_record(root: Path, relative: str) -> dict[str, Any]:
    if not relative or relative.startswith("/") or ".." in Path(relative).parts:
        raise RuntimeError(f"unsafe evaluator source path: {relative!r}")
    path = root / relative
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return {"path": relative, "kind": "missing"}
    mode = stat.S_IMODE(metadata.st_mode)
    if stat.S_ISREG(metadata.st_mode):
        return {
            "path": relative,
            "kind": "file",
            "mode": mode,
            "size": metadata.st_size,
            "sha256": sha256_file(path),
        }
    if stat.S_ISLNK(metadata.st_mode):
        target = os.readlink(path)
        return {
            "path": relative,
            "kind": "symlink",
            "mode": mode,
            "target": target,
            "sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
        }
    if stat.S_ISDIR(metadata.st_mode):
        return {"path": relative, "kind": "directory", "mode": mode}
    raise RuntimeError(f"unsupported evaluator source type: {path}")


def build_evaluator_provenance(
    *,
    repo_root: Path = REPO_ROOT,
    vlmeval_root: Path = DEFAULT_VLMEVAL_ROOT,
) -> dict[str, Any]:
    repo_root = repo_root.expanduser().resolve()
    vlmeval_root = vlmeval_root.expanduser().resolve()
    if not (vlmeval_root / ".git").exists():
        raise RuntimeError(f"VLMEvalKit git checkout does not exist: {vlmeval_root}")
    if not (repo_root / ".git").exists():
        # Git worktrees use a .git file rather than a directory.
        if not (repo_root / ".git").is_file():
            raise RuntimeError(f"TRACE git checkout does not exist: {repo_root}")

    upstream_paths = _git_paths(vlmeval_root)
    local_paths = _git_paths(repo_root, LOCAL_SOURCE_PATHS)
    apply_path = "rlvr/evaluation/scripts/apply_vlmevalkit_trace_extensions.py"
    if apply_path not in local_paths:
        raise RuntimeError(f"evaluator extension installer is not tracked or visible: {apply_path}")

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "vlmevalkit": {
            "git_head": _git(vlmeval_root, "rev-parse", "HEAD").decode("ascii").strip(),
            "files": [_path_record(vlmeval_root, path) for path in upstream_paths],
        },
        "trace_extensions": {
            "files": [_path_record(repo_root, path) for path in local_paths],
        },
    }
    manifest["sha256"] = hashlib.sha256(canonical_json(manifest).encode("utf-8")).hexdigest()
    return manifest


def evaluator_provenance_sha256(
    *,
    repo_root: Path = REPO_ROOT,
    vlmeval_root: Path = DEFAULT_VLMEVAL_ROOT,
) -> str:
    return str(
        build_evaluator_provenance(repo_root=repo_root, vlmeval_root=vlmeval_root)["sha256"]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--vlmeval-root", type=Path, default=DEFAULT_VLMEVAL_ROOT)
    parser.add_argument("--hash-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest = build_evaluator_provenance(
        repo_root=args.repo_root,
        vlmeval_root=args.vlmeval_root,
    )
    if args.hash_only:
        print(manifest["sha256"])
    else:
        print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
