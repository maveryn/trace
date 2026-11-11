"""Configuration and paths of the follow-up study.

The follow-up reuses the reviewed study's audited helpers (``experiments/trace_merge``) by import and its pinned roles,
protocol and interface file lists through ``base_config``; it never writes into that study's work root. The work root
comes from ``--work-root`` or ``TRACE_FOLLOWUP_ROOT``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]           # experiments/trace_module_followup
MERGE_STUDY_ROOT = PACKAGE_ROOT.parent / "trace_merge"        # experiments/trace_merge
if str(MERGE_STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(MERGE_STUDY_ROOT))

from trace_merge.config import StudyConfig, load_config, sha256_bytes, sha256_file, canonical_json, write_json  # noqa: E402,F401

DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "followup_v1.yaml"
ROLES = ("B", "T", "V")
WORK_SUBDIRS = ("audit", "hybrids", "views", "runs", "manifests", "tables", "reports", "figures", "logs", "dispatch",
                "splits", "aligned")


class FollowupError(RuntimeError):
    pass


@dataclass
class FollowupConfig:
    path: Path
    raw: dict[str, Any]
    sha256: str
    base: StudyConfig
    work_root: Path | None

    def require_work_root(self) -> Path:
        if self.work_root is None:
            raise FollowupError("work root is required (--work-root or TRACE_FOLLOWUP_ROOT)")
        return self.work_root

    def work(self, *parts: str) -> Path:
        return self.require_work_root().joinpath(*parts)

    @property
    def hybrids(self) -> dict[str, dict[str, Any]]:
        return {k: dict(v) for k, v in self.raw["hybrids"].items()}

    @property
    def cells(self) -> dict[str, str]:
        """The seven F1 cells (cell -> served-slug stem)."""
        return dict(self.raw["cells"])

    @property
    def extension_cells(self) -> dict[str, str]:
        """Conditional-extension checkpoints (F2), evaluated only as their decision record says."""
        return dict(self.raw.get("extension_cells") or {})

    @property
    def scaled(self) -> dict[str, dict[str, Any]]:
        """F3 interpolated checkpoints (design/decision_F3.md), built into ``scaled/<id>`` by the reviewed builder."""
        return {k: dict(v) for k, v in (self.raw.get("scaled") or {}).items()}

    def constructed(self, source: str) -> tuple[Path, Path] | None:
        """(directory, manifest) of a built hybrid or F3 checkpoint; None for a parent role."""
        if source in self.hybrids:
            d = self.work("hybrids", source)
            return d, d / "manifest.json"
        if source in self.scaled:
            d = self.work("scaled", source)
            return d, d / "merge_manifest.json"
        return None

    def parent_dir(self, role: str) -> Path:
        return self.base.hf_snapshot(role)

    def revision(self, role: str) -> str:
        return str(self.base.role(role)["source_revision"])


def load(path: Path | str | None = None, *, work_root: Path | str | None = None,
         protocol_clone: Path | str | None = None) -> FollowupConfig:
    path = Path(path) if path else DEFAULT_CONFIG
    data = path.read_bytes()
    raw = yaml.safe_load(data)
    for key in ("study_id", "base_config", "parts", "partitions", "expected_counts", "hybrids", "cells", "evaluation"):
        if key not in raw:
            raise FollowupError(f"{path}: missing {key!r}")
    if any(v is not False for v in (raw.get("policy") or {}).values()) or not raw.get("policy"):
        raise FollowupError("policy switches (training, uploads, merge_search, averaging) must all be false")
    base_path = (path.parent / raw["base_config"]).resolve()
    # the base config resolves its own work root; the follow-up never uses it, so it is not passed
    base = load_config(base_path, work_root=None, protocol_clone=protocol_clone)
    wr = work_root or os.environ.get("TRACE_FOLLOWUP_ROOT") or raw.get("work_root")
    return FollowupConfig(path=path.resolve(), raw=raw, sha256=sha256_bytes(data), base=base,
                          work_root=Path(wr).resolve() if wr else None)


def ensure_work_tree(cfg: FollowupConfig) -> Path:
    root = cfg.require_work_root()
    for sub in WORK_SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root
