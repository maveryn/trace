"""Validated study configuration and path resolution.

The YAML carries no machine paths. ``work_root`` and the prepared protocol clone come from the command line or the
environment (``TRACE_MERGE_WORK_ROOT``, ``TRACE_MERGE_PROTOCOL_CLONE``); they are never invented. Policy switches that
would enable training, uploads or expansions must stay false in this phase.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]          # experiments/trace_merge
REPO_ROOT = PROJECT_ROOT.parents[1]                          # repository checkout
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "trace_merge_qwen25_v1.yaml"
ROLES = ("B", "T", "V")
FORBIDDEN_TRUE = ("allow_training", "allow_uploads", "allow_coefficient_expansion", "allow_component_experiment")
WORK_SUBDIRS = ("audit", "manifests", "aligned", "labels", "splits", "merges", "views", "runs", "tables", "figures",
                "reports", "tests", "dispatch", "logs")
SUITE_KEYS_EXPECTED = 24


class ConfigError(ValueError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json(path: Path, value: Any) -> None:
    """Atomic JSON write (temp file in the same directory, then rename)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


@dataclass(frozen=True)
class StudyConfig:
    path: Path
    raw: dict[str, Any]
    sha256: str
    work_root: Path | None = None
    protocol_clone: Path | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ accessors
    @property
    def study_id(self) -> str:
        return str(self.raw["study_id"])

    def role(self, role: str) -> dict[str, Any]:
        return dict(self.raw["model_roles"][role])

    @property
    def merges(self) -> dict[str, dict[str, Any]]:
        return {k: dict(v) for k, v in self.raw["merges"].items()}

    def repo_path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else REPO_ROOT / p

    def require_work_root(self) -> Path:
        if self.work_root is None:
            raise ConfigError("work_root is required (--work-root or TRACE_MERGE_WORK_ROOT)")
        return self.work_root

    def work(self, *parts: str) -> Path:
        return self.require_work_root().joinpath(*parts)

    def require_protocol_clone(self) -> Path:
        if self.protocol_clone is None:
            raise ConfigError("the prepared protocol clone is required (--protocol-clone or TRACE_MERGE_PROTOCOL_CLONE)")
        return self.protocol_clone

    @property
    def archive_run_root(self) -> Path:
        arch = self.raw["archive"]
        root = Path(arch["local_root"])
        if not root.is_absolute():
            root = (self.require_work_root() / root).resolve()
        return root / "runs" / arch["run_id"]

    @property
    def dataset_manifest(self) -> Path:
        return self.require_protocol_clone() / "rlvr/evaluation/.work/LMUData/trace_eval_v1_dataset_manifest.json"

    def suite(self) -> dict[str, Any]:
        return json.loads(self.repo_path(self.raw["external_suite"]).read_text(encoding="utf-8"))

    def benchmark_keys(self) -> list[str]:
        return [b["key"] for b in self.suite()["benchmarks"]]

    def categories(self) -> dict[str, list[str]]:
        return {k: list(v) for k, v in self.suite()["categories"].items()}

    def hf_snapshot(self, role: str) -> Path:
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        r = self.role(role)
        return hf_home / "hub" / f"models--{r['repo_id'].replace('/', '--')}" / "snapshots" / r["source_revision"]


def validate(raw: dict[str, Any]) -> None:
    for key in ("study_id", "phase", "model_roles", "archive", "external_suite", "merges", "split", "statistics",
                "protocols", "sampling", "evaluation_matrix"):
        if key not in raw:
            raise ConfigError(f"config is missing {key!r}")
    if raw["phase"] != "analysis_only":
        raise ConfigError("this implementation only supports phase: analysis_only")
    for key in FORBIDDEN_TRUE:
        if raw.get(key) is not False:
            raise ConfigError(f"{key} must be false in the analysis-only phase")
    roles = raw["model_roles"]
    if set(roles) != set(ROLES):
        raise ConfigError(f"model_roles must be exactly {ROLES}")
    for r, spec in roles.items():
        rev = str(spec.get("source_revision", ""))
        if len(rev) != 40 or any(c not in "0123456789abcdef" for c in rev):
            raise ConfigError(f"{r}: source_revision must be a 40-hex commit")
        for k in ("repo_id", "archived_model_id", "archived_model_revision"):
            if not spec.get(k):
                raise ConfigError(f"{r}: missing {k}")
    if float(raw.get("initial_merge_alpha", -1)) != 0.5:
        raise ConfigError("initial_merge_alpha must be 0.5 (expansions need approval)")
    for mid, spec in raw["merges"].items():
        coeffs = spec.get("coefficients") or {}
        if set(coeffs) - set(ROLES) or len(coeffs) != 2:
            raise ConfigError(f"{mid}: coefficients must name two of {ROLES}")
        if abs(sum(float(v) for v in coeffs.values()) - 1.0) > 1e-12 or any(float(v) != 0.5 for v in coeffs.values()):
            raise ConfigError(f"{mid}: only 50:50 merges are enabled")
    split = raw["split"]
    if not 0.0 < float(split["development_fraction"]) < 1.0:
        raise ConfigError("development_fraction must be in (0, 1)")
    if raw["protocols"].get("P1_backbone", {}).get("role") != "primary":
        raise ConfigError("P1_backbone must be the primary protocol")
    sampling = raw["sampling"]
    for tier, key in (("smoke", "development_seeds"), ("development", "development_seeds"),
                      ("confirmation", "confirmation_seeds")):
        seeds = [int(s) for s in (raw["evaluation_matrix"].get(tier) or {}).get("seeds") or []]
        if not seeds or seeds != [int(s) for s in sampling.get(key) or []]:
            raise ConfigError(f"evaluation_matrix.{tier}.seeds must equal sampling.{key}")
    stats = raw["statistics"]
    if stats.get("multiplicity") != "holm":
        raise ConfigError("statistics.multiplicity must be holm (the only adjustment implemented)")
    family = stats.get("primary_family") or []
    pairs = [str(c).split("-") for c in family]
    if not family or any(len(p) != 2 or not all(p) or p[0] == p[1] for p in pairs) or len(set(family)) != len(family):
        raise ConfigError("statistics.primary_family must list distinct 'A-B' comparisons of two different labels")


def load_config(path: Path | str | None = None, *, work_root: Path | str | None = None,
                protocol_clone: Path | str | None = None) -> StudyConfig:
    path = Path(path) if path else DEFAULT_CONFIG
    data = path.read_bytes()
    raw = yaml.safe_load(data)
    validate(raw)
    wr = work_root or os.environ.get("TRACE_MERGE_WORK_ROOT") or raw.get("work_root")
    pc = protocol_clone or os.environ.get("TRACE_MERGE_PROTOCOL_CLONE") or (raw.get("protocol") or {}).get("clone")
    return StudyConfig(
        path=path.resolve(), raw=raw, sha256=sha256_bytes(data),
        work_root=Path(wr).resolve() if wr else None,
        protocol_clone=Path(pc).resolve() if pc else None,
    )


def ensure_work_tree(cfg: StudyConfig) -> Path:
    root = cfg.require_work_root()
    for sub in WORK_SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root
