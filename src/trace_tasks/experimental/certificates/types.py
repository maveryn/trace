"""Typed records for the experimental certificate extension.

Nothing here changes the Trace release ABI: certificates are an opt-in extension keyed
to existing ``instance_id`` / ``trace_record_hash`` identities (DESIGN sections 2.4-2.6).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core import canonical as _canonical
from trace_tasks.core.canonical import canonical_json_bytes
from trace_tasks.core.hash_utils import blake3_hex

SCORER_VERSION = "cert-scorer-1.1.0"  # 1.1.0: review fixes (A object, F annotation schema, path, EX-5)
COORD_FRAMES = ("px_source", "rel1000")
MAX_OBJECT_BYTES = 16 * 1024
CHECK_KINDS = ("truth", "local")
ERROR_CLASSES = ("envelope", "grounding", "operand", "computation", "answer_inconsistent", "correct")
FAMILY_STAGES = ("grounding", "operand", "computation")
COVERAGE_STATUSES = ("supported", "unsupported", "defined_not_exercised", "quarantined")
SIGNALS = ("A", "G", "M", "F")


class CertificateError(Exception):
    """Base class for extension/data problems. These are never turned into model zeros."""


class QuarantineError(CertificateError):
    """A record violates an adapter invariant; the exporter quarantines it with ``reason``."""

    def __init__(self, reason: str, message: str = "") -> None:
        self.reason = str(reason)
        super().__init__(f"{reason}: {message}" if message else reason)


class CertificateVersionError(CertificateError):
    """Missing, unknown or mismatched extension, adapter, schema or verifier version."""


class ExtensionMismatchError(CertificateVersionError):
    """The extension does not belong to the scored row (id or instance mismatch)."""


_JSON_SAFE_MAX_DEPTH = 64


def _printable_int(value: int) -> bool:
    """True when ``value`` has a decimal form under ``sys.int_max_str_digits`` (which is never
    changed here: the limit protects the process, so a too-large int is replaced instead)."""
    if -(1 << 63) <= value < (1 << 63):
        return True
    try:
        int.__repr__(value)
    except ValueError:
        return False
    return True


def json_safe(value: Any, _depth: int = 0) -> Any:
    """``value`` ready for a JSON writer, never raising on model-derived numbers (review VER-1).

    Every int (key or value) with no decimal form under ``sys.int_max_str_digits``, e.g. the
    sum of two 4300-digit certificate costs that a check logs, becomes the string
    ``"<int: N bits>"``; ``json.dumps`` would otherwise raise ValueError on it. Everything else
    is returned unchanged, and the same object is returned when nothing changes, so logged
    records of every scorable response are byte-identical. Containers deeper than 64 levels
    are left as they are.
    """
    if value is None or isinstance(value, (bool, str, float)):
        return value
    if isinstance(value, int):
        return value if _printable_int(value) else f"<int: {value.bit_length()} bits>"
    if _depth >= _JSON_SAFE_MAX_DEPTH:
        return value
    if isinstance(value, dict):
        items = [(json_safe(key, _depth + 1), json_safe(item, _depth + 1)) for key, item in value.items()]
        if all(new_key is key and new_item is item for (new_key, new_item), (key, item) in zip(items, value.items())):
            return value
        return dict(items)
    if isinstance(value, (list, tuple)):
        items = [json_safe(item, _depth + 1) for item in value]
        if all(new is old for new, old in zip(items, value)):
            return value
        return type(value)(items) if type(value) in (list, tuple) else list(items)
    return value


def require_rfc8785() -> None:
    """Fail loudly when TRACE's canonical JSON would silently fall back to ``json.dumps``."""
    if getattr(_canonical, "rfc8785", None) is None:
        raise RuntimeError(
            "rfc8785 is not importable: trace_tasks.core.canonical would fall back to "
            "json.dumps(sort_keys=True) and certificate hashes would differ between envs"
        )


def canonical_digest(payload: Any) -> str:
    """``"blake3:" + blake3(JCS(payload))`` using TRACE's canonical helpers (RFC 8785 enforced)."""
    require_rfc8785()
    return blake3_hex(canonical_json_bytes(payload))


def certificate_extension_id(
    *,
    instance_id: str,
    trace_record_hash: str,
    adapter_id: str,
    adapter_version: str,
    schema_id: str,
) -> str:
    """DESIGN 2.5: hash of the base trace identity, adapter version and schema."""
    return canonical_digest(
        {
            "instance_id": str(instance_id),
            "trace_record_hash": str(trace_record_hash),
            "adapter_id": str(adapter_id),
            "adapter_version": str(adapter_version),
            "schema_id": str(schema_id),
        }
    )


def scene_group(task: str, instance_seed: int) -> str:
    """DESIGN 2.5: statistical cluster id ``blake3(JCS({task, instance_seed}))``."""
    return canonical_digest({"task": str(task), "instance_seed": int(instance_seed)})


@dataclass(frozen=True)
class AdapterSpec:
    """Static identity and coverage of one adapter (what the registry and receipts record)."""

    adapter_id: str
    kind: str
    version: str
    schema_id: str
    verifier_version: str
    supported: Tuple[Tuple[str, Tuple[str, ...]], ...]
    instance_versions: Tuple[str, ...]
    annotation_type: str
    list_key: str | None
    list_cap: int

    def identity(self) -> Dict[str, str]:
        """The ``adapter`` block stored in every private extension record."""
        return {
            "id": self.adapter_id,
            "version": self.version,
            "schema_id": self.schema_id,
            "verifier_version": self.verifier_version,
        }

    def supports(self, task: str, query_id: str, instance_version: str) -> bool:
        if str(instance_version) not in self.instance_versions:
            return False
        return any(t == task and query_id in qids for t, qids in self.supported)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["supported"] = {task: list(qids) for task, qids in self.supported}
        data["instance_versions"] = list(self.instance_versions)
        return data


@dataclass(frozen=True)
class PublicContract:
    """What the model may know about the certificate contract (DESIGN section 4)."""

    kind: str
    schema_id: str
    coord_frame: str
    image_size: Tuple[int, int]
    annotation_type: str
    list_key: str | None
    list_cap: int
    max_object_bytes: int = MAX_OBJECT_BYTES

    def __post_init__(self) -> None:
        if self.coord_frame not in COORD_FRAMES:
            raise ValueError(f"coord_frame must be one of {COORD_FRAMES}, got {self.coord_frame!r}")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["image_size"] = [int(self.image_size[0]), int(self.image_size[1])]
        return data


@dataclass(frozen=True)
class CheckResult:
    """One verifier check. ``score`` is in [0, 1]; ``kind`` is ``truth`` or ``local``."""

    check_id: str
    family: str
    kind: str
    score: float
    applicable: bool = True
    valid_input: bool = True
    error_code: str | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in CHECK_KINDS:
            raise ValueError(f"check kind must be one of {CHECK_KINDS}, got {self.kind!r}")
        score = float(self.score)
        if not math.isfinite(score) or score < 0.0 or score > 1.0:
            raise ValueError(f"check {self.check_id!r} score must be finite in [0, 1], got {self.score!r}")
        object.__setattr__(self, "score", score)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_id": self.check_id,
            "family": self.family,
            "kind": self.kind,
            "score": self.score,
            "applicable": bool(self.applicable),
            "valid_input": bool(self.valid_input),
            "error_code": self.error_code,
            "detail": json_safe(dict(self.detail)),
        }


@dataclass
class ScoreResult:
    """Per-response scores (DESIGN 2.4/5). ``overall`` holds the reward of every arm."""

    answer: float
    grounding: float
    semantics: float
    format: float
    overall: Dict[str, float]
    joint: float
    checks: List[CheckResult]
    diagnostics: Dict[str, Any]
    coverage_status: str
    scorer_version: str
    error_class: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "A": self.answer,
            "G": self.grounding,
            "M": self.semantics,
            "F": self.format,
            "overall": dict(self.overall),
            "joint": self.joint,
            "checks": [check.to_dict() for check in self.checks],
            "diagnostics": json_safe(dict(self.diagnostics)),
            "coverage_status": self.coverage_status,
            "scorer_version": self.scorer_version,
            "error_class": self.error_class,
        }


@dataclass(frozen=True)
class Corruption:
    """A deliberately corrupted response object and the checks it must move.

    ``expected`` maps a check id (or one of the signals ``A``, ``G``, ``M``, ``F``) to
    ``"pass"`` (score == 1), ``"fail"`` (score < 1), ``"unchanged"`` (equal, within 0.01, to
    the score of the uncorrupted SFT object) or a float (exact score, abs tol 1e-9).
    """

    name: str
    response_obj: Mapping[str, Any]
    expected: Mapping[str, Any]
    description: str = ""


_EXTENSION_REQUIRED = {
    "base": ("instance_id", "trace_record_hash", "task", "query_id", "instance_seed"),
    "adapter": ("id", "version", "schema_id", "verifier_version"),
    "labels": ("answer_gt", "annotation_gt", "reward_contract"),
    "public": ("image_size",),
}


@dataclass
class CertificateExtension:
    """Private extension record (DESIGN 2.6). Labels and oracle never reach a prompt."""

    certificate_extension_id: str
    base: Dict[str, Any]
    adapter: Dict[str, Any]
    labels: Dict[str, Any]
    public: Dict[str, Any]
    oracle: Dict[str, Any]
    sft_target: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CertificateExtension":
        if not isinstance(payload, Mapping):
            raise CertificateVersionError("certificate extension must be a JSON object")
        for block, keys in _EXTENSION_REQUIRED.items():
            value = payload.get(block)
            if not isinstance(value, Mapping):
                raise CertificateVersionError(f"certificate extension is missing block {block!r}")
            missing = [key for key in keys if key not in value]
            if missing:
                raise CertificateVersionError(f"certificate extension block {block!r} is missing {missing}")
        if not isinstance(payload.get("oracle"), Mapping):
            raise CertificateVersionError("certificate extension is missing block 'oracle'")
        if not isinstance(payload.get("certificate_extension_id"), str):
            raise CertificateVersionError("certificate extension is missing certificate_extension_id")
        return cls(
            certificate_extension_id=str(payload["certificate_extension_id"]),
            base=dict(payload["base"]),
            adapter=dict(payload["adapter"]),
            labels=dict(payload["labels"]),
            public=dict(payload["public"]),
            oracle=dict(payload["oracle"]),
            sft_target=dict(payload.get("sft_target") or {}),
            provenance=dict(payload.get("provenance") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "certificate_extension_id": self.certificate_extension_id,
            "base": dict(self.base),
            "adapter": dict(self.adapter),
            "labels": dict(self.labels),
            "public": dict(self.public),
            "oracle": dict(self.oracle),
            "sft_target": dict(self.sft_target),
            "provenance": dict(self.provenance),
        }

    @property
    def instance_id(self) -> str:
        return str(self.base["instance_id"])

    @property
    def adapter_id(self) -> str:
        return str(self.adapter["id"])

    @property
    def image_size(self) -> Tuple[int, int]:
        size = self.public["image_size"]
        if not isinstance(size, Sequence) or len(size) != 2:
            raise CertificateVersionError(f"extension public.image_size must be [W, H], got {size!r}")
        width, height = int(size[0]), int(size[1])
        if width <= 0 or height <= 0:
            raise CertificateVersionError(f"extension public.image_size must be positive, got {size!r}")
        return width, height

    def expected_id(self) -> str:
        return certificate_extension_id(
            instance_id=self.base["instance_id"],
            trace_record_hash=self.base["trace_record_hash"],
            adapter_id=self.adapter["id"],
            adapter_version=self.adapter["version"],
            schema_id=self.adapter["schema_id"],
        )

    def verify_id(self) -> None:
        """Raise when the stored id does not hash from the base identity and adapter version."""
        expected = self.expected_id()
        if expected != self.certificate_extension_id:
            raise ExtensionMismatchError(
                f"certificate_extension_id {self.certificate_extension_id} does not match {expected}"
            )

    def payload_digest(self) -> str:
        """sha256 of the canonical (RFC 8785) labels, oracle and SFT target."""
        require_rfc8785()
        payload = {"labels": self.labels, "oracle": self.oracle, "sft_target": self.sft_target}
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    def verify_payload(self) -> None:
        """Raise unless ``provenance.payload_sha256`` matches the labels, oracle and SFT
        target: an edited label or oracle under an unchanged id is detected (review LEAK-04)."""
        recorded = self.provenance.get("payload_sha256")
        if not isinstance(recorded, str) or not recorded:
            raise CertificateVersionError(f"extension {self.certificate_extension_id} has no provenance.payload_sha256")
        found = self.payload_digest()
        if found != recorded:
            raise ExtensionMismatchError(
                f"extension {self.certificate_extension_id}: labels/oracle/sft_target hash {found[:12]} "
                f"!= recorded payload_sha256 {recorded[:12]} (the store row was edited)"
            )

    def verify(self) -> None:
        """``verify_id`` and ``verify_payload``."""
        self.verify_id()
        self.verify_payload()


@dataclass(frozen=True)
class TraceRecord:
    """One generated instance: the TrainInstance dict plus its TraceInstance sidecar dict."""

    train: Mapping[str, Any]
    trace: Mapping[str, Any]
    image_size: Tuple[int, int] | None = None
    image_path: str | None = None

    @property
    def task(self) -> str:
        return str(self.train["task"])

    @property
    def query_id(self) -> str:
        return str(self.train["query_id"])

    @property
    def instance_id(self) -> str:
        return str(self.train["instance_id"])

    @property
    def instance_seed(self) -> int:
        return int(self.train["instance_seed"])

    @property
    def instance_version(self) -> str:
        return str(self.train.get("instance_version", ""))

    @property
    def trace_record_hash(self) -> str:
        return str(self.train["trace_ref"]["trace_record_hash"])

    @property
    def answer_gt(self) -> Dict[str, Any]:
        return dict(self.train["answer_gt"])

    @property
    def annotation_gt(self) -> Dict[str, Any]:
        return dict(self.train["annotation_gt"])

    @property
    def reward_contract(self) -> Dict[str, Any]:
        return dict(self.train["reward_contract"])

    @property
    def scene_ir(self) -> Mapping[str, Any]:
        return self.trace.get("scene_ir") or {}

    @property
    def execution_trace(self) -> Mapping[str, Any]:
        return self.trace.get("execution_trace") or {}

    @property
    def render_map(self) -> Mapping[str, Any]:
        return self.trace.get("render_map") or {}

    @property
    def render_spec(self) -> Mapping[str, Any]:
        return self.trace.get("render_spec") or {}

    def source_size(self) -> Tuple[int, int]:
        """Source image (W, H): the PNG size when known, else the persisted canvas size."""
        if self.image_size is not None:
            return int(self.image_size[0]), int(self.image_size[1])
        spec = self.render_spec
        if spec.get("canvas_width") and spec.get("canvas_height"):
            return int(spec["canvas_width"]), int(spec["canvas_height"])
        canvas = spec.get("canvas_size")
        if isinstance(canvas, Sequence) and len(canvas) == 2:
            return int(canvas[0]), int(canvas[1])
        renderer = self.execution_trace.get("renderer") or {}
        if renderer.get("canvas_width") and renderer.get("canvas_height"):
            return int(renderer["canvas_width"]), int(renderer["canvas_height"])
        raise QuarantineError("missing_image_size", self.instance_id)


def read_trace_records(dataset_dir: str | Path, *, read_image_size: bool = True) -> List[TraceRecord]:
    """Load ``train_instances.jsonl`` and join each row with its sidecar trace record.

    The join is by ``trace_ref`` (shard id, line index) and is checked by ``instance_id``.
    """
    from trace_tasks.core.trace_store import read_trace_shard

    root = Path(dataset_dir)
    rows = [
        json.loads(line)
        for line in (root / "train_instances.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    shards: Dict[str, List[Dict[str, Any]]] = {}
    records: List[TraceRecord] = []
    for row in rows:
        ref = row["trace_ref"]
        shard_id = str(ref["shard_id"])
        if shard_id not in shards:
            shards[shard_id] = read_trace_shard(root / "traces" / shard_id)
        trace = shards[shard_id][int(ref["line_index"])]
        if trace.get("instance_id") != row.get("instance_id"):
            raise QuarantineError("trace_join_mismatch", str(row.get("instance_id")))
        size = None
        image_path = None
        images = row.get("images") or []
        if images:
            image_path = str(root / images[0]["path"])
            if read_image_size:
                from PIL import Image

                with Image.open(image_path) as image:
                    size = (int(image.size[0]), int(image.size[1]))
        records.append(TraceRecord(train=row, trace=trace, image_size=size, image_path=image_path))
    return records


__all__ = [
    "AdapterSpec",
    "COORD_FRAMES",
    "CertificateError",
    "CertificateExtension",
    "CertificateVersionError",
    "CheckResult",
    "Corruption",
    "ERROR_CLASSES",
    "ExtensionMismatchError",
    "FAMILY_STAGES",
    "MAX_OBJECT_BYTES",
    "PublicContract",
    "QuarantineError",
    "SCORER_VERSION",
    "ScoreResult",
    "TraceRecord",
    "canonical_digest",
    "certificate_extension_id",
    "json_safe",
    "read_trace_records",
    "require_rfc8785",
    "scene_group",
]
