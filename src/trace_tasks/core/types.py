"""Typed core ABI records for Trace datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from .reward_contracts import RewardContract


@dataclass(frozen=True)
class TypedValue:
    """Typed envelope for answer/annotation values."""

    type: str
    value: Any

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "value": self.value}


@dataclass(frozen=True)
class ImageRecord:
    """Training-facing image metadata for one image artifact."""

    image_id: str
    format: str
    image_hash: str
    path: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TraceRef:
    """Pointer from TrainInstance to a sidecar trace record."""

    shard_id: str
    line_index: int
    trace_record_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainInstance:
    """Training-facing dataset ABI record."""

    instance_version: str
    instance_id: str
    instance_seed: int
    domain: str
    task: str
    scene_id: str
    query_id: str
    prompt: str
    images: List[ImageRecord]
    answer_gt: TypedValue
    annotation_gt: TypedValue
    reward_contract: RewardContract
    trace_ref: TraceRef
    versions: Dict[str, str]
    prompt_variants: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "instance_version": self.instance_version,
            "instance_id": self.instance_id,
            "instance_seed": int(self.instance_seed),
            "domain": self.domain,
            "task": self.task,
            "scene_id": self.scene_id,
            "query_id": self.query_id,
            "prompt": self.prompt,
            "prompt_variants": dict(self.prompt_variants),
            "images": [image.to_dict() for image in self.images],
            "answer_gt": self.answer_gt.to_dict(),
            "annotation_gt": self.annotation_gt.to_dict(),
            "reward_contract": self.reward_contract.to_dict(),
            "trace_ref": self.trace_ref.to_dict(),
            "versions": dict(self.versions),
        }
        return data


@dataclass
class TraceInstance:
    """Heavy sidecar trace payload for replay/debugging."""

    instance_id: str
    scene_ir: Dict[str, Any]
    query_spec: Dict[str, Any]
    render_spec: Dict[str, Any]
    render_map: Dict[str, Any]
    execution_trace: Dict[str, Any]
    witness_symbolic: Dict[str, Any]
    projected_annotation: Dict[str, Any]
    taxonomy: Dict[str, Any] | None = None
    seed_map: Dict[str, int] | None = None
    answer_gt: TypedValue | None = None
    annotation_gt: TypedValue | None = None
    reward_contract: RewardContract | None = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "instance_id": self.instance_id,
            "scene_ir": self.scene_ir,
            "query_spec": self.query_spec,
            "render_spec": self.render_spec,
            "render_map": self.render_map,
            "execution_trace": self.execution_trace,
            "witness_symbolic": self.witness_symbolic,
            "projected_annotation": self.projected_annotation,
        }
        if self.taxonomy is not None:
            data["taxonomy"] = dict(self.taxonomy)
        if self.seed_map is not None:
            data["seed_map"] = dict(self.seed_map)
        if self.answer_gt is not None:
            data["answer_gt"] = self.answer_gt.to_dict()
        if self.annotation_gt is not None:
            data["annotation_gt"] = self.annotation_gt.to_dict()
        if self.reward_contract is not None:
            data["reward_contract"] = self.reward_contract.to_dict()
        return data


@dataclass(frozen=True)
class CurriculumIndex:
    """Curriculum sampling record keyed by instance_id."""

    instance_id: str
    domain: str
    task: str
    scene_id: str
    query_id: str

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "instance_id": self.instance_id,
            "domain": self.domain,
            "task": self.task,
            "scene_id": self.scene_id,
            "query_id": self.query_id,
        }
        return data
