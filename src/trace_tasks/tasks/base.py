"""Base task interface for Trace generators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Protocol

from PIL import Image

from ..core.types import TypedValue


@dataclass
class TaskOutput:
    """Task-level generation output before builder packaging."""

    prompt: str
    answer_gt: TypedValue
    annotation_gt: TypedValue
    image: Image.Image
    image_id: str
    trace_payload: Dict[str, Any]
    task_versions: Dict[str, str]
    # Canonical task-internal branch id used for sampling, replay, and diagnostics.
    scene_id: str = ""
    query_id: str = ""
    prompt_variants: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Keep prompt trace metadata at the canonical query-spec location."""
        query_spec = self.trace_payload.get("query_spec")
        if not isinstance(query_spec, dict) or "prompt_variants" in query_spec:
            return

        candidates = (
            query_spec.get("prompt"),
            self.trace_payload.get("render_spec", {}).get("prompt")
            if isinstance(self.trace_payload.get("render_spec"), Mapping)
            else None,
            self.trace_payload.get("prompt_spec"),
            self.trace_payload.get("prompt"),
        )
        for candidate in candidates:
            if not isinstance(candidate, Mapping) or "prompt_variants" not in candidate:
                continue
            if "template_id" in candidate and "template_id" not in query_spec:
                query_spec["template_id"] = candidate["template_id"]
            for key in ("prompt_variant", "prompt_variant_active_key", "prompt_variants"):
                if key in candidate:
                    query_spec[key] = candidate[key]
            break


class Task(Protocol):
    """Protocol that every registered Trace task must satisfy."""

    task_id: str
    domain: str
    scene_id: str | None
    reasoning_operations: tuple[str, ...]
    default_dataset_enabled: bool

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one deterministic task output for the given seed and params."""
