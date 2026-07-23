"""Public API for the Trace visual reasoning task environment."""

from __future__ import annotations

from typing import Any, Mapping

from .core.builder import BuildError, build_dataset
from .core.config import BuildConfig, BuildTaskConfig, load_build_config
from .core.reward_scoring import score_trace_response
from .core.rlvr_export import RLVRExportResult, export_trace_dataset_to_rlvr
from .tasks.base import TaskOutput
from .tasks.registry import create_task, list_default_task_ids, list_task_ids

__version__ = "0.1.0"


def generate_task(
    task_id: str,
    *,
    seed: int,
    params: Mapping[str, Any] | None = None,
    max_attempts: int = 100,
) -> TaskOutput:
    """Generate one deterministic output from a public task id."""

    task = create_task(task_id)
    return task.generate(
        int(seed),
        params=dict(params or {}),
        max_attempts=int(max_attempts),
    )


__all__ = [
    "BuildConfig",
    "BuildError",
    "BuildTaskConfig",
    "RLVRExportResult",
    "TaskOutput",
    "__version__",
    "build_dataset",
    "create_task",
    "export_trace_dataset_to_rlvr",
    "generate_task",
    "list_default_task_ids",
    "list_task_ids",
    "load_build_config",
    "score_trace_response",
]
