"""Build and task configuration models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass(frozen=True)
class BuildTaskConfig:
    """Task request descriptor inside a build configuration."""

    task_id: str
    count: int | None = None
    weight: float | None = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BuildConfig:
    """Top-level dataset build configuration."""

    output_root: str
    dataset_name: str
    instance_version: str
    image_format: str
    tasks: List[BuildTaskConfig]
    num_instances: int | None = None
    strict_repro: bool = False
    max_attempts_per_instance: int = 100
    sampling_seed: int = 0
    workers: int = 1
    max_in_flight: int = 0


def load_build_config(path: str | Path) -> BuildConfig:
    """Load YAML build config from disk."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    tasks = [
        BuildTaskConfig(
            task_id=str(item["task_id"]),
            count=(int(item["count"]) if item.get("count") is not None else None),
            weight=(float(item["weight"]) if item.get("weight") is not None else None),
            params=dict(item.get("params", {})),
        )
        for item in data.get("tasks", [])
    ]
    return BuildConfig(
        output_root=str(data["output_root"]),
        dataset_name=str(data["dataset_name"]),
        instance_version=str(data.get("instance_version", "v0")),
        image_format=str(data.get("image_format", "png")).lower(),
        tasks=tasks,
        num_instances=(int(data["num_instances"]) if data.get("num_instances") is not None else None),
        strict_repro=bool(data.get("strict_repro", False)),
        max_attempts_per_instance=int(data.get("max_attempts_per_instance", 100)),
        sampling_seed=int(data.get("sampling_seed", 0)),
        workers=int(data.get("workers", 1)),
        max_in_flight=int(data.get("max_in_flight", 0)),
    )
