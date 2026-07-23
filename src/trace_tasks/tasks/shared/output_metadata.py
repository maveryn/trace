"""Shared task-output metadata helpers.

These helpers keep common task-output metadata contracts consistent across
tasks (for example task-version maps).
"""

from __future__ import annotations

from typing import Dict


_DEFAULT_TASK_VERSIONS: Dict[str, str] = {
    "dsl_spec_version": "v0",
    "template_version": "v0",
    "operator_bundle_version": "v0",
    "domain_capability_version": "v0",
    "renderer_version": "v0",
}


def default_task_versions() -> Dict[str, str]:
    """Return a copy of the standard per-task version payload."""
    return dict(_DEFAULT_TASK_VERSIONS)


__all__ = [
    "default_task_versions",
]
