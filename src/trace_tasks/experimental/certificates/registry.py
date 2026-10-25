"""Adapter registry: ``(task, query_id, instance_version) -> Adapter | Unsupported``.

Adapter modules are imported lazily (``adapters/<name>.py`` exposing ``create_adapter()``),
so scoring one adapter never imports the others. Tasks without an adapter are reported as
:class:`Unsupported` with a reason (coverage, DESIGN 1.1 and 10); a supported row whose
extension does not match the loaded adapter version raises :class:`CertificateVersionError`.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Dict, List, Mapping

from .adapters.base import Adapter
from .types import CertificateExtension, CertificateVersionError

ADAPTER_MODULES: Dict[str, str] = {
    "selected_set": "trace_tasks.experimental.certificates.adapters.selected_set",
    "filtered_mean": "trace_tasks.experimental.certificates.adapters.filtered_mean",
    "weighted_sum": "trace_tasks.experimental.certificates.adapters.weighted_sum",
    "path": "trace_tasks.experimental.certificates.adapters.path",
}

TASK_ADAPTERS: Dict[str, str] = {
    "task_games__chess__colored_piece_kind_count": "selected_set",
    "task_games__chess__piece_kind_count": "selected_set",
    "task_charts__table__filtered_column_mean": "filtered_mean",
    "task_physics__lever__side_torque_value": "weighted_sum",
    "task_illustrations__rpg_tactical_map__movement_cost_value": "path",
}

KNOWN_UNSUPPORTED: Dict[str, str] = {
    "task_physics__lever__missing_weight_balance_value": "needs a balance-equation certificate kind",
    "task_graph__node_link__shortest_path_length": "unique witness; label adapter deferred",
    "task_graph__metro__shortest_path_length": "unique witness; label adapter deferred",
    "task_graph__pipe_network__shortest_path_length": "unique witness; label adapter deferred",
    "task_graph__node_link__longest_path_length": "path-like task without an adapter in v1 (DESIGN 10)",
    "task_puzzles__cell_board__shortest_path_length_value": "path-like task without an adapter in v1",
    "task_puzzles__cell_board__reachable_region_size": "path-like task without an adapter in v1 (DESIGN 10)",
    "task_puzzles__maze__nearest_exit_label": "path-like task without an adapter in v1",
    "task_puzzles__maze__exit_reachability_label": "path-like task without an adapter in v1 (DESIGN 10)",
    "task_charts__table__threshold_count": "selected-set backup without empty sets; not in v1",
}


@dataclass(frozen=True)
class Unsupported:
    """A task/query the certificate extension does not cover, with the reason."""

    reason: str
    task: str = ""
    query_id: str = ""


_LOADED: Dict[str, Adapter] = {}
_EXTRA: Dict[str, Adapter] = {}


def load_adapter(name: str) -> Adapter:
    """Import ``adapters/<name>.py`` and return its (cached) adapter instance.

    Raises NotImplementedError while the module is still a placeholder.
    """
    if name in _LOADED:
        return _LOADED[name]
    if name not in ADAPTER_MODULES:
        raise CertificateVersionError(f"unknown adapter module {name!r}")
    module = importlib.import_module(ADAPTER_MODULES[name])
    factory = getattr(module, "create_adapter", None)
    if factory is None:
        raise NotImplementedError(f"adapter module {name!r} defines no create_adapter()")
    adapter = factory()
    if not isinstance(adapter, Adapter):
        raise TypeError(f"{name}.create_adapter() must return an Adapter, got {type(adapter).__name__}")
    type(adapter).validate_definition()
    _LOADED[name] = adapter
    return adapter


def register_adapter(adapter: Adapter) -> None:
    """Register an extra adapter instance by ``adapter_id`` (tests and local extensions)."""
    if not isinstance(adapter, Adapter):
        raise TypeError("register_adapter expects an Adapter instance")
    type(adapter).validate_definition()
    _EXTRA[adapter.adapter_id] = adapter


def unregister_adapter(adapter_id: str) -> None:
    _EXTRA.pop(adapter_id, None)


def resolve(task: str, query_id: str, instance_version: str = "v0") -> Adapter | Unsupported:
    """The adapter for a generated row, or Unsupported(reason)."""
    for adapter in _EXTRA.values():
        if adapter.supports(task, query_id, instance_version):
            return adapter
    name = TASK_ADAPTERS.get(task)
    if name is None:
        return Unsupported(KNOWN_UNSUPPORTED.get(task, "no_adapter"), task, query_id)
    try:
        adapter = load_adapter(name)
    except NotImplementedError as exc:
        return Unsupported(f"adapter_not_implemented: {exc}", task, query_id)
    if adapter.supports(task, query_id, instance_version):
        return adapter
    if task in adapter.supported and query_id in adapter.supported[task]:
        return Unsupported(f"instance_version_not_supported: {instance_version}", task, query_id)
    return Unsupported(f"query_not_supported: {query_id}", task, query_id)


def adapter_by_id(adapter_id: str) -> Adapter:
    """The adapter whose ``adapter_id`` matches; the module is ``adapter_id``'s first segment."""
    if adapter_id in _EXTRA:
        return _EXTRA[adapter_id]
    name = str(adapter_id).split(".", 1)[0]
    if name not in ADAPTER_MODULES:
        raise CertificateVersionError(f"unknown adapter id {adapter_id!r}")
    adapter = load_adapter(name)
    if adapter.adapter_id != adapter_id:
        raise CertificateVersionError(f"adapter id {adapter_id!r} not provided by module {name!r}")
    return adapter


def check_adapter_identity(adapter: Adapter, extension: CertificateExtension) -> None:
    """Raise unless the extension's adapter block equals the loaded adapter's identity."""
    expected = adapter.identity()
    found = {key: extension.adapter.get(key) for key in expected}
    if found != expected:
        raise CertificateVersionError(f"extension adapter {found} does not match loaded adapter {expected}")
    base = extension.base
    if not adapter.supports(str(base["task"]), str(base["query_id"]), str(base.get("instance_version", "v0"))):
        raise CertificateVersionError(
            f"adapter {adapter.adapter_id} does not support {base['task']}/{base['query_id']}"
        )


def adapter_for_extension(extension: CertificateExtension | Mapping) -> Adapter:
    """Load the adapter named by an extension and check version, schema and verifier."""
    if not isinstance(extension, CertificateExtension):
        extension = CertificateExtension.from_dict(extension)
    adapter = adapter_by_id(extension.adapter_id)
    check_adapter_identity(adapter, extension)
    return adapter


def supported_adapters() -> List[Adapter]:
    """Every implemented adapter (placeholders are skipped)."""
    adapters: List[Adapter] = []
    for name in ADAPTER_MODULES:
        try:
            adapters.append(load_adapter(name))
        except NotImplementedError:
            continue
    return adapters


__all__ = [
    "ADAPTER_MODULES",
    "KNOWN_UNSUPPORTED",
    "TASK_ADAPTERS",
    "Unsupported",
    "adapter_by_id",
    "adapter_for_extension",
    "check_adapter_identity",
    "load_adapter",
    "register_adapter",
    "resolve",
    "supported_adapters",
    "unregister_adapter",
]
