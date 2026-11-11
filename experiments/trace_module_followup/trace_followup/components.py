"""Stored-tensor ownership and donor assignment for component-swap hybrids (design/followup_v1.md, F1/F2).

Every stored tensor belongs to exactly one part (E encoder, J vision-language merger, L language component) through the
audited ``trace_merge.compatibility.tensor_group`` map; a partition groups parts (F1: frontend = (E, J), language = L).
An unclassified name is an error, never a default owner. A hybrid assigns one donor role to each group of its
partition and copies every tensor from that donor; nothing is averaged.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from .common import ROLES, FollowupConfig, FollowupError, canonical_json, sha256_bytes

from trace_merge.compatibility import tensor_group  # noqa: E402

_LAYER = re.compile(r"^model\.layers\.(\d+)\.")


def _owns(spec: list[str] | dict[str, Any], name: str, group: str) -> bool:
    """A part spec is a list of tensor groups, or {groups: [...], layers: [first, last]} for a decoder block range."""

    if isinstance(spec, dict):
        if group not in spec["groups"]:
            return False
        if "layers" not in spec:
            return True
        m = _LAYER.match(name)
        if m is None:
            raise FollowupError(f"{name}: a layer-range part needs a model.layers.<n> tensor")
        lo, hi = (int(x) for x in spec["layers"])
        return lo <= int(m.group(1)) <= hi
    return group in spec


def part_of(name: str, parts: dict[str, Any]) -> str:
    """The part owning a stored tensor name; raises on an unclassified or doubly owned name."""

    group = tensor_group(name)
    owners = [p for p, spec in parts.items() if _owns(spec, name, group)]
    if len(owners) != 1:
        raise FollowupError(f"{name}: tensor group {group!r} has {len(owners)} owners in {sorted(parts)}")
    return owners[0]


def partition_def(cfg: FollowupConfig, partition_id: str) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """(parts, partition) of a partition id: F1/F2a partitions use ``parts``; F2c's decoder-block partitions use
    ``fine_parts`` (layer ranges)."""

    if partition_id in cfg.raw["partitions"]:
        return cfg.raw["parts"], cfg.raw["partitions"][partition_id]
    fine = cfg.raw.get("fine_partitions") or {}
    if partition_id in fine:
        return cfg.raw["fine_parts"], fine[partition_id]
    raise FollowupError(f"unknown partition {partition_id!r}")


def group_of(name: str, parts: dict[str, list[str]], partition: dict[str, list[str]]) -> str:
    """The partition group owning a stored tensor name (no fallback ownership)."""

    part = part_of(name, parts)
    owners = [g for g, members in partition.items() if part in members]
    if len(owners) != 1:
        raise FollowupError(f"{name}: part {part} has {len(owners)} owners in partition {sorted(partition)}")
    return owners[0]


def ownership(cfg: FollowupConfig, names: Iterable[str], partition_id: str) -> dict[str, str]:
    """name -> group for one partition, with every part of the config covered by exactly one group."""

    parts, partition = partition_def(cfg, partition_id)
    covered = sorted(p for members in partition.values() for p in members)
    if covered != sorted(parts):
        raise FollowupError(f"partition {partition_id} covers parts {covered}, expected {sorted(parts)}")
    return {n: group_of(n, parts, partition) for n in sorted(names)}


def mask_sha256(owner: dict[str, str]) -> str:
    return sha256_bytes(canonical_json(dict(sorted(owner.items()))).encode("utf-8"))


def check_counts(cfg: FollowupConfig, names: Iterable[str]) -> dict[str, int]:
    """Counts per part and per F1 group; must equal the expected inventory (F = 390, L = 339, total 729)."""

    names = sorted(names)
    parts = cfg.raw["parts"]
    counts: dict[str, int] = {p: 0 for p in parts}
    for n in names:
        counts[part_of(n, parts)] += 1
    owner = ownership(cfg, names, "component")
    for g in cfg.raw["partitions"]["component"]:
        counts[g] = sum(1 for v in owner.values() if v == g)
    counts["total"] = len(names)
    expected = {k: int(v) for k, v in cfg.raw["expected_counts"].items()}
    if counts != expected:
        raise FollowupError(f"tensor inventory {counts} differs from the expected {expected}")
    return counts


def donor_map(cfg: FollowupConfig, hybrid_id: str, names: Iterable[str]) -> dict[str, str]:
    """name -> donor role for one configured hybrid."""

    spec = cfg.hybrids[hybrid_id]
    _parts, partition = partition_def(cfg, spec["partition"])
    donors = dict(spec["donors"])
    if set(donors) != set(partition):
        raise FollowupError(f"{hybrid_id}: donors {sorted(donors)} must name exactly the groups {sorted(partition)}")
    if any(r not in ROLES for r in donors.values()):
        raise FollowupError(f"{hybrid_id}: donors must be roles {ROLES}")
    owner = ownership(cfg, names, spec["partition"])
    return {n: donors[g] for n, g in owner.items()}


def component_map(cfg: FollowupConfig, names: Iterable[str], numels: dict[str, int]) -> dict[str, Any]:
    """audit/component_map.json: every tensor's part and F1 group, sorted key lists, element counts, mask hashes, and
    each hybrid's intended donor per tensor."""

    names = sorted(names)
    counts = check_counts(cfg, names)
    parts = cfg.raw["parts"]
    out: dict[str, Any] = {
        "schema": "trace-followup-component-map-v1",
        "tensor_count": len(names),
        "counts": counts,
        "parts": {n: part_of(n, parts) for n in names},
        "partitions": {},
        "hybrids": {},
    }
    for pid in cfg.raw["partitions"]:
        owner = ownership(cfg, names, pid)
        groups = sorted(set(owner.values()))
        out["partitions"][pid] = {
            "groups": {g: sorted(n for n in names if owner[n] == g) for g in groups},
            "elements": {g: sum(numels[n] for n in names if owner[n] == g) for g in groups},
            "mask_sha256": mask_sha256(owner),
        }
    for hid in cfg.hybrids:
        dm = donor_map(cfg, hid, names)
        out["hybrids"][hid] = {"spec": cfg.hybrids[hid], "tensors_by_donor": {r: sum(1 for v in dm.values() if v == r)
                                                                                for r in sorted(set(dm.values()))},
                               "donor_map_sha256": mask_sha256(dm)}
    return out
