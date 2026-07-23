"""Shared label helpers for node-link objectives."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ...shared.label_assets import resolve_graph_edge_label_support


def build_edge_label_support_resolver(
    axes: Any,
    *,
    default_support_size: int,
    default_min_chars: int,
    default_max_chars: int,
):
    """Return a resolver that samples edge labels after node labels are known."""

    def _resolve(rng: Any, node_labels: Sequence[str]):
        raw_weights = axes.values.get("edge_label_bucket_weights")
        bucket_weights = dict(raw_weights) if isinstance(raw_weights, Mapping) else None
        excluded = sorted(str(label).strip().lower() for label in node_labels if str(label).strip())
        explicit_support = axes.values.get("edge_label_support")
        if explicit_support is not None:
            support = tuple(
                str(label).strip().lower()
                for label in explicit_support
                if str(label).strip() and str(label).strip().lower() not in set(excluded)
            )
            if len(set(support)) != len(support) or len(support) < 2:
                raise ValueError(
                    "edge_label_support must contain at least two unique labels after excluding node labels"
                )
            return tuple(str(label) for label in support), {
                "edge_label_source_kind": "explicit_support",
                "edge_label_bucket": "",
                "edge_label_manifest": "",
                "edge_label_filter": {"excluded_node_labels": excluded},
                "edge_label_bucket_probabilities": {},
            }
        resolved = resolve_graph_edge_label_support(
            rng,
            support_size=int(axes.values.get("edge_label_support_size", int(default_support_size))),
            min_chars=int(axes.values.get("edge_label_min_chars", int(default_min_chars))),
            max_chars=int(axes.values.get("edge_label_max_chars", int(default_max_chars))),
            bucket_weights=bucket_weights,
            exclude_labels=tuple(str(label) for label in node_labels),
        )
        return tuple(str(label) for label in resolved.labels), {
            "edge_label_source_kind": str(resolved.label_source_kind),
            "edge_label_bucket": str(resolved.label_bucket),
            "edge_label_manifest": str(resolved.label_manifest),
            "edge_label_filter": dict(resolved.label_filter) | {"excluded_node_labels": excluded},
            "edge_label_bucket_probabilities": dict(resolved.label_bucket_probabilities),
        }

    return _resolve


__all__ = ["build_edge_label_support_resolver"]
