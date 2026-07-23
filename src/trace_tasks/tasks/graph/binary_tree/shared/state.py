"""State records and scene constants for graph binary-tree diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from PIL import Image


SCENE_ID = "binary_tree"
SCENE_VARIANTS: Tuple[str, ...] = (
    "classic_tree",
    "paper_tree",
    "boxed_tree",
)
CONNECTOR_STYLE_VARIANTS: Tuple[str, ...] = (
    "diagonal_edges",
    "elbow_edges",
)


@dataclass(frozen=True)
class BinaryTreeNode:
    """One labeled node in a rooted ordered binary tree."""

    node_id: str
    label: str
    parent_id: str | None
    left_id: str | None
    right_id: str | None
    depth: int


@dataclass(frozen=True)
class BinaryTreeSample:
    """Trace-ready binary-tree structure and derived orders."""

    nodes: Tuple[BinaryTreeNode, ...]
    root_id: str
    label_variant: str
    label_source_kind: str
    label_bucket: str
    label_manifest: str
    label_filter: Dict[str, Any]
    label_bucket_probabilities: Dict[str, float]
    preorder_labels: Tuple[str, ...]
    inorder_labels: Tuple[str, ...]
    postorder_labels: Tuple[str, ...]
    level_order_labels: Tuple[str, ...]

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def max_depth(self) -> int:
        return max((int(node.depth) for node in self.nodes), default=0)


@dataclass(frozen=True)
class RenderedBinaryTreeNode:
    """Rendered node projection for one binary-tree diagram."""

    node_id: str
    label: str
    parent_label: str | None
    left_label: str | None
    right_label: str | None
    depth: int
    center_xy: Tuple[int, int]
    bbox_xyxy: Tuple[int, int, int, int]


@dataclass(frozen=True)
class RenderedBinaryTreeEdge:
    """Rendered parent-child edge projection for one binary-tree diagram."""

    edge_id: str
    parent_label: str
    child_label: str
    child_side: str
    segment_px: Tuple[Tuple[int, int], Tuple[int, int]]
    connector_path_px: Tuple[Tuple[int, int], ...]
    connector_style_variant: str


@dataclass(frozen=True)
class RenderedBinaryTreeScene:
    """Full render output for one binary-tree diagram."""

    image: Image.Image
    panel_geometry: Dict[str, Any]
    nodes: Tuple[RenderedBinaryTreeNode, ...]
    edges: Tuple[RenderedBinaryTreeEdge, ...]
    scene_variant: str
    connector_style_variant: str
    resolved_label_font_size_px: int
    resolved_label_stroke_width_px: int


@dataclass(frozen=True)
class RelationSelection:
    """One sampled binary-tree relation query."""

    query_labels: Tuple[str, ...]
    answer_label: str
    annotation_labels: Tuple[str, ...]
    query_node_ids: Tuple[str, ...]
    answer_node_id: str
    answer_scope: str = ""


@dataclass(frozen=True)
class OperationSelection:
    """One sampled operation query over a numeric binary tree."""

    target_key: int | None
    answer_label: str
    annotation_labels: Tuple[str, ...]
    query_node_ids: Tuple[str, ...]
    answer_node_id: str
    operation_kind: str


__all__ = [
    "CONNECTOR_STYLE_VARIANTS",
    "SCENE_ID",
    "SCENE_VARIANTS",
    "BinaryTreeNode",
    "BinaryTreeSample",
    "OperationSelection",
    "RelationSelection",
    "RenderedBinaryTreeEdge",
    "RenderedBinaryTreeNode",
    "RenderedBinaryTreeScene",
]
