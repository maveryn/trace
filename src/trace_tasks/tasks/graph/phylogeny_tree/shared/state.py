"""State records for phylogeny-tree graph scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


SUPPORTED_PHYLOGENY_SCENE_VARIANTS: Tuple[str, ...] = (
    "rectangular_cladogram",
    "diagonal_cladogram",
    "paper_cladogram",
)
PHYLOGENY_TAXON_LABEL_POOL: Tuple[str, ...] = tuple("ABCDEFGHIJKLM")


@dataclass(frozen=True)
class PhylogenyNode:
    """One node in a rooted phylogeny tree."""

    node_id: str
    parent_id: str | None
    child_ids: Tuple[str, ...]
    leaf_label: str | None
    depth: int

    @property
    def is_leaf(self) -> bool:
        return self.leaf_label is not None


@dataclass(frozen=True)
class PhylogenySample:
    """Trace-ready rooted phylogeny sample."""

    nodes: Tuple[PhylogenyNode, ...]
    root_id: str
    leaf_labels: Tuple[str, ...]
    canonical_signature: Tuple[Tuple[str, ...], ...]

    @property
    def leaf_count(self) -> int:
        return len(self.leaf_labels)

    @property
    def max_depth(self) -> int:
        return max((int(node.depth) for node in self.nodes), default=0)


@dataclass(frozen=True)
class RenderedPhylogenyNode:
    """Rendered node projection for one phylogeny diagram."""

    node_id: str
    leaf_label: str | None
    descendant_leaf_labels: Tuple[str, ...]
    center_xy: Tuple[int, int]
    bbox_xyxy: Tuple[int, int, int, int]
    label_bbox_xyxy: Tuple[int, int, int, int] | None
    is_leaf: bool


@dataclass(frozen=True)
class RenderedPhylogenyEdge:
    """Rendered parent-child branch projection."""

    edge_id: str
    parent_id: str
    child_id: str
    path_px: Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class RenderedPhylogenyScene:
    """Full render output for one phylogeny diagram."""

    image: Image.Image
    panel_geometry: Dict[str, Any]
    nodes: Tuple[RenderedPhylogenyNode, ...]
    edges: Tuple[RenderedPhylogenyEdge, ...]
    scene_variant: str
    resolved_label_font_size_px: int
    resolved_label_stroke_width_px: int
    option_panel_bboxes: Dict[str, List[float]]


def _node_map(sample: PhylogenySample) -> Dict[str, PhylogenyNode]:
    return {str(node.node_id): node for node in sample.nodes}


def _leaf_node_id_by_label(sample: PhylogenySample) -> Dict[str, str]:
    return {
        str(node.leaf_label): str(node.node_id)
        for node in sample.nodes
        if node.leaf_label is not None
    }


__all__ = [
    "PHYLOGENY_TAXON_LABEL_POOL",
    "SUPPORTED_PHYLOGENY_SCENE_VARIANTS",
    "PhylogenyNode",
    "PhylogenySample",
    "RenderedPhylogenyEdge",
    "RenderedPhylogenyNode",
    "RenderedPhylogenyScene",
    "_leaf_node_id_by_label",
    "_node_map",
]
