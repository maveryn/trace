"""State containers and semantic resources for process-flow page scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]
Color = Tuple[int, int, int]

SCENE_VARIANTS: Tuple[str, ...] = (
    "incident_response",
    "editorial_review",
    "order_fulfillment",
    "model_release",
    "lab_sample",
    "support_ticket",
)
LAYOUT_VARIANTS: Tuple[str, ...] = (
    "horizontal_swimlane",
)
STYLE_VARIANTS: Tuple[str, ...] = (
    "blueprint",
    "pastel_cards",
    "graphite",
    "warm_memo",
)

CONDITION_POOLS: Tuple[Tuple[str, str], ...] = (
    ("yes", "no"),
    ("pass", "fail"),
    ("approve", "revise"),
    ("ready", "hold"),
    ("auto", "manual"),
    ("ship", "return"),
    ("accept", "reject"),
)
STATUS_POOL: Tuple[str, ...] = (
    "Ready",
    "Hold",
    "Queued",
    "Review",
    "Done",
    "Retry",
)
ROLE_DESCRIPTIONS: Dict[str, str] = {
    "process": "process steps",
    "decision": "decision steps",
    "review": "review steps",
    "data": "data steps",
    "output": "output steps",
}
SHAPE_DESCRIPTIONS: Dict[str, str] = {
    "diamond": "decision diamonds",
    "rounded": "rounded process boxes",
    "parallelogram": "slanted data boxes",
    "ellipse": "terminal ovals",
}

CONTEXTS: Dict[str, Dict[str, Any]] = {
    "incident_response": {
        "title": "Incident response flow",
        "lanes": ["Monitor", "Triage", "Security", "Ops", "Comms"],
        "steps": [
            "Alert",
            "Classify",
            "Scope",
            "Contain",
            "Patch",
            "Notify",
            "Review",
            "Close",
            "Escalate",
            "Snapshot",
            "Approve",
            "Archive",
            "Restore",
            "Report",
        ],
    },
    "editorial_review": {
        "title": "Publishing workflow",
        "lanes": ["Author", "Editor", "Legal", "Design", "Release"],
        "steps": [
            "Draft",
            "Edit",
            "Fact Check",
            "Layout",
            "Proof",
            "Revise",
            "Approve",
            "Publish",
            "Caption",
            "Archive",
            "Assign",
            "Review",
            "Upload",
            "Notify",
        ],
    },
    "order_fulfillment": {
        "title": "Order fulfillment path",
        "lanes": ["Sales", "Inventory", "Packing", "Carrier", "Billing"],
        "steps": [
            "Order",
            "Verify",
            "Reserve",
            "Pick",
            "Pack",
            "Label",
            "Dispatch",
            "Invoice",
            "Backorder",
            "Refund",
            "Inspect",
            "Confirm",
            "Ship",
            "Close",
        ],
    },
    "model_release": {
        "title": "Model release pipeline",
        "lanes": ["Data", "Training", "Eval", "Safety", "Deploy"],
        "steps": [
            "Collect",
            "Clean",
            "Train",
            "Score",
            "Audit",
            "Tune",
            "Approve",
            "Package",
            "Shadow",
            "Launch",
            "Rollback",
            "Log",
            "Monitor",
            "Signoff",
        ],
    },
    "lab_sample": {
        "title": "Lab sample workflow",
        "lanes": ["Intake", "Prep", "Assay", "Review", "Records"],
        "steps": [
            "Receive",
            "Barcode",
            "Aliquot",
            "Spin",
            "Assay",
            "Repeat",
            "Validate",
            "Record",
            "Store",
            "Reject",
            "Release",
            "Notify",
            "Archive",
            "Seal",
        ],
    },
    "support_ticket": {
        "title": "Support ticket workflow",
        "lanes": ["Customer", "Support", "Specialist", "QA", "Billing"],
        "steps": [
            "Request",
            "Triage",
            "Lookup",
            "Assign",
            "Diagnose",
            "Escalate",
            "Patch",
            "Verify",
            "Reply",
            "Credit",
            "Close",
            "Survey",
            "Reopen",
            "Document",
        ],
    },
}

STYLE_PALETTES: Dict[str, Dict[str, Any]] = {
    "blueprint": {
        "panel_fill": (246, 250, 255),
        "panel_border": (44, 72, 102),
        "title": (25, 41, 65),
        "lane_header": (219, 234, 249),
        "lane_fills": [
            (237, 246, 255),
            (245, 251, 255),
            (231, 241, 251),
            (240, 247, 253),
            (235, 243, 250),
        ],
        "node_fill": (255, 255, 255),
        "node_border": (45, 77, 112),
        "edge": (46, 78, 112),
        "text": (20, 30, 45),
        "muted": (86, 100, 116),
    },
    "pastel_cards": {
        "panel_fill": (255, 253, 248),
        "panel_border": (135, 103, 84),
        "title": (65, 43, 33),
        "lane_header": (250, 230, 207),
        "lane_fills": [
            (253, 244, 232),
            (244, 250, 236),
            (236, 247, 249),
            (249, 239, 248),
            (245, 242, 233),
        ],
        "node_fill": (255, 255, 252),
        "node_border": (123, 94, 75),
        "edge": (105, 86, 76),
        "text": (42, 34, 28),
        "muted": (114, 92, 78),
    },
    "graphite": {
        "panel_fill": (244, 245, 244),
        "panel_border": (58, 64, 70),
        "title": (32, 37, 42),
        "lane_header": (226, 229, 231),
        "lane_fills": [
            (247, 248, 248),
            (238, 241, 242),
            (250, 250, 247),
            (242, 244, 246),
            (246, 243, 241),
        ],
        "node_fill": (255, 255, 255),
        "node_border": (69, 74, 82),
        "edge": (63, 70, 78),
        "text": (25, 28, 31),
        "muted": (87, 93, 100),
    },
    "warm_memo": {
        "panel_fill": (255, 250, 239),
        "panel_border": (111, 83, 108),
        "title": (53, 37, 66),
        "lane_header": (239, 223, 241),
        "lane_fills": [
            (255, 247, 232),
            (247, 239, 250),
            (238, 247, 240),
            (248, 240, 232),
            (239, 245, 249),
        ],
        "node_fill": (255, 255, 252),
        "node_border": (99, 75, 111),
        "edge": (88, 78, 105),
        "text": (38, 31, 48),
        "muted": (104, 89, 112),
    },
}
BADGE_COLORS: Dict[str, Color] = {
    "Ready": (214, 238, 222),
    "Hold": (249, 224, 196),
    "Queued": (218, 230, 249),
    "Review": (238, 224, 249),
    "Done": (206, 234, 221),
    "Retry": (247, 216, 213),
}
ROLE_FILL_ADJUST: Dict[str, Color] = {
    "start": (226, 242, 235),
    "process": (255, 255, 255),
    "decision": (255, 249, 217),
    "review": (239, 236, 255),
    "data": (230, 245, 252),
    "output": (230, 239, 223),
}


@dataclass(frozen=True)
class ProcessFlowDefaults:
    """Fallback knobs for lane-based process-flow diagrams."""

    node_count_min: int = 10
    node_count_max: int = 10
    lane_count_min: int = 3
    lane_count_max: int = 3
    canvas_width: int = 1200
    canvas_height: int = 900
    outer_margin_px: int = 46
    panel_padding_px: int = 24
    panel_corner_radius_px: int = 24
    title_band_height_px: int = 68
    lane_header_height_px: int = 42
    node_width_px: int = 146
    node_height_px: int = 64
    node_corner_radius_px: int = 16
    node_border_width_px: int = 3
    edge_width_px: int = 3
    arrow_head_length_px: int = 15
    arrow_head_width_px: int = 13
    title_font_size_px: int = 30
    lane_font_size_px: int = 18
    node_label_font_size_px: int = 17
    badge_font_size_px: int = 12
    edge_label_font_size_px: int = 15


@dataclass(frozen=True)
class ProcessFlowRenderParams:
    """Resolved render parameters for one process-flow diagram."""

    canvas_width: int
    canvas_height: int
    outer_margin_px: int
    panel_padding_px: int
    panel_corner_radius_px: int
    title_band_height_px: int
    lane_header_height_px: int
    node_width_px: int
    node_height_px: int
    node_corner_radius_px: int
    node_border_width_px: int
    edge_width_px: int
    arrow_head_length_px: int
    arrow_head_width_px: int
    title_font_size_px: int
    lane_font_size_px: int
    node_label_font_size_px: int
    badge_font_size_px: int
    edge_label_font_size_px: int
    layout_jitter_meta: Dict[str, Any]


@dataclass(frozen=True)
class ProcessFlowSceneCase:
    """Sampled scene graph and resolved layout before final rendering."""

    scene_variant: str
    layout_variant: str
    style_variant: str
    flow_family: str
    lane_pattern_index: int
    scene_title: str
    lanes: Tuple[str, ...]
    nodes: Tuple[Dict[str, Any], ...]
    edges: Tuple[Dict[str, Any], ...]
    condition_map: Dict[str, str]
    panel_bbox: BBox
    title_bbox: BBox
    content_bbox: BBox
    lane_bboxes: Dict[str, List[float]]
    render_params: ProcessFlowRenderParams
    scene_variant_probabilities: Dict[str, float]
    layout_variant_probabilities: Dict[str, float]
    style_variant_probabilities: Dict[str, float]


@dataclass(frozen=True)
class RenderedProcessFlow:
    """Rendered process-flow image and pixel-space projection maps."""

    image: Image.Image
    render_map: Dict[str, Any]
