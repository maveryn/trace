"""Trace payload helpers for rendered pair-grid icon scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import Image

from .rendering import panel_geometry_to_trace, render_two_panel_icon_pair_grid_scene
from .state import IconPairSpec


@dataclass(frozen=True)
class PairGridRenderPayload:
    """Rendered image plus neutral trace payloads for a pair-grid scene."""

    image: Image.Image
    panel_geometry: Dict[str, Any]
    reference_pair: Dict[str, Any]
    scene_cells: Tuple[Dict[str, Any], ...]


def _pair_payload(
    rendered_pair: Any,
    *,
    panel: str,
    extra_fields: Mapping[str, Any] | None = None,
    is_match: bool | None = None,
    index: int | None = None,
) -> Dict[str, Any]:
    """Serialize rendered pair metadata without task-specific routing."""

    payload = {
        "panel": str(panel),
        "icon_id": str(rendered_pair.icon_id),
        "transform_id": str(rendered_pair.transform_id),
        "tint_rgb": list(rendered_pair.tint_rgb),
        "left_tint_rgb": list(rendered_pair.left_tint_rgb),
        "right_tint_rgb": list(rendered_pair.right_tint_rgb),
        "left_size_scale": float(rendered_pair.left_size_scale),
        "right_size_scale": float(rendered_pair.right_size_scale),
        "left_bbox_xyxy": list(rendered_pair.left_bbox_xyxy),
        "right_bbox_xyxy": list(rendered_pair.right_bbox_xyxy),
        "left_noise_edits": [dict(edit) for edit in rendered_pair.left_noise_edits],
        "left_noise_seed": None if rendered_pair.left_noise_seed is None else int(rendered_pair.left_noise_seed),
        "right_noise_edits": [dict(edit) for edit in rendered_pair.right_noise_edits],
        "right_noise_seed": None if rendered_pair.right_noise_seed is None else int(rendered_pair.right_noise_seed),
    }
    if hasattr(rendered_pair, "label"):
        payload["label"] = str(rendered_pair.label)
    if hasattr(rendered_pair, "cell_bbox_xyxy"):
        payload["cell_bbox_xyxy"] = list(rendered_pair.cell_bbox_xyxy)
    if is_match is not None:
        payload["is_match"] = bool(is_match)
    if index is not None:
        payload["index"] = int(index)
    if extra_fields:
        payload.update({str(key): value for key, value in extra_fields.items()})
    return payload


def render_pair_grid_payload(
    *,
    reference_pair: IconPairSpec,
    scene_pairs: Sequence[IconPairSpec],
    scene_labels: Sequence[str],
    render_params: Mapping[str, Any],
    matching_labels: Sequence[str] = (),
    reference_extra: Mapping[str, Any] | None = None,
    cell_extra_by_label: Mapping[str, Mapping[str, Any]] | None = None,
) -> PairGridRenderPayload:
    """Render the pair-grid scene and return neutral trace payloads."""

    rendered = render_two_panel_icon_pair_grid_scene(
        reference_pair=reference_pair,
        scene_pairs=scene_pairs,
        scene_labels=scene_labels,
        canvas_width=int(render_params["canvas_width"]),
        canvas_height=int(render_params["canvas_height"]),
        reference_panel_width_px=int(render_params["reference_panel_width_px"]),
        outer_margin_px=int(render_params["outer_margin_px"]),
        panel_gap_px=int(render_params["panel_gap_px"]),
        panel_padding_px=int(render_params["panel_padding_px"]),
        panel_corner_radius_px=int(render_params["panel_corner_radius_px"]),
        cell_padding_px=int(render_params["cell_padding_px"]),
        scene_icon_size_min_px=int(render_params["scene_icon_size_min_px"]),
        scene_icon_size_max_px=int(render_params["scene_icon_size_max_px"]),
        reference_icon_size_px=int(render_params["reference_icon_size_px"]),
        pair_arrow_stroke_px=int(render_params["pair_arrow_stroke_px"]),
        cell_label_font_size_px=int(render_params["cell_label_font_size_px"]),
        panel_title_font_size_px=int(render_params["panel_title_font_size_px"]),
        background_rgb=tuple(int(v) for v in render_params["background_color_rgb"]),
        panel_fill_rgb=tuple(int(v) for v in render_params["panel_fill_rgb"]),
        panel_border_rgb=tuple(int(v) for v in render_params["panel_border_rgb"]),
        title_color_rgb=tuple(int(v) for v in render_params["header_text_rgb"]),
        cell_border_rgb=tuple(int(v) for v in render_params["cell_border_rgb"]),
        cell_label_color_rgb=tuple(int(v) for v in render_params["cell_label_color_rgb"]),
        cell_label_stroke_rgb=tuple(int(v) for v in render_params["cell_label_stroke_rgb"]),
        cell_label_stroke_width_px=1,
        arrow_color_rgb=tuple(int(v) for v in render_params["arrow_color_rgb"]),
        icon_canvas_style=render_params.get("_icon_canvas_style_object"),
    )
    matching_set = {str(label) for label in matching_labels}
    cell_extra_by_label = cell_extra_by_label or {}
    scene_cells = tuple(
        _pair_payload(
            cell,
            panel="scene",
            extra_fields=cell_extra_by_label.get(str(cell.label), {}),
            is_match=str(cell.label) in matching_set,
            index=int(index),
        )
        for index, cell in enumerate(rendered.scene_cells)
    )
    return PairGridRenderPayload(
        image=rendered.image,
        panel_geometry=panel_geometry_to_trace(rendered.layout),
        reference_pair=_pair_payload(
            rendered.reference_pair,
            panel="reference",
            extra_fields=reference_extra or {},
        ),
        scene_cells=scene_cells,
    )


def selected_scene_cell_bbox(
    *,
    scene_cells: Sequence[Mapping[str, Any]],
    selected_label: str,
) -> list[int]:
    """Return the rendered Scene-cell bbox for one selected label."""

    for cell in scene_cells:
        if str(cell.get("label")) == str(selected_label):
            bbox = cell.get("cell_bbox_xyxy")
            if not isinstance(bbox, Sequence) or len(bbox) != 4:
                raise RuntimeError(f"invalid Scene cell bbox for label {selected_label}: {bbox}")
            return [int(round(float(value))) for value in bbox]
    raise RuntimeError(f"missing Scene cell for selected label {selected_label}")


__all__ = [
    "PairGridRenderPayload",
    "render_pair_grid_payload",
    "selected_scene_cell_bbox",
]
