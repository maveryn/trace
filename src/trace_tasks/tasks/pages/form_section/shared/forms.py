"""Structured-document defaults and render parameters for form-section pages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import normalize_positive_weights, weighted_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.render_variation import resolve_layout_jitter, resolve_render_int, resolve_render_rgb


SUPPORTED_DOCUMENT_SCENE_VARIANTS: Tuple[str, ...] = (
    "form_sheet",
    "invoice_sheet",
    "receipt_sheet",
)
SUPPORTED_DOCUMENT_LAYOUT_MODES: Tuple[str, ...] = (
    "centered",
    "left_weighted",
    "right_weighted",
    "upper_left",
    "upper_right",
    "lower_left",
    "lower_right",
)
DEFAULT_DOCUMENT_LAYOUT_MODE_WEIGHTS: Dict[str, float] = {
    "centered": 0.22,
    "left_weighted": 0.22,
    "right_weighted": 0.22,
    "upper_left": 0.12,
    "upper_right": 0.12,
    "lower_left": 0.05,
    "lower_right": 0.05,
}
DOCUMENT_SCENE_TITLES: Dict[str, str] = {
    "form_sheet": "Application Form",
    "invoice_sheet": "Invoice",
    "receipt_sheet": "Receipt",
}


@dataclass(frozen=True)
class DocumentDefaults:
    """Stable fallback defaults for structured-document readout tasks."""

    canvas_width: int = 1280
    canvas_height: int = 920
    sheet_page_width_px: int = 960
    sheet_page_height_px: int = 760
    receipt_page_width_px: int = 520
    receipt_page_height_px: int = 760
    page_shadow_offset_px: int = 14
    page_corner_radius_px: int = 20
    field_corner_radius_px: int = 14
    page_outline_width_px: int = 2
    field_outline_width_px: int = 2
    title_font_size_px: int = 42
    section_font_size_px: int = 24
    label_font_size_px: int = 22
    value_font_size_px: int = 28
    page_fill_rgb: Tuple[int, int, int] = (251, 250, 246)
    page_outline_rgb: Tuple[int, int, int] = (120, 126, 138)
    page_shadow_rgb: Tuple[int, int, int] = (221, 224, 230)
    field_fill_rgb: Tuple[int, int, int] = (255, 255, 255)
    field_outline_rgb: Tuple[int, int, int] = (202, 207, 214)
    label_fill_rgb: Tuple[int, int, int] = (60, 66, 76)
    label_stroke_rgb: Tuple[int, int, int] = (255, 255, 255)
    value_fill_rgb: Tuple[int, int, int] = (24, 29, 35)
    divider_rgb: Tuple[int, int, int] = (214, 217, 224)


@dataclass(frozen=True)
class DocumentRenderParams:
    """Resolved rendering parameters for structured-document scenes."""

    canvas_width: int
    canvas_height: int
    sheet_page_width_px: int
    sheet_page_height_px: int
    receipt_page_width_px: int
    receipt_page_height_px: int
    page_shadow_offset_px: int
    page_corner_radius_px: int
    field_corner_radius_px: int
    page_outline_width_px: int
    field_outline_width_px: int
    title_font_size_px: int
    section_font_size_px: int
    label_font_size_px: int
    value_font_size_px: int
    page_fill_rgb: Tuple[int, int, int]
    page_outline_rgb: Tuple[int, int, int]
    page_shadow_rgb: Tuple[int, int, int]
    field_fill_rgb: Tuple[int, int, int]
    field_outline_rgb: Tuple[int, int, int]
    label_fill_rgb: Tuple[int, int, int]
    label_stroke_rgb: Tuple[int, int, int]
    value_fill_rgb: Tuple[int, int, int]
    divider_rgb: Tuple[int, int, int]
    document_layout_mode: str
    document_layout_mode_meta: Dict[str, Any]
    layout_jitter_meta: Dict[str, Any]


def resolve_document_layout_mode(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    instance_seed: int | None = None,
) -> Tuple[str, Dict[str, Any]]:
    """Resolve the nonsemantic document placement mode for one rendered page."""

    supported = [str(item) for item in SUPPORTED_DOCUMENT_LAYOUT_MODES]
    supported_set = set(supported)
    explicit = params.get(
        "document_layout_mode",
        params.get(
            "page_layout_mode",
            group_default(
                render_defaults,
                "document_layout_mode",
                group_default(render_defaults, "page_layout_mode", None),
            ),
        ),
    )
    if explicit is not None:
        selected = str(explicit).strip()
        if selected not in supported_set:
            raise ValueError(f"unsupported document_layout_mode: {selected}")
        return selected, {
            "explicit": True,
            "probabilities": {key: (1.0 if key == selected else 0.0) for key in supported},
            "supported_modes": list(supported),
        }

    raw_weights = params.get("document_layout_mode_weights")
    if raw_weights is None:
        raw_weights = params.get("page_layout_mode_weights")
    if raw_weights is None:
        raw_weights = group_default(render_defaults, "document_layout_mode_weights", None)
    if raw_weights is None:
        raw_weights = group_default(render_defaults, "page_layout_mode_weights", DEFAULT_DOCUMENT_LAYOUT_MODE_WEIGHTS)
    if not isinstance(raw_weights, Mapping):
        raise ValueError("document_layout_mode_weights must be a mapping when provided")
    weights = {
        str(key): float(value)
        for key, value in raw_weights.items()
        if str(key) in supported_set
    }
    probabilities = normalize_positive_weights(weights, default_keys=supported)
    rng = spawn_rng(0 if instance_seed is None else int(instance_seed), "pages.document.layout_mode")
    selected = weighted_choice(rng, probabilities, sort_keys=True)
    return str(selected), {
        "explicit": False,
        "probabilities": {str(key): float(value) for key, value in sorted(probabilities.items())},
        "supported_modes": list(supported),
    }


def resolve_document_render_params(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    defaults: DocumentDefaults = DocumentDefaults(),
    instance_seed: int | None = None,
) -> DocumentRenderParams:
    """Resolve render parameters for structured-document scenes."""

    def _int(key: str, fallback: int) -> int:
        return resolve_render_int(
            params,
            render_defaults,
            key,
            fallback,
            instance_seed=instance_seed,
            namespace="pages.document",
        )

    def _rgb(key: str, fallback: Sequence[int]) -> Tuple[int, int, int]:
        return resolve_render_rgb(
            params,
            render_defaults,
            key,
            fallback,
            instance_seed=instance_seed,
            namespace="pages.document",
        )

    layout_jitter_meta = resolve_layout_jitter(
        params,
        render_defaults,
        instance_seed=instance_seed,
        namespace="pages.document.layout",
    )
    document_layout_mode, document_layout_mode_meta = resolve_document_layout_mode(
        params,
        render_defaults=render_defaults,
        instance_seed=instance_seed,
    )

    return DocumentRenderParams(
        canvas_width=_int("canvas_width", defaults.canvas_width),
        canvas_height=_int("canvas_height", defaults.canvas_height),
        sheet_page_width_px=_int("sheet_page_width_px", defaults.sheet_page_width_px),
        sheet_page_height_px=_int("sheet_page_height_px", defaults.sheet_page_height_px),
        receipt_page_width_px=_int("receipt_page_width_px", defaults.receipt_page_width_px),
        receipt_page_height_px=_int("receipt_page_height_px", defaults.receipt_page_height_px),
        page_shadow_offset_px=_int("page_shadow_offset_px", defaults.page_shadow_offset_px),
        page_corner_radius_px=_int("page_corner_radius_px", defaults.page_corner_radius_px),
        field_corner_radius_px=_int("field_corner_radius_px", defaults.field_corner_radius_px),
        page_outline_width_px=_int("page_outline_width_px", defaults.page_outline_width_px),
        field_outline_width_px=_int("field_outline_width_px", defaults.field_outline_width_px),
        title_font_size_px=_int("title_font_size_px", defaults.title_font_size_px),
        section_font_size_px=_int("section_font_size_px", defaults.section_font_size_px),
        label_font_size_px=_int("label_font_size_px", defaults.label_font_size_px),
        value_font_size_px=_int("value_font_size_px", defaults.value_font_size_px),
        page_fill_rgb=_rgb("page_fill_rgb", defaults.page_fill_rgb),
        page_outline_rgb=_rgb("page_outline_rgb", defaults.page_outline_rgb),
        page_shadow_rgb=_rgb("page_shadow_rgb", defaults.page_shadow_rgb),
        field_fill_rgb=_rgb("field_fill_rgb", defaults.field_fill_rgb),
        field_outline_rgb=_rgb("field_outline_rgb", defaults.field_outline_rgb),
        label_fill_rgb=_rgb("label_fill_rgb", defaults.label_fill_rgb),
        label_stroke_rgb=_rgb("label_stroke_rgb", defaults.label_stroke_rgb),
        value_fill_rgb=_rgb("value_fill_rgb", defaults.value_fill_rgb),
        divider_rgb=_rgb("divider_rgb", defaults.divider_rgb),
        document_layout_mode=str(document_layout_mode),
        document_layout_mode_meta=dict(document_layout_mode_meta),
        layout_jitter_meta=dict(layout_jitter_meta),
    )


__all__ = [
    "DEFAULT_DOCUMENT_LAYOUT_MODE_WEIGHTS",
    "DOCUMENT_SCENE_TITLES",
    "DocumentDefaults",
    "DocumentRenderParams",
    "SUPPORTED_DOCUMENT_LAYOUT_MODES",
    "SUPPORTED_DOCUMENT_SCENE_VARIANTS",
    "resolve_document_layout_mode",
    "resolve_document_render_params",
]
