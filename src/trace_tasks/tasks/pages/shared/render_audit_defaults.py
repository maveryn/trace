"""Pages-domain rendering defaults for font and non-answer context text."""

from __future__ import annotations

import random
from functools import wraps
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence, Tuple

from PIL import ImageDraw

from ....core.seed import spawn_rng
from ....core.scene_config import get_scene_defaults
from ...base import TaskOutput
from ...shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from ...shared.context_text_assets import sample_context_text
from ...shared.font_assets import font_asset_version, font_role_trace, sample_font_family
from ...shared.text_rendering import load_font, resolve_text_stroke_fill, temporary_default_font_family
from ...shared.visual_style.context_layer import ContextTextElement, context_text_layer_metadata
from ...shared.text_legibility import draw_text_traced
from .information_style import PagesInformationStyle, resolve_pages_information_style


BBox = Tuple[float, float, float, float]
RGB = Tuple[int, int, int]
_CONTEXT_PROFILE_WEIGHTS = {
    "dense_clean_minimal": {
        "clean": 0.70,
        "minimal": 0.30,
        "paragraph_box": 0.0,
    },
    "report_paragraph": {
        "clean": 0.30,
        "minimal": 0.40,
        "paragraph_box": 0.30,
    },
}
_DEFAULT_CONTEXT_PROFILE = "dense_clean_minimal"
_CONTEXT_MODE_CONFIG = {
    "clean": {"simple_min": 0, "simple_max": 0, "paragraph_min": 0, "paragraph_max": 0},
    "minimal": {"simple_min": 2, "simple_max": 4, "paragraph_min": 0, "paragraph_max": 0},
    "paragraph_box": {"simple_min": 1, "simple_max": 3, "paragraph_min": 1, "paragraph_max": 2},
}
_CONTEXT_MODE_ALIASES = {
    "light": "minimal",
    "light_context": "minimal",
    "large": "paragraph_box",
    "large_distractor": "paragraph_box",
    "paragraph": "paragraph_box",
    "sidebar": "paragraph_box",
    "side_note": "paragraph_box",
    "one_side_note": "paragraph_box",
    "two_side_notes": "paragraph_box",
    "right_sidebar": "paragraph_box",
    "left_sidebar": "paragraph_box",
    "bottom_band": "paragraph_box",
}


def wrap_pages_generation(
    original_generate: Callable[..., TaskOutput],
    *,
    task_id: str,
    scene_id: str,
) -> Callable[..., TaskOutput]:
    """Wrap one pages task generator with domain-wide render audit defaults."""

    @wraps(original_generate)
    def _generate_with_pages_render_defaults(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        font_family = resolve_pages_default_font_family(
            task_id=str(task_id),
            scene_id=str(scene_id),
            instance_seed=int(instance_seed),
            params=params,
        )
        with temporary_default_font_family(str(font_family)):
            output = original_generate(self, int(instance_seed), params=params, max_attempts=int(max_attempts))
        annotate_pages_font_assets(
            output,
            font_family=str(font_family),
            task_id=str(task_id),
            scene_id=str(scene_id),
        )
        ensure_pages_information_style(
            output,
            instance_seed=int(instance_seed),
            params=params,
            task_id=str(task_id),
            scene_id=str(scene_id),
        )
        add_pages_safe_context_text(
            output,
            instance_seed=int(instance_seed),
            params=params,
            task_id=str(task_id),
            scene_id=str(scene_id),
        )
        return output

    return _generate_with_pages_render_defaults


def wrap_pages_scene_generation(
    original_generate: Callable[..., TaskOutput],
    *,
    task_id: str,
    scene_id: str,
) -> Callable[..., TaskOutput]:
    """Wrap one migrated pages scene task with render audit defaults."""

    @wraps(original_generate)
    def _generate_with_pages_render_defaults(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        font_family = resolve_pages_scene_default_font_family(
            task_id=str(task_id),
            scene_id=str(scene_id),
            instance_seed=int(instance_seed),
            params=params,
        )
        with temporary_default_font_family(str(font_family)):
            output = original_generate(self, int(instance_seed), params=params, max_attempts=int(max_attempts))
        annotate_pages_scene_font_assets(
            output,
            font_family=str(font_family),
            task_id=str(task_id),
            scene_id=str(scene_id),
        )
        ensure_pages_information_style(
            output,
            instance_seed=int(instance_seed),
            params=params,
            task_id=str(task_id),
            scene_id=str(scene_id),
        )
        add_pages_scene_safe_context_text(
            output,
            instance_seed=int(instance_seed),
            params=params,
            task_id=str(task_id),
            scene_id=str(scene_id),
        )
        return output

    return _generate_with_pages_render_defaults


def resolve_pages_default_font_family(
    *,
    task_id: str,
    scene_id: str,
    instance_seed: int,
    params: Mapping[str, Any] | None,
) -> str:
    """Sample one default page font family for read-required page text."""

    return _resolve_pages_default_font_family_for_route(
        task_id=str(task_id),
        route_id=str(scene_id),
        config_kind="scene_id",
        instance_seed=int(instance_seed),
        params=params,
    )


def resolve_pages_scene_default_font_family(
    *,
    task_id: str,
    scene_id: str,
    instance_seed: int,
    params: Mapping[str, Any] | None,
) -> str:
    """Sample one default page font family for a migrated pages scene task."""

    return _resolve_pages_default_font_family_for_route(
        task_id=str(task_id),
        route_id=str(scene_id),
        config_kind="scene",
        instance_seed=int(instance_seed),
        params=params,
    )


def _resolve_pages_default_font_family_for_route(
    *,
    task_id: str,
    route_id: str,
    config_kind: str,
    instance_seed: int,
    params: Mapping[str, Any] | None,
) -> str:
    """Sample one default page font family for a config-routing unit."""

    resolved_params = _resolve_context_text_params(
        params=params,
        task_id=str(task_id),
        route_id=str(route_id),
        config_kind=str(config_kind),
    )
    if "pages_font_family" not in resolved_params and "font_family" in resolved_params:
        resolved_params["pages_font_family"] = resolved_params["font_family"]
    if "pages_font_family_weights" not in resolved_params and "font_family_weights" in resolved_params:
        resolved_params["pages_font_family_weights"] = resolved_params["font_family_weights"]
    return sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{task_id}.{route_id}.pages_default_font",
        params=resolved_params,
        explicit_key="pages_font_family",
        weights_key="pages_font_family_weights",
    )


def annotate_pages_font_assets(
    output: TaskOutput,
    *,
    font_family: str,
    task_id: str,
    scene_id: str,
) -> None:
    """Record the pages default font family in render metadata."""

    _annotate_pages_font_assets_for_route(
        output,
        font_family=str(font_family),
        task_id=str(task_id),
        route_kind="scene_id",
        route_id=str(scene_id),
    )


def annotate_pages_scene_font_assets(
    output: TaskOutput,
    *,
    font_family: str,
    task_id: str,
    scene_id: str,
) -> None:
    """Record the pages default font family in source-layout scene render metadata."""

    _annotate_pages_font_assets_for_route(
        output,
        font_family=str(font_family),
        task_id=str(task_id),
        route_kind="scene",
        route_id=str(scene_id),
    )


def _annotate_pages_font_assets_for_route(
    output: TaskOutput,
    *,
    font_family: str,
    task_id: str,
    route_kind: str,
    route_id: str,
) -> None:
    """Record the pages default font family in render metadata."""

    trace_payload = output.trace_payload if isinstance(output.trace_payload, MutableMapping) else None
    if trace_payload is None:
        return
    render_spec = trace_payload.setdefault("render_spec", {})
    if not isinstance(render_spec, MutableMapping):
        return
    font_assets = render_spec.setdefault("font_assets", {})
    if not isinstance(font_assets, MutableMapping):
        font_assets = {}
        render_spec["font_assets"] = font_assets
    font_assets.setdefault("asset_version", font_asset_version())
    font_assets["pages_default_font_family"] = str(font_family)
    font_assets["pages_default_font_role"] = "readout"
    font_assets["pages_default_font_trace"] = font_role_trace(str(font_family), role="readout")
    font_assets["pages_font_sampling_policy"] = "readout_pool_single_family_per_page_instance"
    font_assets["task_id"] = str(task_id)
    if str(route_kind) == "scene":
        font_assets["scene_id"] = str(route_id)
    else:
        font_assets["scene_id"] = str(route_id)


def ensure_pages_information_style(
    output: TaskOutput,
    *,
    instance_seed: int,
    params: Mapping[str, Any] | None,
    task_id: str,
    scene_id: str,
) -> None:
    """Ensure every pages task records the shared 25-treatment style metadata."""

    trace_payload = output.trace_payload if isinstance(output.trace_payload, MutableMapping) else None
    if trace_payload is None:
        return
    render_spec = trace_payload.setdefault("render_spec", {})
    if not isinstance(render_spec, MutableMapping):
        return

    existing_meta = _find_information_scene_style_meta(render_spec)
    if isinstance(existing_meta, Mapping) and existing_meta:
        render_spec.setdefault("information_scene_style", dict(existing_meta))
        _record_information_style_in_execution(
            trace_payload,
            style_meta=dict(existing_meta),
            policy="scene_renderer",
        )
        return

    try:
        style, style_meta = resolve_pages_information_style(
            instance_seed=int(instance_seed),
            params=params,
            scene_id=str(scene_id),
        )
    except Exception:
        return

    render_spec["information_scene_style"] = dict(style_meta)
    background_style = render_spec.setdefault("background_style", {})
    if isinstance(background_style, MutableMapping):
        background_style.setdefault("information_scene_style", dict(style_meta))
        background_style.setdefault("selected_style", f"information_scene_style:{style.style_pack}")
        background_style.setdefault("style_spec", dict(style_meta))
    render_spec["pages_information_style_policy"] = {
        "domain_wrapper": "fallback_outer_chrome",
        "scene_renderer_recorded_style": False,
        "task_id": str(task_id),
        "scene_id": str(scene_id),
    }
    _record_information_style_in_execution(
        trace_payload,
        style_meta=dict(style_meta),
        policy="domain_wrapper_fallback_outer_chrome",
    )
    _draw_information_style_fallback_chrome(output, trace_payload=trace_payload, style=style)


def _find_information_scene_style_meta(render_spec: Mapping[str, Any]) -> Mapping[str, Any] | None:
    candidates: list[Any] = [render_spec.get("information_scene_style")]
    for container_key in ("calendar_style", "background_style"):
        container = render_spec.get(str(container_key))
        if isinstance(container, Mapping):
            candidates.append(container.get("information_scene_style"))
            style_spec = container.get("style_spec")
            if isinstance(style_spec, Mapping) and str(style_spec.get("kind", "")) == "information_scene_style":
                candidates.append(style_spec)
    for candidate in candidates:
        if isinstance(candidate, Mapping) and str(candidate.get("kind", "")) == "information_scene_style":
            return candidate
    return None


def _record_information_style_in_execution(
    trace_payload: MutableMapping[str, Any],
    *,
    style_meta: Mapping[str, Any],
    policy: str,
) -> None:
    execution = trace_payload.setdefault("execution_trace", {})
    if isinstance(execution, MutableMapping):
        execution.setdefault("information_scene_treatment", str(style_meta.get("treatment", "")))
        execution.setdefault("information_scene_palette_id", str(style_meta.get("palette_id", "")))
        execution.setdefault("information_scene_style_pack", str(style_meta.get("style_pack", "")))
        execution.setdefault("information_scene_style_policy", str(policy))
    query_spec = trace_payload.setdefault("query_spec", {})
    if isinstance(query_spec, MutableMapping):
        query_params = query_spec.setdefault("params", {})
        if isinstance(query_params, MutableMapping):
            query_params.setdefault("information_scene_treatment", str(style_meta.get("treatment", "")))
            query_params.setdefault("information_scene_palette_id", str(style_meta.get("palette_id", "")))
            query_params.setdefault("information_scene_style_pack", str(style_meta.get("style_pack", "")))


def _draw_information_style_fallback_chrome(
    output: TaskOutput,
    *,
    trace_payload: Mapping[str, Any],
    style: PagesInformationStyle,
) -> None:
    """Draw non-semantic outer treatment cues for legacy Pages renderers."""

    if output.image is None:
        return
    image = output.image.convert("RGB").copy()
    width, height = image.size
    draw = ImageDraw.Draw(image)
    boxes = [
        _clip_bbox_float(box, width=int(width), height=int(height))
        for box in _collect_occupied_bboxes(trace_payload)
    ]
    boxes = [box for box in boxes if box is not None]
    if boxes:
        min_x = max(0.0, min(float(box[0]) for box in boxes))
        min_y = max(0.0, min(float(box[1]) for box in boxes))
        max_x = min(float(width), max(float(box[2]) for box in boxes))
        max_y = min(float(height), max(float(box[3]) for box in boxes))
        margin_regions = (
            (0.0, 0.0, float(width), min_y),
            (0.0, max_y, float(width), float(height)),
            (0.0, min_y, min_x, max_y),
            (max_x, min_y, float(width), max_y),
        )
        for region in margin_regions:
            if float(region[2] - region[0]) >= 10.0 and float(region[3] - region[1]) >= 10.0:
                draw.rectangle(region, fill=tuple(int(value) for value in style.canvas_rgb))

    inset = max(3, min(14, int(style.content_margin_px // 3)))
    frame_width = max(1, min(4, int(style.frame_width_px)))
    frame_color = style.accent_rgb if str(style.chrome_mode) == "accent_frame" else style.panel_border_rgb
    draw.rectangle(
        (inset, inset, int(width) - inset - 1, int(height) - inset - 1),
        outline=tuple(int(value) for value in frame_color),
        width=int(frame_width),
    )
    if str(style.surface_kind) in {"dashboard", "app_window", "console", "dark_board"}:
        band_h = max(5, min(14, int(style.title_band_height_px // 5)))
        draw.rectangle((inset, inset, int(width) - inset - 1, inset + band_h), fill=tuple(int(value) for value in style.canvas_accent_rgb))
    elif str(style.surface_kind) in {"infographic", "poster", "annotated"}:
        band_w = max(5, min(14, int(style.content_margin_px // 4)))
        draw.rectangle((inset, inset, inset + band_w, int(height) - inset - 1), fill=tuple(int(value) for value in style.canvas_accent_rgb))
    output.image = image


def _clip_bbox_float(bbox: BBox, *, width: int, height: int) -> BBox | None:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    x0 = max(0.0, min(float(width), x0))
    y0 = max(0.0, min(float(height), y0))
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return (float(x0), float(y0), float(x1), float(y1))


def add_pages_safe_context_text(
    output: TaskOutput,
    *,
    instance_seed: int,
    params: Mapping[str, Any] | None,
    task_id: str,
    scene_id: str,
) -> None:
    """Draw non-answer context text into safe empty page margins and trace it."""

    _add_pages_safe_context_text_for_route(
        output,
        instance_seed=int(instance_seed),
        params=params,
        task_id=str(task_id),
        route_id=str(scene_id),
        config_kind="scene_id",
    )


def add_pages_scene_safe_context_text(
    output: TaskOutput,
    *,
    instance_seed: int,
    params: Mapping[str, Any] | None,
    task_id: str,
    scene_id: str,
) -> None:
    """Draw non-answer context text for a migrated pages scene task."""

    _add_pages_safe_context_text_for_route(
        output,
        instance_seed=int(instance_seed),
        params=params,
        task_id=str(task_id),
        route_id=str(scene_id),
        config_kind="scene",
    )


def _add_pages_safe_context_text_for_route(
    output: TaskOutput,
    *,
    instance_seed: int,
    params: Mapping[str, Any] | None,
    task_id: str,
    route_id: str,
    config_kind: str,
) -> None:
    """Draw non-answer context text using scene or scene config routing."""

    trace_payload = output.trace_payload if isinstance(output.trace_payload, MutableMapping) else None
    if trace_payload is None or output.image is None:
        return
    render_spec = trace_payload.get("render_spec", {})
    context_policy = render_spec.get("context_text_policy", {}) if isinstance(render_spec, Mapping) else {}
    allow_inside_container = bool(
        isinstance(context_policy, Mapping)
        and context_policy.get("container_bboxes_are_background", False)
    )

    resolved_params = _resolve_context_text_params(
        params=params,
        task_id=str(task_id),
        route_id=str(route_id),
        config_kind=str(config_kind),
    )
    context_color_source = _apply_information_scene_context_colors(
        resolved_params,
        trace_payload=trace_payload,
        explicit_params=params,
    )
    context_profile = _resolve_context_profile(resolved_params)
    enabled = bool(resolved_params.get("pages_context_text_enabled", resolved_params.get("context_text_enabled", True)))
    if not enabled:
        _record_context_text_layer(
            trace_payload,
            elements=(),
            enabled=False,
            layout_mode="none",
            layout_spec={
                "reason": "disabled",
                "context_profile": str(context_profile),
                "context_color_source": str(context_color_source),
            },
        )
        return

    image = output.image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    width, height = image.size
    occupied = _collect_occupied_bboxes(trace_payload)
    rng = spawn_rng(int(instance_seed), f"{task_id}.{route_id}.pages_context_text")
    context_font_family = sample_font_family(
        role="context",
        instance_seed=int(instance_seed),
        namespace=f"{task_id}.{route_id}.pages_context_text_font",
        params=_context_font_params(resolved_params),
        explicit_key="pages_context_text_font_family",
        weights_key="pages_context_text_font_family_weights",
    )
    text_fill = _coerce_rgb(
        resolved_params.get("pages_context_text_rgb", resolved_params.get("context_text_rgb", (62, 70, 82))),
        fallback=(62, 70, 82),
    )
    muted_fill = _coerce_rgb(
        resolved_params.get("pages_context_muted_text_rgb", resolved_params.get("context_muted_text_rgb", (101, 109, 123))),
        fallback=(101, 109, 123),
    )
    box_fill = _coerce_rgb(
        resolved_params.get("pages_context_box_fill_rgb", resolved_params.get("context_box_fill_rgb", (255, 255, 255))),
        fallback=(255, 255, 255),
    )
    box_border = _coerce_rgb(
        resolved_params.get("pages_context_box_border_rgb", resolved_params.get("context_box_border_rgb", (204, 211, 222))),
        fallback=(204, 211, 222),
    )
    padding_px = int(
        resolved_params.get(
            "pages_context_overlap_padding_px",
            resolved_params.get("context_text_overlap_padding_px", 8),
        )
    )
    mode, normalized_weights = _resolve_context_mode(
        resolved_params,
        rng=rng,
        context_profile=str(context_profile),
    )
    mode_config = _resolve_mode_count_config(str(mode), resolved_params)
    simple_requested = _sample_int_range(
        rng,
        int(mode_config["simple_min"]),
        int(mode_config["simple_max"]),
    )
    paragraph_requested = _sample_int_range(
        rng,
        int(mode_config["paragraph_min"]),
        int(mode_config["paragraph_max"]),
    )
    simple_count_override = resolved_params.get(
        "pages_context_simple_count",
        resolved_params.get("context_text_simple_count"),
    )
    if simple_count_override is not None:
        simple_requested = max(0, int(simple_count_override))
    paragraph_count_override = resolved_params.get(
        "pages_context_paragraph_box_count",
        resolved_params.get(
            "context_text_paragraph_box_count",
            resolved_params.get(
                "pages_context_side_note_count",
                resolved_params.get("context_text_side_note_count"),
            ),
        ),
    )
    if paragraph_count_override is not None:
        paragraph_requested = max(0, int(paragraph_count_override))
    default_max_elements = int(simple_requested) + (2 * int(paragraph_requested))
    max_elements = max(
        0,
        int(
            resolved_params.get(
                "pages_context_text_max_elements",
                resolved_params.get("context_text_max_elements", default_max_elements),
            )
        ),
    )
    candidates = _context_candidates(width=int(width), height=int(height))
    selected_candidates = _select_candidate_order(candidates, rng=rng)[: int(simple_requested)]
    elements: list[ContextTextElement] = []
    occupied_with_context = list(occupied)

    side_notes_added = _draw_side_notes(
        draw,
        elements=elements,
        scene_occupied=occupied,
        occupied=occupied_with_context,
        rng=rng,
        width=int(width),
        height=int(height),
        requested_count=int(paragraph_requested),
        max_elements=int(max_elements),
        allow_inside_container=bool(allow_inside_container),
        font_family=str(context_font_family),
        text_fill=text_fill,
        muted_fill=muted_fill,
        box_fill=box_fill,
        box_border=box_border,
        padding_px=int(padding_px),
    )

    simple_added = 0
    for candidate in selected_candidates:
        if len(elements) >= int(max_elements):
            break
        role = str(candidate["role"])
        selection = sample_context_text(str(candidate["manifest_path"]), rng=rng)
        font_size = int(candidate["font_size_px"])
        font = load_font(font_size, bold=bool(candidate["bold"]), font_family=str(context_font_family))
        fitted = _fit_one_line(draw, str(selection.text), font=font, max_width_px=int(candidate["max_width_px"]))
        if not fitted:
            continue
        bbox = _candidate_bbox(
            draw,
            text=fitted,
            font=font,
            slot=str(candidate["slot"]),
            width=int(width),
            height=int(height),
            margin_px=int(candidate["margin_px"]),
        )
        if not _bbox_is_safe(bbox, occupied_with_context, width=int(width), height=int(height), padding_px=int(padding_px)):
            continue
        stroke_width = 1
        fill = text_fill if role == "header" else muted_fill
        draw_text_traced(draw,
            (float(bbox[0]), float(bbox[1])),
            fitted,
            font=font,
            fill=tuple(int(value) for value in fill),
            stroke_width=int(stroke_width),
            stroke_fill=resolve_text_stroke_fill(tuple(fill)),
         role="readout", required=False,)
        trace_bbox = tuple(int(round(value)) for value in bbox)
        element = ContextTextElement(
            context_id=f"pages_context_{len(elements):02d}",
            role=str(role),
            text=str(fitted),
            bbox_xyxy=trace_bbox,
            manifest_path=str(selection.manifest_path),
            source_ids=tuple(selection.source_ids),
            row_index=int(selection.row_index),
            layout_mode=f"safe_margin:{candidate['slot']}",
            font_family=str(context_font_family),
        )
        elements.append(element)
        occupied_with_context.append(tuple(float(value) for value in trace_bbox))
        simple_added += 1

    output.image = image
    layout_mode = f"pages_context:{mode}" if elements else f"pages_context:{mode}:no_safe_slots"
    layout_spec = {
        "placement_policy": "post_render_safe_margin",
        "context_profile": str(context_profile),
        "mode": str(mode),
        "mode_weights": dict(normalized_weights),
        "candidate_slots": [str(candidate["slot"]) for candidate in candidates],
        "max_elements": int(max_elements),
        "requested_simple_count": int(simple_requested),
        "simple_count": int(simple_added),
        "requested_paragraph_box_count": int(paragraph_requested),
        "paragraph_box_count": int(side_notes_added),
        "font_family": str(context_font_family),
        "overlap_padding_px": int(padding_px),
        "context_text_source": "assets/context_text",
        "context_color_source": str(context_color_source),
    }
    _record_context_text_layer(
        trace_payload,
        elements=tuple(elements),
        enabled=True,
        layout_mode=str(layout_mode),
        layout_spec=layout_spec,
    )


def _resolve_context_text_params(
    *,
    params: Mapping[str, Any] | None,
    task_id: str,
    route_id: str,
    config_kind: str = "scene_id",
) -> dict[str, Any]:
    """Merge task-config context text defaults with explicit generation params."""

    resolved = _context_text_config_defaults(
        task_id=str(task_id),
        route_id=str(route_id),
        config_kind=str(config_kind),
    )
    resolved.update(dict(params or {}))
    return resolved


def _apply_information_scene_context_colors(
    resolved_params: MutableMapping[str, Any],
    *,
    trace_payload: Mapping[str, Any],
    explicit_params: Mapping[str, Any] | None,
) -> str:
    """Use the selected pages information-scene style for context text boxes."""

    roles = _information_scene_roles_from_trace(trace_payload)
    if not roles:
        return "configured_context_colors"

    changed: list[str] = []
    for primary_key, alias_keys, role_names in (
        ("pages_context_text_rgb", ("context_text_rgb",), ("text",)),
        ("pages_context_muted_text_rgb", ("context_muted_text_rgb",), ("muted_text", "text")),
        ("pages_context_box_fill_rgb", ("context_box_fill_rgb",), ("surface_alt", "panel_fill", "surface")),
        ("pages_context_box_border_rgb", ("context_box_border_rgb",), ("panel_border", "guide", "axis")),
    ):
        if _has_explicit_context_color(explicit_params, primary_key, alias_keys):
            continue
        role_rgb = _first_role_rgb(roles, role_names)
        if role_rgb is None:
            continue
        resolved_params[str(primary_key)] = tuple(int(value) for value in role_rgb)
        changed.append(str(primary_key))

    return "information_scene_style" if changed else "configured_context_colors"


def _information_scene_roles_from_trace(trace_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    render_spec = trace_payload.get("render_spec", {})
    if not isinstance(render_spec, Mapping):
        return {}

    style_candidates: list[Any] = [render_spec.get("information_scene_style", {})]
    for container_key in ("calendar_style", "background_style"):
        container = render_spec.get(str(container_key), {})
        if isinstance(container, Mapping):
            style_candidates.append(container.get("information_scene_style", {}))

    for style_meta in style_candidates:
        if not isinstance(style_meta, Mapping):
            continue
        roles = style_meta.get("roles_rgb", {})
        if isinstance(roles, Mapping) and roles:
            return roles
    return {}


def _has_explicit_context_color(
    explicit_params: Mapping[str, Any] | None,
    primary_key: str,
    alias_keys: Sequence[str],
) -> bool:
    if not isinstance(explicit_params, Mapping):
        return False
    for key in (str(primary_key), *(str(alias) for alias in alias_keys)):
        if key in explicit_params and explicit_params.get(key) is not None:
            return True
    return False


def _first_role_rgb(roles: Mapping[str, Any], role_names: Sequence[str]) -> RGB | None:
    for role_name in role_names:
        role_value = roles.get(str(role_name))
        if isinstance(role_value, Sequence) and not isinstance(role_value, (str, bytes)) and len(role_value) >= 3:
            try:
                return tuple(max(0, min(255, int(channel))) for channel in role_value[:3])  # type: ignore[return-value]
            except Exception:
                continue
    return None


def _context_text_config_defaults(*, task_id: str, route_id: str, config_kind: str) -> dict[str, Any]:
    cfg = get_scene_defaults("pages", str(route_id))
    _generation, rendering, _prompt = split_scene_generation_rendering_prompt_defaults(
        cfg if isinstance(cfg, Mapping) else {},
        task_id=str(task_id),
    )
    resolved: dict[str, Any] = dict(rendering)

    # Legacy migration bridge: older Pages scene configs placed context text
    # under visual.context_text. Canonical shared defaults now live under
    # rendering.shared, matching Charts.
    visual = cfg.get("visual", {}) if isinstance(cfg, Mapping) else {}
    if not isinstance(visual, Mapping):
        return resolved

    shared = visual.get("shared", {})
    if isinstance(shared, Mapping):
        shared_context = shared.get("context_text", {})
        if isinstance(shared_context, Mapping):
            resolved.update(dict(shared_context))

    direct_context = visual.get("context_text", {})
    if isinstance(direct_context, Mapping):
        resolved.update(dict(direct_context))

    task_overrides = visual.get("task_overrides", {})
    if isinstance(task_overrides, Mapping):
        task_values = task_overrides.get(str(task_id), {})
        if isinstance(task_values, Mapping):
            task_context = task_values.get("context_text", {})
            if isinstance(task_context, Mapping):
                resolved.update(dict(task_context))
    return resolved


def _context_font_params(params: Mapping[str, Any]) -> dict[str, Any]:
    resolved = dict(params)
    if "pages_context_text_font_family" not in resolved and "context_text_font_family" in resolved:
        resolved["pages_context_text_font_family"] = resolved["context_text_font_family"]
    if "pages_context_text_font_family_weights" not in resolved and "context_text_font_family_weights" in resolved:
        resolved["pages_context_text_font_family_weights"] = resolved["context_text_font_family_weights"]
    return resolved


def _resolve_context_profile(params: Mapping[str, Any]) -> str:
    explicit_profile = params.get("pages_context_profile", params.get("context_text_profile"))
    if explicit_profile is None:
        legacy_mode = params.get(
            "pages_context_mode",
            params.get("context_text_mode", params.get("pages_context_density", params.get("context_text_density"))),
        )
        if legacy_mode is not None and _normalize_mode(str(legacy_mode)) == "paragraph_box":
            return "report_paragraph"
    raw_profile = explicit_profile if explicit_profile is not None else _DEFAULT_CONTEXT_PROFILE
    normalized = str(raw_profile).strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized in {"dense", "clean_minimal", "clean_minimal_only", "dense_clean_minimal"}:
        return "dense_clean_minimal"
    if normalized in {"report", "paragraph", "paragraph_capable", "report_paragraph"}:
        return "report_paragraph"
    if normalized not in _CONTEXT_PROFILE_WEIGHTS:
        raise ValueError(f"unsupported pages context profile: {raw_profile!r}")
    return str(normalized)


def _resolve_context_mode(
    params: Mapping[str, Any],
    *,
    rng: random.Random,
    context_profile: str,
) -> tuple[str, dict[str, float]]:
    supported = tuple(_CONTEXT_MODE_CONFIG)
    profile_weights = dict(
        _CONTEXT_PROFILE_WEIGHTS.get(str(context_profile), _CONTEXT_PROFILE_WEIGHTS[_DEFAULT_CONTEXT_PROFILE])
    )
    explicit = params.get(
        "pages_context_mode",
        params.get("context_text_mode", params.get("pages_context_density", params.get("context_text_density"))),
    )
    if explicit is not None:
        mode = _normalize_mode(str(explicit))
        if mode not in set(supported):
            raise ValueError(f"unsupported pages context mode: {explicit!r}")
        if float(profile_weights.get(str(mode), 0.0)) <= 0.0:
            raise ValueError(
                f"pages context mode {mode!r} is not supported by profile {context_profile!r}"
            )
        return str(mode), {key: 1.0 if key == mode else 0.0 for key in supported}

    raw_weights = params.get(
        "pages_context_mode_weights",
        params.get(
            "context_text_mode_weights",
            params.get("pages_context_density_weights", params.get("context_text_density_weights", {})),
        ),
    )
    weights = _normalize_mode_weights(raw_weights if isinstance(raw_weights, Mapping) else {}, profile_weights=profile_weights)
    mode = _weighted_choice(rng, weights, fallback=supported)
    total = sum(max(0.0, float(value)) for value in weights.values())
    if total <= 0.0:
        normalized = {key: 1.0 / float(len(supported)) for key in supported}
    else:
        normalized = {key: max(0.0, float(value)) / total for key, value in weights.items()}
    return str(mode), dict(normalized)


def _normalize_mode(value: str) -> str:
    normalized = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    return str(_CONTEXT_MODE_ALIASES.get(normalized, normalized))


def _normalize_mode_weights(raw_weights: Mapping[str, Any], *, profile_weights: Mapping[str, Any]) -> dict[str, float]:
    weights = {str(key): max(0.0, float(value)) for key, value in profile_weights.items()}
    for raw_key, raw_value in raw_weights.items():
        mode = _normalize_mode(str(raw_key))
        if mode not in weights:
            continue
        try:
            value = float(raw_value)
        except Exception:
            continue
        if float(profile_weights.get(str(mode), 0.0)) <= 0.0:
            weights[mode] = 0.0
        else:
            weights[mode] = max(0.0, float(value))
    if sum(max(0.0, float(value)) for value in weights.values()) <= 0.0:
        weights = {str(key): max(0.0, float(value)) for key, value in profile_weights.items()}
    return weights


def _resolve_mode_count_config(mode: str, params: Mapping[str, Any]) -> dict[str, int]:
    resolved = dict(_CONTEXT_MODE_CONFIG[str(mode)])
    configured_modes = params.get("pages_context_mode_configs", params.get("context_text_mode_configs", {}))
    if isinstance(configured_modes, Mapping):
        configured = configured_modes.get(str(mode), {})
        if isinstance(configured, Mapping):
            aliases = {
                "simple_min": ("simple_count_min", "simple_min"),
                "simple_max": ("simple_count_max", "simple_max"),
                "paragraph_min": ("paragraph_box_count_min", "paragraph_count_min", "side_note_count_min", "paragraph_min"),
                "paragraph_max": ("paragraph_box_count_max", "paragraph_count_max", "side_note_count_max", "paragraph_max"),
            }
            for target_key, source_keys in aliases.items():
                for source_key in source_keys:
                    if source_key in configured:
                        resolved[target_key] = max(0, int(configured[source_key]))
                        break

    prefix = f"pages_context_{str(mode)}_"
    fallback_prefix = f"context_text_{str(mode)}_"
    key_pairs = (
        ("simple_min", f"{prefix}simple_count_min", f"{fallback_prefix}simple_count_min"),
        ("simple_max", f"{prefix}simple_count_max", f"{fallback_prefix}simple_count_max"),
        ("paragraph_min", f"{prefix}paragraph_box_count_min", f"{fallback_prefix}paragraph_box_count_min"),
        ("paragraph_max", f"{prefix}paragraph_box_count_max", f"{fallback_prefix}paragraph_box_count_max"),
    )
    for target_key, pages_key, fallback_key in key_pairs:
        if pages_key in params:
            resolved[target_key] = max(0, int(params[pages_key]))
        elif fallback_key in params:
            resolved[target_key] = max(0, int(params[fallback_key]))
    if int(resolved["simple_max"]) < int(resolved["simple_min"]):
        resolved["simple_max"] = int(resolved["simple_min"])
    if int(resolved["paragraph_max"]) < int(resolved["paragraph_min"]):
        resolved["paragraph_max"] = int(resolved["paragraph_min"])
    return resolved


def _sample_int_range(rng: random.Random, low: int, high: int) -> int:
    low_i = int(low)
    high_i = int(high)
    if high_i <= low_i:
        return int(low_i)
    return int(rng.randint(low_i, high_i))


def _weighted_choice(rng: random.Random, weights: Mapping[str, float], *, fallback: Sequence[str]) -> str:
    weighted = [(str(key), max(0.0, float(weights.get(str(key), 0.0)))) for key in fallback]
    weighted = [(key, value) for key, value in weighted if value > 0.0]
    if not weighted:
        weighted = [(str(key), 1.0) for key in fallback]
    total = sum(value for _key, value in weighted)
    cursor = rng.random() * float(total)
    running = 0.0
    for key, value in weighted:
        running += float(value)
        if cursor <= running:
            return str(key)
    return str(weighted[-1][0])


def _draw_side_notes(
    draw: ImageDraw.ImageDraw,
    *,
    elements: list[ContextTextElement],
    scene_occupied: Sequence[BBox],
    occupied: list[BBox],
    rng: random.Random,
    width: int,
    height: int,
    requested_count: int,
    max_elements: int,
    allow_inside_container: bool = False,
    font_family: str,
    text_fill: Tuple[int, int, int],
    muted_fill: Tuple[int, int, int],
    box_fill: Tuple[int, int, int],
    box_border: Tuple[int, int, int],
    padding_px: int,
) -> int:
    if int(requested_count) <= 0:
        return 0
    candidates = list(
        _side_note_candidates(
            width=int(width),
            height=int(height),
            occupied=scene_occupied,
            rng=rng,
            allow_inside_container=bool(allow_inside_container),
        )
    )
    added = 0
    for candidate in candidates:
        if added >= int(requested_count) or len(elements) + 2 > int(max_elements):
            break
        note_bbox = tuple(float(value) for value in candidate["bbox"])
        if not _bbox_is_safe(note_bbox, occupied, width=int(width), height=int(height), padding_px=int(padding_px)):
            continue
        if _draw_one_side_note(
            draw,
            elements=elements,
            note_bbox=note_bbox,
            side=str(candidate["side"]),
            rng=rng,
            width=int(width),
            height=int(height),
            font_family=str(font_family),
            text_fill=text_fill,
            muted_fill=muted_fill,
            box_fill=box_fill,
            box_border=box_border,
        ):
            occupied.append(note_bbox)
            added += 1
    return int(added)


def _side_note_candidates(
    *,
    width: int,
    height: int,
    occupied: Sequence[BBox],
    rng: random.Random,
    allow_inside_container: bool = False,
) -> tuple[dict[str, Any], ...]:
    if not occupied:
        occupied = tuple()
    margin = max(12, min(22, int(round(min(width, height) * 0.018))))
    gutter = max(8, min(16, int(round(min(width, height) * 0.014))))
    min_x = max(0.0, min(float(box[0]) for box in occupied)) if occupied else float(width * 0.5)
    max_x = min(float(width), max(float(box[2]) for box in occupied)) if occupied else float(width * 0.5)
    top_limit = float(max(margin + 42, int(height * 0.16)))
    note_height = float(min(max(132, int(height * 0.23)), max(140, int(height - (2 * margin) - 96))))
    max_top = float(max(top_limit, height - margin - note_height - 40))
    if max_top > top_limit:
        top = float(rng.randint(int(top_limit), int(max_top)))
    else:
        top = float(max(margin, (height - note_height) * 0.5))
    min_width = max(96, min(126, int(round(width * 0.09))))
    max_note_width = max(min_width, min(238, int(round(width * 0.18))))
    candidates: list[dict[str, Any]] = []

    left_available = int(min_x - float(margin) - float(gutter))
    if left_available >= min_width:
        note_width = min(max_note_width, left_available)
        candidates.append(
            {
                "side": "left",
                "bbox": (
                    float(margin),
                    float(top),
                    float(margin + note_width),
                    float(top + note_height),
                ),
            }
        )

    right_available = int(float(width) - max_x - float(margin) - float(gutter))
    if right_available >= min_width:
        note_width = min(max_note_width, right_available)
        candidates.append(
            {
                "side": "right",
                "bbox": (
                    float(width - margin - note_width),
                    float(top),
                    float(width - margin),
                    float(top + note_height),
                ),
            }
        )

    bottom_available = int(float(height) - max(float(box[3]) for box in occupied) - float(margin) - float(gutter))
    bottom_note_height = max(104, min(150, int(round(height * 0.18))))
    if bottom_available >= bottom_note_height:
        bottom_width = max(260, min(int(round(width * 0.58)), int(width - (2 * margin))))
        bottom_left_min = int(margin)
        bottom_left_max = max(bottom_left_min, int(width - margin - bottom_width))
        bottom_left = float(rng.randint(bottom_left_min, bottom_left_max)) if bottom_left_max > bottom_left_min else float(bottom_left_min)
        bottom_top = float(height - margin - bottom_note_height)
        candidates.append(
            {
                "side": "bottom",
                "bbox": (
                    float(bottom_left),
                    float(bottom_top),
                    float(bottom_left + bottom_width),
                    float(bottom_top + bottom_note_height),
                ),
            }
        )

    if bool(allow_inside_container):
        inside_width = float(max(138, min(238, int(round(width * 0.20)))))
        inside_height = float(max(120, min(150, int(round(height * 0.20)))))
        x_mid = float((width - inside_width) * 0.5)
        y_mid = float((height - inside_height) * 0.5)
        y_upper = float(max(margin + 54, int(round(height * 0.18))))
        y_lower = float(height - margin - inside_height)
        inside_candidates = (
            ("inside_left", (float(margin), y_mid, float(margin + inside_width), float(y_mid + inside_height))),
            ("inside_right", (float(width - margin - inside_width), y_mid, float(width - margin), float(y_mid + inside_height))),
            ("inside_top_right", (float(width - margin - inside_width), y_upper, float(width - margin), float(y_upper + inside_height))),
            ("inside_bottom_left", (float(margin), y_lower, float(margin + inside_width), float(y_lower + inside_height))),
            ("inside_bottom_right", (float(width - margin - inside_width), y_lower, float(width - margin), float(y_lower + inside_height))),
            ("inside_bottom", (x_mid, float(height - margin - inside_height), float(x_mid + inside_width), float(height - margin))),
        )
        for side, bbox in inside_candidates:
            candidates.append({"side": str(side), "bbox": tuple(float(value) for value in bbox)})

    rng.shuffle(candidates)
    return tuple(candidates)


def _draw_one_side_note(
    draw: ImageDraw.ImageDraw,
    *,
    elements: list[ContextTextElement],
    note_bbox: BBox,
    side: str,
    rng: random.Random,
    width: int,
    height: int,
    font_family: str,
    text_fill: Tuple[int, int, int],
    muted_fill: Tuple[int, int, int],
    box_fill: Tuple[int, int, int],
    box_border: Tuple[int, int, int],
) -> bool:
    left, top, right, bottom = [float(value) for value in note_bbox]
    inner_pad = max(8, min(12, int(round((right - left) * 0.08))))
    inner_width = max(30, int(right - left - (2 * inner_pad)))
    inner_height = max(40, int(bottom - top - (2 * inner_pad)))
    heading_font = load_font(max(10, min(13, int(round(height * 0.014)))), bold=True, font_family=str(font_family))
    body_font = load_font(max(9, min(12, int(round(height * 0.012)))), bold=False, font_family=str(font_family))
    heading_selection = _sample_context_text_filtered("phrases/callout_phrases.txt", rng=rng, avoid_digits=True)
    body_manifest = "paragraphs/context_template_blocks.txt" if inner_width >= 120 and inner_height >= 120 else "sentences/context_template_sentences.txt"
    body_selection = _sample_context_text_filtered(body_manifest, rng=rng, avoid_digits=True)
    heading_text = _fit_one_line(draw, str(heading_selection.text), font=heading_font, max_width_px=int(inner_width))
    max_body_lines = max(2, min(6, int((inner_height - 28) / 15)))
    body_text = _wrap_text_to_lines(
        draw,
        str(body_selection.text),
        font=body_font,
        max_width_px=int(inner_width),
        max_lines=int(max_body_lines),
    )
    if not heading_text or not body_text:
        return False

    draw.rounded_rectangle(
        tuple(int(round(value)) for value in note_bbox),
        radius=8,
        fill=tuple(int(value) for value in box_fill),
        outline=tuple(int(value) for value in box_border),
        width=1,
    )
    divider_y = float(top + inner_pad + 22)
    draw.line(
        (left + inner_pad, divider_y, right - inner_pad, divider_y),
        fill=tuple(int(value) for value in box_border),
        width=1,
    )

    heading_xy = (float(left + inner_pad), float(top + inner_pad + 1))
    body_xy = (float(left + inner_pad), float(top + inner_pad + 32))
    heading_stroke = resolve_text_stroke_fill(tuple(text_fill))
    body_stroke = resolve_text_stroke_fill(tuple(muted_fill))
    heading_bbox_raw = draw.textbbox(heading_xy, heading_text, font=heading_font, stroke_width=1)
    draw_text_traced(draw,
        heading_xy,
        heading_text,
        font=heading_font,
        fill=tuple(int(value) for value in text_fill),
        stroke_width=1,
        stroke_fill=tuple(int(value) for value in heading_stroke),
     role="readout", required=False,)
    body_bbox_raw = draw.multiline_textbbox(body_xy, body_text, font=body_font, spacing=3, stroke_width=1)
    draw.multiline_text(
        body_xy,
        body_text,
        font=body_font,
        fill=tuple(int(value) for value in muted_fill),
        spacing=3,
        stroke_width=1,
        stroke_fill=tuple(int(value) for value in body_stroke),
    )
    elements.extend(
        [
            ContextTextElement(
                context_id=f"pages_context_{len(elements):02d}",
                role="side_note_heading",
                text=str(heading_text),
                bbox_xyxy=_clip_trace_bbox(heading_bbox_raw, width=int(width), height=int(height)),
                manifest_path=str(heading_selection.manifest_path),
                source_ids=tuple(heading_selection.source_ids),
                row_index=int(heading_selection.row_index),
                layout_mode=f"safe_margin:side_note:{side}",
                font_family=str(font_family),
            ),
            ContextTextElement(
                context_id=f"pages_context_{len(elements) + 1:02d}",
                role="side_note_body",
                text=str(body_text),
                bbox_xyxy=_clip_trace_bbox(body_bbox_raw, width=int(width), height=int(height)),
                manifest_path=str(body_selection.manifest_path),
                source_ids=tuple(body_selection.source_ids),
                row_index=int(body_selection.row_index),
                layout_mode=f"safe_margin:side_note:{side}",
                font_family=str(font_family),
            ),
        ]
    )
    return True


def _sample_context_text_filtered(manifest_path: str, *, rng: random.Random, avoid_digits: bool) -> Any:
    last_selection = None
    for _attempt in range(32):
        selection = sample_context_text(str(manifest_path), rng=rng)
        last_selection = selection
        text = str(selection.text)
        if "$" in text or "€" in text or "£" in text or "¥" in text:
            continue
        if bool(avoid_digits) and any(char.isdigit() for char in text):
            continue
        return selection
    if last_selection is None:
        return sample_context_text(str(manifest_path), rng=rng)
    return last_selection


def _record_context_text_layer(
    trace_payload: MutableMapping[str, Any],
    *,
    elements: Iterable[ContextTextElement],
    enabled: bool,
    layout_mode: str,
    layout_spec: Mapping[str, Any],
) -> None:
    element_tuple = tuple(elements)
    render_spec = trace_payload.setdefault("render_spec", {})
    if isinstance(render_spec, MutableMapping):
        render_spec["context_text_layer"] = context_text_layer_metadata(
            element_tuple,
            enabled=bool(enabled),
            layout_mode=str(layout_mode),
            layout_spec=dict(layout_spec),
        )
    render_map = trace_payload.setdefault("render_map", {})
    if isinstance(render_map, MutableMapping):
        render_map["context_text_bboxes_px"] = {
            str(element.context_id): [int(value) for value in element.bbox_xyxy]
            for element in element_tuple
        }


def _context_candidates(width: int, height: int) -> tuple[dict[str, Any], ...]:
    margin = max(10, min(18, int(round(min(width, height) * 0.018))))
    return (
        {
            "role": "header",
            "manifest_path": "phrases/headlines.txt",
            "slot": "top_left",
            "font_size_px": max(11, min(15, int(round(height * 0.016)))),
            "bold": True,
            "margin_px": margin,
            "max_width_px": max(180, int(width * 0.36)),
        },
        {
            "role": "source_note",
            "manifest_path": "phrases/source_notes.txt",
            "slot": "top_center",
            "font_size_px": max(10, min(13, int(round(height * 0.014)))),
            "bold": False,
            "margin_px": margin,
            "max_width_px": max(160, int(width * 0.28)),
        },
        {
            "role": "source_note",
            "manifest_path": "phrases/source_notes.txt",
            "slot": "top_right",
            "font_size_px": max(10, min(13, int(round(height * 0.014)))),
            "bold": False,
            "margin_px": margin,
            "max_width_px": max(180, int(width * 0.36)),
        },
        {
            "role": "footer",
            "manifest_path": "phrases/footers.txt",
            "slot": "bottom_left",
            "font_size_px": max(10, min(13, int(round(height * 0.014)))),
            "bold": False,
            "margin_px": margin,
            "max_width_px": max(220, int(width * 0.44)),
        },
        {
            "role": "caption",
            "manifest_path": "phrases/captions.txt",
            "slot": "bottom_center",
            "font_size_px": max(10, min(13, int(round(height * 0.014)))),
            "bold": False,
            "margin_px": margin,
            "max_width_px": max(160, int(width * 0.28)),
        },
        {
            "role": "caption",
            "manifest_path": "phrases/captions.txt",
            "slot": "bottom_right",
            "font_size_px": max(10, min(13, int(round(height * 0.014)))),
            "bold": False,
            "margin_px": margin,
            "max_width_px": max(180, int(width * 0.36)),
        },
    )


def _select_candidate_order(candidates: Sequence[dict[str, Any]], *, rng: random.Random) -> tuple[dict[str, Any], ...]:
    ordered = list(candidates)
    rng.shuffle(ordered)
    ordered.sort(key=lambda item: 0 if str(item["role"]) in {"header", "footer"} else 1)
    return tuple(ordered)


def _fit_one_line(draw: ImageDraw.ImageDraw, text: str, *, font: Any, max_width_px: int) -> str:
    raw = " ".join(str(text).split())
    if not raw:
        return ""
    if draw.textbbox((0, 0), raw, font=font, stroke_width=1)[2] <= int(max_width_px):
        return raw
    ellipsis = "..."
    words = raw.split()
    fitted = ""
    for word in words:
        candidate = f"{fitted} {word}".strip()
        if draw.textbbox((0, 0), f"{candidate}{ellipsis}", font=font, stroke_width=1)[2] > int(max_width_px):
            break
        fitted = candidate
    if fitted:
        return f"{fitted}{ellipsis}"
    chars: list[str] = []
    for char in raw:
        candidate = "".join(chars) + char
        if draw.textbbox((0, 0), f"{candidate}{ellipsis}", font=font, stroke_width=1)[2] > int(max_width_px):
            break
        chars.append(char)
    return f"{''.join(chars).strip()}{ellipsis}" if chars else ""


def _wrap_text_to_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    font: Any,
    max_width_px: int,
    max_lines: int,
) -> str:
    words = " ".join(str(text).split()).split()
    if not words:
        return ""
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font, stroke_width=1)[2] <= int(max_width_px):
            current = candidate
            continue
        if current:
            lines.append(current)
        current = str(word)
        if len(lines) >= max(1, int(max_lines)):
            break
    if current and len(lines) < max(1, int(max_lines)):
        lines.append(current)
    if not lines:
        return ""
    if len(lines) >= max(1, int(max_lines)):
        last = lines[-1]
        ellipsis = "..."
        while last and draw.textbbox((0, 0), f"{last}{ellipsis}", font=font, stroke_width=1)[2] > int(max_width_px):
            last = " ".join(last.split()[:-1])
        lines[-1] = f"{last}{ellipsis}" if last else ellipsis
    return "\n".join(lines)


def _clip_trace_bbox(bbox: Sequence[float], *, width: int, height: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = [int(round(float(value))) for value in bbox[:4]]
    clipped_x0 = min(max(0, x0), max(0, int(width) - 1))
    clipped_y0 = min(max(0, y0), max(0, int(height) - 1))
    clipped_x1 = min(max(clipped_x0 + 1, x1), int(width))
    clipped_y1 = min(max(clipped_y0 + 1, y1), int(height))
    return (int(clipped_x0), int(clipped_y0), int(clipped_x1), int(clipped_y1))


def _candidate_bbox(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    font: Any,
    slot: str,
    width: int,
    height: int,
    margin_px: int,
) -> BBox:
    bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=1)
    text_width = float(bbox[2] - bbox[0])
    text_height = float(bbox[3] - bbox[1])
    margin = float(margin_px)
    if str(slot) == "top_right":
        left = float(width) - margin - text_width
        top = margin
    elif str(slot) == "top_center":
        left = (float(width) - text_width) * 0.5
        top = margin
    elif str(slot) == "bottom_left":
        left = margin
        top = float(height) - margin - text_height
    elif str(slot) == "bottom_center":
        left = (float(width) - text_width) * 0.5
        top = float(height) - margin - text_height
    elif str(slot) == "bottom_right":
        left = float(width) - margin - text_width
        top = float(height) - margin - text_height
    else:
        left = margin
        top = margin
    return (
        float(left),
        float(top),
        float(left + text_width),
        float(top + text_height),
    )


def _collect_occupied_bboxes(trace_payload: Mapping[str, Any]) -> list[BBox]:
    render_map = trace_payload.get("render_map", {})
    render_spec = trace_payload.get("render_spec", {})
    context_policy = render_spec.get("context_text_policy", {}) if isinstance(render_spec, Mapping) else {}
    containers_are_background = bool(
        isinstance(context_policy, Mapping)
        and context_policy.get("container_bboxes_are_background", False)
    )
    boxes: list[BBox] = []
    if isinstance(render_map, Mapping):
        for map_key in (
            "event_bboxes_by_id",
            "node_bboxes_px",
            "node_label_bboxes_px",
            "edge_bboxes_px",
            "card_bboxes_px",
            "label_bboxes_px",
            "value_bboxes_px",
            "caption_bboxes_px",
            "section_bboxes_px",
            "section_title_bboxes_px",
        ):
            bbox_map = render_map.get(str(map_key))
            if isinstance(bbox_map, Mapping):
                for value in bbox_map.values():
                    if _looks_like_bbox(value):
                        boxes.append(tuple(float(item) for item in value[:4]))  # type: ignore[index]
        render_map_keys = (
            "scene_bbox_px",
            "page_bbox_px",
            "calendar_panel_bbox_px",
            "panel_bbox_px",
            "title_bbox_px",
            "subtitle_bbox_px",
            "content_bbox_px",
        )
        if containers_are_background:
            render_map_keys = tuple(
                key
                for key in render_map_keys
                if key
                not in {
                    "scene_bbox_px",
                    "page_bbox_px",
                    "calendar_panel_bbox_px",
                    "panel_bbox_px",
                    "title_bbox_px",
                    "content_bbox_px",
                }
            )
        for key in render_map_keys:
            value = render_map.get(key)
            if _looks_like_bbox(value):
                boxes.append(tuple(float(item) for item in value[:4]))  # type: ignore[index]
        lane_bboxes = render_map.get("lane_bboxes_px")
        if isinstance(lane_bboxes, Mapping):
            for value in lane_bboxes.values():
                if _looks_like_bbox(value):
                    boxes.append(tuple(float(item) for item in value[:4]))  # type: ignore[index]

    scene_ir = trace_payload.get("scene_ir", {})
    entities = scene_ir.get("entities", []) if isinstance(scene_ir, Mapping) else []
    if not isinstance(entities, Sequence):
        return boxes
    for entity in entities:
        if not isinstance(entity, Mapping):
            continue
        if containers_are_background and str(entity.get("entity_type", "")) in {
            "diagram_panel",
            "page_panel",
            "timeline_panel",
            "calendar_panel",
        }:
            continue
        for key in ("bbox_px", "bbox_xyxy"):
            value = entity.get(key)
            if _looks_like_bbox(value):
                boxes.append(tuple(float(item) for item in value[:4]))  # type: ignore[index]
    return boxes


def _looks_like_bbox(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 4:
        return False
    try:
        x0, y0, x1, y1 = [float(item) for item in value[:4]]
    except Exception:
        return False
    return x1 > x0 and y1 > y0


def _bbox_is_safe(
    bbox: BBox,
    occupied: Sequence[BBox],
    *,
    width: int,
    height: int,
    padding_px: int,
) -> bool:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    if x0 < 0 or y0 < 0 or x1 > float(width) or y1 > float(height):
        return False
    padded = (
        float(x0 - padding_px),
        float(y0 - padding_px),
        float(x1 + padding_px),
        float(y1 + padding_px),
    )
    return not any(_bboxes_overlap(padded, box) for box in occupied)


def _bboxes_overlap(left: BBox, right: BBox) -> bool:
    return not (
        float(left[2]) <= float(right[0])
        or float(left[0]) >= float(right[2])
        or float(left[3]) <= float(right[1])
        or float(left[1]) >= float(right[3])
    )


def _coerce_rgb(value: Any, *, fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 3:
        try:
            return tuple(max(0, min(255, int(channel))) for channel in value[:3])  # type: ignore[return-value]
        except Exception:
            return tuple(int(channel) for channel in fallback)
    return tuple(int(channel) for channel in fallback)


__all__ = [
    "add_pages_safe_context_text",
    "add_pages_scene_safe_context_text",
    "annotate_pages_font_assets",
    "annotate_pages_scene_font_assets",
    "ensure_pages_information_style",
    "resolve_pages_default_font_family",
    "resolve_pages_scene_default_font_family",
    "wrap_pages_generation",
    "wrap_pages_scene_generation",
]
