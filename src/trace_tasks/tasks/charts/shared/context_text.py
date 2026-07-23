"""Chart-domain context-text wrapper for generated chart images."""

from __future__ import annotations

import random
from functools import wraps
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....core.seed import spawn_rng
from ....core.scene_config import get_scene_defaults
from ...base import TaskOutput
from ...shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from ...shared.context_text_assets import sample_context_text
from ...shared.font_assets import sample_font_family
from ...shared.text_legibility import draw_text_traced
from ...shared.text_rendering import load_font, resolve_text_stroke_fill
from ...shared.visual_style.context_layer import ContextTextElement, context_text_layer_metadata


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
    "right_sidebar": "paragraph_box",
    "left_sidebar": "paragraph_box",
    "bottom_band": "paragraph_box",
}


def wrap_charts_generation(
    original_generate: Callable[..., TaskOutput],
    *,
    task_id: str,
    scene_id: str,
) -> Callable[..., TaskOutput]:
    """Wrap one chart task generator with domain-wide render audit defaults."""

    @wraps(original_generate)
    def _generate_with_chart_render_defaults(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        output = original_generate(self, int(instance_seed), params=params, max_attempts=int(max_attempts))
        add_chart_safe_context_text(
            output,
            instance_seed=int(instance_seed),
            params=params,
            task_id=str(task_id),
            scene_id=str(scene_id),
        )
        return output

    return _generate_with_chart_render_defaults


def add_chart_safe_context_text(
    output: TaskOutput,
    *,
    instance_seed: int,
    params: Mapping[str, Any] | None,
    task_id: str,
    scene_id: str,
) -> None:
    """Draw shared non-answer chart context into safe empty margins."""

    trace_payload = output.trace_payload if isinstance(output.trace_payload, MutableMapping) else None
    if trace_payload is None or output.image is None:
        return

    render_spec = trace_payload.get("render_spec", {})
    if isinstance(render_spec, Mapping) and isinstance(render_spec.get("context_text_layer"), Mapping):
        _record_existing_context_policy(
            trace_payload,
            task_id=str(task_id),
            scene_id=str(scene_id),
        )
        return

    resolved_params = _resolve_chart_context_params(
        params=params,
        task_id=str(task_id),
        scene_id=str(scene_id),
    )
    context_color_source = _apply_information_scene_context_colors(
        resolved_params,
        trace_payload=trace_payload,
        explicit_params=params,
    )
    context_profile = _resolve_context_profile(resolved_params)
    enabled = bool(
        resolved_params.get(
            "chart_context_text_enabled",
            resolved_params.get("context_text_enabled", True),
        )
    )
    if not enabled:
        _record_context_text_layer(
            trace_payload,
            elements=(),
            enabled=False,
            layout_mode="chart_context:none",
            layout_spec={
                "reason": "disabled",
                "context_profile": str(context_profile),
                "task_id": str(task_id),
                "scene_id": str(scene_id),
                },
        )
        return

    routing_key = _chart_context_routing_key(task_id=task_id, scene_id=scene_id)
    rng = spawn_rng(int(instance_seed), f"{task_id}.{routing_key}.chart_context_text")
    mode, normalized_weights = _resolve_context_mode(resolved_params, rng=rng, context_profile=str(context_profile))
    mode_config = _resolve_mode_count_config(str(mode), resolved_params)

    image = output.image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    width, height = image.size
    occupied = _collect_occupied_bboxes(trace_payload)
    _record_context_protected_bboxes(trace_payload, occupied)
    occupied_with_context = list(occupied)
    elements: list[ContextTextElement] = []

    context_font_family = sample_font_family(
        role="context",
        instance_seed=int(instance_seed),
        namespace=f"{task_id}.{routing_key}.chart_context_text_font",
        params=_context_font_params(resolved_params),
        explicit_key="chart_context_text_font_family",
        weights_key="chart_context_text_font_family_weights",
    )
    text_fill = _coerce_rgb(
        resolved_params.get("chart_context_text_rgb", resolved_params.get("context_text_rgb", (62, 70, 82))),
        fallback=(62, 70, 82),
    )
    muted_fill = _coerce_rgb(
        resolved_params.get(
            "chart_context_muted_text_rgb",
            resolved_params.get("context_muted_text_rgb", (101, 109, 123)),
        ),
        fallback=(101, 109, 123),
    )
    box_fill = _coerce_rgb(
        resolved_params.get("chart_context_box_fill_rgb", resolved_params.get("context_box_fill_rgb", (255, 255, 255))),
        fallback=(255, 255, 255),
    )
    box_border = _coerce_rgb(
        resolved_params.get(
            "chart_context_box_border_rgb",
            resolved_params.get("context_box_border_rgb", (204, 211, 222)),
        ),
        fallback=(204, 211, 222),
    )
    padding_px = int(
        resolved_params.get(
            "chart_context_overlap_padding_px",
            resolved_params.get("context_text_overlap_padding_px", 8),
        )
    )

    paragraph_requested = _sample_int_range(
        rng,
        int(mode_config["paragraph_min"]),
        int(mode_config["paragraph_max"]),
    )
    paragraph_added = _draw_paragraph_boxes(
        draw,
        elements=elements,
        occupied=occupied_with_context,
        rng=rng,
        width=int(width),
        height=int(height),
        requested_count=int(paragraph_requested),
        font_family=str(context_font_family),
        text_fill=text_fill,
        muted_fill=muted_fill,
        box_fill=box_fill,
        box_border=box_border,
        padding_px=int(padding_px),
    )
    extension_meta: dict[str, Any] = {"canvas_extension_added": False}
    if (
        str(context_profile) == "report_paragraph"
        and str(mode) == "paragraph_box"
        and int(paragraph_requested) > 0
        and int(paragraph_added) <= 0
    ):
        image, paragraph_extension_added, extension_meta = _draw_paragraph_extension(
            image,
            elements=elements,
            rng=rng,
            requested_count=int(paragraph_requested),
            font_family=str(context_font_family),
            text_fill=text_fill,
            muted_fill=muted_fill,
            box_fill=box_fill,
            box_border=box_border,
        )
        paragraph_added += int(paragraph_extension_added)
        draw = ImageDraw.Draw(image)
        width, height = image.size

    simple_requested = _sample_int_range(
        rng,
        int(mode_config["simple_min"]),
        int(mode_config["simple_max"]),
    )
    simple_added = _draw_simple_context(
        draw,
        elements=elements,
        occupied=occupied_with_context,
        rng=rng,
        width=int(width),
        height=int(height),
        requested_count=int(simple_requested),
        font_family=str(context_font_family),
        text_fill=text_fill,
        muted_fill=muted_fill,
        padding_px=int(padding_px),
    )

    if elements:
        output.image = image
    _record_canvas_size(trace_payload, width=int(width), height=int(height))
    layout_mode = f"chart_context:{mode}"
    if mode != "clean" and not elements:
        layout_mode = f"{layout_mode}:no_safe_slots"
    if str(context_profile) == "report_paragraph" and str(mode) == "paragraph_box" and int(paragraph_added) <= 0:
        raise RuntimeError(
            "report_paragraph chart context selected paragraph_box but no paragraph context box was rendered "
            f"for {task_id}"
        )
    _record_context_text_layer(
        trace_payload,
        elements=tuple(elements),
        enabled=True,
        layout_mode=str(layout_mode),
        layout_spec={
            "placement_policy": "post_render_safe_margin",
            "context_profile": str(context_profile),
            "mode": str(mode),
            "mode_weights": dict(normalized_weights),
            "task_id": str(task_id),
            "scene_id": str(scene_id),
            "font_family": str(context_font_family),
            "overlap_padding_px": int(padding_px),
            "occupied_bbox_count": int(len(occupied)),
            "requested_simple_count": int(simple_requested),
            "simple_count": int(simple_added),
            "requested_paragraph_box_count": int(paragraph_requested),
            "paragraph_box_count": int(paragraph_added),
            "context_text_source": "assets/context_text",
            "context_color_source": str(context_color_source),
            **dict(extension_meta),
        },
    )


def _record_existing_context_policy(
    trace_payload: MutableMapping[str, Any],
    *,
    task_id: str,
    scene_id: str,
) -> None:
    render_spec = trace_payload.setdefault("render_spec", {})
    if not isinstance(render_spec, MutableMapping):
        return
    context_profile = ""
    context_layer = render_spec.get("context_text_layer")
    if isinstance(context_layer, Mapping):
        layout_spec = context_layer.get("layout_spec", {})
        if isinstance(layout_spec, Mapping):
            context_profile = str(layout_spec.get("context_profile", ""))
    render_spec["chart_context_text_policy"] = {
        "source": "scene_specific_context_text_layer",
        "domain_wrapper": "skipped_existing_context_text_layer",
        "context_profile": str(context_profile),
        "task_id": str(task_id),
        "scene_id": str(scene_id),
    }


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


def _record_canvas_size(trace_payload: MutableMapping[str, Any], *, width: int, height: int) -> None:
    render_spec = trace_payload.setdefault("render_spec", {})
    if isinstance(render_spec, MutableMapping):
        render_spec["canvas_width"] = int(width)
        render_spec["canvas_height"] = int(height)


def _record_context_protected_bboxes(
    trace_payload: MutableMapping[str, Any],
    occupied: Sequence[BBox],
) -> None:
    """Record the pre-context visual regions that chart context text avoided."""

    render_map = trace_payload.setdefault("render_map", {})
    if not isinstance(render_map, MutableMapping):
        return
    protected: dict[str, list[float]] = {}
    existing = render_map.get("context_protected_bboxes_px")
    if isinstance(existing, Mapping):
        for key, value in existing.items():
            if _looks_like_bbox(value):
                protected[str(key)] = _round_trace_bbox(value)  # type: ignore[arg-type]
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                for index, item in enumerate(value):
                    if _looks_like_bbox(item):
                        protected[f"{key}_{index:02d}"] = _round_trace_bbox(item)  # type: ignore[arg-type]

    seen = {tuple(int(round(float(item))) for item in box) for box in protected.values()}
    for index, box in enumerate(occupied):
        rounded = _round_trace_bbox(box)
        key = tuple(int(round(float(item))) for item in rounded)
        if key in seen:
            continue
        seen.add(key)
        protected[f"occupied_{index:03d}"] = rounded

    render_map["context_protected_bboxes_px"] = protected


def _resolve_chart_context_params(
    *,
    params: Mapping[str, Any] | None,
    task_id: str,
    scene_id: str,
) -> dict[str, Any]:
    resolved = _chart_context_config_defaults(
        task_id=str(task_id),
        scene_id=str(scene_id),
    )
    resolved.update(dict(params or {}))
    return resolved


def _apply_information_scene_context_colors(
    resolved_params: MutableMapping[str, Any],
    *,
    trace_payload: Mapping[str, Any],
    explicit_params: Mapping[str, Any] | None,
) -> str:
    """Use the selected chart information-scene style for context text boxes."""

    roles = _information_scene_roles_from_trace(trace_payload)
    if not roles:
        return "configured_context_colors"

    changed: list[str] = []
    for primary_key, alias_keys, role_names in (
        ("chart_context_text_rgb", ("context_text_rgb",), ("text",)),
        ("chart_context_muted_text_rgb", ("context_muted_text_rgb",), ("muted_text", "text")),
        ("chart_context_box_fill_rgb", ("context_box_fill_rgb",), ("surface_alt", "panel_fill", "surface")),
        ("chart_context_box_border_rgb", ("context_box_border_rgb",), ("panel_border", "guide", "axis")),
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
    style_meta = render_spec.get("information_scene_style", {})
    if not isinstance(style_meta, Mapping):
        return {}
    roles = style_meta.get("roles_rgb", {})
    return roles if isinstance(roles, Mapping) else {}


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


def _chart_context_config_defaults(
    *,
    task_id: str,
    scene_id: str,
) -> dict[str, Any]:
    try:
        cfg = get_scene_defaults("charts", str(scene_id))
    except Exception:
        return {}
    if not isinstance(cfg, Mapping):
        return {}
    _generation, rendering, _prompt = split_scene_generation_rendering_prompt_defaults(cfg, task_id=str(task_id))
    return dict(rendering)


def _chart_context_routing_key(
    *,
    task_id: str,
    scene_id: str,
) -> str:
    return str(scene_id)


def _context_font_params(params: Mapping[str, Any]) -> dict[str, Any]:
    resolved = dict(params)
    if "chart_context_text_font_family" not in resolved and "context_text_font_family" in resolved:
        resolved["chart_context_text_font_family"] = resolved["context_text_font_family"]
    if "chart_context_text_font_family_weights" not in resolved and "context_text_font_family_weights" in resolved:
        resolved["chart_context_text_font_family_weights"] = resolved["context_text_font_family_weights"]
    return resolved


def _resolve_context_profile(params: Mapping[str, Any]) -> str:
    raw_profile = params.get("chart_context_profile", params.get("context_text_profile", _DEFAULT_CONTEXT_PROFILE))
    normalized = str(raw_profile).strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized in {"dense", "clean_minimal", "clean_minimal_only", "dense_clean_minimal"}:
        return "dense_clean_minimal"
    if normalized in {"report", "paragraph", "paragraph_capable", "report_paragraph"}:
        return "report_paragraph"
    if normalized not in _CONTEXT_PROFILE_WEIGHTS:
        raise ValueError(f"unsupported chart context profile: {raw_profile!r}")
    return str(normalized)


def _resolve_context_mode(
    params: Mapping[str, Any],
    *,
    rng: random.Random,
    context_profile: str,
) -> tuple[str, dict[str, float]]:
    supported = tuple(_CONTEXT_MODE_CONFIG)
    profile_weights = dict(_CONTEXT_PROFILE_WEIGHTS.get(str(context_profile), _CONTEXT_PROFILE_WEIGHTS[_DEFAULT_CONTEXT_PROFILE]))
    explicit = params.get("chart_context_mode", params.get("context_text_mode"))
    if explicit is not None:
        mode = _normalize_mode(str(explicit))
        if mode not in set(supported):
            raise ValueError(f"unsupported chart context mode: {explicit!r}")
        if float(profile_weights.get(str(mode), 0.0)) <= 0.0:
            raise ValueError(
                f"chart context mode {mode!r} is not supported by profile {context_profile!r}"
            )
        return str(mode), {key: 1.0 if key == mode else 0.0 for key in supported}
    weights = _normalize_mode_weights(profile_weights, profile_weights=profile_weights)
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
    configured_modes = params.get("chart_context_mode_configs", {})
    if isinstance(configured_modes, Mapping):
        configured = configured_modes.get(str(mode), {})
        if isinstance(configured, Mapping):
            aliases = {
                "simple_min": ("simple_count_min", "simple_min"),
                "simple_max": ("simple_count_max", "simple_max"),
                "paragraph_min": ("paragraph_box_count_min", "paragraph_count_min", "paragraph_min"),
                "paragraph_max": ("paragraph_box_count_max", "paragraph_count_max", "paragraph_max"),
            }
            for target_key, source_keys in aliases.items():
                for source_key in source_keys:
                    if source_key in configured:
                        resolved[target_key] = max(0, int(configured[source_key]))
                        break
    prefix = f"chart_context_{str(mode)}_"
    fallback_prefix = f"context_text_{str(mode)}_"
    key_pairs = (
        ("simple_min", f"{prefix}simple_count_min", f"{fallback_prefix}simple_count_min"),
        ("simple_max", f"{prefix}simple_count_max", f"{fallback_prefix}simple_count_max"),
        ("paragraph_min", f"{prefix}paragraph_box_count_min", f"{fallback_prefix}paragraph_box_count_min"),
        ("paragraph_max", f"{prefix}paragraph_box_count_max", f"{fallback_prefix}paragraph_box_count_max"),
    )
    for target_key, chart_key, fallback_key in key_pairs:
        if chart_key in params:
            resolved[target_key] = max(0, int(params[chart_key]))
        elif fallback_key in params:
            resolved[target_key] = max(0, int(params[fallback_key]))
    if int(resolved["simple_max"]) < int(resolved["simple_min"]):
        resolved["simple_max"] = int(resolved["simple_min"])
    if int(resolved["paragraph_max"]) < int(resolved["paragraph_min"]):
        resolved["paragraph_max"] = int(resolved["paragraph_min"])
    return resolved


def _draw_simple_context(
    draw: ImageDraw.ImageDraw,
    *,
    elements: list[ContextTextElement],
    occupied: list[BBox],
    rng: random.Random,
    width: int,
    height: int,
    requested_count: int,
    font_family: str,
    text_fill: RGB,
    muted_fill: RGB,
    padding_px: int,
) -> int:
    if int(requested_count) <= 0:
        return 0
    added = 0
    candidates = list(_simple_context_candidates(width=int(width), height=int(height)))
    rng.shuffle(candidates)
    candidates.sort(key=lambda item: 0 if str(item["role"]) in {"header", "footer"} else 1)
    for candidate in candidates:
        if added >= int(requested_count):
            break
        selection = sample_context_text(str(candidate["manifest_path"]), rng=rng)
        font = load_font(
            int(candidate["font_size_px"]),
            bold=bool(candidate["bold"]),
            font_family=str(font_family),
        )
        text = _fit_one_line(
            draw,
            str(selection.text),
            font=font,
            max_width_px=int(candidate["max_width_px"]),
        )
        if not text:
            continue
        bbox = _candidate_bbox(
            draw,
            text=str(text),
            font=font,
            slot=str(candidate["slot"]),
            width=int(width),
            height=int(height),
            margin_px=int(candidate["margin_px"]),
        )
        if not _bbox_is_safe(bbox, occupied, width=int(width), height=int(height), padding_px=int(padding_px)):
            continue
        fill = text_fill if str(candidate["role"]) == "header" else muted_fill
        stroke_width = 1 if str(candidate["role"]) in {"header", "source_note"} else 0
        stroke_fill = resolve_text_stroke_fill(tuple(fill))
        draw_text_traced(
            draw,
            (float(bbox[0]), float(bbox[1])),
            str(text),
            font=font,
            fill=tuple(int(value) for value in fill),
            stroke_width=int(stroke_width),
            stroke_fill=tuple(int(value) for value in stroke_fill),
            role="non_answer_context_text",
            required=False,
            extra_metadata={"answer_excluded": True, "context_layer": True},
        )
        trace_bbox = _clip_trace_bbox(bbox, width=int(width), height=int(height))
        elements.append(
            ContextTextElement(
                context_id=f"chart_context_{len(elements):02d}",
                role=str(candidate["role"]),
                text=str(text),
                bbox_xyxy=trace_bbox,
                manifest_path=str(selection.manifest_path),
                source_ids=tuple(selection.source_ids),
                row_index=int(selection.row_index),
                layout_mode=f"chart_context:simple:{candidate['slot']}",
                font_family=str(font_family),
            )
        )
        occupied.append(tuple(float(value) for value in trace_bbox))
        added += 1
    return int(added)


def _draw_paragraph_boxes(
    draw: ImageDraw.ImageDraw,
    *,
    elements: list[ContextTextElement],
    occupied: list[BBox],
    rng: random.Random,
    width: int,
    height: int,
    requested_count: int,
    font_family: str,
    text_fill: RGB,
    muted_fill: RGB,
    box_fill: RGB,
    box_border: RGB,
    padding_px: int,
) -> int:
    if int(requested_count) <= 0:
        return 0
    candidates = list(_paragraph_box_candidates(width=int(width), height=int(height), occupied=occupied, rng=rng))
    added = 0
    for candidate in candidates:
        if added >= int(requested_count):
            break
        bbox = tuple(float(value) for value in candidate["bbox"])
        if not _bbox_is_safe(bbox, occupied, width=int(width), height=int(height), padding_px=int(padding_px)):
            continue
        if _draw_one_paragraph_box(
            draw,
            elements=elements,
            box_bbox=bbox,
            placement=str(candidate["placement"]),
            rng=rng,
            width=int(width),
            height=int(height),
            font_family=str(font_family),
            text_fill=text_fill,
            muted_fill=muted_fill,
            box_fill=box_fill,
            box_border=box_border,
        ):
            occupied.append(bbox)
            added += 1
    return int(added)


def _draw_paragraph_extension(
    image: Image.Image,
    *,
    elements: list[ContextTextElement],
    rng: random.Random,
    requested_count: int,
    font_family: str,
    text_fill: RGB,
    muted_fill: RGB,
    box_fill: RGB,
    box_border: RGB,
) -> tuple[Image.Image, int, dict[str, Any]]:
    """Append a reserved context band and draw paragraph boxes there."""

    width, height = image.size
    requested = max(1, int(requested_count))
    fill = tuple(int(value) for value in box_fill)
    border = tuple(int(value) for value in box_border)
    if int(height) >= 520:
        extension_width = max(280, min(340, int(round(width * 0.27))))
        new_image = Image.new("RGB", (int(width + extension_width), int(height)), fill)
        new_image.paste(image, (0, 0))
        draw = ImageDraw.Draw(new_image)
        draw.line((int(width), 0, int(width), int(height)), fill=border, width=1)
        margin = 18
        gap = 18
        max_count = min(2, requested)
        available_height = max(160, int(height) - (2 * margin) - ((max_count - 1) * gap))
        box_height = max(120, min(190, int(available_height / max_count)))
        box_width = max(220, int(extension_width) - (2 * margin))
        left = float(width + margin)
        boxes = []
        for index in range(max_count):
            top = float(margin + index * (box_height + gap))
            if top + box_height > int(height) - margin:
                break
            boxes.append((left, top, float(left + box_width), float(top + box_height)))
        placement = "right_extension"
    else:
        extension_height = max(150, min(210, int(round(height * 0.26))))
        new_image = Image.new("RGB", (int(width), int(height + extension_height)), fill)
        new_image.paste(image, (0, 0))
        draw = ImageDraw.Draw(new_image)
        draw.line((0, int(height), int(width), int(height)), fill=border, width=1)
        margin = 18
        gap = 18
        max_count = min(2, requested)
        available_width = max(260, int(width) - (2 * margin) - ((max_count - 1) * gap))
        box_width = max(260, min(420, int(available_width / max_count)))
        box_height = max(110, int(extension_height) - (2 * margin))
        boxes = []
        for index in range(max_count):
            left = float(margin + index * (box_width + gap))
            if left + box_width > int(width) - margin:
                break
            boxes.append((left, float(height + margin), float(left + box_width), float(height + margin + box_height)))
        placement = "bottom_extension"

    added = 0
    new_width, new_height = new_image.size
    for bbox in boxes:
        if _draw_one_paragraph_box(
            draw,
            elements=elements,
            box_bbox=tuple(float(value) for value in bbox),
            placement=str(placement),
            rng=rng,
            width=int(new_width),
            height=int(new_height),
            font_family=str(font_family),
            text_fill=text_fill,
            muted_fill=muted_fill,
            box_fill=box_fill,
            box_border=box_border,
        ):
            added += 1
    return new_image, int(added), {
        "canvas_extension_added": bool(added),
        "canvas_extension_placement": str(placement),
        "canvas_extension_original_size": [int(width), int(height)],
        "canvas_extension_size": [int(new_width), int(new_height)],
        "paragraph_extension_count": int(added),
    }


def _draw_one_paragraph_box(
    draw: ImageDraw.ImageDraw,
    *,
    elements: list[ContextTextElement],
    box_bbox: BBox,
    placement: str,
    rng: random.Random,
    width: int,
    height: int,
    font_family: str,
    text_fill: RGB,
    muted_fill: RGB,
    box_fill: RGB,
    box_border: RGB,
) -> bool:
    left, top, right, bottom = [float(value) for value in box_bbox]
    inner_pad = max(8, min(14, int(round((right - left) * 0.07))))
    inner_width = max(40, int(right - left - (2 * inner_pad)))
    inner_height = max(44, int(bottom - top - (2 * inner_pad)))
    heading_font = load_font(max(10, min(14, int(round(height * 0.015)))), bold=True, font_family=str(font_family))
    body_font = load_font(max(9, min(12, int(round(height * 0.012)))), bold=False, font_family=str(font_family))
    heading_selection = _sample_context_text_filtered("phrases/callout_phrases.txt", rng=rng)
    body_manifest = "paragraphs/context_template_blocks.txt" if inner_width >= 130 and inner_height >= 116 else "sentences/context_template_sentences.txt"
    body_selection = _sample_context_text_filtered(body_manifest, rng=rng)
    heading_text = _fit_one_line(draw, str(heading_selection.text), font=heading_font, max_width_px=int(inner_width))
    body_lines = max(2, min(7, int((inner_height - 30) / 15)))
    body_text = _wrap_text_to_lines(
        draw,
        str(body_selection.text),
        font=body_font,
        max_width_px=int(inner_width),
        max_lines=int(body_lines),
    )
    if not heading_text or not body_text:
        return False

    int_bbox = tuple(int(round(value)) for value in box_bbox)
    draw.rounded_rectangle(
        int_bbox,
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
    heading_bbox = draw.textbbox(heading_xy, heading_text, font=heading_font, stroke_width=1)
    draw_text_traced(
        draw,
        heading_xy,
        heading_text,
        font=heading_font,
        fill=tuple(int(value) for value in text_fill),
        stroke_width=1,
        stroke_fill=tuple(int(value) for value in heading_stroke),
        role="non_answer_context_text",
        required=False,
        extra_metadata={"answer_excluded": True, "context_layer": True, "context_box": True},
    )
    body_bbox = draw.multiline_textbbox(body_xy, body_text, font=body_font, spacing=3, stroke_width=1)
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
                context_id=f"chart_context_{len(elements):02d}",
                role="paragraph_box_heading",
                text=str(heading_text),
                bbox_xyxy=_clip_trace_bbox(heading_bbox, width=int(width), height=int(height)),
                manifest_path=str(heading_selection.manifest_path),
                source_ids=tuple(heading_selection.source_ids),
                row_index=int(heading_selection.row_index),
                layout_mode=f"chart_context:paragraph_box:{placement}",
                font_family=str(font_family),
            ),
            ContextTextElement(
                context_id=f"chart_context_{len(elements) + 1:02d}",
                role="paragraph_box_body",
                text=str(body_text),
                bbox_xyxy=_clip_trace_bbox(body_bbox, width=int(width), height=int(height)),
                manifest_path=str(body_selection.manifest_path),
                source_ids=tuple(body_selection.source_ids),
                row_index=int(body_selection.row_index),
                layout_mode=f"chart_context:paragraph_box:{placement}",
                font_family=str(font_family),
            ),
        ]
    )
    return True


def _simple_context_candidates(width: int, height: int) -> tuple[dict[str, Any], ...]:
    margin = max(10, min(20, int(round(min(width, height) * 0.018))))
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
            "max_width_px": max(160, int(width * 0.30)),
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
            "max_width_px": max(160, int(width * 0.30)),
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


def _paragraph_box_candidates(
    *,
    width: int,
    height: int,
    occupied: Sequence[BBox],
    rng: random.Random,
) -> tuple[dict[str, Any], ...]:
    if not occupied:
        return tuple()
    margin = max(12, min(24, int(round(min(width, height) * 0.02))))
    gutter = max(8, min(18, int(round(min(width, height) * 0.014))))
    min_x = max(0.0, min(float(box[0]) for box in occupied))
    max_x = min(float(width), max(float(box[2]) for box in occupied))
    min_y = max(0.0, min(float(box[1]) for box in occupied))
    max_y = min(float(height), max(float(box[3]) for box in occupied))
    candidates: list[dict[str, Any]] = []

    box_height = max(112, min(180, int(round(height * 0.22))))
    top_low = max(margin + 32, int(min_y))
    top_high = max(top_low, int(height - margin - box_height - 24))
    top = float(rng.randint(int(top_low), int(top_high))) if top_high > top_low else float(max(margin, (height - box_height) * 0.5))
    min_box_width = max(118, min(150, int(round(width * 0.11))))
    max_box_width = max(min_box_width, min(260, int(round(width * 0.20))))

    left_available = int(min_x - float(margin) - float(gutter))
    if left_available >= min_box_width:
        box_width = min(int(max_box_width), int(left_available))
        candidates.append(
            {
                "placement": "left_margin",
                "bbox": (float(margin), float(top), float(margin + box_width), float(top + box_height)),
            }
        )

    right_available = int(float(width) - max_x - float(margin) - float(gutter))
    if right_available >= min_box_width:
        box_width = min(int(max_box_width), int(right_available))
        candidates.append(
            {
                "placement": "right_margin",
                "bbox": (float(width - margin - box_width), float(top), float(width - margin), float(top + box_height)),
            }
        )

    bottom_available = int(float(height) - max_y - float(margin) - float(gutter))
    bottom_height = max(100, min(150, int(round(height * 0.18))))
    if bottom_available >= bottom_height:
        bottom_width = max(260, min(int(round(width * 0.58)), int(width - (2 * margin))))
        left_min = int(margin)
        left_max = max(left_min, int(width - margin - bottom_width))
        left = float(rng.randint(left_min, left_max)) if left_max > left_min else float(left_min)
        top = float(height - margin - bottom_height)
        candidates.append(
            {
                "placement": "bottom_margin",
                "bbox": (float(left), float(top), float(left + bottom_width), float(top + bottom_height)),
            }
        )

    top_available = int(min_y - float(margin) - float(gutter))
    top_height = max(76, min(118, int(round(height * 0.14))))
    if top_available >= top_height:
        top_width = max(260, min(int(round(width * 0.56)), int(width - (2 * margin))))
        left_min = int(margin)
        left_max = max(left_min, int(width - margin - top_width))
        left = float(rng.randint(left_min, left_max)) if left_max > left_min else float(left_min)
        candidates.append(
            {
                "placement": "top_margin",
                "bbox": (float(left), float(margin), float(left + top_width), float(margin + top_height)),
            }
        )

    rng.shuffle(candidates)
    return tuple(candidates)


def _collect_occupied_bboxes(trace_payload: Mapping[str, Any]) -> list[BBox]:
    boxes: list[BBox] = []
    render_map = trace_payload.get("render_map", {})
    if isinstance(render_map, Mapping):
        _collect_bboxes_recursive(render_map, boxes)
    projected_annotation = trace_payload.get("projected_annotation", {})
    if isinstance(projected_annotation, Mapping):
        _collect_bboxes_recursive(projected_annotation, boxes)
    scene_ir = trace_payload.get("scene_ir", {})
    entities = scene_ir.get("entities", ()) if isinstance(scene_ir, Mapping) else ()
    if isinstance(entities, Sequence) and not isinstance(entities, (str, bytes)):
        for entity in entities:
            if not isinstance(entity, Mapping):
                continue
            for key in ("bbox_px", "bbox_xyxy", "bbox"):
                value = entity.get(key)
                if _looks_like_bbox(value):
                    boxes.append(tuple(float(item) for item in value[:4]))  # type: ignore[index]
    return _dedupe_bboxes(boxes)


def _collect_bboxes_recursive(value: Any, boxes: list[BBox]) -> None:
    if _looks_like_bbox(value):
        boxes.append(tuple(float(item) for item in value[:4]))  # type: ignore[index]
        return
    if isinstance(value, Mapping):
        for child in value.values():
            _collect_bboxes_recursive(child, boxes)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            _collect_bboxes_recursive(child, boxes)


def _dedupe_bboxes(boxes: Sequence[BBox]) -> list[BBox]:
    seen: set[tuple[int, int, int, int]] = set()
    deduped: list[BBox] = []
    for box in boxes:
        key = tuple(int(round(float(value))) for value in box[:4])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(tuple(float(value) for value in box[:4]))
    return deduped


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


def _round_trace_bbox(bbox: Sequence[float]) -> list[float]:
    return [round(float(value), 3) for value in bbox[:4]]


def _sample_context_text_filtered(manifest_path: str, *, rng: random.Random) -> Any:
    last_selection = None
    for _attempt in range(32):
        selection = sample_context_text(str(manifest_path), rng=rng)
        last_selection = selection
        text = str(selection.text)
        if "$" in text or "€" in text or "£" in text or "¥" in text:
            continue
        return selection
    return last_selection if last_selection is not None else sample_context_text(str(manifest_path), rng=rng)


def _sample_int_range(rng: random.Random, low: int, high: int) -> int:
    resolved_low = int(low)
    resolved_high = int(high)
    if resolved_high < resolved_low:
        resolved_high = resolved_low
    if resolved_low == resolved_high:
        return int(resolved_low)
    return int(rng.randrange(int(resolved_low), int(resolved_high) + 1))


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


def _coerce_rgb(value: Any, *, fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 3:
        try:
            return tuple(max(0, min(255, int(channel))) for channel in value[:3])  # type: ignore[return-value]
        except Exception:
            return tuple(int(channel) for channel in fallback)
    return tuple(int(channel) for channel in fallback)


__all__ = [
    "add_chart_safe_context_text",
    "wrap_charts_generation",
]
