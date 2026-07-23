"""Shared wallpaper-pattern rendering helpers for icon pattern tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ...shared.icon_assets import render_icon_rgba, resolve_icon_pool
from ...shared.icon_noise import serialize_icon_noise_edits
from ...shared.icon_scene import RenderedIconInstance, serialize_rendered_icon_instance
from ...shared.icon_style import sample_single_icon_tint
from ...shared.icon_task_rendering import sample_icon_instance_noise
from ...shared.scene_style import draw_icon_panel_chrome, make_icon_canvas_background
from ....shared.visual_style.panel import PANEL_SCENE_TREATMENTS

from .defaults import LATTICE_COLS, LATTICE_ROWS, OPTION_LABELS, REFERENCE_LABEL, WALLPAPER_GROUP_IDS
from .layout import draw_panel_label, option_panel_geometry, reference_panel_geometry
from .state import WallpaperScenePayload


WALLPAPER_CANVAS_TREATMENTS: Tuple[str, ...] = tuple(PANEL_SCENE_TREATMENTS)
WALLPAPER_PANEL_CHROME_POLICY = "shared_panel_canvas_all_treatments"


@dataclass(frozen=True)
class WallpaperElementSpec:
    """One motif instance in an invisible wallpaper lattice."""

    element_index: int
    lattice_row: int
    lattice_col: int
    local_index: int
    u: float
    v: float
    rotation_degrees: int
    mirror_x: bool
    group_role: str


def uniform_str_probability_map(values: Sequence[str], *, selected: str | None = None) -> Dict[str, float]:
    support = tuple(str(value) for value in values)
    if not support:
        return {}
    if selected is not None:
        return {str(selected): 1.0}
    probability = 1.0 / float(len(support))
    return {str(value): float(probability) for value in support}


def resolve_wallpaper_group_support(raw: Any, *, fallback: Sequence[str] = WALLPAPER_GROUP_IDS) -> Tuple[str, ...]:
    if raw is None:
        raw = list(fallback)
    if not isinstance(raw, (list, tuple)):
        raise ValueError("wallpaper_group_ids must be a sequence")
    allowed = set(WALLPAPER_GROUP_IDS)
    support = tuple(str(value).strip() for value in raw if str(value).strip() in allowed)
    if len(support) < 2:
        raise ValueError("wallpaper_group_ids must contain at least two supported group ids")
    return support


def wallpaper_canvas_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Return params with wallpaper panels opted into all shared canvas treatments."""

    resolved = dict(params)
    allowed = set(WALLPAPER_CANVAS_TREATMENTS)
    explicit_treatment = str(resolved.get("icon_canvas_treatment", "")).strip()
    if explicit_treatment and explicit_treatment not in allowed:
        raise ValueError(
            "wallpaper panel tasks only support shared icon canvas treatments; "
            f"got icon_canvas_treatment={explicit_treatment!r}"
        )
    explicit_treatments = resolved.get("icon_canvas_treatments")
    if explicit_treatments is not None:
        if isinstance(explicit_treatments, (str, bytes)) or not isinstance(explicit_treatments, Sequence):
            raise ValueError("icon_canvas_treatments must be a sequence for wallpaper panel tasks")
        requested = tuple(str(value).strip() for value in explicit_treatments if str(value).strip())
        unsupported = tuple(value for value in requested if value not in allowed)
        if unsupported:
            raise ValueError(
                "wallpaper panel tasks only support shared icon canvas treatments; "
                f"got icon_canvas_treatments={list(unsupported)!r}"
            )
    resolved["icon_canvas_treatments"] = list(WALLPAPER_CANVAS_TREATMENTS)
    resolved["icon_canvas_treatment_weights"] = {
        str(treatment): 1.0 for treatment in WALLPAPER_CANVAS_TREATMENTS
    }
    return resolved


def wallpaper_chrome_policy_trace() -> Dict[str, Any]:
    """Return trace metadata for wallpaper-panel chrome constraints."""

    return {
        "wallpaper_panel_chrome_policy": WALLPAPER_PANEL_CHROME_POLICY,
        "available_canvas_treatments": list(WALLPAPER_CANVAS_TREATMENTS),
    }


def cell_element_for_group(group_id: str, *, row: int, col: int) -> Tuple[float, float, int, bool, str]:
    """Return the single visible motif placement for one lattice cell.

    These tasks need distinguishable repeat arrangements, not faithful
    wallpaper-group construction. One motif per lattice cell keeps every group
    at exactly 16 visible icons on the fixed 4 x 4 panel.
    """

    group = str(group_id)
    row_i = int(row)
    col_i = int(col)
    parity = int((row_i + col_i) % 2)
    if group == "p1":
        return (0.50, 0.50, 0, False, "translation_center")
    if group == "p2":
        offset = 0.60 if parity else 0.40
        return (offset, offset, 180 if parity else 0, False, "twofold_checker")
    if group == "pm":
        mirror = bool(col_i % 2)
        return (0.62 if mirror else 0.38, 0.50, 0, mirror, "vertical_mirror_stripe")
    if group == "pg":
        mirror = bool(row_i % 2)
        return (0.64 if mirror else 0.36, 0.50, 180 if mirror else 0, mirror, "row_glide_shift")
    if group == "cm":
        return (
            0.38 if parity == 0 else 0.62,
            0.34 if row_i % 2 == 0 else 0.66,
            0,
            bool(parity),
            "centered_checker_mirror",
        )
    if group == "pmm":
        return (
            0.38 if col_i % 2 == 0 else 0.62,
            0.38 if row_i % 2 == 0 else 0.62,
            180 if row_i % 2 else 0,
            bool(col_i % 2),
            "quadrant_mirror_grid",
        )
    if group == "p4":
        phase = int((row_i + col_i) % 4)
        positions = ((0.50, 0.32), (0.68, 0.50), (0.50, 0.68), (0.32, 0.50))
        u, v = positions[phase]
        return (u, v, int(phase * 90), False, "quarter_turn_cycle")
    if group == "p3":
        phase = int((row_i + (2 * col_i)) % 3)
        positions = ((0.50, 0.32), (0.34, 0.66), (0.66, 0.66))
        u, v = positions[phase]
        return (u, v, int(phase * 120), False, "third_turn_cycle")
    raise ValueError(f"unsupported wallpaper group id: {group_id}")


def elements_for_group(*, group_id: str, rows: int, cols: int) -> Tuple[WallpaperElementSpec, ...]:
    elements: List[WallpaperElementSpec] = []
    for row in range(int(rows)):
        for col in range(int(cols)):
            u, v, rotation, mirror_x, role = cell_element_for_group(str(group_id), row=row, col=col)
            elements.append(
                WallpaperElementSpec(
                    element_index=len(elements),
                    lattice_row=int(row),
                    lattice_col=int(col),
                    local_index=0,
                    u=float(u),
                    v=float(v),
                    rotation_degrees=int(rotation) % 360,
                    mirror_x=bool(mirror_x),
                    group_role=str(role),
                )
            )
    return tuple(elements)


def element_to_trace(element: WallpaperElementSpec) -> Dict[str, Any]:
    return {
        "element_index": int(element.element_index),
        "lattice_row": int(element.lattice_row),
        "lattice_col": int(element.lattice_col),
        "local_index": int(element.local_index),
        "u": round(float(element.u), 4),
        "v": round(float(element.v), 4),
        "rotation_degrees": int(element.rotation_degrees),
        "mirror_x": bool(element.mirror_x),
        "group_role": str(element.group_role),
    }


def element_center_xy(
    *,
    content_bbox: Sequence[int | float],
    element: WallpaperElementSpec,
    lattice_rows: int,
    lattice_cols: int,
) -> Tuple[float, float]:
    x0, y0, x1, y1 = [float(value) for value in content_bbox]
    cell_w = max(1.0, float(x1 - x0) / float(max(1, int(lattice_cols))))
    cell_h = max(1.0, float(y1 - y0) / float(max(1, int(lattice_rows))))
    cx = x0 + ((float(element.lattice_col) + float(element.u)) * cell_w)
    cy = y0 + ((float(element.lattice_row) + float(element.v)) * cell_h)
    return float(cx), float(cy)


def centered_sprite_bbox(
    *,
    sprite_size: Tuple[int, int],
    center_xy: Tuple[float, float],
    content_bbox: Sequence[int | float],
) -> Tuple[int, int, int, int]:
    content_x0, content_y0, content_x1, content_y1 = [int(round(float(value))) for value in content_bbox]
    sprite_w, sprite_h = int(sprite_size[0]), int(sprite_size[1])
    if sprite_w <= 0 or sprite_h <= 0:
        raise ValueError("sprite size must be positive")
    margin = 1
    min_x0 = int(content_x0 + margin)
    max_x0 = int(content_x1 - margin - sprite_w)
    min_y0 = int(content_y0 + margin)
    max_y0 = int(content_y1 - margin - sprite_h)
    if min_x0 > max_x0 or min_y0 > max_y0:
        raise ValueError("wallpaper icon sprite does not fit inside content panel")
    paste_x0 = int(round(float(center_xy[0]) - (float(sprite_w) / 2.0)))
    paste_y0 = int(round(float(center_xy[1]) - (float(sprite_h) / 2.0)))
    paste_x0 = int(max(min_x0, min(max_x0, paste_x0)))
    paste_y0 = int(max(min_y0, min(max_y0, paste_y0)))
    return (int(paste_x0), int(paste_y0), int(paste_x0 + sprite_w), int(paste_y0 + sprite_h))


def select_distinct_icon_ids(rng: Any, *, pool_manifest: str, labels: Sequence[str]) -> Dict[str, str]:
    pool = list(resolve_icon_pool(str(pool_manifest)))
    if len(pool) < len(labels):
        raise ValueError("wallpaper icon pool must contain at least one unique icon per labeled panel")
    rng.shuffle(pool)
    return {str(label): str(icon_id) for label, icon_id in zip(labels, pool[: len(labels)])}


def draw_wallpaper_motifs(
    image: Image.Image,
    *,
    noise_namespace: str,
    instance_seed: int,
    panel_label: str,
    group_id: str,
    icon_id: str,
    tint_rgb: Sequence[int],
    nominal_icon_size_px: int,
    content_bbox: Sequence[int | float],
    render_params: Mapping[str, Any],
    sprite_cache: MutableMapping[Tuple[str, int, bool], Image.Image],
    entity_kind_prefix: str = "wallpaper",
    is_answer_panel: bool = False,
) -> Tuple[Tuple[Dict[str, Any], ...], Tuple[Dict[str, Any], ...]]:
    """Draw one panel's resolved wallpaper lattice without choosing task answers.

    Invariant: the caller supplies the panel label, wallpaper group, icon id,
    and answer flag; this helper only projects motif elements into pixels and
    serializes the corresponding neutral scene entities.
    """

    scene_elements: List[Dict[str, Any]] = []
    scene_icon_instances: List[Dict[str, Any]] = []
    elements = elements_for_group(
        group_id=str(group_id),
        rows=int(render_params["lattice_rows"]),
        cols=int(render_params["lattice_cols"]),
    )
    for element in elements:
        noise_edits, noise_seed = sample_icon_instance_noise(
            instance_seed=int(instance_seed),
            namespace=f"{noise_namespace}:panel_{str(panel_label)}_element_{int(element.element_index)}",
            render_params=render_params,
        )
        cache_key = (str(icon_id), int(element.rotation_degrees) % 360, bool(element.mirror_x))
        if cache_key not in sprite_cache or noise_edits:
            sprite = render_icon_rgba(
                icon_id=str(icon_id),
                size_px=int(nominal_icon_size_px),
                tint_rgb=tuple(int(v) for v in tint_rgb),
                rotation_degrees=int(element.rotation_degrees),
                mirror_x=bool(element.mirror_x),
                noise_edits=tuple(noise_edits),
                noise_seed=int(noise_seed),
            )
            if not noise_edits:
                sprite_cache[cache_key] = sprite
        else:
            sprite = sprite_cache[cache_key]
        center_xy = element_center_xy(
            content_bbox=content_bbox,
            element=element,
            lattice_rows=int(render_params["lattice_rows"]),
            lattice_cols=int(render_params["lattice_cols"]),
        )
        paste_bbox = centered_sprite_bbox(sprite_size=sprite.size, center_xy=center_xy, content_bbox=content_bbox)
        image.alpha_composite(sprite, (int(paste_bbox[0]), int(paste_bbox[1])))
        rendered_instance = RenderedIconInstance(
            instance_id=f"panel_{str(panel_label)}_element_{int(element.element_index)}",
            icon_id=str(icon_id),
            panel=str(panel_label),
            bbox_xyxy=tuple(int(value) for value in paste_bbox),
            nominal_size_px=int(nominal_icon_size_px),
            rotation_degrees=int(element.rotation_degrees) % 360,
            mirror_x=bool(element.mirror_x),
            tint_rgb=tuple(int(value) for value in tint_rgb),
            noise_edits=serialize_icon_noise_edits(tuple(noise_edits)),
            noise_seed=int(noise_seed),
        )
        element_trace = element_to_trace(element)
        scene_elements.append(
            {
                "entity_kind": f"{entity_kind_prefix}_motif_element",
                "panel_label": str(panel_label),
                "wallpaper_group_id": str(group_id),
                "is_answer_panel": bool(is_answer_panel),
                "element_bbox_xyxy": [int(value) for value in paste_bbox],
                **element_trace,
            }
        )
        scene_icon_instances.append(
            serialize_rendered_icon_instance(
                rendered_instance,
                entity_kind=f"{entity_kind_prefix}_motif_icon",
                extra_fields={
                    "panel_label": str(panel_label),
                    "wallpaper_group_id": str(group_id),
                    "is_answer_panel": bool(is_answer_panel),
                    **element_trace,
                },
            )
        )
    return tuple(scene_elements), tuple(scene_icon_instances)


def render_option_wallpaper_scene(
    *,
    rng: Any,
    instance_seed: int,
    option_labels: Sequence[str],
    wallpaper_group_ids_by_label: Mapping[str, str],
    answer_labels: Sequence[str],
    pool_manifest: str,
    render_params: Mapping[str, Any],
    noise_namespace: str,
) -> Tuple[WallpaperScenePayload, Image.Image]:
    """Render option-only wallpaper panels from resolved semantic group ids."""

    return _render_wallpaper_scene(
        rng=rng,
        instance_seed=int(instance_seed),
        panel_labels=tuple(str(label) for label in option_labels),
        option_labels=tuple(str(label) for label in option_labels),
        wallpaper_group_ids_by_label=wallpaper_group_ids_by_label,
        answer_labels=answer_labels,
        reference_group_id=None,
        pool_manifest=str(pool_manifest),
        render_params=render_params,
        noise_namespace=str(noise_namespace),
        reference_layout=False,
    )


def render_reference_wallpaper_scene(
    *,
    rng: Any,
    instance_seed: int,
    option_labels: Sequence[str],
    reference_wallpaper_group_id: str,
    wallpaper_group_ids_by_label: Mapping[str, str],
    answer_labels: Sequence[str],
    pool_manifest: str,
    render_params: Mapping[str, Any],
    noise_namespace: str,
) -> Tuple[WallpaperScenePayload, Image.Image]:
    """Render one Reference panel and resolved candidate wallpaper panels."""

    panel_labels = (REFERENCE_LABEL, *tuple(str(label) for label in option_labels))
    return _render_wallpaper_scene(
        rng=rng,
        instance_seed=int(instance_seed),
        panel_labels=panel_labels,
        option_labels=tuple(str(label) for label in option_labels),
        wallpaper_group_ids_by_label=wallpaper_group_ids_by_label,
        answer_labels=answer_labels,
        reference_group_id=str(reference_wallpaper_group_id),
        pool_manifest=str(pool_manifest),
        render_params=render_params,
        noise_namespace=str(noise_namespace),
        reference_layout=True,
    )


def _render_wallpaper_scene(
    *,
    rng: Any,
    instance_seed: int,
    panel_labels: Sequence[str],
    option_labels: Sequence[str],
    wallpaper_group_ids_by_label: Mapping[str, str],
    answer_labels: Sequence[str],
    reference_group_id: str | None,
    pool_manifest: str,
    render_params: Mapping[str, Any],
    noise_namespace: str,
    reference_layout: bool,
) -> Tuple[WallpaperScenePayload, Image.Image]:
    """Render fully resolved wallpaper panels without task-specific branching.

    Invariant: public task files already selected the reference group, candidate
    groups, and answer labels; this helper only renders those resolved inputs
    into a scene payload and image.
    """

    answer_label_set = set(str(label) for label in answer_labels)
    icon_ids_by_label = select_distinct_icon_ids(rng, pool_manifest=str(pool_manifest), labels=panel_labels)
    tint_rgb, sampled_palette_rgb = sample_single_icon_tint(
        rng,
        channel_min=int(render_params["color_channel_min"]),
        channel_max=int(render_params["color_channel_max"]),
        anchor_colors=(
            tuple(int(v) for v in render_params["background_color_rgb"]),
            tuple(int(v) for v in render_params["panel_fill_rgb"]),
            tuple(int(v) for v in render_params["panel_border_rgb"]),
            tuple(int(v) for v in render_params["header_text_rgb"]),
        ),
        min_color_distance=float(render_params["min_color_distance"]),
        distance_space=str(render_params["color_distance_space"]),
    )
    nominal_icon_size_px = int(
        rng.randint(
            int(render_params["scene_icon_size_min_px"]),
            int(render_params["scene_icon_size_max_px"]),
        )
    )
    if reference_layout:
        panel_geometry, panels = reference_panel_geometry(render_params=render_params, option_labels=option_labels)
    else:
        panel_geometry, panels = option_panel_geometry(render_params=render_params, option_labels=option_labels)
    image = make_icon_canvas_background(
        canvas_width=int(render_params["canvas_width"]),
        canvas_height=int(render_params["canvas_height"]),
        style=render_params.get("_icon_canvas_style_object"),
        fallback_rgb=tuple(int(v) for v in render_params["background_color_rgb"]),
    )
    draw = ImageDraw.Draw(image)
    scene_panels: List[Dict[str, Any]] = []
    scene_elements: List[Dict[str, Any]] = []
    scene_icon_instances: List[Dict[str, Any]] = []
    sprite_cache: Dict[Tuple[str, int, bool], Image.Image] = {}

    for label in panel_labels:
        panel_info = panels[str(label)]
        panel_bbox = tuple(int(value) for value in panel_info["panel_bbox_xyxy"])
        content_bbox = tuple(int(value) for value in panel_info["content_bbox_xyxy"])
        is_reference = bool(str(label) == REFERENCE_LABEL and reference_layout)
        group_id = str(reference_group_id if is_reference else wallpaper_group_ids_by_label[str(label)])
        is_answer = bool((not is_reference) and str(label) in answer_label_set)
        draw_icon_panel_chrome(
            draw,
            bbox=panel_bbox,
            style=render_params.get("_icon_canvas_style_object"),
            fallback_fill_rgb=tuple(int(v) for v in render_params["panel_fill_rgb"]),
            fallback_border_rgb=tuple(int(v) for v in render_params["panel_border_rgb"]),
            radius=int(render_params["panel_corner_radius_px"]),
            border_width=2,
        )
        draw_panel_label(draw, label=str(label), panel_bbox=panel_bbox, render_params=render_params)
        icon_id = str(icon_ids_by_label[str(label)])
        elements, instances = draw_wallpaper_motifs(
            image,
            noise_namespace=str(noise_namespace),
            instance_seed=int(instance_seed),
            panel_label=str(label),
            group_id=str(group_id),
            icon_id=str(icon_id),
            tint_rgb=tuple(int(v) for v in tint_rgb),
            nominal_icon_size_px=int(nominal_icon_size_px),
            content_bbox=content_bbox,
            render_params=render_params,
            sprite_cache=sprite_cache,
            is_answer_panel=bool(is_answer),
        )
        scene_elements.extend(dict(element) for element in elements)
        scene_icon_instances.extend(dict(instance) for instance in instances)
        scene_panels.append(
            {
                "entity_kind": "wallpaper_panel",
                "label": str(label),
                "panel_role": "reference" if is_reference else "candidate",
                "panel_bbox_xyxy": [int(value) for value in panel_bbox],
                "content_bbox_xyxy": [int(value) for value in content_bbox],
                "icon_id": str(icon_id),
                "wallpaper_group_id": str(group_id),
                "is_reference": bool(is_reference),
                "is_answer": bool(is_answer),
                "matches_reference_pattern": bool(is_reference or is_answer),
            }
        )

    return (
        WallpaperScenePayload(
            option_count=int(len(option_labels)),
            option_labels=tuple(str(label) for label in option_labels),
            icon_ids_by_label=dict(icon_ids_by_label),
            sampled_palette_rgb=tuple(tuple(int(channel) for channel in color) for color in sampled_palette_rgb),
            nominal_icon_size_px=int(nominal_icon_size_px),
            panel_geometry=dict(panel_geometry),
            scene_panels=tuple(scene_panels),
            scene_elements=tuple(scene_elements),
            scene_icon_instances=tuple(scene_icon_instances),
        ),
        image.convert("RGB"),
    )


__all__ = [
    "LATTICE_COLS",
    "LATTICE_ROWS",
    "OPTION_LABELS",
    "WALLPAPER_CANVAS_TREATMENTS",
    "WALLPAPER_PANEL_CHROME_POLICY",
    "WALLPAPER_GROUP_IDS",
    "WallpaperElementSpec",
    "centered_sprite_bbox",
    "cell_element_for_group",
    "draw_wallpaper_motifs",
    "element_center_xy",
    "element_to_trace",
    "elements_for_group",
    "resolve_wallpaper_group_support",
    "render_option_wallpaper_scene",
    "render_reference_wallpaper_scene",
    "select_distinct_icon_ids",
    "uniform_str_probability_map",
    "wallpaper_chrome_policy_trace",
    "wallpaper_canvas_params",
]
