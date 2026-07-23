"""Source-image helpers for pixel-village reconstruction tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

from PIL import Image

from .....core.seed import hash64
from ....shared.config_defaults import group_default
from ...shared.option_rendering import fit_source_image
from ...shared.rpg_tile_profiles import resolve_rpg_tile_profile
from .rendering import DEFAULT_DISPLAY_TILE_PX, PixelVillageScene, render_pixel_village_map
from .sampling import _DEFAULTS


@dataclass(frozen=True)
class PixelVillageSourceSpec:
    """Neutral source-scene parameters for visual reconstruction tasks."""

    source_size: Tuple[int, int]
    canvas_size: Tuple[int, int]
    tile_px: int
    grid_cols: int
    grid_rows: int
    canvas_profile: str
    canvas_profile_probabilities: Mapping[str, float]
    theme_mode: str
    cemetery_mode: str
    orchard_mode: str
    windmill_mode: str
    river_mode: str
    river_orientation: str
    river_placement: str


def build_pixel_village_source_spec(
    *,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    fallback_source_width: int,
    fallback_source_height: int,
    instance_seed: int,
    namespace: str,
) -> PixelVillageSourceSpec:
    """Resolve source-image render parameters from task params and defaults."""

    profile = resolve_rpg_tile_profile(
        params=params,
        defaults=rendering_defaults,
        tile_px_key="pixel_village_tile_px",
        fallback_tile_px=DEFAULT_DISPLAY_TILE_PX,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
        width_key="source_width",
        height_key="source_height",
    )
    return PixelVillageSourceSpec(
        source_size=(int(profile.width), int(profile.height)),
        canvas_size=(int(profile.width), int(profile.height)),
        tile_px=int(profile.tile_px),
        grid_cols=int(profile.grid_cols),
        grid_rows=int(profile.grid_rows),
        canvas_profile=str(profile.profile_id),
        canvas_profile_probabilities=dict(profile.probabilities),
        theme_mode=str(params.get("theme_mode", group_default(rendering_defaults, "pixel_village_theme_mode", _DEFAULTS.theme_mode))),
        cemetery_mode=str(params.get("cemetery_mode", group_default(rendering_defaults, "pixel_village_cemetery_mode", "force"))),
        orchard_mode=str(params.get("orchard_mode", group_default(rendering_defaults, "pixel_village_orchard_mode", "force"))),
        windmill_mode=str(params.get("windmill_mode", group_default(rendering_defaults, "pixel_village_windmill_mode", "force"))),
        river_mode=str(params.get("river_mode", group_default(rendering_defaults, "pixel_village_river_mode", "force"))),
        river_orientation=str(params.get("river_orientation", group_default(rendering_defaults, "pixel_village_river_orientation", "auto"))),
        river_placement=str(params.get("river_placement", group_default(rendering_defaults, "pixel_village_river_placement", "balanced"))),
    )


def render_pixel_village_source_scene(
    *,
    seed_namespace: str,
    instance_seed: int,
    attempt_index: int,
    source_spec: PixelVillageSourceSpec,
) -> PixelVillageScene:
    """Render a reusable dense pixel-village source scene."""

    seed = hash64(int(instance_seed), f"{seed_namespace}:source_scene", int(attempt_index))
    return render_pixel_village_map(
        int(seed),
        width=int(source_spec.canvas_size[0]),
        height=int(source_spec.canvas_size[1]),
        tile_px=int(source_spec.tile_px),
        grid_cols=int(source_spec.grid_cols),
        grid_rows=int(source_spec.grid_rows),
        cemetery_mode=str(source_spec.cemetery_mode),
        orchard_mode=str(source_spec.orchard_mode),
        windmill_mode=str(source_spec.windmill_mode),
        theme_mode=str(source_spec.theme_mode),
        river_mode=str(source_spec.river_mode),
        river_orientation=str(source_spec.river_orientation),
        river_placement=str(source_spec.river_placement),
    )


def source_panel_for_scene(scene: PixelVillageScene, source_size: Sequence[int]) -> Image.Image:
    """Return a fixed-size RGB source panel from a rendered pixel village."""

    if int(scene.image.width) == int(source_size[0]) and int(scene.image.height) == int(source_size[1]):
        return scene.image.convert("RGB")
    return fit_source_image(scene.image, width=int(source_size[0]), height=int(source_size[1]))


def candidate_crop_boxes_from_scene_entities(
    *,
    scene: PixelVillageScene,
    source_size: Sequence[int],
    patch_size: Sequence[int],
    margin_px: int,
) -> Tuple[Tuple[int, int, int, int], ...]:
    """Build patch-sized crop candidates centered on source-scene entities."""

    source_w, source_h = int(source_size[0]), int(source_size[1])
    patch_w, patch_h = int(patch_size[0]), int(patch_size[1])
    margin = int(margin_px)
    scale_x = float(source_w) / max(1.0, float(scene.image.width))
    scale_y = float(source_h) / max(1.0, float(scene.image.height))
    boxes: list[Tuple[int, int, int, int]] = []
    seen: set[Tuple[int, int, int, int]] = set()
    for entity in scene.entities:
        if str(entity.category) in {"barrier", "path_feature", "territory_feature"}:
            continue
        x0, y0, x1, y1 = [float(value) for value in entity.bbox_xyxy]
        cx = int(round((x0 + x1) * 0.5 * scale_x))
        cy = int(round((y0 + y1) * 0.5 * scale_y))
        crop_x0 = min(max(margin, cx - patch_w // 2), max(margin, source_w - patch_w - margin))
        crop_y0 = min(max(margin, cy - patch_h // 2), max(margin, source_h - patch_h - margin))
        box = (int(crop_x0), int(crop_y0), int(crop_x0 + patch_w), int(crop_y0 + patch_h))
        if box in seen:
            continue
        if box[0] < margin or box[1] < margin or box[2] > source_w - margin or box[3] > source_h - margin:
            continue
        seen.add(box)
        boxes.append(box)
    return tuple(boxes)


__all__ = [
    "PixelVillageSourceSpec",
    "build_pixel_village_source_spec",
    "candidate_crop_boxes_from_scene_entities",
    "render_pixel_village_source_scene",
    "source_panel_for_scene",
]
