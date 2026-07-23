"""Sampling primitives for paired-canvas icon scenes."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ....shared.counting_sampling import resolve_counting_target_and_distractor_triplet
from ...shared.icon_assets import resolve_icon_pool
from ...shared.icon_scene import BBox
from ...shared.icon_style import sample_icon_palette
from ...shared.icon_task_rendering import sample_icon_instance_noise

from .defaults import PairedCanvasDefaults
from .state import PairedIconSpec


def resolve_paired_counts(
    rng,
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    defaults: PairedCanvasDefaults,
) -> Tuple[int, Dict[str, float], int, Dict[str, float], int, Dict[str, float]]:
    """Resolve object, target, and distractor counts for one count task."""

    return resolve_counting_target_and_distractor_triplet(
        rng,
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        fallback_total_min=int(defaults.object_count_min),
        fallback_total_max=int(defaults.object_count_max),
        fallback_target_min=int(defaults.target_count_min),
        fallback_target_max=int(defaults.target_count_max),
        fallback_distractor_min=int(defaults.distractor_count_min),
        fallback_distractor_max=int(defaults.distractor_count_max),
    )


def load_icon_pool_from_params(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    defaults: PairedCanvasDefaults,
) -> Tuple[str, ...]:
    """Load the configured curated icon pool for a paired-canvas task."""

    manifest = str(params.get("pool_manifest", gen_defaults.get("pool_manifest", defaults.pool_manifest)))
    return tuple(str(value) for value in resolve_icon_pool(manifest))


def sample_palette(rng, *, render_params: Mapping[str, Any]) -> Tuple[Tuple[int, int, int], ...]:
    """Sample a background-safe icon palette."""

    palette_size_min = max(2, int(render_params["palette_size_min"]))
    palette_size_max = max(palette_size_min, int(render_params["palette_size_max"]))
    return tuple(
        tuple(int(channel) for channel in color)
        for color in sample_icon_palette(
            rng,
            palette_size=int(rng.randint(palette_size_min, palette_size_max)),
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
    )


def sample_positions(
    rng,
    *,
    count: int,
    min_gap_frac: float,
    max_attempts: int = 600,
) -> Tuple[Tuple[float, float], ...]:
    """Sample normalized panel positions with a minimum center gap."""

    positions: List[Tuple[float, float]] = []
    gap = max(0.02, float(min_gap_frac))
    for _ in range(int(max_attempts)):
        if len(positions) >= int(count):
            break
        x = float(rng.uniform(0.12, 0.88))
        y = float(rng.uniform(0.13, 0.88))
        if all(((x - ox) ** 2 + (y - oy) ** 2) ** 0.5 >= gap for ox, oy in positions):
            positions.append((x, y))
    if len(positions) < int(count):
        raise ValueError("failed to sample separated paired-canvas positions")
    return tuple((float(x), float(y)) for x, y in positions)


def make_icon_spec(
    *,
    instance_seed: int,
    namespace: str,
    render_params: Mapping[str, Any],
    instance_id: str,
    identity_id: str,
    icon_id: str,
    panel: str,
    position: Tuple[float, float],
    tint_rgb: Tuple[int, int, int],
    size_px: int,
    rotation_degrees: int,
) -> PairedIconSpec:
    """Build one render spec with deterministic per-icon noise."""

    noise_edits, noise_seed = sample_icon_instance_noise(
        instance_seed=int(instance_seed),
        namespace=str(namespace),
        render_params=render_params,
    )
    return PairedIconSpec(
        instance_id=str(instance_id),
        identity_id=str(identity_id),
        icon_id=str(icon_id),
        panel=str(panel),
        x_frac=float(position[0]),
        y_frac=float(position[1]),
        nominal_size_px=int(size_px),
        rotation_degrees=int(rotation_degrees) % 360,
        tint_rgb=tuple(int(value) for value in tint_rgb),
        noise_edits=tuple(noise_edits),
        noise_seed=int(noise_seed),
    )


def sample_base_attributes(
    rng,
    *,
    pool: Sequence[str],
    palette: Sequence[Tuple[int, int, int]],
    count: int,
    render_params: Mapping[str, Any],
    rotation_candidates: Sequence[int],
) -> List[Dict[str, Any]]:
    """Sample visually matchable icon attributes."""

    if len(pool) < int(count):
        raise ValueError("icon pool is too small for paired-canvas task")
    icon_ids = [str(value) for value in pool[: int(count)]]
    attrs: List[Dict[str, Any]] = []
    min_size = int(render_params["scene_icon_size_min_px"])
    max_size = int(render_params["scene_icon_size_max_px"])
    for index, icon_id in enumerate(icon_ids):
        attrs.append(
            {
                "identity_id": f"icon_{index}",
                "icon_id": str(icon_id),
                "tint_rgb": tuple(int(value) for value in rng.choice(tuple(palette))),
                "size_px": int(rng.randint(min_size, max_size)),
                "rotation_degrees": int(rng.choice(tuple(int(v) for v in rotation_candidates))),
            }
        )
    return attrs


def bbox_center(bbox: BBox) -> Tuple[float, float]:
    """Return the center point of a pixel bbox."""

    return (0.5 * float(int(bbox[0]) + int(bbox[2])), 0.5 * float(int(bbox[1]) + int(bbox[3])))


__all__ = [
    "bbox_center",
    "load_icon_pool_from_params",
    "make_icon_spec",
    "resolve_paired_counts",
    "sample_base_attributes",
    "sample_palette",
    "sample_positions",
]
