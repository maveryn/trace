"""Shared canvas profiles for top-down RPG tile illustration scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ....core.sampling import uniform_choice
from ...shared.config_defaults import group_default
from .canvas_profiles import (
    CANVAS_PROFILE_CUSTOM,
    CANVAS_PROFILE_LANDSCAPE,
    CANVAS_PROFILE_PORTRAIT,
    CANVAS_PROFILE_SQUARE,
    CANVAS_PROFILE_SUPPORT,
)


DEFAULT_RPG_TILE_PX = 48

RPG_TILE_PROFILE_GRIDS: Mapping[str, Tuple[int, int]] = {
    CANVAS_PROFILE_LANDSCAPE: (27, 18),
    CANVAS_PROFILE_PORTRAIT: (18, 27),
    CANVAS_PROFILE_SQUARE: (21, 21),
}


@dataclass(frozen=True)
class RpgTileProfile:
    """Resolved top-down RPG tile canvas profile."""

    profile_id: str
    grid_cols: int
    grid_rows: int
    tile_px: int
    probabilities: Dict[str, float]

    @property
    def width(self) -> int:
        return int(self.grid_cols) * int(self.tile_px)

    @property
    def height(self) -> int:
        return int(self.grid_rows) * int(self.tile_px)

    @property
    def size(self) -> Tuple[int, int]:
        return (int(self.width), int(self.height))

    def trace(self) -> Dict[str, Any]:
        return {
            "canvas_profile": str(self.profile_id),
            "canvas_profile_size": [int(self.width), int(self.height)],
            "canvas_profile_probabilities": dict(self.probabilities),
            "rpg_tile_profile": {
                "tile_px": int(self.tile_px),
                "grid_cols": int(self.grid_cols),
                "grid_rows": int(self.grid_rows),
            },
        }


def _uniform_probability_map(values: Sequence[str]) -> Dict[str, float]:
    support = tuple(str(value) for value in values)
    if not support:
        return {}
    probability = 1.0 / float(len(support))
    return {str(value): float(probability) for value in support}


def _profile_support(params: Mapping[str, Any], defaults: Mapping[str, Any]) -> Tuple[str, ...]:
    raw = params.get("canvas_profile_support", group_default(defaults, "canvas_profile_support", CANVAS_PROFILE_SUPPORT))
    if isinstance(raw, str):
        values = (raw,)
    elif isinstance(raw, Sequence):
        values = tuple(raw)
    else:
        values = tuple(CANVAS_PROFILE_SUPPORT)
    support = tuple(
        dict.fromkeys(
            str(value)
            for value in values
            if str(value) in {CANVAS_PROFILE_LANDSCAPE, CANVAS_PROFILE_SQUARE, CANVAS_PROFILE_PORTRAIT}
        )
    )
    if not support:
        raise ValueError("canvas_profile_support must include at least one supported RPG tile profile")
    return support


def _resolve_tile_px(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    tile_px_key: str,
    fallback_tile_px: int,
) -> int:
    value = int(params.get("tile_px", group_default(defaults, str(tile_px_key), int(fallback_tile_px))))
    if value <= 0:
        raise ValueError("RPG tile size must be positive")
    return int(value)


def resolve_rpg_tile_profile(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    tile_px_key: str,
    fallback_tile_px: int = DEFAULT_RPG_TILE_PX,
    instance_seed: int | None = None,
    namespace: str = "illustrations:rpg_tile_profile",
    width_key: str = "canvas_width",
    height_key: str = "canvas_height",
) -> RpgTileProfile:
    """Resolve a tile-aligned top-down RPG canvas profile."""

    tile_px = _resolve_tile_px(params, defaults, tile_px_key=str(tile_px_key), fallback_tile_px=int(fallback_tile_px))
    if str(width_key) in params or str(height_key) in params:
        fallback_cols, fallback_rows = RPG_TILE_PROFILE_GRIDS[CANVAS_PROFILE_LANDSCAPE]
        width = int(params.get(str(width_key), int(fallback_cols) * int(tile_px)))
        height = int(params.get(str(height_key), int(fallback_rows) * int(tile_px)))
        if width % tile_px != 0 or height % tile_px != 0:
            raise ValueError(f"custom RPG tile canvas {width}x{height} is not divisible by tile size {tile_px}")
        return RpgTileProfile(
            profile_id=CANVAS_PROFILE_CUSTOM,
            grid_cols=int(width // tile_px),
            grid_rows=int(height // tile_px),
            tile_px=int(tile_px),
            probabilities={CANVAS_PROFILE_CUSTOM: 1.0},
        )

    support = _profile_support(params, defaults)
    explicit = params.get("canvas_profile", group_default(defaults, "canvas_profile", None))
    if explicit is not None:
        profile_id = str(explicit)
        if profile_id not in set(support):
            raise ValueError(f"canvas_profile must be one of {support}")
        grid_cols, grid_rows = RPG_TILE_PROFILE_GRIDS[profile_id]
        return RpgTileProfile(
            profile_id=profile_id,
            grid_cols=int(grid_cols),
            grid_rows=int(grid_rows),
            tile_px=int(tile_px),
            probabilities={profile_id: 1.0},
        )

    if instance_seed is not None:
        rng = spawn_rng(int(instance_seed), str(namespace))
        profile_id = str(uniform_choice(rng, support, sort_keys=False))
    else:
        profile_id = str(support[0])
    grid_cols, grid_rows = RPG_TILE_PROFILE_GRIDS[profile_id]
    return RpgTileProfile(
        profile_id=profile_id,
        grid_cols=int(grid_cols),
        grid_rows=int(grid_rows),
        tile_px=int(tile_px),
        probabilities=_uniform_probability_map(support),
    )


def resolve_rpg_tile_render_params(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    tile_px_key: str,
    fallback_tile_px: int = DEFAULT_RPG_TILE_PX,
    instance_seed: int | None = None,
    namespace: str = "illustrations:rpg_tile_profile",
) -> Dict[str, Any]:
    """Return renderer params for one resolved top-down RPG tile profile."""

    profile = resolve_rpg_tile_profile(
        params,
        defaults,
        tile_px_key=str(tile_px_key),
        fallback_tile_px=int(fallback_tile_px),
        instance_seed=instance_seed,
        namespace=str(namespace),
    )
    return {
        "canvas_width": int(profile.width),
        "canvas_height": int(profile.height),
        "tile_px": int(profile.tile_px),
        "grid_cols": int(profile.grid_cols),
        "grid_rows": int(profile.grid_rows),
        **profile.trace(),
    }


def rpg_rotated_tile_grid_for_size(width: int, height: int) -> Tuple[int, int]:
    """Return the rotated-tile split for a top-down RPG source size."""

    w = int(width)
    h = int(height)
    if w <= 0 or h <= 0:
        raise ValueError("RPG rotated-tile source size must be positive")
    if w == h:
        return (3, 3)
    if h > w:
        return (3, 2)
    return (2, 3)


__all__ = [
    "DEFAULT_RPG_TILE_PX",
    "RPG_TILE_PROFILE_GRIDS",
    "RpgTileProfile",
    "resolve_rpg_tile_profile",
    "resolve_rpg_tile_render_params",
    "rpg_rotated_tile_grid_for_size",
]
