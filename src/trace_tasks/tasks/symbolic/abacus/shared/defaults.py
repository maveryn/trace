"""Scene-local defaults for symbolic abacus tasks."""

from __future__ import annotations

from ...shared.visual_defaults import load_symbolic_noise_defaults

from .rules import DEFAULT_OPTION_LABELS


POST_IMAGE_NOISE_DEFAULTS = load_symbolic_noise_defaults(scene_id="abacus", apply_prob=0.18)


__all__ = [
    "DEFAULT_OPTION_LABELS",
    "POST_IMAGE_NOISE_DEFAULTS",
]
