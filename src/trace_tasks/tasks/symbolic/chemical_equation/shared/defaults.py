"""Scene-local defaults for symbolic chemical-equation tasks."""

from __future__ import annotations

from ...shared.visual_defaults import load_symbolic_noise_defaults
from .state import COEFFICIENT_SUPPORT, OPTION_LABELS


POST_IMAGE_NOISE_DEFAULTS = load_symbolic_noise_defaults(
    scene_id="chemical_equation",
    apply_prob=0.12,
)


__all__ = [
    "COEFFICIENT_SUPPORT",
    "OPTION_LABELS",
    "POST_IMAGE_NOISE_DEFAULTS",
]
