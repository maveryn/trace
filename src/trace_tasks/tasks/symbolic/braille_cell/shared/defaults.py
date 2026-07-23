"""Scene-local defaults for Braille-cell tasks."""

from __future__ import annotations

from ...shared.visual_defaults import load_symbolic_noise_defaults


POST_IMAGE_NOISE_DEFAULTS = load_symbolic_noise_defaults(scene_id="braille_cell", apply_prob=0.20)


__all__ = ["POST_IMAGE_NOISE_DEFAULTS"]
