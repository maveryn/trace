"""Scene-local defaults for radial code-wheel tasks."""

from __future__ import annotations

from ...shared.visual_defaults import load_symbolic_noise_defaults


POST_IMAGE_NOISE_DEFAULTS = load_symbolic_noise_defaults(scene_id="radial_code_wheel", apply_prob=0.20)


__all__ = ["POST_IMAGE_NOISE_DEFAULTS"]
