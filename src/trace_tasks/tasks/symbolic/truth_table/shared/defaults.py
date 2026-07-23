"""Default values for symbolic truth-table rendering."""

from __future__ import annotations

from ...shared.visual_defaults import load_symbolic_noise_defaults


POST_IMAGE_NOISE_DEFAULTS = load_symbolic_noise_defaults(
    scene_id="truth_table",
    apply_prob=0.16,
)


__all__ = ["POST_IMAGE_NOISE_DEFAULTS"]
