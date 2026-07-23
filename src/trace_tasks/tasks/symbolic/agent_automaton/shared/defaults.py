"""Scene-local defaults for symbolic agent-automaton tasks."""

from __future__ import annotations

from ...shared.visual_defaults import load_symbolic_noise_defaults


POST_IMAGE_NOISE_DEFAULTS = load_symbolic_noise_defaults(scene_id="agent_automaton", apply_prob=0.5)
