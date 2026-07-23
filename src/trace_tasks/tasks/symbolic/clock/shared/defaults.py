"""Default values for symbolic clock-display scenes."""

from __future__ import annotations

from dataclasses import dataclass

from ...shared.visual_defaults import load_symbolic_noise_defaults


@dataclass(frozen=True)
class ClockDefaults:
    """Stable fallback defaults for symbolic clock-display scenes."""

    hour_min: int = 1
    hour_max: int = 12
    minute_min: int = 0
    minute_max: int = 55
    minute_step: int = 5
    min_hand_angle_gap_deg: float = 10.0
    canvas_width: int = 640
    canvas_height: int = 640
    outer_margin_px: int = 36
    face_radius_px: int = 236
    bezel_width_px: int = 10
    numeral_font_size_px: int = 28
    major_tick_length_px: int = 18
    minor_tick_length_px: int = 8
    major_tick_width_px: int = 4
    minor_tick_width_px: int = 2
    minor_tick_dot_radius_px: int = 3
    hour_hand_width_px: int = 12
    minute_hand_width_px: int = 8
    second_hand_width_px: int = 3
    hand_bbox_padding_px: int = 6
    center_dot_radius_px: int = 8
    inner_ring_inset_px: int = 18
    inner_ring_width_px: int = 4


DEFAULTS = ClockDefaults()
POST_IMAGE_NOISE_DEFAULTS = load_symbolic_noise_defaults(scene_id="clock", apply_prob=0.0)


__all__ = ["DEFAULTS", "POST_IMAGE_NOISE_DEFAULTS", "ClockDefaults"]
