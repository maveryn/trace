from trace_tasks.tasks.shared.render_variation import apply_resolved_layout_jitter_to_margins, resolve_layout_jitter


def test_slack_fraction_layout_jitter_uses_available_safe_space() -> None:
    jitter = {
        "enabled": True,
        "mode": "slack_fraction",
        "requested_x_slack_fraction": 0.5,
        "requested_y_slack_fraction": -0.25,
        "min_margin_px": 20,
    }

    left, right, top, bottom, resolved = apply_resolved_layout_jitter_to_margins(
        left_px=100,
        right_px=300,
        top_px=80,
        bottom_px=200,
        jitter=jitter,
    )

    assert (left, right, top, bottom) == (240, 160, 65, 215)
    assert resolved["mode"] == "slack_fraction"
    assert resolved["requested_dx_px"] == 140
    assert resolved["requested_dy_px"] == -15
    assert resolved["dx_px"] == 140
    assert resolved["dy_px"] == -15
    assert resolved["clamp_x_px"] == [-80, 280]
    assert resolved["clamp_y_px"] == [-60, 180]


def test_resolve_layout_jitter_preserves_pixel_mode_by_default() -> None:
    jitter = resolve_layout_jitter(
        {"layout_jitter_enabled": True, "layout_jitter_x_px": 7, "layout_jitter_y_px": -5},
        {},
        instance_seed=123,
        namespace="test.layout",
    )

    assert jitter["mode"] == "pixel"
    assert jitter["requested_dx_px"] == 7
    assert jitter["requested_dy_px"] == -5
