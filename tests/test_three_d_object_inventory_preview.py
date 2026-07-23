from collections import Counter

import pytest

from trace_tasks.tasks.three_d.shared.object_inventory_preview import render_three_d_object_profile_preview
from trace_tasks.tasks.three_d.shared.object_resources import THREE_D_OBJECT_PROFILES


def test_inventory_preview_dispatches_all_registered_profiles_to_native_adapters():
    errors = []
    preview_renderer_counts = Counter()
    for profile in THREE_D_OBJECT_PROFILES:
        try:
            preview = render_three_d_object_profile_preview(
                profile,
                canvas_width=320,
                canvas_height=250,
                instance_seed=17,
            )
        except Exception as exc:  # pragma: no cover - failure aggregation for clearer review output.
            errors.append(f"{profile.profile_id}: {type(exc).__name__}: {exc}")
            continue
        preview_renderer_counts[str(preview.preview_renderer)] += 1
        assert preview.image.width > 0
        assert preview.image.height > 0
        assert preview.object_bbox_px[2] > preview.object_bbox_px[0]
        assert preview.object_bbox_px[3] > preview.object_bbox_px[1]
        assert preview.metadata["profile_renderer"] == profile.renderer
        assert preview.metadata["profile_source_scene"] == profile.source_scene

    assert errors == []
    assert preview_renderer_counts == Counter(str(profile.renderer) for profile in THREE_D_OBJECT_PROFILES)


def test_street_candidate_inventory_previews_use_candidate_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_candidate(draw, spec, *, camera, frame):
        calls.append(("candidate", str(spec["object_type"]), str(spec.get("object_role", ""))))
        return [10.0, 10.0, 42.0, 42.0]

    def fake_context(draw, spec, *, camera, frame):
        calls.append(("context", str(spec["object_type"]), str(spec.get("object_role", ""))))
        return [10.0, 10.0, 42.0, 42.0]

    monkeypatch.setattr("trace_tasks.tasks.three_d.shared.street_object_rendering._draw_candidate_object", fake_candidate)
    monkeypatch.setattr("trace_tasks.tasks.three_d.shared.street_object_rendering._draw_context_object", fake_context)
    profile = next(
        profile
        for profile in THREE_D_OBJECT_PROFILES
        if profile.profile_id == "street:street_candidate:bicycle"
    )

    preview = render_three_d_object_profile_preview(
        profile,
        canvas_width=320,
        canvas_height=250,
        instance_seed=17,
    )

    assert preview.preview_renderer == "street_object"
    assert calls == [("candidate", "bicycle", "street_candidate")]


def test_surface_fixture_inventory_preview_uses_fixture_renderer() -> None:
    profile = next(
        profile
        for profile in THREE_D_OBJECT_PROFILES
        if profile.profile_id == "surface_fixture:surface_fixture_variant:solar_panel_array"
    )

    preview = render_three_d_object_profile_preview(
        profile,
        canvas_width=320,
        canvas_height=250,
        instance_seed=17,
    )

    assert preview.preview_renderer == "surface_fixture"
    assert preview.metadata["scene_variant"] == "solar_panel_array"
    assert preview.metadata["element_type"] == "solar_panel"
    assert preview.object_bbox_px[2] > preview.object_bbox_px[0]
    assert preview.object_bbox_px[3] > preview.object_bbox_px[1]
