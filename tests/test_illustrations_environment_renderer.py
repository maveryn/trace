"""Smoke tests for the shared illustrations environment renderer."""

from __future__ import annotations

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.illustrations.environment.shared.rendering import (
    ENVIRONMENT_THEME_IDS,
    LAND_OBJECT_TYPES,
    RIVER_OBJECT_TYPES,
    ROAD_OBJECT_TYPES,
    SKY_OBJECT_TYPES,
    environment_scene_entities,
    render_environment_object_scene,
    serialize_environment_scene,
)


def test_environment_renderer_themes_and_semantic_entities() -> None:
    expected_feature_types = {
        "park_road": {"road"},
        "river_meadow": {"river", "bridge"},
        "road_and_river": {"road", "river", "bridge"},
        "canal_city": {"river", "bridge"},
        "skyline_street": {"road", "crosswalk"},
    }
    for index, theme_id in enumerate(ENVIRONMENT_THEME_IDS):
        rng = spawn_rng(2026052001, "illustrations-environment-test", index)
        scene = render_environment_object_scene(
            rng=rng,
            object_count=10,
            theme_weights={theme: (1.0 if theme == theme_id else 0.0) for theme in ENVIRONMENT_THEME_IDS},
        )
        assert scene.image.size == (1280, 840)
        assert scene.theme_id == theme_id
        assert len(scene.objects) == 10
        feature_types = {feature.feature_type for feature in scene.features}
        assert expected_feature_types[theme_id].issubset(feature_types)
        if theme_id in {"canal_city", "skyline_street"}:
            assert scene.buildings
            assert all(building.window_bboxes for building in scene.buildings)
        entities = environment_scene_entities(scene)
        assert any(entity["entity_type"] == "illustration_object" for entity in entities)
        assert any(entity["entity_type"] == "environment_feature" for entity in entities)
        payload = serialize_environment_scene(scene)
        assert payload["scene_id"] == "environment"
        assert payload["theme_id"] == theme_id
        for placement in scene.placements:
            if placement.zone_id == "road":
                assert placement.object_type in ROAD_OBJECT_TYPES
            elif placement.zone_id == "river":
                assert placement.object_type in RIVER_OBJECT_TYPES
            elif placement.zone_id == "sky":
                assert placement.object_type in SKY_OBJECT_TYPES
            else:
                assert placement.object_type in LAND_OBJECT_TYPES
            if feature_types.intersection({"road", "river"}):
                assert placement.relations
                for relation in placement.relations.values():
                    if isinstance(relation, dict) and relation.get("feature_type") in {"road", "river"}:
                        assert len(relation["nearest_point_px"]) == 2
