from __future__ import annotations

from pathlib import Path
import sys
import types


def _install_trace_tasks_namespace() -> None:
    if "trace_tasks.tasks" in sys.modules:
        return
    repo_root = Path(__file__).resolve().parents[1]
    tasks_module = types.ModuleType("trace_tasks.tasks")
    tasks_module.__path__ = [str(repo_root / "trace" / "tasks")]  # type: ignore[attr-defined]
    sys.modules["trace_tasks.tasks"] = tasks_module


_install_trace_tasks_namespace()

from trace_tasks.tasks.illustrations.shared.pixel_isometric_farmstead_rendering import (  # noqa: E402
    CANONICAL_ISO_TILE_H_PX,
    CANONICAL_ISO_TILE_W_PX,
    CANONICAL_LEVEL_PX,
    render_pixel_isometric_farmstead,
)
from trace_tasks.tasks.illustrations.shared.pixel_world_objects import PIXEL_DOMESTIC_ANIMALS, PIXEL_PERSON_VARIANTS  # noqa: E402


def test_pixel_isometric_farmstead_renderer_is_deterministic_and_semantic() -> None:
    first = render_pixel_isometric_farmstead(20260604, width=960, height=720)
    second = render_pixel_isometric_farmstead(20260604, width=960, height=720)

    assert first.trace == second.trace
    assert first.image.size == (960, 720)
    assert first.trace["renderer_id"] == "pixel_isometric_farmstead_v0"
    assert first.trace["uses_external_sprites"] is False
    assert first.trace["projection"]["type"] == "2:1_isometric"
    assert first.trace["projection"]["canonical_tile_size_px"] == [CANONICAL_ISO_TILE_W_PX, CANONICAL_ISO_TILE_H_PX]
    assert first.trace["projection"]["display_tile_size_px"] == [64, 32]
    assert first.trace["projection"]["canonical_level_px"] == CANONICAL_LEVEL_PX
    assert first.trace["projection"]["display_level_px"] == 24
    assert 13 <= int(first.trace["grid_cols"]) <= 15
    assert 11 <= int(first.trace["grid_rows"]) <= 13
    assert first.trace["scale"] == 2
    upper_level = int(first.trace["upper_level"])
    assert upper_level in {1, 2}
    assert first.trace["lower_level"] == 0
    assert first.trace["elevation_depth"] == upper_level
    assert first.trace["supported_elevation_depths"] == [1, 2]
    assert first.trace["levels"] == [0, upper_level]
    assert int(first.trace["level_count"]) == 2
    assert int(first.trace["level_tile_counts"][str(upper_level)]) > 0
    assert int(first.trace["tile_count"]) == int(first.trace["grid_cols"]) * int(first.trace["grid_rows"])
    assert first.trace["terrain_counts"]["crop"] > 0
    assert first.trace["path_tiles"]
    assert first.trace["crop_tiles"]
    assert int(first.trace["retaining_wall_face_count"]) > 0
    assert len(first.trace["retaining_wall_faces"]) == int(first.trace["retaining_wall_face_count"])
    assert len(first.trace["transition_open_edges"]) == 2
    assert {edge["side"] for edge in first.trace["transition_open_edges"]} == {"south"}
    for wall_face in first.trace["retaining_wall_faces"]:
        assert wall_face["from_level"] == upper_level
        assert wall_face["to_level"] == 0
        assert wall_face["drop_px"] == upper_level * int(first.trace["projection"]["display_level_px"])
        assert len(wall_face["screen_polygon"]) == 4

    region_types = {region.region_type for region in first.regions}
    assert {"raised_terrace", "lower_field", "fenced_pen"}.issubset(region_types)
    regions = {region.region_id: region for region in first.regions}
    assert set(regions) == {"lower_field", "lower_pen", "upper_pen", "upper_terrace"}
    assert regions["upper_terrace"].level == upper_level
    assert regions["lower_field"].level == 0
    assert regions["upper_pen"].level == upper_level
    assert regions["lower_pen"].level == 0
    assert regions["upper_pen"].metadata["contains_fence_border"] is True
    assert regions["lower_pen"].metadata["contains_fence_border"] is True

    transitions = {transition.transition_type: transition for transition in first.transitions}
    assert set(transitions) == {"ramp", "stair"}
    for transition in first.transitions:
        assert transition.lower_level == 0
        assert transition.upper_level == upper_level
        assert transition.metadata["connects_region_ids"] == ["lower_field", "upper_terrace"]
        assert transition.metadata["elevation_depth"] == upper_level
        x0, y0, x1, y1 = transition.bbox_xyxy
        assert 0 <= x0 < x1 <= 960
        assert 0 <= y0 < y1 <= 720

    assert int(first.trace["entity_count"]) >= 20
    assert int(first.trace["animal_count"]) >= 7
    assert int(first.trace["person_count"]) == 2
    assert int(first.trace["inside_pen_animal_count"]) > 0
    assert int(first.trace["outside_pen_animal_count"]) > 0
    assert set(PIXEL_DOMESTIC_ANIMALS).issubset(set(first.trace["public_name_counts"]))
    assert {"animal", "building", "farm_fixture", "person", "plant"}.issubset(set(first.trace["category_counts"]))
    assert "barn_00" in first.trace["entity_draw_order"]
    assert "coop_00" in first.trace["entity_draw_order"]

    for region in first.regions:
        x0, y0, x1, y1 = region.bbox_xyxy
        assert 0 <= x0 < x1 <= 960
        assert 0 <= y0 < y1 <= 720
        assert len(region.polygon_xy) == 4
        assert region.metadata.get("level") == region.level

    for entity in first.entities:
        x0, y0, x1, y1 = entity.bbox_xyxy
        assert 0 <= x0 < x1 <= 960
        assert 0 <= y0 < y1 <= 720
        assert len(entity.footprint_polygon_xy) == 4
        assert entity.public_name
        assert entity.category
        assert entity.metadata.get("variant")
        assert entity.metadata.get("base_level") == entity.level
        anchor_x, anchor_y = entity.anchor_screen_xy
        assert 0 <= anchor_x <= 960
        assert 0 <= anchor_y <= 720
        if entity.category == "animal":
            assert entity.metadata.get("animal_type") in PIXEL_DOMESTIC_ANIMALS
            assert entity.metadata.get("region_id")
            assert entity.metadata.get("facing") in {"left", "right"}
            assert isinstance(entity.metadata.get("inside_pen"), bool)
            assert entity.metadata.get("body_rgb")
            record = entity.metadata.get("object_record")
            assert record["object_type"] == "domestic_animal"
            assert record["semantic_attributes"]["animal_type"] == entity.public_name
            assert record["semantic_attributes"]["inside_pen"] == entity.metadata.get("inside_pen")
            assert record["visual_attributes"]["renderer_style"] == "isometric_pixel_rpg"
        if entity.category == "building":
            assert entity.metadata.get("building_door_state") in {"open", "closed"}
            assert entity.metadata.get("building_facing") == "front_isometric"
        if entity.category == "person":
            assert entity.public_name == "person"
            assert entity.metadata.get("person_variant_id") in PIXEL_PERSON_VARIANTS
            assert entity.metadata.get("person_variant_id") in {"adult", "farmer", "worker"}
            assert entity.metadata.get("object_variant_id") == entity.metadata.get("person_variant_id")
            assert entity.metadata.get("renderer_style") == "isometric_pixel_rpg"
            assert entity.metadata.get("facing") in {"down", "up", "left", "right"}
            assert entity.metadata.get("skin_rgb")
            assert entity.metadata.get("object_record")["object_type"] == "person"
        if entity.public_name == "tree":
            assert entity.metadata.get("tree_style")
            assert entity.metadata.get("object_variant_id") == entity.metadata.get("tree_style")
            assert entity.metadata.get("renderer_style") == "isometric_pixel_rpg"
            assert entity.metadata.get("leaf_rgb")
            assert entity.metadata.get("object_record")["object_type"] == "tree"
        if entity.public_name == "flower":
            assert entity.metadata.get("flower_rgb")
            assert entity.metadata.get("leaf_rgb")
            assert entity.metadata.get("object_record")["object_type"] == "flower"
        if entity.category == "farm_fixture" and entity.public_name in {"crate", "hay bale", "trough"}:
            assert entity.metadata.get("object_record")["object_type"] == str(entity.metadata.get("variant"))
            assert entity.metadata.get("object_record")["visual_attributes"]["renderer_style"] == "isometric_pixel_rpg"
        if entity.category in {"animal", "person"} or entity.public_name in {"crate", "flower", "hay bale", "tree", "trough"}:
            payload = entity.as_dict()
            assert "object_record" in payload
            assert "object_record" not in payload["metadata"]


def test_pixel_isometric_farmstead_species_cover_reusable_variants() -> None:
    species: set[str] = set()
    for seed in range(20260604, 20260612):
        scene = render_pixel_isometric_farmstead(seed, width=960, height=720)
        species.update(entity.public_name for entity in scene.entities if entity.category == "animal")

    assert species == set(PIXEL_DOMESTIC_ANIMALS)


def test_pixel_isometric_farmstead_buildings_cover_door_states() -> None:
    door_states: set[str] = set()
    for seed in range(20260604, 20260624):
        scene = render_pixel_isometric_farmstead(seed, width=960, height=720)
        door_states.update(
            str(entity.metadata.get("building_door_state"))
            for entity in scene.entities
            if entity.category == "building"
        )

    assert door_states == {"open", "closed"}


def test_pixel_isometric_farmstead_samples_one_and_two_unit_elevation_depths() -> None:
    depths: set[int] = set()
    for seed in range(20260604, 20260620):
        scene = render_pixel_isometric_farmstead(seed, width=960, height=720)
        depths.add(int(scene.trace["elevation_depth"]))

    assert depths == {1, 2}


def test_pixel_isometric_farmstead_supports_explicit_grid_and_scale() -> None:
    scene = render_pixel_isometric_farmstead(
        123,
        width=960,
        height=720,
        scale=2,
        grid_cols=14,
        grid_rows=12,
        elevation_depth=2,
    )

    assert scene.trace["grid_cols"] == 14
    assert scene.trace["grid_rows"] == 12
    assert scene.trace["scale"] == 2
    assert scene.trace["canonical_canvas_size_px"] == [480, 360]
    assert scene.image.size == (960, 720)
    assert scene.trace["levels"] == [0, 2]
    assert scene.trace["upper_level"] == 2
    assert scene.trace["elevation_depth"] == 2
    assert scene.trace["level_tile_counts"]["2"] > 0
    assert {transition.upper_level for transition in scene.transitions} == {2}
    assert scene.trace["transition_count"] == 2
    assert scene.trace["inside_pen_animal_count"] > 0
    assert scene.trace["outside_pen_animal_count"] > 0
