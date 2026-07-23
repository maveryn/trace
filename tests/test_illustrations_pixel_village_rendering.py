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

from trace_tasks.tasks.illustrations.shared.pixel_world_objects import (
    PIXEL_GRAVE_MARKER_STYLES,
    PIXEL_PERSON_VARIANTS,
    PIXEL_TREE_STYLES,
)
from trace_tasks.tasks.illustrations.pixel_village.shared.rendering import render_pixel_village_map


def test_pixel_village_renderer_is_deterministic_and_semantic() -> None:
    first = render_pixel_village_map(
        20260604,
        width=960,
        height=720,
        cemetery_mode="force",
        orchard_mode="force",
        windmill_mode="force",
    )
    second = render_pixel_village_map(
        20260604,
        width=960,
        height=720,
        cemetery_mode="force",
        orchard_mode="force",
        windmill_mode="force",
    )

    assert first.trace == second.trace
    assert first.image.size == (960, 720)
    assert first.trace["renderer_id"] == "pixel_village_map_v0"
    assert first.trace["uses_external_sprites"] is False
    assert first.trace["theme_mode"] == "temperate"
    assert first.trace["theme_id"] == "temperate"
    assert first.trace["snow_intensity"] == "none"
    assert first.trace["autumn_intensity"] == "none"
    assert 18 <= int(first.trace["grid_cols"]) <= 34
    assert 15 <= int(first.trace["grid_rows"]) <= 30
    assert first.trace["tile_px"] == 48
    assert first.trace["canonical_tile_px"] == 16
    assert first.trace["map_size_px"] == [960, 720]
    assert first.trace["map_offset_xy"] == [0, 0]
    assert first.trace["uses_outer_canvas_background"] is False
    assert first.trace["uses_pixel_frame_border"] is False
    assert first.trace["path_tiles"]
    assert int(first.trace["entity_count"]) >= 20
    assert int(first.trace["territory_count"]) == 2
    assert first.trace["cemetery_mode"] == "force"
    assert first.trace["cemetery_present"] is True
    assert int(first.trace["cemetery_grave_marker_count"]) >= 4
    assert first.trace["orchard_mode"] == "force"
    assert first.trace["orchard_present"] is True
    assert int(first.trace["orchard_tree_count"]) >= 4
    assert first.trace["windmill_mode"] == "force"
    assert first.trace["windmill_present"] is True
    assert first.trace["territory_type_counts"] == {"cemetery": 1, "orchard": 1}
    assert int(first.trace["category_counts"]["building"]) >= 2
    assert int(first.trace["category_counts"]["grave_marker"]) >= 4
    assert first.trace["public_name_counts"]
    assert first.trace["public_name_counts"]["cemetery gate"] == 1
    assert first.trace["public_name_counts"]["windmill"] == 1
    assert first.territories
    territories = {territory.territory_type: territory for territory in first.territories}
    assert territories["cemetery"].metadata["grave_marker_count"] == first.trace["cemetery_grave_marker_count"]
    assert territories["orchard"].metadata["tree_count"] == first.trace["orchard_tree_count"]
    assert territories["orchard"].metadata["row_count"] >= 1
    assert territories["orchard"].metadata["column_count"] >= 1

    categories = {entity.category for entity in first.entities}
    assert {"building", "landmark", "plant", "person", "grave_marker", "territory_feature"}.issubset(categories)
    people = [entity for entity in first.entities if entity.category == "person"]
    assert people
    assert {entity.metadata["facing"] for entity in people}.issubset({"down", "up", "left", "right"})
    assert {entity.metadata["gender_id"] for entity in people}.issubset({"male", "female"})
    assert {entity.metadata["person_variant_id"] for entity in people}.issubset(set(PIXEL_PERSON_VARIANTS))
    trees = [entity for entity in first.entities if entity.public_name == "tree"]
    assert trees
    orchard_trees = [entity for entity in trees if entity.metadata.get("territory_id") == "orchard_0"]
    assert orchard_trees
    assert all(entity.metadata.get("tree_style") == "fruit_tree" for entity in orchard_trees)
    tree_styles = {str(entity.metadata.get("tree_style")) for entity in trees}
    assert tree_styles.issubset(set(PIXEL_TREE_STYLES))
    flowers = [entity for entity in first.entities if entity.public_name == "flower"]
    assert flowers
    markers = [entity for entity in first.entities if entity.category == "grave_marker"]
    assert markers
    assert {str(entity.metadata.get("marker_style")) for entity in markers}.issubset(set(PIXEL_GRAVE_MARKER_STYLES))

    for entity in first.entities:
        x0, y0, x1, y1 = entity.bbox_xyxy
        assert 0 <= x0 < x1 <= 960
        assert 0 <= y0 < y1 <= 720
        assert entity.public_name
        assert entity.category
        assert entity.metadata.get("variant")
        if entity.category == "person":
            assert entity.metadata.get("skin_rgb")
            assert entity.metadata.get("shirt_rgb")
            assert entity.metadata.get("pants_rgb")
            assert entity.metadata.get("hair_rgb")
            assert entity.metadata.get("object_variant_id") == entity.metadata.get("person_variant_id")
            assert entity.metadata.get("renderer_style") == "top_down_pixel_rpg"
            assert entity.metadata.get("object_record")["object_type"] == "person"
        if entity.category == "building":
            assert entity.metadata.get("building_facing") == "front"
            assert entity.metadata.get("building_door_state") in {"open", "closed"}
            if entity.public_name in {"castle", "church", "tower"}:
                assert entity.metadata.get("building_roof_style") == "landmark"
                assert entity.metadata.get("building_wall_style") == "landmark"
            if entity.public_name in {"house", "shop", "inn"}:
                assert entity.metadata.get("building_roof_style") in {"shingle", "wood_plank", "tile"}
                assert entity.metadata.get("building_wall_style") in {"stucco", "wood", "stone"}
                _, _, w, h = entity.tile_xywh
                assert w > h
        if entity.public_name == "tree":
            assert entity.metadata.get("tree_style")
            assert entity.metadata.get("object_variant_id") == entity.metadata.get("tree_style")
            assert entity.metadata.get("renderer_style") == "top_down_pixel_rpg"
            assert entity.metadata.get("leaf_rgb")
            assert entity.metadata.get("object_record")["object_type"] == "tree"
        if entity.public_name == "flower":
            assert entity.metadata.get("flower_rgb")
            assert entity.metadata.get("leaf_rgb")
            assert entity.metadata.get("object_record")["object_type"] == "flower"
        if entity.category == "person" or entity.public_name in {"tree", "flower"}:
            payload = entity.as_dict()
            assert "object_record" in payload
            assert "object_record" not in payload["metadata"]
        if entity.public_name == "windmill":
            assert entity.category == "landmark"
            assert entity.metadata.get("blade_pose") in {"plus", "diagonal"}
            assert entity.metadata.get("body_rgb")
            assert entity.metadata.get("roof_rgb")
        if entity.public_name == "barrel":
            assert entity.category == "object"
            assert entity.metadata.get("barrel_rgb")
            assert entity.metadata.get("band_rgb")
        if entity.public_name == "bench":
            assert entity.category == "object"
            assert entity.metadata.get("orientation") in {"horizontal", "vertical"}
            assert entity.metadata.get("wood_rgb")
            assert entity.metadata.get("object_record")["object_type"] == "bench"
        if entity.public_name == "lamp post":
            assert entity.category == "landmark"
            assert entity.metadata.get("glow_rgb")
            assert entity.metadata.get("metal_rgb")
            assert entity.metadata.get("object_record")["object_type"] == "lamp_post"
        if entity.public_name == "notice board":
            assert entity.category == "landmark"
            assert entity.metadata.get("board_rgb")
            assert entity.metadata.get("paper_rgb")
        if entity.public_name == "cart":
            assert entity.category == "object"
            assert entity.metadata.get("facing") in {"left", "right"}
            assert entity.metadata.get("body_rgb")
        if entity.public_name == "market stall":
            assert entity.category == "landmark"
            assert entity.metadata.get("canopy_rgb")
            assert entity.metadata.get("wood_rgb")
            assert entity.metadata.get("goods_type") in {"fruit", "cloth", "crates"}
            assert entity.metadata.get("reference_source_ids")
            assert entity.metadata.get("object_record")["object_type"] == "market_stall"
        if entity.public_name == "wagon":
            assert entity.category == "object"
            assert entity.metadata.get("facing") in {"left", "right"}
            assert entity.metadata.get("body_rgb")
            assert entity.metadata.get("reference_source_ids")
            assert entity.metadata.get("object_record")["object_type"] == "wagon"
        if entity.public_name == "statue":
            assert entity.category == "landmark"
            assert entity.metadata.get("material_rgb")
            assert entity.metadata.get("reference_source_ids")
        if entity.public_name == "gazebo":
            assert entity.category == "landmark"
            assert entity.metadata.get("roof_rgb")
            assert entity.metadata.get("wood_rgb")
            assert entity.metadata.get("reference_source_ids")
            assert entity.metadata.get("object_record")["object_type"] == "gazebo"
        if entity.public_name == "woodpile":
            assert entity.category == "object"
            assert entity.metadata.get("log_rgb")
            assert entity.metadata.get("stack_variant") in {"low", "tall"}
            assert entity.metadata.get("reference_source_ids")
        if entity.public_name == "pond":
            assert entity.category == "landmark"
            assert entity.metadata.get("pond_shape") in {"round", "long", "kidney"}
            assert entity.metadata.get("water_rgb")
            assert entity.metadata.get("rim_rgb")
            assert entity.metadata.get("reference_source_ids")
            assert entity.metadata.get("object_record")["object_type"] == "pond"
        if entity.category == "grave_marker":
            assert entity.metadata.get("marker_style") in PIXEL_GRAVE_MARKER_STYLES
            assert entity.metadata.get("stone_rgb")
            assert entity.metadata.get("territory_id") == "cemetery_0"


def test_pixel_village_tree_styles_cover_reusable_variants() -> None:
    styles: set[str] = set()
    for seed in range(20260604, 20260610):
        scene = render_pixel_village_map(seed, width=960, height=720)
        styles.update(str(entity.metadata.get("tree_style")) for entity in scene.entities if entity.public_name == "tree")

    assert styles == set(PIXEL_TREE_STYLES)


def test_pixel_village_forced_balanced_river_orientation() -> None:
    vertical = render_pixel_village_map(
        20260610,
        width=960,
        height=720,
        river_mode="force",
        river_orientation="vertical",
        river_placement="balanced",
    )
    horizontal = render_pixel_village_map(
        20260611,
        width=960,
        height=720,
        river_mode="force",
        river_orientation="horizontal",
        river_placement="balanced",
    )

    assert vertical.trace["river_present"] is True
    assert vertical.trace["river_orientation"] == "vertical"
    assert vertical.trace["river_placement"] == "balanced"
    assert vertical.trace["river_bounds"]
    assert vertical.trace["bridge_box"]
    vertical_bounds = vertical.trace["river_bounds"]
    vertical_mid = int(vertical.trace["grid_cols"]) // 2
    assert abs(int(vertical_bounds["min_x"]) - vertical_mid) <= 2
    assert int(vertical_bounds["max_x"]) - int(vertical_bounds["min_x"]) == 1

    assert horizontal.trace["river_present"] is True
    assert horizontal.trace["river_orientation"] == "horizontal"
    assert horizontal.trace["river_placement"] == "balanced"
    assert horizontal.trace["river_bounds"]
    assert horizontal.trace["bridge_box"]
    horizontal_bounds = horizontal.trace["river_bounds"]
    horizontal_mid = int(horizontal.trace["grid_rows"]) // 2
    assert abs(int(horizontal_bounds["min_y"]) - horizontal_mid) <= 2
    assert int(horizontal_bounds["max_y"]) - int(horizontal_bounds["min_y"]) == 1


def test_pixel_village_person_variants_cover_reusable_variants() -> None:
    variants: set[str] = set()
    for seed in range(20260604, 20260624):
        scene = render_pixel_village_map(seed, width=960, height=720)
        variants.update(str(entity.metadata.get("person_variant_id")) for entity in scene.entities if entity.category == "person")

    assert variants == set(PIXEL_PERSON_VARIANTS)


def test_pixel_village_covers_new_building_and_plant_templates() -> None:
    public_names: set[str] = set()
    for seed in range(20260604, 20260680):
        scene = render_pixel_village_map(seed, width=960, height=720)
        public_names.update(entity.public_name for entity in scene.entities)

    assert {"castle", "church", "flower"}.issubset(public_names)


def test_pixel_village_covers_small_path_side_prop_templates() -> None:
    public_names: set[str] = set()
    bench_orientations: set[str] = set()
    cart_facings: set[str] = set()
    for seed in range(20260604, 20260616):
        scene = render_pixel_village_map(
            seed,
            width=960,
            height=720,
            cemetery_mode="none",
            orchard_mode="none",
            windmill_mode="none",
        )
        public_names.update(entity.public_name for entity in scene.entities)
        bench_orientations.update(
            str(entity.metadata.get("orientation")) for entity in scene.entities if entity.public_name == "bench"
        )
        cart_facings.update(str(entity.metadata.get("facing")) for entity in scene.entities if entity.public_name == "cart")

    assert {"barrel", "bench", "lamp post", "notice board", "cart"}.issubset(public_names)
    assert bench_orientations.issubset({"horizontal", "vertical"})
    assert bench_orientations
    assert cart_facings.issubset({"left", "right"})
    assert cart_facings


def test_pixel_village_covers_large_prop_templates_and_pond_variants() -> None:
    public_names: set[str] = set()
    pond_shapes: set[str] = set()
    pond_sizes: set[tuple[int, int]] = set()
    for seed in range(20260604, 20260624):
        scene = render_pixel_village_map(
            seed,
            width=960,
            height=720,
            cemetery_mode="none",
            orchard_mode="none",
            windmill_mode="none",
        )
        public_names.update(entity.public_name for entity in scene.entities)
        for entity in scene.entities:
            if entity.public_name == "pond":
                pond_shapes.add(str(entity.metadata.get("pond_shape")))
                _, _, w, h = entity.tile_xywh
                pond_sizes.add((w, h))

    assert {"market stall", "wagon", "statue", "gazebo", "woodpile", "pond"}.issubset(public_names)
    assert len(pond_shapes) >= 2
    assert len(pond_sizes) >= 2


def test_pixel_village_winter_theme_keeps_layout_and_records_visual_metadata() -> None:
    kwargs = {
        "width": 960,
        "height": 720,
        "cemetery_mode": "none",
        "orchard_mode": "none",
        "windmill_mode": "none",
    }
    temperate = render_pixel_village_map(20260604, theme_mode="temperate", **kwargs)
    winter = render_pixel_village_map(20260604, theme_mode="winter", **kwargs)

    assert temperate.trace["theme_id"] == "temperate"
    assert winter.trace["theme_id"] == "winter"
    assert winter.trace["snow_intensity"] in {"light", "medium", "heavy"}
    assert temperate.trace["path_tiles"] == winter.trace["path_tiles"]
    assert temperate.trace["water_tiles"] == winter.trace["water_tiles"]
    assert temperate.trace["public_name_counts"] == winter.trace["public_name_counts"]
    assert [(entity.public_name, entity.tile_xywh) for entity in temperate.entities] == [
        (entity.public_name, entity.tile_xywh) for entity in winter.entities
    ]

    covered_public_names = {"tree", "bench", "lamp post", "market stall", "wagon", "gazebo", "pond"}
    guaranteed_public_names = {"tree", "lamp post", "pond"}
    covered = [entity for entity in winter.entities if entity.public_name in covered_public_names or entity.category == "building"]
    assert covered
    assert any(entity.category == "building" for entity in covered)
    assert guaranteed_public_names.issubset({entity.public_name for entity in covered})
    for entity in covered:
        assert entity.metadata.get("theme_id") == "winter"
        assert entity.metadata.get("snow_intensity") == winter.trace["snow_intensity"]
        assert 0.0 < float(entity.metadata.get("snow_coverage", 0.0)) <= 1.0
        assert entity.metadata.get("snow_style")
        assert entity.metadata.get("snow_rgb")
        assert entity.metadata.get("snow_shadow_rgb")
        if entity.public_name in covered_public_names:
            record = entity.metadata.get("object_record")
            assert record
            assert record["visual_attributes"]["theme_id"] == "winter"
            assert record["visual_attributes"]["snow_intensity"] == winter.trace["snow_intensity"]
            assert record["visual_attributes"]["renderer_style"] == "top_down_pixel_rpg"


def test_pixel_village_autumn_theme_keeps_layout_and_records_visual_metadata() -> None:
    kwargs = {
        "width": 960,
        "height": 720,
        "cemetery_mode": "none",
        "orchard_mode": "none",
        "windmill_mode": "none",
    }
    temperate = render_pixel_village_map(20260604, theme_mode="temperate", **kwargs)
    autumn = render_pixel_village_map(20260604, theme_mode="autumn", **kwargs)

    assert temperate.trace["theme_id"] == "temperate"
    assert autumn.trace["theme_id"] == "autumn"
    assert autumn.trace["autumn_intensity"] in {"early", "peak", "late"}
    assert autumn.trace["snow_intensity"] == "none"
    assert temperate.trace["path_tiles"] == autumn.trace["path_tiles"]
    assert temperate.trace["water_tiles"] == autumn.trace["water_tiles"]
    assert temperate.trace["public_name_counts"] == autumn.trace["public_name_counts"]
    assert [(entity.public_name, entity.tile_xywh) for entity in temperate.entities] == [
        (entity.public_name, entity.tile_xywh) for entity in autumn.entities
    ]

    covered_public_names = {"tree", "flower", "bench", "market stall", "wagon", "gazebo", "pond"}
    covered = [entity for entity in autumn.entities if entity.public_name in covered_public_names]
    assert covered
    assert "tree" in {entity.public_name for entity in covered}
    for entity in covered:
        assert entity.metadata.get("theme_id") == "autumn"
        assert entity.metadata.get("autumn_intensity") == autumn.trace["autumn_intensity"]
        assert 0.0 < float(entity.metadata.get("leaf_coverage", 0.0)) <= 0.5
        assert entity.metadata.get("leaf_style")
        assert entity.metadata.get("leaf_overlay_rgb")
        assert entity.metadata.get("leaf_shadow_rgb")
        assert entity.metadata.get("leaf_accent_rgb")
        record = entity.metadata.get("object_record")
        assert record
        assert record["visual_attributes"]["theme_id"] == "autumn"
        assert record["visual_attributes"]["autumn_intensity"] == autumn.trace["autumn_intensity"]
        assert record["visual_attributes"]["renderer_style"] == "top_down_pixel_rpg"


def test_pixel_village_buildings_cover_front_facing_style_variants() -> None:
    facings: set[str] = set()
    roof_styles: set[str] = set()
    wall_styles: set[str] = set()
    door_states: set[str] = set()
    for seed in range(20260604, 20260620):
        scene = render_pixel_village_map(seed, width=960, height=720, cemetery_mode="none", orchard_mode="none")
        for entity in scene.entities:
            if entity.category == "building":
                facing = str(entity.metadata.get("building_facing"))
                facings.add(facing)
                door_states.add(str(entity.metadata.get("building_door_state")))
                if entity.public_name in {"house", "shop", "inn"}:
                    roof_styles.add(str(entity.metadata.get("building_roof_style")))
                    wall_styles.add(str(entity.metadata.get("building_wall_style")))

    assert facings == {"front"}
    assert door_states == {"open", "closed"}
    assert {"shingle", "wood_plank", "tile"}.issubset(roof_styles)
    assert {"stucco", "wood", "stone"}.issubset(wall_styles)


def test_pixel_village_supports_territory_absent_and_present_modes() -> None:
    absent = render_pixel_village_map(
        123,
        width=960,
        height=720,
        cemetery_mode="none",
        orchard_mode="none",
        windmill_mode="none",
    )
    present = render_pixel_village_map(
        123,
        width=960,
        height=720,
        cemetery_mode="force",
        orchard_mode="force",
        windmill_mode="force",
    )

    assert absent.trace["cemetery_present"] is False
    assert absent.trace["cemetery_grave_marker_count"] == 0
    assert absent.trace["orchard_present"] is False
    assert absent.trace["orchard_tree_count"] == 0
    assert absent.trace["windmill_present"] is False
    assert absent.territories == ()
    assert all(entity.category != "grave_marker" for entity in absent.entities)
    assert all(entity.metadata.get("territory_id") != "orchard_0" for entity in absent.entities)

    assert present.trace["cemetery_present"] is True
    assert present.trace["orchard_present"] is True
    assert present.trace["windmill_present"] is True
    assert len(present.territories) == 2
    assert any(entity.category == "grave_marker" for entity in present.entities)
    assert any(entity.metadata.get("territory_id") == "orchard_0" for entity in present.entities)
    assert any(entity.public_name == "windmill" for entity in present.entities)


def test_pixel_village_orchard_uses_variable_sizes() -> None:
    sizes: set[tuple[int, int]] = set()
    for seed in range(20260604, 20260612):
        scene = render_pixel_village_map(seed, width=1296, height=864, cemetery_mode="none", orchard_mode="force")
        orchards = [territory for territory in scene.territories if territory.territory_type == "orchard"]
        assert len(orchards) == 1
        _, _, w, h = orchards[0].tile_xywh
        sizes.add((w, h))
        assert int(orchards[0].metadata["tree_count"]) >= 4

    assert len(sizes) >= 3


def test_pixel_village_renderer_supports_explicit_full_bleed_grid() -> None:
    scene = render_pixel_village_map(
        123,
        width=960,
        height=720,
        tile_px=32,
        grid_cols=32,
        grid_rows=24,
        cemetery_mode="none",
        orchard_mode="none",
        windmill_mode="none",
    )

    assert scene.trace["grid_cols"] == 32
    assert scene.trace["grid_rows"] == 24
    assert scene.trace["tile_px"] == 30
    assert scene.trace["map_size_px"] == [960, 720]
    assert scene.trace["map_offset_xy"] == [0, 0]
    assert scene.image.size == (960, 720)
