from __future__ import annotations

from pathlib import Path
import sys
import types

from PIL import Image, ImageDraw


def _install_trace_tasks_namespace() -> None:
    if "trace_tasks.tasks" in sys.modules:
        return
    repo_root = Path(__file__).resolve().parents[1]
    tasks_module = types.ModuleType("trace_tasks.tasks")
    tasks_module.__path__ = [str(repo_root / "trace" / "tasks")]  # type: ignore[attr-defined]
    sys.modules["trace_tasks.tasks"] = tasks_module


_install_trace_tasks_namespace()

from trace_tasks.tasks.illustrations.shared.object_rendering import (  # noqa: E402
    PIXEL_RPG_SHARED_OBJECT_TYPES,
    IllustrationObjectSpec,
    RenderContext,
    render_illustration_object,
)
from trace_tasks.tasks.illustrations.shared.object_variants import (  # noqa: E402
    RENDERER_STYLE_ISOMETRIC_PIXEL_RPG,
    RENDERER_STYLE_TOP_DOWN_PIXEL_RPG,
)


_FOOTPRINTS: dict[str, tuple[int, int]] = {
    "archway": (2, 1),
    "barn": (4, 3),
    "basket": (1, 1),
    "bed": (2, 3),
    "bench": (2, 1),
    "boulder": (1, 1),
    "brazier": (1, 1),
    "broken_wall": (2, 1),
    "cart": (2, 1),
    "castle": (5, 4),
    "cave_entrance": (4, 3),
    "chest": (2, 1),
    "candle": (1, 1),
    "chair": (1, 1),
    "chicken_coop": (2, 2),
    "church": (4, 4),
    "coop": (2, 2),
    "counter": (3, 1),
    "crystal_cluster": (1, 1),
    "crop_row": (4, 1),
    "dead_tree": (1, 2),
    "domestic_animal": (2, 1),
    "fireplace": (2, 1),
    "floor_switch": (1, 1),
    "gazebo": (3, 3),
    "house": (4, 3),
    "inn": (4, 3),
    "jar": (1, 1),
    "lamp_post": (1, 2),
    "ladder": (1, 1),
    "magic_circle": (2, 2),
    "market_stall": (3, 2),
    "mine_cart": (2, 1),
    "notice_board": (2, 1),
    "ore_vein": (1, 1),
    "plate": (1, 1),
    "pond": (3, 2),
    "pot": (1, 1),
    "produce_bin": (2, 1),
    "rail_track": (1, 1),
    "rubble": (1, 1),
    "room_divider": (3, 1),
    "rug": (3, 2),
    "sack": (1, 1),
    "scarecrow": (1, 2),
    "sealed_door": (2, 1),
    "shelf": (3, 1),
    "shop": (4, 3),
    "stairs": (2, 2),
    "stalagmite": (1, 1),
    "statue": (2, 2),
    "stone_column": (1, 1),
    "stool": (1, 1),
    "table": (2, 2),
    "tower": (3, 4),
    "torch": (1, 1),
    "tree": (1, 2),
    "trough": (2, 1),
    "vegetable_patch": (1, 1),
    "wagon": (3, 2),
    "well": (2, 2),
    "windmill": (3, 4),
    "woodpile": (2, 1),
    "wood_support": (2, 2),
}


def _spec_for(object_type: str) -> IllustrationObjectSpec:
    width, height = _FOOTPRINTS.get(object_type, (1, 1))
    semantic_attributes = {"animal_type": "cow"} if object_type == "domestic_animal" else {}
    visual_attributes = {"animal_type": "cow", "facing": "right"} if object_type == "domestic_animal" else {}
    if object_type == "vegetable_patch":
        visual_attributes = {"vegetable_style": "carrot"}
    if object_type == "shelf":
        visual_attributes = {"goods_type": "mixed"}
    if object_type == "produce_bin":
        visual_attributes = {"goods_type": "fruit"}
    if object_type == "table":
        visual_attributes = {"table_shape": "square"}
    if object_type == "chair":
        visual_attributes = {"facing": "down"}
    if object_type == "bed":
        visual_attributes = {"bed_size": "single"}
    if object_type == "fireplace":
        visual_attributes = {"fire_state": "lit"}
    if object_type == "room_divider":
        visual_attributes = {"divider_style": "screen"}
    if object_type == "candle":
        visual_attributes = {"flame_state": "lit"}
    if object_type == "ladder":
        visual_attributes = {"orientation": "vertical"}
    if object_type == "mine_cart":
        visual_attributes = {"orientation": "horizontal"}
    if object_type == "rail_track":
        visual_attributes = {"track_shape": "horizontal"}
    if object_type == "stairs":
        visual_attributes = {"stair_direction": "down"}
    if object_type == "floor_switch":
        visual_attributes = {"switch_state": "raised"}
    if object_type == "broken_wall":
        visual_attributes = {"break_style": "cracked"}
    if object_type == "sealed_door":
        visual_attributes = {"door_orientation": "horizontal"}
    if object_type == "brazier":
        visual_attributes = {"fire_state": "lit"}
    return IllustrationObjectSpec(
        object_id=f"{object_type}_00",
        object_type=object_type,
        public_name=object_type.replace("_", " "),
        tile_xywh=(1, 1, width, height),
        semantic_attributes=semantic_attributes,
        visual_attributes=visual_attributes,
        source_entity_type="pixel_rpg_object_test",
    )


def _project_tile_center(tile_xywh: tuple[int, int, int, int], level: int) -> tuple[float, float]:
    x, y, w, h = tile_xywh
    center_x = x + (w - 1) * 0.5
    center_y = y + (h - 1) * 0.5
    return (90.0 + (center_x - center_y) * 16.0, 82.0 + (center_x + center_y) * 8.0 - level * 12.0)


def test_current_village_and_farm_prop_inventory_has_shared_pixel_rpg_object_support() -> None:
    expected = {
        "archway",
        "barrel",
        "barn",
        "basket",
        "bed",
        "bench",
        "boulder",
        "brazier",
        "broken_wall",
        "bridge",
        "candle",
        "cart",
        "castle",
        "cave_entrance",
        "cemetery_gate",
        "chair",
        "chest",
        "church",
        "coop",
        "counter",
        "crate",
        "crystal_cluster",
        "crop_row",
        "dead_tree",
        "domestic_animal",
        "farm_gate",
        "fence",
        "fireplace",
        "floor_switch",
        "flower",
        "fountain",
        "gazebo",
        "grave_marker",
        "hay_bale",
        "house",
        "inn",
        "iron_fence",
        "jar",
        "lamp_post",
        "ladder",
        "magic_circle",
        "market_stall",
        "mine_cart",
        "notice_board",
        "ore_vein",
        "plate",
        "person",
        "pond",
        "pot",
        "produce_bin",
        "rail_track",
        "rock",
        "rubble",
        "room_divider",
        "rug",
        "sack",
        "scarecrow",
        "sealed_door",
        "shelf",
        "shop",
        "sign",
        "stairs",
        "stalagmite",
        "statue",
        "stone_column",
        "stool",
        "table",
        "tower",
        "torch",
        "tree",
        "trough",
        "vegetable_patch",
        "wagon",
        "well",
        "windmill",
        "woodpile",
        "wood_support",
    }

    assert expected.issubset(set(PIXEL_RPG_SHARED_OBJECT_TYPES))
    assert "bottle" not in PIXEL_RPG_SHARED_OBJECT_TYPES
    assert "bowl" not in PIXEL_RPG_SHARED_OBJECT_TYPES
    assert "mug" not in PIXEL_RPG_SHARED_OBJECT_TYPES


def test_shared_pixel_rpg_objects_render_in_top_down_and_isometric_styles() -> None:
    for object_type in PIXEL_RPG_SHARED_OBJECT_TYPES:
        top_down = Image.new("RGBA", (128, 112), (0, 0, 0, 0))
        rendered_top_down = render_illustration_object(
            _spec_for(object_type),
            RenderContext(
                renderer_style=RENDERER_STYLE_TOP_DOWN_PIXEL_RPG,
                draw=ImageDraw.Draw(top_down, "RGBA"),
            ),
        )
        assert top_down.getbbox() is not None, object_type
        top_down_visual = rendered_top_down.object_record["visual_attributes"]
        assert top_down_visual["renderer_style"] == RENDERER_STYLE_TOP_DOWN_PIXEL_RPG
        assert top_down_visual["shadow_policy"] == "none"
        assert top_down_visual["shadow_enabled"] is False
        assert top_down_visual["shadow_kind"] == "none"

        isometric = Image.new("RGBA", (180, 140), (0, 0, 0, 0))

        rendered_iso = render_illustration_object(
            _spec_for(object_type),
            RenderContext(
                renderer_style=RENDERER_STYLE_ISOMETRIC_PIXEL_RPG,
                image=isometric,
                project_tile_center=_project_tile_center,
            ),
        )
        assert isometric.getbbox() is not None, object_type
        iso_visual = rendered_iso.object_record["visual_attributes"]
        assert iso_visual["renderer_style"] == RENDERER_STYLE_ISOMETRIC_PIXEL_RPG
        assert iso_visual["shadow_policy"] == "none"
        assert iso_visual["shadow_enabled"] is False
        assert iso_visual["shadow_kind"] == "none"


def test_feedback_target_top_down_dungeon_variants_render() -> None:
    specs = (
        IllustrationObjectSpec(
            object_id="floor_switch_pressed",
            object_type="floor_switch",
            public_name="floor switch",
            tile_xywh=(1, 1, 1, 1),
            visual_attributes={"switch_state": "pressed"},
            source_entity_type="pixel_rpg_object_test",
        ),
        IllustrationObjectSpec(
            object_id="floor_switch_raised",
            object_type="floor_switch",
            public_name="floor switch",
            tile_xywh=(1, 1, 1, 1),
            visual_attributes={"switch_state": "raised"},
            source_entity_type="pixel_rpg_object_test",
        ),
        IllustrationObjectSpec(
            object_id="magic_circle",
            object_type="magic_circle",
            public_name="magic circle",
            tile_xywh=(1, 1, 2, 2),
            source_entity_type="pixel_rpg_object_test",
        ),
        IllustrationObjectSpec(
            object_id="broken_wall_cracked",
            object_type="broken_wall",
            public_name="broken wall",
            tile_xywh=(1, 1, 2, 1),
            visual_attributes={"break_style": "cracked"},
            source_entity_type="pixel_rpg_object_test",
        ),
        IllustrationObjectSpec(
            object_id="broken_wall_gap",
            object_type="broken_wall",
            public_name="broken wall",
            tile_xywh=(1, 1, 2, 1),
            visual_attributes={"break_style": "gap"},
            source_entity_type="pixel_rpg_object_test",
        ),
        IllustrationObjectSpec(
            object_id="sealed_door_horizontal",
            object_type="sealed_door",
            public_name="sealed door",
            tile_xywh=(1, 1, 2, 1),
            visual_attributes={"door_orientation": "horizontal"},
            source_entity_type="pixel_rpg_object_test",
        ),
        IllustrationObjectSpec(
            object_id="sealed_door_vertical",
            object_type="sealed_door",
            public_name="sealed door",
            tile_xywh=(1, 1, 1, 2),
            visual_attributes={"door_orientation": "vertical"},
            source_entity_type="pixel_rpg_object_test",
        ),
    )
    for spec in specs:
        image = Image.new("RGBA", (128, 112), (0, 0, 0, 0))
        render_illustration_object(
            spec,
            RenderContext(
                renderer_style=RENDERER_STYLE_TOP_DOWN_PIXEL_RPG,
                draw=ImageDraw.Draw(image, "RGBA"),
            ),
        )
        assert image.getbbox() is not None, spec.object_id


def test_feedback_target_isometric_dungeon_variants_render() -> None:
    specs = (
        IllustrationObjectSpec(
            object_id="floor_switch_pressed",
            object_type="floor_switch",
            public_name="floor switch",
            tile_xywh=(1, 1, 1, 1),
            visual_attributes={"switch_state": "pressed"},
            source_entity_type="pixel_rpg_object_test",
        ),
        IllustrationObjectSpec(
            object_id="floor_switch_raised",
            object_type="floor_switch",
            public_name="floor switch",
            tile_xywh=(1, 1, 1, 1),
            visual_attributes={"switch_state": "raised"},
            source_entity_type="pixel_rpg_object_test",
        ),
        IllustrationObjectSpec(
            object_id="archway",
            object_type="archway",
            public_name="archway",
            tile_xywh=(1, 1, 2, 1),
            source_entity_type="pixel_rpg_object_test",
        ),
        IllustrationObjectSpec(
            object_id="broken_wall_cracked",
            object_type="broken_wall",
            public_name="broken wall",
            tile_xywh=(1, 1, 2, 1),
            visual_attributes={"break_style": "cracked"},
            source_entity_type="pixel_rpg_object_test",
        ),
        IllustrationObjectSpec(
            object_id="broken_wall_gap",
            object_type="broken_wall",
            public_name="broken wall",
            tile_xywh=(1, 1, 2, 1),
            visual_attributes={"break_style": "gap"},
            source_entity_type="pixel_rpg_object_test",
        ),
        IllustrationObjectSpec(
            object_id="sealed_door_horizontal",
            object_type="sealed_door",
            public_name="sealed door",
            tile_xywh=(1, 1, 2, 1),
            visual_attributes={"door_orientation": "horizontal"},
            source_entity_type="pixel_rpg_object_test",
        ),
        IllustrationObjectSpec(
            object_id="sealed_door_vertical",
            object_type="sealed_door",
            public_name="sealed door",
            tile_xywh=(1, 1, 1, 2),
            visual_attributes={"door_orientation": "vertical"},
            source_entity_type="pixel_rpg_object_test",
        ),
    )
    for spec in specs:
        image = Image.new("RGBA", (180, 140), (0, 0, 0, 0))
        render_illustration_object(
            spec,
            RenderContext(
                renderer_style=RENDERER_STYLE_ISOMETRIC_PIXEL_RPG,
                image=image,
                project_tile_center=_project_tile_center,
            ),
        )
        assert image.getbbox() is not None, spec.object_id


def test_pixel_rpg_chair_facings_render_in_rpg_styles() -> None:
    for facing in ("down", "left", "right"):
        spec = IllustrationObjectSpec(
            object_id=f"chair_{facing}",
            object_type="chair",
            public_name="chair",
            tile_xywh=(1, 1, 1, 1),
            visual_attributes={"facing": facing},
            source_entity_type="pixel_rpg_object_test",
        )

        top_down = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        render_illustration_object(
            spec,
            RenderContext(
                renderer_style=RENDERER_STYLE_TOP_DOWN_PIXEL_RPG,
                draw=ImageDraw.Draw(top_down, "RGBA"),
            ),
        )
        assert top_down.getbbox() is not None, facing

        isometric = Image.new("RGBA", (180, 140), (0, 0, 0, 0))
        render_illustration_object(
            spec,
            RenderContext(
                renderer_style=RENDERER_STYLE_ISOMETRIC_PIXEL_RPG,
                image=isometric,
                project_tile_center=_project_tile_center,
            ),
        )
        assert isometric.getbbox() is not None, facing


def test_feedback_target_pixel_rpg_objects_stay_within_top_down_tile_footprints() -> None:
    for object_type in {"bed", "bench", "cart", "crop_row", "fireplace", "notice_board", "rock", "room_divider", "table"}:
        spec = _spec_for(object_type)
        assert spec.tile_xywh is not None
        image = Image.new("RGBA", (128, 112), (0, 0, 0, 0))

        render_illustration_object(
            spec,
            RenderContext(
                renderer_style=RENDERER_STYLE_TOP_DOWN_PIXEL_RPG,
                draw=ImageDraw.Draw(image, "RGBA"),
            ),
        )

        tile_x, tile_y, tile_w, tile_h = spec.tile_xywh
        bbox = image.getbbox()
        assert bbox is not None, object_type
        assert bbox[0] >= tile_x * 16, object_type
        assert bbox[1] >= tile_y * 16, object_type
        assert bbox[2] <= (tile_x + tile_w) * 16, object_type
        assert bbox[3] <= (tile_y + tile_h) * 16, object_type


def test_isometric_pixel_rpg_renderer_does_not_draw_generic_shadow_below_sprite() -> None:
    spec = _spec_for("tree")
    assert spec.tile_xywh is not None
    center = _project_tile_center(spec.tile_xywh, spec.level)
    sprite_bottom = int(round(float(center[1]) + 6 - 32)) + 32
    image = Image.new("RGBA", (180, 140), (0, 0, 0, 0))

    render_illustration_object(
        spec,
        RenderContext(
            renderer_style=RENDERER_STYLE_ISOMETRIC_PIXEL_RPG,
            image=image,
            project_tile_center=_project_tile_center,
        ),
    )

    bbox = image.getbbox()
    assert bbox is not None
    assert bbox[3] <= sprite_bottom


def test_top_down_pixel_rpg_renderer_does_not_draw_house_ground_shadow_outside_tile() -> None:
    spec = _spec_for("house")
    assert spec.tile_xywh is not None
    image = Image.new("RGBA", (128, 112), (0, 0, 0, 0))

    render_illustration_object(
        spec,
        RenderContext(
            renderer_style=RENDERER_STYLE_TOP_DOWN_PIXEL_RPG,
            draw=ImageDraw.Draw(image, "RGBA"),
        ),
    )

    tile_x, tile_y, tile_w, tile_h = spec.tile_xywh
    bbox = image.getbbox()
    assert bbox is not None
    assert bbox[2] <= (tile_x + tile_w) * 16
    assert bbox[3] <= (tile_y + tile_h) * 16
