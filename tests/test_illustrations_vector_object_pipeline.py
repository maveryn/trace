from __future__ import annotations

from pathlib import Path
import random

from PIL import Image, ImageDraw

from trace_tasks.tasks.illustrations.environment.shared.rendering import (
    ENVIRONMENT_THEME_IDS,
    environment_scene_entities,
    render_environment_object_scene,
)
from trace_tasks.tasks.illustrations.environment.shared.output import serialize_environment_objects
from trace_tasks.tasks.illustrations.indoor_room.shared.rendering import (
    IndoorObjectSpec,
    indoor_scene_entities,
    render_indoor_room_scene,
)
from trace_tasks.tasks.illustrations.construction_site.shared.output import construction_scene_entities
from trace_tasks.tasks.illustrations.construction_site.shared.rendering import render_construction_site_scene
from trace_tasks.tasks.illustrations.construction_site.shared.state import (
    CONSTRUCTION_EQUIPMENT_TYPES,
    CONSTRUCTION_MATERIAL_TYPES,
    CONSTRUCTION_SETTING_IDS,
    CONSTRUCTION_ZONE_TYPES,
    ConstructionEquipmentSpec,
    ConstructionMaterialSpec,
    ConstructionWorkerSpec,
)
from trace_tasks.tasks.illustrations.library.shared.annotations import library_scene_entities
from trace_tasks.tasks.illustrations.library.shared.rendering import render_library_scene
from trace_tasks.tasks.illustrations.library.shared.state import LibraryBookSpec, LibrarySectionSpec
from trace_tasks.tasks.illustrations.shared.mixed_object_rendering import (
    ObjectPlacementSpec,
    render_mixed_object_scene,
    scene_entities as mixed_scene_entities,
)
from trace_tasks.tasks.illustrations.shared.object_rendering import (
    IllustrationObjectSpec,
    RenderContext,
    render_illustration_object,
    vector_object_record_for_spec,
)
from trace_tasks.tasks.illustrations.shared.object_library import draw_illustration_object, serialize_object
from trace_tasks.tasks.illustrations.shared.object_variants import RENDERER_STYLE_VECTOR
from trace_tasks.tasks.illustrations.park_playground.shared.annotations import park_scene_entities
from trace_tasks.tasks.illustrations.park_playground.shared.rendering import (
    PARK_EQUIPMENT_TYPES,
    ParkEquipmentSpec,
    ParkPersonSpec,
    render_park_playground_scene,
)


def _assert_vector_object_record(entity: dict) -> None:
    record = entity["object_record"]
    assert record["bbox"]
    assert record["visual_attributes"]["renderer_style"] == "vector"
    assert record["visual_attributes"]["style_id"]
    assert record["source_entity_type"]


def test_vector_object_records_capture_style_shadow_policy() -> None:
    spec = IllustrationObjectSpec(
        object_id="tree_00",
        object_type="tree",
        bbox_xyxy=(20, 30, 70, 120),
        source_entity_type="shadow_policy_test",
    )

    flat_visual = vector_object_record_for_spec(spec, style_id="flat_vector")["visual_attributes"]
    assert flat_visual["shadow_policy"] == "vector_style_decorative"
    assert flat_visual["shadow_enabled"] is False
    assert flat_visual["shadow_kind"] == "none"

    shadow_visual = vector_object_record_for_spec(spec, style_id="soft_shadow")["visual_attributes"]
    assert shadow_visual["shadow_policy"] == "vector_style_decorative"
    assert shadow_visual["shadow_enabled"] is True
    assert shadow_visual["shadow_kind"] == "decorative_oval"


def test_registered_vector_person_renderers_emit_parts_and_renderer_metadata() -> None:
    image = Image.new("RGB", (360, 180), (245, 246, 242))
    draw = ImageDraw.Draw(image)
    context = RenderContext(renderer_style=RENDERER_STYLE_VECTOR, draw=draw, render_scale=1, style_id="flat_vector")
    cases = (
        IllustrationObjectSpec(
            object_id="park_00",
            object_type="person",
            bbox_xyxy=(24, 24, 62, 124),
            renderer_id="park_person",
            renderer_variant_id="playing_ball",
            semantic_attributes={"activity": "playing_ball"},
            visual_attributes={"primary_color_rgb": [74, 122, 180], "accent_color_rgb": [220, 169, 75], "gender_id": "male"},
            source_entity_type="park_person",
        ),
        IllustrationObjectSpec(
            object_id="worker_00",
            object_type="worker",
            bbox_xyxy=(240, 24, 288, 136),
            renderer_id="construction_worker",
            semantic_attributes={"hard_hat_color": "yellow", "vest_color": "orange", "tool_type": "hammer"},
            visual_attributes={"hard_hat_color_rgb": [238, 194, 64], "vest_color_rgb": [232, 126, 54], "gender_id": "male"},
            source_entity_type="construction_worker",
        ),
    )

    rendered = [render_illustration_object(spec, context) for spec in cases]

    assert [item.object_record["visual_attributes"]["renderer_id"] for item in rendered] == [
        "park_person",
        "construction_worker",
    ]
    for item in rendered:
        part_kinds = {part["part_kind"] for part in item.parts}
        assert {"head", "arm", "leg"} <= part_kinds
        assert item.object_record["parts"] == list(item.parts)


def test_registered_vector_object_renderers_emit_renderer_metadata() -> None:
    image = Image.new("RGB", (680, 360), (245, 246, 242))
    draw = ImageDraw.Draw(image)
    context = RenderContext(renderer_style=RENDERER_STYLE_VECTOR, draw=draw, render_scale=1, style_id="flat_vector")
    cases = (
        IllustrationObjectSpec(
            object_id="bench_00",
            object_type="bench",
            bbox_xyxy=(20, 30, 150, 94),
            renderer_id="fixture_bench",
            renderer_variant_id="bench",
            semantic_attributes={"fixture_type": "bench"},
            visual_attributes={"style_id": "flat_vector"},
            source_entity_type="park_decor",
        ),
        IllustrationObjectSpec(
            object_id="equipment_00",
            object_type="playground_equipment",
            bbox_xyxy=(280, 24, 430, 126),
            renderer_id="park_equipment",
            renderer_variant_id="slide",
            semantic_attributes={"equipment_type": "slide"},
            visual_attributes={"primary_color_rgb": [216, 96, 82], "accent_color_rgb": [229, 168, 67]},
            source_entity_type="park_decor",
        ),
        IllustrationObjectSpec(
            object_id="material_00",
            object_type="construction_material",
            bbox_xyxy=(34, 150, 134, 220),
            renderer_id="construction_material",
            renderer_variant_id="brick_stack",
            semantic_attributes={"material_type": "brick_stack"},
            visual_attributes={"style_id": "flat_vector"},
            source_entity_type="construction_material",
        ),
        IllustrationObjectSpec(
            object_id="machine_00",
            object_type="construction_equipment",
            bbox_xyxy=(220, 136, 380, 232),
            renderer_id="construction_equipment",
            renderer_variant_id="dump_truck",
            semantic_attributes={"equipment_type": "dump_truck"},
            visual_attributes={"style_id": "flat_vector"},
            source_entity_type="construction_equipment",
        ),
        IllustrationObjectSpec(
            object_id="indoor_furniture_00",
            object_type="furniture",
            bbox_xyxy=(410, 136, 570, 246),
            renderer_id="indoor_furniture",
            renderer_variant_id="table",
            semantic_attributes={"furniture_type": "table"},
            visual_attributes={
                "style_id": "flat_vector",
                "draw_phase": "all",
                "rug_bbox": (386, 236, 594, 314),
                "surface_bbox": (410, 136, 570, 178),
                "leg_width": 14,
            },
            source_entity_type="indoor_furniture",
        ),
        IllustrationObjectSpec(
            object_id="indoor_surface_00",
            object_type="surface",
            bbox_xyxy=(28, 270, 168, 326),
            renderer_id="indoor_surface",
            renderer_variant_id="table",
            semantic_attributes={"surface_type": "table"},
            visual_attributes={
                "style_id": "flat_vector",
                "plane": {
                    "back_left": (42, 274),
                    "back_right": (154, 274),
                    "front_left": (28, 298),
                    "front_right": (168, 298),
                },
                "lip_bottom_y": 326,
            },
            source_entity_type="indoor_surface",
        ),
        IllustrationObjectSpec(
            object_id="indoor_container_00",
            object_type="container",
            bbox_xyxy=(206, 264, 302, 342),
            renderer_id="indoor_container",
            renderer_variant_id="basket",
            semantic_attributes={"container_type": "basket"},
            visual_attributes={"style_id": "flat_vector", "container_style": "woven"},
            source_entity_type="indoor_container",
        ),
    )

    rendered = [render_illustration_object(spec, context) for spec in cases]

    assert [item.object_record["visual_attributes"]["renderer_id"] for item in rendered] == [
        "fixture_bench",
        "park_equipment",
        "construction_material",
        "construction_equipment",
        "indoor_furniture",
        "indoor_surface",
        "indoor_container",
    ]
    assert [item.object_record["visual_attributes"]["renderer_variant_id"] for item in rendered] == [
        "bench",
        "slide",
        "brick_stack",
        "dump_truck",
        "table",
        "table",
        "basket",
    ]
    assert {part["part_kind"] for part in rendered[0].parts} == {"leg"}
    for item in rendered:
        assert item.object_record["visual_attributes"]["renderer_style"] == "vector"
        assert item.object_record["parts"] == list(item.parts)


def test_reviewed_vector_library_objects_emit_stable_parts() -> None:
    image = Image.new("RGB", (720, 360), (245, 246, 242))
    draw = ImageDraw.Draw(image)
    cases = {
        "book": {},
        "soccer_ball": {},
        "table": {"leg": 4},
        "trash_bin": {},
        "umbrella": {"handle": 1},
        "lily_pad": {"leaf": 1, "flower": 1},
        "tree": {"leaf": 6},
        "bicycle": {"wheel": 2, "handle": 1, "light": 1},
        "boat": {"window": 2},
    }

    for index, (object_type, expected_parts) in enumerate(cases.items()):
        row, col = divmod(index, 3)
        x0 = 24 + col * 220
        y0 = 24 + row * 104
        rendered = draw_illustration_object(
            draw,
            object_id=f"{object_type}_00",
            object_type=object_type,
            bbox_xyxy=(x0, y0, x0 + 120, y0 + 84),
            primary_color_rgb=(86, 126, 177),
            accent_color_rgb=(245, 190, 88),
            style_id="flat_vector",
            render_scale=1,
            object_variant_id="maple" if object_type == "tree" else None,
        )
        payload = serialize_object(rendered)
        bbox = payload["object_record"]["bbox"]
        assert bbox[2] > bbox[0]
        assert bbox[3] > bbox[1]
        part_counts: dict[str, int] = {}
        for part in payload["object_record"]["parts"]:
            part_counts[str(part["part_kind"])] = part_counts.get(str(part["part_kind"]), 0) + 1
        for part_kind, expected_count in expected_parts.items():
            assert part_counts.get(part_kind) == expected_count


def test_elongated_indoor_vector_objects_have_countable_short_side() -> None:
    image = Image.new("RGB", (420, 80), (245, 246, 242))
    draw = ImageDraw.Draw(image)

    for index, object_type in enumerate(("pencil", "ruler", "spoon", "key")):
        x0 = 12 + index * 100
        rendered = draw_illustration_object(
            draw,
            object_id=f"{object_type}_small",
            object_type=object_type,
            bbox_xyxy=(float(x0), 20.0, float(x0 + 90), 50.0),
            primary_color_rgb=(86, 126, 177),
            accent_color_rgb=(245, 190, 88),
            style_id="flat_vector",
            render_scale=1,
        )
        x0_box, y0_box, x1_box, y1_box = [float(value) for value in rendered.bbox_xyxy]
        assert min(x1_box - x0_box, y1_box - y0_box) >= 24.0


def test_source_layout_scene_person_renderers_do_not_keep_local_duplicate_drawers() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    banned = {
        "src/trace_tasks/tasks/illustrations/park_playground/shared/rendering.py": (
            "_draw_activity_person",
            "def _draw_bench(",
            "def _draw_climber(",
            "def _draw_equipment(",
            "def _draw_seesaw(",
            "def _draw_slide(",
            "def _draw_swing(",
        ),
        "src/trace_tasks/tasks/illustrations/construction_site/shared/rendering.py": (
            "_draw_worker",
            "def _draw_material(",
            "def _draw_equipment(",
        ),
        "src/trace_tasks/tasks/illustrations/indoor_room/shared/rendering.py": (
            "def _draw_surface_plane(",
            "def _draw_room_furniture(",
        ),
    }
    for relative_path, needles in banned.items():
        content = (repo_root / relative_path).read_text()
        for needle in needles:
            assert needle not in content



def test_mixed_object_scene_uses_shared_vector_records() -> None:
    scene = render_mixed_object_scene(
        placements=(
            ObjectPlacementSpec("obj_00", "person", (40, 30, 82, 128), (72, 115, 166), (245, 190, 88), "flat_vector"),
            ObjectPlacementSpec("obj_01", "tree", (150, 42, 230, 142), (52, 143, 84), (225, 92, 76), "flat_vector"),
        ),
        rng=random.Random(101),
        canvas_width=320,
        canvas_height=220,
        background_weights={"studio": 1.0},
        render_scale=1,
        content_bbox=(16, 16, 304, 204),
        background_id="studio",
    )

    object_entities = [entity for entity in mixed_scene_entities(scene) if entity["entity_type"] == "illustration_object"]
    assert len(object_entities) == 2
    assert any(entity["object_record"]["visual_attributes"].get("gender_id") for entity in object_entities)
    assert any(entity["entity_type"] == "illustration_part" for entity in mixed_scene_entities(scene))
    for entity in object_entities:
        _assert_vector_object_record(entity)


def test_environment_scene_uses_shared_vector_records_in_entities_and_render_map() -> None:
    scene = render_environment_object_scene(
        rng=random.Random(202),
        object_count=6,
        theme_weights={theme: (1.0 if theme == "park_road" else 0.0) for theme in ENVIRONMENT_THEME_IDS},
    )

    object_entities = [entity for entity in environment_scene_entities(scene) if entity["entity_type"] == "illustration_object"]
    assert object_entities
    for entity in object_entities:
        _assert_vector_object_record(entity)
        assert entity["object_record"]["semantic_attributes"]["zone_id"] == entity["zone_id"]
        assert "relations" in entity["object_record"]["semantic_attributes"]
    local_entities = [
        entity
        for entity in environment_scene_entities(scene)
        if entity["entity_type"] in {"environment_feature", "environment_building"}
    ]
    assert local_entities
    for entity in local_entities:
        _assert_vector_object_record(entity)

    serialized_objects, object_bboxes, part_bboxes = serialize_environment_objects(scene)
    assert serialized_objects
    assert object_bboxes
    assert part_bboxes
    assert all(obj["object_record"]["visual_attributes"]["renderer_style"] == "vector" for obj in serialized_objects)


def test_indoor_room_scene_uses_shared_vector_records_for_placed_objects() -> None:
    scene = render_indoor_room_scene(
        rng=random.Random(303),
        object_specs=(
            IndoorObjectSpec("mug", "surface", "table", "target"),
            IndoorObjectSpec("book", "container", "basket", "distractor"),
        ),
        theme_weights={"living_room": 1.0},
        render_scale=1,
    )

    object_entities = [entity for entity in indoor_scene_entities(scene) if entity["entity_type"] == "illustration_object"]
    assert len(object_entities) == 2
    for entity in object_entities:
        _assert_vector_object_record(entity)
        record_semantic = entity["object_record"]["semantic_attributes"]
        assert record_semantic["placement_kind"] == entity["placement_kind"]
        assert record_semantic["relations"]
    fixture_entities = [
        entity
        for entity in indoor_scene_entities(scene)
        if entity["entity_type"] in {"indoor_furniture", "indoor_surface", "indoor_container"}
    ]
    assert fixture_entities
    for entity in fixture_entities:
        _assert_vector_object_record(entity)
    assert all(
        entity["object_record"]["visual_attributes"]["renderer_id"] == entity["entity_type"]
        for entity in fixture_entities
    )


def test_library_scene_uses_shared_vector_records_for_drawn_decor() -> None:
    scene = render_library_scene(
        rng=random.Random(404),
        section_specs=(
            LibrarySectionSpec(
                "history",
                (
                    LibraryBookSpec("history", "red", "upright"),
                    LibraryBookSpec("history", "blue", "horizontal"),
                ),
            ),
        ),
        render_scale=1,
    )

    decor_entities = [
        entity
        for entity in library_scene_entities(scene)
        if entity["entity_type"] == "library_decor"
        and entity["object_record"]["visual_attributes"].get("renderer_style") == "vector"
    ]
    section_and_book_entities = [
        entity
        for entity in library_scene_entities(scene)
        if entity["entity_type"] in {"library_section", "library_book"}
    ]
    assert section_and_book_entities
    for entity in section_and_book_entities:
        _assert_vector_object_record(entity)
    assert decor_entities
    for entity in decor_entities:
        _assert_vector_object_record(entity)
        assert entity["object_record"]["semantic_attributes"]["decor_type"] == entity["decor_type"]


def test_park_playground_scene_uses_shared_vector_records_for_people_and_decor() -> None:
    scene = render_park_playground_scene(
        rng=random.Random(505),
        person_specs=(
            ParkPersonSpec("walking", "target"),
            ParkPersonSpec("playing_ball", "distractor"),
        ),
        equipment_specs=(ParkEquipmentSpec(PARK_EQUIPMENT_TYPES[0], "target"),),
        render_scale=1,
    )

    entities = park_scene_entities(scene)
    assert entities
    for entity in entities:
        _assert_vector_object_record(entity)
    person_entities = [entity for entity in entities if entity["entity_type"] == "park_person"]
    assert person_entities
    assert {entity["object_record"]["semantic_attributes"]["activity"] for entity in person_entities} >= {"walking"}
    assert all(entity["object_record"]["visual_attributes"]["renderer_id"] == "park_person" for entity in person_entities)
    assert all(entity["object_record"]["parts"] for entity in person_entities)
    equipment_entities = [
        entity
        for entity in entities
        if entity["object_record"]["object_type"] == "playground_equipment"
    ]
    assert equipment_entities
    assert all("equipment_type" in entity["object_record"]["semantic_attributes"] for entity in equipment_entities)
    assert all(entity["object_record"]["visual_attributes"]["renderer_id"] == "park_equipment" for entity in equipment_entities)


def test_construction_site_scene_uses_shared_vector_records_for_local_entities() -> None:
    scene = render_construction_site_scene(
        rng=random.Random(707),
        worker_specs=(ConstructionWorkerSpec("yellow", "orange", "hammer", "target"),),
        material_specs=(ConstructionMaterialSpec(CONSTRUCTION_MATERIAL_TYPES[0], "distractor"),),
        equipment_specs=(ConstructionEquipmentSpec(CONSTRUCTION_EQUIPMENT_TYPES[0], CONSTRUCTION_ZONE_TYPES[0], "target"),),
        canvas_width=1280,
        canvas_height=900,
        render_scale=1,
        setting_weights={setting: 1.0 for setting in CONSTRUCTION_SETTING_IDS},
        style_weights={"flat_vector": 1.0},
        instance_seed=707,
    )

    entities = construction_scene_entities(scene)
    assert entities
    for entity in entities:
        _assert_vector_object_record(entity)
    worker_entities = [entity for entity in entities if entity["type"] == "construction_worker"]
    assert worker_entities
    assert all(entity["object_record"]["visual_attributes"]["renderer_id"] == "construction_worker" for entity in worker_entities)
    assert all(entity["object_record"]["parts"] for entity in worker_entities)
    material_entities = [entity for entity in entities if entity["type"] == "construction_material"]
    equipment_entities = [entity for entity in entities if entity["type"] == "construction_equipment"]
    assert material_entities
    assert equipment_entities
    assert all(entity["object_record"]["visual_attributes"]["renderer_id"] == "construction_material" for entity in material_entities)
    assert all(entity["object_record"]["visual_attributes"]["renderer_id"] == "construction_equipment" for entity in equipment_entities)
    assert any(entity["type"] == "construction_zone" for entity in entities)
