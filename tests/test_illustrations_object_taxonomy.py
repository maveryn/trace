"""Tests for shared illustration object taxonomy records."""

from __future__ import annotations

from PIL import Image

from trace_tasks.tasks.illustrations.construction_site.shared.output import construction_scene_entities
from trace_tasks.tasks.illustrations.construction_site.shared.state import (
    ConstructionDecor,
    ConstructionEquipment,
    ConstructionMaterial,
    ConstructionWorker,
    ConstructionZone,
    RenderedConstructionSiteScene,
)
from trace_tasks.tasks.illustrations.library.shared.annotations import library_scene_entities
from trace_tasks.tasks.illustrations.library.shared.state import LibraryDecor, RenderedLibraryScene
from trace_tasks.tasks.illustrations.shared.object_library import IllustrationObject, IllustrationPart, serialize_object
from trace_tasks.tasks.illustrations.shared.object_registry import object_type_definition, registered_object_types
from trace_tasks.tasks.illustrations.park_playground.shared.annotations import park_scene_entities
from trace_tasks.tasks.illustrations.park_playground.shared.rendering import (
    ParkDecor,
    ParkPerson,
    RenderedParkPlaygroundScene,
)
from trace_tasks.tasks.illustrations.shared.scene_objects import extract_scene_object_records, normalized_object_record


def test_registry_contains_shared_and_scene_specific_object_types() -> None:
    registered = set(registered_object_types())
    assert {"duck", "person", "worker", "playground_equipment", "boarding_area"} <= registered
    assert object_type_definition("worker").family == "person"


def test_serialize_object_embeds_normalized_object_record() -> None:
    rendered = IllustrationObject(
        object_id="obj_0",
        object_type="duck",
        family="animal",
        bbox_xyxy=(10.0, 20.0, 50.0, 70.0),
        primary_color_rgb=(100, 120, 140),
        accent_color_rgb=(230, 210, 80),
        style_id="flat_vector",
        parts=(
            IllustrationPart(
                part_id="obj_0_wing_0",
                part_kind="wing",
                bbox_xyxy=(20.0, 30.0, 35.0, 45.0),
                attributes={"visible": True},
            ),
        ),
        attributes={"role": "target"},
    )
    payload = serialize_object(rendered)
    record = payload["object_record"]
    assert record["object_type"] == "duck"
    assert record["public_name"] == "duck"
    assert record["visual_attributes"]["style_id"] == "flat_vector"
    assert record["semantic_attributes"]["role"] == "target"
    assert record["parts"][0]["part_kind"] == "wing"


def test_scene_object_extractor_handles_source_entity_shape() -> None:
    record = normalized_object_record(
        {
            "id": "worker_0",
            "type": "construction_worker",
            "bbox_xyxy": [1, 2, 3, 4],
            "hard_hat_color": "yellow",
            "vest_color": "orange",
            "role": "target",
        }
    )
    assert record is not None
    assert record["object_type"] == "worker"
    assert record["source_entity_type"] == "construction_worker"
    assert record["semantic_attributes"]["hard_hat_color"] == "yellow"


def test_scene_entities_embed_normalized_object_records() -> None:
    image = Image.new("RGB", (16, 16), (255, 255, 255))
    park = RenderedParkPlaygroundScene(
        image=image,
        setting_id="playground_lawn",
        persons=(
            ParkPerson(
                person_id="person_00",
                activity="walking",
                activity_label="walking",
                bbox_xyxy=(10, 20, 30, 60),
                primary_color_rgb=(1, 2, 3),
                accent_color_rgb=(4, 5, 6),
                skin_color_rgb=(7, 8, 9),
                style_id="flat_vector",
                gender_id="male",
                role="target",
                attributes={"zone": "garden"},
            ),
        ),
        decor=(ParkDecor("equipment_00", "slide", (40, 20, 70, 65), {"role": "target"}),),
        canvas_width=100,
        canvas_height=80,
        render_scale=1,
        style_id="flat_vector",
        layout={},
    )
    construction = RenderedConstructionSiteScene(
        image=image,
        setting_id="roadwork",
        zones=(ConstructionZone("zone_0", "Roadwork Zone", (0, 0, 80, 40), (1, 2, 3), (4, 5, 6)),),
        workers=(
            ConstructionWorker(
                worker_id="worker_0",
                hard_hat_color="yellow",
                vest_color="orange",
                tool_type="hammer",
                bbox_xyxy=(5, 5, 20, 30),
                style_id="flat_vector",
                gender_id="female",
                role="target",
                attributes={},
            ),
        ),
        materials=(
            ConstructionMaterial(
                material_id="material_0",
                material_type="brick_stack",
                material_label="brick stacks",
                bbox_xyxy=(30, 5, 50, 25),
                style_id="flat_vector",
                role="distractor",
                attributes={},
            ),
        ),
        equipment=(
            ConstructionEquipment(
                equipment_id="equipment_0",
                equipment_type="excavator",
                equipment_label="excavators",
                zone_id="zone_0",
                bbox_xyxy=(55, 5, 90, 35),
                style_id="flat_vector",
                role="distractor",
                attributes={},
            ),
        ),
        decor=(ConstructionDecor("decor_0", "scaffold", (0, 40, 20, 70), {}),),
        canvas_width=100,
        canvas_height=80,
        render_scale=1,
        style_id="flat_vector",
        layout={},
    )
    library = RenderedLibraryScene(
        image=image,
        setting_id="reading_room",
        sections=(),
        books=(),
        decor=(LibraryDecor("library_person_0", "person", (5, 40, 20, 75), {"gender_id": "female", "role": "distractor"}),),
        canvas_width=100,
        canvas_height=80,
        render_scale=1,
        style_id="flat_vector",
        layout={},
    )

    records = extract_scene_object_records(
        [
            *park_scene_entities(park),
            *construction_scene_entities(construction),
            *library_scene_entities(library),
        ]
    )
    by_id = {record["object_id"]: record for record in records}
    assert by_id["person_00"]["semantic_attributes"]["activity"] == "walking"
    assert by_id["equipment_00"]["object_type"] == "playground_equipment"
    assert by_id["worker_0"]["object_type"] == "worker"
    assert by_id["person_00"]["visual_attributes"]["gender_id"] == "male"
    assert by_id["worker_0"]["visual_attributes"]["gender_id"] == "female"
    assert by_id["library_person_0"]["visual_attributes"]["gender_id"] == "female"
    assert "gender_id" not in by_id["person_00"]["semantic_attributes"]
    assert "gender_id" not in by_id["worker_0"]["semantic_attributes"]
    assert "gender_id" not in by_id["library_person_0"]["semantic_attributes"]
