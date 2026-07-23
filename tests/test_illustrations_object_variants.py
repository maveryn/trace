"""Tests for shared illustration object variant profiles."""

from __future__ import annotations

import random

from PIL import Image, ImageDraw

from trace_tasks.tasks.illustrations.shared.object_rendering import (
    IllustrationObjectSpec,
    RenderContext,
    render_illustration_object,
    serialize_rendered_illustration_object,
)
from trace_tasks.tasks.illustrations.shared.object_variants import (
    ILLUSTRATION_RENDERER_STYLES,
    PERSON_VARIANT_IDS,
    RENDERER_STYLE_ISOMETRIC_PIXEL_RPG,
    RENDERER_STYLE_TOP_DOWN_PIXEL_RPG,
    RENDERER_STYLE_VECTOR,
    TREE_VARIANT_IDS,
    object_variant_ids,
    renderer_supported_variant_ids,
    sample_object_variant_id,
    variant_visual_metadata,
)
from trace_tasks.tasks.illustrations.shared.object_library import draw_illustration_object, serialize_object


def test_tree_and_person_variants_have_profiles_for_all_current_renderers() -> None:
    assert object_variant_ids("tree") == TREE_VARIANT_IDS
    assert object_variant_ids("person") == PERSON_VARIANT_IDS
    assert PERSON_VARIANT_IDS == ("adult", "farmer", "worker", "vendor", "soldier")
    assert "child" not in PERSON_VARIANT_IDS

    for renderer_style in ILLUSTRATION_RENDERER_STYLES:
        assert renderer_supported_variant_ids("tree", renderer_style) == TREE_VARIANT_IDS
        assert renderer_supported_variant_ids("person", renderer_style) == PERSON_VARIANT_IDS

    tree_metadata = variant_visual_metadata("tree", "pine", "isometric_pixel_rpg")
    assert tree_metadata["object_variant_id"] == "pine"
    assert tree_metadata["tree_style"] == "pine"
    assert tree_metadata["renderer_style"] == "isometric_pixel_rpg"

    person_metadata = variant_visual_metadata("person", "farmer", "top_down_pixel_rpg")
    assert person_metadata["object_variant_id"] == "farmer"
    assert person_metadata["person_variant_id"] == "farmer"
    assert person_metadata["renderer_style"] == "top_down_pixel_rpg"

    soldier_metadata = variant_visual_metadata("person", "soldier", "vector")
    assert soldier_metadata["object_variant_id"] == "soldier"
    assert soldier_metadata["person_variant_id"] == "soldier"
    assert soldier_metadata["renderer_variant_id"] == "soldier"


def test_object_variant_sampling_uses_explicit_support() -> None:
    rng = random.Random(123)
    sampled = {sample_object_variant_id(rng, "person", support=("farmer", "worker")) for _ in range(20)}

    assert sampled == {"farmer", "worker"}


def test_vector_object_variant_metadata_stays_visual() -> None:
    image = Image.new("RGB", (120, 120), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    rendered = draw_illustration_object(
        draw,
        object_id="person_0",
        object_type="person",
        bbox_xyxy=(30, 20, 70, 95),
        primary_color_rgb=(72, 115, 166),
        accent_color_rgb=(245, 190, 88),
        style_id="flat_vector",
        render_scale=1,
        gender_id="female",
        object_variant_id="farmer",
    )

    record = serialize_object(rendered)["object_record"]
    assert record["visual_attributes"]["object_variant_id"] == "farmer"
    assert record["visual_attributes"]["person_variant_id"] == "farmer"
    assert record["visual_attributes"]["renderer_style"] == "vector"
    assert "object_variant_id" not in record["semantic_attributes"]
    assert "person_variant_id" not in record["semantic_attributes"]


def test_shared_vector_renderer_dispatch_records_variant_metadata() -> None:
    image = Image.new("RGB", (120, 120), (255, 255, 255))
    rendered = render_illustration_object(
        IllustrationObjectSpec(
            object_id="person_0",
            object_type="person",
            bbox_xyxy=(30, 20, 70, 95),
            variant_id="worker",
            semantic_attributes={"zone": "garden"},
            visual_attributes={"gender_id": "male"},
            role="target",
            source_entity_type="unit_test_object",
        ),
        RenderContext(renderer_style=RENDERER_STYLE_VECTOR, draw=ImageDraw.Draw(image)),
    )

    record = rendered.object_record
    assert record["object_type"] == "person"
    assert record["visual_attributes"]["object_variant_id"] == "worker"
    assert record["visual_attributes"]["person_variant_id"] == "worker"
    assert record["visual_attributes"]["renderer_style"] == "vector"
    assert "object_variant_id" not in record["semantic_attributes"]
    assert record["semantic_attributes"]["zone"] == "garden"
    assert record["role"] == "target"
    assert record["source_entity_type"] == "unit_test_object"
    serialized = serialize_rendered_illustration_object(rendered)
    assert serialized["object_id"] == "person_0"
    assert serialized["object_record"]["visual_attributes"]["renderer_style"] == "vector"
    assert serialized["attributes"]["zone"] == "garden"
    assert serialized["parts"]


def test_shared_top_down_pixel_renderer_dispatches_farm_objects() -> None:
    image = Image.new("RGBA", (80, 48), (255, 255, 255, 255))
    rendered = render_illustration_object(
        IllustrationObjectSpec(
            object_id="animal_0",
            object_type="domestic_animal",
            public_name="cow",
            bbox_xyxy=(0, 0, 64, 32),
            tile_xywh=(1, 1, 2, 1),
            semantic_attributes={"animal_type": "cow", "region_id": "pen_a", "inside_pen": True},
            visual_attributes={"animal_type": "cow", "facing": "right"},
        ),
        RenderContext(renderer_style=RENDERER_STYLE_TOP_DOWN_PIXEL_RPG, draw=ImageDraw.Draw(image, "RGBA")),
    )

    assert rendered.object_record["object_type"] == "domestic_animal"
    assert rendered.object_record["semantic_attributes"]["animal_type"] == "cow"
    assert rendered.object_record["semantic_attributes"]["inside_pen"] is True
    assert rendered.object_record["visual_attributes"]["renderer_style"] == "top_down_pixel_rpg"
    assert any(pixel != (255, 255, 255, 255) for pixel in image.getdata())


def test_shared_isometric_pixel_renderer_dispatches_projected_sprites() -> None:
    image = Image.new("RGBA", (96, 96), (255, 255, 255, 0))

    rendered = render_illustration_object(
        IllustrationObjectSpec(
            object_id="tree_0",
            object_type="tree",
            public_name="tree",
            bbox_xyxy=(30, 20, 46, 52),
            tile_xywh=(2, 3, 1, 1),
            level=2,
            variant_id="pine",
            visual_attributes={"tree_style": "pine", "leaf_rgb": [29, 118, 82]},
        ),
        RenderContext(
            renderer_style=RENDERER_STYLE_ISOMETRIC_PIXEL_RPG,
            image=image,
            project_tile_center=lambda tile_xywh, level: (48.0, 60.0 - 12.0 * level),
        ),
    )

    assert rendered.object_record["object_type"] == "tree"
    assert rendered.object_record["visual_attributes"]["object_variant_id"] == "pine"
    assert rendered.object_record["visual_attributes"]["renderer_style"] == "isometric_pixel_rpg"
    assert any(pixel[3] for pixel in image.getdata())
