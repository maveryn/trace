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

from trace_tasks.tasks.illustrations.shared.pixel_farm_rendering import render_pixel_farm_map
from trace_tasks.tasks.illustrations.shared.pixel_world_objects import PIXEL_DOMESTIC_ANIMALS


def test_pixel_farm_renderer_is_deterministic_and_semantic() -> None:
    first = render_pixel_farm_map(20260604, width=960, height=720)
    second = render_pixel_farm_map(20260604, width=960, height=720)

    assert first.trace == second.trace
    assert first.image.size == (960, 720)
    assert first.trace["renderer_id"] == "pixel_farm_map_v0"
    assert first.trace["uses_external_sprites"] is False
    assert 24 <= int(first.trace["grid_cols"]) <= 30
    assert 17 <= int(first.trace["grid_rows"]) <= 22
    assert first.trace["tile_px"] == 32
    assert first.trace["canonical_tile_px"] == 16
    assert first.trace["map_size_px"] == [
        int(first.trace["grid_cols"]) * 32,
        int(first.trace["grid_rows"]) * 32,
    ]
    assert len(first.regions) == 2
    assert {region.region_type for region in first.regions} == {"fenced_pen"}
    assert int(first.trace["entity_count"]) >= 18
    assert int(first.trace["animal_count"]) >= 9
    assert int(first.trace["inside_pen_animal_count"]) > 0
    assert int(first.trace["outside_pen_animal_count"]) > 0
    assert set(PIXEL_DOMESTIC_ANIMALS).issubset(set(first.trace["public_name_counts"]))

    categories = {entity.category for entity in first.entities}
    assert {"animal", "building", "farm_fixture", "plant"}.issubset(categories)
    animals = [entity for entity in first.entities if entity.category == "animal"]
    assert animals
    assert {entity.public_name for entity in animals}.issubset(set(PIXEL_DOMESTIC_ANIMALS))
    assert any(bool(entity.metadata["inside_pen"]) for entity in animals)
    assert any(not bool(entity.metadata["inside_pen"]) for entity in animals)

    for entity in first.entities:
        x0, y0, x1, y1 = entity.bbox_xyxy
        assert 0 <= x0 < x1 <= 960
        assert 0 <= y0 < y1 <= 720
        assert entity.public_name
        assert entity.category
        assert entity.metadata.get("variant")
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
        if entity.category == "building":
            assert entity.metadata.get("building_door_state") in {"open", "closed"}
        if entity.public_name == "tree":
            assert entity.metadata.get("tree_style") == "fruit_tree"
            assert entity.metadata.get("object_variant_id") == "fruit_tree"
            assert entity.metadata.get("renderer_style") == "top_down_pixel_rpg"
            assert entity.metadata.get("object_record")["object_type"] == "tree"
        if entity.public_name == "flower":
            assert entity.metadata.get("object_record")["object_type"] == "flower"
        if entity.category == "animal" or entity.public_name in {"tree", "flower"}:
            payload = entity.as_dict()
            assert "object_record" in payload
            assert "object_record" not in payload["metadata"]

    for region in first.regions:
        x0, y0, x1, y1 = region.bbox_xyxy
        assert 0 <= x0 < x1 <= 960
        assert 0 <= y0 < y1 <= 720
        assert region.public_name
        assert region.region_id in {"pen_a", "pen_b"}


def test_pixel_farm_animal_species_cover_reusable_variants() -> None:
    species: set[str] = set()
    for seed in range(20260604, 20260612):
        scene = render_pixel_farm_map(seed, width=960, height=720)
        species.update(entity.public_name for entity in scene.entities if entity.category == "animal")

    assert species == set(PIXEL_DOMESTIC_ANIMALS)


def test_pixel_farm_buildings_cover_door_states() -> None:
    door_states: set[str] = set()
    for seed in range(20260604, 20260620):
        scene = render_pixel_farm_map(seed, width=960, height=720)
        door_states.update(
            str(entity.metadata.get("building_door_state"))
            for entity in scene.entities
            if entity.category == "building"
        )

    assert door_states == {"open", "closed"}


def test_pixel_farm_renderer_supports_explicit_grid_and_tile_size() -> None:
    scene = render_pixel_farm_map(123, width=960, height=720, tile_px=32, grid_cols=27, grid_rows=19)

    assert scene.trace["grid_cols"] == 27
    assert scene.trace["grid_rows"] == 19
    assert scene.trace["tile_px"] == 32
    assert scene.trace["map_size_px"] == [864, 608]
    assert scene.image.size == (960, 720)
    assert scene.trace["inside_pen_animal_count"] > 0
    assert scene.trace["outside_pen_animal_count"] > 0
