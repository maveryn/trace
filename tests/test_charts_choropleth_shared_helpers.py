import random

from trace_tasks.tasks.charts.region_map.shared import assets
from trace_tasks.tasks.charts.region_map.shared import projection
from trace_tasks.tasks.charts.region_map.shared import spatial_primitives
from trace_tasks.tasks.charts.region_map.shared import styles


def test_region_map_asset_loader_aliases_cover_world_map() -> None:
    assert assets.normalize_geographic_map_variant(assets.WORLD_MAP_ASSET_ID) == "world_countries"
    assert assets.load_world_map_asset()["asset_id"] == "natural_earth_admin0_world_110m_v0"


def test_region_map_spatial_primitives_cover_connected_grid_cells() -> None:
    points = spatial_primitives._grid_points(rows=2, cols=3, map_bbox=(0, 0, 300, 200), instance_seed=17)
    assert sorted(points) == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
        (3, 0),
        (3, 1),
        (3, 2),
    ]
    polygon = spatial_primitives._region_polygon(row=0, col=1, grid_points=points)
    assert spatial_primitives._polygon_bbox(polygon)[2] > spatial_primitives._polygon_bbox(polygon)[0]

    connected = spatial_primitives._sample_connected_cells(
        rows=4,
        cols=4,
        target_count=5,
        rng=random.Random(23),
    )
    connected_set = set(connected)
    assert len(connected_set) == 5
    assert all(
        any(neighbor in connected_set for neighbor in spatial_primitives._neighbors(cell, rows=4, cols=4))
        for cell in connected_set
    )


def test_region_map_projection_helpers_cover_world_and_synthetic_adjacency() -> None:
    centroid = projection._centroid_lonlat_from_rings([[[0, 0], [2, 0], [2, 2], [0, 2]]])
    assert centroid == [1.0, 1.0]

    filtered = projection._world_filtered_region_candidates(
        [
            {"region_id": "a", "continent": "Africa"},
            {"region_id": "b", "continent": "Oceania"},
        ]
    )
    assert [region["region_id"] for region in filtered] == ["a"]

    adjacency = projection._synthetic_region_adjacency(
        {
            "r0": {"row": 0, "col": 0},
            "r1": {"row": 0, "col": 1},
            "r2": {"row": 2, "col": 2},
        }
    )
    assert adjacency["r0"] == ["r1"]
    assert adjacency["r2"] == []


def test_region_map_style_resolvers_cover_explicit_paths() -> None:
    palette_id, probabilities, palette = styles.resolve_choropleth_palette(
        {"map_palette_rgb": [(1, 2, 3), (4, 5, 6)]},
        render_defaults={},
        namespace="test_region_map",
        style_seed=7,
        required_palette_count=4,
        categorical=False,
    )
    assert palette_id == "custom"
    assert probabilities == {"custom": 1.0}
    assert len(palette) == 4

    style_id, style_probabilities, style = styles.resolve_choropleth_world_map_style(
        {"world_map_style": "warm_print"},
        render_defaults={},
        namespace="test_region_map",
        instance_seed=11,
    )
    assert style_id == "warm_print"
    assert style_probabilities["warm_print"] == 1.0
    assert "ocean_rgb" in style
