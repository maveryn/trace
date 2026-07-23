from __future__ import annotations

from trace_tasks.tasks.registry import create_task
from trace_tasks.tasks.illustrations.rpg_tactical_map.counterfactual_terrain_conversion_cost_value import TASK_ID as COUNTERFACTUAL_COST_TASK_ID
from trace_tasks.tasks.illustrations.rpg_tactical_map.movement_cost_value import TASK_ID as COST_VALUE_TASK_ID
from trace_tasks.tasks.illustrations.rpg_tactical_map.movement_reachable_tile_count import TASK_ID as COUNT_TASK_ID
from trace_tasks.tasks.illustrations.rpg_tactical_map.movement_reachable_tile_label import TASK_ID as LABEL_TASK_ID
from trace_tasks.tasks.illustrations.rpg_tactical_map.movement_sequence_endpoint_label import TASK_ID as SEQUENCE_TASK_ID
from trace_tasks.tasks.illustrations.rpg_tactical_map.terrain_type_tile_count import TASK_ID as TERRAIN_COUNT_TASK_ID
from trace_tasks.tasks.illustrations.rpg_tactical_map.water_barrier_unreachable_tile_label import TASK_ID as WATER_BARRIER_TASK_ID
from trace_tasks.tasks.illustrations.rpg_tactical_map.shared.relations import (
    TERRAIN_GRASS,
    TERRAIN_MOUNTAIN,
    TERRAIN_WATER,
    orthogonal_neighbors,
    shortest_movement_costs,
)
from trace_tasks.tasks.illustrations.rpg_tactical_map.shared.rendering import (
    DEFAULT_TILE_PX,
    render_rpg_tactical_map_scene,
    resolve_tactical_map_render_params,
)
from trace_tasks.tasks.illustrations.rpg_tactical_map.shared.state import RpgTacticalTile


def _assert_bbox_inside_canvas(bbox: list[float], *, width: int, height: int) -> None:
    assert 0 <= float(bbox[0]) < float(bbox[2]) <= float(width)
    assert 0 <= float(bbox[1]) < float(bbox[3]) <= float(height)


def _tile(tile_id: str, row: int, col: int, terrain: str, cost: int | None, passable: bool) -> RpgTacticalTile:
    return RpgTacticalTile(
        tile_id=tile_id,
        row=row,
        col=col,
        terrain=terrain,
        movement_cost=cost,
        passable=passable,
        bbox_xyxy=(float(col * 10), float(row * 10), float(col * 10 + 10), float(row * 10 + 10)),
        point_xy=(float(col * 10 + 5), float(row * 10 + 5)),
        metadata={},
    )


def test_rpg_tactical_map_pathfinding_uses_mountain_cost_three() -> None:
    tiles = {
        (0, 0): _tile("start", 0, 0, TERRAIN_GRASS, 1, True),
        (0, 1): _tile("mountain", 0, 1, TERRAIN_MOUNTAIN, 3, True),
        (0, 2): _tile("after", 0, 2, TERRAIN_GRASS, 1, True),
        (1, 0): _tile("water", 1, 0, TERRAIN_WATER, None, False),
    }
    costs = shortest_movement_costs(tiles, start_coord=(0, 0))
    assert costs["start"] == 0
    assert costs["mountain"] == 3
    assert costs["after"] == 4
    assert "water" not in costs


def test_rpg_tactical_map_renderer_is_deterministic_and_profile_safe() -> None:
    for profile, expected_size in (
        ("landscape", (960, 640)),
        ("square", (800, 800)),
        ("portrait", (640, 960)),
    ):
        params = resolve_tactical_map_render_params({"canvas_profile": profile}, {}, instance_seed=101)
        first = render_rpg_tactical_map_scene(
            12345,
            width=params["canvas_width"],
            height=params["canvas_height"],
            grid_cols=params["grid_cols"],
            grid_rows=params["grid_rows"],
            tile_px=params["tile_px"],
            candidate_tile_ids_by_label={"A": "r00_c00", "B": "r00_c01", "C": "r01_c00", "D": "r01_c01"},
            render_metadata=params,
        )
        second = render_rpg_tactical_map_scene(
            12345,
            width=params["canvas_width"],
            height=params["canvas_height"],
            grid_cols=params["grid_cols"],
            grid_rows=params["grid_rows"],
            tile_px=params["tile_px"],
            candidate_tile_ids_by_label={"A": "r00_c00", "B": "r00_c01", "C": "r01_c00", "D": "r01_c01"},
            render_metadata=params,
        )
        assert first.image.size == expected_size
        assert list(first.image.getdata()) == list(second.image.getdata())
        assert first.trace["tile_px"] == DEFAULT_TILE_PX
        assert first.trace["canvas_profile"] == profile
        assert first.trace["terrain_movement_costs"]["mountain"] == 3
        assert first.trace["blocked_terrain"] == ["water"]
        assert len(first.tiles) == int(params["grid_cols"]) * int(params["grid_rows"])
        assert len(first.units) == 1
        assert sorted(first.label_bboxes_by_tile_id) == ["r00_c00", "r00_c01", "r01_c00", "r01_c01"]
        width, height = first.image.size
        for tile in first.tiles:
            _assert_bbox_inside_canvas(list(tile.bbox_xyxy), width=width, height=height)
        for unit in first.units:
            _assert_bbox_inside_canvas(list(unit.bbox_xyxy), width=width, height=height)


def test_rpg_tactical_map_generated_rivers_use_shared_styles() -> None:
    params = resolve_tactical_map_render_params({"canvas_profile": "square"}, {}, instance_seed=103)
    styles_by_seed: dict[str, list[int]] = {}
    for seed in range(2026062800, 2026062825):
        scene = render_rpg_tactical_map_scene(
            seed,
            width=params["canvas_width"],
            height=params["canvas_height"],
            grid_cols=params["grid_cols"],
            grid_rows=params["grid_rows"],
            tile_px=params["tile_px"],
            render_metadata=params,
        )
        water_feature = scene.trace["terrain_generation"]["water_feature"]
        if water_feature["kind"] != "river":
            continue
        styles_by_seed.setdefault(str(water_feature["style"]), []).append(seed)
        assert water_feature["orientation"] in {"horizontal", "vertical"}
        assert int(water_feature["water_tile_count"]) > 0

    assert styles_by_seed["straight"]
    assert styles_by_seed["zigzag"]


def test_rpg_tactical_map_target_marker_is_red_square_on_tile_border() -> None:
    params = resolve_tactical_map_render_params({"canvas_profile": "landscape"}, {}, instance_seed=102)
    seed = 67890
    probe = render_rpg_tactical_map_scene(
        seed,
        width=params["canvas_width"],
        height=params["canvas_height"],
        grid_cols=params["grid_cols"],
        grid_rows=params["grid_rows"],
        tile_px=params["tile_px"],
        render_metadata=params,
    )
    player_tile_id = str(probe.units[0].tile_id)
    target_tile = next(tile for tile in probe.tiles if str(tile.tile_id) != player_tile_id)
    baseline = render_rpg_tactical_map_scene(
        seed,
        width=params["canvas_width"],
        height=params["canvas_height"],
        grid_cols=params["grid_cols"],
        grid_rows=params["grid_rows"],
        tile_px=params["tile_px"],
        player_tile_id=player_tile_id,
        render_metadata=params,
    )
    marked = render_rpg_tactical_map_scene(
        seed,
        width=params["canvas_width"],
        height=params["canvas_height"],
        grid_cols=params["grid_cols"],
        grid_rows=params["grid_rows"],
        tile_px=params["tile_px"],
        player_tile_id=player_tile_id,
        target_tile_ids=[str(target_tile.tile_id)],
        render_metadata=params,
    )

    x0, y0, x1, y1 = [int(round(value)) for value in target_tile.bbox_xyxy]
    cx, cy = [int(round(value)) for value in target_tile.point_xy]
    assert marked.image.getpixel((cx, cy)) == baseline.image.getpixel((cx, cy))

    edge_points = [
        (x0, y0),
        (x1 - 1, y0),
        (x0, y1 - 1),
        (x1 - 1, y1 - 1),
        ((x0 + x1) // 2, y0),
        ((x0 + x1) // 2, y1 - 1),
        (x0, (y0 + y1) // 2),
        (x1 - 1, (y0 + y1) // 2),
    ]
    for point in edge_points:
        pixel = marked.image.getpixel(point)
        assert int(pixel[0]) > 180 and int(pixel[1]) < 70 and int(pixel[2]) < 80

    crop_pixels = list(marked.image.crop((x0, y0, x1, y1)).getdata())
    bright_red_pixels = [
        pixel
        for pixel in crop_pixels
        if int(pixel[0]) > 180 and int(pixel[1]) < 70 and int(pixel[2]) < 80
    ]
    assert len(bright_red_pixels) > 80


def test_rpg_tactical_map_movement_reachable_tile_contract() -> None:
    task = create_task(LABEL_TASK_ID)
    out = task.generate(
        2026062401,
        params={
            "canvas_profile": "square",
            "movement_budget": 5,
        },
        max_attempts=30,
    )
    assert out.scene_id == "rpg_tactical_map"
    assert out.query_id == "single"
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value in {"A", "B", "C", "D"}
    assert out.annotation_gt.type == "bbox"
    width, height = out.image.size
    _assert_bbox_inside_canvas(out.annotation_gt.value, width=width, height=height)
    assert "mountains cost 3" in out.prompt
    assert "water cannot be entered" in out.prompt
    assert "up, down, left, or right" in out.prompt

    trace = out.trace_payload
    render_map = trace["render_map"]
    answer_label = str(out.answer_gt.value)
    assert trace["projected_annotation"]["type"] == "bbox"
    assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert render_map["selected_label"] == answer_label
    assert render_map["selected_tile_bbox_px"] == out.annotation_gt.value
    assert render_map["movement_budget"] == 5
    assert trace["execution_trace"]["selected_tile_cost"] <= 5
    assert trace["scene_ir"]["relations"]["terrain_movement_costs"]["mountain"] == 3

    reachable_labels = [
        label
        for label, cost in render_map["candidate_shortest_costs_by_label"].items()
        if cost is not None and int(cost) <= int(render_map["movement_budget"])
    ]
    assert reachable_labels == [answer_label]
    assert sorted(render_map["candidate_tile_ids_by_label"]) == ["A", "B", "C", "D"]
    assert trace["query_spec"]["prompt_variant"]["prompt_bundle_id"] == "illustrations_rpg_tactical_map_v0"
    assert trace["query_spec"]["prompt_variant"]["prompt_scene_id"] == "rpg_tactical_map"


def test_rpg_tactical_map_water_barrier_unreachable_tile_contract() -> None:
    task = create_task(WATER_BARRIER_TASK_ID)
    out = task.generate(
        2026062701,
        params={
            "canvas_profile": "square",
            "barrier_orientation": "vertical",
            "barrier_style": "zigzag",
        },
        max_attempts=40,
    )
    assert out.scene_id == "rpg_tactical_map"
    assert out.query_id == "single"
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value in {"A", "B", "C", "D"}
    assert out.annotation_gt.type == "bbox"
    width, height = out.image.size
    _assert_bbox_inside_canvas(out.annotation_gt.value, width=width, height=height)
    prompt_lower = out.prompt.lower()
    assert "water" in prompt_lower
    assert "cannot be crossed" in prompt_lower
    assert "movement points" not in prompt_lower
    assert "mountains cost" not in prompt_lower

    trace = out.trace_payload
    render_map = trace["render_map"]
    answer_label = str(out.answer_gt.value)
    assert trace["projected_annotation"]["type"] == "bbox"
    assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert render_map["selected_label"] == answer_label
    assert render_map["selected_tile_bbox_px"] == out.annotation_gt.value
    assert render_map["candidate_reachable_by_label"][answer_label] is False
    assert [
        label
        for label, reachable in render_map["candidate_reachable_by_label"].items()
        if not bool(reachable)
    ] == [answer_label]
    assert sorted(render_map["candidate_tile_ids_by_label"]) == ["A", "B", "C", "D"]
    assert trace["query_spec"]["prompt_variant"]["prompt_bundle_id"] == "illustrations_rpg_tactical_map_v0"
    assert trace["query_spec"]["prompt_variant"]["prompt_scene_id"] == "rpg_tactical_map"

    tiles_by_id = {str(tile["tile_id"]): tile for tile in trace["scene_ir"]["tiles"]}
    selected_tile_id = str(render_map["selected_tile_id"])
    assert selected_tile_id not in set(render_map["reachable_tile_ids"])
    for label, tile_id in render_map["candidate_tile_ids_by_label"].items():
        if str(label) == answer_label:
            continue
        assert str(tile_id) in set(render_map["reachable_tile_ids"])

    barrier_ids = [str(tile_id) for tile_id in render_map["water_barrier_tile_ids"]]
    assert barrier_ids
    assert all(tiles_by_id[tile_id]["terrain"] == "water" for tile_id in barrier_ids)
    orientation = str(render_map["barrier_orientation"])
    style = str(render_map["barrier_style"])
    thickness = int(render_map["barrier_thickness"])
    barrier_coords = {
        (int(tiles_by_id[tile_id]["row"]), int(tiles_by_id[tile_id]["col"]))
        for tile_id in barrier_ids
    }
    pending = {next(iter(barrier_coords))}
    seen: set[tuple[int, int]] = set()
    while pending:
        coord = pending.pop()
        if coord in seen:
            continue
        seen.add(coord)
        pending.update(
            neighbor
            for neighbor in orthogonal_neighbors(coord)
            if neighbor in barrier_coords and neighbor not in seen
        )
    assert seen == barrier_coords
    if orientation == "vertical":
        rows = {int(tiles_by_id[tile_id]["row"]) for tile_id in barrier_ids}
        cols = {int(tiles_by_id[tile_id]["col"]) for tile_id in barrier_ids}
        assert rows == set(range(int(trace["render_spec"]["style"]["grid_rows"])))
        assert len(cols) >= thickness
        if style == "zigzag":
            cols_by_row = {
                row: {
                    int(tiles_by_id[tile_id]["col"])
                    for tile_id in barrier_ids
                    if int(tiles_by_id[tile_id]["row"]) == row
                }
                for row in rows
            }
            assert len({min(cols) for cols in cols_by_row.values()}) > 1
    else:
        rows = {int(tiles_by_id[tile_id]["row"]) for tile_id in barrier_ids}
        cols = {int(tiles_by_id[tile_id]["col"]) for tile_id in barrier_ids}
        assert len(rows) >= thickness
        assert cols == set(range(int(trace["render_spec"]["style"]["grid_cols"])))
        if style == "zigzag":
            rows_by_col = {
                col: {
                    int(tiles_by_id[tile_id]["row"])
                    for tile_id in barrier_ids
                    if int(tiles_by_id[tile_id]["col"]) == col
                }
                for col in cols
            }
            assert len({min(rows) for rows in rows_by_col.values()}) > 1


def test_rpg_tactical_map_movement_sequence_endpoint_label_contract() -> None:
    task = create_task(SEQUENCE_TASK_ID)
    out = task.generate(
        2026063001,
        params={
            "canvas_profile": "square",
            "sequence_length": 5,
        },
        max_attempts=40,
    )
    assert out.scene_id == "rpg_tactical_map"
    assert out.query_id == "single"
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value in {"A", "B", "C", "D"}
    assert out.annotation_gt.type == "bbox"
    width, height = out.image.size
    _assert_bbox_inside_canvas(out.annotation_gt.value, width=width, height=height)
    prompt_lower = out.prompt.lower()
    assert "up" in prompt_lower or "down" in prompt_lower or "left" in prompt_lower or "right" in prompt_lower
    assert "movement points" not in prompt_lower
    assert "mountains cost" not in prompt_lower
    assert "water cannot be entered" not in prompt_lower

    trace = out.trace_payload
    render_map = trace["render_map"]
    answer_label = str(out.answer_gt.value)
    assert trace["projected_annotation"]["type"] == "bbox"
    assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert render_map["selected_label"] == answer_label
    assert render_map["selected_tile_bbox_px"] == out.annotation_gt.value
    assert sorted(render_map["candidate_tile_ids_by_label"]) == ["A", "B", "C", "D"]
    assert trace["query_spec"]["prompt_variant"]["prompt_bundle_id"] == "illustrations_rpg_tactical_map_v0"
    assert trace["query_spec"]["prompt_variant"]["prompt_scene_id"] == "rpg_tactical_map"

    tiles_by_id = {str(tile["tile_id"]): tile for tile in trace["scene_ir"]["tiles"]}
    tiles_by_coord = {
        (int(tile["row"]), int(tile["col"])): str(tile["tile_id"])
        for tile in trace["scene_ir"]["tiles"]
    }
    move_sequence = [str(direction) for direction in trace["execution_trace"]["move_sequence"]]
    path_tile_ids = [str(tile_id) for tile_id in trace["execution_trace"]["path_tile_ids"]]
    assert len(move_sequence) == 5
    assert len(path_tile_ids) == len(move_sequence) + 1
    assert path_tile_ids == [str(tile_id) for tile_id in render_map["path_tile_ids"]]
    assert path_tile_ids[0] == str(trace["scene_ir"]["units"][0]["tile_id"])
    assert path_tile_ids[-1] == str(render_map["selected_tile_id"])
    assert render_map["candidate_tile_ids_by_label"][answer_label] == path_tile_ids[-1]

    current_coord = (
        int(tiles_by_id[path_tile_ids[0]]["row"]),
        int(tiles_by_id[path_tile_ids[0]]["col"]),
    )
    direction_delta = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }
    replayed_path = [path_tile_ids[0]]
    for direction in move_sequence:
        drow, dcol = direction_delta[direction]
        current_coord = (current_coord[0] + drow, current_coord[1] + dcol)
        assert current_coord in tiles_by_coord
        replayed_tile_id = tiles_by_coord[current_coord]
        assert bool(tiles_by_id[replayed_tile_id]["passable"])
        replayed_path.append(replayed_tile_id)
    assert replayed_path == path_tile_ids
    for left_id, right_id in zip(path_tile_ids, path_tile_ids[1:]):
        left = tiles_by_id[left_id]
        right = tiles_by_id[right_id]
        step = abs(int(left["row"]) - int(right["row"])) + abs(int(left["col"]) - int(right["col"]))
        assert step == 1


def test_rpg_tactical_map_movement_distractors_are_plausible_and_spread() -> None:
    task = create_task(LABEL_TASK_ID)
    for seed in (2026062401, 2026062402, 2026062403, 23, 93):
        for profile in ("landscape", "square", "portrait"):
            out = task.generate(
                seed,
                params={
                    "canvas_profile": profile,
                    "movement_budget": 5,
                },
                max_attempts=30,
            )
            render_map = out.trace_payload["render_map"]
            tiles_by_id = {str(tile["tile_id"]): tile for tile in out.trace_payload["scene_ir"]["tiles"]}
            answer_label = str(out.answer_gt.value)

            reachable_labels = [
                label
                for label, cost in render_map["candidate_shortest_costs_by_label"].items()
                if cost is not None and int(cost) <= int(render_map["movement_budget"])
            ]
            assert reachable_labels == [answer_label]

            candidate_coords: list[tuple[int, int]] = []
            for label, tile_id in render_map["candidate_tile_ids_by_label"].items():
                tile = tiles_by_id[str(tile_id)]
                candidate_coords.append((int(tile["row"]), int(tile["col"])))
                cost = render_map["candidate_shortest_costs_by_label"][str(label)]
                if str(label) != answer_label and cost is not None:
                    assert int(render_map["movement_budget"]) < int(cost) <= int(render_map["movement_budget"]) + 5

            pairwise_distances = []
            for index, first in enumerate(candidate_coords):
                for second in candidate_coords[index + 1 :]:
                    pairwise_distances.append(abs(first[0] - second[0]) + abs(first[1] - second[1]))
            assert pairwise_distances
            assert min(pairwise_distances) >= 2


def test_rpg_tactical_map_movement_reachable_tile_count_contract() -> None:
    task = create_task(COUNT_TASK_ID)
    out = task.generate(
        2026062400,
        params={
            "canvas_profile": "square",
            "movement_budget": 2,
        },
        max_attempts=30,
    )
    assert out.scene_id == "rpg_tactical_map"
    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert 2 <= int(out.answer_gt.value) <= 8
    assert out.annotation_gt.type == "bbox_set"
    width, height = out.image.size
    for bbox in out.annotation_gt.value:
        _assert_bbox_inside_canvas(bbox, width=width, height=height)
    assert "mountains cost 3" in out.prompt
    assert "water cannot be entered" in out.prompt
    assert "Do not count the tile the unit starts on" in out.prompt or "excluding" in out.prompt

    trace = out.trace_payload
    render_map = trace["render_map"]
    relations = trace["scene_ir"]["relations"]
    start_tile_id = str(relations["start_tile_id"])
    counted_tile_ids = [str(tile_id) for tile_id in render_map["counted_tile_ids"]]
    assert start_tile_id not in set(counted_tile_ids)
    assert int(render_map["answer_count"]) == int(out.answer_gt.value)
    assert len(counted_tile_ids) == int(out.answer_gt.value)
    assert trace["projected_annotation"]["type"] == "bbox_set"
    assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
    assert render_map["counted_tile_bboxes_px"] == out.annotation_gt.value
    assert trace["query_spec"]["prompt_variant"]["prompt_bundle_id"] == "illustrations_rpg_tactical_map_v0"
    assert trace["query_spec"]["prompt_variant"]["prompt_scene_id"] == "rpg_tactical_map"

    costs_by_tile_id = trace["execution_trace"]["movement_costs_by_tile_id"]
    movement_budget = int(render_map["movement_budget"])
    for tile_id in counted_tile_ids:
        assert int(costs_by_tile_id[tile_id]) <= movement_budget
    for tile in trace["scene_ir"]["tiles"]:
        tile_id = str(tile["tile_id"])
        if tile_id == start_tile_id:
            continue
        cost = costs_by_tile_id.get(tile_id)
        if cost is not None and int(cost) <= movement_budget:
            assert tile_id in set(counted_tile_ids)


def test_rpg_tactical_map_terrain_type_tile_count_contract() -> None:
    task = create_task(TERRAIN_COUNT_TASK_ID)
    out = task.generate(
        2026062405,
        params={
            "canvas_profile": "square",
            "target_terrain": "forest",
        },
        max_attempts=30,
    )
    assert out.scene_id == "rpg_tactical_map"
    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert 1 <= int(out.answer_gt.value) <= 18
    assert out.annotation_gt.type == "bbox_set"
    width, height = out.image.size
    for bbox in out.annotation_gt.value:
        _assert_bbox_inside_canvas(bbox, width=width, height=height)
    assert "forest" in out.prompt

    trace = out.trace_payload
    render_map = trace["render_map"]
    target_terrain = str(render_map["target_terrain"])
    counted_tile_ids = [str(tile_id) for tile_id in render_map["counted_tile_ids"]]
    assert target_terrain == "forest"
    assert int(render_map["answer_count"]) == int(out.answer_gt.value)
    assert len(counted_tile_ids) == int(out.answer_gt.value)
    assert trace["projected_annotation"]["type"] == "bbox_set"
    assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
    assert render_map["counted_tile_bboxes_px"] == out.annotation_gt.value
    assert trace["query_spec"]["prompt_variant"]["prompt_bundle_id"] == "illustrations_rpg_tactical_map_v0"
    assert trace["query_spec"]["prompt_variant"]["prompt_scene_id"] == "rpg_tactical_map"

    terrain_by_tile_id = trace["execution_trace"]["terrain_by_tile_id"]
    assert set(counted_tile_ids) == {
        str(tile_id)
        for tile_id, terrain in terrain_by_tile_id.items()
        if str(terrain) == target_terrain
    }


def test_rpg_tactical_map_movement_cost_value_contract() -> None:
    task = create_task(COST_VALUE_TASK_ID)
    out = task.generate(
        2026062601,
        params={
            "canvas_profile": "square",
            "min_movement_cost": 3,
            "max_movement_cost": 10,
        },
        max_attempts=40,
    )
    assert out.scene_id == "rpg_tactical_map"
    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert 3 <= int(out.answer_gt.value) <= 10
    assert out.annotation_gt.type == "bbox_map"
    assert set(out.annotation_gt.value) == {"player_cell", "target_cell"}
    width, height = out.image.size
    for bbox in out.annotation_gt.value.values():
        _assert_bbox_inside_canvas(bbox, width=width, height=height)
    assert "marked" in out.prompt
    assert "movement" in out.prompt
    assert "water cannot be entered" in out.prompt

    trace = out.trace_payload
    render_map = trace["render_map"]
    assert trace["projected_annotation"]["type"] == "bbox_map"
    assert trace["projected_annotation"]["bbox_map"] == out.annotation_gt.value
    assert render_map["target_tile_bbox_px"] == out.annotation_gt.value["target_cell"]
    assert int(render_map["answer_value"]) == int(out.answer_gt.value)
    assert int(render_map["target_shortest_movement_cost"]) == int(out.answer_gt.value)
    assert trace["query_spec"]["prompt_variant"]["prompt_bundle_id"] == "illustrations_rpg_tactical_map_v0"
    assert trace["query_spec"]["prompt_variant"]["prompt_scene_id"] == "rpg_tactical_map"

    tiles_by_id = {str(tile["tile_id"]): tile for tile in trace["scene_ir"]["tiles"]}
    target_tile_id = str(render_map["target_tile_id"])
    start_tile_id = str(render_map["start_tile_id"])
    assert target_tile_id != start_tile_id
    assert target_tile_id in tiles_by_id
    assert start_tile_id in tiles_by_id
    assert out.annotation_gt.value["player_cell"] == tiles_by_id[start_tile_id]["bbox"]
    assert out.annotation_gt.value["target_cell"] == tiles_by_id[target_tile_id]["bbox"]
    assert trace["execution_trace"]["annotation_tile_id_map"] == {
        "player_cell": start_tile_id,
        "target_cell": target_tile_id,
    }
    assert trace["scene_ir"]["relations"]["annotation_tile_id_map"] == {
        "player_cell": start_tile_id,
        "target_cell": target_tile_id,
    }
    path_tile_ids = [str(tile_id) for tile_id in render_map["shortest_path_tile_ids"]]
    assert path_tile_ids[0] == start_tile_id
    assert path_tile_ids[-1] == target_tile_id
    assert trace["execution_trace"]["shortest_path_tile_ids"] == path_tile_ids
    assert len(path_tile_ids) == len(render_map["shortest_path_tile_bboxes_px"])
    for left_id, right_id in zip(path_tile_ids, path_tile_ids[1:]):
        left = tiles_by_id[left_id]
        right = tiles_by_id[right_id]
        step = abs(int(left["row"]) - int(right["row"])) + abs(int(left["col"]) - int(right["col"]))
        assert step == 1
    assert render_map["shortest_path_entry_costs"][0] == 0
    assert sum(int(cost) for cost in render_map["shortest_path_entry_costs"]) == int(out.answer_gt.value)
    assert trace["scene_ir"]["relations"]["target_tile_id"] == target_tile_id
    assert trace["execution_trace"]["target_tile_id"] == target_tile_id
    assert trace["execution_trace"]["movement_costs_by_tile_id"][target_tile_id] == int(out.answer_gt.value)

    target_tile = tiles_by_id[target_tile_id]
    start_tile = tiles_by_id[start_tile_id]
    manhattan = abs(int(target_tile["row"]) - int(start_tile["row"])) + abs(int(target_tile["col"]) - int(start_tile["col"]))
    assert render_map["target_manhattan_distance"] == manhattan
    assert int(out.answer_gt.value) >= manhattan


def test_rpg_tactical_map_counterfactual_terrain_conversion_cost_value_contract() -> None:
    task = create_task(COUNTERFACTUAL_COST_TASK_ID)
    out = task.generate(
        2026062900,
        params={
            "canvas_profile": "square",
            "min_counterfactual_cost": 2,
            "max_counterfactual_cost": 6,
        },
        max_attempts=80,
    )
    assert out.scene_id == "rpg_tactical_map"
    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert 2 <= int(out.answer_gt.value) <= 6
    assert out.annotation_gt.type == "bbox_map"
    assert set(out.annotation_gt.value) == {"player_cell", "target_cell", "changed_cell"}
    width, height = out.image.size
    for bbox in out.annotation_gt.value.values():
        _assert_bbox_inside_canvas(bbox, width=width, height=height)
    prompt_lower = out.prompt.lower()
    assert "marked" in prompt_lower
    assert "changed into a road" in prompt_lower or "terrain-to-road" in prompt_lower
    assert "water cannot be entered" in prompt_lower

    trace = out.trace_payload
    render_map = trace["render_map"]
    assert trace["projected_annotation"]["type"] == "bbox_map"
    assert trace["projected_annotation"]["bbox_map"] == out.annotation_gt.value
    assert int(render_map["answer_value"]) == int(out.answer_gt.value)
    assert int(render_map["counterfactual_shortest_movement_cost"]) == int(out.answer_gt.value)
    assert trace["query_spec"]["prompt_variant"]["prompt_bundle_id"] == "illustrations_rpg_tactical_map_v0"
    assert trace["query_spec"]["prompt_variant"]["prompt_scene_id"] == "rpg_tactical_map"

    tiles_by_id = {str(tile["tile_id"]): tile for tile in trace["scene_ir"]["tiles"]}
    start_tile_id = str(render_map["start_tile_id"])
    target_tile_id = str(render_map["target_tile_id"])
    changed_tile_id = str(render_map["changed_tile_id"])
    assert len({start_tile_id, target_tile_id, changed_tile_id}) == 3
    assert out.annotation_gt.value["player_cell"] == tiles_by_id[start_tile_id]["bbox"]
    assert out.annotation_gt.value["target_cell"] == tiles_by_id[target_tile_id]["bbox"]
    assert out.annotation_gt.value["changed_cell"] == tiles_by_id[changed_tile_id]["bbox"]
    assert trace["execution_trace"]["annotation_tile_id_map"] == {
        "player_cell": start_tile_id,
        "target_cell": target_tile_id,
        "changed_cell": changed_tile_id,
    }
    assert trace["scene_ir"]["relations"]["unique_best_changed_tile"] is True
    assert render_map["changed_tile_original_terrain"] in {"water", "mountain", "forest"}
    assert render_map["changed_tile_counterfactual_terrain"] == "road"

    path_tile_ids = [str(tile_id) for tile_id in render_map["shortest_path_tile_ids"]]
    assert path_tile_ids[0] == start_tile_id
    assert path_tile_ids[-1] == target_tile_id
    assert changed_tile_id in set(path_tile_ids)
    assert len(path_tile_ids) == len(render_map["shortest_path_entry_costs_after_conversion"])
    for left_id, right_id in zip(path_tile_ids, path_tile_ids[1:]):
        left = tiles_by_id[left_id]
        right = tiles_by_id[right_id]
        step = abs(int(left["row"]) - int(right["row"])) + abs(int(left["col"]) - int(right["col"]))
        assert step == 1
    assert render_map["shortest_path_entry_costs_after_conversion"][0] == 0
    assert sum(int(cost) for cost in render_map["shortest_path_entry_costs_after_conversion"]) == int(out.answer_gt.value)
