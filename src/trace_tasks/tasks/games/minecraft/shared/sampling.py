"""Identity-free sampling primitives for Minecraft-like block-world scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.tasks.games.shared.sampling import resolve_games_named_axis
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support

from .defaults import (
    DEFAULTS,
    HEIGHT_CONDITION_AT_LEAST,
    HEIGHT_CONDITION_EXACT,
    ORE_KINDS,
    RESOURCE_KINDS,
    ROUTE_DISTRACTOR_MIN_TRACK_DISTANCE,
    SAMPLE_KIND_HEIGHT_FILTER,
    SAMPLE_KIND_ROUTE_COST,
    SAMPLE_KIND_TOP_RESOURCE,
    STYLE_VARIANTS,
)
from .rules import validate_minecraft_sample
from .state import (
    MinecraftBlock,
    MinecraftCell,
    MinecraftRouteOverlay,
    MinecraftSceneSample,
    stack_entity_id,
    terrain_cell_entity_id,
    track_block_entity_id,
    water_cell_entity_id,
)


@dataclass(frozen=True)
class MinecraftAxes:
    """Resolved semantic and visual axes for one block-world instance."""

    style_variant: str
    grid_width: int
    grid_depth: int
    target_answer: int
    target_stack_height: int
    style_variant_probabilities: Dict[str, float]
    grid_width_probabilities: Dict[str, float]
    grid_depth_probabilities: Dict[str, float]
    answer_probabilities: Dict[str, float]
    target_stack_height_probabilities: Dict[str, float]


def resolve_top_resource_axes(
    instance_seed: int,
    *,
    gen_defaults: Mapping[str, Any],
    namespace: str,
    params: Mapping[str, Any],
) -> MinecraftAxes:
    """Resolve axes for a top-resource stack count scene."""

    return _resolve_axes(
        int(instance_seed),
        gen_defaults=gen_defaults,
        namespace=str(namespace),
        params=params,
        answer_support_key="top_ore_stack_answer_support",
        answer_fallback=DEFAULTS.top_resource_answer_support,
    )


def resolve_route_cost_axes(
    instance_seed: int,
    *,
    gen_defaults: Mapping[str, Any],
    namespace: str,
    params: Mapping[str, Any],
) -> MinecraftAxes:
    """Resolve axes for a single-track route-cost scene."""

    return _resolve_axes(
        int(instance_seed),
        gen_defaults=gen_defaults,
        namespace=str(namespace),
        params=params,
        answer_support_key="route_answer_support",
        answer_fallback=DEFAULTS.route_answer_support,
        width_support_key="route_grid_width_support",
        depth_support_key="route_grid_depth_support",
        width_fallback=DEFAULTS.route_grid_width_support,
        depth_fallback=DEFAULTS.route_grid_depth_support,
    )


def resolve_height_filter_axes(
    instance_seed: int,
    *,
    gen_defaults: Mapping[str, Any],
    namespace: str,
    params: Mapping[str, Any],
    height_condition: str,
) -> MinecraftAxes:
    """Resolve axes for exact-height or at-least-height stack counting."""

    if str(height_condition) == HEIGHT_CONDITION_EXACT:
        height_support_key = "exact_target_height_support"
        height_fallback = DEFAULTS.exact_target_height_support
    elif str(height_condition) == HEIGHT_CONDITION_AT_LEAST:
        height_support_key = "at_least_target_height_support"
        height_fallback = DEFAULTS.at_least_target_height_support
    else:
        raise ValueError(f"unsupported stack height condition: {height_condition}")

    axes = _resolve_axes(
        int(instance_seed),
        gen_defaults=gen_defaults,
        namespace=str(namespace),
        params=params,
        answer_support_key="stack_height_answer_support",
        answer_fallback=DEFAULTS.height_filter_answer_support,
    )
    target_stack_height, target_stack_height_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=height_support_key,
        explicit_key="target_stack_height",
        fallback_support=height_fallback,
        namespace=f"{namespace}.target_stack_height",
        balanced_flag_key="balanced_target_stack_height_sampling",
        namespace_support_permutation=True,
    )
    return MinecraftAxes(
        **{
            **axes.__dict__,
            "target_stack_height": int(target_stack_height),
            "target_stack_height_probabilities": dict(target_stack_height_probabilities),
        }
    )


def sample_top_resource_scene(
    *,
    rng: Any,
    axes: MinecraftAxes,
    gen_defaults: Mapping[str, Any],
    params: Mapping[str, Any],
) -> MinecraftSceneSample:
    """Construct a scene with an exact count of target-resource top cubes."""

    answer = int(axes.target_answer)
    target_kind = _resource_kind(rng=rng, params=params, allowed_kinds=ORE_KINDS)
    distractor_ore_kinds = tuple(kind for kind in ORE_KINDS if str(kind) != str(target_kind))
    height_support = tuple(sorted(set(_stack_height_support(params, gen_defaults=gen_defaults))))
    if not height_support:
        raise ValueError("top-resource stack count requires stack height support")
    distractor_count = int(rng.randrange(1, 4))
    target_specs = _sample_spaced_stack_specs(
        rng=rng,
        grid_width=int(axes.grid_width),
        grid_depth=int(axes.grid_depth),
        count=int(answer),
        height_support=height_support,
        min_x=1,
        max_x_exclusive=int(axes.grid_width) - 1,
        min_y=1,
        max_y_exclusive=int(axes.grid_depth) - 1,
        min_chebyshev_distance=2,
    )
    target_cells = tuple((int(x), int(y)) for x, y, _height in target_specs)
    distractor_avoid = _expand_cells(target_cells, radius=1)
    try:
        distractor_cells = _sample_distinct_cells(
            rng=rng,
            grid_width=int(axes.grid_width),
            grid_depth=int(axes.grid_depth),
            count=distractor_count,
            avoid=distractor_avoid,
            min_x=1,
            max_x_exclusive=int(axes.grid_width) - 1,
            min_y=1,
            max_y_exclusive=int(axes.grid_depth) - 1,
        )
    except ValueError:
        distractor_cells = _sample_distinct_cells(
            rng=rng,
            grid_width=int(axes.grid_width),
            grid_depth=int(axes.grid_depth),
            count=distractor_count,
            avoid=target_cells,
            min_x=1,
            max_x_exclusive=int(axes.grid_width) - 1,
            min_y=1,
            max_y_exclusive=int(axes.grid_depth) - 1,
        )
    stack_specs = tuple(target_specs) + tuple(
        (int(x), int(y), int(rng.choice(height_support)))
        for x, y in distractor_cells
    )
    blocks: list[MinecraftBlock] = []
    annotation_ids: list[str] = []
    for stack_index, (x, y, height) in enumerate(stack_specs):
        qualifies = int(stack_index) < int(answer)
        top_kind = str(target_kind) if qualifies else str(rng.choice(("stone", "dirt", *distractor_ore_kinds)))
        if qualifies:
            annotation_ids.append(stack_entity_id(int(x), int(y)))
        for z in range(int(height)):
            kind = str(top_kind if int(z) == int(height) - 1 else rng.choice(("stone", "dirt")))
            blocks.append(
                MinecraftBlock(
                    block_id=f"top_resource_stack_{int(stack_index):02d}_z{int(z):02d}",
                    x=int(x),
                    y=int(y),
                    z=int(z),
                    kind=kind,
                )
            )
    rng.shuffle(blocks)
    sample = MinecraftSceneSample(
        grid_width=int(axes.grid_width),
        grid_depth=int(axes.grid_depth),
        sample_kind=SAMPLE_KIND_TOP_RESOURCE,
        style_variant=str(axes.style_variant),
        answer=answer,
        terrain_cells=_terrain_cells_from_kinds(grid_width=int(axes.grid_width), grid_depth=int(axes.grid_depth)),
        blocks=tuple(blocks),
        player_cell=None,
        target_cell=None,
        river_width=0,
        scaffold_cost=0,
        ladder_present=False,
        annotation_entity_ids=tuple(sorted(annotation_ids)),
        construction_mode=f"top_resource_{answer}",
        target_resource_kind=target_kind,
        counted_resource_kind=target_kind,
    )
    validate_minecraft_sample(sample)
    return sample


def sample_route_cost_scene(*, rng: Any, axes: MinecraftAxes) -> MinecraftSceneSample:
    """Construct one visible track with an exact count of raised blocks."""

    answer = int(axes.target_answer)
    grid_width = int(axes.grid_width)
    grid_depth = int(axes.grid_depth)
    distractor_count = int(rng.randrange(2, 6))
    for _track_attempt in range(128):
        track_cells = _sample_track_cells(rng=rng, grid_width=grid_width, grid_depth=grid_depth)
        countable_track_cells = tuple(track_cells[1:-1])
        if int(answer) > len(countable_track_cells):
            continue
        try:
            distractor_cells = _sample_off_track_distractor_cells(
                rng=rng,
                grid_width=grid_width,
                grid_depth=grid_depth,
                track_cells=track_cells,
                count=distractor_count,
                min_track_distance=ROUTE_DISTRACTOR_MIN_TRACK_DISTANCE,
            )
        except ValueError:
            continue
        break
    else:
        raise ValueError("could not sample route distractors far enough from track")

    cell_kinds: dict[Tuple[int, int], str] = {cell: "route_path" for cell in track_cells}
    blocks: list[MinecraftBlock] = []
    annotation_ids: list[str] = []
    obstacle_cells = tuple(rng.sample(countable_track_cells, int(answer))) if answer else ()
    for obstacle_index, (x, y) in enumerate(obstacle_cells):
        block_id = track_block_entity_id(int(obstacle_index))
        blocks.append(
            MinecraftBlock(
                block_id=block_id,
                x=int(x),
                y=int(y),
                z=0,
                kind=str(rng.choice(("stone", "dirt"))),
            )
        )
        annotation_ids.append(block_id)

    for distractor_index, (x, y) in enumerate(distractor_cells):
        blocks.append(
            MinecraftBlock(
                block_id=f"route_distractor_block_{int(distractor_index):02d}",
                x=int(x),
                y=int(y),
                z=0,
                kind=str(rng.choice(("stone", "dirt"))),
            )
        )

    track_rgb = (224, 184, 45) if float(rng.random()) < 0.5 else (73, 210, 220)
    route_overlay = MinecraftRouteOverlay(label="", cells=track_cells, rgb=track_rgb)

    sample = MinecraftSceneSample(
        grid_width=grid_width,
        grid_depth=grid_depth,
        sample_kind=SAMPLE_KIND_ROUTE_COST,
        style_variant=str(axes.style_variant),
        answer=answer,
        terrain_cells=_terrain_cells_from_kinds(grid_width=grid_width, grid_depth=grid_depth, cell_kinds=cell_kinds),
        blocks=tuple(blocks),
        player_cell=None,
        target_cell=None,
        river_width=0,
        scaffold_cost=0,
        ladder_present=False,
        annotation_entity_ids=tuple(annotation_ids),
        construction_mode=f"single_track_route_cost_{answer}_distractors_{distractor_count}",
        route_overlays=(route_overlay,),
        track_cells=track_cells,
    )
    validate_minecraft_sample(sample)
    return sample


def sample_height_filter_scene(
    *,
    rng: Any,
    axes: MinecraftAxes,
    gen_defaults: Mapping[str, Any],
    params: Mapping[str, Any],
    height_condition: str,
) -> MinecraftSceneSample:
    """Construct a stack field with an exact count for one height predicate."""

    answer = int(axes.target_answer)
    target_height = int(axes.target_stack_height)
    height_support = tuple(sorted(set(_stack_height_support(params, gen_defaults=gen_defaults))))
    if target_height not in height_support:
        raise ValueError("target stack height must be in stack height support")
    if str(height_condition) == HEIGHT_CONDITION_EXACT:
        qualifying_heights = tuple(height for height in height_support if int(height) == target_height)
        distractor_heights = tuple(height for height in height_support if int(height) != target_height)
    elif str(height_condition) == HEIGHT_CONDITION_AT_LEAST:
        qualifying_heights = tuple(height for height in height_support if int(height) >= target_height)
        distractor_heights = tuple(height for height in height_support if int(height) < target_height)
    else:
        raise ValueError(f"unsupported stack height condition: {height_condition}")
    if not qualifying_heights or not distractor_heights:
        raise ValueError("height-filter query needs both qualifying and distractor heights")

    distractor_count = int(rng.randrange(2, 5))
    total_stacks = int(answer) + int(distractor_count)
    stack_cells = _sample_distinct_cells(
        rng=rng,
        grid_width=int(axes.grid_width),
        grid_depth=int(axes.grid_depth),
        count=total_stacks,
        min_x=1,
        max_x_exclusive=int(axes.grid_width) - 1,
        min_y=1,
        max_y_exclusive=int(axes.grid_depth) - 1,
    )
    blocks: list[MinecraftBlock] = []
    annotation_ids: list[str] = []
    for stack_index, (x, y) in enumerate(stack_cells):
        qualifies = int(stack_index) < int(answer)
        height = int(rng.choice(qualifying_heights if qualifies else distractor_heights))
        if qualifies:
            annotation_ids.append(stack_entity_id(int(x), int(y)))
        for z in range(int(height)):
            blocks.append(
                MinecraftBlock(
                    block_id=f"height_stack_{int(stack_index):02d}_z{int(z):02d}",
                    x=int(x),
                    y=int(y),
                    z=int(z),
                    kind=str(rng.choice(("stone", "dirt"))),
                )
            )
    rng.shuffle(blocks)
    sample = MinecraftSceneSample(
        grid_width=int(axes.grid_width),
        grid_depth=int(axes.grid_depth),
        sample_kind=SAMPLE_KIND_HEIGHT_FILTER,
        style_variant=str(axes.style_variant),
        answer=answer,
        terrain_cells=_terrain_cells_from_kinds(grid_width=int(axes.grid_width), grid_depth=int(axes.grid_depth)),
        blocks=tuple(blocks),
        player_cell=None,
        target_cell=None,
        river_width=0,
        scaffold_cost=0,
        ladder_present=False,
        annotation_entity_ids=tuple(sorted(annotation_ids)),
        construction_mode=f"height_filter_{height_condition}_{target_height}_{answer}",
        target_stack_height=int(target_height),
        stack_height_condition=str(height_condition),
    )
    validate_minecraft_sample(sample)
    return sample


def stack_height_support(params: Mapping[str, Any], *, gen_defaults: Mapping[str, Any]) -> Tuple[int, ...]:
    """Resolve allowed visible stack heights for block-world constructions."""

    return _stack_height_support(params, gen_defaults=gen_defaults)


def _resolve_style_variant(
    instance_seed: int,
    *,
    gen_defaults: Mapping[str, Any],
    namespace: str,
    params: Mapping[str, Any],
) -> tuple[str, Dict[str, float]]:
    """Resolve the scene style axis shared by all block-world tasks."""

    return resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=f"{namespace}.style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported_variants=STYLE_VARIANTS,
    )


def _resolve_axes(
    instance_seed: int,
    *,
    gen_defaults: Mapping[str, Any],
    namespace: str,
    params: Mapping[str, Any],
    answer_support_key: str,
    answer_fallback: Sequence[int],
    width_support_key: str = "grid_width_support",
    depth_support_key: str = "grid_depth_support",
    width_fallback: Sequence[int] = DEFAULTS.grid_width_support,
    depth_fallback: Sequence[int] = DEFAULTS.grid_depth_support,
) -> MinecraftAxes:
    """Resolve common style, board-size, and target-answer axes."""

    style_variant, style_variant_probabilities = _resolve_style_variant(
        int(instance_seed),
        gen_defaults=gen_defaults,
        namespace=str(namespace),
        params=params,
    )
    grid_width, grid_width_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(width_support_key),
        explicit_key="grid_width",
        fallback_support=tuple(int(value) for value in width_fallback),
        namespace=f"{namespace}.grid_width",
        balanced_flag_key="balanced_grid_width_sampling",
        namespace_support_permutation=True,
    )
    grid_depth, grid_depth_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(depth_support_key),
        explicit_key="grid_depth",
        fallback_support=tuple(int(value) for value in depth_fallback),
        namespace=f"{namespace}.grid_depth",
        balanced_flag_key="balanced_grid_depth_sampling",
        namespace_support_permutation=True,
    )
    target_answer, answer_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(answer_support_key),
        explicit_key="target_answer",
        fallback_support=tuple(int(value) for value in answer_fallback),
        namespace=f"{namespace}.answer",
        balanced_flag_key="balanced_answer_sampling",
        namespace_support_permutation=True,
    )
    return MinecraftAxes(
        style_variant=str(style_variant),
        grid_width=int(grid_width),
        grid_depth=int(grid_depth),
        target_answer=int(target_answer),
        target_stack_height=0,
        style_variant_probabilities=dict(style_variant_probabilities),
        grid_width_probabilities=dict(grid_width_probabilities),
        grid_depth_probabilities=dict(grid_depth_probabilities),
        answer_probabilities=dict(answer_probabilities),
        target_stack_height_probabilities={"0": 1.0},
    )


def _terrain_cells_from_kinds(
    *,
    grid_width: int,
    grid_depth: int,
    cell_kinds: Mapping[Tuple[int, int], str] | None = None,
) -> Tuple[MinecraftCell, ...]:
    """Build terrain cells for every visible ground coordinate."""

    kind_by_cell = {(int(x), int(y)): str(kind) for (x, y), kind in dict(cell_kinds or {}).items()}
    cells: list[MinecraftCell] = []
    for y in range(int(grid_depth)):
        for x in range(int(grid_width)):
            kind = str(kind_by_cell.get((int(x), int(y)), "ground"))
            if kind == "water":
                cells.append(MinecraftCell(cell_id=water_cell_entity_id(int(x), int(y)), x=int(x), y=int(y), kind="water"))
            else:
                cells.append(MinecraftCell(cell_id=terrain_cell_entity_id(int(x), int(y)), x=int(x), y=int(y), kind=kind))
    return tuple(cells)


def _resource_kind(*, rng: Any, params: Mapping[str, Any], allowed_kinds: Sequence[str] | None = None) -> str:
    """Resolve or sample one prompt-facing resource block kind."""

    explicit = params.get("resource_kind", params.get("target_resource_kind"))
    supported = tuple(str(kind) for kind in (allowed_kinds or RESOURCE_KINDS))
    if explicit is not None:
        kind = str(explicit)
        if kind not in supported:
            raise ValueError(f"unsupported resource kind: {kind}")
        return str(kind)
    return str(rng.choice(supported))


def _sample_distinct_cells(
    *,
    rng: Any,
    grid_width: int,
    grid_depth: int,
    count: int,
    avoid: Sequence[Tuple[int, int]] = (),
    min_x: int = 1,
    max_x_exclusive: int | None = None,
    min_y: int = 1,
    max_y_exclusive: int | None = None,
) -> Tuple[Tuple[int, int], ...]:
    """Sample distinct cells from an interior rectangular range."""

    avoid_set = {(int(x), int(y)) for x, y in avoid}
    max_x = int(max_x_exclusive) if max_x_exclusive is not None else int(grid_width) - 1
    max_y = int(max_y_exclusive) if max_y_exclusive is not None else int(grid_depth) - 1
    candidates = [
        (int(x), int(y))
        for y in range(int(min_y), int(max_y))
        for x in range(int(min_x), int(max_x))
        if (int(x), int(y)) not in avoid_set
    ]
    if len(candidates) < int(count):
        raise ValueError("not enough minecraft cells to sample")
    return tuple(rng.sample(candidates, int(count)))


def _sample_spaced_cells(
    *,
    rng: Any,
    grid_width: int,
    grid_depth: int,
    count: int,
    min_x: int = 1,
    max_x_exclusive: int | None = None,
    min_y: int = 1,
    max_y_exclusive: int | None = None,
    min_chebyshev_distance: int = 2,
) -> Tuple[Tuple[int, int], ...]:
    """Sample cells whose grid centers remain visually separated."""

    max_x = int(max_x_exclusive) if max_x_exclusive is not None else int(grid_width) - 1
    max_y = int(max_y_exclusive) if max_y_exclusive is not None else int(grid_depth) - 1
    candidates = [
        (int(x), int(y))
        for y in range(int(min_y), int(max_y))
        for x in range(int(min_x), int(max_x))
    ]
    for _attempt in range(128):
        remaining = list(candidates)
        rng.shuffle(remaining)
        selected: list[Tuple[int, int]] = []
        while remaining and len(selected) < int(count):
            cell = remaining.pop()
            if all(
                max(abs(int(cell[0]) - int(other[0])), abs(int(cell[1]) - int(other[1])))
                >= int(min_chebyshev_distance)
                for other in selected
            ):
                selected.append((int(cell[0]), int(cell[1])))
        if len(selected) == int(count):
            return tuple(selected)
    raise ValueError("not enough separated minecraft cells to sample")


def _sample_spaced_stack_specs(
    *,
    rng: Any,
    grid_width: int,
    grid_depth: int,
    count: int,
    height_support: Sequence[int],
    min_x: int = 1,
    max_x_exclusive: int | None = None,
    min_y: int = 1,
    max_y_exclusive: int | None = None,
    min_chebyshev_distance: int = 2,
) -> Tuple[Tuple[int, int, int], ...]:
    """Sample stack cells and heights with distinct isometric top projections."""

    heights = tuple(int(height) for height in height_support)
    if not heights:
        raise ValueError("spaced stack specs require height support")
    for _attempt in range(256):
        cells = _sample_spaced_cells(
            rng=rng,
            grid_width=int(grid_width),
            grid_depth=int(grid_depth),
            count=int(count),
            min_x=int(min_x),
            max_x_exclusive=max_x_exclusive,
            min_y=int(min_y),
            max_y_exclusive=max_y_exclusive,
            min_chebyshev_distance=int(min_chebyshev_distance),
        )
        specs = tuple((int(x), int(y), int(rng.choice(heights))) for x, y in cells)
        if _stack_top_projection_keys_are_unique(specs):
            return specs
    raise ValueError("not enough separated minecraft stack projections to sample")


def _stack_top_projection_keys_are_unique(specs: Sequence[Tuple[int, int, int]]) -> bool:
    """Approximate top-center projection uniqueness for isometric stacks."""

    keys: set[Tuple[int, int]] = set()
    for x, y, height in specs:
        key = (int(x) - int(y), int(x) + int(y) - (2 * (int(height) - 1)))
        if key in keys:
            return False
        keys.add(key)
    return True


def _expand_cells(cells: Sequence[Tuple[int, int]], *, radius: int) -> Tuple[Tuple[int, int], ...]:
    """Return the Chebyshev neighborhood around cells for visual de-crowding."""

    expanded: set[Tuple[int, int]] = set()
    for x, y in cells:
        for dy in range(-int(radius), int(radius) + 1):
            for dx in range(-int(radius), int(radius) + 1):
                expanded.add((int(x) + int(dx), int(y) + int(dy)))
    return tuple(sorted(expanded))


def _sample_track_cells(*, rng: Any, grid_width: int, grid_depth: int) -> Tuple[Tuple[int, int], ...]:
    """Sample one connected left-to-right track with at least one gentle bend."""

    if int(grid_width) < 6 or int(grid_depth) < 5:
        raise ValueError("track route needs a grid at least 6 x 5")
    min_y = 1
    max_y = int(grid_depth) - 2
    y = int(rng.randrange(min_y, max_y + 1))
    force_turn_x = int(rng.randrange(2, max(3, int(grid_width) - 2)))
    cells: list[Tuple[int, int]] = []
    for x in range(1, int(grid_width) - 1):
        if int(x) > 1:
            turn_candidates = [0]
            if int(y) > min_y:
                turn_candidates.append(-1)
            if int(y) < max_y:
                turn_candidates.append(1)
            if int(x) == force_turn_x:
                nonzero = [candidate for candidate in turn_candidates if int(candidate) != 0]
                delta = int(rng.choice(nonzero or [0]))
            elif float(rng.random()) < 0.30:
                delta = int(rng.choice(turn_candidates))
            else:
                delta = 0
            y = max(min_y, min(max_y, int(y) + int(delta)))
        cells.append((int(x), int(y)))
    return tuple(cells)


def _sample_off_track_distractor_cells(
    *,
    rng: Any,
    grid_width: int,
    grid_depth: int,
    track_cells: Sequence[Tuple[int, int]],
    count: int,
    min_track_distance: int,
) -> Tuple[Tuple[int, int], ...]:
    """Sample raised-block distractor cells far from the visible track."""

    track_set = {(int(x), int(y)) for x, y in track_cells}
    candidates = [
        (int(x), int(y))
        for y in range(1, int(grid_depth) - 1)
        for x in range(1, int(grid_width) - 1)
        if (int(x), int(y)) not in track_set
        and _min_chebyshev_distance_to_cells((int(x), int(y)), track_cells) >= int(min_track_distance)
    ]
    if len(candidates) < int(count):
        raise ValueError("not enough off-track cells for route distractors")
    return tuple(rng.sample(candidates, int(count)))


def _min_chebyshev_distance_to_cells(
    cell: Tuple[int, int],
    cells: Sequence[Tuple[int, int]],
) -> int:
    """Return the nearest Chebyshev grid distance from one cell to a cell set."""

    if not cells:
        raise ValueError("distance requires at least one reference cell")
    cx, cy = int(cell[0]), int(cell[1])
    return min(max(abs(cx - int(x)), abs(cy - int(y))) for x, y in cells)


def _stack_height_support(params: Mapping[str, Any], *, gen_defaults: Mapping[str, Any]) -> Tuple[int, ...]:
    """Resolve configured stack-height support."""

    return tuple(
        int(value)
        for value in resolve_integer_support(
            params,
            gen_defaults=gen_defaults,
            key="stack_height_support",
            fallback=DEFAULTS.stack_height_support,
        )
    )


__all__ = [
    "MinecraftAxes",
    "resolve_height_filter_axes",
    "resolve_route_cost_axes",
    "resolve_top_resource_axes",
    "sample_height_filter_scene",
    "sample_route_cost_scene",
    "sample_top_resource_scene",
    "stack_height_support",
]
