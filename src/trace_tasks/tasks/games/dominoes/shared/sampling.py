"""Identity-free sampling primitives for dominoes scene tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.tasks.games.shared.layout import resolve_games_layout_jitter
from trace_tasks.tasks.games.shared.sampling import resolve_games_integer_axis, resolve_games_named_axis
from trace_tasks.tasks.games.shared.style import SUPPORTED_DOMINO_STYLE_VARIANTS
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.font_assets import sample_font_family
from trace_tasks.tasks.shared.support_sampling import resolve_integer_support

from .defaults import DEFAULTS, DOMINOES_NAMESPACE, SUPPORTED_DOMINO_SCENE_VARIANTS
from .rendering import DominoRenderParams
from .rules import CANONICAL_DOMINOES, OPTION_LABELS, PIP_VALUES, canonical_tile
from .state import DominoIntegerAxis, DominoSceneAxes, DominoTileInstance, SampledDominoScene


@dataclass(frozen=True)
class CountedCandidateRecipe:
    """Task-owned counted-candidate setup over one sampled domino chain."""

    oriented_chain: Tuple[Tuple[int, int], ...]
    is_annotation_tile: Callable[[Tuple[int, int]], bool]
    reference_role: str | None
    highlight_open_end: bool
    open_end_value: int | None = None
    reference_sum: int | None = None


def _resolve_named_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace: str,
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    supported: Sequence[str],
) -> Tuple[str, Dict[str, float]]:
    """Resolve one scene-level string axis with optional balanced sampling."""

    return resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=f"{DOMINOES_NAMESPACE}.{str(namespace)}",
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        balance_flag_key=str(balance_flag_key),
        supported_variants=[str(item) for item in supported],
    )


def resolve_domino_scene_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> DominoSceneAxes:
    """Resolve the scene layout and domino visual style axes."""

    scene_variant, scene_variant_probabilities = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace="scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported=SUPPORTED_DOMINO_SCENE_VARIANTS,
    )
    style_variant, style_variant_probabilities = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace="style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported=SUPPORTED_DOMINO_STYLE_VARIANTS,
    )
    return DominoSceneAxes(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        style_variant_probabilities=dict(style_variant_probabilities),
    )


def resolve_domino_integer_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[int],
    namespace: str,
    balanced_flag_key: str,
) -> DominoIntegerAxis:
    """Resolve one task-owned integer axis and return trace metadata."""

    value, support, probabilities = resolve_games_integer_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(support_key),
        explicit_key=str(explicit_key),
        fallback_support=tuple(int(value) for value in fallback_support),
        namespace=f"{DOMINOES_NAMESPACE}.{str(namespace)}",
        balanced_flag_key=str(balanced_flag_key),
    )
    return DominoIntegerAxis(value=int(value), support=tuple(int(v) for v in support), probabilities=dict(probabilities))


def resolve_domino_target_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    fallback_support: Sequence[int],
    namespace: str,
) -> DominoIntegerAxis:
    """Resolve a task-owned target-answer axis."""

    return resolve_domino_integer_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(support_key),
        explicit_key="target_answer",
        fallback_support=fallback_support,
        namespace=str(namespace),
        balanced_flag_key="balanced_target_answer_sampling",
    )


def resolve_domino_candidate_count_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    scene_variant: str,
    minimum_candidate_count: int,
    namespace: str,
) -> DominoIntegerAxis:
    """Resolve a visible candidate count after the task supplies feasibility."""

    support_key = {
        "single_row": "single_row_candidate_count_support",
        "two_row": "two_row_candidate_count_support",
    }[str(scene_variant)]
    raw_support = resolve_integer_support(
        params,
        gen_defaults=gen_defaults,
        key=support_key,
        fallback=getattr(DEFAULTS, support_key),
    )
    feasible = tuple(int(value) for value in raw_support if int(value) >= int(minimum_candidate_count))
    if not feasible:
        raise ValueError(f"no feasible candidate_count values remain for minimum {int(minimum_candidate_count)}")
    candidate_params = dict(params)
    candidate_params[support_key] = [int(value) for value in feasible]
    return resolve_domino_integer_axis(
        instance_seed=int(instance_seed),
        params=candidate_params,
        gen_defaults=gen_defaults,
        support_key=support_key,
        explicit_key="candidate_count",
        fallback_support=feasible,
        namespace=f"candidate_count.{str(scene_variant)}.{str(namespace)}",
        balanced_flag_key="balanced_candidate_count_sampling",
    )


def build_tile_instance(
    *,
    tile_id: str,
    oriented_tile: Tuple[int, int],
    role: str,
    is_reference: bool = False,
    highlight_right_half: bool = False,
    option_label: str | None = None,
    right_join_label: str | None = None,
) -> DominoTileInstance:
    """Build one rendered domino tile payload from oriented half values."""

    return DominoTileInstance(
        tile_id=str(tile_id),
        left_value=int(oriented_tile[0]),
        right_value=int(oriented_tile[1]),
        role=str(role),
        is_reference=bool(is_reference),
        highlight_right_half=bool(highlight_right_half),
        option_label=None if option_label is None else str(option_label),
        right_join_label=None if right_join_label is None else str(right_join_label),
    )


def sample_chain_with_end(
    rng,
    *,
    end_tile: Tuple[int, int],
    avoid_prefix_values: Sequence[int] = (),
    avoid_prefix_doubles: bool = False,
) -> Tuple[Tuple[int, int], ...]:
    """Sample one valid oriented 3-tile chain whose rightmost tile is fixed."""

    end_left = int(end_tile[0])
    end_right = int(end_tile[1])
    blocked_values = {int(value) for value in avoid_prefix_values}
    disallowed = {canonical_tile(int(end_left), int(end_right))}
    middle_options: List[Tuple[int, int]] = []
    for middle_left in PIP_VALUES:
        middle_canonical = canonical_tile(int(middle_left), int(end_left))
        if middle_canonical in disallowed:
            continue
        if int(middle_left) in blocked_values or int(end_left) in blocked_values:
            continue
        if bool(avoid_prefix_doubles) and int(middle_left) == int(end_left):
            continue
        middle_options.append((int(middle_left), int(end_left)))
    if not middle_options:
        raise ValueError("unable to sample middle domino for chain")
    middle_tile = middle_options[int(rng.randrange(len(middle_options)))]
    disallowed.add(canonical_tile(int(middle_tile[0]), int(middle_tile[1])))

    first_options: List[Tuple[int, int]] = []
    for first_left in PIP_VALUES:
        first_canonical = canonical_tile(int(first_left), int(middle_tile[0]))
        if first_canonical in disallowed:
            continue
        if int(first_left) in blocked_values or int(middle_tile[0]) in blocked_values:
            continue
        if bool(avoid_prefix_doubles) and int(first_left) == int(middle_tile[0]):
            continue
        first_options.append((int(first_left), int(middle_tile[0])))
    if not first_options:
        raise ValueError("unable to sample first domino for chain")
    first_tile = first_options[int(rng.randrange(len(first_options)))]
    return (
        (int(first_tile[0]), int(first_tile[1])),
        (int(middle_tile[0]), int(middle_tile[1])),
        (int(end_left), int(end_right)),
    )


def sample_generic_chain(rng, *, avoid_doubles: bool) -> Tuple[Tuple[int, int], ...]:
    """Sample one generic valid 3-tile domino chain."""

    for _ in range(96):
        end_left = int(PIP_VALUES[int(rng.randrange(len(PIP_VALUES)))])
        end_right = int(PIP_VALUES[int(rng.randrange(len(PIP_VALUES)))])
        if bool(avoid_doubles) and int(end_left) == int(end_right):
            continue
        try:
            chain = sample_chain_with_end(
                rng,
                end_tile=(int(end_left), int(end_right)),
                avoid_prefix_doubles=bool(avoid_doubles),
            )
        except ValueError:
            continue
        if bool(avoid_doubles) and any(int(left_value) == int(right_value) for left_value, right_value in chain):
            continue
        return chain
    raise ValueError("unable to sample generic domino chain")


def candidate_pool_for_chain(oriented_chain: Sequence[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    """Return canonical dominoes not already consumed by the visible chain."""

    chain_tiles = {canonical_tile(int(left_value), int(right_value)) for left_value, right_value in oriented_chain}
    return tuple(tile for tile in CANONICAL_DOMINOES if tile not in chain_tiles)


def random_orientation(rng, *, tile: Tuple[int, int]) -> Tuple[int, int]:
    """Return one random visible orientation for a canonical domino tile."""

    left_value, right_value = int(tile[0]), int(tile[1])
    if int(left_value) == int(right_value):
        return (int(left_value), int(right_value))
    if bool(rng.randrange(2)):
        return (int(left_value), int(right_value))
    return (int(right_value), int(left_value))


def build_scene_instances(
    *,
    rng,
    oriented_chain: Sequence[Tuple[int, int]],
    candidate_tiles: Sequence[Tuple[int, int]],
    annotation_tiles: Sequence[Tuple[int, int]],
    reference_role: str | None,
    highlight_open_end: bool,
    shuffle_candidates: bool = True,
    label_candidates: bool = False,
) -> Tuple[Tuple[DominoTileInstance, ...], Tuple[DominoTileInstance, ...], Tuple[str, ...], str | None]:
    """Build rendered chain/candidate instances and witness ids."""

    annotation_canonicals = {canonical_tile(int(tile[0]), int(tile[1])) for tile in annotation_tiles}
    chain_instances: List[DominoTileInstance] = []
    candidate_instances: List[DominoTileInstance] = []
    annotation_tile_ids: List[str] = []
    reference_tile_id: str | None = None

    for index, oriented_tile in enumerate(oriented_chain, start=1):
        tile_id = f"chain_{index:02d}"
        is_reference = bool(index == len(oriented_chain) and reference_role is not None)
        chain_instances.append(
            build_tile_instance(
                tile_id=str(tile_id),
                oriented_tile=(int(oriented_tile[0]), int(oriented_tile[1])),
                role=str(reference_role) if bool(is_reference) else "chain",
                is_reference=bool(is_reference),
                highlight_right_half=bool(is_reference and highlight_open_end),
            )
        )
        if bool(is_reference):
            reference_tile_id = str(tile_id)

    ordered_candidates = list(candidate_tiles)
    if bool(shuffle_candidates):
        rng.shuffle(ordered_candidates)
    for index, canonical in enumerate(ordered_candidates, start=1):
        tile_id = f"candidate_{index:02d}"
        candidate_instances.append(
            build_tile_instance(
                tile_id=str(tile_id),
                oriented_tile=random_orientation(rng, tile=canonical),
                role="candidate",
                option_label=OPTION_LABELS[index - 1] if bool(label_candidates) else None,
            )
        )
        if canonical_tile(int(canonical[0]), int(canonical[1])) in annotation_canonicals:
            annotation_tile_ids.append(str(tile_id))

    return (
        tuple(chain_instances),
        tuple(candidate_instances),
        tuple(annotation_tile_ids),
        None if reference_tile_id is None else str(reference_tile_id),
    )


def build_tableau_instances(
    *,
    rng,
    candidate_tiles: Sequence[Tuple[int, int]],
    annotation_tiles: Sequence[Tuple[int, int]],
    reference_tile: Tuple[int, int] | None = None,
    reference_role: str | None = None,
    shuffle_tiles: bool = True,
    label_candidates: bool = False,
) -> Tuple[Tuple[DominoTileInstance, ...], Tuple[str, ...], str | None]:
    """Build a face-up domino tableau with an optional in-table reference tile."""

    annotation_canonicals = {canonical_tile(int(tile[0]), int(tile[1])) for tile in annotation_tiles}
    visible_entries: List[Tuple[str, Tuple[int, int]]] = [("candidate", tile) for tile in candidate_tiles]
    if reference_tile is not None:
        visible_entries.append(("reference", canonical_tile(int(reference_tile[0]), int(reference_tile[1]))))
    if bool(shuffle_tiles):
        rng.shuffle(visible_entries)

    instances: List[DominoTileInstance] = []
    annotation_tile_ids: List[str] = []
    reference_tile_id: str | None = None
    candidate_index = 0
    for _display_index, (entry_kind, canonical) in enumerate(visible_entries, start=1):
        if str(entry_kind) == "reference":
            tile_id = "reference_01"
            reference_tile_id = str(tile_id)
            instances.append(
                build_tile_instance(
                    tile_id=str(tile_id),
                    oriented_tile=random_orientation(rng, tile=canonical),
                    role=str(reference_role or "reference"),
                    is_reference=True,
                )
            )
            continue

        candidate_index += 1
        tile_id = f"candidate_{candidate_index:02d}"
        instances.append(
            build_tile_instance(
                tile_id=str(tile_id),
                oriented_tile=random_orientation(rng, tile=canonical),
                role="candidate",
                option_label=OPTION_LABELS[candidate_index - 1] if bool(label_candidates) else None,
            )
        )
        if canonical_tile(int(canonical[0]), int(canonical[1])) in annotation_canonicals:
            annotation_tile_ids.append(str(tile_id))

    return (
        tuple(instances),
        tuple(annotation_tile_ids),
        None if reference_tile_id is None else str(reference_tile_id),
    )


def tile_id_for_canonical(
    candidate_instances: Sequence[DominoTileInstance],
    canonical: Tuple[int, int],
) -> str | None:
    """Return the rendered candidate id for one canonical domino tile."""

    target = canonical_tile(int(canonical[0]), int(canonical[1]))
    for tile in candidate_instances:
        if canonical_tile(int(tile.left_value), int(tile.right_value)) == target:
            return str(tile.tile_id)
    return None


def build_sampled_scene(
    *,
    chain_instances: Sequence[DominoTileInstance],
    candidate_instances: Sequence[DominoTileInstance],
    annotation_tile_ids: Sequence[str],
    answer_value: int | str,
    reference_tile_id: str | None,
    open_end_value: int | None = None,
    reference_sum: int | None = None,
    target_total: int | None = None,
    first_step_tile_id: str | None = None,
    second_step_tile_id: str | None = None,
    bridge_value: int | None = None,
    candidate_extra_flags: Mapping[str, Mapping[str, Any]] | None = None,
) -> SampledDominoScene:
    """Build a JSON-friendly sampled scene record after task construction."""

    extra_flags = {str(key): dict(value) for key, value in dict(candidate_extra_flags or {}).items()}
    return SampledDominoScene(
        chain_tiles=tuple(chain_instances),
        candidate_tiles=tuple(candidate_instances),
        annotation_tile_ids=tuple(str(tile_id) for tile_id in annotation_tile_ids),
        answer_value=answer_value,
        reference_tile_id=None if reference_tile_id is None else str(reference_tile_id),
        open_end_value=None if open_end_value is None else int(open_end_value),
        reference_sum=None if reference_sum is None else int(reference_sum),
        target_total=None if target_total is None else int(target_total),
        first_step_tile_id=None if first_step_tile_id is None else str(first_step_tile_id),
        second_step_tile_id=None if second_step_tile_id is None else str(second_step_tile_id),
        bridge_value=None if bridge_value is None else int(bridge_value),
        chain_tile_specs=tuple(
            {
                "tile_id": str(tile.tile_id),
                "left_value": int(tile.left_value),
                "right_value": int(tile.right_value),
                "role": str(tile.role),
                "is_reference": bool(tile.is_reference),
                "option_label": None if tile.option_label is None else str(tile.option_label),
                "right_join_label": None if tile.right_join_label is None else str(tile.right_join_label),
            }
            for tile in chain_instances
        ),
        candidate_tile_specs=tuple(
            {
                "tile_id": str(tile.tile_id),
                "left_value": int(tile.left_value),
                "right_value": int(tile.right_value),
                "role": str(tile.role),
                "is_reference": bool(tile.is_reference),
                "option_label": None if tile.option_label is None else str(tile.option_label),
                "right_join_label": None if tile.right_join_label is None else str(tile.right_join_label),
                **extra_flags.get(str(tile.tile_id), {}),
            }
            for tile in candidate_instances
        ),
    )


def sample_counted_candidate_scene(
    rng,
    *,
    oriented_chain: Sequence[Tuple[int, int]],
    candidate_count: int,
    target_answer: int,
    is_annotation_tile: Callable[[Tuple[int, int]], bool],
    reference_role: str | None,
    highlight_open_end: bool,
    open_end_value: int | None = None,
    reference_sum: int | None = None,
) -> SampledDominoScene:
    """Sample target-count candidate witnesses plus fillers for one chain."""

    candidate_pool = candidate_pool_for_chain(oriented_chain)
    annotation_pool = [tile for tile in candidate_pool if bool(is_annotation_tile(tile))]
    filler_pool = [tile for tile in candidate_pool if not bool(is_annotation_tile(tile))]
    if int(len(annotation_pool)) < int(target_answer):
        raise ValueError("insufficient annotation candidates")
    if int(len(filler_pool)) < int(candidate_count - target_answer):
        raise ValueError("insufficient filler candidates")
    annotation_tiles = list(rng.sample(annotation_pool, int(target_answer)))
    selected_candidates = annotation_tiles + list(rng.sample(filler_pool, int(candidate_count - target_answer)))
    chain_instances, candidate_instances, annotation_tile_ids, reference_tile_id = build_scene_instances(
        rng=rng,
        oriented_chain=oriented_chain,
        candidate_tiles=selected_candidates,
        annotation_tiles=annotation_tiles,
        reference_role=reference_role,
        highlight_open_end=bool(highlight_open_end),
    )
    return build_sampled_scene(
        chain_instances=chain_instances,
        candidate_instances=candidate_instances,
        annotation_tile_ids=annotation_tile_ids,
        answer_value=int(target_answer),
        reference_tile_id=reference_tile_id,
        open_end_value=open_end_value,
        reference_sum=reference_sum,
    )


def sample_counted_candidate_scene_from_recipe(
    rng,
    *,
    attempts: int,
    candidate_count: int,
    target_answer: int,
    build_recipe: Callable[[Any], CountedCandidateRecipe],
) -> SampledDominoScene:
    """Retry a task-owned counted-candidate recipe until it yields a scene."""

    for _ in range(max(1, int(attempts))):
        try:
            recipe = build_recipe(rng)
            return sample_counted_candidate_scene(
                rng,
                oriented_chain=recipe.oriented_chain,
                candidate_count=int(candidate_count),
                target_answer=int(target_answer),
                is_annotation_tile=recipe.is_annotation_tile,
                reference_role=recipe.reference_role,
                highlight_open_end=bool(recipe.highlight_open_end),
                open_end_value=recipe.open_end_value,
                reference_sum=recipe.reference_sum,
            )
        except ValueError:
            continue
    raise ValueError("unable to sample counted domino candidate scene")


def resolve_domino_render_params(params: Mapping[str, Any], *, render_defaults: Mapping[str, Any], instance_seed: int) -> DominoRenderParams:
    """Resolve domino-scene rendering parameters from config/defaults."""

    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace="games.dominoes.font_family",
        params=params,
    )
    return DominoRenderParams(
        canvas_width=int(params.get("canvas_width", group_default(render_defaults, "canvas_width", DEFAULTS.canvas_width))),
        canvas_height=int(params.get("canvas_height", group_default(render_defaults, "canvas_height", DEFAULTS.canvas_height))),
        panel_margin_px=int(params.get("panel_margin_px", group_default(render_defaults, "panel_margin_px", DEFAULTS.panel_margin_px))),
        chain_top_px=int(params.get("chain_top_px", group_default(render_defaults, "chain_top_px", DEFAULTS.chain_top_px))),
        tile_width_px=int(params.get("tile_width_px", group_default(render_defaults, "tile_width_px", DEFAULTS.tile_width_px))),
        tile_height_px=int(params.get("tile_height_px", group_default(render_defaults, "tile_height_px", DEFAULTS.tile_height_px))),
        chain_gap_px=int(params.get("chain_gap_px", group_default(render_defaults, "chain_gap_px", DEFAULTS.chain_gap_px))),
        candidate_gap_px=int(params.get("candidate_gap_px", group_default(render_defaults, "candidate_gap_px", DEFAULTS.candidate_gap_px))),
        row_gap_px=int(params.get("row_gap_px", group_default(render_defaults, "row_gap_px", DEFAULTS.row_gap_px))),
        tile_corner_radius_px=int(
            params.get("tile_corner_radius_px", group_default(render_defaults, "tile_corner_radius_px", DEFAULTS.tile_corner_radius_px))
        ),
        pip_radius_px=int(params.get("pip_radius_px", group_default(render_defaults, "pip_radius_px", DEFAULTS.pip_radius_px))),
        divider_width_px=int(params.get("divider_width_px", group_default(render_defaults, "divider_width_px", DEFAULTS.divider_width_px))),
        reference_tag_font_size_px=int(
            params.get("reference_tag_font_size_px", group_default(render_defaults, "reference_tag_font_size_px", DEFAULTS.reference_tag_font_size_px))
        ),
        reference_tag_gap_px=int(
            params.get("reference_tag_gap_px", group_default(render_defaults, "reference_tag_gap_px", DEFAULTS.reference_tag_gap_px))
        ),
        section_label_font_size_px=int(
            params.get("section_label_font_size_px", group_default(render_defaults, "section_label_font_size_px", DEFAULTS.section_label_font_size_px))
        ),
        section_separator_width_px=int(
            params.get("section_separator_width_px", group_default(render_defaults, "section_separator_width_px", DEFAULTS.section_separator_width_px))
        ),
        font_family=str(font_family),
        layout_jitter_meta=resolve_games_layout_jitter(
            params,
            render_defaults,
            instance_seed=int(instance_seed),
            namespace="games.dominoes.layout",
        ),
    )


__all__ = [
    "build_sampled_scene",
    "build_scene_instances",
    "build_tableau_instances",
    "build_tile_instance",
    "candidate_pool_for_chain",
    "CountedCandidateRecipe",
    "random_orientation",
    "resolve_domino_candidate_count_axis",
    "resolve_domino_integer_axis",
    "resolve_domino_render_params",
    "resolve_domino_scene_axes",
    "resolve_domino_target_axis",
    "sample_counted_candidate_scene",
    "sample_chain_with_end",
    "sample_counted_candidate_scene_from_recipe",
    "sample_generic_chain",
    "tile_id_for_canonical",
]
