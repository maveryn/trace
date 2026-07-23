"""Compute the longest one-sided domino chain from the marked reference end."""

from __future__ import annotations

from functools import lru_cache
from typing import FrozenSet, Sequence, Tuple

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import DominoObjectivePlan, domino_bbox_set_attempt, resolve_domino_count_axes, run_domino_lifecycle
from .shared.defaults import DEFAULTS, SCENE_ID
from .shared.prompts import domino_integer_json_examples, domino_output_slots
from .shared.rules import PIP_VALUES, can_connect, canonical_tile, chain_open_end_after_play
from .shared.sampling import build_sampled_scene, build_scene_instances, candidate_pool_for_chain, sample_chain_with_end
from .shared.state import DominoSceneAxes


TASK_ID = "task_games__dominoes__longest_chain_length_value"
QUERY_ID = "longest_chain_length_value"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _longest_chain_index_sets(
    *,
    candidate_tiles: Sequence[Tuple[int, int]],
    open_end_value: int,
) -> tuple[int, tuple[FrozenSet[int], ...]]:
    """Return the maximum chain length and unique tile-index sets achieving it."""

    tiles = tuple(canonical_tile(int(tile[0]), int(tile[1])) for tile in candidate_tiles)

    @lru_cache(maxsize=None)
    def search(current_open_end: int, remaining_indices: tuple[int, ...]) -> tuple[int, tuple[FrozenSet[int], ...]]:
        best_length = 0
        best_sets: set[FrozenSet[int]] = {frozenset()}
        for index in remaining_indices:
            tile = tiles[int(index)]
            if not can_connect(tile, int(current_open_end)):
                continue
            next_open_end = chain_open_end_after_play(tile, int(current_open_end))
            next_remaining = tuple(other for other in remaining_indices if int(other) != int(index))
            child_length, child_sets = search(int(next_open_end), next_remaining)
            total_length = int(child_length) + 1
            total_sets = {frozenset({int(index), *child_set}) for child_set in child_sets}
            if int(total_length) > int(best_length):
                best_length = int(total_length)
                best_sets = total_sets
            elif int(total_length) == int(best_length):
                best_sets.update(total_sets)
        return int(best_length), tuple(sorted(best_sets, key=lambda value: tuple(sorted(value))))

    return search(int(open_end_value), tuple(range(len(tiles))))


def _sample_path_tiles(
    rng,
    *,
    candidate_pool: Sequence[Tuple[int, int]],
    open_end_value: int,
    path_length: int,
) -> list[Tuple[int, int]]:
    """Sample one valid loose-domino path of the requested length."""

    available = {canonical_tile(int(tile[0]), int(tile[1])) for tile in candidate_pool}
    current_open_end = int(open_end_value)
    path_tiles: list[Tuple[int, int]] = []
    for _ in range(int(path_length)):
        options = [tile for tile in available if can_connect(tile, int(current_open_end))]
        if not options:
            raise ValueError("unable to sample requested domino path")
        tile = options[int(rng.randrange(len(options)))]
        path_tiles.append(tile)
        available.remove(tile)
        current_open_end = chain_open_end_after_play(tile, int(current_open_end))
    return path_tiles


def _sample_longest_chain_scene(rng, *, candidate_count: int, target_answer: int):
    """Construct a chain scene whose unique longest extension has target length."""

    target_length = int(target_answer)
    for _ in range(1200):
        open_end_value = int(PIP_VALUES[int(rng.randrange(len(PIP_VALUES)))])
        connector_options = [value for value in PIP_VALUES if int(value) != int(open_end_value)]
        connector_value = int(connector_options[int(rng.randrange(len(connector_options)))])
        try:
            oriented_chain = sample_chain_with_end(
                rng,
                end_tile=(int(connector_value), int(open_end_value)),
            )
        except ValueError:
            continue

        candidate_pool = tuple(candidate_pool_for_chain(oriented_chain))
        try:
            path_tiles = _sample_path_tiles(
                rng,
                candidate_pool=candidate_pool,
                open_end_value=int(open_end_value),
                path_length=int(target_length),
            )
        except ValueError:
            continue

        path_set = {canonical_tile(int(tile[0]), int(tile[1])) for tile in path_tiles}
        filler_pool = [tile for tile in candidate_pool if tile not in path_set]
        filler_count = int(candidate_count) - int(target_length)
        if int(filler_count) < 0 or int(len(filler_pool)) < int(filler_count):
            continue
        selected_candidates = list(path_tiles) + list(rng.sample(filler_pool, int(filler_count)))
        longest_length, longest_sets = _longest_chain_index_sets(
            candidate_tiles=selected_candidates,
            open_end_value=int(open_end_value),
        )
        if int(longest_length) != int(target_length) or len(longest_sets) != 1:
            continue

        longest_indices = set(next(iter(longest_sets)))
        annotation_tiles = [selected_candidates[int(index)] for index in sorted(longest_indices)]
        chain_instances, candidate_instances, annotation_tile_ids, reference_tile_id = build_scene_instances(
            rng=rng,
            oriented_chain=oriented_chain,
            candidate_tiles=selected_candidates,
            annotation_tiles=annotation_tiles,
            reference_role="reference_end",
            highlight_open_end=True,
        )
        return build_sampled_scene(
            chain_instances=chain_instances,
            candidate_instances=candidate_instances,
            annotation_tile_ids=annotation_tile_ids,
            answer_value=int(target_length),
            reference_tile_id=reference_tile_id,
            open_end_value=int(open_end_value),
            reference_sum=int(oriented_chain[-1][0] + oriented_chain[-1][1]),
            candidate_extra_flags={
                str(tile.tile_id): {"is_longest_chain_member": bool(str(tile.tile_id) in set(annotation_tile_ids))}
                for tile in candidate_instances
            },
        )
    raise ValueError("unable to sample unique longest-chain domino scene")


def _prepare_longest_chain_objective(
    instance_seed,
    task_params,
    _query_id,
    _query_probabilities,
    axes: DominoSceneAxes,
):
    """Resolve axes and bind one-sided longest-chain semantics."""

    count_axes = resolve_domino_count_axes(
        instance_seed=int(instance_seed),
        task_params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        axes=axes,
        target_support_key="longest_chain_length_answer_support",
        target_fallback_support=DEFAULTS.longest_chain_length_answer_support,
        target_namespace="longest_chain.target_answer",
        minimum_candidate_count=lambda target: max(7, int(target)),
        candidate_namespace="longest_chain",
    )
    json_example, json_example_answer_only = domino_integer_json_examples(answer_value=3)
    prompt_dynamic_slots = domino_output_slots(
        prompt_query_key=QUERY_ID,
        json_example=json_example,
        json_example_answer_only=json_example_answer_only,
    )

    def construct_attempt(rng, _axes: DominoSceneAxes):
        sample = _sample_longest_chain_scene(
            rng,
            candidate_count=int(count_axes.candidate_axis.value),
            target_answer=int(count_axes.target_axis.value),
        )
        return domino_bbox_set_attempt(
            sample=sample,
            answer_gt=TypedValue(type="integer", value=int(sample.answer_value)),
            execution_extra={
                "target_answer": int(sample.answer_value),
                "longest_chain_tile_ids": [str(tile_id) for tile_id in sample.annotation_tile_ids],
            },
        )

    return DominoObjectivePlan(
        attempt_namespace="games.dominoes.longest_chain",
        prompt_query_key=QUERY_ID,
        query_params=dict(count_axes.query_params),
        prompt_dynamic_slots=prompt_dynamic_slots,
        construct_attempt=construct_attempt,
    )


@register_task
class GamesDominoesLongestChainLengthValueTask:
    """Return the maximum number of loose dominoes addable after the reference tile."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'topology')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params, max_attempts):
        return run_domino_lifecycle(
            task_id=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            prepare_objective=_prepare_longest_chain_objective,
        )


__all__ = ["GamesDominoesLongestChainLengthValueTask"]
