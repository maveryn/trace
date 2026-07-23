"""Identify the one invalid labeled join in a domino chain."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant

from ._lifecycle import DominoAttemptResult, DominoObjectivePlan, run_domino_lifecycle
from .shared.annotations import domino_join_segment_annotation
from .shared.defaults import DEFAULTS, DOMINOES_NAMESPACE, SCENE_ID
from .shared.prompts import domino_option_label_segment_json_examples, domino_output_slots
from .shared.rules import OPTION_LABELS, PIP_VALUES, canonical_tile
from .shared.sampling import build_sampled_scene, build_tile_instance
from .shared.state import DominoSceneAxes


TASK_ID = "task_games__dominoes__invalid_join_label"
QUERY_ID = "invalid_join_label"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
JOIN_LABELS = tuple(OPTION_LABELS[:6])
CHAIN_TILE_COUNT = 7
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _resolve_target_join_label(
    *,
    instance_seed: int,
    task_params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> tuple[str, Mapping[str, float]]:
    """Resolve the target invalid join label from the six visible join labels."""

    raw_support = task_params.get(
        "invalid_join_label_support",
        gen_defaults.get("invalid_join_label_support", DEFAULTS.invalid_join_label_support),
    )
    support = tuple(str(label) for label in raw_support)
    if not support or any(label not in JOIN_LABELS for label in support):
        raise ValueError("invalid_join_label_support must be a non-empty subset of A..F")
    selected, probabilities = resolve_variant(
        spawn_rng(int(instance_seed), f"{DOMINOES_NAMESPACE}.invalid_join.target_label"),
        params=task_params,
        gen_defaults=gen_defaults,
        supported_variants=support,
        explicit_key="target_label",
        weights_key="invalid_join_label_weights",
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
        selected_variant=str(selected),
        variant_probabilities=dict(probabilities),
        supported_variants=support,
        balance_flag_key="balanced_target_answer_sampling",
        explicit_key="target_label",
        weights_key="invalid_join_label_weights",
        sampling_namespace=f"{DOMINOES_NAMESPACE}.invalid_join.target_label",
    )
    return str(selected), dict(probabilities)


def _sample_oriented_invalid_join_chain(rng, *, invalid_join_index: int) -> Tuple[Tuple[int, int], ...]:
    """Sample seven unique dominoes with exactly one invalid adjacent join."""

    target_index = int(invalid_join_index)
    for _ in range(800):
        oriented_tiles: list[Tuple[int, int]] = []
        used_canonicals: set[Tuple[int, int]] = set()
        previous_right: int | None = None
        failed = False
        for tile_index in range(CHAIN_TILE_COUNT):
            left_options = list(PIP_VALUES)
            if previous_right is not None:
                if int(tile_index - 1) == int(target_index):
                    left_options = [value for value in PIP_VALUES if int(value) != int(previous_right)]
                else:
                    left_options = [int(previous_right)]
            rng.shuffle(left_options)

            chosen_tile: Tuple[int, int] | None = None
            for left_value in left_options:
                right_options = list(PIP_VALUES)
                rng.shuffle(right_options)
                for right_value in right_options:
                    canonical = canonical_tile(int(left_value), int(right_value))
                    if canonical in used_canonicals:
                        continue
                    chosen_tile = (int(left_value), int(right_value))
                    break
                if chosen_tile is not None:
                    break
            if chosen_tile is None:
                failed = True
                break
            oriented_tiles.append(chosen_tile)
            used_canonicals.add(canonical_tile(int(chosen_tile[0]), int(chosen_tile[1])))
            previous_right = int(chosen_tile[1])

        if failed or len(oriented_tiles) != CHAIN_TILE_COUNT:
            continue
        invalid_indices = [
            index
            for index in range(CHAIN_TILE_COUNT - 1)
            if int(oriented_tiles[index][1]) != int(oriented_tiles[index + 1][0])
        ]
        if invalid_indices == [int(target_index)]:
            return tuple(oriented_tiles)
    raise ValueError("unable to sample domino chain with unique invalid join")


def _sample_invalid_join_scene(rng, *, answer_label: str):
    """Build one labeled chain scene where exactly one join breaks the rule."""

    label = str(answer_label)
    invalid_join_index = JOIN_LABELS.index(label)
    oriented_chain = _sample_oriented_invalid_join_chain(rng, invalid_join_index=int(invalid_join_index))
    chain_instances = tuple(
        build_tile_instance(
            tile_id=f"chain_{index + 1:02d}",
            oriented_tile=oriented_tile,
            role="chain",
            right_join_label=JOIN_LABELS[index] if index < len(JOIN_LABELS) else None,
        )
        for index, oriented_tile in enumerate(oriented_chain)
    )
    return build_sampled_scene(
        chain_instances=chain_instances,
        candidate_instances=(),
        annotation_tile_ids=(),
        answer_value=str(label),
        reference_tile_id=None,
    )


def _prepare_invalid_join_objective(
    instance_seed,
    task_params,
    _query_id,
    _query_probabilities,
    _axes: DominoSceneAxes,
):
    """Bind the labeled invalid-join objective for one generated instance."""

    answer_label, label_probabilities = _resolve_target_join_label(
        instance_seed=int(instance_seed),
        task_params=task_params,
        gen_defaults=_GEN_DEFAULTS,
    )
    json_example, json_example_answer_only = domino_option_label_segment_json_examples(answer_value="C")
    prompt_dynamic_slots = domino_output_slots(
        prompt_query_key=QUERY_ID,
        json_example=json_example,
        json_example_answer_only=json_example_answer_only,
    )

    def construct_attempt(rng, _axes: DominoSceneAxes):
        sample = _sample_invalid_join_scene(rng, answer_label=str(answer_label))
        left_tile_id = f"chain_{JOIN_LABELS.index(str(answer_label)) + 1:02d}"
        right_tile_id = f"chain_{JOIN_LABELS.index(str(answer_label)) + 2:02d}"
        return DominoAttemptResult(
            sample=sample,
            answer_gt=TypedValue(type="option_letter", value=str(answer_label)),
            annotation_entity_ids=(f"join_{str(answer_label)}",),
            build_annotation=lambda rendered_context: domino_join_segment_annotation(
                rendered_context,
                str(answer_label),
            ),
            execution_extra={
                "answer_option_label": str(answer_label),
                "invalid_join_label": str(answer_label),
                "invalid_join_index": int(JOIN_LABELS.index(str(answer_label))),
                "invalid_join_tile_ids": [str(left_tile_id), str(right_tile_id)],
                "option_labels": list(JOIN_LABELS),
                "target_label_probabilities": dict(label_probabilities),
            },
        )

    return DominoObjectivePlan(
        attempt_namespace="games.dominoes.invalid_join",
        prompt_query_key=QUERY_ID,
        query_params={
            "target_label": str(answer_label),
            "option_labels": list(JOIN_LABELS),
            "target_label_probabilities": dict(label_probabilities),
        },
        prompt_dynamic_slots=prompt_dynamic_slots,
        construct_attempt=construct_attempt,
    )


@register_task
class GamesDominoesInvalidJoinLabelTask:
    """Return the label of the one adjacent domino join whose touching halves do not match."""

    task_id = TASK_ID
    reasoning_operations = ('matching',)
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
            prepare_objective=_prepare_invalid_join_objective,
        )


__all__ = ["GamesDominoesInvalidJoinLabelTask"]
