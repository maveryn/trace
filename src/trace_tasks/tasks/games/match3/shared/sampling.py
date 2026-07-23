"""Neutral sampling helpers for match-3 board, style, and option construction."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.games.shared.sampling import resolve_games_named_axis
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice

from .defaults import DEFAULTS, GEM_KEYS, OPTION_LABELS, SUPPORTED_SCENE_VARIANTS, SUPPORTED_STYLE_VARIANTS
from .rules import all_move_outcomes, find_runs, generate_board
from .state import Match3BoardSpec, Match3IntegerAxis, Match3SceneAxes, MoveOutcome, SwapOption


def resolve_match3_integer_axis(
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[int],
    namespace: str,
    balanced_flag_key: str,
) -> Match3IntegerAxis:
    """Resolve one integer scene/task axis from params, config, or fallback support."""

    value, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(support_key),
        explicit_key=str(explicit_key),
        fallback_support=tuple(int(item) for item in fallback_support),
        namespace=str(namespace),
        balanced_flag_key=str(balanced_flag_key),
        namespace_support_permutation=True,
    )
    support = group_default(gen_defaults, str(support_key), tuple(int(item) for item in fallback_support))
    return Match3IntegerAxis(
        value=int(value),
        support=tuple(int(item) for item in support),
        probabilities=dict(probabilities),
    )


def resolve_match3_scene_axes(
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
) -> Match3SceneAxes:
    """Sample visual scene and style variants without task-objective branching."""

    scene_variant, scene_probs = resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=f"{namespace}.scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported_variants=SUPPORTED_SCENE_VARIANTS,
    )
    style_variant, style_probs = resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=f"{namespace}.style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported_variants=SUPPORTED_STYLE_VARIANTS,
    )
    return Match3SceneAxes(
        scene_variant=str(scene_variant),
        scene_variant_probabilities=dict(scene_probs),
        style_variant=str(style_variant),
        style_variant_probabilities=dict(style_probs),
    )


def resolve_board_dimensions(
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    scene_variant: str,
    namespace: str,
) -> tuple[int, int, dict[str, float], dict[str, float]]:
    """Resolve row/column counts while enforcing the selected board shape variant."""

    row_axis = resolve_match3_integer_axis(
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        params=params,
        support_key="row_count_support",
        explicit_key="row_count",
        fallback_support=DEFAULTS.row_count_support,
        namespace=f"{namespace}.row_count",
        balanced_flag_key="balanced_row_count_sampling",
    )
    col_axis = resolve_match3_integer_axis(
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        params=params,
        support_key="col_count_support",
        explicit_key="col_count",
        fallback_support=DEFAULTS.col_count_support,
        namespace=f"{namespace}.col_count",
        balanced_flag_key="balanced_col_count_sampling",
    )
    rows = int(row_axis.value)
    cols = int(col_axis.value)
    if str(scene_variant) == "wide_board":
        cols = max(int(cols), int(rows))
    elif str(scene_variant) == "tall_board":
        rows = max(int(rows), int(cols))
    else:
        cols = rows
    return int(rows), int(cols), dict(row_axis.probabilities), dict(col_axis.probabilities)


def sample_gem_keys(rng: Any, *, gem_type_count: int) -> Tuple[str, ...]:
    """Sample the canonical named-color keys used by one match-3 board."""

    values = list(GEM_KEYS)
    rng.shuffle(values)
    return tuple(values[: int(gem_type_count)])


def make_base_board(
    rng: Any,
    *,
    gen_defaults: Mapping[str, Any],
    namespace: str,
    instance_seed: int,
    params: Mapping[str, Any],
    scene_variant: str,
) -> Match3BoardSpec:
    """Construct a run-free board and record all neutral generation axes."""

    rows, cols, row_probs, col_probs = resolve_board_dimensions(
        gen_defaults=gen_defaults,
        namespace=str(namespace),
        instance_seed=int(instance_seed),
        params=params,
        scene_variant=str(scene_variant),
    )
    gem_axis = resolve_match3_integer_axis(
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        params=params,
        support_key="gem_type_count_support",
        explicit_key="gem_type_count",
        fallback_support=DEFAULTS.gem_type_count_support,
        namespace=f"{namespace}.gem_type_count",
        balanced_flag_key="balanced_gem_type_count_sampling",
    )
    gem_keys = sample_gem_keys(rng, gem_type_count=int(gem_axis.value))
    board = generate_board(rng, rows=int(rows), cols=int(cols), gem_keys=gem_keys)
    if find_runs(board):
        raise ValueError("generated board contains a pre-existing run")
    return Match3BoardSpec(
        board=board,
        gem_keys=tuple(gem_keys),
        rows=int(rows),
        cols=int(cols),
        metadata={
            "rows": int(rows),
            "cols": int(cols),
            "gem_type_count": int(gem_axis.value),
            "gem_keys": [str(key) for key in gem_keys],
            "row_count_probabilities": dict(row_probs),
            "col_count_probabilities": dict(col_probs),
            "gem_type_count_probabilities": dict(gem_axis.probabilities),
        },
    )


def resolve_option_count_axis(
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
) -> Match3IntegerAxis:
    """Resolve how many visible swap-arrow options to draw."""

    return resolve_match3_integer_axis(
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        params=params,
        support_key="option_count_support",
        explicit_key="option_count",
        fallback_support=DEFAULTS.option_count_support,
        namespace=f"{namespace}.option_count",
        balanced_flag_key="balanced_option_count_sampling",
    )


def answer_option_index(instance_seed: int, *, params: Mapping[str, Any], option_count: int, namespace: str) -> int:
    """Resolve which visible option slot should contain the correct swap."""

    explicit = params.get("answer_option_index")
    if explicit is not None:
        value = int(explicit)
        if 0 <= value < int(option_count):
            return int(value)
        raise ValueError("answer_option_index outside option_count")
    rng = spawn_rng(int(instance_seed), f"{str(namespace)}.answer_option_index")
    return int(uniform_choice(rng, tuple(range(int(option_count)))))


def select_random_outcomes(outcomes: Sequence[MoveOutcome], rng: Any, *, count: int) -> Tuple[MoveOutcome, ...]:
    """Pick a random subset of move outcomes without imposing objective semantics."""

    values = list(outcomes)
    rng.shuffle(values)
    return tuple(values[: int(count)])


def labeled_swap_options(
    *,
    option_count: int,
    answer_index: int,
    answer_outcome: MoveOutcome,
    distractor_outcomes: Sequence[MoveOutcome],
) -> Tuple[SwapOption, ...]:
    """Bind labels to one answer outcome and neutral distractor outcomes."""

    labels = OPTION_LABELS[: int(option_count)]
    option_outcomes = list(distractor_outcomes)
    option_outcomes.insert(int(answer_index), answer_outcome)
    return tuple(
        SwapOption(label=str(label), outcome=outcome, is_answer=(int(index) == int(answer_index)))
        for index, (label, outcome) in enumerate(zip(labels, option_outcomes))
    )


def all_outcomes_for_board(board) -> Tuple[MoveOutcome, ...]:
    """Expose all move outcomes from rules.py through the sampling role layer."""

    return tuple(all_move_outcomes(board))


def outcome_histograms(outcomes: Sequence[MoveOutcome]) -> dict[str, dict[str, int]]:
    """Summarize candidate clear/run counts for trace metadata."""

    clear_hist: Dict[str, int] = {}
    run_hist: Dict[str, int] = {}
    for outcome in outcomes:
        clear_key = str(int(outcome.clear_count))
        run_key = str(int(outcome.run_count))
        clear_hist[clear_key] = int(clear_hist.get(clear_key, 0) + 1)
        run_hist[run_key] = int(run_hist.get(run_key, 0) + 1)
    return {
        "clear_count_histogram": dict(sorted(clear_hist.items(), key=lambda item: int(item[0]))),
        "run_count_histogram": dict(sorted(run_hist.items(), key=lambda item: int(item[0]))),
    }


__all__ = [
    "all_outcomes_for_board",
    "answer_option_index",
    "labeled_swap_options",
    "make_base_board",
    "outcome_histograms",
    "resolve_board_dimensions",
    "resolve_match3_integer_axis",
    "resolve_match3_scene_axes",
    "resolve_option_count_axis",
    "sample_gem_keys",
    "select_random_outcomes",
]
