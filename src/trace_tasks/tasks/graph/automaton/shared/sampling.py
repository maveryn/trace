"""Sampling primitives for graph automaton scenes."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import networkx as nx

from .....core.sampling import uniform_choice
from .....core.seed import spawn_rng
from ....shared.config_defaults import group_default
from ...shared.style import SUPPORTED_NODE_COLOR_NAMES
from ...shared.task_support import resolve_graph_named_variant
from .labels import state_labels
from .state import (
    AUTOMATON_SYMBOLS,
    OPTION_LABELS,
    SUPPORTED_AUTOMATON_LAYOUT_VARIANTS,
    AcceptanceAxes,
    AcceptanceSample,
)
from .topology import build_automaton_topology_sample


@dataclass(frozen=True)
class AcceptanceDefaults:
    """Stable fallback defaults for automaton string-acceptance scenes."""

    state_count_min: int = 4
    state_count_max: int = 6
    input_length_min: int = 3
    input_length_max: int = 6
    candidate_count: int = 6
    candidate_count_support: Tuple[int, ...] = (4, 6)
    canvas_width: int = 864
    canvas_height: int = 640
    option_panel_height_px: int = 150
    outer_margin_px: int = 28
    panel_padding_px: int = 24
    panel_corner_radius_px: int = 20
    panel_title_font_size_px: int = 24
    node_shape_variant: str = "circle"
    node_radius_min_px: int = 20
    node_radius_max_px: int = 25
    edge_width_px: int = 4
    arrow_length_px: int = 12
    arrow_width_px: int = 7
    node_border_width_px: int = 2
    label_font_size_px: int = 20
    node_color_name: str = "blue"


def _uniform_probability(values: Sequence[int | str], *, selected: int | str | None = None) -> Dict[str, float]:
    """Return a uniform probability map over support values."""

    support = tuple(str(value) for value in values)
    if not support:
        return {}
    if selected is not None:
        selected_text = str(selected)
        return {str(value): (1.0 if str(value) == selected_text else 0.0) for value in support}
    probability = 1.0 / float(len(support))
    return {str(value): float(probability) for value in support}


def _coerce_int_support(
    raw_value: Any,
    *,
    fallback: Sequence[int],
    min_value: int,
    max_value: int,
    field_name: str,
) -> Tuple[int, ...]:
    """Normalize one integer support list from config/params."""

    if raw_value is None:
        raw_items = tuple(int(value) for value in fallback)
    elif isinstance(raw_value, str):
        raw_items = tuple(int(part.strip()) for part in raw_value.split(",") if part.strip())
    elif isinstance(raw_value, int):
        raw_items = (int(raw_value),)
    else:
        raw_items = tuple(int(value) for value in raw_value)

    support = tuple(sorted(set(int(value) for value in raw_items if int(min_value) <= int(value) <= int(max_value))))
    if not support:
        raise ValueError(f"{field_name} has no feasible values")
    return support


def _resolve_named_variant(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    context_key: str,
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    supported: Tuple[str, ...],
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve one balanced visual axis without public task identity."""

    return resolve_graph_named_variant(
        spawn_rng(int(instance_seed), f"{str(context_key)}.{str(namespace)}"),
        params=params,
        gen_defaults=gen_defaults,
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        balance_flag_key=str(balance_flag_key),
        supported=tuple(str(value) for value in supported),
        instance_seed=int(instance_seed),
        task_id=str(context_key),
        namespace=str(namespace),
    )


def resolve_acceptance_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    defaults: AcceptanceDefaults,
    automaton_kind: str,
) -> AcceptanceAxes:
    """Resolve non-query sampling and visual axes for a DFA or NFA instance."""

    kind = str(automaton_kind)
    if kind not in {"dfa", "nfa"}:
        raise ValueError("automaton_kind must be 'dfa' or 'nfa'")
    context_key = f"graph_automaton_{kind}_acceptance"

    state_count_min = int(params.get("state_count_min", group_default(gen_defaults, "state_count_min", defaults.state_count_min)))
    state_count_max = int(params.get("state_count_max", group_default(gen_defaults, "state_count_max", defaults.state_count_max)))
    state_count_support = tuple(int(value) for value in range(max(3, int(state_count_min)), int(state_count_max) + 1))
    if not state_count_support:
        raise ValueError("no feasible state_count support exists for automaton string acceptance")
    explicit_state_count = params.get("state_count")
    if explicit_state_count is not None:
        state_count = int(explicit_state_count)
        if int(state_count) not in set(state_count_support):
            raise ValueError("state_count is outside feasible support")
    else:
        state_count = int(
            uniform_choice(
                spawn_rng(int(instance_seed), f"{context_key}:state_count"),
                state_count_support,
            )
        )

    input_length_min = int(params.get("input_length_min", group_default(gen_defaults, "input_length_min", defaults.input_length_min)))
    input_length_max = int(params.get("input_length_max", group_default(gen_defaults, "input_length_max", defaults.input_length_max)))
    input_length_support = tuple(int(value) for value in range(max(1, int(input_length_min)), int(input_length_max) + 1))
    if not input_length_support:
        raise ValueError("no feasible input length support exists for automaton string acceptance")
    explicit_input_length = params.get("input_length")
    if explicit_input_length is not None:
        input_length = int(explicit_input_length)
        if int(input_length) not in set(input_length_support):
            raise ValueError("input_length is outside feasible support")
    else:
        input_length = int(
            uniform_choice(
                spawn_rng(int(instance_seed), f"{context_key}:input_length"),
                input_length_support,
            )
        )

    candidate_count_support = _coerce_int_support(
        params.get(
            "candidate_count_support",
            group_default(gen_defaults, "candidate_count_support", defaults.candidate_count_support),
        ),
        fallback=defaults.candidate_count_support,
        min_value=2,
        max_value=len(OPTION_LABELS),
        field_name="candidate_count_support",
    )
    explicit_candidate_count = params.get("candidate_count")
    if explicit_candidate_count is not None:
        candidate_count = int(explicit_candidate_count)
        if int(candidate_count) not in set(candidate_count_support):
            raise ValueError("candidate_count is outside feasible support")
    else:
        candidate_count = int(
            uniform_choice(
                spawn_rng(int(instance_seed), f"{context_key}:candidate_count"),
                candidate_count_support,
            )
        )

    explicit_answer_option = str(params.get("answer_option", params.get("answer_option_label", ""))).strip().upper()
    if explicit_answer_option:
        if explicit_answer_option not in set(OPTION_LABELS[: int(candidate_count)]):
            raise ValueError("answer_option is outside feasible option-label support")
        answer_option_index = int(OPTION_LABELS.index(explicit_answer_option))
    else:
        answer_option_index = int(
            uniform_choice(
                spawn_rng(int(instance_seed), f"{context_key}:answer_option"),
                tuple(range(int(candidate_count))),
            )
        )

    layout_variant, layout_variant_probabilities = _resolve_named_variant(
        int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        context_key=context_key,
        explicit_key="layout_variant",
        weights_key="layout_variant_weights",
        balance_flag_key="balanced_layout_variant_sampling",
        supported=SUPPORTED_AUTOMATON_LAYOUT_VARIANTS,
        namespace="layout_variant",
    )
    layout_transform_variant, layout_transform_variant_probabilities = _resolve_named_variant(
        int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        context_key=context_key,
        explicit_key="layout_transform_variant",
        weights_key="layout_transform_variant_weights",
        balance_flag_key="balanced_layout_transform_variant_sampling",
        supported=("identity", "mirror_left_right", "mirror_up_down"),
        namespace="layout_transform_variant",
    )
    edge_routing_variant, edge_routing_variant_probabilities = _resolve_named_variant(
        int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        context_key=context_key,
        explicit_key="edge_routing_variant",
        weights_key="edge_routing_variant_weights",
        balance_flag_key="balanced_edge_routing_variant_sampling",
        supported=("straight", "mixed_arc"),
        namespace="edge_routing_variant",
    )
    node_color_name, node_color_name_probabilities = _resolve_named_variant(
        int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        context_key=context_key,
        explicit_key="node_color_name",
        weights_key="node_color_name_weights",
        balance_flag_key="balanced_node_color_name_sampling",
        supported=SUPPORTED_NODE_COLOR_NAMES,
        namespace="node_color_name",
    )

    return AcceptanceAxes(
        automaton_kind=kind,
        state_count=int(state_count),
        input_length=int(input_length),
        input_length_min=int(min(input_length_support)),
        input_length_max=int(max(input_length_support)),
        candidate_count=int(candidate_count),
        candidate_count_support=tuple(int(value) for value in candidate_count_support),
        answer_option_index=int(answer_option_index),
        layout_variant=str(layout_variant),
        layout_transform_variant=str(layout_transform_variant),
        edge_routing_variant=str(edge_routing_variant),
        node_color_name=str(node_color_name),
        state_count_probabilities=_uniform_probability(
            state_count_support,
            selected=int(state_count) if explicit_state_count is not None else None,
        ),
        input_length_probabilities=_uniform_probability(
            input_length_support,
            selected=int(input_length) if explicit_input_length is not None else None,
        ),
        candidate_count_probabilities=_uniform_probability(
            candidate_count_support,
            selected=int(candidate_count) if explicit_candidate_count is not None else None,
        ),
        answer_option_probabilities=_uniform_probability(
            OPTION_LABELS[: int(candidate_count)],
            selected=str(OPTION_LABELS[int(answer_option_index)]) if explicit_answer_option else None,
        ),
        layout_variant_probabilities=dict(layout_variant_probabilities),
        layout_transform_variant_probabilities=dict(layout_transform_variant_probabilities),
        edge_routing_variant_probabilities=dict(edge_routing_variant_probabilities),
        node_color_name_probabilities=dict(node_color_name_probabilities),
    )


def _all_binary_strings(*, min_len: int, max_len: int) -> Tuple[str, ...]:
    """Return all binary strings in the requested length interval."""

    values: list[str] = []
    for length in range(int(min_len), int(max_len) + 1):
        for number in range(2 ** int(length)):
            values.append(format(int(number), f"0{int(length)}b"))
    return tuple(values)


def _sample_transition_function(
    rng: random.Random,
    *,
    state_count: int,
    automaton_kind: str,
) -> Dict[int, Dict[str, Tuple[int, ...]]]:
    """Sample DFA or NFA transitions over the binary alphabet."""

    transitions: Dict[int, Dict[str, Tuple[int, ...]]] = {int(state): {} for state in range(int(state_count))}
    has_nondeterministic_branch = False
    for state in range(int(state_count)):
        for symbol in AUTOMATON_SYMBOLS:
            candidates = [int(value) for value in range(int(state_count)) if int(value) != int(state)]
            first_target = int(rng.choice(candidates))
            targets = {int(first_target)}
            if str(automaton_kind) == "nfa" and rng.random() < 0.35:
                extra_candidates = [int(value) for value in candidates if int(value) not in targets]
                if extra_candidates:
                    targets.add(int(rng.choice(extra_candidates)))
                    has_nondeterministic_branch = True
            transitions[int(state)][str(symbol)] = tuple(sorted(int(value) for value in targets))

    if str(automaton_kind) == "nfa" and not has_nondeterministic_branch:
        state = int(rng.randrange(int(state_count)))
        symbol = str(rng.choice(AUTOMATON_SYMBOLS))
        existing = set(int(value) for value in transitions[int(state)][str(symbol)])
        extra_candidates = [
            int(value)
            for value in range(int(state_count))
            if int(value) != int(state) and int(value) not in existing
        ]
        if extra_candidates:
            existing.add(int(rng.choice(extra_candidates)))
            transitions[int(state)][str(symbol)] = tuple(sorted(existing))
    return dict(transitions)


def accepting_paths(
    *,
    transitions: Mapping[int, Mapping[str, Sequence[int]]],
    input_string: str,
    accepting_states: set[int],
    max_paths: int = 256,
) -> Tuple[Tuple[int, ...], ...]:
    """Return accepting state paths for one candidate input."""

    paths: list[Tuple[int, ...]] = [(0,)]
    for symbol in str(input_string):
        next_paths: list[Tuple[int, ...]] = []
        for path in paths:
            source = int(path[-1])
            for target in transitions.get(int(source), {}).get(str(symbol), ()):
                next_paths.append(tuple([*path, int(target)]))
                if len(next_paths) >= int(max_paths):
                    break
            if len(next_paths) >= int(max_paths):
                break
        paths = list(next_paths)
        if not paths:
            break
    return tuple(path for path in paths if int(path[-1]) in set(int(value) for value in accepting_states))


def _transition_label_map(
    *,
    transitions: Mapping[int, Mapping[str, Sequence[int]]],
    labels: Sequence[str],
) -> Tuple[Dict[Tuple[str, str], str], Dict[str, Dict[str, Tuple[str, ...]]], nx.DiGraph]:
    """Convert symbolic transitions into renderable edge labels and a graph."""

    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(tuple(labels))))
    symbols_by_edge: Dict[Tuple[int, int], list[str]] = {}
    transition_function: Dict[str, Dict[str, Tuple[str, ...]]] = {str(label): {} for label in labels}
    for source, per_symbol in sorted(transitions.items()):
        source_label = str(labels[int(source)])
        for symbol, targets in sorted(per_symbol.items()):
            target_labels: list[str] = []
            for target in targets:
                graph.add_edge(int(source), int(target))
                symbols_by_edge.setdefault((int(source), int(target)), []).append(str(symbol))
                target_labels.append(str(labels[int(target)]))
            transition_function[str(source_label)][str(symbol)] = tuple(str(label) for label in target_labels)

    transition_labels_by_edge: Dict[Tuple[str, str], str] = {}
    for (source, target), symbols in sorted(symbols_by_edge.items()):
        edge_label = ",".join(sorted(set(str(symbol) for symbol in symbols)))
        transition_labels_by_edge[(str(labels[int(source)]), str(labels[int(target)]))] = str(edge_label)
    return (
        dict(transition_labels_by_edge),
        {str(key): {str(symbol): tuple(str(target) for target in targets) for symbol, targets in value.items()} for key, value in transition_function.items()},
        graph,
    )


def sample_acceptance_automaton(
    rng: random.Random,
    *,
    axes: AcceptanceAxes,
) -> AcceptanceSample:
    """Construct one automaton with exactly one accepted displayed candidate."""

    labels = state_labels(int(axes.state_count))
    option_labels = OPTION_LABELS[: int(axes.candidate_count)]
    all_strings = list(_all_binary_strings(min_len=int(axes.input_length_min), max_len=int(axes.input_length_max)))
    for _ in range(2000):
        transitions = _sample_transition_function(
            rng,
            state_count=int(axes.state_count),
            automaton_kind=str(axes.automaton_kind),
        )
        accepting_count = int(rng.randint(1, max(1, min(2, int(axes.state_count) - 1))))
        accepting_states = set(int(value) for value in rng.sample(range(1, int(axes.state_count)), k=int(accepting_count)))
        rng.shuffle(all_strings)
        accepted: list[Tuple[str, Tuple[int, ...]]] = []
        rejected: list[str] = []
        for candidate in all_strings:
            candidate_accepting_paths = accepting_paths(
                transitions=transitions,
                input_string=str(candidate),
                accepting_states=set(accepting_states),
            )
            if candidate_accepting_paths:
                if len(str(candidate)) == int(axes.input_length):
                    accepted.append((str(candidate), tuple(int(value) for value in candidate_accepting_paths[0])))
            else:
                rejected.append(str(candidate))
        if not accepted or len(rejected) < int(axes.candidate_count) - 1:
            continue
        answer_string, accepting_path = rng.choice(accepted)
        distractor_pool = [str(value) for value in rejected if str(value) != str(answer_string)]
        same_length = [str(value) for value in distractor_pool if len(str(value)) == len(str(answer_string))]
        other_lengths = [str(value) for value in distractor_pool if len(str(value)) != len(str(answer_string))]
        selected_distractors: list[str] = []
        rng.shuffle(same_length)
        rng.shuffle(other_lengths)
        for value in [*same_length, *other_lengths]:
            if len(selected_distractors) >= int(axes.candidate_count) - 1:
                break
            if str(value) not in selected_distractors:
                selected_distractors.append(str(value))
        if len(selected_distractors) < int(axes.candidate_count) - 1:
            continue

        candidate_strings_by_option: Dict[str, str] = {}
        distractor_iter = iter(selected_distractors)
        answer_label = str(option_labels[int(axes.answer_option_index)])
        for index, option_label in enumerate(option_labels):
            if int(index) == int(axes.answer_option_index):
                candidate_strings_by_option[str(option_label)] = str(answer_string)
            else:
                candidate_strings_by_option[str(option_label)] = str(next(distractor_iter))
        accepted_option_labels = tuple(
            str(option_label)
            for option_label, candidate in candidate_strings_by_option.items()
            if accepting_paths(
                transitions=transitions,
                input_string=str(candidate),
                accepting_states=set(accepting_states),
            )
        )
        if accepted_option_labels != (answer_label,):
            continue

        transition_labels_by_edge, transition_function, graph = _transition_label_map(
            transitions=transitions,
            labels=labels,
        )
        graph_sample = build_automaton_topology_sample(
            graph=graph,
            labels=labels,
            transition_labels_by_edge=transition_labels_by_edge,
        )
        accepting_labels = tuple(str(labels[int(index)]) for index in sorted(int(value) for value in accepting_states))
        accepting_path_labels = tuple(str(labels[int(index)]) for index in accepting_path)
        return AcceptanceSample(
            graph_sample=graph_sample,
            start_label=str(labels[0]),
            accepting_labels=tuple(str(label) for label in accepting_labels),
            answer_option_label=str(answer_label),
            answer_input_string=str(answer_string),
            accepting_path_labels=tuple(str(label) for label in accepting_path_labels),
            candidate_strings_by_option={str(key): str(value) for key, value in candidate_strings_by_option.items()},
            accepted_option_labels=tuple(str(value) for value in accepted_option_labels),
            transition_labels_by_edge={(str(left), str(right)): str(value) for (left, right), value in transition_labels_by_edge.items()},
            transition_function={str(key): {str(symbol): tuple(str(target) for target in targets) for symbol, targets in value.items()} for key, value in transition_function.items()},
        )
    raise ValueError("failed to sample automaton with one accepted displayed string")


__all__ = [
    "AcceptanceDefaults",
    "accepting_paths",
    "resolve_acceptance_axes",
    "sample_acceptance_automaton",
]
