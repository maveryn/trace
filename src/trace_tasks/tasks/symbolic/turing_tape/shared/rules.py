"""Sampling and simulation rules for symbolic Turing tape scenes."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .....core.seed import spawn_rng
from .....core.sampling import uniform_choice
from ...shared.common import get_int_range

from .state import TURING_MOVES, TURING_SYMBOLS, TuringDataset, TuringTrace, TuringTransition


def move_delta(move: str) -> int:
    """Return the head-position delta for a Turing move."""

    return -1 if str(move) == "L" else 1


def _sample_head_path(
    rng,
    *,
    tape_length: int,
    steps: int,
) -> Tuple[int, Tuple[str, ...], Tuple[int, ...]]:
    for _attempt in range(200):
        head = int(rng.randrange(1, max(2, int(tape_length) - 1)))
        positions: List[int] = []
        moves: List[str] = []
        for _step in range(int(steps)):
            allowed: List[str] = []
            if head > 0:
                allowed.append("L")
            if head < int(tape_length) - 1:
                allowed.append("R")
            move = str(rng.choice(allowed or list(TURING_MOVES)))
            positions.append(int(head))
            moves.append(move)
            head += move_delta(move)
        if len(set(positions)) >= min(3, int(steps)):
            return int(positions[0]), tuple(moves), tuple(positions)
    start = int(max(1, min(int(tape_length) - 2, int(tape_length) // 2)))
    positions = []
    moves = []
    head = start
    direction = 1
    for _step in range(int(steps)):
        positions.append(int(head))
        if head >= int(tape_length) - 2:
            direction = -1
        elif head <= 1:
            direction = 1
        move = "R" if direction > 0 else "L"
        moves.append(move)
        head += move_delta(move)
    return int(start), tuple(moves), tuple(positions)


def transition_key(transition: TuringTransition) -> Tuple[str, str]:
    """Return the state/read-symbol lookup key for one transition."""

    return (str(transition.state), str(transition.read_symbol))


def simulate_turing(
    *,
    initial_tape: Sequence[str],
    start_state: str,
    start_head: int,
    transitions: Sequence[TuringTransition],
    steps: int,
) -> Tuple[Tuple[str, ...], Tuple[TuringTrace, ...]]:
    """Execute the transition table for a fixed number of steps."""

    table = {transition_key(transition): transition for transition in transitions}
    tape = [str(symbol) for symbol in initial_tape]
    state = str(start_state)
    head = int(start_head)
    traces: List[TuringTrace] = []
    for step in range(1, int(steps) + 1):
        read_symbol = str(tape[head])
        transition = table[(state, read_symbol)]
        tape[head] = str(transition.write_symbol)
        traces.append(
            TuringTrace(
                step=int(step),
                state=str(state),
                head_position=int(head),
                read_symbol=str(read_symbol),
                write_symbol=str(transition.write_symbol),
                move=str(transition.move),
                next_state=str(transition.next_state),
            )
        )
        head = max(0, min(len(tape) - 1, int(head + move_delta(str(transition.move)))))
        state = str(transition.next_state)
    return tuple(tape), tuple(traces)


def final_head_position_after_steps(
    *,
    start_head: int,
    tape_length: int,
    traces: Sequence[TuringTrace],
) -> int:
    """Return the zero-based head position after replaying executed moves."""

    head = int(start_head)
    max_index = max(0, int(tape_length) - 1)
    for trace in traces:
        head = max(0, min(max_index, int(trace.head_position) + move_delta(str(trace.move))))
    return int(head)


def build_turing_dataset(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    sampling_namespace: str,
    desired_answer_namespace: str,
    query_symbol_namespace: str,
) -> TuringDataset:
    """Sample a Turing tape instance with a controlled final symbol count."""

    rng = spawn_rng(int(instance_seed), str(sampling_namespace))
    tape_min, tape_max = get_int_range(params, gen_defaults, min_key="turing_tape_length_min", max_key="turing_tape_length_max", fallback_min=8, fallback_max=11)
    steps_min, steps_max = get_int_range(params, gen_defaults, min_key="turing_steps_min", max_key="turing_steps_max", fallback_min=3, fallback_max=6)
    symbol_min, symbol_max = get_int_range(params, gen_defaults, min_key="turing_symbol_count_min", max_key="turing_symbol_count_max", fallback_min=2, fallback_max=2)
    answer_min, answer_max = get_int_range(params, gen_defaults, min_key="turing_answer_min", max_key="turing_answer_max", fallback_min=1, fallback_max=7)
    symbol_count = int(max(2, min(len(TURING_SYMBOLS), rng.randint(symbol_min, symbol_max))))
    symbols = tuple(TURING_SYMBOLS[:symbol_count])
    tape_length = int(rng.randint(tape_min, tape_max))
    max_answer = int(min(answer_max, tape_length - 1))
    answer_support = list(range(int(answer_min), int(max_answer) + 1))
    if not answer_support:
        answer_support = [max(0, min(tape_length, int(answer_min)))]
    desired_rng = spawn_rng(int(instance_seed), str(desired_answer_namespace))
    desired_answer = int(uniform_choice(desired_rng, tuple(answer_support), sort_keys=True))
    symbol_rng = spawn_rng(int(instance_seed), str(query_symbol_namespace))
    query_symbol = str(uniform_choice(symbol_rng, symbols, sort_keys=False))
    steps = int(rng.randint(steps_min, steps_max))
    states = tuple(f"S{index}" for index in range(int(steps)))
    start_state = states[0]
    start_head, planned_moves, planned_positions = _sample_head_path(rng, tape_length=tape_length, steps=steps)

    final_tape: List[str] = [str(rng.choice([symbol for symbol in symbols if symbol != query_symbol])) for _ in range(tape_length)]
    query_positions = list(range(tape_length))
    rng.shuffle(query_positions)
    for pos in query_positions[:desired_answer]:
        final_tape[int(pos)] = str(query_symbol)

    initial_tape = list(final_tape)
    first_visit_positions = []
    seen_positions: set[int] = set()
    for pos in planned_positions:
        if int(pos) not in seen_positions:
            first_visit_positions.append(int(pos))
            seen_positions.add(int(pos))
    for pos in first_visit_positions:
        if rng.random() < 0.65:
            alternatives = [symbol for symbol in symbols if str(symbol) != str(final_tape[pos])]
            initial_tape[pos] = str(rng.choice(alternatives))
    if first_visit_positions and all(str(initial_tape[pos]) == str(final_tape[pos]) for pos in first_visit_positions):
        pos = int(first_visit_positions[0])
        alternatives = [symbol for symbol in symbols if str(symbol) != str(final_tape[pos])]
        initial_tape[pos] = str(alternatives[0])

    transitions_by_key: Dict[Tuple[str, str], TuringTransition] = {}
    tape = list(initial_tape)
    for step_index, (position, move) in enumerate(zip(planned_positions, planned_moves)):
        state = str(states[step_index])
        read_symbol = str(tape[int(position)])
        write_symbol = str(final_tape[int(position)])
        next_state = str(states[step_index + 1]) if step_index + 1 < len(states) else str(states[0])
        transition = TuringTransition(
            state=state,
            read_symbol=read_symbol,
            write_symbol=write_symbol,
            move=str(move),
            next_state=next_state,
        )
        transitions_by_key[(state, read_symbol)] = transition
        tape[int(position)] = write_symbol

    for state in states:
        for symbol in symbols:
            key = (str(state), str(symbol))
            if key in transitions_by_key:
                continue
            transitions_by_key[key] = TuringTransition(
                state=str(state),
                read_symbol=str(symbol),
                write_symbol=str(rng.choice(symbols)),
                move=str(rng.choice(TURING_MOVES)),
                next_state=str(rng.choice(states)),
            )

    transitions = tuple(
        transitions_by_key[(str(state), str(symbol))]
        for state in states
        for symbol in symbols
    )
    simulated_final_tape, traces = simulate_turing(
        initial_tape=initial_tape,
        start_state=start_state,
        start_head=start_head,
        transitions=transitions,
        steps=steps,
    )
    answer_count = int(sum(1 for symbol in simulated_final_tape if str(symbol) == str(query_symbol)))
    if answer_count != desired_answer:
        raise RuntimeError(f"constructed Turing tape answer mismatch: {answer_count} != {desired_answer}")
    return TuringDataset(
        tape_length=int(tape_length),
        symbol_count=int(symbol_count),
        symbols=tuple(symbols),
        query_symbol=str(query_symbol),
        steps=int(steps),
        states=tuple(states),
        start_state=str(start_state),
        start_head=int(start_head),
        initial_tape=tuple(str(symbol) for symbol in initial_tape),
        final_tape=tuple(str(symbol) for symbol in simulated_final_tape),
        transitions=transitions,
        traces=traces,
        answer_count=int(answer_count),
    )
