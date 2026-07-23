"""Rubik cube-net symbolic constructors and answer-option assembly."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.named_colors import (
    available_named_colors,
    sample_named_color_palette,
)

from .rules import (
    apply_move,
    apply_sequence,
    face_color_count,
    format_sequence,
    invert_sequence,
    make_solved_state,
    sample_move,
    sample_move_sequence,
    state_signature,
    sticker_id,
)
from .state import (
    FACE_DISPLAY_NAMES,
    FACE_ORDER,
    OPTION_LABELS,
    StickerKey,
)


def _as_range(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    *,
    min_key: str,
    max_key: str,
    fallback_min: int,
    fallback_max: int,
) -> tuple[int, int]:
    low = int(
        params.get(
            str(min_key), group_default(gen_defaults, str(min_key), int(fallback_min))
        )
    )
    high = int(
        params.get(
            str(max_key), group_default(gen_defaults, str(max_key), int(fallback_max))
        )
    )
    if int(low) > int(high):
        raise ValueError(f"{min_key} must be <= {max_key}")
    return int(low), int(high)


def _sample_base_state(
    *,
    rng,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> tuple[dict[StickerKey, str], dict[str, dict[str, Any]], list[str], dict[str, str]]:
    """Sample the scrambled cube state and palette shared by all objectives."""

    color_count = int(
        params.get(
            "rubiks_palette_size",
            group_default(gen_defaults, "rubiks_palette_size", 6),
        )
    )
    color_count = max(6, min(6, int(color_count)))
    palette = sample_named_color_palette(rng, palette_size=int(color_count))
    if len(palette) < 6:
        palette = list(available_named_colors())[:6]
    palette_entries = [
        {
            "color_name": str(name),
            "color_rgb": [int(channel) for channel in rgb],
        }
        for name, rgb in palette[:6]
    ]
    color_names = [str(item["color_name"]) for item in palette_entries]
    shuffled = list(color_names)
    rng.shuffle(shuffled)
    face_color_names = {
        str(face): str(shuffled[index]) for index, face in enumerate(FACE_ORDER)
    }
    solved = make_solved_state(face_color_names)
    scramble_min, scramble_max = _as_range(
        params,
        gen_defaults,
        min_key="scramble_move_count_min",
        max_key="scramble_move_count_max",
        fallback_min=4,
        fallback_max=8,
    )
    scramble_len = int(rng.randint(int(scramble_min), int(scramble_max)))
    scramble_sequence = sample_move_sequence(rng, length=int(scramble_len))
    state = apply_sequence(solved, scramble_sequence)
    color_map = {
        str(item["color_name"]): {
            "color_name": str(item["color_name"]),
            "color_rgb": [int(channel) for channel in item["color_rgb"]],
        }
        for item in palette_entries
    }
    return state, color_map, scramble_sequence, face_color_names


def _target_key(rng) -> StickerKey:
    face = str(FACE_ORDER[int(rng.randrange(len(FACE_ORDER)))])
    row = int(rng.randrange(3))
    col = int(rng.randrange(3))
    return (str(face), int(row), int(col))


def _make_option_specs(
    *,
    rng,
    labels: Sequence[str],
    answer_value: Any,
    distractor_values: Sequence[Any],
    answer_label_index: int | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Place the true value at a resolved option slot and fill unique distractors."""

    values: list[Any] = []
    for value in distractor_values:
        if value != answer_value and value not in values:
            values.append(value)
        if len(values) >= len(labels):
            break
    if answer_label_index is None:
        values = [answer_value, *values]
        values = values[: len(labels)]
        rng.shuffle(values)
    else:
        resolved_index = int(answer_label_index)
        if not 0 <= resolved_index < len(labels):
            raise ValueError("answer option index is outside the option range")
        distractor_iter = iter(values)
        arranged: list[Any] = []
        for index in range(len(labels)):
            if int(index) == int(resolved_index):
                arranged.append(answer_value)
            else:
                arranged.append(next(distractor_iter))
        values = arranged
    specs: list[dict[str, Any]] = []
    answer_label = ""
    for index, value in enumerate(values):
        label = str(labels[int(index)])
        if value == answer_value:
            answer_label = str(label)
        specs.append(
            {
                "option_id": f"option_{label}",
                "option_label": str(label),
                "value": value,
                "is_correct": bool(value == answer_value),
            }
        )
    if not answer_label:
        raise RuntimeError("unable to construct Rubik option labels")
    return specs, str(answer_label)


def _color_option_specs(
    *,
    rng,
    color_map: Mapping[str, Mapping[str, Any]],
    answer_color_name: str,
    option_count: int,
    answer_label_index: int,
) -> tuple[list[dict[str, Any]], str]:
    labels = OPTION_LABELS[: int(option_count)]
    color_names = [str(name) for name in color_map]
    distractors = [
        str(name) for name in color_names if str(name) != str(answer_color_name)
    ]
    rng.shuffle(distractors)
    specs, answer_label = _make_option_specs(
        rng=rng,
        labels=labels,
        answer_value=str(answer_color_name),
        distractor_values=distractors,
        answer_label_index=int(answer_label_index),
    )
    for spec in specs:
        color_entry = color_map[str(spec["value"])]
        spec["color_name"] = str(color_entry["color_name"])
        spec["color_rgb"] = [int(channel) for channel in color_entry["color_rgb"]]
    return specs, str(answer_label)


def _count_option_specs(
    *,
    rng,
    answer_count: int,
    option_count: int,
    answer_label_index: int,
) -> tuple[list[dict[str, Any]], str]:
    labels = OPTION_LABELS[: int(option_count)]
    candidates = list(range(10))
    candidates.sort(key=lambda value: (abs(int(value) - int(answer_count)), int(value)))
    near = [int(value) for value in candidates if int(value) != int(answer_count)]
    specs, answer_label = _make_option_specs(
        rng=rng,
        labels=labels,
        answer_value=int(answer_count),
        distractor_values=near,
        answer_label_index=int(answer_label_index),
    )
    for spec in specs:
        spec["count_value"] = int(spec["value"])
    return specs, str(answer_label)


def _base_record(
    *,
    start_state: Mapping[StickerKey, str],
    final_state: Mapping[StickerKey, str],
    color_map: Mapping[str, Mapping[str, Any]],
    face_color_names: Mapping[str, str],
    scramble_sequence: Sequence[str],
    query_sequence: Sequence[str],
) -> dict[str, Any]:
    return {
        "start_state": dict(start_state),
        "final_state": dict(final_state),
        "color_map": dict(color_map),
        "face_color_names": dict(face_color_names),
        "scramble_sequence": [str(item) for item in scramble_sequence],
        "query_sequence": [str(item) for item in query_sequence],
        "move_sequence_text": format_sequence(query_sequence),
    }


def _sample_sequence_in_range(
    rng,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    *,
    min_key: str,
    max_key: str,
    fallback_min: int,
    fallback_max: int,
) -> list[str]:
    low, high = _as_range(
        params,
        gen_defaults,
        min_key=min_key,
        max_key=max_key,
        fallback_min=int(fallback_min),
        fallback_max=int(fallback_max),
    )
    return sample_move_sequence(rng, length=int(rng.randint(int(low), int(high))))


def build_post_move_sticker_sample(
    *,
    rng,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    option_count: int,
    answer_option_index: int,
) -> dict[str, Any]:
    """Construct a sticker color option-label sample after sampled moves."""

    sequence = _sample_sequence_in_range(
        rng,
        params,
        gen_defaults,
        min_key="post_move_sequence_count_min",
        max_key="post_move_sequence_count_max",
        fallback_min=1,
        fallback_max=3,
    )
    return build_sticker_sample(
        rng=rng,
        params=params,
        gen_defaults=gen_defaults,
        move_sequence=sequence,
        option_count=int(option_count),
        answer_option_index=int(answer_option_index),
    )


def build_sticker_sample(
    *,
    rng,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    move_sequence: Sequence[str],
    option_count: int,
    answer_option_index: int,
) -> dict[str, Any]:
    """Construct a sticker-color record after a sampled move sequence."""

    start_state, color_map, scramble_sequence, face_color_names = _sample_base_state(
        rng=rng,
        params=params,
        gen_defaults=gen_defaults,
    )
    final_state = apply_sequence(start_state, move_sequence)
    target_key = _target_key(rng)
    answer_color_name = str(final_state[target_key])
    option_specs, answer_option_label = _color_option_specs(
        rng=rng,
        color_map=color_map,
        answer_color_name=str(answer_color_name),
        option_count=int(option_count),
        answer_label_index=int(answer_option_index),
    )
    return {
        **_base_record(
            start_state=start_state,
            final_state=final_state,
            color_map=color_map,
            face_color_names=face_color_names,
            scramble_sequence=scramble_sequence,
            query_sequence=move_sequence,
        ),
        "render_mode": "color_options",
        "rubiks_operation": "read_sticker_color",
        "target_face": str(target_key[0]),
        "target_face_name": str(FACE_DISPLAY_NAMES[str(target_key[0])]),
        "target_row": int(target_key[1]),
        "target_col": int(target_key[2]),
        "target_sticker_id": sticker_id(*target_key),
        "answer_color_name": str(answer_color_name),
        "answer_color_rgb": list(color_map[str(answer_color_name)]["color_rgb"]),
        "option_specs": option_specs,
        "option_count": int(option_count),
        "answer_option_label": str(answer_option_label),
        "solver_trace": {
            "operation": "apply_sequence_then_read_sticker_color",
            "final_target_color_name": str(answer_color_name),
        },
    }


def build_post_move_face_count_sample(
    *,
    rng,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    option_count: int,
    answer_option_index: int,
) -> dict[str, Any]:
    """Construct a face color-count option-label sample after sampled moves."""

    sequence = _sample_sequence_in_range(
        rng,
        params,
        gen_defaults,
        min_key="post_move_sequence_count_min",
        max_key="post_move_sequence_count_max",
        fallback_min=1,
        fallback_max=3,
    )
    return build_face_count_sample(
        rng=rng,
        params=params,
        gen_defaults=gen_defaults,
        move_sequence=sequence,
        option_count=int(option_count),
        answer_option_index=int(answer_option_index),
    )


def build_face_count_sample(
    *,
    rng,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    move_sequence: Sequence[str],
    option_count: int,
    answer_option_index: int,
) -> dict[str, Any]:
    """Construct a color-count record after a sampled move sequence."""

    count_min, count_max = _as_range(
        params,
        gen_defaults,
        min_key="face_color_count_answer_min",
        max_key="face_color_count_answer_max",
        fallback_min=1,
        fallback_max=6,
    )
    for _attempt in range(80):
        start_state, color_map, scramble_sequence, face_color_names = (
            _sample_base_state(
                rng=rng,
                params=params,
                gen_defaults=gen_defaults,
            )
        )
        final_state = apply_sequence(start_state, move_sequence)
        target_face = str(FACE_ORDER[int(rng.randrange(len(FACE_ORDER)))])
        color_names = list(color_map.keys())
        rng.shuffle(color_names)
        for color_name in color_names:
            answer_count = face_color_count(
                final_state,
                face=target_face,
                color_name=str(color_name),
            )
            if not int(count_min) <= int(answer_count) <= int(count_max):
                continue
            option_specs, answer_option_label = _count_option_specs(
                rng=rng,
                answer_count=int(answer_count),
                option_count=int(option_count),
                answer_label_index=int(answer_option_index),
            )
            counted_sticker_ids = [
                sticker_id(str(target_face), int(row), int(col))
                for row in range(3)
                for col in range(3)
                if str(final_state[(str(target_face), int(row), int(col))])
                == str(color_name)
            ]
            return {
                **_base_record(
                    start_state=start_state,
                    final_state=final_state,
                    color_map=color_map,
                    face_color_names=face_color_names,
                    scramble_sequence=scramble_sequence,
                    query_sequence=move_sequence,
                ),
                "render_mode": "count_options",
                "rubiks_operation": "count_color_on_face",
                "target_face": str(target_face),
                "target_face_name": str(FACE_DISPLAY_NAMES[str(target_face)]),
                "target_color_name": str(color_name),
                "target_color_rgb": list(color_map[str(color_name)]["color_rgb"]),
                "counted_sticker_ids": list(counted_sticker_ids),
                "answer_count": int(answer_count),
                "option_specs": option_specs,
                "option_count": int(option_count),
                "answer_option_label": str(answer_option_label),
                "solver_trace": {
                    "operation": "apply_sequence_then_count_color_on_face",
                    "final_counted_sticker_ids": list(counted_sticker_ids),
                },
            }
    raise RuntimeError("unable to sample Rubik face color-count constraints")


def build_result_sample(
    *,
    rng,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    option_count: int,
    answer_option_index: int,
    result_mode: str,
) -> dict[str, Any]:
    """Construct a candidate-net option task for direct or inverse moves."""

    start_state, color_map, scramble_sequence, face_color_names = _sample_base_state(
        rng=rng,
        params=params,
        gen_defaults=gen_defaults,
    )
    if str(result_mode) == "inverse":
        base_sequence = _sample_sequence_in_range(
            rng,
            params,
            gen_defaults,
            min_key="inverse_base_move_count_min",
            max_key="inverse_base_move_count_max",
            fallback_min=2,
            fallback_max=2,
        )
        query_sequence = invert_sequence(base_sequence)
    elif str(result_mode) == "direct":
        base_sequence = []
        query_sequence = _sample_sequence_in_range(
            rng,
            params,
            gen_defaults,
            min_key="direct_result_move_count_min",
            max_key="direct_result_move_count_max",
            fallback_min=1,
            fallback_max=2,
        )
    else:
        raise ValueError(f"unsupported Rubik result mode: {result_mode}")

    answer_state = apply_sequence(start_state, query_sequence)
    candidate_states: list[dict[StickerKey, str]] = [dict(answer_state)]
    seen = {state_signature(answer_state)}
    attempts = 0
    while len(candidate_states) < int(option_count) and attempts < 200:
        attempts += 1
        alt_len = max(1, len(query_sequence))
        alt_sequence = sample_move_sequence(rng, length=int(alt_len))
        candidate = apply_sequence(start_state, alt_sequence)
        sig = state_signature(candidate)
        if sig in seen:
            candidate = apply_move(answer_state, sample_move(rng))
            sig = state_signature(candidate)
        if sig in seen:
            continue
        seen.add(sig)
        candidate_states.append(dict(candidate))
    if len(candidate_states) < int(option_count):
        raise RuntimeError("unable to sample distinct Rubik result options")

    labels = list(OPTION_LABELS[: int(option_count)])
    distractor_states = [
        dict(state) for state in candidate_states[1 : int(option_count)]
    ]
    option_specs: list[dict[str, Any]] = []
    answer_option_label = ""
    for option_index, label in enumerate(labels):
        is_correct = int(option_index) == int(answer_option_index)
        state = (
            dict(answer_state) if bool(is_correct) else dict(distractor_states.pop(0))
        )
        if bool(is_correct):
            answer_option_label = str(label)
        option_specs.append(
            {
                "option_id": f"option_{label}",
                "option_label": str(label),
                "is_correct": bool(is_correct),
                "state": dict(state),
            }
        )

    return {
        **_base_record(
            start_state=start_state,
            final_state=answer_state,
            color_map=color_map,
            face_color_names=face_color_names,
            scramble_sequence=scramble_sequence,
            query_sequence=query_sequence,
        ),
        "render_mode": "candidate_nets",
        "rubiks_operation": "match_result_net",
        "base_sequence": [str(item) for item in base_sequence],
        "base_sequence_text": format_sequence(base_sequence),
        "result_mode": str(result_mode),
        "option_specs": option_specs,
        "option_count": int(option_count),
        "answer_option_label": str(answer_option_label),
        "solver_trace": {
            "operation": "apply_sequence_and_match_candidate_net",
            "answer_state_signature": state_signature(answer_state),
        },
    }


__all__ = [
    "build_post_move_face_count_sample",
    "build_post_move_sticker_sample",
    "build_result_sample",
]
