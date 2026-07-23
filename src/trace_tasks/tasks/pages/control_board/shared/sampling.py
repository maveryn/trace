"""Sampling primitives for control-board page scenes."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.deterministic_sampling import resolve_selection_index
from trace_tasks.tasks.shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant

from .defaults import (
    COMMAND_OPTIONS,
    DEFAULTS,
    DISABLED_MODE,
    GENERATION_DEFAULTS,
    NAMESPACE_ROOT,
    SCENE_VARIANTS,
    SELECTED_ENABLED_MODE,
)
from .state import ControlBoardCase, ControlSpec


def _normalize_int_support(params: Mapping[str, Any], key: str, fallback: Sequence[int]) -> Tuple[int, ...]:
    raw_values = params.get(str(key), group_default(GENERATION_DEFAULTS, str(key), fallback))
    support: List[int] = []
    for raw_value in raw_values:
        value = int(raw_value)
        if value not in support:
            support.append(value)
    if not support:
        raise ValueError(f"{key} must not be empty for control-board sampling")
    return tuple(int(value) for value in support)


def _normalize_str_support(params: Mapping[str, Any], key: str, fallback: Sequence[str]) -> Tuple[str, ...]:
    raw_values = params.get(str(key), group_default(GENERATION_DEFAULTS, str(key), fallback))
    support: List[str] = []
    for raw_value in raw_values:
        value = str(raw_value).strip()
        if value and value not in support:
            support.append(value)
    if not support:
        raise ValueError(f"{key} must not be empty for control-board sampling")
    return tuple(str(value) for value in support)


def _support_selection_index(params: Mapping[str, Any], *, instance_seed: int, namespace: str) -> int:
    return int(
        resolve_selection_index(
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{NAMESPACE_ROOT}:{namespace}",
        )
    )


def _resolve_axis(
    rng,
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    supported: Tuple[str, ...],
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    selected, probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=GENERATION_DEFAULTS,
        supported_variants=supported,
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
    )
    balanced = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GENERATION_DEFAULTS,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=supported,
        balance_flag_key=str(balance_flag_key),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        sampling_namespace=f"{NAMESPACE_ROOT}:{namespace}",
    )
    return str(balanced), {str(key): float(value) for key, value in sorted(probabilities.items())}


def _select_indices(*, instance_seed: int, namespace: str, count: int, size: int) -> Tuple[int, ...]:
    if int(count) > int(size):
        raise ValueError("cannot select more control indices than group size")
    indices = list(range(int(size)))
    rng = spawn_rng(int(instance_seed), f"{NAMESPACE_ROOT}.{namespace}")
    rng.shuffle(indices)
    return tuple(sorted(int(value) for value in indices[: int(count)]))


def _select_values(
    *,
    instance_seed: int,
    namespace: str,
    values: Sequence[Tuple[int, int]],
    count: int,
) -> Tuple[Tuple[int, int], ...]:
    if int(count) > len(values):
        raise ValueError("cannot select more values than population size")
    shuffled = list(values)
    rng = spawn_rng(int(instance_seed), f"{NAMESPACE_ROOT}.{namespace}")
    rng.shuffle(shuffled)
    return tuple(sorted(shuffled[: int(count)]))


def _group_count_state_sets(
    *,
    instance_seed: int,
    count_mode: str,
    target_group_index: int,
    target_group_size: int,
    answer_value: int,
    group_sizes: Sequence[int],
) -> Tuple[set[Tuple[int, int]], set[Tuple[int, int]]]:
    """Place target states and non-target distractor states across groups."""

    disabled: set[Tuple[int, int]] = set()
    selected: set[Tuple[int, int]] = set()
    target_indices = _select_indices(
        instance_seed=int(instance_seed),
        namespace=f"target_state.{count_mode}.{target_group_index}",
        count=int(answer_value),
        size=int(target_group_size),
    )
    target_keys = {(int(target_group_index), int(idx)) for idx in target_indices}
    if str(count_mode) == DISABLED_MODE:
        disabled.update(target_keys)
    elif str(count_mode) == SELECTED_ENABLED_MODE:
        selected.update(target_keys)

    for group_index, group_size in enumerate(group_sizes):
        if str(count_mode) == SELECTED_ENABLED_MODE:
            group_keys = [(int(group_index), int(idx)) for idx in range(int(group_size))]
            if int(group_index) == int(target_group_index):
                remaining = [key for key in group_keys if key not in target_keys]
                distractor_count = min(2, len(remaining))
                for key in _select_values(
                    instance_seed=int(instance_seed) + int(group_index),
                    namespace=f"selected_disabled_target_distractors.{group_index}",
                    values=remaining,
                    count=distractor_count,
                ):
                    disabled.add(key)
                    selected.add(key)
                continue

            selected_enabled_count = min(2, len(group_keys))
            selected_enabled = set(
                _select_values(
                    instance_seed=int(instance_seed) + (53 * int(group_index)),
                    namespace=f"selected_enabled_group_distractors.{group_index}",
                    values=group_keys,
                    count=selected_enabled_count,
                )
            )
            selected.update(selected_enabled)
            remaining = [key for key in group_keys if key not in selected_enabled]
            selected_disabled_count = min(1, len(remaining))
            for key in _select_values(
                instance_seed=int(instance_seed) + (97 * int(group_index)),
                namespace=f"selected_disabled_group_distractors.{group_index}",
                values=remaining,
                count=selected_disabled_count,
            ):
                disabled.add(key)
                selected.add(key)
            continue

        non_target_state_count = 1 + int(
            _support_selection_index(
                {"count_mode": count_mode},
                instance_seed=int(instance_seed) + int(group_index),
                namespace=f"state_distractors.{group_index}",
            )
            % 2
        )
        if int(group_index) == int(target_group_index):
            distractor_count = 1 if int(group_size) - int(answer_value) >= 2 else 0
        else:
            distractor_count = min(non_target_state_count, max(0, int(group_size) - 1))
        if distractor_count <= 0:
            continue
        choices = _select_indices(
            instance_seed=int(instance_seed) + (97 * int(group_index)),
            namespace=f"distractor_state.{count_mode}.{group_index}",
            count=int(distractor_count),
            size=int(group_size),
        )
        for idx in choices:
            key = (int(group_index), int(idx))
            if key in disabled or key in selected:
                continue
            if str(count_mode) == DISABLED_MODE:
                selected.add(key)
            elif str(count_mode) == SELECTED_ENABLED_MODE:
                disabled.add(key)
                selected.add(key)
    return disabled, selected


def _group_state_sets_from_counts(
    *,
    instance_seed: int,
    count_mode: str,
    group_sizes: Sequence[int],
    group_state_counts: Sequence[int],
) -> Tuple[set[Tuple[int, int]], set[Tuple[int, int]]]:
    """Place requested matching-state counts in every group.

    The caller owns the objective-specific meaning of the per-group counts.
    This helper only turns semantic state counts into disabled/selected flags
    for the shared control-board renderer.
    """

    if len(group_state_counts) != len(group_sizes):
        raise ValueError("group_state_counts must match group_sizes length")
    disabled: set[Tuple[int, int]] = set()
    selected: set[Tuple[int, int]] = set()
    for group_index, (group_size, state_count) in enumerate(zip(group_sizes, group_state_counts)):
        count = int(state_count)
        if count < 0 or count > int(group_size):
            raise ValueError("group_state_counts values must fit each group size")
        selected_indices = _select_indices(
            instance_seed=int(instance_seed) + (211 * int(group_index)),
            namespace=f"group_state_count.{count_mode}.{group_index}.{count}",
            count=int(count),
            size=int(group_size),
        )
        for idx in selected_indices:
            key = (int(group_index), int(idx))
            if str(count_mode) == DISABLED_MODE:
                disabled.add(key)
            elif str(count_mode) == SELECTED_ENABLED_MODE:
                selected.add(key)
            else:
                raise ValueError(f"unsupported control state count mode: {count_mode}")

    for group_index, group_size in enumerate(group_sizes):
        group_keys = [(int(group_index), int(idx)) for idx in range(int(group_size))]
        if str(count_mode) == DISABLED_MODE:
            remaining = [key for key in group_keys if key not in disabled]
            if remaining:
                selected_only_count = min(1 + (int(group_index) % 2), len(remaining))
                for key in _select_values(
                    instance_seed=int(instance_seed) + (307 * int(group_index)),
                    namespace=f"selected_distractors.{group_index}",
                    values=remaining,
                    count=int(selected_only_count),
                ):
                    selected.add(key)
        elif str(count_mode) == SELECTED_ENABLED_MODE:
            remaining = [key for key in group_keys if key not in selected]
            if remaining:
                selected_disabled_count = min(1, len(remaining))
                for key in _select_values(
                    instance_seed=int(instance_seed) + (401 * int(group_index)),
                    namespace=f"selected_disabled_distractors.{group_index}",
                    values=remaining,
                    count=int(selected_disabled_count),
                ):
                    selected.add(key)
                    disabled.add(key)
    return disabled, selected


def _assemble_controls(
    *,
    instance_seed: int,
    group_names: Sequence[str],
    group_sizes: Sequence[int],
    candidate_label_pool: Sequence[str],
    disabled_set: set[Tuple[int, int]],
    selected_set: set[Tuple[int, int]],
    annotation_predicate: Callable[[int, int, bool, bool], bool],
) -> Tuple[Tuple[ControlSpec, ...], Tuple[str, ...]]:
    """Create visible controls and annotation ids from final state sets."""

    total_controls = int(sum(int(value) for value in group_sizes))
    if total_controls > len(candidate_label_pool):
        raise ValueError("candidate_label_pool must cover all rendered GUI controls")
    if total_controls > len(COMMAND_OPTIONS):
        raise ValueError("not enough command options for rendered GUI controls")

    command_rng = spawn_rng(int(instance_seed), f"{NAMESPACE_ROOT}.commands")
    command_options = list(COMMAND_OPTIONS)
    command_rng.shuffle(command_options)

    controls: List[ControlSpec] = []
    label_index = 0
    annotation_ids: List[str] = []
    for group_index, group_name in enumerate(group_names):
        for order_in_group in range(int(group_sizes[int(group_index)])):
            label = str(candidate_label_pool[int(label_index)])
            control_ref = f"control_{str(label).lower()}"
            disabled = (int(group_index), int(order_in_group)) in disabled_set
            selected = (int(group_index), int(order_in_group)) in selected_set
            if annotation_predicate(int(group_index), int(order_in_group), bool(disabled), bool(selected)):
                annotation_ids.append(str(control_ref))
            controls.append(
                ControlSpec(
                    control_id=str(control_ref),
                    candidate_label=str(label),
                    group_name=str(group_name),
                    group_index=int(group_index),
                    order_in_group=int(order_in_group),
                    global_order_index=int(label_index),
                    command=command_options[int(label_index)],
                    enabled=not bool(disabled),
                    selected=bool(selected),
                    is_reference=False,
                )
            )
            label_index += 1
    return tuple(controls), tuple(str(value) for value in annotation_ids)


def build_control_board_case(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    count_mode: str,
    default_answer_support: Sequence[int],
) -> ControlBoardCase:
    """Resolve one deterministic control-board case for a public objective."""

    rng = spawn_rng(int(instance_seed), f"{NAMESPACE_ROOT}.scene")
    scene_variant, scene_variant_probabilities = _resolve_axis(
        rng,
        instance_seed=int(instance_seed),
        params=params,
        supported=SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        namespace="scene_variant",
    )
    group_names = _normalize_str_support(params, "group_name_pool", DEFAULTS.group_name_pool)
    if len(group_names) != 4:
        raise ValueError("control-board sampling requires exactly four group names")
    if params.get("state_count_support") is None:
        answer_support = tuple(int(value) for value in default_answer_support)
    else:
        answer_support = _normalize_int_support(params, "state_count_support", tuple(int(value) for value in default_answer_support))
    candidate_label_pool = _normalize_str_support(params, "candidate_label_pool", DEFAULTS.candidate_label_pool)

    explicit_answer_value = params.get("answer_value")
    if explicit_answer_value is not None:
        answer_value = int(explicit_answer_value)
        if int(answer_value) not in set(int(value) for value in answer_support):
            raise ValueError("answer_value must be in the active answer support")
    else:
        answer_value = int(
            answer_support[
                _support_selection_index(
                    params,
                    instance_seed=int(instance_seed),
                    namespace=f"answer_value.{count_mode}",
                )
                % len(answer_support)
            ]
        )

    target_group_index = int(
        _support_selection_index(
            params,
            instance_seed=int(instance_seed),
            namespace=f"target_group.{count_mode}",
        )
        % len(group_names)
    )
    if str(count_mode) == SELECTED_ENABLED_MODE:
        target_group_size = 8
    else:
        target_group_size = min(8, max(int(answer_value) + 2, 6))

    group_sizes: List[int] = []
    for group_index in range(len(group_names)):
        if int(group_index) == int(target_group_index):
            group_sizes.append(int(target_group_size))
        elif str(count_mode) == SELECTED_ENABLED_MODE:
            group_sizes.append(6)
        else:
            group_sizes.append(5)
    disabled_set, selected_set = _group_count_state_sets(
        instance_seed=int(instance_seed),
        count_mode=str(count_mode),
        target_group_index=int(target_group_index),
        target_group_size=int(target_group_size),
        answer_value=int(answer_value),
        group_sizes=group_sizes,
    )

    def _count_annotation_predicate(group_index: int, _order: int, disabled: bool, selected: bool) -> bool:
        if int(group_index) != int(target_group_index):
            return False
        if str(count_mode) == DISABLED_MODE:
            return bool(disabled)
        if str(count_mode) == SELECTED_ENABLED_MODE:
            return bool(selected) and not bool(disabled)
        return False

    controls, annotation_ids = _assemble_controls(
        instance_seed=int(instance_seed),
        group_names=group_names,
        group_sizes=group_sizes,
        candidate_label_pool=candidate_label_pool,
        disabled_set=disabled_set,
        selected_set=selected_set,
        annotation_predicate=_count_annotation_predicate,
    )

    if len(annotation_ids) != int(answer_value):
        raise RuntimeError(
            "control-board annotation cardinality does not match answer: "
            f"{len(annotation_ids)} != {answer_value}"
        )

    return ControlBoardCase(
        count_mode=str(count_mode),
        scene_variant=str(scene_variant),
        controls=tuple(controls),
        group_names=tuple(str(value) for value in group_names),
        target_group_name=str(group_names[int(target_group_index)]),
        target_group_index=int(target_group_index),
        answer_value=int(answer_value),
        annotation_control_ids=tuple(str(value) for value in annotation_ids),
        answer_support=tuple(int(value) for value in answer_support),
        candidate_label_pool=tuple(str(value) for value in candidate_label_pool),
        scene_variant_probabilities=dict(scene_variant_probabilities),
    )


def build_control_board_case_from_group_state_counts(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    count_mode: str,
    target_group_index: int,
    group_state_counts: Sequence[int],
    answer_support: Sequence[int],
) -> ControlBoardCase:
    """Build one control board from explicit per-group state counts."""

    rng = spawn_rng(int(instance_seed), f"{NAMESPACE_ROOT}.scene")
    scene_variant, scene_variant_probabilities = _resolve_axis(
        rng,
        instance_seed=int(instance_seed),
        params=params,
        supported=SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        namespace="scene_variant",
    )
    group_names = _normalize_str_support(params, "group_name_pool", DEFAULTS.group_name_pool)
    if len(group_names) != 4:
        raise ValueError("control-board sampling requires exactly four group names")
    if len(group_state_counts) != len(group_names):
        raise ValueError("group_state_counts must contain one count per group")
    candidate_label_pool = _normalize_str_support(params, "candidate_label_pool", DEFAULTS.candidate_label_pool)
    group_sizes = [6 for _ in group_names]
    disabled_set, selected_set = _group_state_sets_from_counts(
        instance_seed=int(instance_seed),
        count_mode=str(count_mode),
        group_sizes=group_sizes,
        group_state_counts=tuple(int(value) for value in group_state_counts),
    )

    def _target_state_predicate(group_index: int, _order: int, disabled: bool, selected: bool) -> bool:
        if int(group_index) != int(target_group_index):
            return False
        if str(count_mode) == DISABLED_MODE:
            return bool(disabled)
        if str(count_mode) == SELECTED_ENABLED_MODE:
            return bool(selected) and not bool(disabled)
        return False

    controls, annotation_ids = _assemble_controls(
        instance_seed=int(instance_seed),
        group_names=group_names,
        group_sizes=group_sizes,
        candidate_label_pool=candidate_label_pool,
        disabled_set=disabled_set,
        selected_set=selected_set,
        annotation_predicate=_target_state_predicate,
    )
    answer_value = int(group_state_counts[int(target_group_index)])
    if len(annotation_ids) != int(answer_value):
        raise RuntimeError(
            "control-board target state count does not match annotation ids: "
            f"{len(annotation_ids)} != {answer_value}"
        )
    return ControlBoardCase(
        count_mode=str(count_mode),
        scene_variant=str(scene_variant),
        controls=tuple(controls),
        group_names=tuple(str(value) for value in group_names),
        target_group_name=str(group_names[int(target_group_index)]),
        target_group_index=int(target_group_index),
        answer_value=int(answer_value),
        annotation_control_ids=tuple(str(value) for value in annotation_ids),
        answer_support=tuple(int(value) for value in answer_support),
        candidate_label_pool=tuple(str(value) for value in candidate_label_pool),
        scene_variant_probabilities=dict(scene_variant_probabilities),
    )
