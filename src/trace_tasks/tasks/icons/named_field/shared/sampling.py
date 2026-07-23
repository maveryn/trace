"""Sampling primitives for named-field icon tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from .....core.seed import spawn_rng
from ....shared.color_format import format_named_color_with_hex
from ....shared.config_defaults import group_default
from ....shared.deterministic_sampling import uniform_probability_map
from ....shared.named_colors import available_named_colors, named_color
from ....shared.weighted_sampling import sample_weighted_value, weighted_probability_map
from ...shared.procedural_named_icon_field_scene import (
    resolve_named_icon_fill_style_probabilities,
    resolve_named_icon_fill_style_support,
    resolve_named_icon_int_bounds,
    uniform_string_probability_map,
)
from ...shared.procedural_named_icons import (
    PROCEDURAL_NAMED_ICON_SHAPES,
    procedural_named_icon_display_name,
    sample_procedural_named_icon_fill_style,
)
from .defaults import (
    BOOLEAN_DEFAULTS as _BOOLEAN_DEFAULTS,
    COUNTERFACTUAL_DEFAULTS as _COUNTERFACTUAL_DEFAULTS,
    NON_STACK_LAYOUT_MODES as _NON_STACK_LAYOUT_MODES,
    PAIR_ARITHMETIC_DEFAULTS as _PAIR_ARITHMETIC_DEFAULTS,
    SHAPE_COUNT_ARRANGEMENT_PROFILES as _SHAPE_COUNT_ARRANGEMENT_PROFILES,
    SHAPE_COUNT_DEFAULTS as _SHAPE_COUNT_DEFAULTS,
)
from .metrics import (
    BOOLEAN_PREDICATES,
    BOOLEAN_PREDICATE_AND,
    BOOLEAN_PREDICATE_ATTRIBUTE_WITHOUT_SHAPE,
    BOOLEAN_PREDICATE_NEITHER,
    BOOLEAN_PREDICATE_OR,
    BOOLEAN_PREDICATE_SHAPE_WITHOUT_ATTRIBUTE,
    BOOLEAN_PREDICATE_XOR,
    COUNTERFACTUAL_EDIT_KINDS,
    COUNTERFACTUAL_SHAPE_REMOVAL,
    COUNTERFACTUAL_SHAPE_REPLACEMENT,
)
from .state import (
    BooleanIconSemanticSpec as _BooleanIconSemanticSpec,
    BooleanSampleSpec as _BooleanSampleSpec,
    CounterfactualIconSemanticSpec as _CounterfactualIconSemanticSpec,
    CounterfactualSampleSpec as _CounterfactualSampleSpec,
    NamedColorEntry as _NamedColorEntry,
    PairArithmeticIconSemanticSpec,
    PairArithmeticOperandSpec,
    PairArithmeticSampleSpec,
    ShapeCountSampleSpec,
)

_ATTRIBUTE_AXES: Tuple[str, ...] = ("color",)


def _int_param(params: Mapping[str, Any], gen_defaults: Mapping[str, Any], key: str, fallback: int) -> int:
    return int(params.get(key, group_default(gen_defaults, key, fallback)))


def _shape_support(params: Mapping[str, Any], gen_defaults: Mapping[str, Any], *, min_count: int = 2) -> Tuple[str, ...]:
    raw = params.get("shape_id_support", group_default(gen_defaults, "shape_id_support", PROCEDURAL_NAMED_ICON_SHAPES))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("shape_id_support must be a sequence")
    values = tuple(str(value) for value in raw)
    unsupported = sorted(set(values) - set(PROCEDURAL_NAMED_ICON_SHAPES))
    if unsupported:
        raise ValueError(f"unsupported procedural named icon shapes: {unsupported}")
    support = tuple(dict.fromkeys(values))
    if len(support) < int(min_count):
        raise ValueError(f"shape_id_support must include at least {int(min_count)} shapes")
    return support


def _color_support(params: Mapping[str, Any], gen_defaults: Mapping[str, Any]) -> Tuple[_NamedColorEntry, ...]:
    color_by_name = {str(name): tuple(int(channel) for channel in rgb) for name, rgb in available_named_colors()}
    raw = params.get("named_color_support", group_default(gen_defaults, "named_color_support", tuple(color_by_name)))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("named_color_support must be a sequence")
    names = tuple(dict.fromkeys(str(value).strip().lower() for value in raw if str(value).strip()))
    unsupported = sorted(set(names) - set(color_by_name))
    if unsupported:
        raise ValueError(f"unsupported named colors: {unsupported}")
    if len(names) < 2:
        raise ValueError("named_color_support must include at least two colors")
    return tuple(
        _NamedColorEntry(
            name=str(name),
            rgb=tuple(int(channel) for channel in named_color(str(name))),
            label=format_named_color_with_hex(str(name), named_color(str(name))),
        )
        for name in names
    )


def _attribute_axis_probability_map(params: Mapping[str, Any], gen_defaults: Mapping[str, Any]) -> Dict[str, float]:
    raw = params.get(
        "attribute_axis_probabilities",
        group_default(gen_defaults, "attribute_axis_probabilities", {"color": 1.0}),
    )
    if not isinstance(raw, Mapping):
        raw = {"color": 1.0}
    weights = {str(axis): max(0.0, float(raw.get(str(axis), 0.0))) for axis in _ATTRIBUTE_AXES}
    total = sum(float(value) for value in weights.values())
    if total <= 0.0:
        raise ValueError("attribute_axis_probabilities must assign positive mass")
    return {str(axis): float(weights[str(axis)]) / float(total) for axis in _ATTRIBUTE_AXES}


def _sample_attribute_axis(params: Mapping[str, Any], gen_defaults: Mapping[str, Any], rng) -> Tuple[str, Dict[str, float]]:
    explicit = params.get("attribute_axis")
    probabilities = _attribute_axis_probability_map(params, gen_defaults)
    if explicit is not None:
        axis = str(explicit)
        if axis not in _ATTRIBUTE_AXES:
            raise ValueError(f"attribute_axis must be one of {_ATTRIBUTE_AXES}")
        return axis, {axis: 1.0}
    threshold = float(rng.random())
    cumulative = 0.0
    for axis in _ATTRIBUTE_AXES:
        cumulative += float(probabilities[str(axis)])
        if threshold <= cumulative:
            return str(axis), dict(probabilities)
    return str(_ATTRIBUTE_AXES[-1]), dict(probabilities)


def _arrangement_mode_support(params: Mapping[str, Any], render_defaults: Mapping[str, Any], fallback_modes: Sequence[str]) -> Tuple[str, ...]:
    raw = params.get("named_icon_layout_modes", group_default(render_defaults, "named_icon_layout_modes", tuple(fallback_modes)))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        values = tuple(fallback_modes)
    else:
        values = tuple(str(value) for value in raw if str(value).strip())
    unsupported = sorted(set(values) - set(_NON_STACK_LAYOUT_MODES))
    if unsupported:
        raise ValueError(f"named-field count tasks only support non-stack layouts; got {unsupported}")
    modes = tuple(dict.fromkeys(values))
    if not modes:
        raise ValueError("named_icon_layout_modes resolved no supported non-stack layouts")
    return modes


def _compose(total: int, buckets: int, rng, *, require_positive_if_possible: bool) -> Tuple[int, ...]:
    if buckets <= 0:
        return ()
    if total < 0:
        raise ValueError("cannot compose a negative total")
    if total == 0:
        return tuple(0 for _ in range(buckets))
    if require_positive_if_possible and total >= buckets:
        remaining = int(total) - int(buckets)
        values = [1 for _ in range(buckets)]
    else:
        remaining = int(total)
        values = [0 for _ in range(buckets)]
    for _ in range(int(remaining)):
        values[int(rng.randrange(0, int(buckets)))] += 1
    rng.shuffle(values)
    return tuple(int(value) for value in values)

def _initial_partition_counts(predicate_kind: str, target_answer: int, rng) -> Dict[str, int]:
    target = int(target_answer)
    if target < 1:
        raise ValueError("target_answer must be positive")
    if predicate_kind == BOOLEAN_PREDICATE_AND:
        return {"both": target, "shape_only": 1, "attribute_only": 1, "neither": 1}
    if predicate_kind == BOOLEAN_PREDICATE_OR:
        both, shape_only, attribute_only = _compose(target, 3, rng, require_positive_if_possible=True)
        return {"both": both, "shape_only": shape_only, "attribute_only": attribute_only, "neither": 1}
    if predicate_kind == BOOLEAN_PREDICATE_SHAPE_WITHOUT_ATTRIBUTE:
        return {"both": 1, "shape_only": target, "attribute_only": 1, "neither": 1}
    if predicate_kind == BOOLEAN_PREDICATE_ATTRIBUTE_WITHOUT_SHAPE:
        return {"both": 1, "shape_only": 1, "attribute_only": target, "neither": 1}
    if predicate_kind == BOOLEAN_PREDICATE_NEITHER:
        return {"both": 1, "shape_only": 1, "attribute_only": 1, "neither": target}
    if predicate_kind == BOOLEAN_PREDICATE_XOR:
        shape_only, attribute_only = _compose(target, 2, rng, require_positive_if_possible=True)
        return {"both": 1, "shape_only": shape_only, "attribute_only": attribute_only, "neither": 1}
    raise ValueError(f"unsupported Boolean predicate kind: {predicate_kind}")


def _safe_fill_partitions(predicate_kind: str) -> Tuple[str, ...]:
    if predicate_kind == BOOLEAN_PREDICATE_AND:
        return ("shape_only", "attribute_only", "neither")
    if predicate_kind == BOOLEAN_PREDICATE_OR:
        return ("neither",)
    if predicate_kind == BOOLEAN_PREDICATE_SHAPE_WITHOUT_ATTRIBUTE:
        return ("both", "attribute_only", "neither")
    if predicate_kind == BOOLEAN_PREDICATE_ATTRIBUTE_WITHOUT_SHAPE:
        return ("both", "shape_only", "neither")
    if predicate_kind == BOOLEAN_PREDICATE_NEITHER:
        return ("both", "shape_only", "attribute_only")
    if predicate_kind == BOOLEAN_PREDICATE_XOR:
        return ("both", "neither")
    raise ValueError(f"unsupported Boolean predicate kind: {predicate_kind}")


def _answer_from_partitions(predicate_kind: str, partition_counts: Mapping[str, int]) -> int:
    counts = {str(key): int(value) for key, value in partition_counts.items()}
    if predicate_kind == BOOLEAN_PREDICATE_AND:
        return counts.get("both", 0)
    if predicate_kind == BOOLEAN_PREDICATE_OR:
        return counts.get("both", 0) + counts.get("shape_only", 0) + counts.get("attribute_only", 0)
    if predicate_kind == BOOLEAN_PREDICATE_SHAPE_WITHOUT_ATTRIBUTE:
        return counts.get("shape_only", 0)
    if predicate_kind == BOOLEAN_PREDICATE_ATTRIBUTE_WITHOUT_SHAPE:
        return counts.get("attribute_only", 0)
    if predicate_kind == BOOLEAN_PREDICATE_NEITHER:
        return counts.get("neither", 0)
    if predicate_kind == BOOLEAN_PREDICATE_XOR:
        return counts.get("shape_only", 0) + counts.get("attribute_only", 0)
    raise ValueError(f"unsupported Boolean predicate kind: {predicate_kind}")


def _partition_counts_for_predicate(*, predicate_kind: str, target_answer: int, object_count: int, rng) -> Dict[str, int]:
    counts = _initial_partition_counts(str(predicate_kind), int(target_answer), rng)
    current_total = sum(int(value) for value in counts.values())
    if int(object_count) < int(current_total):
        raise ValueError("object_count leaves no room for required Boolean distractor partitions")
    fillable = _safe_fill_partitions(str(predicate_kind))
    while current_total < int(object_count):
        counts[str(rng.choice(fillable))] += 1
        current_total += 1
    answer = _answer_from_partitions(str(predicate_kind), counts)
    if answer != int(target_answer):
        raise RuntimeError("partition fill changed Boolean target answer")
    return {str(key): int(value) for key, value in counts.items()}


def _other_value(rng, values: Sequence[str], excluded: str) -> str:
    candidates = [str(value) for value in values if str(value) != str(excluded)]
    if not candidates:
        raise ValueError("no alternate value available")
    return str(rng.choice(candidates))


def _boolean_semantic_specs_from_partitions(
    *,
    partition_counts: Mapping[str, int],
    target_shape_id: str,
    target_color_name: str,
    attribute_axis: str,
    shape_support: Sequence[str],
    color_support: Sequence[_NamedColorEntry],
    fill_style_support: Sequence[str],
    fill_style_probabilities: Dict[str, float],
    rng,
) -> Tuple[_BooleanIconSemanticSpec, ...]:
    """Build target-relative icon partitions for one Boolean count plan.

    The invariant is that membership in the four symbolic partitions alone
    determines the final answer; rendering can only project those instances.
    """

    color_names = tuple(str(entry.name) for entry in color_support)
    def build_spec(*, shape_matches: bool, attribute_matches: bool, partition: str) -> _BooleanIconSemanticSpec:
        shape_id = str(target_shape_id) if bool(shape_matches) else _other_value(rng, shape_support, str(target_shape_id))
        if str(attribute_axis) == "color":
            color_name = str(target_color_name) if bool(attribute_matches) else _other_value(rng, color_names, str(target_color_name))
            fill_style = sample_procedural_named_icon_fill_style(
                rng,
                support=fill_style_support,
                probabilities=fill_style_probabilities,
            )
        else:
            raise ValueError(f"unsupported attribute axis: {attribute_axis}")
        return _BooleanIconSemanticSpec(
            shape_id=str(shape_id),
            color_name=str(color_name),
            fill_style=str(fill_style),
            partition=str(partition),
        )

    specs: list[_BooleanIconSemanticSpec] = []
    for _ in range(int(partition_counts.get("both", 0))):
        specs.append(build_spec(shape_matches=True, attribute_matches=True, partition="both"))
    for _ in range(int(partition_counts.get("shape_only", 0))):
        specs.append(build_spec(shape_matches=True, attribute_matches=False, partition="shape_only"))
    for _ in range(int(partition_counts.get("attribute_only", 0))):
        specs.append(build_spec(shape_matches=False, attribute_matches=True, partition="attribute_only"))
    for _ in range(int(partition_counts.get("neither", 0))):
        specs.append(build_spec(shape_matches=False, attribute_matches=False, partition="neither"))
    rng.shuffle(specs)
    return tuple(specs)


def _sample_boolean_spec(
    *,
    run_namespace: str,
    prompt_query_key: str,
    predicate_kind: str,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
) -> _BooleanSampleSpec:
    """Sample one feasible Boolean named-field scene from task-owned semantics.

    The public task has already selected the predicate; this helper only
    chooses operands, counts, and distractor partitions that realize it exactly.
    """

    if str(predicate_kind) not in set(BOOLEAN_PREDICATES):
        raise ValueError(f"unsupported Boolean predicate kind: {predicate_kind}")
    rng = spawn_rng(int(instance_seed), f"{run_namespace}.sample")
    shape_support = _shape_support(params, gen_defaults)
    color_support = _color_support(params, gen_defaults)
    fill_style_support = resolve_named_icon_fill_style_support(
        params,
        gen_defaults,
        fallback_support=_BOOLEAN_DEFAULTS.named_icon_fill_style_support,
    )
    fill_style_probabilities = resolve_named_icon_fill_style_probabilities(params, gen_defaults, fill_style_support)
    arrangement_support = _arrangement_mode_support(params, render_defaults, _BOOLEAN_DEFAULTS.named_icon_layout_modes)
    attribute_axis, attribute_axis_probabilities = _sample_attribute_axis(params, gen_defaults, rng)
    answer_min, answer_max = resolve_named_icon_int_bounds(
        params,
        gen_defaults,
        "target_count_min",
        "target_count_max",
        _BOOLEAN_DEFAULTS.target_count_min,
        _BOOLEAN_DEFAULTS.target_count_max,
    )
    object_min, object_max = resolve_named_icon_int_bounds(
        params,
        gen_defaults,
        "object_count_min",
        "object_count_max",
        _BOOLEAN_DEFAULTS.object_count_min,
        _BOOLEAN_DEFAULTS.object_count_max,
    )
    object_max_answer_offset = _int_param(
        params,
        gen_defaults,
        "object_count_max_answer_offset",
        _BOOLEAN_DEFAULTS.object_count_max_answer_offset,
    )
    if answer_min < 1:
        raise ValueError("Boolean named-icon counting uses target_count_min >= 1")
    if object_max_answer_offset < 0:
        raise ValueError("object_count_max_answer_offset must be non-negative")

    answer_support = tuple(range(int(answer_min), int(answer_max) + 1))
    target_count_probabilities = weighted_probability_map(
        answer_support,
        params.get("target_count_weights", group_default(gen_defaults, "target_count_weights", None)),
    )
    explicit_target = params.get("target_count", params.get("target_answer"))
    if explicit_target is not None:
        target_answer = int(explicit_target)
        if target_answer not in set(answer_support):
            raise ValueError(f"target_count must be in {answer_support}")
    else:
        target_answer = int(sample_weighted_value(rng, answer_support, target_count_probabilities))

    explicit_shape = params.get("shape_id", params.get("target_shape_id"))
    if explicit_shape is not None:
        target_shape_id = str(explicit_shape)
        if target_shape_id not in set(shape_support):
            raise ValueError(f"target shape must be one of {shape_support}")
    else:
        target_shape_id = str(rng.choice(shape_support))

    color_by_name = {str(entry.name): entry for entry in color_support}
    explicit_color = params.get("color_name", params.get("target_color_name"))
    target_color: _NamedColorEntry | None = None
    if str(attribute_axis) == "color":
        if explicit_color is not None:
            target_color_name = str(explicit_color).strip().lower()
            if target_color_name not in color_by_name:
                raise ValueError(f"target color must be one of {tuple(color_by_name)}")
        else:
            target_color_name = str(rng.choice(color_support).name)
        target_color = color_by_name[str(target_color_name)]
        target_attribute_value = str(target_color.name)
        target_attribute_label = str(target_color.label)
    else:
        raise ValueError(f"unsupported attribute axis: {attribute_axis}")

    min_required_counts = _initial_partition_counts(str(predicate_kind), int(target_answer), rng)
    min_required_total = sum(int(value) for value in min_required_counts.values())
    min_object_count = max(int(object_min), int(min_required_total))
    answer_relative_object_max = int(target_answer) + int(object_max_answer_offset)
    dynamic_object_max = min(int(object_max), max(int(min_required_total), int(answer_relative_object_max)))
    if min_object_count > int(dynamic_object_max):
        raise ValueError("object_count range cannot support requested Boolean target")
    object_support = tuple(range(int(min_object_count), int(dynamic_object_max) + 1))
    explicit_object_count = params.get("object_count")
    if explicit_object_count is not None:
        object_count = int(explicit_object_count)
        if object_count < int(min_object_count) or object_count > int(dynamic_object_max):
            raise ValueError("object_count is outside configured support")
    else:
        object_count = int(rng.choice(object_support))

    partition_counts = _partition_counts_for_predicate(
        predicate_kind=str(predicate_kind),
        target_answer=int(target_answer),
        object_count=int(object_count),
        rng=rng,
    )
    semantic_specs = _boolean_semantic_specs_from_partitions(
        partition_counts=partition_counts,
        target_shape_id=str(target_shape_id),
        target_color_name=str(target_color.name) if target_color is not None else "",
        attribute_axis=str(attribute_axis),
        shape_support=shape_support,
        color_support=color_support,
        fill_style_support=fill_style_support,
        fill_style_probabilities=fill_style_probabilities,
        rng=rng,
    )

    explicit_arrangement = params.get("arrangement_mode", params.get("layout_mode"))
    if explicit_arrangement is not None:
        arrangement_mode = str(explicit_arrangement)
        if arrangement_mode not in set(arrangement_support):
            raise ValueError(f"Boolean named-icon counting only supports non-stack layouts: {arrangement_support}")
    else:
        arrangement_mode = str(rng.choice(arrangement_support))

    return _BooleanSampleSpec(
        prompt_query_key=str(prompt_query_key),
        predicate_kind=str(predicate_kind),
        target_shape_id=str(target_shape_id),
        target_shape_name=procedural_named_icon_display_name(str(target_shape_id)),
        target_attribute_axis=str(attribute_axis),
        target_attribute_value=str(target_attribute_value),
        target_attribute_label=str(target_attribute_label),
        target_color=target_color,
        target_answer=int(target_answer),
        object_count=int(object_count),
        object_count_max_answer_offset=int(object_max_answer_offset),
        arrangement_mode=str(arrangement_mode),
        partition_counts=dict(partition_counts),
        semantic_specs=tuple(semantic_specs),
        shape_probabilities=uniform_string_probability_map(shape_support, selected=str(target_shape_id) if explicit_shape is not None else None),
        color_probabilities=uniform_string_probability_map(
            tuple(color_by_name),
            selected=str(target_color.name) if explicit_color is not None and target_color is not None else None,
        ),
        fill_style_probabilities=dict(fill_style_probabilities),
        attribute_axis_probabilities=dict(attribute_axis_probabilities),
        target_count_probabilities=dict(
            uniform_probability_map(answer_support, selected=int(target_answer))
            if explicit_target is not None
            else target_count_probabilities
        ),
        object_count_probabilities=dict(
            uniform_probability_map(object_support, selected=int(object_count) if explicit_object_count is not None else None)
        ),
        arrangement_mode_probabilities=uniform_string_probability_map(
            arrangement_support,
            selected=str(arrangement_mode) if explicit_arrangement is not None else None,
        ),
    )



def _counterfactual_other_shapes(rng, support: Sequence[str], excluded: Sequence[str], *, count: int) -> Tuple[str, ...]:
    excluded_set = {str(value) for value in excluded}
    candidates = [str(value) for value in support if str(value) not in excluded_set]
    if len(candidates) < int(count):
        raise ValueError("not enough alternate shapes available")
    rng.shuffle(candidates)
    return tuple(str(value) for value in candidates[: int(count)])


def _counterfactual_split_answer_into_source_and_target(rng, answer: int) -> Tuple[int, int]:
    if int(answer) <= 0:
        raise ValueError("answer must be positive")
    if int(answer) == 1:
        return 1, 0
    source_count = int(rng.randint(1, int(answer) - 1))
    existing_target_count = int(answer) - int(source_count)
    return int(source_count), int(existing_target_count)


def _sample_counterfactual_spec(
    *,
    run_namespace: str,
    prompt_query_key: str,
    edit_kind: str,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
) -> _CounterfactualSampleSpec:
    """Sample one feasible counterfactual named-field scene.

    The public task fixes the edit kind; this helper constructs only visible
    pre-edit icons whose post-edit counted set has the requested answer.
    """

    if str(edit_kind) not in set(COUNTERFACTUAL_EDIT_KINDS):
        raise ValueError(f"unsupported counterfactual edit kind: {edit_kind}")
    rng = spawn_rng(int(instance_seed), f"{run_namespace}.sample")
    shape_support = _shape_support(params, gen_defaults, min_count=6)
    fill_style_support = resolve_named_icon_fill_style_support(
        params,
        gen_defaults,
        fallback_support=_COUNTERFACTUAL_DEFAULTS.named_icon_fill_style_support,
    )
    fill_style_probabilities = resolve_named_icon_fill_style_probabilities(params, gen_defaults, fill_style_support)
    arrangement_support = _arrangement_mode_support(params, render_defaults, _COUNTERFACTUAL_DEFAULTS.named_icon_layout_modes)
    answer_min, answer_max = resolve_named_icon_int_bounds(
        params,
        gen_defaults,
        "target_count_min",
        "target_count_max",
        _COUNTERFACTUAL_DEFAULTS.target_count_min,
        _COUNTERFACTUAL_DEFAULTS.target_count_max,
    )
    removal_min, removal_max = resolve_named_icon_int_bounds(
        params,
        gen_defaults,
        "removal_count_min",
        "removal_count_max",
        _COUNTERFACTUAL_DEFAULTS.removal_count_min,
        _COUNTERFACTUAL_DEFAULTS.removal_count_max,
    )
    distractor_min, distractor_max = resolve_named_icon_int_bounds(
        params,
        gen_defaults,
        "distractor_count_min",
        "distractor_count_max",
        _COUNTERFACTUAL_DEFAULTS.distractor_count_min,
        _COUNTERFACTUAL_DEFAULTS.distractor_count_max,
    )
    if answer_min < 1:
        raise ValueError("named-icon counterfactual count uses target_count_min >= 1")
    answer_support = tuple(range(int(answer_min), int(answer_max) + 1))
    removal_support = tuple(range(int(removal_min), int(removal_max) + 1))
    distractor_support = tuple(range(int(distractor_min), int(distractor_max) + 1))

    explicit_answer = params.get("target_count", params.get("target_answer"))
    if explicit_answer is not None:
        target_answer = int(explicit_answer)
        if target_answer not in set(answer_support):
            raise ValueError(f"target answer must be in {answer_support}")
    else:
        target_answer = int(rng.choice(answer_support))

    explicit_arrangement = params.get("arrangement_mode", params.get("layout_mode"))
    if explicit_arrangement is not None:
        arrangement_mode = str(explicit_arrangement)
        if arrangement_mode not in set(arrangement_support):
            raise ValueError(f"arrangement_mode must be one of {arrangement_support}")
    else:
        arrangement_mode = str(rng.choice(arrangement_support))

    semantic_specs: list[_CounterfactualIconSemanticSpec] = []
    source_shape_id = ""
    target_shape_id = ""
    remove_shape_id = ""
    source_count = 0
    existing_target_count = 0
    removal_count = 0
    distractor_count = 0

    if edit_kind == COUNTERFACTUAL_SHAPE_REPLACEMENT:
        source_shape_id, target_shape_id = _counterfactual_other_shapes(rng, shape_support, (), count=2)
        source_count, existing_target_count = _counterfactual_split_answer_into_source_and_target(rng, int(target_answer))
        distractor_count = int(rng.choice(distractor_support))
        for _ in range(int(source_count)):
            semantic_specs.append(
                _CounterfactualIconSemanticSpec(
                    shape_id=str(source_shape_id),
                    counterfactual_role="source_shape_changed_to_target",
                    counted_after_edit=True,
                )
            )
        for _ in range(int(existing_target_count)):
            semantic_specs.append(
                _CounterfactualIconSemanticSpec(
                    shape_id=str(target_shape_id),
                    counterfactual_role="existing_target_shape",
                    counted_after_edit=True,
                )
            )
        pool = _counterfactual_other_shapes(
            rng,
            shape_support,
            (source_shape_id, target_shape_id),
            count=min(4, len(shape_support) - 2),
        )
        for shape_id in rng.choices(pool, k=int(distractor_count)):
            semantic_specs.append(
                _CounterfactualIconSemanticSpec(
                    shape_id=str(shape_id),
                    counterfactual_role="unaffected_distractor",
                    counted_after_edit=False,
                )
            )
    elif edit_kind == COUNTERFACTUAL_SHAPE_REMOVAL:
        remove_shape_id = str(rng.choice(shape_support))
        removal_count = int(rng.choice(removal_support))
        remaining_pool = _counterfactual_other_shapes(rng, shape_support, (remove_shape_id,), count=min(5, len(shape_support) - 1))
        for shape_id in rng.choices(remaining_pool, k=int(target_answer)):
            semantic_specs.append(
                _CounterfactualIconSemanticSpec(
                    shape_id=str(shape_id),
                    counterfactual_role="remaining_after_removal",
                    counted_after_edit=True,
                )
            )
        for _ in range(int(removal_count)):
            semantic_specs.append(
                _CounterfactualIconSemanticSpec(
                    shape_id=str(remove_shape_id),
                    counterfactual_role="removed_shape",
                    counted_after_edit=False,
                )
            )
    else:
        raise ValueError(f"unsupported counterfactual edit kind: {edit_kind}")

    rng.shuffle(semantic_specs)
    object_count = len(semantic_specs)
    if sum(1 for spec in semantic_specs if spec.counted_after_edit) != int(target_answer):
        raise RuntimeError("counterfactual construction did not match target answer")
    if target_shape_id:
        target_shape_name = procedural_named_icon_display_name(str(target_shape_id))
    elif edit_kind == COUNTERFACTUAL_SHAPE_REMOVAL:
        target_shape_name = ""
    else:
        raise RuntimeError("target shape missing for target-count query")
    return _CounterfactualSampleSpec(
        prompt_query_key=str(prompt_query_key),
        edit_kind=str(edit_kind),
        target_answer=int(target_answer),
        object_count=int(object_count),
        target_shape_id=str(target_shape_id),
        target_shape_name=str(target_shape_name),
        source_shape_id=str(source_shape_id),
        source_shape_name=procedural_named_icon_display_name(str(source_shape_id)) if source_shape_id else "",
        remove_shape_id=str(remove_shape_id),
        remove_shape_name=procedural_named_icon_display_name(str(remove_shape_id)) if remove_shape_id else "",
        source_count=int(source_count),
        existing_target_count=int(existing_target_count),
        removal_count=int(removal_count),
        distractor_count=int(distractor_count),
        arrangement_mode=str(arrangement_mode),
        semantic_specs=tuple(semantic_specs),
        shape_probabilities=uniform_string_probability_map(shape_support),
        target_count_probabilities=dict(
            uniform_probability_map(answer_support, selected=int(target_answer) if explicit_answer is not None else None)
        ),
        removal_count_probabilities=dict(uniform_probability_map(removal_support)),
        distractor_count_probabilities=dict(uniform_probability_map(distractor_support)),
        arrangement_mode_probabilities=uniform_string_probability_map(
            arrangement_support,
            selected=str(arrangement_mode) if explicit_arrangement is not None else None,
        ),
        fill_style_support=tuple(fill_style_support),
        fill_style_probabilities=dict(fill_style_probabilities),
    )



sample_boolean_spec = _sample_boolean_spec
sample_counterfactual_spec = _sample_counterfactual_spec
shape_support = _shape_support
color_support = _color_support
arrangement_mode_support = _arrangement_mode_support


def _int_profile_value(profile: Mapping[str, Any], key: str, fallback: int) -> int:
    value = profile.get(str(key), fallback)
    return int(value)


def shape_count_arrangement_profiles(params: Mapping[str, Any], gen_defaults: Mapping[str, Any]) -> Dict[str, Dict[str, int]]:
    """Resolve arrangement-specific count ranges for direct shape-count tasks."""

    profiles = {
        str(key): {str(k): int(v) for k, v in value.items()}
        for key, value in _SHAPE_COUNT_ARRANGEMENT_PROFILES.items()
    }
    raw = params.get("named_icon_arrangement_profiles", group_default(gen_defaults, "named_icon_arrangement_profiles", {}))
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            if not isinstance(value, Mapping):
                continue
            current = dict(profiles.get(str(key), {}))
            for field, field_value in value.items():
                current[str(field)] = int(field_value)
            profiles[str(key)] = current
    return profiles


def shape_count_arrangement_mode_support(
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> Tuple[str, ...]:
    """Resolve supported layout modes for direct shape-count tasks."""

    raw = params.get(
        "named_icon_layout_modes",
        group_default(render_defaults, "named_icon_layout_modes", _SHAPE_COUNT_DEFAULTS.named_icon_layout_modes),
    )
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        values = tuple(str(value) for value in _SHAPE_COUNT_DEFAULTS.named_icon_layout_modes)
    else:
        values = tuple(str(value) for value in raw if str(value).strip())
    supported = set(shape_count_arrangement_profiles(params, gen_defaults))
    modes = tuple(value for value in dict.fromkeys(values) if value in supported)
    if not modes:
        raise ValueError("named_icon_layout_modes resolved no supported arrangement modes")
    return modes


def _sample_limited_distractor_pool(
    rng,
    *,
    support: Sequence[str],
    target_shape_id: str,
    min_groups: int,
    max_groups: int,
) -> Tuple[str, ...]:
    distractor_pool = [str(value) for value in support if str(value) != str(target_shape_id)]
    if len(distractor_pool) < 4:
        raise ValueError("named-shape count needs at least four distractor shapes")
    rng.shuffle(distractor_pool)
    group_count = int(rng.randint(max(1, int(min_groups)), max(max(1, int(min_groups)), min(int(max_groups), len(distractor_pool)))))
    return tuple(distractor_pool[: int(group_count)])


def sample_shape_count_spec(
    *,
    run_namespace: str,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
) -> ShapeCountSampleSpec:
    """Resolve direct named-shape count axes and construct a symbolic scene."""

    rng = spawn_rng(int(instance_seed), f"{run_namespace}:sample")
    support = _shape_support(params, gen_defaults)
    fill_style_support = resolve_named_icon_fill_style_support(
        params,
        gen_defaults,
        fallback_support=_SHAPE_COUNT_DEFAULTS.named_icon_fill_style_support,
    )
    fill_style_probabilities = resolve_named_icon_fill_style_probabilities(params, gen_defaults, fill_style_support)
    profiles = shape_count_arrangement_profiles(params, gen_defaults)
    arrangement_support = shape_count_arrangement_mode_support(params, render_defaults, gen_defaults)
    explicit_arrangement = params.get("arrangement_mode", params.get("layout_mode"))
    if explicit_arrangement is not None:
        arrangement_mode = str(explicit_arrangement)
        if arrangement_mode not in profiles:
            raise ValueError(f"unsupported named-icon arrangement mode: {arrangement_mode}")
    else:
        arrangement_mode = str(rng.choice(arrangement_support))
    profile = dict(profiles[str(arrangement_mode)])
    target_min = _int_profile_value(profile, "target_count_min", _SHAPE_COUNT_DEFAULTS.target_count_min)
    target_max = _int_profile_value(profile, "target_count_max", _SHAPE_COUNT_DEFAULTS.target_count_max)
    object_min = _int_profile_value(profile, "object_count_min", _SHAPE_COUNT_DEFAULTS.object_count_min)
    object_max = _int_profile_value(profile, "object_count_max", _SHAPE_COUNT_DEFAULTS.object_count_max)
    if "target_count_min" in params or "target_count_max" in params:
        target_min, target_max = resolve_named_icon_int_bounds(params, gen_defaults, "target_count_min", "target_count_max", target_min, target_max)
    if "object_count_min" in params or "object_count_max" in params:
        object_min, object_max = resolve_named_icon_int_bounds(params, gen_defaults, "object_count_min", "object_count_max", object_min, object_max)
    if target_min < 1:
        raise ValueError("named-shape count uses target_count_min >= 1 so every queried shape is visible")

    explicit_shape = params.get("shape_id", params.get("target_shape_id"))
    if explicit_shape is not None:
        target_shape_id = str(explicit_shape)
        if target_shape_id not in set(support):
            raise ValueError(f"target shape must be one of {support}")
    else:
        target_shape_id = str(rng.choice(support))

    target_support = tuple(range(int(target_min), int(target_max) + 1))
    explicit_target = params.get("target_count")
    if explicit_target is not None:
        target_count = int(explicit_target)
        if target_count < 1:
            raise ValueError("target_count must be positive")
    else:
        target_count = int(rng.choice(target_support))

    explicit_object_count = params.get("object_count")
    oddball_count = 1 if str(arrangement_mode) == "target_stack_with_oddballs" else 0
    if str(arrangement_mode) == "target_stack_with_oddballs":
        object_count = int(target_count) + int(oddball_count)
        object_support = (int(object_count),)
        if explicit_object_count is not None and explicit_arrangement is not None and int(explicit_object_count) != int(object_count):
            raise ValueError("target_stack_with_oddballs uses object_count = target_count + 1")
    else:
        min_object_count = max(int(object_min), int(target_count) + 4)
        object_support = tuple(range(int(min_object_count), int(object_max) + 1))
        if not object_support:
            raise ValueError("object_count range leaves no room for named-shape distractors")
        if explicit_object_count is not None:
            object_count = int(explicit_object_count)
            if object_count < int(target_count) + 4 or object_count > int(object_max):
                raise ValueError("object_count is outside configured support")
        else:
            object_count = int(rng.choice(object_support))

    distractor_pool = tuple(str(value) for value in support if str(value) != str(target_shape_id))
    if len(distractor_pool) < 4:
        raise ValueError("named-shape count needs at least four distractor shapes")
    stack_modes = {"shape_stacks", "target_stack_with_oddballs", "mixed_stacks"}
    if str(arrangement_mode) == "target_stack_with_oddballs":
        distractor_pool = (str(rng.choice(distractor_pool)),)
    elif str(arrangement_mode) in stack_modes:
        min_groups = int(params.get("named_icon_stack_distractor_group_min", group_default(gen_defaults, "named_icon_stack_distractor_group_min", _SHAPE_COUNT_DEFAULTS.named_icon_stack_distractor_group_min)))
        max_groups = int(params.get("named_icon_stack_distractor_group_max", group_default(gen_defaults, "named_icon_stack_distractor_group_max", _SHAPE_COUNT_DEFAULTS.named_icon_stack_distractor_group_max)))
        distractor_pool = _sample_limited_distractor_pool(
            rng,
            support=support,
            target_shape_id=str(target_shape_id),
            min_groups=int(min_groups),
            max_groups=int(max_groups),
        )
    target_group = f"target_stack:{target_shape_id}" if str(arrangement_mode) == "target_stack_with_oddballs" else str(target_shape_id)
    shape_ids: list[str] = []
    placement_groups: list[str] = []
    for _ in range(int(target_count)):
        shape_ids.append(str(target_shape_id))
        placement_groups.append(str(target_group) if str(arrangement_mode) in stack_modes else "")
    for _ in range(int(oddball_count)):
        oddball_shape = str(rng.choice(distractor_pool))
        shape_ids.append(oddball_shape)
        placement_groups.append(str(target_group))
    extra_distractor_count = 0 if str(arrangement_mode) == "target_stack_with_oddballs" else int(object_count) - int(target_count)
    for _ in range(int(extra_distractor_count)):
        shape_id = str(rng.choice(distractor_pool))
        shape_ids.append(shape_id)
        placement_groups.append(str(shape_id) if str(arrangement_mode) in stack_modes else "")
    paired = list(zip(shape_ids, placement_groups))
    rng.shuffle(paired)
    shape_ids = [str(shape_id) for shape_id, _group in paired]
    placement_groups = [str(group) for _shape_id, group in paired]
    arrangement_details = {
        "mode": str(arrangement_mode),
        "oddball_count": int(oddball_count),
        "target_stack_total": int(target_count + oddball_count) if str(arrangement_mode) == "target_stack_with_oddballs" else None,
        "stack_distractor_shape_count": len(set(distractor_pool)) if str(arrangement_mode) in stack_modes else None,
    }
    return ShapeCountSampleSpec(
        arrangement_mode=str(arrangement_mode),
        target_shape_id=str(target_shape_id),
        target_shape_name=procedural_named_icon_display_name(str(target_shape_id)),
        target_count=int(target_count),
        object_count=int(object_count),
        shape_ids=tuple(str(value) for value in shape_ids),
        placement_groups=tuple(str(value) for value in placement_groups),
        arrangement_details=dict(arrangement_details),
        arrangement_mode_probabilities=uniform_string_probability_map(arrangement_support, selected=str(arrangement_mode) if explicit_arrangement is not None else None),
        shape_probabilities=uniform_string_probability_map(support, selected=str(target_shape_id) if explicit_shape is not None else None),
        target_count_probabilities=dict(uniform_probability_map(target_support, selected=int(target_count) if explicit_target is not None else None)),
        object_count_probabilities=dict(uniform_probability_map(object_support, selected=int(object_count) if explicit_object_count is not None else None)),
        fill_style_support=tuple(fill_style_support),
        fill_style_probabilities=dict(fill_style_probabilities),
    )


def _choose_other_value(rng, values: Sequence[str], excluded: Sequence[str]) -> str:
    excluded_set = {str(value) for value in excluded}
    candidates = [str(value) for value in values if str(value) not in excluded_set]
    if not candidates:
        raise ValueError("no alternate value available")
    return str(rng.choice(candidates))


def _pair_arithmetic_answer_support(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    *,
    operation: str,
) -> Tuple[int, ...]:
    if str(operation) == "total":
        low, high = resolve_named_icon_int_bounds(
            params,
            gen_defaults,
            "total_answer_min",
            "total_answer_max",
            _PAIR_ARITHMETIC_DEFAULTS.total_answer_min,
            _PAIR_ARITHMETIC_DEFAULTS.total_answer_max,
        )
    elif str(operation) == "absolute_difference":
        low, high = resolve_named_icon_int_bounds(
            params,
            gen_defaults,
            "difference_answer_min",
            "difference_answer_max",
            _PAIR_ARITHMETIC_DEFAULTS.difference_answer_min,
            _PAIR_ARITHMETIC_DEFAULTS.difference_answer_max,
        )
    else:
        raise ValueError(f"unsupported pair arithmetic operation: {operation}")
    return tuple(range(int(low), int(high) + 1))


def _feasible_pair_arithmetic_operand_pairs(
    *,
    operation: str,
    target_answer: int,
    operand_min: int,
    operand_max: int,
) -> Tuple[Tuple[int, int], ...]:
    pairs: list[Tuple[int, int]] = []
    for left in range(int(operand_min), int(operand_max) + 1):
        for right in range(int(operand_min), int(operand_max) + 1):
            if str(operation) == "total" and int(left) + int(right) == int(target_answer):
                pairs.append((int(left), int(right)))
            elif str(operation) == "absolute_difference" and abs(int(left) - int(right)) == int(target_answer):
                pairs.append((int(left), int(right)))
    return tuple(pairs)


def _sample_pair_arithmetic_operand_counts(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    rng,
    operation: str,
) -> Tuple[int, int, int, Dict[str, float], Dict[str, float]]:
    """Choose operand counts that exactly realize the requested arithmetic answer."""

    operand_min, operand_max = resolve_named_icon_int_bounds(
        params,
        gen_defaults,
        "operand_count_min",
        "operand_count_max",
        _PAIR_ARITHMETIC_DEFAULTS.operand_count_min,
        _PAIR_ARITHMETIC_DEFAULTS.operand_count_max,
    )
    if operand_min < 1:
        raise ValueError("named-icon pair arithmetic uses operand_count_min >= 1")
    answer_support = _pair_arithmetic_answer_support(params, gen_defaults, operation=str(operation))
    feasible_answer_support = tuple(
        int(answer)
        for answer in answer_support
        if _feasible_pair_arithmetic_operand_pairs(
            operation=str(operation),
            target_answer=int(answer),
            operand_min=int(operand_min),
            operand_max=int(operand_max),
        )
    )
    if not feasible_answer_support:
        raise ValueError("answer support has no feasible operand-count pairs")
    answer_probabilities = weighted_probability_map(
        feasible_answer_support,
        params.get("answer_weights", group_default(gen_defaults, "answer_weights", None)),
    )
    explicit_answer = params.get("target_answer", params.get("answer"))
    if explicit_answer is not None:
        target_answer = int(explicit_answer)
        if target_answer not in set(feasible_answer_support):
            raise ValueError(f"target_answer must be in feasible support {feasible_answer_support}")
    else:
        target_answer = int(sample_weighted_value(rng, feasible_answer_support, answer_probabilities))

    explicit_left = params.get("left_count")
    explicit_right = params.get("right_count")
    if explicit_left is not None or explicit_right is not None:
        if explicit_left is None or explicit_right is None:
            raise ValueError("left_count and right_count must be provided together")
        left_count = int(explicit_left)
        right_count = int(explicit_right)
        if not (int(operand_min) <= left_count <= int(operand_max) and int(operand_min) <= right_count <= int(operand_max)):
            raise ValueError("left_count/right_count outside operand count support")
        expected = left_count + right_count if str(operation) == "total" else abs(left_count - right_count)
        if int(expected) != int(target_answer):
            raise ValueError("left_count/right_count do not match target_answer")
    else:
        pairs = _feasible_pair_arithmetic_operand_pairs(
            operation=str(operation),
            target_answer=int(target_answer),
            operand_min=int(operand_min),
            operand_max=int(operand_max),
        )
        left_count, right_count = tuple(int(value) for value in rng.choice(pairs))

    operand_support = tuple(range(int(operand_min), int(operand_max) + 1))
    return (
        int(left_count),
        int(right_count),
        int(target_answer),
        dict(uniform_probability_map(feasible_answer_support, selected=int(target_answer)) if explicit_answer is not None else answer_probabilities),
        dict(uniform_probability_map(operand_support)),
    )


def _pair_arithmetic_operand_specs(
    *,
    params: Mapping[str, Any],
    rng,
    shape_values: Sequence[str],
    color_values: Sequence[_NamedColorEntry],
    uses_color_binding: bool,
) -> Tuple[PairArithmeticOperandSpec, PairArithmeticOperandSpec, Dict[str, float], Dict[str, float]]:
    """Choose the two visible operand descriptions and their optional color bindings."""

    explicit_left_shape = params.get("left_shape_id")
    explicit_right_shape = params.get("right_shape_id")
    if explicit_left_shape is not None:
        left_shape_id = str(explicit_left_shape)
        if left_shape_id not in set(shape_values):
            raise ValueError(f"left_shape_id must be one of {tuple(shape_values)}")
    else:
        left_shape_id = str(rng.choice(tuple(shape_values)))
    if explicit_right_shape is not None:
        right_shape_id = str(explicit_right_shape)
        if right_shape_id not in set(shape_values):
            raise ValueError(f"right_shape_id must be one of {tuple(shape_values)}")
        if right_shape_id == left_shape_id:
            raise ValueError("left_shape_id and right_shape_id must be distinct")
    else:
        right_shape_id = _choose_other_value(rng, shape_values, (left_shape_id,))

    color_by_name = {str(entry.name): entry for entry in color_values}
    color_names = tuple(color_by_name)
    explicit_left_color = params.get("left_color_name")
    explicit_right_color = params.get("right_color_name")
    if uses_color_binding:
        if explicit_left_color is not None:
            left_color_name = str(explicit_left_color).strip().lower()
            if left_color_name not in color_by_name:
                raise ValueError(f"left_color_name must be one of {color_names}")
        else:
            left_color_name = str(rng.choice(color_values).name)
        if explicit_right_color is not None:
            right_color_name = str(explicit_right_color).strip().lower()
            if right_color_name not in color_by_name:
                raise ValueError(f"right_color_name must be one of {color_names}")
            if right_color_name == left_color_name:
                raise ValueError("left_color_name and right_color_name must be distinct for color-bound queries")
        else:
            right_color_name = _choose_other_value(rng, color_names, (left_color_name,))
    else:
        left_color_name = ""
        right_color_name = ""

    def make_operand(shape_id: str, color_name: str) -> PairArithmeticOperandSpec:
        shape_name = procedural_named_icon_display_name(str(shape_id))
        if str(color_name):
            color_label = str(color_by_name[str(color_name)].label)
            label = f'{color_label} "{shape_name}"'
        else:
            color_label = ""
            label = str(shape_name)
        return PairArithmeticOperandSpec(
            shape_id=str(shape_id),
            shape_name=str(shape_name),
            color_name=str(color_name),
            color_label=str(color_label),
            label=str(label),
        )

    return (
        make_operand(str(left_shape_id), str(left_color_name)),
        make_operand(str(right_shape_id), str(right_color_name)),
        uniform_string_probability_map(shape_values),
        uniform_string_probability_map(color_names),
    )


def _pair_arithmetic_matches_operand(
    spec: PairArithmeticIconSemanticSpec,
    operand: PairArithmeticOperandSpec,
    *,
    uses_color_binding: bool,
) -> bool:
    if str(spec.shape_id) != str(operand.shape_id):
        return False
    if bool(uses_color_binding):
        return str(spec.color_name) == str(operand.color_name)
    return True


def _pair_arithmetic_semantic_specs(
    *,
    rng,
    left_operand: PairArithmeticOperandSpec,
    right_operand: PairArithmeticOperandSpec,
    left_count: int,
    right_count: int,
    distractor_count: int,
    uses_color_binding: bool,
    shape_values: Sequence[str],
    color_values: Sequence[_NamedColorEntry],
    fill_style_support: Sequence[str],
    fill_style_probabilities: Mapping[str, float],
) -> Tuple[PairArithmeticIconSemanticSpec, ...]:
    """Build operand and distractor icon semantics while preserving operand counts."""

    color_names = tuple(str(entry.name) for entry in color_values)
    specs: list[PairArithmeticIconSemanticSpec] = []

    def random_fill_style() -> str:
        return sample_procedural_named_icon_fill_style(
            rng,
            support=fill_style_support,
            probabilities=dict(fill_style_probabilities),
        )

    def random_color(excluded: Sequence[str] = ()) -> str:
        return _choose_other_value(rng, color_names, tuple(str(value) for value in excluded))

    for _ in range(int(left_count)):
        specs.append(
            PairArithmeticIconSemanticSpec(
                shape_id=str(left_operand.shape_id),
                color_name=str(left_operand.color_name or random_color()),
                fill_style=random_fill_style(),
                role="left_operand",
            )
        )
    for _ in range(int(right_count)):
        specs.append(
            PairArithmeticIconSemanticSpec(
                shape_id=str(right_operand.shape_id),
                color_name=str(right_operand.color_name or random_color()),
                fill_style=random_fill_style(),
                role="right_operand",
            )
        )

    for _ in range(int(distractor_count)):
        if bool(uses_color_binding):
            shape_id = str(rng.choice(tuple(shape_values)))
            color_name = str(rng.choice(color_names))
            for _attempt in range(40):
                candidate = PairArithmeticIconSemanticSpec(
                    shape_id=str(shape_id),
                    color_name=str(color_name),
                    fill_style=random_fill_style(),
                    role="distractor",
                )
                if not _pair_arithmetic_matches_operand(candidate, left_operand, uses_color_binding=True) and not _pair_arithmetic_matches_operand(
                    candidate,
                    right_operand,
                    uses_color_binding=True,
                ):
                    specs.append(candidate)
                    break
                shape_id = str(rng.choice(tuple(shape_values)))
                color_name = str(rng.choice(color_names))
            else:
                raise RuntimeError("could not sample color-bound distractor")
        else:
            specs.append(
                PairArithmeticIconSemanticSpec(
                    shape_id=_choose_other_value(rng, shape_values, (left_operand.shape_id, right_operand.shape_id)),
                    color_name=str(rng.choice(color_names)),
                    fill_style=random_fill_style(),
                    role="distractor",
                )
            )

    rng.shuffle(specs)
    return tuple(specs)


def sample_pair_arithmetic_spec(
    *,
    run_namespace: str,
    query_key: str,
    operation: str,
    uses_color_binding: bool,
    query_probabilities: Mapping[str, float],
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
) -> PairArithmeticSampleSpec:
    """Sample a feasible two-operand named-field arithmetic plan."""

    rng = spawn_rng(int(instance_seed), f"{run_namespace}:sample")
    shape_values = _shape_support(params, gen_defaults, min_count=3)
    color_values = _color_support(params, gen_defaults)
    fill_style_support = resolve_named_icon_fill_style_support(
        params,
        gen_defaults,
        fallback_support=_PAIR_ARITHMETIC_DEFAULTS.named_icon_fill_style_support,
    )
    fill_style_probabilities = resolve_named_icon_fill_style_probabilities(params, gen_defaults, fill_style_support)
    arrangement_support = _arrangement_mode_support(params, render_defaults, _PAIR_ARITHMETIC_DEFAULTS.named_icon_layout_modes)

    left_count, right_count, target_answer, answer_probabilities, operand_count_probabilities = _sample_pair_arithmetic_operand_counts(
        params=params,
        gen_defaults=gen_defaults,
        rng=rng,
        operation=str(operation),
    )
    left_operand, right_operand, shape_probabilities, color_probabilities = _pair_arithmetic_operand_specs(
        params=params,
        rng=rng,
        shape_values=shape_values,
        color_values=color_values,
        uses_color_binding=bool(uses_color_binding),
    )

    distractor_min, distractor_max = resolve_named_icon_int_bounds(
        params,
        gen_defaults,
        "distractor_count_min",
        "distractor_count_max",
        _PAIR_ARITHMETIC_DEFAULTS.distractor_count_min,
        _PAIR_ARITHMETIC_DEFAULTS.distractor_count_max,
    )
    explicit_distractor_count = params.get("distractor_count")
    distractor_support = tuple(range(int(distractor_min), int(distractor_max) + 1))
    if explicit_distractor_count is not None:
        distractor_count = int(explicit_distractor_count)
        if distractor_count not in set(distractor_support):
            raise ValueError(f"distractor_count must be in {distractor_support}")
    else:
        distractor_count = int(rng.choice(distractor_support))

    semantic_specs = _pair_arithmetic_semantic_specs(
        rng=rng,
        left_operand=left_operand,
        right_operand=right_operand,
        left_count=int(left_count),
        right_count=int(right_count),
        distractor_count=int(distractor_count),
        uses_color_binding=bool(uses_color_binding),
        shape_values=shape_values,
        color_values=color_values,
        fill_style_support=fill_style_support,
        fill_style_probabilities=fill_style_probabilities,
    )

    explicit_arrangement = params.get("arrangement_mode", params.get("layout_mode"))
    if explicit_arrangement is not None:
        arrangement_mode = str(explicit_arrangement)
        if arrangement_mode not in set(arrangement_support):
            raise ValueError(f"named-icon pair arithmetic only supports non-stack layouts: {arrangement_support}")
    else:
        arrangement_mode = str(rng.choice(arrangement_support))

    return PairArithmeticSampleSpec(
        query_key=str(query_key),
        operation=str(operation),
        uses_color_binding=bool(uses_color_binding),
        left_operand=left_operand,
        right_operand=right_operand,
        left_count=int(left_count),
        right_count=int(right_count),
        target_answer=int(target_answer),
        distractor_count=int(distractor_count),
        object_count=int(len(semantic_specs)),
        arrangement_mode=str(arrangement_mode),
        semantic_specs=tuple(semantic_specs),
        query_probabilities=dict(query_probabilities),
        shape_probabilities=dict(shape_probabilities),
        color_probabilities=dict(color_probabilities),
        answer_probabilities=dict(answer_probabilities),
        operand_count_probabilities=dict(operand_count_probabilities),
        distractor_count_probabilities=dict(uniform_probability_map(distractor_support, selected=int(distractor_count) if explicit_distractor_count is not None else None)),
        fill_style_support=tuple(fill_style_support),
        fill_style_probabilities=dict(fill_style_probabilities),
        arrangement_mode_probabilities=uniform_string_probability_map(arrangement_support, selected=str(arrangement_mode) if explicit_arrangement is not None else None),
    )


__all__ = [
    "arrangement_mode_support",
    "color_support",
    "sample_boolean_spec",
    "sample_counterfactual_spec",
    "sample_pair_arithmetic_spec",
    "sample_shape_count_spec",
    "shape_support",
    "shape_count_arrangement_mode_support",
    "shape_count_arrangement_profiles",
]
