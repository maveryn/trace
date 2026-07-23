"""Internal count-relation builders for object-cluster tasks."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.color_format import format_named_color_with_hex
from trace_tasks.tasks.three_d.shared.task_support import resolve_axis_variant_for_namespace
from trace_tasks.tasks.three_d.shared.semantic_colors import (
    colors_conflict as shared_colors_conflict,
    confusable_color_names as shared_confusable_color_names,
)

from .defaults import (
    CLUSTER_COMPOSITION_MODES,
    COLOR_READOUT_CLUSTER_SHAPE_TYPES,
    COLOR_SAFE_CLUSTER_SHAPE_TYPES,
    NAMED_CLUSTER_SHAPE_TYPES,
    PROMPT_COLOR_RGB,
    compatible_distractor_pool,
    object_name_for_shape,
    object_plural,
    shape_plural,
)
from .sampling import (
    configured_int,
    count_bounds,
    one_hot_int_probability_map,
    resolve_named_choice,
    resolve_string_subset,
    resolve_uniform_count,
    resolve_weighted_count,
    selected_probability_map,
    uniform_string_probability_map,
)
from .state import ClusterSequenceItem, PredicateTarget


def color_support() -> Tuple[str, ...]:
    """Return semantic color labels supported by object-cluster prompts."""

    return tuple(str(color) for color in PROMPT_COLOR_RGB)


def semantic_color_label(color_name: str) -> str:
    """Return the visible prompt label for one semantic color."""

    rgb = PROMPT_COLOR_RGB[str(color_name)]
    return format_named_color_with_hex(str(color_name), rgb)


def _confusable_color_names(color_name: str) -> Tuple[str, ...]:
    """Return generated color distractors too close to one semantic target."""

    return shared_confusable_color_names(str(color_name))


def _colors_conflict(left: str, right: str) -> bool:
    """Return whether two semantic colors are too visually close for readout."""

    return shared_colors_conflict(str(left), str(right))


def readout_color_support(*, anchors: Sequence[str] = (), exclude: Sequence[str] = ()) -> Tuple[str, ...]:
    """Return semantic colors after removing target-confusable distractors."""

    blocked = {str(color) for color in exclude}
    for anchor in anchors:
        blocked.add(str(anchor))
        blocked.update(_confusable_color_names(str(anchor)))
    pool = tuple(str(color) for color in color_support() if str(color) not in blocked)
    if not pool:
        raise ValueError("object-cluster color readout needs at least one compatible color")
    return pool


def _sample_nonconflicting_readout_colors(*, instance_seed: int, namespace: str, count: int) -> Tuple[str, ...]:
    """Sample generated semantic colors with no pairwise near-color conflicts."""

    if int(count) < 1:
        raise ValueError("color readout needs at least one color")
    rng = spawn_rng(int(instance_seed), f"{namespace}.nonconflicting_colors")
    candidates = list(color_support())
    for _attempt in range(96):
        rng.shuffle(candidates)
        selected: list[str] = []
        for color in candidates:
            if all(not _colors_conflict(str(color), chosen) for chosen in selected):
                selected.append(str(color))
            if len(selected) >= int(count):
                return tuple(selected[: int(count)])
    raise ValueError("could not sample enough non-conflicting semantic colors")


def safe_shape_support() -> Tuple[str, ...]:
    """Return named object shapes available for shape/type questions."""

    return tuple(str(shape) for shape in COLOR_SAFE_CLUSTER_SHAPE_TYPES)


def named_shape_support() -> Tuple[str, ...]:
    """Return the scene-approved named object pool for object-cluster tasks."""

    return tuple(str(shape) for shape in NAMED_CLUSTER_SHAPE_TYPES)


def color_readout_shape_support() -> Tuple[str, ...]:
    """Return object shapes suitable for generated semantic-color questions."""

    return tuple(str(shape) for shape in COLOR_READOUT_CLUSTER_SHAPE_TYPES)


def random_color(rng) -> str:
    """Sample one semantic color for non-color-targeted objects."""

    return str(rng.choice(list(color_support())))


def sample_visual_color_sequence(
    *,
    rng,
    object_count: int,
    min_colors: int = 2,
    max_colors: int = 4,
) -> Tuple[Tuple[str, ...], Tuple[str, ...], Dict[str, int]]:
    """Assign non-semantic canonical colors across a count-only sequence."""

    count = int(object_count)
    if count < 1:
        raise ValueError("visual color sequence needs at least one object")
    support = list(color_support())
    rng.shuffle(support)
    active_count = min(count, max(int(min_colors), min(int(max_colors), 2 + int(rng.randrange(3)))))
    active_colors = tuple(str(color) for color in support[:active_count])
    colors = list(active_colors)
    for _ in range(max(0, count - active_count)):
        colors.append(str(rng.choice(active_colors)))
    rng.shuffle(colors)
    color_counts = Counter(str(color) for color in colors)
    return tuple(str(color) for color in colors), tuple(active_colors), {str(color): int(color_counts[str(color)]) for color in active_colors}


def target_mapping(target: PredicateTarget) -> Dict[str, Any]:
    """Convert an internal target object into trace/prompt metadata."""

    return {
        "mode": str(target.mode),
        "target_shape_type": target.target_shape_type,
        "target_color_name": target.target_color_name,
        "target_shape_types": list(target.target_shape_types),
        "singleton_shape_types": list(target.singleton_shape_types),
        "target_object_name": target.target_object_name,
        "target_object_plural": target.target_object_plural,
        "target_object_union_phrase": target.target_object_union_phrase,
        "target_color_names": list(target.target_color_names),
        "left_operand_phrase": target.left_operand_phrase,
        "right_operand_phrase": target.right_operand_phrase,
        "arithmetic_operation": target.arithmetic_operation,
        "left_operand_count": target.left_count,
        "right_operand_count": target.right_count,
        "target_property_phrase": target.target_property_phrase,
        **dict(target.extras),
    }


def resolve_composition_mode(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve homogeneous/mixed cluster composition from defaults or counts."""

    explicit_mode = params.get("composition_mode")
    explicit_object_count = params.get("object_count")
    explicit_target_count = params.get("target_count")
    if explicit_mode is None and explicit_object_count is not None and explicit_target_count is not None:
        object_count = int(explicit_object_count)
        target_count = int(explicit_target_count)
        distractor_count = int(object_count) - int(target_count)
        if distractor_count < 0:
            raise ValueError("object_count must be at least target_count")
        if distractor_count == 0:
            selected = "single_type_cluster"
        elif 1 <= distractor_count <= 4:
            selected = "near_homogeneous_cluster"
        else:
            selected = "mixed_type_cluster"
        return selected, uniform_string_probability_map(CLUSTER_COMPOSITION_MODES, selected=str(selected))

    return resolve_axis_variant_for_namespace(
        params,
        namespace=str(namespace),
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=CLUSTER_COMPOSITION_MODES,
        explicit_key="composition_mode",
        weights_key="composition_mode_weights",
        balance_flag_key="balanced_composition_mode_sampling",
    )


def resolve_membership_counts(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    composition_mode: str,
    instance_seed: int,
    namespace: str,
) -> Dict[str, Any]:
    """Resolve object/target/distractor counts for one type-membership cluster."""

    explicit_object_count = params.get("object_count")
    explicit_target_count = params.get("target_count")

    if str(composition_mode) == "single_type_cluster":
        minimum, maximum = count_bounds(
            params=params,
            gen_defaults=gen_defaults,
            minimum_key="single_type_count_min",
            maximum_key="single_type_count_max",
            fallback_minimum=configured_int(params, gen_defaults, "target_count_min", 4),
            fallback_maximum=configured_int(params, gen_defaults, "target_count_max", 25),
            lower=4,
            upper=25,
        )
        if explicit_object_count is not None and explicit_target_count is not None:
            if int(explicit_object_count) != int(explicit_target_count):
                raise ValueError("single-type clusters require matching object and target counts")
            selected = int(explicit_target_count)
            support = tuple(range(int(minimum), int(maximum) + 1))
            if selected not in set(support):
                raise ValueError("single-type explicit count is outside configured support")
            probabilities = one_hot_int_probability_map(support, selected=int(selected))
        else:
            selected, probabilities = resolve_weighted_count(
                params={**dict(params), "target_count": int(explicit_object_count)} if explicit_object_count is not None else params,
                gen_defaults=gen_defaults,
                instance_seed=int(instance_seed),
                explicit_key="target_count",
                weights_key="target_count_weights",
                minimum=int(minimum),
                maximum=int(maximum),
                namespace=f"{namespace}.single_type_count",
            )
        return {
            "object_count": int(selected),
            "target_count": int(selected),
            "distractor_count": 0,
            "object_count_probabilities": dict(probabilities),
            "target_count_probabilities": dict(probabilities),
            "distractor_count_probabilities": {"0": 1.0},
        }

    if str(composition_mode) == "near_homogeneous_cluster":
        target_minimum, target_maximum = count_bounds(
            params=params,
            gen_defaults=gen_defaults,
            minimum_key="near_homogeneous_target_count_min",
            maximum_key="near_homogeneous_target_count_max",
            fallback_minimum=configured_int(params, gen_defaults, "target_count_min", 4),
            fallback_maximum=configured_int(params, gen_defaults, "target_count_max", 25),
            lower=4,
            upper=25,
        )
        object_minimum, object_maximum = count_bounds(
            params=params,
            gen_defaults=gen_defaults,
            minimum_key="near_homogeneous_object_count_min",
            maximum_key="near_homogeneous_object_count_max",
            fallback_minimum=8,
            fallback_maximum=29,
            lower=7,
            upper=30,
        )
        distractor_minimum, distractor_maximum = count_bounds(
            params=params,
            gen_defaults=gen_defaults,
            minimum_key="near_homogeneous_distractor_count_min",
            maximum_key="near_homogeneous_distractor_count_max",
            fallback_minimum=1,
            fallback_maximum=4,
            lower=1,
            upper=4,
        )
        if explicit_object_count is not None and explicit_target_count is not None:
            object_count = int(explicit_object_count)
            target_count = int(explicit_target_count)
            distractor_count = int(object_count) - int(target_count)
            if not (int(target_minimum) <= target_count <= int(target_maximum)):
                raise ValueError("near-homogeneous explicit target count is outside configured support")
            if not (int(object_minimum) <= object_count <= int(object_maximum)):
                raise ValueError("near-homogeneous explicit object count is outside configured support")
            if not (int(distractor_minimum) <= distractor_count <= int(distractor_maximum)):
                raise ValueError("near-homogeneous clusters require 1-4 distractor objects")
            return {
                "object_count": int(object_count),
                "target_count": int(target_count),
                "distractor_count": int(distractor_count),
                "object_count_probabilities": one_hot_int_probability_map(range(object_minimum, object_maximum + 1), selected=int(object_count)),
                "target_count_probabilities": one_hot_int_probability_map(range(target_minimum, target_maximum + 1), selected=int(target_count)),
                "distractor_count_probabilities": one_hot_int_probability_map(range(distractor_minimum, distractor_maximum + 1), selected=int(distractor_count)),
            }

        target_count, target_probabilities = resolve_weighted_count(
            params=params,
            gen_defaults=gen_defaults,
            instance_seed=int(instance_seed),
            explicit_key="target_count",
            weights_key="target_count_weights",
            minimum=int(target_minimum),
            maximum=int(target_maximum),
            namespace=f"{namespace}.near_target",
        )
        feasible_minimum = max(int(distractor_minimum), int(object_minimum) - int(target_count))
        feasible_maximum = min(int(distractor_maximum), int(object_maximum) - int(target_count))
        distractor_count, distractor_probabilities = resolve_uniform_count(
            params=params,
            explicit_key="distractor_count",
            minimum=int(feasible_minimum),
            maximum=int(feasible_maximum),
            instance_seed=int(instance_seed),
            namespace=f"{namespace}.near_distractor",
        )
        object_count = int(target_count) + int(distractor_count)
        return {
            "object_count": int(object_count),
            "target_count": int(target_count),
            "distractor_count": int(distractor_count),
            "object_count_probabilities": one_hot_int_probability_map(range(object_minimum, object_maximum + 1), selected=int(object_count)),
            "target_count_probabilities": dict(target_probabilities),
            "distractor_count_probabilities": dict(distractor_probabilities),
        }

    target_minimum, target_maximum = count_bounds(
        params=params,
        gen_defaults=gen_defaults,
        minimum_key="mixed_type_target_count_min",
        maximum_key="mixed_type_target_count_max",
        fallback_minimum=4,
        fallback_maximum=18,
        lower=4,
        upper=25,
    )
    object_minimum, object_maximum = count_bounds(
        params=params,
        gen_defaults=gen_defaults,
        minimum_key="mixed_type_object_count_min",
        maximum_key="mixed_type_object_count_max",
        fallback_minimum=16,
        fallback_maximum=30,
        lower=12,
        upper=32,
    )
    minimum_mixed_distractors = configured_int(params, gen_defaults, "mixed_type_distractor_count_min", 5)
    if explicit_object_count is not None and explicit_target_count is not None:
        object_count = int(explicit_object_count)
        target_count = int(explicit_target_count)
        distractor_count = int(object_count) - int(target_count)
        if target_count < int(target_minimum) or target_count > int(target_maximum):
            raise ValueError("mixed explicit target count is outside configured support")
        if object_count < int(object_minimum) or object_count > int(object_maximum):
            raise ValueError("mixed explicit object count is outside configured support")
        if distractor_count < int(minimum_mixed_distractors):
            raise ValueError("mixed clusters require enough distractor objects")
        return {
            "object_count": int(object_count),
            "target_count": int(target_count),
            "distractor_count": int(distractor_count),
            "object_count_probabilities": one_hot_int_probability_map(range(object_minimum, object_maximum + 1), selected=int(object_count)),
            "target_count_probabilities": one_hot_int_probability_map(range(target_minimum, target_maximum + 1), selected=int(target_count)),
            "distractor_count_probabilities": {"derived": 1.0},
        }
    target_count, target_probabilities = resolve_weighted_count(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        explicit_key="target_count",
        weights_key="mixed_type_target_count_weights",
        minimum=int(target_minimum),
        maximum=int(target_maximum),
        namespace=f"{namespace}.mixed_target",
    )
    feasible_object_minimum = max(int(object_minimum), int(target_count) + int(minimum_mixed_distractors))
    object_count, object_probabilities = resolve_uniform_count(
        params=params,
        explicit_key="object_count",
        minimum=int(feasible_object_minimum),
        maximum=int(object_maximum),
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.mixed_object",
    )
    return {
        "object_count": int(object_count),
        "target_count": int(target_count),
        "distractor_count": int(object_count) - int(target_count),
        "object_count_probabilities": dict(object_probabilities),
        "target_count_probabilities": dict(target_probabilities),
        "distractor_count_probabilities": {"derived": 1.0},
    }


def build_total_sequence(
    *,
    shape_type: str,
    object_count: int,
    rng,
) -> tuple[list[ClusterSequenceItem], PredicateTarget]:
    """Build a one-type cluster where every object is counted."""

    color_sequence, visual_color_names, visual_color_counts = sample_visual_color_sequence(rng=rng, object_count=int(object_count))
    sequence = [
        ClusterSequenceItem(str(shape_type), str(color), True, "target")
        for color in color_sequence
    ]
    name = object_name_for_shape(str(shape_type))
    return sequence, PredicateTarget(
        mode="all_objects",
        target_shape_type=str(shape_type),
        target_object_name=str(name),
        target_object_plural=object_plural(str(name)),
        target_property_phrase="all objects",
        extras={
            "color_role": "non_semantic_visual_variation",
            "visual_color_names": list(visual_color_names),
            "visual_color_counts": dict(visual_color_counts),
        },
    )


def build_type_membership_sequence(
    *,
    shape_type: str,
    composition_mode: str,
    target_count: int,
    object_count: int,
    rng,
) -> tuple[list[ClusterSequenceItem], PredicateTarget]:
    """Build a target-type cluster with composition-controlled distractors."""

    sequence = [
        ClusterSequenceItem(str(shape_type), random_color(rng), True, "target")
        for _ in range(int(target_count))
    ]
    if str(composition_mode) != "single_type_cluster":
        distractor_pool = list(compatible_distractor_pool(str(shape_type), support=NAMED_CLUSTER_SHAPE_TYPES))
        while len(sequence) < int(object_count):
            sequence.append(ClusterSequenceItem(str(rng.choice(distractor_pool)), random_color(rng), False, "distractor"))
    rng.shuffle(sequence)
    name = object_name_for_shape(str(shape_type))
    return sequence, PredicateTarget(
        mode="by_type",
        target_shape_type=str(shape_type),
        target_object_name=str(name),
        target_object_plural=object_plural(str(name)),
        target_property_phrase=object_plural(str(name)),
        extras={
            "cluster_composition_mode": str(composition_mode),
            "distractor_count": int(object_count) - int(target_count),
            "cluster_object_pool_size": len(NAMED_CLUSTER_SHAPE_TYPES),
        },
    )


def build_type_and_color_sequence(
    *,
    shape_type: str,
    color_name: str,
    target_count: int,
    object_count: int,
    rng,
) -> tuple[list[ClusterSequenceItem], PredicateTarget]:
    """Build an exact type-and-color conjunction with structured distractors."""

    sequence = [
        ClusterSequenceItem(str(shape_type), str(color_name), True, "target")
        for _ in range(int(target_count))
    ]
    wrong_colors = list(readout_color_support(anchors=(str(color_name),)))
    wrong_shapes = list(compatible_distractor_pool(str(shape_type), support=color_readout_shape_support()))
    rng.shuffle(wrong_colors)
    rng.shuffle(wrong_shapes)
    if len(sequence) < int(object_count):
        sequence.append(ClusterSequenceItem(str(shape_type), str(wrong_colors[0]), False, "same_type_wrong_color"))
    if len(sequence) < int(object_count):
        sequence.append(ClusterSequenceItem(str(wrong_shapes[0]), str(color_name), False, "same_color_wrong_type"))
    while len(sequence) < int(object_count):
        shape = str(rng.choice(wrong_shapes))
        color = str(rng.choice(wrong_colors))
        sequence.append(ClusterSequenceItem(shape, color, False, "distractor"))
    rng.shuffle(sequence)
    name = object_name_for_shape(str(shape_type))
    plural = object_plural(str(name))
    phrase = f"{color_name} {plural}"
    prompt_phrase = f"{semantic_color_label(str(color_name))} {plural}"
    return sequence, PredicateTarget(
        mode="by_type_and_color",
        target_shape_type=str(shape_type),
        target_color_name=str(color_name),
        target_object_name=str(name),
        target_object_plural=str(plural),
        target_property_phrase=str(phrase),
        extras={"target_property_prompt_phrase": str(prompt_phrase)},
    )


def build_color_membership_sequence(
    *,
    color_name: str,
    target_count: int,
    object_count: int,
    rng,
) -> tuple[list[ClusterSequenceItem], PredicateTarget]:
    """Build a cluster where color alone determines membership."""

    shapes = list(color_readout_shape_support())
    wrong_colors = list(readout_color_support(anchors=(str(color_name),)))
    sequence = [
        ClusterSequenceItem(str(rng.choice(shapes)), str(color_name), True, "target")
        for _ in range(int(target_count))
    ]
    while len(sequence) < int(object_count):
        sequence.append(ClusterSequenceItem(str(rng.choice(shapes)), str(rng.choice(wrong_colors)), False, "distractor"))
    rng.shuffle(sequence)
    return sequence, PredicateTarget(
        mode="by_color",
        target_color_name=str(color_name),
        target_property_phrase=f"{color_name} objects",
        extras={"target_property_prompt_phrase": f"{semantic_color_label(str(color_name))} objects"},
    )


def build_or_sequence(
    *,
    shape_type: str,
    color_name: str,
    target_count: int,
    object_count: int,
    rng,
) -> tuple[list[ClusterSequenceItem], PredicateTarget]:
    """Build an inclusive OR set and ensure overlap is counted once."""

    if int(target_count) < 1:
        raise ValueError("inclusive OR count needs at least one target")
    wrong_shapes = list(compatible_distractor_pool(str(shape_type), support=color_readout_shape_support()))
    wrong_colors = list(readout_color_support(anchors=(str(color_name),)))
    sequence = [ClusterSequenceItem(str(shape_type), str(color_name), True, "target")]
    while len(sequence) < int(target_count):
        if len(sequence) % 2 == 0:
            sequence.append(ClusterSequenceItem(str(shape_type), str(rng.choice(wrong_colors)), True, "target"))
        else:
            sequence.append(ClusterSequenceItem(str(rng.choice(wrong_shapes)), str(color_name), True, "target"))
    while len(sequence) < int(object_count):
        sequence.append(ClusterSequenceItem(str(rng.choice(wrong_shapes)), str(rng.choice(wrong_colors)), False, "distractor"))
    rng.shuffle(sequence)
    name = object_name_for_shape(str(shape_type))
    plural = object_plural(str(name))
    return sequence, PredicateTarget(
        mode="by_type_or_color",
        target_shape_type=str(shape_type),
        target_color_name=str(color_name),
        target_object_name=str(name),
        target_object_plural=str(plural),
        target_property_phrase=f"{plural} or {color_name} objects",
        extras={"target_property_prompt_phrase": f"{plural} or {semantic_color_label(str(color_name))} objects"},
    )


def build_xor_sequence(
    *,
    shape_type: str,
    color_name: str,
    target_count: int,
    object_count: int,
    rng,
) -> tuple[list[ClusterSequenceItem], PredicateTarget]:
    """Build an exclusive OR set: target type or target color, but not both."""

    if int(target_count) < 1:
        raise ValueError("exclusive OR count needs at least one target")
    wrong_shapes = list(compatible_distractor_pool(str(shape_type), support=color_readout_shape_support()))
    wrong_colors = list(readout_color_support(anchors=(str(color_name),)))
    sequence: list[ClusterSequenceItem] = []
    while len(sequence) < int(target_count):
        if len(sequence) % 2 == 0:
            sequence.append(ClusterSequenceItem(str(shape_type), str(rng.choice(wrong_colors)), True, "target_type_only"))
        else:
            sequence.append(ClusterSequenceItem(str(rng.choice(wrong_shapes)), str(color_name), True, "target_color_only"))
    if len(sequence) < int(object_count):
        sequence.append(ClusterSequenceItem(str(shape_type), str(color_name), False, "excluded_overlap"))
    while len(sequence) < int(object_count):
        sequence.append(ClusterSequenceItem(str(rng.choice(wrong_shapes)), str(rng.choice(wrong_colors)), False, "distractor"))
    rng.shuffle(sequence)
    name = object_name_for_shape(str(shape_type))
    plural = object_plural(str(name))
    return sequence, PredicateTarget(
        mode="by_exactly_one_type_or_color",
        target_shape_type=str(shape_type),
        target_color_name=str(color_name),
        target_object_name=str(name),
        target_object_plural=str(plural),
        target_property_phrase=f"exactly one of {plural} or {color_name} objects",
        extras={
            "target_property_prompt_phrase": f"exactly one of {plural} or {semantic_color_label(str(color_name))} objects"
        },
    )


def build_exclusion_sequence(
    *,
    mode: str,
    shape_type: str,
    color_name: str,
    target_count: int,
    object_count: int,
    rng,
) -> tuple[list[ClusterSequenceItem], PredicateTarget]:
    """Build a count set that matches one attribute while excluding another."""

    wrong_shapes = list(compatible_distractor_pool(str(shape_type), support=color_readout_shape_support()))
    wrong_colors = list(readout_color_support(anchors=(str(color_name),)))
    sequence: list[ClusterSequenceItem] = []
    if str(mode) == "type_without_color":
        for _ in range(int(target_count)):
            sequence.append(ClusterSequenceItem(str(shape_type), str(rng.choice(wrong_colors)), True, "target"))
        if len(sequence) < int(object_count):
            sequence.append(ClusterSequenceItem(str(shape_type), str(color_name), False, "excluded_intersection"))
        while len(sequence) < int(object_count):
            sequence.append(ClusterSequenceItem(str(rng.choice(wrong_shapes)), str(rng.choice(wrong_colors)), False, "distractor"))
        phrase = f"{shape_plural(str(shape_type))} that are not {color_name}"
        prompt_phrase = f"{shape_plural(str(shape_type))} that are not {semantic_color_label(str(color_name))}"
    else:
        for _ in range(int(target_count)):
            sequence.append(ClusterSequenceItem(str(rng.choice(wrong_shapes)), str(color_name), True, "target"))
        if len(sequence) < int(object_count):
            sequence.append(ClusterSequenceItem(str(shape_type), str(color_name), False, "excluded_intersection"))
        while len(sequence) < int(object_count):
            sequence.append(ClusterSequenceItem(str(shape_type), str(rng.choice(wrong_colors)), False, "distractor"))
        phrase = f"{color_name} objects that are not {shape_plural(str(shape_type))}"
        prompt_phrase = f"{semantic_color_label(str(color_name))} objects that are not {shape_plural(str(shape_type))}"
    rng.shuffle(sequence)
    name = object_name_for_shape(str(shape_type))
    return sequence, PredicateTarget(
        mode=str(mode),
        target_shape_type=str(shape_type),
        target_color_name=str(color_name),
        target_object_name=str(name),
        target_object_plural=object_plural(str(name)),
        target_property_phrase=str(phrase),
        extras={"target_property_prompt_phrase": str(prompt_phrase)},
    )


def build_arithmetic_sequence(
    *,
    operand_kind: str,
    operation: str,
    left_value: str,
    right_value: str,
    left_count: int,
    right_count: int,
    object_count: int,
    rng,
) -> tuple[list[ClusterSequenceItem], PredicateTarget]:
    """Build two disjoint operand groups for total or absolute-difference counts."""

    sequence: list[ClusterSequenceItem] = []
    shapes = list(safe_shape_support())
    if str(operand_kind) == "shape":
        left_phrase = shape_plural(str(left_value))
        right_phrase = shape_plural(str(right_value))
        for _ in range(int(left_count)):
            sequence.append(ClusterSequenceItem(str(left_value), random_color(rng), True, "left_operand"))
        for _ in range(int(right_count)):
            sequence.append(ClusterSequenceItem(str(right_value), random_color(rng), True, "right_operand"))
        distractor_shapes = [shape for shape in shapes if str(shape) not in {str(left_value), str(right_value)}]
        while len(sequence) < int(object_count):
            sequence.append(ClusterSequenceItem(str(rng.choice(distractor_shapes)), random_color(rng), False, "distractor"))
        extras = {"target_shape_types": [str(left_value), str(right_value)]}
    else:
        left_phrase = f"{left_value} objects"
        right_phrase = f"{right_value} objects"
        left_prompt_phrase = f"{semantic_color_label(str(left_value))} objects"
        right_prompt_phrase = f"{semantic_color_label(str(right_value))} objects"
        color_distractor_pool = list(readout_color_support(anchors=(str(left_value), str(right_value))))
        shapes = list(color_readout_shape_support())
        for _ in range(int(left_count)):
            sequence.append(ClusterSequenceItem(str(rng.choice(shapes)), str(left_value), True, "left_operand"))
        for _ in range(int(right_count)):
            sequence.append(ClusterSequenceItem(str(rng.choice(shapes)), str(right_value), True, "right_operand"))
        while len(sequence) < int(object_count):
            sequence.append(ClusterSequenceItem(str(rng.choice(shapes)), str(rng.choice(color_distractor_pool)), False, "distractor"))
        extras = {
            "target_color_names": [str(left_value), str(right_value)],
            "left_operand_prompt_phrase": str(left_prompt_phrase),
            "right_operand_prompt_phrase": str(right_prompt_phrase),
            "target_property_prompt_phrase": f"{left_prompt_phrase} and {right_prompt_phrase}",
        }
    rng.shuffle(sequence)
    return sequence, PredicateTarget(
        mode="operand_arithmetic",
        left_operand_phrase=str(left_phrase),
        right_operand_phrase=str(right_phrase),
        arithmetic_operation=str(operation),
        left_count=int(left_count),
        right_count=int(right_count),
        target_property_phrase=f"{left_phrase} and {right_phrase}",
        extras=extras,
    )


def resolve_shape_choice(
    *,
    params: Mapping[str, Any],
    key: str,
    instance_seed: int,
    namespace: str,
    support: Sequence[str] = NAMED_CLUSTER_SHAPE_TYPES,
) -> tuple[str, Dict[str, float]]:
    """Resolve one object shape from the requested support."""

    return resolve_named_choice(
        params=params,
        key=str(key),
        support=tuple(str(shape) for shape in support),
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )


def resolve_color_choice(
    *,
    params: Mapping[str, Any],
    key: str,
    instance_seed: int,
    namespace: str,
) -> tuple[str, Dict[str, float]]:
    """Resolve one semantic color name."""

    return resolve_named_choice(
        params=params,
        key=str(key),
        support=color_support(),
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )


def resolve_two_shapes(
    *,
    params: Mapping[str, Any],
    key: str,
    instance_seed: int,
    namespace: str,
) -> tuple[list[str], Dict[str, float]]:
    """Resolve two distinct object shapes for union/arithmetic contracts."""

    return resolve_string_subset(
        params=params,
        key=str(key),
        support=safe_shape_support(),
        instance_seed=int(instance_seed),
        namespace=str(namespace),
        count=2,
    )


def resolve_two_colors(
    *,
    params: Mapping[str, Any],
    key: str,
    instance_seed: int,
    namespace: str,
) -> tuple[list[str], Dict[str, float]]:
    """Resolve two distinct semantic colors for arithmetic contracts."""

    explicit_value = params.get(str(key))
    if explicit_value is not None:
        return resolve_string_subset(
            params=params,
            key=str(key),
            support=color_support(),
            instance_seed=int(instance_seed),
            namespace=str(namespace),
            count=2,
        )
    selected = _sample_nonconflicting_readout_colors(
        instance_seed=int(instance_seed),
        namespace=str(namespace),
        count=2,
    )
    return list(selected), selected_probability_map(color_support(), selected)


__all__ = [
    "build_arithmetic_sequence",
    "build_color_membership_sequence",
    "build_exclusion_sequence",
    "build_or_sequence",
    "build_xor_sequence",
    "build_total_sequence",
    "build_type_and_color_sequence",
    "build_type_membership_sequence",
    "color_readout_shape_support",
    "named_shape_support",
    "resolve_color_choice",
    "resolve_composition_mode",
    "resolve_membership_counts",
    "resolve_shape_choice",
    "resolve_two_colors",
    "resolve_two_shapes",
    "safe_shape_support",
    "semantic_color_label",
    "selected_probability_map",
    "target_mapping",
]
