"""ClusterRequest assembly helpers for public object-cluster tasks."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.three_d.shared.object_scene import ObjectSceneRenderParams
from trace_tasks.tasks.three_d.shared.task_support import shuffled_repeated_support

from .defaults import (
    CLUSTER_SCENE_VARIANTS,
    COLOR_READOUT_CLUSTER_SHAPE_TYPES,
    NAMED_CLUSTER_SHAPE_TYPES,
    PROMPT_COLOR_RGB,
    object_name_for_shape,
    object_plural,
)
from .objects import build_dataset_from_sequence
from .relations import (
    build_arithmetic_sequence,
    build_color_membership_sequence,
    build_exclusion_sequence,
    build_or_sequence,
    build_total_sequence,
    build_type_and_color_sequence,
    build_type_membership_sequence,
    build_xor_sequence,
    color_support,
    resolve_color_choice,
    resolve_composition_mode,
    resolve_membership_counts,
    resolve_shape_choice,
    resolve_two_colors,
    resolve_two_shapes,
    selected_probability_map,
    semantic_color_label,
    shape_plural,
    target_mapping,
)
from .sampling import count_bounds, configured_int, resolve_named_choice, resolve_scene_variant, resolve_uniform_count, resolve_weighted_count
from .state import ClusterRequest


COUNTERFACTUAL_PREDICATE_KINDS: Tuple[str, ...] = ("color", "object", "color_object")


def _resolve_counterfactual_axis(
    *,
    params: Mapping[str, Any],
    key: str,
    support: Sequence[str],
    instance_seed: int,
    namespace: str,
) -> tuple[str, Dict[str, float]]:
    """Resolve one internal counterfactual axis without exposing it as query id."""

    return resolve_named_choice(
        params=params,
        key=str(key),
        support=tuple(str(value) for value in support),
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )


def _resolve_counterfactual_shape(
    *,
    predicate_kind: str,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> tuple[str | None, Dict[str, float]]:
    """Resolve target shape support for one counterfactual predicate kind."""

    if str(predicate_kind) == "color":
        return None, {}
    support = COLOR_READOUT_CLUSTER_SHAPE_TYPES if str(predicate_kind) == "color_object" else NAMED_CLUSTER_SHAPE_TYPES
    shape, probabilities = resolve_named_choice(
        params=params,
        key="target_shape_type",
        support=support,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    return str(shape), dict(probabilities)


def _resolve_counterfactual_color(
    *,
    predicate_kind: str,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> tuple[str | None, Dict[str, float]]:
    """Resolve target color support for one counterfactual predicate kind."""

    if str(predicate_kind) == "object":
        return None, {}
    color, probabilities = resolve_named_choice(
        params=params,
        key="target_color_name",
        support=color_support(),
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    return str(color), dict(probabilities)


def _target_phrase_for_edit(target_spec: Mapping[str, Any]) -> str:
    """Return a concise prompt phrase for the exact target predicate."""

    return str(target_spec.get("target_property_prompt_phrase") or target_spec.get("target_property_phrase") or "counted objects")


def _counterfactual_shape_support() -> Tuple[str, ...]:
    """Return all shape types available for counterfactual edit predicates."""

    return tuple(dict.fromkeys([*COLOR_READOUT_CLUSTER_SHAPE_TYPES, *NAMED_CLUSTER_SHAPE_TYPES]))


def _property_key(shape_type: str, color_name: str) -> Tuple[str, str]:
    """Return the exact shape/color key used by count metadata."""

    return (str(shape_type), str(color_name))


def _predicate_matches_key(predicate: Mapping[str, Any], key: Tuple[str, str]) -> bool:
    """Return whether one exact shape/color key satisfies a broad predicate."""

    shape_type, color_name = _property_key(str(key[0]), str(key[1]))
    predicate_shape = predicate.get("shape_type")
    predicate_color = predicate.get("color_name")
    if predicate_shape is not None and str(predicate_shape) != str(shape_type):
        return False
    if predicate_color is not None and str(predicate_color) != str(color_name):
        return False
    return True


def _predicate_relation_to_target(edit_predicate: Mapping[str, Any], target_predicate: Mapping[str, Any]) -> str:
    """Classify whether an edit predicate affects, misses, or ambiguously overlaps the target."""

    edit_shape = edit_predicate.get("shape_type")
    edit_color = edit_predicate.get("color_name")
    target_shape = target_predicate.get("shape_type")
    target_color = target_predicate.get("color_name")

    if edit_shape is not None and target_shape is not None and str(edit_shape) != str(target_shape):
        return "disjoint"
    if edit_color is not None and target_color is not None and str(edit_color) != str(target_color):
        return "disjoint"

    target_attrs = {"shape_type": target_shape, "color_name": target_color}
    edit_attrs = {"shape_type": edit_shape, "color_name": edit_color}
    is_subset = True
    for attr_name, target_value in target_attrs.items():
        if target_value is None:
            continue
        edit_value = edit_attrs[attr_name]
        if edit_value is None or str(edit_value) != str(target_value):
            is_subset = False
            break
    return "subset" if bool(is_subset) else "ambiguous"


def _counterfactual_predicate(
    *,
    predicate_kind: str,
    shape_type: str | None = None,
    color_name: str | None = None,
) -> Dict[str, Any]:
    """Build one edit predicate with explicit shape/color requirements."""

    kind = str(predicate_kind)
    if kind not in set(COUNTERFACTUAL_PREDICATE_KINDS):
        raise ValueError(f"unsupported counterfactual predicate kind: {predicate_kind}")
    if kind in {"object", "color_object"} and not shape_type:
        raise ValueError(f"predicate kind {kind} requires shape_type")
    if kind in {"color", "color_object"} and not color_name:
        raise ValueError(f"predicate kind {kind} requires color_name")
    return {
        "predicate_kind": str(kind),
        "shape_type": str(shape_type) if shape_type is not None else None,
        "color_name": str(color_name) if color_name is not None else None,
    }


def _predicate_phrase(predicate: Mapping[str, Any], *, count: int | None = None, prompt_facing: bool = False) -> str:
    """Return a singular/plural phrase for a counterfactual edit predicate."""

    kind = str(predicate["predicate_kind"])
    amount = int(count) if count is not None else 2
    shape_type = predicate.get("shape_type")
    color_name = predicate.get("color_name")
    if color_name is not None and bool(prompt_facing):
        color_phrase = semantic_color_label(str(color_name))
    else:
        color_phrase = str(color_name) if color_name is not None else ""
    if kind == "color":
        noun = "object" if int(amount) == 1 else "objects"
        return f"{color_phrase} {noun}".strip()
    if kind == "object":
        name = object_name_for_shape(str(shape_type))
        return str(name if int(amount) == 1 else object_plural(str(name)))
    if kind == "color_object":
        name = object_name_for_shape(str(shape_type))
        noun = str(name if int(amount) == 1 else object_plural(str(name)))
        return f"{color_phrase} {noun}".strip()
    raise ValueError(f"unsupported predicate kind: {kind}")


def _quantity_phrase(amount: int, predicate: Mapping[str, Any], *, prompt_facing: bool = False) -> str:
    """Return one prompt/edit quantity phrase."""

    return f"{int(amount)} {_predicate_phrase(predicate, count=int(amount), prompt_facing=bool(prompt_facing))}"


def _all_exact_property_keys() -> Tuple[Tuple[str, str], ...]:
    """Return exact shape/color keys for counterfactual count bookkeeping."""

    return tuple(
        _property_key(str(shape), str(color))
        for shape in _counterfactual_shape_support()
        for color in color_support()
    )


def _candidate_edit_predicates(target_predicate: Mapping[str, Any], *, relation: str) -> Tuple[Dict[str, Any], ...]:
    """Return edit predicates with a requested relation to the target predicate."""

    candidates: list[Dict[str, Any]] = []
    for color_name in color_support():
        candidates.append(_counterfactual_predicate(predicate_kind="color", color_name=str(color_name)))
    for shape_type in _counterfactual_shape_support():
        candidates.append(_counterfactual_predicate(predicate_kind="object", shape_type=str(shape_type)))
    for shape_type in _counterfactual_shape_support():
        for color_name in color_support():
            candidates.append(
                _counterfactual_predicate(
                    predicate_kind="color_object",
                    shape_type=str(shape_type),
                    color_name=str(color_name),
                )
            )
    return tuple(
        dict(predicate)
        for predicate in candidates
        if _predicate_relation_to_target(predicate, target_predicate) == str(relation)
    )


def _predicate_count(current_counts: Counter[Tuple[str, str]], predicate: Mapping[str, Any]) -> int:
    """Count exact shape/color records matching one broad predicate."""

    return int(sum(int(count) for key, count in current_counts.items() if _predicate_matches_key(predicate, key)))


def _remove_visible_from_counts(
    *,
    rng,
    current_counts: Counter[Tuple[str, str]],
    visible_remaining_counts: Counter[Tuple[str, str]],
    predicate: Mapping[str, Any],
    amount: int,
) -> Dict[str, int]:
    """Remove only exact properties still represented by visible starting objects."""

    remaining = int(amount)
    exact_keys = [
        key
        for key in _all_exact_property_keys()
        if _predicate_matches_key(predicate, key)
        and int(visible_remaining_counts[key]) > 0
        and int(current_counts[key]) > 0
    ]
    rng.shuffle(exact_keys)
    removed: Dict[str, int] = {}
    for key in exact_keys:
        if remaining <= 0:
            break
        take = min(int(visible_remaining_counts[key]), int(current_counts[key]), int(remaining))
        visible_remaining_counts[key] -= int(take)
        current_counts[key] -= int(take)
        removed[f"{key[1]}_{key[0]}"] = int(take)
        remaining -= int(take)
    if remaining != 0:
        raise ValueError("counterfactual step would remove more visible starting objects than available")
    return dict(removed)


def _add_to_counts(
    *,
    rng,
    current_counts: Counter[Tuple[str, str]],
    predicate: Mapping[str, Any],
    amount: int,
) -> Dict[str, int]:
    """Add exact shape/color records satisfying one broad predicate."""

    exact_keys = [key for key in _all_exact_property_keys() if _predicate_matches_key(predicate, key)]
    if not exact_keys:
        raise ValueError("counterfactual add predicate has no exact support")
    added: Dict[str, int] = {}
    for key in shuffled_repeated_support(rng, exact_keys, int(amount)):
        current_counts[key] += 1
        added[f"{key[1]}_{key[0]}"] = int(added.get(f"{key[1]}_{key[0]}", 0)) + 1
    return dict(added)


def _sample_edit_step(
    *,
    rng,
    current_counts: Counter[Tuple[str, str]],
    visible_remaining_counts: Counter[Tuple[str, str]],
    target_predicate: Mapping[str, Any],
    relation: str,
    edit_amount_max: int,
) -> Dict[str, Any]:
    """Sample one deterministic count edit with either target-subset or disjoint semantics."""

    candidate_predicates = list(_candidate_edit_predicates(target_predicate, relation=str(relation)))
    rng.shuffle(candidate_predicates)
    operations = ("remove", "add") if bool(rng.random() < 0.45) else ("add", "remove")
    for operation in operations:
        for predicate in candidate_predicates:
            available = (
                _predicate_count(visible_remaining_counts, predicate)
                if str(operation) == "remove"
                else _predicate_count(current_counts, predicate)
            )
            if str(operation) == "remove" and int(available) <= 0:
                continue
            max_amount = min(int(edit_amount_max), int(available)) if str(operation) == "remove" else int(edit_amount_max)
            if int(max_amount) <= 0:
                continue
            amount = int(rng.randint(1, int(max_amount)))
            relation_to_target = _predicate_relation_to_target(predicate, target_predicate)
            if relation_to_target == "ambiguous":
                continue
            if str(operation) == "remove":
                exact_deltas = _remove_visible_from_counts(
                    rng=rng,
                    current_counts=current_counts,
                    visible_remaining_counts=visible_remaining_counts,
                    predicate=predicate,
                    amount=int(amount),
                )
                signed_amount = -int(amount)
            else:
                exact_deltas = _add_to_counts(
                    rng=rng,
                    current_counts=current_counts,
                    predicate=predicate,
                    amount=int(amount),
                )
                signed_amount = int(amount)
            target_delta = int(signed_amount) if relation_to_target == "subset" else 0
            prompt_phrase = _predicate_phrase(predicate, count=int(amount), prompt_facing=True)
            plain_phrase = _predicate_phrase(predicate, count=int(amount), prompt_facing=False)
            verb = "Add" if str(operation) == "add" else "Remove"
            return {
                "operation": str(operation),
                "amount": int(amount),
                "predicate": dict(predicate),
                "predicate_kind": str(predicate["predicate_kind"]),
                "target_shape_type": predicate.get("shape_type"),
                "target_color_name": predicate.get("color_name"),
                "target_property_phrase": str(plain_phrase),
                "target_property_prompt_phrase": str(prompt_phrase),
                "predicate_relation_to_target": str(relation_to_target),
                "affects_target_property": bool(relation_to_target == "subset"),
                "target_delta": int(target_delta),
                "exact_property_deltas": dict(exact_deltas),
                "step_text": f"{verb} {_quantity_phrase(int(amount), predicate, prompt_facing=True)}.",
            }
    raise ValueError(f"could not sample {relation} counterfactual edit step")


def _build_counterfactual_steps(
    *,
    rng,
    initial_counts: Counter[Tuple[str, str]],
    target_predicate: Mapping[str, Any],
    edit_step_count: int,
    edit_amount_max: int,
) -> Tuple[list[Dict[str, Any]], str, int]:
    """Build numbered multi-step edits and return final target count."""

    current_counts: Counter[Tuple[str, str]] = Counter(initial_counts)
    visible_remaining_counts: Counter[Tuple[str, str]] = Counter(initial_counts)
    initial_target_count = _predicate_count(current_counts, target_predicate)
    target_step_indices = {0}
    if int(edit_step_count) >= 3:
        target_step_indices.add(2)
    steps: list[Dict[str, Any]] = []
    for step_index in range(int(edit_step_count)):
        relation = "subset" if int(step_index) in target_step_indices else "disjoint"
        step = _sample_edit_step(
            rng=rng,
            current_counts=current_counts,
            visible_remaining_counts=visible_remaining_counts,
            target_predicate=target_predicate,
            relation=str(relation),
            edit_amount_max=int(edit_amount_max),
        )
        step.update({"step_index": int(step_index) + 1})
        steps.append(dict(step))
    final_target_count = _predicate_count(current_counts, target_predicate)
    if int(final_target_count) <= 0:
        raise ValueError("counterfactual final target count must remain positive")
    if int(final_target_count) == int(initial_target_count):
        raise ValueError("counterfactual target count did not change")
    edit_steps_text = "\n".join(f"{step['step_index']}. {step['step_text']}" for step in steps)
    return list(steps), str(edit_steps_text), int(final_target_count)


def _build_counterfactual_initial_sequence(
    *,
    predicate_kind: str,
    shape_type: str | None,
    color_name: str | None,
    target_count: int,
    object_count: int,
    rng,
):
    """Build the starting visible cluster for one target predicate."""

    if str(predicate_kind) == "color":
        if color_name is None:
            raise ValueError("color counterfactual target needs a color")
        return build_color_membership_sequence(
            color_name=str(color_name),
            target_count=int(target_count),
            object_count=int(object_count),
            rng=rng,
        )
    if str(predicate_kind) == "object":
        if shape_type is None:
            raise ValueError("object counterfactual target needs a shape")
        return build_type_membership_sequence(
            shape_type=str(shape_type),
            composition_mode="mixed_type_cluster",
            target_count=int(target_count),
            object_count=int(object_count),
            rng=rng,
        )
    if shape_type is None or color_name is None:
        raise ValueError("color+object counterfactual target needs a shape and color")
    return build_type_and_color_sequence(
        shape_type=str(shape_type),
        color_name=str(color_name),
        target_count=int(target_count),
        object_count=int(object_count),
        rng=rng,
    )


def build_counterfactual_count_request(
    *,
    external_query: str,
    prompt_key: str,
    branch_probabilities: Mapping[str, float],
    namespace: str,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    render_params: ObjectSceneRenderParams,
) -> ClusterRequest:
    """Assemble a dense-cluster count-after-edits request.

    The invariant is that annotation witnesses are the visible starting target
    objects, while the answer is the final count after deterministic add/remove
    edits. Predicate kind and edit program details are generation axes, not
    public query ids.
    """

    scene_variant, scene_probabilities = resolve_scene_variant(
        params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        support=CLUSTER_SCENE_VARIANTS,
        namespace=f"{namespace}.scene_variant",
    )
    predicate_kind, predicate_probabilities = _resolve_counterfactual_axis(
        params=params,
        key="predicate_kind",
        support=COUNTERFACTUAL_PREDICATE_KINDS,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.predicate_kind",
    )
    shape_type, shape_probabilities = _resolve_counterfactual_shape(
        predicate_kind=str(predicate_kind),
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.target_shape_type",
    )
    color_name, color_probabilities = _resolve_counterfactual_color(
        predicate_kind=str(predicate_kind),
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.target_color_name",
    )

    target_min = configured_int(params, gen_defaults, "target_count_min", 3)
    target_max = configured_int(params, gen_defaults, "target_count_max", 8)
    target_count, target_probabilities = resolve_uniform_count(
        params=params,
        explicit_key="target_count",
        minimum=max(2, int(target_min)),
        maximum=max(max(2, int(target_min)), int(target_max)),
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.target_count",
    )
    distractor_min = configured_int(params, gen_defaults, "distractor_count_min", 5)
    distractor_max = configured_int(params, gen_defaults, "distractor_count_max", 10)
    object_min = configured_int(params, gen_defaults, "object_count_min", 12)
    object_max = configured_int(params, gen_defaults, "object_count_max", 20)
    min_distractors = max(1, int(distractor_min), int(object_min) - int(target_count))
    max_distractors = max(int(min_distractors), min(int(distractor_max), int(object_max) - int(target_count)))
    distractor_count, distractor_probabilities = resolve_uniform_count(
        params=params,
        explicit_key="distractor_count",
        minimum=int(min_distractors),
        maximum=int(max_distractors),
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.distractor_count",
    )
    object_count = int(target_count) + int(distractor_count)

    edit_step_min = configured_int(params, gen_defaults, "edit_step_count_min", 2)
    edit_step_max = configured_int(params, gen_defaults, "edit_step_count_max", 3)
    edit_step_count, edit_step_count_probabilities = resolve_uniform_count(
        params=params,
        explicit_key="edit_step_count",
        minimum=max(2, int(edit_step_min)),
        maximum=max(max(2, int(edit_step_min)), int(edit_step_max)),
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.edit_step_count",
    )
    edit_amount_max = max(1, configured_int(params, gen_defaults, "edit_amount_max", 2))

    rng = spawn_rng(int(instance_seed), f"{namespace}.sequence")
    sequence, target = _build_counterfactual_initial_sequence(
        predicate_kind=str(predicate_kind),
        shape_type=shape_type,
        color_name=color_name,
        target_count=int(target_count),
        object_count=int(object_count),
        rng=rng,
    )
    target_spec = target_mapping(target)
    target_spec["base_predicate_mode"] = str(target_spec.get("mode", ""))
    target_spec["mode"] = "count_after_edits"
    target_phrase = _target_phrase_for_edit(target_spec)
    target_predicate = _counterfactual_predicate(
        predicate_kind=str(predicate_kind),
        shape_type=shape_type,
        color_name=color_name,
    )
    initial_counts: Counter[Tuple[str, str]] = Counter(
        _property_key(str(item.shape_type), str(item.color_name))
        for item in sequence
    )
    counterfactual_steps, edit_steps_text, final_count = _build_counterfactual_steps(
        rng=rng,
        initial_counts=initial_counts,
        target_predicate=target_predicate,
        edit_step_count=int(edit_step_count),
        edit_amount_max=int(edit_amount_max),
    )

    extra_trace = {
        "counterfactual_step_count": int(edit_step_count),
        "counterfactual_steps": [dict(step) for step in counterfactual_steps],
        "edit_steps_text": str(edit_steps_text),
        "edit_amount_max": int(edit_amount_max),
        "initial_target_count": int(target_count),
        "final_target_count": int(final_count),
        "target_delta_total": int(final_count) - int(target_count),
        "distractor_count": int(distractor_count),
        "counterfactual_predicate_kind": str(predicate_kind),
        "counterfactual_target_exact_edit": False,
        "counterfactual_target_multistep": True,
        "counterfactual_target_phrase": str(target_spec.get("target_property_phrase", "")),
        "counterfactual_target_prompt_phrase": str(target_phrase),
    }
    dataset = build_dataset_from_sequence(
        source_namespace=str(namespace),
        scene_kind="three_d_object_cluster_count_after_edits",
        prompt_query_key=str(prompt_key),
        scene_variant=str(scene_variant),
        render_params=render_params,
        instance_seed=int(instance_seed),
        sequence=sequence,
        target_spec=dict(target_spec),
        answer_value=int(final_count),
        expected_annotation_count=int(target_count),
        extra_trace=dict(extra_trace),
    )
    return ClusterRequest(
        external_query=str(external_query),
        prompt_query_key=str(prompt_key),
        query_probabilities=dict(branch_probabilities),
        scene_variant=str(scene_variant),
        scene_probabilities=dict(scene_probabilities),
        dataset=dict(dataset),
        count_probabilities={
            "object_count_probabilities": {str(object_count): 1.0},
            "target_count_probabilities": dict(target_probabilities),
            "distractor_count_probabilities": dict(distractor_probabilities),
            "edit_step_count_probabilities": dict(edit_step_count_probabilities),
            "edit_amount_max": int(edit_amount_max),
            "predicate_kind_probabilities": dict(predicate_probabilities),
            "target_shape_probabilities": dict(shape_probabilities),
            "target_color_probabilities": dict(color_probabilities),
            "cluster_object_pool_size": len(NAMED_CLUSTER_SHAPE_TYPES),
            "color_readout_cluster_object_pool_size": len(COLOR_READOUT_CLUSTER_SHAPE_TYPES),
        },
        prompt_slots={
            "edit_steps_text": str(edit_steps_text),
            "initial_target_count": str(int(target_count)),
            "edit_step_count": str(int(edit_step_count)),
            "final_target_count": str(int(final_count)),
            "target_property_phrase": str(
                target_spec.get("target_property_prompt_phrase") or target_spec.get("target_property_phrase") or ""
            ),
            "target_color_name": semantic_color_label(str(color_name)) if color_name is not None else "",
            "target_object_plural": shape_plural(str(shape_type)) if shape_type is not None else "",
        },
        keyed_annotation=False,
    )


def build_count_request(
    *,
    mode: str,
    external_query: str,
    prompt_key: str,
    branch_probabilities: Mapping[str, float],
    namespace: str,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    render_params: ObjectSceneRenderParams,
) -> ClusterRequest:
    """Assemble a neutral request from a public task's already-resolved contract."""

    scene_variant, scene_probabilities = resolve_scene_variant(
        params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        support=CLUSTER_SCENE_VARIANTS,
        namespace=f"{namespace}.scene_variant",
    )
    rng = spawn_rng(int(instance_seed), f"{namespace}.sequence")
    count_probabilities: Dict[str, Any] = {"cluster_object_pool_size": len(NAMED_CLUSTER_SHAPE_TYPES)}
    keyed_annotation = False

    if str(mode) == "all_objects":
        minimum, maximum = count_bounds(
            params=params,
            gen_defaults=gen_defaults,
            minimum_key="object_count_min",
            maximum_key="object_count_max",
            fallback_minimum=configured_int(params, gen_defaults, "single_type_count_min", 6),
            fallback_maximum=configured_int(params, gen_defaults, "single_type_count_max", 25),
            lower=6,
            upper=25,
        )
        object_count, object_probabilities = resolve_weighted_count(
            params=params,
            gen_defaults=gen_defaults,
            instance_seed=int(instance_seed),
            explicit_key="object_count",
            weights_key="object_count_weights",
            minimum=int(minimum),
            maximum=int(maximum),
            namespace=f"{namespace}.object_count",
        )
        primary_shape, shape_probabilities = resolve_shape_choice(
            params=params,
            key="primary_shape_type",
            instance_seed=int(instance_seed),
            namespace=f"{namespace}.primary_shape_type",
            support=NAMED_CLUSTER_SHAPE_TYPES,
        )
        sequence, target = build_total_sequence(shape_type=str(primary_shape), object_count=int(object_count), rng=rng)
        answer_value = int(object_count)
        expected_annotation_count = int(object_count)
        scene_kind = "three_d_object_cluster_total_count"
        extra_trace = {"cluster_composition_mode": "single_type_cluster", "distractor_count": 0, "cluster_object_pool_size": len(NAMED_CLUSTER_SHAPE_TYPES)}
        count_probabilities.update({"object_count_probabilities": dict(object_probabilities), "target_count_probabilities": dict(object_probabilities), "target_shape_probabilities": dict(shape_probabilities), "cluster_object_pool_size": len(NAMED_CLUSTER_SHAPE_TYPES)})
    elif str(mode) == "type_membership":
        composition_mode, composition_probabilities = resolve_composition_mode(params=params, gen_defaults=gen_defaults, instance_seed=int(instance_seed), namespace=f"{namespace}.composition_mode")
        count_record = resolve_membership_counts(params=params, gen_defaults=gen_defaults, composition_mode=str(composition_mode), instance_seed=int(instance_seed), namespace=f"{namespace}.counts")
        target_shape, shape_probabilities = resolve_shape_choice(params=params, key="target_shape_type", instance_seed=int(instance_seed), namespace=f"{namespace}.target_shape_type", support=NAMED_CLUSTER_SHAPE_TYPES)
        sequence, target = build_type_membership_sequence(shape_type=str(target_shape), composition_mode=str(composition_mode), target_count=int(count_record["target_count"]), object_count=int(count_record["object_count"]), rng=rng)
        answer_value = int(count_record["target_count"])
        expected_annotation_count = int(count_record["target_count"])
        scene_kind = "three_d_object_cluster_instance_count"
        extra_trace = {"cluster_composition_mode": str(composition_mode), "distractor_count": int(count_record["distractor_count"]), "cluster_object_pool_size": len(NAMED_CLUSTER_SHAPE_TYPES)}
        count_probabilities.update({**dict(count_record), "composition_mode_probabilities": dict(composition_probabilities), "target_shape_probabilities": dict(shape_probabilities), "cluster_object_pool_size": len(NAMED_CLUSTER_SHAPE_TYPES)})
    elif str(mode) in {"type_color", "color_membership", "type_or_color", "type_xor_color", "type_without_color", "color_without_type"}:
        object_count, object_probabilities = resolve_uniform_count(params=params, explicit_key="object_count", minimum=int(gen_defaults.get("object_count_min", 16)), maximum=int(gen_defaults.get("object_count_max", 30)), instance_seed=int(instance_seed), namespace=f"{namespace}.object_count")
        target_min = int(gen_defaults.get("target_count_min", 2))
        target_max = min(int(gen_defaults.get("target_count_max", 12)), max(1, int(object_count) - 4))
        target_count, target_probabilities = resolve_uniform_count(params=params, explicit_key="target_count", minimum=target_min, maximum=target_max, instance_seed=int(instance_seed), namespace=f"{namespace}.target_count")
        if str(mode) == "color_membership":
            target_color, color_probabilities = resolve_color_choice(params=params, key="target_color_name", instance_seed=int(instance_seed), namespace=f"{namespace}.target_color_name")
            sequence, target = build_color_membership_sequence(color_name=str(target_color), target_count=int(target_count), object_count=int(object_count), rng=rng)
            shape_probabilities: Dict[str, float] = {}
        else:
            target_shape, shape_probabilities = resolve_shape_choice(params=params, key="target_shape_type", instance_seed=int(instance_seed), namespace=f"{namespace}.target_shape_type", support=COLOR_READOUT_CLUSTER_SHAPE_TYPES)
            target_color, color_probabilities = resolve_color_choice(params=params, key="target_color_name", instance_seed=int(instance_seed), namespace=f"{namespace}.target_color_name")
            if str(mode) == "type_color":
                sequence, target = build_type_and_color_sequence(shape_type=str(target_shape), color_name=str(target_color), target_count=int(target_count), object_count=int(object_count), rng=rng)
            elif str(mode) == "type_or_color":
                sequence, target = build_or_sequence(shape_type=str(target_shape), color_name=str(target_color), target_count=int(target_count), object_count=int(object_count), rng=rng)
            elif str(mode) == "type_xor_color":
                sequence, target = build_xor_sequence(shape_type=str(target_shape), color_name=str(target_color), target_count=int(target_count), object_count=int(object_count), rng=rng)
            else:
                sequence, target = build_exclusion_sequence(mode=str(mode), shape_type=str(target_shape), color_name=str(target_color), target_count=int(target_count), object_count=int(object_count), rng=rng)
        answer_value = int(target_count)
        expected_annotation_count = int(target_count)
        scene_kind = f"three_d_object_cluster_{mode}"
        pool_size = len(COLOR_READOUT_CLUSTER_SHAPE_TYPES)
        extra_trace = {"cluster_object_pool_size": int(pool_size)}
        count_probabilities.update({"object_count_probabilities": dict(object_probabilities), "target_count_probabilities": dict(target_probabilities), "target_shape_probabilities": dict(shape_probabilities), "target_color_probabilities": dict(color_probabilities), "cluster_object_pool_size": int(pool_size)})
    elif str(mode).startswith("arithmetic_"):
        operand_min = max(1, int(gen_defaults.get("operand_count_min", 1)))
        operand_max = max(int(operand_min), int(gen_defaults.get("operand_count_max", 7)))
        operation = "total" if str(mode).endswith("_total") else "absolute_difference"
        explicit_left = params.get("left_operand_count") is not None
        explicit_right = params.get("right_operand_count") is not None
        if bool(explicit_left) != bool(explicit_right):
            raise ValueError("left_operand_count and right_operand_count must be provided together")
        if str(operation) == "absolute_difference" and not bool(explicit_left):
            difference_min = max(0, int(gen_defaults.get("difference_answer_min", 0)))
            difference_max = max(
                int(difference_min),
                min(int(gen_defaults.get("difference_answer_max", int(operand_max) - int(operand_min))), int(operand_max) - int(operand_min)),
            )
            support = tuple(range(int(difference_min), int(difference_max) + 1))
            explicit_answer = params.get("answer_value")
            if explicit_answer is not None:
                answer_value = int(explicit_answer)
                if int(answer_value) not in set(support):
                    raise ValueError(f"unsupported arithmetic answer_value: {answer_value}")
            else:
                answer_value, _answer_probabilities = resolve_uniform_count(
                    params=params,
                    explicit_key="answer_value",
                    minimum=int(difference_min),
                    maximum=int(difference_max),
                    instance_seed=int(instance_seed),
                    namespace=f"{namespace}.difference_answer_value",
                )
            count_pairs = [
                (left, right)
                for left in range(int(operand_min), int(operand_max) + 1)
                for right in range(int(operand_min), int(operand_max) + 1)
                if abs(int(left) - int(right)) == int(answer_value)
            ]
            if not count_pairs:
                raise ValueError(f"no arithmetic operand counts for difference answer {answer_value}")
            left_count, right_count = count_pairs[int(rng.randrange(len(count_pairs)))]
            left_probabilities = {str(left_count): 1.0}
            right_probabilities = {str(right_count): 1.0}
        else:
            left_count, left_probabilities = resolve_uniform_count(params=params, explicit_key="left_operand_count", minimum=int(operand_min), maximum=int(operand_max), instance_seed=int(instance_seed), namespace=f"{namespace}.left_operand_count")
            right_count, right_probabilities = resolve_uniform_count(params=params, explicit_key="right_operand_count", minimum=int(operand_min), maximum=int(operand_max), instance_seed=int(instance_seed), namespace=f"{namespace}.right_operand_count")
            answer_value = int(left_count) + int(right_count) if operation == "total" else abs(int(left_count) - int(right_count))
        operand_total = int(left_count) + int(right_count)
        object_minimum = max(int(gen_defaults.get("object_count_min", 16)), int(operand_total) + 4)
        object_count, object_probabilities = resolve_uniform_count(params=params, explicit_key="object_count", minimum=int(object_minimum), maximum=max(object_minimum, int(gen_defaults.get("object_count_max", 30))), instance_seed=int(instance_seed), namespace=f"{namespace}.object_count")
        if "_type_" in str(mode):
            if params.get("left_shape_type") is not None and params.get("right_shape_type") is not None:
                operands = [str(params["left_shape_type"]), str(params["right_shape_type"])]
                if len(set(operands)) != 2 or any(value not in set(NAMED_CLUSTER_SHAPE_TYPES) for value in operands):
                    raise ValueError("shape operands must be two distinct supported shapes")
                operand_probabilities = selected_probability_map(NAMED_CLUSTER_SHAPE_TYPES, operands)
            else:
                operands, operand_probabilities = resolve_two_shapes(params=params, key="operand_shape_types", instance_seed=int(instance_seed), namespace=f"{namespace}.operand_shape_types")
            operand_kind = "shape"
            shape_probabilities = dict(operand_probabilities)
            color_probabilities = {}
        else:
            if params.get("left_color_name") is not None and params.get("right_color_name") is not None:
                operands = [str(params["left_color_name"]), str(params["right_color_name"])]
                if len(set(operands)) != 2 or any(value not in set(color_support()) for value in operands):
                    raise ValueError("color operands must be two distinct supported colors")
                operand_probabilities = selected_probability_map(color_support(), operands)
            else:
                operands, operand_probabilities = resolve_two_colors(params=params, key="operand_color_names", instance_seed=int(instance_seed), namespace=f"{namespace}.operand_color_names")
            operand_kind = "color"
            shape_probabilities = {}
            color_probabilities = dict(operand_probabilities)
        sequence, target = build_arithmetic_sequence(operand_kind=operand_kind, operation=operation, left_value=str(operands[0]), right_value=str(operands[1]), left_count=int(left_count), right_count=int(right_count), object_count=int(object_count), rng=rng)
        expected_annotation_count = int(operand_total)
        scene_kind = "three_d_object_cluster_count_arithmetic"
        pool_size = len(COLOR_READOUT_CLUSTER_SHAPE_TYPES) if str(operand_kind) == "color" else len(NAMED_CLUSTER_SHAPE_TYPES)
        extra_trace = {"cluster_object_pool_size": int(pool_size)}
        keyed_annotation = True
        count_probabilities.update({"object_count_probabilities": dict(object_probabilities), "target_count_probabilities": {"derived_from_operands": 1.0}, "left_operand_count": int(left_count), "left_operand_count_probabilities": dict(left_probabilities), "right_operand_count": int(right_count), "right_operand_count_probabilities": dict(right_probabilities), "target_shape_probabilities": shape_probabilities, "target_color_probabilities": color_probabilities, "cluster_object_pool_size": int(pool_size)})
    else:
        raise ValueError(f"unsupported object-cluster count mode: {mode}")

    dataset = build_dataset_from_sequence(
        source_namespace=str(namespace),
        scene_kind=str(scene_kind),
        prompt_query_key=str(prompt_key),
        scene_variant=str(scene_variant),
        render_params=render_params,
        instance_seed=int(instance_seed),
        sequence=sequence,
        target_spec=target_mapping(target),
        answer_value=int(answer_value),
        expected_annotation_count=int(expected_annotation_count),
        extra_trace=dict(extra_trace),
    )
    return ClusterRequest(
        external_query=str(external_query),
        prompt_query_key=str(prompt_key),
        query_probabilities=dict(branch_probabilities),
        scene_variant=str(scene_variant),
        scene_probabilities=dict(scene_probabilities),
        dataset=dict(dataset),
        count_probabilities=dict(count_probabilities),
        prompt_slots={},
        keyed_annotation=bool(keyed_annotation),
    )


__all__ = ["build_count_request", "build_counterfactual_count_request"]
