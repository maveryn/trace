"""Neutral dice-probability sampling and probability helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import gcd
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from .....core.seed import spawn_rng
from .....core.sampling import uniform_choice
from ...shared.common import get_int_param as _get_int
from ...shared.common import get_int_range as _get_range


DICE_COLOR_PALETTE: Tuple[Tuple[str, Tuple[int, int, int]], ...] = (
    ("red", (204, 67, 65)),
    ("blue", (55, 105, 190)),
    ("green", (53, 143, 91)),
    ("yellow", (235, 190, 58)),
    ("purple", (132, 89, 191)),
    ("orange", (218, 124, 53)),
)

PROBABILITY_OPTION_LABELS: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")

SINGLE_COLOR_OR_ANSWER_TARGETS: Tuple[str, ...] = (
    "1/2",
    "2/3",
    "3/4",
    "3/5",
    "4/5",
    "4/7",
    "5/6",
    "5/7",
    "5/8",
    "6/7",
    "7/8",
    "7/9",
    "8/9",
    "9/10",
    "10/11",
    "11/12",
)

CONDITIONAL_ANSWER_TARGETS: Tuple[str, ...] = (
    "1/3",
    "1/2",
    "2/5",
    "3/5",
    "3/4",
)


def format_fraction(numerator: int, denominator: int) -> str:
    """Return one reduced probability fraction."""

    if int(denominator) <= 0:
        raise ValueError("probability denominator must be positive")
    common = gcd(abs(int(numerator)), abs(int(denominator)))
    return f"{int(numerator) // common}/{int(denominator) // common}"


def parse_fraction_label(value: str) -> Tuple[int, int]:
    """Parse one reduced probability fraction label into integer counts."""

    raw = str(value).strip()
    if "/" not in raw:
        raise ValueError(f"probability option must be a fraction label: {value!r}")
    numerator_text, denominator_text = raw.split("/", 1)
    numerator = int(numerator_text.strip())
    denominator = int(denominator_text.strip())
    if denominator <= 0:
        raise ValueError("probability denominator must be positive")
    return int(numerator), int(denominator)


def _fraction_value(label: str) -> float:
    numerator, denominator = parse_fraction_label(str(label))
    return float(numerator) / float(denominator)


def _candidate_fraction_labels(favorable: int, total: int) -> List[str]:
    """Return plausible reduced-fraction distractors for a probability event."""

    favorable = int(favorable)
    total = int(total)
    correct = format_fraction(favorable, total)
    labels: List[str] = []

    def add(numerator: int, denominator: int) -> None:
        if int(denominator) <= 1:
            return
        if not (0 < int(numerator) < int(denominator)):
            return
        label = format_fraction(int(numerator), int(denominator))
        if label == correct or label in labels:
            return
        labels.append(str(label))

    # Common probability mistakes: off-by-one favorable counts, complement,
    # and using a nearby or wrong denominator.
    for delta in (-2, -1, 1, 2):
        add(favorable + int(delta), total)
    add(total - favorable, total)
    for denominator_delta in (-2, -1, 1, 2, 3):
        add(favorable, total + int(denominator_delta))
        add(favorable - 1, total + int(denominator_delta))
        add(favorable + 1, total + int(denominator_delta))
    add(total - favorable - 1, total)
    add(total - favorable + 1, total)

    # Fill with globally plausible nearby fractions so every task can show six
    # options even when the local count perturbations collapse after reduction.
    correct_value = _fraction_value(correct)
    global_candidates: List[str] = []
    max_denominator = max(12, min(48, int(total) + 16))
    for denominator in range(2, int(max_denominator) + 1):
        for numerator in range(1, int(denominator)):
            label = format_fraction(int(numerator), int(denominator))
            if label == correct or label in labels or label in global_candidates:
                continue
            global_candidates.append(str(label))
    global_candidates.sort(
        key=lambda label: (
            abs(_fraction_value(str(label)) - float(correct_value)),
            parse_fraction_label(str(label))[1],
            parse_fraction_label(str(label))[0],
        )
    )
    labels.extend(global_candidates)
    return labels


def build_probability_option_set(
    *,
    favorable: int,
    total: int,
    rng,
    labels: Sequence[str] = PROBABILITY_OPTION_LABELS,
    correct_label: str | None = None,
) -> Dict[str, Any]:
    """Bind one exact reduced probability fraction to fixed visible MCQ labels."""

    option_labels = tuple(str(label) for label in labels)
    if len(option_labels) != 6:
        raise ValueError("dice probability MCQ tasks require exactly six option labels")
    if len(set(option_labels)) != len(option_labels):
        raise ValueError("dice probability option labels must be unique")
    correct_fraction = format_fraction(int(favorable), int(total))
    requested_label = str(correct_label).strip().upper() if correct_label is not None else ""
    if requested_label:
        if requested_label not in option_labels:
            raise ValueError("explicit correct_label is outside dice probability option labels")
        selected_label = str(requested_label)
    else:
        selected_label = str(uniform_choice(rng, option_labels))

    candidates = _candidate_fraction_labels(int(favorable), int(total))
    if len(candidates) < len(option_labels) - 1:
        raise RuntimeError("failed to build enough dice probability distractors")
    nearby_pool = list(candidates[: min(24, len(candidates))])
    rng.shuffle(nearby_pool)
    distractors = list(nearby_pool[: len(option_labels) - 1])
    if len(distractors) < len(option_labels) - 1:
        raise RuntimeError("failed to sample enough dice probability distractors")

    text_by_label: Dict[str, str] = {}
    distractor_index = 0
    for label in option_labels:
        if str(label) == str(selected_label):
            text_by_label[str(label)] = str(correct_fraction)
        else:
            text_by_label[str(label)] = str(distractors[distractor_index])
            distractor_index += 1

    return {
        "labels": [str(label) for label in option_labels],
        "correct_label": str(selected_label),
        "correct_fraction": str(correct_fraction),
        "text_by_label": dict(text_by_label),
        "value_by_label": dict(text_by_label),
        "correct_label_probabilities": {
            str(label): (1.0 / float(len(option_labels))) for label in option_labels
        },
    }


def valid_favorable_count(count: int, total: int, *, min_count: int, max_count: int) -> bool:
    """Check that an event is nontrivial and within configured support."""

    return int(min_count) <= int(count) <= min(int(total) - 1, int(max_count))


def normalize_int_with_bounds(value: int, bounds: Sequence[int]) -> float:
    """Normalize an integer within inclusive calibration bounds."""

    low = int(bounds[0])
    high = int(bounds[1])
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (float(value) - float(low)) / float(high - low)))


def select_event_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    rng,
    favorable_key: str,
    denominator_key: str | None = None,
    fixed_total: int | None = None,
    answer_targets: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Choose one event candidate while balancing over feasible fraction labels."""

    if not candidates:
        raise RuntimeError("failed to choose a nontrivial dice probability event")

    enriched: List[Dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)
        favorable_count = int(len(item[str(favorable_key)]))
        total_count = int(len(item[str(denominator_key)])) if denominator_key else int(fixed_total or 0)
        item["favorable_outcome_count"] = int(favorable_count)
        item["total_outcome_count"] = int(total_count)
        item["answer_value"] = format_fraction(int(favorable_count), int(total_count))
        enriched.append(item)

    by_answer: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in enriched:
        by_answer[str(item["answer_value"])].append(dict(item))
    if answer_targets:
        target = str(uniform_choice(rng, tuple(str(value) for value in answer_targets)))
        bucket = list(by_answer.get(target, []))
        if not bucket:
            raise RuntimeError(f"no candidate for balanced answer target {target}")
        rng.shuffle(bucket)
        return dict(bucket[0])
    target = str(uniform_choice(rng, tuple(sorted(by_answer))))
    bucket = list(by_answer[target])
    rng.shuffle(bucket)
    return dict(bucket[0])


def is_prime_die_value(value: int) -> bool:
    """Return whether a standard die value is prime."""

    return int(value) in {2, 3, 5}


def numeric_properties() -> List[Tuple[str, Callable[[int], bool], Dict[str, Any]]]:
    """Return visible die-value predicates used by dice probability tasks."""

    return [
        ("shows an even value", lambda value: int(value) % 2 == 0, {"property": "even"}),
        ("shows an odd value", lambda value: int(value) % 2 == 1, {"property": "odd"}),
        ("shows a prime value", is_prime_die_value, {"property": "prime"}),
        ("shows a value at least 4", lambda value: int(value) >= 4, {"property": "at_least", "threshold": 4}),
        ("shows a value at most 3", lambda value: int(value) <= 3, {"property": "at_most", "threshold": 3}),
    ]


def conditional_value_properties() -> List[Tuple[str, Callable[[int], bool], Dict[str, Any]]]:
    """Return value predicates for conditional dice events."""

    properties = list(numeric_properties())
    for first in range(1, 7):
        for second in range(first + 1, 7):
            values = {int(first), int(second)}
            properties.append(
                (
                    f"shows either {first} or {second}",
                    lambda value, values=values: int(value) in values,
                    {"property": "value_set", "target_values": [int(first), int(second)]},
                )
            )
    return properties


def sample_dice(
    *,
    tray_id: str,
    count: int,
    rng,
    color_pool_size: int,
) -> List[Dict[str, Any]]:
    """Sample visible die values and colors for one tray."""

    color_pool = list(DICE_COLOR_PALETTE[: max(3, min(len(DICE_COLOR_PALETTE), int(color_pool_size)))])
    dice: List[Dict[str, Any]] = []
    for index in range(int(count)):
        color_name, color_rgb = uniform_choice(rng, tuple(color_pool))
        value = int(rng.randint(1, 6))
        dice.append(
            {
                "die_id": f"{tray_id}_die_{index}",
                "tray_id": str(tray_id),
                "die_index": int(index),
                "value": int(value),
                "color_name": str(color_name),
                "color_rgb": [int(channel) for channel in color_rgb],
            }
        )
    return dice


def color_names(dice: Sequence[Mapping[str, Any]]) -> List[str]:
    """Return sorted visible color names in a dice collection."""

    return sorted({str(die["color_name"]) for die in dice})


def pair_ids(
    dice_a: Sequence[Mapping[str, Any]],
    dice_b: Sequence[Mapping[str, Any]],
    predicate: Callable[[Mapping[str, Any], Mapping[str, Any]], bool],
) -> List[List[str]]:
    """Return favorable ordered die-pair ids for two trays."""

    return [
        [str(a["die_id"]), str(b["die_id"])]
        for a in dice_a
        for b in dice_b
        if predicate(a, b)
    ]


def fraction_count_pairs(
    *,
    target: str,
    min_denominator: int,
    max_denominator: int,
    min_favorable: int,
    max_favorable: int,
) -> List[Tuple[int, int]]:
    """Find count pairs that reduce to one target fraction."""

    pairs: List[Tuple[int, int]] = []
    for denominator in range(int(min_denominator), int(max_denominator) + 1):
        for favorable in range(int(min_favorable), min(int(max_favorable), int(denominator) - 1) + 1):
            if format_fraction(int(favorable), int(denominator)) == str(target):
                pairs.append((int(favorable), int(denominator)))
    return pairs


def sample_value_property_given_color_dice(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    count_min: int,
    count_max: int,
    color_pool_size: int,
    fixed_count: int | None,
    rng,
) -> Tuple[List[Dict[str, Any]], str]:
    """Construct a conditional-color subset with a feasible balanced answer."""

    min_denominator = _get_int(params, gen_defaults, "conditional_denominator_count_min", 3)
    raw_max_denominator = _get_int(params, gen_defaults, "conditional_denominator_count_max", 0)
    min_favorable = _get_int(params, gen_defaults, "conditional_favorable_count_min", 1)
    raw_max_favorable = _get_int(params, gen_defaults, "conditional_favorable_count_max", 0)
    max_total_count = int(fixed_count) if fixed_count is not None else int(count_max)
    max_denominator = int(raw_max_denominator) if int(raw_max_denominator) > 0 else int(count_max)
    max_favorable = int(raw_max_favorable) if int(raw_max_favorable) > 0 else max(1, int(max_denominator) - 1)
    feasible_targets: List[Tuple[str, List[Tuple[int, int]]]] = []
    for target in CONDITIONAL_ANSWER_TARGETS:
        pairs = fraction_count_pairs(
            target=str(target),
            min_denominator=int(min_denominator),
            max_denominator=min(int(max_denominator), int(max_total_count) - 1),
            min_favorable=int(min_favorable),
            max_favorable=int(max_favorable),
        )
        if pairs:
            feasible_targets.append((str(target), list(pairs)))
    if not feasible_targets:
        raise RuntimeError("no feasible conditional dice answer targets for current count constraints")

    target_answer, count_pairs = uniform_choice(rng, tuple(feasible_targets))
    favorable_count, denominator_count = uniform_choice(rng, tuple(count_pairs))
    if fixed_count is not None:
        total_count = int(fixed_count)
    else:
        min_total = max(int(count_min), int(denominator_count) + 1)
        if int(min_total) > int(count_max):
            raise RuntimeError("conditional dice count range cannot fit the target color group plus distractors")
        total_count = int(rng.randint(int(min_total), int(count_max)))

    color_pool = list(DICE_COLOR_PALETTE[: max(3, min(len(DICE_COLOR_PALETTE), int(color_pool_size)))])
    target_color_name, target_color_rgb = uniform_choice(rng, tuple(color_pool))
    distractor_colors = [color for color in color_pool if str(color[0]) != str(target_color_name)]
    if not distractor_colors:
        raise RuntimeError("conditional dice construction needs at least one distractor color")

    properties = conditional_value_properties()
    _description, predicate, _meta = uniform_choice(rng, tuple(properties))
    favorable_values = [value for value in range(1, 7) if predicate(int(value))]
    unfavorable_values = [value for value in range(1, 7) if not predicate(int(value))]
    if not favorable_values or not unfavorable_values:
        raise RuntimeError("conditional dice value predicate must have favorable and unfavorable values")

    raw_dice: List[Dict[str, Any]] = []
    for _ in range(int(favorable_count)):
        raw_dice.append(
            {
                "value": int(uniform_choice(rng, tuple(favorable_values))),
                "color_name": str(target_color_name),
                "color_rgb": [int(channel) for channel in target_color_rgb],
            }
        )
    for _ in range(int(denominator_count) - int(favorable_count)):
        raw_dice.append(
            {
                "value": int(uniform_choice(rng, tuple(unfavorable_values))),
                "color_name": str(target_color_name),
                "color_rgb": [int(channel) for channel in target_color_rgb],
            }
        )
    for _ in range(int(total_count) - int(denominator_count)):
        color_name, color_rgb = uniform_choice(rng, tuple(distractor_colors))
        raw_dice.append(
            {
                "value": int(rng.randint(1, 6)),
                "color_name": str(color_name),
                "color_rgb": [int(channel) for channel in color_rgb],
            }
        )
    rng.shuffle(raw_dice)
    dice: List[Dict[str, Any]] = []
    for index, die in enumerate(raw_dice):
        dice.append(
            {
                "die_id": f"tray_die_{index}",
                "tray_id": "tray",
                "die_index": int(index),
                "value": int(die["value"]),
                "color_name": str(die["color_name"]),
                "color_rgb": [int(channel) for channel in die["color_rgb"]],
            }
        )
    return dice, str(target_answer)


def build_single_dataset(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    rng_namespace: str,
    candidate_builder: Callable[[Sequence[Mapping[str, Any]], Mapping[str, Any], Mapping[str, Any], Any], Sequence[Mapping[str, Any]]],
    answer_targets: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Sample a one-tray dice dataset and bind one task-owned event."""

    rng = spawn_rng(int(instance_seed), str(rng_namespace))
    count_min, count_max = _get_range(
        params,
        gen_defaults,
        min_key="single_dice_count_min",
        max_key="single_dice_count_max",
        fallback_min=8,
        fallback_max=16,
    )
    color_pool_size = _get_int(params, gen_defaults, "dice_color_pool_size", 5)
    for _attempt in range(400):
        count = int(params.get("single_dice_count", rng.randint(int(count_min), int(count_max))))
        dice = sample_dice(tray_id="tray", count=count, rng=rng, color_pool_size=int(color_pool_size))
        try:
            candidates = candidate_builder(dice, params, gen_defaults, rng)
            event = select_event_candidate(
                candidates,
                rng=rng,
                favorable_key="favorable_die_ids",
                fixed_total=int(len(dice)),
                answer_targets=answer_targets,
            )
            return {
                "mode": "single",
                "tray_specs": [{"tray_id": "tray", "title": "Dice tray", "dice": dice}],
                "dice_count": int(count),
                "dice_count_range": [int(count_min), int(count_max)],
                "event": dict(event),
                "answer_value": str(event["answer_value"]),
                "calculation_supporting_item_ids": list(event["favorable_die_ids"]),
                "annotation_item_ids": ["tray"],
            }
        except RuntimeError:
            continue
    raise RuntimeError("failed to build single-dice probability dataset")


def build_pair_dataset(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    rng_namespace: str,
    candidate_builder: Callable[[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]], Mapping[str, Any], Mapping[str, Any], Any], Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    """Sample a two-tray dice dataset and bind one task-owned event."""

    rng = spawn_rng(int(instance_seed), str(rng_namespace))
    count_min, count_max = _get_range(
        params,
        gen_defaults,
        min_key="pair_dice_count_min",
        max_key="pair_dice_count_max",
        fallback_min=4,
        fallback_max=6,
    )
    color_pool_size = _get_int(params, gen_defaults, "dice_color_pool_size", 5)
    for _attempt in range(500):
        count_a = int(params.get("pair_dice_count_a", rng.randint(int(count_min), int(count_max))))
        count_b = int(params.get("pair_dice_count_b", rng.randint(int(count_min), int(count_max))))
        dice_a = sample_dice(tray_id="tray_a", count=count_a, rng=rng, color_pool_size=int(color_pool_size))
        dice_b = sample_dice(tray_id="tray_b", count=count_b, rng=rng, color_pool_size=int(color_pool_size))
        try:
            candidates = candidate_builder(dice_a, dice_b, params, gen_defaults, rng)
            event = select_event_candidate(
                candidates,
                rng=rng,
                favorable_key="favorable_pairs",
                fixed_total=int(len(dice_a) * len(dice_b)),
            )
            event["supporting_die_ids"] = sorted({die_id for pair in event["favorable_pairs"] for die_id in pair})
            return {
                "mode": "pair",
                "tray_specs": [
                    {"tray_id": "tray_a", "title": "Tray A", "dice": dice_a},
                    {"tray_id": "tray_b", "title": "Tray B", "dice": dice_b},
                ],
                "dice_count_a": int(count_a),
                "dice_count_b": int(count_b),
                "dice_count_range": [int(count_min), int(count_max)],
                "event": dict(event),
                "answer_value": str(event["answer_value"]),
                "calculation_supporting_item_ids": list(event["supporting_die_ids"]),
                "annotation_item_ids": ["tray_a", "tray_b"],
            }
        except RuntimeError:
            continue
    raise RuntimeError("failed to build pair-dice probability dataset")


def build_conditional_dataset(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    rng_namespace: str,
    candidate_builder: Callable[[Sequence[Mapping[str, Any]], Mapping[str, Any], Mapping[str, Any], Any], Sequence[Mapping[str, Any]]],
    dice_builder: Callable[[Mapping[str, Any], Mapping[str, Any], int, int, int, int | None, Any], Tuple[Sequence[Mapping[str, Any]], Sequence[str] | None]] | None = None,
    answer_targets: Sequence[str] | None = CONDITIONAL_ANSWER_TARGETS,
) -> Dict[str, Any]:
    """Sample a one-tray conditional dice dataset and bind one task-owned event."""

    rng = spawn_rng(int(instance_seed), str(rng_namespace))
    count_min, count_max = _get_range(
        params,
        gen_defaults,
        min_key="conditional_dice_count_min",
        max_key="conditional_dice_count_max",
        fallback_min=12,
        fallback_max=18,
    )
    color_pool_size = _get_int(params, gen_defaults, "dice_color_pool_size", 5)
    for _attempt in range(600):
        fixed_count = int(params["conditional_dice_count"]) if "conditional_dice_count" in params else None
        count = int(fixed_count) if fixed_count is not None else int(rng.randint(int(count_min), int(count_max)))
        current_targets = answer_targets
        if dice_builder is not None:
            dice_sequence, current_targets = dice_builder(
                params,
                gen_defaults,
                int(count_min),
                int(count_max),
                int(color_pool_size),
                fixed_count,
                rng,
            )
            dice = [dict(item) for item in dice_sequence]
            count = int(len(dice))
        else:
            dice = sample_dice(tray_id="tray", count=count, rng=rng, color_pool_size=int(color_pool_size))
        try:
            candidates = candidate_builder(dice, params, gen_defaults, rng)
            event = select_event_candidate(
                candidates,
                rng=rng,
                favorable_key="favorable_die_ids",
                denominator_key="denominator_die_ids",
                answer_targets=current_targets,
            )
            return {
                "mode": "conditional",
                "tray_specs": [{"tray_id": "tray", "title": "Dice tray", "dice": dice}],
                "dice_count": int(count),
                "dice_count_range": [int(count_min), int(count_max)],
                "event": dict(event),
                "answer_value": str(event["answer_value"]),
                "calculation_supporting_item_ids": list(event["favorable_die_ids"]),
                "denominator_supporting_item_ids": list(event["denominator_die_ids"]),
                "annotation_item_ids": ["tray"],
            }
        except RuntimeError:
            continue
    raise RuntimeError("failed to build conditional-dice probability dataset")


@dataclass(frozen=True)
class ThresholdEventSpec:
    property_name: str
    description_template: str
    thresholds: Tuple[int, ...]
    comparator: Callable[[int, int], bool]


def threshold_at_least(value: int, threshold: int) -> bool:
    return int(value) >= int(threshold)


def threshold_at_most(value: int, threshold: int) -> bool:
    return int(value) <= int(threshold)


def value_threshold_candidates(
    dice: Sequence[Mapping[str, Any]],
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    spec: ThresholdEventSpec,
) -> List[Dict[str, Any]]:
    """Return single-tray favorable die ids for one neutral threshold spec."""

    total = len(dice)
    min_count = _get_int(params, gen_defaults, "single_favorable_count_min", 2)
    max_count = _get_int(params, gen_defaults, "single_favorable_count_max", max(2, int(total) - 2))
    candidates: List[Dict[str, Any]] = []
    for threshold in spec.thresholds:
        favorable = [
            str(die["die_id"])
            for die in dice
            if spec.comparator(int(die["value"]), int(threshold))
        ]
        if valid_favorable_count(len(favorable), total, min_count=int(min_count), max_count=int(max_count)):
            candidates.append(
                {
                    "event_description": str(spec.description_template.format(threshold=int(threshold))),
                    "favorable_die_ids": list(favorable),
                    "property": str(spec.property_name),
                    "threshold": int(threshold),
                }
            )
    return candidates


def pair_sum_threshold_candidates(
    dice_a: Sequence[Mapping[str, Any]],
    dice_b: Sequence[Mapping[str, Any]],
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    spec: ThresholdEventSpec,
) -> List[Dict[str, Any]]:
    """Return favorable ordered die-pair ids for one neutral sum threshold spec."""

    total = int(len(dice_a) * len(dice_b))
    min_count = _get_int(params, gen_defaults, "pair_favorable_count_min", 2)
    max_count = _get_int(params, gen_defaults, "pair_favorable_count_max", max(2, int(total) - 2))
    candidates: List[Dict[str, Any]] = []
    for threshold in spec.thresholds:
        favorable_pairs = pair_ids(
            dice_a,
            dice_b,
            lambda a, b, threshold=threshold: spec.comparator(
                int(a["value"]) + int(b["value"]),
                int(threshold),
            ),
        )
        if valid_favorable_count(len(favorable_pairs), total, min_count=int(min_count), max_count=int(max_count)):
            candidates.append(
                {
                    "event_description": str(spec.description_template.format(threshold=int(threshold))),
                    "property": str(spec.property_name),
                    "threshold": int(threshold),
                    "favorable_pairs": list(favorable_pairs),
                }
            )
    return candidates


__all__ = [
    "CONDITIONAL_ANSWER_TARGETS",
    "DICE_COLOR_PALETTE",
    "PROBABILITY_OPTION_LABELS",
    "SINGLE_COLOR_OR_ANSWER_TARGETS",
    "ThresholdEventSpec",
    "build_probability_option_set",
    "build_conditional_dataset",
    "build_pair_dataset",
    "build_single_dataset",
    "color_names",
    "conditional_value_properties",
    "format_fraction",
    "normalize_int_with_bounds",
    "numeric_properties",
    "pair_ids",
    "pair_sum_threshold_candidates",
    "parse_fraction_label",
    "sample_value_property_given_color_dice",
    "select_event_candidate",
    "threshold_at_least",
    "threshold_at_most",
    "valid_favorable_count",
    "value_threshold_candidates",
]
