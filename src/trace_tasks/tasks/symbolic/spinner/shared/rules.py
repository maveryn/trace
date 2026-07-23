"""Neutral spinner probability sampling and probability helpers."""

from __future__ import annotations

from math import gcd
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from .....core.seed import spawn_rng
from .....core.sampling import uniform_choice
from ...shared.common import get_int_param as _get_int
from ...shared.common import get_int_range as _get_range


COLOR_PALETTE: Tuple[Tuple[str, Tuple[int, int, int]], ...] = (
    ("red", (213, 76, 76)),
    ("blue", (58, 112, 194)),
    ("green", (55, 151, 103)),
    ("yellow", (232, 184, 57)),
    ("purple", (139, 101, 201)),
    ("orange", (220, 126, 58)),
    ("teal", (42, 154, 166)),
    ("pink", (207, 88, 143)),
)
SHAPE_POOL: Tuple[str, ...] = ("circle", "triangle", "square", "diamond", "star")
PROBABILITY_OPTION_LABELS: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")


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
    if int(denominator) <= 0:
        raise ValueError("probability denominator must be positive")
    return int(numerator), int(denominator)


def _fraction_value(label: str) -> float:
    numerator, denominator = parse_fraction_label(str(label))
    return float(numerator) / float(denominator)


def _candidate_fraction_labels(favorable: int, total: int) -> List[str]:
    """Return plausible reduced-fraction distractors for one spinner event."""

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

    for delta in (-2, -1, 1, 2):
        add(favorable + int(delta), total)
    add(total - favorable, total)
    for denominator_delta in (-2, -1, 1, 2, 3):
        add(favorable, total + int(denominator_delta))
        add(favorable - 1, total + int(denominator_delta))
        add(favorable + 1, total + int(denominator_delta))
    add(total - favorable - 1, total)
    add(total - favorable + 1, total)

    correct_value = _fraction_value(correct)
    global_candidates: List[str] = []
    max_denominator = max(12, min(64, int(total) + 24))
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
    """Bind one exact reduced probability fraction to fixed visible option labels."""

    option_labels = tuple(str(label) for label in labels)
    if len(option_labels) != 6:
        raise ValueError("spinner probability MCQ tasks require exactly six option labels")
    if len(set(option_labels)) != len(option_labels):
        raise ValueError("spinner probability option labels must be unique")
    correct_fraction = format_fraction(int(favorable), int(total))
    requested_label = str(correct_label).strip().upper() if correct_label is not None else ""
    if requested_label:
        if requested_label not in option_labels:
            raise ValueError("explicit correct_label is outside spinner probability option labels")
        selected_label = str(requested_label)
    else:
        selected_label = str(uniform_choice(rng, option_labels))

    candidates = _candidate_fraction_labels(int(favorable), int(total))
    if len(candidates) < len(option_labels) - 1:
        raise RuntimeError("failed to build enough spinner probability distractors")
    nearby_pool = list(candidates[: min(30, len(candidates))])
    rng.shuffle(nearby_pool)
    distractors = list(nearby_pool[: len(option_labels) - 1])
    if len(distractors) < len(option_labels) - 1:
        raise RuntimeError("failed to sample enough spinner probability distractors")

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


def normalize_int_with_bounds(value: int, bounds: Sequence[int]) -> float:
    """Normalize an integer within inclusive calibration bounds."""

    low = int(bounds[0])
    high = int(bounds[1])
    if int(high) <= int(low):
        return 0.0
    return max(0.0, min(1.0, (float(value) - float(low)) / float(high - low)))


def valid_favorable_count(count: int, total: int, *, min_count: int, max_count: int) -> bool:
    """Check that an event is nontrivial and within configured support."""

    return int(min_count) <= int(count) <= min(int(total) - 1, int(max_count))


def color_names(sectors: Sequence[Mapping[str, Any]]) -> List[str]:
    """Return sorted color names present in spinner sectors."""

    return sorted({str(sector["color_name"]) for sector in sectors})


def shape_names(sectors: Sequence[Mapping[str, Any]]) -> List[str]:
    """Return sorted marker-shape names present in spinner sectors."""

    return sorted({str(sector["shape"]) for sector in sectors})


def configured_single_count_limits(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    *,
    total: int,
) -> tuple[int, int]:
    """Resolve the single-spinner favorable-count range."""

    min_count = _get_int(params, gen_defaults, "single_favorable_count_min", 2)
    max_count = _get_int(params, gen_defaults, "single_favorable_count_max", max(2, int(total) - 2))
    return int(min_count), int(max_count)


def color_shape_event_candidates(
    *,
    sectors: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    operator: str,
) -> List[Dict[str, Any]]:
    """Return candidates for one color/shape conjunction or disjunction event."""

    total = len(sectors)
    min_count, max_count = configured_single_count_limits(params, gen_defaults, total=int(total))
    candidates: List[Dict[str, Any]] = []
    operator_text = str(operator)
    if operator_text not in {"and", "or"}:
        raise ValueError(f"unsupported color/shape spinner operator: {operator}")
    for color in color_names(sectors):
        for shape in shape_names(sectors):
            if operator_text == "and":
                favorable = [
                    str(sector["sector_id"])
                    for sector in sectors
                    if str(sector["color_name"]) == str(color) and str(sector["shape"]) == str(shape)
                ]
            else:
                favorable = [
                    str(sector["sector_id"])
                    for sector in sectors
                    if str(sector["color_name"]) == str(color) or str(sector["shape"]) == str(shape)
                ]
            if valid_favorable_count(len(favorable), total, min_count=min_count, max_count=max_count):
                candidates.append(
                    {
                        "event_description": f"{color} {operator_text} marked with a {shape}",
                        "target_color": str(color),
                        "target_shape": str(shape),
                        "favorable_sector_ids": list(favorable),
                    }
                )
    return candidates


def sample_spinner_sectors(
    *,
    spinner_id: str,
    sector_count: int,
    rng,
    color_pool_size: int,
    number_min: int,
    number_max: int,
    show_number: bool = True,
    show_shape: bool = True,
) -> List[Dict[str, Any]]:
    """Sample visible sector records for one equal-sector spinner."""

    color_pool = list(COLOR_PALETTE[: max(3, min(len(COLOR_PALETTE), int(color_pool_size)))])
    sectors: List[Dict[str, Any]] = []
    for index in range(int(sector_count)):
        color_name, color_rgb = uniform_choice(rng, tuple(color_pool), sort_keys=False)
        shape = str(uniform_choice(rng, SHAPE_POOL, sort_keys=False))
        number = int(rng.randint(int(number_min), int(number_max)))
        sectors.append(
            {
                "sector_id": f"{spinner_id}_sector_{index}",
                "spinner_id": str(spinner_id),
                "sector_index": int(index),
                "color_name": str(color_name),
                "color_rgb": [int(value) for value in color_rgb],
                "shape": str(shape),
                "number": int(number),
                "show_number": bool(show_number),
                "show_shape": bool(show_shape),
            }
        )
    return sectors


def select_event_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    rng,
    favorable_key: str,
    total_outcome_count: int,
) -> Dict[str, Any]:
    """Choose one event candidate and bind reduced probability fields."""

    if not candidates:
        raise RuntimeError("failed to choose a nontrivial spinner probability event")
    shuffled = [dict(candidate) for candidate in candidates]
    rng.shuffle(shuffled)
    selected = dict(shuffled[0])
    favorable_count = len(selected[str(favorable_key)])
    selected["favorable_outcome_count"] = int(favorable_count)
    selected["total_outcome_count"] = int(total_outcome_count)
    selected["answer_value"] = format_fraction(int(favorable_count), int(total_outcome_count))
    return selected


def build_single_spinner_dataset(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    rng_namespace: str,
    event_builder: Callable[..., Mapping[str, Any]],
) -> Dict[str, Any]:
    """Build a single-spinner dataset using a task-owned event builder."""

    rng = spawn_rng(int(instance_seed), str(rng_namespace))
    sector_min, sector_max = _get_range(
        params,
        gen_defaults,
        min_key="single_sector_count_min",
        max_key="single_sector_count_max",
        fallback_min=6,
        fallback_max=10,
    )
    color_pool_size = _get_int(params, gen_defaults, "single_color_pool_size", 6)
    number_min, number_max = _get_range(
        params,
        gen_defaults,
        min_key="number_min",
        max_key="number_max",
        fallback_min=1,
        fallback_max=12,
    )
    for _attempt in range(300):
        sector_count = int(params.get("single_sector_count", rng.randint(int(sector_min), int(sector_max))))
        sectors = sample_spinner_sectors(
            spinner_id="spinner",
            sector_count=int(sector_count),
            rng=rng,
            color_pool_size=int(color_pool_size),
            number_min=int(number_min),
            number_max=int(number_max),
            show_number=False,
            show_shape=True,
        )
        try:
            event = dict(
                event_builder(
                    sectors=sectors,
                    params=params,
                    gen_defaults=gen_defaults,
                    rng=rng,
                )
            )
        except RuntimeError:
            continue
        return {
            "mode": "single",
            "spinner_specs": [{"spinner_id": "spinner", "title": "Spinner", "sectors": sectors}],
            "sector_count": int(sector_count),
            "sector_count_range": [int(sector_min), int(sector_max)],
            "event": event,
            "answer_value": str(event["answer_value"]),
            "calculation_supporting_item_ids": list(event["favorable_sector_ids"]),
            "annotation_item_ids": ["spinner_panel"],
        }
    raise RuntimeError("failed to build single-spinner probability dataset")


def build_pair_spinner_dataset(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    rng_namespace: str,
    event_builder: Callable[..., Mapping[str, Any]],
) -> Dict[str, Any]:
    """Build a two-spinner dataset using a task-owned event builder."""

    rng = spawn_rng(int(instance_seed), str(rng_namespace))
    sector_min, sector_max = _get_range(
        params,
        gen_defaults,
        min_key="pair_sector_count_min",
        max_key="pair_sector_count_max",
        fallback_min=4,
        fallback_max=6,
    )
    color_pool_size = _get_int(params, gen_defaults, "pair_color_pool_size", 5)
    number_min, number_max = _get_range(
        params,
        gen_defaults,
        min_key="number_min",
        max_key="number_max",
        fallback_min=1,
        fallback_max=12,
    )
    for _attempt in range(500):
        count_a = int(params.get("pair_sector_count_a", rng.randint(int(sector_min), int(sector_max))))
        count_b = int(params.get("pair_sector_count_b", rng.randint(int(sector_min), int(sector_max))))
        sectors_a = sample_spinner_sectors(
            spinner_id="spinner_a",
            sector_count=int(count_a),
            rng=rng,
            color_pool_size=int(color_pool_size),
            number_min=int(number_min),
            number_max=int(number_max),
            show_number=False,
            show_shape=False,
        )
        sectors_b = sample_spinner_sectors(
            spinner_id="spinner_b",
            sector_count=int(count_b),
            rng=rng,
            color_pool_size=int(color_pool_size),
            number_min=int(number_min),
            number_max=int(number_max),
            show_number=False,
            show_shape=False,
        )
        try:
            event = dict(
                event_builder(
                    sectors_a=sectors_a,
                    sectors_b=sectors_b,
                    params=params,
                    gen_defaults=gen_defaults,
                    rng=rng,
                )
            )
        except RuntimeError:
            continue
        return {
            "mode": "pair",
            "spinner_specs": [
                {"spinner_id": "spinner_a", "title": "Spinner A", "sectors": sectors_a},
                {"spinner_id": "spinner_b", "title": "Spinner B", "sectors": sectors_b},
            ],
            "sector_count_a": int(count_a),
            "sector_count_b": int(count_b),
            "sector_count_range": [int(sector_min), int(sector_max)],
            "event": event,
            "answer_value": str(event["answer_value"]),
            "calculation_supporting_item_ids": list(event["supporting_sector_ids"]),
            "annotation_item_ids": ["spinner_a_panel", "spinner_b_panel"],
        }
    raise RuntimeError("failed to build pair-spinner probability dataset")


__all__ = [
    "COLOR_PALETTE",
    "PROBABILITY_OPTION_LABELS",
    "SHAPE_POOL",
    "build_probability_option_set",
    "build_pair_spinner_dataset",
    "build_single_spinner_dataset",
    "color_names",
    "color_shape_event_candidates",
    "configured_single_count_limits",
    "format_fraction",
    "normalize_int_with_bounds",
    "parse_fraction_label",
    "select_event_candidate",
    "shape_names",
    "valid_favorable_count",
]
