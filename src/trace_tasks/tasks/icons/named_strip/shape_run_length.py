"""Measure longest or shortest consecutive runs of named icons in a row."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from ...shared.variant_sampling import resolve_variant
from ._lifecycle import (
    NamedStripOutputPlan,
    build_named_strip_task_output,
    int_bounds_from_defaults,
    sample_named_strip_common_fields,
)
from .shared.defaults import DOMAIN, SCENE_ID
from .shared.sampling import (
    target_run_lengths as named_strip_target_run_lengths,
    target_runs as named_strip_target_runs,
)


TASK_ID = "task_icons__named_strip__shape_run_length"

QUERY_IDS: Tuple[str, ...] = (
    "longest_shape_run_length",
    "shortest_shape_run_length",
)


@dataclass(frozen=True)
class _TaskDefaults:
    """Stable fallback defaults for named-icon run rows."""

    strip_length_min: int = 12
    strip_length_max: int = 16
    longest_run_length_min: int = 2
    longest_run_length_max: int = 6
    shortest_run_length_min: int = 1
    shortest_run_length_max: int = 5


@dataclass(frozen=True)
class _SampleSpec:
    """Symbolic sample for one named-icon run row."""

    query_id: str
    target_shape_id: str
    target_shape_name: str
    answer: int
    strip_length: int
    shape_ids: Tuple[str, ...]
    selected_run_indices: Tuple[int, ...]
    target_runs: Tuple[Tuple[int, int], ...]
    shape_support: Tuple[str, ...]
    query_probabilities: Dict[str, float]
    answer_probabilities: Dict[str, float]
    strip_length_probabilities: Dict[str, float]
    shape_probabilities: Dict[str, float]
    fill_style_support: Tuple[str, ...]
    fill_style_probabilities: Dict[str, float]


_DEFAULTS = _TaskDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
    task_id=TASK_ID,
)


def _answer_support(params: Mapping[str, Any], query_id: str) -> Tuple[int, ...]:
    if str(query_id) == "longest_shape_run_length":
        low, high = int_bounds_from_defaults(
            params,
            _GEN_DEFAULTS,
            low_key="longest_run_length_min",
            high_key="longest_run_length_max",
            fallback_low=_DEFAULTS.longest_run_length_min,
            fallback_high=_DEFAULTS.longest_run_length_max,
        )
    elif str(query_id) == "shortest_shape_run_length":
        low, high = int_bounds_from_defaults(
            params,
            _GEN_DEFAULTS,
            low_key="shortest_run_length_min",
            high_key="shortest_run_length_max",
            fallback_low=_DEFAULTS.shortest_run_length_min,
            fallback_high=_DEFAULTS.shortest_run_length_max,
        )
    else:
        raise ValueError(f"unsupported query_id: {query_id}")
    if str(query_id) == "longest_shape_run_length" and low < 2:
        raise ValueError("longest run support starts at 2 so the task is not a singleton-detection branch")
    return tuple(range(int(low), int(high) + 1))


def _target_runs(shape_ids: Sequence[str], *, target_shape_id: str) -> Tuple[Tuple[int, int], ...]:
    return named_strip_target_runs(shape_ids, target_shape_id=str(target_shape_id))


def _target_run_lengths(runs: Sequence[Tuple[int, int]]) -> Tuple[int, ...]:
    return named_strip_target_run_lengths(runs)


def _build_runs_for_query(rng, *, query_id: str, answer: int, strip_length: int) -> Tuple[Tuple[int, bool], ...]:
    """Return target-run lengths, marking the run that witnesses the answer."""

    run_blocks: List[Tuple[int, bool]] = [(int(answer), True)]
    if str(query_id) == "longest_shape_run_length":
        optional_candidates = tuple(range(1, int(answer)))
        desired_optional_count = int(rng.randint(1, 4)) if optional_candidates else 0
        for _ in range(int(desired_optional_count)):
            candidate = int(rng.choice(optional_candidates))
            tentative = run_blocks + [(candidate, False)]
            min_needed = sum(length for length, _ in tentative) + max(0, len(tentative) - 1)
            if int(min_needed) <= int(strip_length):
                run_blocks.append((int(candidate), False))
    elif str(query_id) == "shortest_shape_run_length":
        max_other = min(6, int(strip_length) - int(answer) - 1)
        if max_other <= int(answer):
            raise ValueError("strip length leaves no room for a longer target-shape run")
        run_blocks.append((int(rng.randint(int(answer) + 1, int(max_other))), False))
        desired_optional_count = int(rng.randint(0, 3))
        for _ in range(int(desired_optional_count)):
            candidate = int(rng.randint(int(answer) + 1, 6))
            tentative = run_blocks + [(candidate, False)]
            min_needed = sum(length for length, _ in tentative) + max(0, len(tentative) - 1)
            if int(min_needed) <= int(strip_length):
                run_blocks.append((int(candidate), False))
    else:
        raise ValueError(f"unsupported query_id: {query_id}")
    rng.shuffle(run_blocks)
    return tuple((int(length), bool(selected)) for length, selected in run_blocks)


def _construct_shape_row(
    rng,
    *,
    support: Sequence[str],
    target_shape_id: str,
    query_id: str,
    answer: int,
    strip_length: int,
) -> Tuple[Tuple[str, ...], Tuple[int, ...], Tuple[Tuple[int, int], ...]]:
    """Construct one row with a unique target-shape extremum run."""

    run_blocks = list(
        _build_runs_for_query(
            rng,
            query_id=str(query_id),
            answer=int(answer),
            strip_length=int(strip_length),
        )
    )
    min_needed = sum(length for length, _ in run_blocks) + max(0, len(run_blocks) - 1)
    if int(min_needed) > int(strip_length):
        raise ValueError("target runs do not fit strip length")
    gaps = [0] + ([1] * max(0, len(run_blocks) - 1)) + [0]
    remaining = int(strip_length) - int(min_needed)
    for _ in range(max(0, int(remaining))):
        gaps[int(rng.randint(0, len(gaps) - 1))] += 1

    distractor_support = tuple(str(value) for value in support if str(value) != str(target_shape_id))
    if not distractor_support:
        raise ValueError("target row needs at least one distractor shape")

    row: List[str] = []
    selected_indices: List[int] = []
    for run_index, (run_length, selected) in enumerate(run_blocks):
        for _ in range(int(gaps[int(run_index)])):
            row.append(str(rng.choice(distractor_support)))
        run_start = len(row)
        for _ in range(int(run_length)):
            row.append(str(target_shape_id))
        if bool(selected):
            selected_indices.extend(range(int(run_start), int(run_start) + int(run_length)))
    for _ in range(int(gaps[-1])):
        row.append(str(rng.choice(distractor_support)))

    runs = _target_runs(row, target_shape_id=str(target_shape_id))
    lengths = _target_run_lengths(runs)
    selected_run_length = int(len(selected_indices))
    if str(query_id) == "longest_shape_run_length":
        if selected_run_length != max(lengths) or lengths.count(int(selected_run_length)) != 1:
            raise ValueError("constructed row does not have a unique longest target run")
    elif str(query_id) == "shortest_shape_run_length":
        if selected_run_length != min(lengths) or lengths.count(int(selected_run_length)) != 1:
            raise ValueError("constructed row does not have a unique shortest target run")
    return tuple(str(value) for value in row), tuple(int(index) for index in selected_indices), tuple(runs)


def _sample_spec(*, instance_seed: int, params: Mapping[str, Any]) -> _SampleSpec:
    """Sample the task-owned run-length program before rendering.

    This owns the semantic query branch, target shape, answer support, strip
    length, and unique witness-run construction. Scene helpers only receive the
    resolved row and visual style arguments.
    """

    rng = spawn_rng(int(instance_seed), f"{TASK_ID}:sample")
    query_id, query_probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        supported_variants=QUERY_IDS,
        explicit_key="query_id",
        weights_key="query_id_weights",
    )
    answer_support = _answer_support(params, str(query_id))
    if str(query_id) == "longest_shape_run_length":
        min_strip_by_answer = {int(value): int(value) for value in answer_support}
    else:
        min_strip_by_answer = {int(value): (2 * int(value)) + 2 for value in answer_support}
    common = sample_named_strip_common_fields(
        rng=rng,
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        answer_support=answer_support,
        explicit_answer_keys=("target_run_length", "answer", "run_length"),
        min_strip_length=1,
        fallback_strip_length_min=_DEFAULTS.strip_length_min,
        fallback_strip_length_max=_DEFAULTS.strip_length_max,
        min_strip_length_by_answer=min_strip_by_answer,
    )
    if str(query_id) == "shortest_shape_run_length":
        max_other = min(6, int(common.strip_length) - int(common.answer) - 1)
        if max_other <= int(common.answer):
            raise ValueError("strip length range cannot support the requested run-length query")
    shape_ids, selected_indices, target_runs = _construct_shape_row(
        rng,
        support=common.shape_support,
        target_shape_id=str(common.target_shape_id),
        query_id=str(query_id),
        answer=int(common.answer),
        strip_length=int(common.strip_length),
    )
    return _SampleSpec(
        query_id=str(query_id),
        target_shape_id=str(common.target_shape_id),
        target_shape_name=str(common.target_shape_name),
        answer=int(common.answer),
        strip_length=int(common.strip_length),
        shape_ids=tuple(str(value) for value in shape_ids),
        selected_run_indices=tuple(int(index) for index in selected_indices),
        target_runs=tuple((int(start), int(end)) for start, end in target_runs),
        shape_support=common.shape_support,
        query_probabilities=dict(query_probabilities),
        answer_probabilities=dict(common.answer_probabilities),
        strip_length_probabilities=dict(common.strip_length_probabilities),
        shape_probabilities=dict(common.shape_probabilities),
        fill_style_support=tuple(common.fill_style_support),
        fill_style_probabilities=dict(common.fill_style_probabilities),
    )


@register_task
class IconsNamedStripShapeRunLengthTask:
    """Ask for longest/shortest consecutive run length of a named icon shape."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'ranking', 'topology')
    domain = DOMAIN
    supported_query_ids = QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one run-length task instance from a sampled symbolic row.

        The task file owns answer and annotation binding from the same rendered
        selected run; shared code only supplies scene rendering, prompt, and
        trace serialization primitives.
        """

        sample = _sample_spec(instance_seed=int(instance_seed), params=params)
        plan = NamedStripOutputPlan(
            query_id=str(sample.query_id),
            prompt_query_key=str(sample.query_id),
            target_shape_id=str(sample.target_shape_id),
            target_shape_name=str(sample.target_shape_name),
            answer=int(sample.answer),
            strip_length=int(sample.strip_length),
            shape_ids=sample.shape_ids,
            selected_indices=sample.selected_run_indices,
            target_runs=sample.target_runs,
            query_probabilities=sample.query_probabilities,
            answer_probabilities=sample.answer_probabilities,
            strip_length_probabilities=sample.strip_length_probabilities,
            shape_probabilities=sample.shape_probabilities,
            fill_style_support=sample.fill_style_support,
            fill_style_probabilities=sample.fill_style_probabilities,
            scene_kind="icons_named_strip_run_length",
            row_rule="consecutive_target_shape_runs",
            question_format="named_shape_run_length",
            selected_indices_key="selected_run_indices",
            selected_instance_ids_key="selected_run_instance_ids",
        )
        output = build_named_strip_task_output(
            task_id=TASK_ID,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            render_defaults=_RENDER_DEFAULTS,
            prompt_defaults=_PROMPT_DEFAULTS,
            plan=plan,
            shape_support=sample.shape_support,
        )
        return output


__all__ = ["IconsNamedStripShapeRunLengthTask"]
