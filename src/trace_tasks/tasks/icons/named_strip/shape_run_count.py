"""Count separate consecutive runs of a named icon shape in a row."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from ._lifecycle import (
    NamedStripOutputPlan,
    build_named_strip_task_output,
    int_bounds_from_defaults,
    sample_named_strip_common_fields,
)
from .shared.defaults import DOMAIN, SCENE_ID
from .shared.sampling import (
    target_runs as named_strip_target_runs,
)


TASK_ID = "task_icons__named_strip__shape_run_count"

QUERY_IDS: Tuple[str, ...] = ("single",)
PROMPT_QUERY_KEY = "shape_run_count"


@dataclass(frozen=True)
class _TaskDefaults:
    """Stable fallback defaults for named-icon run-count rows."""

    strip_length_min: int = 12
    strip_length_max: int = 16
    run_count_min: int = 1
    run_count_max: int = 4


@dataclass(frozen=True)
class _SampleSpec:
    """Symbolic sample for one named-icon run-count row."""

    query_id: str
    target_shape_id: str
    target_shape_name: str
    answer: int
    strip_length: int
    shape_ids: Tuple[str, ...]
    selected_run_start_indices: Tuple[int, ...]
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


def _query_id(params: Mapping[str, Any]) -> str:
    query_id = str(params.get("query_id", QUERY_IDS[0]))
    if query_id not in set(QUERY_IDS):
        raise ValueError(f"unsupported query_id: {query_id}")
    return query_id


def _answer_support(params: Mapping[str, Any]) -> Tuple[int, ...]:
    low, high = int_bounds_from_defaults(
        params,
        _GEN_DEFAULTS,
        low_key="run_count_min",
        high_key="run_count_max",
        fallback_low=_DEFAULTS.run_count_min,
        fallback_high=_DEFAULTS.run_count_max,
    )
    if low < 1:
        raise ValueError("run count support must start at 1 or higher")
    return tuple(range(int(low), int(high) + 1))


def _target_runs(shape_ids: Sequence[str], *, target_shape_id: str) -> Tuple[Tuple[int, int], ...]:
    return named_strip_target_runs(shape_ids, target_shape_id=str(target_shape_id))


def _construct_shape_row(
    rng,
    *,
    support: Sequence[str],
    target_shape_id: str,
    run_count: int,
    strip_length: int,
) -> Tuple[Tuple[str, ...], Tuple[int, ...], Tuple[Tuple[int, int], ...]]:
    """Construct one row with exactly ``run_count`` target-shape runs."""

    if int(run_count) < 1:
        raise ValueError("run_count must be at least 1")
    min_needed = int(run_count) + max(0, int(run_count) - 1)
    if int(min_needed) > int(strip_length):
        raise ValueError("target runs do not fit strip length")

    distractor_support = tuple(str(value) for value in support if str(value) != str(target_shape_id))
    if not distractor_support:
        raise ValueError("target row needs at least one distractor shape")

    run_lengths = [1 for _ in range(int(run_count))]
    gaps = [0] + ([1] * max(0, int(run_count) - 1)) + [0]
    remaining = int(strip_length) - int(min_needed)
    for _ in range(max(0, int(remaining))):
        if float(rng.random()) < 0.45:
            run_lengths[int(rng.randint(0, len(run_lengths) - 1))] += 1
        else:
            gaps[int(rng.randint(0, len(gaps) - 1))] += 1

    row: List[str] = []
    selected_run_start_indices: List[int] = []
    for run_index, run_length in enumerate(run_lengths):
        for _ in range(int(gaps[int(run_index)])):
            row.append(str(rng.choice(distractor_support)))
        run_start = len(row)
        selected_run_start_indices.append(int(run_start))
        for _ in range(int(run_length)):
            row.append(str(target_shape_id))
    for _ in range(int(gaps[-1])):
        row.append(str(rng.choice(distractor_support)))

    runs = _target_runs(row, target_shape_id=str(target_shape_id))
    run_starts = tuple(int(start) for start, _ in runs)
    if len(runs) != int(run_count):
        raise ValueError("constructed row does not have the requested target run count")
    if tuple(int(index) for index in selected_run_start_indices) != run_starts:
        raise ValueError("selected run starts do not match constructed target runs")
    return (
        tuple(str(value) for value in row),
        tuple(int(index) for index in selected_run_start_indices),
        tuple((int(start), int(end)) for start, end in runs),
    )


def _sample_spec(*, instance_seed: int, params: Mapping[str, Any]) -> _SampleSpec:
    """Sample the task-owned run-count program before rendering."""

    rng = spawn_rng(int(instance_seed), f"{TASK_ID}:sample")
    query_id = _query_id(params)
    query_probabilities = {QUERY_IDS[0]: 1.0}
    answer_support = _answer_support(params)
    common = sample_named_strip_common_fields(
        rng=rng,
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        answer_support=answer_support,
        explicit_answer_keys=("run_count", "target_run_count", "answer"),
        min_strip_length=1,
        fallback_strip_length_min=_DEFAULTS.strip_length_min,
        fallback_strip_length_max=_DEFAULTS.strip_length_max,
        min_strip_length_by_answer={int(value): (2 * int(value)) - 1 for value in answer_support},
    )
    shape_ids, selected_run_start_indices, target_runs = _construct_shape_row(
        rng,
        support=common.shape_support,
        target_shape_id=str(common.target_shape_id),
        run_count=int(common.answer),
        strip_length=int(common.strip_length),
    )
    return _SampleSpec(
        query_id=str(query_id),
        target_shape_id=str(common.target_shape_id),
        target_shape_name=str(common.target_shape_name),
        answer=int(common.answer),
        strip_length=int(common.strip_length),
        shape_ids=tuple(str(value) for value in shape_ids),
        selected_run_start_indices=tuple(int(index) for index in selected_run_start_indices),
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
class IconsNamedStripShapeRunCountTask:
    """Ask for the number of separate consecutive runs of a named icon shape."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'topology')
    domain = DOMAIN
    supported_query_ids = QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one run-count task instance from a sampled symbolic row."""

        sample = _sample_spec(instance_seed=int(instance_seed), params=params)
        plan = NamedStripOutputPlan(
            query_id=str(sample.query_id),
            prompt_query_key=PROMPT_QUERY_KEY,
            target_shape_id=str(sample.target_shape_id),
            target_shape_name=str(sample.target_shape_name),
            answer=int(sample.answer),
            strip_length=int(sample.strip_length),
            shape_ids=sample.shape_ids,
            selected_indices=sample.selected_run_start_indices,
            target_runs=sample.target_runs,
            query_probabilities=sample.query_probabilities,
            answer_probabilities=sample.answer_probabilities,
            strip_length_probabilities=sample.strip_length_probabilities,
            shape_probabilities=sample.shape_probabilities,
            fill_style_support=sample.fill_style_support,
            fill_style_probabilities=sample.fill_style_probabilities,
            scene_kind="icons_named_strip_run_count",
            row_rule="separate_consecutive_target_shape_runs",
            question_format="named_shape_run_count",
            selected_indices_key="selected_run_start_indices",
            selected_instance_ids_key="selected_run_start_instance_ids",
            query_params_extra={"run_count": int(sample.answer)},
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


__all__ = ["IconsNamedStripShapeRunCountTask"]
