"""Private lifecycle plumbing for Minesweeper games public tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Mapping, Sequence

from PIL import Image

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.font_assets import get_font_family_record
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.games.shared.scene_style import make_panel_scene_background, resolve_game_panel_scene_style

from .shared.annotations import (
    cell_ids_for_coords,
    minesweeper_bbox_set_annotation,
    minesweeper_point_annotation,
    minesweeper_point_set_annotation,
)
from .shared.defaults import POST_IMAGE_NOISE_DEFAULTS, SCENE_ID
from .shared.output import build_minesweeper_common_trace_params, build_minesweeper_trace_payload
from .shared.prompts import MinesweeperPromptSlots, build_minesweeper_prompt_artifacts
from .shared.rendering import (
    RenderedMinesweeperScene,
    render_minesweeper_grid_scene,
    resolve_minesweeper_render_params,
)
from .shared.sampling import MinesweeperAxes
from .shared.state import Coord, MinesweeperSample


AnnotationBinder = Callable[[RenderedMinesweeperScene], AnnotationArtifacts]
AttemptBuilder = Callable[[Any, MinesweeperAxes], "MinesweeperAttemptResult"]
ObjectivePreparer = Callable[
    [int, Mapping[str, Any], str, Mapping[str, float], Mapping[str, Any]],
    "MinesweeperObjectivePlan",
]


@dataclass(frozen=True)
class MinesweeperAttemptResult:
    """Task-owned sample plus answer, prompt, and annotation hooks."""

    answer_gt: TypedValue
    sample: MinesweeperSample
    prompt_slots: MinesweeperPromptSlots
    bind_annotation: AnnotationBinder
    annotation_entity_ids: tuple[str, ...]
    keyed_annotation_entity_ids: Mapping[str, Sequence[str]] = field(default_factory=dict)
    highlighted_clue_coords: tuple[Coord, ...] = tuple()
    extra_query_params: Mapping[str, Any] = field(default_factory=dict)
    execution_extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MinesweeperObjectivePlan:
    """Prepared task-owned objective for one generated instance."""

    axes: MinesweeperAxes
    attempt_namespace: str
    construct_attempt: AttemptBuilder


def minesweeper_integer_point_attempt(
    *,
    sample: MinesweeperSample,
    prompt_key: str,
    object_description_key: str,
    answer_hint_key: str,
    annotation_hint_key: str,
    example_annotation: Any,
    example_answer: int,
    coord: Coord,
    highlighted_clue_coords: Sequence[Coord] = tuple(),
    extra_query_params: Mapping[str, Any] | None = None,
) -> MinesweeperAttemptResult:
    """Package an integer answer whose annotation is one board cell point."""

    return MinesweeperAttemptResult(
        answer_gt=TypedValue(type="integer", value=int(sample.answer)),
        sample=sample,
        prompt_slots=MinesweeperPromptSlots(
            prompt_query_key=str(prompt_key),
            object_description_key=str(object_description_key),
            answer_hint_key=str(answer_hint_key),
            annotation_hint_key=str(annotation_hint_key),
            example_annotation=example_annotation,
            example_answer=int(example_answer),
        ),
        bind_annotation=lambda rendered: minesweeper_point_annotation(rendered=rendered, coord=coord),
        annotation_entity_ids=cell_ids_for_coords((coord,)),
        highlighted_clue_coords=tuple(highlighted_clue_coords),
        extra_query_params=dict(extra_query_params or {}),
    )


def minesweeper_option_letter_point_attempt(
    *,
    sample: MinesweeperSample,
    prompt_key: str,
    object_description_key: str,
    answer_hint_key: str,
    annotation_hint_key: str,
    example_annotation: Any,
    example_answer: str,
    coord: Coord,
    highlighted_clue_coords: Sequence[Coord] = tuple(),
    execution_extra: Mapping[str, Any] | None = None,
    extra_query_params: Mapping[str, Any] | None = None,
) -> MinesweeperAttemptResult:
    """Package an option-letter answer whose annotation is one board cell point."""

    return MinesweeperAttemptResult(
        answer_gt=TypedValue(type="option_letter", value=str(sample.answer)),
        sample=sample,
        prompt_slots=MinesweeperPromptSlots(
            prompt_query_key=str(prompt_key),
            object_description_key=str(object_description_key),
            answer_hint_key=str(answer_hint_key),
            annotation_hint_key=str(annotation_hint_key),
            example_annotation=example_annotation,
            example_answer=str(example_answer),
        ),
        bind_annotation=lambda rendered: minesweeper_point_annotation(rendered=rendered, coord=coord),
        annotation_entity_ids=cell_ids_for_coords((coord,)),
        highlighted_clue_coords=tuple(highlighted_clue_coords),
        extra_query_params=dict(extra_query_params or {}),
        execution_extra=dict(execution_extra or {}),
    )


def minesweeper_integer_point_set_attempt(
    *,
    sample: MinesweeperSample,
    prompt_key: str,
    object_description_key: str,
    answer_hint_key: str,
    annotation_hint_key: str,
    example_annotation: Any,
    example_answer: int,
    coords: Sequence[Coord],
    highlighted_clue_coords: Sequence[Coord] = tuple(),
    execution_extra: Mapping[str, Any] | None = None,
    extra_query_params: Mapping[str, Any] | None = None,
) -> MinesweeperAttemptResult:
    """Package an integer answer whose annotation is a homogeneous point set."""

    return MinesweeperAttemptResult(
        answer_gt=TypedValue(type="integer", value=int(sample.answer)),
        sample=sample,
        prompt_slots=MinesweeperPromptSlots(
            prompt_query_key=str(prompt_key),
            object_description_key=str(object_description_key),
            answer_hint_key=str(answer_hint_key),
            annotation_hint_key=str(annotation_hint_key),
            example_annotation=example_annotation,
            example_answer=int(example_answer),
        ),
        bind_annotation=lambda rendered: minesweeper_point_set_annotation(rendered=rendered, coords=tuple(coords)),
        annotation_entity_ids=cell_ids_for_coords(tuple(coords)),
        highlighted_clue_coords=tuple(highlighted_clue_coords),
        extra_query_params=dict(extra_query_params or {}),
        execution_extra=dict(execution_extra or {}),
    )


def minesweeper_integer_bbox_set_attempt(
    *,
    sample: MinesweeperSample,
    prompt_key: str,
    object_description_key: str,
    answer_hint_key: str,
    annotation_hint_key: str,
    example_annotation: Any,
    example_answer: int,
    coords: Sequence[Coord],
    highlighted_clue_coords: Sequence[Coord] = tuple(),
    execution_extra: Mapping[str, Any] | None = None,
    extra_query_params: Mapping[str, Any] | None = None,
) -> MinesweeperAttemptResult:
    """Package an integer answer whose annotation is a homogeneous bbox set."""

    return MinesweeperAttemptResult(
        answer_gt=TypedValue(type="integer", value=int(sample.answer)),
        sample=sample,
        prompt_slots=MinesweeperPromptSlots(
            prompt_query_key=str(prompt_key),
            object_description_key=str(object_description_key),
            answer_hint_key=str(answer_hint_key),
            annotation_hint_key=str(annotation_hint_key),
            example_annotation=example_annotation,
            example_answer=int(example_answer),
        ),
        bind_annotation=lambda rendered: minesweeper_bbox_set_annotation(rendered=rendered, coords=tuple(coords)),
        annotation_entity_ids=cell_ids_for_coords(tuple(coords)),
        highlighted_clue_coords=tuple(highlighted_clue_coords),
        extra_query_params=dict(extra_query_params or {}),
        execution_extra=dict(execution_extra or {}),
    )


@dataclass(frozen=True)
class MinesweeperLifecycleResult:
    """Rendered prompt/image/annotation payload returned by neutral plumbing."""

    prompt: str
    prompt_variants: Mapping[str, str]
    annotation_artifacts: AnnotationArtifacts
    image: Image.Image
    trace_payload: Mapping[str, Any]


@lru_cache(maxsize=None)
def minesweeper_task_defaults(public_id: str) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Load and cache scene defaults for one public Minesweeper task."""

    return load_scene_generation_rendering_prompt_defaults(
        "games",
        SCENE_ID,
        task_id=str(public_id),
    )


def run_minesweeper_registered_task(
    task_obj: Any,
    instance_seed: int,
    *,
    params: Mapping[str, Any] | None = None,
    max_attempts: int = 100,
) -> TaskOutput:
    """Run the private Minesweeper lifecycle for a registered public task."""

    gen_defaults, render_defaults, prompt_defaults = minesweeper_task_defaults(str(task_obj.task_id))
    return run_minesweeper_lifecycle(
        public_id=str(task_obj.task_id),
        domain=str(task_obj.domain),
        gen_defaults=gen_defaults,
        render_defaults=render_defaults,
        prompt_defaults=prompt_defaults,
        instance_seed=int(instance_seed),
        params=dict(params or {}),
        max_attempts=int(max_attempts),
        supported_branches=tuple(str(value) for value in task_obj.supported_query_ids),
        default_branch=str(task_obj._default_branch),
        prepare_objective=task_obj._prepare_objective,
        namespace=str(task_obj._namespace),
    )


def run_minesweeper_lifecycle(
    *,
    public_id: str,
    domain: str,
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    supported_branches: tuple[str, ...],
    default_branch: str,
    prepare_objective: ObjectivePreparer,
    namespace: str,
) -> TaskOutput:
    """Run branch selection, retry, render, prompt, annotation, and output assembly."""

    selected_branch, branch_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=tuple(str(value) for value in supported_branches),
        default_query_id=str(default_branch),
        task_id=str(public_id),
        namespace=f"{namespace}.query",
    )
    objective = prepare_objective(
        int(instance_seed),
        task_params,
        str(selected_branch),
        dict(branch_probabilities),
        gen_defaults,
    )
    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(int(instance_seed), f"{objective.attempt_namespace}.attempt.{int(attempt_index)}")
        try:
            attempt = objective.construct_attempt(rng, objective.axes)
        except ValueError:
            continue
        lifecycle = render_minesweeper_lifecycle(
            domain=str(domain),
            selected_branch=str(selected_branch),
            branch_probabilities=dict(branch_probabilities),
            task_params=task_params,
            axes=objective.axes,
            attempt=attempt,
            prompt_defaults=prompt_defaults,
            render_defaults=render_defaults,
            instance_seed=int(instance_seed),
            namespace=str(namespace),
        )
        return TaskOutput(
            prompt=str(lifecycle.prompt),
            prompt_variants=dict(lifecycle.prompt_variants),
            answer_gt=attempt.answer_gt,
            annotation_gt=lifecycle.annotation_artifacts.annotation_gt,
            image=lifecycle.image,
            image_id="img0",
            trace_payload=dict(lifecycle.trace_payload),
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(selected_branch),
        )
    raise RuntimeError(f"{public_id} failed to generate after {max_attempts} attempts")


def render_minesweeper_lifecycle(
    *,
    domain: str,
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    task_params: Mapping[str, Any],
    axes: MinesweeperAxes,
    attempt: MinesweeperAttemptResult,
    prompt_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> MinesweeperLifecycleResult:
    """Run neutral render, prompt, noise, annotation, and trace assembly."""

    rendered_scene, background_meta, panel_style_meta, render_params = _render_scene(
        sample=attempt.sample,
        axes=axes,
        params=task_params,
        render_defaults=render_defaults,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
        highlighted_clue_coords=attempt.highlighted_clue_coords,
    )
    annotation_artifacts = attempt.bind_annotation(rendered_scene)
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=task_params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    prompt_defaults_used, prompt_artifacts = build_minesweeper_prompt_artifacts(
        domain=str(domain),
        prompt_defaults=prompt_defaults,
        slots=attempt.prompt_slots,
        instance_seed=int(instance_seed),
    )
    query_params = build_minesweeper_common_trace_params(
        axes=axes,
        branch_probabilities=branch_probabilities,
        extra_params={
            **dict(attempt.extra_query_params),
            "hidden_count": len(attempt.sample.hidden_coords),
            "flagged_count": len(attempt.sample.flagged_coords),
        },
    )
    prompt_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_branch),
        params=query_params,
    )
    trace_payload = build_minesweeper_trace_payload(
        annotation_artifacts=annotation_artifacts,
        annotation_entity_ids=attempt.annotation_entity_ids,
        keyed_annotation_entity_ids=attempt.keyed_annotation_entity_ids,
        axes=axes,
        sample=attempt.sample,
        rendered=rendered_scene,
        prompt_defaults=prompt_defaults_used,
        prompt_query_spec=prompt_spec,
        post_noise_meta=post_noise_meta,
        background_meta=background_meta,
        panel_style_meta=panel_style_meta,
        image_size=(int(image.size[0]), int(image.size[1])),
        answer_value=attempt.answer_gt.value,
        execution_extra=attempt.execution_extra,
    )
    trace_payload["scene_ir"]["relations"]["query_id"] = str(selected_branch)
    trace_payload["execution_trace"]["query_id"] = str(selected_branch)
    trace_payload["render_spec"]["font_asset"] = get_font_family_record(str(render_params.font_family)).to_trace()
    return MinesweeperLifecycleResult(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        annotation_artifacts=annotation_artifacts,
        image=image,
        trace_payload=dict(trace_payload),
    )


def _render_scene(
    *,
    sample: MinesweeperSample,
    axes: MinesweeperAxes,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
    highlighted_clue_coords: Sequence[Coord],
) -> tuple[RenderedMinesweeperScene, Mapping[str, Any], Mapping[str, Any], Any]:
    """Resolve panel styling and render one Minesweeper sample."""

    render_params = resolve_minesweeper_render_params(
        params,
        render_defaults=render_defaults,
        namespace=str(namespace),
        instance_seed=int(instance_seed),
    )
    panel_style, panel_style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.panel_scene",
        treatment_weights=render_defaults.get("panel_scene_treatment_weights"),
        palette_weights=render_defaults.get("panel_scene_palette_weights"),
    )
    background, background_meta = make_panel_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=panel_style,
    )
    rendered_scene = render_minesweeper_grid_scene(
        size=int(sample.size),
        mine_coords=sample.mine_coords,
        revealed_coords=sample.revealed_coords,
        flagged_coords=sample.flagged_coords,
        hidden_coords=sample.hidden_coords,
        background=background,
        style_variant=str(axes.style_variant),
        params=render_params,
        highlighted_clue_coords=tuple(highlighted_clue_coords),
        option_label_coords=tuple(sample.candidate_option_coords),
    )
    return rendered_scene, dict(background_meta), dict(panel_style_meta), render_params


__all__ = [
    "MinesweeperAttemptResult",
    "MinesweeperObjectivePlan",
    "minesweeper_integer_bbox_set_attempt",
    "minesweeper_integer_point_attempt",
    "minesweeper_integer_point_set_attempt",
    "minesweeper_option_letter_point_attempt",
    "run_minesweeper_lifecycle",
    "run_minesweeper_registered_task",
]
