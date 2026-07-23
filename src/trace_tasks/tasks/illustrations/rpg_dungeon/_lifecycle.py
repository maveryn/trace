"""Identity-free retry plumbing for RPG dungeon rendered count scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, Mapping, Sequence, TypeVar

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.config_defaults import (
    required_group_defaults,
    split_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.illustrations.shared.rpg_tile_profiles import resolve_rpg_tile_render_params

from .shared.output import (
    _json_safe,
    bbox_set_projection,
    rounded_bbox_set,
    rpg_dungeon_render_spec,
    rpg_dungeon_scene_ir,
)
from .shared.prompts import build_rpg_dungeon_prompt_artifacts
from .shared.rendering import DEFAULT_TILE_PX, MAX_TOTAL_CHEST_COUNT, MIN_TOTAL_CHEST_COUNT, SCENE_ID
from .shared.rendering import render_rpg_dungeon_profile_scene
from .shared.sampling import select_count_from_support
from .shared.state import RpgDungeonScene


WitnessT = TypeVar("WitnessT")


@dataclass(frozen=True)
class RenderedCountScene(Generic[WitnessT]):
    """Rendered scene plus task-owned witness data."""

    scene: RpgDungeonScene
    witnesses: WitnessT


@dataclass(frozen=True)
class CountTarget:
    """Sampled count target and render profile for one public objective."""

    query_id: str
    query_probabilities: Mapping[str, float]
    total_value: int
    target_value: int
    total_probabilities: Mapping[str, float]
    target_probabilities: Mapping[str, float]
    render_params: Mapping[str, Any]


@dataclass(frozen=True)
class CountWitnesses:
    """Task-owned answer, annotation, and verifier fields."""

    answer: int
    annotation_bboxes: Sequence[Sequence[float]]
    relations: Mapping[str, Any]
    execution_fields: Mapping[str, Any]
    witness_fields: Mapping[str, Any]


@dataclass(frozen=True)
class CountPromptKeys:
    """Prompt-default keys selected by a public count task."""

    answer_hint: str
    annotation_hint: str
    json_example: str
    json_example_answer_only: str

    def required(self) -> tuple[str, ...]:
        return (
            "bundle_id",
            "scene_key",
            "task_key",
            "json_output_contract",
            "json_output_contract_answer_only",
            str(self.answer_hint),
            str(self.annotation_hint),
            str(self.json_example),
            str(self.json_example_answer_only),
        )


@dataclass(frozen=True)
class CountTaskConfig:
    """Identity values and parameter keys supplied by one public count task."""

    task_identifier: str
    supported_query_ids: Sequence[str]
    prompt_query_key: str
    operation: str
    prompt_keys: CountPromptKeys
    total_support_key: str
    total_explicit_key: str
    total_fallback_support: Sequence[int]
    total_namespace_suffix: str
    target_support_key: str
    target_explicit_key: str
    target_fallback_support: Sequence[int]
    target_namespace_suffix: str
    target_max_delta: int = 0


def make_count_task_config(
    *,
    task_identifier: str,
    supported_query_ids: Sequence[str],
    prompt_query_key: str,
    operation: str,
    target_support_key: str,
    target_explicit_key: str,
    target_fallback_support: Sequence[int],
    target_namespace_suffix: str,
    target_max_delta: int = 0,
) -> CountTaskConfig:
    """Create the standard count-task configuration from public semantic keys."""

    prompt_suffix = f"rpg_dungeon_{prompt_query_key}"
    return CountTaskConfig(
        task_identifier=str(task_identifier),
        supported_query_ids=tuple(str(value) for value in supported_query_ids),
        prompt_query_key=str(prompt_query_key),
        operation=str(operation),
        prompt_keys=CountPromptKeys(
            answer_hint=f"answer_hint_{prompt_suffix}",
            annotation_hint=f"annotation_hint_{prompt_suffix}",
            json_example=f"json_example_{prompt_suffix}",
            json_example_answer_only=f"json_example_answer_only_{prompt_suffix}",
        ),
        total_support_key="total_chest_count_support",
        total_explicit_key="total_chest_count",
        total_fallback_support=tuple(range(MIN_TOTAL_CHEST_COUNT, MAX_TOTAL_CHEST_COUNT + 1)),
        total_namespace_suffix="total_chest_count",
        target_support_key=str(target_support_key),
        target_explicit_key=str(target_explicit_key),
        target_fallback_support=tuple(int(value) for value in target_fallback_support),
        target_namespace_suffix=str(target_namespace_suffix),
        target_max_delta=int(target_max_delta),
    )


def count_task_defaults(
    task_identifier: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Load scene defaults for one public count task without routing behavior."""

    defaults = get_scene_defaults("illustrations", SCENE_ID)
    return split_scene_generation_rendering_prompt_defaults(
        defaults if isinstance(defaults, Mapping) else {},
        task_id=str(task_identifier),
    )


def count_prompt_slots(
    prompt_defaults: Mapping[str, Any],
    prompt_keys: CountPromptKeys,
) -> dict[str, str]:
    """Resolve the standard answer and annotation prompt slots."""

    return {
        "json_output_contract": str(prompt_defaults["json_output_contract"]),
        "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
        "answer_hint": str(prompt_defaults[prompt_keys.answer_hint]),
        "annotation_hint": str(prompt_defaults[prompt_keys.annotation_hint]),
        "json_example": str(prompt_defaults[prompt_keys.json_example]),
        "json_example_answer_only": str(prompt_defaults[prompt_keys.json_example_answer_only]),
    }


def select_configured_count(
    *,
    instance_seed: int,
    task_identifier: str,
    task_params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[int],
    namespace_suffix: str,
    max_value: int | None = None,
) -> tuple[int, Mapping[str, float]]:
    """Select one configured integer operand for a public count task."""

    return select_count_from_support(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=generation_defaults,
        support_key=str(support_key),
        explicit_key=str(explicit_key),
        fallback_support=tuple(int(value) for value in fallback_support),
        namespace=f"{task_identifier}:{namespace_suffix}",
        max_value=max_value,
    )


def resolve_count_render_params(
    *,
    instance_seed: int,
    task_identifier: str,
    task_params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Resolve RPG tile profile parameters for a count task render."""

    return resolve_rpg_tile_render_params(
        task_params,
        rendering_defaults,
        tile_px_key="rpg_dungeon_tile_px",
        fallback_tile_px=DEFAULT_TILE_PX,
        instance_seed=int(instance_seed),
        namespace=f"{task_identifier}:canvas_profile",
    )


def sample_count_target(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    config: CountTaskConfig,
) -> CountTarget:
    """Sample query, total count, target count, and render profile."""

    from trace_tasks.tasks.shared.fixed_query import select_task_query_id

    query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in config.supported_query_ids),
        default_query_id=str(tuple(config.supported_query_ids)[0]),
        task_id=str(config.task_identifier),
        namespace=f"{config.task_identifier}:query",
    )
    total_value, total_probabilities = select_configured_count(
        instance_seed=int(instance_seed),
        task_identifier=str(config.task_identifier),
        task_params=task_params,
        generation_defaults=generation_defaults,
        support_key=str(config.total_support_key),
        explicit_key=str(config.total_explicit_key),
        fallback_support=tuple(int(value) for value in config.total_fallback_support),
        namespace_suffix=str(config.total_namespace_suffix),
    )
    target_value, target_probabilities = select_configured_count(
        instance_seed=int(instance_seed),
        task_identifier=str(config.task_identifier),
        task_params=task_params,
        generation_defaults=generation_defaults,
        support_key=str(config.target_support_key),
        explicit_key=str(config.target_explicit_key),
        fallback_support=tuple(int(value) for value in config.target_fallback_support),
        namespace_suffix=str(config.target_namespace_suffix),
        max_value=int(total_value) + int(config.target_max_delta),
    )
    render_params = resolve_count_render_params(
        instance_seed=int(instance_seed),
        task_identifier=str(config.task_identifier),
        task_params=task_params,
        rendering_defaults=rendering_defaults,
    )
    return CountTarget(
        query_id=str(query_id),
        query_probabilities=dict(query_probabilities),
        total_value=int(total_value),
        target_value=int(target_value),
        total_probabilities=dict(total_probabilities),
        target_probabilities=dict(target_probabilities),
        render_params=dict(render_params),
    )


def render_bound_count_scene(
    *,
    instance_seed: int,
    max_attempts: int,
    render_params: Mapping[str, Any],
    scene_kwargs: Mapping[str, Any],
    bind_scene: Callable[[RpgDungeonScene], WitnessT],
) -> RenderedCountScene[WitnessT]:
    """Render a sampled dungeon repeatedly until the task-owned binder accepts it."""

    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        try:
            scene_seed = int(instance_seed) + int(attempt) * 1009
            scene = render_rpg_dungeon_profile_scene(
                scene_seed,
                render_params=render_params,
                tile_px=int(render_params["tile_px"]),
                **dict(scene_kwargs),
            )
            return RenderedCountScene(scene=scene, witnesses=bind_scene(scene))
        except Exception as exc:  # pragma: no cover - layout feasibility is retry based.
            last_error = exc
    raise RuntimeError(f"could not generate RPG dungeon count scene: {last_error}") from last_error


def run_count_task(
    *,
    domain: str,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    config: CountTaskConfig,
    build_scene_kwargs: Callable[[CountTarget, int], Mapping[str, Any]],
    bind_scene: Callable[[RpgDungeonScene, CountTarget, Mapping[str, Any]], CountWitnesses],
    build_render_map: Callable[[RpgDungeonScene], Mapping[str, Any]],
) -> TaskOutput:
    """Run a public count task using task-owned semantic hooks."""

    target = sample_count_target(
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        rendering_defaults=rendering_defaults,
        config=config,
    )
    scene_kwargs = dict(build_scene_kwargs(target, int(instance_seed)))
    rendered = render_bound_count_scene(
        instance_seed=int(instance_seed),
        max_attempts=int(max_attempts),
        render_params=target.render_params,
        scene_kwargs=scene_kwargs,
        bind_scene=lambda scene: bind_scene(scene, target, scene_kwargs),
    )
    query_params = {
        "query_id": str(target.query_id),
        "prompt_query_key": str(config.prompt_query_key),
        "query_id_probabilities": dict(target.query_probabilities),
        str(config.total_explicit_key): int(target.total_value),
        f"{config.total_explicit_key}_probabilities": dict(target.total_probabilities),
        str(config.target_explicit_key): int(target.target_value),
        f"{config.target_explicit_key}_probabilities": dict(target.target_probabilities),
        **scene_kwargs,
        "canvas_profile": str(target.render_params.get("canvas_profile", "")),
        "canvas_profile_probabilities": dict(target.render_params.get("canvas_profile_probabilities", {})),
    }
    return package_count_result(
        domain=str(domain),
        task_identifier=str(config.task_identifier),
        scene=rendered.scene,
        query_identifier=str(target.query_id),
        prompt_query_key=str(config.prompt_query_key),
        prompt_defaults=prompt_defaults,
        prompt_keys=config.prompt_keys,
        instance_seed=int(instance_seed),
        query_params=query_params,
        operation=str(config.operation),
        relations=rendered.witnesses.relations,
        render_map=build_render_map(scene=rendered.scene),
        execution_fields=rendered.witnesses.execution_fields,
        witness_fields=rendered.witnesses.witness_fields,
        annotation_bboxes=rendered.witnesses.annotation_bboxes,
        answer_value=int(rendered.witnesses.answer),
    )


def package_count_result(
    *,
    domain: str,
    task_identifier: str,
    scene: RpgDungeonScene,
    query_identifier: str,
    prompt_query_key: str,
    prompt_defaults: Mapping[str, Any],
    prompt_keys: CountPromptKeys,
    instance_seed: int,
    query_params: Mapping[str, Any],
    operation: str,
    relations: Mapping[str, Any],
    render_map: Mapping[str, Any],
    execution_fields: Mapping[str, Any],
    witness_fields: Mapping[str, Any],
    annotation_bboxes: Sequence[Sequence[float]],
    answer_value: int,
) -> TaskOutput:
    """Package a fully bound public count task result into Trace output."""

    checked_prompt_defaults = required_group_defaults(
        prompt_defaults,
        list(prompt_keys.required()),
        context=f"prompt defaults for {task_identifier}",
    )
    prompt_artifacts = build_rpg_dungeon_prompt_artifacts(
        domain=str(domain),
        scene_id=SCENE_ID,
        prompt_defaults=checked_prompt_defaults,
        prompt_query_key=str(prompt_query_key),
        slots=count_prompt_slots(checked_prompt_defaults, prompt_keys),
        instance_seed=int(instance_seed),
    )
    annotation_value = rounded_bbox_set(annotation_bboxes)
    trace_payload = count_trace_payload(
        domain=str(domain),
        scene=scene,
        task_identifier=str(task_identifier),
        query_identifier=str(query_identifier),
        prompt_query_key=str(prompt_query_key),
        prompt_artifacts=prompt_artifacts,
        query_params=query_params,
        operation=str(operation),
        relations=relations,
        render_map=render_map,
        execution_fields=execution_fields,
        witness_fields=witness_fields,
        annotation_value=annotation_value,
    )
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants={
            str(key): str(value)
            for key, value in prompt_artifacts.prompt_variants.items()
        },
        answer_gt=TypedValue(type="integer", value=int(answer_value)),
        annotation_gt=TypedValue(type="bbox_set", value=annotation_value),
        image=scene.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(query_identifier),
    )


def count_trace_payload(
    *,
    domain: str,
    scene: RpgDungeonScene,
    task_identifier: str,
    query_identifier: str,
    prompt_query_key: str,
    prompt_artifacts: Any,
    query_params: Mapping[str, Any],
    operation: str,
    relations: Mapping[str, Any],
    render_map: Mapping[str, Any],
    execution_fields: Mapping[str, Any],
    witness_fields: Mapping[str, Any],
    annotation_value: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Assemble the neutral trace skeleton for bbox-set count tasks."""

    annotation = rounded_bbox_set(annotation_value)
    return {
        "scene_ir": rpg_dungeon_scene_ir(
            domain=str(domain),
            scene_id=SCENE_ID,
            scene=scene,
            relations={
                "operation": str(operation),
                "query_id": str(query_identifier),
                "prompt_query_key": str(prompt_query_key),
                **_json_safe(dict(relations)),
            },
        ),
        "query_spec": {
            "task_id": str(task_identifier),
            "query_id": str(query_identifier),
            "prompt_query_key": str(prompt_query_key),
            "prompt_variant_active_key": prompt_artifacts.prompt_variant_active_key,
            "prompt_variant": _json_safe(dict(prompt_artifacts.prompt_variant)),
            "prompt_variants": _json_safe(dict(prompt_artifacts.prompt_variants_for_trace)),
            "params": _json_safe(dict(query_params)),
        },
        "render_spec": rpg_dungeon_render_spec(scene, scene_id=SCENE_ID),
        "render_map": _json_safe(dict(render_map)),
        "execution_trace": {
            "query_id": str(query_identifier),
            "prompt_query_key": str(prompt_query_key),
            "scene_id": SCENE_ID,
            "operation": str(operation),
            **_json_safe(dict(execution_fields)),
            "renderer": _json_safe(dict(scene.trace)),
        },
        "witness_symbolic": _json_safe(dict(witness_fields)),
        "projected_annotation": _json_safe(bbox_set_projection(annotation)),
    }


__all__ = [
    "CountPromptKeys",
    "CountTarget",
    "CountTaskConfig",
    "CountWitnesses",
    "RenderedCountScene",
    "count_task_defaults",
    "make_count_task_config",
    "package_count_result",
    "render_bound_count_scene",
    "resolve_count_render_params",
    "run_count_task",
    "sample_count_target",
    "select_configured_count",
]
