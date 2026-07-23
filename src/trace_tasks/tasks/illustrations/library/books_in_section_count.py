"""Count all books in one labeled library section."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ....core.scene_config import get_scene_defaults
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import required_group_defaults, split_scene_generation_rendering_prompt_defaults
from ...shared.output_metadata import default_task_versions
from .shared.annotations import (
    book_bbox_map,
    library_scene_entities,
    serialize_library_scene,
    sort_library_points,
)
from .shared.output import library_base_render_map, library_render_spec, library_scene_relations, render_fallback_from_defaults
from .shared.prompts import build_library_prompt_artifacts
from .shared.rendering import render_library_scene
from .shared.sampling import (
    bounds,
    color_support,
    make_library_section_specs,
    random_book_specs,
    render_params,
    sample_count,
    section_keys_for_scene,
    section_support,
    setting_weights,
    spawned_task_rng,
    style_weights,
    support_choice,
    uniform_string_probability_map,
)
from .shared.state import LibraryBookSpec, library_section_display_name


TASK_ID = "task_illustrations__library__books_in_section_count"
SCENE_ID = "library"
QUERY_ID = SINGLE_QUERY_ID
PROMPT_QUERY_KEY = "books_in_section_count"


@dataclass(frozen=True)
class _Defaults:
    section_count_min: int = 4
    section_count_max: int = 6
    target_count_min: int = 3
    target_count_max: int = 8
    section_book_count_min: int = 5
    section_book_count_max: int = 12
    canvas_width: int = 1280
    canvas_height: int = 900
    render_scale: int = 2


@dataclass(frozen=True)
class _SampleSpec:
    section_key: str
    section_name: str
    section_count: int
    target_count: int
    section_specs: Tuple[Any, ...]
    section_keys: Tuple[str, ...]
    section_key_probabilities: Dict[str, float]
    section_count_probabilities: Dict[str, float]
    target_count_probabilities: Dict[str, float]


_DEFAULTS = _Defaults()
_SCENE_DEFAULTS = get_scene_defaults("illustrations", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)


def _sample_spec(*, instance_seed: int, params: Mapping[str, Any], attempt_index: int) -> _SampleSpec:
    """Sample exact target books in one section plus section distractors."""

    target_min, target_max = bounds(
        params,
        _GEN_DEFAULTS,
        "target_count_min",
        "target_count_max",
        _DEFAULTS.target_count_min,
        _DEFAULTS.target_count_max,
    )
    target_count, target_probabilities = sample_count(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:target_count",
        low=int(target_min),
        high=int(target_max),
        explicit_key="target_count",
    )
    section_min, section_max = bounds(
        params,
        _GEN_DEFAULTS,
        "section_count_min",
        "section_count_max",
        _DEFAULTS.section_count_min,
        _DEFAULTS.section_count_max,
    )
    section_values = section_support(params, _GEN_DEFAULTS)
    section_count, section_count_probabilities = sample_count(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:section_count",
        low=int(section_min),
        high=min(int(section_max), len(section_values)),
        explicit_key="section_count",
    )
    section_key, section_probabilities = support_choice(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:section",
        support=section_values,
        explicit_key="section_key",
    )
    rng = spawned_task_rng(int(instance_seed), TASK_ID, int(attempt_index))
    section_keys = section_keys_for_scene(
        rng=rng,
        params=params,
        defaults=_GEN_DEFAULTS,
        target_section_key=str(section_key),
        section_count=int(section_count),
    )
    colors = color_support(params, _GEN_DEFAULTS)
    section_book_min, section_book_max = bounds(
        params,
        _GEN_DEFAULTS,
        "section_book_count_min",
        "section_book_count_max",
        _DEFAULTS.section_book_count_min,
        _DEFAULTS.section_book_count_max,
    )
    specs_by_section: Dict[str, Tuple[LibraryBookSpec, ...]] = {}
    for key in section_keys:
        if str(key) == str(section_key):
            specs_by_section[str(key)] = random_book_specs(
                rng=rng,
                section_key=str(key),
                count=int(target_count),
                colors=colors,
                role="target",
            )
        else:
            count = int(rng.randint(int(section_book_min), int(section_book_max)))
            specs_by_section[str(key)] = random_book_specs(
                rng=rng,
                section_key=str(key),
                count=int(count),
                colors=colors,
                role="distractor",
            )
    return _SampleSpec(
        section_key=str(section_key),
        section_name=library_section_display_name(str(section_key)),
        section_count=int(section_count),
        target_count=int(target_count),
        section_specs=make_library_section_specs(section_keys=section_keys, specs_by_section=specs_by_section),
        section_keys=tuple(section_keys),
        section_key_probabilities=dict(section_probabilities),
        section_count_probabilities=dict(section_count_probabilities),
        target_count_probabilities=dict(target_probabilities),
    )


def _prompt_slots(sample: _SampleSpec, prompt_defaults: Mapping[str, Any]) -> Dict[str, str | int]:
    return {
        "section_count": int(sample.section_count),
        "section_name": str(sample.section_name),
        "json_output_contract": str(prompt_defaults["json_output_contract"]),
        "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
        "answer_hint": str(prompt_defaults["answer_hint_books_in_section"]).format(section_name=str(sample.section_name)),
        "annotation_hint": str(prompt_defaults["annotation_hint_books_in_section"]).format(section_name=str(sample.section_name)),
        "json_example": str(prompt_defaults["json_example_books_in_section"]),
        "json_example_answer_only": str(prompt_defaults["json_example_answer_only_books_in_section"]),
    }


@register_task
class IllustrationsLibraryBooksInSectionCountTask:
    """Count all books in one named library shelf section."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = "illustrations"
    supported_query_ids = (QUERY_ID,)
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one scoped book-count instance and bind answer/annotation locally."""

        last_error: Exception | None = None
        sample: _SampleSpec | None = None
        scene = None
        fallback = render_fallback_from_defaults(_DEFAULTS)
        for attempt in range(max(1, int(max_attempts))):
            try:
                sample = _sample_spec(instance_seed=int(instance_seed), params=params, attempt_index=int(attempt))
                scene_rng = spawn_rng(int(instance_seed), f"{TASK_ID}:scene", int(attempt))
                rp = render_params(
                    params,
                    _RENDER_DEFAULTS,
                    fallback_width=int(fallback["canvas_width"]),
                    fallback_height=int(fallback["canvas_height"]),
                    fallback_scale=int(fallback["render_scale"]),
                    instance_seed=int(instance_seed),
                    namespace=f"{TASK_ID}:canvas_profile",
                )
                scene = render_library_scene(
                    rng=scene_rng,
                    section_specs=sample.section_specs,
                    canvas_width=int(rp["canvas_width"]),
                    canvas_height=int(rp["canvas_height"]),
                    render_scale=int(rp["render_scale"]),
                    setting_weights=setting_weights(params, _RENDER_DEFAULTS),
                    style_weights=style_weights(params, _RENDER_DEFAULTS),
                    instance_seed=int(instance_seed),
                    font_params=params,
                )
                break
            except Exception as exc:  # pragma: no cover
                last_error = exc
                sample = None
                scene = None
        if scene is None or sample is None:
            raise RuntimeError(f"could not generate {TASK_ID}: {last_error}") from last_error

        serialized_scene, _book_bboxes, _section_bboxes = serialize_library_scene(scene)
        counted_book_ids = tuple(
            str(book.book_id)
            for book in scene.books
            if str(book.section_key) == str(sample.section_key)
        )
        if len(counted_book_ids) != int(sample.target_count):
            raise RuntimeError("rendered library book count did not match sample target")
        annotation_value = sort_library_points(book_bbox_map(scene), counted_book_ids)

        prompt_defaults = required_group_defaults(
            _PROMPT_DEFAULTS,
            [
                "bundle_id",
                "scene_key",
                "task_key",
                "json_output_contract",
                "json_output_contract_answer_only",
                "answer_hint_books_in_section",
                "annotation_hint_books_in_section",
                "json_example_books_in_section",
                "json_example_answer_only_books_in_section",
            ],
            context=f"prompt defaults for {TASK_ID}",
        )
        prompt_artifacts = build_library_prompt_artifacts(
            domain=self.domain,
            scene_id=SCENE_ID,
            prompt_defaults=prompt_defaults,
            prompt_query_key=PROMPT_QUERY_KEY,
            slots=_prompt_slots(sample, prompt_defaults),
            instance_seed=int(instance_seed),
        )
        query_probabilities = {QUERY_ID: 1.0}
        trace_payload = {
            "scene_ir": {
                "domain": self.domain,
                "scene_id": SCENE_ID,
                "entities": library_scene_entities(scene),
                "relations": library_scene_relations(
                    prompt_query_key=PROMPT_QUERY_KEY,
                    section_key=str(sample.section_key),
                ),
            },
            "query_spec": {
                "task_id": self.task_id,
                "query_id": QUERY_ID,
                "prompt_query_key": PROMPT_QUERY_KEY,
                "prompt_variant_active_key": prompt_artifacts.prompt_variant_active_key,
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
                "params": {
                    "query_id": QUERY_ID,
                    "prompt_query_key": PROMPT_QUERY_KEY,
                    "section_key": str(sample.section_key),
                    "section_name": str(sample.section_name),
                    "section_count": int(sample.section_count),
                    "target_count": int(sample.target_count),
                    "section_keys": list(sample.section_keys),
                    "query_id_probabilities": dict(query_probabilities),
                    "section_key_probabilities": dict(sample.section_key_probabilities),
                    "section_count_probabilities": dict(sample.section_count_probabilities),
                    "target_count_probabilities": dict(sample.target_count_probabilities),
                },
            },
            "render_spec": library_render_spec(scene, scene_id=SCENE_ID),
            "render_map": library_base_render_map(scene, counted_book_ids=counted_book_ids),
            "execution_trace": {
                "query_id": QUERY_ID,
                "prompt_query_key": PROMPT_QUERY_KEY,
                "scene_id": SCENE_ID,
                "target_section_key": str(sample.section_key),
                "target_section_name": str(sample.section_name),
                "target_count": int(sample.target_count),
                "section_count": int(sample.section_count),
                "counted_book_ids": list(counted_book_ids),
                "sections": serialized_scene[0]["sections"],
                "books": serialized_scene[0]["books"],
                "decor": serialized_scene[0]["decor"],
            },
            "witness_symbolic": {
                "counted_book_ids": list(counted_book_ids),
                "target_section_key": str(sample.section_key),
                "answer": int(sample.target_count),
            },
            "projected_annotation": {
                "type": "point_set",
                "point_set": list(annotation_value),
                "pixel_point_set": list(annotation_value),
            },
        }
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants={str(key): str(value) for key, value in prompt_artifacts.prompt_variants.items()},
            answer_gt=TypedValue(type="integer", value=int(sample.target_count)),
            annotation_gt=TypedValue(type="point_set", value=list(annotation_value)),
            image=scene.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=QUERY_ID,
        )


__all__ = ["IllustrationsLibraryBooksInSectionCountTask", "_sample_spec"]
