"""Select the Braille option matching a reference cell."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ....core.sampling import uniform_choice
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from ..shared.common import get_int_range as _get_range

from ._lifecycle import build_braille_task_output
from .shared.annotations import role_bbox_map
from .shared.output import draw_braille_scene_artifacts
from .shared.prompts import render_braille_prompt
from .shared.rendering import render_braille_match_scene
from .shared.rules import pattern_from_mask, pattern_key, sample_pattern_with_count
from .shared.sampling import build_with_retries, resolve_braille_scene_variant
from .shared.state import OPTION_LABELS, BrailleCellSpec


DOMAIN = "symbolic"
SCENE_ID = "braille_cell"
TASK_ID = "task_symbolic__braille_cell__matching_pattern_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
INTERNAL_QUERY_KEY = "matching_pattern_label"
TASK_PROMPT_KEY = "braille_matching_pattern_label"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
    task_id=TASK_ID,
)


@dataclass(frozen=True)
class _Dataset:
    scene_variant: str
    answer_value: str
    target_answer_support: tuple[str, ...]
    annotation_item_ids: tuple[str, str]
    reference: BrailleCellSpec
    options: tuple[BrailleCellSpec, ...]
    metadata: dict[str, Any]
    scene_variant_probabilities: dict[str, float]


def _build_dataset(
    *,
    instance_seed: int,
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
    params: Mapping[str, Any],
) -> _Dataset:
    """Sample the reference cell and exactly one matching visual option."""

    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.dataset")
    option_count = int(params.get("option_count", _GEN_DEFAULTS.get("option_count", 6)))
    if int(option_count) != 6:
        raise ValueError("Braille matching task requires exactly six visual options")

    raised_min, raised_max = _get_range(
        params,
        _GEN_DEFAULTS,
        min_key="reference_raised_dot_count_min",
        max_key="reference_raised_dot_count_max",
        fallback_min=1,
        fallback_max=6,
    )
    if int(raised_min) < 1 or int(raised_max) > 6:
        raise ValueError("Braille reference raised-dot count support must stay within 1..6")
    reference_count = int(params.get("reference_raised_dot_count", rng.randint(int(raised_min), int(raised_max))))
    reference_pattern = sample_pattern_with_count(rng, int(reference_count))
    labels = tuple(str(label) for label in OPTION_LABELS)
    correct_label = str(params.get("correct_label", uniform_choice(rng, labels, sort_keys=False)))
    if correct_label not in labels:
        raise ValueError(f"correct_label must be one of {labels}")

    distractor_masks = [mask for mask in range(1, 64) if pattern_from_mask(mask) != reference_pattern]
    rng.shuffle(distractor_masks)
    distractor_patterns = [pattern_from_mask(mask) for mask in distractor_masks[:5]]
    options: list[BrailleCellSpec] = []
    distractor_index = 0
    for label in labels:
        if str(label) == str(correct_label):
            pattern = reference_pattern
            role = "correct_option"
        else:
            pattern = distractor_patterns[int(distractor_index)]
            distractor_index += 1
            role = "distractor_option"
        options.append(
            BrailleCellSpec(
                item_id=f"option_{label}",
                raised_positions=tuple(pattern),
                label=str(label),
                role=str(role),
                marked=False,
            )
        )

    reference = BrailleCellSpec(
        item_id="reference_cell",
        raised_positions=tuple(reference_pattern),
        label="REF",
        role="reference_cell",
        marked=True,
    )
    correct_option = next(option for option in options if str(option.label) == str(correct_label))
    return _Dataset(
        scene_variant=str(scene_variant),
        answer_value=str(correct_label),
        target_answer_support=labels,
        annotation_item_ids=(str(reference.item_id), str(correct_option.item_id)),
        reference=reference,
        options=tuple(options),
        metadata={
            "reference_pattern": pattern_key(reference_pattern),
            "reference_raised_positions": [int(pos) for pos in reference_pattern],
            "correct_option_label": str(correct_label),
            "correct_option_id": str(correct_option.item_id),
            "option_patterns": {str(option.label): pattern_key(option.raised_positions) for option in options},
        },
        scene_variant_probabilities={str(key): float(value) for key, value in scene_variant_probabilities.items()},
    )


@register_task
class SymbolicBrailleMatchingPatternLabelTask:
    """Choose the visual option with the same raised-dot pattern as the reference."""

    task_id = TASK_ID
    reasoning_operations = ('matching',)
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one Braille reference-to-option matching instance."""

        scene_variant, scene_variant_probabilities = resolve_braille_scene_variant(
            params,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}.scene_variant",
        )
        dataset = build_with_retries(
            lambda retry_seed: _build_dataset(
                instance_seed=int(retry_seed),
                scene_variant=str(scene_variant),
                scene_variant_probabilities=scene_variant_probabilities,
                params=params,
            ),
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            failure_message=f"failed to generate Braille matching instance for {TASK_ID}",
        )

        render_artifacts = draw_braille_scene_artifacts(
            instance_seed=int(instance_seed),
            params=params,
            render_defaults=_RENDER_DEFAULTS,
            namespace=f"{TASK_ID}.background",
            render_scene=render_braille_match_scene,
            render_kwargs={"reference": dataset.reference, "options": dataset.options},
            annotation_source="item_bboxes_px",
        )
        prompt_runtime = render_braille_prompt(
            _PROMPT_DEFAULTS,
            domain=DOMAIN,
            scene_id=SCENE_ID,
            scene_variant=str(dataset.scene_variant),
            task_key=TASK_PROMPT_KEY,
            object_description_key=f"object_description_matching_pattern_label_{dataset.scene_variant}",
            annotation_hint_key="annotation_hint_matching_pattern_label",
            answer_hint_key="answer_hint_matching_pattern_label",
            json_example_key="json_example_matching_pattern_label",
            json_example_answer_only_key="json_example_answer_only_matching_pattern_label",
            instance_seed=int(instance_seed),
            context=f"prompt defaults for {TASK_ID}",
        )
        projection = render_artifacts.projection

        annotation_artifacts = role_bbox_map(
            projection.item_bboxes,
            {
                "reference_cell": str(dataset.annotation_item_ids[0]),
                "selected_option": str(dataset.annotation_item_ids[1]),
            },
        )
        answer_gt = TypedValue(type="string", value=str(dataset.answer_value))
        return build_braille_task_output(
            prompt_runtime=prompt_runtime,
            render_artifacts=render_artifacts,
            scene_id=SCENE_ID,
            scene_variant=str(dataset.scene_variant),
            internal_query_key=INTERNAL_QUERY_KEY,
            scene_variant_probabilities=dataset.scene_variant_probabilities,
            target_answer_support=dataset.target_answer_support,
            annotation_artifacts=annotation_artifacts,
            answer_gt=answer_gt,
            annotation_item_ids=dataset.annotation_item_ids,
            annotation_dot_ids=(),
            braille_metadata=dataset.metadata,
        )


__all__ = [
    "INTERNAL_QUERY_KEY",
    "OPTION_LABELS",
    "SCENE_ID",
    "TASK_ID",
    "SymbolicBrailleMatchingPatternLabelTask",
]
