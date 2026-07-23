from __future__ import annotations

from typing import Any, Mapping

from ....core.query_ids import SINGLE_QUERY_ID
from ...registry import register_task
from ...shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import (
    DOMAIN,
    SCENE_ID,
    IconGridCategoryObjective,
    IconGridCategoryTaskLifecycle,
    build_category_binding_payload,
    run_icon_grid_category_lifecycle,
)
from .shared.annotations import representative_cell_bboxes_by_field
from .shared.defaults import IconGridDefaults
from .shared.sampling import resolve_distinct_type_frequency_spec


TASK_ID = "task_icons__icon_grid__distinct_type_count"
PROMPT_QUERY_KEY = "distinct_type_count"
QUERY_ID = SINGLE_QUERY_ID
SUPPORTED_QUERY_IDS = (QUERY_ID,)

_DEFAULTS = IconGridDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
    task_id=TASK_ID,
)


def _resolve_frequency_spec(instance_seed: int, params: Mapping[str, Any]):
    return resolve_distinct_type_frequency_spec(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        defaults=_DEFAULTS,
        selection_namespace=f"{TASK_ID}:frequency_spec",
    )


def _bind_answer_annotation(scene_payload, frequency_spec):
    annotation_bboxes, annotation_indices, counted_icon_ids = representative_cell_bboxes_by_field(
        scene_payload,
        field_name="icon_id",
    )
    answer_value = int(scene_payload.distinct_type_count)
    if len(annotation_bboxes) != int(answer_value):
        raise ValueError("distinct-type representative cell count did not match answer")
    return build_category_binding_payload(
        scene_payload=scene_payload,
        frequency_spec=frequency_spec,
        counting_rule="distinct_icon_type_count",
        question_format="count_distinct_icon_types_in_grid",
        counted_icon_ids=tuple(str(icon_id) for icon_id in counted_icon_ids),
        annotation_indices=tuple(int(index) for index in annotation_indices),
        annotation_bboxes=tuple(annotation_bboxes),
        target_count=int(answer_value),
    )


@register_task
class IconsIconGridDistinctTypeCountTask(IconGridCategoryTaskLifecycle):
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True
    prompt_query_key = PROMPT_QUERY_KEY
    scene_kind = "icons_grid_distinct_type_counting"

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        return run_icon_grid_category_lifecycle(
            self,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            objective=IconGridCategoryObjective(
                params=params,
                generation_defaults=_GEN_DEFAULTS,
                render_defaults=_RENDER_DEFAULTS,
                prompt_defaults=_PROMPT_DEFAULTS,
                defaults=_DEFAULTS,
                resolve_frequency_spec=_resolve_frequency_spec,
                bind_answer_annotation=_bind_answer_annotation,
            ),
        )
