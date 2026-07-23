"""Prompt slot helpers for size-encoded chart tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .state import DOMAIN, SCENE_ID


BUNDLE_ID = "charts_size_encoding_v1"
SCENE_KEY = "size_encoded_chart_scene"
TASK_KEY = "size_encoding_label_query"

OBJECT_DESCRIPTION_BY_VARIANT: dict[str, str] = {
    "rect_word_cloud": "a rectangular word cloud where item text size indicates value and a colored marker beside each word indicates category",
    "circle_word_cloud": "a circular word cloud where item text size indicates value and a colored marker beside each word indicates category",
    "packed_bubble_cloud": "a packed bubble chart where bubble size indicates value and color indicates category",
    "small_multiple_bubble_cloud": "multiple packed bubble-chart panels where bubble size indicates value and color indicates category",
}

ANNOTATION_HINT_BY_KIND: dict[str, str] = {
    "answer_item_bbox": 'set "annotation" to one bounding box [x0, y0, x1, y1] around the answer item',
    "reference_answer_bbox_map": 'set "annotation" to an object mapping "reference_item" and "answer_item" to their [x0, y0, x1, y1] pixel boxes',
    "reference_counted_bbox_set_map": 'set "annotation" to an object mapping "reference_item" to an array containing one [x0, y0, x1, y1] box for the reference item and "counted_items" to an array of [x0, y0, x1, y1] boxes for the counted items',
    "answer_category_bbox_set": 'set "annotation" to an array of bounding boxes [x0, y0, x1, y1] for the items in the answer category',
}

JSON_EXAMPLE_BY_KIND: dict[str, str] = {
    "answer_item_bbox": '{"annotation":[410,235,548,292],"answer":"Aero"}',
    "reference_answer_bbox_map": '{"annotation":{"reference_item":[238,460,322,544],"answer_item":[628,306,710,388]},"answer":"Cair"}',
    "reference_counted_bbox_set_map": '{"annotation":{"reference_item":[[238,460,322,544]],"counted_items":[[628,306,710,388],[720,250,780,310]]},"answer":2}',
    "answer_category_bbox_set": '{"annotation":[[118,250,236,318],[330,410,470,485],[710,205,842,280]],"answer":"Coastal"}',
}


def object_description(scene_variant: str) -> str:
    return str(OBJECT_DESCRIPTION_BY_VARIANT.get(str(scene_variant), "a size-encoded category chart"))


def render_prompt_artifacts(
    *,
    prompt_key: str,
    dynamic_slot_values: Mapping[str, Any],
    instance_seed: int,
) -> PromptTraceArtifacts:
    rendered = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=BUNDLE_ID,
        scene_key=SCENE_KEY,
        task_key=TASK_KEY,
        query_key=str(prompt_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=dict(dynamic_slot_values),
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(rendered)
