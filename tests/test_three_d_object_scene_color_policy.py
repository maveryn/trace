from __future__ import annotations

from typing import Any, Mapping

import pytest

from trace_tasks.tasks import create_task
from trace_tasks.tasks.shared.named_colors import available_named_colors
from trace_tasks.tasks.three_d.shared.option_panel import PROMPT_COLOR_RGB_BY_NAME


OBJECT_SCENE_OPTION_TASKS = (
    "task_three_d__object_scene__between_references_label",
    "task_three_d__object_scene__camera_distance_extremum_label",
    "task_three_d__object_scene__object_relation_label",
    "task_three_d__object_scene__occlusion_order_label",
    "task_three_d__object_scene__reference_nearest_label",
)


def test_object_scene_option_prompt_colors_use_canonical_palette() -> None:
    canonical = {
        str(name): (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        for name, rgb in available_named_colors()
    }

    assert dict(PROMPT_COLOR_RGB_BY_NAME) == canonical


def _answer_spec(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    answer_label = str(trace["answer_label"])
    for spec in trace["point_specs"]:
        if str(spec["point_label"]) == answer_label:
            return spec
    raise AssertionError(f"missing answer spec for label {answer_label}")


@pytest.mark.parametrize("task_id", OBJECT_SCENE_OPTION_TASKS)
def test_object_scene_option_answer_color_is_not_slot_locked(task_id: str) -> None:
    task = create_task(task_id)
    answer_colors = set()
    answer_labels = set()
    for seed in range(20260701, 20260709):
        output = task.generate(
            seed,
            params={"post_image_noise_apply_prob": 0.0},
            max_attempts=240,
        )
        trace = output.trace_payload["execution_trace"]
        spec = _answer_spec(trace)
        answer_labels.add(str(trace["answer_label"]))
        answer_colors.add(str(spec["option_color_name"]))
        assert spec["color_assignment_policy"] == "independent_prompt_color_by_option_label"

    assert len(answer_labels) >= 2
    assert len(answer_colors) >= 2
