"""Count all repeated elements on a projected fixture surface."""

from __future__ import annotations

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import ResolvedSurfaceFixtureAxes, SurfaceFixturePlan, run_surface_fixture_lifecycle
from .shared.metrics import build_repeated_surface_data
from .shared.state import SCENE_ID, SUPPORTED_SCENE_VARIANTS


TASK_ID = "task_three_d__surface_fixture__repeated_element_count"
QUERY_ID = "single"
PROMPT_QUERY_KEY = "element_type_count"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "three_d",
    SCENE_ID,
    task_id=TASK_ID,
)


def _prepare_repeated_objective(
    instance_seed,
    params,
    axes: ResolvedSurfaceFixtureAxes,
    _branch_probabilities,
    _selected_branch,
):
    """Bind the all-visible-elements count objective."""

    dataset, answer_probabilities = build_repeated_surface_data(
        namespace=f"{TASK_ID}.objective",
        scene_variant=axes.scene_variant,
        element_type=axes.element_type,
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
    )
    return SurfaceFixturePlan(
        dataset=dataset,
        answer_gt=TypedValue(type="integer", value=int(dataset["answer_value"])),
        target_element_ids=tuple(str(element_id) for element_id in dataset["target_element_ids"]),
        answer_value_probabilities=dict(answer_probabilities),
        object_description=f"a {dataset['fixture_display_name']} with visible {dataset['target_element_plural']}",
        objective_params={
            "target_count": int(dataset["answer_value"]),
            "layout_family": str(dataset["layout_family"]),
            "layout_style": str(dataset["layout_style"]),
            "visual_color_names": list(dataset.get("visual_color_names", [])),
            "visual_color_counts": dict(dataset.get("visual_color_counts", {})),
            "color_role": str(dataset.get("color_role", "non_semantic_visual_variation")),
        },
    )


@register_task
class ThreeDSurfaceFixtureRepeatedElementCountTask:
    """Count all repeated elements on a projected fixture surface."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = "three_d"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params, max_attempts):
        return run_surface_fixture_lifecycle(
            public_name=TASK_ID,
            domain_name=self.domain,
            prompt_query_key=PROMPT_QUERY_KEY,
            supported_branches=SUPPORTED_QUERY_IDS,
            default_branch=QUERY_ID,
            supported_scenes=SUPPORTED_SCENE_VARIANTS,
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_repeated_objective,
        )


__all__ = ["SUPPORTED_QUERY_IDS", "TASK_ID", "ThreeDSurfaceFixtureRepeatedElementCountTask"]
