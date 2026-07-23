from __future__ import annotations

from typing import Any, Dict

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ...registry import register_task
from ...shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import RadialChoiceTaskBinding, run_bound_radial_choice_instance
from .shared.sampling import build_output_code_match_choice
from .shared.state import OPTION_LABELS


TASK_ID = "task_symbolic__radial_code_wheel__output_code_match_label"
INTERNAL_QUERY_KEY = "output_code_match_label"
TASK_PROMPT_KEY = "output_code_match_label"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "symbolic",
    "radial_code_wheel",
    task_id=TASK_ID,
)
_BINDING = RadialChoiceTaskBinding(
    seed_namespace=TASK_ID,
    internal_query_key=INTERNAL_QUERY_KEY,
    task_prompt_key=TASK_PROMPT_KEY,
    object_description_prefix="object_description_output_code_match_label",
    annotation_role_names=("inner_ring_symbol", "middle_ring_symbol", "outer_ring_symbol"),
    annotation_hint_key="annotation_hint_output_code_match_label",
    answer_hint_key="answer_hint_output_code_match_label",
    json_example_key="json_example_output_code_match_label",
    json_example_answer_only_key="json_example_answer_only_output_code_match_label",
    failure_message=f"failed to generate radial output-code instance for {TASK_ID}",
)


@register_task
class SymbolicRadialOutputCodeMatchLabelTask:
    task_id = TASK_ID
    reasoning_operations = ('transformation', 'matching')
    domain = "symbolic"
    supported_query_ids = (SINGLE_QUERY_ID,)
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int):
        # This task binds a target output label to its matching code option.
        factory = lambda retry_seed, variant, probs: build_output_code_match_choice(
            rng=spawn_rng(int(retry_seed), f"{TASK_ID}.dataset"),
            scene_variant=str(variant),
            scene_variant_probabilities=probs,
            params=params,
            gen_defaults=_GEN_DEFAULTS,
            labels=OPTION_LABELS,
        )
        return run_bound_radial_choice_instance(
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            prompt_defaults=_PROMPT_DEFAULTS,
            dataset_factory=factory,
            binding=_BINDING,
        )
