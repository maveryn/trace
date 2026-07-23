"""Select which Battleship fleet shape is not yet sunk."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.games.battleship.shared import annotations as battleship_annotations
from trace_tasks.tasks.games.battleship.shared import output as battleship_output
from trace_tasks.tasks.games.battleship.shared import rendering as battleship_rendering
from trace_tasks.tasks.games.battleship.shared import rules as battleship_rules
from trace_tasks.tasks.games.battleship.shared import sampling as battleship_sampling
from trace_tasks.tasks.games.battleship.shared import state as battleship_state
from trace_tasks.tasks.games.battleship.shared.prompts import build_battleship_prompt_artifacts
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support


TASK_ID = "task_games__battleship__remaining_ship_shape_label"
REMAINING_SHIP_SHAPE_LABEL_QUERY_IDS: Tuple[str, ...] = ("remaining_ship_shape_label",)
REMAINING_SHIP_SHAPE_LABEL_INDEX_SUPPORT: Tuple[int, ...] = (0, 1, 2, 3, 4)
_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    battleship_state.SCENE_ID,
    task_id=TASK_ID,
)


def _label_index_choice(instance_seed: int, params: Mapping[str, Any]) -> tuple[int, Tuple[int, ...], Dict[str, float]]:
    selected, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="remaining_ship_shape_label_index_support",
        explicit_key="target_answer",
        fallback_support=REMAINING_SHIP_SHAPE_LABEL_INDEX_SUPPORT,
        namespace="games.battleship.remaining_shape.option_label",
        balanced_flag_key="balanced_target_answer_sampling",
        namespace_support_permutation=True,
    )
    support = resolve_integer_support(
        params,
        gen_defaults=_GEN_DEFAULTS,
        key="remaining_ship_shape_label_index_support",
        fallback=REMAINING_SHIP_SHAPE_LABEL_INDEX_SUPPORT,
    )
    return int(selected), tuple(int(value) for value in support), dict(probabilities)


def _one_untouched_sample(
    *,
    rng: Any,
    board_size: int,
    scene_variant: str,
    target_shape_id: str,
    answer_slot: int,
    params: Mapping[str, Any],
) -> tuple[battleship_state.BattleshipSample, str]:
    """Construct a fleet where exactly one target ship remains untouched.

    The invariant is that every non-target ship is fully hit, the target ship
    has no hits, and exactly one answer option names that target shape.
    """

    base = battleship_sampling.place_fleet(rng=rng, board_size=int(board_size))
    hit_plan = {
        str(ship.ship_id): (tuple() if str(ship.ship_id) == str(target_shape_id) else tuple(ship.coords))
        for ship in base
    }
    placements = battleship_sampling.build_ship_placements_with_hits(base, hit_coords_by_ship_id=hit_plan)
    ship_cells = {coord for ship in placements for coord in ship.coords}
    options = battleship_sampling.sample_shape_options(
        rng=rng,
        answer_shape_id=str(target_shape_id),
        answer_label_index=int(answer_slot),
        option_count=len(battleship_state.SHAPE_OPTION_LABELS),
    )
    sample = battleship_sampling.build_battleship_scene_state(
        board_size=int(board_size),
        scene_variant=str(scene_variant),
        placements=placements,
        miss_coords=battleship_sampling.sample_miss_coords(
            rng=rng,
            board_size=int(board_size),
            occupied_ship_coords=ship_cells,
            excluded_coords=tuple(),
            params=params,
            gen_defaults=_GEN_DEFAULTS,
        ),
        construction_mode="placed_fleet_with_one_untouched_remaining_shape",
        shape_options=options,
    )
    if int(sample.sunk_ship_count) != len(sample.ship_placements) - 1 or int(sample.partial_ship_count) != 0 or int(sample.untouched_ship_count) != 1:
        raise ValueError("Battleship remaining-shape sample must have four sunk ships and one untouched ship")
    answer_options = [option for option in sample.shape_options if bool(option.is_answer)]
    if len(answer_options) != 1 or str(answer_options[0].shape_id) != str(target_shape_id):
        raise ValueError("Battleship remaining-shape options must expose exactly one matching answer")
    untouched = [ship for ship in sample.ship_placements if not bool(ship.hit_coords)]
    if len(untouched) != 1 or str(untouched[0].ship_id) != str(target_shape_id):
        raise ValueError("Battleship remaining-shape target must be the only untouched ship")
    return sample, str(answer_options[0].label)


@register_task
class GamesBattleshipRemainingShipShapeLabelTask:
    """Select the answer-choice fleet shape for the only untouched ship."""

    task_id = TASK_ID
    reasoning_operations = ('matching',)
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = REMAINING_SHIP_SHAPE_LABEL_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Sample target shape/options, render the board, and bind option annotation.

        This task owns the untouched-ship construction and only accepts samples
        where the board state and shape-choice answer agree exactly.
        """

        branch, branch_probs, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=self.supported_query_ids,
            default_query_id=self.supported_query_ids[0],
            task_id=self.task_id,
            namespace=f"{self.task_id}.query",
        )
        axes = battleship_sampling.resolve_battleship_scene_axes(int(instance_seed), params=task_params, gen_defaults=_GEN_DEFAULTS)
        target_shape_id, target_shape_probs = battleship_sampling.resolve_battleship_target_ship_id(
            instance_seed=int(instance_seed),
            params=task_params,
            gen_defaults=_GEN_DEFAULTS,
            namespace="games.battleship.remaining_shape.target",
            supported_shape_ids=battleship_state.SUPPORTED_BATTLESHIP_TARGET_SHIP_IDS,
        )
        answer_slot, answer_support, answer_probs = _label_index_choice(int(instance_seed), task_params)
        sample = answer = None
        for attempt in range(max(1, int(max_attempts))):
            rng = spawn_rng(int(instance_seed), f"{self.task_id}.attempt.{int(attempt)}")
            try:
                sample, answer = _one_untouched_sample(
                    rng=rng,
                    board_size=int(axes.board_size),
                    scene_variant=str(axes.scene_variant),
                    target_shape_id=str(target_shape_id),
                    answer_slot=int(answer_slot),
                    params=task_params,
                )
            except ValueError:
                continue
            break
        if sample is None or answer is None:
            raise RuntimeError(f"{self.task_id} failed to generate a valid Battleship remaining-shape scene after {max_attempts} attempts")

        shape = battleship_rules.fleet_shape_by_id()[str(target_shape_id)]
        rendered = battleship_rendering.render_battleship_sample(
            sample=sample,
            style_variant=str(axes.style_variant),
            params=task_params,
            instance_seed=int(instance_seed),
        )
        projection = battleship_annotations.project_shape_option_annotation(
            option_label=str(answer),
            rendered_scene=rendered.rendered_scene,
        )
        answer_gt = TypedValue(type="string", value=str(answer))
        annotation_gt = TypedValue(type="bbox", value=list(projection.annotation_bboxes[0]))
        prompt_defaults, prompt_artifacts = build_battleship_prompt_artifacts(
            domain=self.domain,
            instance_seed=int(instance_seed),
            prompt_query_key=str(branch),
            dynamic_slots={},
        )
        trace = battleship_output.common_trace_sections(
            axes=axes,
            sample=sample,
            rendered_context=rendered,
            annotation_projection=projection,
        )
        shared_params = {
            "scene_variant": str(axes.scene_variant),
            "query_id": str(branch),
            "style_variant": str(axes.style_variant),
            "board_size": int(sample.board_size),
            "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
            "query_id_probabilities": dict(branch_probs),
            "style_variant_probabilities": dict(axes.style_variant_probabilities),
            "board_size_probabilities": dict(axes.board_size_probabilities),
            "target_answer": int(answer_slot),
            "target_answer_support": [int(value) for value in answer_support],
            "target_answer_probabilities": dict(answer_probs),
            "target_ship_id": str(target_shape_id),
            "target_ship_display_name": str(shape.display_name),
            "target_ship_shape_id": str(target_shape_id),
            "target_ship_id_probabilities": dict(target_shape_probs),
            "target_cell_status": "untouched",
            "target_missing_coord": None,
            "candidate_labels": [],
            "shape_option_count": len(sample.shape_options),
            "shape_option_labels": [str(option.label) for option in sample.shape_options],
            "shape_option_shape_ids": [str(option.shape_id) for option in sample.shape_options],
            "hit_count": len(sample.hit_coords),
            "miss_count": len(sample.miss_coords),
            "sunk_ship_count": int(sample.sunk_ship_count),
            "partial_ship_count": int(sample.partial_ship_count),
            "untouched_ship_count": int(sample.untouched_ship_count),
        }
        trace["scene_ir"]["relations"].update(shared_params)
        trace["query_spec"] = build_prompt_query_spec(prompt_artifacts=prompt_artifacts, query_id=str(branch), params=shared_params)
        trace["execution_trace"].update(
            {
                **shared_params,
                "annotation_coords": [],
                "annotation_ship_ids": [str(target_shape_id)],
                "target_ship_cell_ids": battleship_output.target_ship_cell_ids(sample, target_ship_id=str(target_shape_id)),
            }
        )
        trace["witness_symbolic"] = battleship_output.witness_symbolic_payload(annotation_gt=annotation_gt, annotation_projection=projection)
        trace["projected_annotation"] = battleship_output.projected_annotation_payload(annotation_gt=annotation_gt, annotation_projection=projection)
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=rendered.image,
            image_id="img0",
            trace_payload=trace,
            task_versions=default_task_versions(),
            scene_id=battleship_state.SCENE_ID,
            query_id=str(branch),
        )


__all__ = [
    "GamesBattleshipRemainingShipShapeLabelTask",
    "REMAINING_SHIP_SHAPE_LABEL_INDEX_SUPPORT",
    "REMAINING_SHIP_SHAPE_LABEL_QUERY_IDS",
]
