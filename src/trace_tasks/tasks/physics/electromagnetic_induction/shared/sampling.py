"""Sampling primitives for electromagnetic induction panels."""

from __future__ import annotations

from collections.abc import Mapping
from random import Random
from typing import Any

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default

from .formulas import induced_current_class
from .layout import panel_layout
from .state import (
    ANSWER_SUPPORT,
    CURRENT_CLASSES,
    FIELD_ORIENTATIONS,
    PANEL_COUNT,
    PANEL_MECHANISMS_BY_FLUX_CHANGE,
    InductionScenario,
    PanelSpec,
)


def probability_map(values: tuple[int, ...]) -> dict[str, float]:
    """Return a uniform probability map over integer support values."""

    if not values:
        return {}
    probability = 1.0 / float(len(values))
    return {str(value): float(probability) for value in values}


def selected_probability_map(selected: int) -> dict[str, float]:
    """Return a degenerate probability map for a pinned support value."""

    return {str(int(selected)): 1.0}


def resolve_target_answer(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> tuple[int, dict[str, float]]:
    """Resolve the requested count while preserving the full 0..6 support."""

    raw_support = params.get("target_answer_support", group_default(defaults, "target_answer_support", ANSWER_SUPPORT))
    support = tuple(int(value) for value in raw_support)
    if not support:
        raise ValueError("target_answer_support cannot be empty")
    for value in support:
        if int(value) < 0 or int(value) > PANEL_COUNT:
            raise ValueError(f"target_answer_support values must be in 0..{PANEL_COUNT}")
    explicit = params.get("target_answer")
    if explicit is not None:
        target_answer = int(explicit)
        if target_answer not in set(support):
            raise ValueError(f"target_answer must be in configured support {sorted(support)}")
        return int(target_answer), selected_probability_map(target_answer)
    rng = spawn_rng(int(instance_seed), f"{namespace}.target_answer")
    return int(rng.choice(support)), probability_map(support)


def make_panel_spec(
    *,
    panel_id: str,
    current_class: str,
    rng: Random,
    bbox_px: list[float],
) -> PanelSpec:
    """Construct one panel whose visible flux cue yields the requested current class."""

    if current_class == "no_current":
        field_orientation = str(rng.choice(FIELD_ORIENTATIONS))
        flux_change = "none"
    else:
        field_orientation = str(rng.choice(FIELD_ORIENTATIONS))
        desired_induced_field = "into_page" if current_class == "clockwise" else "out_of_page"
        flux_change = "decreasing" if desired_induced_field == field_orientation else "increasing"
    mechanism = str(rng.choice(PANEL_MECHANISMS_BY_FLUX_CHANGE[str(flux_change)]))
    region_side = str(rng.choice(("left", "right")))
    resolved = induced_current_class(str(field_orientation), str(flux_change))
    if str(resolved) != str(current_class):
        raise RuntimeError(f"internal induction mapping error: expected {current_class}, got {resolved}")
    return PanelSpec(
        panel_id=str(panel_id),
        current_class=str(current_class),
        field_orientation=str(field_orientation),
        flux_change=str(flux_change),
        mechanism=str(mechanism),
        region_side=str(region_side),
        bbox_px=list(bbox_px),
    )


def make_induction_scenario(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    target_current_class: str,
    namespace: str,
) -> InductionScenario:
    """Sample six panels with exactly the requested number of target-current panels."""

    target_answer, answer_probs = resolve_target_answer(
        instance_seed=int(instance_seed),
        params=params,
        defaults=generation_defaults,
        namespace=str(namespace),
    )
    target_class = str(target_current_class)
    distractor_classes = [value for value in CURRENT_CLASSES if str(value) != target_class]
    rng = spawn_rng(int(instance_seed), f"{namespace}.panel_classes")
    panel_classes = [target_class for _ in range(int(target_answer))]
    panel_classes.extend(str(rng.choice(distractor_classes)) for _ in range(PANEL_COUNT - int(target_answer)))
    rng.shuffle(panel_classes)

    panel_bboxes = panel_layout(
        canvas_width=int(params.get("canvas_width", group_default(rendering_defaults, "canvas_width", 1180))),
        canvas_height=int(params.get("canvas_height", group_default(rendering_defaults, "canvas_height", 820))),
    )
    panels: list[PanelSpec] = []
    for index, current_class in enumerate(panel_classes):
        panel_rng = spawn_rng(int(instance_seed), f"{namespace}.panel_spec", index)
        panels.append(
            make_panel_spec(
                panel_id=f"panel_{index + 1}",
                current_class=str(current_class),
                rng=panel_rng,
                bbox_px=list(panel_bboxes[index]),
            )
        )
    return InductionScenario(
        target_current_class=str(target_class),
        target_answer=int(target_answer),
        panels=tuple(panels),
        target_answer_probabilities=dict(answer_probs),
    )
