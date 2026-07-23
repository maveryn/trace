"""Sampling helpers for navigation-flow page scenes."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.deterministic_sampling import resolve_selection_index
from trace_tasks.tasks.shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant

from .defaults import DEFAULTS, GENERATION_DEFAULTS, NAMESPACE_ROOT, SCENE_VARIANTS
from .state import (
    MENU_SURFACE,
    RIBBON_SURFACE,
    SIDEBAR_SURFACE,
    ControlSpec,
    NavigationFlowCase,
)


NAVIGATION_SURFACES: Tuple[str, ...] = (MENU_SURFACE, SIDEBAR_SURFACE, RIBBON_SURFACE)
_NAV_COMMAND_SYMBOLS: Tuple[str, ...] = ("@", "%", "&", "#")
_SIDEBAR_ITEM_SYMBOLS: Tuple[str, ...] = ("@", "%", "&", "#")


def _selection_index(params: Mapping[str, Any], *, instance_seed: int, namespace: str) -> int:
    return int(
        resolve_selection_index(
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{NAMESPACE_ROOT}.{namespace}",
        )
    )


def _resolve_axis_bounds(
    params: Mapping[str, Any],
    *,
    exact_key: str,
    min_key: str,
    max_key: str,
    fallback_exact: int,
    fallback_min: int,
    fallback_max: int,
) -> Tuple[int, int]:
    if str(exact_key) in params or str(exact_key) in GENERATION_DEFAULTS:
        value = int(params.get(str(exact_key), group_default(GENERATION_DEFAULTS, str(exact_key), int(fallback_exact))))
        return int(value), int(value)
    min_value = int(params.get(str(min_key), group_default(GENERATION_DEFAULTS, str(min_key), int(fallback_min))))
    max_value = int(params.get(str(max_key), group_default(GENERATION_DEFAULTS, str(max_key), int(fallback_max))))
    if int(min_value) > int(max_value):
        raise ValueError(f"{min_key} must be <= {max_key}")
    return int(min_value), int(max_value)


def _resolve_menu_command_count(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    candidate_label_capacity: int,
) -> Tuple[int, Tuple[int, int]]:
    min_value, max_value = _resolve_axis_bounds(
        params,
        exact_key="menu_command_count",
        min_key="menu_command_count_min",
        max_key="menu_command_count_max",
        fallback_exact=3,
        fallback_min=3,
        fallback_max=4,
    )
    candidates = [
        int(value)
        for value in range(int(min_value), int(max_value) + 1)
        if 2 * 2 * 2 * int(value) <= int(candidate_label_capacity)
    ]
    if not candidates:
        raise ValueError("menu command count requires more candidate labels than available")
    selected_index = _selection_index(
        params,
        instance_seed=int(instance_seed),
        namespace="menu_command_count",
    ) % len(candidates)
    return int(candidates[int(selected_index)]), (int(min_value), int(max_value))


def _resolve_ribbon_counts(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    candidate_label_capacity: int,
) -> Tuple[int, Tuple[int, int], int, Tuple[int, int], int, Tuple[int, int]]:
    """Resolve ribbon dimensions while preserving candidate-label uniqueness."""

    tab_min, tab_max = _resolve_axis_bounds(
        params,
        exact_key="ribbon_tab_count",
        min_key="ribbon_tab_count_min",
        max_key="ribbon_tab_count_max",
        fallback_exact=3,
        fallback_min=3,
        fallback_max=5,
    )
    group_min, group_max = _resolve_axis_bounds(
        params,
        exact_key="ribbon_group_count",
        min_key="ribbon_group_count_min",
        max_key="ribbon_group_count_max",
        fallback_exact=2,
        fallback_min=2,
        fallback_max=3,
    )
    command_min, command_max = _resolve_axis_bounds(
        params,
        exact_key="ribbon_command_count",
        min_key="ribbon_command_count_min",
        max_key="ribbon_command_count_max",
        fallback_exact=3,
        fallback_min=3,
        fallback_max=4,
    )
    candidates = [
        (int(tab_count), int(group_count), int(command_count))
        for tab_count in range(int(tab_min), int(tab_max) + 1)
        for group_count in range(int(group_min), int(group_max) + 1)
        for command_count in range(int(command_min), int(command_max) + 1)
        if int(tab_count) * int(group_count) * int(command_count) <= int(candidate_label_capacity)
    ]
    if not candidates:
        raise ValueError("ribbon counts require more candidate labels than available")
    selected_index = _selection_index(
        params,
        instance_seed=int(instance_seed),
        namespace="ribbon_count_tuple",
    ) % len(candidates)
    tab_count, group_count, command_count = candidates[int(selected_index)]
    return (
        int(tab_count),
        (int(tab_min), int(tab_max)),
        int(group_count),
        (int(group_min), int(group_max)),
        int(command_count),
        (int(command_min), int(command_max)),
    )


def _resolve_named_axis(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    supported: Sequence[str],
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    rng = spawn_rng(int(instance_seed), f"{NAMESPACE_ROOT}.{namespace}")
    selected, probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=GENERATION_DEFAULTS,
        supported_variants=tuple(str(value) for value in supported),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
    )
    balanced = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GENERATION_DEFAULTS,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=tuple(str(value) for value in supported),
        balance_flag_key=str(balance_flag_key),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        sampling_namespace=f"{NAMESPACE_ROOT}.{namespace}",
    )
    if str(balanced) != str(selected) and params.get(str(explicit_key)) is not None:
        return str(balanced), {str(key): (1.0 if str(key) == str(balanced) else 0.0) for key in supported}
    return str(balanced), {str(key): float(value) for key, value in sorted(probabilities.items())}


def resolve_navigation_surface(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve the navigation surface axis for the merged public task."""

    return _resolve_named_axis(
        params=params,
        instance_seed=int(instance_seed),
        supported=NAVIGATION_SURFACES,
        explicit_key="navigation_surface",
        weights_key="navigation_surface_weights",
        balance_flag_key="balanced_navigation_surface_sampling",
        namespace=str(namespace),
    )


def normalize_str_support(params: Mapping[str, Any], key: str, fallback: Sequence[str]) -> Tuple[str, ...]:
    """Resolve a non-empty string support list."""

    raw_values = params.get(str(key), group_default(GENERATION_DEFAULTS, str(key), fallback))
    support: List[str] = []
    for raw_value in raw_values:
        value = str(raw_value).strip()
        if value and value not in support:
            support.append(value)
    if not support:
        raise ValueError(f"{key} must not be empty")
    return tuple(str(value) for value in support)


def _base_control_specs(
    *,
    navigation_surface: str,
    menus: Sequence[str],
    submenus: Sequence[str],
    menu_groups: Sequence[str],
    menu_commands: Sequence[str],
    sidebar_sections: Sequence[str],
    sidebar_groups: Sequence[str],
    sidebar_items: Sequence[str],
    ribbon_tabs: Sequence[str],
    ribbon_groups: Sequence[str],
    ribbon_commands: Sequence[str],
) -> Tuple[ControlSpec, ...]:
    """Construct unlabeled controls for the requested navigation surface."""

    controls: List[ControlSpec] = []
    order = 0
    if str(navigation_surface) == MENU_SURFACE:
        for menu_index, menu in enumerate(menus):
            for submenu_index, submenu in enumerate(submenus):
                for group_index, group in enumerate(menu_groups):
                    for command_index, command in enumerate(menu_commands):
                        controls.append(
                            ControlSpec(
                                control_id=(
                                    f"menu_{menu_index:02d}_{submenu_index:02d}_"
                                    f"{group_index:02d}_{command_index:02d}"
                                ),
                                candidate_label="",
                                role="menu_item",
                                display_text=str(_NAV_COMMAND_SYMBOLS[int(command_index) % len(_NAV_COMMAND_SYMBOLS)]),
                                nav_kind=MENU_SURFACE,
                                path_keys=(str(menu), str(submenu), str(group), str(command)),
                                order_index=int(order),
                            )
                        )
                        order += 1
        return tuple(controls)
    if str(navigation_surface) == SIDEBAR_SURFACE:
        for section_index, section in enumerate(sidebar_sections):
            for group_index, group in enumerate(sidebar_groups):
                for item_index, item in enumerate(sidebar_items):
                    controls.append(
                        ControlSpec(
                            control_id=f"sidebar_{section_index:02d}_{group_index:02d}_{item_index:02d}",
                            candidate_label="",
                            role="sidebar_tree_item",
                            display_text=str(_SIDEBAR_ITEM_SYMBOLS[int(item_index) % len(_SIDEBAR_ITEM_SYMBOLS)]),
                            nav_kind=SIDEBAR_SURFACE,
                            path_keys=(str(section), str(group), str(item)),
                            order_index=int(order),
                        )
                    )
                    order += 1
        return tuple(controls)
    if str(navigation_surface) != RIBBON_SURFACE:
        raise ValueError(f"unsupported navigation surface: {navigation_surface}")
    for tab_index, tab in enumerate(ribbon_tabs):
        for group_index, group in enumerate(ribbon_groups):
            for command_index, command in enumerate(ribbon_commands):
                controls.append(
                    ControlSpec(
                        control_id=f"ribbon_{tab_index:02d}_{group_index:02d}_{command_index:02d}",
                        candidate_label="",
                        role="ribbon_command",
                        display_text=str(_NAV_COMMAND_SYMBOLS[int(command_index) % len(_NAV_COMMAND_SYMBOLS)]),
                        nav_kind=RIBBON_SURFACE,
                        path_keys=(str(tab), str(group), str(command)),
                        order_index=int(order),
                    )
                )
                order += 1
    return tuple(controls)


def _with_candidate_labels(
    controls: Sequence[ControlSpec],
    *,
    target_control_id: str,
    target_label: str,
    candidate_label_pool: Sequence[str],
    instance_seed: int,
    namespace: str,
) -> Tuple[ControlSpec, ...]:
    if len(controls) > len(candidate_label_pool):
        raise ValueError("candidate_label_pool must cover all navigation controls")
    remaining_labels = [str(value) for value in candidate_label_pool if str(value) != str(target_label)]
    rng = spawn_rng(int(instance_seed), f"{NAMESPACE_ROOT}.{namespace}.candidate_labels")
    rng.shuffle(remaining_labels)
    assigned: List[ControlSpec] = []
    cursor = 0
    for control in controls:
        label = str(target_label) if str(control.control_id) == str(target_control_id) else str(remaining_labels[cursor])
        if str(control.control_id) != str(target_control_id):
            cursor += 1
        assigned.append(
            ControlSpec(
                control_id=str(control.control_id),
                candidate_label=str(label),
                role=str(control.role),
                display_text=str(control.display_text),
                nav_kind=str(control.nav_kind),
                path_keys=tuple(str(value) for value in control.path_keys),
                order_index=int(control.order_index),
            )
        )
    return tuple(assigned)


def build_navigation_flow_case(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    navigation_surface: str,
    namespace: str,
) -> NavigationFlowCase:
    """Sample one navigation-flow case for a fixed surface."""

    if str(navigation_surface) not in NAVIGATION_SURFACES:
        raise ValueError(f"unsupported navigation surface: {navigation_surface}")
    scene_variant, scene_variant_probabilities = _resolve_named_axis(
        params=params,
        instance_seed=int(instance_seed),
        supported=SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        namespace=f"{namespace}.scene_variant",
    )
    candidate_label_pool = normalize_str_support(params, "candidate_label_pool", DEFAULTS.candidate_label_pool)
    menus = normalize_str_support(params, "nav_menu_pool", DEFAULTS.nav_menu_pool)[:2]
    submenus = normalize_str_support(params, "nav_submenu_pool", DEFAULTS.nav_submenu_pool)[:2]
    menu_groups = normalize_str_support(params, "nav_menu_group_pool", DEFAULTS.nav_menu_group_pool)[:2]
    menu_command_count, menu_command_count_range = _resolve_menu_command_count(
        params,
        instance_seed=int(instance_seed),
        candidate_label_capacity=len(candidate_label_pool),
    )
    menu_commands = normalize_str_support(params, "nav_command_pool", DEFAULTS.nav_command_pool)[: int(menu_command_count)]
    sidebar_sections = normalize_str_support(params, "nav_sidebar_section_pool", DEFAULTS.nav_sidebar_section_pool)[:3]
    sidebar_groups = normalize_str_support(params, "nav_sidebar_group_pool", DEFAULTS.nav_sidebar_group_pool)[:2]
    sidebar_items = normalize_str_support(params, "nav_sidebar_item_pool", DEFAULTS.nav_sidebar_item_pool)[:2]
    (
        ribbon_tab_count,
        ribbon_tab_count_range,
        ribbon_group_count,
        ribbon_group_count_range,
        ribbon_command_count,
        ribbon_command_count_range,
    ) = _resolve_ribbon_counts(
        params,
        instance_seed=int(instance_seed),
        candidate_label_capacity=len(candidate_label_pool),
    )
    ribbon_tabs = normalize_str_support(params, "nav_ribbon_tab_pool", DEFAULTS.nav_ribbon_tab_pool)[: int(ribbon_tab_count)]
    ribbon_groups = normalize_str_support(params, "nav_ribbon_group_pool", DEFAULTS.nav_ribbon_group_pool)[: int(ribbon_group_count)]
    ribbon_commands = normalize_str_support(params, "nav_ribbon_command_pool", DEFAULTS.nav_ribbon_command_pool)[
        : int(ribbon_command_count)
    ]
    if (
        len(menus) < 2
        or len(submenus) < 2
        or len(menu_groups) < 2
        or len(menu_commands) < 2
        or len(sidebar_sections) < 3
        or len(sidebar_groups) < 2
        or len(sidebar_items) < 2
        or len(ribbon_tabs) < 3
        or len(ribbon_groups) < 2
        or len(ribbon_commands) < 2
    ):
        raise ValueError("navigation pools are too small for the active scene")
    controls_without_labels = _base_control_specs(
        navigation_surface=str(navigation_surface),
        menus=menus,
        submenus=submenus,
        menu_groups=menu_groups,
        menu_commands=menu_commands,
        sidebar_sections=sidebar_sections,
        sidebar_groups=sidebar_groups,
        sidebar_items=sidebar_items,
        ribbon_tabs=ribbon_tabs,
        ribbon_groups=ribbon_groups,
        ribbon_commands=ribbon_commands,
    )
    target_index = _selection_index(
        params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.target.{navigation_surface}",
    ) % len(controls_without_labels)
    target = controls_without_labels[int(target_index)]
    target_label = str(
        params.get(
            "target_label",
            candidate_label_pool[
                _selection_index(
                    params,
                    instance_seed=int(instance_seed),
                    namespace=f"{namespace}.answer_label.{navigation_surface}",
                )
                % len(candidate_label_pool)
            ],
        )
    )
    controls = _with_candidate_labels(
        controls_without_labels,
        target_control_id=str(target.control_id),
        target_label=str(target_label),
        candidate_label_pool=candidate_label_pool,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    path_labels = tuple(str(value) for value in target.path_keys)
    return NavigationFlowCase(
        navigation_surface=str(navigation_surface),
        scene_variant=str(scene_variant),
        controls=tuple(controls),
        target_control_id=str(target.control_id),
        target_label=str(target_label),
        path_labels=path_labels,
        path_display=" > ".join(path_labels),
        command_label=str(path_labels[-1]),
        menu_command_count=int(menu_command_count),
        menu_command_count_range=tuple(int(value) for value in menu_command_count_range),
        ribbon_tab_count=int(ribbon_tab_count),
        ribbon_tab_count_range=tuple(int(value) for value in ribbon_tab_count_range),
        ribbon_group_count=int(ribbon_group_count),
        ribbon_group_count_range=tuple(int(value) for value in ribbon_group_count_range),
        ribbon_command_count=int(ribbon_command_count),
        ribbon_command_count_range=tuple(int(value) for value in ribbon_command_count_range),
        candidate_label_pool=tuple(str(value) for value in candidate_label_pool),
        surface_probabilities={str(navigation_surface): 1.0},
        scene_variant_probabilities=dict(scene_variant_probabilities),
    )
