"""Scene-private lifecycle for workspace target tasks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.sampling import integer_range_choice, support_probability_map, uniform_choice_with_probabilities
from ....core.seed import hash64, spawn_rng
from ....core.scene_config import get_scene_defaults, resolve_scene_section_defaults
from ....core.types import TypedValue
from ....core.visual.noise import apply_post_image_noise
from ...base import TaskOutput
from ..shared.information_style import make_pages_information_background, resolve_pages_information_style
from ...shared.config_defaults import group_default, required_group_defaults, split_generation_rendering_prompt_defaults
from ...shared.deterministic_sampling import resolve_selection_index
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_query_spec,
    build_prompt_trace_artifacts,
    render_task_prompt_variants,
)
from ...shared.render_variation import resolve_render_int
from ...shared.text_rendering import load_font
from ...shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant
from ...shared.text_legibility import contrast_ratio, draw_text_traced, normalize_rgb
from .shared.styles import (
    SUPPORTED_SCENE_VARIANTS,
    _bbox_list,
    _clamp_unit,
    _draw_app_chrome,
    _draw_badge,
    _draw_text_center_fit,
    _draw_text_left,
    _normalize_str_support as _unused_normalize,
    _rounded_rect,
    _theme_from_information_style,
)


BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]
SCENE = "workspace"
TASK_NAMESPACE = "pages.workspace"
PROMPT_BUNDLE = "pages_workspace_v1"
PROMPT_SCENE_KEY = "workspace"
PROMPT_TASK_KEY = "workspace_control_query"
PROMPT_CONTROL_LABEL_KEY = "target_control_lookup"
PROMPT_CONTEXT_COUNT_KEY = "row_state_filter_count"
PROMPT_CONTEXT_GUIDE_LABEL_KEY = "context_guide_control_lookup"
_GUIDE_CODES: Tuple[str, ...] = ("K1", "M2", "R3", "T4", "V5")
_CONTEXT_CUE_CODES: Tuple[str, ...] = ("Q1", "S2", "D3", "F4", "L5")
_ACTION_SYMBOLS: Tuple[str, ...] = ("@", "%", "&", "#", "*")
_CONTROL_STATE_NORMAL = "normal"
_COUNT_CONTROL_STATES: Tuple[str, ...] = ("blue_highlighted", "gray_disabled", "orange_warning")
_CONTROL_STATE_PHRASES: Mapping[str, str] = {
    "blue_highlighted": "blue-highlighted",
    "gray_disabled": "gray-disabled",
    "orange_warning": "orange warning",
}
_ACCENT_FILLS: Tuple[Color, ...] = (
    (225, 239, 255),
    (231, 246, 236),
    (255, 239, 216),
    (246, 230, 242),
    (228, 246, 247),
)
_ACCENT_LINES: Tuple[Color, ...] = (
    (93, 142, 205),
    (92, 158, 110),
    (208, 139, 54),
    (174, 105, 164),
    (69, 145, 157),
)


@dataclass(frozen=True)
class ProfessionalVariantSpec:
    name: str
    layout: str
    scene_title: str
    context_title: str
    guide_title: str
    header_title: str
    context_kind: str
    guide_kind: str
    header_kind: str
    control_role: str
    context_pool_key: str
    action_pool_key: str
    cue_pool_key: str
    context_pool: Tuple[str, ...]
    action_pool: Tuple[str, ...]
    cue_pool: Tuple[str, ...]
    instruction_templates: Tuple[str, ...]


@dataclass(frozen=True)
class ProfessionalTaskDefinition:
    task_namespace: str
    scene_kind: str
    question_format: str
    supported_query_ids: Tuple[str, ...]
    variants: Tuple[ProfessionalVariantSpec, ...]


@dataclass(frozen=True)
class _TaskDefaults:
    canvas_width: int = 1280
    canvas_height: int = 800
    window_margin_px: int = 42
    title_bar_height_px: int = 46
    menu_bar_height_px: int = 34
    corner_radius_px: int = 16
    control_corner_radius_px: int = 8
    control_outline_width_px: int = 2
    badge_size_px: int = 24
    title_font_size_px: int = 24
    body_font_size_px: int = 16
    small_font_size_px: int = 13
    label_font_size_px: int = 16
    context_count: int = 5
    candidate_label_pool: Tuple[str, ...] = tuple(chr(ord("A") + idx) for idx in range(26))


@dataclass(frozen=True)
class _RenderParams:
    canvas_width: int
    canvas_height: int
    window_margin_px: int
    title_bar_height_px: int
    menu_bar_height_px: int
    corner_radius_px: int
    control_corner_radius_px: int
    control_outline_width_px: int
    badge_size_px: int
    title_font_size_px: int
    body_font_size_px: int
    small_font_size_px: int
    label_font_size_px: int


@dataclass(frozen=True)
class _ControlSpec:
    control_id: str
    candidate_label: str
    role: str
    display_text: str
    context_label: str
    context_cue_label: str
    action_label: str
    cue_label: str
    code_label: str
    context_index: int
    action_index: int
    order_index: int
    state_id: str = _CONTROL_STATE_NORMAL
    state_phrase: str = "normal"


@dataclass(frozen=True)
class _ResolvedQuery:
    objective_key: str
    query_id: str
    scene_variant: str
    workspace_variant: str
    variant_spec: ProfessionalVariantSpec
    controls: Tuple[_ControlSpec, ...]
    target_control_id: str
    target_label: str
    context_label: str
    action_label: str
    cue_label: str
    context_cue_label: str
    code_label: str
    instruction_text: str
    guide_order: Tuple[int, ...]
    context_guide_order: Tuple[int, ...]
    context_count: int
    context_count_range: Tuple[int, int]
    candidate_label_pool: Tuple[str, ...]
    query_id_probabilities: Dict[str, float]
    workspace_variant_probabilities: Dict[str, float]
    scene_variant_probabilities: Dict[str, float]
    answer_value: int = 0
    target_context_index: int = -1
    target_state_id: str = ""
    target_state_phrase: str = ""
    counted_control_ids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class _RenderedScene:
    control_bboxes_by_id: Dict[str, List[float]]
    badge_bboxes_by_id: Dict[str, List[float]]
    support_bboxes_by_id: Dict[str, List[float]]
    control_records: Tuple[Dict[str, Any], ...]
    support_records: Tuple[Dict[str, Any], ...]
    scene_bbox_px: List[float]
    window_bbox_px: List[float]
    profile: Any
    theme: Any


_DEFAULTS = _TaskDefaults()


def _workspace_defaults() -> Tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
]:
    scene_id_defaults = get_scene_defaults("pages", SCENE)
    gen_defaults, render_defaults, prompt_defaults = split_generation_rendering_prompt_defaults(
        scene_id_defaults if isinstance(scene_id_defaults, Mapping) else {},
    )
    visual_defaults = scene_id_defaults.get("visual", {}) if isinstance(scene_id_defaults, Mapping) else {}
    background_defaults = dict(visual_defaults.get("background", {})) if isinstance(visual_defaults.get("background"), Mapping) else {}
    noise_defaults = dict(visual_defaults.get("noise", {})) if isinstance(visual_defaults.get("noise"), Mapping) else {}
    return gen_defaults, render_defaults, prompt_defaults, background_defaults, noise_defaults


def _normalize_support(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    *,
    key: str,
    fallback: Sequence[str],
    task_id: str,
) -> Tuple[str, ...]:
    raw_values = params.get(str(key), group_default(gen_defaults, str(key), fallback))
    support: List[str] = []
    for raw_value in raw_values:
        value = str(raw_value).strip()
        if value and value not in support:
            support.append(value)
    if not support:
        raise ValueError(f"{key} must not be empty for {task_id}")
    return tuple(str(value) for value in support)


def _decoupled_params(params: Mapping[str, Any], *, task_id: str, divisor: int, namespace: str) -> Mapping[str, Any]:
    _ = task_id, int(divisor), namespace
    return params


def _support_selection_index(
    params: Mapping[str, Any],
    *,
    task_id: str,
    supported_query_ids: Sequence[str],
    instance_seed: int,
    namespace: str,
) -> int:
    _ = supported_query_ids
    return int(resolve_selection_index(params=params, instance_seed=int(instance_seed), namespace=f"{task_id}:{namespace}"))


def _resolve_context_count(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    *,
    task_id: str,
    supported_query_ids: Sequence[str],
    instance_seed: int,
) -> Tuple[int, Tuple[int, int]]:
    if "context_count" in params or "context_count" in gen_defaults:
        value = int(params.get("context_count", group_default(gen_defaults, "context_count", _DEFAULTS.context_count)))
        min_value = max_value = int(value)
    else:
        min_value = int(params.get("context_count_min", group_default(gen_defaults, "context_count_min", _DEFAULTS.context_count)))
        max_value = int(params.get("context_count_max", group_default(gen_defaults, "context_count_max", _DEFAULTS.context_count)))
    if int(min_value) > int(max_value):
        raise ValueError(f"context_count_min must be <= context_count_max for {task_id}")
    if int(min_value) < 2 or int(max_value) > 5:
        raise ValueError(f"context_count range must stay within 2..5 for {task_id}, got {min_value}..{max_value}")
    span = int(max_value) - int(min_value) + 1
    selected = int(min_value) + (
        _support_selection_index(
            params,
            task_id=str(task_id),
            supported_query_ids=supported_query_ids,
            instance_seed=int(instance_seed),
            namespace="context_count",
        )
        % max(1, int(span))
    )
    return int(selected), (int(min_value), int(max_value))


def _resolve_named_axis(
    rng,
    *,
    task_id: str,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    supported: Tuple[str, ...],
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    selected, probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=gen_defaults,
        supported_variants=supported,
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
    )
    balanced = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=supported,
        balance_flag_key=str(balance_flag_key),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        sampling_namespace=f"{task_id}:{namespace}",
    )
    return str(balanced), {str(key): float(value) for key, value in sorted(probabilities.items())}


def _resolve_render_params(
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    *,
    instance_seed: int | None = None,
) -> _RenderParams:
    values = asdict(_DEFAULTS)

    def _int_value(key: str) -> int:
        return resolve_render_int(
            params,
            render_defaults,
            str(key),
            int(values[str(key)]),
            instance_seed=instance_seed,
            namespace=TASK_NAMESPACE,
        )

    return _RenderParams(
        canvas_width=_int_value("canvas_width"),
        canvas_height=_int_value("canvas_height"),
        window_margin_px=_int_value("window_margin_px"),
        title_bar_height_px=_int_value("title_bar_height_px"),
        menu_bar_height_px=_int_value("menu_bar_height_px"),
        corner_radius_px=_int_value("corner_radius_px"),
        control_corner_radius_px=_int_value("control_corner_radius_px"),
        control_outline_width_px=_int_value("control_outline_width_px"),
        badge_size_px=_int_value("badge_size_px"),
        title_font_size_px=_int_value("title_font_size_px"),
        body_font_size_px=_int_value("body_font_size_px"),
        small_font_size_px=_int_value("small_font_size_px"),
        label_font_size_px=_int_value("label_font_size_px"),
    )


def _state_phrase(state_id: str) -> str:
    if str(state_id) == _CONTROL_STATE_NORMAL:
        return "normal"
    return str(_CONTROL_STATE_PHRASES[str(state_id)])


def _with_state(control: _ControlSpec, *, state_id: str) -> _ControlSpec:
    return _ControlSpec(
        control_id=str(control.control_id),
        candidate_label=str(control.candidate_label),
        role=str(control.role),
        display_text=str(control.display_text),
        context_label=str(control.context_label),
        context_cue_label=str(control.context_cue_label),
        action_label=str(control.action_label),
        cue_label=str(control.cue_label),
        code_label=str(control.code_label),
        context_index=int(control.context_index),
        action_index=int(control.action_index),
        order_index=int(control.order_index),
        state_id=str(state_id),
        state_phrase=_state_phrase(str(state_id)),
    )


def _resolve_explicit_or_integer_range(
    params: Mapping[str, Any],
    *,
    rng: Any,
    explicit_keys: Sequence[str],
    minimum: int,
    maximum: int,
) -> Tuple[int, Dict[str, float]]:
    support = tuple(range(int(minimum), int(maximum) + 1))
    for key in explicit_keys:
        if str(key) in params and params.get(str(key)) is not None:
            value = int(params[str(key)])
            if int(value) not in set(support):
                raise ValueError(f"{key} must be within {minimum}..{maximum}, got {value}")
            return int(value), support_probability_map(support, selected=int(value), sort_keys=True)
    return integer_range_choice(rng, int(minimum), int(maximum))


def _resolve_explicit_or_uniform(
    params: Mapping[str, Any],
    *,
    rng: Any,
    explicit_keys: Sequence[str],
    support: Sequence[str],
) -> Tuple[str, Dict[str, float]]:
    supported = tuple(str(value) for value in support)
    for key in explicit_keys:
        if str(key) in params and params.get(str(key)) is not None:
            value = str(params[str(key)]).strip()
            if value not in set(supported):
                raise ValueError(f"unsupported {key}: {value}; supported: {supported}")
            return value, support_probability_map(supported, selected=value, sort_keys=True)
    selected, probabilities = uniform_choice_with_probabilities(rng, supported, sort_keys=True)
    return str(selected), dict(probabilities)


def _resolve_query(
    definition: ProfessionalTaskDefinition,
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    objective_key: str,
    gen_defaults: Mapping[str, Any],
) -> _ResolvedQuery:
    """Resolve one public workspace task wrapper into controls, target, and witnesses."""

    task_namespace = str(definition.task_namespace)
    rng = spawn_rng(int(instance_seed), f"{task_namespace}.query")
    objective = str(objective_key)
    if objective not in SUPPORTED_PROMPT_QUERY_KEYS:
        raise ValueError(
            f"unsupported workspace objective key: {objective}; "
            f"supported: {SUPPORTED_PROMPT_QUERY_KEYS}"
        )
    objective_probabilities = {objective: 1.0}
    scene_variant, scene_variant_probabilities = _resolve_named_axis(
        rng,
        task_id=task_namespace,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        params=_decoupled_params(params, task_id=task_namespace, divisor=len(definition.supported_query_ids), namespace="scene_variant"),
        supported=SUPPORTED_SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        namespace="scene_variant",
    )
    workspace_variant, workspace_variant_probabilities = _resolve_named_axis(
        rng,
        task_id=task_namespace,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        params=params,
        supported=SUPPORTED_WORKSPACE_VARIANTS,
        explicit_key="workspace_variant",
        weights_key="workspace_variant_weights",
        balance_flag_key="balanced_workspace_variant_sampling",
        namespace="workspace_variant",
    )
    variant_by_name = {str(spec.name): spec for spec in definition.variants}
    if str(workspace_variant) not in variant_by_name:
        raise ValueError(f"unsupported workspace variant {workspace_variant!r} for {task_namespace}")
    spec = variant_by_name[str(workspace_variant)]
    candidate_label_pool = _normalize_support(params, gen_defaults, key="candidate_label_pool", fallback=_DEFAULTS.candidate_label_pool, task_id=task_namespace)
    context_count, context_count_range = _resolve_context_count(
        params,
        gen_defaults,
        task_id=task_namespace,
        supported_query_ids=definition.supported_query_ids,
        instance_seed=int(instance_seed),
    )
    contexts = _normalize_support(params, gen_defaults, key=str(spec.context_pool_key), fallback=spec.context_pool, task_id=task_namespace)[:context_count]
    actions = _normalize_support(params, gen_defaults, key=str(spec.action_pool_key), fallback=spec.action_pool, task_id=task_namespace)[:5]
    cues = _normalize_support(params, gen_defaults, key=str(spec.cue_pool_key), fallback=spec.cue_pool, task_id=task_namespace)[:5]
    context_cues = _CONTEXT_CUE_CODES[:context_count]
    if len(contexts) < context_count or len(actions) < 5 or len(cues) < 5:
        raise ValueError(f"{task_namespace}/{objective} requires at least {context_count} contexts plus 5 actions and cues")

    controls_without_labels: List[_ControlSpec] = []
    order = 0
    for context_index, context_label in enumerate(contexts):
        for action_index, action_label in enumerate(actions):
            controls_without_labels.append(
                _ControlSpec(
                    control_id=f"ctx_{context_index:02d}_act_{action_index:02d}",
                    candidate_label="",
                    role=str(spec.control_role),
                    display_text=str(_ACTION_SYMBOLS[int(action_index) % len(_ACTION_SYMBOLS)]),
                    context_label=str(context_label),
                    context_cue_label=str(context_cues[int(context_index)]),
                    action_label=str(action_label),
                    cue_label=str(cues[int(action_index)]),
                    code_label=str(_GUIDE_CODES[int(action_index) % len(_GUIDE_CODES)]),
                    context_index=int(context_index),
                    action_index=int(action_index),
                    order_index=int(order),
                )
            )
            order += 1
    target_rng = spawn_rng(int(instance_seed), f"{task_namespace}.{objective}.target")
    target_context_index = -1
    target_state_id = ""
    target_state_phrase = ""
    counted_control_ids: Tuple[str, ...] = ()
    answer_value = 0
    if str(objective) == PROMPT_CONTEXT_COUNT_KEY:
        state_rng = spawn_rng(int(instance_seed), f"{task_namespace}.{objective}.state")
        count_rng = spawn_rng(int(instance_seed), f"{task_namespace}.{objective}.count")
        target_context_index, context_index_probabilities = _resolve_explicit_or_integer_range(
            params,
            rng=target_rng,
            explicit_keys=("target_context_index",),
            minimum=0,
            maximum=int(context_count) - 1,
        )
        target_state_id, state_probabilities = _resolve_explicit_or_uniform(
            params,
            rng=state_rng,
            explicit_keys=("target_state_id", "control_state"),
            support=_COUNT_CONTROL_STATES,
        )
        answer_value, answer_value_probabilities = _resolve_explicit_or_integer_range(
            params,
            rng=count_rng,
            explicit_keys=("answer_value", "target_count"),
            minimum=0,
            maximum=5,
        )
        action_indices = list(range(5))
        action_choice_rng = spawn_rng(int(instance_seed), f"{task_namespace}.{objective}.matching_actions")
        action_choice_rng.shuffle(action_indices)
        matching_actions = set(action_indices[: int(answer_value)])
        non_target_states = tuple(state for state in (_CONTROL_STATE_NORMAL, *_COUNT_CONTROL_STATES) if str(state) != str(target_state_id))
        state_assign_rng = spawn_rng(int(instance_seed), f"{task_namespace}.{objective}.state_assignments")
        state_controls: List[_ControlSpec] = []
        for control in controls_without_labels:
            if int(control.context_index) == int(target_context_index):
                state_id = str(target_state_id) if int(control.action_index) in matching_actions else str(non_target_states[int(state_assign_rng.randrange(len(non_target_states)))])
            else:
                support = (_CONTROL_STATE_NORMAL, *_COUNT_CONTROL_STATES)
                state_id = str(support[int(state_assign_rng.randrange(len(support)))])
            state_controls.append(_with_state(control, state_id=str(state_id)))
        controls_without_labels = state_controls
        counted_control_ids = tuple(
            str(control.control_id)
            for control in controls_without_labels
            if int(control.context_index) == int(target_context_index) and str(control.state_id) == str(target_state_id)
        )
        target_state_phrase = _state_phrase(str(target_state_id))
        target = next(control for control in controls_without_labels if int(control.context_index) == int(target_context_index))
    else:
        target_index = _support_selection_index(
            params,
            task_id=task_namespace,
            supported_query_ids=definition.supported_query_ids,
            instance_seed=int(instance_seed),
            namespace=f"target.{objective}.{workspace_variant}",
        ) % len(controls_without_labels)
        target = controls_without_labels[int(target_index)]
        context_index_probabilities = {}
        state_probabilities = {}
        answer_value_probabilities = {}

    target_label = str(
        params.get(
            "target_label",
            candidate_label_pool[
                _support_selection_index(
                    params,
                    task_id=task_namespace,
                    supported_query_ids=definition.supported_query_ids,
                    instance_seed=int(instance_seed),
                    namespace=f"answer_label.{objective}.{workspace_variant}",
                )
                % len(candidate_label_pool)
            ],
        )
    )
    remaining_labels = [str(value) for value in candidate_label_pool if str(value) != str(target_label)]
    label_rng = spawn_rng(int(instance_seed), f"{task_namespace}.candidate_labels")
    label_rng.shuffle(remaining_labels)
    controls: List[_ControlSpec] = []
    cursor = 0
    for control in controls_without_labels:
        label = str(target_label) if str(control.control_id) == str(target.control_id) else str(remaining_labels[int(cursor)])
        if str(control.control_id) != str(target.control_id):
            cursor += 1
        controls.append(
            _ControlSpec(
                control_id=str(control.control_id),
                candidate_label=str(label),
                role=str(control.role),
                display_text=str(control.display_text),
                context_label=str(control.context_label),
                context_cue_label=str(control.context_cue_label),
                action_label=str(control.action_label),
                cue_label=str(control.cue_label),
                code_label=str(control.code_label),
                context_index=int(control.context_index),
                action_index=int(control.action_index),
                order_index=int(control.order_index),
                state_id=str(control.state_id),
                state_phrase=str(control.state_phrase),
            )
        )
    guide_order = list(range(len(actions)))
    spawn_rng(int(instance_seed), f"{task_namespace}.guide_order.{workspace_variant}").shuffle(guide_order)
    context_guide_order = list(range(len(contexts)))
    spawn_rng(int(instance_seed), f"{task_namespace}.context_guide_order.{workspace_variant}").shuffle(context_guide_order)
    if str(objective) == PROMPT_CONTEXT_GUIDE_LABEL_KEY:
        instruction_templates = _CONTEXT_GUIDE_INSTRUCTIONS
    else:
        instruction_templates = spec.instruction_templates
    template_index = _support_selection_index(
        params,
        task_id=task_namespace,
        supported_query_ids=definition.supported_query_ids,
        instance_seed=int(instance_seed),
        namespace=f"instruction_template.{objective}.{workspace_variant}",
    ) % len(instruction_templates)
    instruction_text = str(instruction_templates[int(template_index)]).format(
        context_label=str(target.context_label),
        context_cue_label=str(target.context_cue_label),
        cue_label=str(target.cue_label),
        action_label=str(target.action_label),
        code_label=str(target.code_label),
    )
    return _ResolvedQuery(
        objective_key=str(objective),
        query_id=str(objective),
        scene_variant=str(scene_variant),
        workspace_variant=str(workspace_variant),
        variant_spec=spec,
        controls=tuple(controls),
        target_control_id=str(target.control_id),
        target_label=str(target_label),
        context_label=str(target.context_label),
        action_label=str(target.action_label),
        cue_label=str(target.cue_label),
        context_cue_label=str(target.context_cue_label),
        code_label=str(target.code_label),
        instruction_text=str(instruction_text),
        guide_order=tuple(int(value) for value in guide_order),
        context_guide_order=tuple(int(value) for value in context_guide_order),
        context_count=int(context_count),
        context_count_range=tuple(int(value) for value in context_count_range),
        candidate_label_pool=tuple(str(value) for value in candidate_label_pool),
        query_id_probabilities=dict(objective_probabilities),
        workspace_variant_probabilities=dict(workspace_variant_probabilities),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        answer_value=int(answer_value),
        target_context_index=int(target_context_index),
        target_state_id=str(target_state_id),
        target_state_phrase=str(target_state_phrase),
        counted_control_ids=tuple(str(control_id) for control_id in counted_control_ids),
    )


def _add_support(
    support_bboxes: Dict[str, List[float]],
    support_records: List[Dict[str, Any]],
    support_id: str,
    support_kind: str,
    display_text: str,
    bbox: BBox,
) -> None:
    support_bboxes[str(support_id)] = _bbox_list(bbox)
    support_records.append(
        {
            "support_id": str(support_id),
            "support_kind": str(support_kind),
            "display_text": str(display_text),
            "bbox_px": _bbox_list(bbox),
        }
    )


def _layout_tints(layout: str, theme: Any) -> Tuple[Color, Color, Color]:
    if str(layout) == "property_panel":
        return (theme.panel_alt_fill, (245, 247, 250), theme.selected_fill)
    if str(layout) == "canvas_tool":
        return ((237, 242, 239), (250, 250, 246), (228, 238, 250))
    if str(layout) == "code_workspace":
        return ((238, 241, 246), (249, 250, 252), (232, 241, 250))
    if str(layout) == "file_dialog":
        return ((245, 247, 250), (255, 255, 255), (231, 240, 248))
    return (theme.panel_alt_fill, theme.control_fill, theme.selected_fill)


def _text_on_fill(fill: Sequence[int], theme: Any) -> Color:
    """Choose a high-contrast text color for one rendered surface fill."""

    surface = normalize_rgb(fill)
    candidates = (
        getattr(theme, "control_text", (250, 252, 255)),
        getattr(theme, "muted_text", (92, 102, 118)),
        getattr(theme, "badge_text", (255, 255, 255)),
        getattr(theme, "title_text", (255, 255, 255)),
        (10, 14, 22),
        (255, 255, 255),
        (0, 0, 0),
    )
    return max(
        (normalize_rgb(candidate) for candidate in candidates),
        key=lambda candidate: contrast_ratio(candidate, surface),
    )


def _control_visual_style(control: _ControlSpec, theme: Any) -> Tuple[Color, Color, Color, int]:
    state_id = str(control.state_id)
    if state_id == "blue_highlighted":
        return (220, 238, 255), (59, 116, 190), (13, 45, 96), 3
    if state_id == "gray_disabled":
        return (229, 233, 238), (146, 156, 170), (95, 104, 118), 2
    if state_id == "orange_warning":
        return (255, 237, 203), (204, 123, 38), (92, 52, 9), 3
    return theme.control_fill, theme.control_outline, theme.control_text, int(theme.control_outline_width_px if hasattr(theme, "control_outline_width_px") else 0) or 0


def _draw_professional_scene(
    image: Image.Image,
    *,
    query: _ResolvedQuery,
    render_params: _RenderParams,
    theme: Any,
) -> _RenderedScene:
    """Render the selected workspace layout and collect control/support bboxes from it."""

    draw = ImageDraw.Draw(image)
    content_bbox, profile = _draw_app_chrome(draw, query=query, render_params=render_params, theme=theme)
    x1, y1, x2, y2 = [float(value) for value in content_bbox]
    spec = query.variant_spec
    pale_fill, surface_fill, selected_fill = _layout_tints(str(spec.layout), theme)

    title_bar = (x1 + 18.0, y1 + 10.0, x2 - 18.0, y1 + 50.0)
    _rounded_rect(draw, title_bar, radius=10, fill=theme.panel_alt_fill, outline=theme.chrome_line, width=1)
    _draw_text_left(
        draw,
        text=str(spec.scene_title),
        bbox=(title_bar[0] + 18.0, title_bar[1] + 8.0, title_bar[0] + 480.0, title_bar[3] - 8.0),
        fill=theme.control_text,
        max_size_px=int(render_params.body_font_size_px),
        bold=True,
    )
    draw_text_traced(draw,(title_bar[2] - 150.0, title_bar[1] + 13.0), str(profile.status_text), fill=theme.muted_text, font=load_font(int(render_params.small_font_size_px)), role="readout", required=False)

    workspace = (x1 + 18.0, title_bar[3] + 12.0, x2 - 18.0, y2 - 16.0)
    _rounded_rect(draw, workspace, radius=10, fill=theme.panel_fill, outline=theme.chrome_line)
    support_bboxes: Dict[str, List[float]] = {}
    support_records: List[Dict[str, Any]] = []
    control_bboxes: Dict[str, List[float]] = {}
    badge_bboxes: Dict[str, List[float]] = {}

    has_context_guide = str(query.objective_key) == PROMPT_CONTEXT_GUIDE_LABEL_KEY
    show_action_guide = str(query.objective_key) != PROMPT_CONTEXT_GUIDE_LABEL_KEY
    guide_y1 = workspace[1] + 14.0
    guide_h = 58.0 if has_context_guide else 72.0
    guide_x1 = workspace[0] + 205.0
    guide_x2 = workspace[2] - 18.0
    if has_context_guide:
        _draw_text_left(
            draw,
            text="Context Cue Guide",
            bbox=(workspace[0] + 18.0, guide_y1, workspace[0] + 190.0, guide_y1 + 30.0),
            fill=theme.control_text,
            max_size_px=int(render_params.body_font_size_px),
            bold=True,
        )
        contexts = sorted(
            {
                (int(control.context_index), str(control.context_label), str(control.context_cue_label))
                for control in query.controls
            }
        )
        context_guide_w = (guide_x2 - guide_x1) / float(len(contexts))
        context_by_index = {int(index): (str(context_label), str(context_cue_label)) for index, context_label, context_cue_label in contexts}
        for visual_index, context_index in enumerate(query.context_guide_order):
            context_label, context_cue_label = context_by_index[int(context_index)]
            bbox = (
                guide_x1 + visual_index * context_guide_w,
                guide_y1,
                guide_x1 + (visual_index + 1) * context_guide_w - 8.0,
                guide_y1 + guide_h,
            )
            fill = _ACCENT_FILLS[int(context_index) % len(_ACCENT_FILLS)]
            outline = _ACCENT_LINES[int(context_index) % len(_ACCENT_LINES)]
            _rounded_rect(draw, bbox, radius=8, fill=fill, outline=outline, width=2)
            _draw_text_center_fit(
                draw,
                text=f"{context_cue_label}\n{context_label}",
                bbox=(bbox[0] + 8.0, bbox[1] + 5.0, bbox[2] - 8.0, bbox[3] - 5.0),
                fill=_text_on_fill(fill, theme),
                max_size_px=int(render_params.small_font_size_px),
                bold=True,
            )
            _add_support(
                support_bboxes,
                support_records,
                f"context_guide_{context_index}",
                "context_cue_guide",
                f"{context_cue_label} -> {context_label}",
                bbox,
            )
        guide_y1 = guide_y1 + guide_h + 10.0

    guide_w = (guide_x2 - guide_x1) / 5.0
    actions = sorted({(int(control.action_index), str(control.action_label), str(control.cue_label), str(control.code_label)) for control in query.controls})
    if show_action_guide:
        _draw_text_left(
            draw,
            text=str(spec.guide_title),
            bbox=(workspace[0] + 18.0, guide_y1, workspace[0] + 190.0, guide_y1 + 30.0),
            fill=theme.control_text,
            max_size_px=int(render_params.body_font_size_px),
            bold=True,
        )
        for visual_index, action_index in enumerate(query.guide_order):
            action_tuple = actions[int(action_index)]
            _idx, action_label, cue_label, code_label = action_tuple
            bbox = (
                guide_x1 + visual_index * guide_w,
                guide_y1,
                guide_x1 + (visual_index + 1) * guide_w - 8.0,
                guide_y1 + guide_h,
            )
            fill = _ACCENT_FILLS[int(action_index) % len(_ACCENT_FILLS)]
            outline = _ACCENT_LINES[int(action_index) % len(_ACCENT_LINES)]
            _rounded_rect(draw, bbox, radius=8, fill=fill, outline=outline, width=2)
            guide_text_fill = _text_on_fill(fill, theme)
            _draw_text_center_fit(
                draw,
                text=f"{cue_label}\nKey {code_label}",
                bbox=(bbox[0] + 8.0, bbox[1] + 6.0, bbox[2] - 8.0, bbox[3] - 6.0),
                fill=guide_text_fill,
                max_size_px=int(render_params.small_font_size_px),
                bold=True,
            )
            _add_support(support_bboxes, support_records, f"guide_{action_index}", str(spec.guide_kind), f"{cue_label} -> {code_label}", bbox)

    body_y1 = guide_y1 + (guide_h + 14.0 if show_action_guide else 14.0)
    body_y2 = workspace[3] - 18.0
    context_x1 = workspace[0] + 18.0
    context_x2 = workspace[0] + 300.0
    matrix_x1 = context_x2 + 16.0
    matrix_x2 = workspace[2] - 18.0
    context_header = (context_x1, body_y1, context_x2, body_y1 + 36.0)
    matrix_header = (matrix_x1, body_y1, matrix_x2, body_y1 + 36.0)
    _rounded_rect(draw, context_header, radius=8, fill=pale_fill, outline=theme.chrome_line)
    _rounded_rect(draw, matrix_header, radius=8, fill=pale_fill, outline=theme.chrome_line)
    pale_text_fill = _text_on_fill(pale_fill, theme)
    _draw_text_center_fit(
        draw,
        text=str(spec.context_title),
        bbox=context_header,
        fill=pale_text_fill,
        max_size_px=int(render_params.small_font_size_px),
        bold=True,
    )
    _draw_text_center_fit(
        draw,
        text=str(spec.header_title),
        bbox=matrix_header,
        fill=pale_text_fill,
        max_size_px=int(render_params.small_font_size_px),
        bold=True,
    )

    header_y1 = body_y1 + 46.0
    header_h = 40.0
    row_y1 = header_y1 + header_h + 8.0
    col_w = (matrix_x2 - matrix_x1) / 5.0
    action_indices = sorted({int(control.action_index) for control in query.controls})
    context_indices = sorted({int(control.context_index) for control in query.controls})
    row_h = (body_y2 - row_y1) / float(len(context_indices))
    for action_index in action_indices:
        header_bbox = (
            matrix_x1 + action_index * col_w,
            header_y1,
            matrix_x1 + (action_index + 1) * col_w - 8.0,
            header_y1 + header_h,
        )
        fill = _ACCENT_FILLS[int(action_index) % len(_ACCENT_FILLS)]
        outline = _ACCENT_LINES[int(action_index) % len(_ACCENT_LINES)]
        code_label = next(str(control.code_label) for control in query.controls if int(control.action_index) == int(action_index))
        _rounded_rect(draw, header_bbox, radius=8, fill=fill, outline=outline, width=2)
        header_text_fill = _text_on_fill(fill, theme)
        _draw_text_center_fit(
            draw,
            text=f"Key {code_label}",
            bbox=header_bbox,
            fill=header_text_fill,
            max_size_px=int(render_params.small_font_size_px),
            bold=True,
        )
        action_label = next(str(control.action_label) for control in query.controls if int(control.action_index) == int(action_index))
        _add_support(support_bboxes, support_records, f"header_{action_index}", str(spec.header_kind), f"{code_label}: {action_label}", header_bbox)

    for context_index in context_indices:
        row_bbox = (
            context_x1,
            row_y1 + context_index * row_h,
            context_x2,
            row_y1 + (context_index + 1) * row_h - 8.0,
        )
        row_fill = selected_fill if context_index % 2 == 0 else surface_fill
        _rounded_rect(draw, row_bbox, radius=8, fill=row_fill, outline=theme.chrome_line)
        context_label = next(str(control.context_label) for control in query.controls if int(control.context_index) == int(context_index))
        prefix = {
            "toolbar_palette": "Mode",
            "property_panel": "Section",
            "canvas_tool": "Object",
            "code_workspace": "Target",
            "file_dialog": "Location",
        }.get(str(spec.layout), "Context")
        if str(spec.layout) == "file_dialog":
            draw.rectangle([row_bbox[0] + 16.0, row_bbox[1] + 16.0, row_bbox[0] + 42.0, row_bbox[1] + 38.0], fill=theme.accent_alt)
        row_text_x = row_bbox[0] + (54.0 if str(spec.layout) == "file_dialog" else 12.0)
        _draw_text_left(
            draw,
            text=f"{prefix}: {context_label}",
            bbox=(row_text_x, row_bbox[1] + 8.0, row_bbox[2] - 10.0, row_bbox[3] - 8.0),
            fill=_text_on_fill(row_fill, theme),
            max_size_px=int(render_params.small_font_size_px),
            bold=True,
        )
        _add_support(support_bboxes, support_records, f"context_{context_index}", str(spec.context_kind), str(context_label), row_bbox)

        if str(spec.layout) == "code_workspace":
            draw.rectangle([row_bbox[0] + 14.0, row_bbox[3] - 18.0, row_bbox[2] - 14.0, row_bbox[3] - 15.0], fill=theme.accent)

    controls_by_pos = {(int(control.context_index), int(control.action_index)): control for control in query.controls}
    for context_index in context_indices:
        for action_index in action_indices:
            control = controls_by_pos[(int(context_index), int(action_index))]
            bbox = (
                matrix_x1 + action_index * col_w,
                row_y1 + context_index * row_h,
                matrix_x1 + (action_index + 1) * col_w - 8.0,
                row_y1 + (context_index + 1) * row_h - 8.0,
            )
            control_fill, control_outline, control_text, state_outline_width = _control_visual_style(control, theme)
            _rounded_rect(
                draw,
                bbox,
                radius=int(render_params.control_corner_radius_px),
                fill=control_fill,
                outline=control_outline,
                width=max(int(render_params.control_outline_width_px), int(state_outline_width)),
            )
            if str(control.state_id) == "gray_disabled":
                draw.line([bbox[0] + 12.0, bbox[3] - 12.0, bbox[2] - 12.0, bbox[1] + 12.0], fill=(168, 176, 188), width=1)
            elif str(control.state_id) == "orange_warning":
                marker = (bbox[2] - 23.0, bbox[1] + 10.0, bbox[2] - 9.0, bbox[1] + 24.0)
                draw.ellipse([float(value) for value in marker], fill=(204, 123, 38))
                _draw_text_center_fit(
                    draw,
                    text="!",
                    bbox=(marker[0] + 2.0, marker[1] - 1.0, marker[2] - 2.0, marker[3] + 1.0),
                    fill=(255, 255, 255),
                    max_size_px=10,
                    bold=True,
                )
            _draw_text_center_fit(
                draw,
                text=str(control.display_text),
                bbox=(bbox[0] + 46.0, bbox[1] + 5.0, bbox[2] - 8.0, bbox[3] - 5.0),
                fill=control_text,
                max_size_px=int(render_params.small_font_size_px),
                bold=True,
            )
            badge_bboxes[str(control.control_id)] = _draw_badge(draw, control_bbox=bbox, label=str(control.candidate_label), render_params=render_params, theme=theme)
            control_bboxes[str(control.control_id)] = _bbox_list(bbox)

    control_records: List[Dict[str, Any]] = []
    for control in query.controls:
        control_records.append(
            {
                "control_id": str(control.control_id),
                "candidate_label": str(control.candidate_label),
                "role": str(control.role),
                "display_text": str(control.display_text),
                "context_label": str(control.context_label),
                "context_cue_label": str(control.context_cue_label),
                "action_label": str(control.action_label),
                "cue_label": str(control.cue_label),
                "code_label": str(control.code_label),
                "context_index": int(control.context_index),
                "action_index": int(control.action_index),
                "order_index": int(control.order_index),
                "state_id": str(control.state_id),
                "state_phrase": str(control.state_phrase),
                "bbox_px": list(control_bboxes[str(control.control_id)]),
                "candidate_label_bbox_px": list(badge_bboxes[str(control.control_id)]),
            }
        )
    m = int(render_params.window_margin_px)
    window_bbox = [float(m), float(m - 6), float(render_params.canvas_width - m), float(render_params.canvas_height - m + 6)]
    return _RenderedScene(
        control_bboxes_by_id={str(key): list(value) for key, value in control_bboxes.items()},
        badge_bboxes_by_id={str(key): list(value) for key, value in badge_bboxes.items()},
        support_bboxes_by_id={str(key): list(value) for key, value in support_bboxes.items()},
        control_records=tuple(control_records),
        support_records=tuple(dict(record) for record in support_records),
        scene_bbox_px=[0.0, 0.0, float(render_params.canvas_width), float(render_params.canvas_height)],
        window_bbox_px=_bbox_list(tuple(window_bbox)),
        profile=profile,
        theme=theme,
    )


def _annotation_support_ids(query: _ResolvedQuery) -> Tuple[str, str, str]:
    target = next(control for control in query.controls if str(control.control_id) == str(query.target_control_id))
    return (f"guide_{int(target.action_index)}", f"context_{int(target.context_index)}", f"header_{int(target.action_index)}")


def _context_guide_annotation_support_ids(query: _ResolvedQuery) -> Tuple[str, str, str]:
    target = next(control for control in query.controls if str(control.control_id) == str(query.target_control_id))
    return (
        f"context_guide_{int(target.context_index)}",
        f"context_{int(target.context_index)}",
        f"header_{int(target.action_index)}",
    )


def _annotation_roles(query: _ResolvedQuery) -> Tuple[str, str, str, str]:
    spec = query.variant_spec
    return (
        str(spec.guide_kind),
        f"{str(spec.context_kind)}_row",
        str(spec.header_kind),
        f"target_{str(spec.control_role)}",
    )


def _prompt_json_examples(query: _ResolvedQuery) -> Tuple[str, str]:
    if str(query.objective_key) == PROMPT_CONTEXT_COUNT_KEY:
        answer_and_annotation = {
            "annotation": [[410, 378, 560, 418], [580, 378, 730, 418]],
            "answer": 2,
        }
        answer_only = {"answer": 2}
    elif str(query.objective_key) == PROMPT_CONTEXT_GUIDE_LABEL_KEY:
        answer_and_annotation = {
            "annotation": [520, 360, 690, 444],
            "answer": "G",
        }
        answer_only = {"answer": "G"}
    else:
        answer_and_annotation = {
            "annotation": [520, 360, 690, 444],
            "answer": "G",
        }
        answer_only = {"answer": "G"}
    return (
        json.dumps(answer_and_annotation, ensure_ascii=True, allow_nan=False, separators=(",", ":")),
        json.dumps(answer_only, ensure_ascii=True, allow_nan=False, separators=(",", ":")),
    )

def _prompt_key(*parts: str) -> str:
    return "_".join(str(part) for part in parts)


_TOOLBAR_WORKSPACE_VARIANT = "toolbar_palette"
_PROPERTY_WORKSPACE_VARIANT = "property_panel"
_CANVAS_WORKSPACE_VARIANT = "canvas_tool"
_CODE_WORKSPACE_VARIANT = "code_workspace"
_FILE_WORKSPACE_VARIANT = "file_dialog"
SUPPORTED_WORKSPACE_VARIANTS = (
    _TOOLBAR_WORKSPACE_VARIANT,
    _PROPERTY_WORKSPACE_VARIANT,
    _CANVAS_WORKSPACE_VARIANT,
    _CODE_WORKSPACE_VARIANT,
    _FILE_WORKSPACE_VARIANT,
)
SUPPORTED_PROMPT_QUERY_KEYS = (
    PROMPT_CONTROL_LABEL_KEY,
    PROMPT_CONTEXT_COUNT_KEY,
    PROMPT_CONTEXT_GUIDE_LABEL_KEY,
)


def _variant(
    name: str,
    *,
    layout: str,
    scene_title: str,
    context_title: str,
    guide_title: str,
    header_title: str,
    context_kind: str,
    guide_kind: str,
    header_kind: str,
    control_role: str,
    context_pool: Tuple[str, ...],
    actions: Tuple[str, ...],
    cues: Tuple[str, ...],
    instruction_templates: Tuple[str, ...],
) -> ProfessionalVariantSpec:
    return ProfessionalVariantSpec(
        name=name,
        layout=layout,
        scene_title=scene_title,
        context_title=context_title,
        guide_title=guide_title,
        header_title=header_title,
        context_kind=context_kind,
        guide_kind=guide_kind,
        header_kind=header_kind,
        control_role=control_role,
        context_pool_key=f"{name}_context_pool",
        action_pool_key=f"{name}_action_pool",
        cue_pool_key=f"{name}_cue_pool",
        context_pool=context_pool,
        action_pool=actions,
        cue_pool=cues,
        instruction_templates=instruction_templates,
    )


_DEFAULT_INSTRUCTIONS = (
    'For "{context_label}", use the guide cue "{cue_label}".',
    'In "{context_label}", choose the control for "{cue_label}".',
    'Find the "{cue_label}" control associated with "{context_label}".',
    'Use "{cue_label}" for the visible context "{context_label}".',
    'Select the labeled control for "{cue_label}" in "{context_label}".',
)
_CONTEXT_GUIDE_INSTRUCTIONS = (
    'Use context cue "{context_cue_label}" and Key {code_label}.',
    'Find the row for "{context_cue_label}", then use Key {code_label}.',
    'Choose the control in the row from "{context_cue_label}" and column Key {code_label}.',
    'Map "{context_cue_label}" to its row and use Key {code_label}.',
    'Resolve context cue "{context_cue_label}" and the stated Key {code_label}.',
)


TASK_DEFINITION = ProfessionalTaskDefinition(
    task_namespace=TASK_NAMESPACE,
    scene_kind="gui_professional_target",
    question_format="workspace_control_label",
    supported_query_ids=(SINGLE_QUERY_ID,),
    variants=(
        _variant(
            _TOOLBAR_WORKSPACE_VARIANT,
            layout="toolbar_palette",
            scene_title="Tool Palette Workspace",
            context_title="Tool contexts",
            guide_title="Tool Cue Guide",
            header_title="Coded tool headers",
            context_kind="tool_context",
            guide_kind="tool_cue_card",
            header_kind="tool_code_header",
            control_role="toolbar_palette_control",
            context_pool=("Sketch", "Model", "Inspect", "Annotate", "Review"),
            actions=("Copy", "Mirror", "Align", "Measure", "Reset"),
            cues=("copy item", "flip item", "line up", "read span", "fresh state"),
            instruction_templates=_DEFAULT_INSTRUCTIONS,
        ),
        _variant(
            _PROPERTY_WORKSPACE_VARIANT,
            layout="property_panel",
            scene_title="Inspector Settings Panel",
            context_title="Inspector sections",
            guide_title="Setting Cue Guide",
            header_title="Coded setting headers",
            context_kind="inspector_section",
            guide_kind="setting_cue_card",
            header_kind="setting_code_header",
            control_role="property_panel_control",
            context_pool=("General", "Layout", "Display", "Data", "Output"),
            actions=("Name", "Width", "Visible", "Required", "Format"),
            cues=("title field", "wide value", "show item", "must fill", "output type"),
            instruction_templates=_DEFAULT_INSTRUCTIONS,
        ),
        _variant(
            _CANVAS_WORKSPACE_VARIANT,
            layout="canvas_tool",
            scene_title="Canvas And Viewport Controls",
            context_title="Canvas targets",
            guide_title="Canvas Cue Guide",
            header_title="Coded viewport headers",
            context_kind="canvas_target",
            guide_kind="canvas_cue_card",
            header_kind="canvas_code_header",
            control_role="canvas_workspace_control",
            context_pool=("Object A", "Object B", "Object C", "Object D", "Object E"),
            actions=("Bounds", "Guide", "Mask", "Note", "Measure"),
            cues=("edge box", "helper line", "cover area", "text note", "read length"),
            instruction_templates=_DEFAULT_INSTRUCTIONS,
        ),
        _variant(
            _CODE_WORKSPACE_VARIANT,
            layout="code_workspace",
            scene_title="IDE Workspace Controls",
            context_title="Code targets",
            guide_title="IDE Cue Guide",
            header_title="Coded IDE headers",
            context_kind="code_target",
            guide_kind="ide_cue_card",
            header_kind="ide_code_header",
            control_role="code_workspace_control",
            context_pool=("App", "Tests", "API", "Worker", "Docs"),
            actions=("Open", "Save", "Split", "Search", "Kill"),
            cues=("show file", "write disk", "second pane", "find text", "end shell"),
            instruction_templates=_DEFAULT_INSTRUCTIONS,
        ),
        _variant(
            _FILE_WORKSPACE_VARIANT,
            layout="file_dialog",
            scene_title="File Dialog And Window Controls",
            context_title="Dialog locations",
            guide_title="Dialog Cue Guide",
            header_title="Coded dialog headers",
            context_kind="dialog_location",
            guide_kind="dialog_cue_card",
            header_kind="dialog_code_header",
            control_role="file_dialog_control",
            context_pool=("Desktop", "Downloads", "Projects", "Shared", "Archive"),
            actions=("Select", "Preview", "Save", "Open", "Options"),
            cues=("pick file", "quick view", "write file", "choose file", "extra choices"),
            instruction_templates=_DEFAULT_INSTRUCTIONS,
        ),
    ),
)


class WorkspaceTargetTaskBase:
    """Base class for GUI target-grounding tasks."""

    domain = "pages"
    scene_id = SCENE
    definition: ProfessionalTaskDefinition

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one workspace instance for a public workspace objective."""

        del max_attempts
        definition = self.definition
        task_namespace = str(definition.task_namespace)
        objective_key = str(params.get("_objective_key", "")).strip()
        if not objective_key:
            raise ValueError("_objective_key is required for workspace lifecycle generation")
        question_format = str(params.get("_question_format", str(definition.question_format)))
        gen_defaults, render_defaults, prompt_defaults, _background_defaults, noise_defaults = _workspace_defaults()
        query = _resolve_query(
            definition,
            int(instance_seed),
            params=params,
            objective_key=str(objective_key),
            gen_defaults=gen_defaults,
        )
        render_params = _resolve_render_params(params, render_defaults, instance_seed=int(instance_seed))
        information_style, information_style_meta = resolve_pages_information_style(
            instance_seed=int(instance_seed),
            params=params,
            scene_id=SCENE,
        )
        background, background_meta = make_pages_information_background(
            canvas_width=int(render_params.canvas_width),
            canvas_height=int(render_params.canvas_height),
            style=information_style,
            instance_seed=int(instance_seed),
            namespace=f"{task_namespace}.background",
        )
        theme = _theme_from_information_style(information_style)
        image = background.copy().convert("RGB")
        rendered = _draw_professional_scene(image, query=query, render_params=render_params, theme=theme)
        image, post_noise_meta = apply_post_image_noise(
            image,
            instance_seed=int(instance_seed),
            params=params,
            default_config=noise_defaults,
        )

        support_records = [dict(record) for record in rendered.support_records]
        control_records = [dict(record) for record in rendered.control_records]
        control_record_by_id = {str(record["control_id"]): dict(record) for record in control_records}
        if str(query.objective_key) == PROMPT_CONTEXT_COUNT_KEY:
            target_context_support_id = f"context_{int(query.target_context_index)}"
            target_context_record = next(record for record in support_records if str(record["support_id"]) == str(target_context_support_id))
            context_role = f"{str(query.variant_spec.context_kind)}_row"
            counted_records = [dict(control_record_by_id[str(control_id)]) for control_id in query.counted_control_ids]
            annotation_bboxes = [list(record["bbox_px"]) for record in counted_records]
            annotation_support_ids = (str(target_context_support_id),)
            annotation_support_records = [dict(target_context_record)]
            annotation_role_support_ids: Dict[str, str] = {str(context_role): str(target_context_support_id)}
            support_bbox_map: Dict[str, List[float]] = {str(context_role): list(target_context_record["bbox_px"])}
            answer_gt = TypedValue(type="integer", value=int(query.answer_value))
            annotation_gt = TypedValue(type="bbox_set", value=list(annotation_bboxes))
            witness_symbolic = {
                "type": "bbox_set",
                "annotation_support_ids": [str(value) for value in annotation_support_ids],
                "annotation_role_support_ids": dict(annotation_role_support_ids),
                "target_context_index": int(query.target_context_index),
                "target_state_id": str(query.target_state_id),
                "target_state_phrase": str(query.target_state_phrase),
                "counted_control_ids": [str(value) for value in query.counted_control_ids],
                "support_bbox_map": dict(support_bbox_map),
                "value": list(annotation_bboxes),
            }
            projected_annotation = {
                "type": "bbox_set",
                "bbox_set": list(annotation_bboxes),
                "pixel_bbox_set": list(annotation_bboxes),
            }
            target_record = dict(control_record_by_id[str(query.target_control_id)])
        elif str(query.objective_key) == PROMPT_CONTEXT_GUIDE_LABEL_KEY:
            target_record = dict(control_record_by_id[str(query.target_control_id)])
            annotation_support_ids = _context_guide_annotation_support_ids(query)
            annotation_support_records = [
                next(record for record in support_records if str(record["support_id"]) == str(support_id))
                for support_id in annotation_support_ids
            ]
            support_bbox_map = {
                "context_cue_guide": list(annotation_support_records[0]["bbox_px"]),
                "context_row": list(annotation_support_records[1]["bbox_px"]),
                "action_code_header": list(annotation_support_records[2]["bbox_px"]),
                "target_control": list(target_record["bbox_px"]),
            }
            annotation_role_support_ids = {
                "context_cue_guide": str(annotation_support_ids[0]),
                "context_row": str(annotation_support_ids[1]),
                "action_code_header": str(annotation_support_ids[2]),
                "target_control": str(query.target_control_id),
            }
            target_annotation_bbox = list(target_record["bbox_px"])
            answer_gt = TypedValue(type="option_letter", value=str(query.target_label))
            annotation_gt = TypedValue(type="bbox", value=list(target_annotation_bbox))
            witness_symbolic = {
                "type": "bbox",
                "annotation_support_ids": [str(value) for value in annotation_support_ids],
                "annotation_role_support_ids": dict(annotation_role_support_ids),
                "target_control_id": str(query.target_control_id),
                "support_bbox_map": dict(support_bbox_map),
                "value": list(target_annotation_bbox),
            }
            projected_annotation = {
                "type": "bbox",
                "bbox": list(target_annotation_bbox),
                "pixel_bbox": list(target_annotation_bbox),
            }
        else:
            target_record = dict(control_record_by_id[str(query.target_control_id)])
            annotation_support_ids = _annotation_support_ids(query)
            guide_role, context_role, header_role, target_role = _annotation_roles(query)
            annotation_support_records = [
                next(record for record in support_records if str(record["support_id"]) == str(support_id))
                for support_id in annotation_support_ids
            ]
            support_bbox_map = {
                str(guide_role): list(annotation_support_records[0]["bbox_px"]),
                str(context_role): list(annotation_support_records[1]["bbox_px"]),
                str(header_role): list(annotation_support_records[2]["bbox_px"]),
                str(target_role): list(target_record["bbox_px"]),
            }
            target_annotation_bbox = list(target_record["bbox_px"])
            annotation_role_support_ids = {
                str(guide_role): str(annotation_support_ids[0]),
                str(context_role): str(annotation_support_ids[1]),
                str(header_role): str(annotation_support_ids[2]),
                str(target_role): str(query.target_control_id),
            }
            answer_gt = TypedValue(type="option_letter", value=str(query.target_label))
            annotation_gt = TypedValue(type="bbox", value=list(target_annotation_bbox))
            witness_symbolic = {
                "type": "bbox",
                "annotation_support_ids": [str(value) for value in annotation_support_ids],
                "annotation_role_support_ids": dict(annotation_role_support_ids),
                "target_control_id": str(query.target_control_id),
                "target_control_role": str(target_role),
                "support_bbox_map": dict(support_bbox_map),
                "value": list(target_annotation_bbox),
            }
            projected_annotation = {
                "type": "bbox",
                "bbox": list(target_annotation_bbox),
                "pixel_bbox": list(target_annotation_bbox),
            }

        prompt_defaults_required = required_group_defaults(
            prompt_defaults,
            ("bundle_id",),
            context=f"prompt defaults for {getattr(self, 'task_id', task_namespace)}",
        )
        prompt_selection = render_task_prompt_variants(
            domain=self.domain,
            scene_id=self.scene_id,
            bundle_id=str(prompt_defaults_required["bundle_id"]),
            scene_key=PROMPT_SCENE_KEY,
            task_key=PROMPT_TASK_KEY,
            query_key=str(query.query_id),
            answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
            dynamic_slots={
                "instruction_text": str(query.instruction_text),
                "context_label": str(query.context_label),
                "context_cue_label": str(query.context_cue_label),
                "cue_label": str(query.cue_label),
                "action_label": str(query.action_label),
                "code_label": str(query.code_label),
                "state_phrase": str(query.target_state_phrase),
            },
            instance_seed=int(instance_seed),
        )
        prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)
        query_params = {
            "query_id": SINGLE_QUERY_ID,
            "prompt_query_key": str(query.query_id),
            "source_query_id": str(query.query_id),
            "scene_variant": str(query.scene_variant),
            "workspace_variant": str(query.workspace_variant),
            "target_control_id": str(query.target_control_id),
            "target_label": str(query.target_label),
            "context_label": str(query.context_label),
            "context_cue_label": str(query.context_cue_label),
            "action_label": str(query.action_label),
            "cue_label": str(query.cue_label),
            "code_label": str(query.code_label),
            "instruction_text": str(query.instruction_text),
            "guide_order": [int(value) for value in query.guide_order],
            "context_guide_order": [int(value) for value in query.context_guide_order],
            "candidate_label_pool": [str(value) for value in query.candidate_label_pool],
            "context_count": int(query.context_count),
            "context_count_range": [int(value) for value in query.context_count_range],
            "action_count": int(len({int(control.action_index) for control in query.controls})),
            "target_context_index": int(query.target_context_index),
            "target_state_id": str(query.target_state_id),
            "target_state_phrase": str(query.target_state_phrase),
            "answer_value": int(query.answer_value),
            "counted_control_ids": [str(value) for value in query.counted_control_ids],
            "annotation_role_support_ids": dict(annotation_role_support_ids),
            "query_id_probabilities": {SINGLE_QUERY_ID: 1.0},
            "prompt_query_key_probabilities": dict(query.query_id_probabilities),
            "workspace_variant_probabilities": dict(query.workspace_variant_probabilities),
            "scene_variant_probabilities": dict(query.scene_variant_probabilities),
        }
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=SINGLE_QUERY_ID,
            params=dict(query_params),
        )
        query_spec["scene_id"] = SCENE
        trace_payload = {
            "scene_ir": {
                "scene_id": SCENE,
                "scene_kind": str(definition.scene_kind),
                "entities": [
                    {
                        "entity_id": str(record["control_id"]),
                        "entity_type": "gui_control",
                        "attrs": {
                            "candidate_label": str(record["candidate_label"]),
                            "role": str(record["role"]),
                            "display_text": str(record["display_text"]),
                            "context_label": str(record["context_label"]),
                            "context_cue_label": str(record["context_cue_label"]),
                            "action_label": str(record["action_label"]),
                            "cue_label": str(record["cue_label"]),
                            "code_label": str(record["code_label"]),
                            "state_id": str(record["state_id"]),
                            "state_phrase": str(record["state_phrase"]),
                            "bbox_px": list(record["bbox_px"]),
                        },
                    }
                    for record in control_records
                ],
                "relations": {
                    "query_id": SINGLE_QUERY_ID,
                    "prompt_query_key": str(query.query_id),
                    "source_query_id": str(query.query_id),
                    "scene_variant": str(query.scene_variant),
                    "workspace_variant": str(query.workspace_variant),
                    "target_control_id": str(query.target_control_id),
                    "target_label": str(query.target_label),
                    "context_label": str(query.context_label),
                    "context_cue_label": str(query.context_cue_label),
                    "action_label": str(query.action_label),
                    "cue_label": str(query.cue_label),
                    "code_label": str(query.code_label),
                    "instruction_text": str(query.instruction_text),
                    "context_count": int(query.context_count),
                    "target_context_index": int(query.target_context_index),
                    "target_state_id": str(query.target_state_id),
                    "target_state_phrase": str(query.target_state_phrase),
                    "answer_value": int(query.answer_value),
                    "counted_control_ids": [str(value) for value in query.counted_control_ids],
                    "annotation_support_ids": [str(value) for value in annotation_support_ids],
                    "annotation_role_support_ids": dict(annotation_role_support_ids),
                },
                "frames": {
                    "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
                },
            },
            "query_spec": query_spec,
            "render_spec": {
                "scene_id": SCENE,
                "query_id": SINGLE_QUERY_ID,
                "prompt_query_key": str(query.query_id),
                "canvas_width": int(render_params.canvas_width),
                "canvas_height": int(render_params.canvas_height),
                "coord_space": "pixel",
                "background_style": dict(background_meta),
                "information_scene_style": dict(information_style_meta),
                "pages_information_style_policy": {
                    "domain_wrapper": "scene_renderer",
                    "scene_renderer_recorded_style": True,
                    "task_id": str(getattr(self, "task_id", task_namespace)),
                    "scene_id": SCENE,
                },
                "post_image_noise": dict(post_noise_meta),
                "scene_variant": str(query.scene_variant),
                "workspace_variant": str(query.workspace_variant),
                "window_bbox_px": list(rendered.window_bbox_px),
                "scene_bbox_px": list(rendered.scene_bbox_px),
                "render_params": asdict(render_params),
                "theme": {
                    "name": str(rendered.theme.name),
                    "accent_rgb": [int(value) for value in rendered.theme.accent],
                    "accent_alt_rgb": [int(value) for value in rendered.theme.accent_alt],
                    "badge_fill_rgb": [int(value) for value in rendered.theme.badge_fill],
                },
            },
            "render_map": {
                "image_id": "img0",
                "scene_bbox_px": list(rendered.scene_bbox_px),
                "window_bbox_px": list(rendered.window_bbox_px),
                "app_profile": asdict(rendered.profile),
                "control_bboxes_by_id": dict(rendered.control_bboxes_by_id),
                "candidate_label_badge_bboxes_by_id": dict(rendered.badge_bboxes_by_id),
                "support_bboxes_by_id": dict(rendered.support_bboxes_by_id),
                "target_control_id": str(query.target_control_id),
                "target_context_index": int(query.target_context_index),
                "target_state_id": str(query.target_state_id),
                "counted_control_ids": [str(value) for value in query.counted_control_ids],
                "annotation_support_ids": [str(value) for value in annotation_support_ids],
                "annotation_role_support_ids": dict(annotation_role_support_ids),
            },
            "execution_trace": {
                **dict(query_params),
                "annotation_support_ids": [str(value) for value in annotation_support_ids],
                "annotation_role_support_ids": dict(annotation_role_support_ids),
                "annotation_support_records": [dict(record) for record in annotation_support_records],
                "target_control": dict(target_record),
                "controls": list(control_records),
                "support_records": list(support_records),
                "total_control_count": int(len(query.controls)),
                "question_format": str(question_format),
                "answer": answer_gt.to_dict(),
            },
            "witness_symbolic": dict(witness_symbolic),
            "projected_annotation": dict(projected_annotation),
        }

        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE,
            query_id=SINGLE_QUERY_ID,
            prompt_variants=dict(prompt_artifacts.prompt_variants),
        )


class PagesWorkspaceSceneTask(WorkspaceTargetTaskBase):
    """Scene-private generator for workspace control-target tasks."""

    definition = TASK_DEFINITION


def build_workspace_response(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    task_id: str,
    objective_key: str,
    question_format: str,
) -> TaskOutput:
    task = PagesWorkspaceSceneTask()
    task.task_id = str(task_id)
    lifecycle_params = dict(params)
    lifecycle_params["_objective_key"] = str(objective_key)
    lifecycle_params["_question_format"] = str(question_format)
    return task.generate(
        int(instance_seed),
        params=lifecycle_params,
        max_attempts=1,
    )


__all__ = [
    "PagesWorkspaceSceneTask",
    "ProfessionalTaskDefinition",
    "ProfessionalVariantSpec",
    "WorkspaceTargetTaskBase",
    "SUPPORTED_PROMPT_QUERY_KEYS",
    "SUPPORTED_WORKSPACE_VARIANTS",
    "PROMPT_CONTROL_LABEL_KEY",
    "PROMPT_CONTEXT_COUNT_KEY",
    "PROMPT_CONTEXT_GUIDE_LABEL_KEY",
    "build_workspace_response",
]
