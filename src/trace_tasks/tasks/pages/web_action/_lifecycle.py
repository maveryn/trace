"""Scene-private lifecycle for web-action page tasks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

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
from ...shared.text_rendering import draw_text_centered, fit_font_to_box, load_font
from ...shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant
from ...shared.text_legibility import draw_text_traced
from .shared.styles import (
    _bbox_list,
    _clamp_unit,
    _draw_text_center_fit,
    _draw_text_left,
    _normalize_str_support,
    _rounded_rect,
    _theme_from_information_style,
)


SCENE = "web_action"
TASK_NAMESPACE = "pages.web_action"
PROMPT_BUNDLE = "pages_web_action_v1"
PROMPT_SCENE_KEY = "web_action"
PROMPT_TASK_KEY = "web_action_query"


def _prompt_key(*parts: str) -> str:
    return "_".join(str(part) for part in parts)


_CLICK_PROMPT_KEY = _prompt_key("click", "target", "label")
_TYPE_FIELD_PROMPT_KEY = _prompt_key("type", "field", "label")
_SELECT_OPTION_PROMPT_KEY = _prompt_key("select", "option", "label")
_CLICK_GUIDE_CODE_COUNT_KEY = _prompt_key("click", "guide", "code", "target", "count")
_TYPE_FIELD_GUIDE_CODE_COUNT_KEY = _prompt_key("type", "field", "guide", "code", "target", "count")
_SELECT_OPTION_GUIDE_CODE_COUNT_KEY = _prompt_key("select", "option", "guide", "code", "target", "count")
SUPPORTED_ACTION_TARGET_QUERY_IDS: Tuple[str, ...] = (
    _CLICK_PROMPT_KEY,
    _TYPE_FIELD_PROMPT_KEY,
    _SELECT_OPTION_PROMPT_KEY,
)
SUPPORTED_GUIDE_CODE_COUNT_QUERY_IDS: Tuple[str, ...] = (
    _CLICK_GUIDE_CODE_COUNT_KEY,
    _TYPE_FIELD_GUIDE_CODE_COUNT_KEY,
    _SELECT_OPTION_GUIDE_CODE_COUNT_KEY,
)
SUPPORTED_PROMPT_QUERY_KEYS: Tuple[str, ...] = SUPPORTED_ACTION_TARGET_QUERY_IDS + SUPPORTED_GUIDE_CODE_COUNT_QUERY_IDS
SUPPORTED_QUERY_IDS = SUPPORTED_ACTION_TARGET_QUERY_IDS
GUIDE_CODE_COUNT_CONTROL_FAMILY_BY_QUERY_ID: Dict[str, str] = {
    _CLICK_GUIDE_CODE_COUNT_KEY: _CLICK_PROMPT_KEY,
    _TYPE_FIELD_GUIDE_CODE_COUNT_KEY: _TYPE_FIELD_PROMPT_KEY,
    _SELECT_OPTION_GUIDE_CODE_COUNT_KEY: _SELECT_OPTION_PROMPT_KEY,
}
CONTROL_FAMILY_BY_QUERY_ID: Dict[str, str] = {
    **{str(value): str(value) for value in SUPPORTED_ACTION_TARGET_QUERY_IDS},
    **GUIDE_CODE_COUNT_CONTROL_FAMILY_BY_QUERY_ID,
}
SUPPORTED_WEB_SCENE_VARIANTS: Tuple[str, ...] = (
    "shop_catalog",
    "travel_booking",
    "support_center",
    "learning_portal",
    "finance_portal",
    "content_cms",
)
_BALANCE_SALT = 112573
_GUIDE_CODE_LABELS: Tuple[str, ...] = ("K1", "M2", "R3", "T4", "V5", "X6")
_GUIDE_CUE_LABELS: Tuple[str, ...] = (
    "next step",
    "follow-up",
    "priority route",
    "review flag",
    "saved setting",
    "quick change",
)

BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]


@dataclass(frozen=True)
class _TaskDefaults:
    canvas_width: int = 1280
    canvas_height: int = 800
    browser_margin_px: int = 34
    browser_bar_height_px: int = 58
    instruction_height_px: int = 60
    corner_radius_px: int = 14
    control_corner_radius_px: int = 8
    control_outline_width_px: int = 2
    badge_size_px: int = 24
    title_font_size_px: int = 24
    body_font_size_px: int = 16
    small_font_size_px: int = 13
    label_font_size_px: int = 16
    candidate_label_pool: Tuple[str, ...] = tuple(chr(ord("A") + idx) for idx in range(26))
    web_item_pool: Tuple[str, ...] = (
        "Trail Jacket",
        "Desk Lamp",
        "Noise Filter",
        "Canvas Tote",
        "Graph Notebook",
        "Travel Mug",
        "Studio Headset",
        "Cable Kit",
    )
    web_click_action_pool: Tuple[str, ...] = ("Details", "Compare", "Save", "Open")
    web_click_category_pool: Tuple[str, ...] = ("Audio", "Office", "Travel", "Home", "Outdoor", "Creative")
    web_click_status_pool: Tuple[str, ...] = ("Ready", "Backorder", "Featured", "Clearance", "Reserved", "Limited")
    web_section_pool: Tuple[str, ...] = ("Account", "Traveler", "Billing", "Delivery", "Notifications")
    web_field_pool: Tuple[str, ...] = ("Email", "Phone", "City", "Reference", "Notes")
    web_option_group_pool: Tuple[str, ...] = ("Delivery speed", "Plan type", "Seat zone", "Alert channel")
    web_option_pool: Tuple[str, ...] = ("Standard", "Priority", "Economy", "Window", "Monthly", "Email")


@dataclass(frozen=True)
class _RenderParams:
    canvas_width: int
    canvas_height: int
    browser_margin_px: int
    browser_bar_height_px: int
    instruction_height_px: int
    corner_radius_px: int
    control_corner_radius_px: int
    control_outline_width_px: int
    badge_size_px: int
    title_font_size_px: int
    body_font_size_px: int
    small_font_size_px: int
    label_font_size_px: int


@dataclass(frozen=True)
class _WebTheme:
    name: str
    page_fill: Color
    browser_fill: Color
    browser_line: Color
    chrome_fill: Color
    nav_fill: Color
    panel_fill: Color
    panel_alt_fill: Color
    control_fill: Color
    control_outline: Color
    text: Color
    muted_text: Color
    accent: Color
    accent_alt: Color
    instruction_fill: Color
    instruction_line: Color
    badge_fill: Color
    badge_text: Color


@dataclass(frozen=True)
class _WebProfile:
    site_name: str
    url_path: str
    page_title: str
    nav_items: Tuple[str, ...]
    status_text: str


@dataclass(frozen=True)
class _ControlSpec:
    control_id: str
    candidate_label: str
    role: str
    display_text: str
    context_label: str
    context_display_label: str
    context_attribute_1: str
    context_attribute_2: str
    action_label: str
    action_cue_label: str
    action_code_label: str
    support_id: str
    support_kind: str
    row_index: int
    col_index: int
    order_index: int


@dataclass(frozen=True)
class _GuideEntry:
    support_id: str
    support_kind: str
    cue_label: str
    code_label: str
    action_label: str
    col_index: int
    order_index: int


@dataclass(frozen=True)
class _ResolvedQuery:
    query_id: str
    control_family_key: str
    scene_variant: str
    controls: Tuple[_ControlSpec, ...]
    target_control_id: str
    target_label: str
    context_label: str
    action_label: str
    instruction_cue_label: str
    instruction_code_label: str
    instruction_text: str
    instruction_template_index: int
    guide_entries: Tuple[_GuideEntry, ...]
    instruction_support_id: str
    guide_support_id: str
    context_support_id: str
    context_support_kind: str
    candidate_label_pool: Tuple[str, ...]
    query_id_probabilities: Dict[str, float]
    scene_variant_probabilities: Dict[str, float]


@dataclass(frozen=True)
class _RenderedScene:
    control_bboxes_by_id: Dict[str, List[float]]
    badge_bboxes_by_id: Dict[str, List[float]]
    support_bboxes_by_id: Dict[str, List[float]]
    control_records: Tuple[Dict[str, Any], ...]
    support_records: Tuple[Dict[str, Any], ...]
    scene_bbox_px: List[float]
    browser_bbox_px: List[float]
    profile: _WebProfile
    theme: _WebTheme


_DEFAULTS = _TaskDefaults()
_TASK_GROUP_DEFAULTS = get_scene_defaults("pages", SCENE)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_generation_rendering_prompt_defaults(
    _TASK_GROUP_DEFAULTS if isinstance(_TASK_GROUP_DEFAULTS, Mapping) else {},
)
_VISUAL_DEFAULTS = _TASK_GROUP_DEFAULTS.get("visual", {}) if isinstance(_TASK_GROUP_DEFAULTS, Mapping) else {}
POST_IMAGE_NOISE_DEFAULTS = (
    dict(_VISUAL_DEFAULTS.get("noise", {})) if isinstance(_VISUAL_DEFAULTS.get("noise"), Mapping) else {}
)


_WEB_PROFILES: Dict[str, _WebProfile] = {
    "shop_catalog": _WebProfile(
        "MarketLane",
        "/catalog/deals",
        "Product Picks",
        ("Deals", "Orders", "Saved", "Help"),
        "8 items",
    ),
    "travel_booking": _WebProfile(
        "TripNest",
        "/book/stays",
        "Trip Planner",
        ("Flights", "Stays", "Cars", "Trips"),
        "Draft trip",
    ),
    "support_center": _WebProfile(
        "Assistly",
        "/support/tickets",
        "Support Queue",
        ("Inbox", "Customers", "Reports", "Macros"),
        "12 open",
    ),
    "learning_portal": _WebProfile(
        "CoursePad",
        "/learn/dashboard",
        "Learning Hub",
        ("Courses", "Calendar", "Grades", "Messages"),
        "3 due",
    ),
    "finance_portal": _WebProfile(
        "LedgerWay",
        "/payments/settings",
        "Payment Center",
        ("Cards", "Bills", "Transfers", "Settings"),
        "Secure",
    ),
    "content_cms": _WebProfile(
        "PublishKit",
        "/cms/articles",
        "Editorial Desk",
        ("Drafts", "Assets", "Review", "Publish"),
        "Autosaved",
    ),
}


def _web_theme_from_information_style(style: Any) -> _WebTheme:
    """Map the shared Pages information style into browser-page roles."""

    theme = _theme_from_information_style(style)
    return _WebTheme(
        name=str(theme.name),
        page_fill=tuple(style.canvas_rgb),
        browser_fill=tuple(theme.app_fill),
        browser_line=tuple(theme.chrome_line),
        chrome_fill=tuple(theme.panel_alt_fill),
        nav_fill=tuple(theme.panel_fill),
        panel_fill=tuple(theme.panel_fill),
        panel_alt_fill=tuple(theme.panel_alt_fill),
        control_fill=tuple(theme.control_fill),
        control_outline=tuple(theme.control_outline),
        text=tuple(theme.control_text),
        muted_text=tuple(theme.muted_text),
        accent=tuple(theme.accent),
        accent_alt=tuple(theme.accent_alt),
        instruction_fill=tuple(style.callout_fill_rgb),
        instruction_line=tuple(style.callout_border_rgb),
        badge_fill=tuple(theme.badge_fill),
        badge_text=tuple(theme.badge_text),
    )


def _decoupled_params(params: Mapping[str, Any], *, divisor: int, namespace: str) -> Mapping[str, Any]:
    _ = int(divisor), namespace
    return params


def _support_selection_index(params: Mapping[str, Any], *, instance_seed: int, namespace: str) -> int:
    return int(resolve_selection_index(params=params, instance_seed=int(instance_seed), namespace=f"{TASK_NAMESPACE}:{namespace}"))


def _resolve_named_axis(
    rng,
    *,
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
        gen_defaults=_GEN_DEFAULTS,
        supported_variants=supported,
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=supported,
        balance_flag_key=str(balance_flag_key),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        sampling_namespace=f"{TASK_NAMESPACE}:{namespace}",
    )
    return str(selected), dict(probabilities)


def _resolve_axis_bounds(
    params: Mapping[str, Any],
    *,
    min_key: str,
    max_key: str,
    fallback_min: int,
    fallback_max: int,
) -> Tuple[int, int]:
    min_value = int(params.get(str(min_key), group_default(_GEN_DEFAULTS, str(min_key), int(fallback_min))))
    max_value = int(params.get(str(max_key), group_default(_GEN_DEFAULTS, str(max_key), int(fallback_max))))
    if int(min_value) > int(max_value):
        raise ValueError(f"{min_key} must be <= {max_key} for {TASK_NAMESPACE}")
    return int(min_value), int(max_value)


def _resolve_count(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    min_key: str,
    max_key: str,
    fallback_min: int,
    fallback_max: int,
    namespace: str,
) -> int:
    min_value, max_value = _resolve_axis_bounds(
        params,
        min_key=str(min_key),
        max_key=str(max_key),
        fallback_min=int(fallback_min),
        fallback_max=int(fallback_max),
    )
    span = int(max_value) - int(min_value) + 1
    return int(min_value) + (_support_selection_index(params, instance_seed=int(instance_seed), namespace=str(namespace)) % max(1, span))


def _resolve_render_params(params: Mapping[str, Any], *, instance_seed: int) -> _RenderParams:
    values = asdict(_DEFAULTS)

    def _int_value(key: str) -> int:
        return int(
            resolve_render_int(
                params,
                _RENDER_DEFAULTS,
                str(key),
                int(values[str(key)]),
                instance_seed=instance_seed,
                namespace=TASK_NAMESPACE,
            )
        )

    return _RenderParams(
        canvas_width=_int_value("canvas_width"),
        canvas_height=_int_value("canvas_height"),
        browser_margin_px=_int_value("browser_margin_px"),
        browser_bar_height_px=_int_value("browser_bar_height_px"),
        instruction_height_px=_int_value("instruction_height_px"),
        corner_radius_px=_int_value("corner_radius_px"),
        control_corner_radius_px=_int_value("control_corner_radius_px"),
        control_outline_width_px=_int_value("control_outline_width_px"),
        badge_size_px=_int_value("badge_size_px"),
        title_font_size_px=_int_value("title_font_size_px"),
        body_font_size_px=_int_value("body_font_size_px"),
        small_font_size_px=_int_value("small_font_size_px"),
        label_font_size_px=_int_value("label_font_size_px"),
    )


def _sample_values(values: Sequence[str], *, count: int, rng) -> Tuple[str, ...]:
    support = [str(value) for value in values if str(value).strip()]
    if int(count) > len(support):
        raise ValueError(f"not enough GUI web support values: requested {count}, have {len(support)}")
    rng.shuffle(support)
    return tuple(support[: int(count)])


def _sample_click_attribute_pairs(params: Mapping[str, Any], *, count: int, rng) -> Tuple[Tuple[str, str], ...]:
    categories = _normalize_str_support(params, "web_click_category_pool", _DEFAULTS.web_click_category_pool)
    statuses = _normalize_str_support(params, "web_click_status_pool", _DEFAULTS.web_click_status_pool)
    pairs = [(str(category), str(status)) for category in categories for status in statuses]
    if int(count) > len(pairs):
        raise ValueError(f"not enough click attribute pairs: requested {count}, have {len(pairs)}")
    rng.shuffle(pairs)
    return tuple(pairs[: int(count)])


def _base_click_controls(params: Mapping[str, Any], *, instance_seed: int, rng) -> Tuple[_ControlSpec, ...]:
    """Build clickable item/action controls before guide codes and labels."""

    item_count = _resolve_count(
        params,
        instance_seed=int(instance_seed),
        min_key="web_click_item_count_min",
        max_key="web_click_item_count_max",
        fallback_min=4,
        fallback_max=6,
        namespace="click.item_count",
    )
    action_count = _resolve_count(
        params,
        instance_seed=int(instance_seed),
        min_key="web_click_action_count_min",
        max_key="web_click_action_count_max",
        fallback_min=3,
        fallback_max=4,
        namespace="click.action_count",
    )
    items = _sample_values(
        _normalize_str_support(params, "web_item_pool", _DEFAULTS.web_item_pool),
        count=int(item_count),
        rng=rng,
    )
    actions = _sample_values(
        _normalize_str_support(params, "web_click_action_pool", _DEFAULTS.web_click_action_pool),
        count=int(action_count),
        rng=rng,
    )
    attribute_pairs = _sample_click_attribute_pairs(params, count=int(item_count), rng=rng)
    controls: List[_ControlSpec] = []
    for row_index, item_label in enumerate(items):
        category, status = attribute_pairs[int(row_index)]
        context_label = f'category "{category}" and status "{status}"'
        visual_order = list(range(len(actions)))
        spawn_rng(int(instance_seed), f"{TASK_NAMESPACE}.click.button_order.{row_index}").shuffle(visual_order)
        visual_slot_by_col = {int(col_index): int(slot_index) for slot_index, col_index in enumerate(visual_order)}
        for col_index, action_label in enumerate(actions):
            controls.append(
                _ControlSpec(
                    control_id=f"click_{row_index}_{col_index}",
                    candidate_label="",
                    role="web_button",
                    display_text=str(action_label),
                    context_label=str(context_label),
                    context_display_label=str(item_label),
                    context_attribute_1=str(category),
                    context_attribute_2=str(status),
                    action_label=str(action_label),
                    action_cue_label="",
                    action_code_label="",
                    support_id=f"support_click_card_{row_index}",
                    support_kind="item_card",
                    row_index=int(row_index),
                    col_index=int(col_index),
                    order_index=int(visual_slot_by_col[int(col_index)]),
                )
            )
    return tuple(controls)


def _base_type_controls(params: Mapping[str, Any], *, instance_seed: int, rng) -> Tuple[_ControlSpec, ...]:
    """Build form-section/input controls before guide codes and labels."""

    section_count = _resolve_count(
        params,
        instance_seed=int(instance_seed),
        min_key="web_type_section_count_min",
        max_key="web_type_section_count_max",
        fallback_min=3,
        fallback_max=4,
        namespace="type.section_count",
    )
    field_count = _resolve_count(
        params,
        instance_seed=int(instance_seed),
        min_key="web_type_field_count_min",
        max_key="web_type_field_count_max",
        fallback_min=3,
        fallback_max=4,
        namespace="type.field_count",
    )
    sections = _sample_values(
        _normalize_str_support(params, "web_section_pool", _DEFAULTS.web_section_pool),
        count=int(section_count),
        rng=rng,
    )
    fields = _sample_values(
        _normalize_str_support(params, "web_field_pool", _DEFAULTS.web_field_pool),
        count=int(field_count),
        rng=rng,
    )
    controls: List[_ControlSpec] = []
    order_index = 0
    for row_index, section_label in enumerate(sections):
        for col_index, field_label in enumerate(fields):
            controls.append(
                _ControlSpec(
                    control_id=f"type_{row_index}_{col_index}",
                    candidate_label="",
                    role="web_input",
                    display_text=f"Enter {str(field_label).lower()}",
                    context_label=str(section_label),
                    context_display_label=str(section_label),
                    context_attribute_1="",
                    context_attribute_2="",
                    action_label=str(field_label),
                    action_cue_label="",
                    action_code_label="",
                    support_id=f"support_type_section_{row_index}",
                    support_kind="form_section",
                    row_index=int(row_index),
                    col_index=int(col_index),
                    order_index=int(order_index),
                )
            )
            order_index += 1
    return tuple(controls)


def _base_select_controls(params: Mapping[str, Any], *, instance_seed: int, rng) -> Tuple[_ControlSpec, ...]:
    """Build option-group controls before guide codes and labels."""

    group_count = _resolve_count(
        params,
        instance_seed=int(instance_seed),
        min_key="web_select_group_count_min",
        max_key="web_select_group_count_max",
        fallback_min=3,
        fallback_max=4,
        namespace="select.group_count",
    )
    option_count = _resolve_count(
        params,
        instance_seed=int(instance_seed),
        min_key="web_select_option_count_min",
        max_key="web_select_option_count_max",
        fallback_min=3,
        fallback_max=4,
        namespace="select.option_count",
    )
    groups = _sample_values(
        _normalize_str_support(params, "web_option_group_pool", _DEFAULTS.web_option_group_pool),
        count=int(group_count),
        rng=rng,
    )
    options = _sample_values(
        _normalize_str_support(params, "web_option_pool", _DEFAULTS.web_option_pool),
        count=int(option_count),
        rng=rng,
    )
    controls: List[_ControlSpec] = []
    order_index = 0
    for row_index, group_label in enumerate(groups):
        for col_index, option_label in enumerate(options):
            controls.append(
                _ControlSpec(
                    control_id=f"select_{row_index}_{col_index}",
                    candidate_label="",
                    role="web_option",
                    display_text=str(option_label),
                    context_label=str(group_label),
                    context_display_label=str(group_label),
                    context_attribute_1="",
                    context_attribute_2="",
                    action_label=str(option_label),
                    action_cue_label="",
                    action_code_label="",
                    support_id=f"support_select_group_{row_index}",
                    support_kind="option_group",
                    row_index=int(row_index),
                    col_index=int(col_index),
                    order_index=int(order_index),
                )
            )
            order_index += 1
    return tuple(controls)


def _instruction_for_target(
    *,
    control_family_key: str,
    target: _ControlSpec,
    instance_seed: int,
    params: Mapping[str, Any],
) -> Tuple[str, int]:
    """Compose the visible instruction from the selected control's guide code and context."""

    explicit = params.get("instruction_text")
    if explicit is not None:
        return str(explicit), 0
    if str(control_family_key) == _TYPE_FIELD_PROMPT_KEY:
        templates = (
            'In "{context}", use guide code "{code}" from the Field Guide',
            'Enter text for "{context}" using guide code "{code}"',
            'Use the input in "{context}" whose guide code is "{code}"',
        )
    elif str(control_family_key) == _SELECT_OPTION_PROMPT_KEY:
        templates = (
            'For "{context}", use guide code "{code}" from the Option Guide',
            'Choose the option for "{context}" with guide code "{code}"',
            'Set "{context}" according to guide code "{code}"',
        )
    else:
        templates = (
            'For the item with {context}, use guide code "{code}" from the Action Guide',
            'Click the control on the item with {context} and guide code "{code}"',
            'Use guide code "{code}" for the item with {context}',
        )
    index = _support_selection_index(params, instance_seed=int(instance_seed), namespace=f"instruction.{control_family_key}") % len(templates)
    return (
        str(templates[int(index)]).format(code=str(target.action_code_label), context=str(target.context_label)),
        int(index),
    )


def _guide_support_kind(prompt_key: str) -> str:
    if str(prompt_key) == _TYPE_FIELD_PROMPT_KEY:
        return "field_guide_card"
    if str(prompt_key) == _SELECT_OPTION_PROMPT_KEY:
        return "option_guide_card"
    return "action_guide_card"


def _coded_display_text(prompt_key: str, code_label: str) -> str:
    if str(prompt_key) == _TYPE_FIELD_PROMPT_KEY:
        return "Enter value"
    if str(prompt_key) == _SELECT_OPTION_PROMPT_KEY:
        return str(code_label)
    return str(code_label)


def _with_guide_codes(
    controls: Sequence[_ControlSpec],
    *,
    prompt_key: str,
    instance_seed: int,
) -> Tuple[Tuple[_ControlSpec, ...], Tuple[_GuideEntry, ...]]:
    """Assign shuffled guide cues/codes while preserving branch semantics."""

    columns = sorted({int(control.col_index) for control in controls})
    if len(columns) > min(len(_GUIDE_CODE_LABELS), len(_GUIDE_CUE_LABELS)):
        raise ValueError(f"not enough guide codes/cues for {TASK_NAMESPACE}")
    code_labels = list(_GUIDE_CODE_LABELS)
    cue_labels = list(_GUIDE_CUE_LABELS)
    spawn_rng(int(instance_seed), f"{TASK_NAMESPACE}.guide_codes.{prompt_key}").shuffle(code_labels)
    spawn_rng(int(instance_seed), f"{TASK_NAMESPACE}.guide_cues.{prompt_key}").shuffle(cue_labels)
    action_by_col = {
        int(col_index): str(next(control.action_label for control in controls if int(control.col_index) == int(col_index)))
        for col_index in columns
    }
    code_by_col = {int(col_index): str(code_labels[index]) for index, col_index in enumerate(columns)}
    cue_by_col = {int(col_index): str(cue_labels[index]) for index, col_index in enumerate(columns)}
    guide_entries = [
        _GuideEntry(
            support_id=f"support_guide_{int(col_index)}",
            support_kind=_guide_support_kind(str(prompt_key)),
            cue_label=str(cue_by_col[int(col_index)]),
            code_label=str(code_by_col[int(col_index)]),
            action_label=str(action_by_col[int(col_index)]),
            col_index=int(col_index),
            order_index=index,
        )
        for index, col_index in enumerate(columns)
    ]
    order = list(range(len(guide_entries)))
    spawn_rng(int(instance_seed), f"{TASK_NAMESPACE}.guide_order.{prompt_key}").shuffle(order)
    shuffled_entries = tuple(guide_entries[index] for index in order)
    coded_controls: List[_ControlSpec] = []
    for control in controls:
        col_index = int(control.col_index)
        code_label = str(code_by_col[col_index])
        cue_label = str(cue_by_col[col_index])
        coded_controls.append(
            _ControlSpec(
                control_id=str(control.control_id),
                candidate_label=str(control.candidate_label),
                role=str(control.role),
                display_text=_coded_display_text(str(prompt_key), code_label),
                context_label=str(control.context_label),
                context_display_label=str(control.context_display_label),
                context_attribute_1=str(control.context_attribute_1),
                context_attribute_2=str(control.context_attribute_2),
                action_label=str(control.action_label),
                action_cue_label=str(cue_label),
                action_code_label=str(code_label),
                support_id=str(control.support_id),
                support_kind=str(control.support_kind),
                row_index=int(control.row_index),
                col_index=int(control.col_index),
                order_index=int(control.order_index),
            )
        )
    return tuple(coded_controls), shuffled_entries


def _with_candidate_labels(
    controls: Sequence[_ControlSpec],
    *,
    target_control_id: str,
    target_label: str,
    candidate_label_pool: Sequence[str],
    instance_seed: int,
) -> Tuple[_ControlSpec, ...]:
    """Assign one target label and deterministic distractor labels to controls."""

    labels = [str(value) for value in candidate_label_pool]
    if str(target_label) not in labels:
        raise ValueError(f"target_label must be in candidate_label_pool for {TASK_NAMESPACE}")
    if len(controls) > len(labels):
        raise ValueError(f"candidate_label_pool has {len(labels)} labels for {len(controls)} controls in {TASK_NAMESPACE}")
    remaining = [label for label in labels if str(label) != str(target_label)]
    spawn_rng(int(instance_seed), f"{TASK_NAMESPACE}.candidate_labels").shuffle(remaining)
    out: List[_ControlSpec] = []
    next_index = 0
    for control in controls:
        label = str(target_label) if str(control.control_id) == str(target_control_id) else str(remaining[next_index])
        if str(control.control_id) != str(target_control_id):
            next_index += 1
        out.append(
            _ControlSpec(
                control_id=str(control.control_id),
                candidate_label=str(label),
                role=str(control.role),
                display_text=str(control.display_text),
                context_label=str(control.context_label),
                context_display_label=str(control.context_display_label),
                context_attribute_1=str(control.context_attribute_1),
                context_attribute_2=str(control.context_attribute_2),
                action_label=str(control.action_label),
                action_cue_label=str(control.action_cue_label),
                action_code_label=str(control.action_code_label),
                support_id=str(control.support_id),
                support_kind=str(control.support_kind),
                row_index=int(control.row_index),
                col_index=int(control.col_index),
                order_index=int(control.order_index),
            )
        )
    return tuple(out)


def _resolve_query(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    query_id: str,
    control_family_key: str,
    query_id_probabilities: Mapping[str, float],
) -> _ResolvedQuery:
    """Resolve one public task wrapper into a complete web-action scene trace."""

    rng = spawn_rng(int(instance_seed), f"{TASK_NAMESPACE}.query")
    public_query_id = str(query_id)
    if public_query_id not in SUPPORTED_PROMPT_QUERY_KEYS:
        raise ValueError(
            f"unsupported web_action query_id: {public_query_id}; "
            f"supported: {SUPPORTED_PROMPT_QUERY_KEYS}"
        )
    prompt_key = str(control_family_key)
    if prompt_key not in SUPPORTED_ACTION_TARGET_QUERY_IDS:
        raise ValueError(
            f"unsupported web_action control family: {prompt_key}; "
            f"supported: {SUPPORTED_ACTION_TARGET_QUERY_IDS}"
        )
    public_query_probabilities = {str(key): float(value) for key, value in dict(query_id_probabilities).items()}
    if not public_query_probabilities:
        public_query_probabilities = {public_query_id: 1.0}
    scene_variant, scene_variant_probabilities = _resolve_named_axis(
        rng,
        instance_seed=int(instance_seed),
        params=_decoupled_params(params, divisor=len(SUPPORTED_PROMPT_QUERY_KEYS), namespace="scene_variant"),
        supported=SUPPORTED_WEB_SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        namespace="scene_variant",
    )

    if str(prompt_key) == _TYPE_FIELD_PROMPT_KEY:
        base_controls = _base_type_controls(params, instance_seed=int(instance_seed), rng=rng)
    elif str(prompt_key) == _SELECT_OPTION_PROMPT_KEY:
        base_controls = _base_select_controls(params, instance_seed=int(instance_seed), rng=rng)
    else:
        base_controls = _base_click_controls(params, instance_seed=int(instance_seed), rng=rng)
    if not base_controls:
        raise ValueError(f"{TASK_NAMESPACE} generated no target controls")
    coded_controls, guide_entries = _with_guide_codes(
        base_controls,
        prompt_key=str(prompt_key),
        instance_seed=int(instance_seed),
    )

    target_index = _support_selection_index(params, instance_seed=int(instance_seed), namespace=f"target.{prompt_key}") % len(coded_controls)
    target_without_label = coded_controls[int(target_index)]
    candidate_label_pool = _normalize_str_support(params, "candidate_label_pool", _DEFAULTS.candidate_label_pool)
    target_label = str(
        params.get(
            "target_label",
            candidate_label_pool[
                _support_selection_index(params, instance_seed=int(instance_seed), namespace=f"answer_label.{prompt_key}")
                % len(candidate_label_pool)
            ],
        )
    )
    controls = _with_candidate_labels(
        coded_controls,
        target_control_id=str(target_without_label.control_id),
        target_label=str(target_label),
        candidate_label_pool=candidate_label_pool,
        instance_seed=int(instance_seed),
    )
    target = next(control for control in controls if str(control.control_id) == str(target_without_label.control_id))
    instruction_text, instruction_template_index = _instruction_for_target(
        control_family_key=str(prompt_key),
        target=target,
        instance_seed=int(instance_seed),
        params=params,
    )
    return _ResolvedQuery(
        query_id=str(public_query_id),
        control_family_key=str(prompt_key),
        scene_variant=str(scene_variant),
        controls=tuple(controls),
        target_control_id=str(target.control_id),
        target_label=str(target_label),
        context_label=str(target.context_label),
        action_label=str(target.action_label),
        instruction_cue_label=str(target.action_cue_label),
        instruction_code_label=str(target.action_code_label),
        instruction_text=str(instruction_text),
        instruction_template_index=int(instruction_template_index),
        guide_entries=tuple(guide_entries),
        instruction_support_id="support_instruction",
        guide_support_id=f"support_guide_{int(target.col_index)}",
        context_support_id=str(target.support_id),
        context_support_kind=str(target.support_kind),
        candidate_label_pool=tuple(str(value) for value in candidate_label_pool),
        query_id_probabilities=dict(public_query_probabilities),
        scene_variant_probabilities=dict(scene_variant_probabilities),
    )


def _add_support(
    support_bboxes: Dict[str, List[float]],
    support_records: List[Dict[str, Any]],
    support_id: str,
    support_kind: str,
    display_text: str,
    bbox: BBox,
    attrs: Mapping[str, Any] | None = None,
) -> None:
    if str(support_id) in support_bboxes:
        return
    support_bboxes[str(support_id)] = _bbox_list(bbox)
    record = {
        "support_id": str(support_id),
        "support_kind": str(support_kind),
        "display_text": str(display_text),
        "bbox_px": _bbox_list(bbox),
    }
    if attrs:
        record.update({str(key): value for key, value in attrs.items()})
    support_records.append(record)


def _draw_browser_frame(
    draw: ImageDraw.ImageDraw,
    *,
    query: _ResolvedQuery,
    render_params: _RenderParams,
    theme: _WebTheme,
) -> Tuple[BBox, _WebProfile, BBox]:
    """Draw browser chrome and return the inner content area used by page branches."""

    profile = _WEB_PROFILES[str(query.scene_variant)]
    m = float(render_params.browser_margin_px)
    width = float(render_params.canvas_width)
    height = float(render_params.canvas_height)
    browser = (m, m - 6.0, width - m, height - m + 6.0)
    _rounded_rect(draw, browser, radius=int(render_params.corner_radius_px), fill=theme.browser_fill, outline=theme.browser_line, width=2)
    bar_h = float(render_params.browser_bar_height_px)
    bar = (browser[0], browser[1], browser[2], browser[1] + bar_h)
    draw.rounded_rectangle([bar[0], bar[1], bar[2], bar[3] + int(render_params.corner_radius_px)], radius=int(render_params.corner_radius_px), fill=theme.chrome_fill)
    draw.rectangle([bar[0], bar[3] - int(render_params.corner_radius_px), bar[2], bar[3]], fill=theme.chrome_fill)
    for idx, fill in enumerate(((226, 78, 69), (236, 178, 67), (88, 176, 98))):
        draw.ellipse([bar[0] + 18.0 + idx * 22.0, bar[1] + 18.0, bar[0] + 30.0 + idx * 22.0, bar[1] + 30.0], fill=fill)
    address = (bar[0] + 104.0, bar[1] + 13.0, bar[2] - 266.0, bar[1] + 41.0)
    _rounded_rect(draw, address, radius=14, fill=(255, 255, 255), outline=theme.browser_line, width=1)
    _draw_text_left(
        draw,
        text=f"https://{profile.site_name.lower()}.example{profile.url_path}",
        bbox=(address[0] + 16.0, address[1] + 5.0, address[2] - 16.0, address[3] - 5.0),
        fill=theme.muted_text,
        max_size_px=int(render_params.small_font_size_px),
    )
    status = (bar[2] - 238.0, bar[1] + 13.0, bar[2] - 26.0, bar[1] + 41.0)
    _rounded_rect(draw, status, radius=14, fill=(255, 255, 255), outline=theme.browser_line, width=1)
    _draw_text_center_fit(
        draw,
        text=str(profile.status_text),
        bbox=(status[0] + 8.0, status[1] + 4.0, status[2] - 8.0, status[3] - 4.0),
        fill=theme.muted_text,
        max_size_px=int(render_params.small_font_size_px),
        bold=True,
    )

    page_header = (browser[0], bar[3], browser[2], bar[3] + 66.0)
    draw.rectangle([page_header[0], page_header[1], page_header[2], page_header[3]], fill=theme.page_fill)
    logo = (page_header[0] + 26.0, page_header[1] + 16.0, page_header[0] + 54.0, page_header[1] + 44.0)
    _rounded_rect(draw, logo, radius=8, fill=theme.accent, outline=None)
    draw_text_centered(
        draw,
        text=str(profile.site_name)[:1],
        center=((logo[0] + logo[2]) / 2.0, (logo[1] + logo[3]) / 2.0),
        font=load_font(int(render_params.small_font_size_px), bold=True),
        fill=(255, 255, 255),
    )
    draw_text_traced(draw,(page_header[0] + 66.0, page_header[1] + 14.0), str(profile.site_name), fill=theme.text, font=load_font(int(render_params.body_font_size_px), bold=True), role="readout", required=False)
    nav_x = page_header[0] + 238.0
    for idx, nav_label in enumerate(profile.nav_items):
        nav_w = 92.0 if len(str(nav_label)) <= 8 else 118.0
        nav_bbox = (nav_x, page_header[1] + 18.0, nav_x + nav_w, page_header[1] + 46.0)
        if idx == 0:
            _rounded_rect(draw, nav_bbox, radius=14, fill=theme.nav_fill, outline=theme.accent, width=1)
            fill = theme.text
        else:
            fill = theme.muted_text
        _draw_text_center_fit(
            draw,
            text=str(nav_label),
            bbox=(nav_bbox[0] + 8.0, nav_bbox[1] + 4.0, nav_bbox[2] - 8.0, nav_bbox[3] - 4.0),
            fill=fill,
            max_size_px=int(render_params.small_font_size_px),
            bold=True,
        )
        nav_x += nav_w + 8.0
    draw.line([browser[0], page_header[3], browser[2], page_header[3]], fill=theme.browser_line, width=1)
    content = (browser[0] + 24.0, page_header[3] + 18.0, browser[2] - 24.0, browser[3] - 20.0)
    return content, profile, browser


def _draw_instruction(
    draw: ImageDraw.ImageDraw,
    *,
    content_bbox: BBox,
    query: _ResolvedQuery,
    render_params: _RenderParams,
    theme: _WebTheme,
    support_bboxes: Dict[str, List[float]],
    support_records: List[Dict[str, Any]],
) -> BBox:
    x1, y1, x2, _y2 = [float(value) for value in content_bbox]
    instruction = (x1, y1, x2, y1 + float(render_params.instruction_height_px))
    _rounded_rect(draw, instruction, radius=12, fill=theme.instruction_fill, outline=theme.instruction_line, width=2)
    draw_text_traced(draw,(instruction[0] + 20.0, instruction[1] + 10.0), "Action instruction", fill=theme.muted_text, font=load_font(int(render_params.small_font_size_px), bold=True), role="readout", required=False)
    _draw_text_left(
        draw,
        text=str(query.instruction_text),
        bbox=(instruction[0] + 20.0, instruction[1] + 30.0, instruction[2] - 20.0, instruction[3] - 8.0),
        fill=theme.text,
        max_size_px=int(render_params.body_font_size_px),
        bold=True,
    )
    _add_support(
        support_bboxes,
        support_records,
        str(query.instruction_support_id),
        "instruction_banner",
        str(query.instruction_text),
        instruction,
    )
    return instruction


def _draw_action_guide(
    draw: ImageDraw.ImageDraw,
    *,
    content_bbox: BBox,
    top_y: float,
    query: _ResolvedQuery,
    render_params: _RenderParams,
    theme: _WebTheme,
    support_bboxes: Dict[str, List[float]],
    support_records: List[Dict[str, Any]],
) -> BBox:
    """Draw the cue guide; guide cards are annotation support witnesses."""

    x1, _y1, x2, _y2 = [float(value) for value in content_bbox]
    guide_h = 74.0
    guide = (x1, float(top_y), x2, float(top_y) + guide_h)
    _rounded_rect(draw, guide, radius=12, fill=theme.panel_alt_fill, outline=theme.browser_line, width=1)
    title = "Action Guide"
    if str(query.control_family_key) == _TYPE_FIELD_PROMPT_KEY:
        title = "Field Guide"
    elif str(query.control_family_key) == _SELECT_OPTION_PROMPT_KEY:
        title = "Option Guide"
    _draw_text_left(
        draw,
        text=title,
        bbox=(guide[0] + 18.0, guide[1] + 14.0, guide[0] + 178.0, guide[3] - 14.0),
        fill=theme.text,
        max_size_px=int(render_params.body_font_size_px),
        bold=True,
    )
    entries = tuple(query.guide_entries)
    if not entries:
        return guide
    gap = 10.0
    card_x1 = guide[0] + 190.0
    card_w = (guide[2] - card_x1 - 18.0 - gap * (len(entries) - 1)) / float(len(entries))
    for slot_index, entry in enumerate(entries):
        card = (
            card_x1 + slot_index * (card_w + gap),
            guide[1] + 12.0,
            card_x1 + slot_index * (card_w + gap) + card_w,
            guide[3] - 12.0,
        )
        _rounded_rect(draw, card, radius=8, fill=theme.control_fill, outline=theme.accent if slot_index % 2 == 0 else theme.accent_alt, width=2)
        _draw_text_center_fit(
            draw,
            text=str(entry.cue_label),
            bbox=(card[0] + 8.0, card[1] + 5.0, card[2] - 8.0, card[1] + 27.0),
            fill=theme.text,
            max_size_px=int(render_params.small_font_size_px),
            bold=True,
        )
        _draw_text_center_fit(
            draw,
            text=f"key {entry.code_label}",
            bbox=(card[0] + 8.0, card[1] + 28.0, card[2] - 8.0, card[3] - 5.0),
            fill=theme.muted_text,
            max_size_px=int(render_params.small_font_size_px),
        )
        _add_support(
            support_bboxes,
            support_records,
            str(entry.support_id),
            str(entry.support_kind),
            str(entry.cue_label),
            card,
            attrs={
                "cue_label": str(entry.cue_label),
                "code_label": str(entry.code_label),
                "action_label": str(entry.action_label),
                "col_index": int(entry.col_index),
                "guide_slot": int(slot_index),
            },
        )
    return guide


def _draw_candidate_badge(
    draw: ImageDraw.ImageDraw,
    *,
    control_bbox: BBox,
    label: str,
    render_params: _RenderParams,
    theme: _WebTheme,
) -> List[float]:
    x1, y1, x2, y2 = [float(value) for value in control_bbox]
    size = max(
        18,
        min(
            int(render_params.badge_size_px),
            int(min(float(x2 - x1), float(y2 - y1)) - 8.0),
        ),
    )
    badge = (x1 + 6.0, y1 + 6.0, x1 + 6.0 + float(size), y1 + 6.0 + float(size))
    _rounded_rect(draw, badge, radius=5, fill=theme.badge_fill, outline=(255, 255, 255), width=1)
    font = fit_font_to_box(
        draw,
        text=str(label),
        max_width=float(size) * 0.60,
        max_height=float(size) * 0.60,
        bold=False,
        min_size_px=8,
        max_size_px=int(render_params.label_font_size_px),
        fill_ratio=0.90,
    )
    draw_text_centered(
        draw,
        text=str(label),
        center=((badge[0] + badge[2]) / 2.0, (badge[1] + badge[3]) / 2.0),
        font=font,
        fill=theme.badge_text,
    )
    return _bbox_list(badge)


def _draw_control(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: BBox,
    control: _ControlSpec,
    render_params: _RenderParams,
    theme: _WebTheme,
) -> List[float]:
    role = str(control.role)
    if role == "web_button":
        fill = theme.control_fill
        text_fill = theme.text
        outline = theme.control_outline
        _rounded_rect(draw, bbox, radius=int(render_params.control_corner_radius_px), fill=fill, outline=outline, width=int(render_params.control_outline_width_px))
        text_bbox = (bbox[0] + 36.0, bbox[1] + 5.0, bbox[2] - 10.0, bbox[3] - 5.0)
    elif role == "web_option":
        _rounded_rect(draw, bbox, radius=18, fill=theme.control_fill, outline=theme.control_outline, width=int(render_params.control_outline_width_px))
        dot = (bbox[0] + 38.0, bbox[1] + 14.0, bbox[0] + 50.0, bbox[1] + 26.0)
        draw.ellipse([float(value) for value in dot], fill=theme.panel_alt_fill, outline=theme.accent, width=2)
        text_fill = theme.text
        text_bbox = (bbox[0] + 58.0, bbox[1] + 6.0, bbox[2] - 10.0, bbox[3] - 6.0)
    else:
        _rounded_rect(draw, bbox, radius=int(render_params.control_corner_radius_px), fill=theme.control_fill, outline=theme.control_outline, width=int(render_params.control_outline_width_px))
        text_fill = theme.muted_text
        text_bbox = (bbox[0] + 38.0, bbox[1] + 7.0, bbox[2] - 12.0, bbox[3] - 7.0)
    _draw_text_left(
        draw,
        text=str(control.display_text),
        bbox=text_bbox,
        fill=text_fill,
        max_size_px=int(render_params.small_font_size_px),
        bold=(role != "web_input"),
    )
    return _draw_candidate_badge(draw, control_bbox=bbox, label=str(control.candidate_label), render_params=render_params, theme=theme)


def _render_click_scene(
    draw: ImageDraw.ImageDraw,
    *,
    query: _ResolvedQuery,
    content_bbox: BBox,
    profile: _WebProfile,
    render_params: _RenderParams,
    theme: _WebTheme,
    control_bboxes: Dict[str, List[float]],
    badge_bboxes: Dict[str, List[float]],
    support_bboxes: Dict[str, List[float]],
    support_records: List[Dict[str, Any]],
) -> None:
    """Render the item-card branch where candidate controls are action buttons."""

    x1, y1, x2, y2 = [float(value) for value in content_bbox]
    instruction = _draw_instruction(draw, content_bbox=content_bbox, query=query, render_params=render_params, theme=theme, support_bboxes=support_bboxes, support_records=support_records)
    guide = _draw_action_guide(draw, content_bbox=content_bbox, top_y=instruction[3] + 10.0, query=query, render_params=render_params, theme=theme, support_bboxes=support_bboxes, support_records=support_records)
    work_y1 = guide[3] + 16.0
    draw_text_traced(draw,(x1 + 2.0, work_y1), str(profile.page_title), fill=theme.text, font=load_font(int(render_params.title_font_size_px), bold=True), role="readout", required=False)
    grid_y1 = work_y1 + 42.0
    controls_by_row: Dict[int, List[_ControlSpec]] = {}
    for control in query.controls:
        controls_by_row.setdefault(int(control.row_index), []).append(control)
    row_count = len(controls_by_row)
    col_count = 2
    gap = 16.0
    card_w = (x2 - x1 - gap * (col_count - 1)) / float(col_count)
    card_h = (y2 - grid_y1 - gap * (max(1, (row_count + 1) // 2) - 1)) / float(max(1, (row_count + 1) // 2))
    for row_index, controls in sorted(controls_by_row.items()):
        grid_row = row_index // col_count
        grid_col = row_index % col_count
        card = (
            x1 + grid_col * (card_w + gap),
            grid_y1 + grid_row * (card_h + gap),
            x1 + grid_col * (card_w + gap) + card_w,
            grid_y1 + grid_row * (card_h + gap) + card_h,
        )
        _rounded_rect(draw, card, radius=12, fill=theme.panel_fill, outline=theme.browser_line, width=1)
        stripe = (card[0], card[1], card[2], card[1] + 7.0)
        draw.rectangle([float(value) for value in stripe], fill=theme.accent_alt if row_index % 2 else theme.accent)
        title_bbox = (card[0] + 18.0, card[1] + 14.0, card[2] - 18.0, card[1] + 38.0)
        _draw_text_left(draw, text=str(controls[0].context_display_label), bbox=title_bbox, fill=theme.text, max_size_px=int(render_params.body_font_size_px), bold=True)
        meta = f"Category: {controls[0].context_attribute_1}   Status: {controls[0].context_attribute_2}"
        _draw_text_left(draw, text=meta, bbox=(card[0] + 18.0, card[1] + 40.0, card[2] - 18.0, card[1] + 62.0), fill=theme.muted_text, max_size_px=int(render_params.small_font_size_px))
        _add_support(
            support_bboxes,
            support_records,
            str(controls[0].support_id),
            str(controls[0].support_kind),
            str(controls[0].context_display_label),
            card,
            attrs={
                "context_label": str(controls[0].context_label),
                "context_display_label": str(controls[0].context_display_label),
                "category": str(controls[0].context_attribute_1),
                "status": str(controls[0].context_attribute_2),
                "row_index": int(row_index),
            },
        )
        button_gap = 10.0
        button_h = 34.0
        button_w = (card[2] - card[0] - 36.0 - button_gap * (len(controls) - 1)) / float(len(controls))
        by_visual_slot = sorted(controls, key=lambda value: int(value.order_index))
        for col_index, control in enumerate(by_visual_slot):
            button = (
                card[0] + 18.0 + col_index * (button_w + button_gap),
                card[3] - 14.0 - button_h,
                card[0] + 18.0 + col_index * (button_w + button_gap) + button_w,
                card[3] - 14.0,
            )
            badge_bboxes[str(control.control_id)] = _draw_control(draw, bbox=button, control=control, render_params=render_params, theme=theme)
            control_bboxes[str(control.control_id)] = _bbox_list(button)


def _render_type_scene(
    draw: ImageDraw.ImageDraw,
    *,
    query: _ResolvedQuery,
    content_bbox: BBox,
    profile: _WebProfile,
    render_params: _RenderParams,
    theme: _WebTheme,
    control_bboxes: Dict[str, List[float]],
    badge_bboxes: Dict[str, List[float]],
    support_bboxes: Dict[str, List[float]],
    support_records: List[Dict[str, Any]],
) -> None:
    """Render the form-section branch where candidate controls are input fields."""

    x1, y1, x2, y2 = [float(value) for value in content_bbox]
    instruction = _draw_instruction(draw, content_bbox=content_bbox, query=query, render_params=render_params, theme=theme, support_bboxes=support_bboxes, support_records=support_records)
    guide = _draw_action_guide(draw, content_bbox=content_bbox, top_y=instruction[3] + 10.0, query=query, render_params=render_params, theme=theme, support_bboxes=support_bboxes, support_records=support_records)
    work_y1 = guide[3] + 16.0
    draw_text_traced(draw,(x1 + 2.0, work_y1), str(profile.page_title), fill=theme.text, font=load_font(int(render_params.title_font_size_px), bold=True), role="readout", required=False)
    grid_y1 = work_y1 + 42.0
    controls_by_section: Dict[int, List[_ControlSpec]] = {}
    for control in query.controls:
        controls_by_section.setdefault(int(control.row_index), []).append(control)
    section_count = len(controls_by_section)
    panel_cols = 2
    gap = 16.0
    panel_w = (x2 - x1 - gap * (panel_cols - 1)) / float(panel_cols)
    panel_h = (y2 - grid_y1 - gap * (max(1, (section_count + 1) // 2) - 1)) / float(max(1, (section_count + 1) // 2))
    for section_index, controls in sorted(controls_by_section.items()):
        panel_row = section_index // panel_cols
        panel_col = section_index % panel_cols
        panel = (
            x1 + panel_col * (panel_w + gap),
            grid_y1 + panel_row * (panel_h + gap),
            x1 + panel_col * (panel_w + gap) + panel_w,
            grid_y1 + panel_row * (panel_h + gap) + panel_h,
        )
        _rounded_rect(draw, panel, radius=12, fill=theme.panel_fill, outline=theme.browser_line, width=1)
        _draw_text_left(draw, text=str(controls[0].context_label), bbox=(panel[0] + 16.0, panel[1] + 14.0, panel[2] - 16.0, panel[1] + 42.0), fill=theme.text, max_size_px=int(render_params.body_font_size_px), bold=True)
        _add_support(
            support_bboxes,
            support_records,
            str(controls[0].support_id),
            str(controls[0].support_kind),
            str(controls[0].context_label),
            panel,
            attrs={"context_label": str(controls[0].context_label), "row_index": int(section_index)},
        )
        sorted_controls = sorted(controls, key=lambda value: int(value.col_index))
        field_gap = 10.0
        field_h = min(58.0, (panel[3] - panel[1] - 58.0 - field_gap * (len(sorted_controls) - 1)) / float(len(sorted_controls)))
        for field_index, control in enumerate(sorted_controls):
            fy1 = panel[1] + 52.0 + field_index * (field_h + field_gap)
            label_bbox = (panel[0] + 18.0, fy1, panel[0] + 160.0, fy1 + field_h)
            input_bbox = (panel[0] + 172.0, fy1, panel[2] - 18.0, fy1 + field_h)
            _draw_text_left(draw, text=f"Key {control.action_code_label}", bbox=(label_bbox[0], label_bbox[1] + 6.0, label_bbox[2], label_bbox[3] - 6.0), fill=theme.text, max_size_px=int(render_params.small_font_size_px), bold=True)
            badge_bboxes[str(control.control_id)] = _draw_control(draw, bbox=input_bbox, control=control, render_params=render_params, theme=theme)
            control_bboxes[str(control.control_id)] = _bbox_list((label_bbox[0], label_bbox[1], input_bbox[2], input_bbox[3]))


def _render_select_scene(
    draw: ImageDraw.ImageDraw,
    *,
    query: _ResolvedQuery,
    content_bbox: BBox,
    profile: _WebProfile,
    render_params: _RenderParams,
    theme: _WebTheme,
    control_bboxes: Dict[str, List[float]],
    badge_bboxes: Dict[str, List[float]],
    support_bboxes: Dict[str, List[float]],
    support_records: List[Dict[str, Any]],
) -> None:
    """Render the option-group branch where candidate controls are selectable choices."""

    x1, y1, x2, y2 = [float(value) for value in content_bbox]
    instruction = _draw_instruction(draw, content_bbox=content_bbox, query=query, render_params=render_params, theme=theme, support_bboxes=support_bboxes, support_records=support_records)
    guide = _draw_action_guide(draw, content_bbox=content_bbox, top_y=instruction[3] + 10.0, query=query, render_params=render_params, theme=theme, support_bboxes=support_bboxes, support_records=support_records)
    work_y1 = guide[3] + 16.0
    draw_text_traced(draw,(x1 + 2.0, work_y1), str(profile.page_title), fill=theme.text, font=load_font(int(render_params.title_font_size_px), bold=True), role="readout", required=False)
    grid_y1 = work_y1 + 42.0
    controls_by_group: Dict[int, List[_ControlSpec]] = {}
    for control in query.controls:
        controls_by_group.setdefault(int(control.row_index), []).append(control)
    group_count = len(controls_by_group)
    gap = 14.0
    group_h = (y2 - grid_y1 - gap * (group_count - 1)) / float(max(1, group_count))
    for group_index, controls in sorted(controls_by_group.items()):
        group = (x1, grid_y1 + group_index * (group_h + gap), x2, grid_y1 + group_index * (group_h + gap) + group_h)
        _rounded_rect(draw, group, radius=12, fill=theme.panel_fill, outline=theme.browser_line, width=1)
        title_bbox = (group[0] + 18.0, group[1] + 14.0, group[0] + 240.0, group[3] - 14.0)
        _draw_text_left(draw, text=str(controls[0].context_label), bbox=(title_bbox[0], title_bbox[1], title_bbox[2], title_bbox[1] + 28.0), fill=theme.text, max_size_px=int(render_params.body_font_size_px), bold=True)
        _draw_text_left(draw, text="Choose one", bbox=(title_bbox[0], title_bbox[1] + 31.0, title_bbox[2], title_bbox[3]), fill=theme.muted_text, max_size_px=int(render_params.small_font_size_px))
        _add_support(
            support_bboxes,
            support_records,
            str(controls[0].support_id),
            str(controls[0].support_kind),
            str(controls[0].context_label),
            title_bbox,
            attrs={"context_label": str(controls[0].context_label), "row_index": int(group_index)},
        )
        sorted_controls = sorted(controls, key=lambda value: int(value.col_index))
        option_gap = 12.0
        option_w = (group[2] - group[0] - 280.0 - option_gap * (len(sorted_controls) - 1)) / float(len(sorted_controls))
        option_h = min(46.0, group_h - 28.0)
        for option_index, control in enumerate(sorted_controls):
            option = (
                group[0] + 260.0 + option_index * (option_w + option_gap),
                group[1] + (group_h - option_h) / 2.0,
                group[0] + 260.0 + option_index * (option_w + option_gap) + option_w,
                group[1] + (group_h + option_h) / 2.0,
            )
            badge_bboxes[str(control.control_id)] = _draw_control(draw, bbox=option, control=control, render_params=render_params, theme=theme)
            control_bboxes[str(control.control_id)] = _bbox_list(option)


def _render_web_scene(
    image: Image.Image,
    *,
    query: _ResolvedQuery,
    render_params: _RenderParams,
    theme: _WebTheme,
) -> _RenderedScene:
    """Render one web-action page and collect bboxes from the same trace."""

    draw = ImageDraw.Draw(image)
    content_bbox, profile, browser_bbox = _draw_browser_frame(draw, query=query, render_params=render_params, theme=theme)
    control_bboxes: Dict[str, List[float]] = {}
    badge_bboxes: Dict[str, List[float]] = {}
    support_bboxes: Dict[str, List[float]] = {}
    support_records: List[Dict[str, Any]] = []
    if str(query.control_family_key) == _TYPE_FIELD_PROMPT_KEY:
        _render_type_scene(
            draw,
            query=query,
            content_bbox=content_bbox,
            profile=profile,
            render_params=render_params,
            theme=theme,
            control_bboxes=control_bboxes,
            badge_bboxes=badge_bboxes,
            support_bboxes=support_bboxes,
            support_records=support_records,
        )
    elif str(query.control_family_key) == _SELECT_OPTION_PROMPT_KEY:
        _render_select_scene(
            draw,
            query=query,
            content_bbox=content_bbox,
            profile=profile,
            render_params=render_params,
            theme=theme,
            control_bboxes=control_bboxes,
            badge_bboxes=badge_bboxes,
            support_bboxes=support_bboxes,
            support_records=support_records,
        )
    else:
        _render_click_scene(
            draw,
            query=query,
            content_bbox=content_bbox,
            profile=profile,
            render_params=render_params,
            theme=theme,
            control_bboxes=control_bboxes,
            badge_bboxes=badge_bboxes,
            support_bboxes=support_bboxes,
            support_records=support_records,
        )

    control_records: List[Dict[str, Any]] = []
    for control in query.controls:
        control_records.append(
            {
                "control_id": str(control.control_id),
                "candidate_label": str(control.candidate_label),
                "role": str(control.role),
                "display_text": str(control.display_text),
                "context_label": str(control.context_label),
                "context_display_label": str(control.context_display_label),
                "context_attribute_1": str(control.context_attribute_1),
                "context_attribute_2": str(control.context_attribute_2),
                "action_label": str(control.action_label),
                "action_cue_label": str(control.action_cue_label),
                "action_code_label": str(control.action_code_label),
                "support_id": str(control.support_id),
                "support_kind": str(control.support_kind),
                "row_index": int(control.row_index),
                "col_index": int(control.col_index),
                "order_index": int(control.order_index),
                "bbox_px": list(control_bboxes[str(control.control_id)]),
                "candidate_label_bbox_px": list(badge_bboxes[str(control.control_id)]),
            }
        )
    return _RenderedScene(
        control_bboxes_by_id={str(key): list(value) for key, value in control_bboxes.items()},
        badge_bboxes_by_id={str(key): list(value) for key, value in badge_bboxes.items()},
        support_bboxes_by_id={str(key): list(value) for key, value in support_bboxes.items()},
        control_records=tuple(control_records),
        support_records=tuple(dict(record) for record in support_records),
        scene_bbox_px=[0.0, 0.0, float(render_params.canvas_width), float(render_params.canvas_height)],
        browser_bbox_px=_bbox_list(browser_bbox),
        profile=profile,
        theme=theme,
    )


def _target_annotation_role_for_query(prompt_key: str) -> str:
    """Return the trace-facing role name for the answer-bearing control."""

    if str(prompt_key) == _CLICK_PROMPT_KEY:
        return "target_button"
    if str(prompt_key) == _TYPE_FIELD_PROMPT_KEY:
        return "target_input"
    return "target_option"


def _bbox_union(*boxes: Sequence[float]) -> List[float]:
    """Return the smallest bbox covering all provided pixel boxes."""

    if not boxes:
        raise ValueError("at least one bbox is required")
    normalized = [[float(value) for value in box] for box in boxes]
    return _bbox_list(
        (
            min(box[0] for box in normalized),
            min(box[1] for box in normalized),
            max(box[2] for box in normalized),
            max(box[3] for box in normalized),
        )
    )


def _prompt_json_examples(*, answer_mode: str) -> Tuple[str, str]:
    if str(answer_mode) == "guide_code_count":
        answer_and_annotation = {
            "annotation": [[410, 378, 584, 442], [410, 452, 584, 516], [410, 526, 584, 590]],
            "answer": 3,
        }
        answer_only = {"answer": 3}
    else:
        answer_and_annotation = {
            "annotation": [410, 378, 584, 442],
            "answer": "G",
        }
        answer_only = {"answer": "G"}
    return (
        json.dumps(answer_and_annotation, ensure_ascii=True, allow_nan=False, separators=(",", ":")),
        json.dumps(answer_only, ensure_ascii=True, allow_nan=False, separators=(",", ":")),
    )


def _web_profile_dict(profile: _WebProfile) -> Dict[str, Any]:
    return {
        "site_name": str(profile.site_name),
        "url_path": str(profile.url_path),
        "page_title": str(profile.page_title),
        "nav_items": [str(value) for value in profile.nav_items],
        "status_text": str(profile.status_text),
    }




class PagesWebActionSceneTask:
    """Build web-action page instances for public task wrappers."""

    domain = "pages"
    scene_id = SCENE

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one instance; answer and annotation come from the resolved trace."""

        del max_attempts
        public_query_id = str(params.get("_query_id", "")).strip()
        if not public_query_id:
            public_query_id = str(params.get("_prompt_query_key", "")).strip()
        if not public_query_id:
            raise ValueError("_query_id is required for web_action lifecycle generation")
        control_family_key = str(params.get("_control_family_key", CONTROL_FAMILY_BY_QUERY_ID.get(public_query_id, ""))).strip()
        if not control_family_key:
            raise ValueError("_control_family_key is required for web_action lifecycle generation")
        question_format = str(params.get("_question_format", "web_action_query"))
        answer_mode = str(params.get("_answer_mode", "selection"))
        query_id_probabilities = params.get("_query_id_probabilities", {public_query_id: 1.0})
        if not isinstance(query_id_probabilities, Mapping):
            raise ValueError("_query_id_probabilities must be a mapping for web_action lifecycle generation")
        query = _resolve_query(
            int(instance_seed),
            params=params,
            query_id=str(public_query_id),
            control_family_key=str(control_family_key),
            query_id_probabilities=query_id_probabilities,
        )
        render_params = _resolve_render_params(params, instance_seed=int(instance_seed))
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
            namespace=f"{TASK_NAMESPACE}.background",
        )
        theme = _web_theme_from_information_style(information_style)
        image = background.copy().convert("RGB")
        rendered = _render_web_scene(image, query=query, render_params=render_params, theme=theme)
        image, post_noise_meta = apply_post_image_noise(
            image,
            instance_seed=int(instance_seed),
            params=params,
            default_config=POST_IMAGE_NOISE_DEFAULTS,
        )

        target_control_bbox = list(rendered.control_bboxes_by_id[str(query.target_control_id)])
        target_badge_bbox = list(rendered.badge_bboxes_by_id[str(query.target_control_id)])
        target_annotation_bbox = _bbox_union(target_control_bbox, target_badge_bbox)
        reasoning_support_ids = (
            str(query.instruction_support_id),
            str(query.guide_support_id),
            str(query.context_support_id),
        )
        target_annotation_role = _target_annotation_role_for_query(str(query.control_family_key))

        control_records = [dict(record) for record in rendered.control_records]
        support_records = [dict(record) for record in rendered.support_records]
        target_record = next(record for record in control_records if str(record["control_id"]) == str(query.target_control_id))
        reasoning_support_records = [
            next(record for record in support_records if str(record["support_id"]) == str(support_id))
            for support_id in reasoning_support_ids
        ]

        matching_control_ids: List[str] = []
        matching_annotation_bboxes: List[List[float]] = []
        if answer_mode == "guide_code_count":
            guide_code = str(query.instruction_code_label)
            matching_control_ids = [
                str(record["control_id"])
                for record in control_records
                if str(record["action_code_label"]) == guide_code
            ]
            matching_annotation_bboxes = [
                _bbox_union(
                    rendered.control_bboxes_by_id[str(control_id)],
                    rendered.badge_bboxes_by_id[str(control_id)],
                )
                for control_id in matching_control_ids
            ]
            answer_gt = TypedValue(type="integer", value=int(len(matching_control_ids)))
            annotation_gt = TypedValue(type="bbox_set", value=[list(bbox) for bbox in matching_annotation_bboxes])
        elif answer_mode == "selection":
            guide_code = str(query.instruction_code_label)
            answer_gt = TypedValue(type="option_letter", value=str(query.target_label))
            annotation_gt = TypedValue(type="bbox", value=list(target_annotation_bbox))
        else:
            raise ValueError(f"unsupported web_action answer mode: {answer_mode}")

        prompt_defaults = required_group_defaults(
            _PROMPT_DEFAULTS,
            ("bundle_id",),
            context=f"prompt defaults for {getattr(self, 'task_id', TASK_NAMESPACE)}",
        )
        prompt_selection = render_task_prompt_variants(
            domain=self.domain,
            scene_id=self.scene_id,
            bundle_id=str(prompt_defaults["bundle_id"]),
            scene_key=PROMPT_SCENE_KEY,
            task_key=PROMPT_TASK_KEY,
            query_key=str(query.query_id),
            answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
            dynamic_slots={
                "instruction_text": str(query.instruction_text),
                "context_label": str(query.context_label),
                "action_label": str(query.action_label),
                "guide_code": str(query.instruction_code_label),
            },
            instance_seed=int(instance_seed),
        )
        prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)

        query_params = {
            "query_id": str(query.query_id),
            "prompt_query_key": str(query.query_id),
            "source_query_id": str(query.query_id),
            "control_family_key": str(query.control_family_key),
            "answer_mode": str(answer_mode),
            "scene_variant": str(query.scene_variant),
            "target_control_id": str(query.target_control_id),
            "target_label": str(query.target_label),
            "context_label": str(query.context_label),
            "context_display_label": str(target_record["context_display_label"]),
            "context_attribute_1": str(target_record["context_attribute_1"]),
            "context_attribute_2": str(target_record["context_attribute_2"]),
            "action_label": str(query.action_label),
            "instruction_cue_label": str(query.instruction_cue_label),
            "instruction_code_label": str(query.instruction_code_label),
            "instruction_text": str(query.instruction_text),
            "instruction_template_index": int(query.instruction_template_index),
            "instruction_support_id": str(query.instruction_support_id),
            "guide_support_id": str(query.guide_support_id),
            "guide_entries": [asdict(entry) for entry in query.guide_entries],
            "context_support_id": str(query.context_support_id),
            "context_support_kind": str(query.context_support_kind),
            "candidate_label_pool": [str(value) for value in query.candidate_label_pool],
            "target_annotation_role": str(target_annotation_role),
            "target_annotation_control_id": str(query.target_control_id),
            "guide_code": str(guide_code),
            "matching_control_ids": [str(value) for value in matching_control_ids],
            "matching_control_count": int(len(matching_control_ids)),
            "query_id_probabilities": dict(query.query_id_probabilities),
            "prompt_query_key_probabilities": dict(query.query_id_probabilities),
            "scene_variant_probabilities": dict(query.scene_variant_probabilities),
        }
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(query.query_id),
            params=dict(query_params),
        )
        query_spec["scene_id"] = SCENE
        scene_kind = "gui_web_action_guide_code_count" if answer_mode == "guide_code_count" else "gui_web_action_target"
        if answer_mode == "guide_code_count":
            witness_symbolic = {
                "type": "bbox_set",
                "guide_code": str(guide_code),
                "matching_control_ids": [str(value) for value in matching_control_ids],
                "value": [list(bbox) for bbox in matching_annotation_bboxes],
            }
            projected_annotation = {
                "type": "bbox_set",
                "bbox_set": [list(bbox) for bbox in matching_annotation_bboxes],
                "pixel_bbox_set": [list(bbox) for bbox in matching_annotation_bboxes],
            }
        else:
            witness_symbolic = {
                "type": "bbox",
                "reasoning_support_ids": [str(value) for value in reasoning_support_ids],
                "target_annotation_role": str(target_annotation_role),
                "target_control_id": str(query.target_control_id),
                "value": list(target_annotation_bbox),
            }
            projected_annotation = {
                "type": "bbox",
                "bbox": list(target_annotation_bbox),
                "pixel_bbox": list(target_annotation_bbox),
            }
        trace_payload = {
            "scene_ir": {
                "scene_id": SCENE,
                "scene_kind": str(scene_kind),
                "entities": [
                    {
                        "entity_id": str(record["control_id"]),
                        "entity_type": "web_control",
                        "attrs": {
                            "candidate_label": str(record["candidate_label"]),
                            "role": str(record["role"]),
                            "display_text": str(record["display_text"]),
                            "context_label": str(record["context_label"]),
                            "context_display_label": str(record["context_display_label"]),
                            "context_attribute_1": str(record["context_attribute_1"]),
                            "context_attribute_2": str(record["context_attribute_2"]),
                            "action_label": str(record["action_label"]),
                            "action_cue_label": str(record["action_cue_label"]),
                            "action_code_label": str(record["action_code_label"]),
                            "bbox_px": list(record["bbox_px"]),
                        },
                    }
                    for record in control_records
                ],
                "relations": {
                    "query_id": str(query.query_id),
                    "prompt_query_key": str(query.query_id),
                    "source_query_id": str(query.query_id),
                    "control_family_key": str(query.control_family_key),
                    "answer_mode": str(answer_mode),
                    "scene_variant": str(query.scene_variant),
                    "target_control_id": str(query.target_control_id),
                    "target_label": str(query.target_label),
                    "context_label": str(query.context_label),
                    "context_display_label": str(target_record["context_display_label"]),
                    "context_attribute_1": str(target_record["context_attribute_1"]),
                    "context_attribute_2": str(target_record["context_attribute_2"]),
                    "action_label": str(query.action_label),
                    "instruction_cue_label": str(query.instruction_cue_label),
                    "instruction_code_label": str(query.instruction_code_label),
                    "instruction_text": str(query.instruction_text),
                    "instruction_template_index": int(query.instruction_template_index),
                    "instruction_support_id": str(query.instruction_support_id),
                    "guide_support_id": str(query.guide_support_id),
                    "guide_entries": [asdict(entry) for entry in query.guide_entries],
                    "context_support_id": str(query.context_support_id),
                    "context_support_kind": str(query.context_support_kind),
                    "reasoning_support_ids": [str(value) for value in reasoning_support_ids],
                    "target_annotation_role": str(target_annotation_role),
                    "target_annotation_control_id": str(query.target_control_id),
                    "guide_code": str(guide_code),
                    "matching_control_ids": [str(value) for value in matching_control_ids],
                    "matching_control_count": int(len(matching_control_ids)),
                },
                "frames": {
                    "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
                },
            },
            "query_spec": query_spec,
            "render_spec": {
                "scene_id": SCENE,
                "query_id": str(query.query_id),
                "prompt_query_key": str(query.query_id),
                "control_family_key": str(query.control_family_key),
                "canvas_width": int(render_params.canvas_width),
                "canvas_height": int(render_params.canvas_height),
                "coord_space": "pixel",
                "background_style": dict(background_meta),
                "information_scene_style": dict(information_style_meta),
                "pages_information_style_policy": {
                    "domain_wrapper": "scene_renderer",
                    "scene_renderer_recorded_style": True,
                    "task_id": str(getattr(self, "task_id", TASK_NAMESPACE)),
                    "scene_id": SCENE,
                },
                "post_image_noise": dict(post_noise_meta),
                "scene_variant": str(query.scene_variant),
                "browser_bbox_px": list(rendered.browser_bbox_px),
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
                "browser_bbox_px": list(rendered.browser_bbox_px),
                "web_profile": _web_profile_dict(rendered.profile),
                "control_bboxes_by_id": dict(rendered.control_bboxes_by_id),
                "candidate_label_badge_bboxes_by_id": dict(rendered.badge_bboxes_by_id),
                "support_bboxes_by_id": dict(rendered.support_bboxes_by_id),
                "target_control_id": str(query.target_control_id),
                "guide_entries": [asdict(entry) for entry in query.guide_entries],
                "reasoning_support_ids": [str(value) for value in reasoning_support_ids],
                "target_annotation_role": str(target_annotation_role),
                "target_annotation_bbox": list(target_annotation_bbox),
                "matching_control_ids": [str(value) for value in matching_control_ids],
                "matching_annotation_bboxes": [list(bbox) for bbox in matching_annotation_bboxes],
            },
            "execution_trace": {
                **dict(query_params),
                "reasoning_support_ids": [str(value) for value in reasoning_support_ids],
                "reasoning_support_records": [dict(record) for record in reasoning_support_records],
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
            query_id=str(query.query_id),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
        )


def build_web_action_response(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    task_id: str,
    prompt_query_key: str,
    question_format: str,
    public_query_id: str | None = None,
    query_id_probabilities: Mapping[str, float] | None = None,
    control_family_key: str | None = None,
    answer_mode: str = "selection",
) -> TaskOutput:
    task = PagesWebActionSceneTask()
    task.task_id = str(task_id)
    lifecycle_params = dict(params)
    resolved_public_query_id = str(public_query_id or prompt_query_key)
    resolved_control_family_key = str(control_family_key or CONTROL_FAMILY_BY_QUERY_ID.get(resolved_public_query_id, prompt_query_key))
    lifecycle_params["_prompt_query_key"] = str(prompt_query_key)
    lifecycle_params["_query_id"] = str(resolved_public_query_id)
    lifecycle_params["_query_id_probabilities"] = dict(query_id_probabilities or {str(resolved_public_query_id): 1.0})
    lifecycle_params["_control_family_key"] = str(resolved_control_family_key)
    lifecycle_params["_answer_mode"] = str(answer_mode)
    lifecycle_params["_question_format"] = str(question_format)
    return task.generate(
        int(instance_seed),
        params=lifecycle_params,
        max_attempts=1,
    )


__all__ = [
    "PagesWebActionSceneTask",
    "CONTROL_FAMILY_BY_QUERY_ID",
    "GUIDE_CODE_COUNT_CONTROL_FAMILY_BY_QUERY_ID",
    "SUPPORTED_ACTION_TARGET_QUERY_IDS",
    "SUPPORTED_GUIDE_CODE_COUNT_QUERY_IDS",
    "SUPPORTED_PROMPT_QUERY_KEYS",
    "SUPPORTED_QUERY_IDS",
    "build_web_action_response",
]
