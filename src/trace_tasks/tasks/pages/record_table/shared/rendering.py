"""Rendering and neutral sampling primitives for record-table pages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.seed import hash64, spawn_rng
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.visual.background import make_background_canvas
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.pages.shared.gui_chrome import (
    bbox_list as _bbox_list,
    clamp_unit as _clamp_unit,
    draw_text_center_fit as _draw_text_center_fit,
    draw_text_left as _draw_text_left,
    rounded_rect as _rounded_rect,
)
from trace_tasks.tasks.shared.config_defaults import group_default, split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.deterministic_sampling import resolve_selection_index
from trace_tasks.tasks.shared.render_variation import resolve_render_int
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import load_font
from trace_tasks.tasks.shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant


NAMESPACE_ROOT = "pages.record_table"
FILTER_KEYS: Tuple[str, ...] = (
    "selected_status_filter",
    "enabled_type_action_filter",
    "section_size_threshold_filter",
)
SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = (
    "office_document",
    "creative_workspace",
    "developer_ide",
    "cad_workspace",
    "scientific_plotter",
    "os_file_manager",
)
SUPPORTED_STYLE_VARIANTS: Tuple[str, ...] = ("standard", "compact", "contrast", "cool", "warm", "sage")
_BALANCE_SALT = 93751

BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]


@dataclass(frozen=True)
class RecordTableDefaults:
    canvas_width: int = 1280
    canvas_height: int = 800
    window_margin_px: int = 42
    title_bar_height_px: int = 46
    menu_bar_height_px: int = 34
    corner_radius_px: int = 16
    table_corner_radius_px: int = 10
    row_height_px: int = 28
    section_header_height_px: int = 25
    table_header_height_px: int = 32
    title_font_size_px: int = 24
    body_font_size_px: int = 16
    small_font_size_px: int = 13
    row_count_support: Tuple[int, ...] = (9, 10, 11, 12, 13, 14, 15)
    section_count_support: Tuple[int, ...] = (2, 3)
    answer_count_support: Tuple[int, ...] = (2, 3, 4, 5, 6, 7)
    section_name_pool: Tuple[str, ...] = ("Inbox", "Assets", "Backlog")
    type_label_pool: Tuple[str, ...] = ("Image", "Document", "Dataset", "Report")
    status_label_pool: Tuple[str, ...] = ("Ready", "Warning", "Queued", "Blocked")
    action_label_pool: Tuple[str, ...] = ("Sync", "Export", "Archive", "Review")
    size_threshold_support: Tuple[int, ...] = (25, 35, 45, 55, 65)


@dataclass(frozen=True)
class RecordTableRenderParams:
    canvas_width: int
    canvas_height: int
    window_margin_px: int
    title_bar_height_px: int
    menu_bar_height_px: int
    corner_radius_px: int
    table_corner_radius_px: int
    row_height_px: int
    section_header_height_px: int
    table_header_height_px: int
    title_font_size_px: int
    body_font_size_px: int
    small_font_size_px: int


@dataclass(frozen=True)
class RecordTableTheme:
    name: str
    app_fill: Color
    title_bar: Color
    title_text: Color
    chrome_line: Color
    panel_fill: Color
    panel_alt_fill: Color
    row_fill: Color
    row_alt_fill: Color
    text: Color
    muted_text: Color
    disabled_fill: Color
    disabled_outline: Color
    accent: Color
    accent_alt: Color
    success_fill: Color
    warning_fill: Color
    blocked_fill: Color


@dataclass(frozen=True)
class RecordTableProfile:
    app_title: str
    window_title: str
    primary_tab: str
    secondary_tab: str
    workspace_title: str
    status_text: str


@dataclass(frozen=True)
class RecordTableRow:
    row_id: str
    row_label: str
    section_name: str
    section_index: int
    order_in_section: int
    global_order_index: int
    item_name: str
    type_label: str
    status_label: str
    size_mb: int
    selected: bool
    action_label: str
    action_enabled: bool


@dataclass(frozen=True)
class RecordTableCase:
    filter_key: str
    scene_variant: str
    style_variant: str
    rows: Tuple[RecordTableRow, ...]
    section_names: Tuple[str, ...]
    target_status: str
    target_type: str
    target_action_label: str
    target_section_name: str
    target_section_index: int
    size_threshold_mb: int
    answer_value: int
    annotation_row_ids: Tuple[str, ...]
    row_count_support: Tuple[int, ...]
    answer_count_support: Tuple[int, ...]
    section_count_support: Tuple[int, ...]
    filter_key_probabilities: Dict[str, float]
    scene_variant_probabilities: Dict[str, float]
    style_variant_probabilities: Dict[str, float]


@dataclass(frozen=True)
class RenderedRecordTable:
    row_bboxes_by_id: Dict[str, List[float]]
    section_bboxes_by_name: Dict[str, List[float]]
    cell_bboxes_by_row_id: Dict[str, Dict[str, List[float]]]
    row_records: Tuple[Dict[str, Any], ...]
    scene_bbox_px: List[float]
    window_bbox_px: List[float]
    profile: RecordTableProfile
    theme: RecordTableTheme


@dataclass(frozen=True)
class RenderedRecordTableBundle:
    """Rendered record-table image and pixel-space metadata."""

    image: Image.Image
    render_params: RecordTableRenderParams
    rendered_table: RenderedRecordTable
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]


_DEFAULTS = RecordTableDefaults()
_SCENE_DEFAULTS = get_scene_defaults("pages", "record_table")
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
)
_VISUAL_DEFAULTS = _SCENE_DEFAULTS.get("visual", {}) if isinstance(_SCENE_DEFAULTS, Mapping) else {}
POST_IMAGE_BACKGROUND_DEFAULTS = (
    dict(_VISUAL_DEFAULTS.get("background", {})) if isinstance(_VISUAL_DEFAULTS.get("background"), Mapping) else {}
)
POST_IMAGE_NOISE_DEFAULTS = (
    dict(_VISUAL_DEFAULTS.get("noise", {})) if isinstance(_VISUAL_DEFAULTS.get("noise"), Mapping) else {}
)

_APP_PROFILES: Dict[str, RecordTableProfile] = {
    "office_document": RecordTableProfile("Document Studio", "Review Queue", "Home", "Review", "Document Table", "Rows synced"),
    "creative_workspace": RecordTableProfile("Canvas Lab", "Asset Catalog", "Design", "Assets", "Asset Table", "RGB / 100%"),
    "developer_ide": RecordTableProfile("Code Desk", "Issue Board", "Build", "Debug", "Issue Table", "main / clean"),
    "cad_workspace": RecordTableProfile("Model Works", "Part Library", "Sketch", "Inspect", "Parts Table", "Units: mm"),
    "scientific_plotter": RecordTableProfile("Lab Plot", "Run Registry", "Analyze", "Plot", "Sample Table", "Sample 2.4k"),
    "os_file_manager": RecordTableProfile("File Center", "Shared Folder", "Files", "View", "File Table", "23 items"),
}

_ITEM_STEMS: Tuple[str, ...] = (
    "Atlas",
    "Beacon",
    "Canvas",
    "Delta",
    "Echo",
    "Flux",
    "Grid",
    "Helix",
    "Ion",
    "Juno",
    "Keystone",
    "Lumen",
    "Matrix",
    "Nova",
    "Orbit",
)


def _theme(style_variant: str) -> RecordTableTheme:
    """Resolve a table palette while keeping row-state colors consistent."""

    if str(style_variant) == "cool":
        return RecordTableTheme(
            name="cool",
            app_fill=(253, 254, 255),
            title_bar=(49, 80, 112),
            title_text=(255, 255, 255),
            chrome_line=(198, 211, 224),
            panel_fill=(244, 248, 252),
            panel_alt_fill=(235, 244, 250),
            row_fill=(255, 255, 255),
            row_alt_fill=(248, 251, 254),
            text=(35, 48, 63),
            muted_text=(85, 101, 118),
            disabled_fill=(231, 236, 241),
            disabled_outline=(164, 179, 193),
            accent=(38, 113, 171),
            accent_alt=(64, 142, 137),
            success_fill=(218, 241, 230),
            warning_fill=(255, 241, 207),
            blocked_fill=(244, 225, 229),
        )
    if str(style_variant) == "warm":
        return RecordTableTheme(
            name="warm",
            app_fill=(255, 254, 250),
            title_bar=(116, 77, 49),
            title_text=(255, 255, 255),
            chrome_line=(219, 207, 193),
            panel_fill=(250, 247, 241),
            panel_alt_fill=(244, 238, 228),
            row_fill=(255, 255, 252),
            row_alt_fill=(251, 248, 242),
            text=(55, 44, 34),
            muted_text=(112, 93, 73),
            disabled_fill=(236, 231, 224),
            disabled_outline=(177, 160, 145),
            accent=(159, 90, 45),
            accent_alt=(46, 126, 119),
            success_fill=(219, 240, 228),
            warning_fill=(255, 239, 205),
            blocked_fill=(244, 224, 228),
        )
    if str(style_variant) == "sage":
        return RecordTableTheme(
            name="sage",
            app_fill=(253, 255, 253),
            title_bar=(53, 92, 79),
            title_text=(255, 255, 255),
            chrome_line=(198, 216, 208),
            panel_fill=(244, 250, 247),
            panel_alt_fill=(234, 245, 240),
            row_fill=(255, 255, 255),
            row_alt_fill=(248, 252, 250),
            text=(35, 53, 47),
            muted_text=(80, 104, 96),
            disabled_fill=(230, 238, 234),
            disabled_outline=(160, 181, 172),
            accent=(41, 123, 100),
            accent_alt=(166, 91, 65),
            success_fill=(217, 241, 228),
            warning_fill=(255, 241, 210),
            blocked_fill=(244, 225, 229),
        )
    if str(style_variant) == "compact":
        return RecordTableTheme(
            name="compact",
            app_fill=(251, 252, 253),
            title_bar=(45, 53, 67),
            title_text=(250, 252, 255),
            chrome_line=(203, 209, 218),
            panel_fill=(242, 245, 248),
            panel_alt_fill=(232, 240, 240),
            row_fill=(255, 255, 255),
            row_alt_fill=(247, 249, 250),
            text=(38, 44, 55),
            muted_text=(91, 99, 112),
            disabled_fill=(229, 232, 236),
            disabled_outline=(159, 167, 176),
            accent=(0, 126, 145),
            accent_alt=(225, 90, 71),
            success_fill=(215, 241, 225),
            warning_fill=(255, 239, 205),
            blocked_fill=(246, 226, 230),
        )
    if str(style_variant) == "contrast":
        return RecordTableTheme(
            name="contrast",
            app_fill=(250, 250, 247),
            title_bar=(34, 34, 38),
            title_text=(255, 255, 255),
            chrome_line=(184, 184, 178),
            panel_fill=(241, 241, 236),
            panel_alt_fill=(229, 239, 246),
            row_fill=(255, 255, 252),
            row_alt_fill=(246, 246, 242),
            text=(26, 28, 32),
            muted_text=(77, 81, 88),
            disabled_fill=(224, 224, 218),
            disabled_outline=(142, 142, 136),
            accent=(184, 53, 71),
            accent_alt=(24, 121, 108),
            success_fill=(217, 239, 224),
            warning_fill=(255, 237, 194),
            blocked_fill=(238, 218, 222),
        )
    return RecordTableTheme(
        name="standard",
        app_fill=(255, 255, 255),
        title_bar=(63, 78, 104),
        title_text=(255, 255, 255),
        chrome_line=(205, 211, 220),
        panel_fill=(246, 248, 250),
        panel_alt_fill=(237, 243, 248),
        row_fill=(255, 255, 255),
        row_alt_fill=(248, 250, 252),
        text=(40, 48, 61),
        muted_text=(91, 101, 117),
        disabled_fill=(231, 235, 240),
        disabled_outline=(151, 163, 179),
        accent=(43, 114, 197),
        accent_alt=(213, 111, 54),
        success_fill=(219, 242, 228),
        warning_fill=(255, 239, 205),
        blocked_fill=(244, 224, 228),
    )


def _normalize_int_support(params: Mapping[str, Any], key: str, fallback: Sequence[int]) -> Tuple[int, ...]:
    raw_values = params.get(str(key), group_default(_GEN_DEFAULTS, str(key), fallback))
    support: List[int] = []
    for raw_value in raw_values:
        value = int(raw_value)
        if value not in support:
            support.append(value)
    if not support:
        raise ValueError(f"{key} must not be empty for {NAMESPACE_ROOT}")
    return tuple(int(value) for value in support)


def _normalize_str_support(params: Mapping[str, Any], key: str, fallback: Sequence[str]) -> Tuple[str, ...]:
    raw_values = params.get(str(key), group_default(_GEN_DEFAULTS, str(key), fallback))
    support: List[str] = []
    for raw_value in raw_values:
        value = str(raw_value).strip()
        if value and value not in support:
            support.append(value)
    if not support:
        raise ValueError(f"{key} must not be empty for {NAMESPACE_ROOT}")
    return tuple(str(value) for value in support)


def _decoupled_params(params: Mapping[str, Any], *, divisor: int, namespace: str) -> Mapping[str, Any]:
    _ = int(divisor), namespace
    return params


def _support_selection_index(params: Mapping[str, Any], *, instance_seed: int, namespace: str) -> int:
    return int(resolve_selection_index(params=params, instance_seed=int(instance_seed), namespace=f"{NAMESPACE_ROOT}:{namespace}"))


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
    balanced = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=supported,
        balance_flag_key=str(balance_flag_key),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        sampling_namespace=f"{NAMESPACE_ROOT}:{namespace}",
    )
    return str(balanced), {str(key): float(value) for key, value in sorted(probabilities.items())}


def _resolve_render_params(params: Mapping[str, Any], *, instance_seed: int | None = None) -> RecordTableRenderParams:
    values = asdict(_DEFAULTS)

    def _int_value(key: str) -> int:
        return resolve_render_int(
            params,
            _RENDER_DEFAULTS,
            str(key),
            int(values[str(key)]),
            instance_seed=instance_seed,
            namespace=NAMESPACE_ROOT,
        )

    return RecordTableRenderParams(
        canvas_width=_int_value("canvas_width"),
        canvas_height=_int_value("canvas_height"),
        window_margin_px=_int_value("window_margin_px"),
        title_bar_height_px=_int_value("title_bar_height_px"),
        menu_bar_height_px=_int_value("menu_bar_height_px"),
        corner_radius_px=_int_value("corner_radius_px"),
        table_corner_radius_px=_int_value("table_corner_radius_px"),
        row_height_px=_int_value("row_height_px"),
        section_header_height_px=_int_value("section_header_height_px"),
        table_header_height_px=_int_value("table_header_height_px"),
        title_font_size_px=_int_value("title_font_size_px"),
        body_font_size_px=_int_value("body_font_size_px"),
        small_font_size_px=_int_value("small_font_size_px"),
    )


def _select_indices(*, instance_seed: int, namespace: str, count: int, size: int) -> Tuple[int, ...]:
    if int(count) > int(size):
        raise ValueError("cannot select more indices than population size")
    indices = list(range(int(size)))
    rng = spawn_rng(int(instance_seed), f"{NAMESPACE_ROOT}.{namespace}")
    rng.shuffle(indices)
    return tuple(sorted(int(value) for value in indices[: int(count)]))


def _choose_from_support(
    support: Sequence[Any],
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> Any:
    return support[
        _support_selection_index(params, instance_seed=int(instance_seed), namespace=str(namespace))
        % len(support)
    ]


def _section_sizes(
    *,
    row_count: int,
    section_count: int,
    target_section_index: int,
    min_target_size: int,
) -> Tuple[int, ...]:
    sizes = [2 for _ in range(int(section_count))]
    remaining = int(row_count) - sum(sizes)
    required_extra = max(0, int(min_target_size) - int(sizes[int(target_section_index)]))
    add_to_target = min(int(remaining), int(required_extra))
    sizes[int(target_section_index)] += int(add_to_target)
    remaining -= int(add_to_target)
    cursor = 0
    while remaining > 0:
        section_index = (int(target_section_index) + int(cursor)) % int(section_count)
        sizes[int(section_index)] += 1
        remaining -= 1
        cursor += 1
    return tuple(int(value) for value in sizes)


def _status_fill(status: str, theme: RecordTableTheme) -> Color:
    normalized = str(status).lower()
    if normalized == "ready":
        return theme.success_fill
    if normalized == "warning":
        return theme.warning_fill
    if normalized == "blocked":
        return theme.blocked_fill
    return theme.panel_alt_fill


def _row_matches(row: RecordTableRow, query: RecordTableCase) -> bool:
    if str(query.filter_key) == "selected_status_filter":
        return bool(row.selected) and str(row.status_label) == str(query.target_status)
    if str(query.filter_key) == "enabled_type_action_filter":
        return (
            str(row.type_label) == str(query.target_type)
            and str(row.action_label) == str(query.target_action_label)
            and bool(row.action_enabled)
        )
    return (
        str(row.section_name) == str(query.target_section_name)
        and int(row.size_mb) >= int(query.size_threshold_mb)
    )


def build_record_table_case(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    filter_key: str,
    default_row_count_support: Sequence[int],
    default_answer_count_support: Sequence[int],
) -> RecordTableCase:
    """Build one table state for a task-owned neutral row predicate."""

    if str(filter_key) not in FILTER_KEYS:
        raise ValueError(f"unsupported record-table filter: {filter_key!r}")
    rng = spawn_rng(int(instance_seed), f"{NAMESPACE_ROOT}.case")
    scene_variant, scene_variant_probabilities = _resolve_named_axis(
        rng,
        instance_seed=int(instance_seed),
        params=params,
        supported=SUPPORTED_SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        namespace="scene_variant",
    )
    style_variant, style_variant_probabilities = _resolve_named_axis(
        rng,
        instance_seed=int(instance_seed),
        params=params,
        supported=SUPPORTED_STYLE_VARIANTS,
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        namespace="style_variant",
    )

    row_count_support = _normalize_int_support(params, "row_count_support", default_row_count_support)
    section_count_support = _normalize_int_support(params, "section_count_support", _DEFAULTS.section_count_support)
    answer_count_support = _normalize_int_support(params, "answer_count_support", default_answer_count_support)
    section_name_pool = _normalize_str_support(params, "section_name_pool", _DEFAULTS.section_name_pool)
    type_label_pool = _normalize_str_support(params, "type_label_pool", _DEFAULTS.type_label_pool)
    status_label_pool = _normalize_str_support(params, "status_label_pool", _DEFAULTS.status_label_pool)
    action_label_pool = _normalize_str_support(params, "action_label_pool", _DEFAULTS.action_label_pool)
    size_threshold_support = _normalize_int_support(params, "size_threshold_support", _DEFAULTS.size_threshold_support)
    if len(section_name_pool) < max(section_count_support):
        raise ValueError("section_name_pool must cover section_count_support")
    if len(type_label_pool) < 3 or len(status_label_pool) < 3 or len(action_label_pool) < 3:
        raise ValueError("GUI table row-filter pools must each contain at least three labels")

    explicit_answer_value = params.get("answer_value")
    if explicit_answer_value is not None:
        answer_value = int(explicit_answer_value)
        if answer_value not in set(int(value) for value in answer_count_support):
            raise ValueError("answer_value must be in answer_count_support")
    else:
        answer_value = int(
            answer_count_support[
                _support_selection_index(params, instance_seed=int(instance_seed), namespace=f"answer_value.{filter_key}")
                % len(answer_count_support)
            ]
        )

    requested_row_count = int(
        params.get(
            "row_count",
            _choose_from_support(row_count_support, params=params, instance_seed=int(instance_seed), namespace="row_count"),
        )
    )
    section_count = int(
        params.get(
            "section_count",
            _choose_from_support(
                section_count_support,
                params=params,
                instance_seed=int(instance_seed),
                namespace=f"section_count.{filter_key}",
            ),
        )
    )
    target_status = str(
        params.get(
            "target_status",
            _choose_from_support(status_label_pool, params=params, instance_seed=int(instance_seed), namespace="target_status"),
        )
    )
    target_type = str(
        params.get(
            "target_type",
            _choose_from_support(type_label_pool, params=params, instance_seed=int(instance_seed), namespace="target_type"),
        )
    )
    target_action_label = str(
        params.get(
            "target_action_label",
            _choose_from_support(
                action_label_pool,
                params=params,
                instance_seed=int(instance_seed),
                namespace="target_action",
            ),
        )
    )
    size_threshold_mb = int(
        params.get(
            "size_threshold_mb",
            _choose_from_support(
                size_threshold_support,
                params=params,
                instance_seed=int(instance_seed),
                namespace="size_threshold",
            ),
        )
    )

    target_section_index = int(
        _support_selection_index(params, instance_seed=int(instance_seed), namespace=f"target_section.{filter_key}")
        % int(section_count)
    )
    min_target_size = int(answer_value) + 1 if str(filter_key) == "section_size_threshold_filter" else 3
    row_count = min(max(row_count_support), max(int(requested_row_count), int(answer_value) + 5, int(min_target_size) + 2))
    if row_count < min_target_size + (section_count - 1) * 2:
        row_count = min(max(row_count_support), min_target_size + (section_count - 1) * 2)
    if row_count < int(answer_value):
        raise ValueError("row_count_support cannot support requested answer count")

    section_names = tuple(str(value) for value in section_name_pool[: int(section_count)])
    target_section_name = str(section_names[int(target_section_index)])
    sizes = _section_sizes(
        row_count=int(row_count),
        section_count=int(section_count),
        target_section_index=int(target_section_index),
        min_target_size=int(min_target_size),
    )

    flat_keys: List[Tuple[int, int]] = []
    for section_index, size in enumerate(sizes):
        for order_in_section in range(int(size)):
            flat_keys.append((int(section_index), int(order_in_section)))
    annotation_keys = set(
        flat_keys[int(index)]
        for index in _select_indices(
            instance_seed=int(instance_seed),
            namespace=f"annotation.{filter_key}",
            count=int(answer_value),
            size=len(flat_keys),
        )
    )
    if str(filter_key) == "section_size_threshold_filter":
        target_keys = [key for key in flat_keys if int(key[0]) == int(target_section_index)]
        annotation_keys = set(
            target_keys[int(index)]
            for index in _select_indices(
                instance_seed=int(instance_seed),
                namespace=f"annotation.{filter_key}.{target_section_index}",
                count=int(answer_value),
                size=len(target_keys),
            )
        )

    rows: List[RecordTableRow] = []
    annotation_row_ids: List[str] = []
    non_target_statuses = [str(value) for value in status_label_pool if str(value) != str(target_status)]
    non_target_types = [str(value) for value in type_label_pool if str(value) != str(target_type)]
    non_target_actions = [str(value) for value in action_label_pool if str(value) != str(target_action_label)]

    global_index = 0
    for section_index, section_name in enumerate(section_names):
        for order_in_section in range(int(sizes[int(section_index)])):
            key = (int(section_index), int(order_in_section))
            row_id = f"row_{global_index:02d}"
            is_match = key in annotation_keys
            row_label = chr(ord("A") + int(global_index))
            item_name = f"{_ITEM_STEMS[int(global_index) % len(_ITEM_STEMS)]}-{100 + ((int(instance_seed) + int(global_index) * 17) % 900)}"
            type_label = str(type_label_pool[(int(global_index) + int(section_index)) % len(type_label_pool)])
            status_label = str(status_label_pool[(int(global_index) + int(order_in_section)) % len(status_label_pool)])
            action_label = str(action_label_pool[(int(global_index) + int(section_index)) % len(action_label_pool)])
            selected = bool((int(global_index) + int(instance_seed)) % 5 == 0)
            action_enabled = bool((int(global_index) + int(section_index)) % 3 != 0)
            size_mb = 10 + int((abs(hash64(int(instance_seed), row_id, 17)) % 82))

            if str(filter_key) == "selected_status_filter":
                if is_match:
                    selected = True
                    status_label = str(target_status)
                elif selected and status_label == str(target_status):
                    status_label = non_target_statuses[int(global_index) % len(non_target_statuses)]
                elif not selected and int(global_index) % 4 == 0:
                    status_label = str(target_status)
            elif str(filter_key) == "enabled_type_action_filter":
                if is_match:
                    type_label = str(target_type)
                    action_label = str(target_action_label)
                    action_enabled = True
                elif type_label == str(target_type) and action_label == str(target_action_label) and bool(action_enabled):
                    action_enabled = False
                elif int(global_index) % 4 == 0:
                    type_label = str(target_type)
                    action_label = str(target_action_label)
                    action_enabled = False
                elif int(global_index) % 4 == 1:
                    type_label = non_target_types[int(global_index) % len(non_target_types)]
                    action_label = str(target_action_label)
                    action_enabled = True
                elif int(global_index) % 4 == 2:
                    type_label = str(target_type)
                    action_label = non_target_actions[int(global_index) % len(non_target_actions)]
                    action_enabled = True
            else:
                if int(section_index) == int(target_section_index):
                    if is_match:
                        size_mb = int(size_threshold_mb) + 4 + int((global_index * 7) % 31)
                    else:
                        size_mb = max(1, int(size_threshold_mb) - 4 - int((global_index * 5) % 19))
                elif int(global_index) % 2 == 0:
                    size_mb = int(size_threshold_mb) + 6 + int((global_index * 3) % 25)
                else:
                    size_mb = max(1, int(size_threshold_mb) - 5 - int((global_index * 2) % 17))

            row = RecordTableRow(
                row_id=str(row_id),
                row_label=str(row_label),
                section_name=str(section_name),
                section_index=int(section_index),
                order_in_section=int(order_in_section),
                global_order_index=int(global_index),
                item_name=str(item_name),
                type_label=str(type_label),
                status_label=str(status_label),
                size_mb=int(size_mb),
                selected=bool(selected),
                action_label=str(action_label),
                action_enabled=bool(action_enabled),
            )
            rows.append(row)
            global_index += 1

    placeholder = RecordTableCase(
        filter_key=str(filter_key),
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        rows=tuple(rows),
        section_names=tuple(section_names),
        target_status=str(target_status),
        target_type=str(target_type),
        target_action_label=str(target_action_label),
        target_section_name=str(target_section_name),
        target_section_index=int(target_section_index),
        size_threshold_mb=int(size_threshold_mb),
        answer_value=int(answer_value),
        annotation_row_ids=(),
        row_count_support=tuple(int(value) for value in row_count_support),
        answer_count_support=tuple(int(value) for value in answer_count_support),
        section_count_support=tuple(int(value) for value in section_count_support),
        filter_key_probabilities={str(filter_key): 1.0},
        scene_variant_probabilities=dict(scene_variant_probabilities),
        style_variant_probabilities=dict(style_variant_probabilities),
    )
    for row in rows:
        if _row_matches(row, placeholder):
            annotation_row_ids.append(str(row.row_id))
    if len(annotation_row_ids) != int(answer_value):
        raise RuntimeError(
            f"GUI table row-filter annotation cardinality does not match answer for {filter_key}: "
            f"{len(annotation_row_ids)} != {answer_value}"
        )

    return RecordTableCase(
        filter_key=str(filter_key),
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        rows=tuple(rows),
        section_names=tuple(section_names),
        target_status=str(target_status),
        target_type=str(target_type),
        target_action_label=str(target_action_label),
        target_section_name=str(target_section_name),
        target_section_index=int(target_section_index),
        size_threshold_mb=int(size_threshold_mb),
        answer_value=int(answer_value),
        annotation_row_ids=tuple(str(value) for value in annotation_row_ids),
        row_count_support=tuple(int(value) for value in row_count_support),
        answer_count_support=tuple(int(value) for value in answer_count_support),
        section_count_support=tuple(int(value) for value in section_count_support),
        filter_key_probabilities={str(filter_key): 1.0},
        scene_variant_probabilities=dict(scene_variant_probabilities),
        style_variant_probabilities=dict(style_variant_probabilities),
    )


def _draw_app_chrome(
    draw: ImageDraw.ImageDraw,
    *,
    query: RecordTableCase,
    render_params: RecordTableRenderParams,
    theme: RecordTableTheme,
) -> Tuple[BBox, RecordTableProfile]:
    """Draw the desktop-app frame and return the content region."""

    profile = _APP_PROFILES[str(query.scene_variant)]
    m = int(render_params.window_margin_px)
    width = int(render_params.canvas_width)
    height = int(render_params.canvas_height)
    window = (float(m), float(m - 6), float(width - m), float(height - m + 6))
    _rounded_rect(draw, window, radius=int(render_params.corner_radius_px), fill=theme.app_fill, outline=theme.chrome_line, width=2)
    title_bar = (window[0], window[1], window[2], window[1] + int(render_params.title_bar_height_px))
    draw.rounded_rectangle(
        [title_bar[0], title_bar[1], title_bar[2], title_bar[3] + int(render_params.corner_radius_px)],
        radius=int(render_params.corner_radius_px),
        fill=theme.title_bar,
    )
    draw.rectangle([title_bar[0], title_bar[3] - int(render_params.corner_radius_px), title_bar[2], title_bar[3]], fill=theme.title_bar)
    dot_y = (title_bar[1] + title_bar[3]) / 2.0
    for idx, color in enumerate(((221, 91, 84), (229, 174, 65), (88, 174, 104))):
        draw.ellipse([window[0] + 18 + idx * 22, dot_y - 6, window[0] + 30 + idx * 22, dot_y + 6], fill=color)
    title_font = load_font(int(render_params.title_font_size_px), bold=True)
    draw_text_traced(draw,(window[0] + 98, title_bar[1] + 10), str(profile.app_title), fill=theme.title_text, font=title_font, role="readout", required=False)
    small_font = load_font(int(render_params.small_font_size_px), bold=False)
    draw_text_traced(draw,(window[2] - 255, title_bar[1] + 16), str(profile.window_title), fill=theme.title_text, font=small_font, role="readout", required=False)
    menu_y1 = title_bar[3]
    menu_y2 = menu_y1 + int(render_params.menu_bar_height_px)
    draw.rectangle([window[0], menu_y1, window[2], menu_y2], fill=theme.panel_fill, outline=theme.chrome_line)
    tab_font = load_font(int(render_params.small_font_size_px), bold=True)
    tab_x = window[0] + 26
    for idx, tab in enumerate(("File", str(profile.primary_tab), str(profile.secondary_tab), "View", "Help")):
        fill = theme.accent if idx == 1 else theme.muted_text
        draw_text_traced(draw,(tab_x, menu_y1 + 9), tab, fill=fill, font=tab_font, role="readout", required=False)
        tab_x += 86 if idx else 64
    return (window[0] + 22, menu_y2 + 18, window[2] - 22, window[3] - 18), profile


def _draw_checkbox(draw: ImageDraw.ImageDraw, bbox: BBox, *, selected: bool, theme: RecordTableTheme) -> None:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255), outline=theme.disabled_outline, width=2)
    if bool(selected):
        draw.line([(x1 + 4, y1 + 9), (x1 + 9, y2 - 4), (x2 - 3, y1 + 4)], fill=theme.accent, width=3)


def _draw_status_pill(draw: ImageDraw.ImageDraw, bbox: BBox, *, status: str, theme: RecordTableTheme, max_size_px: int) -> None:
    fill = _status_fill(str(status), theme)
    _rounded_rect(draw, bbox, radius=8, fill=fill, outline=theme.chrome_line, width=1)
    _draw_text_center_fit(draw, text=str(status), bbox=bbox, fill=theme.text, max_size_px=int(max_size_px), bold=True)


def _draw_action_button(
    draw: ImageDraw.ImageDraw,
    bbox: BBox,
    *,
    label: str,
    enabled: bool,
    theme: RecordTableTheme,
    max_size_px: int,
) -> None:
    fill = theme.success_fill if bool(enabled) else theme.disabled_fill
    outline = theme.accent if bool(enabled) else theme.disabled_outline
    text_fill = theme.accent if bool(enabled) else theme.muted_text
    _rounded_rect(draw, bbox, radius=7, fill=fill, outline=outline, width=2)
    state = "ON" if bool(enabled) else "OFF"
    _draw_text_center_fit(draw, text=f"{label} {state}", bbox=bbox, fill=text_fill, max_size_px=int(max_size_px), bold=True)


def _draw_record_table_scene(
    image: Image.Image,
    *,
    query: RecordTableCase,
    render_params: RecordTableRenderParams,
) -> RenderedRecordTable:
    """Project table rows and section groups into the app-window layout."""

    draw = ImageDraw.Draw(image)
    theme = _theme(str(query.style_variant))
    content_bbox, profile = _draw_app_chrome(draw, query=query, render_params=render_params, theme=theme)
    x1, y1, x2, y2 = [float(value) for value in content_bbox]
    title_bar = (x1 + 18.0, y1 + 10.0, x2 - 18.0, y1 + 50.0)
    _rounded_rect(draw, title_bar, radius=10, fill=theme.panel_alt_fill, outline=theme.chrome_line, width=1)
    _draw_text_left(
        draw,
        text=str(profile.workspace_title),
        bbox=(title_bar[0] + 18.0, title_bar[1] + 8.0, title_bar[0] + 360.0, title_bar[3] - 8.0),
        fill=theme.text,
        max_size_px=int(render_params.body_font_size_px),
        bold=True,
    )
    status_font = load_font(int(render_params.small_font_size_px), bold=False)
    draw_text_traced(draw,(title_bar[2] - 142.0, title_bar[1] + 13.0), str(profile.status_text), fill=theme.muted_text, font=status_font, role="readout", required=False)

    table = (x1 + 18.0, title_bar[3] + 12.0, x2 - 18.0, y2 - 16.0)
    _rounded_rect(draw, table, radius=int(render_params.table_corner_radius_px), fill=theme.panel_fill, outline=theme.chrome_line, width=1)

    table_x1, table_y1, table_x2, _table_y2 = [float(value) for value in table]
    pad_x = 12.0
    col_x = {
        "select": table_x1 + pad_x,
        "name": table_x1 + 76.0,
        "type": table_x1 + 398.0,
        "status": table_x1 + 548.0,
        "size": table_x1 + 718.0,
        "action": table_x1 + 872.0,
    }
    col_right = {
        "select": table_x1 + 62.0,
        "name": table_x1 + 384.0,
        "type": table_x1 + 532.0,
        "status": table_x1 + 702.0,
        "size": table_x1 + 850.0,
        "action": table_x2 - pad_x,
    }
    header_y1 = table_y1
    header_y2 = table_y1 + int(render_params.table_header_height_px)
    draw.rectangle([table_x1, header_y1, table_x2, header_y2], fill=theme.panel_alt_fill, outline=theme.chrome_line)
    for key, label in (
        ("select", "Sel"),
        ("name", "Name"),
        ("type", "Type"),
        ("status", "Status"),
        ("size", "Size"),
        ("action", "Action"),
    ):
        _draw_text_left(
            draw,
            text=str(label),
            bbox=(col_x[key], header_y1 + 4.0, col_right[key], header_y2 - 4.0),
            fill=theme.muted_text,
            max_size_px=int(render_params.small_font_size_px),
            bold=True,
        )

    rows_by_section: Dict[str, List[RecordTableRow]] = {str(name): [] for name in query.section_names}
    for row in query.rows:
        rows_by_section[str(row.section_name)].append(row)

    row_bboxes: Dict[str, List[float]] = {}
    section_bboxes: Dict[str, List[float]] = {}
    cell_bboxes: Dict[str, Dict[str, List[float]]] = {}
    y_cursor = header_y2
    section_font = load_font(int(render_params.small_font_size_px), bold=True)
    for section_index, section_name in enumerate(query.section_names):
        section_y1 = y_cursor
        section_y2 = section_y1 + int(render_params.section_header_height_px)
        draw.rectangle([table_x1, section_y1, table_x2, section_y2], fill=theme.panel_alt_fill, outline=theme.chrome_line)
        draw_text_traced(draw,(table_x1 + 14.0, section_y1 + 6.0), str(section_name), fill=theme.text, font=section_font, role="readout", required=False)
        draw_text_traced(draw,
            (table_x2 - 92.0, section_y1 + 6.0),
            f"{len(rows_by_section[str(section_name)])} rows",
            fill=theme.muted_text,
            font=section_font,
         role="readout", required=False,)
        y_cursor = section_y2
        first_row_y = y_cursor
        for local_index, row in enumerate(rows_by_section[str(section_name)]):
            row_y1 = y_cursor
            row_y2 = row_y1 + int(render_params.row_height_px)
            fill = theme.row_alt_fill if local_index % 2 else theme.row_fill
            draw.rectangle([table_x1, row_y1, table_x2, row_y2], fill=fill, outline=theme.chrome_line)
            checkbox = (col_x["select"] + 7.0, row_y1 + 6.0, col_x["select"] + 23.0, row_y1 + 22.0)
            _draw_checkbox(draw, checkbox, selected=bool(row.selected), theme=theme)
            _draw_text_left(
                draw,
                text=f"{row.row_label}  {row.item_name}",
                bbox=(col_x["name"], row_y1 + 3.0, col_right["name"], row_y2 - 3.0),
                fill=theme.text,
                max_size_px=int(render_params.small_font_size_px + 1),
                bold=False,
            )
            _draw_text_left(
                draw,
                text=str(row.type_label),
                bbox=(col_x["type"], row_y1 + 3.0, col_right["type"], row_y2 - 3.0),
                fill=theme.text,
                max_size_px=int(render_params.small_font_size_px + 1),
                bold=False,
            )
            _draw_status_pill(
                draw,
                (col_x["status"], row_y1 + 4.0, col_right["status"] - 16.0, row_y2 - 4.0),
                status=str(row.status_label),
                theme=theme,
                max_size_px=int(render_params.small_font_size_px),
            )
            _draw_text_left(
                draw,
                text=f"{row.size_mb} MB",
                bbox=(col_x["size"], row_y1 + 3.0, col_right["size"], row_y2 - 3.0),
                fill=theme.text,
                max_size_px=int(render_params.small_font_size_px + 1),
                bold=False,
            )
            _draw_action_button(
                draw,
                (col_x["action"], row_y1 + 4.0, col_right["action"] - 8.0, row_y2 - 4.0),
                label=str(row.action_label),
                enabled=bool(row.action_enabled),
                theme=theme,
                max_size_px=int(render_params.small_font_size_px),
            )
            row_bbox = (table_x1, row_y1, table_x2, row_y2)
            row_bboxes[str(row.row_id)] = _bbox_list(row_bbox)
            cell_bboxes[str(row.row_id)] = {
                "selection": _bbox_list((col_x["select"], row_y1, col_right["select"], row_y2)),
                "name": _bbox_list((col_x["name"], row_y1, col_right["name"], row_y2)),
                "type": _bbox_list((col_x["type"], row_y1, col_right["type"], row_y2)),
                "status": _bbox_list((col_x["status"], row_y1, col_right["status"], row_y2)),
                "size": _bbox_list((col_x["size"], row_y1, col_right["size"], row_y2)),
                "action": _bbox_list((col_x["action"], row_y1, col_right["action"], row_y2)),
            }
            y_cursor = row_y2
        section_bboxes[str(section_name)] = _bbox_list((table_x1, section_y1, table_x2, y_cursor if y_cursor > first_row_y else section_y2))

    row_records: List[Dict[str, Any]] = []
    for row in query.rows:
        row_records.append(
            {
                "row_id": str(row.row_id),
                "row_label": str(row.row_label),
                "section_name": str(row.section_name),
                "section_index": int(row.section_index),
                "order_in_section": int(row.order_in_section),
                "global_order_index": int(row.global_order_index),
                "item_name": str(row.item_name),
                "type_label": str(row.type_label),
                "status_label": str(row.status_label),
                "size_mb": int(row.size_mb),
                "selected": bool(row.selected),
                "action_label": str(row.action_label),
                "action_enabled": bool(row.action_enabled),
                "bbox_px": list(row_bboxes[str(row.row_id)]),
                "cell_bboxes_px": dict(cell_bboxes[str(row.row_id)]),
            }
        )

    m = int(render_params.window_margin_px)
    window_bbox = [float(m), float(m - 6), float(render_params.canvas_width - m), float(render_params.canvas_height - m + 6)]
    return RenderedRecordTable(
        row_bboxes_by_id={str(key): list(value) for key, value in row_bboxes.items()},
        section_bboxes_by_name={str(key): list(value) for key, value in section_bboxes.items()},
        cell_bboxes_by_row_id={str(key): dict(value) for key, value in cell_bboxes.items()},
        row_records=tuple(row_records),
        scene_bbox_px=[0.0, 0.0, float(render_params.canvas_width), float(render_params.canvas_height)],
        window_bbox_px=_bbox_list(tuple(window_bbox)),
        profile=profile,
        theme=theme,
    )


def render_record_table_case(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    case: RecordTableCase,
) -> RenderedRecordTableBundle:
    """Render one sampled record-table case to an image and pixel metadata."""

    render_params = _resolve_render_params(params, instance_seed=int(instance_seed))
    background, background_meta = make_background_canvas(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_BACKGROUND_DEFAULTS,
    )
    image = background.copy().convert("RGB")
    rendered_table = _draw_record_table_scene(image, query=case, render_params=render_params)
    image, post_noise_meta = apply_post_image_noise(
        image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return RenderedRecordTableBundle(
        image=image,
        render_params=render_params,
        rendered_table=rendered_table,
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
    )


__all__ = [
    "FILTER_KEYS",
    "RecordTableCase",
    "RecordTableRenderParams",
    "RecordTableRow",
    "RenderedRecordTable",
    "RenderedRecordTableBundle",
    "SUPPORTED_SCENE_VARIANTS",
    "SUPPORTED_STYLE_VARIANTS",
    "build_record_table_case",
    "render_record_table_case",
]
