"""Scene-local sampling and render defaults for paired-form tasks."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

from .....core.seed import spawn_rng
from ....shared.config_defaults import group_default, resolve_required_int_bounds
from ....shared.render_variation import resolve_layout_jitter, resolve_render_int, resolve_render_rgb
from ...shared.common import resolve_pages_axis_variant
from ...shared.text_generation import sample_company_name, sample_identifier


SUPPORTED_PAIRED_FORMS_SCENE_VARIANTS: Tuple[str, ...] = ("purchase_receipt_pair",)
AnswerFunction = Callable[[Sequence[Mapping[str, Any]]], int]

_ITEM_NAMES: Tuple[str, ...] = (
    "Valve",
    "Cable",
    "Filter",
    "Gasket",
    "Bracket",
    "Sensor",
    "Clamp",
    "Switch",
    "Hose",
    "Panel",
    "Relay",
    "Bearing",
    "Nozzle",
    "Adapter",
    "Gauge",
    "Latch",
    "Seal",
    "Coupler",
)
_ITEM_PREFIXES: Tuple[str, ...] = ("AX", "BR", "CN", "DK", "EL", "FT", "GX", "HY", "JP", "KR")
_DOCK_CODES: Tuple[str, ...] = ("A1", "A2", "B1", "B2", "C1", "C2", "D1", "D2")


@dataclass(frozen=True)
class ReconciliationDefaults:
    """Fallback knobs for paired-form reconciliation scenes."""

    item_count_min: int = 8
    item_count_max: int = 12
    quantity_min: int = 30
    quantity_max: int = 99
    unit_value_min: int = 20
    unit_value_max: int = 75
    discrepancy_min: int = 4
    discrepancy_max: int = 18
    mismatch_count_min: int = 4
    mismatch_count_max: int = 12
    direction_count_min: int = 2
    canvas_width: int = 1600
    canvas_height: int = 980
    outer_margin_px: int = 44
    panel_gap_px: int = 34
    panel_corner_radius_px: int = 18
    panel_border_width_px: int = 2
    title_band_height_px: int = 70
    header_field_height_px: int = 52
    table_header_height_px: int = 38
    row_min_height_px: int = 44
    row_max_height_px: int = 58
    title_font_size_px: int = 32
    subtitle_font_size_px: int = 19
    label_font_size_px: int = 16
    cell_font_size_px: int = 20
    panel_fill_rgb: Tuple[int, int, int] = (252, 251, 247)
    panel_border_rgb: Tuple[int, int, int] = (96, 105, 116)
    title_fill_rgb: Tuple[int, int, int] = (67, 105, 128)
    title_text_rgb: Tuple[int, int, int] = (255, 255, 255)
    field_fill_rgb: Tuple[int, int, int] = (255, 255, 255)
    field_border_rgb: Tuple[int, int, int] = (203, 209, 216)
    table_header_fill_rgb: Tuple[int, int, int] = (232, 237, 239)
    row_alt_fill_rgb: Tuple[int, int, int] = (246, 248, 248)
    label_rgb: Tuple[int, int, int] = (66, 73, 82)
    text_rgb: Tuple[int, int, int] = (24, 30, 38)
    stroke_rgb: Tuple[int, int, int] = (255, 255, 255)
    divider_rgb: Tuple[int, int, int] = (212, 218, 224)


@dataclass(frozen=True)
class ReconciliationRenderParams:
    """Resolved rendering parameters for paired-form reconciliation scenes."""

    canvas_width: int
    canvas_height: int
    outer_margin_px: int
    panel_gap_px: int
    panel_corner_radius_px: int
    panel_border_width_px: int
    title_band_height_px: int
    header_field_height_px: int
    table_header_height_px: int
    row_min_height_px: int
    row_max_height_px: int
    title_font_size_px: int
    subtitle_font_size_px: int
    label_font_size_px: int
    cell_font_size_px: int
    panel_fill_rgb: Tuple[int, int, int]
    panel_border_rgb: Tuple[int, int, int]
    title_fill_rgb: Tuple[int, int, int]
    title_text_rgb: Tuple[int, int, int]
    field_fill_rgb: Tuple[int, int, int]
    field_border_rgb: Tuple[int, int, int]
    table_header_fill_rgb: Tuple[int, int, int]
    row_alt_fill_rgb: Tuple[int, int, int]
    label_rgb: Tuple[int, int, int]
    text_rgb: Tuple[int, int, int]
    stroke_rgb: Tuple[int, int, int]
    divider_rgb: Tuple[int, int, int]
    layout_jitter_meta: Dict[str, Any]


def resolve_reconciliation_render_params(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    defaults: ReconciliationDefaults = ReconciliationDefaults(),
    instance_seed: int | None = None,
) -> ReconciliationRenderParams:
    """Resolve render parameters for one paired-form reconciliation scene."""

    def _int(key: str, fallback: int) -> int:
        return resolve_render_int(
            params,
            render_defaults,
            key,
            fallback,
            instance_seed=instance_seed,
            namespace="pages.paired_forms",
        )

    def _rgb(key: str, fallback: Sequence[int]) -> Tuple[int, int, int]:
        return resolve_render_rgb(
            params,
            render_defaults,
            key,
            fallback,
            instance_seed=instance_seed,
            namespace="pages.paired_forms",
        )

    layout_jitter_meta = resolve_layout_jitter(
        params,
        render_defaults,
        instance_seed=instance_seed,
        namespace="pages.paired_forms.layout",
    )

    return ReconciliationRenderParams(
        canvas_width=_int("canvas_width", defaults.canvas_width),
        canvas_height=_int("canvas_height", defaults.canvas_height),
        outer_margin_px=_int("outer_margin_px", defaults.outer_margin_px),
        panel_gap_px=_int("panel_gap_px", defaults.panel_gap_px),
        panel_corner_radius_px=_int("panel_corner_radius_px", defaults.panel_corner_radius_px),
        panel_border_width_px=_int("panel_border_width_px", defaults.panel_border_width_px),
        title_band_height_px=_int("title_band_height_px", defaults.title_band_height_px),
        header_field_height_px=_int("header_field_height_px", defaults.header_field_height_px),
        table_header_height_px=_int("table_header_height_px", defaults.table_header_height_px),
        row_min_height_px=_int("row_min_height_px", defaults.row_min_height_px),
        row_max_height_px=_int("row_max_height_px", defaults.row_max_height_px),
        title_font_size_px=_int("title_font_size_px", defaults.title_font_size_px),
        subtitle_font_size_px=_int("subtitle_font_size_px", defaults.subtitle_font_size_px),
        label_font_size_px=_int("label_font_size_px", defaults.label_font_size_px),
        cell_font_size_px=_int("cell_font_size_px", defaults.cell_font_size_px),
        panel_fill_rgb=_rgb("panel_fill_rgb", defaults.panel_fill_rgb),
        panel_border_rgb=_rgb("panel_border_rgb", defaults.panel_border_rgb),
        title_fill_rgb=_rgb("title_fill_rgb", defaults.title_fill_rgb),
        title_text_rgb=_rgb("title_text_rgb", defaults.title_text_rgb),
        field_fill_rgb=_rgb("field_fill_rgb", defaults.field_fill_rgb),
        field_border_rgb=_rgb("field_border_rgb", defaults.field_border_rgb),
        table_header_fill_rgb=_rgb("table_header_fill_rgb", defaults.table_header_fill_rgb),
        row_alt_fill_rgb=_rgb("row_alt_fill_rgb", defaults.row_alt_fill_rgb),
        label_rgb=_rgb("label_rgb", defaults.label_rgb),
        text_rgb=_rgb("text_rgb", defaults.text_rgb),
        stroke_rgb=_rgb("stroke_rgb", defaults.stroke_rgb),
        divider_rgb=_rgb("divider_rgb", defaults.divider_rgb),
        layout_jitter_meta=dict(layout_jitter_meta),
    )


def resolve_reconciliation_scene_variant(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    sampling_namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve the visual paired-form scene variant."""

    return resolve_pages_axis_variant(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_PAIRED_FORMS_SCENE_VARIANTS,
        task_id=str(sampling_namespace),
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        axis_namespace="scene_variant",
    )


def _sample_item_specs(
    *,
    item_count: int,
    quantity_min: int,
    quantity_max: int,
    unit_value_min: int,
    unit_value_max: int,
    discrepancy_min: int,
    discrepancy_max: int,
    mismatch_count_min: int,
    mismatch_count_max: int,
    direction_count_min: int,
    rng: Any,
) -> list[Dict[str, Any]]:
    """Sample matched purchase-order and receiving-slip line items."""

    names = list(rng.sample(list(_ITEM_NAMES), int(item_count)))
    prefixes = list(rng.sample(list(_ITEM_PREFIXES), min(len(_ITEM_PREFIXES), int(item_count))))
    while len(prefixes) < int(item_count):
        prefixes.append(str(rng.choice(_ITEM_PREFIXES)))

    effective_mismatch_min = max(0, min(int(mismatch_count_min), int(item_count)))
    effective_mismatch_max = max(0, min(int(mismatch_count_max), int(item_count)))
    if int(effective_mismatch_min) > int(effective_mismatch_max):
        raise ValueError("mismatch_count_min must be <= mismatch_count_max after item-count clamping")
    mismatch_count = int(rng.randint(int(effective_mismatch_min), int(effective_mismatch_max)))
    mismatch_indices = set(rng.sample(list(range(int(item_count))), int(mismatch_count)))
    min_direction_count = max(0, int(direction_count_min))
    if int(mismatch_count) == 0:
        negative_count = 0
    else:
        min_negative_count = min(int(min_direction_count), int(mismatch_count))
        max_negative_count = int(mismatch_count) - int(min_direction_count)
        if int(max_negative_count) < int(min_negative_count):
            raise ValueError("mismatch_count is too small for direction_count_min")
        negative_count = int(rng.randint(int(min_negative_count), int(max_negative_count)))
    negative_indices = set(rng.sample(list(mismatch_indices), int(negative_count)))
    available_diffs = list(range(int(discrepancy_min), int(discrepancy_max) + 1))
    if len(available_diffs) < int(mismatch_count):
        raise ValueError("discrepancy range must support unique non-zero differences")
    diffs = list(rng.sample(available_diffs, int(mismatch_count)))
    diff_by_index = {index: int(diff) for index, diff in zip(sorted(mismatch_indices), diffs, strict=True)}

    specs: list[Dict[str, Any]] = []
    seen_item_codes: set[str] = set()
    for index, name in enumerate(names):
        order_qty = int(rng.randint(int(quantity_min), int(quantity_max)))
        unit_value = int(rng.randint(int(unit_value_min), int(unit_value_max)))
        signed_difference = 0
        if index in mismatch_indices:
            diff = int(diff_by_index[int(index)])
            signed_difference = -diff if index in negative_indices else diff
            if signed_difference < 0 and int(order_qty + signed_difference) < int(quantity_min):
                order_qty = int(quantity_min - signed_difference)
            if signed_difference > 0 and int(order_qty + signed_difference) > int(quantity_max):
                order_qty = int(quantity_max - signed_difference)
        received_qty = int(order_qty + signed_difference)
        if received_qty < int(quantity_min) or received_qty > int(quantity_max):
            raise ValueError("received quantity outside supported range")
        item_id = f"item_{index:02d}"
        for _ in range(32):
            item_code = f"{prefixes[index]}-{rng.randint(100, 999)}"
            if item_code not in seen_item_codes:
                seen_item_codes.add(str(item_code))
                break
        else:
            raise ValueError("failed to sample unique item code")
        specs.append(
            {
                "item_id": item_id,
                "item_code": item_code,
                "item_name": str(name),
                "order_qty": int(order_qty),
                "received_qty": int(received_qty),
                "unit_value": int(unit_value),
                "quantity_difference": int(order_qty - received_qty),
                "absolute_quantity_difference": abs(int(order_qty - received_qty)),
                "dock_code": str(rng.choice(_DOCK_CODES)),
                "po_code_bbox_id": f"po:{item_id}:item_code",
                "po_item_bbox_id": f"po:{item_id}:item",
                "po_order_qty_bbox_id": f"po:{item_id}:order_qty",
                "po_unit_value_bbox_id": f"po:{item_id}:unit_value",
                "recv_code_bbox_id": f"recv:{item_id}:item_code",
                "recv_item_bbox_id": f"recv:{item_id}:item",
                "recv_received_qty_bbox_id": f"recv:{item_id}:received_qty",
                "recv_dock_bbox_id": f"recv:{item_id}:dock",
            }
        )
    return specs


def _supporting_bbox_ids(*, item_specs: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return receiving-slip row bbox ids for all mismatched items."""

    support_items = [spec for spec in item_specs if int(spec["absolute_quantity_difference"]) > 0]
    return [f"recv:{spec['item_id']}" for spec in support_items]


def _supporting_cell_bbox_ids(
    *,
    item_specs: Sequence[Mapping[str, Any]],
    include_unit_value: bool,
) -> Dict[str, str]:
    """Return private role-keyed cell bbox ids for audit/debug traces."""

    support_items = [spec for spec in item_specs if int(spec["absolute_quantity_difference"]) > 0]
    role_bbox_ids: Dict[str, str] = {}
    for mismatch_index, spec in enumerate(support_items, start=1):
        prefix = f"mismatch_{int(mismatch_index)}"
        role_bbox_ids[f"{prefix}_purchase_code"] = str(spec["po_code_bbox_id"])
        role_bbox_ids[f"{prefix}_ordered_quantity"] = str(spec["po_order_qty_bbox_id"])
        role_bbox_ids[f"{prefix}_receiving_code"] = str(spec["recv_code_bbox_id"])
        role_bbox_ids[f"{prefix}_received_quantity"] = str(spec["recv_received_qty_bbox_id"])
        if bool(include_unit_value):
            role_bbox_ids[f"{prefix}_unit_value"] = str(spec["po_unit_value_bbox_id"])
    return role_bbox_ids


def build_paired_forms_reconciliation_dataset(
    *,
    operation_key: str,
    answer_fn: AnswerFunction,
    include_unit_value_support: bool,
    scene_variant: str,
    params: Mapping[str, Any],
    instance_seed: int,
    gen_defaults: Mapping[str, Any],
    defaults: ReconciliationDefaults,
    sampling_namespace: str,
) -> Dict[str, Any]:
    """Build one paired-form reconciliation dataset instance."""

    if str(scene_variant) != "purchase_receipt_pair":
        raise ValueError(f"unsupported reconciliation scene_variant '{scene_variant}'")

    item_count_min, item_count_max = resolve_required_int_bounds(
        params,
        gen_defaults,
        min_key="item_count_min",
        max_key="item_count_max",
        fallback_min=int(defaults.item_count_min),
        fallback_max=int(defaults.item_count_max),
        context=f"{sampling_namespace} item count",
    )
    quantity_min, quantity_max = resolve_required_int_bounds(
        params,
        gen_defaults,
        min_key="quantity_min",
        max_key="quantity_max",
        fallback_min=int(defaults.quantity_min),
        fallback_max=int(defaults.quantity_max),
        context=f"{sampling_namespace} quantity range",
    )
    unit_value_min, unit_value_max = resolve_required_int_bounds(
        params,
        gen_defaults,
        min_key="unit_value_min",
        max_key="unit_value_max",
        fallback_min=int(defaults.unit_value_min),
        fallback_max=int(defaults.unit_value_max),
        context=f"{sampling_namespace} unit value range",
    )
    discrepancy_min, discrepancy_max = resolve_required_int_bounds(
        params,
        gen_defaults,
        min_key="discrepancy_min",
        max_key="discrepancy_max",
        fallback_min=int(defaults.discrepancy_min),
        fallback_max=int(defaults.discrepancy_max),
        context=f"{sampling_namespace} discrepancy range",
    )
    mismatch_count_min, mismatch_count_max = resolve_required_int_bounds(
        params,
        gen_defaults,
        min_key="mismatch_count_min",
        max_key="mismatch_count_max",
        fallback_min=int(defaults.mismatch_count_min),
        fallback_max=int(defaults.mismatch_count_max),
        context=f"{sampling_namespace} mismatch count",
    )
    direction_count_min = int(params.get("direction_count_min", group_default(gen_defaults, "direction_count_min", int(defaults.direction_count_min))))
    for attempt in range(128):
        rng = spawn_rng(
            int(instance_seed),
            f"{sampling_namespace}.paired_forms_reconciliation",
            index=int(attempt),
        )
        item_count = int(rng.randint(int(item_count_min), int(item_count_max)))
        item_specs = _sample_item_specs(
            item_count=int(item_count),
            quantity_min=int(quantity_min),
            quantity_max=int(quantity_max),
            unit_value_min=int(unit_value_min),
            unit_value_max=int(unit_value_max),
            discrepancy_min=int(discrepancy_min),
            discrepancy_max=int(discrepancy_max),
            mismatch_count_min=int(mismatch_count_min),
            mismatch_count_max=int(mismatch_count_max),
            direction_count_min=int(direction_count_min),
            rng=rng,
        )
        shortfall_items = [spec for spec in item_specs if int(spec["order_qty"]) > int(spec["received_qty"])]
        overage_items = [spec for spec in item_specs if int(spec["order_qty"]) < int(spec["received_qty"])]
        mismatch_items = [spec for spec in item_specs if int(spec["absolute_quantity_difference"]) > 0]
        unique_abs_diffs = {
            int(spec["absolute_quantity_difference"])
            for spec in item_specs
            if int(spec["absolute_quantity_difference"]) > 0
        }
        if (
            len(shortfall_items) < int(direction_count_min)
            or len(overage_items) < int(direction_count_min)
            or len(mismatch_items) < int(mismatch_count_min)
        ):
            continue
        if len(unique_abs_diffs) != len(mismatch_items):
            continue

        answer_value = int(answer_fn(item_specs))
        visible_numbers = {
            int(value)
            for spec in item_specs
            for value in (int(spec["order_qty"]), int(spec["received_qty"]), int(spec["unit_value"]))
        }
        if int(answer_value) <= 0 or int(answer_value) in visible_numbers:
            continue

        receiving_item_specs = [dict(spec) for spec in item_specs]
        rng.shuffle(receiving_item_specs)
        if [str(spec["item_id"]) for spec in receiving_item_specs] == [str(spec["item_id"]) for spec in item_specs]:
            receiving_item_specs = list(reversed(receiving_item_specs))

        purchase_date = dt.date(2026, 5, 1) + dt.timedelta(days=int(rng.randint(0, 18)))
        receipt_date = purchase_date + dt.timedelta(days=int(rng.randint(2, 9)))
        vendor_name = sample_company_name(rng)
        purchase_header_specs = [
            {"field_id": "po_number", "field_label": "PO Number", "field_value": sample_identifier(rng, prefix="PO", digits=5)},
            {"field_id": "supplier", "field_label": "Supplier", "field_value": vendor_name},
            {"field_id": "order_date", "field_label": "Order Date", "field_value": purchase_date.strftime("%Y-%m-%d")},
        ]
        receiving_header_specs = [
            {"field_id": "slip_number", "field_label": "Slip Number", "field_value": sample_identifier(rng, prefix="RS", digits=5)},
            {"field_id": "dock", "field_label": "Dock", "field_value": str(rng.choice(_DOCK_CODES))},
            {"field_id": "received_date", "field_label": "Received", "field_value": receipt_date.strftime("%Y-%m-%d")},
        ]
        supporting_bbox_ids = _supporting_bbox_ids(item_specs=item_specs)
        supporting_cell_bbox_ids = _supporting_cell_bbox_ids(
            item_specs=item_specs,
            include_unit_value=bool(include_unit_value_support),
        )
        return {
            "scene_variant": str(scene_variant),
            "operation_key": str(operation_key),
            "scene_title": "Order Reconciliation Packet",
            "query_prompt_slots": {},
            "question_format": str(operation_key),
            "view_family": "paired_forms_reconciliation",
            "purchase_title": "Purchase Order",
            "receiving_title": "Receiving Slip",
            "purchase_header_specs": list(purchase_header_specs),
            "receiving_header_specs": list(receiving_header_specs),
            "item_specs": list(item_specs),
            "receiving_item_specs": list(receiving_item_specs),
            "item_count": int(item_count),
            "item_count_range": [int(item_count_min), int(item_count_max)],
            "quantity_range": [int(quantity_min), int(quantity_max)],
            "unit_value_range": [int(unit_value_min), int(unit_value_max)],
            "discrepancy_range": [int(discrepancy_min), int(discrepancy_max)],
            "mismatch_count_range": [int(mismatch_count_min), int(mismatch_count_max)],
            "direction_count_min": int(direction_count_min),
            "shortfall_item_ids": [str(spec["item_id"]) for spec in shortfall_items],
            "overage_item_ids": [str(spec["item_id"]) for spec in overage_items],
            "mismatch_item_ids": [str(spec["item_id"]) for spec in mismatch_items],
            "receiving_item_order_ids": [str(spec["item_id"]) for spec in receiving_item_specs],
            "answer_value": int(answer_value),
            "supporting_cell_bbox_ids": dict(supporting_cell_bbox_ids),
            "supporting_bbox_ids": list(supporting_bbox_ids),
            "annotation_bbox_ids": list(supporting_bbox_ids),
        }
    raise ValueError("failed to build a paired-form reconciliation scene with valid answer constraints")


__all__ = [
    "ReconciliationDefaults",
    "ReconciliationRenderParams",
    "SUPPORTED_PAIRED_FORMS_SCENE_VARIANTS",
    "build_paired_forms_reconciliation_dataset",
    "resolve_reconciliation_render_params",
    "resolve_reconciliation_scene_variant",
]
