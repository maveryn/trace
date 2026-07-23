"""Resource and spec construction for mixed infographic page tasks."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

from .....core.seed import spawn_rng
from ...shared.page_text_resources import (
    PageTextBatch,
    page_text_resource_metadata,
    sample_page_context_batch,
    sample_page_label_batch,
)
from ...shared.page_visual_assets import sample_page_visual_asset
from .state import (
    CATEGORICAL_FIELD_LABELS,
    MODULE_KINDS,
    NUMERIC_FIELD_LABELS,
    _ACCENTS,
    _FIELD_VALUE_BANKS,
    _TEXT_BLOCK_KIND_ORDER,
    _InfographicTextBlock,
    _MixedField,
    _MixedInfographicSpec,
    _MixedItem,
    _MixedModule,
)


def _build_infographic_text_blocks(
    *,
    count: int,
    instance_seed: int,
    resource_namespace: str,
    paragraph_batch: PageTextBatch,
    note_batch: PageTextBatch,
) -> Tuple[_InfographicTextBlock, ...]:
    """Sample non-answer page text blocks; paragraph/context regions must stay unique."""

    rng = spawn_rng(int(instance_seed), f"{resource_namespace}.native_text_blocks")
    target_count = max(4, min(5, int(count)))
    kinds = [kind for kind in _TEXT_BLOCK_KIND_ORDER if str(kind) != "paragraph_note"]
    rng.shuffle(kinds)
    placement_regions = ["header_callout", "footer_badge", "footer_source"]
    blocks: List[_InfographicTextBlock] = []
    used_phrases: set[str] = set()
    paragraph_phrases = list(paragraph_batch.values)
    for index, region in enumerate(("paragraph_left", "paragraph_right")):
        text = str(paragraph_phrases[int(index) % len(paragraph_phrases)])
        used_phrases.add(str(text))
        blocks.append(
            _InfographicTextBlock(
                block_id=f"text_block_{len(blocks) + 1}",
                kind="paragraph_note",
                text=str(text),
                placement_region=str(region),
                font_role="context",
            )
        )
    note_phrases = list(note_batch.values)
    note_cursor = 0
    for index in range(target_count):
        if len(blocks) >= target_count:
            break
        kind = str(kinds[int(index) % len(kinds)])
        while note_cursor < len(note_phrases) and str(note_phrases[note_cursor]) in used_phrases:
            note_cursor += 1
        text = str(note_phrases[int(note_cursor) % len(note_phrases)])
        note_cursor += 1
        used_phrases.add(str(text))
        blocks.append(
            _InfographicTextBlock(
                block_id=f"text_block_{len(blocks) + 1}",
                kind=str(kind),
                text=str(text),
                placement_region=str(placement_regions[int(index) % len(placement_regions)]),
                font_role="context",
            )
        )
    return tuple(blocks)


def _unique_value(
    *,
    rng: Any,
    field_label: str,
    values: Sequence[str],
    used_values: set[str],
    counter: int,
) -> str:
    candidates = [str(value) for value in values]
    rng.shuffle(candidates)
    for candidate in candidates:
        if str(candidate) not in used_values:
            used_values.add(str(candidate))
            return str(candidate)
    while True:
        fallback_base = int(counter)
        if str(field_label) == "Reach":
            fallback = f"{fallback_base}k"
        elif str(field_label) == "Rate":
            fallback = f"{fallback_base}%"
        elif str(field_label) == "Cost":
            fallback = f"${fallback_base}"
        elif str(field_label) == "Rank":
            fallback = f"#{fallback_base}"
        elif str(field_label) in {"Score", "Count"}:
            fallback = str(fallback_base)
        else:
            fallback = f"{str(field_label)[:2].upper()}{fallback_base:03d}"
        counter += 1
        if fallback not in used_values:
            used_values.add(str(fallback))
            return str(fallback)


def _build_mixed_spec(
    *,
    module_count: int,
    item_count_support: Sequence[int],
    field_count_support: Sequence[int],
    native_text_block_count: int,
    instance_seed: int,
    resource_namespace: str,
    allow_categorical_value_reuse: bool = False,
    ensure_shared_numeric_field: bool = False,
    shared_numeric_field_label: str | None = None,
    shared_numeric_field_choices: Sequence[str] | None = None,
) -> _MixedInfographicSpec:
    """Build semantic module/item/value state; visible labels and values drive answers."""

    rng = spawn_rng(int(instance_seed), f"{resource_namespace}.spec")
    title_batch = sample_page_context_batch(
        rng,
        role="mixed_infographic_title",
        count=1,
        manifest_names=("phrases/headlines.txt",),
    )
    subtitle_batch = sample_page_context_batch(
        rng,
        role="mixed_infographic_subtitle",
        count=1,
        manifest_names=("phrases/captions.txt", "phrases/legend_notes.txt"),
    )
    module_title_batch = sample_page_label_batch(
        rng,
        role="mixed_infographic_module_title",
        count=int(module_count),
        manifest_name="categories/product_labels.txt",
        min_chars=3,
        max_chars=16,
        allow_spaces=True,
        allow_punctuation=False,
    )
    max_item_count = int(module_count) * max(int(value) for value in item_count_support)
    item_label_batch = sample_page_label_batch(
        rng,
        role="mixed_infographic_item_label",
        count=max_item_count,
        manifest_name="categories/abstract_group_labels.txt",
        min_chars=3,
        max_chars=14,
        allow_spaces=True,
        allow_punctuation=False,
        exclude=module_title_batch.values,
    )
    paragraph_batch = sample_page_context_batch(
        rng,
        role="mixed_infographic_paragraph_note",
        count=2,
        manifest_names=("paragraphs/context_long_blocks.txt", "paragraphs/context_template_blocks.txt"),
    )
    note_batch = sample_page_context_batch(
        rng,
        role="mixed_infographic_native_note",
        count=6,
        manifest_names=(
            "phrases/callout_phrases.txt",
            "phrases/source_notes.txt",
            "phrases/captions.txt",
            "phrases/sidebar_notes.txt",
        ),
    )
    text_resource_meta = page_text_resource_metadata(
        title_batch,
        subtitle_batch,
        module_title_batch,
        item_label_batch,
        paragraph_batch,
        note_batch,
    )
    module_titles = list(module_title_batch.values)
    item_labels = list(item_label_batch.values)
    field_banks = list(_FIELD_VALUE_BANKS)
    kinds = list(MODULE_KINDS)
    visual_asset_rng = spawn_rng(int(instance_seed), f"{resource_namespace}.page_visual_assets")
    hero_asset_selection = sample_page_visual_asset(visual_asset_rng, role="hero_anchor")
    rng.shuffle(field_banks)
    rng.shuffle(kinds)
    shared_numeric_label = ""
    shared_numeric_module_count = 0
    if bool(ensure_shared_numeric_field):
        shared_numeric_choices = tuple(str(label) for label in (shared_numeric_field_choices or NUMERIC_FIELD_LABELS))
        invalid_shared_labels = [label for label in shared_numeric_choices if str(label) not in set(NUMERIC_FIELD_LABELS)]
        if invalid_shared_labels:
            raise ValueError("shared_numeric_field_choices must be supported numeric mixed-infographic fields")
        requested_shared_label = str(shared_numeric_field_label) if shared_numeric_field_label is not None else ""
        if requested_shared_label and requested_shared_label not in set(NUMERIC_FIELD_LABELS):
            raise ValueError("shared_numeric_field_label must be a supported numeric mixed-infographic field")
        if requested_shared_label and requested_shared_label not in set(shared_numeric_choices):
            raise ValueError("shared_numeric_field_label must be allowed by shared_numeric_field_choices")
        shared_numeric_label = requested_shared_label or str(
            shared_numeric_choices[int(rng.randrange(len(shared_numeric_choices)))]
        )
        shared_numeric_module_count = min(int(module_count), max(3, int(math.ceil(float(module_count) * 0.45))))
    if int(module_count) >= 4:
        required_kinds = ("radial_bubbles", "ring_summary")
        required_start = max(0, int(module_count) - len(required_kinds))
        for offset, required_kind in enumerate(required_kinds):
            if required_kind in kinds:
                kinds.remove(required_kind)
            kinds.insert(required_start + int(offset), required_kind)
    if int(module_count) > len(module_titles):
        raise ValueError("module_count exceeds available unique module titles")
    used_values: set[str] = set()
    used_items = 0
    fallback_counter = 1
    modules: List[_MixedModule] = []
    for module_index in range(int(module_count)):
        module_kind = str(kinds[int(module_index) % len(kinds)])
        section_asset_selection = sample_page_visual_asset(visual_asset_rng, role="section_illustration")
        item_count = int(item_count_support[int(rng.randrange(len(item_count_support)))])
        field_count = int(field_count_support[int(rng.randrange(len(field_count_support)))])
        if module_kind in {"radial_bubbles", "ring_summary"}:
            item_count = min(int(item_count), 3)
            field_count = 1
        elif module_kind in {"profile_cards", "callout_stats"}:
            item_count = min(int(item_count), 3)
            field_count = min(int(field_count), 2)
        bank_offset = int(rng.randrange(len(field_banks)))
        module_fields: List[_MixedField] = []
        for field_index in range(int(field_count)):
            label = str(field_banks[(bank_offset + field_index) % len(field_banks)][0])
            module_fields.append(_MixedField(field_id=f"field_{field_index + 1}", label=label))
        if bool(ensure_shared_numeric_field) and int(module_index) < int(shared_numeric_module_count):
            module_fields[0] = _MixedField(field_id=str(module_fields[0].field_id), label=str(shared_numeric_label))
        if not any(str(field.label) in set(NUMERIC_FIELD_LABELS) for field in module_fields):
            used_labels = {str(field.label) for field in module_fields}
            preferred_numeric_labels = ("Score", "Count", "Reach", "Rate", "Cost", "Rank")
            replacement_label = next(
                label for label in preferred_numeric_labels if str(label) not in used_labels or len(module_fields) == 1
            )
            module_fields[0] = _MixedField(field_id=str(module_fields[0].field_id), label=str(replacement_label))
        module_items: List[_MixedItem] = []
        for item_index in range(int(item_count)):
            if used_items >= len(item_labels):
                item_label = f"Item {used_items + 1}"
            else:
                item_label = str(item_labels[used_items])
            used_items += 1
            values_by_field_id: Dict[str, str] = {}
            for field_index, field in enumerate(module_fields):
                bank = next(values for label, values in field_banks if str(label) == str(field.label))
                if bool(allow_categorical_value_reuse) and str(field.label) in set(CATEGORICAL_FIELD_LABELS):
                    repeat_pool_size = min(len(bank), 2 if int(item_count) <= 4 else 3)
                    value = str(bank[(int(module_index) + int(field_index) + int(item_index)) % int(repeat_pool_size)])
                else:
                    value = _unique_value(
                        rng=rng,
                        field_label=str(field.label),
                        values=bank,
                        used_values=used_values,
                        counter=fallback_counter,
                    )
                    fallback_counter += 1
                values_by_field_id[str(field.field_id)] = str(value)
            module_items.append(
                _MixedItem(
                    item_id=f"item_{item_index + 1}",
                    label=str(item_label),
                    visual_asset_selection=sample_page_visual_asset(
                        visual_asset_rng,
                        role="badge_spot",
                        render_modes=("monochrome",),
                    ),
                    values_by_field_id=dict(values_by_field_id),
                )
            )
        modules.append(
            _MixedModule(
                module_id=f"module_{module_index + 1}",
                kind=str(module_kind),
                title=str(module_titles[int(module_index)]),
                accent_rgb=tuple(int(value) for value in _ACCENTS[int(module_index) % len(_ACCENTS)]),
                section_asset_selection=section_asset_selection,
                fields=tuple(module_fields),
                items=tuple(module_items),
            )
        )
    return _MixedInfographicSpec(
        title=str(title_batch.values[0]),
        subtitle=str(subtitle_batch.values[0]),
        hero_asset_selection=hero_asset_selection,
        modules=tuple(modules),
        text_blocks=_build_infographic_text_blocks(
            count=int(native_text_block_count),
            instance_seed=int(instance_seed),
            resource_namespace=str(resource_namespace),
            paragraph_batch=paragraph_batch,
            note_batch=note_batch,
        ),
        text_resource_metadata=dict(text_resource_meta),
    )

__all__ = ["_build_mixed_spec"]
