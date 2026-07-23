"""Sampling helpers for category-grid scene packages."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.deterministic_sampling import resolve_selection_index
from trace_tasks.tasks.pages.shared.sampling import (
    resolve_int_support as resolve_pages_int_support,
    resolve_named_axis as resolve_pages_named_axis,
    resolve_supported_int as resolve_pages_supported_int,
)
from trace_tasks.tasks.pages.shared.page_text_resources import (
    page_text_resource_metadata,
    sample_page_context_batch,
    sample_page_label_batch,
)

from .defaults import ACCENTS, DEFAULTS, GENERATION_DEFAULTS, NAMESPACE_ROOT, ORDINALS, SCENE_VARIANTS
from .state import Category, CategoryGridCase, CategoryGridSpec, CategoryItem, Subcategory


def ordinal_label(value: int) -> str:
    """Return a compact ordinal label used by visible prompts and trace."""

    return str(ORDINALS.get(int(value), f"{int(value)}th"))


def resolve_named_axis(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    supported: Sequence[str],
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve one balanced named category-grid axis."""

    return resolve_pages_named_axis(
        params=params,
        gen_defaults=GENERATION_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace_root=NAMESPACE_ROOT,
        supported=supported,
        explicit_key=explicit_key,
        weights_key=weights_key,
        balance_flag_key=balance_flag_key,
        namespace=namespace,
    )


def resolve_int_support(params: Mapping[str, Any], key: str, fallback: Sequence[int]) -> Tuple[int, ...]:
    """Resolve an integer support list from config or explicit params."""

    return resolve_pages_int_support(params=params, gen_defaults=GENERATION_DEFAULTS, key=key, fallback=fallback)


def resolve_supported_int(
    *,
    params: Mapping[str, Any],
    explicit_key: str,
    support_key: str,
    fallback: Sequence[int],
    instance_seed: int,
    namespace: str,
) -> Tuple[int, Tuple[int, ...], Dict[str, float]]:
    """Resolve one supported integer operand and probability map."""

    return resolve_pages_supported_int(
        params=params,
        gen_defaults=GENERATION_DEFAULTS,
        namespace_root=NAMESPACE_ROOT,
        explicit_key=explicit_key,
        support_key=support_key,
        fallback=fallback,
        instance_seed=int(instance_seed),
        namespace=namespace,
    )


def build_category_grid_spec(
    *,
    category_count: int,
    subcategory_count: int,
    item_count_support: Sequence[int],
    instance_seed: int,
) -> CategoryGridSpec:
    """Build visible category, subcategory, and item labels."""

    rng = spawn_rng(int(instance_seed), f"{NAMESPACE_ROOT}.spec")
    max_items = int(category_count) * int(subcategory_count) * max(int(value) for value in item_count_support)
    title_batch = sample_page_context_batch(
        rng,
        role="category_grid_title",
        count=1,
        manifest_names=("phrases/headlines.txt",),
    )
    subtitle_batch = sample_page_context_batch(
        rng,
        role="category_grid_subtitle",
        count=1,
        manifest_names=("phrases/captions.txt", "phrases/legend_notes.txt"),
    )
    category_batch = sample_page_label_batch(
        rng,
        role="category_grid_category_label",
        count=int(category_count),
        manifest_name="categories/product_labels.txt",
        min_chars=3,
        max_chars=16,
        allow_spaces=True,
        allow_punctuation=False,
    )
    subcategory_batch = sample_page_label_batch(
        rng,
        role="category_grid_subcategory_label",
        count=int(category_count) * int(subcategory_count),
        manifest_name="panel_titles/technical_topics.txt",
        min_chars=3,
        max_chars=18,
        allow_spaces=True,
        allow_punctuation=False,
        exclude=category_batch.values,
    )
    item_batch = sample_page_label_batch(
        rng,
        role="category_grid_item_label",
        count=int(max_items),
        manifest_name="mixed/compact_labels.txt",
        min_chars=5,
        max_chars=18,
        allow_spaces=True,
        allow_punctuation=False,
        exclude=tuple(category_batch.values) + tuple(subcategory_batch.values),
    )
    category_titles = list(category_batch.values)
    subcategory_titles = list(subcategory_batch.values)
    item_labels = list(item_batch.values)

    item_cursor = 0
    subcategory_cursor = 0
    accent_offset = int(rng.randrange(len(ACCENTS)))
    categories: List[Category] = []
    for category_index in range(int(category_count)):
        subcategories: List[Subcategory] = []
        for subcategory_index in range(int(subcategory_count)):
            item_count = int(item_count_support[int(rng.randrange(len(item_count_support)))])
            items: List[CategoryItem] = []
            for item_index in range(int(item_count)):
                items.append(
                    CategoryItem(
                        item_id=(
                            f"category_{int(category_index) + 1}_subcategory_{int(subcategory_index) + 1}"
                            f"_item_{int(item_index) + 1}"
                        ),
                        label=str(item_labels[int(item_cursor)]),
                    )
                )
                item_cursor += 1
            subcategories.append(
                Subcategory(
                    subcategory_id=f"category_{int(category_index) + 1}_subcategory_{int(subcategory_index) + 1}",
                    label=str(subcategory_titles[int(subcategory_cursor)]),
                    items=tuple(items),
                )
            )
            subcategory_cursor += 1
        categories.append(
            Category(
                category_id=f"category_{int(category_index) + 1}",
                label=str(category_titles[int(category_index)]),
                accent_rgb=tuple(int(value) for value in ACCENTS[(int(category_index) + int(accent_offset)) % len(ACCENTS)]),
                subcategories=tuple(subcategories),
            )
        )
    return CategoryGridSpec(
        title=str(title_batch.values[0]),
        subtitle=str(subtitle_batch.values[0]),
        categories=tuple(categories),
        text_resource_metadata=page_text_resource_metadata(
            title_batch,
            subtitle_batch,
            category_batch,
            subcategory_batch,
            item_batch,
        ),
    )


def select_target(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    spec: CategoryGridSpec,
    include_slot: bool,
) -> Tuple[Category, Subcategory, int | None, CategoryItem | None]:
    """Select the task target category, subcategory, and optional item slot."""

    category_count = len(spec.categories)
    category_index = int(params["target_category_index"]) if params.get("target_category_index") is not None else int(
        resolve_selection_index(
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{NAMESPACE_ROOT}.target_category",
        )
        % int(category_count)
    )
    if category_index < 0 or category_index >= int(category_count):
        raise ValueError("target_category_index out of range")
    target_category = spec.categories[int(category_index)]
    subcategory_count = len(target_category.subcategories)
    subcategory_index = (
        int(params["target_subcategory_index"])
        if params.get("target_subcategory_index") is not None
        else int(
            resolve_selection_index(
                params=params,
                instance_seed=int(instance_seed),
                namespace=f"{NAMESPACE_ROOT}.target_subcategory.{category_index}",
            )
            % int(subcategory_count)
        )
    )
    if subcategory_index < 0 or subcategory_index >= int(subcategory_count):
        raise ValueError("target_subcategory_index out of range")
    target_subcategory = target_category.subcategories[int(subcategory_index)]
    if not bool(include_slot):
        return target_category, target_subcategory, None, None
    item_count = len(target_subcategory.items)
    slot_index = int(params["target_slot_index"]) if params.get("target_slot_index") is not None else int(
        resolve_selection_index(
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{NAMESPACE_ROOT}.target_slot.{category_index}.{subcategory_index}",
        )
        % int(item_count)
    )
    if slot_index < 0 or slot_index >= int(item_count):
        raise ValueError("target_slot_index out of range")
    return target_category, target_subcategory, int(slot_index), target_subcategory.items[int(slot_index)]


def build_category_grid_case(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    include_slot: bool,
) -> CategoryGridCase:
    """Resolve one complete category-grid scene and target selection."""

    scene_variant, scene_variant_probabilities = resolve_named_axis(
        params=params,
        instance_seed=int(instance_seed),
        supported=SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        namespace="scene_variant",
    )
    category_count, category_count_support, category_count_probabilities = resolve_supported_int(
        params=params,
        explicit_key="category_count",
        support_key="category_count_support",
        fallback=DEFAULTS.category_count_support,
        instance_seed=int(instance_seed),
        namespace="category_count",
    )
    subcategory_count, subcategory_count_support, subcategory_count_probabilities = resolve_supported_int(
        params=params,
        explicit_key="subcategory_count",
        support_key="subcategory_count_support",
        fallback=DEFAULTS.subcategory_count_support,
        instance_seed=int(instance_seed),
        namespace="subcategory_count",
    )
    item_count_support = resolve_int_support(params, "item_count_support", DEFAULTS.item_count_support)
    spec = build_category_grid_spec(
        category_count=int(category_count),
        subcategory_count=int(subcategory_count),
        item_count_support=tuple(item_count_support),
        instance_seed=int(instance_seed),
    )
    target_category, target_subcategory, target_slot_index, target_item = select_target(
        params=params,
        instance_seed=int(instance_seed),
        spec=spec,
        include_slot=bool(include_slot),
    )
    return CategoryGridCase(
        scene_variant=str(scene_variant),
        category_count=int(category_count),
        subcategory_count=int(subcategory_count),
        category_count_support=tuple(int(value) for value in category_count_support),
        subcategory_count_support=tuple(int(value) for value in subcategory_count_support),
        item_count_support=tuple(int(value) for value in item_count_support),
        spec=spec,
        target_category=target_category,
        target_subcategory=target_subcategory,
        target_slot_index=target_slot_index,
        target_item=target_item,
        scene_variant_probabilities=dict(scene_variant_probabilities),
        category_count_probabilities=dict(category_count_probabilities),
        subcategory_count_probabilities=dict(subcategory_count_probabilities),
    )


def build_slot_item_case(instance_seed: int, *, params: Mapping[str, Any]) -> CategoryGridCase:
    """Build a category-grid case with a target ordinal item."""

    return build_category_grid_case(int(instance_seed), params=params, include_slot=True)


def build_item_count_case(instance_seed: int, *, params: Mapping[str, Any]) -> CategoryGridCase:
    """Build a category-grid case with a target item-count block."""

    return build_category_grid_case(int(instance_seed), params=params, include_slot=False)
