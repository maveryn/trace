"""Neutral sampling primitives for pictogram tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .....core.sampling import uniform_choice
from .....core.seed import spawn_rng
from ....shared.config_defaults import group_default, resolve_required_int_bounds
from ...shared.label_assets import resolve_chart_category_labels

from .defaults import (
    GEN_DEFAULTS,
    RENDER_DEFAULTS,
    SCENE_NAMESPACE,
    resolve_glyph,
    resolve_scene_variant,
    sample_balanced_choice,
    sample_balanced_int,
    support_probability_map,
)
from .state import PictogramBaseSample, PictogramCategory, RGB


def resolve_mark_bounds(params: Mapping[str, object]) -> tuple[int, int]:
    mark_min, mark_max = resolve_required_int_bounds(
        params,
        GEN_DEFAULTS,
        min_key="mark_count_min",
        max_key="mark_count_max",
        fallback_min=4,
        fallback_max=16,
        context=f"generation defaults for {SCENE_NAMESPACE}",
    )
    if int(mark_min) < 2:
        raise ValueError("mark_count_min must be at least 2")
    if int(mark_max) <= int(mark_min):
        raise ValueError("mark_count_max must exceed mark_count_min")
    return int(mark_min), int(mark_max)


def resolve_unit_scale(params: Mapping[str, object], *, instance_seed: int) -> tuple[int, dict[str, float]]:
    raw_values = params.get("unit_scale_values", group_default(GEN_DEFAULTS, "unit_scale_values", [1, 2, 3, 4, 5]))
    support = [int(value) for value in raw_values]  # type: ignore[arg-type]
    if not support:
        raise ValueError("unit_scale_values must contain at least one integer")
    explicit = params.get("unit_scale")
    if explicit is not None:
        selected = int(explicit)
        if selected not in set(support):
            raise ValueError(f"unsupported unit_scale: {selected}")
        return selected, support_probability_map(support, selected=selected)
    selected = sample_balanced_choice(
        support,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.unit_scale",
    )
    return selected, support_probability_map(support)


def category_palette(params: Mapping[str, object]) -> tuple[RGB, ...]:
    default_palette: list[RGB] = [
        (37, 99, 235),
        (220, 84, 45),
        (16, 132, 96),
        (139, 92, 246),
        (202, 138, 4),
        (14, 116, 144),
        (190, 58, 90),
        (86, 105, 38),
        (91, 76, 181),
        (181, 94, 36),
    ]
    raw = params.get("category_palette_rgb", group_default(RENDER_DEFAULTS, "category_palette_rgb", ()))
    palette = [
        (int(item[0]), int(item[1]), int(item[2]))
        for item in raw  # type: ignore[union-attr]
        if isinstance(item, Sequence) and len(item) == 3
    ]
    if not palette:
        palette = list(default_palette)
    for fallback in default_palette:
        if len(palette) >= len(default_palette):
            break
        if fallback not in palette:
            palette.append(fallback)
    return tuple(palette)


def resolve_categories(
    *,
    mark_counts: Sequence[int],
    unit_scale: int,
    params: Mapping[str, object],
    instance_seed: int,
) -> tuple[PictogramCategory, ...]:
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.labels")
    labels = list(
        resolve_chart_category_labels(
            rng,
            count=len(mark_counts),
            min_chars=2,
            max_chars=8,
            allow_spaces=False,
        ).labels
    )
    palette = category_palette(params)
    categories: list[PictogramCategory] = []
    for index, mark_count in enumerate(mark_counts):
        color = palette[index]
        categories.append(
            PictogramCategory(
                category_id=f"cat_{index}",
                label=str(labels[index]),
                mark_count=int(mark_count),
                total=int(mark_count) * int(unit_scale),
                color_rgb=(int(color[0]), int(color[1]), int(color[2])),
            )
        )
    return tuple(categories)


def sample_base(params: Mapping[str, object], *, instance_seed: int) -> PictogramBaseSample:
    """Sample task-neutral pictogram frame axes before objective binding."""

    scene_variant, scene_variant_probabilities = resolve_scene_variant(params, instance_seed=int(instance_seed))
    glyph, glyph_probabilities = resolve_glyph(params, instance_seed=int(instance_seed))
    category_min, category_max = resolve_required_int_bounds(
        params,
        GEN_DEFAULTS,
        min_key="category_count_min",
        max_key="category_count_max",
        fallback_min=6,
        fallback_max=10,
        context=f"generation defaults for {SCENE_NAMESPACE}",
    )
    mark_min, mark_max = resolve_mark_bounds(params)
    category_count = sample_balanced_int(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.category_count",
        low=int(category_min),
        high=int(category_max),
    )
    unit_scale, unit_scale_probabilities = resolve_unit_scale(params, instance_seed=int(instance_seed))
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.base_counts")
    mark_counts = tuple(rng.randint(int(mark_min), int(mark_max)) for _ in range(int(category_count)))
    title_options = ("Unit Quantity Chart", "Category Unit Chart", "Pictogram Totals", "Repeated-Mark Summary")
    title = str(
        uniform_choice(
            spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.title"),
            title_options,
        )
    )
    return PictogramBaseSample(
        scene_variant=str(scene_variant),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        glyph_name=str(glyph),
        glyph_probabilities=dict(glyph_probabilities),
        unit_scale=int(unit_scale),
        unit_scale_probabilities=dict(unit_scale_probabilities),
        category_count_range=(int(category_min), int(category_max)),
        mark_count_range=(int(mark_min), int(mark_max)),
        mark_counts=tuple(int(value) for value in mark_counts),
        title=str(title),
    )


def categories_by_id(categories: Sequence[PictogramCategory]) -> dict[str, PictogramCategory]:
    return {str(category.category_id): category for category in categories}
