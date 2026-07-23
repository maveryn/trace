"""Scene-private lifecycle helpers for mixed infographic page tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from PIL import Image

from trace_tasks.core.seed import hash64
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.pages.shared.information_style import make_pages_information_background, resolve_pages_information_style
from trace_tasks.tasks.pages.shared.page_visual_assets import page_visual_asset_version
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_query_spec,
    build_prompt_trace_artifacts,
    render_task_prompt_variants,
)

from .shared.assets import _build_mixed_spec
from .shared.defaults import (
    DOMAIN,
    GENERATION_DEFAULTS,
    NAMESPACE_ROOT,
    POST_IMAGE_NOISE_DEFAULTS,
    PROMPT_BUNDLE,
    PROMPT_SCENE_KEY,
    PROMPT_TASK_KEY,
    RENDERING_DEFAULTS,
    SCENE,
    SCENE_VARIANTS,
)
from .shared.layout import (
    _RenderParams,
    _resolve_native_layout_mode,
    _resolve_native_text_block_count,
    _resolve_render_params,
)
from .shared.rendering import (
    _MixedInfographicLayoutError,
    _RenderedMixedInfographic,
    _render_mixed_infographic,
    _resolve_mixed_font_profile,
)
from .shared.sampling import (
    resolve_int_support,
    resolve_named_variant,
    resolve_supported_int,
)
from .shared.state import (
    ADDITIVE_FIELD_LABELS,
    NUMERIC_FIELD_LABELS,
    _MixedInfographicSpec,
)


MIN_PUBLIC_BBOX_ANNOTATION_SIDE_PX = 24.0
_ORDINAL_WORDS = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
    11: "eleventh",
    12: "twelfth",
}


@dataclass(frozen=True)
class MixedSceneContext:
    """Rendered scene and metadata shared by one task instance."""

    gen_defaults: Dict[str, Any]
    render_defaults: Dict[str, Any]
    scene_variant: str
    scene_variant_probabilities: Dict[str, float]
    native_layout_mode: str
    native_layout_mode_probabilities: Dict[str, float]
    module_count: int
    module_count_support: Tuple[int, ...]
    module_count_probabilities: Dict[str, float]
    item_count_support: Tuple[int, ...]
    field_count_support: Tuple[int, ...]
    native_text_block_count: int
    native_text_block_count_support: Tuple[int, ...]
    native_text_block_count_probabilities: Dict[str, float]
    spec: _MixedInfographicSpec
    rendered: _RenderedMixedInfographic
    image: Image.Image
    render_params: _RenderParams
    background_meta: Dict[str, Any]
    style_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]
    modules_payload: Tuple[Dict[str, Any], ...]


@dataclass(frozen=True)
class BoundTaskPayload:
    """Task-owned answer, annotation, prompt slots, and trace extras."""

    answer_type: str
    answer_value: Any
    annotation_type: str
    annotation_value: Any
    prompt_slots: Mapping[str, Any]
    target_payload: Mapping[str, Any]
    annotation_keys: Sequence[str]
    task_params_extra: Mapping[str, Any]
    execution_extra: Mapping[str, Any]
    witness_type: str
    diagnostic_bbox_map: Mapping[str, Sequence[float]] | None = None


TargetBinder = Callable[[MixedSceneContext, Mapping[str, Any], int], BoundTaskPayload]


def task_generation_defaults(
    *,
    module_count_support: Sequence[int],
    item_count_support: Sequence[int],
    field_count_support: Sequence[int],
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return scene defaults overlaid with public-task operand supports."""

    defaults = dict(GENERATION_DEFAULTS)
    defaults.update(
        {
            "module_count_support": [int(value) for value in module_count_support],
            "item_count_support": [int(value) for value in item_count_support],
            "field_count_support": [int(value) for value in field_count_support],
        }
    )
    defaults.update(dict(extra or {}))
    return defaults


def task_render_defaults(
    *,
    native_text_block_count_support: Sequence[int] = (4, 5),
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return scene render defaults overlaid with task-safe render supports."""

    defaults = dict(RENDERING_DEFAULTS)
    defaults["native_text_block_count_support"] = [
        int(value) for value in native_text_block_count_support
    ]
    defaults.update(dict(extra or {}))
    return defaults


def select_public_branch(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    supported: tuple[str, ...],
    default: str,
    public_task: str,
) -> tuple[str, Dict[str, float], Dict[str, Any]]:
    """Resolve the public branch through the shared single-query policy."""

    branch, probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in supported),
        default_query_id=str(default),
        task_id=str(public_task),
    )
    return str(branch), dict(probabilities), dict(task_params)


def _build_modules_payload(
    *,
    spec: _MixedInfographicSpec,
    rendered: _RenderedMixedInfographic,
) -> Tuple[Dict[str, Any], ...]:
    """Flatten rendered module metadata; ids must match bbox maps used by annotations."""

    modules_payload: List[Dict[str, Any]] = []
    for module in spec.modules:
        module_id = str(module.module_id)
        modules_payload.append(
            {
                "module_id": module_id,
                "module_title": str(module.title),
                "module_kind": str(module.kind),
                "bbox_px": [float(value) for value in rendered.module_bboxes_px[module_id]],
                "title_bbox_px": [
                    float(value) for value in rendered.module_title_bboxes_px[module_id]
                ],
                "section_visual_asset": module.section_asset_selection.to_metadata(),
                "section_visual_asset_bbox_px": [
                    float(value) for value in rendered.section_asset_bboxes_px[module_id]
                ],
                "fields": [
                    {
                        "field_id": str(field.field_id),
                        "field_label": str(field.label),
                        "bbox_px": [
                            float(value)
                            for value in rendered.field_label_bboxes_px[module_id][
                                str(field.field_id)
                            ]
                        ],
                    }
                    for field in module.fields
                ],
                "items": [
                    {
                        "item_id": str(item.item_id),
                        "item_label": str(item.label),
                        "visual_asset": item.visual_asset_selection.to_metadata(),
                        "label_bbox_px": [
                            float(value)
                            for value in rendered.item_label_bboxes_px[module_id][
                                str(item.item_id)
                            ]
                        ],
                        "item_container_bbox_px": [
                            float(value)
                            for value in rendered.item_container_bboxes_px[module_id][
                                str(item.item_id)
                            ]
                        ],
                        "visual_asset_bbox_px": [
                            float(value)
                            for value in rendered.icon_bboxes_px[module_id][str(item.item_id)]
                        ],
                        "values_by_field_id": dict(item.values_by_field_id),
                        "value_bboxes_px": dict(
                            rendered.value_cell_bboxes_px[module_id][str(item.item_id)]
                        ),
                    }
                    for item in module.items
                ],
            }
        )
    return tuple(modules_payload)


def _ordinal_word(value: int) -> str:
    """Return a compact ordinal phrase for page-position prompt slots."""

    ivalue = int(value)
    if ivalue in _ORDINAL_WORDS:
        return _ORDINAL_WORDS[ivalue]
    suffix = "th"
    if ivalue % 100 not in (11, 12, 13):
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(ivalue % 10, "th")
    return f"{ivalue}{suffix}"


def _module_reading_order(
    *,
    ctx: MixedSceneContext,
) -> Tuple[str, ...]:
    """Order modules by visual reading order from top to bottom, then left to right."""

    entries: List[Tuple[float, float, str]] = []
    for module in ctx.spec.modules:
        module_id = str(module.module_id)
        bbox = ctx.rendered.module_bboxes_px[module_id]
        cx = (float(bbox[0]) + float(bbox[2])) / 2.0
        cy = (float(bbox[1]) + float(bbox[3])) / 2.0
        entries.append((cy, cx, module_id))
    return tuple(module_id for _, _, module_id in sorted(entries))


def _module_position_descriptor(
    *,
    ctx: MixedSceneContext,
    module_id: str,
) -> Tuple[str, int, Tuple[str, ...]]:
    """Build the visible-position locator used by the value lookup prompt."""

    order = _module_reading_order(ctx=ctx)
    position_index = order.index(str(module_id)) + 1
    descriptor = (
        f"{_ordinal_word(position_index)} module in reading order "
        "(top to bottom, left to right)"
    )
    return descriptor, position_index, order


def build_scene_context(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    sampling_namespace: str,
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    allow_categorical_value_reuse: bool = False,
    ensure_shared_numeric_field: bool = False,
    shared_numeric_field_label: str | None = None,
    shared_numeric_field_choices: Sequence[str] | None = None,
) -> MixedSceneContext:
    """Sample, render, and package one mixed infographic scene."""

    scene_variant, scene_variant_probabilities = resolve_named_variant(
        sampling_namespace=str(sampling_namespace),
        gen_defaults=gen_defaults,
        params=params,
        instance_seed=int(instance_seed),
        supported=SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        namespace="scene_variant",
    )
    native_layout_mode, native_layout_mode_probabilities = _resolve_native_layout_mode(
        sampling_namespace=str(sampling_namespace),
        params=params,
        render_defaults=render_defaults,
        instance_seed=int(instance_seed),
    )
    module_count, module_count_support, module_count_probabilities = resolve_supported_int(
        sampling_namespace=str(sampling_namespace),
        params=params,
        gen_defaults=gen_defaults,
        explicit_key="module_count",
        support_key="module_count_support",
        fallback=(7, 8, 9),
        instance_seed=int(instance_seed),
        namespace="module_count",
    )
    item_count_support = resolve_int_support(
        params=params,
        gen_defaults=gen_defaults,
        key="item_count_support",
        fallback=(2, 3, 4),
    )
    field_count_support = resolve_int_support(
        params=params,
        gen_defaults=gen_defaults,
        key="field_count_support",
        fallback=(2, 3),
    )
    native_text_count, native_text_support, native_text_probabilities = (
        _resolve_native_text_block_count(
            sampling_namespace=str(sampling_namespace),
            params=params,
            render_defaults=render_defaults,
            instance_seed=int(instance_seed),
        )
    )
    shared_choices = shared_numeric_field_choices or NUMERIC_FIELD_LABELS
    if ensure_shared_numeric_field and shared_numeric_field_choices is None:
        shared_choices = NUMERIC_FIELD_LABELS
    spec = _build_mixed_spec(
        module_count=int(module_count),
        item_count_support=tuple(item_count_support),
        field_count_support=tuple(field_count_support),
        native_text_block_count=int(native_text_count),
        instance_seed=int(instance_seed),
        resource_namespace=NAMESPACE_ROOT,
        allow_categorical_value_reuse=bool(allow_categorical_value_reuse),
        ensure_shared_numeric_field=bool(ensure_shared_numeric_field),
        shared_numeric_field_label=shared_numeric_field_label,
        shared_numeric_field_choices=tuple(str(value) for value in shared_choices),
    )
    font_profile = _resolve_mixed_font_profile(
        modules=spec.modules,
        params=params,
        instance_seed=int(instance_seed),
    )
    render_params = _resolve_render_params(params, render_defaults)
    style, style_meta = resolve_pages_information_style(
        instance_seed=int(instance_seed),
        params={**dict(render_defaults), **dict(params or {})},
        scene_id=SCENE,
    )
    background, background_meta = make_pages_information_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=style,
        instance_seed=int(instance_seed),
        namespace=f"{NAMESPACE_ROOT}.information_scene_background",
    )
    rendered = _render_mixed_infographic(
        background,
        spec=spec,
        scene_variant=str(scene_variant),
        native_layout_mode=str(native_layout_mode),
        style=style,
        render_params=render_params,
        instance_seed=int(instance_seed),
        font_profile=font_profile,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return MixedSceneContext(
        gen_defaults=dict(gen_defaults),
        render_defaults=dict(render_defaults),
        scene_variant=str(scene_variant),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        native_layout_mode=str(native_layout_mode),
        native_layout_mode_probabilities=dict(native_layout_mode_probabilities),
        module_count=int(module_count),
        module_count_support=tuple(int(value) for value in module_count_support),
        module_count_probabilities=dict(module_count_probabilities),
        item_count_support=tuple(int(value) for value in item_count_support),
        field_count_support=tuple(int(value) for value in field_count_support),
        native_text_block_count=int(native_text_count),
        native_text_block_count_support=tuple(int(value) for value in native_text_support),
        native_text_block_count_probabilities=dict(native_text_probabilities),
        spec=spec,
        rendered=rendered,
        image=image,
        render_params=render_params,
        background_meta=dict(background_meta),
        style_meta=dict(style_meta),
        post_noise_meta=dict(post_noise_meta),
        modules_payload=_build_modules_payload(spec=spec, rendered=rendered),
    )


def render_prompt(
    *,
    prompt_key: str,
    dynamic_slots: Mapping[str, Any],
    instance_seed: int,
) -> Any:
    """Render prompt variants for one task-owned prompt key."""

    prompt_selection = render_task_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE,
        bundle_id=PROMPT_BUNDLE,
        scene_key=PROMPT_SCENE_KEY,
        task_key=PROMPT_TASK_KEY,
        query_key=str(prompt_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=dict(dynamic_slots),
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(prompt_selection)


def _normalized_bbox(value: Sequence[float]) -> List[float]:
    """Normalize one pixel bbox for trace/review payloads."""

    return [float(coord) for coord in value]


def _normalized_bbox_map(value: Mapping[str, Sequence[float]] | None) -> Dict[str, List[float]]:
    """Normalize optional diagnostic bbox maps."""

    return {
        str(key): _normalized_bbox(bbox)
        for key, bbox in dict(value or {}).items()
    }


def projected_annotation(
    annotation_type: str,
    annotation_value: Any,
    *,
    diagnostic_bbox_map: Mapping[str, Sequence[float]] | None = None,
) -> Dict[str, Any]:
    """Project public annotations while keeping diagnostic role boxes separate."""

    diagnostics = _normalized_bbox_map(diagnostic_bbox_map)
    if str(annotation_type) == "bbox":
        bbox = _normalized_bbox(annotation_value)
        payload = {
            "type": "bbox",
            "bbox": list(bbox),
            "pixel_bbox": list(bbox),
            "bbox_set": [list(bbox)],
            "pixel_bbox_set": [list(bbox)],
        }
        if diagnostics:
            payload["bbox_map"] = dict(diagnostics)
            payload["pixel_bbox_map"] = dict(diagnostics)
        return payload
    if str(annotation_type) == "bbox_map":
        keyed = _normalized_bbox_map(annotation_value)
        return {
            "type": "bbox_map",
            "bbox_map": dict(keyed),
            "pixel_bbox_map": dict(keyed),
            "bbox_set": list(keyed.values()),
            "pixel_bbox_set": list(keyed.values()),
        }
    bbox_set = [_normalized_bbox(bbox) for bbox in list(annotation_value)]
    payload = {
        "type": "bbox_set",
        "bbox_set": list(bbox_set),
        "pixel_bbox_set": list(bbox_set),
    }
    if diagnostics:
        payload["bbox_map"] = dict(diagnostics)
        payload["pixel_bbox_map"] = dict(diagnostics)
    return payload


def build_trace_payload(
    *,
    ctx: MixedSceneContext,
    prompt_artifacts: Any,
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    prompt_key: str,
    target_payload: Mapping[str, Any],
    answer_value: Any,
    annotation_type: str,
    annotation_value: Any,
    witness_type: str,
    annotation_keys: Sequence[str],
    task_params_extra: Mapping[str, Any] | None = None,
    execution_extra: Mapping[str, Any] | None = None,
    diagnostic_bbox_map: Mapping[str, Sequence[float]] | None = None,
) -> Dict[str, Any]:
    """Build trace payload from a rendered scene and task-bound target."""

    target = dict(target_payload)
    branch_probs = {str(key): float(value) for key, value in branch_probabilities.items()}
    style_meta = dict(ctx.style_meta)
    surface_style = style_meta.get("surface_style", {})
    if not isinstance(surface_style, Mapping):
        surface_style = {}
    information_style_fields = {
        "information_scene_treatment": str(style_meta.get("treatment", "")),
        "information_scene_palette_id": str(style_meta.get("palette_id", "")),
        "information_scene_style_pack": str(style_meta.get("style_pack", "")),
        "information_scene_surface_kind": str(surface_style.get("kind", "")),
        "information_scene_chrome_mode": str(surface_style.get("chrome_mode", "")),
    }
    params_payload = {
        "query_id": str(selected_branch),
        "prompt_query_key": str(prompt_key),
        "source_query_id": str(prompt_key),
        "scene_variant": str(ctx.scene_variant),
        "native_layout_mode": str(ctx.native_layout_mode),
        **information_style_fields,
        "module_count": int(ctx.module_count),
        "native_text_block_count": int(ctx.native_text_block_count),
        "target": dict(target),
        "target_answer": answer_value,
        "query_id_probabilities": dict(branch_probs),
        "scene_variant_probabilities": dict(ctx.scene_variant_probabilities),
        "native_layout_mode_probabilities": dict(ctx.native_layout_mode_probabilities),
        "module_count_probabilities": dict(ctx.module_count_probabilities),
        "native_text_block_count_probabilities": dict(
            ctx.native_text_block_count_probabilities
        ),
    }
    params_payload.update(dict(task_params_extra or {}))
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_branch),
        params=params_payload,
    )
    query_spec["scene_id"] = SCENE

    execution_trace = {
        "query_id": str(selected_branch),
        "prompt_query_key": str(prompt_key),
        "source_query_id": str(prompt_key),
        "scene_variant": str(ctx.scene_variant),
        "native_layout_mode": str(ctx.native_layout_mode),
        **information_style_fields,
        "question_format": str(prompt_key),
        "module_count": int(ctx.module_count),
        "module_count_support": [int(value) for value in ctx.module_count_support],
        "item_count_support": [int(value) for value in ctx.item_count_support],
        "field_count_support": [int(value) for value in ctx.field_count_support],
        "native_text_block_count": int(ctx.native_text_block_count),
        "native_text_block_count_support": [
            int(value) for value in ctx.native_text_block_count_support
        ],
        "target": dict(target),
        "answer_value": answer_value,
        "modules": [dict(module) for module in ctx.modules_payload],
        "page_text_resources": dict(ctx.spec.text_resource_metadata),
        "query_id_probabilities": dict(branch_probs),
        "scene_variant_probabilities": dict(ctx.scene_variant_probabilities),
        "native_layout_mode_probabilities": dict(ctx.native_layout_mode_probabilities),
        "module_count_probabilities": dict(ctx.module_count_probabilities),
        "native_text_block_count_probabilities": dict(
            ctx.native_text_block_count_probabilities
        ),
    }
    execution_trace.update(dict(execution_extra or {}))
    return {
        "scene_ir": {
            "scene_id": SCENE,
            "scene_kind": "pages_mixed_infographic_page",
            "entities": [dict(entity) for entity in ctx.rendered.entities],
            "relations": {
                "query_id": str(selected_branch),
                "prompt_query_key": str(prompt_key),
                "source_query_id": str(prompt_key),
                "scene_variant": str(ctx.scene_variant),
                "native_layout_mode": str(ctx.native_layout_mode),
                **information_style_fields,
                "target": dict(target),
                "answer_value": answer_value,
            },
        },
        "query_spec": query_spec,
        "render_spec": {
            "canvas_width": int(ctx.render_params.canvas_width),
            "canvas_height": int(ctx.render_params.canvas_height),
            "coord_space": "pixel",
            "scene_id": SCENE,
            "scene_variant": str(ctx.scene_variant),
            "native_layout_mode": str(ctx.native_layout_mode),
            "background_style": dict(ctx.background_meta),
            "information_scene_style": dict(ctx.style_meta),
            "post_image_noise": dict(ctx.post_noise_meta),
            "page_bbox_px": list(ctx.rendered.page_bbox_px),
            "layout": dict(ctx.rendered.layout_meta),
            "infographic_text_blocks": list(ctx.rendered.text_blocks),
            "font_assets": {
                "mixed_infographic_font_profile": dict(ctx.rendered.font_profile_meta),
            },
            "page_text_resources": dict(ctx.spec.text_resource_metadata),
            "module_kinds": [str(module.kind) for module in ctx.spec.modules],
            "page_visual_assets": {
                "asset_version": page_visual_asset_version(),
                "asset_root": "assets/pages/visual_assets",
                "semantic_policy": "non_answer_visual_context",
                "hero_anchor_drawn": "hero_anchor" in ctx.rendered.decorative_asset_bboxes_px,
                "roles": {
                    "hero_anchor": ctx.spec.hero_asset_selection.to_metadata(),
                    "module_section_assets": {
                        str(module.module_id): module.section_asset_selection.to_metadata()
                        for module in ctx.spec.modules
                    },
                    "item_badge_assets": {
                        str(module.module_id): {
                            str(item.item_id): item.visual_asset_selection.to_metadata()
                            for item in module.items
                        }
                        for module in ctx.spec.modules
                    },
                },
            },
        },
        "render_map": {
            "image_id": "img0",
            "page_bbox_px": list(ctx.rendered.page_bbox_px),
            "module_bboxes_px": dict(ctx.rendered.module_bboxes_px),
            "module_title_bboxes_px": dict(ctx.rendered.module_title_bboxes_px),
            "item_label_bboxes_px": dict(ctx.rendered.item_label_bboxes_px),
            "item_container_bboxes_px": dict(ctx.rendered.item_container_bboxes_px),
            "field_label_bboxes_px": dict(ctx.rendered.field_label_bboxes_px),
            "value_cell_bboxes_px": dict(ctx.rendered.value_cell_bboxes_px),
            "icon_bboxes_px": dict(ctx.rendered.icon_bboxes_px),
            "visual_asset_bboxes_px": {
                "hero_anchor": dict(ctx.rendered.decorative_asset_bboxes_px),
                "module_section_assets": dict(ctx.rendered.section_asset_bboxes_px),
                "item_badge_assets": dict(ctx.rendered.icon_bboxes_px),
            },
            "infographic_text_block_bboxes_px": dict(ctx.rendered.text_block_bboxes_px),
        },
        "execution_trace": dict(execution_trace),
        "witness_symbolic": {
            "type": str(witness_type),
            "target": dict(target),
            "answer_value": answer_value,
            "annotation_keys": [str(key) for key in annotation_keys],
        },
        "projected_annotation": projected_annotation(
            str(annotation_type),
            annotation_value,
            diagnostic_bbox_map=diagnostic_bbox_map,
        ),
    }


def build_task_output(
    *,
    prompt_artifacts: Any,
    answer_type: str,
    answer_value: Any,
    annotation_type: str,
    annotation_value: Any,
    ctx: MixedSceneContext,
    trace_payload: Mapping[str, Any],
    selected_branch: str,
) -> TaskOutput:
    """Construct the final public task output."""

    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=TypedValue(type=str(answer_type), value=answer_value),
        annotation_gt=TypedValue(type=str(annotation_type), value=annotation_value),
        image=ctx.image,
        image_id="img0",
        trace_payload=dict(trace_payload),
        task_versions=default_task_versions(),
        scene_id=SCENE,
        query_id=str(selected_branch),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
    )


def _public_annotation_bboxes(annotation_type: str, annotation_value: Any) -> List[List[float]]:
    """Flatten public bbox annotations for minimum-size validation."""

    if str(annotation_type) == "bbox":
        return [_normalized_bbox(annotation_value)]
    if str(annotation_type) == "bbox_set":
        return [_normalized_bbox(bbox) for bbox in list(annotation_value)]
    if str(annotation_type) == "bbox_map":
        return [_normalized_bbox(bbox) for bbox in dict(annotation_value).values()]
    if str(annotation_type) == "bbox_set_map":
        bboxes: List[List[float]] = []
        for group in dict(annotation_value).values():
            bboxes.extend(_normalized_bbox(bbox) for bbox in list(group))
        return bboxes
    return []


def _bbox_min_side_px(bbox: Sequence[float]) -> float:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return float(min(max(0.0, x1 - x0), max(0.0, y1 - y0)))


def _public_annotation_min_side_px(annotation_type: str, annotation_value: Any) -> float | None:
    bboxes = _public_annotation_bboxes(str(annotation_type), annotation_value)
    if not bboxes:
        return None
    return min(_bbox_min_side_px(bbox) for bbox in bboxes)


def _attempt_seed(instance_seed: int, public_task: str, attempt_index: int) -> int:
    if int(attempt_index) <= 0:
        return int(instance_seed)
    return int(
        hash64(
            int(instance_seed),
            f"{str(public_task)}.mixed_infographic_annotation_retry",
            int(attempt_index),
        )
    )


def run_bound_task(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    public_task: str,
    supported_branches: tuple[str, ...],
    prompt_key: str,
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    bind_target: TargetBinder,
    context_options: Mapping[str, Any] | Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> TaskOutput:
    """Run common scene lifecycle around a public task's target binder."""

    selected_branch, branch_probabilities, task_params = select_public_branch(
        instance_seed=int(instance_seed),
        params=params,
        supported=tuple(str(value) for value in supported_branches),
        default=tuple(str(value) for value in supported_branches)[0],
        public_task=str(public_task),
    )
    resolved_context_options = (
        context_options(task_params)
        if callable(context_options)
        else dict(context_options or {})
    )
    attempts = max(1, int(max_attempts))
    min_required = float(MIN_PUBLIC_BBOX_ANNOTATION_SIDE_PX)
    last_min_side: float | None = None
    last_layout_error: _MixedInfographicLayoutError | None = None
    for attempt_index in range(int(attempts)):
        seed_for_attempt = _attempt_seed(
            int(instance_seed),
            str(public_task),
            int(attempt_index),
        )
        try:
            ctx = build_scene_context(
                params=task_params,
                instance_seed=int(seed_for_attempt),
                sampling_namespace=str(public_task),
                gen_defaults=gen_defaults,
                render_defaults=render_defaults,
                **dict(resolved_context_options),
            )
        except _MixedInfographicLayoutError as exc:
            last_layout_error = exc
            last_min_side = None
            continue
        bound = bind_target(ctx, task_params, int(seed_for_attempt))
        annotation_min_side = _public_annotation_min_side_px(
            str(bound.annotation_type),
            bound.annotation_value,
        )
        last_min_side = annotation_min_side
        if annotation_min_side is not None and float(annotation_min_side) < min_required:
            last_layout_error = None
            continue
        prompt_artifacts = render_prompt(
            prompt_key=str(prompt_key),
            dynamic_slots=dict(bound.prompt_slots),
            instance_seed=int(instance_seed),
        )
        annotation_attempt_meta = {
            "generation_attempt_index": int(attempt_index),
            "generation_attempt_seed": int(seed_for_attempt),
            "annotation_min_side_px": (
                None if annotation_min_side is None else float(annotation_min_side)
            ),
            "annotation_min_side_required_px": float(min_required),
        }
        trace_payload = build_trace_payload(
            ctx=ctx,
            prompt_artifacts=prompt_artifacts,
            selected_branch=str(selected_branch),
            branch_probabilities=branch_probabilities,
            prompt_key=str(prompt_key),
            target_payload=dict(bound.target_payload),
            answer_value=bound.answer_value,
            annotation_type=str(bound.annotation_type),
            annotation_value=bound.annotation_value,
            witness_type=str(bound.witness_type),
            annotation_keys=tuple(str(value) for value in bound.annotation_keys),
            task_params_extra={
                **dict(bound.task_params_extra),
                **dict(annotation_attempt_meta),
            },
            execution_extra={
                **dict(bound.execution_extra),
                **dict(annotation_attempt_meta),
            },
            diagnostic_bbox_map=bound.diagnostic_bbox_map,
        )
        return build_task_output(
            prompt_artifacts=prompt_artifacts,
            answer_type=str(bound.answer_type),
            answer_value=bound.answer_value,
            annotation_type=str(bound.annotation_type),
            annotation_value=bound.annotation_value,
            ctx=ctx,
            trace_payload=trace_payload,
            selected_branch=str(selected_branch),
        )
    if last_layout_error is not None:
        raise RuntimeError(
            f"{public_task} failed to build and render a valid mixed infographic "
            f"after {attempts} attempts"
        ) from last_layout_error
    raise ValueError(
        "mixed infographic public annotation bbox below minimum side "
        f"{min_required:.1f}px after {attempts} attempts; last min side={last_min_side}"
    )


def module_value_rank_annotation(
    *,
    ctx: MixedSceneContext,
    module_id: str,
    field_id: str,
    item_id: str,
    item_key: str,
    value_key: str,
    candidates: Sequence[Mapping[str, Any]],
) -> Dict[str, List[float]]:
    """Project one module-rank answer and compared value cells to bboxes."""

    annotation = {
        "module_title": [float(value) for value in ctx.rendered.module_title_bboxes_px[module_id]],
        "field_label": [
            float(value) for value in ctx.rendered.field_label_bboxes_px[module_id][field_id]
        ],
        str(item_key): [
            float(value)
            for value in ctx.rendered.item_container_bboxes_px[module_id][item_id]
        ],
        f"{str(item_key)}_label": [
            float(value) for value in ctx.rendered.item_label_bboxes_px[module_id][item_id]
        ],
        str(value_key): [
            float(value)
            for value in ctx.rendered.value_cell_bboxes_px[module_id][item_id][field_id]
        ],
    }
    for index, candidate in enumerate(candidates, start=1):
        annotation[f"candidate_{index}"] = [
            float(value)
            for value in ctx.rendered.value_cell_bboxes_px[module_id][
                str(candidate["item_id"])
            ][field_id]
        ]
    return annotation


def module_field_value_payload(
    *,
    ctx: MixedSceneContext,
    target_module: Any,
    target_item: Any,
    target_field: Any,
    module_probs: Mapping[str, float],
    item_probs: Mapping[str, float],
    field_probs: Mapping[str, float],
    prompt_key: str,
) -> BoundTaskPayload:
    """Bind one item/field value lookup; public annotation marks the answer cell."""

    answer_value = str(target_item.values_by_field_id[str(target_field.field_id)])
    module_id = str(target_module.module_id)
    item_id = str(target_item.item_id)
    field_id = str(target_field.field_id)
    module_position_phrase, module_position_index, module_reading_order = (
        _module_position_descriptor(ctx=ctx, module_id=module_id)
    )
    annotation_value = {
        "module_title": [float(value) for value in ctx.rendered.module_title_bboxes_px[module_id]],
        "item_label": [
            float(value) for value in ctx.rendered.item_label_bboxes_px[module_id][item_id]
        ],
        "item_container": [
            float(value)
            for value in ctx.rendered.item_container_bboxes_px[module_id][item_id]
        ],
        "field_label": [
            float(value) for value in ctx.rendered.field_label_bboxes_px[module_id][field_id]
        ],
        "value_cell": [
            float(value)
            for value in ctx.rendered.value_cell_bboxes_px[module_id][item_id][field_id]
        ],
    }
    return BoundTaskPayload(
        answer_type="string",
        answer_value=str(answer_value),
        annotation_type="bbox",
        annotation_value=list(annotation_value["value_cell"]),
        prompt_slots={
            "module_position_phrase": str(module_position_phrase),
            "item_label": str(target_item.label),
            "field_label": str(target_field.label),
        },
        target_payload={
            "module_id": str(target_module.module_id),
            "module_title": str(target_module.title),
            "module_position_phrase": str(module_position_phrase),
            "module_position_index": int(module_position_index),
            "module_reading_order": list(module_reading_order),
            "module_kind": str(target_module.kind),
            "item_id": str(target_item.item_id),
            "item_label": str(target_item.label),
            "field_id": str(target_field.field_id),
            "field_label": str(target_field.label),
            "value": str(answer_value),
        },
        annotation_keys=("value_cell",),
        task_params_extra={
            "target_module_index_probabilities": dict(module_probs),
            "target_item_index_probabilities": dict(item_probs),
            "target_field_index_probabilities": dict(field_probs),
        },
        execution_extra={
            "module_position_phrase": str(module_position_phrase),
            "module_position_index": int(module_position_index),
            "module_reading_order": list(module_reading_order),
        },
        witness_type=str(prompt_key),
        diagnostic_bbox_map=dict(annotation_value),
    )


def module_ranked_field_value_payload(
    *,
    ctx: MixedSceneContext,
    target_module: Any,
    selector_field: Any,
    answer_field: Any,
    target: Mapping[str, Any],
    module_probs: Mapping[str, float],
    selector_field_probs: Mapping[str, float],
    answer_field_probs: Mapping[str, float],
    direction_probs: Mapping[str, float],
    rank_probs: Mapping[str, float],
    prompt_key: str,
) -> BoundTaskPayload:
    """Bind a ranked-item field lookup; public annotation marks the answer value cell."""

    module_id = str(target_module.module_id)
    item_id = str(target["item_id"])
    selector_field_id = str(selector_field.field_id)
    answer_field_id = str(answer_field.field_id)
    answer_value = str(target["answer_value"])
    module_position_phrase, module_position_index, module_reading_order = (
        _module_position_descriptor(ctx=ctx, module_id=module_id)
    )
    annotation_value = {
        "module_title": [float(value) for value in ctx.rendered.module_title_bboxes_px[module_id]],
        "selector_field_label": [
            float(value)
            for value in ctx.rendered.field_label_bboxes_px[module_id][selector_field_id]
        ],
        "answer_field_label": [
            float(value)
            for value in ctx.rendered.field_label_bboxes_px[module_id][answer_field_id]
        ],
        "ranked_item_container": [
            float(value)
            for value in ctx.rendered.item_container_bboxes_px[module_id][item_id]
        ],
        "ranked_item_label": [
            float(value) for value in ctx.rendered.item_label_bboxes_px[module_id][item_id]
        ],
        "ranked_selector_value": [
            float(value)
            for value in ctx.rendered.value_cell_bboxes_px[module_id][item_id][selector_field_id]
        ],
        "answer_value_cell": [
            float(value)
            for value in ctx.rendered.value_cell_bboxes_px[module_id][item_id][answer_field_id]
        ],
    }
    for index, candidate in enumerate(target["candidate_values"], start=1):
        annotation_value[f"rank_candidate_{index}"] = [
            float(value)
            for value in ctx.rendered.value_cell_bboxes_px[module_id][
                str(candidate["item_id"])
            ][selector_field_id]
        ]
    target_payload = dict(target)
    target_payload.update(
        {
            "module_position_phrase": str(module_position_phrase),
            "module_position_index": int(module_position_index),
            "module_reading_order": list(module_reading_order),
            "selector_field_id": str(selector_field_id),
            "selector_field_label": str(selector_field.label),
            "answer_field_id": str(answer_field_id),
            "answer_field_label": str(answer_field.label),
            "answer_value": str(answer_value),
        }
    )
    return BoundTaskPayload(
        answer_type="string",
        answer_value=str(answer_value),
        annotation_type="bbox",
        annotation_value=list(annotation_value["answer_value_cell"]),
        prompt_slots={
            "module_title": str(target_module.title),
            "selector_field_label": str(selector_field.label),
            "rank_ordinal": str(target["rank_ordinal"]),
            "rank_order_phrase": str(target["rank_order_phrase"]),
            "answer_field_label": str(answer_field.label),
        },
        target_payload=target_payload,
        annotation_keys=("answer_value_cell",),
        task_params_extra={
            "target_module_index_probabilities": dict(module_probs),
            "selector_field_index_probabilities": dict(selector_field_probs),
            "answer_field_index_probabilities": dict(answer_field_probs),
            "rank_direction_probabilities": dict(direction_probs),
            "rank_position_probabilities": dict(rank_probs),
        },
        execution_extra={
            "module_position_phrase": str(module_position_phrase),
            "module_position_index": int(module_position_index),
            "module_reading_order": list(module_reading_order),
            "rank_direction": str(target["rank_direction"]),
            "rank_position": int(target["rank_position"]),
            "rank_ordinal": str(target["rank_ordinal"]),
        },
        witness_type=str(prompt_key),
        diagnostic_bbox_map=dict(annotation_value),
    )


def module_rank_payload(
    *,
    ctx: MixedSceneContext,
    target_module: Any,
    target_field: Any,
    target: Mapping[str, Any],
    module_probs: Mapping[str, float],
    field_probs: Mapping[str, float],
    direction_probs: Mapping[str, float],
    prompt_key: str,
    item_key: str,
    value_key: str,
    rank_probs: Mapping[str, float] | None = None,
) -> BoundTaskPayload:
    """Bind one module-local rank target; public annotation marks the answer item."""

    annotation_value = module_value_rank_annotation(
        ctx=ctx,
        module_id=str(target_module.module_id),
        field_id=str(target_field.field_id),
        item_id=str(target["item_id"]),
        item_key=str(item_key),
        value_key=str(value_key),
        candidates=target["candidate_values"],
    )
    prompt_slots = {
        "module_title": str(target_module.title),
        "field_label": str(target_field.label),
        "rank_direction": str(target["rank_direction"]),
        "rank_order_phrase": str(target["rank_order_phrase"]),
    }
    task_params_extra = {
        "target_module_index_probabilities": dict(module_probs),
        "target_field_index_probabilities": dict(field_probs),
        "rank_direction_probabilities": dict(direction_probs),
    }
    execution_extra = {"rank_direction": str(target["rank_direction"])}
    if rank_probs is not None:
        prompt_slots["rank_ordinal"] = str(target["rank_ordinal"])
        task_params_extra["rank_position_probabilities"] = dict(rank_probs)
        execution_extra.update(
            {
                "rank_position": int(target["rank_position"]),
                "rank_ordinal": str(target["rank_ordinal"]),
            }
        )
    return BoundTaskPayload(
        answer_type="string",
        answer_value=str(target["item_label"]),
        annotation_type="bbox",
        annotation_value=list(annotation_value[str(item_key)]),
        prompt_slots=prompt_slots,
        target_payload=dict(target),
        annotation_keys=(str(item_key),),
        task_params_extra=task_params_extra,
        execution_extra=execution_extra,
        witness_type=str(prompt_key),
        diagnostic_bbox_map=dict(annotation_value),
    )


def page_field_extremum_payload(
    *,
    ctx: MixedSceneContext,
    target_module: Any,
    target_field: Any,
    target: Mapping[str, Any],
    field_probs: Mapping[str, float],
    direction_probs: Mapping[str, float],
    prompt_key: str,
) -> BoundTaskPayload:
    """Bind the page-wide field extremum; public annotation marks the answer module."""

    module_id = str(target_module.module_id)
    field_id = str(target_field.field_id)
    item_id = str(target["item_id"])
    annotation_value = {
        "winning_module_title": [
            float(value) for value in ctx.rendered.module_title_bboxes_px[module_id]
        ],
        "winning_field_label": [
            float(value) for value in ctx.rendered.field_label_bboxes_px[module_id][field_id]
        ],
        "winning_item": [
            float(value) for value in ctx.rendered.item_label_bboxes_px[module_id][item_id]
        ],
        "winning_value": [
            float(value)
            for value in ctx.rendered.value_cell_bboxes_px[module_id][item_id][field_id]
        ],
    }
    for index, candidate in enumerate(target["candidate_values"], start=1):
        annotation_value[f"candidate_{index}"] = [
            float(value)
            for value in ctx.rendered.value_cell_bboxes_px[str(candidate["module_id"])][
                str(candidate["item_id"])
            ][str(candidate["field_id"])]
        ]
    return BoundTaskPayload(
        answer_type="string",
        answer_value=str(target["answer_value"]),
        annotation_type="bbox",
        annotation_value=[
            float(value) for value in ctx.rendered.module_bboxes_px[module_id]
        ],
        prompt_slots={
            "field_label": str(target["field_label"]),
            "rank_direction": str(target["rank_direction"]),
            "rank_order_phrase": str(target["rank_order_phrase"]),
        },
        target_payload=dict(target),
        annotation_keys=("winning_module",),
        task_params_extra={
            "target_field_label_probabilities": dict(field_probs),
            "rank_direction_probabilities": dict(direction_probs),
        },
        execution_extra={
            "rank_direction": str(target["rank_direction"]),
            "target_field_label": str(target["field_label"]),
            "candidate_module_count": int(target["candidate_module_count"]),
        },
        witness_type=str(prompt_key),
        diagnostic_bbox_map=dict(annotation_value),
    )


def two_field_condition_payload(
    *,
    ctx: MixedSceneContext,
    target_module: Any,
    numeric_field: Any,
    category_field: Any,
    target: Mapping[str, Any],
    probability_payload: Mapping[str, Mapping[str, float]],
    prompt_key: str,
) -> BoundTaskPayload:
    """Bind the unique item satisfying numeric and categorical predicates."""

    module_id = str(target_module.module_id)
    item_id = str(target["item_id"])
    numeric_field_id = str(numeric_field.field_id)
    category_field_id = str(category_field.field_id)
    annotation_value = {
        "module_title": [float(value) for value in ctx.rendered.module_title_bboxes_px[module_id]],
        "numeric_field_label": [
            float(value)
            for value in ctx.rendered.field_label_bboxes_px[module_id][numeric_field_id]
        ],
        "category_field_label": [
            float(value)
            for value in ctx.rendered.field_label_bboxes_px[module_id][category_field_id]
        ],
        "matching_item": [
            float(value)
            for value in ctx.rendered.item_container_bboxes_px[module_id][item_id]
        ],
        "matching_item_label": [
            float(value) for value in ctx.rendered.item_label_bboxes_px[module_id][item_id]
        ],
        "numeric_value_cell": [
            float(value)
            for value in ctx.rendered.value_cell_bboxes_px[module_id][item_id][numeric_field_id]
        ],
        "category_value_cell": [
            float(value)
            for value in ctx.rendered.value_cell_bboxes_px[module_id][item_id][category_field_id]
        ],
    }
    return BoundTaskPayload(
        answer_type="string",
        answer_value=str(target["answer_value"]),
        annotation_type="bbox",
        annotation_value=list(annotation_value["matching_item"]),
        prompt_slots={
            "module_title": str(target_module.title),
            "numeric_field_label": str(numeric_field.label),
            "condition_phrase": str(target["condition_phrase"]),
            "threshold_value": str(target["threshold_visible"]),
            "category_field_label": str(category_field.label),
            "category_value": str(target["category_value"]),
        },
        target_payload=dict(target),
        annotation_keys=("matching_item",),
        task_params_extra={str(key): dict(value) for key, value in probability_payload.items()},
        execution_extra={
            "condition_operator": str(target["condition_operator"]),
            "threshold_value": int(target["threshold_value"]),
            "threshold_visible": str(target["threshold_visible"]),
            "category_value": str(target["category_value"]),
        },
        witness_type=str(prompt_key),
        diagnostic_bbox_map=dict(annotation_value),
    )


def condition_count_payload(
    *,
    ctx: MixedSceneContext,
    target_module: Any,
    target_field: Any,
    target: Mapping[str, Any],
    probability_payload: Mapping[str, Mapping[str, float]],
    prompt_key: str,
) -> BoundTaskPayload:
    """Bind all value cells matching a module-local numeric condition."""

    module_id = str(target_module.module_id)
    field_id = str(target_field.field_id)
    annotation_value = [
        [
            float(value)
            for value in ctx.rendered.value_cell_bboxes_px[module_id][str(match["item_id"])][
                field_id
            ]
        ]
        for match in target["matching_values"]
    ]
    return BoundTaskPayload(
        answer_type="integer",
        answer_value=int(target["answer_value"]),
        annotation_type="bbox_set",
        annotation_value=list(annotation_value),
        prompt_slots={
            "module_title": str(target_module.title),
            "field_label": str(target_field.label),
            "condition_phrase": str(target["condition_phrase"]),
            "threshold_value": str(target["threshold_visible"]),
        },
        target_payload=dict(target),
        annotation_keys=tuple(str(match["item_id"]) for match in target["matching_values"]),
        task_params_extra={str(key): dict(value) for key, value in probability_payload.items()},
        execution_extra={
            "condition_operator": str(target["condition_operator"]),
            "condition_answer_count": int(target["answer_value"]),
            "threshold_value": int(target["threshold_value"]),
            "threshold_visible": str(target["threshold_visible"]),
        },
        witness_type=str(prompt_key),
    )


def module_total_payload(
    *,
    ctx: MixedSceneContext,
    target_module: Any,
    target_field: Any,
    target: Mapping[str, Any],
    module_probs: Mapping[str, float],
    field_probs: Mapping[str, float],
    prompt_key: str,
) -> BoundTaskPayload:
    """Bind all value cells summed for one module additive field."""

    module_id = str(target_module.module_id)
    field_id = str(target_field.field_id)
    annotation_value = [
        [
            float(value)
            for value in ctx.rendered.value_cell_bboxes_px[module_id][
                str(value_payload["item_id"])
            ][field_id]
        ]
        for value_payload in target["summed_values"]
    ]
    return BoundTaskPayload(
        answer_type="integer",
        answer_value=int(target["answer_value"]),
        annotation_type="bbox_set",
        annotation_value=list(annotation_value),
        prompt_slots={
            "module_title": str(target_module.title),
            "field_label": str(target_field.label),
        },
        target_payload=dict(target),
        annotation_keys=tuple(
            str(value_payload["item_id"]) for value_payload in target["summed_values"]
        ),
        task_params_extra={
            "target_module_index_probabilities": dict(module_probs),
            "target_field_index_probabilities": dict(field_probs),
        },
        execution_extra={},
        witness_type=str(prompt_key),
    )


def two_module_total_comparison_payload(
    *,
    ctx: MixedSceneContext,
    module_a: Any,
    field_a: Any,
    module_b: Any,
    field_b: Any,
    target: Mapping[str, Any],
    field_probs: Mapping[str, float],
    pair_probs: Mapping[str, float],
    prompt_key: str,
) -> BoundTaskPayload:
    """Bind two module totals; public annotation marks the winning module."""

    module_a_id = str(module_a.module_id)
    module_b_id = str(module_b.module_id)
    field_a_id = str(field_a.field_id)
    field_b_id = str(field_b.field_id)
    annotation_value = {
        "module_a_title": [
            float(value) for value in ctx.rendered.module_title_bboxes_px[module_a_id]
        ],
        "module_b_title": [
            float(value) for value in ctx.rendered.module_title_bboxes_px[module_b_id]
        ],
        "field_label_a": [
            float(value) for value in ctx.rendered.field_label_bboxes_px[module_a_id][field_a_id]
        ],
        "field_label_b": [
            float(value) for value in ctx.rendered.field_label_bboxes_px[module_b_id][field_b_id]
        ],
    }
    for index, value_payload in enumerate(target["module_a"]["summed_values"], start=1):
        annotation_value[f"module_a_value_{index}"] = [
            float(value)
            for value in ctx.rendered.value_cell_bboxes_px[module_a_id][
                str(value_payload["item_id"])
            ][field_a_id]
        ]
    for index, value_payload in enumerate(target["module_b"]["summed_values"], start=1):
        annotation_value[f"module_b_value_{index}"] = [
            float(value)
            for value in ctx.rendered.value_cell_bboxes_px[module_b_id][
                str(value_payload["item_id"])
            ][field_b_id]
        ]
    winning_module_id = (
        module_a_id if str(target["winning_side"]) == "module_a" else module_b_id
    )
    return BoundTaskPayload(
        answer_type="string",
        answer_value=str(target["answer_value"]),
        annotation_type="bbox",
        annotation_value=[
            float(value) for value in ctx.rendered.module_bboxes_px[winning_module_id]
        ],
        prompt_slots={
            "module_a_title": str(module_a.title),
            "module_b_title": str(module_b.title),
            "field_label": str(target["field_label"]),
        },
        target_payload=dict(target),
        annotation_keys=("winning_module",),
        task_params_extra={
            "target_field_label_probabilities": dict(field_probs),
            "target_module_pair_probabilities": dict(pair_probs),
        },
        execution_extra={
            "target_field_label": str(target["field_label"]),
            "module_a_total": int(target["module_a_total"]),
            "module_b_total": int(target["module_b_total"]),
            "winning_side": str(target["winning_side"]),
        },
        witness_type=str(prompt_key),
        diagnostic_bbox_map=dict(annotation_value),
    )


__all__ = [
    "ADDITIVE_FIELD_LABELS",
    "BoundTaskPayload",
    "DOMAIN",
    "MixedSceneContext",
    "NAMESPACE_ROOT",
    "build_scene_context",
    "build_task_output",
    "build_trace_payload",
    "condition_count_payload",
    "module_field_value_payload",
    "module_rank_payload",
    "module_total_payload",
    "module_value_rank_annotation",
    "page_field_extremum_payload",
    "render_prompt",
    "run_bound_task",
    "select_public_branch",
    "task_generation_defaults",
    "task_render_defaults",
    "two_field_condition_payload",
    "two_module_total_comparison_payload",
]
