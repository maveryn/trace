"""Scene-private lifecycle helpers for symbolic chemical-equation tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from PIL import Image

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ....core.types import TypedValue
from ....core.visual.noise import apply_post_image_noise
from ...base import TaskOutput
from ...shared.annotation_artifacts import AnnotationArtifacts
from ...shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import build_prompt_query_spec
from ..shared.scene_style import (
    make_symbolic_scene_background,
    resolve_symbolic_scene_style,
)
from .shared.defaults import POST_IMAGE_NOISE_DEFAULTS
from .shared.prompts import render_chemical_prompt
from .shared.rendering import render_chemical_equation_scene
from .shared.sampling import resolve_chemical_scene_variant
from .shared.state import (
    SCENE_ID,
    SCENE_KIND,
    ChemicalEquationDataset,
    RenderedChemicalEquationScene,
)
from .shared.styles import resolve_render_params

DOMAIN = "symbolic"


@dataclass(frozen=True)
class ChemicalEquationTaskBinding:
    """Task-owned prompt and output binding for one chemical-equation objective."""

    public_task_id: str
    internal_query_id: str
    task_prompt_key: str
    object_description_prefix: str
    annotation_hint_key: str
    answer_hint_key: str
    json_example_key: str
    json_example_answer_only_key: str
    answer_type: str
    annotation_source: str
    failure_message: str


@dataclass(frozen=True)
class ChemicalRenderArtifacts:
    image: Image.Image
    rendered_scene: RenderedChemicalEquationScene
    render_spec: dict[str, Any]
    render_map: dict[str, Any]
    task_versions: dict[str, str]


def load_chemical_equation_defaults(
    public_task_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load scene config defaults for one chemical-equation objective."""

    return load_scene_generation_rendering_prompt_defaults(
        DOMAIN,
        SCENE_ID,
        task_id=str(public_task_id),
    )


def _make_canvas(
    *,
    instance_seed: int,
    namespace: str,
    canvas_width: int,
    canvas_height: int,
) -> tuple[Image.Image, Any, dict[str, Any], dict[str, Any]]:
    scene_style, scene_style_meta = resolve_symbolic_scene_style(
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    image, background_meta = make_symbolic_scene_background(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        style=scene_style,
    )
    return image, scene_style, dict(scene_style_meta), dict(background_meta)


def _draw_chemical_artifacts(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    namespace: str,
    dataset: ChemicalEquationDataset,
    annotation_source: str,
) -> ChemicalRenderArtifacts:
    """Render one finalized chemical-equation scene with projection metadata."""

    render_params = resolve_render_params(params, render_defaults)
    canvas, style, scene_style_meta, background_meta = _make_canvas(
        instance_seed=int(instance_seed),
        namespace=str(namespace),
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
    )
    rendered_scene = render_chemical_equation_scene(
        canvas,
        dataset=dataset,
        params=render_params,
        style=style,
        scene_variant=str(dataset.scene_variant),
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    render_spec = {
        "scene_id": SCENE_ID,
        "canvas_width": int(render_params.canvas_width),
        "canvas_height": int(render_params.canvas_height),
        "coord_space": "pixel",
        "scene_variant": str(dataset.scene_variant),
        "scene_style": dict(scene_style_meta),
        "background_style": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
        "scene_bbox_px": list(rendered_scene.scene_bbox_px),
        "chemical_equation_style": dict(rendered_scene.style_metadata),
    }
    def _bbox_map(mapping: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
        return {
            str(key): [round(float(value), 3) for value in bbox[:4]]
            for key, bbox in dict(mapping).items()
        }

    render_map = {
        "image_id": "img0",
        "scene_bbox_px": list(rendered_scene.scene_bbox_px),
        "item_bboxes_px": _bbox_map(rendered_scene.item_bboxes),
        "coefficient_slot_bboxes_px": _bbox_map(rendered_scene.coefficient_slot_bboxes),
        "molecule_card_bboxes_px": _bbox_map(rendered_scene.molecule_card_bboxes),
        "option_bboxes_px": _bbox_map(rendered_scene.option_bboxes),
        "atom_chip_bboxes_px": _bbox_map(rendered_scene.atom_chip_bboxes),
        "annotation_source": str(annotation_source),
    }
    return ChemicalRenderArtifacts(
        image=image,
        rendered_scene=rendered_scene,
        render_spec=render_spec,
        render_map=render_map,
        task_versions=default_task_versions(),
    )


def _entity_trace_records(
    entities: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for entity in entities:
        record = {str(key): value for key, value in dict(entity).items()}
        records.append(record)
    return records


def _term_records(dataset: ChemicalEquationDataset) -> list[dict[str, Any]]:
    return [
        {
            "item_id": str(term.item_id),
            "coefficient_slot_id": str(term.coefficient_slot_id),
            "molecule_card_id": str(term.molecule_card_id),
            "term_index": int(term.term_index),
            "side": str(term.side),
            "side_index": int(term.side_index),
            "formula": str(term.formula),
            "coefficient": int(term.coefficient),
            "hidden_coefficient": bool(term.hidden_coefficient),
            "atom_counts": {
                str(key): int(value) for key, value in term.atom_counts.items()
            },
        }
        for term in dataset.terms
    ]


def run_chemical_equation_instance(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    binding: ChemicalEquationTaskBinding,
    dataset_factory: Callable[[Any, str, Mapping[str, float]], ChemicalEquationDataset],
    annotation_factory: Callable[
        [RenderedChemicalEquationScene, ChemicalEquationDataset], AnnotationArtifacts
    ],
    annotation_item_ids_factory: Callable[[ChemicalEquationDataset], Sequence[str]],
) -> TaskOutput:
    """Run the common chemical-equation render/prompt/output lifecycle."""

    scene_variant, scene_variant_probabilities = resolve_chemical_scene_variant(
        params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        sampling_scope=str(binding.public_task_id),
    )
    last_error: Exception | None = None
    dataset: ChemicalEquationDataset | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        retry_seed = int(instance_seed) + int(attempt_index)
        try:
            dataset_rng = spawn_rng(
                int(retry_seed),
                f"{binding.public_task_id}.dataset",
            )
            dataset = dataset_factory(
                dataset_rng,
                str(scene_variant),
                scene_variant_probabilities,
            )
            break
        except Exception as exc:
            last_error = exc
    if dataset is None:
        raise RuntimeError(str(binding.failure_message)) from last_error

    render_artifacts = _draw_chemical_artifacts(
        instance_seed=int(instance_seed),
        params=params,
        render_defaults=render_defaults,
        namespace=f"{binding.public_task_id}.background",
        dataset=dataset,
        annotation_source=str(binding.annotation_source),
    )
    prompt_runtime = render_chemical_prompt(
        prompt_defaults,
        domain=DOMAIN,
        scene_id=SCENE_ID,
        scene_variant=str(dataset.scene_variant),
        task_key=str(binding.task_prompt_key),
        object_description_key=f"{binding.object_description_prefix}_{dataset.scene_variant}",
        annotation_hint_key=str(binding.annotation_hint_key),
        answer_hint_key=str(binding.answer_hint_key),
        json_example_key=str(binding.json_example_key),
        json_example_answer_only_key=str(binding.json_example_answer_only_key),
        instance_seed=int(instance_seed),
        context=f"prompt defaults for {binding.public_task_id}",
    )
    annotation_artifacts = annotation_factory(render_artifacts.rendered_scene, dataset)
    answer_gt = TypedValue(
        type=str(binding.answer_type),
        value=dataset.answer_value,
    )
    annotation_item_ids = [
        str(item_id) for item_id in annotation_item_ids_factory(dataset)
    ]
    query_params = {
        "query_id": SINGLE_QUERY_ID,
        "query_id_probabilities": {SINGLE_QUERY_ID: 1.0},
        "internal_query_id": str(binding.internal_query_id),
        "scene_id": SCENE_ID,
        "scene_variant": str(dataset.scene_variant),
        "scene_variant_probabilities": {
            str(key): float(value)
            for key, value in dataset.scene_variant_probabilities.items()
        },
        "question_format": str(binding.internal_query_id),
        "target_answer_support": [
            value for value in tuple(dataset.target_answer_support)
        ],
        "target_answer_probabilities": {
            str(key): float(value)
            for key, value in dataset.target_answer_probabilities.items()
        },
        "reaction_id": str(dataset.reaction.reaction_id),
    }
    prompt_query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_runtime.artifacts,
        query_id=SINGLE_QUERY_ID,
        params=query_params,
    )
    trace_payload = {
        "scene_ir": {
            "scene_kind": SCENE_KIND,
            "entities": _entity_trace_records(render_artifacts.rendered_scene.entities),
            "relations": {
                "query_id": SINGLE_QUERY_ID,
                "internal_query_id": str(binding.internal_query_id),
                "scene_id": SCENE_ID,
                "scene_variant": str(dataset.scene_variant),
                "reaction_id": str(dataset.reaction.reaction_id),
                "answer_value": answer_gt.value,
            },
        },
        "query_spec": {
            **dict(prompt_query_spec),
            "internal_query_id": str(binding.internal_query_id),
            "template_id": str(prompt_runtime.metadata["bundle_id"]),
        },
        "render_spec": dict(render_artifacts.render_spec),
        "render_map": dict(render_artifacts.render_map),
        "execution_trace": {
            **dict(query_params),
            "answer_value": answer_gt.value,
            "answer_type": str(answer_gt.type),
            "annotation_item_ids": list(annotation_item_ids),
            "terms": _term_records(dataset),
            "options": [
                {
                    "item_id": str(option.item_id),
                    "label": str(option.label),
                    "coefficients": [int(value) for value in option.coefficients],
                    "balances_equation": bool(option.balances_equation),
                }
                for option in dataset.options
            ],
            "chemical_equation_metadata": dict(dataset.metadata or {}),
        },
        "witness_symbolic": {
            "type": str(annotation_artifacts.annotation_gt.type),
            "ids": list(annotation_item_ids),
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "answer_gt": answer_gt.to_dict(),
        "annotation_gt": annotation_artifacts.annotation_gt.to_dict(),
    }
    return TaskOutput(
        prompt=str(prompt_runtime.prompt),
        answer_gt=answer_gt,
        annotation_gt=annotation_artifacts.annotation_gt,
        image=render_artifacts.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=dict(render_artifacts.task_versions),
        scene_id=SCENE_ID,
        query_id=SINGLE_QUERY_ID,
        prompt_variants=dict(prompt_runtime.prompt_variants),
    )


__all__ = [
    "ChemicalEquationTaskBinding",
    "load_chemical_equation_defaults",
    "run_chemical_equation_instance",
]
