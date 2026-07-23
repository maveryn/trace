"""Select the endpoint reached by following bearing-route instructions."""

from __future__ import annotations
from typing import Any, Dict, Tuple
from trace_tasks.core import scene_config as scene_config_core
from trace_tasks.core import types as core_types
from trace_tasks.tasks import base as task_base
from trace_tasks.tasks import registry as task_registry
from trace_tasks.tasks.geometry.shared import option_count as geometry_options
from trace_tasks.tasks.shared import config_defaults as shared_config_defaults
from trace_tasks.tasks.shared import fixed_query as shared_fixed_query
from trace_tasks.tasks.shared import output_metadata as shared_output_metadata
from . import _lifecycle as bearing_lifecycle
from .shared import construction as bearing_construction
from .shared import rendering as bearing_rendering
from .shared import state as bearing_state

TASK_ID = "task_geometry__bearing_route__endpoint_position_label"
QUERY_ID = "endpoint_position_label"
SCENE_VARIANT = "instruction_endpoint_candidates"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (QUERY_ID,)
_SCENE_DEFAULTS = scene_config_core.get_scene_defaults(
    "geometry", bearing_state.SCENE_ID
)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = (
    shared_config_defaults.split_scene_generation_rendering_prompt_defaults(
        _SCENE_DEFAULTS, task_id=TASK_ID
    )
)


@task_registry.register_task
class GeometryBearingRouteEndpointPositionLabelTask:
    """Select the candidate endpoint reached from the start point."""

    task_id = TASK_ID
    reasoning_operations = ('spatial_relations', 'state_update')
    domain = "geometry"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def _resolve_endpoint_case(
        self, *, instance_seed: int, params: Dict[str, Any]
    ) -> tuple[
        str,
        dict[str, float],
        dict[str, Any],
        bearing_state.RouteCase,
        dict[str, float],
        dict[str, float],
    ]:
        """Select the endpoint query, option count, and unique target route."""
        selected_query, query_probabilities, task_params = (
            shared_fixed_query.select_task_query_id(
                instance_seed=int(instance_seed),
                params=params,
                supported_query_ids=SUPPORTED_QUERY_IDS,
                default_query_id=QUERY_ID,
                task_id=TASK_ID,
            )
        )
        option_count, option_count_probabilities = (
            geometry_options.resolve_geometry_option_count(
                params=task_params,
                gen_defaults=_GEN_DEFAULTS,
                field_name="option_count",
                supported_counts=(4, 6),
                task_id=TASK_ID,
                instance_seed=int(instance_seed),
            )
        )
        route_case, target_index_probabilities = (
            bearing_construction.resolve_endpoint_route_case(
                params=task_params,
                instance_seed=int(instance_seed),
                option_count=option_count,
                route_case_namespace=f"{TASK_ID}.{selected_query}.route_case",
                orientation_namespace=f"{TASK_ID}.{selected_query}.orientation",
                target_index_namespace=f"{TASK_ID}.{selected_query}.target_index",
                labels_namespace=f"{TASK_ID}.{selected_query}.labels",
            )
        )
        if route_case.target_index is None:
            raise ValueError("endpoint-position task requires a selected endpoint")
        return (
            str(selected_query),
            dict(query_probabilities),
            dict(task_params),
            route_case,
            dict(target_index_probabilities),
            dict(option_count_probabilities),
        )

    def _build_endpoint_output(
        self,
        *,
        prepared: bearing_lifecycle.PreparedBearingRoute,
        route_case: bearing_state.RouteCase,
        selected_query: str,
        query_probabilities: dict[str, float],
        target_index_probabilities: dict[str, float],
        option_count_probabilities: dict[str, float],
    ) -> task_base.TaskOutput:
        """Bind endpoint answer/annotation and construct the final TaskOutput."""
        rendered = prepared.runtime.rendered
        answer_value = str(route_case.option_labels[int(route_case.target_index)])
        answer_gt = core_types.TypedValue(type="option_letter", value=answer_value)
        annotation_gt = core_types.TypedValue(
            type="point_map", value=dict(prepared.annotation_keyed_points)
        )
        query_params = {
            "scene_id": bearing_state.SCENE_ID,
            "scene_variant": SCENE_VARIANT,
            "query_id": str(selected_query),
            "query_id_probabilities": dict(query_probabilities),
            "target_index_probabilities": dict(target_index_probabilities),
            "option_count_probabilities": dict(option_count_probabilities),
            **dict(rendered.witness),
        }
        trace_payload = bearing_lifecycle.build_bearing_route_trace_payload(
            runtime=prepared.runtime,
            prompt_artifacts=prepared.prompt_artifacts,
            noise_meta=prepared.noise_meta,
            branch_name=str(selected_query),
            branch_params=query_params,
            scene_variant=SCENE_VARIANT,
            answer_type="option_letter",
            answer_value=answer_value,
            witness_kind="bearing_route_endpoint_position",
            annotation_bboxes=prepared.annotation_bboxes,
            annotation_points=prepared.annotation_points,
            annotation_keyed_bboxes=prepared.annotation_keyed_bboxes,
            annotation_keyed_points=prepared.annotation_keyed_points,
        )
        return task_base.TaskOutput(
            prompt=str(prepared.prompt_artifacts.prompt),
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=prepared.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=shared_output_metadata.default_task_versions(),
            scene_id=bearing_state.SCENE_ID,
            query_id=str(selected_query),
            prompt_variants=dict(prepared.prompt_artifacts.prompt_variants),
        )

    def generate(
        self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int
    ) -> task_base.TaskOutput:
        """Own endpoint selection and annotation after final rendered layout."""
        (
            selected_query,
            query_probabilities,
            task_params,
            route_case,
            target_index_probabilities,
            option_count_probabilities,
        ) = self._resolve_endpoint_case(instance_seed=int(instance_seed), params=params)
        prepared = bearing_lifecycle.prepare_bearing_route_rendering(
            route_case=route_case,
            scene_renderer=bearing_rendering.render_endpoint_label_scene,
            domain=self.domain,
            prompt_defaults=_PROMPT_DEFAULTS,
            prompt_key=str(selected_query),
            instance_seed=int(instance_seed),
            params=task_params,
            render_defaults=_RENDER_DEFAULTS,
            max_attempts=int(max_attempts),
            style_namespace=f"{TASK_ID}.style",
        )
        task_output = self._build_endpoint_output(
            prepared=prepared,
            route_case=route_case,
            selected_query=str(selected_query),
            query_probabilities=dict(query_probabilities),
            target_index_probabilities=dict(target_index_probabilities),
            option_count_probabilities=dict(option_count_probabilities),
        )
        return task_output


__all__ = ["GeometryBearingRouteEndpointPositionLabelTask"]
