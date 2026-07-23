"""Shared helpers for public tasks backed by internal query branches."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, Dict, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.query_ids import LEGACY_DEFAULT_QUERY_ID, SINGLE_QUERY_ID

from ..base import TaskOutput
from .deterministic_sampling import resolve_selection_index


_UNSET = object()
DEFAULT_QUERY_ID = SINGLE_QUERY_ID
QUERY_ID_PARAM_KEYS: tuple[str, str] = ("query_id", "query_variant")
QUERY_ID_WEIGHT_KEYS: tuple[str, str] = ("query_id_weights", "query_variant_weights")


def explicit_query_id_param(params: Mapping[str, Any], *, allow_default: bool = False) -> str | None:
    """Return the requested query id from canonical or legacy param names.

    `query_variant` is kept as an input-only legacy alias. Generated metadata
    should continue to use only `query_id`.
    """

    selected: str | None = None
    selected_key: str | None = None
    for key in QUERY_ID_PARAM_KEYS:
        value = params.get(str(key))
        if value is None:
            continue
        text = str(value)
        if text == LEGACY_DEFAULT_QUERY_ID and not bool(allow_default):
            continue
        if selected is not None and text != selected:
            raise ValueError(f"{selected_key} conflicts with {key}")
        selected = text
        selected_key = str(key)
    return selected


def has_explicit_query_id_param(params: Mapping[str, Any], *, allow_default: bool = False) -> bool:
    """Return whether params explicitly pin a query id or legacy variant."""

    return explicit_query_id_param(params, allow_default=bool(allow_default)) is not None


def strip_query_id_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Return params without public query-id selector aliases."""

    stripped = dict(params)
    for key in QUERY_ID_PARAM_KEYS:
        stripped.pop(str(key), None)
    return stripped


def resolve_task_query_id_param(
    params: Mapping[str, Any],
    *,
    supported_query_ids: Sequence[str],
    default_query_id: str = DEFAULT_QUERY_ID,
    task_id: str = "",
) -> str:
    """Resolve and validate a caller-selected internal query id.

    Public task files own this validation. Scene shared helpers should receive
    semantic arguments, not the returned query-id string.
    """

    supported = tuple(str(value) for value in supported_query_ids if str(value))
    if not supported:
        context = f" for {task_id}" if task_id else ""
        raise ValueError(f"supported_query_ids must be non-empty{context}")

    default_text = str(default_query_id)
    if default_text not in supported:
        context = f" for {task_id}" if task_id else ""
        raise ValueError(
            f"default query_id '{default_text}' is not in supported_query_ids{context}: {supported}"
        )

    selected = explicit_query_id_param(params, allow_default=True)
    if selected is None:
        return default_text
    selected_text = str(selected)
    if selected_text == LEGACY_DEFAULT_QUERY_ID:
        return default_text
    if selected_text not in supported:
        context = f" for {task_id}" if task_id else ""
        raise ValueError(f"unsupported query_id{context}: {selected_text}; supported: {supported}")
    return selected_text


def normalize_query_id_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize legacy query-id aliases onto canonical `query_id` params."""

    normalized = dict(params)
    selected = explicit_query_id_param(normalized)
    if selected is not None:
        normalized["query_id"] = str(selected)
    normalized.pop("query_variant", None)

    canonical_weights = normalized.get("query_id_weights")
    legacy_weights = normalized.get("query_variant_weights")
    if canonical_weights is not None and legacy_weights is not None and canonical_weights != legacy_weights:
        raise ValueError("query_id_weights conflicts with query_variant_weights")
    if canonical_weights is None and legacy_weights is not None:
        normalized["query_id_weights"] = legacy_weights
    normalized.pop("query_variant_weights", None)
    return normalized


def force_query_id_params(params: Mapping[str, Any], *, query_id: str) -> Dict[str, Any]:
    """Return params that force one internal query branch."""

    forced = normalize_query_id_params(params)
    requested_query_id = forced.get("query_id")
    if requested_query_id is not None and str(requested_query_id) != str(query_id):
        raise ValueError(
            "public task query_id must match query_id "
            f"'{query_id}' (got: {requested_query_id})"
        )
    forced["query_id"] = str(query_id)
    return forced


def probability_map(values: Sequence[str]) -> Dict[str, float]:
    """Return a uniform probability map over string values."""

    resolved = tuple(str(value) for value in values)
    if not resolved:
        return {}
    weight = 1.0 / float(len(resolved))
    return {str(value): float(weight) for value in resolved}


def normalize_probability_map(values: Mapping[str, float]) -> Dict[str, float]:
    """Return a JSON-stable string-keyed probability map."""

    return {str(key): float(value) for key, value in values.items()}


def select_task_query_id(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    supported_query_ids: Sequence[str],
    default_query_id: str = DEFAULT_QUERY_ID,
    task_id: str = "",
    namespace: str | None = None,
) -> tuple[str, Dict[str, float], Dict[str, Any]]:
    """Select one internal query branch for a public task.

    Public task files own the supported query ids and translate the selected id
    into semantic arguments. This helper owns only the shared selection policy:
    explicit query pinning, finite-support cursor cycling, and uniform random
    fallback.
    """

    supported = tuple(str(value) for value in supported_query_ids if str(value))
    if not supported:
        context = f" for {task_id}" if task_id else ""
        raise ValueError(f"supported_query_ids must be non-empty{context}")

    explicit = explicit_query_id_param(params, allow_default=True)
    if explicit is not None:
        selected = resolve_task_query_id_param(
            params,
            supported_query_ids=supported,
            default_query_id=str(default_query_id),
            task_id=str(task_id),
        )
        return (
            str(selected),
            {query: (1.0 if query == str(selected) else 0.0) for query in supported},
            strip_query_id_params(params),
        )

    sample_cursor = params.get("_sample_cursor")
    if sample_cursor is not None:
        cursor = abs(int(sample_cursor))
        selected = supported[cursor % len(supported)]
        task_params = strip_query_id_params(params)
        task_params["_sample_cursor"] = cursor // len(supported)
        return str(selected), probability_map(supported), task_params

    resolved_namespace = str(namespace) if namespace else f"{task_id}.query"
    rng = spawn_rng(int(instance_seed), resolved_namespace)
    return str(rng.choice(supported)), probability_map(supported), strip_query_id_params(params)


def rewrite_public_query_output(
    output: TaskOutput,
    *,
    query_id: str,
    scene_id: str | None = None,
    task_id: str | None = None,
    include_render_spec: bool = False,
    include_scene_ir_root: bool = False,
    query_id_probabilities: Mapping[str, float] | object = _UNSET,
    params_query_id_probabilities: Mapping[str, float] | object = _UNSET,
    scene_variant_probabilities: Mapping[str, float] | object = _UNSET,
    preserve_internal_query_id_as: str | Sequence[str] | None = None,
    preserve_prior_task_id_as: str | None = None,
    clear_keys: Sequence[str] = (),
    extra_fields: Mapping[str, Any] | None = None,
    extra_param_fields: Mapping[str, Any] | None = None,
    prompt_metadata: Mapping[str, Any] | None = None,
    update_existing_taxonomy: bool = False,
) -> TaskOutput:
    """Rewrite trace/query metadata for a narrowed public task.

    Domain wrappers choose which optional probability fields to emit so this
    helper can preserve existing generation and replay contracts.
    """

    payload = deepcopy(output.trace_payload if isinstance(output.trace_payload, Mapping) else {})
    query_id_text = str(query_id)
    scene_id_text = None if scene_id is None else str(scene_id)
    task_id_text = None if task_id is None else str(task_id)
    clear_key_values = tuple(str(key) for key in clear_keys)
    extra_field_map = {str(key): value for key, value in dict(extra_fields or {}).items()}
    extra_param_field_map = {str(key): value for key, value in dict(extra_param_fields or extra_field_map).items()}
    prompt_metadata_map = {str(key): value for key, value in dict(prompt_metadata or {}).items()}
    if preserve_internal_query_id_as is None:
        preserve_internal_keys: tuple[str, ...] = ()
    elif isinstance(preserve_internal_query_id_as, str):
        preserve_internal_keys = (str(preserve_internal_query_id_as),)
    else:
        preserve_internal_keys = tuple(str(key) for key in preserve_internal_query_id_as)
    top_query_probabilities = (
        _UNSET
        if query_id_probabilities is _UNSET
        else normalize_probability_map(query_id_probabilities)  # type: ignore[arg-type]
    )
    if params_query_id_probabilities is _UNSET:
        params_probabilities = top_query_probabilities
    else:
        params_probabilities = normalize_probability_map(params_query_id_probabilities)  # type: ignore[arg-type]
    scene_variant_probability_map = (
        _UNSET
        if scene_variant_probabilities is _UNSET
        else normalize_probability_map(scene_variant_probabilities)  # type: ignore[arg-type]
    )

    def _preserve_internal(value: Dict[str, Any]) -> None:
        if not preserve_internal_keys:
            return
        prior_query_id = value.get("query_id")
        if prior_query_id is not None and str(prior_query_id) != LEGACY_DEFAULT_QUERY_ID:
            for key in preserve_internal_keys:
                value.setdefault(str(key), str(prior_query_id))

    def _rewrite_task_id(value: Dict[str, Any]) -> None:
        if task_id_text is None:
            return
        prior_task_id = value.get("task_id")
        if (
            preserve_prior_task_id_as
            and prior_task_id is not None
            and str(prior_task_id) != task_id_text
        ):
            value.setdefault(str(preserve_prior_task_id_as), str(prior_task_id))
        value["task_id"] = task_id_text

    def _rewrite_prompt_metadata(value: Dict[str, Any]) -> None:
        if not prompt_metadata_map:
            return
        prompt_variant = value.get("prompt_variant")
        if isinstance(prompt_variant, dict):
            prompt_variant.update(prompt_metadata_map)
        prompt_variants = value.get("prompt_variants")
        if not isinstance(prompt_variants, dict):
            return
        for prompt_variant_record in prompt_variants.values():
            if not isinstance(prompt_variant_record, dict):
                continue
            metadata = prompt_variant_record.get("metadata")
            if isinstance(metadata, dict):
                metadata.update(prompt_metadata_map)

    def _clear_mapping_keys(value: Dict[str, Any]) -> None:
        for key in clear_key_values:
            value.pop(str(key), None)

    def _rewrite_mapping(value: Any) -> None:
        if not isinstance(value, dict):
            return
        _clear_mapping_keys(value)
        _preserve_internal(value)
        _rewrite_task_id(value)
        if scene_id_text is not None:
            value["scene_id"] = scene_id_text
        value["query_id"] = query_id_text
        if top_query_probabilities is not _UNSET:
            value["query_id_probabilities"] = dict(top_query_probabilities)  # type: ignore[arg-type]
        if scene_variant_probability_map is not _UNSET and scene_variant_probability_map:
            value["scene_variant_probabilities"] = dict(scene_variant_probability_map)  # type: ignore[arg-type]
        value.update(extra_field_map)
        _rewrite_prompt_metadata(value)

        params = value.get("params")
        if not isinstance(params, dict):
            return
        _clear_mapping_keys(params)
        _preserve_internal(params)
        _rewrite_task_id(params)
        if scene_id_text is not None:
            params["scene_id"] = scene_id_text
        params["query_id"] = query_id_text
        if params_probabilities is not _UNSET:
            params["query_id_probabilities"] = dict(params_probabilities)  # type: ignore[arg-type]
        if scene_variant_probability_map is not _UNSET and scene_variant_probability_map:
            params["scene_variant_probabilities"] = dict(scene_variant_probability_map)  # type: ignore[arg-type]
        params.update(extra_param_field_map)

    def _rewrite_scene_ir_root(value: Any) -> None:
        if not isinstance(value, dict):
            return
        _clear_mapping_keys(value)
        _preserve_internal(value)
        _rewrite_task_id(value)
        if scene_id_text is not None:
            value["scene_id"] = scene_id_text
        value["query_id"] = query_id_text

    def _rewrite_existing_taxonomy() -> None:
        if not update_existing_taxonomy:
            return
        taxonomy = payload.get("taxonomy")
        if not isinstance(taxonomy, dict):
            return
        if scene_id_text is not None:
            taxonomy["scene_id"] = scene_id_text
        if task_id_text is not None:
            prior_task_id = taxonomy.get("task_id")
            if (
                preserve_prior_task_id_as
                and prior_task_id is not None
                and str(prior_task_id) != task_id_text
            ):
                taxonomy.setdefault(str(preserve_prior_task_id_as), str(prior_task_id))
            taxonomy["task_id"] = task_id_text
        taxonomy["query_id"] = query_id_text
        public = taxonomy.get("public")
        if isinstance(public, dict):
            if scene_id_text is not None:
                public["scene_id"] = scene_id_text
            if task_id_text is not None:
                public["task_id"] = task_id_text
            public["query_id"] = query_id_text

    keys = ["query_spec", "execution_trace"]
    if include_render_spec:
        keys.append("render_spec")
    for key in keys:
        _rewrite_mapping(payload.get(str(key)))

    scene_ir = payload.get("scene_ir")
    if isinstance(scene_ir, dict):
        if include_scene_ir_root:
            _rewrite_scene_ir_root(scene_ir)
        _rewrite_mapping(scene_ir.get("relations"))
    _rewrite_mapping(payload.get("witness_symbolic"))
    _rewrite_existing_taxonomy()

    return replace(
        output,
        trace_payload=payload,
        query_id=query_id_text,
        scene_id=scene_id_text if scene_id_text is not None else output.scene_id,
    )


def _validated_sequence(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    resolved = tuple(str(value).strip() for value in values if str(value).strip())
    if not resolved:
        raise ValueError(f"{field_name} must contain at least one non-empty value")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{field_name} must not contain duplicate values: {resolved!r}")
    return resolved


def _probability_keys(
    values: Sequence[Any],
    *,
    key_fn: Any | None = None,
    sort_unique: bool = False,
) -> tuple[str, ...]:
    formatter = key_fn or (lambda value: str(value))
    keys = tuple(str(formatter(value)) for value in values)
    if bool(sort_unique):
        return tuple(sorted(set(keys)))
    return keys


def geometry_probability_map(
    values: Sequence[Any],
    *,
    key_fn: Any | None = None,
    sort_unique: bool = False,
) -> Dict[str, float]:
    """Return a uniform probability map for geometry support values."""

    return probability_map(_probability_keys(values, key_fn=key_fn, sort_unique=sort_unique))


def geometry_selected_probability_map(
    values: Sequence[Any],
    selected: Any | None = None,
    *,
    key_fn: Any | None = None,
    is_selected: Any | None = None,
    sort_unique: bool = False,
) -> Dict[str, float]:
    """Return a uniform or one-hot probability map for support values."""

    raw_values = tuple(values)
    resolved = _probability_keys(raw_values, key_fn=key_fn, sort_unique=sort_unique)
    if selected is None:
        return probability_map(resolved)
    selected_key = _probability_keys((selected,), key_fn=key_fn)[0]
    if bool(sort_unique) or is_selected is None:
        return {value: (1.0 if value == selected_key else 0.0) for value in resolved}
    return {
        key: (1.0 if bool(is_selected(raw_value, selected)) else 0.0)
        for raw_value, key in zip(raw_values, resolved)
    }


def geometry_query_ids_for_task(
    task_id: str,
    query_ids_by_task_id: Mapping[str, Sequence[str]],
    *,
    context: str = "geometry task",
) -> tuple[str, ...]:
    """Return validated query ids for one public task id."""

    task_id_text = str(task_id)
    if task_id_text not in query_ids_by_task_id:
        raise ValueError(f"unsupported {context} task_id: {task_id_text}")
    return _validated_sequence(
        tuple(str(query_id) for query_id in query_ids_by_task_id[task_id_text]),
        field_name=f"query ids for {task_id_text}",
    )


def select_geometry_query_id(
    params: Mapping[str, Any],
    *,
    query_ids: Sequence[str],
    task_id: str = "",
    instance_seed: int = 0,
) -> tuple[str, Dict[str, float]]:
    """Select one internal query id for a public task."""

    supported = _validated_sequence(query_ids, field_name="query_ids")
    selected = explicit_query_id_param(params, allow_default=False)
    if selected is not None:
        selected_text = str(selected)
        if selected_text not in set(supported):
            raise ValueError(
                f"query_id={selected_text!r} is not valid for {task_id or 'task'}; expected {supported!r}"
            )
        return selected_text, {selected_text: 1.0}
    rng = spawn_rng(int(instance_seed), f"{str(task_id)}.fixed_query_id")
    sampled = str(supported[int(rng.randrange(len(supported)))])
    return sampled, probability_map(supported)


def select_indexed_geometry_query_id(
    params: Mapping[str, Any],
    *,
    query_ids: Sequence[str],
    task_id: str = "",
    instance_seed: int = 0,
    default_means_sample: bool = True,
) -> tuple[str, Dict[str, float]]:
    """Select a query id with the deterministic sample-index convention."""

    supported = _validated_sequence(query_ids, field_name="query_ids")
    explicit_value = params.get("query_id")
    if explicit_value is not None and str(explicit_value) == LEGACY_DEFAULT_QUERY_ID and bool(default_means_sample):
        explicit_value = None
    if explicit_value is not None:
        selected = str(explicit_value)
        if selected not in set(supported):
            raise ValueError(
                f"query_id={selected!r} is not valid for {task_id or 'task'}; expected {supported!r}"
            )
        return selected, geometry_selected_probability_map(supported, selected=selected)
    index = resolve_selection_index(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{str(task_id)}.query_id",
    )
    selected = str(supported[int(index) % len(supported)])
    return selected, probability_map(supported)


def forced_query_params(params: Mapping[str, Any], *, query_id: str) -> Dict[str, Any]:
    """Return params that force one internal query id."""

    return force_query_id_params(params, query_id=str(query_id))


def forced_query_id_params(params: Mapping[str, Any], *, query_id: str) -> Dict[str, Any]:
    """Return params that force one internal query id."""

    return force_query_id_params(params, query_id=str(query_id))


def forced_geometry_query_params(
    params: Mapping[str, Any],
    *,
    query_id: str,
    allowed_scene_variants: Sequence[str] = (),
    task_id: str = "",
    instance_seed: int = 0,
) -> Dict[str, Any]:
    """Return params that force one geometry query and optional scene variant."""

    forced = force_query_id_params(params, query_id=str(query_id))
    allowed_scenes = tuple(str(scene) for scene in allowed_scene_variants if str(scene).strip())
    if not allowed_scenes:
        return forced
    explicit_scene = forced.get("scene_variant")
    if explicit_scene is not None:
        if str(explicit_scene) not in set(allowed_scenes):
            raise ValueError(
                f"scene_variant={explicit_scene!r} is not valid for query_id={query_id!r}; "
                f"expected one of {allowed_scenes!r}"
            )
        return forced
    rng = spawn_rng(int(instance_seed), f"{str(task_id)}.fixed_scene_variant")
    forced["scene_variant"] = str(allowed_scenes[int(rng.randrange(len(allowed_scenes)))])
    return forced


def rewrite_fixed_query_output(
    output: TaskOutput,
    *,
    query_id: str,
    query_id_probabilities: Mapping[str, float] | None = None,
) -> TaskOutput:
    """Rewrite generated output for a task that pins one query id."""

    query_id_text = str(query_id)
    return rewrite_public_query_output(
        output,
        query_id=query_id_text,
        query_id_probabilities=dict(query_id_probabilities or {query_id_text: 1.0}),
    )


def rewrite_fixed_geometry_query_output(
    output: TaskOutput,
    *,
    query_id: str,
    scene_id: str,
    allowed_scene_variants: Sequence[str] = (),
    query_id_probabilities: Mapping[str, float] | None = None,
) -> TaskOutput:
    """Rewrite geometry output to the selected public query id."""

    scene_values = tuple(str(scene) for scene in allowed_scene_variants if str(scene).strip())
    return rewrite_public_query_output(
        output,
        scene_id=str(scene_id),
        query_id=str(query_id),
        include_render_spec=True,
        query_id_probabilities=dict(query_id_probabilities or {str(query_id): 1.0}),
        scene_variant_probabilities=geometry_probability_map(scene_values) if scene_values else {},
    )


def rewrite_fixed_puzzle_query_output(output: TaskOutput, *, query_id: str, scene_id: str = "") -> TaskOutput:
    """Rewrite generated puzzle output under one public query id."""

    return rewrite_public_query_output(
        output,
        query_id=str(query_id),
        scene_id=str(scene_id) if str(scene_id).strip() else None,
        include_render_spec=True,
        query_id_probabilities={str(query_id): 1.0},
        preserve_internal_query_id_as="internal_query_id",
    )


def forced_puzzle_query_params(params: Mapping[str, Any], *, query_id: str) -> Dict[str, Any]:
    """Return params that force one internal puzzle query id."""

    return force_query_id_params(params, query_id=str(query_id))


def rewrite_physics_query_output(output: TaskOutput, *, query_id: str) -> TaskOutput:
    """Rewrite generated physics output under one public query id."""

    return rewrite_public_query_output(
        output,
        query_id=str(query_id),
        params_query_id_probabilities={str(query_id): 1.0},
        preserve_internal_query_id_as="internal_query_id",
    )


def merged_query_params(
    params: Mapping[str, Any],
    *,
    allowed_query_ids: Sequence[str],
) -> Dict[str, Any]:
    """Return params restricted to a public query set."""

    merged = normalize_query_id_params(params)
    allowed = tuple(str(value) for value in allowed_query_ids if str(value).strip())
    allowed_set = set(allowed)
    explicit_query = explicit_query_id_param(merged)
    if allowed and explicit_query is not None:
        selected = str(explicit_query)
        if selected not in allowed_set:
            raise ValueError(f"unsupported query id for merged task: {selected}")
        return merged
    raw_weights = merged.get("query_id_weights")
    if allowed and raw_weights is None:
        merged["query_id_weights"] = {str(value): 1.0 for value in allowed}
    elif allowed and isinstance(raw_weights, Mapping):
        positive = {str(key) for key, value in raw_weights.items() if float(value) > 0.0}
        invalid = sorted(positive.difference(allowed_set))
        if invalid:
            raise ValueError(f"unsupported positive query weights for merged task: {invalid}")
    elif allowed and raw_weights is not None:
        raise ValueError("query_id_weights must be a mapping when provided")
    return merged


def infer_query_id_from_output(output: TaskOutput) -> str:
    """Infer the concrete generated query id from output metadata."""

    if str(output.query_id).strip() and str(output.query_id) != LEGACY_DEFAULT_QUERY_ID:
        return str(output.query_id)
    payload = output.trace_payload if isinstance(output.trace_payload, Mapping) else {}
    sources = [
        payload.get("query_spec") if isinstance(payload, Mapping) else None,
        payload.get("execution_trace") if isinstance(payload, Mapping) else None,
    ]
    scene_ir = payload.get("scene_ir") if isinstance(payload, Mapping) else None
    if isinstance(scene_ir, Mapping):
        sources.append(scene_ir.get("relations"))
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        params = source.get("params")
        for candidate in (source, params if isinstance(params, Mapping) else None):
            if not isinstance(candidate, Mapping):
                continue
            for key in ("query_id", "query_variant"):
                value = candidate.get(str(key))
                if value is not None and str(value).strip() and str(value) != LEGACY_DEFAULT_QUERY_ID:
                    return str(value)
    return ""


def forced_pages_query_params(params: Mapping[str, Any], *, query_id: str) -> Dict[str, Any]:
    """Return params that force one internal pages query id."""

    return force_query_id_params(params, query_id=str(query_id))


def merged_pages_query_params(
    params: Mapping[str, Any],
    *,
    allowed_query_ids: Sequence[str],
) -> Dict[str, Any]:
    """Return params restricted to one public pages query set."""

    return merged_query_params(params, allowed_query_ids=allowed_query_ids)


def infer_pages_query_id(output: TaskOutput) -> str:
    """Infer the concrete generated pages query id from output metadata."""

    return infer_query_id_from_output(output)


def query_probabilities_from_pages_output(
    output: TaskOutput,
    *,
    fallback_query_ids: Sequence[str],
) -> Dict[str, float]:
    """Return query probability metadata from a generated pages output."""

    payload = output.trace_payload if isinstance(output.trace_payload, Mapping) else {}
    for source in (
        payload.get("execution_trace") if isinstance(payload, Mapping) else None,
        payload.get("query_spec", {}).get("params") if isinstance(payload.get("query_spec"), Mapping) else None,
    ):
        if isinstance(source, Mapping) and isinstance(source.get("query_id_probabilities"), Mapping):
            return {str(key): float(value) for key, value in source["query_id_probabilities"].items()}
    return probability_map(tuple(str(query_id) for query_id in fallback_query_ids))


def rewrite_pages_public_task_output(
    output: TaskOutput,
    *,
    task_id: str,
    scene_id: str,
    query_id: str,
    query_probabilities: Mapping[str, float] | None = None,
) -> TaskOutput:
    """Rewrite generated pages output under the public task id."""

    rewritten = rewrite_public_query_output(
        output,
        task_id=str(task_id),
        scene_id=str(scene_id),
        query_id=str(query_id),
        include_render_spec=True,
        include_scene_ir_root=True,
        query_id_probabilities=dict(query_probabilities or {str(query_id): 1.0}),
        preserve_prior_task_id_as="source_task_id",
        preserve_internal_query_id_as=("source_query_id", "internal_query_id"),
        update_existing_taxonomy=True,
    )
    return replace(rewritten, image_id=f"{str(task_id)}_image")


class FixedQueryVariantTaskMixin:
    """Mixin for legacy tasks that force one internal query id."""

    default_dataset_enabled = True
    fixed_query_id: str

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        output = super().generate(  # type: ignore[misc]
            int(instance_seed),
            params=force_query_id_params(params, query_id=str(self.fixed_query_id)),
            max_attempts=int(max_attempts),
        )
        return rewrite_fixed_query_output(output, query_id=str(self.fixed_query_id))


class QuerySubsetTaskMixin:
    """Mixin for legacy tasks that sample a constrained query set."""

    default_dataset_enabled = True
    supported_query_ids: Sequence[str]

    def _select_query_id(self, instance_seed: int, params: Mapping[str, Any]) -> tuple[str, Dict[str, float]]:
        supported = tuple(str(value) for value in self.supported_query_ids)
        if not supported:
            raise ValueError("supported_query_ids must contain at least one query id")
        explicit = explicit_query_id_param(params, allow_default=True)
        if explicit == LEGACY_DEFAULT_QUERY_ID:
            explicit = None
        probabilities = probability_map(supported)
        if explicit is not None:
            query_id = str(explicit)
            if query_id not in supported:
                raise ValueError(f"unsupported query_id={query_id!r}; expected one of {supported}")
            return query_id, {str(item): (1.0 if str(item) == query_id else 0.0) for item in supported}
        sampling_index = params.get("_sample_cursor")
        if sampling_index is not None:
            return str(supported[abs(int(sampling_index)) % len(supported)]), probabilities
        rng = spawn_rng(int(instance_seed), f"{self.task_id}.query_id")
        return str(rng.choice(supported)), probabilities

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        query_id, probabilities = self._select_query_id(int(instance_seed), params)
        raw_cursor = params.get("_sample_cursor")
        explicit_query = explicit_query_id_param(params, allow_default=True)
        next_params = force_query_id_params(params, query_id=str(query_id))
        if raw_cursor is not None and explicit_query in {None, LEGACY_DEFAULT_QUERY_ID}:
            next_params["_sample_cursor"] = abs(int(raw_cursor)) // max(1, len(tuple(self.supported_query_ids)))
        output = super().generate(  # type: ignore[misc]
            int(instance_seed),
            params=next_params,
            max_attempts=int(max_attempts),
        )
        return rewrite_fixed_query_output(output, query_id=str(query_id), query_id_probabilities=probabilities)


class FixedChartQueryVariantTaskMixin(FixedQueryVariantTaskMixin):
    """Compatibility alias for chart tasks with one internal query id."""


class MergedChartQueryVariantTaskMixin(QuerySubsetTaskMixin):
    """Compatibility alias for chart tasks with a restricted query set."""

    allowed_query_ids: Sequence[str] = ()
    fixed_query_ids: Sequence[str] = ()

    @property
    def supported_query_ids(self) -> Sequence[str]:  # type: ignore[override]
        return self.allowed_query_ids or self.fixed_query_ids


class FixedPuzzleQueryVariantTaskMixin(FixedQueryVariantTaskMixin):
    """Compatibility alias for puzzle tasks with one internal query id."""

    public_scene_id: str = ""

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        output = super(FixedQueryVariantTaskMixin, self).generate(  # type: ignore[misc]
            int(instance_seed),
            params=force_query_id_params(params, query_id=str(self.fixed_query_id)),
            max_attempts=int(max_attempts),
        )
        return rewrite_fixed_puzzle_query_output(
            output,
            query_id=str(self.fixed_query_id),
            scene_id=str(self.public_scene_id),
        )


class FixedPhysicsQueryVariantTaskMixin(FixedQueryVariantTaskMixin):
    """Compatibility alias for physics tasks with one internal query id."""

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        output = super(FixedQueryVariantTaskMixin, self).generate(  # type: ignore[misc]
            int(instance_seed),
            params=force_query_id_params(params, query_id=str(self.fixed_query_id)),
            max_attempts=int(max_attempts),
        )
        return rewrite_physics_query_output(output, query_id=str(self.fixed_query_id))


class FixedPagesQueryTaskMixin:
    """Compatibility mixin for pages tasks backed by one source query branch."""

    default_dataset_enabled = True
    fixed_query_id: str
    public_scene_id: str
    source_task_cls: type

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        source_task = self.source_task_cls()
        output = source_task.generate(
            int(instance_seed),
            params=force_query_id_params(params, query_id=str(self.fixed_query_id)),
            max_attempts=int(max_attempts),
        )
        return rewrite_pages_public_task_output(
            output,
            task_id=str(self.task_id),
            scene_id=str(self.public_scene_id),
            query_id=str(self.fixed_query_id),
            query_probabilities={str(self.fixed_query_id): 1.0},
        )


class MergedPagesQueryTaskMixin:
    """Compatibility mixin for pages tasks that sample source query branches."""

    default_dataset_enabled = True
    allowed_query_ids: Sequence[str] = ()
    public_scene_id: str
    source_task_cls: type

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        allowed = tuple(str(value) for value in self.allowed_query_ids if str(value).strip())
        source_task = self.source_task_cls()
        output = source_task.generate(
            int(instance_seed),
            params=merged_query_params(params, allowed_query_ids=allowed),
            max_attempts=int(max_attempts),
        )
        query_id = infer_query_id_from_output(output)
        if str(query_id) not in set(allowed):
            raise ValueError(f"generated unsupported query id for public pages task: {query_id}")
        return rewrite_pages_public_task_output(
            output,
            task_id=str(self.task_id),
            scene_id=str(self.public_scene_id),
            query_id=str(query_id),
            query_probabilities=query_probabilities_from_pages_output(output, fallback_query_ids=allowed),
        )


__all__ = [
    "DEFAULT_QUERY_ID",
    "QUERY_ID_PARAM_KEYS",
    "QUERY_ID_WEIGHT_KEYS",
    "explicit_query_id_param",
    "force_query_id_params",
    "has_explicit_query_id_param",
    "normalize_query_id_params",
    "normalize_probability_map",
    "probability_map",
    "resolve_task_query_id_param",
    "rewrite_public_query_output",
    "strip_query_id_params",
]
