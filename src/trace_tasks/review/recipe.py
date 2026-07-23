"""Capture and load the canonical deterministic task-review recipe."""

from __future__ import annotations

from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass
from io import BytesIO
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

from PIL import Image

from trace_tasks.core.annotation_sanitization import (
    sanitize_trace_payload_for_public_annotation,
)
from trace_tasks.core.canonical import canonical_json_bytes
from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.reward_contracts import resolve_reward_contract
from trace_tasks.core.seed import hash64
from trace_tasks.core.source_layout_policy import (
    parse_public_task_id,
    uses_current_source_layout,
)
from trace_tasks.core.taxonomy import inject_taxonomy_metadata, resolve_task_taxonomy
from trace_tasks.tasks.registry import create_task

from .models import (
    REQUESTS_PER_TASK,
    ArtifactHashes,
    NonDeterministicGenerationError,
    RecipeCaptureError,
    RecipeManifest,
    RecipeShard,
    RecipeValidationError,
    ReviewProvenance,
    ReviewRequest,
)
from .provenance import (
    collect_review_provenance,
    default_repo_root,
    sha256_bytes,
)

GeneratorFn = Callable[[str, int, Mapping[str, Any], int], Any]


@dataclass(frozen=True)
class PreparedOutput:
    """Canonical semantic payload and rendering bytes for a generated output."""

    semantic_payload: Mapping[str, Any]
    png_bytes: bytes
    hashes: ArtifactHashes
    image_mode: str
    image_size: tuple[int, int]


@dataclass(frozen=True)
class _CaptureSpec:
    """Pickle-safe arguments for one independently captured request."""

    task_id: str
    query_id: str
    ordinal: int
    capture_seed: int
    base_params: Mapping[str, Any]
    max_attempts: int
    max_request_retries: int


def _default_generator(
    task_id: str,
    seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> Any:
    task = create_task(str(task_id))
    return task.generate(
        int(seed),
        params=dict(params),
        max_attempts=int(max_attempts),
    )


def _typed_value_dict(value: Any, *, name: str) -> dict[str, Any]:
    to_dict = getattr(value, "to_dict", None)
    if not callable(to_dict):
        raise RecipeCaptureError(
            f"generated output {name} does not implement to_dict()"
        )
    payload = to_dict()
    if not isinstance(payload, Mapping):
        raise RecipeCaptureError(
            f"generated output {name}.to_dict() must return an object"
        )
    return {str(key): item for key, item in payload.items()}


def resolve_output_query_id(output: Any) -> str:
    """Resolve one generated output's public query id."""

    query_id = str(getattr(output, "query_id", "") or "")
    trace_payload = getattr(output, "trace_payload", {})
    if not query_id and isinstance(trace_payload, Mapping):
        query_spec = trace_payload.get("query_spec")
        execution_trace = trace_payload.get("execution_trace")
        if isinstance(query_spec, Mapping):
            query_id = str(query_spec.get("query_id", "") or "")
        if not query_id and isinstance(execution_trace, Mapping):
            query_id = str(execution_trace.get("query_id", "") or "")
    return query_id or SINGLE_QUERY_ID


def _png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(
        buffer,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    return buffer.getvalue()


def _raw_pixel_hash(image: Image.Image) -> str:
    rgba = image.convert("RGBA")
    header = canonical_json_bytes(
        {"mode": "RGBA", "width": int(rgba.width), "height": int(rgba.height)}
    )
    return sha256_bytes(header + b"\0" + rgba.tobytes())


def semantic_payload(output: Any, *, task_id: str = "") -> dict[str, Any]:
    """Build the host-independent structured identity for one generated output."""

    image = getattr(output, "image", None)
    if not isinstance(image, Image.Image):
        raise RecipeCaptureError("generated output must contain a PIL image")
    answer = getattr(output, "answer_gt", None)
    annotation = getattr(output, "annotation_gt", None)
    answer_payload = _typed_value_dict(answer, name="answer_gt")
    annotation_payload = _typed_value_dict(annotation, name="annotation_gt")
    try:
        reward_contract = resolve_reward_contract(
            answer_type=str(answer_payload.get("type", "")),
            annotation_type=str(annotation_payload.get("type", "")),
        ).to_dict()
    except ValueError as exc:
        raise RecipeCaptureError(str(exc)) from exc
    trace_payload = getattr(output, "trace_payload", {})
    if not isinstance(trace_payload, Mapping):
        raise RecipeCaptureError("generated output trace_payload must be an object")
    sanitized_trace = sanitize_trace_payload_for_public_annotation(
        trace_payload,
        annotation_gt=annotation,
    )
    query_id = resolve_output_query_id(output)
    taxonomy_payload: dict[str, str] = {}
    if task_id:
        parts = parse_public_task_id(task_id)
        taxonomy = resolve_task_taxonomy(
            task_id,
            source_domain=parts.domain,
            source_scene_id=parts.scene_id,
        )
        sanitized_trace = inject_taxonomy_metadata(
            sanitized_trace,
            task_id=task_id,
            taxonomy=taxonomy,
            query_id=query_id,
            registered_domain=parts.domain,
            registered_scene_id=(
                None
                if uses_current_source_layout(task_id, domain=parts.domain)
                else parts.scene_id
            ),
        )
        taxonomy_payload = {
            "domain": parts.domain,
            "scene_id": parts.scene_id,
            "task_id": task_id,
            "query_id": query_id,
        }
    prompt = getattr(output, "prompt", None)
    if not isinstance(prompt, str) or not prompt.strip():
        raise RecipeCaptureError("generated output prompt must be a non-empty string")
    prompt_variants = getattr(output, "prompt_variants", {}) or {}
    versions = getattr(output, "task_versions", {}) or {}
    if not isinstance(prompt_variants, Mapping) or not isinstance(versions, Mapping):
        raise RecipeCaptureError(
            "generated prompt_variants and task_versions must be objects"
        )
    return {
        "scene_id": str(getattr(output, "scene_id", "") or ""),
        "query_id": query_id,
        "taxonomy": taxonomy_payload,
        "prompt": prompt,
        "prompt_variants": {
            str(key): str(item) for key, item in prompt_variants.items()
        },
        "answer_gt": answer_payload,
        "annotation_gt": annotation_payload,
        "reward_contract": reward_contract,
        "trace_payload": sanitized_trace,
        "versions": {str(key): str(item) for key, item in versions.items()},
    }


def prepare_output(output: Any, *, task_id: str = "") -> PreparedOutput:
    """Compute semantic, raw-pixel, and encoded-PNG identities."""

    image = getattr(output, "image", None)
    if not isinstance(image, Image.Image):
        raise RecipeCaptureError("generated output must contain a PIL image")
    semantic = semantic_payload(output, task_id=task_id)
    encoded = _png_bytes(image)
    hashes = ArtifactHashes(
        semantic_hash=sha256_bytes(canonical_json_bytes(semantic)),
        raw_pixel_hash=_raw_pixel_hash(image),
        png_hash=sha256_bytes(encoded),
    )
    return PreparedOutput(
        semantic_payload=semantic,
        png_bytes=encoded,
        hashes=hashes,
        image_mode=str(image.mode),
        image_size=(int(image.width), int(image.height)),
    )


def _declared_query_ids(task_id: str) -> tuple[str, ...]:
    task = create_task(str(task_id))
    raw = getattr(task, "supported_query_ids", ()) or ()
    if isinstance(raw, str):
        raw = (raw,)
    query_ids = tuple(sorted({str(value) for value in raw if str(value).strip()}))
    return query_ids or (SINGLE_QUERY_ID,)


def _normalized_task_query_ids(
    task_ids: Sequence[str],
    query_ids_by_task: Mapping[str, Sequence[str]] | None,
) -> dict[str, tuple[str, ...]]:
    provided = dict(query_ids_by_task or {})
    unknown = sorted(set(provided) - set(task_ids))
    if unknown:
        raise RecipeCaptureError(
            "query_ids_by_task contains unknown tasks: " + ", ".join(unknown)
        )
    resolved: dict[str, tuple[str, ...]] = {}
    for task_id in task_ids:
        raw = provided.get(task_id)
        if raw is None:
            query_ids = _declared_query_ids(task_id)
        else:
            if isinstance(raw, str):
                raw = (raw,)
            query_ids = tuple(sorted({str(value) for value in raw if str(value)}))
        if not query_ids:
            raise RecipeCaptureError(f"{task_id} has no declared query ids")
        if len(query_ids) > REQUESTS_PER_TASK:
            raise RecipeCaptureError(
                f"{task_id} declares {len(query_ids)} queries, exceeding the "
                f"{REQUESTS_PER_TASK}-request canonical recipe budget"
            )
        resolved[task_id] = query_ids
    return resolved


def _base_params(
    task_id: str,
    params_by_task: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    params = dict((params_by_task or {}).get(task_id, {}))
    forbidden = [
        key
        for key in params
        if key in {"query_id", "query_variant"}
        or str(key).endswith("sample_cursor")
        or str(key).endswith("sampling_index")
    ]
    if forbidden:
        raise RecipeCaptureError(
            f"{task_id} base params cannot override recipe controls: {sorted(forbidden)}"
        )
    return params


def _request_seed(
    *,
    capture_seed: int,
    task_id: str,
    query_id: str,
    sample_cursor: int,
    retry: int,
) -> int:
    namespace = f"trace-review-recipe-v1:{task_id}:{query_id}:retry:{int(retry)}"
    return int(hash64(int(capture_seed), namespace, int(sample_cursor)))


def _capture_request(
    *,
    task_id: str,
    query_id: str,
    ordinal: int,
    capture_seed: int,
    base_params: Mapping[str, Any],
    max_attempts: int,
    max_request_retries: int,
    generator: GeneratorFn,
) -> ReviewRequest:
    parts = parse_public_task_id(task_id)
    last_error: Exception | None = None
    for retry in range(max_request_retries + 1):
        seed = _request_seed(
            capture_seed=capture_seed,
            task_id=task_id,
            query_id=query_id,
            sample_cursor=ordinal,
            retry=retry,
        )
        params = dict(base_params)
        params["query_id"] = query_id
        params["_sample_cursor"] = int(ordinal)
        try:
            first_output = generator(task_id, seed, params, max_attempts)
            second_output = generator(task_id, seed, params, max_attempts)
            first = prepare_output(first_output, task_id=task_id)
            second = prepare_output(second_output, task_id=task_id)
        except Exception as exc:
            last_error = exc
            continue
        first_query = resolve_output_query_id(first_output)
        second_query = resolve_output_query_id(second_output)
        if first_query != query_id or second_query != query_id:
            last_error = RecipeCaptureError(
                f"requested query {query_id!r}, observed {first_query!r} and {second_query!r}"
            )
            continue
        first_scene = str(getattr(first_output, "scene_id", "") or "")
        second_scene = str(getattr(second_output, "scene_id", "") or "")
        if first_scene != parts.scene_id or second_scene != parts.scene_id:
            last_error = RecipeCaptureError(
                f"requested scene {parts.scene_id!r}, observed {first_scene!r} "
                f"and {second_scene!r}"
            )
            continue
        if first.hashes != second.hashes:
            differing = [
                name
                for name in ("semantic_hash", "raw_pixel_hash", "png_hash")
                if getattr(first.hashes, name) != getattr(second.hashes, name)
            ]
            raise NonDeterministicGenerationError(
                f"{task_id} query={query_id} ordinal={ordinal} changed across "
                f"identical generations: {', '.join(differing)}"
            )
        return ReviewRequest(
            task_id=task_id,
            domain=parts.domain,
            scene_id=parts.scene_id,
            query_id=query_id,
            ordinal=ordinal,
            seed=seed,
            sample_cursor=ordinal,
            params=params,
            max_attempts=max_attempts,
            retry=retry,
            hashes=first.hashes,
        )
    detail = f": {last_error}" if last_error is not None else ""
    raise RecipeCaptureError(
        f"{task_id} query={query_id} ordinal={ordinal} failed after "
        f"{max_request_retries + 1} deterministic choices{detail}"
    ) from last_error


def _capture_spec(spec: _CaptureSpec, generator: GeneratorFn) -> ReviewRequest:
    """Capture one request from a serializable scheduling record."""

    return _capture_request(
        task_id=spec.task_id,
        query_id=spec.query_id,
        ordinal=spec.ordinal,
        capture_seed=spec.capture_seed,
        base_params=spec.base_params,
        max_attempts=spec.max_attempts,
        max_request_retries=spec.max_request_retries,
        generator=generator,
    )


def _process_pool_context() -> mp.context.BaseContext | None:
    """Prefer process start methods that do not inherit mutable task state."""

    for method in ("forkserver", "spawn"):
        try:
            return mp.get_context(method)
        except ValueError:
            continue
    return None


def _parallel_capture(
    specs: Sequence[_CaptureSpec],
    *,
    generator: GeneratorFn,
    workers: int,
    use_processes: bool,
) -> list[ReviewRequest]:
    """Capture with a bounded queue while restoring canonical spec order."""

    if workers == 1:
        return [_capture_spec(spec, generator) for spec in specs]
    executor: Executor
    if use_processes:
        context = _process_pool_context()
        executor = ProcessPoolExecutor(max_workers=workers, mp_context=context)
    else:
        # Custom generators are frequently local test/application callables and
        # therefore not pickleable. Threads preserve that API while each spec
        # still creates its own task outputs and never shares a task instance.
        executor = ThreadPoolExecutor(max_workers=workers)
    results: list[ReviewRequest | None] = [None] * len(specs)
    pending: dict[Future[ReviewRequest], int] = {}
    next_index = 0
    max_in_flight = max(workers, workers * 2)
    try:
        while next_index < len(specs) and len(pending) < max_in_flight:
            future = executor.submit(_capture_spec, specs[next_index], generator)
            pending[future] = next_index
            next_index += 1
        while pending:
            completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in sorted(completed, key=lambda item: pending[item]):
                index = pending.pop(future)
                results[index] = future.result()
                if next_index < len(specs):
                    replacement = executor.submit(
                        _capture_spec,
                        specs[next_index],
                        generator,
                    )
                    pending[replacement] = next_index
                    next_index += 1
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    if any(result is None for result in results):  # pragma: no cover - defensive
        raise RecipeCaptureError("parallel capture ended without every request")
    return [result for result in results if result is not None]


def _jsonl_bytes(records: Sequence[ReviewRequest]) -> bytes:
    lines = [
        json.dumps(
            record.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for record in records
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def manifest_digest(manifest: RecipeManifest) -> str:
    """Compute a recipe manifest's identity, excluding its self-digest field."""

    return sha256_bytes(canonical_json_bytes(manifest.to_dict(include_digest=False)))


def _write_captured_recipe(
    recipe_root: Path,
    *,
    requests: Sequence[ReviewRequest],
    provenance: ReviewProvenance,
) -> RecipeManifest:
    if recipe_root.exists():
        raise FileExistsError(
            f"refusing to overwrite existing recipe root: {recipe_root}"
        )
    recipe_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{recipe_root.name}.capture-",
            dir=str(recipe_root.parent),
        )
    )
    try:
        by_domain: dict[str, list[ReviewRequest]] = {}
        for request in requests:
            by_domain.setdefault(request.domain, []).append(request)
        shards: list[RecipeShard] = []
        for domain in sorted(by_domain):
            domain_requests = sorted(
                by_domain[domain], key=lambda row: (row.task_id, row.ordinal)
            )
            payload = _jsonl_bytes(domain_requests)
            relative = Path("requests") / f"{domain}.jsonl"
            path = stage / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            shards.append(
                RecipeShard(
                    domain=domain,
                    path=relative.as_posix(),
                    request_count=len(domain_requests),
                    sha256=sha256_bytes(payload),
                )
            )
        task_count = len({request.task_id for request in requests})
        provisional = RecipeManifest(
            provenance=provenance,
            shards=tuple(shards),
            task_count=task_count,
            request_count=len(requests),
            recipe_digest="sha256:" + ("0" * 64),
        )
        manifest = RecipeManifest(
            provenance=provenance,
            shards=tuple(shards),
            task_count=task_count,
            request_count=len(requests),
            recipe_digest=manifest_digest(provisional),
        )
        (stage / "manifest.json").write_text(
            json.dumps(
                manifest.to_dict(),
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        stage.rename(recipe_root)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def capture_recipe(
    task_ids: Sequence[str],
    recipe_root: Path | str,
    *,
    seed: int = 42,
    requests_per_task: int = REQUESTS_PER_TASK,
    max_attempts: int = 100,
    max_request_retries: int = 32,
    workers: int = 1,
    params_by_task: Mapping[str, Mapping[str, Any]] | None = None,
    query_ids_by_task: Mapping[str, Sequence[str]] | None = None,
    generator: GeneratorFn | None = None,
    provenance: ReviewProvenance | None = None,
    repo_root: Path | str | None = None,
    source_repository: str = "maveryn/trace",
    require_clean_source: bool = True,
) -> RecipeManifest:
    """Capture exactly 25 query-stratified, double-generated requests per task."""

    target_root = Path(recipe_root)
    if target_root.exists():
        raise FileExistsError(
            f"refusing to overwrite existing recipe root: {target_root}"
        )
    if isinstance(requests_per_task, bool) or requests_per_task != REQUESTS_PER_TASK:
        raise RecipeCaptureError(
            f"trace-review-recipe-v1 requires exactly {REQUESTS_PER_TASK} requests per task"
        )
    if isinstance(max_attempts, bool) or int(max_attempts) < 1:
        raise RecipeCaptureError("max_attempts must be at least 1")
    if isinstance(max_request_retries, bool) or int(max_request_retries) < 0:
        raise RecipeCaptureError("max_request_retries must be non-negative")
    if isinstance(workers, bool) or int(workers) < 1:
        raise RecipeCaptureError("workers must be at least 1")
    selected_tasks = sorted({str(task_id) for task_id in task_ids})
    if not selected_tasks:
        raise RecipeCaptureError("capture requires at least one task id")
    for task_id in selected_tasks:
        parse_public_task_id(task_id)
    unknown_params = sorted(set(params_by_task or {}) - set(selected_tasks))
    if unknown_params:
        raise RecipeCaptureError(
            "params_by_task contains unknown tasks: " + ", ".join(unknown_params)
        )

    task_queries = _normalized_task_query_ids(selected_tasks, query_ids_by_task)
    if provenance is None:
        provenance = collect_review_provenance(
            repo_root=repo_root or default_repo_root(),
            task_query_ids=task_queries,
            source_repository=source_repository,
            require_clean_source=require_clean_source,
        )
    generate = generator or _default_generator
    specs: list[_CaptureSpec] = []
    for task_id in selected_tasks:
        queries = task_queries[task_id]
        base = _base_params(task_id, params_by_task)
        # The sorted round-robin assignment covers every query before filling
        # subsequent positions and is independent of registry import order.
        for ordinal in range(REQUESTS_PER_TASK):
            query_id = queries[ordinal % len(queries)]
            specs.append(
                _CaptureSpec(
                    task_id=task_id,
                    query_id=query_id,
                    ordinal=ordinal,
                    capture_seed=int(seed),
                    base_params=base,
                    max_attempts=int(max_attempts),
                    max_request_retries=int(max_request_retries),
                )
            )
    captured = _parallel_capture(
        specs,
        generator=generate,
        workers=int(workers),
        use_processes=generator is None,
    )
    return _write_captured_recipe(
        target_root,
        requests=captured,
        provenance=provenance,
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecipeValidationError(f"invalid JSON object {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise RecipeValidationError(f"expected a JSON object: {path}")
    return {str(key): item for key, item in raw.items()}


def load_recipe(
    recipe_root: Path | str,
) -> tuple[RecipeManifest, list[ReviewRequest]]:
    """Load and fully validate a versioned manifest and all domain shards."""

    root = Path(recipe_root)
    manifest = RecipeManifest.from_dict(_load_json_object(root / "manifest.json"))
    actual_digest = manifest_digest(manifest)
    if actual_digest != manifest.recipe_digest:
        raise RecipeValidationError(
            f"recipe manifest digest mismatch: expected {manifest.recipe_digest}, "
            f"observed {actual_digest}"
        )
    requests: list[ReviewRequest] = []
    root_resolved = root.resolve()
    for shard in manifest.shards:
        path = root / shard.path
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise RecipeValidationError(f"missing recipe shard: {path}") from exc
        if root_resolved not in resolved.parents:
            raise RecipeValidationError(
                f"recipe shard escapes recipe root: {shard.path}"
            )
        payload = path.read_bytes()
        observed_digest = sha256_bytes(payload)
        if observed_digest != shard.sha256:
            raise RecipeValidationError(
                f"recipe shard digest mismatch for {shard.path}: expected "
                f"{shard.sha256}, observed {observed_digest}"
            )
        shard_requests: list[ReviewRequest] = []
        for line_number, raw_line in enumerate(payload.decode("utf-8").splitlines(), 1):
            if not raw_line.strip():
                continue
            try:
                raw_request = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise RecipeValidationError(
                    f"invalid JSON on {shard.path}:{line_number}: {exc}"
                ) from exc
            request = ReviewRequest.from_dict(raw_request)
            if request.domain != shard.domain:
                raise RecipeValidationError(
                    f"request domain mismatch on {shard.path}:{line_number}"
                )
            shard_requests.append(request)
        if len(shard_requests) != shard.request_count:
            raise RecipeValidationError(
                f"request count mismatch for {shard.path}: expected "
                f"{shard.request_count}, observed {len(shard_requests)}"
            )
        requests.extend(shard_requests)
    if len(requests) != manifest.request_count:
        raise RecipeValidationError("manifest request count does not match its shards")
    keys = [(request.task_id, request.ordinal) for request in requests]
    if len(keys) != len(set(keys)):
        raise RecipeValidationError("recipe contains duplicate task/ordinal requests")
    by_task: dict[str, list[int]] = {}
    for request in requests:
        by_task.setdefault(request.task_id, []).append(request.ordinal)
    expected_ordinals = list(range(REQUESTS_PER_TASK))
    for task_id, ordinals in by_task.items():
        if sorted(ordinals) != expected_ordinals:
            raise RecipeValidationError(
                f"{task_id} must contain every ordinal from 0 through "
                f"{REQUESTS_PER_TASK - 1}"
            )
    if len(by_task) != manifest.task_count:
        raise RecipeValidationError("manifest task count does not match its shards")
    return manifest, sorted(requests, key=lambda row: (row.task_id, row.ordinal))


def filter_requests(
    requests: Iterable[ReviewRequest],
    *,
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
    scene_ids: Sequence[str] | None = None,
    query_ids: Sequence[str] | None = None,
) -> list[ReviewRequest]:
    """Apply optional public taxonomy filters in deterministic order."""

    tasks = {str(value) for value in task_ids or ()}
    domain_values = {str(value) for value in domains or ()}
    bare_scenes: set[str] = set()
    qualified_scenes: set[tuple[str, str]] = set()
    for value in scene_ids or ():
        text = str(value)
        if "/" not in text:
            bare_scenes.add(text)
            continue
        domain, separator, scene_id = text.partition("/")
        if not separator or not domain or not scene_id or "/" in scene_id:
            raise RecipeValidationError(
                "scene filters must be SCENE_ID or DOMAIN/SCENE_ID"
            )
        qualified_scenes.add((domain, scene_id))
    queries = {str(value) for value in query_ids or ()}
    selected = [
        request
        for request in requests
        if (not tasks or request.task_id in tasks)
        and (not domain_values or request.domain in domain_values)
        and (
            (not bare_scenes and not qualified_scenes)
            or request.scene_id in bare_scenes
            or (request.domain, request.scene_id) in qualified_scenes
        )
        and (not queries or request.query_id in queries)
    ]
    return sorted(selected, key=lambda row: (row.task_id, row.ordinal))


def iter_recipe_requests(
    recipe_root: Path | str,
    *,
    task_ids: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
    scene_ids: Sequence[str] | None = None,
    query_ids: Sequence[str] | None = None,
) -> Iterable[ReviewRequest]:
    """Yield validated recipe rows matching optional taxonomy filters."""

    _, requests = load_recipe(recipe_root)
    yield from filter_requests(
        requests,
        task_ids=task_ids,
        domains=domains,
        scene_ids=scene_ids,
        query_ids=query_ids,
    )


__all__ = [
    "GeneratorFn",
    "PreparedOutput",
    "capture_recipe",
    "filter_requests",
    "iter_recipe_requests",
    "load_recipe",
    "manifest_digest",
    "prepare_output",
    "resolve_output_query_id",
    "semantic_payload",
]
