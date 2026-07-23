"""Trace dataset export helpers for the local RLVR stack."""

from __future__ import annotations

import json
import os
import math
import re
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping

from tqdm.auto import tqdm
from PIL import Image

from .taxonomy import resolve_task_query_id, resolve_task_taxonomy


PromptVariantMode = Literal["active", "answer_only", "answer_and_annotation"]
PromptVariantInput = Literal["active", "answer", "answer_only", "annotation", "answer_and_annotation"] | str
ImagePathMode = Literal["relative", "absolute", "dataset_relative"]
ImageStorageMode = Literal["path_dict", "embedded_bytes"]
OutputFormat = Literal["jsonl", "parquet"]
_PARQUET_JSON_COLUMNS = ("answer_gt", "annotation_gt", "reward_contract", "trace_ref")
_PARQUET_WRITE_CHUNK_SIZE = 4096
_PROGRESS_DISABLED_VALUES = {"0", "false", "no", "off"}
_EXPORT_PROGRESS_ENABLED = (
    os.environ.get("TRACE_EXPORT_PROGRESS")
    or os.environ.get("TRACE_BUILD_PROGRESS")
    or "1"
).strip().lower() not in _PROGRESS_DISABLED_VALUES

_PROMPT_VARIANT_ALIASES: dict[str, PromptVariantMode] = {
    "active": "active",
    "answer": "answer_only",
    "answer_only": "answer_only",
    "answer_and_annotation": "answer_and_annotation",
    "annotation": "answer_and_annotation",
}

_ANSWER_ONLY_SCHEMA_LINE_RE = re.compile(
    r'^Use a valid JSON object with key "answer" for the final answer\.\s*$',
    re.IGNORECASE,
)
_ANSWER_AND_ANNOTATION_SCHEMA_LINE_RE = re.compile(
    r'^Use a valid JSON object with keys (?:"annotation" and "answer" in that order|"answer" and "annotation") for the final answer\.\s*$',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RLVRExportResult:
    """Summary for one Trace-to-RLVR export run."""

    source_dataset_root: Path
    train_instances_path: Path
    output_path: Path
    output_format: OutputFormat
    prompt_variant: PromptVariantMode
    image_path_mode: ImagePathMode
    row_count: int


@dataclass(frozen=True)
class ExportedImageInfo:
    """Export-time image geometry needed to keep annotations in image space."""

    original_width: int
    original_height: int
    exported_width: int
    exported_height: int

    @property
    def scale_x(self) -> float:
        return float(self.exported_width) / float(max(1, self.original_width))

    @property
    def scale_y(self) -> float:
        return float(self.exported_height) / float(max(1, self.original_height))

    def to_dict(self) -> dict[str, int]:
        return {
            "original_width": int(self.original_width),
            "original_height": int(self.original_height),
            "exported_width": int(self.exported_width),
            "exported_height": int(self.exported_height),
        }


def _normalize_prompt_variant(prompt_variant: str) -> PromptVariantMode:
    normalized = _PROMPT_VARIANT_ALIASES.get(prompt_variant.strip().lower())
    if normalized is None:
        raise ValueError(f"unsupported prompt variant: {prompt_variant}")
    return normalized

def _iter_chunks(rows: list[dict[str, Any]], chunk_size: int) -> Iterable[list[dict[str, Any]]]:
    """Yield fixed-size row chunks in order."""

    for start in range(0, len(rows), chunk_size):
        yield rows[start : start + chunk_size]


def _iter_iterable_chunks(rows: Iterable[Any], chunk_size: int) -> Iterable[list[Any]]:
    """Yield fixed-size chunks from any row iterable."""

    chunk: list[Any] = []
    for row in rows:
        chunk.append(row)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _resolve_parquet_cpu_count(parquet_cpu_count: int | None) -> int | None:
    """Normalize the requested parquet CPU count."""

    if parquet_cpu_count is None:
        return None
    parsed = int(parquet_cpu_count)
    if parsed < 0:
        raise ValueError("parquet_cpu_count must be >= 0")
    if parsed == 0:
        return max(1, int(os.cpu_count() or 1))
    return parsed


def _resolve_parquet_row_worker_count(parquet_cpu_count: int | None) -> int:
    """Resolve Python-side row preparation workers for parquet export."""

    env_value = os.environ.get("TRACE_EXPORT_PARQUET_ROW_WORKERS")
    if env_value is not None and env_value.strip():
        parsed = int(env_value)
        if parsed < 1:
            raise ValueError("TRACE_EXPORT_PARQUET_ROW_WORKERS must be >= 1")
        return parsed
    requested_cpu_count = _resolve_parquet_cpu_count(parquet_cpu_count)
    if requested_cpu_count is None:
        return 1
    return max(1, int(requested_cpu_count))


def resolve_train_instances_source(path: str | Path) -> tuple[Path, Path]:
    """Resolve a Trace dataset root plus its train-instances file."""

    candidate = Path(path).expanduser().resolve()
    if candidate.is_dir():
        train_instances_path = candidate / "train_instances.jsonl"
        dataset_root = candidate
    else:
        train_instances_path = candidate
        dataset_root = candidate.parent

    if not train_instances_path.exists():
        raise FileNotFoundError(f"Trace train-instances file not found: {train_instances_path}")
    if train_instances_path.name != "train_instances.jsonl":
        raise ValueError(
            "Trace RLVR export expects a dataset root or a file named "
            f"'train_instances.jsonl', got: {train_instances_path.name}"
        )
    return dataset_root, train_instances_path


def resolve_export_output_path(
    output_path: str | Path,
    *,
    output_format: OutputFormat | None = None,
) -> tuple[Path, OutputFormat]:
    """Resolve the concrete export file path plus normalized format."""

    candidate = Path(output_path).expanduser()
    suffix = candidate.suffix.lower()

    inferred_format: OutputFormat | None = None
    if suffix == ".parquet":
        inferred_format = "parquet"
    elif suffix in {".jsonl", ".json"}:
        inferred_format = "jsonl"

    final_format = output_format or inferred_format or "jsonl"
    if final_format not in {"jsonl", "parquet"}:
        raise ValueError(f"unsupported RLVR export format: {final_format}")

    if inferred_format is not None and output_format is not None and inferred_format != output_format:
        raise ValueError(
            "output format mismatch: "
            f"path {candidate} implies {inferred_format}, but --format={output_format}"
        )

    if candidate.exists() and candidate.is_dir():
        filename = "train.parquet" if final_format == "parquet" else "train.jsonl"
        return candidate / filename, final_format
    if candidate.suffix:
        return candidate, final_format

    candidate.mkdir(parents=True, exist_ok=True)
    filename = "train.parquet" if final_format == "parquet" else "train.jsonl"
    return candidate / filename, final_format


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    progress_enabled = _EXPORT_PROGRESS_ENABLED
    records: list[dict[str, Any]] = []
    total_bytes = path.stat().st_size
    with (
        path.open("r", encoding="utf-8") as handle,
        tqdm(
            total=total_bytes,
            desc="Read Trace JSONL",
            unit="B",
            unit_scale=True,
            dynamic_ncols=True,
            disable=not progress_enabled,
        ) as progress_bar,
    ):
        for line_number, raw_line in enumerate(handle, start=1):
            progress_bar.update(len(raw_line.encode("utf-8")))
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object on line {line_number} of {path}")
            records.append(payload)
    return records


def _select_prompt(record: Mapping[str, Any], prompt_variant: PromptVariantMode) -> str:
    prompt = str(record.get("prompt", ""))
    if prompt_variant == "active":
        return prompt

    prompt_variants = record.get("prompt_variants")
    if isinstance(prompt_variants, Mapping):
        candidate = prompt_variants.get(prompt_variant)
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return prompt


def _normalize_multimodal_prompt(prompt: str, *, image_count: int) -> str:
    """Prefix RLVR image placeholders so exported multimodal rows match their image payload.

    vLLM's multimodal replacement expects prompt tokens to contain one `<image>` marker
    per image item. Trace prompts intentionally avoid transport-specific placeholders, so
    RLVR export normalizes them into the local RLVR multimodal convention.
    """

    if image_count <= 0:
        return prompt
    cleaned_prompt = prompt.replace("<image>", "").strip()
    return f"{'<image>' * image_count}{cleaned_prompt}"


def _strip_redundant_rlvr_output_contract(prompt: str, *, prompt_variant: PromptVariantMode) -> str:
    if prompt_variant == "active":
        return prompt

    schema_line_re = (
        _ANSWER_ONLY_SCHEMA_LINE_RE
        if prompt_variant == "answer_only"
        else _ANSWER_AND_ANNOTATION_SCHEMA_LINE_RE
    )
    filtered_lines = [line for line in prompt.splitlines() if not schema_line_re.match(line.strip())]

    cleaned_lines: list[str] = []
    previous_blank = False
    for line in filtered_lines:
        is_blank = not line.strip()
        if is_blank and previous_blank:
            continue
        cleaned_lines.append(line.rstrip())
        previous_blank = is_blank

    return "\n".join(cleaned_lines).strip()


def _build_prompt_columns(record: Mapping[str, Any], *, image_count: int) -> dict[str, str]:
    """Export both public Trace prompt variants so one parquet can drive multiple ablations."""

    prompt_answer = _normalize_multimodal_prompt(
        _strip_redundant_rlvr_output_contract(
            _select_prompt(record, "answer_only"),
            prompt_variant="answer_only",
        ),
        image_count=image_count,
    )
    return {
        "prompt_active": _normalize_multimodal_prompt(_select_prompt(record, "active"), image_count=image_count),
        "prompt_answer": prompt_answer,
        "prompt_answer_only": prompt_answer,
        "prompt_answer_and_annotation": _normalize_multimodal_prompt(
            _strip_redundant_rlvr_output_contract(
                _select_prompt(record, "answer_and_annotation"),
                prompt_variant="answer_and_annotation",
            ),
            image_count=image_count,
        ),
    }


def _iter_image_paths(record: Mapping[str, Any], dataset_root: Path) -> Iterable[Path]:
    images = record.get("images")
    if not isinstance(images, list):
        raise ValueError("Trace RLVR export requires images to be a list")

    for image in images:
        if isinstance(image, Mapping):
            raw_path = image.get("path")
        elif isinstance(image, str):
            raw_path = image
        else:
            raise ValueError(f"unsupported image entry for RLVR export: {image!r}")

        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"image entry is missing a usable path: {image!r}")

        image_path = Path(raw_path)
        yield image_path if image_path.is_absolute() else (dataset_root / image_path)


def _format_image_path(
    image_path: Path,
    *,
    dataset_root: Path,
    output_parent: Path,
    image_path_mode: ImagePathMode,
) -> str:
    resolved = image_path.resolve()
    if image_path_mode == "absolute":
        return str(resolved)
    if image_path_mode == "dataset_relative":
        try:
            return str(resolved.relative_to(dataset_root.resolve()).as_posix())
        except ValueError as exc:
            raise ValueError(
                f"cannot express image path {resolved} relative to dataset root {dataset_root}"
            ) from exc
    if image_path_mode != "relative":
        raise ValueError(f"unsupported image-path mode: {image_path_mode}")
    relative = os.path.relpath(str(resolved), start=str(output_parent.resolve()))
    return str(Path(relative).as_posix())


def _resize_image_bytes_to_pixel_cap(
    image_path: Path,
    *,
    max_pixels: int | None,
) -> tuple[bytes, str, ExportedImageInfo]:
    """Read one image, optionally resize it, and return bytes plus geometry."""

    suffix_format = image_path.suffix.lower().lstrip(".") or "png"
    with Image.open(image_path) as image:
        image.load()
        original_width, original_height = int(image.width), int(image.height)
        exported = image
        if max_pixels is not None and int(max_pixels) > 0:
            pixel_count = int(image.width) * int(image.height)
            if pixel_count > int(max_pixels):
                resize_factor = math.sqrt(float(max_pixels) / float(pixel_count))
                exported_width = max(1, int(image.width * resize_factor))
                exported_height = max(1, int(image.height * resize_factor))
                while exported_width * exported_height > int(max_pixels):
                    if exported_width >= exported_height:
                        exported_width -= 1
                    else:
                        exported_height -= 1
                exported = image.resize((exported_width, exported_height), Image.Resampling.LANCZOS)

        if exported.mode not in {"RGB", "RGBA", "L"}:
            exported = exported.convert("RGB")
        buffer = BytesIO()
        save_format = "PNG" if suffix_format in {"", "png"} else suffix_format.upper()
        if save_format == "JPG":
            save_format = "JPEG"
        try:
            exported.save(buffer, format=save_format)
        except KeyError:
            save_format = "PNG"
            suffix_format = "png"
            exported.save(buffer, format=save_format)

        info = ExportedImageInfo(
            original_width=original_width,
            original_height=original_height,
            exported_width=int(exported.width),
            exported_height=int(exported.height),
        )
        return buffer.getvalue(), suffix_format, info


def _build_exported_images(
    train_record: Mapping[str, Any],
    *,
    dataset_root: Path,
    output_parent: Path,
    image_path_mode: ImagePathMode,
    image_storage_mode: ImageStorageMode,
    max_embedded_image_pixels: int | None = None,
) -> tuple[list[dict[str, Any]], list[ExportedImageInfo]]:
    """Build RLVR image records in the path-dict shape that the loader normalizes."""

    exported: list[dict[str, Any]] = []
    image_infos: list[ExportedImageInfo] = []
    for image_path in _iter_image_paths(train_record, dataset_root):
        if image_storage_mode == "embedded_bytes":
            image_bytes, image_format, image_info = _resize_image_bytes_to_pixel_cap(
                image_path,
                max_pixels=max_embedded_image_pixels,
            )
            image_infos.append(image_info)
            exported.append(
                {
                    "bytes": image_bytes,
                    "format": image_format,
                }
            )
            continue
        if image_storage_mode != "path_dict":
            raise ValueError(f"unsupported image-storage mode: {image_storage_mode}")
        exported.append(
            {
                "path": _format_image_path(
                    image_path,
                    dataset_root=dataset_root,
                    output_parent=output_parent,
                    image_path_mode=image_path_mode,
                )
            }
        )
    return exported, image_infos


def _scale_point(value: Any, *, scale_x: float, scale_y: float) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"expected point [x,y], got {value!r}")
    return [round(float(value[0]) * float(scale_x), 3), round(float(value[1]) * float(scale_y), 3)]


def _scale_bbox(value: Any, *, scale_x: float, scale_y: float) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"expected bbox [x0,y0,x1,y1], got {value!r}")
    return [
        round(float(value[0]) * float(scale_x), 3),
        round(float(value[1]) * float(scale_y), 3),
        round(float(value[2]) * float(scale_x), 3),
        round(float(value[3]) * float(scale_y), 3),
    ]


def _scale_segment(value: Any, *, scale_x: float, scale_y: float) -> list[list[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"expected segment [[x0,y0],[x1,y1]], got {value!r}")
    return [
        _scale_point(value[0], scale_x=scale_x, scale_y=scale_y),
        _scale_point(value[1], scale_x=scale_x, scale_y=scale_y),
    ]


def _scale_annotation_value(
    annotation_type: str,
    value: Any,
    *,
    scale_x: float,
    scale_y: float,
) -> Any:
    """Scale one public annotation value into exported-image pixel space."""

    if annotation_type == "point":
        return _scale_point(value, scale_x=scale_x, scale_y=scale_y)
    if annotation_type in {"point_set", "point_sequence"}:
        return [_scale_point(item, scale_x=scale_x, scale_y=scale_y) for item in value]
    if annotation_type == "point_map":
        return {str(key): _scale_point(item, scale_x=scale_x, scale_y=scale_y) for key, item in value.items()}
    if annotation_type == "point_set_map":
        return {
            str(key): [_scale_point(item, scale_x=scale_x, scale_y=scale_y) for item in items]
            for key, items in value.items()
        }
    if annotation_type == "bbox":
        return _scale_bbox(value, scale_x=scale_x, scale_y=scale_y)
    if annotation_type in {"bbox_set", "bbox_sequence"}:
        return [_scale_bbox(item, scale_x=scale_x, scale_y=scale_y) for item in value]
    if annotation_type == "bbox_map":
        return {str(key): _scale_bbox(item, scale_x=scale_x, scale_y=scale_y) for key, item in value.items()}
    if annotation_type == "bbox_set_map":
        return {
            str(key): [_scale_bbox(item, scale_x=scale_x, scale_y=scale_y) for item in items]
            for key, items in value.items()
        }
    if annotation_type == "segment":
        return _scale_segment(value, scale_x=scale_x, scale_y=scale_y)
    if annotation_type == "segment_set":
        return [_scale_segment(item, scale_x=scale_x, scale_y=scale_y) for item in value]
    raise ValueError(f"unsupported annotation type for image scaling: {annotation_type}")


def _scale_annotation_gt_for_export(
    annotation_gt: Mapping[str, Any],
    *,
    image_infos: list[ExportedImageInfo],
) -> dict[str, Any]:
    """Return annotation_gt in exported-image pixel coordinates."""

    if not image_infos:
        return dict(annotation_gt)
    if len(image_infos) != 1:
        if any((info.exported_width, info.exported_height) != (info.original_width, info.original_height) for info in image_infos):
            raise ValueError("cannot scale annotation_gt for multi-image rows without per-image annotation ownership")
        return dict(annotation_gt)

    info = image_infos[0]
    annotation_type = str(annotation_gt.get("type", "")).strip()
    if not annotation_type:
        raise ValueError("annotation_gt.type is required for image scaling")
    scale_x = info.scale_x
    scale_y = info.scale_y
    if abs(scale_x - 1.0) < 1e-12 and abs(scale_y - 1.0) < 1e-12:
        return dict(annotation_gt)
    return {
        **dict(annotation_gt),
        "value": _scale_annotation_value(
            annotation_type,
            annotation_gt.get("value"),
            scale_x=scale_x,
            scale_y=scale_y,
        ),
    }


def _build_curriculum_assignments(records: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Assign stable task-local curriculum buckets without difficulty scores."""

    task_rows: dict[str, list[str]] = {}
    for record in records:
        instance_id = str(record.get("instance_id", "")).strip()
        if not instance_id:
            raise ValueError("Trace RLVR export requires instance_id")
        task = str(record.get("task", "")).strip()
        if not task:
            raise ValueError(f"Trace RLVR export requires task on {instance_id}")
        task_rows.setdefault(task, []).append(instance_id)

    assignments: dict[str, dict[str, Any]] = {}
    for task, instance_ids in task_rows.items():
        bucket_id = f"{task}::q0"
        for instance_id in sorted(instance_ids):
            assignments[instance_id] = {
                "difficulty_bin": 0,
                "bucket_id_str": bucket_id,
            }
    return assignments


def _json_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped[0] in "{[":
            parsed = json.loads(stripped)
            return parsed if isinstance(parsed, Mapping) else None
    return None


def _trace_shard_path(dataset_root: Path, shard_id: str) -> Path:
    traces_root = dataset_root / "traces"
    candidate = traces_root / shard_id
    if candidate.exists():
        return candidate
    if not shard_id.endswith(".zst"):
        zstd_candidate = traces_root / f"{shard_id}.zst"
        if zstd_candidate.exists():
            return zstd_candidate
    raise FileNotFoundError(f"Trace sidecar shard not found: {candidate}")


def _iter_trace_shard_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".zst":
        import io
        import zstandard as zstd

        with path.open("rb") as handle:
            stream = zstd.ZstdDecompressor().stream_reader(handle)
            with io.TextIOWrapper(stream, encoding="utf-8") as text:
                for line in text:
                    yield json.loads(line)
        return

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def _query_fields_from_trace_record(trace_record: Mapping[str, Any]) -> dict[str, str]:
    query_spec = trace_record.get("query_spec") if isinstance(trace_record.get("query_spec"), Mapping) else {}
    execution_trace = (
        trace_record.get("execution_trace") if isinstance(trace_record.get("execution_trace"), Mapping) else {}
    )
    render_spec = trace_record.get("render_spec") if isinstance(trace_record.get("render_spec"), Mapping) else {}
    scene_ir = trace_record.get("scene_ir") if isinstance(trace_record.get("scene_ir"), Mapping) else {}
    taxonomy = trace_record.get("taxonomy") if isinstance(trace_record.get("taxonomy"), Mapping) else {}

    query_query_id = query_spec.get("query_id")
    execution_query_id = execution_trace.get("query_id")
    if query_query_id is not None and execution_query_id is not None:
        if str(query_query_id) != str(execution_query_id):
            raise ValueError(
                "Trace sidecar query_id mismatch: "
                f"query_spec={query_query_id!r} execution_trace={execution_query_id!r}"
            )

    query_id = query_query_id if query_query_id is not None else execution_query_id
    scene_variant = render_spec.get("scene_variant")
    if scene_variant is None:
        scene_variant = execution_trace.get("scene_variant")

    scene_id = taxonomy.get("scene_id") or query_spec.get("scene_id") or render_spec.get("scene_id")
    if scene_id is None:
        scene_id = scene_ir.get("scene_id")
    query_id = (
        taxonomy.get("query_id")
        or query_spec.get("query_id")
        or execution_trace.get("query_id")
        or resolve_task_query_id(query_id="" if query_id is None else str(query_id), trace_payload=trace_record)
    )

    return {
        "scene_variant": "" if scene_variant is None else str(scene_variant),
        "scene_id": "" if scene_id is None else str(scene_id),
        "query_id": "" if query_id is None else str(query_id),
    }


def _fields_from_train_record(record: Mapping[str, Any]) -> dict[str, str]:
    task_id = str(record.get("task", "")).strip()
    scene_id = str(record.get("scene_id", "")).strip()
    if not scene_id:
        raise ValueError("Trace train record requires scene_id")
    taxonomy = resolve_task_taxonomy(
        task_id,
        source_domain=str(record.get("domain", "")),
        source_scene_id=scene_id,
    )
    return {
        "scene_variant": str(record.get("scene_variant", "") or ""),
        "scene_id": scene_id,
        "query_id": str(record.get("query_id", "") or resolve_task_query_id()),
    }


def _build_query_field_assignments(
    records: list[Mapping[str, Any]],
    *,
    dataset_root: Path,
) -> dict[str, dict[str, str]]:
    """Recover task/scene variant columns from mandatory sidecar traces."""

    instance_to_ref: dict[str, tuple[str, int]] = {}
    line_refs_by_shard: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    direct_assignments: dict[str, dict[str, str]] = {}
    records_by_instance: dict[str, Mapping[str, Any]] = {}
    for record in records:
        instance_id = str(record.get("instance_id", "")).strip()
        if not instance_id:
            continue
        records_by_instance[instance_id] = record

        trace_ref = _json_mapping(record.get("trace_ref"))
        if not trace_ref:
            direct_assignments[instance_id] = _fields_from_train_record(record)
            continue
        shard_id = str(trace_ref.get("shard_id", "")).strip()
        line_index = trace_ref.get("line_index")
        if not shard_id or line_index is None:
            raise ValueError(f"Trace trace_ref for {instance_id} is missing shard_id/line_index")
        parsed_line_index = int(line_index)
        instance_to_ref[instance_id] = (shard_id, parsed_line_index)
        line_refs_by_shard[shard_id][parsed_line_index].append(instance_id)

    assignments = dict(direct_assignments)
    for shard_id, instances_by_line in sorted(line_refs_by_shard.items()):
        try:
            shard_path = _trace_shard_path(dataset_root, shard_id)
        except FileNotFoundError:
            for instance_ids in instances_by_line.values():
                for instance_id in instance_ids:
                    assignments[instance_id] = _fields_from_train_record(records_by_instance.get(instance_id, {}))
            continue
        shard_target_count = sum(len(instance_ids) for instance_ids in instances_by_line.values())
        shard_found_count = 0
        for line_index, trace_record in enumerate(_iter_trace_shard_records(shard_path)):
            instance_ids = instances_by_line.get(line_index)
            if not instance_ids:
                continue
            query_fields = _query_fields_from_trace_record(trace_record)
            for instance_id in instance_ids:
                fallback_fields = _fields_from_train_record(records_by_instance.get(instance_id, {}))
                assignments[instance_id] = {
                    key: str(query_fields.get(key, "") or fallback_fields.get(key, ""))
                    for key in ("scene_variant", "scene_id", "query_id")
                }
                shard_found_count += 1
            if shard_found_count >= shard_target_count:
                break

    missing = sorted(set(instance_to_ref) - set(assignments))
    if missing:
        sample = ", ".join(missing[:5])
        raise ValueError(f"failed to recover Trace query fields for {len(missing)} rows, sample: {sample}")
    return assignments


def build_rlvr_row(
    train_record: Mapping[str, Any],
    *,
    dataset_root: Path,
    output_parent: Path,
    prompt_variant: PromptVariantInput = "answer_and_annotation",
    image_path_mode: ImagePathMode = "relative",
    image_storage_mode: ImageStorageMode = "path_dict",
    max_embedded_image_pixels: int | None = None,
) -> dict[str, Any]:
    """Convert one Trace train record into an RLVR-ready row."""

    prompt_variant = _normalize_prompt_variant(prompt_variant)

    instance_id = str(train_record.get("instance_id", "")).strip()
    if not instance_id:
        raise ValueError("Trace RLVR export requires instance_id")
    scene_id = str(train_record.get("scene_id", "")).strip()
    if not scene_id:
        raise ValueError(f"Trace RLVR export requires scene_id on {instance_id}")

    answer_gt = train_record.get("answer_gt")
    annotation_gt = train_record.get("annotation_gt")
    reward_contract = train_record.get("reward_contract")
    if not isinstance(answer_gt, Mapping):
        raise ValueError(f"Trace RLVR export requires answer_gt on {instance_id}")
    if not isinstance(annotation_gt, Mapping):
        raise ValueError(f"Trace RLVR export requires annotation_gt on {instance_id}")
    if not isinstance(reward_contract, Mapping):
        raise ValueError(f"Trace RLVR export requires reward_contract on {instance_id}")
    exported_images, image_infos = _build_exported_images(
        train_record,
        dataset_root=dataset_root,
        output_parent=output_parent,
        image_path_mode=image_path_mode,
        image_storage_mode=image_storage_mode,
        max_embedded_image_pixels=max_embedded_image_pixels,
    )
    exported_annotation_gt = _scale_annotation_gt_for_export(annotation_gt, image_infos=image_infos)
    prompt_columns = _build_prompt_columns(train_record, image_count=len(exported_images))
    prompt = {
        "active": prompt_columns["prompt_active"],
        "answer_only": prompt_columns["prompt_answer"],
        "answer_and_annotation": prompt_columns["prompt_answer_and_annotation"],
    }[prompt_variant]
    task_id = str(train_record.get("task", ""))
    taxonomy = resolve_task_taxonomy(
        task_id,
        source_domain=str(train_record.get("domain", "")),
        source_scene_id=scene_id,
    )

    exported = {
        "uid": instance_id,
        "instance_id": instance_id,
        "domain": taxonomy.domain,
        "task": task_id,
        "scene_id": scene_id,
        "query_id": str(train_record.get("query_id", "") or resolve_task_query_id()),
        "scene_variant": str(train_record.get("scene_variant", "") or ""),
        "prompt": prompt,
        **prompt_columns,
        "prompt_mode": prompt_variant,
        "images": exported_images,
        "image_sizes_original": [
            {"width": info.original_width, "height": info.original_height}
            for info in image_infos
        ],
        "image_sizes_exported": [
            {"width": info.exported_width, "height": info.exported_height}
            for info in image_infos
        ],
        "answer_gt": dict(answer_gt),
        "annotation_gt": exported_annotation_gt,
        "reward_contract": dict(reward_contract),
        "trace_ref": dict(train_record.get("trace_ref", {}))
        if isinstance(train_record.get("trace_ref"), Mapping)
        else train_record.get("trace_ref"),
    }
    return exported


def _write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("w", encoding="utf-8") as handle,
        tqdm(
            total=len(rows),
            desc="Write RLVR JSONL",
            unit="row",
            dynamic_ncols=True,
            disable=not _EXPORT_PROGRESS_ENABLED,
        ) as progress_bar,
    ):
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True))
            handle.write("\n")
            progress_bar.update(1)


def _prepare_parquet_row(
    row: dict[str, Any],
    *,
    image_storage_mode: ImageStorageMode,
) -> dict[str, Any]:
    parquet_row = dict(row)
    if image_storage_mode == "embedded_bytes":
        image_entries = []
        for image_entry in parquet_row.get("images") or []:
            if isinstance(image_entry, Mapping) and image_entry.get("bytes") is not None:
                image_entries.append(
                    {
                        "bytes": image_entry.get("bytes"),
                        "path": image_entry.get("path"),
                    }
                )
                continue
            image_entries.append(image_entry)
        parquet_row["images"] = image_entries
    for key in _PARQUET_JSON_COLUMNS:
        if key in parquet_row:
            parquet_row[key] = json.dumps(
                parquet_row[key],
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            )
    return parquet_row


def _write_parquet_items(
    path: Path,
    items: Iterable[Any],
    *,
    row_count: int,
    row_builder: Callable[[Any], dict[str, Any]],
    parquet_cpu_count: int | None = None,
    image_storage_mode: ImageStorageMode = "path_dict",
) -> None:
    """Write parquet from an item stream with optional parallel row preparation."""

    import pyarrow as pa

    path.parent.mkdir(parents=True, exist_ok=True)
    progress_enabled = _EXPORT_PROGRESS_ENABLED
    if image_storage_mode not in {"path_dict", "embedded_bytes"}:
        raise ValueError(f"unsupported image-storage mode: {image_storage_mode}")

    requested_cpu_count = _resolve_parquet_cpu_count(parquet_cpu_count)
    row_worker_count = _resolve_parquet_row_worker_count(parquet_cpu_count)
    prior_cpu_count = pa.cpu_count()
    prior_io_thread_count = pa.io_thread_count()
    try:
        if requested_cpu_count is not None:
            pa.set_cpu_count(int(requested_cpu_count))
            pa.set_io_thread_count(int(requested_cpu_count))

        import pyarrow.parquet as pq

        writer = None
        features = None
        executor: ThreadPoolExecutor | None = None
        try:
            if row_worker_count > 1:
                executor = ThreadPoolExecutor(max_workers=row_worker_count)
                if progress_enabled:
                    tqdm.write(
                        "Write parquet row preparation workers: "
                        f"{row_worker_count}"
                    )
            with tqdm(
                total=int(row_count),
                desc="Write parquet",
                unit="row",
                dynamic_ncols=True,
                disable=not progress_enabled,
            ) as progress_bar:
                for chunk in _iter_iterable_chunks(items, _PARQUET_WRITE_CHUNK_SIZE):
                    if executor is None:
                        parquet_chunk_rows = [row_builder(item) for item in chunk]
                    else:
                        parquet_chunk_rows = list(executor.map(row_builder, chunk))
                    if not parquet_chunk_rows:
                        continue
                    table = pa.Table.from_pylist(parquet_chunk_rows)
                    if image_storage_mode == "embedded_bytes":
                        if features is None:
                            from datasets import Features, Sequence
                            from datasets import Image as HFImage

                            features = Features.from_arrow_schema(table.schema)
                            features["images"] = Sequence(HFImage())
                        table = table.cast(features.arrow_schema)
                    if writer is None:
                        writer = pq.ParquetWriter(path, table.schema, compression="snappy")
                    writer.write_table(table, row_group_size=len(parquet_chunk_rows))
                    progress_bar.update(len(chunk))
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            if writer is not None:
                writer.close()
    finally:
        pa.set_cpu_count(int(prior_cpu_count))
        pa.set_io_thread_count(int(prior_io_thread_count))


def _write_parquet_rows(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    parquet_cpu_count: int | None = None,
    image_storage_mode: ImageStorageMode = "path_dict",
) -> None:
    if image_storage_mode not in {"path_dict", "embedded_bytes"}:
        raise ValueError(f"unsupported image-storage mode: {image_storage_mode}")
    _write_parquet_items(
        path,
        rows,
        row_count=len(rows),
        row_builder=lambda row: _prepare_parquet_row(
            row,
            image_storage_mode=image_storage_mode,
        ),
        parquet_cpu_count=parquet_cpu_count,
        image_storage_mode=image_storage_mode,
    )


def _write_parquet_row_iter(
    path: Path,
    rows: Iterable[dict[str, Any]],
    *,
    row_count: int,
    parquet_cpu_count: int | None = None,
    image_storage_mode: ImageStorageMode = "path_dict",
) -> None:
    """Write parquet from a row iterator without retaining all rows in memory."""
    if image_storage_mode not in {"path_dict", "embedded_bytes"}:
        raise ValueError(f"unsupported image-storage mode: {image_storage_mode}")
    _write_parquet_items(
        path,
        rows,
        row_count=row_count,
        row_builder=lambda row: _prepare_parquet_row(
            row,
            image_storage_mode=image_storage_mode,
        ),
        parquet_cpu_count=parquet_cpu_count,
        image_storage_mode=image_storage_mode,
    )


def export_trace_dataset_to_rlvr(
    source_path: str | Path,
    output_path: str | Path,
    *,
    output_format: OutputFormat | None = None,
    prompt_variant: PromptVariantInput = "answer_and_annotation",
    image_path_mode: ImagePathMode = "relative",
    image_storage_mode: ImageStorageMode = "path_dict",
    parquet_cpu_count: int | None = None,
    max_embedded_image_pixels: int | None = None,
) -> RLVRExportResult:
    """Export one Trace dataset to an RLVR-ready JSONL or parquet file."""

    prompt_variant = _normalize_prompt_variant(prompt_variant)
    if image_path_mode not in {"relative", "absolute", "dataset_relative"}:
        raise ValueError(f"unsupported image-path mode: {image_path_mode}")
    if image_storage_mode not in {"path_dict", "embedded_bytes"}:
        raise ValueError(f"unsupported image-storage mode: {image_storage_mode}")
    if max_embedded_image_pixels is not None:
        if int(max_embedded_image_pixels) <= 0:
            raise ValueError("max_embedded_image_pixels must be > 0 when provided")
        if image_storage_mode != "embedded_bytes":
            raise ValueError("max_embedded_image_pixels requires image_storage_mode='embedded_bytes'")

    dataset_root, train_instances_path = resolve_train_instances_source(source_path)
    final_output_path, final_format = resolve_export_output_path(output_path, output_format=output_format)
    if final_format == "jsonl" and image_storage_mode == "embedded_bytes":
        raise ValueError("embedded_bytes image storage is only supported for parquet exports")
    output_parent = final_output_path.parent.resolve()

    records = _read_jsonl_records(train_instances_path)
    if _EXPORT_PROGRESS_ENABLED:
        tqdm.write(f"Assign task-local buckets for {len(records)} rows")
    curriculum_assignments = _build_curriculum_assignments(records)
    if _EXPORT_PROGRESS_ENABLED:
        tqdm.write(f"Recover query fields for {len(records)} rows")
    query_assignments = _build_query_field_assignments(records, dataset_root=dataset_root)

    def iter_export_rows() -> Iterable[dict[str, Any]]:
        for record in records:
            instance_id = str(record.get("instance_id", "")).strip()
            yield {
                **build_rlvr_row(
                    record,
                    dataset_root=dataset_root,
                    output_parent=output_parent,
                    prompt_variant=prompt_variant,
                    image_path_mode=image_path_mode,
                    image_storage_mode=image_storage_mode,
                    max_embedded_image_pixels=max_embedded_image_pixels,
                ),
                **query_assignments[instance_id],
                **curriculum_assignments[instance_id],
            }

    if final_format == "parquet":
        def build_parquet_row(record: dict[str, Any]) -> dict[str, Any]:
            instance_id = str(record.get("instance_id", "")).strip()
            return _prepare_parquet_row(
                {
                    **build_rlvr_row(
                        record,
                        dataset_root=dataset_root,
                        output_parent=output_parent,
                        prompt_variant=prompt_variant,
                        image_path_mode=image_path_mode,
                        image_storage_mode=image_storage_mode,
                        max_embedded_image_pixels=max_embedded_image_pixels,
                    ),
                    **query_assignments[instance_id],
                    **curriculum_assignments[instance_id],
                },
                image_storage_mode=image_storage_mode,
            )

        _write_parquet_items(
            final_output_path,
            records,
            row_count=len(records),
            row_builder=build_parquet_row,
            parquet_cpu_count=parquet_cpu_count,
            image_storage_mode=image_storage_mode,
        )
    else:
        rows = []
        with tqdm(
            total=len(records),
            desc="Build RLVR rows",
            unit="row",
            dynamic_ncols=True,
            disable=not _EXPORT_PROGRESS_ENABLED,
        ) as progress_bar:
            for row in iter_export_rows():
                rows.append(row)
                progress_bar.update(1)
        _write_jsonl_rows(final_output_path, rows)

    return RLVRExportResult(
        source_dataset_root=dataset_root,
        train_instances_path=train_instances_path,
        output_path=final_output_path.resolve(),
        output_format=final_format,
        prompt_variant=prompt_variant,
        image_path_mode=image_path_mode,
        row_count=len(records),
    )
