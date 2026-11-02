"""Archive and local-run ingestion into the study's normalized aligned table (handoff 4.1-4.2, DESIGN.md section 4).

Why this module exists: the archive and every new local run expose the protocol's *public* row schema (responses,
extractions, scores) whose ``sample_id`` deliberately differs across models and seeds. Cross-model comparison needs a
content identity that is the same for every model, so each row is keyed by ``(benchmark_id, source_ordinal,
source_index, source_row_sha256, source_record_sha256)`` and verified against the prepared dataset manifest (plus
``media_set_sha256`` and the ordered per-image digests) before it enters the aligned table. Every digest the exporter
wrote (media set, source record, response, sample/response/score ids, public request, prompt contract, extraction) is
recomputed with the exporter's own helpers, so a row the protocol would not have produced cannot be ingested.

Fail-closed rules: duplicate identities, ambiguous or conflicting joins, digest mismatches, key sets that differ across
the slices of a benchmark, or rows that do not match exactly one dataset-manifest row stop ingestion with an error that
names the slice. Missing rows are listed in ``audit/missingness.csv``; they are never turned into wrong answers.

Local runs (new generations, or the existing seed-42 reruns of B and T) are read from the protocol's private archive
spool (``<run_dir>/hf_archive/{ready,payloads}``). The spool is converted to public rows by the *exporter's own* row
builders (``_response_rows``, ``_extraction_rows``, ``_score_rows``) run on a private copy of the exporter whose parquet
record reader is replaced by a spool reader; after that, local rows go through exactly the same validation as archive
rows.

A local run is accepted only if, in addition (handoff 4.2, 4.4; DESIGN.md section 3):

* its model label is a declared study label and names its protocol (``B``/``T``/``V@P0`` native, ``*@P1`` backbone
  view, ``*@P2`` VERO view); the spool's served model is the prepared native model at the pinned commit (P0 reruns)
  or the registered study view of that label (P1/P2: view manifest, ``manifests/views.json`` and the protocol's
  content-bound revision marker all agree, and the weights are the label's weight source);
* every descriptor names the pinned protocol commit and VLMEvalKit commit, with one code hash and one campaign config;
* its normalized generation settings (minus only ``model_id`` and ``seed``) equal the configured protocol and the
  archive's, so no run with another budget or temperature is ever compared with the cached parents;
* it covers exactly the planned benchmarks and rows (``plan.seed<S>.json``; a subset's row list must recompute its
  ``row_list_sha256`` and, for study runs, equal the frozen split's), and each aggregate equals the campaign summary;
* its rendered inputs conform to the archived reference of its protocol (P1: archived ``B``; P2: archived ``V@P0``;
  P0 rerun: the same archived label): identical ``prompt_sha256``, model-neutral request digest and prompt tokens.

Two verified, documented exceptions keep identities joinable without weakening the checks:

* CountQA (``PRODUCER_ROW_HASH_ALIASES``): the archive's producer hashed pandas-inferred integer answers, the public
  code hashes the answer string, so every archived CountQA ``source_row_sha256`` differs from the prepared manifest
  while ordinal, index and media agree. The alias is rebuilt from the licensed TSV and accepted only if it reproduces
  the manifest hash of every row; matched rows take the manifest identity and keep the archived digests in
  ``archived_source_row_sha256`` / ``archived_source_record_sha256`` with an explicit ``row_identity_status``. Because
  DESIGN.md section 4 requires a direct manifest match, the alias is **off by default**: it is enabled only by the
  config (``identity.producer_row_hash_aliases``) or an explicit argument, applies to archive rows only (local runs
  are produced by the public code and must match directly), and the ingest report lists it under
  ``pending_design_amendments`` until DESIGN.md records an amendment naming its patch id.
* Row-subset local runs (``repair_subset_stage_ordinals``): the official-VLMEvalKit route renumbers extraction/score
  ordinals by subset position; records are re-attached to their generation record only by (index, row hash) plus an
  exact positional check.

Layout written under the work root (local only, never committed):

* ``aligned/predictions.parquet``: archive rows (protocol ``P0_native``), one row group per model x seed x benchmark
  slice, written benchmark-major; read it column- and slice-selectively with :func:`load_aligned`.
* ``aligned/official_aggregates.parquet``: archived official aggregates (one row per slice).
* ``aligned/local_runs/<name>.parquet`` (+ ``.aggregates.parquet``, ``.report.json``): one file per ingested local run,
  e.g. ``B_seed42_existing_local_rerun``. Local reruns keep the archive label (``B``/``T``) and differ by ``source``
  and ``run_id``, so a sensitivity analysis can substitute them for the archived slice without relabelling;
  :func:`load_aligned` refuses to return two runs of one label and seed unless ``source`` and ``run_id`` are selected.
* ``aligned/index.json``: digests, row counts, slices and this layout description.
* ``audit/missingness.csv``, ``audit/archive_ingest.json``.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import sys
import types
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import PROJECT_ROOT, REPO_ROOT, ROLES, StudyConfig, canonical_json, sha256_bytes, sha256_file, write_json

Log = Callable[[str], None]


class IngestError(RuntimeError):
    """Archive or local-run content violates an identity, integrity, coverage or join rule."""


# --------------------------------------------------------------------------------------------- protocol code access
PROTOCOL_SCRIPTS_REL = "rlvr/evaluation/scripts"
EXPORTER = "trace_eval_public_export"
ARCHIVE_LIB = "trace_eval_hf_archive_lib"
MEDIA_CONTRACT = "trace_eval_media_contract"
_MODULES: dict[str, types.ModuleType] = {}


def protocol_module_path(name: str) -> Path:
    """The pinned protocol module in this checkout (``rlvr/`` at af32fe0; read-only, never edited)."""

    path = REPO_ROOT / PROTOCOL_SCRIPTS_REL / f"{name}.py"
    if not path.is_file():
        raise IngestError(f"pinned protocol module {name!r} not found under {PROTOCOL_SCRIPTS_REL}")
    return path


def _load_module_copy(name: str, alias: str) -> types.ModuleType:
    path = protocol_module_path(name)
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise IngestError(f"cannot import protocol module {name!r}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module  # dataclasses resolve their module through sys.modules
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(alias, None)
        raise
    return module


def protocol_module(name: str) -> types.ModuleType:
    """Import a pinned protocol module under a private alias, so nothing here can shadow or patch the shared one."""

    alias = f"_trace_merge_protocol__{name}"
    if alias not in _MODULES:
        _MODULES[alias] = _load_module_copy(name, alias)
    return _MODULES[alias]


def protocol_code_provenance(cfg: StudyConfig | None = None) -> dict[str, Any]:
    """sha256 of every protocol module this package imports; with a prepared clone, require byte equality.

    The local runs were produced by the clone's copy; identity helpers that differed from this checkout's frozen copy
    would make recomputed digests meaningless, so a difference is an error rather than a note.
    """

    out: dict[str, Any] = {}
    for name in (EXPORTER, ARCHIVE_LIB, MEDIA_CONTRACT):
        path = protocol_module_path(name)
        entry: dict[str, Any] = {"path": f"{PROTOCOL_SCRIPTS_REL}/{name}.py", "sha256": sha256_file(path)}
        if cfg is not None and cfg.protocol_clone is not None:
            clone_copy = cfg.protocol_clone / PROTOCOL_SCRIPTS_REL / f"{name}.py"
            if clone_copy.is_file():
                entry["clone_sha256"] = sha256_file(clone_copy)
                if entry["clone_sha256"] != entry["sha256"]:
                    raise IngestError(f"protocol clone's {name}.py differs from this checkout's pinned copy")
        out[name] = entry
    return out


def _make_spool_reader(module: types.ModuleType) -> Callable[[Any], list[dict[str, Any]]]:
    def _source_records(source: Any) -> list[dict[str, Any]]:
        records = [dict(item) for item in source.records]
        expected = int(source.manifest.get("rows", -1))
        if len(records) != expected:
            raise module.PublicExportIntegrityError(f"spool row count mismatch: {len(records)} != {expected}")
        for ordinal, record in enumerate(records):
            if not isinstance(record.get("record_id"), str) or not record["record_id"]:
                raise module.PublicExportError(f"spool record {ordinal} has no record_id")
        # identical ordering to the exporter's parquet reader
        records.sort(key=lambda item: (int(item.get("source_ordinal", 0)), str(item.get("source_index", "")),
                                       str(item["record_id"])))
        return records

    return _source_records


def spool_exporter() -> types.ModuleType:
    """A private exporter copy whose private-parquet record reader is replaced by a spool reader.

    ``ArchiveDaemon.build_descriptor`` copies a spool descriptor's ``aggregate``/``identity``/``rows`` into the private
    slice manifest and each JSONL record into ``record_json``, so the exporter's row builders see the same inputs; only
    the reader differs. A separate module object keeps the shared exporter unmodified.
    """

    alias = "_trace_merge_protocol__exporter_spool"
    if alias not in _MODULES:
        module = _load_module_copy(EXPORTER, alias)
        module._source_records = _make_spool_reader(module)
        _MODULES[alias] = module
    return _MODULES[alias]


# --------------------------------------------------------------------------------------------- schema and labels
IDENTITY_FIELDS = ("benchmark_id", "source_ordinal", "source_index", "source_row_sha256", "source_record_sha256")
LINK_FIELDS = ("sample_id", "dataset_sha256", "media_set_sha256", "prompt_sha256", "request_sha256",
               "prompt_contract_sha256")
CONFIG_NAMES = ("responses", "extractions", "scores")
SPOOL_STAGES = {"responses": "generation", "extractions": "extraction", "scores": "score"}
MODEL_ROLES = ("B", "T", "V", "M050", "CV050", "CT050")
PROTOCOL_IDS = ("P0_native", "P1_backbone", "P2_vero")
SOURCES = ("archive", "local_run", "existing_local_rerun")
LOCAL_SOURCES = ("local_run", "existing_local_rerun")
ARCHIVE_LABELS = {"B": "B", "T": "T", "V": "V@P0"}
EXTRACTION_STATUSES = ("resolved", "ambiguous", "invalid", "excluded")
ROW_IDENTITY_VERIFIED = "verified_dataset_manifest_and_digests"
ROW_IDENTITY_ALIASED = "verified_dataset_manifest_via_producer_row_hash_alias"
DEFAULT_JUDGE_SOURCE_SLUG = "qwen3-32b-judge"   # served slug of the pinned judge in the prepared clone
MAX_PROBLEMS = 12
SUMMARY_TOLERANCE_PP = 1e-6
MODEL_MARKER = ".trace_model_revision.json"
# served slugs of the prepared protocol clone's native models (rlvr/evaluation/.work/models/<slug>); V is not prepared
NATIVE_PROTOCOL_SLUGS = {"B": "qwen2.5-vl-7b-instruct", "T": "trace-qwen2.5-vl-7b"}
LABEL_SUFFIX_PROTOCOLS = {"P0": "P0_native", "P1": "P1_backbone", "P2": "P2_vero"}
# archived label whose rendered inputs a local run of a protocol must reproduce (P0 reruns: their own archived label)
CONFORMANCE_REFERENCE = {"P1_backbone": "B", "P2_vero": "V@P0"}
IDENTITY_SETTING_FIELDS = ("model_id", "seed")   # documented identity fields of normalize_generation_settings
# normalized generation setting -> configured protocol.generation key
GENERATION_SETTING_KEYS = {"temperature": "temperature", "top_p": "top_p", "top_k": "top_k",
                           "presence_penalty": "presence_penalty", "repetition_penalty": "repetition_penalty",
                           "max_output_tokens": "max_tokens", "min_image_pixels": "min_image_pixels",
                           "max_image_pixels": "max_image_pixels", "media_transport": "media_transport"}
PROVENANCE_FIELDS = ("trace_git_commit", "vlmevalkit_git_commit", "final25_code_hash", "campaign_config_hash",
                     "archive_hook_version")
# DESIGN.md section 5, fixed before any row was read: the archive holds 0/1 row scores whose mean is the aggregate
# (binary_row), 0/1 row scores under a group statistic (grouped_row, WeMath), or only the aggregate. Ingestion checks
# the data against this map instead of inferring a kind from each slice's observed values.
BENCHMARK_SCORE_KINDS: dict[str, str] = {
    **{b: "binary_row" for b in ("charxivreason", "evochart", "mathverse", "phyx_mini_mc", "mmmu_pro_vision",
                                 "realworldqa", "mmstar", "embspatial", "spatialvizbench_cot", "cvbench_3d", "erqa",
                                 "blink", "countqa", "treebench", "logicvista", "mme_reasoning")},
    "wemath": "grouped_row",
    **{b: "aggregate_only" for b in ("chartqapro", "tablevqabench", "mathvision", "mathvista", "countbenchqa",
                                     "puzzlevqa", "visualpuzzles")},
}
ROW_SCORE_KINDS = ("binary_row", "grouped_row")

NORMALIZED_SCHEMA = pa.schema(
    [
        pa.field("study_id", pa.string(), nullable=False),
        pa.field("protocol_id", pa.string(), nullable=False),
        pa.field("model_role", pa.string(), nullable=False),
        pa.field("model_label", pa.string(), nullable=False),
        pa.field("model_revision", pa.string(), nullable=False),
        pa.field("runtime_revision", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("seed", pa.int64(), nullable=False),
        pa.field("benchmark_id", pa.string(), nullable=False),
        pa.field("source_ordinal", pa.int64(), nullable=False),
        pa.field("source_index", pa.string(), nullable=False),
        pa.field("source_row_sha256", pa.string(), nullable=False),
        pa.field("source_record_sha256", pa.string(), nullable=False),
        pa.field("media_set_sha256", pa.string(), nullable=False),
        pa.field("content_key", pa.string(), nullable=False),
        pa.field("official_metric_id", pa.string(), nullable=False),
        pa.field("score_kind", pa.string(), nullable=False),
        pa.field("archived_row_score", pa.float64()),
        pa.field("score_available", pa.bool_(), nullable=False),
        pa.field("official_excluded", pa.bool_(), nullable=False),
        pa.field("scorer_id", pa.string(), nullable=False),
        pa.field("response_id", pa.string(), nullable=False),
        pa.field("response_text", pa.string(), nullable=False),
        pa.field("finish_reason", pa.string()),
        pa.field("completion_tokens", pa.int64()),
        pa.field("prompt_tokens", pa.int64()),
        pa.field("token_count_source", pa.string()),
        pa.field("extraction_status", pa.string()),
        pa.field("extraction_value_type", pa.string()),
        pa.field("extraction_value_json", pa.string()),
        pa.field("extraction_method", pa.string()),
        pa.field("used_judge", pa.bool_()),
        pa.field("judge_output", pa.string()),
        pa.field("source_manifest_sha256", pa.string(), nullable=False),
        pa.field("row_identity_status", pa.string(), nullable=False),
        # provenance and diagnostics beyond the shared interface
        pa.field("valid_protocol_ids", pa.string(), nullable=False),
        pa.field("model_id", pa.string(), nullable=False),
        pa.field("dataset_split", pa.string(), nullable=False),
        pa.field("dataset_revision", pa.string(), nullable=False),
        pa.field("dataset_sha256", pa.string(), nullable=False),
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("response_sha256", pa.string(), nullable=False),
        pa.field("ordered_media_sha256", pa.list_(pa.string()), nullable=False),
        pa.field("prompt_sha256", pa.string(), nullable=False),
        pa.field("request_sha256", pa.string(), nullable=False),
        pa.field("prompt_contract_sha256", pa.string(), nullable=False),
        pa.field("generation_settings_json", pa.string(), nullable=False),
        pa.field("extraction_sha256", pa.string()),
        pa.field("extraction_candidates_json", pa.string()),
        pa.field("retry_count", pa.int64()),
        pa.field("judge_model_id", pa.string()),
        pa.field("judge_model_revision", pa.string()),
        pa.field("score_unit", pa.string()),
        pa.field("score_value_json", pa.string()),
        pa.field("row_scorer", pa.string()),
        pa.field("row_scorer_output_json", pa.string()),
        pa.field("source_parts_json", pa.string(), nullable=False),
        pa.field("archived_source_row_sha256", pa.string()),
        pa.field("archived_source_record_sha256", pa.string()),
    ],
    metadata={b"trace_merge_schema": b"aligned_predictions_v1"},
)
NORMALIZED_COLUMNS = tuple(NORMALIZED_SCHEMA.names)

AGGREGATE_SCHEMA = pa.schema(
    [
        pa.field("protocol_id", pa.string(), nullable=False),
        pa.field("model_label", pa.string(), nullable=False),
        pa.field("model_role", pa.string(), nullable=False),
        pa.field("seed", pa.int64(), nullable=False),
        pa.field("benchmark_id", pa.string(), nullable=False),
        pa.field("official_value", pa.float64(), nullable=False),
        pa.field("evaluated_rows", pa.int64(), nullable=False),
        pa.field("scorer_id", pa.string(), nullable=False),
        pa.field("metric_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("model_id", pa.string(), nullable=False),
        pa.field("score_kind", pa.string(), nullable=False),
        pa.field("results_file_value", pa.float64()),
        # local runs: the run's own campaign summary value and how the spool aggregate relates to it
        pa.field("campaign_summary_value", pa.float64()),
        pa.field("campaign_summary_status", pa.string()),
    ]
)

LAYOUT_DOC = (
    "aligned/predictions.parquet holds the archive rows (P0_native) in one file, one row group per model x seed x "
    "benchmark slice (benchmark-major). aligned/local_runs/<name>.parquet holds one ingested local run each, with the "
    "same schema. Unique key: (source, run_id, model_label, seed, content_key); local reruns of B/T keep the archive "
    "label, so load_aligned refuses to mix two runs of one label and seed unless source and run_id are selected. Read "
    "with trace_merge.ingest.load_aligned(cfg, columns=..., labels=..., seeds=..., benchmarks=..., local_runs=...)."
)


def content_key(benchmark_id: str, source_ordinal: int, source_index: str, source_row_sha256: str,
                source_record_sha256: str) -> str:
    """Model- and seed-independent row identity (the shared interface's definition)."""

    material = [str(benchmark_id), int(source_ordinal), str(source_index), str(source_row_sha256),
                str(source_record_sha256)]
    return sha256_bytes(canonical_json(material).encode("utf-8"))


def role_of_label(label: str) -> str:
    return str(label).split("@", 1)[0]


def valid_protocols_for(role: str, protocol_id: str) -> str:
    """B and T native inputs are byte-identical to the P1 (backbone view) inputs (DESIGN.md section 3)."""

    if protocol_id == "P0_native" and role in ("B", "T"):
        return "P0_native,P1_backbone"
    return protocol_id


def protocol_for_label(label: str) -> str:
    """The protocol a model label belongs to (DESIGN.md section 3), so a label can never be filed under another one.

    Bare ``B``/``T`` are the native runs (also valid P1 outputs); every other label carries its protocol as a suffix:
    ``V@P0`` (native VERO), ``<role>@P1`` (backbone view), ``<role>@P2`` (VERO view). Merges have no native view.
    """

    role, sep, suffix = str(label).partition("@")
    if role not in MODEL_ROLES:
        raise IngestError(f"model label {label!r} does not name a study role {MODEL_ROLES}")
    if not sep:
        if role in ("B", "T"):
            return "P0_native"
        raise IngestError(f"model label {label!r} needs a protocol suffix (@P0, @P1 or @P2)")
    if suffix not in LABEL_SUFFIX_PROTOCOLS:
        raise IngestError(f"model label {label!r} has an unknown protocol suffix {suffix!r}")
    if suffix == "P0" and role != "V":
        raise IngestError(f"model label {label!r}: native labels are 'B', 'T' and 'V@P0' (merges have no native view)")
    return LABEL_SUFFIX_PROTOCOLS[suffix]


def study_labels(cfg: StudyConfig) -> set[str]:
    """Every label the study declares: the archive labels plus the configured evaluation matrix."""

    out = set(ARCHIVE_LABELS.values())
    for tier in (cfg.raw.get("evaluation_matrix") or {}).values():
        out.update(str(m) for m in tier.get("models") or [])
    return out


def content_revision(file_sha256: Mapping[str, str]) -> str:
    """The protocol's ``sha256set:`` content revision of a served directory (prepare_trace_eval_models)."""

    exp = protocol_module(EXPORTER)  # same canonical JSON (ensure_ascii, sorted keys, compact separators)
    return "sha256set:" + sha256_bytes(exp.canonical_json(dict(file_sha256)).encode("utf-8"))


def normalized_settings_json(settings_json: str) -> str:
    """Generation settings with only the documented identity fields (model_id, seed) removed, canonical JSON."""

    settings = json.loads(settings_json)
    for key in IDENTITY_SETTING_FIELDS:
        settings.pop(key, None)
    return canonical_json(settings)


def neutral_request_sha256(prompt_sha256: str, ordered_media: Sequence[str], settings_json: str) -> str:
    """The public request digest recomputed without model_id/seed: equal iff prompt, media and decoding are equal."""

    exp = protocol_module(EXPORTER)
    return exp.public_request_sha256_from_hashes(prompt_digest=prompt_sha256, ordered_media_sha256=list(ordered_media),
                                                 generation_settings=json.loads(normalized_settings_json(settings_json)))


def configured_settings_problems(cfg: StudyConfig, normalized: str) -> list[str]:
    """Differences between normalized generation settings and the configured protocol decoding."""

    settings = json.loads(normalized)
    gen = cfg.raw["protocol"]["generation"]
    return [f"{k}={settings.get(k)!r} (configured {gen[c]!r})" for k, c in GENERATION_SETTING_KEYS.items()
            if settings.get(k) != gen[c]]


# --------------------------------------------------------------------------------------------- value coercion
_NULL_STRINGS = {"", "none", "null", "nan", "<na>"}


def _is_null(value: Any, *, null_strings: bool = False) -> bool:
    """Missing value test. Text such as a judge output of literally ``"None"`` is content, so string spellings of
    null count only for typed fields (ints, bools, floats) that some exports store as strings."""

    if value is None or value is pd.NA:
        return True
    if isinstance(value, (float, np.floating)) and math.isnan(value):
        return True
    return null_strings and isinstance(value, str) and value.strip().lower() in _NULL_STRINGS


def _as_int(value: Any) -> int | None:
    """Integers may arrive as int, float (nullable parquet ints) or strings ('42'); bools are rejected."""

    if _is_null(value, null_strings=True):
        return None
    if isinstance(value, (bool, np.bool_)):
        raise IngestError(f"boolean where an integer was expected: {value!r}")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not float(value).is_integer():
            raise IngestError(f"non-integral value where an integer was expected: {value!r}")
        return int(value)
    text = str(value).strip()
    if re.fullmatch(r"-?\d+(\.0+)?", text):
        return int(float(text))
    raise IngestError(f"not an integer: {value!r}")


def _as_bool(value: Any) -> bool | None:
    if _is_null(value, null_strings=True):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("true", "1"):
        return True
    if text in ("false", "0"):
        return False
    raise IngestError(f"not a boolean: {value!r}")


def _as_str(value: Any) -> str | None:
    return None if _is_null(value) else str(value)


def _as_float(value: Any) -> float | None:
    if _is_null(value, null_strings=True):
        return None
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    out = float(value)
    return None if math.isnan(out) else out


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise IngestError(f"expected a JSON list of digests, got {value[:80]!r}")
        return [str(v) for v in parsed]
    return [str(v) for v in list(value)]


def read_parquet_frame(path: Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Open one parquet file directly (the hive dataset API fails on the archive's seed partitions)."""

    table = pq.ParquetFile(path).read(columns=list(columns) if columns else None)
    return table.to_pandas(types_mapper={pa.int64(): pd.Int64Dtype()}.get)


# --------------------------------------------------------------------------------------------- dataset manifest
@dataclass(frozen=True)
class DatasetManifestIndex:
    """Per-benchmark index of the prepared dataset manifest: identity 5-tuple -> ordered per-image sha256."""

    sha256: str
    schema_version: str
    snapshot_sha256: str
    rows: dict[str, dict[tuple[int, str, str, str, str], tuple[str, ...]]]
    receipts: dict[str, dict[str, Any]]
    # benchmark -> producer identity 5-tuple -> canonical (manifest) 5-tuple; see PRODUCER_ROW_HASH_ALIASES
    aliases: dict[str, dict[tuple[int, str, str, str, str], tuple[int, str, str, str, str]]] = field(
        default_factory=dict)
    alias_reports: dict[str, dict[str, Any]] = field(default_factory=dict)

    def expected_content_keys(self, benchmark_id: str) -> set[str]:
        return {content_key(benchmark_id, k[0], k[1], k[2], k[3]) for k in self.rows[benchmark_id]}


# Benchmarks whose *producer* (archive) source_row_sha256 was computed over a different in-memory representation of
# the same licensed row than the public protocol code computes. CountQA: the producer reloaded its cached TSV and
# hashed pandas-inferred integer answers; the public adapter (post-run patch countqa-cached-answer-normalization)
# hashes the answer string. Ordinal, index and media are identical. The alias is accepted only if, for every manifest
# row, the public representation reproduces the manifest's row hash and the producer representation (same row, the
# patched column left as pandas inferred it) yields a distinct, unique producer hash.
PRODUCER_ROW_HASH_ALIASES: dict[str, dict[str, str]] = {
    "countqa": {"patch_id": "countqa-cached-answer-normalization", "column": "answer"},
}
_ROW_MEDIA_COLUMNS = ("image", "images", "image_path", "image_paths", "img", "picture")


def producer_row_hash_alias(bench: str, row_media: Sequence[Mapping[str, Any]], tsv_path: Path,
                            ) -> tuple[dict[tuple[int, str, str, str, str], tuple[int, str, str, str, str]],
                                       dict[str, Any]]:
    """Verified map from producer identity to manifest identity for one PRODUCER_ROW_HASH_ALIASES benchmark."""

    spec = PRODUCER_ROW_HASH_ALIASES[bench]
    column = spec["column"]
    mc = protocol_module(MEDIA_CONTRACT)
    frame = pd.read_csv(tsv_path, sep="\t")          # the producer's cached reload (pandas type inference)
    records = frame.to_dict("records")
    if len(records) != len(row_media):
        raise IngestError(f"{bench}: licensed TSV has {len(records)} rows, manifest {len(row_media)}")
    alias: dict[tuple[int, str, str, str, str], tuple[int, str, str, str, str]] = {}
    changed = 0
    for item in row_media:
        canon = (int(item["ordinal"]), str(item["index"]), str(item["source_row_hash"]),
                 str(item["source_record_sha256"]), str(item["media_set_sha256"]))
        # manifest ordinals enumerate the loaded table in TSV order; indices need not be unique (CountQA "6710"),
        # so rows are paired by position and the index must agree
        raw = records[canon[0]]
        row = {k: v for k, v in raw.items() if k not in _ROW_MEDIA_COLUMNS}
        # the loaded dataset carries integer indices (Python int() of the TSV's "<image>_<question>" label)
        row["index"] = int(str(raw["index"]))
        if str(row["index"]) != canon[1]:
            raise IngestError(f"{bench}: TSV row {canon[0]} index {raw['index']!r} != manifest index {canon[1]}")
        public = dict(row)
        value = public[column]
        public[column] = "" if pd.isna(value) else str(value).strip()
        if mc.source_row_sha256(public) != canon[2]:
            raise IngestError(f"{bench}: cannot reproduce the manifest row hash at ordinal {canon[0]}; "
                              f"the producer alias is not verifiable")
        producer_hash = mc.source_row_sha256(row)
        producer = (canon[0], canon[1], producer_hash, mc.source_record_sha256(producer_hash, canon[4]), canon[4])
        if producer in alias:
            raise IngestError(f"{bench}: producer identity collision at ordinal {canon[0]}")
        changed += producer_hash != canon[2]
        alias[producer] = canon
    report = {"patch_id": spec["patch_id"], "column": column, "rows": len(alias), "rows_with_distinct_hash": changed,
              "tsv": tsv_path.name, "tsv_sha256": sha256_file(tsv_path),
              "rule": "public hash = column as stripped string (reproduces the manifest); producer hash = column as "
                      "pandas inferred it from the cached TSV"}
    return alias, report


def enabled_producer_aliases(cfg: StudyConfig, override: Sequence[str] | None = None) -> tuple[list[str], str]:
    """Producer row-hash aliases enabled for archive ingestion, and by what (off unless explicitly enabled).

    DESIGN.md section 4 requires every archived row to match the manifest directly, so a verified alias is still a
    deviation: it must be switched on by the config (``identity.producer_row_hash_aliases``) or an explicit argument.
    """

    if override is not None:
        names, source = [str(n) for n in override], "argument"
    else:
        names = [str(n) for n in ((cfg.raw.get("identity") or {}).get("producer_row_hash_aliases") or [])]
        source = "config identity.producer_row_hash_aliases" if names else "none"
    unknown = sorted(set(names) - set(PRODUCER_ROW_HASH_ALIASES))
    if unknown:
        raise IngestError(f"no verified producer row-hash alias rule exists for {unknown}")
    return sorted(set(names)), source


def design_amendment_recorded(patch_id: str, design: Path | None = None) -> bool:
    """True if DESIGN.md's Amendments section names ``patch_id`` (the frozen part may not be edited)."""

    path = design or (PROJECT_ROOT / "DESIGN.md")
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    head, sep, amendments = text.partition("\n## Amendments")
    return bool(sep) and patch_id in amendments


def load_dataset_manifest_index(path: Path, benchmarks: Sequence[str], *,
                                aliases: Sequence[str] = ()) -> DatasetManifestIndex:
    """Load and self-check the prepared dataset manifest (media-set, source-record and snapshot digests).

    ``aliases`` names the producer row-hash aliases to build (off by default; see :func:`enabled_producer_aliases`).
    """

    if not path.is_file():
        raise IngestError(f"dataset manifest not found: {path}")
    raw_bytes = path.read_bytes()
    data = json.loads(raw_bytes)
    mc = protocol_module(MEDIA_CONTRACT)
    datasets = data.get("datasets") or {}
    rows: dict[str, dict[tuple[int, str, str, str, str], tuple[str, ...]]] = {}
    receipts: dict[str, dict[str, Any]] = {}
    alias_maps: dict[str, dict[tuple[int, str, str, str, str], tuple[int, str, str, str, str]]] = {}
    alias_reports: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    for bench in benchmarks:
        rec = datasets.get(bench)
        if not isinstance(rec, dict) or rec.get("status") != "ready":
            raise IngestError(f"dataset manifest has no ready receipt for {bench!r}")
        index: dict[tuple[int, str, str, str, str], tuple[str, ...]] = {}
        ordinals: set[int] = set()
        for item in rec.get("row_media") or []:
            media = item.get("media") or []
            mset = str(item["media_set_sha256"])
            if mc.media_set_sha256(media) != mset:
                problems.append(f"{bench} ordinal {item.get('ordinal')}: manifest media_set_sha256 does not recompute")
            if mc.source_record_sha256(item["source_row_hash"], mset) != item["source_record_sha256"]:
                problems.append(f"{bench} ordinal {item.get('ordinal')}: manifest source_record_sha256 mismatch")
            key = (int(item["ordinal"]), str(item["index"]), str(item["source_row_hash"]),
                   str(item["source_record_sha256"]), mset)
            if key in index or key[0] in ordinals:
                problems.append(f"{bench}: duplicate manifest row identity at ordinal {key[0]}")
            ordinals.add(key[0])
            index[key] = tuple(str(m["sha256"]) for m in media)
        if len(index) != int(rec.get("rows", -1)):
            problems.append(f"{bench}: manifest rows {rec.get('rows')} != {len(index)} indexed rows")
        snapshot = mc.dataset_snapshot_sha256(rec.get("row_media") or [])
        if snapshot != rec.get("dataset_snapshot_sha256"):
            problems.append(f"{bench}: per-dataset snapshot digest does not recompute")
        meta_files = [{"name": Path(str(m.get("path", ""))).name, "sha256": m.get("sha256"),
                       "size_bytes": m.get("size_bytes")} for m in rec.get("metadata_files") or []]
        receipts[bench] = {"rows": len(index), "dataset_snapshot_sha256": rec.get("dataset_snapshot_sha256"),
                           "alias": rec.get("alias"), "metadata_files": meta_files}
        rows[bench] = index
        if bench in PRODUCER_ROW_HASH_ALIASES and bench in aliases and not problems:
            files = rec.get("metadata_files") or []
            if len(files) != 1:
                raise IngestError(f"{bench}: expected one licensed metadata file for the producer alias")
            tsv = path.parent / Path(str(files[0]["path"])).name
            if not tsv.is_file() or sha256_file(tsv) != files[0].get("sha256"):
                raise IngestError(f"{bench}: licensed TSV missing or not the manifest's recorded file")
            alias_maps[bench], alias_reports[bench] = producer_row_hash_alias(bench, rec.get("row_media") or [], tsv)
        if len(problems) > MAX_PROBLEMS:
            break
    if problems:
        raise IngestError("dataset manifest self-check failed: " + "; ".join(problems[:MAX_PROBLEMS]))
    return DatasetManifestIndex(sha256=sha256_bytes(raw_bytes), schema_version=str(data.get("schema_version")),
                                snapshot_sha256=str(data.get("dataset_snapshot_sha256")), rows=rows,
                                receipts=receipts, aliases=alias_maps, alias_reports=alias_reports)


# --------------------------------------------------------------------------------------------- slice validation
@dataclass
class PublicSlice:
    """One model x seed x benchmark slice in the protocol's public schema."""

    benchmark_id: str
    model_id: str
    model_revision: str
    run_id: str
    seed: int
    responses: pd.DataFrame
    extractions: pd.DataFrame
    scores: pd.DataFrame
    parts: dict[str, str]
    source_manifest_sha256: str

    @property
    def label(self) -> str:
        return f"{self.model_id}/seed={self.seed}/{self.benchmark_id}"


@dataclass
class SliceResult:
    frame: pd.DataFrame
    aggregate: dict[str, Any]
    keys: frozenset[str]
    stats: dict[str, Any] = field(default_factory=dict)


class _Problems:
    def __init__(self, label: str) -> None:
        self.label = label
        self.items: list[str] = []
        self.total = 0

    def add(self, message: str, *, first: bool = False) -> None:
        self.total += 1
        if first:
            self.items.insert(0, message)
            del self.items[MAX_PROBLEMS:]
        elif len(self.items) < MAX_PROBLEMS:
            self.items.append(message)

    def raise_if_any(self) -> None:
        if self.total:
            more = f" (+{self.total - len(self.items)} more)" if self.total > len(self.items) else ""
            raise IngestError(f"{self.label}: " + "; ".join(self.items) + more)


def _normalize_identity_columns(df: pd.DataFrame, label: str, name: str) -> pd.DataFrame:
    df = df.copy()
    for col in IDENTITY_FIELDS + ("response_id", "response_sha256") + LINK_FIELDS:
        if col not in df.columns:
            raise IngestError(f"{label}: {name} lacks column {col!r}")
    df["source_ordinal"] = [_as_int(v) for v in df["source_ordinal"]]
    df["source_index"] = [str(v) for v in df["source_index"]]
    if df["source_ordinal"].isna().any():
        raise IngestError(f"{label}: {name} has null source_ordinal")
    df["source_ordinal"] = df["source_ordinal"].astype(np.int64)
    return df


def _check_constants(df: pd.DataFrame, expected: Mapping[str, Any], name: str, problems: _Problems) -> None:
    for col, value in expected.items():
        if col not in df.columns:
            problems.add(f"{name} lacks column {col}")
            continue
        observed = {(_as_int(v) if col == "seed" else str(v)) for v in df[col].unique()}
        if observed != {value}:
            problems.add(f"{name} {col} {sorted(map(str, observed))[:3]} != {value!r}")


def _check_unique(df: pd.DataFrame, name: str, problems: _Problems) -> None:
    dup = df.duplicated(list(IDENTITY_FIELDS), keep=False)
    if dup.any():
        first = df.loc[dup, list(IDENTITY_FIELDS)].head(2).to_dict("records")
        problems.add(f"{name}: {int(dup.sum())} rows share a content identity, e.g. {first}")
    for col in ("sample_id", "response_id"):
        if col in df.columns and df[col].duplicated().any():
            problems.add(f"{name}: duplicate {col}")


def _join_one_to_one(left: pd.DataFrame, right: pd.DataFrame, compare: Sequence[str], name: str,
                     problems: _Problems) -> pd.DataFrame:
    """Join on the full identity tuple plus response id/digest (never on sample_id alone) and compare links."""

    on = list(IDENTITY_FIELDS) + ["response_id", "response_sha256"]
    try:
        merged = left.merge(right, on=on, how="outer", suffixes=("", "__r"), indicator=True, validate="one_to_one")
    except pd.errors.MergeError as error:
        problems.add(f"{name}: join is not one-to-one ({error})")
        return left.iloc[0:0]
    unmatched = merged["_merge"] != "both"
    if unmatched.any():
        sides = Counter(merged.loc[unmatched, "_merge"].astype(str))
        problems.add(f"{name}: {int(unmatched.sum())} rows do not join one-to-one with responses ({dict(sides)})")
    both = merged[~unmatched]
    for col in compare:
        if col not in both.columns or f"{col}__r" not in both.columns:
            continue
        lhs = both[col]
        rhs = both[f"{col}__r"]
        if col == "ordered_media_sha256":
            diff = [tuple(_as_str_list(a)) != tuple(_as_str_list(b)) for a, b in zip(lhs, rhs)]
            n_diff = int(sum(diff))
        else:
            n_diff = int((lhs.astype(str) != rhs.astype(str)).sum())
        if n_diff:
            problems.add(f"{name}: {n_diff} joined rows disagree on {col}")
    return both.drop(columns=["_merge"])


def validate_public_slice(sl: PublicSlice, manifest: DatasetManifestIndex, *,
                          allow_aliases: bool = False) -> SliceResult:
    """Recompute every exporter digest, check joins and the dataset manifest, and build normalized content columns.

    ``allow_aliases`` lets rows match the manifest through an enabled producer row-hash alias; only archive rows
    (written by the producer) may, local runs come from the public code and must match directly.
    """

    exp = protocol_module(EXPORTER)
    problems = _Problems(sl.label)
    if sl.benchmark_id not in manifest.rows:
        raise IngestError(f"{sl.label}: benchmark not indexed in the dataset manifest")
    if sl.benchmark_id not in BENCHMARK_SCORE_KINDS:
        raise IngestError(f"{sl.label}: benchmark has no DESIGN.md section 5 score kind")
    score_kind = BENCHMARK_SCORE_KINDS[sl.benchmark_id]
    resp = _normalize_identity_columns(sl.responses, sl.label, "responses")
    ext = _normalize_identity_columns(sl.extractions, sl.label, "extractions")
    if resp.empty:
        raise IngestError(f"{sl.label}: no responses")
    constants = {"run_id": sl.run_id, "model_id": sl.model_id, "model_revision": sl.model_revision, "seed": sl.seed,
                 "benchmark_id": sl.benchmark_id}
    _check_constants(resp, constants, "responses", problems)
    _check_constants(ext, constants, "extractions", problems)
    _check_constants(sl.scores, constants, "scores", problems)
    for col in ("dataset_split", "dataset_revision", "dataset_sha256"):
        values = set(resp[col].astype(str)) | set(ext[col].astype(str)) | set(sl.scores[col].astype(str))
        if len(values) != 1:
            problems.add(f"{col} is not constant across the slice: {sorted(values)[:3]}")
    dataset_sha = str(resp["dataset_sha256"].iloc[0])
    if str(resp["dataset_revision"].iloc[0]) != f"{exp.DATASET_REVISION_PREFIX}:{dataset_sha}":
        problems.add("dataset_revision does not bind dataset_sha256")
    _check_unique(resp, "responses", problems)
    _check_unique(ext, "extractions", problems)
    problems.raise_if_any()

    # responses: recompute every digest with the exporter's helpers and match the dataset manifest
    mrows = manifest.rows[sl.benchmark_id]
    aliases = manifest.aliases.get(sl.benchmark_id, {}) if allow_aliases else {}
    canonical: dict[tuple[int, str, str, str, str], tuple[int, str, str, str, str]] = {}
    unmatched = 0
    settings_seen: set[str] = set()
    for r in resp.itertuples(index=False):
        media = _as_str_list(r.ordered_media_sha256)
        try:
            mset = exp.media_set_sha256(media)
            if mset != r.media_set_sha256:
                problems.add(f"ordinal {r.source_ordinal}: media_set_sha256 does not recompute")
            if exp.source_record_sha256(r.source_row_sha256, mset) != r.source_record_sha256:
                problems.add(f"ordinal {r.source_ordinal}: source_record_sha256 does not recompute")
            if not isinstance(r.response_text, str):
                problems.add(f"ordinal {r.source_ordinal}: response_text is not text")
                continue
            rsha = exp.response_sha256(r.response_text)
            if rsha != r.response_sha256:
                problems.add(f"ordinal {r.source_ordinal}: response_sha256 does not recompute")
            sid = exp._sample_id(run_id=sl.run_id, model_id=sl.model_id, model_revision=sl.model_revision,
                                 seed=sl.seed, benchmark_id=sl.benchmark_id, source_index=r.source_index,
                                 source_ordinal=int(r.source_ordinal), source_row_sha256=r.source_row_sha256)
            if sid != r.sample_id:
                problems.add(f"ordinal {r.source_ordinal}: sample_id does not recompute")
            if exp._response_id(sid, rsha) != r.response_id:
                problems.add(f"ordinal {r.source_ordinal}: response_id does not recompute")
            settings = json.loads(r.generation_settings_json)
            if exp.canonical_json(settings) != r.generation_settings_json:
                problems.add(f"ordinal {r.source_ordinal}: generation_settings_json is not canonical")
            if settings.get("model_id") != sl.model_id or _as_int(settings.get("seed")) != sl.seed:
                problems.add(f"ordinal {r.source_ordinal}: generation settings name another model or seed")
            settings_seen.add(r.generation_settings_json)
            request = exp.public_request_sha256_from_hashes(prompt_digest=r.prompt_sha256, ordered_media_sha256=media,
                                                            generation_settings=settings)
            if request != r.request_sha256:
                problems.add(f"ordinal {r.source_ordinal}: request_sha256 does not recompute")
            contract = exp.prompt_contract_sha256(dataset_sha256=r.dataset_sha256,
                                                  source_row_sha256=r.source_row_sha256,
                                                  source_record_digest=r.source_record_sha256, media_digest=mset,
                                                  prompt_digest=r.prompt_sha256, request_digest=request)
            if contract != r.prompt_contract_sha256:
                problems.add(f"ordinal {r.source_ordinal}: prompt_contract_sha256 does not recompute")
        except (exp.PublicExportError, ValueError, TypeError) as error:
            problems.add(f"ordinal {r.source_ordinal}: {error}")
            continue
        mkey = (int(r.source_ordinal), str(r.source_index), str(r.source_row_sha256), str(r.source_record_sha256),
                str(r.media_set_sha256))
        mmedia = mrows.get(mkey)
        if mmedia is None and mkey in aliases:
            canonical[mkey] = aliases[mkey]
            mmedia = mrows.get(aliases[mkey])
        if mmedia is None:
            unmatched += 1
            problems.add(f"ordinal {r.source_ordinal}: no dataset-manifest row matches the identity and media set")
        elif tuple(media) != mmedia:
            problems.add(f"ordinal {r.source_ordinal}: ordered media digests differ from the dataset manifest")
    if unmatched and sl.benchmark_id in PRODUCER_ROW_HASH_ALIASES and not aliases:
        rule = PRODUCER_ROW_HASH_ALIASES[sl.benchmark_id]["patch_id"]
        problems.add(f"{unmatched} rows do not match the manifest directly; a verified producer row-hash alias "
                     f"({rule}) exists for this benchmark but is not enabled here (archive only: config "
                     f"identity.producer_row_hash_aliases or --producer-alias, with a DESIGN.md amendment)", first=True)
    problems.raise_if_any()

    # extractions: one-to-one with responses, same links, extraction digest recomputes
    ext_cols = list(IDENTITY_FIELDS) + ["response_id", "response_sha256", "ordered_media_sha256", *LINK_FIELDS,
                                        "extraction_status", "extraction_value_type", "extraction_value_json",
                                        "extraction_candidates_json", "extraction_sha256", "extraction_method",
                                        "used_judge", "judge_model_id", "judge_model_revision", "judge_output",
                                        "judge_output_sha256", "retry_count"]
    missing = [c for c in ext_cols if c not in ext.columns]
    if missing:
        raise IngestError(f"{sl.label}: extractions lack {missing}")
    base_cols = list(IDENTITY_FIELDS) + ["response_id", "response_sha256", "ordered_media_sha256", *LINK_FIELDS]
    joined_ext = _join_one_to_one(resp[base_cols], ext[ext_cols], ["ordered_media_sha256", *LINK_FIELDS],
                                  "extractions", problems)
    for r in joined_ext.itertuples(index=False):
        status = str(r.extraction_status)
        if status not in EXTRACTION_STATUSES:
            problems.add(f"ordinal {r.source_ordinal}: unknown extraction status {status!r}")
        try:
            value = json.loads(r.extraction_value_json)
            candidates = json.loads(r.extraction_candidates_json)
            digest = exp.canonical_sha256({"contract_version": exp.EXTRACTION_CONTRACT_VERSION,
                                           "sample_id": r.sample_id, "status": status, "value": value,
                                           "candidates": candidates})
            if digest != r.extraction_sha256:
                problems.add(f"ordinal {r.source_ordinal}: extraction_sha256 does not recompute")
            if exp._json_value_type(value) != r.extraction_value_type:
                problems.add(f"ordinal {r.source_ordinal}: extraction_value_type mismatch")
        except (exp.PublicExportError, ValueError, TypeError) as error:
            problems.add(f"ordinal {r.source_ordinal}: extraction value unreadable ({error})")
        used = _as_bool(r.used_judge)
        output = _as_str(r.judge_output)
        if used is None or used != (output is not None):
            problems.add(f"ordinal {r.source_ordinal}: used_judge disagrees with judge_output presence")
        if output is not None and exp.sha256_bytes(output.encode("utf-8")) != _as_str(r.judge_output_sha256):
            problems.add(f"ordinal {r.source_ordinal}: judge_output_sha256 does not recompute")
    problems.raise_if_any()

    # scores: one aggregate row, and either one row score per response or none (aggregate-only benchmark)
    scores = sl.scores.copy()
    scopes = set(scores["score_scope"].astype(str))
    if scopes - {"row", "aggregate"}:
        problems.add(f"unknown score scopes {sorted(scopes - {'row', 'aggregate'})}")
    agg = scores[scores["score_scope"] == "aggregate"]
    row_scores = scores[scores["score_scope"] == "row"]
    expected_scorer = f"{sl.benchmark_id}-primary-v1"
    identity = {"run_id": sl.run_id, "model_id": sl.model_id, "seed": sl.seed, "benchmark_id": sl.benchmark_id}
    aggregate: dict[str, Any] = {}
    if len(agg) != 1:
        problems.add(f"{len(agg)} aggregate score rows (expected exactly 1)")
    else:
        a = agg.iloc[0]
        for col in ("sample_id", "response_id", "source_index", "source_row_sha256", "response_sha256"):
            if not _is_null(a[col]):
                problems.add(f"aggregate score row carries {col}")
        value = _as_float(a["score_value"])
        evaluated = _as_int(a["evaluated_rows"])
        if value is None or not math.isfinite(value):
            problems.add("aggregate score is not finite")
        if evaluated != len(resp):
            problems.add(f"aggregate evaluated_rows {evaluated} != {len(resp)} responses")
        if str(a["score_unit"]) != "percent" or str(a["metric_id"]) != "primary":
            problems.add("aggregate row is not the primary percent metric")
        if str(a["scorer_id"]) != expected_scorer:
            problems.add(f"aggregate scorer_id {a['scorer_id']!r} != {expected_scorer!r}")
        if exp._score_id(identity, scope="aggregate", sample_id=None) != str(a["score_id"]):
            problems.add("aggregate score_id does not recompute")
        aggregate = {"official_value": value, "evaluated_rows": evaluated, "scorer_id": str(a["scorer_id"]),
                     "metric_id": str(a["metric_id"])}
    score_frame: pd.DataFrame | None = None
    if len(row_scores):
        rs = _normalize_identity_columns(row_scores, sl.label, "row scores")
        _check_unique(rs, "row scores", problems)
        if len(rs) != len(resp):
            problems.add(f"{len(rs)} row scores for {len(resp)} responses (partial row scoring)")
        bad_meta = rs[(rs["score_unit"].astype(str) != "fraction") | (rs["metric_id"].astype(str) != "primary")
                      | (rs["scorer_id"].astype(str) != expected_scorer)]
        if len(bad_meta):
            problems.add(f"{len(bad_meta)} row scores are not the primary fraction metric of {expected_scorer}")
        sc_cols = list(IDENTITY_FIELDS) + ["response_id", "response_sha256", *LINK_FIELDS, "score_id", "score_value",
                                           "score_value_json", "excluded", "score_unit"]
        joined_sc = _join_one_to_one(resp[base_cols], rs[sc_cols], LINK_FIELDS, "row scores", problems)
        values: list[float | None] = []
        excluded: list[bool] = []
        for r in joined_sc.itertuples(index=False):
            exc = _as_bool(r.excluded)
            val = _as_float(r.score_value)
            if exc is None:
                problems.add(f"ordinal {r.source_ordinal}: null excluded flag")
                exc = False
            if val is None and not exc:
                problems.add(f"ordinal {r.source_ordinal}: non-excluded row score is null")
            try:
                raw = json.loads(r.score_value_json)
                if (val is None) != (raw is None) or (val is not None and float(raw) != val):
                    problems.add(f"ordinal {r.source_ordinal}: score_value_json disagrees with score_value")
            except (ValueError, TypeError):
                problems.add(f"ordinal {r.source_ordinal}: unreadable score_value_json")
            if exp._score_id(identity, scope="row", sample_id=r.sample_id) != r.score_id:
                problems.add(f"ordinal {r.source_ordinal}: row score_id does not recompute")
            values.append(val)
            excluded.append(bool(exc))
        score_frame = joined_sc[list(IDENTITY_FIELDS) + ["score_value_json", "score_unit"]].copy()
        score_frame["archived_row_score"] = values
        score_frame["official_excluded"] = excluded
    problems.raise_if_any()

    # the data must be what DESIGN.md section 5 says this benchmark's archive holds
    if score_kind in ROW_SCORE_KINDS:
        if score_frame is None:
            raise IngestError(f"{sl.label}: {score_kind} benchmark has no row scores")
        observed = {v for v, e in zip(score_frame["archived_row_score"], score_frame["official_excluded"])
                    if v is not None and not e}
        if not observed <= {0.0, 1.0}:
            raise IngestError(f"{sl.label}: {score_kind} benchmark has row scores outside {{0, 1}}: "
                              f"{sorted(observed - {0.0, 1.0})[:3]}")
    elif score_frame is not None:
        raise IngestError(f"{sl.label}: aggregate_only benchmark carries {len(score_frame)} row scores")

    # normalized content columns
    keep = list(IDENTITY_FIELDS) + ["response_id", "response_sha256", "ordered_media_sha256", *LINK_FIELDS,
                                    "dataset_split", "dataset_revision", "response_text", "finish_reason",
                                    "completion_tokens", "prompt_tokens", "generation_settings_json"]
    frame = resp[keep].merge(
        joined_ext[list(IDENTITY_FIELDS) + ["extraction_status", "extraction_value_type", "extraction_value_json",
                                           "extraction_candidates_json", "extraction_sha256", "extraction_method",
                                           "used_judge", "judge_model_id", "judge_model_revision", "judge_output",
                                           "retry_count"]],
        on=list(IDENTITY_FIELDS), how="left", validate="one_to_one")
    if score_frame is not None:
        frame = frame.merge(score_frame, on=list(IDENTITY_FIELDS), how="left", validate="one_to_one")
    else:
        frame["archived_row_score"] = None
        frame["official_excluded"] = False
        frame["score_value_json"] = None
        frame["score_unit"] = None
    # rows matched through a verified producer alias take the manifest (public) identity; originals are kept
    frame["archived_source_row_sha256"] = None
    frame["archived_source_record_sha256"] = None
    frame["row_identity_status"] = ROW_IDENTITY_VERIFIED
    if canonical:
        patch = PRODUCER_ROW_HASH_ALIASES[sl.benchmark_id]["patch_id"]
        keys5 = list(zip(frame["source_ordinal"].astype(int), frame["source_index"].astype(str),
                         frame["source_row_sha256"], frame["source_record_sha256"], frame["media_set_sha256"]))
        hit = [k in canonical for k in keys5]
        frame["archived_source_row_sha256"] = [k[2] if h else None for k, h in zip(keys5, hit)]
        frame["archived_source_record_sha256"] = [k[3] if h else None for k, h in zip(keys5, hit)]
        frame["source_row_sha256"] = [canonical[k][2] if h else k[2] for k, h in zip(keys5, hit)]
        frame["source_record_sha256"] = [canonical[k][3] if h else k[3] for k, h in zip(keys5, hit)]
        frame["row_identity_status"] = [f"{ROW_IDENTITY_ALIASED}:{patch}" if h else ROW_IDENTITY_VERIFIED
                                        for h in hit]
    frame["content_key"] = [content_key(sl.benchmark_id, o, i, a, b) for o, i, a, b in zip(
        frame["source_ordinal"], frame["source_index"], frame["source_row_sha256"], frame["source_record_sha256"])]
    frame["score_kind"] = score_kind
    frame["official_metric_id"] = f"{sl.benchmark_id}:{aggregate.get('metric_id', 'primary')}"
    frame["scorer_id"] = aggregate.get("scorer_id", expected_scorer)
    frame["archived_row_score"] = [_as_float(v) for v in frame["archived_row_score"]]
    frame["official_excluded"] = [bool(_as_bool(v)) for v in frame["official_excluded"]]
    frame["score_available"] = [v is not None and not e for v, e in zip(frame["archived_row_score"],
                                                                         frame["official_excluded"])]
    frame["model_id"] = sl.model_id
    frame["run_id"] = sl.run_id
    frame["seed"] = int(sl.seed)
    frame["source_manifest_sha256"] = sl.source_manifest_sha256
    frame["source_parts_json"] = canonical_json(sl.parts)
    frame = frame.sort_values("source_ordinal", kind="stable").reset_index(drop=True)
    stats = {
        "rows": int(len(frame)),
        "row_scores": 0 if score_frame is None else int(len(score_frame)),
        "score_kind": score_kind,
        "score_available": int(sum(frame["score_available"])),
        "excluded": int(sum(frame["official_excluded"])),
        "finish_reason_null": int(frame["finish_reason"].isna().sum()),
        "completion_tokens_null": int(frame["completion_tokens"].isna().sum()),
        "prompt_tokens_null": int(frame["prompt_tokens"].isna().sum()),
        "generation_settings_variants": sorted(settings_seen),
        "dataset_sha256": dataset_sha,
        "rows_identity_aliased": len(canonical),
    }
    if frame["content_key"].duplicated().any():
        raise IngestError(f"{sl.label}: canonical content keys are not unique")
    return SliceResult(frame=frame, aggregate=aggregate, keys=frozenset(frame["content_key"]), stats=stats)


def finalize_frame(frame: pd.DataFrame, *, study_id: str, protocol_id: str, model_role: str, model_label: str,
                   model_revision: str, runtime_revision: str, source: str) -> pd.DataFrame:
    """Attach study/model labels and conform to NORMALIZED_SCHEMA column order and dtypes."""

    if protocol_id not in PROTOCOL_IDS:
        raise IngestError(f"unknown protocol_id {protocol_id!r}")
    if source not in SOURCES:
        raise IngestError(f"unknown source {source!r}")
    out = frame.copy()
    out["study_id"] = study_id
    out["protocol_id"] = protocol_id
    out["model_role"] = model_role
    out["model_label"] = model_label
    out["model_revision"] = model_revision
    out["runtime_revision"] = runtime_revision
    out["source"] = source
    out["valid_protocol_ids"] = valid_protocols_for(model_role, protocol_id)
    known_tokens = out["completion_tokens"].notna() | out["prompt_tokens"].notna()
    out["token_count_source"] = np.where(known_tokens, "engine_usage", None)
    for col in ("row_scorer", "row_scorer_output_json"):
        if col not in out.columns:
            out[col] = None
    for col in ("completion_tokens", "prompt_tokens", "retry_count"):
        out[col] = pd.array([_as_int(v) for v in out[col]], dtype="Int64")
    out["used_judge"] = pd.array([_as_bool(v) for v in out["used_judge"]], dtype="boolean")
    out["finish_reason"] = [_as_str(v) for v in out["finish_reason"]]
    out["ordered_media_sha256"] = [_as_str_list(v) for v in out["ordered_media_sha256"]]
    missing = [c for c in NORMALIZED_COLUMNS if c not in out.columns]
    if missing:
        raise IngestError(f"normalized frame lacks {missing}")
    return out[list(NORMALIZED_COLUMNS)].reset_index(drop=True)


def to_table(frame: pd.DataFrame) -> pa.Table:
    return pa.Table.from_pandas(frame[list(NORMALIZED_COLUMNS)], schema=NORMALIZED_SCHEMA, preserve_index=False)


# --------------------------------------------------------------------------------------------- archive ingestion
def _archive_model_roles(cfg: StudyConfig) -> dict[str, str]:
    out = {}
    for role in ROLES:
        mid = cfg.role(role)["archived_model_id"]
        if mid in out:
            raise IngestError(f"archived model id {mid!r} is mapped to two roles")
        out[mid] = role
    return out


def _check_archive_manifest(cfg: StudyConfig) -> tuple[Path, dict[str, Any], str]:
    root = cfg.archive_run_root
    path = root / "metadata" / "manifest.json"
    if not path.is_file():
        raise IngestError(f"archive manifest not found under the archive run root ({path})")
    digest = sha256_file(path)
    expected = cfg.raw["archive"]["manifest_sha256"]
    if digest != expected:
        raise IngestError(f"archive manifest sha256 {digest} != configured {expected}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("run_ids") != [cfg.raw["archive"]["run_id"]]:
        raise IngestError(f"archive manifest run_ids {manifest.get('run_ids')} != [{cfg.raw['archive']['run_id']}]")
    return root, manifest, digest


def archive_results_index(cfg: StudyConfig, manifest: Mapping[str, Any]) -> tuple[str, dict[tuple[str, int, str], tuple[float, int]]]:
    """Locate the results file pinned by ``result_sha256`` (metadata/results/benchmark_scores.json) and index it."""

    expected = cfg.raw["archive"]["result_sha256"]
    matches = [m for m in manifest.get("metadata_files") or [] if m.get("sha256") == expected]
    if len(matches) != 1:
        raise IngestError(f"result_sha256 {expected} matches {len(matches)} archive metadata files")
    rel = str(matches[0]["path"])
    path = cfg.archive_run_root / rel
    if not path.is_file() or sha256_file(path) != expected:
        raise IngestError(f"archive results file {rel} is missing or does not match result_sha256")
    data = json.loads(path.read_text(encoding="utf-8"))
    index: dict[tuple[str, int, str], tuple[float, int]] = {}
    for item in data.get("benchmark_scores") or []:
        key = (str(item["model_id"]), int(item["seed"]), str(item["benchmark_id"]))
        if key in index:
            raise IngestError(f"duplicate results entry {key}")
        index[key] = (float(item["score"]), int(item["evaluated_rows"]))
    return rel, index


def _archive_artifact_index(manifest: Mapping[str, Any]) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    index: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for entry in manifest.get("artifacts") or []:
        key = (str(entry["config_name"]), str(entry["model_id"]), int(entry["seed"]), str(entry["benchmark_id"]))
        if key in index:
            raise IngestError(f"archive manifest lists {key} twice")
        index[key] = entry
    return index


def _read_archive_part(root: Path, entry: Mapping[str, Any], *, verify: bool) -> pd.DataFrame:
    """Read one archive part after checking its digest, size and part manifest against the archive manifest."""

    path = root / str(entry["parquet_path"])
    mpath = root / str(entry["manifest_path"])
    if not path.is_file() or not mpath.is_file():
        raise IngestError(f"archive part missing: {entry['parquet_path']}")
    if verify:
        if path.stat().st_size != int(entry["parquet_size"]) or sha256_file(path) != entry["parquet_sha256"]:
            raise IngestError(f"archive part digest/size mismatch: {entry['parquet_path']}")
        if sha256_file(mpath) != entry["manifest_sha256"]:
            raise IngestError(f"archive part manifest digest mismatch: {entry['manifest_path']}")
        part = json.loads(mpath.read_text(encoding="utf-8"))
        ident = part.get("identity") or {}
        if (part.get("parquet_sha256") != entry["parquet_sha256"] or int(part.get("rows", -1)) != int(entry["rows"])
                or ident.get("model_id") != entry["model_id"] or int(ident.get("seed", -1)) != int(entry["seed"])
                or ident.get("benchmark_id") != entry["benchmark_id"]):
            raise IngestError(f"archive part manifest disagrees with the archive manifest: {entry['manifest_path']}")
    frame = read_parquet_frame(path)
    if len(frame) != int(entry["rows"]):
        raise IngestError(f"archive part {entry['parquet_path']} has {len(frame)} rows, manifest says {entry['rows']}")
    return frame


def archive_slice(cfg: StudyConfig, root: Path, artifacts: Mapping[tuple[str, str, int, str], Mapping[str, Any]],
                  model_id: str, seed: int, benchmark: str, *, verify: bool = True) -> PublicSlice:
    entries = {}
    for config_name in CONFIG_NAMES:
        entry = artifacts.get((config_name, model_id, seed, benchmark))
        if entry is None:
            raise IngestError(f"archive has no {config_name} part for {model_id}/seed={seed}/{benchmark}")
        entries[config_name] = entry
    frames = {c: _read_archive_part(root, e, verify=verify) for c, e in entries.items()}
    role = _archive_model_roles(cfg)[model_id]
    return PublicSlice(
        benchmark_id=benchmark, model_id=model_id, model_revision=cfg.role(role)["archived_model_revision"],
        run_id=cfg.raw["archive"]["run_id"], seed=seed, responses=frames["responses"],
        extractions=frames["extractions"], scores=frames["scores"],
        parts={c: str(e["manifest_sha256"]) for c, e in entries.items()},
        source_manifest_sha256=str(entries["responses"]["manifest_sha256"]))


def _missingness_row(*, source: str, run_name: str, protocol_id: str, label: str, seed: int, benchmark: str,
                     expected_keys: set[str], keys: Iterable[str], stats: Mapping[str, Any] | None,
                     status: str) -> dict[str, Any]:
    keys = set(keys)
    stats = stats or {}
    score_kind = stats.get("score_kind")
    n_rows = int(stats.get("rows", len(keys)))
    row_missing = (n_rows - int(stats.get("score_available", 0))) if score_kind and score_kind != "aggregate_only" else 0
    return {
        "source": source, "run_name": run_name, "protocol_id": protocol_id, "model_label": label, "seed": seed,
        "benchmark_id": benchmark, "expected_rows": len(expected_keys), "response_rows": n_rows,
        "row_score_rows": int(stats.get("row_scores", 0)), "missing_rows": len(expected_keys - keys),
        "extra_rows": len(keys - expected_keys), "score_kind": score_kind,
        "score_available_rows": int(stats.get("score_available", 0)), "excluded_rows": int(stats.get("excluded", 0)),
        "row_score_unavailable_rows": row_missing, "finish_reason_null": int(stats.get("finish_reason_null", 0)),
        "completion_tokens_null": int(stats.get("completion_tokens_null", 0)),
        "prompt_tokens_null": int(stats.get("prompt_tokens_null", 0)), "status": status,
    }


def _write_missingness(cfg: StudyConfig, rows: list[dict[str, Any]], *, replace_run: str) -> Path:
    """audit/missingness.csv; every writer (archive, each local run) replaces only its own ``run_name`` rows."""

    path = cfg.work("audit", "missingness.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if path.is_file():
        old = pd.read_csv(path)
        old = old[old["run_name"] != replace_run]
        new = pd.concat([old, new], ignore_index=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    new.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return path


def _update_index(cfg: StudyConfig, update: Callable[[dict[str, Any]], None]) -> Path:
    path = cfg.work("aligned", "index.json")
    index = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    index.setdefault("layout", LAYOUT_DOC)
    index.setdefault("schema", "aligned_predictions_v1")
    index.setdefault("local_runs", {})
    update(index)
    write_json(path, index)
    return path


def ingest_archive(cfg: StudyConfig, *, benchmarks: Sequence[str] | None = None, seeds: Sequence[int] | None = None,
                   verify_files: bool = True, require_full_coverage: bool = True,
                   producer_aliases: Sequence[str] | None = None, log: Log = print) -> dict[str, Path]:
    """Load, validate and normalize the archived B/T/V run; write the aligned archive table and audits.

    Fails (after writing ``audit/missingness.csv`` and ``audit/archive_ingest.json``) if any slice violates an
    identity, integrity or join rule, if the 9 slices of a benchmark do not share one key set, if the slices do not
    share one normalized generation setting equal to the configured protocol, or (by default) if a slice does not
    cover every prepared row of its benchmark. ``benchmarks``/``seeds`` restrict the run (tests);
    ``producer_aliases`` overrides the config's enabled producer row-hash aliases (default: none).
    """

    root, manifest, manifest_sha = _check_archive_manifest(cfg)
    results_rel, results = archive_results_index(cfg, manifest)
    artifacts = _archive_artifact_index(manifest)
    model_roles = _archive_model_roles(cfg)
    seeds = [int(s) for s in (seeds or cfg.raw["archive"]["seeds"])]
    benchmarks = list(benchmarks or cfg.benchmark_keys())
    alias_names, alias_source = enabled_producer_aliases(cfg, producer_aliases)
    dmi = load_dataset_manifest_index(cfg.dataset_manifest, benchmarks, aliases=alias_names)
    provenance = protocol_code_provenance(cfg)
    out_dir = cfg.work("aligned")
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "predictions.parquet"
    tmp = out_dir / f".predictions.parquet.tmp.{os.getpid()}"
    writer = pq.ParquetWriter(tmp, NORMALIZED_SCHEMA, compression="zstd")
    problems: list[str] = []
    aggregates: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    slices: list[dict[str, Any]] = []
    settings_variants: Counter[str] = Counter()
    aliased_rows: dict[str, int] = {}
    dataset_shas: Counter[str] = Counter()
    total_rows = 0
    try:
        for bench in benchmarks:
            expected_keys = dmi.expected_content_keys(bench)
            keysets: dict[tuple[str, int], frozenset[str]] = {}
            for model_id, role in model_roles.items():
                label = ARCHIVE_LABELS[role]
                for seed in seeds:
                    try:
                        sl = archive_slice(cfg, root, artifacts, model_id, seed, bench, verify=verify_files)
                        res = validate_public_slice(sl, dmi, allow_aliases=True)
                    except IngestError as error:
                        problems.append(str(error))
                        missing_rows.append(_missingness_row(
                            source="archive", run_name="archive", protocol_id="P0_native", label=label, seed=seed,
                            benchmark=bench, expected_keys=expected_keys, keys=(), stats=None, status="invalid"))
                        continue
                    agg = res.aggregate
                    published = results.get((model_id, seed, bench))
                    if published is None:
                        problems.append(f"{sl.label}: no entry in the archive results file")
                    elif (abs(published[0] - agg["official_value"]) > 1e-9
                          or published[1] != agg["evaluated_rows"]):
                        problems.append(f"{sl.label}: aggregate {agg['official_value']} / {agg['evaluated_rows']} "
                                        f"!= results file {published}")
                    frame = finalize_frame(res.frame, study_id=cfg.study_id, protocol_id="P0_native",
                                           model_role=role, model_label=label,
                                           model_revision=cfg.role(role)["source_revision"],
                                           runtime_revision=sl.model_revision, source="archive")
                    writer.write_table(to_table(frame))
                    total_rows += len(frame)
                    keysets[(label, seed)] = res.keys
                    settings_variants.update(res.stats["generation_settings_variants"])
                    aliased_rows[f"{label}/seed={seed}/{bench}"] = res.stats["rows_identity_aliased"]
                    dataset_shas[res.stats["dataset_sha256"]] += 1
                    status = "complete" if res.keys == expected_keys else "incomplete"
                    missing_rows.append(_missingness_row(
                        source="archive", run_name="archive", protocol_id="P0_native", label=label, seed=seed,
                        benchmark=bench, expected_keys=expected_keys, keys=res.keys, stats=res.stats, status=status))
                    if require_full_coverage and res.keys != expected_keys:
                        problems.append(f"{sl.label}: covers {len(res.keys & expected_keys)}/{len(expected_keys)} "
                                        f"prepared rows")
                    aggregates.append({
                        "protocol_id": "P0_native", "model_label": label, "model_role": role, "seed": seed,
                        "benchmark_id": bench, "official_value": agg["official_value"],
                        "evaluated_rows": agg["evaluated_rows"], "scorer_id": agg["scorer_id"],
                        "metric_id": agg["metric_id"], "source": "archive", "run_id": sl.run_id,
                        "model_id": model_id, "score_kind": res.stats["score_kind"],
                        "results_file_value": None if published is None else published[0]})
                    slices.append({"model_label": label, "seed": seed, "benchmark_id": bench,
                                   "rows": res.stats["rows"], "score_kind": res.stats["score_kind"],
                                   "parts": sl.parts})
                    del frame, res, sl
            distinct = {ks for ks in keysets.values()}
            if len(distinct) > 1:
                sizes = {f"{k[0]}/seed={k[1]}": len(v) for k, v in keysets.items()}
                problems.append(f"{bench}: content key sets differ across model x seed slices {sizes}")
            log(f"[ingest] {bench}: {len(keysets)} slices, {len(expected_keys)} prepared rows")
    finally:
        writer.close()
    # one decoding contract for every archived slice, equal to the configured protocol (handoff 4.2.5)
    normalized = sorted({normalized_settings_json(s) for s in settings_variants})
    if len(normalized) != 1:
        problems.append(f"archived slices carry {len(normalized)} distinct normalized generation settings")
    for value in normalized:
        bad = configured_settings_problems(cfg, value)
        if bad:
            problems.append(f"archived generation settings differ from the configured protocol: {bad}")
    alias_status = {}
    for name in alias_names:
        patch = PRODUCER_ROW_HASH_ALIASES[name]["patch_id"]
        alias_status[name] = {"patch_id": patch, "enabled_by": alias_source,
                              "design_amendment_recorded": design_amendment_recorded(patch)}
    report = {
        "study_id": cfg.study_id, "config_sha256": cfg.sha256, "archive": {
            "repository_id": cfg.raw["archive"]["repository_id"], "revision": cfg.raw["archive"]["revision"],
            "run_id": cfg.raw["archive"]["run_id"], "manifest_sha256": manifest_sha, "results_file": results_rel,
            "result_sha256": cfg.raw["archive"]["result_sha256"]},
        "dataset_manifest": {"sha256": dmi.sha256, "schema_version": dmi.schema_version,
                             "dataset_snapshot_sha256": dmi.snapshot_sha256,
                             "producer_row_hash_aliases": dmi.alias_reports},
        "producer_row_hash_aliases_enabled": alias_status,
        "pending_design_amendments": [f"{n}: {v['patch_id']} (DESIGN.md section 4 requires a direct manifest match)"
                                      for n, v in alias_status.items() if not v["design_amendment_recorded"]],
        "rows_identity_aliased": aliased_rows,
        "archive_dataset_sha256": dict(dataset_shas),
        "protocol_code": provenance, "benchmarks": benchmarks, "seeds": seeds,
        "model_labels": {mid: ARCHIVE_LABELS[r] for mid, r in model_roles.items()},
        "generation_settings_variants": dict(settings_variants), "normalized_generation_settings": normalized,
        "rows": total_rows, "slices": slices,
        "checks": [
            "archive manifest sha256 equals the configured receipt", "every part's sha256/size/part manifest verified",
            "response/extraction/score digests recomputed with the pinned exporter helpers",
            "content identity unique per slice; extraction and row-score joins one-to-one on identity + response id",
            "every row matches exactly one dataset-manifest row (5-tuple + media set + ordered media), directly or "
            "through a verified producer row-hash alias (PRODUCER_ROW_HASH_ALIASES; originals kept in archived_*)",
            "identical key sets across the 9 slices of each benchmark",
            "one normalized generation setting (minus model_id/seed) across slices, equal to protocol.generation",
            "slice aggregates equal the archive results file"],
        "problems": problems, "status": "failed" if problems else "verified",
    }
    _write_missingness(cfg, missing_rows, replace_run="archive")   # keeps registered local runs' rows
    write_json(cfg.work("audit", "archive_ingest.json"), report)
    if problems:
        tmp.unlink(missing_ok=True)
        raise IngestError(f"archive ingestion failed with {len(problems)} problem(s): " + " | ".join(problems[:8]))
    os.replace(tmp, final)
    agg_path = out_dir / "official_aggregates.parquet"
    pq.write_table(pa.Table.from_pylist(aggregates, schema=AGGREGATE_SCHEMA), agg_path, compression="zstd")

    def _upd(index: dict[str, Any]) -> None:
        index["predictions"] = {"path": "predictions.parquet", "rows": total_rows, "sha256": sha256_file(final),
                                "source": "archive", "protocol_id": "P0_native", "slices": len(slices)}
        index["official_aggregates"] = {"path": "official_aggregates.parquet", "rows": len(aggregates),
                                        "sha256": sha256_file(agg_path)}

    index_path = _update_index(cfg, _upd)
    log(f"[ingest] archive verified: {total_rows} rows, {len(slices)} slices -> {final}")
    if report["pending_design_amendments"]:
        log(f"[ingest] WARNING pending DESIGN.md amendments: {report['pending_design_amendments']}")
    return {"predictions": final, "official_aggregates": agg_path, "missingness": cfg.work("audit", "missingness.csv"),
            "report": cfg.work("audit", "archive_ingest.json"), "index": index_path}


# --------------------------------------------------------------------------------------------- local runs
def _slugify(value: str) -> str:
    out = re.sub(r"[^a-z0-9._-]+", "-", str(value).lower()).strip("-._")
    if not out:
        raise IngestError(f"cannot form a public model id from {value!r}")
    return out


_SCORER_DROP_KEYS = frozenset({"question", "ground_truth", "options", "prompt", "source_row", "record_id",
                               "request_hash", "source_index", "source_ordinal", "source_row_hash", "metadata",
                               "scorer"})


def _looks_like_path(value: Any) -> bool:
    return isinstance(value, str) and (value.startswith("/") or value.startswith("file://"))


def sanitize_scorer_output(record: Mapping[str, Any]) -> tuple[str | None, str]:
    """Per-row scorer outputs from the score spool, without benchmark inputs, prompts or machine paths."""

    out = {k: v for k, v in record.items() if k not in _SCORER_DROP_KEYS}
    meta = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
    out["metadata"] = {k: v for k, v in meta.items()
                       if not any(t in k.lower() for t in ("prompt", "path", "artifact")) and not _looks_like_path(v)}
    scorer = record.get("scorer")
    return (None if scorer is None else str(scorer)), canonical_json(out)


def _load_spool(run_dir: Path, seed: int) -> dict[tuple[str, str], tuple[dict[str, Any], list[dict[str, Any]]]]:
    hf = protocol_module(ARCHIVE_LIB)
    spool = run_dir / "hf_archive"
    ready = sorted((spool / "ready").glob("*.ready.json"))
    if not ready:
        raise IngestError(f"no ready spool descriptors under {spool}")
    out: dict[tuple[str, str], tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for path in ready:
        # the seeds of one model and subset share a run root and spool (eval_subset.sh ROOT=<slug>/<subset>): another
        # seed's descriptors are skipped after a cheap identity read; this seed's are fully verified below
        try:
            other = int(json.loads(path.read_text(encoding="utf-8"))["identity"]["seed"]) != int(seed)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise IngestError(f"spool descriptor {path.name} is unreadable: {error}") from error
        if other:
            continue
        try:
            desc = hf.load_descriptor(path, spool_root=spool)   # descriptor digest + payload sha256
        except hf.ArchiveError as error:
            raise IngestError(f"spool descriptor {path.name} failed verification: {error}") from error
        ident = desc["identity"]
        if int(ident["seed"]) != int(seed):
            raise IngestError(f"spool descriptor {path.name} is seed {ident['seed']}, expected {seed}")
        key = (str(ident["benchmark"]), str(ident["stage"]))
        if key in out:
            raise IngestError(f"spool holds two {key[1]} slices for {key[0]}")
        payload = spool / desc["payload_path"]
        records = [json.loads(line) for line in payload.read_text(encoding="utf-8").splitlines() if line]
        if len(records) != int(desc["rows"]):
            raise IngestError(f"spool payload {payload.name} row count mismatch")
        for rec in records:
            expected = hf.stable_record_id(
                run_id=ident["run_id"], model_slug=ident["model_slug"], model_revision=ident["model_revision"],
                seed=int(ident["seed"]), benchmark=ident["benchmark"], dataset_alias=ident["dataset_alias"],
                dataset_split=ident["dataset_split"], dataset_revision=ident["dataset_revision"],
                source_index=rec.get("source_index"), source_ordinal=rec.get("source_ordinal"),
                source_row_hash=rec.get("source_row_hash"))
            if rec.get("record_id") != expected:
                raise IngestError(f"spool record in {payload.name} is not bound to its descriptor identity")
        out[key] = (desc, records)
    if not out:
        raise IngestError(f"no spool descriptors for seed {seed} under {spool}")
    common = {(d["identity"]["run_id"], d["identity"]["model_slug"], d["identity"]["model_revision"],
               d["identity"]["dataset_revision"]) for d, _ in out.values()}
    if len(common) != 1:
        raise IngestError(f"spool descriptors disagree on run/model/dataset identity: {sorted(common)[:3]}")
    return out


def repair_subset_stage_ordinals(generation: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]],
                                 identity: Mapping[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """Re-attach extraction/score records whose ordinal was renumbered by subset position (row-subset runs).

    In a row-subset run the official-VLMEvalKit route re-reads the subset table, so its extraction and score records
    carry the row's *position within the subset* as ``source_ordinal`` (and a ``record_id`` derived from it), while the
    generation record keeps the full-split ordinal; the exporter joins stages by ``record_id`` and rejects such a run.
    A record is re-attached only if exactly one generation record has the same ``(source_index, source_row_hash)`` and
    the record's ordinal equals that generation record's position among the run's generation records sorted by
    ordinal; its ordinal and ``record_id`` (recomputed with the spool's ``stable_record_id``) then become the
    generation's. Anything else is left unchanged for the exporter to reject; nothing is matched by position alone.
    """

    hf = protocol_module(ARCHIVE_LIB)
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for rec in generation:
        key = (str(rec["source_index"]), str(rec["source_row_hash"]))
        if key in by_key:
            raise IngestError(f"generation records share (source_index, source_row_hash) {key}")
        by_key[key] = rec
    ordered = sorted(generation, key=lambda r: (int(r["source_ordinal"]), str(r["source_index"])))
    position = {str(r["record_id"]): i for i, r in enumerate(ordered)}
    out: list[dict[str, Any]] = []
    repaired = 0
    for rec in records:
        gen = by_key.get((str(rec.get("source_index")), str(rec.get("source_row_hash"))))
        if (str(rec.get("source_index")) == "__aggregate__" or gen is None
                or int(rec["source_ordinal"]) == int(gen["source_ordinal"])):
            out.append(dict(rec))
            continue
        if int(rec["source_ordinal"]) != position[str(gen["record_id"])]:
            raise IngestError(f"stage record ordinal {rec['source_ordinal']} for index {rec.get('source_index')} is "
                              f"neither the generation ordinal {gen['source_ordinal']} nor its subset position")
        fixed = dict(rec, source_ordinal=int(gen["source_ordinal"]))
        fixed["record_id"] = hf.stable_record_id(
            run_id=identity["run_id"], model_slug=identity["model_slug"], model_revision=identity["model_revision"],
            seed=int(identity["seed"]), benchmark=identity["benchmark"], dataset_alias=identity["dataset_alias"],
            dataset_split=identity["dataset_split"], dataset_revision=identity["dataset_revision"],
            source_index=fixed["source_index"], source_ordinal=fixed["source_ordinal"],
            source_row_hash=fixed["source_row_hash"])
        if fixed["record_id"] != gen["record_id"]:
            raise IngestError(f"re-attached record for index {rec.get('source_index')} does not reproduce the "
                              f"generation record id")
        out.append(fixed)
        repaired += 1
    return out, repaired


def spool_provenance(cfg: StudyConfig,
                     spool: Mapping[tuple[str, str], tuple[dict[str, Any], list[dict[str, Any]]]]) -> dict[str, Any]:
    """The run's producer provenance; it must be the pinned protocol (handoff 4.2.5).

    Every descriptor must name the configured protocol commit and VLMEvalKit commit, and the run must carry exactly
    one non-empty code hash, campaign config hash and archive hook version (the publish worker's own rule), so a run
    produced by other code cannot pass as the pinned protocol.
    """

    proto = cfg.raw["protocol"]
    values: dict[str, set[str]] = {k: set() for k in PROVENANCE_FIELDS}
    contracts: dict[str, set[str]] = {}
    for (_bench, stage), (desc, _records) in spool.items():
        prov = desc.get("provenance") or {}
        for k in PROVENANCE_FIELDS:
            values[k].add(str(prov.get(k) or ""))
        contracts.setdefault(stage, set()).add(str(prov.get("contract_version") or ""))
    problems = [f"{k} missing or not unique across descriptors: {sorted(v)[:3]}" for k, v in values.items()
                if "" in v or len(v) != 1]
    if values["trace_git_commit"] != {proto["code_revision"]}:
        problems.append(f"trace_git_commit {sorted(values['trace_git_commit'])[:2]} != protocol.code_revision")
    if values["vlmevalkit_git_commit"] != {proto["vlmevalkit_revision"]}:
        problems.append(f"vlmevalkit_git_commit {sorted(values['vlmevalkit_git_commit'])[:2]} != "
                        f"protocol.vlmevalkit_revision")
    if problems:
        raise IngestError("spool provenance is not the pinned protocol: " + "; ".join(problems))
    out: dict[str, Any] = {k: next(iter(v)) for k, v in values.items()}
    out["contract_versions"] = {s: sorted(v) for s, v in sorted(contracts.items())}
    return out


def _read_json(path: Path, what: str) -> dict[str, Any]:
    if not path.is_file():
        raise IngestError(f"{what} not found ({path.name})")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_served_model(cfg: StudyConfig, label: str, role: str, protocol_id: str,
                        identity: Mapping[str, Any]) -> dict[str, Any]:
    """Tie the spool's served model to the label (DESIGN.md section 3); return weight/runtime revisions.

    * P0 (local reruns of B/T): the protocol clone's prepared native model, at the pinned commit (its revision marker
      names the commit; the served file set's content revision is recorded, since it can differ from the archived
      runtime view, e.g. T's 18-file snapshot versus its archived 10-file ``sha256set`` view).
    * P1/P2: the registered study view of the label. ``manifests/views.json``, the view directory's
      ``view_manifest.json`` and the protocol's content-bound marker must agree; the view must carry the protocol's
      view parent and file set, its weights must be the label's weight source (a role's pinned commit, whose LFS blob
      names are checked against the marker, or the merge's content revision in ``manifests/merges.json``), and the
      spool's revision must be the marker's.
    """

    slug, revision = str(identity["model_slug"]), str(identity["model_revision"])
    if protocol_id == "P0_native":
        native = NATIVE_PROTOCOL_SLUGS.get(role)
        if native is None:
            raise IngestError(f"{label}: the protocol clone has no prepared native model for role {role}; a native "
                              f"local run of it is not part of the design")
        pinned = cfg.role(role)["source_revision"]
        if slug != native or revision != pinned:
            raise IngestError(f"{label}: spool serves {slug}@{revision}, expected the prepared native {native}@{pinned}")
        marker = _read_json(cfg.require_protocol_clone() / "rlvr/evaluation/.work/models" / native / MODEL_MARKER,
                            f"{label}: the prepared model's revision marker")
        if (marker.get("slug") != native or marker.get("resolved_commit") != pinned
                or marker.get("immutable_revision") != revision):
            raise IngestError(f"{label}: the prepared model marker does not name {native}@{pinned}")
        files = {str(k): str(v) for k, v in (marker.get("file_sha256") or {}).items()}
        return {"model_revision": pinned, "runtime_revision": revision, "served_view": {
            "kind": "protocol_prepared_native", "slug": native, "immutable_revision": revision,
            "resolved_commit": marker.get("resolved_commit"), "file_count": len(files),
            "content_revision": content_revision(files), "files": sorted(files)}}
    views = _read_json(cfg.work("manifests", "views.json"), "manifests/views.json")
    entry = views.get(label)
    if not isinstance(entry, dict):
        raise IngestError(f"{label}: not a registered study view in manifests/views.json")
    if entry.get("protocol_id") != protocol_id or entry.get("label") != label:
        raise IngestError(f"{label}: manifests/views.json files it under {entry.get('protocol_id')}, not {protocol_id}")
    if slug != entry.get("slug"):
        raise IngestError(f"{label}: spool serves {slug!r}, the registered view is {entry.get('slug')!r}")
    vdir = cfg.work("views", slug)
    manifest = _read_json(vdir / "view_manifest.json", f"{label}: view manifest")
    keys = ("label", "slug", "protocol_id", "weight_source", "weight_source_revision", "view_parent",
            "view_parent_revision", "view_files_sha256", "weights")
    differ = [k for k in keys if manifest.get(k) != entry.get(k)]
    if differ:
        raise IngestError(f"{label}: the view directory's manifest differs from manifests/views.json on {differ}")
    parent = str(cfg.raw["protocols"][protocol_id]["view"])
    if manifest["view_parent"] != parent or manifest["view_parent_revision"] != cfg.role(parent)["source_revision"]:
        raise IngestError(f"{label}: view parent {manifest['view_parent']}@{manifest['view_parent_revision']} is not "
                          f"{protocol_id}'s pinned {parent}")
    if set(manifest["view_files_sha256"]) != set(cfg.raw["protocols"][protocol_id]["files"]):
        raise IngestError(f"{label}: the view does not carry exactly {protocol_id}'s non-weight files")
    if manifest["weight_source"] != role:
        raise IngestError(f"{label}: the view serves {manifest['weight_source']} weights, not {role}")
    if role in ROLES:
        expected = cfg.role(role)["source_revision"]
    else:
        merges = _read_json(cfg.work("manifests", "merges.json"), "manifests/merges.json")
        expected = (merges.get(role) or {}).get("content_revision")
    if not expected or manifest["weight_source_revision"] != expected:
        raise IngestError(f"{label}: view weight revision {manifest['weight_source_revision']} != {role}'s {expected}")
    marker = _read_json(vdir / MODEL_MARKER, f"{label}: the view's registered revision marker")
    files = {str(k): str(v) for k, v in (marker.get("file_sha256") or {}).items()}
    if content_revision(files) != marker.get("immutable_revision") or marker.get("slug") != slug:
        raise IngestError(f"{label}: the view marker's revision does not recompute from its file digests")
    if marker["immutable_revision"] != revision:
        raise IngestError(f"{label}: spool revision {revision} is not the registered view {marker['immutable_revision']}")
    if files.get("view_manifest.json") != sha256_file(vdir / "view_manifest.json"):
        raise IngestError(f"{label}: the registered marker does not bind this view_manifest.json")
    bad = sorted(n for n, h in manifest["view_files_sha256"].items() if files.get(n) != h)
    if role in ROLES:
        # the full parent weight hashes verified by `parent-hashes` are authoritative; a resolved HF blob name is the
        # sha256 only when the cache does not link blobs onward into a deduplicated store (B's and T's do)
        audited = cfg.work("audit", "parent_weight_sha256.json")
        full = (_read_json(audited, "audit/parent_weight_sha256.json").get(role) or {}) if audited.is_file() else {}
        if full:
            bad += sorted(n for n in manifest["weights"] if n.endswith(".safetensors") and files.get(n) != full.get(n))
        else:
            bad += sorted(n for n, blob in manifest["weights"].items()
                          if re.fullmatch(r"[0-9a-f]{64}", str(blob)) and files.get(n) != blob)
    if bad:
        raise IngestError(f"{label}: the served files differ from the view manifest: {bad[:4]}")
    return {"model_revision": expected, "runtime_revision": revision, "served_view": {
        "kind": "study_view", "slug": slug, "immutable_revision": revision, "view_parent": parent,
        "weight_source": role, "weight_source_revision": expected, "file_count": len(files)}}


def load_run_plan(run_dir: Path, seed: int) -> dict[str, Any]:
    """The run's own plan (``plan.seed<S>.json``): planned benchmarks, row counts and (subsets) the row list."""

    plan = _read_json(run_dir / f"plan.seed{seed}.json", "run plan")
    if _as_int(plan.get("seed")) != seed:
        raise IngestError(f"run plan is for seed {plan.get('seed')}, expected {seed}")
    rows = plan.get("expected_rows")
    if not isinstance(rows, dict) or not rows:
        raise IngestError("run plan lists no expected_rows")
    return plan


def row_list_sha256(payload: Mapping[str, Any], suite_keys: Sequence[str]) -> str:
    """The protocol's row-list digest: suite order, then manifest ordinal order, one ``key<TAB>row_id`` line each."""

    lines = []
    for key in suite_keys:
        for r in sorted((payload["benchmarks"].get(key) or {}).get("rows") or [], key=lambda r: int(r["ordinal"])):
            lines.append(f"{key}\t{r['row_id']}\n")
    return sha256_bytes("".join(lines).encode("utf-8"))


def planned_content_keys(cfg: StudyConfig, run_dir: Path, plan: Mapping[str, Any], dmi: DatasetManifestIndex, *,
                         strict: bool) -> tuple[dict[str, frozenset[str]], dict[str, Any]]:
    """Content keys each planned benchmark must cover exactly (handoff 4.2.6: missing rows are listed, not dropped).

    A full-split plan expects every prepared row. A subset plan's row list (the copy in the run directory) must
    recompute its ``row_list_sha256``, equal the plan's and, for study runs, the frozen split's row list of that
    subset; each row must be the prepared dataset row at its ordinal.
    """

    expected = {str(k): int(v) for k, v in plan["expected_rows"].items() if int(v) > 0}
    info: dict[str, Any] = {"subset": plan.get("subset"), "row_list_sha256": plan.get("row_list_sha256"),
                            "planned_rows": expected, "warnings": []}
    unknown = sorted(set(expected) - set(dmi.rows))
    if unknown:
        raise IngestError(f"run plan names benchmarks outside the suite/manifest: {unknown}")
    if not plan.get("rows_file"):
        keys = {b: frozenset(dmi.expected_content_keys(b)) for b in expected}
        bad = {b: (n, len(keys[b])) for b, n in expected.items() if n != len(keys[b])}
        if bad:
            raise IngestError(f"full-split plan row counts differ from the prepared dataset: {bad}")
        info["mode"] = "full_split"
        return keys, info
    payload = _read_json(run_dir / Path(str(plan["rows_file"])).name, "the run's row list")
    sha = row_list_sha256(payload, cfg.benchmark_keys())
    if sha != payload.get("row_list_sha256") or sha != plan.get("row_list_sha256"):
        raise IngestError(f"row list digest {sha} != recorded {payload.get('row_list_sha256')} / plan "
                          f"{plan.get('row_list_sha256')}")
    split_path = cfg.work("splits", "split_manifest.json")
    frozen = {}
    if split_path.is_file():
        frozen = (json.loads(split_path.read_text(encoding="utf-8")).get("row_lists") or {}).get(
            str(plan.get("subset"))) or {}
    if frozen:
        if frozen.get("row_list_sha256") != sha:
            raise IngestError(f"subset {plan.get('subset')!r} row list {sha} != the frozen split's "
                              f"{frozen.get('row_list_sha256')}")
        info["frozen_split_row_list"] = True
    elif strict:
        raise IngestError(f"subset {plan.get('subset')!r} is not a frozen row list of this study "
                          f"(splits/split_manifest.json)")
    else:
        info["warnings"].append(f"subset {plan.get('subset')!r} is not a frozen row list of this study")
    keys: dict[str, frozenset[str]] = {}
    for bench, n in expected.items():
        by_ordinal = {k[0]: k for k in dmi.rows[bench]}
        rows = (payload["benchmarks"].get(bench) or {}).get("rows") or []
        found: set[str] = set()
        for r in rows:
            k = by_ordinal.get(int(r["ordinal"]))
            if k is None or k[1] != str(r["index"]) or k[2] != str(r["source_row_hash"]):
                raise IngestError(f"{bench}: planned row {r['ordinal']}/{r['index']} is not the prepared dataset row")
            ck = content_key(bench, k[0], k[1], k[2], k[3])
            if r.get("content_key") is not None and str(r["content_key"]) != ck:
                raise IngestError(f"{bench}: planned row {r['ordinal']} carries another content key")
            found.add(ck)
        if len(found) != n:
            raise IngestError(f"{bench}: row list has {len(found)} distinct rows, the plan expects {n}")
        keys[bench] = frozenset(found)
    info["mode"] = "row_subset"
    return keys, info


def load_campaign_summary(run_dir: Path, seed: int, plan: Mapping[str, Any]) -> dict[str, dict[str, Any]] | None:
    """The run's campaign summary per benchmark (``summary.seed<S>.json``), checked to be this run's; None if absent."""

    path = run_dir / f"summary.seed{seed}.json"
    if not path.is_file():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    if _as_int(summary.get("seed")) != seed or summary.get("row_list_sha256") != plan.get("row_list_sha256"):
        raise IngestError("campaign summary is for another seed or row list than the run plan")
    if summary.get("complete") is not True:
        raise IngestError("campaign summary says the run is incomplete")
    return {str(b["key"]): b for b in summary.get("benchmarks") or []}


def summary_check(aggregate: Mapping[str, Any], entry: Mapping[str, Any] | None) -> tuple[str, float | None, str | None]:
    """How a spool aggregate relates to the campaign summary: (status, summary value, problem or None).

    ``equal``: the summary reports the spool's value. ``summary_rescaled``: the spool holds the official evaluator's
    value (the summary's ``score_protocol``) and the summary reports another value for the subset (e.g. WeMath's
    strict score over its complete problem groups); both are kept, metrics.py decides which one an adapter must
    reproduce. Anything else is a problem.
    """

    if entry is None:
        return "missing", None, "benchmark is absent from the campaign summary"
    value = float(aggregate["official_value"])
    score = _as_float(entry.get("score"))
    if _as_int(entry.get("rows")) != int(aggregate["evaluated_rows"]):
        return "rows_differ", score, f"summary rows {entry.get('rows')} != {aggregate['evaluated_rows']}"
    if score is not None and abs(score - value) <= SUMMARY_TOLERANCE_PP:
        return "equal", score, None
    protocol = _as_float(entry.get("score_protocol"))
    if protocol is not None and abs(protocol - value) <= SUMMARY_TOLERANCE_PP:
        return "summary_rescaled", score, None
    return "differs", score, f"spool aggregate {value} != summary score {score} (score_protocol {protocol})"


def family_completeness(cfg: StudyConfig, dmi: DatasetManifestIndex, bench: str,
                        keys: frozenset[str]) -> dict[str, Any] | None:
    """For benchmarks with problem families (WeMath ``ID``): does the run hold every row of each family it touches?

    A grouped official metric computed on a partial family is not the official statistic; the study's subsets are
    unions of connected components (DESIGN.md section 4), so families are complete by construction.
    """

    cols = list((cfg.raw.get("split", {}).get("family_columns") or {}).get(bench) or [])
    if not cols:
        return None
    files = dmi.receipts[bench]["metadata_files"]
    if len(files) != 1:
        raise IngestError(f"{bench}: expected one licensed metadata file for its problem families")
    tsv = cfg.dataset_manifest.parent / files[0]["name"]
    if not tsv.is_file() or sha256_file(tsv) != files[0]["sha256"]:
        raise IngestError(f"{bench}: licensed TSV missing or not the manifest's recorded file")
    frame = pd.read_csv(tsv, sep="\t", usecols=["index", *cols], dtype=str, keep_default_na=False)
    if len(frame) != len(dmi.rows[bench]):
        raise IngestError(f"{bench}: licensed TSV has {len(frame)} rows, the manifest {len(dmi.rows[bench])}")
    family_of: dict[str, tuple[str, ...]] = {}
    members: dict[tuple[str, ...], set[str]] = {}
    for k in dmi.rows[bench]:
        row = frame.iloc[k[0]]
        if str(row["index"]) != k[1]:
            raise IngestError(f"{bench}: TSV row {k[0]} index {row['index']!r} != manifest index {k[1]}")
        fam = tuple(str(row[c]) for c in cols)
        ck = content_key(bench, k[0], k[1], k[2], k[3])
        family_of[ck] = fam
        members.setdefault(fam, set()).add(ck)
    touched = {family_of[ck] for ck in keys}
    incomplete = sorted(f for f in touched if not members[f] <= keys)
    return {"columns": cols, "families_touched": len(touched), "families_incomplete": len(incomplete),
            "examples": [list(f) for f in incomplete[:3]]}


def prompt_conformance(local: pd.DataFrame, reference: pd.DataFrame) -> dict[str, Any]:
    """Row-level agreement of a local run's rendered inputs with its archived reference (handoff 4.4).

    Compared on every local row, joined one-to-one on (benchmark_id, content_key): ``prompt_sha256`` (the message
    payload), the model-neutral request digest (prompt + ordered media + decoding settings minus model_id/seed) and
    ``prompt_tokens`` (the rendered token count, which differs when another chat template or system prompt is
    applied). A local row without a reference row, or with unknown prompt tokens, does not conform.
    """

    cols = ["benchmark_id", "content_key", "prompt_sha256", "prompt_tokens", "ordered_media_sha256",
            "generation_settings_json"]
    cache: dict[tuple[Any, ...], str] = {}

    def neutral(frame: pd.DataFrame) -> list[str]:
        out = []
        for p, m, s in zip(frame["prompt_sha256"], frame["ordered_media_sha256"], frame["generation_settings_json"]):
            key = (p, tuple(_as_str_list(m)), s)
            if key not in cache:
                cache[key] = neutral_request_sha256(p, key[1], s)
            out.append(cache[key])
        return out

    loc = local[cols].copy()
    ref = reference[cols].copy()
    loc["neutral"] = neutral(loc)
    ref["neutral"] = neutral(ref)
    m = loc.merge(ref, on=["benchmark_id", "content_key"], how="left", suffixes=("", "_ref"), indicator=True,
                  validate="one_to_one")
    per: dict[str, Any] = {}
    for bench, g in m.groupby("benchmark_id", sort=True):
        hit = g["_merge"] == "both"
        pt, rt = pd.to_numeric(g["prompt_tokens"]), pd.to_numeric(g["prompt_tokens_ref"])
        known = hit & pt.notna() & rt.notna()
        per[str(bench)] = {
            "rows": int(len(g)), "reference_rows_matched": int(hit.sum()),
            "prompt_sha256_equal": int((hit & (g["prompt_sha256"] == g["prompt_sha256_ref"])).sum()),
            "neutral_request_sha256_equal": int((hit & (g["neutral"] == g["neutral_ref"])).sum()),
            "prompt_tokens_known_both": int(known.sum()),
            "prompt_tokens_equal": int((known & (pt == rt)).sum()),
            "prompt_tokens_median_local": float(pt.median()) if pt.notna().any() else None,
            "prompt_tokens_median_reference": float(rt.median()) if rt.notna().any() else None,
        }
        e = per[str(bench)]
        e["conforms"] = all(e[k] == e["rows"] for k in ("reference_rows_matched", "prompt_sha256_equal",
                                                        "neutral_request_sha256_equal", "prompt_tokens_equal"))
    fields = ("rows", "reference_rows_matched", "prompt_sha256_equal", "neutral_request_sha256_equal",
              "prompt_tokens_known_both", "prompt_tokens_equal")
    total = {k: int(sum(v[k] for v in per.values())) for k in fields}
    return {"total": total, "conforms": bool(per) and all(v["conforms"] for v in per.values()),
            "nonconforming_benchmarks": sorted(b for b, v in per.items() if not v["conforms"]), "per_benchmark": per}


CONFORMANCE_COLUMNS = ["benchmark_id", "content_key", "prompt_sha256", "prompt_tokens", "ordered_media_sha256",
                       "generation_settings_json"]


def conformance_reference_label(label: str, protocol_id: str) -> str:
    return label if protocol_id == "P0_native" else CONFORMANCE_REFERENCE[protocol_id]


def archive_reference_settings(cfg: StudyConfig) -> str:
    """The archive's single normalized generation setting (local runs must equal it); needs archive ingestion first."""

    path = cfg.work("audit", "archive_ingest.json")
    if not path.is_file():
        raise IngestError("the archive must be ingested before local runs (audit/archive_ingest.json is missing): "
                          "local runs are checked against its decoding settings and rendered inputs")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "verified":
        raise IngestError("archive ingestion is not verified; local runs cannot be checked against it")
    values = {normalized_settings_json(s) for s in report.get("generation_settings_variants") or {}}
    if len(values) != 1:
        raise IngestError(f"the archive carries {len(values)} normalized generation settings, expected exactly 1")
    return values.pop()


def ingest_local_run(cfg: StudyConfig, run_dir: Path | str, model_label: str, protocol_id: str | None, seed: int,
                     source: str, *, public_model_id: str | None = None, run_id: str | None = None,
                     model_role: str | None = None, strict_roles: bool = True,
                     judge_source_model_id: str = DEFAULT_JUDGE_SOURCE_SLUG,
                     manifest_index: DatasetManifestIndex | None = None, log: Log = print) -> pd.DataFrame:
    """Ingest a protocol run directory (``hf_archive`` spool, possibly a row subset) into the normalized schema.

    ``protocol_id`` may be None (derived from the label) and must otherwise equal the label's protocol.
    ``public_model_id`` is the model id written into the public generation settings (and therefore into
    ``request_sha256``); it defaults to the archived id for B/T/V so that a rerun's requests are directly comparable to
    the archive's, and to a slug of the label otherwise. ``strict_roles=False`` (a format example from another study,
    never written) skips only the label, served-model, frozen-split, family and conformance checks and requires an
    explicit ``protocol_id``. The returned frame carries ``attrs['official_aggregates']``, ``attrs['missingness']``
    and ``attrs['ingest_report']``; persist them with :func:`write_local_run`.
    """

    run_dir = Path(run_dir)
    if source not in LOCAL_SOURCES:
        raise IngestError(f"local source must be one of {LOCAL_SOURCES}, got {source!r}")
    if strict_roles:
        derived = protocol_for_label(model_label)
        if protocol_id is not None and protocol_id != derived:
            raise IngestError(f"model label {model_label!r} belongs to {derived}, not {protocol_id}")
        protocol_id = derived
        if model_label not in study_labels(cfg):
            raise IngestError(f"model label {model_label!r} is not declared by the study (archive labels and the "
                              f"evaluation matrix)")
        role = role_of_label(model_label)
        if model_role is not None and model_role != role:
            raise IngestError(f"model_role {model_role!r} contradicts label {model_label!r}")
    else:
        if protocol_id is None:
            raise IngestError("an unregistered role needs an explicit protocol_id")
        role = model_role or role_of_label(model_label)
    if protocol_id not in PROTOCOL_IDS:
        raise IngestError(f"unknown protocol_id {protocol_id!r}")
    seed = int(seed)
    spool = _load_spool(run_dir, seed)
    provenance = spool_provenance(cfg, spool)
    reference_settings = archive_reference_settings(cfg)
    if configured_settings_problems(cfg, reference_settings):
        raise IngestError("the archive's generation settings differ from the configured protocol")
    plan = load_run_plan(run_dir, seed)
    suite_keys = cfg.benchmark_keys()
    present = sorted({b for b, _ in spool})
    unknown = sorted(set(present) - set(suite_keys))
    if unknown:
        raise IngestError(f"spool holds benchmarks outside the suite: {unknown}")
    planned_names = [b for b in suite_keys if int(plan["expected_rows"].get(b, 0) or 0) > 0]
    missing_bench = sorted(set(planned_names) - set(present))
    extra_bench = sorted(set(present) - set(planned_names))
    if missing_bench or extra_bench:
        raise IngestError(f"run does not hold exactly the planned benchmarks: missing {missing_bench}, "
                          f"unplanned {extra_bench}")
    selected = planned_names
    incomplete = [b for b in selected if any((b, s) not in spool for s in SPOOL_STAGES.values())]
    if incomplete:
        raise IngestError(f"run is incomplete (missing generation/extraction/score slices) for {incomplete}")
    dmi = manifest_index or load_dataset_manifest_index(cfg.dataset_manifest, selected)
    planned, plan_info = planned_content_keys(cfg, run_dir, plan, dmi, strict=strict_roles)
    summary = load_campaign_summary(run_dir, seed, plan)
    if summary is None and strict_roles:
        raise IngestError(f"campaign summary summary.seed{seed}.json not found: each aggregate is checked against it")
    exp = spool_exporter()
    first = spool[(selected[0], "generation")][0]["identity"]
    served = verify_served_model(cfg, model_label, role, protocol_id, first) if strict_roles else {
        "model_revision": str(first["model_revision"]), "runtime_revision": str(first["model_revision"]),
        "served_view": {"kind": "not_checked_unregistered_role"}}
    if public_model_id is None:
        public_model_id = cfg.role(role)["archived_model_id"] if role in ROLES and protocol_id == "P0_native" \
            else _slugify(model_label)
    run_id = run_id or f"{first['run_id']}/{first['model_slug']}"
    repo_id = cfg.role(role)["repo_id"] if role in ROLES else "local/derived"
    model = exp.ModelMapping(source_model_id=str(first["model_slug"]), source_revision=str(first["model_revision"]),
                             model_id=public_model_id, model_revision=str(first["model_revision"]),
                             display_name=model_label, repository_id=repo_id,
                             repository_revision=str(first["model_revision"]))
    judge_cfg = cfg.raw["protocol"]["judge"]
    judge = exp.JudgeMapping(source_model_id=judge_source_model_id, model_id=judge_cfg["repo_id"],
                             model_revision=judge_cfg["revision"])
    plan_ns = types.SimpleNamespace(run_id=run_id, judge=judge)
    frames: list[pa.Table] = []
    aggregates: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    slice_reports: list[dict[str, Any]] = []
    problems: list[str] = []
    warnings: list[str] = list(plan_info.get("warnings", []))
    settings_seen: set[str] = set()
    full_split = plan_info.get("mode") == "full_split"
    for bench in selected:
        descs = {c: spool[(bench, s)] for c, s in SPOOL_STAGES.items()}
        label = f"{model_label}/seed={seed}/{bench}"
        repairs: dict[str, int] = {}
        try:
            for config_name in ("extractions", "scores"):
                desc, records = descs[config_name]
                fixed, repairs[config_name] = repair_subset_stage_ordinals(descs["responses"][1], records,
                                                                           desc["identity"])
                descs[config_name] = (desc, fixed)
        except IngestError as error:
            problems.append(f"{label}: {error}")
            continue

        def _src(config_name: str) -> types.SimpleNamespace:
            desc, records = descs[config_name]
            return types.SimpleNamespace(config_name=config_name, manifest=desc, model=model, records=records)

        try:
            resp_rows, contexts, _settings = exp._response_rows(_src("responses"), plan_ns)
            ext_rows = exp._extraction_rows(_src("extractions"), contexts, judge)
            score_rows, _scope, _agg = exp._score_rows(_src("scores"), contexts)
        except exp.PublicExportError as error:
            problems.append(f"{label}: exporter rejected the spool ({error})")
            continue
        parts = {c: str(d[0]["descriptor_id"]) for c, d in descs.items()}
        sl = PublicSlice(benchmark_id=bench, model_id=public_model_id, model_revision=str(first["model_revision"]),
                         run_id=run_id, seed=seed, responses=pd.DataFrame(resp_rows),
                         extractions=pd.DataFrame(ext_rows), scores=pd.DataFrame(score_rows), parts=parts,
                         source_manifest_sha256=parts["responses"])
        try:
            res = validate_public_slice(sl, dmi, allow_aliases=False)
        except IngestError as error:
            problems.append(str(error))
            continue
        # decoding contract: the archive's normalized settings, which equal the configured protocol
        for s in res.stats["generation_settings_variants"]:
            settings_seen.add(s)
            if normalized_settings_json(s) != reference_settings:
                problems.append(f"{label}: generation settings differ from the archive/protocol "
                                f"({configured_settings_problems(cfg, normalized_settings_json(s)) or 'media fields'})")
        expected_keys = planned[bench]
        if res.keys != expected_keys:
            problems.append(f"{label}: covers {len(res.keys & expected_keys)}/{len(expected_keys)} planned rows "
                            f"(missing {len(expected_keys - res.keys)}, unplanned {len(res.keys - expected_keys)})")
        families = None if full_split else family_completeness(cfg, dmi, bench, res.keys)
        if families and families["families_incomplete"]:
            msg = f"{label}: {families['families_incomplete']} problem families are only partly in the run"
            (problems if strict_roles else warnings).append(msg)
        agg = res.aggregate
        status, summary_value, summary_problem = ("not_checked", None, None) if summary is None else \
            summary_check(agg, summary.get(bench))
        if summary_problem:
            problems.append(f"{label}: {summary_problem}")
        elif status == "summary_rescaled":
            warnings.append(f"{label}: campaign summary reports {summary_value} for the subset; the spool and the "
                            f"official evaluator report {agg['official_value']}")
        scorer_out: dict[str, tuple[str | None, str]] = {}
        for rec in descs["scores"][1]:
            ctx = contexts.get(rec.get("record_id"))
            if ctx is not None:
                scorer_out[ctx.public_row["sample_id"]] = sanitize_scorer_output(rec)
        frame = res.frame
        frame["row_scorer"] = [scorer_out.get(s, (None, None))[0] for s in frame["sample_id"]]
        frame["row_scorer_output_json"] = [scorer_out.get(s, (None, None))[1] for s in frame["sample_id"]]
        frames.append(to_table(finalize_frame(frame, study_id=cfg.study_id, protocol_id=protocol_id,
                                              model_role=role, model_label=model_label,
                                              model_revision=served["model_revision"],
                                              runtime_revision=served["runtime_revision"], source=source)))
        complete_split = res.keys == dmi.expected_content_keys(bench)
        missing_rows.append(_missingness_row(
            source=source, run_name=f"{model_label}_seed{seed}_{source}", protocol_id=protocol_id, label=model_label,
            seed=seed, benchmark=bench, expected_keys=set(expected_keys), keys=res.keys, stats=res.stats,
            status=("complete" if complete_split else "subset_complete") if res.keys == expected_keys
            else "incomplete"))
        aggregates.append({
            "protocol_id": protocol_id, "model_label": model_label, "model_role": role, "seed": seed,
            "benchmark_id": bench, "official_value": agg["official_value"], "evaluated_rows": agg["evaluated_rows"],
            "scorer_id": agg["scorer_id"], "metric_id": agg["metric_id"], "source": source, "run_id": run_id,
            "model_id": public_model_id, "score_kind": res.stats["score_kind"], "results_file_value": None,
            "campaign_summary_value": summary_value, "campaign_summary_status": status})
        slice_reports.append({"benchmark_id": bench, "rows": res.stats["rows"], "score_kind": res.stats["score_kind"],
                              "parts": parts, "dataset_sha256": res.stats["dataset_sha256"],
                              "subset_stage_ordinals_reattached": repairs, "campaign_summary": status,
                              "problem_families": families,
                              "row_scorer_outputs": int(sum(o is not None for o in frame["row_scorer_output_json"]))})
        log(f"[ingest-local] {label}: {len(frame)} rows ({res.stats['score_kind']})")
    if problems:
        raise IngestError(f"local run {model_label} seed {seed} failed with {len(problems)} problem(s): "
                          + " | ".join(problems[:8]))
    out = pa.concat_tables(frames).to_pandas()   # schema-conformed per benchmark; consistent dtypes
    conformance: dict[str, Any] = {"status": "not_checked_unregistered_role"}
    if strict_roles:
        ref_label = conformance_reference_label(model_label, protocol_id)
        reference = load_aligned(cfg, columns=CONFORMANCE_COLUMNS, labels=[ref_label], seeds=[seed],
                                 benchmarks=selected)
        conformance = prompt_conformance(out, reference)
        conformance["reference"] = {"source": "archive", "model_label": ref_label, "seed": seed}
        del reference
        if not conformance["conforms"]:
            raise IngestError(f"local run {model_label} seed {seed}: rendered inputs do not conform to archived "
                              f"{ref_label} on {conformance['nonconforming_benchmarks'][:6]} "
                              f"(totals {conformance['total']})")
    out.attrs["official_aggregates"] = aggregates
    out.attrs["missingness"] = missing_rows
    out.attrs["ingest_report"] = {
        "model_label": model_label, "model_role": role, "protocol_id": protocol_id, "seed": seed, "source": source,
        "run_id": run_id, "public_model_id": public_model_id, "spool_identity": {
            k: first[k] for k in ("run_id", "model", "model_slug", "model_revision", "dataset_revision")},
        "spool_provenance": provenance, "served_model": served, "model_revision": served["model_revision"],
        "runtime_revision": served["runtime_revision"], "generation_settings_variants": sorted(settings_seen),
        "normalized_generation_settings": reference_settings, "run_dir_name": run_dir.name, "plan": plan_info,
        "rows": int(len(out)), "benchmarks": selected, "slices": slice_reports, "dataset_manifest_sha256": dmi.sha256,
        "prompt_conformance": conformance, "warnings": warnings,
        "protocol_code": protocol_code_provenance(cfg), "status": "verified"}
    return out


def write_local_run(cfg: StudyConfig, df: pd.DataFrame, name: str) -> dict[str, Path]:
    """Persist an ingested local run (predictions, aggregates, report) and register it in aligned/index.json."""

    if not re.fullmatch(r"[A-Za-z0-9@._-]+", name):
        raise IngestError(f"unsafe local run name {name!r}")
    out_dir = cfg.work("aligned", "local_runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = out_dir / f"{name}.parquet"
    tmp = out_dir / f".{name}.parquet.tmp.{os.getpid()}"
    pq.write_table(to_table(df), tmp, compression="zstd")
    os.replace(tmp, pred)
    aggs = out_dir / f"{name}.aggregates.parquet"
    pq.write_table(pa.Table.from_pylist(df.attrs.get("official_aggregates", []), schema=AGGREGATE_SCHEMA), aggs,
                   compression="zstd")
    report = dict(df.attrs.get("ingest_report", {}))
    report["name"] = name
    rpath = out_dir / f"{name}.report.json"
    write_json(rpath, report)
    missing = [dict(r, run_name=name) for r in df.attrs.get("missingness", [])]
    if missing:
        _write_missingness(cfg, missing, replace_run=name)

    def _upd(index: dict[str, Any]) -> None:
        index["local_runs"][name] = {
            "path": f"local_runs/{name}.parquet", "rows": int(len(df)), "sha256": sha256_file(pred),
            "aggregates": f"local_runs/{name}.aggregates.parquet", "report": f"local_runs/{name}.report.json",
            "model_label": report.get("model_label"), "seed": report.get("seed"), "source": report.get("source"),
            "protocol_id": report.get("protocol_id"), "run_id": report.get("run_id")}

    index_path = _update_index(cfg, _upd)
    return {"predictions": pred, "aggregates": aggs, "report": rpath, "index": index_path}


def compare_identity_with_archive(cfg: StudyConfig, local: pd.DataFrame, label: str, seed: int) -> dict[str, Any]:
    """Per-benchmark key-set and identity agreement between a local run and the archived slice of the same label/seed."""

    cols = ["benchmark_id", "content_key", "media_set_sha256", "dataset_sha256", "prompt_sha256", "request_sha256",
            "generation_settings_json"]
    arch = load_aligned(cfg, columns=cols, labels=[label], seeds=[seed])
    loc = local[cols]
    out: dict[str, Any] = {"label": label, "seed": seed, "benchmarks": {}}
    all_equal = True
    for bench, a in arch.groupby("benchmark_id", sort=False):
        lb = loc[loc["benchmark_id"] == bench]
        ak, lk = set(a["content_key"]), set(lb["content_key"])
        m = a.merge(lb, on=["benchmark_id", "content_key"], suffixes=("_a", "_l"))
        entry = {
            "archive_rows": len(ak), "local_rows": len(lk), "shared": len(ak & lk), "archive_only": len(ak - lk),
            "local_only": len(lk - ak),
            "media_set_equal": int((m["media_set_sha256_a"] == m["media_set_sha256_l"]).sum()),
            "prompt_sha256_equal": int((m["prompt_sha256_a"] == m["prompt_sha256_l"]).sum()),
            "request_sha256_equal": int((m["request_sha256_a"] == m["request_sha256_l"]).sum()),
            "generation_settings_equal": int((m["generation_settings_json_a"] == m["generation_settings_json_l"]).sum()),
            "dataset_sha256": {"archive": sorted(set(a["dataset_sha256"])), "local": sorted(set(lb["dataset_sha256"]))},
        }
        entry["identical_key_set"] = entry["archive_only"] == 0 and entry["local_only"] == 0
        all_equal &= entry["identical_key_set"] and entry["media_set_equal"] == entry["shared"]
        out["benchmarks"][bench] = entry
    extra = sorted(set(loc["benchmark_id"]) - set(arch["benchmark_id"]))
    out["local_benchmarks_not_in_archive"] = extra
    out["total"] = {k: int(sum(v[k] for v in out["benchmarks"].values()))
                    for k in ("archive_rows", "local_rows", "shared", "archive_only", "local_only", "media_set_equal",
                              "prompt_sha256_equal", "request_sha256_equal", "generation_settings_equal")}
    out["identical_identities"] = bool(all_equal and not extra)
    return out


# --------------------------------------------------------------------------------------------- reading
def aligned_index(cfg: StudyConfig) -> dict[str, Any]:
    path = cfg.work("aligned", "index.json")
    if not path.is_file():
        raise IngestError("aligned/index.json not found; run archive ingestion first")
    return json.loads(path.read_text(encoding="utf-8"))


def load_aligned(cfg: StudyConfig, *, columns: Sequence[str] | None = None, labels: Sequence[str] | None = None,
                 seeds: Sequence[int] | None = None, benchmarks: Sequence[str] | None = None,
                 include_archive: bool = True, local_runs: Sequence[str] | None = None) -> pd.DataFrame:
    """Column- and slice-selective reader for the aligned tables (bounded memory on the login node).

    The shared interface has one row per model_label x seed x content row *within a run*; local reruns keep the
    archive labels B/T, so a read that would return two runs of the same (model_label, seed) is refused unless
    ``source`` and ``run_id`` are among the selected columns (no silent duplicate rows).
    """

    filters = []
    if labels:
        filters.append(("model_label", "in", list(labels)))
    if seeds:
        filters.append(("seed", "in", [int(s) for s in seeds]))
    if benchmarks:
        filters.append(("benchmark_id", "in", list(benchmarks)))
    paths: list[Path] = []
    if include_archive:
        paths.append(cfg.work("aligned", "predictions.parquet"))
    if local_runs:
        index = aligned_index(cfg)
        for name in local_runs:
            entry = index["local_runs"].get(name)
            if entry is None:
                raise IngestError(f"local run {name!r} is not registered in aligned/index.json")
            paths.append(cfg.work("aligned", entry["path"]))
    for path in paths:
        if not path.is_file():
            raise IngestError(f"aligned table not found: {path.name}")
    if len(paths) > 1 and columns is not None and not {"source", "run_id"} <= set(columns):
        seen: dict[tuple[str, int], str] = {}
        for path in paths:
            pairs = pq.read_table(path, columns=["model_label", "seed"], filters=filters or None).to_pandas()
            for label, seed in set(zip(pairs["model_label"], pairs["seed"])):
                other = seen.setdefault((str(label), int(seed)), path.name)
                if other != path.name:
                    raise IngestError(f"{label}/seed={seed} is in both {other} and {path.name}; select 'source' and "
                                      f"'run_id' to tell the runs apart")
    frames = []
    for path in paths:
        table = pq.read_table(path, columns=list(columns) if columns else None, filters=filters or None)
        frames.append(table.to_pandas())
    if not frames:
        return pd.DataFrame(columns=list(columns or NORMALIZED_COLUMNS))
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def load_aggregates(cfg: StudyConfig, *, local_runs: Sequence[str] | None = None) -> pd.DataFrame:
    frames = [pq.read_table(cfg.work("aligned", "official_aggregates.parquet")).to_pandas()]
    if local_runs:
        index = aligned_index(cfg)
        for name in local_runs:
            frames.append(pq.read_table(cfg.work("aligned", index["local_runs"][name]["aggregates"])).to_pandas())
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------------------------- command line
EXISTING_RERUNS = {"B": "qwen2.5-vl-7b-instruct/full", "T": "trace-qwen2.5-vl-7b/full"}


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m trace_merge.ingest {archive|local-run|existing-reruns}`` (paths from CLI/env, never invented)."""

    import argparse

    from .config import ensure_work_tree, load_config

    p = argparse.ArgumentParser(prog="trace_merge.ingest", description=__doc__.split("\n\n")[0])
    p.add_argument("command", choices=["archive", "local-run", "existing-reruns"])
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--work-root", type=Path, default=None)
    p.add_argument("--protocol-clone", type=Path, default=None)
    p.add_argument("--runs-root", type=Path, default=None, help="directory holding <slug>/<subset> local run dirs")
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--protocol-id", default=None, help="optional; must equal the label's protocol (derived if omitted)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--source", default="local_run")
    p.add_argument("--name", default=None)
    p.add_argument("--public-model-id", default=None)
    p.add_argument("--producer-alias", action="append", default=None,
                   help="archive: enable a verified producer row-hash alias (e.g. countqa); needs a DESIGN.md "
                        "amendment, overrides the config's identity.producer_row_hash_aliases")
    p.add_argument("--no-write", action="store_true", help="validate only (e.g. a format example run)")
    p.add_argument("--allow-unregistered-role", action="store_true",
                   help="validate a run of a model outside the study (format example); needs --no-write and "
                        "--protocol-id")
    args = p.parse_args(argv)
    cfg = load_config(args.config, work_root=args.work_root, protocol_clone=args.protocol_clone)
    ensure_work_tree(cfg)
    if args.command == "archive":
        paths = ingest_archive(cfg, producer_aliases=args.producer_alias)
        print(json.dumps({k: str(v.relative_to(cfg.require_work_root())) for k, v in paths.items()}, indent=1))
        return 0
    if args.command == "local-run":
        if args.run_dir is None or args.label is None:
            p.error("local-run needs --run-dir and --label")
        if args.allow_unregistered_role and not (args.no_write and args.protocol_id):
            p.error("--allow-unregistered-role needs --no-write and an explicit --protocol-id")
        df = ingest_local_run(cfg, args.run_dir, args.label, args.protocol_id, args.seed, args.source,
                              public_model_id=args.public_model_id,
                              model_role="example" if args.allow_unregistered_role else None,
                              strict_roles=not args.allow_unregistered_role)
        report = df.attrs["ingest_report"]
        summary = {"rows": len(df), "benchmarks": int(df["benchmark_id"].nunique()),
                   "protocol_id": report["protocol_id"], "score_kinds": dict(Counter(
                       df.drop_duplicates("benchmark_id")["score_kind"])),
                   "prompt_conformance": report["prompt_conformance"].get("total",
                                                                          report["prompt_conformance"].get("status")),
                   "warnings": report["warnings"]}
        if not args.no_write:
            name = args.name or f"{args.label.replace('@', '_at_')}_seed{args.seed}_{args.source}"
            write_local_run(cfg, df, name)
            summary["name"] = name
        print(json.dumps(summary, indent=1))
        return 0
    if args.runs_root is None:
        p.error("existing-reruns needs --runs-root")
    results = {}
    for label, rel in EXISTING_RERUNS.items():
        df = ingest_local_run(cfg, args.runs_root / rel, label, None, 42, "existing_local_rerun")
        name = f"{label}_seed42_existing_local_rerun"
        write_local_run(cfg, df, name)
        ident = compare_identity_with_archive(cfg, df, label, 42)
        write_json(cfg.work("audit", f"identity_{name}.json"), ident)
        results[name] = {"rows": len(df), "identical_identities": ident["identical_identities"], **ident["total"],
                         "prompt_conformance": df.attrs["ingest_report"]["prompt_conformance"]["total"],
                         "served_view": {k: df.attrs["ingest_report"]["served_model"]["served_view"].get(k)
                                         for k in ("slug", "immutable_revision", "content_revision", "file_count")}}
        del df
    print(json.dumps(results, indent=1))
    return 0 if all(r["identical_identities"] and r["rows"] == 32805 for r in results.values()) else 4


if __name__ == "__main__":
    sys.exit(main())
