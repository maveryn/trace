#!/usr/bin/env python3
"""Fail-closed entrypoint for the two TRACE RLVR training runs."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RLVR_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = RLVR_ROOT / "configs"
BACKEND_ROOT = RLVR_ROOT / "easyr1_backend"
SOURCE_ROOT = REPO_ROOT / "src"
PROMPT_PATH = (
    RLVR_ROOT / "examples" / "prompts" / "trace_vero_json_system_prompt_answer.txt"
)
REWARD_PATH = BACKEND_ROOT / "examples" / "reward_function" / "trace_rlvr.py"
MERGER_PATH = BACKEND_ROOT / "scripts" / "model_merger.py"
INPUT_RECEIPT_NAME = "input_receipt.json"
RUN_RECEIPT_NAME = "run_receipt.json"
RELEASE_FILES_PATH = RLVR_ROOT / "release_files.v1.json"
VENDOR_MANIFEST_PATH = RLVR_ROOT / "easyr1_backend" / "VENDOR_MANIFEST.v1.json"

CONFIG_IDS = ("trace-qwen2.5-vl-3b", "trace-qwen2.5-vl-7b")
PROMPT_SHA256 = "f394927d9abcfb7a1e43ef48a30c29b8c70e6facdbda314d7b27c59d8c3ae900"
DATASET_REPOSITORY = "maveryn/trace"
HISTORICAL_DATASET_REVISION = "e317b746b258630682367cc6a9d87dedd195113c"
COMPARED_DATASET_REVISION = "78f09b5482abc8e447a0a722cdf39e7d32f483c8"
DATASET_REVISION = "4e5b54361360296a855542b40cfd8b7f81b355fe"
DATASET_RELEASE_TAG = "dataset-v1"
DATASET_EQUIVALENCE_RECEIPT_RELATIVE_PATH = "rlvr/dataset_equivalence.v1.json"
DATASET_EQUIVALENCE_RECEIPT_PATH = REPO_ROOT / DATASET_EQUIVALENCE_RECEIPT_RELATIVE_PATH
DATASET_EQUIVALENCE_RECEIPT_SHA256 = (
    "40afcb59e0c67d7b7f47b78bb673a6c152320eb0656a990910ff52c634d43ae7"
)
DATASET_ADVISORY_COLUMN = "trace_supervision_mode"
DATASET_RELEASE_IDENTITY = {
    "repository_id": DATASET_REPOSITORY,
    "historical_training_revision": HISTORICAL_DATASET_REVISION,
    "reproduction_dataset_revision": DATASET_REVISION,
    "release_tag": DATASET_RELEASE_TAG,
    "equivalence_receipt": {
        "path": DATASET_EQUIVALENCE_RECEIPT_RELATIVE_PATH,
        "sha256": DATASET_EQUIVALENCE_RECEIPT_SHA256,
    },
}
REQUIRED_DATASET_COLUMNS = [
    "images",
    "image_sizes",
    "prompt_answer",
    "prompt_answer_and_annotation",
    "answer_gt",
    "annotation_gt",
    "reward_contract",
    "instance_id",
    "domain",
    "task",
    "scene_id",
    "query_id",
    "scene_variant",
    "trace_ref",
]
BASE_MODELS = {
    "trace-qwen2.5-vl-3b": (
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "66285546d2b821cf421d4f5eb2576359d3770cd3",
    ),
    "trace-qwen2.5-vl-7b": (
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "cc594898137f460bfe9f0759e9844b3ce807cfb5",
    ),
}
BASE_MODEL_FILES = {
    "trace-qwen2.5-vl-3b": [
        "chat_template.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ],
    "trace-qwen2.5-vl-7b": [
        "chat_template.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model-00001-of-00005.safetensors",
        "model-00002-of-00005.safetensors",
        "model-00003-of-00005.safetensors",
        "model-00004-of-00005.safetensors",
        "model-00005-of-00005.safetensors",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ],
}
RELEASED_MODELS = {
    "trace-qwen2.5-vl-3b": (
        "maveryn/trace-qwen2.5-vl-3b",
        "2ec2374d5c219e6b12e26bda93d3b3adeb1e30c5",
    ),
    "trace-qwen2.5-vl-7b": (
        "maveryn/trace-qwen2.5-vl-7b",
        "4d0f1ae8ee25022058090dbdbff61957ece7331d",
    ),
}
SOURCE_RECEIPTS = {
    "trace-qwen2.5-vl-3b": {
        "source_revision": "847e9f5279f8111fdbfef1c8b8631fc621c23456",
        "source_config_sha256": "2d8a1461468c5950fe2a4daafa9df524d3f5c64bf1a1cff3201f9931f9bc8b35",
        "source_environment_sha256": "798ce596e11a5266591ffe01f0ca0cf29d6968308784cab476960fc1de0ecca8",
    },
    "trace-qwen2.5-vl-7b": {
        "source_revision": "d29b23f6085764ea831adb5edc5a23f89b0d98f3",
        "source_config_sha256": "19dea84aa0fb748566f42974819f331ef58fde8738f0b18d8dc0876be9baab31",
        "source_environment_sha256": "132c380fac1fc11bea08ff130b978e997703b303f989e5b72aca4bcdfd5dd276",
    },
}


class ReleaseError(RuntimeError):
    """Raised when an RLVR training invariant is not satisfied."""


def _sha256_path(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"cannot read JSON file {path}: {exc}") from exc


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _config_path(config_id: str) -> Path:
    if config_id not in CONFIG_IDS:
        raise ReleaseError(f"unknown configuration {config_id!r}")
    return CONFIG_ROOT / f"{config_id}.yaml"


def _load_config(config_id: str) -> dict[str, Any]:
    path = _config_path(config_id)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ReleaseError(f"cannot read configuration {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReleaseError(f"configuration {config_id} must be an object")
    return payload


def _nested(payload: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ReleaseError(f"configuration is missing {dotted_path}")
        value = value[part]
    return value


def _expect(payload: Mapping[str, Any], dotted_path: str, expected: Any) -> None:
    actual = _nested(payload, dotted_path)
    if actual != expected:
        raise ReleaseError(
            f"{dotted_path} is frozen: expected {expected!r}, found {actual!r}"
        )


def _all_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


def _validate_config(config_id: str, config: Mapping[str, Any]) -> None:
    _expect(config, "schema_version", "trace-rlvr-training-config-v1")
    _expect(config, "config_id", config_id)
    _expect(config, "release.scope", "answer_only")
    for key, expected in SOURCE_RECEIPTS[config_id].items():
        _expect(config, f"release.{key}", expected)
    _expect(config, "release.selected_checkpoint_step", 500)
    base_repo, base_revision = BASE_MODELS[config_id]
    _expect(config, "release.base_model.repo_id", base_repo)
    _expect(config, "release.base_model.revision", base_revision)
    _expect(config, "release.base_model.required_files", BASE_MODEL_FILES[config_id])
    release_repo, release_revision = RELEASED_MODELS[config_id]
    _expect(config, "release.released_model.repo_id", release_repo)
    _expect(config, "release.released_model.revision", release_revision)

    frozen_values = {
        "release.dataset.repo_id": DATASET_REPOSITORY,
        "release.dataset.revision": DATASET_REVISION,
        "release.dataset.historical_training_revision": HISTORICAL_DATASET_REVISION,
        "release.dataset.equivalence_receipt_path": (
            DATASET_EQUIVALENCE_RECEIPT_RELATIVE_PATH
        ),
        "release.dataset.equivalence_receipt_sha256": (
            DATASET_EQUIVALENCE_RECEIPT_SHA256
        ),
        "release.dataset.train_rows": 64_000,
        "release.dataset.validation_rows": 2_000,
        "release.dataset.task_count": 1_000,
        "release.dataset.train_rows_per_task": 64,
        "release.dataset.validation_rows_per_task": 2,
        "release.dataset.validation_file": "data/validation/trace_rlvr_validation_iid_2000_all1000_seed1042.parquet",
        "release.dataset.train_manifest": "metadata/train/trace_rlvr_train_64000_all1000_seed42.parquet.manifest.json",
        "release.dataset.validation_manifest": "metadata/validation_iid/trace_rlvr_validation_iid_2000_all1000_seed1042.parquet.manifest.json",
        "release.prompt.path": "rlvr/examples/prompts/trace_vero_json_system_prompt_answer.txt",
        "release.prompt.sha256": PROMPT_SHA256,
        "release.reference_hardware.gpu_count": 8,
        "release.reference_hardware.cuda": "12.8",
        "easyr1.data.prompt_key": "prompt_answer",
        "easyr1.data.answer_key": "answer_gt",
        "easyr1.data.rollout_batch_size": 128,
        "easyr1.data.val_batch_size": 1024,
        "easyr1.data.max_prompt_length": 2048,
        "easyr1.data.max_response_length": 2048,
        "easyr1.data.shuffle": True,
        "easyr1.data.seed": 1,
        "easyr1.data.min_pixels": 262_144,
        "easyr1.data.max_pixels": 4_194_304,
        "easyr1.data.filter_overlong_prompts": False,
        "easyr1.data.train_files": None,
        "easyr1.data.val_files": None,
        "easyr1.data.system_prompt_file": None,
        "easyr1.algorithm.adv_estimator": "grpo",
        "easyr1.algorithm.disable_kl": True,
        "easyr1.algorithm.use_kl_loss": False,
        "easyr1.algorithm.kl_coef": 0.0,
        "easyr1.worker.actor.global_batch_size": 128,
        "easyr1.worker.actor.micro_batch_size_per_device_for_experience": 2,
        "easyr1.worker.actor.micro_batch_size_per_device_for_update": 1,
        "easyr1.worker.actor.max_grad_norm": 1.0,
        "easyr1.worker.actor.clip_ratio_low": 0.2,
        "easyr1.worker.actor.clip_ratio_high": 0.3,
        "easyr1.worker.actor.clip_ratio_dual": 3.0,
        "easyr1.worker.actor.loss_avg_mode": "token",
        "easyr1.worker.actor.loss_type": "default",
        "easyr1.worker.actor.optim.lr": 1.0e-6,
        "easyr1.worker.actor.optim.weight_decay": 0.01,
        "easyr1.worker.actor.optim.lr_scheduler_type": "constant",
        "easyr1.worker.actor.ppo_epochs": 1,
        "easyr1.worker.actor.padding_free": True,
        "easyr1.worker.actor.dynamic_batching": True,
        "easyr1.worker.actor.ulysses_size": 1,
        "easyr1.worker.actor.use_torch_compile": True,
        "easyr1.worker.actor.model.model_path": None,
        "easyr1.worker.actor.model.tokenizer_path": None,
        "easyr1.worker.actor.model.enable_gradient_checkpointing": True,
        "easyr1.worker.actor.model.trust_remote_code": False,
        "easyr1.worker.actor.model.freeze_vision_tower": False,
        "easyr1.worker.actor.model.lora.rank": 0,
        "easyr1.worker.actor.fsdp.enable_full_shard": True,
        "easyr1.worker.actor.fsdp.enable_cpu_offload": False,
        "easyr1.worker.actor.fsdp.enable_rank0_init": True,
        "easyr1.worker.actor.fsdp.mp_param_dtype": "bf16",
        "easyr1.worker.actor.fsdp.mp_reduce_dtype": "fp32",
        "easyr1.worker.actor.offload.offload_params": True,
        "easyr1.worker.actor.offload.offload_optimizer": True,
        "easyr1.worker.rollout.n": 8,
        "easyr1.worker.rollout.temperature": 1.0,
        "easyr1.worker.rollout.top_p": 1.0,
        "easyr1.worker.rollout.top_k": -1,
        "easyr1.worker.rollout.seed": 1,
        "easyr1.worker.rollout.dtype": "bf16",
        "easyr1.worker.rollout.tensor_parallel_size": 2,
        "easyr1.worker.rollout.gpu_memory_utilization": 0.6,
        "easyr1.worker.rollout.max_num_batched_tokens": 8192,
        "easyr1.worker.rollout.val_override_config.temperature": 0.6,
        "easyr1.worker.rollout.val_override_config.top_p": 0.95,
        "easyr1.worker.rollout.val_override_config.n": 1,
        "easyr1.worker.ref.fsdp.enable_cpu_offload": True,
        "easyr1.worker.ref.offload.offload_params": False,
        "easyr1.worker.reward.reward_function": None,
        "easyr1.worker.reward.reward_function_kwargs": {},
        "easyr1.trainer.total_epochs": 15,
        "easyr1.trainer.max_steps": 500,
        "easyr1.trainer.project_name": "trace_easyr1",
        "easyr1.trainer.experiment_name": config_id,
        "easyr1.trainer.logger": ["console"],
        "easyr1.trainer.nnodes": 1,
        "easyr1.trainer.n_gpus_per_node": 8,
        "easyr1.trainer.val_freq": 100,
        "easyr1.trainer.val_before_train": False,
        "easyr1.trainer.save_freq": 100,
        "easyr1.trainer.save_limit": 1,
        "easyr1.trainer.save_model_only": False,
        "easyr1.trainer.save_checkpoint_path": None,
        "easyr1.trainer.load_checkpoint_path": None,
        "easyr1.trainer.find_last_checkpoint": False,
    }
    for dotted_path, expected in frozen_values.items():
        _expect(config, dotted_path, expected)

    expected_train_files = [
        f"data/train/trace_rlvr_train_64000_all1000_seed42-{index:05d}-of-00016.parquet"
        for index in range(16)
    ]
    _expect(config, "release.dataset.train_files", expected_train_files)
    _expect(
        config,
        "release.dataset.required_columns",
        REQUIRED_DATASET_COLUMNS,
    )
    forbidden_keys = {
        key
        for key in _all_keys(_nested(config, "easyr1"))
        if "annotation" in key
        or "task_conditioned" in key
        or key == "trace_output_mode"
    }
    if forbidden_keys:
        raise ReleaseError(
            f"non-answer training configuration keys found: {sorted(forbidden_keys)}"
        )


def _normalized_profile(config: Mapping[str, Any]) -> dict[str, Any]:
    profile = copy.deepcopy(dict(config))
    profile.pop("config_id")
    release = profile["release"]
    for key in (
        "source_revision",
        "source_config_sha256",
        "source_environment_sha256",
        "released_model",
        "base_model",
    ):
        release.pop(key)
    profile["easyr1"]["trainer"].pop("experiment_name")
    return profile


def _validate_profile_pair() -> None:
    paths = sorted(path.name for path in CONFIG_ROOT.glob("*.yaml"))
    expected_paths = [f"{config_id}.yaml" for config_id in CONFIG_IDS]
    if paths != expected_paths:
        raise ReleaseError(
            f"expected exactly two canonical configs: {expected_paths}, found {paths}"
        )
    configs = {config_id: _load_config(config_id) for config_id in CONFIG_IDS}
    for config_id, config in configs.items():
        _validate_config(config_id, config)
    if _normalized_profile(configs[CONFIG_IDS[0]]) != _normalized_profile(
        configs[CONFIG_IDS[1]]
    ):
        raise ReleaseError(
            "3B and 7B profiles differ outside their approved identity fields"
        )


def _validate_vendor_manifest() -> None:
    manifest = _read_json(VENDOR_MANIFEST_PATH)
    if manifest.get("schema_version") != "trace-easyr1-vendor-manifest-v1":
        raise ReleaseError("unexpected EasyR1 vendor manifest schema")
    if manifest.get("upstream_revision") != "dd71bbd252694f5f850213eec15795b6b88d9fea":
        raise ReleaseError("unexpected EasyR1 upstream revision")
    if (
        manifest.get("internal_source_revision")
        != SOURCE_RECEIPTS[CONFIG_IDS[0]]["source_revision"]
    ):
        raise ReleaseError("unexpected EasyR1 source revision")
    entries = manifest.get("files")
    if not isinstance(entries, list) or len(entries) != 64:
        raise ReleaseError("EasyR1 vendor manifest must contain exactly 64 verl files")
    dispositions = Counter()
    expected_paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ReleaseError("invalid EasyR1 vendor manifest entry")
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative.startswith("verl/"):
            raise ReleaseError(f"invalid vendored path {relative!r}")
        expected_paths.add(relative)
        dispositions[entry.get("disposition")] += 1
        path = BACKEND_ROOT / relative
        if _sha256_path(path) != entry.get("public_sha256"):
            raise ReleaseError(f"vendored file hash mismatch: {relative}")
    actual_paths = {
        path.relative_to(BACKEND_ROOT).as_posix()
        for path in (BACKEND_ROOT / "verl").rglob("*.py")
        if "__pycache__" not in path.parts
    }
    if actual_paths != expected_paths:
        raise ReleaseError(
            "vendored EasyR1 file inventory differs from the expected 64-file unit"
        )
    if dispositions != Counter(
        {
            "upstream_exact": 57,
            "internal_runtime_patch": 3,
            "public_answer_only_adaptation": 4,
        }
    ):
        raise ReleaseError(
            f"unexpected EasyR1 vendor dispositions: {dict(dispositions)}"
        )


def _dataset_equivalence_file_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    dataset = _nested(config, "release.dataset")
    if (
        _sha256_path(DATASET_EQUIVALENCE_RECEIPT_PATH)
        != DATASET_EQUIVALENCE_RECEIPT_SHA256
    ):
        raise ReleaseError("dataset equivalence receipt hash mismatch")
    receipt = _read_json(DATASET_EQUIVALENCE_RECEIPT_PATH)
    expected_identity = {
        "receipt_version": "trace-dataset-equivalence-v1",
        "repository": DATASET_REPOSITORY,
        "original_training_revision": HISTORICAL_DATASET_REVISION,
        "compared_current_revision": COMPARED_DATASET_REVISION,
        "added_advisory_column": DATASET_ADVISORY_COLUMN,
        "compared_columns": REQUIRED_DATASET_COLUMNS,
    }
    for key, expected in expected_identity.items():
        if receipt.get(key) != expected:
            raise ReleaseError(f"dataset equivalence receipt {key} mismatch")
    expected_aggregate = {
        "current_differs_only_by_added_trace_supervision_mode": True,
        "existing_columns_value_equal": True,
        "existing_field_schemas_equal": True,
        "image_bytes_equal": True,
        "instance_ids_equal_and_unique": True,
        "outcome": "pass",
        "parquet_file_count": 17,
        "row_count": 66_000,
        "row_counts_equal": True,
        "row_order_equal": True,
        "schema_metadata_only_adds_trace_supervision_mode": True,
        "task_distributions_equal": True,
        "train_validation_instance_ids_disjoint": True,
    }
    if receipt.get("aggregate") != expected_aggregate:
        raise ReleaseError("dataset equivalence receipt aggregate outcome mismatch")
    original_schema = receipt.get("original_schema")
    current_schema = receipt.get("current_schema")
    if (
        not isinstance(original_schema, list)
        or not all(isinstance(field, dict) for field in original_schema)
        or [field.get("name") for field in original_schema] != REQUIRED_DATASET_COLUMNS
    ):
        raise ReleaseError("dataset equivalence receipt original schema mismatch")
    if (
        not isinstance(current_schema, list)
        or not all(isinstance(field, dict) for field in current_schema)
        or [field.get("name") for field in current_schema]
        != [*REQUIRED_DATASET_COLUMNS, DATASET_ADVISORY_COLUMN]
    ):
        raise ReleaseError("dataset equivalence receipt current schema mismatch")

    expected_paths = {*dataset["train_files"], dataset["validation_file"]}
    entries = receipt.get("parquet_files")
    if not isinstance(entries, list):
        raise ReleaseError("dataset equivalence receipt must contain parquet_files")
    expected_column_equality = {column: True for column in REQUIRED_DATASET_COLUMNS}
    hashes: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ReleaseError("invalid dataset equivalence receipt file entry")
        relative = entry["path"]
        if relative in hashes:
            raise ReleaseError(
                f"duplicate dataset equivalence receipt path: {relative}"
            )
        current = entry.get("current")
        expected_rows = 4_000 if relative in dataset["train_files"] else 2_000
        equality_checks = {
            "rows": expected_rows,
            "compared_columns_equal_in_row_order": True,
            "existing_field_schemas_equal": True,
            "image_bytes_equal_in_row_order": True,
            "schema_metadata_only_adds_trace_supervision_mode": True,
            "task_distribution_equal": True,
            "column_equality": expected_column_equality,
        }
        for key, expected in equality_checks.items():
            if entry.get(key) != expected:
                raise ReleaseError(
                    f"dataset equivalence receipt {relative} {key} mismatch"
                )
        if not isinstance(current, dict):
            raise ReleaseError(
                f"dataset equivalence receipt {relative} lacks current hash"
            )
        current_hash = current.get("sha256")
        if (
            not isinstance(current_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", current_hash) is None
            or current.get("lfs_sha256") != current_hash
            or current.get("lfs_oid_matches_sha256") is not True
        ):
            raise ReleaseError(
                f"dataset equivalence receipt {relative} current hash mismatch"
            )
        hashes[relative] = current_hash
    if set(hashes) != expected_paths:
        raise ReleaseError("dataset equivalence receipt Parquet inventory mismatch")
    return hashes


def _validate_release_files() -> None:
    manifest = _read_json(RELEASE_FILES_PATH)
    if manifest.get("schema_version") != "trace-rlvr-release-files-v1":
        raise ReleaseError("unexpected RLVR release-file manifest schema")
    if manifest.get("dataset_release") != DATASET_RELEASE_IDENTITY:
        raise ReleaseError("unexpected dataset release identity")
    if manifest.get("canonical_trace_validation") != {
        "evaluation_harness_revision": "b7e4bcf2bae88684a442834419d41d74c58e3eac",
        "results_sha256": "1e2b78d46dbbd606210653b584ed77dc436d16bbdc0d329c5aaf55718dc37e11",
        "source_revision": "cf0d14aed86db2661d397ce8b68b36171873478d",
        "suite_id": "trace-validation-2000-seed42-v1",
        "suite_sha256": "f9cccdcdddb6135c16d3a9d434f985b51e4105c07ff0c74a54a71a4dfe7c85c7",
    }:
        raise ReleaseError("unexpected TRACE validation release identity")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ReleaseError("RLVR release-file manifest must contain a files list")
    expected: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ReleaseError("invalid RLVR release-file entry")
        expected[entry["path"]] = entry.get("sha256")
    actual = {
        path.relative_to(RLVR_ROOT).as_posix()
        for path in RLVR_ROOT.rglob("*")
        if path.is_file()
        and path != RELEASE_FILES_PATH
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        raise ReleaseError(
            f"RLVR release-file inventory mismatch; missing={missing}, extra={extra}"
        )
    for relative, expected_hash in expected.items():
        if _sha256_path(RLVR_ROOT / relative) != expected_hash:
            raise ReleaseError(f"RLVR release-file hash mismatch: {relative}")


def check_release(config_id: str) -> dict[str, Any]:
    _validate_profile_pair()
    _validate_vendor_manifest()
    _validate_release_files()
    if _sha256_path(PROMPT_PATH) != PROMPT_SHA256:
        raise ReleaseError("answer prompt hash does not match the configured value")
    config = _load_config(config_id)
    _dataset_equivalence_file_hashes(config)
    return {
        "config_id": config_id,
        "config_sha256": _sha256_path(_config_path(config_id)),
        "dataset_revision": _nested(config, "release.dataset.revision"),
        "prompt_sha256": PROMPT_SHA256,
        "selected_checkpoint_step": 500,
        "status": "ok",
    }


def _ensure_empty_directory(path: Path, *, label: str) -> Path:
    if path.is_symlink():
        raise ReleaseError(f"{label} must not be a symlink: {path}")
    resolved = path.expanduser().resolve()
    if resolved.exists():
        if not resolved.is_dir():
            raise ReleaseError(f"{label} must be a directory: {resolved}")
        if any(resolved.iterdir()):
            raise ReleaseError(f"{label} must be empty: {resolved}")
    else:
        resolved.mkdir(parents=True)
    return resolved


def _file_record(path: Path, root: Path, **extra: Any) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size": path.stat().st_size,
        "sha256": _sha256_path(path),
        **extra,
    }


def _validate_dataset_files(
    config: Mapping[str, Any], dataset_root: Path
) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ReleaseError(
            "prepare requires pyarrow from rlvr/requirements-cu128.txt"
        ) from exc

    dataset = _nested(config, "release.dataset")
    expected_columns = [*dataset["required_columns"], DATASET_ADVISORY_COLUMN]
    expected_hashes = _dataset_equivalence_file_hashes(config)
    train_paths = [dataset_root / relative for relative in dataset["train_files"]]
    validation_path = dataset_root / dataset["validation_file"]
    records: list[dict[str, Any]] = []
    train_tasks: Counter[str] = Counter()
    train_rows = 0
    for path in train_paths:
        if not path.is_file():
            raise ReleaseError(f"missing frozen training shard: {path}")
        metadata = pq.read_metadata(path)
        if pq.read_schema(path).names != expected_columns:
            raise ReleaseError(
                f"training shard schema does not match the release: {path}"
            )
        train_rows += metadata.num_rows
        train_tasks.update(
            str(value)
            for value in pq.read_table(path, columns=["task"])["task"].to_pylist()
        )
        record = _file_record(path, dataset_root, rows=metadata.num_rows)
        if record["sha256"] != expected_hashes[record["path"]]:
            raise ReleaseError(
                f"training shard hash does not match the release: {path}"
            )
        records.append(record)
    if train_rows != dataset["train_rows"]:
        raise ReleaseError(
            f"expected {dataset['train_rows']} training rows, found {train_rows}"
        )
    if len(train_tasks) != dataset["task_count"] or set(train_tasks.values()) != {
        dataset["train_rows_per_task"]
    }:
        raise ReleaseError(
            "training task cardinality does not match the frozen 1000 x 64 contract"
        )

    if not validation_path.is_file():
        raise ReleaseError(f"missing frozen validation file: {validation_path}")
    validation_metadata = pq.read_metadata(validation_path)
    if pq.read_schema(validation_path).names != expected_columns:
        raise ReleaseError("validation shard schema does not match the release")
    validation_tasks = Counter(
        str(value)
        for value in pq.read_table(validation_path, columns=["task"])[
            "task"
        ].to_pylist()
    )
    if validation_metadata.num_rows != dataset["validation_rows"]:
        raise ReleaseError("validation row count does not match the frozen contract")
    if len(validation_tasks) != dataset["task_count"] or set(
        validation_tasks.values()
    ) != {dataset["validation_rows_per_task"]}:
        raise ReleaseError(
            "validation task cardinality does not match the frozen 1000 x 2 contract"
        )
    record = _file_record(
        validation_path, dataset_root, rows=validation_metadata.num_rows
    )
    if record["sha256"] != expected_hashes[record["path"]]:
        raise ReleaseError("validation shard hash does not match the release")
    records.append(record)

    for key in ("train_manifest", "validation_manifest"):
        path = dataset_root / dataset[key]
        payload = _read_json(path)
        if not isinstance(payload, dict) or not payload:
            raise ReleaseError(f"dataset manifest is empty: {path}")
        records.append(_file_record(path, dataset_root))
    return sorted(records, key=lambda item: item["path"])


def _validate_model_files(
    config: Mapping[str, Any], model_root: Path
) -> list[dict[str, Any]]:
    required = _nested(config, "release.base_model.required_files")
    if not isinstance(required, list) or not required:
        raise ReleaseError("base-model required_files must be a nonempty list")
    paths = [model_root / relative for relative in required]
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise ReleaseError(f"missing or empty frozen base-model file: {path}")
    config_payload = _read_json(model_root / "config.json")
    if not isinstance(config_payload, dict) or not config_payload.get("model_type"):
        raise ReleaseError("base-model config.json is invalid")
    if not any(path.suffix == ".safetensors" for path in paths):
        raise ReleaseError("base-model weights are missing")
    return sorted(
        (_file_record(path, model_root) for path in paths),
        key=lambda item: item["path"],
    )


def prepare_inputs(
    config_id: str, input_dir: Path, cache_dir: Path | None
) -> dict[str, Any]:
    check_release(config_id)
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise ReleaseError(
            "prepare requires huggingface_hub from rlvr/requirements-cu128.txt"
        ) from exc

    destination = _ensure_empty_directory(input_dir, label="input directory")
    cache = cache_dir.expanduser().resolve() if cache_dir else None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    config = _load_config(config_id)
    dataset = _nested(config, "release.dataset")
    base_model = _nested(config, "release.base_model")
    api = HfApi()
    dataset_info = api.dataset_info(dataset["repo_id"], revision=dataset["revision"])
    if str(dataset_info.sha) != dataset["revision"]:
        raise ReleaseError("Hugging Face resolved a different dataset revision")
    model_info = api.model_info(base_model["repo_id"], revision=base_model["revision"])
    if str(model_info.sha) != base_model["revision"]:
        raise ReleaseError("Hugging Face resolved a different base-model revision")

    dataset_files = [
        *dataset["train_files"],
        dataset["validation_file"],
        dataset["train_manifest"],
        dataset["validation_manifest"],
    ]
    dataset_root = destination / "dataset"
    model_root = destination / "base_model"
    snapshot_download(
        repo_id=dataset["repo_id"],
        repo_type="dataset",
        revision=dataset["revision"],
        allow_patterns=dataset_files,
        local_dir=dataset_root,
        cache_dir=cache,
    )
    snapshot_download(
        repo_id=base_model["repo_id"],
        revision=base_model["revision"],
        allow_patterns=base_model["required_files"],
        local_dir=model_root,
        cache_dir=cache,
    )

    receipt = {
        "schema_version": "trace-rlvr-input-receipt-v1",
        "config_id": config_id,
        "config_sha256": _sha256_path(_config_path(config_id)),
        "prompt_sha256": _sha256_path(PROMPT_PATH),
        "dataset": {
            "repo_id": dataset["repo_id"],
            "revision": dataset["revision"],
            "historical_training_revision": dataset["historical_training_revision"],
            "reproduction_dataset_revision": dataset["revision"],
            "equivalence_receipt": {
                "path": dataset["equivalence_receipt_path"],
                "sha256": dataset["equivalence_receipt_sha256"],
            },
            "files": _validate_dataset_files(config, dataset_root),
        },
        "base_model": {
            "repo_id": base_model["repo_id"],
            "revision": base_model["revision"],
            "files": _validate_model_files(config, model_root),
        },
    }
    _write_json(destination / INPUT_RECEIPT_NAME, receipt)
    return receipt


def _verify_input_receipt(
    config_id: str, input_dir: Path
) -> tuple[Path, dict[str, Any]]:
    root = input_dir.expanduser().resolve()
    receipt_path = root / INPUT_RECEIPT_NAME
    receipt = _read_json(receipt_path)
    expected = {
        "schema_version": "trace-rlvr-input-receipt-v1",
        "config_id": config_id,
        "config_sha256": _sha256_path(_config_path(config_id)),
        "prompt_sha256": PROMPT_SHA256,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ReleaseError(f"input receipt {key} mismatch")
    config = _load_config(config_id)
    dataset = _nested(config, "release.dataset")
    base_model = _nested(config, "release.base_model")
    expected_groups = {
        "dataset": {
            "repo_id": dataset["repo_id"],
            "revision": dataset["revision"],
            "historical_training_revision": dataset["historical_training_revision"],
            "reproduction_dataset_revision": dataset["revision"],
            "equivalence_receipt": {
                "path": dataset["equivalence_receipt_path"],
                "sha256": dataset["equivalence_receipt_sha256"],
            },
            "paths": {
                *dataset["train_files"],
                dataset["validation_file"],
                dataset["train_manifest"],
                dataset["validation_manifest"],
            },
        },
        "base_model": {
            "repo_id": base_model["repo_id"],
            "revision": base_model["revision"],
            "paths": set(base_model["required_files"]),
        },
    }
    for group, directory in (
        ("dataset", root / "dataset"),
        ("base_model", root / "base_model"),
    ):
        group_receipt = receipt.get(group)
        if not isinstance(group_receipt, dict):
            raise ReleaseError(f"input receipt is missing {group}")
        for key, expected_value in expected_groups[group].items():
            if key == "paths":
                continue
            if group_receipt.get(key) != expected_value:
                raise ReleaseError(f"input receipt {group}.{key} mismatch")
        records = group_receipt.get("files")
        if not isinstance(records, list):
            raise ReleaseError(f"input receipt {group}.files must be a list")
        record_paths = {
            record.get("path") for record in records if isinstance(record, dict)
        }
        if record_paths != expected_groups[group]["paths"] or len(records) != len(
            record_paths
        ):
            raise ReleaseError(f"input receipt {group} file inventory mismatch")
        for record in records:
            relative = record.get("path")
            if (
                not isinstance(relative, str)
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
            ):
                raise ReleaseError(f"unsafe {group} receipt path: {relative!r}")
            path = directory / relative
            if not path.is_file() or path.stat().st_size != record.get("size"):
                raise ReleaseError(f"input file size mismatch: {path}")
            if _sha256_path(path) != record.get("sha256"):
                raise ReleaseError(f"input file hash mismatch: {path}")
    return root, receipt


def _gpu_inventory(
    cuda_visible_devices: str | None,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    env_update: dict[str, str] = {}
    requested = (
        cuda_visible_devices
        if cuda_visible_devices is not None
        else os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    if requested:
        devices = [value.strip() for value in requested.split(",") if value.strip()]
        if len(devices) != 8 or len(set(devices)) != 8:
            raise ReleaseError(
                "canonical training requires exactly eight distinct visible CUDA devices"
            )
        if any(not re.fullmatch(r"[A-Za-z0-9_.:-]+", value) for value in devices):
            raise ReleaseError("CUDA device selection contains an unsupported value")
        env_update["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReleaseError(
            "nvidia-smi is required to validate the training hardware"
        ) from exc
    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", maxsplit=3)]
        if len(parts) == 4:
            rows.append(
                dict(zip(("index", "name", "driver_version", "memory_mib"), parts))
            )
    visible_count = 8 if requested else len(rows)
    if visible_count != 8 or len(rows) < 8:
        raise ReleaseError(
            f"canonical training requires eight visible GPUs; found {visible_count}"
        )
    return env_update, rows


def _resolved_config(
    config_id: str,
    input_root: Path,
    output_root: Path,
    *,
    smoke: bool,
    wandb_enabled: bool,
    wandb_project: str,
) -> dict[str, Any]:
    config = copy.deepcopy(_load_config(config_id)["easyr1"])
    config["data"]["train_files"] = str(input_root / "dataset" / "data" / "train")
    config["data"]["val_files"] = str(
        input_root
        / "dataset"
        / "data"
        / "validation"
        / "trace_rlvr_validation_iid_2000_all1000_seed1042.parquet"
    )
    config["data"]["system_prompt_file"] = str(PROMPT_PATH)
    config["worker"]["actor"]["model"]["model_path"] = str(input_root / "base_model")
    config["worker"]["actor"]["model"]["tokenizer_path"] = str(
        input_root / "base_model"
    )
    config["worker"]["reward"]["reward_function"] = f"{REWARD_PATH}:compute_score"
    config["trainer"]["save_checkpoint_path"] = str(output_root / "checkpoint")
    config["trainer"]["logger"] = ["console", "wandb"] if wandb_enabled else ["console"]
    config["trainer"]["project_name"] = wandb_project
    if smoke:
        config["trainer"]["experiment_name"] = f"{config_id}-smoke"
        config["trainer"]["max_steps"] = 1
        config["trainer"]["save_freq"] = 1
        config["trainer"]["val_freq"] = -1
        config["trainer"]["val_before_train"] = False
    return config


def _runtime_environment(
    output_root: Path,
    *,
    cache_dir: Path | None,
    cuda_update: Mapping[str, str],
    wandb_enabled: bool,
    wandb_entity: str | None,
) -> dict[str, str]:
    env = dict(os.environ)
    python_paths = [str(BACKEND_ROOT), str(SOURCE_ROOT)]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "true"
    env["RAY_TMPDIR"] = str(output_root / "ray-temp")
    env["WANDB_MODE"] = "online" if wandb_enabled else "disabled"
    for key in ("WANDB_NAME", "WANDB_PROJECT", "WANDB_RESUME", "WANDB_RUN_ID"):
        env.pop(key, None)
    if wandb_entity:
        env["WANDB_ENTITY"] = wandb_entity
    else:
        env.pop("WANDB_ENTITY", None)
    if cache_dir:
        cache = cache_dir.expanduser().resolve()
        cache.mkdir(parents=True, exist_ok=True)
        env["HF_HOME"] = str(cache)
    env.update(cuda_update)
    env.pop("USE_MODELSCOPE_HUB", None)
    return env


def run_training(
    config_id: str,
    input_dir: Path,
    output_dir: Path,
    *,
    smoke: bool,
    cache_dir: Path | None,
    cuda_visible_devices: str | None,
    wandb_enabled: bool,
    wandb_project: str,
    wandb_entity: str | None,
) -> int:
    check_release(config_id)
    if wandb_entity and not wandb_enabled:
        raise ReleaseError("--wandb-entity requires --wandb")
    input_root, input_receipt = _verify_input_receipt(config_id, input_dir)
    cuda_update, hardware = _gpu_inventory(cuda_visible_devices)
    output_root = _ensure_empty_directory(output_dir, label="output directory")
    resolved = _resolved_config(
        config_id,
        input_root,
        output_root,
        smoke=smoke,
        wandb_enabled=wandb_enabled,
        wandb_project=wandb_project,
    )
    resolved_path = output_root / "resolved_easyr1.yaml"
    resolved_path.write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    receipt = {
        "schema_version": "trace-rlvr-run-receipt-v1",
        "config_id": config_id,
        "config_sha256": _sha256_path(_config_path(config_id)),
        "input_receipt_sha256": _sha256_path(input_root / INPUT_RECEIPT_NAME),
        "mode": "smoke" if smoke else "canonical",
        "expected_step": 1 if smoke else 500,
        "resolved_config_sha256": _sha256_path(resolved_path),
        "hardware": hardware,
        "reference_hardware_match": all("H100" in row["name"] for row in hardware[:8]),
        "status": "launching",
        "wandb_enabled": wandb_enabled,
    }
    receipt_path = output_root / RUN_RECEIPT_NAME
    _write_json(receipt_path, receipt)
    env = _runtime_environment(
        output_root,
        cache_dir=cache_dir,
        cuda_update=cuda_update,
        wandb_enabled=wandb_enabled,
        wandb_entity=wandb_entity,
    )
    command = [sys.executable, "-m", "verl.trainer.main", f"config={resolved_path}"]
    try:
        subprocess.run(command, cwd=BACKEND_ROOT, env=env, check=True)
    except subprocess.CalledProcessError as exc:
        receipt["status"] = "failed"
        receipt["exit_code"] = exc.returncode
        _write_json(receipt_path, receipt)
        return int(exc.returncode)
    receipt["status"] = "completed"
    _write_json(receipt_path, receipt)
    return 0


def merge_checkpoint(
    config_id: str, checkpoint_dir: Path, output_dir: Path
) -> dict[str, Any]:
    check_release(config_id)
    checkpoint_root = checkpoint_dir.expanduser().resolve()
    run_receipt_path = checkpoint_root.parent / RUN_RECEIPT_NAME
    run_receipt = _read_json(run_receipt_path)
    if (
        run_receipt.get("config_id") != config_id
        or run_receipt.get("status") != "completed"
    ):
        raise ReleaseError(
            "checkpoint run receipt is missing, incomplete, or for another config"
        )
    step = run_receipt.get("expected_step")
    if step not in (1, 500) or (step == 1 and run_receipt.get("mode") != "smoke"):
        raise ReleaseError("checkpoint receipt has an invalid selected step")
    step_root = checkpoint_root / f"global_step_{step}"
    actor_root = step_root / "actor"
    tracker = _read_json(checkpoint_root / "checkpoint_tracker.json")
    if int(tracker.get("last_global_step", -1)) != step:
        raise ReleaseError("checkpoint tracker does not match the selected step")
    model_shards = sorted(actor_root.glob("model_world_size_8_rank_*.pt"))
    expected_names = {f"model_world_size_8_rank_{rank}.pt" for rank in range(8)}
    if {path.name for path in model_shards} != expected_names:
        raise ReleaseError(
            "checkpoint does not contain the expected eight model shards"
        )
    if any(path.stat().st_size == 0 for path in model_shards):
        raise ReleaseError("checkpoint contains an empty model shard")
    if not (actor_root / "huggingface" / "config.json").is_file():
        raise ReleaseError("checkpoint is missing its Hugging Face config")
    output = output_dir.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ReleaseError(f"merge output must not already exist: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(BACKEND_ROOT), str(SOURCE_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    subprocess.run(
        [
            sys.executable,
            str(MERGER_PATH),
            "--local_dir",
            str(actor_root),
            "--output_dir",
            str(output),
        ],
        cwd=BACKEND_ROOT,
        env=env,
        check=True,
    )
    receipt = {
        "schema_version": "trace-rlvr-merge-receipt-v1",
        "config_id": config_id,
        "checkpoint_step": step,
        "run_receipt_sha256": _sha256_path(run_receipt_path),
        "publication_performed": False,
    }
    _write_json(output / "trace_merge_receipt.json", receipt)
    return receipt


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, choices=CONFIG_IDS)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--wandb", action="store_true", help="Opt in to W&B telemetry")
    parser.add_argument("--wandb-project", default="trace_easyr1")
    parser.add_argument("--wandb-entity")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check_parser = subparsers.add_parser(
        "check", help="validate the RLVR configuration offline"
    )
    check_parser.add_argument("--config", required=True, choices=CONFIG_IDS)
    prepare_parser = subparsers.add_parser(
        "prepare", help="download and verify immutable inputs"
    )
    prepare_parser.add_argument("--config", required=True, choices=CONFIG_IDS)
    prepare_parser.add_argument("--input-dir", required=True, type=Path)
    prepare_parser.add_argument("--cache-dir", type=Path)
    _add_run_arguments(
        subparsers.add_parser("run", help="launch the canonical 500-step run")
    )
    _add_run_arguments(
        subparsers.add_parser("smoke", help="launch a marked one-step GPU smoke run")
    )
    merge_parser = subparsers.add_parser(
        "merge", help="merge a validated EasyR1 checkpoint"
    )
    merge_parser.add_argument("--config", required=True, choices=CONFIG_IDS)
    merge_parser.add_argument("--checkpoint-dir", required=True, type=Path)
    merge_parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "check":
            print(
                json.dumps(
                    check_release(args.config), sort_keys=True, separators=(",", ":")
                )
            )
            return 0
        if args.command == "prepare":
            receipt = prepare_inputs(args.config, args.input_dir, args.cache_dir)
            print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
            return 0
        if args.command in {"run", "smoke"}:
            return run_training(
                args.config,
                args.input_dir,
                args.output_dir,
                smoke=args.command == "smoke",
                cache_dir=args.cache_dir,
                cuda_visible_devices=args.cuda_visible_devices,
                wandb_enabled=args.wandb,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
            )
        if args.command == "merge":
            receipt = merge_checkpoint(
                args.config, args.checkpoint_dir, args.output_dir
            )
            print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
            return 0
        raise ReleaseError(f"unsupported command {args.command!r}")
    except (ReleaseError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
