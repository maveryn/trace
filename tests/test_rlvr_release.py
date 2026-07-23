"""CPU-only checks for the TRACE RLVR training workflow."""

from __future__ import annotations

import ast
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RLVR_ROOT = REPO_ROOT / "rlvr"
TRAIN_SCRIPT = RLVR_ROOT / "train.py"
REWARD_ADAPTER = (
    RLVR_ROOT / "easyr1_backend" / "examples" / "reward_function" / "trace_rlvr.py"
)
CONFIG_IDS = ("trace-qwen2.5-vl-3b", "trace-qwen2.5-vl-7b")

if not RLVR_ROOT.is_dir():
    pytest.skip("the clone-only RLVR release is not present", allow_module_level=True)


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _release_tree_fingerprint() -> dict[str, str]:
    return {
        path.relative_to(RLVR_ROOT).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in RLVR_ROOT.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }


def _offline_environment() -> dict[str, str]:
    environment = dict(os.environ)
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(REPO_ROOT / "src") + (
        f"{os.pathsep}{existing_pythonpath}" if existing_pythonpath else ""
    )
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "offline",
        }
    )
    return environment


def _reward_input(response: str) -> dict[str, Any]:
    return {
        "response": response,
        "ground_truth": {"type": "integer", "value": 2},
        "annotation_gt": {"type": "bbox", "value": [10, 10, 20, 20]},
        "reward_contract": {
            "reward_contract_version": "v0",
            "answer": {"id": "answer_exact_match_v0", "type": "integer"},
            "annotation": {"id": "bbox_soft_iou_v0", "type": "bbox"},
        },
    }


def _mapping_keys(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _mapping_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _mapping_keys(child)


def _parser_option_strings(parser) -> set[str]:
    options: set[str] = set()
    for action in parser._actions:
        options.update(action.option_strings)
        choices = getattr(action, "choices", None)
        if isinstance(choices, dict):
            for child in choices.values():
                options.update(_parser_option_strings(child))
    return options


def test_canonical_config_pair_matches_training_contract() -> None:
    train = _load_module("trace_public_rlvr_train", TRAIN_SCRIPT)

    assert train.CONFIG_IDS == CONFIG_IDS
    assert sorted(path.name for path in (RLVR_ROOT / "configs").glob("*.yaml")) == [
        f"{config_id}.yaml" for config_id in CONFIG_IDS
    ]
    train._validate_profile_pair()

    for config_id in CONFIG_IDS:
        config = train._load_config(config_id)
        dataset = config["release"]["dataset"]
        easyr1 = config["easyr1"]
        assert config["release"]["scope"] == "answer_only"
        assert dataset["revision"] == "4e5b54361360296a855542b40cfd8b7f81b355fe"
        assert dataset["historical_training_revision"] == (
            "e317b746b258630682367cc6a9d87dedd195113c"
        )
        assert dataset["equivalence_receipt_path"] == "rlvr/dataset_equivalence.v1.json"
        assert dataset["equivalence_receipt_sha256"] == (
            "40afcb59e0c67d7b7f47b78bb673a6c152320eb0656a990910ff52c634d43ae7"
        )
        assert dataset["required_columns"] == train.REQUIRED_DATASET_COLUMNS
        assert "trace_supervision_mode" not in dataset["required_columns"]
        assert easyr1["data"]["prompt_key"] == "prompt_answer"
        assert easyr1["data"]["answer_key"] == "answer_gt"
        assert easyr1["worker"]["reward"]["reward_function_kwargs"] == {}
        assert easyr1["trainer"]["max_steps"] == 500
        assert easyr1["trainer"]["save_limit"] == 1
        assert easyr1["trainer"]["load_checkpoint_path"] is None
        assert easyr1["trainer"]["find_last_checkpoint"] is False


def test_dataset_equivalence_receipt_pins_all_training_parquets() -> None:
    train = _load_module("trace_public_rlvr_train_dataset_receipt", TRAIN_SCRIPT)
    config = train._load_config(CONFIG_IDS[0])

    hashes = train._dataset_equivalence_file_hashes(config)

    assert len(hashes) == 17
    assert set(hashes) == {
        *config["release"]["dataset"]["train_files"],
        config["release"]["dataset"]["validation_file"],
    }
    assert all(len(value) == 64 for value in hashes.values())


@pytest.mark.parametrize("config_id", CONFIG_IDS)
def test_offline_cli_check_is_deterministic_and_read_only(config_id: str) -> None:
    before = _release_tree_fingerprint()
    command = [sys.executable, str(TRAIN_SCRIPT), "check", "--config", config_id]
    outputs = []
    for _ in range(2):
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=_offline_environment(),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert result.returncode == 0, result.stderr
        outputs.append(result.stdout)

    assert outputs[0] == outputs[1]
    payload = json.loads(outputs[0])
    assert payload["config_id"] == config_id
    assert payload["status"] == "ok"
    assert _release_tree_fingerprint() == before


def test_reward_adapter_preserves_batch_order_and_canonical_values() -> None:
    reward = _load_module("trace_public_rlvr_reward", REWARD_ADAPTER)
    inputs = [
        _reward_input('{"answer":2}'),
        _reward_input('{"answer":3}'),
        _reward_input('{"answer":2,"extra":1}'),
        _reward_input("not JSON"),
    ]

    scores = reward.compute_score(inputs)

    assert [score["overall"] for score in scores] == pytest.approx(
        [1.0, 0.05, 0.95, 0.0]
    )
    assert [score["score"] for score in scores] == pytest.approx([1.0, 0.05, 0.95, 0.0])
    assert [score["answer_reward"] for score in scores] == [1.0, 0.0, 1.0, 0.0]
    assert [score["format"] for score in scores] == [1.0, 1.0, 0.0, 0.0]


def test_reward_adapter_accepts_serialized_metadata() -> None:
    reward = _load_module("trace_public_rlvr_reward_serialized", REWARD_ADAPTER)
    reward_input = _reward_input('{"answer":2}')
    for field in ("ground_truth", "annotation_gt", "reward_contract"):
        reward_input[field] = json.dumps(reward_input[field])

    assert reward.compute_score([reward_input])[0]["overall"] == 1.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ground_truth", None),
        ("ground_truth", "not JSON"),
        ("ground_truth", {}),
        ("annotation_gt", None),
        ("annotation_gt", "not JSON"),
        ("reward_contract", None),
        ("reward_contract", "not JSON"),
    ],
)
def test_reward_adapter_fails_closed_for_missing_or_malformed_metadata(
    field: str,
    value: Any,
) -> None:
    reward = _load_module(f"trace_public_rlvr_reward_invalid_{field}", REWARD_ADAPTER)
    reward_input = _reward_input('{"answer":2}')
    if value is None:
        reward_input.pop(field)
    else:
        reward_input[field] = value

    with pytest.raises(ValueError):
        reward.compute_score([reward_input])


def test_vendored_python_is_parseable_and_has_no_private_trace_imports() -> None:
    backend_root = RLVR_ROOT / "easyr1_backend"
    python_files = sorted(backend_root.rglob("*.py"))
    assert python_files
    private_imports: list[str] = []
    public_imports: list[tuple[str, str]] = []

    for path in python_files:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path), feature_version=(3, 10))
        assert "/" + "home/" not in source
        assert "/" + "dev/shm" not in source
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            for name in names:
                relative = path.relative_to(backend_root).as_posix()
                if name == "trace" or name.startswith("trace."):
                    private_imports.append(f"{relative}: {name}")
                if name == "trace_tasks" or name.startswith("trace_tasks."):
                    public_imports.append((relative, name))

    assert private_imports == []
    assert public_imports == [
        (
            "examples/reward_function/trace_rlvr.py",
            "trace_tasks.core.reward_scoring",
        )
    ]


def test_release_exposes_no_annotation_training_mode() -> None:
    train = _load_module("trace_public_rlvr_train_surface", TRAIN_SCRIPT)
    forbidden_identifiers = {
        "annotation_profile",
        "annotation_reward_formula",
        "final_answer_only_manifest",
        "task_conditioned",
        "trace_output_mode",
    }

    for config_id in CONFIG_IDS:
        config = train._load_config(config_id)
        easyr1 = config["easyr1"]
        keys = set(_mapping_keys(easyr1))
        assert not any("annotation" in key for key in keys)
        assert forbidden_identifiers.isdisjoint(keys)
        assert easyr1["data"]["prompt_key"] == "prompt_answer"
        assert easyr1["data"]["answer_key"] == "answer_gt"
        assert "trace_supervision_mode" not in keys
        assert (
            "trace_supervision_mode"
            not in config["release"]["dataset"]["required_columns"]
        )

    options = _parser_option_strings(train._parser())
    assert not any("annotation" in option for option in options)

    backend_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((RLVR_ROOT / "easyr1_backend" / "verl").rglob("*.py"))
    )
    for identifier in forbidden_identifiers:
        assert identifier not in backend_source
