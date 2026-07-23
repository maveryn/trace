from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import json
import stat
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = REPO_ROOT / "rlvr" / "evaluation"
SCRIPTS_ROOT = EVALUATION_ROOT / "scripts"
EXTENSIONS_ROOT = EVALUATION_ROOT / "vlmevalkit_extensions"

if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))


EXPECTED_BENCHMARK_KEYS = (
    "chartqapro",
    "charxivreason",
    "tablevqabench",
    "evochart",
    "mathvision",
    "mathvista",
    "mathverse",
    "wemath",
    "phyx_mini_mc",
    "mmmu_pro_vision",
    "realworldqa",
    "mmstar",
    "embspatial",
    "spatialvizbench_cot",
    "cvbench_3d",
    "erqa",
    "blink",
    "countbenchqa",
    "countqa",
    "treebench",
    "puzzlevqa",
    "visualpuzzles",
    "logicvista",
    "mme_reasoning",
)


def _script(name: str):
    importlib.invalidate_caches()
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def _load_extension(monkeypatch: pytest.MonkeyPatch, stem: str):
    """Load one adapter with a minimal, offline VLMEvalKit boundary."""

    package_name = f"_trace_eval_test_{stem}"
    package = types.ModuleType(package_name)
    package.__path__ = [str(EXTENSIONS_ROOT)]  # type: ignore[attr-defined]

    class ImageBaseDataset:
        def build_prompt(self, line):
            return [
                {"type": "image", "value": line.get("image_path", "image.png")},
                {"type": "text", "value": str(line.get("question", ""))},
            ]

    image_base = types.ModuleType(f"{package_name}.image_base")
    image_base.ImageBaseDataset = ImageBaseDataset

    vlmeval = types.ModuleType("vlmeval")
    vlmeval.__path__ = []  # type: ignore[attr-defined]
    smp = types.ModuleType("vlmeval.smp")
    smp.LMUDataRoot = lambda: "/unused"
    smp.dump = lambda *_args, **_kwargs: None
    smp.get_intermediate_file_path = lambda path, *_args: path
    smp.load = lambda _path: pd.DataFrame()

    monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, image_base.__name__, image_base)
    monkeypatch.setitem(sys.modules, "vlmeval", vlmeval)
    monkeypatch.setitem(sys.modules, "vlmeval.smp", smp)

    module_name = f"{package_name}.{stem}"
    spec = importlib.util.spec_from_file_location(
        module_name, EXTENSIONS_ROOT / f"{stem}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_trace_eval_v1_is_the_only_queue_route_and_has_exact_partition() -> None:
    suite_module = _script("trace_eval_suite")
    queue_lib = _script("benchmark_queue_lib")
    generation = _script("run_external_benchmark_generation_api_queue")
    mme = _script("run_mme_reasoning_eval")
    score_campaign = _script("run_trace_eval_official_score_campaign")

    suite = suite_module.load_trace_eval_suite()
    assert suite.path == EVALUATION_ROOT / "trace_eval" / "suite.v1.json"
    assert suite.suite_id == "trace_eval_v1"
    assert suite.benchmark_keys == EXPECTED_BENCHMARK_KEYS
    assert suite.rows_per_model_seed == 32_805
    assert tuple(len(keys) for keys in suite.categories.values()) == (4,) * 6
    assert {name: len(keys) for name, keys in suite.routes.items()} == {
        "direct_score": 7,
        "official_vlmevalkit": 16,
        "dedicated_score": 1,
    }
    routed = tuple(key for keys in suite.routes.values() for key in keys)
    assert len(routed) == len(set(routed)) == 24
    assert set(routed) == set(EXPECTED_BENCHMARK_KEYS)

    assert queue_lib.BENCHMARK_RUN_SETS == ("trace_eval_v1",)
    assert queue_lib.TRACE_EVAL_V1_BENCHMARKS == EXPECTED_BENCHMARK_KEYS
    assert (
        tuple(
            spec.key for spec in queue_lib.benchmark_specs_for_run_set("trace_eval_v1")
        )
        == EXPECTED_BENCHMARK_KEYS
    )

    args = generation.build_parser().parse_args(
        [
            "--model",
            "model",
            "--model-slug",
            "model",
            "--api-model",
            "model",
            "--dataset-manifest-view",
            "trace_eval_v1",
            "--run-set",
            "trace_eval_v1",
        ]
    )
    assert args.dataset_manifest_view == args.run_set == "trace_eval_v1"
    generation_parser = generation.build_parser()
    actions = {
        option: action
        for action in generation_parser._actions
        for option in action.option_strings
    }
    assert actions["--media-transport"].choices == ("file-url",)
    assert "--subset-root" not in actions
    assert "--limit" not in actions

    mme_parser = mme.build_parser()
    mme_subcommands = next(
        action
        for action in mme_parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    assert set(mme_subcommands.choices) == {"score"}
    mme_args = mme_parser.parse_args(
        [
            "score",
            "--model",
            "model",
            "--model-slug",
            "model",
            "--run-root",
            "/tmp/trace-eval-offline-runs",
            "--benchmark-root",
            "/tmp/trace-eval-offline-benchmarks",
        ]
    )
    assert mme_args.command == "score"
    assert 'run_set="full"' not in (
        SCRIPTS_ROOT / "run_mme_reasoning_eval.py"
    ).read_text(encoding="utf-8")

    score_args = score_campaign.build_parser().parse_args(
        [
            "--campaign",
            "model",
            "model",
            "/tmp/trace-eval-offline-campaign",
            "--score-root",
            "/tmp/trace-eval-offline-scores",
        ]
    )
    expected_work_root = EVALUATION_ROOT / ".work"
    assert score_args.eval_deps == expected_work_root / "eval_deps"
    assert score_args.lmu_data == expected_work_root / "LMUData"
    assert score_args.judge_model == expected_work_root / "models" / "qwen3-32b-judge"


def test_runtime_sources_are_neutral_and_import_closed() -> None:
    required = {
        SCRIPTS_ROOT / "run_external_benchmark_generation_api_queue.py",
        SCRIPTS_ROOT / "run_external_benchmark_score_queue.py",
        SCRIPTS_ROOT / "run_official_vlmevalkit_saved_score.py",
        SCRIPTS_ROOT / "trace_eval_public_export.py",
        SCRIPTS_ROOT / "run_trace_eval_publish_worker.py",
        EXTENSIONS_ROOT / "batched_vlmevalkit_qwen3vl.py",
        EXTENSIONS_ROOT / "evochart.py",
        EXTENSIONS_ROOT / "trace_eval_answer_parsing.py",
        EXTENSIONS_ROOT / "trace_eval_local_vqa.py",
    }
    assert all(path.is_file() for path in required)

    forbidden_paths = (
        REPO_ROOT / "rlvr" / "experiments" / "final_answer_only_manifest.json",
        SCRIPTS_ROOT / "run_trace_eval_rl_baselines.sh",
        SCRIPTS_ROOT / "run_external_benchmark_generation_queue.py",
        EXTENSIONS_ROOT / "batched_chartmuseum_vllm.py",
        EXTENSIONS_ROOT / "batched_chartqapro_vllm.py",
    )
    assert not any(path.exists() for path in forbidden_paths)
    excluded_name_fragments = (
        "final24",
        "final25",
        "final26",
        "final31",
        "repair",
        "recover",
        "reuse",
    )
    assert not any(
        fragment in path.name.lower()
        for path in EVALUATION_ROOT.rglob("*")
        for fragment in excluded_name_fragments
    )

    python_files = sorted(SCRIPTS_ROOT.glob("*.py")) + sorted(
        EXTENSIONS_ROOT.glob("*.py")
    )
    script_modules = {path.stem for path in SCRIPTS_ROOT.glob("*.py")}
    extension_modules = {path.stem for path in EXTENSIONS_ROOT.glob("*.py")}
    local_modules = script_modules | extension_modules
    missing_local: list[str] = []
    forbidden_imports: list[str] = []
    forbidden_names = {
        "screenspot_json_contract",
        "trace_eval_archive_migration_lib",
        "batched_chartmuseum_vllm",
        "batched_chartqapro_vllm",
    }

    for path in python_files:
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        for imported in _imports(path):
            parts = imported.split(".")
            leaf = parts[-1]
            if any(
                name in parts or imported.startswith((f"{name}.", f"scripts.{name}"))
                for name in forbidden_names
            ):
                forbidden_imports.append(f"{path.name}: {imported}")
            if "final25" in imported.lower() or "final26" in imported.lower():
                forbidden_imports.append(f"{path.name}: {imported}")

            root = parts[0]
            looks_local = root not in {"trace_tasks"} and root.startswith(
                (
                    "apply_",
                    "benchmark_",
                    "batched_",
                    "build_",
                    "prepare_",
                    "run_",
                    "status_",
                    "summarize_",
                    "trace_",
                    "verify_",
                )
            )
            if root == "scripts" and len(parts) > 1:
                looks_local = True
                leaf = parts[1]
            if looks_local and leaf not in local_modules:
                missing_local.append(f"{path.name}: {imported}")

    assert forbidden_imports == []
    assert missing_local == []


def test_runtime_implementation_is_trace_eval_v1_only() -> None:
    runtime_paths = (
        SCRIPTS_ROOT / "benchmark_queue_lib.py",
        SCRIPTS_ROOT / "prepare_trace_eval_datasets.py",
        SCRIPTS_ROOT / "prepare_trace_eval_manifest.py",
        SCRIPTS_ROOT / "run_external_benchmark_generation_api_queue.py",
        SCRIPTS_ROOT / "run_external_benchmark_score_queue.py",
        SCRIPTS_ROOT / "run_mme_reasoning_eval.py",
        SCRIPTS_ROOT / "run_official_vlmevalkit_saved_score.py",
        SCRIPTS_ROOT / "run_trace_eval_official_score_campaign.py",
        EXTENSIONS_ROOT / "batched_vlmevalkit_qwen3vl.py",
    )
    forbidden_fragments = (
        "candidate37",
        "external_eval_v1",
        "trace_grounding",
        "chartmuseum",
        "screenspot",
        "qbench_video",
        "videommmu",
        "video_tt",
        "tempcompass",
        "seephys",
        "scienceqa",
        "mmhelix",
        "visulogic",
    )
    violations = []
    for path in runtime_paths:
        source = path.read_text(encoding="utf-8").lower()
        violations.extend(
            f"{path.name}: {fragment}"
            for fragment in forbidden_fragments
            if fragment in source
        )
    assert violations == []

    queue_lib = _script("benchmark_queue_lib")
    assert tuple(spec.key for spec in queue_lib.ALL_BENCHMARKS) == (
        EXPECTED_BENCHMARK_KEYS
    )
    with pytest.raises(KeyError, match="not a trace_eval_v1 benchmark"):
        queue_lib.spec_by_key("excluded-benchmark")


def test_producer_era_names_are_confined_to_wire_compatibility() -> None:
    allowed_line_markers = {
        "run_external_benchmark_generation_api_queue.py": ("final25_code_hash",),
        "run_trace_eval_publish_worker.py": ("final25_code_hash",),
        "trace_eval_archive_hooks.py": (
            "trace-final25-archive-hooks-v1",
            "final25_code_hash",
        ),
        "trace_eval_campaign_verification.py": ("final25_code_hash",),
        "trace_eval_hf_archive_lib.py": ("trace-final25-hf-archive-v1",),
        "trace_eval_media_contract.py": (
            "trace-final25-datasets-v2",
            "trace-final25-media-v2",
            "trace-final25-generation-v7",
        ),
        "trace_eval_public_export.py": (
            '"final24"',
            '"final25"',
            '"final26"',
            '"final31"',
            "final25_code_hash",
        ),
        "verify_trace_eval.py": ("final25_code_hash",),
    }
    legacy_fragments = ("final24", "final25", "final26", "final31")
    violations: list[str] = []
    for path in sorted(SCRIPTS_ROOT.glob("*.py")):
        markers = allowed_line_markers.get(path.name, ())
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            lowered = line.lower()
            if not any(fragment in lowered for fragment in legacy_fragments):
                continue
            if not any(marker.lower() in lowered for marker in markers):
                violations.append(f"{path.name}:{line_number}: {line.strip()}")

    assert violations == []


def test_manifest_preparation_has_one_canonical_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_module = _script("prepare_trace_eval_manifest")
    materialization = _script("prepare_trace_eval_datasets")
    monkeypatch.setattr(manifest_module, "_git_commit", lambda _path: "a" * 40)

    payload = manifest_module._new_manifest(
        lmu_root=tmp_path / "LMUData",
        vlmeval_root=tmp_path / "VLMEvalKit",
    )
    assert payload["suite_id"] == "trace_eval_v1"
    assert payload["dataset_views"] == {"trace_eval_v1": list(EXPECTED_BENCHMARK_KEYS)}
    assert set(payload["datasets"]) == set()

    with pytest.raises(ValueError, match="non-canonical dataset alias"):
        materialization._materialize_dataset(
            key="chartqapro",
            alias="not-chartqapro",
            expected_rows=1948,
            lmu_root=tmp_path,
            workers=1,
            token=None,
        )

    receipts = {
        key: {
            "status": "ready",
            "alias": benchmark.official_alias,
            "rows": benchmark.rows,
            "expected_rows": benchmark.rows,
        }
        for key, benchmark in {
            item.key: item
            for item in _script("trace_eval_suite").load_trace_eval_suite().benchmarks
        }.items()
    }
    complete = {
        **payload,
        "datasets": receipts,
        "view_snapshot_sha256": {"trace_eval_v1": "b" * 64},
    }
    monkeypatch.setattr(
        manifest_module,
        "_receipt_is_complete",
        lambda receipt, *, alias, expected_rows: (
            receipt["alias"] == alias and receipt["rows"] == expected_rows
        ),
    )
    manifest_module.validate_trace_eval_manifest(complete)

    polluted = json.loads(json.dumps(complete))
    polluted["dataset_views"]["historical"] = ["chartqapro"]
    with pytest.raises(ValueError, match="only the trace_eval_v1 view"):
        manifest_module.validate_trace_eval_manifest(polluted)


def test_every_runtime_script_imports_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("WANDB_MODE", "offline")

    failures: dict[str, str] = {}
    for path in sorted(SCRIPTS_ROOT.glob("*.py")):
        try:
            _script(path.stem)
        except Exception as error:  # pragma: no cover - assertion renders the detail
            failures[path.name] = f"{type(error).__name__}: {error}"
    assert failures == {}


def test_generation_api_queue_preserves_duplicate_row_identity(tmp_path: Path) -> None:
    queue_lib = _script("benchmark_queue_lib")
    generation_api = _script("run_external_benchmark_generation_api_queue")

    frame = pd.DataFrame(
        [
            {"index": "duplicate", "question": "first"},
            {"index": "duplicate", "question": "second"},
        ]
    )
    first_hash = generation_api._row_hash(frame.iloc[0])
    second_hash = generation_api._row_hash(frame.iloc[1])
    assert first_hash != second_hash
    assert second_hash == (
        "b48e9befa89a01decd971f591f0433a696fd0642a8677657b17f152a0223d631"
    )

    row = {"index": "duplicate", "question": "second"}
    job = generation_api.RowJob(
        spec=queue_lib.BenchmarkSpec("demo", "Demo", "Demo", "demo"),
        row=row,
        rank=0,
        row_key=generation_api._row_key(row, 0),
        output_dir=tmp_path,
        result_path=tmp_path / "result.json",
        source_ordinal=17,
    )
    result = generation_api._result_from_response(
        job,
        {
            "choices": [{"message": {"content": "A"}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 1, "prompt_tokens": 4},
        },
        "http://offline-endpoint/v1",
    )
    assert result["prediction"] == "A"
    assert result["source_ordinal"] == 17
    assert len(result["source_row_hash"]) == 64


def test_score_queue_uses_strict_canonical_parsers() -> None:
    scorer = _script("run_external_benchmark_score_queue")

    accepted = {
        "0": 0,
        "Judgement: 1": 1,
        "**Judgement: 0**": 0,
        "Judgement: 1\n\n**Explanation**: equivalent": 1,
    }
    for output, expected in accepted.items():
        assert scorer._parse_mathverse_score_output(output) == expected
    for malformed in ("1.", "True", "0 or 1", "Judgement: 1**", ""):
        assert scorer._parse_mathverse_score_output(malformed) is None


def test_official_saved_score_extracts_primary_metric_and_redacts_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved_score = _script("run_official_vlmevalkit_saved_score")
    answer_parsing = _load_extension(monkeypatch, "trace_eval_answer_parsing")

    class OfficialDataset:
        dataset_name = "OfficialAlias"

        @classmethod
        def report_primary_metric(cls, metrics):
            return {"Overall Accuracy": metrics["split=Overall|acc"]}

    score, receipt = saved_score._primary_score(
        OfficialDataset(),
        pd.DataFrame([{"split": "Overall", "acc": 0.625}]),
        lambda result: {"split=Overall|acc": result.iloc[0]["acc"]},
        None,
        "auto",
    )
    assert score == 62.5
    assert receipt["key"] == "Overall Accuracy"
    assert answer_parsing.unwrap_single_answer_block("<answer>42</answer>") == "42"
    assert answer_parsing.extract_unambiguous_abcd(r"reasoning \boxed{C}") == "C"
    assert answer_parsing.extract_unambiguous_abcd("Answer: A then Answer: B") == "Z"
    assert saved_score._redact(
        {"max_tokens": 256, "api_key": "secret", "nested": {"token": "secret"}}
    ) == {
        "max_tokens": 256,
        "api_key": "<redacted>",
        "nested": {"token": "<redacted>"},
    }


def test_official_saved_score_is_confined_to_registered_official_routes() -> None:
    saved_score = _script("run_official_vlmevalkit_saved_score")

    with pytest.raises(ValueError, match="--benchmark-key is required"):
        saved_score._resolve_benchmark(None, "ScreenSpot")
    with pytest.raises(ValueError, match="direct_score"):
        saved_score._resolve_benchmark("mathvision", None)
    with pytest.raises(ValueError, match="dedicated_score"):
        saved_score._resolve_benchmark("mme_reasoning", None)

    assert saved_score._resolve_benchmark("chartqapro", None) == (
        "chartqapro",
        "ChartQAPro_CoT",
        "vlmevalkit_faithful_cot",
    )


def test_public_export_rejects_schema_and_opaque_output_leaks() -> None:
    public_export = _script("trace_eval_public_export")

    with pytest.raises(public_export.PublicExportError, match="forbidden public field"):
        public_export._assert_no_internal_markers({"prompt": "private prompt"})
    with pytest.raises(public_export.PublicExportError, match="local path"):
        public_export._assert_private_output_safe(
            {"response_text": "/" + "dev/shm/private-response.bin"}
        )
    with pytest.raises(public_export.PublicExportError, match="private identifier"):
        public_export._assert_private_output_safe(
            {"response_text": "from an unpublished campaign"},
            private_markers=("unpublished campaign",),
        )


def test_publication_interface_is_offline_until_explicit_upload_and_lock_safe(
    tmp_path: Path,
) -> None:
    worker = _script("run_trace_eval_publish_worker")

    parser = worker._parser()
    options = {option for action in parser._actions for option in action.option_strings}
    assert {
        "--paper-repo",
        "--token-env",
        "--allow-paper-run-upload",
        "--confirm-paper-run",
    } <= options
    token_action = next(
        action for action in parser._actions if "--token-env" in action.option_strings
    )
    assert token_action.default == "HF_TOKEN"

    lock_path = tmp_path / "locks" / "worker.lock"
    with worker.HeldLock(lock_path, blocking=False):
        with pytest.raises(
            worker.PublishWorkerError, match="another publication worker"
        ):
            with worker.HeldLock(lock_path, blocking=False):
                pass
    with worker.HeldLock(lock_path, blocking=False):
        assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600


def test_evochart_and_countqa_adapters_are_deterministic_and_cache_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evochart = _load_extension(monkeypatch, "evochart")
    assert evochart.score_prediction("<answer>32</answer>", "32", True) == 1.0
    assert evochart.score_prediction(r"\boxed{38%}", "38", True) == 1.0
    assert evochart.score_prediction("31", "32", False) == 1.0
    assert evochart.score_prediction("28", "32", False) == 0.0
    assert evochart.score_prediction("0.04", "0", False) == 0.0

    dataset = object.__new__(evochart.EvoChart)
    dataset.dataset_name = "EvoChart_Qwen3_ZS"
    prompt = dataset.build_prompt(
        {"question": "What is the value?", "image_path": "offline.png"}
    )
    assert prompt[-1]["value"].endswith(
        "Answer the question using a single word or phrase."
    )

    local_vqa = _load_extension(monkeypatch, "trace_eval_local_vqa")
    monkeypatch.setattr(local_vqa, "LMUDataRoot", lambda: str(tmp_path))
    source_cache = pd.DataFrame(
        {
            "index": ["a", "b", "c"],
            "answer": pd.Series([3, None, " 4 "], dtype=object),
        }
    )
    monkeypatch.setattr(
        local_vqa,
        "_load_cached_with_images",
        lambda path: source_cache if path == tmp_path / "CountQA.tsv" else None,
    )
    (tmp_path / "CountQA.tsv").write_text("offline cache\n", encoding="utf-8")
    countqa = object.__new__(local_vqa.CountQA)
    cached = countqa.load_data("CountQA")
    assert cached is not source_cache
    assert cached["answer"].tolist() == ["3", "", "4"]
    assert source_cache["answer"].tolist() == [3, None, " 4 "]
    assert local_vqa._extract_last_int(r"reasoning \boxed{12}") == 12

    countqa.dataset_name = "CountQA"
    countqa_prompt = countqa.build_prompt(
        {"question": "How many?", "image_path": "offline.png"}
    )
    assert countqa_prompt[-1]["value"].endswith(
        "Put the final answer inside \\boxed{}."
    )
    assert "Put only" not in countqa_prompt[-1]["value"]


def test_reviewed_post_run_runtime_patches_are_explicit_and_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = (SCRIPTS_ROOT / "start_vllm_endpoint_pool.sh").read_text(
        encoding="utf-8"
    )
    assert '--mm-processor-kwargs "${MM_PROCESSOR_KWARGS}"' in launcher
    assert '--generation-config "${GENERATION_CONFIG}"' in launcher
    assert "sitecustomize" not in launcher
    assert not (SCRIPTS_ROOT / "vllm_sitecustomize").exists()

    archive = _script("trace_eval_hf_archive")
    calls: list[dict[str, object]] = []

    class FakeLedger:
        @staticmethod
        def rows(statuses=None):
            del statuses
            return []

    class FakeArchiveDaemon:
        def __init__(self, **kwargs):
            calls.append(dict(kwargs))
            self.ledger = FakeLedger()

        @staticmethod
        def build_ready():
            return 2

    monkeypatch.setattr(archive, "ArchiveDaemon", FakeArchiveDaemon)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    assert archive.main(["--spool-root", str(tmp_path), "build"]) == 0
    assert len(calls) == 1
    assert calls[0]["token"] is None
