from __future__ import annotations

from copy import deepcopy
import io
import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile
import zipfile

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "check_public_release.py"
)
SPEC = importlib.util.spec_from_file_location("check_public_release", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
release_check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release_check)


def test_sensitive_text_scan_detects_credentials_and_machine_paths() -> None:
    fake_hf_token = "hf_" + ("a" * 32)
    local_path = "/" + "home/alice/trace/checkpoint"
    findings = release_check.scan_text_for_sensitive_values(
        f"TOKEN={fake_hf_token}\ncheckpoint={local_path}\n"
    )

    assert findings == [
        (1, "Hugging Face token"),
        (2, f"absolute machine path {local_path!r}"),
    ]


def test_sensitive_text_scan_detects_wandb_key_contexts() -> None:
    fake_key = "a" * 40
    for text in (
        f"WANDB_API_KEY={fake_key}",
        f'wandb.login(key="{fake_key}")',
        f'{{"api_key": "{fake_key}"}}',
    ):
        assert release_check.scan_text_for_sensitive_values(text) == [
            (1, "Weights & Biases API key")
        ]


def test_sensitive_text_scan_detects_checkpoint_roots() -> None:
    for root in ("mnt", "data", "scratch", "opt"):
        local_path = "/" + root + "/trace/checkpoint"
        assert release_check.scan_text_for_sensitive_values(local_path) == [
            (1, f"absolute machine path {local_path!r}")
        ]


def test_sensitive_text_scan_does_not_treat_source_urls_as_machine_paths() -> None:
    assert (
        release_check.scan_text_for_sensitive_values(
            "https://example.test/project/src/home/icon.svg"
        )
        == []
    )


def test_forbidden_tracked_path_scan_is_component_aware() -> None:
    findings = release_check.find_forbidden_tracked_paths(
        [
            "docs/tokenization.md",
            "src/trace_tasks/token_counter.py",
            "artifacts/" + "hf-token.txt",
            "rlvr/evaluation/.work/status.json",
            "build/output.txt",
            "results/run.sqlite3",
        ]
    )

    assert findings == [
        "artifacts/" + "hf-token.txt",
        "rlvr/evaluation/.work/status.json",
        "build/output.txt",
        "results/run.sqlite3",
    ]


def test_historical_dataset_revision_is_rejected_only_in_markdown(
    tmp_path: Path,
) -> None:
    revision = release_check.HISTORICAL_DATASET_REVISION
    markdown = tmp_path / "docs" / "guide.md"
    markdown.parent.mkdir()
    markdown.write_text(f"Load maveryn/trace@{revision}.\n", encoding="utf-8")
    receipt = tmp_path / "metadata" / "receipt.json"
    receipt.parent.mkdir()
    receipt.write_text(f'{{"original_revision": "{revision}"}}\n', encoding="utf-8")

    release_check.check_repository_hygiene(tmp_path, ["metadata/receipt.json"])
    with pytest.raises(
        release_check.ReleaseCheckError,
        match="historical dataset revision appears in public Markdown",
    ):
        release_check.check_repository_hygiene(
            tmp_path,
            ["docs/guide.md", "metadata/receipt.json"],
        )


def test_release_recipe_gate_rejects_every_reviewed_field_drift() -> None:
    contract = release_check._expected_release_recipe_contract()
    fields = contract["fields"]
    splits = contract["splits"]
    source_input_paths = contract["source_input_paths"]
    assert isinstance(fields, dict)
    assert isinstance(splits, list)
    assert isinstance(source_input_paths, list)
    valid = {
        **deepcopy(fields),
        "source": {"source_input_paths": deepcopy(source_input_paths)},
        "splits": deepcopy(splits),
    }
    release_check._check_release_dataset_recipe(valid)

    for field in fields:
        mutated = deepcopy(valid)
        mutated[field] = None
        with pytest.raises(
            release_check.ReleaseCheckError,
            match="recipe changed unexpectedly",
        ):
            release_check._check_release_dataset_recipe(mutated)

    for split_index, expected_split in enumerate(splits):
        assert isinstance(expected_split, dict)
        for field in expected_split:
            mutated = deepcopy(valid)
            mutated["splits"][split_index][field] = None
            with pytest.raises(
                release_check.ReleaseCheckError,
                match="recipe changed unexpectedly",
            ):
                release_check._check_release_dataset_recipe(mutated)

    mutated_source = deepcopy(valid)
    mutated_source["source"]["source_input_paths"] = []
    with pytest.raises(
        release_check.ReleaseCheckError,
        match="recipe changed unexpectedly",
    ):
        release_check._check_release_dataset_recipe(mutated_source)


def test_artifact_content_check_accepts_complete_minimal_archives(
    tmp_path: Path,
) -> None:
    expected = {"trace_tasks/__init__.py", "trace_tasks/resources/prompt.json"}
    expected_checkout = {
        "docs/README.md",
        "examples/example.py",
        "scripts/release_check.py",
    }
    wheel = tmp_path / "trace_tasks-0.1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in expected:
            archive.writestr(name, "content")
        archive.writestr(
            "trace_tasks-0.1.dist-info/entry_points.txt",
            "[console_scripts]\n"
            "trace-list = trace_tasks.cli:list_main\n"
            "trace-generate = trace_tasks.cli:generate_main\n"
            "trace-validate = trace_tasks.cli:validate_main\n"
            "trace-export = trace_tasks.cli:export_main\n",
        )

    sdist = tmp_path / "trace_tasks-0.1.tar.gz"
    required = (
        {
            "CITATION.cff",
            "LICENSE",
            "MANIFEST.in",
            "NOTICE",
            "README.md",
            "THIRD_PARTY_NOTICES.md",
            "constraints/compat-py314.txt",
            "constraints/release.txt",
            "mkdocs.yml",
            "pyproject.toml",
        }
        | {f"src/{path}" for path in expected}
        | expected_checkout
    )
    with tarfile.open(sdist, "w:gz") as archive:
        for name in sorted(required):
            payload = b"content"
            info = tarfile.TarInfo(f"trace_tasks-0.1/{name}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

    release_check.check_built_artifact_contents(
        wheel,
        sdist,
        expected,
        expected_checkout,
    )


def test_artifact_content_check_rejects_untracked_grafted_file(
    tmp_path: Path,
) -> None:
    expected = {"trace_tasks/__init__.py"}
    expected_checkout = {"docs/README.md"}
    wheel = tmp_path / "trace_tasks-0.1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("trace_tasks/__init__.py", "content")
        archive.writestr(
            "trace_tasks-0.1.dist-info/entry_points.txt",
            "[console_scripts]\n"
            "trace-list = trace_tasks.cli:list_main\n"
            "trace-generate = trace_tasks.cli:generate_main\n"
            "trace-validate = trace_tasks.cli:validate_main\n"
            "trace-export = trace_tasks.cli:export_main\n",
        )

    sdist = tmp_path / "trace_tasks-0.1.tar.gz"
    required = {
        "CITATION.cff",
        "LICENSE",
        "MANIFEST.in",
        "NOTICE",
        "README.md",
        "THIRD_PARTY_NOTICES.md",
        "constraints/compat-py314.txt",
        "constraints/release.txt",
        "docs/README.md",
        "mkdocs.yml",
        "pyproject.toml",
        "scripts/local-secret.txt",
        "src/trace_tasks/__init__.py",
    }
    with tarfile.open(sdist, "w:gz") as archive:
        for name in sorted(required):
            payload = b"content"
            info = tarfile.TarInfo(f"trace_tasks-0.1/{name}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

    with pytest.raises(
        release_check.ReleaseCheckError,
        match="untracked files from grafted release directories",
    ):
        release_check.check_built_artifact_contents(
            wheel,
            sdist,
            expected,
            expected_checkout,
        )


def test_artifact_content_check_rejects_non_regular_grafted_member(
    tmp_path: Path,
) -> None:
    expected = {"trace_tasks/__init__.py"}
    expected_checkout = {"docs/README.md"}
    wheel = tmp_path / "trace_tasks-0.1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("trace_tasks/__init__.py", "content")
        archive.writestr(
            "trace_tasks-0.1.dist-info/entry_points.txt",
            "[console_scripts]\n"
            "trace-list = trace_tasks.cli:list_main\n"
            "trace-generate = trace_tasks.cli:generate_main\n"
            "trace-validate = trace_tasks.cli:validate_main\n"
            "trace-export = trace_tasks.cli:export_main\n",
        )

    sdist = tmp_path / "trace_tasks-0.1.tar.gz"
    required = {
        "CITATION.cff",
        "LICENSE",
        "MANIFEST.in",
        "NOTICE",
        "README.md",
        "THIRD_PARTY_NOTICES.md",
        "constraints/compat-py314.txt",
        "constraints/release.txt",
        "docs/README.md",
        "mkdocs.yml",
        "pyproject.toml",
        "src/trace_tasks/__init__.py",
    }
    with tarfile.open(sdist, "w:gz") as archive:
        for name in sorted(required):
            payload = b"content"
            info = tarfile.TarInfo(f"trace_tasks-0.1/{name}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        link = tarfile.TarInfo("trace_tasks-0.1/docs/latest")
        link.type = tarfile.SYMTYPE
        link.linkname = "README.md"
        archive.addfile(link)

    with pytest.raises(
        release_check.ReleaseCheckError,
        match="non-regular members in grafted release directories",
    ):
        release_check.check_built_artifact_contents(
            wheel,
            sdist,
            expected,
            expected_checkout,
        )


@pytest.mark.parametrize(
    ("wheel_extra", "sdist_extra", "message"),
    [
        ("rlvr/train.py", None, "wheel contains clone-only RLVR files"),
        (None, "rlvr/train.py", "sdist contains clone-only RLVR files"),
    ],
)
def test_artifact_content_check_rejects_clone_only_rlvr_files(
    tmp_path: Path,
    wheel_extra: str | None,
    sdist_extra: str | None,
    message: str,
) -> None:
    expected = {"trace_tasks/__init__.py"}
    wheel = tmp_path / "trace_tasks-0.1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("trace_tasks/__init__.py", "content")
        archive.writestr(
            "trace_tasks-0.1.dist-info/entry_points.txt",
            "[console_scripts]\n"
            "trace-list = trace_tasks.cli:list_main\n"
            "trace-generate = trace_tasks.cli:generate_main\n"
            "trace-validate = trace_tasks.cli:validate_main\n"
            "trace-export = trace_tasks.cli:export_main\n",
        )
        if wheel_extra is not None:
            archive.writestr(wheel_extra, "excluded content")

    sdist = tmp_path / "trace_tasks-0.1.tar.gz"
    required = {
        "CITATION.cff",
        "LICENSE",
        "MANIFEST.in",
        "NOTICE",
        "README.md",
        "THIRD_PARTY_NOTICES.md",
        "constraints/compat-py314.txt",
        "constraints/release.txt",
        "pyproject.toml",
        "src/trace_tasks/__init__.py",
    }
    if sdist_extra is not None:
        required.add(sdist_extra)
    with tarfile.open(sdist, "w:gz") as archive:
        for name in sorted(required):
            payload = b"content"
            info = tarfile.TarInfo(f"trace_tasks-0.1/{name}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

    with pytest.raises(release_check.ReleaseCheckError, match=message):
        release_check.check_built_artifact_contents(wheel, sdist, expected, set())


def test_optional_rlvr_check_is_a_noop_when_tree_is_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        release_check,
        "_run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("command ran")),
    )

    assert release_check.check_optional_rlvr_release(tmp_path) is False


def test_optional_rlvr_check_runs_both_profiles_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_script = tmp_path / "rlvr" / "train.py"
    train_script.parent.mkdir()
    train_script.write_text("# test checker\n", encoding="utf-8")
    calls: list[tuple[list[object], dict[str, str]]] = []

    def fake_run(command, *, cwd, env):
        config_id = str(command[-1])
        calls.append((list(command), dict(env)))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"config_id": config_id, "status": "ok"}),
            stderr="",
        )

    monkeypatch.setattr(release_check, "_run", fake_run)

    assert release_check.check_optional_rlvr_release(tmp_path) is True
    assert [str(command[-1]) for command, _ in calls] == list(
        release_check.RLVR_CONFIG_IDS
    )
    for _, env in calls:
        assert env["HF_HUB_OFFLINE"] == "1"
        assert env["TRANSFORMERS_OFFLINE"] == "1"
        assert env["WANDB_MODE"] == "offline"
        assert env["CUDA_VISIBLE_DEVICES"] == ""


def test_optional_rlvr_check_propagates_checker_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_script = tmp_path / "rlvr" / "train.py"
    train_script.parent.mkdir()
    train_script.write_text("# test checker\n", encoding="utf-8")

    def fail(*args, **kwargs):
        raise release_check.ReleaseCheckError("RLVR checker failed")

    monkeypatch.setattr(release_check, "_run", fail)

    with pytest.raises(release_check.ReleaseCheckError, match="RLVR checker failed"):
        release_check.check_optional_rlvr_release(tmp_path)


def test_optional_rlvr_check_validates_curated_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_script = tmp_path / "rlvr" / "train.py"
    train_script.parent.mkdir()
    train_script.write_text("# test checker\n", encoding="utf-8")
    validator = (
        tmp_path
        / "rlvr"
        / "evaluation"
        / "scripts"
        / "validate_release_inputs.py"
    )
    validator.parent.mkdir(parents=True)
    validator.write_text("# test validator\n", encoding="utf-8")
    commands: list[list[object]] = []

    def fake_run(command, *, cwd, env):
        commands.append(list(command))
        if Path(command[1]) == validator:
            payload = {"status": "ok", "suite_id": "trace_eval_v1"}
        else:
            payload = {"status": "ok", "config_id": str(command[-1])}
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(payload),
            stderr="",
        )

    monkeypatch.setattr(release_check, "_run", fake_run)

    assert release_check.check_optional_rlvr_release(tmp_path) is True
    assert [Path(command[1]) for command in commands] == [
        train_script,
        train_script,
        validator,
    ]


def test_source_only_skips_build_and_installed_smoke(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(release_check, "_git_tracked_files", lambda root: [])
    monkeypatch.setattr(release_check, "check_registry", lambda: None)
    monkeypatch.setattr(
        release_check,
        "check_inventory_and_source_manifest",
        lambda root: None,
    )
    monkeypatch.setattr(release_check, "check_representative_generation", lambda: None)
    monkeypatch.setattr(
        release_check,
        "check_repository_hygiene",
        lambda root, tracked: None,
    )
    monkeypatch.setattr(
        release_check,
        "check_package_sources_are_tracked",
        lambda root, tracked: set(),
    )
    monkeypatch.setattr(
        release_check,
        "build_artifacts",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("build ran")),
    )

    release_check.run_release_checks(
        repo_root=tmp_path,
        source_only=True,
        constraints_path=None,
    )


def test_package_only_builds_and_checks_installed_cli_without_frozen_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    constraints = tmp_path / "constraints" / "compat-py314.txt"
    constraints.parent.mkdir()
    constraints.write_text("numpy==2.3.4\n", encoding="utf-8")
    calls: list[str] = []

    monkeypatch.setattr(release_check, "_git_tracked_files", lambda root: [])
    monkeypatch.setattr(
        release_check,
        "check_inventory_and_source_manifest",
        lambda root: (_ for _ in ()).throw(AssertionError("inventory check ran")),
    )
    monkeypatch.setattr(
        release_check,
        "check_registry",
        lambda: (_ for _ in ()).throw(AssertionError("registry check ran")),
    )
    monkeypatch.setattr(
        release_check,
        "check_representative_generation",
        lambda: (_ for _ in ()).throw(AssertionError("source generation ran")),
    )
    monkeypatch.setattr(
        release_check,
        "check_repository_hygiene",
        lambda root, tracked: None,
    )
    monkeypatch.setattr(
        release_check,
        "check_package_sources_are_tracked",
        lambda root, tracked: {"trace_tasks/__init__.py"},
    )
    monkeypatch.setattr(
        release_check,
        "build_artifacts",
        lambda root, workspace: (
            calls.append("build") or workspace / "trace.whl",
            workspace / "trace.tar.gz",
        ),
    )
    monkeypatch.setattr(
        release_check,
        "check_built_artifact_contents",
        lambda *args: calls.append("artifacts"),
    )
    monkeypatch.setattr(
        release_check,
        "check_installed_cli",
        lambda *args: calls.append("installed-cli"),
    )

    release_check.run_release_checks(
        repo_root=tmp_path,
        source_only=False,
        constraints_path=constraints,
        package_only=True,
    )

    assert calls == ["build", "artifacts", "installed-cli"]


def test_default_constraints_are_required(tmp_path: Path) -> None:
    with pytest.raises(
        release_check.ReleaseCheckError,
        match="default constraints file does not exist",
    ):
        release_check._resolve_constraints(tmp_path, None)
