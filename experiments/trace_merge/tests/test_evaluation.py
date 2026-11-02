"""Evaluation planning and lock tests: task lines follow the configured matrix; the lock fails closed."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trace_merge import config, evaluation, grouping  # noqa: E402


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    monkeypatch.delenv("TVM_RUNS", raising=False)
    c = config.load_config(work_root=tmp_path / "work", protocol_clone=tmp_path / "clone")
    config.ensure_work_tree(c)
    return c


@pytest.fixture()
def lockable(cfg):
    """Every input the lock reads: split manifest, metric registry, merge manifests, built and registered views."""

    config.write_json(cfg.work("splits", "split_manifest.json"),
                      {"rule": grouping.split_rule(cfg.raw["split"]),
                       "row_lists": {"confirmation": {"row_list_sha256": "a" * 64}}})
    config.write_json(cfg.work("manifests", "metric_registry.json"), {"accepted_benchmarks": ["b", "a"]})
    for mid in cfg.merges:
        config.write_json(cfg.work("merges", mid, "merge_manifest.json"),
                          {"content_revision": f"sha256set:{mid}", "recipe_sha256": mid})
    for _, slug, _, _ in evaluation.tier_runs(cfg, "confirmation"):
        config.write_json(cfg.work("views", slug, "view_manifest.json"), {"slug": slug})
        config.write_json(cfg.work("views", slug, ".trace_model_revision.json"), {"immutable_revision": f"r:{slug}"})
    return cfg


@pytest.fixture()
def design(tmp_path):
    p = tmp_path / "DESIGN.md"
    p.write_text("# frozen\n\n## Amendments\n\n**A1 (x).** first.\n")
    return p


def _start_confirmation(cfg, slug="m050-p1", name="plan.seed42.json", root=None):
    run = (root / slug / "confirmation") if root else cfg.work("runs", slug, "confirmation")
    run.mkdir(parents=True, exist_ok=True)
    (run / name).write_text("{}")
    return run


def test_config_policy_switches_fail_closed(tmp_path):
    raw = (config.DEFAULT_CONFIG).read_text()
    bad = tmp_path / "bad.yaml"
    bad.write_text(raw.replace("allow_training: false", "allow_training: true"))
    with pytest.raises(config.ConfigError):
        config.load_config(bad)
    bad.write_text(raw.replace("M050: {coefficients: {V: 0.5, T: 0.5}", "M050: {coefficients: {V: 0.25, T: 0.75}"))
    with pytest.raises(config.ConfigError):
        config.load_config(bad)


def test_work_root_is_required():
    c = config.load_config(work_root=None) if not __import__("os").environ.get("TRACE_MERGE_WORK_ROOT") else None
    if c is not None:
        with pytest.raises(config.ConfigError):
            c.require_work_root()


def test_tier_runs_follow_matrix(cfg):
    conf = evaluation.tier_runs(cfg, "confirmation")
    assert len(conf) == 4 * 3
    assert {label for label, *_ in conf} == {"V@P1", "M050@P1", "CV050@P1", "CT050@P1"}
    assert all(subset == "confirmation" for _, _, subset, _ in conf)
    dev = evaluation.tier_runs(cfg, "development")
    assert {s for _, s, _, _ in dev} >= {"vero-p1", "m050-p1", "qwen25vl7b-p2"}


def test_task_lines_one_per_model_seeds_in_order(cfg, tmp_path):
    lines = evaluation.task_lines(cfg, "confirmation", tmp_path / "env.sh", tmp_path / "jobs")
    assert len(lines) == 4
    assert all(line.endswith(" confirmation:42 confirmation:43 confirmation:44") for line in lines)


def test_task_lines_smoke_only_for_smoke_matrix_models(cfg, tmp_path):
    smoke = {slug for _, slug, _, _ in evaluation.tier_runs(cfg, "smoke")}
    for tier, first in (("development", "development_screen:42"), ("confirmation", "confirmation:42")):
        lines = evaluation.task_lines(cfg, tier, tmp_path / "env.sh", tmp_path / "jobs", with_smoke=True)
        assert len(lines) == len({slug for _, slug, _, _ in evaluation.tier_runs(cfg, tier)})
        for line in lines:
            slug, *specs = line.split("run_tiers.sh ")[1].split()
            assert specs[0] == ("smoke:42" if slug in smoke else first), line
            assert sum(s.startswith("smoke:") for s in specs) == (slug in smoke)
    assert not any("qwen25vl7b-p2 smoke" in line
                   for line in evaluation.task_lines(cfg, "development", tmp_path, tmp_path, with_smoke=True))


def test_lock_refuses_after_confirmation_scores(lockable, design):
    cfg = lockable
    run = _start_confirmation(cfg, name="validation.seed42.json")
    with pytest.raises(RuntimeError):
        evaluation.lock_selection(cfg, design)
    (run / "validation.seed42.json").unlink()
    lock = evaluation.lock_selection(cfg, design)
    assert lock["selected_candidates"] == ["CT050@P1", "CV050@P1", "M050@P1", "V@P1"]
    assert lock["accepted_benchmarks"] == ["a", "b"] and lock["design_amendments_at_lock"] == ["A1"]
    assert lock["runner_code_sha256"] and lock["split_config_sha256"]
    assert evaluation.lock_selection(cfg, design)["locked_at"] == lock["locked_at"]  # idempotent
    design.write_text("changed")
    with pytest.raises(RuntimeError):
        evaluation.lock_selection(cfg, design)


@pytest.mark.parametrize("name", ["plan.seed42.json", "rows.confirmation.json", "campaign/seed_42/predictions.jsonl"])
def test_lock_refuses_once_confirmation_generation_has_started(lockable, design, name):
    run = lockable.work("runs", "m050-p1", "confirmation")
    (run / name).parent.mkdir(parents=True, exist_ok=True)
    (run / name).write_text("{}")
    with pytest.raises(RuntimeError, match="already started"):
        evaluation.lock_selection(lockable, design)
    with pytest.raises(RuntimeError, match="already started"):
        evaluation.lock_selection(lockable, design, force=True)


def test_lock_refuses_when_any_confirmation_run_exists(lockable, design, tmp_path, monkeypatch):
    lockable.work("runs", "vero-p1", "confirmation").mkdir(parents=True)  # an empty directory is not an output
    assert evaluation.confirmation_outputs(lockable) == {}
    run = _start_confirmation(lockable, slug="m050-p2")  # a view outside the matrix on the confirmation partition
    with pytest.raises(RuntimeError, match="already started"):
        evaluation.lock_selection(lockable, design)
    (run / "plan.seed42.json").unlink()
    monkeypatch.setenv("TVM_RUNS", str(tmp_path / "elsewhere"))  # the runner's run-root override
    _start_confirmation(lockable, root=tmp_path / "elsewhere")
    assert list(evaluation.confirmation_outputs(lockable)) == ["TVM_RUNS:m050-p1/confirmation"]
    with pytest.raises(RuntimeError, match="already started"):
        evaluation.lock_selection(lockable, design)


def test_force_relock_keeps_previous_lock_and_never_follows_outputs(lockable, design):
    first = evaluation.lock_selection(lockable, design)
    design.write_text(design.read_text() + "\n**A2 (y).** second.\n")
    second = evaluation.lock_selection(lockable, design, force=True)
    prev = second["previous_locks"]
    assert len(prev) == 1 and prev[0]["design_sha256"] == first["design_sha256"]
    archived = json.loads(lockable.work("reports", prev[0]["file"]).read_text())
    assert archived == first
    _start_confirmation(lockable)
    design.write_text(design.read_text() + "\n**A3 (z).** written after confirmation started.\n")
    with pytest.raises(RuntimeError, match="already started"):
        evaluation.lock_selection(lockable, design, force=True)
    assert json.loads(lockable.work("reports", evaluation.LOCK_NAME).read_text()) == second


def test_lock_requires_built_and_registered_views(lockable, design):
    marker = lockable.work("views", "ct050-p1", ".trace_model_revision.json")
    marker.unlink()
    with pytest.raises(RuntimeError, match="not built and registered"):
        evaluation.lock_selection(lockable, design)
    config.write_json(marker, {"immutable_revision": "r"})
    lockable.work("views", "vero-p1", "view_manifest.json").unlink()
    with pytest.raises(RuntimeError, match="not built and registered"):
        evaluation.lock_selection(lockable, design)


def test_lock_requires_registry_and_matching_split_rule(lockable, design):
    split = lockable.work("splits", "split_manifest.json")
    m = json.loads(split.read_text())
    config.write_json(split, {**m, "rule": m["rule"].replace("271828", "1")})
    with pytest.raises(RuntimeError, match="split"):
        evaluation.lock_selection(lockable, design)
    config.write_json(split, m)
    lockable.work("manifests", "metric_registry.json").unlink()
    with pytest.raises(FileNotFoundError):
        evaluation.lock_selection(lockable, design)


def test_verify_lock_allows_only_appended_amendments(lockable, design):
    with pytest.raises(RuntimeError, match="missing"):
        evaluation.verify_lock(lockable, design)
    evaluation.lock_selection(lockable, design)
    rep = evaluation.verify_lock(lockable, design)
    assert rep["ok"] and rep["post_lock_amendments"] == [] and rep["not_recorded"] == []
    frozen = design.read_text()
    design.write_text(frozen + "\n**A2 (later).** appended after the lock.\n")
    rep = evaluation.verify_lock(lockable, design)
    assert rep["ok"] and rep["amendments_at_lock"] == ["A1"] and rep["post_lock_amendments"] == ["A2"]
    design.write_text(frozen.replace("# frozen", "# edited") + "\n**A2 (later).** appended.\n")
    with pytest.raises(RuntimeError, match="design_sha256"):
        evaluation.verify_lock(lockable, design)
    design.write_text(frozen.replace("first.", "first, edited.") + "\n**A2 (later).** appended.\n")
    assert evaluation.verify_lock(lockable, design, strict=False)["mismatches"] == ["design_sha256"]


def test_verify_lock_reads_a_v1_lock_and_flags_changed_inputs(lockable, design, monkeypatch):
    lock = evaluation.lock_selection(lockable, design)
    v1 = {k: v for k, v in lock.items() if k not in {"design_amendments_at_lock", "split_config_sha256",
                                                      "metric_registry_sha256", "accepted_benchmarks",
                                                      "runner_code_sha256"}}
    config.write_json(lockable.work("reports", evaluation.LOCK_NAME), v1)
    rep = evaluation.verify_lock(lockable, design)
    assert rep["ok"] and rep["not_recorded"] == ["accepted_benchmarks", "design_amendments_at_lock",
                                                 "metric_registry_sha256", "runner_code_sha256", "split_config_sha256"]
    config.write_json(lockable.work("reports", evaluation.LOCK_NAME), lock)
    monkeypatch.setattr(evaluation, "runner_code", lambda: {**lock["runner_code_sha256"], "eval_subset.sh": "0" * 64})
    rep = evaluation.verify_lock(lockable, design)
    assert rep["ok"] and rep["runner_code_changed"] == ["eval_subset.sh"]
    config.write_json(lockable.work("manifests", "metric_registry.json"), {"accepted_benchmarks": ["a"]})
    assert evaluation.verify_lock(lockable, design, strict=False)["mismatches"] == ["metric_registry_sha256",
                                                                                     "accepted_benchmarks"]


def test_verify_lock_flags_confirmation_outputs_older_than_the_lock(lockable, design):
    lock = evaluation.lock_selection(lockable, design)
    run = _start_confirmation(lockable)
    assert evaluation.verify_lock(lockable, design)["ok"]
    t = __import__("datetime").datetime.fromisoformat(lock["locked_at"]).timestamp() - 3600
    os.utime(run / "plan.seed42.json", (t, t))
    rep = evaluation.verify_lock(lockable, design, strict=False)
    assert rep["mismatches"] == ["confirmation_outputs_before_lock"]
    assert list(rep["confirmation_outputs_before_lock"]) == ["m050-p1/confirmation"]
