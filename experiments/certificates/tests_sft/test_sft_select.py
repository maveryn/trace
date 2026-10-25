"""The pre-registered stop rule (DESIGN 7.2) and jobs/sft_select.py's outputs (pure; synthetic run roots)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

RULE = {"min_F": 0.95, "max_abs_dA": 0.015, "max_abs_dM": 0.02}


def _curve(*points):
    return [{"step": s, "A": a, "M": m, "F": f} for s, a, m, f in points]


def test_first_qualifying_checkpoint(select):
    res = select.apply_rule(_curve((94, 0.40, 0.50, 0.99), (188, 0.45, 0.60, 0.99), (282, 0.455, 0.61, 0.99), (376, 0.456, 0.612, 0.99)), **RULE)
    assert res["selected_step"] == 282 and res["outcome"] == "rule"
    by = {p["step"]: p for p in res["curve"]}
    assert by[94]["reasons"] == ["no previous evaluation"]  # the first save cannot qualify
    assert not by[188]["qualifies"] and by[282]["qualifies"] and by[376]["qualifies"]
    assert by[282]["dA"] == pytest.approx(0.005) and by[282]["dM"] == pytest.approx(0.01)


def test_format_gate_and_inclusive_thresholds(select):
    # F below 0.95 blocks; |dA| == 0.015 and |dM| == 0.02 (to float noise) pass
    res = select.apply_rule(_curve((94, 0.40, 0.50, 0.90), (188, 0.40, 0.50, 0.949), (282, 0.415, 0.52, 0.95)), **RULE)
    assert res["selected_step"] == 282
    res = select.apply_rule(_curve((94, 0.40, 0.50, 0.99), (188, 0.4151, 0.50, 0.99), (282, 0.40, 0.5201, 0.99)), **RULE)
    assert res["outcome"] == "fallback_last" and res["selected_step"] == 282
    assert [p["qualifies"] for p in res["curve"]] == [False, False, False]


def test_drops_count_like_rises(select):
    res = select.apply_rule(_curve((94, 0.50, 0.60, 0.99), (188, 0.47, 0.60, 0.99), (282, 0.47, 0.57, 0.99), (376, 0.47, 0.57, 0.99)), **RULE)
    assert res["selected_step"] == 376


def test_fallback_is_the_last_save_and_order_does_not_matter(select):
    pts = _curve((564, 0.5, 0.5, 0.5), (94, 0.1, 0.1, 0.5), (282, 0.3, 0.3, 0.5))
    res = select.apply_rule(pts, **RULE)
    assert res["selected_step"] == 564 and res["outcome"] == "fallback_last"
    assert [p["step"] for p in res["curve"]] == [94, 282, 564]


def test_smoke_single_save(select):
    res = select.apply_rule(_curve((20, 1.0, 1.0, 1.0)), **RULE)
    assert res["selected_step"] == 20 and res["outcome"] == "fallback_last"


def test_bad_inputs(select):
    with pytest.raises(select.SelectError):
        select.apply_rule([], **RULE)
    with pytest.raises(select.SelectError):
        select.apply_rule(_curve((94, 0.1, 0.1, 1.0), (94, 0.1, 0.1, 1.0)), **RULE)


# ------------------------------------------------------------------------------- end to end


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mk(path: Path) -> Path:
    path.mkdir()
    return path


def _fake_root(tmp_path: Path, metrics, *, smoke=False) -> Path:
    """A finished run root as jobs/sft.sh leaves it, with synthetic dev scores."""
    root = tmp_path / "run"
    steps = [s for s, *_ in metrics]
    plan = {"model": {"repo_id": "Qwen/Qwen2.5-VL-3B-Instruct", "revision": "r", "tag": "qwen2.5-vl-3b", "architecture": "X"},
            "config_id": "cfg", "math_sha256": "m" * 64, "save_steps": steps, "smoke": smoke, "n_train": 6000,
            "dev_rows_per_task": 8 if smoke else None, "select": {**RULE, "fallback": "last"}, "eval": {"temperature": 0.0}}
    root.mkdir()
    (root / "plan.json").write_text(json.dumps(plan))
    (root / "check.json").write_text(json.dumps({"ok": True, "data": {"dev_rows": 512, "files": {"sft/sft.jsonl": "a" * 64}}, "config": "c",
                                                 "config_sha256": "c" * 64, "base": {"dir": "/base", "files": {}}}))
    (root / "train_done.json").write_text(json.dumps({"plan": plan, "finished": "now", "provenance": {"git_head": "h", "env_lock_sha256": "e"}}))
    (root / "runs.jsonl").write_text(json.dumps({"world_size": 2, "resume_step": 0}) + "\n")
    n = 32 if smoke else 512
    for step, a, m, f in metrics:
        name = f"step_{step:06d}"
        ck = root / "ckpt" / name
        ck.mkdir(parents=True)
        (ck / "model.safetensors").write_bytes(b"w" + bytes([step % 256]))
        files = {"model.safetensors": {"sha256": _sha(ck / "model.safetensors"), "bytes": 2}}
        (ck / "sft_checkpoint.json").write_text(json.dumps({"step": step, "epoch": step / 188, "math_sha256": plan["math_sha256"], "files": files}))
        run = root / "dev" / name
        (run / "cert").mkdir(parents=True)
        (run / "responses.jsonl").write_text(json.dumps({"instance_id": "x", "step": step}) + "\n")
        (run / "run_config.json").write_text(json.dumps({"checkpoint": {"files": files}}))
        (run / "cert" / "summary.json").write_text(json.dumps({"all": [{"n": n, "A": a, "M": m, "F": f, "G": 0.5, "joint": 0.1, "tokens": {"mean": 150.0}}]}))
        checks = {"dataset_frozen": True, "allow_unfrozen": False, "prompt_check": True, "prompts_checked": n, "allow_settings_mismatch": True,
                  "settings_problems": ["decoding 'greedy', expected the 'trace' preset"]}
        (run / "cert" / "score_receipt.json").write_text(json.dumps({"responses": {"sha256": _sha(run / "responses.jsonl")}, "checks": checks,
                                                                      "outputs": {"cert_scores.jsonl": {"sha256": "s" * 64}}, "scorer_version": "v"}))
    return root


def test_select_writes_symlink_curve_and_receipt(select, tmp_path, capsys):
    root = _fake_root(tmp_path, [(94, 0.40, 0.50, 0.99), (188, 0.405, 0.51, 0.99), (282, 0.41, 0.52, 0.99)])
    assert select.main(["--root", str(root), "--data", str(tmp_path)]) == 0
    assert os.readlink(root / "selected") == "ckpt/step_000188"
    assert (root / "selected" / "sft_checkpoint.json").exists()
    receipt = json.loads((root / "receipt.json").read_text())
    assert receipt["selected"]["step"] == 188 and receipt["outcome"] == "rule"
    assert [p["step"] for p in receipt["dev_eval"]["curve"]] == [94, 188, 282]
    assert receipt["data"]["files_sha256"] == {"sft/sft.jsonl": "a" * 64}
    assert receipt["model"]["revision"] == "r" and receipt["git"]["trace"]["head"] == "h"
    assert receipt["env"]["train_env_lock_sha256"] == "e" and receipt["selected"]["files"]["model.safetensors"]["sha256"]
    curve = json.loads((root / "dev_curve.json").read_text())
    assert curve["selected_step"] == 188
    # rerun replaces the link atomically
    assert select.main(["--root", str(root), "--data", str(tmp_path)]) == 0
    assert os.readlink(root / "selected") == "ckpt/step_000188"
    assert "selected step_000188 (rule)" in capsys.readouterr().out


def test_select_refuses_stale_or_foreign_evaluations(select, tmp_path):
    root = _fake_root(tmp_path, [(20, 1.0, 1.0, 1.0)], smoke=True)
    assert select.main(["--root", str(root), "--data", str(tmp_path)]) == 0
    assert os.readlink(root / "selected") == "ckpt/step_000020"
    # responses changed after scoring -> incomplete (rerun scores again)
    (root / "dev" / "step_000020" / "responses.jsonl").write_text("{}\n")
    assert select.main(["--root", str(root), "--data", str(tmp_path)]) == 1
    # generated from other weights -> configuration error
    root2 = _fake_root(_mk(tmp_path / "b"), [(20, 1.0, 1.0, 1.0)], smoke=True)
    (root2 / "dev" / "step_000020" / "run_config.json").write_text(json.dumps({"checkpoint": {"files": {"model.safetensors": {"sha256": "0"}}}}))
    assert select.main(["--root", str(root2), "--data", str(tmp_path)]) == 4
    # a missing evaluation is incomplete, a wrong row count too
    root3 = _fake_root(_mk(tmp_path / "c"), [(20, 1.0, 1.0, 1.0)], smoke=True)
    (root3 / "dev" / "step_000020" / "cert" / "summary.json").unlink()
    assert select.main(["--root", str(root3), "--data", str(tmp_path)]) == 1
    root4 = _fake_root(_mk(tmp_path / "d"), [(20, 1.0, 1.0, 1.0)], smoke=False)
    assert select.main(["--root", str(root4), "--data", str(tmp_path), "--expected-rows", "511"]) == 1
    # a scorer override other than the greedy decoding (here: another system prompt) is refused
    root5 = _fake_root(_mk(tmp_path / "e"), [(20, 1.0, 1.0, 1.0)], smoke=True)
    rec = root5 / "dev" / "step_000020" / "cert" / "score_receipt.json"
    payload = json.loads(rec.read_text())
    payload["checks"]["settings_problems"].append("system_prompt_sha256 aaaa != bbbb expected for cert_px")
    rec.write_text(json.dumps(payload))
    assert select.main(["--root", str(root5), "--data", str(tmp_path)]) == 4
    payload["checks"]["settings_problems"].pop()
    payload["checks"]["prompt_check"] = False
    rec.write_text(json.dumps(payload))
    assert select.main(["--root", str(root5), "--data", str(tmp_path)]) == 4
