#!/usr/bin/env python
"""Pre-registered SFT stop rule (DESIGN 7.2) and the M2 receipt.

    python experiments/certificates/jobs/sft_select.py --root $TRACE_CERT_RUNS/sft/<tag> --data $TRC_DATA

Reads what jobs/sft.sh left under ROOT:
  plan.json                      sft.py plan (save steps, stop-rule thresholds, dev rows)
  check.json                     sft.py check (data/receipt sha256s, base snapshot, EasyR1 token identity)
  train_done.json, runs.jsonl    training summary and every (re)launch (world size, resume step, env)
  ckpt/step_XXXXXX/              saved HF dirs (sft_checkpoint.json: step, epoch, file sha256s)
  dev/step_XXXXXX/               greedy E1-dev run (responses.jsonl, run_config.json) and its score
                                 (cert/summary.json, cert/score_receipt.json)
Rule: walking the saves in step order, pick the FIRST checkpoint with dev F >= min_F,
|A_k - A_{k-1}| <= max_abs_dA and |M_k - M_{k-1}| <= max_abs_dM against the previous evaluation
(the first save has no previous evaluation, so it cannot qualify); if none qualifies, take the last
save (the end of epoch 3; for --smoke the single save). A, M, F are the means over every scored
E1-dev row. Writes ROOT/selected (relative symlink to ckpt/step_XXXXXX, replaced atomically),
ROOT/dev_curve.json and ROOT/receipt.json. Exit 0 selected; 1 an evaluation is missing (rerun
sft.sh, it resumes); 4 inconsistent inputs.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

SELECT_VERSION = "cert-sft-select-v1"
RECEIPT_VERSION = "cert-sft-receipt-v1"
EPS = 1e-9  # thresholds are inclusive; guards 0.015 vs 0.015000000000000013
CURVE_KEYS = ("n", "A", "A_legacy", "G", "G_id", "G_shown", "M", "M_shown", "F", "joint", "valid_schema_rate",
              "envelope_failure_rate", "completion_rate", "truncation_rate", "within_2048_rate")


class SelectError(RuntimeError):
    def __init__(self, msg: str, code: int = 4):
        super().__init__(msg)
        self.code = code


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def step_name(step: int) -> str:
    return f"step_{int(step):06d}"


def dev_point(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """Pooled E1-dev metrics of one scored run (``summary["all"][0]`` of the certificate scorer)."""
    rows = summary.get("all") or []
    if len(rows) != 1:
        raise SelectError("summary.json must hold exactly one pooled row")
    agg = rows[0]
    out = {key: agg.get(key) for key in CURVE_KEYS}
    out["tokens_mean"] = (agg.get("tokens") or {}).get("mean")
    out["error_classes"] = agg.get("error_classes")
    for key in ("A", "M", "F"):
        if not isinstance(out.get(key), (int, float)):
            raise SelectError(f"summary has no numeric {key}")
    return out


def apply_rule(curve: Sequence[Mapping[str, Any]], *, min_F: float, max_abs_dA: float, max_abs_dM: float, fallback: str = "last") -> Dict[str, Any]:
    """The DESIGN 7.2 stop rule over ``curve`` (dicts with step, A, M, F; any order)."""
    if not curve:
        raise SelectError("no evaluated checkpoint")
    if fallback != "last":
        raise SelectError(f"unknown fallback {fallback!r}")
    points = sorted((dict(p) for p in curve), key=lambda p: int(p["step"]))
    if len({int(p["step"]) for p in points}) != len(points):
        raise SelectError("duplicate steps in the dev curve")
    selected = None
    for k, p in enumerate(points):
        reasons = []
        if float(p["F"]) < float(min_F) - EPS:
            reasons.append(f"F {float(p['F']):.4f} < {min_F}")
        if k == 0:
            p["dA"] = p["dM"] = None
            reasons.append("no previous evaluation")
        else:
            prev = points[k - 1]
            p["dA"] = float(p["A"]) - float(prev["A"])
            p["dM"] = float(p["M"]) - float(prev["M"])
            if abs(p["dA"]) > float(max_abs_dA) + EPS:
                reasons.append(f"|dA| {abs(p['dA']):.4f} > {max_abs_dA}")
            if abs(p["dM"]) > float(max_abs_dM) + EPS:
                reasons.append(f"|dM| {abs(p['dM']):.4f} > {max_abs_dM}")
        p["qualifies"] = not reasons
        p["reasons"] = reasons
        if selected is None and not reasons:
            selected = int(p["step"])
    outcome = "rule" if selected is not None else "fallback_last"
    if selected is None:
        selected = int(points[-1]["step"])
    return {
        "selected_step": selected,
        "outcome": outcome,
        "rule": {"min_F": float(min_F), "max_abs_dA": float(max_abs_dA), "max_abs_dM": float(max_abs_dM), "fallback": fallback,
                 "text": "first checkpoint with dev F >= min_F, |dA| <= max_abs_dA and |dM| <= max_abs_dM against the "
                         "previous evaluation; else the last save (end of epoch 3)"},
        "curve": points,
    }


def link_selected(root: Path, step: int) -> Path:
    """ROOT/selected -> ckpt/step_XXXXXX (relative), replaced atomically."""
    target = Path("ckpt") / step_name(step)
    if not (root / target / "sft_checkpoint.json").exists():
        raise SelectError(f"{root / target} is not a finished checkpoint")
    tmp = root / f".selected.tmp.{os.getpid()}"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    os.symlink(target, tmp)
    os.replace(tmp, root / "selected")
    return root / "selected"


def _git(path: Optional[str], *args: str) -> Optional[str]:
    if not path:
        return None
    try:
        return subprocess.run(["git", "-C", path, *args], capture_output=True, text=True, check=True, timeout=60).stdout.strip()
    except Exception:
        return None


def _provenance() -> Dict[str, Any]:
    try:
        from trace_tasks.experimental.certificates.export import code_provenance

        return dict(code_provenance())
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def collect(root: Path, plan: Mapping[str, Any], expected_rows: Optional[int]) -> List[Dict[str, Any]]:
    """One curve point per planned save; raises SelectError(code 1) if an evaluation is missing."""
    points = []
    for step in plan["save_steps"]:
        name = step_name(step)
        ckpt = root / "ckpt" / name
        run = root / "dev" / name
        if not (ckpt / "sft_checkpoint.json").exists():
            raise SelectError(f"checkpoint {ckpt} is missing", 1)
        if not (run / "cert" / "summary.json").exists() or not (run / "cert" / "score_receipt.json").exists():
            raise SelectError(f"dev evaluation of {name} is not scored ({run}/cert)", 1)
        info = read_json(ckpt / "sft_checkpoint.json")
        if int(info["step"]) != int(step) or info.get("math_sha256") != plan["math_sha256"]:
            raise SelectError(f"{ckpt}: step/config does not match plan.json")
        run_config = read_json(run / "run_config.json")
        want_files = {k: v.get("sha256") for k, v in (info.get("files") or {}).items()}
        got_files = {k: v.get("sha256") for k, v in ((run_config.get("checkpoint") or {}).get("files") or {}).items()}
        if want_files != got_files:
            raise SelectError(f"{run} was generated from other weights than {ckpt}")
        receipt = read_json(run / "cert" / "score_receipt.json")
        if receipt["responses"]["sha256"] != file_sha256(run / "responses.jsonl"):
            raise SelectError(f"{run}/cert is stale: responses.jsonl changed after scoring", 1)
        checks = receipt.get("checks")
        if checks is not None:  # scorer integrity checks (frozen dataset, prompts, DESIGN 7.1 settings)
            others = [p for p in checks.get("settings_problems") or [] if not str(p).startswith("decoding ")]
            if not checks.get("dataset_frozen") or checks.get("allow_unfrozen") or not checks.get("prompt_check") or others:
                raise SelectError(f"{run}/cert: scorer integrity checks not met (only the greedy decoding may differ): {checks}")
        summary = read_json(run / "cert" / "summary.json")
        point = {"step": int(step), "epoch": info.get("epoch"), **dev_point(summary)}
        if expected_rows is not None and int(point["n"] or 0) != int(expected_rows):
            raise SelectError(f"{name}: {point['n']} scored dev rows, want {expected_rows}", 1)
        point["responses_sha256"] = receipt["responses"]["sha256"]
        point["cert_scores_sha256"] = receipt["outputs"]["cert_scores.jsonl"]["sha256"]
        point["scorer_version"] = receipt.get("scorer_version")
        point["scorer_checks"] = checks
        points.append(point)
    return points


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--root", required=True, help="SFT run root ($TRACE_CERT_RUNS/sft/<tag> or sft_smoke/<tag>)")
    ap.add_argument("--data", required=True, help="frozen data root")
    ap.add_argument("--expected-rows", type=int, help="scored E1-dev rows per checkpoint (default: plan / receipt)")
    ap.add_argument("--rlvr-root", help="trace-rlvr checkout (git HEAD recorded)")
    ap.add_argument("--tool-root", help="trace-tool checkout (git HEAD recorded)")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    try:
        plan = read_json(root / "plan.json")
        check = read_json(root / "check.json")
        if not check.get("ok"):
            raise SelectError(f"{root}/check.json is not ok: {check.get('error')}")
        done = read_json(root / "train_done.json") if (root / "train_done.json").exists() else None
        if done is None:
            raise SelectError("training has not finished (no train_done.json)", 1)
        if done["plan"]["math_sha256"] != plan["math_sha256"] or done["plan"]["save_steps"] != plan["save_steps"]:
            raise SelectError("train_done.json does not match plan.json")
        expected = args.expected_rows
        if expected is None:
            if plan.get("dev_rows_per_task"):
                expected = 4 * int(plan["dev_rows_per_task"])
            else:
                expected = (check.get("data") or {}).get("dev_rows")
        points = collect(root, plan, expected)
        sel = plan["select"]
        result = apply_rule(points, min_F=sel["min_F"], max_abs_dA=sel["max_abs_dA"], max_abs_dM=sel["max_abs_dM"], fallback=sel.get("fallback", "last"))
        step = result["selected_step"]
        link = link_selected(root, step)
        ckpt_info = read_json(root / "ckpt" / step_name(step) / "sft_checkpoint.json")
        write_json(root / "dev_curve.json", {"select_version": SELECT_VERSION, **result})
        runs = [json.loads(ln) for ln in (root / "runs.jsonl").read_text().splitlines() if ln.strip()] if (root / "runs.jsonl").exists() else []
        receipt = {
            "receipt_version": RECEIPT_VERSION,
            "command": "sft",
            "written": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "smoke": bool(plan.get("smoke")),
            "model": {**plan["model"], "base_dir": (check.get("base") or {}).get("dir"), "base_files_sha256": (check.get("base") or {}).get("files")},
            "config": {"path": check.get("config"), "sha256": check.get("config_sha256"), "config_id": plan["config_id"], "math_sha256": plan["math_sha256"]},
            "data": {"root": str(Path(args.data).resolve()), "files_sha256": (check.get("data") or {}).get("files"), "sft_rows": (check.get("data") or {}).get("sft_rows"),
                     "train_rows": plan["n_train"], "export_provenance": (check.get("data") or {}).get("export_provenance"),
                     "dev_rows_scored": expected, "lengths": check.get("lengths")},
            "token_identity_easyr1": check.get("easyr1_equivalence"),
            "training": {"plan": plan, "launches": runs, "finished": done.get("finished"), "train_log_sha256": done.get("train_log_sha256"),
                         "model": done.get("model"), "versions": done.get("versions"), "provenance": done.get("provenance")},
            "git": {"trace": {"head": (done.get("provenance") or {}).get("git_head"), "dirty": (done.get("provenance") or {}).get("git_dirty"),
                              "cert_code_sha256": (done.get("provenance") or {}).get("cert_code_sha256")},
                    "trace_rlvr_head": _git(args.rlvr_root, "rev-parse", "HEAD") or ((check.get("easyr1_equivalence") or {}).get("rlvr_head")),
                    "trace_tool_head": _git(args.tool_root, "rev-parse", "HEAD")},
            "env": {"train_env_lock_sha256": (done.get("provenance") or {}).get("env_lock_sha256"), "train_versions": done.get("versions"),
                    "check_env_lock_sha256": (check.get("provenance") or {}).get("env_lock_sha256"), "scorer_provenance": _provenance()},
            "dev_eval": {"split": "e1_dev", "decoding": plan.get("eval"), "curve": result["curve"]},
            "rule": result["rule"],
            "outcome": result["outcome"],
            "selected": {"step": step, "epoch": ckpt_info.get("epoch"), "dir": str((root / "ckpt" / step_name(step)).resolve()), "link": str(link),
                         "files": ckpt_info.get("files")},
        }
        write_json(root / "receipt.json", receipt)
        print(f"{'step':>6} {'epoch':>6} {'A':>7} {'M':>7} {'F':>7} {'G':>7} {'dA':>8} {'dM':>8}  qualifies")
        for p in result["curve"]:
            fmt = lambda v: f"{v:+.4f}" if isinstance(v, float) else "     -"  # noqa: E731
            print(f"{p['step']:>6} {p['epoch'] or 0:>6.2f} {p['A']:>7.4f} {p['M']:>7.4f} {p['F']:>7.4f} {(p.get('G') or 0):>7.4f} "
                  f"{fmt(p['dA']):>8} {fmt(p['dM']):>8}  {'yes' if p['qualifies'] else 'no: ' + '; '.join(p['reasons'])}")
        print(f"selected {step_name(step)} ({result['outcome']}) -> {link}")
        return 0
    except SelectError as exc:
        print(f"sft_select: {exc}", file=sys.stderr)
        return exc.code
    except (OSError, KeyError, ValueError) as exc:
        print(f"sft_select: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
