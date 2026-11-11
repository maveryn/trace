"""Tests of the evaluation job helpers (jobs/eval_lib.py, jobs/common.sh, jobs/run_tiers.sh, jobs/eval_subset.sh).

The eval_lib tests run against small fake protocol modules, so they need neither the protocol clone nor GPUs. A few
tests use the real protocol queue (benchmark_queue_lib) and run eval_subset.sh end to end against a stub protocol clone
(fake nvidia-smi, stub pools and scorers); they are skipped unless the protocol scripts are importable and LMUData
points at the prepared dataset manifest.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
import types
from pathlib import Path

import pytest

JOBS = Path(__file__).resolve().parents[1] / "jobs"
sys.path.insert(0, str(JOBS))

import eval_lib as L  # noqa: E402

BASH = shutil.which("bash")
needs_bash = pytest.mark.skipif(BASH is None, reason="bash not available")
MANIFEST = Path(os.environ.get("LMUData") or "/nonexistent") / "trace_eval_v1_dataset_manifest.json"
PROTOCOL = (importlib.util.find_spec("benchmark_queue_lib") is not None
            and importlib.util.find_spec("trace_eval_suite") is not None)
needs_protocol = pytest.mark.skipif(not PROTOCOL, reason="protocol scripts (benchmark_queue_lib) not importable")
needs_protocol_data = pytest.mark.skipif(not (PROTOCOL and MANIFEST.is_file() and BASH and shutil.which("git")),
                                         reason="protocol scripts, LMUData dataset manifest, bash and git needed")


def dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


# ============================================================================== fake protocol modules


class Spec(types.SimpleNamespace):
    pass


def fake_protocol(monkeypatch, keys=("a", "b")):
    """Fake benchmark_queue_lib and trace_eval_suite with benchmarks `keys` (category 'cat', route 'direct')."""

    bql = types.ModuleType("benchmark_queue_lib")

    def spec_by_key(key):
        if key not in keys:
            raise KeyError(key)
        return Spec(key=key, run_name="run")

    bql.spec_by_key = spec_by_key
    bql.run_dir = lambda spec, slug, root: Path(root) / spec.key / slug / spec.run_name
    bql.score_path = lambda spec, slug, root: Path(root) / spec.key / slug / spec.run_name / "scores.json"
    bql.extract_score_and_rows = lambda payload: (payload.get("score"), payload.get("rows"))
    bql.file_lock = lambda path: contextlib.nullcontext()
    suite = types.SimpleNamespace(
        suite_id="trace_eval_v1", manifest_sha256="0" * 64, benchmark_keys=tuple(keys),
        benchmarks=[types.SimpleNamespace(key=k, display=k.upper(), rows=10) for k in keys],
        categories={"cat": list(keys)}, routes={"direct": list(keys)}, rows_by_benchmark={k: 10 for k in keys})
    tes = types.ModuleType("trace_eval_suite")
    tes.load_trace_eval_suite = lambda: suite
    monkeypatch.setitem(sys.modules, "benchmark_queue_lib", bql)
    monkeypatch.setitem(sys.modules, "trace_eval_suite", tes)
    return bql, suite


def media(n, dup=False):
    return [{"ordinal": o, "index": str(7 if dup and o < 2 else 100 + o), "source_row_hash": f"h{o}"} for o in range(n)]


def row_list(manifest, picks, keys=("a", "b"), snapshot="f" * 64):
    """A row list in trace_merge.grouping.rowlist's format: picks = {key: [ordinal, ...]}."""

    out = {"schema": L.SUBSET_SCHEMA, "mode": "smoke", "dataset_snapshot_sha256": snapshot, "benchmarks": {}}
    lines = []
    for key in keys:
        rows_media = manifest["datasets"][key]["row_media"]
        unique = len({m["index"] for m in rows_media}) == len(rows_media)
        rows = []
        for o in sorted(picks.get(key, [])):
            m = rows_media[o]
            rid = m["index"] if unique else f"{o}:{m['index']}"
            rows.append({"ordinal": o, "index": m["index"], "source_row_hash": m["source_row_hash"], "row_id": rid})
            lines.append(f"{key}\t{rid}\n")
        out["benchmarks"][key] = {"total": len(rows_media), "selected": len(rows), "rows": rows}
    out["total_selected"] = sum(len(v) for v in picks.values())
    out["row_list_sha256"] = hashlib.sha256("".join(lines).encode()).hexdigest()
    return out


@pytest.fixture()
def manifest():
    return {"datasets": {"a": {"row_media": media(5)}, "b": {"row_media": media(4, dup=True)}}}


# ============================================================================== rows-check


def frozen_of(payload):
    return {"row_list_sha256": payload["row_list_sha256"], "total_selected": payload["total_selected"]}


def test_row_list_check_accepts_a_frozen_list(manifest):
    payload = row_list(manifest, {"a": [1, 3], "b": [0, 2]})
    assert payload["benchmarks"]["b"]["rows"][0]["row_id"] == "0:7"  # duplicate indices -> ordinal:index
    assert L.check_row_list(payload, ["a", "b"], manifest, frozen_of(payload)) == (None, payload["row_list_sha256"])


@pytest.mark.parametrize("tamper, expect", [
    # the reviewer's case: same row_id (so the same digest) re-pointed at another ordinal, selected inflated
    (lambda p: p["benchmarks"]["a"]["rows"][0].update(ordinal=0, index="100", source_row_hash="h0"), "row_id"),
    (lambda p: p["benchmarks"]["a"].update(selected=9), "selected"),
    (lambda p: p.update(total_selected=99), "total_selected"),
    (lambda p: p["benchmarks"]["a"]["rows"].append(dict(p["benchmarks"]["a"]["rows"][0])), "twice"),
    (lambda p: p["benchmarks"]["a"]["rows"][0].update(source_row_hash="other"), "prepared dataset"),
    (lambda p: p["benchmarks"]["a"]["rows"][0].update(ordinal=-1), "prepared dataset"),
    (lambda p: p["benchmarks"].pop("b"), "every suite benchmark"),
])
def test_row_list_check_rejects_rows_the_digest_does_not_bind(manifest, tamper, expect):
    payload = row_list(manifest, {"a": [1, 3], "b": [0, 2]})
    frozen = frozen_of(payload)
    tamper(payload)
    problem, _ = L.check_row_list(payload, ["a", "b"], manifest, frozen)
    assert problem is not None and expect in problem


def test_row_list_check_needs_the_split_manifest_digest(manifest):
    payload = row_list(manifest, {"a": [1], "b": [0]})
    problem, _ = L.check_row_list(payload, ["a", "b"], manifest, {"row_list_sha256": "0" * 64})
    assert problem and "split manifest" in problem


def test_plan_counts_listed_rows(tmp_path, monkeypatch, manifest):
    fake_protocol(monkeypatch)
    payload = row_list(manifest, {"a": [1, 3], "b": [0]})
    payload["benchmarks"]["a"]["selected"] = 9  # rows-check refuses this; the plan does not trust it either
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps(payload))
    args = L.build_parser().parse_args(["plan", "--model", "m", "--slug", "s", "--model-path", "m", "--revision", "r",
                                        "--source", "o/r@x", "--seed", "42", "--subset", "smoke", "--rows", str(rows),
                                        "--out", str(tmp_path / "plan.json")])
    assert L.cmd_plan(args) == 0
    first = (tmp_path / "plan.json").read_text()
    assert json.loads(first)["expected_rows"] == {"a": 2, "b": 1}
    time.sleep(1.1)
    assert L.cmd_plan(args) == 0 and (tmp_path / "plan.json").read_text() == first  # unchanged plan: file kept
    args.revision = "r2"
    assert L.cmd_plan(args) == 0 and json.loads((tmp_path / "plan.json").read_text())["revision"] == "r2"


# ============================================================================== generation row filter


class FakeGen:
    def __init__(self, data_by_key):
        import pandas as pd

        self.data = {k: pd.DataFrame(v) for k, v in data_by_key.items()}

    def _row_hash(self, row):
        return f"h{row['ordinal']}"

    def _prepare_handles_and_jobs(self, args, specs):
        handles = {k: types.SimpleNamespace(dataset=types.SimpleNamespace(data=df), source_ordinals=None)
                   for k, df in self.data.items()}
        jobs = [types.SimpleNamespace(spec=types.SimpleNamespace(key=k), row=df.iloc[o].to_dict(), rank=o,
                                      source_ordinal=o) for k, df in self.data.items() for o in range(len(df))]
        return handles, jobs


def test_row_filter_keeps_full_handles_until_finalization():
    gen = FakeGen({"a": [{"ordinal": o, "index": str(100 + o)} for o in range(5)]})
    selection = {"a": {(1, "101", "h1"), (3, "103", "h3")}}
    L._install_row_filter(gen, selection)
    handles, jobs = gen._prepare_handles_and_jobs(types.SimpleNamespace(finalize_only=False), [])
    assert [j.source_ordinal for j in jobs] == [1, 3]
    assert len(handles["a"].dataset.data) == 5  # ranks, row keys and request hashes stay those of a full run
    handles, _ = gen._prepare_handles_and_jobs(types.SimpleNamespace(finalize_only=True), [])
    assert handles["a"].source_ordinals == [1, 3] and list(handles["a"].dataset.data["index"]) == ["101", "103"]


def test_row_filter_refuses_rows_of_another_dataset():
    gen = FakeGen({"a": [{"ordinal": o, "index": str(100 + o)} for o in range(5)]})
    L._install_row_filter(gen, {"a": {(1, "999", "h1")}})
    with pytest.raises(RuntimeError, match="does not match"):
        gen._prepare_handles_and_jobs(types.SimpleNamespace(finalize_only=False), [])


# ============================================================================== validate


def write_run(tmp_path, manifest, picks, *, revision="sha256set:" + "a" * 64, scores=True, extra=None, snapshot=None):
    payload = row_list(manifest, picks)
    rows = tmp_path / "rows.smoke.json"
    rows.write_text(json.dumps(payload))
    plan = {"slug": "s", "seed": 42, "revision": "sha256set:" + "a" * 64, "rows_file": str(rows),
            "row_list_sha256": payload["row_list_sha256"], "subset": "smoke", "model": "m",
            "expected_rows": {k: len(v["rows"]) for k, v in payload["benchmarks"].items()}}
    run_root, score_root = tmp_path / "runs", tmp_path / "scoring"
    for key, entry in payload["benchmarks"].items():
        d = run_root / key / "s" / "run"
        d.mkdir(parents=True)
        recs = [{"index": r["index"], "source_ordinal": str(r["ordinal"]), "source_row_hash": r["source_row_hash"],
                 "prediction": "B"} for r in entry["rows"]]
        if extra and key in extra:
            recs = extra[key](recs)
        n = len(entry["rows"])
        (d / "generation_summary.json").write_text(json.dumps({"rows": n, "expected_rows": n, "generation": {
            "seed": 42, "max_tokens": 4096, "model_revision": revision,
            "dataset_snapshot_sha256": snapshot or payload["dataset_snapshot_sha256"]}}))
        (d / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        if scores:
            p = score_root / "seed_42" / "benchmark" / key / "s" / "run" / "scores.json"
            p.parent.mkdir(parents=True)
            p.write_text(json.dumps({"score": 50.0, "rows": n}))
    return plan, run_root, score_root


def test_validate_accepts_the_planned_rows(tmp_path, monkeypatch, manifest):
    fake_protocol(monkeypatch)
    plan, run_root, score_root = write_run(tmp_path, manifest, {"a": [1, 3], "b": [0]})
    res = L.validate(plan, run_root, score_root, "score")
    assert res["complete"], res["problems"]
    assert res["generated_total"] == res["expected_total"] == 3


def test_validate_rejects_another_model_revision(tmp_path, monkeypatch, manifest):
    fake_protocol(monkeypatch)
    plan, run_root, score_root = write_run(tmp_path, manifest, {"a": [1], "b": [0]}, revision="sha256set:" + "f" * 64)
    res = L.validate(plan, run_root, score_root, "score")
    assert not res["complete"] and any("model revision" in p for p in res["problems"])


def test_validate_rejects_unplanned_rows_with_equal_counts(tmp_path, monkeypatch, manifest):
    fake_protocol(monkeypatch)
    swap = lambda recs: [{**recs[0], "source_ordinal": "0", "index": "100", "source_row_hash": "h0"}, *recs[1:]]  # noqa
    plan, run_root, score_root = write_run(tmp_path, manifest, {"a": [1, 3], "b": [0]}, extra={"a": swap})
    res = L.validate(plan, run_root, None, "generation")
    assert not res["complete"]
    assert any("not the planned rows (1 missing, 1 unplanned" in p for p in res["problems"])


def test_validate_rejects_rows_without_identity_and_other_snapshots(tmp_path, monkeypatch, manifest):
    fake_protocol(monkeypatch)
    strip = lambda recs: [{"index": r["index"], "prediction": "B"} for r in recs]  # noqa: E731
    plan, run_root, _ = write_run(tmp_path, manifest, {"a": [1], "b": [0]}, extra={"b": strip}, snapshot="e" * 64)
    problems = L.validate(plan, run_root, None, "generation")["problems"]
    assert any("1 without identity" in p for p in problems) and any("dataset snapshot" in p for p in problems)


def test_validate_rejects_a_row_list_that_is_not_the_plans(tmp_path, monkeypatch, manifest):
    fake_protocol(monkeypatch)
    plan, run_root, score_root = write_run(tmp_path, manifest, {"a": [1], "b": [0]})
    plan["row_list_sha256"] = "0" * 64
    res = L.validate(plan, run_root, score_root, "score")
    assert not res["complete"] and any("is not the plan's" in p for p in res["problems"])


def test_validate_score_phase_needs_every_scores_json(tmp_path, monkeypatch, manifest):
    fake_protocol(monkeypatch)
    plan, run_root, score_root = write_run(tmp_path, manifest, {"a": [1], "b": [0]})
    (score_root / "seed_42" / "benchmark" / "b" / "s" / "run" / "scores.json").unlink()
    assert L.validate(plan, run_root, None, "generation")["complete"]
    res = L.validate(plan, run_root, score_root, "score")
    assert not res["complete"] and any("b: no scores.json" in p for p in res["problems"])


# ============================================================================== summary


def test_summary_reports_only_protocol_aggregates(tmp_path, monkeypatch, manifest):
    fake_protocol(monkeypatch)
    plan, _, score_root = write_run(tmp_path, manifest, {"a": [1], "b": [0]})
    s = L.summarize(plan, score_root, None)
    assert s["complete"] and s["macro"]["overall_24"] == 50.0 and s["macro"]["categories"] == {"cat": 50.0}
    flat = json.dumps(s)
    assert "natural" not in flat and "synthetic" not in flat and "groups" not in s
    assert "| benchmark | category |" in L.summary_markdown(s)


# ============================================================================== judge placement


@pytest.mark.parametrize("mib, n, tp, endpoints, batched", [
    (143771, 1, 1, 1, 32768),    # H200: 124.9 - 61.0 = 63.8 GiB headroom, one TP1 endpoint per card
    (143771, 2, 1, 2, 32768),
    (81559, 1, 1, 1, 8192),      # one 80 GB card: 9.1 GiB headroom -> tight TP1, fewer batched tokens
    (81559, 2, 2, 1, 32768),     # two: comfortable TP2 is preferred over two tight TP1 endpoints
    (46068, 2, 2, 1, 32768),     # 2 x 45 GiB: TP2 with 16.9 GiB headroom
    (24564, 4, 4, 1, 32768),     # 4 x 24 GiB: TP4 with 19.3 GiB headroom
])
def test_judge_plan_groups(mib, n, tp, endpoints, batched):
    plan = L.judge_plan([str(i) for i in range(n)], [mib] * n, 61.04)
    assert plan["feasible"] and plan["tp"] == tp and plan["endpoints"] == endpoints
    assert plan["max_num_batched_tokens"] == batched
    assert all(len(g.split(",")) == tp for g in plan["gpu_groups"].split())


def test_judge_plan_infeasible_has_a_reason():
    plan = L.judge_plan(["0"], [46068], 61.04)
    assert plan["feasible"] is False and "cannot hold Qwen3-32B" in plan["reason"]
    assert L.judge_plan(["0", "1"], [46068], 61.04) == {"feasible": False, "reason": "no GPUs"}


def test_judge_plan_fails_loudly_on_a_dangling_shard(tmp_path):
    """A broken judge directory must fail the plan (eval_subset.sh then exits 4), never yield a pending judge."""

    (tmp_path / "model-00001-of-00017.safetensors").symlink_to(tmp_path / "gone")
    with pytest.raises(FileNotFoundError):
        L.judge_weight_gib(tmp_path)
    r = subprocess.run([sys.executable, str(JOBS / "eval_lib.py"), "judge-plan", "--devices", "0", "--mib", "143771",
                        "--judge-dir", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode != 0 and r.stdout == ""


# ============================================================================== score queue recovery


def queue(tmp_path, jobs):
    q = tmp_path / "queues" / "score_x.json"
    q.parent.mkdir(parents=True, exist_ok=True)
    q.write_text(json.dumps({"jobs": jobs}))
    return q


def test_reset_stale_queue_resets_dead_running_and_exhausted_failed(tmp_path, monkeypatch):
    fake_protocol(monkeypatch, keys=("a", "b", "c", "d", "e"))
    dead, now = dead_pid(), time.time()
    q = queue(tmp_path, {
        "a": {"status": "running", "pid": dead, "attempts": 1, "updated_at": now},
        "b": {"status": "failed", "pid": dead, "attempts": 2, "updated_at": now},
        "c": {"status": "failed", "pid": dead, "attempts": 2, "updated_at": now},   # its scores.json exists
        "d": {"status": "running", "pid": os.getpid(), "attempts": 1, "updated_at": now},  # alive on this host
        "e": {"status": "done", "pid": dead, "attempts": 1, "updated_at": now},
    })
    bench = tmp_path / "benchmark"
    (bench / "c" / "s" / "run").mkdir(parents=True)
    (bench / "c" / "s" / "run" / "scores.json").write_text("{}")
    res = L.reset_stale_queue(q.parent, bench, "s")
    assert [i["job"] for i in res["reset"]] == ["a", "b"] and [i["job"] for i in res["kept"]] == ["d"]
    jobs = json.loads(q.read_text())["jobs"]
    assert {k: (v["status"], v["attempts"]) for k, v in jobs.items()} == {
        "a": ("failed", 0), "b": ("failed", 0), "c": ("failed", 2), "d": ("running", 1), "e": ("done", 1)}
    assert "reset under the run lease" in jobs["a"]["note"]


@needs_protocol
def test_reset_stale_queue_makes_the_protocol_claim_again(tmp_path):
    """The protocol's own claim_next_job skips both entries before the reset and claims them after it."""

    import benchmark_queue_lib as bql

    dead, now = dead_pid(), time.time()
    q = queue(tmp_path, {"logicvista": {"status": "failed", "pid": dead, "attempts": 2, "updated_at": now},
                         "mathvista": {"status": "running", "pid": dead, "attempts": 1, "updated_at": now}})
    bench = tmp_path / "benchmark"
    jobs = [(k, bql.score_path(bql.spec_by_key(k), "s", bench)) for k in ("logicvista", "mathvista")]
    assert bql.claim_next_job(queue_path=q, jobs=jobs, worker_id="w") is None
    res = L.reset_stale_queue(q.parent, bench, "s")
    assert sorted(i["job"] for i in res["reset"]) == ["logicvista", "mathvista"]
    claimed = {bql.claim_next_job(queue_path=q, jobs=jobs, worker_id="w") for _ in range(2)}
    assert claimed == {"logicvista", "mathvista"}


# ============================================================================== shell helpers


def bash(script: str, env: dict[str, str] | None = None, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run([BASH, "-c", script], capture_output=True, text=True, timeout=timeout,
                          env={**os.environ, **(env or {})})


def claim_file(claims: Path, token: str) -> Path:
    host = subprocess.run(["hostname", "-s"], capture_output=True, text=True).stdout.strip()
    f = claims / f"tvm_unknown_none_{host}_11111111"
    hb = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    hb = hb[:-2] + ":" + hb[-2:]
    f.write_text(json.dumps({"claimed": hb, "owner": f"claude session TRACE-VERO merge study (x; token {token})",
                             "heartbeat": hb}) + "\n")
    return f


TAKE = f"""set -u; source {JOBS}/common.sh; SLURM_CLUSTER_NAME=unknown; unset SLURM_JOB_ID
TRC_CLAIM_STALE=900; TRC_GPU_UUID=(GPU-11111111-a); TRC_GPU_IDX=(0); trc_claim_take; echo "rc=$?" """


@needs_bash
def test_claim_of_a_dead_runner_on_this_host_is_replaced(tmp_path):
    host = subprocess.run(["hostname", "-s"], capture_output=True, text=True).stdout.strip()
    f = claim_file(tmp_path, f"tvm-{host}-{dead_pid()}-1700000000")
    r = bash(TAKE, {"TRC_CLAIM_DIR": str(tmp_path)})
    assert "rc=0" in r.stdout and "is gone from this host" in r.stdout, r.stdout + r.stderr
    assert "tvm-" in f.read_text() and "(x; token" not in f.read_text()


@needs_bash
@pytest.mark.parametrize("where", ["live_here", "other_host"])
def test_fresh_claim_of_a_live_or_remote_runner_collides(tmp_path, where):
    host = subprocess.run(["hostname", "-s"], capture_output=True, text=True).stdout.strip()
    live = subprocess.Popen(["sleep", "30"])
    try:
        token = f"tvm-{host}-{live.pid}-1700000000" if where == "live_here" else f"tvm-elsewhere-{dead_pid()}-1"
        claim_file(tmp_path, token)
        r = bash(TAKE, {"TRC_CLAIM_DIR": str(tmp_path)})
    finally:
        live.kill()
    assert "rc=3" in r.stdout and "claim collision" in r.stdout, r.stdout + r.stderr


@needs_bash
@pytest.mark.parametrize("case, rc", [("dead_here", 0), ("dead_here_orphan", 3), ("other_host", 3)])
def test_lease_of_a_dead_runner_is_taken_over_only_without_orphans(tmp_path, case, rc):
    host = subprocess.run(["hostname", "-s"], capture_output=True, text=True).stdout.strip()
    run = tmp_path / "run+dir"  # a regex character in the path must not defeat the orphan check
    run.mkdir()
    pid = dead_pid()
    owner = f"tvm-{host if case != 'other_host' else 'elsewhere'}-{pid}-1700000000"
    (run / ".owner.seed42").write_text(f"{owner} {host} none none {pid} {int(time.time())}\n")
    orphan = None
    if case == "dead_here_orphan":  # e.g. a generation queue that outlived its runner
        orphan = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", str(run / "campaign")])
    try:
        # (the path goes in through the environment: a process whose argv names the run directory counts as an orphan)
        r = bash(f'set -u; source {JOBS}/common.sh; trc_lease_take "$LEASE"; echo "rc=$?"',
                 {"LEASE": str(run / ".owner.seed42")})
    finally:
        if orphan:
            orphan.kill()
    assert f"rc={rc}" in r.stdout, r.stdout + r.stderr
    assert (("taking over" in r.stdout) == (rc == 0))


def tiers_env(tmp_path, rc_by_subset, pending=()):
    """run_tiers.sh next to a stub eval_subset.sh that exits with rc_by_subset[subset] and writes a pending marker."""

    jobs = tmp_path / "jobs"
    jobs.mkdir()
    shutil.copy(JOBS / "run_tiers.sh", jobs / "run_tiers.sh")
    (jobs / "eval_subset.sh").write_text(textwrap.dedent("""\
        #!/usr/bin/env bash
        slug=$2; seed=$4; subset=$8
        echo "$subset" >> "$TVM_WORK/called"
        if [[ " $STUB_PENDING " == *" $subset "* ]]; then
          d=${TVM_RUNS:-$TVM_WORK/runs}/$slug/$subset; mkdir -p "$d"; echo '{}' > "$d/PENDING_JUDGE.seed$seed"
        fi
        eval "exit \\${STUB_RC_$subset:-0}"
        """))
    work = tmp_path / "work"
    (work / "views" / "v").mkdir(parents=True)
    (work / "views" / "v" / "view_manifest.json").write_text("{}")
    env = {"TVM_WORK": str(work), "STUB_PENDING": " ".join(pending), "TVM_RUNS": str(tmp_path / "elsewhere")}
    env.update({f"STUB_RC_{k}": str(v) for k, v in rc_by_subset.items()})
    return jobs / "run_tiers.sh", work, env


@needs_bash
def test_run_tiers_exits_6_on_a_pending_judge_and_stops(tmp_path):
    script, work, env = tiers_env(tmp_path, {}, pending=("smoke",))
    r = bash(f'bash "{script}" v smoke:42 development_screen:42', env)
    assert r.returncode == 6, r.stdout + r.stderr
    assert (work / "called").read_text().split() == ["smoke"]
    assert "rc=6: judge pending" in r.stdout


@needs_bash
def test_run_tiers_stops_at_the_first_failing_tier(tmp_path):
    script, work, env = tiers_env(tmp_path, {"smoke": 0, "development_screen": 5})
    r = bash(f'bash "{script}" v smoke:42 development_screen:42 confirmation:42', env)
    assert r.returncode == 5 and (work / "called").read_text().split() == ["smoke", "development_screen"]
    r = bash(f'bash "{script}" v smoke:42', {**env, "STUB_RC_development_screen": "0"})
    assert r.returncode == 0


# ============================================================================== eval_subset.sh end to end (stub clone)

STUBS = {
    "rlvr/evaluation/scripts/setup_trace_eval_env.sh": "#!/bin/bash\nexit 0\n",
    "rlvr/evaluation/scripts/run_mme_reasoning_eval.py": "# trace-cert post-release fix-2 trace-cert post-release fix-3\n",
    "rlvr/evaluation/scripts/run_external_benchmark_score_queue.py": "# trace-cert post-release fix-3\n",
    "rlvr/evaluation/scripts/trace_eval_evaluator_provenance.py": "print('e' * 64)\n",
    "rlvr/evaluation/scripts/trace_eval_code_provenance.py": "print('c' * 64)\n",
    "rlvr/evaluation/scripts/run_external_benchmark_generation_api_queue.py": "raise SystemExit('stub: not used')\n",
    "rlvr/evaluation/scripts/run_trace_eval_score_campaign.py":
        "import os, sys, time\ntime.sleep(float(os.environ.get('STUB_SCORE_SLEEP', '0')))\nsys.exit(1)\n",
    "rlvr/evaluation/scripts/prepare_trace_eval_models.py": textwrap.dedent("""\
        import json, sys
        from pathlib import Path
        MARKER_NAME = ".trace_model_revision.json"
        def _write_marker(path, marker): (Path(path) / MARKER_NAME).write_text(json.dumps(marker))
        def verify(entries, deep): return None
        def register_local(slug, path, source):
            rev = "sha256set:" + "a" * 64
            _write_marker(path, {"slug": slug, "immutable_revision": rev, "source": source}); return rev
        if __name__ == "__main__":
            sys.exit(0)
        """),
    "rlvr/evaluation/scripts/start_vllm_endpoint_pool.sh": textwrap.dedent("""\
        #!/bin/bash
        mkdir -p "$LOG_DIR"; : > "$PID_FILE"
        echo "GPU_GROUPS=[$GPU_GROUPS] SERVED=$SERVED_MODEL_NAME" >> "$STUB_POOL_LOG"
        [ -n "${STUB_POOL_OK:-}" ] || exit 5
        read -r -a G <<< "$GPU_GROUPS"
        for i in "${!G[@]}"; do  # one static /v1/models server per group
          port=$((PORT_START + i)); d="$LOG_DIR/www$i"; mkdir -p "$d/v1"; echo '{"data":[]}' > "$d/v1/models"
          setsid "$PYTHON_BIN" -m http.server "$port" --bind 127.0.0.1 --directory "$d" >"$LOG_DIR/endpoint_$i.log" 2>&1 &
          echo "$! $port ${G[$i]} x" >> "$PID_FILE"
        done
        """),
    "rlvr/evaluation/scripts/stop_vllm_endpoint_pool.sh": textwrap.dedent("""\
        #!/bin/bash
        while read -r pid rest; do kill -TERM -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null; done < "$PID_FILE"
        sleep "${STOP_GRACE_SEC:-1}"
        """),
    "external/VLMEvalKit/scripts/batched_vlmevalkit_qwen3vl.py": "def _parse_charxiv_judge():\n    pass\n",
    "rlvr/examples/prompts/chat_template_no_think.jinja": "{{ messages }}\n",
    "bin/nvidia-smi": textwrap.dedent("""\
        #!/bin/bash
        # FAKE_GPUS = "idx|uuid|name|totalMiB|usedMiB;..."
        case "$*" in *query-compute-apps*) exit 0 ;; esac
        IFS=';' read -ra G <<< "$FAKE_GPUS"
        for g in "${G[@]}"; do IFS='|' read -r i u n t used <<< "$g"
          case "$*" in
            *index,uuid,name,memory.total,memory.used*) echo "$i, $u, $n, $t, $used" ;;
            *index,uuid,memory.used*) echo "$i, $u, $used" ;;
            *utilization.gpu*) echo "$i, $u, 50, $used, $t" ;;
          esac
        done
        """),
}


@pytest.fixture()
def stub_world(tmp_path):
    """A stub protocol clone (git commit, stub scripts, the real suite/queue modules and dataset manifest), a work root
    with a one-row-per-benchmark smoke list and a finalized generation of it, and the runner's environment."""

    import benchmark_queue_lib as bql
    import trace_eval_suite

    clone = tmp_path / "clone"
    for rel, text in STUBS.items():
        base = tmp_path if rel.startswith("bin/") else clone
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        p.chmod(0o755)
    scripts = clone / "rlvr/evaluation/scripts"
    for mod in (bql, trace_eval_suite):
        (scripts / Path(mod.__file__).name).symlink_to(Path(mod.__file__).resolve())
    lmu = clone / "rlvr/evaluation/.work/LMUData"
    lmu.mkdir(parents=True)
    (lmu / MANIFEST.name).symlink_to(MANIFEST.resolve())
    (clone / "rlvr/evaluation/.work/eval_deps").mkdir()
    git = ["git", "-C", str(clone), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-qm", "stub"], check=True)
    commit = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()

    suite_keys = list(trace_eval_suite.load_trace_eval_suite().benchmark_keys)
    manifest = json.loads(MANIFEST.read_text())
    payload = {"schema": L.SUBSET_SCHEMA, "mode": "smoke", "benchmarks": {},
               "dataset_snapshot_sha256": manifest["view_snapshot_sha256"]["trace_eval_v1"]}
    lines = []
    for key in suite_keys:
        rows_media = manifest["datasets"][key]["row_media"]
        unique = len({str(m["index"]) for m in rows_media}) == len(rows_media)
        m = rows_media[0]
        rid = str(m["index"]) if unique else f"0:{m['index']}"
        payload["benchmarks"][key] = {"total": len(rows_media), "selected": 1, "rows": [
            {"ordinal": 0, "index": str(m["index"]), "source_row_hash": m["source_row_hash"], "row_id": rid}]}
        lines.append(f"{key}\t{rid}\n")
    payload["total_selected"] = len(suite_keys)
    payload["row_list_sha256"] = hashlib.sha256("".join(lines).encode()).hexdigest()
    work = tmp_path / "work"
    (work / "splits").mkdir(parents=True)
    (work / "splits" / "rows.smoke.json").write_text(json.dumps(payload))
    (work / "splits" / "split_manifest.json").write_text(json.dumps({"row_lists": {"smoke": {
        "file": "rows.smoke.json", "row_list_sha256": payload["row_list_sha256"], "total_selected": len(suite_keys)}}}))
    view = work / "views" / "fake"
    view.mkdir(parents=True)
    (view / "config.json").write_text("{}")
    (view / "view_manifest.json").write_text("{}")

    root = work / "runs" / "fake" / "smoke"

    def finalize(scores=False):
        for key, entry in payload["benchmarks"].items():
            r = entry["rows"][0]
            d = bql.run_dir(bql.spec_by_key(key), "fake", root / "campaign" / "seed_42" / "runs")
            d.mkdir(parents=True, exist_ok=True)
            (d / "generation_summary.json").write_text(json.dumps({"rows": 1, "expected_rows": 1, "generation": {
                "seed": 42, "max_tokens": 4096, "model_revision": "sha256set:" + "a" * 64,
                "dataset_snapshot_sha256": payload["dataset_snapshot_sha256"]}}))
            (d / "predictions.jsonl").write_text(json.dumps({
                "index": r["index"], "source_ordinal": "0", "source_row_hash": r["source_row_hash"],
                "prediction": "B"}) + "\n")
            if scores:
                p = bql.score_path(bql.spec_by_key(key), "fake", root / "scoring" / "seed_42" / "benchmark")
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps({"accuracy": 50.0, "rows": 1}))

    (tmp_path / "claims").mkdir()
    (tmp_path / "tmp").mkdir()
    (tmp_path / "judge").mkdir()
    env = {k: v for k, v in os.environ.items() if k not in ("TVM_RUNS", "CUDA_VISIBLE_DEVICES", "N", "PYTHONPATH")}
    env.update({"TVM_WORK": str(work), "TVM_PROTOCOL_CLONE": str(clone), "TVM_PYTHON": sys.executable,
                "HF_HOME": str(tmp_path / "hf"), "TVM_CLAIM_DIR": str(tmp_path / "claims"),
                "TMPDIR": str(tmp_path / "tmp"), "TRC_NVIDIA_SMI": str(tmp_path / "bin/nvidia-smi"),
                "TRC_EXT_COMMIT": commit, "TRC_EXT_VERIFY_DATA": "0", "TRC_EXT_JUDGE_MODEL": str(tmp_path / "judge"),
                "PYTHONDONTWRITEBYTECODE": "1", "STUB_POOL_LOG": str(tmp_path / "pool.log"),
                "TRC_EXCLUDE_NODES": "^nomatch$", "SLURM_JOB_ID": "999", "SLURM_STEP_ID": "1", "TVM_VIEW_WAIT": "0"})

    def run(*extra, gpus="0|GPU-11111111-a|NVIDIA H200|143771|0", tiers=False, cvd=None, **extra_env):
        cmd = ([BASH, str(JOBS / "run_tiers.sh"), "fake", "smoke:42", "development_screen:42"] if tiers else
               [BASH, str(JOBS / "eval_subset.sh"), str(view), "fake", "--seed", "42", "--rows",
                str(work / "splits" / "rows.smoke.json"), "--subset", "smoke", *extra])
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                           env={**env, "FAKE_GPUS": gpus, **({"CUDA_VISIBLE_DEVICES": cvd} if cvd else {}),
                                **extra_env})
        logs = "".join(p.read_text() for p in sorted((root / "logs").glob("eval_subset.seed42.*.log")))
        return r.returncode, logs + r.stdout + r.stderr

    def start(gpus="0|GPU-11111111-a|NVIDIA H200|143771|0", **extra_env):
        cmd = [BASH, str(JOBS / "eval_subset.sh"), str(view), "fake", "--seed", "42", "--rows",
               str(work / "splits" / "rows.smoke.json"), "--subset", "smoke"]
        return subprocess.Popen(cmd, stdout=open(tmp_path / "start.out", "w"), stderr=subprocess.STDOUT,
                                start_new_session=True, env={**env, "FAKE_GPUS": gpus, **extra_env})

    def log():
        return "".join(p.read_text() for p in sorted((root / "logs").glob("eval_subset.seed42.*.log")))

    return types.SimpleNamespace(root=root, finalize=finalize, run=run, start=start, log=log, clone=clone,
                                 claims=tmp_path / "claims", judge=tmp_path / "judge", pool_log=tmp_path / "pool.log")


@needs_protocol_data
def test_runner_judge_plan_failure_is_exit_4_not_pending(stub_world):
    stub_world.finalize()
    (stub_world.judge / "model-00001-of-00017.safetensors").symlink_to(stub_world.judge / "gone")
    rc, out = stub_world.run()
    assert rc == 4, out
    assert "judge-plan failed" in out and not (stub_world.root / "PENDING_JUDGE.seed42").exists()
    assert not list(stub_world.claims.iterdir()) and not (stub_world.root / ".owner.seed42").exists()


@needs_protocol_data
def test_runner_finished_run_exits_0_on_busy_cards_without_claiming(stub_world):
    stub_world.finalize(scores=True)
    rc, out = stub_world.run(gpus="0|GPU-11111111-a|NVIDIA H200|143771|60000")
    assert rc == 0, out
    assert "no GPU claimed" in out and "COMPLETE" in out and "claimed tvm_" not in out
    attempt = json.loads((stub_world.root / "timing.seed42.json").read_text())["attempts"][-1]
    assert set(attempt["jobs_sha256"]) >= {"common.sh", "eval_lib.py", "eval_subset.sh", "run_tiers.sh"}


@needs_protocol_data
def test_runner_judge_only_without_generation_fails_before_claiming(stub_world):
    rc, out = stub_world.run("--judge-only")
    assert rc == 1, out
    assert "--judge-only but the generation is not finalized" in out and "claimed tvm_" not in out


@needs_protocol_data
def test_runner_pending_judge_marker_and_run_tiers_6(stub_world):
    stub_world.finalize()
    rc, out = stub_world.run(gpus="0|GPU-33333333-d|NVIDIA A40|46068|0", tiers=True)
    assert rc == 6, out
    marker = json.loads((stub_world.root / "PENDING_JUDGE.seed42").read_text())
    assert "cannot hold Qwen3-32B" in marker["reason"] and "--judge-only" in marker["task_line"]
    assert not list(stub_world.claims.iterdir())


@needs_protocol_data
def test_runner_serves_cuda_ordinals_for_uuid_devices(stub_world):
    stub_world.finalize()
    two = "0|GPU-11111111-a|NVIDIA H200|143771|0;1|GPU-22222222-c|NVIDIA H200|143771|0"
    rc, out = stub_world.run(gpus=two, cvd="GPU-22222222-c", TRC_POOL_GRACE="1")
    assert rc == 5, out  # the stub judge pool never becomes ready
    assert stub_world.pool_log.read_text().split()[0] == "GPU_GROUPS=[1]"


def procs_naming(path: Path) -> list[int]:
    found = []
    for d in Path("/proc").iterdir():
        try:
            if d.name.isdigit() and str(path).encode() in (d / "cmdline").read_bytes():
                found.append(int(d.name))
        except OSError:
            continue
    return found


@needs_protocol_data
def test_runner_term_during_scoring_releases_claims_and_lease(stub_world):
    """A TERM to the runner alone (not its group) during the hours-long score campaign is handled at once: the
    campaign and the judge pool are stopped and the claims and the lease released well inside Slurm's KillWait."""

    stub_world.finalize()
    p = stub_world.start(STUB_POOL_OK="1", STUB_SCORE_SLEEP="600")
    try:
        deadline = time.time() + 120
        while "score queue" not in stub_world.log() and time.time() < deadline and p.poll() is None:
            time.sleep(1)
        assert "score queue" in stub_world.log(), stub_world.log()
        time.sleep(2)
        assert list(stub_world.claims.iterdir()) and (stub_world.root / ".owner.seed42").exists()
        t0 = time.time()
        p.send_signal(15)
        rc = p.wait(timeout=60)
        took = time.time() - t0
    finally:
        if p.poll() is None:
            os.killpg(p.pid, 9)
    assert rc == 143 and took < 40, (rc, took)
    out = stub_world.log()
    assert "signal TERM" in out and "exit rc=143" in out
    assert not list(stub_world.claims.iterdir()) and not (stub_world.root / ".owner.seed42").exists()
    time.sleep(1)
    assert not procs_naming(stub_world.clone / "rlvr/evaluation/scripts/run_trace_eval_score_campaign.py")
