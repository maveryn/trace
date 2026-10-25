"""Runner orchestration on CPU against a stand-in EasyR1: launch command, cwd, PYTHONPATH, RAY_TMPDIR,
keep/ linking while the trainer deletes old saves, a crash mid-run, resume on rerun with the attempt's
experiment log kept, merges of the kept steps + final, completion bookkeeping."""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

FAKE_MAIN = textwrap.dedent('''
    """Stand-in for `python -m verl.trainer.main config=...` with EasyR1's save / resume / log semantics."""
    import json, os, shutil, sys, time
    from pathlib import Path

    import yaml

    cfg = yaml.safe_load(open(sys.argv[1].split("=", 1)[1]))
    save = Path(cfg["trainer"]["save_checkpoint_path"])
    n = int(cfg["trainer"]["n_gpus_per_node"])
    max_steps, freq = int(cfg["trainer"]["max_steps"]), int(cfg["trainer"]["save_freq"])
    save.mkdir(parents=True, exist_ok=True)
    probe = {"cwd": os.getcwd(), "pythonpath": os.environ.get("PYTHONPATH", ""), "ray_tmpdir": os.environ.get("RAY_TMPDIR"),
             "hf_datasets_cache": os.environ.get("HF_DATASETS_CACHE"), "dont_write_bytecode": os.environ.get("PYTHONDONTWRITEBYTECODE"),
             "argv": sys.argv, "executable": sys.executable}
    with open(save.parent / "probe.jsonl", "a") as handle:
        handle.write(json.dumps(probe) + "\\n")
    open(save / "experiment_log.jsonl", "w").close()           # FileLogger truncates at every start
    tracker = save / "checkpoint_tracker.json"
    step = json.load(open(tracker))["last_global_step"] if tracker.exists() else 0
    if step:
        print(f"Load from checkpoint: {save}/global_step_{step}.", flush=True)
    best = 2
    crash = int(os.environ.get("FAKE_CRASH_AT", "0"))
    while step < max_steps:
        step += 1
        time.sleep(0.6)
        if crash and step == crash and not (save.parent / "crashed").exists():
            (save.parent / "crashed").write_text("1")
            print("simulated preemption", flush=True)
            sys.exit(1)
        with open(save / "experiment_log.jsonl", "a") as handle:
            handle.write(json.dumps({"step": step, "timing_s": {"step": 10.0, "gen": 6.0}, "reward": {"overall": step / 100,
                                     "accuracy": 0.5, "grounding": 0.4, "semantic": 0.3, "cert_format": 1.0}}) + "\\n")
        if step % freq == 0:
            for old in save.glob("global_step_*"):              # save_limit 2 with a best step: best + current
                if int(old.name.split("_")[-1]) not in (best, step):
                    shutil.rmtree(old)
            actor = save / f"global_step_{step}" / "actor"
            (actor / "huggingface").mkdir(parents=True)
            for rank in range(n):
                (actor / f"model_world_size_{n}_rank_{rank}.pt").write_bytes(f"{step}:{rank}".encode())
            (actor / "huggingface" / "config.json").write_text("{}")
            json.dump({"last_global_step": step, "best_global_step": best}, open(tracker, "w"))
    print("Finish validation.", flush=True)
''')


@pytest.fixture()
def fake_backend(tmp_path):
    backend = tmp_path / "backend"
    (backend / "verl" / "trainer").mkdir(parents=True)
    (backend / "verl" / "__init__.py").write_text("")
    (backend / "verl" / "trainer" / "__init__.py").write_text("")
    (backend / "verl" / "trainer" / "main.py").write_text(FAKE_MAIN)
    return backend


def test_crash_resume_keep_merge(trl, specs, rlvr_env, fake_backend, tmp_path, monkeypatch):
    spec = specs["3b"]
    paths = trl.Paths(spec)
    paths.backend = fake_backend
    run = tmp_path / "rl" / "3b" / "P" / "s1"
    (run / "logs").mkdir(parents=True)
    config = {"trainer": {"save_checkpoint_path": str(run / "ckpt"), "n_gpus_per_node": 2, "max_steps": 12, "save_freq": 2}}
    config_path = run / "resolved_easyr1.yaml"
    config_path.write_text(yaml.safe_dump(config))
    receipt = {"attempts": [], "keep": {}, "merged": {}, "status": "created", "model": "3b", "n_gpus": 2}
    (run / trl.RUN_RECEIPT).write_text(json.dumps(receipt))
    merged = []

    def fake_merge(spec_, paths_, run_dir, step, *, n_gpus, model_path):
        src = run_dir / "keep" / f"global_step_{step}"
        if not src.is_dir():
            src = run_dir / "ckpt" / f"global_step_{step}"
        assert trl.step_complete(src, n_gpus), step
        merged.append(step)
        out = run_dir / "hf" / f"global_step_{step}"
        out.mkdir(parents=True, exist_ok=True)
        payload = {"status": "ok", "step": step, "source": str(src)}
        (out / trl.MERGE_RECEIPT).write_text(json.dumps(payload))
        return payload

    monkeypatch.setattr(trl, "merge_step", fake_merge)
    monkeypatch.setenv("FAKE_CRASH_AT", "7")
    hardware = {"n": 2, "name": "FAKE GPU", "uuids": ["a", "b"]}
    kwargs = dict(hardware=hardware, save_freq=2, reason="test", max_steps=12, keep_steps=[4, 8], model_path=tmp_path,
                  n_gpus=2, vllm_python=None, stall_minutes=60, poll_seconds=0.2)

    with pytest.raises(trl.Incomplete):
        trl.train_merge_verify(spec, paths, run, trl.read_json(run / trl.RUN_RECEIPT), config_path, **kwargs)
    receipt = trl.read_json(run / trl.RUN_RECEIPT)
    assert receipt["attempts"][0]["rc"] == 1 and receipt["attempts"][0]["last_step"] == 6
    assert (run / "keep" / "global_step_4").is_dir()               # linked before the trainer deleted it
    assert not (run / "ckpt" / "global_step_4").exists()
    assert merged == []

    rc = trl.train_merge_verify(spec, paths, run, trl.read_json(run / trl.RUN_RECEIPT), config_path, **kwargs)
    assert rc == 0
    receipt = trl.read_json(run / trl.RUN_RECEIPT)
    assert receipt["status"] == "completed" and len(receipt["attempts"]) == 2
    assert receipt["attempts"][1]["resumed_from_step"] == 6 and receipt["attempts"][1]["rc"] == 0
    assert sorted(merged) == [4, 8, 12] and sorted(receipt["keep"]) == ["4", "8"]
    assert "Load from checkpoint" in (run / "logs" / "trainer.attempt2.log").read_text()
    # attempt 1's FileLogger output was kept before attempt 2 truncated it; steps 1..12 all present
    assert (run / "logs" / "experiment_log.attempt1.jsonl").is_file()
    assert [r["step"] for r in trl.experiment_records(run)] == list(range(1, 13))
    assert receipt["summary"]["logged_steps"] == 12 and receipt["summary"]["step_seconds_median"] == 10.0

    probes = [json.loads(line) for line in (run / "probe.jsonl").read_text().splitlines()]
    assert len(probes) == 2
    for probe in probes:
        assert Path(probe["cwd"]) == fake_backend
        parts = probe["pythonpath"].split(os.pathsep)
        assert parts[0] == str(fake_backend) and Path(parts[1]) == trl.TRACE_ROOT / "src"
        assert probe["ray_tmpdir"].startswith("/tmp/trc_ray_") and len(probe["ray_tmpdir"]) < 40
        assert not Path(probe["ray_tmpdir"]).exists()                # removed after the launch
        assert probe["hf_datasets_cache"] == str(run / "hf_datasets_cache")
        assert probe["dont_write_bytecode"] == "1"
        assert probe["argv"][1:] == [f"config={config_path}"] and probe["executable"] == sys.executable

    # a rerun of a completed run does not launch the trainer again
    rc = trl.train_merge_verify(spec, paths, run, trl.read_json(run / trl.RUN_RECEIPT), config_path, **kwargs)
    assert rc == 0 and len(trl.read_json(run / trl.RUN_RECEIPT)["attempts"]) == 2
    assert len((run / "probe.jsonl").read_text().splitlines()) == 2


def test_lost_keep_step_is_reported(trl, specs, rlvr_env, fake_backend, tmp_path, monkeypatch):
    """A kept step deleted before it could be linked makes the run incomplete (never silently dropped)."""
    spec = specs["3b"]
    paths = trl.Paths(spec)
    paths.backend = fake_backend
    run = tmp_path / "run"
    (run / "logs").mkdir(parents=True)
    config_path = run / "resolved_easyr1.yaml"
    config_path.write_text(yaml.safe_dump({"trainer": {"save_checkpoint_path": str(run / "ckpt"), "n_gpus_per_node": 2,
                                                       "max_steps": 6, "save_freq": 2}}))
    (run / trl.RUN_RECEIPT).write_text(json.dumps({"attempts": [], "keep": {}, "merged": {}, "status": "created"}))
    monkeypatch.setattr(trl, "link_keep", lambda *a, **k: [])      # the poller never runs
    monkeypatch.setattr(trl, "merge_step", lambda *a, **k: {"status": "ok"})
    with pytest.raises(trl.Incomplete, match="lost"):
        trl.train_merge_verify(spec, paths, run, trl.read_json(run / trl.RUN_RECEIPT), config_path,
                               hardware={"n": 2, "name": "FAKE", "uuids": []}, save_freq=2, reason="t", max_steps=6,
                               keep_steps=[4], model_path=tmp_path, n_gpus=2, vllm_python=None, stall_minutes=60,
                               poll_seconds=0.2)
    assert trl.read_json(run / trl.RUN_RECEIPT)["status"] == "incomplete_keep_lost"
