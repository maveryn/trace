"""jobs/sft.sh end to end on CPU: stand-in nvidia-smi (2 fake 80 GB cards), a scratch claim dir,
the tiny model through a test config, and an oracle stand-in for the vLLM generator; everything
else is real (preflight with the EasyR1 token check, torchrun training, merge, the certificate
scorer in the trace_cert env, sft_select.py). Outputs go to a scratch root; the frozen data is
only read."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

import sft_helpers as h


@pytest.fixture(scope="module")
def job_env(tiny_model, data_root, cpu_env, rlvr_root, tmp_path_factory):
    for var in ("TRACE_CERT_RUNS", "TRACE_TOOL_ROOT", "HF_HOME", "TRACE_CERT_PYTHON"):
        if not os.environ.get(var):
            pytest.skip(f"{var} not set (source $TRACE_CERT_RUNS/host/env.sh)")
    tmp = tmp_path_factory.mktemp("sft_job")
    cfg = h.tiny_config(h.CONFIG_3B, tmp / "cfg.yaml", tiny_model, train__global_batch=4, train__per_device_batch=1,
                        smoke__train_rows=16, smoke__max_steps=2, smoke__dev_rows_per_task=2,
                        resources__min_card_mib=1000, resources__min_total_mib=2000)
    claims = tmp / "claims"
    claims.mkdir()
    env = dict(cpu_env)
    env.update({
        "TRC_SFT_CONFIG": str(cfg), "TRC_SFT_OUT_ROOT": str(tmp / "out"), "TRC_NVIDIA_SMI": str(h.write_fake_smi(tmp / "nvidia-smi")),
        "TRC_CLAIM_DIR": str(claims), "TRC_SFT_GEN_CMD": f"{sys.executable} {h.write_oracle_gen(tmp / 'oracle_gen.py')} {h.CERT_DIR}",
        "TRC_SFT_MIN_FREE_GB": "1", "TRC_CLAIM_SETTLE": "0", "TRC_SFT_POLL": "1", "TRC_SFT_UTIL_MIB": "0", "FAKE_GPUS": "2:81920",
        "TRC_DATA": str(data_root), "TRACE_RLVR_ROOT": str(rlvr_root), "TRC_RL_PYTHON": sys.executable,
    })
    for var in ("CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID", "SLURM_STEP_ID", "N", "FAKE_GPU_USED"):
        env.pop(var, None)
    return {"env": env, "tmp": tmp, "root": tmp / "out" / "qwen2.5-vl-3b", "claims": claims}


def _sft_sh(job_env, *args, **extra_env):
    env = {**job_env["env"], **extra_env}
    return h.run(["bash", str(h.SFT_SH), "3b", *args], env=env, timeout=2400)


def _log(res):
    return res.stdout[-4000:] + "\n--- stderr ---\n" + res.stderr[-4000:]


def test_sft_sh_smoke_end_to_end(job_env):
    root = job_env["root"]
    dry = _sft_sh(job_env, "--smoke", "--dry-run")
    assert dry.returncode == 0, _log(dry)
    assert (root / "check.json").exists() and json.loads((root / "check.json").read_text())["ok"]
    assert not (root / "ckpt").exists() and "dry run: no GPU touched" in dry.stdout

    res = _sft_sh(job_env, "--smoke")
    assert res.returncode == 0, _log(res)
    assert "COMPLETE: qwen2.5-vl-3b smoke SFT" in res.stdout
    assert os.readlink(root / "selected") == "ckpt/step_000002"
    receipt = json.loads((root / "receipt.json").read_text())
    assert receipt["smoke"] is True and receipt["selected"]["step"] == 2 and receipt["outcome"] == "fallback_last"
    (point,) = receipt["dev_eval"]["curve"]
    assert point["n"] == 8 and point["A"] == point["M"] == point["F"] == 1.0  # oracle responses score perfectly
    eq = receipt["token_identity_easyr1"]
    assert eq["sft_rows"]["all_equal"] and eq["e1_dev_rows"]["all_equal"]
    assert receipt["data"]["files_sha256"]["sft/sft.jsonl"]
    assert receipt["training"]["launches"][0]["world_size"] == 2
    run_config = json.loads((root / "dev" / "step_000002" / "run_config.json").read_text())
    assert run_config["model_id"] == "Qwen/Qwen2.5-VL-3B-Instruct" and run_config["condition"] == "cert_v1"
    # GPUs and the run lease are released; the claims were ours (trc_*) while they existed
    assert list(job_env["claims"].iterdir()) == []
    assert not (root / ".owner").exists()
    log = "".join(p.read_text() for p in sorted((root / "logs").glob("sft.*.log")))  # dry run + this run
    assert "claimed trc_" in log and "UTILCHECK ok" in log and "GPUs released" in log

    again = _sft_sh(job_env, "--smoke")  # a rerun only re-selects
    assert again.returncode == 0, _log(again)
    assert "training done; generation pending for 0 of 1 saves; scoring pending for 0" in again.stdout
    assert os.readlink(root / "selected") == "ckpt/step_000002"


def test_sft_sh_refuses_bad_allocations(job_env, tmp_path):
    env = {"TRC_SFT_OUT_ROOT": str(tmp_path / "out")}
    bad_n = _sft_sh(job_env, "--smoke", FAKE_GPUS="3:81920", **env)
    assert bad_n.returncode == 4 and "does not divide the global batch" in bad_n.stdout + bad_n.stderr, _log(bad_n)
    small = _sft_sh(job_env, "--smoke", FAKE_GPUS="2:900", **env)
    assert small.returncode == 4, _log(small)
    busy = _sft_sh(job_env, "--smoke", FAKE_GPU_USED="5000", **env)
    assert busy.returncode == 3 and "GPUs busy" in busy.stdout + busy.stderr, _log(busy)
    assert list(job_env["claims"].iterdir()) == []
    usage = h.run(["bash", str(h.SFT_SH), "13b"], env=job_env["env"])
    assert usage.returncode == 2
