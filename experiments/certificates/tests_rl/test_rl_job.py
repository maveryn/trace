"""jobs/rl.sh gates on CPU with a stand-in nvidia-smi: N not in {2,4} -> 4, VRAM below the spec -> 4,
usage -> 2, --dry-run on a good allocation runs `train_rl.py check` and exits 0 without claiming."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from rl_helpers import CERT_DIR, TRACE_ROOT

STUB = r"""#!/usr/bin/env bash
n=${STUB_N:-2}; mib=${STUB_MIB:-81920}; name=${STUB_NAME:-NVIDIA A100 80GB PCIe}
uuid() { printf 'GPU-%08d-aaaa-bbbb-cccc-dddddddddddd' "$1"; }
case "$*" in
  *query-compute-apps*) ;;
  *query-gpu=index,uuid,name,memory.total,memory.used*) for i in $(seq 0 $((n - 1))); do echo "$i, $(uuid "$i"), $name, $mib, 0"; done ;;
  *query-gpu=index,uuid,name,memory.total,driver_version*) for i in $(seq 0 $((n - 1))); do echo "$i, $(uuid "$i"), $name, $mib, 595.00"; done ;;
  *query-gpu=index,uuid,memory.used,utilization.gpu*) for i in $(seq 0 $((n - 1))); do echo "$i, $(uuid "$i"), 0, 0"; done ;;
  *query-gpu=index,uuid,memory.used*) for i in $(seq 0 $((n - 1))); do echo "$i, $(uuid "$i"), 0"; done ;;
  *) echo "stub nvidia-smi" ;;
esac
"""


@pytest.fixture(scope="module")
def job_env(tmp_path_factory, rlvr_env):
    for name in ("TRACE_TOOL_ROOT", "HF_HOME", "TRACE_CERT_PYTHON", "TRC_CLAIM_DIR"):
        if not os.environ.get(name):
            pytest.skip(f"{name} not set (source $TRACE_CERT_RUNS/host/env.sh)")
    root = tmp_path_factory.mktemp("rljob")
    stub = root / "nvidia-smi"
    stub.write_text(STUB)
    stub.chmod(0o755)
    claims = root / "claims"
    claims.mkdir()
    env = dict(os.environ)
    env.update({
        "TRC_NVIDIA_SMI": str(stub),
        "TRC_CLAIM_DIR": str(claims),
        "TRC_RL_ROOT": str(root / "rl"),
        "TRC_RL_SMOKE_ROOT": str(root / "rl_smoke"),
        "TRC_RL_PYTHON": sys.executable,
        "SLURM_JOB_ID": "test",
    })
    for name in ("CUDA_VISIBLE_DEVICES", "N", "SLURM_STEP_ID"):
        env.pop(name, None)
    return env, root


def run_job(env, *args, **stub):
    env = dict(env)
    env.update({f"STUB_{k.upper()}": str(v) for k, v in stub.items()})
    proc = subprocess.run(["bash", str(CERT_DIR / "jobs" / "rl.sh"), *args], cwd=str(TRACE_ROOT), env=env,
                          capture_output=True, text=True, timeout=1800)
    return proc.returncode, proc.stdout + proc.stderr


@pytest.mark.parametrize("n", [1, 3, 8])
def test_refuses_gpu_counts(job_env, n):
    env, root = job_env
    rc, out = run_job(env, "--model", "3b", "--arm", "P", "--seed", "1", "--smoke", n=n)
    assert rc == 4, out
    assert "N in {2,4}" in out
    assert not any((root / "claims").iterdir())


@pytest.mark.parametrize("model,n,mib", [("3b", 2, 49140), ("3b", 4, 24463), ("7b", 2, 97887), ("7b", 2, 81920)])
def test_refuses_small_cards(job_env, model, n, mib):
    env, _ = job_env
    rc, out = run_job(env, "--model", model, "--arm", "G", "--seed", "1", "--smoke", n=n, mib=mib)
    assert rc == 4, out
    assert "needs >=" in out


def test_usage(job_env):
    env, _ = job_env
    assert run_job(env, "--model", "3b", "--arm", "P", "--seed", "1")[0] == 2          # neither --sft-dir nor --smoke
    assert run_job(env, "--model", "5b", "--arm", "P", "--seed", "1", "--smoke")[0] == 2
    assert run_job(env, "--model", "3b", "--arm", "Q", "--seed", "1", "--smoke")[0] == 2
    assert run_job(env, "--model", "3b", "--arm", "P", "--seed", "x", "--smoke")[0] == 2


def test_dry_run_passes_the_gate_and_checks(job_env, base_snapshot, data_ready):
    env, root = job_env
    rc, out = run_job(env, "--model", "3b", "--arm", "P_gated", "--seed", "2", "--sft-dir", str(base_snapshot), "--dry-run",
                      n=2, mib=143771, name="NVIDIA H200 NVL")
    if rc == 4:
        log = next((root / "rl" / "3b" / "P_gated" / "s2" / "logs").glob("rl.*.log")).read_text()
        if "cannot score the frozen" in log:
            pytest.skip("check refuses: the frozen store predates the current certificate code (rebuild the data)")
    assert rc == 0, out
    assert "check ok" in out and "dry run: no GPU touched" in out
    assert not any((root / "claims").iterdir())
    logs = list((root / "rl" / "3b" / "P_gated" / "s2" / "logs").glob("job.*/check.json"))
    assert logs, out
