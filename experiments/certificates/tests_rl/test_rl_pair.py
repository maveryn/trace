"""Pair mode on CPU: jobs/rl_pair.sh splits one 4-GPU allocation into two disjoint 2-GPU runs (narrowed masks,
N=2, slots, host-RAM share), combines their exit codes, restarts an incomplete run once, forwards signals; and the
train_rl.py pieces it relies on: the host-RAM gate per share, one Ray temp dir / vLLM cache per launch, a Ray
cleanup that never touches the other run's processes, hardware resolution on a GPU subset.

rl_pair.sh runs against a stand-in nvidia-smi and a stand-in child (TRC_RL_PAIR_CHILD) that records its
environment; nothing here touches a GPU. The optional two-Ray-instances test needs TRC_TEST_RAY=1 (it starts two
real local Ray instances; run it on a compute node, not on a login node)."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from rl_helpers import CERT_DIR, TRACE_ROOT

STUB_SMI = r"""#!/usr/bin/env bash
n=${STUB_N:-4}; mib=${STUB_MIB:-97887}
IFS=, read -r -a names <<<"${STUB_NAMES:-NVIDIA RTX PRO 6000 Blackwell Server Edition}"
uuid() { printf 'GPU-%08d-aaaa-bbbb-cccc-dddddddddddd' "$1"; }
name() { local k=$(( $1 % ${#names[@]} )); printf '%s' "${names[$k]}"; }
case "$*" in
  *query-compute-apps*) ;;
  *query-gpu=index,uuid,name,memory.total,memory.used*) for i in $(seq 0 $((n - 1))); do echo "$i, $(uuid "$i"), $(name "$i"), $mib, 0"; done ;;
  *query-gpu=index,uuid,name,memory.total,driver_version*) for i in $(seq 0 $((n - 1))); do echo "$i, $(uuid "$i"), $(name "$i"), $mib, 595.00"; done ;;
  *query-gpu=index,uuid,memory.used,utilization.gpu*) for i in $(seq 0 $((n - 1))); do echo "$i, $(uuid "$i"), 0, 0"; done ;;
  *query-gpu=index,uuid,memory.used*) for i in $(seq 0 $((n - 1))); do echo "$i, $(uuid "$i"), 0"; done ;;
  *query-gpu=index,pci.bus_id*) for i in $(seq 0 $((n - 1))); do printf '%s, 00000000:%02X:00.0\n' "$i" $((16 * i + 33)); done ;;
  *) echo "stub nvidia-smi $*" ;;
esac
"""

# stand-in rl.sh: records its environment, then exits with the code for this try (STUB_RC_<slot>="1,0" = rc 1 on
# the first try, 0 on the second), after STUB_SLEEP_<slot> seconds; TERM -> marker file + exit 143
STUB_CHILD = r"""#!/usr/bin/env bash
slot=${TRC_RL_SLOT:?}
out=${PAIR_TEST_OUT:?}
try=1; while [ -e "$out/$slot.$try.json" ]; do try=$((try + 1)); done
trap 'echo term > "$out/$slot.$try.term"; exit 143' TERM
printf '{"pid": %s, "pgid": %s, "cvd": "%s", "n": "%s", "slot": "%s", "pair": "%s", "share": "%s", "args": "%s", "cpus": "%s"}\n' \
  "$$" "$(ps -o pgid= -p $$ | tr -d ' ')" "$CUDA_VISIBLE_DEVICES" "$N" "$slot" "$TRC_RL_PAIR" "$TRC_RL_HOST_RAM_SHARE" "$*" \
  "$(sed -n 's/^Cpus_allowed_list:[[:space:]]*//p' /proc/self/status)" > "$out/$slot.$try.json"
echo "[stub] slot $slot try $try"
rcs_var=STUB_RC_$slot; sleep_var=STUB_SLEEP_$slot
IFS=, read -r -a rcs <<<"${!rcs_var:-0}"
rc=${rcs[$((try - 1))]:-${rcs[-1]}}
s=${!sleep_var:-0}
end=$(( $(date +%s) + s ))
while [ "$(date +%s)" -lt "$end" ]; do sleep 0.2; done
exit "$rc"
"""

# the wrapper under test (TRC_TEST_RL_PAIR_SCRIPT: a candidate copy next to a copy of common.sh)
PAIR_SCRIPT = Path(os.environ.get("TRC_TEST_RL_PAIR_SCRIPT") or CERT_DIR / "jobs" / "rl_pair.sh")
RUN_A = "--model 3b --arm G --seed 3 --sft-dir /sft/dir"
RUN_B = "--model 3b --arm P --seed 3 --sft-dir /sft/dir"


@pytest.fixture()
def pair_env(tmp_path, rlvr_env):
    for name in ("TRACE_TOOL_ROOT", "HF_HOME", "TRACE_CERT_PYTHON", "TRC_CLAIM_DIR"):
        if not os.environ.get(name):
            pytest.skip(f"{name} not set (source $TRACE_CERT_RUNS/host/env.sh)")
    smi = tmp_path / "nvidia-smi"
    smi.write_text(STUB_SMI)
    smi.chmod(0o755)
    child = tmp_path / "child.sh"
    child.write_text(STUB_CHILD)
    out = tmp_path / "out"
    out.mkdir()
    claims = tmp_path / "claims"
    claims.mkdir()
    env = dict(os.environ)
    env.update({
        "TRC_NVIDIA_SMI": str(smi), "TRC_CLAIM_DIR": str(claims), "TRC_RL_PAIR_CHILD": str(child),
        "TRC_RL_PAIR_ROOT": str(tmp_path / "rl_pair"), "PAIR_TEST_OUT": str(out), "SLURM_JOB_ID": "test",
        "TRC_RL_PAIR_STAGGER": "0", "TRC_RL_PAIR_RESTART_DELAY": "1", "TRC_RL_PAIR_STOP_WAIT": "20",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    })
    for name in ("N", "SLURM_STEP_ID", "TRC_RL_SLOT", "TRC_RL_PAIR", "TRC_RL_HOST_RAM_SHARE"):
        env.pop(name, None)
    return env, tmp_path


def run_pair(env, *args, timeout=120, **extra):
    env = dict(env)
    env.update({k: str(v) for k, v in extra.items()})
    proc = subprocess.run(["bash", str(PAIR_SCRIPT), *args], cwd=str(TRACE_ROOT), env=env,
                          capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout + proc.stderr


def records(root):
    out = {}
    for path in sorted((root / "out").glob("*.json")):
        slot, attempt = path.stem.split(".")
        out[(slot, int(attempt))] = json.loads(path.read_text())
    return out


def attempt_json(root):
    files = list((root / "rl_pair").glob("*/attempts/*.json"))
    assert len(files) == 1, files
    return json.loads(files[0].read_text())


def test_pair_splits_four_gpus_into_disjoint_narrowed_pairs(pair_env):
    env, root = pair_env
    rc, out = run_pair(env, RUN_A, RUN_B)
    assert rc == 0, out
    rec = records(root)
    a, b = rec[("a", 1)], rec[("b", 1)]
    assert a["cvd"] == "0,1" and b["cvd"] == "2,3"
    assert a["n"] == b["n"] == "2" and a["share"] == b["share"] == "2"
    assert a["pair"] == b["pair"] == "3b_G_s3+P_s3"
    assert a["args"] == RUN_A and b["args"] == RUN_B
    assert a["pid"] != b["pid"] and a["pid"] == a["pgid"] and b["pid"] == b["pgid"]   # own process group each
    att = attempt_json(root)
    assert att["final_rc"] == 0 and att["n_gpus_per_run"] == 2 and "RTX PRO 6000" in att["gpu_class"]
    assert set(att["runs"]["a"]["uuids"].split(",")).isdisjoint(att["runs"]["b"]["uuids"].split(","))
    assert "slot a (3b/G/s3): CUDA_VISIBLE_DEVICES=0,1" in out and "-> exit 0" in out


def test_pair_keeps_the_inherited_token_form_and_never_widens(pair_env):
    env, root = pair_env
    uuid = lambda i: f"GPU-{i:08d}-aaaa-bbbb-cccc-dddddddddddd"  # noqa: E731
    rc, out = run_pair(env, RUN_A, RUN_B, STUB_N=6, CUDA_VISIBLE_DEVICES=",".join(uuid(i) for i in (5, 1, 4, 2)))
    assert rc == 0, out
    rec = records(root)
    assert rec[("a", 1)]["cvd"] == f"{uuid(1)},{uuid(2)}" and rec[("b", 1)]["cvd"] == f"{uuid(4)},{uuid(5)}"
    rc, out = run_pair(env, RUN_A, RUN_B, STUB_N=6, CUDA_VISIBLE_DEVICES="1,2,4,5")
    assert rc == 0, out
    assert {r["cvd"] for r in records(root).values()} >= {"1,2", "4,5"}


@pytest.mark.parametrize("n,names", [(2, None), (6, None), (4, "NVIDIA RTX PRO 6000 Blackwell Server Edition,NVIDIA H200 NVL")])
def test_pair_refuses_other_allocations(pair_env, n, names):
    env, root = pair_env
    extra = {"STUB_N": n, "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(n))}
    if names:
        extra["STUB_NAMES"] = names
    rc, out = run_pair(env, RUN_A, RUN_B, **extra)
    assert rc == 4, out
    assert not records(root)
    assert ("exactly 4 GPUs" in out) if not names else ("mixed GPU types" in out)


def test_usage(pair_env):
    env, root = pair_env
    assert run_pair(env, RUN_A)[0] == 2
    assert run_pair(env, RUN_A, RUN_A)[0] == 2                                   # the same run twice
    assert run_pair(env, RUN_A, "--model 3b --arm P --seed 3")[0] == 2           # neither --sft-dir nor --smoke
    assert run_pair(env, RUN_A, "--model 3b --arm P --seed 3 --sft-dir /x --bogus")[0] == 2
    assert run_pair(env, RUN_A, "--model 3b --arm Q --seed 3 --smoke")[0] == 2
    assert not records(root)


@pytest.mark.parametrize("rc_a,rc_b,want", [("0", "1", 1), ("0", "3", 3), ("4", "0", 4), ("3", "1", 1), ("2", "3", 2),
                                            ("137", "0", 137), ("0", "0", 0)])
def test_exit_code_is_the_worst_child(pair_env, rc_a, rc_b, want):
    env, _ = pair_env
    rc, out = run_pair(env, RUN_A, RUN_B, STUB_RC_a=rc_a, STUB_RC_b=rc_b, TRC_RL_PAIR_RESTARTS=0)
    assert rc == want, out


def test_incomplete_run_restarts_once_while_the_other_trains(pair_env):
    env, root = pair_env
    rc, out = run_pair(env, RUN_A, RUN_B, STUB_RC_a="1,0", STUB_SLEEP_b=6)
    assert rc == 0, out
    rec = records(root)
    assert ("a", 2) in rec and ("a", 3) not in rec and ("b", 2) not in rec
    assert rec[("a", 2)]["cvd"] == "0,1"
    assert attempt_json(root)["runs"]["a"]["tries"] == 2
    # a second failure is not restarted again
    for f in (root / "out").iterdir():
        f.unlink()
    rc, out = run_pair(env, RUN_A, RUN_B, STUB_RC_a="1,1,0", STUB_SLEEP_b=8)
    assert rc == 1, out
    assert ("a", 3) not in records(root)


def test_no_restart_once_the_other_run_is_done(pair_env):
    env, root = pair_env
    rc, out = run_pair(env, RUN_A, RUN_B, STUB_RC_a="1,0", STUB_SLEEP_a=2)
    assert rc == 1, out
    assert ("a", 2) not in records(root)


def test_signal_is_forwarded_to_both_runs(pair_env):
    env, root = pair_env
    env = dict(env, STUB_SLEEP_a="60", STUB_SLEEP_b="60")
    proc = subprocess.Popen(["bash", str(PAIR_SCRIPT), RUN_A, RUN_B], cwd=str(TRACE_ROOT), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + 30
    while time.time() < deadline and len(records(root)) < 2:
        time.sleep(0.2)
    assert len(records(root)) == 2
    t0 = time.time()
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=30)
    assert proc.returncode == 143, out
    assert time.time() - t0 < 15
    assert (root / "out" / "a.1.term").is_file() and (root / "out" / "b.1.term").is_file()
    assert attempt_json(root)["final_rc"] == 143


def fake_sysfs(root, gpu_nodes, halves):
    for i, node in enumerate(gpu_nodes):
        dev = root / "bus" / "pci" / "devices" / f"0000:{16 * i + 33:02x}:00.0"
        dev.mkdir(parents=True)
        (dev / "numa_node").write_text(f"{node}\n")
    for node, cpus in enumerate(halves):
        d = root / "devices" / "system" / "node" / f"node{node}"
        d.mkdir(parents=True)
        (d / "cpulist").write_text(",".join(str(c) for c in cpus) + "\n")


def cpulist(text):
    out = set()
    for part in filter(None, text.split(",")):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def test_pair_binds_each_run_to_the_numa_node_of_its_gpus(pair_env):
    env, root = pair_env
    mine = sorted(os.sched_getaffinity(0))
    if len(mine) < 24:
        pytest.skip(f"needs >= 24 CPUs in this process's affinity, has {len(mine)}")
    halves = (mine[: len(mine) // 2], mine[len(mine) // 2:])
    fake_sysfs(root / "sys", [0, 0, 1, 1], halves)
    allowed = ",".join(str(c) for c in mine)
    rc, out = run_pair(env, RUN_A, RUN_B, TRC_SYSFS=root / "sys", TRC_RL_PAIR_CPUS_ALLOWED=allowed)
    assert rc == 0, out
    rec = records(root)
    assert cpulist(rec[("a", 1)]["cpus"]) == set(halves[0]) and cpulist(rec[("b", 1)]["cpus"]) == set(halves[1])
    att = attempt_json(root)
    assert att["runs"]["a"]["numa"] == "0" and att["runs"]["b"]["numa"] == "1"
    # a pair split across nodes, or TRC_RL_PAIR_NUMA=0: no binding (the children keep the step's cpuset)
    for f in (root / "out").iterdir():
        f.unlink()
    rc, out = run_pair(env, RUN_A, RUN_B, TRC_SYSFS=root / "sys", TRC_RL_PAIR_CPUS_ALLOWED=allowed, TRC_RL_PAIR_NUMA=0)
    assert rc == 0 and "no CPU binding" in out, out
    assert cpulist(records(root)[("a", 1)]["cpus"]) == set(mine)


def test_pair_without_a_common_numa_node_is_not_bound(pair_env):
    env, root = pair_env
    mine = sorted(os.sched_getaffinity(0))
    fake_sysfs(root / "sys", [0, 1, 1, 1], (mine[:1], mine[1:]))
    rc, out = run_pair(env, RUN_A, RUN_B, TRC_SYSFS=root / "sys")
    assert rc == 0 and "no CPU binding: NUMA node of a GPU pair unknown or split" in out, out
    assert cpulist(records(root)[("b", 1)]["cpus"]) == set(mine)


def test_stagger_delays_slot_b(pair_env):
    env, root = pair_env
    rc, out = run_pair(env, RUN_A, RUN_B, TRC_RL_PAIR_STAGGER=3, STUB_SLEEP_a=5)
    assert rc == 0, out
    a = (root / "out" / "a.1.json").stat().st_mtime
    b = (root / "out" / "b.1.json").stat().st_mtime
    assert b - a >= 2.5


# ------------------------------------------------------------------------------ train_rl.py side


def test_host_ram_gate_uses_the_share(trl, specs, monkeypatch):
    monkeypatch.delenv("TRC_RL_MIN_HOST_RAM_GB", raising=False)
    mem = lambda gb: (lambda: {"cgroup": "v1:/x", "limit_gb": gb, "used_gb": 1.0, "peak_gb": 2.0})  # noqa: E731
    monkeypatch.setattr(trl, "host_memory", mem(800.0))
    monkeypatch.setenv("TRC_RL_HOST_RAM_SHARE", "2")
    info = trl.check_host_ram(specs["3b"], 2)
    assert info["share"] == 2 and info["share_gb"] == 400.0 and info["min_gb"] == 200
    monkeypatch.setattr(trl, "host_memory", mem(399.0))              # 199.5 GB each < 200
    with pytest.raises(trl.ConfigError, match="shared by 2 concurrent runs: 199.5 GB each"):
        trl.check_host_ram(specs["3b"], 2)
    monkeypatch.setenv("TRC_RL_HOST_RAM_SHARE", "1")
    assert trl.check_host_ram(specs["3b"], 2)["share_gb"] == 399.0
    monkeypatch.delenv("TRC_RL_HOST_RAM_SHARE")
    assert trl.check_host_ram(specs["3b"], 2)["share"] == 1
    monkeypatch.setattr(trl, "host_memory", mem(640.0))              # 7B on N=2 needs 320 each
    monkeypatch.setenv("TRC_RL_HOST_RAM_SHARE", "2")
    assert trl.check_host_ram(specs["7b"], 2)["share_gb"] == 320.0
    for bad in ("0", "x", "9", "-1"):
        monkeypatch.setenv("TRC_RL_HOST_RAM_SHARE", bad)
        with pytest.raises(trl.ConfigError, match="TRC_RL_HOST_RAM_SHARE"):
            trl.check_host_ram(specs["3b"], 2)


def test_ray_tmp_and_vllm_cache_are_unique_per_launch(trl, tmp_path):
    a = trl.ray_tmp_dir("2190785", 4321, "a")
    b = trl.ray_tmp_dir("2190785", 4321, "b")
    solo = trl.ray_tmp_dir("2190785", 4321)
    assert len({a, b, solo}) == 3 and a.parent == Path("/tmp")
    assert solo.name == "trc_ray_2190785_4321"                        # single runs keep the old name
    assert trl.vllm_cache_dir(a) != trl.vllm_cache_dir(b) and trl.vllm_cache_dir(a).name.startswith("trc_vllm_")
    assert len(str(trl.ray_tmp_dir("123456789012345", 4194303, "b"))) <= 40   # AF_UNIX socket paths stay short


def test_launch_env_per_slot(trl, specs, rlvr_env, tmp_path, monkeypatch):
    paths = trl.Paths(specs["3b"])
    monkeypatch.setenv("RAY_ADDRESS", "10.0.0.1:6379")
    monkeypatch.setenv("VLLM_CACHE_ROOT", "/shared/nfs/vllm")      # a shared default is always replaced
    envs = {}
    for slot in ("a", "b", None):
        if slot:
            monkeypatch.setenv("TRC_RL_SLOT", slot)
        else:
            monkeypatch.delenv("TRC_RL_SLOT")
        ray_tmp = trl.ray_tmp_dir("j", 77, slot)
        envs[slot] = trl.launch_env(paths, tmp_path / str(slot), ray_tmp, 2)
        assert envs[slot]["RAY_TMPDIR"] == str(ray_tmp) and "RAY_ADDRESS" not in envs[slot]
        assert envs[slot]["VLLM_CACHE_ROOT"] == str(trl.vllm_cache_dir(ray_tmp))
        assert envs[slot]["TRITON_CACHE_DIR"] == str(trl.compile_cache_dirs(ray_tmp)["TRITON_CACHE_DIR"])
        assert all(Path(envs[slot][k]).parent == Path("/tmp") for k in ("VLLM_CACHE_ROOT", "TRITON_CACHE_DIR"))
    for key in ("VLLM_CACHE_ROOT", "TRITON_CACHE_DIR", "HF_DATASETS_CACHE", "RAY_TMPDIR"):
        assert len({envs[s][key] for s in envs}) == 3, key
    monkeypatch.setenv("TRC_RL_SLOT", "ab")
    with pytest.raises(trl.ConfigError):
        trl.pair_slot()


def test_ray_cleanup_matches_whole_path_components_only(trl, tmp_path):
    proc = tmp_path / "proc"
    cmdlines = {
        101: ["raylet", "--temp_dir=/tmp/trc_ray_J_12_a", "--session_dir=/tmp/trc_ray_J_12_a/session_x"],
        102: ["python", "default_worker.py", "--session-dir=/tmp/trc_ray_J_12_a/session_x/logs"],
        103: ["raylet", "--temp_dir=/tmp/trc_ray_J_123_a"],                      # another launch, longer pid
        104: ["raylet", "--temp_dir=/tmp/trc_ray_J_12_b", "--x=/tmp/trc_ray_J_12_b/s"],  # the other slot
        105: ["gcs_server", "--log_dir=/tmp/trc_ray_J_12_ab/session/logs"],
        106: ["vllm", "--cache=/tmp/trc_vllm_J_12_a/x"],
    }
    for pid, args in cmdlines.items():
        (proc / str(pid)).mkdir(parents=True)
        (proc / str(pid) / "cmdline").write_bytes(b"\0".join(a.encode() for a in args) + b"\0")
    (proc / "self").mkdir()
    assert sorted(trl.ray_session_pids(Path("/tmp/trc_ray_J_12_a"), proc=proc)) == [101, 102]
    assert sorted(trl.ray_session_pids(Path("/tmp/trc_ray_J_12_b"), proc=proc)) == [104]
    assert trl.ray_session_pids(Path("/tmp/trc_ray_J_1"), proc=proc) == []


def test_placement_record(trl, monkeypatch):
    hw = {"name": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "n": 2}
    monkeypatch.setenv("TRC_RL_SLOT", "b")
    monkeypatch.setenv("TRC_RL_PAIR", "3b_G_s3+P_s3")
    monkeypatch.setenv("TRC_RL_HOST_RAM_SHARE", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    got = trl.placement(hw)
    assert got == {"gpu_class": "NVIDIA RTX PRO 6000 Blackwell Server Edition x2", "n_gpus": 2, "mode": "pair", "slot": "b",
                   "pair": "3b_G_s3+P_s3", "host_ram_share": 2, "cuda_visible_devices": "2,3"}
    for name in ("TRC_RL_SLOT", "TRC_RL_PAIR", "TRC_RL_HOST_RAM_SHARE"):
        monkeypatch.delenv(name)
    assert trl.placement(hw)["mode"] == "single" and trl.placement(hw)["host_ram_share"] == 1


def test_hardware_resolves_on_a_gpu_subset(trl, specs, tmp_path, monkeypatch):
    smi = tmp_path / "nvidia-smi"
    smi.write_text(STUB_SMI)
    smi.chmod(0o755)
    monkeypatch.setenv("TRC_NVIDIA_SMI", str(smi))
    monkeypatch.setenv("STUB_N", "4")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    rows = trl.gpu_inventory()
    assert [r["index"] for r in rows] == ["2", "3"]
    hw = trl.check_hardware(specs["3b"], rows, 2)
    assert hw["n"] == 2 and hw["name"].startswith("NVIDIA RTX PRO 6000") and len(set(hw["uuids"])) == 2
    with pytest.raises(trl.ConfigError):
        trl.check_hardware(specs["7b"], rows, 2)                          # 7B on 2 cards needs 120 GB each
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2,3")
    with pytest.raises(trl.ConfigError, match="--n-gpus 2 but 3"):
        trl.check_hardware(specs["3b"], trl.gpu_inventory(), 2)


def test_rl_job_dry_run_on_a_narrowed_pair(pair_env, base_snapshot, data_ready):
    """rl.sh itself on slot b's two cards of a 4-GPU stand-in: gate passes with N=2, check resolves N=2.
    The check imports vLLM, which asks NVML about CUDA_VISIBLE_DEVICES=2,3: run it where 4 real GPUs exist."""
    try:
        real = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60).stdout.count("GPU ")
    except (OSError, subprocess.SubprocessError):
        real = 0
    if real < 4:
        pytest.skip(f"needs a node with >= 4 GPUs (vLLM's NVML lookup of devices 2,3); this one has {real}")
    env, root = pair_env
    env = dict(env, CUDA_VISIBLE_DEVICES="2,3", TRC_RL_SLOT="b", TRC_RL_PAIR="t", TRC_RL_HOST_RAM_SHARE="2", N="2",
               TRC_RL_ROOT=str(root / "rl"), TRC_RL_SMOKE_ROOT=str(root / "rl_smoke"), TRC_RL_PYTHON=sys.executable)
    proc = subprocess.run(["bash", str(CERT_DIR / "jobs" / "rl.sh"), "--model", "3b", "--arm", "G", "--seed", "3",
                           "--sft-dir", str(base_snapshot), "--dry-run"], cwd=str(TRACE_ROOT), env=env,
                          capture_output=True, text=True, timeout=1800)
    out = proc.stdout + proc.stderr
    if proc.returncode == 4 and "cannot score the frozen" in out:
        pytest.skip("check refuses: the frozen store predates the current certificate code")
    assert proc.returncode == 0, out
    assert "gpu2 GPU-00000002" in out and "gpu3 GPU-00000003" in out and "gpu0 " not in out
    check = json.loads(next((root / "rl" / "3b" / "G" / "s3" / "logs").glob("job.*/check.json")).read_text())
    assert check["arm_diff"].keys() == {"2"}
    assert not any((root / "claims").iterdir())


RAY_PROBE = r"""
import json, os, sys, time
import ray
ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true"}}, _temp_dir=os.environ["RAY_TMPDIR"],
         num_cpus=2, include_dashboard=False)
ctx = ray.get_runtime_context()
nodes = [n["NodeID"] for n in ray.nodes() if n["Alive"]]
print("RAY " + json.dumps({"gcs": ctx.gcs_address, "node": ctx.get_node_id(), "nodes": nodes}), flush=True)
stop = sys.argv[1]
while not os.path.exists(stop):
    time.sleep(0.2)
ray.shutdown()
"""


@pytest.mark.skipif(os.environ.get("TRC_TEST_RAY") != "1", reason="starts two real Ray instances (TRC_TEST_RAY=1)")
def test_two_ray_instances_on_one_node_do_not_share_a_cluster(trl, tmp_path):
    """EasyR1 calls ray.init(runtime_env=..., _temp_dir=$RAY_TMPDIR) with no address: the second launch on a
    node must start its own instance (not attach to the first one's GCS) and choose different ports."""
    procs, info = [], {}
    stop = tmp_path / "stop"
    try:
        for slot in ("a", "b"):
            env = dict(os.environ, RAY_TMPDIR=str(trl.ray_tmp_dir("t", os.getpid(), slot)))
            env.pop("RAY_ADDRESS", None)
            proc = subprocess.Popen([sys.executable, "-c", RAY_PROBE, str(stop)], env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            procs.append(proc)
            for line in proc.stdout:
                if line.startswith("RAY "):
                    info[slot] = json.loads(line[4:])
                    break
        assert set(info) == {"a", "b"}, info
        assert info["a"]["gcs"] != info["b"]["gcs"]
        assert info["a"]["node"] != info["b"]["node"]
        assert info["a"]["nodes"] == [info["a"]["node"]] and info["b"]["nodes"] == [info["b"]["node"]]
    finally:
        stop.write_text("1")
        for proc in procs:
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
        for slot in ("a", "b"):
            trl.kill_ray(trl.ray_tmp_dir("t", os.getpid(), slot))


def test_ray_listen_ports_keeps_only_this_launch(trl, monkeypatch):
    ss = "\n".join([
        'LISTEN 0 4096 10.0.0.5:41001 0.0.0.0:* users:(("gcs_server",pid=11,fd=19))',
        'LISTEN 0 4096 10.0.0.5:41002 0.0.0.0:* users:(("raylet",pid=12,fd=30))',
        'LISTEN 0 4096 127.0.0.1:52365 0.0.0.0:* users:(("python",pid=13,fd=8))',
        'LISTEN 0 4096 10.0.0.5:42001 0.0.0.0:* users:(("gcs_server",pid=21,fd=19))',   # the other run
        'LISTEN 0 4096 [::]:8265 [::]:* users:(("python",pid=99,fd=5))',               # the driver
    ])
    monkeypatch.setattr(trl, "ray_session_pids", lambda ray_tmp, proc=None: [11, 12, 13])
    monkeypatch.setattr(trl.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=ss, stderr=""))
    assert trl.ray_listen_ports(Path("/tmp/trc_ray_J_1_a"), [99]) == {"gcs_server": [41001], "python": [8265, 52365],
                                                                      "raylet": [41002]}
