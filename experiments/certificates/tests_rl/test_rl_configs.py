"""Resolved EasyR1 configs: arms differ only in reward kwargs; only DESIGN 7.3 keys move off the release."""

from __future__ import annotations

import itertools
import re
from pathlib import Path

import pytest

from rl_helpers import CERT_DIR, DUMMY_SHA, TRACE_ROOT, resolve

ARM_KEYS = {"worker.reward.reward_function_kwargs.arm", "worker.reward.reward_function_kwargs.gated"}

# DESIGN 7.3 table (values identical for every arm)
DESIGN_VALUES = {
    "worker.rollout.tensor_parallel_size": 1,
    "data.rollout_batch_size": 64,
    "worker.actor.global_batch_size": 64,
    "worker.rollout.n": 8,
    "trainer.max_steps": 300,
    "data.max_prompt_length": 3072,
    "data.max_response_length": 2048,
    "trainer.val_freq": 50,
    "data.prompt_key": "prompt_cert",
    "data.answer_key": "answer_gt",
    "worker.reward.num_cpus": 4,
    "trainer.save_freq": 25,
    "trainer.save_limit": 2,
    "trainer.find_last_checkpoint": True,
}
# release values the runner must NOT change (optimizer, clip, sampling, batching, pixels, KL, ...)
UNCHANGED = (
    "worker.actor.optim.lr", "worker.actor.optim.lr_scheduler_type", "worker.actor.optim.weight_decay",
    "worker.actor.clip_ratio_low", "worker.actor.clip_ratio_high", "worker.actor.clip_ratio_dual",
    "worker.actor.micro_batch_size_per_device_for_update", "worker.actor.micro_batch_size_per_device_for_experience",
    "worker.actor.ppo_epochs", "worker.actor.loss_avg_mode", "worker.actor.max_grad_norm", "worker.actor.padding_free",
    "worker.actor.dynamic_batching", "worker.actor.model.freeze_vision_tower", "worker.actor.model.lora.rank",
    "worker.actor.offload.offload_params", "worker.actor.offload.offload_optimizer",
    "worker.rollout.temperature", "worker.rollout.top_p", "worker.rollout.top_k", "worker.rollout.gpu_memory_utilization",
    "worker.rollout.max_num_batched_tokens", "worker.rollout.val_override_config", "worker.rollout.enforce_eager",
    "data.min_pixels", "data.max_pixels", "data.val_batch_size", "data.shuffle", "data.filter_overlong_prompts",
    "data.image_key", "algorithm.adv_estimator", "algorithm.disable_kl", "algorithm.kl_coef", "algorithm.online_filtering",
    "trainer.val_before_train", "trainer.nnodes", "trainer.total_epochs", "worker.reward.skip_special_tokens",
)


@pytest.mark.parametrize("model", ["3b", "7b"])
@pytest.mark.parametrize("n_gpus", [2, 4])
def test_arms_differ_only_in_reward_kwargs(trl, specs, releases, model, n_gpus):
    spec, release = specs[model], releases[model]
    configs = {arm: resolve(trl, spec, release, arm=arm, n_gpus=n_gpus) for arm in trl.ARMS}
    diff = trl.arm_diff(configs)
    assert set(diff["differing_keys"]) == ARM_KEYS
    pairs = {(v["arm"], v["gated"]) for v in diff["reward_kwargs"].values()}
    assert pairs == {("A", False), ("G", False), ("P", False), ("P", True), ("G", True)}
    # without normalisation the only other differences are the run identity (directory, name)
    for a, b in itertools.combinations(trl.ARMS, 2):
        raw = set(trl.config_diff(configs[a], configs[b]))
        assert raw <= ARM_KEYS | trl.arm_identity_keys(), (a, b, raw)
        for key in raw & trl.arm_identity_keys():
            va, vb = trl.get_dotted(configs[a], key), trl.get_dotted(configs[b], key)
            assert va.replace(f"/{a}/", "/X/").replace(f"-{a}-", "-X-") == vb.replace(f"/{b}/", "/X/").replace(f"-{b}-", "-X-")


@pytest.mark.parametrize("model", ["3b", "7b"])
def test_only_design_keys_change(trl, specs, releases, model):
    spec, release = specs[model], releases[model]
    for arm, n_gpus in itertools.product(trl.ARMS, (2, 4)):
        config = resolve(trl, spec, release, arm=arm, n_gpus=n_gpus, seed=2)
        changed = set(trl.config_diff(release, config))
        allowed = trl.allowed_release_diff(spec)
        assert all(trl.key_allowed(key, allowed) for key in changed), sorted(changed)
        for key, value in DESIGN_VALUES.items():
            assert trl.get_dotted(config, key) == value, key
        assert config["trainer"]["n_gpus_per_node"] == n_gpus
        assert config["data"]["seed"] == 2 and config["worker"]["rollout"]["seed"] == 2
        assert config["worker"]["actor"]["model"]["model_path"] == "/sft/dir"
        assert config["worker"]["actor"]["model"]["tokenizer_path"] == "/sft/dir"
        for key in UNCHANGED:
            assert trl.get_dotted(config, key) == trl.get_dotted(release, key), key
        paths = trl.Paths(spec)
        assert config["data"]["train_files"] == str(paths.train_dir)
        assert config["data"]["val_files"] == str(paths.val_parquet)
        assert Path(config["data"]["system_prompt_file"]) == trl.SYSTEM_PROMPT
        fn_path, fn_name = config["worker"]["reward"]["reward_function"].rsplit(":", 1)
        assert Path(fn_path) == CERT_DIR / "reward_entry.py" and Path(fn_path).is_file() and fn_name == "compute_score"
        kwargs = config["worker"]["reward"]["reward_function_kwargs"]
        assert kwargs["alpha"] == 0.8 and kwargs["format_weight"] == 0.05 and kwargs["coord_frame"] == "px_source"
        assert kwargs["store_path"] == str(paths.store) and kwargs["store_sha256"] == DUMMY_SHA
        assert kwargs["arm"] == spec["reward"]["arms"][arm]["arm"] and kwargs["gated"] == spec["reward"]["arms"][arm]["gated"]
        assert kwargs["details_dir"].endswith(f"/{arm}/s2/reward_details")
        assert config["trainer"]["save_checkpoint_path"] == str(paths.run_dir(spec, arm, 2) / "ckpt")
        assert config["trainer"]["logger"] == ["console", "file"]


def test_reward_kwargs_match_compute_score_signature(trl, specs, releases):
    import inspect

    from trace_tasks.experimental.certificates.reward import compute_score

    params = inspect.signature(compute_score).parameters
    config = resolve(trl, specs["3b"], releases["3b"], arm="P_gated")
    kwargs = config["worker"]["reward"]["reward_function_kwargs"]
    assert set(kwargs) <= set(params) - {"reward_inputs"}
    required = {name for name, p in params.items() if p.default is inspect.Parameter.empty and p.kind == p.KEYWORD_ONLY}
    assert required <= set(kwargs)


@pytest.mark.parametrize("bad_n", [1, 3, 8])
def test_refuses_other_gpu_counts(trl, specs, releases, bad_n):
    with pytest.raises(trl.ConfigError):
        resolve(trl, specs["3b"], releases["3b"], n_gpus=bad_n)


@pytest.mark.parametrize("n_gpus", [2, 4])
def test_smoke_config(trl, specs, releases, n_gpus):
    spec, release = specs["3b"], releases["3b"]
    config = resolve(trl, spec, release, n_gpus=n_gpus, smoke=True, max_steps=1, val_files="/tmp/val.parquet")
    assert config["data"]["rollout_batch_size"] == 8 and config["worker"]["actor"]["global_batch_size"] == 8
    assert config["worker"]["rollout"]["n"] == 4 and config["trainer"]["val_freq"] == -1
    assert config["trainer"]["save_freq"] == 1 and config["trainer"]["max_steps"] == 1
    assert config["data"]["val_files"] == "/tmp/val.parquet"
    changed = set(trl.config_diff(release, config))
    assert all(trl.key_allowed(key, trl.allowed_release_diff(spec, smoke=True)) for key in changed)
    with pytest.raises(trl.ConfigError):
        resolve(trl, spec, release, n_gpus=n_gpus, smoke=False, max_steps=5)


def test_specs_differ_only_in_model_identity(trl, specs):
    allowed = {"config_id", "model", "release.config", "release.config_id", "release.sha256", "base_model.repo_id",
               "base_model.revision", "base_model.hidden_size", "base_model.num_hidden_layers", "hardware.min_vram_gb.2",
               "hardware.min_host_ram_gb.2", "hardware.min_host_ram_gb.4",
               "schedule.min_free_disk_gb", "_path", "_sha256"}
    assert set(trl.config_diff(specs["3b"], specs["7b"])) <= allowed
    assert specs["3b"]["hardware"]["n_gpus"] == [2, 4]
    assert min(specs["3b"]["hardware"]["min_vram_gb"].values()) >= 80
    # host-RAM gate: above the 164 GB measured on 2 x H200 NVL (and the 120 GB allocation that was OOM-killed)
    for spec in specs.values():
        assert set(spec["hardware"]["min_host_ram_gb"]) == set(spec["hardware"]["n_gpus"])
        assert min(spec["hardware"]["min_host_ram_gb"].values()) >= 200


def test_host_ram_gate(trl, specs, monkeypatch):
    monkeypatch.setattr(trl, "host_memory", lambda: {"cgroup": "v1:/x", "limit_gb": 120.0, "used_gb": 1.0, "peak_gb": 2.0})
    monkeypatch.delenv("TRC_RL_MIN_HOST_RAM_GB", raising=False)
    with pytest.raises(trl.ConfigError, match="host RAM"):
        trl.check_host_ram(specs["3b"], 2)
    monkeypatch.setattr(trl, "host_memory", lambda: {"cgroup": "v1:/x", "limit_gb": 360.0, "used_gb": 1.0, "peak_gb": 2.0})
    assert trl.check_host_ram(specs["3b"], 2)["min_gb"] == 200
    monkeypatch.setattr(trl, "host_memory", lambda: {"cgroup": None, "limit_gb": None, "used_gb": None, "peak_gb": None})
    assert trl.check_host_ram(specs["3b"], 2)["limit_gb"] is None   # unknown limit: no refusal
    monkeypatch.setattr(trl, "host_memory", lambda: {"cgroup": "v1:/x", "limit_gb": 120.0, "used_gb": 1.0, "peak_gb": 2.0})
    monkeypatch.setenv("TRC_RL_MIN_HOST_RAM_GB", "0")
    assert trl.check_host_ram(specs["3b"], 2)["min_gb"] == 0


def test_easyr1_dataclasses_accept_every_arm(trl, specs, releases):
    paths = trl.Paths(specs["3b"])
    for arm in trl.ARMS:
        trl.easyr1_structured(resolve(trl, specs["3b"], releases["3b"], arm=arm), paths.backend)
    trl.easyr1_structured(resolve(trl, specs["7b"], releases["7b"], arm="G", n_gpus=4), paths.backend)
    trl.easyr1_structured(resolve(trl, specs["3b"], releases["3b"], smoke=True, val_files="/tmp/v.parquet"), paths.backend)


def test_validate_config_rejects_bad_batches(trl, specs, releases):
    config = resolve(trl, specs["3b"], releases["3b"])
    config["worker"]["actor"]["global_batch_size"] = 32
    with pytest.raises(trl.ConfigError):
        trl.validate_config(config, 2)
    config = resolve(trl, specs["3b"], releases["3b"])
    config["data"]["max_prompt_length"] = 7000
    with pytest.raises(trl.ConfigError):
        trl.validate_config(config, 2)


def test_placeholders(trl):
    assert trl.substitute("${A}/x", {"A": "a"}) == "a/x"
    assert trl.substitute("${N}", {"N": 4}) == 4
    assert trl.substitute(["${N}", {"k": "${A}"}], {"N": 2, "A": "z"}) == [2, {"k": "z"}]
    with pytest.raises(trl.ConfigError):
        trl.substitute("${MISSING}", {})


def test_no_machine_paths_in_runner_files():
    """scripts/check_public_release.py hygiene: no absolute machine paths in our files (DESIGN 1)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("check_public_release", TRACE_ROOT / "scripts" / "check_public_release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files = [CERT_DIR / "train_rl.py", CERT_DIR / "configs" / "rl_3b.yaml", CERT_DIR / "configs" / "rl_7b.yaml",
             CERT_DIR / "jobs" / "rl.sh", *sorted((CERT_DIR / "tests_rl").glob("*.py"))]
    findings = {}
    for path in files:
        found = module.scan_text_for_sensitive_values(path.read_text(encoding="utf-8"))
        if found:
            findings[path.name] = found
    assert not findings
    assert not any(re.search(r"\.parquet$", p.name) for p in (CERT_DIR / "tests_rl").iterdir())
