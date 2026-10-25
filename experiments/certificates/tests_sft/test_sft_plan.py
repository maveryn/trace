"""Step plan, batch layout, schedule and data order of sft.py (pure; no model, no data needed
except where a fixture says so)."""

from __future__ import annotations

import math

import pytest


def test_design_plan_3b(sft, cfg3b):
    plan = sft.make_plan(cfg3b, 6000, smoke=False, world_size=2)
    assert plan["steps_per_epoch"] == 188 and plan["total_steps"] == 564  # DESIGN 7.2: about 188 steps/epoch, 3 epochs
    assert plan["save_steps"] == [94, 188, 282, 376, 470, 564]           # every half epoch: 6 checkpoints
    assert plan["save_epochs"] == [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    assert plan["warmup_steps"] == math.ceil(0.03 * 564) == 17
    assert plan["lr"] == 1e-5 and plan["seed"] == 1 and plan["global_batch"] == 32
    assert plan["layout"] == {"world_size": 2, "per_rank": 16, "per_device_batch": 4, "grad_accum": 4}
    assert plan["select"] == {"min_F": 0.95, "max_abs_dA": 0.015, "max_abs_dM": 0.02, "fallback": "last"}
    assert plan["eval"]["temperature"] == 0.0 and plan["eval"]["max_new_tokens"] == 2048
    assert plan["dev_rows_per_task"] is None


def test_design_config_7b_differs_only_where_designed(sft, cfg3b, cfg7b):
    assert cfg7b["train"]["lr"] == 5e-6 and cfg3b["train"]["lr"] == 1e-5
    assert cfg7b["model"]["repo_id"] == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert cfg7b["model"]["revision"] == "cc594898137f460bfe9f0759e9844b3ce807cfb5"
    assert cfg3b["model"]["revision"] == "66285546d2b821cf421d4f5eb2576359d3770cd3"
    for block in ("data", "eval", "select", "smoke"):
        assert cfg7b[block] == cfg3b[block], block
    same_train = {k: v for k, v in cfg3b["train"].items() if k not in ("lr", "per_device_batch")}
    assert {k: v for k, v in cfg7b["train"].items() if k not in ("lr", "per_device_batch")} == same_train
    t = cfg3b["train"]
    assert (t["scheduler"], t["warmup_ratio"], t["weight_decay"], t["global_batch"], t["epochs"], t["seed"], t["save"]) == (
        "cosine", 0.03, 0.0, 32, 3, 1, "half_epoch")
    assert cfg3b["data"]["max_length"] == 4096 and (cfg3b["data"]["min_pixels"], cfg3b["data"]["max_pixels"]) == (262144, 4194304)
    assert cfg3b["model"]["attn_implementation"] == "flash_attention_2" and cfg3b["model"]["trainable"] == "llm_and_merger"
    assert cfg3b["fsdp"]["param_dtype"] == "bf16"


def test_smoke_plan(sft, cfg3b):
    plan = sft.make_plan(cfg3b, 256, smoke=True, world_size=4)
    assert plan["total_steps"] == 20 and plan["save_steps"] == [20] and plan["steps_per_epoch"] == 8
    assert plan["dev_rows_per_task"] == 8 and plan["smoke"] is True
    assert plan["math_sha256"] != sft.make_plan(cfg3b, 256, smoke=False)["math_sha256"]


def test_smoke_subset_is_64_rows_per_adapter_in_file_order(sft, cfg3b, data_root):
    full = sft.load_sft_records(data_root, cfg3b, smoke=False)
    smoke = sft.load_sft_records(data_root, cfg3b, smoke=True)
    assert len(smoke) == 256
    for kind in ("selected_set", "filtered_mean", "weighted_sum", "path"):
        mine = [r["line"] for r in smoke if r["adapter_id"].startswith(kind + ".")]
        first = [r["line"] for r in full if r["adapter_id"].startswith(kind + ".")][:64]
        assert mine == first


@pytest.mark.parametrize("spe,total,want", [(188, 564, [94, 188, 282, 376, 470, 564]), (187, 561, [94, 187, 281, 374, 468, 561]),
                                            (8, 20, [4, 8, 12, 16, 20]), (8, 21, [4, 8, 12, 16, 20, 21])])
def test_half_epoch_saves(sft, spe, total, want):
    assert sft.save_steps("half_epoch", spe, total) == want
    assert sft.save_steps("final", spe, total) == [total]


def test_batch_layout(sft):
    assert sft.batch_layout(32, 1, 4) == {"world_size": 1, "per_rank": 32, "per_device_batch": 4, "grad_accum": 8}
    assert sft.batch_layout(32, 4, 4)["grad_accum"] == 2
    assert sft.batch_layout(32, 8, 4)["grad_accum"] == 1
    assert sft.batch_layout(32, 8, 16) == {"world_size": 8, "per_rank": 4, "per_device_batch": 4, "grad_accum": 1}
    assert sft.batch_layout(32, 2, 3)["per_device_batch"] == 2  # largest divisor of 16 that is <= 3
    for bad in (3, 5, 6, 64):
        with pytest.raises(sft.ConfigError):
            sft.batch_layout(32, bad, 4)


def test_lr_schedule_is_the_transformers_cosine_with_warmup(sft):
    import torch
    from transformers import get_cosine_schedule_with_warmup

    total, warmup = 564, 17
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=total)
    for step in range(total):
        assert sft.lr_factor(step, warmup, total) == pytest.approx(opt.param_groups[0]["lr"], abs=1e-12)
        opt.step()
        sched.step()
    assert sft.lr_factor(warmup, warmup, total) == 1.0 and sft.lr_factor(total - 1, warmup, total) > 0


@pytest.mark.parametrize("n", [6000, 256, 100])
def test_data_order_is_independent_of_world_size(sft, n):
    """Step j of an epoch holds positions [32j, 32j+32) of the epoch permutation for every N
    (N divides n here, so DistributedSampler adds no padding); every row once per epoch."""
    import torch

    gb, seed = 32, 1
    for epoch in (0, 1):
        g = torch.Generator()
        g.manual_seed(seed + epoch)
        perm = torch.randperm(n, generator=g).tolist()  # what DistributedSampler shuffles with
        ref = None
        for world in (1, 2, 4):
            steps = sft.epoch_steps(n, world, seed, epoch, gb)
            assert len(steps) == math.ceil(n / gb)
            members = [sorted(i for share in step for i in share) for step in steps]
            assert members == [sorted(perm[j * gb : (j + 1) * gb]) for j in range(len(steps))]
            assert sorted(i for m in members for i in m) == list(range(n))
            ref = ref or members
            assert members == ref
            for j in (0, len(steps) - 1):
                assert sorted(sft.step_members(n, world, seed, epoch, gb, j)) == members[j]
    assert sft.epoch_steps(n, 2, seed, 0, gb) != sft.epoch_steps(n, 2, seed, 1, gb)


def test_rank_chunks_are_torch_distributed_sampler(sft):
    from torch.utils.data import DistributedSampler

    s = DistributedSampler(range(6000), num_replicas=2, rank=1, shuffle=True, seed=1, drop_last=False)
    s.set_epoch(2)
    flat = [i for chunk in sft.epoch_rank_chunks(6000, 2, 1, 1, 2, 32) for i in chunk]
    assert flat == list(s)


def test_micro_batches_partition_a_step(sft):
    lengths = [10, 50, 20, 40, 30, 60, 5, 70]
    mbs = sft.micro_batches([0, 1, 2, 3, 4, 5, 6, 7], 3, lengths)
    assert sorted(i for mb in mbs for i in mb) == list(range(8)) and [len(mb) for mb in mbs] == [3, 3, 2]
    assert mbs[0] == [7, 5, 1]  # longest first: similar lengths share a micro-batch


def test_math_sha_ignores_free_knobs_only(sft, cfg3b):
    import copy

    base = sft.math_sha256(cfg3b, False)
    c = copy.deepcopy(cfg3b)
    c["train"]["per_device_batch"] = 1
    c["train"]["dataloader_workers"] = 0
    c["fsdp"]["sharding_strategy"] = "SHARD_GRAD_OP"
    assert sft.math_sha256(c, False) == base
    for block, key, value in (("train", "lr", 2e-5), ("train", "seed", 2), ("data", "max_length", 2048), ("model", "revision", "x")):
        c2 = copy.deepcopy(cfg3b)
        c2[block][key] = value
        assert sft.math_sha256(c2, False) != base, key


def test_checkpoint_key_mapping(sft):
    assert sft.checkpoint_key("hf.model.language_model.layers.0.mlp.up_proj.weight") == "model.layers.0.mlp.up_proj.weight"
    assert sft.checkpoint_key("hf.model.language_model.embed_tokens.weight") == "model.embed_tokens.weight"
    assert sft.checkpoint_key("hf.model.visual.merger.mlp.0.weight") == "visual.merger.mlp.0.weight"
    assert sft.checkpoint_key("hf.lm_head.weight") == "lm_head.weight"


def test_processed_size_matches_easyr1_process_image(sft):
    from PIL import Image

    for w, h in ((737, 745), (300, 200), (4000, 3000), (512, 512), (1280, 640)):
        img = sft.process_image(Image.new("RGB", (w, h)), 262144, 4194304)
        assert (img.width, img.height) == sft.processed_size(w, h, 262144, 4194304)
