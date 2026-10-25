"""keep/ hard links survive EasyR1's own checkpoint retention (DESIGN 7.3 retention bug), save cadence,
attempt logs."""

from __future__ import annotations

import json
import os

import pytest


def fake_save(ckpt, step, n_gpus=2, payload=None):
    actor = ckpt / f"global_step_{step}" / "actor"
    (actor / "huggingface").mkdir(parents=True)
    for rank in range(n_gpus):
        (actor / f"model_world_size_{n_gpus}_rank_{rank}.pt").write_bytes(payload or f"step{step}rank{rank}".encode())
        (actor / f"optim_world_size_{n_gpus}_rank_{rank}.pt").write_bytes(b"o")
    (actor / "huggingface" / "config.json").write_text("{}")
    (ckpt / f"global_step_{step}" / "dataloader.pt").write_bytes(b"d")


def tracker(ckpt, step, best):
    (ckpt / "checkpoint_tracker.json").write_text(json.dumps({"last_global_step": step, "best_global_step": best}))


def test_keep_survives_remove_obsolete(trl, specs, rlvr_env, tmp_path):
    """Replays EasyR1's save loop with the vendored remove_obsolete_ckpt (save_limit 2)."""
    paths = trl.Paths(specs["3b"])
    with trl.backend_on_path(paths.backend):
        from verl.utils.checkpoint.checkpoint_manager import remove_obsolete_ckpt

    run = tmp_path / "run"
    ckpt = run / "ckpt"
    ckpt.mkdir(parents=True)
    best = 25  # what ray_trainer records at the first save (val score 0.0 > -1.0)
    inode_100 = None
    for step in range(25, 301, 25):
        remove_obsolete_ckpt(str(ckpt), step, best, 2)
        fake_save(ckpt, step)
        tracker(ckpt, step, best)
        trl.link_keep(run, [100, 200, 300], 2)
        if step == 100:
            inode_100 = os.stat(ckpt / "global_step_100" / "actor" / "model_world_size_2_rank_0.pt").st_ino
    assert not (ckpt / "global_step_100").exists() and not (ckpt / "global_step_200").exists()  # the bug
    for step in (100, 200, 300):
        kept = run / "keep" / f"global_step_{step}" / "actor"
        assert trl.step_complete(run / "keep" / f"global_step_{step}", 2)
        assert (kept / "model_world_size_2_rank_1.pt").read_bytes() == f"step{step}rank1".encode()
    assert os.stat(run / "keep" / "global_step_100" / "actor" / "model_world_size_2_rank_0.pt").st_ino == inode_100
    # with the best step present, save_limit 2 keeps only best + current: every other save is deleted
    assert sorted(p.name for p in ckpt.glob("global_step_*")) == ["global_step_25", "global_step_300"]


def test_keep_waits_for_a_complete_save(trl, tmp_path):
    run = tmp_path / "run"
    ckpt = run / "ckpt"
    ckpt.mkdir(parents=True)
    fake_save(ckpt, 100)
    tracker(ckpt, 75, 25)  # step-100 save still in progress: tracker not yet written
    assert trl.link_keep(run, [100], 2) == []
    tracker(ckpt, 100, 25)
    (ckpt / "global_step_100" / "actor" / "model_world_size_2_rank_1.pt").unlink()  # a shard missing
    assert trl.link_keep(run, [100], 2) == []
    fake_save(ckpt, 125)
    (ckpt / "global_step_100" / "actor" / "model_world_size_2_rank_1.pt").write_bytes(b"x")
    assert trl.link_keep(run, [100], 2) == [100]
    assert trl.link_keep(run, [100], 2) == []  # idempotent
    assert trl.link_keep(run, [100], 4) == []  # wrong world size is never "complete"


@pytest.mark.parametrize(
    "host,step_seconds,expect",
    [("grn067", None, 25), ("grn067", 3000.0, 25), ("notch369", None, 5), ("notch369", 60.0, 25), ("notch369", 200.0, 10),
     ("grn077", 600.0, 4), ("grn077", 4000.0, 1)],
)
def test_save_freq_rule(trl, specs, host, step_seconds, expect):
    value, _ = trl.save_freq_for(specs["3b"], host, step_seconds)
    assert value == expect
    assert 100 % value == 0  # steps 100, 200, 300 are always saved


def test_experiment_records_merge_attempts(trl, tmp_path):
    run = tmp_path / "run"
    (run / "logs").mkdir(parents=True)
    (run / "ckpt").mkdir()
    (run / "logs" / "experiment_log.attempt1.jsonl").write_text(
        "".join(json.dumps({"step": s, "timing_s": {"step": 100.0 + s}, "reward": {"overall": 0.1}}) + "\n" for s in (1, 2, 3)))
    (run / "ckpt" / "experiment_log.jsonl").write_text(
        "".join(json.dumps({"step": s, "timing_s": {"step": 200.0}, "reward": {"overall": 0.5}}) + "\n" for s in (3, 4)))
    records = trl.experiment_records(run)
    assert [r["step"] for r in records] == [1, 2, 3, 4]
    assert records[2]["reward"]["overall"] == 0.5  # the later attempt wins
    assert trl.measured_step_seconds(run) == pytest.approx(151.0)
    summary = trl.step_summary(records[-1])
    assert summary["R"] == 0.5 and summary["step_s"] == 200.0
    assert "step 4/300" in trl.format_step(summary, 300)


def test_experiment_records_merge_val_record_of_a_step(trl, tmp_path):
    # EasyR1 logs a validation pass as a second record with the same step: it must not hide the train metrics
    run = tmp_path / "run"
    (run / "logs").mkdir(parents=True)
    (run / "ckpt").mkdir()
    (run / "logs" / "experiment_log.attempt1.jsonl").write_text(
        json.dumps({"step": 1, "timing_s": {"step": 61.0}, "reward": {"overall": 0.06}}) + "\n"
        + json.dumps({"step": 1, "val": {"reward_score": 0.17}, "val_response_length": {"mean": 200.0}}) + "\n")
    (run / "ckpt" / "experiment_log.jsonl").write_text(
        json.dumps({"step": 2, "timing_s": {"step": 59.0}, "reward": {"overall": 0.05}}) + "\n"
        + json.dumps({"step": 2, "val": {"reward_score": 0.18}}) + "\n")
    records = trl.experiment_records(run)
    assert [r["step"] for r in records] == [1, 2]
    summaries = [trl.step_summary(r) for r in records]
    assert [s["step_s"] for s in summaries] == [61.0, 59.0]
    assert summaries[1]["val"]["reward_score"] == 0.18 and summaries[0]["R"] == 0.06
    assert trl.measured_step_seconds(run) == pytest.approx(60.0)
    assert trl.run_summary(run, 2)["step_seconds_median"] == 60.0


def test_bf16_config_patch(trl):
    config = {"torch_dtype": "float32", "text_config": {"dtype": "float32"}, "vision_config": {"torch_dtype": "bfloat16"}}
    patched = trl.bf16_config(config)
    assert config["torch_dtype"] == "bfloat16" and config["text_config"]["dtype"] == "bfloat16"
    assert len(patched) == 2
    assert trl.bf16_config({"x": 1}) == ["torch_dtype: <absent> -> bfloat16"]
