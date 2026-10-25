"""The reward kwargs each arm's resolved config passes reach compute_score exactly as EasyR1 calls it:
the entry file loaded by path, functools.partial(**kwargs), reward_inputs built like
verl/workers/reward/function.py from real RL rows (val split), details records for the smoke stats."""

from __future__ import annotations

import functools
import importlib.util
import json

import pytest

from rl_helpers import CERT_DIR


@pytest.fixture(scope="module")
def rows_and_store(trl, specs, data_ready):
    import pyarrow.parquet as pq

    from trace_tasks.experimental.certificates.export import make_sft_target
    from trace_tasks.experimental.certificates.reward import ExtensionStore

    paths = data_ready
    data = trl.check_data(specs["3b"], paths, trl.ShaCache(paths.cache / "sha256.json"))
    store = ExtensionStore.load(paths.store, data["store"]["sha256"])
    table = pq.read_table(paths.val_parquet, columns=["prompt_cert", "instance_id", "cert_extension_id", "answer_gt", "task",
                                                      "domain", "scene_id", "query_id", "scene_variant"]).to_pylist()
    picked, seen = [], set()
    for row in table:
        if row["task"] not in seen:
            seen.add(row["task"])
            picked.append(row)
    from trace_tasks.experimental.certificates.reward import compute_score
    from trace_tasks.experimental.certificates.types import CertificateVersionError

    inputs = []
    for row in picked:
        ext = store.get(row["cert_extension_id"])
        response = make_sft_target(None, ext)
        reward_input = {"response": response, "response_length": len(response) // 3, "ground_truth": row.pop("answer_gt")}
        reward_input.update(row)  # every other non-tensor column rides along (function.py:57-72)
        inputs.append(reward_input)
    try:  # the store must be scoreable by the current certificate code (train_rl.py check fails hard on this)
        compute_score([dict(r) for r in inputs], store_path=str(paths.store), store_sha256=data["store"]["sha256"], arm="P")
    except CertificateVersionError as exc:
        if "does not match loaded adapter" in str(exc):
            pytest.skip(f"frozen store predates the current certificate adapters (rebuild the data): {exc}")
        raise
    return data, inputs


def easyr1_reward_fn(config):
    """What verl/workers/reward/function.py does with worker.reward.* of a resolved config."""
    path, name = config["worker"]["reward"]["reward_function"].rsplit(":", 1)
    spec = importlib.util.spec_from_file_location("custom_reward_fn", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert getattr(module, "REWARD_TYPE", "batch") == "batch"
    return functools.partial(getattr(module, name), **config["worker"]["reward"]["reward_function_kwargs"])


def test_every_arm_scores_through_its_config(trl, specs, releases, rows_and_store, tmp_path):
    from trace_tasks.experimental.certificates.reward import resolve_arm

    data, inputs = rows_and_store
    spec, release = specs["3b"], releases["3b"]
    paths = trl.Paths(spec)
    results = {}
    for arm in trl.ARMS:
        config = trl.resolve_config(spec, release, arm=arm, seed=1, n_gpus=2, model_path="/sft", run_dir=str(tmp_path / arm),
                                    data_root=str(paths.data), store_sha256=data["store"]["sha256"])
        scores = easyr1_reward_fn(config)([dict(r) for r in inputs])
        assert len(scores) == len(inputs)
        for score in scores:
            assert set(score) >= {"overall", "accuracy", "answer_reward", "grounding", "semantic", "cert_format"}
            assert all(0.0 <= v <= 1.0 for k, v in score.items() if k != "verifier_ms")
        results[arm] = scores
        # details records (val phase: e1_dev extensions) carry every arm's overall for the same responses
        kwargs = config["worker"]["reward"]["reward_function_kwargs"]
        details = trl.details_summary(tmp_path / arm, "val")
        assert details["rows"] == len(inputs)
        arm_key = resolve_arm(kwargs["arm"], kwargs["gated"])
        assert arm_key == arm
        mean_overall = sum(s["overall"] for s in scores) / len(scores)
        assert details["by_adapter"]["all"][f"overall[{arm_key}]"] == pytest.approx(mean_overall, abs=1e-4)
    # the SFT target (oracle certificate, rounded boxes): A = M = F = 1 in every arm; arms differ only in overall
    for arm, scores in results.items():
        for score in scores:
            assert score["accuracy"] == 1.0 and score["semantic"] == 1.0 and score["cert_format"] == 1.0
        assert [s["grounding"] for s in scores] == [s["grounding"] for s in results["A"]]
    assert all(s["overall"] == pytest.approx(1.0) for s in results["A"])


def test_wrong_store_sha_raises(trl, specs, releases, rows_and_store, tmp_path):
    data, inputs = rows_and_store
    spec = specs["3b"]
    paths = trl.Paths(spec)
    config = trl.resolve_config(spec, releases["3b"], arm="P", seed=1, n_gpus=2, model_path="/sft", run_dir=str(tmp_path),
                                data_root=str(paths.data), store_sha256="f" * 64)
    with pytest.raises(Exception, match="sha256"):
        easyr1_reward_fn(config)([dict(inputs[0])])


def test_reward_entry_is_the_package_function(trl):
    spec = importlib.util.spec_from_file_location("custom_reward_fn", CERT_DIR / "reward_entry.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from trace_tasks.experimental.certificates.reward import compute_score

    assert module.compute_score is compute_score
    assert json.dumps(module.REWARD_NAME) == '"trace_certificate_v1"'
