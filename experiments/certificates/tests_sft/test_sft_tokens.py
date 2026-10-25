"""SFT rows tokenize exactly like EasyR1's RLHFDataset rows (DESIGN 7.2, leakage test 4).

The central test builds, for the same RL-split instance, one SFT row with the library's own
``export_sft`` (into a scratch root; the frozen data is only read) and takes the frozen EasyR1
train-parquet row of that instance through EasyR1's ``RLHFDataset``, and compares the prompt token
ids, the rollout prompt ids, the actor's pixel values and grid, the label mask, the chat-template
rendering and the M-RoPE positions. The same comparison runs for the E1-dev rows the stop-rule
generation reads. The SFT user text is checked byte-identical to the eval public prompt (every
SFT row against its instance's public prompt hash; E1-dev against the public parquet itself).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

import sft_helpers as h

ADAPTERS = ("selected_set", "filtered_mean", "weighted_sum", "path")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source_build(data_root: Path) -> Path:
    """The frozen build dir (``private/build`` in the current export layout, ``build`` in older roots)."""
    from trace_tasks.experimental.certificates import export

    for cand in ([export.build_root(data_root)] if hasattr(export, "build_root") else []) + [data_root / "build"]:
        if (cand / "rl").is_dir():
            return cand
    pytest.skip(f"no build dir with the rl split under {data_root}")


@pytest.fixture(scope="module")
def rl_case(data_root, tmp_path_factory):
    """Two RL-split instances per adapter: their SFT rows from the library's own ``export_sft``
    (scratch root holding a subset of the build; the frozen files are only read) and their frozen
    EasyR1 train-parquet rows (``rl/rl/rl.parquet``, copied byte for byte into a small parquet)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from trace_tasks.experimental.certificates import export

    src = _source_build(data_root)
    out = tmp_path_factory.mktemp("export_sft_rl")
    (out / "receipts").mkdir()
    (out / "private").mkdir()
    os.symlink(data_root / "receipts" / "rl.json", out / "receipts" / "rl.json")
    os.symlink(data_root / "private" / "rl.extensions_v1.jsonl.zst", out / "private" / "rl.extensions_v1.jsonl.zst")
    dst = export.build_root(out) if hasattr(export, "build_root") else out / "build"
    picked = []
    for kind in ADAPTERS:
        lines = (src / "rl" / kind / "accepted.jsonl").read_text().splitlines()[:2]
        (dst / "rl" / kind).mkdir(parents=True)
        (dst / "rl" / kind / "accepted.jsonl").write_text("".join(ln + "\n" for ln in lines))
        os.symlink(src / "rl" / kind / "images", dst / "rl" / kind / "images")
        picked += [json.loads(ln)["instance_id"] for ln in lines]
    receipt = export.export_sft("rl", out)
    sft_rows = {r["instance_id"]: r for r in h.read_jsonl(out / "sft" / "rl.jsonl")}
    assert receipt["rows"] == len(sft_rows) == 8 and set(sft_rows) == set(picked)
    pf = pq.ParquetFile(data_root / "rl" / "rl" / "rl.parquet")
    want, tables, offset = set(picked), [], 0
    for rg in range(pf.metadata.num_row_groups):
        ids = pf.read_row_group(rg, columns=["instance_id"]).column("instance_id").to_pylist()
        hit = [i for i, iid in enumerate(ids) if iid in want]
        if hit:
            t = pf.read_row_group(rg)
            tables.append(t.take(pa.array(hit)))
        offset += len(ids)
    rl_table = pa.concat_tables(tables)
    assert sorted(rl_table.column("instance_id").to_pylist()) == sorted(picked)
    rl_path = out / "rl_subset.parquet"
    pq.write_table(rl_table, rl_path)
    return out, sft_rows, rl_path


def _rl_dataset(easyr1, path: Path, cfg3b):
    from trace_tasks.experimental.certificates import prompts

    ds = easyr1["dataset"].RLHFDataset(
        str(path), easyr1["tokenizer"], easyr1["processor"], prompt_key="prompt_cert", answer_key="answer_gt", max_prompt_length=3072,
        truncation="right", system_prompt_file=str(prompts.system_prompt_path()), min_pixels=cfg3b["data"]["min_pixels"],
        max_pixels=cfg3b["data"]["max_pixels"], filter_overlong_prompts=False,
    )
    return ds, {iid: i for i, iid in enumerate(ds.dataset["instance_id"])}


@pytest.fixture(scope="module")
def public_dev(data_root):
    import pyarrow.parquet as pq

    return {r["instance_id"]: r for r in pq.read_table(data_root / "public" / "e1_dev.cert_px.parquet",
                                                         columns=["instance_id", "prompt_answer", "task"]).to_pylist()}


def _meta_rope_model(base_dir):
    from accelerate import init_empty_weights
    from transformers import AutoConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel

    with init_empty_weights():
        return Qwen2_5_VLModel._from_config(AutoConfig.from_pretrained(str(base_dir)))


def _assert_same_tokens(sft, encoder, processor, easyr1, rope, system, user, target, image, rl, where):
    """One SFT example (sft.py's Encoder) against one RLHFDataset item of the same instance."""
    ex = sft.Encoder(processor, system, encoder.min_pixels, encoder.max_pixels).encode_example(user, image, target)
    mask = rl["attention_mask"].bool()
    rl_ids = rl["input_ids"][mask].tolist()
    n = ex["n_prompt"]
    assert n == len(rl_ids) and len(rl_ids) < 3072, where
    assert ex["input_ids"][:n].tolist() == rl_ids, f"{where}: prompt ids differ from RLHFDataset"
    # what EasyR1 hands to vLLM for the rollout, and what the actor feeds the ViT
    assert encoder.prompt_token_ids(user) == list(rl["raw_prompt_ids"]), where
    rl_img = easyr1["dataset"].process_image(rl["multi_modal_data"]["images"][0], encoder.min_pixels, encoder.max_pixels)
    actor = dict(easyr1["processor"].image_processor(images=[rl_img], return_tensors="pt"))
    assert torch.equal(actor["pixel_values"], ex["pixel_values"]), where
    assert torch.equal(actor["image_grid_thw"], ex["image_grid_thw"]), where
    # assistant-only loss: every prompt position (system, user text, image) is masked; target + <|im_end|> is not
    labels = ex["labels"]
    assert (labels[:n] == sft.IGNORE_INDEX).all()
    assert torch.equal(labels[n:], ex["input_ids"][n:])
    assert labels[-1].item() == encoder.eos_id
    assert encoder.tokenizer.decode(labels[n:-1].tolist()) == target
    # the whole sequence is the chat template's rendering of the conversation (minus the newline
    # after <|im_end|>, which generation never produces)
    full_text = processor.apply_chat_template(
        [{"role": "system", "content": system}, *encoder.messages(user)[1:], {"role": "assistant", "content": target}], tokenize=False)
    full = processor(images=[image], text=[full_text], add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
    mine = ex["input_ids"].tolist()
    assert full[: len(mine)] == mine and encoder.tokenizer.decode(full[len(mine):]) == "\n", where
    # M-RoPE: the training forward's positions (HF get_rope_index) equal EasyR1's for the prompt and
    # continue by +1 per target token, as EasyR1 extends them for the response
    pos, _ = rope.get_rope_index(ex["input_ids"][None], ex["image_grid_thw"], None, attention_mask=torch.ones(1, len(mine), dtype=torch.long))
    rl_pos = rl["position_ids"][:, mask]
    assert torch.equal(pos[:, 0, :n], rl_pos[1:]), where
    assert torch.equal(rl_pos[0], torch.arange(n)), where
    assert torch.equal(pos[:, 0, n:], pos[:, 0, n - 1 : n] + torch.arange(1, len(mine) - n + 1)), where


def test_sft_row_and_rl_row_of_one_instance_have_identical_prompt_tokens(sft, encoder, processor, easyr1, rl_case, base_dir, cfg3b):
    """For 2 RL-split instances per adapter: the SFT row written by export_sft and the frozen RL
    train-parquet row of the same instance give identical prompt token ids through sft.py and
    through EasyR1's RLHFDataset (plus rollout ids, actor pixels, labels, template, M-RoPE)."""
    out, sft_rows, rl_path = rl_case
    ds, rl_index = _rl_dataset(easyr1, rl_path, cfg3b)
    rope = _meta_rope_model(base_dir)
    kinds = set()
    for iid, row in sft_rows.items():
        system, user, assistant = (m["content"] for m in row["messages"])
        rl_row = ds.dataset[rl_index[iid]]
        # the SFT user text is the RL prompt (= the cert_px public prompt), byte for byte
        assert user.encode("utf-8") == rl_row["prompt_cert"].encode("utf-8")
        assert row["prompt_sha256"] == _sha(user) and system == ds.system_prompt and assistant == row["target"]
        image = sft.process_image(out / row["images"][0], encoder.min_pixels, encoder.max_pixels)
        _assert_same_tokens(sft, encoder, processor, easyr1, rope, system, user, row["target"], image, ds[rl_index[iid]], f"rl {iid}")
        kinds.add(row["adapter_id"].split(".", 1)[0])
    assert kinds == set(ADAPTERS)


def test_dev_generation_inputs_match_easyr1_val_rows(sft, encoder, processor, easyr1, public_dev, dev_store, data_root, base_dir, cfg3b):
    """The stop-rule generation (sft.py gen) reads the public E1-dev prompt; with the instance's
    SFT target appended it tokenizes exactly like EasyR1's val row (rl/e1_dev/e1_dev.parquet)."""
    from trace_tasks.experimental.certificates.export import make_sft_target
    from trace_tasks.experimental.certificates.prompts import system_prompt

    ds, rl_index = _rl_dataset(easyr1, data_root / "rl" / "e1_dev" / "e1_dev.parquet", cfg3b)
    rope = _meta_rope_model(base_dir)
    rows = sft.dev_rows(data_root / "public" / "e1_dev.cert_px.parquet", 2)
    assert len(rows) == 8
    for r in rows:
        assert r["user"] == public_dev[r["instance_id"]]["prompt_answer"] == ds.dataset[rl_index[r["instance_id"]]]["prompt_cert"]
        target = make_sft_target({"instance_id": r["instance_id"]}, dev_store[r["instance_id"]])
        image = sft.process_image(r["image"], encoder.min_pixels, encoder.max_pixels)
        _assert_same_tokens(sft, encoder, processor, easyr1, rope, system_prompt(), r["user"], target, image, ds[rl_index[r["instance_id"]]], f"e1_dev {r['instance_id']}")


def test_sft_user_text_is_the_eval_public_prompt_for_every_row(sft, cfg3b, data_root, public_dev, dev_store):
    from trace_tasks.experimental.certificates.export import RL_FRAME

    # the extension field public.prompt_sha256.px_source is the hash of the eval public prompt ...
    assert len(public_dev) == len(dev_store) == 512
    for iid, row in public_dev.items():
        assert _sha(row["prompt_answer"]) == dev_store[iid].public["prompt_sha256"][RL_FRAME]
    # ... every one of the 6000 SFT user texts has its instance's hash (plus system and target checks) ...
    records = sft.load_sft_records(data_root, cfg3b, smoke=False)
    assert len(records) == 6000
    report = sft.check_records(cfg3b, data_root, records)
    assert report["rows_checked"] == 6000
    # ... and is byte-identical to the cert_px prompt the exporter rendered for that instance
    src = _source_build(data_root)
    rendered = {}
    for kind in ADAPTERS:
        for line in (src / "sft" / kind / "accepted.jsonl").read_text().splitlines():
            row = json.loads(line)
            rendered[row["instance_id"]] = row["prompts"]["cert_px"]
    assert all(rec["user"] == rendered[rec["instance_id"]] for rec in records)
    # for E1-dev the exporter's cert_px prompt is exactly the public parquet prompt
    for kind in ADAPTERS:
        for line in (src / "e1_dev" / kind / "accepted.jsonl").read_text().splitlines():
            row = json.loads(line)
            assert row["prompts"]["cert_px"] == public_dev[row["instance_id"]]["prompt_answer"]


def test_training_rows_through_easyr1(sft, encoder, base_dir, rlvr_root, cfg3b, data_root):
    """The equivalence check that jobs/sft.sh runs before training, on 2 SFT rows per adapter."""
    from trace_tasks.experimental.certificates import prompts

    records = sft.load_sft_records(data_root, cfg3b, smoke=True)
    picks = []
    for kind in ADAPTERS:
        picks += [dict(r, image=data_root / r["image"]) for r in records if r["adapter_id"].startswith(kind + ".")][:2]
    res = sft.rl_equivalence(encoder, base_dir, rlvr_root, picks, prompts.system_prompt_path(), cfg3b["data"]["min_pixels"], cfg3b["data"]["max_pixels"])
    assert res["all_equal"], res
    assert len(res["rows"]) == 8


def test_exact_length_formula(sft, encoder, cfg3b, data_root):
    records = sft.load_sft_records(data_root, cfg3b, smoke=True)
    picks = records[::37][:8]
    predicted = encoder.prompt_lengths([r["user"] for r in picks], [r["image_size"] for r in picks])
    for rec, n in zip(picks, predicted):
        img = sft.process_image(data_root / rec["image"], encoder.min_pixels, encoder.max_pixels)
        assert int(encoder.encode_prompt(rec["user"], img)["input_ids"].numel()) == n
    assert encoder.target_lengths([r["target"] for r in picks]) == [len(encoder.target_ids(r["target"])) for r in picks]


def test_targets_are_compact_json_with_evidence_first(sft, cfg3b, data_root):
    for rec in sft.load_sft_records(data_root, cfg3b, smoke=False):
        obj = json.loads(rec["target"])
        assert list(obj) == ["annotation", "certificate", "answer"]
        assert next(iter(obj["certificate"])) == "kind"
        assert json.dumps(obj, separators=(",", ":"), ensure_ascii=False) == rec["target"]


def test_collate_pads_right_and_masks(sft, encoder, cfg3b, data_root):
    records = sft.load_sft_records(data_root, cfg3b, smoke=True)[:2]
    exs = []
    for i, rec in enumerate(records):
        img = sft.process_image(data_root / rec["image"], encoder.min_pixels, encoder.max_pixels)
        ex = encoder.encode_example(rec["user"], img, rec["target"])
        ex["index"] = i
        exs.append(ex)
    batch = sft.collate(exs, encoder.pad_id)
    L = [int(e["input_ids"].numel()) for e in exs]
    assert batch["input_ids"].shape == (2, max(L))
    for i, e in enumerate(exs):
        assert torch.equal(batch["input_ids"][i, : L[i]], e["input_ids"])
        assert (batch["input_ids"][i, L[i]:] == encoder.pad_id).all()
        assert (batch["labels"][i, L[i]:] == sft.IGNORE_INDEX).all()
        assert batch["attention_mask"][i].sum().item() == L[i]
    assert torch.equal(batch["pixel_values"], torch.cat([e["pixel_values"] for e in exs]))
    assert batch["image_grid_thw"].shape == (2, 3) and batch["index"] == [0, 1]


def test_label_position_loss_equals_the_hf_causal_lm_loss(sft, encoder, tiny_model, cfg3b):
    """The root module runs lm_head on label positions only; its summed CE over the batch equals
    HF's own (token-mean) causal-LM loss times the number of label tokens. Short text-only rows
    keep the full-vocabulary reference logits small; the image path is exercised by the
    training tests."""
    from transformers import AutoModelForImageTextToText

    torch.manual_seed(0)
    hf = AutoModelForImageTextToText.from_pretrained(str(tiny_model), torch_dtype=torch.float32, attn_implementation="sdpa")
    hf.eval()
    exs = []
    for i, (n_prompt, target) in enumerate([(40, '{"annotation":[],"answer":0}'), (25, '{"annotation":[[1,2,3,4]],"answer":12}')]):
        prompt = torch.randint(0, 150000, (n_prompt,))
        tgt = torch.tensor(encoder.target_ids(target))
        exs.append({"input_ids": torch.cat([prompt, tgt]), "labels": torch.cat([torch.full((n_prompt,), sft.IGNORE_INDEX), tgt]),
                    "pixel_values": torch.zeros(0, 1176), "image_grid_thw": torch.zeros(0, 3, dtype=torch.long), "index": i})
    batch = sft.collate(exs, encoder.pad_id)
    batch.pop("index")
    batch["pixel_values"] = batch["image_grid_thw"] = None
    module = sft._sft_module_class()(hf)
    with torch.no_grad():
        loss_sum, correct = module(**batch)
        ref = hf(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"], use_cache=False).loss
    n_tokens = int((batch["labels"][:, 1:] != sft.IGNORE_INDEX).sum())
    assert n_tokens == sum(int((e["labels"] != sft.IGNORE_INDEX).sum()) for e in exs)
    assert torch.allclose(loss_sum / n_tokens, ref, rtol=1e-5, atol=1e-6)
    assert 0 <= int(correct) <= n_tokens
    # chunked (recomputed) CE gives the same loss and the same gradients as one chunk
    grads = []
    for chunk in (10**9, 3):
        hf.zero_grad(set_to_none=True)
        loss, corr = sft._sft_module_class()(hf, ce_chunk=chunk)(**batch)
        loss.backward()
        grads.append((float(loss), int(corr), hf.lm_head.weight.grad.clone(), hf.model.language_model.layers[0].mlp.up_proj.weight.grad.clone()))
    (l1, c1, g1, u1), (l2, c2, g2, u2) = grads
    assert l1 == pytest.approx(l2, rel=1e-6) and c1 == c2 == int(correct)
    for a, b in ((g1, g2), (u1, u2)):  # equal up to float summation order
        assert float((a - b).norm() / a.norm()) < 1e-5
