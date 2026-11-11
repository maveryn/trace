"""Merge construction tests (handoff 7.4) on tiny synthetic Qwen2.5-VL-shaped checkpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trace_merge import compatibility, merge  # noqa: E402
from trace_merge.safetensors_io import StreamingShardWriter, index_checkpoint, read_rows, tensor_bytes  # noqa: E402

NAMES = {
    "model.embed_tokens.weight": (16, 4),
    "model.layers.0.mlp.up_proj.weight": (6, 4),
    "model.layers.0.input_layernorm.weight": (4,),
    "model.norm.weight": (4,),
    "lm_head.weight": (16, 4),
    "visual.blocks.0.attn.qkv.weight": (12, 4),
    "visual.merger.mlp.0.weight": (4, 4),
    "visual.patch_embed.proj.weight": (4, 3, 2, 2, 2),
}
CONFIG = {
    "architectures": ["Qwen2_5_VLForConditionalGeneration"], "model_type": "qwen2_5_vl", "hidden_size": 4,
    "intermediate_size": 6, "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
    "vocab_size": 16, "rms_norm_eps": 1e-6, "rope_theta": 1e6, "tie_word_embeddings": False,
    "image_token_id": 13, "video_token_id": 14, "vision_start_token_id": 11, "vision_end_token_id": 12,
    "vision_config": {"depth": 1, "hidden_size": 4, "out_hidden_size": 4, "patch_size": 2, "spatial_merge_size": 2,
                      "temporal_patch_size": 2, "window_size": 4, "fullatt_block_indexes": [0]},
    "rope_scaling": {"type": "mrope", "mrope_section": [1, 1, 0]},
}


def _tokenizer(path: Path, extra: str | None = None) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders

    vocab = {c: i for i, c in enumerate("abcdefghij")}
    if extra:
        vocab[extra] = len(vocab)
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(["<|im_end|>", "<|endoftext|>"])
    tok.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|im_end|>", "pad_token": "<|endoftext|>"}))


def make_ckpt(path: Path, seed: int, *, names=None, config=None, shards: int = 2, extra_token: str | None = None,
              int_buffer: torch.Tensor | None = None) -> Path:
    path.mkdir(parents=True)
    g = torch.Generator().manual_seed(seed)
    names = dict(NAMES if names is None else names)
    tensors = {n: torch.randn(*shape, generator=g).to(torch.bfloat16) for n, shape in names.items()}
    for t in tensors.values():  # a signed zero at a seed-dependent position (another parent is nonzero there)
        t.view(-1)[seed % t.numel()] = -0.0
    if int_buffer is not None:
        tensors["visual.blocks.0.attn.counter"] = int_buffer
    keys = sorted(tensors)
    weight_map = {}
    for s in range(shards):
        part = keys[s::shards]
        fname = f"model-{s + 1:05d}-of-{shards:05d}.safetensors"
        save_file({k: tensors[k] for k in part}, str(path / fname), metadata={"format": "pt"})
        weight_map.update({k: fname for k in part})
    (path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    (path / "config.json").write_text(json.dumps(config or CONFIG))
    (path / "preprocessor_config.json").write_text(json.dumps({"min_pixels": 3136, "max_pixels": 12845056,
                                                               "patch_size": 14, "merge_size": 2,
                                                               "temporal_patch_size": 2,
                                                               "image_processor_type": "Qwen2VLImageProcessor"}))
    _tokenizer(path, extra_token)
    return path


def load_all(path: Path) -> dict[str, torch.Tensor]:
    out = {}
    for f in sorted(path.glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as h:
            for k in h.keys():
                out[k] = h.get_tensor(k)
    return out


def build(tmp: Path, coeffs: dict[str, float], parents: dict[str, Path], mid: str = "X", **kw):
    revs = {r: "0" * 40 for r in parents}
    return merge.build_merge(mid, coeffs, parents, tmp / "out", layout_role=kw.pop("layout_role", "B"),
                             source_revisions=revs, log=lambda *_: None, max_slice_elements=kw.pop("max_slice", 7),
                             **kw)


@pytest.fixture()
def parents(tmp_path):
    return {r: make_ckpt(tmp_path / r, s) for r, s in (("B", 1), ("T", 2), ("V", 3))}


def test_streaming_writer_matches_safetensors(tmp_path):
    t = {"a": torch.randn(5, 3).to(torch.bfloat16), "b": torch.arange(6, dtype=torch.int64).reshape(2, 3)}
    w = StreamingShardWriter(tmp_path / "x.safetensors", [("a", "BF16", (5, 3)), ("b", "I64", (2, 3))], {"format": "pt"})
    assert w.order == ["b", "a"]  # wider dtypes first, so every offset is aligned to its dtype
    for n in w.order:
        w.begin(n)
        w.write(tensor_bytes(t[n]))
        w.end()
    w.close()
    got = load_all(tmp_path)
    assert torch.equal(got["a"].view(torch.int16), t["a"].view(torch.int16)) and torch.equal(got["b"], t["b"])


def test_row_slice_reads(parents):
    idx = index_checkpoint(parents["B"])
    full = load_all(parents["B"])
    info = idx["model.embed_tokens.weight"]
    assert torch.equal(read_rows(info, 3, 9).view(torch.int16), full[info.name][3:9].view(torch.int16))


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_endpoints_recover_parents_bitwise(tmp_path, parents, alpha):
    man = build(tmp_path, {"T": alpha, "B": 1.0 - alpha}, parents)
    got = load_all(tmp_path / "out" / "X")
    want = load_all(parents["T"] if alpha == 1.0 else parents["B"])
    assert set(got) == set(want)
    for k in want:
        assert torch.equal(got[k].view(torch.int16), want[k].view(torch.int16)), k
    assert man["no_training"] is True and man["tensor_count"] == len(NAMES)


def test_swap_parents_with_complement_is_identical(tmp_path, parents):
    build(tmp_path, {"T": 0.5, "V": 0.5}, parents, mid="A")
    build(tmp_path, {"V": 0.5, "T": 0.5}, parents, mid="B2")
    a, b = load_all(tmp_path / "out" / "A"), load_all(tmp_path / "out" / "B2")
    assert all(torch.equal(a[k].view(torch.int16), b[k].view(torch.int16)) for k in a)
    # alpha vs 1 - alpha with roles swapped
    build(tmp_path, {"T": 0.25, "V": 0.75}, parents, mid="C")
    build(tmp_path, {"V": 0.75, "T": 0.25}, parents, mid="D")
    c, d = load_all(tmp_path / "out" / "C"), load_all(tmp_path / "out" / "D")
    assert all(torch.equal(c[k].view(torch.int16), d[k].view(torch.int16)) for k in c)


def bf16_rne_reference(x: torch.Tensor) -> torch.Tensor:
    """Exact float64 values rounded to bf16 (8 significant bits, ties to even) without torch's cast."""

    import numpy as np

    a = x.double().numpy()
    m, e = np.frexp(a)
    return torch.from_numpy(np.ldexp(np.rint(m * 256.0), e - 8)).to(torch.bfloat16)


def test_hand_computed_blend(tmp_path, parents):
    build(tmp_path, {"T": 0.5, "V": 0.5}, parents)
    got = load_all(tmp_path / "out" / "X")
    t, v = load_all(parents["T"]), load_all(parents["V"])
    for k in got:
        want = bf16_rne_reference(0.5 * t[k].double() + 0.5 * v[k].double())  # exact in float64, rounded once
        assert torch.equal(got[k].view(torch.int16), want.view(torch.int16)), k
        assert torch.isfinite(got[k].float()).all()


def test_output_keeps_layout_parent_shards(tmp_path, parents):
    build(tmp_path, {"T": 0.5, "V": 0.5}, parents)
    out = json.loads((tmp_path / "out" / "X" / "model.safetensors.index.json").read_text())["weight_map"]
    ref = json.loads((parents["B"] / "model.safetensors.index.json").read_text())["weight_map"]
    assert out == ref


def test_mismatched_keys_fail_closed(tmp_path, parents):
    names = dict(NAMES)
    names.pop("model.norm.weight")
    bad = make_ckpt(tmp_path / "bad", 9, names=names)
    with pytest.raises(compatibility.IncompatibleParents):
        build(tmp_path, {"T": 0.5, "V": 0.5}, {**parents, "V": bad})


def test_mismatched_shapes_fail_closed(tmp_path, parents):
    names = dict(NAMES)
    names["model.layers.0.mlp.up_proj.weight"] = (7, 4)
    bad = make_ckpt(tmp_path / "bad", 9, names=names)
    with pytest.raises(compatibility.IncompatibleParents):
        build(tmp_path, {"T": 0.5, "V": 0.5}, {**parents, "V": bad})


def test_mismatched_token_map_fails_closed(tmp_path, parents):
    bad = make_ckpt(tmp_path / "bad", 9, extra_token="zz")
    with pytest.raises(compatibility.IncompatibleParents):
        build(tmp_path, {"T": 0.5, "V": 0.5}, {**parents, "V": bad})


def test_architecture_mismatch_fails_closed(tmp_path, parents):
    cfg = json.loads(json.dumps(CONFIG))
    cfg["vision_config"]["window_size"] = 8
    bad = make_ckpt(tmp_path / "bad", 9, config=cfg)
    with pytest.raises(compatibility.IncompatibleParents):
        build(tmp_path, {"T": 0.5, "V": 0.5}, {**parents, "V": bad})


def test_unclassified_tensor_fails_closed(tmp_path, parents):
    names = dict(NAMES)
    names["mystery.weight"] = (2, 2)
    rs = {r: make_ckpt(tmp_path / f"u{r}", s, names=names) for r, s in (("B", 1), ("T", 2), ("V", 3))}
    with pytest.raises(compatibility.IncompatibleParents):
        build(tmp_path, {"T": 0.5, "V": 0.5}, rs)


def test_integer_buffer_conflict_fails(tmp_path):
    rs = {"B": make_ckpt(tmp_path / "B", 1, int_buffer=torch.arange(4)),
          "T": make_ckpt(tmp_path / "T", 2, int_buffer=torch.arange(4)),
          "V": make_ckpt(tmp_path / "V", 3, int_buffer=torch.arange(4) + 1)}
    with pytest.raises(merge.MergeError):
        build(tmp_path, {"T": 0.5, "V": 0.5}, rs)


def test_integer_buffer_equal_is_copied(tmp_path):
    rs = {r: make_ckpt(tmp_path / r, s, int_buffer=torch.arange(4)) for r, s in (("B", 1), ("T", 2), ("V", 3))}
    build(tmp_path, {"T": 0.5, "V": 0.5}, rs)
    got = load_all(tmp_path / "out" / "X")
    assert torch.equal(got["visual.blocks.0.attn.counter"], torch.arange(4))


def test_parents_unchanged(tmp_path, parents):
    before = {r: {f.name: f.read_bytes() for f in p.iterdir()} for r, p in parents.items()}
    build(tmp_path, {"T": 0.5, "V": 0.5}, parents)
    after = {r: {f.name: f.read_bytes() for f in p.iterdir()} for r, p in parents.items()}
    assert before == after


def test_rebuild_same_recipe_is_noop_and_different_recipe_rejected(tmp_path, parents):
    m1 = build(tmp_path, {"T": 0.5, "V": 0.5}, parents)
    m2 = build(tmp_path, {"T": 0.5, "V": 0.5}, parents)
    assert m1["recipe_sha256"] == m2["recipe_sha256"]
    with pytest.raises(merge.MergeError):
        build(tmp_path, {"T": 0.5, "B": 0.5}, parents)


def test_stale_staging_with_other_recipe_rejected(tmp_path, parents):
    staging = tmp_path / "out" / ".X.staging"
    staging.mkdir(parents=True)
    (staging / "recipe.json").write_text(json.dumps({"recipe_sha256": "f" * 64}))
    with pytest.raises(merge.MergeError):
        build(tmp_path, {"T": 0.5, "V": 0.5}, parents)


def test_coefficients_must_sum_to_one(tmp_path, parents):
    with pytest.raises(merge.MergeError):
        build(tmp_path, {"T": 0.6, "V": 0.5}, parents)


def test_tensor_groups_are_a_partition():
    groups = [compatibility.tensor_group(n) for n in NAMES]
    assert "unclassified" not in groups
    assert len(groups) == len(NAMES)  # each tensor in exactly one group by construction (a function)
    vis = {n for n in NAMES if compatibility.tensor_group(n).startswith("visual")}
    lang = {n for n in NAMES if compatibility.tensor_group(n).startswith("language")}
    assert vis.isdisjoint(lang) and vis | lang == set(NAMES)


def test_update_statistics_identify_untouched_groups(tmp_path):
    b = make_ckpt(tmp_path / "B", 1)
    t = tmp_path / "T"
    t.mkdir()
    for f in b.iterdir():
        (t / f.name).write_bytes(f.read_bytes())
    v = make_ckpt(tmp_path / "V", 3)
    stats = merge.parent_update_statistics({"B": b, "T": t, "V": v}, log=lambda *_: None)
    assert stats["bitwise_equal_tensor_counts"]["T"] == len(NAMES)
    assert stats["groups"]["visual_blocks"]["T_delta_norm"] == 0.0
    assert stats["groups"]["visual_blocks"]["V_delta_norm"] > 0.0
