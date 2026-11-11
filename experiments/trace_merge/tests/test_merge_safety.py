"""Fail-closed behaviour of the compatibility audit, the merge builder and the inference views (handoff 7.1-7.4).

Synthetic tests run everywhere; the real-data checks at the end read (never write) the study work root and run only when
TRACE_MERGE_WORK_ROOT is set.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import sys
import threading
import time
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_merge import CONFIG, NAMES, build, load_all, make_ckpt  # noqa: E402
from trace_merge import compatibility, merge, views  # noqa: E402
from trace_merge.config import DEFAULT_CONFIG, load_config, sha256_file  # noqa: E402
from trace_merge.safetensors_io import StreamingShardWriter, tensor_bytes  # noqa: E402

REAL_WORK_ROOT = os.environ.get("TRACE_MERGE_WORK_ROOT")


@pytest.fixture()
def parents(tmp_path):
    return {r: make_ckpt(tmp_path / r, s) for r, s in (("B", 1), ("T", 2), ("V", 3))}


def _cfg(**over):
    c = json.loads(json.dumps(CONFIG))
    c.update(over)
    return c


def _rewrite(d: Path, edit) -> None:
    """Apply ``edit`` to the full tensor dict of a synthetic checkpoint and save it with the same shard layout."""

    full = load_all(d)
    edit(full)
    wm = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    for shard in set(wm.values()):
        save_file({k: full[k] for k, s in wm.items() if s == shard}, str(d / shard), metadata={"format": "pt"})


# ----------------------------------------------------------------------------------------------- audit (R1, R2, R11)
def test_config_token_id_mismatch_is_hard_stop(tmp_path, parents):
    bad = make_ckpt(tmp_path / "bad", 9, config=_cfg(image_token_id=15))
    rep = compatibility.audit({"B": parents["B"], "V": bad})
    assert rep["status"] == "incompatible" and "token.image_token_id" in rep["hard_stops"][0]


def test_architecture_field_missing_on_one_side_is_hard_stop(tmp_path, parents):
    cfg = _cfg()
    del cfg["vision_config"]["fullatt_block_indexes"]
    bad = make_ckpt(tmp_path / "bad", 9, config=cfg)
    rep = compatibility.audit({"B": parents["B"], "V": bad})
    assert rep["status"] == "incompatible" and "vision.fullatt_block_indexes" in rep["hard_stops"][0]


def test_config_eos_scalar_vs_list_is_not_a_stop(tmp_path):
    a = make_ckpt(tmp_path / "a", 1, config=_cfg(eos_token_id=15))
    b = make_ckpt(tmp_path / "b", 2, config=_cfg(eos_token_id=[15]))
    c = make_ckpt(tmp_path / "c", 3, config=_cfg(eos_token_id=[15, 14]))
    assert compatibility.audit({"A": a, "B": b})["status"] == "compatible"
    rep = compatibility.audit({"A": a, "C": c})
    assert rep["status"] == "compatible" and any("config eos_token_id" in f for f in rep["findings"])


@pytest.mark.parametrize("kind", ["py", "auto_map"])
def test_parent_with_custom_code_is_hard_stop(tmp_path, parents, kind):
    v = parents["V"]
    if kind == "py":
        (v / "modeling_qwen2_5_vl.py").write_text("# custom code\n")
    else:
        cfg = json.loads((v / "config.json").read_text())
        cfg["auto_map"] = {"AutoModel": "modeling_x.Model"}
        (v / "config.json").write_text(json.dumps(cfg))
    rep = compatibility.audit({"B": parents["B"], "V": v})
    assert rep["status"] == "incompatible" and any("custom code" in h for h in rep["hard_stops"])


def test_weights_outside_safetensors_are_hard_stop(tmp_path, parents):
    (parents["V"] / "pytorch_model.bin").write_bytes(b"\0")
    rep = compatibility.audit({"B": parents["B"], "V": parents["V"]})
    assert rep["status"] == "incompatible" and any("outside safetensors" in h for h in rep["hard_stops"])


def test_unaudited_files_are_recorded_and_doc_files_ignored(tmp_path, parents):
    v = parents["V"]
    for n in ("processor_config.json", "tokenizer.model", "README.md", ".gitattributes"):
        (v / n).write_text("x")
    (v / "assets").mkdir()
    rep = compatibility.audit({"B": parents["B"], "V": v})
    assert rep["status"] == "compatible"
    assert rep["unaudited_files"] == {"B": [], "V": ["processor_config.json", "tokenizer.model"]}
    assert any("processor_config.json" in f for f in rep["findings"])


def test_processor_resample_or_normalize_mismatch_is_hard_stop(tmp_path, parents):
    v = parents["V"]
    pc = json.loads((v / "preprocessor_config.json").read_text())
    pc["resample"] = 0
    (v / "preprocessor_config.json").write_text(json.dumps(pc))
    rep = compatibility.audit({"B": parents["B"], "V": v})
    assert rep["status"] == "incompatible" and "resample" in rep["hard_stops"][0]


def test_processor_implicit_defaults_equal_explicit_values():
    """The B-style (implicit) and T/V-style (explicit) Qwen2.5-VL processor configs take the same effect."""

    implicit = {"min_pixels": 3136, "max_pixels": 12845056, "patch_size": 14, "temporal_patch_size": 2, "merge_size": 2,
                "image_mean": [0.48145466, 0.4578275, 0.40821073], "image_std": [0.26862954, 0.26130258, 0.27577711],
                "image_processor_type": "Qwen2VLImageProcessor"}
    explicit = {**implicit, "do_convert_rgb": True, "do_normalize": True, "do_rescale": True, "do_resize": True,
                "resample": 3, "rescale_factor": 0.00392156862745098, "do_pad": None,
                "size": {"longest_edge": 12845056, "shortest_edge": 3136},
                "image_processor_type": "Qwen2VLImageProcessorFast"}
    assert compatibility.effective_processor(implicit) == compatibility.effective_processor(explicit)
    assert compatibility.effective_processor({**explicit, "max_pixels": 1003520})["max_pixels"] == 1003520


# ----------------------------------------------------------------------------------------------- recipe and arithmetic (R7, R8)
@pytest.mark.parametrize("coeffs", [{"T": 1.5, "V": -0.5}, {"T": float("nan"), "V": 0.5},
                                    {"T": float("inf"), "V": 0.5}])
def test_coefficients_must_be_finite_interpolation_weights(parents, coeffs):
    with pytest.raises(merge.MergeError):
        merge.recipe_for("X", coeffs, parents, "B", {r: "0" * 40 for r in parents})


def test_zero_coefficient_term_is_skipped():
    a = torch.tensor([-0.0, 1.0, -0.0], dtype=torch.bfloat16)
    b = torch.tensor([1.0, float("nan"), 0.0], dtype=torch.bfloat16)
    out = merge.interpolate_slice([a, b], [1.0, 0.0])
    assert torch.equal(out.view(torch.int16), a.view(torch.int16))


def test_interpolation_rounds_ties_to_even_literal():
    a = torch.tensor([1.0, 1.0078125, -1.0, 3.0e38], dtype=torch.bfloat16)
    b = torch.tensor([1.0078125, 1.015625, -1.0078125, 3.0e38], dtype=torch.bfloat16)
    out = merge.interpolate_slice([a, b], [0.5, 0.5])
    # exact midpoints 1.00390625 and 1.01171875 are bf16 ties: they round to the even mantissa (1.0 and 1.015625)
    assert out.tolist()[:3] == [1.0, 1.015625, -1.0]
    assert torch.isfinite(out).all()


def test_nonfinite_parent_fails_closed_without_final(tmp_path, parents):
    d = tmp_path / "inf"
    shutil.copytree(parents["V"], d)
    _rewrite(d, lambda full: full["model.norm.weight"].__setitem__(1, float("inf")))
    with pytest.raises(merge.MergeError):
        build(tmp_path, {"T": 0.5, "V": 0.5}, {**parents, "V": d})
    assert not (tmp_path / "out" / "X").exists()
    build(tmp_path, {"T": 0.5, "V": 0.5}, parents)  # same recipe, good parents: stale staging replaced, lock released
    assert (tmp_path / "out" / "X" / merge.MANIFEST_NAME).is_file()


def test_float32_parents_refused(tmp_path, parents):
    rs = {r: tmp_path / f"f{r}" for r in parents}
    for r, d in rs.items():
        shutil.copytree(parents[r], d)
        _rewrite(d, lambda full: full.update({k: v.float() for k, v in full.items()}))
    with pytest.raises(merge.MergeError):
        build(tmp_path, {"T": 0.5, "V": 0.5}, rs)


def test_tied_config_without_lm_head_merges_without_head(tmp_path):
    names = {k: v for k, v in NAMES.items() if k != "lm_head.weight"}
    cfg = _cfg(tie_word_embeddings=True)
    rs = {r: make_ckpt(tmp_path / r, s, names=names, config=cfg) for r, s in (("B", 1), ("T", 2), ("V", 3))}
    build(tmp_path, {"T": 0.5, "V": 0.5}, rs)
    got = load_all(tmp_path / "out" / "X")
    assert "lm_head.weight" not in got and "model.embed_tokens.weight" in got


def test_untied_config_without_lm_head_is_hard_stop(tmp_path):
    names = {k: v for k, v in NAMES.items() if k != "lm_head.weight"}
    rs = {r: make_ckpt(tmp_path / r, s, names=names) for r, s in (("B", 1), ("T", 2), ("V", 3))}
    with pytest.raises(compatibility.IncompatibleParents):
        build(tmp_path, {"T": 0.5, "V": 0.5}, rs)


def test_untied_embeddings_and_head_are_merged_independently(tmp_path, parents):
    build(tmp_path, {"T": 0.5, "V": 0.5}, parents)
    got = load_all(tmp_path / "out" / "X")
    assert not torch.equal(got["lm_head.weight"].view(torch.int16), got["model.embed_tokens.weight"].view(torch.int16))


def test_multi_slice_matches_single_slice(tmp_path, parents):
    build(tmp_path, {"T": 0.5, "V": 0.5}, parents, mid="S1", max_slice=1)
    build(tmp_path, {"T": 0.5, "V": 0.5}, parents, mid="S2", max_slice=1 << 20)
    a, b = load_all(tmp_path / "out" / "S1"), load_all(tmp_path / "out" / "S2")
    assert all(torch.equal(a[k].view(torch.int16), b[k].view(torch.int16)) for k in a)


# ----------------------------------------------------------------------------------------------- build checks (R4, R5, R6, R9, R12)
def test_rebuild_of_finished_merge_detects_corrupted_output(tmp_path, parents):
    build(tmp_path, {"T": 0.5, "V": 0.5}, parents)
    shard = sorted((tmp_path / "out" / "X").glob("*.safetensors"))[0]
    raw = bytearray(shard.read_bytes())
    raw[-1] ^= 0xFF
    shard.write_bytes(bytes(raw))
    build(tmp_path, {"T": 0.5, "V": 0.5}, parents, dry_run=True)  # a dry run checks structure only
    with pytest.raises(merge.MergeError, match="sha256"):
        build(tmp_path, {"T": 0.5, "V": 0.5}, parents)


def test_rebuild_of_finished_merge_detects_truncation_and_extra_files(tmp_path, parents):
    build(tmp_path, {"T": 0.5, "V": 0.5}, parents)
    final = tmp_path / "out" / "X"
    (final / "stray.txt").write_text("x")
    with pytest.raises(merge.MergeError, match="disagree"):
        build(tmp_path, {"T": 0.5, "V": 0.5}, parents, dry_run=True)
    (final / "stray.txt").unlink()
    shard = sorted(final.glob("*.safetensors"))[0]
    shard.write_bytes(shard.read_bytes()[:-2])
    with pytest.raises(merge.MergeError):
        build(tmp_path, {"T": 0.5, "V": 0.5}, parents, dry_run=True)


def test_rebuild_with_a_different_layout_parent_is_refused(tmp_path, parents):
    build(tmp_path, {"T": 0.5, "V": 0.5}, parents)
    b1 = make_ckpt(tmp_path / "B1", 1, shards=1)
    with pytest.raises(merge.MergeError, match="layout"):
        build(tmp_path, {"T": 0.5, "V": 0.5}, {**parents, "B": b1})


def test_manifest_records_layout_parent_code_and_runtime_revisions(tmp_path, parents):
    lineage = {r: {"repo_id": f"x/{r}", "runtime_revision": f"rt-{r}"} for r in parents}
    man = build(tmp_path, {"T": 0.5, "V": 0.5}, parents, lineage=lineage)
    assert man["layout_parent"] == {"role": "B", "source_revision": "0" * 40,
                                    "index_sha256": sha256_file(parents["B"] / "model.safetensors.index.json")}
    assert man["runtime_revisions"] == {"T": "rt-T", "V": "rt-V"}
    assert set(man["implementation_code_sha256"]) == {f"trace_merge/{n}" for n in merge.CODE_FILES}


def test_parent_hashes_are_rechecked_after_the_build(tmp_path, parents):
    recorded = {r: {f.name: sha256_file(f) for f in sorted(parents[r].glob("*.safetensors"))} for r in ("T", "V")}
    man = build(tmp_path, {"T": 0.5, "V": 0.5}, parents, parent_sha256=recorded)
    assert man["parent_weight_file_sha256_rechecked_after_build"] is True
    wrong = {**recorded, "V": {k: "0" * 64 for k in recorded["V"]}}
    with pytest.raises(merge.MergeError, match="after the build"):
        build(tmp_path, {"T": 0.5, "V": 0.5}, parents, mid="Y", parent_sha256=wrong)
    assert not (tmp_path / "out" / "Y").exists()


@pytest.mark.parametrize("full", [False, True])
def test_inbuild_verification_catches_a_corrupted_later_slice(tmp_path, parents, monkeypatch, full):
    """A writer/offset bug that only hits rows >= 2 of a tensor must not be finalized."""

    real_bytes, real_iter = merge.tensor_bytes, merge.iter_row_slices
    cur = {"r0": 0, "writing": True}

    def it(info, max_elements):
        for r0, r1 in real_iter(info, max_elements):
            cur["r0"] = r0
            yield r0, r1

    def bad(t):
        raw = bytearray(real_bytes(t))
        if cur["r0"] >= 2:
            raw[0] ^= 0x01
        return bytes(raw)

    monkeypatch.setattr(merge, "iter_row_slices", it)
    monkeypatch.setattr(merge, "tensor_bytes", bad)
    with pytest.raises(merge.MergeError, match="verification failed"):
        build(tmp_path, {"T": 0.5, "V": 0.5}, parents, max_slice=4, full_verify=full)
    assert not (tmp_path / "out" / "X").exists()


def test_concurrent_builds_of_one_merge_are_serialized(tmp_path, parents, monkeypatch):
    real = merge.merge_tensor
    started = threading.Event()

    def slow(*a, **k):
        if threading.current_thread().name == "A" and not started.is_set():
            started.set()
            time.sleep(1.0)
        return real(*a, **k)

    monkeypatch.setattr(merge, "merge_tensor", slow)
    errs: dict[str, str] = {}

    def run(name):
        try:
            build(tmp_path, {"T": 0.5, "V": 0.5}, parents)
        except Exception as exc:  # noqa: BLE001
            errs[name] = f"{type(exc).__name__}: {exc}"

    a = threading.Thread(target=run, args=("A",), name="A")
    a.start()
    assert started.wait(10)
    b = threading.Thread(target=run, args=("B",), name="B")
    b.start()
    a.join()
    b.join()
    assert set(errs) == {"B"} and "lock held" in errs["B"], errs
    final = tmp_path / "out" / "X"
    man = json.loads((final / merge.MANIFEST_NAME).read_text())
    merge.verify_finished(final, man, deep=True)


# ----------------------------------------------------------------------------------------------- writer (R10)
def test_streaming_writer_aligns_mixed_dtypes(tmp_path):
    t = {"a": torch.randn(3).to(torch.bfloat16), "b": torch.arange(3, dtype=torch.int64),
         "c": torch.tensor([True, False, True])}
    spec = [("a", "BF16", (3,)), ("b", "I64", (3,)), ("c", "BOOL", (3,))]
    w = StreamingShardWriter(tmp_path / "x.safetensors", spec, None)
    hdr = json.loads(w.header_bytes[8:])
    for n in w.order:
        w.begin(n)
        w.write(tensor_bytes(t[n]))
        w.end()
    w.close()
    assert all(hdr[n]["data_offsets"][0] % {"a": 2, "b": 8, "c": 1}[n] == 0 for n in t), hdr
    got = load_all(tmp_path)
    assert torch.equal(got["b"], t["b"]) and torch.equal(got["c"], t["c"])
    assert torch.equal(got["a"].view(torch.int16), t["a"].view(torch.int16))


def test_streaming_writer_header_follows_spec(tmp_path):
    t = torch.randn(3, 5).to(torch.bfloat16)
    w = StreamingShardWriter(tmp_path / "x.safetensors", [("a", "BF16", (3, 5))], {"format": "pt"})
    w.begin("a")
    w.write(tensor_bytes(t))
    w.end()
    w.close()
    raw = (tmp_path / "x.safetensors").read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    hdr = raw[8:8 + n]
    assert n % 8 == 0 and hdr[:1] == b"{" and set(hdr[len(hdr.rstrip(b" ")):]) <= {0x20}
    j = json.loads(hdr)
    assert j["__metadata__"] == {"format": "pt"} and j["a"]["data_offsets"] == [0, 30] and len(raw) == 8 + n + 30


# ----------------------------------------------------------------------------------------------- tensor ownership (R13)
def qwen25_vl_7b_names() -> set[str]:
    names = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight", "visual.patch_embed.proj.weight",
             "visual.merger.ln_q.weight", *(f"visual.merger.mlp.{i}.{p}" for i in (0, 2) for p in ("weight", "bias"))}
    for i in range(28):
        pre = f"model.layers.{i}."
        names |= {pre + f"self_attn.{m}_proj.{p}" for m in "qkv" for p in ("weight", "bias")}
        names |= {pre + n for n in ("self_attn.o_proj.weight", "input_layernorm.weight",
                                    "post_attention_layernorm.weight")}
        names |= {pre + f"mlp.{m}_proj.weight" for m in ("gate", "up", "down")}
    for i in range(32):
        pre = f"visual.blocks.{i}."
        names |= {pre + "norm1.weight", pre + "norm2.weight"}
        names |= {pre + f"attn.{m}.{p}" for m in ("qkv", "proj") for p in ("weight", "bias")}
        names |= {pre + f"mlp.{m}_proj.{p}" for m in ("gate", "up", "down") for p in ("weight", "bias")}
    return names


EXPECTED_GROUPS = {"language_embeddings": 1, "language_final_norm": 1, "language_layers": 336,
                   "language_output_head": 1, "visual_blocks": 384, "visual_merger": 5, "visual_patch_embed": 1}


def _groups(names) -> dict[str, int]:
    out: dict[str, int] = {}
    for n in names:
        g = compatibility.tensor_group(n)
        out[g] = out.get(g, 0) + 1
    return out


def test_tensor_groups_cover_the_full_qwen25_vl_7b_layout():
    names = qwen25_vl_7b_names()
    assert len(names) == 729 and _groups(names) == EXPECTED_GROUPS
    assert compatibility.tensor_group("visual.extra.weight") == "unclassified"


# ----------------------------------------------------------------------------------------------- views (R2, R3)
def _view_env(tmp_path, monkeypatch, *, extra_b_file: str | None = None, parent_hashes: bool = True):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    cfg = load_config(DEFAULT_CONFIG, work_root=tmp_path / "work")
    b, t, v = (cfg.hf_snapshot(r) for r in ("B", "T", "V"))
    make_ckpt(t, 2)
    make_ckpt(v, 3)
    b.mkdir(parents=True)
    for n in cfg.raw["protocols"]["P1_backbone"]["files"]:
        (b / n).write_text(json.dumps({"file": n}))
    for n in ("README.md", ".gitattributes"):
        (b / n).write_text("doc")
    (b / "assets").mkdir()
    if extra_b_file:
        (b / extra_b_file).write_text("{{ 'runtime-relevant' }}")
    if parent_hashes:
        hashes = {r: {f.name: sha256_file(f) for f in sorted(cfg.hf_snapshot(r).glob("*.safetensors"))}
                  for r in ("B", "T", "V")}
        (tmp_path / "work" / "audit").mkdir(parents=True)
        (tmp_path / "work" / "audit" / "parent_weight_sha256.json").write_text(json.dumps(hashes))
    return cfg


def test_view_serves_only_weights_and_listed_files(tmp_path, monkeypatch):
    cfg = _view_env(tmp_path, monkeypatch)
    views.build_view(cfg, "V@P1")
    out = cfg.work("views", "vero-p1")
    listed = set(cfg.raw["protocols"]["P1_backbone"]["files"])
    v = cfg.hf_snapshot("V")
    weights = {p.name for p in v.iterdir() if p.suffix == ".safetensors" or p.name == "model.safetensors.index.json"}
    assert {p.name for p in out.iterdir()} == listed | weights | {views.VIEW_MANIFEST}
    for n in listed:
        assert (out / n).read_bytes() == (cfg.hf_snapshot("B") / n).read_bytes() and not (out / n).is_symlink()
    for n in weights:
        assert (out / n).resolve() == (v / n).resolve()
    assert views.build_view(cfg, "V@P1")["slug"] == "vero-p1"  # reuse after the checks pass


def test_view_refuses_unlisted_runtime_file_in_view_parent(tmp_path, monkeypatch):
    cfg = _view_env(tmp_path, monkeypatch, extra_b_file="chat_template.jinja")
    with pytest.raises(views.ViewError, match="chat_template.jinja"):
        views.build_view(cfg, "V@P1")


@pytest.mark.parametrize("tamper", ["view_file", "retarget_link", "extra_file"])
def test_view_reuse_detects_tampering(tmp_path, monkeypatch, tamper):
    cfg = _view_env(tmp_path, monkeypatch)
    views.build_view(cfg, "V@P1")
    out = cfg.work("views", "vero-p1")
    (out / views.MARKER_NAME).write_text("{}")  # the protocol's registration marker is allowed
    views.build_view(cfg, "V@P1")
    if tamper == "view_file":
        (out / "chat_template.json").write_text("tampered")
    elif tamper == "retarget_link":
        link = sorted(out.glob("*.safetensors"))[0]
        link.unlink()
        link.symlink_to(sorted(cfg.hf_snapshot("T").glob("*.safetensors"))[0])
    else:
        (out / "added_tokens.json").write_text("{}")
    with pytest.raises(views.ViewError):
        views.build_view(cfg, "V@P1")


def test_view_of_a_parent_requires_matching_recorded_weight_hashes(tmp_path, monkeypatch):
    cfg = _view_env(tmp_path, monkeypatch, parent_hashes=False)
    with pytest.raises(views.ViewError, match="parent-hashes"):
        views.build_view(cfg, "V@P1")
    path = cfg.work("audit", "parent_weight_sha256.json")
    path.parent.mkdir(parents=True)
    shards = sorted(cfg.hf_snapshot("V").glob("*.safetensors"))
    path.write_text(json.dumps({"V": {f.name: "0" * 64 for f in shards}}))
    with pytest.raises(views.ViewError, match="sha256"):
        views.build_view(cfg, "V@P1")
    assert not cfg.work("views", "vero-p1").exists()


def test_view_of_a_merge_checks_its_manifest_hashes(tmp_path, monkeypatch):
    cfg = _view_env(tmp_path, monkeypatch)
    ps = {r: cfg.hf_snapshot(r) for r in ("T", "V")}
    merge.build_merge("M050", {"T": 0.5, "V": 0.5}, {**ps, "B": ps["T"]}, cfg.work("merges"), layout_role="T",
                      source_revisions={r: "0" * 40 for r in ("B", "T", "V")}, log=lambda *_: None,
                      max_slice_elements=7)
    man_path = cfg.work("merges", "M050", merge.MANIFEST_NAME)
    man = json.loads(man_path.read_text())
    assert views.build_view(cfg, "M050@P1")["weight_source_revision"] == man["content_revision"]
    man["output_file_hashes"][sorted(man["output_file_hashes"])[0]] = "0" * 64
    man_path.write_text(json.dumps(man))
    with pytest.raises(views.ViewError, match="sha256"):
        views.build_view(cfg, "M050@P1", force=True)
    assert (cfg.work("views", "m050-p1") / views.VIEW_MANIFEST).is_file()  # a failed forced rebuild keeps the view


# ----------------------------------------------------------------------------------------------- real data (read-only)
def _real_cfg():
    cfg = load_config(DEFAULT_CONFIG, work_root=REAL_WORK_ROOT)
    if not cfg.work("merges").is_dir():
        pytest.skip("work root has no merges")
    return cfg


@pytest.mark.skipif(not REAL_WORK_ROOT, reason="real-data check needs TRACE_MERGE_WORK_ROOT")
def test_real_merges_unchanged_by_the_code_and_still_reusable():
    """Recipes reproduce the recorded recipe_sha256, outputs keep their structure and B's layout, groups partition."""

    cfg = _real_cfg()
    revisions = {r: cfg.role(r)["source_revision"] for r in ("B", "T", "V")}
    mb = cfg.raw["merge_build"]
    for mid, spec in cfg.merges.items():
        final = cfg.work("merges", mid)
        man = json.loads((final / merge.MANIFEST_NAME).read_text(encoding="utf-8"))
        coeffs = {k: float(v) for k, v in spec["coefficients"].items()}
        recipe = merge.recipe_for(mid, coeffs, {r: Path(r) for r in coeffs}, mb["shard_layout"], revisions,
                                  max_slice_elements=int(mb["max_slice_elements"]))
        assert merge.recipe_sha256(recipe) == man["recipe_sha256"], mid
        merge.verify_finished(final, man, deep=False)
        names = json.loads((final / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
        assert _groups(names) == EXPECTED_GROUPS and set(names) == qwen25_vl_7b_names()
        b = cfg.hf_snapshot("B")
        if b.is_dir():
            assert merge._layout(final, set(names)) == merge._layout(b, set(names))


@pytest.mark.skipif(not REAL_WORK_ROOT, reason="real-data check needs TRACE_MERGE_WORK_ROOT")
def test_real_views_pass_the_reuse_checks():
    cfg = _real_cfg()
    for label, (slug, source, protocol_id) in views.VIEW_SPECS.items():
        out = cfg.work("views", slug)
        if not (out / views.VIEW_MANIFEST).is_file():
            continue
        wdir = views.weight_source_dir(cfg, source)
        if not wdir.is_dir():
            pytest.skip(f"weight source of {label} not present")
        manifest = json.loads((out / views.VIEW_MANIFEST).read_text(encoding="utf-8"))
        weights = sorted({p for g in views.WEIGHT_GLOBS for p in wdir.glob(g)})
        views._check_existing(out, manifest, weights)


@pytest.mark.skipif(not REAL_WORK_ROOT, reason="real-data check needs TRACE_MERGE_WORK_ROOT")
def test_real_parents_remain_compatible_under_the_stricter_audit():
    cfg = _real_cfg()
    parents = {r: cfg.hf_snapshot(r) for r in ("B", "T", "V")}
    if not all(p.is_dir() for p in parents.values()):
        pytest.skip("pinned parent snapshots not present")
    rep = compatibility.audit(parents)
    assert rep["status"] == "compatible", rep["hard_stops"]
    assert rep["architecture"]["differences_vs_reference"] == {}
    for label, (slug, source, protocol_id) in views.VIEW_SPECS.items():
        vdir = parents[views.VIEW_PARENT[protocol_id]]
        listed = set(cfg.raw["protocols"][protocol_id]["files"])
        extra = [p.name for p in vdir.iterdir() if p.name not in listed and p.name not in compatibility.DOC_FILES
                 and not p.name.endswith(".safetensors") and p.name != "model.safetensors.index.json"]
        assert extra == [], (label, extra)
