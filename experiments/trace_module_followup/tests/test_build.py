"""Component ownership and exact-copy hybrid construction on tiny synthetic Qwen2.5-VL-shaped checkpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trace_followup import build, components, inventory  # noqa: E402
from trace_followup.common import FollowupError, load  # noqa: E402
from trace_merge.merge import file_identity  # noqa: E402

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
FRONTEND = {n for n in NAMES if n.startswith("visual.")}


def make_ckpt(path: Path, seed: int, shards: int, same_as: dict | None = None) -> dict[str, torch.Tensor]:
    path.mkdir(parents=True)
    g = torch.Generator().manual_seed(seed)
    tensors = {n: torch.randn(*s, generator=g).to(torch.bfloat16) for n, s in NAMES.items()}
    for n, t in (same_as or {}).items():  # tensors this parent leaves unchanged from B
        tensors[n] = t.clone()
    names = sorted(tensors)
    weight_map = {}
    for i in range(shards):
        part = names[i::shards]
        fname = f"model-{i + 1:05d}-of-{shards:05d}.safetensors"
        save_file({n: tensors[n] for n in part}, str(path / fname), metadata={"format": "pt"})
        weight_map.update({n: fname for n in part})
    (path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    return tensors


@pytest.fixture()
def setup(tmp_path, monkeypatch):
    b = make_ckpt(tmp_path / "B", 0, 2)
    t = make_ckpt(tmp_path / "T", 1, 1, same_as={"model.norm.weight": b["model.norm.weight"]})
    v = make_ckpt(tmp_path / "V", 2, 3, same_as={"visual.merger.mlp.0.weight": b["visual.merger.mlp.0.weight"]})
    cfg = load(work_root=tmp_path / "work")
    dirs = {"B": tmp_path / "B", "T": tmp_path / "T", "V": tmp_path / "V"}
    monkeypatch.setattr(type(cfg), "parent_dir", lambda self, r: dirs[r])
    scans = {r: build.scan_checkpoint(d) for r, d in dirs.items()}
    inv = {"inventory_sha256": "synthetic",
           "parents": {r: {"files": scans[r]["files"],
                           "file_identities": {f.name: file_identity(f) for f in sorted(d.glob("*.safetensors"))},
                           "tensors": {n: {"sha256": v["sha256"]} for n, v in scans[r]["tensors"].items()}}
                       for r, d in dirs.items()}}
    return cfg, inv, {"B": b, "T": t, "V": v}, dirs


def load_tensors(d: Path) -> dict[str, torch.Tensor]:
    out = {}
    for f in sorted(d.glob("*.safetensors")):
        with safe_open(str(f), "pt") as h:
            for n in h.keys():
                out[n] = h.get_tensor(n)
    return out


def test_ownership_has_no_fallback():
    cfg = load()
    with pytest.raises(FollowupError):
        components.part_of("visual.unknown.weight", cfg.raw["parts"])
    owner = components.ownership(cfg, NAMES, "component")
    assert {n for n, g in owner.items() if g == "frontend"} == FRONTEND
    fine = components.ownership(cfg, NAMES, "encoder_merger")
    assert {n for n, g in fine.items() if g == "merger"} == {"visual.merger.mlp.0.weight"}


def test_hybrids_copy_donor_tensors_exactly(setup):
    cfg, inv, parents, dirs = setup
    for hid, spec in ((h, cfg.hybrids[h]) for h in ("TF", "TL", "VF", "VL")):
        m = build.build_hybrid(cfg, hid, inv, log=lambda _m: None)
        got = load_tensors(cfg.work("hybrids", hid))
        assert set(got) == set(NAMES)
        for n, t in got.items():
            donor = spec["donors"]["frontend" if n in FRONTEND else "language"]
            assert torch.equal(t.view(torch.int16), parents[donor][n].view(torch.int16)), (hid, n)
        # B's tensor-to-shard layout, read-only files, manifest records every tensor's donor
        idx = json.loads((cfg.work("hybrids", hid) / "model.safetensors.index.json").read_text())["weight_map"]
        assert idx == json.loads((dirs["B"] / "model.safetensors.index.json").read_text())["weight_map"]
        assert all(m["tensor_sources"][n]["donor"] == spec["donors"]["frontend" if n in FRONTEND else "language"]
                   for n in NAMES)
        assert build.verify_hybrid(cfg, hid, inv)["donor_equal"]
    # rebuilding with the same recipe reuses, never overwrites
    assert build.build_hybrid(cfg, "TF", inv, log=lambda _m: None)["hybrid_id"] == "TF"


def test_identity_and_ownership_checks(setup):
    cfg, inv, _parents, _dirs = setup
    ident = inventory.identity_checks(cfg, inv)
    assert all(v["equal"] for v in ident.values())
    own = inventory.ownership_checks(cfg, inv)
    # T leaves model.norm unchanged, V leaves the merger unchanged
    assert own["T"] == {"changed_vs_B": len(NAMES) - 1, "TF": len(FRONTEND), "TL": len(NAMES) - len(FRONTEND) - 1,
                        "partition": True}
    assert own["V"]["VF"] == len(FRONTEND) - 1


def test_identity_check_detects_wrong_bytes(setup):
    cfg, inv, _parents, _dirs = setup
    inv["parents"]["T"]["tensors"]["lm_head.weight"]["sha256"] = "0" * 64
    with pytest.raises(FollowupError):
        inventory.identity_checks(cfg, inv)


def test_nonfinite_count():
    t = torch.tensor([1.0, float("inf"), float("nan"), -0.0, -float("inf")], dtype=torch.bfloat16)
    assert build.nonfinite_count(t.view(torch.int16).numpy().tobytes(), "BF16") == 3


def test_decoder_block_reversion_ownership():
    cfg = load()
    names = [f"model.layers.{i}.mlp.up_proj.weight" for i in range(28)] + [
        "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight", "visual.blocks.0.attn.qkv.weight",
        "visual.merger.mlp.0.weight", "visual.patch_embed.proj.weight"]
    for hid, (lo, hi) in {"TR0": (0, 8), "TR1": (9, 18), "TR2": (19, 27)}.items():
        dm = components.donor_map(cfg, hid, names)
        reverted = {n for n, r in dm.items() if r == "B"}
        assert reverted == {f"model.layers.{i}.mlp.up_proj.weight" for i in range(lo, hi + 1)}, hid
        assert all(r == "T" for n, r in dm.items() if n not in reverted)
    with pytest.raises(FollowupError):  # a layer outside every range has no owner
        components.part_of("model.layers.28.mlp.up_proj.weight", cfg.raw["fine_parts"])


def test_f1_recipes_unchanged_by_fine_partitions():
    """The F1 hybrids' recipes (and so their recipe hashes) must not change when F2c parts are added."""
    cfg = load()
    for hid in ("TF", "TL", "VF", "VL"):
        r = build.recipe_for(cfg, hid, list(NAMES))
        assert r["parts"] == cfg.raw["parts"] and r["partition_groups"] == cfg.raw["partitions"]["component"]
