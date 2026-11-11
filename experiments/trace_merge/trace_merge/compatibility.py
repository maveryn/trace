"""Mandatory parent compatibility audit before any weights are materialized (handoff 7.1).

Hard stops (no padding, truncation, vocabulary remapping or ignored keys): differing tensor names, shapes or dtypes;
differing token-to-id maps, special/added tokens or config token ids; differing architecture settings (a field present
in one parent and missing in another counts as different: runtime defaults are not assumed); differing effective image
processor settings; adapters, quantized parents, custom model code, or weights outside safetensors.
Everything else (serialization format of configs, processor class spelling, config eos lists, files outside the audited
set, lineage that a model card does not pin) is recorded as a finding with its consequence rather than silently
normalized.
"""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Any

from .config import sha256_file
from .safetensors_io import FLOAT_DTYPES, index_checkpoint

# Architecture fields that must agree (looked up at the top level, then under text_config / vision_config).
TEXT_FIELDS = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
               "vocab_size", "rms_norm_eps", "rope_theta", "max_window_layers", "hidden_act", "tie_word_embeddings",
               "max_position_embeddings", "use_sliding_window")
VISION_FIELDS = ("depth", "hidden_size", "intermediate_size", "num_heads", "out_hidden_size", "patch_size",
                 "spatial_merge_size", "temporal_patch_size", "window_size", "fullatt_block_indexes", "hidden_act",
                 "in_chans", "tokens_per_second")
TOKEN_ID_FIELDS = ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id",
                   "vision_token_id", "bos_token_id", "eos_token_id")
NON_WEIGHT_FILES = ("config.json", "generation_config.json", "preprocessor_config.json", "tokenizer.json",
                    "tokenizer_config.json", "vocab.json", "merges.txt", "added_tokens.json", "special_tokens_map.json",
                    "chat_template.json", "chat_template.jinja", "video_preprocessor_config.json")
WEIGHT_PATTERNS = ("*.safetensors", "model.safetensors.index.json")
# repository/download bookkeeping that neither the loader nor the inference view reads (the last is the protocol's own
# register-local marker)
DOC_FILES = ("README.md", ".gitattributes", ".merge.lock", "assets", ".trace_model_revision.json")
# weight files in other formats: the merge reads only safetensors, so these would be silently ignored
OTHER_WEIGHT_PATTERNS = ("pytorch_model*.bin", "*.pt", "*.pth", "*.ckpt", "*.gguf", "tf_model*.h5",
                         "flax_model*.msgpack")
# Effective image processor settings: fast Qwen2-VL image processor defaults in the pinned transformers (4.57), which
# the pinned environment loads for every parent; ``size`` defaults to these edges and min/max_pixels override it.
PROCESSOR_DEFAULTS = {"do_resize": True, "resample": 3, "do_rescale": True, "rescale_factor": 1 / 255,
                      "do_normalize": True, "image_mean": [0.48145466, 0.4578275, 0.40821073],
                      "image_std": [0.26862954, 0.26130258, 0.27577711], "do_convert_rgb": True, "patch_size": 14,
                      "temporal_patch_size": 2, "merge_size": 2}
PROCESSOR_DEFAULT_SIZE = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}


class IncompatibleParents(RuntimeError):
    pass


def _cfg_get(cfg: dict[str, Any], section: str | None, key: str) -> Any:
    """Field lookup tolerant of the two Qwen2.5-VL config layouts (transformers 4.41 flat vs 4.57 nested)."""

    if section is None:
        if key in cfg and cfg[key] is not None:
            return cfg[key]
        return (cfg.get("text_config") or {}).get(key)
    sub = cfg.get(section) or {}
    return sub.get(key)


def _rope_sections(cfg: dict[str, Any]) -> Any:
    for src in (cfg.get("rope_scaling"), (cfg.get("text_config") or {}).get("rope_scaling")):
        if isinstance(src, dict) and "mrope_section" in src:
            return list(src["mrope_section"])
    return None


def architecture_signature(cfg: dict[str, Any]) -> dict[str, Any]:
    archs = cfg.get("architectures") or (cfg.get("text_config") or {}).get("architectures")
    sig: dict[str, Any] = {"architectures": archs, "model_type": cfg.get("model_type"),
                           "mrope_section": _rope_sections(cfg)}
    for k in TEXT_FIELDS:
        sig[f"text.{k}"] = _cfg_get(cfg, None, k)
    for k in VISION_FIELDS:
        sig[f"vision.{k}"] = _cfg_get(cfg, "vision_config", k)
    for k in TOKEN_ID_FIELDS:
        v = cfg.get(k)
        if v is None:
            v = (cfg.get("text_config") or {}).get(k)
        sig[f"token.{k}"] = v
    return sig


def tokenizer_signature(model_dir: Path) -> dict[str, Any]:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    vocab = tok.get_vocab(with_added_tokens=True)
    spec = json.loads(tok.to_str())
    added = sorted((a["id"], a["content"], bool(a.get("special"))) for a in spec.get("added_tokens", []))
    import hashlib

    vocab_digest = hashlib.sha256(json.dumps(sorted(vocab.items()), ensure_ascii=False).encode("utf-8")).hexdigest()
    tcfg = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    return {
        "vocab_size_with_added": len(vocab),
        "vocab_sha256": vocab_digest,
        "added_tokens": added,
        "model_type": spec["model"]["type"],
        "pre_tokenizer": (spec.get("pre_tokenizer") or {}).get("type"),
        "normalizer": (spec.get("normalizer") or {}).get("type"),
        "decoder": (spec.get("decoder") or {}).get("type"),
        "eos_token": tcfg.get("eos_token"),
        "pad_token": tcfg.get("pad_token"),
        "bos_token": tcfg.get("bos_token"),
        "additional_special_tokens": tcfg.get("additional_special_tokens"),
    }


def _files_present(model_dir: Path) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in NON_WEIGHT_FILES:
        p = model_dir / name
        out[name] = sha256_file(p) if p.is_file() else None
    return out


def _matches(name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(name, g) for g in patterns)


def effective_processor(pc: dict[str, Any]) -> dict[str, Any]:
    """The image processor settings that take effect at run time (declared values over the class defaults)."""

    out = {k: (pc[k] if pc.get(k) is not None else v) for k, v in PROCESSOR_DEFAULTS.items()}
    size = dict(pc.get("size") or PROCESSOR_DEFAULT_SIZE)
    if pc.get("min_pixels") is not None:
        size["shortest_edge"] = pc["min_pixels"]
    if pc.get("max_pixels") is not None:
        size["longest_edge"] = pc["max_pixels"]
    out["min_pixels"], out["max_pixels"] = size.get("shortest_edge"), size.get("longest_edge")
    return out


def _token_ids(value: Any) -> list[Any]:
    """A config eos entry as a sorted id list (a scalar and a one-element list are the same stop set)."""

    if value is None:
        return []
    return sorted(set(value)) if isinstance(value, list) else [value]


def audit(parents: dict[str, Path], lineage: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Compare every parent against the first one; raise IncompatibleParents on a hard stop."""

    roles = list(parents)
    ref = roles[0]
    report: dict[str, Any] = {"schema": "trace-merge-compatibility-v1", "reference_role": ref, "roles": roles,
                              "hard_stops": [], "findings": []}

    # ---- adapters / quantization / custom code / files outside the audited set
    unaudited: dict[str, list[str]] = {}
    for r, d in parents.items():
        if (d / "adapter_config.json").exists() or (d / "adapter_model.safetensors").exists():
            report["hard_stops"].append(f"{r}: is an unmerged adapter checkpoint")
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        if cfg.get("quantization_config") or (cfg.get("text_config") or {}).get("quantization_config"):
            report["hard_stops"].append(f"{r}: is quantized ({cfg.get('quantization_config')})")
        entries = sorted(p.name for p in d.iterdir())
        auto_map = [n for n in ("config.json", "tokenizer_config.json", "preprocessor_config.json")
                    if (d / n).is_file() and "auto_map" in json.loads((d / n).read_text(encoding="utf-8"))]
        code = [n for n in entries if n.endswith(".py")]
        if code or auto_map:
            report["hard_stops"].append(f"{r}: ships custom code ({code}, auto_map in {auto_map}); the stock "
                                        "architecture/processor is not guaranteed and needs explicit review")
        other_weights = [n for n in entries if _matches(n, OTHER_WEIGHT_PATTERNS)]
        if other_weights:
            report["hard_stops"].append(f"{r}: has weights outside safetensors that the merge would ignore: "
                                        f"{other_weights}")
        unaudited[r] = [n for n in entries if n not in NON_WEIGHT_FILES and n not in DOC_FILES and n not in code
                        and n not in other_weights and not _matches(n, WEIGHT_PATTERNS)]
        if unaudited[r]:
            report["findings"].append(f"{r}: files outside the audited set (not hashed, not part of the weights or of "
                                      f"an inference view built from the listed files): {unaudited[r]}")
    report["unaudited_files"] = unaudited

    # ---- tensors
    indexes = {r: index_checkpoint(d) for r, d in parents.items()}
    ref_idx = indexes[ref]
    tensor_report = {"tensor_count": {r: len(ix) for r, ix in indexes.items()},
                     "dtypes": {r: sorted({t.dtype for t in ix.values()}) for r, ix in indexes.items()},
                     "parameters": {r: int(sum(t.numel for t in ix.values())) for r, ix in indexes.items()}}
    for r, ix in indexes.items():
        if set(ix) != set(ref_idx):
            missing = sorted(set(ref_idx) - set(ix))[:10]
            extra = sorted(set(ix) - set(ref_idx))[:10]
            report["hard_stops"].append(f"{r}: tensor names differ from {ref} (missing {missing}, extra {extra})")
            continue
        for name, t in ix.items():
            rt = ref_idx[name]
            if t.shape != rt.shape or t.dtype != rt.dtype:
                report["hard_stops"].append(f"{r}: {name} is {t.dtype}{list(t.shape)} vs {rt.dtype}{list(rt.shape)}")
    nonfloat = sorted(n for n, t in ref_idx.items() if t.dtype not in FLOAT_DTYPES)
    tensor_report["non_floating_tensors"] = nonfloat
    groups: dict[str, int] = {}
    for n in ref_idx:
        groups[tensor_group(n)] = groups.get(tensor_group(n), 0) + 1
    tensor_report["groups"] = groups
    unclassified = [n for n in ref_idx if tensor_group(n) == "unclassified"]
    if unclassified:
        report["hard_stops"].append(f"unclassified tensors (ownership unknown): {unclassified[:10]}")
    report["tensors"] = tensor_report

    # ---- configs
    cfgs = {r: json.loads((d / "config.json").read_text(encoding="utf-8")) for r, d in parents.items()}
    sigs = {r: architecture_signature(c) for r, c in cfgs.items()}
    arch_diffs = {}
    for r, sig in sigs.items():
        diff = {k: (sigs[ref][k], v) for k, v in sig.items() if v != sigs[ref][k]}
        # every architecture value and token id must agree, a field missing on one side included; only the config eos
        # entry (a stop policy the inference view owns) is compared as an id set and recorded instead
        hard = {k: v for k, v in diff.items() if k != "token.eos_token_id"}
        if hard:
            report["hard_stops"].append(f"{r}: architecture or token ids differ from {ref}: {hard}")
        ref_eos, eos = diff.get("token.eos_token_id", (None, None))
        if _token_ids(ref_eos) != _token_ids(eos):
            report["findings"].append(f"{r}: config eos_token_id {eos} vs {ref_eos} (stop-token policy belongs to the "
                                      "inference view, not to the weights)")
        if diff:
            arch_diffs[r] = {k: [str(a), str(b)] for k, (a, b) in diff.items()}
    report["architecture"] = {"signature": sigs[ref], "differences_vs_reference": arch_diffs,
                              "tie_word_embeddings": {r: _cfg_get(c, None, "tie_word_embeddings") for r, c in cfgs.items()},
                              "declared_dtype": {r: c.get("torch_dtype") or c.get("dtype") for r, c in cfgs.items()}}
    for r, c in cfgs.items():
        tied = _cfg_get(c, None, "tie_word_embeddings")
        names = set(indexes[r])
        if tied and "lm_head.weight" in names:
            report["findings"].append(f"{r}: config ties embeddings but stores lm_head.weight; the merge keeps both and "
                                      "the runtime ties them")
        if not tied and "lm_head.weight" not in names:
            report["hard_stops"].append(f"{r}: untied embeddings but no lm_head.weight")

    # ---- tokenizers
    toks = {r: tokenizer_signature(d) for r, d in parents.items()}
    for r, t in toks.items():
        for k in ("vocab_sha256", "added_tokens", "model_type", "pre_tokenizer", "normalizer", "decoder",
                  "eos_token", "pad_token", "additional_special_tokens"):
            if t[k] != toks[ref][k]:
                report["hard_stops"].append(f"{r}: tokenizer {k} differs from {ref}")
    vocab_rows = ref_idx["model.embed_tokens.weight"].shape[0] if "model.embed_tokens.weight" in ref_idx else None
    report["tokenizer"] = {"signature": {k: v for k, v in toks[ref].items() if k != "added_tokens"},
                           "added_token_count": len(toks[ref]["added_tokens"]),
                           "embedding_rows": vocab_rows,
                           "identical_across_parents": not any("tokenizer" in h for h in report["hard_stops"])}
    if vocab_rows is not None and vocab_rows < toks[ref]["vocab_size_with_added"]:
        report["hard_stops"].append("embedding matrix has fewer rows than tokenizer ids")

    # ---- non-weight files, processors, generation configs, templates (recorded; the inference view is chosen separately)
    files = {r: _files_present(d) for r, d in parents.items()}
    report["non_weight_files"] = files
    gens = {}
    procs = {}
    for r, d in parents.items():
        g = d / "generation_config.json"
        gens[r] = json.loads(g.read_text(encoding="utf-8")) if g.is_file() else None
        p = d / "preprocessor_config.json"
        pc = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
        procs[r] = {"image_processor_type": pc.get("image_processor_type"), **effective_processor(pc)}
    report["generation_configs"] = gens
    report["processors"] = procs
    for r in roles:
        for k in procs[r]:
            if k != "image_processor_type" and procs[r][k] != procs[ref][k]:
                report["hard_stops"].append(f"{r}: image processor {k} differs ({procs[r][k]} vs {procs[ref][k]})")
        if procs[r]["image_processor_type"] != procs[ref]["image_processor_type"]:
            report["findings"].append(
                f"{r}: declares {procs[r]['image_processor_type']} vs {procs[ref]['image_processor_type']}; the pinned "
                "transformers loads the fast Qwen2-VL processor for both (vLLM log), same pixel bounds and normalization")
        eos_r = (gens[r] or {}).get("eos_token_id")
        eos_ref = (gens[ref] or {}).get("eos_token_id")
        if eos_r != eos_ref:
            report["findings"].append(f"{r}: generation eos_token_id {eos_r} vs {eos_ref} (stop-token policy belongs to the "
                                      "inference view, not to the weights)")
    templates = {r: next((n for n in ("chat_template.json", "chat_template.jinja") if files[r].get(n)), None)
                 for r in roles}
    report["chat_templates"] = {r: {"file": templates[r], "sha256": files[r].get(templates[r]) if templates[r] else None}
                                for r in roles}
    report["lineage"] = lineage or {}
    if report["hard_stops"]:
        report["status"] = "incompatible"
    else:
        report["status"] = "compatible"
    return report


def tensor_group(name: str) -> str:
    """Ownership map for Qwen2.5-VL tensors (recorded for diagnostics and any later component experiment)."""

    if name.startswith("visual.merger."):
        return "visual_merger"
    if name.startswith("visual.patch_embed."):
        return "visual_patch_embed"
    if name.startswith("visual.blocks."):
        return "visual_blocks"
    if name == "model.embed_tokens.weight":
        return "language_embeddings"
    if name.startswith("model.layers."):
        return "language_layers"
    if name == "model.norm.weight":
        return "language_final_norm"
    if name == "lm_head.weight":
        return "language_output_head"
    return "unclassified"


def require_compatible(report: dict[str, Any]) -> None:
    if report.get("status") != "compatible":
        raise IncompatibleParents("; ".join(report.get("hard_stops", [])) or "compatibility audit did not pass")
