"""External transfer on trace_eval_v1 (RQ2; DESIGN 6 "External suite", 8 "Statistics", 10 "format shift").

Two stages, because the per-row inputs of eight benchmarks only exist as .xlsx tables (openpyxl lives in the
trace_eval env, numpy/pytest in trace_cert):

    # 1. per-item tables (needs pandas + openpyxl: $TRC_EXT_PYTHON); cached, rebuilt when an input changes
    $TRC_EXT_PYTHON experiments/certificates/analysis/compare_ext.py items --ext-root "$TRC_EXT_ROOT" --out-dir OUT
    # 2. tables, pre-registered external family, descriptive comparisons, format-shift view (numpy only)
    $TRACE_CERT_PYTHON experiments/certificates/analysis/compare_ext.py analyze --ext-root "$TRC_EXT_ROOT" --out-dir OUT

Inputs (read only): ``<ext-root>/runs/<served>/full/summary.seed<N>.json`` (official macros, ``complete`` flag,
extraction and evaluator-exception counts), the run's ``hf_archive`` (per-row scores where the archive has them,
row identities for every benchmark) and, for the eight benchmarks whose archive score is aggregate-only, the
evaluator tables under ``scoring/seed_<N>/`` named by each benchmark's ``scores.json``. Neither
``jobs/ext_eval_lib.py`` nor the trace-rlvr clone is imported; the eight per-row rules below are frozen copies
of the pinned evaluators and every reconstructed benchmark score is checked against the official one.

Per-item values (``items``):

- 16 benchmarks: the archive's per-row ``score`` (0/1, float or bool) keyed by ``source_ordinal``.
- countbenchqa: ``str(answer) in str(prediction)`` on the evaluator input table (VLMEvalKit CountBenchQA.evaluate).
- puzzlevqa / visualpuzzles: ``extract_unambiguous_abcd(prediction)`` against the option letter (VLMEvalKit
  utils/puzzlevqa.py, utils/visualpuzzles.py with TRACE's utils/trace_eval_answer_parsing.py).
- mathvista / mathvision: ``post_check(line, prefetch=False)`` on the judged table's ``res`` (utils/mathvista.py,
  utils/mathv.py; MathVision's latex2sympy fallback is not reproduced, see the validation report).
- chartqapro: ``relaxed_correctness_chartqapro`` (utils/chartqapro.py with the anls 0.0.2 score).
- tablevqabench: the scored table's per-row ``scores``; the official score is the mean of five split values
  (fintabnetqa relieved + exact, vtabfact, vwtq, vwtq_syn), so items carry a split stratum and weight.
- wemath: the official ``Score (Strict)`` = (525 - 0.5 IG - RM_strict - IK) / 525 is a mean over the 525
  two- and three-step problem groups, so the WeMath item is the problem group with value 1 (complete mastery),
  0.5 (inadequate generalization) or 0 (insufficient knowledge, rote memorization); one-step rows do not enter.

Every benchmark score is then ``sum_h W_h * mean_{i in h} v_i`` (W_h = 1 for a single stratum), which is what
the bootstrap resamples.

Statistics (``analyze``; DESIGN 8, conventions of ``compare_cert.py``):

- Pairing: items present in both runs; an item's value is its mean over the decoding seeds both runs have
  (seed-1 G/P/A/S have 42-44 when complete; everything else 42).
- Macro = mean of the benchmark scores of a group (natural 9, chart/diagram/synthetic 15, overall 24,
  categories, MMStar alone). CI: paired bootstrap that resamples items within every benchmark (and within the
  TableVQABench split strata) and recomputes the macro (10,000 resamples, seed 0, percentile 95%). Test: item
  sign-flip permutation of the paired macro difference (10,000 flips, seed 0), the continuous-metric test of
  DESIGN 8 (McNemar does not apply to a benchmark-mean macro).
- Training seeds: matched pairs (P@k with G@k); the estimate is the mean of per-seed differences, the CI a
  two-level bootstrap (training seeds, then items), the sign rule needs max(2, ceil(2K/3)) of K seeds.
- External family (pre-registered, DESIGN 8): P - G on the natural macro and on the chart/diagram/synthetic
  macro, Holm across the two. With K > 1 training seeds the Holm input is the two-level bootstrap p
  (2 * the smaller tail share of resampled means at 0, +1 corrected), and a directional claim needs the
  Holm-adjusted p < alpha and the sign rule; compare_cert's own convention (unadjusted CI excluding 0 plus the
  sign rule; per-seed sign-flip p Holm-adjusted within each training-seed index) is reported next to it.
  With K = 1 the Holm input is the sign-flip p and the CI reflects instance sampling only.
- Secondary: P - A and G - A, same machinery, Holm across their four members. Everything else (S, base,
  legacy, gated, 7B, per benchmark, categories) is descriptive and unadjusted.

Format-shift view (DESIGN 10): per model the deterministic-extraction failure rate and the judge-needed rate
(proxy over the judge-extraction routes and the protocol rate of MathVision/MathVista), certificate-shaped JSON
and length-cap rates, fix-2 ``evaluator_exception_scored_incorrect`` and, when a summary carries them, the
fix-3 counters (``mathverse_all_retries_failed_scored_incorrect``, ``extraction_failed_scored_incorrect``);
a pair of compared models is flagged when a rate differs by more than 2 pp.

Provenance: per complete run (full and q25), the ``final25_code_hash`` of its generation and of the score slices it
reports (newest archive slice per benchmark; superseded slices listed), and the fix-2 / fix-3 counters both from the
summary and from the scorers' own ``scores.json`` (MathVerse ``artifacts.mathverse_all_retries_failed_scored_incorrect``,
MME-Reasoning ``evaluation.extraction_failed_scored_incorrect`` / ``evaluator_exception_scored_incorrect``).

Transfer over training (descriptive; DESIGN 6 "intermediate checkpoints use trace_eval_v1_q25"): q25 macros of
the step-100/200 checkpoints from their own q25 runs (``runs/<served>-{100,200}/q25/``), and of step 300, S
(step 0: RL starts from the M2 SFT checkpoint, DESIGN 7.3) and base from their full seed-42 runs restricted to the
same q25 rows (``subsets/trace_eval_v1_q25.rows.json``; WeMath keeps whole problem groups, as the subset rule does).
Every q25 run's row identities must equal the subset's and every full run's generation must contain them with the
same row hashes. Paired deltas per step (P - G, P - A, arm - S, gated and 7B pairs) use the same machinery as above
at decoding seed 42 and are unadjusted. Missing q25 runs are listed and picked up on the next run.

Training hardware: the GPU class of every RL checkpoint from its run's ``run_receipt.json`` (DESIGN 7.3 nests
hardware in the training seed); comparisons whose two RL runs used different GPU classes (the gated runs against
seed-1 arms) are marked as not matched. Decoding-seed shift (descriptive): official seed-43/44 minus seed-42 scores
per model with several decoding seeds, a shift that matched-seed pairs cancel but single-seed levels carry.

Outputs: ``compare_ext.json`` (every number) and ``compare_ext.md`` (rendered only from that JSON).
"""

from __future__ import annotations

import argparse
import ast
import copy
import gzip
import hashlib
import importlib.util
import json
import math
import os
import re
import string
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

VERSION = "compare_ext-1.2.0"
ITEMS_SCHEMA = "compare_ext-items-v2"

# DESIGN 6: frozen natural vs chart/diagram/synthetic mapping; MMStar also alone (mixed content).
NATURAL_KEYS = ("realworldqa", "mmstar", "embspatial", "cvbench_3d", "erqa", "blink", "countbenchqa", "countqa",
                "treebench")
SYNTHETIC_KEYS = ("chartqapro", "charxivreason", "tablevqabench", "evochart", "mathvision", "mathvista", "mathverse",
                  "wemath", "phyx_mini_mc", "mmmu_pro_vision", "spatialvizbench_cot", "puzzlevqa", "visualpuzzles",
                  "logicvista", "mme_reasoning")
ALL_KEYS = NATURAL_KEYS + SYNTHETIC_KEYS
ALONE_KEYS = ("mmstar",)
GROUPS = {"natural_9": NATURAL_KEYS, "chart_diagram_synthetic_15": SYNTHETIC_KEYS, "overall_24": ALL_KEYS,
          "mmstar": ALONE_KEYS}
# descriptive sensitivity (not pre-registered): drop the two benchmarks whose official option-letter extractor
# (explicit "Answer: X" / one answer block / final box) is the format-shift channel of DESIGN 10
LETTER_KEYS = ("puzzlevqa", "visualpuzzles")
SENS_GROUPS = {"sens:chart_diagram_synthetic_13_excl_puzzle_letters": tuple(k for k in SYNTHETIC_KEYS if k not in LETTER_KEYS),
               "sens:overall_22_excl_puzzle_letters": tuple(k for k in ALL_KEYS if k not in LETTER_KEYS)}
FAMILY_GROUPS = ("natural_9", "chart_diagram_synthetic_15")

# fix-2 (README sec. 11) and fix-3 (2026-10-23, code hash d9319e7db480) counters, looked up anywhere in a summary
EXCEPTION_KEYS = ("evaluator_exception_scored_incorrect", "mathverse_all_retries_failed_scored_incorrect",
                  "extraction_failed_scored_incorrect")
RATE_FLAG_KEYS = ("det_fail_rate", "judge_needed_rate_proxy", "judge_needed_protocol_mathvision",
                  "judge_needed_protocol_mathvista", "abcd_unresolved_puzzlevqa", "abcd_unresolved_visualpuzzles")


def _m(served: str, label: str, arm: str, scale: str, tseed: str, role: str, seeds: Sequence[int]) -> Dict[str, Any]:
    return {"served": served, "label": label, "arm": arm, "scale": scale, "tseed": tseed, "role": role,
            "expected_seeds": list(seeds)}


# Models of the RQ2 matrix (DESIGN 6/7.4 and the stage plan). expected_seeds = decoding seeds pre-registered
# for the external suite (G and P also 43/44 at 3B); extra complete seeds found on disk are used as well.
MODELS: Tuple[Dict[str, Any], ...] = (
    _m("qwen2.5-vl-3b-instruct", "base-3B", "base", "3B", "", "context", (42,)),
    _m("trace-qwen2.5-vl-3b", "legacy-3B", "legacy", "3B", "", "context", (42,)),
    _m("cert-sft-3b-s564", "S", "S", "3B", "", "arm", (42,)),
    *(_m(f"rl3b-a-s{t}-300", f"A@{t}", "A", "3B", str(t), "arm", (42,)) for t in (1, 2, 3)),
    *(_m(f"rl3b-g-s{t}-300", f"G@{t}", "G", "3B", str(t), "arm", (42, 43, 44)) for t in (1, 2, 3)),
    *(_m(f"rl3b-p-s{t}-300", f"P@{t}", "P", "3B", str(t), "arm", (42, 43, 44)) for t in (1, 2, 3)),
    _m("rl3b-pgated-s1-300", "P_gated@1", "P_gated", "3B", "1", "gated", (42,)),
    _m("rl3b-ggated-s1-300", "G_gated@1", "G_gated", "3B", "1", "gated", (42,)),
    _m("qwen2.5-vl-7b-instruct", "base-7B", "base7", "7B", "", "context", (42,)),
    _m("trace-qwen2.5-vl-7b", "legacy-7B", "legacy7", "7B", "", "context", (42,)),
    _m("cert-sft-7b-s564", "S7", "S7", "7B", "", "arm", (42,)),
    _m("rl7b-g-s1-300", "G7@1", "G7", "7B", "1", "arm", (42,)),
    _m("rl7b-p-s1-300", "P7@1", "P7", "7B", "1", "arm", (42,)),
)
MODEL_BY_SERVED = {m["served"]: m for m in MODELS}

# (name, X arm, Y arm, kind); kind family/secondary are the pre-registered families, the rest descriptive.
COMPARISONS: Tuple[Tuple[str, str, str, str], ...] = (
    ("P - G", "P", "G", "family"),
    ("P - A", "P", "A", "secondary"),
    ("G - A", "G", "A", "secondary"),
    ("A - S", "A", "S", "descriptive"),
    ("G - S", "G", "S", "descriptive"),
    ("P - S", "P", "S", "descriptive"),
    ("S - base-3B", "S", "base", "descriptive"),
    ("legacy-3B - base-3B", "legacy", "base", "descriptive"),
    ("P - legacy-3B", "P", "legacy", "descriptive"),
    ("G - legacy-3B", "G", "legacy", "descriptive"),
    ("P_gated - G_gated", "P_gated", "G_gated", "descriptive"),
    ("P_gated - P", "P_gated", "P", "descriptive"),
    ("P_gated - G", "P_gated", "G", "descriptive"),
    ("G_gated - G", "G_gated", "G", "descriptive"),
    ("P7 - G7", "P7", "G7", "descriptive"),
    ("G7 - S7", "G7", "S7", "descriptive"),
    ("P7 - S7", "P7", "S7", "descriptive"),
    ("S7 - base-7B", "S7", "base7", "descriptive"),
    ("legacy-7B - base-7B", "legacy7", "base7", "descriptive"),
)

# ---------------------------------------------------------------------------- transfer over training (q25)
Q25 = "q25"
Q25_SUBSET_FILE = "subsets/trace_eval_v1_q25.rows.json"
Q25_STEPS = (100, 200, 300)  # 100/200: own q25 runs; 300: the full run restricted to the q25 rows
Q25_DECODING_SEED = 42       # the only decoding seed of the q25 checkpoint runs (DESIGN 6 / 7.4)
# (arm, served code, served prefix, scale, training seeds); start checkpoint (step 0) per scale
Q25_ARMS = (("A", "a", "rl3b", "3B", (1, 2, 3)), ("G", "g", "rl3b", "3B", (1, 2, 3)), ("P", "p", "rl3b", "3B", (1, 2, 3)),
            ("P_gated", "pgated", "rl3b", "3B", (1,)), ("G_gated", "ggated", "rl3b", "3B", (1,)),
            ("G7", "g", "rl7b", "7B", (1,)), ("P7", "p", "rl7b", "7B", (1,)))
Q25_START = {"3B": "S", "7B": "S7"}       # RL starts from the M2 SFT checkpoint (DESIGN 7.3)
Q25_CONTEXT = ("S", "S7", "base", "base7")  # full seed-42 runs restricted to the q25 rows
Q25_PROBE_PREFIX = "q25run:"                # store key of a context model's own q25 run (the 2026-10-19 probes)
# descriptive paired deltas per step; S / S7 are step 0
Q25_PAIRS = (("P - G", "P", "G"), ("P - A", "P", "A"), ("G - A", "G", "A"), ("A - S", "A", "S"), ("G - S", "G", "S"),
             ("P - S", "P", "S"),
             ("P_gated - G_gated", "P_gated", "G_gated"), ("P7 - G7", "P7", "G7"), ("G7 - S7", "G7", "S7"),
             ("P7 - S7", "P7", "S7"))
Q25_GROUPS = ("natural_9", "chart_diagram_synthetic_15", "overall_24", "mmstar",
              "sens:chart_diagram_synthetic_13_excl_puzzle_letters")


def training_hardware(summary_paths: Iterable[str]) -> Dict[str, Optional[str]]:
    """GPU class an RL checkpoint was trained on, from its run's ``run_receipt.json`` (``<run>/hf/global_step_N``
    is the summary's ``model``; ``gpu.name`` x number of GPU uuids of every attempt that trained). DESIGN 7.3 nests
    hardware in the training seed, so a pair of RL runs on different GPU classes is not a matched pair.
    None = not an RL run / no receipt."""
    out: Dict[str, Optional[str]] = {}
    for sp in summary_paths:
        try:
            s = _read_json(Path(sp))
        except (OSError, ValueError):
            continue
        served = s.get("served") or Path(sp).parent.parent.name
        if served in out or not str(served).startswith("rl"):
            continue
        model = Path(str(s.get("model") or ""))
        rec = model.parent.parent / "run_receipt.json" if model.parent.name == "hf" else None
        if rec is None or not rec.is_file():
            out[served] = None
            continue
        r = _read_json(rec)
        classes = sorted({f"{(a.get('gpu') or {}).get('name')} x{len((a.get('gpu') or {}).get('uuids') or [])}"
                          for a in r.get("attempts", [])
                          if (a.get("gpu") or {}).get("name")
                          and (a.get("last_step") or 0) > (a.get("resumed_from_step") or 0)})
        out[served] = " + ".join(classes) if classes else None
    return out


def annotate_hardware(comps: Sequence[Dict[str, Any]], hw: Mapping[str, Optional[str]]) -> None:
    """Training GPU class of both sides of every training-seed pair; ``hardware_crossed`` lists the pairs whose
    RL runs used different GPU classes (not matched in the DESIGN 7.3 sense; descriptive comparisons only)."""
    for c in comps:
        crossed = []
        for ps in c.get("per_seed", []):
            hx, hy = hw.get(ps["x_served"]), hw.get(ps["y_served"])
            ps["hardware"] = [hx, hy]
            if hx and hy and hx != hy:
                crossed.append({"tseeds": ps["tseeds"], "x": hx, "y": hy})
        c["hardware_crossed"] = crossed


def q25_checkpoints() -> List[Dict[str, Any]]:
    """Every RL checkpoint of the transfer-over-training matrix (steps 100/200 from q25 runs, 300 restricted)."""
    out = []
    for arm, code, prefix, scale, tseeds in Q25_ARMS:
        for t in tseeds:
            for step in Q25_STEPS:
                served = f"{prefix}-{code}-s{t}-{step}"
                out.append({"key": served, "served": served, "label": f"{arm}@{t}", "arm": arm, "scale": scale,
                            "tseed": str(t), "step": step,
                            "source": "full run restricted to q25 rows" if step == 300 else "q25 run"})
    return out


def q25_models(store_keys: Iterable[str]) -> List[Dict[str, Any]]:
    """Model rows for ``run_comparison`` on the q25 store: arm names carry the step (``P@100``, ``S@0``)."""
    have = set(store_keys)
    out = []
    for c in q25_checkpoints():
        out.append({"served": c["key"], "arm": f"{c['arm']}@{c['step']}", "tseed": c["tseed"],
                    "label": f"{c['label']} step {c['step']}"})
    for m in MODELS:
        if m["arm"] in Q25_CONTEXT:
            step = 0 if m["arm"] in Q25_START.values() else None
            out.append({"served": m["served"], "arm": f"{m['arm']}@{step if step is not None else 'ctx'}",
                        "tseed": "", "label": m["label"]})
    return [m for m in out if m["served"] in have] + [m for m in out if m["served"] not in have]


class ExtError(ValueError):
    """Inconsistent or missing inputs (never silently turned into a result)."""


# ============================================================================== small io helpers


def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _write_json(path: Path, obj: Any, gz: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    data = json.dumps(obj, indent=None if gz else 1, sort_keys=False, allow_nan=False).encode("utf-8")
    if gz:
        with gzip.GzipFile(tmp, "wb", mtime=0) as f:
            f.write(data)
    else:
        tmp.write_bytes(data + b"\n")
    os.replace(tmp, path)


def _read_items(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def _finite(x: Any) -> Optional[float]:
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _fingerprint(paths: Iterable[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted({str(p) for p in paths}):
        st = os.stat(p)
        h.update(f"{p}\t{st.st_size}\t{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


# ============================================================================== inventory


def discover(ext_root: Path) -> Dict[str, Any]:
    """Every summary on disk plus the expected matrix: complete runs, incomplete ones, missing ones."""
    runs_dir = ext_root / "runs"
    found: List[Dict[str, Any]] = []
    for sdir in sorted(runs_dir.glob("*/full")):
        served = sdir.parent.name
        for sp in sorted(sdir.glob("summary.seed*.json")):
            m = re.fullmatch(r"summary\.seed(\d+)\.json", sp.name)
            if not m:
                continue
            s = _read_json(sp)
            found.append({"served": served, "seed": int(m.group(1)), "summary_path": str(sp),
                          "complete": bool(s.get("complete")), "benchmarks_scored": s.get("benchmarks_scored"),
                          "created": s.get("created"), "known_model": served in MODEL_BY_SERVED})
    have = {(r["served"], r["seed"]): r for r in found}
    pending = []
    for m in MODELS:
        for seed in m["expected_seeds"]:
            r = have.get((m["served"], seed))
            if r is None:
                pending.append({"served": m["served"], "label": m["label"], "seed": seed, "status": "missing"})
            elif not r["complete"]:
                pending.append({"served": m["served"], "label": m["label"], "seed": seed,
                                "status": f"incomplete ({r['benchmarks_scored']}/24 benchmarks scored)"})
    for r in found:  # non-expected seeds that exist but are incomplete
        m = MODEL_BY_SERVED.get(r["served"])
        if not r["complete"] and (m is None or r["seed"] not in m["expected_seeds"]):
            pending.append({"served": r["served"], "label": (m or {}).get("label", r["served"]), "seed": r["seed"],
                            "status": f"incomplete ({r['benchmarks_scored']}/24 benchmarks scored), not pre-registered"})
    return {"runs": found, "pending": pending}


def discover_q25(ext_root: Path) -> Dict[str, Any]:
    """q25 summaries on disk plus the expected step-100/200 checkpoint runs: complete, in progress, missing."""
    runs_dir = ext_root / "runs"
    found: List[Dict[str, Any]] = []
    for sdir in sorted(runs_dir.glob(f"*/{Q25}")):
        served = sdir.parent.name
        for sp in sorted(sdir.glob("summary.seed*.json")):
            m = re.fullmatch(r"summary\.seed(\d+)\.json", sp.name)
            if not m:
                continue
            s = _read_json(sp)
            found.append({"served": served, "seed": int(m.group(1)), "summary_path": str(sp), "subset": Q25,
                          "complete": bool(s.get("complete")), "benchmarks_scored": s.get("benchmarks_scored"),
                          "created": s.get("created"), "row_list_sha256": s.get("row_list_sha256")})
    have = {(r["served"], r["seed"]): r for r in found}
    expected, pending = [], []
    for c in q25_checkpoints():
        if c["step"] == 300:
            continue
        expected.append(c["served"])
        r = have.get((c["served"], Q25_DECODING_SEED))
        if r is None:
            d = runs_dir / c["served"] / Q25
            status = "in progress (run directory without a summary)" if d.is_dir() else "missing"
        elif not r["complete"]:
            status = f"incomplete ({r['benchmarks_scored']}/24 benchmarks scored)"
        else:
            continue
        pending.append({"served": c["served"], "label": f"{c['label']} step {c['step']}", "seed": Q25_DECODING_SEED,
                        "status": status})
    return {"runs": found, "pending": pending, "expected": expected}


# ============================================================================== frozen per-row rules
# Each is a copy of the pinned evaluator (VLMEvalKit a8b12bf with the TRACE af32fe0 patches) restricted to what
# decides one row; ``items`` checks every reconstructed benchmark score against the official one.

_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.I | re.S)
_OFFICIAL_OPTION_RE = re.compile(r"\banswer\s*:\s*([A-D])\b", flags=re.I)
_ANSWER_LINE_RE = re.compile(r"^\s*answer\s*:\s*(.*?)\s*$", flags=re.I | re.M)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _last_boxed(text: str) -> Optional[str]:
    values: List[str] = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start = match.end()
        depth = 1
        pos = start
        while pos < len(text) and depth:
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
            pos += 1
        if depth == 0:
            values.append(text[start: pos - 1])
    return values[-1] if values else None


def _atomic_abcd(value: Any) -> Optional[str]:
    text = _text(value).strip()
    answer = re.fullmatch(r"answer\s*:\s*(.*)", text, flags=re.I | re.S)
    if answer:
        text = answer.group(1).strip()
    while text:
        previous = text
        wrappers = (
            r"\\\(\s*(.*?)\s*\\\)",
            r"\\\[\s*(.*?)\s*\\\]",
            r"\$\s*(.*?)\s*\$",
            r"\\(?:text|mathrm|mathbf)\s*\{\s*(.*?)\s*\}",
            r"[\(\[\{]\s*(.*?)\s*[\)\]\}]",
        )
        for pattern in wrappers:
            wrapped = re.fullmatch(pattern, text, flags=re.I | re.S)
            if wrapped:
                text = wrapped.group(1).strip()
                break
        if text == previous:
            break
    return text.upper() if re.fullmatch(r"[A-D]", text, flags=re.I) else None


def extract_unambiguous_abcd(value: Any) -> str:
    """utils/trace_eval_answer_parsing.py: explicit ``Answer: X``, one answer block, final box; else ``Z``."""
    text = _text(value)
    candidates = [match.group(1).upper() for match in _OFFICIAL_OPTION_RE.finditer(text)]
    tags = list(_ANSWER_TAG_RE.finditer(text))
    if len(tags) == 1:
        tagged = tags[0].group(1).strip()
        tagged_box = _last_boxed(tagged)
        tagged_candidate = _atomic_abcd(tagged_box) if tagged_box is not None else _atomic_abcd(tagged)
        if tagged_candidate is None:
            answer_lines = list(_ANSWER_LINE_RE.finditer(tagged))
            if answer_lines:
                tagged_candidate = _atomic_abcd(answer_lines[-1].group(1))
        if tagged_candidate is not None:
            candidates.append(tagged_candidate)
    boxed = _last_boxed(text)
    boxed_candidate = _atomic_abcd(boxed) if boxed is not None else None
    if boxed_candidate is not None:
        candidates.append(boxed_candidate)
    distinct = list(dict.fromkeys(candidates))
    return distinct[0] if len(distinct) == 1 else "Z"


def puzzlevqa_hit(prediction: Any, options: Sequence[Any], answer: Any) -> int:
    """utils/puzzlevqa.py PuzzleVQA_acc: extracted letter vs the letter of the answer among the options."""
    answer_str = str(answer).strip()
    letter = None
    for i, opt in enumerate(options):
        if str(opt).strip() == answer_str:
            letter = chr(ord("A") + i)
            break
    if letter is None:
        raise ExtError(f"PuzzleVQA answer {answer!r} not among options {options!r}")
    return int(extract_unambiguous_abcd(prediction).lower() == letter.lower())


def visualpuzzles_hit(prediction: Any, answer: str) -> int:
    """utils/visualpuzzles.py VisulPuzzles_acc."""
    return int(extract_unambiguous_abcd(prediction).lower() == answer.lower())


def countbenchqa_hit(prediction: Any, answer: Any) -> int:
    """CountBenchQA.evaluate (image_vqa.py): the answer string occurs in the prediction string."""
    return int(str(answer) in str(prediction))


_VERBOSE_ANSWER_RE = re.compile(r"(?i)(?:correct\s+)?answer\s+is\s+\**([ABCD])\**")


def can_infer_option(answer: str, choices: Mapping[str, Any]) -> Any:
    """vlmeval/utils/matching_util.py (TRACE-patched; VERBOSE unset)."""
    if "Failed to obtain answer via API" in answer:
        return False
    for err in ("Sorry, I can't help with images of people yet.", "I can't process this file.",
                "I'm sorry, but without the image provided", "Cannot determine the answer"):
        if err in answer:
            return "Z"

    def count_choice(splits, chs, prefix="", suffix=""):
        return sum(1 for c in chs if prefix + c + suffix in splits)

    answer_mod = copy.copy(answer)
    for c in ".()[],:;!*#{}":
        answer_mod = answer_mod.replace(c, " ")
    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)
    if count == 1:
        for ch in choices:
            if ch in splits and splits.index(ch) > (len(splits) - 5):
                return ch
    elif count == 0 and count_choice(splits, {"Z", ""}) == 1:
        return "Z"
    match = _VERBOSE_ANSWER_RE.search(answer or "")
    if match and match.group(1).upper() in choices:
        return match.group(1).upper()
    return False


def can_infer_text(answer: str, choices: Dict[str, Any]) -> Any:
    answer = answer.lower()
    if len(answer) > 2 * sum(len(str(v)) for v in choices.values()):
        return False
    for k in choices:
        if k not in string.ascii_uppercase:
            raise ExtError(f"choice key {k!r}")
        choices[k] = str(choices[k]).lower()
    cands = [k for k in choices if choices[k] in answer]
    return cands[0] if len(cands) == 1 else False


def can_infer(answer: Any, choices: Dict[str, Any]) -> Any:
    answer = str(answer)
    copt = can_infer_option(answer, choices)
    return copt if copt else can_infer_text(answer, choices)


def _list_to_dict(lst: Sequence[Any]) -> Dict[str, Any]:
    return {chr(65 + i): val for i, val in enumerate(lst)}


def mathvista_hit(row: Mapping[str, Any]) -> int:
    """utils/mathvista.py post_check(line, prefetch=False) on the judged table (``res`` = judge extraction)."""
    res: Any = None
    ans: Any = row["answer"]
    response = row["res"]
    try:
        if row["question_type"] == "multi_choice":
            ans = row["answer_option"]
            choices = _list_to_dict(ast.literal_eval(row["choices"]))
            res = can_infer(response, choices)
        else:
            if row["answer_type"] == "integer":
                res = int(response)
                ans = int(row["answer"])
            elif row["answer_type"] == "float":
                res = float(response)
                ans = float(row["answer"])
            else:
                res = str(res)
                ans = str(ans)
    except ValueError:
        pass
    return int(res == ans)


_SAFE_BINOPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
                ast.Div: lambda a, b: a / b, ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b,
                ast.Pow: lambda a, b: a ** b, ast.BitXor: lambda a, b: a ^ b, ast.BitOr: lambda a, b: a | b,
                ast.BitAnd: lambda a, b: a & b, ast.RShift: lambda a, b: a >> b, ast.LShift: lambda a, b: a << b}
_SAFE_UNARY = {ast.UAdd: lambda a: +a, ast.USub: lambda a: -a, ast.Not: lambda a: not a, ast.Invert: lambda a: ~a}
_SAFE_CMP = {ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b, ast.Gt: lambda a, b: a > b,
             ast.GtE: lambda a, b: a >= b, ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b}
_NAME_HITS = {"used": False}
_SAFE_CALLS = {"abs": abs, "round": round, "min": min, "max": max, "pow": pow, "int": int, "float": float}


def _safe_eval(text: str, names: Optional[Mapping[str, Any]] = None) -> Any:
    """Python ``eval`` restricted to literal arithmetic plus the given names.

    The official ``is_equal`` calls bare ``eval`` inside the function, so the function's locals at that moment
    (``asw``, ``gt_asw`` and, once assigned, ``a`` and ``b``) resolve; e.g. the response ``a`` evaluates to the
    already-parsed ground truth and counts as equal. Unknown names, attributes and oversized powers raise, as the
    official eval raises NameError on names absent from mathv.py's namespace."""
    names = names or {}
    node = ast.parse(text, mode="eval")

    def ev(n: ast.AST) -> Any:
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, (ast.Tuple, ast.List)):
            vals = [ev(e) for e in n.elts]
            return tuple(vals) if isinstance(n, ast.Tuple) else vals
        if isinstance(n, ast.BinOp) and type(n.op) in _SAFE_BINOPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, (ast.Pow, ast.LShift)) and isinstance(b, (int, float)) and abs(b) > 100000:
                raise OverflowError("power too large")
            return _SAFE_BINOPS[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _SAFE_UNARY:
            return _SAFE_UNARY[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Compare) and all(type(o) in _SAFE_CMP for o in n.ops):
            left = ev(n.left)
            for op, comp in zip(n.ops, n.comparators):
                right = ev(comp)
                if not _SAFE_CMP[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(n, ast.Name):
            if n.id in names:
                _NAME_HITS["used"] = True
                return names[n.id]
            if n.id in _SAFE_CALLS:
                return _SAFE_CALLS[n.id]
            raise NameError(n.id)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _SAFE_CALLS and not n.keywords \
                and n.func.id not in names:
            return _SAFE_CALLS[n.func.id](*[ev(a) for a in n.args])
        raise NameError(type(n).__name__)

    return ev(node)


_LATEX: Dict[str, Any] = {"fn": None, "cache": {}, "source": None}


def use_latex2sympy(vendor_dir: Optional[str]) -> Optional[str]:
    """Load latex2sympy2_extended (the MathVision fallback) from a read-only copy of the scoring env's package.

    The pinned scorer imports it from the trace-rlvr clone's ``.work/eval_deps``; run.sh copies that package (and
    its antlr4 runtime) into ``<out>/vendor`` so nothing is imported from, or written into, the clone."""
    if not vendor_dir:
        return None
    sys.dont_write_bytecode = True
    if vendor_dir not in sys.path:
        sys.path.insert(0, vendor_dir)
    from latex2sympy2_extended import latex2sympy  # noqa: E402  (vendored copy)

    _LATEX["fn"] = latex2sympy
    _LATEX["source"] = str(Path(sys.modules["latex2sympy2_extended"].__file__).parent)
    return _LATEX["source"]


def _latex(text: str) -> Tuple[bool, Any]:
    cache = _LATEX["cache"]
    if text not in cache:
        try:
            cache[text] = (True, _LATEX["fn"](text))
        except Exception as exc:  # noqa: BLE001 - the official fallback swallows every error
            cache[text] = (False, exc)
    return cache[text]


def mathv_is_equal(asw: Any, gt_asw: Any) -> Tuple[bool, bool]:
    """utils/mathv.py is_equal. Returns (equal, latex_fallback_unevaluated).

    The official ``eval`` runs in mathv.py's module namespace; here a literal-arithmetic evaluator stands in
    (sympy's printed forms with any name, e.g. ``sqrt(3)`` or ``pi``, raise NameError there as well). The
    latex2sympy fallback runs when the vendored package is loaded (``use_latex2sympy``)."""
    asw = str(asw).lower().strip()
    gt_asw = str(gt_asw).lower().strip()
    if gt_asw == asw:
        return True, False
    scope: Dict[str, Any] = {"asw": asw, "gt_asw": gt_asw}  # the function's locals, as bare eval sees them
    try:
        a = _safe_eval(gt_asw, scope)
        scope["a"] = a
        b = _safe_eval(asw, scope)
        scope["b"] = b
        if abs(a - b) < 1e-6:
            return True, False
    except Exception:
        pass
    if _LATEX["fn"] is None:
        return False, True
    try:
        ok_a, a = _latex(gt_asw)
        if not ok_a:
            raise a
        scope["a"] = a
        ok_b, b = _latex(asw)
        if not ok_b:
            raise b
        scope["b"] = b
        if abs(_safe_eval(str(a), scope) - _safe_eval(str(b), scope)) < 1e-6:
            return True, False
        if abs(a - b) < 1e-6:
            return True, False
    except Exception:
        pass
    return False, False


def mathvision_hit(row: Mapping[str, Any]) -> Tuple[int, bool]:
    """utils/mathv.py post_check(line, prefetch=False); second value: the latex fallback was not evaluated
    and could have changed the outcome."""
    res: Any = None
    ans: Any = row["answer"]
    response = row["res"]
    try:
        if len(ast.literal_eval(row["choices"])) > 0:
            ans = row["answer"]
            choices = _list_to_dict(ast.literal_eval(row["choices"]))
            res = can_infer(response, choices)
        else:
            res = str(response)
            ans = str(ans)
    except ValueError:
        pass
    _NAME_HITS["used"] = False
    try:
        eq, fallback = mathv_is_equal(res, ans)
    except Exception:
        return 0, False
    if eq and _NAME_HITS["used"]:
        _NAME_HITS["count"] = _NAME_HITS.get("count", 0) + 1
    if not fallback:
        return int(eq), False
    gt = str(ans).lower().strip()
    # latex2sympy can only change the outcome when one side looks like latex or a non-literal expression
    unresolved = (not eq) and fallback and bool(re.search(r"[\\{}^]|sqrt|pi|frac", gt + " " + str(res).lower()))
    return int(eq), unresolved


def _levenshtein(s1: str, s2: str) -> int:
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = list(range(len(s1) + 1))
    for i2, c2 in enumerate(s2):
        dists = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                dists.append(distances[i1])
            else:
                dists.append(1 + min((distances[i1], distances[i1 + 1], dists[-1])))
        distances = dists
    return distances[-1]


def anls_score(prediction: str, gold_labels: Sequence[str], threshold: float = 0.5) -> float:
    """anls 0.0.2 (anls.metrics.score.anls_score)."""
    y_pred = " ".join(prediction.strip().lower().split())
    scores = []
    for gold in gold_labels:
        y_true = " ".join(gold.strip().lower().split())
        dist = _levenshtein(y_pred, y_true)
        length = max(len(y_pred.upper()), len(y_true.upper()))
        nl = 0.0 if length == 0 else dist / length
        scores.append(1 - nl if nl < threshold else 0.0)
    return max(scores)


def _fix_list_format(item: Any) -> Any:
    if not isinstance(item, str):
        return item
    match = re.match(r"^\[(.*)\]$", item.strip())
    if not match:
        return item
    content = match.group(1)
    corrected = re.sub(r"(?<!['\w])(\w[^,]*?)(?!['\w])", r"'\1'", content)
    try:
        return ast.literal_eval(f"[{corrected}]")
    except (SyntaxError, ValueError):
        return item


def _parse_to_list(text: Any) -> Optional[List[str]]:
    if not isinstance(text, str):
        return None
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return None
    if isinstance(parsed, list):
        return [str(x).strip(" '") for x in parsed]
    return None


def _to_float(text: str) -> Optional[float]:
    try:
        return float(text.strip().strip("%"))
    except ValueError:
        return None


def _chartqapro_single(target: str, prediction: str, max_relative_change: float = 0.05) -> float:
    t = target.strip().strip("%").strip()
    p = prediction.strip().strip("%").strip()
    t_f, p_f = _to_float(t), _to_float(p)
    if t_f is not None and p_f is not None:
        if t_f == 0.0:
            return 1.0 if p_f == 0.0 else 0.0
        change = abs(p_f - t_f) / abs(t_f)
        return 1.0 if change <= max_relative_change else 0.0
    return anls_score(prediction=p.lower(), gold_labels=[t.lower()], threshold=0.5)


def chartqapro_score(answer_list: Sequence[Any], question_type: str, year_flags: Sequence[str], prediction: Any) -> float:
    """utils/chartqapro.py evaluate_predictions_chartqapro for one row (always_use_exact_match stays False, as in
    the pinned code, which computes but never passes it)."""
    target = answer_list[-1].strip(".").strip("\n")
    pred = str(prediction).strip(".").strip("\n")
    flags = list(year_flags)
    if question_type == "Conversational":
        flags = flags[-1:]
    fixed_t = _fix_list_format(target)
    t_list = _parse_to_list(str(fixed_t)) or [str(target)]
    p_list = _parse_to_list(str(pred)) or [str(pred)]
    n = len(t_list)
    if flags is not None and len(flags) < n:
        flags = flags * n
    scores: List[float] = []
    for idx in range(max(len(t_list), len(p_list))):
        if idx >= len(t_list) or idx >= len(p_list):
            scores.append(0.0)
            continue
        t_item, p_item, flag = t_list[idx], p_list[idx], flags[idx]
        if flag.upper() == "YES":
            scores.append(1.0 if t_item.strip().lower() == p_item.strip().lower() else 0.0)
        else:
            scores.append(_chartqapro_single(t_item, p_item))
    return sum(scores) / len(scores) if scores else 0.0


def wemath_groups(rows: Sequence[Mapping[str, Any]], total: int = 525) -> Tuple[Dict[str, float], Dict[str, int]]:
    """utils/wemath.py four-dimensional metrics as per-group values: rows carry ``ID``, ``key``, ``hit``.

    A group is one multi-step problem (merge of ``<k>steps_1..k`` and ``<k>steps_multi`` on ``ID``). Strict:
    CM = all steps and the composite right (value 1), IG = all steps right and the composite wrong (0.5),
    IK = a step wrong and the composite wrong (0), RM = a step wrong and the composite right (0). The official
    Score (Strict) = (total - 0.5 IG - RM - IK) / total equals the mean of these values when the groups number
    ``total``."""
    by: Dict[Tuple[int, str], Dict[str, bool]] = defaultdict(dict)
    for r in rows:
        key = str(r["key"])
        m = re.fullmatch(r"([23])steps_(\d+|multi)", key)
        if not m:
            continue
        steps, part = int(m.group(1)), m.group(2)
        slot = by[(steps, str(r["ID"]))]
        if part in slot:
            raise ExtError(f"WeMath group {(steps, r['ID'])} has {key} twice")
        slot[part] = bool(int(r["hit"]))
    values: Dict[str, float] = {}
    counts = Counter()
    for (steps, gid), slot in sorted(by.items()):
        parts = [str(i) for i in range(1, steps + 1)]
        if "1" not in slot:
            continue  # the official left merge starts from step 1
        if any(p not in slot for p in parts) or "multi" not in slot:
            raise ExtError(f"WeMath group {(steps, gid)} incomplete: {sorted(slot)}")
        subs = all(slot[p] for p in parts)
        multi = slot["multi"]
        if subs and multi:
            values[f"{steps}:{gid}"] = 1.0
            counts["CM_strict"] += 1
        elif subs and not multi:
            values[f"{steps}:{gid}"] = 0.5
            counts["IG"] += 1
        elif multi:
            values[f"{steps}:{gid}"] = 0.0
            counts["RM_strict"] += 1
        else:
            values[f"{steps}:{gid}"] = 0.0
            counts["IK"] += 1
    counts["groups"] = len(values)
    counts["score_strict_x100"] = (total - 0.5 * counts["IG"] - counts["RM_strict"] - counts["IK"]) / total * 100
    return values, dict(counts)


# ============================================================================== stage 1: per-item tables


def _archive_index(run_root: Path) -> Dict[Tuple[str, int, str], List[Dict[str, Any]]]:
    idx: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = defaultdict(list)
    for f in sorted((run_root / "hf_archive" / "ready").glob("*.ready.json")):
        r = _read_json(f)
        i = r["identity"]
        idx[(i["benchmark"], int(i["seed"]), i["stage"])].append(
            {"ready": str(f), "payload": str(run_root / "hf_archive" / r["payload_path"]), "created": r.get("created_at"),
             "code_hash": (r.get("provenance") or {}).get("final25_code_hash"), "rows": r.get("rows")})
    for v in idx.values():
        v.sort(key=lambda e: e["created"] or "", reverse=True)
    return idx


def _row_digest(pairs: Iterable[Tuple[int, str]]) -> str:
    h = hashlib.sha256()
    for o, rh in sorted(pairs):
        h.update(f"{o}:{rh}\n".encode())
    return h.hexdigest()


def _generation_identity(arch: Mapping, bench: str, seed: int
                         ) -> Tuple[Dict[int, str], str, str, Dict[Tuple[str, str], Optional[int]]]:
    """Generation rows: {source ordinal: row hash}, their digest, the payload, and {(source_index, row hash):
    ordinal} (None when that key is not unique) used to re-key archive rows that carry another ordinal."""
    cands = arch.get((bench, seed, "generation")) or []
    if not cands:
        raise ExtError(f"no generation archive for {bench} seed {seed}")
    pairs: Dict[int, str] = {}
    keys: Dict[Tuple[str, str], Optional[int]] = {}
    for r in _iter_jsonl(Path(cands[0]["payload"])):
        o, h = int(r["source_ordinal"]), str(r["source_row_hash"])
        pairs[o] = h
        k = (str(r.get("source_index")), h)
        keys[k] = None if k in keys else o
    return pairs, _row_digest(pairs.items()), cands[0]["payload"], keys


def _archive_rows(payload: str, gen_pairs: Mapping[int, str], gen_keys: Mapping[Tuple[str, str], Optional[int]],
                  stats: Dict[str, Any]) -> Iterable[Tuple[int, Dict[str, Any]]]:
    """Archive extraction/score rows keyed by the generation's source ordinal.

    Full runs carry the generation ordinal everywhere. In q25 runs the extraction and score payloads of the
    deterministic VLMEvalKit route number rows by their position in the subset (0..n-1) while keeping the true
    ``source_index`` and ``source_row_hash``; such rows are re-keyed through (source_index, row hash), and the
    re-keying is recorded (``stats``)."""
    pos = sorted(gen_pairs)
    for r in _iter_jsonl(Path(payload)):
        o, h = int(r["source_ordinal"]), str(r["source_row_hash"])
        if gen_pairs.get(o) == h:
            yield o, r
            continue
        g = gen_keys.get((str(r.get("source_index")), h))
        if g is None:
            raise ExtError(f"{payload}: row (ordinal {o}, index {r.get('source_index')}) not in the generation archive")
        stats["rekeyed"] = stats.get("rekeyed", 0) + 1
        stats["rekeyed_as_position"] = stats.get("rekeyed_as_position", True) and 0 <= o < len(pos) and pos[o] == g
        yield g, r


def _read_xlsx(path: str):
    import pandas as pd  # trace_eval env (openpyxl); the official scorers read the same tables with pd.read_excel

    return pd.read_excel(path)


def _per_row_from_archive(arch: Mapping, bench: str, seed: int, official: float, gen_pairs: Mapping[int, str],
                          gen_keys: Mapping[Tuple[str, str], Optional[int]]) -> Dict[str, Any]:
    cands = arch.get((bench, seed, "score")) or []
    tried = []
    for c in cands:
        ids, vals, excluded, bad = [], [], 0, 0
        stats: Dict[str, Any] = {}
        for o, r in _archive_rows(c["payload"], gen_pairs, gen_keys, stats):
            if r.get("excluded") and str(r.get("excluded")).lower() not in ("false", "0"):
                excluded += 1
                continue
            v = r.get("score")
            if isinstance(v, bool):
                v = float(v)
            v = _finite(v)
            if v is None:
                bad += 1
                continue
            ids.append(o)
            vals.append(v)
        if bad or not vals:
            tried.append({"payload": c["payload"], "problem": f"{bad} rows without a numeric score"})
            continue
        recon = 100.0 * sum(vals) / len(vals)
        entry = {"method": "archive_per_row", "ids": ids, "values": vals, "inputs": [c["payload"]],
                 "excluded_rows": excluded, "reconstructed": recon, "candidates": len(cands),
                 "archive_rows_rekeyed": stats.get("rekeyed", 0),
                 "archive_rekeyed_as_position": stats.get("rekeyed_as_position") if stats.get("rekeyed") else None}
        if abs(recon - official) <= 1e-6 * max(1.0, abs(official)) or c is cands[-1]:
            entry["tried"] = tried
            return entry
        tried.append({"payload": c["payload"], "problem": f"reconstructed {recon} != official {official}"})
    raise ExtError(f"{bench} seed {seed}: no per-row score payload ({tried})")


def _scores_json(bench_entry: Mapping[str, Any]) -> Dict[str, Any]:
    p = bench_entry.get("scores_path")
    if not p or not Path(p).is_file():
        raise ExtError(f"{bench_entry.get('key')}: scores.json missing ({p})")
    return _read_json(Path(p))


def _sub_checks(expected: Mapping[str, Any], got: Mapping[str, Any], tol: float = 1e-9) -> Dict[str, Any]:
    bad = {k: {"official": expected[k], "reconstructed": got.get(k)} for k in expected
           if got.get(k) is None or abs(float(got[k]) - float(expected[k])) > tol}
    return {"n": len(expected), "ok": len(expected) - len(bad), "mismatch": bad}


def _bench_countbenchqa(sj: Mapping[str, Any]) -> Dict[str, Any]:
    path = sj["artifacts"]["prediction_table"]
    df = _read_xlsx(path).sort_values(by="index")
    ids = [int(x) for x in df["source_ordinal"]]
    vals = [float(countbenchqa_hit(p, a)) for p, a in zip(df["prediction"], df["answer"])]
    return {"method": "countbenchqa_contains", "ids": ids, "values": vals, "inputs": [path]}


def _bench_puzzlevqa(sj: Mapping[str, Any]) -> Dict[str, Any]:
    path = sj["artifacts"]["prediction_table"]
    df = _read_xlsx(path)
    ids, vals = [], []
    hit, tot = Counter(), Counter()
    unresolved = 0
    for r in df.to_dict(orient="records"):
        v = puzzlevqa_hit(r["prediction"], ast.literal_eval(r["options"]), r["answer"])
        unresolved += int(extract_unambiguous_abcd(r["prediction"]) == "Z")
        ids.append(int(r["source_ordinal"]))
        vals.append(float(v))
        for k in ("overall", r["category"]):
            tot[k] += 1
            hit[k] += v
    exp = {f"{row['category']}": row["hit"] for row in sj["scores"]["table"]}
    return {"method": "puzzlevqa_abcd", "ids": ids, "values": vals, "inputs": [path],
            "extractor_unresolved_rate": unresolved / len(ids), "sub_checks": _sub_checks(exp, dict(hit))}


def _bench_visualpuzzles(sj: Mapping[str, Any]) -> Dict[str, Any]:
    path = sj["artifacts"]["prediction_table"]
    df = _read_xlsx(path)
    ids, vals = [], []
    hit = Counter()
    unresolved = 0
    for r in df.to_dict(orient="records"):
        v = visualpuzzles_hit(r["prediction"], r["answer"])
        unresolved += int(extract_unambiguous_abcd(r["prediction"]) == "Z")
        ids.append(int(r["source_ordinal"]))
        vals.append(float(v))
        for k in ("overall", f"category={r['category']}", f"level={r['difficulty']}"):
            hit[k] += v
    exp = {}
    for row in sj["scores"]["table"]:
        name = "overall" if row["group"] == "overall" else f"{row['type']}={row['group']}"
        exp[name] = row["hit"]
    return {"method": "visualpuzzles_abcd", "ids": ids, "values": vals, "inputs": [path],
            "extractor_unresolved_rate": unresolved / len(ids), "sub_checks": _sub_checks(exp, dict(hit))}


def _bench_mathvista(sj: Mapping[str, Any]) -> Dict[str, Any]:
    path = sj["artifacts"]["judged_table"]
    df = _read_xlsx(path)
    ids, vals = [], []
    hit = Counter()
    for r in df.to_dict(orient="records"):
        v = mathvista_hit(r)
        ids.append(int(r["source_ordinal"]))
        vals.append(float(v))
        hit["Overall"] += v
        hit[str(r["task"])] += v
    exp = {str(row["Task&Skill"]): row["hit"] for row in sj["scores"]["table"]
           if str(row["Task&Skill"]) in hit or str(row["Task&Skill"]) == "Overall"}
    return {"method": "mathvista_post_check", "ids": ids, "values": vals, "inputs": [path],
            "sub_checks": _sub_checks(exp, dict(hit))}


def _bench_mathvision(sj: Mapping[str, Any]) -> Dict[str, Any]:
    path = sj["artifacts"]["judged_table"]
    df = _read_xlsx(path)
    ids, vals = [], []
    hit = Counter()
    unresolved = 0
    _NAME_HITS["count"] = 0
    for r in df.to_dict(orient="records"):
        v, unres = mathvision_hit(r)
        unresolved += int(unres)
        ids.append(int(r["source_ordinal"]))
        vals.append(float(v))
        hit["Overall"] += v
        hit[str(r["category"])] += v
    exp = {str(row["Subject"]): row["hit"] for row in sj["scores"]["table"]}
    method = "mathvision_post_check" if _LATEX["fn"] is not None else "mathvision_post_check_no_latex2sympy"
    return {"method": method, "ids": ids, "values": vals, "inputs": [path], "latex2sympy": _LATEX["source"],
            "latex_fallback_rows": unresolved, "local_name_hits": _NAME_HITS.get("count", 0),
            "sub_checks": _sub_checks(exp, dict(hit))}


def _bench_chartqapro(sj: Mapping[str, Any]) -> Dict[str, Any]:
    path = sj["artifacts"]["prediction_table"]
    df = _read_xlsx(path)
    ids, vals = [], []
    by_type: Dict[str, List[float]] = defaultdict(list)
    for r in df.to_dict(orient="records"):
        v = chartqapro_score(ast.literal_eval(r["answer"]), r["question_type"], ast.literal_eval(r["year"]),
                             r["prediction"])
        ids.append(int(r["source_ordinal"]))
        vals.append(float(v))
        by_type[str(r["question_type"])].append(v)
        by_type["Overall"].append(v)
    got = {k: sum(v) / len(v) for k, v in by_type.items()}
    exp = {str(row["0"]): row["1"] for row in sj["scores"]["table"]}
    return {"method": "chartqapro_relaxed", "ids": ids, "values": vals, "inputs": [path],
            "sub_checks": _sub_checks(exp, got, tol=1e-12)}


TABLEVQA_W = {"fintabnetqa": 0.4, "vtabfact": 0.2, "vwtq": 0.2, "vwtq_syn": 0.2}


def _bench_tablevqabench(sj: Mapping[str, Any]) -> Dict[str, Any]:
    path = sj["artifacts"]["judged_table"]
    df = _read_xlsx(path)
    ids, vals, strata = [], [], []
    sums: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for r in df.to_dict(orient="records"):
        sc = ast.literal_eval(r["scores"]) if isinstance(r["scores"], str) else dict(r["scores"])
        split = str(r["split"])
        acc = float(sc.get("accuracy") or 0)
        if split == "fintabnetqa":
            ex = float(sc.get("exact_score") or 0)
            v = (acc + ex) / 2.0
            sums[split][1] += ex
        else:
            v = acc
        sums[split][0] += acc
        sums[split][2] += 1
        ids.append(int(r["source_ordinal"]))
        vals.append(v)
        strata.append(split)
    got, exp = {}, {}
    for row in sj["scores"]["table"]:
        s = row["split"]
        n = sums[s][2]
        rec = [100 * sums[s][0] / n] + ([100 * sums[s][1] / n] if s == "fintabnetqa" else [])
        for j, val in enumerate(row["average_scores"]):
            exp[f"{s}#{j}"] = val
            got[f"{s}#{j}"] = rec[j] if j < len(rec) else None
    if set(strata) != set(TABLEVQA_W):
        raise ExtError(f"TableVQABench splits {sorted(set(strata))}")
    return {"method": "tablevqabench_split_macro", "ids": ids, "values": vals, "strata": strata,
            "weights": dict(TABLEVQA_W), "inputs": [path], "sub_checks": _sub_checks(exp, got, tol=1e-5)}


def _bench_wemath(sj: Mapping[str, Any], arch: Mapping, seed: int, gen_pairs: Mapping[int, str],
                  gen_keys: Mapping[Tuple[str, str], Optional[int]]) -> Dict[str, Any]:
    outs = [p for p in sj["artifacts"].get("official_outputs", []) if p.endswith(".xlsx")]
    judged = [p for p in outs if "judge" in Path(p).name and not p.endswith("_result.xlsx")]
    if not judged:
        cand = sorted(Path(sj["artifacts"]["prediction_table"]).parent.glob("*_qwen3-32b-judge.xlsx"))
        judged = [str(c) for c in cand]
    if len(judged) != 1:
        raise ExtError(f"WeMath judged table not unique: {judged}")
    df = _read_xlsx(judged[0])
    rows = df[["ID", "key", "hit", "source_ordinal"]].to_dict(orient="records")
    # the archive's per-row score is this hit; check that both views agree
    agree = None
    cands = arch.get(("wemath", seed, "score")) or []
    if cands:
        arch_hit = {o: _finite(r.get("score")) for o, r in _archive_rows(cands[0]["payload"], gen_pairs, gen_keys, {})}
        agree = sum(1 for r in rows if arch_hit.get(int(r["source_ordinal"])) == float(r["hit"]))
    values, counts = wemath_groups(rows)
    # source ordinals of each group's rows: a subset keeps a group only when it keeps all of them
    members: Dict[str, List[int]] = defaultdict(list)
    for r in rows:
        m = re.fullmatch(r"([23])steps_(\d+|multi)", str(r["key"]))
        if m:
            members[f"{m.group(1)}:{r['ID']}"].append(int(r["source_ordinal"]))
    tab = sj["scores"]["table"][0]
    exp = {}
    for name, key in (("IK", "InsufficientKnowledge (Strict)"), ("IG", "InadequateGeneralization (Strict)"),
                      ("CM_strict", "CompleteMastery (Strict)"), ("RM_strict", "RoteMemorization (Strict)")):
        m = re.search(r"\((\d+)\)", str(tab.get(key, "")))
        if m:
            exp[name] = int(m.group(1))
    keys = sorted(values)
    return {"method": "wemath_strict_groups", "ids": keys, "values": [values[k] for k in keys], "inputs": [judged[0]],
            "group_members": {k: sorted(members[k]) for k in keys},
            "wemath_counts": counts, "archive_hit_agreement": {"rows": len(rows), "agree": agree},
            "sub_checks": _sub_checks(exp, counts)}


XLSX_BENCH = {"countbenchqa": _bench_countbenchqa, "puzzlevqa": _bench_puzzlevqa,
              "visualpuzzles": _bench_visualpuzzles, "mathvista": _bench_mathvista, "mathvision": _bench_mathvision,
              "chartqapro": _bench_chartqapro, "tablevqabench": _bench_tablevqabench}
SCORE_TOL = {"wemath": 0.0051, "tablevqabench": 1e-5}  # official values rounded (2 decimals; 8-decimal fractions)


def _reconstructed_score(entry: Mapping[str, Any]) -> float:
    vals, strata = entry["values"], entry.get("strata")
    if not strata:
        return 100.0 * sum(vals) / len(vals)
    sums: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0])
    for v, h in zip(vals, strata):
        sums[h][0] += v
        sums[h][1] += 1
    return 100.0 * sum(entry["weights"][h] * s / n for h, (s, n) in sums.items())


WEMATH_GROUPS = 525


def _scorer_counter(sj: Mapping[str, Any], key: str) -> Optional[int]:
    """A fix-2 / fix-3 counter where the scorers write it: ``artifacts`` (MathVerse), ``evaluation``
    (MME-Reasoning) or top level; None when this scores.json predates the counter."""
    for path in (("artifacts", key), ("evaluation", key), (key,)):
        node: Any = sj
        for p in path:
            node = node.get(p) if isinstance(node, dict) else None
        if node is not None and not isinstance(node, bool) and isinstance(node, (int, float)):
            return int(node)
    return None


def _short(h: Optional[str]) -> str:
    return (h or "?")[:12]


def archive_provenance(arch: Mapping, seed: int) -> Dict[str, Dict[str, int]]:
    """Code hash of the newest archive slice per benchmark and stage (the one a rerun of the summary reads), and
    the hashes of older (superseded) score slices of the same benchmark, e.g. a stale pre-fix-3 scoring."""
    prov: Dict[str, Counter] = {"generation": Counter(), "extraction": Counter(), "score": Counter(),
                                "score_superseded": Counter()}
    for (bench, s, stage), recs in arch.items():
        if s != seed or stage not in prov or not recs:
            continue
        prov[stage][_short(recs[0]["code_hash"])] += 1
        if stage == "score":
            for r in recs[1:]:
                prov["score_superseded"][_short(r["code_hash"])] += 1
    return {k: dict(sorted(v.items())) for k, v in prov.items()}


def build_run_items(run_root: Path, served: str, seed: int, summary_path: Path,
                    subset: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Per-item table of one complete run. ``subset`` = {"name", "rows": {bench: {ordinal: row hash}}, "path"}:
    on a full run the restricted row digest is stored per benchmark; on a run of that subset its rows must equal
    the subset's exactly."""
    summary = _read_json(summary_path)
    if not summary.get("complete"):
        raise ExtError(f"{summary_path} is not complete")
    run_subset = summary.get("subset") or "full"
    arch = _archive_index(run_root)
    out: Dict[str, Any] = {"schema": ITEMS_SCHEMA, "version": VERSION, "served": served, "seed": seed,
                           "subset": run_subset, "summary_path": str(summary_path),
                           "summary_created": summary.get("created"), "benchmarks": {}}
    inputs: List[Path] = [summary_path]
    if subset is not None:
        inputs.append(Path(subset["path"]))
    counters: Dict[str, Dict[str, Optional[int]]] = {}
    for b in summary["benchmarks"]:
        key = b["key"]
        official = float(b["score"])
        gen_pairs, digest, gen_payload, gen_keys = _generation_identity(arch, key, seed)
        sj = _scores_json(b)
        inputs.append(Path(b["scores_path"]))
        counters[key] = {k: _scorer_counter(sj, k) for k in EXCEPTION_KEYS}
        for k in EXCEPTION_KEYS:
            if not counters[key][k]:
                continue
            for node in (sj.get("artifacts"), sj.get("evaluation"), sj):
                if isinstance(node, dict) and node.get(k + "_indices"):
                    counters[key][k + "_indices"] = [str(x) for x in node[k + "_indices"]]
                    break
        if key in XLSX_BENCH or key == "wemath":
            entry = (_bench_wemath(sj, arch, seed, gen_pairs, gen_keys) if key == "wemath" else XLSX_BENCH[key](sj))
            if key != "wemath":
                ords = set(entry["ids"])
                if ords != set(gen_pairs):
                    raise ExtError(f"{served} seed {seed} {key}: table ordinals differ from the archive's")
            else:
                members = {o for ms in entry["group_members"].values() for o in ms}
                if members - set(gen_pairs):
                    raise ExtError(f"{served} seed {seed} wemath: group rows outside the generation archive")
        else:
            entry = _per_row_from_archive(arch, key, seed, official, gen_pairs, gen_keys)
            if set(entry["ids"]) - set(gen_pairs):
                raise ExtError(f"{served} seed {seed} {key}: score ordinals outside the generation archive")
        if len(set(entry["ids"])) != len(entry["ids"]):
            raise ExtError(f"{served} seed {seed} {key}: duplicate item ids")
        entry["reconstructed"] = _reconstructed_score(entry)
        entry["official"] = official
        tol = SCORE_TOL.get(key, 1e-6)
        protocol_ok = True
        if key == "wemath" and len(entry["ids"]) != WEMATH_GROUPS:
            # subset run: the summary rescales the official 525-group score (2 decimals) to its groups, so the
            # rounding grows by 525/groups; the official protocol value itself is checked at 2-decimal tolerance
            n_g = len(entry["ids"])
            tol = SCORE_TOL["wemath"] * WEMATH_GROUPS / n_g
            loss = sum(1.0 - v for v in entry["values"])
            rec_protocol = 100.0 * (WEMATH_GROUPS - loss) / WEMATH_GROUPS
            off_protocol = _finite(b.get("score_protocol"))
            entry["protocol_check"] = {"official": off_protocol, "reconstructed": rec_protocol,
                                       "note": b.get("score_note")}
            protocol_ok = off_protocol is not None and abs(rec_protocol - off_protocol) <= SCORE_TOL["wemath"]
        entry["abs_diff"] = abs(entry["reconstructed"] - official)
        sub = entry.get("sub_checks")
        entry["valid"] = bool(entry["abs_diff"] <= tol and (sub is None or not sub["mismatch"]) and protocol_ok)
        entry["tolerance"] = tol
        entry["row_digest"] = digest
        entry["n_items"] = len(entry["ids"])
        entry["generation_payload"] = gen_payload
        if subset is not None:
            srows = subset["rows"].get(key) or {}
            if run_subset == subset["name"]:
                entry["subset_exact"] = {subset["name"]: dict(gen_pairs) == srows}
            elif run_subset == "full":
                entry["subset_digest"] = {subset["name"]: _row_digest((o, gen_pairs.get(o, "")) for o in srows)}
                entry["subset_rows_match"] = {subset["name"]: all(gen_pairs.get(o) == h for o, h in srows.items())}
        inputs += [Path(p) for p in entry["inputs"]] + [Path(gen_payload)]
        out["benchmarks"][key] = entry
    out["scorer_counters"] = counters
    out["provenance"] = archive_provenance(arch, seed)
    out["latex2sympy"] = _LATEX["source"]
    out["code_sha256"] = code_sha256()
    out["fingerprint"] = _fingerprint(inputs)
    out["input_files"] = sorted({str(p) for p in inputs})
    return out


def code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _items_path(out_dir: Path, served: str, seed: int, subset: str = "full") -> Path:
    if subset == "full":
        return out_dir / "items" / f"{served}.seed{seed}.json.gz"
    return out_dir / "items" / subset / f"{served}.seed{seed}.json.gz"


def load_subset(path: Path) -> Dict[str, Any]:
    """The frozen subset row list: per benchmark {source ordinal: source row hash} and its row digest."""
    d = _read_json(path)
    name = d.get("mode") or Q25
    rows = {k: {int(r["ordinal"]): str(r["source_row_hash"]) for r in v["rows"]} for k, v in d["benchmarks"].items()}
    return {"name": name, "path": str(path), "row_list_sha256": d.get("row_list_sha256"), "rows": rows,
            "digests": {k: _row_digest(v.items()) for k, v in rows.items()},
            "selected": {k: v.get("selected") for k, v in d["benchmarks"].items()},
            "total": {k: v.get("total") for k, v in d["benchmarks"].items()}}


_SUBSET_CACHE: Dict[str, Any] = {}


def _items_job(args: Tuple[str, str, int, str, str, bool, Optional[str], str, Optional[str]]) -> Dict[str, Any]:
    ext_root, served, seed, summary_path, out_dir, force, vendor, run_subset, subset_path = args
    use_latex2sympy(vendor)
    path = _items_path(Path(out_dir), served, seed, run_subset)
    if path.is_file() and not force:
        old = _read_items(path)
        try:
            same = (old.get("version") == VERSION and old.get("code_sha256") == code_sha256()
                    and old.get("fingerprint") == _fingerprint(old.get("input_files", [])))
        except OSError:
            same = False
        same = same and bool(old.get("latex2sympy")) == bool(vendor)
        if same and old.get("summary_created") == _read_json(Path(summary_path)).get("created"):
            return {"served": served, "seed": seed, "subset": run_subset, "status": "cached", "path": str(path)}
    t0 = time.time()
    try:
        subset = None
        if subset_path:
            if subset_path not in _SUBSET_CACHE:
                _SUBSET_CACHE[subset_path] = load_subset(Path(subset_path))
            subset = _SUBSET_CACHE[subset_path]
        items = build_run_items(Path(ext_root) / "runs" / served / run_subset, served, seed, Path(summary_path), subset)
    except Exception as exc:  # recorded per run; analyze treats the run as unusable for paired statistics
        return {"served": served, "seed": seed, "subset": run_subset, "status": "error",
                "error": f"{type(exc).__name__}: {exc}"}
    _write_json(path, items, gz=True)
    bad = [k for k, e in items["benchmarks"].items() if not e["valid"]]
    return {"served": served, "seed": seed, "subset": run_subset, "status": "built", "path": str(path),
            "seconds": round(time.time() - t0, 1), "invalid_benchmarks": bad}


def cmd_items(args: argparse.Namespace) -> int:
    ext_root, out_dir = Path(args.ext_root), Path(args.out_dir)
    inv = discover(ext_root)
    subset_path = Path(args.subset_file) if args.subset_file else ext_root / Q25_SUBSET_FILE
    sp = str(subset_path) if subset_path.is_file() else None
    runs = [dict(r, subset="full") for r in inv["runs"]] + list(discover_q25(ext_root)["runs"])
    jobs = [(str(ext_root), r["served"], r["seed"], r["summary_path"], str(out_dir), args.force, args.vendor,
             r["subset"], sp)
            for r in runs if r["complete"] and (not args.only or r["served"] in args.only.split(","))]
    log = []
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        for res in ex.map(_items_job, jobs):
            log.append(res)
            print(json.dumps(res), flush=True)
    _write_json(out_dir / "items" / "items_log.json", {"version": VERSION, "jobs": log})
    errors = [r for r in log if r["status"] == "error"]
    return 1 if errors else 0


# ============================================================================== stage 2: statistics


def _np():
    import numpy as np

    return np


def _load_compare_cert():
    sys.dont_write_bytecode = True  # read-only use of the checkout: no __pycache__ next to compare_cert.py
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("compare_cert", here / "compare_cert.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("compare_cert", mod)
    spec.loader.exec_module(mod)
    return mod


class BenchArrays:
    """One benchmark's paired item values for a comparison: d (X - Y) and x, y per stratum, in [0, 1]."""

    def __init__(self, key: str, strata: Dict[str, Tuple[Any, Any]], weights: Dict[str, float]):
        np = _np()
        self.key = key
        self.strata = {h: (np.asarray(x, float), np.asarray(y, float)) for h, (x, y) in strata.items()}
        self.weights = weights
        self.n = sum(len(x) for x, _ in self.strata.values())

    def means(self) -> Tuple[float, float]:
        mx = sum(self.weights[h] * float(x.mean()) for h, (x, _) in self.strata.items())
        my = sum(self.weights[h] * float(y.mean()) for h, (_, y) in self.strata.items())
        return 100.0 * mx, 100.0 * my

    def boot(self, n: int, rng, chunk: int = 500):
        """(n,) resampled benchmark differences in points (items resampled within each stratum)."""
        np = _np()
        out = np.zeros(n)
        for h, (x, y) in self.strata.items():
            d = x - y
            k = len(d)
            for b0 in range(0, n, chunk):
                b1 = min(n, b0 + chunk)
                idx = rng.integers(0, k, size=(b1 - b0, k), dtype=np.int32)
                out[b0:b1] += self.weights[h] * d[idx].mean(axis=1)
        return 100.0 * out

    def coef_diff(self):
        """Per-item coefficient c and difference d with sum(c * d) = benchmark difference in points."""
        np = _np()
        cs, ds = [], []
        for h, (x, y) in self.strata.items():
            cs.append(np.full(len(x), 100.0 * self.weights[h] / len(x)))
            ds.append(x - y)
        return np.concatenate(cs), np.concatenate(ds)


def group_members(categories: Mapping[str, Sequence[str]]) -> Dict[str, Tuple[str, ...]]:
    g = dict(GROUPS)
    g.update(SENS_GROUPS)
    for cat, keys in categories.items():
        g[f"category:{cat}"] = tuple(keys)
    for k in ALL_KEYS:
        g[f"benchmark:{k}"] = (k,)
    return g


def macro_from_bench(vals: Mapping[str, Any], members: Sequence[str]):
    np = _np()
    return np.mean(np.stack([vals[k] for k in members]), axis=0)


def signflip_groups(benches: Mapping[str, BenchArrays], groups: Mapping[str, Sequence[str]], n_perm: int, rng,
                    chunk: int = 250) -> Dict[str, float]:
    """Two-sided item sign-flip p of every group's macro difference (one shared set of flips)."""
    np = _np()
    keys = list(benches)
    cd, offs = [], {}
    pos = 0
    for k in keys:
        c, d = benches[k].coef_diff()
        cd.append(c * d)
        offs[k] = (pos, pos + len(d))
        pos += len(d)
    v = np.concatenate(cd)
    gidx = {}
    for g, members in groups.items():
        sel = np.zeros(len(v))
        for k in members:
            a, b = offs[k]
            sel[a:b] = 1.0 / len(members)
        gidx[g] = sel * v
    obs = {g: abs(float(w.sum())) for g, w in gidx.items()}
    W = np.stack([gidx[g] for g in groups], axis=1)  # (N, G)
    exceed = np.zeros(len(groups))
    obs_v = np.array([obs[g] for g in groups])
    for b0 in range(0, n_perm, chunk):
        b1 = min(n_perm, b0 + chunk)
        s = rng.integers(0, 2, size=(b1 - b0, len(v))).astype(np.float64) * 2.0 - 1.0
        t = np.abs(s @ W)
        exceed += (t >= obs_v[None, :] - 1e-12 * np.maximum(1.0, obs_v[None, :])).sum(axis=0)
    p = (1.0 + exceed) / (1.0 + n_perm)
    return {g: float(p[i]) for i, g in enumerate(groups)}


def _ci(arr) -> List[Optional[float]]:
    np = _np()
    a = arr[~np.isnan(arr)]
    if not a.size:
        return [None, None]
    lo, hi = np.percentile(a, [2.5, 97.5])
    return [float(lo), float(hi)]


def _boot_p(arr) -> Optional[float]:
    """Two-sided percentile-bootstrap p of 0 (+1 corrected): consistent with the percentile CI excluding 0."""
    np = _np()
    a = arr[~np.isnan(arr)]
    if not a.size:
        return None
    lo = (1.0 + float((a <= 0).sum())) / (1.0 + a.size)
    hi = (1.0 + float((a >= 0).sum())) / (1.0 + a.size)
    return min(1.0, 2.0 * min(lo, hi))


def _bench_view(e: Mapping[str, Any]) -> Dict[str, Any]:
    strata = e.get("strata") or ["-"] * len(e["ids"])
    return {"values": {str(i): (float(v), h) for i, v, h in zip(e["ids"], e["values"], strata)},
            "weights": e.get("weights") or {"-": 1.0}, "digest": e["row_digest"],
            "valid": e["valid"], "abs_diff": e["abs_diff"], "method": e["method"],
            "official": e["official"], "reconstructed": e["reconstructed"],
            "n_items": e["n_items"], "latex_fallback_rows": e.get("latex_fallback_rows"),
            "local_name_hits": e.get("local_name_hits"),
            "extractor_unresolved_rate": e.get("extractor_unresolved_rate"),
            "sub_checks": e.get("sub_checks"), "wemath_counts": e.get("wemath_counts"),
            "archive_hit_agreement": e.get("archive_hit_agreement"),
            "group_members": e.get("group_members"), "subset_digest": e.get("subset_digest") or {},
            "subset_rows_match": e.get("subset_rows_match") or {}, "subset_exact": e.get("subset_exact") or {},
            "archive_rows_rekeyed": e.get("archive_rows_rekeyed"),
            "archive_rekeyed_as_position": e.get("archive_rekeyed_as_position"),
            "protocol_check": e.get("protocol_check")}


def bench_score(b: Mapping[str, Any]) -> float:
    """Benchmark score (points) of a store entry: sum_h W_h * mean of its items in h."""
    sums: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0])
    for v, h in b["values"].values():
        sums[h][0] += v
        sums[h][1] += 1
    return 100.0 * sum(b["weights"][h] * s / n for h, (s, n) in sums.items())


class Store:
    """Item tables of every usable run: served -> seed -> {benchmark: {id: value}, strata, weights, digest}."""

    def __init__(self):
        self.runs: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
        self.meta: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self.problems: List[Dict[str, Any]] = []

    def load(self, out_dir: Path, inv: Mapping[str, Any], subset: str = "full") -> None:
        for r in inv["runs"]:
            if not r["complete"]:
                continue
            p = _items_path(out_dir, r["served"], r["seed"], subset)
            if not p.is_file():
                self.problems.append({"served": r["served"], "seed": r["seed"], "problem": "no item table (run items)"})
                continue
            it = _read_items(p)
            if it.get("summary_created") != _read_json(Path(r["summary_path"])).get("created"):
                self.problems.append({"served": r["served"], "seed": r["seed"],
                                      "problem": "item table older than the summary (re-run items)"})
                continue
            self.runs[r["served"]][r["seed"]] = {k: _bench_view(e) for k, e in it["benchmarks"].items()}
            self.meta[(r["served"], r["seed"])] = {"scorer_counters": it.get("scorer_counters"),
                                                   "provenance": it.get("provenance")}

    def digests_consistent(self) -> Dict[str, Any]:
        out = {}
        for k in ALL_KEYS:
            ds = Counter(b[k]["digest"] for by in self.runs.values() for b in by.values() if k in b)
            out[k] = {"distinct_row_sets": len(ds), "runs": sum(ds.values())}
        return out


def paired_benches(store: Store, xs: str, ys: str, seeds: Optional[Sequence[int]] = None
                   ) -> Tuple[Dict[str, BenchArrays], List[int], Dict[str, Any]]:
    """Pair run ``xs`` with ``ys`` per benchmark: item value = mean over the decoding seeds both have."""
    sx, sy = store.runs.get(xs, {}), store.runs.get(ys, {})
    common = sorted(set(sx) & set(sy))
    if seeds is not None:
        common = [s for s in common if s in seeds]
    if not common:
        raise ExtError(f"{xs} and {ys} share no complete decoding seed")
    benches, info = {}, {"invalid": []}
    for k in ALL_KEYS:
        digests = {sx[s][k]["digest"] for s in common} | {sy[s][k]["digest"] for s in common}
        if len(digests) != 1:
            raise ExtError(f"{k}: row identities differ between {xs} and {ys}")
        for s in common:
            for who, run in ((xs, sx[s][k]), (ys, sy[s][k])):
                if not run["valid"]:
                    info["invalid"].append(f"{who}@{s}:{k}")
        ids = set.intersection(*[set(sx[s][k]["values"]) for s in common], *[set(sy[s][k]["values"]) for s in common])
        weights = sx[common[0]][k]["weights"]
        strata: Dict[str, Tuple[List[float], List[float]]] = defaultdict(lambda: ([], []))
        for i in sorted(ids):
            h = sx[common[0]][k]["values"][i][1]
            xv = sum(sx[s][k]["values"][i][0] for s in common) / len(common)
            yv = sum(sy[s][k]["values"][i][0] for s in common) / len(common)
            strata[h][0].append(xv)
            strata[h][1].append(yv)
        benches[k] = BenchArrays(k, dict(strata), weights)
    return benches, common, info


def seed_stats(benches: Mapping[str, BenchArrays], groups: Mapping[str, Sequence[str]], st: Mapping[str, Any],
               rng_key: Sequence[int]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """One training-seed pair: per group delta, means, bootstrap CI and sign-flip p; also the bootstrap matrix."""
    np = _np()
    rng_b = np.random.default_rng([st["seed"], 1, *rng_key])
    reps = {k: b.boot(st["n_boot"], rng_b) for k, b in benches.items()}
    p = signflip_groups(benches, groups, st["n_perm"], np.random.default_rng([st["seed"], 2, *rng_key]))
    means = {k: b.means() for k, b in benches.items()}
    out = {}
    for g, members in groups.items():
        mx = float(np.mean([means[k][0] for k in members]))
        my = float(np.mean([means[k][1] for k in members]))
        boot = macro_from_bench(reps, members)
        out[g] = {"mean_x": mx, "mean_y": my, "delta": mx - my, "ci": _ci(boot), "boot_p": _boot_p(boot),
                  "perm_p": p[g], "n_items": int(sum(benches[k].n for k in members)), "n_benchmarks": len(members)}
    return out, reps


def two_level(per_seed_benches: Sequence[Mapping[str, BenchArrays]], groups: Mapping[str, Sequence[str]],
              st: Mapping[str, Any], rng_key: Sequence[int]) -> Dict[str, Any]:
    """compare_cert.two_level_bootstrap on macros: resample training seeds, then items within benchmarks."""
    np = _np()
    K = len(per_seed_benches)
    rng = np.random.default_rng([st["seed"], 3, *rng_key])
    pick = rng.integers(0, K, size=(st["n_boot"], K))
    vals = {k: np.full((st["n_boot"], K), np.nan) for k in ALL_KEYS}
    for kk in range(K):
        where = np.argwhere(pick == kk)
        if len(where):
            for k, b in per_seed_benches[kk].items():
                vals[k][where[:, 0], where[:, 1]] = b.boot(len(where), rng)
    bench_mean = {k: v.mean(axis=1) for k, v in vals.items()}
    return {g: macro_from_bench(bench_mean, members) for g, members in groups.items()}


def run_comparison(store: Store, name: str, xarm: str, yarm: str, kind: str, groups: Mapping[str, Sequence[str]],
                   st: Mapping[str, Any], cc, seeds: Optional[Sequence[int]] = None,
                   models: Sequence[Mapping[str, Any]] = MODELS) -> Dict[str, Any]:
    np = _np()
    xm = {m["tseed"]: m for m in models if m["arm"] == xarm and m["served"] in store.runs}
    ym = {m["tseed"]: m for m in models if m["arm"] == yarm and m["served"] in store.runs}
    want_x = [m for m in models if m["arm"] == xarm]
    want_y = [m for m in models if m["arm"] == yarm]
    res: Dict[str, Any] = {"name": name, "x": xarm, "y": yarm, "kind": kind,
                           "decoding_seed_filter": list(seeds) if seeds else None,
                           "missing_x": [m["label"] for m in want_x if m["served"] not in store.runs],
                           "missing_y": [m["label"] for m in want_y if m["served"] not in store.runs]}
    if not xm or not ym:
        res["status"] = "pending"
        return res
    # matched training seeds; a single run on one side is shared (compare_cert.tseed_pairs)
    common = sorted(set(xm) & set(ym))
    if common and not (common == [""] and (len(xm) > 1 or len(ym) > 1)):
        tps = [(t, t) for t in common]
    elif len(ym) == 1:
        tps = [(t, next(iter(ym))) for t in sorted(xm)]
    elif len(xm) == 1:
        tps = [(next(iter(xm)), t) for t in sorted(ym)]
    else:
        res["status"] = "pending"
        res["note"] = "no matched training seeds"
        return res
    res["training_seed_pairs"] = [[a, b] for a, b in tps]
    per_seed, bench_sets = [], []
    for i, (tx, ty) in enumerate(tps):
        sx, sy = xm[tx]["served"], ym[ty]["served"]
        benches, used, info = paired_benches(store, sx, sy, seeds)
        key = [abs(hash_int(name)) % (2 ** 31), i]
        stats, _ = seed_stats(benches, groups, st, key)
        per_seed.append({"tseeds": [tx, ty], "x_served": sx, "y_served": sy, "decoding_seeds": used,
                         "reconstruction_invalid": info["invalid"], "groups": stats})
        bench_sets.append(benches)
    res["per_seed"] = per_seed
    K = len(per_seed)
    res["K"] = K
    combined: Dict[str, Any] = {}
    if K == 1:
        for g, r in per_seed[0]["groups"].items():
            combined[g] = {**r, "K": 1, "ci_scope": "instance sampling only (1 training seed)",
                           "p": r["perm_p"], "test": "item sign-flip permutation"}
    else:
        boots = two_level(bench_sets, groups, st, [abs(hash_int(name)) % (2 ** 31), 99])
        need = cc.sign_needed(K)
        for g in groups:
            deltas = [ps["groups"][g]["delta"] for ps in per_seed]
            delta = float(np.mean(deltas))
            sgn = 1 if delta > 0 else -1 if delta < 0 else 0
            agree = sum(1 for d in deltas if sgn and d * sgn > 0)
            ci = _ci(boots[g])
            excl = ci[0] is not None and (ci[0] > 0 or ci[1] < 0)
            combined[g] = {
                "K": K, "mean_x": float(np.mean([ps["groups"][g]["mean_x"] for ps in per_seed])),
                "mean_y": float(np.mean([ps["groups"][g]["mean_y"] for ps in per_seed])),
                "delta": delta, "ci": ci, "ci_scope": f"two-level bootstrap over {K} training seeds and items",
                "boot_p": _boot_p(boots[g]), "p": _boot_p(boots[g]), "test": "two-level bootstrap p",
                "sign_agree": agree, "sign_needed": need, "claim_compare_cert": bool(excl and agree >= need),
                "per_seed_delta": deltas, "per_seed_perm_p": [ps["groups"][g]["perm_p"] for ps in per_seed]}
    res["combined"] = combined
    res["status"] = "ok"
    return res


def hash_int(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def apply_family(comps: Sequence[Dict[str, Any]], kind: str, holm, alpha: float) -> Dict[str, Any]:
    tests = []
    for c in comps:
        if c["kind"] != kind:
            continue
        for g in FAMILY_GROUPS:
            r = (c.get("combined") or {}).get(g)
            tests.append({"comparison": c["name"], "group": g, "status": c.get("status"), "result": r})
    present = [t for t in tests if t["result"] is not None]
    if present:
        adj = holm([1.0 if t["result"]["p"] is None else float(t["result"]["p"]) for t in present])
        for t, a in zip(present, adj):
            t["holm_p"] = float(a)
        # compare_cert convention: per-seed sign-flip p, Holm within each training-seed index
        maxk = max(t["result"]["K"] for t in present)
        if maxk > 1:
            for i in range(maxk):
                items = [(t, t["result"]["per_seed_perm_p"][i]) for t in present
                         if t["result"]["K"] > 1 and i < len(t["result"]["per_seed_perm_p"])]
                if items:
                    a2 = holm([p for _, p in items])
                    for (t, _), a in zip(items, a2):
                        t.setdefault("holm_p_per_seed", []).append(float(a))
        for t in present:
            r = t["result"]
            direction = "X > Y" if (r["delta"] or 0) > 0 else "X < Y"
            if r["K"] == 1:
                t["significant"] = bool(t["holm_p"] < alpha)
                t["decision"] = direction if t["significant"] else "not significant"
            else:
                t["significant"] = bool(t["holm_p"] < alpha and r["sign_agree"] >= r["sign_needed"])
                t["decision"] = direction if t["significant"] else "no claim"
                t["decision_compare_cert_convention"] = direction if r["claim_compare_cert"] else "no claim"
    return {"family": kind, "groups": list(FAMILY_GROUPS), "tests": tests,
            "complete": bool(tests) and all(t["result"] is not None for t in tests)}


def _find_counts(obj: Any, key: str) -> List[Any]:
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                found.append(v)
            else:
                found.extend(_find_counts(v, key))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(_find_counts(v, key))
    return found


def _count_total(v: Any) -> Optional[int]:
    if isinstance(v, dict):
        if "total" in v and isinstance(v["total"], (int, float)):
            return int(v["total"])
        return sum(int(x or 0) for x in v.values() if isinstance(x, (int, float)))
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return int(v)
    return None


ABCD_KEYS = ("puzzlevqa", "visualpuzzles")
AVG_RATES = ("det_fail_rate", "judge_needed_rate_proxy", "judge_needed_protocol_mathvision",
             "judge_needed_protocol_mathvista", "cert_json_rate", "json_answer_rate", "length_cap_rate",
             "abcd_unresolved_puzzlevqa", "abcd_unresolved_visualpuzzles")


def format_shift(inv: Mapping[str, Any], store: Optional["Store"] = None) -> Dict[str, Any]:
    rows = []
    for r in inv["runs"]:
        if not r["complete"]:
            continue
        s = _read_json(Path(r["summary_path"]))
        ex = s.get("extraction") or {}
        jp = s.get("judge_needed_protocol") or {}
        m = MODEL_BY_SERVED.get(r["served"], {})
        entry = {"served": r["served"], "label": m.get("label", r["served"]), "arm": m.get("arm"), "seed": r["seed"],
                 "det_fail_rate": ex.get("det_fail_rate"), "judge_needed_rate_proxy": ex.get("judge_needed_rate_proxy"),
                 "judge_needed_protocol_mathvision": jp.get("mathvision"),
                 "judge_needed_protocol_mathvista": jp.get("mathvista"),
                 "cert_json_rate": ex.get("cert_json_rate"), "json_answer_rate": ex.get("json_answer_rate"),
                 "length_cap_rate": ex.get("length_cap_rate"), "counters": {}}
        for k in EXCEPTION_KEYS:
            top = s.get(k)
            if top is not None:
                entry["counters"][k] = {"total": _count_total(top),
                                        "by_benchmark": {b: n for b, n in (top.get("by_benchmark") or {}).items() if n}
                                        if isinstance(top, dict) else None}
            else:
                per = [(b["key"], b.get(k)) for b in s.get("benchmarks", []) if b.get(k) is not None]
                deep = _find_counts(s, k)
                if per:
                    entry["counters"][k] = {"total": sum(int(n or 0) for _, n in per),
                                            "by_benchmark": {b: n for b, n in per if n}}
                elif deep:
                    entry["counters"][k] = {"total": sum(_count_total(v) or 0 for v in deep), "by_benchmark": None}
                else:
                    entry["counters"][k] = None  # not reported by this summary (fix-3 not applied yet)
        entry["benchmarks_det_fail"] = {b["key"]: b.get("det_fail_rate") for b in s.get("benchmarks", [])}
        # the official option-letter extractor of PuzzleVQA / VisualPuzzles (explicit "Answer: X", one answer block
        # or a final box; anything else is "Z" and scored wrong) -- from the item tables
        items = (store.runs.get(r["served"], {}).get(r["seed"]) if store else None) or {}
        for k in ABCD_KEYS:
            entry[f"abcd_unresolved_{k}"] = (items.get(k) or {}).get("extractor_unresolved_rate")
        rows.append(entry)
    # per model mean over its complete decoding seeds
    models = []
    by = defaultdict(list)
    for e in rows:
        by[e["served"]].append(e)
    for served, es in by.items():
        avg = {k: (float(sum(e[k] for e in es) / len(es)) if all(e[k] is not None for e in es) else None)
               for k in AVG_RATES}
        counters = {}
        for k in EXCEPTION_KEYS:
            vals = [e["counters"][k]["total"] for e in es if e["counters"].get(k)]
            counters[k] = {"per_seed": {str(e["seed"]): (e["counters"][k] or {}).get("total") if e["counters"].get(k) else None
                                        for e in es}, "reported": bool(vals)}
        m = MODEL_BY_SERVED.get(served, {})
        models.append({"served": served, "label": m.get("label", served), "arm": m.get("arm"),
                       "seeds": sorted(e["seed"] for e in es), **avg, "counters": counters})
    order = {m["served"]: i for i, m in enumerate(MODELS)}
    models.sort(key=lambda e: order.get(e["served"], 999))
    rows.sort(key=lambda e: (order.get(e["served"], 999), e["seed"]))
    return {"runs": rows, "models": models}


def flag_pairs(fs: Mapping[str, Any], comps: Sequence[Mapping[str, Any]], threshold_pp: float) -> List[Dict[str, Any]]:
    """Rates of the decoding seeds a comparison actually used, per training-seed pair; flagged above threshold."""
    by_run = {(e["served"], e["seed"]): e for e in fs["runs"]}
    out = []
    for c in comps:
        for ps in c.get("per_seed", []):
            for k in RATE_FLAG_KEYS:
                vx = [by_run[(ps["x_served"], s)][k] for s in ps["decoding_seeds"]]
                vy = [by_run[(ps["y_served"], s)][k] for s in ps["decoding_seeds"]]
                if any(v is None for v in vx + vy):
                    continue
                dx, dy = sum(vx) / len(vx), sum(vy) / len(vy)
                diff = 100.0 * (dx - dy)
                out.append({"comparison": c["name"], "kind": c["kind"], "tseeds": ps["tseeds"], "x": ps["x_served"],
                            "y": ps["y_served"], "rate": k, "x_rate": dx, "y_rate": dy, "diff_pp": diff,
                            "flag": bool(abs(diff) > threshold_pp)})
    return out


def official_tables(inv: Mapping[str, Any]) -> Dict[str, Any]:
    runs = []
    categories: Dict[str, List[str]] = {}
    for r in inv["runs"]:
        s = _read_json(Path(r["summary_path"]))
        m = MODEL_BY_SERVED.get(r["served"], {})
        mac = s.get("macro") or {}
        for b in s.get("benchmarks", []):
            categories.setdefault(b["category"], [])
            if b["key"] not in categories[b["category"]]:
                categories[b["category"]].append(b["key"])
        runs.append({"served": r["served"], "label": m.get("label", r["served"]), "arm": m.get("arm"),
                     "scale": m.get("scale"), "tseed": m.get("tseed"), "seed": r["seed"], "complete": r["complete"],
                     "overall_24": mac.get("overall_24"), "overall_category_mean": mac.get("overall_category_mean"),
                     "natural_9": mac.get("natural_9"), "chart_diagram_synthetic_15": mac.get("chart_diagram_synthetic_15"),
                     "mmstar": mac.get("mmstar"), "categories": mac.get("categories") or {},
                     "benchmarks": {b["key"]: b.get("score") for b in s.get("benchmarks", [])},
                     "groups_in_summary": s.get("groups")})
    order = {m["served"]: i for i, m in enumerate(MODELS)}
    runs.sort(key=lambda e: (order.get(e["served"], 999), e["seed"]))
    for e in runs:  # the summary's own groups must be DESIGN 6's frozen mapping
        g = e.pop("groups_in_summary") or {}
        if set(g.get("natural", [])) != set(NATURAL_KEYS) or set(g.get("chart_diagram_synthetic", [])) != set(SYNTHETIC_KEYS):
            raise ExtError(f"{e['served']} seed {e['seed']}: summary groups differ from DESIGN 6")
    complete = [e for e in runs if e["complete"]]
    fields = ("overall_24", "overall_category_mean", "natural_9", "chart_diagram_synthetic_15", "mmstar")

    def avg(es: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        out = {f: (float(sum(e[f] for e in es) / len(es)) if all(e[f] is not None for e in es) else None)
               for f in fields}
        out["categories"] = {c: float(sum(e["categories"][c] for e in es) / len(es)) for c in es[0]["categories"]}
        out["benchmarks"] = {b: float(sum(e["benchmarks"][b] for e in es) / len(es)) for b in ALL_KEYS}
        return out

    models = []
    by = defaultdict(list)
    for e in complete:
        by[e["served"]].append(e)
    for served in sorted(by, key=lambda s: order.get(s, 999)):
        es = by[served]
        m = MODEL_BY_SERVED.get(served, {})
        models.append({"served": served, "label": m.get("label", served), "arm": m.get("arm"),
                       "seeds": [e["seed"] for e in es], "mean_over_decoding_seeds": avg(es),
                       "seed42": next(({f: e[f] for f in fields} for e in es if e["seed"] == 42), None)})
    arms = []
    for arm in ("A", "G", "P"):
        es = [e for e in complete if e["arm"] == arm and e["seed"] == 42]
        if not es:
            continue
        arms.append({"arm": arm, "decoding_seed": 42, "training_seeds": [e["tseed"] for e in es],
                     "mean": avg(es), "per_training_seed": {e["tseed"]: {f: e[f] for f in fields} for e in es},
                     "min": {f: min((e[f] for e in es if e[f] is not None), default=None) for f in fields},
                     "max": {f: max((e[f] for e in es if e[f] is not None), default=None) for f in fields}})
    return {"runs": runs, "models": models, "arms_seed42": arms, "categories": categories}


SHIFT_BASE_SEED = 42


def decoding_seed_shift(tables: Mapping[str, Any]) -> Dict[str, Any]:
    """Descriptive: official score at decoding seed s minus seed 42, per model that has 42 and s complete, for the
    macros and every benchmark, and how consistent the shift is across models. A shift shared by every model
    (seeded per-request sampling drawing similar outputs from near-identical models) cancels in matched-seed pairs
    but moves single-seed levels."""
    off = {(e["served"], e["seed"]): e for e in tables["runs"] if e["complete"]}
    seeds = sorted({s for _, s in off if s != SHIFT_BASE_SEED})
    scopes = ("natural_9", "chart_diagram_synthetic_15", "overall_24") + tuple(f"benchmark:{k}" for k in ALL_KEYS)

    def val(e: Mapping[str, Any], scope: str) -> Optional[float]:
        return e["benchmarks"].get(scope.split(":", 1)[1]) if scope.startswith("benchmark:") else e.get(scope)

    out: Dict[str, Any] = {"base_seed": SHIFT_BASE_SEED, "seeds": {}}
    order = {m["served"]: i for i, m in enumerate(MODELS)}
    for s in seeds:
        models = sorted({sv for sv, sd in off if sd == s and (sv, SHIFT_BASE_SEED) in off}, key=lambda x: order.get(x, 999))
        if not models:
            continue
        rows = {}
        for scope in scopes:
            per = {}
            for sv in models:
                a, b = val(off[(sv, SHIFT_BASE_SEED)], scope), val(off[(sv, s)], scope)
                if a is not None and b is not None:
                    per[sv] = float(b) - float(a)
            v = list(per.values())
            rows[scope] = {"per_model": per, "mean": float(sum(v) / len(v)) if v else None,
                           "min": min(v, default=None), "max": max(v, default=None),
                           "n_up": sum(1 for x in v if x > 0), "n_down": sum(1 for x in v if x < 0), "n": len(v)}
        out["seeds"][str(s)] = {"models": models,
                                "labels": [MODEL_BY_SERVED.get(sv, {}).get("label", sv) for sv in models],
                                "rows": rows}
    return out


def macro_checks(comps: Sequence[Mapping[str, Any]], tables: Mapping[str, Any]) -> Dict[str, Any]:
    """Reconstructed paired macros vs the official summary macros averaged over the same decoding seeds."""
    off = {(e["served"], e["seed"]): e for e in tables["runs"] if e["complete"]}
    worst, n = 0.0, 0
    for c in comps:
        for ps in c.get("per_seed", []):
            for side, served in (("mean_x", ps["x_served"]), ("mean_y", ps["y_served"])):
                for g in ("natural_9", "chart_diagram_synthetic_15", "overall_24", "mmstar"):
                    o = sum(off[(served, sd)][g] for sd in ps["decoding_seeds"]) / len(ps["decoding_seeds"])
                    worst = max(worst, abs(ps["groups"][g][side] - o))
                    n += 1
    return {"compared": n, "max_abs_diff_points": worst,
            "note": "differences come only from official rounding (WeMath 2 decimals, TableVQABench 8-decimal "
                    "fractions, MathVerse/LogicVista 10 decimals)"}


def reconstruction_report(store: Store) -> Dict[str, Any]:
    per_bench = {}
    for k in ALL_KEYS:
        rows = []
        for served, by in store.runs.items():
            for seed, b in by.items():
                if k in b:
                    e = b[k]
                    rows.append({"served": served, "seed": seed, "valid": e["valid"], "abs_diff": e["abs_diff"],
                                 "official": e["official"], "reconstructed": e["reconstructed"],
                                 "latex_fallback_rows": e.get("latex_fallback_rows"),
                                 "local_name_hits": e.get("local_name_hits"),
                                 "extractor_unresolved_rate": e.get("extractor_unresolved_rate"),
                                 "sub_mismatch": (e.get("sub_checks") or {}).get("mismatch") or None})
        methods = Counter(b[k]["method"] for by in store.runs.values() for b in by.values() if k in b)
        per_bench[k] = {"methods": dict(methods), "runs": len(rows), "valid": sum(r["valid"] for r in rows),
                        "max_abs_diff": max((r["abs_diff"] for r in rows), default=None),
                        "invalid": [r for r in rows if not r["valid"]],
                        "latex_fallback_rows_max": max((r["latex_fallback_rows"] or 0 for r in rows), default=None)
                        if k == "mathvision" else None,
                        "local_name_hits": {f"{r['served']}@{r['seed']}": r["local_name_hits"] for r in rows}
                        if k == "mathvision" else None}
    return per_bench


# ============================================================================== provenance and fix-2/fix-3 counts


def _q25_label(served: str) -> str:
    for c in q25_checkpoints():
        if c["served"] == served:
            return f"{c['label']} step {c['step']}"
    m = MODEL_BY_SERVED.get(served)
    return f"{m['label']} (q25 probe)" if m else served


def provenance_table(inv: Mapping[str, Any], inv_q25: Mapping[str, Any], store: Store,
                     store_q25: Store) -> Dict[str, Any]:
    """Per complete run: code hashes of its generation and of the score slices it reports, and the fix-2 / fix-3
    counters from the summary and, independently, from the scorers' scores.json files."""
    rows = []
    for subset, iv, st_ in (("full", inv, store), (Q25, inv_q25, store_q25)):
        for r in iv["runs"]:
            if not r["complete"]:
                continue
            s = _read_json(Path(r["summary_path"]))
            meta = st_.meta.get((r["served"], r["seed"])) or {}
            sc = meta.get("scorer_counters")
            summ, scor, by = {}, {}, {}
            for k in EXCEPTION_KEYS:
                v = s.get(k)
                if v is None:
                    per = [b.get(k) for b in s.get("benchmarks", []) if b.get(k) is not None]
                    summ[k] = sum(int(x) for x in per) if per else None
                else:
                    summ[k] = _count_total(v)
                if sc is None:
                    scor[k] = None
                    by[k] = {}
                else:
                    vals = [c.get(k) for c in sc.values() if c.get(k) is not None]
                    scor[k] = int(sum(vals)) if vals else None
                    by[k] = {b: {"n": c[k], "indices": c.get(k + "_indices")} for b, c in sc.items() if c.get(k)}
            prov = meta.get("provenance") or {}
            m = MODEL_BY_SERVED.get(r["served"])
            label = (m["label"] if m else r["served"]) if subset == "full" else _q25_label(r["served"])
            rows.append({"served": r["served"], "label": label, "subset": subset, "seed": r["seed"],
                         "summary_created": s.get("created"), "generation_code": prov.get("generation"),
                         "extraction_code": prov.get("extraction"), "score_code": prov.get("score"),
                         "score_superseded": prov.get("score_superseded"), "summary_counters": summ,
                         "scorer_counters": scor, "by_benchmark": by, "item_table": sc is not None,
                         "counters_agree": sc is not None and all(summ[k] == scor[k] for k in EXCEPTION_KEYS)})
    order = {m["served"]: i for i, m in enumerate(MODELS)}
    qorder = {c["served"]: i for i, c in enumerate(q25_checkpoints())}
    rows.sort(key=lambda e: (e["subset"] != "full", order.get(e["served"], 999) if e["subset"] == "full"
                             else qorder.get(e["served"], -1), e["served"], e["seed"]))
    by_hash: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for e in rows:
        h = "+".join(sorted(e["score_code"] or {})) or "?"
        by_hash[e["subset"]][h].append(f"{e['served']}@{e['seed']}")
    fix3 = [{"served": e["served"], "subset": e["subset"], "seed": e["seed"], "label": e["label"], "key": k,
             "summary": e["summary_counters"][k], "scorer": e["scorer_counters"][k], "by_benchmark": e["by_benchmark"][k]}
            for e in rows for k in EXCEPTION_KEYS[1:] if e["summary_counters"][k] or e["scorer_counters"][k]]
    return {"runs": rows, "score_code_by_subset": {k: dict(v) for k, v in by_hash.items()},
            "fix3_nonzero": fix3, "counters_disagree": [f"{e['served']}@{e['seed']} ({e['subset']})" for e in rows
                                                        if e["item_table"] and not e["counters_agree"]]}


# ============================================================================== transfer over training (q25)


def restrict_to_subset(b: Mapping[str, Any], key: str, subset: Mapping[str, Any]
                       ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """A full run's benchmark entry restricted to the subset rows (WeMath: whole problem groups only)."""
    name = subset["name"]
    ords = subset["rows"].get(key) or {}
    if b["subset_digest"].get(name) != subset["digests"].get(key) or not b["subset_rows_match"].get(name):
        return None, "row identities differ from the subset"
    if key == "wemath":
        keep, partial = [], 0
        for g, members in (b.get("group_members") or {}).items():
            inside = sum(1 for o in members if o in ords)
            if inside == len(members):
                keep.append(g)
            elif inside:
                partial += 1
        if partial:
            return None, f"{partial} WeMath problem groups only partly in the subset"
        vals = {g: b["values"][g] for g in keep if g in b["values"]}
    else:
        vals = {i: v for i, v in b["values"].items() if int(i) in ords}
    out = dict(b)
    out.update({"values": vals, "digest": subset["digests"][key], "n_items": len(vals), "official": None,
                "restricted_from_full": True})
    return out, None


def build_q25_store(inv: Mapping[str, Any], inv_q25: Mapping[str, Any], store: Store, store_q25: Store,
                    subset: Mapping[str, Any]) -> Tuple[Store, Dict[str, Any]]:
    qs = Store()
    entries: Dict[str, Dict[str, Any]] = {}
    problems: List[Dict[str, Any]] = []
    for r in inv_q25["runs"]:
        if not r["complete"] or r["seed"] != Q25_DECODING_SEED:
            continue
        benches = store_q25.runs.get(r["served"], {}).get(r["seed"])
        if benches is None:
            continue  # recorded by the q25 item load
        if r.get("row_list_sha256") != subset["row_list_sha256"]:
            problems.append({"served": r["served"], "problem": "summary row_list_sha256 differs from the subset file"})
            continue
        bad = sorted(k for k, b in benches.items()
                     if not b["subset_exact"].get(subset["name"]) or b["digest"] != subset["digests"].get(k))
        if bad or set(benches) != set(ALL_KEYS):
            problems.append({"served": r["served"], "problem": f"rows differ from the subset: {bad}"})
            continue
        key = (Q25_PROBE_PREFIX + r["served"]) if r["served"] in MODEL_BY_SERVED else r["served"]
        qs.runs[key][r["seed"]] = benches
        s = _read_json(Path(r["summary_path"]))
        entries[key] = {"key": key, "served": r["served"], "label": _q25_label(r["served"]), "source": "q25 run",
                        "summary_path": r["summary_path"],
                        "invalid": sorted(k for k, b in benches.items() if not b["valid"]),
                        "official_macro": {g: (s.get("macro") or {}).get(g) for g in GROUPS}}
    want = {c["served"] for c in q25_checkpoints() if c["step"] == 300} | \
           {m["served"] for m in MODELS if m["arm"] in Q25_CONTEXT}
    for served in sorted(want):
        benches = store.runs.get(served, {}).get(Q25_DECODING_SEED)
        if benches is None:
            continue
        out = {}
        for k, b in benches.items():
            rb, err = restrict_to_subset(b, k, subset)
            if err:
                problems.append({"served": served, "benchmark": k, "problem": err})
            else:
                out[k] = rb
        if set(out) == set(ALL_KEYS):
            qs.runs[served][Q25_DECODING_SEED] = out
            m = MODEL_BY_SERVED.get(served)
            entries[served] = {"key": served, "served": served,
                               "label": f"{m['label']} step 300" if m and m["arm"] not in Q25_CONTEXT else m["label"],
                               "source": "full run restricted to q25 rows",
                               "invalid": sorted(k for k, b in out.items() if not b["valid"]), "official_macro": None}
    counts = {k: sorted({by[Q25_DECODING_SEED][k]["n_items"] for by in qs.runs.values()}) for k in ALL_KEYS}
    expected = {k: (132 if k == "wemath" else subset["selected"].get(k)) for k in ALL_KEYS}
    return qs, {"entries": entries, "problems": problems, "n_items": counts, "n_items_expected": expected,
                "n_items_consistent": all(len(v) == 1 and v[0] == expected[k] for k, v in counts.items()
                                          if k != "wemath") and len(counts["wemath"]) == 1}


def q25_levels(qs: Store, info: Mapping[str, Any], groups: Mapping[str, Sequence[str]]) -> Dict[str, Any]:
    np = _np()
    out = {}
    for key, by in qs.runs.items():
        b = by[Q25_DECODING_SEED]
        scores = {k: bench_score(b[k]) for k in ALL_KEYS}
        mac = {g: float(np.mean([scores[k] for k in groups[g]])) for g in Q25_GROUPS}
        e = info["entries"][key]
        off = e.get("official_macro")
        diff = None
        if off:
            diff = max(abs(mac[g] - off[g]) for g in GROUPS if off.get(g) is not None)
        bench_diff = None
        if e["source"] == "q25 run":
            bench_diff = max(b[k]["abs_diff"] for k in ALL_KEYS)
        out[key] = {"key": key, "served": e["served"], "label": e["label"], "source": e["source"], "macros": mac,
                    "benchmarks": scores, "official_macro": off, "max_abs_diff_official_macro": diff,
                    "max_abs_diff_official_benchmark": bench_diff, "invalid": e["invalid"]}
    return out


def q25_trajectory(levels: Mapping[str, Any]) -> Dict[str, Any]:
    np = _np()
    start = {scale: next(m["served"] for m in MODELS if m["arm"] == arm) for scale, arm in Q25_START.items()}
    rows, means = [], []
    for arm, code, prefix, scale, tseeds in Q25_ARMS:
        per_step: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        for t in tseeds:
            steps = {"0": (levels.get(start[scale]) or {}).get("macros")}
            for step in Q25_STEPS:
                steps[str(step)] = (levels.get(f"{prefix}-{code}-s{t}-{step}") or {}).get("macros")
            rows.append({"arm": arm, "scale": scale, "tseed": str(t), "start": start[scale], "steps": steps})
            for s, v in steps.items():
                if v is not None:
                    per_step[s].append(v)
        means.append({"arm": arm, "scale": scale, "training_seeds": [str(t) for t in tseeds],
                      "steps": {s: {"n": len(per_step.get(s, [])),
                                    "macros": {g: float(np.mean([v[g] for v in per_step[s]])) for g in Q25_GROUPS}
                                    if per_step.get(s) else None}
                                for s in ("0",) + tuple(str(x) for x in Q25_STEPS)}})
    return {"rows": rows, "arm_means": means}


def q25_regeneration(qs: Store, groups: Mapping[str, Sequence[str]], st: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Same model, same rows, same seed and request hashes: its own q25 probe run against its full run restricted
    to the q25 rows. The difference is generation nondeterminism plus re-judging, a yardstick for step-to-step
    changes (steps 100/200 come from q25 runs, step 300 from the full runs)."""
    np = _np()
    out = []
    for m in MODELS:
        probe = Q25_PROBE_PREFIX + m["served"]
        if probe not in qs.runs or m["served"] not in qs.runs:
            continue
        benches, used, _ = paired_benches(qs, probe, m["served"], [Q25_DECODING_SEED])
        stats, _ = seed_stats(benches, {g: groups[g] for g in Q25_GROUPS}, st, [hash_int("regen " + probe) % (2 ** 31), 0])
        same = {k: float(np.mean(np.concatenate([x == y for x, y in b.strata.values()]))) for k, b in benches.items()}
        n_same = sum(sum(int((x == y).sum()) for x, y in b.strata.values()) for b in benches.values())
        n_all = sum(b.n for b in benches.values())
        out.append({"model": m["label"], "x": probe, "y": m["served"], "groups": stats,
                    "identical_item_share": n_same / n_all, "identical_item_share_by_benchmark": same,
                    "min_identical_benchmark": min(same, key=same.get)})
    return out


def q25_section(inv: Mapping[str, Any], inv_q25: Mapping[str, Any], store: Store, store_q25: Store,
                subset: Mapping[str, Any], groups: Mapping[str, Sequence[str]], st: Mapping[str, Any], cc
                ) -> Dict[str, Any]:
    qs, info = build_q25_store(inv, inv_q25, store, store_q25, subset)
    levels = q25_levels(qs, info, groups)
    models = q25_models(qs.runs)
    ggroups = {g: groups[g] for g in Q25_GROUPS}
    comps = []
    for step in Q25_STEPS:
        for name, x, y in Q25_PAIRS:
            ya = f"{y}@0" if y in Q25_START.values() else f"{y}@{step}"
            c = run_comparison(qs, f"{name} @ step {step}", f"{x}@{step}", ya, "descriptive", ggroups, st, cc,
                               seeds=[Q25_DECODING_SEED], models=models)
            c.update({"pair": name, "step": step})
            comps.append(c)
            print(f"[compare_ext] q25 {name} @ {step}: {c['status']}", flush=True)
    present = sorted(k for k in qs.runs if not k.startswith(Q25_PROBE_PREFIX))
    # per-benchmark reconstruction of the q25 runs themselves (restricted full runs inherit the full-run check)
    recon = {"benchmarks": 0, "valid": 0, "max_abs_diff_excl_wemath": 0.0, "max_abs_diff_wemath": 0.0,
             "wemath_protocol_max_abs_diff": 0.0, "rekeyed_rows": 0, "rekeyed_as_position": True}
    for by in store_q25.runs.values():
        for b in by.values():
            for k, e in b.items():
                recon["benchmarks"] += 1
                recon["valid"] += int(bool(e["valid"]))
                fld = "max_abs_diff_wemath" if k == "wemath" else "max_abs_diff_excl_wemath"
                recon[fld] = max(recon[fld], e["abs_diff"])
                pc = e.get("protocol_check") or {}
                if pc.get("official") is not None:
                    recon["wemath_protocol_max_abs_diff"] = max(recon["wemath_protocol_max_abs_diff"],
                                                                abs(pc["official"] - pc["reconstructed"]))
                if e.get("archive_rows_rekeyed"):
                    recon["rekeyed_rows"] += e["archive_rows_rekeyed"]
                    recon["rekeyed_as_position"] = recon["rekeyed_as_position"] and bool(e["archive_rekeyed_as_position"])
    return {"subset": {"path": subset["path"], "row_list_sha256": subset["row_list_sha256"],
                       "rows": sum(len(v) for v in subset["rows"].values()), "selected": subset["selected"]},
            "decoding_seed": Q25_DECODING_SEED, "expected_q25_runs": inv_q25["expected"],
            "pending": inv_q25["pending"], "store_keys": present, "problems": info["problems"],
            "n_items": info["n_items"], "n_items_expected": info["n_items_expected"],
            "n_items_consistent": info["n_items_consistent"], "levels": levels, "trajectory": q25_trajectory(levels),
            "comparisons": comps, "regeneration": q25_regeneration(qs, groups, st), "q25_reconstruction": recon,
            "q25_runs_max_abs_diff_official_macro": max((v["max_abs_diff_official_macro"] for v in levels.values()
                                                         if v["max_abs_diff_official_macro"] is not None), default=None),
            "q25_runs_invalid": {k: v["invalid"] for k, v in levels.items() if v["invalid"]}}


def cmd_analyze(args: argparse.Namespace) -> int:
    np = _np()
    cc = _load_compare_cert()
    _, holm = cc.load_trace_tool(args.trace_tool_src)
    ext_root, out_dir = Path(args.ext_root), Path(args.out_dir)
    inv = discover(ext_root)
    tables = official_tables(inv)
    store = Store()
    store.load(out_dir, inv)
    groups = group_members(tables["categories"])
    st = {"n_boot": args.n_boot, "n_perm": args.n_perm, "seed": args.seed}
    comps = []
    for name, x, y, kind in COMPARISONS:
        t0 = time.time()
        c = run_comparison(store, name, x, y, kind, groups, st, cc)
        c["seconds"] = round(time.time() - t0, 1)
        comps.append(c)
        print(f"[compare_ext] {name}: {c['status']} ({c['seconds']} s)", flush=True)
    # sensitivity: P - G on decoding seed 42 only (the pre-registered single seed; 43/44 run for G and P only)
    c = run_comparison(store, "P - G (decoding seed 42 only)", "P", "G", "descriptive", groups, st, cc, seeds=[42])
    comps.append(c)
    inv_q25 = discover_q25(ext_root)
    hw = training_hardware([r["summary_path"] for r in list(inv["runs"]) + list(inv_q25["runs"]) if r["complete"]])
    annotate_hardware(comps, hw)
    checks = macro_checks(comps, tables)
    fam = apply_family(comps, "family", holm, args.alpha)
    sec = apply_family(comps, "secondary", holm, args.alpha)
    fs = format_shift(inv, store)
    flags = flag_pairs(fs, comps, args.flag_pp)
    # q25 runs, provenance, transfer over training
    store_q25 = Store()
    store_q25.load(out_dir, inv_q25, Q25)
    prov = provenance_table(inv, inv_q25, store, store_q25)
    subset_path = Path(args.subset_file) if args.subset_file else ext_root / Q25_SUBSET_FILE
    if subset_path.is_file():
        q25 = q25_section(inv, inv_q25, store, store_q25, load_subset(subset_path), groups, st, cc)
        q25["item_problems"] = store_q25.problems
        annotate_hardware(q25["comparisons"], hw)
    else:
        q25 = {"status": f"subset file missing: {subset_path}"}
    doc = {
        "version": VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": {"ext_root": str(ext_root), "n_boot": args.n_boot, "n_perm": args.n_perm, "seed": args.seed,
                   "alpha": args.alpha, "flag_pp": args.flag_pp, "natural": list(NATURAL_KEYS),
                   "chart_diagram_synthetic": list(SYNTHETIC_KEYS), "reported_alone": list(ALONE_KEYS),
                   "categories": tables["categories"], "models": list(MODELS),
                   "comparisons": [list(c) for c in COMPARISONS]},
        "inputs": {"runs": inv["runs"], "pending": inv["pending"], "item_problems": store.problems,
                   "row_identity": store.digests_consistent()},
        "reconstruction": reconstruction_report(store),
        "macro_checks": checks,
        "tables": tables,
        "decoding_seed_shift": decoding_seed_shift(tables),
        "training_hardware": hw,
        "families": {"external": fam, "secondary": sec},
        "comparisons": comps,
        "format_shift": {**fs, "pairs": flags, "flagged": [f for f in flags if f["flag"]]},
        "provenance": prov,
        "transfer_over_training_q25": q25,
    }
    _write_json(out_dir / "compare_ext.json", doc)
    (out_dir / "compare_ext.md").write_text(render_markdown(doc), encoding="utf-8")
    print(f"[compare_ext] wrote {out_dir / 'compare_ext.json'} and compare_ext.md", flush=True)
    return 0


# ============================================================================== markdown (from the JSON only)


def _f2(v: Any, signed: bool = False) -> str:
    if v is None:
        return "-"
    return f"{v:+.2f}" if signed else f"{v:.2f}"


def _pct(v: Any) -> str:
    return "-" if v is None else f"{100 * v:.1f}%"


def _p(v: Any) -> str:
    return "-" if v is None else f"{v:.3g}"


def _ci_txt(ci: Sequence[Any]) -> str:
    return "-" if not ci or ci[0] is None else f"[{ci[0]:+.2f}, {ci[1]:+.2f}]"


GROUP_TITLES = {"natural_9": "natural (9)", "chart_diagram_synthetic_15": "chart/diagram/synthetic (15)",
                "overall_24": "overall (24)", "mmstar": "MMStar (alone)",
                "sens:chart_diagram_synthetic_13_excl_puzzle_letters": "chart/diag/syn without PuzzleVQA+VisualPuzzles (13)",
                "sens:overall_22_excl_puzzle_letters": "overall without PuzzleVQA+VisualPuzzles (22)"}


def render_markdown(doc: Mapping[str, Any]) -> str:
    cfg = doc["config"]
    L: List[str] = ["# External transfer on trace_eval_v1 (RQ2)", "",
                    f"Generated by `compare_ext.py` {doc['version']} at {doc['created']} from `{cfg['ext_root']}`. "
                    "Every number below is read from `compare_ext.json`. Scores are official benchmark scores in "
                    "points (0-100); differences are X - Y in points. Macros are means of benchmark scores (DESIGN 6 "
                    "groups); MMStar is natural and is also listed alone.", ""]
    L += ["## Summary (generated)", ""]
    fam = doc["families"]["external"]
    for t in fam["tests"]:
        r = t["result"]
        if r is None:
            L.append(f"- External family, {t['comparison']} on the {GROUP_TITLES[t['group']]} macro: pending.")
            continue
        extra = (f"; sign agreement {r['sign_agree']}/{r['K']} (need {r['sign_needed']}), per-seed "
                 + ", ".join(_f2(d, True) for d in r["per_seed_delta"])) if r["K"] > 1 else ""
        L.append(f"- External family, {t['comparison']} on the {GROUP_TITLES[t['group']]} macro: "
                 f"{_f2(r['delta'], True)} points (P {_f2(r['mean_x'])}, G {_f2(r['mean_y'])}), 95% CI {_ci_txt(r['ci'])} "
                 f"({r['ci_scope']}), p {_p(r['p'])}, Holm p {_p(t.get('holm_p'))}{extra}: {t.get('decision')}.")
    for t in doc["families"]["secondary"]["tests"]:
        r = t["result"]
        if r is None:
            continue
        L.append(f"- Secondary, {t['comparison']} on the {GROUP_TITLES[t['group']]} macro: {_f2(r['delta'], True)} "
                 f"{_ci_txt(r['ci'])}, K={r['K']}, Holm p {_p(t.get('holm_p'))}: {t.get('decision')}.")
    for name in ("A - S", "G - S", "P - S", "S - base-3B", "legacy-3B - base-3B", "P - legacy-3B",
                 "P_gated - G_gated", "P7 - G7", "G7 - S7", "P7 - S7", "S7 - base-7B"):
        c = next((x for x in doc["comparisons"] if x["name"] == name and x.get("status") == "ok"), None)
        if c:
            nat, syn = c["combined"]["natural_9"], c["combined"]["chart_diagram_synthetic_15"]
            sen = c["combined"]["sens:chart_diagram_synthetic_13_excl_puzzle_letters"]
            tail = (f"{nat['sign_agree']}/{c['K']} and {syn['sign_agree']}/{c['K']} seeds agree" if c["K"] > 1 else
                    f"1 training seed, sign-flip p {_p(nat['p'])} / {_p(syn['p'])}")
            L.append(f"- Descriptive {name}: natural {_f2(nat['delta'], True)} {_ci_txt(nat['ci'])}, chart/diagram/"
                     f"synthetic {_f2(syn['delta'], True)} {_ci_txt(syn['ci'])}; without PuzzleVQA+VisualPuzzles "
                     f"{_f2(sen['delta'], True)} {_ci_txt(sen['ci'])} (K={c['K']}; {tail}{_hw_note(c)}).")
    sh = doc.get("decoding_seed_shift") or {}
    for s, d in (sh.get("seeds") or {}).items():
        rows = d["rows"]
        common = [(k.split(":", 1)[1], r) for k, r in rows.items() if k.startswith("benchmark:") and r["n"]
                  and (r["n_down"] == r["n"] or r["n_up"] == r["n"]) and abs(r["mean"]) >= 3.0]
        L.append(f"- Decoding seed {s} vs {sh['base_seed']} ({len(d['models'])} models with both; descriptive): natural "
                 f"{_f2(rows['natural_9']['mean'], True)} [{_f2(rows['natural_9']['min'], True)}, "
                 f"{_f2(rows['natural_9']['max'], True)}], chart/diagram/synthetic "
                 f"{_f2(rows['chart_diagram_synthetic_15']['mean'], True)} [{_f2(rows['chart_diagram_synthetic_15']['min'], True)}, "
                 f"{_f2(rows['chart_diagram_synthetic_15']['max'], True)}]; benchmarks moving the same way in every model "
                 "by at least 3 points on average: " + (", ".join(f"{k} {_f2(r['mean'], True)}" for k, r in common) or "none")
                 + ".")
    pg = next((x for x in doc["comparisons"] if x["name"] == "P - G" and x.get("status") == "ok"), None)
    if pg:
        L.append("- Decoding seeds pooled in P - G per training seed: " + "; ".join(
            f"{'/'.join(ps['tseeds'])}: {','.join(map(str, ps['decoding_seeds']))}" for ps in pg["per_seed"]) + ".")
    pv = doc.get("provenance") or {}
    if pv:
        L.append("- Scorer code of the reported score slices: " + "; ".join(
            f"{sub}: " + ", ".join(f"{h} ({len(v)} runs)" for h, v in sorted(d.items()))
            for sub, d in pv["score_code_by_subset"].items()) + ". Rows scored incorrect by fix-3: "
            + (", ".join(f"{f['label']} ({f['subset']}, seed {f['seed']}) {f['key']} = {f['summary']}"
                         for f in pv["fix3_nonzero"]) or "none") +
            f"; summary and scores.json counters disagree for {len(pv['counters_disagree'])} runs.")
    q = doc.get("transfer_over_training_q25") or {}
    if q.get("comparisons"):
        L.append(f"- Transfer over training (q25, descriptive): {len(q['store_keys'])} checkpoint/context models; "
                 f"{len(q['pending'])} expected q25 runs not complete yet (listed in that section).")
        for name in ("P - G", "P - A", "G - A", "P_gated - G_gated", "P7 - G7"):
            cells = []
            for c in q["comparisons"]:
                if c["pair"] != name:
                    continue
                if c.get("status") != "ok":
                    cells.append(f"step {c['step']} pending")
                    continue
                nat, syn = c["combined"]["natural_9"], c["combined"]["chart_diagram_synthetic_15"]
                cells.append(f"step {c['step']} natural {_f2(nat['delta'], True)} {_ci_txt(nat['ci'])}, chart "
                             f"{_f2(syn['delta'], True)} {_ci_txt(syn['ci'])} (K={c['K']})")
            L.append(f"  - {name} on q25: " + "; ".join(cells) + ".")
        for m in q["trajectory"]["arm_means"]:
            st_ = m["steps"]
            cells = [f"{s} {_f2(st_[s]['macros']['natural_9'])} / {_f2(st_[s]['macros']['chart_diagram_synthetic_15'])}"
                     if st_[s]["macros"] else f"{s} -" for s in ("0", "100", "200", "300")]
            L.append(f"  - {m['arm']} (training seeds {','.join(m['training_seeds'])}, mean) natural / chart macro by "
                     f"step: " + "; ".join(cells) + ".")
        for r in q.get("regeneration") or []:
            g = r["groups"]
            L.append(f"  - Regeneration yardstick {r['model']} (own q25 run - full run on q25 rows): natural "
                     f"{_f2(g['natural_9']['delta'], True)} {_ci_txt(g['natural_9']['ci'])}, chart "
                     f"{_f2(g['chart_diagram_synthetic_15']['delta'], True)} {_ci_txt(g['chart_diagram_synthetic_15']['ci'])}; "
                     f"identical item values {_pct(r['identical_item_share'])}.")
    fl = doc["format_shift"]["flagged"]
    fam_fl = [f for f in fl if f["kind"] in ("family", "secondary")]
    L.append(f"- Format shift: {len(fam_fl)} rate differences above {cfg['flag_pp']} pp within the family/secondary "
             f"pairs; {len(fl)} across all compared pairs (see the format-shift section).")
    mc = doc["macro_checks"]
    L.append(f"- Reconstruction: per-item tables reproduce the official macros within {mc['max_abs_diff_points']:.2g} "
             f"points over {mc['compared']} paired-macro checks.")
    L.append("")
    # ---------------------------------------------------------------- inputs
    L += ["## Inputs and pending", ""]
    comp = [r for r in doc["inputs"]["runs"] if r["complete"]]
    L.append(f"Complete runs used: {len(comp)} (model, decoding seed) summaries.")
    L.append("")
    if doc["inputs"]["pending"]:
        L += ["| pending model | served | decoding seed | status |", "|---|---|---|---|"]
        for p in doc["inputs"]["pending"]:
            L.append(f"| {p['label']} | {p['served']} | {p['seed']} | {p['status']} |")
        L.append("")
    if doc["inputs"]["item_problems"]:
        L += ["Item tables not usable:", ""]
        for p in doc["inputs"]["item_problems"]:
            L.append(f"- {p['served']} seed {p['seed']}: {p['problem']}")
        L.append("")
    # ---------------------------------------------------------------- family
    fam = doc["families"]["external"]
    L += ["## Pre-registered external family (DESIGN 8): P - G", "",
          "Natural macro and chart/diagram/synthetic macro, Holm across the two. Item values pool the decoding seeds "
          "both runs have; training seeds are matched. With K > 1 training seeds the Holm input is the two-level "
          "bootstrap p and a claim needs Holm p < alpha and the sign rule; `compare_cert` convention = unadjusted "
          "two-level CI excluding 0 plus the sign rule.", ""]
    L += _family_table(fam, cfg["alpha"])
    sec = doc["families"]["secondary"]
    L += ["", "## Secondary family: P - A and G - A (Holm across the four)", ""]
    L += _family_table(sec, cfg["alpha"])
    # ---------------------------------------------------------------- per-seed detail of the family
    L += ["", "### Per training seed (family and secondary)", "",
          "| comparison | training seeds | decoding seeds | group | X | Y | delta | 95% CI (items) | sign-flip p |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c in doc["comparisons"]:
        if c["kind"] not in ("family", "secondary") or c.get("status") != "ok":
            continue
        for ps in c["per_seed"]:
            for g in FAMILY_GROUPS + ("overall_24", "mmstar"):
                r = ps["groups"][g]
                L.append(f"| {c['name']} | {'/'.join(t or '-' for t in ps['tseeds'])} | {','.join(map(str, ps['decoding_seeds']))} | "
                         f"{GROUP_TITLES[g]} | {_f2(r['mean_x'])} | {_f2(r['mean_y'])} | {_f2(r['delta'], True)} | "
                         f"{_ci_txt(r['ci'])} | {_p(r['perm_p'])} |")
    # ---------------------------------------------------------------- descriptive comparisons
    L += ["", "## All comparisons (macros; descriptive except the family rows above)", "",
          "K = training-seed pairs; with K > 1 the CI is the two-level bootstrap and `k/K` the sign agreement; with "
          "K = 1 the CI reflects instance sampling only and p is the item sign-flip p (unadjusted).", "",
          "| comparison | kind | K | natural (9) | chart/diag/syn (15) | overall (24) | MMStar | chart/diag/syn w/o PuzzleVQA+VisualPuzzles (13, sensitivity) | status |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c in doc["comparisons"]:
        if c.get("status") != "ok":
            miss = ", ".join(c.get("missing_x", []) + c.get("missing_y", []))
            L.append(f"| {c['name']} | {c['kind']} | - | - | - | - | - | - | pending ({miss or c.get('note', '')}) |")
            continue
        cells = []
        for g in ("natural_9", "chart_diagram_synthetic_15", "overall_24", "mmstar",
                  "sens:chart_diagram_synthetic_13_excl_puzzle_letters"):
            r = c["combined"][g]
            tail = f" {r['sign_agree']}/{r['K']}" if r["K"] > 1 else f" p={_p(r['p'])}"
            cells.append(f"{_f2(r['delta'], True)} {_ci_txt(r['ci'])}{tail}")
        inv = sorted({x for ps in c["per_seed"] for x in ps["reconstruction_invalid"]})
        status = "ok" + (f"; unvalidated reconstruction: {', '.join(inv)}" if inv else "") + _hw_note(c)
        L.append(f"| {c['name']} | {c['kind']} | {c['K']} | " + " | ".join(cells) + f" | {status} |")
    # ---------------------------------------------------------------- categories and benchmarks for P - G
    pg = next((c for c in doc["comparisons"] if c["name"] == "P - G" and c.get("status") == "ok"), None)
    if pg:
        L += ["", "### P - G by category and benchmark (descriptive, unadjusted)", "",
              "| scope | P | G | delta | 95% CI | per-seed deltas | sign agree |", "|---|---|---|---|---|---|---|"]
        for g, r in pg["combined"].items():
            if not (g.startswith("category:") or g.startswith("benchmark:")):
                continue
            ps = ", ".join(_f2(d, True) for d in r.get("per_seed_delta", [])) or "-"
            sa = f"{r['sign_agree']}/{r['K']}" if r["K"] > 1 else f"p={_p(r['p'])}"
            L.append(f"| {g.split(':', 1)[1]} | {_f2(r['mean_x'])} | {_f2(r['mean_y'])} | {_f2(r['delta'], True)} | "
                     f"{_ci_txt(r['ci'])} | {ps} | {sa} |")
    # ---------------------------------------------------------------- cross-model tables
    t = doc["tables"]
    L += ["", "## Cross-model table (official summaries, complete runs)", "",
          "### Macros per model and decoding seed", "",
          "| model | served | seed | overall (24) | natural (9) | chart/diag/syn (15) | MMStar | category mean |",
          "|---|---|---|---|---|---|---|---|"]
    for e in t["runs"]:
        if not e["complete"]:
            continue
        L.append(f"| {e['label']} | {e['served']} | {e['seed']} | {_f2(e['overall_24'])} | {_f2(e['natural_9'])} | "
                 f"{_f2(e['chart_diagram_synthetic_15'])} | {_f2(e['mmstar'])} | {_f2(e['overall_category_mean'])} |")
    if t["arms_seed42"]:
        L += ["", "### 3B RL arms over training seeds (decoding seed 42)", "",
              "| arm | training seeds | overall (24) | natural (9) | chart/diag/syn (15) | MMStar | natural min-max | chart min-max |",
              "|---|---|---|---|---|---|---|---|"]
        for a in t["arms_seed42"]:
            m = a["mean"]
            L.append(f"| {a['arm']} | {','.join(a['training_seeds'])} | {_f2(m['overall_24'])} | {_f2(m['natural_9'])} | "
                     f"{_f2(m['chart_diagram_synthetic_15'])} | {_f2(m['mmstar'])} | "
                     f"{_f2(a['min']['natural_9'])}-{_f2(a['max']['natural_9'])} | "
                     f"{_f2(a['min']['chart_diagram_synthetic_15'])}-{_f2(a['max']['chart_diagram_synthetic_15'])} |")
    models = t["models"]
    cats = list(t["categories"])
    L += ["", "### Categories (mean over each model's complete decoding seeds)", "",
          "| model | seeds | " + " | ".join(cats) + " |", "|---|---|" + "---|" * len(cats)]
    for mm in models:
        v = mm["mean_over_decoding_seeds"]["categories"]
        L.append(f"| {mm['label']} | {','.join(map(str, mm['seeds']))} | " + " | ".join(_f2(v.get(c)) for c in cats) + " |")
    L += ["", "### Per benchmark (mean over each model's complete decoding seeds)", "",
          "| benchmark | group | " + " | ".join(mm["label"] for mm in models) + " |", "|---|---|" + "---|" * len(models)]
    for k in ALL_KEYS:
        grp = "nat" if k in NATURAL_KEYS else "syn"
        L.append(f"| {k} | {grp} | " + " | ".join(_f2(mm["mean_over_decoding_seeds"]["benchmarks"][k]) for mm in models) + " |")
    if doc.get("decoding_seed_shift"):
        L += _render_seed_shift(doc["decoding_seed_shift"])
    # ---------------------------------------------------------------- format shift
    fs = doc["format_shift"]
    L += ["", "## Format-shift view (DESIGN 10)", "",
          "Deterministic-extraction failure (no answer tag, box, JSON answer, answer marker or bare short answer), "
          "judge-needed proxy (rows of the judge-extraction routes without a deterministic answer), MathVision / "
          "MathVista protocol judge rates (official prefetch failed), the share of PuzzleVQA / VisualPuzzles rows the "
          "official option-letter extractor leaves unresolved (scored wrong), certificate-shaped JSON and length-cap "
          "rates; mean over each model's complete decoding seeds. Counters per decoding seed: fix-2 evaluator "
          "exceptions scored incorrect and the fix-3 counters (`-` = the summary predates fix-3, i.e. it was scored "
          "under 16d37ed061fd; such a scoring cannot contain a fix-3 row, because one would have failed its slice).", "",
          "| model | seeds | det. fail | judge-needed (proxy) | MathVision judge | MathVista judge | PuzzleVQA letter unresolved | "
          "VisualPuzzles letter unresolved | cert JSON | length cap | fix-2 exceptions (per seed) | "
          "fix-3 MathVerse retries failed | fix-3 extraction failed |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in fs["models"]:
        def ctr(k):
            c = e["counters"][k]
            if not c["reported"]:
                return "-"
            return ", ".join(f"{s}:{'-' if v is None else v}" for s, v in c["per_seed"].items())
        L.append(f"| {e['label']} | {','.join(map(str, e['seeds']))} | {_pct(e['det_fail_rate'])} | "
                 f"{_pct(e['judge_needed_rate_proxy'])} | {_pct(e['judge_needed_protocol_mathvision'])} | "
                 f"{_pct(e['judge_needed_protocol_mathvista'])} | {_pct(e['abcd_unresolved_puzzlevqa'])} | "
                 f"{_pct(e['abcd_unresolved_visualpuzzles'])} | {_pct(e['cert_json_rate'])} | {_pct(e['length_cap_rate'])} | "
                 f"{ctr('evaluator_exception_scored_incorrect')} | {ctr('mathverse_all_retries_failed_scored_incorrect')} | "
                 f"{ctr('extraction_failed_scored_incorrect')} |")
    flagged = fs["flagged"]
    L += ["", f"Pairs flagged (|difference| > {cfg['flag_pp']} pp in a rate, decoding seeds the comparison used): "
          f"{len(flagged)} of {len(fs['pairs'])} checked.", ""]
    fam_names = {c["name"] for c in doc["comparisons"] if c["kind"] in ("family", "secondary")}
    shown = [f for f in fs["pairs"] if f["comparison"] in fam_names or f["flag"]]
    if shown:
        L += ["| comparison | training seeds | rate | X | Y | X - Y (pp) | flag |", "|---|---|---|---|---|---|---|"]
        for f in shown:
            L.append(f"| {f['comparison']} | {'/'.join(t or '-' for t in f['tseeds'])} | {f['rate']} | {_pct(f['x_rate'])} | "
                     f"{_pct(f['y_rate'])} | {f['diff_pp']:+.2f} | {'FLAG' if f['flag'] else ''} |")
    # ---------------------------------------------------------------- reconstruction
    L += ["", "## Per-item reconstruction check", "",
          "Each run's per-item values must reproduce the official benchmark score (and the evaluator's sub-totals "
          "where it reports them).", "",
          "| benchmark | method | runs valid | max abs diff (points) | note |", "|---|---|---|---|---|"]
    for k, r in doc["reconstruction"].items():
        note = ""
        if k == "mathvision":
            nh = r["local_name_hits"] or {}
            note = (f"latex2sympy fallback unevaluated rows (max per run): {r['latex_fallback_rows_max']}; hits that "
                    f"only the official eval-locals quirk grants (response `a`/`b` etc.): total {sum(v or 0 for v in nh.values())}, "
                    f"max per run {max((v or 0 for v in nh.values()), default=0)}")
        if r["invalid"]:
            note += ("; " if note else "") + "invalid: " + ", ".join(f"{x['served']}@{x['seed']}" for x in r["invalid"])
        L.append(f"| {k} | {', '.join(r['methods'])} | {r['valid']}/{r['runs']} | "
                 f"{'-' if r['max_abs_diff'] is None else f'{r['max_abs_diff']:.2g}'} | {note} |")
    rid = doc["inputs"]["row_identity"]
    multi = [k for k, v in rid.items() if v["distinct_row_sets"] > 1]
    L += ["", "Row identities (source ordinal and row hash) are identical across all runs for "
          f"{len(rid) - len(multi)} of {len(rid)} benchmarks" + (f"; differing: {', '.join(multi)}" if multi else "") + "."]
    if doc.get("provenance"):
        L += _render_provenance(doc["provenance"])
    if (doc.get("transfer_over_training_q25") or {}).get("levels") is not None:
        L += _render_q25(doc["transfer_over_training_q25"])
    return "\n".join(L) + "\n"


def _short_gpu(name: Optional[str]) -> str:
    n = str(name or "?")
    return n.replace("NVIDIA ", "").replace(" Blackwell", "").replace(" Workstation Edition", "")


def _hw_note(c: Mapping[str, Any]) -> str:
    cr = c.get("hardware_crossed") or []
    if not cr:
        return ""
    return "; training hardware differs (not a matched pair): " + ", ".join(
        f"{'/'.join(t or '-' for t in x['tseeds'])} {_short_gpu(x['x'])} vs {_short_gpu(x['y'])}" for x in cr)


def _render_seed_shift(sh: Mapping[str, Any]) -> List[str]:
    L = ["", "## Decoding-seed shift (descriptive)", "",
         f"Official score at decoding seed s minus seed {sh['base_seed']}, per model with both complete: mean over "
         "those models [min, max] and how many moved down / up. A shift that every model shares cancels in the "
         "matched-seed pairs above but moves single-seed levels (q25 and all K = 1 comparisons use seed 42 only).", ""]
    if not sh["seeds"]:
        return L + ["No model has a second complete decoding seed."]
    seeds = list(sh["seeds"])
    for s in seeds:
        L.append(f"- seed {s}: " + ", ".join(sh["seeds"][s]["labels"]))
    L += ["", "| scope | " + " | ".join(f"seed {s} - {sh['base_seed']}: mean [min, max] (down/up)" for s in seeds) + " |",
          "|---|" + "---|" * len(seeds)]
    scopes = list(next(iter(sh["seeds"].values()))["rows"])
    for scope in scopes:
        cells = []
        for s in seeds:
            r = sh["seeds"][s]["rows"].get(scope) or {}
            cells.append("-" if r.get("mean") is None else
                         f"{_f2(r['mean'], True)} [{_f2(r['min'], True)}, {_f2(r['max'], True)}] ({r['n_down']}/{r['n_up']})")
        name = GROUP_TITLES.get(scope, scope.split(":", 1)[-1])
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    return L


def _hashes(d: Optional[Mapping[str, int]]) -> str:
    return ", ".join(f"{h} x{n}" for h, n in sorted((d or {}).items())) or "-"


def _render_provenance(pv: Mapping[str, Any]) -> List[str]:
    def c2(e, k):
        a, b = e["summary_counters"][k], e["scorer_counters"][k]
        return f"{'-' if a is None else a} / {'-' if b is None else b}"

    L = ["", "## Provenance: scorer code and fix-2 / fix-3 counts per run", "",
         "Code hash (`final25_code_hash`, 12 characters) of the newest archive slice per benchmark: generation, "
         "and the score slices the summary reports (`x n` = number of benchmarks). Superseded score slices are older "
         "scorings kept in `hf_archive/` (e.g. a stale pre-fix-3 campaign moved to `scoring/stale/`). Counters are "
         "`summary / scores.json` (MathVerse `artifacts`, MME-Reasoning `evaluation`); `-` = the file predates that "
         "counter. fix-2: MME-Reasoning official-evaluator exceptions scored incorrect; fix-3: a judge output still "
         "unusable after its 5 attempts scored incorrect (MathVerse score judgement, MME-Reasoning answer extraction).",
         "", "| model | subset | seed | generation code | score code | superseded score slices | fix-2 | "
         "fix-3 MathVerse | fix-3 MME extraction | counters agree |", "|---|---|---|---|---|---|---|---|---|---|"]
    for e in pv["runs"]:
        L.append(f"| {e['label']} | {e['subset']} | {e['seed']} | {_hashes(e['generation_code'])} | "
                 f"{_hashes(e['score_code'])} | {_hashes(e['score_superseded'])} | "
                 f"{c2(e, 'evaluator_exception_scored_incorrect')} | "
                 f"{c2(e, 'mathverse_all_retries_failed_scored_incorrect')} | "
                 f"{c2(e, 'extraction_failed_scored_incorrect')} | "
                 f"{'yes' if e['counters_agree'] else ('no item table' if not e['item_table'] else 'NO')} |")
    L += ["", "Runs per score code: " + "; ".join(
        f"{sub}: " + ", ".join(f"{h}: {len(v)}" for h, v in sorted(d.items())) for sub, d in pv["score_code_by_subset"].items()) + "."]
    full_new = [r for h, v in pv["score_code_by_subset"].get("full", {}).items() if "d9319e7db480" in h for r in v]
    L.append("Full runs scored under fix-3 (d9319e7db480): " + (", ".join(full_new) or "none") + ".")
    if pv["fix3_nonzero"]:
        L += ["", "Rows scored incorrect by fix-3:", ""]
        for f in pv["fix3_nonzero"]:
            det = "; ".join(f"{b}: {v['n']} (index {', '.join(v['indices'] or [])})" for b, v in f["by_benchmark"].items())
            L.append(f"- {f['label']} ({f['subset']}, seed {f['seed']}), `{f['key']}` = {f['summary']} ({det}).")
    else:
        L += ["", "No run has a row scored incorrect by fix-3."]
    if pv["counters_disagree"]:
        L.append("Summary and scores.json counters disagree for: " + ", ".join(pv["counters_disagree"]) + ".")
    return L


def _render_q25(q: Mapping[str, Any]) -> List[str]:
    G3 = ("natural_9", "chart_diagram_synthetic_15", "overall_24")
    L = ["", "## Transfer over training on trace_eval_v1_q25 (descriptive)", "",
         f"Steps 100 and 200 come from each checkpoint's own q25 run; step 300, S (step 0: RL starts from the M2 SFT "
         f"checkpoint) and base come from their full decoding-seed-{q['decoding_seed']} runs restricted to the same "
         f"q25 rows ({q['subset']['rows']} rows, row list sha256 `{(q['subset']['row_list_sha256'] or '')[:12]}`; "
         "WeMath keeps whole problem groups, 132 of 525). Every q25 run's row identities equal the subset's, and "
         "every full run's generation contains them with the same row hashes. All levels are reconstructed from per-item "
         "tables; macros are means of the q25 benchmark scores (WeMath = mean over its 132 groups, the summaries' "
         "rescaled score). Deltas are paired on items at decoding seed 42, K = training seeds (two-level bootstrap "
         "when K > 1, `k/K` = sign agreement; K = 1: instance sampling only, sign-flip p). Nothing here is "
         "pre-registered or Holm-adjusted.", ""]
    rc = q["q25_reconstruction"]
    L.append(f"q25 runs used: {len([k for k, v in q['levels'].items() if v['source'] == 'q25 run'])}. Their per-item "
             f"tables reproduce {rc['valid']} of {rc['benchmarks']} official benchmark scores (max abs diff "
             f"{_p(rc['max_abs_diff_excl_wemath'])} points without WeMath; WeMath {_p(rc['max_abs_diff_wemath'])}, "
             f"its 2-decimal official 525-group value rescaled by 525/132, and {_p(rc['wemath_protocol_max_abs_diff'])} "
             f"on the unrescaled value) and the official macros within {_p(q['q25_runs_max_abs_diff_official_macro'])} "
             f"points. In q25 runs the archive extraction/score rows of the deterministic VLMEvalKit route are numbered "
             f"by position in the subset, not by source ordinal: {rc['rekeyed_rows']} rows were re-keyed through "
             f"(source_index, row hash), all of them position = rank of the source ordinal: "
             f"{'yes' if rc['rekeyed_as_position'] else 'NO'}. Items per benchmark identical across all q25 models: "
             f"{'yes' if q['n_items_consistent'] else 'NO'}"
             + (f"; reconstruction not validated: {q['q25_runs_invalid']}" if q["q25_runs_invalid"] else "") + ".")
    if q["pending"]:
        L += ["", "Expected q25 checkpoint runs not complete yet (picked up on the next run):", "",
              "| checkpoint | served | decoding seed | status |", "|---|---|---|---|"]
        for p in q["pending"]:
            L.append(f"| {p['label']} | {p['served']} | {p['seed']} | {p['status']} |")
    else:
        L += ["", f"All {len(q['expected_q25_runs'])} expected step-100/200 q25 runs are complete."]
    if q["problems"] or q.get("item_problems"):
        L += ["", "Not usable: " + "; ".join(f"{p.get('served')}: {p.get('problem')}"
                                             for p in list(q["problems"]) + list(q.get("item_problems") or []))]
    steps = ["0"] + [str(s) for s in (100, 200, 300)]
    for g in G3:
        L += ["", f"### {GROUP_TITLES[g]} macro on q25 by training step", "",
              "| arm | training seed | " + " | ".join(f"step {s}" + (" (S)" if s == "0" else "") for s in steps) + " |",
              "|---|---|" + "---|" * len(steps)]
        for r in q["trajectory"]["rows"]:
            L.append(f"| {r['arm']} | {r['tseed']} | " + " | ".join(
                _f2((r["steps"].get(s) or {}).get(g)) for s in steps) + " |")
        for m in q["trajectory"]["arm_means"]:
            if len(m["training_seeds"]) < 2:
                continue
            L.append(f"| {m['arm']} mean | " + ",".join(m["training_seeds"]) + " | " + " | ".join(
                (f"{_f2(m['steps'][s]['macros'][g])} (n={m['steps'][s]['n']})" if m["steps"][s]["macros"] else "-")
                for s in steps) + " |")
    L += ["", "### Context models on q25", "", "| model | source | natural (9) | chart/diag/syn (15) | overall (24) | MMStar |",
          "|---|---|---|---|---|---|"]
    for k, v in q["levels"].items():
        if v["source"] == "q25 run" and not k.startswith(Q25_PROBE_PREFIX):
            continue
        if v["label"] and "step 300" in v["label"]:
            continue
        mm = v["macros"]
        L.append(f"| {v['label']} | {v['source']} | {_f2(mm['natural_9'])} | {_f2(mm['chart_diagram_synthetic_15'])} | "
                 f"{_f2(mm['overall_24'])} | {_f2(mm['mmstar'])} |")
    if q["regeneration"]:
        L += ["", "### Regeneration yardstick: same model, own q25 run vs full run restricted to q25", "",
              "Same rows, decoding seed and request hashes; the difference is generation nondeterminism (hardware, "
              "batching) plus a second judge pass, the noise floor between steps 100/200 (q25 runs) and 300 "
              "(restricted full runs).", "",
              "| model | identical item values | natural (9) | chart/diag/syn (15) | overall (24) |", "|---|---|---|---|---|"]
        for r in q["regeneration"]:
            cells = [f"{_f2(r['groups'][g]['delta'], True)} {_ci_txt(r['groups'][g]['ci'])} p={_p(r['groups'][g]['perm_p'])}"
                     for g in G3]
            L.append(f"| {r['model']} | {_pct(r['identical_item_share'])} (lowest: {r['min_identical_benchmark']} "
                     f"{_pct(r['identical_item_share_by_benchmark'][r['min_identical_benchmark']])}) | " + " | ".join(cells) + " |")
    L += ["", "### Paired deltas per step (X - Y on q25 items, decoding seed 42)", "",
          "| comparison | step | K | natural (9) | chart/diag/syn (15) | overall (24) | chart w/o puzzles (13) | status |",
          "|---|---|---|---|---|---|---|---|"]
    for c in q["comparisons"]:
        if c.get("status") != "ok":
            miss = ", ".join(c.get("missing_x", []) + c.get("missing_y", []))
            L.append(f"| {c['pair']} | {c['step']} | - | - | - | - | - | pending ({miss}) |")
            continue
        cells = []
        for g in G3 + ("sens:chart_diagram_synthetic_13_excl_puzzle_letters",):
            r = c["combined"][g]
            tail = f" {r['sign_agree']}/{r['K']}" if r["K"] > 1 else f" p={_p(r['p'])}"
            cells.append(f"{_f2(r['delta'], True)} {_ci_txt(r['ci'])}{tail}")
        miss = c.get("missing_x", []) + c.get("missing_y", [])
        L.append(f"| {c['pair']} | {c['step']} | {c['K']} | " + " | ".join(cells) + " | "
                 + ("ok" + (f"; missing {', '.join(miss)}" if miss else "") + _hw_note(c)) + " |")
    return L


def _family_table(fam: Mapping[str, Any], alpha: float) -> List[str]:
    L = ["| comparison | group | K | X | Y | delta | 95% CI | CI scope | p (test) | Holm p | sign agree | per-seed deltas | "
         "decision | compare_cert convention |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in fam["tests"]:
        r = t["result"]
        if r is None:
            L.append(f"| {t['comparison']} | {GROUP_TITLES[t['group']]} | - | - | - | - | - | - | - | - | - | - | "
                     f"pending | - |")
            continue
        sa = f"{r['sign_agree']}/{r['K']} (need {r['sign_needed']})" if r["K"] > 1 else "-"
        psd = ", ".join(_f2(d, True) for d in r.get("per_seed_delta", [])) or "-"
        L.append(f"| {t['comparison']} | {GROUP_TITLES[t['group']]} | {r['K']} | {_f2(r['mean_x'])} | {_f2(r['mean_y'])} | "
                 f"{_f2(r['delta'], True)} | {_ci_txt(r['ci'])} | {r['ci_scope']} | {_p(r['p'])} ({r['test']}) | "
                 f"{_p(t.get('holm_p'))} | {sa} | {psd} | {t.get('decision', '-')} | "
                 f"{t.get('decision_compare_cert_convention', '-')} |")
    if not fam["complete"]:
        L.append("")
        L.append("Family incomplete: at least one member has no result (pending runs).")
    return L


# ============================================================================== main


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("items", help="per-item tables of every complete run (needs pandas + openpyxl)")
    pi.add_argument("--ext-root", default=os.environ.get("TRC_EXT_ROOT"), required=os.environ.get("TRC_EXT_ROOT") is None)
    pi.add_argument("--out-dir", required=True)
    pi.add_argument("--jobs", type=int, default=4)
    pi.add_argument("--only", help="comma list of served names")
    pi.add_argument("--force", action="store_true", help="rebuild cached tables")
    pi.add_argument("--vendor", help="dir holding a copy of latex2sympy2_extended + antlr4 (MathVision fallback)")
    pi.add_argument("--subset-file", help=f"q25 row list (default <ext-root>/{Q25_SUBSET_FILE})")
    pa = sub.add_parser("analyze", help="tables, families, comparisons, format shift (numpy)")
    pa.add_argument("--ext-root", default=os.environ.get("TRC_EXT_ROOT"), required=os.environ.get("TRC_EXT_ROOT") is None)
    pa.add_argument("--out-dir", required=True)
    pa.add_argument("--n-boot", type=int, default=10000)
    pa.add_argument("--n-perm", type=int, default=10000)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--alpha", type=float, default=0.05)
    pa.add_argument("--flag-pp", type=float, default=2.0)
    pa.add_argument("--trace-tool-src", help="trace-tool's src dir (default $TRACE_TOOL_SRC or $TRACE_TOOL_ROOT/src)")
    pa.add_argument("--subset-file", help=f"q25 row list (default <ext-root>/{Q25_SUBSET_FILE})")
    args = p.parse_args(argv)
    try:
        return cmd_items(args) if args.cmd == "items" else cmd_analyze(args)
    except ExtError as exc:
        print(f"compare_ext: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
