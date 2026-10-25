"""Re-score the zero-shot certificate responses for the Section 5 report (DESIGN 7.1, 2.2, 2.3).

    PYTHONPATH="$TRACE_ROOT/src" python experiments/certificates/analysis/section5_zs_rescore.py \\
        --zs-root "$TRACE_CERT_RUNS/eval/zs" --data "$TRACE_CERT_RUNS/data/v2" \\
        --out "$TRACE_CERT_RUNS/results/final/section5/rescore" --workers 4

Read-only on its inputs. For every ``cert_v1`` run of every model under ``--zs-root``
(``<model>/<split>/cert_v1/seed<k>/{responses.jsonl, cert/cert_scores.jsonl}``) it re-scores each
response with the current certificate scorer and writes one compact row per response:

1. **Reproduction check.** The response is scored again in its declared contract frame (the
   ``coord_frame`` of its stored row) and A, A_legacy, G, G_id, G_shown, M, M_shown, F, joint,
   error_class and valid_schema are compared with the stored ``cert_scores.jsonl`` row
   (floats within 1e-9). Any mismatch is listed; the stored rows stay the source of record.
2. **Other contract frame.** The same response is scored with the other public frame
   (``px_source`` <-> ``rel1000``): A, G, M, F, joint, error_class. For Qwen3.5 (declared
   rel1000) this is the frame-confound diagnostic: what its certificates would score if its
   coordinates were source pixels. G and M are also re-scored in the response's shown frame
   (``shown_image_size`` the server reported) as ``G_px_shown`` / ``M_px_shown`` (M with the
   scorer's own ``_semantics`` path and a ``px_shown`` frame, exactly as its Qwen2.5-VL
   ``_shown_diagnostics`` does; for Qwen2.5-VL rows both are checked against the stored
   ``G_shown`` / ``M_shown`` and any difference is counted under ``shown_replication``).
   ``joint_px_shown`` = A = F = 1 and ``M_px_shown`` = 1 (A and F do not depend on the frame;
   checked on every row: ``af_frame_dependent`` counts rows whose other-frame A or F differ).
3. **Coordinate ranges.** Every number pair of the scored object's annotation (else the legacy
   tolerant extraction, as the scorer does for G): count, max x / max y, whether all are <= 1000,
   inside the source image, inside the shown image.
4. **Identity sensitivity (E1-dev only, DESIGN 2.3).** M, G_id, joint and error_class with the
   identity IoU threshold at 0.3 and 0.7 (containment 0.8 kept), declared frame.

Outputs: ``rows.jsonl.gz`` (sorted by model, split, seed, row_index; gzip header without file name
or mtime, so its sha256 is reproducible; ``rows_jsonl_sha256`` is the sha256 of the decompressed
stream) and ``rescore_summary.json`` (per-run counts, mismatches, the scorer version and certificate
code sha256, input sha256s). Exit code 1 when any stored row is not reproduced or the shown-frame
replication differs from a stored Qwen2.5-VL ``G_shown`` / ``M_shown``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

VERSION = "section5_zs_rescore-1.1.0"
SPLITS = ("e1_dev", "e1_test", "e2_test")
SEEDS = (42, 43, 44)
CHECK_FIELDS = ("A", "A_legacy", "G", "G_id", "G_shown", "M", "M_shown", "F", "joint", "error_class", "valid_schema")
IOU_SENSITIVITY = (0.3, 0.7)
TOL = 1e-9


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ worker state (one per process)
_STORES: Dict[str, Any] = {}


def _store(data: str, split: str):
    from trace_tasks.experimental.certificates.evaluate import load_store

    key = f"{data}|{split}"
    if key not in _STORES:
        _STORES[key] = load_store(Path(data) / "private" / f"{split}.extensions_v1.jsonl.zst")
    return _STORES[key]


def _flatten(value: Any, depth: int = 0) -> List[float]:
    if depth > 8 or isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        try:
            f = float(value)
        except (OverflowError, ValueError):
            return []
        return [f] if math.isfinite(f) else []
    if isinstance(value, (list, tuple)):
        out: List[float] = []
        for v in value:
            out.extend(_flatten(v, depth + 1))
        return out
    if isinstance(value, dict):
        out = []
        for v in value.values():
            out.extend(_flatten(v, depth + 1))
        return out
    return []


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, str) or isinstance(b, str):
        return a == b
    return abs(float(a) - float(b)) <= TOL


def _score_fields(res: Any) -> Dict[str, Any]:
    diag = res.diagnostics
    return {
        "A": res.answer,
        "G": res.grounding,
        "G_id": diag.get("G_id"),
        "M": res.semantics,
        "F": res.format,
        "joint": res.joint,
        "error_class": res.error_class,
    }


def shown_semantics(response: str, ext: Any, adapter: Any, frame: Any) -> Optional[float]:
    """M with the response's coordinates read in ``frame`` (a ``px_shown`` frame).

    Same calls as ``scoring._shown_diagnostics`` (the scorer's Qwen2.5-VL shown-frame
    diagnostic), with the frame built from the server-reported shown size so that it also works
    for Qwen3.5. Returns None when the diagnostic raises (as the scorer records ``shown_error``).
    """
    from trace_tasks.experimental.certificates import scoring
    from trace_tasks.experimental.certificates.adapters.base import PublicContext
    from trace_tasks.experimental.certificates.identity import DEFAULT_IDENTITY
    from trace_tasks.experimental.certificates.parsing import locate_scored_object

    try:
        ctx = PublicContext(
            task=str(ext.base["task"]), query_id=str(ext.base["query_id"]), image_size=ext.image_size,
            coord_frame="px_shown", frame=frame, identity=DEFAULT_IDENTITY,
        )
        applicable = dict(adapter.family_applicable(ext.oracle))
        _, m_shown, _, _, _ = scoring._semantics(adapter, locate_scored_object(response), ext.oracle, ctx, applicable)
        return scoring._unit("M_shown", m_shown)
    except Exception:  # noqa: BLE001 - a diagnostic; mirrors the scorer's shown_error handling
        return None


def score_run(task: Tuple[str, str, str, str, int]) -> Dict[str, Any]:
    """Re-score one ``<model>/<split>/cert_v1/seed<k>`` run; returns rows and counts."""
    from trace_tasks.experimental.certificates import registry
    from trace_tasks.experimental.certificates.evaluate import POSITION_ONLY_REASONS, read_responses
    from trace_tasks.experimental.certificates.geometry import Frame
    from trace_tasks.experimental.certificates.identity import IdentityParams
    from trace_tasks.experimental.certificates.parsing import locate_scored_object
    from trace_tasks.experimental.certificates.scoring import (
        legacy_prediction,
        score_certificate_response,
        score_grounding,
    )

    zs_root, data, model, split, seed = task
    run = Path(zs_root) / model / split / "cert_v1" / f"seed{seed}"
    started = time.time()
    stored: Dict[str, Dict[str, Any]] = {}
    with open(run / "cert" / "cert_scores.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                stored[row["instance_id"]] = row
    records, failures, n_lines = read_responses(run / "responses.jsonl")
    store = _store(data, split)
    out_rows: List[Dict[str, Any]] = []
    mismatches: List[Dict[str, Any]] = []
    shown_rep = {"n_compared": 0, "n_G_mismatch": 0, "n_M_mismatch": 0, "examples": []}
    af_dependent = 0
    m_shown_errors = 0
    missing = 0
    for rec in records:
        st = stored.get(rec["instance_id"])
        if st is None:
            missing += 1
            continue
        ext = store.for_instance(rec["instance_id"])
        adapter = registry.adapter_for_extension(ext)
        response = rec.get("final_response")
        response = response if isinstance(response, str) else ("" if response is None else str(response))
        base = {"instance_id": rec["instance_id"], "cert_extension_id": ext.certificate_extension_id}
        declared = str(st["coord_frame"])
        family = str(st["family"])
        alt = "px_source" if declared == "rel1000" else "rel1000"

        # 1. reproduction in the declared frame (same call as evaluate.score_record)
        r0 = score_certificate_response(
            response, base, ext, coord_frame=declared, adapter=adapter,
            shown_frame_diagnostics=family == "qwen2.5-vl" and declared == "px_source",
        )
        d0 = r0.diagnostics
        reasons = list(d0.get("format_reasons") or [])
        again = {
            "A": r0.answer, "A_legacy": d0.get("A_legacy"), "G": r0.grounding, "G_id": d0.get("G_id"),
            "G_shown": d0.get("G_shown"), "M": r0.semantics, "M_shown": d0.get("M_shown"), "F": r0.format,
            "joint": r0.joint, "error_class": r0.error_class,
            "valid_schema": d0.get("envelope_error") is None and set(reasons) <= POSITION_ONLY_REASONS,
        }
        bad = [k for k in CHECK_FIELDS if not _same(again[k], st.get(k))]
        if bad:
            mismatches.append({"instance_id": rec["instance_id"], "fields": {k: [st.get(k), again[k]] for k in bad}})

        # 2. the other contract frame
        r1 = score_certificate_response(response, base, ext, coord_frame=alt, adapter=adapter)

        # 3. coordinate ranges of the annotation the scorer reads
        located = locate_scored_object(response)
        if located.scored is not None:
            annotation, source = located.scored.get("annotation"), "scored_object"
        else:
            _, annotation, _ = legacy_prediction(response)
            source = "legacy_extraction"
        nums = _flatten(annotation)
        pairs = [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
        w, h = (int(v) for v in ext.image_size)
        shown = rec.get("shown_image_size") or [w, h]
        sw, sh = int(shown[0]), int(shown[1])
        shown_frame = Frame("px_shown", w / sw, h / sh)
        try:
            g_px_shown = score_grounding(annotation, ext.labels, adapter.annotation_type, shown_frame, (w, h))[0]
        except Exception:  # noqa: BLE001 - a malformed annotation scores 0 (as zs_report frame-diag)
            g_px_shown = 0.0
        m_px_shown = shown_semantics(response, ext, adapter, shown_frame)
        if m_px_shown is None:
            m_shown_errors += 1
        joint_px_shown = (None if m_px_shown is None else
                          1.0 if (r0.answer == 1.0 and r0.format == 1.0 and m_px_shown == 1.0) else 0.0)
        if r1.answer != r0.answer or r1.format != r0.format:
            af_dependent += 1
        if st.get("G_shown") is not None or st.get("M_shown") is not None:
            shown_rep["n_compared"] += 1
            bad_g = not _same(st.get("G_shown"), float(g_px_shown))
            bad_m = not _same(st.get("M_shown"), m_px_shown)
            shown_rep["n_G_mismatch"] += int(bad_g)
            shown_rep["n_M_mismatch"] += int(bad_m)
            if (bad_g or bad_m) and len(shown_rep["examples"]) < 20:
                shown_rep["examples"].append({"instance_id": rec["instance_id"],
                                              "G_shown": [st.get("G_shown"), g_px_shown],
                                              "M_shown": [st.get("M_shown"), m_px_shown]})
        xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
        row: Dict[str, Any] = {
            "model": model, "family": family, "split": split, "seed": int(seed),
            "row_index": st.get("row_index"), "instance_id": rec["instance_id"],
            "adapter_id": st["adapter_id"], "cell": st.get("cell"), "scene_group": st.get("scene_group"),
            "declared_frame": declared, "alt_frame": alt,
            "reproduced": not bad,
            "stored": {k: st.get(k) for k in ("A", "G", "M", "F", "joint", "error_class")},
            "alt": _score_fields(r1),
            "G_px_shown": float(g_px_shown),
            "M_px_shown": m_px_shown,
            "joint_px_shown": joint_px_shown,
            "annotation_source": source,
            "n_coords": len(pairs),
            "max_x": max(xs) if xs else None, "max_y": max(ys) if ys else None,
            "source_wh": [w, h], "shown_wh": [sw, sh],
            "le_1000": bool(pairs) and max(xs + ys) <= 1000.0 and min(xs + ys) >= 0.0,
            "in_source": bool(pairs) and max(xs) <= w and max(ys) <= h and min(xs + ys) >= 0.0,
            "in_shown": bool(pairs) and max(xs) <= sw and max(ys) <= sh and min(xs + ys) >= 0.0,
        }
        # 4. identity sensitivity on E1-dev
        if split == "e1_dev":
            for iou in IOU_SENSITIVITY:
                ri = score_certificate_response(
                    response, base, ext, coord_frame=declared, adapter=adapter,
                    identity=IdentityParams(iou_threshold=iou, containment_threshold=0.8),
                )
                row[f"iou{iou:.1f}"] = _score_fields(ri)
        out_rows.append(row)
    return {
        "model": model, "split": split, "seed": int(seed),
        "responses_sha256": sha256_file(run / "responses.jsonl"),
        "cert_scores_sha256": sha256_file(run / "cert" / "cert_scores.jsonl"),
        "n_lines": n_lines, "n_records": len(records), "n_infra_failures": len(failures),
        "n_stored": len(stored), "n_rescored": len(out_rows), "missing_stored": missing,
        "n_mismatch": len(mismatches), "mismatches": mismatches[:50],
        "shown_replication": shown_rep, "af_frame_dependent": af_dependent, "m_px_shown_errors": m_shown_errors,
        "elapsed_s": round(time.time() - started, 2),
        "rows": out_rows,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zs-root", required=True)
    ap.add_argument("--data", required=True, help="frozen data dir (data/v2): private/<split>.extensions_v1.jsonl.zst")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="", help="comma list; default every model dir with a plan.json")
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)

    from trace_tasks.experimental.certificates.export import cert_code_sha256
    from trace_tasks.experimental.certificates.scoring import SCORER_VERSION

    zs_root = Path(args.zs_root).resolve()
    models = [m for m in args.models.split(",") if m] or sorted(
        p.name for p in zs_root.iterdir() if (p / "plan.json").is_file()
    )
    splits = [s for s in args.splits.split(",") if s]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    tasks = []
    absent = []
    for m in models:
        for sp in splits:
            for sd in seeds:
                run = zs_root / m / sp / "cert_v1" / f"seed{sd}"
                if (run / "responses.jsonl").is_file() and (run / "cert" / "cert_scores.jsonl").is_file():
                    tasks.append((str(zs_root), str(Path(args.data).resolve()), m, sp, sd))
                else:
                    absent.append(str(run))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for res in ex.map(score_run, tasks):
            results.append(res)
            print(f"{res['model']} {res['split']} seed{res['seed']}: {res['n_rescored']} rows, "
                  f"{res['n_mismatch']} mismatches, {res['elapsed_s']} s", flush=True)
    results.sort(key=lambda r: (r["model"], SPLITS.index(r["split"]) if r["split"] in SPLITS else 9, r["seed"]))
    rows_path = out / "rows.jsonl.gz"
    n_rows = 0
    plain = hashlib.sha256()
    # filename="" and mtime=0: the gzip header carries no name or time, so the .gz bytes (and their
    # sha256) are reproducible across reruns; plain is the sha256 of the decompressed stream.
    with open(rows_path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as gz:
        for res in results:
            for row in sorted(res["rows"], key=lambda r: (r["row_index"] if r["row_index"] is not None else -1, r["instance_id"])):
                line = (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
                gz.write(line)
                plain.update(line)
                n_rows += 1
    stores = {}
    for sp in splits:
        p = Path(args.data) / "private" / f"{sp}.extensions_v1.jsonl.zst"
        stores[sp] = {"path": str(p), "sha256": sha256_file(p)}
    summary = {
        "version": VERSION,
        "scorer_version": SCORER_VERSION,
        "cert_code_sha256": cert_code_sha256(),
        "zs_root": str(zs_root), "data": str(Path(args.data).resolve()), "stores": stores,
        "models": models, "splits": splits, "seeds": seeds,
        "iou_sensitivity": list(IOU_SENSITIVITY), "containment_threshold": 0.8,
        "absent_runs": absent,
        "runs": [{k: v for k, v in r.items() if k != "rows"} for r in results],
        "n_rows": n_rows,
        "n_mismatch_total": sum(r["n_mismatch"] for r in results),
        "shown_replication": {
            "n_compared": sum(r["shown_replication"]["n_compared"] for r in results),
            "n_G_mismatch": sum(r["shown_replication"]["n_G_mismatch"] for r in results),
            "n_M_mismatch": sum(r["shown_replication"]["n_M_mismatch"] for r in results),
        },
        "af_frame_dependent_total": sum(r["af_frame_dependent"] for r in results),
        "m_px_shown_errors_total": sum(r["m_px_shown_errors"] for r in results),
        "rows_jsonl_gz_sha256": sha256_file(rows_path),
        "rows_jsonl_sha256": plain.hexdigest(),
        "gzip_header": "no file name, mtime 0 (reproducible)",
        "elapsed_s": round(time.time() - started, 1),
        "command": " ".join([sys.executable] + list(sys.argv)),
    }
    (out / "rescore_summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    sr = summary["shown_replication"]
    print(f"rows {n_rows}, mismatches {summary['n_mismatch_total']}, absent runs {len(absent)}, "
          f"shown-frame replication {sr['n_compared']} rows compared, {sr['n_G_mismatch']} G / {sr['n_M_mismatch']} M "
          f"differ; A/F frame-dependent rows {summary['af_frame_dependent_total']}; "
          f"M_px_shown errors {summary['m_px_shown_errors_total']}; {summary['elapsed_s']} s")
    bad = summary["n_mismatch_total"] or sr["n_G_mismatch"] or sr["n_M_mismatch"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
