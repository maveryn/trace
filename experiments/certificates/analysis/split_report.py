"""Split report and integrity checks of an exported data root (DESIGN 6).

    python experiments/certificates/analysis/split_report.py --data "$TRACE_CERT_RUNS/data/v2" \\
        --out "$TRACE_CERT_RUNS/data/v2/split_report.json" [--times build_times.tsv]

Reads every ``receipts/<split>.json``, ``private/build/<split>/<adapter>/accepted.jsonl``, private
store and ``receipts/freeze.json``, then asserts (exit 1 on a violation):

- held-out E2 cells never appear in SFT or RL (``split_cell`` and the extension ``base.cell``),
  and every SFT/RL/E1 row is a declared train cell, every E2 row a declared held-out cell;
- a ``scene_group`` (and an instance id, an extension id) never crosses splits;
- no test row's semantic key appears in SFT or RL, and no test split holds one semantic key
  twice (the exporter drops both; the counts are kept);
- every store matches its receipt, every public prompt (both frames and the legacy ``ans`` /
  ``ann`` prompts) matches the sha256 frozen in its extension, every extension's payload hash
  verifies, the oracle round trip passed at export, and ``freeze.json`` names the stores on disk;
- no oracle-bearing ``build/`` dir sits outside ``private/``.

Prints a Markdown summary of counts per adapter/cell, quarantine reasons, dedupe drops and
(with ``--times``, a ``name<TAB>workers<TAB>seconds<TAB>rc`` file) wall times.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from trace_tasks.experimental.certificates import export, registry
from trace_tasks.experimental.certificates.types import CertificateExtension

TRAIN_SPLITS = export.TRAIN_SPLITS
TEST_SPLITS = export.TEST_SPLITS


def _accepted(data: Path, split: str, name: str) -> List[Dict[str, Any]]:
    path = export.build_root(data) / split / name / "accepted.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def build_report(data: Path, times: Optional[Path] = None) -> Dict[str, Any]:
    problems: List[str] = []
    splits = [s for s in export.SPLIT_SEEDS if (data / "receipts" / f"{s}.json").exists()]
    report: Dict[str, Any] = {"splits": {}, "problems": problems}
    groups: Dict[str, set] = defaultdict(set)
    instances: Dict[str, set] = defaultdict(set)
    ext_ids: Counter = Counter()
    keys: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))  # adapter -> split -> semantic keys
    for split in splits:
        receipt = json.loads((data / "receipts" / f"{split}.json").read_text(encoding="utf-8"))
        store_rel = f"private/{split}.{export.STORE_NAME}"
        if export.file_sha256(data / store_rel) != receipt["outputs"][store_rel]["sha256"]:
            problems.append(f"{split}: store does not match its receipt")
        store = export.read_store(data / store_rel)
        cells_from_store: Dict[str, Counter] = defaultdict(Counter)
        prompt_shas = {row["certificate_extension_id"]: row["public"]["prompt_sha256"] for row in store}
        for row in store:
            base = row["base"]
            try:
                CertificateExtension.from_dict(row).verify()
            except Exception as exc:  # noqa: BLE001 - reported as a problem
                problems.append(f"{split}: extension {row['certificate_extension_id']} fails verification: {exc}")
            groups[base["scene_group"]].add(split)
            instances[base["instance_id"]].add(split)
            ext_ids[row["certificate_extension_id"]] += 1
            cells_from_store[row["adapter"]["id"]][base["cell"]] += 1
            if base["split"] != split:
                problems.append(f"{split}: extension {row['certificate_extension_id']} says split {base['split']}")
        entry: Dict[str, Any] = {
            "sampling_seed": receipt["sampling_seed"],
            "count_per_adapter": receipt["count_per_adapter"],
            "totals": receipt["totals"],
            "dedupe": {"enabled": receipt["dedupe"]["enabled"]},
            "adapters": {},
        }
        for name in export.ADAPTER_ORDER:
            adapter = registry.load_adapter(name)
            counts = receipt["counts"].get(adapter.adapter_id)
            if counts is None:
                continue
            rows = _accepted(data, split, name)
            cells = Counter(row["cell"] for row in rows)
            if cells != cells_from_store[adapter.adapter_id]:
                problems.append(f"{split}/{name}: accepted.jsonl cells differ from the store")
            for row in rows:
                keys[name][split].add(row["semantic_key"])
                shas = prompt_shas.get(row["cert_extension_id"], {})
                for key, condition in export.PROMPT_KEYS.items():
                    if shas.get(key) != export.prompt_sha256(row["prompts"][condition]):
                        problems.append(f"{split}/{name}: {row['instance_id']} {condition} prompt sha differs from the store")
                holdout = adapter.holdout_type(row["cell"])
                if holdout != row["holdout"]:
                    problems.append(f"{split}/{name}: {row['instance_id']} holdout {row['holdout']!r} != {holdout!r}")
                if split in TRAIN_SPLITS + ("e1_dev", "e1_test") and holdout:
                    problems.append(f"{split}/{name}: held-out cell {row['cell']} in a train/E1 split")
                if split == "e2_test" and not holdout:
                    problems.append(f"{split}/{name}: train cell {row['cell']} in E2")
            entry["adapters"][adapter.adapter_id] = {
                "accepted": counts["accepted"],
                "cells": dict(sorted(cells.items())),
                "strata": {k: v["accepted"] for k, v in counts["strata"].items()},
                "candidates": counts["candidates"],
                "excluded_other_cells": counts["excluded"],
                "rejected_by_trace": counts["rejected"],
                "rejected_reasons": counts["rejected_reasons"],
                "quarantined": counts["quarantined"],
                "quarantine_reasons": counts["quarantine_reasons"],
                "dedupe_dropped": counts["dedupe_dropped"],
                "dedupe_within_dropped": counts.get("dedupe_within_dropped", 0),
                "duplicate_semantic_keys_within_split": counts["duplicate_semantic_keys_within_split"],
                "roundtrip_failed": len(counts["roundtrip_failed"]),
            }
            if counts["roundtrip_failed"]:
                problems.append(f"{split}/{name}: {len(counts['roundtrip_failed'])} oracle round-trip failures")
            if split in TEST_SPLITS and receipt["dedupe"]["enabled"]:
                seen = [row["semantic_key"] for row in rows]
                if len(seen) != len(set(seen)):
                    problems.append(f"{split}/{name}: {len(seen) - len(set(seen))} semantic keys twice in one test split")
        report["splits"][split] = entry
    crossing = {g: sorted(s) for g, s in groups.items() if len(s) > 1}
    report["scene_groups"] = {"total": len(groups), "crossing_splits": len(crossing)}
    if crossing:
        problems.append(f"{len(crossing)} scene groups cross splits, e.g. {next(iter(crossing.items()))}")
    shared = [i for i, s in instances.items() if len(s) > 1]
    if shared:
        problems.append(f"{len(shared)} instance ids appear in more than one split")
    if any(n > 1 for n in ext_ids.values()):
        problems.append("duplicate certificate_extension_id across splits")
    overlap: Dict[str, Dict[str, int]] = {}
    for name, by_split in keys.items():
        train = set().union(*(by_split.get(s, set()) for s in TRAIN_SPLITS))
        overlap[name] = {s: len(by_split.get(s, set()) & train) for s in TEST_SPLITS if s in by_split}
        for s, n in overlap[name].items():
            if n and report["splits"][s]["dedupe"]["enabled"]:
                problems.append(f"{s}/{name}: {n} semantic keys also in SFT/RL despite dedupe")
    report["test_keys_in_train"] = overlap
    if (data / "build").exists():
        problems.append("an oracle-bearing build/ dir exists outside private/")
    freeze_path = data / "receipts" / export.FREEZE_RECEIPT
    if freeze_path.exists():
        frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
        report["freeze"] = {key: frozen[key]["sha256"] for key in ("merged_store", "rl_reward_store")}
        for key in ("merged_store", "rl_reward_store"):
            if export.file_sha256(data / frozen[key]["path"]) != frozen[key]["sha256"]:
                problems.append(f"freeze.json {key} does not match {frozen[key]['path']}")
        for split, entry in frozen["splits"].items():
            if export.file_sha256(data / entry["receipt"]["path"]) != entry["receipt"]["sha256"]:
                problems.append(f"freeze.json names an older receipts/{split}.json")
    else:
        problems.append("receipts/freeze.json is missing (run `freeze` after the last export)")
    if times is not None and times.exists():
        report["wall_seconds"] = {
            parts[0]: {"workers": parts[1], "seconds": int(parts[2]), "rc": int(parts[3])}
            for parts in (line.split("\t") for line in times.read_text(encoding="utf-8").splitlines() if line.strip())
        }
    return report


def markdown(report: Mapping[str, Any]) -> str:
    out = ["| split | adapter | accepted | cells | candidates | excluded (other cells) | quarantined | dedupe dropped (vs SFT/RL, within split) |",
           "|---|---|---:|---|---:|---:|---:|---:|"]
    for split, entry in report["splits"].items():
        for adapter_id, a in entry["adapters"].items():
            cells = ", ".join(f"{c} {n}" for c, n in a["cells"].items())
            out.append(f"| {split} | {adapter_id.split('.')[0]} | {a['accepted']} | {cells} | {a['candidates']} | "
                       f"{a['excluded_other_cells']} | {a['quarantined']} | {a['dedupe_dropped']}, {a['dedupe_within_dropped']} |")
    out.append("")
    out.append(f"Scene groups: {report['scene_groups']['total']}, crossing splits: {report['scene_groups']['crossing_splits']}. "
               f"Test semantic keys found in SFT/RL: {json.dumps(report['test_keys_in_train'])}.")
    if "wall_seconds" in report:
        out.append("")
        out.append("| step | workers | wall s |")
        out.append("|---|---:|---:|")
        out += [f"| {k} | {v['workers']} | {v['seconds']} |" for k, v in report["wall_seconds"].items()]
    out.append("")
    out.append("Problems: " + ("none" if not report["problems"] else "; ".join(report["problems"])))
    return "\n".join(out) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--times")
    args = parser.parse_args(argv)
    report = build_report(Path(args.data), Path(args.times) if args.times else None)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(markdown(report))
    return 1 if report["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
