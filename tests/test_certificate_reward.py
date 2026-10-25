"""EasyR1 batch reward of the certificate extension (DESIGN 5 "Batch API", 9).

Covers the arm formulas on synthetic ScoreResults, the batch contract (order under shuffle,
fixed float keys, bounds, no wall-clock dependence), store and join integrity (every
mismatch raises), the details files, the EasyR1 file loader replayed on ``reward_entry.py``,
the frozen adapter fixtures through the exporter's store format, and the release
``trace_rlvr.compute_score`` goldens.

The release goldens and the EasyR1 source drift guard need ``TRACE_RLVR_ROOT`` (the
trace-rlvr checkout that holds ``rlvr/easyr1_backend``); they skip when it is unset.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pytest

from trace_tasks.core.reward_contracts import resolve_reward_contract
from trace_tasks.experimental.certificates import registry, reward, scoring
from trace_tasks.experimental.certificates.adapters.base import Adapter, PublicContext
from trace_tasks.experimental.certificates.checks import (
    bind_elements,
    bound_exactly_once,
    counted_truth,
    gated_check,
    indices_bijective,
    local_check,
)
from trace_tasks.experimental.certificates.identity import CandidateSet
from trace_tasks.experimental.certificates.parsing import INDEX, INT
from trace_tasks.experimental.certificates.types import (
    CertificateExtension,
    CertificateVersionError,
    Corruption,
    ExtensionMismatchError,
    ScoreResult,
    certificate_extension_id,
)

REPO = Path(__file__).resolve().parents[1]
ENTRY = REPO / "experiments" / "certificates" / "reward_entry.py"
FIXTURES = REPO / "tests" / "fixtures" / "certificates"
TOY_TASK = "task_toy__blocks__reward_sum"
IMAGE_SIZE = (640, 480)
BOXES = (
    [100.0, 280.0, 158.0, 332.0],
    [170.0, 280.0, 228.0, 332.0],
    [240.0, 280.0, 298.0, 332.0],
    [400.0, 280.0, 458.0, 332.0],
)
UNIT_KEYS = reward.METRIC_KEYS[:-1]


def gate_skip(reason: str) -> None:
    """Skip, except under the M1 gate (``CERT_GATE=1``), where a required test must run (review CONF-12)."""
    if os.environ.get("CERT_GATE") == "1":
        pytest.fail(f"CERT_GATE=1: required test cannot run: {reason}")
    pytest.skip(reason)
ARMS = ("A", "G", "P", "G_gated", "P_gated")


class ToyRewardAdapter(Adapter):
    """Sum of the values printed on the first three blocks (bbox_set annotation)."""

    _allow_foreign_module = True
    adapter_id = "toy_reward.blocks"
    kind = "toy_sum"
    version = "1.0.0"
    schema_id = "cert.toy_sum.v1"
    verifier_version = "1.0.0"
    supported = {TOY_TASK: ("single",)}
    annotation_type = "bbox_set"
    list_key = "terms"
    list_cap = 8
    cert_fields = {"total": INT}
    element_fields = {"annotation_index": INDEX, "value": INT}
    m_families = ("toy.items.truth", "toy.value.truth", "toy.total.truth")
    family_stages = {"toy.items.truth": "grounding", "toy.value.truth": "operand", "toy.total.truth": "computation"}
    local_check_ids = ("toy.ref.local",)

    def build_oracle(self, record):  # pragma: no cover - extensions are built directly
        raise NotImplementedError

    def split_cell(self, record):  # pragma: no cover
        return "toy"

    def semantic_key(self, record):  # pragma: no cover
        return "toy"

    def sft_certificate(self, oracle, rng_seed):
        selected = [block for block in oracle["blocks"] if block["selected"]]
        terms = [{"annotation_index": i, "value": block["value"]} for i, block in enumerate(selected)]
        certificate = {"kind": self.kind, "terms": terms, "total": oracle["total"]}
        return [block["bbox"] for block in selected], certificate, oracle["total"]

    def candidates(self, oracle):
        blocks = oracle["blocks"]
        selected = tuple(i for i, block in enumerate(blocks) if block["selected"])
        return {"": CandidateSet(tuple(tuple(block["bbox"]) for block in blocks), selected)}

    def checks(self, cert, annotation, oracle, ctx: PublicContext):
        candidates = self.candidates(oracle)
        bindings = bind_elements(cert.elements, self.resolve_role(annotation, "", oracle, ctx, candidates))
        once = bound_exactly_once(bindings, candidates[""].selected)
        n, k = cert.n_claimed, len(candidates[""].selected)
        items = counted_truth("toy.items.truth", len(once), n, k)
        blocks = oracle["blocks"]
        hits = sum(1 for t, pos in once.items() if cert.elements[pos].get("value") == blocks[t]["value"])
        value = counted_truth("toy.value.truth", hits, n, k)
        values = [element.get("value") for element in cert.elements if element.valid]
        total = cert.fields.get("total")
        total_ok = total is not None and len(values) == n and total == sum(values)
        return [
            items,
            value,
            gated_check("toy.total.truth", [items, value], total_ok),
            local_check("toy.ref.local", indices_bijective(cert.elements, annotation.size())),
        ]

    def corruptions(self, oracle, sft_obj, rng_seed):  # pragma: no cover
        return [Corruption("noop", dict(sft_obj), {"M": "pass"})]

    def certificate_line(self):  # pragma: no cover
        return 'Set "certificate" to {"kind": "toy_sum", "terms": [...], "total": t}.'

    def example_object(self):  # pragma: no cover
        certificate = {"kind": "toy_sum", "terms": [], "total": 0}
        return {"annotation": [[1, 2, 3, 4]], "certificate": certificate, "answer": 0}


ADAPTER = ToyRewardAdapter()


@pytest.fixture(autouse=True)
def _toy_adapter():
    registry.register_adapter(ADAPTER)
    yield
    registry.unregister_adapter(ADAPTER.adapter_id)


def make_extension(
    *,
    base: Mapping[str, Any],
    identity: Mapping[str, str],
    labels: Mapping[str, Any],
    image_size: Sequence[int],
    oracle: Mapping[str, Any],
    sft_target: Mapping[str, Any] | None = None,
) -> CertificateExtension:
    ext_id = certificate_extension_id(
        instance_id=base["instance_id"],
        trace_record_hash=base["trace_record_hash"],
        adapter_id=identity["id"],
        adapter_version=identity["version"],
        schema_id=identity["schema_id"],
    )
    ext = CertificateExtension(
        certificate_extension_id=ext_id,
        base=dict(base),
        adapter=dict(identity),
        labels=dict(labels),
        public={"image_size": list(image_size), "prompt_sha256": dict(TOY_PROMPT_SHAS)},
        oracle=dict(oracle),
        sft_target=dict(sft_target or {}),
    )
    ext.provenance["payload_sha256"] = ext.payload_digest()  # real stores always carry it (LEAK-04)
    return ext


TOY_PROMPTS = {"px_source": "<image>public prompt", "rel1000": "<image>public prompt (rel1000)"}
TOY_PROMPT_SHAS = {frame: hashlib.sha256(text.encode("utf-8")).hexdigest() for frame, text in TOY_PROMPTS.items()}


def toy_extension(i: int, split: str = "rl", version: str = ADAPTER.version) -> CertificateExtension:
    values = [(i + j) % 9 + 1 for j in range(4)]
    blocks = [{"bbox": list(b), "value": v, "selected": j < 3} for j, (b, v) in enumerate(zip(BOXES, values))]
    total = sum(values[:3])
    contract = resolve_reward_contract(answer_type="integer", annotation_type="bbox_set").to_dict()
    return make_extension(
        base={
            "instance_id": f"blake3:{i:064x}",
            "trace_record_hash": f"blake3:{i + 10**6:064x}",
            "task": TOY_TASK,
            "query_id": "single",
            "instance_seed": i,
            "split": split,
        },
        identity=dict(ADAPTER.identity(), version=version),
        labels={
            "answer_gt": {"type": "integer", "value": total},
            "annotation_gt": {"type": "bbox_set", "value": [b["bbox"] for b in blocks[:3]]},
            "reward_contract": contract,
        },
        image_size=IMAGE_SIZE,
        oracle={"blocks": blocks, "total": total},
    )


def write_store(path: Path, records: Sequence[Mapping[str, Any]], compress: bool = True) -> str:
    data = "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records).encode("utf-8")
    if compress:
        import zstandard as zstd

        data = zstd.ZstdCompressor(level=3).compress(data)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def responses(ext: CertificateExtension) -> List[Any]:
    """Deterministic variants: oracle, position-only defect, operand, arithmetic, cap, junk."""
    annotation, certificate, answer = ADAPTER.sft_certificate(ext.oracle, 0)
    good = {"annotation": annotation, "certificate": certificate, "answer": answer}
    wrong_operand = copy.deepcopy(good)
    wrong_operand["certificate"]["terms"][0]["value"] += 1
    wrong_operand["certificate"]["total"] += 1
    wrong_operand["answer"] += 1
    wrong_total = copy.deepcopy(good)
    wrong_total["certificate"]["total"] += 2
    over_cap = copy.deepcopy(good)
    over_cap["certificate"]["terms"] = over_cap["certificate"]["terms"] * 3  # 9 > list_cap 8
    return [
        json.dumps(good),
        "Reasoning first.\n" + json.dumps(good) + "\nDone.",
        json.dumps(wrong_operand),
        json.dumps(wrong_total),
        json.dumps(over_cap),
        json.dumps({"answer": answer}),
        '{"annotation": [], "certificate": {"kind": "toy_sum", "terms": [], "total": 0}, "answer": true}',
        '{"answer": NaN}',
        "no json at all",
        "",
        None,
        "{" * 5000,
    ]


def reward_input(ext: CertificateExtension, response: Any, **extra: Any) -> Dict[str, Any]:
    row = {
        "response": response,
        "response_length": 17,
        "ground_truth": json.dumps({"cert_extension_id": ext.certificate_extension_id}),
        "instance_id": ext.instance_id,
        "cert_extension_id": ext.certificate_extension_id,
        "prompt_cert": "<image>public prompt",
        "uid": "u-" + ext.instance_id[-4:],
    }
    row.update(extra)
    return row


@pytest.fixture()
def store(tmp_path: Path) -> Dict[str, Any]:
    extensions = [toy_extension(i, split="rl" if i < 3 else "e1_dev") for i in range(5)]
    path = tmp_path / "extensions_v1.jsonl.zst"
    sha = write_store(path, [ext.to_dict() for ext in extensions])
    # One reward call never mixes phases (train rollouts vs the val pass), so the train rows
    # (split rl) and the val rows (split e1_dev) are separate batches.
    rows = [reward_input(ext, response) for ext in extensions[:3] for response in responses(ext)]
    val_rows = [reward_input(ext, response) for ext in extensions[3:] for response in responses(ext)]
    return {"path": str(path), "sha": sha, "extensions": extensions, "rows": rows, "val_rows": val_rows}


def score(store: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], arm: str = "P", **kwargs: Any):
    return reward.compute_score(rows, store_path=store["path"], store_sha256=store["sha"], arm=arm, **kwargs)


def strip_time(outputs: Sequence[Mapping[str, float]]) -> List[Dict[str, float]]:
    return [{key: value for key, value in row.items() if key != "verifier_ms"} for row in outputs]


def read_details(directory: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for path in sorted(directory.iterdir()) for line in path.read_text().splitlines()]


# ------------------------------------------------------------------------------------------
# arm formulas


def expected_overall(arm: str, a: float, g: float, m: float, f: float, alpha: float, fw: float) -> float:
    b = {"A": a, "G": g, "G_gated": g}.get(arm, (g + m) / 2)
    if arm.endswith("_gated"):
        return (1 - fw) * a * (alpha + (1 - alpha) * b) + fw * f
    return (1 - fw) * (alpha * a + (1 - alpha) * b) + fw * f


def synthetic(a: float, g: float, m: float, f: float) -> ScoreResult:
    return ScoreResult(
        answer=a,
        grounding=g,
        semantics=m,
        format=f,
        overall={},
        joint=float(a == f == m == 1.0),
        checks=[],
        diagnostics={"verifier_ms": 0.25},
        coverage_status="supported",
        scorer_version="test",
        error_class="correct",
    )


SIGNALS = [(1, 1, 1, 1), (1, 0.5, 0.25, 1), (0, 0.9, 1, 1), (1, 0, 0, 0), (0, 0, 0, 0), (0, 1, 1, 0)]


@pytest.mark.parametrize("signals", SIGNALS)
@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("weights", [(0.8, 0.05), (0.6, 0.1)])
def test_arm_formulas_on_synthetic_score_results(signals, arm, weights) -> None:
    a, g, m, f = (float(x) for x in signals)
    alpha, fw = weights
    metrics = reward.row_metrics(synthetic(a, g, m, f), arm, alpha=alpha, format_weight=fw)
    assert tuple(metrics) == reward.METRIC_KEYS
    assert metrics["overall"] == pytest.approx(expected_overall(arm, a, g, m, f, alpha, fw), abs=1e-12)
    assert metrics["score"] == metrics["overall"]
    assert metrics["accuracy"] == metrics["answer_reward"] == a
    assert (metrics["grounding"], metrics["semantic"], metrics["cert_format"]) == (g, m, f)
    assert metrics["b_term"] == {"A": a, "G": g, "G_gated": g}.get(arm, (g + m) / 2)
    assert metrics["verifier_ms"] == 0.25


def test_worked_values_and_arm_resolution() -> None:
    right = {arm: reward.row_metrics(synthetic(1.0, 0.5, 0.25, 1.0), arm)["overall"] for arm in ARMS}
    assert right == pytest.approx({"A": 1.0, "G": 0.905, "P": 0.88125, "G_gated": 0.905, "P_gated": 0.88125})
    wrong = {arm: reward.row_metrics(synthetic(0.0, 1.0, 1.0, 1.0), arm)["overall"] for arm in ARMS}
    assert wrong == pytest.approx({"A": 0.05, "G": 0.24, "P": 0.24, "G_gated": 0.05, "P_gated": 0.05})
    pairs = (("A", False), ("G", False), ("P", False), ("G", True), ("P", True), ("P_gated", False))
    assert [reward.resolve_arm(arm, gated) for arm, gated in pairs] == ["A", "G", "P", "G_gated", "P_gated", "P_gated"]
    for arm, gated in (("A", True), ("S", False), ("p", False)):
        with pytest.raises(ValueError):
            reward.resolve_arm(arm, gated)


def test_out_of_range_metric_raises() -> None:
    bad = synthetic(1.0, 1.0, 1.0, 1.0)
    bad.diagnostics["verifier_ms"] = float("nan")
    with pytest.raises(ValueError):
        reward.row_metrics(bad, "P")


# ------------------------------------------------------------------------------------------
# batch contract


def test_every_row_has_the_same_bounded_float_keys(store) -> None:
    for arm, gated in (("A", False), ("G", False), ("P", False), ("G", True), ("P", True)):
        outputs = score(store, store["rows"], arm=arm, gated=gated)
        assert len(outputs) == len(store["rows"])
        for row in outputs:
            assert tuple(row) == reward.METRIC_KEYS
            assert all(type(value) is float and math.isfinite(value) for value in row.values())
            assert all(0.0 <= row[key] <= 1.0 for key in UNIT_KEYS) and row["verifier_ms"] >= 0.0


def test_signals_match_the_scorer_and_arms_differ_only_in_overall(store) -> None:
    by_arm = {arm: score(store, store["rows"], arm=arm) for arm in ARMS}
    for i, row in enumerate(store["rows"]):
        ext = next(e for e in store["extensions"] if e.instance_id == row["instance_id"])
        base = {"instance_id": ext.instance_id}
        result = scoring.score_certificate_response(row["response"] or "", base, ext, coord_frame="px_source")
        signals = (result.answer, result.grounding, result.semantics, result.format, result.joint)
        for arm, outputs in by_arm.items():
            out = outputs[i]
            assert (out["accuracy"], out["grounding"], out["semantic"], out["cert_format"], out["joint"]) == signals
            assert out["overall"] == result.overall[arm]
    for arm in ("G", "P"):
        assert strip_time(score(store, store["rows"], arm=arm, gated=True)) == strip_time(by_arm[f"{arm}_gated"])
    per_ext = len(responses(store["extensions"][0]))
    outputs = by_arm["P"]
    assert all(out["overall"] == out["joint"] == 1.0 for out in outputs[0::per_ext])  # oracle
    prose = outputs[1::per_ext]
    assert all(o["cert_format"] == 0.0 and o["semantic"] == o["grounding"] == o["accuracy"] == 1.0 for o in prose)
    operand = outputs[2::per_ext]
    assert all(o["accuracy"] == 0.0 and o["grounding"] == 1.0 and o["semantic"] < 1.0 for o in operand)


def test_batch_order_is_preserved_under_shuffle(store) -> None:
    rows = store["rows"]
    reference = strip_time(score(store, rows))
    order = list(range(len(rows)))
    random.Random(0).shuffle(order)
    shuffled = strip_time(score(store, [rows[i] for i in order]))
    assert [shuffled[order.index(i)] for i in range(len(rows))] == reference
    assert len({json.dumps(row, sort_keys=True) for row in reference}) > 3  # the rows really differ


def test_scores_do_not_depend_on_wall_clock_or_cpu_load(store, tmp_path, monkeypatch) -> None:
    first = strip_time(score(store, store["rows"]))
    stop = threading.Event()

    def burn() -> None:
        x = 0
        while not stop.is_set():
            x = (x * 31 + 7) % 1_000_003

    threads = [threading.Thread(target=burn, daemon=True) for _ in range(2)]
    for thread in threads:
        thread.start()
    try:
        loaded = strip_time(score(store, store["rows"]))
    finally:
        stop.set()
        for thread in threads:
            thread.join()
    clock = iter(i * 0.2 for i in range(10**6))  # every row now looks like 200 ms > SLOW_ROW_MS
    monkeypatch.setattr(scoring.time, "perf_counter", lambda: next(clock))
    slow = score(store, store["rows"], details_dir=str(tmp_path / "slow"))
    monkeypatch.undo()
    assert first == loaded == strip_time(slow)
    assert all(row["verifier_ms"] >= scoring.SLOW_ROW_MS for row in slow)
    records = read_details(tmp_path / "slow")
    assert len(records) == len(store["rows"]) and all(r["diagnostics"]["slow_row"] for r in records)


def test_budget_cap_is_a_deterministic_envelope_failure(store, tmp_path) -> None:
    ext = store["extensions"][0]
    outputs = score(store, [reward_input(ext, responses(ext)[4])] * 2, details_dir=str(tmp_path / "d"))
    assert outputs[0]["semantic"] == outputs[0]["cert_format"] == 0.0
    assert strip_time(outputs)[0] == strip_time(outputs)[1]
    records = read_details(tmp_path / "d")
    assert [r["diagnostics"]["envelope_error"] for r in records] == ["budget", "budget"]
    assert all(check["error_code"] == "budget" for check in records[0]["checks"])


def test_prompt_and_extra_columns_are_ignored(store) -> None:
    rows = store["rows"][:12]
    extra = {"uid": "other", "domain": "toy", "image_sizes": [{"width": 1}]}
    assert strip_time(score(store, rows)) == strip_time(score(store, [dict(row, **extra) for row in rows]))
    no_prompt = [{key: value for key, value in row.items() if key != "prompt_cert"} for row in rows]
    assert strip_time(score(store, rows)) == strip_time(score(store, no_prompt))  # integrity only, never a score


def test_prompt_must_be_the_frozen_prompt_of_the_reward_frame(store) -> None:
    """Review EX-6 / LEAK-02: a reward frame the prompt did not declare, or a drifted prompt, raises."""
    rows = store["rows"][:4]
    with pytest.raises(CertificateVersionError, match="prompt sha256"):
        score(store, [dict(row, prompt_cert="<image>something else") for row in rows])
    with pytest.raises(CertificateVersionError, match="px_source prompt"):
        score(store, [dict(row, prompt_cert=TOY_PROMPTS["rel1000"]) for row in rows])
    with pytest.raises(CertificateVersionError, match="rel1000 prompt"):
        score(store, rows, coord_frame="rel1000")
    rel = score(store, [dict(row, prompt_cert=TOY_PROMPTS["rel1000"]) for row in rows], coord_frame="rel1000")
    assert len(rel) == len(rows)


def test_empty_batch_and_store_encodings(tmp_path) -> None:
    import zstandard as zstd

    first, second = toy_extension(7), toy_extension(8)
    plain = tmp_path / "plain.jsonl"
    write_store(plain, [first.to_dict()], compress=False)
    frames = tmp_path / "two_frames.jsonl.zst"  # concatenated zstd frames are read across
    lines = [json.dumps(ext.to_dict()).encode() + b"\n" for ext in (first, second)]
    frames.write_bytes(b"".join(zstd.ZstdCompressor().compress(line) for line in lines))
    for path, extensions in ((plain, [first]), (frames, [first, second])):
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        kwargs = {"store_path": str(path), "store_sha256": sha, "arm": "G"}
        assert reward.compute_score([], **kwargs) == []
        batch = [reward_input(ext, responses(ext)[0]) for ext in extensions]
        assert [out["overall"] for out in reward.compute_score(batch, **kwargs)] == [1.0] * len(extensions)


def test_store_is_loaded_once_per_process(store, monkeypatch) -> None:
    calls = []
    original = reward.ExtensionStore.load.__func__

    def counting(cls, path, sha256):
        calls.append(path)
        return original(cls, path, sha256)

    monkeypatch.setattr(reward, "_STORES", {})
    monkeypatch.setattr(reward.ExtensionStore, "load", classmethod(counting))
    for arm in ("A", "G", "P"):
        score(store, store["rows"][:4], arm=arm)
    assert len(calls) == 1
    assert reward.load_store(store["path"], store["sha"]) is reward.load_store(store["path"], store["sha"].upper())


def test_scipy_assignment_is_required(store, monkeypatch) -> None:
    monkeypatch.setattr(scoring._legacy, "linear_sum_assignment", None)
    with pytest.raises(RuntimeError, match="linear_sum_assignment"):
        score(store, store["rows"][:1])


# ------------------------------------------------------------------------------------------
# integrity: every mismatch raises (never a model zero)


def test_store_integrity_errors_raise(store, tmp_path) -> None:
    rows = store["rows"][:1]

    def run(path: Path, sha: str, batch: Sequence[Mapping[str, Any]] = rows):
        return reward.compute_score(batch, store_path=str(path), store_sha256=sha, arm="P")

    with pytest.raises(CertificateVersionError, match="sha256"):
        run(store["path"], "0" * 64)
    with pytest.raises(CertificateVersionError, match="64 hex"):
        run(store["path"], "sha256:abc")
    ext = store["extensions"][0]
    with pytest.raises(CertificateVersionError, match="duplicate"):
        run(tmp_path / "dup.zst", write_store(tmp_path / "dup.zst", [ext.to_dict()] * 2))
    tampered = ext.to_dict()
    tampered["base"]["trace_record_hash"] = "blake3:" + "f" * 64  # id no longer hashes from the base
    with pytest.raises(ExtensionMismatchError):
        run(tmp_path / "tampered.zst", write_store(tmp_path / "tampered.zst", [tampered]))
    stale = toy_extension(0, version="0.9.0")
    sha = write_store(tmp_path / "stale.zst", [stale.to_dict()])
    with pytest.raises(CertificateVersionError, match="does not match loaded adapter"):
        run(tmp_path / "stale.zst", sha, [reward_input(stale, "{}")])


def test_join_errors_raise(store) -> None:
    ext, other = store["extensions"][0], store["extensions"][1]
    good = reward_input(ext, responses(ext)[0])
    unknown = json.dumps({"cert_extension_id": "blake3:" + "e" * 64})
    cases = [
        (dict(good, instance_id=other.instance_id), ExtensionMismatchError),
        ({key: value for key, value in good.items() if key != "instance_id"}, ExtensionMismatchError),
        (dict(good, cert_extension_id=other.certificate_extension_id), ExtensionMismatchError),
        (dict(good, ground_truth=unknown, cert_extension_id=None), CertificateVersionError),
        (dict(good, ground_truth=json.dumps({"type": "integer", "value": 3})), ValueError),
        (dict(good, ground_truth="not json"), ValueError),
        (dict(good, ground_truth=json.dumps({"cert_extension_id": ext.certificate_extension_id, "x": 1})), ValueError),
    ]
    for row, error in cases:
        with pytest.raises(error):
            score(store, [store["rows"][0], row])
    as_mapping = dict(good, ground_truth={"cert_extension_id": ext.certificate_extension_id})
    assert score(store, [as_mapping])[0]["overall"] == 1.0
    bad_kwargs = ({"arm": "S"}, {"gated": True, "arm": "A"}, {"coord_frame": "px_shown"}, {"alpha": 1.5})
    for kwargs in bad_kwargs + ({"format_weight": -0.1},):
        with pytest.raises(ValueError):
            score(store, [good], **{"arm": "P", **kwargs})


# ------------------------------------------------------------------------------------------
# details files


def test_details_files_name_phase_and_rows(store, tmp_path) -> None:
    details = tmp_path / "details"
    train_rows, val_rows = store["rows"], store["val_rows"]  # splits "rl" and "e1_dev"
    train_out = score(store, train_rows, arm="G", details_dir=str(details))
    score(store, val_rows, arm="G", details_dir=str(details))
    pattern = re.compile(rf"^(train|val)_{os.getpid()}_(\d{{6}})\.jsonl$")
    files = sorted(details.iterdir(), key=lambda path: int(pattern.match(path.name).group(2)))
    matches = [pattern.match(path.name) for path in files]
    assert [m.group(1) for m in matches] == ["train", "val"]
    assert int(matches[1].group(2)) == int(matches[0].group(2)) + 1
    records = [json.loads(line) for line in files[0].read_text().splitlines()]
    assert [r["row"] for r in records] == list(range(len(train_rows)))
    assert [r["instance_id"] for r in records] == [row["instance_id"] for row in train_rows]
    assert [r["metrics"] for r in records] == train_out
    first = records[0]
    assert (first["phase"], first["arm"], first["split"]) == ("train", "G", "rl")
    assert first["reward_name"] == reward.REWARD_NAME
    assert first["response_length"] == 17 and first["store_sha256"] == store["sha"]
    assert {check["check_id"] for check in first["checks"]} >= set(ADAPTER.m_families)
    assert {"G_shown", "M_shown"} <= set(first["diagnostics"])
    assert "public prompt" not in files[0].read_text()  # the prompt column is never logged


def test_details_never_overwrite(tmp_path) -> None:
    first = reward._write_details(tmp_path, "train", 3, [{"a": 1}])
    second = reward._write_details(tmp_path, "train", 3, [{"a": 2}])
    assert first != second
    assert (json.loads(first.read_text()), json.loads(second.read_text())) == ({"a": 1}, {"a": 2})


# ------------------------------------------------------------------------------------------
# EasyR1 loader (verl/workers/reward/function.py and config.py, replayed without torch)

SKIP_KEYS = {
    "input_ids",
    "attention_mask",
    "position_ids",
    "responses",
    "raw_prompt_ids",
    "multi_modal_data",
    "multi_modal_inputs",
}


def easyr1_load(reward_function: str, kwargs: Mapping[str, Any], module_name: str = "custom_reward_fn"):
    """``RewardConfig.post_init`` plus ``AutoRewardManager.__init__``, step for step."""
    if ":" not in reward_function:
        path, name = reward_function, "main"
    else:
        path, name = reward_function.rsplit(":", maxsplit=1)
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    reward_fn = getattr(module, name)
    return module, functools.partial(reward_fn, **kwargs), getattr(module, "REWARD_TYPE", "batch")


def easyr1_reward_inputs(batch: Mapping[str, np.ndarray], responses: Sequence[str], lengths: Sequence[int]):
    """``_build_reward_input`` over a DataProto-like ``non_tensor_batch``."""
    inputs = []
    for index, (response, length) in enumerate(zip(responses, lengths)):
        row = {"response": response, "response_length": length, "ground_truth": batch["ground_truth"][index]}
        for key, values in batch.items():
            if key in SKIP_KEYS or key in row:
                continue
            try:
                if len(values) != len(responses):
                    continue
                row[key] = values[index]
            except Exception:
                continue
        inputs.append(row)
    return inputs


def test_reward_files_pass_the_release_hygiene_scan() -> None:
    script = REPO / "scripts" / "check_public_release.py"
    spec = importlib.util.spec_from_file_location("release_check_for_reward", script)
    release_check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(release_check)
    assert len(ENTRY.read_text(encoding="utf-8").splitlines()) <= 20
    for path in (ENTRY, Path(reward.__file__), Path(__file__)):
        assert release_check.scan_text_for_sensitive_values(path.read_text(encoding="utf-8")) == [], path.name


def test_reward_entry_loads_like_easyr1(store, tmp_path) -> None:
    kwargs = {"store_path": store["path"], "store_sha256": store["sha"], "arm": "P", "details_dir": str(tmp_path)}
    saved = sys.modules.get("custom_reward_fn")
    try:
        module, reward_fn, reward_type = easyr1_load(f"{ENTRY}:compute_score", kwargs)
        assert (module.REWARD_NAME, reward_type) == ("trace_certificate_v1", "batch")
        assert module.compute_score is reward.compute_score
        rows = store["rows"]
        columns = ("ground_truth", "instance_id", "cert_extension_id", "prompt_cert", "uid")
        batch = {key: np.array([row[key] for row in rows], dtype=object) for key in columns}
        batch["multi_modal_data"] = np.array([{"images": ["x"]}] * len(rows), dtype=object)
        batch["short_column"] = np.array(["only one"], dtype=object)
        inputs = easyr1_reward_inputs(batch, [row["response"] or "" for row in rows], [17] * len(rows))
        assert "multi_modal_data" not in inputs[0] and "short_column" not in inputs[0]
        scores = reward_fn(inputs)
        last_token = [score_row["overall"] for score_row in scores]  # reward_tensor[i, len - 1]
        metrics: Dict[str, List[float]] = {}
        for score_row in scores:
            for key, value in score_row.items():
                metrics.setdefault(key, []).append(value)
        assert len(last_token) == len(rows) and set(metrics) == set(reward.METRIC_KEYS)
        assert all(len(values) == len(rows) for values in metrics.values())
        reduced = {key: float(np.mean(np.asarray(values, dtype=float))) for key, values in metrics.items()}
        assert all(math.isfinite(value) for value in reduced.values())  # reduce_reward_metrics
    finally:
        if saved is None:
            sys.modules.pop("custom_reward_fn", None)
        else:
            sys.modules["custom_reward_fn"] = saved


def test_reward_entry_loads_in_a_fresh_interpreter(tmp_path) -> None:
    store_path = tmp_path / "empty.jsonl"
    sha = write_store(store_path, [], compress=False)
    script = "\n".join(
        [
            "import importlib.util, sys",
            f"spec = importlib.util.spec_from_file_location('custom_reward_fn', {str(ENTRY)!r})",
            "module = importlib.util.module_from_spec(spec)",
            "sys.modules['custom_reward_fn'] = module",
            "spec.loader.exec_module(module)",
            f"out = module.compute_score([], store_path={str(store_path)!r}, store_sha256={sha!r}, arm='P')",
            "assert out == [] and 'torch' not in sys.modules and 'transformers' not in sys.modules",
            "print(module.REWARD_NAME, module.REWARD_TYPE, module.compute_score.__module__)",
        ]
    )
    pythonpath = os.pathsep.join(filter(None, [str(REPO / "src"), os.environ.get("PYTHONPATH")]))
    env = dict(os.environ, PYTHONPATH=pythonpath)
    done = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["trace_certificate_v1", "batch", "trace_tasks.experimental.certificates.reward"]


def rlvr_backend() -> Path:
    root = os.environ.get("TRACE_RLVR_ROOT")
    if not root:
        gate_skip("TRACE_RLVR_ROOT is unset; point it at the trace-rlvr checkout to pin the release reward")
    backend = Path(root) / "rlvr" / "easyr1_backend"
    if not backend.is_dir():
        gate_skip(f"TRACE_RLVR_ROOT has no rlvr/easyr1_backend: {root}")
    return backend


def test_easyr1_loader_replay_matches_the_vendored_source() -> None:
    reward_dir = rlvr_backend() / "verl" / "workers" / "reward"
    function = (reward_dir / "function.py").read_text(encoding="utf-8")
    config = (reward_dir / "config.py").read_text(encoding="utf-8")
    for snippet in (
        'spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)',
        'sys.modules["custom_reward_fn"] = module',
        'reward_type = getattr(module, "REWARD_TYPE", "batch")',
        "self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)",
        'reward_tensor[i, cur_response_length - 1] = score["overall"]',
        '"ground_truth": data.non_tensor_batch["ground_truth"][index]',
        "if key in _REWARD_INPUT_SKIP_KEYS or key in reward_input:",
    ):
        assert snippet in function, snippet
    skip_block = function.split("_REWARD_INPUT_SKIP_KEYS = {", 1)[1].split("}", 1)[0]
    assert set(re.findall(r'"(\w+)"', skip_block)) == SKIP_KEYS
    assert 'self.reward_function.rsplit(":", maxsplit=1)' in config


# ------------------------------------------------------------------------------------------
# the release reward (rlvr/easyr1_backend/examples/reward_function/trace_rlvr.py) is unchanged

RELEASE_BOXES = [[545.0, 278.5, 603.0, 330.5], [413.0, 278.5, 471.0, 330.5], [281.0, 278.5, 339.0, 330.5]]
RELEASE_CERT = {"kind": "weighted_sum", "terms": [{"annotation_index": 0, "weight": 2, "distance": 3, "product": 6}]}
RELEASE_GOLDEN = [  # (answer value, response, overall, accuracy, format, zero_reward)
    (12, 'Reasoning...\n{"answer": 12}', 1.0, 1.0, 1.0, 0.0),
    (12, '{"answer": 1}', 0.05, 0.0, 1.0, 1.0),
    (12, json.dumps({"annotation": RELEASE_BOXES, "answer": 12}), 0.95, 1.0, 0.0, 0.0),
    (12, json.dumps({"annotation": RELEASE_BOXES, "certificate": RELEASE_CERT, "answer": 12}), 0.95, 1.0, 0.0, 0.0),
    (12, json.dumps({"annotation": RELEASE_BOXES, "certificate": RELEASE_CERT, "answer": 3}), 0.0, 0.0, 0.0, 1.0),
    (12, "the answer is 12", 0.0, 0.0, 0.0, 1.0),
    (12, '```json\n{"answer": true}\n```', 0.05, 0.0, 1.0, 1.0),
    (12, "", 0.0, 0.0, 0.0, 1.0),
    (1, '{"answer": true}', 1.0, 1.0, 1.0, 0.0),  # legacy true == 1 (the certificate A guards this)
    (15, '{"answer": 15.0}', 1.0, 1.0, 1.0, 0.0),
]


def test_release_trace_rlvr_reward_is_unchanged(store) -> None:
    path = rlvr_backend() / "examples" / "reward_function" / "trace_rlvr.py"
    module, release_fn, reward_type = easyr1_load(f"{path}:compute_score", {}, module_name="release_trace_rlvr_test")
    sys.modules.pop("release_trace_rlvr_test", None)
    assert (module.REWARD_NAME, reward_type) == ("trace_answer_only", "batch")
    contract = json.dumps(resolve_reward_contract(answer_type="integer", annotation_type="bbox_set").to_dict())
    rows = [
        {
            "response": response,
            "response_length": 10,
            "ground_truth": json.dumps({"type": "integer", "value": value}),
            "annotation_gt": json.dumps({"type": "bbox_set", "value": RELEASE_BOXES}),
            "reward_contract": contract,
            "instance_id": "row",
        }
        for value, response, *_ in RELEASE_GOLDEN
    ]
    before = release_fn(rows)
    score(store, store["rows"], arm="P")  # the certificate reward leaves no global state behind
    assert release_fn(rows) == before
    for out, (_, _, overall, accuracy, fmt, zero) in zip(before, RELEASE_GOLDEN):
        expected = {"overall": overall, "score": overall, "accuracy": accuracy, "answer_reward": accuracy}
        assert out == dict(expected, format=fmt, zero_reward=zero, trace_reward=1.0)


# ------------------------------------------------------------------------------------------
# real adapters (frozen fixture oracles) through the exporter's store format


def fixture_extension(record: Mapping[str, Any], adapter: Adapter) -> CertificateExtension:
    """The extension of a frozen adapter fixture record (its stored oracle and SFT object)."""
    train = record["train"]
    base = {key: train[key] for key in ("instance_id", "task", "query_id", "instance_version", "instance_seed")}
    base.update(trace_record_hash=train["trace_ref"]["trace_record_hash"], split="rl")
    return make_extension(
        base=base,
        identity=adapter.identity(),
        labels={key: train[key] for key in ("answer_gt", "annotation_gt", "reward_contract")},
        image_size=record["image_size"],
        oracle=record["oracle"],
        sft_target=record["sft"],
    )


@pytest.mark.parametrize("name", ["selected_set", "filtered_mean", "weighted_sum", "path"])
def test_real_adapter_rows_match_the_scorer(name, tmp_path) -> None:
    from trace_tasks.experimental.certificates.export import STORE_NAME, write_store as export_write_store

    records = []
    for path in sorted((FIXTURES / name).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records += [r for r in payload.get("records", []) if "oracle" in r and "sft" in r]
    if not records:
        pytest.skip(f"no {name} fixture records")
    adapter = registry.resolve(records[0]["train"]["task"], records[0]["train"]["query_id"])
    if not isinstance(adapter, Adapter):
        pytest.skip(f"{name}: {adapter.reason}")
    extensions = [fixture_extension(record, adapter) for record in records[:3]]
    path = tmp_path / "private" / STORE_NAME
    path.parent.mkdir()
    export_write_store(path, [ext.to_dict() for ext in extensions])
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    rows, expected, oracle_rows = [], [], []
    for ext in extensions:
        oracle_rows.append(len(rows))
        corruptions = adapter.corruptions(ext.oracle, dict(ext.sft_target), int(ext.base["instance_seed"]))
        for obj in [ext.sft_target] + [corruption.response_obj for corruption in corruptions]:
            response = json.dumps(obj)
            rows.append(reward_input(ext, response))
            base = {"instance_id": ext.instance_id}
            expected.append(scoring.score_certificate_response(response, base, ext, coord_frame="px_source"))
    for arm in ("A", "G", "P", "P_gated"):
        kwargs = {"store_path": str(path), "store_sha256": sha, "arm": arm, "details_dir": str(tmp_path / "d")}
        outputs = reward.compute_score(rows, **kwargs)
        for out, result in zip(outputs, expected):
            assert out["overall"] == result.overall[arm]
            signals = (result.answer, result.grounding, result.semantics, result.format)
            assert (out["accuracy"], out["grounding"], out["semantic"], out["cert_format"]) == signals
        oracle = [outputs[i] for i in oracle_rows]
        assert all(out["accuracy"] == out["semantic"] == out["cert_format"] == 1.0 for out in oracle)
    assert len({out["overall"] for out in outputs}) > 1
