"""Ingestion, identity audit and inventory on a small synthetic archive, dataset manifest and local-run spool.

The fixtures are built with the pinned exporter / archive-spool helpers themselves, so a passing test means the module
accepts exactly what the protocol produces and rejects tampering. Local runs carry what a study run carries (plan,
row list, campaign summary, served-model marker or registered view, pinned provenance). A real-data check runs only
when TRACE_MERGE_WORK_ROOT points at an ingested work root.
"""

from __future__ import annotations

import hashlib
import json
import os
import types
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from trace_merge import ingest, inventory
from trace_merge.config import DEFAULT_CONFIG, ROLES, load_config, sha256_file

RUN_ID = "qwen2.5-vl-7b-comparison-temp06-seeds42-44-v1"
BENCH_ROW = "erqa"          # row-scored binary benchmark
BENCH_AGG = "mathvista"     # aggregate-only benchmark
N_ROWS = 4
SEED = 42
MEDIA_GEN = {"media_contract_version": "trace-final25-media-v2", "media_transport": "file-url",
             "min_image_pixels": 3136, "max_image_pixels": 12845056, "max_image_side": None,
             "image_jpeg_quality": None}
SAMPLING = {"max_tokens": 4096, "presence_penalty": 0.0, "repetition_penalty": 1.0, "seed": SEED, "temperature": 0.6,
            "top_k": -1, "top_p": 1.0}
VIEW_SLUGS = {"V@P1": "vero-p1", "M050@P1": "m050-p1", "CV050@P1": "cv050-p1", "CT050@P1": "ct050-p1",
              "B@P2": "qwen25vl7b-p2", "M050@P2": "m050-p2"}


def h(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


EXP = ingest.protocol_module(ingest.EXPORTER)
MC = ingest.protocol_module(ingest.MEDIA_CONTRACT)
HF = ingest.protocol_module(ingest.ARCHIVE_LIB)
ARCHIVE_DATASET_SHA = h("producer-dataset-snapshot")


def manifest_rows(bench: str) -> list[dict]:
    rows = []
    for i in range(N_ROWS):
        image = h(f"{bench}-image-{i // 2}")      # rows 0/1 and 2/3 share an image
        media = [{"type": "image", "sha256": image}]
        mset = MC.media_set_sha256(media)
        row_hash = h(f"{bench}-row-{i}")
        rows.append({"index": str(100 + i), "ordinal": i, "source_row_hash": row_hash,
                     "source_record_sha256": MC.source_record_sha256(row_hash, mset), "media_set_sha256": mset,
                     "media": media})
    return rows


def prompt_text(bench: str, i: int) -> str:
    return f"question {bench} {i}"


def prompt_tokens_of(role: str) -> int:
    return 1100 if role == "V" else 600


# --------------------------------------------------------------------------------------------- archive builder
def response_rows(cfg, role: str, bench: str, *, seed: int = SEED, identity_rows: list[dict] | None = None,
                  ) -> list[dict]:
    spec = cfg.role(role)
    mid, rev = spec["archived_model_id"], spec["archived_model_revision"]
    media_inputs = EXP._source_media_inputs(types.SimpleNamespace(manifest={"aggregate": {"generation": MEDIA_GEN}}))
    settings = EXP.normalize_generation_settings(dict(SAMPLING, seed=seed), model_id=mid, media_inputs=media_inputs)
    rows = []
    for m in identity_rows or manifest_rows(bench):
        media = [x["sha256"] for x in m["media"]]
        prompt = EXP.prompt_sha256(prompt_text(bench, m["ordinal"]))
        request = EXP.public_request_sha256_from_hashes(prompt_digest=prompt, ordered_media_sha256=media,
                                                        generation_settings=settings)
        contract = EXP.prompt_contract_sha256(dataset_sha256=ARCHIVE_DATASET_SHA,
                                              source_row_sha256=m["source_row_hash"],
                                              source_record_digest=m["source_record_sha256"],
                                              media_digest=m["media_set_sha256"], prompt_digest=prompt,
                                              request_digest=request)
        sid = EXP._sample_id(run_id=RUN_ID, model_id=mid, model_revision=rev, seed=seed, benchmark_id=bench,
                             source_index=m["index"], source_ordinal=m["ordinal"],
                             source_row_sha256=m["source_row_hash"])
        text = f"Reasoning for {bench} {m['ordinal']} by {role} (seed {seed}). The answer is {m['ordinal'] % 2}"
        rsha = EXP.response_sha256(text)
        rows.append({
            "schema_version": EXP.SCHEMA_VERSION, "run_id": RUN_ID, "model_id": mid, "model_revision": rev,
            "seed": seed, "benchmark_id": bench, "dataset_split": "default",
            "dataset_revision": f"{EXP.DATASET_REVISION_PREFIX}:{ARCHIVE_DATASET_SHA}", "sample_id": sid,
            "response_id": EXP._response_id(sid, rsha), "source_index": m["index"], "source_ordinal": m["ordinal"],
            "dataset_sha256": ARCHIVE_DATASET_SHA, "source_row_sha256": m["source_row_hash"],
            "source_record_sha256": m["source_record_sha256"], "media_set_sha256": m["media_set_sha256"],
            "ordered_media_sha256": media, "prompt_sha256": prompt, "request_sha256": request,
            "prompt_contract_sha256": contract, "response_sha256": rsha,
            "contract_version": EXP.RESPONSE_CONTRACT_VERSION, "response_text": text,
            "finish_reason": "length" if m["ordinal"] == 3 and seed != 43 else "stop",
            "generation_settings_json": EXP.canonical_json(settings), "prompt_tokens": prompt_tokens_of(role),
            "completion_tokens": 50 + m["ordinal"] + (seed - SEED)})
    return rows


def extraction_rows(resp: list[dict]) -> list[dict]:
    common = [f.name for f in EXP.COMMON_SAMPLE_FIELDS]
    out = []
    for r in resp:
        judged = r["source_ordinal"] == 1
        value = "None" if judged else str(r["source_ordinal"] % 2)   # a literal "None" answer is content, not null
        judge_output = "None" if judged else None
        out.append({**{k: r[k] for k in common}, "contract_version": EXP.EXTRACTION_CONTRACT_VERSION,
                    "extraction_status": "resolved", "extraction_value_type": "string",
                    "extraction_value_json": EXP.canonical_json(value), "extraction_candidates_json": "[]",
                    "extraction_sha256": EXP.canonical_sha256({
                        "contract_version": EXP.EXTRACTION_CONTRACT_VERSION, "sample_id": r["sample_id"],
                        "status": "resolved", "value": value, "candidates": []}),
                    "extraction_method": "judge" if judged else "deterministic_parser", "used_judge": judged,
                    "judge_model_id": "Qwen/Qwen3-32B" if judged else None,
                    "judge_model_revision": "9216db5781bf21249d130ec9da846c4624c16137" if judged else None,
                    "judge_output": judge_output,
                    "judge_output_sha256": h(judge_output) if judge_output is not None else None, "retry_count": 0})
    return out


def score_rows(resp: list[dict], bench: str, aggregate_only: bool) -> tuple[list[dict], float]:
    first = resp[0]
    identity = {k: first[k] for k in ("run_id", "model_id", "seed", "benchmark_id")}
    common = {k: first[k] for k in ("schema_version", "run_id", "model_id", "model_revision", "seed", "benchmark_id",
                                    "dataset_split", "dataset_revision", "dataset_sha256")}
    rows = []
    values = []
    for r in resp:
        value = float(r["source_ordinal"] % 2 == 0)
        values.append(value)
        if aggregate_only:
            continue
        rows.append({**common, "score_id": EXP._score_id(identity, scope="row", sample_id=r["sample_id"]),
                     "score_scope": "row", **{k: r[k] for k in (
                         "sample_id", "response_id", "source_index", "source_ordinal", "source_row_sha256",
                         "source_record_sha256", "media_set_sha256", "prompt_sha256", "request_sha256",
                         "prompt_contract_sha256", "response_sha256")},
                     "contract_version": EXP.SCORE_CONTRACT_VERSION, "scorer_id": f"{bench}-primary-v1",
                     "metric_id": "primary", "score_unit": "fraction", "score_value": value,
                     "score_value_json": EXP.canonical_json(value), "excluded": False, "evaluated_rows": None})
    aggregate = 100.0 * sum(values) / len(values)
    rows.append({**common, "score_id": EXP._score_id(identity, scope="aggregate", sample_id=None),
                 "score_scope": "aggregate", "sample_id": None, "response_id": None, "source_index": None,
                 "source_ordinal": None, "source_row_sha256": None, "source_record_sha256": None,
                 "media_set_sha256": None, "prompt_sha256": None, "request_sha256": None,
                 "prompt_contract_sha256": None, "response_sha256": None,
                 "contract_version": EXP.SCORE_CONTRACT_VERSION, "scorer_id": f"{bench}-primary-v1",
                 "metric_id": "primary", "score_unit": "percent", "score_value": aggregate,
                 "score_value_json": EXP.canonical_json(aggregate), "excluded": False, "evaluated_rows": len(resp)})
    return rows, aggregate


def write_json(path: Path, value) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=1, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest(), len(data)


class World:
    """Synthetic archive + dataset manifest + config; mutate() hooks let tests inject faults before writing."""

    def __init__(self, tmp: Path, seeds: tuple[int, ...] = (SEED,)):
        self.tmp = tmp
        self.seeds = list(seeds)
        self.archive_root = tmp / "archive"
        self.run_root = self.archive_root / "runs" / RUN_ID
        self.clone = tmp / "clone"
        self.work = tmp / "work"
        self.base_cfg = load_config(DEFAULT_CONFIG, work_root=self.work, protocol_clone=self.clone)
        self.benchmarks = [BENCH_ROW, BENCH_AGG]
        self.manifest_override: dict[str, list[dict]] = {}     # benchmark -> manifest (public) identity rows
        self.archive_override: dict[str, list[dict]] = {}      # benchmark -> identity rows the archive carries
        self.metadata_files: dict[str, list[dict]] = {}

    def build(self, mutate=None) -> "World":
        mutate = mutate or (lambda kind, role, bench, rows: rows)
        cfg = self.base_cfg
        dm_path = self.clone / "rlvr/evaluation/.work/LMUData/trace_eval_v1_dataset_manifest.json"
        datasets = {}
        for bench in self.benchmarks:
            rows = mutate("manifest", None, bench, self.manifest_override.get(bench) or manifest_rows(bench))
            datasets[bench] = {"status": "ready", "rows": len(rows), "row_media": rows, "alias": bench,
                               "dataset_snapshot_sha256": MC.dataset_snapshot_sha256(rows),
                               "metadata_files": self.metadata_files.get(bench, [])}
        write_json(dm_path, {"schema_version": MC.DATASET_MANIFEST_SCHEMA, "datasets": datasets,
                             "dataset_snapshot_sha256": h("clone-snapshot")})
        artifacts, results, meta_files = [], [], []
        configs = {c: {"config_name": c, "contract_version": v, "parts": 0, "rows": 0}
                   for c, v in (("responses", EXP.RESPONSE_CONTRACT_VERSION),
                                ("extractions", EXP.EXTRACTION_CONTRACT_VERSION),
                                ("scores", EXP.SCORE_CONTRACT_VERSION))}
        schemas = {"responses": EXP.RESPONSE_SCHEMA, "extractions": EXP.EXTRACTION_SCHEMA, "scores": EXP.SCORE_SCHEMA}
        for role in ("B", "T", "V"):
            mid = cfg.role(role)["archived_model_id"]
            for seed in self.seeds:
                for bench in self.benchmarks:
                    ident = self.archive_override.get(bench)

                    def fresh():
                        return response_rows(cfg, role, bench, seed=seed, identity_rows=ident)

                    resp = mutate("responses", role, bench, fresh())
                    ext = mutate("extractions", role, bench, extraction_rows(fresh()))
                    sc, agg = score_rows(fresh(), bench, bench == BENCH_AGG)
                    sc = mutate("scores", role, bench, sc)
                    results.append({"benchmark_id": bench, "evaluated_rows": N_ROWS, "model_id": mid,
                                    "score": mutate("aggregate", role, bench, agg), "scoring_scope": "aggregate",
                                    "seed": seed})
                    for cname, rows in (("responses", resp), ("extractions", ext), ("scores", sc)):
                        part_dir = f"{cname}/run={RUN_ID}/model={mid}/seed={seed}/benchmark={bench}"
                        tmp_path = self.run_root / "data" / part_dir / "tmp.parquet"
                        tmp_path.parent.mkdir(parents=True, exist_ok=True)
                        pq.write_table(pa.Table.from_pylist(rows, schema=schemas[cname]), tmp_path)
                        psha = sha256_file(tmp_path)
                        ppath = self.run_root / "data" / part_dir / f"part-{psha}.parquet"
                        tmp_path.rename(ppath)
                        prel = ppath.relative_to(self.run_root).as_posix()
                        mrel = f"metadata/parts/{part_dir}/part-{psha}.manifest.json"
                        msha, msize = write_json(self.run_root / mrel, {
                            "config_name": cname, "contract_version": configs[cname]["contract_version"],
                            "identity": {"benchmark_id": bench, "model_id": mid, "run_id": RUN_ID, "seed": seed},
                            "parquet_path": prel, "parquet_sha256": psha, "parquet_size": ppath.stat().st_size,
                            "rows": len(rows), "provenance": {"producer_code_revision": "x" * 40,
                                                              "producer_code_sha256": h("producer")}})
                        artifacts.append({"benchmark_id": bench, "config_name": cname, "manifest_path": mrel,
                                          "manifest_sha256": msha, "manifest_size": msize, "model_id": mid,
                                          "parquet_path": prel, "parquet_sha256": psha,
                                          "parquet_size": ppath.stat().st_size, "rows": len(rows), "run_id": RUN_ID,
                                          "seed": seed})
                        configs[cname]["parts"] += 1
                        configs[cname]["rows"] += len(rows)
            spec = cfg.role(role)
            sha, size = write_json(self.run_root / f"metadata/models/{mid}.json", {
                "model_id": mid, "model_revision": spec["archived_model_revision"], "repository_id": spec["repo_id"],
                "repository_revision": spec["source_revision"]})
            meta_files.append({"path": f"metadata/models/{mid}.json", "sha256": sha, "size": size})
        rsha, rsize = write_json(self.run_root / "metadata/results/benchmark_scores.json",
                                 {"benchmark_scores": results})
        meta_files.append({"path": "metadata/results/benchmark_scores.json", "sha256": rsha, "size": rsize})
        gen = [{"model_id": cfg.role(r)["archived_model_id"], "seed": s, "temperature": 0.6, "top_p": 1.0,
                "top_k": -1, "max_output_tokens": 4096, "presence_penalty": 0.0, "repetition_penalty": 1.0,
                "min_image_pixels": 3136, "max_image_pixels": 12845056, "media_transport": "file-url"}
               for r in ("B", "T", "V") for s in self.seeds]
        sha, size = write_json(self.run_root / f"metadata/runs/{RUN_ID}.json", {
            "generation_settings": gen,
            "judge_model": {"model_id": "Qwen/Qwen3-32B", "model_revision": "9216db5781bf21249d130ec9da846c4624c16137"}})
        meta_files.append({"path": f"metadata/runs/{RUN_ID}.json", "sha256": sha, "size": size})
        msha, _ = write_json(self.run_root / "metadata/manifest.json", {
            "artifacts": artifacts, "configs": list(configs.values()), "metadata_files": meta_files,
            "release_files": [], "run_ids": [RUN_ID]})
        raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        # the study config enables the CountQA alias (DESIGN amendment A1); synthetic worlds start without it
        raw.pop("identity", None)
        raw["archive"].update(manifest_sha256=msha, result_sha256=rsha, local_root=str(self.archive_root),
                              seeds=list(self.seeds))
        cfg_path = self.tmp / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        self.cfg = load_config(cfg_path, work_root=self.work, protocol_clone=self.clone)
        self.receipt = {"manifest_sha256": msha, "result_sha256": rsha}
        return self

    def ingest(self, **kw):
        return ingest.ingest_archive(self.cfg, benchmarks=self.benchmarks, seeds=self.seeds, log=lambda m: None, **kw)


@pytest.fixture()
def world(tmp_path):
    return World(tmp_path).build()


@pytest.fixture()
def ingested(world):
    world.ingest()
    return world


# --------------------------------------------------------------------------------------------- archive tests
def test_archive_ingestion_writes_normalized_table(world):
    paths = world.ingest()
    df = pq.read_table(paths["predictions"]).to_pandas()
    assert len(df) == 3 * 2 * N_ROWS
    assert list(df.columns) == list(ingest.NORMALIZED_COLUMNS)
    assert set(df["model_label"]) == {"B", "T", "V@P0"}
    assert set(df.loc[df["model_label"] == "B", "valid_protocol_ids"]) == {"P0_native,P1_backbone"}
    assert set(df.loc[df["model_label"] == "V@P0", "valid_protocol_ids"]) == {"P0_native"}
    assert (df["row_identity_status"] == ingest.ROW_IDENTITY_VERIFIED).all()
    t = df[df["model_label"] == "T"].iloc[0]
    assert t["runtime_revision"] == world.cfg.role("T")["archived_model_revision"]
    assert t["model_revision"] == world.cfg.role("T")["source_revision"]
    # content key: shared definition, identical across models although sample ids differ
    r = df.iloc[0]
    expected = hashlib.sha256(json.dumps([r.benchmark_id, int(r.source_ordinal), r.source_index, r.source_row_sha256,
                                          r.source_record_sha256], separators=(",", ":"),
                                         ensure_ascii=False).encode()).hexdigest()
    assert r["content_key"] == expected
    per_model = df.groupby("model_label")["content_key"].apply(frozenset)
    assert len(set(per_model)) == 1
    assert df.groupby("content_key")["sample_id"].nunique().eq(3).all()
    # score kinds come from DESIGN.md section 5 (verified against the data), with null-preserving scores
    kinds = df.drop_duplicates("benchmark_id").set_index("benchmark_id")["score_kind"].to_dict()
    assert kinds == {BENCH_ROW: "binary_row", BENCH_AGG: "aggregate_only"}
    agg_rows = df[df["benchmark_id"] == BENCH_AGG]
    assert agg_rows["archived_row_score"].isna().all() and not agg_rows["score_available"].any()
    # a literal "None" judge output is content
    judged = df[(df["source_ordinal"] == 1)]
    assert (judged["judge_output"] == "None").all() and judged["used_judge"].all()
    aggs = pq.read_table(paths["official_aggregates"]).to_pandas()
    assert len(aggs) == 6 and (aggs["official_value"] == aggs["results_file_value"]).all()
    miss = pd.read_csv(paths["missingness"])
    assert (miss["missing_rows"] == 0).all() and (miss["status"] == "complete").all()
    index = json.loads(paths["index"].read_text())
    assert index["predictions"]["rows"] == len(df)
    report = json.loads(paths["report"].read_text())
    assert len(report["normalized_generation_settings"]) == 1 and report["pending_design_amendments"] == []


def test_score_kind_map_matches_design_and_metric_registry():
    from trace_merge import metrics

    kinds = {"binary_row": "binary_row", "grouped_row": "grouped_row", "aggregate_only_rederived": "aggregate_only"}
    assert {b: kinds[s.kind] for b, s in metrics.SPECS.items()} == ingest.BENCHMARK_SCORE_KINDS
    assert sum(k == "binary_row" for k in ingest.BENCHMARK_SCORE_KINDS.values()) == 16


def test_score_values_must_fit_the_design_kind(tmp_path):
    def half(rows):
        rows = [dict(r) for r in rows]
        body = next(r for r in rows if r["score_scope"] == "row")
        body.update(score_value=0.5, score_value_json=EXP.canonical_json(0.5))
        return rows

    w = World(tmp_path).build(_fault("scores", "B", BENCH_ROW, half))
    with pytest.raises(ingest.IngestError, match=r"row scores outside \{0, 1\}"):
        w.ingest()


def test_excluded_row_keeps_its_value_but_is_not_available(tmp_path):
    def exclude(rows):
        rows = [dict(r) for r in rows]
        rows[0]["excluded"] = True
        return rows

    w = World(tmp_path).build(_fault("scores", "B", BENCH_ROW, exclude))
    paths = w.ingest()
    df = pq.read_table(paths["predictions"]).to_pandas()
    r = df[(df.model_label == "B") & (df.benchmark_id == BENCH_ROW) & (df.source_ordinal == 0)].iloc[0]
    assert (not r.score_available) and r.official_excluded and r.archived_row_score == 1.0
    miss = pd.read_csv(paths["missingness"])
    row = miss[(miss.model_label == "B") & (miss.benchmark_id == BENCH_ROW)].iloc[0]
    assert row["excluded_rows"] == 1 and row["row_score_unavailable_rows"] == 1


def test_load_aligned_is_selective(world):
    world.ingest()
    df = ingest.load_aligned(world.cfg, columns=["model_label", "benchmark_id", "content_key"], labels=["T"],
                             benchmarks=[BENCH_ROW])
    assert set(df["model_label"]) == {"T"} and len(df) == N_ROWS and list(df.columns) == [
        "model_label", "benchmark_id", "content_key"]


def _fault(kind, role, bench, fn):
    def mutate(k, r, b, rows):
        return fn(rows) if (k, r, b) == (kind, role, bench) else rows
    return mutate


def test_duplicate_content_key_fails(tmp_path):
    w = World(tmp_path).build(_fault("responses", "T", BENCH_ROW, lambda rows: rows + [dict(rows[0])]))
    with pytest.raises(ingest.IngestError, match="share a content identity"):
        w.ingest(verify_files=True)


def _drop_last_v_row(kind, role, bench, rows):
    if role == "V" and bench == BENCH_ROW and kind in ("responses", "extractions"):
        return rows[:-1]
    if role == "V" and bench == BENCH_ROW and kind == "scores":
        body = [r for r in rows if r["score_scope"] == "row"][:-1]
        agg = dict([r for r in rows if r["score_scope"] == "aggregate"][0], evaluated_rows=N_ROWS - 1)
        return body + [agg]
    return rows


def test_key_set_mismatch_across_slices_fails(tmp_path):
    w = World(tmp_path).build(_drop_last_v_row)
    with pytest.raises(ingest.IngestError, match="covers 3/4 prepared rows"):
        w.ingest()
    miss = pd.read_csv(w.cfg.work("audit", "missingness.csv"))
    row = miss[(miss["model_label"] == "V@P0") & (miss["benchmark_id"] == BENCH_ROW)].iloc[0]
    assert row["missing_rows"] == 1 and row["status"] == "incomplete"
    assert not w.cfg.work("aligned", "predictions.parquet").exists()


def test_cross_slice_key_set_check_without_full_coverage(tmp_path):
    w = World(tmp_path).build(_drop_last_v_row)
    with pytest.raises(ingest.IngestError, match="content key sets differ across model x seed slices"):
        w.ingest(require_full_coverage=False)


def test_dataset_manifest_mismatch_fails(tmp_path):
    def mutate(kind, role, bench, rows):
        if kind == "manifest" and bench == BENCH_ROW:
            rows = [dict(r) for r in rows]
            other = h("another-image")
            rows[2]["media"] = [{"type": "image", "sha256": other}]
            rows[2]["media_set_sha256"] = MC.media_set_sha256(rows[2]["media"])
            rows[2]["source_record_sha256"] = MC.source_record_sha256(rows[2]["source_row_hash"],
                                                                      rows[2]["media_set_sha256"])
        return rows

    w = World(tmp_path).build(mutate)
    with pytest.raises(ingest.IngestError, match="dataset-manifest"):
        w.ingest()


def test_extraction_join_conflict_fails(tmp_path):
    def bad_ext(rows):
        rows = [dict(r) for r in rows]
        rows[0]["prompt_sha256"] = h("different prompt")
        return rows

    w = World(tmp_path).build(_fault("extractions", "B", BENCH_ROW, bad_ext))
    with pytest.raises(ingest.IngestError, match="disagree on prompt_sha256"):
        w.ingest()


def test_response_digest_tamper_fails(tmp_path):
    def tamper(rows):
        rows = [dict(r) for r in rows]
        rows[1]["response_text"] += " (edited)"
        return rows

    w = World(tmp_path).build(_fault("responses", "B", BENCH_AGG, tamper))
    with pytest.raises(ingest.IngestError, match="response_sha256 does not recompute"):
        w.ingest()


def test_aggregate_mismatch_with_results_file_fails(tmp_path):
    w = World(tmp_path).build(_fault("aggregate", "T", BENCH_AGG, lambda v: v + 1.0))
    with pytest.raises(ingest.IngestError, match="results file"):
        w.ingest()


def test_part_digest_mismatch_fails(world):
    part = next((world.run_root / "data" / "scores").rglob("*.parquet"))
    data = part.read_bytes()
    part.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(ingest.IngestError, match="digest/size mismatch"):
        world.ingest()


def test_manifest_sha_must_match_config(world):
    m = world.run_root / "metadata/manifest.json"
    m.write_text(m.read_text() + " ")
    with pytest.raises(ingest.IngestError, match="manifest sha256"):
        world.ingest()


# --------------------------------------------------------------------------------------------- inventory tests
def test_inventory_verifies_archive(world):
    rep = inventory.verify_archive(world.cfg, benchmarks=world.benchmarks, receipt=world.receipt, log=lambda m: None)
    assert rep["status"] == "verified", rep["problems"]
    assert rep["result_file"] == "metadata/results/benchmark_scores.json"


def test_inventory_flags_extra_and_resized_files(world):
    (world.run_root / "data" / "stray.parquet").write_bytes(b"x")
    meta = world.run_root / "metadata/models" / f"{world.cfg.role('B')['archived_model_id']}.json"
    meta.write_text(meta.read_text() + "\n")
    rep = inventory.verify_archive(world.cfg, benchmarks=world.benchmarks, receipt=world.receipt, log=lambda m: None)
    assert rep["status"] == "failed"
    text = " ".join(rep["problems"])
    assert "not listed in the manifest" in text and "size mismatch" in text


def test_inventory_flags_missing_artifact(world):
    rep = inventory.verify_archive(world.cfg, benchmarks=world.benchmarks + ["blink"], receipt=world.receipt,
                                   log=lambda m: None)
    assert any("missing from the manifest" in p for p in rep["problems"])


def test_snapshot_inventory_uses_immediate_blob_name(tmp_path):
    blobs = tmp_path / "blobs"
    store = tmp_path / "store"
    blobs.mkdir()
    store.mkdir()
    content = b"tokenizer"
    digest = hashlib.sha256(content).hexdigest()
    (store / ("f" * 64)).write_bytes(content)          # deduplicated store under another name
    (blobs / digest).symlink_to(store / ("f" * 64))
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "tokenizer.json").symlink_to(blobs / digest)
    inv = inventory.snapshot_inventory(snap)
    assert inv["files"][0]["blob"] == digest and not inv["problems"]
    assert inventory.content_revision({"a": digest}).startswith("sha256set:")


# --------------------------------------------------------------------------------------------- labels and protocols
def test_protocol_is_derived_from_the_label():
    assert ingest.protocol_for_label("B") == "P0_native" and ingest.protocol_for_label("V@P0") == "P0_native"
    assert ingest.protocol_for_label("M050@P1") == "P1_backbone"
    assert ingest.protocol_for_label("B@P2") == "P2_vero"
    for bad in ("V", "M050", "M050@P0", "B@P0", "X@P1", "V@P3"):
        with pytest.raises(ingest.IngestError):
            ingest.protocol_for_label(bad)


# --------------------------------------------------------------------------------------------- local-run spool
def prepare_native(world: World, role: str) -> tuple[str, str]:
    """The protocol clone's prepared native model directory marker (as prepare_trace_eval_models writes it)."""

    slug = ingest.NATIVE_PROTOCOL_SLUGS[role]
    rev = world.cfg.role(role)["source_revision"]
    write_json(world.clone / "rlvr/evaluation/.work/models" / slug / ingest.MODEL_MARKER, {
        "slug": slug, "resolved_commit": rev, "immutable_revision": rev,
        "file_sha256": {"config.json": h(f"{role}-config"), "model.safetensors": h(f"{role}-weights")}})
    return slug, rev


def register_view(world: World, label: str) -> tuple[str, str]:
    """A registered study view (views.py manifest + manifests/views.json + the protocol's content-bound marker)."""

    cfg = world.cfg
    slug = VIEW_SLUGS[label]
    role = ingest.role_of_label(label)
    protocol = ingest.protocol_for_label(label)
    parent = cfg.raw["protocols"][protocol]["view"]
    files = {n: h(f"{parent}-{n}") for n in cfg.raw["protocols"][protocol]["files"]}
    if role in ROLES:
        wrev = cfg.role(role)["source_revision"]
        weights = {"model.safetensors": h(f"{role}-weights")}
        weight_digest = weights["model.safetensors"]
    else:
        wrev = "sha256set:" + h(f"{role}-merge")
        weights = {"model.safetensors": "model.safetensors"}
        weight_digest = h(f"{role}-merged-weights")
        merges_path = world.work / "manifests" / "merges.json"
        merges = json.loads(merges_path.read_text()) if merges_path.is_file() else {}
        merges[role] = {"content_revision": wrev, "recipe_sha256": h(role)}
        write_json(merges_path, merges)
    manifest = {"schema": "trace-merge-view-v1", "label": label, "slug": slug, "protocol_id": protocol,
                "weight_source": role, "weight_source_revision": wrev, "view_parent": parent,
                "view_parent_revision": cfg.role(parent)["source_revision"], "weights": weights,
                "view_files_sha256": files}
    vdir = world.work / "views" / slug
    vsha, _ = write_json(vdir / "view_manifest.json", manifest)
    marker_files = {**files, "view_manifest.json": vsha, "model.safetensors": weight_digest}
    rev = ingest.content_revision(marker_files)
    write_json(vdir / ingest.MODEL_MARKER, {"file_sha256": marker_files, "immutable_revision": rev, "slug": slug,
                                            "model_origin": "local_registration"})
    views_path = world.work / "manifests" / "views.json"
    views = json.loads(views_path.read_text()) if views_path.is_file() else {}
    views[label] = manifest
    write_json(views_path, views)
    return slug, rev


def emit_local_run(world: World, run_dir: Path, *, label: str = "B", rows_subset: dict[str, list[int]] | None = None,
                   skip: tuple[str, str] | None = None, local_dataset_sha: str = h("clone-snapshot"),
                   renumber_stage_ordinals: bool = False, renumber_offset: int = 0, sampling: dict | None = None,
                   provenance: dict | None = None, prompt_tokens: int | None = None, slug: str | None = None,
                   revision: str | None = None, identity_rows: dict[str, list[dict]] | None = None,
                   emit_benchmarks: list[str] | None = None, plan: bool = True, plan_rows: dict | None = None,
                   frozen: bool = True, subset_name: str = "development_screen", summary: bool = True,
                   summary_override: dict | None = None, seed: int = SEED) -> Path:
    """A protocol run directory: spool, plan, (subset) row list, campaign summary, served-model registration."""

    cfg = world.cfg
    protocol = ingest.protocol_for_label(label)
    role = ingest.role_of_label(label)
    if protocol == "P0_native":
        served_slug, served_rev = prepare_native(world, role)
        model = f"{cfg.role(role)['repo_id']}@{served_rev}"
    else:
        served_slug, served_rev = register_view(world, label)
        model = f"trace-merge-local/{served_slug}@{served_rev.split(':')[-1]}"
    slug, revision = slug or served_slug, revision or served_rev
    reference_role = {"P0_native": role, "P1_backbone": "B", "P2_vero": "V"}[protocol]
    tokens = prompt_tokens if prompt_tokens is not None else prompt_tokens_of(reference_role)
    spool = run_dir / "hf_archive"
    common = dict(run_id=f"trace_eval_v1_temp06_seed{seed}", model=model, model_revision=revision, seed=seed,
                  dataset_split="default", dataset_revision=f"trace-final25-datasets-v2:{local_dataset_sha}",
                  model_slug=slug)
    prov = {"trace_git_commit": cfg.raw["protocol"]["code_revision"],
            "vlmevalkit_git_commit": cfg.raw["protocol"]["vlmevalkit_revision"], "contract_version": "c",
            "campaign_config_hash": "d" * 64, "final25_code_hash": "e" * 64,
            "archive_hook_version": "trace-final25-archive-hooks-v1", **(provenance or {})}
    samp = dict(sampling or SAMPLING, seed=seed)
    selected: dict[str, list[dict]] = {}
    for bench in world.benchmarks:
        base_rows = (identity_rows or {}).get(bench) or world.manifest_override.get(bench) or manifest_rows(bench)
        keep = set(rows_subset.get(bench, range(N_ROWS))) if rows_subset else set(range(N_ROWS))
        selected[bench] = [m for m in base_rows if m["ordinal"] in keep]
    for bench in emit_benchmarks or world.benchmarks:
        mrows = selected[bench]
        gen, ext, sco = [], [], []
        for position, m in enumerate(mrows):
            base = {"source_index": m["index"], "source_ordinal": m["ordinal"], "source_row_hash": m["source_row_hash"],
                    "request_hash": h(f"req{bench}{m['ordinal']}"), "question": "q", "ground_truth": "1"}
            # subset runs: the official route renumbers extraction/score ordinals by subset position
            stage_base = dict(base, source_ordinal=position + renumber_offset) if renumber_stage_ordinals else base
            text = f"local {bench} {m['ordinal']}. The answer is 1"
            gen.append({**base, "source_row": {"answer": "1"}, "prompt": prompt_text(bench, m["ordinal"]),
                        "model_response": text, "sampling": samp, "finish_reason": "stop",
                        "usage": {"prompt_tokens": tokens, "completion_tokens": 40},
                        "metadata": {"dataset_snapshot_sha256": local_dataset_sha,
                                     "image_hash": [x["sha256"] for x in m["media"]],
                                     "media_hash": m["media_set_sha256"],
                                     "source_record_sha256": m["source_record_sha256"]}})
            ext.append({**stage_base, "model_response": text, "judge_prompt": "", "judge_response": "",
                        "normalized_extraction": {"method": "regex", "status": "resolved", "value": "1"},
                        "retries": {"count": 0}})
            if ingest.BENCHMARK_SCORE_KINDS[bench] != "aggregate_only":
                sco.append({**stage_base, "prediction": "1", "score": m["ordinal"] % 2 == 1, "scorer": "exact",
                            "excluded": False, "metadata": {"evaluation_status": "ok", "judge_prompt": "secret q",
                                                            "artifact_path": "/abs/path"}})
            else:
                sco.append({**stage_base, "prediction": "1", "score": None, "scorer": "agg", "excluded": False,
                            "metadata": {"score_contract": "aggregate_only", "aggregate_score": 50.0}})
        aggs = {"generation": {"generation": MEDIA_GEN}, "extraction": {"rows": len(mrows)},
                "score": {"score": 50.0, "rows": len(mrows)}}
        for stage, records in (("generation", gen), ("extraction", ext), ("score", sco)):
            if skip == (bench, stage):
                continue
            HF.emit_slice_ready(spool, stage=stage, benchmark=bench, dataset_alias=bench, records=records,
                                provenance=prov, aggregate=aggs[stage], **common)
    sha = None
    if plan:
        rows_file = None
        if rows_subset:
            payload = {"schema": "trace-cert-ext-eval-subset-v1", "mode": subset_name, "benchmarks": {
                b: {"selected": len(selected[b]), "rows": [
                    {"ordinal": m["ordinal"], "index": m["index"], "source_row_hash": m["source_row_hash"],
                     "row_id": m["index"]} for m in selected[b]]} for b in world.benchmarks}}
            sha = ingest.row_list_sha256(payload, world.cfg.benchmark_keys())
            payload["row_list_sha256"] = sha
            rows_file = run_dir / f"rows.{subset_name}.json"
            write_json(rows_file, payload)
            if frozen:
                split = world.work / "splits" / "split_manifest.json"
                data = json.loads(split.read_text()) if split.is_file() else {"row_lists": {}}
                data["row_lists"][subset_name] = {"file": rows_file.name, "row_list_sha256": sha}
                write_json(split, data)
        expected = plan_rows or {b: len(selected[b]) for b in world.benchmarks}
        write_json(run_dir / f"plan.seed{seed}.json", {
            "schema": "trace-merge-eval-plan-v1", "slug": slug, "seed": seed, "subset": subset_name if rows_subset
            else "full", "rows_file": None if rows_file is None else str(rows_file), "row_list_sha256": sha,
            "expected_rows": expected})
    if summary:
        entries = [dict({"key": b, "score": 50.0, "rows": len(selected[b])}, **(summary_override or {}).get(b, {}))
                   for b in world.benchmarks]
        write_json(run_dir / f"summary.seed{seed}.json", {"seed": seed, "row_list_sha256": sha, "complete": True,
                                                         "benchmarks": entries})
    return run_dir


def local(world, run, label="B", protocol=None, source=None, **kw):
    source = source or ("existing_local_rerun" if "@" not in label else "local_run")
    return ingest.ingest_local_run(world.cfg, run, label, protocol, SEED, source, log=lambda m: None, **kw)


def test_local_run_ingestion_matches_archive_identity(ingested):
    world = ingested
    run = emit_local_run(world, world.tmp / "runs" / "b_full")
    df = local(world, run, "B", "P0_native")
    assert len(df) == 2 * N_ROWS and list(df.columns) == list(ingest.NORMALIZED_COLUMNS)
    assert set(df["source"]) == {"existing_local_rerun"} and set(df["model_label"]) == {"B"}
    ident = ingest.compare_identity_with_archive(world.cfg, df, "B", SEED)
    assert ident["identical_identities"]
    # same prompt, media, settings and public model id => identical public requests
    assert ident["total"]["request_sha256_equal"] == 2 * N_ROWS
    report = df.attrs["ingest_report"]
    assert report["prompt_conformance"]["conforms"] and report["prompt_conformance"]["total"]["rows"] == 2 * N_ROWS
    assert report["spool_provenance"]["trace_git_commit"] == world.cfg.raw["protocol"]["code_revision"]
    assert report["served_model"]["served_view"]["content_revision"].startswith("sha256set:")
    row = df[df["benchmark_id"] == BENCH_ROW]
    assert row["score_available"].all() and set(row["archived_row_score"]) == {0.0, 1.0}
    out = json.loads(row["row_scorer_output_json"].iloc[0])
    assert "question" not in out and "ground_truth" not in out
    assert out["metadata"] == {"evaluation_status": "ok"}
    assert set(df.loc[df["benchmark_id"] == BENCH_AGG, "score_kind"]) == {"aggregate_only"}
    paths = ingest.write_local_run(world.cfg, df, "B_seed42_existing_local_rerun")
    back = ingest.load_aligned(world.cfg, include_archive=False, local_runs=["B_seed42_existing_local_rerun"])
    assert len(back) == len(df)
    aggs = pq.read_table(paths["aggregates"]).to_pandas()
    assert set(aggs["benchmark_id"]) == set(world.benchmarks)
    assert set(aggs["campaign_summary_status"]) == {"equal"}


def test_local_runs_need_the_archive_first(world):
    run = emit_local_run(world, world.tmp / "runs" / "early")
    with pytest.raises(ingest.IngestError, match="archive must be ingested before local runs"):
        local(world, run, "B")


def test_local_view_subset_run_is_accepted(ingested):
    world = ingested
    run = emit_local_run(world, world.tmp / "runs" / "subset", label="V@P1",
                         rows_subset={BENCH_ROW: [0, 2], BENCH_AGG: [1]})
    df = local(world, run, "V@P1", None)
    assert len(df) == 3
    assert set(df["protocol_id"]) == {"P1_backbone"} and set(df["valid_protocol_ids"]) == {"P1_backbone"}
    assert set(df["model_revision"]) == {world.cfg.role("V")["source_revision"]}
    assert set(df["runtime_revision"]) == {df.attrs["ingest_report"]["served_model"]["runtime_revision"]}
    statuses = {r["benchmark_id"]: r["status"] for r in df.attrs["missingness"]}
    assert statuses == {BENCH_ROW: "subset_complete", BENCH_AGG: "subset_complete"}
    assert {r["benchmark_id"]: r["expected_rows"] for r in df.attrs["missingness"]} == {BENCH_ROW: 2, BENCH_AGG: 1}
    conf = df.attrs["ingest_report"]["prompt_conformance"]
    assert conf["conforms"] and conf["reference"]["model_label"] == "B"


def test_merge_view_revision_is_the_merge_content_revision(ingested):
    world = ingested
    run = emit_local_run(world, world.tmp / "runs" / "m050", label="M050@P1")
    df = local(world, run, "M050@P1")
    merges = json.loads((world.work / "manifests" / "merges.json").read_text())
    assert set(df["model_revision"]) == {merges["M050"]["content_revision"]}


def test_label_and_protocol_must_agree(ingested):
    world = ingested
    run = emit_local_run(world, world.tmp / "runs" / "p2", label="B@P2")
    with pytest.raises(ingest.IngestError, match="belongs to P2_vero, not P1_backbone"):
        local(world, run, "B@P2", "P1_backbone")
    with pytest.raises(ingest.IngestError, match="belongs to P1_backbone"):
        local(world, run, "M050@P1", "P2_vero")


def test_wrong_model_spool_is_rejected(ingested):
    world = ingested
    run = emit_local_run(world, world.tmp / "runs" / "bspool", label="B")       # native B spool
    register_view(world, "V@P1")
    with pytest.raises(ingest.IngestError, match="spool serves 'qwen2.5-vl-7b-instruct', the registered view is"):
        local(world, run, "V@P1")


def test_view_revision_must_be_the_registered_one(ingested):
    world = ingested
    run = emit_local_run(world, world.tmp / "runs" / "stale", label="V@P1", revision="sha256set:" + h("stale"))
    with pytest.raises(ingest.IngestError, match="is not the registered view"):
        local(world, run, "V@P1")


def test_native_rerun_must_serve_the_pinned_commit(ingested):
    world = ingested
    run = emit_local_run(world, world.tmp / "runs" / "oldb", label="B", revision="0" * 40)
    with pytest.raises(ingest.IngestError, match="expected the prepared native"):
        local(world, run, "B")


def test_undeclared_label_is_rejected(ingested):
    world = ingested
    run = emit_local_run(world, world.tmp / "runs" / "role")
    with pytest.raises(ingest.IngestError, match="study role|not declared"):
        local(world, run, "X@P1")


def test_local_run_with_other_decoding_is_rejected(ingested):
    world = ingested
    run = emit_local_run(world, world.tmp / "runs" / "hot", label="M050@P1",
                         sampling=dict(SAMPLING, max_tokens=8192, temperature=0.7))
    with pytest.raises(ingest.IngestError, match="generation settings differ.*max_output_tokens=8192"):
        local(world, run, "M050@P1")


def test_local_run_provenance_must_be_the_pinned_protocol(ingested):
    world = ingested
    run = emit_local_run(world, world.tmp / "runs" / "prov", label="M050@P1",
                         provenance={"trace_git_commit": "a" * 40, "vlmevalkit_git_commit": "b" * 40})
    with pytest.raises(ingest.IngestError, match="trace_git_commit .* != protocol.code_revision"):
        local(world, run, "M050@P1")


def test_missing_planned_benchmark_fails(ingested):
    world = ingested
    run = emit_local_run(world, world.tmp / "runs" / "partial", label="M050@P1", emit_benchmarks=[BENCH_ROW])
    with pytest.raises(ingest.IngestError, match=r"missing \['mathvista'\]"):
        local(world, run, "M050@P1")


def test_run_without_plan_fails(ingested):
    world = ingested
    run = emit_local_run(world, world.tmp / "runs" / "noplan", label="M050@P1", plan=False)
    with pytest.raises(ingest.IngestError, match="run plan not found"):
        local(world, run, "M050@P1")


def test_lost_row_of_a_full_run_fails(ingested):
    world = ingested
    run = emit_local_run(world, world.tmp / "runs" / "lost", label="M050@P1",
                         rows_subset={BENCH_ROW: [0, 1, 2], BENCH_AGG: [0, 1, 2, 3]}, plan=False)
    write_json(run / "plan.seed42.json", {"seed": SEED, "subset": "full", "rows_file": None, "row_list_sha256": None,
                                          "expected_rows": {BENCH_ROW: N_ROWS, BENCH_AGG: N_ROWS}})
    with pytest.raises(ingest.IngestError, match="covers 3/4 planned rows"):
        local(world, run, "M050@P1")


def test_subset_rows_must_be_the_planned_rows(ingested):
    world = ingested
    planned = emit_local_run(world, world.tmp / "runs" / "planned", label="M050@P1",
                             rows_subset={BENCH_ROW: [0, 2], BENCH_AGG: [0]})
    produced = emit_local_run(world, world.tmp / "runs" / "wrongrows", label="M050@P1",
                              rows_subset={BENCH_ROW: [1, 3], BENCH_AGG: [0]}, plan=False, summary=False)
    for name in ("plan.seed42.json", "rows.development_screen.json", "summary.seed42.json"):
        (produced / name).write_bytes((planned / name).read_bytes())
    plan = json.loads((produced / "plan.seed42.json").read_text())
    plan["rows_file"] = str(produced / "rows.development_screen.json")
    write_json(produced / "plan.seed42.json", plan)
    with pytest.raises(ingest.IngestError, match="covers 0/2 planned rows"):
        local(world, produced, "M050@P1")


def test_subset_row_list_must_be_frozen_for_study_runs(ingested):
    world = ingested
    run = emit_local_run(world, world.tmp / "runs" / "unfrozen", label="M050@P1",
                         rows_subset={BENCH_ROW: [0, 2], BENCH_AGG: [0]}, frozen=False, subset_name="q25")
    with pytest.raises(ingest.IngestError, match="not a frozen row list of this study"):
        local(world, run, "M050@P1")


def test_aggregate_must_match_the_campaign_summary(ingested):
    world = ingested
    run = emit_local_run(world, world.tmp / "runs" / "summ", label="M050@P1",
                         summary_override={BENCH_AGG: {"score": 40.0}})
    with pytest.raises(ingest.IngestError, match="spool aggregate 50.0 != summary score 40.0"):
        local(world, run, "M050@P1")


def test_rescaled_summary_value_is_kept_next_to_the_spool_value(ingested):
    world = ingested
    run = emit_local_run(world, world.tmp / "runs" / "rescaled", label="M050@P1",
                         summary_override={BENCH_AGG: {"score": 40.0, "score_protocol": 50.0}})
    df = local(world, run, "M050@P1")
    agg = {a["benchmark_id"]: a for a in df.attrs["official_aggregates"]}[BENCH_AGG]
    assert (agg["official_value"], agg["campaign_summary_value"], agg["campaign_summary_status"]) == (
        50.0, 40.0, "summary_rescaled")
    assert any("campaign summary reports 40.0" in w for w in df.attrs["ingest_report"]["warnings"])


def test_nonconforming_rendered_inputs_are_rejected(ingested):
    world = ingested
    run = emit_local_run(world, world.tmp / "runs" / "vtemplate", label="V@P1", prompt_tokens=1090)
    with pytest.raises(ingest.IngestError, match="do not conform to archived B"):
        local(world, run, "V@P1")


def test_p2_run_conforms_to_archived_vero(ingested):
    world = ingested
    run = emit_local_run(world, world.tmp / "runs" / "bp2", label="B@P2")
    df = local(world, run, "B@P2")
    conf = df.attrs["ingest_report"]["prompt_conformance"]
    assert conf["conforms"] and conf["reference"]["model_label"] == "V@P0"
    assert set(df["protocol_id"]) == {"P2_vero"}


def test_tampered_spool_payload_fails(ingested):
    world = ingested
    run = emit_local_run(world, world.tmp / "runs" / "tampered")
    payload = next((run / "hf_archive" / "payloads").glob("*.jsonl"))
    payload.chmod(0o600)
    payload.write_text(payload.read_text().replace("local", "LOCAL", 1))
    with pytest.raises(ingest.IngestError, match="failed verification"):
        local(world, run, "B")


def test_incomplete_local_run_fails(ingested):
    world = ingested
    run = emit_local_run(world, world.tmp / "runs" / "incomplete", skip=(BENCH_ROW, "score"))
    with pytest.raises(ingest.IngestError, match="incomplete"):
        local(world, run, "B")


def test_load_aligned_refuses_to_mix_runs_of_one_label(ingested):
    world = ingested
    run = emit_local_run(world, world.tmp / "runs" / "b_full")
    ingest.write_local_run(world.cfg, local(world, run, "B"), "B_seed42_existing_local_rerun")
    with pytest.raises(ingest.IngestError, match="select 'source' and 'run_id'"):
        ingest.load_aligned(world.cfg, columns=["model_label", "seed", "content_key"], labels=["B"], seeds=[SEED],
                            local_runs=["B_seed42_existing_local_rerun"])
    both = ingest.load_aligned(world.cfg, columns=["source", "run_id", "model_label", "seed", "content_key"],
                               labels=["B"], seeds=[SEED], local_runs=["B_seed42_existing_local_rerun"])
    assert not both.duplicated(["source", "run_id", "model_label", "seed", "content_key"]).any()
    assert len(both) == 2 * 2 * N_ROWS


def test_reingesting_the_archive_keeps_local_missingness_rows(ingested):
    world = ingested
    run = emit_local_run(world, world.tmp / "runs" / "b_full2")
    ingest.write_local_run(world.cfg, local(world, run, "B"), "B_seed42_existing_local_rerun")
    world.ingest()
    after = pd.read_csv(world.cfg.work("audit", "missingness.csv"))
    assert {"archive", "B_seed42_existing_local_rerun"} <= set(after["run_name"])
    assert (after["run_name"] == "archive").sum() == 3 * 2


def countqa_fixture(world: World) -> None:
    """CountQA-like licensed TSV: the manifest hashes the answer as a string, the 'producer' as pandas' int."""

    lmu = world.clone / "rlvr/evaluation/.work/LMUData"
    lmu.mkdir(parents=True, exist_ok=True)
    tsv = lmu / "CountQA.tsv"
    pd.DataFrame([{"index": f"{i}_0", "question": f"how many {i}", "answer": i + 3, "image_path": f"img/{i}.jpg",
                   "objects": "['a']", "categories": "['c']", "is_focused": False, "full_config": "{}"}
                  for i in range(N_ROWS)]).to_csv(tsv, sep="\t", index=False)
    public_rows, producer_rows = [], []
    for i, raw in enumerate(pd.read_csv(tsv, sep="\t").to_dict("records")):
        base = {k: v for k, v in raw.items() if k != "image_path"}
        base["index"] = int(str(raw["index"]))
        media = [{"type": "image", "sha256": h(f"countqa-image-{i}")}]
        mset = MC.media_set_sha256(media)
        for rows, variant in ((public_rows, dict(base, answer=str(base["answer"]))), (producer_rows, base)):
            rh = MC.source_row_sha256(variant)
            rows.append({"index": str(base["index"]), "ordinal": i, "source_row_hash": rh,
                         "source_record_sha256": MC.source_record_sha256(rh, mset), "media_set_sha256": mset,
                         "media": media})
    world.benchmarks = [BENCH_ROW, "countqa"]
    world.manifest_override["countqa"] = public_rows
    world.archive_override["countqa"] = producer_rows
    world.metadata_files["countqa"] = [{"path": str(tsv), "sha256": sha256_file(tsv), "size_bytes": tsv.stat().st_size}]


def test_producer_alias_is_off_by_default(tmp_path):
    w = World(tmp_path)
    countqa_fixture(w)
    w.build()
    with pytest.raises(ingest.IngestError, match="producer row-hash alias .* is not enabled"):
        w.ingest()


def test_producer_row_hash_alias_is_verified_and_canonicalized(tmp_path):
    w = World(tmp_path)
    countqa_fixture(w)
    w.build()
    paths = w.ingest(producer_aliases=["countqa"])
    df = pq.read_table(paths["predictions"]).to_pandas()
    cq = df[df["benchmark_id"] == "countqa"]
    public = {r["ordinal"]: r for r in w.manifest_override["countqa"]}
    producer = {r["ordinal"]: r for r in w.archive_override["countqa"]}
    assert len(cq) == 3 * N_ROWS
    assert cq["row_identity_status"].str.startswith(ingest.ROW_IDENTITY_ALIASED).all()
    for r in cq.itertuples():
        assert r.source_row_sha256 == public[r.source_ordinal]["source_row_hash"]
        assert r.archived_source_row_sha256 == producer[r.source_ordinal]["source_row_hash"]
        assert r.source_record_sha256 == public[r.source_ordinal]["source_record_sha256"]
    dmi = ingest.load_dataset_manifest_index(w.cfg.dataset_manifest, w.benchmarks, aliases=["countqa"])
    assert set(cq["content_key"]) == dmi.expected_content_keys("countqa")
    other = df[df["benchmark_id"] == BENCH_ROW]
    assert other["archived_source_row_sha256"].isna().all()
    report = json.loads(paths["report"].read_text())
    assert report["dataset_manifest"]["producer_row_hash_aliases"]["countqa"]["rows_with_distinct_hash"] == N_ROWS
    assert report["producer_row_hash_aliases_enabled"]["countqa"]["enabled_by"] == "argument"
    # DESIGN.md amendment A1 names the patch, so nothing is pending
    assert report["pending_design_amendments"] == []
    # a local CountQA run (public code) joins the aliased archive rows on the content key
    run = emit_local_run(w, w.tmp / "runs" / "b_countqa")
    loc = local(w, run, "B")
    assert set(loc.loc[loc.benchmark_id == "countqa", "content_key"]) == set(cq["content_key"])
    assert (loc["row_identity_status"] == ingest.ROW_IDENTITY_VERIFIED).all()


def test_producer_alias_never_applies_to_local_runs(tmp_path):
    w = World(tmp_path)
    countqa_fixture(w)
    w.build()
    w.ingest(producer_aliases=["countqa"])
    run = emit_local_run(w, w.tmp / "runs" / "producer_hashes",
                         identity_rows={"countqa": w.archive_override["countqa"]})
    with pytest.raises(ingest.IngestError, match="no dataset-manifest row matches"):
        local(w, run, "B")


def test_producer_alias_requires_reproducible_manifest_hash(tmp_path):
    w = World(tmp_path)
    countqa_fixture(w)
    w.manifest_override["countqa"] = [dict(r, source_row_hash=h(f"other{r['ordinal']}"),
                                           source_record_sha256=MC.source_record_sha256(h(f"other{r['ordinal']}"),
                                                                                        r["media_set_sha256"]))
                                      for r in w.manifest_override["countqa"]]
    w.build()
    with pytest.raises(ingest.IngestError, match="cannot reproduce the manifest row hash"):
        w.ingest(producer_aliases=["countqa"])


def test_subset_stage_ordinals_are_reattached(ingested):
    world = ingested
    run = emit_local_run(world, world.tmp / "runs" / "renumbered", label="CT050@P1",
                         rows_subset={BENCH_ROW: [1, 3], BENCH_AGG: [0, 2]}, renumber_stage_ordinals=True)
    df = local(world, run, "CT050@P1")
    assert sorted(df.loc[df["benchmark_id"] == BENCH_ROW, "source_ordinal"]) == [1, 3]
    assert df.loc[df["benchmark_id"] == BENCH_ROW, "score_available"].all()
    repairs = {s["benchmark_id"]: s["subset_stage_ordinals_reattached"] for s in df.attrs["ingest_report"]["slices"]}
    assert repairs[BENCH_ROW] == {"extractions": 2, "scores": 2}
    assert repairs[BENCH_AGG] == {"extractions": 1, "scores": 1}   # ordinal 0 is already at position 0


def test_subset_stage_ordinals_must_be_positional(ingested):
    world = ingested
    run = emit_local_run(world, world.tmp / "runs" / "offset", label="CT050@P1",
                         rows_subset={BENCH_ROW: [1, 3], BENCH_AGG: [0, 2]}, renumber_stage_ordinals=True,
                         renumber_offset=5)
    with pytest.raises(ingest.IngestError, match="neither the generation ordinal .* nor its subset position"):
        local(world, run, "CT050@P1")


# --------------------------------------------------------------------------------------------- real data (opt-in)
@pytest.mark.skipif(not os.environ.get("TRACE_MERGE_WORK_ROOT"), reason="real work root not configured")
def test_real_aligned_tables():
    cfg = load_config()
    index_path = cfg.work("aligned", "index.json")
    if not index_path.is_file():
        pytest.skip("archive not ingested in this work root")
    index = json.loads(index_path.read_text())
    assert index["predictions"]["rows"] == 9 * 32805
    assert pq.ParquetFile(cfg.work("aligned", "predictions.parquet")).metadata.num_rows == 9 * 32805
    archive = json.loads(cfg.work("audit", "archive_ingest.json").read_text())
    assert archive["status"] == "verified" and len(archive["normalized_generation_settings"]) == 1
    for name, entry in index.get("local_runs", {}).items():
        report = json.loads(cfg.work("aligned", entry["report"]).read_text())
        assert report["status"] == "verified" and report["prompt_conformance"]["conforms"], name
        if entry.get("source") == "existing_local_rerun":
            assert entry["rows"] == 32805, name
