"""Component-effect analysis (design §6) on synthetic data: decomposition identity, Holm, centered p-values, the
paired cluster bootstrap with a known effect, identical draws for official and format-normalized-v2 scores, and the
ingest shim's path routing and label patch.

The bootstrap tests use the real reviewed engine (``trace_merge.statistics`` through F0's ``paired_contrasts``) and
real binary-row metric adapters, which need no licensed metadata or pinned scorers."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trace_followup import analyze as az  # noqa: E402
from trace_merge import ingest as ig  # noqa: E402
from trace_merge import metrics as mt  # noqa: E402
from trace_merge import statistics as st  # noqa: E402
from trace_merge.config import StudyConfig  # noqa: E402

CELLS = az.F1_CELLS
LABELS = [f"{c}@P1" for c in CELLS]
BENCHES = ["blink", "mmstar"]
CATEGORIES = {"Perception": ["blink"], "General": ["mmstar"]}
N_ROWS = 200


# --------------------------------------------------------------------------------------------- synthetic matrix
def _accuracy_pattern() -> dict[str, float]:
    """Fraction of rows (a prefix) each cell solves: gF_T = 0, gL_T = +20, I_T = +20; V's effects all 0."""

    return {"B": 0.5, "TF": 0.5, "TL": 0.7, "T": 0.9, "V": 0.5, "VF": 0.5, "VL": 0.5}


def synthetic_rows(seeds=(42, 43), normalized_bump: dict[tuple[str, str], float] | None = None) -> tuple[
        pd.DataFrame, pd.DataFrame]:
    """Rows (7 labels x seeds x 2 benchmarks x 200 contents) and bootstrap units (pairs of contents).

    Cell c solves content i of a benchmark iff i / N < its accuracy. ``normalized_bump[(cell, bench)] = f`` makes the
    normalized score 1 on the first ``f`` share of that cell's wrong rows (normalized >= official, as in F0)."""

    acc = _accuracy_pattern()
    recs = []
    for bench in BENCHES:
        for i in range(N_ROWS):
            for cell in CELLS:
                for s in seeds:
                    off = float(i / N_ROWS < acc[cell])
                    norm = off
                    bump = (normalized_bump or {}).get((cell, bench))
                    if bump and not off and (i / N_ROWS - acc[cell]) < bump * (1 - acc[cell]):
                        norm = 1.0
                    recs.append({"model_label": f"{cell}@P1", "seed": s, "benchmark_id": bench,
                                 "content_key": f"{bench}:{i}", "source_ordinal": i, "source_index": str(i),
                                 "official_row_score": off, "normalized_row_score": norm})
    rows = pd.DataFrame(recs)
    keys = rows[["content_key"]].drop_duplicates().reset_index(drop=True)
    keys["component_id"] = [f"u{k // 2:04d}" for k in range(len(keys))]   # two contents per unit
    return rows, keys


def _metrics() -> dict[str, mt.BenchmarkMetric]:
    return {b: mt.get_metric(b) for b in BENCHES}


@pytest.fixture(scope="module")
def boot():
    rows, ga = synthetic_rows()
    return az.scale_bootstrap(rows, _metrics(), ga, labels=LABELS, seeds=[42, 43], categories=CATEGORIES,
                              n_replicates=1000, seed=314159, confidence=0.95)


# ---------------------------------------------------------------------------------------------------- identity
def test_decomposition_weights_satisfy_the_identities_algebraically():
    eff = az.decomposition_contrasts("P1")
    for p in az.PARENTS:
        full = az.combine((1.0, eff[f"full_{p}"]), (-1.0, eff[f"gF_{p}"]), (-1.0, eff[f"gL_{p}"]),
                          (-1.0, eff[f"I_{p}"]))
        assert full == {}
        assert az.combine((1.0, eff[f"revF_{p}"]), (-1.0, eff[f"gF_{p}"]), (-1.0, eff[f"I_{p}"])) == {}
        assert az.combine((1.0, eff[f"revL_{p}"]), (-1.0, eff[f"gL_{p}"]), (-1.0, eff[f"I_{p}"])) == {}
    assert eff["I_T"] == {"T@P1": 1.0, "TF@P1": -1.0, "TL@P1": -1.0, "B@P1": 1.0}
    assert eff["revF_V"] == {"V@P1": 1.0, "VL@P1": -1.0}      # S(P) - S(F_B, L_P)
    assert eff["revL_V"] == {"V@P1": 1.0, "VF@P1": -1.0}      # S(P) - S(F_P, L_B)
    cross = az.cross_parent_contrasts(eff)
    assert cross["gF_T_minus_gF_V"] == {"TF@P1": 1.0, "VF@P1": -1.0}   # B cancels


def test_identity_holds_on_points_and_every_replicate(boot):
    for res in boot.values():
        chk = az.identity_check(res, "P1")
        for p in az.PARENTS:
            assert chk[p]["holds"], chk
            assert chk[p]["replicates"] == 1000


def test_identity_check_reports_missing_cells():
    rows, ga = synthetic_rows(seeds=(42,))
    keep = [lab for lab in LABELS if lab != "VL@P1"]
    res = az.scale_bootstrap(rows[rows["model_label"].isin(keep)], _metrics(), ga, labels=keep, seeds=[42],
                             categories=CATEGORIES, n_replicates=50, seed=1, confidence=0.95)
    chk = az.identity_check(res[az.OFFICIAL], "P1")
    assert chk["T"]["holds"] and "not computable" in chk["V"]["status"]
    table = az.contrast_rows(res[az.OFFICIAL], az.decomposition_contrasts("P1"), scale=az.OFFICIAL)
    assert table.loc[table["effect"] == "gL_V", "status"].iloc[0].startswith("not computable")


# ---------------------------------------------------------------------------------------------- known effects
def test_bootstrap_recovers_the_known_effects(boot):
    eff = az.decomposition_contrasts("P1")
    table = az.contrast_rows(boot[az.OFFICIAL], eff, scale=az.OFFICIAL)
    ov = table[(table["level"] == "overall") & (table["seed_scope"] == "mean_over_seeds")].set_index("effect")
    expected = {"gF_T": 0.0, "gL_T": 20.0, "I_T": 20.0, "full_T": 40.0, "revF_T": 20.0, "revL_T": 40.0,
                "gF_V": 0.0, "gL_V": 0.0, "I_V": 0.0, "full_V": 0.0}
    for name, value in expected.items():
        assert ov.loc[name, "estimate"] == pytest.approx(value, abs=1e-9)
    # real effects: interval excludes 0, smallest attainable centered p
    for name in ("gL_T", "I_T", "full_T"):
        assert ov.loc[name, "ci_low"] > 0
        assert ov.loc[name, "p_value"] == pytest.approx(1 / 1001)
    # exact zeros (identical rows): degenerate interval, p = 1
    for name in ("gF_T", "gF_V", "gL_V", "I_V"):
        assert ov.loc[name, "ci_low"] == ov.loc[name, "ci_high"] == 0.0
        assert ov.loc[name, "p_value"] == 1.0
    assert (ov["n_valid_replicates"] == 1000).all()
    # per-seed estimates are reported and equal here (same pattern on both seeds)
    seeds = table[(table["effect"] == "gL_T") & (table["level"] == "overall") & (table["seed_scope"] == "single_seed")]
    assert sorted(seeds["seed"].tolist()) == [42, 43] and np.allclose(seeds["estimate"], 20.0)
    # benchmark and category scopes exist, and the overall macro is named by its benchmark count
    assert set(table["level"]) == {"overall", "category", "benchmark"}
    assert ov.loc["gL_T", "scope"] == st.macro_name(2)


def test_performance_table_matches_the_pattern(boot):
    perf = az.performance_table(boot)
    ov = perf[(perf["level"] == "overall") & (perf["seed_scope"] == "mean_over_seeds")].set_index("cell")
    for cell, acc in _accuracy_pattern().items():
        assert ov.loc[cell, "value_official"] == pytest.approx(100 * acc)
        assert ov.loc[cell, "value_normalized"] == pytest.approx(100 * acc)
    assert (perf["normalized_minus_official"].abs() < 1e-12).all()


def test_effect_with_noise_has_a_proper_interval():
    """A noisy +5 pp effect: the percentile interval covers the estimate and is not degenerate."""

    rows, ga = synthetic_rows(seeds=(42,))
    rng = np.random.default_rng(0)
    sel = (rows["model_label"] == "VL@P1")
    rows.loc[sel, "official_row_score"] = (rng.random(int(sel.sum())) < 0.55).astype(float)
    rows.loc[sel, "normalized_row_score"] = rows.loc[sel, "official_row_score"]
    res = az.scale_bootstrap(rows, _metrics(), ga, labels=LABELS, seeds=[42], categories=CATEGORIES,
                             n_replicates=2000, seed=7, confidence=0.95)
    t = az.contrast_rows(res[az.OFFICIAL], {"gL_V": {"VL@P1": 1.0, "B@P1": -1.0}}, scale=az.OFFICIAL)
    r = t[(t["level"] == "overall")].iloc[0]
    assert r["ci_low"] < r["estimate"] < r["ci_high"] and r["ci_high"] - r["ci_low"] > 1.0


# -------------------------------------------------------------------------------------------- identical draws
def test_official_and_normalized_use_identical_draws():
    """Normalized differs from official only for TF on mmstar; every other replicate value must be bit-identical,
    which holds only if both scales were resampled with the same unit multiplicities."""

    rows, ga = synthetic_rows(normalized_bump={("TF", "mmstar"): 0.5})
    res = az.scale_bootstrap(rows, _metrics(), ga, labels=LABELS, seeds=[42, 43], categories=CATEGORIES,
                             n_replicates=500, seed=314159, confidence=0.95)
    o, n = res[az.OFFICIAL], res[az.NORMALIZED]
    assert np.array_equal(o.component_ids, n.component_ids)
    ro, rn = o.replicate_benchmark_scores, n.replicate_benchmark_scores
    tf, mm = o.labels.index("TF@P1"), o.benchmarks.index("mmstar")
    mask = np.ones(ro.shape[1:], dtype=bool)
    mask[tf, mm] = False
    assert np.array_equal(ro[:, mask], rn[:, mask])
    assert (rn[:, tf, mm] > ro[:, tf, mm]).all()
    # the official result does not depend on the normalized column at all (same draws as a run without the bump)
    rows0, _ = synthetic_rows()
    ref = az.scale_bootstrap(rows0, _metrics(), ga, labels=LABELS, seeds=[42, 43], categories=CATEGORIES,
                             n_replicates=500, seed=314159, confidence=0.95)[az.OFFICIAL]
    assert np.array_equal(ref.replicate_benchmark_scores, ro)
    rec = az.draws_record(res)
    assert rec["identical_draws_official_and_normalized"] and rec["n_units"] == len(ga) // 2


# -------------------------------------------------------------------------------------------------------- Holm
def test_holm_known_values_and_monotonicity():
    p = {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005, "e": 0.5, "f": 0.2}
    adj = st.holm_adjust(p)
    # step-down: sorted d(.005) a(.01) c(.03) b(.04) f(.2) e(.5) with multipliers 6..1, running max
    assert adj["d"] == pytest.approx(0.03)
    assert adj["a"] == pytest.approx(0.05)
    assert adj["c"] == pytest.approx(0.12)
    assert adj["b"] == pytest.approx(0.12)     # 3 * 0.04 = 0.12, monotone with c
    assert adj["f"] == pytest.approx(0.4)
    assert adj["e"] == pytest.approx(0.5)


def _family_table(pvals: dict[str, float], scale: str) -> pd.DataFrame:
    rows = [{"effect": e, "scale": scale, "level": "overall", "scope": "m", "seed_scope": "mean_over_seeds",
             "p_value": p, "status": "ok"} for e, p in pvals.items()]
    rows += [{"effect": "gF_T", "scale": scale, "level": "benchmark", "scope": "blink",
              "seed_scope": "mean_over_seeds", "p_value": 0.001, "status": "ok"},
             {"effect": "full_T", "scale": scale, "level": "overall", "scope": "m", "seed_scope": "mean_over_seeds",
              "p_value": 0.0001, "status": "ok"}]
    return pd.DataFrame(rows)


def test_holm_is_applied_to_the_overall_family_only_and_per_scale():
    p_off = dict(zip(az.PRIMARY_FAMILY, [0.01, 0.04, 0.03, 0.005, 0.5, 0.2]))
    p_norm = dict(zip(az.PRIMARY_FAMILY, [0.2, 0.2, 0.2, 0.2, 0.2, 0.001]))
    t = az.apply_holm(pd.concat([_family_table(p_off, az.OFFICIAL), _family_table(p_norm, az.NORMALIZED)]),
                      az.PRIMARY_FAMILY)
    fam = t[t["family_member"]]
    assert len(fam) == 12                                   # 6 per scale; benchmark rows and full_T excluded
    off = fam[fam["scale"] == az.OFFICIAL].set_index("effect")["p_holm"].to_dict()
    assert off == pytest.approx(st.holm_adjust(p_off))
    nrm = fam[fam["scale"] == az.NORMALIZED].set_index("effect")["p_holm"].to_dict()
    assert nrm == pytest.approx(st.holm_adjust(p_norm))     # its own family, not pooled with official
    assert t.loc[~t["family_member"], "p_holm"].isna().all()


def test_holm_refuses_a_smaller_family():
    p = dict(zip(az.PRIMARY_FAMILY[:5], [0.01] * 5))
    t = az.apply_holm(_family_table(p, az.OFFICIAL), az.PRIMARY_FAMILY)
    assert not t["family_member"].any() and t["p_holm"].isna().all()


def test_effects_table_applies_holm_on_the_bootstrap(boot):
    eff = az.decomposition_contrasts("P1")
    table = az.effects_table(boot, eff, az.effect_roles(eff, az.PRIMARY_FAMILY), az.PRIMARY_FAMILY)
    fam = table[table["family_member"] & (table["scale"] == az.OFFICIAL)].set_index("effect")
    assert sorted(fam.index) == sorted(az.PRIMARY_FAMILY)
    assert fam["p_holm"].to_dict() == pytest.approx(st.holm_adjust(fam["p_value"].to_dict()))
    assert set(table.loc[table["scale"] == az.NORMALIZED, "analysis"]) == {"sensitivity_analysis"}
    assert table.loc[table["effect"] == "full_T", "role"].iloc[0] == "decomposition_reported"


# --------------------------------------------------------------------------------------------- centered p-value
def test_centered_p_value():
    r = np.random.default_rng(1).normal(0.0, 1.0, 9999)
    assert az.centered_p_value(r, 0.0) == 1.0
    # estimate 1.96 sd from 0: about 5% of |d* - d| exceed 1.96
    shifted = r + 1.96
    assert az.centered_p_value(shifted, 1.96) == pytest.approx(0.05, abs=0.01)
    assert az.centered_p_value(np.full(99, 3.0), 3.0) == pytest.approx(1 / 100)
    assert np.isnan(az.centered_p_value(np.array([np.nan]), 1.0))


# ------------------------------------------------------------------------------------------------ ingest shim
def test_shim_routes_reviewed_reads_and_follow_up_writes(tmp_path):
    new, old = tmp_path / "new", tmp_path / "old"
    cfg = az.FollowupIngestConfig(path=tmp_path / "c.yaml", raw={}, sha256="x", work_root=new, reviewed_root=old)
    assert cfg.work("aligned", "predictions.parquet") == old / "aligned" / "predictions.parquet"
    assert cfg.work("audit", "archive_ingest.json") == old / "audit" / "archive_ingest.json"
    assert cfg.work("manifests", "metric_registry.json") == old / "manifests" / "metric_registry.json"
    assert cfg.work("aligned", "index.json") == new / "aligned" / "index.json"
    assert cfg.work("aligned", "local_runs") == new / "aligned" / "local_runs"
    assert cfg.work("audit", "missingness.csv") == new / "audit" / "missingness.csv"
    assert cfg.work("manifests", "views.json") == new / "manifests" / "views.json"
    assert cfg.work("manifests", "merges.json") == new / "manifests" / "hybrids.json"
    assert cfg.work("splits", "split_manifest.json") == new / "splits" / "split_manifest.json"
    assert isinstance(cfg, StudyConfig)


def test_follow_up_roles_are_patched_only_inside_the_block():
    fake = types.SimpleNamespace(cells={c: c.lower() for c in CELLS}, raw={"extension_cells": {"TR0": "tr0"}})
    before = ig.MODEL_ROLES
    with pytest.raises(ig.IngestError):
        ig.protocol_for_label("TF@P1")
    with az.followup_roles(fake):
        assert ig.protocol_for_label("TF@P1") == "P1_backbone"
        assert ig.protocol_for_label("VL@P2") == "P2_vero"
        assert ig.protocol_for_label("TR0@P1") == "P1_backbone"
    assert ig.MODEL_ROLES == before
    with pytest.raises(ig.IngestError):
        ig.protocol_for_label("TF@P1")


# ---------------------------------------------------------------------------------------- checks and rates
class _Pooled:
    """Pooled 0/1 mean whose campaign value equals finalize (any non-WeMath adapter)."""

    def row_contributions(self, rows, scores):
        return np.column_stack([scores.to_numpy(), np.ones(len(scores))])

    def finalize(self, totals):
        return 100.0 * totals[0] / totals[1]

    campaign_value = finalize


def test_campaign_reproduction_tolerance():
    scored = pd.DataFrame({"benchmark_id": ["x"] * 4 + ["y"] * 2, "official_row_score": [1, 0, 1, 1, 0, 1.0]})
    aggs = pd.DataFrame({"benchmark_id": ["x", "y"], "official_value": [75.0, 50.0 + 1e-3],
                         "campaign_summary_value": [75.0, 50.0], "campaign_summary_status": ["equal", "equal"]})
    rep = az.campaign_reproduction(scored, {"x": _Pooled(), "y": _Pooled()}, aggs)
    assert rep["x"]["reproduced"] and rep["x"]["abs_error_pp"] == 0.0
    assert not rep["y"]["reproduced"] and rep["y"]["abs_error_pp"] == pytest.approx(1e-3)


def test_format_rates_and_technical_thresholds():
    rows = pd.DataFrame({"model_label": "TF@P1", "seed": 42, "benchmark_id": ["a", "a", "b", "b"],
                         "response_empty": [True, False, False, False],
                         "finish_reason": ["stop", "length", "length", None],
                         "completion_tokens": [0, 4096, 4096, 10], "fn_status": ["out_of_scope", "repaired",
                                                                                  "official_read", "unresolved"]})
    r = az.format_rates(rows, 4096)
    ov = r[r["level"] == "overall"].iloc[0]
    assert ov["rows"] == 4 and ov["empty_pct"] == 25.0 and ov["truncated_pct"] == 50.0
    assert ov["at_token_limit_rows"] == 2 and ov["finish_reason_null_rows"] == 1 and ov["fn_repaired_rows"] == 1
    assert set(r.loc[r["level"] == "benchmark", "scope"]) == {"a", "b"}
    assert az.EMPTY_STOP_PCT == 50.0 and az.TRUNCATION_STOP_PCT == 80.0


# ------------------------------------------------------------------------------- overlay-rescored provenance (A1/A3)
PINNED, OV1, OV2 = "a" * 64, "b" * 64, "c" * 64


def _registry(tmp_path):
    import json
    from trace_followup.common import load
    cfg = load(work_root=tmp_path)
    d = tmp_path / "dispatch" / "protocol_overlay"
    d.mkdir(parents=True)
    (d / "approved_code_hashes.json").write_text(json.dumps({
        "pinned_shared_clone": PINNED, "overlay": {OV1: "fix-4/5"},
        "rescored_runs": {"TF@P2.development_screen.s42": "A1"}}))
    return cfg


def _spool(slug, gen_hash, score_hash, seed=42, subset="development_screen"):
    def desc(stage, h, camp):
        return ({"identity": {"model_slug": slug, "seed": seed, "stage": stage,
                              "run_id": f"trace_eval_v1_{subset}_temp06_seed{seed}"},
                 "provenance": {"final25_code_hash": h, "campaign_config_hash": camp, "trace_git_commit": "x"}}, [])
    gen_camp = "s" if gen_hash == score_hash else "g"   # one campaign config per code state
    return {("chartqapro", "generation"): desc("generation", gen_hash, gen_camp),
            ("chartqapro", "extraction"): desc("extraction", score_hash, "s"),
            ("chartqapro", "score"): desc("score", score_hash, "s")}


def _strict(cfg, spool):  # stand-in for the reviewed check: every provenance field unique
    for k in ("final25_code_hash", "campaign_config_hash"):
        if len({(d.get("provenance") or {}).get(k) for d, _r in spool.values()}) != 1:
            raise ig.IngestError(f"{k} not unique")
    return {"final25_code_hash": next(iter(spool.values()))[0]["provenance"]["final25_code_hash"]}


def test_overlay_split_accepted_only_for_listed_runs(tmp_path):
    check = az.overlay_tolerant_provenance(_registry(tmp_path), _strict)
    out = check(None, _spool("tf-p2", PINNED, OV1))
    assert out["overlay_rescore"]["amendment"] == "A1"
    assert out["overlay_rescore"]["generation"]["final25_code_hash"] == PINNED
    with pytest.raises(az.FollowupAnalysisError):          # an overlay-scored run that is not listed
        check(None, _spool("vl-p2", PINNED, OV1))
    with pytest.raises(ig.IngestError):                    # unapproved overlay hash: the reviewed check decides
        check(None, _spool("tf-p2", PINNED, OV2))
    with pytest.raises(ig.IngestError):                    # generation not from the pinned clone
        check(None, _spool("tf-p2", OV2, OV1))
    assert "overlay_rescore" not in check(None, _spool("tf-p2", PINNED, PINNED))   # ordinary run: unchanged path
