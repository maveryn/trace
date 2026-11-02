"""Protocol comparison, environment drift and length/format/judge audit on small synthetic frames."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from trace_merge import diagnostics, ingest


def h(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


SETTINGS = {"image_jpeg_quality": None, "max_image_pixels": 12845056, "max_image_side": None,
            "max_output_tokens": 4096, "media_contract_version": "trace_eval_media_v1", "media_transport": "file-url",
            "min_image_pixels": 3136, "presence_penalty": 0.0, "repetition_penalty": 1.0,
            "source_media_contract_sha256": h("media"), "temperature": 0.6, "top_k": -1, "top_p": 1.0}


def settings_json(model_id: str, seed: int, **override) -> str:
    s = dict(SETTINGS, model_id=model_id, seed=seed, **override)
    return json.dumps(s, sort_keys=True, separators=(",", ":"))


def protocol_frame(temperature_for_v: float = 0.6) -> pd.DataFrame:
    rows = []
    for label, mid, ptoks in (("B", "base", 600), ("T", "trace", 600), ("V@P0", "vero", 1090)):
        for i in range(3):
            override = {"temperature": temperature_for_v} if label == "V@P0" else {}
            rows.append({
                "source": "archive", "run_id": "run", "protocol_id": "P0_native", "model_label": label, "seed": 42,
                "benchmark_id": "erqa", "content_key": h(f"key{i}"), "dataset_revision": "rev",
                "dataset_sha256": h("ds"), "runtime_revision": mid, "prompt_sha256": h(f"prompt{i}"),
                "request_sha256": h(f"req{label}{i}"), "prompt_contract_sha256": h(f"pc{label}{i}"),
                "generation_settings_json": settings_json(mid, 42, **override), "scorer_id": "erqa-primary-v1",
                "prompt_tokens": ptoks + i, "completion_tokens": 100 * (i + 1),
                "finish_reason": "length" if (label == "B" and i == 2) else "stop",
                "ordered_media_sha256": [h(f"img{i}")], "source_row_sha256": h(f"row{i}"),
                "source_record_sha256": h(f"rec{i}"), "media_set_sha256": h(f"ms{i}"), "source_ordinal": i,
                "source_index": str(i)})
    return pd.DataFrame(rows)


def test_normalized_settings_drop_only_identity_fields():
    a = diagnostics.normalized_settings_json(settings_json("base", 42))
    b = diagnostics.normalized_settings_json(settings_json("vero", 44))
    c = diagnostics.normalized_settings_json(settings_json("base", 42, top_p=0.9))
    assert a == b and a != c
    assert "model_id" not in json.loads(a) and "seed" not in json.loads(a) and "temperature" in json.loads(a)


def test_compare_protocols_matching_settings():
    rep = diagnostics.compare_protocols(protocol_frame())
    assert rep["normalized_settings"]["distinct"] == 1
    bt = rep["cross_model_pairs"]["B|T"]
    assert bt["prompt_sha256_equal_rate"] == 1.0
    assert bt["neutral_request_sha256_equal_rate"] == 1.0
    assert bt["neutral_prompt_contract_sha256_equal_rate"] == 1.0
    assert bt["prompt_contract_sha256_equal_rate"] == 0.0        # binds model_id/seed by construction
    assert bt["prompt_tokens_equal_rate"] == 1.0
    bv = rep["cross_model_pairs"]["B|V@P0"]
    assert bv["prompt_tokens_equal_rate"] == 0.0 and bv["prompt_tokens_median_paired_diff_b_minus_a"] == 490.0
    assert rep["scorers"]["all_identical_across_labels"]
    assert rep["finish_reasons"]["archive/B"] == {"stop": 2, "length": 1}
    assert len(rep["slices"]) == 3


def test_compare_protocols_detects_setting_difference():
    rep = diagnostics.compare_protocols(protocol_frame(temperature_for_v=0.7))
    assert rep["normalized_settings"]["distinct"] == 2
    assert rep["cross_model_pairs"]["B|V@P0"]["neutral_request_sha256_equal_rate"] == 0.0
    assert rep["cross_model_pairs"]["B|T"]["neutral_request_sha256_equal_rate"] == 1.0


def test_pair_summary_pools_counts_and_the_paired_median():
    df = protocol_frame()
    # a second benchmark where V is 10 tokens longer still: per-group medians 490 and 500 (+i offsets cancel)
    extra = protocol_frame().assign(benchmark_id="blink")
    extra.loc[extra["model_label"] == "V@P0", "prompt_tokens"] += 10
    rep = diagnostics.compare_protocols(pd.concat([df, extra], ignore_index=True))
    bv = rep["cross_model_pairs"]["B|V@P0"]
    assert bv["shared_rows"] == 6 and bv["prompt_tokens_known_both"] == 6
    assert bv["prompt_tokens_median_paired_diff_b_minus_a"] == 495.0     # pooled over 3x490 and 3x500
    assert diagnostics.counter_median({1: 1, 5: 2, 9: 1}) == 5.0 and diagnostics.counter_median({}) is None
    assert diagnostics.counter_median({1: 1, 2: 1}) == 1.5


def test_cross_label_agreement_refuses_two_runs_of_one_label():
    df = protocol_frame()
    b = df[df.model_label == "B"].assign(source="existing_local_rerun", run_id="rerun")
    with pytest.raises(ingest.IngestError, match="one run per label"):
        diagnostics.cross_label_agreement(pd.concat([df, b], ignore_index=True))


def test_local_runs_are_keyed_by_source_and_run_id():
    df = protocol_frame()
    rerun = df[df.model_label == "B"].assign(source="existing_local_rerun", run_id="rerun42")
    rep = diagnostics.compare_protocols(pd.concat([df, rerun], ignore_index=True))
    assert set(rep["finish_reasons"]) == {"archive/B", "archive/T", "archive/V@P0", "existing_local_rerun/rerun42/B"}
    assert set(rep["cross_model_pairs"]) == {"B|T", "B|V@P0", "T|V@P0"}


def conformance_frame(label_tokens: int, settings_override: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref = protocol_frame()
    ref = ref[ref.model_label == "B"]
    loc = ref.assign(model_label="V@P1", source="local_run", protocol_id="P1_backbone",
                     prompt_tokens=ref["prompt_tokens"] - 600 + label_tokens,
                     generation_settings_json=settings_json("v-p1", 42, **(settings_override or {})))
    return loc, ref


def test_prompt_conformance_accepts_backbone_rendering():
    loc, ref = conformance_frame(600)
    res = ingest.prompt_conformance(loc, ref)
    assert res["conforms"] and res["total"]["prompt_tokens_equal"] == 3
    assert res["total"]["neutral_request_sha256_equal"] == 3        # model_id/seed differ, nothing else


def test_prompt_conformance_flags_another_template_or_decoding():
    loc, ref = conformance_frame(1090)                               # V's own template rendered under "P1"
    res = ingest.prompt_conformance(loc, ref)
    assert not res["conforms"] and res["nonconforming_benchmarks"] == ["erqa"]
    assert res["total"]["prompt_sha256_equal"] == 3 and res["total"]["prompt_tokens_equal"] == 0
    loc, ref = conformance_frame(600, {"max_output_tokens": 8192})
    res = ingest.prompt_conformance(loc, ref)
    assert not res["conforms"] and res["total"]["neutral_request_sha256_equal"] == 0
    res = ingest.prompt_conformance(loc.iloc[:2], ref.iloc[:1].assign(generation_settings_json=loc[
        "generation_settings_json"].iloc[0]))
    assert res["total"]["reference_rows_matched"] == 1 and not res["conforms"]


def drift_frame(scores, responses, tokens, bench="erqa", kind="binary_row", finish=None):
    n = len(scores)
    return pd.DataFrame({
        "benchmark_id": bench, "content_key": [h(f"{bench}{i}") for i in range(n)],
        "archived_row_score": [np.nan if s is None else s for s in scores],
        "score_available": [s is not None for s in scores], "score_kind": kind,
        "response_sha256": [h(r) for r in responses], "completion_tokens": [np.nan if t is None else t for t in tokens], "prompt_tokens": [600] * n,
        "finish_reason": finish or ["stop"] * n, "request_sha256": [h(f"req{i}") for i in range(n)],
        "prompt_sha256": [h(f"p{i}") for i in range(n)], "extraction_value_json": ['"A"'] * n,
        "extraction_status": "resolved"})


def test_compare_runs_counts_agreement_and_deltas():
    ref = pd.concat([drift_frame([1.0, 1.0, 0.0, 0.0], ["a", "b", "c", "d"], [10, 20, 30, 40],
                                 finish=["stop", "stop", "stop", "length"]),
                     drift_frame([None, None], ["x", "y"], [5, None], bench="mathvista", kind="aggregate_only")])
    oth = pd.concat([drift_frame([1.0, 0.0, 1.0, 0.0], ["a", "B", "C", "d"], [12, 18, 30, 50]),
                     drift_frame([None, None], ["x", "Y"], [7, 9], bench="mathvista", kind="aggregate_only")])
    ref_aggs = pd.DataFrame({"benchmark_id": ["erqa", "mathvista"], "official_value": [50.0, 70.0]})
    oth_aggs = pd.DataFrame({"benchmark_id": ["erqa", "mathvista"], "official_value": [50.0, 72.5]})
    table, summary = diagnostics.compare_runs(ref, oth, ref_aggs, oth_aggs)
    erqa = table.set_index("benchmark_id").loc["erqa"]
    assert erqa["rows_scored_both"] == 4 and erqa["row_score_agreement_rate"] == 0.5
    assert (erqa["both_correct"], erqa["both_wrong"], erqa["reference_only_correct"],
            erqa["other_only_correct"]) == (1, 1, 1, 1)
    assert erqa["response_sha256_equal_rate"] == 0.5
    assert erqa["completion_tokens_mean_paired_shift"] == pytest.approx((2 - 2 + 0 + 10) / 4)
    assert erqa["length_stops_reference"] == 1 and erqa["length_stops_other"] == 0
    mv = table.set_index("benchmark_id").loc["mathvista"]
    assert mv["rows_scored_both"] == 0 and mv["official_delta_other_minus_reference"] == pytest.approx(2.5)
    assert mv["completion_tokens_mean_paired_shift"] == pytest.approx(2.0)   # only rows known in both runs
    assert summary["official_macro_delta"] == pytest.approx(1.25)
    assert summary["row_score_agreement_rate_pooled"] == 0.5
    assert summary["response_text_equal_rate"] == pytest.approx(3 / 6)


def test_compare_runs_pools_token_shift_over_rows_known_in_both():
    ref = pd.concat([drift_frame([1.0] * 4, ["a"] * 4, [10, 10, 10, 10]),
                     drift_frame([None] * 4, ["x"] * 4, [10, None, None, None], bench="mathvista",
                                 kind="aggregate_only")])
    oth = pd.concat([drift_frame([1.0] * 4, ["a"] * 4, [10, 10, 10, 10]),
                     drift_frame([None] * 4, ["x"] * 4, [50, 7, 7, 7], bench="mathvista", kind="aggregate_only")])
    _, summary = diagnostics.compare_runs(ref, oth)
    assert summary["completion_tokens_known_both"] == 5
    assert summary["completion_tokens_mean_paired_shift"] == pytest.approx(40 / 5)


def test_compare_runs_benchmark_without_known_tokens_does_not_dilute_the_shift():
    ref = pd.concat([drift_frame([1.0] * 4, ["a"] * 4, [10] * 4),
                     drift_frame([None] * 4, ["x"] * 4, [None] * 4, bench="mathvista", kind="aggregate_only")])
    oth = pd.concat([drift_frame([1.0] * 4, ["a"] * 4, [30] * 4),
                     drift_frame([None] * 4, ["x"] * 4, [None] * 4, bench="mathvista", kind="aggregate_only")])
    table, summary = diagnostics.compare_runs(ref, oth)
    assert summary["completion_tokens_mean_paired_shift"] == pytest.approx(20.0)
    assert table.set_index("benchmark_id").loc["mathvista", "completion_tokens_known_both"] == 0


def test_compare_runs_prompt_token_rate_ignores_unknown_rows():
    ref = drift_frame([1.0] * 2, ["a", "b"], [1, 1]).assign(prompt_tokens=[600, None])
    oth = drift_frame([1.0] * 2, ["a", "b"], [1, 1])
    table, summary = diagnostics.compare_runs(ref, oth)
    assert summary["prompt_tokens_known_both"] == 1 and summary["prompt_tokens_equal_rate"] == 1.0


def audit_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "source": "archive", "run_id": "run", "protocol_id": "P0_native", "model_label": "V@P0", "seed": 42,
        "benchmark_id": "erqa",
        "response_text": ["<think>x</think><answer>\\boxed{B}</answer>", "The answer is C", "Answer: D", None,
                          "rambling without a marker"],
        "finish_reason": ["stop", "stop", "length", None, "length"],
        "completion_tokens": pd.array([100, 4096, 4096, None, 50], dtype="Int64"),
        "extraction_status": ["resolved", "resolved", "invalid", None, "ambiguous"],
        "extraction_method": ["judge", "benchmark_adapter", "judge", None, "deterministic_parser"],
        "extraction_value_json": ['"B"', '"The answer is C"', '"The model response is incomplete. Extracted: D"',
                                  None, '"rambling without a marker"'],
        "used_judge": pd.array([True, False, False, None, False], dtype="boolean"),
        "archived_row_score": [1.0, 0.0, None, None, 0.0],
        "score_available": [True, True, False, False, True],
        "row_scorer": [None, None, None, None, None]})


def test_length_format_judge_audit_counts_and_null_handling():
    table = diagnostics.length_format_judge_audit(audit_frame(), max_tokens=4096)
    assert len(table) == 1
    r = table.iloc[0]
    assert r["n_rows"] == 5
    assert r["completion_tokens_n_known"] == 4 and r["completion_tokens_max"] == 4096
    assert r["completion_tokens_mean"] == pytest.approx((100 + 4096 + 4096 + 50) / 4)
    # a 4096-token response that stopped normally is not a length stop
    assert r["n_tokens_at_or_above_cap"] == 2 and r["finish_reason_length"] == 2
    assert r["finish_reason_null"] == 1 and r["length_stop_rate"] == pytest.approx(2 / 4)
    assert (r["extraction_resolved"], r["extraction_invalid"], r["extraction_ambiguous"],
            r["extraction_status_null"]) == (2, 1, 1, 1)
    assert r["extraction_status_not_resolved_rate"] == pytest.approx(2 / 4)
    assert "extraction_failure_rate" not in table.columns
    assert (r["extraction_method_judge"], r["extraction_method_benchmark_adapter"],
            r["extraction_method_deterministic_parser"], r["extraction_method_other"]) == (2, 1, 1, 0)
    # passthrough values and the judge's 'incomplete' sentinel are measured deterministically
    assert r["extraction_value_is_response_rate"] == pytest.approx(2 / 4)
    assert r["judge_incomplete_sentinel_rows"] == 1
    # length-stop cross-tab: one truncated row is unscored with the sentinel, the other passed through and scored 0
    assert (r["length_stop_score_available"], r["length_stop_score_zero"], r["length_stop_value_is_response"],
            r["length_stop_judge_incomplete_sentinel"], r["length_stop_marker_any_final"]) == (1, 1, 1, 1, 1)
    assert r["used_judge_n_known"] == 4 and r["used_judge_rate"] == pytest.approx(1 / 4)
    assert r["row_scorer_known"] == 0 and pd.isna(r["row_scorer_judge_rows"])
    assert r["response_text_n_known"] == 4
    assert r["marker_answer_tag_rate"] == pytest.approx(1 / 4)
    assert r["marker_boxed_rate"] == pytest.approx(1 / 4)
    assert r["marker_think_close_rate"] == pytest.approx(1 / 4)
    assert r["marker_answer_is_rate"] == pytest.approx(1 / 4)
    assert r["marker_answer_colon_line_rate"] == pytest.approx(1 / 4)
    assert r["marker_any_final_rate"] == pytest.approx(3 / 4)


def test_length_audit_counts_judge_backed_row_scorers():
    df = audit_frame().assign(source="local_run", row_scorer=["vlmevalkit_defaults_qwen32b_judge", "choice_function",
                                                              None, None, "qwen3_open_correctness"])
    r = diagnostics.length_format_judge_audit(df).iloc[0]
    assert r["row_scorer_known"] == 3 and r["row_scorer_judge_rows"] == 2


def test_length_audit_groups_by_source_run_and_seed():
    a = audit_frame()
    b = audit_frame().assign(source="existing_local_rerun")
    c = audit_frame().assign(seed=43)
    d = audit_frame().assign(source="local_run", run_id="dev")
    e = audit_frame().assign(source="local_run", run_id="confirmation")
    table = diagnostics.length_format_judge_audit(pd.concat([a, b, c, d, e], ignore_index=True))
    assert len(table) == 5 and "n_tokens_at_or_above_cap" not in table.columns
    assert set(table["n_rows"]) == {5}


def test_length_audit_all_null_group_stays_null():
    df = audit_frame().iloc[[3]]
    r = diagnostics.length_format_judge_audit(df).iloc[0]
    assert r["completion_tokens_mean"] is None or np.isnan(r["completion_tokens_mean"])
    assert r["length_stop_rate"] is None or np.isnan(r["length_stop_rate"])
    assert r["marker_any_final_rate"] is None or np.isnan(r["marker_any_final_rate"])


def test_audit_protocols_end_to_end_on_a_synthetic_world(tmp_path):
    """Archive with three seeds + a B rerun + a V@P1 view run: conformance, scorer code, seed-pair drift reference."""

    import test_ingest as ti

    world = ti.World(tmp_path, seeds=(42, 43, 44)).build()
    world.ingest()
    rerun = ti.emit_local_run(world, tmp_path / "runs" / "b_full")
    ingest.write_local_run(world.cfg, ti.local(world, rerun, "B"), "B_seed42_existing_local_rerun")
    view = ti.emit_local_run(world, tmp_path / "runs" / "v_p1", label="V@P1")
    ingest.write_local_run(world.cfg, ti.local(world, view, "V@P1"), "V_at_P1_seed42_local_run")
    rep = diagnostics.audit_protocols(world.cfg, log=lambda m: None)
    assert rep["all_local_runs_conform"]
    conf = rep["prompt_conformance"]["V_at_P1_seed42_local_run"]
    (key, entry), = conf.items()
    assert entry["reference"] == "archive/B/seed=42" and entry["total"]["prompt_tokens_equal"] == 2 * ti.N_ROWS
    assert rep["cross_model_pairs"]["B|T"]["shared_rows"] == 3 * 2 * ti.N_ROWS
    assert rep["scorer_code"]["archive_identical_across_models"]
    assert rep["scorer_code"]["local_code_hashes"] == ["e" * 64]
    assert any(k.startswith("local_run/") and k.endswith("/V@P1") for k in rep["finish_reasons"])
    drift = rep["environment_drift"]["B_seed42_existing_local_rerun"]
    assert set(drift["seed_pairs"]) == {"42-43", "42-44", "43-44"}
    assert drift["length_stops"] == {"archived_by_seed": {"42": 2, "43": 0, "44": 2}, "rerun": 0,
                                     "rerun_outside_archived_range": False}
    shift = drift["ranges"]["completion_tokens_mean_paired_shift"]
    assert shift["seed_pairs_min"] == pytest.approx(1.0) and shift["seed_pairs_max"] == pytest.approx(2.0)
    assert shift["rerun"] < 0 and shift["rerun_vs_seed_pair_range"] == "below"
    table = pd.read_csv(world.cfg.work("tables", "environment_drift.csv"))
    assert set(table["comparison"]) == {"archive B seed 42 vs B_seed42_existing_local_rerun",
                                        "archive B seeds 42-43", "archive B seeds 42-44", "archive B seeds 43-44"}
    path = diagnostics.write_length_format_judge_audit(world.cfg, local_runs=["B_seed42_existing_local_rerun"],
                                                       log=lambda m: None)
    audit = pd.read_csv(path)
    assert len(audit) == 3 * 3 * 2 + 2 and set(audit["scoring_route"]) == {"official_vlmevalkit", "direct_score"}
    assert json.loads(path.with_name("length_format_judge_audit.columns.json").read_text())["columns"]


def test_answer_format_sensitivity_reads_latex_letters_leniently():
    import pandas as pd
    from trace_merge import diagnostics as dg

    rows = pd.DataFrame({"source_ordinal": [0, 1, 2, 3],
                         "response_text": ["... Answer: \\( C \\)", "so the answer is B.", "\\boxed{A}", "no letter here"]})
    meta = pd.DataFrame({"source_ordinal": [0, 1, 2, 3], "answer": ["red", "blue", "5", "x"],
                         "options": ["['5', 'blue', 'red']", "['red', 'blue']", "['5', '6']", "['x']"]})
    scores = pd.Series([0.0, 1.0, 0.0, 0.0])
    out = dg.answer_format_sensitivity(rows, scores, meta, "puzzlevqa")
    assert out["official_accuracy_pct"] == 25.0
    assert out["latex_letter_ending_pct"] == 25.0
    # row 0: C -> 'red' == answer (officially 0) ; row 2: A -> '5' == answer (officially 0); row 3 has no letter
    assert out["official_zero_but_final_letter_correct_pct"] == 50.0
    assert dg.final_option_letter("Answer: \\( D \\).") == "D" and dg.final_option_letter("nothing") is None
