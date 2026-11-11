"""Figures and sanitized result bundles (handoff 5.5, 11.2, 11.3).

Figures are static PNGs, one question each, with denominators and interval meaning in the caption and the evaluation
tier in the title. Entity colors are fixed across every figure (color follows the model, never its rank); base B is a
neutral reference. Exported tables contain aggregates only: no benchmark inputs, no responses, no machine paths.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, REPO_ROOT, StudyConfig, sha256_bytes, write_json

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
COLORS = {  # reference palette slots, assigned per entity in a fixed order
    "B": "#8f8e89",
    "T": "#2a78d6",
    "V@P0": "#eb6834",
    "V@P1": "#6250d6",
    "M050@P1": "#1baf7a",
    "CV050@P1": "#eda100",
    "CT050@P1": "#e87ba4",
}
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6")   # positional fallback, fixed order
DISPLAY = {"B": "Base (Qwen2.5-VL-7B)", "T": "TRACE", "V@P0": "VERO (own template)", "V@P1": "VERO (backbone template)",
           "M050@P1": "M050 = ½VERO + ½TRACE", "CV050@P1": "CV050 = ½VERO + ½base", "CT050@P1": "CT050 = ½TRACE + ½base",
           "B@P2": "Base, VERO interface", "T@P2": "TRACE, VERO interface", "M050@P2": "M050, VERO interface",
           "CV050@P2": "CV050, VERO interface", "CT050@P2": "CT050, VERO interface"}
CAPTURE_EVENTS = (("trace_only_capture", "TRACE-only capture"), ("vero_only_capture", "VERO-only capture"),
                  ("shared_success_damage", "shared-success damage"), ("joint_failure_rescue", "joint-failure rescue"))
# a whole absolute machine path, from its root segment up to the next delimiter
ABS_PATH = re.compile(r"(?<![\w.~-])/(?:scratch|uufs|home|Users|users|tmp|mnt|data|proj|opt|root|var|workspace|content"
                      r"|dev/shm|kaggle)(?=[/\s'\"`),;|\]}>]|$)[^\s'\"`),;|\]}>]*")
TOKEN = re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")


def color_of(label: str) -> str:
    """Entity color: the exact label, else the same weights' P1 label (``M050``/``M050@P2`` -> ``M050@P1``,
    ``B@P2`` -> ``B``); unknown labels get the secondary ink."""

    base = label.split("@")[0]
    for cand in (label, f"{base}@P1", base):
        if cand in COLORS:
            return COLORS[cand]
    return INK_2


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
                         "text.color": INK, "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})
    return plt


def suite_order(cfg: StudyConfig) -> list[tuple[str, str]]:
    """(category, benchmark) in suite order."""

    return [(cat, b) for cat, keys in cfg.categories().items() for b in keys]


def display_names(cfg: StudyConfig) -> dict[str, str]:
    return {b["key"]: b["display"] for b in cfg.suite()["benchmarks"]}


def fig_benchmark_gains(cfg: StudyConfig, diffs: pd.DataFrame, out: Path, comparisons: Sequence[str],
                        title: str, caption: str) -> Path:
    """Dot-and-interval plot: per benchmark, each comparison's paired difference with its 95% interval."""

    plt = _plt()
    order = suite_order(cfg)
    names = display_names(cfg)
    d = diffs[diffs["level"] == "benchmark"].set_index(["comparison", "scope"])
    fig, ax = plt.subplots(figsize=(7.2, 0.27 * len(order) + 1.6))
    offsets = np.linspace(-0.18, 0.18, len(comparisons)) if len(comparisons) > 1 else [0.0]
    ylabels = []
    for i, (cat, b) in enumerate(order):
        ylabels.append(f"{names[b]}")
    # color by the compared model; when comparisons share it (M050 - T, M050 - V@P1, ...), by the reference model
    pairs = [comp.split("-") for comp in comparisons]
    side = 0 if len({p[0] for p in pairs}) == len(pairs) else 1
    colors = [color_of(p[side]) for p in pairs]
    if len(set(colors)) < len(colors):
        colors = [PALETTE[j % len(PALETTE)] for j in range(len(colors))]
    missing = [comp for comp in comparisons if comp not in d.index.get_level_values(0)]
    if missing:
        raise ValueError(f"no benchmark rows for {missing} in the differences table")
    for j, comp in enumerate(comparisons):
        color = colors[j]
        ys, xs, lo, hi = [], [], [], []
        for i, (_, b) in enumerate(order):
            if (comp, b) not in d.index:
                continue
            r = d.loc[(comp, b)]
            ys.append(i + offsets[j]); xs.append(r["estimate"]); lo.append(r["ci_low"]); hi.append(r["ci_high"])
        ax.hlines(ys, lo, hi, color=color, linewidth=2, alpha=0.85)
        ax.plot(xs, ys, "o", color=color, markersize=5, markeredgecolor=SURFACE, markeredgewidth=1,
                label=" − ".join(short_label(x) for x in comp.split("-", 1)))
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(ylabels)
    ax.invert_yaxis()
    cat_starts = {}
    for i, (cat, _) in enumerate(order):
        cat_starts.setdefault(cat, i)
    for cat, i in cat_starts.items():
        if i:
            ax.axhline(i - 0.5, color=GRID, linewidth=1.2)
        ax.text(1.01, i, cat, transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=8, color=INK_2)
    ax.set_xlabel("difference in official score (percentage points)")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.045), frameon=False, ncol=min(2, len(comparisons)))
    fig.text(0.01, 0.005, caption, fontsize=7, color=INK_2, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.06, 0.86, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def fig_parent_overlap(overlap: pd.DataFrame, out: Path, caption: str, category_order: Sequence[str] = (),
                       title: str = "Discovery (Stage B): parent outcome cells") -> Path:
    """Stacked bars: share of rows in each parent-outcome cell, per category in suite order (seed-averaged, pooled
    rows), each labelled with its rows per seed."""

    plt = _plt()
    p = overlap[(overlap["seed_scope"] == "mean_over_seeds") & (overlap["aggregation"] == "pooled_rows")
                & (overlap["level"].isin(["category", "overall"]))].copy()
    p["order"] = (p["level"] == "overall").astype(int)
    rank = {c: i for i, c in enumerate(category_order)}
    p["suite_rank"] = [rank.get(str(s).split(" (partial")[0], len(rank)) for s in p["scope"]]
    p = p.sort_values(["order", "suite_rank", "scope"])
    cells = [("both_correct_pct", "both correct", "#6b6a66"), ("trace_only_pct", "TRACE only", COLORS["T"]),
             ("vero_only_pct", "VERO only", COLORS["V@P0"]), ("both_wrong_pct", "both wrong", "#d8d7d1")]
    fig, ax = plt.subplots(figsize=(7.2, 0.6 * len(p) + 1.4))
    left = np.zeros(len(p))
    for col, lab, color in cells:
        vals = p[col].to_numpy()
        ax.barh(range(len(p)), vals, left=left, color=color, edgecolor=SURFACE, linewidth=2, label=lab, height=0.62)
        for k, (x0, v) in enumerate(zip(left, vals)):
            if col != "both_wrong_pct" and v > 4:
                ax.text(x0 + v / 2, k, f"{v:.1f}", ha="center", va="center", fontsize=7, color="#ffffff")
        left += vals
    ax.set_yticks(range(len(p)))
    ax.set_yticklabels([f"{s.replace(' (partial', chr(10) + '(partial')}\nn = {int(n):,} rows/seed"
                        for s, n in zip(p["scope"], p["n_units"])])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of binary-eligible rows (mean over seeds 42-44)")
    ax.set_title(title, loc="left")
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, -0.16), frameon=False, fontsize=8, handlelength=1.2,
              columnspacing=1.0)
    fig.text(0.01, 0.005, caption, fontsize=7, color=INK_2, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def fig_response_lengths(cfg: StudyConfig, audit: pd.DataFrame, out: Path, labels: Sequence[str], caption: str,
                         source: str = "archive",
                         title: str = "Discovery (Stage B): response length") -> Path:
    """Median completion tokens per benchmark and model (seed-averaged)."""

    plt = _plt()
    order = suite_order(cfg)
    names = display_names(cfg)
    a = audit[(audit["source"] == source) & (audit["model_label"].isin(labels))]
    g = a.groupby(["model_label", "benchmark_id"])["completion_tokens_median"].mean()
    fig, ax = plt.subplots(figsize=(7.2, 0.27 * len(order) + 1.6))
    offsets = np.linspace(-0.2, 0.2, len(labels))
    for j, lab in enumerate(labels):
        ys = [i + offsets[j] for i, (_, b) in enumerate(order) if (lab, b) in g.index]
        xs = [g[(lab, b)] for _, b in order if (lab, b) in g.index]
        ax.plot(xs, ys, "o", color=color_of(lab), markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=1, label=DISPLAY.get(lab, lab))
    ax.set_xscale("log")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[b] for _, b in order])
    ax.invert_yaxis()
    ax.set_xlabel("median completion tokens (log scale)")
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right", frameon=False)
    fig.text(0.01, 0.005, caption, fontsize=7, color=INK_2, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def fig_model_scores(scores: pd.DataFrame, out: Path, labels: Sequence[str], title: str, caption: str,
                     scope_level: str = "overall") -> Path:
    """One row per model: macro score with its 95% interval (identity by position, color by entity)."""

    plt = _plt()
    s = scores[scores["level"] == scope_level].set_index("model_label")
    labels = [l for l in labels if l in s.index]
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(labels) + 1.3))
    for i, lab in enumerate(labels):
        r = s.loc[lab]
        c = color_of(lab)
        ax.hlines(i, r["ci_low"], r["ci_high"], color=c, linewidth=2.2)
        ax.plot(r["estimate"], i, "o", color=c, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.text(r["ci_high"] + 0.08, i, f"{r['estimate']:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([DISPLAY.get(l, l) for l in labels])
    ax.invert_yaxis()
    ax.set_xlabel(f"{s['scope'].iloc[0]} (official scores, percent)")
    ax.set_title(title, loc="left")
    fig.text(0.01, 0.005, caption, fontsize=7, color=INK_2, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def stage_b_figures(cfg: StudyConfig) -> list[Path]:
    t = cfg.work("tables", "stage_b")
    fig = cfg.work("figures")
    diffs = pd.read_csv(t / "differences.all.csv")
    units = json.loads((t / "bootstrap.all.json").read_text(encoding="utf-8"))["n_components"]
    out = [fig_benchmark_gains(
        cfg, diffs, fig / "stage_b_benchmark_gains.png", ["T-B", "V@P0-B"],
        "Discovery (Stage B): gains over the base model",
        "Discovery (Stage B), archived runs, full split, seeds 42-44 averaged. Dots: paired difference of official "
        f"scores; lines: 95% cluster-bootstrap intervals over {units:,} bootstrap units (image/problem groups, "
        "near-duplicate twins joined; 10,000 replicates). VERO with its own template.")]
    out.append(fig_benchmark_gains(
        cfg, diffs, fig / "stage_b_trace_minus_vero.png", ["T-V@P0"],
        "Discovery (Stage B): TRACE − VERO, per benchmark",
        f"Discovery (Stage B), archived runs, full split, seeds 42-44. 95% cluster-bootstrap intervals over {units:,} "
        "bootstrap units; 24 unadjusted exploratory comparisons. VERO with its own system prompt, TRACE with the "
        "backbone template."))
    out.append(fig_parent_overlap(pd.read_csv(t / "parent_overlap.csv"), fig / "stage_b_parent_overlap.png",
                                  "21 binary-eligible benchmarks (27,617 rows per seed); same-seed cells averaged over "
                                  "seeds 42-44. 'TRACE only' = T correct, VERO wrong. Hindsight cells, not a router.",
                                  category_order=list(cfg.categories())))
    audit = pd.read_csv(cfg.work("tables", "length_format_judge_audit.csv"))
    out.append(fig_response_lengths(cfg, audit, fig / "stage_b_response_lengths.png", ["B", "T", "V@P0"],
                                    "Archived runs, seeds 42-44 (per-seed medians averaged). VERO's responses include "
                                    "its <think> block; length stops are rare (B 313, T 14, VERO 8 of 98,415 each)."))
    return out


SHORT = {"B": "base", "T": "TRACE", "V@P0": "VERO (own template)", "V@P1": "VERO (backbone template)",
         "B@P2": "base@P2", "T@P2": "TRACE@P2"}


def short_label(label: str) -> str:
    """Compact legend name: parents spelled out, merges without the primary interface suffix."""

    return SHORT.get(label, label.removesuffix("@P1"))


def fig_category_contrasts(diffs: pd.DataFrame, out: Path, comparisons: Sequence[str], categories: Sequence[str],
                           title: str, caption: str) -> Path:
    """Dot-and-interval plot: the macro and each category, every comparison's paired difference with its 95%
    interval (Holm-adjusted p in the legend when the comparison is in the primary family)."""

    plt = _plt()
    d = diffs[diffs["level"].isin(["overall", "category"])].set_index(["comparison", "scope"])
    overall = diffs.loc[diffs["level"] == "overall", "scope"].iloc[0]
    scopes = [overall, *categories]
    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(scopes) + 1.6))
    offsets = np.linspace(-0.22, 0.22, len(comparisons)) if len(comparisons) > 1 else [0.0]
    refs = [comp.split("-", 1)[1] for comp in comparisons]
    colors = [color_of(r) for r in refs]
    if len(set(colors)) < len(colors):
        colors = [PALETTE[j % len(PALETTE)] for j in range(len(colors))]
    for j, comp in enumerate(comparisons):
        color = colors[j]
        ys, xs, lo, hi = [], [], [], []
        for i, sc in enumerate(scopes):
            if (comp, sc) in d.index:
                r = d.loc[(comp, sc)]
                ys.append(i + offsets[j]); xs.append(r["estimate"]); lo.append(r["ci_low"]); hi.append(r["ci_high"])
        if not xs:
            raise ValueError(f"no overall or category rows for {comp} in the differences table")
        label = " − ".join(short_label(x) for x in comp.split("-", 1))
        if (comp, overall) in d.index and "p_holm" in d.columns:
            ph = d.loc[(comp, overall)]["p_holm"]
            label += f" (Holm p = {ph:.3g})" if ph == ph else ""
        ax.hlines(ys, lo, hi, color=color, linewidth=2, alpha=0.85)
        ax.plot(xs, ys, "o", color=color, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1, label=label)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.axhline(0.5, color=GRID, linewidth=1.2)
    ax.set_yticks(range(len(scopes)))
    ax.set_yticklabels(scopes)
    ax.invert_yaxis()
    ax.set_xlabel("difference in official score (percentage points)")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.11), frameon=False, fontsize=8)
    fig.text(0.01, 0.005, caption, fontsize=7, color=INK_2, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def capture_overall(cd: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Overall pooled-row capture/damage rows of one variant (pooled over seeds or seed pairs when there are several)."""

    c = cd[(cd["level"] == "overall") & (cd["variant"] == variant)]
    if "seed_scope" in c.columns:
        pooled = c[c["seed_scope"].astype(str).str.contains("pool|mean", regex=True)]
        c = pooled if len(pooled) else c
    if "aggregation" in c.columns and (c["aggregation"] == "pooled_rows").any():
        c = c[c["aggregation"] == "pooled_rows"]
    return c


def fig_capture(cd: pd.DataFrame, out: Path, merges: Sequence[str], variant: str, title: str, caption: str) -> Path:
    """Per capture/damage event, each merge's conditional rate with its 95% interval; group sizes in the labels."""

    plt = _plt()
    c = capture_overall(cd, variant).set_index(["metric", "merge_label"])
    fig, ax = plt.subplots(figsize=(7.2, 0.62 * len(CAPTURE_EVENTS) * max(1, len(merges)) / 2 + 1.8))
    offsets = np.linspace(-0.24, 0.24, len(merges)) if len(merges) > 1 else [0.0]
    ylabels = []
    for i, (metric, name) in enumerate(CAPTURE_EVENTS):
        n = next((int(round(float(c.loc[(metric, m)]["n_group"]))) for m in merges if (metric, m) in c.index), None)
        ylabels.append(f"{name}\nn = {n:,} rows per seed" if n is not None else name)
        for j, m in enumerate(merges):
            if (metric, m) not in c.index:
                continue
            r = c.loc[(metric, m)]
            y = i + offsets[j]
            ax.hlines(y, r["conditional_rate_pct_ci_low"], r["conditional_rate_pct_ci_high"], color=color_of(m),
                      linewidth=2.2)
            ax.plot(r["conditional_rate_pct"], y, "o", color=color_of(m), markersize=6, markeredgecolor=SURFACE,
                    markeredgewidth=1, label=DISPLAY.get(m, m) if i == 0 else None)
    ax.set_yticks(range(len(CAPTURE_EVENTS)))
    ax.set_yticklabels(ylabels)
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_xlabel("conditional rate (% of the group's rows)")
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.text(0.01, 0.005, caption, fontsize=7, color=INK_2, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


P1_MODELS = ("B", "T", "V@P0", "V@P1", "M050@P1", "CV050@P1", "CT050@P1")
P2_MODELS = ("B@P2", "T@P2", "V@P0", "M050@P2", "CV050@P2", "CT050@P2")
MERGES = ("M050@P1", "CV050@P1", "CT050@P1")


def screen_figures(cfg: StudyConfig) -> list[Path]:
    t = cfg.work("tables", "screen")
    fig = cfg.work("figures")
    cats = list(cfg.categories())
    n = "development screen, seed 42, 2,910 rows (81-129 per benchmark)"
    ci = "95% paired cluster-bootstrap intervals over the screen's bootstrap units (10,000 replicates), unadjusted"
    out = [fig_model_scores(pd.read_csv(t / "scores.P1.csv"), fig / "screen_model_scores.P1.png", P1_MODELS,
                            "Screen (P1, backbone interface): 24-benchmark macro",
                            f"Development tier, descriptive only: {n}. Lines: {ci}.")]
    out.append(fig_model_scores(pd.read_csv(t / "scores.P2.csv"), fig / "screen_model_scores.P2.png", P2_MODELS,
                                "Screen diagnostic (P2, VERO interface): macro",
                                f"Development tier, diagnostic only: {n}. VERO's tokenizer, template and processor for "
                                f"every model; VERO itself is its archived own-template run. Lines: {ci}."))
    out.append(fig_category_contrasts(
        pd.read_csv(t / "differences.P1.csv"), fig / "screen_contrasts.P1.png",
        ["M050@P1-T", "M050@P1-V@P1", "M050@P1-CV050@P1", "M050@P1-CT050@P1"], cats,
        "Screen (P1): M050 against parents and controls",
        f"Development tier, descriptive only: {n}. Dots: paired difference of official scores; lines: {ci}."))
    out.append(fig_category_contrasts(
        pd.read_csv(t / "differences.P2.csv"), fig / "screen_contrasts.P2.png",
        ["M050@P2-V@P0", "M050@P2-CV050@P2", "M050@P2-CT050@P2"], cats,
        "Screen diagnostic (P2): M050 against VERO, controls",
        f"Development tier, diagnostic only: {n}. Dots: paired difference of official scores; lines: {ci}."))
    out.append(fig_capture(
        pd.read_csv(t / "merge_capture_damage.csv"), fig / "screen_capture_damage.png", MERGES, "contemporaneous",
        "Screen (P1): capture and damage, same-seed parents",
        "Development tier, seed 42, 21 binary-eligible benchmarks, rows pooled. Groups from T and VERO (backbone "
        f"template) on the same seed, so they include lucky parental failures. Lines: {ci}."))
    return out


def confirmation_figures(cfg: StudyConfig) -> list[Path]:
    t = cfg.work("tables", "confirmation")
    fig = cfg.work("figures")
    cats = list(cfg.categories())
    summary = json.loads((t / "summary.json").read_text(encoding="utf-8"))
    seeds = "-".join(str(s) for s in (summary["seeds"][0], summary["seeds"][-1]))
    n = f"confirmation split, 26,228 rows, seeds {seeds} averaged within model"
    ci = "95% paired cluster-bootstrap intervals over bootstrap units, seeds resampled together (10,000 replicates)"
    diffs = pd.read_csv(t / "differences.csv")
    # the lock names the family by model ids ("M050-T"); the table carries the resolved labels ("M050@P1-T")
    flagged = diffs["family_member"].astype(str).str.lower() == "true"
    family = list(dict.fromkeys(diffs.loc[flagged & (diffs["level"] == "overall"), "comparison"]))
    if len(family) != len(summary["family"]):
        raise ValueError(f"family {summary['family']} resolves to {family} in differences.csv")
    out = [fig_model_scores(pd.read_csv(t / "scores.csv"), fig / "confirmation_model_scores.png", P1_MODELS,
                            "Confirmation (P1): 24-benchmark macro",
                            f"Confirmation tier, prespecified models: {n}. Lines: {ci}.")]
    out.append(fig_category_contrasts(
        diffs, fig / "confirmation_primary_family.png", family, cats,
        "Confirmation (P1): the prespecified primary family",
        f"Confirmation tier: {n}. Dots: paired difference of official scores; lines: {ci}, unadjusted. Holm "
        f"adjustment over the {len(family)} macro comparisons."))
    out.append(fig_benchmark_gains(
        cfg, diffs, fig / "confirmation_benchmark_contrasts.png", [c for c in family if "CV050" in c or "CT050" in c],
        "Confirmation (P1): each half-update, per benchmark",
        f"Confirmation tier: {n}. M050 − CV050 adds TRACE's half-update to a VERO blend; M050 − CT050 adds VERO's to a "
        f"TRACE blend. Lines: {ci}, unadjusted (exploratory per benchmark)."))
    cd = pd.read_csv(t / "merge_capture_damage.csv")
    for variant in ("contemporaneous", "independent_repeat"):
        if (cd["variant"] == variant).any():
            out.append(fig_capture(
                cd, fig / f"confirmation_capture_damage.{variant}.png", MERGES, variant,
                f"Confirmation (P1): capture/damage, {variant.replace('_', ' ')}",
                f"Confirmation tier, 21 binary-eligible benchmarks, rows pooled over seeds. "
                + ("Parent groups and merge outcome from the same seed. " if variant == "contemporaneous" else
                   "Parent groups from one seed, merge outcome from a different seed (no shared draw). ")
                + f"Lines: {ci}."))
    return out


# ------------------------------------------------------------------------------------------------ export


@functools.lru_cache(maxsize=1)
def _release_patterns() -> tuple[list[tuple[str, re.Pattern[str]]], re.Pattern[str] | None]:
    """The repository scanner's secret patterns and machine-path pattern (scripts/check_public_release.py), when present."""

    path = REPO_ROOT / "scripts" / "check_public_release.py"
    if not path.is_file():
        return [], None
    spec = importlib.util.spec_from_file_location("_trace_merge_release_check", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(getattr(mod, "SECRET_PATTERNS", ())), getattr(mod, "MACHINE_PATH_PATTERN", None)


def sanitization_problems(text: str) -> list[str]:
    """Machine paths (this module's pattern and the repository scanner's) and token-shaped strings in ``text``."""

    secrets, machine = _release_patterns()
    problems = []
    if ABS_PATH.search(text) or (machine is not None and machine.search(text)):
        problems.append("an absolute machine path")
    if TOKEN.search(text) or any(pat.search(text) for _, pat in secrets):
        problems.append("a token-shaped string")
    return problems


def assert_sanitized(path: Path) -> None:
    problems = sanitization_problems(path.read_text(encoding="utf-8", errors="replace"))
    if problems:
        raise RuntimeError(f"{path} contains {' and '.join(problems)}")


# work-root files of the public bundle (handoff 11.2), as globs relative to the work root; row-level artifacts (row
# lists, aligned predictions, group/unit assignments, run directories) stay in the work root
BUNDLE_GLOBS = ("audit/*.md", "audit/*.json", "audit/*.csv", "manifests/*.json", "splits/split_manifest.json",
                "splits/bootstrap_units.json", "reports/*.md", "reports/*.json", "tables/*.csv", "tables/*.json",
                "tables/*/*.csv", "tables/*/*.json", "figures/*.png", "merges/*/merge_manifest.json",
                "merges/*/recipe.json", "tests/test_report.txt")


def bundle_files(cfg: StudyConfig) -> dict[str, Path]:
    """Published path -> work-root source for every file of the public bundle that exists."""

    root = cfg.work()
    files = {}
    for pattern in BUNDLE_GLOBS:
        for src in sorted(root.glob(pattern)):
            if src.is_file():
                files[src.relative_to(root).as_posix()] = src
    return files


def write_costs(cfg: StudyConfig) -> Path:
    costs = evaluation_costs(cfg)
    out = cfg.work("tables", "evaluation_costs.csv")
    costs.to_csv(out, index=False)
    totals = {"gpu_hours": round(float(costs["gpu_hours"].sum()), 2) if len(costs) else 0.0,
              "completion_tokens": int(pd.to_numeric(costs.get("completion_tokens"), errors="coerce").fillna(0).sum())
              if len(costs) else 0,
              "attempts": int(costs["attempts"].sum()) if len(costs) else 0,
              "failed_or_incomplete_attempts": int((costs["attempts"] - costs["successful_attempts"]).sum())
              if len(costs) else 0,
              "merged_weights_gib": costs.attrs.get("merged_weights_gib", 0.0),
              # run_dir_gib is the size of the run directory the seeds of one model and tier share: count it once
              "run_dirs_gib": round(float(costs.drop_duplicates(["view_slug", "tier"])["run_dir_gib"].sum()), 2)
              if len(costs) else 0.0}
    write_json(cfg.work("tables", "evaluation_costs.totals.json"), totals)
    return out


def export_results(cfg: StudyConfig, files: dict[str, Path], dest: Path | None = None) -> dict[str, str]:
    """Copy aggregate tables, reports and figures into experiments/trace_merge/results (sanitized, hashed).

    Every text file is prepared and checked in memory first (JSON has each whole machine path replaced by
    ``<local>``; other text must already be clean); nothing is written unless every file passes.
    """

    dest = dest or PROJECT_ROOT / "results"
    staged: dict[str, bytes] = {}
    bad = []
    for rel, src in files.items():
        if not src.is_file():
            continue
        if src.suffix in (".csv", ".md", ".json", ".txt"):
            text = src.read_text(encoding="utf-8")
            text = ABS_PATH.sub("<local>", text) if src.suffix == ".json" else text
            problems = sanitization_problems(text)
            if problems:
                bad.append(f"{rel}: {' and '.join(problems)}")
            staged[rel] = text.encode("utf-8")
        else:
            staged[rel] = src.read_bytes()
    if bad:
        raise RuntimeError("export refused, nothing written: " + "; ".join(bad))
    hashes = {}
    for rel, data in staged.items():
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        hashes[rel] = sha256_bytes(data)
    write_json(dest / "MANIFEST.json", {"files": hashes, "study_id": cfg.study_id,
                                        "note": "aggregates only; written by trace_merge.reports.export_results"})
    return hashes


# ------------------------------------------------------------------------------------------------ markdown tables


def _ci(r: Any, digits: int = 2) -> str:
    return f"{r['estimate']:.{digits}f} [{r['ci_low']:.{digits}f}, {r['ci_high']:.{digits}f}]"


def md_scores(scores: pd.DataFrame, labels: Sequence[str], categories: Sequence[str]) -> str:
    """Rows = models; columns = overall macro and each category, estimate [95% CI]."""

    s = scores.set_index(["model_label", "level", "scope"])
    overall_scope = scores.loc[scores["level"] == "overall", "scope"].iloc[0]
    head = f"| model | {overall_scope} | " + " | ".join(categories) + " |"
    lines = [head, "|" + "---|" * (len(categories) + 2)]
    for lab in labels:
        cells = [DISPLAY.get(lab, lab), _ci(s.loc[(lab, "overall", overall_scope)])]
        for c in categories:
            key = (lab, "category", c)
            cells.append(f"{s.loc[key]['estimate']:.2f}" if key in s.index else "-")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def md_differences(diffs: pd.DataFrame, comparisons: Sequence[str], levels: Sequence[str] = ("overall", "category"),
                   holm: bool = False) -> str:
    d = diffs[diffs["level"].isin(levels) & diffs["comparison"].isin(comparisons)]
    head = "| comparison | scope | difference (pp) [95% CI] | p |" + (" Holm p |" if holm else "")
    lines = [head, "|---|---|---|---|" + ("---|" if holm else "")]
    for comp in comparisons:
        for r in d[d["comparison"] == comp].itertuples():
            row = f"| {comp} | {r.scope} | {r.estimate:+.2f} [{r.ci_low:+.2f}, {r.ci_high:+.2f}] | {r.p_value:.3g} |"
            if holm:
                ph = getattr(r, "p_holm", None)
                row += f" {ph:.3g} |" if ph is not None and ph == ph else " - |"
            lines.append(row)
    return "\n".join(lines)


def md_capture(cd: pd.DataFrame, variant: str = "contemporaneous") -> str:
    """Overall capture/damage rows of each merge for one variant (pooled over seeds or seed pairs)."""

    c = cd[(cd["level"] == "overall") & (cd["variant"] == variant)]
    if "seed_scope" in c.columns:
        pooled = c[c["seed_scope"].astype(str).str.contains("pool|mean", regex=True)]
        c = pooled if len(pooled) else c
    if "aggregation" in c.columns and (c["aggregation"] == "pooled_rows").any():
        c = c[c["aggregation"] == "pooled_rows"]
    cols = [x for x in ("merge_label", "event", "n_group", "conditional_rate_pct", "conditional_rate_pct_ci_low",
                        "conditional_rate_pct_ci_high", "contribution_pp") if x in c.columns]
    lines = ["| merge | event | group size | rate % [95% CI] | contribution (pp of all rows) |", "|---|---|---|---|---|"]
    for r in c[cols].itertuples(index=False):
        d = r._asdict()
        ci = (f" [{d['conditional_rate_pct_ci_low']:.1f}, {d['conditional_rate_pct_ci_high']:.1f}]"
              if d.get("conditional_rate_pct_ci_low") == d.get("conditional_rate_pct_ci_low") else "")
        lines.append(f"| {d['merge_label']} | {d['event']} | {d.get('n_group', '-')} | "
                     f"{d['conditional_rate_pct']:.1f}{ci} | {d.get('contribution_pp', float('nan')):.2f} |")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ evaluation costs


def evaluation_costs(cfg: StudyConfig) -> pd.DataFrame:
    """GPU time, generated tokens, attempts and storage per run directory (handoff 8.1: log the real costs).

    GPU-hours are wall-clock seconds of each attempt (start to its last recorded phase) times the cards it held, from
    the runner's own ``timing.seed<S>.json``; tokens are completion tokens of the ingested rows; storage is the run
    directory's apparent size. Failed attempts count: they were real costs.
    """

    import re as _re

    rows = []
    runs = cfg.work("runs")
    index = {}
    idx = cfg.work("aligned", "index.json")
    if idx.is_file():
        index = json.loads(idx.read_text(encoding="utf-8")).get("local_runs", {})
    for timing in sorted(runs.glob("*/*/timing.seed*.json")):
        slug, subset = timing.parent.parent.name, timing.parent.name
        seed = int(_re.search(r"seed(\d+)", timing.name).group(1))
        data = json.loads(timing.read_text(encoding="utf-8"))
        gpu_h = wall_h = 0.0
        attempts = ok = 0
        for a in data.get("attempts", []):
            ep = a.get("epoch") or {}
            if "start" not in ep or len(ep) < 2:
                continue
            m = _re.match(r"(\d+)x", str(a.get("gpus") or ""))
            n = int(m.group(1)) if m else 1
            secs = max(ep.values()) - ep["start"]
            attempts += 1
            ok += int(a.get("rc") == 0)
            wall_h += secs / 3600
            gpu_h += n * secs / 3600
        tokens = None
        for name, entry in index.items():
            if entry.get("seed") == seed and name.endswith(f".{subset}.s{seed}") and slug_matches(cfg, name, slug):
                df = pd.read_parquet(cfg.work("aligned", entry["path"]), columns=["completion_tokens"])
                tokens = int(pd.to_numeric(df["completion_tokens"], errors="coerce").fillna(0).sum())
        size = sum(p.stat().st_size for p in timing.parent.rglob("*") if p.is_file())
        rows.append({"view_slug": slug, "tier": subset, "seed": seed, "attempts": attempts, "successful_attempts": ok,
                     "wall_hours": round(wall_h, 3), "gpu_hours": round(gpu_h, 3), "completion_tokens": tokens,
                     "run_dir_gib": round(size / 2**30, 3)})
    out = pd.DataFrame(rows)
    merges = cfg.work("merges")
    merge_gib = sum(p.stat().st_size for p in merges.rglob("*.safetensors")) / 2**30 if merges.is_dir() else 0.0
    if len(out):
        out.attrs["merged_weights_gib"] = round(merge_gib, 2)
    return out


def slug_matches(cfg: StudyConfig, run_name: str, slug: str) -> bool:
    from .views import VIEW_SPECS

    label = run_name.split(".")[0]
    return VIEW_SPECS.get(label, ("",))[0] == slug
