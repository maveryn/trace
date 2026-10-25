#!/usr/bin/env bash
# experiments/certificates/jobs/ckpt_eval.sh CKPT_DIR SERVED_NAME [--splits S,...] [--conds C,...]
#                                            [--seeds K,...] [--no-protocol] [--base HF_ID]
#                                            [--smoke] [--dry-run]
#
# DESIGN 7.4 evaluation of one LOCAL Hugging Face checkpoint (the M2 SFT dir, a merged RL actor, a
# kept intermediate step), written as ONE use-gpu task line. It is zs_eval.sh's pipeline with a
# local-path model, built from the same pieces: common.sh (environment, claims, lease, GPU checks,
# conditions file, trace-tool's scripts/serve_vllm.sh started with MODEL=<dir>, REVISION= empty and
# SERVED=<name>), payload_check.py, zs_report.py (count, check-data, validate, report, frame-diag),
# trace-tool-eval and the certificate CLI. What zs_eval.sh cannot do without edits, and this script
# adds: a chosen condition list, a served name that is not the path's basename, the model family read
# from config.json (not from the name), a content fingerprint as the trace-tool resume revision, and
# the released TRACE validation split.
#
#   source "$TRACE_CERT_RUNS/host/env.sh" && cd "$TRACE_ROOT" && \
#     bash experiments/certificates/jobs/ckpt_eval.sh "$TRACE_CERT_RUNS/sft/qwen2.5-vl-3b/final" s3b-sft
#
# Default plan: splits e1_dev, e1_test, e2_test (frozen public parquets) x conditions cert_v1 (px
# frame for Qwen2.5-VL, DESIGN D8) and ans_v1 x decoding seeds 42, 43, 44; plus `trace_val`, the
# released TRACE validation (maveryn/trace dataset-v1 4e5b5436, 2,000 rows, loaded by trace-tool's own
# pinned loader trace_tool.data with its sha256 check) under ans_v1 for the same seeds, and the TRACE
# protocol run (trace-tool's own `no_tool` condition: one turn, 2048 tokens) at seed 42 (--no-protocol
# drops it; it runs only when 42 is among the seeds). For intermediate checkpoints (DESIGN 7.4: steps
# 100/200 on E1-dev only) pass --splits e1_dev.
# Outputs (the zs_eval.sh layout, so zs_report.py and the host watcher read them the same way):
#   $TRACE_CERT_RUNS/eval/ckpt/<served>/<split>/<cond>/seed<k>/{responses.jsonl,run_config.json,cert/}
#   $TRACE_CERT_RUNS/eval/ckpt/<served>/trace_val/{ans_v1,no_tool}/seed<k>/{responses.jsonl,run_config.json,
#                                                  scores.jsonl,summary.json}  (trace-tool-score)
#   $TRACE_CERT_RUNS/eval/ckpt/<served>/{ckpt_manifest.json,plan.json,summary.json,summary.<split>.json,
#        summary.trace_val.json,summary.md,validation.json,validation.trace_val.json,
#        trace_tool_conditions_v1.yaml,logs/}
# trace-tool run_config.json records model=<served>, model_id=<checkpoint dir>, and
# model_revision=sha256:<fingerprint of weights, config, tokenizer, chat template, preprocessor>; a
# rerun into the same <served> with other weights is refused (exit 4) before anything is generated.
# The official TRACE score of trace_val needs the Qwen3-32B judge for rows the deterministic
# extractor cannot resolve: they are listed in trace_val/judge_pending.jsonl and count as wrong in
# `official_acc` (reported as a lower bound next to the deterministic and strict accuracies).
#
# --smoke: e1_dev x conditions x seed 42, 5 rows per task, and trace_val (+ protocol) seed 42 on the
#          first 20 rows, into eval/ckpt_smoke/<served>/.
# --dry-run: CPU preflight only (checkpoint files and fingerprint, chat template and image processor
#          against the base snapshot, conditions file, frozen-data sha256, request payload check,
#          trace-tool --dry-run of every condition, protocol condition check) and print the plan.
#
# Exit: 0 done; 1 incomplete (gaps, endpoint errors, server death, scoring failure); 2 usage;
#       3 GPUs busy or the run is owned by another live job; 4 configuration mismatch (checkpoint
#       files, family, template, fingerprint changed under the same served name, VRAM, data, env);
#       5 vLLM did not start; 130/143 signalled.
# Knobs: TRC_PAR, TRC_CONCURRENCY, TRC_REQUEST_TIMEOUT (1800), TRC_ERR_PASSES (3), TRC_MAX_LEN (16384),
#        TRC_MAX_IMAGES (1), TRC_SMOKE_PER_TASK (5), TRC_SMOKE_VAL_ROWS (20), TRC_OUT_ROOT (default
#        $TRACE_CERT_RUNS/eval/ckpt or eval/ckpt_smoke), TRC_ALLOW_TEMPLATE_DIFF=1 (proceed when the
#        checkpoint's rendered chat template or tokenizer differs from the base snapshot's; recorded).
set -uo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
# shellcheck source=common.sh
source "$(dirname "$SELF")/common.sh"

FROZEN_SPLITS="e1_dev e1_test e2_test"
VAL_SPLIT=trace_val      # directory name of the released TRACE validation split
VAL_COND=ans_v1          # its condition (answer-only prompt; the parquet has no cert/ann prompts)
PROTO_COND=no_tool       # trace-tool configs/conditions.yaml: the TRACE validation protocol exactly
PROTO_SEED=42

# ============================================================================== generation worker
# One trace-tool-eval run, started by the pool below with setsid so the parent can stop its whole
# process group; repeats while rows end in endpoint errors (zs_eval.sh's worker, plus the run's own
# config file, trace-tool split, dataset path and row selection, which differ for trace_val).

gen_worker() {  # split cond seed config tt_split dataset_path|- expected concurrency rcfile [selection]
  local split=$1 cond=$2 seed=$3 config=$4 ttsplit=$5 dsp=$6 expected=$7 conc=$8 rcfile=$9 sel=()
  read -ra sel <<<"${10:-}"
  local out=$TRC_W_ROOT/$split resp stat pass rc dsargs=()
  [ "$dsp" = - ] || dsargs=(--dataset-path "$dsp")
  resp=$out/$cond/seed$seed/responses.jsonl
  # shellcheck disable=SC1091
  source "$TRC_VENV/bin/activate"
  for pass in $(seq 1 "$TRC_W_PASSES"); do
    trc_log "gen $split/$cond/seed$seed pass $pass: concurrency $conc, split $ttsplit ${dsargs[*]:-(trace-tool pinned loader)}"
    trace-tool-eval --endpoint "$TRC_W_ENDPOINT" --model "$TRC_W_SERVED" --model-id "$TRC_W_MODEL_ID" \
      --model-revision "$TRC_W_REV" --family "$TRC_W_FAMILY" --config "$config" --conditions "$cond" \
      --seeds "$seed" --split "$ttsplit" ${dsargs[@]+"${dsargs[@]}"} --out "$out" \
      --concurrency "$conc" --request-timeout "$TRC_W_TIMEOUT" ${sel[@]+"${sel[@]}"}
    rc=$?
    stat=$(python "$TRC_JOBS_DIR/zs_report.py" count --responses "$resp" --expected "$expected")
    trc_log "gen $split/$cond/seed$seed pass $pass: rc=$rc $stat"
    case "$stat" in *complete=1*) echo 0 >"$rcfile"; return 0 ;; esac
    [ "$rc" -eq 0 ] || break   # not an endpoint problem (e.g. resume-key mismatch): do not loop
    trc_serve_healthy "$TRC_W_PORT" "$TRC_W_SERVED" || { trc_log "server unhealthy; no further pass"; break; }
  done
  echo 1 >"$rcfile"
  return 1
}

if [ "${1:-}" = "--_gen-worker" ]; then
  shift
  TRC_TAG="gen"
  gen_worker "$@"
  exit $?
fi

# ============================================================================== trace_val helper
# Python (trace-tool's env; stdlib + trace_tool + zs_report.py) for the released validation runs:
#   summary  ROOT RUNS_JSON OUT    per-run and pooled accuracies from trace-tool-score's scores.jsonl
#   validate ROOT RUNS_JSON REV OUT  gaps, endpoint errors, pinned dataset sha256, revision, stale scores
# RUNS_JSON: [{"cond": ..., "seed": ..., "expected": ...}, ...]

trval_py() {
  "$TRC_VENV/bin/python" - "$TRC_JOBS_DIR" "$@" <<'PY'
import json, math, sys
from pathlib import Path

jobs_dir, cmd, root = sys.argv[1], sys.argv[2], Path(sys.argv[3])
sys.path.insert(0, jobs_dir)
import zs_report  # noqa: E402  (stdlib only: response_counts, non_finite, count_lines)

runs = json.loads(sys.argv[4])
CAP = 2048


def rows(run):
    path = run / "scores.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def agg(rs):
    if not rs:
        return {"n": 0}
    tok = [int(r.get("completion_tokens") or 0) for r in rs]
    within = [r.get("stop_reason") == "answer" and t <= CAP for r, t in zip(rs, tok)]
    return {
        "n": len(rs),
        "strict_acc": mean(float(r["strict_correct"]) for r in rs),
        "deterministic_acc": mean(float(r["deterministic_correct"]) for r in rs),
        "official_acc_lower": mean(float(r["official_correct"]) for r in rs),
        "judge_pending": sum(r.get("official_source") == "judge_pending" for r in rs),
        "strict_format_rate": mean(float(r["strict_format"]) for r in rs),
        "completion_rate": mean(float(r.get("stop_reason") == "answer") for r in rs),
        "truncation_rate": mean(float(r.get("stop_reason") == "length") for r in rs),
        "within_2048_rate": mean(float(w) for w in within),
        "strict_acc_within_2048": mean(float(r["strict_correct"]) * w for r, w in zip(rs, within)),
        "deterministic_acc_within_2048": mean(float(r["deterministic_correct"]) * w for r, w in zip(rs, within)),
        "tokens_mean": mean(tok),
    }


if cmd == "summary":
    out = {"split": "validation (maveryn/trace dataset-v1, released)", "runs": [], "by_cond": {}}
    pooled = {}
    for spec in runs:
        run = root / spec["cond"] / f"seed{spec['seed']}"
        rs = rows(run)
        out["runs"].append({"cond": spec["cond"], "seed": spec["seed"], **agg(rs)})
        pooled.setdefault(spec["cond"], []).extend(rs)
    for cond, rs in pooled.items():
        out["by_cond"][cond] = {"seeds_pooled": sorted({s["seed"] for s in runs if s["cond"] == cond}), **agg(rs)}
    Path(sys.argv[5]).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    f = lambda v, p=True: "-" if v is None else (f"{100 * v:.2f}" if p else f"{v:.0f}")
    lines = ["## trace_val (released TRACE validation, answer only)", "",
             "| cond | seed | n | strict | deterministic | official (lower) | judge pending | format | complete | trunc | <=2048 | strict@2048 | tok mean |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in out["runs"] + [{"cond": c, "seed": "pooled", **v} for c, v in out["by_cond"].items()]:
        if not r.get("n"):
            lines.append(f"| {r['cond']} | {r['seed']} | 0 | - | - | - | - | - | - | - | - | - | - |")
            continue
        lines.append(f"| {r['cond']} | {r['seed']} | {r['n']} | {f(r['strict_acc'])} | {f(r['deterministic_acc'])} | "
                     f"{f(r['official_acc_lower'])} | {r['judge_pending']} | {f(r['strict_format_rate'])} | {f(r['completion_rate'])} | "
                     f"{f(r['truncation_rate'])} | {f(r['within_2048_rate'])} | {f(r['strict_acc_within_2048'])} | {f(r['tokens_mean'], False)} |")
    lines += ["", "Accuracies in %. strict = TRACE's GRPO exact_json reward accuracy; deterministic = validation-protocol",
              "extraction without the judge; official (lower) counts judge-pending rows as wrong. <=2048 and strict@2048",
              f"derive the 2048-token cap from the 8192 run (`{'no_tool'}` is the real 2048-token protocol run).", ""]
    print("\n".join(lines))
elif cmd == "validate":
    from trace_tool.data import REVISION, SPLIT_SHA256
    want_rev, out_path = sys.argv[5], Path(sys.argv[6])
    res = []
    for spec in runs:
        run = root / spec["cond"] / f"seed{spec['seed']}"
        r = {"run": f"trace_val/{spec['cond']}/seed{spec['seed']}", "expected": spec["expected"], "problems": []}
        resp = run / "responses.jsonl"
        if not resp.exists():
            r["problems"].append("missing")
            res.append(r)
            continue
        c = zs_report.response_counts(resp)
        r.update(c)
        if c["done"] < spec["expected"]:
            r["problems"].append(f"gap {spec['expected'] - c['done']} rows")
        if c["error_only"]:
            r["problems"].append(f"endpoint_errors {c['error_only']} rows")
        try:
            cfg = json.loads((run / "run_config.json").read_text())
            ds = cfg.get("dataset") or {}
            if ds.get("sha256") != SPLIT_SHA256["validation"] or ds.get("revision") != REVISION:
                r["problems"].append(f"dataset is not the pinned TRACE validation ({ds.get('sha256')}, {ds.get('revision')})")
            if cfg.get("model_revision") != want_rev:
                r["problems"].append(f"model_revision {cfg.get('model_revision')} != {want_rev}")
        except (OSError, ValueError) as exc:
            r["problems"].append(f"run_config: {exc}")
        scores = run / "scores.jsonl"
        if not scores.exists():
            r["problems"].append("unscored")
        else:
            r["scored"] = zs_report.count_lines(scores)
            if scores.stat().st_mtime < resp.stat().st_mtime:
                r["problems"].append("scores older than responses")
            if r["scored"] != c["done"]:
                r["problems"].append(f"scored {r['scored']} != done {c['done']}")
            try:
                bad = zs_report.non_finite(json.loads((run / "summary.json").read_text()))
                if bad:
                    r["problems"].append(f"nan in summary: {bad[:3]}")
            except (OSError, ValueError) as exc:
                r["problems"].append(f"summary: {exc}")
        res.append(r)
    failed = [r for r in res if r["problems"]]
    for r in failed:
        for p in r["problems"]:
            kind = "endpoint_errors" if p.startswith("endpoint_errors") else "nan" if p.startswith("nan") else "gap"
            print(f"FLAG {kind} {r['run']}: {p}")
    done, want = sum(r.get("done", 0) for r in res), sum(r["expected"] for r in res)
    out_path.write_text(json.dumps({"runs": res, "done": done, "expected": want, "ok": not failed}, indent=1, sort_keys=True) + "\n")
    print(f"VALIDATION trace_val {'OK' if not failed else 'FAILED'}: {done}/{want} rows, {len(res) - len(failed)}/{len(res)} runs clean")
    sys.exit(0 if not failed else 1)
else:
    sys.exit(f"unknown command {cmd}")
PY
}

# ============================================================================== main

usage() {
  echo "usage: ckpt_eval.sh CKPT_DIR SERVED_NAME [--splits e1_dev,e1_test,e2_test,trace_val] [--conds cert_v1,ans_v1]" >&2
  echo "                    [--seeds 42,43,44] [--no-protocol] [--base HF_ID] [--smoke] [--dry-run]" >&2
  exit 2
}
CKPT=""; SERVED=""; SMOKE=0; DRY=0; PROTO=1; BASE=""
SPLITS_ARG=""; CONDS_ARG=""; SEEDS_ARG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --splits) [ $# -ge 2 ] || usage; SPLITS_ARG=$2; shift ;;
    --conds) [ $# -ge 2 ] || usage; CONDS_ARG=$2; shift ;;
    --seeds) [ $# -ge 2 ] || usage; SEEDS_ARG=$2; shift ;;
    --base) [ $# -ge 2 ] || usage; BASE=$2; shift ;;
    --no-protocol) PROTO=0 ;;
    --smoke) SMOKE=1 ;;
    --dry-run) DRY=1 ;;
    -h|--help) usage ;;
    -*) echo "unknown option $1" >&2; usage ;;
    *) if [ -z "$CKPT" ]; then CKPT=$1; elif [ -z "$SERVED" ]; then SERVED=$1; else usage; fi ;;
  esac
  shift
done
[ -n "$CKPT" ] && [ -n "$SERVED" ] || usage
# the served name is a directory name, the vLLM model name and the trace-tool `model` resume key
[[ $SERVED =~ ^[a-z0-9][a-z0-9._-]{0,99}$ ]] || { echo "SERVED_NAME must match ^[a-z0-9][a-z0-9._-]*\$ (got '$SERVED')" >&2; exit 2; }

trc_require_env
trc_node_allowed || { echo "FATAL: config error: $(hostname -s) is excluded by fleet policy (${TRC_EXCLUDE_NODES:-grn030, grn078/079})" >&2; exit 4; }

# ---------------------------------------------------------------------------- checkpoint
[ -d "$CKPT" ] || { echo "FATAL: config error: checkpoint dir $CKPT does not exist" >&2; exit 4; }
CKPT=$(cd "$CKPT" && pwd)   # absolute, symlinks kept (the path is the recorded model_id)
for f in config.json preprocessor_config.json; do
  [ -e "$CKPT/$f" ] || { echo "FATAL: config error: $CKPT has no $f (not a servable HF checkpoint dir)" >&2; exit 4; }
done
if ! compgen -G "$CKPT/*.safetensors" >/dev/null; then
  echo "FATAL: config error: no *.safetensors in $CKPT" >&2; exit 4
fi
if ! { [ -e "$CKPT/tokenizer.json" ] || [ -e "$CKPT/tokenizer_config.json" ]; }; then
  echo "FATAL: config error: no tokenizer files in $CKPT" >&2; exit 4
fi
# family and size from config.json (a local path need not carry the family in its name)
read -r MODEL_TYPE HIDDEN < <("$TRACE_CERT_PYTHON" -I -c '
import json, sys
c = json.load(open(sys.argv[1]))
h = c.get("hidden_size") or (c.get("text_config") or {}).get("hidden_size") or 0
print(c.get("model_type", "?"), h)' "$CKPT/config.json") \
  || { echo "FATAL: config error: unreadable $CKPT/config.json" >&2; exit 4; }
case "$MODEL_TYPE" in
  qwen2_5_vl) FAMILY=qwen2.5-vl ;;
  qwen3_5|qwen3_5_moe) FAMILY=qwen3.5 ;;
  qwen3_vl|qwen3_vl_moe) FAMILY=qwen3-vl ;;
  *) FAMILY=other ;;
esac
FRAME=$(trc_frame "$FAMILY") || { echo "FATAL: config error: no certificate frame for model_type $MODEL_TYPE (family $FAMILY)" >&2; exit 4; }
if [ -z "$BASE" ]; then
  case "$FAMILY:$HIDDEN" in
    qwen2.5-vl:2048) BASE=Qwen/Qwen2.5-VL-3B-Instruct ;;
    qwen2.5-vl:3584) BASE=Qwen/Qwen2.5-VL-7B-Instruct ;;
    *) echo "FATAL: config error: cannot tell the base model of $CKPT ($MODEL_TYPE, hidden_size $HIDDEN); pass --base HF_ID" >&2; exit 4 ;;
  esac
fi
[ "$(trc_family "$BASE")" = "$FAMILY" ] || { echo "FATAL: config error: --base $BASE is not family $FAMILY" >&2; exit 4; }
trc_model_env "$BASE"
BASE_REV=$TRC_REVISION
[ -n "$BASE_REV" ] || { echo "FATAL: config error: no pinned revision for base $BASE in _model_env.sh" >&2; exit 4; }
BASE_DIR=$HF_HOME/hub/models--${BASE//\//--}/snapshots/$BASE_REV
[ -f "$BASE_DIR/config.json" ] || { echo "FATAL: config error: base snapshot $BASE_DIR is not cached" >&2; exit 4; }

if [ "$SMOKE" = 1 ]; then OUT_ROOT=$TRACE_CERT_RUNS/eval/ckpt_smoke; else OUT_ROOT=$TRACE_CERT_RUNS/eval/ckpt; fi
OUT_ROOT=${TRC_OUT_ROOT:-$OUT_ROOT}
ROOT=$OUT_ROOT/$SERVED
JOBTAG=${SLURM_JOB_ID:-local}.${SLURM_STEP_ID:-0}.$(date +%m%d_%H%M%S)
STATE=$ROOT/logs/job.$JOBTAG
TRC_TAG="ckpt:$SERVED"
mkdir -p "$STATE"
trc_log_open "$ROOT/logs/ckpt_eval.$JOBTAG.log"
trc_say "ckpt_eval.sh $CKPT as $SERVED: model_type $MODEL_TYPE, family $FAMILY, frame $FRAME, base $BASE@$BASE_REV, smoke=$SMOKE dry=$DRY"
trc_say "host $(hostname -s) job ${SLURM_JOB_ID:-none} step ${SLURM_STEP_ID:-none} cluster $(trc_cluster) N(dispatcher)=${N:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

declare -A WPID=()
GEN_DEAD=0
# shellcheck disable=SC2329  # invoked by the EXIT trap
on_exit() {
  local rc=$? pid
  trap - EXIT INT TERM HUP
  for pid in "${!WPID[@]}"; do trc_stop_tree "$pid" "generation worker" "${TRC_STOP_WAIT:-30}"; done
  if [ -n "${TRC_SERVER_PID:-}" ]; then
    trc_log "stopping vLLM (pid $TRC_SERVER_PID)"; trc_stop_tree "$TRC_SERVER_PID" vLLM "${TRC_STOP_WAIT:-120}"; TRC_SERVER_PID=""
  fi
  trc_heartbeat_stop   # first, so no heartbeat rewrite can resurrect a released claim
  trc_claim_release
  trc_lease_release
  trc_say "exit rc=$rc"
  exit "$rc"
}
trap on_exit EXIT
trap 'TRC_STOP_WAIT=20; trc_say "signal TERM"; exit 143' TERM
trap 'TRC_STOP_WAIT=20; trc_say "signal INT"; exit 130' INT
trap 'TRC_STOP_WAIT=20; trc_say "signal HUP"; exit 129' HUP

# ---------------------------------------------------------------------------- fingerprint
# sha256 of every file that changes what the server generates (weights, config, tokenizer, chat
# template, image processor), cached by (realpath, size, mtime). The digest is trace-tool's
# --model-revision (a resume key) and is pinned in ckpt_manifest.json for this served name.
MANIFEST=$ROOT/ckpt_manifest.json
REV=$("$TRACE_CERT_PYTHON" -I - "$CKPT" "$TRACE_CERT_RUNS/eval/.ckpt_sha256_cache.json" "$MANIFEST" "$SERVED" \
      "$FAMILY" "$FRAME" "$BASE" "$BASE_REV" "$MODEL_TYPE" 2>"$STATE/fingerprint.err" <<'PY'
import hashlib, json, os, sys, time
ckpt, cache_path, manifest, served, family, frame, base, base_rev, model_type = sys.argv[1:10]
SERVING = ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
           "merges.txt", "special_tokens_map.json", "added_tokens.json", "chat_template.json", "chat_template.jinja",
           "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json")
names = sorted(n for n in os.listdir(ckpt) if n.endswith(".safetensors")) + [n for n in SERVING if os.path.exists(os.path.join(ckpt, n))]
try:
    cache = json.load(open(cache_path))
except (OSError, ValueError):
    cache = {}
files = []
for name in names:
    real = os.path.realpath(os.path.join(ckpt, name))
    st = os.stat(real)
    key = f"{real}|{st.st_size}|{st.st_mtime_ns}"
    sha = cache.get(key)
    if not sha:
        h = hashlib.sha256()
        with open(real, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        sha = cache[key] = h.hexdigest()
    files.append({"name": name, "bytes": st.st_size, "sha256": sha})
tmp = f"{cache_path}.{os.getpid()}.tmp"
with open(tmp, "w") as f:
    json.dump(cache, f, indent=1, sort_keys=True)
os.replace(tmp, cache_path)
rev = "sha256:" + hashlib.sha256(json.dumps([[f["name"], f["sha256"]] for f in files], separators=(",", ":")).encode()).hexdigest()
new = {"served": served, "ckpt_dir": ckpt, "ckpt_realpath": os.path.realpath(ckpt), "model_type": model_type,
       "family": family, "frame": frame, "base": base, "base_revision": base_rev, "revision": rev, "files": files}
if os.path.exists(manifest):
    old = json.load(open(manifest))
    if old.get("revision") != rev or old.get("ckpt_dir") != ckpt:
        sys.exit(f"config error: {served} was evaluated from {old.get('ckpt_dir')} ({old.get('revision')}); "
                 f"this checkpoint is {ckpt} ({rev}). Use a new served name.")
else:
    new["created"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(manifest + ".tmp", "w") as f:
        json.dump(new, f, indent=1, sort_keys=True)
    os.replace(manifest + ".tmp", manifest)
print(rev)
PY
) || { cat "$STATE/fingerprint.err"; trc_say "FATAL: $(tail -n 1 "$STATE/fingerprint.err")"; exit 4; }
trc_say "checkpoint revision $REV ($MANIFEST)"

# ---------------------------------------------------------------------------- chat template
# vLLM serves the checkpoint with its own chat template and image processor: both must render and
# tokenize a fixed request exactly as the base snapshot does (SFT copies them from the base; the
# RL merger copies the actor's saved processor), and the image processor must not resize trace-tool's
# client-side images (patch/merge size equal, min_pixels <= 262144, max_pixels >= 4194304).
if ! "$TRC_VENV/bin/python" - "$CKPT" "$BASE_DIR" "$ROOT/template_check.json" >"$STATE/template_check.txt" 2>&1 <<'PY'
import json, sys
from transformers import AutoProcessor

ckpt, base, out = sys.argv[1:4]
MSGS = [{"role": "system", "content": "You are checked."},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "How many pieces? End with JSON."}]},
        {"role": "assistant", "content": '{"answer": 3}'},
        {"role": "user", "content": [{"type": "text", "text": "Again."}]}]


def probe(path):
    proc = AutoProcessor.from_pretrained(path, local_files_only=True)
    text = proc.apply_chat_template(MSGS, tokenize=False, add_generation_prompt=True)
    ids = proc.tokenizer(text)["input_ids"]
    ip = proc.image_processor
    size = getattr(ip, "size", None)  # dict (transformers 4.x) or SizeDict (5.x): both have .get

    def pick(*keys):
        for key in keys:
            v = getattr(ip, key, None)
            if v is None and size is not None and hasattr(size, "get"):
                v = size.get(key)
            if v is not None:
                return v
        return None

    lo, hi = pick("min_pixels", "shortest_edge"), pick("max_pixels", "longest_edge")
    return {"text": text, "ids": ids, "patch_size": getattr(ip, "patch_size", None),
            "merge_size": getattr(ip, "merge_size", None), "min_pixels": lo, "max_pixels": hi}


c, b = probe(ckpt), probe(base)
res = {"template_equal": c["text"] == b["text"], "token_ids_equal": c["ids"] == b["ids"],
       "geometry_equal": (c["patch_size"], c["merge_size"]) == (b["patch_size"], b["merge_size"]),
       "no_server_resize": (c["min_pixels"] or 0) <= 262144 and (c["max_pixels"] or 0) >= 4194304,
       "ckpt": {k: v for k, v in c.items() if k not in ("ids",)}, "base": {k: v for k, v in b.items() if k not in ("ids",)}}
json.dump(res, open(out, "w"), indent=1, sort_keys=True)
print(json.dumps({k: v for k, v in res.items() if k not in ("ckpt", "base")}, sort_keys=True))
geom_ok = res["geometry_equal"] and res["no_server_resize"]
tmpl_ok = res["template_equal"] and res["token_ids_equal"]
sys.exit(0 if geom_ok and tmpl_ok else (3 if geom_ok else 4))
PY
then
  tc_last=$(tail -n 1 "$STATE/template_check.txt")
  cat "$STATE/template_check.txt"
  if grep -q '"geometry_equal": true' "$STATE/template_check.txt" && grep -q '"no_server_resize": true' "$STATE/template_check.txt" \
     && [ "${TRC_ALLOW_TEMPLATE_DIFF:-0}" = 1 ]; then
    trc_flag "template_diff $SERVED: chat template/tokenizer differs from $BASE (TRC_ALLOW_TEMPLATE_DIFF=1; $ROOT/template_check.json)"
  else
    trc_say "FATAL: config error: checkpoint chat template / image processor check failed ($ROOT/template_check.json): $tc_last"; exit 4
  fi
else
  trc_say "template check ok vs $BASE: $(tail -n 1 "$STATE/template_check.txt")"
fi

# ---------------------------------------------------------------------------- plan
if [ "$SMOKE" = 1 ]; then
  SPLITS=(e1_dev "$VAL_SPLIT"); CONDS=(cert_v1 ans_v1); SEEDS=(42)
  SEL_FROZEN="--per-task ${TRC_SMOKE_PER_TASK:-5}"; SEL_VAL="--limit ${TRC_SMOKE_VAL_ROWS:-20}"
else
  SPLITS=(e1_dev e1_test e2_test "$VAL_SPLIT"); CONDS=(cert_v1 ans_v1); SEEDS=(42 43 44)
  SEL_FROZEN=""; SEL_VAL=""
fi
if [ -n "$SPLITS_ARG" ]; then IFS=, read -ra SPLITS <<<"$SPLITS_ARG"; fi
if [ -n "$CONDS_ARG" ]; then IFS=, read -ra CONDS <<<"$CONDS_ARG"; fi
if [ -n "$SEEDS_ARG" ]; then IFS=, read -ra SEEDS <<<"$SEEDS_ARG"; fi
FSPLITS=(); WANT_VAL=0
for s in "${SPLITS[@]}"; do
  case " $FROZEN_SPLITS " in
    *" $s "*) FSPLITS+=("$s") ;;
    *) if [ "$s" = "$VAL_SPLIT" ]; then WANT_VAL=1; else trc_say "FATAL: unknown split $s (use ${FROZEN_SPLITS// /,},$VAL_SPLIT)"; exit 2; fi ;;
  esac
done
for c in "${CONDS[@]}"; do
  case "$c" in ans_v1|ann_v1|cert_v1) ;; *) trc_say "FATAL: unknown condition $c (ans_v1, ann_v1, cert_v1)"; exit 2 ;; esac
done
for k in "${SEEDS[@]}"; do [[ $k =~ ^[0-9]+$ ]] || { trc_say "FATAL: bad seed $k"; exit 2; }; done
[ ${#FSPLITS[@]} -gt 0 ] || [ "$WANT_VAL" = 1 ] || usage
[ ${#SEEDS[@]} -gt 0 ] || { trc_say "FATAL: empty --seeds"; exit 2; }
[ ${#CONDS[@]} -gt 0 ] || [ ${#FSPLITS[@]} -eq 0 ] || { trc_say "FATAL: empty --conds"; exit 2; }
CONFIG=$ROOT/trace_tool_conditions_v1.yaml
TT_CONFIG=$TRACE_TOOL_ROOT/configs/conditions.yaml
DO_PROTO=0
if [ "$WANT_VAL" = 1 ] && [ "$PROTO" = 1 ] && [[ " ${SEEDS[*]} " == *" $PROTO_SEED "* ]]; then DO_PROTO=1; fi

parquet_of() { echo "$TRC_DATA/public/$1.$(trc_public_cond "$2" "$FRAME").parquet"; }
expected_of() {  # split parquet
  if [ "$SMOKE" = 1 ]; then
    "$TRACE_CERT_PYTHON" -I -c '
import collections, sys
import pyarrow.parquet as pq
n = int(sys.argv[2])
tasks = collections.Counter(pq.read_table(sys.argv[1], columns=["task"]).column("task").to_pylist())
print(sum(min(n, c) for c in tasks.values()))' "$2" "${TRC_SMOKE_PER_TASK:-5}"
  else
    trc_receipt_value "$1" 'd["totals"]["accepted"]'
  fi
}
declare -A EXPECT=()
for split in ${FSPLITS[@]+"${FSPLITS[@]}"}; do
  for cond in "${CONDS[@]}"; do
    pq=$(parquet_of "$split" "$cond")
    [ -n "${EXPECT[$split.$cond]:-}" ] || EXPECT[$split.$cond]=$(expected_of "$split" "$pq") \
      || { trc_say "FATAL: config error: cannot read the row count of $pq"; exit 4; }
  done
done
VAL_PARQUET=""; VAL_ROWS=0
if [ "$WANT_VAL" = 1 ]; then
  # the cached dataset-v1 parquet through trace-tool's own resolver (HF_HUB_OFFLINE=1: cache only)
  read -r VAL_PARQUET VAL_ROWS < <("$TRC_VENV/bin/python" -c '
import pyarrow.parquet as pq
from trace_tool.data import resolve_split_path
p = resolve_split_path("validation")
print(p, pq.ParquetFile(str(p)).metadata.num_rows)' 2>>"$STATE/trace_val_resolve.txt") \
    || { trc_say "FATAL: config error: the released TRACE validation parquet is not in the HF cache ($STATE/trace_val_resolve.txt)"; exit 4; }
  [[ $VAL_ROWS =~ ^[0-9]+$ ]] && [ "$VAL_ROWS" -gt 0 ] || { trc_say "FATAL: config error: cannot read $VAL_PARQUET"; exit 4; }
  if [ "$SMOKE" = 1 ] && [ "${TRC_SMOKE_VAL_ROWS:-20}" -lt "$VAL_ROWS" ]; then VAL_ROWS=${TRC_SMOKE_VAL_ROWS:-20}; fi
fi

# RUNS entries: "split cond seed expected"; everything else derives from (split, cond)
RUNS=()
for seed in "${SEEDS[@]}"; do
  for split in ${FSPLITS[@]+"${FSPLITS[@]}"}; do
    for cond in "${CONDS[@]}"; do RUNS+=("$split $cond $seed ${EXPECT[$split.$cond]}"); done
  done
  if [ "$WANT_VAL" = 1 ]; then
    RUNS+=("$VAL_SPLIT $VAL_COND $seed $VAL_ROWS")
    if [ "$DO_PROTO" = 1 ] && [ "$seed" = "$PROTO_SEED" ]; then RUNS+=("$VAL_SPLIT $PROTO_COND $seed $VAL_ROWS"); fi
  fi
done
run_args() {  # split cond -> "config tt_split dataset_path|- selection" (selection last, may hold spaces)
  if [ "$1" = "$VAL_SPLIT" ]; then
    if [ "$2" = "$PROTO_COND" ]; then printf '%s validation - %s' "$TT_CONFIG" "$SEL_VAL"
    else printf '%s validation - %s' "$CONFIG" "$SEL_VAL"; fi
  else
    printf '%s cert_%s %s %s' "$CONFIG" "$1" "$(parquet_of "$1" "$2")" "$SEL_FROZEN"
  fi
}
dataset_name() {  # split cond -> parquet basename for plan.json
  if [ "$1" = "$VAL_SPLIT" ]; then basename "$VAL_PARQUET"; else basename "$(parquet_of "$1" "$2")"; fi
}
desc=""
[ ${#FSPLITS[@]} -gt 0 ] && desc="${FSPLITS[*]} x ${CONDS[*]}"
[ "$WANT_VAL" = 1 ] && desc="${desc:+$desc + }$VAL_SPLIT x $VAL_COND$([ "$DO_PROTO" = 1 ] && echo " (+ $PROTO_COND@$PROTO_SEED)")"
trc_say "plan: ${#RUNS[@]} runs ($desc; seeds ${SEEDS[*]}${SEL_FROZEN:+; $SEL_FROZEN}${SEL_VAL:+; trace_val $SEL_VAL}) -> $ROOT"
write_plan() {
  local entry split cond seed exp sep="" total=0 tmp=$ROOT/plan.json.tmp.$$
  {
    printf '{"model": "%s", "served": "%s", "ckpt": true, "revision": "%s", "base": "%s", "family": "%s", "frame": "%s", "smoke": %s, "updated": "%s", "runs": [' \
      "$CKPT" "$SERVED" "$REV" "$BASE" "$FAMILY" "$FRAME" "$([ "$SMOKE" = 1 ] && echo true || echo false)" "$(trc_iso)"
    for entry in "${RUNS[@]}"; do
      read -r split cond seed exp <<<"$entry"
      printf '%s\n  {"split": "%s", "cond": "%s", "seed": %s, "dir": "%s/%s/seed%s", "parquet": "%s", "expected": %s}' \
        "$sep" "$split" "$cond" "$seed" "$split" "$cond" "$seed" "$(dataset_name "$split" "$cond")" "$exp"
      sep=","; total=$((total + exp))
    done
    printf '\n], "expected_total": %s}\n' "$total"
  } >"$tmp" && mv -f "$tmp" "$ROOT/plan.json"
}
write_plan

# ---------------------------------------------------------------------------- CPU preflight
trc_render_conditions "$CONFIG" || exit 4
trc_say "conditions: $CONFIG"
for split in ${FSPLITS[@]+"${FSPLITS[@]}"}; do
  args=()
  for cond in "${CONDS[@]}"; do args+=(--cond "$(trc_public_cond "$cond" "$FRAME")"); done
  "$TRACE_CERT_PYTHON" "$TRC_JOBS_DIR/zs_report.py" check-data --data "$TRC_DATA" --split "$split" "${args[@]}" \
    --cache "$TRACE_CERT_RUNS/eval/.data_sha256_cache.json" || { trc_say "FATAL: config error: frozen data check failed"; exit 4; }
done
# request payload (system + user text + one image, nothing else) on e1_dev for every condition this
# job sends, ans_v1 included when only trace_val runs (same condition); the family is the base's
PCONDS=("${CONDS[@]}")
if [ "$WANT_VAL" = 1 ] && [[ " ${PCONDS[*]} " != *" $VAL_COND "* ]]; then PCONDS+=("$VAL_COND"); fi
if [ ${#FSPLITS[@]} -eq 0 ]; then PCONDS=("$VAL_COND"); fi
CERT_SYSTEM=$TRACE_ROOT/src/trace_tasks/resources/prompts/experimental/certificates/cert_system_v1.txt
cases=()
for cond in "${PCONDS[@]}"; do
  if [ "$cond" = ans_v1 ]; then sys=$TRACE_TOOL_ROOT/prompts/trace_system_prompt.txt; else sys=$CERT_SYSTEM; fi
  cases+=(--case "$cond=$(parquet_of e1_dev "$cond")=$sys")
done
if ! "$TRC_VENV/bin/python" "$TRC_JOBS_DIR/payload_check.py" --config "$CONFIG" --model-id "$BASE" --served "$SERVED" \
    "${cases[@]}" >"$STATE/payload_check.jsonl"; then
  cat "$STATE/payload_check.jsonl"
  trc_say "FATAL: config error: the request payload check failed ($STATE/payload_check.jsonl)"; exit 4
fi
trc_say "payload check ok (${PCONDS[*]}): system text + user text + one image per request"
# the TRACE protocol condition is trace-tool's own; check it is still the protocol
if [ "$DO_PROTO" = 1 ]; then
  if ! "$TRC_VENV/bin/python" -c '
import sys
from trace_tool.conditions import load_conditions
s = load_conditions(sys.argv[1])[sys.argv[2]]
want = {"max_turns": 1, "max_tokens_per_turn": 2048, "max_total_tokens": 2048, "decoding": "trace", "tools": [],
        "oracle_crop": False, "min_pixels": 262144, "max_pixels": 4194304,
        "system_prompt_files": ["prompts/trace_system_prompt.txt"]}
bad = {k: s.get(k) for k, v in want.items() if s.get(k) != v}
assert not bad, f"{sys.argv[2]} is not the TRACE protocol: {bad}"
' "$TT_CONFIG" "$PROTO_COND" >"$STATE/protocol_check.txt" 2>&1; then
    cat "$STATE/protocol_check.txt"; trc_say "FATAL: config error: $TT_CONFIG $PROTO_COND check failed"; exit 4
  fi
  trc_say "protocol condition ok: $TT_CONFIG $PROTO_COND (1 turn, 2048 tokens, TRACE system prompt)"
fi
# trace-tool's own --dry-run rendering of the first row of every run kind (kept for the record);
# for trace_val this also runs trace-tool's sha256 check of the pinned validation parquet
dry_kinds=()
for cond in "${CONDS[@]}"; do [ ${#FSPLITS[@]} -gt 0 ] && dry_kinds+=("${FSPLITS[0]} $cond"); done
if [ "$WANT_VAL" = 1 ]; then dry_kinds+=("$VAL_SPLIT $VAL_COND"); [ "$DO_PROTO" = 1 ] && dry_kinds+=("$VAL_SPLIT $PROTO_COND"); fi
for kind in "${dry_kinds[@]}"; do
  read -r split cond <<<"$kind"
  read -r cfg ttsplit dsp _ <<<"$(run_args "$split" "$cond")"
  dsargs=(); [ "$dsp" = - ] || dsargs=(--dataset-path "$dsp")
  # shellcheck disable=SC1091
  if ! (source "$TRC_VENV/bin/activate" && trace-tool-eval --dry-run --model "$SERVED" --model-id "$CKPT" \
        --model-revision "$REV" --family "$FAMILY" --config "$cfg" --conditions "$cond" --split "$ttsplit" \
        ${dsargs[@]+"${dsargs[@]}"} --limit 1) >"$STATE/tt_dryrun.$split.$cond.txt" 2>&1 \
     || ! grep -q "tools=\[\]" "$STATE/tt_dryrun.$split.$cond.txt" \
     || grep -q " sha256 .* != pinned" "$STATE/tt_dryrun.$split.$cond.txt"; then
    cat "$STATE/tt_dryrun.$split.$cond.txt"
    trc_say "FATAL: config error: trace-tool-eval --dry-run failed for $split/$cond ($STATE/tt_dryrun.$split.$cond.txt)"; exit 4
  fi
done
trc_say "trace-tool --dry-run ok for: $(printf '%s; ' "${dry_kinds[@]}")($STATE/tt_dryrun.*.txt)"

count_run() {  # split cond seed expected
  "$TRACE_CERT_PYTHON" "$TRC_JOBS_DIR/zs_report.py" count --responses "$ROOT/$1/$2/seed$3/responses.jsonl" --expected "$4"
}
PENDING=()
for entry in "${RUNS[@]}"; do
  read -r split cond seed exp <<<"$entry"
  stat=$(count_run "$split" "$cond" "$seed" "$exp")
  case "$stat" in *complete=1*) ;; *) PENDING+=("$entry") ;; esac
done
trc_say "pending: ${#PENDING[@]} of ${#RUNS[@]} runs"
if [ "$DRY" = 1 ]; then
  for entry in "${PENDING[@]+"${PENDING[@]}"}"; do
    read -r split cond seed exp <<<"$entry"
    trc_say "  todo: $entry <- $(run_args "$split" "$cond")"
  done
  trc_say "dry run: no GPU touched"; exit 0
fi

trc_lease_take "$ROOT/.owner" || exit 3
trc_heartbeat_start   # lease now; restarted below once the claims exist (the loop copies the claim list)

# ---------------------------------------------------------------------------- GPUs + vLLM + generation
generate() {
  local total=${#PENDING[@]} next=0 par conc per fails=0 t0 util_done=0 unhealthy=0
  local pid key rc rcfile split cond seed exp cfg ttsplit dsp sel
  par=${TRC_PAR:-3}; [ "$SMOKE" = 1 ] && par=2
  conc=${TRC_CONCURRENCY:-$((64 * TRC_N))}
  per=$(( (conc + par - 1) / par ))
  export TRC_W_ENDPOINT="http://127.0.0.1:$TRC_SERVER_PORT/v1" TRC_W_PORT=$TRC_SERVER_PORT TRC_W_SERVED=$SERVED \
    TRC_W_MODEL_ID=$CKPT TRC_W_REV=$REV TRC_W_FAMILY=$FAMILY TRC_W_ROOT=$ROOT \
    TRC_W_TIMEOUT=${TRC_REQUEST_TIMEOUT:-1800} TRC_W_PASSES=${TRC_ERR_PASSES:-3}
  t0=$(date +%s)
  while :; do
    for pid in "${!WPID[@]}"; do
      kill -0 "$pid" 2>/dev/null && continue
      wait "$pid" 2>/dev/null
      key=${WPID[$pid]}; unset "WPID[$pid]"
      rc=$(cat "$STATE/gen.${key//\//.}.rc" 2>/dev/null || echo '?')
      if [ "$rc" = 0 ]; then trc_say "done $key"; else trc_say "INCOMPLETE $key (rc=$rc, log logs/gen.${key//\//.}.log)"; fails=$((fails + 1)); fi
    done
    if trc_serve_healthy "$TRC_SERVER_PORT" "$SERVED"; then unhealthy=0; else unhealthy=$((unhealthy + 1)); trc_log "health check failed ($unhealthy)"; fi
    if [ "$unhealthy" -ge 3 ] || ! kill -0 "$TRC_SERVER_PID" 2>/dev/null; then
      trc_flag "server_dead vLLM on port $TRC_SERVER_PORT stopped answering; last lines of its log follow"
      tail -n 40 "$VLLM_LOG"
      for pid in "${!WPID[@]}"; do trc_stop_tree "$pid" "generation worker" 30; unset "WPID[$pid]"; done
      GEN_DEAD=1
      break
    fi
    while [ ${#WPID[@]} -lt "$par" ] && [ "$next" -lt "$total" ]; do
      read -r split cond seed exp <<<"${PENDING[$next]}"
      next=$((next + 1))
      read -r cfg ttsplit dsp sel <<<"$(run_args "$split" "$cond")"
      key="$split/$cond/seed$seed"
      rcfile=$STATE/gen.$split.$cond.seed$seed.rc
      rm -f "$rcfile"
      setsid bash "$SELF" --_gen-worker "$split" "$cond" "$seed" "$cfg" "$ttsplit" "$dsp" "$exp" "$per" "$rcfile" "$sel" \
        >>"$ROOT/logs/gen.$split.$cond.seed$seed.log" 2>&1 </dev/null 3>&- &
      WPID[$!]=$key
      trc_say "start $key ($exp rows, concurrency $per)"
    done
    if [ "$util_done" = 0 ] && [ $(( $(date +%s) - t0 )) -ge 60 ]; then
      util_done=1
      if [ ${#WPID[@]} -gt 0 ]; then trc_util_check || true; fi
    fi
    if [ ${#WPID[@]} -eq 0 ] && [ "$next" -ge "$total" ]; then break; fi
    sleep 15
  done
  [ "$util_done" = 1 ] || trc_say "UTILCHECK skipped: generation finished within 60 s"
  GEN_FAILS=$fails
}

GEN_FAILS=0
if [ ${#PENDING[@]} -gt 0 ]; then
  trc_gpu_discover || exit 4
  while read -r line; do trc_say "  $line"; done < <(trc_gpu_describe)
  [ -z "${N:-}" ] || [ "$N" = "$TRC_N" ] || trc_say "note: dispatcher N=$N, this step sees $TRC_N GPUs (using $TRC_N)"
  need_gb=$(trc_model_min_vram_gb "$BASE")
  if [ "$TRC_MIN_MIB" -lt $((need_gb * 1000)) ]; then
    trc_say "FATAL: config error: $SERVED ($BASE size) needs >= ${need_gb} GB per card; the smallest card here has ${TRC_MIN_MIB} MiB"; exit 4
  fi
  if busy=$(trc_gpu_busy); then trc_say "GPUs busy, not starting: $busy"; exit 3; fi
  if busy=$(trc_foreign_claimed); then trc_say "GPUs claimed, not starting: $busy"; exit 3; fi
  TRC_CLAIM_WHO="ckpt_eval.sh $SERVED$([ "$SMOKE" = 1 ] && echo ' --smoke')"
  TRC_CLAIM_RUN=$ROOT; TRC_CLAIM_TASK="trc:ckpt_eval:$SERVED"
  trc_claim_take >"$STATE/claims.txt" 2>&1; rc=$?
  trc_say "$(tr '\n' ' ' <"$STATE/claims.txt")"
  [ "$rc" -eq 0 ] || exit 3
  trc_heartbeat_stop; trc_heartbeat_start
  sleep 10
  if busy=$(trc_gpu_busy); then trc_say "GPUs became busy after claiming, releasing: $busy"; exit 3; fi
  PORT=$(trc_pick_port) || { trc_say "no free port"; exit 5; }
  VLLM_LOG=$ROOT/logs/vllm.$JOBTAG.log
  # shellcheck disable=SC1091
  source "$TRC_VENV/bin/activate"
  # local path: MODEL=<dir>, REVISION= (empty: serve_vllm.sh passes no --revision), SERVED=<name>
  trc_serve_start "$CKPT" "" "$SERVED" "$PORT" "$TRC_N" "$VLLM_LOG" || exit 5
  generate
  trc_serve_stop
  trc_heartbeat_stop; trc_claim_release; trc_heartbeat_start   # GPUs free again; keep the run lease
  trc_say "generation finished: ${GEN_FAILS} incomplete runs, server_dead=$GEN_DEAD; GPUs released"
fi

# ---------------------------------------------------------------------------- scoring (CPU)
SCORE_FAILS=0
SCORE_FILES=()
declare -A SPLIT_FILES=()
VAL_SPECS=""
for entry in "${RUNS[@]}"; do
  read -r split cond seed exp <<<"$entry"
  if [ "$split" = "$VAL_SPLIT" ]; then
    VAL_SPECS="$VAL_SPECS${VAL_SPECS:+,}{\"cond\": \"$cond\", \"seed\": $seed, \"expected\": $exp}"
    continue
  fi
  run=$ROOT/$split/$cond/seed$seed
  [ -f "$run/responses.jsonl" ] || continue
  stat=$(count_run "$split" "$cond" "$seed" "$exp")
  case "$stat" in *complete=1*) ;; *) trc_log "not scoring $split/$cond/seed$seed: $stat"; continue ;; esac
  if ! { [ -f "$run/cert/score_receipt.json" ] && [ "$run/cert/cert_scores.jsonl" -nt "$run/responses.jsonl" ]; }; then
    if (cd "$TRACE_ROOT" && nice -n 5 "$TRACE_CERT_PYTHON" -m trace_tasks.experimental.certificates score \
          --responses "$run/responses.jsonl" --store "$TRC_DATA/private/$split.extensions_v1.jsonl.zst" \
          --receipt "$TRC_DATA/receipts/$split.json" --family "$FAMILY" --out "$run/cert") >>"$ROOT/logs/score.log" 2>&1; then
      trc_say "scored $split/$cond/seed$seed"
    else
      trc_flag "score_failed $split/$cond/seed$seed (see logs/score.log)"; SCORE_FAILS=$((SCORE_FAILS + 1)); continue
    fi
  fi
  SCORE_FILES+=("$run/cert/cert_scores.jsonl")
  SPLIT_FILES[$split]="${SPLIT_FILES[$split]:-} $run/cert/cert_scores.jsonl"
done
summarize() {  # OUT FILES...
  local out=$1; shift
  (cd "$TRACE_ROOT" && "$TRACE_CERT_PYTHON" -m trace_tasks.experimental.certificates summarize --scores "$@" --out "$out") \
    >>"$ROOT/logs/score.log" 2>&1 || { trc_flag "summarize_failed $out"; SCORE_FAILS=$((SCORE_FAILS + 1)); }
}
REPORT_MD=""
if [ ${#SCORE_FILES[@]} -gt 0 ]; then
  summarize "$ROOT/summary.json" "${SCORE_FILES[@]}"
  for split in "${!SPLIT_FILES[@]}"; do
    read -ra files <<<"${SPLIT_FILES[$split]}"
    summarize "$ROOT/summary.$split.json" "${files[@]}"
  done
  REPORT_MD=$("$TRACE_CERT_PYTHON" "$TRC_JOBS_DIR/zs_report.py" report --model-root "$ROOT" --splits "$(IFS=,; echo "${FSPLITS[*]}")") \
    || { trc_flag "report_failed $ROOT/summary.md"; REPORT_MD=""; }
  REPORT_MD=${REPORT_MD/\# Zero-shot certificate evaluation:/# Checkpoint certificate evaluation:}
fi
VAL_MD=""
if [ -n "$VAL_SPECS" ] && compgen -G "$ROOT/$VAL_SPLIT/*/seed*/responses.jsonl" >/dev/null; then
  # shellcheck disable=SC1091
  if (source "$TRC_VENV/bin/activate" && nice -n 5 trace-tool-score "$ROOT/$VAL_SPLIT" --split validation \
        --pending-out "$ROOT/$VAL_SPLIT/judge_pending.jsonl" --report "$ROOT/$VAL_SPLIT/report.md") >>"$ROOT/logs/score.log" 2>&1; then
    trc_say "scored $VAL_SPLIT ($(wc -l <"$ROOT/$VAL_SPLIT/judge_pending.jsonl") responses await the Qwen3-32B judge)"
  else
    trc_flag "score_failed $VAL_SPLIT (see logs/score.log)"; SCORE_FAILS=$((SCORE_FAILS + 1))
  fi
  VAL_MD=$(trval_py summary "$ROOT/$VAL_SPLIT" "[$VAL_SPECS]" "$ROOT/summary.$VAL_SPLIT.json" 2>>"$ROOT/logs/score.log") \
    || { trc_flag "summarize_failed $ROOT/summary.$VAL_SPLIT.json"; SCORE_FAILS=$((SCORE_FAILS + 1)); VAL_MD=""; }
fi
if [ -n "$REPORT_MD$VAL_MD" ]; then
  {
    [ -n "$REPORT_MD" ] && printf '%s\n\n' "$REPORT_MD" || printf '# Checkpoint evaluation: %s\n\n' "$SERVED"
    printf 'Checkpoint: %s (%s), base %s, revision %s\n\n' "$CKPT" "$MODEL_TYPE" "$BASE" "$REV"
    # an `if`, not `[ ] && printf`: the group's status gates the mv, and without trace_val (--splits
    # e1_dev) a false test as the last command left summary.md.tmp.<pid> behind
    if [ -n "$VAL_MD" ]; then printf '%s\n' "$VAL_MD"; fi
  } >"$ROOT/summary.md.tmp.$$" && mv -f "$ROOT/summary.md.tmp.$$" "$ROOT/summary.md"
  trc_say "summary: $ROOT/summary.md"
  if [ "$SMOKE" = 1 ]; then
    while IFS= read -r line; do trc_say "$line"; done < <(grep '^|' "$ROOT/summary.md")
  fi
fi

# frame obedience on e1_dev cert_v1 seed 42 (logged; trained models should answer in source px)
diag_run=$ROOT/e1_dev/cert_v1/seed42
if [ -f "$diag_run/cert/cert_scores.jsonl" ]; then
  diag=$("$TRACE_CERT_PYTHON" "$TRC_JOBS_DIR/zs_report.py" frame-diag --run "$diag_run" \
    --store "$TRC_DATA/private/e1_dev.extensions_v1.jsonl.zst" --out "$diag_run/cert/frame_diag.json" 2>&1) \
    || trc_flag "frame_diag_failed $diag_run"
  printf '%s\n' "$diag"; trc_say "$(grep '^frame obedience' <<<"$diag")"
fi

# ---------------------------------------------------------------------------- validation
VRC=0
if [ ${#FSPLITS[@]} -gt 0 ]; then
  vargs=(--model-root "$ROOT" --data "$TRC_DATA" --splits "$(IFS=,; echo "${FSPLITS[*]}")" --conds "$(IFS=,; echo "${CONDS[*]}")"
         --seeds "$(IFS=,; echo "${SEEDS[*]}")" --frame "$FRAME" --out "$ROOT/validation.json")
  if [ "$SMOKE" = 1 ]; then vargs+=(--expect-rows "${EXPECT[e1_dev.${CONDS[0]}]}"); fi
  vout=$("$TRACE_CERT_PYTHON" "$TRC_JOBS_DIR/zs_report.py" validate "${vargs[@]}" 2>&1) || VRC=1
  while IFS= read -r line; do trc_say "$line"; done <<<"$vout"
fi
if [ -n "$VAL_SPECS" ]; then
  vout=$(trval_py validate "$ROOT/$VAL_SPLIT" "[$VAL_SPECS]" "$REV" "$ROOT/validation.$VAL_SPLIT.json" 2>&1) || VRC=1
  while IFS= read -r line; do trc_say "$line"; done <<<"$vout"
fi
if [ "$VRC" -ne 0 ] || [ "$GEN_DEAD" = 1 ] || [ "$SCORE_FAILS" -gt 0 ]; then
  trc_say "INCOMPLETE: rerun the same task line; finished rows are kept"
  exit 1
fi
trc_say "COMPLETE: $SERVED $([ "$SMOKE" = 1 ] && echo smoke || echo full) checkpoint evaluation validated"
exit 0
