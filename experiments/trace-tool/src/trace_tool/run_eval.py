"""Generate responses for TRACE rows under one or more conditions.

Example:
    trace-tool-eval --endpoint http://127.0.0.1:18100/v1 --model qwen2.5-vl-7b-instruct \
        --conditions no_tool,zoom,ocr --seeds 42 --out runs/full/qwen2.5-vl-7b-instruct

Writes ``<out>/<condition>/seed<seed>/responses.jsonl`` (resumable) and
``run_config.json`` next to it.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from tqdm import tqdm

from . import __version__
from .agent import OpenAIChatBackend, build_initial_messages, run_episode
from .conditions import DEFAULT_CONFIG, REPO_ROOT, build_episode_config, describe, load_conditions
from .data import REVISION, SPLIT_SHA256, file_sha256, load_split, resolve_split_path, select_rows
from .jsonl import append_jsonl, exclusive_lock, read_jsonl, repair_tail
from .models import FAMILIES, chat_template_kwargs, detect_family, get_family
from .prompting import render_text

log = logging.getLogger("trace_tool.run_eval")


def _git(*cmd: str) -> str | None:
    try:
        return subprocess.check_output(["git", *cmd], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _git_state() -> dict[str, Any]:
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"git_commit": _git("rev-parse", "HEAD"), "git_dirty": None if status is None else bool(status)}


def _code_sha256() -> str:
    """Hash of the code and configs that shape a run, so edits between resumes are flagged
    even without a commit (git_commit/git_dirty cannot tell two dirty trees apart)."""
    digest = hashlib.sha256()
    for sub in ("src/trace_tool", "configs", "prompts"):
        for path in sorted((REPO_ROOT / sub).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".yaml", ".txt", ".jinja"}:
                digest.update(str(path.relative_to(REPO_ROOT)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def _done_ids(path: Path) -> set[str]:
    """Rows with at least one non-error record (endpoint errors are retried; see score_runs.load_responses)."""
    records, _ = read_jsonl(path)
    return {r["instance_id"] for r in records if r.get("stop_reason") != "error" and "instance_id" in r}


# A rerun into an existing run directory must not change any of these.
_RESUME_KEYS = ("model", "model_id", "model_revision", "condition", "seed", "settings", "dataset")
# These may change (e.g. a bug fix mid-sweep) but are flagged.
_CODE_KEYS = ("trace_tool_version", "git_commit", "git_dirty", "code_sha256")


def _check_or_write_config(path: Path, config: dict[str, Any]) -> None:
    if path.exists():
        old = json.loads(path.read_text())
        diff = [k for k in _RESUME_KEYS if old.get(k) != config.get(k)]
        if diff:
            raise SystemExit(f"{path} exists with different {diff}; use a new --out or delete the directory")
        changed = [k for k in _CODE_KEYS if old.get(k) != config.get(k)]
        if changed:
            print(f"warning: resuming {path.parent} with different {changed} "
                  f"({[old.get(k) for k in changed]} -> {[config.get(k) for k in changed]})", file=sys.stderr)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")


async def _run_condition(
    name: str,
    spec: dict[str, Any],
    rows: list,
    args: argparse.Namespace,
    backend: OpenAIChatBackend,
    pool: ThreadPoolExecutor,
    seed: int,
) -> None:
    cfg = build_episode_config(spec, seed=seed, family=args.family_obj)
    out_dir = Path(args.out) / name / f"seed{seed}"
    config = {
        "model": args.model,
        "model_id": args.model_id or args.model,
        "model_revision": args.model_revision,
        "condition": name,
        "seed": seed,
        "settings": describe(spec, cfg),
        "dataset": {
            "split": args.split,
            "revision": REVISION if args.dataset_sha256 == SPLIT_SHA256.get(args.split) else None,
            "path": args.dataset_path,
            "sha256": args.dataset_sha256,
            "selection": args.selection,
        },
        "endpoint": args.endpoint,
        "trace_tool_version": __version__,
        **_git_state(),
        "code_sha256": args.code_sha256,
        "python": platform.python_version(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(out_dir / ".lock"):
        _check_or_write_config(out_dir / "run_config.json", config)
        await _generate(name, seed, cfg, rows, args, backend, pool, out_dir / "responses.jsonl")


async def _generate(name, seed, cfg, rows, args, backend, pool, out_path: Path) -> None:
    repair_tail(out_path)
    done = _done_ids(out_path)
    todo = [row for row in rows if row.instance_id not in done]
    if not todo:
        print(f"[{name} seed{seed}] complete ({len(done)} rows)")
        return

    semaphore = asyncio.Semaphore(args.concurrency)
    progress = tqdm(total=len(rows), initial=len(rows) - len(todo), desc=f"{name}/seed{seed}", dynamic_ncols=True)
    lock = asyncio.Lock()
    errors = 0

    async def one(row) -> None:
        nonlocal errors
        async with semaphore:
            record = await run_episode(row, cfg, backend, pool)
        record.update(condition=name, seed=seed, model=args.model)
        if not args.keep_transcripts:
            record.pop("transcript", None)
        async with lock:
            append_jsonl(out_path, record)
            errors += record["stop_reason"] == "error"
            progress.update(1)
            progress.set_postfix(errors=errors)

    await asyncio.gather(*(one(row) for row in todo))
    progress.close()
    if errors:
        print(f"[{name} seed{seed}] {errors} episodes ended with an endpoint error (see `error` in {out_path}); "
              "rerun the same command to retry them", file=sys.stderr)


def _resolve_family(args: argparse.Namespace):
    if args.family:
        return get_family(args.family)
    for candidate in (args.model_id, args.model):
        if candidate:
            try:
                return detect_family(candidate)
            except ValueError:
                continue
    raise SystemExit(f"cannot tell the model family of {args.model_id or args.model!r}; pass --family {'/'.join(sorted(FAMILIES))}")


def _dry_run(name: str, spec: dict[str, Any], row, family) -> None:
    cfg = build_episode_config(spec, seed=0, family=family)
    messages, ctx, shown = build_initial_messages(row, cfg)
    sizes = [list(img.size) for img in ctx.images]
    print(f"===== {name}: {row.instance_id} ({family.name}, shown image {list(shown[0].size)}, all images {sizes}, tools={sorted(cfg.tools)})")
    print(render_text(messages))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--endpoint", default=os.environ.get("TRACE_TOOL_ENDPOINT", "http://127.0.0.1:18100/v1"))
    p.add_argument("--model", required=True, help="served model name on the endpoint")
    p.add_argument("--model-id", help="HF id recorded in run_config.json (default: --model)")
    p.add_argument("--model-revision", help="checkpoint revision recorded in run_config.json (a resume key)")
    p.add_argument("--family", choices=sorted(FAMILIES), help="model family (default: detected from --model-id / --model)")
    p.add_argument("--conditions", required=True, help="comma-separated names from the config")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--seeds", default="42", help="comma-separated decoding seeds")
    p.add_argument("--out", help="output root (required unless --dry-run)")
    p.add_argument("--split", default="validation")
    p.add_argument("--dataset-path", help="local parquet instead of downloading from the Hub")
    p.add_argument("--per-task", type=int, help="keep the first K rows of each task (1 -> 1000 rows)")
    p.add_argument("--domains", help="comma-separated domain filter")
    p.add_argument("--limit", type=int)
    p.add_argument("--shard", help="i/n: take every n-th row starting at i")
    p.add_argument("--concurrency", type=int, default=64, help="episodes in flight")
    p.add_argument("--tool-threads", type=int, default=16, help="thread pool for tool execution")
    p.add_argument("--request-timeout", type=float, default=600.0)
    p.add_argument("--keep-transcripts", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dry-run", action="store_true", help="print the first row's rendered prompt per condition and exit")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    conditions = load_conditions(args.config)
    names = [c.strip() for c in args.conditions.split(",") if c.strip()]
    missing = [n for n in names if n not in conditions]
    if missing:
        raise SystemExit(f"unknown conditions {missing}; available: {sorted(conditions)}")
    shard = tuple(int(x) for x in args.shard.split("/")) if args.shard else None
    args.selection = {"per_task": args.per_task, "domains": args.domains, "limit": args.limit, "shard": args.shard}

    if not args.dry_run and not args.out:
        raise SystemExit("--out is required")
    args.family_obj = _resolve_family(args)
    dataset_path = resolve_split_path(args.split, args.dataset_path)
    args.dataset_sha256 = file_sha256(dataset_path)
    all_rows = load_split(args.split, local_path=str(dataset_path) if args.dataset_path else None)
    domains = [d.strip() for d in args.domains.split(",") if d.strip()] if args.domains else None
    unknown = sorted(set(domains or []) - {r.domain for r in all_rows})
    if unknown:
        raise SystemExit(f"unknown domains {unknown}; available: {sorted({r.domain for r in all_rows})}")
    rows = select_rows(all_rows, per_task=args.per_task, domains=domains, limit=args.limit, shard=shard)  # type: ignore[arg-type]
    if not rows:
        raise SystemExit("the selection is empty")
    print(f"{len(rows)} rows selected")
    if args.dry_run:
        for name in names:
            _dry_run(name, conditions[name], rows[0], args.family_obj)
        return

    seeds = [int(s) for s in args.seeds.split(",")]
    # Build every condition's tools now: a broken backend (e.g. OCR without OpenCV)
    # must stop the sweep before any episode runs, not when it reaches that condition.
    for name in names:
        try:
            build_episode_config(conditions[name], seed=seeds[0], family=args.family_obj)
        except RuntimeError as exc:
            raise SystemExit(f"condition {name}: {exc}") from None
    args.code_sha256 = _code_sha256()
    state = _git_state()
    if state["git_commit"] is None:
        print("warning: not a git checkout (or no commits); run_config.json records only code_sha256", file=sys.stderr)

    async def run_all() -> None:
        # one event loop for the whole sweep: the AsyncOpenAI client is bound to it
        backend = OpenAIChatBackend(
            args.endpoint, args.model, timeout=args.request_timeout, chat_template_kwargs=chat_template_kwargs(args.family_obj)
        )
        with ThreadPoolExecutor(max_workers=args.tool_threads) as pool:
            for seed in seeds:
                for name in names:
                    await _run_condition(name, conditions[name], rows, args, backend, pool, seed)

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
