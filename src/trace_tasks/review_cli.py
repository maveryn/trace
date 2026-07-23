"""Command-line interface for public Trace contributor review workflows."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from . import __version__


def _json_value(value: Any) -> Any:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_value(to_dict())
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(child) for child in value]
    return value


def _report_ok(value: Any) -> bool:
    """Return a report's explicit success state without hiding failures."""

    if hasattr(value, "ok"):
        return bool(value.ok)
    if hasattr(value, "passed"):
        return bool(value.passed)
    if isinstance(value, dict):
        if "ok" in value:
            return bool(value["ok"])
        if "passed" in value:
            return bool(value["passed"])
    raise TypeError(f"review report {type(value).__name__} has no success field")


def _print_json(value: Any) -> None:
    print(json.dumps(_json_value(value), indent=2, sort_keys=True, ensure_ascii=False))


def _selection_arguments(parser: argparse.ArgumentParser, *, allow_all: bool) -> None:
    parser.add_argument(
        "--task", action="append", default=[], help="Task id; repeatable"
    )
    parser.add_argument(
        "--domain", action="append", default=[], help="Public domain; repeatable"
    )
    parser.add_argument(
        "--scene",
        action="append",
        default=[],
        metavar="DOMAIN/SCENE_ID",
        help="Public scene; repeatable",
    )
    if allow_all:
        parser.add_argument(
            "--all", action="store_true", help="Select all public tasks"
        )


def _resolve_capture_tasks(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> list[str]:
    from .core.source_layout_policy import parse_public_task_id
    from .tasks.registry import list_default_task_ids

    available = list_default_task_ids()
    available_set = set(available)
    explicit = [str(value) for value in args.task]
    unknown = sorted(set(explicit) - available_set)
    if unknown:
        parser.error(f"unknown or inactive task id(s): {', '.join(unknown)}")
    if args.all and (explicit or args.domain or args.scene):
        parser.error("--all cannot be combined with task, domain, or scene filters")
    if args.all:
        return list(available)

    selected = set(explicit)
    requested_domains = {str(value) for value in args.domain}
    requested_scenes: set[tuple[str, str]] = set()
    for value in args.scene:
        domain, separator, scene_id = str(value).partition("/")
        if not separator or not domain or not scene_id:
            parser.error("--scene values must use DOMAIN/SCENE_ID")
        requested_scenes.add((domain, scene_id))

    known_domains: set[str] = set()
    known_scenes: set[tuple[str, str]] = set()
    for task_id in available:
        parts = parse_public_task_id(task_id)
        known_domains.add(parts.domain)
        known_scenes.add((parts.domain, parts.scene_id))
        if (
            parts.domain in requested_domains
            or (parts.domain, parts.scene_id) in requested_scenes
        ):
            selected.add(task_id)
    unknown_domains = sorted(requested_domains - known_domains)
    unknown_scenes = sorted(requested_scenes - known_scenes)
    if unknown_domains:
        parser.error(f"unknown public domain(s): {', '.join(unknown_domains)}")
    if unknown_scenes:
        parser.error(
            "unknown public scene(s): "
            + ", ".join(f"{domain}/{scene}" for domain, scene in unknown_scenes)
        )
    if not selected:
        parser.error("select tasks with --task, --domain, --scene, or --all")
    return sorted(selected)


def _capture(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from .review.recipe import capture_recipe

    task_ids = _resolve_capture_tasks(args, parser)
    result = capture_recipe(
        task_ids,
        Path(args.recipe),
        seed=int(args.seed),
        requests_per_task=int(args.requests_per_task),
        max_attempts=int(args.max_attempts),
        max_request_retries=int(args.max_request_retries),
        workers=int(args.workers),
        repo_root=Path(args.repo_root),
    )
    _print_json(result)
    return 0


def _materialize(args: argparse.Namespace, _parser: argparse.ArgumentParser) -> int:
    from .review.materialize import materialize_recipe

    result = materialize_recipe(
        Path(args.recipe),
        Path(args.output),
        task_ids=args.task or None,
        domains=args.domain or None,
        scene_ids=args.scene or None,
        strict_rendering=bool(args.strict_rendering),
    )
    _print_json(result)
    return 0


def _verify(args: argparse.Namespace, _parser: argparse.ArgumentParser) -> int:
    from .review.materialize import verify_recipe

    result = verify_recipe(
        Path(args.recipe),
        task_ids=args.task or None,
        domains=args.domain or None,
        scene_ids=args.scene or None,
        query_ids=args.query or None,
        strict_rendering=bool(args.strict_rendering),
    )
    _print_json(result)
    return 0 if _report_ok(result) else 1


def _audit(args: argparse.Namespace, _parser: argparse.ArgumentParser) -> int:
    from .review.audits import audit_materialized, audit_recipe

    recipe_report = audit_recipe(Path(args.recipe))
    payload: dict[str, Any] = {"recipe": recipe_report}
    passed = _report_ok(recipe_report)
    if args.output is not None:
        materialized_report = audit_materialized(
            Path(args.recipe),
            Path(args.output),
            task_ids=args.task or None,
            domains=args.domain or None,
            scene_ids=args.scene or None,
            query_ids=args.query or None,
        )
        payload["materialized"] = materialized_report
        passed = passed and _report_ok(materialized_report)
    _print_json(payload)
    return 0 if passed else 1


def _serve(args: argparse.Namespace, _parser: argparse.ArgumentParser) -> int:
    from .review.app import serve_review_app

    review_root = Path(args.review_root)
    database_path = (
        Path(args.database)
        if args.database is not None
        else review_root.parent / "feedback" / "review_feedback.sqlite"
    )
    serve_review_app(
        review_root=review_root,
        database_path=database_path,
        repo_root=Path(args.repo_root),
        host=str(args.host),
        port=int(args.port),
        auth_token_env=str(args.token_env),
        trusted_hosts=tuple(args.trusted_host),
    )
    return 0


def _export(args: argparse.Namespace, _parser: argparse.ArgumentParser) -> int:
    from .review.export import export_review_report

    review_root = Path(args.review_root)
    database_path = (
        Path(args.database)
        if args.database is not None
        else review_root.parent / "feedback" / "review_feedback.sqlite"
    )
    output = export_review_report(
        review_root=review_root,
        database_path=database_path,
        output_path=Path(args.output),
        output_format=str(args.format),
    )
    _print_json({"output": output})
    return 0


def _calibrate(args: argparse.Namespace, _parser: argparse.ArgumentParser) -> int:
    from .review.calibration import CalibrationConfig, calibrate_review_root

    config = CalibrationConfig(
        endpoint=str(args.endpoint),
        model=str(args.model),
        api_key_env=str(args.api_key_env),
        timeout_seconds=float(args.timeout),
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
        rollouts_per_sample=int(args.rollouts),
        max_retries=int(args.max_retries),
        diagnostic_accuracy_threshold=args.diagnostic_accuracy_threshold,
    )
    result = calibrate_review_root(
        review_root=Path(args.review_root),
        config=config,
        output_path=Path(args.output),
        task_ids=tuple(args.task),
        limit=args.limit,
    )
    _print_json(result)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trace-review",
        description="Generate, inspect, and verify public Trace task-review artifacts.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    capture = commands.add_parser(
        "capture", help="Freeze a deterministic review recipe"
    )
    capture.add_argument("--recipe", type=Path, required=True)
    capture.add_argument("--repo-root", type=Path, default=Path.cwd())
    capture.add_argument("--seed", type=int, default=42)
    capture.add_argument("--requests-per-task", type=int, default=25)
    capture.add_argument("--max-attempts", type=int, default=100)
    capture.add_argument("--max-request-retries", type=int, default=32)
    capture.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Bounded parallel capture workers; output ordering remains deterministic",
    )
    _selection_arguments(capture, allow_all=True)
    capture.set_defaults(handler=_capture)

    materialize = commands.add_parser("materialize", help="Regenerate recipe artifacts")
    materialize.add_argument("--recipe", type=Path, required=True)
    materialize.add_argument("--output", type=Path, default=Path("review/task-reviews"))
    materialize.add_argument("--strict-rendering", action="store_true")
    _selection_arguments(materialize, allow_all=False)
    materialize.set_defaults(handler=_materialize)

    verify = commands.add_parser("verify", help="Replay and verify recipe hashes")
    verify.add_argument("--recipe", type=Path, required=True)
    verify.add_argument("--strict-rendering", action="store_true")
    _selection_arguments(verify, allow_all=False)
    verify.add_argument(
        "--query", action="append", default=[], help="Query id; repeatable"
    )
    verify.set_defaults(handler=_verify)

    audit = commands.add_parser("audit", help="Audit a recipe and optional artifacts")
    audit.add_argument("--recipe", type=Path, required=True)
    audit.add_argument("--output", type=Path)
    _selection_arguments(audit, allow_all=False)
    audit.add_argument(
        "--query", action="append", default=[], help="Query id; repeatable"
    )
    audit.set_defaults(handler=_audit)

    serve = commands.add_parser("serve", help="Run the local contributor review app")
    serve.add_argument(
        "--review-root",
        "--root",
        dest="review_root",
        type=Path,
        default=Path("review/task-reviews"),
    )
    serve.add_argument("--database", type=Path)
    serve.add_argument("--repo-root", type=Path, default=Path.cwd())
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--token-env", default="TRACE_REVIEW_APP_TOKEN")
    serve.add_argument(
        "--trusted-host",
        action="append",
        default=[],
        help="Allowed HTTP Host value; repeat for non-loopback access",
    )
    serve.set_defaults(handler=_serve)

    export = commands.add_parser("export", help="Export portable review reports")
    export.add_argument(
        "--review-root",
        "--root",
        dest="review_root",
        type=Path,
        default=Path("review/task-reviews"),
    )
    export.add_argument("--database", type=Path)
    export.add_argument("--output", "--out", dest="output", type=Path, required=True)
    export.add_argument("--format", choices=("json", "jsonl", "xlsx"), default="json")
    export.set_defaults(handler=_export)

    calibrate = commands.add_parser(
        "calibrate", help="Probe a caller-managed model endpoint"
    )
    calibrate.add_argument(
        "--review-root",
        "--root",
        dest="review_root",
        type=Path,
        default=Path("review/task-reviews"),
    )
    calibrate.add_argument("--endpoint", required=True)
    calibrate.add_argument("--model", required=True)
    calibrate.add_argument("--api-key-env", default="TRACE_REVIEW_API_KEY")
    calibrate.add_argument("--timeout", type=float, default=60.0)
    calibrate.add_argument("--max-tokens", type=int, default=512)
    calibrate.add_argument("--temperature", type=float, default=0.0)
    calibrate.add_argument("--rollouts", type=int, default=1)
    calibrate.add_argument("--max-retries", type=int, default=2)
    calibrate.add_argument("--diagnostic-accuracy-threshold", type=float)
    calibrate.add_argument("--task", action="append", default=[])
    calibrate.add_argument("--limit", type=int)
    calibrate.add_argument("--output", "--out", dest="output", type=Path, required=True)
    calibrate.set_defaults(handler=_calibrate)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the contributor review CLI."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args, parser))
    except ModuleNotFoundError as exc:
        print(
            "trace-review requires the optional review dependencies; "
            "install trace-tasks[review] "
            f"(missing module: {exc.name})",
            file=sys.stderr,
        )
        return 1
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"trace-review failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
