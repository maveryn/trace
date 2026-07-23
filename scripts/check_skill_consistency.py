#!/usr/bin/env python3
"""Validate public repo-local Codex skills and their documentation routing."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import yaml

EXPECTED_SKILLS = {
    "trace-code-review",
    "trace-prompt-design",
    "trace-task-design",
    "trace-task-implementation",
    "trace-task-unit-audit",
    "trace-verification-review",
}

NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
FRONTMATTER_RE = re.compile(r"\A---\n(?P<yaml>.*?)\n---\n(?P<body>.*)\Z", re.DOTALL)
SKILL_REF_RE = re.compile(r"\$(trace-[a-z0-9-]+)")
DOC_REF_RE = re.compile(r"docs/[A-Za-z0-9_./<>-]+\.md")
CONCRETE_TASK_ID_RE = re.compile(r"\btask_[a-z0-9]+__[a-z0-9_]+__[a-z0-9_]+\b")

FORBIDDEN_TEXT = (
    (re.compile(r"(?:^|[\s`(])/(?:home|Users)/"), "machine-specific absolute path"),
    (re.compile(r"[A-Za-z]:\\"), "machine-specific Windows path"),
    (
        re.compile(
            r"(?<![A-Za-z0-9_.-])skills/(?:code-review|prompt-design|task-design|"
            r"task-implementation|task-unit-audit|verification-review)"
        ),
        "legacy skill path",
    ),
    (re.compile(r"docs/TODO\.md"), "internal TODO reference"),
    (
        re.compile(r"(?<!src/)trace/(?:core|tasks|review_app)(?:/|\b)"),
        "legacy source-layout path",
    ),
    (re.compile(r"(?:^|[\s`])rlvr/"), "paper/RLVR campaign path"),
    (re.compile(r"\b(?:Qwen/|qwen(?:2|25|3))", re.IGNORECASE), "fixed model reference"),
)


def _frontmatter(skill_md: Path) -> tuple[dict[str, object] | None, str, list[str]]:
    errors: list[str] = []
    text = skill_md.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.fullmatch(text)
    if match is None:
        return None, text, ["SKILL.md must have LF-delimited YAML frontmatter"]
    try:
        value = yaml.safe_load(match.group("yaml"))
    except yaml.YAMLError as exc:
        return None, text, [f"invalid SKILL.md frontmatter: {exc}"]
    if not isinstance(value, dict):
        return None, text, ["SKILL.md frontmatter must be a mapping"]
    return value, text, errors


def _validate_interface(path: Path, skill_name: str) -> list[str]:
    if not path.is_file():
        return ["agents/openai.yaml is missing"]

    text = path.read_text(encoding="utf-8")
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return [f"invalid agents/openai.yaml: {exc}"]
    if not isinstance(value, dict) or set(value) != {"interface"}:
        return ["agents/openai.yaml must contain only the interface mapping"]
    interface = value.get("interface")
    required = {"display_name", "short_description", "default_prompt"}
    if not isinstance(interface, dict) or set(interface) != required:
        return [
            "agents/openai.yaml interface must contain exactly display_name, "
            "short_description, and default_prompt"
        ]

    errors: list[str] = []
    for key in sorted(required):
        item = interface.get(key)
        if not isinstance(item, str) or not item.strip():
            errors.append(f"interface.{key} must be a non-empty string")
        if not re.search(rf"^\s{{2}}{key}:\s+\".*\"\s*$", text, re.MULTILINE):
            errors.append(f"interface.{key} must be double-quoted")

    short_description = interface.get("short_description")
    if isinstance(short_description, str) and not 25 <= len(short_description) <= 64:
        errors.append("interface.short_description must contain 25-64 characters")
    default_prompt = interface.get("default_prompt")
    if isinstance(default_prompt, str) and f"${skill_name}" not in default_prompt:
        errors.append(f"interface.default_prompt must mention ${skill_name}")
    return errors


def validate_skill(
    skill_dir: Path,
    *,
    repo_root: Path,
    known_skills: set[str],
) -> list[str]:
    """Return validation errors for one skill directory."""

    errors: list[str] = []
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        return ["SKILL.md is missing"]

    frontmatter, text, parse_errors = _frontmatter(skill_md)
    errors.extend(parse_errors)
    if frontmatter is not None:
        if set(frontmatter) != {"name", "description"}:
            errors.append(
                "SKILL.md frontmatter must contain exactly name and description"
            )
        name = frontmatter.get("name")
        if not isinstance(name, str) or not NAME_RE.fullmatch(name):
            errors.append("frontmatter name must be lowercase hyphen-case")
        elif name != skill_dir.name:
            errors.append(
                f"frontmatter name {name!r} does not match folder {skill_dir.name!r}"
            )
        description = frontmatter.get("description")
        if not isinstance(description, str) or not description.strip():
            errors.append("frontmatter description must be a non-empty string")
        elif "use " not in description.lower():
            errors.append("frontmatter description must explain when to use the skill")

    if len(text.splitlines()) > 500:
        errors.append("SKILL.md must remain at or below 500 lines")
    if CONCRETE_TASK_ID_RE.search(text):
        errors.append("SKILL.md must not embed concrete public task ids")
    for pattern, label in FORBIDDEN_TEXT:
        if pattern.search(text):
            errors.append(f"SKILL.md contains a {label}")

    for referenced_name in sorted(set(SKILL_REF_RE.findall(text))):
        if referenced_name not in known_skills:
            errors.append(f"unknown cross-skill reference ${referenced_name}")

    for referenced_doc in sorted(set(DOC_REF_RE.findall(text))):
        if "<" in referenced_doc or ">" in referenced_doc:
            continue
        if not (repo_root / referenced_doc).is_file():
            errors.append(f"referenced document does not exist: {referenced_doc}")

    errors.extend(
        _validate_interface(skill_dir / "agents" / "openai.yaml", skill_dir.name)
    )
    return errors


def validate_repository(repo_root: Path) -> list[str]:
    """Return all repo-local skill consistency errors."""

    repo_root = repo_root.resolve()
    skills_root = repo_root / ".agents" / "skills"
    if not skills_root.is_dir():
        return [".agents/skills is missing"]

    skill_dirs = sorted(path for path in skills_root.iterdir() if path.is_dir())
    actual = {path.name for path in skill_dirs}
    errors: list[str] = []
    for missing in sorted(EXPECTED_SKILLS - actual):
        errors.append(f"missing required skill: {missing}")
    for unexpected in sorted(actual - EXPECTED_SKILLS):
        errors.append(f"unexpected repo-local skill: {unexpected}")

    for skill_dir in skill_dirs:
        for error in validate_skill(
            skill_dir, repo_root=repo_root, known_skills=actual
        ):
            errors.append(f"{skill_dir.relative_to(repo_root)}: {error}")
    return errors


def _print_errors(errors: Iterable[str]) -> int:
    errors = list(errors)
    if not errors:
        print("skill consistency check passed")
        return 0
    print("skill consistency check failed:")
    for error in errors:
        print(f"- {error}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Trace checkout root (defaults to this script's checkout)",
    )
    args = parser.parse_args()
    return _print_errors(validate_repository(args.repo_root))


if __name__ == "__main__":
    raise SystemExit(main())
