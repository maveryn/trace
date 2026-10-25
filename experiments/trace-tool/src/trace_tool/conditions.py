"""Load evaluation conditions from configs/conditions.yaml."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from .agent import DECODING_PRESETS, DEFAULT_ANSWER_TOKEN_RESERVE, EpisodeConfig
from .models import ModelFamily, chat_template_kwargs
from .prompting import FORCED_PROMPT_SHA256
from .tools import TOOL_GROUPS, build_tools

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "conditions.yaml"

_EPISODE_KEYS = {
    "max_turns",
    "max_tokens_per_turn",
    "max_total_tokens",
    "max_calls_per_turn",
    "decoding",
    "min_pixels",
    "max_pixels",
    "oracle_crop",
}
# Forced conditions only: allowed on a condition, never in `defaults`, and recorded by
# describe() only for conditions that enable them (other conditions' settings are unchanged).
_FORCED_KEYS = ("forced_answer", "tool_call_budget", "answer_token_reserve")


def load_conditions(path: str | Path = DEFAULT_CONFIG) -> dict[str, dict[str, Any]]:
    raw = yaml.safe_load(Path(path).read_text())
    defaults = raw.get("defaults", {})
    in_defaults = sorted(set(defaults) & set(_FORCED_KEYS))
    if in_defaults:
        raise ValueError(f"{in_defaults} cannot be defaults; set them on the forced conditions")
    out = {}
    for name, overrides in (raw.get("conditions") or {}).items():
        merged = {**defaults, **(overrides or {})}
        unknown = set(merged) - _EPISODE_KEYS - {"tools", "system_prompt_files"} - set(_FORCED_KEYS)
        if unknown:
            raise ValueError(f"condition {name!r}: unknown keys {sorted(unknown)}")
        if merged["decoding"] not in DECODING_PRESETS:
            raise ValueError(f"condition {name!r}: unknown decoding preset {merged['decoding']!r}")
        _check_forced(name, merged)
        out[name] = merged
    return out


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_forced(name: str, spec: dict[str, Any]) -> None:
    keys = [k for k in _FORCED_KEYS if k in spec]
    if not keys:
        return
    if not isinstance(spec.get("forced_answer", False), bool):
        raise ValueError(f"condition {name!r}: forced_answer must be true or false")
    if not spec.get("tools"):
        raise ValueError(f"condition {name!r}: {keys} need a non-empty tools list")
    if not spec.get("forced_answer"):
        if keys != ["forced_answer"]:
            raise ValueError(f"condition {name!r}: {[k for k in keys if k != 'forced_answer']} need forced_answer: true")
        return
    if spec["max_turns"] < 2:
        raise ValueError(f"condition {name!r}: forced_answer needs max_turns >= 2 (tool turns, then the answer turn)")
    per_turn = spec["max_calls_per_turn"]
    if not _is_int(per_turn) or per_turn < 1:  # also with an explicit tool_call_budget
        raise ValueError(f"condition {name!r}: forced_answer needs an integer max_calls_per_turn >= 1, got {per_turn!r}")
    budget = spec.get("tool_call_budget", (spec["max_turns"] - 1) * spec["max_calls_per_turn"])
    if not _is_int(budget) or budget < 1:
        raise ValueError(f"condition {name!r}: tool_call_budget must be an integer >= 1, got {budget!r}")
    reserve = spec.get("answer_token_reserve", DEFAULT_ANSWER_TOKEN_RESERVE)
    if not _is_int(reserve) or not 1 <= reserve <= spec["max_tokens_per_turn"]:
        raise ValueError(f"condition {name!r}: answer_token_reserve must be in [1, max_tokens_per_turn], got {reserve!r}")
    if reserve >= spec["max_total_tokens"]:
        raise ValueError(f"condition {name!r}: answer_token_reserve must be below max_total_tokens (the first turn needs tokens)")


TRACE_PROMPT_FILE = "prompts/trace_system_prompt.txt"
TRACE_PROMPT_FILE_SHA256 = "f394927d9abcfb7a1e43ef48a30c29b8c70e6facdbda314d7b27c59d8c3ae900"


def read_system_prompt(files: list[str], root: Path = REPO_ROOT) -> str:
    """Each file is stripped, as TRACE's generator and EasyR1 dataset do."""
    parts = []
    for name in files:
        raw = (root / name).read_bytes()
        if name == TRACE_PROMPT_FILE and hashlib.sha256(raw).hexdigest() != TRACE_PROMPT_FILE_SHA256:
            raise RuntimeError(f"{name} differs from TRACE's pinned system prompt")
        parts.append(raw.decode("utf-8").strip())
    return "\n\n".join(parts)


def build_episode_config(spec: dict[str, Any], *, seed: int, family: ModelFamily, root: Path = REPO_ROOT) -> EpisodeConfig:
    tools = build_tools(list(spec.get("tools") or []), family)
    return EpisodeConfig(
        system_prompt=read_system_prompt(list(spec["system_prompt_files"]), root),
        family=family,
        tools=tools,
        seed=seed,
        **{k: spec[k] for k in _EPISODE_KEYS},
        **{k: spec[k] for k in _FORCED_KEYS if k in spec},
    )


def describe(spec: dict[str, Any], cfg: EpisodeConfig) -> dict[str, Any]:
    """Settings recorded in run_config.json (and compared on resume)."""
    return {
        "family": cfg.family.name,
        "patch_factor": cfg.family.patch_factor,
        "coords": cfg.family.coords,
        **_family_settings(cfg.family),
        **{k: spec[k] for k in sorted(_EPISODE_KEYS)},
        "tools": sorted(cfg.tools),
        "tool_groups_requested": list(spec.get("tools") or []),
        "system_prompt_files": list(spec["system_prompt_files"]),
        "system_prompt_sha256": hashlib.sha256(cfg.system_prompt.encode()).hexdigest(),
        "tool_schemas": [tool.schema() for tool in cfg.tools.values()],
        "decoding_params": DECODING_PRESETS[spec["decoding"]],
        **_forced_settings(cfg),
        **_ocr_settings(cfg),
    }


def _family_settings(family: ModelFamily) -> dict[str, Any]:
    """Only for families that differ from the defaults (Hermes calls, no thinking switch)."""
    if family.tool_format == "hermes" and not family.thinking_off:
        return {}
    return {"tool_format": family.tool_format, "chat_template_kwargs": chat_template_kwargs(family)}


def _forced_settings(cfg: EpisodeConfig) -> dict[str, Any]:
    """Only for forced conditions; the prompt hash covers the budget statement and both notices."""
    if not cfg.forced_answer:
        return {}
    return {
        "forced_answer": True,
        "tool_call_budget": cfg.call_budget,
        "answer_token_reserve": cfg.answer_token_reserve,
        "forced_prompt_sha256": FORCED_PROMPT_SHA256,
    }


def _ocr_settings(cfg: EpisodeConfig) -> dict[str, Any]:
    """OCR output depends on the backend and its package versions, so they are resume keys."""
    tool = cfg.tools.get("ocr")
    if tool is None:
        return {}
    from .tools.ocr import backend_versions

    return {"ocr_backend": tool.backend_name, "ocr_min_score": tool.min_score, "ocr_versions": backend_versions(tool.backend_name)}


__all__ = ["DEFAULT_CONFIG", "TOOL_GROUPS", "build_episode_config", "describe", "load_conditions", "read_system_prompt"]
