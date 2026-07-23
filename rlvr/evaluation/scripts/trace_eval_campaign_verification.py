#!/usr/bin/env python3
"""Shared generation-verification helpers for canonical trace_eval_v1 runs.

The generation contract keeps the producer's serialized ``final25_code_hash``
field name because changing it would change the frozen response hash. Public
CLI names and benchmark selection are trace_eval_v1-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_eval_media_contract import (
    GENERATION_CONTRACT_VERSION,
    MEDIA_CONTRACT_VERSION,
    MEDIA_TRANSPORT,
    QWEN_MAX_IMAGE_PIXELS,
    QWEN_MIN_IMAGE_PIXELS,
)
from run_external_benchmark_generation_api_queue import _generation_contract_hash


def _parse_model_entries(values: list[str]) -> dict[str, tuple[str, str]]:
    entries: dict[str, tuple[str, str]] = {}
    for value in values:
        parts = value.split("=", 2)
        if len(parts) != 3 or not all(part.strip() for part in parts):
            raise ValueError(
                f"invalid --model-entry {value!r}; expected SLUG=MODEL=REVISION"
            )
        slug, model, revision = (part.strip() for part in parts)
        if slug in entries:
            raise ValueError(f"duplicate --model-entry for {slug!r}")
        entries[slug] = (model, revision)
    return entries


def _expected_generation_contract_hash(
    *,
    model: str,
    model_slug: str,
    model_revision: str,
    seed: int,
    dataset_snapshot_sha256: str,
    dataset_revision: str,
    final25_code_hash: str,
    max_tokens: int = 4096,
) -> str:
    """Rebuild the immutable producer hash used by the canonical reruns."""

    contract_args = argparse.Namespace(
        model=model,
        model_slug=model_slug,
        api_model=model_slug,
        temperature=0.6,
        top_p=1.0,
        top_k=-1,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        max_tokens=max_tokens,
        seed=seed,
        media_transport=MEDIA_TRANSPORT,
        min_image_pixels=QWEN_MIN_IMAGE_PIXELS,
        max_image_pixels=QWEN_MAX_IMAGE_PIXELS,
        max_image_side=1280,
        image_jpeg_quality=85,
        dataset_snapshot_sha256=dataset_snapshot_sha256,
        subset_root=None,
        limit=None,
        sample_seed=0,
    )
    return _generation_contract_hash(
        contract_args,
        model_revision=model_revision,
        dataset_revision=dataset_revision,
        final25_code_hash=final25_code_hash,
    )


def _generation_complete(
    path: Path,
    seed: int,
    dataset_revision: str,
    *,
    expected_model_revision: str,
    expected_contract_hash: str,
    expected_max_tokens: int = 4096,
) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        generation = payload["generation"]
        rows = int(payload["rows"])
        expected = int(payload["expected_rows"])
        selection = generation.get("selection") or {}
        checks = {
            "rows": rows > 0 and rows == expected,
            "seed": int(generation["seed"]) == seed,
            "temperature": abs(float(generation["temperature"]) - 0.6) < 1e-9,
            "top_p": abs(float(generation["top_p"]) - 1.0) < 1e-9,
            "top_k": int(generation["top_k"]) == -1,
            "presence_penalty": abs(float(generation["presence_penalty"])) < 1e-9,
            "repetition_penalty": abs(float(generation["repetition_penalty"]) - 1.0)
            < 1e-9,
            "max_tokens": int(generation["max_tokens"]) == expected_max_tokens,
            "compact_tables": bool(generation.get("compact_prediction_tables")),
            "contract_version": (
                generation.get("contract_version") == GENERATION_CONTRACT_VERSION
            ),
            "contract_hash": generation.get("contract_hash") == expected_contract_hash,
            "media_contract": (
                generation.get("media_contract_version") == MEDIA_CONTRACT_VERSION
            ),
            "media_transport": generation.get("media_transport") == MEDIA_TRANSPORT,
            "min_pixels": int(generation.get("min_image_pixels", 0))
            == QWEN_MIN_IMAGE_PIXELS,
            "max_pixels": int(generation.get("max_image_pixels", 0))
            == QWEN_MAX_IMAGE_PIXELS,
            "dataset_snapshot": bool(generation.get("dataset_snapshot_sha256")),
            "dataset_revision": generation.get("dataset_revision") == dataset_revision,
            "model_revision": generation.get("model_revision")
            == expected_model_revision,
            "full_selection": (
                selection.get("mode") == "full"
                and selection.get("limit") is None
                and selection.get("sample_seed") is None
                and selection.get("subset_manifest_sha256") is None
                and payload.get("subset_manifest") is None
            ),
        }
        failed = [name for name, passed in checks.items() if not passed]
        detail = f"{rows}/{expected}" + (
            f" mismatch={','.join(failed)}" if failed else ""
        )
        return not failed, detail
    except Exception as error:
        return False, f"invalid:{type(error).__name__}"
