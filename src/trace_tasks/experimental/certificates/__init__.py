"""Experimental program-verifiable reasoning certificates for Trace (opt-in extension).

A certificate response is ``{"annotation": ..., "certificate": {"kind": ...}, "answer": ...}``.
The scorer checks the certificate against private oracle data built from the generated scene;
legacy datasets, response modes, reward ids and release receipts are untouched.

Plan API (DESIGN 1.1): ``score_certificate_response`` lives here (``scoring``);
``build_extension``/``make_sft_target`` live in ``export`` and ``render_public_prompt`` in
``prompts`` (imported lazily by their callers).
"""

from __future__ import annotations

from .identity import CONTAINMENT_THRESHOLD, IOU_THRESHOLD, IdentityParams
from .registry import Unsupported, adapter_by_id, adapter_for_extension, resolve
from .scoring import reward_arms, score_certificate_response
from .types import (
    SCORER_VERSION,
    AdapterSpec,
    CertificateExtension,
    CertificateVersionError,
    CheckResult,
    Corruption,
    ExtensionMismatchError,
    PublicContract,
    QuarantineError,
    ScoreResult,
    TraceRecord,
    certificate_extension_id,
    read_trace_records,
    scene_group,
)

__all__ = [
    "AdapterSpec",
    "CONTAINMENT_THRESHOLD",
    "CertificateExtension",
    "CertificateVersionError",
    "CheckResult",
    "Corruption",
    "ExtensionMismatchError",
    "IOU_THRESHOLD",
    "IdentityParams",
    "PublicContract",
    "QuarantineError",
    "SCORER_VERSION",
    "ScoreResult",
    "TraceRecord",
    "Unsupported",
    "adapter_by_id",
    "adapter_for_extension",
    "certificate_extension_id",
    "read_trace_records",
    "resolve",
    "reward_arms",
    "scene_group",
    "score_certificate_response",
]
