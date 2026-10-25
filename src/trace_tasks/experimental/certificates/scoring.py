"""Certificate scorer: A, G, M, F, reward arms and diagnostics (DESIGN 2, 5).

- **A**: legacy ``_score_trace_answer`` (TRACE's exact_json) on the scored object's
  ``answer``, so A, G and M read one object (review EX-4); without a scored object, on
  ``extract_trace_prediction`` (bare-text fallback included). Guards, applied to the object A
  was read from (review EX-7): a boolean answer for a numeric type scores 0, and so do
  conflicting duplicate ``answer`` keys. ``A_legacy`` (TRACE's own reading) is logged; the
  model's answer is never replaced.
- **G**: legacy ``_score_trace_annotation`` on the scored object's annotation (legacy tolerant
  extraction when there is no scored object), converted to source pixels first.
- **M**: the adapter's truth families on the scored object; envelope failures give M=0.
- **F**: strict schema of the original response (``parsing.format_check``), including a strict
  annotation structure check against the public schema (review CONF-02).

Missing or mismatched extensions and versions raise; they are never model zeros. Any model
output scores without raising: numbers that do not fit a float are invalid geometry, and an
exception of the legacy answer/annotation helpers on model data is a scored zero with a guard
(review EX-1, CONTRACT-1/2/3). Scores do not depend on wall-clock time: ``verifier_ms`` and
``slow_row`` are diagnostics only.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import time
import warnings
from typing import Any, Dict, List, Mapping, Optional, Tuple

from trace_tasks.core import reward_scoring as _legacy
from trace_tasks.core.reward_scoring import (
    _score_trace_annotation,
    _score_trace_answer,
    extract_trace_prediction,
)

from . import registry
from .adapters.base import Adapter, PublicContext
from .checks import envelope_failure_checks, family_results, read_int, semantics_score
from .geometry import (
    AnnotationView,
    Frame,
    annotation_view,
    sanitize_numbers,
    scale_annotation,
    shown_frame,
    shown_size,
    source_frame,
)
from .identity import DEFAULT_IDENTITY, IdentityParams
from .parsing import (
    CertificateView,
    LocatedObject,
    annotation_schema_error,
    answer_duplicate_conflict,
    check_envelope,
    format_check,
    legacy_answer_source,
    locate_scored_object,
    text_answer_conflict,
)
from .types import (
    COORD_FRAMES,
    SCORER_VERSION,
    CertificateExtension,
    CertificateVersionError,
    CheckResult,
    ExtensionMismatchError,
    ScoreResult,
)

ALPHA = 0.80
FORMAT_WEIGHT = 0.05
ARMS = ("A", "G", "P", "P_gated", "G_gated")
SLOW_ROW_MS = 50.0
_NUMERIC_ANSWER_TYPES = ("integer", "number")
_POINT_HALF_LIFE = (0.035, 20.0, 80.0)  # legacy fraction of the image diagonal, min, max


def require_scipy_assignment() -> None:
    """Legacy G silently falls back to greedy matching without scipy; refuse to score then."""
    if getattr(_legacy, "linear_sum_assignment", None) is None:
        raise RuntimeError("scipy.optimize.linear_sum_assignment is not importable; legacy G would be greedy")


def reward_arms(
    answer: float,
    grounding: float,
    semantics: float,
    fmt: float,
    *,
    alpha: float = ALPHA,
    format_weight: float = FORMAT_WEIGHT,
) -> Dict[str, float]:
    """``R = (1-f)[aA + (1-a)B] + fF`` per arm; gated arms use ``(1-f)A[a + (1-a)B] + fF``."""
    for name, value in (("alpha", alpha), ("format_weight", format_weight)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a real number (not a bool or string), got {value!r}")
    if not 0.0 <= alpha <= 1.0 or not 0.0 <= format_weight <= 1.0:
        raise ValueError(f"alpha and format_weight must be in [0, 1], got {alpha}, {format_weight}")
    f, a = float(format_weight), float(alpha)
    b_p = (grounding + semantics) / 2.0
    out = {
        "A": (1 - f) * answer + f * fmt,
        "G": (1 - f) * (a * answer + (1 - a) * grounding) + f * fmt,
        "P": (1 - f) * (a * answer + (1 - a) * b_p) + f * fmt,
        "P_gated": (1 - f) * answer * (a + (1 - a) * b_p) + f * fmt,
        "G_gated": (1 - f) * answer * (a + (1 - a) * grounding) + f * fmt,
    }
    return {arm: _unit(f"overall.{arm}", value) for arm, value in out.items()}


def _unit(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < -1e-12 or value > 1.0 + 1e-12:
        raise ValueError(f"{name} must be finite in [0, 1], got {value!r}")
    return min(1.0, max(0.0, value))


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    return isinstance(value, str) and value.strip().lower() in ("true", "false")


def legacy_prediction(response: str) -> Tuple[Any, Any, bool]:
    """``extract_trace_prediction`` with the SyntaxWarnings of its literal_eval fallback muted."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return extract_trace_prediction(response)


# Exceptions TRACE's legacy answer/annotation helpers raise on model-producible values (for
# example a Python-literal ``{1j}`` answer, or a number too large for a float): scored zeros.
_LEGACY_MODEL_ERRORS = (RecursionError, OverflowError, TypeError, ValueError)


def _legacy_answer_score(answer_value: Any, answer_gt: Mapping[str, Any]) -> Optional[float]:
    """TRACE's exact_json answer score, or None when the legacy helper raises on the value."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            legacy, _ = _score_trace_answer(answer_value, dict(answer_gt))
    except _LEGACY_MODEL_ERRORS:
        return None
    return float(legacy)


def score_answer(
    response: str,
    answer_gt: Mapping[str, Any],
    located: Optional[LocatedObject] = None,
    *,
    use_scored_object: bool = True,
) -> Tuple[float, float, Optional[str]]:
    """``(A, A_legacy, guard)``: exact_json answer score plus the DESIGN 5 guards.

    With ``use_scored_object`` (certificate conditions), A reads the ``answer`` of the scored
    object, the same object G and M read (review EX-4); when there is no scored object, and
    always for the legacy ``ans``/``ann`` conditions, A is TRACE's ``extract_trace_prediction``
    reading. The guards apply to whichever object A was read from (review EX-7). ``A_legacy``
    is TRACE's unguarded reading. ``located`` only avoids re-locating the objects.
    """
    numeric = str(answer_gt.get("type", "")).strip() in _NUMERIC_ANSWER_TYPES
    legacy_value, _, _ = legacy_prediction(response)
    if legacy_value is None:
        legacy_value = response
    legacy_score = _legacy_answer_score(legacy_value, answer_gt)
    answer_legacy = 0.0 if legacy_score is None else legacy_score

    if use_scored_object:
        if located is None:
            located = locate_scored_object(response)
        scored = located.scored
        if scored is not None:
            if "answer" not in scored:
                return 0.0, answer_legacy, "missing_answer"
            value = scored["answer"]
            score = _legacy_answer_score(value, answer_gt)
            if score is None:
                return 0.0, answer_legacy, "unscorable_answer"
            guard: Optional[str] = None
            if numeric and _boolish(value):
                guard = "bool_answer"
            elif answer_duplicate_conflict(scored):
                guard = "duplicate_answer"
            return (0.0 if guard else score), answer_legacy, guard

    if legacy_score is None:
        # e.g. a pathologically nested answer, or a Python-literal set holding a complex
        # number: the legacy reward would crash too; it cannot equal a scalar answer_gt.
        return 0.0, answer_legacy, "unscorable_answer"
    guard = None
    if numeric and _boolish(legacy_value):
        guard = "bool_answer"
    elif text_answer_conflict(legacy_answer_source(response)):
        guard = "duplicate_answer"
    return (0.0 if guard else legacy_score), answer_legacy, guard


def point_half_life_px(image_size: Tuple[int, int]) -> float:
    """Legacy point half-life for a known image size (unused by box contracts)."""
    fraction, low, high = _POINT_HALF_LIFE
    return float(min(max(fraction * math.hypot(*image_size), low), high))


def score_grounding(
    annotation_value: Any,
    labels: Mapping[str, Any],
    annotation_type: str,
    frame: Frame,
    image_size: Tuple[int, int],
) -> Tuple[float, bool, Dict[str, Any]]:
    """Legacy G on the annotation converted to source pixels: ``(G, parse_ok, details)``.

    Numbers that do not fit a float are replaced by None first, so the legacy normalizer
    rejects that box instead of raising; any other exception of the legacy helper on model
    data is a scored G=0 with ``reason = unscorable_annotation:<type>`` (review EX-1).
    """
    if annotation_value is None:
        return 0.0, False, {"reason": "missing_annotation"}
    converted = sanitize_numbers(scale_annotation(annotation_value, annotation_type, frame))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            score, parse_ok, details = _score_trace_annotation(
                converted,
                dict(labels["annotation_gt"]),
                dict(labels["reward_contract"]),
                point_half_life_px=point_half_life_px(image_size),
            )
    except _LEGACY_MODEL_ERRORS as exc:
        return 0.0, False, {"reason": f"unscorable_annotation:{type(exc).__name__}"}
    return float(score), bool(parse_ok), dict(details)


def _semantics(
    adapter: Adapter,
    located: LocatedObject,
    oracle: Mapping[str, Any],
    ctx: PublicContext,
    applicable: Mapping[str, bool],
    annotation: Optional[AnnotationView] = None,
) -> Tuple[List[CheckResult], float, Optional[CertificateView], Optional[str], Dict[str, Any]]:
    """``(checks, M, certificate view, envelope error, adapter diagnostics)``.

    ``annotation`` is the scored object's annotation in ``ctx.frame`` (built here when None).
    An annotation role longer than the adapter's cap is a deterministic ``budget`` envelope
    failure (one element per box, so it can never be valid; review CONTRACT-5).
    """
    error = located.envelope_error if located.scored is None else check_envelope(located.scored)
    if error is not None:
        return envelope_failure_checks(adapter.m_families, error, applicable), 0.0, None, error, {}
    obj = located.scored
    if annotation is None:
        annotation = annotation_view(obj.get("annotation"), adapter.annotation_type, ctx.frame)
    if adapter.annotation_over_cap(annotation):
        return envelope_failure_checks(adapter.m_families, "budget", applicable), 0.0, None, "budget", {}
    view = adapter.validate_certificate(obj["certificate"], adapter.annotation_size(annotation))
    view = dataclasses.replace(view, answer=read_int(obj.get("answer")).value)
    if view.envelope_error is not None:
        failed = envelope_failure_checks(adapter.m_families, view.envelope_error, applicable)
        return failed, 0.0, view, view.envelope_error, {}
    checks = list(adapter.checks(view, annotation, oracle, ctx))
    families = family_results(checks, adapter.m_families)
    for family, result in families.items():
        if bool(result.applicable) != bool(applicable.get(family, True)):
            raise ValueError(f"{adapter.adapter_id}: applicability of {family} must be oracle-only")
    diagnostics = adapter.diagnostics(view, annotation, oracle, ctx, checks)
    return checks, semantics_score(checks, adapter.m_families), view, None, dict(diagnostics)


def check_prompt(prompt: Any, extension: CertificateExtension, coord_frame: str, where: str) -> bool:
    """Integrity only (never a score input): the row's public prompt must be the one the
    extension froze for ``coord_frame``. Returns False when there is no prompt to check."""
    if prompt is None:
        return False
    if isinstance(prompt, (bytes, bytearray)):
        prompt = prompt.decode("utf-8")
    if not isinstance(prompt, str):
        raise CertificateVersionError(f"{where}: prompt must be a string, got {type(prompt).__name__}")
    frozen = (extension.public.get("prompt_sha256") or {}).get(coord_frame)
    found = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if frozen is None or found != frozen:
        matches = [
            frame for frame, sha in sorted((extension.public.get("prompt_sha256") or {}).items()) if sha == found
        ]
        raise CertificateVersionError(
            f"{where}: prompt sha256 {found[:12]} is not the {coord_frame} prompt of "
            f"{extension.certificate_extension_id} (matches {matches or 'no frozen prompt'}); "
            "the reward's coord_frame must be the frame the row's prompt declared"
        )
    return True


# The public rows' prompt columns: ``prompt_cert`` (RL parquet) and ``prompt_answer``
# (trace-tool parquets). They are always checked against the extension's frozen prompts.
CERT_PROMPT_FIELDS = ("prompt_cert", "prompt_answer")
# TRACE's own base row (the TrainInstance the plan API passes to ``build_extension``) carries
# TRACE's legacy prompt under ``prompt``. It is a public prompt of the extension only when its
# sha256 is one the extension froze; otherwise it states no certificate frame (review VER-2).
TRACE_PROMPT_FIELD = "prompt"
PROMPT_FIELDS = CERT_PROMPT_FIELDS + (TRACE_PROMPT_FIELD,)


def _row_prompt(base: Mapping[str, Any], extension: CertificateExtension) -> Optional[str]:
    """The row's public prompt to check, or None (see ``CERT_PROMPT_FIELDS``/``TRACE_PROMPT_FIELD``)."""
    for key in CERT_PROMPT_FIELDS:
        if isinstance(base.get(key), str):
            return base[key]
    prompt = base.get(TRACE_PROMPT_FIELD)
    if not isinstance(prompt, str):
        return None
    found = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    frozen = (extension.public.get("prompt_sha256") or {}).values()
    return prompt if found in frozen else None


def _frame_of_base(base: Mapping[str, Any], extension: CertificateExtension, coord_frame: Optional[str]) -> str:
    """The contract frame of the row (plan API, review CONF-09, VER-2), never guessed:

    1. ``coord_frame`` when given;
    2. else the row's own ``coord_frame`` field;
    3. else the frame whose frozen prompt sha256 the row's public prompt has (``prompt_cert``
       / ``prompt_answer``, or a ``prompt`` that is one of the extension's frozen prompts);
    4. else the extension's public contract, when it froze exactly one certificate frame.

    TRACE's own base row is accepted: its ``prompt`` (TRACE's legacy prompt) is no certificate
    prompt, so its frame comes from 1, 2 or 4; with both frames frozen and none given, a
    ValueError asks for ``coord_frame``. A frame from 1 or 2 is checked against the row's public prompt
    when it carries one (review EX-6).
    """
    prompt = _row_prompt(base, extension)
    if coord_frame is None:
        coord_frame = base.get("coord_frame") if isinstance(base.get("coord_frame"), str) else None
    if coord_frame is None and prompt is not None:
        found = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        frames = [f for f in COORD_FRAMES if (extension.public.get("prompt_sha256") or {}).get(f) == found]
        if len(frames) != 1:
            raise CertificateVersionError(
                f"the row's prompt matches no frozen certificate prompt of {extension.certificate_extension_id}"
            )
        coord_frame = frames[0]
    frozen_frames = [f for f in COORD_FRAMES if f in (extension.public.get("prompt_sha256") or {})]
    if coord_frame is None and len(frozen_frames) == 1:
        coord_frame = frozen_frames[0]
    if coord_frame is None:
        raise ValueError(
            "coord_frame is required unless the base row carries its public certificate prompt "
            "(prompt_cert/prompt_answer) or a coord_frame field, or the extension froze a single frame "
            f"(it froze {frozen_frames or 'none'})"
        )
    if coord_frame not in COORD_FRAMES:
        raise ValueError(f"coord_frame must be one of {COORD_FRAMES}, got {coord_frame!r}")
    if prompt is not None:
        check_prompt(prompt, extension, coord_frame, "base row")
    return coord_frame


def _check_base(base: Mapping[str, Any], extension: CertificateExtension) -> None:
    if not isinstance(base, Mapping) or "instance_id" not in base:
        raise ExtensionMismatchError("base row must carry instance_id")
    if str(base["instance_id"]) != extension.instance_id:
        raise ExtensionMismatchError(
            f"row instance_id {base['instance_id']} != extension instance_id {extension.instance_id}"
        )
    for key in ("cert_extension_id", "certificate_extension_id"):
        if key in base and base[key] is not None and str(base[key]) != extension.certificate_extension_id:
            raise ExtensionMismatchError(f"row {key} {base[key]} != {extension.certificate_extension_id}")


def score_certificate_response(
    response: str,
    base: Mapping[str, Any],
    extension: CertificateExtension | Mapping[str, Any],
    *,
    coord_frame: Optional[str] = None,
    adapter: Optional[Adapter] = None,
    shown_frame_diagnostics: bool = False,
    identity: IdentityParams = DEFAULT_IDENTITY,
    alpha: float = ALPHA,
    format_weight: float = FORMAT_WEIGHT,
) -> ScoreResult:
    """Score one response against its private extension (plan API, DESIGN 5).

    ``base`` is the public row or TRACE's own base row (needs ``instance_id``;
    ``cert_extension_id`` is checked when present). ``coord_frame`` is the frame the row's
    prompt declared; when omitted it is read from the row (its ``coord_frame`` field or its
    public prompt's frozen sha256) or from an extension that froze one frame, never guessed
    (:func:`_frame_of_base`), and a given frame is checked against the row's public prompt.
    TRACE's legacy ``prompt`` (a base row's) states no frame. ``shown_frame_diagnostics`` adds
    ``G_shown``/``M_shown`` for Qwen2.5-VL rows (px_source prompts only).
    """
    started = time.perf_counter()
    if extension is None:
        raise CertificateVersionError("missing certificate extension")
    if not isinstance(extension, CertificateExtension):
        extension = CertificateExtension.from_dict(extension)
    _check_base(base, extension)
    coord_frame = _frame_of_base(base, extension, coord_frame)
    extension.verify_id()
    if adapter is None:
        adapter = registry.adapter_for_extension(extension)
    else:
        registry.check_adapter_identity(adapter, extension)
    require_scipy_assignment()
    labels = extension.labels
    if str(labels["annotation_gt"].get("type")) != adapter.annotation_type:
        raise CertificateVersionError(
            f"annotation_gt type {labels['annotation_gt'].get('type')!r} "
            f"!= adapter {adapter.annotation_type!r}"
        )
    image_size = extension.image_size
    oracle = extension.oracle
    frame = source_frame(coord_frame, image_size)
    ctx = PublicContext(
        task=str(extension.base["task"]),
        query_id=str(extension.base["query_id"]),
        image_size=image_size,
        coord_frame=coord_frame,
        frame=frame,
        identity=identity,
    )
    response = response if isinstance(response, str) else ("" if response is None else str(response))
    located = locate_scored_object(response)

    answer, answer_legacy, answer_guard = score_answer(response, labels["answer_gt"], located)

    if located.scored is not None:
        annotation_value, annotation_source = located.scored.get("annotation"), "scored_object"
    else:
        _, annotation_value, _ = legacy_prediction(response)
        annotation_source = "legacy_extraction"
    grounding, annotation_parse_ok, grounding_details = score_grounding(
        annotation_value, labels, adapter.annotation_type, frame, image_size
    )
    annotation = annotation_view(annotation_value, adapter.annotation_type, frame)

    applicable = dict(adapter.family_applicable(oracle))
    checks, semantics, view, envelope_error, adapter_diagnostics = _semantics(
        adapter, located, oracle, ctx, applicable, annotation if located.scored is not None else None
    )

    scored = located.scored
    answer_ok = scored is not None and "answer" in scored and adapter.answer_schema_ok(scored["answer"])
    annotation_error = (
        annotation_schema_error(
            scored.get("annotation"),
            adapter.annotation_type,
            keys=adapter.annotation_keys(),
            cap=adapter.annotation_cap(),
        )
        if scored is not None
        else None
    )
    fmt = format_check(
        located,
        view,
        annotation_parse_ok=annotation_parse_ok and annotation_source == "scored_object",
        answer_ok=answer_ok,
        annotation_error=annotation_error,
    )

    answer = _unit("A", answer)
    grounding = _unit("G", grounding)
    semantics = _unit("M", semantics)
    fmt_score = _unit("F", fmt.score)
    overall = reward_arms(answer, grounding, semantics, fmt_score, alpha=alpha, format_weight=format_weight)
    joint = 1.0 if answer == 1.0 and fmt_score == 1.0 and semantics == 1.0 else 0.0
    families = family_results(checks, adapter.m_families)
    error_class = adapter.error_class(
        families, envelope_failure=envelope_error is not None, semantics=semantics, answer=answer
    )

    identity_budget = adapter.annotation_over_cap(annotation)
    if identity_budget:
        report_dict: Dict[str, Any] = {}
        g_id: Optional[float] = None
    else:
        report, g_id = adapter.identity_report(annotation, oracle, ctx)
        report_dict = report.to_dict()
    diagnostics: Dict[str, Any] = {
        "A_legacy": answer_legacy,
        "answer_guard": answer_guard,
        "G_id": g_id,
        "G_details": grounding_details,
        "annotation_source": annotation_source,
        "identity": report_dict,
        "identity_budget": identity_budget,
        "identity_params": identity.to_dict(),
        "located": located.to_dict(),
        "envelope_error": envelope_error,
        "format_reasons": list(fmt.reasons),
        "coord_frame": coord_frame,
        "n_claimed": view.n_claimed if view is not None and view.envelope_ok else None,
        "adapter": adapter_diagnostics,
    }
    if shown_frame_diagnostics and coord_frame == "px_source":
        diagnostics.update(
            _shown_diagnostics(adapter, located, annotation_value, labels, oracle, ctx, applicable)
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    diagnostics["verifier_ms"] = elapsed_ms
    diagnostics["slow_row"] = elapsed_ms > SLOW_ROW_MS
    return ScoreResult(
        answer=answer,
        grounding=grounding,
        semantics=semantics,
        format=fmt_score,
        overall=overall,
        joint=joint,
        checks=checks,
        diagnostics=diagnostics,
        coverage_status="supported",
        scorer_version=SCORER_VERSION,
        error_class=error_class,
    )


def _shown_diagnostics(
    adapter: Adapter,
    located: LocatedObject,
    annotation_value: Any,
    labels: Mapping[str, Any],
    oracle: Mapping[str, Any],
    ctx: PublicContext,
    applicable: Mapping[str, bool],
) -> Dict[str, Any]:
    """G and M when the response's coordinates are read in Qwen2.5-VL's shown frame.

    Logged only: this diagnostic never changes whether a row scores. The main path already
    scored the same object in the contract frame, so an exception here is recorded as
    ``shown_error`` (with ``G_shown``/``M_shown`` None) instead of failing the row
    (review CONTRACT-2).
    """
    try:
        frame = shown_frame(ctx.image_size)
        shown_ctx = PublicContext(
            task=ctx.task,
            query_id=ctx.query_id,
            image_size=ctx.image_size,
            coord_frame="px_shown",
            frame=frame,
            identity=ctx.identity,
        )
        g_shown, _, _ = score_grounding(annotation_value, labels, adapter.annotation_type, frame, ctx.image_size)
        _, m_shown, _, _, _ = _semantics(adapter, located, oracle, shown_ctx, applicable)
        return {
            "G_shown": _unit("G_shown", g_shown),
            "M_shown": _unit("M_shown", m_shown),
            "shown_size": list(shown_size(ctx.image_size)),
            "shown_error": None,
        }
    except Exception as exc:  # noqa: BLE001 - a diagnostic must not decide whether a row scores
        return {"G_shown": None, "M_shown": None, "shown_size": None, "shown_error": type(exc).__name__}


__all__ = [
    "ALPHA",
    "ARMS",
    "FORMAT_WEIGHT",
    "check_prompt",
    "SLOW_ROW_MS",
    "legacy_prediction",
    "require_scipy_assignment",
    "reward_arms",
    "score_answer",
    "score_certificate_response",
    "score_grounding",
]
