"""Public prompt, system prompt and schema resources of the certificate condition.

DESIGN section 4. The texts live in ``resources/prompts/experimental/certificates/
cert_prompts_v1.json`` (the single source of truth); this module only assembles them. A
rendered prompt is::

    <image>{question}
    {frame line}
    {envelope line}
    Annotation format: {the task's legacy annotation hint, frame wording removed}.
    {certificate line}
    Answer format: {the task's legacy answer hint}.
    Example JSON:
    {fixed example}

Only the question line and the image size vary between rows of one task and frame. Nothing
in this module reads ``answer_gt``, ``annotation_gt``, ``reward_contract``, the trace
sidecar or any private extension: the public view is the image, the question, ``W x H``, the
contract (kind, field types, caps) and the frame.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.resources import resource_path

from .types import COORD_FRAMES, CertificateVersionError, PublicContract, QuarantineError

PROMPTS_VERSION = "cert_prompts_v1"
SYSTEM_PROMPT_NAME = "cert_system_v1.txt"
# TRACE's answer-only system prompt as trace-tool records it (sha256 of the stripped text of
# trace-tool prompts/trace_system_prompt.txt, whose raw-file sha256 is f394927d...): the
# system prompt of the ans_v1 condition (DESIGN 7.1).
TRACE_SYSTEM_PROMPT_SHA256 = "d6105d53a9a11d7cdcaa6f3d045882b49998ad0c5978d7d4ec8b489479a491e4"
QUESTION_SLOT = "{question}"
LEGACY_CONDITIONS: Mapping[str, str] = {
    "ans_v1": "prompt_answer",
    "ann_v1": "prompt_answer_and_annotation",
}
CERT_KINDS = ("selected_set", "filtered_mean", "weighted_sum", "path")

_PUBLIC_SCHEMA_DIR = ("schemas", "public")
_PRIVATE_SCHEMA_NAME = "extension.v1.schema.json"


def _resource(*parts: str) -> Path:
    return resource_path("prompts", "experimental", "certificates", *parts)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _bundle() -> Dict[str, Any]:
    bundle = _read_json(_resource("cert_prompts_v1.json"))
    if bundle.get("bundle_id") != PROMPTS_VERSION:
        raise CertificateVersionError(f"unexpected prompt bundle id {bundle.get('bundle_id')!r}")
    if tuple(bundle["frames"]) != COORD_FRAMES:
        raise CertificateVersionError(f"prompt bundle frames {tuple(bundle['frames'])} != {COORD_FRAMES}")
    for kind, block in bundle["kinds"].items():
        if block["example"]["certificate"]["kind"] != kind:
            raise CertificateVersionError(f"example of kind {kind!r} carries another certificate kind")
    return bundle


def load_prompt_bundle() -> Dict[str, Any]:
    """A private copy of ``cert_prompts_v1.json`` (the cached original is never handed out)."""
    return copy.deepcopy(_bundle())


# ----------------------------------------------------------------------------------------
# resources: system prompt, schemas, digests


def system_prompt_path() -> Path:
    """Filesystem path of ``cert_system_v1.txt`` (trace-tool and EasyR1 configs read it)."""
    return _resource(SYSTEM_PROMPT_NAME)


def system_prompt() -> str:
    """The system prompt text, stripped the way TRACE's generator and EasyR1 strip prompt files."""
    return system_prompt_path().read_text(encoding="utf-8").strip()


def system_prompt_sha256() -> str:
    """sha256 of :func:`system_prompt` (what trace-tool records as ``system_prompt_sha256``)."""
    return hashlib.sha256(system_prompt().encode("utf-8")).hexdigest()


def expected_system_prompt_sha256(condition: str) -> str:
    """DESIGN 7.1 system prompt per condition target: ``ans_v1`` uses TRACE's answer-only
    prompt; ``ann_v1`` and both certificate frames use ``cert_system_v1``."""
    if condition == "ans_v1":
        return TRACE_SYSTEM_PROMPT_SHA256
    if condition in ("ann_v1", *COORD_FRAMES):
        return system_prompt_sha256()
    raise ValueError(f"unknown condition target {condition!r}")


def public_schema(kind: str) -> Dict[str, Any]:
    """JSON Schema (draft 2020-12) of the response object of one certificate kind."""
    if kind not in CERT_KINDS:
        raise KeyError(f"unknown certificate kind {kind!r}; expected one of {CERT_KINDS}")
    return _read_json(_resource(*_PUBLIC_SCHEMA_DIR, f"{kind}.v1.schema.json"))


def private_schema() -> Dict[str, Any]:
    """JSON Schema of one private extension record (labels and oracle; never public)."""
    return _read_json(_resource("schemas", "private", _PRIVATE_SCHEMA_NAME))


def resource_digests() -> Dict[str, str]:
    """sha256 of every prompt, system-prompt and schema file, keyed by relative path (for receipts)."""
    names = [
        "cert_prompts_v1.json",
        SYSTEM_PROMPT_NAME,
        *(f"{'/'.join(_PUBLIC_SCHEMA_DIR)}/{kind}.v1.schema.json" for kind in CERT_KINDS),
        f"schemas/private/{_PRIVATE_SCHEMA_NAME}",
    ]
    return {name: hashlib.sha256(_resource(*name.split("/")).read_bytes()).hexdigest() for name in sorted(names)}


def prompt_sha256(prompt: str) -> str:
    """sha256 of a rendered public prompt (``public.prompt_sha256`` of the extension record)."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------------------
# fixed texts per kind


def _kind_block(kind: str) -> Mapping[str, Any]:
    try:
        return _bundle()["kinds"][kind]
    except KeyError:
        raise CertificateVersionError(
            f"no prompt block for certificate kind {kind!r}; known kinds: {sorted(_bundle()['kinds'])}"
        ) from None


def certificate_line(kind: str) -> str:
    """The fixed English rendering of the certificate schema (what ``Adapter.certificate_line`` returns)."""
    return str(_kind_block(kind)["certificate_line"])


def example_object(kind: str) -> Dict[str, Any]:
    """The fixed, instance-independent example response object of one kind."""
    return copy.deepcopy(_kind_block(kind)["example"])


def example_json(kind: str) -> str:
    """Compact JSON of the fixed example (keys annotation, certificate, answer), like TRACE's own."""
    return json.dumps(_kind_block(kind)["example"], separators=(",", ":"), ensure_ascii=False)


# ----------------------------------------------------------------------------------------
# row-level pieces


def _train_row(base: Any) -> Mapping[str, Any]:
    """Accept a TrainInstance dict, a ``TraceRecord`` or anything with a ``train`` mapping."""
    row = getattr(base, "train", base)
    if not isinstance(row, Mapping):
        raise TypeError(f"base must be a mapping or a TraceRecord, got {type(base).__name__}")
    return row


def _variants(row: Mapping[str, Any]) -> Mapping[str, Any]:
    variants = row.get("prompt_variants")
    if not isinstance(variants, Mapping) or not isinstance(variants.get("answer_only"), str):
        raise QuarantineError("missing_prompt_variants", str(row.get("instance_id")))
    return variants


def question_line(base: Any) -> str:
    """``prompt_variants["answer_only"].split("\\n", 1)[0]`` with the DESIGN 4 quarantine assertion.

    The first line must be pure question: it may not mention JSON, answer or annotation (any
    case) and may not carry an image marker. rpg prompts put "Return JSON ..." on the next
    line, which is why only the first line is taken.
    """
    row = _train_row(base)
    line = _variants(row)["answer_only"].split("\n", 1)[0]
    if not line.strip():
        raise QuarantineError("empty_question", str(row.get("instance_id")))
    lowered = line.lower()
    for word in _bundle()["question_banned_words"]:
        if word.lower() in lowered:
            raise QuarantineError("question_contains_format_word", f"{word!r} in question of {row.get('instance_id')}")
    if _bundle()["image_marker"] in line:
        raise QuarantineError("question_contains_image_marker", str(row.get("instance_id")))
    return line


def strip_frame_words(text: str) -> str:
    """Remove the legacy frame wording ("pixel-space", "pixel") from an annotation hint."""
    for word in _bundle()["frame_words_removed"]:
        text = re.sub(rf"\b{re.escape(word)}\s+", "", text)
    if "pixel" in text.lower():
        raise QuarantineError("frame_wording_left", text)
    return text


def _legacy_hint(prompt: str, prefixes: Sequence[str]) -> str | None:
    """The hint of the first line (after the question) that starts with one of the legacy labels."""
    for line in prompt.split("\n")[1:]:
        for prefix in prefixes:
            if line.startswith(f"{prefix}:"):
                text = line[len(prefix) + 1 :].strip()
                return text[:-1] if text.endswith(".") else text
    return None


def _task_texts(kind: str, row: Mapping[str, Any]) -> Tuple[str, str]:
    """(annotation hint, answer hint) of the row's task, cross-checked against TRACE's own prompt.

    The JSON is the source of text. When the row carries TRACE's legacy variants, their
    annotation (frame wording removed) and answer hints must equal the stored ones; a change
    in TRACE's task prompts is therefore a quarantine, never a silent prompt change.
    """
    task = str(row.get("task"))
    tasks = _kind_block(kind)["tasks"]
    if task not in tasks:
        raise QuarantineError("task_not_in_prompt_block", f"{task!r} is not a {kind!r} task")
    texts = tasks[task]
    variants = _variants(row)
    bundle = _bundle()
    answer_hint = _legacy_hint(variants["answer_only"], bundle["legacy_answer_prefixes"])
    if answer_hint != texts["answer_hint"]:
        raise QuarantineError("answer_hint_drift", f"{task}: legacy {answer_hint!r} != stored {texts['answer_hint']!r}")
    annotated = variants.get("answer_and_annotation")
    if isinstance(annotated, str):
        legacy = _legacy_hint(annotated, bundle["legacy_annotation_prefixes"])
        if legacy is None or strip_frame_words(legacy) != texts["annotation_hint"]:
            raise QuarantineError(
                "annotation_hint_drift", f"{task}: legacy {legacy!r} != stored {texts['annotation_hint']!r}"
            )
    return str(texts["annotation_hint"]), str(texts["answer_hint"])


def frame_line(coord_frame: str, image_size: Sequence[int] | None = None) -> str:
    """DESIGN 2.2 frame line. ``px_source`` states ``W x H``; ``rel1000`` states no size."""
    frames = _bundle()["frames"]
    if coord_frame not in frames:
        raise ValueError(f"coord_frame must be one of {COORD_FRAMES}, got {coord_frame!r}")
    text = str(frames[coord_frame])
    if "{W}" not in text:
        return text
    if image_size is None or len(image_size) != 2 or int(image_size[0]) <= 0 or int(image_size[1]) <= 0:
        raise ValueError(f"image_size must be positive [W, H], got {image_size!r}")
    return text.replace("{W}", str(int(image_size[0]))).replace("{H}", str(int(image_size[1])))


def _check_contract(kind: str, contract: PublicContract) -> None:
    """The stored texts must describe the contract the scorer enforces (types, caps, schema id)."""
    block = _kind_block(kind)
    expected = {
        "schema_id": block["schema_id"],
        "annotation_type": block["annotation_type"],
        "list_key": block["list_key"],
        "list_cap": block["list_cap"],
    }
    actual = {name: getattr(contract, name) for name in expected}
    if actual != expected:
        raise CertificateVersionError(f"public contract {actual} does not match prompt block {expected}")


def _body(kind: str, annotation_hint: str, answer_hint: str) -> list[str]:
    bundle = _bundle()
    return [
        str(bundle["envelope"]),
        f"{bundle['annotation_label']}{annotation_hint}.",
        certificate_line(kind),
        f"{bundle['answer_label']}{answer_hint}.",
        str(bundle["example_header"]),
        example_json(kind),
    ]


def render_public_prompt(base: Any, public_contract: PublicContract) -> str:
    """The certificate-condition user prompt of one row (DESIGN 4).

    ``base`` is the row's TrainInstance dict (or a ``TraceRecord``); only ``task`` and
    ``prompt_variants`` are read. ``public_contract`` supplies the kind, the image size and
    the coordinate frame. Raises ``QuarantineError`` when the row cannot be rendered safely
    (question with format words, unknown task, TRACE hint drift) and
    ``CertificateVersionError`` when the contract and the stored texts disagree.
    """
    kind = public_contract.kind
    _check_contract(kind, public_contract)
    row = _train_row(base)
    question = question_line(row)
    annotation_hint, answer_hint = _task_texts(kind, row)
    frame = frame_line(public_contract.coord_frame, public_contract.image_size)
    return "\n".join([f"{_bundle()['image_marker']}{question}", frame, *_body(kind, annotation_hint, answer_hint)])


def prompt_template(task: str, public_contract: PublicContract) -> str:
    """The constant part of the prompt of one (task, frame): ``{question}``, ``{W}``, ``{H}`` stay slots.

    ``render_public_prompt`` equals this template with the question line and ``W x H``
    substituted; the isolation tests (DESIGN 4, leakage test 1) rely on that identity.
    """
    kind = public_contract.kind
    _check_contract(kind, public_contract)
    tasks = _kind_block(kind)["tasks"]
    if task not in tasks:
        raise QuarantineError("task_not_in_prompt_block", f"{task!r} is not a {kind!r} task")
    texts = tasks[task]
    frame = str(_bundle()["frames"][public_contract.coord_frame])
    return "\n".join(
        [
            f"{_bundle()['image_marker']}{QUESTION_SLOT}",
            frame,
            *_body(kind, str(texts["annotation_hint"]), str(texts["answer_hint"])),
        ]
    )


def legacy_prompt(base: Any, condition: str) -> str:
    """TRACE's own prompt for the legacy conditions ``ans_v1`` / ``ann_v1``, unchanged.

    This is exactly the ``prompt_answer`` / ``prompt_answer_and_annotation`` column that
    TRACE's RLVR export writes for the row (image marker prefixed, redundant output-contract
    line removed), so legacy rows are prompted like the released validation set.
    """
    if condition not in LEGACY_CONDITIONS:
        raise ValueError(f"legacy condition must be one of {sorted(LEGACY_CONDITIONS)}, got {condition!r}")
    from trace_tasks.core.rlvr_export import _build_prompt_columns

    return _build_prompt_columns(_train_row(base), image_count=1)[LEGACY_CONDITIONS[condition]]


__all__ = [
    "CERT_KINDS",
    "LEGACY_CONDITIONS",
    "PROMPTS_VERSION",
    "QUESTION_SLOT",
    "SYSTEM_PROMPT_NAME",
    "TRACE_SYSTEM_PROMPT_SHA256",
    "certificate_line",
    "expected_system_prompt_sha256",
    "example_json",
    "example_object",
    "frame_line",
    "legacy_prompt",
    "load_prompt_bundle",
    "private_schema",
    "prompt_sha256",
    "prompt_template",
    "public_schema",
    "question_line",
    "render_public_prompt",
    "resource_digests",
    "strip_frame_words",
    "system_prompt",
    "system_prompt_path",
    "system_prompt_sha256",
]
