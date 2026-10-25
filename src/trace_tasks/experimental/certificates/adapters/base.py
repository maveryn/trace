"""Abstract certificate adapter (the contract every adapter implements).

See ``ADAPTER_INTERFACE.md`` next to the package for the full contract. An adapter is a
small explicit verifier for one task family: it builds a private oracle from a generated
trace record, declares the certificate schema, resolves annotation boxes to scene objects
with the shared identity rule, and returns typed checks. It never evaluates descriptive
strings or model output and never sees anything but source-pixel geometry.
"""

from __future__ import annotations

import abc
import copy
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, ClassVar, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.resources import resource_path

from ..checks import classify_error
from ..geometry import BOX_ANNOTATION_TYPES, POINT_ANNOTATION_TYPES, AnnotationView, Frame
from ..identity import (
    DEFAULT_IDENTITY,
    CandidateSet,
    IdentityParams,
    IdentityReport,
    Resolution,
    resolve_boxes,
    resolve_points,
)
from ..parsing import CertificateSchema, CertificateView, FieldSpec, validate_certificate
from ..types import (
    FAMILY_STAGES,
    AdapterSpec,
    CertificateVersionError,
    CheckResult,
    Corruption,
    PublicContract,
    QuarantineError,
    TraceRecord,
)

SftObject = Tuple[Any, Dict[str, Any], Any]
HOLDOUT_TYPES = ("render", "program", "both")  # DESIGN 6 E2 held-out cell types
PROMPT_BUNDLE = ("prompts", "experimental", "certificates", "cert_prompts_v1.json")


@lru_cache(maxsize=1)
def prompt_bundle() -> Mapping[str, Any]:
    """The versioned certificate prompt bundle (the single source of the public prompt text)."""
    return json.loads(resource_path(*PROMPT_BUNDLE).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class PublicContext:
    """Public facts available to ``checks``: the row's task/query, image size and frame.

    ``frame`` maps response-frame coordinates to source pixels. Adapters must convert every
    coordinate that appears in the *certificate* (for example path ``at`` points) with
    ``frame.point``/``frame.box``; annotation geometry arrives already converted. During the
    shown-frame diagnostic the scorer passes ``coord_frame="px_shown"`` with another frame.
    ``identity`` carries the identity thresholds (sensitivity analysis) and must be used for
    every box resolution.
    """

    task: str
    query_id: str
    image_size: Tuple[int, int]
    coord_frame: str
    frame: Frame
    identity: IdentityParams = DEFAULT_IDENTITY


class Adapter(abc.ABC):
    """Base class of the four certificate adapters.

    Subclasses set the class attributes below and implement every abstract method. Each
    adapter module exposes ``create_adapter() -> Adapter`` (the registry imports it lazily).
    """

    # --- identity and versions -----------------------------------------------------------
    adapter_id: ClassVar[str]
    kind: ClassVar[str]
    version: ClassVar[str]
    schema_id: ClassVar[str]
    verifier_version: ClassVar[str]
    supported: ClassVar[Mapping[str, Tuple[str, ...]]]
    instance_versions: ClassVar[Tuple[str, ...]] = ("v0",)

    # --- contract shape --------------------------------------------------------------------
    annotation_type: ClassVar[str]
    annotation_roles: ClassVar[Tuple[str, ...]] = ("",)
    list_key: ClassVar[str | None]
    list_cap: ClassVar[int]
    cert_fields: ClassVar[Mapping[str, FieldSpec]]
    element_fields: ClassVar[Mapping[str, FieldSpec]]

    # --- M ------------------------------------------------------------------------------------
    m_families: ClassVar[Tuple[str, ...]]
    family_stages: ClassVar[Mapping[str, str]]
    local_check_ids: ClassVar[Tuple[str, ...]] = ()

    # --- split cells (DESIGN 6, D13): every ``split_cell`` value is in exactly one of these ----
    # (empty for scoring-only test adapters; the exporter quarantines undeclared cells)
    train_cells: ClassVar[Tuple[str, ...]] = ()
    heldout_cells: ClassVar[Mapping[str, str]] = {}  # E2 cell -> "render" | "program" | "both"

    # ======================================================================================
    # identity

    @classmethod
    def validate_definition(cls) -> None:
        """Static consistency of the class attributes (the registry calls this on load)."""
        for name in (
            "adapter_id", "kind", "version", "schema_id", "verifier_version",
            "annotation_type", "list_cap", "cert_fields", "element_fields",
            "m_families", "family_stages", "supported", "list_key",
        ):
            if not hasattr(cls, name):
                raise TypeError(f"{cls.__name__} must define {name}")
        for name in ("adapter_id", "kind", "version", "schema_id", "verifier_version"):
            if not isinstance(getattr(cls, name), str) or not getattr(cls, name):
                raise TypeError(f"{cls.__name__}.{name} must be a non-empty string")
        if not cls.adapter_id.startswith(cls.__module__.rsplit(".", 1)[-1] + ".") and not getattr(
            cls, "_allow_foreign_module", False
        ):
            raise TypeError(f"{cls.__name__}.adapter_id must start with its module name")
        if cls.annotation_type not in BOX_ANNOTATION_TYPES + POINT_ANNOTATION_TYPES:
            raise TypeError(f"{cls.__name__}.annotation_type {cls.annotation_type!r} is not a box/point type")
        if not cls.m_families or len(set(cls.m_families)) != len(cls.m_families):
            raise TypeError(f"{cls.__name__}.m_families must be a non-empty list of distinct ids")
        for family in cls.m_families:
            if cls.family_stages.get(family) not in FAMILY_STAGES:
                raise TypeError(f"{cls.__name__}.family_stages[{family!r}] must be one of {FAMILY_STAGES}")
        if not cls.supported:
            raise TypeError(f"{cls.__name__}.supported must list at least one task")
        if cls.list_key is not None and not cls.element_fields:
            raise TypeError(f"{cls.__name__}.element_fields must be set when list_key is set")
        if not isinstance(cls.train_cells, tuple) or not isinstance(cls.heldout_cells, Mapping):
            raise TypeError(f"{cls.__name__}: train_cells must be a tuple and heldout_cells a mapping")
        if set(cls.train_cells) & set(cls.heldout_cells):
            raise TypeError(f"{cls.__name__}: a cell cannot be both train and held out")
        if any(kind not in HOLDOUT_TYPES for kind in cls.heldout_cells.values()):
            raise TypeError(f"{cls.__name__}.heldout_cells values must be in {HOLDOUT_TYPES}")

    def spec(self) -> AdapterSpec:
        return AdapterSpec(
            adapter_id=self.adapter_id,
            kind=self.kind,
            version=self.version,
            schema_id=self.schema_id,
            verifier_version=self.verifier_version,
            supported=tuple((task, tuple(qids)) for task, qids in sorted(self.supported.items())),
            instance_versions=tuple(self.instance_versions),
            annotation_type=self.annotation_type,
            list_key=self.list_key,
            list_cap=int(self.list_cap),
        )

    def identity(self) -> Dict[str, str]:
        """The ``adapter`` block of a private extension record."""
        return self.spec().identity()

    def supports(self, task: str, query_id: str, instance_version: str) -> bool:
        return self.spec().supports(task, query_id, instance_version)

    def public_contract(self, image_size: Sequence[int], coord_frame: str) -> PublicContract:
        return PublicContract(
            kind=self.kind,
            schema_id=self.schema_id,
            coord_frame=coord_frame,
            image_size=(int(image_size[0]), int(image_size[1])),
            annotation_type=self.annotation_type,
            list_key=self.list_key,
            list_cap=int(self.list_cap),
        )

    # ======================================================================================
    # export side (private; runs on generated trace records)

    @abc.abstractmethod
    def build_oracle(self, record: TraceRecord) -> Dict[str, Any]:
        """JSON-safe private oracle (DESIGN 3.x). Raise QuarantineError(reason) on any invariant
        failure; never return a partial oracle. Geometry is in source pixels."""

    @abc.abstractmethod
    def split_cell(self, record: TraceRecord) -> str:
        """Held-out cell label used by the split builder (DESIGN 6, D13), e.g. ``"above|zebra"``."""

    def holdout_type(self, cell: str) -> str:
        """``""`` for a train/E1 cell, else the E2 type of a held-out cell (DESIGN 6).

        A cell in neither list is a data problem: it raises ``QuarantineError("unknown_cell")``.
        """
        if cell in self.heldout_cells:
            return str(self.heldout_cells[cell])
        if cell in self.train_cells:
            return ""
        raise QuarantineError("unknown_cell", f"{self.adapter_id}: split cell {cell!r} is not declared")

    @abc.abstractmethod
    def semantic_key(self, record: TraceRecord) -> str:
        """Dedupe key across splits (DESIGN 6 "Group and dedupe rules"); a stable string."""

    @abc.abstractmethod
    def sft_certificate(self, oracle: Mapping[str, Any], rng_seed: int) -> SftObject:
        """``(annotation, certificate, answer)`` of the canonical SFT target.

        Annotation geometry is unrounded source pixels (the exporter rounds half-up); the
        certificate dict has ``kind`` first and its list in canonical order; ``answer`` is the
        oracle answer. Any choice among valid witnesses uses ``random.Random(rng_seed)``.
        """

    # ======================================================================================
    # schema (parse side)

    def schema(self, cert: Any) -> CertificateSchema:
        """Schema of the certificate variant ``cert`` claims (override for several variants)."""
        return CertificateSchema(
            kind=self.kind,
            fields=dict(self.cert_fields),
            list_key=self.list_key,
            element_fields=dict(self.element_fields),
            list_cap=int(self.list_cap),
        )

    def annotation_size(self, annotation: AnnotationView) -> int:
        """Bound for ``annotation_index``: the smallest list length over ``annotation_roles``
        (0 when the certificate does not bind by index)."""
        if not self.annotation_roles:
            return 0
        return min(annotation.size(role) for role in self.annotation_roles)

    def validate_certificate(self, cert: Any, n_annotation: int) -> CertificateView:
        return validate_certificate(cert, self.schema(cert), n_annotation)

    def element_schema_ok(self, cert: Any, n_annotation: int) -> List[bool]:
        """Per-element validity (empty on an envelope failure)."""
        return self.validate_certificate(cert, n_annotation).element_validity()

    def answer_schema_ok(self, answer: Any) -> bool:
        """F condition on the answer value: a JSON integer (no bool, no float)."""
        return type(answer) is int

    # ======================================================================================
    # verification

    @abc.abstractmethod
    def candidates(self, oracle: Mapping[str, Any]) -> Mapping[str, CandidateSet]:
        """Per annotation role, **all** scene objects of the adapter's class (source px) and
        the oracle-selected positions. Used for identity, ``G_id`` and identity logs."""

    def family_applicable(self, oracle: Mapping[str, Any]) -> Mapping[str, bool]:
        """Applicability of each M family; depends on the oracle only (default: all)."""
        return {family: True for family in self.m_families}

    @abc.abstractmethod
    def checks(
        self,
        cert: CertificateView,
        annotation: AnnotationView,
        oracle: Mapping[str, Any],
        ctx: PublicContext,
    ) -> List[CheckResult]:
        """Truth and local checks of an envelope-valid certificate.

        Must return exactly one family-level ``truth`` result (``check_id == family``) per
        entry of ``m_families`` with ``applicable == family_applicable(oracle)[family]``,
        plus the local checks; optional per-element results use a distinct ``check_id`` with
        ``family`` set. Must not raise on any envelope-valid input.
        """

    def diagnostics(
        self,
        cert: CertificateView,
        annotation: AnnotationView,
        oracle: Mapping[str, Any],
        ctx: PublicContext,
        checks: Sequence[CheckResult],
    ) -> Dict[str, Any]:
        """Extra JSON-safe diagnostics (for example operand-read accuracy on distractors)."""
        return {}

    def error_class(
        self,
        families: Mapping[str, CheckResult],
        *,
        envelope_failure: bool,
        semantics: float,
        answer: float,
    ) -> str:
        """First failing stage (DESIGN 5); ``family_stages`` maps families to stages."""
        return classify_error(
            envelope_failure=envelope_failure,
            families=families,
            family_stages=self.family_stages,
            semantics=semantics,
            answer=answer,
        )

    # --- shared helpers ------------------------------------------------------------------

    def resolve_role(
        self,
        annotation: AnnotationView,
        role: str,
        oracle: Mapping[str, Any],
        ctx: PublicContext,
        candidates: Mapping[str, CandidateSet] | None = None,
    ) -> List[Resolution]:
        """Resolve one annotation role against its candidates (boxes: D4 rule with one-to-one
        dedupe; points: strict point-in-tile). Results are cached on the view, so checks,
        diagnostics and the identity report of one scoring call resolve every box once."""
        candidate_set = (candidates or self.candidates(oracle))[role]
        key = (role, ctx.identity, candidate_set.boxes)
        cached = annotation.cache.get(key)
        if cached is None:
            cached = annotation.cache[key] = tuple(self._resolve_role(annotation, role, candidate_set, ctx))
        return list(cached)

    def _resolve_role(
        self,
        annotation: AnnotationView,
        role: str,
        candidate_set: CandidateSet,
        ctx: PublicContext,
    ) -> List[Resolution]:
        """Uncached resolution of one role (adapters may override with an equivalent fast path)."""
        items = annotation.items(role)
        if annotation.geometry == "point":
            return resolve_points(items, candidate_set.boxes)
        return resolve_boxes(items, candidate_set.boxes, ctx.identity)

    def annotation_cap(self) -> int | None:
        """Most boxes one annotation role may hold: the certificate list cap when the
        certificate binds to the annotation by index (one element per box), else None."""
        return int(self.list_cap) if self.annotation_roles and self.list_key is not None else None

    def annotation_keys(self) -> Tuple[str, ...]:
        """Exact keys of a map-type annotation (the legacy contract's roles)."""
        return tuple(role for role in self.annotation_roles if role)

    def annotation_over_cap(self, annotation: AnnotationView) -> bool:
        """An annotation role longer than :meth:`annotation_cap` (deterministic budget failure)."""
        cap = self.annotation_cap()
        return cap is not None and annotation.max_size() > cap

    def identity_report(
        self,
        annotation: AnnotationView,
        oracle: Mapping[str, Any],
        ctx: PublicContext,
    ) -> Tuple[IdentityReport, float]:
        """Resolutions of every role plus ``G_id`` (micro identity-rule set F1)."""
        candidates = self.candidates(oracle)
        report = IdentityReport()
        hits = predicted = selected = 0
        for role, candidate_set in candidates.items():
            resolutions = self.resolve_role(annotation, role, oracle, ctx, candidates)
            report.roles[role] = resolutions
            target = set(candidate_set.selected)
            hits += len({r.candidate for r in resolutions if r.resolved and r.candidate in target})
            predicted += len(resolutions)
            selected += len(target)
        g_id = 1.0 if predicted + selected == 0 else 2.0 * hits / (predicted + selected)
        return report, g_id

    # ======================================================================================
    # counterexamples

    @abc.abstractmethod
    def corruptions(
        self,
        oracle: Mapping[str, Any],
        sft_obj: Mapping[str, Any],
        rng_seed: int,
    ) -> List[Corruption]:
        """Deliberate corruptions of the SFT object ``{"annotation", "certificate", "answer"}``
        (source px) with the checks each one must move (see :class:`Corruption`)."""

    # ======================================================================================
    # prompt pieces (instance-independent; ``cert_prompts_v1.json`` is the single source)

    def prompt_block(self) -> Mapping[str, Any]:
        """This kind's block of the prompt bundle, checked against the adapter's contract."""
        block = prompt_bundle()["kinds"][self.kind]
        found = tuple(block.get(key) for key in ("schema_id", "annotation_type", "list_key", "list_cap"))
        expected = (self.schema_id, self.annotation_type, self.list_key, self.list_cap)
        if found != expected or set(block.get("tasks", ())) != set(self.supported):
            raise CertificateVersionError(f"prompt bundle block {self.kind} {found} != adapter {expected}")
        return block

    def certificate_line(self) -> str:
        """Fixed English rendering of the certificate schema (exactly what prompts.py renders)."""
        return str(self.prompt_block()["certificate_line"])

    def example_object(self) -> Dict[str, Any]:
        """Fixed, self-consistent example ``{"annotation", "certificate", "answer"}`` (DESIGN 4)."""
        return copy.deepcopy(dict(self.prompt_block()["example"]))

    def example_json(self) -> str:
        """Compact JSON of :meth:`example_object` (key order annotation, certificate, answer)."""
        example = self.example_object()
        ordered = {key: example[key] for key in ("annotation", "certificate", "answer")}
        return json.dumps(ordered, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


__all__ = ["HOLDOUT_TYPES", "Adapter", "PublicContext", "SftObject", "prompt_bundle"]
