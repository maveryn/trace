"""``weighted_sum`` certificate adapter for the lever side-torque task (DESIGN 3.3).

The model lists one term per annotation box: the weight printed on the block, the distance
mark below it and their product. The answer is the sum of the products; it is scored by A
and logged by ``ws.answer.local``, never by M (there is no ``total`` or ``side`` field).

Terms bind to scene blocks only through the model's own annotation boxes (``annotation_index``
plus the D4 identity rule) against every weight block on **both** sides, so a wrong-side block
is a resolved extra claim rather than an off-target box. Weight digits repeat on one side in
about 41% of scenes, so nothing is ever bound by label.

Oracle source (persisted trace record, which survives the annotation sanitizer):
``scene_ir.entities[entity_type == "physics_lever_weight"]`` with
``{bbox_px, meta{side, distance_units, value, missing, relevant_to_task}}`` and
``execution_trace.{torque_side, query_side, target_answer, weight_specs}``. ``weight_specs``
carries no ids; it is joined to the entities by index and the join is checked at export.
"""

from __future__ import annotations

import copy
import random
from itertools import combinations
from typing import Any, ClassVar, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..checks import (
    Binding,
    bind_elements,
    bound_exactly_once,
    counted_truth,
    indices_bijective,
    local_check,
)
from ..geometry import AnnotationView, normalize_box
from ..identity import CandidateSet
from ..parsing import INDEX, INT, CertificateView, ElementView, FieldSpec
from ..types import (
    CheckResult,
    Corruption,
    QuarantineError,
    TraceRecord,
    canonical_digest,
)
from .base import Adapter, PublicContext

TASK_ID = "task_physics__lever__side_torque_value"
QUERY_ID = "single"
WEIGHT_ENTITY_TYPE = "physics_lever_weight"
SIDES = ("left", "right")
WEIGHT_RANGE = (1, 9)
DISTANCE_RANGE = (1, 8)
SCENE_VARIANTS = ("center_fulcrum", "offset_fulcrum", "textured_beam")
TRAIN_CELLS = ("center_fulcrum", "offset_fulcrum")
HELDOUT_CELLS = {"textured_beam": "render"}

TERMS = "ws.terms.truth"
WEIGHT = "ws.weight.truth"
DISTANCE = "ws.distance.truth"
PRODUCT = "ws.product.truth"
REF_LOCAL = "ws.ref.local"
PRODUCT_LOCAL = "ws.product.local"
ANSWER_LOCAL = "ws.answer.local"


def _int(value: Any) -> Optional[int]:
    """Exact JSON integer from a trusted record (bool and float rejected)."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _require(condition: bool, reason: str, message: str = "") -> None:
    if not condition:
        raise QuarantineError(reason, message)


def _scene_variant(record: TraceRecord) -> str:
    """The lever scene variant, cross-checked between the trace views that carry it."""
    variant = record.execution_trace.get("scene_variant")
    _require(isinstance(variant, str), "missing_field", "execution_trace.scene_variant")
    relations = record.scene_ir.get("relations") or {}
    for source, value in (
        ("scene_ir.relations.scene_variant", relations.get("scene_variant")),
        ("render_spec.scene_variant", record.render_spec.get("scene_variant")),
    ):
        _require(value is None or value == variant, "scene_variant_mismatch", f"{source}={value!r} != {variant!r}")
    _require(variant in SCENE_VARIANTS, "unknown_scene_variant", repr(variant))
    return variant


def _weight_entities(record: TraceRecord) -> List[Mapping[str, Any]]:
    entities = record.scene_ir.get("entities")
    _require(isinstance(entities, list), "missing_field", "scene_ir.entities")
    weights = [e for e in entities if isinstance(e, Mapping) and e.get("entity_type") == WEIGHT_ENTITY_TYPE]
    _require(bool(weights), "missing_weights", "no physics_lever_weight entity")
    return weights


def _product_consistent(element: ElementView) -> bool:
    """``product == weight * distance`` on the model's own typed values."""
    weight, distance, product = element.get("weight"), element.get("distance"), element.get("product")
    if weight is None or distance is None or product is None:
        return False
    return product == weight * distance


class LeverWeightedSumAdapter(Adapter):
    """Lever side torque as a weighted sum of (weight, distance) reads (DESIGN 3.3)."""

    adapter_id = "weighted_sum.lever_side_torque"
    kind = "weighted_sum"
    version = "1.0.0"
    schema_id = "cert.weighted_sum.v1"
    verifier_version = "1.1.0"  # 1.1.0: annotation boxes count as term claims
    supported = {TASK_ID: (QUERY_ID,)}
    instance_versions = ("v0",)

    annotation_type = "bbox_set"
    annotation_roles = ("",)
    list_key = "terms"
    list_cap = 8
    cert_fields: ClassVar[Mapping[str, FieldSpec]] = {}
    element_fields: ClassVar[Mapping[str, FieldSpec]] = {
        "annotation_index": INDEX,
        "weight": INT,
        "distance": INT,
        "product": INT,
    }

    m_families = (TERMS, WEIGHT, DISTANCE, PRODUCT)
    family_stages = {TERMS: "grounding", WEIGHT: "operand", DISTANCE: "operand", PRODUCT: "computation"}
    local_check_ids = (REF_LOCAL, PRODUCT_LOCAL, ANSWER_LOCAL)

    # DESIGN 6 split cells: scene variants; textured_beam is the (render) held-out E2 cell.
    train_cells: ClassVar[Tuple[str, ...]] = TRAIN_CELLS
    heldout_cells: ClassVar[Mapping[str, str]] = HELDOUT_CELLS

    # ======================================================================================
    # export side

    def build_oracle(self, record: TraceRecord) -> Dict[str, Any]:
        """``{"query_side", "blocks": [{oid, bbox, side, weight, distance, relevant}], "total"}``.

        ``oid`` is the block's position in entity order; entity ids stay out of the oracle.
        Every export invariant of DESIGN 3.3 is checked; the first failure raises
        :class:`QuarantineError`.
        """
        _require(
            record.task in self.supported and record.query_id in self.supported[record.task],
            "unsupported_task",
            f"{record.task}/{record.query_id}",
        )
        execution = record.execution_trace
        query_side = execution.get("torque_side")
        _require(query_side in SIDES, "missing_field", "execution_trace.torque_side")
        relations = record.scene_ir.get("relations") or {}
        for source, value in (
            ("execution_trace.query_side", execution.get("query_side")),
            ("scene_ir.relations.torque_side", relations.get("torque_side", query_side)),
        ):
            _require(value == query_side, "side_mismatch", f"{source}={value!r} != {query_side!r}")

        entities = _weight_entities(record)
        specs = execution.get("weight_specs")
        _require(
            isinstance(specs, list) and len(specs) == len(entities),
            "weight_specs_misaligned",
            "weight_specs and scene_ir weight entities differ in length",
        )
        blocks: List[Dict[str, Any]] = []
        for oid, (entity, spec) in enumerate(zip(entities, specs)):
            meta = entity.get("meta")
            _require(isinstance(meta, Mapping), "missing_field", f"weight entity {oid} meta")
            side, missing, relevant = meta.get("side"), meta.get("missing"), meta.get("relevant_to_task")
            weight, distance = _int(meta.get("value")), _int(meta.get("distance_units"))
            _require(
                side in SIDES and weight is not None and distance is not None
                and isinstance(missing, bool) and isinstance(relevant, bool),
                "missing_field",
                f"weight entity {oid} meta {dict(meta)!r}",
            )
            _require(not missing, "missing_weight", f"weight entity {oid} is a placeholder block")
            _require(
                WEIGHT_RANGE[0] <= weight <= WEIGHT_RANGE[1] and DISTANCE_RANGE[0] <= distance <= DISTANCE_RANGE[1],
                "operand_out_of_range",
                f"weight entity {oid}: weight {weight}, distance {distance}",
            )
            raw_box = entity.get("bbox_px")
            box = normalize_box(raw_box)
            _require(
                box is not None and [float(v) for v in raw_box] == list(box),
                "bad_bbox",
                f"weight entity {oid} bbox_px {raw_box!r}",
            )
            _require(
                isinstance(spec, Mapping)
                and (spec.get("side"), _int(spec.get("distance_units")), _int(spec.get("value")))
                == (side, distance, weight)
                and spec.get("missing") is missing
                and spec.get("relevant_to_query") is relevant,
                "weight_specs_misaligned",
                f"weight_specs[{oid}] {spec!r} != entity meta {dict(meta)!r}",
            )
            blocks.append(
                {
                    "oid": oid,
                    "bbox": [float(v) for v in box],
                    "side": side,
                    "weight": weight,
                    "distance": distance,
                    "relevant": relevant,
                }
            )

        _require(
            all(block["relevant"] == (block["side"] == query_side) for block in blocks),
            "relevant_not_queried_side",
            "relevant blocks must be exactly the blocks on torque_side",
        )
        relevant_blocks = [block for block in blocks if block["relevant"]]
        _require(1 <= len(relevant_blocks) <= self.list_cap, "witness_size", str(len(relevant_blocks)))
        slots = [(block["side"], block["distance"]) for block in blocks]
        _require(len(set(slots)) == len(slots), "duplicate_slot", "two blocks share a side and distance")
        for first, second in combinations(blocks, 2):
            a, b = first["bbox"], second["bbox"]
            overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            _require(overlap <= 0.0, "overlapping_blocks", f"blocks {first['oid']} and {second['oid']}")

        total = sum(block["weight"] * block["distance"] for block in relevant_blocks)
        answer_gt = record.answer_gt
        _require(
            answer_gt.get("type") == "integer" and _int(answer_gt.get("value")) == total,
            "total_mismatch",
            f"answer_gt {answer_gt!r} != sum of relevant products {total}",
        )
        _require(_int(execution.get("target_answer")) == total, "total_mismatch", "execution_trace.target_answer")
        known = execution.get(f"known_torque_{query_side}")
        _require(known is None or _int(known) == total, "total_mismatch", f"known_torque_{query_side}")

        annotation_gt = record.annotation_gt
        try:
            gt_boxes = sorted(tuple(float(v) for v in box) for box in annotation_gt.get("value"))
        except (TypeError, ValueError):
            gt_boxes = None
        _require(
            annotation_gt.get("type") == self.annotation_type
            and gt_boxes == sorted(tuple(block["bbox"]) for block in relevant_blocks),
            "annotation_gt_mismatch",
            "annotation_gt must equal the boxes of the relevant blocks",
        )
        return {"query_side": query_side, "blocks": blocks, "total": total}

    def split_cell(self, record: TraceRecord) -> str:
        """The scene variant: ``center_fulcrum``/``offset_fulcrum`` (train, E1) or ``textured_beam`` (E2)."""
        return _scene_variant(record)

    def semantic_key(self, record: TraceRecord) -> str:
        """DESIGN 6 lever key: queried side and sorted ``(side, distance, weight)`` of every block.

        The render variant (fulcrum position, textured beam) is *not* part of the key: it does
        not change an operand or the answer, so alternate renders of one torque problem are one
        semantic scene and are grouped into one split (plan sec. 7; review CONF-13 / DATA-5).
        """
        execution = record.execution_trace
        _require(execution.get("torque_side") in SIDES, "missing_field", "execution_trace.torque_side")
        blocks = []
        for entity in _weight_entities(record):
            meta = entity.get("meta") or {}
            blocks.append([str(meta.get("side")), _int(meta.get("distance_units")), _int(meta.get("value"))])
        _require(all(d is not None and w is not None for _, d, w in blocks), "missing_field", "weight meta")
        return canonical_digest(
            {
                "task": record.task,
                "query_side": execution["torque_side"],
                "blocks": sorted(blocks),
            }
        )

    def sft_certificate(self, oracle: Mapping[str, Any], rng_seed: int) -> Tuple[Any, Dict[str, Any], Any]:
        """Relevant blocks in reading order (ascending box x); the witness is unique, so no RNG."""
        relevant = sorted(
            (block for block in oracle["blocks"] if block["relevant"]),
            key=lambda block: (block["bbox"][0], block["bbox"][1]),
        )
        annotation = [[float(v) for v in block["bbox"]] for block in relevant]
        terms = [
            {
                "annotation_index": index,
                "weight": block["weight"],
                "distance": block["distance"],
                "product": block["weight"] * block["distance"],
            }
            for index, block in enumerate(relevant)
        ]
        return annotation, {"kind": self.kind, "terms": terms}, int(oracle["total"])

    # ======================================================================================
    # verification

    def candidates(self, oracle: Mapping[str, Any]) -> Mapping[str, CandidateSet]:
        blocks = oracle["blocks"]
        return {
            "": CandidateSet(
                boxes=tuple(tuple(float(v) for v in block["bbox"]) for block in blocks),
                selected=tuple(position for position, block in enumerate(blocks) if block["relevant"]),
                keys=tuple(int(block["oid"]) for block in blocks),
            )
        }

    def _bind(
        self,
        cert: CertificateView,
        annotation: AnnotationView,
        oracle: Mapping[str, Any],
        ctx: PublicContext,
    ) -> Tuple[List[Binding], Tuple[int, ...], Dict[int, int]]:
        """``(bindings, relevant positions, {relevant position: the single term bound to it})``."""
        candidates = self.candidates(oracle)
        resolutions = self.resolve_role(annotation, "", oracle, ctx, candidates)
        bindings = bind_elements(cert.elements, resolutions)
        selected = candidates[""].selected
        return bindings, selected, bound_exactly_once(bindings, selected)

    def checks(
        self,
        cert: CertificateView,
        annotation: AnnotationView,
        oracle: Mapping[str, Any],
        ctx: PublicContext,
    ) -> List[CheckResult]:
        blocks = oracle["blocks"]
        elements = cert.elements
        bindings, selected, once = self._bind(cert, annotation, oracle, ctx)
        n, r = cert.n_claimed, len(selected)

        weight_ok = {oid: elements[pos].get("weight") == blocks[oid]["weight"] for oid, pos in once.items()}
        distance_ok = {oid: elements[pos].get("distance") == blocks[oid]["distance"] for oid, pos in once.items()}
        # Per-term gate: a product earns credit only when that term's weight and distance are true.
        product_ok = {
            oid: weight_ok[oid] and distance_ok[oid] and _product_consistent(elements[pos])
            for oid, pos in once.items()
        }
        term_status = [
            {
                "position": binding.position,
                "annotation_index": binding.annotation_index,
                "status": binding.status,
                "oid": binding.candidate,
                "relevant": None if binding.candidate is None else bool(blocks[binding.candidate]["relevant"]),
                "repeated_index": binding.repeated_index,
            }
            for binding in bindings
        ]
        # Every annotation box is a claim (review EX-5): unreferenced boxes raise the term
        # selection denominator like extra terms.
        n_annotation = annotation.size("")
        terms = counted_truth(
            TERMS, len(once), max(n, n_annotation), r, detail={"terms": term_status, "n_boxes": n_annotation}
        )
        weight = counted_truth(WEIGHT, sum(weight_ok.values()), n, r)
        distance = counted_truth(DISTANCE, sum(distance_ok.values()), n, r)
        product = counted_truth(
            PRODUCT,
            sum(product_ok.values()),
            n,
            r,
            detail={"gate_open": sum(weight_ok[oid] and distance_ok[oid] for oid in once)},
        )

        consistent = [_product_consistent(element) for element in elements]
        products = [element.get("product") for element in elements]
        answer_ok = (
            cert.answer is not None and all(p is not None for p in products) and cert.answer == sum(products)
        )
        return [
            terms,
            weight,
            distance,
            product,
            local_check(REF_LOCAL, indices_bijective(elements, annotation.size(""))),
            local_check(PRODUCT_LOCAL, sum(consistent) / n if n else 1.0),
            local_check(ANSWER_LOCAL, answer_ok, detail={"answer": cert.answer}),
        ]

    def diagnostics(
        self,
        cert: CertificateView,
        annotation: AnnotationView,
        oracle: Mapping[str, Any],
        ctx: PublicContext,
        checks: Sequence[CheckResult],
    ) -> Dict[str, Any]:
        """Operand reads on every bound term, and separately on wrong-side (distractor) blocks."""
        blocks = oracle["blocks"]
        bindings, selected, _ = self._bind(cert, annotation, oracle, ctx)
        reads = []
        for binding in bindings:
            if binding.bound and binding.candidate is not None:
                element, block = cert.elements[binding.position], blocks[binding.candidate]
                reads.append(
                    (
                        bool(block["relevant"]),
                        element.get("weight") == block["weight"],
                        element.get("distance") == block["distance"],
                    )
                )
        wrong_side = [read for read in reads if not read[0]]
        return {
            "ws.n_terms": cert.n_claimed,
            "ws.n_relevant": len(selected),
            "ws.operand_read_acc": (
                sum(int(w) + int(d) for _, w, d in reads) / (2.0 * len(reads)) if reads else None
            ),
            "ws.wrong_side_reads": {
                "terms": len(wrong_side),
                "weight_correct": sum(int(w) for _, w, _ in wrong_side),
                "distance_correct": sum(int(d) for _, _, d in wrong_side),
            },
        }

    # ======================================================================================
    # counterexamples

    def _expect(
        self, fail: Iterable[str], *, answer: int, total: int, grounding: str, fmt: str = "pass"
    ) -> Dict[str, Any]:
        """Complete expectation: every M family and local check, plus A, G, M and F."""
        failing = set(fail)
        unknown = failing - set(self.m_families) - set(self.local_check_ids)
        if unknown:
            raise ValueError(f"unknown check ids {sorted(unknown)}")
        expected: Dict[str, Any] = {
            check: ("fail" if check in failing else "pass") for check in self.m_families + self.local_check_ids
        }
        expected["M"] = "fail" if failing & set(self.m_families) else "pass"
        expected["A"] = "pass" if answer == total else "fail"
        expected["G"] = grounding
        expected["F"] = fmt
        return expected

    def corruptions(
        self,
        oracle: Mapping[str, Any],
        sft_obj: Mapping[str, Any],
        rng_seed: int,
    ) -> List[Corruption]:
        """DESIGN 9 counterexamples for the lever (seeded; never mutates ``sft_obj``).

        Each corruption's ``expected`` lists every M family, every local check and A/G/M/F.
        ``G`` is ``"unchanged"`` whenever the annotation is untouched.
        """
        rng = random.Random(rng_seed)
        total = int(oracle["total"])
        blocks = oracle["blocks"]
        base_terms = sft_obj["certificate"]["terms"]
        r = len(base_terms)
        pick = rng.randrange(r)
        all_families = list(self.m_families)
        out: List[Corruption] = []

        def fresh() -> Tuple[Dict[str, Any], List[Any], List[Dict[str, Any]]]:
            obj = copy.deepcopy(dict(sft_obj))
            return obj, obj["annotation"], obj["certificate"]["terms"]

        def consistent_answer(obj: Dict[str, Any]) -> None:
            obj["answer"] = sum(term["product"] for term in obj["certificate"]["terms"])

        def add(
            name: str, obj: Dict[str, Any], fail: Iterable[str], grounding: str, description: str, fmt: str = "pass"
        ) -> None:
            expected = self._expect(fail, answer=obj["answer"], total=total, grounding=grounding, fmt=fmt)
            out.append(Corruption(name, obj, expected, description))

        term = base_terms[pick]
        weight, distance = term["weight"], term["distance"]
        wrong_weight = rng.choice([v for v in range(WEIGHT_RANGE[0], WEIGHT_RANGE[1] + 1) if v != weight])
        wrong_distance = rng.choice([v for v in range(DISTANCE_RANGE[0], DISTANCE_RANGE[1] + 1) if v != distance])

        obj, _, terms = fresh()
        terms[pick].update(weight=wrong_weight, product=wrong_weight * distance)
        consistent_answer(obj)
        add("wrong_weight_consistent", obj, [WEIGHT, PRODUCT], "unchanged",
            "one weight misread; product and answer follow the misread value")

        obj, _, terms = fresh()
        terms[pick].update(distance=wrong_distance, product=weight * wrong_distance)
        consistent_answer(obj)
        add("wrong_distance_consistent", obj, [DISTANCE, PRODUCT], "unchanged",
            "one distance misread; product and answer follow the misread value")

        obj, _, terms = fresh()
        terms[pick]["weight"] = wrong_weight
        add("wrong_weight_answer_kept", obj, [WEIGHT, PRODUCT, PRODUCT_LOCAL], "unchanged",
            "one weight misread while product and answer stay right (correct answer, wrong certificate)")

        obj, _, terms = fresh()
        terms[pick]["product"] += 1
        add("wrong_product_answer_kept", obj, [PRODUCT, PRODUCT_LOCAL, ANSWER_LOCAL], "unchanged",
            "right operands, product off by one, answer unchanged")

        obj, _, terms = fresh()
        terms[pick]["product"] += 1
        consistent_answer(obj)
        add("wrong_product_consistent", obj, [PRODUCT, PRODUCT_LOCAL], "unchanged",
            "right operands, product off by one, answer is the sum of the claimed products")

        obj, _, _ = fresh()
        obj["answer"] = total + 1
        add("answer_sum_slip", obj, [ANSWER_LOCAL], "unchanged",
            "certificate fully right, answer is not the sum of the products (answer_inconsistent)")

        obj, annotation, terms = fresh()
        del annotation[pick], terms[pick]
        for index, item in enumerate(terms):
            item["annotation_index"] = index
        consistent_answer(obj)
        add("omit_term", obj, all_families, "fail", "one term and its box omitted; answer re-summed")

        obj, _, terms = fresh()
        del terms[pick]
        consistent_answer(obj)
        add("omit_term_keep_box", obj, all_families + [REF_LOCAL], "unchanged",
            "one term omitted while its box stays in the annotation")

        obj, _, terms = fresh()
        terms.append(copy.deepcopy(terms[pick]))
        consistent_answer(obj)
        add("duplicate_term", obj, all_families + [REF_LOCAL], "unchanged",
            "one term repeated with the same annotation_index")

        obj, annotation, terms = fresh()
        annotation.append(list(annotation[terms[pick]["annotation_index"]]))
        terms.append(dict(copy.deepcopy(terms[pick]), annotation_index=len(annotation) - 1))
        consistent_answer(obj)
        add("duplicate_box_term", obj, all_families, "fail",
            "a second box on the same block with its own term (one-to-one dedupe)")

        obj, _, terms = fresh()
        terms[pick]["annotation_index"] = -1
        add("negative_index", obj, all_families + [REF_LOCAL], "unchanged",
            "annotation_index -1 (never Python negative indexing)", fmt="fail")

        obj, annotation, terms = fresh()
        terms[pick]["annotation_index"] = len(annotation)
        add("index_out_of_range", obj, all_families + [REF_LOCAL], "unchanged",
            "annotation_index == len(annotation)", fmt="fail")

        obj, annotation, _ = fresh()
        for box in annotation:
            height = box[3] - box[1]
            box[1], box[3] = box[1] - height - 10.0, box[3] - height - 10.0
        add("boxes_shifted_off_target", obj, all_families, "fail",
            "every box moved up by its height + 10 px (no weight block there)")

        obj, annotation, _ = fresh()
        for box in annotation:
            box[0], box[2] = box[0] + 15.0, box[2] + 15.0
        add("boxes_shifted_within_tolerance", obj, [], "fail",
            "every box moved 15 px right: IoU about 0.59, still resolves (G drops, M does not)")

        distractors = [block for block in blocks if not block["relevant"]]
        if distractors:
            other = distractors[rng.randrange(len(distractors))]
            values = {"weight": other["weight"], "distance": other["distance"],
                      "product": other["weight"] * other["distance"]}

            obj, annotation, terms = fresh()
            annotation.append([float(v) for v in other["bbox"]])
            terms.append(dict(annotation_index=len(annotation) - 1, **values))
            consistent_answer(obj)
            add("wrong_side_extra_term", obj, all_families, "fail",
                "an extra term for a distractor-side block, read correctly")

            obj, annotation, terms = fresh()
            annotation[terms[pick]["annotation_index"]] = [float(v) for v in other["bbox"]]
            terms[pick].update(values)
            consistent_answer(obj)
            add("wrong_side_swap", obj, all_families, "fail",
                "one relevant block replaced by a distractor-side block, read correctly")

        if r >= 2:
            i, j = sorted(rng.sample(range(r), 2))
            obj, _, terms = fresh()
            terms[i]["annotation_index"], terms[j]["annotation_index"] = (
                terms[j]["annotation_index"], terms[i]["annotation_index"]
            )
            fail = [DISTANCE, PRODUCT] + ([WEIGHT] if base_terms[i]["weight"] != base_terms[j]["weight"] else [])
            add("swap_annotation_index", obj, fail, "unchanged",
                "two terms swap their boxes: still a bijection, but each read now belongs to the other block")

        obj, _, terms = fresh()
        terms[pick]["weight"] = True
        add("bool_weight", obj, all_families + [REF_LOCAL, PRODUCT_LOCAL], "unchanged",
            "weight given as a JSON boolean (element failure)", fmt="fail")

        obj, _, terms = fresh()
        terms[pick]["weight"] = float(weight)
        add("integral_float_weight", obj, [], "unchanged",
            "weight given as an integral float: scored by value, F=0", fmt="fail")

        obj = {"annotation": [], "certificate": {"kind": self.kind, "terms": []}, "answer": 0}
        add("empty_certificate", obj, all_families, "fail", "no boxes and no terms (the oracle set is never empty)")
        return out

def create_adapter() -> Adapter:
    return LeverWeightedSumAdapter()


__all__ = [
    "HELDOUT_CELLS",
    "LeverWeightedSumAdapter",
    "TASK_ID",
    "TRAIN_CELLS",
    "create_adapter",
]
