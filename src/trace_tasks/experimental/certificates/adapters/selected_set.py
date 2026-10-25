"""``selected_set.chess_piece_count``: the control certificate adapter (DESIGN 3.1).

Tasks ``task_games__chess__colored_piece_kind_count`` and ``task_games__chess__piece_kind_count``
ask how many pieces of one kind (and, in the colored task, one color) are on a partial chess
board. The certificate lists one member per annotation box with the piece's color and kind,
plus the member count:

    {"kind": "selected_set",
     "members": [{"annotation_index": 0, "color": "white", "piece": "pawn"}, ...],
     "count": 3}

The oracle is read from the persisted scene (``scene_ir.entities[type == "chess_piece"]`` and
``scene_ir.relations``); member boxes are bound to pieces with the shared identity rule (D4),
never by attribute. In the kind-only task ``color`` is a free attribute: it is a read checked
by ``ss.attr.truth`` and never gates membership. This adapter largely re-expresses G, so it is
a plumbing control and is excluded from the go/no-go decision (DESIGN 8, 10).
"""

from __future__ import annotations

import copy
import math
import random
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..checks import (
    Binding,
    bind_elements,
    bound_exactly_once,
    counted_truth,
    gated_check,
    indices_bijective,
    local_check,
)
from ..geometry import AnnotationView
from ..identity import CandidateSet
from ..parsing import INDEX, INT, CertificateView, enum
from ..types import (
    CheckResult,
    Corruption,
    QuarantineError,
    TraceRecord,
    canonical_digest,
)
from .base import Adapter, PublicContext, SftObject

COLORED_TASK = "task_games__chess__colored_piece_kind_count"
KIND_TASK = "task_games__chess__piece_kind_count"
PIECE_COLORS: Tuple[str, ...] = ("white", "black")
PIECE_KINDS: Tuple[str, ...] = ("pawn", "knight", "bishop", "rook", "queen", "king")

# DESIGN 6: the program (predicate structure) is the held-out axis; all styles are trained.
TASK_CELLS: Dict[str, str] = {
    COLORED_TASK: "colored_piece_kind_count",
    KIND_TASK: "piece_kind_count",
}
TRAIN_CELLS: Tuple[str, ...] = ("colored_piece_kind_count",)
HELDOUT_CELLS: Dict[str, str] = {"piece_kind_count": "program"}

SET, ATTR, COUNT = "ss.set.truth", "ss.attr.truth", "ss.count.truth"
REF_LOCAL = "ss.ref.local"
PRED_LOCAL = "ss.pred.local"
COUNT_LOCAL = "ss.count.local"
ANSWER_LOCAL = "ss.answer.local"

_BOX_TOLERANCE = 1e-6
# Off-target shift, as a fraction of the box width. Piece boxes are 0.76 of their cell, so a
# 0.6-width shift moves the center out of the piece and into the gap before the next piece.
_OFF_TARGET_SHIFT = 0.6


def _box(value: Any) -> Optional[List[float]]:
    """Four finite non-bool numbers with positive area, else None."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return None
    box = [float(item) for item in value]
    if not all(math.isfinite(item) for item in box) or box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _same_box(left: Sequence[float], right: Sequence[float]) -> bool:
    if len(left) != 4 or len(right) != 4:
        return False
    return all(abs(float(a) - float(b)) <= _BOX_TOLERANCE for a, b in zip(left, right))


def _overlap(left: Sequence[float], right: Sequence[float]) -> bool:
    width = min(left[2], right[2]) - max(left[0], right[0])
    height = min(left[3], right[3]) - max(left[1], right[1])
    return width > 0 and height > 0


def _satisfies(predicate: Mapping[str, Any], color: Any, piece: Any) -> bool:
    """The public prompt predicate: the kind, and the color only when the task names one."""
    if piece != predicate["piece"]:
        return False
    return predicate["color"] is None or color == predicate["color"]


def _read_scene(record: TraceRecord) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    """``(pieces, piece ids, predicate)`` from the persisted scene, validated.

    Pieces keep entity order (``oid`` = position); entity ids stay inside the exporter.
    """
    if record.task not in TASK_CELLS:
        raise QuarantineError("unsupported_task", record.task)
    entities = record.scene_ir.get("entities")
    relations = record.scene_ir.get("relations")
    if not isinstance(entities, list) or not isinstance(relations, Mapping):
        raise QuarantineError("missing_scene_ir", "scene_ir.entities / scene_ir.relations")
    pieces: List[Dict[str, Any]] = []
    ids: List[str] = []
    cells = set()
    for entity in entities:
        if not isinstance(entity, Mapping) or entity.get("type") != "chess_piece":
            continue
        piece_id, color, kind = entity.get("id"), entity.get("color"), entity.get("kind")
        row, col = entity.get("row"), entity.get("col")
        if color not in PIECE_COLORS or kind not in PIECE_KINDS:
            raise QuarantineError("bad_piece_attribute", f"{piece_id}: {color!r} {kind!r}")
        if type(row) is not int or type(col) is not int or row < 0 or col < 0:
            raise QuarantineError("bad_piece_cell", str(piece_id))
        bbox = _box(entity.get("bbox_px"))
        if bbox is None:
            raise QuarantineError("bad_piece_bbox", str(piece_id))
        if not isinstance(piece_id, str) or piece_id in ids or (row, col) in cells:
            raise QuarantineError("duplicate_piece", str(piece_id))
        ids.append(piece_id)
        cells.add((row, col))
        pieces.append(
            {
                "oid": len(pieces),
                "bbox": bbox,
                "piece": kind,
                "color": color,
                "row": row,
                "col": col,
                "selected": False,
            }
        )

    kind = relations.get("target_piece_kind")
    color = relations.get("target_piece_color")
    if kind not in PIECE_KINDS:
        raise QuarantineError("bad_predicate", f"target_piece_kind {kind!r}")
    colored = record.task == COLORED_TASK
    if (colored and color not in PIECE_COLORS) or (not colored and color not in ("", None)):
        raise QuarantineError("predicate_variant_mismatch", f"{record.task}: target_piece_color {color!r}")
    return pieces, ids, {"piece": kind, "color": color or None}


class SelectedSetAdapter(Adapter):
    """Chess piece counting as a selected set (control adapter, DESIGN 3.1)."""

    adapter_id = "selected_set.chess_piece_count"
    kind = "selected_set"
    version = "1.0.0"
    schema_id = "cert.selected_set.v1"
    verifier_version = "1.1.0"  # 1.1.0: annotation boxes count as selection claims
    supported = {COLORED_TASK: ("single",), KIND_TASK: ("single",)}

    annotation_type = "bbox_set"
    list_key = "members"
    list_cap = 64
    cert_fields = {"count": INT}
    element_fields = {
        "annotation_index": INDEX,
        "color": enum(*PIECE_COLORS),
        "piece": enum(*PIECE_KINDS),
    }

    m_families = (SET, ATTR, COUNT)
    family_stages = {SET: "grounding", ATTR: "operand", COUNT: "computation"}
    local_check_ids = (REF_LOCAL, PRED_LOCAL, COUNT_LOCAL, ANSWER_LOCAL)

    train_cells = TRAIN_CELLS
    heldout_cells = HELDOUT_CELLS

    # ======================================================================================
    # export side

    def build_oracle(self, record: TraceRecord) -> Dict[str, Any]:
        """``{"predicate", "pieces", "selected"}`` (DESIGN 3.1) with every export invariant checked.

        ``selected`` keeps ``annotation_entity_ids`` order (row-major), which is the SFT order.
        Each piece also carries its ``row``/``col`` (candidate keys and semantic key).
        """
        pieces, ids, predicate = _read_scene(record)
        if record.query_id not in self.supported[record.task]:
            raise QuarantineError("unsupported_query", record.query_id)
        width, height = record.source_size()
        for left, piece in enumerate(pieces):
            box = piece["bbox"]
            if box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
                raise QuarantineError("bbox_out_of_image", ids[left])
            for right in range(left + 1, len(pieces)):
                if _overlap(box, pieces[right]["bbox"]):
                    raise QuarantineError("pieces_overlap", f"{ids[left]} / {ids[right]}")

        entity_ids = record.scene_ir["relations"].get("annotation_entity_ids")
        if not isinstance(entity_ids, list) or not all(isinstance(i, str) for i in entity_ids):
            raise QuarantineError("missing_annotation_entity_ids")
        if len(set(entity_ids)) != len(entity_ids):
            raise QuarantineError("duplicate_annotation_entity")
        position = {piece_id: oid for oid, piece_id in enumerate(ids)}
        unknown = [item for item in entity_ids if item not in position]
        if unknown:
            raise QuarantineError("unknown_annotation_entity", ", ".join(unknown))
        selected = [position[item] for item in entity_ids]
        matching = {p["oid"] for p in pieces if _satisfies(predicate, p["color"], p["piece"])}
        if matching != set(selected):
            raise QuarantineError("predicate_mismatch", "annotation_entity_ids != predicate matches")
        for oid in selected:
            pieces[oid]["selected"] = True

        # Cross-checks only: the labels never derive the oracle.
        answer_gt = record.answer_gt
        if answer_gt.get("type") != "integer" or type(answer_gt.get("value")) is not int:
            raise QuarantineError("bad_answer_gt", repr(answer_gt))
        if answer_gt["value"] != len(selected):
            raise QuarantineError("answer_mismatch", f"answer_gt {answer_gt['value']} != {len(selected)}")
        target_answer = record.execution_trace.get("target_answer")
        if target_answer is not None and target_answer != len(selected):
            raise QuarantineError("answer_mismatch", f"execution_trace.target_answer {target_answer!r}")
        annotation_gt = record.annotation_gt
        boxes = annotation_gt.get("value")
        if annotation_gt.get("type") != self.annotation_type or not isinstance(boxes, list):
            raise QuarantineError("bad_annotation_gt", repr(annotation_gt.get("type")))
        if len(boxes) != len(selected) or not all(
            _same_box(box, pieces[oid]["bbox"]) for box, oid in zip(boxes, selected)
        ):
            raise QuarantineError("annotation_mismatch", "annotation_gt != selected piece boxes in order")
        return {"predicate": predicate, "pieces": pieces, "selected": selected}

    def split_cell(self, record: TraceRecord) -> str:
        """``colored_piece_kind_count`` (train/E1) or ``piece_kind_count`` (E2 program holdout)."""
        _read_scene(record)
        return TASK_CELLS[record.task]

    def semantic_key(self, record: TraceRecord) -> str:
        """DESIGN 6: the target predicate and the sorted pieces (row, col, color, kind)."""
        pieces, _, predicate = _read_scene(record)
        board = sorted([p["row"], p["col"], p["color"], p["piece"]] for p in pieces)
        return canonical_digest({"adapter": "selected_set", "target": predicate, "pieces": board})

    def sft_certificate(self, oracle: Mapping[str, Any], rng_seed: int) -> SftObject:
        """Members in ``annotation_gt`` order. The witness is unique, so ``rng_seed`` is unused."""
        chosen = [oracle["pieces"][oid] for oid in oracle["selected"]]
        annotation = [list(piece["bbox"]) for piece in chosen]
        members = [
            {"annotation_index": index, "color": piece["color"], "piece": piece["piece"]}
            for index, piece in enumerate(chosen)
        ]
        return annotation, {"kind": self.kind, "members": members, "count": len(members)}, len(members)

    # ======================================================================================
    # verification

    def candidates(self, oracle: Mapping[str, Any]) -> Mapping[str, CandidateSet]:
        """Every piece on the board (piece boxes in oid order); ``selected`` is the oracle set."""
        pieces = oracle["pieces"]
        return {
            "": CandidateSet(
                boxes=tuple(tuple(float(v) for v in piece["bbox"]) for piece in pieces),
                selected=tuple(int(oid) for oid in oracle["selected"]),
                keys=tuple((int(piece["row"]), int(piece["col"])) for piece in pieces),
            )
        }

    def family_applicable(self, oracle: Mapping[str, Any]) -> Mapping[str, bool]:
        return {SET: True, ATTR: len(oracle["selected"]) > 0, COUNT: True}

    def _bindings(
        self,
        cert: CertificateView,
        annotation: AnnotationView,
        oracle: Mapping[str, Any],
        ctx: PublicContext,
    ) -> Tuple[Tuple[int, ...], List[Binding]]:
        candidates = self.candidates(oracle)
        resolutions = self.resolve_role(annotation, "", oracle, ctx, candidates)
        return candidates[""].selected, bind_elements(cert.elements, resolutions)

    def checks(
        self,
        cert: CertificateView,
        annotation: AnnotationView,
        oracle: Mapping[str, Any],
        ctx: PublicContext,
    ) -> List[CheckResult]:
        """DESIGN 3.1 table. Attribute credit, like set credit, needs a target claimed by
        exactly one bound member, so repeated or duplicate claims never add credit."""
        pieces, predicate = oracle["pieces"], oracle["predicate"]
        selected, bindings = self._bindings(cert, annotation, oracle, ctx)
        k, n = len(selected), cert.n_claimed
        once = bound_exactly_once(bindings, selected)

        # Every annotation box is a claim that its piece is selected (review EX-5): an
        # unreferenced box (no member) raises the selection denominator like an extra member.
        n_annotation = annotation.size("")
        status = {"member_status": [binding.status for binding in bindings], "n_boxes": n_annotation}
        set_check = counted_truth(SET, len(once), max(n, n_annotation), k, detail=status)
        attr_hits = sum(
            1
            for target, position in once.items()
            if cert.elements[position].get("color") == pieces[target]["color"]
            and cert.elements[position].get("piece") == pieces[target]["piece"]
        )
        attr_check = counted_truth(ATTR, attr_hits, n, k, applicable=k > 0)
        count = cert.fields.get("count")
        count_check = gated_check(COUNT, [set_check], count is not None and count == n)

        predicate_ok = all(_satisfies(predicate, e.get("color"), e.get("piece")) for e in cert.elements)
        return [
            set_check,
            attr_check,
            count_check,
            local_check(REF_LOCAL, indices_bijective(cert.elements, n_annotation)),
            local_check(PRED_LOCAL, predicate_ok),
            local_check(COUNT_LOCAL, count is not None and count == n == n_annotation),
            local_check(ANSWER_LOCAL, count is not None and cert.answer == count),
        ]

    def diagnostics(
        self,
        cert: CertificateView,
        annotation: AnnotationView,
        oracle: Mapping[str, Any],
        ctx: PublicContext,
        checks: Sequence[CheckResult],
    ) -> Dict[str, Any]:
        """``ss.attr_read_acc``: attribute reads of every bound member, distractors included."""
        pieces = oracle["pieces"]
        _, bindings = self._bindings(cert, annotation, oracle, ctx)
        reads = [(cert.elements[b.position], pieces[b.candidate]) for b in bindings if b.bound]
        color = [element.get("color") == piece["color"] for element, piece in reads]
        kind = [element.get("piece") == piece["piece"] for element, piece in reads]

        def rate(hits: Sequence[bool]) -> Optional[float]:
            return sum(hits) / len(hits) if hits else None

        return {
            "ss.attr_read_acc": rate([c and p for c, p in zip(color, kind)]),
            "ss.color_read_acc": rate(color),
            "ss.piece_read_acc": rate(kind),
            "n_bound": len(reads),
            "n_bound_distractor": sum(1 for _, piece in reads if not piece["selected"]),
            "member_status": dict(sorted(Counter(b.status for b in bindings).items())),
        }

    # ======================================================================================
    # counterexamples

    def corruptions(
        self,
        oracle: Mapping[str, Any],
        sft_obj: Mapping[str, Any],
        rng_seed: int,
    ) -> List[Corruption]:
        """DESIGN 9 corruptions of the SFT object, each listing every check it must fail.

        Checks not listed as ``"fail"`` stay at 1 (the tests assert the exact failing set).
        ``ss.attr.truth`` is dropped from the list when it is not applicable (k = 0).
        """
        rng = random.Random(int(rng_seed))
        predicate, pieces = oracle["predicate"], oracle["pieces"]
        boxes = [list(box) for box in sft_obj["annotation"]]
        members = [dict(member) for member in sft_obj["certificate"]["members"]]
        k = len(members)
        out: List[Corruption] = []

        def add(
            name: str,
            annotation: Any,
            claimed: Any,
            count: Any,
            answer: Any,
            fails: Sequence[str],
            description: str,
            **signals: str,
        ) -> None:
            """Append one corruption; ``signals`` are the A/G/M/F expectations."""
            certificate = {"kind": self.kind, "members": copy.deepcopy(claimed), "count": count}
            obj = {"annotation": copy.deepcopy(annotation), "certificate": certificate, "answer": answer}
            expected: Dict[str, Any] = {c: "fail" for c in fails if c != ATTR or k > 0}
            expected.update(signals)
            out.append(Corruption(name, obj, expected, description))

        add("wrong_count", boxes, members, k + 1, k, [COUNT, COUNT_LOCAL, ANSWER_LOCAL],
            "count off by one; members and answer right (wrong arithmetic on right values)",
            A="pass", G="unchanged", F="pass", M="fail")
        add("wrong_count_and_answer", boxes, members, k + 1, k + 1, [COUNT, COUNT_LOCAL],
            "count and answer agree with each other but not with the members",
            A="fail", G="unchanged", F="pass", M="fail")
        add("answer_inconsistent", boxes, members, k, k + 1, [ANSWER_LOCAL],
            "certificate right, answer disagrees with its count",
            A="fail", G="unchanged", F="pass", M="pass")

        distractors = [piece for piece in pieces if not piece["selected"]]
        if distractors:
            extra = rng.choice(distractors)
            claim = {"annotation_index": k, "color": extra["color"], "piece": extra["piece"]}
            fails = [SET, ATTR, COUNT]
            if not _satisfies(predicate, extra["color"], extra["piece"]):
                fails.append(PRED_LOCAL)
            add("extra_distractor", boxes + [list(extra["bbox"])], members + [claim], k + 1, k + 1,
                fails, "a correctly described non-selected piece added (irrelevant true statement)",
                A="fail", G="fail", F="pass", M="fail")

        if k == 0:
            phantom = {"color": predicate["color"] or "white", "piece": predicate["piece"]}
            for name, index in (("index_out_of_range", 0), ("negative_index", -1)):
                add(name, [], [dict(phantom, annotation_index=index)], 1, 1,
                    [SET, COUNT, REF_LOCAL, COUNT_LOCAL],
                    "a member pointing at no annotation box on an empty scene",
                    A="fail", G="unchanged", F="fail", M="fail")
            return out

        j = rng.randrange(k)
        misread = [dict(member) for member in members]
        misread[j]["piece"] = rng.choice([p for p in PIECE_KINDS if p != members[j]["piece"]])
        add("wrong_piece", boxes, misread, k, k, [ATTR, PRED_LOCAL],
            "one member's kind misread, count consistent (wrong value, consistent arithmetic)",
            A="pass", G="unchanged", F="pass", M="fail")
        misread = [dict(member) for member in members]
        misread[j]["color"] = "black" if members[j]["color"] == "white" else "white"
        add("wrong_color", boxes, misread, k, k,
            [ATTR] + ([PRED_LOCAL] if predicate["color"] is not None else []),
            "one member's color misread (a free attribute in the kind-only task)",
            A="pass", G="unchanged", F="pass", M="fail")

        kept = [index for index in range(k) if index != j]
        reindexed = [dict(members[old], annotation_index=new) for new, old in enumerate(kept)]
        add("omit_member", [boxes[index] for index in kept], reindexed, k - 1, k - 1,
            [SET, ATTR, COUNT], "one selected piece left out of annotation and members",
            A="fail", G="fail", F="pass", M="fail")
        add("omit_member_keep_box", boxes, [members[index] for index in kept], k - 1, k - 1,
            [SET, ATTR, COUNT, REF_LOCAL, COUNT_LOCAL], "box kept but its member dropped (G is blind)",
            A="fail", G="unchanged", F="pass", M="fail")
        add("duplicate_member", boxes, members + [dict(members[j])], k, k,
            [SET, ATTR, COUNT, REF_LOCAL, COUNT_LOCAL], "one member repeated with the same index",
            A="pass", G="unchanged", F="pass", M="fail")
        add("duplicate_box", boxes + [list(boxes[j])], members + [dict(members[j], annotation_index=k)],
            k + 1, k + 1, [SET, ATTR, COUNT], "one selected piece boxed and claimed twice",
            A="fail", G="fail", F="pass", M="fail")
        for name, index in (("negative_index", -1), ("index_out_of_range", k)):
            bad = [dict(member) for member in members]
            bad[j]["annotation_index"] = index
            add(name, boxes, bad, k, k, [SET, ATTR, COUNT, REF_LOCAL],
                "annotation_index outside [0, len(annotation)): an invalid member, no Python indexing",
                A="pass", G="unchanged", F="fail", M="fail")
        shifted = []
        for x0, y0, x1, y1 in boxes:
            dx = _OFF_TARGET_SHIFT * (x1 - x0)
            shifted.append([x0 + dx, y0, x1 + dx, y1])
        add("boxes_shifted", shifted, members, k, k, [SET, ATTR, COUNT],
            "every box moved off its piece by 0.6 of its width; certificate otherwise right",
            A="pass", G="fail", F="pass", M="fail")
        add("empty_claim", [], [], 0, 0, [SET, ATTR, COUNT], "empty set claimed on a non-empty scene",
            A="fail", G="fail", F="pass", M="fail")
        return out

def create_adapter() -> SelectedSetAdapter:
    return SelectedSetAdapter()


__all__ = [
    "COLORED_TASK",
    "HELDOUT_CELLS",
    "KIND_TASK",
    "PIECE_COLORS",
    "PIECE_KINDS",
    "SelectedSetAdapter",
    "TASK_CELLS",
    "TRAIN_CELLS",
    "create_adapter",
]
