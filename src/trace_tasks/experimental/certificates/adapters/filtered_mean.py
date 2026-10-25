"""``filtered_mean.table_filtered_column_mean`` (DESIGN 3.2).

Task ``task_charts__table__filtered_column_mean`` asks for the mean of a target column T over
the data rows whose filter column F satisfies a public predicate: strict ``>`` (above), strict
``<`` (below) or an inclusive, possibly degenerate interval. The certificate claims one row per
selected row; ``annotation_index`` i binds the row to **both** ``annotation.filter_cells[i]`` and
``annotation.target_cells[i]``::

    {"kind": "filtered_mean",
     "rows": [{"annotation_index": 0, "filter_value": 24, "target_value": 9}, ...],
     "sum": 105, "count": 7, "mean": 15}

A row is *bound* when its filter box resolves (identity rule, DESIGN 2.3) to a value cell of
column F, its target box to a value cell of column T, and both cells sit in the same data row.
The candidates are the value cells of every column, so a box in another column still resolves
and is logged as ``wrong_column``. Rows are bound by geometry only, never by value or label
(values repeat within a table in most records).

The oracle is built from the persisted trace record: ``scene_ir.entities`` (``table_cell``
value cells with ``bbox_px`` and ``attrs``) and ``scene_ir.relations`` (columns, predicate,
selected rows), cross-checked against ``execution_trace.values_by_row``, ``answer_gt`` and
``annotation_gt``. ``answer_gt`` is never used to derive an oracle value.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..checks import (
    Binding,
    bind_elements,
    counted_truth,
    gated_check,
    indices_bijective,
    local_check,
)
from ..geometry import AnnotationView
from ..identity import CandidateSet, Resolution, dedupe, resolve_box
from ..parsing import INDEX, INT, CertificateView
from ..types import CheckResult, Corruption, QuarantineError, TraceRecord, canonical_digest
from .base import Adapter, PublicContext, SftObject

TASK_ID = "task_charts__table__filtered_column_mean"
QUERY_FILTERS: Dict[str, str] = {
    "above_threshold_filtered_mean": "above_threshold",
    "below_threshold_filtered_mean": "below_threshold",
    "interval_filtered_mean": "in_interval",
}
QUERY_SHORT: Dict[str, str] = {
    "above_threshold_filtered_mean": "above",
    "below_threshold_filtered_mean": "below",
    "interval_filtered_mean": "interval",
}
SCENE_VARIANTS = ("spreadsheet", "zebra", "ledger", "card_table")

# DESIGN 6: train / E1 cells and the E2 held-out cells with their holdout type.
TRAIN_CELLS: Tuple[str, ...] = tuple(
    f"{query}|{variant}" for query in ("above", "below") for variant in ("spreadsheet", "zebra", "card_table")
)
HELD_OUT_CELLS: Dict[str, str] = {
    "above|ledger": "render",
    "below|ledger": "render",
    "interval|spreadsheet": "program",
    "interval|zebra": "program",
    "interval|card_table": "program",
    "interval|ledger": "both",
}

FILTER_ROLE = "filter_cells"
TARGET_ROLE = "target_cells"

ROWS = "fm.rows.truth"
FILTER_VALUE = "fm.filter_value.truth"
TARGET_VALUE = "fm.target_value.truth"
COUNT = "fm.count.truth"
SUM = "fm.sum.truth"
MEAN = "fm.mean.truth"
REF_LOCAL = "fm.ref.local"
PRED_LOCAL = "fm.pred.local"
COUNT_LOCAL = "fm.count.local"
SUM_LOCAL = "fm.sum.local"
MEAN_LOCAL = "fm.mean.local"
ANSWER_LOCAL = "fm.answer.local"

_ANNOTATION_TOL = 1e-6


# ------------------------------------------------------------------------------------------
# predicate


def predicate_holds(predicate: Mapping[str, Any], value: Any) -> bool:
    """The public row filter: ``gt``/``lt`` are strict, ``in`` is inclusive on both ends."""
    if isinstance(value, bool) or not isinstance(value, int):
        return False
    op = predicate["op"]
    if op == "gt":
        return value > int(predicate["t"])
    if op == "lt":
        return value < int(predicate["t"])
    if op == "in":
        return int(predicate["lo"]) <= value <= int(predicate["hi"])
    raise ValueError(f"unknown predicate op {op!r}")


def holdout_type(cell: str) -> Optional[str]:
    """``None`` for a train/E1 cell, else ``"render"``, ``"program"`` or ``"both"`` (E2)."""
    if cell in HELD_OUT_CELLS:
        return HELD_OUT_CELLS[cell]
    if cell in TRAIN_CELLS:
        return None
    raise ValueError(f"unknown filtered_mean cell {cell!r}")


# ------------------------------------------------------------------------------------------
# persisted-record readers (QuarantineError on any data problem)


def _require(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if not isinstance(mapping, Mapping) or key not in mapping or mapping[key] is None:
        raise QuarantineError("missing_field", f"{where}.{key}")
    return mapping[key]


def _as_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise QuarantineError("bad_field", f"{where} must be an integer, got {value!r}")
    return int(value)


def _as_box(value: Any, where: str) -> Tuple[float, float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise QuarantineError("bad_cell_bbox", where)
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value):
        raise QuarantineError("bad_cell_bbox", where)
    x0, y0, x1, y1 = (float(v) for v in value)
    if not (x0 < x1 and y0 < y1):
        raise QuarantineError("bad_cell_bbox", where)
    return (x0, y0, x1, y1)


def _relations(record: TraceRecord) -> Mapping[str, Any]:
    relations = record.scene_ir.get("relations") if isinstance(record.scene_ir, Mapping) else None
    if not isinstance(relations, Mapping):
        raise QuarantineError("missing_field", "scene_ir.relations")
    return relations


def _predicate(relations: Mapping[str, Any]) -> Dict[str, Any]:
    variant = _require(relations, "filter_variant", "scene_ir.relations")
    if variant == "above_threshold":
        return {"op": "gt", "t": _as_int(_require(relations, "threshold_value", "scene_ir.relations"), "threshold_value")}
    if variant == "below_threshold":
        return {"op": "lt", "t": _as_int(_require(relations, "threshold_value", "scene_ir.relations"), "threshold_value")}
    if variant == "in_interval":
        low = _as_int(_require(relations, "interval_min", "scene_ir.relations"), "interval_min")
        high = _as_int(_require(relations, "interval_max", "scene_ir.relations"), "interval_max")
        if low > high:
            raise QuarantineError("bad_predicate", f"interval [{low}, {high}] is empty")
        return {"op": "in", "lo": low, "hi": high}
    raise QuarantineError("bad_predicate", f"unknown filter_variant {variant!r}")


def _scene_variant(record: TraceRecord) -> str:
    for source in (_relations(record), record.execution_trace, record.render_spec):
        if isinstance(source, Mapping) and isinstance(source.get("scene_variant"), str):
            variant = source["scene_variant"]
            if variant not in SCENE_VARIANTS:
                raise QuarantineError("unknown_scene_variant", repr(variant))
            return variant
    raise QuarantineError("missing_field", "scene_variant")


def _check_query(record: TraceRecord) -> str:
    if record.task != TASK_ID or record.query_id not in QUERY_FILTERS:
        raise QuarantineError("unsupported_query", f"{record.task}/{record.query_id}")
    return record.query_id


def _overlap(a: Sequence[float], b: Sequence[float]) -> bool:
    return min(a[2], b[2]) - max(a[0], b[0]) > 0.0 and min(a[3], b[3]) - max(a[1], b[1]) > 0.0


def _same_box(a: Any, b: Sequence[float]) -> bool:
    try:
        return len(a) == 4 and all(abs(float(x) - float(y)) <= _ANNOTATION_TOL for x, y in zip(a, b))
    except (TypeError, ValueError):
        return False


# ------------------------------------------------------------------------------------------
# adapter


class FilteredMeanAdapter(Adapter):
    """Certificate adapter for the table filtered-column mean (DESIGN 3.2)."""

    adapter_id = "filtered_mean.table_filtered_column_mean"
    kind = "filtered_mean"
    version = "1.0.0"
    schema_id = "cert.filtered_mean.v1"
    verifier_version = "1.1.0"  # 1.1.0: annotation boxes count as row claims
    supported = {TASK_ID: tuple(QUERY_FILTERS)}
    instance_versions = ("v0",)

    annotation_type = "bbox_set_map"
    annotation_roles = (FILTER_ROLE, TARGET_ROLE)
    list_key = "rows"
    list_cap = 16
    cert_fields = {"sum": INT, "count": INT, "mean": INT}
    element_fields = {"annotation_index": INDEX, "filter_value": INT, "target_value": INT}

    m_families = (ROWS, FILTER_VALUE, TARGET_VALUE, COUNT, SUM, MEAN)
    family_stages = {
        ROWS: "grounding",
        FILTER_VALUE: "operand",
        TARGET_VALUE: "operand",
        COUNT: "computation",
        SUM: "computation",
        MEAN: "computation",
    }
    local_check_ids = (REF_LOCAL, PRED_LOCAL, COUNT_LOCAL, SUM_LOCAL, MEAN_LOCAL, ANSWER_LOCAL)

    train_cells = TRAIN_CELLS
    heldout_cells = HELD_OUT_CELLS

    # ======================================================================================
    # export side

    def build_oracle(self, record: TraceRecord) -> Dict[str, Any]:
        query_id = _check_query(record)
        relations = _relations(record)
        if relations.get("filter_variant") != QUERY_FILTERS[query_id]:
            raise QuarantineError(
                "query_variant_mismatch", f"{query_id} vs filter_variant {relations.get('filter_variant')!r}"
            )
        where = "scene_ir.relations"
        n_rows = _as_int(_require(relations, "row_count", where), "row_count")
        n_cols = _as_int(_require(relations, "numeric_column_count", where), "numeric_column_count")
        f_col = _as_int(_require(relations, "filter_column_index", where), "filter_column_index")
        t_col = _as_int(_require(relations, "target_column_index", where), "target_column_index")
        f_name = str(_require(relations, "filter_column", where))
        t_name = str(_require(relations, "target_column", where))
        if n_rows < 1 or n_cols < 2 or not (0 <= f_col < n_cols and 0 <= t_col < n_cols) or f_col == t_col:
            raise QuarantineError("bad_columns", f"rows={n_rows} cols={n_cols} F={f_col} T={t_col}")
        predicate = _predicate(relations)

        boxes, values, headers, labels = self._value_grid(record, n_rows, n_cols)
        if headers[f_col] != f_name or headers[t_col] != t_name:
            raise QuarantineError("column_header_mismatch", f"{headers[f_col]}/{headers[t_col]} vs {f_name}/{t_name}")
        self._check_values_by_row(record, values, headers, labels)
        self._check_geometry(record, boxes)

        selected = [r for r in range(n_rows) if predicate_holds(predicate, values[r][f_col])]
        claimed = _require(relations, "selected_row_indices", where)
        if not isinstance(claimed, list) or sorted(_as_int(v, "selected_row_indices") for v in claimed) != selected:
            raise QuarantineError("predicate_mismatch", f"recomputed {selected} vs selected_row_indices {claimed}")
        if not selected:
            raise QuarantineError("empty_selection", "no row satisfies the predicate")
        total = sum(values[r][t_col] for r in selected)
        count = len(selected)
        if total % count != 0:
            raise QuarantineError("sum_not_divisible", f"sum {total} over {count} rows")
        mean = total // count

        answer_gt = record.answer_gt
        if answer_gt.get("type") != "integer" or isinstance(answer_gt.get("value"), bool) or not isinstance(
            answer_gt.get("value"), int
        ):
            raise QuarantineError("answer_type", repr(answer_gt))
        if answer_gt["value"] != mean:
            raise QuarantineError("mean_mismatch", f"oracle mean {mean} != answer_gt {answer_gt['value']}")
        self._check_annotation(record, [boxes[r][f_col] for r in selected], [boxes[r][t_col] for r in selected])

        chosen = set(selected)
        rows = [
            {
                "r": r,
                "label": labels[r],
                "filter_cell": list(boxes[r][f_col]),
                "target_cell": list(boxes[r][t_col]),
                "filter_value": values[r][f_col],
                "target_value": values[r][t_col],
                "selected": r in chosen,
            }
            for r in range(n_rows)
        ]
        return {
            "filter_column": f_name,
            "target_column": t_name,
            "filter_col": f_col,
            "target_col": t_col,
            "predicate": predicate,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "cells": [[list(boxes[r][c]) for c in range(n_cols)] for r in range(n_rows)],
            "rows": rows,
            "sum": total,
            "count": count,
            "mean": mean,
        }

    @staticmethod
    def _value_grid(
        record: TraceRecord, n_rows: int, n_cols: int
    ) -> Tuple[List[List[Any]], List[List[int]], List[str], List[str]]:
        """Boxes, values, column headers and row labels of every value cell (0-based r, c)."""
        entities = record.scene_ir.get("entities")
        if not isinstance(entities, list):
            raise QuarantineError("missing_field", "scene_ir.entities")
        boxes: List[List[Any]] = [[None] * n_cols for _ in range(n_rows)]
        values: List[List[Any]] = [[None] * n_cols for _ in range(n_rows)]
        headers: List[Optional[str]] = [None] * n_cols
        labels: List[Optional[str]] = [None] * n_rows
        for entity in entities:
            if not isinstance(entity, Mapping) or entity.get("entity_type") != "table_cell":
                continue
            attrs = entity.get("attrs")
            if not isinstance(attrs, Mapping) or attrs.get("cell_role") != "value":
                continue
            where = f"scene_ir.entities[{entity.get('entity_id')}]"
            # Header row is row_index 0 and the Name column is column_index 0.
            r = _as_int(attrs.get("row_index"), f"{where}.row_index") - 1
            c = _as_int(attrs.get("column_index"), f"{where}.column_index") - 1
            if not (0 <= r < n_rows and 0 <= c < n_cols):
                raise QuarantineError("grid_out_of_range", f"{where} at ({r}, {c})")
            if boxes[r][c] is not None:
                raise QuarantineError("duplicate_cell", where)
            value = _as_int(attrs.get("value"), f"{where}.value")
            if attrs.get("text") != str(value):
                raise QuarantineError("text_value_mismatch", f"{where}: {attrs.get('text')!r} vs {value}")
            header, label = attrs.get("column_header"), attrs.get("row_label")
            if not isinstance(header, str) or not isinstance(label, str):
                raise QuarantineError("missing_field", f"{where}.column_header/row_label")
            if headers[c] not in (None, header) or labels[r] not in (None, label):
                raise QuarantineError("inconsistent_grid", where)
            headers[c], labels[r] = header, label
            boxes[r][c] = _as_box(entity.get("bbox_px"), f"{where}.bbox_px")
            values[r][c] = value
        if any(box is None for row in boxes for box in row):
            raise QuarantineError("incomplete_grid", f"expected {n_rows}x{n_cols} value cells")
        if len(set(headers)) != n_cols or len(set(labels)) != n_rows:
            raise QuarantineError("inconsistent_grid", "column headers and row labels must be unique")
        return boxes, values, list(headers), list(labels)  # type: ignore[arg-type]

    @staticmethod
    def _check_values_by_row(
        record: TraceRecord, values: List[List[int]], headers: List[str], labels: List[str]
    ) -> None:
        """Cross-check the scene cells against the persisted ``execution_trace`` table."""
        trace = record.execution_trace
        by_row = trace.get("values_by_row") if isinstance(trace, Mapping) else None
        if by_row is None:
            return
        trace_headers = trace.get("column_headers")
        if not isinstance(by_row, Mapping) or list(by_row) != labels or (
            trace_headers is not None and list(trace_headers) != headers
        ):
            raise QuarantineError("values_by_row_mismatch", "row labels or headers differ from scene_ir")
        for r, label in enumerate(labels):
            row = by_row[label]
            if not isinstance(row, Mapping) or [row.get(h) for h in headers] != values[r]:
                raise QuarantineError("values_by_row_mismatch", f"row {label!r}")

    @staticmethod
    def _check_geometry(record: TraceRecord, boxes: List[List[Any]]) -> None:
        """Value cells lie inside the image and are pairwise disjoint (unique identity, D4)."""
        width, height = record.source_size()
        flat = [box for row in boxes for box in row]
        for box in flat:
            if box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
                raise QuarantineError("cell_outside_image", repr(box))
        for i, a in enumerate(flat):
            for b in flat[i + 1 :]:
                if _overlap(a, b):
                    raise QuarantineError("cells_overlap", f"{a} / {b}")

    @staticmethod
    def _check_annotation(record: TraceRecord, filter_boxes: List[Any], target_boxes: List[Any]) -> None:
        """``annotation_gt`` must be the selected rows' F and T cells in ascending row order."""
        annotation = record.annotation_gt
        value = annotation.get("value")
        if annotation.get("type") != "bbox_set_map" or not isinstance(value, Mapping):
            raise QuarantineError("annotation_mismatch", "annotation_gt is not a bbox_set_map")
        if set(value) != {FILTER_ROLE, TARGET_ROLE}:
            raise QuarantineError("annotation_mismatch", f"annotation_gt keys {sorted(value)}")
        for role, expected in ((FILTER_ROLE, filter_boxes), (TARGET_ROLE, target_boxes)):
            given = value[role]
            if not isinstance(given, list) or len(given) != len(expected):
                raise QuarantineError("annotation_mismatch", f"{role} length")
            if not all(_same_box(a, b) for a, b in zip(given, expected)):
                raise QuarantineError("annotation_mismatch", f"{role} boxes differ from the scene cells")

    def split_cell(self, record: TraceRecord) -> str:
        """``"<above|below|interval>|<scene_variant>"`` (DESIGN 6); see ``HELD_OUT_CELLS``."""
        return f"{QUERY_SHORT[_check_query(record)]}|{_scene_variant(record)}"

    def semantic_key(self, record: TraceRecord) -> str:
        """DESIGN 6: query plus a hash of the column headers and ``values_by_row``."""
        query_id = _check_query(record)
        trace = record.execution_trace
        payload = {
            "task": record.task,
            "query_id": query_id,
            "column_headers": list(_require(trace, "column_headers", "execution_trace")),
            "values_by_row": _require(trace, "values_by_row", "execution_trace"),
        }
        return f"{self.kind}:{canonical_digest(payload)}"

    def sft_certificate(self, oracle: Mapping[str, Any], rng_seed: int) -> SftObject:
        """Selected rows in ascending data-row order; there is no witness choice to make."""
        chosen = [row for row in oracle["rows"] if row["selected"]]
        annotation = {
            FILTER_ROLE: [list(row["filter_cell"]) for row in chosen],
            TARGET_ROLE: [list(row["target_cell"]) for row in chosen],
        }
        certificate = {
            "kind": self.kind,
            "rows": [
                {"annotation_index": i, "filter_value": row["filter_value"], "target_value": row["target_value"]}
                for i, row in enumerate(chosen)
            ],
            "sum": int(oracle["sum"]),
            "count": int(oracle["count"]),
            "mean": int(oracle["mean"]),
        }
        return annotation, certificate, int(oracle["mean"])

    # ======================================================================================
    # verification

    def candidates(self, oracle: Mapping[str, Any]) -> Mapping[str, CandidateSet]:
        """Every value cell (row-major, position ``r * n_cols + c``) for both roles."""
        n_cols = int(oracle["n_cols"])
        cells = oracle["cells"]
        boxes = tuple(tuple(float(v) for v in cells[r][c]) for r in range(len(cells)) for c in range(n_cols))
        keys = tuple((r, c) for r in range(len(cells)) for c in range(n_cols))
        chosen = [row["r"] for row in oracle["rows"] if row["selected"]]
        f_col, t_col = int(oracle["filter_col"]), int(oracle["target_col"])
        return {
            FILTER_ROLE: CandidateSet(boxes, tuple(r * n_cols + f_col for r in chosen), keys),
            TARGET_ROLE: CandidateSet(boxes, tuple(r * n_cols + t_col for r in chosen), keys),
        }

    def _resolve_role(
        self,
        annotation: AnnotationView,
        role: str,
        candidate_set: CandidateSet,
        ctx: PublicContext,
    ) -> List[Resolution]:
        """The shared D4 rule (``identity.resolve_box`` + ``dedupe``) on a smaller candidate list.

        A table has up to 80 value cells, so the full scan dominated scoring time. A cell with
        zero-area intersection has IoU 0 and containment 0, cannot match, and cannot beat a
        positive-IoU cell as the logged best candidate; when nothing intersects, the full scan
        logs cell 0 with zero stats. Scanning only the intersecting cells (or cell 0) therefore
        returns the same ``Resolution`` field for field, for any thresholds.
        """
        boxes = candidate_set.boxes
        resolutions: List[Resolution] = []
        for index, box in enumerate(annotation.items(role)):
            near: List[int] = []
            if box is not None:  # normalized: positive area, so overlap <=> positive intersection
                bx0, by0, bx1, by1 = box
                near = [p for p, (x0, y0, x1, y1) in enumerate(boxes) if x0 < bx1 and bx0 < x1 and y0 < by1 and by0 < y1]
            subset = near or list(range(min(1, len(boxes))))
            quick = resolve_box(box, [boxes[p] for p in subset], ctx.identity, index=index)
            if near and quick.iou <= 0.0:  # float underflow only: keep the exact full scan
                resolutions.append(resolve_box(box, boxes, ctx.identity, index=index))
                continue
            resolutions.append(
                dataclasses.replace(
                    quick,
                    candidate=None if quick.candidate is None else subset[quick.candidate],
                    hits=tuple(subset[h] for h in quick.hits),
                    best_candidate=None if quick.best_candidate is None else subset[quick.best_candidate],
                )
            )
        return dedupe(resolutions)

    def row_bindings(
        self,
        cert: CertificateView,
        annotation: AnnotationView,
        oracle: Mapping[str, Any],
        ctx: PublicContext,
    ) -> List[Tuple[str, Optional[int]]]:
        """Per certificate row: ``(status, data row)``; status ``bound`` or why it is not
        (``invalid_element``, ``unresolved``, ``ambiguous``, ``duplicate``, ``invalid``,
        ``wrong_column``, ``wrong_pairing``)."""
        candidates = self.candidates(oracle)
        by_filter = bind_elements(cert.elements, self.resolve_role(annotation, FILTER_ROLE, oracle, ctx, candidates))
        by_target = bind_elements(cert.elements, self.resolve_role(annotation, TARGET_ROLE, oracle, ctx, candidates))
        n_cols, f_col, t_col = int(oracle["n_cols"]), int(oracle["filter_col"]), int(oracle["target_col"])
        out: List[Tuple[str, Optional[int]]] = []
        for f, t in zip(by_filter, by_target):
            out.append(_row_status(f, t, n_cols, f_col, t_col))
        return out

    def checks(
        self,
        cert: CertificateView,
        annotation: AnnotationView,
        oracle: Mapping[str, Any],
        ctx: PublicContext,
    ) -> List[CheckResult]:
        bindings = self.row_bindings(cert, annotation, oracle, ctx)
        rows = oracle["rows"]
        chosen = [row["r"] for row in rows if row["selected"]]
        claims: Dict[int, List[int]] = {}
        for position, (status, r) in enumerate(bindings):
            if status == "bound" and r is not None:
                claims.setdefault(r, []).append(position)
        once = {r: claims[r][0] for r in chosen if len(claims.get(r, ())) == 1}
        n, k = cert.n_claimed, len(chosen)
        elements = cert.elements
        n_filter, n_target = annotation.size(FILTER_ROLE), annotation.size(TARGET_ROLE)

        # Every annotation cell box is a claim (review EX-5): boxes no row references raise the
        # row-selection denominator like extra rows.
        rows_check = counted_truth(
            ROWS,
            len(once),
            max(n, n_filter, n_target),
            k,
            detail={"elements": [[status, r] for status, r in bindings], "n_boxes": [n_filter, n_target]},
        )
        filter_hits = sum(1 for r, p in once.items() if elements[p].get("filter_value") == rows[r]["filter_value"])
        target_hits = sum(1 for r, p in once.items() if elements[p].get("target_value") == rows[r]["target_value"])
        filter_check = counted_truth(FILTER_VALUE, filter_hits, n, k)
        target_check = counted_truth(TARGET_VALUE, target_hits, n, k)

        count, total, mean = (cert.fields.get(key) for key in ("count", "sum", "mean"))
        targets = _claimed(elements, "target_value")
        sum_ok = total is not None and targets is not None and total == sum(targets)
        mean_ok = count is not None and total is not None and mean is not None and count * mean == total
        count_check = gated_check(COUNT, [rows_check], count is not None and count == n)
        sum_check = gated_check(SUM, [rows_check, target_check], sum_ok)
        mean_check = gated_check(MEAN, [count_check, sum_check], mean_ok)

        filters = [element.get("filter_value") for element in elements]
        satisfied = sum(1 for value in filters if predicate_holds(oracle["predicate"], value))
        return [
            rows_check,
            filter_check,
            target_check,
            count_check,
            sum_check,
            mean_check,
            local_check(
                REF_LOCAL,
                indices_bijective(elements, n_filter) and n_filter == n_target == n,
                detail={"n_filter_cells": n_filter, "n_target_cells": n_target, "n_rows": n},
            ),
            local_check(PRED_LOCAL, satisfied == n, detail={"n_satisfied": satisfied, "n_rows": n}),
            local_check(COUNT_LOCAL, count is not None and count == n),
            local_check(SUM_LOCAL, sum_ok),
            local_check(MEAN_LOCAL, mean_ok and count is not None and count >= 1),
            local_check(ANSWER_LOCAL, cert.answer is not None and mean is not None and cert.answer == mean),
        ]

    def diagnostics(
        self,
        cert: CertificateView,
        annotation: AnnotationView,
        oracle: Mapping[str, Any],
        ctx: PublicContext,
        checks: Sequence[CheckResult],
    ) -> Dict[str, Any]:
        """Binding statuses, extra rows and operand-read accuracy on every bound row
        (selected or not), which measures operand extraction for RQ3."""
        rows_check = next(check for check in checks if check.check_id == ROWS)
        bindings = rows_check.detail.get("elements", [])
        rows = oracle["rows"]
        statuses: Dict[str, int] = {}
        filter_reads = target_reads = bound = extra = 0
        for position, (status, r) in enumerate(bindings):
            statuses[status] = statuses.get(status, 0) + 1
            if status != "bound" or r is None:
                continue
            bound += 1
            extra += 0 if rows[r]["selected"] else 1
            element = cert.elements[position]
            filter_reads += element.get("filter_value") == rows[r]["filter_value"]
            target_reads += element.get("target_value") == rows[r]["target_value"]
        return {
            "fm.binding_status": statuses,
            "fm.extra_rows": extra,
            "fm.filter_read_acc": filter_reads / bound if bound else None,
            "fm.target_read_acc": target_reads / bound if bound else None,
        }

    # ======================================================================================
    # counterexamples (DESIGN 9)

    def corruptions(self, oracle: Mapping[str, Any], sft_obj: Mapping[str, Any], rng_seed: int) -> List[Corruption]:
        """Deterministic corruptions of the SFT object with every check's expected outcome.

        Each ``expected`` lists all six truth families, all local checks and A/G/M/F; exact
        floats are given where the partial credit is known (k = oracle selected rows)."""
        rng = random.Random(rng_seed)
        rows = oracle["rows"]
        chosen = [row for row in rows if row["selected"]]
        others = [row for row in rows if not row["selected"]]
        k = len(chosen)
        mean = int(oracle["mean"])
        p = rng.randrange(k)
        q = (p + 1 + rng.randrange(k - 1)) % k if k > 1 else p
        f_col, t_col, n_cols = int(oracle["filter_col"]), int(oracle["target_col"]), int(oracle["n_cols"])
        cells = oracle["cells"]
        predicate = oracle["predicate"]
        out: List[Corruption] = []

        def base() -> Dict[str, Any]:
            return copy.deepcopy(dict(sft_obj))

        def add(name: str, obj: Mapping[str, Any], description: str, **expect: Any) -> None:
            out.append(Corruption(name, obj, self._expected(**expect), description))

        def resized(name: str, obj: Dict[str, Any], description: str, frac: Dict[str, float], **expect: Any) -> None:
            """After rows were added or removed: recompute count/sum/mean the way a model would."""
            cert = obj["certificate"]
            cert["count"] = len(cert["rows"])
            cert["sum"] = sum(row["target_value"] for row in cert["rows"])
            cert["mean"] = cert["sum"] // cert["count"] if cert["count"] else 0
            obj["answer"] = cert["mean"]
            divisible = cert["count"] >= 1 and cert["sum"] % cert["count"] == 0
            expect.setdefault("local", {})[MEAN_LOCAL] = "pass" if divisible else "fail"
            expect["A"] = "pass" if obj["answer"] == mean else "fail"
            fail = {ROWS: frac[ROWS], FILTER_VALUE: frac[FILTER_VALUE], TARGET_VALUE: frac[TARGET_VALUE]}
            add(name, obj, description, families={**fail, COUNT: "fail", SUM: "fail", MEAN: "fail"}, **expect)

        def append_row(obj: Dict[str, Any], row: Mapping[str, Any]) -> None:
            annotation = obj["annotation"]
            index = len(annotation[FILTER_ROLE])
            annotation[FILTER_ROLE].append(list(row["filter_cell"]))
            annotation[TARGET_ROLE].append(list(row["target_cell"]))
            obj["certificate"]["rows"].append(
                {"annotation_index": index, "filter_value": row["filter_value"], "target_value": row["target_value"]}
            )

        # --- operands wrong, arithmetic self-consistent ------------------------------------
        if k > 1:
            obj = base()
            obj["certificate"]["rows"][p]["target_value"] += 1
            obj["certificate"]["rows"][q]["target_value"] -= 1
            add("wrong_target_value_consistent", obj, "two target values misread by +1/-1; sum and mean unchanged",
                families={TARGET_VALUE: (k - 2) / k, SUM: "fail", MEAN: "fail"})
        obj = base()
        obj["certificate"]["rows"][p]["target_value"] += k
        obj["certificate"]["sum"] += k
        obj["certificate"]["mean"] += 1
        obj["answer"] += 1
        add("wrong_target_value_answer", obj, "one target value misread; sum, mean and answer follow it",
            families={TARGET_VALUE: (k - 1) / k, SUM: "fail", MEAN: "fail"}, A="fail")
        obj = base()
        true_filter = obj["certificate"]["rows"][p]["filter_value"]
        alternative = _alternative_satisfying(predicate, true_filter)
        obj["certificate"]["rows"][p]["filter_value"] = alternative if alternative is not None else true_filter + 1
        add("wrong_filter_value_consistent", obj, "one filter value misread as another value that passes the filter",
            families={FILTER_VALUE: (k - 1) / k}, local={PRED_LOCAL: "pass" if alternative is not None else "fail"})
        obj = base()
        obj["certificate"]["rows"][p]["filter_value"] = _boundary_violation(predicate)
        add("claimed_boundary_value", obj, "one filter value claimed as the boundary value, which fails the filter",
            families={FILTER_VALUE: (k - 1) / k}, local={PRED_LOCAL: "fail"})

        # --- arithmetic wrong, values right ----------------------------------------------------
        obj = base()
        obj["certificate"]["sum"] += 1
        add("wrong_sum", obj, "sum off by one with every value right",
            families={SUM: "fail", MEAN: "fail"}, local={SUM_LOCAL: "fail", MEAN_LOCAL: "fail"})
        obj = base()
        obj["certificate"]["mean"] += 1
        obj["answer"] += 1
        add("wrong_mean", obj, "mean (and answer) off by one with sum and count right",
            families={MEAN: "fail"}, local={MEAN_LOCAL: "fail"}, A="fail")
        obj = base()
        obj["certificate"]["count"] += 1
        add("wrong_count", obj, "count off by one",
            families={COUNT: "fail", MEAN: "fail"}, local={COUNT_LOCAL: "fail", MEAN_LOCAL: "fail"})
        obj = base()
        obj["answer"] += 1
        add("wrong_answer_only", obj, "certificate right, final answer off by one",
            local={ANSWER_LOCAL: "fail"}, A="fail")

        # --- omitted, duplicate, extra rows -----------------------------------------------------
        obj = base()
        for role in (FILTER_ROLE, TARGET_ROLE):
            del obj["annotation"][role][p]
        del obj["certificate"]["rows"][p]
        for index, row in enumerate(obj["certificate"]["rows"]):
            row["annotation_index"] = index
        resized("omit_row", obj, "one selected row omitted from annotation and certificate",
                {ROWS: (k - 1) / k, FILTER_VALUE: (k - 1) / k, TARGET_VALUE: (k - 1) / k}, G="fail")
        obj = base()
        obj["certificate"]["rows"].append(dict(obj["certificate"]["rows"][p]))
        resized("duplicate_row_same_index", obj, "one row repeated with the same annotation index",
                {ROWS: (k - 1) / (k + 1), FILTER_VALUE: (k - 1) / (k + 1), TARGET_VALUE: (k - 1) / (k + 1)},
                local={REF_LOCAL: "fail"})
        obj = base()
        append_row(obj, chosen[p])
        resized("duplicate_row_new_box", obj, "one row repeated with copies of its two boxes (identity duplicate)",
                {ROWS: k / (k + 1), FILTER_VALUE: k / (k + 1), TARGET_VALUE: k / (k + 1)}, G="fail")
        if others:
            extra = others[rng.randrange(len(others))]
            obj = base()
            append_row(obj, extra)
            resized("extra_row", obj, "a non-selected row added with its true values",
                    {ROWS: k / (k + 1), FILTER_VALUE: k / (k + 1), TARGET_VALUE: k / (k + 1)},
                    local={PRED_LOCAL: "fail"}, G="fail")
            boundary = min(others, key=lambda row: (_boundary_distance(predicate, row["filter_value"]), row["r"]))
            obj = base()
            append_row(obj, boundary)
            resized("boundary_value_row", obj, "the non-selected row nearest the filter boundary added "
                    "(value == threshold for above/below)",
                    {ROWS: k / (k + 1), FILTER_VALUE: k / (k + 1), TARGET_VALUE: k / (k + 1)},
                    local={PRED_LOCAL: "fail"}, G="fail")
        obj = base()
        obj["annotation"] = {FILTER_ROLE: [], TARGET_ROLE: []}
        obj["certificate"].update(rows=[], sum=0, count=0, mean=0)
        obj["answer"] = 0
        add("empty_certificate", obj, "empty selection claimed (count 0)",
            families={ROWS: 0.0, FILTER_VALUE: 0.0, TARGET_VALUE: 0.0, COUNT: "fail", SUM: "fail", MEAN: "fail"},
            local={MEAN_LOCAL: "fail"}, A="pass" if mean == 0 else "fail", G="fail")

        # --- binding: swapped roles, pairing, columns, boxes off target --------------------------
        obj = base()
        annotation = obj["annotation"]
        annotation[FILTER_ROLE], annotation[TARGET_ROLE] = annotation[TARGET_ROLE], annotation[FILTER_ROLE]
        add("swap_filter_target", obj, "filter_cells and target_cells exchanged",
            families={ROWS: 0.0, FILTER_VALUE: 0.0, TARGET_VALUE: 0.0, COUNT: "fail", SUM: "fail", MEAN: "fail"},
            G="fail")
        if k > 1:
            obj = base()
            targets = obj["annotation"][TARGET_ROLE]
            targets[p], targets[q] = targets[q], targets[p]
            add("wrong_pairing", obj, "target boxes of two rows exchanged (same sets, wrong rows)",
                families=_binding_failure((k - 2) / k))
        obj = base()
        r_p = chosen[p]["r"]
        other_col = next((c for c in range(n_cols) if c not in (f_col, t_col)), t_col)
        obj["annotation"][FILTER_ROLE][p] = list(cells[r_p][other_col])
        add("wrong_column", obj, "one filter box on another column of the same row",
            families=_binding_failure((k - 1) / k), G="fail")
        obj = base()
        neighbour = r_p + 1 if r_p + 1 < int(oracle["n_rows"]) else r_p - 1
        obj["annotation"][FILTER_ROLE][p] = list(cells[neighbour][f_col])
        obj["annotation"][TARGET_ROLE][p] = list(cells[neighbour][t_col])
        add("box_shift_row", obj, "both boxes of one row moved to the neighbouring data row",
            families={**_binding_failure((k - 1) / k), FILTER_VALUE: "fail", TARGET_VALUE: "fail"}, G="fail")
        obj = base()
        x0, y0, x1, y1 = obj["annotation"][TARGET_ROLE][p]
        shift = (y1 - y0) / 2.0 * (1 if neighbour > r_p else -1)
        obj["annotation"][TARGET_ROLE][p] = [x0, y0 + shift, x1, y1 + shift]
        add("box_straddle", obj, "one target box shifted half a row (straddles two cells)",
            families=_binding_failure((k - 1) / k), G="fail")

        # --- schema and index failures -----------------------------------------------------------
        for name, index in (("index_out_of_range", k), ("negative_index", -1)):
            obj = base()
            obj["certificate"]["rows"][p]["annotation_index"] = index
            add(name, obj, f"annotation_index {index} on one row (never Python-indexed)",
                families=_binding_failure((k - 1) / k), local={REF_LOCAL: "fail"}, F="fail")
        obj = base()
        obj["certificate"]["count"] = True
        add("bool_count", obj, "count given as JSON true",
            families={COUNT: "fail", MEAN: "fail"}, local={COUNT_LOCAL: "fail", MEAN_LOCAL: "fail"}, F="fail")
        obj = base()
        obj["certificate"]["mean"] = float(obj["certificate"]["mean"])
        add("float_mean", obj, "integral float mean: scored by value, F=0", F="fail")
        return out

    def _expected(
        self,
        *,
        families: Optional[Mapping[str, Any]] = None,
        local: Optional[Mapping[str, str]] = None,
        A: str = "pass",
        G: str = "unchanged",
        F: str = "pass",
    ) -> Dict[str, Any]:
        expected: Dict[str, Any] = {family: "pass" for family in self.m_families}
        expected.update({check: "pass" for check in self.local_check_ids})
        expected.update(families or {})
        expected.update(local or {})
        failing = any(expected[family] != "pass" for family in self.m_families)
        expected.update(M="fail" if failing else "pass", A=A, G=G, F=F)
        return expected

# ------------------------------------------------------------------------------------------
# helpers


def _row_status(f: Binding, t: Binding, n_cols: int, f_col: int, t_col: int) -> Tuple[str, Optional[int]]:
    if not f.bound or f.candidate is None:
        return f.status, None
    if not t.bound or t.candidate is None:
        return t.status, None
    f_row, f_c = divmod(int(f.candidate), n_cols)
    t_row, t_c = divmod(int(t.candidate), n_cols)
    if f_c != f_col or t_c != t_col:
        return "wrong_column", None
    if f_row != t_row:
        return "wrong_pairing", None
    return "bound", f_row


def _claimed(elements: Sequence[Any], key: str) -> Optional[List[int]]:
    """Every element's typed ``key`` value, or None when any element lacks it."""
    values = [element.get(key) for element in elements]
    return None if any(value is None for value in values) else values


def _binding_failure(fraction: float) -> Dict[str, Any]:
    return {ROWS: fraction, FILTER_VALUE: fraction, TARGET_VALUE: fraction, COUNT: "fail", SUM: "fail", MEAN: "fail"}


def _alternative_satisfying(predicate: Mapping[str, Any], value: int) -> Optional[int]:
    """Another integer that still passes the filter (None for a one-value interval)."""
    for candidate in (value + 1, value - 1):
        if predicate_holds(predicate, candidate):
            return candidate
    return None


def _boundary_violation(predicate: Mapping[str, Any]) -> int:
    """The nearest integer that fails the filter: the threshold, or ``lo - 1``."""
    return int(predicate["lo"]) - 1 if predicate["op"] == "in" else int(predicate["t"])


def _boundary_distance(predicate: Mapping[str, Any], value: int) -> int:
    if predicate["op"] == "in":
        return max(int(predicate["lo"]) - value, value - int(predicate["hi"]), 0)
    return abs(value - int(predicate["t"]))


def create_adapter() -> Adapter:
    return FilteredMeanAdapter()


__all__ = [
    "FilteredMeanAdapter",
    "HELD_OUT_CELLS",
    "QUERY_FILTERS",
    "TASK_ID",
    "TRAIN_CELLS",
    "create_adapter",
    "holdout_type",
    "predicate_holds",
]
