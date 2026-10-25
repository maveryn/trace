"""Box/point -> scene object identity (DESIGN 2.3, decision D4).

``match(b, c)`` holds when the center of ``b`` is strictly inside ``c`` and either
``IoU(b, c) >= iou_threshold`` or ``area(b & c) >= containment_threshold * area(b)``.
A box resolves to the unique matching candidate; 0 hits is UNRESOLVED and >= 2 hits is
AMBIGUOUS. Boxes are then deduplicated one-to-one in increasing annotation index: a later box
that resolves to an already-claimed object is DUPLICATE (no credit, still counted in n).
Soft IoU alone is never identity. Thresholds are frozen before M2; they are parameters only
for the pre-registered sensitivity analysis (IoU 0.3 / 0.7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

IOU_THRESHOLD = 0.5
CONTAINMENT_THRESHOLD = 0.8

RESOLVED = "resolved"
UNRESOLVED = "unresolved"
AMBIGUOUS = "ambiguous"
DUPLICATE = "duplicate"
INVALID = "invalid"
STATUSES = (RESOLVED, UNRESOLVED, AMBIGUOUS, DUPLICATE, INVALID)

BoxLike = Sequence[float]


@dataclass(frozen=True)
class IdentityParams:
    """Identity thresholds (module constants by default)."""

    iou_threshold: float = IOU_THRESHOLD
    containment_threshold: float = CONTAINMENT_THRESHOLD

    def to_dict(self) -> Dict[str, float]:
        return {"iou_threshold": self.iou_threshold, "containment_threshold": self.containment_threshold}


DEFAULT_IDENTITY = IdentityParams()


@dataclass(frozen=True)
class CandidateSet:
    """All scene objects of an adapter's class for one annotation role (source pixels).

    ``selected`` lists the candidate positions the oracle selects (the witness set); ``keys``
    optionally carries an adapter key per candidate (oid, ``(row, col)``, tile index, ...).
    """

    boxes: Tuple[Tuple[float, float, float, float], ...]
    selected: Tuple[int, ...] = ()
    keys: Tuple[Any, ...] = ()

    def key(self, position: int) -> Any:
        return self.keys[position] if self.keys else position


@dataclass(frozen=True)
class MatchStats:
    iou: float
    containment: float
    center_inside: bool


@dataclass(frozen=True)
class Resolution:
    """Resolution of one annotation element (``index`` is its annotation index).

    ``iou``/``containment``/``center_inside`` are logged for the resolved candidate, or for the
    best-IoU candidate when nothing resolved, so thresholds can be recalibrated offline.
    """

    index: int
    status: str
    candidate: Optional[int] = None
    hits: Tuple[int, ...] = ()
    best_candidate: Optional[int] = None
    iou: float = 0.0
    containment: float = 0.0
    center_inside: bool = False
    duplicate_of: Optional[int] = None

    @property
    def resolved(self) -> bool:
        return self.status == RESOLVED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "status": self.status,
            "candidate": self.candidate,
            "hits": list(self.hits),
            "best_candidate": self.best_candidate,
            "iou": round(self.iou, 6),
            "containment": round(self.containment, 6),
            "center_inside": self.center_inside,
            "duplicate_of": self.duplicate_of,
        }


def _area(box: BoxLike) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def match_stats(box: BoxLike, candidate: BoxLike) -> MatchStats:
    """Raw IoU, containment ``area(b & c) / area(b)`` and strict center-inside of ``b`` in ``c``."""
    bx0, by0, bx1, by1 = (float(v) for v in box)
    cx0, cy0, cx1, cy1 = (float(v) for v in candidate)
    iw = max(0.0, min(bx1, cx1) - max(bx0, cx0))
    ih = max(0.0, min(by1, cy1) - max(by0, cy0))
    inter = iw * ih
    area_b = _area(box)
    union = area_b + _area(candidate) - inter
    iou = inter / union if union > 0.0 else 0.0
    containment = inter / area_b if area_b > 0.0 else 0.0
    center_x, center_y = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
    center_inside = cx0 < center_x < cx1 and cy0 < center_y < cy1
    return MatchStats(iou=iou, containment=containment, center_inside=center_inside)


def box_matches(box: BoxLike, candidate: BoxLike, params: IdentityParams = DEFAULT_IDENTITY) -> bool:
    stats = match_stats(box, candidate)
    return stats.center_inside and (
        stats.iou >= params.iou_threshold or stats.containment >= params.containment_threshold
    )


def resolve_box(
    box: Optional[BoxLike],
    candidates: Sequence[BoxLike],
    params: IdentityParams = DEFAULT_IDENTITY,
    *,
    index: int = 0,
) -> Resolution:
    """Resolve one source-pixel box against all candidates (no dedupe)."""
    if box is None:
        return Resolution(index=index, status=INVALID)
    hits: List[int] = []
    best: Optional[int] = None
    best_stats: Optional[MatchStats] = None
    hit_stats: Dict[int, MatchStats] = {}
    for position, candidate in enumerate(candidates):
        stats = match_stats(box, candidate)
        if best_stats is None or stats.iou > best_stats.iou:
            best, best_stats = position, stats
        if stats.center_inside and (
            stats.iou >= params.iou_threshold or stats.containment >= params.containment_threshold
        ):
            hits.append(position)
            hit_stats[position] = stats
    if len(hits) == 1:
        stats = hit_stats[hits[0]]
        return Resolution(
            index=index,
            status=RESOLVED,
            candidate=hits[0],
            hits=(hits[0],),
            best_candidate=best,
            iou=stats.iou,
            containment=stats.containment,
            center_inside=True,
        )
    logged = best_stats or MatchStats(0.0, 0.0, False)
    return Resolution(
        index=index,
        status=AMBIGUOUS if hits else UNRESOLVED,
        hits=tuple(hits),
        best_candidate=best,
        iou=logged.iou,
        containment=logged.containment,
        center_inside=logged.center_inside,
    )


def dedupe(resolutions: Sequence[Resolution]) -> List[Resolution]:
    """One-to-one: in increasing annotation index, a repeat claim on an object is DUPLICATE."""
    claimed: Dict[int, int] = {}
    out: List[Resolution] = []
    for resolution in sorted(resolutions, key=lambda item: item.index):
        if resolution.status == RESOLVED and resolution.candidate is not None:
            if resolution.candidate in claimed:
                out.append(
                    Resolution(
                        index=resolution.index,
                        status=DUPLICATE,
                        candidate=resolution.candidate,
                        hits=resolution.hits,
                        best_candidate=resolution.best_candidate,
                        iou=resolution.iou,
                        containment=resolution.containment,
                        center_inside=resolution.center_inside,
                        duplicate_of=claimed[resolution.candidate],
                    )
                )
                continue
            claimed[resolution.candidate] = resolution.index
        out.append(resolution)
    return out


def resolve_boxes(
    boxes: Sequence[Optional[BoxLike]],
    candidates: Sequence[BoxLike],
    params: IdentityParams = DEFAULT_IDENTITY,
) -> List[Resolution]:
    """Resolve every annotation box (index = position) and apply one-to-one dedupe."""
    return dedupe(
        [resolve_box(box, candidates, params, index=index) for index, box in enumerate(boxes)]
    )


def resolve_point(
    point: Optional[Sequence[float]],
    tiles: Sequence[BoxLike],
    *,
    index: int = 0,
) -> Resolution:
    """A point resolves to the tile whose interior strictly contains it; boundaries are UNRESOLVED."""
    if point is None:
        return Resolution(index=index, status=INVALID)
    x, y = float(point[0]), float(point[1])
    hits = tuple(
        position
        for position, tile in enumerate(tiles)
        if float(tile[0]) < x < float(tile[2]) and float(tile[1]) < y < float(tile[3])
    )
    if len(hits) == 1:
        return Resolution(index=index, status=RESOLVED, candidate=hits[0], hits=hits, center_inside=True)
    return Resolution(index=index, status=AMBIGUOUS if hits else UNRESOLVED, hits=hits)


def resolve_points(points: Sequence[Optional[Sequence[float]]], tiles: Sequence[BoxLike]) -> List[Resolution]:
    """Resolve each point independently (no dedupe; repeated tiles are a path-legality question)."""
    return [resolve_point(point, tiles, index=index) for index, point in enumerate(points)]


def set_f1(resolutions: Sequence[Resolution], selected: Sequence[int]) -> float:
    """Identity-rule set F1 between resolved (deduplicated) claims and the oracle selection."""
    target = set(int(item) for item in selected)
    predicted = len(resolutions)
    if predicted == 0 and not target:
        return 1.0
    hit = len({r.candidate for r in resolutions if r.status == RESOLVED and r.candidate in target})
    denominator = predicted + len(target)
    return 2.0 * hit / denominator if denominator else 0.0


@dataclass(frozen=True)
class IdentityReport:
    """Per-role resolutions of a whole annotation (logged in diagnostics)."""

    roles: Dict[str, List[Resolution]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {role: [r.to_dict() for r in items] for role, items in self.roles.items()}


__all__ = [
    "AMBIGUOUS",
    "CONTAINMENT_THRESHOLD",
    "CandidateSet",
    "DEFAULT_IDENTITY",
    "DUPLICATE",
    "INVALID",
    "IOU_THRESHOLD",
    "IdentityParams",
    "IdentityReport",
    "MatchStats",
    "RESOLVED",
    "Resolution",
    "UNRESOLVED",
    "box_matches",
    "dedupe",
    "match_stats",
    "resolve_box",
    "resolve_boxes",
    "resolve_point",
    "resolve_points",
    "set_f1",
]
