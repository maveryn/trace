"""Path certificate adapter: a cheapest route on the RPG tactical map (DESIGN 3.4).

Task ``task_illustrations__rpg_tactical_map__movement_cost_value`` (query ``single``): a blue
unit, one marked target tile and a terrain grid whose entry costs are public in the question
(grass, road and bridge 1, forest 2, mountain 3, water blocked; 4-neighbour moves). The
answer is the minimum movement cost. The cheapest route is often not unique, so the
certificate claims *a* route and the verifier accepts every route whose oracle cost equals
the optimum::

    {"kind": "path", "reachable": true,
     "steps": [{"at": [x, y], "cost": 0}, {"at": [x, y], "cost": 1}, ...], "total": 4}

``at`` is a point strictly inside a tile (DESIGN 2.3) in the response frame and ``cost`` is
the cost of entering that tile (0 for the starting tile). A route is a route *of this
instance*: legality needs ``steps[0]`` on the unit's start tile, and cost reads earn credit
only along the longest legal simple prefix that starts there, so an image-blind fixed walk,
a reversed route, repeated tiles or a "stutter" on the start earn no cost credit (review
EX-2, EX-3, CONF-03, LEAK-03; DESIGN 3.4 note). Optimality is decided from Dijkstra
distance labels recomputed from ``scene_ir.tiles``; the stored ``shortest_path_tile_ids`` (one
heap tie-break) is cross-checked at export and never used as the witness. The unreachable
form ``{"kind": "path", "reachable": false}`` is defined but never exercised: TRACE never
generates an unreachable target, so such a record is quarantined at export.
"""

from __future__ import annotations

import copy
import heapq
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..checks import counted_truth, gated_check, local_check, truth_check
from ..geometry import AnnotationView
from ..identity import CandidateSet, resolve_points
from ..parsing import INT, POINT, CertificateSchema, CertificateView, const
from ..types import CheckResult, Corruption, QuarantineError, TraceRecord, canonical_digest
from .base import Adapter, PublicContext

TASK_ID = "task_illustrations__rpg_tactical_map__movement_cost_value"
PUBLIC_TERRAIN_COSTS: Dict[str, int] = {"grass": 1, "road": 1, "bridge": 1, "forest": 2, "mountain": 3}
BLOCKED_TERRAINS: Tuple[str, ...] = ("water",)
PROFILES: Tuple[str, ...] = ("landscape", "portrait", "square")
MAX_STEPS = 32
SFT_PATH_CAP = 16

FAM_ENDPOINTS = "pa.endpoints.truth"
FAM_LEGAL = "pa.legal.truth"
FAM_COST = "pa.cost.truth"
FAM_TOTAL = "pa.total.truth"
FAM_OPTIMAL = "pa.optimal.truth"
LOCAL_ANNO = "pa.anno.local"
LOCAL_TOTAL = "pa.total.local"
LOCAL_ANSWER = "pa.answer.local"


# ------------------------------------------------------------------------------------------
# grid geometry and graph search (tiles are indexed row-major: i = row * cols + col)


@dataclass(frozen=True)
class Grid:
    """Exact tile grid of one map: ``entry[i]`` is the cost of entering tile i (None = water)."""

    rows: int
    cols: int
    tile_px: int
    entry: Tuple[Optional[int], ...]

    @classmethod
    def from_oracle(cls, oracle: Mapping[str, Any]) -> "Grid":
        grid = oracle["grid"]
        entry = tuple(value for row in oracle["cost"] for value in row)
        return cls(int(grid["rows"]), int(grid["cols"]), int(grid["tile_px"]), entry)

    @property
    def size(self) -> int:
        return self.rows * self.cols

    def index(self, row: int, col: int) -> int:
        return int(row) * self.cols + int(col)

    def rc(self, index: int) -> Tuple[int, int]:
        return divmod(int(index), self.cols)

    def box(self, index: int) -> Tuple[float, float, float, float]:
        row, col = self.rc(index)
        size = float(self.tile_px)
        return (col * size, row * size, (col + 1) * size, (row + 1) * size)

    @property
    def boxes(self) -> Tuple[Tuple[float, float, float, float], ...]:
        return tuple(self.box(index) for index in range(self.size))

    def center(self, index: int) -> List[float | int]:
        x0, y0, x1, y1 = self.box(index)
        return [_compact((x0 + x1) / 2.0), _compact((y0 + y1) / 2.0)]

    def passable(self, index: int) -> bool:
        return self.entry[index] is not None

    def neighbors(self, index: int) -> List[int]:
        """4-neighbours in increasing index order (up, left, right, down)."""
        row, col = self.rc(index)
        out = []
        for d_row, d_col in ((-1, 0), (0, -1), (0, 1), (1, 0)):
            r, c = row + d_row, col + d_col
            if 0 <= r < self.rows and 0 <= c < self.cols:
                out.append(self.index(r, c))
        return out

    def adjacent(self, a: int, b: int) -> bool:
        (ra, ca), (rb, cb) = self.rc(a), self.rc(b)
        return abs(ra - rb) + abs(ca - cb) == 1


def _compact(value: float) -> float | int:
    """Integral coordinates as JSON integers (tile centres are exact, not rounded)."""
    return int(value) if float(value).is_integer() else float(value)


def dijkstra(
    grid: Grid,
    source: int,
    enter: Optional[Callable[[int], Optional[int]]] = None,
    *,
    reverse: bool = False,
) -> Tuple[List[Optional[int]], List[Optional[int]]]:
    """Distance labels and parents from ``source`` with entry costs ``enter`` (default: terrain).

    Forward: ``dist[v]`` = cheapest cost of a route source -> v (entry costs after the source).
    Reverse: ``dist[u]`` = cheapest cost of a route u -> source, and ``parent[u]`` is the next
    tile toward ``source``. Ties pop in index order, so parents are deterministic.
    """
    enter = enter or (lambda index: grid.entry[index])
    dist: List[Optional[int]] = [None] * grid.size
    parent: List[Optional[int]] = [None] * grid.size
    dist[source] = 0
    heap = [(0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d != dist[u]:
            continue
        for v in grid.neighbors(u):
            if enter(v) is None or (reverse and enter(u) is None):
                continue
            nd = d + int(enter(u) if reverse else enter(v))
            if dist[v] is None or nd < dist[v]:
                dist[v], parent[v] = nd, u
                heapq.heappush(heap, (nd, v))
    return dist, parent


def min_tiles(grid: Grid, start: int, target: int) -> Optional[int]:
    """Fewest tiles (start and target included) on any legal start -> target route (BFS)."""
    depth = {start: 1}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        if u == target:
            return depth[u]
        for v in grid.neighbors(u):
            if grid.passable(v) and v not in depth:
                depth[v] = depth[u] + 1
                queue.append(v)
    return None


def _tight_predecessors(grid: Grid, dist: Sequence[Optional[int]], v: int) -> List[int]:
    if dist[v] is None or grid.entry[v] is None:
        return []
    return [u for u in grid.neighbors(v) if dist[u] is not None and dist[u] + grid.entry[v] == dist[v]]


def count_optimal(grid: Grid, dist: Sequence[Optional[int]], start: int, target: int) -> int:
    """Number of optimal start -> target routes (path count in the tight-edge DAG; costs >= 1)."""
    if dist[target] is None:
        return 0
    ways = [0] * grid.size
    ways[start] = 1
    for v in sorted((i for i in range(grid.size) if dist[i] is not None), key=lambda i: (dist[i], i)):
        if v != start:
            ways[v] = sum(ways[u] for u in _tight_predecessors(grid, dist, v))
    return ways[target]


def optimal_paths(
    grid: Grid, dist: Sequence[Optional[int]], start: int, target: int, cap: Optional[int] = None
) -> List[List[int]]:
    """Optimal routes as tile-index lists, enumerated backwards from the target in a fixed
    order (every tight-edge chain reaches the start, so there are no dead ends)."""
    found: List[List[int]] = []
    if dist[target] is None:
        return found

    def walk(v: int, suffix: List[int]) -> None:
        if cap is not None and len(found) >= cap:
            return
        if v == start:
            found.append(suffix[::-1])
            return
        for u in _tight_predecessors(grid, dist, v):
            walk(u, suffix + [u])

    walk(target, [target])
    return found


def grid_oracle(
    *,
    rows: int,
    cols: int,
    tile_px: int,
    profile: str,
    cost: Sequence[Sequence[Optional[int]]],
    start: Sequence[int],
    target: Sequence[int],
) -> Dict[str, Any]:
    """The private oracle of DESIGN 3.4 from a cost grid and the two endpoints (no record checks)."""
    grid = Grid(int(rows), int(cols), int(tile_px), tuple(value for row in cost for value in row))
    s, t = grid.index(*start), grid.index(*target)
    dist, _ = dijkstra(grid, s)
    reachable = dist[t] is not None
    return {
        "grid": {"rows": grid.rows, "cols": grid.cols, "tile_px": grid.tile_px, "profile": str(profile)},
        "cost": [list(row) for row in cost],
        "start": [int(start[0]), int(start[1])],
        "target": [int(target[0]), int(target[1])],
        "reachable": reachable,
        "opt": dist[t],
        "min_tiles": min_tiles(grid, s, t) if reachable else None,
        "dist": [dist[r * grid.cols : (r + 1) * grid.cols] for r in range(grid.rows)],
        "n_optimal": count_optimal(grid, dist, s, t),
    }


def _endpoints(grid: Grid, oracle: Mapping[str, Any]) -> Tuple[int, int]:
    return grid.index(*oracle["start"]), grid.index(*oracle["target"])


def path_cost(grid: Grid, tiles: Sequence[int]) -> Optional[int]:
    """Oracle cost of a route (sum of entry costs after the first tile); None through water."""
    costs = [grid.entry[t] for t in tiles[1:]]
    return None if any(c is None for c in costs) else int(sum(costs))


def oracle_paths(oracle: Mapping[str, Any], cap: Optional[int] = None) -> List[List[List[int]]]:
    """Every optimal route of an oracle as ``[[row, col], ...]`` lists (fixtures and tests)."""
    grid = Grid.from_oracle(oracle)
    start, target = _endpoints(grid, oracle)
    dist = [value for row in oracle["dist"] for value in row]
    return [[list(grid.rc(t)) for t in path] for path in optimal_paths(grid, dist, start, target, cap)]


def steps_for(grid: Grid, tiles: Sequence[int], costs: Optional[Sequence[Any]] = None) -> List[Dict[str, Any]]:
    """Certificate steps at tile centres; ``costs`` defaults to the true entry costs."""
    if costs is None:
        costs = [0] + [grid.entry[t] for t in tiles[1:]]
    return [{"at": grid.center(t), "cost": c} for t, c in zip(tiles, costs)]


# ------------------------------------------------------------------------------------------
# record checks (export invariants -> QuarantineError)


def _require(condition: bool, reason: str, message: str = "") -> None:
    if not condition:
        raise QuarantineError(reason, message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), "missing_field", name)
    return value


def _positive_int(value: Any, name: str) -> int:
    _require(type(value) is int and value > 0, "missing_field", name)
    return int(value)


class PathAdapter(Adapter):
    """``path.rpg_movement_cost``: endpoints, legality, step costs, total and optimality."""

    adapter_id = "path.rpg_movement_cost"
    kind = "path"
    version = "1.0.0"
    schema_id = "cert.path.v1"
    # 1.1.0: legality anchored on the start tile; cost credit on the legal prefix from it.
    verifier_version = "1.1.0"
    supported = {TASK_ID: ("single",)}
    annotation_type = "bbox_map"
    annotation_roles = ()
    list_key = "steps"
    list_cap = MAX_STEPS
    cert_fields = {"reachable": const(True), "total": INT}
    element_fields = {"at": POINT, "cost": INT}
    m_families = (FAM_ENDPOINTS, FAM_LEGAL, FAM_COST, FAM_TOTAL, FAM_OPTIMAL)
    family_stages = {
        FAM_ENDPOINTS: "grounding",
        FAM_LEGAL: "grounding",
        FAM_COST: "operand",
        FAM_TOTAL: "computation",
        FAM_OPTIMAL: "computation",
    }
    local_check_ids = (LOCAL_ANNO, LOCAL_TOTAL, LOCAL_ANSWER)
    # DESIGN 6: canvas profile cells; "square" (800x800, 10x10) is the held-out render cell.
    train_cells = ("landscape", "portrait")
    heldout_cells = {"square": "render"}

    # ======================================================================================
    # export side

    def build_oracle(self, record: TraceRecord) -> Dict[str, Any]:
        _require(
            self.supports(record.task, record.query_id, record.instance_version or "v0"),
            "unsupported_row",
            f"{record.task}/{record.query_id}/{record.instance_version}",
        )
        scene = _mapping(record.scene_ir, "scene_ir")
        relations = _mapping(scene.get("relations"), "scene_ir.relations")
        renderer = _mapping(record.execution_trace.get("renderer"), "execution_trace.renderer")
        rows = _positive_int(renderer.get("grid_rows"), "renderer.grid_rows")
        cols = _positive_int(renderer.get("grid_cols"), "renderer.grid_cols")
        tile_px = _positive_int(renderer.get("tile_px"), "renderer.tile_px")
        profile = self._profile(record)
        width, height = record.source_size()
        _require((width, height) == (cols * tile_px, rows * tile_px), "canvas_mismatch", f"{width}x{height}")
        for key, value in (("canvas_width", width), ("canvas_height", height)):
            _require(renderer.get(key) in (None, value), "canvas_mismatch", key)

        _require(
            dict(relations.get("terrain_movement_costs") or {}) == PUBLIC_TERRAIN_COSTS
            and sorted(relations.get("blocked_terrain") or []) == sorted(BLOCKED_TERRAINS),
            "terrain_costs_mismatch",
            "scene costs differ from the costs stated in the public question",
        )
        cost, ids, terrain = self._cost_grid(scene.get("tiles"), rows, cols, tile_px)
        terrain_rows = renderer.get("terrain_rows")
        if terrain_rows is not None:
            _require(terrain_rows == terrain, "terrain_grid_mismatch", "renderer.terrain_rows")

        start = ids.get(relations.get("start_tile_id"))
        target = ids.get(relations.get("target_tile_id"))
        _require(start is not None and target is not None, "unknown_endpoint", "start/target tile id")
        _require(start != target, "degenerate_query", "start == target")
        _require(
            cost[start[0]][start[1]] is not None and cost[target[0]][target[1]] is not None,
            "impassable_endpoint",
        )
        blue = [u for u in scene.get("units") or [] if isinstance(u, Mapping) and u.get("team") == "blue"]
        _require(
            len(blue) == 1 and blue[0].get("tile_id") == relations.get("start_tile_id"),
            "unit_not_on_start",
        )

        oracle = grid_oracle(
            rows=rows, cols=cols, tile_px=tile_px, profile=profile, cost=cost, start=start, target=target
        )
        grid = Grid.from_oracle(oracle)
        s, t = _endpoints(grid, oracle)
        annotation = record.annotation_gt
        _require(
            annotation.get("type") == "bbox_map"
            and annotation.get("value") == {"player_cell": list(grid.box(s)), "target_cell": list(grid.box(t))},
            "annotation_mismatch",
        )
        _require(oracle["reachable"], "unreachable_target", "the task never poses unreachable targets")
        opt = oracle["opt"]
        answer = record.answer_gt
        _require(
            answer.get("type") == "integer" and type(answer.get("value")) is int and answer["value"] == opt,
            "answer_mismatch",
            f"answer_gt {answer.get('value')!r} != Dijkstra optimum {opt}",
        )
        _require(relations.get("target_shortest_movement_cost", opt) == opt, "answer_mismatch", "relations")
        _require(opt + 1 <= MAX_STEPS, "path_over_cap", f"optimum {opt} may need more than {MAX_STEPS} steps")
        self._cross_check_stored(record, grid, ids, oracle)
        return oracle

    @staticmethod
    def _profile(record: TraceRecord) -> str:
        renderer = record.execution_trace.get("renderer") or {}
        profile = renderer.get("canvas_profile")
        _require(profile in PROFILES, "unknown_profile", repr(profile))
        width, height = record.source_size()
        shape = "landscape" if width > height else "portrait" if width < height else "square"
        _require(shape == profile, "profile_mismatch", f"{profile} vs {width}x{height}")
        return str(profile)

    @staticmethod
    def _cost_grid(
        tiles: Any, rows: int, cols: int, tile_px: int
    ) -> Tuple[List[List[Optional[int]]], Dict[str, Tuple[int, int]], List[List[str]]]:
        _require(isinstance(tiles, list) and len(tiles) == rows * cols, "tile_grid_mismatch", "tile count")
        cost: List[List[Optional[int]]] = [[None] * cols for _ in range(rows)]
        terrain: List[List[str]] = [[""] * cols for _ in range(rows)]
        ids: Dict[str, Tuple[int, int]] = {}
        for tile in tiles:
            _require(isinstance(tile, Mapping), "tile_grid_mismatch", "tile is not an object")
            row, col = tile.get("row"), tile.get("col")
            _require(
                type(row) is int and type(col) is int and 0 <= row < rows and 0 <= col < cols,
                "tile_grid_mismatch",
                f"tile position {row!r},{col!r}",
            )
            _require(not terrain[row][col], "tile_grid_mismatch", f"duplicate tile {row},{col}")
            expected_box = [col * tile_px, row * tile_px, (col + 1) * tile_px, (row + 1) * tile_px]
            box = tile.get("bbox")
            _require(
                isinstance(box, list)
                and len(box) == 4
                and all(type(v) in (int, float) for v in box)
                and [float(v) for v in box] == [float(v) for v in expected_box],
                "tile_geometry_mismatch",
                str(tile.get("tile_id")),
            )
            name = tile.get("terrain")
            _require(
                isinstance(name, str) and (name in PUBLIC_TERRAIN_COSTS or name in BLOCKED_TERRAINS),
                "unknown_terrain",
                repr(name),
            )
            passable = name not in BLOCKED_TERRAINS
            entry = PUBLIC_TERRAIN_COSTS[name] if passable else None
            _require(
                tile.get("passable") is passable and tile.get("movement_cost") == entry
                and type(tile.get("movement_cost")) is type(entry),
                "tile_cost_mismatch",
                str(tile.get("tile_id")),
            )
            tile_id = str(tile.get("tile_id"))
            _require(tile_id not in ids, "tile_grid_mismatch", f"duplicate tile id {tile_id}")
            cost[row][col], terrain[row][col], ids[tile_id] = entry, str(name), (row, col)
        return cost, ids, terrain

    @staticmethod
    def _cross_check_stored(
        record: TraceRecord, grid: Grid, ids: Mapping[str, Tuple[int, int]], oracle: Mapping[str, Any]
    ) -> None:
        """The persisted witness and cost cache must agree with the recomputed labels."""
        relations = record.scene_ir.get("relations") or {}
        stored = relations.get("shortest_path_tile_ids")
        if stored is not None:
            _require(
                isinstance(stored, list) and all(isinstance(i, str) and i in ids for i in stored),
                "stored_path_inconsistent",
            )
            tiles = [grid.index(*ids[i]) for i in stored]
            s, t = _endpoints(grid, oracle)
            legal = len(tiles) >= 2 and all(grid.adjacent(a, b) for a, b in zip(tiles, tiles[1:]))
            _require(
                legal and tiles[0] == s and tiles[-1] == t and path_cost(grid, tiles) == oracle["opt"],
                "stored_path_inconsistent",
            )
        cache = record.render_map.get("movement_costs_by_tile_id")
        if cache is not None:
            dist = [value for row in oracle["dist"] for value in row]
            reachable = {tile_id for tile_id, rc in ids.items() if dist[grid.index(*rc)] is not None}
            _require(
                isinstance(cache, Mapping)
                and set(cache) == reachable
                and all(cache[i] == dist[grid.index(*ids[i])] for i in reachable),
                "dist_mismatch",
                "render_map.movement_costs_by_tile_id",
            )

    def split_cell(self, record: TraceRecord) -> str:
        return self._profile(record)

    def semantic_key(self, record: TraceRecord) -> str:
        relations = record.scene_ir.get("relations") or {}
        tiles = sorted(record.scene_ir.get("tiles") or [], key=lambda t: (t.get("row"), t.get("col")))
        positions = {str(t.get("tile_id")): [t.get("row"), t.get("col")] for t in tiles}
        return canonical_digest(
            {
                "profile": self._profile(record),
                "terrain": [[t.get("row"), t.get("col"), t.get("terrain")] for t in tiles],
                "start": positions.get(str(relations.get("start_tile_id"))),
                "target": positions.get(str(relations.get("target_tile_id"))),
            }
        )

    def sft_certificate(self, oracle: Mapping[str, Any], rng_seed: int) -> Tuple[Any, Dict[str, Any], Any]:
        """One optimal route drawn uniformly from the first ``SFT_PATH_CAP`` enumerated optima."""
        if not oracle.get("reachable"):
            raise QuarantineError("unreachable_target", "no SFT target for an unreachable instance")
        grid = Grid.from_oracle(oracle)
        start, target = _endpoints(grid, oracle)
        dist = [value for row in oracle["dist"] for value in row]
        paths = optimal_paths(grid, dist, start, target, SFT_PATH_CAP)
        tiles = paths[random.Random(rng_seed).randrange(len(paths))]
        annotation = {"player_cell": list(grid.box(start)), "target_cell": list(grid.box(target))}
        certificate = {"kind": self.kind, "reachable": True, "steps": steps_for(grid, tiles), "total": int(oracle["opt"])}
        return annotation, certificate, int(oracle["opt"])

    # ======================================================================================
    # schema

    def schema(self, cert: Any) -> CertificateSchema:
        if isinstance(cert, dict) and cert.get("reachable") is False:
            return CertificateSchema(
                kind=self.kind,
                fields={"reachable": const(False)},
                list_key=None,
                element_fields={},
                list_cap=self.list_cap,
                variant="unreachable",
            )
        return CertificateSchema(
            kind=self.kind,
            fields=dict(self.cert_fields),
            list_key=self.list_key,
            element_fields=dict(self.element_fields),
            list_cap=self.list_cap,
            variant="steps",
        )

    # ======================================================================================
    # verification

    def annotation_keys(self) -> Tuple[str, ...]:
        return ("player_cell", "target_cell")

    def candidates(self, oracle: Mapping[str, Any]) -> Mapping[str, CandidateSet]:
        grid = Grid.from_oracle(oracle)
        start, target = _endpoints(grid, oracle)
        boxes = grid.boxes
        keys = tuple(grid.rc(index) for index in range(grid.size))
        return {
            "player_cell": CandidateSet(boxes, (start,), keys),
            "target_cell": CandidateSet(boxes, (target,), keys),
        }

    def checks(
        self,
        cert: CertificateView,
        annotation: AnnotationView,
        oracle: Mapping[str, Any],
        ctx: PublicContext,
    ) -> List[CheckResult]:
        if cert.variant == "unreachable" or not oracle["reachable"]:
            return self._reachability_checks(cert, oracle)
        grid = Grid.from_oracle(oracle)
        start, target = _endpoints(grid, oracle)
        steps = cert.elements
        n = len(steps)
        points = [ctx.frame.point(step.get("at")) if step.valid else None for step in steps]
        tiles = [r.candidate if r.resolved else None for r in resolve_points(points, grid.boxes)]
        claimed = [step.get("cost") if step.valid else None for step in steps]

        start_ok = n >= 1 and tiles[0] == start
        target_ok = n >= 1 and tiles[-1] == target
        endpoints = truth_check(
            FAM_ENDPOINTS,
            (start_ok + target_ok) / 2.0,
            error_code=None if start_ok and target_ok else "wrong_endpoint",
            detail={"start": start_ok, "target": target_ok},
        )

        valid = [step.valid for step in steps]
        reason, at_step = _legality(grid, tiles, valid)
        if reason is None and not start_ok:
            reason, at_step = "wrong_start", 0  # a legal walk elsewhere is not a route of the unit
        moves = [
            a is not None and b is not None and grid.adjacent(a, b) and grid.passable(b)
            for a, b in zip(tiles, tiles[1:])
        ]
        legal = truth_check(
            FAM_LEGAL,
            1.0 if reason is None else 0.0,
            error_code=reason,
            detail={
                "at_step": at_step,
                "resolved_steps": sum(t is not None for t in tiles),
                "legal_move_frac": sum(moves) / len(moves) if moves else None,
            },
        )

        # Cost credit only along the longest legal simple prefix from the start tile: a read is
        # credited when the unit can actually be on that tile at that step of the route.
        prefix = _legal_prefix(grid, tiles, valid, start)
        hits = sum(
            1
            for position in range(prefix)
            if claimed[position] is not None and claimed[position] == (0 if position == 0 else grid.entry[tiles[position]])
        )
        confusion: Dict[str, int] = {}
        for position, (tile, value) in enumerate(zip(tiles, claimed)):
            if position == 0 or tile is None or not grid.passable(tile) or value is None:
                continue
            truth = grid.entry[tile]
            if value != truth:
                confusion[f"{truth}->{value}"] = confusion.get(f"{truth}->{value}", 0) + 1
        cost = counted_truth(
            FAM_COST, hits, n, int(oracle["min_tiles"]), detail={"misreads": confusion, "credited_prefix": prefix}
        )

        total_claim = cert.fields.get("total")
        claimed_sum = sum(claimed) if all(c is not None for c in claimed) else None
        total_ok = total_claim is not None and claimed_sum is not None and total_claim == claimed_sum
        total = gated_check(
            FAM_TOTAL, [endpoints, legal, cost], total_ok, detail={"total": total_claim, "claimed_sum": claimed_sum}
        )

        route_cost = path_cost(grid, tiles) if reason is None else None
        optimal = gated_check(
            FAM_OPTIMAL,
            [endpoints, legal],
            route_cost is not None and route_cost == oracle["opt"],
            detail={"path_cost": route_cost, "opt": oracle["opt"], "first_detour": _first_detour(grid, oracle, tiles)},
        )

        anno_ok = n >= 1 and _inside(points[0], annotation.items("player_cell")) and _inside(
            points[-1], annotation.items("target_cell")
        )
        return [
            endpoints,
            legal,
            cost,
            total,
            optimal,
            local_check(LOCAL_ANNO, anno_ok),
            local_check(LOCAL_TOTAL, total_ok),
            local_check(LOCAL_ANSWER, cert.answer is not None and total_claim is not None and cert.answer == total_claim),
        ]

    def _reachability_checks(self, cert: CertificateView, oracle: Mapping[str, Any]) -> List[CheckResult]:
        """``reachable:false`` is right only on an unreachable oracle; a route never is there."""
        right = (cert.variant == "unreachable") != bool(oracle["reachable"])
        code = None if right else ("claimed_unreachable" if cert.variant == "unreachable" else "target_unreachable")
        results = [truth_check(family, 1.0 if right else 0.0, error_code=code) for family in self.m_families]
        results += [
            local_check(check_id, False, applicable=False, error_code="not_applicable")
            for check_id in self.local_check_ids
        ]
        return results

    def diagnostics(
        self,
        cert: CertificateView,
        annotation: AnnotationView,
        oracle: Mapping[str, Any],
        ctx: PublicContext,
        checks: Sequence[CheckResult],
    ) -> Dict[str, Any]:
        by_id = {check.check_id: check for check in checks}
        legal, cost, optimal = by_id[FAM_LEGAL].detail, by_id[FAM_COST].detail, by_id[FAM_OPTIMAL].detail
        route_cost = optimal.get("path_cost")
        return {
            "variant": cert.variant,
            "n_steps": cert.n_claimed,
            "min_tiles": oracle.get("min_tiles"),
            "n_optimal": oracle.get("n_optimal"),
            "resolved_steps": legal.get("resolved_steps"),
            "legal_move_frac": legal.get("legal_move_frac"),
            "path_cost": route_cost,
            "cost_ratio": (oracle["opt"] / route_cost) if route_cost else None,
            "cost_misreads": cost.get("misreads", {}),
        }

    # ======================================================================================
    # counterexamples

    def corruptions(self, oracle: Mapping[str, Any], sft_obj: Mapping[str, Any], rng_seed: int) -> List[Corruption]:
        """DESIGN 9 path corruptions of the SFT object, each with the exact score of every check.

        Index corruptions (negative / out-of-range ``annotation_index``) do not apply: path
        steps bind by points. Their analogues are ``point_off_canvas`` and ``boundary_points``.
        """
        grid = Grid.from_oracle(oracle)
        rng = random.Random(rng_seed)
        start, target = _endpoints(grid, oracle)
        opt, m = int(oracle["opt"]), int(oracle["min_tiles"])
        annotation = sft_obj["annotation"]
        path = self._sft_tiles(grid, sft_obj)
        true = [0] + [grid.entry[t] for t in path[1:]]
        n = len(path)
        out: List[Corruption] = []

        def obj(tiles: Sequence[int], costs: Sequence[Any], **override: Any) -> Dict[str, Any]:
            steps = override.pop("steps", None) or steps_for(grid, tiles, costs)
            total = override.pop("total", sum(costs))
            cert = {"kind": "path", "reachable": True, "steps": steps, "total": total}
            return {
                "annotation": override.pop("annotation", annotation),
                "certificate": override.pop("certificate", cert),
                "answer": override.pop("answer", total),
            }

        def add(name: str, response: Mapping[str, Any], description: str, **expected: Any) -> None:
            out.append(Corruption(name, copy.deepcopy(response), _expected(**expected), description))

        def ratio(numerator: int, length: int) -> float:
            return numerator / max(length, m)

        i = rng.randrange(1, n)
        wrong = rng.choice([v for v in (1, 2, 3) if v != true[i]])
        costs = true[:i] + [wrong] + true[i + 1 :]
        add("wrong_cost_consistent", obj(path, costs), f"step {i} cost {true[i]} read as {wrong}; total and answer follow",
            e=1.0, lg=1.0, c=ratio(n - 1, n), t=0.0, o=1.0, a=False)
        add("wrong_total", obj(path, true, total=opt + 1), "right step costs, total off by one, answer = total",
            e=1.0, lg=1.0, c=1.0, t=0.0, o=1.0, tl=False, a=False)
        add("answer_inconsistent", obj(path, true, answer=opt + 1), "certificate right, answer != total",
            e=1.0, lg=1.0, c=1.0, t=1.0, o=1.0, al=False, a=False)
        if n >= 3:
            j = rng.randrange(1, n - 1)
            tiles, costs = path[:j] + path[j + 1 :], true[:j] + true[j + 1 :]
            add("skipped_tile", obj(tiles, costs), f"step {j} omitted (non-adjacent move; reads after it are not on the route)",
                e=1.0, lg=0.0, c=ratio(j, n - 1), t=0.0, o=0.0, a=False)
        add("truncated_path", obj(path[:-1], true[:-1]), "last step (the target) omitted",
            e=0.5, lg=1.0 if n - 1 >= 2 else 0.0, c=ratio(n - 1, n - 1), t=0.0, o=0.0, anno=False, a=False)
        j = rng.randrange(1, n)
        tiles, costs = path[: j + 1] + [path[j]] + path[j + 1 :], true[: j + 1] + [true[j]] + true[j + 1 :]
        add("duplicate_step", obj(tiles, costs), f"step {j} repeated: credit stops at the repeat",
            e=1.0, lg=0.0, c=ratio(j + 1, n + 1), t=0.0, o=0.0, a=False)
        extra = next((v for v in grid.neighbors(target) if grid.passable(v) and v not in path), None)
        if extra is not None:
            add("extra_step_after_target", obj(path + [extra], true + [grid.entry[extra]]), "one legal step past the target",
                e=0.5, lg=1.0, c=1.0, t=0.0, o=0.0, anno=False, a=False)
        detour = _detour(grid, path)
        if detour is not None:
            costs = [0] + [grid.entry[t] for t in detour[1:]]
            add("suboptimal_path", obj(detour, costs), "legal route with a two-tile detour, true costs",
                e=1.0, lg=1.0, c=1.0, t=1.0, o=0.0, a=sum(costs) == opt)
        water = _water_route(grid, start, target)
        if water is not None:
            costs = [0] + [grid.entry[t] if grid.passable(t) else 1 for t in water[1:]]
            n_water = sum(not grid.passable(t) for t in water)
            first_water = next(p for p, t in enumerate(water) if not grid.passable(t))
            add("water_tile", obj(water, costs), f"route through {n_water} water tile(s) read as cost 1",
                e=1.0, lg=0.0, c=ratio(first_water, len(water)), t=0.0, o=0.0, a=sum(costs) == opt)
        tiles = path[::-1]
        costs = [0] + [grid.entry[t] for t in tiles[1:]]
        add("reversed_endpoints", obj(tiles, costs), "route walked from the target back to the start",
            e=0.0, lg=0.0, c=0.0, t=0.0, o=0.0, anno=False, a=sum(costs) == opt)
        if any(c != 1 for c in true[1:]):
            ones = [0] + [1] * (n - 1)
            add("all_costs_one", obj(path, ones), "every entry cost read as 1 (arithmetic consistent)",
                e=1.0, lg=1.0, c=ratio(1 + sum(c == 1 for c in true[1:]), n), t=0.0, o=1.0, a=n - 1 == opt)
        add("claims_unreachable", obj(path, true, certificate={"kind": "path", "reachable": False}, answer=opt),
            "reachable:false on a reachable instance", e=0.0, lg=0.0, c=0.0, t=0.0, o=0.0, anno=None, tl=None, al=None)
        if m > 2:
            add("teleport_start_target", obj([start, target], [0, grid.entry[target]]), "two steps [start, target]",
                e=1.0, lg=0.0, c=ratio(1, 2), t=0.0, o=0.0, a=grid.entry[target] == opt)
            stutter = [start] * (m - 1) + [target]
            costs = [0] + [grid.entry[start]] * (m - 2) + [grid.entry[target]]
            add("stutter_on_start", obj(stutter, costs), "start repeated, then the target: min_tiles steps, no movement",
                e=1.0, lg=0.0, c=ratio(1, m), t=0.0, o=0.0, a=sum(costs) == opt)
        add("teleport_start_start", obj([start, start], [0, grid.entry[start]]), "two steps [start, start]",
            e=0.5, lg=0.0, c=ratio(1, 2), t=0.0, o=0.0, anno=False, a=grid.entry[start] == opt)
        nxt = next((v for v in grid.neighbors(start) if grid.passable(v) and v != target), None)
        if nxt is not None:
            length = max(m, 4)
            walk = [start if p % 2 == 0 else nxt for p in range(length)]
            costs = [0] + [grid.entry[t] for t in walk[1:]]
            add("backtrack", obj(walk, costs), "start and one neighbour alternated (true entry costs)",
                e=0.5, lg=0.0, c=ratio(2, length), t=0.0, o=0.0, anno=False, a=sum(costs) == opt)
        detached = _detached_walk(grid, start, target, max(m, 2))
        if detached is not None:
            costs = [0] + [grid.entry[t] for t in detached[1:]]
            add("detached_walk", obj(detached, costs), "a legal walk with true costs that never touches start or target",
                e=0.0, lg=0.0, c=0.0, t=0.0, o=0.0, anno=False, a=sum(costs) == opt)
        shifted = {role: _shift_box(grid, box) for role, box in annotation.items()}
        add("annotation_shifted", obj(path, true, annotation=shifted), "annotation boxes moved one tile off target",
            e=1.0, lg=1.0, c=1.0, t=1.0, o=1.0, anno=False, g="fail")
        corners = [{"at": [_compact(grid.box(t)[0]), _compact(grid.box(t)[1])], "cost": c} for t, c in zip(path, true)]
        add("boundary_points", obj(path, true, steps=corners), "every point on a tile corner (unresolved)",
            e=0.0, lg=0.0, c=0.0, t=0.0, o=0.0, anno=False)
        steps = steps_for(grid, path)
        steps[1] = {"at": [-steps[1]["at"][0], steps[1]["at"][1]], "cost": true[1]}
        add("point_off_canvas", obj(path, true, steps=steps), "step 1 at a negative x (outside every tile)",
            e=0.5 if n == 2 else 1.0, lg=0.0, c=ratio(1, n), t=0.0, o=0.0, anno=n > 2)
        k = next((p for p in range(1, n - 1) if true[p] == 1), next((p for p in range(1, n) if true[p] == 1), 1))
        steps = steps_for(grid, path)
        steps[k] = {"at": steps[k]["at"], "cost": True}
        add("bool_cost", obj(path, true, steps=steps), f"step {k} cost given as JSON true (invalid element)",
            e=0.5 if k == n - 1 else 1.0, lg=0.0, c=ratio(k, n), t=0.0, o=0.0, anno=k != n - 1, tl=False, f=False)
        add("float_costs", obj(path, true, steps=steps_for(grid, path, [float(c) for c in true])),
            "integral float costs: scored by value in M, F = 0", e=1.0, lg=1.0, c=1.0, t=1.0, o=1.0, f=False)
        cert = {"kind": "path", "reachable": True, "steps": steps_for(grid, path)}
        add("total_missing", obj(path, true, certificate=cert, answer=opt), "certificate without total",
            e=1.0, lg=1.0, c=1.0, t=0.0, o=1.0, tl=False, al=False, f=False)
        return out

    @staticmethod
    def _sft_tiles(grid: Grid, sft_obj: Mapping[str, Any]) -> List[int]:
        points = [step["at"] for step in sft_obj["certificate"]["steps"]]
        resolutions = resolve_points(points, grid.boxes)
        if not all(r.resolved for r in resolutions):
            raise ValueError("SFT steps must resolve to tiles in source pixels")
        return [int(r.candidate) for r in resolutions]

# ------------------------------------------------------------------------------------------
# helpers


def _legality(
    grid: Grid, tiles: Sequence[Optional[int]], valid: Sequence[bool]
) -> Tuple[Optional[str], Optional[int]]:
    """First reason a claimed route is not a legal simple route (None when legal).

    Every tile of a route, the first included, must be passable: a unit never stands on water.
    """
    if len(tiles) < 2:
        return "too_short", None
    seen = set()
    for position, tile in enumerate(tiles):
        if not valid[position]:
            return "invalid_step", position
        if tile is None:
            return "unresolved_step", position
        if not grid.passable(tile):
            return "impassable", position
        if position and not grid.adjacent(tiles[position - 1], tile):
            return "not_adjacent", position
        if tile in seen:
            return "repeated_tile", position
        seen.add(tile)
    return None, None


def _legal_prefix(grid: Grid, tiles: Sequence[Optional[int]], valid: Sequence[bool], start: int) -> int:
    """Length of the longest prefix that is a legal simple route from the start tile: step 0
    on ``start``, then valid, resolved, passable, 4-adjacent, never-repeated steps."""
    seen: set = set()
    for position, tile in enumerate(tiles):
        if not valid[position] or tile is None or not grid.passable(tile):
            return position
        if position == 0 and tile != start:
            return 0
        if position and not grid.adjacent(tiles[position - 1], tile):
            return position
        if tile in seen:
            return position
        seen.add(tile)
    return len(tiles)


def _detached_walk(grid: Grid, start: int, target: int, length: int) -> Optional[List[int]]:
    """A simple 4-neighbour walk of ``length`` passable tiles avoiding start and target
    (deterministic depth-first search in index order), or None."""
    banned = {start, target}

    def extend(walk: List[int]) -> Optional[List[int]]:
        if len(walk) == length:
            return walk
        for v in grid.neighbors(walk[-1]):
            if grid.passable(v) and v not in banned and v not in walk:
                found = extend(walk + [v])
                if found is not None:
                    return found
        return None

    for first in range(grid.size):
        if grid.passable(first) and first not in banned:
            found = extend([first])
            if found is not None:
                return found
    return None


def _first_detour(grid: Grid, oracle: Mapping[str, Any], tiles: Sequence[Optional[int]]) -> Optional[int]:
    """First step that leaves the tight-edge DAG (``dist[v] != dist[u] + cost(v)``), if any."""
    dist = oracle["dist"]
    for position in range(1, len(tiles)):
        u, v = tiles[position - 1], tiles[position]
        if u is None or v is None or not grid.passable(v):
            return position
        du, dv = dist[u // grid.cols][u % grid.cols], dist[v // grid.cols][v % grid.cols]
        if du is None or dv is None or du + grid.entry[v] != dv:
            return position
    return None


def _inside(point: Optional[Sequence[float]], boxes: Sequence[Optional[Sequence[float]]]) -> bool:
    if point is None or len(boxes) != 1 or boxes[0] is None:
        return False
    x0, y0, x1, y1 = boxes[0]
    return x0 < point[0] < x1 and y0 < point[1] < y1


def _expected(
    *,
    e: float,
    lg: float,
    c: float,
    t: float,
    o: float,
    anno: Optional[bool] = True,
    tl: Optional[bool] = True,
    al: Optional[bool] = True,
    a: bool = True,
    g: str = "unchanged",
    f: bool = True,
) -> Dict[str, Any]:
    """Exact family scores, M, and pass/fail of the locals (None = not applicable) and signals."""
    families = {FAM_ENDPOINTS: e, FAM_LEGAL: lg, FAM_COST: c, FAM_TOTAL: t, FAM_OPTIMAL: o}
    expected: Dict[str, Any] = {key: float(value) for key, value in families.items()}
    expected["M"] = sum(families.values()) / len(families)
    for check_id, ok in ((LOCAL_ANNO, anno), (LOCAL_TOTAL, tl), (LOCAL_ANSWER, al)):
        if ok is not None:
            expected[check_id] = "pass" if ok else "fail"
    expected.update({"A": "pass" if a else "fail", "G": g, "F": "pass" if f else "fail"})
    return expected


def _detour(grid: Grid, path: Sequence[int]) -> Optional[List[int]]:
    """Replace one move u -> v by u -> a -> b -> v around a side (a, b passable and unused)."""
    used = set(path)
    for position in range(len(path) - 1):
        u, v = path[position], path[position + 1]
        (ru, cu), (rv, cv) = grid.rc(u), grid.rc(v)
        sides = ((0, -1), (0, 1)) if cu == cv else ((-1, 0), (1, 0))
        for d_row, d_col in sides:
            ra, ca, rb, cb = ru + d_row, cu + d_col, rv + d_row, cv + d_col
            if not (0 <= ra < grid.rows and 0 <= rb < grid.rows and 0 <= ca < grid.cols and 0 <= cb < grid.cols):
                continue
            a, b = grid.index(ra, ca), grid.index(rb, cb)
            if grid.passable(a) and grid.passable(b) and a not in used and b not in used:
                route = list(path[: position + 1]) + [a, b] + list(path[position + 1 :])
                if len(route) <= MAX_STEPS:
                    return route
    return None


def _water_route(grid: Grid, start: int, target: int) -> Optional[List[int]]:
    """Cheapest simple route through some water tile when water is misread as cost 1."""
    water = [index for index in range(grid.size) if not grid.passable(index)]
    if not water:
        return None

    def misread(index: int) -> int:
        return 1 if grid.entry[index] is None else int(grid.entry[index])

    forward, to_start = dijkstra(grid, start, misread)
    backward, to_target = dijkstra(grid, target, misread, reverse=True)
    for w in sorted(water, key=lambda index: (forward[index] + backward[index], index)):
        head = [w]
        while head[-1] != start:
            head.append(to_start[head[-1]])
        tail = [w]
        while tail[-1] != target:
            tail.append(to_target[tail[-1]])
        route = head[::-1] + tail[1:]
        if len(set(route)) == len(route) and len(route) <= MAX_STEPS:
            return route
    return None


def _shift_box(grid: Grid, box: Sequence[float]) -> List[float]:
    """Move a box one tile sideways, staying on the canvas."""
    dx = grid.tile_px if box[2] + grid.tile_px <= grid.cols * grid.tile_px else -grid.tile_px
    return [box[0] + dx, box[1], box[2] + dx, box[3]]


def create_adapter() -> Adapter:
    return PathAdapter()


__all__ = [
    "Grid",
    "PathAdapter",
    "TASK_ID",
    "count_optimal",
    "create_adapter",
    "dijkstra",
    "grid_oracle",
    "min_tiles",
    "optimal_paths",
    "oracle_paths",
    "path_cost",
    "steps_for",
]
