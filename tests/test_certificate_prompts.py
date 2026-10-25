"""Public prompts, system prompt and JSON Schemas of the certificate extension (DESIGN 4, 1.1).

Covers the prompt pieces (frame lines, template invariance, quarantine assertions, hint drift
checks, legacy conditions), the fixed examples (they validate against the public schemas and
are self-consistent), the private extension schema, and the absence of machine paths. The
adapters are written concurrently: the cross-check against ``Adapter.certificate_line`` and
``Adapter.example_json`` skips while an adapter is still a placeholder.
"""

from __future__ import annotations

import glob
import importlib
import itertools
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping

import pytest

from trace_tasks.experimental.certificates import prompts as P
from trace_tasks.experimental.certificates.types import (
    CertificateExtension,
    CertificateVersionError,
    PublicContract,
    QuarantineError,
    certificate_extension_id,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESOURCE_DIR = REPO_ROOT / "src" / "trace_tasks" / "resources" / "prompts" / "experimental" / "certificates"
PACKAGE_DIR = REPO_ROOT / "src" / "trace_tasks" / "experimental" / "certificates"

FRAMES = ("px_source", "rel1000")
BUNDLE = P.load_prompt_bundle()

# TRACE's own task texts as generated at d3440e95 (with the frame wording the certificate
# prompt removes). Kept literal here so that a drift in the stored hints is caught.
LEGACY: Dict[str, Dict[str, str]] = {
    "task_games__chess__colored_piece_kind_count": dict(
        kind="selected_set",
        question="The figure shows a partial chess board with white and black pieces. Count all White kings shown on the board.",
        ann='set "annotation" to a JSON array of pixel-space piece boxes [x0, y0, x1, y1] for every matching visible piece; use an empty array if none are shown',
        ans='set "answer" to the count of matching visible pieces',
        ann_example='{"annotation":[[116,116,164,164],[276,116,324,164]],"answer":2}',
    ),
    "task_games__chess__piece_kind_count": dict(
        kind="selected_set",
        question="The visual shows a partial chess board with white and black pieces. How many kings are shown?",
        ann='set "annotation" to a JSON array of pixel-space piece boxes [x0, y0, x1, y1] for every matching visible piece; use an empty array if none are shown',
        ans='set "answer" to the count of matching visible pieces',
        ann_example='{"annotation":[[116,116,164,164],[276,116,324,164]],"answer":2}',
    ),
    "task_charts__table__filtered_column_mean": dict(
        kind="filtered_mean",
        question='The image shows a table with one Name column and several data columns. Compute the mean of "Atena" over rows where "Eten" is greater than 19.',
        ann='set "annotation" to an object with "filter_cells" and "target_cells", each mapping to arrays of [x0,y0,x1,y1] boxes for the selected rows',
        ans='set "answer" to the exact integer result',
        ann_example='{"annotation":{"filter_cells":[[260,180,372,236]],"target_cells":[[374,180,486,236]]},"answer":14}',
    ),
    "task_physics__lever__side_torque_value": dict(
        kind="weighted_sum",
        question="The image shows a horizontal lever beam with a fulcrum, numbered distance marks, and weight blocks. Read the left-side weights and distance marks. What torque do they produce about the fulcrum?",
        ann='set "annotation" to an array of [x0,y0,x1,y1] pixel boxes around every weight block on the queried side of the fulcrum',
        ans='set "answer" to the integer torque value',
        ann_example='{"annotation":[[192,282,250,334],[326,282,384,334]],"answer":12}',
        contract_line='Use a valid JSON object with keys "annotation" and "answer" in that order for the final answer.',
    ),
    "task_illustrations__rpg_tactical_map__movement_cost_value": dict(
        kind="path",
        question="The picture shows a pixel-art tactical movement map containing terrain tiles, a blue unit, and a target marker on one tile. The blue unit starts on its current tile. Grass, roads, and bridges cost 1 movement point; forests cost 2; mountains cost 3; water cannot be entered. Moves are only up, down, left, or right. How many movement points are needed to reach the marked destination tile using the cheapest path?",
        ann='set "annotation" to an object with keys "player_cell" and "target_cell", each mapping to one [x0, y0, x1, y1] pixel bounding box around that tile',
        ans='set "answer" to the fewest number of movement points needed to reach the marked target tile',
        ann_example='{"annotation":{"player_cell":[160,320,240,400],"target_cell":[320,320,400,400]},"answer":7}',
        contract_line='Return JSON with keys "annotation" and "answer".',
    ),
}
ANN_PREFIXES = ("Annotation format", "Annotation format", "Required annotation format", "Annotation format", 'Format for the "annotation" field')
ANS_PREFIXES = ("Answer format", "Answer field", "Required answer format", "Final answer format", 'Format for the "answer" field')
IID = "blake3:" + "ab" * 32


def make_base(task: str, *, template: int = 0, question: str | None = None, swap: bool = False) -> Dict[str, Any]:
    """A TrainInstance-shaped dict whose ``prompt_variants`` mimic TRACE's output templates."""
    spec = LEGACY[task]
    q = question if question is not None else spec["question"]
    ann_line = f"{ANN_PREFIXES[template]}: {spec['ann']}."
    ans_line = f"{ANS_PREFIXES[template]}: {spec['ans']}."
    contract = spec.get("contract_line")
    ao_example = '{"answer":2}'
    head = f"{q}\n" + (f"{contract}\n" if contract else "")
    answer_only = f"{q}\n{ans_line}\nExample JSON:\n{ao_example}"
    lines = [ans_line, ann_line] if swap else [ann_line, ans_line]
    both = f"{head}" + "\n".join(lines) + f"\nExample JSON:\n{spec['ann_example']}"
    return {
        "task": task,
        "query_id": "single",
        "instance_id": IID,
        "prompt": both,
        "prompt_variants": {"answer_and_annotation": both, "answer_only": answer_only},
    }


def contract_for(kind: str, frame: str = "px_source", size=(900, 700)) -> PublicContract:
    block = BUNDLE["kinds"][kind]
    return PublicContract(
        kind=kind,
        schema_id=block["schema_id"],
        coord_frame=frame,
        image_size=tuple(size),
        annotation_type=block["annotation_type"],
        list_key=block["list_key"],
        list_cap=block["list_cap"],
    )


TASKS = sorted(LEGACY)
CHESS = "task_games__chess__colored_piece_kind_count"
LEVER = "task_physics__lever__side_torque_value"


def render(task: str, frame: str = "px_source", size=(900, 700), **kwargs: Any) -> str:
    return P.render_public_prompt(make_base(task, **kwargs), contract_for(LEGACY[task]["kind"], frame, size))


# ----------------------------------------------------------------------------------------
# a tiny JSON Schema (2020-12 subset) validator, so the tests do not need ``jsonschema``


class SchemaError(Exception):
    pass


_IGNORED = {"$schema", "$id", "title", "description", "$defs"}


def _same(a: Any, b: Any) -> bool:
    return type(a) is type(b) and a == b


def _is_type(value: Any, name: str) -> bool:
    if name == "object":
        return isinstance(value, dict)
    if name == "array":
        return isinstance(value, list)
    if name == "string":
        return isinstance(value, str)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "integer":  # mathematical integers, like jsonschema: 15.0 is an integer
        return not isinstance(value, bool) and (isinstance(value, int) or (isinstance(value, float) and value.is_integer()))
    if name == "number":
        return not isinstance(value, bool) and isinstance(value, (int, float))
    raise SchemaError(f"unsupported type {name!r}")


def validate(value: Any, schema: Mapping[str, Any], root: Mapping[str, Any] | None = None, path: str = "$") -> List[str]:
    """Return the list of violations of ``schema`` by ``value`` (empty = valid)."""
    root = root if root is not None else schema
    if "$ref" in schema:
        node: Any = root
        for part in schema["$ref"].removeprefix("#/").split("/"):
            node = node[part]
        return validate(value, node, root, path)
    errors: List[str] = []
    for key, rule in schema.items():
        if key in _IGNORED:
            continue
        if key == "type":
            if not _is_type(value, rule):
                return [f"{path}: expected {rule}"]
        elif key == "const":
            if not _same(value, rule):
                errors.append(f"{path}: not const {rule!r}")
        elif key == "enum":
            if not any(_same(value, option) for option in rule):
                errors.append(f"{path}: not in enum")
        elif key == "minimum":
            if _is_type(value, "number") and value < rule:
                errors.append(f"{path}: below minimum")
        elif key == "minLength":
            if isinstance(value, str) and len(value) < rule:
                errors.append(f"{path}: too short")
        elif key == "pattern":
            if isinstance(value, str) and re.search(rule, value) is None:
                errors.append(f"{path}: pattern")
        elif key in ("minItems", "maxItems", "items"):
            if isinstance(value, list):
                if key == "minItems" and len(value) < rule:
                    errors.append(f"{path}: fewer than {rule} items")
                if key == "maxItems" and len(value) > rule:
                    errors.append(f"{path}: more than {rule} items")
                if key == "items":
                    for i, item in enumerate(value):
                        errors += validate(item, rule, root, f"{path}[{i}]")
        elif key in ("properties", "required", "additionalProperties"):
            if isinstance(value, dict):
                if key == "required":
                    errors += [f"{path}: missing {name!r}" for name in rule if name not in value]
                if key == "properties":
                    for name, sub in rule.items():
                        if name in value:
                            errors += validate(value[name], sub, root, f"{path}.{name}")
                if key == "additionalProperties" and rule is False:
                    errors += [f"{path}: unexpected {name!r}" for name in value if name not in schema["properties"]]
        elif key == "oneOf":
            hits = sum(1 for sub in rule if not validate(value, sub, root, path))
            if hits != 1:
                errors.append(f"{path}: matches {hits} oneOf branches")
        else:
            raise SchemaError(f"keyword {key!r} is outside the tested subset")
    return errors


def _refs(node: Any) -> Iterator[str]:
    if isinstance(node, dict):
        if "$ref" in node:
            yield node["$ref"]
        for sub in node.values():
            yield from _refs(sub)
    elif isinstance(node, list):
        for sub in node:
            yield from _refs(sub)


def _numbers(node: Any) -> Iterator[float]:
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        yield node
    elif isinstance(node, dict):
        for sub in node.values():
            yield from _numbers(sub)
    elif isinstance(node, list):
        for sub in node:
            yield from _numbers(sub)


# ----------------------------------------------------------------------------------------
# schemas and fixed examples


@pytest.mark.parametrize("kind", P.CERT_KINDS)
def test_example_validates_against_public_schema(kind: str) -> None:
    schema = P.public_schema(kind)
    example = P.example_object(kind)
    assert validate(example, schema) == []
    assert json.loads(P.example_json(kind)) == example
    # the literal in the prompt is the compact, key-ordered form
    assert list(example) == ["annotation", "certificate", "answer"]
    assert next(iter(example["certificate"])) == "kind"
    assert P.example_json(kind) == json.dumps(example, separators=(",", ":"), ensure_ascii=False)


@pytest.mark.parametrize("kind", P.CERT_KINDS)
def test_schema_is_draft_2020_12_and_refs_resolve(kind: str) -> None:
    schema = P.public_schema(kind)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == BUNDLE["kinds"][kind]["schema_id"] == f"cert.{kind}.v1"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["annotation", "certificate", "answer"]
    for ref in _refs(schema):
        node: Any = schema
        for part in ref.removeprefix("#/").split("/"):
            node = node[part]
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator.check_schema(schema)
    assert list(jsonschema.Draft202012Validator(schema).iter_errors(P.example_object(kind))) == []


def test_private_schema_is_draft_2020_12() -> None:
    schema = P.private_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    for ref in _refs(schema):
        node: Any = schema
        for part in ref.removeprefix("#/").split("/"):
            node = node[part]
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize("kind", P.CERT_KINDS)
def test_examples_stay_inside_both_frames_and_caps(kind: str) -> None:
    """One literal serves both frames (all values <= 1000) and stays under the list cap."""
    example = P.example_object(kind)
    assert max(abs(x) for x in _numbers(example)) <= 1000
    block = BUNDLE["kinds"][kind]
    assert len(example["certificate"].get(block["list_key"], [])) <= block["list_cap"]
    assert str(block["list_cap"]) in P.certificate_line(kind)


def test_selected_set_example_is_self_consistent() -> None:
    ex = P.example_object("selected_set")
    members = ex["certificate"]["members"]
    assert [m["annotation_index"] for m in members] == list(range(len(ex["annotation"])))
    assert ex["certificate"]["count"] == len(members) == ex["answer"]


def test_filtered_mean_example_is_self_consistent() -> None:
    ex = P.example_object("filtered_mean")
    rows, cert = ex["certificate"]["rows"], ex["certificate"]
    assert len(ex["annotation"]["filter_cells"]) == len(ex["annotation"]["target_cells"]) == len(rows)
    assert [r["annotation_index"] for r in rows] == list(range(len(rows)))
    assert cert["sum"] == sum(r["target_value"] for r in rows)
    assert cert["count"] == len(rows)
    assert cert["count"] * cert["mean"] == cert["sum"]
    assert ex["answer"] == cert["mean"]


def test_weighted_sum_example_is_self_consistent() -> None:
    ex = P.example_object("weighted_sum")
    terms = ex["certificate"]["terms"]
    assert [t["annotation_index"] for t in terms] == list(range(len(ex["annotation"])))
    assert all(t["product"] == t["weight"] * t["distance"] for t in terms)
    assert ex["answer"] == sum(t["product"] for t in terms)


def test_path_example_is_self_consistent() -> None:
    ex = P.example_object("path")
    cert, anno = ex["certificate"], ex["annotation"]
    steps = cert["steps"]
    tile = 80  # the example boxes use TRACE's own 80 px tiles

    def inside(point, box):
        return box[0] < point[0] < box[2] and box[1] < point[1] < box[3]

    def tile_of(point):
        return (point[0] // tile, point[1] // tile)

    assert inside(steps[0]["at"], anno["player_cell"]) and inside(steps[-1]["at"], anno["target_cell"])
    assert steps[0]["cost"] == 0
    tiles = [tile_of(s["at"]) for s in steps]
    assert len(set(tiles)) == len(tiles)
    for (r0, c0), (r1, c1) in zip(tiles, tiles[1:]):
        assert abs(r0 - r1) + abs(c0 - c1) == 1
    assert cert["total"] == sum(s["cost"] for s in steps) == ex["answer"]
    assert cert["reachable"] is True


@pytest.mark.parametrize("kind", P.CERT_KINDS)
def test_schema_rejects_malformed_examples(kind: str) -> None:
    schema = P.public_schema(kind)

    def bad(mutate) -> List[str]:
        obj = P.example_object(kind)
        mutate(obj)
        return validate(obj, schema)

    assert bad(lambda o: o.update(extra=1))
    assert bad(lambda o: o.pop("certificate"))
    assert bad(lambda o: o.update(answer="3"))
    assert bad(lambda o: o.update(answer=True))
    assert bad(lambda o: o["certificate"].update(kind="other"))
    assert bad(lambda o: o["certificate"].update(extra=1))
    block = BUNDLE["kinds"][kind]
    key = block["list_key"]
    cap = block["list_cap"]
    element = P.example_object(kind)["certificate"][key][0]
    assert bad(lambda o: o["certificate"].update({key: [element] * (cap + 1)}))
    assert bad(lambda o: o["certificate"][key][0].update(extra=1))
    assert bad(lambda o: o["certificate"][key][0].pop(next(iter(o["certificate"][key][0]))))
    assert bad(lambda o: o["certificate"][key][0].update(annotation_index=-1)) or kind == "path"


def test_schema_type_rules_per_kind() -> None:
    sel, tab, lev, pth = (P.public_schema(k) for k in P.CERT_KINDS[:4])

    def mutated(kind: str, fn) -> List[str]:
        obj = P.example_object(kind)
        fn(obj)
        return validate(obj, P.public_schema(kind))

    assert mutated("selected_set", lambda o: o["certificate"]["members"][0].update(color="White"))
    assert mutated("selected_set", lambda o: o["certificate"]["members"][0].update(piece="horse"))
    assert mutated("selected_set", lambda o: o["certificate"]["members"][0].update(annotation_index=True))
    assert mutated("selected_set", lambda o: o["certificate"].update(count=2.5))
    assert mutated("filtered_mean", lambda o: o["certificate"]["rows"][0].update(filter_value="22"))
    assert mutated("filtered_mean", lambda o: o["annotation"].pop("target_cells"))
    assert mutated("weighted_sum", lambda o: o["certificate"]["terms"][0].update(weight=3.5))
    assert mutated("weighted_sum", lambda o: o["annotation"].__setitem__(0, [1, 2, 3]))
    assert mutated("path", lambda o: o["certificate"]["steps"][0].update(at=[200]))
    assert mutated("path", lambda o: o["certificate"].update(reachable=False))  # steps/total not allowed
    assert not mutated("path", lambda o: o.update(certificate={"kind": "path", "reachable": False}))
    assert mutated("path", lambda o: o.update(certificate={"kind": "path", "reachable": 0}))
    assert mutated("path", lambda o: o["certificate"]["steps"][0].update(at=[200.5, "x"]))
    # the schema accepts float coordinates and integral floats (the scorer's F is stricter)
    assert not mutated("path", lambda o: o["certificate"]["steps"][0].update(at=[200.5, 360.25]))
    assert not mutated("weighted_sum", lambda o: o["certificate"]["terms"][0].update(weight=3.0))
    assert sel and tab and lev and pth


def _sample_extension() -> Dict[str, Any]:
    iid, trh = "blake3:" + "11" * 32, "blake3:" + "22" * 32
    adapter = dict(id="weighted_sum.lever_side_torque", version="1.0.0", schema_id="cert.weighted_sum.v1", verifier_version="1.0.0")
    ext_id = certificate_extension_id(
        instance_id=iid, trace_record_hash=trh, adapter_id=adapter["id"], adapter_version=adapter["version"], schema_id=adapter["schema_id"]
    )
    return {
        "certificate_extension_id": ext_id,
        "base": dict(
            instance_id=iid, trace_record_hash=trh, task="task_physics__lever__side_torque_value", query_id="single",
            instance_seed=3, sampling_seed=7401, seed_index=17, scene_group="blake3:" + "33" * 32, split="E1_test",
            cell="offset_fulcrum", code_hash="cert-v1",
        ),
        "adapter": adapter,
        "labels": {
            "answer_gt": {"type": "integer", "value": 23},
            "annotation_gt": {"type": "bbox_set", "value": [[235.0, 278.5, 293.0, 330.5]]},
            "reward_contract": {"annotation": {"id": "bbox_set_soft_iou_v0"}, "answer": {"id": "answer_exact_match_v0"}},
        },
        "public": {"image_size": [1280, 640], "prompt_sha256": {"px_source": "0" * 64, "rel1000": "f" * 64}},
        "oracle": {"query_side": "left", "blocks": []},
        "sft_target": {"annotation": [[235, 279, 293, 331]], "certificate": {"kind": "weighted_sum", "terms": []}, "answer": 23},
        "provenance": {"trace_tasks_commit": "d3440e95", "cert_commit": "x", "env_lock_sha256": "0" * 64},
    }


def test_private_schema_accepts_extension_record_that_types_accepts() -> None:
    record = _sample_extension()
    assert validate(record, P.private_schema()) == []
    extension = CertificateExtension.from_dict(record)
    extension.verify_id()
    assert extension.to_dict() == record


def test_private_schema_rejects_malformed_records() -> None:
    schema = P.private_schema()

    def bad(mutate) -> List[str]:
        record = _sample_extension()
        mutate(record)
        return validate(record, schema)

    assert bad(lambda r: r.pop("oracle"))
    assert bad(lambda r: r.update(unexpected={}))
    assert bad(lambda r: r.update(certificate_extension_id="sha256:abc"))
    assert bad(lambda r: r["base"].pop("trace_record_hash"))
    assert bad(lambda r: r["adapter"].update(version=""))
    assert bad(lambda r: r["adapter"].update(extra="x"))
    assert bad(lambda r: r["public"].update(image_size=[1280]))
    assert bad(lambda r: r["public"].update(image_size=[0, 640]))
    assert bad(lambda r: r["labels"].pop("reward_contract"))
    assert bad(lambda r: r["sft_target"].pop("answer"))
    assert not bad(lambda r: r.pop("provenance"))
    assert not bad(lambda r: r.pop("sft_target"))


# ----------------------------------------------------------------------------------------
# adapters (skipped while a module is still the placeholder)


@pytest.mark.parametrize("kind", P.CERT_KINDS)
def test_adapter_prompt_pieces_match_the_json(kind: str) -> None:
    module = importlib.import_module(f"trace_tasks.experimental.certificates.adapters.{kind}")
    try:
        adapter = module.create_adapter()
    except NotImplementedError:
        pytest.skip(f"{kind} adapter is still a placeholder")
    block = BUNDLE["kinds"][kind]
    assert adapter.kind == kind
    assert adapter.schema_id == block["schema_id"]
    assert adapter.annotation_type == block["annotation_type"]
    assert adapter.list_key == block["list_key"]
    assert adapter.list_cap == block["list_cap"]
    assert set(adapter.supported) == set(block["tasks"])
    # prompts.py reads the JSON (single source of truth), so the adapter's own texts are advisory:
    # they must describe the same schema and their example must satisfy the public schema.
    line = adapter.certificate_line()
    for name in (adapter.kind, adapter.list_key, *adapter.element_fields, *adapter.cert_fields):
        assert f'"{name}"' in line, f"adapter certificate_line does not mention {name!r}"
    example = adapter.example_object()
    assert validate(example, P.public_schema(kind)) == []
    assert json.loads(adapter.example_json()) == example
    assert max(abs(x) for x in _numbers(example)) <= 1000
    contract = adapter.public_contract((900, 700), "px_source")
    assert contract.schema_id == block["schema_id"]
    base = make_base(next(iter(block["tasks"])))
    assert P.render_public_prompt(base, contract) == render(next(iter(block["tasks"])))


# ----------------------------------------------------------------------------------------
# prompt structure and template invariance


@pytest.mark.parametrize("task,frame", list(itertools.product(TASKS, FRAMES)))
def test_prompt_structure(task: str, frame: str) -> None:
    kind = LEGACY[task]["kind"]
    prompt = render(task, frame)
    lines = prompt.split("\n")
    assert prompt.startswith("<image>") and prompt.count("<image>") == 1
    assert lines[0] == "<image>" + LEGACY[task]["question"]
    assert lines[1] == P.frame_line(frame, (900, 700))
    assert lines[2] == BUNDLE["envelope"]
    assert lines[3].startswith("Annotation format: set \"annotation\" to ") and lines[3].endswith(".")
    assert lines[4] == P.certificate_line(kind)
    assert lines[5].startswith("Answer format: set \"answer\" to ") and lines[5].endswith(".")
    assert lines[6] == "Example JSON:"
    assert lines[7] == P.example_json(kind)
    assert len(lines) == 8
    assert "900x700" in lines[1] if frame == "px_source" else "900" not in lines[1]
    # frame wording is gone from the hint, in both frames
    assert "pixel" not in lines[3].lower()
    if frame == "rel1000":
        assert "pixel" not in "\n".join(lines[1:]).lower()
        assert "0-1000" in lines[1]


def test_frame_lines_follow_design() -> None:
    assert P.frame_line("px_source", (1280, 640)) == (
        "Image size: 1280x640 pixels. Give every box as [x0, y0, x1, y1] and every point as [x, y] "
        "in pixels of this image (origin at the top-left corner, x to the right, y downward)."
    )
    assert P.frame_line("rel1000") == P.frame_line("rel1000", (1, 1)) == P.frame_line("rel1000", (999, 5))
    assert "normalized to 0-1000" in P.frame_line("rel1000")
    assert "0,0 is the top-left corner and 1000,1000 the bottom-right corner" in P.frame_line("rel1000")
    with pytest.raises(ValueError):
        P.frame_line("shown")
    with pytest.raises(ValueError):
        P.frame_line("px_source")
    with pytest.raises(ValueError):
        P.frame_line("px_source", (0, 640))


def test_envelope_line_is_design_text() -> None:
    assert BUNDLE["envelope"] == (
        'End your response with one JSON object with exactly the keys "annotation", "certificate" and '
        '"answer". All numbers in "certificate" and "answer" are integers.'
    )


@pytest.mark.parametrize("task", TASKS)
def test_template_invariance_apart_from_question_and_size(task: str) -> None:
    kind = LEGACY[task]["kind"]
    questions = [
        LEGACY[task]["question"],
        "How many things are there? Look carefully at the picture.",
        "What is 3 + 4 in the image?",
    ]
    sizes = [(900, 700), (1280, 640), (640, 960), (433, 1001)]
    for frame in FRAMES:
        template = P.prompt_template(task, contract_for(kind, frame))
        assert template.count(P.QUESTION_SLOT) == 1
        rendered = set()
        for template_id, question, size, swap in itertools.product(range(5), questions, sizes, (False, True)):
            base = make_base(task, template=template_id, question=question, swap=swap)
            prompt = P.render_public_prompt(base, contract_for(kind, frame, size))
            expected = template.replace(P.QUESTION_SLOT, question).replace("{W}", str(size[0])).replace("{H}", str(size[1]))
            assert prompt == expected
            # remove question line and W x H: what is left is one constant per (task, frame)
            rest = prompt.split("\n", 1)[1].replace(f"{size[0]}x{size[1]}", "{W}x{H}")
            rendered.add(rest)
        assert len(rendered) == 1
        if frame == "rel1000":
            assert "{W}" not in rendered.copy().pop()


def test_prompt_is_independent_of_everything_but_task_and_variants() -> None:
    """render_public_prompt reads only ``task`` and ``prompt_variants`` (never labels or ids)."""
    seen: List[str] = []

    class Recorder(dict):
        def __getitem__(self, key):
            seen.append(key)
            return super().__getitem__(key)

        def get(self, key, default=None):
            seen.append(key)
            return super().get(key, default)

    task = CHESS
    base = Recorder(make_base(task))
    base.update(
        answer_gt={"type": "integer", "value": 777777},
        annotation_gt={"type": "bbox_set", "value": [[1, 2, 3, 4]]},
        reward_contract={"x": 1},
        trace_ref={"shard_id": "s"},
    )
    prompt = P.render_public_prompt(base, contract_for("selected_set"))
    assert set(seen) <= {"task", "prompt_variants", "instance_id"}
    assert "777777" not in prompt
    other = dict(make_base(task), answer_gt={"type": "integer", "value": 1}, instance_id="blake3:" + "00" * 32)
    assert P.render_public_prompt(other, contract_for("selected_set")) == prompt


def test_render_accepts_trace_record_like_objects() -> None:
    class Record:
        train = make_base(CHESS)

    assert P.render_public_prompt(Record(), contract_for("selected_set")) == render(CHESS)
    with pytest.raises(TypeError):
        P.render_public_prompt("not a row", contract_for("selected_set"))


def test_no_oracle_or_instance_vocabulary_in_fixed_texts() -> None:
    forbidden = re.compile(r"lever_weight_|cell_r\d|piece_\d|\br\d\d_c\d\d|values_by_row|selected_row|shortest_path|oracle|ground.?truth", re.I)
    texts = [BUNDLE["envelope"], *BUNDLE["frames"].values(), P.system_prompt()]
    for kind in P.CERT_KINDS:
        block = BUNDLE["kinds"][kind]
        texts += [block["certificate_line"], P.example_json(kind)]
        for task in block["tasks"].values():
            texts += list(task.values())
    for text in texts:
        assert forbidden.search(text) is None, text
    # fixed texts never state a set size or a target value of any instance
    for kind in P.CERT_KINDS:
        assert not re.search(r"\b(exactly|total of)\s+\d+\s+(rows|terms|members|steps)", P.certificate_line(kind))


# ----------------------------------------------------------------------------------------
# question assertion, drift checks, contract checks


@pytest.mark.parametrize("word", ["JSON", "json", "answer", "Answer", "ANSWER", "annotation", "Annotation"])
def test_question_with_format_word_is_quarantined(word: str) -> None:
    task = CHESS
    base = make_base(task, question=f"How many pieces can you {word} on the board?")
    with pytest.raises(QuarantineError) as info:
        P.render_public_prompt(base, contract_for("selected_set"))
    assert info.value.reason == "question_contains_format_word"


def test_question_edge_cases() -> None:
    task = CHESS
    with pytest.raises(QuarantineError) as info:
        P.question_line(make_base(task, question="   "))
    assert info.value.reason == "empty_question"
    with pytest.raises(QuarantineError) as info:
        P.question_line(make_base(task, question="<image>How many pieces?"))
    assert info.value.reason == "question_contains_image_marker"
    base = make_base(task)
    del base["prompt_variants"]["answer_only"]
    with pytest.raises(QuarantineError) as info:
        P.question_line(base)
    assert info.value.reason == "missing_prompt_variants"
    assert P.question_line(make_base(task)) == LEGACY[task]["question"]
    # a multi-line prompt contributes only its first line, even when line 2 mentions JSON
    for task in TASKS:
        base = make_base(task)
        assert P.question_line(base) == LEGACY[task]["question"]
        assert "\n" not in P.question_line(base)


def test_question_is_identical_in_both_legacy_variants() -> None:
    for task in TASKS:
        variants = make_base(task)["prompt_variants"]
        assert variants["answer_only"].split("\n", 1)[0] == variants["answer_and_annotation"].split("\n", 1)[0]


def test_unknown_task_and_hint_drift_are_quarantined() -> None:
    kind_contract = contract_for("selected_set")
    with pytest.raises(QuarantineError) as info:
        P.render_public_prompt(make_base("task_physics__lever__side_torque_value"), kind_contract)
    assert info.value.reason == "task_not_in_prompt_block"
    with pytest.raises(QuarantineError) as info:
        P.prompt_template("task_unknown", kind_contract)
    assert info.value.reason == "task_not_in_prompt_block"

    task = CHESS
    base = make_base(task)
    base["prompt_variants"]["answer_and_annotation"] = base["prompt_variants"]["answer_and_annotation"].replace("piece boxes", "square boxes")
    with pytest.raises(QuarantineError) as info:
        P.render_public_prompt(base, kind_contract)
    assert info.value.reason == "annotation_hint_drift"

    base = make_base(task)
    base["prompt_variants"]["answer_only"] = base["prompt_variants"]["answer_only"].replace("count of", "number of")
    with pytest.raises(QuarantineError) as info:
        P.render_public_prompt(base, kind_contract)
    assert info.value.reason == "answer_hint_drift"

    base = make_base(task)
    base["prompt_variants"]["answer_and_annotation"] = LEGACY[task]["question"] + "\nno hint here"
    with pytest.raises(QuarantineError):
        P.render_public_prompt(base, kind_contract)
    # a row without the annotation variant is still rendered from the stored texts
    base = make_base(task)
    del base["prompt_variants"]["answer_and_annotation"]
    assert P.render_public_prompt(base, kind_contract) == render(task)


def test_frame_word_removal() -> None:
    assert P.strip_frame_words("an array of [x0,y0,x1,y1] pixel boxes around") == "an array of [x0,y0,x1,y1] boxes around"
    assert P.strip_frame_words("a JSON array of pixel-space piece boxes") == "a JSON array of piece boxes"
    assert P.strip_frame_words("one [x0, y0, x1, y1] pixel bounding box") == "one [x0, y0, x1, y1] bounding box"
    assert P.strip_frame_words("boxes with no frame words") == "boxes with no frame words"
    with pytest.raises(QuarantineError) as info:
        P.strip_frame_words("pixels everywhere")
    assert info.value.reason == "frame_wording_left"
    for task, spec in LEGACY.items():
        assert P.strip_frame_words(spec["ann"]) == BUNDLE["kinds"][spec["kind"]]["tasks"][task]["annotation_hint"]


def test_stored_hints_equal_trace_texts_for_every_output_template() -> None:
    """All five TRACE output-template phrasings and both line orders map to the same prompt."""
    for task in TASKS:
        expected = render(task)
        for template, swap in itertools.product(range(5), (False, True)):
            assert render(task, template=template, swap=swap) == expected


def test_contract_mismatch_is_a_version_error() -> None:
    block = BUNDLE["kinds"]["selected_set"]
    base = make_base(CHESS)
    good = contract_for("selected_set")
    for field, value in (("list_cap", block["list_cap"] + 1), ("schema_id", "cert.selected_set.v2"), ("annotation_type", "bbox_map"), ("list_key", "rows")):
        kwargs = {**good.__dict__, field: value}
        with pytest.raises(CertificateVersionError):
            P.render_public_prompt(base, PublicContract(**kwargs))
    with pytest.raises(CertificateVersionError):
        P.render_public_prompt(base, PublicContract(**{**good.__dict__, "kind": "unknown_kind"}))
    with pytest.raises(KeyError):
        P.public_schema("unknown_kind")


def test_bundle_matches_public_contract_constants() -> None:
    from trace_tasks.experimental.certificates.types import MAX_OBJECT_BYTES

    assert tuple(BUNDLE["frames"]) == ("px_source", "rel1000")
    assert tuple(BUNDLE["kinds"]) == P.CERT_KINDS
    assert {k: v["list_cap"] for k, v in BUNDLE["kinds"].items()} == {"selected_set": 64, "filtered_mean": 16, "weighted_sum": 8, "path": 32}
    assert contract_for("path").max_object_bytes == MAX_OBJECT_BYTES
    # every supported task of the design has exactly one prompt block
    assert sorted(t for v in BUNDLE["kinds"].values() for t in v["tasks"]) == TASKS


# ----------------------------------------------------------------------------------------
# legacy conditions and system prompt


def test_legacy_prompts_are_trace_exports() -> None:
    for task in TASKS:
        base = make_base(task)
        variants = base["prompt_variants"]
        assert P.legacy_prompt(base, "ans_v1") == "<image>" + variants["answer_only"].strip()
        # TRACE's export drops the redundant "Use a valid JSON object ..." line and nothing else
        kept = [ln for ln in variants["answer_and_annotation"].split("\n") if not ln.startswith("Use a valid JSON object")]
        assert P.legacy_prompt(base, "ann_v1") == "<image>" + "\n".join(kept).strip()
        # the legacy conditions keep TRACE's own wording, frame words included
        assert P.legacy_prompt(base, "ann_v1").count("<image>") == 1
    with pytest.raises(ValueError):
        P.legacy_prompt(make_base(CHESS), "cert_v1")


def test_legacy_prompt_strips_redundant_contract_line_like_trace() -> None:
    base = make_base("task_physics__lever__side_torque_value")
    base["prompt_variants"]["answer_only"] = (
        LEGACY[base["task"]]["question"] + '\nUse a valid JSON object with key "answer" for the final answer.\nRequired answer format: x.\nExample JSON:\n{"answer":12}'
    )
    assert "Use a valid JSON object" not in P.legacy_prompt(base, "ans_v1")
    assert P.legacy_prompt(base, "ans_v1").startswith("<image>" + LEGACY[base["task"]]["question"])


def test_system_prompt_is_design_text() -> None:
    expected = (
        "You are a helpful, conversational assistant tasked with answering a question about an image. "
        "Reason carefully from the image and the question to determine the answer. "
        "End your response with a JSON object in the format the question specifies."
    )
    assert P.system_prompt() == expected
    raw = P.system_prompt_path().read_bytes()
    assert raw.decode("utf-8") == expected + "\n" and raw.isascii()
    # the first two sentences are TRACE's own system prompt, verbatim
    trace_sentences = (
        "You are a helpful, conversational assistant tasked with answering a question about an image.",
        "Reason carefully from the image and the question to determine the answer.",
    )
    assert all(sentence in expected for sentence in trace_sentences)
    assert "{" not in expected and "answer\":" not in expected


def test_resource_digests() -> None:
    digests = P.resource_digests()
    assert len(digests) == 2 + len(P.CERT_KINDS) + 1
    assert all(re.fullmatch(r"[0-9a-f]{64}", d) for d in digests.values())
    assert digests == P.resource_digests()
    assert P.prompt_sha256(render(CHESS)) == P.prompt_sha256(render(CHESS))
    assert P.prompt_sha256(render(CHESS, "px_source")) != P.prompt_sha256(render(CHESS, "rel1000"))


def test_load_prompt_bundle_returns_a_copy() -> None:
    copy = P.load_prompt_bundle()
    copy["kinds"]["path"]["list_cap"] = 1
    assert P.load_prompt_bundle()["kinds"]["path"]["list_cap"] == 32
    assert P.certificate_line("path") == BUNDLE["kinds"]["path"]["certificate_line"]


# ----------------------------------------------------------------------------------------
# hygiene


def _released_files() -> List[Path]:
    return sorted(p for p in RESOURCE_DIR.rglob("*") if p.is_file()) + [PACKAGE_DIR / "prompts.py", Path(__file__)]


def test_no_machine_paths_in_prompt_resources() -> None:
    for path in _released_files():
        text = path.read_text(encoding="utf-8")
        for needle in ("/" + "scratch", "/" + "home/", "/" + "uufs", "/" + "Users/"):
            assert needle not in text, (path.name, needle)


def test_release_hygiene_scanner_accepts_the_new_files() -> None:
    scanner = pytest.importorskip("scripts.check_public_release", reason="release checker not importable")
    for path in _released_files():
        findings = scanner.scan_text_for_sensitive_values(path.read_text(encoding="utf-8"))
        assert not findings, (path.name, findings)


def test_resources_are_valid_utf8_json_with_trailing_newline() -> None:
    for path in sorted(RESOURCE_DIR.rglob("*.json")):
        raw = path.read_bytes()
        assert raw.endswith(b"\n")
        json.loads(raw.decode("utf-8"))


# ----------------------------------------------------------------------------------------
# optional: every row of real generated datasets (local runs; set TRACE_CERT_PROMPT_DATASETS)


def _dataset_rows() -> List[Dict[str, Any]]:
    rows = []
    for pattern in filter(None, os.environ.get("TRACE_CERT_PROMPT_DATASETS", "").split(os.pathsep)):
        for path in glob.glob(os.path.join(pattern, "**", "train_instances.jsonl"), recursive=True):
            with open(path, encoding="utf-8") as handle:
                rows += [json.loads(line) for line in handle if line.strip()]
    return [r for r in rows if r.get("task") in LEGACY]


@pytest.mark.skipif(not os.environ.get("TRACE_CERT_PROMPT_DATASETS"), reason="set TRACE_CERT_PROMPT_DATASETS to run on generated datasets")
def test_generated_rows_render_with_one_template_per_task_and_frame() -> None:
    rows = _dataset_rows()
    assert rows, "no supported rows found under TRACE_CERT_PROMPT_DATASETS"
    constants: Dict[Any, str] = {}
    for row in rows:
        kind = LEGACY[row["task"]]["kind"]
        for frame in FRAMES:
            prompt = P.render_public_prompt(row, contract_for(kind, frame, (800, 600)))
            question = P.question_line(row)
            rest = prompt.split("\n", 1)[1].replace("800x600", "{W}x{H}")
            assert prompt.split("\n", 1)[0] == "<image>" + question
            assert constants.setdefault((row["task"], frame), rest) == rest
            assert "answer_gt" not in prompt
        assert P.legacy_prompt(row, "ans_v1").startswith("<image>")
        assert P.legacy_prompt(row, "ann_v1").startswith("<image>")
