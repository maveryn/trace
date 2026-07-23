"""Contract tests for the physics mechanics pulley mechanical-advantage task."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.tasks.physics.pulley.pulley_mechanical_advantage import PhysicsPulleyMechanicalAdvantageTask
from tests.helpers import read_jsonl


@pytest.mark.parametrize(
    ("params", "expected_answer", "expected_annotation_count", "expected_query_id", "expected_solve_for"),
    (
        (
            {
                "scene_variant": "compact_block",
                "query_id": "missing_effort_force_value",
                "target_answer": 12,
                "support_segment_count": 5,
                "disconnected_segment_count": 4,
            },
            12,
            3,
            "missing_effort_force_value",
            "effort_force",
        ),
        (
            {
                "scene_variant": "tall_block",
                "query_id": "missing_load_force_value",
                "target_answer": 60,
                "support_segment_count": 5,
                "disconnected_segment_count": 4,
            },
            60,
            3,
            "missing_load_force_value",
            "load_force",
        ),
    ),
)
def test_physics_mechanics_pulley_emits_expected_contract(
    params: dict[str, int | str],
    expected_answer: int,
    expected_annotation_count: int | None,
    expected_query_id: str,
    expected_solve_for: str,
) -> None:
    out = PhysicsPulleyMechanicalAdvantageTask().generate(39001, params=params, max_attempts=40)
    trace = out.trace_payload
    execution = trace["execution_trace"]


    assert out.answer_gt.type == "integer"

    assert int(out.answer_gt.value) == int(expected_answer)

    assert out.annotation_gt.type == "bbox_map"

    assert out.query_id == expected_query_id

    assert trace["query_spec"]["query_id"] == expected_query_id

    assert trace["query_spec"]["params"]["query_id"] == expected_query_id

    assert execution["query_id"] == expected_query_id
    assert execution["solve_for"] == expected_solve_for
    expected_count = (
        int(expected_annotation_count)
        if expected_annotation_count is not None
        else int(execution["support_segment_count"]) + 6
    )

    assert len(out.annotation_gt.value) == int(expected_count)

    assert trace["projected_annotation"]["type"] == "bbox_map"
    assert trace["projected_annotation"]["bbox_map"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_bbox_map"] == out.annotation_gt.value
    assert trace["render_map"]["annotation_bbox_map_px"] == out.annotation_gt.value
    assert set(out.annotation_gt.value) == {
        "supporting_strands_region",
        "known_force_label",
        "unknown_force_label",
    }
    assert (
        out.annotation_gt.value["supporting_strands_region"]
        == trace["render_map"]["supporting_strands_region_bbox_px"]
    )
    assert trace["witness_symbolic"]["type"] == "object_map"
    assert trace["render_spec"]["font"]["selection_policy"]["pool"] == "global_approved_font_pool"
    assert trace["render_spec"]["layout_placement"]["mode"] == "whole_pulley_diagram_offset"

    assert 2 <= int(execution["support_segment_count"]) <= 6

    assert 0 <= int(execution["disconnected_segment_count"]) <= 4

    assert len(execution["cut_segments"]) == int(execution["disconnected_segment_count"])

    assert all(0.25 <= float(segment["cut_fraction"]) <= 0.75 for segment in execution["cut_segments"])

    assert all(segment["supports_moving_block"] is False for segment in execution["cut_segments"])

    assert int(execution["load_force_value"]) == int(execution["effort_force_value"]) * int(execution["support_segment_count"])
    if str(execution["solve_for"]) == "effort_force":

        assert execution["shown_effort_force_value"] is None

        assert int(execution["shown_load_force_value"]) == int(execution["load_force_value"])
    else:

        assert str(execution["solve_for"]) == "load_force"

        assert int(execution["shown_effort_force_value"]) == int(execution["effort_force_value"])

        assert execution["shown_load_force_value"] is None


def test_physics_mechanics_pulley_accepts_legacy_solve_for_param() -> None:
    out = PhysicsPulleyMechanicalAdvantageTask().generate(
        39011,
        params={
            "scene_variant": "compact_block",
            "solve_for": "load_force",
            "target_answer": 60,
            "support_segment_count": 5,
            "disconnected_segment_count": 1,
        },
        max_attempts=40,
    )

    assert out.query_id == "missing_load_force_value"
    assert out.trace_payload["execution_trace"]["query_id"] == "missing_load_force_value"
    assert out.trace_payload["execution_trace"]["solve_for"] == "load_force"


def test_physics_mechanics_pulley_is_deterministic() -> None:
    params = {
        "scene_variant": "tall_block",
        "solve_for": "load_force",
        "target_answer": 84,
        "support_segment_count": 6,
        "disconnected_segment_count": 4,
        "accent_color_name": "cyan",
    }
    task = PhysicsPulleyMechanicalAdvantageTask()
    out_a = task.generate(39021, params=params, max_attempts=40)
    out_b = task.generate(39021, params=params, max_attempts=40)

    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()

    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()

    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]

    assert out_a.prompt == out_b.prompt

    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_physics_mechanics_pulley_accepts_explicit_accent_color() -> None:
    out = PhysicsPulleyMechanicalAdvantageTask().generate(
        39031,
        params={
            "scene_variant": "open_block",
            "solve_for": "effort_force",
            "target_answer": 10,
            "support_segment_count": 6,
            "disconnected_segment_count": 0,
            "accent_color_name": "orange",
        },
        max_attempts=40,
    )

    assert int(out.trace_payload["execution_trace"]["disconnected_segment_count"]) == 0

    assert out.trace_payload["execution_trace"]["cut_segments"] == []

    assert str(out.trace_payload["execution_trace"]["accent_color_name"]) == "orange"

    assert str(out.trace_payload["render_map"]["accent_color_name"]) == "orange"


def test_physics_mechanics_pulley_rejects_unknown_scene_variant() -> None:
    with pytest.raises(ValueError):
        PhysicsPulleyMechanicalAdvantageTask().generate(
            39041,
            params={"scene_variant": "sideways_block", "solve_for": "effort_force"},
            max_attempts=20,
        )


def test_physics_mechanics_pulleyseeded_sampler_decouples_variant_and_answer_support() -> None:
    task = PhysicsPulleyMechanicalAdvantageTask()
    answers_by_solve_for: dict[str, set[int]] = {
        "effort_force": set(),
        "load_force": set(),
    }
    scenes_by_solve_for: dict[str, Counter[str]] = defaultdict(Counter)
    cuts_by_support: dict[int, set[int]] = defaultdict(set)
    support_cut_pairs: Counter[tuple[int, int]] = Counter()
    combos: Counter[tuple[str, str]] = Counter()
    for sampling_index in range(240):
        out = task.generate(
            39100 + sampling_index,
            params={},
            max_attempts=60,
        )

        execution = out.trace_payload["execution_trace"]
        solve_for = str(execution["solve_for"])
        expected_query_id = (
            "missing_effort_force_value" if solve_for == "effort_force" else "missing_load_force_value"
        )
        assert str(out.query_id) == expected_query_id
        assert str(execution["query_id"]) == expected_query_id
        assert str(out.trace_payload["query_spec"]["query_id"]) == expected_query_id
        scene_variant = str(out.trace_payload["query_spec"]["params"]["scene_variant"])
        support_count = int(execution["support_segment_count"])
        cut_count = int(execution["disconnected_segment_count"])
        answers_by_solve_for[solve_for].add(int(out.answer_gt.value))
        scenes_by_solve_for[solve_for][scene_variant] += 1
        cuts_by_support[support_count].add(cut_count)
        support_cut_pairs[(support_count, cut_count)] += 1
        combos[(solve_for, scene_variant)] += 1


    assert answers_by_solve_for["effort_force"] == set(range(4, 19))

    assert answers_by_solve_for["load_force"].issubset({
        8,
        10,
        12,
        14,
        15,
        16,
        18,
        20,
        21,
        22,
        24,
        25,
        26,
        27,
        28,
        30,
        32,
        33,
        34,
        35,
        36,
        39,
        40,
        42,
        44,
        45,
        48,
        50,
        51,
        52,
        54,
        55,
        56,
        60,
        64,
        65,
        66,
        68,
        70,
        72,
        75,
        78,
        80,
        84,
        85,
        90,
        96,
        102,
        108,
    })
    assert len(answers_by_solve_for["load_force"]) >= 40
    for solve_for, counts in scenes_by_solve_for.items():

        assert set(counts.keys()) == {"open_block", "compact_block", "tall_block"}, solve_for

        assert len(combos) == 6

        assert set(cuts_by_support.keys()) == {2, 3, 4, 5, 6}

        assert all(len(cut_counts) >= 3 for cut_counts in cuts_by_support.values())

        assert len(support_cut_pairs) >= 18


def test_physics_mechanics_pulley_prompt_bundle_supports_variants() -> None:
    bundle = json.loads(Path("src/trace_tasks/resources/prompts/physics/pulley/physics_pulley_v1.json").read_text(encoding="utf-8"))

    assert len(bundle["templates"]["scene"]["pulley_system_diagram"]) == 5

    assert set(bundle["templates"]["query"]) == {
        "missing_effort_force_value",
        "missing_load_force_value",
    }

    assert len(bundle["templates"]["query"]["missing_effort_force_value"]) == 5

    assert len(bundle["templates"]["query"]["missing_load_force_value"]) == 5


def test_physics_mechanics_pulley_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_physics__pulley__pulley_mechanical_advantage"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_physics__pulley__pulley_mechanical_advantage",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_physics__pulley__pulley_mechanical_advantage",
                count=4,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=40,
        sampling_seed=91,
    )
    final_path = build_dataset(config, code_hash="physics-mechanics-pulley-smoke")

    assert final_path.exists()
    train_records = read_jsonl(final_path / "train_instances.jsonl")

    assert len(train_records) == 4

    assert all(record["domain"] == "physics" for record in train_records)

    assert all(record["scene_id"] == "pulley" for record in train_records)

    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))

    assert int(build_report["accepted_counts_by_task"]["task_physics__pulley__pulley_mechanical_advantage"]) == 4

    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))

    assert validation["total_errors"] == 0
