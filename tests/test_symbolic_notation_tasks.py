"""Contract tests for symbolic music-staff notation tasks."""

from __future__ import annotations

from pathlib import Path

import pytest

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks import TASK_REGISTRY
from trace_tasks.tasks.symbolic.music_staff.articulation_symbol_label import (
    TASK_ID as ARTICULATION_SYMBOL_TASK_ID,
    SymbolicArticulationSymbolLabelTask,
)
from trace_tasks.tasks.symbolic.music_staff.chord_inversion_label import (
    TASK_ID as CHORD_INVERSION_TASK_ID,
    SymbolicChordInversionLabelTask,
)
from trace_tasks.tasks.symbolic.music_staff.chord_quality_label import (
    TASK_ID as CHORD_QUALITY_TASK_ID,
    SymbolicChordQualityLabelTask,
)
from trace_tasks.tasks.symbolic.music_staff.duration_equivalence_label import (
    TASK_ID as DURATION_EQUIVALENCE_TASK_ID,
    SymbolicDurationEquivalenceLabelTask,
)
from trace_tasks.tasks.symbolic.music_staff.interval_name_label import (
    TASK_ID as INTERVAL_NAME_TASK_ID,
    SymbolicIntervalNameLabelTask,
)
from trace_tasks.tasks.symbolic.music_staff.key_signature_label import (
    TASK_ID as KEY_SIGNATURE_TASK_ID,
    SymbolicKeySignatureLabelTask,
)
from trace_tasks.tasks.symbolic.music_staff.meter_type_count import (
    TASK_ID as METER_TYPE_TASK_ID,
    SymbolicMeterTypeCountTask,
)
from trace_tasks.tasks.symbolic.music_staff.note_name_label import (
    TASK_ID as NOTE_NAME_TASK_ID,
    SymbolicNoteNameLabelTask,
)
from trace_tasks.tasks.symbolic.music_staff.roman_numeral_label import (
    TASK_ID as ROMAN_NUMERAL_TASK_ID,
    SymbolicRomanNumeralLabelTask,
)
from trace_tasks.tasks.symbolic.music_staff.scale_degree_function_label import (
    TASK_ID as SCALE_DEGREE_FUNCTION_TASK_ID,
    SymbolicScaleDegreeFunctionLabelTask,
)
from trace_tasks.tasks.symbolic.music_staff.scale_validation_count import (
    TASK_ID as SCALE_VALIDATION_COUNT_TASK_ID,
    SymbolicScaleValidationCountTask,
)
from trace_tasks.tasks.symbolic.music_staff.shared.state import SCENE_ID
from trace_tasks.tasks.symbolic.music_staff.shared.components import (
    Pitch,
    _duration_visual_style,
    _stem_up_for_pitch,
)
from trace_tasks.tasks.symbolic.music_staff.transposed_pitch_pair_count import (
    TASK_ID as TRANSPOSED_PITCH_PAIR_COUNT_TASK_ID,
    SymbolicTransposedPitchPairCountTask,
)


TASKS = (
    (NOTE_NAME_TASK_ID, SymbolicNoteNameLabelTask, ((SINGLE_QUERY_ID, "note_name_label"),), "bbox"),
    (INTERVAL_NAME_TASK_ID, SymbolicIntervalNameLabelTask, ((SINGLE_QUERY_ID, "interval_name_label"),), "bbox"),
    (TRANSPOSED_PITCH_PAIR_COUNT_TASK_ID, SymbolicTransposedPitchPairCountTask, ((SINGLE_QUERY_ID, "transposed_pitch_pair_count"),), "bbox_set"),
    (KEY_SIGNATURE_TASK_ID, SymbolicKeySignatureLabelTask, ((SINGLE_QUERY_ID, "key_signature_label"),), "bbox"),
    (SCALE_VALIDATION_COUNT_TASK_ID, SymbolicScaleValidationCountTask, ((SINGLE_QUERY_ID, "scale_validation_count"),), "bbox_set"),
    (SCALE_DEGREE_FUNCTION_TASK_ID, SymbolicScaleDegreeFunctionLabelTask, ((SINGLE_QUERY_ID, "scale_degree_function_label"),), "bbox_map"),
    (CHORD_QUALITY_TASK_ID, SymbolicChordQualityLabelTask, ((SINGLE_QUERY_ID, "chord_quality_label"),), "bbox"),
    (CHORD_INVERSION_TASK_ID, SymbolicChordInversionLabelTask, ((SINGLE_QUERY_ID, "chord_inversion_label"),), "bbox"),
    (ROMAN_NUMERAL_TASK_ID, SymbolicRomanNumeralLabelTask, ((SINGLE_QUERY_ID, "roman_numeral_label"),), "bbox_map"),
    (DURATION_EQUIVALENCE_TASK_ID, SymbolicDurationEquivalenceLabelTask, ((SINGLE_QUERY_ID, "duration_equivalence_label"),), "bbox"),
    (METER_TYPE_TASK_ID, SymbolicMeterTypeCountTask, ((SINGLE_QUERY_ID, "meter_type_count"),), "bbox_set"),
    (ARTICULATION_SYMBOL_TASK_ID, SymbolicArticulationSymbolLabelTask, ((SINGLE_QUERY_ID, "articulation_symbol_label"),), "point"),
)

QUERY_CASES = tuple(
    (task_cls, public_query_id, internal_branch_key, annotation_schema)
    for _task_id, task_cls, query_cases, annotation_schema in TASKS
    for public_query_id, internal_branch_key in query_cases
)


def _iter_bboxes(annotation_type: str, value):
    if annotation_type == "bbox":
        yield value
    elif annotation_type == "bbox_map":
        yield from value.values()
    else:
        yield from value


def _bbox_min_side(bbox) -> float:
    return min(float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1]))


def _bbox_center_x(bbox) -> float:
    return (float(bbox[0]) + float(bbox[2])) / 2.0


def _entities(out, entity_type: str):
    return [
        dict(entity)
        for entity in out.trace_payload["scene_ir"]["entities"]
        if str(entity.get("entity_type")) == str(entity_type)
    ]


def test_music_staff_tasks_are_registered() -> None:
    for task_id, task_cls, _query_cases, _annotation_schema in TASKS:
        assert TASK_REGISTRY[task_id] is task_cls
        task = task_cls()
        assert task.domain == "symbolic"
        assert not hasattr(task, "scene_id")
        module_path = Path(task_cls.__module__.replace(".", "/") + ".py")
        assert str(module_path).endswith(f"src/trace_tasks/tasks/symbolic/music_staff/{task_id.rsplit('__', 1)[-1]}.py")


@pytest.mark.parametrize("task_cls, public_query_id, internal_branch_key, annotation_schema", QUERY_CASES)
def test_music_staff_branches_emit_public_contract(
    task_cls,
    public_query_id: str,
    internal_branch_key: str,
    annotation_schema: str,
) -> None:
    out = task_cls().generate(
        2026052401,
        params={"query_id": public_query_id, "scene_variant": "engraved_sheet"},
        max_attempts=80,
    )
    trace = out.trace_payload

    assert out.scene_id == SCENE_ID
    assert out.query_id == public_query_id
    assert out.annotation_gt.type == annotation_schema
    assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]
    assert '"answer"' in out.prompt_variants["answer_only"]
    assert "annotation" in out.prompt_variants["answer_and_annotation"].lower()

    assert trace["query_spec"]["query_id"] == public_query_id
    assert trace["query_spec"]["internal_query_id"] == internal_branch_key
    assert trace["query_spec"]["params"]["scene_id"] == SCENE_ID
    assert trace["query_spec"]["params"]["query_id"] == public_query_id
    assert trace["query_spec"]["params"]["internal_query_id"] == internal_branch_key
    assert trace["query_spec"]["params"]["prompt_query_key"] == internal_branch_key
    assert trace["render_spec"]["scene_id"] == SCENE_ID
    assert trace["render_spec"]["scene_style"]
    assert trace["render_spec"]["music_style"]
    assert trace["render_map"]["annotation_source"] == "item_bboxes_px"
    assert "scene_title" not in trace["render_map"]["item_bboxes_px"]
    assert trace["projected_annotation"]["type"] == out.annotation_gt.type
    assert trace["execution_trace"]["answer_value"] == out.answer_gt.value
    assert trace["execution_trace"]["answer_type"] == out.answer_gt.type
    notation_meta = trace["execution_trace"]["notation_metadata"]
    if "option_texts" in notation_meta:
        option_texts = tuple(str(value) for value in notation_meta["option_texts"])
        option_ids = tuple(sorted(key for key in trace["render_map"]["item_bboxes_px"] if str(key).startswith("option_")))
        assert str(notation_meta["correct_option_text"]) in option_texts
        assert len(option_ids) == len(option_texts)
        assert out.answer_gt.value in trace["query_spec"]["params"]["target_answer_support"]
        assert trace["render_map"]["item_bboxes_px"][f"option_{out.answer_gt.value}"]

    width, height = out.image.size
    assert width == int(trace["render_spec"]["canvas_width"])
    assert height == int(trace["render_spec"]["canvas_height"])
    if out.annotation_gt.type == "point":
        point = out.annotation_gt.value
        assert len(point) == 2
        assert trace["projected_annotation"]["point"] == point
        assert trace["projected_annotation"]["pixel_point"] == point
        assert 0 <= float(point[0]) <= width
        assert 0 <= float(point[1]) <= height
    else:
        for bbox in _iter_bboxes(out.annotation_gt.type, out.annotation_gt.value):
            assert len(bbox) == 4
            assert 0 <= float(bbox[0]) < float(bbox[2]) <= width
            assert 0 <= float(bbox[1]) < float(bbox[3]) <= height


def test_music_staff_uses_compact_canvas_height() -> None:
    out = SymbolicArticulationSymbolLabelTask().generate(
        2026062601,
        params={"query_id": SINGLE_QUERY_ID, "scene_variant": "engraved_sheet"},
        max_attempts=80,
    )
    assert out.image.size == (1180, 480)
    assert int(out.trace_payload["render_spec"]["canvas_height"]) == 480


def test_music_staff_uses_vector_clef_entities() -> None:
    out = SymbolicNoteNameLabelTask().generate(
        2026062806,
        params={"query_id": SINGLE_QUERY_ID, "scene_variant": "engraved_sheet"},
        max_attempts=80,
    )
    clef_entities = _entities(out, "music_clef")
    assert clef_entities
    assert all(entity["rendering"] == "vector" for entity in clef_entities)
    assert not any("text" in entity for entity in clef_entities)


@pytest.mark.parametrize(
    ("duration_units", "expected"),
    (
        (1, {"filled_notehead": True, "has_stem": True, "has_dot": False, "has_flag": True}),
        (2, {"filled_notehead": True, "has_stem": True, "has_dot": False, "has_flag": False}),
        (3, {"filled_notehead": True, "has_stem": True, "has_dot": True, "has_flag": False}),
        (4, {"filled_notehead": False, "has_stem": True, "has_dot": False, "has_flag": False}),
        (6, {"filled_notehead": False, "has_stem": True, "has_dot": True, "has_flag": False}),
        (8, {"filled_notehead": False, "has_stem": False, "has_dot": False, "has_flag": False}),
    ),
)
def test_music_staff_duration_units_map_to_standard_glyph_parts(duration_units: int, expected: dict[str, bool]) -> None:
    style = _duration_visual_style(int(duration_units))
    assert {
        "filled_notehead": style.filled,
        "has_stem": style.stem,
        "has_dot": style.dotted,
        "has_flag": style.flag,
    } == expected


def test_music_staff_stem_direction_is_pitch_based() -> None:
    assert _stem_up_for_pitch(Pitch("E", 4), "treble") is True
    assert _stem_up_for_pitch(Pitch("B", 4), "treble") is True
    assert _stem_up_for_pitch(Pitch("D", 5), "treble") is False


def test_music_staff_text_mcqs_render_text_only() -> None:
    cases = (
        (SymbolicNoteNameLabelTask, {"query_id": SINGLE_QUERY_ID}),
        (SymbolicIntervalNameLabelTask, {"query_id": SINGLE_QUERY_ID}),
        (SymbolicKeySignatureLabelTask, {"query_id": SINGLE_QUERY_ID}),
        (SymbolicScaleDegreeFunctionLabelTask, {"query_id": SINGLE_QUERY_ID}),
        (SymbolicChordQualityLabelTask, {"query_id": SINGLE_QUERY_ID}),
        (SymbolicRomanNumeralLabelTask, {"query_id": SINGLE_QUERY_ID}),
        (SymbolicChordInversionLabelTask, {"query_id": SINGLE_QUERY_ID}),
        (SymbolicDurationEquivalenceLabelTask, {"query_id": SINGLE_QUERY_ID}),
        (SymbolicArticulationSymbolLabelTask, {"query_id": SINGLE_QUERY_ID}),
    )
    for task_cls, params in cases:
        out = task_cls().generate(2026062801, params={**params, "scene_variant": "engraved_sheet"}, max_attempts=80)
        option_cards = _entities(out, "music_option_card")
        assert option_cards
        assert all(card["option_mode"] == "text" for card in option_cards)
        assert all(str(card["text"]) for card in option_cards)


def test_duration_equivalence_uses_text_duration_options_only() -> None:
    out = SymbolicDurationEquivalenceLabelTask().generate(
        2026062802,
        params={"query_id": SINGLE_QUERY_ID, "scene_variant": "engraved_sheet"},
        max_attempts=80,
    )
    option_cards = _entities(out, "music_option_card")
    assert len(option_cards) == 6
    assert all(card["option_mode"] == "text" for card in option_cards)
    assert all(str(card["text"]) for card in option_cards)
    assert out.annotation_gt.type == "bbox"
    assert len(out.annotation_gt.value) == 4


def test_interval_name_uses_single_range_bbox_annotation() -> None:
    out = SymbolicIntervalNameLabelTask().generate(
        2026062805,
        params={"query_id": SINGLE_QUERY_ID, "scene_variant": "engraved_sheet"},
        max_attempts=80,
    )
    assert out.annotation_gt.type == "bbox"
    assert out.trace_payload["execution_trace"]["annotation_item_ids"] == ["target_interval_range"]
    item_bboxes = out.trace_payload["render_map"]["item_bboxes_px"]
    assert out.annotation_gt.value == [float(value) for value in item_bboxes["target_interval_range"]]
    assert "target_interval_note_1" in item_bboxes
    assert "target_interval_note_2" in item_bboxes


def test_music_staff_marker_entities_are_not_annotation_witnesses() -> None:
    cases = (
        (SymbolicNoteNameLabelTask, {"query_id": SINGLE_QUERY_ID}),
        (SymbolicScaleDegreeFunctionLabelTask, {"query_id": SINGLE_QUERY_ID}),
        (SymbolicChordQualityLabelTask, {"query_id": SINGLE_QUERY_ID}),
        (SymbolicChordInversionLabelTask, {"query_id": SINGLE_QUERY_ID}),
        (SymbolicDurationEquivalenceLabelTask, {"query_id": SINGLE_QUERY_ID}),
        (SymbolicArticulationSymbolLabelTask, {"query_id": SINGLE_QUERY_ID}),
    )
    for task_cls, params in cases:
        out = task_cls().generate(2026062803, params={**params, "scene_variant": "engraved_sheet"}, max_attempts=80)
        marker_entities = _entities(out, "music_item_marker")
        item_bboxes = out.trace_payload["render_map"]["item_bboxes_px"]
        annotation_item_ids = tuple(out.trace_payload["execution_trace"]["annotation_item_ids"])
        assert marker_entities
        assert not any(str(item_id).endswith("_marker") for item_id in annotation_item_ids)
        assert not any(str(marker["entity_id"]) in annotation_item_ids for marker in marker_entities)
        for marker in marker_entities:
            target_bbox = item_bboxes[str(marker["target_item_id"])]
            assert float(marker["bbox_px"][3]) + 8.0 <= float(target_bbox[1])


def test_music_staff_numbered_items_have_clear_horizontal_marker_spacing() -> None:
    single_target_cases = (
        (SymbolicNoteNameLabelTask, 2026062804, {"query_id": SINGLE_QUERY_ID}),
        (SymbolicScaleDegreeFunctionLabelTask, 2026062804, {"query_id": SINGLE_QUERY_ID}),
        (SymbolicDurationEquivalenceLabelTask, 2026062802, {"query_id": SINGLE_QUERY_ID}),
        (SymbolicArticulationSymbolLabelTask, 6127900801642325, {"query_id": SINGLE_QUERY_ID}),
        (SymbolicChordQualityLabelTask, 2026062804, {"query_id": SINGLE_QUERY_ID}),
    )
    for task_cls, seed, params in single_target_cases:
        out = task_cls().generate(int(seed), params={**params, "scene_variant": "engraved_sheet"}, max_attempts=80)
        marker_centers = sorted(
            _bbox_center_x(marker["bbox_px"])
            for marker in _entities(out, "music_item_marker")
        )
        assert len(marker_centers) >= 4
        assert min(
            marker_centers[index + 1] - marker_centers[index]
            for index in range(len(marker_centers) - 1)
        ) >= 85.0


def test_music_staff_pair_ranges_have_clear_spacing_and_high_labels() -> None:
    cases = (
        (SymbolicTransposedPitchPairCountTask, {"query_id": SINGLE_QUERY_ID}),
    )
    for task_cls, params in cases:
        out = task_cls().generate(2026062703, params={**params, "scene_variant": "engraved_sheet"}, max_attempts=80)
        item_bboxes = out.trace_payload["render_map"]["item_bboxes_px"]
        range_bboxes = [item_bboxes[f"pair_{index}"] for index in range(1, 5)]
        label_bboxes = [item_bboxes[f"pair_{index}_label"] for index in range(1, 5)]
        range_centers = sorted(_bbox_center_x(bbox) for bbox in range_bboxes)
        assert min(
            range_centers[index + 1] - range_centers[index]
            for index in range(len(range_centers) - 1)
        ) >= 90.0
        for label_bbox, range_bbox in zip(label_bboxes, range_bboxes):
            assert float(label_bbox[3]) + 6.0 <= float(range_bbox[1])


def test_transposed_pitch_pair_accidentals_are_visually_attached_to_second_note() -> None:
    out = SymbolicTransposedPitchPairCountTask().generate(
        2026062700,
        params={"query_id": SINGLE_QUERY_ID, "scene_variant": "engraved_sheet"},
        max_attempts=80,
    )
    note_entities = {
        str(entity["entity_id"]): entity
        for entity in _entities(out, "music_note")
    }
    accidental_checked = 0
    for pair_index in range(1, 5):
        source_bbox = note_entities[f"pair_{pair_index}_source"]["bbox_px"]
        shown_entity = note_entities[f"pair_{pair_index}_shown"]
        if str(shown_entity["display_accidental"]):
            accidental_checked += 1
            assert float(shown_entity["bbox_px"][0]) - float(source_bbox[2]) >= 8.0
    assert accidental_checked >= 1


def test_music_staff_label_tasks_use_contextual_excerpts() -> None:
    excerpt_cases = (
        (SymbolicNoteNameLabelTask, {"query_id": SINGLE_QUERY_ID}, "music_note", 5),
        (SymbolicScaleDegreeFunctionLabelTask, {"query_id": SINGLE_QUERY_ID}, "music_note", 5),
        (SymbolicChordQualityLabelTask, {"query_id": SINGLE_QUERY_ID}, "music_chord", 4),
        (SymbolicDurationEquivalenceLabelTask, {"query_id": SINGLE_QUERY_ID}, "music_note", 5),
        (SymbolicArticulationSymbolLabelTask, {"query_id": SINGLE_QUERY_ID}, "music_articulation_symbol", 5),
        (SymbolicIntervalNameLabelTask, {"query_id": SINGLE_QUERY_ID}, "music_staff_range", 4),
    )
    for task_cls, params, entity_type, minimum_count in excerpt_cases:
        out = task_cls().generate(2026062804, params={**params, "scene_variant": "engraved_sheet"}, max_attempts=80)
        assert len(_entities(out, entity_type)) >= int(minimum_count)

@pytest.mark.parametrize(
    "task_cls, public_query_id",
    (
        (SymbolicNoteNameLabelTask, SINGLE_QUERY_ID),
        (SymbolicKeySignatureLabelTask, SINGLE_QUERY_ID),
        (SymbolicChordQualityLabelTask, SINGLE_QUERY_ID),
    ),
)
def test_music_staff_generation_is_deterministic(task_cls, public_query_id: str) -> None:
    params = {"query_id": public_query_id, "scene_variant": "notebook_staff"}
    out_a = task_cls().generate(2026052499, params=params, max_attempts=80)
    out_b = task_cls().generate(2026052499, params=params, max_attempts=80)

    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt == out_b.answer_gt
    assert out_a.annotation_gt == out_b.annotation_gt
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_count_annotations_contain_only_counted_items() -> None:
    cases = (
        (SymbolicTransposedPitchPairCountTask, "transposed_pitch_pair_count", "target_pair_indices_1based", "pair_{}"),
        (SymbolicScaleValidationCountTask, "scale_validation_count", "target_fragment_indices_1based", "fragment_{}"),
        (SymbolicMeterTypeCountTask, "meter_type_count", "target_measure_indices_1based", "measure_{}"),
    )
    for task_cls, internal_branch_key, metadata_key, item_pattern in cases:
        for target_answer in range(0, 5):
            params = {
                "query_id": SINGLE_QUERY_ID,
                "scene_variant": "engraved_sheet",
                "target_answer": int(target_answer),
            }
            if internal_branch_key == "meter_type_count":
                params["target_meter_type"] = "compound"
            out = task_cls().generate(2026060900 + int(target_answer), params=params, max_attempts=80)
            assert out.answer_gt.value == int(target_answer)
            assert out.query_id == SINGLE_QUERY_ID
            assert out.trace_payload["query_spec"]["internal_query_id"] == internal_branch_key
            assert out.annotation_gt.type == "bbox_set"
            assert len(out.annotation_gt.value) == int(target_answer)
            assert out.trace_payload["execution_trace"]["annotation_item_ids"] == [
                item_pattern.format(position)
                for position in out.trace_payload["execution_trace"]["notation_metadata"][metadata_key]
            ]


def test_chord_annotations_respect_minimum_bbox_side() -> None:
    cases = (
        (SymbolicChordQualityLabelTask, {"query_id": SINGLE_QUERY_ID}, "bbox"),
        (SymbolicChordInversionLabelTask, {"query_id": SINGLE_QUERY_ID}, "bbox"),
        (SymbolicRomanNumeralLabelTask, {"query_id": SINGLE_QUERY_ID}, "bbox_map"),
    )
    for task_cls, params, expected_schema in cases:
        for seed in range(2026062700, 2026062710):
            out = task_cls().generate(
                int(seed),
                params={**params, "scene_variant": "engraved_sheet"},
                max_attempts=80,
            )
            assert out.annotation_gt.type == expected_schema
            if expected_schema == "bbox":
                assert out.trace_payload["execution_trace"]["annotation_item_ids"] == ["target_chord"]
                assert _bbox_min_side(out.annotation_gt.value) >= 24.0
            else:
                assert out.trace_payload["execution_trace"]["annotation_item_ids"] == ["key_signature", "target_chord"]
                assert set(out.annotation_gt.value) == {"key_signature", "target_chord"}
                assert set(out.trace_payload["projected_annotation"]["bbox_map"]) == {"key_signature", "target_chord"}
                for bbox in out.annotation_gt.value.values():
                    assert _bbox_min_side(bbox) >= 24.0


def test_key_signature_annotations_respect_minimum_bbox_side() -> None:
    for seed in range(2026062700, 2026062710):
        out = SymbolicKeySignatureLabelTask().generate(
            int(seed),
            params={"query_id": SINGLE_QUERY_ID, "scene_variant": "engraved_sheet"},
            max_attempts=80,
        )
        assert out.annotation_gt.type == "bbox"
        assert _bbox_min_side(out.annotation_gt.value) >= 24.0


def test_key_signature_renders_positioned_treble_accidentals() -> None:
    expected = {
        "G major": (("#",), (8,)),
        "D major": (("#", "#"), (8, 5)),
        "F major": (("b",), (4,)),
        "Bb major": (("b", "b"), (4, 7)),
        "Eb major": (("b", "b", "b"), (4, 7, 3)),
    }
    seen: set[str] = set()
    for seed in range(2026062700, 2026062900):
        out = SymbolicKeySignatureLabelTask().generate(
            int(seed),
            params={"query_id": SINGLE_QUERY_ID, "scene_variant": "engraved_sheet"},
            max_attempts=80,
        )
        key_label = str(out.trace_payload["execution_trace"]["notation_metadata"]["key_label"])
        tokens, staff_steps = expected[key_label]
        entity = next(entity for entity in _entities(out, "music_key_signature"))
        assert tuple(entity["tokens"]) == tokens
        assert tuple(int(value) for value in entity["staff_steps"]) == staff_steps
        assert str(out.trace_payload["execution_trace"]["notation_metadata"]["correct_option_text"]) == key_label
        seen.add(key_label)
        if seen == set(expected):
            break
    assert seen == set(expected)


def test_scale_validation_uses_key_aware_accidentals() -> None:
    out = SymbolicScaleValidationCountTask().generate(
        2026062801,
        params={"query_id": SINGLE_QUERY_ID, "scene_variant": "engraved_sheet"},
        max_attempts=80,
    )
    metadata = out.trace_payload["execution_trace"]["notation_metadata"]
    assert str(metadata["fragments"][1]["display_accidentals"][0]) == "natural"
    assert str(metadata["fragments"][2]["display_accidentals"][0]) == "natural"
    natural_note_ids = {
        str(metadata["fragments"][1]["note_ids"][0]),
        str(metadata["fragments"][2]["note_ids"][0]),
    }
    note_entities = {
        str(entity["entity_id"]): entity
        for entity in _entities(out, "music_note")
    }
    assert all(str(note_entities[note_id]["display_accidental"]) == "♮" for note_id in natural_note_ids)
    assert all(not value for value in metadata["fragments"][0]["display_accidentals"])
    assert all(not value for value in metadata["fragments"][3]["display_accidentals"])


def test_scale_degree_suppresses_key_implied_accidentals() -> None:
    out = SymbolicScaleDegreeFunctionLabelTask().generate(
        2026062800,
        params={"query_id": SINGLE_QUERY_ID, "scene_variant": "engraved_sheet"},
        max_attempts=80,
    )
    metadata = out.trace_payload["execution_trace"]["notation_metadata"]
    assert metadata["note"] == "Eb4"
    assert metadata["correct_option_text"] == "subdominant"
    note_entities = {
        str(entity["entity_id"]): entity
        for entity in _entities(out, "music_note")
    }
    assert str(note_entities["degree_note"]["display_accidental"]) == ""
