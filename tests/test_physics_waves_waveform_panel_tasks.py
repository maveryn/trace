"""Contract tests for physics waveform-panel tasks."""

from __future__ import annotations

from trace_tasks.tasks.physics.waveform_panel.wave_property_extremum_label import (
    PhysicsWaveformPanelWavePropertyExtremumLabelTask,
)


def _assert_bbox_in_bounds(out) -> None:
    width, height = out.image.size
    assert out.annotation_gt.type == "bbox"
    bbox = out.annotation_gt.value
    assert 0 <= bbox[0] < bbox[2] <= width
    assert 0 <= bbox[1] < bbox[3] <= height


def _panel_by_label(out, label: str) -> dict:
    panels = out.trace_payload["render_map"]["panels"]
    return next(panel for panel in panels if str(panel["label"]) == str(label))


def test_physics_waveform_highest_amplitude_contract() -> None:
    out = PhysicsWaveformPanelWavePropertyExtremumLabelTask().generate(
        81101,
        params={
            "query_id": "highest_amplitude_label",
            "panel_count": 5,
            "target_label": "C",
        },
        max_attempts=20,
    )
    panels = out.trace_payload["render_map"]["panels"]
    selected = _panel_by_label(out, "C")

    assert out.scene_id == "waveform_panel"
    assert out.query_id == "highest_amplitude_label"
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "C"
    assert int(selected["amplitude_rank"]) == max(int(panel["amplitude_rank"]) for panel in panels)
    assert out.annotation_gt.value == selected["bbox_px"]
    assert out.trace_payload["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert out.prompt_variants["answer_only"]
    assert out.prompt_variants["answer_and_annotation"]
    _assert_bbox_in_bounds(out)


def test_physics_waveform_shortest_wavelength_uses_max_cycle_count() -> None:
    out = PhysicsWaveformPanelWavePropertyExtremumLabelTask().generate(
        81103,
        params={
            "query_id": "shortest_wavelength_label",
            "panel_count": 6,
            "target_label": "B",
        },
        max_attempts=20,
    )
    panels = out.trace_payload["render_map"]["panels"]
    selected = _panel_by_label(out, "B")

    assert out.query_id == "shortest_wavelength_label"
    assert out.answer_gt.value == "B"
    assert int(selected["cycle_count"]) == max(int(panel["cycle_count"]) for panel in panels)
    assert float(selected["wavelength_relative"]) == min(float(panel["wavelength_relative"]) for panel in panels)
    assert out.trace_payload["execution_trace"]["query_property"] == "wavelength"
    assert out.trace_payload["execution_trace"]["query_extremum"] == "shortest"
    _assert_bbox_in_bounds(out)


def test_physics_waveform_all_query_branches_have_unique_selected_extrema() -> None:
    task = PhysicsWaveformPanelWavePropertyExtremumLabelTask()
    query_checks = {
        "highest_amplitude_label": lambda panel, panels: int(panel["amplitude_rank"])
        == max(int(item["amplitude_rank"]) for item in panels),
        "lowest_amplitude_label": lambda panel, panels: int(panel["amplitude_rank"])
        == min(int(item["amplitude_rank"]) for item in panels),
        "highest_frequency_label": lambda panel, panels: int(panel["cycle_count"])
        == max(int(item["cycle_count"]) for item in panels),
        "lowest_frequency_label": lambda panel, panels: int(panel["cycle_count"])
        == min(int(item["cycle_count"]) for item in panels),
        "longest_wavelength_label": lambda panel, panels: int(panel["cycle_count"])
        == min(int(item["cycle_count"]) for item in panels),
        "shortest_wavelength_label": lambda panel, panels: int(panel["cycle_count"])
        == max(int(item["cycle_count"]) for item in panels),
    }
    for offset, (query_id, predicate) in enumerate(query_checks.items()):
        out = task.generate(
            81130 + offset,
            params={"query_id": query_id, "panel_count": 6, "target_label": "E"},
            max_attempts=20,
        )
        panels = out.trace_payload["render_map"]["panels"]
        selected = _panel_by_label(out, "E")

        assert out.answer_gt.value == "E"
        assert predicate(selected, panels)
        assert out.trace_payload["query_spec"]["params"]["answer_support"] == ["A", "B", "C", "D", "E", "F"]


def test_physics_waveform_generation_is_deterministic() -> None:
    task = PhysicsWaveformPanelWavePropertyExtremumLabelTask()
    params = {"query_id": "lowest_frequency_label", "panel_count": 4}

    first = task.generate(81177, params=params, max_attempts=20)
    second = task.generate(81177, params=params, max_attempts=20)

    assert first.answer_gt == second.answer_gt
    assert first.annotation_gt == second.annotation_gt
    assert first.trace_payload["render_map"] == second.trace_payload["render_map"]
    assert first.image.tobytes() == second.image.tobytes()
