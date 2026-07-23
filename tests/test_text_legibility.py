"""Tests for shared text legibility utilities and style integration."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from scripts.audit_text_legibility import (
    font_routing_findings,
    renderer_role_metadata_findings,
    renderer_text_migration_findings,
    semantic_text_color_findings,
)
from trace_tasks.core import error_codes
from trace_tasks.core.validation import _validate_text_legibility_contract
from trace_tasks.tasks.shared.text_legibility import (
    READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO,
    TextLegibilityRecorder,
    collect_traced_text_records,
    contrast_ratio,
    draw_text_traced,
    resolve_readable_text_style,
    traced_text_records_summary,
)
from trace_tasks.tasks.shared.text_rendering import load_font, symbol_safe_font_for_text, text_needs_symbol_safe_font
from trace_tasks.tasks.shared.visual_style.information_scene import resolve_information_scene_style
from trace_tasks.tasks.shared.visual_style.panel import resolve_panel_scene_style
from trace_tasks.tasks.shared.visual_style.technical_diagram import resolve_technical_diagram_style


def test_resolve_readable_text_style_passes_required_contrast() -> None:
    style = resolve_readable_text_style(
        instance_seed=12027,
        namespace="test.text_legibility",
        role="read_required_test",
        surface_rgbs=((255, 255, 255), (245, 247, 250)),
        preferred_rgbs=((90, 92, 95),),
    )

    assert style.min_contrast_ratio >= READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO
    assert contrast_ratio(style.fill_rgb, (255, 255, 255)) >= READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO
    assert style.metadata()["passes"] is True


def test_information_scene_records_required_text_legibility() -> None:
    _style, metadata = resolve_information_scene_style(
        instance_seed=12028,
        namespace="test.information_scene",
        treatments=("presentation_slide",),
        palettes=("pastel_dashboard",),
    )

    legibility = metadata["text_legibility"]
    assert legibility["enabled"] is True
    assert legibility["required_role_count"] >= 3
    assert legibility["failure_count"] == 0
    assert metadata["text_color_policy"] == "read_required_text_uses_random_nonsemantic_readable_ink"


def test_technical_diagram_records_required_text_legibility() -> None:
    _style, metadata = resolve_technical_diagram_style(
        instance_seed=12029,
        namespace="test.technical_diagram",
        treatments=("exam_problem_box",),
        palettes=("graphite_blue",),
    )

    legibility = metadata["text_legibility"]
    assert legibility["enabled"] is True
    assert legibility["required_role_count"] >= 1
    assert legibility["failure_count"] == 0
    assert metadata["text_color_policy"] == "read_required_text_uses_random_nonsemantic_readable_ink"


def test_panel_scene_records_required_text_legibility() -> None:
    _style, metadata = resolve_panel_scene_style(
        instance_seed=12030,
        namespace="test.panel_scene",
        treatments=("worksheet_panel",),
    )

    legibility = metadata["text_legibility"]
    assert legibility["enabled"] is True
    assert legibility["required_role_count"] == 1
    assert legibility["failure_count"] == 0
    assert metadata["text_color_policy"] == "read_required_text_uses_random_nonsemantic_readable_ink"


def test_active_assets_do_not_make_glyph_text_color_semantic() -> None:
    findings = semantic_text_color_findings(Path(".").resolve())
    assert findings == []


def test_text_legibility_recorder_tracks_drawn_bbox() -> None:
    image = Image.new("RGB", (180, 90), (250, 252, 255))
    draw = ImageDraw.Draw(image)
    style = resolve_readable_text_style(
        instance_seed=12031,
        namespace="test.recorder",
        role="node_label",
        surface_rgbs=((250, 252, 255),),
    )
    recorder = TextLegibilityRecorder(canvas_size_px=(180, 90))
    record = recorder.draw_centered_text(
        draw,
        center=(90, 45),
        text="A",
        font=load_font(24),
        style=style,
        stroke_width=1,
    )

    metadata = recorder.metadata()
    assert metadata["policy_version"] == "text_legibility_v1"
    assert metadata["drawn_text_record_count"] == 1
    assert metadata["failure_count"] == 0
    assert record["bbox_px"][2] > record["bbox_px"][0]
    assert record["bbox_px"][3] > record["bbox_px"][1]


def test_core_validation_rejects_failing_text_legibility_record() -> None:
    trace_record = {
        "render_spec": {
            "text_legibility": {
                "enabled": True,
                "failure_count": 1,
                "records": [
                    {
                        "role": "axis_tick",
                        "required": True,
                        "passes": False,
                        "min_contrast_ratio": 2.0,
                        "min_contrast_required": 7.0,
                        "min_lab_distance": 12.0,
                        "min_lab_distance_required": 38.0,
                        "bbox_px": [10, 10, 30, 24],
                    }
                ],
            }
        }
    }

    errors = _validate_text_legibility_contract(trace_record, instance_id="iid")
    assert any(error.error_code == error_codes.TEXT_LEGIBILITY_CONTRAST_FAILED for error in errors)


def test_renderer_text_migration_audit_flags_direct_task_draw_text(tmp_path: Path) -> None:
    direct_path = tmp_path / "trace" / "tasks" / "demo.py"
    direct_path.parent.mkdir(parents=True)
    direct_path.write_text("def render(draw):\n    draw.text((0, 0), 'A')\n", encoding="utf-8")

    findings = renderer_text_migration_findings(tmp_path, scan_roots=("src/trace_tasks/tasks",))
    assert findings == ["src/trace_tasks/tasks/demo.py:2: draw.text((0, 0), 'A')"]


def test_global_text_collection_records_compatibility_draw() -> None:
    image = Image.new("RGB", (160, 80), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = load_font(18)

    with collect_traced_text_records() as records:
        draw_text_traced(
            draw,
            (12, 18),
            "Label",
            font=font,
            fill=(20, 26, 34),
            role="test_label",
            required=False,
        )

    assert len(records) == 1
    assert records[0]["role"] == "test_label"
    assert records[0]["bbox_px"][2] > records[0]["bbox_px"][0]
    summary = traced_text_records_summary(records)
    assert summary["source"] == "automatic_drawn_text_collector"
    assert summary["drawn_text_record_count"] == 1
    assert summary["failure_count"] == 0


def test_symbol_safe_font_fallback_is_used_for_math_readout_tokens() -> None:
    sampled_font = load_font(24, bold=True, font_family="yanone_kaffeesatz")
    symbol_font = symbol_safe_font_for_text("∠ABC=?", sampled_font)

    assert text_needs_symbol_safe_font("∠ABC=?") is True
    assert text_needs_symbol_safe_font("ABC=?") is False
    assert str(getattr(symbol_font, "path", "")) != str(getattr(sampled_font, "path", ""))
    assert "vollkorn" in str(getattr(symbol_font, "path", "")).casefold()


def test_role_metadata_audit_flags_missing_role_and_required(tmp_path: Path) -> None:
    direct_path = tmp_path / "trace" / "tasks" / "demo.py"
    direct_path.parent.mkdir(parents=True)
    direct_path.write_text(
        "def render(draw, font):\n"
        "    draw_text_traced(draw, (0, 0), 'A', font=font, fill=(0, 0, 0))\n",
        encoding="utf-8",
    )

    findings = renderer_role_metadata_findings(tmp_path, scan_roots=("src/trace_tasks/tasks",))
    assert findings == ["src/trace_tasks/tasks/demo.py:2: draw_text_traced missing explicit required, role"]


def test_font_routing_audit_flags_direct_imagefont_loading(tmp_path: Path) -> None:
    direct_path = tmp_path / "trace" / "tasks" / "demo.py"
    direct_path.parent.mkdir(parents=True)
    direct_path.write_text(
        "from PIL import ImageFont\n"
        "def render():\n"
        "    return ImageFont.truetype('/tmp/example.ttf', 12)\n",
        encoding="utf-8",
    )

    findings = font_routing_findings(tmp_path, scan_roots=("src/trace_tasks/tasks",))
    assert findings == ["src/trace_tasks/tasks/demo.py:3: direct ImageFont.truetype bypasses shared font routing"]
