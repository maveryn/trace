from __future__ import annotations

from trace_tasks.tasks.geometry.shared.metadata_serialization import geometry_json_ready


def test_geometry_json_ready_rounds_nested_float_values() -> None:
    payload = {
        7: (1.23456, {"bbox": [2.34567, 3.45678]}),
        "label": "A",
    }

    assert geometry_json_ready(payload) == {
        "7": [1.235, {"bbox": [2.346, 3.457]}],
        "label": "A",
    }


def test_geometry_json_ready_preserves_float_values_when_disabled() -> None:
    payload = {
        "segments": ((1.23456, 2.34567),),
        3: [{"value": 4.56789}],
    }

    assert geometry_json_ready(payload, round_floats=False) == {
        "segments": [[1.23456, 2.34567]],
        "3": [{"value": 4.56789}],
    }


def test_geometry_json_ready_uses_custom_float_precision() -> None:
    assert geometry_json_ready({"x": 1.23456}, ndigits=1) == {"x": 1.2}
