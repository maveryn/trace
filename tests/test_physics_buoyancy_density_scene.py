"""Focused tests for the migrated buoyancy-density physics scene."""

from __future__ import annotations

import math

import pytest

from trace_tasks.tasks.physics.buoyancy_density.object_density_value import (
    PhysicsBuoyancyDensityObjectDensityValueTask,
)


def test_buoyancy_density_scene_uses_single_query_contract() -> None:
    task = PhysicsBuoyancyDensityObjectDensityValueTask()
    out = task.generate(
        81231,
        params={
            "query_id": "single",
            "scene_variant": "beaker_tank",
            "object_shape": "rounded_block",
            "submerged_fraction": "3/4",
            "liquid_density_tenths": 16,
            "target_answer": 1.2,
        },
        max_attempts=10,
    )

    assert out.scene_id == "buoyancy_density"
    assert out.query_id == "single"
    assert out.trace_payload["query_spec"]["query_id"] == "single"
    assert out.answer_gt.type == "number"
    assert math.isclose(float(out.answer_gt.value), 1.2, abs_tol=1e-9)
    assert out.annotation_gt.type == "bbox"
    assert out.annotation_gt.value == out.trace_payload["render_map"]["floating_object_bbox_px"]
    assert out.trace_payload["projected_annotation"]["bbox"] == out.annotation_gt.value


def test_buoyancy_density_rejects_retired_branch_name() -> None:
    task = PhysicsBuoyancyDensityObjectDensityValueTask()
    with pytest.raises(ValueError, match="unsupported query_id"):
        task.generate(
            81232,
            params={"query_id": "floating_object_density_value"},
            max_attempts=10,
        )
