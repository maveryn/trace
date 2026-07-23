from trace_tasks.tasks.three_d.street import intersection_nearest_label as intersection_nearest
from trace_tasks.tasks.three_d.street import _lifecycle as street_lifecycle
from trace_tasks.tasks.three_d.street.shared import rendering as intersection_rendering
from trace_tasks.tasks.three_d.street.shared import state as intersection_scene


def test_intersection_nearest_uses_shared_street_scene_and_renderer_boundaries() -> None:
    assert intersection_nearest._StreetRenderParams is intersection_scene._StreetRenderParams
    assert street_lifecycle.render_street_intersection_scene_3d is intersection_rendering.render_street_intersection_scene_3d
    assert intersection_nearest.build_street_option_label_task_output is street_lifecycle.build_street_option_label_task_output
    assert intersection_nearest._sample_context_specs is intersection_scene._sample_context_specs


def test_street_orientation_dimensions_are_axis_aware() -> None:
    x_dims = intersection_scene._dimensions_for_orientation("car", orientation_axis="x", scale=1.0)
    y_dims = intersection_scene._dimensions_for_orientation("car", orientation_axis="y", scale=1.0)
    assert x_dims[0] == y_dims[1]
    assert x_dims[1] == y_dims[0]
