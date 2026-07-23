from ....core.seed import spawn_rng
from ....core.scene_config import get_domain_defaults, get_scene_defaults
from ...registry import register_task
from ...shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from ..shared.object_scene import SCENE_ID, _camera_yaw_band_for_instance
from .shared.annotations import draw_marked_points as _render_marked_point_overlay
from ..shared.object_scene_point_order_output import generate_point_order_once
from .shared.sampling import CAMERA_DISTANCE_BRANCH, MIN_CAMERA_DISTANCE_MARGIN, MIN_SCREEN_DEPTH_STEP_PX, build_camera_distance_order_dataset


TASK_ID = "task_three_d__object_scene__point_camera_distance_order_label"
SUPPORTED_QUERY_IDS = (CAMERA_DISTANCE_BRANCH,)


_SCENE_DEFAULTS = get_scene_defaults("three_d", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, dict) else {},
    task_id=TASK_ID,
)
_DOMAIN_DEFAULTS = get_domain_defaults("three_d")
_VISUAL_DEFAULTS = _DOMAIN_DEFAULTS.get("visual", {}) if isinstance(_DOMAIN_DEFAULTS, dict) else {}
_BACKGROUND_DEFAULTS = _VISUAL_DEFAULTS.get("background", {}) if isinstance(_VISUAL_DEFAULTS, dict) else {}
_NOISE_DEFAULTS = _VISUAL_DEFAULTS.get("noise", {}) if isinstance(_VISUAL_DEFAULTS, dict) else {}


@register_task
class ThreeDObjectScenePointCameraDistanceOrderLabelTask:
    task_id = TASK_ID
    reasoning_operations = ('ranking', 'spatial_relations')
    supported_query_ids = SUPPORTED_QUERY_IDS
    domain = "three_d"
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        last_error = None
        camera_yaw_band = _camera_yaw_band_for_instance(int(instance_seed))
        for attempt_index in range(max(1, int(max_attempts))):
            attempt_seed = (
                int(instance_seed)
                if attempt_index == 0
                else int(spawn_rng(int(instance_seed), f"{TASK_ID}.attempt_seed.{attempt_index}").randrange(1, 2**62))
            )
            try:
                return generate_point_order_once(
                    owner_domain=self.domain,
                    objective_name=TASK_ID,
                    branch_options=SUPPORTED_QUERY_IDS,
                    dataset_builder=build_camera_distance_order_dataset,
                    order_axis="camera_distance_near_to_far",
                    scene_kind="three_d_object_scene_point_camera_distance_order",
                    instance_seed=int(attempt_seed),
                    params=params,
                    camera_yaw_band=camera_yaw_band,
                    gen_defaults=_GEN_DEFAULTS,
                    render_defaults=_RENDER_DEFAULTS,
                    prompt_defaults=_PROMPT_DEFAULTS,
                    background_defaults=_BACKGROUND_DEFAULTS,
                    noise_defaults=_NOISE_DEFAULTS,
                    draw_marked_points_fn=_render_marked_point_overlay,
                )
            except Exception as exc:  # pragma: no cover - unlucky sampling fallback.
                last_error = exc
        raise RuntimeError(f"{self.task_id} failed to generate a valid scene after {max_attempts} attempts: {last_error}")
