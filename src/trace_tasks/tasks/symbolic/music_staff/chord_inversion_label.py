from __future__ import annotations

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ...registry import register_task

from ._lifecycle import DOMAIN, MusicStaffRuntime, load_music_staff_defaults, run_music_staff_runtime
from .shared.components import MusicSceneSpec, StaffSystem
from .shared.output import build_text_option_dataset
from .shared.rules import INVERSION_NAMES, build_numbered_inversion_chords, inversion_quality_support, title_for
from .shared.sampling import resolve_named_choice


TASK_ID = "task_symbolic__music_staff__chord_inversion_label"
PROMPT_QUERY_KEY = "chord_inversion_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_KEY = "music_chord_inversion_label"
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_music_staff_defaults(TASK_ID)


def _build_dataset(
    *,
    branch_key,
    instance_seed,
    scene_variant,
    params,
    gen_defaults,
):
    """Sample the target inversion and bind one numbered chord as the witness."""

    if str(branch_key) != PROMPT_QUERY_KEY:
        raise ValueError(f"unsupported music-staff branch for chord-inversion task: {branch_key}")
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.dataset")
    inversion_name, _probabilities = resolve_named_choice(
        params,
        gen_defaults,
        instance_seed=int(instance_seed),
        namespace=TASK_ID,
        values=INVERSION_NAMES,
        explicit_key="chord_inversion",
        weights_key="chord_inversion_weights",
        balance_flag_key="balanced_chord_inversion_sampling",
        axis_namespace="chord_inversion",
    )
    inversion = int(INVERSION_NAMES.index(str(inversion_name)))
    target_index = rng.randrange(4)
    quality = rng.choice(inversion_quality_support(inversion))
    return build_text_option_dataset(
        rng,
        branch_key=PROMPT_QUERY_KEY,
        correct_text=str(INVERSION_NAMES[inversion]),
        candidate_texts=INVERSION_NAMES,
        annotation_item_ids=("target_chord",),
        spec=MusicSceneSpec(
            title=title_for(scene_variant, "chord inversion"),
            systems=(StaffSystem(clef="treble", slot_count=7, chords=build_numbered_inversion_chords(rng, target_index=target_index, target_quality=str(quality), target_inversion=inversion)),),
        ),
        scene_variant=str(scene_variant),
        prompt_slots={"target_marker": str(target_index + 1)},
        metadata={
            "inversion_index": int(inversion),
            "target_marker": str(target_index + 1),
        },
    )


_RUNTIME = MusicStaffRuntime(TASK_ID, SUPPORTED_QUERY_IDS, PROMPT_KEY, "bbox", _build_dataset, _GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS, PROMPT_QUERY_KEY, PROMPT_QUERY_KEY)


@register_task
class SymbolicChordInversionLabelTask:
    task_id = TASK_ID
    reasoning_operations = ('matching',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params, max_attempts):
        return run_music_staff_runtime(
            _RUNTIME,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )
