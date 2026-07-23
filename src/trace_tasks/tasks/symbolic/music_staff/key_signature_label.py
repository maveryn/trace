"""Read the key name from a visible staff key signature."""

from __future__ import annotations

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ...registry import register_task

from ._lifecycle import DOMAIN, MusicStaffRuntime, load_music_staff_defaults, run_music_staff_runtime
from .shared.components import MusicSceneSpec, StaffSystem, make_staff_note_sequence
from .shared.output import build_text_option_dataset
from .shared.rules import KEY_SIGNATURES, key_signature_text, random_staff_pitch, title_for


TASK_ID = "task_symbolic__music_staff__key_signature_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_KEY = "music_key_signature_label"
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_music_staff_defaults(TASK_ID)


def _build_dataset(
    *,
    branch_key,
    instance_seed,
    scene_variant,
    params,
    gen_defaults,
):
    """Bind the key-signature bbox while rendering context notes on the staff."""

    if str(branch_key) != "key_signature_label":
        raise ValueError(f"unsupported music-staff branch for key-signature task: {branch_key}")
    _ = params, gen_defaults
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.dataset")
    visible_key_labels = tuple(key for key in KEY_SIGNATURES if key != "C major")
    key_label = rng.choice(visible_key_labels)
    notes = make_staff_note_sequence(
        tuple(random_staff_pitch(rng, low_step=0, high_step=7, allow_accidental=False) for _ in range(4)),
        item_ids=tuple(f"context_note_{index + 1}" for index in range(4)),
        start_slot=1.0,
        slot_step=0.85,
    )
    spec = MusicSceneSpec(
        title=title_for(scene_variant, "key signature"),
        systems=(
            StaffSystem(
                clef="treble",
                slot_count=6,
                key_signature=key_signature_text(key_label),
                key_signature_id="key_signature",
                notes=notes,
            ),
        ),
    )
    dataset_kwargs = dict(
        branch_key=str(branch_key),
        correct_text=str(key_label),
        candidate_texts=tuple(KEY_SIGNATURES.keys()),
        annotation_item_ids=("key_signature",),
        spec=spec,
        scene_variant=str(scene_variant),
        prompt_slots={},
        metadata={"key_label": str(key_label)},
    )
    return build_text_option_dataset(rng, **dataset_kwargs)


_RUNTIME = MusicStaffRuntime(TASK_ID, SUPPORTED_QUERY_IDS, PROMPT_KEY, "bbox", _build_dataset, _GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS, "key_signature_label", "key_signature_label")

@register_task
class SymbolicKeySignatureLabelTask:
    """Read a visible key signature label."""

    task_id = TASK_ID
    reasoning_operations = ('matching',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params, max_attempts):
        return run_music_staff_runtime(_RUNTIME, instance_seed, params=params, max_attempts=max_attempts)
