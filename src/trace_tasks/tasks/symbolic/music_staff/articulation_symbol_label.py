"""Read the articulation symbol attached to a marked note."""

from __future__ import annotations

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ...registry import register_task

from ._lifecycle import DOMAIN, MusicStaffRuntime, load_music_staff_defaults, run_music_staff_runtime
from .shared.components import MusicSceneSpec, Pitch, StaffSymbol, StaffSystem, make_staff_note_sequence
from .shared.output import build_text_option_dataset
from .shared.rules import ARTICULATION_SYMBOLS, title_for


TASK_ID = "task_symbolic__music_staff__articulation_symbol_label"
PROMPT_QUERY_KEY = "articulation_symbol_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_KEY = "music_articulation_symbol_label"
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_music_staff_defaults(TASK_ID)


_ARTICULATION_PITCH_SUPPORT = (Pitch("A", 4), Pitch("B", 4), Pitch("G", 4), Pitch("C", 5), Pitch("F", 4))
_ARTICULATION_SLOT_STEP = 2.20


def _sample_articulation_excerpt(rng, target_index: int, target_symbol: str):
    """Return note and symbol layers for one numbered articulation target."""

    notes = make_staff_note_sequence(
        _ARTICULATION_PITCH_SUPPORT,
        item_ids=tuple(
            "symbol_note" if index == int(target_index) else f"distractor_symbol_note_{index + 1}"
            for index in range(5)
        ),
    )
    symbols = []
    for note_index, pitch in enumerate(_ARTICULATION_PITCH_SUPPORT):
        symbol = str(target_symbol) if note_index == int(target_index) else rng.choice(ARTICULATION_SYMBOLS)
        symbol_id = "target_symbol" if note_index == int(target_index) else f"distractor_symbol_{note_index + 1}"
        symbols.append(
            StaffSymbol(
                symbol_id,
                0,
                1.0 + note_index * _ARTICULATION_SLOT_STEP,
                pitch,
                symbol,
                marker=str(note_index + 1),
            )
        )
    return notes, tuple(symbols)


def _build_dataset(
    *,
    branch_key,
    instance_seed,
    scene_variant,
    params,
    gen_defaults,
):
    """Bind one numbered articulation mark."""

    if str(branch_key) != "articulation_symbol_label":
        raise ValueError(f"unsupported music-staff branch for articulation task: {branch_key}")
    _ = params, gen_defaults
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.dataset")
    target_index = rng.randrange(5)
    target_symbol = rng.choice(ARTICULATION_SYMBOLS)
    notes, symbols = _sample_articulation_excerpt(rng, target_index, target_symbol)
    spec = MusicSceneSpec(
        title=title_for(scene_variant, "articulation"),
        systems=(StaffSystem(clef="treble", slot_count=7, notes=tuple(notes), symbols=tuple(symbols)),),
    )
    articulation_kwargs = {
        "branch_key": str(branch_key),
        "correct_text": str(target_symbol),
        "candidate_texts": ARTICULATION_SYMBOLS,
        "annotation_item_ids": ("target_symbol",),
        "spec": spec,
        "scene_variant": str(scene_variant),
        "prompt_slots": {"target_marker": str(target_index + 1)},
        "metadata": {"symbol": str(target_symbol), "target_marker": str(target_index + 1)},
    }
    return build_text_option_dataset(rng, **articulation_kwargs)


_RUNTIME = MusicStaffRuntime(TASK_ID, SUPPORTED_QUERY_IDS, PROMPT_KEY, "point", _build_dataset, _GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS, PROMPT_QUERY_KEY, PROMPT_QUERY_KEY)

@register_task
class SymbolicArticulationSymbolLabelTask:
    """Read the articulation symbol attached to a marked note."""

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
