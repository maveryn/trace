"""Identity-free sampling primitives for cube-net puzzle cases."""

from __future__ import annotations

from string import ascii_uppercase
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import sample_without_replacement, uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.named_colors import sample_named_color_palette
from trace_tasks.tasks.shared.config_defaults import group_default

from .rules import (
    canonical_face_assignment_signature,
    cube_rotation_matrices,
    face_across_display_side,
    rotate_face_assignment,
)
from .state import (
    DEFAULTS,
    FACE_IDS,
    FACE_LABEL_POOL,
    NET_COORDS,
    NetEquivalenceDataset,
    NetEquivalenceOption,
    FaceOption,
    FaceRelationDataset,
    OPPOSITE_FACE,
    SIDE_OFFSETS,
)


def resolve_scene_int(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    key: str,
    fallback: int,
) -> int:
    """Resolve one integer from task params, scene defaults, then code fallback."""

    return int(params.get(str(key), group_default(defaults, str(key), int(fallback))))


def sample_face_labels(instance_seed: int, namespace: str) -> Dict[str, str]:
    """Assign unique visible labels to the six cube faces."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.face_labels")
    labels = sample_without_replacement(rng, list(FACE_LABEL_POOL), len(FACE_IDS))
    return {face: str(label) for face, label in zip(FACE_IDS, labels)}


def resolve_option_count(
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
) -> int:
    """Resolve the visible MCQ option count for cube-net answer panels."""

    option_count = resolve_scene_int(
        params,
        generation_defaults,
        "option_count",
        DEFAULTS.option_count,
    )
    if int(option_count) < 2 or int(option_count) > len(FACE_IDS):
        raise ValueError("cube-net option_count must be between 2 and the number of cube faces")
    return int(option_count)


def option_order(
    *,
    face_labels: Mapping[str, str],
    correct_face: str,
    option_count: int,
    instance_seed: int,
    namespace: str,
    excluded_faces: Sequence[str] = (),
) -> Tuple[Tuple[FaceOption, ...], str]:
    """Build one face-label option set with exactly one correct option."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.option_order")
    if int(option_count) < 2 or int(option_count) > len(FACE_IDS):
        raise ValueError("cube-net option_count must be between 2 and the number of cube faces")
    excluded = {str(face) for face in excluded_faces}
    excluded.discard(str(correct_face))
    available_faces = [
        str(face)
        for face in FACE_IDS
        if str(face) != str(correct_face) and str(face) not in excluded
    ]
    if int(option_count) - 1 > len(available_faces):
        raise ValueError("cube-net option_count exceeds available distractor faces")
    correct_index = int(rng.randrange(int(option_count)))
    distractors = list(available_faces)
    rng.shuffle(distractors)
    ordered_faces = list(distractors[: max(0, int(option_count) - 1)])
    ordered_faces.insert(correct_index, str(correct_face))
    labels = tuple(ascii_uppercase[index] for index in range(len(ordered_faces)))
    options = tuple(
        FaceOption(
            option_label=str(label),
            face_id=str(face),
            face_label=str(face_labels[str(face)]),
        )
        for label, face in zip(labels, ordered_faces)
    )
    return options, str(labels[correct_index])


def _exposed_net_edge_pairs() -> Tuple[Tuple[str, str], ...]:
    """Return face-side pairs whose side is not shared in the flat net."""

    occupied_coords = {tuple(coord) for coord in NET_COORDS.values()}
    pairs: list[tuple[str, str]] = []
    for face in FACE_IDS:
        x, y = NET_COORDS[str(face)]
        for side, (dx, dy) in SIDE_OFFSETS.items():
            if (int(x + dx), int(y + dy)) not in occupied_coords:
                pairs.append((str(face), str(side)))
    return tuple(pairs)


def sample_face_relation_dataset(
    *,
    relation_kind: str,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> FaceRelationDataset:
    """Sample a face-relation puzzle without receiving public query identity."""

    option_count = resolve_option_count(params, generation_defaults)
    rng = spawn_rng(int(instance_seed), f"{namespace}.{relation_kind}.face_relation")
    face_labels = sample_face_labels(int(instance_seed), f"{namespace}.{relation_kind}")
    reference_face = str(uniform_choice(rng, FACE_IDS))
    marked_side: str | None = None
    if str(relation_kind) == "opposite":
        correct_face = str(OPPOSITE_FACE[reference_face])
    elif str(relation_kind) == "edge_neighbor":
        reference_face, marked_side = uniform_choice(rng, _exposed_net_edge_pairs())
        correct_face = face_across_display_side(reference_face, marked_side)
    else:
        raise ValueError(f"unsupported face relation kind: {relation_kind}")
    options, correct_label = option_order(
        face_labels=face_labels,
        correct_face=correct_face,
        option_count=int(option_count),
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.{relation_kind}",
        excluded_faces=(reference_face,),
    )
    return FaceRelationDataset(
        relation_kind=str(relation_kind),
        face_labels=dict(face_labels),
        reference_face=reference_face,
        marked_side=marked_side,
        correct_face=correct_face,
        options=tuple(options),
        correct_option_label=str(correct_label),
    )


def _face_assignment_tuple(face_color_names: Mapping[str, str]) -> Tuple[str, ...]:
    """Return one stable face-order tuple for duplicate-option rejection."""

    return tuple(str(face_color_names[str(face)]) for face in FACE_IDS)


def _swap_face_colors(
    face_color_names: Mapping[str, str],
    first_face: str,
    second_face: str,
) -> Dict[str, str]:
    """Return a copy with two face colors exchanged."""

    swapped = {str(face): str(color) for face, color in face_color_names.items()}
    swapped[str(first_face)], swapped[str(second_face)] = (
        swapped[str(second_face)],
        swapped[str(first_face)],
    )
    return swapped


def sample_equivalent_net_dataset(
    *,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> NetEquivalenceDataset:
    """Sample one colored-net equivalence task with exactly one matching option."""

    option_count = resolve_option_count(params, generation_defaults)
    if int(option_count) != 4:
        raise ValueError("cube-net equivalence currently requires exactly four options")
    rng = spawn_rng(int(instance_seed), f"{namespace}.equivalent_net")
    palette = sample_named_color_palette(rng, palette_size=len(FACE_IDS))
    if len(palette) != len(FACE_IDS):
        raise ValueError("cube-net equivalence requires six named colors")
    reference = {
        str(face): str(color_name)
        for face, (color_name, _rgb) in zip(FACE_IDS, palette)
    }
    reference_signature = canonical_face_assignment_signature(reference)

    rotated_candidates = [
        rotate_face_assignment(reference, rotation)
        for rotation in cube_rotation_matrices()
    ]
    non_identity_rotations = [
        candidate
        for candidate in rotated_candidates
        if _face_assignment_tuple(candidate) != _face_assignment_tuple(reference)
    ]
    correct_face_colors = dict(uniform_choice(rng, tuple(non_identity_rotations)))
    correct_signature = canonical_face_assignment_signature(correct_face_colors)
    if correct_signature != reference_signature:
        raise ValueError("sampled correct net is not equivalent to the reference")

    distractors: list[NetEquivalenceOption] = []
    seen_assignments = {
        _face_assignment_tuple(reference),
        _face_assignment_tuple(correct_face_colors),
    }
    seen_signatures = {reference_signature}
    for attempt in range(240):
        base = correct_face_colors if attempt % 2 == 0 else reference
        first_face, second_face = rng.sample(list(FACE_IDS), k=2)
        candidate = _swap_face_colors(base, str(first_face), str(second_face))
        assignment_key = _face_assignment_tuple(candidate)
        if assignment_key in seen_assignments:
            continue
        signature = canonical_face_assignment_signature(candidate)
        if signature in seen_signatures:
            continue
        distractors.append(
            NetEquivalenceOption(
                option_label="",
                face_color_names=dict(candidate),
                equivalence_kind="non_equivalent_swap",
                canonical_signature=tuple(signature),
            )
        )
        seen_assignments.add(assignment_key)
        seen_signatures.add(tuple(signature))
        if len(distractors) == 3:
            break
    if len(distractors) != 3:
        raise ValueError("could not sample three unique non-equivalent cube-net distractors")

    correct_index = int(rng.randrange(4))
    labels = tuple(ascii_uppercase[index] for index in range(4))
    unlabeled_options: list[NetEquivalenceOption] = list(distractors)
    unlabeled_options.insert(
        correct_index,
        NetEquivalenceOption(
            option_label="",
            face_color_names=dict(correct_face_colors),
            equivalence_kind="equivalent_by_cube_rotation",
            canonical_signature=tuple(correct_signature),
        ),
    )
    options = tuple(
        NetEquivalenceOption(
            option_label=str(label),
            face_color_names=dict(option.face_color_names),
            equivalence_kind=str(option.equivalence_kind),
            canonical_signature=tuple(option.canonical_signature),
        )
        for label, option in zip(labels, unlabeled_options)
    )
    return NetEquivalenceDataset(
        reference_face_color_names=dict(reference),
        reference_signature=tuple(reference_signature),
        options=tuple(options),
        correct_option_label=str(labels[correct_index]),
    )


def face_option_specs(options: Sequence[FaceOption]) -> list[dict[str, str]]:
    """Convert face options to JSON-friendly trace records."""

    return [
        {
            "option_label": str(option.option_label),
            "face_id": str(option.face_id),
            "face_label": str(option.face_label),
        }
        for option in options
    ]


def equivalent_net_option_specs(options: Sequence[NetEquivalenceOption]) -> list[dict[str, Any]]:
    """Convert colored-net options to JSON-friendly trace records."""

    return [
        {
            "option_label": str(option.option_label),
            "face_color_names": dict(option.face_color_names),
            "equivalence_kind": str(option.equivalence_kind),
            "canonical_signature": list(option.canonical_signature),
        }
        for option in options
    ]


__all__ = [
    "equivalent_net_option_specs",
    "face_option_specs",
    "option_order",
    "resolve_option_count",
    "resolve_scene_int",
    "sample_equivalent_net_dataset",
    "sample_face_labels",
    "sample_face_relation_dataset",
]
