"""Scene-neutral sampling helpers for cyclic-order puzzles."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.puzzles.shared.symbol_rendering import PUZZLE_OBJECT_TYPES
from trace_tasks.tasks.shared.color_distance import color_distance
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.render_variation import resolve_render_int, resolve_render_rgb
from trace_tasks.tasks.shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant

from .rules import (
    invalid_candidate_sequences,
    rotate_token_sequence,
    token_sequences_are_rotation_equivalent,
)
from .state import (
    DEFAULTS,
    LOOP_PATH_STYLES,
    LOOP_SHAPE_VARIANTS,
    LOOP_START_ANGLES_DEG,
    SCENE_VARIANTS,
    SOURCE_SCENE_VARIANT_MAP,
    SOURCE_TOKEN_MODE_MAP,
    TOKEN_COLOR_SPECS,
    TOKEN_RENDER_STYLES,
    TOKEN_STYLE_SOURCE_MODE,
    CyclicOrderDefaults,
    CyclicOrderRenderParams,
)


def _axis_variant(
    *,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    instance_seed: int,
    supported: Sequence[str],
    explicit_key: str,
    weights_key: str,
    balance_key: str,
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve one visual/sampling axis with explicit, weighted, and balanced modes."""

    rng = spawn_rng(int(instance_seed), str(namespace))
    selected, probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=generation_defaults,
        supported_variants=supported,
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=generation_defaults,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=supported,
        balance_flag_key=str(balance_key),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        sampling_namespace=f"{str(namespace)}.balanced",
    )
    return str(selected), {str(key): float(value) for key, value in sorted(probabilities.items())}


def resolve_scene_variant(
    *,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace_base: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve the scene presentation variant for one cyclic-order sample."""

    explicit = params.get("scene_variant")
    if explicit is not None and str(explicit) in SOURCE_SCENE_VARIANT_MAP:
        selected = str(SOURCE_SCENE_VARIANT_MAP[str(explicit)])
        return selected, {
            value: (1.0 if str(value) == selected else 0.0)
            for value in SCENE_VARIANTS
        }
    return _axis_variant(
        params=params,
        generation_defaults=generation_defaults,
        instance_seed=int(instance_seed),
        supported=SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_key="balanced_scene_variant_sampling",
        namespace=f"{str(namespace_base)}.scene_variant",
    )


def resolve_token_render_style(
    *,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace_base: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve how ordered loop tokens are drawn."""

    explicit_style = params.get("token_render_style")
    if explicit_style is None:
        source_mode = params.get("bead_token_mode")
        if source_mode is not None:
            selected = str(SOURCE_TOKEN_MODE_MAP.get(str(source_mode), str(source_mode)))
            if selected not in set(TOKEN_RENDER_STYLES):
                raise ValueError(f"unsupported bead_token_mode: {source_mode}")
            return selected, {
                value: (1.0 if str(value) == selected else 0.0)
                for value in TOKEN_RENDER_STYLES
            }
    return _axis_variant(
        params=params,
        generation_defaults=generation_defaults,
        instance_seed=int(instance_seed),
        supported=TOKEN_RENDER_STYLES,
        explicit_key="token_render_style",
        weights_key="token_render_style_weights",
        balance_key="balanced_token_render_style_sampling",
        namespace=f"{str(namespace_base)}.token_render_style",
    )


def resolve_loop_path_style(
    *,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace_base: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve the closed-loop path style used by reference and options."""

    return _axis_variant(
        params=params,
        generation_defaults=generation_defaults,
        instance_seed=int(instance_seed),
        supported=LOOP_PATH_STYLES,
        explicit_key="loop_path_style",
        weights_key="loop_path_style_weights",
        balance_key="balanced_loop_path_style_sampling",
        namespace=f"{str(namespace_base)}.loop_path_style",
    )


def resolve_answer_option_label(
    *,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    instance_seed: int,
    labels: Sequence[str],
    namespace_base: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve the answer label location without depending on seed modulo."""

    return _axis_variant(
        params=params,
        generation_defaults=generation_defaults,
        instance_seed=int(instance_seed),
        supported=tuple(str(label) for label in labels),
        explicit_key="answer_option_label",
        weights_key="answer_option_label_weights",
        balance_key="balanced_answer_option_label_sampling",
        namespace=f"{str(namespace_base)}.answer_option_label",
    )


def resolve_render_params(
    params: Mapping[str, Any],
    *,
    rendering_defaults: Mapping[str, Any],
    instance_seed: int | None = None,
) -> CyclicOrderRenderParams:
    """Resolve rendering parameters for cyclic-order puzzle scenes."""

    def _int(key: str, fallback: int) -> int:
        return resolve_render_int(
            params,
            rendering_defaults,
            str(key),
            int(fallback),
            instance_seed=instance_seed,
            namespace="cyclic_order_render",
        )

    def _rgb(key: str, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
        return resolve_render_rgb(
            params,
            rendering_defaults,
            str(key),
            fallback,
            instance_seed=instance_seed,
            namespace="cyclic_order_render",
        )

    return CyclicOrderRenderParams(
        canvas_width=int(_int("canvas_width", 1200)),
        canvas_height=int(_int("canvas_height", 920)),
        scene_margin_left_px=int(_int("scene_margin_left_px", 64)),
        scene_margin_right_px=int(_int("scene_margin_right_px", 64)),
        scene_margin_top_px=int(_int("scene_margin_top_px", 56)),
        scene_margin_bottom_px=int(_int("scene_margin_bottom_px", 56)),
        reference_panel_height_px=int(_int("reference_panel_height_px", 238)),
        reference_panel_padding_px=int(_int("reference_panel_padding_px", 24)),
        reference_loop_width_px=int(_int("reference_loop_width_px", 356)),
        reference_loop_height_px=int(_int("reference_loop_height_px", 170)),
        reference_label_font_size_px=int(_int("reference_label_font_size_px", 28)),
        reference_to_options_gap_px=int(_int("reference_to_options_gap_px", 54)),
        option_image_width_px=int(_int("option_image_width_px", 230)),
        option_image_height_px=int(_int("option_image_height_px", 180)),
        option_gap_px=int(_int("option_gap_px", 28)),
        option_row_gap_px=int(_int("option_row_gap_px", 38)),
        option_label_gap_px=int(_int("option_label_gap_px", 16)),
        option_label_font_size_px=int(_int("option_label_font_size_px", 34)),
        panel_corner_radius_px=int(_int("panel_corner_radius_px", 28)),
        border_width_px=int(_int("border_width_px", 3)),
        loop_stroke_width_px=int(_int("loop_stroke_width_px", 5)),
        bead_size_px=int(_int("bead_size_px", 45)),
        shape_bead_inset_px=int(_int("shape_bead_inset_px", 2)),
        panel_fill_rgb=_rgb("panel_fill_rgb", (248, 249, 252)),
        instruction_fill_rgb=_rgb("instruction_fill_rgb", (240, 244, 250)),
        border_color_rgb=_rgb("border_color_rgb", (86, 94, 108)),
        loop_color_rgb=_rgb("loop_color_rgb", (78, 87, 102)),
        text_color_rgb=_rgb("text_color_rgb", (28, 32, 38)),
        text_stroke_rgb=_rgb("text_stroke_rgb", (255, 255, 255)),
        shape_fill_rgb=_rgb("shape_fill_rgb", (73, 110, 186)),
    )


def _int_bound(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    key: str,
    fallback: int,
) -> int:
    """Resolve one integer generation bound from params or scene defaults."""

    return int(params.get(str(key), group_default(defaults, str(key), int(fallback))))


def _token_count_bounds(
    token_render_style: str,
    *,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    defaults: CyclicOrderDefaults,
) -> tuple[int, int]:
    """Return token-count bounds, clamping shape-only modes to their fallback max."""

    min_count = _int_bound(params, generation_defaults, "bead_count_min", int(defaults.bead_count_min))
    max_count = _int_bound(params, generation_defaults, "bead_count_max", int(defaults.bead_count_max))
    if str(token_render_style) in {"shape_tokens", "outline_shape_tokens"}:
        max_count = min(int(max_count), int(defaults.shape_bead_count_max))
    min_count = min(int(min_count), int(max_count))
    return int(min_count), int(max_count)


def _swap_positions(tokens: Sequence[str], first_index: int, second_index: int) -> Tuple[str, ...]:
    """Return a copy of tokens with two zero-based positions exchanged."""

    swapped = [str(token) for token in tokens]
    swapped[int(first_index)], swapped[int(second_index)] = (
        swapped[int(second_index)],
        swapped[int(first_index)],
    )
    return tuple(swapped)


def _sample_distinct_color_specs(
    *,
    token_count: int,
    min_distance: float,
    distance_space: str,
    rng,
) -> list[tuple[str, tuple[int, int, int]]]:
    """Sample colors whose pairwise distance stays legible across token styles."""

    shuffled = list(TOKEN_COLOR_SPECS)
    rng.shuffle(shuffled)
    for candidate_specs in combinations(shuffled, int(token_count)):
        if all(
            float(color_distance(color_a, color_b, distance_space=str(distance_space))) >= float(min_distance)
            for (_, color_a), (_, color_b) in combinations(candidate_specs, 2)
        ):
            selected = list(candidate_specs)
            rng.shuffle(selected)
            return selected
    raise RuntimeError("no cyclic-order color subset satisfied the required separation")


def _token_catalog_for_style(
    token_render_style: str,
    *,
    token_count: int,
    min_color_distance: float,
    color_distance_space: str,
    rng,
) -> Dict[str, Dict[str, Any]]:
    """Build render specs for the ordered token identities in one sample."""

    selected_style = str(token_render_style)
    if selected_style == "colored_beads":
        colors = _sample_distinct_color_specs(
            token_count=int(token_count),
            min_distance=float(min_color_distance),
            distance_space=str(color_distance_space),
            rng=rng,
        )
        return {
            str(color_name): {
                "token_label": str(color_name),
                "render_mode": "color",
                "object_type": "circle",
                "fill_rgb": [int(value) for value in color_rgb],
            }
            for color_name, color_rgb in colors
        }

    if selected_style in {"shape_tokens", "outline_shape_tokens"}:
        shapes = rng.sample(list(PUZZLE_OBJECT_TYPES), int(token_count))
        return {
            str(shape): {
                "token_label": str(shape),
                "render_mode": "outline_shape" if selected_style == "outline_shape_tokens" else "shape",
                "object_type": str(shape),
                "fill_rgb": None,
            }
            for shape in shapes
        }

    colors = _sample_distinct_color_specs(
        token_count=int(token_count),
        min_distance=float(min_color_distance),
        distance_space=str(color_distance_space),
        rng=rng,
    )
    shapes = rng.sample(list(PUZZLE_OBJECT_TYPES), int(token_count))
    render_mode = "symbol_badge" if selected_style == "symbol_badges" else "mixed"
    return {
        f"{shape}:{color_name}": {
            "token_label": f"{shape}:{color_name}",
            "render_mode": str(render_mode),
            "object_type": str(shape),
            "fill_rgb": [int(value) for value in color_rgb],
        }
        for shape, (color_name, color_rgb) in zip(shapes, colors)
    }


def build_cyclic_order_dataset(
    *,
    token_render_style: str,
    loop_path_style: str,
    params: Mapping[str, Any],
    instance_seed: int,
    generation_defaults: Mapping[str, Any],
    defaults: CyclicOrderDefaults = DEFAULTS,
    namespace_base: str = "cyclic_order",
) -> Dict[str, Any]:
    """Build one reference loop with exactly one matching option loop."""

    selected_token_style = str(token_render_style)
    if selected_token_style not in set(TOKEN_RENDER_STYLES):
        raise ValueError(f"unsupported cyclic-order token_render_style: {token_render_style}")
    selected_loop_path_style = str(loop_path_style)
    if selected_loop_path_style not in set(LOOP_PATH_STYLES):
        raise ValueError(f"unsupported cyclic-order loop_path_style: {loop_path_style}")

    option_count_min = _int_bound(
        params,
        generation_defaults,
        "option_count_min",
        int(defaults.option_count_min),
    )
    option_count_max = _int_bound(
        params,
        generation_defaults,
        "option_count_max",
        int(defaults.option_count_max),
    )
    if int(option_count_min) > int(option_count_max):
        raise ValueError("option_count_min must be <= option_count_max")
    if int(option_count_max) < 2:
        raise ValueError("cyclic-order options need at least one match and one distractor")

    rng = spawn_rng(int(instance_seed), f"{str(namespace_base)}.dataset")
    option_count = int(rng.randint(int(option_count_min), int(option_count_max)))
    labels = [chr(ord("A") + index) for index in range(int(option_count))]
    answer_label, answer_label_probabilities = resolve_answer_option_label(
        params=params,
        generation_defaults=generation_defaults,
        instance_seed=int(instance_seed),
        labels=labels,
        namespace_base=str(namespace_base),
    )
    answer_position = labels.index(str(answer_label))

    bead_count_min, bead_count_max = _token_count_bounds(
        selected_token_style,
        params=params,
        generation_defaults=generation_defaults,
        defaults=defaults,
    )
    if int(bead_count_min) > int(bead_count_max):
        raise ValueError("bead_count_min must be <= bead_count_max")
    bead_count = int(rng.randint(int(bead_count_min), int(bead_count_max)))

    min_color_distance = float(
        params.get(
            "min_color_distance",
            group_default(generation_defaults, "min_color_distance", float(defaults.min_color_distance)),
        )
    )
    color_distance_space = str(
        params.get(
            "color_distance_space",
            group_default(generation_defaults, "color_distance_space", str(defaults.color_distance_space)),
        )
    ).strip().lower()

    token_catalog = _token_catalog_for_style(
        selected_token_style,
        token_count=int(bead_count),
        min_color_distance=float(min_color_distance),
        color_distance_space=str(color_distance_space),
        rng=rng,
    )
    reference_tokens = tuple(str(token) for token in rng.sample(list(token_catalog.keys()), int(bead_count)))
    valid_offset = int(rng.choice(list(range(int(bead_count)))))
    valid_sequence = rotate_token_sequence(reference_tokens, int(valid_offset))

    invalid_needed = int(option_count - 1)
    invalid_candidates = invalid_candidate_sequences(reference_tokens)
    if len(invalid_candidates) < int(invalid_needed):
        raise RuntimeError("insufficient cyclic-order distractors")
    rng.shuffle(invalid_candidates)

    option_records: list[dict[str, Any]] = [
        {
            "is_valid": False,
            "rotation_offset": None,
            "token_sequence": list(sequence),
        }
        for sequence in invalid_candidates[: int(invalid_needed)]
    ]
    option_records.insert(
        int(answer_position),
        {
            "is_valid": True,
            "rotation_offset": int(valid_offset),
            "token_sequence": list(valid_sequence),
        },
    )

    option_specs: list[dict[str, Any]] = []
    valid_option_choice_ids: list[str] = []
    valid_option_labels: list[str] = []
    for option_index, (option_label, option_record) in enumerate(zip(labels, option_records), start=1):
        option_choice_id = f"option_{int(option_index)}"
        loop_shape_variant = str(rng.choice(list(LOOP_SHAPE_VARIANTS)))
        start_angle_deg = int(rng.choice(list(LOOP_START_ANGLES_DEG)))
        bead_specs = [
            dict(token_catalog[str(token_label)])
            for token_label in option_record["token_sequence"]
        ]
        spec = {
            "option_index": int(option_index - 1),
            "option_label": str(option_label),
            "option_choice_id": str(option_choice_id),
            "is_valid": bool(option_record["is_valid"]),
            "rotation_offset": option_record["rotation_offset"],
            "token_sequence": [str(token) for token in option_record["token_sequence"]],
            "loop_shape_variant": str(loop_shape_variant),
            "loop_path_style": str(selected_loop_path_style),
            "start_angle_deg": int(start_angle_deg),
            "bead_specs": bead_specs,
        }
        option_specs.append(spec)
        if bool(spec["is_valid"]):
            valid_option_choice_ids.append(str(option_choice_id))
            valid_option_labels.append(str(option_label))

    reference_bead_specs = [dict(token_catalog[str(token_label)]) for token_label in reference_tokens]
    return {
        "reference_token_sequence": [str(token) for token in reference_tokens],
        "reference_bead_specs": reference_bead_specs,
        "reference_loop_shape_variant": str(rng.choice(list(LOOP_SHAPE_VARIANTS))),
        "reference_loop_path_style": str(selected_loop_path_style),
        "reference_start_angle_deg": int(-90),
        "option_specs": option_specs,
        "option_count": int(option_count),
        "option_count_range": [int(option_count_min), int(option_count_max)],
        "valid_option_count": 1,
        "valid_option_choice_ids": [str(value) for value in valid_option_choice_ids],
        "valid_option_labels": [str(value) for value in valid_option_labels],
        "answer_option_choice_id": str(valid_option_choice_ids[0]),
        "answer_option_label": str(valid_option_labels[0]),
        "answer_option_label_probabilities": dict(answer_label_probabilities),
        "bead_count": int(bead_count),
        "bead_count_range": [int(bead_count_min), int(bead_count_max)],
        "token_render_style": str(selected_token_style),
        "bead_token_mode": str(TOKEN_STYLE_SOURCE_MODE[str(selected_token_style)]),
        "loop_path_style": str(selected_loop_path_style),
        "equivalence_rule": "same_cyclic_order_up_to_rotation_no_reflection",
        "color_distance_space": str(color_distance_space),
        "min_color_distance": float(min_color_distance),
        "question_format": "equivalent_loop_label",
        "view_family": "loop_option_label",
        "solver_trace": {
            "token_render_style": str(selected_token_style),
            "bead_token_mode": str(TOKEN_STYLE_SOURCE_MODE[str(selected_token_style)]),
            "loop_path_style": str(selected_loop_path_style),
            "reference_token_sequence": [str(token) for token in reference_tokens],
            "valid_option_labels": [str(value) for value in valid_option_labels],
            "valid_option_choice_ids": [str(value) for value in valid_option_choice_ids],
            "answer_option_choice_id": str(valid_option_choice_ids[0]),
            "answer_option_label": str(valid_option_labels[0]),
            "equivalence_rule": "same_cyclic_order_up_to_rotation_no_reflection",
            "color_distance_space": str(color_distance_space),
            "min_color_distance": float(min_color_distance),
            "rotation_allowed": True,
            "reflection_allowed": False,
        },
    }


def build_swap_repair_dataset(
    *,
    token_render_style: str,
    loop_path_style: str,
    params: Mapping[str, Any],
    instance_seed: int,
    generation_defaults: Mapping[str, Any],
    defaults: CyclicOrderDefaults = DEFAULTS,
    namespace_base: str = "cyclic_order",
) -> Dict[str, Any]:
    """Build a broken loop where exactly one visible swap repairs the order."""

    selected_token_style = str(token_render_style)
    if selected_token_style not in set(TOKEN_RENDER_STYLES):
        raise ValueError(f"unsupported cyclic-order token_render_style: {token_render_style}")
    selected_loop_path_style = str(loop_path_style)
    if selected_loop_path_style not in set(LOOP_PATH_STYLES):
        raise ValueError(f"unsupported cyclic-order loop_path_style: {loop_path_style}")

    option_count_min = _int_bound(
        params,
        generation_defaults,
        "option_count_min",
        int(defaults.option_count_min),
    )
    option_count_max = _int_bound(
        params,
        generation_defaults,
        "option_count_max",
        int(defaults.option_count_max),
    )
    if int(option_count_min) != 4 or int(option_count_max) != 4:
        raise ValueError("cyclic-order swap repair currently uses exactly four options")

    rng = spawn_rng(int(instance_seed), f"{str(namespace_base)}.swap_repair_dataset")
    option_count = 4
    labels = [chr(ord("A") + index) for index in range(int(option_count))]
    answer_label, answer_label_probabilities = resolve_answer_option_label(
        params=params,
        generation_defaults=generation_defaults,
        instance_seed=int(instance_seed),
        labels=labels,
        namespace_base=f"{str(namespace_base)}.swap_repair",
    )
    answer_position = labels.index(str(answer_label))

    bead_count_min, bead_count_max = _token_count_bounds(
        selected_token_style,
        params=params,
        generation_defaults=generation_defaults,
        defaults=defaults,
    )
    if int(bead_count_min) > int(bead_count_max):
        raise ValueError("bead_count_min must be <= bead_count_max")
    bead_count = int(rng.randint(int(bead_count_min), int(bead_count_max)))
    all_pairs = tuple(combinations(range(int(bead_count)), 2))
    if len(all_pairs) < int(option_count):
        raise ValueError("cyclic-order swap repair needs at least four position pairs")

    min_color_distance = float(
        params.get(
            "min_color_distance",
            group_default(generation_defaults, "min_color_distance", float(defaults.min_color_distance)),
        )
    )
    color_distance_space = str(
        params.get(
            "color_distance_space",
            group_default(generation_defaults, "color_distance_space", str(defaults.color_distance_space)),
        )
    ).strip().lower()

    token_catalog = _token_catalog_for_style(
        selected_token_style,
        token_count=int(bead_count),
        min_color_distance=float(min_color_distance),
        color_distance_space=str(color_distance_space),
        rng=rng,
    )
    reference_tokens = tuple(str(token) for token in rng.sample(list(token_catalog.keys()), int(bead_count)))
    solved_offset = int(rng.choice(list(range(int(bead_count)))))
    solved_sequence = rotate_token_sequence(reference_tokens, int(solved_offset))
    correct_pair = tuple(int(index) for index in uniform_choice(rng, all_pairs))
    broken_sequence = _swap_positions(solved_sequence, int(correct_pair[0]), int(correct_pair[1]))

    distractor_pairs: list[Tuple[int, int]] = []
    shuffled_pairs = [tuple(int(index) for index in pair) for pair in all_pairs if tuple(pair) != correct_pair]
    rng.shuffle(shuffled_pairs)
    for pair in shuffled_pairs:
        repaired = _swap_positions(broken_sequence, int(pair[0]), int(pair[1]))
        if token_sequences_are_rotation_equivalent(reference_tokens, repaired):
            continue
        distractor_pairs.append(tuple(pair))
        if len(distractor_pairs) == int(option_count - 1):
            break
    if len(distractor_pairs) != int(option_count - 1):
        raise ValueError("insufficient non-repairing swap distractors")

    option_records = [
        {
            "is_valid": False,
            "swap_pair": tuple(pair),
            "repaired_token_sequence": list(_swap_positions(broken_sequence, int(pair[0]), int(pair[1]))),
        }
        for pair in distractor_pairs
    ]
    option_records.insert(
        int(answer_position),
        {
            "is_valid": True,
            "swap_pair": tuple(correct_pair),
            "repaired_token_sequence": list(solved_sequence),
        },
    )

    option_specs: list[dict[str, Any]] = []
    valid_option_choice_ids: list[str] = []
    valid_option_labels: list[str] = []
    for option_index, (option_label, option_record) in enumerate(zip(labels, option_records), start=1):
        first_index, second_index = tuple(option_record["swap_pair"])
        option_choice_id = f"option_{int(option_index)}"
        spec = {
            "option_index": int(option_index - 1),
            "option_label": str(option_label),
            "option_choice_id": str(option_choice_id),
            "is_valid": bool(option_record["is_valid"]),
            "first_position": int(first_index + 1),
            "second_position": int(second_index + 1),
            "swap_pair_zero_based": [int(first_index), int(second_index)],
            "repaired_token_sequence": [str(value) for value in option_record["repaired_token_sequence"]],
        }
        option_specs.append(spec)
        if bool(spec["is_valid"]):
            valid_option_choice_ids.append(str(option_choice_id))
            valid_option_labels.append(str(option_label))

    reference_bead_specs = [dict(token_catalog[str(token_label)]) for token_label in reference_tokens]
    broken_bead_specs = [dict(token_catalog[str(token_label)]) for token_label in broken_sequence]
    return {
        "reference_token_sequence": [str(token) for token in reference_tokens],
        "reference_bead_specs": reference_bead_specs,
        "reference_loop_shape_variant": str(rng.choice(list(LOOP_SHAPE_VARIANTS))),
        "reference_loop_path_style": str(selected_loop_path_style),
        "reference_start_angle_deg": int(-90),
        "solved_token_sequence": [str(token) for token in solved_sequence],
        "solved_rotation_offset": int(solved_offset),
        "broken_token_sequence": [str(token) for token in broken_sequence],
        "broken_bead_specs": broken_bead_specs,
        "broken_loop_shape_variant": str(rng.choice(list(LOOP_SHAPE_VARIANTS))),
        "broken_loop_path_style": str(selected_loop_path_style),
        "broken_start_angle_deg": int(-90),
        "option_specs": option_specs,
        "option_count": int(option_count),
        "option_count_range": [int(option_count_min), int(option_count_max)],
        "valid_option_count": 1,
        "valid_option_choice_ids": [str(value) for value in valid_option_choice_ids],
        "valid_option_labels": [str(value) for value in valid_option_labels],
        "answer_option_choice_id": str(valid_option_choice_ids[0]),
        "answer_option_label": str(valid_option_labels[0]),
        "answer_option_label_probabilities": dict(answer_label_probabilities),
        "bead_count": int(bead_count),
        "bead_count_range": [int(bead_count_min), int(bead_count_max)],
        "token_render_style": str(selected_token_style),
        "bead_token_mode": str(TOKEN_STYLE_SOURCE_MODE[str(selected_token_style)]),
        "loop_path_style": str(selected_loop_path_style),
        "equivalence_rule": "repair_broken_loop_by_one_position_swap_then_rotation_allowed",
        "color_distance_space": str(color_distance_space),
        "min_color_distance": float(min_color_distance),
        "question_format": "swap_repair_option",
        "view_family": "loop_swap_repair",
        "solver_trace": {
            "token_render_style": str(selected_token_style),
            "bead_token_mode": str(TOKEN_STYLE_SOURCE_MODE[str(selected_token_style)]),
            "loop_path_style": str(selected_loop_path_style),
            "reference_token_sequence": [str(token) for token in reference_tokens],
            "solved_token_sequence": [str(token) for token in solved_sequence],
            "broken_token_sequence": [str(token) for token in broken_sequence],
            "correct_swap_positions": [int(correct_pair[0] + 1), int(correct_pair[1] + 1)],
            "valid_option_labels": [str(value) for value in valid_option_labels],
            "valid_option_choice_ids": [str(value) for value in valid_option_choice_ids],
            "answer_option_choice_id": str(valid_option_choice_ids[0]),
            "answer_option_label": str(valid_option_labels[0]),
            "equivalence_rule": "repair_broken_loop_by_one_position_swap_then_rotation_allowed",
            "color_distance_space": str(color_distance_space),
            "min_color_distance": float(min_color_distance),
            "rotation_allowed": True,
            "reflection_allowed": False,
        },
    }


def build_insertion_position_dataset(
    *,
    token_render_style: str,
    loop_path_style: str,
    params: Mapping[str, Any],
    instance_seed: int,
    generation_defaults: Mapping[str, Any],
    defaults: CyclicOrderDefaults = DEFAULTS,
    namespace_base: str = "cyclic_order",
) -> Dict[str, Any]:
    """Build a partial loop where one loose token belongs in exactly one gap."""

    selected_token_style = str(token_render_style)
    if selected_token_style not in set(TOKEN_RENDER_STYLES):
        raise ValueError(f"unsupported cyclic-order token_render_style: {token_render_style}")
    selected_loop_path_style = str(loop_path_style)
    if selected_loop_path_style not in set(LOOP_PATH_STYLES):
        raise ValueError(f"unsupported cyclic-order loop_path_style: {loop_path_style}")

    rng = spawn_rng(int(instance_seed), f"{str(namespace_base)}.insertion_position_dataset")
    option_count = 4
    bead_count = 5
    labels = [chr(ord("A") + index) for index in range(int(option_count))]
    answer_label, answer_label_probabilities = resolve_answer_option_label(
        params=params,
        generation_defaults=generation_defaults,
        instance_seed=int(instance_seed),
        labels=labels,
        namespace_base=f"{str(namespace_base)}.insertion_position",
    )
    answer_position = labels.index(str(answer_label))

    min_color_distance = float(
        params.get(
            "min_color_distance",
            group_default(generation_defaults, "min_color_distance", float(defaults.min_color_distance)),
        )
    )
    color_distance_space = str(
        params.get(
            "color_distance_space",
            group_default(generation_defaults, "color_distance_space", str(defaults.color_distance_space)),
        )
    ).strip().lower()

    token_catalog = _token_catalog_for_style(
        selected_token_style,
        token_count=int(bead_count),
        min_color_distance=float(min_color_distance),
        color_distance_space=str(color_distance_space),
        rng=rng,
    )
    reference_tokens = tuple(str(token) for token in rng.sample(list(token_catalog.keys()), int(bead_count)))
    solved_offset = int(rng.choice(list(range(int(bead_count)))))
    solved_sequence = rotate_token_sequence(reference_tokens, int(solved_offset))

    # With five solved tokens and four displayed gaps, removing token g+1 makes
    # gap g the unique insertion point after the preceding displayed token.
    missing_index = int(answer_position + 1)
    missing_token = str(solved_sequence[int(missing_index)])
    partial_sequence = tuple(
        str(token)
        for token_index, token in enumerate(solved_sequence)
        if int(token_index) != int(missing_index)
    )

    option_specs: list[dict[str, Any]] = []
    valid_option_choice_ids: list[str] = []
    valid_option_labels: list[str] = []
    for option_index, option_label in enumerate(labels, start=1):
        gap_index = int(option_index - 1)
        insertion_index = int(gap_index + 1)
        inserted_sequence = (
            tuple(partial_sequence[:insertion_index])
            + (str(missing_token),)
            + tuple(partial_sequence[insertion_index:])
        )
        is_valid = token_sequences_are_rotation_equivalent(reference_tokens, inserted_sequence)
        option_choice_id = f"option_{int(option_index)}"
        spec = {
            "option_index": int(option_index - 1),
            "option_label": str(option_label),
            "option_choice_id": str(option_choice_id),
            "is_valid": bool(is_valid),
            "gap_label": str(option_label),
            "gap_number": int(gap_index + 1),
            "insert_after_position": int(gap_index + 1),
            "inserted_token_sequence": [str(value) for value in inserted_sequence],
        }
        option_specs.append(spec)
        if bool(spec["is_valid"]):
            valid_option_choice_ids.append(str(option_choice_id))
            valid_option_labels.append(str(option_label))

    if valid_option_labels != [str(answer_label)]:
        raise ValueError("cyclic-order insertion task must have exactly one valid insertion option")

    reference_bead_specs = [dict(token_catalog[str(token_label)]) for token_label in reference_tokens]
    partial_bead_specs = [dict(token_catalog[str(token_label)]) for token_label in partial_sequence]
    missing_bead_spec = dict(token_catalog[str(missing_token)])
    return {
        "reference_token_sequence": [str(token) for token in reference_tokens],
        "reference_bead_specs": reference_bead_specs,
        "reference_loop_shape_variant": str(rng.choice(list(LOOP_SHAPE_VARIANTS))),
        "reference_loop_path_style": str(selected_loop_path_style),
        "reference_start_angle_deg": int(-90),
        "solved_token_sequence": [str(token) for token in solved_sequence],
        "solved_rotation_offset": int(solved_offset),
        "partial_token_sequence": [str(token) for token in partial_sequence],
        "partial_bead_specs": partial_bead_specs,
        "partial_loop_shape_variant": str(rng.choice(list(LOOP_SHAPE_VARIANTS))),
        "partial_loop_path_style": str(selected_loop_path_style),
        "partial_start_angle_deg": int(-90),
        "partial_gap_labels": [str(label) for label in labels],
        "missing_token": str(missing_token),
        "missing_token_spec": missing_bead_spec,
        "missing_token_position_in_solved": int(missing_index + 1),
        "option_specs": option_specs,
        "option_count": int(option_count),
        "option_count_range": [int(option_count), int(option_count)],
        "valid_option_count": 1,
        "valid_option_choice_ids": [str(value) for value in valid_option_choice_ids],
        "valid_option_labels": [str(value) for value in valid_option_labels],
        "answer_option_choice_id": str(valid_option_choice_ids[0]),
        "answer_option_label": str(valid_option_labels[0]),
        "answer_option_label_probabilities": dict(answer_label_probabilities),
        "bead_count": int(bead_count),
        "bead_count_range": [int(bead_count), int(bead_count)],
        "token_render_style": str(selected_token_style),
        "bead_token_mode": str(TOKEN_STYLE_SOURCE_MODE[str(selected_token_style)]),
        "loop_path_style": str(selected_loop_path_style),
        "equivalence_rule": "insert_loose_token_then_rotation_allowed",
        "color_distance_space": str(color_distance_space),
        "min_color_distance": float(min_color_distance),
        "question_format": "insertion_position_option",
        "view_family": "loop_insertion_position",
        "solver_trace": {
            "token_render_style": str(selected_token_style),
            "bead_token_mode": str(TOKEN_STYLE_SOURCE_MODE[str(selected_token_style)]),
            "loop_path_style": str(selected_loop_path_style),
            "reference_token_sequence": [str(token) for token in reference_tokens],
            "solved_token_sequence": [str(token) for token in solved_sequence],
            "partial_token_sequence": [str(token) for token in partial_sequence],
            "missing_token": str(missing_token),
            "correct_gap_number": int(answer_position + 1),
            "valid_option_labels": [str(value) for value in valid_option_labels],
            "valid_option_choice_ids": [str(value) for value in valid_option_choice_ids],
            "answer_option_choice_id": str(valid_option_choice_ids[0]),
            "answer_option_label": str(valid_option_labels[0]),
            "equivalence_rule": "insert_loose_token_then_rotation_allowed",
            "color_distance_space": str(color_distance_space),
            "min_color_distance": float(min_color_distance),
            "rotation_allowed": True,
            "reflection_allowed": False,
        },
    }


__all__ = [
    "build_cyclic_order_dataset",
    "build_insertion_position_dataset",
    "build_swap_repair_dataset",
    "resolve_loop_path_style",
    "resolve_render_params",
    "resolve_scene_variant",
    "resolve_token_render_style",
]
