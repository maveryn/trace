"""Query and dataset construction for infographic metric-card page tasks."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default, resolve_required_int_bounds
from trace_tasks.tasks.shared.deterministic_sampling import resolve_selection_index
from trace_tasks.tasks.shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant
from trace_tasks.tasks.pages.shared.page_text_resources import page_text_resource_metadata, sample_page_label_batch
from trace_tasks.tasks.pages.shared.infographic_metric_common import (
    METRIC_RANKED_ITEM_VARIANTS,
    SUPPORTED_QUERY_IDS,
    TASK_ID,
    _EXTREMA_OPERATIONS,
    _EXTREMUM_KINDS,
    _GEN_DEFAULTS,
    _ICON_KINDS,
    _PALETTE,
    _RANK_DIRECTIONS,
    _RANK_ORDINALS,
    _RANK_POSITION_SUPPORT,
    _MetricCard,
    _adjust_section_total,
    _adjust_value_to_break_tie,
    _labels_by_section,
    _partition_cards,
    _section_totals,
)

def _probability_map_for_selection(supported: Sequence[str], selected: str) -> Dict[str, float]:
    return {str(key): (1.0 if str(key) == str(selected) else 0.0) for key in supported}


def _resolve_query_id(params: Mapping[str, Any], *, instance_seed: int) -> Tuple[str, Dict[str, float]]:
    raw_supported = params.get("_supported_query_ids", SUPPORTED_QUERY_IDS)
    if isinstance(raw_supported, str):
        supported_variants = tuple(value.strip() for value in raw_supported.split(",") if value.strip())
    else:
        supported_variants = tuple(str(value) for value in raw_supported)
    if not supported_variants:
        supported_variants = tuple(SUPPORTED_QUERY_IDS)
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.query_id")
    query_id, probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        supported_variants=supported_variants,
        explicit_key="query_id",
        weights_key="query_id_weights",
    )
    balanced = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        selected_variant=str(query_id),
        variant_probabilities=probabilities,
        supported_variants=supported_variants,
        balance_flag_key="balanced_query_id_sampling",
        explicit_key="query_id",
        weights_key="query_id_weights",
        sampling_namespace=f"{TASK_ID}.query_id",
    )
    if balanced != query_id and params.get("query_id") is not None:
        return str(balanced), _probability_map_for_selection(supported_variants, str(balanced))
    return str(balanced), dict(probabilities)


def _resolve_int_range(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    min_key: str,
    max_key: str,
    explicit_key: str,
    fallback_min: int,
    fallback_max: int,
    instance_seed: int,
    namespace: str,
    balanced_flag_key: str = "balanced_count_sampling",
) -> Tuple[int, Tuple[int, int], Dict[str, float]]:
    lower, upper = resolve_required_int_bounds(
        params,
        gen_defaults,
        min_key=str(min_key),
        max_key=str(max_key),
        fallback_min=int(fallback_min),
        fallback_max=int(fallback_max),
        context=f"generation defaults for {TASK_ID}",
    )
    support = [int(value) for value in range(int(lower), int(upper) + 1)]
    explicit = params.get(str(explicit_key))
    if explicit is not None:
        selected = int(explicit)
        if int(selected) not in set(support):
            raise ValueError(f"{explicit_key} must be in {lower}..{upper}")
        return int(selected), (int(lower), int(upper)), {str(int(selected)): 1.0}
    balanced = bool(params.get(str(balanced_flag_key), group_default(gen_defaults, str(balanced_flag_key), True)))
    if balanced:
        index = resolve_selection_index(params=params, instance_seed=int(instance_seed), namespace=str(namespace))
        selected = int(support[int(index) % len(support)])
    else:
        rng = spawn_rng(int(instance_seed), str(namespace))
        selected = int(support[int(rng.randrange(len(support)))])
    probability = 1.0 / float(len(support))
    return int(selected), (int(lower), int(upper)), {str(value): float(probability) for value in support}


def _resolve_section_count(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    query_id: str,
    card_count: int,
    instance_seed: int,
) -> Tuple[int, Tuple[int, int], Dict[str, float]]:
    lower, upper = resolve_required_int_bounds(
        params,
        gen_defaults,
        min_key="section_count_min",
        max_key="section_count_max",
        fallback_min=2,
        fallback_max=4,
        context=f"generation defaults for {TASK_ID}",
    )
    support = [int(value) for value in range(int(lower), int(upper) + 1)]
    if str(query_id) == "section_total_except_named":
        _, exclusion_max = resolve_required_int_bounds(
            params,
            gen_defaults,
            min_key="excluded_metric_count_min",
            max_key="excluded_metric_count_max",
            fallback_min=2,
            fallback_max=2,
            context=f"generation defaults for {TASK_ID}",
        )
        included_count_min = int(
            params.get(
                "section_except_included_count_min",
                group_default(gen_defaults, "section_except_included_count_min", 3),
            )
        )
        required_section_size = int(exclusion_max) + int(included_count_min)
        support = [
            int(section_count)
            for section_count in support
            if max(_partition_cards(int(card_count), int(section_count))) >= int(required_section_size)
        ]
    if not support:
        raise ValueError(f"no feasible section_count values for {query_id} with card_count={card_count}")
    explicit = params.get("section_count")
    if explicit is not None:
        selected = int(explicit)
        if int(selected) not in set(support):
            raise ValueError(f"section_count={selected} is infeasible for {query_id} with card_count={card_count}")
        return int(selected), (min(support), max(support)), {str(int(selected)): 1.0}
    balanced = bool(params.get("balanced_count_sampling", group_default(gen_defaults, "balanced_count_sampling", True)))
    if balanced:
        index = resolve_selection_index(
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}.section_count.{query_id}.{card_count}",
        )
        selected = int(support[int(index) % len(support)])
    else:
        rng = spawn_rng(int(instance_seed), f"{TASK_ID}.section_count.{query_id}")
        selected = int(support[int(rng.randrange(len(support)))])
    probability = 1.0 / float(len(support))
    return int(selected), (min(support), max(support)), {str(value): float(probability) for value in support}


def _caption_tag_for_index(index: int) -> str:
    """Return a compact non-numeric card tag for non-reference-code tasks."""

    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    first = alphabet[(int(index) // len(alphabet)) % len(alphabet)]
    second = alphabet[int(index) % len(alphabet)]
    return f"Tag {first}{second}"


def _build_cards(
    *,
    query_id: str,
    card_count: int,
    section_titles: Sequence[str],
    section_card_counts: Sequence[int],
    values_by_label: Mapping[str, int],
    percent_mode: bool,
    instance_seed: int,
) -> List[_MetricCard]:
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.cards")
    labels = list(values_by_label.keys())
    cards: List[_MetricCard] = []
    label_index = 0
    repeated_icon_offset = int(rng.randrange(len(_ICON_KINDS))) if str(query_id) == "section_icon_extremum_label" else 0
    for section_index, section_title in enumerate(section_titles):
        for local_index in range(int(section_card_counts[section_index])):
            label = labels[label_index]
            value = int(values_by_label[str(label)])
            unit = "%" if bool(percent_mode) else ""
            display = f"{value}%" if bool(percent_mode) else str(value)
            color = _PALETTE[(label_index + int(rng.randrange(len(_PALETTE)))) % len(_PALETTE)]
            if str(query_id) == "section_icon_extremum_label":
                icon_kind = _ICON_KINDS[(int(local_index) + int(repeated_icon_offset)) % len(_ICON_KINDS)]
            else:
                icon_kind = _ICON_KINDS[(label_index + int(rng.randrange(len(_ICON_KINDS)))) % len(_ICON_KINDS)]
            caption_number = 10 + int(rng.randrange(80))
            caption_text = _caption_tag_for_index(int(label_index))
            cards.append(
                _MetricCard(
                    card_id=f"metric_{label_index}",
                    label=str(label),
                    value=int(value),
                    display_text=str(display),
                    unit=str(unit),
                    section=str(section_title),
                    icon_kind=str(icon_kind),
                    color_rgb=tuple(int(channel) for channel in color),
                    caption_number=int(caption_number),
                    caption_text=str(caption_text),
                )
            )
            label_index += 1
    if len(cards) != int(card_count):
        raise ValueError(f"expected {card_count} cards, built {len(cards)} for {query_id}")
    return cards


def _unique_extremum_for_section(
    labels: Sequence[str],
    values_by_label: Mapping[str, int],
    *,
    extremum_kind: str,
) -> Tuple[str, int] | None:
    section_values = [(str(label), int(values_by_label[str(label)])) for label in labels]
    if not section_values:
        return None
    if str(extremum_kind) == "maximum":
        target_value = max(value for _, value in section_values)
    elif str(extremum_kind) == "minimum":
        target_value = min(value for _, value in section_values)
    else:
        raise ValueError(f"unsupported extremum_kind: {extremum_kind}")
    winners = [(label, value) for label, value in section_values if int(value) == int(target_value)]
    if len(winners) != 1:
        return None
    return str(winners[0][0]), int(winners[0][1])


def _sample_section_extrema_query(
    *,
    rng: Any,
    section_titles: Sequence[str],
    labels_by_section: Mapping[str, Sequence[str]],
    values_by_label: Mapping[str, int],
) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    for section_a in section_titles:
        for section_b in section_titles:
            if str(section_a) == str(section_b):
                continue
            for extremum_a in _EXTREMUM_KINDS:
                selected_a = _unique_extremum_for_section(
                    labels_by_section[str(section_a)],
                    values_by_label,
                    extremum_kind=str(extremum_a),
                )
                if selected_a is None:
                    continue
                for extremum_b in _EXTREMUM_KINDS:
                    selected_b = _unique_extremum_for_section(
                        labels_by_section[str(section_b)],
                        values_by_label,
                        extremum_kind=str(extremum_b),
                    )
                    if selected_b is None:
                        continue
                    label_a, value_a = selected_a
                    label_b, value_b = selected_b
                    for operation in _EXTREMA_OPERATIONS:
                        if str(operation) == "sum":
                            answer = int(value_a) + int(value_b)
                        elif str(operation) == "absolute_difference":
                            answer = abs(int(value_a) - int(value_b))
                            if int(answer) == 0:
                                continue
                        else:
                            raise ValueError(f"unsupported extrema operation: {operation}")
                        candidates.append(
                            {
                                "section_a": str(section_a),
                                "section_b": str(section_b),
                                "extremum_a": str(extremum_a),
                                "extremum_b": str(extremum_b),
                                "operation": str(operation),
                                "label_a": str(label_a),
                                "label_b": str(label_b),
                                "value_a": int(value_a),
                                "value_b": int(value_b),
                                "answer": int(answer),
                            }
                        )
    if not candidates:
        raise ValueError("could not build section extrema query with unique section extrema")
    return dict(candidates[int(rng.randrange(len(candidates)))])


def _sample_section_total_extrema_query(
    *,
    section_titles: Sequence[str],
    labels_by_section: Mapping[str, Sequence[str]],
    values_by_label: Dict[str, int],
    value_min: int,
    value_max: int,
) -> Dict[str, Any]:
    """Select the unique highest-total and lowest-total sections."""

    for _ in range(12):
        section_totals = _section_totals(labels_by_section, values_by_label)
        max_total = max(int(value) for value in section_totals.values())
        min_total = min(int(value) for value in section_totals.values())
        max_sections = [str(section) for section in section_titles if int(section_totals[str(section)]) == int(max_total)]
        min_sections = [str(section) for section in section_titles if int(section_totals[str(section)]) == int(min_total)]
        if len(max_sections) == 1 and len(min_sections) == 1 and int(max_total) > int(min_total):
            high_section = str(max_sections[0])
            low_section = str(min_sections[0])
            return {
                "high_section": high_section,
                "low_section": low_section,
                "high_total": int(max_total),
                "low_total": int(min_total),
                "answer": int(max_total) - int(min_total),
                "section_totals": dict(section_totals),
            }

        if len(max_sections) > 1:
            kept_section = str(max_sections[0])
            for rank, section in enumerate(max_sections[1:], start=1):
                lowered = _adjust_section_total(
                    values_by_label,
                    labels_by_section[str(section)],
                    delta=-int(rank),
                    value_min=int(value_min),
                    value_max=int(value_max),
                )
                if not lowered:
                    _adjust_section_total(
                        values_by_label,
                        labels_by_section[kept_section],
                        delta=int(rank),
                        value_min=int(value_min),
                        value_max=int(value_max),
                    )
            continue

        if len(min_sections) > 1:
            kept_section = str(min_sections[0])
            for rank, section in enumerate(min_sections[1:], start=1):
                raised = _adjust_section_total(
                    values_by_label,
                    labels_by_section[str(section)],
                    delta=int(rank),
                    value_min=int(value_min),
                    value_max=int(value_max),
                )
                if not raised:
                    _adjust_section_total(
                        values_by_label,
                        labels_by_section[kept_section],
                        delta=-int(rank),
                        value_min=int(value_min),
                        value_max=int(value_max),
                    )
            continue

    raise ValueError("could not build section-total extrema query with unique highest and lowest sections")


def _resolve_rank_direction(params: Mapping[str, Any], *, instance_seed: int) -> Tuple[str, Dict[str, float]]:
    explicit = params.get("rank_direction")
    if explicit is not None:
        selected = str(explicit)
        if selected not in set(_RANK_DIRECTIONS):
            raise ValueError(f"rank_direction must be one of {_RANK_DIRECTIONS}")
        return selected, {selected: 1.0}
    index = resolve_selection_index(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.rank_direction",
    )
    selected = str(_RANK_DIRECTIONS[int(index) % len(_RANK_DIRECTIONS)])
    probability = 1.0 / float(len(_RANK_DIRECTIONS))
    return selected, {str(direction): probability for direction in _RANK_DIRECTIONS}


def _resolve_rank_position(
    params: Mapping[str, Any],
    *,
    section_count: int,
    instance_seed: int,
) -> Tuple[int, Dict[str, float]]:
    raw_support = params.get(
        "rank_position_support",
        group_default(_GEN_DEFAULTS, "rank_position_support", _RANK_POSITION_SUPPORT),
    )
    support = [int(value) for value in raw_support if int(value) <= int(section_count)]
    if not support:
        support = [2] if int(section_count) >= 2 else [1]
    explicit = params.get("rank_position")
    if explicit is not None:
        selected = int(explicit)
        if int(selected) not in set(support):
            raise ValueError(f"rank_position must be in {support} for section_count={section_count}")
        return int(selected), {str(int(selected)): 1.0}
    index = resolve_selection_index(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.rank_position.{section_count}",
    )
    selected = int(support[int(index) % len(support)])
    probability = 1.0 / float(len(support))
    return int(selected), {str(value): probability for value in support}


def _nudge_duplicate_section_totals(
    values_by_label: Dict[str, int],
    *,
    labels_by_section: Mapping[str, Sequence[str]],
    section_titles: Sequence[str],
    value_min: int,
    value_max: int,
) -> bool:
    totals = _section_totals(labels_by_section, values_by_label)
    seen: Dict[int, str] = {}
    for section in section_titles:
        total = int(totals[str(section)])
        if total not in seen:
            seen[total] = str(section)
            continue
        labels = list(labels_by_section[str(section)])
        if _adjust_section_total(
            values_by_label,
            labels,
            delta=1,
            value_min=int(value_min),
            value_max=int(value_max),
        ):
            return True
        return _adjust_section_total(
            values_by_label,
            labels,
            delta=-1,
            value_min=int(value_min),
            value_max=int(value_max),
        )
    return False


def _sample_section_ranked_total_query(
    *,
    params: Mapping[str, Any],
    section_titles: Sequence[str],
    labels_by_section: Mapping[str, Sequence[str]],
    values_by_label: Dict[str, int],
    value_min: int,
    value_max: int,
    instance_seed: int,
) -> Dict[str, Any]:
    """Select a unique ranked section total."""

    direction, direction_probabilities = _resolve_rank_direction(params, instance_seed=int(instance_seed))
    rank_position, rank_position_probabilities = _resolve_rank_position(
        params,
        section_count=len(section_titles),
        instance_seed=int(instance_seed),
    )
    for _ in range(16):
        section_totals = _section_totals(labels_by_section, values_by_label)
        total_values = [int(value) for value in section_totals.values()]
        if len(set(total_values)) == len(total_values):
            reverse = str(direction) == "highest"
            ranked = sorted(
                ((str(section), int(section_totals[str(section)])) for section in section_titles),
                key=lambda item: item[1],
                reverse=reverse,
            )
            section, total = ranked[int(rank_position) - 1]
            return {
                "answer_section": str(section),
                "answer_total": int(total),
                "rank_direction": str(direction),
                "rank_position": int(rank_position),
                "rank_ordinal": str(_RANK_ORDINALS.get(int(rank_position), f"{rank_position}th")),
                "section_totals": dict(section_totals),
                "rank_direction_probabilities": dict(direction_probabilities),
                "rank_position_probabilities": dict(rank_position_probabilities),
            }
        if not _nudge_duplicate_section_totals(
            values_by_label,
            labels_by_section=labels_by_section,
            section_titles=section_titles,
            value_min=int(value_min),
            value_max=int(value_max),
        ):
            break
    raise ValueError("could not build ranked section-total query with unique section totals")


def _ensure_unique_metric_values(
    labels: Sequence[str],
    values_by_label: Dict[str, int],
    *,
    value_min: int,
    value_max: int,
) -> None:
    if len(labels) > (int(value_max) - int(value_min) + 1):
        raise ValueError("metric ranked item query has more labels than distinct value support")

    used: set[int] = set()
    for raw_label in labels:
        label = str(raw_label)
        current = int(values_by_label[label])
        if int(value_min) <= current <= int(value_max) and current not in used:
            used.add(current)
            continue

        replacement = None
        max_distance = max(abs(current - int(value_min)), abs(current - int(value_max)))
        for distance in range(1, int(max_distance) + 1):
            for candidate in (current + int(distance), current - int(distance)):
                if int(value_min) <= int(candidate) <= int(value_max) and int(candidate) not in used:
                    replacement = int(candidate)
                    break
            if replacement is not None:
                break
        if replacement is None:
            raise ValueError("could not construct unique metric values for ranked item query")
        values_by_label[label] = int(replacement)
        used.add(int(replacement))


def _rank_metric_candidates(
    *,
    labels: Sequence[str],
    labels_by_section: Mapping[str, Sequence[str]],
    values_by_label: Mapping[str, int],
    rank_direction: str,
) -> List[Dict[str, Any]]:
    section_by_label = {
        str(label): str(section)
        for section, section_labels in labels_by_section.items()
        for label in section_labels
    }
    reverse = str(rank_direction) == "highest"
    ranked = sorted(
        ((str(label), int(values_by_label[str(label)])) for label in labels),
        key=lambda item: (item[1], item[0]),
        reverse=reverse,
    )
    return [
        {
            "rank_position": int(index + 1),
            "label": str(label),
            "value": int(value),
            "section": str(section_by_label[str(label)]),
        }
        for index, (label, value) in enumerate(ranked)
    ]


def _sample_metric_ranked_item_query(
    *,
    params: Mapping[str, Any],
    query_id: str,
    labels: Sequence[str],
    section_titles: Sequence[str],
    labels_by_section: Mapping[str, Sequence[str]],
    values_by_label: Dict[str, int],
    value_min: int,
    value_max: int,
    instance_seed: int,
) -> Dict[str, Any]:
    """Select an individual metric card by value rank."""

    rank_direction = "highest" if "highest" in str(query_id) else "lowest"
    rank_scope = "section" if str(query_id).endswith("_in_section_label") else "global"
    if str(rank_scope) == "section":
        eligible_sections = [
            str(section)
            for section in section_titles
            if len(labels_by_section[str(section)]) >= max(_RANK_POSITION_SUPPORT)
        ]
        if not eligible_sections:
            raise ValueError("no section has enough metric cards for ranked item query")
        explicit_section = params.get("target_section")
        if explicit_section is not None:
            target_section = str(explicit_section)
            if target_section not in set(eligible_sections):
                raise ValueError(f"target_section must be one of {eligible_sections}")
        else:
            section_index = resolve_selection_index(
                params=params,
                instance_seed=int(instance_seed),
                namespace=f"{TASK_ID}.metric_ranked_item.section.{query_id}",
            ) % len(eligible_sections)
            target_section = str(eligible_sections[int(section_index)])
        scope_labels = [str(label) for label in labels_by_section[str(target_section)]]
    else:
        target_section = ""
        scope_labels = [str(label) for label in labels]
    _ensure_unique_metric_values(
        scope_labels,
        values_by_label,
        value_min=int(value_min),
        value_max=int(value_max),
    )
    rank_position, rank_position_probabilities = _resolve_rank_position(
        params,
        section_count=len(scope_labels),
        instance_seed=int(instance_seed),
    )
    ranked_candidates = _rank_metric_candidates(
        labels=scope_labels,
        labels_by_section=labels_by_section,
        values_by_label=values_by_label,
        rank_direction=str(rank_direction),
    )
    target = dict(ranked_candidates[int(rank_position) - 1])
    return {
        "answer_label": str(target["label"]),
        "answer_value": int(target["value"]),
        "rank_direction": str(rank_direction),
        "rank_position": int(rank_position),
        "rank_ordinal": str(_RANK_ORDINALS.get(int(rank_position), f"{rank_position}th")),
        "rank_scope": str(rank_scope),
        "target_section": str(target_section),
        "scope_labels": [str(label) for label in scope_labels],
        "ranked_candidates": [dict(candidate) for candidate in ranked_candidates],
        "rank_direction_probabilities": {str(rank_direction): 1.0},
        "rank_position_probabilities": dict(rank_position_probabilities),
    }


def _sample_section_icon_extremum_query(
    *,
    params: Mapping[str, Any],
    section_titles: Sequence[str],
    cards: Sequence[_MetricCard],
    instance_seed: int,
) -> Dict[str, Any]:
    """Select a unique section whose filtered icon-card total is highest/lowest."""

    direction, direction_probabilities = _resolve_rank_direction(params, instance_seed=int(instance_seed))
    cards_by_section_icon: Dict[Tuple[str, str], List[_MetricCard]] = {}
    for card in cards:
        cards_by_section_icon.setdefault((str(card.section), str(card.icon_kind)), []).append(card)

    candidates: List[Dict[str, Any]] = []
    for icon_kind in _ICON_KINDS:
        filtered_totals: Dict[str, int] = {}
        filtered_labels_by_section: Dict[str, List[str]] = {}
        for section in section_titles:
            icon_cards = list(cards_by_section_icon.get((str(section), str(icon_kind)), []))
            if not icon_cards:
                continue
            filtered_totals[str(section)] = int(sum(int(card.value) for card in icon_cards))
            filtered_labels_by_section[str(section)] = [str(card.label) for card in icon_cards]
        if len(filtered_totals) < 2:
            continue
        target_value = max(filtered_totals.values()) if str(direction) == "highest" else min(filtered_totals.values())
        winners = [str(section) for section, total in filtered_totals.items() if int(total) == int(target_value)]
        if len(winners) != 1:
            continue
        answer_section = str(winners[0])
        candidates.append(
            {
                "icon_kind": str(icon_kind),
                "answer_section": str(answer_section),
                "answer_total": int(target_value),
                "target_labels": [str(label) for label in filtered_labels_by_section[str(answer_section)]],
                "filtered_section_totals": {str(section): int(total) for section, total in filtered_totals.items()},
                "rank_direction": str(direction),
                "rank_direction_probabilities": dict(direction_probabilities),
            }
        )
    if not candidates:
        raise ValueError("could not build section-icon extremum query with a unique filtered section total")
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.section_icon_extremum")
    return dict(candidates[int(rng.randrange(len(candidates)))])


def _build_dataset(
    *,
    query_id: str,
    params: Mapping[str, Any],
    instance_seed: int,
) -> Dict[str, Any]:
    card_count, card_count_range, card_count_probabilities = _resolve_int_range(
        params,
        gen_defaults=_GEN_DEFAULTS,
        min_key="card_count_min",
        max_key="card_count_max",
        explicit_key="card_count",
        fallback_min=8,
        fallback_max=16,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.card_count",
    )
    section_count, section_count_range, section_count_probabilities = _resolve_section_count(
        params,
        gen_defaults=_GEN_DEFAULTS,
        query_id=str(query_id),
        card_count=int(card_count),
        instance_seed=int(instance_seed),
    )
    if int(section_count) > int(card_count):
        section_count = int(card_count)

    value_min, value_max = resolve_required_int_bounds(
        params,
        _GEN_DEFAULTS,
        min_key="value_min",
        max_key="value_max",
        fallback_min=10,
        fallback_max=99,
        context=f"generation defaults for {TASK_ID}",
    )
    percent_min, percent_max = resolve_required_int_bounds(
        params,
        _GEN_DEFAULTS,
        min_key="percent_value_min",
        max_key="percent_value_max",
        fallback_min=10,
        fallback_max=90,
        context=f"generation defaults for {TASK_ID}",
    )

    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.dataset.{query_id}")
    label_batch = sample_page_label_batch(
        rng,
        role="metric_card_label",
        count=int(card_count),
        manifest_name="categories/abstract_group_labels.txt",
        min_chars=3,
        max_chars=14,
        allow_spaces=False,
        allow_punctuation=False,
    )
    section_batch = sample_page_label_batch(
        rng,
        role="metric_section_title",
        count=int(section_count),
        manifest_name="categories/product_labels.txt",
        min_chars=3,
        max_chars=16,
        allow_spaces=True,
        allow_punctuation=False,
        exclude=label_batch.values,
    )
    labels = list(label_batch.values)
    section_titles = list(section_batch.values)
    page_text_resources = page_text_resource_metadata(label_batch, section_batch)
    section_card_counts = _partition_cards(int(card_count), int(section_count))

    values_by_label: Dict[str, int] = {
        str(label): int(rng.randint(int(value_min), int(value_max)))
        for label in labels
    }
    labels_by_section = _labels_by_section(
        labels=labels,
        section_titles=section_titles,
        section_card_counts=section_card_counts,
    )
    target_labels: List[str]
    target_values: List[int]
    target_groups: Dict[str, List[str]] = {}
    target_sections: List[str] = []
    excluded_labels: List[str] = []
    target_extrema: List[str] = []
    extrema_operation = ""
    rank_direction = ""
    rank_ordinal = ""
    rank_position = 0
    rank_scope = ""
    ranked_candidates: List[Dict[str, Any]] = []
    rank_direction_probabilities: Dict[str, float] = {}
    rank_position_probabilities: Dict[str, float] = {}
    target_operand_count = 1
    percent_mode = False
    answer_value: int | str
    answer_type = "integer"
    arithmetic_expression: str
    filter_icon_kind = ""
    comparison_icon_kind = ""
    filtered_section_totals: Dict[str, int] = {}
    prebuilt_cards: List[_MetricCard] | None = None
    lookup_target_value = ""
    lookup_target_detail = ""
    annotation_targets: List[Dict[str, str]] = []

    if str(query_id) == "sum_named_metrics":
        operand_count, operand_count_range, operand_count_probabilities = _resolve_int_range(
            params,
            gen_defaults=_GEN_DEFAULTS,
            min_key="sum_operand_count_min",
            max_key="sum_operand_count_max",
            explicit_key="operand_count",
            fallback_min=4,
            fallback_max=6,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}.sum_operand_count",
        )
        target_operand_count = int(operand_count)
        target_labels = rng.sample(labels, k=int(target_operand_count))
        target_values = [int(values_by_label[label]) for label in target_labels]
        answer_value = int(sum(target_values))
        arithmetic_expression = " + ".join(str(value) for value in target_values)
    elif str(query_id) == "section_extrema_arithmetic":
        extrema_query = _sample_section_extrema_query(
            rng=rng,
            section_titles=section_titles,
            labels_by_section=labels_by_section,
            values_by_label=values_by_label,
        )
        section_a = str(extrema_query["section_a"])
        section_b = str(extrema_query["section_b"])
        label_a = str(extrema_query["label_a"])
        label_b = str(extrema_query["label_b"])
        value_a = int(extrema_query["value_a"])
        value_b = int(extrema_query["value_b"])
        target_sections = [section_a, section_b]
        target_extrema = [str(extrema_query["extremum_a"]), str(extrema_query["extremum_b"])]
        extrema_operation = str(extrema_query["operation"])
        target_groups = {"section_a_extremum": [label_a], "section_b_extremum": [label_b]}
        target_labels = [label_a, label_b]
        target_values = [value_a, value_b]
        annotation_targets = [
            {"key": "section_a_extremum", "label": label_a, "bbox_kind": "card"},
            {"key": "section_b_extremum", "label": label_b, "bbox_kind": "card"},
        ]
        target_operand_count = 2
        operand_count_range = (2, 2)
        operand_count_probabilities = {"2": 1.0}
        answer_value = int(extrema_query["answer"])
        if str(extrema_operation) == "sum":
            arithmetic_expression = f"{target_extrema[0]}({section_a}) + {target_extrema[1]}({section_b})"
        else:
            arithmetic_expression = f"abs({target_extrema[0]}({section_a}) - {target_extrema[1]}({section_b}))"
    elif str(query_id) == "section_total_extrema_difference":
        total_query = _sample_section_total_extrema_query(
            section_titles=section_titles,
            labels_by_section=labels_by_section,
            values_by_label=values_by_label,
            value_min=int(value_min),
            value_max=int(value_max),
        )
        section_a = str(total_query["high_section"])
        section_b = str(total_query["low_section"])
        group_a = list(labels_by_section[section_a])
        group_b = list(labels_by_section[section_b])
        target_sections = [section_a, section_b]
        target_groups = {"highest_total_section": list(group_a), "lowest_total_section": list(group_b)}
        target_labels = list(group_a) + list(group_b)
        sum_a = int(total_query["high_total"])
        sum_b = int(total_query["low_total"])
        target_values = [int(values_by_label[label]) for label in target_labels]
        target_operand_count = len(target_labels)
        operand_count_range = (min(section_card_counts) * 2, max(section_card_counts) * 2)
        operand_count_probabilities = {str(target_operand_count): 1.0}
        answer_value = int(sum_a - sum_b)
        target_extrema = ["highest_total", "lowest_total"]
        extrema_operation = "absolute_difference"
        arithmetic_expression = f"sum({section_a}) - sum({section_b})"
    elif str(query_id) == "section_total_except_named":
        exclusion_count, exclusion_count_range, _exclusion_count_probabilities = _resolve_int_range(
            params,
            gen_defaults=_GEN_DEFAULTS,
            min_key="excluded_metric_count_min",
            max_key="excluded_metric_count_max",
            explicit_key="excluded_metric_count",
            fallback_min=2,
            fallback_max=2,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}.excluded_metric_count",
        )
        included_count_min = int(params.get(
            "section_except_included_count_min",
            group_default(_GEN_DEFAULTS, "section_except_included_count_min", 3),
        ))
        eligible_sections = [
            str(section_title)
            for section_title, section_labels in labels_by_section.items()
            if len(section_labels) - int(exclusion_count) >= int(included_count_min)
        ]
        if not eligible_sections:
            raise ValueError(
                f"no section has at least {included_count_min} included cards after excluding "
                f"{exclusion_count} cards for {query_id}"
            )
        target_section = str(eligible_sections[int(rng.randrange(len(eligible_sections)))])
        section_labels = list(labels_by_section[target_section])
        excluded_labels = [str(label) for label in rng.sample(section_labels, k=int(exclusion_count))]
        included_labels = [str(label) for label in section_labels if str(label) not in set(excluded_labels)]
        target_sections = [target_section]
        target_groups = {"included": list(included_labels), "excluded": list(excluded_labels)}
        target_labels = list(included_labels)
        target_values = [int(values_by_label[label]) for label in target_labels]
        target_operand_count = len(included_labels)
        operand_count_range = (
            max(1, min(section_card_counts) - int(exclusion_count_range[1])),
            max(section_card_counts) - int(exclusion_count_range[0]),
        )
        operand_count_probabilities = {str(target_operand_count): 1.0}
        answer_value = int(sum(int(values_by_label[label]) for label in included_labels))
        arithmetic_expression = f"sum({target_section}) - ({' + '.join(str(values_by_label[label]) for label in excluded_labels)})"
    elif str(query_id) == "section_ranked_total_label":
        ranked_query = _sample_section_ranked_total_query(
            params=params,
            section_titles=section_titles,
            labels_by_section=labels_by_section,
            values_by_label=values_by_label,
            value_min=int(value_min),
            value_max=int(value_max),
            instance_seed=int(instance_seed),
        )
        target_section = str(ranked_query["answer_section"])
        section_labels = [str(label) for label in labels_by_section[target_section]]
        target_sections = [target_section]
        target_groups = {"answer_section": list(section_labels)}
        target_labels = list(section_labels)
        target_values = [int(values_by_label[label]) for label in target_labels]
        target_operand_count = len(target_labels)
        operand_count_range = (min(section_card_counts), max(section_card_counts))
        operand_count_probabilities = {str(target_operand_count): 1.0}
        answer_value = str(target_section)
        answer_type = "string"
        rank_direction = str(ranked_query["rank_direction"])
        rank_ordinal = str(ranked_query["rank_ordinal"])
        rank_position = int(ranked_query["rank_position"])
        rank_direction_probabilities = dict(ranked_query["rank_direction_probabilities"])
        rank_position_probabilities = dict(ranked_query["rank_position_probabilities"])
        arithmetic_expression = f"rank_{rank_position}_{rank_direction}_section_total"
    elif str(query_id) == "section_icon_total_value":
        prebuilt_cards = _build_cards(
            query_id=str(query_id),
            card_count=int(card_count),
            section_titles=section_titles,
            section_card_counts=section_card_counts,
            values_by_label=values_by_label,
            percent_mode=bool(percent_mode),
            instance_seed=int(instance_seed),
        )
        cards_by_section_icon: Dict[Tuple[str, str], List[_MetricCard]] = {}
        for card in prebuilt_cards:
            cards_by_section_icon.setdefault((str(card.section), str(card.icon_kind)), []).append(card)
        preferred = [
            (section, icon, cards_for_icon)
            for (section, icon), cards_for_icon in cards_by_section_icon.items()
            if len(cards_for_icon) >= 2
        ]
        candidates = preferred or [
            (section, icon, cards_for_icon)
            for (section, icon), cards_for_icon in cards_by_section_icon.items()
            if cards_for_icon
        ]
        if not candidates:
            raise ValueError("could not build section-icon filtered total query")
        target_section, filter_icon_kind, icon_cards = candidates[int(rng.randrange(len(candidates)))]
        filtered_labels = [str(card.label) for card in icon_cards]
        target_sections = [str(target_section)]
        target_groups = {"filtered_icon_cards": list(filtered_labels)}
        target_labels = list(filtered_labels)
        target_values = [int(values_by_label[label]) for label in target_labels]
        target_operand_count = len(target_labels)
        operand_count_range = (1, max(section_card_counts))
        operand_count_probabilities = {str(target_operand_count): 1.0}
        answer_value = int(sum(target_values))
        arithmetic_expression = f"sum({target_section}, icon={filter_icon_kind})"
    elif str(query_id) == "section_icon_total_difference_value":
        prebuilt_cards = _build_cards(
            query_id=str(query_id),
            card_count=int(card_count),
            section_titles=section_titles,
            section_card_counts=section_card_counts,
            values_by_label=values_by_label,
            percent_mode=bool(percent_mode),
            instance_seed=int(instance_seed),
        )
        cards_by_section_icon: Dict[Tuple[str, str], List[_MetricCard]] = {}
        for card in prebuilt_cards:
            cards_by_section_icon.setdefault((str(card.section), str(card.icon_kind)), []).append(card)
        candidates: List[Dict[str, Any]] = []
        for icon_kind in _ICON_KINDS:
            sections_with_icon = [
                str(section)
                for section in section_titles
                if cards_by_section_icon.get((str(section), str(icon_kind)))
            ]
            for index_a, section_a in enumerate(sections_with_icon):
                for section_b in sections_with_icon[index_a + 1 :]:
                    group_a = cards_by_section_icon[(str(section_a), str(icon_kind))]
                    group_b = cards_by_section_icon[(str(section_b), str(icon_kind))]
                    total_a = sum(int(card.value) for card in group_a)
                    total_b = sum(int(card.value) for card in group_b)
                    gap = abs(int(total_a) - int(total_b))
                    if int(gap) == 0:
                        continue
                    candidates.append(
                        {
                            "icon_kind": str(icon_kind),
                            "section_a": str(section_a),
                            "section_b": str(section_b),
                            "group_a": [str(card.label) for card in group_a],
                            "group_b": [str(card.label) for card in group_b],
                            "gap": int(gap),
                        }
                    )
        if not candidates:
            raise ValueError("could not build section-icon total difference query")
        selected = dict(candidates[int(rng.randrange(len(candidates)))])
        comparison_icon_kind = str(selected["icon_kind"])
        section_a = str(selected["section_a"])
        section_b = str(selected["section_b"])
        group_a = [str(label) for label in selected["group_a"]]
        group_b = [str(label) for label in selected["group_b"]]
        target_sections = [section_a, section_b]
        target_groups = {"section_a_filtered_icon_cards": list(group_a), "section_b_filtered_icon_cards": list(group_b)}
        target_labels = list(group_a) + list(group_b)
        target_values = [int(values_by_label[label]) for label in target_labels]
        target_operand_count = len(target_labels)
        operand_count_range = (2, max(section_card_counts) * 2)
        operand_count_probabilities = {str(target_operand_count): 1.0}
        answer_value = int(selected["gap"])
        arithmetic_expression = f"abs(sum({section_a}, icon={comparison_icon_kind}) - sum({section_b}, icon={comparison_icon_kind}))"
    elif str(query_id) == "section_icon_extremum_label":
        prebuilt_cards = _build_cards(
            query_id=str(query_id),
            card_count=int(card_count),
            section_titles=section_titles,
            section_card_counts=section_card_counts,
            values_by_label=values_by_label,
            percent_mode=bool(percent_mode),
            instance_seed=int(instance_seed),
        )
        extremum_query = _sample_section_icon_extremum_query(
            params=params,
            section_titles=section_titles,
            cards=prebuilt_cards,
            instance_seed=int(instance_seed),
        )
        comparison_icon_kind = str(extremum_query["icon_kind"])
        target_section = str(extremum_query["answer_section"])
        filtered_labels = [str(label) for label in extremum_query["target_labels"]]
        filtered_section_totals = {
            str(section): int(total)
            for section, total in dict(extremum_query["filtered_section_totals"]).items()
        }
        target_sections = [str(target_section)]
        target_groups = {"answer_section_filtered_icon_cards": list(filtered_labels)}
        target_labels = list(filtered_labels)
        target_values = [int(values_by_label[label]) for label in target_labels]
        target_operand_count = len(target_labels)
        operand_count_range = (1, max(section_card_counts))
        operand_count_probabilities = {str(target_operand_count): 1.0}
        answer_value = str(target_section)
        answer_type = "string"
        rank_direction = str(extremum_query["rank_direction"])
        rank_direction_probabilities = dict(extremum_query["rank_direction_probabilities"])
        annotation_targets = [
            {
                "key": "answer_section",
                "section": str(target_section),
                "bbox_kind": "section",
            }
        ]
        arithmetic_expression = f"{rank_direction}_section_total(icon={comparison_icon_kind})"
    elif str(query_id) in METRIC_RANKED_ITEM_VARIANTS:
        ranked_query = _sample_metric_ranked_item_query(
            params=params,
            query_id=str(query_id),
            labels=labels,
            section_titles=section_titles,
            labels_by_section=labels_by_section,
            values_by_label=values_by_label,
            value_min=int(value_min),
            value_max=int(value_max),
            instance_seed=int(instance_seed),
        )
        target_label = str(ranked_query["answer_label"])
        target_section = str(ranked_query["target_section"])
        scope_labels = [str(label) for label in ranked_query["scope_labels"]]
        target_sections = [str(target_section)] if target_section else []
        target_groups = {"rank_scope": list(scope_labels), "target_metric": [str(target_label)]}
        target_labels = [str(target_label)]
        target_values = [int(values_by_label[str(target_label)])]
        target_operand_count = len(scope_labels)
        operand_count_range = (max(_RANK_POSITION_SUPPORT), int(card_count))
        operand_count_probabilities = {str(target_operand_count): 1.0}
        answer_value = str(target_label)
        answer_type = "string"
        rank_direction = str(ranked_query["rank_direction"])
        rank_ordinal = str(ranked_query["rank_ordinal"])
        rank_position = int(ranked_query["rank_position"])
        rank_scope = str(ranked_query["rank_scope"])
        ranked_candidates = [dict(candidate) for candidate in ranked_query["ranked_candidates"]]
        rank_direction_probabilities = dict(ranked_query["rank_direction_probabilities"])
        rank_position_probabilities = dict(ranked_query["rank_position_probabilities"])
        annotation_targets = []
        if target_section:
            annotation_targets.append(
                {
                    "key": "section",
                    "section": str(target_section),
                    "bbox_kind": "section",
                }
            )
        annotation_targets.append({"key": "target_card", "label": str(target_label), "bbox_kind": "card"})
        arithmetic_expression = f"rank_{rank_position}_{rank_direction}_metric(scope={rank_scope})"
    else:
        raise ValueError(f"unsupported query_id: {query_id}")

    cards = prebuilt_cards or _build_cards(
        query_id=str(query_id),
        card_count=int(card_count),
        section_titles=section_titles,
        section_card_counts=section_card_counts,
        values_by_label=values_by_label,
        percent_mode=bool(percent_mode),
        instance_seed=int(instance_seed),
    )
    return {
        "cards": cards,
        "labels": [str(label) for label in labels],
        "values_by_label": dict(values_by_label),
        "section_titles": [str(title) for title in section_titles],
        "section_card_counts": [int(value) for value in section_card_counts],
        "section_totals": dict(_section_totals(labels_by_section, values_by_label)),
        "card_count": int(card_count),
        "card_count_range": [int(card_count_range[0]), int(card_count_range[1])],
        "card_count_probabilities": dict(card_count_probabilities),
        "section_count": int(section_count),
        "section_count_range": [int(section_count_range[0]), int(section_count_range[1])],
        "section_count_probabilities": dict(section_count_probabilities),
        "target_labels": [str(label) for label in target_labels],
        "target_values": [int(value) for value in target_values],
        "target_groups": {str(key): [str(label) for label in value] for key, value in target_groups.items()},
        "target_sections": [str(section) for section in target_sections],
        "excluded_labels": [str(label) for label in excluded_labels],
        "target_extrema": [str(extremum) for extremum in target_extrema],
        "extrema_operation": str(extrema_operation),
        "target_operand_count": int(target_operand_count),
        "target_operand_count_range": [int(operand_count_range[0]), int(operand_count_range[1])],
        "target_operand_count_probabilities": dict(operand_count_probabilities),
        "answer_value": answer_value if str(answer_type) == "string" else int(answer_value),
        "answer_type": str(answer_type),
        "rank_direction": str(rank_direction),
        "rank_ordinal": str(rank_ordinal),
        "rank_position": int(rank_position),
        "rank_scope": str(rank_scope),
        "ranked_candidates": [dict(candidate) for candidate in ranked_candidates],
        "rank_direction_probabilities": dict(rank_direction_probabilities),
        "rank_position_probabilities": dict(rank_position_probabilities),
        "filter_icon_kind": str(filter_icon_kind),
        "comparison_icon_kind": str(comparison_icon_kind),
        "filtered_section_totals": {str(section): int(total) for section, total in filtered_section_totals.items()},
        "target_value_text": str(lookup_target_value),
        "target_detail_text": str(lookup_target_detail),
        "annotation_targets": [dict(item) for item in annotation_targets],
        "arithmetic_expression": str(arithmetic_expression),
        "percent_mode": bool(percent_mode),
        "value_min": int(value_min),
        "value_max": int(value_max),
        "percent_value_min": int(percent_min),
        "percent_value_max": int(percent_max),
        "page_text_resources": dict(page_text_resources),
    }
