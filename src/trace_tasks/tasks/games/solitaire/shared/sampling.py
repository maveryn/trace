"""Identity-free sampling primitives for solitaire tableau scenes."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice

from .defaults import DEFAULTS, GEN_DEFAULTS
from .rules import (
    card_color,
    deck,
    is_legal_foundation_move,
    is_legal_tableau_move,
    remove_card,
)
from .state import (
    MOVE_OPTION_LABELS,
    SUITS,
    SUIT_SHORT,
    Card,
    CardOption,
    Foundation,
    MoveOption,
    SolitaireSample,
)


def sample_integer_axis(
    *,
    namespace: str,
    instance_seed: int,
    params: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[int],
    axis_name: str,
    balanced_flag_key: str,
) -> Tuple[int, Dict[str, float]]:
    value, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GEN_DEFAULTS,
        support_key=str(support_key),
        explicit_key=str(explicit_key),
        fallback_support=tuple(int(item) for item in fallback_support),
        namespace=f"{str(namespace)}.{str(axis_name)}",
        balanced_flag_key=str(balanced_flag_key),
        namespace_support_permutation=True,
    )
    return int(value), dict(probabilities)


def sample_foundations(rng) -> Tuple[Foundation, ...]:
    foundations: List[Foundation] = []
    for suit in SUITS:
        top_rank = int(rng.randrange(0, 7))
        foundations.append(
            Foundation(
                foundation_id=f"foundation_{SUIT_SHORT[str(suit)].lower()}",
                suit_name=str(suit),
                top_rank_value=int(top_rank),
            )
        )
    return tuple(foundations)


def deck_after_foundations(foundations: Sequence[Foundation]) -> List[Tuple[int, str]]:
    top_by_suit = {str(foundation.suit_name): int(foundation.top_rank_value) for foundation in foundations}
    return [
        (int(rank), str(suit))
        for rank, suit in deck()
        if int(rank) > int(top_by_suit.get(str(suit), 0))
    ]


def build_columns_from_exposed(
    rng,
    *,
    exposed_cards: Sequence[Card],
    pool: List[Tuple[int, str]],
    scene_variant: str,
) -> Tuple[Tuple[Card, ...], ...]:
    columns: List[Tuple[Card, ...]] = []
    for index, exposed in enumerate(exposed_cards):
        filler_count = int(rng.randrange(0, 3 if str(scene_variant) == "freecell_tableau" else 4))
        filler_cards: List[Card] = []
        for filler_index in range(filler_count):
            if not pool:
                break
            rank, suit = pool.pop(int(rng.randrange(len(pool))))
            filler_cards.append(
                Card(
                    card_id=f"col_{index + 1:02d}_card_{filler_index + 1:02d}",
                    rank_value=int(rank),
                    suit_name=str(suit),
                    badge_text=None,
                )
            )
        exposed_id = f"col_{index + 1:02d}_card_{len(filler_cards) + 1:02d}"
        columns.append(
            tuple(
                [
                    *filler_cards,
                    Card(
                        card_id=str(exposed_id),
                        rank_value=int(exposed.rank_value),
                        suit_name=str(exposed.suit_name),
                        badge_text=str(exposed.badge_text) if exposed.badge_text else None,
                    ),
                ]
            )
        )
    return tuple(columns)


def build_columns_from_optional_exposed(
    rng,
    *,
    exposed_cards_or_empty: Sequence[Tuple[int, str] | None],
    pool: List[Tuple[int, str]],
    scene_variant: str,
) -> Tuple[Tuple[Card, ...], ...]:
    """Build tableau columns where `None` means an empty destination slot."""

    columns: List[Tuple[Card, ...]] = []
    for col_index, raw in enumerate(exposed_cards_or_empty):
        if raw is None:
            columns.append(())
            continue
        filler_count = int(rng.randrange(0, 3 if str(scene_variant) == "freecell_tableau" else 4))
        filler_cards: List[Card] = []
        for filler_index in range(filler_count):
            if not pool:
                break
            rank, suit = pool.pop(int(rng.randrange(len(pool))))
            filler_cards.append(
                Card(
                    card_id=f"col_{col_index + 1:02d}_card_{filler_index + 1:02d}",
                    rank_value=int(rank),
                    suit_name=str(suit),
                    badge_text=None,
                )
            )
        columns.append(
            tuple(
                [
                    *filler_cards,
                    Card(
                        card_id=f"col_{col_index + 1:02d}_card_{len(filler_cards) + 1:02d}",
                        rank_value=int(raw[0]),
                        suit_name=str(raw[1]),
                        badge_text=None,
                    ),
                ]
            )
        )
    return tuple(columns)


def visible_exposed_cards(columns: Sequence[Sequence[Card]]) -> Tuple[Card, ...]:
    return tuple(column[-1] for column in columns if len(column) > 0)


def foundation_by_id(foundations: Sequence[Foundation]) -> Dict[str, Foundation]:
    return {str(foundation.foundation_id): foundation for foundation in foundations}


def card_by_id(columns: Sequence[Sequence[Card]]) -> Dict[str, Card]:
    return {str(card.card_id): card for column in columns for card in column}


def card_column_label(columns: Sequence[Sequence[Card]], card_id: str) -> str:
    """Return the visible tableau column label for a sampled card id."""

    for col_index, column in enumerate(columns):
        if any(str(card.card_id) == str(card_id) for card in column):
            return f"Col {int(col_index) + 1}"
    raise KeyError(f"unknown solitaire card id: {card_id}")


def target_is_legal(source: Card, target_id: str, *, columns: Sequence[Sequence[Card]], foundations: Sequence[Foundation]) -> bool:
    cards = card_by_id(columns)
    if str(target_id) in cards:
        return is_legal_tableau_move(source, cards[str(target_id)])
    foundations_by_id = foundation_by_id(foundations)
    if str(target_id) in foundations_by_id:
        return is_legal_foundation_move(source, foundations_by_id[str(target_id)])
    return False


def sample_move_option_count(*, namespace: str, instance_seed: int, params: Mapping[str, Any]) -> Tuple[int, Dict[str, float]]:
    return sample_integer_axis(
        namespace=str(namespace),
        instance_seed=int(instance_seed),
        params=params,
        support_key="move_option_count_support",
        explicit_key="option_count",
        fallback_support=DEFAULTS.move_option_count_support,
        axis_name="move_option_count",
        balanced_flag_key="balanced_option_count_sampling",
    )


def answer_option_label(*, instance_seed: int, params: Mapping[str, Any], option_count: int) -> str:
    raw = params.get("answer_option_label")
    labels = MOVE_OPTION_LABELS[: int(option_count)]
    if raw is not None:
        label = str(raw).strip().upper()
        if label not in labels:
            raise ValueError(f"unsupported answer_option_label for {option_count} options: {label}")
        return str(label)
    rng = spawn_rng(int(instance_seed), "games.solitaire.answer_option_label")
    return str(uniform_choice(rng, labels))


def sample_card_option_count(*, namespace: str, instance_seed: int, params: Mapping[str, Any]) -> Tuple[int, Dict[str, float]]:
    return sample_integer_axis(
        namespace=str(namespace),
        instance_seed=int(instance_seed),
        params=params,
        support_key="card_option_count_support",
        explicit_key="option_count",
        fallback_support=DEFAULTS.card_option_count_support,
        axis_name="card_option_count",
        balanced_flag_key="balanced_option_count_sampling",
    )


def ordinal_label(value: int) -> str:
    """Return a compact English ordinal for a small tableau depth."""

    number = int(value)
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def sample_cascade_card_at_depth(
    rng,
    *,
    namespace: str,
    instance_seed: int,
    params: Mapping[str, Any],
    scene_variant: str,
    target_depth: int,
    target_column: int | None = None,
) -> SolitaireSample:
    """Construct a tableau and card-face options for one column/depth lookup."""

    option_count, option_count_probabilities = sample_card_option_count(
        namespace=str(namespace),
        instance_seed=int(instance_seed),
        params=params,
    )
    answer_label = answer_option_label(instance_seed=int(instance_seed), params=params, option_count=int(option_count))
    column_count = 7 if str(scene_variant) == "klondike_tableau" else 8
    target_col_index = int(target_column) - 1 if target_column is not None else int(rng.randrange(column_count))
    if target_col_index < 0 or target_col_index >= int(column_count):
        raise ValueError(f"target_column out of range for solitaire scene: {target_column}")
    depth = int(target_depth)
    if depth < 1 or depth > 6:
        raise ValueError(f"unsupported solitaire target depth: {target_depth}")

    for _attempt in range(300):
        pool = deck()
        columns: List[Tuple[Card, ...]] = []
        for col_index in range(column_count):
            if int(col_index) == int(target_col_index):
                length = int(rng.randint(depth, 6))
            else:
                length = int(rng.randint(1, 6))
            if len(pool) < int(length):
                raise ValueError("not enough cards to sample solitaire cascade-depth scene")
            cards: List[Card] = []
            for row_index in range(int(length)):
                rank, suit = pool.pop(int(rng.randrange(len(pool))))
                cards.append(
                    Card(
                        card_id=f"col_{col_index + 1:02d}_card_{row_index + 1:02d}",
                        rank_value=int(rank),
                        suit_name=str(suit),
                        badge_text=None,
                    )
                )
            columns.append(tuple(cards))

        target_card = columns[int(target_col_index)][int(depth) - 1]
        visible_cards = [card for column in columns for card in column if str(card.card_id) != str(target_card.card_id)]
        rng.shuffle(visible_cards)
        distractor_cards = visible_cards[: max(0, int(option_count) - 1)]
        if len(distractor_cards) < int(option_count) - 1:
            continue
        options: List[CardOption] = []
        distractor_cursor = 0
        for label in MOVE_OPTION_LABELS[: int(option_count)]:
            if str(label) == str(answer_label):
                option_card = Card(
                    card_id=f"card_option_{str(label).lower()}_card",
                    rank_value=int(target_card.rank_value),
                    suit_name=str(target_card.suit_name),
                    badge_text=None,
                )
                options.append(CardOption(f"card_option_{str(label).lower()}", str(label), option_card, True))
            else:
                source = distractor_cards[int(distractor_cursor)]
                distractor_cursor += 1
                option_card = Card(
                    card_id=f"card_option_{str(label).lower()}_card",
                    rank_value=int(source.rank_value),
                    suit_name=str(source.suit_name),
                    badge_text=None,
                )
                options.append(CardOption(f"card_option_{str(label).lower()}", str(label), option_card, False))
        return SolitaireSample(
            scene_variant=str(scene_variant),
            columns=tuple(columns),
            foundations=sample_foundations(rng),
            answer=str(answer_label),
            answer_type="option_letter",
            annotation_entity_ids=(str(target_card.card_id),),
            move_options=(),
            metadata={
                "target_column_index": int(target_col_index),
                "target_column_number": int(target_col_index) + 1,
                "target_depth": int(depth),
                "target_depth_ordinal": ordinal_label(int(depth)),
                "target_card_id": str(target_card.card_id),
                "target_card_label": str(target_card.label),
                "answer_option_label": str(answer_label),
                "option_count": int(option_count),
                "option_count_probabilities": dict(option_count_probabilities),
                "card_options": [
                    {
                        "label": str(option.label),
                        "option_id": str(option.option_id),
                        "rank_value": int(option.card.rank_value),
                        "rank_label": str(option.card.rank_label),
                        "suit_name": str(option.card.suit_name),
                        "suit_short": str(option.card.suit_short),
                        "card_label": str(option.card.label),
                        "is_answer": bool(option.is_answer),
                    }
                    for option in options
                ],
            },
            card_options=tuple(options),
        )
    raise ValueError("failed to sample solitaire cascade-card-at-depth scene")


def sample_move_legality(
    rng,
    *,
    namespace: str,
    instance_seed: int,
    params: Mapping[str, Any],
    scene_variant: str,
) -> SolitaireSample:
    """Construct a tableau where exactly one displayed source-to-target move option is legal."""

    option_count, option_count_probabilities = sample_move_option_count(
        namespace=str(namespace),
        instance_seed=int(instance_seed),
        params=params,
    )
    answer_label = answer_option_label(instance_seed=int(instance_seed), params=params, option_count=int(option_count))
    for _attempt in range(600):
        foundations = sample_foundations(rng)
        pool = deck_after_foundations(foundations)
        column_count = 7 if str(scene_variant) == "klondike_tableau" else 8
        mode = "foundation" if int(rng.randrange(2)) == 0 else "tableau"
        exposed_raw: List[Tuple[int, str]] = []
        answer_source: Tuple[int, str]
        answer_target_id: str
        answer_target_label: str
        if mode == "foundation":
            candidate_foundations = [
                foundation
                for foundation in foundations
                if int(foundation.top_rank_value) < 13 and (int(foundation.top_rank_value) + 1, str(foundation.suit_name)) in pool
            ]
            if not candidate_foundations:
                continue
            foundation = candidate_foundations[int(rng.randrange(len(candidate_foundations)))]
            answer_source = (int(foundation.top_rank_value) + 1, str(foundation.suit_name))
            answer_target_id = str(foundation.foundation_id)
            answer_target_label = str(foundation.label)
            remove_card(pool, answer_source)
            exposed_raw.append(answer_source)
        else:
            target_rank = int(rng.randrange(2, 14))
            target_suit = str(SUITS[int(rng.randrange(len(SUITS)))])
            opposite = [suit for suit in SUITS if card_color(str(suit)) != card_color(str(target_suit))]
            source_suit = str(opposite[int(rng.randrange(len(opposite)))])
            answer_source = (int(target_rank - 1), str(source_suit))
            target_raw = (int(target_rank), str(target_suit))
            if answer_source not in pool or target_raw not in pool:
                continue
            remove_card(pool, answer_source)
            remove_card(pool, target_raw)
            exposed_raw.extend([answer_source, target_raw])
            answer_target_id = "pending_target_card"
            answer_target_label = "pending"

        while len(exposed_raw) < int(column_count):
            if not pool:
                break
            exposed_raw.append(pool.pop(int(rng.randrange(len(pool)))))
        if len(exposed_raw) != int(column_count):
            continue
        rng.shuffle(exposed_raw)
        exposed_cards = tuple(
            Card(
                card_id=f"exposed_{index + 1:02d}",
                rank_value=int(rank),
                suit_name=str(suit),
                badge_text=None,
            )
            for index, (rank, suit) in enumerate(exposed_raw)
        )
        columns = build_columns_from_exposed(rng, exposed_cards=exposed_cards, pool=pool, scene_variant=str(scene_variant))
        exposed_by_raw = {
            (int(card.rank_value), str(card.suit_name)): card
            for card in visible_exposed_cards(columns)
        }
        source_card = exposed_by_raw.get((int(answer_source[0]), str(answer_source[1])))
        if source_card is None:
            continue
        if mode == "tableau":
            legal_targets = [
                card
                for card in visible_exposed_cards(columns)
                if str(card.card_id) != str(source_card.card_id) and is_legal_tableau_move(source_card, card)
            ]
            if not legal_targets:
                continue
            target_card = legal_targets[0]
            answer_target_id = str(target_card.card_id)
            answer_target_label = card_column_label(columns, str(target_card.card_id))
        answer_source_label = card_column_label(columns, str(source_card.card_id))

        all_targets: List[Tuple[str, str]] = [
            (str(card.card_id), card_column_label(columns, str(card.card_id)))
            for card in visible_exposed_cards(columns)
            if str(card.card_id) != str(source_card.card_id)
        ] + [(str(foundation.foundation_id), str(foundation.label)) for foundation in foundations]
        all_sources = [
            (str(card.card_id), card_column_label(columns, str(card.card_id)))
            for card in visible_exposed_cards(columns)
        ]
        distractor_pairs: List[Tuple[str, str, str, str]] = []
        for src_id, src_label in all_sources:
            src_card = card_by_id(columns)[str(src_id)]
            for target_id, target_label in all_targets:
                if str(src_id) == str(target_id):
                    continue
                is_answer_pair = str(src_id) == str(source_card.card_id) and str(target_id) == str(answer_target_id)
                if is_answer_pair:
                    continue
                if not target_is_legal(src_card, str(target_id), columns=columns, foundations=foundations):
                    distractor_pairs.append((str(src_id), str(src_label), str(target_id), str(target_label)))
        rng.shuffle(distractor_pairs)
        if len(distractor_pairs) < int(option_count) - 1:
            continue
        options: List[MoveOption] = []
        distractor_cursor = 0
        for label in MOVE_OPTION_LABELS[: int(option_count)]:
            if str(label) == str(answer_label):
                options.append(
                    MoveOption(
                        option_id=f"move_option_{str(label).lower()}",
                        label=str(label),
                        source_card_id=str(source_card.card_id),
                        source_label=str(answer_source_label),
                        target_id=str(answer_target_id),
                        target_label=str(answer_target_label),
                        is_answer=True,
                    )
                )
            else:
                src_id, src_label, target_id, target_label = distractor_pairs[int(distractor_cursor)]
                distractor_cursor += 1
                options.append(
                    MoveOption(
                        option_id=f"move_option_{str(label).lower()}",
                        label=str(label),
                        source_card_id=str(src_id),
                        source_label=str(src_label),
                        target_id=str(target_id),
                        target_label=str(target_label),
                        is_answer=False,
                    )
                )
        legal_options = [
            option
            for option in options
            if target_is_legal(card_by_id(columns)[str(option.source_card_id)], str(option.target_id), columns=columns, foundations=foundations)
        ]
        if len(legal_options) != 1 or str(legal_options[0].label) != str(answer_label):
            continue
        annotation = (str(source_card.card_id), str(answer_target_id))
        return SolitaireSample(
            scene_variant=str(scene_variant),
            columns=tuple(columns),
            foundations=tuple(foundations),
            answer=str(answer_label),
            answer_type="option_letter",
            annotation_entity_ids=tuple(annotation),
            move_options=tuple(options),
            metadata={
                "legal_move_kind": str(mode),
                "answer_option_label": str(answer_label),
                "option_count": int(option_count),
                "option_count_probabilities": dict(option_count_probabilities),
                "legal_source_id": str(source_card.card_id),
                "legal_source_label": str(answer_source_label),
                "legal_target_id": str(answer_target_id),
                "legal_target_label": str(answer_target_label),
                "legal_move_answer": f"{str(answer_source_label)} -> {str(answer_target_label)}",
                "move_options": [
                    {
                        "label": str(option.label),
                        "source_card_id": str(option.source_card_id),
                        "source_label": str(option.source_label),
                        "target_id": str(option.target_id),
                        "target_label": str(option.target_label),
                        "move": str(option.move_text),
                        "is_answer": bool(option.is_answer),
                    }
                    for option in options
                ],
            },
        )
    raise ValueError("failed to sample solitaire move-legality scene")


def sample_foundation_ready(
    rng,
    *,
    namespace: str,
    instance_seed: int,
    params: Mapping[str, Any],
    scene_variant: str,
) -> SolitaireSample:
    """Construct exposed tableau cards with a requested count of legal foundation moves."""

    target_answer, target_probabilities = sample_integer_axis(
        namespace=str(namespace),
        instance_seed=int(instance_seed),
        params=params,
        support_key="foundation_ready_target_answer_support",
        explicit_key="target_answer",
        fallback_support=DEFAULTS.foundation_ready_target_answer_support,
        axis_name="foundation_ready_target_answer",
        balanced_flag_key="balanced_target_answer_sampling",
    )
    for _attempt in range(500):
        foundations = sample_foundations(rng)
        pool = deck_after_foundations(foundations)
        ready_raw = [
            (int(foundation.top_rank_value) + 1, str(foundation.suit_name))
            for foundation in foundations
            if int(foundation.top_rank_value) < 13 and (int(foundation.top_rank_value) + 1, str(foundation.suit_name)) in pool
        ]
        if len(ready_raw) < int(target_answer):
            continue
        rng.shuffle(ready_raw)
        selected_ready = ready_raw[: int(target_answer)]
        exposed_raw: List[Tuple[int, str]] = []
        for raw in selected_ready:
            remove_card(pool, raw)
            exposed_raw.append(raw)
        non_ready_pool = []
        foundation_by_suit = {str(f.suit_name): f for f in foundations}
        for raw in list(pool):
            rank, suit = raw
            candidate_card = Card(card_id="candidate", rank_value=int(rank), suit_name=str(suit))
            if not is_legal_foundation_move(candidate_card, foundation_by_suit[str(suit)]):
                non_ready_pool.append(raw)
        column_count = 7 if str(scene_variant) == "klondike_tableau" else 8
        if len(non_ready_pool) < int(column_count) - len(exposed_raw):
            continue
        rng.shuffle(non_ready_pool)
        for raw in non_ready_pool[: int(column_count) - len(exposed_raw)]:
            remove_card(pool, raw)
            exposed_raw.append(raw)
        rng.shuffle(exposed_raw)
        exposed_cards = tuple(
            Card(
                card_id=f"exposed_{index + 1:02d}",
                rank_value=int(rank),
                suit_name=str(suit),
                badge_text=None,
            )
            for index, (rank, suit) in enumerate(exposed_raw)
        )
        columns = build_columns_from_exposed(rng, exposed_cards=exposed_cards, pool=pool, scene_variant=str(scene_variant))
        foundation_by_suit = {str(f.suit_name): f for f in foundations}
        ready_ids = tuple(
            str(card.card_id)
            for card in visible_exposed_cards(columns)
            if is_legal_foundation_move(card, foundation_by_suit[str(card.suit_name)])
        )
        if len(ready_ids) != int(target_answer):
            continue
        return SolitaireSample(
            scene_variant=str(scene_variant),
            columns=tuple(columns),
            foundations=tuple(foundations),
            answer=int(target_answer),
            answer_type="integer",
            annotation_entity_ids=tuple(ready_ids),
            move_options=(),
            metadata={
                "target_answer": int(target_answer),
                "target_answer_probabilities": dict(target_probabilities),
                "ready_card_ids": list(ready_ids),
                "ready_card_column_labels": [card_column_label(columns, card_id) for card_id in ready_ids],
            },
        )
    raise ValueError("failed to sample solitaire foundation-ready scene")


def sample_column_card_count(
    rng,
    *,
    namespace: str,
    instance_seed: int,
    params: Mapping[str, Any],
    scene_variant: str,
) -> SolitaireSample:
    """Construct a tableau and count visible cards in one numbered column."""

    target_answer, target_probabilities = sample_integer_axis(
        namespace=str(namespace),
        instance_seed=int(instance_seed),
        params=params,
        support_key="column_card_count_target_answer_support",
        explicit_key="target_answer",
        fallback_support=DEFAULTS.column_card_count_target_answer_support,
        axis_name="column_card_count_target_answer",
        balanced_flag_key="balanced_target_answer_sampling",
    )
    column_count = 7 if str(scene_variant) == "klondike_tableau" else 8
    explicit_column = params.get("target_column")
    if explicit_column is not None:
        target_col_index = int(explicit_column) - 1
        if target_col_index < 0 or target_col_index >= int(column_count):
            raise ValueError(f"target_column out of range for solitaire scene: {explicit_column}")
    else:
        target_col_index = int(rng.randrange(column_count))
    for _attempt in range(300):
        pool = deck()
        columns: List[Tuple[Card, ...]] = []
        for col_index in range(column_count):
            if int(col_index) == int(target_col_index):
                length = int(target_answer)
            else:
                length = int(rng.randrange(1, 7))
            if len(pool) < int(length):
                raise ValueError("not enough cards to sample solitaire column-card count")
            cards: List[Card] = []
            for row_index in range(int(length)):
                raw = pool.pop(int(rng.randrange(len(pool))))
                cards.append(
                    Card(
                        card_id=f"col_{col_index + 1:02d}_card_{row_index + 1:02d}",
                        rank_value=int(raw[0]),
                        suit_name=str(raw[1]),
                        badge_text=None,
                    )
                )
            columns.append(tuple(cards))
        target_ids = tuple(str(card.card_id) for card in columns[int(target_col_index)])
        if len(target_ids) != int(target_answer):
            continue
        return SolitaireSample(
            scene_variant=str(scene_variant),
            columns=tuple(columns),
            foundations=sample_foundations(rng),
            answer=int(target_answer),
            answer_type="integer",
            annotation_entity_ids=tuple(target_ids),
            move_options=(),
            metadata={
                "target_answer": int(target_answer),
                "target_answer_probabilities": dict(target_probabilities),
                "target_column_index": int(target_col_index),
                "target_column_number": int(target_col_index) + 1,
                "target_column_card_ids": list(target_ids),
            },
        )
    raise ValueError("failed to sample solitaire column-card count scene")


def sample_tableau_movable_card_count(
    rng,
    *,
    namespace: str,
    instance_seed: int,
    params: Mapping[str, Any],
    scene_variant: str,
) -> SolitaireSample:
    """Construct a tableau with a controlled count of exposed cards that can move in tableau."""

    target_answer, target_probabilities = sample_integer_axis(
        namespace=str(namespace),
        instance_seed=int(instance_seed),
        params=params,
        support_key="tableau_movable_card_count_target_answer_support",
        explicit_key="target_answer",
        fallback_support=DEFAULTS.tableau_movable_card_count_target_answer_support,
        axis_name="tableau_movable_card_count_target_answer",
        balanced_flag_key="balanced_target_answer_sampling",
    )
    if int(target_answer) < 0 or int(target_answer) > 4:
        raise ValueError(f"unsupported solitaire tableau movable-card count: {target_answer}")

    column_count = 7 if str(scene_variant) == "klondike_tableau" else 8
    for _attempt in range(3000):
        pool = deck()
        rng.shuffle(pool)
        exposed_raw_or_empty: List[Tuple[int, str] | None] = [
            pool.pop() for _ in range(int(column_count))
        ]
        columns = build_columns_from_optional_exposed(
            rng,
            exposed_cards_or_empty=exposed_raw_or_empty,
            pool=pool,
            scene_variant=str(scene_variant),
        )
        exposed = visible_exposed_cards(columns)
        movable_ids = tuple(
            str(source.card_id)
            for source in exposed
            if any(
                str(target.card_id) != str(source.card_id)
                and is_legal_tableau_move(source, target)
                for target in exposed
            )
        )
        if len(movable_ids) != int(target_answer):
            continue
        return SolitaireSample(
            scene_variant=str(scene_variant),
            columns=tuple(columns),
            foundations=(),
            answer=int(target_answer),
            answer_type="integer",
            annotation_entity_ids=tuple(movable_ids),
            move_options=(),
            metadata={
                "target_answer": int(target_answer),
                "target_answer_probabilities": dict(target_probabilities),
                "movable_card_ids": list(movable_ids),
                "movable_card_column_labels": [card_column_label(columns, card_id) for card_id in movable_ids],
                "show_foundations": False,
            },
        )
    raise ValueError("failed to sample solitaire tableau movable-card count scene")
