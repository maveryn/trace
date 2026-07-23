"""Identity-free sampling and card-rule helpers for the cards scene."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .state import (
    CardInstance,
    CANDIDATE_LABELS,
    HAND_LABELS,
    PLAYER_LABELS,
    POKER_CATEGORY_LABEL_BY_KEY,
    POKER_CATEGORY_SCORE_BY_KEY,
    RANK_LABEL_BY_VALUE,
    RANK_VALUES,
    RuleSample,
    SUPPORTED_POKER_DRAW_TARGET_CATEGORIES,
    SUPPORTED_POKER_WINNING_CATEGORIES,
    SUPPORTED_TRICK_PLAY_TRUMP_MODES,
    SUIT_NAMES,
    SampledHand,
)

def _rank_multiplicities_without_exact_count(*, forbidden_count: int) -> Tuple[int, ...]:
    """Return possible per-rank card counts excluding one exact multiplicity."""

    return tuple(
        int(count)
        for count in range(0, len(SUIT_NAMES) + 1)
        if int(count) != int(forbidden_count)
    )


def _can_fill_rank_counts_without_exact_count(
    *,
    rank_count: int,
    total_count: int,
    forbidden_count: int,
) -> bool:
    """Return whether ranks can fill `total_count` cards without a forbidden exact count."""

    possible = {0}
    allowed_counts = _rank_multiplicities_without_exact_count(forbidden_count=int(forbidden_count))
    for _ in range(int(rank_count)):
        possible = {
            int(current + count)
            for current in possible
            for count in allowed_counts
            if int(current + count) <= int(total_count)
        }
    return int(total_count) in possible


def _can_fill_non_triple_rank_counts(*, rank_count: int, total_count: int) -> bool:
    """Return whether non-triple ranks can contribute `total_count` cards without exact triples."""

    return _can_fill_rank_counts_without_exact_count(
        rank_count=int(rank_count),
        total_count=int(total_count),
        forbidden_count=3,
    )


def feasible_card_count_support(
    *,
    hand_kind: str,
    target_answer: int,
    raw_support: Sequence[int],
) -> Tuple[int, ...]:
    """Return the subset of card-count support that can realize the active query."""

    feasible: List[int] = []
    for raw_value in raw_support:
        card_count = int(raw_value)
        if str(hand_kind) == "exact_triple":
            if int(card_count) < 3 * int(target_answer):
                continue
            if not _can_fill_non_triple_rank_counts(
                rank_count=int(len(RANK_VALUES) - int(target_answer)),
                total_count=int(card_count) - (3 * int(target_answer)),
            ):
                continue
        elif str(hand_kind) == "longest_run":
            if int(card_count) < int(target_answer):
                continue
        else:
            if int(card_count) < int(target_answer) + 1:
                continue
        feasible.append(int(card_count))
    return tuple(int(value) for value in feasible)


def make_deck() -> List[Tuple[int, str]]:
    """Return the canonical 52-card deck as `(rank_value, suit_name)` tuples."""

    return [(int(rank_value), str(suit_name)) for suit_name in SUIT_NAMES for rank_value in RANK_VALUES]


def _top_row_left_card_index(*, card_count: int, max_cards_per_row: int) -> int:
    """Return the zero-based index of the leftmost card in the renderer's top row."""

    max_per_row = int(max_cards_per_row)
    if max_per_row <= 0:
        raise ValueError("max_cards_per_row must be positive")
    return 0


def _label_card(
    *,
    card_id: str,
    rank_value: int,
    suit_name: str,
    is_reference: bool,
) -> CardInstance:
    """Build one rendered card payload from rank/suit metadata."""

    return CardInstance(
        card_id=str(card_id),
        rank_label=str(RANK_LABEL_BY_VALUE[int(rank_value)]),
        rank_value=int(rank_value),
        suit_name=str(suit_name),
        is_reference=bool(is_reference),
    )


def sample_same_suit_hand(
    rng,
    *,
    card_count: int,
    target_answer: int,
    order_by_suit: bool,
    reference_anchor_index: int,
) -> SampledHand:
    """Sample one hand with exactly `target_answer` cards matching the reference suit."""

    reference_suit = str(SUIT_NAMES[int(rng.randrange(len(SUIT_NAMES)))])
    reference_rank = int(RANK_VALUES[int(rng.randrange(len(RANK_VALUES)))])
    reference_card = (int(reference_rank), str(reference_suit))
    same_suit_pool = [(int(rank), str(reference_suit)) for rank in RANK_VALUES if int(rank) != int(reference_rank)]
    matching_cards = list(rng.sample(same_suit_pool, int(target_answer)))
    other_pool = [(int(rank), str(suit)) for rank, suit in make_deck() if str(suit) != str(reference_suit)]
    filler_cards = list(rng.sample(other_pool, int(card_count) - 1 - int(target_answer)))
    non_reference_cards = matching_cards + filler_cards
    if bool(order_by_suit):
        suit_order = {str(suit_name): int(index) for index, suit_name in enumerate(SUIT_NAMES)}
        non_reference_cards.sort(key=lambda item: (int(suit_order[str(item[1])]), int(item[0])))
    else:
        rng.shuffle(non_reference_cards)
    anchor_index = max(0, min(int(card_count) - 1, int(reference_anchor_index)))
    ordered = list(non_reference_cards)
    ordered.insert(int(anchor_index), reference_card)
    cards: List[CardInstance] = []
    annotation_card_ids: List[str] = []
    reference_card_id: str | None = None
    for index, (rank_value, suit_name) in enumerate(ordered, start=1):
        is_reference = bool((int(rank_value), str(suit_name)) == reference_card and reference_card_id is None)
        card_id = f"card_{index:02d}"
        cards.append(
            _label_card(
                card_id=str(card_id),
                rank_value=int(rank_value),
                suit_name=str(suit_name),
                is_reference=bool(is_reference),
            )
        )
        if bool(is_reference):
            reference_card_id = str(card_id)
        elif str(suit_name) == str(reference_suit):
            annotation_card_ids.append(str(card_id))
    if reference_card_id is None:
        raise ValueError("same-suit hand must contain one reference card")
    return SampledHand(
        cards=tuple(cards),
        annotation_card_ids=tuple(annotation_card_ids),
        reference_card_id=str(reference_card_id),
        reference_rank_value=int(reference_rank),
        reference_rank_label=str(RANK_LABEL_BY_VALUE[int(reference_rank)]),
        reference_suit_name=str(reference_suit),
        rank_sequence=tuple(int(card.rank_value) for card in cards),
    )


def sample_higher_rank_hand(
    rng,
    *,
    card_count: int,
    target_answer: int,
    order_by_rank: bool,
    reference_anchor_index: int,
) -> SampledHand:
    """Sample one hand with exactly `target_answer` cards ranked above the reference card."""

    feasible_reference_ranks: List[int] = []
    required_not_higher = int(card_count) - 1 - int(target_answer)
    for rank_value in RANK_VALUES:
        higher_count = sum(1 for candidate_rank, _ in make_deck() if int(candidate_rank) > int(rank_value))
        not_higher_count = sum(
            1
            for candidate_rank, _ in make_deck()
            if int(candidate_rank) < int(rank_value)
        ) + 3
        if int(higher_count) >= int(target_answer) and int(not_higher_count) >= int(required_not_higher):
            feasible_reference_ranks.append(int(rank_value))
    if not feasible_reference_ranks:
        raise ValueError("no feasible reference rank for higher-than query")
    reference_rank = int(feasible_reference_ranks[int(rng.randrange(len(feasible_reference_ranks)))])
    reference_suit = str(SUIT_NAMES[int(rng.randrange(len(SUIT_NAMES)))])
    reference_card = (int(reference_rank), str(reference_suit))
    higher_pool = [(int(rank), str(suit)) for rank, suit in make_deck() if int(rank) > int(reference_rank)]
    lower_equal_pool = [
        (int(rank), str(suit))
        for rank, suit in make_deck()
        if (int(rank), str(suit)) != reference_card and int(rank) <= int(reference_rank)
    ]
    higher_cards = list(rng.sample(higher_pool, int(target_answer)))
    lower_equal_cards = list(rng.sample(lower_equal_pool, int(required_not_higher)))
    non_reference_cards = higher_cards + lower_equal_cards
    if bool(order_by_rank):
        suit_order = {str(suit_name): int(index) for index, suit_name in enumerate(SUIT_NAMES)}
        non_reference_cards.sort(key=lambda item: (int(item[0]), int(suit_order[str(item[1])])))
    else:
        rng.shuffle(non_reference_cards)
    anchor_index = max(0, min(int(card_count) - 1, int(reference_anchor_index)))
    ordered = list(non_reference_cards)
    ordered.insert(int(anchor_index), reference_card)
    cards: List[CardInstance] = []
    annotation_card_ids: List[str] = []
    reference_card_id: str | None = None
    for index, (rank_value, suit_name) in enumerate(ordered, start=1):
        is_reference = bool((int(rank_value), str(suit_name)) == reference_card and reference_card_id is None)
        card_id = f"card_{index:02d}"
        cards.append(
            _label_card(
                card_id=str(card_id),
                rank_value=int(rank_value),
                suit_name=str(suit_name),
                is_reference=bool(is_reference),
            )
        )
        if bool(is_reference):
            reference_card_id = str(card_id)
        elif int(rank_value) > int(reference_rank):
            annotation_card_ids.append(str(card_id))
    if reference_card_id is None:
        raise ValueError("higher-rank hand must contain one reference card")
    return SampledHand(
        cards=tuple(cards),
        annotation_card_ids=tuple(annotation_card_ids),
        reference_card_id=str(reference_card_id),
        reference_rank_value=int(reference_rank),
        reference_rank_label=str(RANK_LABEL_BY_VALUE[int(reference_rank)]),
        reference_suit_name=str(reference_suit),
        rank_sequence=tuple(int(card.rank_value) for card in cards),
    )


def sample_exact_triple_count_hand(
    rng,
    *,
    card_count: int,
    target_answer: int,
    order_by_rank: bool,
) -> SampledHand:
    """Sample one hand with exactly `target_answer` distinct exact triples."""

    triple_rank_count = int(target_answer)
    if int(card_count) < 3 * int(triple_rank_count):
        raise ValueError("exact-triple hand requires enough cards to realize the requested triples")
    triple_ranks = list(rng.sample(list(RANK_VALUES), int(triple_rank_count)))
    remaining_ranks = [
        int(rank_value)
        for rank_value in RANK_VALUES
        if int(rank_value) not in set(triple_ranks)
    ]
    filler_count = int(card_count) - (3 * int(triple_rank_count))
    filler_rank_counts = _sample_non_triple_rank_counts(
        rng,
        ranks=remaining_ranks,
        total_count=int(filler_count),
    )

    raw_cards: List[Tuple[int, str, bool]] = []
    for rank_value in triple_ranks:
        suits = list(rng.sample(list(SUIT_NAMES), 3))
        for suit_name in suits:
            raw_cards.append((int(rank_value), str(suit_name), True))
    for rank_value, count in filler_rank_counts:
        suits = list(rng.sample(list(SUIT_NAMES), int(count)))
        for suit_name in suits:
            raw_cards.append((int(rank_value), str(suit_name), False))
    if bool(order_by_rank):
        suit_order = {str(suit_name): int(index) for index, suit_name in enumerate(SUIT_NAMES)}
        raw_cards.sort(key=lambda item: (int(item[0]), int(suit_order[str(item[1])])))
    else:
        rng.shuffle(raw_cards)

    cards: List[CardInstance] = []
    annotation_card_ids: List[str] = []
    keyed_annotation_ids: Dict[str, List[str]] = {}
    for index, (rank_value, suit_name, in_triple) in enumerate(raw_cards, start=1):
        card_id = f"card_{index:02d}"
        cards.append(
            _label_card(
                card_id=str(card_id),
                rank_value=int(rank_value),
                suit_name=str(suit_name),
                is_reference=False,
            )
        )
        if bool(in_triple):
            annotation_card_ids.append(str(card_id))
            rank_label = str(RANK_LABEL_BY_VALUE[int(rank_value)])
            keyed_annotation_ids.setdefault(str(rank_label), []).append(str(card_id))
    rank_order = {str(RANK_LABEL_BY_VALUE[int(rank_value)]): int(rank_value) for rank_value in RANK_VALUES}
    return SampledHand(
        cards=tuple(cards),
        annotation_card_ids=tuple(annotation_card_ids),
        reference_card_id=None,
        reference_rank_value=None,
        reference_rank_label=None,
        reference_suit_name=None,
        rank_sequence=tuple(int(card.rank_value) for card in cards),
        keyed_annotation_card_ids=tuple(
            (str(rank_label), tuple(str(card_id) for card_id in card_ids))
            for rank_label, card_ids in sorted(
                keyed_annotation_ids.items(),
                key=lambda item: int(rank_order[str(item[0])]),
            )
        ),
    )


def _sample_non_triple_rank_counts(
    rng,
    *,
    ranks: Sequence[int],
    total_count: int,
) -> Tuple[Tuple[int, int], ...]:
    """Sample per-rank card counts that never create additional exact triples."""

    return _sample_rank_counts_without_exact_count(
        rng,
        ranks=ranks,
        total_count=int(total_count),
        forbidden_count=3,
    )


def _sample_rank_counts_without_exact_count(
    rng,
    *,
    ranks: Sequence[int],
    total_count: int,
    forbidden_count: int,
) -> Tuple[Tuple[int, int], ...]:
    """Sample per-rank card counts while excluding one exact multiplicity."""

    shuffled_ranks = [int(rank_value) for rank_value in ranks]
    rng.shuffle(shuffled_ranks)
    allowed_counts = _rank_multiplicities_without_exact_count(forbidden_count=int(forbidden_count))

    feasible_cache: Dict[Tuple[int, int], bool] = {}

    def feasible(index: int, remaining: int) -> bool:
        key = (int(index), int(remaining))
        if key in feasible_cache:
            return bool(feasible_cache[key])
        if int(remaining) < 0:
            feasible_cache[key] = False
            return False
        if int(index) >= len(shuffled_ranks):
            feasible_cache[key] = int(remaining) == 0
            return bool(feasible_cache[key])
        feasible_cache[key] = any(
            feasible(int(index) + 1, int(remaining) - int(count))
            for count in allowed_counts
        )
        return bool(feasible_cache[key])

    if not feasible(0, int(total_count)):
        raise ValueError("hand cannot fill remaining cards without creating extra exact rank counts")

    out: List[Tuple[int, int]] = []
    remaining = int(total_count)
    for index, rank_value in enumerate(shuffled_ranks):
        options = [
            int(count)
            for count in allowed_counts
            if feasible(int(index) + 1, int(remaining) - int(count))
        ]
        chosen = int(options[int(rng.randrange(len(options)))])
        if int(chosen) > 0:
            out.append((int(rank_value), int(chosen)))
        remaining -= int(chosen)
    return tuple(out)


def _sample_non_run_cards(
    rng,
    *,
    count: int,
    blocked_ranks: Sequence[int],
    used_cards: set[Tuple[int, str]],
    initial_previous_rank: int | None = None,
    forbidden_final_rank: int | None = None,
) -> List[Tuple[int, str]]:
    """Sample `count` cards that avoid forming ascending adjacent runs."""

    safe_cards = [
        (int(rank_value), str(suit_name))
        for rank_value, suit_name in make_deck()
        if int(rank_value) not in {int(value) for value in blocked_ranks}
    ]
    out: List[Tuple[int, str]] = []
    for offset in range(int(count)):
        previous_rank = int(initial_previous_rank) if not out and initial_previous_rank is not None else None if not out else int(out[-1][0])
        options = [
            (int(rank_value), str(suit_name))
            for rank_value, suit_name in safe_cards
            if (int(rank_value), str(suit_name)) not in used_cards
            and (previous_rank is None or int(rank_value) != int(previous_rank + 1))
            and (
                int(offset) != int(count) - 1
                or forbidden_final_rank is None
                or int(rank_value) != int(forbidden_final_rank)
            )
        ]
        if not options:
            raise ValueError("run sampler exhausted safe non-run cards")
        chosen = options[int(rng.randrange(len(options)))]
        out.append((int(chosen[0]), str(chosen[1])))
        used_cards.add((int(chosen[0]), str(chosen[1])))
    return out


def sample_longest_run_hand(rng, *, card_count: int, target_answer: int) -> SampledHand:
    """Sample one hand with a unique longest contiguous ascending run."""

    run_length = int(target_answer)
    if int(card_count) < int(run_length):
        raise ValueError("run hand requires at least as many cards as the requested run length")
    run_start_rank = int(rng.randrange(2, 15 - int(run_length)))
    run_ranks = [int(run_start_rank + offset) for offset in range(int(run_length))]
    run_start_index = int(rng.randrange(int(card_count) - int(run_length) + 1))

    used_cards: set[Tuple[int, str]] = set()
    before_cards = _sample_non_run_cards(
        rng,
        count=int(run_start_index),
        blocked_ranks=(),
        used_cards=used_cards,
        forbidden_final_rank=(int(run_start_rank) - 1 if int(run_start_rank) > 2 else None),
    )
    run_cards: List[Tuple[int, str]] = []
    for rank_value in run_ranks:
        options = [(int(rank_value), str(suit_name)) for suit_name in SUIT_NAMES if (int(rank_value), str(suit_name)) not in used_cards]
        chosen = options[int(rng.randrange(len(options)))]
        run_cards.append(chosen)
        used_cards.add(chosen)
    after_cards = _sample_non_run_cards(
        rng,
        count=int(card_count) - int(run_start_index) - int(run_length),
        blocked_ranks=(),
        used_cards=used_cards,
        initial_previous_rank=int(run_start_rank + run_length - 1),
    )
    ordered = before_cards + run_cards + after_cards
    if _unique_longest_run_span(tuple(int(rank_value) for rank_value, _ in ordered)) != (
        int(run_start_index),
        int(run_start_index + run_length - 1),
    ):
        raise ValueError("run sampler failed to preserve a unique target-length run")
    cards: List[CardInstance] = []
    annotation_card_ids: List[str] = []
    for index, (rank_value, suit_name) in enumerate(ordered, start=1):
        card_id = f"card_{index:02d}"
        cards.append(
            _label_card(
                card_id=str(card_id),
                rank_value=int(rank_value),
                suit_name=str(suit_name),
                is_reference=False,
            )
        )
        if int(run_start_index) < int(index) <= int(run_start_index + run_length):
            annotation_card_ids.append(str(card_id))
    return SampledHand(
        cards=tuple(cards),
        annotation_card_ids=tuple(annotation_card_ids),
        reference_card_id=None,
        reference_rank_value=None,
        reference_rank_label=None,
        reference_suit_name=None,
        rank_sequence=tuple(int(card.rank_value) for card in cards),
    )


def _unique_longest_run_span(rank_sequence: Sequence[int]) -> Tuple[int, int] | None:
    """Return the only longest ascending-by-one span, or None when tied."""

    best_spans: List[Tuple[int, int]] = []
    start_index = 0
    for index in range(1, len(rank_sequence) + 1):
        continues = (
            index < len(rank_sequence)
            and int(rank_sequence[index]) == int(rank_sequence[index - 1]) + 1
        )
        if continues:
            continue
        end_index = int(index - 1)
        run_length = int(end_index - start_index + 1)
        if not best_spans or run_length > int(best_spans[0][1] - best_spans[0][0] + 1):
            best_spans = [(int(start_index), int(end_index))]
        elif run_length == int(best_spans[0][1] - best_spans[0][0] + 1):
            best_spans.append((int(start_index), int(end_index)))
        start_index = int(index)
    if len(best_spans) != 1:
        return None
    return best_spans[0]



def _option_letter(label: str) -> str:
    """Return the compact answer option from a rendered label such as `Hand C`."""

    return str(label).strip().split()[-1]

def _make_labelled_card(
    *,
    card_id: str,
    rank_value: int,
    suit_name: str,
    badge_text: str | None = None,
    group_label: str | None = None,
    is_reference: bool = False,
) -> CardInstance:
    """Build one rendered card with optional game-rule labels."""

    return CardInstance(
        card_id=str(card_id),
        rank_label=str(RANK_LABEL_BY_VALUE[int(rank_value)]),
        rank_value=int(rank_value),
        suit_name=str(suit_name),
        is_reference=bool(is_reference),
        badge_text=None if badge_text is None else str(badge_text),
        group_label=None if group_label is None else str(group_label),
    )


def _blackjack_total(cards: Sequence[Tuple[int, str]]) -> int:
    """Return the best blackjack total, counting aces as 11 or 1."""

    total = 0
    ace_count = 0
    for rank_value, _suit_name in cards:
        rank = int(rank_value)
        if int(rank) == 14:
            total += 11
            ace_count += 1
        elif int(rank) >= 11:
            total += 10
        else:
            total += int(rank)
    while int(total) > 21 and int(ace_count) > 0:
        total -= 10
        ace_count -= 1
    return int(total)


def sample_blackjack_best_hand(
    rng,
    *,
    hand_count: int,
    cards_per_hand: int,
    hand_count_support: Sequence[int] = (),
    hand_count_probabilities: Mapping[str, float] | None = None,
    cards_per_hand_support: Sequence[int] = (),
    cards_per_hand_probabilities: Mapping[str, float] | None = None,
) -> RuleSample:
    """Sample labelled blackjack hands with one unique best non-bust hand."""

    hand_count_probabilities = dict(hand_count_probabilities or {})
    cards_per_hand_probabilities = dict(cards_per_hand_probabilities or {})
    labels = HAND_LABELS[: int(hand_count)]
    deck = make_deck()
    for _attempt in range(200):
        raw_cards = list(rng.sample(deck, int(hand_count) * int(cards_per_hand)))
        hands = [
            raw_cards[index * int(cards_per_hand) : (index + 1) * int(cards_per_hand)]
            for index in range(int(hand_count))
        ]
        totals = [_blackjack_total(hand) for hand in hands]
        playable_scores = [int(total) if int(total) <= 21 else -1 for total in totals]
        best_score = max(playable_scores)
        if int(best_score) < 0 or playable_scores.count(int(best_score)) != 1:
            continue
        winner_index = int(playable_scores.index(int(best_score)))
        winner_label = str(labels[int(winner_index)])
        winner_option = _option_letter(winner_label)
        cards: List[CardInstance] = []
        annotation_ids: List[str] = []
        for hand_index, hand in enumerate(hands):
            label = str(labels[int(hand_index)])
            for card_index, (rank_value, suit_name) in enumerate(hand, start=1):
                card_id = f"hand_{hand_index + 1:02d}_card_{card_index:02d}"
                cards.append(
                    _make_labelled_card(
                        card_id=str(card_id),
                        rank_value=int(rank_value),
                        suit_name=str(suit_name),
                        group_label=str(label),
                    )
                )
                if int(hand_index) == int(winner_index):
                    annotation_ids.append(str(card_id))
        return RuleSample(
            pattern_kind="blackjack_best",
            scene_variant="blackjack_multi_hand",
            cards=tuple(cards),
            answer=str(winner_option),
            annotation_card_ids=tuple(annotation_ids),
            option_count=int(hand_count),
            cards_per_row=int(cards_per_hand),
            center_label_mode="rank_suit",
            render_overrides={
                "canvas_height": 880,
                "card_width_px": 78,
                "card_height_px": 108,
                "card_gap_px": 12,
                "row_gap_px": 18,
                "rank_font_size_px": 17,
                "center_symbol_font_size_px": 38,
                "max_cards_per_row": int(cards_per_hand),
            },
            prompt_slots={},
            metadata={
                "hand_labels": list(labels),
                "hand_totals": {str(labels[index]): int(total) for index, total in enumerate(totals)},
                "playable_scores": {str(labels[index]): int(score) for index, score in enumerate(playable_scores)},
                "winning_label": str(winner_label),
                "winning_option": str(winner_option),
                "hand_count_support": [int(value) for value in hand_count_support],
                "hand_count_probabilities": dict(hand_count_probabilities),
                "cards_per_hand_support": [int(value) for value in cards_per_hand_support],
                "cards_per_hand_probabilities": dict(cards_per_hand_probabilities),
            },
        )
    raise ValueError("failed to sample unique blackjack best hand")


def _straight_high_rank(ranks: Sequence[int]) -> int | None:
    """Return the high card of a five-card straight, with wheel as five-high."""

    unique = sorted({int(rank) for rank in ranks})
    if len(unique) != 5:
        return None
    if unique == [2, 3, 4, 5, 14]:
        return 5
    if int(unique[-1] - unique[0]) == 4:
        return int(unique[-1])
    return None


def poker_score(cards: Sequence[Tuple[int, str]]) -> Tuple[int, Tuple[int, ...], str]:
    """Return a comparable standard five-card poker score."""

    ranks = [int(rank) for rank, _suit_name in cards]
    suits = [str(suit_name) for _rank, suit_name in cards]
    rank_counts: Dict[int, int] = {}
    for rank in ranks:
        rank_counts[int(rank)] = int(rank_counts.get(int(rank), 0) + 1)
    groups = sorted(rank_counts.items(), key=lambda item: (int(item[1]), int(item[0])), reverse=True)
    flush = len(set(suits)) == 1
    straight_high = _straight_high_rank(ranks)
    if flush and straight_high is not None:
        return (8, (int(straight_high),), "straight flush")
    if int(groups[0][1]) == 4:
        quad_rank = int(groups[0][0])
        kicker = max(int(rank) for rank in ranks if int(rank) != int(quad_rank))
        return (7, (int(quad_rank), int(kicker)), "four of a kind")
    if sorted((int(count) for count in rank_counts.values()), reverse=True) == [3, 2]:
        triple_rank = max(int(rank) for rank, count in rank_counts.items() if int(count) == 3)
        pair_rank = max(int(rank) for rank, count in rank_counts.items() if int(count) == 2)
        return (6, (int(triple_rank), int(pair_rank)), "full house")
    if flush:
        return (5, tuple(sorted((int(rank) for rank in ranks), reverse=True)), "flush")
    if straight_high is not None:
        return (4, (int(straight_high),), "straight")
    if int(groups[0][1]) == 3:
        triple_rank = int(groups[0][0])
        kickers = tuple(sorted((int(rank) for rank in ranks if int(rank) != int(triple_rank)), reverse=True))
        return (3, (int(triple_rank), *kickers), "three of a kind")
    pair_ranks = sorted((int(rank) for rank, count in rank_counts.items() if int(count) == 2), reverse=True)
    if len(pair_ranks) == 2:
        kicker = max(int(rank) for rank in ranks if int(rank) not in set(pair_ranks))
        return (2, (int(pair_ranks[0]), int(pair_ranks[1]), int(kicker)), "two pair")
    if len(pair_ranks) == 1:
        pair_rank = int(pair_ranks[0])
        kickers = tuple(sorted((int(rank) for rank in ranks if int(rank) != int(pair_rank)), reverse=True))
        return (1, (int(pair_rank), *kickers), "one pair")
    return (0, tuple(sorted((int(rank) for rank in ranks), reverse=True)), "high card")


def _straight_rank_sequences() -> Tuple[Tuple[int, ...], ...]:
    """Return five-rank straight sequences using Ace high and wheel forms."""

    return ((14, 2, 3, 4, 5),) + tuple(tuple(range(start, start + 5)) for start in range(2, 11))


def _cards_available(cards: Sequence[Tuple[int, str]], used_cards: set[Tuple[int, str]]) -> bool:
    """Return true when all cards are unused."""

    return all((int(rank), str(suit)) not in used_cards for rank, suit in cards)


def _sample_poker_hand_for_category(
    rng,
    *,
    category_key: str,
    used_cards: set[Tuple[int, str]],
) -> Tuple[Tuple[int, str], ...]:
    """Sample one unused five-card hand for an exact poker category."""

    category = str(category_key)
    if category not in POKER_CATEGORY_SCORE_BY_KEY:
        raise ValueError(f"unsupported poker category: {category}")

    for _attempt in range(1000):
        hand: List[Tuple[int, str]]
        if category == "high_card":
            available = [card for card in make_deck() if card not in used_cards]
            if len(available) < 5:
                raise ValueError("not enough cards remain for high-card hand")
            hand = list(rng.sample(available, 5))
        elif category == "one_pair":
            pair_rank = int(rng.choice(RANK_VALUES))
            pair_suits = list(rng.sample(list(SUIT_NAMES), 2))
            kicker_ranks = list(rng.sample([rank for rank in RANK_VALUES if int(rank) != int(pair_rank)], 3))
            hand = [(int(pair_rank), str(suit)) for suit in pair_suits]
            hand.extend((int(rank), str(SUIT_NAMES[int(rng.randrange(len(SUIT_NAMES)))])) for rank in kicker_ranks)
        elif category == "two_pair":
            pair_ranks = list(rng.sample(list(RANK_VALUES), 2))
            kicker_rank = int(rng.choice([rank for rank in RANK_VALUES if int(rank) not in set(pair_ranks)]))
            hand = []
            for pair_rank in pair_ranks:
                hand.extend((int(pair_rank), str(suit)) for suit in rng.sample(list(SUIT_NAMES), 2))
            hand.append((int(kicker_rank), str(SUIT_NAMES[int(rng.randrange(len(SUIT_NAMES)))])))
        elif category == "three_of_a_kind":
            triple_rank = int(rng.choice(RANK_VALUES))
            kicker_ranks = list(rng.sample([rank for rank in RANK_VALUES if int(rank) != int(triple_rank)], 2))
            hand = [(int(triple_rank), str(suit)) for suit in rng.sample(list(SUIT_NAMES), 3)]
            hand.extend((int(rank), str(SUIT_NAMES[int(rng.randrange(len(SUIT_NAMES)))])) for rank in kicker_ranks)
        elif category == "straight":
            ranks = list(_straight_rank_sequences()[int(rng.randrange(len(_straight_rank_sequences())))])
            suits = [str(SUIT_NAMES[int(rng.randrange(len(SUIT_NAMES)))]) for _rank in ranks]
            if len(set(suits)) == 1:
                continue
            hand = [(int(rank), str(suit)) for rank, suit in zip(ranks, suits)]
        elif category == "flush":
            suit = str(SUIT_NAMES[int(rng.randrange(len(SUIT_NAMES)))])
            ranks = list(rng.sample(list(RANK_VALUES), 5))
            hand = [(int(rank), str(suit)) for rank in ranks]
        elif category == "full_house":
            triple_rank, pair_rank = [int(rank) for rank in rng.sample(list(RANK_VALUES), 2)]
            hand = [(int(triple_rank), str(suit)) for suit in rng.sample(list(SUIT_NAMES), 3)]
            hand.extend((int(pair_rank), str(suit)) for suit in rng.sample(list(SUIT_NAMES), 2))
        elif category == "four_of_a_kind":
            quad_rank = int(rng.choice(RANK_VALUES))
            kicker_rank = int(rng.choice([rank for rank in RANK_VALUES if int(rank) != int(quad_rank)]))
            hand = [(int(quad_rank), str(suit)) for suit in SUIT_NAMES]
            hand.append((int(kicker_rank), str(SUIT_NAMES[int(rng.randrange(len(SUIT_NAMES)))])))
        else:
            suit = str(SUIT_NAMES[int(rng.randrange(len(SUIT_NAMES)))])
            ranks = list(_straight_rank_sequences()[int(rng.randrange(len(_straight_rank_sequences())))])
            hand = [(int(rank), str(suit)) for rank in ranks]

        if len(set(hand)) != 5 or not _cards_available(hand, used_cards):
            continue
        score = poker_score(hand)
        if int(score[0]) != int(POKER_CATEGORY_SCORE_BY_KEY[category]):
            continue
        for card in hand:
            used_cards.add((int(card[0]), str(card[1])))
        rng.shuffle(hand)
        return tuple((int(rank), str(suit)) for rank, suit in hand)
    raise ValueError(f"failed to sample unused poker hand for category: {category}")


def sample_poker_best_hand(
    rng,
    *,
    hand_count: int,
    winning_category_key: str,
    hand_count_support: Sequence[int] = (),
    hand_count_probabilities: Mapping[str, float] | None = None,
    winning_category_probabilities: Mapping[str, float] | None = None,
) -> RuleSample:
    """Sample labelled five-card poker hands with a controlled unique winner category."""

    hand_count_probabilities = dict(hand_count_probabilities or {})
    winning_category_probabilities = dict(winning_category_probabilities or {})
    labels = HAND_LABELS[: int(hand_count)]
    target_score_category = int(POKER_CATEGORY_SCORE_BY_KEY[str(winning_category_key)])
    lower_category_keys = [
        str(key)
        for key, score in POKER_CATEGORY_SCORE_BY_KEY.items()
        if int(score) < int(target_score_category)
    ]
    if not lower_category_keys:
        raise ValueError("poker winning category must have at least one lower distractor category")
    for _attempt in range(500):
        winner_index = int(rng.randrange(int(hand_count)))
        used_cards: set[Tuple[int, str]] = set()
        hands_by_index: Dict[int, Tuple[Tuple[int, str], ...]] = {}
        hand_categories_by_index: Dict[int, str] = {}
        try:
            for hand_index in range(int(hand_count)):
                if int(hand_index) == int(winner_index):
                    category_key = str(winning_category_key)
                else:
                    category_key = str(lower_category_keys[int(rng.randrange(len(lower_category_keys)))])
                hand = _sample_poker_hand_for_category(
                    rng,
                    category_key=str(category_key),
                    used_cards=used_cards,
                )
                hands_by_index[int(hand_index)] = tuple(hand)
                hand_categories_by_index[int(hand_index)] = str(category_key)
        except ValueError:
            continue
        hands = [list(hands_by_index[index]) for index in range(int(hand_count))]
        scores = [poker_score(hand) for hand in hands]
        comparable = [(int(category), tuple(int(value) for value in tiebreakers)) for category, tiebreakers, _name in scores]
        best_score = max(comparable)
        if comparable.count(best_score) != 1:
            continue
        if int(comparable.index(best_score)) != int(winner_index):
            continue
        if int(scores[int(winner_index)][0]) != int(target_score_category):
            continue
        winner_label = str(labels[int(winner_index)])
        winner_option = _option_letter(winner_label)
        cards: List[CardInstance] = []
        annotation_ids: List[str] = []
        for hand_index, hand in enumerate(hands):
            label = str(labels[int(hand_index)])
            for card_index, (rank_value, suit_name) in enumerate(hand, start=1):
                card_id = f"hand_{hand_index + 1:02d}_card_{card_index:02d}"
                cards.append(
                    _make_labelled_card(
                        card_id=str(card_id),
                        rank_value=int(rank_value),
                        suit_name=str(suit_name),
                        group_label=str(label),
                    )
                )
                if int(hand_index) == int(winner_index):
                    annotation_ids.append(str(card_id))
        return RuleSample(
            pattern_kind="poker_best",
            scene_variant="poker_multi_hand",
            cards=tuple(cards),
            answer=str(winner_option),
            annotation_card_ids=tuple(annotation_ids),
            option_count=int(hand_count),
            cards_per_row=5,
            center_label_mode="rank_suit",
            render_overrides={
                "canvas_height": 880,
                "card_width_px": 78,
                "card_height_px": 108,
                "card_gap_px": 12,
                "row_gap_px": 18,
                "rank_font_size_px": 17,
                "center_symbol_font_size_px": 38,
                "max_cards_per_row": 5,
            },
            prompt_slots={},
            metadata={
                "hand_labels": list(labels),
                "hand_categories": {str(labels[index]): str(score[2]) for index, score in enumerate(scores)},
                "target_winning_category": str(POKER_CATEGORY_LABEL_BY_KEY[str(winning_category_key)]),
                "target_winning_category_key": str(winning_category_key),
                "hand_scores": {
                    str(labels[index]): [int(score[0]), [int(value) for value in score[1]]]
                    for index, score in enumerate(scores)
                },
                "winning_label": str(winner_label),
                "winning_option": str(winner_option),
                "winning_category": str(scores[int(winner_index)][2]),
                "winning_category_key": str(winning_category_key),
                "poker_winning_category_probabilities": dict(winning_category_probabilities),
                "hand_count_support": [int(value) for value in hand_count_support],
                "hand_count_probabilities": dict(hand_count_probabilities),
            },
        )
    raise ValueError("failed to sample unique poker best hand")


def poker_score_key(score: Tuple[int, Tuple[int, ...], str]) -> Tuple[int, Tuple[int, ...]]:
    """Return the comparable part of one poker score tuple."""

    return (int(score[0]), tuple(int(value) for value in score[1]))


def sample_poker_draw_card(
    rng,
    *,
    candidate_count: int,
    target_category_key: str,
    target_index: int,
    candidate_count_support: Sequence[int] = (),
    candidate_count_probabilities: Mapping[str, float] | None = None,
    target_category_probabilities: Mapping[str, float] | None = None,
) -> RuleSample:
    """Sample a four-card poker hand plus candidates with one strongest draw."""

    candidate_count_probabilities = dict(candidate_count_probabilities or {})
    target_category_probabilities = dict(target_category_probabilities or {})
    labels = CANDIDATE_LABELS[: int(candidate_count)]
    deck = make_deck()
    for _attempt in range(800):
        used_for_target: set[Tuple[int, str]] = set()
        try:
            target_hand = list(
                _sample_poker_hand_for_category(
                    rng,
                    category_key=str(target_category_key),
                    used_cards=used_for_target,
                )
            )
        except ValueError:
            continue
        correct_offset = int(rng.randrange(len(target_hand)))
        correct_card = target_hand[int(correct_offset)]
        partial_hand = [card for index, card in enumerate(target_hand) if int(index) != int(correct_offset)]
        partial_used = {(int(rank), str(suit)) for rank, suit in partial_hand}
        correct_score = poker_score([*partial_hand, correct_card])
        correct_key = poker_score_key(correct_score)
        lower_candidates: List[Tuple[int, str]] = []
        for card in deck:
            normalized = (int(card[0]), str(card[1]))
            if normalized in partial_used or normalized == (int(correct_card[0]), str(correct_card[1])):
                continue
            candidate_score = poker_score([*partial_hand, normalized])
            if poker_score_key(candidate_score) < correct_key:
                lower_candidates.append(normalized)
        if len(lower_candidates) < int(candidate_count) - 1:
            continue
        distractors = list(rng.sample(lower_candidates, int(candidate_count) - 1))
        candidate_cards = list(distractors)
        candidate_cards.insert(int(target_index), (int(correct_card[0]), str(correct_card[1])))
        completed_scores = [poker_score([*partial_hand, card]) for card in candidate_cards]
        comparable = [poker_score_key(score) for score in completed_scores]
        best_score = max(comparable)
        if comparable.count(best_score) != 1 or int(comparable.index(best_score)) != int(target_index):
            continue

        cards: List[CardInstance] = []
        partial_card_ids: List[str] = []
        for index, (rank_value, suit_name) in enumerate(partial_hand, start=1):
            card_id = f"partial_{index:02d}"
            partial_card_ids.append(str(card_id))
            cards.append(
                _make_labelled_card(
                    card_id=str(card_id),
                    rank_value=int(rank_value),
                    suit_name=str(suit_name),
                    group_label="Hand",
                )
            )

        candidate_ids_by_label: Dict[str, str] = {}
        candidate_specs_by_label: Dict[str, Dict[str, Any]] = {}
        for label, card, score in zip(labels, candidate_cards, completed_scores):
            rank_value, suit_name = card
            card_id = f"candidate_{str(label)}"
            candidate_ids_by_label[str(label)] = str(card_id)
            candidate_specs_by_label[str(label)] = {
                "card_id": str(card_id),
                "rank_value": int(rank_value),
                "rank_label": str(RANK_LABEL_BY_VALUE[int(rank_value)]),
                "suit_name": str(suit_name),
                "completed_score": [int(score[0]), [int(value) for value in score[1]]],
                "completed_category": str(score[2]),
                "is_best_completion": bool(str(label) == str(labels[int(target_index)])),
            }
            cards.append(
                _make_labelled_card(
                    card_id=str(card_id),
                    rank_value=int(rank_value),
                    suit_name=str(suit_name),
                    badge_text=str(label),
                    group_label="Candidates",
                )
            )

        answer_label = str(labels[int(target_index)])
        answer_card_id = str(candidate_ids_by_label[str(answer_label)])
        return RuleSample(
            pattern_kind="poker_draw",
            scene_variant="poker_draw_completion",
            cards=tuple(cards),
            answer=str(answer_label),
            annotation_card_ids=(str(answer_card_id),),
            option_count=int(candidate_count),
            cards_per_row=int(candidate_count),
            center_label_mode="rank_suit",
            render_overrides={
                "canvas_height": 560,
                "card_width_px": 86,
                "card_height_px": 120,
                "card_gap_px": 18,
                "row_gap_px": 54,
                "rank_font_size_px": 18,
                "center_symbol_font_size_px": 40,
                "reference_banner_height_px": 24,
                "reference_font_size_px": 16,
                "max_cards_per_row": int(candidate_count),
            },
            prompt_slots={},
            metadata={
                "partial_card_ids": [str(card_id) for card_id in partial_card_ids],
                "candidate_labels": [str(label) for label in labels],
                "candidate_card_ids_by_label": dict(candidate_ids_by_label),
                "candidate_specs_by_label": dict(candidate_specs_by_label),
                "correct_candidate_label": str(answer_label),
                "correct_candidate_card_id": str(answer_card_id),
                "target_candidate_index": int(target_index),
                "target_winning_category": str(POKER_CATEGORY_LABEL_BY_KEY[str(target_category_key)]),
                "target_winning_category_key": str(target_category_key),
                "winning_category": str(completed_scores[int(target_index)][2]),
                "winning_option": str(answer_label),
                "candidate_count_support": [int(value) for value in candidate_count_support],
                "candidate_count_probabilities": dict(candidate_count_probabilities),
                "poker_draw_target_category_probabilities": dict(target_category_probabilities),
            },
            row_card_counts=(4, int(candidate_count)),
        )
    raise ValueError("failed to sample unique poker draw card")


def _ace_high_straight_sequences() -> Tuple[Tuple[int, ...], ...]:
    """Return only Ace-high-compatible five-rank straight sequences."""

    return tuple(tuple(range(start, start + 5)) for start in range(2, 11))


def target_candidate_index(
    *,
    rng,
    params: Mapping[str, Any],
    candidate_count: int,
    cycle_divisor: int = 1,
) -> int:
    """Resolve the answer option index for a candidate-card scene."""

    raw_index = params.get("target_candidate_index")
    if raw_index is not None:
        index = int(raw_index)
        if index < 0 or index >= int(candidate_count):
            raise ValueError(f"target_candidate_index out of range: {index}")
        return int(index)
    if params.get("_sample_cursor") is not None:
        return int(abs(int(params["_sample_cursor"])) // max(1, int(cycle_divisor))) % int(candidate_count)
    return int(rng.randrange(int(candidate_count)))


def _completion_matches_pattern(
    *,
    pattern_kind: str,
    partial_hand: Sequence[Tuple[int, str]],
    candidate_card: Tuple[int, str],
) -> bool:
    """Return whether adding the candidate card completes the requested pattern."""

    completed = list(partial_hand) + [(int(candidate_card[0]), str(candidate_card[1]))]
    query = str(pattern_kind)
    if query == "missing_flush":
        return len({str(suit_name) for _rank, suit_name in completed}) == 1
    if query == "missing_straight":
        ranks = [int(rank_value) for rank_value, _suit_name in completed]
        return _straight_high_rank(ranks) is not None and set(ranks) in (
            set(sequence) for sequence in _ace_high_straight_sequences()
        )
    if query == "missing_full_house":
        return str(poker_score(completed)[2]) == "full house"
    if query == "missing_three_kind":
        return str(poker_score(completed)[2]) == "three of a kind"
    raise ValueError(f"unsupported missing-card query: {pattern_kind}")


def _missing_card_pattern_label(pattern_kind: str) -> str:
    """Return the human-readable pattern name for one missing-card query."""

    return {
        "missing_flush": "flush",
        "missing_straight": "straight",
        "missing_full_house": "full house",
        "missing_three_kind": "three of a kind",
    }[str(pattern_kind)]


def _sample_missing_card_base(
    rng,
    *,
    pattern_kind: str,
) -> Tuple[Tuple[Tuple[int, str], ...], Tuple[int, str]]:
    """Sample a partial four-card hand and the unique intended completion card."""

    query = str(pattern_kind)
    if query == "missing_flush":
        suit_name = str(SUIT_NAMES[int(rng.randrange(len(SUIT_NAMES)))])
        ranks = list(rng.sample(list(RANK_VALUES), 5))
        partial = tuple((int(rank), str(suit_name)) for rank in ranks[:4])
        correct = (int(ranks[4]), str(suit_name))
        return partial, correct

    if query == "missing_straight":
        sequence = list(_ace_high_straight_sequences()[int(rng.randrange(len(_ace_high_straight_sequences())))])
        missing_offset = int(rng.randrange(1, 4))
        missing_rank = int(sequence[int(missing_offset)])
        partial_ranks = [int(rank) for index, rank in enumerate(sequence) if int(index) != int(missing_offset)]
        partial = tuple(
            (int(rank), str(SUIT_NAMES[int(rng.randrange(len(SUIT_NAMES)))]))
            for rank in partial_ranks
        )
        used_suits_for_missing_rank = {
            str(suit_name)
            for rank_value, suit_name in partial
            if int(rank_value) == int(missing_rank)
        }
        suit_options = [str(suit) for suit in SUIT_NAMES if str(suit) not in used_suits_for_missing_rank]
        correct = (int(missing_rank), str(suit_options[int(rng.randrange(len(suit_options)))]))
        return partial, correct

    if query == "missing_full_house":
        triple_rank, pair_rank = [int(rank) for rank in rng.sample(list(RANK_VALUES), 2)]
        triple_cards = tuple((int(triple_rank), str(suit)) for suit in rng.sample(list(SUIT_NAMES), 3))
        pair_suits = list(rng.sample(list(SUIT_NAMES), 2))
        partial = tuple(triple_cards) + ((int(pair_rank), str(pair_suits[0])),)
        correct = (int(pair_rank), str(pair_suits[1]))
        return partial, correct

    if query == "missing_three_kind":
        triple_rank = int(rng.choice(RANK_VALUES))
        kicker_ranks = list(rng.sample([int(rank) for rank in RANK_VALUES if int(rank) != int(triple_rank)], 2))
        pair_suits = list(rng.sample(list(SUIT_NAMES), 3))
        partial = (
            (int(triple_rank), str(pair_suits[0])),
            (int(triple_rank), str(pair_suits[1])),
            (int(kicker_ranks[0]), str(SUIT_NAMES[int(rng.randrange(len(SUIT_NAMES)))])),
            (int(kicker_ranks[1]), str(SUIT_NAMES[int(rng.randrange(len(SUIT_NAMES)))])),
        )
        correct = (int(triple_rank), str(pair_suits[2]))
        return partial, correct

    raise ValueError(f"unsupported missing-card query: {pattern_kind}")


def sample_missing_card_to_complete_hand(
    rng,
    *,
    candidate_count: int,
    pattern_kind: str,
    target_index: int,
    candidate_count_support: Sequence[int] = (),
    candidate_count_probabilities: Mapping[str, float] | None = None,
    pattern_kind_probabilities: Mapping[str, float] | None = None,
) -> RuleSample:
    """Sample a partial hand and candidates for one requested completion pattern."""

    candidate_count_probabilities = dict(candidate_count_probabilities or {})
    pattern_kind_probabilities = dict(pattern_kind_probabilities or {})
    labels = CANDIDATE_LABELS[: int(candidate_count)]
    deck = make_deck()
    for _attempt in range(400):
        partial_hand, correct_card = _sample_missing_card_base(rng, pattern_kind=str(pattern_kind))
        used_cards = {(int(rank), str(suit)) for rank, suit in partial_hand}
        if (int(correct_card[0]), str(correct_card[1])) in used_cards:
            continue
        invalid_candidates = [
            (int(rank), str(suit))
            for rank, suit in deck
            if (int(rank), str(suit)) not in used_cards
            and (int(rank), str(suit)) != (int(correct_card[0]), str(correct_card[1]))
            and not _completion_matches_pattern(
                pattern_kind=str(pattern_kind),
                partial_hand=partial_hand,
                candidate_card=(int(rank), str(suit)),
            )
        ]
        if len(invalid_candidates) < int(candidate_count) - 1:
            continue
        distractors = list(rng.sample(invalid_candidates, int(candidate_count) - 1))
        candidate_cards = list(distractors)
        candidate_cards.insert(int(target_index), (int(correct_card[0]), str(correct_card[1])))
        completion_matches = [
            bool(
                _completion_matches_pattern(
                    pattern_kind=str(pattern_kind),
                    partial_hand=partial_hand,
                    candidate_card=(int(rank), str(suit)),
                )
            )
            for rank, suit in candidate_cards
        ]
        if completion_matches.count(True) != 1 or not bool(completion_matches[int(target_index)]):
            continue

        cards: List[CardInstance] = []
        partial_card_ids: List[str] = []
        for index, (rank_value, suit_name) in enumerate(partial_hand, start=1):
            card_id = f"partial_{index:02d}"
            partial_card_ids.append(str(card_id))
            cards.append(
                _make_labelled_card(
                    card_id=str(card_id),
                    rank_value=int(rank_value),
                    suit_name=str(suit_name),
                    group_label="Hand",
                )
            )

        candidate_ids_by_label: Dict[str, str] = {}
        candidate_specs_by_label: Dict[str, Dict[str, Any]] = {}
        for index, (label, card) in enumerate(zip(labels, candidate_cards), start=1):
            rank_value, suit_name = card
            card_id = f"candidate_{str(label)}"
            candidate_ids_by_label[str(label)] = str(card_id)
            candidate_specs_by_label[str(label)] = {
                "card_id": str(card_id),
                "rank_value": int(rank_value),
                "rank_label": str(RANK_LABEL_BY_VALUE[int(rank_value)]),
                "suit_name": str(suit_name),
                "completes_pattern": bool(completion_matches[int(index) - 1]),
            }
            cards.append(
                _make_labelled_card(
                    card_id=str(card_id),
                    rank_value=int(rank_value),
                    suit_name=str(suit_name),
                    badge_text=str(label),
                    group_label="Candidates",
                )
            )

        answer_label = str(labels[int(target_index)])
        answer_card_id = str(candidate_ids_by_label[str(answer_label)])
        return RuleSample(
            pattern_kind=str(pattern_kind),
            scene_variant="missing_card_completion",
            cards=tuple(cards),
            answer=str(answer_label),
            annotation_card_ids=(str(answer_card_id),),
            option_count=int(candidate_count),
            cards_per_row=int(candidate_count),
            center_label_mode="rank_suit",
            render_overrides={
                "canvas_height": 560,
                "card_width_px": 86,
                "card_height_px": 120,
                "card_gap_px": 18,
                "row_gap_px": 54,
                "rank_font_size_px": 18,
                "center_symbol_font_size_px": 40,
                "reference_banner_height_px": 24,
                "reference_font_size_px": 16,
                "max_cards_per_row": int(candidate_count),
            },
            prompt_slots={},
            metadata={
                "pattern_name": str(_missing_card_pattern_label(str(pattern_kind))),
                "partial_card_ids": [str(card_id) for card_id in partial_card_ids],
                "candidate_labels": [str(label) for label in labels],
                "candidate_card_ids_by_label": dict(candidate_ids_by_label),
                "candidate_specs_by_label": dict(candidate_specs_by_label),
                "correct_candidate_label": str(answer_label),
                "correct_candidate_card_id": str(answer_card_id),
                "target_candidate_index": int(target_index),
                "candidate_count_support": [int(value) for value in candidate_count_support],
                "candidate_count_probabilities": dict(candidate_count_probabilities),
                "missing_card_pattern_probabilities": dict(pattern_kind_probabilities),
            },
            row_card_counts=(4, int(candidate_count)),
        )
    raise ValueError("failed to sample unique missing-card completion")

def _trick_rank_score(card: Tuple[int, str], *, led_suit: str, trump_suit: str | None) -> Tuple[int, int]:
    """Return trick-taking priority for one played card."""

    rank_value, suit_name = card
    if trump_suit is not None and str(suit_name) == str(trump_suit):
        return (2, int(rank_value))
    if str(suit_name) == str(led_suit):
        return (1, int(rank_value))
    return (0, int(rank_value))


def sample_trick_taking_winner(
    rng,
    *,
    player_count: int,
    target_winner_index: int,
    player_count_support: Sequence[int] = (),
    player_count_probabilities: Mapping[str, float] | None = None,
) -> RuleSample:
    """Sample one labelled trick with a unique winner."""

    player_count_probabilities = dict(player_count_probabilities or {})

    deck = make_deck()
    labels = PLAYER_LABELS[: int(player_count)]
    for _attempt in range(500):
        raw_cards = list(rng.sample(deck, int(player_count)))
        led_suit = str(raw_cards[0][1])
        trump_options: List[str | None] = [None] + [str(suit) for suit in SUIT_NAMES if str(suit) != str(led_suit)]
        trump_suit = trump_options[int(rng.randrange(len(trump_options)))]
        scores = [_trick_rank_score(card, led_suit=str(led_suit), trump_suit=trump_suit) for card in raw_cards]
        best_score = max(scores)
        if scores.count(best_score) != 1:
            continue
        winner_index = int(scores.index(best_score))
        if int(winner_index) != int(target_winner_index):
            continue
        winner_label = str(labels[int(winner_index)])
        winner_option = _option_letter(winner_label)
        cards: List[CardInstance] = []
        for index, (rank_value, suit_name) in enumerate(raw_cards):
            label = str(labels[int(index)])
            cards.append(
                _make_labelled_card(
                    card_id=f"player_{index + 1:02d}_card_01",
                    rank_value=int(rank_value),
                    suit_name=str(suit_name),
                    badge_text=str(label),
                )
            )
        trump_text = "There is no trump suit." if trump_suit is None else f"Trump suit: {trump_suit}."
        return RuleSample(
            pattern_kind="trick_winner",
            scene_variant="trick_row",
            cards=tuple(cards),
            answer=str(winner_option),
            annotation_card_ids=(f"player_{winner_index + 1:02d}_card_01",),
            option_count=int(player_count),
            cards_per_row=int(player_count),
            center_label_mode="rank_suit",
            render_overrides={
                "canvas_height": 360,
                "card_width_px": 96,
                "card_height_px": 136,
                "card_gap_px": 18,
                "row_gap_px": 18,
                "rank_font_size_px": 19,
                "center_symbol_font_size_px": 42,
                "reference_banner_height_px": 24,
                "reference_font_size_px": 13,
                "max_cards_per_row": int(player_count),
            },
            prompt_slots={"trump_text": str(trump_text)},
            metadata={
                "player_labels": list(labels),
                "led_suit": str(led_suit),
                "trump_suit": None if trump_suit is None else str(trump_suit),
                "winning_label": str(winner_label),
                "winning_option": str(winner_option),
                "target_winner_index": int(target_winner_index),
                "player_count_support": [int(value) for value in player_count_support],
                "player_count_probabilities": dict(player_count_probabilities),
                "trick_scores": {str(labels[index]): [int(scores[index][0]), int(scores[index][1])] for index in range(int(player_count))},
            },
        )
    raise ValueError("failed to sample unique trick winner")


def sample_trick_winning_play(
    rng,
    *,
    candidate_count: int,
    played_count: int,
    trump_mode: str,
    target_index: int,
    candidate_count_support: Sequence[int] = (),
    candidate_count_probabilities: Mapping[str, float] | None = None,
    played_count_support: Sequence[int] = (),
    played_count_probabilities: Mapping[str, float] | None = None,
    trump_mode_probabilities: Mapping[str, float] | None = None,
) -> RuleSample:
    """Sample played trick cards plus candidate next cards with one winning play."""

    candidate_count_probabilities = dict(candidate_count_probabilities or {})
    played_count_probabilities = dict(played_count_probabilities or {})
    trump_mode_probabilities = dict(trump_mode_probabilities or {})
    labels = CANDIDATE_LABELS[: int(candidate_count)]
    deck = make_deck()
    for _attempt in range(800):
        raw_cards = list(rng.sample(deck, int(played_count)))
        led_suit = str(raw_cards[0][1])
        if str(trump_mode) == "with_trump":
            trump_candidates = [str(suit_name) for suit_name in SUIT_NAMES if str(suit_name) != str(led_suit)]
            trump_suit = str(trump_candidates[int(rng.randrange(len(trump_candidates)))])
        else:
            trump_suit = None
        played_scores = [_trick_rank_score(card, led_suit=str(led_suit), trump_suit=trump_suit) for card in raw_cards]
        current_best = max(played_scores)
        if played_scores.count(current_best) != 1:
            continue
        current_winner_index = int(played_scores.index(current_best))
        used_cards = {(int(rank), str(suit)) for rank, suit in raw_cards}
        winning_pool: List[Tuple[int, str]] = []
        losing_pool: List[Tuple[int, str]] = []
        for card in deck:
            normalized = (int(card[0]), str(card[1]))
            if normalized in used_cards:
                continue
            score = _trick_rank_score(normalized, led_suit=str(led_suit), trump_suit=trump_suit)
            if tuple(score) > tuple(current_best):
                winning_pool.append(normalized)
            else:
                losing_pool.append(normalized)
        if not winning_pool or len(losing_pool) < int(candidate_count) - 1:
            continue
        correct_card = winning_pool[int(rng.randrange(len(winning_pool)))]
        distractors = list(rng.sample(losing_pool, int(candidate_count) - 1))
        candidate_cards = list(distractors)
        candidate_cards.insert(int(target_index), (int(correct_card[0]), str(correct_card[1])))
        candidate_scores = [
            _trick_rank_score(card, led_suit=str(led_suit), trump_suit=trump_suit)
            for card in candidate_cards
        ]
        winning_labels = [
            str(label)
            for label, score in zip(labels, candidate_scores)
            if tuple(score) > tuple(current_best)
        ]
        if winning_labels != [str(labels[int(target_index)])]:
            continue

        cards: List[CardInstance] = []
        played_card_ids: List[str] = []
        for index, (rank_value, suit_name) in enumerate(raw_cards, start=1):
            card_id = f"played_{index:02d}"
            played_card_ids.append(str(card_id))
            badge_text = "LED" if int(index) == 1 else None
            cards.append(
                _make_labelled_card(
                    card_id=str(card_id),
                    rank_value=int(rank_value),
                    suit_name=str(suit_name),
                    badge_text=badge_text,
                    group_label="Played",
                )
            )

        candidate_ids_by_label: Dict[str, str] = {}
        candidate_specs_by_label: Dict[str, Dict[str, Any]] = {}
        for label, card, score in zip(labels, candidate_cards, candidate_scores):
            rank_value, suit_name = card
            card_id = f"candidate_{str(label)}"
            candidate_ids_by_label[str(label)] = str(card_id)
            candidate_specs_by_label[str(label)] = {
                "card_id": str(card_id),
                "rank_value": int(rank_value),
                "rank_label": str(RANK_LABEL_BY_VALUE[int(rank_value)]),
                "suit_name": str(suit_name),
                "trick_score": [int(score[0]), int(score[1])],
                "wins_if_played": bool(str(label) == str(labels[int(target_index)])),
            }
            cards.append(
                _make_labelled_card(
                    card_id=str(card_id),
                    rank_value=int(rank_value),
                    suit_name=str(suit_name),
                    badge_text=str(label),
                    group_label="Candidates",
                )
            )

        answer_label = str(labels[int(target_index)])
        answer_card_id = str(candidate_ids_by_label[str(answer_label)])
        trump_text = "There is no trump suit." if trump_suit is None else f"Trump suit: {trump_suit}."
        return RuleSample(
            pattern_kind="trick_play",
            scene_variant="trick_candidate_play",
            cards=tuple(cards),
            answer=str(answer_label),
            annotation_card_ids=(str(answer_card_id),),
            option_count=int(candidate_count),
            cards_per_row=int(candidate_count),
            center_label_mode="rank_suit",
            render_overrides={
                "canvas_height": 560,
                "card_width_px": 86,
                "card_height_px": 120,
                "card_gap_px": 18,
                "row_gap_px": 54,
                "rank_font_size_px": 18,
                "center_symbol_font_size_px": 40,
                "reference_banner_height_px": 24,
                "reference_font_size_px": 16,
                "max_cards_per_row": int(candidate_count),
            },
            prompt_slots={"trump_text": str(trump_text)},
            metadata={
                "played_card_ids": [str(card_id) for card_id in played_card_ids],
                "candidate_labels": [str(label) for label in labels],
                "candidate_card_ids_by_label": dict(candidate_ids_by_label),
                "candidate_specs_by_label": dict(candidate_specs_by_label),
                "correct_candidate_label": str(answer_label),
                "correct_candidate_card_id": str(answer_card_id),
                "target_candidate_index": int(target_index),
                "led_suit": str(led_suit),
                "trump_suit": None if trump_suit is None else str(trump_suit),
                "trump_mode": str(trump_mode),
                "current_winning_card_id": str(played_card_ids[int(current_winner_index)]),
                "current_best_score": [int(current_best[0]), int(current_best[1])],
                "played_scores": {
                    str(played_card_ids[index]): [int(score[0]), int(score[1])]
                    for index, score in enumerate(played_scores)
                },
                "winning_option": str(answer_label),
                "candidate_count_support": [int(value) for value in candidate_count_support],
                "candidate_count_probabilities": dict(candidate_count_probabilities),
                "played_count": int(played_count),
                "played_count_support": [int(value) for value in played_count_support],
                "played_count_probabilities": dict(played_count_probabilities),
                "trick_play_trump_mode_probabilities": dict(trump_mode_probabilities),
            },
            row_card_counts=(int(played_count), int(candidate_count)),
        )
    raise ValueError("failed to sample unique trick-winning play")


__all__ = [
    "SUPPORTED_POKER_DRAW_TARGET_CATEGORIES",
    "SUPPORTED_POKER_WINNING_CATEGORIES",
    "SUPPORTED_TRICK_PLAY_TRUMP_MODES",
    "feasible_card_count_support",
    "make_deck",
    "poker_score",
    "sample_blackjack_best_hand",
    "sample_exact_triple_count_hand",
    "sample_higher_rank_hand",
    "sample_longest_run_hand",
    "sample_missing_card_to_complete_hand",
    "sample_poker_best_hand",
    "sample_poker_draw_card",
    "sample_same_suit_hand",
    "sample_trick_taking_winner",
    "sample_trick_winning_play",
    "target_candidate_index",
]
