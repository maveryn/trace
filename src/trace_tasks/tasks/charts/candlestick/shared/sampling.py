"""Neutral OHLC candle sampling for the candlestick scene."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.sampling import uniform_choice_with_probabilities
from trace_tasks.core.seed import hash64, spawn_rng
from trace_tasks.tasks.charts.candlestick.shared.defaults import GENERATION_DEFAULTS, SCENE_NAMESPACE
from trace_tasks.tasks.charts.candlestick.shared.state import Candle
from trace_tasks.tasks.charts.shared.label_assets import sample_chart_labels
from trace_tasks.tasks.shared.config_defaults import resolve_required_int_bounds


def sample_candles(params: Mapping[str, Any], *, instance_seed: int) -> tuple[Candle, ...]:
    """Sample one OHLC series with unique body and wick ranges for objective selection."""

    candle_min, candle_max = resolve_required_int_bounds(
        params,
        GENERATION_DEFAULTS,
        min_key="candle_count_min",
        max_key="candle_count_max",
        fallback_min=7,
        fallback_max=10,
        context="candlestick candle count",
    )
    value_min, value_max = resolve_required_int_bounds(
        params,
        GENERATION_DEFAULTS,
        min_key="value_min",
        max_key="value_max",
        fallback_min=8,
        fallback_max=96,
        context="candlestick value range",
    )
    body_min, body_max = resolve_required_int_bounds(
        params,
        GENERATION_DEFAULTS,
        min_key="body_size_min",
        max_key="body_size_max",
        fallback_min=4,
        fallback_max=20,
        context="candlestick body size",
    )
    wick_min, wick_max = resolve_required_int_bounds(
        params,
        GENERATION_DEFAULTS,
        min_key="wick_size_min",
        max_key="wick_size_max",
        fallback_min=2,
        fallback_max=10,
        context="candlestick wick size",
    )
    if int(body_max) - int(body_min) + 1 < int(candle_max):
        raise ValueError("body size support must allow unique body ranges")

    for attempt in range(250):
        rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.candles.retry", int(attempt))
        candle_count = int(candle_min) + int(rng.randrange(int(candle_max) - int(candle_min) + 1))
        labels = sample_chart_labels(
            count=int(candle_count),
            instance_seed=int(hash64(int(instance_seed), "candlestick.labels", int(attempt))),
            namespace=f"{SCENE_NAMESPACE}.labels:{int(candle_count)}",
        )
        body_sizes = list(range(int(body_min), int(body_max) + 1))
        rng.shuffle(body_sizes)
        body_sizes = body_sizes[: int(candle_count)]
        candles: list[Candle] = []
        for index in range(int(candle_count)):
            body = int(body_sizes[index])
            direction = str(rng.choice(("up", "down")))
            if index == 0:
                direction = "up"
            if index == 1:
                direction = "down"
            if direction == "up":
                open_low = int(value_min) + int(wick_max)
                open_high = int(value_max) - int(body) - int(wick_max)
                if open_low > open_high:
                    break
                open_value = int(rng.randint(open_low, open_high))
                close_value = int(open_value) + int(body)
            else:
                open_low = int(value_min) + int(body) + int(wick_max)
                open_high = int(value_max) - int(wick_max)
                if open_low > open_high:
                    break
                open_value = int(rng.randint(open_low, open_high))
                close_value = int(open_value) - int(body)
            upper_wick = int(rng.randint(int(wick_min), int(wick_max)))
            lower_wick = int(rng.randint(int(wick_min), int(wick_max)))
            high_value = max(int(open_value), int(close_value)) + int(upper_wick)
            low_value = min(int(open_value), int(close_value)) - int(lower_wick)
            if not (int(value_min) <= int(low_value) < int(high_value) <= int(value_max)):
                break
            candles.append(
                Candle(
                    candle_id=f"candle_{index}",
                    label=str(labels[index]),
                    open_value=int(open_value),
                    high_value=int(high_value),
                    low_value=int(low_value),
                    close_value=int(close_value),
                )
            )
        if len(candles) != int(candle_count):
            continue
        directions = [candle.direction for candle in candles]
        if directions.count("up") < 2 or directions.count("down") < 2:
            continue
        if len({int(candle.body_size) for candle in candles}) != len(candles):
            continue
        if len({int(candle.wick_range) for candle in candles}) != len(candles):
            continue
        return tuple(candles)
    raise ValueError("could not sample a valid candlestick OHLC series")


def select_semantic_branch(
    params: Mapping[str, Any],
    *,
    branch_key: str,
    support: tuple[str, ...],
    instance_seed: int,
    namespace: str,
) -> tuple[str, dict[str, float], dict[str, Any]]:
    """Select a task-owned semantic branch without knowing public query identity."""

    requested = params.get(str(branch_key))
    if requested is not None:
        selected = str(requested)
        if selected not in support:
            raise ValueError(f"unsupported {branch_key}: {selected}; supported: {support}")
        branch_params = dict(params)
        branch_params.pop(str(branch_key), None)
        return selected, {value: (1.0 if value == selected else 0.0) for value in support}, branch_params
    selected, probabilities = uniform_choice_with_probabilities(
        spawn_rng(int(instance_seed), str(namespace)),
        support,
    )
    return str(selected), dict(probabilities), dict(params)


__all__ = ["sample_candles", "select_semantic_branch"]
