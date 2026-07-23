"""Density-curve metric and PDF primitives."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

from trace_tasks.tasks.charts.density_curve.shared.state import DensityCurve


def normal_pdf(x_value: float, mean: float, sigma: float) -> float:
    """Return an unnormalized normal-like density value."""

    sigma = max(1e-6, float(sigma))
    z_value = (float(x_value) - float(mean)) / sigma
    return math.exp(-0.5 * z_value * z_value) / sigma


def split_normal_pdf(x_value: float, mode: float, left_sigma: float, right_sigma: float) -> float:
    """Return an asymmetric unimodal density value."""

    sigma = float(left_sigma) if float(x_value) < float(mode) else float(right_sigma)
    z_value = (float(x_value) - float(mode)) / max(1e-6, sigma)
    return math.exp(-0.5 * z_value * z_value)


def student_t_like_pdf(x_value: float, center: float, scale: float, nu: float) -> float:
    """Return a heavy-tailed density value."""

    scaled = (float(x_value) - float(center)) / max(1e-6, float(scale))
    return (1.0 + (scaled * scaled / max(1.0, float(nu)))) ** (
        -(float(nu) + 1.0) / 2.0
    )


def lognormal_like_pdf(x_value: float, shift: float, mu: float, sigma: float) -> float:
    """Return a shifted lognormal-like density value."""

    shifted = max(0.25, float(x_value) - float(shift))
    log_value = math.log(shifted)
    return math.exp(-((log_value - float(mu)) ** 2) / (2.0 * float(sigma) * float(sigma))) / (
        shifted * max(1e-6, float(sigma))
    )


def density_values_for_family(
    rng,
    *,
    family: str,
    xs: Sequence[float],
) -> Tuple[Tuple[float, ...], int, Tuple[dict[str, float], ...]]:
    """Sample one density family over x-values."""

    if str(family) == "gaussian":
        mean = float(rng.uniform(18.0, 82.0))
        sigma = float(rng.uniform(7.0, 15.0))
        return tuple(normal_pdf(x, mean, sigma) for x in xs), 1, (
            {"mean": mean, "sigma": sigma, "weight": 1.0},
        )

    if str(family) in {"gaussian_mixture_2", "gaussian_mixture_3", "asymmetric_bimodal"}:
        component_count = 3 if str(family) == "gaussian_mixture_3" else 2
        base_centers = sorted(float(rng.uniform(14.0, 86.0)) for _ in range(int(component_count)))
        if str(family) == "asymmetric_bimodal":
            base_centers = sorted((float(rng.uniform(16.0, 38.0)), float(rng.uniform(58.0, 86.0))))
        raw_weights = [float(rng.uniform(0.45, 1.35)) for _ in range(int(component_count))]
        weight_total = sum(raw_weights)
        weights = [float(value) / float(weight_total) for value in raw_weights]
        sigmas = [float(rng.uniform(5.0, 12.5)) for _ in range(int(component_count))]
        values = []
        for x_value in xs:
            values.append(
                sum(
                    float(weight) * normal_pdf(float(x_value), float(center), float(sigma))
                    for weight, center, sigma in zip(weights, base_centers, sigmas)
                )
            )
        components = tuple(
            {"mean": float(center), "sigma": float(sigma), "weight": float(weight)}
            for center, sigma, weight in zip(base_centers, sigmas, weights)
        )
        return tuple(values), int(component_count), components

    if str(family) == "skewed_unimodal":
        mode = float(rng.uniform(22.0, 78.0))
        if rng.random() < 0.5:
            left_sigma, right_sigma = float(rng.uniform(5.0, 8.5)), float(rng.uniform(12.0, 20.0))
        else:
            left_sigma, right_sigma = float(rng.uniform(12.0, 20.0)), float(rng.uniform(5.0, 8.5))
        values = tuple(split_normal_pdf(x, mode, left_sigma, right_sigma) for x in xs)
        return values, 1, (
            {"mode": mode, "left_sigma": left_sigma, "right_sigma": right_sigma, "weight": 1.0},
        )

    if str(family) == "lognormal_like":
        shift = float(rng.uniform(-4.0, 8.0))
        mu = float(rng.uniform(3.0, 4.0))
        sigma = float(rng.uniform(0.28, 0.55))
        values = tuple(lognormal_like_pdf(x, shift, mu, sigma) for x in xs)
        return values, 1, (
            {"shift": shift, "mu": mu, "sigma": sigma, "weight": 1.0},
        )

    if str(family) == "student_t_like":
        center = float(rng.uniform(20.0, 80.0))
        scale = float(rng.uniform(7.0, 14.0))
        nu = float(rng.uniform(2.2, 5.5))
        values = tuple(student_t_like_pdf(x, center, scale, nu) for x in xs)
        return values, 1, (
            {"center": center, "scale": scale, "nu": nu, "weight": 1.0},
        )

    raise ValueError(f"unsupported density family: {family}")


def integral(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Return trapezoid integral over sampled points."""

    if len(xs) < 2:
        return 0.0
    total = 0.0
    for index in range(len(xs) - 1):
        dx = float(xs[index + 1]) - float(xs[index])
        total += 0.5 * dx * (float(ys[index]) + float(ys[index + 1]))
    return float(total)


def interval_integral(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    start: float,
    end: float,
) -> float:
    """Return trapezoid integral within a closed x interval."""

    selected_x: list[float] = []
    selected_y: list[float] = []
    for x_value, y_value in zip(xs, ys):
        if float(start) <= float(x_value) <= float(end):
            selected_x.append(float(x_value))
            selected_y.append(float(y_value))
    return integral(selected_x, selected_y)


def interpolated_y_at_x(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    x_value: float,
) -> float:
    """Return linearly interpolated y at one x location."""

    if not xs or not ys:
        return 0.0
    target = float(x_value)
    if target <= float(xs[0]):
        return float(ys[0])
    if target >= float(xs[-1]):
        return float(ys[-1])
    for index in range(len(xs) - 1):
        left_x, right_x = float(xs[index]), float(xs[index + 1])
        if left_x <= target <= right_x:
            left_y, right_y = float(ys[index]), float(ys[index + 1])
            t_value = (target - left_x) / max(1e-9, right_x - left_x)
            return left_y + (t_value * (right_y - left_y))
    return float(ys[-1])


def curve_metrics(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    interval_start: float,
    interval_end: float,
    reference_x: float,
) -> tuple[float, float, float, float, float]:
    """Return mean, mode x/y, interval mass, and density-at-x metrics."""

    mass = max(1e-9, integral(xs, ys))
    mean_x = integral(xs, [float(x) * float(y) for x, y in zip(xs, ys)]) / mass
    peak_index = max(
        range(len(ys)),
        key=lambda index: (float(ys[index]), -abs(float(xs[index]) - mean_x)),
    )
    mode_x = float(xs[int(peak_index)])
    mode_y = float(ys[int(peak_index)])
    interval_mass = interval_integral(
        xs,
        ys,
        start=float(interval_start),
        end=float(interval_end),
    ) / mass
    density_at_x = interpolated_y_at_x(xs, ys, x_value=float(reference_x))
    return float(mean_x), float(mode_x), float(mode_y), float(interval_mass), float(density_at_x)


def metric_value(curve: DensityCurve, metric_name: str) -> float:
    """Return one supported semantic metric from a curve."""

    if str(metric_name) == "mean_x":
        return float(curve.mean_x)
    if str(metric_name) == "mode_x":
        return float(curve.mode_x)
    if str(metric_name) == "interval_mass":
        return float(curve.interval_mass)
    if str(metric_name) == "density_at_x":
        return float(curve.density_at_x)
    raise ValueError(f"unsupported density-curve metric: {metric_name}")


def metric_values_by_label(
    curves: Sequence[DensityCurve],
    *,
    metric_name: str,
) -> dict[str, float]:
    """Return one metric keyed by curve label."""

    return {str(curve.label): float(metric_value(curve, str(metric_name))) for curve in curves}


def extremum_label(
    curves: Sequence[DensityCurve],
    *,
    metric_name: str,
    direction: str,
) -> str:
    """Return the label at the requested metric extremum."""

    labels = tuple(str(curve.label) for curve in curves)
    values = {str(curve.label): metric_value(curve, str(metric_name)) for curve in curves}
    if str(direction) == "max":
        return max(labels, key=lambda label: (float(values[label]), label))
    if str(direction) == "min":
        return min(labels, key=lambda label: (float(values[label]), label))
    raise ValueError(f"unsupported extremum direction: {direction}")


def extremum_gap(
    curves: Sequence[DensityCurve],
    *,
    metric_name: str,
    direction: str,
) -> float:
    """Return the winning margin for one metric extremum."""

    values = sorted(metric_value(curve, str(metric_name)) for curve in curves)
    if len(values) < 2:
        return 0.0
    if str(direction) == "max":
        return abs(float(values[-1] - values[-2]))
    if str(direction) == "min":
        return abs(float(values[1] - values[0]))
    raise ValueError(f"unsupported extremum direction: {direction}")
