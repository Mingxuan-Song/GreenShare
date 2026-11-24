"""Population-weighted descriptions of neighborhood park supply."""

from __future__ import annotations

import numpy as np


def _validate(values, population):
    """Return equally sized one-dimensional float arrays with valid values."""
    values = np.asarray(values, dtype=float)
    population = np.asarray(population, dtype=float)
    if values.ndim != 1 or population.ndim != 1 or values.shape != population.shape:
        raise ValueError("Values and population must be equally sized 1D arrays.")
    if not values.size:
        raise ValueError("At least one neighborhood is required.")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(population)):
        raise ValueError("Values and population must be finite.")
    if np.any(values < 0) or np.any(population <= 0):
        raise ValueError("Values must be nonnegative and populations positive.")
    return values, population


def weighted_gini(values, population):
    """Gini of per-person values, weighting each neighborhood by residents.

    Every resident is assigned their neighborhood's per-person park supply.
    Zero total supply has no normalized Gini and returns NaN.
    """
    values, population = _validate(values, population)
    order = np.argsort(values, kind="stable")
    values, weights = values[order], population[order]
    weights = weights / weights.sum()
    mean = np.dot(weights, values)
    if mean == 0:
        return float("nan")
    # The sorted formula avoids making an n by n matrix of pairwise gaps.
    cumulative = np.cumsum(weights)
    gini = np.sum(weights * values * (2 * cumulative - weights - 1)) / mean
    return float(np.clip(gini, 0.0, 1.0))


def lorenz(values, population):
    """Cumulative population and cumulative park-resource fractions."""
    values, population = _validate(values, population)
    order = np.argsort(values, kind="stable")
    weights = population[order]
    resource = weights * values[order]
    if resource.sum() == 0:
        raise ValueError("A Lorenz curve needs positive total park supply.")
    return (
        np.r_[0.0, np.cumsum(weights) / weights.sum()],
        np.r_[0.0, np.cumsum(resource) / resource.sum()],
    )


def _weighted_quantile(values, weights, probability):
    """Inverse weighted empirical CDF, without interpolating neighborhoods."""
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order]) / weights.sum()
    index = min(np.searchsorted(cumulative, probability, side="left"), len(values) - 1)
    return float(values[order][index])


def summarize(area, population, baseline=None):
    """Describe park areas in square meters and their per-resident distribution.

    ``baseline`` is an optional array of original park areas. The low-supply
    threshold is half the baseline citywide mean, so it stays fixed across
    comparisons. Gains and losses describe neighborhood-level assignments.
    """
    area, population = _validate(area, population)
    supply = area / population
    if baseline is None:
        baseline = area.copy()
    baseline, _ = _validate(baseline, population)
    total_population = population.sum()
    mean = area.sum() / total_population
    threshold = 0.5 * baseline.sum() / total_population
    tolerance = np.maximum(1e-8, np.abs(baseline) * 1e-10)
    return {
        "gini": weighted_gini(supply, population),
        "mean": float(mean),
        "minimum": float(supply.min()),
        "p10": _weighted_quantile(supply, population, 0.1),
        "below_half_mean_share": float(population[supply < threshold].sum() / total_population),
        "population_gaining_share": float(population[area > baseline + tolerance].sum() / total_population),
        "population_losing_share": float(population[area < baseline - tolerance].sum() / total_population),
        "total_area": float(area.sum()),
        "population": float(total_population),
    }
