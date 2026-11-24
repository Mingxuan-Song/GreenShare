"""Simple allocation rules for a fixed pool of neighborhood park area."""

from __future__ import annotations

import numpy as np

from .metrics import _validate


def allocate(area, population, budget_fraction, method="need_first", seed=0):
    """Return final park areas after allocating a shared resource pool.

    A fraction ``budget_fraction`` of every original area enters the same
    pool. The remaining area stays in its original neighborhood. The methods
    allocate the pool equally, in proportion to population, by raising the
    lowest per-person supply first, or by a seeded random Dirichlet draw.
    These are neighborhood-level scenarios, with no parcel feasibility model.
    """
    area, population = _validate(area, population)
    fraction = float(budget_fraction)
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("The budget fraction must lie between zero and one.")
    methods = {"equal", "population", "need_first", "random"}
    if method not in methods:
        raise ValueError(f"Unknown method {method!r}; choose one of {sorted(methods)}.")
    retained = (1 - fraction) * area
    pool = fraction * area.sum()
    if pool == 0:
        return area.copy()
    if method == "equal":
        awards = np.full(area.size, pool / area.size)
    elif method == "population":
        awards = pool * population / population.sum()
    elif method == "random":
        awards = pool * np.random.default_rng(seed).dirichlet(np.ones(area.size))
    else:
        # Find the common level reached by the neighborhoods receiving area.
        # Neighborhoods above this level keep their retained area unchanged.
        lower = float(np.min(retained / population))
        upper = float(area.sum() / population.sum())
        for _ in range(80):
            level = 0.5 * (lower + upper)
            required = np.maximum(0.0, level * population - retained).sum()
            if required < pool:
                lower = level
            else:
                upper = level
        awards = np.maximum(0.0, (lower + upper) * 0.5 * population - retained)
        # Correct floating-point roundoff while keeping all awards nonnegative.
        awards *= pool / awards.sum()
    return retained + awards
