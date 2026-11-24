"""Small comparisons with a shared source pool and fixed total park area."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .allocation import allocate
from .metrics import summarize, weighted_gini

METHODS = {"equal": "Equal community", "population": "Population share",
           "need_first": "Need first"}
DEFAULT_BUDGET = 0.05


def compare_policies(frame, budget=DEFAULT_BUDGET):
    """Return one row per rule and each rule's final community areas."""
    area = frame["park_m2"].to_numpy(float)
    population = frame["population"].to_numpy(float)
    rows = [{"method": "baseline", "label": "Baseline", "budget": 0.0,
             **summarize(area, population, baseline=area)}]
    allocations = {"baseline": area.copy()}
    for method, label in METHODS.items():
        final = allocate(area, population, budget, method)
        allocations[method] = final
        incoming = final - (1 - budget) * area
        rows.append({"method": method, "label": label, "budget": budget,
                     **summarize(final, population, baseline=area),
                     "max_incoming_land_fraction": float(np.max(
                         incoming / frame["area_m2"].to_numpy(float))),
                     "net_moved_area": float(np.abs(final-area).sum()/2),
                     "pool_area": float(budget*area.sum())})
    return pd.DataFrame(rows), allocations


def budget_sensitivity(frame, budgets=None):
    """Vary the same pool fraction for all three rules."""
    if budgets is None:
        budgets = np.linspace(0, 0.20, 21)
    area, population = frame["park_m2"].to_numpy(float), frame["population"].to_numpy(float)
    rows = []
    for budget in budgets:
        for method, label in METHODS.items():
            final = allocate(area, population, float(budget), method)
            rows.append({"budget": float(budget), "method": method, "label": label,
                         **summarize(final, population, baseline=area)})
    return pd.DataFrame(rows)


def random_reference(frame, budget=DEFAULT_BUDGET, repeats=200, seed=2026):
    """Dirichlet(1,...,1) pool shares; draw spread is scenario variability."""
    area, population = frame["park_m2"].to_numpy(float), frame["population"].to_numpy(float)
    rows = []
    for trial in range(repeats):
        final = allocate(area, population, budget, "random", seed=seed+trial)
        rows.append({"trial": trial, "seed": seed+trial,
                     **summarize(final, population, baseline=area)})
    return pd.DataFrame(rows)


def verify_results(frame, summary, allocations, sensitivity):
    """Check invariants that are independent of the comparison figures."""
    area, p = frame["park_m2"].to_numpy(float), frame["population"].to_numpy(float)
    budget = float(summary.loc[summary.method == "need_first", "budget"].iloc[0])
    checks = {}
    for method, final in allocations.items():
        checks[f"{method}_area_conserved"] = bool(np.isclose(final.sum(), area.sum(), rtol=1e-12))
        checks[f"{method}_nonnegative"] = bool(np.all(final >= 0))
        checks[f"{method}_source_floor"] = bool(np.all(final >= (1-budget)*area-1e-6))
    expected = (1-budget)*weighted_gini(area/p, p)
    actual = weighted_gini(allocations["population"]/p, p)
    checks["population_gini_identity"] = bool(np.isclose(actual, expected, atol=1e-12))
    levels = allocations["need_first"]/p
    incoming = allocations["need_first"]-(1-budget)*area
    active = incoming > 1e-6
    checks["need_first_equal_recipient_levels"] = bool(not active.any() or np.ptp(levels[active]) < 1e-9)
    checks["need_first_recipient_level_is_minimum"] = bool(not active.any() or np.min(levels) >= np.max(levels[active])-1e-9)
    checks["all_sensitivity_area_conserved"] = bool(np.allclose(sensitivity.total_area, area.sum(), rtol=1e-12))
    checks["all_sensitivity_mean_constant"] = bool(np.allclose(sensitivity["mean"], area.sum()/p.sum(), rtol=1e-12))
    if not all(checks.values()):
        raise AssertionError({k: v for k,v in checks.items() if not v})
    return checks


def save_results(root, frame, quality, summary, allocations, sensitivity, random):
    root = Path(root)
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "policy_comparison.csv", index=False)
    sensitivity.to_csv(out / "budget_sensitivity.csv", index=False)
    random.to_csv(out / "random_allocations.csv", index=False)
    detail = frame.copy()
    for method, final in allocations.items():
        detail[f"{method}_park_m2"] = final
        detail[f"{method}_m2_per_person"] = final/detail.population
    detail.to_csv(out / "community_allocations.csv", index=False)
    checks = verify_results(frame, summary, allocations, sensitivity)
    base = summary.iloc[0]
    need = summary.loc[summary.method == "need_first"].iloc[0]
    info = {"community_count": len(frame), "population": float(frame.population.sum()),
            "park_area_km2": float(frame.park_m2.sum()/1e6),
            "mapped_area_km2": float(frame.area_m2.sum()/1e6),
            "baseline_gini": float(base.gini), "baseline_mean": float(base["mean"]),
            "baseline_minimum": float(base.minimum), "need_first_gini": float(need.gini),
            "need_first_minimum": float(need.minimum),
            "relative_gini_reduction": float(1-need.gini/base.gini),
            "need_first_gaining_share": float(need.population_gaining_share),
            "need_first_losing_share": float(need.population_losing_share),
            "need_first_incoming_communities": int(np.sum(allocations["need_first"] >
                (1-float(need.budget))*frame.park_m2.to_numpy()+1e-6)),
            "random_repeats": len(random), "random_gini_min": float(random.gini.min()),
            "random_gini_max": float(random.gini.max()),
            "random_gini_median": float(random.gini.median()),
            "checks": checks}
    (out / "summary.json").write_text(json.dumps(info, indent=2, allow_nan=False)+"\n")
    return info
