"""Figures shared by the notebook and the research report."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, TwoSlopeNorm
from matplotlib.collections import PatchCollection
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from shapely.geometry.polygon import orient

from .metrics import lorenz
from .analysis import METHODS

COLORS = {"baseline": "#657582", "equal": "#d88b3d", "population": "#5285b5",
          "need_first": "#168773", "random": "#b7bfca"}


def set_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 10, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": "white",
        "savefig.facecolor": "white", "axes.titleweight": "bold",
        "legend.frameon": False, "svg.fonttype": "none"})


def save_figure(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{name}.png", dpi=180, bbox_inches="tight")
    return fig


def _map(ax, geometries, values, cmap="YlGnBu", norm=None):
    patches, repeated = [], []
    minx = min(g.bounds[0] for g in geometries)
    miny = min(g.bounds[1] for g in geometries)
    for geometry, value in zip(geometries, values):
        parts = [geometry] if geometry.geom_type == "Polygon" else geometry.geoms
        for polygon in parts:
            if polygon.geom_type != "Polygon":
                continue
            polygon = orient(polygon, sign=1.0)
            vertices, codes = [], []
            for ring in [polygon.exterior, *polygon.interiors]:
                xy = np.asarray(ring.coords)
                vertices.extend((xy-np.array([minx, miny]))/1000)
                codes.extend([MplPath.MOVETO] + [MplPath.LINETO]*(len(xy)-2) + [MplPath.CLOSEPOLY])
            patches.append(PathPatch(MplPath(vertices, codes)))
            repeated.append(value)
    collection = PatchCollection(patches, cmap=cmap, norm=norm, edgecolor="white", linewidth=0.25)
    collection.set_array(np.asarray(repeated, float))
    ax.add_collection(collection)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.annotate("N", (0.94, 0.91), xycoords="axes fraction", ha="center", fontweight="bold")
    ax.annotate("", (0.94, 0.90), (0.94, 0.82), xycoords="axes fraction",
                arrowprops={"arrowstyle": "->", "color": "#374151"})
    ax.plot([2, 7], [2, 2], color="#374151", lw=2)
    ax.text(4.5, 2.8, "5 km", ha="center", fontsize=8)
    return collection


def baseline_maps(frame, geometries):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 5.6), layout="constrained")
    for ax, values, title, unit, cmap in [
        (axes[0], frame.pop_density_km2, "Where residents live", "Residents per km²", "Blues"),
        (axes[1], frame.park_per_person, "Park space per resident", "Park m² per resident", "YlGnBu")]:
        collection = _map(ax, geometries, values, cmap, PowerNorm(0.5, vmin=0, vmax=float(max(values))))
        ax.set_title(title, loc="left")
        ticks = [0, 2000, 5000, 10000] if ax is axes[0] else [0, 10, 50, 100, 180]
        bar = fig.colorbar(collection, ax=ax, orientation="horizontal", shrink=0.85,
                          pad=0.02, fraction=0.045, ticks=ticks)
        bar.set_label(unit)
        bar.ax.tick_params(labelsize=8)
    return fig


def distribution_figure(frame):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), layout="constrained")
    x, y = lorenz(frame.park_per_person, frame.population)
    axes[0].plot(x*100, y*100, color=COLORS["need_first"], lw=2.5, label="Baseline distribution")
    axes[0].plot([0, 100], [0, 100], "--", color="#9da7b0", label="Equal per-resident supply")
    axes[0].set(xlabel="Cumulative residents (%)", ylabel="Cumulative park area (%)",
                title="A. Population-weighted Lorenz curve", xlim=(0, 100), ylim=(0, 100))
    axes[0].legend(fontsize=8, loc="upper left")
    ordered = frame.sort_values("park_per_person").reset_index(drop=True)
    axes[1].scatter(np.arange(1, len(frame)+1), ordered.park_per_person,
                    s=16, c=COLORS["need_first"], alpha=.8)
    mean = frame.park_m2.sum()/frame.population.sum()
    axes[1].axhline(mean, color=COLORS["equal"], lw=1.5, ls="--", label=f"Citywide mean: {mean:.2f}")
    axes[1].set(xlabel="Community rank (lowest to highest)", ylabel="Park m² per resident (log scale)",
                title="B. Differences between communities", yscale="log")
    axes[1].legend(loc="upper left", fontsize=8)
    return fig


def policy_figure(summary):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5), layout="constrained")
    colors = [COLORS[m] for m in summary.method]
    labels = ["Baseline", "Equal\ncommunity", "Population\nshare", "Need\nfirst"]
    for ax, key, title, ylabel in [(axes[0], "gini", "A. Overall inequality", "Population-weighted Gini"),
                                  (axes[1], "minimum", "B. Lowest community supply", "Park m² per resident")]:
        bars = ax.bar(labels, summary[key], color=colors, width=.62)
        ax.bar_label(bars, fmt="%.3f" if key == "gini" else "%.2f", fontsize=9, padding=4)
        ax.set(title=title, ylabel=ylabel, ylim=(0, float(summary[key].max())*1.22))
        ax.tick_params(axis="x", labelsize=9)
    return fig


def change_figure(frame, geometries, allocations):
    set_style()
    change = (allocations["need_first"]-allocations["baseline"])/frame.population.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.8), gridspec_kw={"width_ratios": [1, 1.1]}, layout="constrained")
    limit = float(np.max(np.abs(change)))
    collection = _map(axes[0], geometries, change, "BrBG", TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit))
    axes[0].set_title("A. Net change under need first", loc="left")
    fig.colorbar(collection, ax=axes[0], orientation="horizontal", shrink=.9,
                 fraction=.045, pad=.02).set_label("Change in park m² per resident")
    order = np.argsort(change)
    selected = np.r_[order[:5], order[-5:]]
    values = change[selected]
    axes[1].barh(np.arange(10), values, color=np.where(values>=0, COLORS["need_first"], "#aa7957"))
    axes[1].set_yticks(np.arange(10), ["O'Hare" if n == "OHARE" else str(n).title()
                                     for n in frame.iloc[selected].name])
    axes[1].axvline(0, color="#657582", lw=.8)
    axes[1].set(xlabel="Change in park m² per resident", title="B. Five largest gains and losses")
    axes[1].tick_params(axis="y", labelsize=8)
    return fig


def sensitivity_figure(sensitivity):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), layout="constrained")
    for method, label in METHODS.items():
        rows = sensitivity.loc[sensitivity.method == method]
        for ax, key in zip(axes, ["gini", "minimum"]):
            ax.plot(rows.budget*100, rows[key], label=label, color=COLORS[method], lw=2)
    for ax in axes:
        ax.set_xlabel("Shared pool (% of total park area)")
        ax.axvline(5, color="#9da7b0", ls=":", lw=1)
        ax.legend(fontsize=8)
    axes[0].set(title="A. Overall inequality", ylabel="Population-weighted Gini")
    axes[1].set(title="B. Lowest community supply", ylabel="Park m² per resident")
    return fig


def random_figure(random, summary):
    set_style()
    fig, ax = plt.subplots(figsize=(8.4, 3.1), layout="constrained")
    ax.hist(random.gini, bins=18, color=COLORS["random"], edgecolor="white", label="200 random pool allocations")
    for method, label in METHODS.items():
        value = summary.loc[summary.method==method, "gini"].iloc[0]
        ax.axvline(value, color=COLORS[method], lw=2, label=label)
    ax.set(xlabel="Population-weighted Gini", ylabel="Number of draws")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(.43, 1))
    return fig
