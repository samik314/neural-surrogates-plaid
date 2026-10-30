"""Draw the README's result figures from docs/figures/results_2d.csv.

    python scripts/make_figures.py        # writes docs/figures/ablation_2d.png, leaderboard_2d.png

Static PNGs on an explicit light surface, so they read the same in GitHub's light and dark
themes. One hue (blue) marks this repository's results; neutral gray marks everything else.
Every bar carries its value as text, so no reading depends on color alone.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"  # this repo's final results
BLUE_LIGHT = "#a9c9f0"  # intermediate steps of the ablation path
GRAY = "#b4b2a9"  # baselines


def load() -> dict[str, list[dict[str, str]]]:
    rows = [r for r in csv.reader((FIG / "results_2d.csv").read_text().splitlines()) if r]
    data = [r for r in rows if not r[0].startswith("#")]
    header, body = data[0], data[1:]
    out: dict[str, list[dict[str, str]]] = {}
    for r in body:
        d = dict(zip(header, r, strict=True))
        out.setdefault(d["kind"], []).append(d)
    return out


def _style(ax: plt.Axes, xlabel: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(TEXT_2)
    ax.tick_params(axis="y", length=0, labelcolor=TEXT, labelsize=10)
    ax.tick_params(axis="x", colors=TEXT_2, labelsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=TEXT_2, fontsize=9)


def _bars(ax, labels, values, colors, texts, xmax, stds=None) -> None:
    y = list(range(len(labels)))[::-1]  # first row on top
    ax.barh(y, values, height=0.62, color=colors, edgecolor=SURFACE, linewidth=2)
    if stds is not None:
        for yi, v, s in zip(y, values, stds, strict=True):
            if s:
                ax.errorbar(v, yi, xerr=s, fmt="none", ecolor=TEXT_2, elinewidth=1, capsize=3)
    pads = stds if stds is not None else [0.0] * len(values)
    for yi, v, s, t in zip(y, values, pads, texts, strict=True):  # label clears the error bar
        ax.text(v + s + xmax * 0.012, yi, t, va="center", ha="left", color=TEXT, fontsize=9)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, xmax)


def ablation(rows: list[dict[str, str]]) -> None:
    labels = [r["label"].replace("->", "→") for r in rows]
    values = [float(r["value"]) for r in rows]
    stds = [float(r["std"]) if r["std"] else 0.0 for r in rows]
    colors = [BLUE if r["ours"] == "1" else BLUE_LIGHT for r in rows]
    texts = [f"{v:.4f}" + (f" ± {s:.4f}" if s else "") for v, s in zip(values, stds, strict=True)]
    fig, ax = plt.subplots(figsize=(8.0, 3.9), dpi=200, facecolor=SURFACE)
    _bars(ax, labels, values, colors, texts, xmax=0.135, stds=stds)
    _style(ax, "dev total relative RMSE (lower is better); mean ± std over 3 seeds")
    ax.set_title(
        "2D_MultiScHypEl: one change per step",
        loc="left", color=TEXT, fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIG / "ablation_2d.png", facecolor=SURFACE)
    plt.close(fig)


def leaderboard(rows: list[dict[str, str]]) -> None:
    rows = sorted(rows, key=lambda r: float(r["value"]))  # best on top
    labels = [r["label"] for r in rows]
    values = [float(r["value"]) for r in rows]
    colors = [BLUE if r["ours"] == "1" else GRAY for r in rows]
    texts = [f"{v:.4f}" for v in values]
    fig, ax = plt.subplots(figsize=(8.0, 3.0), dpi=200, facecolor=SURFACE)
    _bars(ax, labels, values, colors, texts, xmax=0.068)
    _style(ax, "public-leaderboard total relative RMSE (lower is better)")
    ours = {r["label"] for r in rows if r["ours"] == "1"}
    for tick in ax.get_yticklabels():  # match by text: tick order is not row order
        if tick.get_text() in ours:
            tick.set_fontweight("bold")
    ax.set_title(
        "2D_MultiScHypEl leaderboard vs published baselines",
        loc="left", color=TEXT, fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIG / "leaderboard_2d.png", facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    data = load()
    ablation(data["ablation"])
    leaderboard(data["board"])
    print(f"wrote {FIG / 'ablation_2d.png'} and {FIG / 'leaderboard_2d.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
