"""Figures for the README from reports/benchmark.json (means over seeds).

Palette: categorical slots 1-3 (blue, orange, aqua), validated for colour-vision deficiency; aqua is
below 3:1 contrast, so every series is also direct-labelled and all values are tabulated in the
README.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
SERIES = {  # key in benchmark.json -> (label, colour)
    "svm_rbf": ("SVM (RBF)", "#2a78d6"),
    "random_forest": ("Random forest", "#eb6834"),
    "snr_threshold_baseline": ("SNR threshold (no ML)", "#1baf7a"),
}
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def mean_over_seeds(seeds, getter):
    return float(np.mean([getter(s) for s in seeds]))


def main() -> None:
    r = json.loads((ROOT / "reports" / "benchmark.json").read_text(encoding="utf-8"))
    seeds = list(r["seeds"].values())
    FIG.mkdir(parents=True, exist_ok=True)

    # 1. Detection rate by SNR (speaker-grouped CV)
    snrs = ["0", "5", "10", "15", "20"]
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    style(ax)
    for key, (label, colour) in SERIES.items():
        y = [
            mean_over_seeds(
                seeds,
                lambda s, k=k, m=key: s["quality_detection_by_condition"][m]["recall_by_snr_db"][k],
            )
            for k in snrs
        ]
        ax.plot([int(k) for k in snrs], y, color=colour, lw=2, marker="o", ms=6)
        ax.annotate(
            label,
            (20, y[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=TEXT,
        )
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlim(-1, 28)
    ax.set_ylim(0.8, 1.005)
    ax.set_xlabel("Signal-to-noise ratio of the added noise (dB)", color=MUTED, fontsize=9)
    ax.set_ylabel("Degraded segments detected", color=MUTED, fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Quieter noise is harder to detect", loc="left", color=TEXT, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "detection_by_snr.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # 2. Unseen noise type: recall on the held-out type, by model
    held = ["white", "pink", "brown", "hum", "babble"]
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    style(ax)
    width = 0.26
    x = np.arange(len(held))
    for i, (key, (label, colour)) in enumerate(SERIES.items()):
        y = [
            mean_over_seeds(
                seeds,
                lambda s, h=h, m=key: s["quality_unseen_noise_type"][h][m]["recall_unseen_noise"],
            )
            for h in held
        ]
        ax.bar(
            x + (i - 1) * (width + 0.02),
            y,
            width,
            color=colour,
            label=label,
            edgecolor=SURFACE,
            linewidth=1,
        )
    ax.set_xticks(x, [h.capitalize() for h in held])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Noise type held out of training", color=MUTED, fontsize=9)
    ax.set_ylabel("Detected when unseen", color=MUTED, fontsize=9)
    ax.set_title(
        "Detection of a noise type the model never saw", loc="left", color=TEXT, fontsize=11, pad=26
    )
    ax.legend(
        frameon=False,
        fontsize=8,
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=3,
        labelcolor=TEXT,
        borderaxespad=0.2,
    )
    fig.tight_layout()
    fig.savefig(FIG / "unseen_noise_recall.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
