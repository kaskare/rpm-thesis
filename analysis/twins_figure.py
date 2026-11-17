#!/usr/bin/env python3
"""Thesis figure: sum and midpoint under the value axis's convolution, with and without `plus` (outline figure 15).

Rule twins, rows of 3, values 0-50; exact % on the test rule, unseen puzzles 6300-7299. At 100 and 600 puzzles the
points are means of three seeds and the whiskers one sd (docs/ARITHMETIC.md, "Three seeds"; analysis/twins_seeds.py);
at 6000 one run ("Result"). Counting learners: "What a built-in move does" (analysis/rule_orbits.py). The ceiling is 97 first steps over 1,225 rows: the most any answer
chosen by the first step can get on sum.

    python3 analysis/twins_figure.py            # writes docs/figures/twins_sum_midpoint.{png,pdf}
Any Python with matplotlib (the base Anaconda on RCI has it; pynl does not).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "docs/figures/twins_sum_midpoint"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"       # categorical slots 1-3, validated all-pairs (light)

N3 = [100, 600, 6000]
DATA = {   # rule -> series -> (puzzles, exact %)
    "sum": {
        "conv": (N3, [4.5, 6.2, 7.1]),
        "plus": ([100, 600], [98.2, 99.7]),
        "pervalue": (N3, [4.8, 19.4, 69.8]),
        "learner": ([100, 300, 600, 6000], [3.6, 5.3, 6.3, 7.3]),
    },
    "midpoint": {
        "conv": (N3, [60.3, 91.0, 92.7]),
        "plus": ([100, 600], [63.9, 91.7]),
        "pervalue": (N3, [5.0, 18.0, 75.2]),
        "learner": ([100, 300, 600, 6000], [80.0, 98.8, 99.8, 100.0]),
    },
}
SD = {      # rule -> series -> sd over seeds 1-3 at 100 and 600 puzzles
    "sum": {"conv": [2.3, 0.8], "plus": [0.8, 0.4], "pervalue": [0.6, 1.1]},
    "midpoint": {"conv": [2.7, 3.4], "plus": [3.9, 1.8], "pervalue": [0.5, 0.9]},
}
SERIES = {   # key -> (legend label, colour, marker)
    "conv": ("convolution along the values (14,785 weights)", BLUE, "o"),
    "plus": ("convolution + learned use of plus (19,937 weights)", ORANGE, "s"),
    "pervalue": ("weights per value, no symmetry", AQUA, "^"),
}
CEILING = 100 * 97 / 1225   # 7.9


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
                         "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
                         "axes.labelcolor": INK2, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.1), sharey=True, facecolor=SURFACE)
    titles = {"sum": "Sum   c = a + b", "midpoint": "Midpoint   c = (a + b) / 2"}
    for ax, rule in zip(axes, ("sum", "midpoint")):
        d = DATA[rule]
        ax.set_facecolor(SURFACE)
        ax.set_xscale("log")
        ax.set_xlim(70, 9500)
        ax.set_ylim(0, 104)
        ax.set_xticks([100, 300, 600, 6000])
        ax.set_xticklabels(["100", "300", "600", "6000"])
        ax.minorticks_off()
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0, pad=4)
        ax.set_title(titles[rule], loc="left", fontsize=9, color=INK, pad=8, fontweight="bold")
        ax.set_xlabel("training puzzles")

        x, y = d["learner"]                                       # reference: the convolution's move, no network
        ax.plot(x, y, color=MUTED, linewidth=1.2, linestyle=(0, (1.5, 2)), zorder=2)
        for k in ("pervalue", "conv", "plus"):
            x, y = d[k]
            _, c, m = SERIES[k]
            ax.errorbar(x[:2], y[:2], yerr=SD[rule][k], fmt="none", ecolor=c, elinewidth=1.1, capsize=2.5,
                        capthick=1.1, zorder=3)
            ax.plot(x, y, color=c, linewidth=2, marker=m, markersize=5.5, markeredgecolor=SURFACE,
                    markeredgewidth=1.4, solid_capstyle="round", solid_joinstyle="round", zorder=4)

        if rule == "sum":
            ax.axhline(CEILING, color=INK2, linewidth=0.9, linestyle=(0, (5, 3)), zorder=2)
            ax.text(75, CEILING + 2.5, f"ceiling {CEILING:.1f}", fontsize=7, color=INK2, va="bottom")
            ax.text(6600, CEILING + 2.5, "convolution 7.1", fontsize=7, color=INK, va="bottom", ha="center")
            ax.text(640, 99.7 - 6, "with plus 99.7", fontsize=7, color=INK, va="top")
            ax.text(5600, 69.8 + 4, "per value 69.8", fontsize=7, color=INK, va="bottom", ha="center")
        else:
            ax.text(6600, 100 + 1.5, "counting learner 100", fontsize=7, color=INK2, va="bottom", ha="right")
            ax.text(6600, 92.7 - 3.5, "convolution 92.7", fontsize=7, color=INK, va="top", ha="center")
            ax.annotate("with plus 91.7", xy=(640, 90.6), xytext=(1300, 70), fontsize=7, color=INK, ha="center",
                        arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.7))
            ax.text(9300, 61, "per value 75.2", fontsize=7, color=INK, va="center", ha="right")
            ax.annotate("0 of 101 rows with an unseen\nfirst step right, at each seed", xy=(100, 60.3), xytext=(118, 34),
                        fontsize=6.5, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.7))
    axes[0].set_ylabel("exact on unseen puzzles (%)")

    handles = [plt.Line2D([], [], color=c, linewidth=2, marker=m, markersize=5.5, markeredgecolor=SURFACE)
               for _, c, m in SERIES.values()]
    labels = [lab for lab, _, _ in SERIES.values()]
    handles += [plt.Line2D([], [], color=MUTED, linewidth=1.2, linestyle=(0, (1.5, 2)))]
    labels += ["counting learner with the convolution's move only"]
    handles += [plt.Line2D([], [], color=INK2, linewidth=0.9, linestyle=(0, (5, 3)))]
    labels += ["ceiling on sum for that move: 97 first steps / 1,225 rows"]
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=7, handlelength=2.6,
               columnspacing=1.6, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=2.0)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=220, facecolor=SURFACE)
    print(f"wrote {OUT}.png and {OUT}.pdf")


if __name__ == "__main__":
    main()
