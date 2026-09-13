"""Figure for REPORT.md: accuracy vs evaluation length, A5 beside its Z60 control.

Reads `results/summary.json` (produced by `analyze.py --summary`) so the figure can
be regenerated anywhere the numbers are, without the GPU box.

Two rows. The top row is overall accuracy, which is generous: it averages over
positions, and early positions are shallow. The bottom row is accuracy at the FINAL
position, which is the one requiring the deepest composition and the one that
actually separates the arms.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHANCE = 1 / 60
STYLE = {
    "rlt": ("RLT  (alpha=1)", "#1f4e79", "-", "o"),
    "rlt_nope": ("RLT  (alpha=1, no RoPE)", "#2e8b57", "-", "s"),
    "rlt_untied": ("RLT  untied", "#7b68ee", "--", "v"),
    "rlt_a0": ("RLT  alpha=0  [recurrence off]", "#c44e52", "-", "^"),
    "plain": ("plain Transformer", "#8c8c8c", ":", "D"),
}


def arm_of(tag):
    group = tag.split("_")[0]
    return group, tag[len(group) + 1:].rsplit("_s", 1)[0]


def main(path="results/summary.json", out="figure.png"):
    data = json.load(open(path))
    train_len = next(iter(data.values()))["length"]
    series = {}
    for tag, r in data.items():
        g, arm = arm_of(tag)
        series.setdefault((g, arm), []).append(r["eval"])

    groups = [g for g in ("a5", "z60") if any(k[0] == g for k in series)]
    fig, axes = plt.subplots(2, len(groups), figsize=(5.6 * len(groups), 7.6),
                             sharex=True, sharey="row", squeeze=False)
    titles = {"a5": "A5  —  non-abelian, NC1-complete",
              "z60": "Z60  —  abelian, shallow-computable"}

    for col, g in enumerate(groups):
        for row, (idx, label) in enumerate([(0, "accuracy (all positions)"),
                                            (1, "accuracy at final position")]):
            ax = axes[row][col]
            for arm, (name, color, ls, mk) in STYLE.items():
                if (g, arm) not in series:
                    continue
                runs = series[(g, arm)]
                lengths = sorted(int(k) for k in runs[0])
                for i, ev in enumerate(runs):
                    ys = [ev[str(L)][idx] for L in lengths]
                    ax.plot(lengths, ys, color=color, ls=ls, marker=mk, ms=4.5,
                            lw=1.9 if i == 0 else 1.0,
                            alpha=1.0 if i == 0 else 0.45,
                            label=f"{name}  (n={len(runs)})" if i == 0 else None)
            ax.axhline(CHANCE, color="black", lw=0.9, ls=(0, (4, 3)))
            ax.axvline(train_len, color="black", lw=0.8, alpha=0.35)
            ax.set_xscale("log", base=2)
            ax.set_xticks(lengths)
            ax.set_xticklabels([str(L) for L in lengths])
            ax.set_ylim(-0.04, 1.06)
            ax.grid(alpha=0.18)
            if col == 0:
                ax.set_ylabel(label)
            if row == 0:
                ax.set_title(titles[g], fontsize=11)
            if row == 1:
                ax.set_xlabel("evaluation sequence length")

    axes[0][0].text(train_len * 1.06, 0.5, "trained here", rotation=90,
                    va="center", fontsize=8, alpha=0.6)
    axes[0][0].text(lengths[0], CHANCE + 0.03, "chance = 1/60", fontsize=8, alpha=0.7)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(3, len(l)), frameon=False,
               bbox_to_anchor=(0.5, -0.005), fontsize=9)
    fig.suptitle("Unbounded temporal depth on a depth-bound task, and its matched control",
                 fontsize=12.5, y=0.985)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print("wrote", out, os.path.getsize(out), "bytes")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
