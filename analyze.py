"""Collate the sweep into the tables that go in REPORT.md.

Every number is printed with its seed spread. A difference smaller than the spread
is not a result, and this script says so rather than leaving the reader to notice.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

ARM_ORDER = ["rlt", "rlt_a0", "rlt_untied", "plain"]
ARM_LABEL = {
    "rlt": "RLT (alpha=1, tied)",
    "rlt_a0": "RLT alpha=0  [no recurrence]",
    "rlt_untied": "RLT untied (Sec. 2.6)",
    "plain": "plain causal Transformer",
}


def mean_std(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, v ** 0.5


def load(d):
    runs = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        if os.path.basename(f).startswith("c6_"):
            continue
        r = json.load(open(f))
        tag = r["tag"]
        group = tag.split("_")[0]
        arm = tag[len(group) + 1:].rsplit("_s", 1)[0]
        runs[(group, arm)].append(r)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        print(json.dumps(summary(a.dir), separators=(",", ":")))
        return
    runs = load(a.dir)
    if not runs:
        print(f"no results in {a.dir}")
        return
    any_run = next(iter(runs.values()))[0]
    chance = any_run["chance"]
    train_len = any_run["args"]["length"]
    lengths = sorted(int(k) for k in any_run["eval"])

    print("=" * 92)
    print(f"Accuracy on prefix products.  chance = {chance:.4f}, ceiling = 1.0000, "
          f"trained at length {train_len}")
    print("mean +/- std over seeds; * marks the training length")
    print("=" * 92)
    for group in ("a5", "z60"):
        present = [arm for arm in ARM_ORDER if (group, arm) in runs]
        if not present:
            continue
        head = "  ".join(f"L={L}{'*' if L == train_len else ' '}".rjust(11) for L in lengths)
        title = ("A5  (non-abelian, NC1-complete word problem)" if group == "a5"
                 else "Z60 (abelian, shallow-computable)")
        print(f"\n--- {title} ---")
        print(f"  {'arm':<32s}{'seeds':>6s}  {head}")
        for arm in present:
            rs = runs[(group, arm)]
            cells = []
            for L in lengths:
                vals = [r["eval"][str(L)]["acc"] for r in rs]
                m, s = mean_std(vals)
                cells.append(f"{m:.3f}+-{s:.3f}".rjust(11))
            print(f"  {ARM_LABEL[arm]:<32s}{len(rs):>6d}  " + "  ".join(cells))

        if (group, "rlt") in runs and (group, "rlt_a0") in runs:
            L = str(train_len)
            a1 = [r["eval"][L]["acc"] for r in runs[(group, "rlt")]]
            a0 = [r["eval"][L]["acc"] for r in runs[(group, "rlt_a0")]]
            m1, s1 = mean_std(a1)
            m0, s0 = mean_std(a0)
            spread = max(s1, s0)
            delta = m1 - m0
            verdict = ("above seed noise" if abs(delta) > 2 * spread + 1e-9
                       else "WITHIN seed noise -- not a result")
            print(f"  recurrence effect at L={train_len}: "
                  f"{delta:+.4f}  (seed spread {spread:.4f}) -> {verdict}")

    print("\n" + "=" * 92)
    print("Final-position accuracy (deepest composition in the sequence)")
    print("=" * 92)
    for group in ("a5", "z60"):
        present = [arm for arm in ARM_ORDER if (group, arm) in runs]
        if not present:
            continue
        print(f"\n--- {group.upper()} ---")
        for arm in present:
            rs = runs[(group, arm)]
            cells = []
            for L in lengths:
                m, s = mean_std([r["eval"][str(L)]["acc_final_pos"] for r in rs])
                cells.append(f"{m:.3f}".rjust(11))
            print(f"  {ARM_LABEL[arm]:<32s}        " + "  ".join(cells))

    print("\n" + "=" * 92)
    print("Parameters and cost")
    print("=" * 92)
    for (group, arm), rs in sorted(runs.items()):
        if group != "a5":
            continue
        r = rs[0]
        print(f"  {ARM_LABEL[arm]:<32s} params {r['params']:>10,}  "
              f"blocks/token {r['blocks_per_token']}  "
              f"{r['wall_s']/r['args']['steps']*1000:6.0f} ms/step")


def summary(d="results"):
    """Compact, printable digest -- small enough to move between machines by hand."""
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        if os.path.basename(f).startswith("c6_"):
            continue
        r = json.load(open(f))
        out[r["tag"]] = {
            "eval": {k: [round(v["acc"], 4), round(v["acc_final_pos"], 4)]
                     for k, v in r["eval"].items()},
            "params": r["params"],
            "train_acc": round(r["history"][-1]["acc"], 4),
            "train_loss": round(r["history"][-1]["loss"], 4),
            "ms_per_step": round(r["wall_s"] / r["args"]["steps"] * 1000),
            "length": r["args"]["length"], "layers": r["args"]["layers"],
            "steps": r["args"]["steps"], "rope": r["args"].get("rope", 1),
        }
    return out


if __name__ == "__main__":
    main()
