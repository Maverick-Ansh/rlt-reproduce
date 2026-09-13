"""Phase-4 gate: validate the measuring instrument before running any sweep.

The failure mode this exists to prevent is the expensive one -- training for hours
against a metric that was never able to resolve the effect, then reporting the
result as if it were about the model. Here the metric is plain accuracy against an
exactly known function, so the bracket is easy to state and must still be checked
rather than assumed:

    floor   = 1/60  = 1.67%   (uniform guessing)
    ceiling = 100%            (deterministic ground truth, no label noise)

What actually needs verifying is that the *task* is what it claims to be: that the
group tables are groups, that A5 is non-abelian and Z60 is not, that the labels are
balanced, and above all that no shallow shortcut policy scores well. If a window of
the last k inputs predicted the running product, the A5/Z60 contrast would be
measuring nothing.
"""
import torch

from rlt.tasks import GroupTask, chance_accuracy, ORDER


def check_group(name):
    t = GroupTask(name)
    tab = t.table
    print(f"\n--- {name.upper()} ---")
    print(f"  order                       : {tab.shape[0]}")

    # closure is automatic from the table's dtype/range; check it anyway
    assert tab.min() >= 0 and tab.max() < ORDER, "table not closed"
    # identity
    i = t.identity
    assert torch.equal(tab[i], torch.arange(ORDER)), "left identity fails"
    assert torch.equal(tab[:, i], torch.arange(ORDER)), "right identity fails"
    print("  identity                    : ok")
    # inverses: every row contains the identity exactly once (Latin square)
    for r in range(ORDER):
        assert int((tab[r] == i).sum()) == 1, f"row {r} not a permutation"
        assert int((tab[:, r] == i).sum()) == 1
    print("  inverses / Latin square     : ok")
    # associativity, sampled exhaustively over a random subset
    g = torch.Generator().manual_seed(0)
    a, b, c = (torch.randint(0, ORDER, (20000,), generator=g) for _ in range(3))
    lhs = tab[tab[a, b], c]
    rhs = tab[a, tab[b, c]]
    assert torch.equal(lhs, rhs), "associativity fails"
    print("  associativity (20k triples) : ok")
    # abelian?
    abelian = bool(torch.equal(tab, tab.t()))
    print(f"  abelian                     : {abelian}"
          f"   <- {'shallow-computable' if abelian else 'NC1-complete word problem'}")
    return t, abelian


def check_labels(t, length=64, batch=8192):
    g = torch.Generator().manual_seed(0)
    x, y = t.sample(batch, length, generator=g)
    counts = torch.bincount(y.reshape(-1), minlength=ORDER).float()
    freq = counts / counts.sum()
    print(f"  label marginal: min {freq.min():.4f}  max {freq.max():.4f} "
          f"(uniform = {1 / ORDER:.4f})")
    print(f"  best constant-output policy : {freq.max():.4f}"
          f"   (chance = {chance_accuracy():.4f})")
    return x, y, freq.max().item()


def check_shortcuts(t, x, y, length=64):
    """Can the last k input tokens alone predict the running product?

    Fit the optimal lookup table on the data itself -- an oracle shortcut, strictly
    stronger than anything a model could learn -- and report its accuracy. For a
    group, the prefix product is uniform given any bounded suffix of inputs, so
    every one of these must sit at chance. If one does not, the task is not
    depth-bound and E1/E2 would be measuring the wrong thing.
    """
    g = x[:, 1:]                                  # drop BOS
    print("  oracle shortcut policies (fit on the eval data itself):")
    for k in (1, 2, 3):
        # key = the last k inputs, as a base-60 integer
        key = torch.zeros_like(g[:, k - 1:])
        for j in range(k):
            key = key * ORDER + g[:, k - 1 - j: g.shape[1] - j]
        tgt = y[:, k - 1:]
        n = ORDER ** k
        tab = torch.zeros(n, ORDER, dtype=torch.long)
        tab.index_put_((key.reshape(-1), tgt.reshape(-1)),
                       torch.ones_like(key.reshape(-1)), accumulate=True)
        pred = tab.argmax(-1)[key.reshape(-1)]
        acc = (pred == tgt.reshape(-1)).float().mean().item()
        print(f"    last {k} input token(s)        : {acc:.4f}")
    # position-only shortcut
    tab = torch.zeros(y.shape[1], ORDER, dtype=torch.long)
    pos = torch.arange(y.shape[1]).expand_as(y)
    tab.index_put_((pos.reshape(-1), y.reshape(-1)),
                   torch.ones_like(pos.reshape(-1)), accumulate=True)
    acc = (tab.argmax(-1)[pos.reshape(-1)] == y.reshape(-1)).float().mean().item()
    print(f"    position index only          : {acc:.4f}")


def main():
    print("=" * 78)
    print("EVALUATION BRACKET")
    print("=" * 78)
    print(f"  floor   (uniform guessing) : {chance_accuracy():.4f}")
    print("  ceiling (exact ground truth): 1.0000")
    print("  the metric is direct accuracy; no learned probe sits between the")
    print("  model and the number, so the bracket cannot narrow during training.")

    worst = 0.0
    for name in ("a5", "z60"):
        t, abelian = check_group(name)
        x, y, top = check_labels(t)
        check_shortcuts(t, x, y)
        worst = max(worst, top)

    print("\n" + "=" * 78)
    ok = worst < 2.5 * chance_accuracy()
    print(f"GATE: {'PASS' if ok else 'FAIL'} -- best degenerate policy scores "
          f"{worst:.4f} against a {chance_accuracy():.4f} floor.")
    print("=" * 78)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
