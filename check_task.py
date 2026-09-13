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


def _fit_eval_table(key, tgt, n_keys):
    """Fit a lookup table on the first half, score it on the second.

    The held-out split is not optional. Fitting AND scoring on the same data makes
    the table memorise: at k = 3 there are 60^3 = 216,000 keys against ~500,000
    samples, roughly two samples per key, and the argmax of two samples reproduces
    the data it was fitted on. The first version of this file did exactly that and
    reported a 0.40 "shortcut" that does not exist. The number was measuring the
    estimator's variance, not the task.
    """
    n = key.shape[0] // 2
    tab = torch.zeros(n_keys, ORDER, dtype=torch.long)
    tab.index_put_((key[:n], tgt[:n]), torch.ones_like(key[:n]), accumulate=True)
    pred = tab.argmax(-1)[key[n:]]
    return (pred == tgt[n:]).float().mean().item()


def check_shortcuts(t, x, y, length=64):
    """Can the last k input tokens alone predict the running product?

    Fit the optimal lookup table -- an oracle shortcut, strictly stronger than
    anything a model could learn -- and score it on held-out data. For a group,
    p_{t-k} is uniform and independent of the last k inputs, so p_t is uniform
    given them: every one of these must land at chance. If one does not, the task
    is not depth-bound and E1/E2 would be measuring the wrong thing.
    """
    g = x[:, 1:]                                  # drop BOS
    best = 0.0
    print("  oracle shortcut policies (fit on half, scored on held-out half):")
    for k in (1, 2, 3):
        # key = the last k inputs, as a base-60 integer
        key = torch.zeros_like(g[:, k - 1:])
        for j in range(k):
            key = key * ORDER + g[:, k - 1 - j: g.shape[1] - j]
        acc = _fit_eval_table(key.reshape(-1), y[:, k - 1:].reshape(-1), ORDER ** k)
        best = max(best, acc)
        print(f"    last {k} input token(s)        : {acc:.4f}")
    # position-only shortcut
    pos = torch.arange(y.shape[1]).expand_as(y)
    acc = _fit_eval_table(pos.reshape(-1), y.reshape(-1), y.shape[1])
    print(f"    position index only          : {acc:.4f}")
    return max(best, acc)


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
        top = max(top, check_shortcuts(t, x, y))
        worst = max(worst, top)

    print("\n" + "=" * 78)
    ok = worst < 2.5 * chance_accuracy()
    print(f"GATE: {'PASS' if ok else 'FAIL'} -- best degenerate/shortcut policy scores "
          f"{worst:.4f} against a {chance_accuracy():.4f} floor.")
    print("=" * 78)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
