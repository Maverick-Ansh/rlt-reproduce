"""Exactness checks C1-C5. Run this before spending a single GPU-minute on training.

Each check prints a VERDICT line. These are not shape assertions -- they are the
paper's own propositions and its own stated non-equivalences, turned into numbers.
Every tolerance is compared against a *measured* floating-point noise floor rather
than a number picked by hand, because "these two schedules agree" is only
meaningful relative to how well a schedule agrees with itself.

    python smoke.py [--device cuda] [--dtype float32]
"""
import argparse
import torch

from rlt import RLTConfig, RLT
from rlt.tasks import GroupTask


def hdr(s):
    print("\n" + "=" * 78)
    print(s)
    print("=" * 78)


def make(device, dtype, **kw):
    cfg = RLTConfig(vocab_size=61, d_model=64, n_heads=4, L_E=3, L_D=3,
                    d_ff=128, W=4, G=1, alpha=1.0, tied=True, **kw)
    torch.manual_seed(0)
    m = RLT(cfg).to(device=device, dtype=dtype).eval()
    return cfg, m


# --------------------------------------------------------------- noise floor

@torch.no_grad()
def noise_floor(model, x):
    """How much does the SAME computation disagree with itself?

    Padding the batch changes cuBLAS tiling and therefore reduction order, so this
    is a genuine lower bound on any cross-schedule difference we can claim to
    resolve. Every tolerance below is stated as a multiple of this.
    """
    a, _ = model(x, parallel_encode=True)
    xx = torch.cat([x, x.flip(0)], dim=0)
    b, _ = model(xx, parallel_encode=True)
    return (a - b[: x.shape[0]]).abs().max().item()


# ------------------------------------------------------- C1 split invariance

@torch.no_grad()
def check_c1(model, x, floor):
    """Prop. 3.1: "Moving the prompt-response split does not change the conditional
    distribution for a fixed token history." """
    hdr("C1  Prop. 3.1 -- invariance to the serving split")
    ref, _ = model(x, parallel_encode=True)          # one batched prefill
    inc, _ = model(x, parallel_encode=False)         # fully incremental encoder
    d_inc = (ref - inc).abs().max().item()
    print(f"  parallel prefill  vs  fully incremental : {d_inc:.3e}")

    worst, worst_T = 0.0, None
    S = x.shape[1]
    for T in range(1, S):
        st = model.init_state(x.shape[0], x.device, model.embed.weight.dtype)
        l1, st = model.consume(x[:, :T], st, parallel_encode=True)     # prompt
        l2, _ = model.consume(x[:, T:], st, parallel_encode=True)      # response
        d = (torch.cat([l1, l2], 1) - ref).abs().max().item()
        if d > worst:
            worst, worst_T = d, T
    print(f"  worst over all {S - 1} split points T     : {worst:.3e}  (at T={worst_T})")
    print(f"  measured numerical noise floor          : {floor:.3e}")
    ok = worst <= max(floor * 10, 1e-9) and d_inc <= max(floor * 10, 1e-9)
    print(f"  VERDICT: {'CONFIRMED' if ok else 'REFUTED'} "
          f"(split difference is {worst / max(floor, 1e-30):.1f}x the noise floor)")
    return ok, worst, floor


# ------------------------------------------------------------- C2 causality

@torch.no_grad()
def check_c2(model, x):
    """Prop. B.1: H_t depends only on x_{1:t}.

    Masked logits are exactly -inf, so their softmax weights are exactly 0 and
    contribute exactly 0.0 to the reduction. The correct verdict here is therefore
    BITWISE equality, not "small" -- anything else means future information is
    leaking through a mask that is merely small rather than zero.
    """
    hdr("C2  Prop. B.1 -- causality of the complete state H_t")
    t = x.shape[1] // 2
    y = x.clone()
    y[:, t:] = torch.randint(0, 60, y[:, t:].shape, device=x.device)
    la, _, sa = model(x, collect_states=True)
    lb, _, sb = model(y, collect_states=True)
    ds = (sa[:, :t] - sb[:, :t]).abs().max().item()
    dl = (la[:, :t] - lb[:, :t]).abs().max().item()
    after = (la[:, t:] - lb[:, t:]).abs().max().item()
    print(f"  max |s_j(x) - s_j(x')| for j < t (perturbed tail) : {ds:.3e}")
    print(f"  max |logit_j difference| for j < t                : {dl:.3e}")
    print(f"  max |logit_j difference| for j >= t  (sanity)     : {after:.3e}")
    ok = (ds == 0.0) and (dl == 0.0) and (after > 1e-4)
    print(f"  VERDICT: {'CONFIRMED (bitwise)' if ok else 'REFUTED'}")
    return ok, ds


# --------------------------------------- C3 parallel SWA is not equivalent

@torch.no_grad()
def check_c3(device, dtype, x, floor):
    """Sec. 2.4: "a standard parallel SWA decoder pass is not generally equivalent."

    Sharper than a yes/no. A parallel pass must guess s_{t-1}; each Jacobi sweep
    makes exactly one more position exact. If K sweeps fixed more than K positions
    there would be a parallel shortcut, which is what Sec. 4.1 denies: "No exact
    parallel scan for the general nonlinear decoder is assumed."
    """
    hdr("C3  Sec. 2.4 -- a token-parallel SWA decoder pass is not the recurrence")
    for alpha in (1.0, 0.0):
        cfg, m = make(device, dtype, )
        m.cfg.alpha = alpha
        ref, _ = m(x, parallel_encode=True)
        print(f"\n  alpha = {alpha}   (alpha = 0 severs Eq. (2.11)'s feedback term)")
        for K in (1, 2, 3, 4, 8):
            par, _ = m.forward_parallel_swa(x, n_iters=K)
            err = (par - ref).abs().amax(dim=(0, 2))       # per position
            exact = int((err <= max(floor * 10, 1e-9)).long().cumprod(0).sum())
            print(f"    {K} parallel sweep(s): positions exact = {exact:3d}"
                  f"   max err overall = {err.max().item():.3e}")
            if alpha == 1.0 and K == 1:
                one_sweep = err.max().item()
        if alpha == 0.0:
            zero_sweep = err.max().item()
    ok = one_sweep > 1e-3 and zero_sweep <= max(floor * 10, 1e-9)
    print(f"\n  With feedback, one parallel sweep is off by {one_sweep:.3e} (O(1), not O(eps)).")
    print(f"  With alpha = 0 the parallel pass is exact: {zero_sweep:.3e}.")
    print(f"  VERDICT: {'CONFIRMED' if ok else 'REFUTED'}")
    return ok, one_sweep, zero_sweep


# --------------------------------------------------- C4 gradient path detach

def check_c4(device, dtype, task):
    """App. C / Sec. 5.4: detaching only s_t is not a full truncation.

    Probe: || dL / d s_star ||. The learned initial state enters the computation at
    t = 0 and nowhere else, so it can only be reached by a gradient that travelled
    back through the entire prompt recurrence. Encoder memory cannot carry it --
    Sec. 2.2, "This global memory depends on encoder representations, not decoder
    states" -- so the complete detach of Eq. (C.1) must zero it exactly.
    """
    hdr("C4  App. C / Eq. (C.1) -- which detach removes which gradient path")
    cfg, m = make(device, dtype)
    m.train()
    x, y = task.sample(8, 24)
    x, y = x.to(device), y.to(device)
    T = 13                                   # prompt/response boundary

    rows = []
    for name, kw in [("full BPTT (reference)", None),
                     ("detach s_t only", dict(s=True, swa=False, enc=False, mem=False)),
                     ("detach decoder KV only", dict(s=False, swa=True, enc=False, mem=False)),
                     ("complete detach, Eq. (C.1)", dict(s=True, swa=True, enc=False, mem=False))]:
        m.zero_grad(set_to_none=True)
        st = m.init_state(x.shape[0], device, dtype)
        _, st = m.consume(x[:, :T], st)
        if kw is not None:
            st = st.detach(**kw)
        logits, _ = m.consume(x[:, T:], st)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, cfg.vocab_size), y[:, T - 1:].reshape(-1))
        loss.backward()
        g_star = m.s_star.grad.norm().item() if m.s_star.grad is not None else 0.0
        g_all = sum((p.grad.norm().item() ** 2) for p in m.parameters()
                    if p.grad is not None) ** 0.5
        rows.append((name, g_star, g_all))
        print(f"  {name:<30s}  ||dL/ds_star|| = {g_star:.6e}   ||dL/dtheta|| = {g_all:.4e}")

    ref, s_only, kv_only, both = [r[1] for r in rows]
    ok = (both == 0.0) and (s_only > 0.0) and (kv_only > 0.0) and (ref > s_only)
    print("\n  Detaching s_t alone leaves the decoder-KV path open: "
          f"{s_only:.3e} > 0.")
    print(f"  Only the complete detach of Eq. (C.1) zeroes it: {both:.3e}.")
    print(f"  VERDICT: {'CONFIRMED' if ok else 'REFUTED'}")
    return ok, rows


# ------------------------------------------------------ C5 temporal depth

def check_c5(device, dtype, task):
    """Sec. 3.3 / Fig. 2 -- and the hedge attached to it.

    The block count is true by construction (t * L_D), so counting it proves
    nothing interesting. The claim worth measuring is the one the paper itself
    puts at risk: "Gates, contraction, and learned projections may suppress the
    practical contribution of long paths" (Sec. 3.3), and App. B, "Normalization
    alone does not bound products of these Jacobians."

    So: measure || d s_t / d s_star || as t grows. Structural depth is real only if
    this stays measurably non-zero.
    """
    hdr("C5  Sec. 3.3 -- unbounded temporal depth, structural and effective")
    out = {}
    for alpha in (1.0, 0.0):
        cfg, m = make(device, dtype)
        m.cfg.alpha = alpha
        m.train()
        x, _ = task.sample(4, 128)
        x = x.to(device)
        g = torch.Generator(device="cpu").manual_seed(1)
        r = torch.randn(cfg.d_model, generator=g).to(device=device, dtype=dtype)
        r = r / r.norm()
        series = []
        for t in (1, 2, 4, 8, 16, 32, 64, 128):
            m.zero_grad(set_to_none=True)
            _, _, s = m(x[:, :t], collect_states=True)
            (s[:, -1] @ r).sum().backward()
            gn = m.s_star.grad.norm().item() if m.s_star.grad is not None else 0.0
            series.append((t, gn, t * cfg.L_D))
        out[alpha] = series
        print(f"\n  alpha = {alpha}")
        print("    t     decoder blocks on the state path     || d(r.s_t) / d s_star ||")
        for t, gn, blocks in series:
            print(f"    {t:>4d}  {blocks:>10d}  ({t} x L_D={cfg.L_D})" + " " * 12 + f"{gn:.6e}")
    a1 = [g for _, g, _ in out[1.0]]
    a0 = [g for _, g, _ in out[0.0]]
    ok = a1[-1] > 0.0 and max(a0) == 0.0
    print(f"\n  alpha = 0 gives exactly zero at every t: the recurrent path is the only")
    print(f"  route from s_star to s_t, so the ablation really does sever it.")
    print(f"  alpha = 1 at t = 128: {a1[-1]:.3e}  (decay from t=1: {a1[-1] / max(a1[0], 1e-30):.3e}x)")
    print(f"  VERDICT: {'CONFIRMED (path exists and carries gradient)' if ok else 'REFUTED'}")
    return ok, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32")
    a = ap.parse_args()
    dev, dt = a.device, getattr(torch, a.dtype)
    torch.manual_seed(0)

    task = GroupTask("a5", device=dev)
    cfg, m = make(dev, dt)
    x, _ = task.sample(4, 24)
    x = x.to(dev)

    print(f"device={dev} dtype={dt}  model: d={cfg.d_model} L_E={cfg.L_E} "
          f"L_D={cfg.L_D} W={cfg.W} tied={cfg.tied}  params={m.n_params():,}")
    floor = noise_floor(m, x)
    print(f"measured numerical noise floor (same computation, different batch "
          f"padding): {floor:.3e}")

    results = {}
    results["C1"] = check_c1(m, x, floor)[0]
    results["C2"] = check_c2(m, x)[0]
    results["C3"] = check_c3(dev, dt, x, floor)[0]
    results["C4"] = check_c4(dev, dt, task)[0]
    results["C5"] = check_c5(dev, dt, task)[0]

    hdr("SUMMARY")
    for k, v in results.items():
        print(f"  {k}: {'CONFIRMED' if v else 'REFUTED'}")
    if not all(results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
