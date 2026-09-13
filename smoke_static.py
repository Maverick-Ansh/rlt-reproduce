"""C7: the fast fixed-slot kernel must be the same model, and a speed measurement.

Sec. 4.2 is explicit that an optimised kernel is allowed to be faster and not
allowed to be different:

    "Cross-attention should read only valid encoder-prefix memory, and SWA only its
     valid local window; a faster kernel that reads future entries changes the
     model."

The fixed-slot path pads the SWA cache to a constant W-1 slots and hands
cross-attention the whole encoder memory with a mask, so both are now reading
tensors that contain entries they must not use. That is precisely the situation the
sentence above warns about, so it gets checked rather than assumed -- forward
values, and the gradient, against the reference recurrence.

    python smoke_static.py [--length 32] [--compile]
"""
import argparse
import time

import torch
import torch.nn.functional as F

from rlt import RLTConfig, RLT
from rlt.tasks import GroupTask, VOCAB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=32)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--compile", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = RLTConfig(vocab_size=VOCAB, d_model=a.d_model, n_heads=4, L_E=a.layers,
                    L_D=a.layers, d_ff=4 * a.d_model, W=a.window, alpha=1.0,
                    tied=True, max_position=1024)
    torch.manual_seed(0)
    model = RLT(cfg).to(dev)
    task = GroupTask("a5", device=dev)
    x, y = task.sample(a.batch, a.length)
    x, y = x.to(dev), y.to(dev)

    print("=" * 78)
    print("C7  Sec. 4.2 -- the fixed-slot kernel is the same model")
    print("=" * 78)
    ref, _ = model(x, static=False)
    fast, _ = model(x, static=True)
    d = (ref - fast).abs().max().item()
    rel = d / ref.abs().max().item()
    print(f"  max |logit_ref - logit_fast|         : {d:.3e}   (relative {rel:.2e})")

    def grad_of(static):
        model.zero_grad(set_to_none=True)
        logits, _ = model(x, static=static)
        F.cross_entropy(logits[:, 1:].reshape(-1, VOCAB), y.reshape(-1)).backward()
        return torch.cat([p.grad.reshape(-1) for p in model.parameters()
                          if p.grad is not None])
    ga, gb = grad_of(False), grad_of(True)
    cos = F.cosine_similarity(ga[None], gb[None]).item()
    gd = (ga - gb).abs().max().item()
    print(f"  max |grad_ref - grad_fast|           : {gd:.3e}")
    print(f"  cosine(grad_ref, grad_fast)          : {cos:.10f}")
    ok = rel < 1e-4 and cos > 1 - 1e-6
    print(f"  VERDICT: {'CONFIRMED' if ok else 'REFUTED'}")

    print("\n" + "=" * 78)
    print("Speed (this is an implementation number, NOT a claim about the paper's")
    print("hardware co-design -- Sec. 4 makes no measured speedup claim and neither")
    print("does this repository.)")
    print("=" * 78)

    def bench(static, compiled=False, n=8):
        m = model
        if compiled:
            m.enable_compile()
        for _ in range(3):
            logits, _ = m(x, static=static)
            F.cross_entropy(logits[:, 1:].reshape(-1, VOCAB), y.reshape(-1)).backward()
            m.zero_grad(set_to_none=True)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            logits, _ = m(x, static=static)
            F.cross_entropy(logits[:, 1:].reshape(-1, VOCAB), y.reshape(-1)).backward()
            m.zero_grad(set_to_none=True)
        if dev == "cuda":
            torch.cuda.synchronize()
        return (time.time() - t0) / n

    t_ref = bench(False)
    t_fast = bench(True)
    print(f"  reference growing-cache recurrence : {t_ref*1000:8.1f} ms / fwd+bwd")
    print(f"  fixed-slot recurrence              : {t_fast*1000:8.1f} ms "
          f"({t_ref/t_fast:.2f}x)")
    if a.compile:
        t0 = time.time()
        t_c = bench(True, compiled=True)
        print(f"  fixed-slot + torch.compile         : {t_c*1000:8.1f} ms "
              f"({t_ref/t_c:.2f}x)   [compile+warmup included above]")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
