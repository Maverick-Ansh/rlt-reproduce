"""C6: current-policy replay (Sec. 5.3, Sec. 5.4, App. A.4).

The paper's central engineering claim is a statement about equality, which makes it
unusually easy to falsify:

    "At identical parameters and sampling conventions, trainer and sampler
     represent the same computation." (Sec. 5.3)

If that holds, then replaying a rollout at exactly the parameters that produced it
must return importance ratios r_i = exp(log p_Theta - log mu) of exactly 1. Not
approximately 1. The test needs no training and no task performance.

Four replay schedules are compared, three of which the paper explicitly rejects:

    exact    App. A.4 step 3: rebuild s_t and every decoder SWA cache from
             H_0 = (s_star, empty) through the full prompt and response.
    stale    Reuse the sampler's H_T. Sec. 5.4: "A sampler's old hidden states
             cannot replace current-policy replay."
    reset    Start the response from H_0, i.e. treat the prompt/response split as a
             computation boundary. This is the "structural mismatch caused by
             different prompt/response computations" (Sec. 1) that RLT exists to
             remove; no RLT implementation should do this, and it is included to
             show what the mismatch costs.
    nograd   Exact forward values, prompt recurrence under no_grad. Sec. 5.4:
             "preserves the forward probabilities if the state is recomputed at
             current parameters, but omits prompt-state derivatives and is a
             gradient approximation."

The task keeps the boundary load-bearing: the prompt is a word in A5, the response
is N further group elements, and the reward is 1 iff the whole product is the
identity. A policy that forgets the prompt state at the boundary cannot score above
chance, so "carry H_T across the split" is not decoration here.
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F

from rlt import RLTConfig, RLT
from rlt.tasks import GroupTask, VOCAB, BOS, ORDER


def masked_logprobs(logits):
    """Log-probabilities over the 60 group elements; BOS is not a legal action.

    Sampler and trainer must apply the SAME transform or the comparison is
    meaningless -- so there is exactly one function that produces log-probs, used
    by both.
    """
    logits = logits.float().clone()
    logits[..., BOS] = float("-inf")
    return torch.log_softmax(logits, dim=-1)


@torch.no_grad()
def rollout(model, task, batch, prompt_len, n_new, device, temperature=1.0,
            top_k=0, generator=None):
    """Sample a batch of episodes and record everything the trainer will need.

    Sec. 5.3: "Behavior log-probabilities must include temperature, truncation, and
    renormalization; metadata alone cannot restore missing support."  So log_mu is
    computed from the transformed distribution actually sampled, not from the raw
    model.
    """
    model.eval()
    x, targets = task.sample(batch, prompt_len)
    x, targets = x.to(device), targets.to(device)
    st = model.init_state(batch, device, model.embed.weight.dtype)
    logits, st = model.consume(x, st)
    boundary = st                      # H_T, the exact prefix snapshot of App. C

    ys, logmus = [], []
    cur = logits[:, -1]
    for _ in range(n_new):
        l = cur.float() / max(temperature, 1e-6)
        l[..., BOS] = float("-inf")
        if top_k > 0:
            kth = l.topk(top_k, dim=-1).values[:, -1:]
            l = l.masked_fill(l < kth, float("-inf"))
        lp = torch.log_softmax(l, dim=-1)
        y = torch.multinomial(lp.exp(), 1, generator=generator).squeeze(-1)
        ys.append(y)
        logmus.append(lp.gather(-1, y[:, None]).squeeze(-1))
        out, st = model.consume(y[:, None], st)
        cur = out[:, -1]

    y = torch.stack(ys, 1)
    logmu = torch.stack(logmus, 1)

    # Reward: is the complete product the identity?
    p = targets[:, -1]
    for i in range(n_new):
        p = task.compose(p, y[:, i])
    reward = (p == task.identity).float()
    model.train()
    return {"x": x, "y": y, "logmu": logmu, "reward": reward,
            "boundary": boundary, "prompt_len": prompt_len}


def replay(model, batch, mode="exact"):
    """Return current-policy log p_Theta(y_i | c, y_<i) for every action.

    The only thing that differs between modes is which state the response is
    replayed from, and whether the prompt recurrence carries gradient.
    """
    x, y = batch["x"], batch["y"]
    B, N = y.shape
    device = x.device
    dtype = model.embed.weight.dtype

    if mode == "stale":
        # Sec. 5.4: caches built under older weights "generally cease to be
        # current-policy values". Reuse them anyway, which is the mistake.
        st = batch["boundary"].detach()
        logits_prev = model.readout(st.s)
    elif mode == "reset":
        # Pretend the serving split is a computation boundary: throw H_T away.
        st = model.init_state(B, device, dtype)
        logits_prev = model.readout(st.s)
    else:
        ctx = torch.no_grad() if mode == "nograd" else torch.enable_grad()
        with ctx:
            st0 = model.init_state(B, device, dtype)
            logits, st = model.consume(x, st0)
        if mode == "nograd":
            st = st.detach()
            logits = logits.detach()
        logits_prev = logits[:, -1]

    logps = []
    for i in range(N):
        lp = masked_logprobs(logits_prev)
        logps.append(lp.gather(-1, y[:, i:i + 1]).squeeze(-1))
        out, st = model.consume(y[:, i:i + 1], st)
        logits_prev = out[:, -1]
    return torch.stack(logps, 1)


# ------------------------------------------------------------------ C6 checks

def check_replay(model, task, args, device):
    print("=" * 78)
    print("C6  Sec. 5.3 / App. A.4 -- current-policy replay")
    print("=" * 78)
    g = torch.Generator(device=device).manual_seed(0)
    batch = rollout(model, task, args.batch, args.prompt_len, args.n_new, device,
                    temperature=1.0, top_k=0, generator=g)
    print(f"  batch={args.batch} prompt_len={args.prompt_len} n_new={args.n_new} "
          f"reward mean={batch['reward'].mean():.4f} (chance = {1/ORDER:.4f})")

    print("\n  (a) Replay at the SAME parameters that produced the rollout.")
    print("      Sec. 5.3 predicts r_i = 1 exactly for the reference schedule.\n")
    print(f"      {'schedule':<10s} {'mean r':>10s} {'max |r-1|':>12s} "
          f"{'max |log r|':>13s}")
    out = {}
    for mode in ("exact", "nograd", "stale", "reset"):
        with torch.no_grad():
            lp = replay(model, batch, mode)
        logr = lp - batch["logmu"]
        r = logr.exp()
        out[mode] = {"mean_r": r.mean().item(),
                     "max_abs_r_minus_1": (r - 1).abs().max().item(),
                     "max_abs_logr": logr.abs().max().item()}
        print(f"      {mode:<10s} {r.mean().item():>10.6f} "
              f"{(r-1).abs().max().item():>12.3e} {logr.abs().max().item():>13.3e}")

    print("\n  (b) Replay AFTER one optimizer step, i.e. the ordinary training case.")
    print("      Now every ratio moves; the question is which one is correct.")
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    lp = replay(model, batch, "exact")
    adv = batch["reward"] - batch["reward"].mean()
    (-(adv[:, None] * lp).mean()).backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

    with torch.no_grad():
        ref = replay(model, batch, "exact")
        print(f"\n      {'schedule':<10s} {'mean |log r|':>14s} "
              f"{'max |log p - log p_exact|':>28s}")
        for mode in ("exact", "nograd", "stale", "reset"):
            lp = replay(model, batch, mode)
            d = (lp - ref).abs().max().item()
            out[mode]["post_update_mean_abs_logr"] = (lp - batch["logmu"]).abs().mean().item()
            out[mode]["post_update_err_vs_exact"] = d
            print(f"      {mode:<10s} "
                  f"{(lp - batch['logmu']).abs().mean().item():>14.6f} {d:>28.3e}")

    print("\n  (c) Gradient paths. nograd matches exact's FORWARD values; Sec. 5.4")
    print("      says it still omits prompt-state derivatives.\n")
    grads = {}
    for mode in ("exact", "nograd"):
        model.zero_grad(set_to_none=True)
        lp = replay(model, batch, mode)
        (-(adv[:, None] * lp).mean()).backward()
        grads[mode] = torch.cat([p.grad.reshape(-1) for p in model.parameters()
                                 if p.grad is not None])
    ge, gn = grads["exact"], grads["nograd"]
    cos = F.cosine_similarity(ge[None], gn[None]).item()
    print(f"      ||g_exact|| = {ge.norm():.6e}   ||g_nograd|| = {gn.norm():.6e}")
    print(f"      cosine(g_exact, g_nograd) = {cos:.6f}   "
          f"norm ratio = {(gn.norm()/ge.norm()).item():.4f}")
    out["grad"] = {"cos": cos, "norm_exact": ge.norm().item(),
                   "norm_nograd": gn.norm().item()}

    print("\n  (d) Sec. 5.3 on truncated sampling: 'Top-k or top-p behavior sampling")
    print("      generally violates [absolute continuity] for an untruncated softmax")
    print("      target.'  Sample with top-k and replay against the raw policy.\n")
    b2 = rollout(model, task, args.batch, args.prompt_len, args.n_new, device,
                 temperature=1.0, top_k=5, generator=g)
    with torch.no_grad():
        lp = replay(model, b2, "exact")
    logr = lp - b2["logmu"]
    print(f"      top-k=5 sampling, raw-policy target: mean r = {logr.exp().mean():.6f}, "
          f"max |log r| = {logr.abs().max():.4f}")
    print(f"      (r = 1 fails not because replay is wrong but because mu != p_Theta;")
    print(f"       the target distribution must be replaced consistently, not patched.)")
    out["topk_mean_r"] = logr.exp().mean().item()

    ok = (out["exact"]["max_abs_r_minus_1"] < 1e-4
          and out["stale"]["max_abs_r_minus_1"] > 1e-3
          and out["reset"]["max_abs_r_minus_1"] > 1e-3)
    print(f"\n  VERDICT: {'CONFIRMED' if ok else 'REFUTED'} -- exact replay reproduces the "
          f"sampler ({out['exact']['max_abs_r_minus_1']:.2e}); "
          f"stale ({out['stale']['max_abs_r_minus_1']:.2e}) and "
          f"reset ({out['reset']['max_abs_r_minus_1']:.2e}) do not.")
    out["verdict"] = bool(ok)
    return out


# ------------------------------------------------------------- GRPO training

def grpo(model, task, args, device, mode="exact"):
    """Group-relative policy gradient. Eq. (5.4) with a group-mean baseline.

    Sec. 5.3: "No particular reward function, clipping rule, or KL regularizer is
    required by the architecture."  So this is deliberately the plainest estimator
    that isolates the replay schedule -- the independent variable is `mode`.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=args.rl_lr, betas=(0.9, 0.95))
    g = torch.Generator(device=device).manual_seed(args.seed)
    hist = []
    for step in range(args.rl_steps):
        batch = rollout(model, task, args.batch, args.prompt_len, args.n_new,
                        device, temperature=args.temperature, generator=g)
        R = batch["reward"].view(-1, args.group_size)
        adv = (R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True) + 1e-6)
        adv = adv.reshape(-1)
        lp = replay(model, batch, mode)
        loss = -(adv[:, None] * lp).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 5 == 0 or step == args.rl_steps - 1:
            hist.append({"step": step, "reward": batch["reward"].mean().item(),
                         "loss": loss.item()})
            print(f"    [{mode}] step {step:4d}  reward {batch['reward'].mean():.4f}",
                  flush=True)
    return hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--prompt-len", type=int, default=16)
    p.add_argument("--n-new", type=int, default=2)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rl-steps", type=int, default=0)
    p.add_argument("--rl-lr", type=float, default=3e-4)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--modes", default="exact,stale,reset")
    p.add_argument("--out", default="results/c6_replay.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    cfg = RLTConfig(vocab_size=VOCAB, d_model=args.d_model, n_heads=4,
                    L_E=args.layers, L_D=args.layers, d_ff=4 * args.d_model,
                    W=args.window, alpha=1.0, tied=True, max_position=1024)
    task = GroupTask("a5", device=device)
    model = RLT(cfg).to(device)
    res = check_replay(model, task, args, device)

    if args.rl_steps:
        print("\n" + "=" * 78)
        print("Downstream: does the wrong replay schedule actually hurt?")
        print("=" * 78)
        res["rl"] = {}
        for mode in args.modes.split(","):
            torch.manual_seed(args.seed)
            m = RLT(cfg).to(device)
            print(f"  mode = {mode}")
            res["rl"][mode] = grpo(m, task, args, device, mode)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
