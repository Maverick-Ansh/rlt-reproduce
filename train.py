"""Train one configuration on a group word problem and write a JSON result.

Data is sampled fresh every step from an exact generator, so there is no train set
to overfit and no held-out gap to interpret. Whatever separates the arms is
expressivity or optimisation, not memorisation -- which is the point, since E1 is a
claim about computational depth.

    python train.py --group a5 --arch rlt --alpha 1.0 --seed 0 --tag a5_rlt_s0
"""
import argparse
import json
import math
import os
import time

import torch
import torch._dynamo
import torch.nn.functional as F

torch._dynamo.config.cache_size_limit = 32

from rlt import RLTConfig, RLT, PlainCausalTransformer
from rlt.tasks import GroupTask, VOCAB, chance_accuracy


def build(args, device):
    cfg = RLTConfig(
        vocab_size=VOCAB, d_model=args.d_model, n_heads=args.n_heads,
        L_E=args.layers, L_D=args.layers, d_ff=4 * args.d_model,
        W=args.window, G=1, alpha=args.alpha, tied=args.tied,
        max_position=args.max_position,
    )
    torch.manual_seed(args.seed)
    model = (PlainCausalTransformer(cfg) if args.arch == "plain" else RLT(cfg))
    return cfg, model.to(device)


@torch.no_grad()
def evaluate(model, task, length, batch, device, chunks=2):
    """Accuracy over fresh samples. Also reports accuracy at the FINAL position,
    which is the one requiring the deepest composition."""
    model.eval()
    tot = cor = 0
    last_tot = last_cor = 0
    for _ in range(chunks):
        x, y = task.sample(batch, length)
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        pred = logits[:, 1:].argmax(-1)
        cor += int((pred == y).sum()); tot += y.numel()
        last_cor += int((pred[:, -1] == y[:, -1]).sum()); last_tot += y.shape[0]
    model.train()
    return cor / tot, last_cor / last_tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", default="a5", choices=["a5", "z60"])
    p.add_argument("--arch", default="rlt", choices=["rlt", "plain"])
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--tied", type=int, default=1)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--length", type=int, default=64)
    p.add_argument("--batch", type=int, default=192)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-position", type=int, default=1024)
    p.add_argument("--eval-lengths", default="16,32,64,128,256")
    p.add_argument("--eval-batch", type=int, default=256)
    p.add_argument("--out", default="results")
    p.add_argument("--tag", default=None)
    p.add_argument("--static", type=int, default=1,
                   help="fixed-slot recurrence; verified identical in smoke_static.py")
    p.add_argument("--compile", type=int, default=0)
    args = p.parse_args()
    args.tied = bool(args.tied)

    tag = args.tag or (f"{args.group}_{args.arch}_a{args.alpha}_"
                       f"{'tied' if args.tied else 'untied'}_L{args.length}_s{args.seed}")
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, tag + ".json")
    if os.path.exists(path):
        print(f"[skip] {path} exists")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    task = GroupTask(args.group, device=device)
    cfg, model = build(args, device)
    n_par = model.n_params()
    if args.arch == "rlt":
        model.use_static = bool(args.static)
        if args.compile:
            model.enable_compile()

    decay = [p_ for n, p_ in model.named_parameters() if p_.dim() >= 2]
    nodecay = [p_ for n, p_ in model.named_parameters() if p_.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.wd},
         {"params": nodecay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))

    def lr_at(i):
        if i < args.warmup:
            return args.lr * (i + 1) / args.warmup
        prog = (i - args.warmup) / max(1, args.steps - args.warmup)
        return 0.1 * args.lr + 0.9 * args.lr * 0.5 * (1 + math.cos(math.pi * prog))

    print(f"[{tag}] params={n_par:,} device={device} "
          f"blocks/token={cfg.blocks_per_token} chance={chance_accuracy():.4f}")

    hist, t0 = [], time.time()
    model.train()
    for i in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(i)
        x, y = task.sample(args.batch, args.length)
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits[:, 1:].reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        if i % 100 == 0 or i == args.steps - 1:
            with torch.no_grad():
                acc = (logits[:, 1:].argmax(-1) == y).float().mean().item()
            hist.append({"step": i, "loss": loss.item(), "acc": acc,
                         "gnorm": float(gn), "lr": lr_at(i)})
            print(f"  step {i:5d}  loss {loss.item():.4f}  acc {acc:.4f}  "
                  f"|g| {float(gn):.2f}  {time.time() - t0:.0f}s", flush=True)

    # Evaluation runs the UNCOMPILED fixed-slot path. Each eval length is a new
    # shape, and on a 2-vCPU box a fresh Inductor compile per length costs more
    # than the whole evaluation saves. The two paths are checked identical in
    # smoke_static.py, so this changes cost and not results.
    model._compiled_step = None
    evals = {}
    for L in [int(v) for v in args.eval_lengths.split(",")]:
        if L + 1 > cfg.max_position:
            continue
        a, last = evaluate(model, task, L, args.eval_batch, device)
        evals[str(L)] = {"acc": a, "acc_final_pos": last}
        print(f"  eval L={L:4d}  acc {a:.4f}  final-position acc {last:.4f}")

    out = {"tag": tag, "args": vars(args), "params": n_par,
           "blocks_per_token": cfg.blocks_per_token, "chance": chance_accuracy(),
           "history": hist, "eval": evals, "wall_s": time.time() - t0}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{tag}] wrote {path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
