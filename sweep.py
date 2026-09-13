"""Run the E1-E4 grid sequentially on one GPU, resumably.

One process per run so a crash cannot poison the next one, and every run is
skipped if its JSON already exists -- a killed sweep resumes instead of
restarting.

    python sweep.py --steps 2500            # the full grid
    python sweep.py --steps 40 --tag-prefix smoke --out results_smoke
"""
import argparse
import itertools
import os
import subprocess
import sys
import time

# (name, arch, alpha, tied) -- the four arms.
#   rlt      : the reference model, Eq. (2.11) feedback on
#   rlt_a0   : IDENTICAL model with alpha = 0. Sec. 2.5's feedback term is the only
#              difference, so this is E1's single-variable control.
#   rlt_untied : Sec. 2.6's untied E_theta, D_phi -- "An untied E_theta, D_phi
#              preserves the complete-state recurrence while removing depth-wise
#              parameter reuse."  This is E4.
#   plain    : ordinary causal Transformer at matched per-token block count, with
#              FULL attention at every layer (a handicap in RLT's favour would be
#              dishonest, so the baseline gets the stronger attention).
ARMS = [
    ("rlt",        "rlt",   1.0, 1),
    ("rlt_nope",   "rlt",   1.0, 1),
    ("rlt_a0",     "rlt",   0.0, 1),
    ("rlt_untied", "rlt",   1.0, 0),
    ("plain",      "plain", 1.0, 1),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--length", type=int, default=64)
    p.add_argument("--batch", type=int, default=192)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--groups", default="a5,z60")
    p.add_argument("--arms", default="rlt,rlt_a0,rlt_untied,plain")
    p.add_argument("--out", default="results")
    p.add_argument("--logs", default="logs")
    p.add_argument("--compile", type=int, default=1)
    p.add_argument("--shard", default="0/1",
                   help="i/n -- take every n-th job. The decoder loop leaves the GPU "
                        "mostly idle between tiny kernels, so two workers overlap "
                        "almost for free.")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    groups = args.groups.split(",")
    keep = set(args.arms.split(","))
    arms = [a for a in ARMS if a[0] in keep]

    jobs = list(itertools.product(groups, arms, seeds))
    si, sn = (int(v) for v in args.shard.split("/"))
    jobs = [j for k, j in enumerate(jobs) if k % sn == si]
    print(f"{len(jobs)} runs: groups={groups} arms={[a[0] for a in arms]} seeds={seeds}",
          flush=True)
    t0 = time.time()
    for i, (group, (name, arch, alpha, tied), seed) in enumerate(jobs):
        tag = f"{group}_{name}_s{seed}"
        if os.path.exists(os.path.join(args.out, tag + ".json")):
            print(f"[{i+1}/{len(jobs)}] skip {tag}", flush=True)
            continue
        cmd = [sys.executable, "train.py", "--group", group, "--arch", arch,
               "--alpha", str(alpha), "--tied", str(tied), "--seed", str(seed),
               "--steps", str(args.steps), "--length", str(args.length),
               "--batch", str(args.batch), "--layers", str(args.layers),
               "--d-model", str(args.d_model), "--out", args.out, "--tag", tag,
               "--compile", str(args.compile),
               "--rope", "0" if name.endswith("_nope") else "1"]
        print(f"[{i+1}/{len(jobs)}] {tag}  ({time.time()-t0:.0f}s elapsed)", flush=True)
        with open(os.path.join(args.logs, tag + ".log"), "w") as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        if r.returncode:
            print(f"    FAILED rc={r.returncode}; see {args.logs}/{tag}.log", flush=True)
    print(f"sweep done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
