# Recurrent Looped Transformer — a reproduction

A from-scratch implementation of **Recurrent Looped Transformer** (Yifan Zhang,
September 2026), plus the experiments the paper does not contain.

The paper is a specification, not an empirical report. It says so twice:

> "The report develops these mechanisms; it does not report measured efficiency or
> scaling results." (§1)

> "realized reasoning quality, hardware efficiency, and scaling behavior require
> future validation." (§8)

So "reproducing" it means two different jobs:

1. **Verify what the paper asserts.** It contains two propositions and several
   precise non-equivalences. Each becomes a numerical test that either passes to
   floating-point noise or reveals a bug. These are C1–C7 in
   [`CLAIMS.md`](CLAIMS.md).
2. **Run the experiment it defers.** The paper offers "latent reasoning with
   infinite depth" and then withdraws the guarantee — *"structural depth alone is
   not a reasoning guarantee"* (§3.3). E1–E4 decide whether the unbounded temporal
   depth buys measurable computational power, on hardware that fits in a free
   Colab T4.

## What is implemented

| Paper | Here |
|---|---|
| Causal encoder + memory, Eq. (2.1)–(2.2) | `rlt/model.py: RLT.encode_parallel, project_memory` |
| Gated merge, Eq. (2.9)–(2.11) | `RLT.merge` — `alpha` is the feedback scale |
| Decoder block: SWA → cross-attn → FFN, Eq. (2.12)–(2.16) | `DecoderLayer` |
| Complete state `H_t = (s_t, C^D_t)`, Eq. (2.3)–(2.5) | `State`, `RLT.decode_step` |
| Tied configuration, §2.6 | `RLTConfig.tied` |
| Prefill / incremental / chunked schedules, §2.4, App. A.1–A.2 | `RLT.consume(parallel_encode=...)` |
| Fixed-slot cache with validity masks, App. B | `DecoderLayer.forward_step_static` |
| Selective truncation, App. C, Eq. (C.1) | `State.detach(s=, swa=, enc=, mem=)` |
| Current-policy RL replay, §5.3, App. A.4 | `rl.py: replay(mode="exact")` |
| Rejected replay schedules (stale / boundary-reset / no_grad prompt) | `rl.py: replay(mode=...)` |

There is exactly **one** transition function. Prefill, generation, SFT and RL
replay are all loops over it, which is the whole point of §2.3: *"Neither component
of `H_T` is reset at the serving boundary."*

## Running it

```bash
python check_task.py                 # evaluation gate: is the metric able to resolve anything?
python smoke.py --device cuda        # C1-C5, untrained model, ~1 min
python smoke_static.py --compile     # C7: the fast kernel is the same model
python rl.py --rl-steps 0            # C6: replay ratios
python sweep.py --steps 1600 --length 32 --layers 2 --batch 384
python analyze.py
```

## Layout

```
rlt/model.py      the architecture; every module docstring quotes what it implements
rlt/tasks.py      A5 vs Z60 prefix products -- the depth test and its matched control
rlt/baselines.py  plain causal Transformer at matched per-token block count
smoke.py          C1-C5   exactness checks (no training required)
smoke_static.py   C7      optimised kernel equivalence + speed
rl.py             C6      current-policy replay, and GRPO arms
check_task.py     the Phase-4 evaluation gate
train.py sweep.py analyze.py
REPORT.md         results, deviations, and what broke
```

## Results

See [`REPORT.md`](REPORT.md). Read the *What broke* section first — it is the part
that is hardest to get from the paper alone.

## Deviations from the paper, at a glance

The full table with reasoning is in `REPORT.md`. The material ones:

- **Scale.** The paper's illustrative configuration is 48 encoder + 48 decoder
  layers. This runs 2+2 or 3+3 at `d_model = 256` on one T4.
- **Task.** Eq. (2.6)'s readout predicts a supervised per-position label (a group
  prefix product) rather than the next input token. The state transition
  Eq. (2.4)–(2.5) is untouched; feeding the running products back as tokens would
  make the task one-step trivial and destroy the depth requirement being measured.
- **No hardware claim.** §4 is explicitly about "implementation targets, not
  measured speedups", and this repository makes no throughput claim about the
  architecture either. The one speed number reported is about *this* implementation
  and is labelled as such.

## Reference

Yifan Zhang. *Recurrent Looped Transformer.* September 12, 2026.
Project page: https://github.com/yifanzhang-pro/recurrent-looped-tranformer
