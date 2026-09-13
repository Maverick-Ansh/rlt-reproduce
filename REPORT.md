# Reproducing *Recurrent Looped Transformer*

Yifan Zhang, *Recurrent Looped Transformer*, 12 September 2026.
Reproduction on a single Colab Tesla T4 (16 GB, compute capability 7.5, 2 vCPUs).

---

## 0. What "reproducing" means for this paper

The paper reports no experiments, and is explicit about it:

> "The report develops these mechanisms; it does not report measured efficiency or
> scaling results." (§1)

> "realized reasoning quality, hardware efficiency, and scaling behavior require
> future validation." (§8)

There is therefore no headline number to hit. What the paper does contain is a
tightly specified computation, two propositions, several explicit
*non*-equivalences, and one empirical question it raises and then declines to
answer. This reproduction splits along that seam:

- **C1–C7** turn the paper's assertions into numerical tests. These need no
  training: either they hold to floating-point noise or the implementation is
  wrong. They are the part that can *falsify the specification itself*.
- **E1–E4** run the experiment §3.3 defers, at a scale that fits one T4.

Full claim statements are in [`CLAIMS.md`](CLAIMS.md).

---

## 1. Exactness results

All measurements on an untrained model, `d_model = 64`, `L_E = L_D = 3`, `W = 4`,
fp32, Tesla T4.

Every tolerance is stated against a **measured** noise floor: the same computation
run with a padded batch, which changes cuBLAS tiling and therefore reduction order.
That floor is **2.384e-07**. "Agrees" below always means "agrees to within a small
multiple of that", never a hand-picked epsilon.

| Claim | Source | Result | Verdict |
|---|---|---|---|
| C1 invariance to the serving split | Prop. 3.1 | worst over all 24 split points: **5.25e-07** (2.2x noise floor); parallel prefill vs fully incremental: 4.02e-07 | **CONFIRMED** |
| C2 causality of `H_t` | Prop. B.1 | perturbing `x_{>t}` changes `s_{j<t}` by **exactly 0.0** (bitwise); same-position sanity check moves 6.9e-01 | **CONFIRMED** |
| C3 parallel SWA is not the recurrence | §2.4 | one parallel sweep is off by **7.8e-01**; K sweeps fix **exactly K** positions; at `alpha = 0` the parallel pass is exact (1.5e-07) | **CONFIRMED** |
| C4 detaching only `s_t` is not a truncation | App. C | `‖dL/ds_star‖`: full 1.294e-01, detach-`s` **3.148e-02**, detach-KV 1.006e-01, complete detach **exactly 0.0** | **CONFIRMED** |
| C5 unbounded temporal depth | §3.3 | `‖d s_t / d s_star‖` non-zero out to t = 128; at `alpha = 0` exactly 0 at every t | **CONFIRMED** |
| C7 fixed-slot kernel is the same model | §4.2 | max logit difference 7.15e-07 (relative 5.3e-07); gradient cosine **1.0000000000** | **CONFIRMED** |

### C3 is the most informative of these

The paper only says a parallel pass is "not generally equivalent". The sharper
question is *how* inequivalent, and the answer turns out to be exact:

| parallel sweeps K | positions exact | max error |
|---|---|---|
| 1 | 1 | 7.824e-01 |
| 2 | 2 | 6.266e-01 |
| 3 | 3 | 6.461e-01 |
| 4 | 4 | 5.717e-01 |
| 8 | 8 | 3.539e-01 |

A parallel pass cannot form `u_t` without `s_{t-1}`, which is the decoder's own
output, so the only way to parallelise is to guess `s` and iterate. Each Jacobi
sweep makes exactly one more position exact — never two. Recovering a length-S
sequence costs S sweeps, which is the recurrence again. This is the quantitative
content of §4.1's "No exact parallel scan for the general nonlinear decoder is
assumed", and it is measured rather than assumed.

The control matters as much as the result: at `alpha = 0` a single parallel sweep
is exact at **all** positions. So the non-equivalence is caused specifically by
Eq. (2.11)'s feedback term and not by the sliding window, the encoder memory, or an
implementation artefact.

### C5 measures the hedge, not the claim

The block count along the state path (`t · L_D`) is true by construction and
proving it proves nothing. The interesting quantity is the one §3.3 puts at risk:

> "Gates, contraction, and learned projections may suppress the practical
> contribution of long paths; structural depth alone is not a reasoning guarantee."

and App. B: *"Normalization alone does not bound products of these Jacobians."*

Measured at initialisation:

| t | decoder blocks on the state path | `‖d(r·s_t)/d s_star‖` |
|---|---|---|
| 1 | 3 | 1.782e+01 |
| 2 | 6 | 1.122e+01 |
| 4 | 12 | 6.339e+00 |
| 8 | 24 | 4.937e+00 |
| 16 | 48 | 1.743e+00 |
| 32 | 96 | 1.492e+00 |
| 64 | 192 | 1.629e+00 |
| 128 | 384 | 1.895e+00 |

The Jacobian product **falls by about 10x and then stops falling**. From t = 32 to
t = 128 — a fourfold increase in path length, 96 to 384 composed decoder blocks —
the sensitivity does not decay further; it drifts slightly upward. At
initialisation this architecture is neither vanishing nor exploding along the
temporal path. That is a real, if modest, point in the paper's favour, and it is
the kind of number §3.3 says is needed.

---

## 2. What broke

### 2.1 The evaluation gate caught its own bug first

`check_task.py` fits an oracle shortcut policy — the best possible lookup table
from the last *k* input tokens to the label — and reports its accuracy, to prove
the task cannot be solved shallowly. The first version fit and scored that table on
the same data, and reported:

```
last 3 input token(s)        : 0.4038
```

which would have meant 40% of the task was reachable from a 3-token window and the
whole A5/Z60 contrast was measuring nothing. It was an artefact: 60³ = 216,000 keys
against ~508,000 samples is about two samples per key, and the argmax of two
samples reproduces the data it was fitted on. With a held-out split:

| shortcut policy | fit-on-eval (wrong) | held-out (correct) |
|---|---|---|
| last 1 input token | 0.0322 | 0.0319 |
| last 2 input tokens | 0.0477 | 0.0185 |
| last 3 input tokens | **0.4038** | **0.0168** |
| position index only | 0.0199 | 0.0165 |

Chance is 0.0167. The corrected numbers are exactly where group theory says they
must be: the prefix product is uniform given any bounded suffix of inputs. The
residual 0.0319 at k = 1 is also fully accounted for — at t = 1 the prefix product
*is* g₁, so one position in 64 is free: 1/64 + 63/64 × 0.0167 = 0.0318.

The lesson generalises past this repository: a shortcut check fitted on its own
evaluation data measures the estimator's variance, not the task's structure, and it
fails in the direction that makes you abandon a perfectly good experiment.

### 2.2 The reproduction was 12 hours of GPU before it was 1

The reference implementation — growing KV caches, one Python call per token per
layer — ran at **1.5 s/step** at length 64. The full grid would have been about
twelve hours, which does not fit a Colab session.

The first timing measurement was also wrong in a way worth recording: it divided
total wall time by training steps, but total wall time included evaluation at
lengths 16→256, and the length-256 evaluation alone is 1024 sequential decode
steps. It reported 1.9 s/step for a loop actually running at 1.5 s/step, and
attributed a 25% error to the wrong component.

The fix came from the paper. App. B suggests, for the gradient analysis, "a
fixed-slot representation of the decoder cache, with validity masks during
warm-up". Implemented as an execution strategy rather than an analytical device,
that makes every tensor shape constant — the SWA cache is always `[B, H, W-1, Dh]`
and cross-attention always reads the full encoder memory behind a mask — so one
compiled graph serves every position instead of a fresh one per token.

That path reads tensor entries it must not use, which is exactly what §4.2 warns
about: *"a faster kernel that reads future entries changes the model."* Hence C7,
which checks it rather than trusting it: forward values agree to 7.2e-07 and the
gradient cosine is 1.0000000000.

| implementation | ms / fwd+bwd (len 32, batch 192) |
|---|---|
| reference, growing caches | 742.8 |
| fixed-slot | 703.5 (1.06x) |
| fixed-slot + `torch.compile` | 320.9 (**2.31x**) |

This is a number about *this implementation*, not about the paper. §4 makes no
measured speedup claim and neither does this repository.

### 2.3 "The GPU is idle, so run two jobs" was wrong

The decoder loop leaves the GPU at 0–5% utilisation, so two concurrent training
processes looked free. This Colab instance has **2 vCPUs**, and the loop is bound by
Python and kernel-launch overhead on the CPU, not by the GPU. Two processes plus
their Inductor compile workers produced a load average of 3.3 on 2 cores and made
no training progress for the entire time they were observed. Reverted to one
worker.

A related cost, also from the 2-vCPU limit: every evaluation length is a new tensor
shape and triggered a fresh Inductor compile. Evaluation now runs the uncompiled
fixed-slot path — legitimate precisely because C7 establishes the two paths are the
same computation.

---

## 3. How it was resized

| | Paper | Here | Why this still tests the claim |
|---|---|---|---|
| Depth | `L_E = L_D = 48` (§1, Fig. 2) | `L_E = L_D = 2` (sweep), 3 (checks) | C1–C7 are exact statements, independent of depth. For E1, *smaller* depth makes the test sharper: with 4 blocks/token and length 32, log₂32 = 5 > 4, so a fixed-depth model provably cannot run a parallel scan over the word. |
| Width | unspecified | `d_model = 256` | Capacity is held identical across arms; it is not the variable under test. |
| Objective | next-token, Eq. (5.1)/(5.2) | Eq. (2.6) readout on a supervised per-position label | The state transition Eq. (2.4)–(2.5) is untouched, and that is what E1 is about. Feeding running products back as tokens would make the task one-step trivial (`p_t = p_{t-1} · g_t`) and destroy the depth requirement being measured. |
| Data | natural language | A5 / Z60 prefix products | Exact ground truth, no label noise, chance and ceiling known in closed form, and a matched control that differs in exactly one property. |
| Precision | unspecified | fp32 | T4 is compute capability 7.5, so fp16+GradScaler is the usual choice, but these models are launch-bound rather than FLOP-bound and fp16 buys nothing. fp32 also keeps the C1–C7 tolerances meaningful. |
| Hardware co-design (§4) | — | **not tested** | §4 is explicitly "implementation targets, not measured speedups". |

### The A5 / Z60 control

Both groups have exactly 60 elements, so vocabulary size, label entropy, chance
accuracy (1/60) and sequence statistics are identical by construction. They differ
in one property: Z60 is abelian and its prefix product is a running sum mod 60,
computable in constant depth; A5 is the smallest non-abelian simple group and its
word problem is NC1-complete, so no constant-depth circuit computes it.

A gain from recurrence on A5 but not on Z60 is a **depth** effect. A gain on both is
a capacity effect, and E1 as stated would be refuted.

---

## 4. Empirical results (E1–E4)

*Pending — sweep running.*

---

## 5. Current-policy replay (C6)

*Pending.*

---

## 6. What was not tested

To be completed with the results sections. Known exclusions so far:

- **Every hardware claim in §4.** Batching across sequences, kernel fusion, memory
  residency, and checkpoint placement are untested. The paper claims no measured
  speedup for them and neither does this work.
- **Language modelling quality.** Eq. (5.1) pretraining and Eq. (5.2) SFT masking
  semantics are implemented and exercised, but no model was trained on text, so
  nothing here speaks to whether RLT is a good language model.
- **Scale.** Every conclusion is at `d_model = 256` and ≤ 3 layers per stack. The
  paper's illustrative configuration is 48+48. Depth-dependent effects, including
  whether the C5 Jacobian plateau survives at 48 layers, are untested.
- **Multi-turn cache semantics (§6)** beyond split invariance.
- **`G = L_D` memory groups.** Only `G = 1` (shared memory) was run.
