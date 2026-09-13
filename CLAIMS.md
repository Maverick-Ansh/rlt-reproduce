# Falsifiable claims extracted from *Recurrent Looped Transformer* (Zhang, 2026)

The paper reports **no experiments**. It says so itself:

> "The report develops these mechanisms; it does not report measured efficiency or
> scaling results." (§1)

> "realized reasoning quality, hardware efficiency, and scaling behavior require
> future validation." (§8)

So there are no headline numbers to reproduce. What the paper *does* contain is a
set of precise structural statements — two propositions, several stated
non-equivalences, and one explicitly deferred empirical question. Those are the
things that can be falsified, and they are what this repository tests.

Claims are split into **exactness claims** (C*, decided by a numerical test on an
untrained model — either they hold to floating-point noise or the implementation
is wrong) and **empirical claims** (E*, decided by training runs).

---

## Exactness claims

| ID | Claim | Source | Confirming measurement |
|----|-------|--------|------------------------|
| **C1** | *Invariance to the serving split.* Batched encoder prefill followed by recurrent decoder updates gives the same states and next-token distributions as fully incremental processing, for any placement of the prompt/response split. | Prop. 3.1, §3.1 | Max abs logit difference across every split point `T = 1..S`, compared against a same-computation numerical noise floor. |
| **C2** | *Causality.* `H_t = (s_t, C_t^D)` depends only on `x_{1:t}` and the parameters. | Prop. B.1, App. B | Perturb `x_{>t}`; `H_t` must be **bitwise** identical. |
| **C3** | *A standard parallel SWA decoder pass is not equivalent to the recurrence.* "historical decoder KV must be produced by the preceding recurrent updates; a standard parallel SWA decoder pass is not generally equivalent." | §2.4, App. A.1 | Divergence between the recurrent decoder and a naive token-parallel SWA decoder must be **O(1), not O(eps)** — and must collapse to the noise floor exactly when the feedback scale `alpha = 0` (Eq. 2.11). |
| **C4** | *Detaching only `s_t` is not a full truncation.* "Detaching only `s_t` leaves gradient paths through decoder KV and encoder memory." | §5.4, App. C, Eq. (C.1) | Gradient norm reaching prompt-side parameters, under four detach schemes: none / `s` only / KV only / both. `detach-s` must be strictly greater than `detach-both`. |
| **C5** | *Unbounded temporal depth.* The state path to position `t` traverses `t * L_D` decoder blocks, with no fixed architectural bound; per-token block count stays at `L_E + L_D`. | §3.3, Fig. 2 | Block-evaluation counter (exact integer check) **and** measured sensitivity `d s_t / d s_star` remaining non-zero at large `t`, versus a fixed-depth control where the analogous path is bounded. |
| **C6** | *Stale rollout states cannot replace current-policy replay; exact replay removes prompt-boundary mismatch.* "At identical parameters and sampling conventions, trainer and sampler represent the same computation." | §5.3, §5.4, App. A.4, App. C | Importance ratio `r_i` (Eq. 5.5) at **identical** sampler/trainer parameters. Exact replay must give `r_i = 1` to float precision. Stale-state replay and boundary-reset replay must not. |

**C6 is the paper's central engineering claim** and the one with the cleanest test:
if trainer and sampler really do "represent the same computation", then replaying a
rollout at the parameters that produced it must return ratios of exactly 1. Any
deviation is the structural mismatch the paper claims to remove.

---

## Empirical claims — the experiment the paper does not run

The paper advertises "latent reasoning with infinite depth" and then withdraws any
guarantee:

> "Gates, contraction, and learned projections may suppress the practical
> contribution of long paths; structural depth alone is not a reasoning guarantee."
> (§3.3)

> "The architecture makes that path available; learning useful reasoning along it is
> a separate question." (§1)

That open question is E1-E3.

| ID | Claim under test | Confirming measurement |
|----|------------------|------------------------|
| **E1** | Unbounded temporal depth buys real computational power on a task whose difficulty is depth-bound, not memory-bound. | RLT (`alpha > 0`) beats the identical model with `alpha = 0` on A5 prefix products, at matched parameters, FLOPs, and per-token block count. |
| **E2** | The advantage is a depth advantage, not a capacity advantage. | The same gap must **not** appear on Z60, a group with identical vocabulary size and identical sequence statistics but abelian (so the task is shallow-computable). A gap on A5 with no gap on Z60 isolates the mechanism. |
| **E3** | The depth advantage shows up as length extrapolation. | Train at length 32, evaluate at 32/64/128/256. Fixed-depth models are expected to learn length-limited shortcuts; a genuinely recurrent state should not be length-limited. |
| **E4** | The tied configuration of §2.6 costs nothing. | Accuracy of tied vs untied `E_theta, D_phi` at matched depth. |

### Why A5 versus Z60

Both are groups with exactly **60 elements**, so vocabulary size, label entropy,
chance accuracy (1/60 = 1.67%) and sequence statistics are identical by
construction. They differ in one property only:

- **Z60** is abelian, hence solvable. The prefix product is a running sum mod 60 —
  computable in constant depth by a single attention layer.
- **A5** is the smallest non-abelian simple group; its word problem is
  **NC1-complete**. No constant-depth circuit computes it, and a fixed-depth
  transformer can only learn length-limited shortcuts.

This is the "two arms that differ by exactly one thing" the reproduction protocol
asks for, and it is a *matched control*: if `alpha > 0` helps on both groups, the
gain is capacity, not depth, and E1 is refuted as stated.

### Evaluation bracket (measured before any sweep)

- **Floor**: uniform guessing = 1/60 = 1.67%.
- **Ceiling**: 100%. The target is a deterministic, exactly known function of the
  input — there is no label noise and no annotator disagreement.
- **Shortcut check**: inputs are drawn uniformly from the group, so prefix products
  are uniform over the group and the best constant-output policy scores 1/60.
  Verified empirically in `check_task.py` rather than assumed.

The metric is direct accuracy on exact ground truth. No learned probe stands
between the model and the number.

---

## Not tested here

- Any hardware claim. §4 is entirely about "implementation targets, not measured
  speedups"; this reproduction runs one T4 and makes no throughput claim.
- Language-model quality at scale. The pretraining (Eq. 5.1) and SFT (Eq. 5.2)
  objectives are implemented and unit-tested for masking and state-update
  semantics, but not trained to a quality number.
- Multi-turn serving cache semantics (§6) beyond the split-invariance test.
