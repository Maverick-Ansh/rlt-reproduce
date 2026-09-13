"""Group word problems: a depth-bound task and a matched shallow control.

The paper offers the architecture and withdraws the guarantee:

    "Gates, contraction, and learned projections may suppress the practical
     contribution of long paths; structural depth alone is not a reasoning
     guarantee." (Sec. 3.3)

To decide whether unbounded temporal depth buys anything, the task has to be one
where *depth* is the binding constraint and memory access is not. Prefix products
in a finite group are exactly that, and they come with a free matched control.

    A5  -- the alternating group on 5 letters. 60 elements, non-abelian, and the
           smallest non-abelian simple group. Its word problem is NC1-complete:
           no constant-depth circuit family computes it, so a fixed-depth
           transformer can only learn length-limited shortcuts.

    Z60 -- the cyclic group of order 60. Also 60 elements, also uniform inputs,
           identical label entropy and identical chance accuracy, but abelian.
           The prefix product is a running sum mod 60, which one attention layer
           computes at any length.

Same vocabulary, same sequence statistics, same target distribution. The only
difference is whether the group is solvable. A gain that appears on A5 and not on
Z60 is a depth effect; a gain on both is a capacity effect and refutes E1 as
stated.
"""
from itertools import permutations
from typing import Tuple

import torch

BOS = 60          # vocabulary is {0..59} group elements plus one BOS token
VOCAB = 61
ORDER = 60


def _a5_table() -> torch.Tensor:
    """Cayley table of A5 under composition, as a [60, 60] long tensor."""
    perms = [p for p in permutations(range(5)) if _parity(p) == 0]
    assert len(perms) == 60
    index = {p: i for i, p in enumerate(perms)}
    table = torch.empty(60, 60, dtype=torch.long)
    for i, a in enumerate(perms):
        for j, b in enumerate(perms):
            # (a then b): apply a first, then b.
            table[i, j] = index[tuple(b[a[k]] for k in range(5))]
    return table


def _parity(p) -> int:
    seen, par = [False] * len(p), 0
    for i in range(len(p)):
        if seen[i]:
            continue
        j, sz = i, 0
        while not seen[j]:
            seen[j], j, sz = True, p[j], sz + 1
        par += sz - 1
    return par % 2


def _z60_table() -> torch.Tensor:
    i = torch.arange(60)
    return (i[:, None] + i[None, :]) % 60


class GroupTask:
    """Prefix products in a 60-element group.

    A sample is `[BOS, g_1, ..., g_T]`; the target at the position of `g_t` is the
    running product `p_t = p_{t-1} . g_t`, with `p_0` the identity. The model reads
    the target from Eq. (2.6) applied to `s_t`.

    Deviation from the paper, stated plainly: Eq. (2.6) is used for a supervised
    per-position label rather than for the next input token. The state transition
    Eq. (2.4)-(2.5) is untouched, which is what E1-E3 are about. Feeding the
    running products back in as tokens would make the task one-step trivial
    (`p_t = p_{t-1} . g_t`) and destroy the depth requirement being measured.
    """

    def __init__(self, group: str = "a5", device="cpu"):
        assert group in ("a5", "z60")
        self.group = group
        self.table = (_a5_table() if group == "a5" else _z60_table()).to(device)
        self.identity = 0 if group == "a5" else 0
        self.device = device
        if group == "a5":
            # identity permutation is the first even permutation, index 0
            assert int(self.table[0, 7]) == 7 and int(self.table[7, 0]) == 7

    def sample(self, batch: int, length: int, generator=None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (x [B, T+1] with BOS prepended, targets [B, T])."""
        g = torch.randint(0, ORDER, (batch, length), device=self.device,
                          generator=generator)
        targets = self.prefix_products(g)
        x = torch.cat([torch.full((batch, 1), BOS, dtype=torch.long,
                                  device=self.device), g], dim=1)
        return x, targets

    def prefix_products(self, g: torch.Tensor) -> torch.Tensor:
        """p_t = p_{t-1} . g_t, computed exactly. This is the ground truth: there
        is no label noise and no annotator, so the evaluation ceiling is 100%."""
        B, T = g.shape
        p = torch.full((B,), self.identity, dtype=torch.long, device=g.device)
        out = []
        for t in range(T):
            p = self.table[p, g[:, t]]
            out.append(p)
        return torch.stack(out, dim=1)

    def compose(self, p: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return self.table[p, g]

    def inverse(self, p: torch.Tensor) -> torch.Tensor:
        """Element-wise group inverse, by table lookup."""
        row = self.table[p]                       # [B, 60]
        return (row == self.identity).long().argmax(dim=-1)


def chance_accuracy() -> float:
    """Evaluation floor. Inputs are uniform over the group, so prefix products are
    uniform over the group and the best constant-output policy scores 1/60."""
    return 1.0 / ORDER
