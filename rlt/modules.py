"""Primitive layers shared by the encoder and the recurrent decoder.

Attention is written out with explicit matmuls rather than
``F.scaled_dot_product_attention`` on purpose.  Claim C1 (Prop. 3.1) compares a
token-parallel encoder schedule against an incremental one, and the honest
comparison needs both schedules to run *the same* arithmetic; a fused kernel that
switches algorithm based on sequence length would put a floor under the measured
difference that has nothing to do with the model.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


class RoPE(nn.Module):
    """Rotary positions.

    The paper keeps positional handling abstract -- Eq. (2.12)/(2.13) say only that
    "query/key maps include their respective normalizations and positional
    transformations", and Sec. 2.5 requires "Position metadata follows the same
    convention during prefill, sampling, and replay."  That last sentence is the
    load-bearing one: the *absolute* position index t is threaded through every
    execution mode here, so prefill and step decoding cannot silently disagree.
    """

    def __init__(self, d_head: int, base: float = 10000.0, max_position: int = 4096):
        super().__init__()
        assert d_head % 2 == 0
        inv = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        t = torch.arange(max_position).float()
        freqs = torch.outer(t, inv)                      # [P, d_head/2]
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x, pos):
        """x: [B, H, T, Dh]; pos: [T] absolute positions."""
        cos = self.cos[pos].to(x.dtype)[None, None]      # [1, 1, T, Dh/2]
        sin = self.sin[pos].to(x.dtype)[None, None]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        o1 = x1 * cos - x2 * sin
        o2 = x1 * sin + x2 * cos
        out = torch.empty_like(x)
        out[..., 0::2] = o1
        out[..., 1::2] = o2
        return out


class FFN(nn.Module):
    """SwiGLU feed-forward. Eq. (2.16): z = a + FFN(RMSNorm(a))."""

    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d, d_ff, bias=False)
        self.w_up = nn.Linear(d, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class AttentionCore(nn.Module):
    """Q/K/V/O projections only -- no masking policy, no cache policy.

    Sec. 2.6 ties this object between encoder self-attention at layer l and decoder
    SWA at layer l: "Encoder self-attention at layer l and decoder SWA at layer l
    share compatible query, key, value, and output projection matrices."  The
    *wiring* differs (causal full context vs. a bounded decoder-derived window),
    which is why the wiring lives in the callers and only the matrices live here:
    "This is parameter reuse with different attention wiring, not activation
    copying."
    """

    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def split(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,T,Dh]

    def merge(self, x):
        B, H, T, Dh = x.shape
        return x.transpose(1, 2).reshape(B, T, H * Dh)


def attend(q, k, v, mask=None):
    """Explicit scaled dot-product attention.

    q: [B, H, Tq, Dh]; k, v: [B, H, Tk, Dh]; mask: [Tq, Tk] boolean, True = keep.
    """
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    w = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(w, v)


def causal_mask(T: int, device) -> torch.Tensor:
    return torch.ones(T, T, dtype=torch.bool, device=device).tril()


def sliding_window_mask(T: int, W: int, device) -> torch.Tensor:
    """Sec. 2.5, Eq. (2.14): position t attends to max(1, t - W + 1) .. t.

    W counts the current token (Sec. 2.1), so each row keeps W entries.
    """
    idx = torch.arange(T, device=device)
    delta = idx[:, None] - idx[None, :]
    return (delta >= 0) & (delta < W)
