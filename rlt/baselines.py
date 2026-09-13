"""External control: an ordinary causal Transformer at matched depth.

The internal control for E1 is `alpha = 0` inside RLT itself -- same weights, same
wiring, same per-token block count, with Eq. (2.11)'s feedback term switched off.
That is the cleanest single-variable comparison and it is the primary one.

This module supplies a second, cruder reference point: a standard decoder-only
Transformer of depth `L_E + L_D`, so that "each new token evaluates L_E + L_D
blocks" (Sec. 3.2) holds for it too. It answers a different question -- is RLT's
encoder/decoder split itself competitive with a plain stack? -- and guards against
reading an RLT-internal ablation as an absolute statement about the task.

Note the asymmetry, which is deliberate and favours the baseline: this model has
FULL causal attention at every layer, while the RLT decoder has only a
W-token sliding window plus encoder-derived memory. If the baseline still fails
on A5, it is not failing for lack of access to the input.
"""
import torch
import torch.nn as nn

from .config import RLTConfig
from .modules import RMSNorm, RoPE, FFN, AttentionCore, attend, causal_mask


class PlainBlock(nn.Module):
    def __init__(self, cfg: RLTConfig, rope: RoPE):
        super().__init__()
        self.core = AttentionCore(cfg.d_model, cfg.n_heads)
        self.ffn = FFN(cfg.d_model, cfg.d_ff)
        self.norm_attn = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.norm_ffn = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.rope = rope

    def forward(self, x, pos, mask):
        h = self.norm_attn(x)
        c = self.core
        q = self.rope(c.split(c.q_proj(h)), pos)
        k = self.rope(c.split(c.k_proj(h)), pos)
        v = c.split(c.v_proj(h))
        x = x + c.o_proj(c.merge(attend(q, k, v, mask)))
        return x + self.ffn(self.norm_ffn(x))


class PlainCausalTransformer(nn.Module):
    def __init__(self, cfg: RLTConfig):
        super().__init__()
        self.cfg = cfg
        self.depth = cfg.L_E + cfg.L_D
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = (RoPE(cfg.d_head, cfg.rope_base, cfg.max_position)
                     if cfg.use_rope else (lambda x, pos: x))
        self.blocks = nn.ModuleList([PlainBlock(cfg, self.rope) for _ in range(self.depth)])
        self.norm_o = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.W_o = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=self.cfg.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=self.cfg.init_std)

    def forward(self, x, **kw):
        pos = torch.arange(x.shape[1], device=x.device)
        mask = causal_mask(x.shape[1], x.device)
        h = self.embed(x)
        for b in self.blocks:
            h = b(h, pos, mask)
        return self.W_o(self.norm_o(h)), None

    def n_params(self) -> int:
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total
