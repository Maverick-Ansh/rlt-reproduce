"""Recurrent Looped Transformer (Zhang, 2026), Sec. 2.

Every state-bearing object in this file corresponds to something the paper names:

    e_t        encoder representation, Eq. (2.1)
    M_{<=t}    encoder-derived cross-attention memory, Eq. (2.2)
    s_t        recurrent decoder output, Sec. 2.1
    C^D_t      retained key/value projections at every decoder SWA layer, Sec. 2.1
    H_t        the COMPLETE decoder state (s_t, C^D_t), Sec. 2.1

The single most important structural fact, and the one an implementation is most
likely to get quietly wrong, is Sec. 2.3:

    "Neither component of H_T is reset at the serving boundary."

There is therefore exactly one transition function in this file, `decode_step`, and
every execution mode -- prefill, generation, SFT, RL replay -- is a loop over it.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .config import RLTConfig
from .modules import (
    RMSNorm, RoPE, FFN, AttentionCore, attend, causal_mask, sliding_window_mask,
)

KV = Tuple[torch.Tensor, torch.Tensor]


@dataclass
class State:
    """H_t = (s_t, C^D_t) plus the bookkeeping every schedule must agree on.

    App. C: an exact prefix snapshot contains "recurrent output, every decoder SWA
    cache, encoder continuation state, encoder-derived memory, and required
    metadata".  All of it lives here so that a snapshot is a single object and a
    half-restored state is not expressible.
    """
    s: torch.Tensor                  # [B, d]          recurrent output
    swa: List[Optional[KV]]          # per decoder layer, [B,H,n,Dh], n <= W-1
    enc: List[Optional[KV]]          # per encoder layer, [B,H,t,Dh]
    mem: List[Optional[KV]]          # per memory group,  [B,H,t,Dh]
    t: int                           # number of CONSUMED tokens

    def detach(self, s: bool = True, swa: bool = True, enc: bool = True,
               mem: bool = True) -> "State":
        """Selective stop-gradient.

        App. C makes the point this method exists to test: "Detaching only s_t
        leaves possible paths through decoder KV; detaching only decoder KV leaves
        paths through the recurrent output. ... Any truncation scheme must state
        which tensors are detached."  So the scheme is an explicit argument list,
        not a single boolean.
        """
        def d(kvs, flag):
            if not flag:
                return list(kvs)
            return [None if kv is None else (kv[0].detach(), kv[1].detach())
                    for kv in kvs]
        return State(
            s=self.s.detach() if s else self.s,
            swa=d(self.swa, swa), enc=d(self.enc, enc), mem=d(self.mem, mem),
            t=self.t,
        )

    def clone_shallow(self) -> "State":
        return State(s=self.s, swa=list(self.swa), enc=list(self.enc),
                     mem=list(self.mem), t=self.t)


def _append(cache: Optional[KV], k: torch.Tensor, v: torch.Tensor,
            keep: Optional[int]) -> KV:
    """Append current KV, then evict.

    Sec. 2.5: "After the update, retain positions max(1, t - W + 2), ..., t in
    C^D_t; this set is empty for W = 1."  `keep = W - 1`; `keep = None` means an
    unbounded (encoder / memory) cache.
    """
    if cache is not None:
        k = torch.cat([cache[0], k], dim=2)
        v = torch.cat([cache[1], v], dim=2)
    if keep is not None:
        n = k.shape[2]
        if n > keep:
            k, v = k[:, :, n - keep:], v[:, :, n - keep:]
    return (k, v)


class EncoderLayer(nn.Module):
    """A causal encoder block, Eq. (2.1).

    "Positions within each encoder layer can be processed together using a causal
    mask. Encoder layers remain sequential." (Sec. 2.2) -- sequential in DEPTH, not
    in tokens.  Both schedules below are that same causal computation.
    """

    def __init__(self, cfg: RLTConfig, core: AttentionCore, ffn: FFN, rope: RoPE):
        super().__init__()
        self.core, self.ffn, self.rope = core, ffn, rope
        self.norm_attn = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.norm_ffn = RMSNorm(cfg.d_model, cfg.norm_eps)

    def qkv(self, x, pos):
        h = self.norm_attn(x)
        c = self.core
        q = self.rope(c.split(c.q_proj(h)), pos)
        k = self.rope(c.split(c.k_proj(h)), pos)
        v = c.split(c.v_proj(h))
        return q, k, v

    def forward_parallel(self, x, pos, mask):
        q, k, v = self.qkv(x, pos)
        x = x + self.core.o_proj(self.core.merge(attend(q, k, v, mask)))
        return x + self.ffn(self.norm_ffn(x)), (k, v)

    def forward_step(self, x, pos, cache: Optional[KV]):
        """Eq. (2.8): (e_t, C^E_t) = E^step_theta(x_t, C^E_{t-1})."""
        q, k, v = self.qkv(x, pos)
        cache = _append(cache, k, v, keep=None)
        x = x + self.core.o_proj(self.core.merge(attend(q, cache[0], cache[1])))
        return x + self.ffn(self.norm_ffn(x)), cache

    def forward_chunk(self, x, pos, cache: Optional[KV]):
        """Encode several new tokens against an existing prefix cache (Sec. 2.4)."""
        n = x.shape[1]
        q, k, v = self.qkv(x, pos)
        cache = _append(cache, k, v, keep=None)
        total = cache[0].shape[2]
        past = total - n
        i = torch.arange(n, device=x.device)
        j = torch.arange(total, device=x.device)
        mask = j[None, :] <= (i[:, None] + past)
        x = x + self.core.o_proj(self.core.merge(attend(q, cache[0], cache[1], mask)))
        return x + self.ffn(self.norm_ffn(x)), cache


class DecoderLayer(nn.Module):
    """Eq. (2.12)-(2.16): causal SWA, then encoder-memory cross-attention, then FFN.

    When `cfg.tied`, `core` and `ffn` are the *same modules* as encoder layer l
    (Sec. 2.6).  Cross-attention keeps "separate query/output projections and the
    memory projections of Equation (2.2)", so those are always owned here.
    """

    def __init__(self, cfg: RLTConfig, core: AttentionCore, ffn: FFN, rope: RoPE):
        super().__init__()
        self.cfg, self.core, self.ffn, self.rope = cfg, core, ffn, rope
        self.norm_swa = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.norm_xq = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.norm_ffn = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.xq_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.xo_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def swa_qkv(self, z, pos):
        h = self.norm_swa(z)
        c = self.core
        q = self.rope(c.split(c.q_proj(h)), pos)
        k = self.rope(c.split(c.k_proj(h)), pos)
        v = c.split(c.v_proj(h))
        return q, k, v

    def cross(self, b, pos, mem: KV, mask=None):
        """Eq. (2.15). Sec. 4.2: "Cross-attention should read only valid
        encoder-prefix memory ... a faster kernel that reads future entries changes
        the model." """
        c = self.core
        q = self.rope(c.split(self.xq_proj(self.norm_xq(b))), pos)
        return b + self.xo_proj(c.merge(attend(q, mem[0], mem[1], mask)))

    def forward_step(self, z, pos, cache: Optional[KV], mem: KV):
        q, k, v = self.swa_qkv(z, pos)
        # Sec. 2.5: "At each layer, current KV is formed before SWA, using the layer
        # input, so current-position attention introduces no circular dependency."
        if self.cfg.W > 1 and cache is not None:
            kk = torch.cat([cache[0], k], dim=2)
            vv = torch.cat([cache[1], v], dim=2)
        else:
            kk, vv = k, v
        b = z + self.core.o_proj(self.core.merge(attend(q, kk, vv)))   # Eq. (2.14)
        a = self.cross(b, pos, mem)                                    # Eq. (2.15)
        new_cache = _append(cache, k, v, keep=self.cfg.W - 1)
        return a + self.ffn(self.norm_ffn(a)), new_cache               # Eq. (2.16)

    def forward_parallel(self, z, pos, mem: KV, swa_mask, mem_mask):
        """Token-parallel SWA over GIVEN inputs z.

        Correct only when the whole sequence z is already known -- which at l = 0
        requires u_{1:S}, which requires s_{0:S-1}, which is what the recurrence
        computes.  Used only by the Jacobi probe for claim C3 (Sec. 2.4).
        """
        q, k, v = self.swa_qkv(z, pos)
        b = z + self.core.o_proj(self.core.merge(attend(q, k, v, swa_mask)))
        a = self.cross(b, pos, mem, mem_mask)
        return a + self.ffn(self.norm_ffn(a))


class RLT(nn.Module):
    def __init__(self, cfg: RLTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = RoPE(cfg.d_head, cfg.rope_base, cfg.max_position)

        cores = [AttentionCore(cfg.d_model, cfg.n_heads) for _ in range(cfg.L_E)]
        ffns = [FFN(cfg.d_model, cfg.d_ff) for _ in range(cfg.L_E)]
        self.enc_cores = nn.ModuleList(cores)
        self.enc_ffns = nn.ModuleList(ffns)
        self.encoder = nn.ModuleList(
            [EncoderLayer(cfg, cores[l], ffns[l], self.rope) for l in range(cfg.L_E)]
        )
        if cfg.tied:
            # Sec. 2.6: decoder SWA reuses the encoder attention/FFN core at layer l.
            dec_cores, dec_ffns = cores, ffns
        else:
            dec_cores = [AttentionCore(cfg.d_model, cfg.n_heads) for _ in range(cfg.L_D)]
            dec_ffns = [FFN(cfg.d_model, cfg.d_ff) for _ in range(cfg.L_D)]
            self.dec_cores = nn.ModuleList(dec_cores)
            self.dec_ffns = nn.ModuleList(dec_ffns)
        self.decoder = nn.ModuleList(
            [DecoderLayer(cfg, dec_cores[l], dec_ffns[l], self.rope) for l in range(cfg.L_D)]
        )

        # Eq. (2.2) memory projections, one set per group.
        self.mem_norm = nn.ModuleList(
            [RMSNorm(cfg.d_model, cfg.norm_eps) for _ in range(cfg.G)])
        self.mem_k = nn.ModuleList(
            [nn.Linear(cfg.d_model, cfg.d_model, bias=False) for _ in range(cfg.G)])
        self.mem_v = nn.ModuleList(
            [nn.Linear(cfg.d_model, cfg.d_model, bias=False) for _ in range(cfg.G)])

        # Eq. (2.9)-(2.11) merge.
        self.norm_s = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.W_g = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=True)
        self.W_s = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # Eq. (2.3) learned initial state s_star, and Eq. (2.6) readout.
        self.s_star = nn.Parameter(torch.zeros(cfg.d_model))
        self.norm_o = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.W_o = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.block_evals = 0          # instrumentation for claim C5
        self.apply(self._init)
        nn.init.normal_(self.s_star, std=cfg.init_std)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=self.cfg.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=self.cfg.init_std)

    def group_of(self, layer: int) -> int:
        """Sec. 2.2: "Decoder layer l reads group g(l)"."""
        return 0 if self.cfg.G == 1 else layer

    # ---------------------------------------------------------------- encoder

    def encode_parallel(self, x, pos):
        """Eq. (2.1): e_{1:T} = E_theta(x_{1:T}), token-parallel with a causal mask."""
        h = self.embed(x)
        mask = causal_mask(x.shape[1], x.device)
        enc = []
        for layer in self.encoder:
            h, kv = layer.forward_parallel(h, pos, mask)
            enc.append(kv)
            self.block_evals += 1
        return h, enc

    def encode_chunk(self, x, pos, enc):
        h = self.embed(x)
        out = []
        for layer, c in zip(self.encoder, enc):
            h, c = layer.forward_chunk(h, pos, c)
            out.append(c)
            self.block_evals += 1
        return h, out

    def encode_step(self, x, pos, enc):
        """Eq. (2.8): incremental encoder with its continuation cache."""
        h = self.embed(x)
        out = []
        for layer, c in zip(self.encoder, enc):
            h, c = layer.forward_step(h, pos, c)
            out.append(c)
            self.block_evals += 1
        return h, out

    def project_memory(self, e, pos, mem):
        """Eq. (2.2): k^g_t = P^g_K(e_t, t), v^g_t = W^g_V RMSNorm_E(e_t).

        Sec. 2.2: "This global memory depends on encoder representations, not
        decoder states."  Nothing in this method may read s.
        """
        out = []
        split = self.decoder[0].core.split
        for g in range(self.cfg.G):
            h = self.mem_norm[g](e)
            k = self.rope(split(self.mem_k[g](h)), pos)
            v = split(self.mem_v[g](h))
            out.append(_append(mem[g], k, v, keep=None))
        return out

    # ------------------------------------------------------------------ merge

    def merge(self, e_t, s_prev):
        """Eq. (2.9)-(2.11).

            r_{t-1} = RMSNorm_s(s_{t-1})
            g_t     = sigmoid(W_g [e_t ; r_{t-1}] + b_g)
            u_t     = e_t + alpha * g_t (*) W_s r_{t-1}

        alpha = 0 severs the only path by which s_{t-1} reaches u_t.  That is the
        ablation E1 turns on: identical parameters, identical per-token block
        count, no temporal recurrence.
        """
        r = self.norm_s(s_prev)
        g = torch.sigmoid(self.W_g(torch.cat([e_t, r], dim=-1)))
        return e_t + self.cfg.alpha * g * self.W_s(r)

    # ------------------------------------------------------------- transition

    def init_state(self, batch: int, device, dtype=torch.float32) -> State:
        """Eq. (2.3): H_0 = (s_star, empty)."""
        return State(
            s=self.s_star.to(device=device, dtype=dtype).expand(batch, -1).contiguous(),
            swa=[None] * self.cfg.L_D, enc=[None] * self.cfg.L_E,
            mem=[None] * self.cfg.G, t=0,
        )

    def decode_step(self, e_t, st: State, pos, mem=None) -> State:
        """H_t = D_phi(Merge(e_t, s_{t-1}); M_{<=t}, C^D_{t-1}, t).  Eq. (2.4)-(2.5).

        `e_t` is [B, 1, d].  `mem` defaults to `st.mem` and must already include
        position t: Sec. 2.3, "encode it incrementally, append its encoder-derived
        KV, and compute H_{T+1} = F_{T+1}(H_T)".
        """
        mem = st.mem if mem is None else mem
        z = self.merge(e_t, st.s.unsqueeze(1))
        swa = []
        for l, layer in enumerate(self.decoder):
            z, c = layer.forward_step(z, pos, st.swa[l], mem[self.group_of(l)])
            swa.append(c)
            self.block_evals += 1
        return State(s=z.squeeze(1), swa=swa, enc=st.enc, mem=st.mem, t=st.t + 1)

    def readout(self, s):
        """Eq. (2.6): softmax(W_o RMSNorm_o(s_t)) -- returned as logits."""
        return self.W_o(self.norm_o(s))

    # -------------------------------------------------------------- schedules

    def consume(self, x, st: State, parallel_encode: bool = True,
                collect_states: bool = False):
        """Consume tokens x [B, n] from state `st`; return (logits, H_{t+n}).

        `parallel_encode` selects between the two encoder schedules of Sec. 2.4.
        The decoder loop is byte-identical in both, because Prop. 3.1 assumes only
        "mathematically equivalent causal encoder execution" -- claim C1 is the
        test that this implementation actually delivers that.
        """
        B, n = x.shape
        dev = x.device
        pos_all = torch.arange(st.t, st.t + n, device=dev)

        if parallel_encode:
            if st.enc[0] is None:
                e, enc = self.encode_parallel(x, pos_all)
            else:
                e, enc = self.encode_chunk(x, pos_all, st.enc)
            mem = self.project_memory(e, pos_all, st.mem)
        else:
            e, enc, mem = None, st.enc, st.mem

        logits, states = [], []
        s_cur, swa_cur, t0 = st.s, st.swa, st.t
        for i in range(n):
            if parallel_encode:
                e_i = e[:, i:i + 1]
                # Sec. 2.4: "At decoder position t, attention is restricted to
                # M_{<=t} even though the entire prompt memory is available."
                mem_i = [(k[:, :, : t0 + i + 1], v[:, :, : t0 + i + 1]) for (k, v) in mem]
            else:
                e_i, enc = self.encode_step(x[:, i:i + 1], pos_all[i:i + 1], enc)
                mem = self.project_memory(e_i, pos_all[i:i + 1], mem)
                mem_i = mem
            cur = State(s=s_cur, swa=swa_cur, enc=enc, mem=mem, t=t0 + i)
            cur = self.decode_step(e_i, cur, pos_all[i:i + 1], mem=mem_i)
            s_cur, swa_cur = cur.s, cur.swa
            logits.append(self.readout(s_cur))
            if collect_states:
                states.append(s_cur)

        out = torch.stack(logits, dim=1)
        final = State(s=s_cur, swa=swa_cur, enc=enc, mem=mem, t=t0 + n)
        if collect_states:
            return out, final, torch.stack(states, 1)
        return out, final

    def forward(self, x, parallel_encode: bool = True, collect_states: bool = False):
        st = self.init_state(x.shape[0], x.device, self.embed.weight.dtype)
        return self.consume(x, st, parallel_encode=parallel_encode,
                            collect_states=collect_states)

    # ------------------------------------------------- C3: the Jacobi probe

    @torch.no_grad()
    def forward_parallel_swa(self, x, n_iters: int = 1):
        """A token-parallel decoder pass -- the kernel Sec. 2.4 warns against.

            "historical decoder KV must be produced by the preceding recurrent
             updates; a standard parallel SWA decoder pass is not generally
             equivalent."

        Given the whole merged input sequence u_{1:S}, a windowed-causal parallel
        pass over the decoder stack IS exact.  The obstruction sits upstream: u_t
        needs s_{t-1}, which is the stack's own output.  So the only way to
        parallelise is to guess s and iterate (Jacobi).  Each sweep makes exactly
        one more position exact -- the quantitative form of "No exact parallel scan
        for the general nonlinear decoder is assumed" (Sec. 4.1).
        """
        B, S = x.shape
        dev = x.device
        pos = torch.arange(S, device=dev)
        e, _ = self.encode_parallel(x, pos)
        mem = self.project_memory(e, pos, [None] * self.cfg.G)
        swa_mask = sliding_window_mask(S, self.cfg.W, dev)
        mem_mask = causal_mask(S, dev)

        s0 = self.s_star.to(e.dtype).expand(B, 1, -1)
        s_prev = s0.expand(B, S, -1)               # guess: s_star everywhere
        z = None
        for _ in range(max(1, n_iters)):
            z = self.merge(e, s_prev)
            for l, layer in enumerate(self.decoder):
                z = layer.forward_parallel(z, pos, mem[self.group_of(l)],
                                           swa_mask, mem_mask)
            # s^{(k)}_t feeds position t+1 on the next sweep.
            s_prev = torch.cat([s0, z[:, :-1]], dim=1)
        return self.readout(z), z

    # ------------------------------------------------------------- generation

    @torch.no_grad()
    def generate(self, prompt, n_new: int, temperature: float = 1.0,
                 top_k: int = 0, greedy: bool = False, generator=None):
        """Sec. 2.3 / App. A.2.  Returns (tokens, behaviour log-probs, final state).

        The log-probs are log mu(y_i | .) under the ACTUAL sampling distribution --
        temperature and truncation included -- because Sec. 5.3 is blunt about the
        alternative: "Behavior log-probabilities must include temperature,
        truncation, and renormalization; metadata alone cannot restore missing
        support."
        """
        st = self.init_state(prompt.shape[0], prompt.device, self.embed.weight.dtype)
        logits, st = self.consume(prompt, st)
        toks, logps = [], []
        cur = logits[:, -1]
        for _ in range(n_new):
            l = cur / max(temperature, 1e-6)
            if top_k > 0:
                kth = l.topk(top_k, dim=-1).values[:, -1:]
                l = l.masked_fill(l < kth, float("-inf"))
            lp = torch.log_softmax(l.float(), dim=-1)
            if greedy:
                y = lp.argmax(-1)
            else:
                y = torch.multinomial(lp.exp(), 1, generator=generator).squeeze(-1)
            toks.append(y)
            logps.append(lp.gather(-1, y[:, None]).squeeze(-1))
            out, st = self.consume(y[:, None], st)
            cur = out[:, -1]
        return torch.stack(toks, 1), torch.stack(logps, 1), st

    # --------------------------------------------------------------- counting

    def n_params(self) -> int:
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total
