"""Configuration for Recurrent Looped Transformer.

Symbol names follow the paper (Zhang, 2026) exactly:

    L_E, L_D   encoder / decoder depth              (Sec. 2.1)
    d          residual width                       (Sec. 2.1)
    W          SWA window size, INCLUDES the current token, W >= 1   (Sec. 2.1)
    G          number of encoder-memory groups; G = 1 shares memory across
               layers, G = L_D permits layer-specific projections    (Sec. 2.2)
    alpha      feedback scale in the merge, Eq. (2.11)
"""
from dataclasses import dataclass


@dataclass
class RLTConfig:
    vocab_size: int = 64
    d_model: int = 256
    n_heads: int = 4
    L_E: int = 4
    L_D: int = 4
    d_ff: int = 1024
    W: int = 8                 # SWA window, includes current token (Sec. 2.1)
    G: int = 1                 # encoder memory groups (Sec. 2.2)
    alpha: float = 1.0         # feedback scale, Eq. (2.11). alpha = 0 removes recurrence.
    tied: bool = True          # Sec. 2.6 reference tied configuration
    rope_base: float = 10000.0
    norm_eps: float = 1e-6
    max_position: int = 4096
    init_std: float = 0.02

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        assert self.W >= 1, "Sec. 2.1: W >= 1 includes the current token"
        assert self.G in (1, self.L_D), (
            "Sec. 2.2 discusses G = 1 (shared) and G = L_D (layer-specific); "
            "intermediate G is a valid generalisation but not implemented here"
        )
        if self.tied:
            assert self.L_E == self.L_D, (
                "Sec. 2.6: 'The reference tied RLT sets L_E = L_D = L.'"
            )

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @property
    def blocks_per_token(self) -> int:
        """Sec. 3.2: 'Each new token evaluates L_E + L_D blocks.'"""
        return self.L_E + self.L_D
