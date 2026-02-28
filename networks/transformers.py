from typing import Optional

import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import orthogonal, constant
import numpy as np

def causal_mask(L: int) -> jnp.ndarray:
    """(1,1,L,L) causal mask for SelfAttention."""
    return jnp.tril(jnp.ones((1, 1, L, L), dtype=jnp.bool_))


# ------------------------------------------------------------------ #
# Embedding layer                                                    #
# ------------------------------------------------------------------ #
class LatentEmbedding(nn.Module):
    embed_dim: int = 64   # E
    V: int = 32           # number of categorical variables
    C: int = 32           # number of classes per variable
    std: float = 0.1

    @nn.compact
    def __call__(self, z):                   # z: (B, V, C) one-hot
        table = self.param("table",
                           nn.initializers.normal(self.std),
                           (self.V, self.C, self.embed_dim))
        # einsum = gather & matmul in one shot → (B,V,E)
        return jnp.einsum("bvc, vce -> bve", z, table)

# ------------------------------------------------------------------ #
# Transformer encoder block                                          #
# ------------------------------------------------------------------ #
class TransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, *, causal: Optional[jnp.ndarray]=None, deterministic: bool = True):
        # Self-attention
        h = nn.LayerNorm()(x)
        h = nn.SelfAttention(
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                deterministic=deterministic
        )(h, mask=causal)

        x = x + h                          # residual 1

        # Feed-forward
        h = nn.LayerNorm()(x)
        h = nn.Dense(self.mlp_dim)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.d_model,)(h)
        h = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(h)

        x = x + h                          # residual 2

        return x
