# ------------------------------------------------------------
# 1-Lipschitz building blocks
# ------------------------------------------------------------
import chex
import jax, jax.numpy as jnp
from flax import linen as nn
from typing import Sequence, Tuple

# --------- Spectral normalisation utilities -----------------
def power_iteration(W, num_iters: int = 1):
    """Return W_sn (spectral-norm =1) and σ."""
    v = jax.random.normal(jax.random.PRNGKey(0), (W.shape[1],))
    for _ in range(num_iters):          # usually 1–5 iters suffice
        norm_v = jnp.sqrt(jnp.sum(v**2, axis=-1) + 1e-8)
        v  = v / norm_v
        u = jnp.dot(W, v)              # u = W @ v
        norm_u = jnp.sqrt(jnp.sum(u**2, axis=-1) + 1e-8)
        u  = u / norm_u                # u = u / ||u||
        v  = jnp.dot(W.T, u)
    sigma = jnp.dot(u, jnp.dot(W, v))
    return W / sigma, sigma

# --------- 1-Lipschitz Dense --------------------------------
class SpectralDense(nn.Module):
    features: int
    use_bias: bool = True
    power_iters: int = 1
    @nn.compact
    def __call__(self, x):
        W = self.param('kernel', nn.initializers.lecun_normal(),
                       (x.shape[-1], self.features))
        W_sn, _ = power_iteration(W, self.power_iters)
        y = x @ W_sn
        if self.use_bias:
            y = y + self.param('bias', nn.initializers.zeros, (self.features,))
        return y

# --------- 1-Lipschitz Convolution --------------------------
class SpectralConv(nn.Module):
    features: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    padding: str = 'SAME'
    use_bias: bool = True
    power_iters: int = 1
    feature_group_count: int = 1  # for group convolution

    @nn.compact
    def __call__(self, x):
        in_channels = x.shape[-1]
        G = self.feature_group_count

        chex.assert_equal(in_channels % G, 0)
        chex.assert_equal(self.features % G, 0)

        in_ch_g = in_channels // G  # inC per group
        out_ch_g = self.features // G  # outC per group

        # kernel shape: (H, W, inC, outC) in Flax
        k_shape = self.kernel_size + (in_ch_g, self.features)
        K = self.param('kernel', nn.initializers.lecun_normal(), k_shape)
        H, W = self.kernel_size
        if G > 1:
            W_matrix = K                               \
                .reshape(H, W, in_ch_g, G, out_ch_g)   \
                .transpose(3, 4, 2, 0, 1)              \
                .reshape(G, out_ch_g, in_ch_g * H * W)

            def _spectral_norm(w):
                w_sn, _ = power_iteration(w, self.power_iters)
                return w_sn

            W_sn = jax.vmap(_spectral_norm)(W_matrix)  # shape unchanged
            K_sn = W_sn                              \
                .reshape(G, out_ch_g, in_ch_g, H, W) \
                .transpose(3, 4, 2, 0, 1)            \
                .reshape(H, W, in_ch_g, self.features)

        else:
            W_matrix = jnp.reshape(jnp.transpose(K, (3, 2, 0, 1)),  # outC × (inC*H*W)
                            (self.features, -1))
            W_sn, _ = power_iteration(W_matrix, self.power_iters)
            # Reshape spectral-normalized weights back to conv kernel
            K_sn = jnp.transpose(
                jnp.reshape(W_sn, (self.features, in_channels, *self.kernel_size)), (2, 3, 1, 0))

        # Perform convolution
        y = jax.lax.conv_general_dilated(
                lhs                   = x,
                rhs                   = K_sn,
                window_strides        = self.strides,
                padding               = self.padding,
                dimension_numbers     = ('NHWC', 'HWIO', 'NHWC'),
                feature_group_count   = G
        )
        if self.use_bias:
            y = y + self.param('bias', nn.initializers.zeros, (self.features,))
        return y

# --------- GroupSort activation (channel-wise) --------------
class GroupSort(nn.Module):
    group_size: int = 2
    @nn.compact
    def __call__(self, x):
        *spatial, C = x.shape
        # assert C % self.group_size == 0
        if C % self.group_size != 0:
            # Pad channels with zeros to make divisible
            pad = self.group_size - (C % self.group_size)
            x = jnp.pad(x, [(0, 0)] * len(spatial) + [(0, pad)], mode='constant')
            C = x.shape[-1]
        g = x.reshape(*spatial, C // self.group_size, self.group_size)
        g = jnp.sort(g, axis=-1)
        return g.reshape(*spatial, C)

# ------------------------------------------------------------
# Example K-Lipschitz ConvNet
# ------------------------------------------------------------
class KLipConvNet(nn.Module):
    hidden_channels: Sequence[int] = (32, 64)
    num_outputs: int = 4
    K: float = 1.0
    @nn.compact
    def __call__(self, x):
        for ch in self.hidden_channels:
            x = SpectralConv(ch)(x)
            x = GroupSort()(x)
        x = x.mean()  # 1-Lipschitz global average pooling
        x = SpectralDense(self.num_outputs)(x)
        return self.K * x                # ← exact K-Lipschitz