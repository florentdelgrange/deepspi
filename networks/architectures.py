from enum import Enum
from typing import Tuple, Callable

import chex
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from jax._src.nn.initializers import lecun_normal
from tensorflow_probability.substrates import jax as tfp

from networks.softmoe import SoftMoE, ExpertModel
from utils.activations import StableSigmoid
from utils.distributions import SoftMixtureNormalImage, LipschitzNormal, STOneHotCategorical, TransitionDensity, \
    STBernoulli
from networks.lipschitz import SpectralConv, GroupSort, SpectralDense, power_iteration
from networks.transformers import LatentEmbedding, TransformerBlock, causal_mask

tfd = tfp.distributions

class NetType(str, Enum):
    """Enum for different network types."""
    CONV = "conv"
    FC = "fc"
    TRANSFORMER = "transformer"
    SOFTMOE = "softmoe"


class NetworkConv(nn.Module):

    layer_norm_output: bool = False
    use_layer_norm: bool = False
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.use_layer_norm:
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)
        x = nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = normalize(x)
        x = getattr(nn, self.activation)(x)  # e.g., nn.relu, nn.elu, etc.
        x = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = normalize(x)
        x = getattr(nn, self.activation)(x)
        x = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        if self.layer_norm_output:
            x = nn.LayerNorm()(x)
        x = getattr(nn, self.activation)(x)
        return x

class CategoricalEncoder(nn.Module):
    """Dreamer-style categorical encoder."""
    n_cat: int = 32  # discrete latent count
    n_cls: int = 32  # categories per latent
    hadamard_representation: bool = False

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))  # flatten
        if self.hadamard_representation:
            linear_1 = nn.Dense(features=self.n_cat * self.n_cls)(x)
            linear_2 = nn.Dense(features=self.n_cat * self.n_cls)(x)
            representation_1 = nn.tanh(linear_1)
            representation_2 = nn.tanh(linear_2)
            x = representation_1 * representation_2  # Hadamard product
        logits = nn.Dense(features=self.n_cat * self.n_cls)(x)
        logits = logits.reshape((-1, self.n_cat, self.n_cls))
        out = nn.softmax(logits, axis=-1)  # [B, n_cat, n_cls]; probabilities of each class per category
        indices = jnp.argmax(out, axis=-1)
        one_hot = jax.nn.one_hot(indices, self.n_cls, axis=-1).astype(jnp.float32)  # [B, n_cat, n_cls]
        return one_hot + out - jax.lax.stop_gradient(out)  # [B, n_cat, n_cls]

class NetworkFCOutput(nn.Module):
    layers: Tuple[int, ...] = (512, )
    use_layer_norm: bool = False
    activation: str = "relu"
    hadamard_representation: bool = False
    latent_dim_hadamard: int = 512  # Only used if hadamard_representation is True

    def get_kernel_init(self):
        """Get the kernel initializer based on the activation function."""
        if self.activation == "relu":
            return orthogonal(np.sqrt(2))
        else:
            return lecun_normal()

    @nn.compact
    def __call__(self, x):
        if self.use_layer_norm:
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x
        # flatten and dense
        x = x.reshape((x.shape[0], -1))
        if self.hadamard_representation:
            linear_1 = nn.Dense(features=self.latent_dim_hadamard,)(x)
            linear_2 = nn.Dense(features=self.latent_dim_hadamard,)(x)
            representation_1 = nn.tanh(linear_1)
            representation_2 = nn.tanh(linear_2)
            x = representation_1 * representation_2
        for units in self.layers:
            x = nn.Dense(units, kernel_init=self.get_kernel_init(), bias_init=constant(0.0))(x)
            x = normalize(x)
            x = getattr(nn, self.activation)(x)  # e.g., nn.relu, nn.elu, etc.
        return x

class NetworkAttentionOutput(nn.Module):
    """Transformer-style attention output network. Not recommended."""
    embed_dim: int = 32
    num_blocks: int = 2
    num_heads: int = 4
    mlp_dim: int = 128
    out_dim: int = 512
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, *, deterministic: bool = True):
        """
        x : (B, 32, 32) straight-through one-hot latents
        """
        # tokenise
        h = LatentEmbedding(embed_dim=self.embed_dim)(x)      # (B,32,E)

        # add learnable positional embedding
        pos = self.param("pos_emb",
                         nn.initializers.normal(0.02),
                         (1, 32, self.embed_dim))
        h = h + pos

        # Transformer stack
        for _ in range(self.num_blocks):
            h = TransformerBlock(d_model=self.embed_dim,
                                 mlp_dim=self.mlp_dim,
                                 num_heads=self.num_heads,
                                 dropout_rate=self.dropout_rate)(
                                        h, deterministic=deterministic)

        # flatten & final dense (512 dims)
        h = h.reshape((h.shape[0], -1))                          # (B,32*E)
        h = nn.Dense(self.out_dim,
                     kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(h)
        h = nn.relu(h)
        return h

class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)

# ============================================================
# Reward and Transition Networks
# ============================================================

# DiscreteActionTransitionCNN and DiscreteActionRewardCNN are CNN-based networks

class DiscreteActionTransitionCNN(nn.Module):
    num_actions: int
    density: TransitionDensity = TransitionDensity.DETERMINISTIC
    n_distributions: int = 5  # Only used for MIXTURE_NORMAL; essentially the number of ensemble distributions
    gumbel_softmax: bool = False  # Whether to use Gumbel-Softmax for CATEGORICAL density
    feature_group: bool = False
    activation: str = "relu"

    @nn.compact
    def __call__(self, inputs):
        obs, action = inputs

        if self.density == TransitionDensity.CATEGORICAL:
            chex.assert_rank(obs, 3)  # Expecting (batch, n_cat, n_cls)
            obs = obs[..., None]

        n_outputs = self.num_actions

        if self.density == TransitionDensity.NORMAL or \
                (self.density == TransitionDensity.MIXTURE_NORMAL and self.feature_group):
            n_outputs = 2 * n_outputs  # For mean and stddev in normal distribution
            if self.density == TransitionDensity.MIXTURE_NORMAL:
                n_outputs *= self.n_distributions
        elif self.density == TransitionDensity.MIXTURE_NORMAL:
            n_outputs = 3 * n_outputs  # For mean, stddev, and logits in mixture normal distribution
            n_outputs *= self.n_distributions  # For the number of ensemble distributions

        x = z = obs

        depth = 2 if self.density == TransitionDensity.CATEGORICAL or self.feature_group else 1
        for i in range(depth):
            # Conv: output shape (batch, h, w, 64 * num_actions)
            features = 64 * n_outputs if self.density != TransitionDensity.CATEGORICAL or i == 1 else 8
            feature_group_count = features if self.feature_group and i == depth - 1 else 1
            x = nn.Conv(
                features=features,
                kernel_size=(2, 2),
                strides=(1, 1),
                padding='SAME',
                feature_group_count=feature_group_count,
            )(x)
            if i < depth - 1:
                x = getattr(nn, self.activation)(x)
                z = x

        batch_size, h, w, _ = x.shape

        action_one_hot = jax.nn.one_hot(action, self.num_actions)  # (batch, num_actions)

        # Select output corresponding to action
        if self.density == TransitionDensity.DETERMINISTIC:
            x = x.reshape(batch_size, h, w, 64, self.num_actions)
            out = jnp.einsum('bhwca,ba->bhwc', x, action_one_hot)  # (batch, h, w, 64)
        elif self.density == TransitionDensity.NORMAL:
            x = x.reshape(batch_size, h, w, 64, 2, self.num_actions)
            out = jnp.einsum('bhwcna,ba->bhwcn', x, action_one_hot)  # (batch, h, w, 64, N)
        elif self.density == TransitionDensity.MIXTURE_NORMAL:
            n = 2 if self.feature_group else 3
            x = x.reshape(batch_size, self.n_distributions, h, w, 64, n, self.num_actions)
            out = jnp.einsum('bkhwcna,ba->bkhwcn', x, action_one_hot)  # (batch, n_distributions, h, w, 64)
        elif self.density == TransitionDensity.CATEGORICAL:
            x = x.reshape(batch_size, h, w, 64, self.num_actions)
            weights = self.param("category_weights", nn.initializers.zeros, (64, ))
            bias = self.param("category_bias", nn.initializers.zeros, (1,))
            x = jnp.einsum('bhwca,c->bhwa', x, weights) + bias  # (batch, h, w, actions)
            out = jnp.einsum('bhwa,ba->bhw', x, action_one_hot)  # (batch, h, w)

        if self.density == TransitionDensity.DETERMINISTIC:
            out = getattr(nn, self.activation)(out)
            return tfd.Independent(tfd.Deterministic(loc=out), reinterpreted_batch_ndims=3)
        elif self.density == TransitionDensity.CATEGORICAL:
            return STOneHotCategorical(logits=out, use_gumbel_softmax=self.gumbel_softmax)
        else:
            # Masked conv output:  (B, K, H, W, 64, 2|3)
            if self.density == TransitionDensity.NORMAL or \
                    self.density == TransitionDensity.MIXTURE_NORMAL and self.feature_group:
                loc, scale = jnp.split(out, 2, axis=-1)  # each (B, *, H, W, 64, 1)
                if self.density == TransitionDensity.MIXTURE_NORMAL:
                    logits = nn.Dense(
                        features=self.n_distributions * self.num_actions,
                    )(z.reshape(batch_size, -1))
                    logits = jnp.reshape(logits, (batch_size, self.n_distributions, self.num_actions))  # (B, K, A)
                    logits = logits @ action_one_hot[..., None]  # (B, K, A) @ (B, A, 1) -> (B, K, 1)
                    logits = logits.squeeze(-1)  # (B, K)
            else:  # MIXTURE_NORMAL
                loc, scale, logits = jnp.split(out, 3, axis=-1)  # each (B, K, H, W, 64, 1)
                logits_linear_map = self.param(
                    "logits_linear_map", nn.initializers.lecun_normal(), (h, w, 64))
                logits = jnp.einsum('bkhwcl,hwc->bkl', logits, logits_linear_map)  # (B, K, 1)
                logits = jnp.squeeze(logits, axis=-1)  # (B, K)
            loc = getattr(nn, self.activation)(loc.squeeze(-1))
            scale = nn.softplus(scale.squeeze(-1)) + 1e-6  # (B, K, H, W, 64)
            if self.density == TransitionDensity.NORMAL:
                return tfd.Independent(
                    tfd.Normal(loc=loc, scale=scale),
                    reinterpreted_batch_ndims=3)
            else:
                # build distribution – no manual flattening needed
                return SoftMixtureNormalImage(
                    logits=logits,
                    loc=loc,
                    scale=scale,)


class DiscreteActionRewardCNN(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, inputs):
        obs, action = inputs  # unpack the tuple
        batch_size = obs.shape[0]

        if len(obs.shape) == 3:  # If obs is (batch, h, w)
            obs = obs[..., None]  # Add channel dimension to make it (batch, h, w, 1)

        # Conv: output shape (batch, h, w, 64 * num_actions)
        x = nn.Conv(
            features=64 * self.num_actions,
            kernel_size=(2, 2),
            strides=(1, 1),
            padding='SAME',
        )(obs)
        x = nn.relu(x)

        batch_size, h, w, _ = x.shape

        x = x.reshape(batch_size, h, w, 64, self.num_actions)

        # Select output corresponding to action
        action_one_hot = jax.nn.one_hot(action, self.num_actions)  # (batch, num_actions)
        out = jnp.einsum('bhwca,ba->bhwc', x, action_one_hot)  # (batch, h, w, 64)

        out = jnp.reshape(out, (batch_size, -1))  # Flatten to (batch, h * w * 64)

        out = nn.Dense(1)(out)  # Final output layer for reward prediction

        return out.squeeze(-1)

# Lipschitz versions of the networks

class LipDiscreteActionTransitionCNN(nn.Module):
    num_actions: int
    K: float = 1.0  # Lipschitz constant
    channels: Tuple[int, ...] = (64, 64)
    power_iters: int = 2
    group_size: int = 2
    density: TransitionDensity = TransitionDensity.DETERMINISTIC
    n_distributions: int = 5  # Only used for MIXTURE_NORMAL; essentially the number of ensemble distributions
    feature_group: bool = False

    @nn.compact
    def __call__(self, inputs):
        if self.density == TransitionDensity.CATEGORICAL:
            raise ValueError("LipDiscreteActionTransitionNetwork does not support CATEGORICAL density.")

        obs, action = inputs  # unpack the tuple
        x = z = obs

        n_outputs = self.num_actions

        if self.density in [TransitionDensity.NORMAL, TransitionDensity.MIXTURE_NORMAL]:
            n_outputs = 2 * n_outputs  # For mean and stddev in normal distribution
            if self.density == TransitionDensity.MIXTURE_NORMAL:
                n_outputs *= self.n_distributions  # For the number of ensemble distributions

        for i, ch in enumerate(self.channels):
            mult = n_outputs if i == len(self.channels) - 1 else 1
            if self.feature_group and i < len(self.channels) - 1:
                # double the number of channels for the first layers
                mult *= 2
            # allocate 4 channels per parallel group; each kernel will see 4 input channels
            feature_group_count = ch // 4 if i == len(self.channels) - 1 and self.feature_group else 1
            x = SpectralConv(
                features=ch * mult,
                power_iters=self.power_iters,
                kernel_size=(2, 2),
                strides=(1, 1),
                feature_group_count=feature_group_count,
            )(x)
            if i < len(self.channels) - 1:
                x = GroupSort(group_size=self.group_size)(x)
                z = x

        batch_size, h, w, _ = x.shape
        c = self.channels[-1]

        action_one_hot = jax.nn.one_hot(action, self.num_actions)  # (batch, num_actions)

        # Select output corresponding to action
        if self.density == TransitionDensity.DETERMINISTIC:
            x = x.reshape(batch_size, h, w, c, self.num_actions)
            out = jnp.einsum('bhwca,ba->bhwc', x, action_one_hot)  # (B, h, w, c)
        elif self.density == TransitionDensity.NORMAL:
            x = x.reshape(batch_size, h, w, c, 2, self.num_actions)
            out = jnp.einsum('bhwcna,ba->bhwcn', x, action_one_hot)  # (B, h, w, c, 2)
        elif self.density == TransitionDensity.MIXTURE_NORMAL:
            x = x.reshape(batch_size, h, w, c, self.n_distributions, 2, self.num_actions)
            out = jnp.einsum('bhwckna,ba->bhwckn', x, action_one_hot)  # (B, h, w, c, k, 2)

        if self.density == TransitionDensity.DETERMINISTIC:
            return tfd.Independent(tfd.Deterministic(loc=self.K * out), reinterpreted_batch_ndims=3)
        elif self.density in [TransitionDensity.NORMAL, TransitionDensity.MIXTURE_NORMAL]:
            if self.density in (TransitionDensity.NORMAL, TransitionDensity.MIXTURE_NORMAL):
                loc, scale = jnp.split(out, 2, axis=-1)  # each (B, H, W, 64, *, 1)
                if self.density == TransitionDensity.MIXTURE_NORMAL:
                    logits = SpectralDense(
                        features=self.n_distributions * self.num_actions, power_iters=self.power_iters
                    )(z.reshape(batch_size, -1)) # (B, K * A)
                    logits = jnp.reshape(logits, (batch_size, self.n_distributions, self.num_actions))  # (B, K, A)
                    logits = logits @ action_one_hot[..., None]  # (B, K, A) @ (B, A, 1) -> (B, K, 1)
                    logits = logits.squeeze(-1)  # (B, K)
            loc = loc.squeeze(-1)  # (B, H, W, 64, K)
            scale = nn.softplus(scale.squeeze(-1)) + 1e-6  # (B, H, W, 64, K)
            if self.density == TransitionDensity.NORMAL:
                return LipschitzNormal(loc=.5 * self.K * loc, scale=.5 * self.K * scale)
            else:
                loc = loc.transpose(0, 4, 1, 2, 3)  # (B, K, H, W, 64)
                scale = scale.transpose(0, 4, 1, 2, 3)  # (B, K, H, W, 64)
                return SoftMixtureNormalImage(
                    logits=logits,
                    loc=.5 * self.K * loc,
                    scale=.5 * self.K * scale,
                    clip_epsilon=1.)


class LipDiscreteActionRewardNetwork(nn.Module):
    num_actions: int
    K: float = 1.0  # Lipschitz constant
    layers: int = 2
    hidden: int = 256
    power_iters: int = 2
    group_size: int = 2

    @nn.compact
    def __call__(self, inputs):
        obs, action = inputs  # unpack the tuple
        x = obs

        x = jnp.reshape(x, (x.shape[0], -1))  # flatten
        for i in range(self.layers):
            x = SpectralDense(features=self.hidden, power_iters=self.power_iters)(x)
            x = GroupSort(group_size=self.group_size)(x)

        out = SpectralDense(features=self.num_actions, power_iters=self.power_iters)(x)
        action_one_hot = jax.nn.one_hot(action, self.num_actions)  # (batch, num_actions)
        out = jnp.einsum('ba,ba->b', out, action_one_hot)  # (batch,)
        out = self.K * out  # Apply Lipschitz constant

        return out

# SoftMoE-based networks for discrete actions

class DiscreteActionTransitionNetworkSoftMoE(nn.Module):
    num_actions: int
    num_experts: int = 4 # Number of experts in SoftMoE, only used if soft_moe is True
    gumbel_softmax: bool = False  # Whether to use Gumbel-Softmax for CATEGORICAL density
    density: TransitionDensity = TransitionDensity.CATEGORICAL

    @nn.compact
    def __call__(self, inputs):
        chex.assert_equal(self.density, TransitionDensity.CATEGORICAL)
        obs, action = inputs
        # returns a [B, n_cat * n_actions, n_cls] tensor
        _obs = SoftMoE(
            module=ExpertModel(
                expert_hidden_size=512,
                initializer=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
                raw_output=False,
                num_layers=1
            ),
            num_experts=self.num_experts,
            num_outputs=self.num_actions,
        )(obs).values
        # reshape to [B, n_cat, n_actions, n_cls]
        obs = _obs.reshape((obs.shape[0], self.num_actions, obs.shape[1], obs.shape[2]))
        action_one_hot = jax.nn.one_hot(action, self.num_actions)  # (batch, num_actions)
        out = jnp.einsum('bamd,ba->bmd', obs, action_one_hot)  # (batch, n_cat, n_cls)
        return STOneHotCategorical(logits=out, use_gumbel_softmax=self.gumbel_softmax)


class DiscreteActionRewardNetworkSoftMoE(nn.Module):
    num_actions: int
    num_experts: int = 4  # Number of experts in SoftMoE
    density: TransitionDensity = TransitionDensity.CATEGORICAL

    @nn.compact
    def __call__(self, inputs):
        obs, action = inputs
        batch_size = obs.shape[0]
        chex.assert_equal(self.density, TransitionDensity.CATEGORICAL)
        chex.assert_equal(obs.ndim, 3)  # Expecting (batch, n_cat, n_cls)

        # returns a [B, n_cat * n_actions, n_cls] tensor
        _obs = SoftMoE(
            module=ExpertModel(
                expert_hidden_size=512,
                initializer=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
                raw_output=False,
                num_layers=1),
            num_experts=self.num_experts,
            num_outputs=self.num_actions,
        )(obs).values
        # reshape to [B, n_cat, n_actions, n_cls]
        obs = _obs.reshape((obs.shape[0], self.num_actions, obs.shape[1], obs.shape[2]))
        action_one_hot = jax.nn.one_hot(action, self.num_actions)  # (batch, num_actions)
        out = jnp.einsum('bamd,ba->bmd', obs, action_one_hot)  # (batch, n_cat, n_cls)
        out = jnp.reshape(out, (batch_size, -1))  # Flatten to (batch, n_cat * n_cls)
        out = nn.Dense(1)(out)  # Final output layer for reward prediction
        return out.squeeze(-1)

# Feedforward networks

class DiscreteActionTransitionNetwork(nn.Module):
    """A feedforward network for discrete action transitions."""
    num_actions: int
    n_cat: int = 32  # Number of categorical variables
    n_cls: int = 32  # Number of classes per category
    density: TransitionDensity = TransitionDensity.CATEGORICAL
    gumbel_softmax: bool = False  # Whether to use Gumbel-Softmax for CATEGORICAL density
    embed_dim: int = 128
    hidden: int = 512
    layers: int = 1
    use_layer_norm: bool = False
    activation: str = 'relu'

    @nn.compact
    def __call__(self, inputs):
        obs, action = inputs
        chex.assert_equal(self.density, TransitionDensity.CATEGORICAL)
        chex.assert_rank(obs, 3)  # Expecting (B, n_cat, n_cls)

        embedding_flat = self.param(
            'embedding', nn.initializers.normal(stddev=1.0), (self.n_cat * self.n_cls, self.embed_dim))
        embedding = embedding_flat.reshape((self.n_cat, self.n_cls, self.embed_dim))  # (n_cat, n_cls, D)
        z_soft = obs
        tok_emb = (z_soft[..., None] * embedding[None, ...]).sum(axis=2)  # (B, n_cat, D)

        act_emb = nn.Embed(self.num_actions, self.embed_dim)(action) # (B,D)
        act_broadcast = jnp.expand_dims(act_emb, 1)        # (B,1,D)

        x = jnp.concatenate([tok_emb, act_broadcast.repeat(self.n_cat, 1)], axis=-1)  # (B, n_cat, 2D)
        x = x.reshape(x.shape[0], -1)                      # flatten to (B,L*2D)

        for _ in range(self.layers):
            x = nn.Dense(self.hidden)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = getattr(nn, self.activation)(x)  # e.g., nn.relu, nn.elu, etc.
        logits = nn.Dense(self.n_cat * self.n_cls)(x)
        logits = logits.reshape((logits.shape[0], self.n_cat, self.n_cls))

        return STOneHotCategorical(logits=logits, use_gumbel_softmax=self.gumbel_softmax)


class DiscreteActionRewardNetwork(nn.Module):
    """A feedforward network for discrete action rewards."""
    num_actions: int
    embed_dim: int = 128
    hidden: int = 512
    layers: int = 1
    use_embedding: bool = True  # Whether to use an embedding layer for the input
    use_layer_norm: bool = False
    activation: str = 'relu'
    clip_rewards: bool = False

    @nn.compact
    def __call__(self, inputs):
        obs, action = inputs
        if self.use_embedding and jnp.ndim(obs) == 4:
            h, w, c = obs.shape[1:]
            n_cat = h * w
            n_cls = c
            obs = jnp.reshape(obs, (obs.shape[0], n_cat, n_cls))  # Reshape to (B, n_cat, n_cls)
            use_embedding = True
        elif self.use_embedding and jnp.ndim(obs) == 3:
            n_cat, n_cls = obs.shape[1:]
            use_embedding = True
        else:
            use_embedding = False

        if use_embedding:
            embedding_flat = self.param(
                'embedding', nn.initializers.normal(stddev=1.0), (n_cat * n_cls, self.embed_dim))
            embedding = embedding_flat.reshape((n_cat, n_cls, self.embed_dim))  # (n_cat, n_cls, D)
            z_soft = obs
            tok_emb = (z_soft[..., None] * embedding[None, ...]).sum(axis=2)  # (B, n_cat, D)
            x = tok_emb
        else:
            x = obs

        x = x.reshape(x.shape[0], -1)                      # flatten

        for _ in range(self.layers):
            x = nn.Dense(self.hidden)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = getattr(nn, self.activation)(x)  # e.g., nn.relu, nn.elu, etc.

        out = nn.Dense(self.num_actions)(x)  # (batch, num_actions))
        if self.clip_rewards:
            out = StableSigmoid()(out) * 2. - 1.  # Scale to [-1, 1]
        action_one_hot = jax.nn.one_hot(action, self.num_actions)  # (batch, num_actions)
        return jnp.einsum('ba,ba->b', out, action_one_hot)  # (batch,)

class AutoregressiveDiscreteActionTransitionTransformer(nn.Module):
    """Autoregressive Transformer for discrete action transitions."""
    # ---- sizes -----------
    num_actions: int
    d_model: int = 128  # Dimension of the embedding
    n_heads: int = 4  # Number of attention heads
    n_layers: int = 3  # Number of Transformer blocks
    mlp_dim: int = 512  # Dimension of the MLP in each Transformer block
    # ---- token structure --
    num_categories: int = 32    # rows  (L)
    num_classes: int = 32       # vals  (K)
    # ---- dropout ----------
    dropout: float = 0.0
    # ---- output ------------
    density: TransitionDensity = TransitionDensity.CATEGORICAL
    gumbel_softmax: bool = False  # Whether to use Gumbel-Softmax for CATEGORICAL density

    @nn.compact
    def __call__(self, inputs, deterministic: bool = True) -> tfd.Distribution:
        obs, act = inputs
        chex.assert_equal(self.density, TransitionDensity.CATEGORICAL)
        chex.assert_rank(obs, 3)  # Expecting (B, n_cat, n_cls)

        B, L, K = obs.shape[0], self.num_categories, self.num_classes

        # (row,class): single token embedding table
        E_flat = self.param("E_tok",
                            nn.initializers.normal(stddev=1.0),
                            (L * K, self.d_model))
        E = E_flat.reshape(L, K, self.d_model)             # (num_cat,num_cls,D)

        # value embeddings
        val_emb = (obs[..., None] * E[None, ...]).sum(axis=2)

        # Build full sequence with ⟨ACT=a⟩ token at pos 0
        act_emb_tab = nn.Embed(self.num_actions, self.d_model, name="E_act")
        act_emb = act_emb_tab(act)[:, None, :]                      # (B,1,D)
        seq = jnp.concatenate([act_emb, val_emb + act_emb], axis=1)

        # positional encoding
        pos_emb = self.param("pos",
                             nn.initializers.normal(stddev=0.01),
                             (L + 1, self.d_model))
        seq = seq + pos_emb[None, ...]

        # Transformer stack with causal mask (length num_cat + 1)
        mask = causal_mask(L + 1)
        h = seq
        for _ in range(self.n_layers):
            h = TransformerBlock(
                self.d_model, self.n_heads, self.mlp_dim, self.dropout
            )(h, causal=mask, deterministic=deterministic)

        # Project tokens 1...num_cat to logits for num_cls classes
        logits = nn.Dense(K, name="out_proj")(h[:, 1:, ...])     # (B,num_cat,num_cls)

        return STOneHotCategorical(logits=logits, use_gumbel_softmax=self.gumbel_softmax)

# ============================================================
# Done predictor
# ============================================================
class DonePredictor(nn.Module):
    """A simple feedforward network for predicting done states."""
    hidden_size: int = 512
    activation: str = "relu"
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, z):
        z = z.reshape((z.shape[0], -1))  # Flatten the input
        z = nn.Dense(self.hidden_size)(z)
        if self.use_layer_norm:
            z = nn.LayerNorm()(z)
        z = getattr(nn, self.activation)(z)  # e.g., nn.relu, nn.elu, etc.
        logits = nn.Dense(1)(z)
        logits = jnp.squeeze(logits, axis=-1)
        return STBernoulli(logits=logits)

# ============================================================
# Lipschitz variants of the Conv Network
# ============================================================

class NetworkConvLipschitz(nn.Module):
    power_iters: int = 2
    group_size: int = 2

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)
        x = SpectralConv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            power_iters=self.power_iters,
        )(x)
        x = GroupSort(group_size=self.group_size)(x)
        x = SpectralConv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            power_iters=self.power_iters,
        )(x)
        x = GroupSort(group_size=self.group_size)(x)
        x = SpectralConv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            power_iters=self.power_iters,
        )(x)
        x = GroupSort(group_size=self.group_size)(x)
        return x

# ============================================================
# Discriminators (to approximate dTV and Wasserstein)
# ============================================================

class BoundedDiscriminator(nn.Module):
    """Discriminator for discrete actions, used to approximate dTV."""
    num_actions: int
    embed_dim: int = 128
    hidden_size: int = 512
    clip_fn: Callable = StableSigmoid()
    out_scale: float = .5  # scale factor for the output
    use_cnn: bool = True  # Whether to use a CNN for observation processing

    @nn.compact
    def __call__(self, x):
        obs, act, z, z_prime  = x
        n_cat, n_cls = z.shape[1:]

        if self.use_cnn:
            x_1 = NetworkConv(layer_norm_output=True)(obs)
            x_1 = jnp.reshape(x_1, (x_1.shape[0], -1))
            x_1 = nn.Dense(self.embed_dim * n_cat)(x_1)  # (B, n_cat * D)
            x_1 = x_1.reshape((x_1.shape[0], n_cat, self.embed_dim))  # (B, n_cat, D)
        else:
            x_1 = jnp.zeros((obs.shape[0], n_cat, 0))  # ignore obs if not using CNN

        embedding_flat = self.param(
            'discriminator_embedding',
            nn.initializers.normal(stddev=1.0),
            (n_cat * n_cls, self.embed_dim))
        embedding = embedding_flat.reshape((n_cat, n_cls, self.embed_dim))  # (n_cat, n_cls, D)
        x_2 = (z[..., None] * embedding[None, ...]).sum(axis=2)  # (B, n_cat, D)

        x_3 = (z_prime[..., None] * embedding[None, ...]).sum(axis=2)  # (B, n_cat, D)

        x = jnp.concatenate([x_1, x_2, x_3], axis=-1)  # (B, n_cat, 3D)
        x = jnp.reshape(x, (x.shape[0], -1))  # Flatten to (B, n_cat * 3D)

        x = nn.Dense(self.hidden_size)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_actions)(x)
        x = nn.LayerNorm()(x)

        out = 2. * self.clip_fn(2. * x) - 1.  # Stable sigmoid to ensure the output is in [-1, 1]
        out = self.out_scale * out

        action_one_hot = jax.nn.one_hot(act, self.num_actions)  # (batch, num_actions)
        return jnp.einsum('ba,ba->b', out, action_one_hot)  # (batch,)


class LipschitzDiscriminator(nn.Module):
    """Lipschitz discriminator for discrete actions, used to approximate the Wasserstein distance."""
    num_actions: int
    embed_dims: int = 128
    hidden_layers: Tuple[int, ...] = (256, 256)
    power_iters: int = 2
    group_size: int = 2
    K: float = 1. # Lipschitz constant
    use_cnn: bool = True  # Whether to use a CNN for observation processing

    @nn.compact
    def __call__(self, x):
        obs, act, z, z_prime = x
        if self.use_cnn:
            x_1 = NetworkConvLipschitz(power_iters=self.power_iters, group_size=self.group_size)(obs)
            x_1 = jnp.reshape(x_1, (x_1.shape[0], -1))
            x_1 = SpectralDense(self.embed_dims, power_iters=self.power_iters)(x_1)  # (B, D)
        else:
            x_1 = jnp.zeros((obs.shape[0], 0))  # ignore obs if not using CNN

        embedding = SpectralDense(features=self.embed_dims, power_iters=self.power_iters,)
        x_2 = jnp.reshape(z, (z.shape[0], -1))
        x_2 = embedding(x_2)
        x_3 = jnp.reshape(z_prime, (z_prime.shape[0], -1))
        x_3 = embedding(x_3)

        x = jnp.concatenate([x_1, x_2, x_3], axis=-1)

        for layer in self.hidden_layers:
            x = SpectralDense(layer, power_iters=self.power_iters)(x)
            x = GroupSort(group_size=self.group_size)(x)
        x = SpectralDense(self.num_actions, power_iters=self.power_iters)(x)

        out = self.K * x

        action_one_hot = jax.nn.one_hot(act, self.num_actions)  # (batch, num_actions)
        return jnp.einsum('ba,ba->b', out, action_one_hot)  # (batch,)
