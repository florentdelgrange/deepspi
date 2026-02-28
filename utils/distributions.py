from enum import Enum

import jax
import jax.numpy as jnp
import optax
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb
import logging
import re

from utils.loss import js_distance

from jax.tree_util import register_pytree_node_class


class TransitionDensity(str, Enum):
    DETERMINISTIC = "deterministic"
    NORMAL = "normal"
    MIXTURE_NORMAL = "mixture_normal"
    CATEGORICAL = "categorical"

@register_pytree_node_class
class SoftMixtureNormalImage(tfd.Distribution):
    """
    Differentiable mixture of K multivariate diagonal-Gaussians whose samples
    are *automatically* reshaped to (H, W, C).

    Parameters
    ----------
    logits : (B, K)           - mixture logits
    loc    : (B, K, H, W, C)  - component means
    scale  : (B, K, H, W, C)  - component stdevs (positive)
    """

    def __init__(
            self, *, logits, loc, scale,
            clip_epsilon=None,
            validate_args=False,
            allow_nan_stats=True,
            dtype=jnp.float32,
            name="SoftMixtureNormalImage",
    ):

        if logits.ndim != 2 or loc.ndim != 5:
            raise ValueError("logits must be (B,K); loc/scale must be (B,K,H,W,C)")

        parameters = dict(locals())
        B, K, H, W, C = loc.shape
        D = H * W * C                      # flattened dimensionality

        # Store image-shaped tensors
        self.loc = loc          # (B,K,H,W,C)
        self.scale = scale      # (B,K,H,W,C)

        # Flatten loc/scale only for log-prob
        loc_flat   = loc.reshape(B, K, D)
        scale_flat = scale.reshape(B, K, D)
        self._components_flat = tfd.Normal(loc=loc_flat, scale=scale_flat)

        self.image_shape = (H, W, C)
        self.weights = jax.nn.softmax(logits, axis=-1)           # (B,K)
        self.logits  = logits
        self.clip_epsilon = clip_epsilon

        batch_shape = (B,)
        event_shape = self.image_shape

        super().__init__(
            dtype=loc.dtype,
            reparameterization_type=tfd.FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            name=name,
        )

        self._batch_shape_tensor = jnp.array(batch_shape)
        self._event_shape_tensor = jnp.array(event_shape)

    # ------------------------------------------------------------------ #
    # Differentiable sample (weighted sum of reparameterised samples)
    # ------------------------------------------------------------------ #
    def sample(self, sample_shape=(), seed: jax.random.PRNGKey = None, **kwargs):
        sample_shape = tuple(sample_shape)  # (*S)

        eps = jax.random.normal(
            seed,
            shape=sample_shape + self.loc.shape,      # (*S,B,K,H,W,C)
            dtype=self.dtype,
        )
        if self.clip_epsilon is not None:
            eps = jnp.clip(eps, -self.clip_epsilon, self.clip_epsilon)

        comps = self.loc + self.scale * eps      # (*S,B,K,H,W,C)

        soft = jnp.einsum('bk,...bkhwc->...bhwc', self.weights, comps)
        return soft

    # ------------------------------------------------------------------ #
    # Exact mixture log-prob (expects x with image shape)
    # ------------------------------------------------------------------ #
    def log_prob(self, x, **kwargs):
        x_flat = jnp.reshape(x, x.shape[:-3] + (-1,))             # (*S,B,D)
        log_mix = jax.nn.log_softmax(self.logits, axis=-1)       # (B,K)
        log_probs = self._components_flat.log_prob(x_flat[..., None, :])   # (*S,B,K)
        return jax.scipy.special.logsumexp(log_mix + log_probs, axis=-1)

    # Convenience stats
    def mean(self, **kwargs):
        mean_flat = jnp.sum(self.weights[..., None] * self._components_flat.loc, axis=-2)
        return mean_flat.reshape(self.batch_shape + self.event_shape)

    # Required shape accessors
    def batch_shape_tensor(self, **kwargs):  return self._batch_shape_tensor
    def event_shape_tensor(self, **kwargs):  return self._event_shape_tensor

    def tree_flatten(self):
        children = (self.logits, self.loc, self.scale)
        aux_data = {
            'clip_epsilon': self.clip_epsilon,
            'dtype': self.dtype,
            'name': self.name,
            'validate_args': self.validate_args,
            'allow_nan_stats': self.allow_nan_stats,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        logits, loc, scale = children
        return cls(logits=logits, loc=loc, scale=scale, **aux_data)


class LipschitzNormal(tfd.TransformedDistribution):

    def __init__(self, *, loc, scale, noise_bound=1., validate_args=False, allow_nan_stats=True):
        parameters = dict(locals())
        self.loc = loc
        self.scale = scale
        esp_dist = tfd.TruncatedNormal(
            loc=jnp.zeros(loc.shape, dtype=loc.dtype),
            scale=jnp.ones(scale.shape, dtype=scale.dtype),
            low=-noise_bound, high=noise_bound,
        )
        affine = tfb.Chain([
            tfb.Shift(shift=loc),
            tfb.Scale(scale=scale),
        ])
        super().__init__(
            esp_dist, affine, validate_args=validate_args)
        self._parameters = parameters


class DropTFPSubclassParamProps(logging.Filter):
    """Filter out annoying TFP warning messages about subclassing Distributions"""
    _pat = re.compile(r"Distribution subclass .* inherits `_parameter_properties.*", re.DOTALL)
    def filter(self, record: logging.LogRecord) -> bool:
        return not self._pat.search(record.getMessage())

root = logging.getLogger()          # or a more specific logger if you know it
root.addFilter(DropTFPSubclassParamProps())
logging.getLogger('absl').addFilter(DropTFPSubclassParamProps())

class STOneHotCategorical(tfd.OneHotCategorical):

    def __init__(self, logits=None, probs=None, use_gumbel_softmax=False):
        super().__init__(logits=logits, probs=probs)
        self.use_gumbel_softmax = use_gumbel_softmax

    def mode(self, **kwargs):
        return super().mode(**kwargs).astype(jnp.float32)

    def sample(self, sample_shape=(), seed: jax.random.PRNGKey = None, **kwargs):
        if self.use_gumbel_softmax:
            logits = self.logits_parameter()
            probs = self.probs_parameter() if logits is None else None
            sample = tfd.RelaxedOneHotCategorical(
                temperature=1.0, logits=logits, probs=probs
            ).sample(sample_shape=sample_shape, seed=seed)
            indices = jnp.argmax(sample, axis=-1)
            one_hot = jax.nn.one_hot(indices, sample.shape[-1], axis=-1)  # [B, n_cat, n_cls]
            return one_hot.astype(jnp.float32) + sample - jax.lax.stop_gradient(sample)
        else:
            # Straight through biased gradient estimator.
            sample = super().sample(sample_shape=sample_shape, seed=seed)
            probs = self._pad(super().probs_parameter(), sample.shape)
            sample += probs - jax.lax.stop_gradient(probs)
        return sample

    def _pad(self, tensor, shape):
        tensor = super().probs_parameter() if tensor is None else tensor
        while len(tensor.shape) < len(shape):
          tensor = tensor[None]
        return tensor

    def relaxed_cross_entropy(self, target, seed: jax.random.PRNGKey = None):
        """
        Computes the relaxed cross-entropy loss between the target and the distribution.
        """
        logits = self.logits_parameter()
        gumbels = tfd.Gumbel(loc=logits, scale=1.).sample(seed=seed)
        return optax.softmax_cross_entropy(logits=gumbels, labels=target)

    def relaxed_js_distance(self, target, seed: jax.random.PRNGKey = None, eps=1e-8):
        """
        Computes the relaxed Jensen-Shannon distance between the target and the distribution.
        """
        logits = self.logits_parameter()
        gumbels = tfd.Gumbel(loc=logits, scale=1.).sample(seed=seed)
        return js_distance(p=None, q=target, logits_p=gumbels, logits_q=None, eps=1e-8)

class STBernoulli(tfd.Bernoulli):

    def __init__(self, logits=None, probs=None):
        super().__init__(logits=logits, probs=probs)

    def mode(self, **kwargs):
        return super().mode(**kwargs).astype(jnp.float32)

    def sample(self, sample_shape=(), seed: jax.random.PRNGKey = None, **kwargs):
        logits = self.logits_parameter()
        probs = self.probs_parameter() if logits is None else None
        sample = tfd.RelaxedBernoulli(
            temperature=1.0, logits=logits, probs=probs
        ).sample(sample_shape=sample_shape, seed=seed)
        bernoulli = jnp.round(sample)
        return bernoulli.astype(jnp.float32) + sample - jax.lax.stop_gradient(sample)
