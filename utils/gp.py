# gp_utils.py
from functools import partial
import jax
import jax.numpy as jnp
from typing import Callable, Tuple
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

Array = jnp.ndarray

# ----------------------------------------------------------------------------------------------------------
# 1)  Simple ||grad||² penalty (DeepMDP style)
# See https://github.com/jbuckman/dmdp-donutworld/blob/1b59ac59c6eb4935a9e2c00ea097bf0d2a14a8e0/dmdp.py#L72
# ----------------------------------------------------------------------------------------------------------
def lipschitz_gp(
    f: Callable[[Array, jax.random.PRNGKey], Array],
    x: Array,
    keys: Array,
    min_gradient_norm_epsilon: float = 1e-7
) -> Array:
    """
    λ ·  E[ ||∇_x f(x)||² ]    (batched over first axis)
    `x` shape (B, …).  `f(x)` returns a vector per sample.
    """
    def scalar_fn(xi, key):
        return jnp.sum(f(xi, key))  # reduce vector to scalar

    grad_fn = jax.vmap(jax.grad(scalar_fn))
    grads   = grad_fn(x, keys)                           # (B, …)
    squared_slopes  = jnp.sum(grads**2, axis=-1) + 1e-8   # (B,)
    return jnp.mean(squared_slopes)

def lipschitz_gp_conv_output(
    f: Callable[[Array, jax.random.PRNGKey], Array],
    x: Array,
    keys: Array,
) -> Array:
    """
    Computes λ · E[ ||∇ₓ f(x)||_F² ] where f(xᵢ) is vector-valued
    """
    # Get Jacobian of f w.r.t. input xᵢ — output shape: (output_dim, input_dim)
    def single_jac(xi, key):
        return jax.jacrev(f, argnums=0)(xi, key)

    # Vectorize across batch
    batched_jac = jax.vmap(single_jac)  # output: (output_dim, input_dim)

    jacobians = batched_jac(x, keys)

    # Compute Frobenius norm squared per sample
    frob_sq = jnp.sum(jacobians**2, axis=(1, 2))  # shape (B,)
    return jnp.mean(frob_sq)


# ---------------------------------------------------------------------
# 2)  WGAN-GP (||grad|| ≈ 1 on interpolations)
# ---------------------------------------------------------------------
def wgan_gp(
    f: Callable[[Array, jax.random.PRNGKey], Array],
    x_real: Array,
    x_fake: Array,
    rng_key: jax.random.PRNGKey,
) -> Tuple[Array, jax.random.PRNGKey]:
    """
    E[(||∇_{x̃} f(x̃)||₂ - 1)²]   where   x̃ = ε·x_real + (1-ε)·x_fake

    Returns: (penalty, new_key)
    Notes:
      * `f` must output a scalar per sample.
      * `x_real` and `x_fake` must have identical shape (B, …).
    """
    B = x_real.shape[0]

    rng_key, eps_key = jax.random.split(rng_key)
    eps = jax.random.uniform(eps_key, (B,) + (1,) * (x_real.ndim - 1))
    x_tilde = eps * x_real + (1.0 - eps) * x_fake  # (B, …)

    # Batched gradient wrt x_tilde
    rng_key, tr_key = jax.random.split(rng_key)
    keys = jax.random.split(tr_key, B)  # (B, 2)

    def scalar_fn(xi, key):
        return jnp.sum(f(xi, key))  # reduce vector to scalar

    grad_fn = jax.vmap(jax.grad(scalar_fn))
    grads   = grad_fn(x_tilde, keys)                           # (B, …)
    # Compute the L2 norm of the gradients
    slopes  = jnp.sqrt(jnp.sum(grads**2, axis=-1) + 1e-8)      # (B,)

    penalty = jnp.mean((slopes - 1.0) ** 2)
    return penalty, rng_key

# --------------------------------------------------------------------
# Flatten *all* numeric parameters of a TFP distribution
# --------------------------------------------------------------------
def flatten_dist_params(dist: tfd.Distribution) -> jnp.ndarray:
    """
    Convert a TFP distribution into a single 1-D jax.Array of parameters.

    Works for Normal, Categorical, MixtureSameFamily, etc.
    Non-array parameters (strings, booleans) are skipped.
    """
    leaves = []
    for v in jax.tree_util.tree_leaves(dist.parameters):
        # skip non-numeric entries
        if isinstance(v, (float, int)) or hasattr(v, "dtype"):
            leaves.append(jnp.ravel(jnp.asarray(v)))
    if not leaves:
        # Fallback: use mean if parameters unavailable
        leaves.append(jnp.ravel(dist.mean()))
    return jnp.concatenate(leaves, axis=0)