from enum import Enum
import jax.numpy as jnp
import optax
from jax import scipy as jsp
import jax


class CategoricalCost(str, Enum):
    """
    Enum for one-hot encoding cost types.
    """
    L2 = "l2"  # L2 cost
    CROSS_ENTROPY = "cross_entropy"  # Cross-entropy cost
    HAMMING = "hamming"
    JENSEN_SHANNON = "jensen_shannon"  # Jensen-Shannon distance

def smooth_l1_loss(pred: jnp.ndarray, target: jnp.ndarray, beta: float = 1.0, reduction: str = "mean", axis=-1) -> jnp.ndarray:
    """
    Computes the Smooth L1 Loss between `pred` and `target`.

    Args:
        pred: Predicted values (same shape as target).
        target: Ground truth values.
        beta: Transition point from L2 to L1 loss. Default is 1.0.
        reduction: One of 'none', 'mean', or 'sum'.

    Returns:
        Scalar loss if reduction is 'mean' or 'sum', else the loss per element.
    """
    # loss = optax.huber_loss(pred, target, delta=beta)
    error = pred - target
    abs_error = jnp.abs(error)
    loss = jnp.where(abs_error < beta, 0.5 * (error ** 2) / beta, abs_error - 0.5 * beta)


    if reduction == "mean":
        return jnp.mean(loss, axis=axis)
    elif reduction == "sum":
        return jnp.sum(loss, axis=axis)
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}. Choose 'none', 'mean', or 'sum'.")

def hamming_distance(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Hamming distance between two arrays of shape (B, C1, C2)."""
    if a.shape != b.shape:
        raise ValueError("Input arrays must have the same shape.")
    match = jnp.sum(a * b, axis=-1)
    dist = jnp.sum(1. - match, axis=-1)
    return dist / a.shape[-1]  # Normalize by the number of elements in the last dimension

def kl(p, q, axis=-1, eps=1e-8):
    """KL(P‖Q) with no NaNs."""
    p_ok = p > eps
    safe_p = jnp.where(p_ok, p, 1.)
    safe_q = jnp.where(p_ok, q, 1.)
    return jnp.sum(
        jnp.where(p_ok, jax.lax.mul(safe_p, jax.lax.log(safe_p + eps) - jax.lax.log(safe_q + eps)), jnp.zeros_like(p_ok)),
        axis=axis)

def js_distance(p, q, logits_p=None, logits_q=None, eps=1e-8):
    """
    √Jensen–Shannon distance between two “prob” vectors.
    """
    if logits_p is not None:
        log_p = jax.nn.log_softmax(logits_p, axis=-1)
        p = jnp.exp(log_p)
    else:
        p = jnp.clip(p, 2. * eps, 1. - 2. * eps)
    if logits_q is not None:
        log_q = jax.nn.log_softmax(logits_q, axis=-1)
        q = jnp.exp(log_q)
    else:
        q = jnp.clip(q, 2. * eps, 1. - 2. * eps)
    m = 0.5 * (p + q)

    js_div = 0.5 * kl(p, m, eps=eps) + 0.5 * kl(q, m, eps=eps)
    js_div = jnp.where(js_div > eps, js_div, jnp.zeros_like(js_div))  # Avoid NaNs
    return jnp.sqrt(js_div)            # √JS = metric distance
