from tensorflow_probability.substrates import jax as tfp
import jax.numpy as jnp
import jax
tfb = tfp.bijectors

class StableSigmoid(tfb.Sigmoid):
    """
    A stable version of the sigmoid function that avoids numerical instabilities.
    """
    def __init__(self, epsilon: float = 1e-6):
        super(StableSigmoid, self).__init__()
        self._sigmoid = tfb.Sigmoid()
        self._epsilon = epsilon
        self._low = self._sigmoid.inverse(self._epsilon)
        self._high = self._sigmoid.inverse(1. - self._epsilon)

    def _forward(self, logits):
        clipped_logits = jnp.clip(logits, self._low , self._high)
        logits = logits + jax.lax.stop_gradient(clipped_logits - logits)

        # apply straight through gradients for damped logits
        res = self._sigmoid(logits)
        res = jnp.where(
            res  <= 1.1 * self._epsilon,
            res - jax.lax.stop_gradient(res),
            res)
        res = jnp.where(
            res >= 1. - 1.1 * self._epsilon,
            res + jax.lax.stop_gradient(1. - res),
            res)

        return res
