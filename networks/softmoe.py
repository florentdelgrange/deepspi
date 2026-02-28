# coding=utf-8
# Copyright 2023 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implementation of SoftMoE for Dopamine networks."""
from flax import struct
from typing import Any

import math
import chex
import flax.linen as nn
import jax
import jax.numpy as jnp

@struct.dataclass
class RouterReturn:
  output: jax.Array
  probabilities: jax.Array
  top_expert_weights: jax.Array
  top_experts: jax.Array


@struct.dataclass
class MoEModuleReturn:
  values: jax.Array
  router_out: RouterReturn
  experts_hidden: 'jax.Array | None' = None


def l2_normalize(x, axis, eps=1e-6):
  norm = jnp.sqrt(jnp.square(x).sum(axis=axis, keepdims=True))
  return x * jnp.reciprocal(norm + eps)


class SoftMoE(nn.Module):
  """Soft Mixture of Experts (https://arxiv.org/abs/2308.00951)."""

  module: nn.Module
  num_experts: int
  num_outputs: int = 1
  capacity_factor: float = 1.0
  kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
  expert_type: str = "SMALL"
  normalization: bool = False

  @nn.compact
  def __call__(self, x: jax.Array) -> MoEModuleReturn:
    chex.assert_rank(x, 3)

    # Create phi weight matrix of size (d x n.p), where d is the token dim,
    # n is the number of experts and p is the capacity of each expert (#slots).
    # TODO(gsokar) implementation detail missing. Normalize input and weight
    # The paper states that it make a difference for large tokens
    batch_size, num_tokens, token_length = x.shape
    # capacity of each expert
    # we use ceil to allow for per sample token, where the capacity will be 1.
    if self.expert_type == "BIG":
      num_slots = int(
          math.ceil(num_tokens * self.num_outputs * self.capacity_factor / self.num_experts)
      )
      num_slots_sqrt = math.floor(math.sqrt(num_slots))
      num_slots = int(num_slots_sqrt**2)
    else:
      num_slots = int(
          math.ceil(num_tokens * self.num_outputs * self.capacity_factor / self.num_experts)
      )


    if self.num_outputs > 1:
        upscale = self.param('upscale', nn.initializers.ones, (self.num_outputs,))
        x = jnp.einsum("bmd,o->bmod",x, upscale)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])  # (b, m * o, d)

    phi_weights = self.param(
        "phi_weights",
        self.kernel_init,
        (token_length, self.num_experts, num_slots),
    )
    scale_value = self.param("scalar", nn.initializers.ones, (1,))
    # Calculate the weight of each token per slot.
    if self.normalization:
      x_normalized = l2_normalize(x, axis=2)
      phi_weights = scale_value[jnp.newaxis, jnp.newaxis, :].repeat(
          phi_weights.shape[0], axis=0
      ).repeat(phi_weights.shape[1], axis=1).repeat(
          phi_weights.shape[2], axis=2
      ) * l2_normalize(
          phi_weights, axis=0
      )
    else:
      x_normalized = x

    logits = jnp.einsum(
        "bmd,dnp->bmnp",
        x_normalized,
        phi_weights,
    )
    dispatch_weights = jax.nn.softmax(logits, axis=1)
    combine_weights = jax.nn.softmax(logits, axis=(2, 3))

    # Calculate the input tokens to the experts.
    mixture_inputs = jnp.einsum("bmd,bmnp->bnpd", x, dispatch_weights)
    # Make sure to convert out-of-bounds nans to zeros
    mixture_inputs = jnp.nan_to_num(mixture_inputs)

    if self.expert_type == "BIG":
      dim = int(math.sqrt(num_slots))
      mixture_inputs = mixture_inputs.reshape(
          batch_size, self.num_experts, 1, dim, dim, token_length
      )

    # Forward pass the MoE
    # This part is taken from MOE class.
    # The input shape should be (num_experts, max_capacity, -1)
    # nn.vmap will map over num_experts without parameter sharing,
    # i.e., it'll use a different initialization for each expert.
    # From there it'll vmap the model over the `max_capacity` dimension.
    experts, experts_hidden = nn.vmap(
        lambda module, x: jax.vmap(module)(x),
        in_axes=1,
        out_axes=1,
        variable_axes={"params": 0},
        split_rngs={"params": True},
        axis_size=self.num_experts,
        # TODO(jfarebro): Supply logical sharding axes
    )(self.module, mixture_inputs)

    if self.expert_type == "BIG":
      experts = experts.reshape(batch_size, self.num_experts, num_slots, token_length)

    # Keep the batch dimension: result shape (batch_size, num_tokens * num_outputs, token_length)
    outputs = jnp.einsum("bnpd,bmnp->bmd", experts, combine_weights)

    probabilities = combine_weights.mean(axis=-1)
    router_out = RouterReturn(
        output=jnp.empty_like(probabilities),
        probabilities=probabilities,
        top_expert_weights=jnp.empty([1]),
        top_experts=jnp.argmax(
            combine_weights.mean(axis=-1), axis=2, keepdims=True
        ),
    )
    return MoEModuleReturn(
        values=outputs,
        router_out=router_out,
        experts_hidden=experts_hidden,
    )

class ExpertModel(nn.Module):
  """The MLP network of an expert."""

  expert_hidden_size: int
  num_layers: int = 1
  maintain_token_size: bool = True
  eval_mode: bool = False
  initializer: Any = nn.initializers.xavier_uniform()
  bias_init: Any = nn.initializers.zeros_init()
  raw_output: bool = False

  @nn.compact
  def __call__(self, x):
    initializer = nn.initializers.xavier_uniform()
    token_size = x.shape[-1]

    for i in range(self.num_layers):
        x = nn.Dense(
            features=self.expert_hidden_size,
            kernel_init=self.initializer,
            bias_init=self.bias_init,
        )(x)
        if i < self.num_layers - 1 or not self.raw_output:
            x = nn.relu(x)

    hidden_x = x
    if self.maintain_token_size:
      x = nn.Dense(features=token_size, kernel_init=initializer)(x)
    return x, hidden_x
