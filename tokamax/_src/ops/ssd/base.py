# SPDX-License-Identifier: Apache-2.0
"""Full-sequence discretized Mamba2 SSD forward operation."""

from collections.abc import Hashable
from typing import Any, ClassVar, override

import jax
import jax.numpy as jnp
from tokamax._src.ops import op

type AbstractArray = jax.Array | jax.ShapeDtypeStruct
type Outputs = tuple[jax.Array, jax.Array]


class SSD[C, K: Hashable](op.Op[Any, Outputs, None, C, K]):
  """Evaluate SSD outputs and final state with an XLA recurrence."""

  supports_symbolic_shapes: ClassVar[bool] = False
  supports_batched_args_capture: ClassVar[bool] = False

  @override
  def bind(
    self,
    x: AbstractArray,
    log_decay: AbstractArray,
    b: AbstractArray,
    c: AbstractArray,
    initial_state: AbstractArray,
    *,
    return_residuals: bool = False,
  ) -> op.BoundArguments:
    if x.ndim != 4 or b.ndim != 4:
      raise ValueError('x and b must have rank four')
    batch, length, heads, head_dim = x.shape
    groups, state_dim = b.shape[2:]
    if min(batch, length, heads, head_dim, groups, state_dim) < 1:
      raise ValueError('SSD dimensions must be positive')
    if heads % groups:
      raise ValueError('heads must be divisible by groups')
    if log_decay.shape != (batch, length, heads):
      raise ValueError('Incompatible log_decay shape')
    if b.shape != (batch, length, groups, state_dim) or c.shape != b.shape:
      raise ValueError('Incompatible grouped B/C shapes')
    if initial_state.shape != (batch, heads, head_dim, state_dim):
      raise ValueError('Incompatible initial_state shape')
    arrays = (x, log_decay, b, c, initial_state)
    if any(array.dtype != jnp.float32 for array in arrays):
      raise TypeError('SSD currently requires float32 inputs')
    return super().bind(*arrays, return_residuals=return_residuals)

  @override
  def _fwd(
    self,
    x: jax.Array,
    log_decay: jax.Array,
    b: jax.Array,
    c: jax.Array,
    initial_state: jax.Array,
    *,
    return_residuals: bool,
    config: C,
  ) -> tuple[Outputs, None]:
    del return_residuals, config
    groups = jnp.arange(x.shape[2]) // (x.shape[2] // b.shape[2])
    sequence = tuple(
      jnp.swapaxes(array, 0, 1) for array in (x, log_decay, b, c)
    )

    def step(
      state: jax.Array, inputs: tuple[jax.Array, ...]
    ) -> tuple[jax.Array, jax.Array]:
      x_t, decay_t, b_t, c_t = inputs
      state = (
        jnp.exp(decay_t)[..., None, None] * state
        + x_t[..., None] * b_t[:, groups, None, :]
      )
      output = jnp.sum(state * c_t[:, groups, None, :], axis=-1)
      return state, output

    final_state, outputs = jax.lax.scan(step, initial_state, sequence)
    return (jnp.swapaxes(outputs, 0, 1), final_state), None
