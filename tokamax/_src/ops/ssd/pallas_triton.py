# SPDX-License-Identifier: Apache-2.0
"""Pallas-Triton implementation of full-sequence SSD forward."""

import dataclasses
import functools
from typing import Any, ClassVar, override

import immutabledict
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
import jax.numpy as jnp
from tokamax._src import gpu_utils
from tokamax._src.ops import op
from tokamax._src.ops.ssd import base
from tokamax._src.ops.ssd.pallas_triton_config import Config

type Key = immutabledict.immutabledict[str, Any]


def recurrence_kernel(
  x_ref: jax.Array,
  decay_ref: jax.Array,
  b_ref: jax.Array,
  c_ref: jax.Array,
  initial_ref: jax.Array,
  output_ref: jax.Array,
  final_ref: jax.Array,
  *,
  block_p: int,
) -> None:
  batch, head = pl.program_id(0), pl.program_id(1)
  length, heads, head_dim = x_ref.shape[1:]
  group = head // (heads // b_ref.shape[2])
  state_dim = initial_ref.shape[-1]
  p = pl.program_id(2) * block_p + jnp.arange(block_p)
  n = jnp.arange(pl.next_power_of_2(state_dim))
  mask = (p[:, None] < head_dim) & (n[None, :] < state_dim)
  state = plgpu.load(
    initial_ref.at[batch, head, p[:, None], n[None, :]], mask=mask, other=0.0
  )

  def step(t: jax.Array, state: jax.Array) -> jax.Array:
    x_t = plgpu.load(x_ref.at[batch, t, head, p], mask=p < head_dim, other=0.0)
    decay = jnp.exp(plgpu.load(decay_ref.at[batch, t, head]))
    b_t = plgpu.load(
      b_ref.at[batch, t, group, n], mask=n < state_dim, other=0.0
    )
    c_t = plgpu.load(
      c_ref.at[batch, t, group, n], mask=n < state_dim, other=0.0
    )
    state = decay * state + x_t[:, None] * b_t[None, :]
    output = jnp.sum(state * c_t[None, :], axis=1)
    plgpu.store(output_ref.at[batch, t, head, p], output, mask=p < head_dim)
    return state

  state = jax.lax.fori_loop(0, length, step, state)
  plgpu.store(
    final_ref.at[batch, head, p[:, None], n[None, :]], state, mask=mask
  )


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PallasTritonSSD(base.SSD[Config, Key]):
  """Fused recurrent SSD kernel with Tokamax autotuning support."""

  config_cls: ClassVar[type[Config]] = Config

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
    config: Config,
  ) -> tuple[base.Outputs, None]:
    del return_residuals
    kernel = functools.partial(recurrence_kernel, block_p=config.block_p)
    outputs = pl.pallas_call(
      kernel,
      out_shape=(
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        jax.ShapeDtypeStruct(initial_state.shape, initial_state.dtype),
      ),
      grid=(x.shape[0], x.shape[2], pl.cdiv(x.shape[3], config.block_p)),
      compiler_params=plgpu.CompilerParams(num_warps=config.num_warps),
      name='mamba2_ssd_recurrence',
    )(x, log_decay, b, c, initial_state)
    return outputs, None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    del ba
    return Config(block_p=16, num_warps=4)

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    del ba
    return {
      Config(block_p=block_p, num_warps=num_warps)
      for block_p in (4, 8, 16, 32)
      for num_warps in (4, 8)
    }

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return gpu_utils.has_triton_support(device)
