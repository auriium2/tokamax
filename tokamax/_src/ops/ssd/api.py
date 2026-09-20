# SPDX-License-Identifier: Apache-2.0
"""Public API for discretized Mamba2 SSD forward."""

from typing import Literal

import immutabledict
import jax
from tokamax._src import gpu_utils
from tokamax._src.ops.ssd import base
from tokamax._src.ops.ssd import pallas_triton

type Implementation = Literal['xla', 'triton']

IMPLEMENTATIONS = immutabledict.immutabledict(
  xla=base.SSD(), triton=pallas_triton.PallasTritonSSD()
)


def ssd(
  x: jax.Array,
  log_decay: jax.Array,
  b: jax.Array,
  c: jax.Array,
  initial_state: jax.Array,
  *,
  implementation: 'Implementation|None' = None,
) -> base.Outputs:
  """Return SSD sequence outputs and final state from discretized inputs."""
  if implementation is None:
    implementation = 'triton' if gpu_utils.has_triton_support() else 'xla'
  if implementation not in IMPLEMENTATIONS:
    raise ValueError(f'Unknown SSD implementation: {implementation}')
  return IMPLEMENTATIONS[implementation](x, log_decay, b, c, initial_state)
