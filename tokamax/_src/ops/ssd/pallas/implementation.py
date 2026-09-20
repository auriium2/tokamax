# SPDX-License-Identifier: Apache-2.0
"""Agent-owned Pallas implementation entry point."""

from collections.abc import Hashable
from typing import override

import jax
from tokamax._src.ops import op
from tokamax._src.ops.ssd import base


class PallasSSD(base.SSD[op.NullConfig, Hashable]):
  """Unimplemented GPU backend for the SSD assignment."""

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
    config: op.NullConfig,
  ) -> tuple[base.Outputs, None]:
    raise NotImplementedError(
      'The Pallas SSD backend is the assignment to implement.'
    )

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return device.platform == 'gpu'
