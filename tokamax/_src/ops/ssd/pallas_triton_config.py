# SPDX-License-Identifier: Apache-2.0
"""Tuning configuration for the recurrent SSD kernel."""

import pydantic
from tokamax._src import pydantic as pydantic_lib


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  block_p: pydantic_lib.PowerOfTwo
  num_warps: pydantic_lib.PowerOfTwo
