# SPDX-License-Identifier: Apache-2.0

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np
import tokamax
from tokamax._src.ops import op
from tokamax._src.ops.ssd import api
from tokamax._src.ops.ssd import pallas_triton


def make_inputs(length: int) -> tuple[jax.Array, ...]:
  rng = np.random.default_rng(8)
  shapes = ((1, length, 4, 7), (1, length, 4), (1, length, 2, 9))
  x = (0.25 * rng.standard_normal(shapes[0])).astype(np.float32)
  decay = rng.uniform(-0.2, -0.001, shapes[1]).astype(np.float32)
  b = (0.25 * rng.standard_normal(shapes[2])).astype(np.float32)
  c = (0.25 * rng.standard_normal(shapes[2])).astype(np.float32)
  state = rng.standard_normal((1, 4, 7, 9)).astype(np.float32)
  return tuple(jnp.asarray(value) for value in (x, decay, b, c, state))


class SSDTest(parameterized.TestCase):
  def test_scalar_recurrence(self) -> None:
    x = jnp.ones((1, 3, 1, 1), jnp.float32)
    decay = jnp.zeros((1, 3, 1), jnp.float32)
    initial = jnp.full((1, 1, 1, 1), 2, jnp.float32)
    y, state = tokamax.ssd(x, decay, x, x, initial, implementation='xla')
    np.testing.assert_array_equal(y.ravel(), [3, 4, 5])
    np.testing.assert_array_equal(state.ravel(), [5])

  @parameterized.parameters(1, 7, 64)
  def test_state_continuation(self, split: int) -> None:
    inputs = make_inputs(65)
    forward = jax.jit(functools.partial(tokamax.ssd, implementation='xla'))
    expected, final = forward(*inputs)
    first = tuple(value[:, :split] for value in inputs[:4])
    rest = tuple(value[:, split:] for value in inputs[:4])
    left, middle = forward(*first, inputs[4])
    right, actual_final = forward(*rest, middle)
    np.testing.assert_allclose(
      jnp.concatenate((left, right), axis=1), expected, atol=2e-4, rtol=2e-4
    )
    np.testing.assert_allclose(actual_final, final, atol=2e-4, rtol=2e-4)

  @parameterized.product(length=(7, 65), block_p=(4, 16))
  def test_interpreted_pallas_tails(self, length: int, block_p: int) -> None:
    inputs = make_inputs(length)
    x, _, _, _, initial = inputs
    kernel = functools.partial(pallas_triton.recurrence_kernel, block_p=block_p)
    call = pl.pallas_call(
      kernel,
      out_shape=(
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        jax.ShapeDtypeStruct(initial.shape, initial.dtype),
      ),
      grid=(1, x.shape[2], pl.cdiv(x.shape[3], block_p)),
      interpret=True,
    )
    actual = call(*inputs)
    expected = tokamax.ssd(*inputs, implementation='xla')
    for got, want in zip(actual, expected, strict=True):
      np.testing.assert_allclose(got, want, atol=2e-4, rtol=2e-4)

  def test_tuning_configuration_serialization(self) -> None:
    bound = api.IMPLEMENTATIONS['triton'].bind(*make_inputs(7))
    self.assertLen(bound.autotuning_configs, 8)
    self.assertIn(bound.heuristics_config, bound.autotuning_configs)
    abstract = jax.tree.map(
      lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype),
      make_inputs(7),
    )
    bound = api.IMPLEMENTATIONS['triton'].bind(*abstract)
    restored = op.BOUND_ARGS_ADAPTER.validate_json(
      op.BOUND_ARGS_ADAPTER.dump_json(bound)
    )
    self.assertEqual(restored.autotuning_cache_key, bound.autotuning_cache_key)

  def test_invalid_inputs(self) -> None:
    inputs = make_inputs(7)
    with self.assertRaises(TypeError):
      tokamax.ssd(
        inputs[0].astype(jnp.float16), *inputs[1:], implementation='xla'
      )
    with self.assertRaises(ValueError):
      tokamax.ssd(*inputs[:4], inputs[4][..., :1], implementation='xla')
    with self.assertRaises(ValueError):
      tokamax.ssd(*inputs, implementation='unknown')

  def test_explicit_triton_rejects_cpu(self) -> None:
    if jax.default_backend() != 'cpu':
      self.skipTest('CPU-only dispatch check')
    with self.assertRaises(NotImplementedError):
      tokamax.ssd(*make_inputs(7), implementation='triton')


if __name__ == '__main__':
  absltest.main()
