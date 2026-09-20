# SPDX-License-Identifier: Apache-2.0

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import tokamax


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

  def test_explicit_pallas_rejects_cpu(self) -> None:
    if jax.default_backend() != 'cpu':
      self.skipTest('CPU-only dispatch check')
    with self.assertRaises(NotImplementedError):
      tokamax.ssd(*make_inputs(7), implementation='pallas')


if __name__ == '__main__':
  absltest.main()
