"""Focused causal and score checks for the convolutional CIFAR MADE."""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dsb.conv_made import ConvMADE


def _pixel(row, column, channel):
    return 3 * (32 * row + column) + channel


def _raster_gradient(gradient):
    return np.asarray(gradient).transpose(1, 2, 0).reshape(-1)


def test_conditional_parameters_are_raster_causal_at_rows_and_boundaries():
    net = ConvMADE(3072, hidden=6, depth=2, components=1, num_steps=1, key=jax.random.PRNGKey(4))
    x = jax.random.normal(jax.random.PRNGKey(5), (3, 32, 32))

    target = _pixel(4, 0, 2)
    gradient = jax.grad(
        lambda z: net.conditional_parameters(0, z)[1][target, 0]
    )(x)
    raster = _raster_gradient(gradient)

    # The target RGB channel and every later raster channel are forbidden.
    np.testing.assert_allclose(raster[target:], 0.0, atol=1e-7, rtol=0.0)
    # Same-row mixing of vertical features reaches beyond a single-stack
    # causal convolution's upper-right cone.
    assert abs(raster[_pixel(3, 2, 2)]) > 1e-9

    first_channel = _pixel(4, 0, 0)
    first_gradient = jax.grad(
        lambda z: net.conditional_parameters(0, z)[1][first_channel, 0]
    )(x)
    np.testing.assert_allclose(
        _raster_gradient(first_gradient)[first_channel:], 0.0, atol=1e-7, rtol=0.0
    )

    green = _pixel(4, 5, 1)
    gradient = jax.grad(lambda z: net.conditional_parameters(0, z)[1][green, 0])(x)
    raster = _raster_gradient(gradient)
    np.testing.assert_allclose(raster[green:], 0.0, atol=1e-7, rtol=0.0)
    assert abs(raster[_pixel(4, 4, 2)]) > 1e-9
    assert abs(raster[_pixel(4, 5, 0)]) > 1e-9


def test_score_and_parameter_gradients_are_finite_and_mixed():
    net = ConvMADE(3072, hidden=6, depth=2, components=2, num_steps=2,
                   output_scale=1.0, key=jax.random.PRNGKey(8))
    x = jax.random.normal(jax.random.PRNGKey(9), (3, 32, 32))
    params, static = eqx.partition(net, eqx.is_array)
    target = jnp.zeros_like(x)
    loss, gradients = jax.value_and_grad(
        lambda p: jnp.mean((eqx.combine(p, static)(1, x) - target) ** 2)
    )(params)

    assert np.isfinite(float(loss))
    leaves = jax.tree_util.tree_leaves(gradients)
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves)
    assert float(jnp.linalg.norm(gradients.layers[0].horizontal.conv.weight)) > 0


def test_inherited_density_score_uses_raster_rgb_ordering():
    net = ConvMADE(3072, hidden=3, depth=1, components=1, num_steps=2, key=jax.random.PRNGKey(10))
    net = jax.tree_util.tree_map(lambda leaf: jnp.zeros_like(leaf) if eqx.is_array(leaf) else leaf, net)
    net = eqx.tree_at(
        lambda m: (m.layers[0].horizontal.conv.weight, m.head.conv.weight, m.gain),
        net,
        (net.layers[0].horizontal.conv.weight.at[1, 0, 0, 1].set(0.5),
         net.head.conv.weight.at[4, 1, 0, 0].set(1.0), jnp.ones_like(net.gain)),
    )
    x = jax.random.normal(jax.random.PRNGKey(11), (3, 32, 32))
    red, green, blue = x
    z = 0.5 * red
    sigmoid = jax.nn.sigmoid(z)
    mean = z * sigmoid
    derivative = 0.5 * (sigmoid + z * sigmoid * (1 - sigmoid))
    residual = green - mean
    expected = jnp.stack((-red + derivative * residual, -residual, -blue))
    expected_log_prob = -0.5 * jnp.sum(red**2 + residual**2 + blue**2) - 1536 * np.log(2 * np.pi)
    np.testing.assert_allclose(net(1, x), expected, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(net(1, x.reshape(-1)), expected.reshape(-1), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(net.log_prob(1, x), expected_log_prob, atol=1e-3, rtol=1e-6)


def test_constructor_validates_cifar_score_arguments():
    key = jax.random.PRNGKey(12)
    with pytest.raises(ValueError):
        ConvMADE(3, key=key)
    with pytest.raises(ValueError):
        ConvMADE(3072, mode="strict", key=key)
    with pytest.raises(ValueError):
        ConvMADE(3072, hidden=5, key=key)
    with pytest.raises(ValueError):
        ConvMADE(3072, output_scale=float("nan"), key=key)
