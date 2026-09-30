"""Regression checks for strict conditional masks and the full density score."""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from dsb.made import ImageMADE, MADE


def test_conditional_masks_and_full_score_gradient():
    net = MADE(3, hidden=12, depth=2, components=2, key=jax.random.PRNGKey(7))
    x = jnp.array([0.3, -0.5, 0.8])
    forbidden = np.arange(3)[:, None] <= np.arange(3)[None, :]
    for jac in jax.jacfwd(lambda z: net.conditional_parameters(4, z))(x):
        assert np.all(np.asarray(jac).transpose(0, 2, 1)[forbidden] == 0)

    score_jac = np.asarray(jax.jacfwd(lambda z: net(4, z))(x))
    assert np.all(np.abs(np.diag(score_jac)) > 1e-7)
    assert np.any(np.abs(score_jac[np.triu_indices(3, k=1)]) > 1e-7)
    np.testing.assert_allclose(score_jac, score_jac.T, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(net(4, x), net.gain[4] * jax.grad(lambda z: net.log_prob(4, z))(x))
    np.testing.assert_allclose(net(4, x.reshape(1, 3)), net(4, x).reshape(1, 3))

    strict = MADE(3, hidden=12, depth=2, components=2, mode="strict", key=jax.random.PRNGKey(7))
    strict_jac = np.asarray(jax.jacfwd(lambda z: strict(4, z))(x))
    assert np.all(strict_jac[forbidden] == 0)

    params, static = eqx.partition(net, eqx.is_array)
    loss, grads = jax.value_and_grad(
        lambda p: jnp.mean((eqx.combine(p, static)(4, x) - jnp.array([0.2, -0.4, 0.1])) ** 2)
    )(params)
    assert np.isfinite(loss)
    assert all(np.all(np.isfinite(g)) for g in jax.tree_util.tree_leaves(grads))
    density_grads = jax.tree_util.tree_leaves(grads.layers)
    assert sum(float(jnp.sum(g**2)) for g in density_grads) > 1e-12


def test_one_dimensional_mixture_is_normalized_and_time_dependent():
    net = MADE(1, hidden=8, depth=2, components=3, key=jax.random.PRNGKey(12))
    x = jnp.array([0.0])
    parameters = net.conditional_parameters(0, x)
    later_parameters = net.conditional_parameters(10, x)
    assert any(not np.allclose(a, b) for a, b in zip(parameters, later_parameters))
    _, locations, log_scales = parameters
    assert np.ptp(np.asarray(locations)) > 1e-5
    scales = jnp.exp(log_scales)
    grid = jnp.linspace(jnp.min(locations - 10 * scales), jnp.max(locations + 10 * scales), 4097)
    density = jax.vmap(lambda z: jnp.exp(net.log_prob(0, z[None])))(grid)
    integral = jnp.trapezoid(density, grid)
    np.testing.assert_allclose(integral, 1.0, atol=2e-4, rtol=0)
    assert np.all(np.isfinite(net(0, x)))


def test_width_supplies_multiple_data_conditioned_features():
    """A wide 2-D conditioner must not collapse to one scalar hidden channel."""
    net = MADE(2, hidden=16, depth=2, components=2, key=jax.random.PRNGKey(7))

    def conditional_outputs(z):
        logits, locations, _ = net.conditional_parameters(4, jnp.array([z, 0.0]))
        return jnp.concatenate((logits[1], locations[1]))

    derivatives = jax.vmap(jax.jacfwd(conditional_outputs))(jnp.linspace(-3.0, 3.0, 16))
    singular_values = np.linalg.svd(np.asarray(derivatives), compute_uv=False)
    assert singular_values[1] > 1e-5


def test_image_score_respects_raster_rgb_conditionals_and_fine_scales():
    net = ImageMADE(3072, hidden=16, depth=1, components=1, num_steps=1,
                    output_scale=1.0, key=jax.random.PRNGKey(7))
    net = eqx.tree_at(lambda m: m.layers, net, jax.tree_util.tree_map(jnp.zeros_like, net.layers))
    parent, child, fine = (2, 1, 31), (0, 2, 0), (1, 1, 1)
    net = eqx.tree_at(
        lambda m: (m.layers[0].weight, m.layers[-1].weight, m.layers[-1].bias), net,
        (net.layers[0].weight.at[1, 191].set(1.0),
         net.layers[-1].weight.at[3 * 192 + 1, 1].set(0.4),
         net.layers[-1].bias.at[3 * 100 + 2].set(7 * np.arctanh(np.log(0.01) / 7))),
    )
    x = jnp.zeros((3, 32, 32)).at[parent].set(0.7).at[child].set(-0.3).at[fine].set(0.02)
    sigmoid = 1 / (1 + np.exp(-0.7))
    location = 0.4 * 0.7 * sigmoid
    derivative = 0.4 * (sigmoid + 0.7 * sigmoid * (1 - sigmoid))
    expected = (-x).at[parent].set(-0.7 + derivative * (-0.3 - location))
    expected = expected.at[child].set(0.3 + location).at[fine].set(-0.02 / 0.01**2)
    np.testing.assert_allclose(net(1, x), expected, atol=3e-4, rtol=3e-5)
