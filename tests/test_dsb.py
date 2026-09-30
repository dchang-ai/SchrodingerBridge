"""Quick CPU tests of the DSB building blocks (run: python -m pytest tests -q)."""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dsb import ipf
from dsb.data import cifar_to_uint8
from dsb.networks import ScaledIncrement, ToyMLP
from dsb.schedule import OUReference, symmetric_gammas
from dsb.train import output_scales


class RefNet(eqx.Module):
    """Increment network equal to the reference drift gamma_k f(x)."""

    ref: OUReference = eqx.field(static=True)

    def __call__(self, k, x):
        return jnp.asarray(self.ref.gammas)[k] * self.ref.drift(x)


def test_symmetric_schedule():
    g = symmetric_gammas(20, 1e-5, 1e-1)
    assert g.shape == (20,) and np.allclose(g, g[::-1]) and g[0] == np.float32(1e-5)
    g2 = symmetric_gammas(20, 1e-5, 1e-1, power=2)
    assert g2[1] < g[1] and g2.max() <= g.max() and np.allclose(g2, g2[::-1])


def test_closed_form_marginal_matches_simulation():
    ref = OUReference(symmetric_gammas(20), alpha=1.0, mean=0.0)
    key = jax.random.PRNGKey(0)
    x0 = jnp.asarray(np.random.default_rng(0).standard_normal((20000, 2)) * 0.3 + 2.0, jnp.float32)
    c = ipf.simulate(RefNet(ref), x0, jnp.asarray(ref.gammas), key, forward=True)
    for k in (5, 12, 20):
        sim = c.xs[k - 1]
        cf = ref.sample_marginal(jax.random.PRNGKey(k), jnp.full((x0.shape[0],), k), x0)
        assert abs(float(sim.mean()) - float(cf.mean())) < 0.03
        assert abs(float(sim.std()) - float(cf.std())) < 0.03
    assert int(c.ks[0]) == 1 and int(c.ks[-1]) == 20


def test_targets_are_prop3_targets():
    """target_k = F(k, X_k) - F(k, X_{k+1}) for the recorded pair."""
    ref = OUReference(symmetric_gammas(10), alpha=1.0)
    net = RefNet(ref)
    x0 = jnp.ones((4, 3))
    c = ipf.simulate(net, x0, jnp.asarray(ref.gammas), jax.random.PRNGKey(1), forward=True)
    # reconstruct X_k from X_{k+1}? easier: check the k=0 pair with x0 known
    k = jnp.zeros((4,), jnp.int32)
    f_x0 = ipf.batched_mean(net, k, x0)
    f_x1 = ipf.batched_mean(net, k, c.xs[0])
    assert jnp.allclose(c.targets[0], f_x0 - f_x1, atol=1e-6)


def test_backward_indexing_and_noise_free_last_step():
    ref = OUReference(symmetric_gammas(10), alpha=1.0)
    net = RefNet(ref)
    xN = jnp.zeros((5, 2))
    c = ipf.simulate(net, xN, jnp.asarray(ref.gammas), jax.random.PRNGKey(2), forward=False, last_step_noise=False)
    assert list(np.asarray(c.ks)) == list(range(9, -1, -1))
    # last step is deterministic: x_0 = B(1, x_1)
    x1 = c.xs[-2]
    b = ipf.batched_mean(net, jnp.ones((5,), jnp.int32), x1)
    assert jnp.allclose(c.xs[-1], b, atol=1e-6)


def test_reference_pairs_consistent_with_marginal():
    ref = OUReference(symmetric_gammas(20), alpha=1.0)
    x0 = jnp.zeros((50000, 1))
    ks, xk1, tgt = ipf.reference_pairs(ref, jax.random.PRNGKey(3), x0)
    A, S = ref.marginal_coefficients()
    for k in (1, 10, 20):
        sel = xk1[ks == k]
        assert abs(float(sel.std()) - S[k]) < 0.03, (k, float(sel.std()), S[k])


def test_cache_sampling_shapes():
    ref = OUReference(symmetric_gammas(6), alpha=1.0)
    c = ipf.simulate(RefNet(ref), jnp.zeros((7, 2)), jnp.asarray(ref.gammas), jax.random.PRNGKey(0), forward=True)
    ks, xs, tg = ipf.sample_cache(c, jax.random.PRNGKey(1), 32)
    assert ks.shape == (32,) and xs.shape == (32, 2) and tg.shape == (32, 2)
    assert int(ks.min()) >= 1 and int(ks.max()) <= 6


def test_scaled_increment_and_train_step():
    ref = OUReference(symmetric_gammas(6), alpha=1.0)
    sf, sb = output_scales(ref)
    net = ScaledIncrement(ToyMLP(dim=2, hidden=32, key=jax.random.PRNGKey(0)), sb)
    params, static = eqx.partition(net, eqx.is_array)
    assert all(isinstance(s, float) for s in net.scales) and len(sb) == 7
    opt = ipf.make_optimizer(1e-3)
    state = ipf.init_train_state(net, opt)
    step = ipf.make_train_step(opt, static, 0.99, False)
    c = ipf.simulate(RefNet(ref), jnp.zeros((16, 2)), jnp.asarray(ref.gammas), jax.random.PRNGKey(0), forward=True)
    ks, xs, tg = ipf.sample_cache(c, jax.random.PRNGKey(1), 16)
    state, loss = step(state, ks, xs, tg, None, jax.random.PRNGKey(2))
    assert jnp.isfinite(loss) and int(state.step) == 1


def test_shared_noise():
    ref = OUReference(symmetric_gammas(4), alpha=1.0)
    c = ipf.simulate(RefNet(ref), jnp.zeros((3, 2)), jnp.asarray(ref.gammas), jax.random.PRNGKey(0), forward=False, shared_noise=True)
    assert jnp.allclose(c.xs[:, 0], c.xs[:, 1])


def test_reference_residual_untrained_equals_reference():
    """With the reference drift as a fixed residual, a zero-output network is exactly F^0."""
    from dsb.networks import ReferenceResidual

    ref = OUReference(symmetric_gammas(10), alpha=0.7, mean=0.2)

    class Zero(eqx.Module):
        def __call__(self, k, x):
            return jnp.zeros_like(x)

    net = ReferenceResidual(Zero(), ref.gammas, ref.alpha, ref.mean)
    x = jnp.linspace(-2, 2, 6)[:, None]
    k = jnp.array([0, 3, 5, 7, 9, 2])
    assert jnp.allclose(ipf.batched_mean(net, k, x), ref.forward_mean(k, x), atol=1e-6)


def test_gamma_max_reached():
    g = symmetric_gammas(50, 1e-5, 1e-1)
    assert np.isclose(g.max(), 1e-1) and g[0] == np.float32(1e-5) and np.allclose(g, g[::-1])


def test_backward_step_uses_gamma_of_transition_asymmetric_schedule():
    """With an asymmetric schedule, the noise of the backward step k -> k-1 must use gammas[k-1]."""
    gammas = jnp.array([0.001, 0.01, 0.1, 1.0])
    ref = OUReference(np.asarray(gammas), alpha=0.0)
    net = RefNet(ref)  # zero drift: pure noise, so the std of each step is sqrt(2 gamma)
    xN = jnp.zeros((200000, 1))
    c = ipf.simulate(net, xN, gammas, jax.random.PRNGKey(0), forward=False)
    # xs[i] = X_{N-1-i}: the step producing X_3 (from X_4) uses gammas[3] = 1.0, X_0 (from X_1) uses gammas[0]
    stds = [float(jnp.std(c.xs[i] - (xN if i == 0 else c.xs[i - 1]))) for i in range(4)]
    assert np.allclose(stds, np.sqrt(2 * np.array([1.0, 0.1, 0.01, 0.001])), rtol=0.05), stds


def test_legacy_ramp_matches_first_runs():
    g = symmetric_gammas(50, 1e-5, 1e-1, legacy=True)
    assert np.isclose(g.sum(), 2.400, atol=2e-3) and g.max() < 0.1
    g2 = symmetric_gammas(50, 1e-5, 1e-1)
    assert np.isclose(g2.sum(), 2.500, atol=2e-3)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_cifar_export_rejects_nonfinite_pixels(value):
    pixels = jnp.zeros((2, 3, 32, 32)).at[1, 2, 31, 31].set(value)
    with pytest.raises(FloatingPointError, match="nonfinite"):
        cifar_to_uint8(pixels)


def test_cifar_export_preserves_rounding_and_clipping():
    pixels = jnp.asarray([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    exported = cifar_to_uint8(pixels)
    assert isinstance(exported, jax.Array)
    np.testing.assert_array_equal(exported, np.asarray([0, 0, 64, 128, 191, 255, 255], dtype=np.uint8))
    assert exported.dtype == jnp.uint8
