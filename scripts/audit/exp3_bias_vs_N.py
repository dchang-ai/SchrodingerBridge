"""Fixed point of the exact-regression DSB (same as exp2) for N=20 vs N=80 at fixed T=1/2:
if the residual bias is the paper's O(gamma) approximation, it shrinks ~4x."""
import sys; sys.path.insert(0, "/mnt/data0/shared/dominic/SchrodingerBridge")
import jax, jax.numpy as jnp, numpy as np, equinox as eqx
from dsb import ipf
from dsb.schedule import OUReference, constant_gammas
a, M = 0.1, 150_000
BETA = (np.sqrt(5) - 1) / 2
class Affine(eqx.Module):
    w: jax.Array; b: jax.Array
    def __call__(self, k, x): return self.w[k] * x + self.b[k]
def fit(ks, xs, tg, size):
    ks, xs, tg = np.asarray(ks), np.asarray(xs)[..., 0], np.asarray(tg)[..., 0]
    w, b = np.zeros(size, np.float32), np.zeros(size, np.float32)
    for k in np.unique(ks):
        sel = ks == k; A = np.stack([xs[sel], np.ones(sel.sum())], 1)
        (w[k], b[k]), *_ = np.linalg.lstsq(A, tg[sel], rcond=None)
    return Affine(jnp.asarray(w), jnp.asarray(b))
for N in (20, 80):
    gammas = jnp.asarray(constant_gammas(N, 0.5 / N)); key = jax.random.PRNGKey(0)
    f_net = Affine(jnp.zeros(N), jnp.zeros(N)); res = []
    for n in range(10):
        key, k1, k2, k3, k4, k5, k6 = jax.random.split(key, 7)
        c = ipf.simulate(f_net, a + jax.random.normal(k1, (M, 1)), gammas, k2, forward=True)
        b_net = fit(c.ks[:, None].repeat(M, 1).reshape(-1), c.xs.reshape(-1, 1), c.targets.reshape(-1, 1), N + 1)
        c = ipf.simulate(b_net, -a + jax.random.normal(k3, (M, 1)), gammas, k4, forward=False)
        f_net = fit(c.ks[:, None].repeat(M, 1).reshape(-1), c.xs.reshape(-1, 1), c.targets.reshape(-1, 1), N)
        xN = -a + jax.random.normal(k5, (M, 1))
        x0 = np.asarray(ipf.simulate(b_net, xN, gammas, k6, forward=False, last_step_noise=True).xs[-1])[:, 0]
        if n >= 6: res.append((x0.mean(), x0.var(), np.cov(x0, np.asarray(xN)[:, 0])[0, 1]))
    r = np.mean(res, 0)
    print(f"N={N:3d} gamma={0.5/N:.5f}: fixed point mean {r[0]:+.4f} (bias {r[0]-a:+.4f}), var {r[1]:.4f} (bias {r[1]-1:+.4f}), cov {r[2]:.4f} (bias {r[2]-BETA:+.4f})")
