"""End-to-end check of the repo's simulate()/targets/index conventions with EXACT per-step affine
regressors (closed-form least squares) instead of neural nets, d = 1.
Gaussian case of App. G.2: p_data = N(a,1) at k=0, p_prior = N(-a,1) at k=N, Brownian reference,
T = 1/2 (N=20, gamma=1/40).  The SB has Cov(X_0, X_N) = beta = (sqrt5-1)/2 = 0.618.
A wrong sign / wrong index in the targets would give a different fixed point (we also run a
deliberately sign-flipped variant to show the check has power)."""
import sys; sys.path.insert(0, "/mnt/data0/shared/dominic/SchrodingerBridge")
import jax, jax.numpy as jnp, numpy as np, equinox as eqx
from dsb import ipf
from dsb.schedule import OUReference, constant_gammas

a, N, gamma, M = 0.1, 20, 1.0 / 40.0, 200_000
BETA = (np.sqrt(5) - 1) / 2
ref = OUReference(constant_gammas(N, gamma), alpha=0.0, mean=-a)
gammas = jnp.asarray(ref.gammas)

class Affine(eqx.Module):
    """increment(k, x) = w[k] * x + b[k]  (per-step affine mean map, exact for Gaussians)."""
    w: jax.Array; b: jax.Array
    def __call__(self, k, x):
        return self.w[k] * x + self.b[k]

def fit(ks, xs, tg, size, flip=False):
    """exact least squares of tg on (1, x) for each step index -> Affine with size entries."""
    ks, xs, tg = np.asarray(ks), np.asarray(xs)[..., 0], np.asarray(tg)[..., 0]
    if flip: tg = -tg
    w, b = np.zeros(size, np.float32), np.zeros(size, np.float32)
    for k in np.unique(ks):
        sel = ks == k
        A = np.stack([xs[sel], np.ones(sel.sum())], 1)
        sol, *_ = np.linalg.lstsq(A, tg[sel], rcond=None)
        w[k], b[k] = sol
    return Affine(jnp.asarray(w), jnp.asarray(b))

def run(flip=False, iters=8, seed=0):
    key = jax.random.PRNGKey(seed)
    f_net = Affine(jnp.zeros(N), jnp.zeros(N))  # reference: Brownian -> F^0(k,x) = x
    for n in range(iters):
        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        # backward half-step: forward trajectories from data with F^n, fit B^n on (k+1, X_{k+1}) -> target
        x0 = a + jax.random.normal(k1, (M, 1))
        c = ipf.simulate(f_net, x0, gammas, k2, forward=True)
        ks, xs, tg = c.ks[:, None].repeat(M, 1).reshape(-1), c.xs.reshape(-1, 1), c.targets.reshape(-1, 1)
        b_net = fit(ks, xs, tg, N + 1, flip)
        # forward half-step: backward trajectories from prior with B^n, fit F^{n+1}
        xN = -a + jax.random.normal(k3, (M, 1))
        c = ipf.simulate(b_net, xN, gammas, k4, forward=False)
        ks, xs, tg = c.ks[:, None].repeat(M, 1).reshape(-1), c.xs.reshape(-1, 1), c.targets.reshape(-1, 1)
        f_net = fit(ks, xs, tg, N, flip)
        # evaluate generative model B^n (with last-step noise, as in train_gaussian.py)
        key, k5, k6 = jax.random.split(key, 3)
        xN = -a + jax.random.normal(k5, (M, 1))
        g = ipf.simulate(b_net, xN, gammas, k6, forward=False, last_step_noise=True)
        x0g = np.asarray(g.xs[-1])[:, 0]; xNn = np.asarray(xN)[:, 0]
        # forward endpoint of F^{n+1} from data
        key, k7, k8 = jax.random.split(key, 3)
        xf = np.asarray(ipf.simulate(f_net, a + jax.random.normal(k7, (M, 1)), gammas, k8, forward=True).xs[-1])[:, 0]
        print(f"[flip={flip}] iter {n}: gen X_0 mean {x0g.mean():+.4f} var {x0g.var():.4f} cov(X_0,X_N) {np.cov(x0g, xNn)[0,1]:.4f} "
              f"(target {a:+.1f}, 1, {BETA:.4f}) | fwd X_N mean {xf.mean():+.4f} var {xf.var():.4f} (target {-a:+.1f}, 1)")

run(flip=False)
print()
run(flip=True, iters=3)
