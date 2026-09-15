"""Check Technique-2 pairs (X_k, X_{k+1}) from ipf.reference_pairs against the simulated EM chain."""
import sys; sys.path.insert(0, "/mnt/data0/shared/dominic/SchrodingerBridge")
import jax, jax.numpy as jnp, numpy as np, equinox as eqx
from dsb import ipf
from dsb.schedule import OUReference, symmetric_gammas

class RefNet(eqx.Module):
    ref: OUReference = eqx.field(static=True)
    def __call__(self, k, x):
        return jnp.asarray(self.ref.gammas)[k] * self.ref.drift(x)

ref = OUReference(symmetric_gammas(20, 1e-5, 1e-1), alpha=1.0, mean=0.3)
n = ref.num_steps
M = 400_000
x0 = 2.0 + 0.5 * jax.random.normal(jax.random.PRNGKey(0), (M, 1))
# closed form pairs
ks, xk1, tgt = ipf.reference_pairs(ref, jax.random.PRNGKey(1), x0)
ks = np.asarray(ks); xk1 = np.asarray(xk1)[:, 0]; tgt = np.asarray(tgt)[:, 0]
counts = np.bincount(ks, minlength=n + 1)
print("k+1 histogram (should be ~uniform on 1..N, zero at 0):", counts)
# simulated chain
c = ipf.simulate(RefNet(ref), x0, jnp.asarray(ref.gammas), jax.random.PRNGKey(2), forward=True)
xs = np.asarray(c.xs)[:, :, 0]  # xs[i] = X_{i+1}
tg = np.asarray(c.targets)[:, :, 0]
x_all = np.concatenate([np.asarray(x0)[None, :, 0], xs])  # x_all[k] = X_k
print(" k+1 |  mean X_{k+1}: cf / sim |  std X_{k+1}: cf / sim |  E[target]: cf / sim | std target: cf / sim")
for kk in (1, 2, 5, 10, 15, 20):
    sel = ks == kk
    print(f" {kk:3d} | {xk1[sel].mean():+.4f} / {xs[kk-1].mean():+.4f} | {xk1[sel].std():.4f} / {xs[kk-1].std():.4f} | "
          f"{tgt[sel].mean():+.5f} / {tg[kk-1].mean():+.5f} | {tgt[sel].std():.5f} / {tg[kk-1].std():.5f}")
# joint check: recover X_k from the target: target = F(X_k)-F(X_{k+1}) = (1-a g)(X_k - X_{k+1}) -> X_k = X_{k+1} + target/(1-a g)
g = np.asarray(ref.gammas)
for kk in (1, 10, 20):
    sel = ks == kk
    xk_cf = xk1[sel] + tgt[sel] / (1 - ref.alpha * g[kk - 1])
    xk_sim = x_all[kk - 1]
    print(f"k={kk-1}: X_k mean/std  cf {xk_cf.mean():+.4f}/{xk_cf.std():.4f}   sim {xk_sim.mean():+.4f}/{xk_sim.std():.4f}; "
          f"cov(X_k,X_k+1) cf {np.cov(xk_cf, xk1[sel])[0,1]:.4f} sim {np.cov(xk_sim, xs[kk-1])[0,1]:.4f}")
