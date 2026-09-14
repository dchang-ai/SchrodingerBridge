"""Step-size schedule, reference dynamics and closed-form marginals.

Conventions
-----------
We work with a discrete chain X_0, ..., X_N.  ``gammas[k]`` (k = 0..N-1) is the
step size of the transition k -> k+1, i.e. the paper's gamma_{k+1}.

The reference dynamics (paper: f(x) = -alpha x) is the Euler–Maruyama
discretisation of an Ornstein–Uhlenbeck process, f(x) = -alpha (x - mean):

    X_{k+1} = X_k + gammas[k] * f(X_k) + sqrt(2 gammas[k]) Z.

With alpha = 1/var this targets the Gaussian prior N(mean, var I); alpha = 0 is a
Brownian motion.

Because f is linear, the reference marginals X_k | X_0 are Gaussian and can be
sampled in closed form (Appendix I, Technique 2).
"""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np


def symmetric_gammas(num_steps: int, gamma_min: float = 1e-5, gamma_max: float = 1e-1, power: float = 1.0) -> np.ndarray:
    """Time-reversal invariant schedule of Appendix J.2.

    gamma_k = gamma_min + (2k/N)^power (gamma_max - gamma_min) for k <= N/2 and
    symmetric afterwards, so that gamma_k = gamma_{N-k}.  Returns N values.

    power=1 is the linear ramp of the paper.  Its first two steps are 1e-5 and ~4e-3
    (N=50): the noise sqrt(2 gamma_1) ~ 0.09 injected at the second-to-last backward
    step is then 20x larger than the noise level the near-identity last step B(1, .)
    was trained on, and stays in the samples as speckle.  power=2 (or 3) starts the ramp
    gently so that consecutive steps differ by a small factor at both ends.
    """
    n = num_steps
    if n % 2 != 0:
        raise ValueError("num_steps must be even for the symmetric schedule")
    half = np.array([gamma_min + (2 * k / n) ** power * (gamma_max - gamma_min) for k in range(n // 2)])
    return np.concatenate([half, half[::-1]]).astype(np.float32)


def constant_gammas(num_steps: int, gamma: float) -> np.ndarray:
    return np.full((num_steps,), gamma, dtype=np.float32)


@dataclasses.dataclass(frozen=True)
class OUReference:
    """Reference forward dynamics f(x) = -alpha (x - mean).

    alpha = 1/var gives an OU process targeting N(mean, var I) (Ho et al. style);
    alpha = 0 gives a Brownian motion (Song & Ermon style).
    """

    gammas: np.ndarray  # (N,)
    alpha: float = 1.0
    mean: float = 0.0

    @property
    def num_steps(self) -> int:
        return int(self.gammas.shape[0])

    @property
    def horizon(self) -> float:
        return float(np.sum(self.gammas))

    def drift(self, x):
        return -self.alpha * (x - self.mean)

    def forward_mean(self, k, x):
        """F^0(k, x) = x + gamma_k f(x): mean of X_{k+1} given X_k = x."""
        g = jnp.asarray(self.gammas)[k]
        g = jnp.reshape(g, g.shape + (1,) * (x.ndim - jnp.ndim(g)))
        return x + g * self.drift(x)

    def marginal_coefficients(self) -> tuple[np.ndarray, np.ndarray]:
        """Closed-form X_k | X_0 = A_k X_0 + (1 - A_k) mean + S_k Z for k = 0..N.

        The chain is linear Gaussian: X_{k+1} = a_k X_k + (1 - a_k) mean + sqrt(2 g_k) Z with
        a_k = 1 - alpha g_k, so A_{k+1} = a_k A_k and S_{k+1}^2 = a_k^2 S_k^2 + 2 g_k.
        """
        n = self.num_steps
        A = np.ones(n + 1, dtype=np.float64)
        S2 = np.zeros(n + 1, dtype=np.float64)
        for k in range(n):
            a = 1.0 - self.alpha * float(self.gammas[k])
            A[k + 1] = a * A[k]
            S2[k + 1] = a * a * S2[k] + 2.0 * float(self.gammas[k])
        return A.astype(np.float32), np.sqrt(S2).astype(np.float32)

    def sample_marginal(self, key, k, x0):
        """Sample X_k given X_0 = x0 in closed form. ``k`` may be a per-sample int array."""
        A, S = self.marginal_coefficients()
        A = jnp.asarray(A)[k]
        S = jnp.asarray(S)[k]
        shape = (1,) * (x0.ndim - jnp.ndim(A))
        A = jnp.reshape(A, A.shape + shape)
        S = jnp.reshape(S, S.shape + shape)
        z = jax.random.normal(key, x0.shape, x0.dtype)
        return A * x0 + (1.0 - A) * self.mean + S * z
