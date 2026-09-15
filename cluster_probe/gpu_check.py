"""Tiny GPU sanity check: prints devices, versions, and does a matmul on the GPU."""
import os, time
t0 = time.time()
import jax, jax.numpy as jnp
import equinox as eqx, optax, numpy as np
print("host:", os.uname().nodename, "| jax", jax.__version__, "| equinox", eqx.__version__, "| optax", optax.__version__, "| numpy", np.__version__)
print("backend:", jax.default_backend(), "| devices:", jax.devices())
assert jax.default_backend() == "gpu", "GPU backend not active!"
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (4096, 4096), dtype=jnp.float32)
y = jnp.dot(x, x).block_until_ready()
t1 = time.time(); y = jnp.dot(x, x).block_until_ready(); dt = time.time() - t1
print(f"matmul 4096x4096 ok on {y.devices()} | {2*4096**3/dt/1e12:.1f} TFLOP/s | fro-norm {float(jnp.linalg.norm(y)):.3e}")
# tiny equinox model + optax step to make sure the full stack works on GPU
model = eqx.nn.MLP(in_size=3072, out_size=10, width_size=256, depth=2, key=key)
opt = optax.adam(1e-3); state = opt.init(eqx.filter(model, eqx.is_array))
@eqx.filter_jit
def step(model, state, xb, yb):
    def loss(m): return optax.softmax_cross_entropy_with_integer_labels(jax.vmap(m)(xb), yb).mean()
    l, g = eqx.filter_value_and_grad(loss)(model)
    upd, state = opt.update(g, state, eqx.filter(model, eqx.is_array)); return eqx.apply_updates(model, upd), state, l
xb = jax.random.normal(key, (128, 3072)); yb = jnp.zeros(128, dtype=jnp.int32)
for _ in range(3): model, state, l = step(model, state, xb, yb)
print(f"equinox+optax train step ok, loss={float(l):.4f} | total {time.time()-t0:.1f}s")
