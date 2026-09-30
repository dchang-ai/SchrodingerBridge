"""GPU-only score regression diagnostic, not a DSB trajectory experiment.

For X ~ N(0, I_d), regress the Gaussian score -QX. The default pair target
has Q=[[1,-rho],[-rho,1]], with coordinatewise MSE infimum rho**2.
The Hadamard target Q=I+rho*H/sqrt(d) is positive definite for abs(rho)<1.
For any fixed structural score support, omitted independent coordinates
give the exact unrestricted support-class MSE infimum sum(Q_missing**2)/d.
This is a lower bound for the model subclass, not an optimization guarantee.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import hadamard

from dsb import ipf
from dsb.made import MADE


def model_metadata(model):
    metadata = model.metadata()
    if "conditional_parents" in metadata:
        return metadata
    # Dense MADE has no geometric drawing, but its masks give exact reachability.
    reachable = np.zeros((len(model.layers[0].input_ranks), model.dim), dtype=bool)
    reachable[:model.dim] = np.eye(model.dim, dtype=bool)
    incoming = np.zeros(reachable.shape[0], dtype=np.int64)
    max_degree, active_weights = 0, 0
    for layer in model.layers:
        mask = np.asarray(layer.mask(), dtype=bool)
        max_degree = max(max_degree, int(np.max(incoming + mask.sum(axis=0))))
        incoming = mask.sum(axis=1)
        active_weights += int(mask.sum())
        reachable = mask @ reachable
    max_degree = max(max_degree, int(np.max(incoming)))
    metadata.update(
        conditioner_max_degree=max_degree,
        conditioner_max_edge_length=None,
        conditioner_planar_drawing=None,
        active_weights=active_weights,
        conditional_parents=reachable.reshape(model.dim, 3 * model.components, model.dim).any(axis=1).tolist(),
    )
    return metadata


def target(x, precision):
    return -x @ precision.T


@eqx.filter_jit
def squared_errors(model, xs, precision):
    predictions = jax.vmap(lambda x: model(1, x))(xs)
    return jnp.sum(jnp.square(predictions - target(xs, precision)), axis=0)


def evaluate(model, xs, precision, batch):
    total = np.zeros(model.dim, dtype=np.float64)
    for start in range(0, xs.shape[0], batch):
        total += np.asarray(squared_errors(model, xs[start:start + batch], precision))
    errors = total / xs.shape[0]
    if not np.all(np.isfinite(errors)):
        raise FloatingPointError(f"Nonfinite heldout per-coordinate MSE: {errors}")
    return {"mse": float(errors.mean()), "per_coordinate_mse": errors.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("dense", "local"), default="dense")
    parser.add_argument("--target", choices=("pair", "hadamard"), default="pair")
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--components", type=int, default=8)
    parser.add_argument("--max-degree", type=int)
    parser.add_argument("--radius", type=float)
    parser.add_argument("--planar", action="store_true")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--eval-samples", type=int, default=20000)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    for name in ("dim", "hidden", "depth", "components", "steps", "batch", "eval_samples"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.rho) or abs(args.rho) >= 1:
        parser.error("--rho must be finite with abs(rho) < 1 for positive-definite precision")
    if args.target == "pair" and args.dim != 2:
        parser.error("--target pair requires --dim 2")
    if args.target == "hadamard" and args.dim & (args.dim - 1):
        parser.error("--target hadamard requires a power-of-two dimension")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")
    if args.seed < 0 or args.seed >= 2**32:
        parser.error("--seed must be in [0, 2**32)")
    if args.architecture == "dense" and (args.max_degree is not None or args.radius is not None or args.planar):
        parser.error("--max-degree, --radius and --planar require --architecture local")
    if jax.default_backend() != "gpu":
        raise RuntimeError("This experiment requires an allocated GPU; CPU fallback is forbidden")
    device = jax.devices()[0]
    print(f"GPU: {device}; architecture={args.architecture}; seed={args.seed}", flush=True)

    model_key, train_key, eval_key = jax.random.split(jax.random.PRNGKey(args.seed), 3)
    keys = {name: np.asarray(key).tolist() for name, key in
            (("model", model_key), ("train", train_key), ("evaluation", eval_key))}
    model_args = dict(dim=args.dim, hidden=args.hidden, depth=args.depth, components=args.components,
                      mode="score", num_steps=1, output_scale=1.0, key=model_key)
    if args.architecture == "local":
        from dsb.local_made import LocalMADE
        model = LocalMADE(**model_args, max_degree=args.max_degree, radius=args.radius, planar=args.planar)
    else:
        model = MADE(**model_args)
    metadata = model_metadata(model)
    parents = np.asarray(metadata["conditional_parents"], dtype=bool)
    if parents.shape != (args.dim, args.dim) or np.any(np.triu(parents)):
        raise ValueError("Expected strict natural-order conditional parent reachability")
    support = parents | np.eye(args.dim, dtype=bool)
    score_dependencies = support.T @ support
    if args.target == "pair":
        precision = np.array([[1.0, -args.rho], [-args.rho, 1.0]], dtype=np.float32)
    else:
        precision = np.eye(args.dim, dtype=np.float32) + args.rho * hadamard(args.dim).astype(np.float32) / math.sqrt(args.dim)
    missing_variance = np.where(score_dependencies, 0.0, precision.astype(np.float64)**2).sum(axis=1)
    support_infimum = float(missing_variance.mean())
    precision_device = jnp.asarray(precision)
    heldout = jax.random.normal(eval_key, (args.eval_samples, args.dim))
    optimizer = ipf.make_optimizer(args.lr, grad_clip=1.0, warmup=0)
    state = ipf.init_train_state(model, optimizer)
    _, static = eqx.partition(model, eqx.is_array)
    train_step = ipf.make_train_step(optimizer, static, ema_decay=0.0, use_dropout=False)
    ks = jnp.ones((args.batch,), dtype=jnp.int32)
    # A fresh directory prevents mixing seeds or overwriting another run's checkpoint.
    args.outdir.mkdir(parents=True, exist_ok=False)
    initial = evaluate(model, heldout, precision_device, args.batch)
    print(f"initial heldout MSE={initial['mse']:.8g}; structural-support infimum={support_infimum:.8g}", flush=True)
    with (args.outdir / "loss.jsonl").open("w") as log:
        for step in range(1, args.steps + 1):
            train_key, sample_key, update_key = jax.random.split(train_key, 3)
            xs = jax.random.normal(sample_key, (args.batch, args.dim))
            state, loss = train_step(state, ks, xs, target(xs, precision_device), None, update_key)
            loss_value = float(loss)
            if not math.isfinite(loss_value):
                raise FloatingPointError(f"Nonfinite training loss at step {step}: {loss_value}")
            log.write(json.dumps({"step": step, "loss": loss_value}, allow_nan=False) + "\n")
            if step == 1 or step % 100 == 0 or step == args.steps:
                log.flush()
                print(f"step={step}/{args.steps} train MSE={loss_value:.8g}", flush=True)

    model = eqx.combine(state.params, static)
    final = evaluate(model, heldout, precision_device, args.batch)
    gain = float(model.gain[1])
    if not math.isfinite(gain):
        raise FloatingPointError(f"Nonfinite learned time gain: {gain}")
    metrics = {
        "seed": args.seed,
        "args": {**vars(args), "outdir": str(args.outdir)},
        "random_keys": keys,
        "device": str(device),
        "jax_version": jax.__version__,
        "model_metadata": metadata,
        "family_scope": {
            "prediction": "learned scalar gain[1] times grad_x log p_theta(x | k=1)",
            "density": "normalized natural-order Gaussian-mixture MADE; not composed regression",
            "conditioning_graph": "all scalar affine conditioner edges, including sinusoidal time-feature inputs",
            "excluded_from_conditioner_constraints": "time embedding generation, fixed-dimensional mixture-head arithmetic, and reverse differentiation; these add overhead and propagation and do not share the triangular conditioner graph",
            "conditional_parents": parents.tolist(),
            "conditional_parents_semantics": "row=conditional, column=input; actual structural paths, not thresholded learned weights",
            "score_possible_dependencies": score_dependencies.tolist(),
            "score_dependency_semantics": "row=score coordinate, column=input; conservative support from shared log-conditional factors, not a measured nonzero Jacobian",
            "learned_gain_at_k1": gain,
        },
        "regression": {
            "input_distribution": f"N(0, I_{args.dim})",
            "target": "-Qx",
            "target_family": args.target,
            "target_density_precision": precision.tolist(),
            "loss": "unchanged ipf.mean_matching_loss: mean over samples and all coordinates",
            "k": 1,
            "optimizer": {"name": "ipf.make_optimizer/Adam", "lr": args.lr, "grad_clip": 1.0, "warmup": 0},
            "checkpoint_parameters": "last iterate, not EMA",
            "evaluation": "same fixed heldout Gaussian samples before and after; evaluation key independent of model and training keys",
        },
        "support_theorem": {
            "population_infimum": support_infimum,
            "per_coordinate_infimum": missing_variance.tolist(),
            "assumptions": "Independent standard-normal input coordinates; arbitrary square-integrable prediction i measurable only with respect to its structural score support",
            "derivation": "Conditional expectation retains Q entries on the available coordinates. The independent omitted linear terms have variance sum_j Q_ij^2 over unavailable j.",
            "coordinatewise": not bool(parents.any()),
            "max_score_support": int(score_dependencies.sum(axis=1).max()),
            "interpretation": "Exact infimum over all functions with the reported support, hence a model-class lower bound. For the pair target and coordinatewise models it equals rho^2 and is attained by the standard Gaussian score. A finite-sample MSE may be below the population bound. Surviving graph paths do not ensure the model attains the unrestricted support-class optimum.",
        },
        "initial_heldout": initial,
        "final_heldout": final,
        "checkpoint": "model.eqx",
    }
    eqx.tree_serialise_leaves(args.outdir / "model.eqx", model)
    with (args.outdir / "metrics.json").open("w") as output:
        json.dump(metrics, output, indent=2, allow_nan=False)
        output.write("\n")
    print(f"final heldout MSE={final['mse']:.8g}; per-coordinate MSE={final['per_coordinate_mse']}; "
          f"saved to {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
