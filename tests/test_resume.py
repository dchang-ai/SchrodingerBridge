"""Deterministic recovery checks; run on an allocated GPU, not the login node."""
import argparse
import dataclasses
import json
import weakref

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dsb import train
from dsb.utils import Logger, save_model_config


class TinyNet(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, key):
        self.linear = eqx.nn.Linear(2, 2, key=key)

    def __call__(self, k, x):
        return 0.02 * (k + 1) * self.linear(x)


class Interrupted(RuntimeError):
    pass


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _run(path, parallel, *, interrupt_eval=False, cfg=None, ndevices=1, checkpoint_every=None, eval_value=None):
    cfg = cfg or train.DSBConfig(
        num_steps=2, num_dsb_iters=2, train_steps=5, batch_size=4,
        cache_size=4, cache_refresh=4, checkpoint_every=3, log_every=4,
        gamma_schedule="constant", gamma=0.01, lr=0.003, warmup=3,
        reset_optimizer=False,
    )
    if checkpoint_every is not None:
        cfg = dataclasses.replace(cfg, checkpoint_every=checkpoint_every)
    logger = Logger(str(path / "log.jsonl"))

    def evaluate(n, backward, forward, key):
        xs = jax.random.normal(key, (4, 2))
        value = jnp.mean(jax.vmap(lambda x: backward(1, x) + forward(0, x))(xs))
        metrics = {"value": float(value if eval_value is None else eval_value)}
        with (path / "metrics.jsonl").open("a") as fh:
            fh.write(json.dumps(dict(n=n, **metrics)) + "\n")
        if interrupt_eval and n == 0:
            raise Interrupted("evaluation appended its row before interruption")
        return metrics

    driver = train.run_dsb_parallel if parallel else train.run_dsb
    kwargs = {"devices": jax.devices()[:ndevices]} if parallel else {}
    return driver(
        cfg, make_forward_net=TinyNet, make_backward_net=TinyNet,
        sample_data=lambda key, count, data: data + jax.random.normal(key, (count, 2)),
        sample_prior=lambda key, count: jax.random.normal(key, (count, 2)),
        data=jnp.asarray([0.25, -0.5]), outdir=str(path), eval_fn=evaluate,
        logger=logger, **kwargs,
    )


@pytest.mark.parametrize("parallel,interruption,ndevices", [
    (False, "training", 1), (True, "training", 1),
    (False, "evaluation", 1), (True, "evaluation", 1),
    (False, "publication", 1), (True, "training", 2),
])
def test_exact_resume(tmp_path, monkeypatch, parallel, interruption, ndevices):
    if len(jax.devices()) < ndevices:
        pytest.skip("requires two allocated devices")
    expected_dir, resumed_dir = tmp_path / "expected", tmp_path / "resumed"
    expected = _run(expected_dir, parallel, ndevices=ndevices)
    original_save, original_json = train._Checkpoints.save, train._atomic_json

    def save_then_interrupt(self, f_state, b_state, key, cache_key, cursor):
        original_save(self, f_state, b_state, key, cache_key, cursor)
        if cursor["n"] == 1 and cursor["phase"] == "backward" and cursor["it"] == 3:
            raise Interrupted("checkpoint inside a live cache and loss window")

    def interrupt_publication(path, value):
        if str(path).endswith("resume.json"):
            cursor = value["cursor"]
            if cursor["n"] == 1 and cursor["phase"] == "backward" and cursor["it"] == 3:
                raise Interrupted("payload written but pointer not committed")
        original_json(path, value)

    with monkeypatch.context() as patch:
        if interruption == "training":
            patch.setattr(train._Checkpoints, "save", save_then_interrupt)
        elif interruption == "publication":
            patch.setattr(train, "_atomic_json", interrupt_publication)
        with pytest.raises(Interrupted):
            _run(resumed_dir, parallel, interrupt_eval=interruption == "evaluation", ndevices=ndevices)

    actual = _run(resumed_dir, parallel, ndevices=ndevices, checkpoint_every=2)
    for wanted, restored in zip(jax.tree_util.tree_leaves(expected), jax.tree_util.tree_leaves(actual)):
        np.testing.assert_array_equal(wanted, restored)
    assert _rows(expected_dir / "metrics.jsonl") == _rows(resumed_dir / "metrics.jsonl")

    def losses(path):
        return [{k: row[k] for k in ("n", "phase", "step", "loss")}
                for row in _rows(path / "log.jsonl") if "loss" in row]

    assert losses(expected_dir) == losses(resumed_dir)
    # A completed command is idempotent, and inference consumers still load Nets.
    _run(resumed_dir, parallel, ndevices=ndevices)
    assert _rows(expected_dir / "metrics.jsonl") == _rows(resumed_dir / "metrics.jsonl")
    latest = json.loads((resumed_dir / "ckpt" / "latest.json").read_text())
    nets = train.Nets(actual[0], actual[0], actual[1], actual[1])
    loaded = eqx.tree_deserialise_leaves(resumed_dir / "ckpt" / latest["file"], nets)
    for wanted, restored in zip(jax.tree_util.tree_leaves(actual),
                               jax.tree_util.tree_leaves((loaded.f_ema, loaded.b_ema))):
        np.testing.assert_array_equal(wanted, restored)


@pytest.mark.parametrize("parallel", [False, True])
def test_cache_released_before_refresh_and_evaluation(tmp_path, monkeypatch, parallel):
    original_loop = train._run_loop
    live_arrays = []

    def bounded_cache(builder):
        def build(*args):
            assert all(ref() is None for ref in live_arrays), "previous trajectory cache still live"
            cache = builder(*args)
            live_arrays[:] = [weakref.ref(cache.xs), weakref.ref(cache.targets)]
            return cache
        return build

    def run_loop(*args, **kwargs):
        evaluate = kwargs["eval_fn"]

        def without_cache(*eval_args):
            assert all(ref() is None for ref in live_arrays), "trajectory cache retained during evaluation"
            return evaluate(*eval_args)

        kwargs["forward_cache"] = bounded_cache(kwargs["forward_cache"])
        kwargs["backward_cache"] = bounded_cache(kwargs["backward_cache"])
        kwargs["eval_fn"] = without_cache
        return original_loop(*args, **kwargs)

    monkeypatch.setattr(train, "_run_loop", run_loop)
    _run(tmp_path, parallel)


@pytest.mark.parametrize("published", [False, True])
def test_resume_refuses_changed_config_without_overwriting_provenance(tmp_path, published):
    path = tmp_path / "run"
    cfg = train.DSBConfig(num_steps=2, num_dsb_iters=0, checkpoint_every=3)
    _run(path, False, cfg=cfg)
    if not published:
        (path / "ckpt" / "resume.json").unlink()
    original = (path / "config.json").read_bytes()
    with pytest.raises(ValueError):
        _run(path, False, cfg=dataclasses.replace(cfg, lr=cfg.lr * 2))
    assert (path / "config.json").read_bytes() == original


def test_legacy_inference_checkpoint_remains_loadable(tmp_path):
    path = tmp_path / "legacy"
    cfg = train.DSBConfig(num_steps=2, num_dsb_iters=1, train_steps=1,
                          batch_size=4, cache_size=4)
    expected = _run(path, False, cfg=cfg)
    (path / "ckpt" / "resume.json").unlink()
    actual = _run(path, False, cfg=cfg)
    for wanted, restored in zip(jax.tree_util.tree_leaves(expected), jax.tree_util.tree_leaves(actual)):
        np.testing.assert_array_equal(wanted, restored)
    resumes = [row for row in _rows(path / "log.jsonl") if row.get("event") == "resume"]
    assert resumes[-1]["exact"] is False
    assert [row["n"] for row in _rows(path / "metrics.jsonl")] == [0]


def test_legacy_launch_metadata_preserves_configuration(tmp_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", default="unet")
    parser.add_argument("--ch", type=int, default=128)
    parser.add_argument("--iters", type=int, default=10)
    previous = {"ch": 128, "argv": ["--iters", "3"]}
    path = tmp_path / "model.json"
    path.write_text(json.dumps(previous))
    original = path.read_bytes()
    config = vars(parser.parse_args(previous["argv"]))
    save_model_config(str(tmp_path), config, parser)
    assert path.read_bytes() == original
    with pytest.raises(ValueError):
        save_model_config(str(tmp_path), {**config, "architecture": "made"}, parser)
    assert path.read_bytes() == original


@pytest.mark.parametrize("log_every,checkpoint_every,bad_step,saved_step", [
    (5, 2, 3, 2),  # Checkpoint sees a bad earlier loss even though the latest loss is finite.
    (2, 5, 3, 0),  # Logging rejects before the next checkpoint.
    (5, 99, 6, 0),  # The final update cannot publish a failed model.
])
def test_nonfinite_loss_preserves_last_resume(tmp_path, monkeypatch, log_every, checkpoint_every, bad_step, saved_step):
    cfg = train.DSBConfig(num_steps=2, num_dsb_iters=1, train_steps=6, batch_size=4,
                          cache_size=4, log_every=log_every, checkpoint_every=checkpoint_every)
    original_loop = train._run_loop
    saved = {}
    original_save = train._Checkpoints.save

    def remember_save(self, *args):
        original_save(self, *args)
        saved["pointer"] = (tmp_path / "ckpt" / "resume.json").read_bytes()
        meta = json.loads(saved["pointer"])
        saved["payload"] = (tmp_path / "ckpt" / meta["file"]).read_bytes()

    def inject_loss(*args, **kwargs):
        reference_step = kwargs["reference_step"]
        step = 0

        def step_with_bad_loss(*step_args):
            nonlocal step
            state, loss = reference_step(*step_args)
            step += 1
            return state, jnp.asarray(np.nan) if step == bad_step else loss

        kwargs["reference_step"] = step_with_bad_loss
        return original_loop(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(train._Checkpoints, "save", remember_save)
        patch.setattr(train, "_run_loop", inject_loss)
        with pytest.raises(FloatingPointError, match=f"n=0 phase=backward step={bad_step}"):
            _run(tmp_path, False, cfg=cfg)

    meta = json.loads((tmp_path / "ckpt" / "resume.json").read_text())
    assert meta["cursor"]["it"] == saved_step
    assert (tmp_path / "ckpt" / "resume.json").read_bytes() == saved["pointer"]
    assert (tmp_path / "ckpt" / meta["file"]).read_bytes() == saved["payload"]
    assert not (tmp_path / "ckpt" / "latest.json").exists()
    assert not list((tmp_path / "ckpt").glob("iter*.eqx"))
    _run(tmp_path, False, cfg=cfg)
    assert json.loads((tmp_path / "ckpt" / "latest.json").read_text())["phase"] == "forward"


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_evaluation_does_not_advance_or_publish(tmp_path, value):
    cfg = train.DSBConfig(num_steps=2, num_dsb_iters=1, train_steps=1, batch_size=4, cache_size=4)
    with pytest.raises(FloatingPointError, match="n=0 phase=eval metric=value"):
        _run(tmp_path, False, cfg=cfg, eval_value=value)
    pointer = (tmp_path / "ckpt" / "resume.json").read_bytes()
    meta = json.loads(pointer)
    payload = (tmp_path / "ckpt" / meta["file"]).read_bytes()
    assert meta["cursor"]["phase"] == "eval"
    assert not (tmp_path / "ckpt" / "latest.json").exists()
    assert not list((tmp_path / "ckpt").glob("iter*.eqx"))
    with pytest.raises(FloatingPointError, match="n=0 phase=eval metric=value"):
        _run(tmp_path, False, cfg=cfg, eval_value=value)
    assert (tmp_path / "ckpt" / "resume.json").read_bytes() == pointer
    assert (tmp_path / "ckpt" / meta["file"]).read_bytes() == payload
    _run(tmp_path, False, cfg=cfg)
    assert len(_rows(tmp_path / "metrics.jsonl")) == 1
    assert np.isfinite(_rows(tmp_path / "metrics.jsonl")[0]["value"])


def test_nonfinite_pending_loss_rejected_at_resume_boundary(tmp_path, monkeypatch):
    original_save = train._Checkpoints.save

    def interrupt_after_checkpoint(self, *args):
        original_save(self, *args)
        if args[-1]["it"] == 3:
            raise Interrupted("saved pending loss window")

    with monkeypatch.context() as patch:
        patch.setattr(train._Checkpoints, "save", interrupt_after_checkpoint)
        with pytest.raises(Interrupted):
            _run(tmp_path, False)
    path = tmp_path / "ckpt" / "resume.json"
    meta = json.loads(path.read_text())
    meta["cursor"].update(it=5, losses=[np.inf])
    path.write_text(json.dumps(meta))
    original = path.read_bytes()
    with pytest.raises(FloatingPointError, match="n=0 phase=backward step=5"):
        _run(tmp_path, False)
    assert path.read_bytes() == original
    assert not list((tmp_path / "ckpt").glob("iter*.eqx"))


@pytest.mark.parametrize("phase,field,bad_step", [
    ("backward", "params", 2),
    ("backward", "ema_params", 3),
    ("forward", "params", 3),
    ("forward", "opt_state", 3),
])
def test_finite_loss_cannot_publish_nonfinite_state(tmp_path, monkeypatch, phase, field, bad_step):
    cfg = train.DSBConfig(num_steps=2, num_dsb_iters=1, train_steps=3, batch_size=4,
                          cache_size=4, checkpoint_every=2, log_every=3)
    original_loop = train._run_loop

    def inject_bad_update(*args, **kwargs):
        name = "reference_step" if phase == "backward" else "forward_step"
        original_step = kwargs[name]
        step = 0

        def bad_update(*step_args):
            nonlocal step
            state, loss = original_step(*step_args)
            step += 1
            if step == bad_step:
                assert bool(jnp.isfinite(loss))
                corrupted = jax.tree_util.tree_map(
                    lambda leaf: jnp.full_like(leaf, np.nan) if eqx.is_inexact_array(leaf) else leaf,
                    getattr(state, field),
                )
                state = state._replace(**{field: corrupted})
            return state, loss

        kwargs[name] = bad_update
        return original_loop(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(train, "_run_loop", inject_bad_update)
        with pytest.raises(FloatingPointError, match=f"Nonfinite state: n=0 phase={phase} step={bad_step}"):
            _run(tmp_path, False, cfg=cfg)
    meta = json.loads((tmp_path / "ckpt" / "resume.json").read_text())
    assert meta["cursor"]["phase"] == phase
    assert meta["cursor"]["it"] == (0 if bad_step == 2 else 2)
    assert not (tmp_path / "ckpt" / f"iter000_{phase}.eqx").exists()
    _run(tmp_path, False, cfg=cfg)
    assert json.loads((tmp_path / "ckpt" / "latest.json").read_text())["phase"] == "forward"
