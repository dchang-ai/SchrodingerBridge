"""Run with .venv-fid/bin/python tests/test_fid_resume.py on a CPU compute node."""
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

import numpy as np


def check_cached_resume():
    script = Path(__file__).resolve().parents[1] / "scripts" / "fid.py"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ref = root / "reference.npz"
        np.savez(ref, mu=np.zeros(2), sigma=np.eye(2))
        command = [sys.executable, str(script), str(root), "--ref", str(ref), "--threads", "1"]
        journal = root / "fid.jsonl"

        def cached(n, shift):
            # Unreadable images prove the evaluator uses the surviving statistics.
            contents = f"images {n} {shift} must not be read".encode()
            (root / f"gen_iter{n:03d}.npz").write_bytes(contents)
            np.savez(root / f"gen_iter{n:03d}_fidstats.npz",
                     mu=np.array(shift), sigma=np.eye(2), n=5000,
                     sample_sha256=hashlib.sha256(contents).hexdigest())

        def rows():
            return [json.loads(line) for line in journal.read_text().splitlines()]

        cached(0, [3.0, 4.0])
        subprocess.run(command, check=True, timeout=60)
        actual = rows()
        assert len(actual) == 1 and actual[0]["n"] == 0
        assert actual[0]["num_samples"] == 5000
        np.testing.assert_allclose(actual[0]["fid"], 25.0, atol=1e-10)
        previous = journal.read_bytes()
        subprocess.run(command, check=True, timeout=60)
        assert journal.read_bytes() == previous

        # Replay a crash after stats publication, before journal publication.
        journal.unlink()
        cached(1, [6.0, 8.0])
        cached(2, [9.0, 12.0])
        metrics = root / "metrics.jsonl"
        metrics.write_text('{"n": 0}\n{"n": 1')
        resume = root / "ckpt" / "resume.json"
        resume.parent.mkdir()

        def commit(limit):
            temporary = resume.with_suffix(".tmp")
            temporary.write_text(json.dumps({"logs": {"metrics.jsonl": limit}}))
            temporary.replace(resume)

        commit(len(b'{"n": 0}\n'))
        watch = command + ["--watch", "--expected-iters", "2", "--poll-seconds", "0.05"]
        process = subprocess.Popen(watch)
        try:
            deadline = time.monotonic() + 60
            while not journal.exists():
                assert process.poll() is None, "watch exited before evaluating committed iteration 0"
                assert time.monotonic() < deadline, "watch did not publish iteration 0"
                time.sleep(0.05)
            time.sleep(0.2)
            assert process.poll() is None, "watch did not wait for iteration 1"
            assert rows() == actual, "watch consumed an uncommitted sample or duplicated a row"
            # Complete JSON without a newline is still not a committed metrics row.
            with metrics.open("a") as fh:
                fh.write("}")
            time.sleep(0.2)
            assert process.poll() is None and rows() == actual
            with metrics.open("a") as fh:
                fh.write("\n")
            time.sleep(0.2)
            assert process.poll() is None and rows() == actual
            commit(metrics.stat().st_size)
            assert process.wait(timeout=60) == 0
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        actual = rows()
        assert [row["n"] for row in actual] == [0, 1]
        assert [row["num_samples"] for row in actual] == [5000, 5000]
        np.testing.assert_allclose([row["fid"] for row in actual], [25.0, 100.0], atol=1e-10)
        previous = journal.read_bytes()
        subprocess.run(watch, check=True, timeout=60)
        assert journal.read_bytes() == previous

        # A surviving complete journal cannot certify rolled-back metrics.
        commit(0)
        namespace = runpy.run_path(str(script))
        class Waiting(Exception):
            pass
        with patch.object(sys, "argv", watch[1:]), patch.object(time, "sleep", side_effect=Waiting):
            try:
                namespace["main"]()
            except Waiting:
                pass
            else:
                raise AssertionError("watch accepted an uncommitted FID journal")
        commit(metrics.stat().st_size)

        # Regenerated samples invalidate the journal even if their new stats survived.
        cached(1, [9.0, 12.0])
        subprocess.run(watch, check=True, timeout=60)
        np.testing.assert_allclose([row["fid"] for row in rows()], [25.0, 225.0], atol=1e-10)
        np.savez(ref, mu=np.array([1.0, 2.0]), sigma=np.eye(2))
        subprocess.run(watch, check=True, timeout=60)
        np.testing.assert_allclose([row["fid"] for row in rows()], [8.0, 164.0], atol=1e-10)

        # A stale statistics cache must not conceal unreadable replacement samples.
        (root / "gen_iter001.npz").write_bytes(b"replacement images without valid statistics")
        rejected = subprocess.run(watch, capture_output=True, timeout=60)
        assert rejected.returncode != 0
    print("cached-statistics resume and incremental watch checks passed")


if __name__ == "__main__":
    check_cached_resume()
