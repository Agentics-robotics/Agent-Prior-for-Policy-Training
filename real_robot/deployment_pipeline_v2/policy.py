"""Host adapter; the caller owns observation acquisition and robot communication."""

import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile

from .common import ROOT, verify, wire, unwire, locations

class Policy:
    """One persistent policy/inventory memory per instance; use as a context manager.

    Uses this Python environment, not a server-specific Pixi executable. No API
    credential, training cache or robot connection is loaded by the worker.
    Calls are sequential; use a separate instance for a separate scene/session.
    """

    def __init__(
        self,
        bundle,
        policy_id=None,
        *,
        gpu=0,
        weights=None,
        package=None,
        assets=None,
        output=None,
        timeout_s=120,
    ):
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        defaults = locations(bundle)
        package = defaults[0] if package is None else Path(package)
        weights = defaults[1] if weights is None else Path(weights)
        assets = defaults[2] if assets is None else Path(assets)
        manifest = verify(package, weights, assets, policy_id)
        if policy_id is None:
            if len(manifest["policies"]) != 1:
                raise ValueError("Select a policy_id explicitly")
            policy_id = next(iter(manifest["policies"]))
        self.policy_id = policy_id
        self.manifest = manifest
        self.timeout_s = float(timeout_s)
        self.closed = False
        self.output = (
            Path(output).resolve()
            if output
            else Path(tempfile.mkdtemp(prefix=f"push-{policy_id}-"))
        )
        if output:
            self.output.mkdir(parents=True, exist_ok=False)
        self.errors = (self.output / "stderr.log").open("w")
        env = dict(
            PATH=os.defpath,
            LANG="C.UTF-8",
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONNOUSERSITE="1",
            PYTHONPATH=str(ROOT) + ":" + str(ROOT / "src"),
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="1",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
        )
        args = [
            sys.executable,
            "-u",
            "-m",
            "real_robot.deployment_pipeline_v2.worker",
            "--policy",
            policy_id,
            "--gpu",
            str(gpu),
            "--package",
            str(Path(package).resolve()),
            "--weights",
            str(Path(weights).resolve()),
            "--output",
            str(self.output),
            "--assets",
            str(Path(assets).resolve()),
        ]
        try:
            self.process = subprocess.Popen(
                args,
                cwd=ROOT,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.errors,
                text=True,
                bufsize=1,
            )
            ready = self._read()
            if ready.get("status") != "ready" or ready.get("policy_id") != policy_id:
                raise RuntimeError("Unexpected worker startup: " + str(ready))
            self.ready = ready
        except BaseException:
            self._abort()
            raise

    def _abort(self):
        self.closed = True
        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        self.errors.close()

    def _read(self):
        if not select.select([self.process.stdout], [], [], self.timeout_s)[0]:
            self._abort()
            raise TimeoutError("Policy worker timed out; no request was retried")
        line = self.process.stdout.readline()
        if not line:
            message = (self.output / "stderr.log").read_text()[-8000:]
            raise RuntimeError(f"Policy worker exited; logs: {self.output}\n{message}")
        return unwire(json.loads(line))

    def _request(self, request):
        if self.closed:
            raise RuntimeError("Policy worker is closed")
        try:
            self.process.stdin.write(json.dumps(wire(request), allow_nan=False) + "\n")
            self.process.stdin.flush()
            return self._read()
        except BaseException:
            self._abort()
            raise

    def inventory(self, observation, scene_version):
        return self._request(
            dict(
                operation="inventory",
                observation=observation,
                scene_version=scene_version,
            )
        )

    def act(self, observation, call, *, reset=False, executor_contract=None):
        return self._request(
            dict(
                operation="act",
                observation=observation,
                call=call,
                reset=reset,
                executor_contract=executor_contract,
            )
        )

    def close(self):
        if self.closed:
            return
        try:
            self.process.stdin.close()
            code = self.process.wait(timeout=self.timeout_s)
            if code:
                raise RuntimeError(
                    f"Policy exited with status {code}; see {self.output}"
                )
        finally:
            self._abort()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
