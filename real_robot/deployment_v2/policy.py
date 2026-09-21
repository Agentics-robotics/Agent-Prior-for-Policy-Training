"""Host adapter; the caller owns observation acquisition and robot communication."""

import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile

from real_robot.deployment.common import ROOT, verify, wire, unwire

PACKAGE = ROOT / "real_robot/policies/push_v2"
WEIGHTS = ROOT / "real_robot/checkpoints/push_v2"


class PushPolicy:
    """One persistent policy/inventory memory per instance; use as a context manager.

    Uses this Python environment, not a server-specific Pixi executable. No API
    credential, training cache or robot connection is loaded by the worker.
    Calls are sequential; use a separate instance for a separate scene/session.
    """

    def __init__(
        self,
        policy_id,
        *,
        gpu=0,
        weights=WEIGHTS,
        package=PACKAGE,
        output=None,
        timeout_s=120,
    ):
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        manifest = verify(package, weights, policy_id)
        if manifest.get("api_contract") != "push_training_v2":
            raise ValueError(
                "Use the training_v2 source and checkpoints with this adapter"
            )
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
            "real_robot.deployment_v2.worker",
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

    def observe_scene(self, observation, scene_version):
        return self._request(
            dict(
                operation="observe_scene",
                observation=observation,
                scene_version=scene_version,
            )
        )

    def preview_goal(self, template_ref, spatial_goal, workspace_polygon_uv):
        return self._request(
            dict(
                operation="preview_goal",
                template_ref=template_ref,
                spatial_goal=spatial_goal,
                workspace_polygon_uv=workspace_polygon_uv,
            )
        )

    def reexpress_goal(self, new_template, old_goal_mask):
        return self._request(
            dict(
                operation="reexpress_goal",
                new_template=new_template,
                old_goal_mask=old_goal_mask,
            )
        )

    def act(self, observation, call, *, reset=False, controller_contract=None):
        return self._request(
            dict(
                operation="act",
                observation=observation,
                call=call,
                reset=reset,
                controller_contract=controller_contract,
            )
        )

    def decode_action(self, action, controller_contract):
        """Serialize a candidate using the unchanged API-authored controller gate."""
        return self._request(
            dict(
                operation="decode",
                action=action,
                controller_contract=controller_contract,
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
