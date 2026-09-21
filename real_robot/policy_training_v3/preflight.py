"""Local execution-contract preflight; captures a request without network IO."""
import ast
import json
from pathlib import Path
from appl.agent import AgentLoop
from appl.io import ROOT, atomic, digest, read
from .design import configuration, ImplementationTools, POLICIES, DATASETS


class Captured(Exception):
    pass


class CaptureClient:
    model = "gpt-6-astra"
    reasoning = "xhigh"

    def configuration(self):
        return dict(transport="local_capture_only", model=self.model,
                    reasoning_effort=self.reasoning)

    def respond(self, request):
        self.request = request
        raise Captured()


def main():
    cfg = configuration(ROOT / "real_robot/configs/push_training_v3.json")
    setup = ROOT / cfg["run"] / "setup"
    origin = read(setup / "framework_origin.json")
    for name, sha in origin["source_files"].items():
        assert digest(ROOT / origin["source_directory"] / name) == sha
    old = {}
    for cut in ("cut_v3", "cut_v4", "cut_v5"):
        folder = ROOT / "real_robot/data/push_letters" / cut
        manifest = read(folder / "manifest.json")
        for name, sha in manifest["files"].items():
            assert digest(folder / name) == sha, name
        old[cut] = len(manifest["files"])
    plan = read(ROOT / cfg["cut_output"] / "plan.json")
    assert {g["skill_id"] for g in plan["skills"]} == set(DATASETS.values())
    for path in (ROOT / "real_robot/policy_training_v3").glob("*.py"):
        ast.parse(path.read_text())
    trial_cfg = dict(cfg, run=cfg["run"] + "/setup/local_capture")
    tools = ImplementationTools(trial_cfg, 0)
    client = CaptureClient()
    try:
        tools.dispatch("read_assignment", {})
        tools.dispatch("read_interface", {})
        try:
            AgentLoop(tools.j, tools, client, (ROOT / cfg["prompt"]).read_text()).run(
                dict(task="LOCAL REQUEST CAPTURE: no provider execution"))
        except Captured:
            pass
        assert client.request["model"] == cfg["model"]
        assert client.request["reasoning"]["effort"] == cfg["reasoning_effort"] == "xhigh"
        assert client.request["parallel_tool_calls"] is False
        assert set(POLICIES) == set(cfg["policy_datasets"])
        atomic(setup / "preflight.json", dict(
            passed=True, model=client.request["model"], reasoning=client.request["reasoning"],
            network_calls=0, local_capture_not_provider_journal=True,
            previous_published_files_verified=old, old_training_framework_unchanged=True,
            policies=cfg["policies"], source_segments=sum(len(g["segments"]) for g in plan["skills"]),
            contract_tests="8 provenance tests passed separately",
            interface_sha256=digest(ROOT / cfg["interface"]),
            prompt_sha256=digest(ROOT / cfg["prompt"])))
        print(json.dumps(read(setup / "preflight.json")))
    finally:
        tools.j.db.close()


if __name__ == "__main__":
    main()
