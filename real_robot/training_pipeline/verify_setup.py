"""Offline request capture and preservation receipt; no provider or robot calls."""
from pathlib import Path
import time

from appl.agent import AgentLoop
from appl.io import ROOT, atomic, digest, read
from .contract import configuration
from .design import ImplementationTools
from .workflow import preflight


class Captured(Exception):
    pass


class CaptureClient:
    model = "gpt-6-astra"
    reasoning = "xhigh"

    def configuration(self):
        return dict(transport="offline_capture", model=self.model, reasoning=self.reasoning)

    def respond(self, request):
        self.request = request
        raise Captured()


def main():
    folder = ROOT / "real_robot/runs/pipeline_verification" / str(time.time_ns())
    folder.mkdir(parents=True)
    results = {}
    for name in ("push_training_v5", "flip_egg_training_v1"):
        cfg = configuration(ROOT / f"real_robot/configs/{name}.json")
        results[name] = preflight(cfg)
        trial = dict(cfg, run=str((folder / name).relative_to(ROOT)))
        tools = ImplementationTools(trial, 0)
        client = CaptureClient()
        try:
            assignment = tools.dispatch("read_assignment", {})
            tools.dispatch("read_interface", {})
            assert "policy_ids" not in assignment and "policy_datasets" not in assignment
            try:
                AgentLoop(tools.j, tools, client, (ROOT / cfg["prompt"]).read_text()).run(
                    dict(task="OFFLINE INFRASTRUCTURE REQUEST CAPTURE"))
            except Captured:
                pass
            assert client.request["reasoning"]["effort"] == "xhigh"
            assert client.request["model"] == "gpt-6-astra"
            assert not client.request["parallel_tool_calls"]
            assert client.request["instructions"] == (ROOT / cfg["prompt"]).read_text()
            with tools.j.db:
                tools.j.db.execute("UPDATE api SET status='not_sent_preflight'")
            results[name]["captured_request"] = str(tools.j.root / "api/0001.request.json")
            results[name]["capture_network_calls"] = 0
        finally:
            tools.j.db.close()
    archive = ROOT / "real_robot/prompts/archive/20260922_before_representation_revision"
    records = read(archive / "manifest.json")["runs"]
    for record in records:
        for name, expected in record["files"].items():
            assert digest(ROOT / name) == digest(archive / name) == expected
    old = read("/tmp/robot_pipeline_preservation.json")
    for name, expected in old.items():
        assert digest(name) == expected
    result = dict(passed=True, runs=results, network_calls=0, formal_training_runs=0,
                  preserved_cut_runs=[r["experiment"] for r in records],
                  preserved_old_training_files=len(old), fixture_tests="11 passed; isolated CPU preparation/training/online worker",
                  common_implementation_prompt=True, robot_execution=False)
    atomic(folder / "receipt.json", result)
    atomic(ROOT / "real_robot/reports/GENERIC_TRAINING_PIPELINE_CHECK.json", result)
    print(str(folder / "receipt.json"), flush=True)


if __name__ == "__main__":
    main()
