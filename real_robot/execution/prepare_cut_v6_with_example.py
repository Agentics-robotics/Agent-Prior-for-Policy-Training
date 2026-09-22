"""Local cut_v6 preparation and request-field checks, without network calls."""

from pathlib import Path
from types import SimpleNamespace
import json

from appl.agent import AgentLoop
from appl.io import atomic, digest, read
from appl.journal import Journal
from real_robot.data import ROOT, config
from real_robot.execution.cut_v6 import prepare, freeze_interface
from real_robot.cutting_v6.tools import CutTools


class RequestCaptured(Exception):
    pass


def main():
    cfg = config(ROOT / "real_robot/configs/push_cut_v6_with_example.json")
    run = ROOT / cfg["run"]
    setup = run / "setup"
    previous = {}
    for version in ("cut_v3", "cut_v4", "cut_v5"):
        oldrun = ROOT / "real_robot/runs/push_letters" / version
        oldout = ROOT / "real_robot/data/push_letters" / version
        frozen = read(oldrun / "interface_freeze.json")
        manifest = read(oldout / "manifest.json")
        for name, expected in frozen["files"].items():
            assert digest(ROOT / name) == expected, name
        for name, expected in manifest["files"].items():
            assert digest(oldout / name) == expected, name
        previous[version] = dict(frozen_files=len(frozen["files"]), published_files=len(manifest["files"]))
    prepare(cfg)
    freeze = freeze_interface(cfg)
    atomic(setup / "runner_provenance.json", dict(
        source="real_robot/execution/cut_v5.py",
        source_sha256=digest(ROOT / "real_robot/execution/cut_v5.py"),
        derived="real_robot/execution/cut_v6.py",
        derived_sha256=digest(ROOT / "real_robot/execution/cut_v6.py"),
        changes=["Independent v6 tool schema permits learned and supplied-executor groups", "Dense/sparse/executor-only supervision and explicit decision indices", "Neutral policy-count guidance and planner-supported general/task documents", "Separate execution catalog and role-preserving publication/validation"],
        developer_interface_changed=True, scientific_design_owner="Runtime API",
    ))
    atomic(run / "destination_authorization.json", dict(
        date="2026-09-21", run="cut_v6_with_example", destination="https://api.openai.com/v1/responses",
        token_count_destination="https://api.openai.com/v1/responses/input_tokens",
        data_scope="Selected original Push demonstration images, robot states and related metadata",
        purpose="User-requested next-numbered planner-supported cut/prior design",
        latest_user_instruction=cfg["authorization"],
        standing_authorization="real_robot/authorizations/openai_data_processing_20260921.json",
        standing_authorization_sha256=digest(ROOT / "real_robot/authorizations/openai_data_processing_20260921.json"),
        authorization_basis="Explicit current run request and persistent data-processing authorization; no new confirmation required",
        model=cfg["model"], reasoning_effort=cfg["reasoning_effort"],
        policy_code=False, training=False, robot_execution=False,
    ))
    cut_tools = CutTools(cfg)
    journal = Journal(setup / "request_construction_check")
    captured = {}

    class CaptureClient:
        model = cfg["model"]
        reasoning = cfg["reasoning_effort"]

        def configuration(self):
            return dict(preflight_only=True, model=self.model, reasoning_effort=self.reasoning)

        def respond(self, request):
            captured.update(request)
            raise RequestCaptured()

    prompt = (ROOT / cfg["prompt"]).read_text()
    inspection_tools = SimpleNamespace(schemas=cut_tools.schemas, budget=cut_tools.budget, done=lambda: False)
    try:
        AgentLoop(journal, inspection_tools, CaptureClient(), prompt=prompt).run(dict(purpose="Request construction only; no transport"))
    except RequestCaptured:
        pass
    assert captured["model"] == "gpt-6-astra"
    assert captured["reasoning"] == {"effort": "xhigh"}
    assert captured["instructions"] == prompt
    assert captured["max_output_tokens"] == cfg["max_output_tokens"]
    with journal.db:
        journal.db.execute("UPDATE api SET status='not_sent_preflight'")
    journal.db.close()
    cut_tools.j.db.close()
    result = dict(
        date="2026-09-21", passed=True, actual_AgentLoop_request_constructed=True,
        model=captured["model"], reasoning=captured["reasoning"], network_calls=0,
        prompt_sha256=digest(ROOT / cfg["prompt"]), interface_identity=freeze["identity"],
        old_versions_unchanged=previous, source_audit=read(run / "data_audit.json"),
    )
    atomic(setup / "preflight.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
