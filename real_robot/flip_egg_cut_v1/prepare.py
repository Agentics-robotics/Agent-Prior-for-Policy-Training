"""Local source audit and real AgentLoop request check; no network transport."""

import json
from types import SimpleNamespace

from appl.agent import AgentLoop
from appl.io import atomic, digest, read
from appl.journal import Journal
from .data import ROOT, config
from .runner import prepare, freeze_interface
from .tools import CutTools


class RequestCaptured(Exception):
    pass


def main():
    cfg = config(ROOT / "real_robot/configs/flip_egg_cut_v1.json")
    run = ROOT / cfg["run"]
    prepare(cfg)
    freeze = freeze_interface(cfg)
    authorization = ROOT / "real_robot/authorizations/openai_data_processing_20260921.json"
    atomic(run / "destination_authorization.json", dict(
        date="2026-09-21",
        destination="https://api.openai.com/v1/responses",
        token_count_destination="https://api.openai.com/v1/responses/input_tokens",
        data_scope="Selected original Flip egg demonstration images, robot states and metadata from the 20 configured episodes",
        purpose="Explicitly user-requested independent Flip egg cut/prior and model input/output design",
        current_authorization=cfg["authorization"],
        standing_authorization=str(authorization.relative_to(ROOT)),
        standing_authorization_sha256=digest(authorization),
        authorization_basis="Current user request plus standing authorization for future real_robot API cut/prior data processing",
        model=cfg["model"], reasoning_effort=cfg["reasoning_effort"],
        policy_code=False, training=False, robot_execution=False,
    ))
    tools = CutTools(cfg)
    journal = Journal(run / "setup/request_construction_check")
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
    contract = (ROOT / cfg["data_contract"]).read_text()
    assert "Push recordings contain -1" not in contract
    assert "gripper_position=-1" not in json.dumps(read(run / "source_manifest.json")["contract"])
    assert len(tools.sources.episodes) == 20
    assert sum(ep["n"] for ep in tools.sources.episodes.values()) == 45730
    inspection = SimpleNamespace(schemas=tools.schemas, budget=tools.budget, done=lambda: False)
    try:
        AgentLoop(journal, inspection, CaptureClient(), prompt=prompt).run(
            dict(purpose="Local request construction only; no transport"))
    except RequestCaptured:
        pass
    assert captured["model"] == "gpt-6-astra"
    assert captured["reasoning"] == {"effort": "xhigh"}
    assert captured["instructions"] == prompt
    assert captured["max_output_tokens"] == cfg["max_output_tokens"]
    with journal.db:
        journal.db.execute("UPDATE api SET status='not_sent_preflight'")
    journal.db.close()
    tools.j.db.close()
    result = dict(
        date="2026-09-21", passed=True, actual_AgentLoop_request_constructed=True,
        model=captured["model"], reasoning=captured["reasoning"], network_calls=0,
        prompt_sha256=digest(ROOT / cfg["prompt"]), interface_identity=freeze["identity"],
        source_audit=read(run / "data_audit.json"),
        concurrency="Independent package, source, session, output and report; no Push files modified",
    )
    atomic(run / "setup/preflight.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
