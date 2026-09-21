"""Cut-stage preparation, execution, and independent artifact verification."""

from pathlib import Path
from collections import Counter
import argparse
import json
import os
import shutil
import sqlite3
import time

import numpy as np

from appl.agent import AgentLoop
from appl.io import atomic, digest, object_hash, read
from real_robot.data import ROOT, CONTRACT, Sources, config
from real_robot.tools import CutTools
from real_robot.transport import Client


def prepare(cfg):
    run = ROOT / cfg["run"]
    run.mkdir(parents=True, exist_ok=True)
    if (run / "_session/journal.sqlite").exists():
        raise ValueError("Cannot prepare over an API session")
    atomic(
        run / "authorization.json",
        dict(
            date="2026-09-21",
            scope=cfg["scope"],
            user_authorization=cfg["authorization"],
            policy_code=False,
            training=False,
            flip_egg=False,
        ),
    )
    source = Sources(cfg)
    source.inventory(run)
    print(json.dumps(read(run / "data_audit.json")), flush=True)


def freeze_interface(cfg):
    run = ROOT / cfg["run"]
    paths = list((ROOT / "real_robot").glob("*.py")) + [
        Path(__file__).resolve(),
        ROOT / cfg["prompt"],
        ROOT / cfg["task_specification"],
        ROOT / cfg["data_contract"],
        ROOT / cfg["config_path"],
        ROOT / "environments/exp2/pixi.toml",
        ROOT / "environments/exp2/pixi.lock",
        ROOT / "src/appl/agent.py",
        ROOT / "src/appl/journal.py",
        ROOT / "src/appl/io.py",
    ]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}
    frozen = dict(
        config=cfg,
        files=hashes,
        identity=object_hash(hashes),
        source_manifest_sha256=digest(run / "source_manifest.json"),
    )
    marker = run / "interface_freeze.json"
    if marker.exists() and read(marker) != frozen:
        raise ValueError("Interface/source changed since launch; no silent restart")
    if not marker.exists():
        atomic(marker, frozen)
        for path in paths:
            target = run / "interface_snapshot" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    return frozen


def run(cfg):
    runroot = ROOT / cfg["run"]
    freeze = freeze_interface(cfg)
    tools = CutTools(cfg)
    client = Client(cfg, tools.j)
    prompt = (ROOT / cfg["prompt"]).read_text()
    message = dict(
        task="Inspect the authorized demonstrations and submit semantic segments, conditioning/supervision contracts, regrouped datasets, priors and a proposed callable policy catalog. Stop before policy code.",
        task_specification=(ROOT / cfg["task_specification"]).read_text(),
        data_contract=(ROOT / cfg["data_contract"]).read_text(),
        recording_contract=CONTRACT,
        catalog=tools.sources.catalog(),
        interface_identity=freeze["identity"],
        original_records=sum(e["n"] for e in tools.sources.episodes.values()),
        image_default="640x360 bounding box; original/crops available on request",
        range_convention="[start,stop) paired sample indices",
        heuristic_count="one or multiple per group, your scientific decision",
    )
    atomic(runroot / "prompt.json", dict(instructions=prompt, initial_message=message))
    atomic(
        runroot / "status.json",
        dict(
            status="running",
            started=time.time(),
            model=cfg["model"],
            reasoning_effort=cfg["reasoning_effort"],
        ),
    )
    try:
        AgentLoop(tools.j, tools, client, prompt=prompt).run(message)
        result = verify(cfg)
        atomic(
            runroot / "status.json",
            dict(status="complete", finished=time.time(), **result),
        )
        print(json.dumps(result), flush=True)
    except Exception as error:
        atomic(
            runroot / "status.json",
            dict(
                status="stopped",
                time=time.time(),
                error_type=type(error).__name__,
                reason=str(error),
                automatic_retry=False,
            ),
        )
        raise


def verify(cfg):
    runroot = ROOT / cfg["run"]
    out = ROOT / cfg["output"]
    manifest = read(out / "manifest.json")
    plan = read(out / "plan.json")
    if object_hash(plan) != manifest["plan_hash"]:
        raise ValueError("Plan identity mismatch")
    for name, h in manifest["files"].items():
        if digest(out / name) != h:
            raise ValueError("Published artifact changed: " + name)
    if digest(runroot / "source_manifest.json") != manifest["source_manifest_sha256"]:
        raise ValueError("Source manifest changed")
    freeze = read(runroot / "interface_freeze.json")
    for name, h in freeze["files"].items():
        if (
            digest(ROOT / name) != h
            or digest(runroot / "interface_snapshot" / name) != h
        ):
            raise ValueError("Frozen interface/source mismatch: " + name)
    saved_prompt = read(runroot / "prompt.json")
    expected_prompt = (ROOT / cfg["prompt"]).read_text()
    expected_spec = (ROOT / cfg["task_specification"]).read_text()
    expected_contract = (ROOT / cfg["data_contract"]).read_text()
    if (
        saved_prompt["instructions"] != expected_prompt
        or saved_prompt["initial_message"]["task_specification"] != expected_spec
        or saved_prompt["initial_message"]["data_contract"] != expected_contract
    ):
        raise ValueError(
            "General prompt, task specification or data contract differs from frozen input"
        )
    db = sqlite3.connect(
        "file:" + str((runroot / "_session/journal.sqlite").resolve()) + "?mode=ro",
        uri=True,
    )
    requests = list(
        db.execute("SELECT seq,status,request,response FROM api ORDER BY seq")
    )
    authored = []
    submitted = []
    usage = Counter()
    identity = Counter()
    for seq, status, raw_request, raw_response in requests:
        if status != "consumed":
            raise ValueError("Unreconciled API request")
        req = json.loads(raw_request)
        resp = json.loads(raw_response)
        expected = (cfg["model"], cfg["reasoning_effort"])
        if (req["model"], req.get("reasoning", {}).get("effort")) != expected:
            raise ValueError("Saved wire request model/effort mismatch")
        if (
            req["instructions"] != expected_prompt
            or json.loads(req["input"][0]["content"]) != saved_prompt["initial_message"]
        ):
            raise ValueError("Actual wire prompt or task specification mismatch")
        if (resp["model"], resp.get("reasoning", {}).get("effort")) != expected:
            raise ValueError("Response model/effort mismatch")
        if req.get("service_tier") != "default":
            raise ValueError("Wrong requested service tier")
        if read(runroot / "_session/api" / f"{seq:04d}.request.json") != req:
            raise ValueError("Wire-request file differs from journal")
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            usage[key] += resp["usage"][key]
        usage["cached_input_tokens"] += (
            resp["usage"].get("input_tokens_details", {}).get("cached_tokens", 0)
        )
        usage["reasoning_tokens_already_in_output"] += (
            resp["usage"].get("output_tokens_details", {}).get("reasoning_tokens", 0)
        )
        identity["gpt-6-astra/xhigh"] += 1
        for item in resp["output"]:
            if item.get("type") != "function_call":
                continue
            args = json.loads(item["arguments"])
            if item["name"] == "write_plan" and args["plan"] == plan:
                authored.append(seq)
            if (
                item["name"] == "submit_datasets"
                and args["expected_hash"] == manifest["plan_hash"]
            ):
                submitted.append(seq)
    if not authored or not submitted:
        raise ValueError("No exact Runtime API authorship/submission evidence")
    tool_states = dict(db.execute("SELECT status,COUNT(*) FROM tools GROUP BY status"))
    if set(tool_states) != {"completed"}:
        raise ValueError("Unreconciled tools")
    db.close()
    sources = Sources(cfg)
    sources.verify_metadata(read(runroot / "source_manifest.json"))
    checker = CutTools(cfg)
    if checker.validate(plan) != manifest["checks"]:
        raise ValueError(
            "Independent condition/evidence validation differs from publication"
        )
    checker.j.db.close()
    # Verify raw slices and exact original image-index associations independently.
    for item in manifest["datasets"]:
        dataset = read(out / item["dataset"])
        source_skill = next(
            s for s in plan["skills"] if s["skill_id"] == item["skill_id"]
        )
        if dataset["heuristics"] != source_skill["heuristics"]:
            raise ValueError("Heuristics were rewritten")
        for seg, source_seg in zip(
            dataset["segments"], source_skill["segments"], strict=True
        ):
            if any(seg[k] != v for k, v in source_seg.items()):
                raise ValueError("Cut differs from API plan")
            if read(ROOT / seg["directory"] / "supervision.json") != source_seg:
                raise ValueError("Published supervision contract differs from API plan")
            ep = sources.episode(seg["trajectory_id"])
            lo = seg["start"]
            hi = seg["stop"]
            folder = ROOT / seg["directory"]
            with np.load(folder / "records.npz", allow_pickle=False) as z:
                if set(z.files) != set(ep["arrays"]):
                    raise ValueError("Raw fields lost")
                for k, raw in ep["arrays"].items():
                    expected = raw[lo:hi]
                    same = (
                        np.array_equal(z[k], expected, equal_nan=True)
                        if raw.dtype.kind == "f"
                        else np.array_equal(z[k], expected)
                    )
                    if not same:
                        raise ValueError("Published rows differ from original")
            index = read(folder / "samples.json")
            if [x["source_sample_index"] for x in index["samples"]] != list(
                range(lo, hi)
            ):
                raise ValueError("Published image sample indices changed")
            if index["source_frame_records"] != ep["frames"]["samples"][lo:hi]:
                raise ValueError("Frame pairing changed")
    ledger = read(runroot / "cost_ledger.json")
    if ledger["stopped"] or any(
        r["status"] != "usage_reported" for r in ledger["records"]
    ):
        raise ValueError("Unresolved cost accounting")
    if len(ledger["records"]) != len(requests):
        raise ValueError("Cost/request count mismatch")
    result = dict(
        passed=True,
        stage="cut_and_heuristics_complete",
        plan_hash=manifest["plan_hash"],
        datasets=manifest["checks"]["datasets"],
        heuristics=manifest["checks"]["heuristics"],
        proposed_policies=manifest["checks"]["proposed_policies"],
        segments=manifest["checks"]["segments"],
        coverage=manifest["checks"]["coverage"],
        API_calls=len(requests),
        API_identity=dict(identity),
        usage=dict(usage),
        estimated_usd=sum(r["usd"] for r in ledger["records"]),
        cost_is_invoice=False,
        API_authored_plan_requests=authored,
        API_submit_requests=submitted,
        policy_code_written=False,
        training_runs=0,
        flip_egg_started=False,
        original_metadata_and_numeric_data_unchanged=True,
        original_media_integrity="All media verified before API submission publication",
        files_verified=len(manifest["files"]),
        semantic_validation=manifest["checks"]["semantic_validation"],
        general_prompt_sha256=digest(ROOT / cfg["prompt"]),
        task_specification_sha256=digest(ROOT / cfg["task_specification"]),
        actual_request_documents_verified=True,
    )
    atomic(runroot / "validation.json", result)
    report(cfg, plan, manifest, result)
    return result


def report(cfg, plan, manifest, result):
    lines = [
        f"# {cfg['task']}: API segmentation and heuristic designs",
        "",
        "Review date: 2026-09-21. Completed stage: cut/grouping and heuristic documents only.",
        "No policy code, preprocessing implementation, training, or physical robot execution has occurred.",
        "",
        f"Model: `{cfg['model']}`; reasoning: `{cfg['reasoning_effort']}`; verified API calls: {result['API_calls']}.",
        f"Reported input/output tokens: {result['usage']['input_tokens']:,} / {result['usage']['output_tokens']:,}. Estimated standard API cost: USD {result['estimated_usd']:.4f} (not an invoice).",
        "",
        "## Published datasets",
        "",
        "| Dataset | Original segments | Heuristic designs | Documents |",
        "| --- | ---: | ---: | --- |",
    ]
    report_root = ROOT / "real_robot/reports"
    report_root.mkdir(exist_ok=True)
    data_rel = os.path.relpath(ROOT / cfg["output"], report_root)
    run_rel = os.path.relpath(ROOT / cfg["run"], report_root)
    for d in manifest["datasets"]:
        rel = data_rel + "/datasets/" + d["skill_id"]
        lines.append(
            f"| {d['skill_id']} | {d['segments']} | {d['heuristics']} | [heuristics]({rel}/heuristic.md), [dataset]({rel}/dataset.json) |"
        )
    lines += [
        "",
        "## Coverage",
        "",
        json.dumps(result["coverage"], indent=2),
        "",
        "All intervals use original paired indices [start, stop). Raw NPZ slices retain missing values and original action JSON. Each segment has an independent sequence boundary and original media references with SHA-256 hashes. No timestamp re-pairing or cross-segment trajectory concatenation occurred.",
        "",
    ]
    for k in (
        "task_interpretation",
        "goal_conditioning",
        "portfolio_rationale",
        "model_inventory",
        "generalization_design",
        "calling_contract",
        "capability_coverage",
        "sharing_and_diversity_audit",
        "motion_coverage",
        "unobserved_cases",
        "overlap_rationale",
    ):
        lines += [
            "## " + k.replace("_", " ").capitalize() + " (verbatim API output)",
            "",
            plan[k],
            "",
        ]
    lines += [
        "## Evidence",
        "",
        f"- [API-authored plan]({data_rel}/plan.json)",
        f"- [Published artifact hashes]({data_rel}/manifest.json)",
        f"- [Independent validation]({run_rel}/validation.json)",
        f"- [API cost ledger]({run_rel}/cost_ledger.json)",
        f"- [Source inventory]({run_rel}/source_manifest.json)",
        f"- [Proposed policy catalog]({data_rel}/POLICY_CATALOG.md)",
        f"- [Frozen general prompt]({run_rel}/interface_snapshot/{cfg['prompt']})",
        f"- [Frozen task specification]({run_rel}/interface_snapshot/{cfg['task_specification']})",
        "",
    ]
    (report_root / "PUSH_CUT_V5.md").write_text("\n".join(lines))


def status(cfg):
    root = ROOT / cfg["run"]
    value = {}
    if (root / "status.json").exists():
        value["status"] = read(root / "status.json")
    if (root / "cost_ledger.json").exists():
        ledger = read(root / "cost_ledger.json")
        value["cost"] = dict(
            estimated_usd=sum(r.get("usd", 0) for r in ledger["records"]),
            reserved_unknown_usd=sum(
                r["reserved_usd"] for r in ledger["records"] if "usd" not in r
            ),
            stopped=ledger["stopped"],
        )
    path = root / "_session/journal.sqlite"
    if path.exists():
        db = sqlite3.connect("file:" + str(path.resolve()) + "?mode=ro", uri=True)
        value["API_states"] = dict(
            db.execute("SELECT status,COUNT(*) FROM api GROUP BY status")
        )
        value["tools"] = dict(
            db.execute("SELECT name,COUNT(*) FROM tools GROUP BY name")
        )
        last = db.execute(
            "SELECT seq,status,response FROM api ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        if last:
            value["last_API"] = dict(seq=last[0], status=last[1])
            if last[2]:
                response = json.loads(last[2])
                value["last_API"]["outputs"] = [
                    dict(type=i["type"], name=i.get("name"))
                    for i in response.get("output", [])
                ]
        db.close()
    print(json.dumps(value), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["prepare", "run", "verify", "status"])
    p.add_argument("--config", default="real_robot/configs/push_cut_v5.json")
    args = p.parse_args()
    cfg = config(ROOT / args.config)
    result = globals()[args.command](cfg)
    if args.command == "verify":
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
