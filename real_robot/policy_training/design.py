"""Bounded Runtime API source-authoring tools, with immutable submissions."""

from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit
import ast
import json
import math
import shutil
import time
import urllib.request


from appl.agent import AgentLoop
from appl.io import atomic, digest, object_hash, read
from appl.journal import safe
from real_robot.data import ROOT
from real_robot.tools import CutTools, obj
from real_robot.transport import Client, PRICING, RATES
from .security import audit

HOOKS = {
    "prepare_data",
    "make_batch",
    "build_model",
    "compute_loss",
    "predict",
    "act",
    "decode_action",
}
POLICIES = {"contour_push": "boundary_graph", "visual_push": "dual_view_memory"}


def configuration(path):
    cfg = read(path)
    if cfg["schema"] != "real_robot.train_config.v1" or (
        cfg["model"],
        cfg["reasoning_effort"],
    ) != ("gpt-6-astra", "xhigh"):
        raise ValueError("Expected explicit Astra/xhigh training configuration")
    if cfg["policies"] != list(POLICIES):
        raise ValueError("This run implements exactly the two frozen policies")
    return cfg


def validate_package(source):
    source = Path(source)
    required = {"policy.py", "package.json", "PRIOR.md", "CALLING.md"}
    files = {p.name: digest(p) for p in source.iterdir() if p.is_file()}
    if not required <= files.keys():
        raise ValueError("Required files: " + str(sorted(required)))
    try:
        audit(source)
    except SyntaxError as error:
        raise ValueError("Python syntax error: " + str(error)) from error
    definitions = {
        node.name
        for node in ast.parse((source / "policy.py").read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if not HOOKS <= definitions:
        raise ValueError(
            "Missing top-level policy.py hooks: " + str(sorted(HOOKS - definitions))
        )
    metadata = read(source / "package.json")
    if (
        set(metadata)
        != {
            "schema",
            "policies",
            "preprocessing",
            "dependencies",
            "adaptations",
            "limitations",
        }
        or metadata["schema"] != "real_robot.policy_package.v1"
    ):
        raise ValueError("package.json top-level keys/schema do not match INTERFACE.md")
    if set(metadata["policies"]) != set(POLICIES):
        raise ValueError("Implement exactly the two assigned policies")
    for policy_id, heuristic in POLICIES.items():
        p = metadata["policies"][policy_id]
        if (
            set(p)
            != {
                "heuristic_id",
                "batch_size",
                "learning_rate",
                "weight_decay",
                "max_grad_norm",
                "config",
                "description",
            }
            or p["heuristic_id"] != heuristic
        ):
            raise ValueError("Policy metadata/heuristic mismatch: " + policy_id)
        if type(p["batch_size"]) is not int or not 1 <= p["batch_size"] <= 128:
            raise ValueError("batch_size must be an integer 1..128")
        for key, lower, upper in [
            ("learning_rate", 0, 0.1),
            ("weight_decay", -1e-20, 1),
            ("max_grad_norm", 0, 1000),
        ]:
            if (
                not isinstance(p[key], (int, float))
                or not math.isfinite(p[key])
                or not lower < p[key] <= upper
            ):
                raise ValueError("Invalid optimizer setting: " + key)
        if (
            not isinstance(p["config"], dict)
            or not isinstance(p["description"], str)
            or not p["description"].strip()
        ):
            raise ValueError("Each policy requires config and description")
    if not isinstance(metadata["preprocessing"], dict):
        raise ValueError("preprocessing must be a JSON object")
    for name in ["dependencies", "adaptations", "limitations"]:
        if not isinstance(metadata[name], str) or not metadata[name].strip():
            raise ValueError("Nonempty metadata required: " + name)
    for name in ["PRIOR.md", "CALLING.md"]:
        if len((source / name).read_text().strip()) < 100:
            raise ValueError("Complete scientific/calling documentation required")
    return dict(
        package_hash=object_hash(files),
        files=files,
        checks="Syntax/import/metadata/hook checks only; zero candidate optimizer updates.",
    )


def allowed_asset_url(url):
    parsed = urlsplit(url)
    hosts = (
        "huggingface.co",
        "download.pytorch.org",
        "dl.fbaipublicfiles.com",
        "hf.co",
        "xethub.hf.co",
    )
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or not parsed.hostname
        or not any(
            parsed.hostname == h or parsed.hostname.endswith("." + h) for h in hosts
        )
    ):
        raise ValueError("Asset URL must be HTTPS on an approved public model host")


class AssetRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        allowed_asset_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class ImplementationTools(CutTools):
    def __init__(self, cfg, revision, feedback=None, previous_source=None):
        self.training_cfg = cfg
        self.revision = revision
        run = ROOT / cfg["run"] / f"design_{revision:02d}"
        run.mkdir(parents=True, exist_ok=True)
        manifest = run / "source_manifest.json"
        original = ROOT / cfg["cut_run"] / "source_manifest.json"
        if not manifest.exists():
            shutil.copyfile(original, manifest)
        elif digest(manifest) != digest(original):
            raise ValueError("Source identity changed")
        cut_cfg = read(ROOT / cfg["cut_config"])
        cut_cfg.update(run=str(run.relative_to(ROOT)))
        super().__init__(cut_cfg)
        self.budget = SimpleNamespace(
            **{
                k: cfg[k]
                for k in ["max_api_calls", "max_tool_calls", "max_output_tokens"]
            }
        )
        self.work = self.j.root / "work"
        self.work.mkdir(exist_ok=True)
        self.assets = ROOT / cfg["run"] / "assets"
        self.assets.mkdir(exist_ok=True)
        self.feedback = feedback
        if previous_source and not any(self.work.iterdir()):
            for file in Path(previous_source).iterdir():
                if file.is_file():
                    self.j.write_candidate(file.name, file.read_text())
            self.j.event(
                "repair_workspace_initialized", previous_source=str(previous_source)
            )

    def done(self):
        return bool(self.j.get("submitted"))

    def schemas(self):
        schemas = [
            s
            for s in super().schemas()
            if s["name"] in {"read_images", "read_steps", "read_metadata"}
        ]
        text_type = dict(type="string", minLength=1)

        def tool(name, description, properties):
            return dict(
                type="function",
                name=name,
                description=description,
                strict=True,
                parameters=obj(properties),
            )

        return schemas + [
            tool(
                "read_assignment",
                "Read the complete frozen API plan, two priors, paired-data contract and implementation scope.",
                {},
            ),
            tool(
                "read_interface",
                "Read the complete mandatory numerical and invocation contract.",
                {},
            ),
            tool(
                "write_file",
                "Write one unsubmitted API-authored flat .py/.json/.md file; every version is retained.",
                dict(path=text_type, content=text_type),
            ),
            tool(
                "read_file", "Read your current unsubmitted file.", dict(path=text_type)
            ),
            tool(
                "request_asset",
                "Download a concrete public model tensor asset. No automatic retry; disclose provenance/license. Does not install Python packages.",
                dict(
                    name=text_type,
                    url=text_type,
                    expected_sha256=dict(type="string"),
                    license=text_type,
                    purpose=text_type,
                ),
            ),
            tool(
                "check_package",
                "Check both policies syntax/imports/hooks/metadata only. Zero training updates; no preliminary performance study.",
                {},
            ),
            tool(
                "submit_package",
                "Freeze both implementations at the exact successfully checked hash for full preprocessing and independent training.",
                dict(expected_hash=text_type),
            ),
        ]

    def dispatch(self, name, args):
        if name in {"read_images", "read_steps", "read_metadata"}:
            return super().dispatch(name, args)
        if name == "read_assignment":
            self.j.set("assignment_read", True)
            cfg = self.training_cfg
            return dict(
                source_plan=read(ROOT / cfg["cut_output"] / "plan.json"),
                recording_contract=read(ROOT / cfg["cut_run"] / "data_audit.json")[
                    "contract"
                ],
                task_specification=(ROOT / cfg["task_specification"]).read_text(),
                latest_stage_authorization=cfg["authorization"],
                training=cfg["training"],
                feedback=self.feedback,
                existing_files=sorted(
                    p.name for p in self.work.iterdir() if p.is_file()
                ),
                available_assets=sorted(
                    p.name
                    for p in self.assets.iterdir()
                    if p.is_file() and not p.name.endswith(".receipt.json")
                ),
            )
        if name == "read_interface":
            self.j.set("interface_read", True)
            return dict(content=(ROOT / self.training_cfg["interface"]).read_text())
        if name in {"write_file", "read_file"}:
            path = args["path"]
            if Path(path).name != path or Path(path).suffix not in {
                ".py",
                ".json",
                ".md",
            }:
                raise ValueError("Use flat .py/.json/.md files")
            if name == "read_file":
                return dict(content=safe(self.work, path).read_text())
            if self.done():
                raise ValueError("Submitted package is immutable")
            return self.j.write_candidate(path, args["content"])
        if name == "request_asset":
            return self.download_asset(args)
        if name == "check_package":
            if not self.j.get("assignment_read") or not self.j.get("interface_read"):
                raise ValueError("Read both assignment and interface before checking")
            result = validate_package(self.work)
            self.j.set("checked_package", result)
            return result
        if name == "submit_package":
            if self.done():
                raise ValueError("Package already submitted")
            result = validate_package(self.work)
            if (
                args["expected_hash"] != result["package_hash"]
                or self.j.get("checked_package") != result
            ):
                raise ValueError("Submit the exact successfully checked package hash")
            target = ROOT / self.training_cfg["run"] / f"package_{self.revision:02d}"
            if target.exists():
                raise ValueError("Existing package cannot be overwritten")
            shutil.copytree(self.work, target / "source")
            assets = {p.name: digest(p) for p in self.assets.iterdir() if p.is_file()}
            receipt = dict(
                **result,
                source=str(target / "source"),
                revision=self.revision,
                assets=assets,
                source_plan_hash=object_hash(
                    read(ROOT / self.training_cfg["cut_output"] / "plan.json")
                ),
                time=time.time(),
            )
            atomic(target / "submission.json", receipt)
            self.j.set("submitted", receipt)
            self.j.set("frozen", receipt)
            return dict(submitted=True, **receipt)
        raise ValueError("Unknown implementation tool")

    def download_asset(self, args):
        if self.done():
            raise ValueError("Assets cannot change after submission")
        name = args["name"]
        if Path(name).name != name or Path(name).suffix not in {
            ".pt",
            ".pth",
            ".safetensors",
            ".npz",
            ".json",
        }:
            raise ValueError("Use flat tensor/config asset filenames")
        allowed_asset_url(args["url"])
        target = self.assets / name
        receipt = target.with_name(name + ".receipt.json")
        if target.exists() or receipt.exists():
            raise ValueError(
                "Asset request already recorded; no automatic retry or overwrite"
            )
        atomic(
            receipt,
            dict(
                request=args, status="started", time=time.time(), automatic_retry=False
            ),
        )
        partial = target.with_name(name + ".partial")
        try:
            opener = urllib.request.build_opener(AssetRedirect())
            size = 0
            with (
                opener.open(args["url"], timeout=180) as response,
                partial.open("xb") as output,
            ):
                while block := response.read(1024 * 1024):
                    size += len(block)
                    if size > 2_000_000_000:
                        raise ValueError("Asset exceeds 2 GB bound")
                    output.write(block)
            sha = digest(partial)
            if args["expected_sha256"] and sha != args["expected_sha256"]:
                raise ValueError("Asset checksum differs from supplied identity")
            partial.rename(target)
            value = dict(
                request=args,
                status="complete",
                bytes=size,
                sha256=sha,
                automatic_retry=False,
            )
            atomic(receipt, value)
            return value
        except Exception as error:
            atomic(
                receipt,
                dict(
                    request=args,
                    status="failed",
                    error_type=type(error).__name__,
                    automatic_retry=False,
                ),
            )
            raise ValueError(
                "Asset download failed and recorded: " + type(error).__name__
            ) from error


def design(cfg, revision=0, feedback=None, previous_source=None):
    tools = ImplementationTools(cfg, revision, feedback, previous_source)
    run = tools.run
    prompt = (ROOT / cfg["prompt"]).read_text()
    frozen_paths = (
        [
            ROOT / cfg[k]
            for k in ["config_path", "prompt", "interface", "task_specification"]
        ]
        + list((ROOT / "real_robot/policy_training").glob("*.py"))
        + [
            ROOT / "real_robot/transport.py",
            ROOT / "real_robot/tools.py",
            ROOT / "real_robot/data.py",
            ROOT / "src/appl/agent.py",
            ROOT / "src/appl/journal.py",
        ]
    )
    frozen = {str(p.relative_to(ROOT)): digest(p) for p in frozen_paths}
    marker = run / "framework.json"
    if marker.exists() and read(marker) != frozen:
        raise ValueError("Design interface changed; no silent continuation")
    atomic(marker, frozen)
    for p in frozen_paths:
        target = run / "interface_snapshot" / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(p, target)
    ledger = run / "cost_ledger.json"
    if not ledger.exists():
        atomic(
            ledger,
            dict(
                schema="real_robot.api_cost.v1",
                pricing=PRICING,
                rates_usd_per_million=RATES,
                execution_guard_usd=cfg["execution_guard_usd"],
                guard_scope="Developer execution guard for this authorized implementation session; not an asserted user budget.",
                records=[],
                stopped=None,
            ),
        )
    message = dict(
        task="Implement BOTH existing cut_v3 policies and their complete executable observation/goal/action pipelines. Read interface and assignment, write source, check syntax/interface, submit. Outer runner trains independently; no preliminary performance study.",
        revision=revision,
        feedback=feedback,
    )
    atomic(run / "prompt.json", dict(instructions=prompt, initial_message=message))
    atomic(run / "status.json", dict(status="running", started=time.time()))
    try:
        result = AgentLoop(tools.j, tools, Client(cfg, tools.j), prompt).run(message)
        atomic(
            run / "status.json",
            dict(status="complete", finished=time.time(), submission=result),
        )
        return result
    except Exception as error:
        atomic(
            run / "status.json",
            dict(
                status="stopped",
                error_type=type(error).__name__,
                reason=str(error),
                automatic_retry=False,
            ),
        )
        raise
    finally:
        tools.j.db.close()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="real_robot/configs/push_train_v1.json")
    p.add_argument("--revision", type=int, default=0)
    p.add_argument("--feedback")
    p.add_argument("--previous-source")
    a = p.parse_args()
    print(
        json.dumps(
            design(
                configuration(ROOT / a.config),
                a.revision,
                read(a.feedback) if a.feedback else None,
                a.previous_source,
            )
        ),
        flush=True,
    )
