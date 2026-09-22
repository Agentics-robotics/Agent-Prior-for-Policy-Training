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
import fnmatch
import re
import base64


from appl.agent import AgentLoop
from appl.io import atomic, digest, object_hash, read
from appl.journal import safe
from real_robot.data import ROOT
from real_robot.tools import CutTools, obj
from real_robot.transport import Client, PRICING, RATES
from .security import audit

from .contract import configuration, validate_package, recording_facts


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
                "Read task facts, authorized recordings, previous API design as revisable context, resource limits and current feedback.",
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
                "request_model",
                "Download an explicitly selected public Hugging Face model snapshot to a local directory, pin resolved commit and every file hash. Downloads only tensor/config/tokenizer files, not executable remote code. No automatic retry.",
                dict(name=text_type, repo_id=text_type, revision=text_type,
                     allow_patterns=dict(type="array", items=text_type, minItems=1, maxItems=32),
                     license=text_type, purpose=text_type),
            ),
            tool(
                "check_preprocessing",
                "Execute your current checked package's full preprocessing and zero-optimizer-update model interface checks in isolation. Retain per-anchor reports/images and exact source snapshot; bounded by the configured preprocessing check allowance. No training or performance search. Successful unchanged cache is reusable at submission.",
                {},
            ),
            tool(
                "read_preparation_evidence",
                "Inspect a JSON/text report or PNG image produced by the last check_preprocessing. Only paths returned by that check are available; images use an inspection preview.",
                dict(path=text_type),
            ),
            tool(
                "check_package",
                "Check package syntax/imports/hooks/metadata only. Zero training updates.",
                {},
            ),
            tool(
                "submit_package",
                "Freeze the checked implementation after you explicitly assess its actual preparation coverage and training readiness.",
                dict(expected_hash=text_type, preparation_hash=text_type, readiness=obj(dict(decision=dict(type="string", enum=["train", "blocked"]), coverage_assessment=text_type, data_recovery_assessment=text_type, training_rationale=text_type, limitations=text_type))),
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
                recording_contract=recording_facts(self.sources),
                task_specification=(ROOT / cfg["task_specification"]).read_text(),
                latest_stage_authorization=cfg["authorization"],
                training=cfg["training"],
                catalog=self.sources.catalog(),
                prior_design_is_revisable=True,
                previous_execution_evidence={path: (ROOT / path).read_text()
                    for path in cfg.get("previous_execution_evidence", [])},
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
        if name == "request_model":
            return self.download_model(args)
        if name == "check_preprocessing":
            return self.check_preprocessing()
        if name == "read_preparation_evidence":
            return self.read_preparation_evidence(args["path"])
        if name == "check_package":
            if not self.j.get("assignment_read") or not self.j.get("interface_read"):
                raise ValueError("Read both assignment and interface before checking")
            result = validate_package(self.work, self.training_cfg)
            self.j.set("checked_package", result)
            return result
        if name == "submit_package":
            if self.done():
                raise ValueError("Package already submitted")
            result = validate_package(self.work, self.training_cfg)
            if (
                args["expected_hash"] != result["package_hash"]
                or self.j.get("checked_package") != result
            ):
                raise ValueError("Submit the exact successfully checked package hash")
            check = self.j.get("last_preparation_check")
            if not check or check["returncode"] or check["package_hash"] != result["package_hash"]:
                raise ValueError("Inspect successful preparation for this exact source before submission")
            manifest_path = Path(check["directory"]) / "prepared/result.json"
            if args["preparation_hash"] != digest(manifest_path):
                raise ValueError("Readiness must refer to the actual preparation manifest hash")
            readiness = args["readiness"]
            if readiness["decision"] not in {"train", "blocked"} or any(
                not isinstance(v, str) or not v.strip() for v in readiness.values()
            ):
                raise ValueError("Provide an explicit data and training readiness assessment")
            prepared = read(manifest_path)
            if readiness["decision"] == "train" and any(
                not value["passed"] for value in prepared["model_interface_checks"].values()
            ):
                raise ValueError("Training needs eligible examples and working interfaces for each policy")
            target = ROOT / self.training_cfg["run"] / f"package_{self.revision:02d}"
            if target.exists():
                raise ValueError("Existing package cannot be overwritten")
            shutil.copytree(self.work, target / "source")
            assets = {str(p.relative_to(self.assets)): digest(p) for p in self.assets.rglob("*") if p.is_file()}
            receipt = dict(
                **result,
                source=str(target / "source"),
                revision=self.revision,
                assets=assets,
                source_plan_hash=object_hash(
                    read(ROOT / self.training_cfg["cut_output"] / "plan.json")
                ),
                time=time.time(),
                readiness=args["readiness"],
                preparation_hash=args["preparation_hash"],
            )
            check = self.j.get("last_preparation_check")
            if check and check["package_hash"] == result["package_hash"] and check["returncode"] == 0:
                shutil.copytree(Path(check["directory"]) / "prepared", target / "prepared")
                receipt["preparation_check"] = check
            atomic(target / "submission.json", receipt)
            self.j.set("submitted", receipt)
            self.j.set("frozen", receipt)
            return dict(submitted=True, **receipt)
        raise ValueError("Unknown implementation tool")

    def check_preprocessing(self):
        from .runner import start_job, finish_job
        result = validate_package(self.work, self.training_cfg)
        self.j.charge("preprocessing_check", self.training_cfg["max_preprocessing_checks"])
        folder = self.run / "preprocessing_checks" / result["package_hash"]
        if folder.exists():
            raise ValueError("This source hash already has a check; inspect its retained result before a source revision")
        folder.mkdir(parents=True)
        shutil.copytree(self.work, folder / "source")
        job = start_job(self.training_cfg, folder, "prepare", folder / "prepared", self.training_cfg["devices"][0])
        checked = finish_job(job)
        evidence = [str(p.relative_to(folder / "prepared")) for p in (folder / "prepared" / "evidence").glob("*") if p.is_file()]
        record = dict(package_hash=result["package_hash"], directory=str(folder), returncode=checked["returncode"], evidence=evidence, training_updates=0)
        self.j.set("last_preparation_check", record)
        atomic(folder / "check_receipt.json", dict(**record, result=checked))
        if checked["returncode"]:
            return dict(**record, error=checked.get("error", "")[-12000:], repair_allowed=True)
        r=checked["result"]
        return dict(**record, preparation_hash=digest(folder / "prepared/result.json"), num_examples=r["num_examples"], segment_examples=r["segment_examples"], source_coverage=r["source_coverage"], coverage=r["API_declared_coverage"], data_audit=r["API_data_audit"], interface_checks=r["model_interface_checks"])

    def read_preparation_evidence(self, name):
        record = self.j.get("last_preparation_check")
        if not record or name not in record["evidence"]:
            raise ValueError("Choose an evidence path returned by the last preprocessing check")
        path = safe(Path(record["directory"]) / "prepared", name)
        if path.suffix == ".png":
            from PIL import Image
            import io
            with Image.open(path) as im:
                im=im.convert("RGB"); original=im.size; im.thumbnail((1280,960))
                buffer=io.BytesIO(); im.save(buffer,format="PNG")
                return [dict(type="input_text",text=json.dumps(dict(path=name,original_size=original,preview_size=im.size,sha256=digest(path)))),dict(type="input_image",detail="high",image_url="data:image/png;base64,"+base64.b64encode(buffer.getvalue()).decode())]
        text=path.read_text()
        if len(text)>80000:
            raise ValueError("Evidence report exceeds 80000 characters; write smaller per-anchor reports")
        return dict(path=name,content=text,sha256=digest(path))

    def download_model(self, args):
        if self.done():
            raise ValueError("Assets cannot change after submission")
        name=args["name"]
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}",name) or not re.fullmatch(r"[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+",args["repo_id"]) or not re.fullmatch(r"[a-zA-Z0-9_.-]+",args["revision"]):
            raise ValueError("Invalid public model identity")
        folder=self.assets/name; receipt=self.assets/(name+".receipt.json")
        if folder.exists() or receipt.exists():
            raise ValueError("Model request already recorded; no automatic retry or overwrite")
        atomic(receipt,dict(request=args,status="started",automatic_retry=False))
        opener=urllib.request.build_opener(AssetRedirect())
        try:
            url=f"https://huggingface.co/api/models/{args['repo_id']}/revision/{args['revision']}"
            with opener.open(url,timeout=180) as response: info=json.load(response)
            commit=info["sha"]
            if not re.fullmatch(r"[a-f0-9]{40}",commit): raise ValueError("Model revision did not resolve to a commit")
            selected=[f["rfilename"] for f in info["siblings"] if any(fnmatch.fnmatch(f["rfilename"],p) for p in args["allow_patterns"])]
            if not selected: raise ValueError("No snapshot files match requested patterns")
            files={}; total=0; folder.mkdir()
            for filename in selected:
                rel=Path(filename)
                if rel.is_absolute() or ".." in rel.parts or rel.suffix not in {".safetensors",".pt",".pth",".json",".txt",".model",".yaml",".yml",".md"}:
                    raise ValueError("Snapshot includes unsupported or executable file: "+filename)
                target=folder/rel;target.parent.mkdir(parents=True,exist_ok=True)
                source_url=f"https://huggingface.co/{args['repo_id']}/resolve/{commit}/{filename}"
                with opener.open(source_url,timeout=300) as response, target.with_suffix(target.suffix+".partial").open("xb") as output:
                    while block:=response.read(8*1024**2):
                        total+=len(block)
                        if total>24_000_000_000: raise ValueError("Model snapshot exceeds 24 GB bound")
                        output.write(block)
                target.with_suffix(target.suffix+".partial").rename(target)
                files[filename]=dict(sha256=digest(target),bytes=target.stat().st_size,url=source_url)
            result=dict(request=args,status="complete",resolved_commit=commit,files=files,total_bytes=total,automatic_retry=False)
            atomic(folder/"asset_manifest.json",result);atomic(receipt,result)
            return result
        except Exception as error:
            atomic(receipt,dict(request=args,status="failed",error_type=type(error).__name__,reason=str(error),automatic_retry=False))
            raise ValueError("Model download failed with retained receipt: "+str(error)) from error

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
            ".yaml",
            ".yml",
            ".txt",
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
        + list((ROOT / "real_robot/training_pipeline").glob("*.py"))
        + [
            ROOT / "real_robot/transport.py",
            ROOT / "real_robot/tools.py",
            ROOT / "real_robot/data.py",
            ROOT / "src/appl/agent.py",
            ROOT / "src/appl/journal.py",
            ROOT / cfg["cut_config"],
            ROOT / cfg["cut_output"] / "plan.json",
            *[ROOT / path for path in cfg.get("previous_execution_evidence", [])],
            ROOT / cfg["environment_manifest"],
            (ROOT / cfg["environment_manifest"]).with_name("pixi.lock"),
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
        task="Implement and train the supplied API cut/prior design for the task and authorized recordings. Read assignment and interface, inspect the data, choose and implement preprocessing, training examples, representations, models and training settings. Use actual preparation feedback to refine implementation, assess readiness and submit the exact checked package. Retain the original cut/prior artifacts and record evidence-based implementation changes in your derived data plan. This is implementation and training, not a new cut-stage experiment.",
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
    p.add_argument("--config", default="real_robot/configs/push_training_v5.json")
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
