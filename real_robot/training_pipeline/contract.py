"""Execution/provenance constraints; scientific choices belong to the API."""

import ast
import math
from pathlib import Path
import re

import jsonschema
import numpy as np

from appl.io import digest, object_hash, read
from .security import audit

HOOKS = {
    "prepare_data", "make_batch", "build_model", "compute_loss", "predict",
    "configure_optimizer", "sample_indices", "before_update", "after_update",
    "act", "executor_request", "calling_cases",
}
HANDOFF_FIELDS = {
    "responsibility", "selection_cues", "entry_conditions", "continuation_conditions",
    "exit_conditions", "successor_readiness", "switch_procedure", "failure_signatures",
    "memory_reset_rules", "observation_dependencies", "training_evidence", "limitations",
}


def configuration(path):
    cfg = read(path)
    if cfg["schema"] != "real_robot.train_config.pipeline.v1" or (
        cfg["model"], cfg["reasoning_effort"]
    ) != ("gpt-6-astra", "xhigh"):
        raise ValueError("New runs require explicit gpt-6-astra/xhigh")
    if any(k in cfg for k in ("policies", "policy_datasets")):
        raise ValueError("Policy inventory and data mapping are API-owned")
    allowed = {"max_updates", "parameter_limit", "max_cache_bytes", "max_policies",
               "log_every", "checkpoint_every"}
    if set(cfg["training"]) != allowed or any(
        type(v) is not int or v <= 0 for v in cfg["training"].values()
    ):
        raise ValueError("training contains positive execution limits/logging intervals only")
    for key in ("max_preprocessing_checks", "max_implementation_revisions", "max_job_seconds"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError("Invalid execution limit: " + key)
    if not cfg["devices"] or any(type(g) is not int or g < 0 for g in cfg["devices"]):
        raise ValueError("An explicit physical GPU allocation is required")
    return cfg


def recording_facts(sources):
    """Derive factual fields, without importing Push-specific missingness claims."""
    return dict(
        row_semantics="Original paired sample indices, [start,stop); no re-pairing or fabricated terminal row.",
        camera_semantics="Read original metadata; raw depth units and registration require evidence.",
        episodes={tid: dict(
            records=ep["n"], metadata=ep["meta"], fields={
                k: dict(shape=list(a.shape), dtype=str(a.dtype),
                        finite_values=int(np.isfinite(a).sum()) if a.dtype.kind in "fciub" else None,
                        values=int(a.size))
                for k, a in ep["arrays"].items()
            }) for tid, ep in sources.episodes.items()},
    )


def validate_package(source, cfg):
    source = Path(source)
    if any(p.is_symlink() or not p.is_file() or p.suffix not in {".py", ".json", ".md"}
           for p in source.iterdir()):
        raise ValueError("Package must contain flat regular Python/JSON/Markdown files")
    audit(source)
    metadata = read(source / "package.json")
    required_keys = {"schema", "policies", "preprocessing", "data_plan", "dependencies", "adaptations", "limitations"}
    if set(metadata) != required_keys or metadata["schema"] != "real_robot.policy_package.pipeline.v1":
        raise ValueError("package.json schema/keys differ from the generic interface")
    policies = metadata["policies"]
    if not isinstance(policies, dict) or not 1 <= len(policies) <= cfg["training"]["max_policies"]:
        raise ValueError("Declare the learned policy inventory within resource limits")
    required = {"policy.py", "package.json", "PRIOR.md", "CALLING.md", "POLICY_CATALOG.md", "HANDOFF.json"}
    for policy_id, p in policies.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", policy_id):
            raise ValueError("Invalid policy identifier")
        required |= {f"{policy_id}_PRIOR.md", f"{policy_id}_USAGE.md"}
        keys = {"description", "model_family", "architecture_rationale", "config", "updates",
                "batch_size", "seed", "ema_decay", "max_grad_norm", "prediction_dimension", "decision_schema"}
        if set(p) != keys:
            raise ValueError("Policy metadata differs from interface: " + policy_id)
        for key, maximum in (("updates", cfg["training"]["max_updates"]),
                             ("batch_size", 65536), ("prediction_dimension", 1048576)):
            if type(p[key]) is not int or not 1 <= p[key] <= maximum:
                raise ValueError("Invalid declared execution size: " + key)
        if type(p["seed"]) is not int or not 0 <= p["seed"] < 2**32:
            raise ValueError("Declare a reproducible seed")
        for key in ("ema_decay", "max_grad_norm"):
            v = p[key]
            if v is not None and (type(v) not in (int, float) or not math.isfinite(v) or
                                  not (0 <= v < 1 if key == "ema_decay" else v > 0)):
                raise ValueError("Invalid optional training setting: " + key)
        for key in ("description", "model_family", "architecture_rationale"):
            if not isinstance(p[key], str) or not p[key].strip():
                raise ValueError("Document the chosen design: " + key)
        if not isinstance(p["config"], dict) or not isinstance(p["decision_schema"], dict):
            raise ValueError("config and decision_schema must be objects")
        jsonschema.Draft202012Validator.check_schema(p["decision_schema"])
    for key in ("dependencies", "adaptations", "limitations"):
        if not isinstance(metadata[key], str) or not metadata[key].strip():
            raise ValueError("Document " + key)
    if not isinstance(metadata["preprocessing"], dict):
        raise ValueError("preprocessing must be an API-defined object")
    plan = metadata["data_plan"]
    if set(plan) != {"rationale", "changes_from_prior", "segments"} or not plan["rationale"] or not plan["changes_from_prior"]:
        raise ValueError("Provide a derived data plan and explain changes from the prior design")
    if not isinstance(plan["segments"], list) or not plan["segments"]:
        raise ValueError("Declare original-episode boundaries for derived learning windows")
    ids = set()
    for s in plan["segments"]:
        if set(s) != {"segment_id", "trajectory_id", "start", "stop", "rationale"}:
            raise ValueError("Invalid derived segment keys")
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", s["segment_id"]) or s["segment_id"] in ids:
            raise ValueError("Derived segment IDs must be unique")
        ids.add(s["segment_id"])
        if type(s["start"]) is not int or type(s["stop"]) is not int or not 0 <= s["start"] < s["stop"] or not s["rationale"]:
            raise ValueError("Invalid derived original-source interval")
    files = {p.name: digest(p) for p in source.iterdir()}
    if not required <= files.keys():
        raise ValueError("Missing package files: " + str(sorted(required - files.keys())))
    definitions = {n.name for n in ast.parse((source / "policy.py").read_text()).body if isinstance(n, ast.FunctionDef)}
    if not HOOKS <= definitions:
        raise ValueError("Missing executable hooks: " + str(sorted(HOOKS - definitions)))
    for name in required:
        if name.endswith(".md") and len((source / name).read_text().strip()) < 100:
            raise ValueError("Document model, data, calling and deployment assumptions: " + name)
    handoff = read(source / "HANDOFF.json").get("policies", {})
    if set(handoff) != set(policies) or any(not HANDOFF_FIELDS <= handoff[p].keys() for p in policies):
        raise ValueError("Document the calling/handoff contract for every policy")
    return dict(package_hash=object_hash(files), files=files,
                checks="Syntax/import/metadata/hook checks only; zero candidate optimizer updates.")
