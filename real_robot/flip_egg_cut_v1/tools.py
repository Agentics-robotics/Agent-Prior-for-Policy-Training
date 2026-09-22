"""Bounded API evidence, plan validation and exact cut publication."""

from dataclasses import dataclass
import base64
import json
import time

import jsonschema
import numpy as np

from appl.io import atomic, digest, object_hash, read
from appl.journal import Journal, encode
from real_robot.flip_egg_cut_v1.data import CONTRACT, ROOT, Sources


def obj(properties):
    return dict(
        type="object",
        properties=properties,
        required=list(properties),
        additionalProperties=False,
    )


TEXT = dict(type="string", minLength=1, maxLength=16000)
INDEX = dict(type="integer", minimum=0)
ID = dict(type="string", pattern="^[a-z][a-z0-9_]{0,63}$")


def arr(item, maximum=256, minimum=1):
    return dict(type="array", items=item, minItems=minimum, maxItems=maximum)


RANGE = dict(trajectory_id=TEXT, start=INDEX, stop=dict(type="integer", minimum=1))
LOCAL_RANGE = dict(start=INDEX, stop=dict(type="integer", minimum=1))
SEGMENT = obj(
    dict(
        **RANGE,
        segment_id=ID,
        rationale=TEXT,
        objective=TEXT,
        supervision_kind=dict(type="string", enum=["dense", "sparse", "executor_only"]),
        decision_indices=arr(INDEX, 2048, 0),
        supervised_start=dict(anyOf=[INDEX, dict(type="null")]),
        supervised_stop=dict(anyOf=[dict(type="integer", minimum=1), dict(type="null")]),
        training_condition=TEXT,
        label_derivation=TEXT,
        deployment_condition_source=TEXT,
        condition_evidence_indices=arr(INDEX, 32),
        boundary_evidence_indices=arr(INDEX, 32),
        merge_check=TEXT,
        split_check=TEXT,
        uncertainty=TEXT,
        supervision_exclusions=arr(obj(dict(**LOCAL_RANGE, reason=TEXT)), 64, 0),
    )
)
EVIDENCE = obj(dict(trajectory_id=TEXT, indices=arr(INDEX, 32)))
HANDOFF = obj(
    {
        k: TEXT
        for k in (
            "entry_conditions",
            "exit_conditions",
            "overlap_role",
            "successor_readiness",
            "failure_signatures",
        )
    }
)
HEURISTIC = obj(
    dict(
        heuristic_id=ID,
        policy_id=ID,
        policy_contract=obj(
            dict(
                caller_arguments=TEXT,
                input_output_contract=TEXT,
                selection_cues=TEXT,
                status_and_progress=TEXT,
                memory_and_handoff=TEXT,
            )
        ),
        statement=TEXT,
        evidence=arr(EVIDENCE, 32),
        rationale=TEXT,
        applicability=TEXT,
        pipeline_implications=TEXT,
        goal_conditioning=TEXT,
        input_preprocessing=TEXT,
        augmentation=TEXT,
        action_decoding=TEXT,
        limitations=TEXT,
        handoff=HANDOFF,
    )
)
SKILL = obj(
    dict(
        skill_id=ID,
        name=TEXT,
        subgoal=TEXT,
        segmentation_rationale=TEXT,
        component_kind=dict(type="string", enum=["learned_policy", "supplied_executor"]),
        execution_contract=TEXT,
        segments=arr(SEGMENT),
        heuristics=arr(HEURISTIC, 32, 0),
    )
)
PLAN = obj(
    dict(
        task_interpretation=TEXT,
        goal_conditioning=TEXT,
        portfolio_rationale=TEXT,
        model_inventory=TEXT,
        generalization_design=TEXT,
        calling_contract=TEXT,
        capability_coverage=TEXT,
        sharing_and_diversity_audit=TEXT,
        skills=arr(SKILL, 64),
        exclusions=arr(obj(dict(**RANGE, reason=TEXT)), 512, 0),
        overlap_rationale=TEXT,
        motion_coverage=TEXT,
        unobserved_cases=TEXT,
    )
)


@dataclass(frozen=True)
class Limits:
    max_api_calls: int
    max_tool_calls: int
    max_plan_versions: int
    max_output_tokens: int


class CutTools:
    def __init__(self, cfg):
        self.cfg = cfg
        self.run = ROOT / cfg["run"]
        self.output = ROOT / cfg["output"]
        self.sources = Sources(cfg)
        self.source_manifest = read(self.run / "source_manifest.json")
        self.sources.verify_metadata(self.source_manifest)
        self.j = Journal(self.run / "_session")
        self.budget = Limits(**{k: cfg[k] for k in Limits.__dataclass_fields__})

    def schemas(self):
        def tool(name, desc, props):
            return dict(
                type="function",
                name=name,
                description=desc,
                strict=True,
                parameters=obj(props),
            )

        return [
            tool(
                "list_demonstrations",
                "Read authorized demonstrations and their actual observation/action contract.",
                {},
            ),
            tool(
                "read_metadata",
                "Read recorded per-episode calibration, TCP and teleoperation metadata.",
                dict(trajectory_id=TEXT),
            ),
            tool(
                "read_motion",
                "Numerical summaries over API-chosen bins; not semantic cuts. Does not mark boundary observations as inspected.",
                dict(**RANGE, bins=dict(type="integer", minimum=1, maximum=64)),
            ),
            tool(
                "read_steps",
                "Inspect paired robot states/actions at explicit original sample indices. Inspect first and last included row of every proposed cut.",
                dict(trajectory_id=TEXT, indices=arr(INDEX, 32)),
            ),
            tool(
                "read_images",
                "Read 1-12 original paired RGB frames. Use preview normally; original/crop for details. Crop coordinates are in original pixels; null means full frame.",
                dict(
                    trajectory_id=TEXT,
                    indices=arr(INDEX, 12),
                    camera=dict(type="string", enum=["third", "wrist"]),
                    resolution=dict(type="string", enum=["preview", "original"]),
                    crop=dict(
                        anyOf=[
                            dict(type="null"),
                            obj({k: INDEX for k in ("left", "top", "right", "bottom")}),
                        ]
                    ),
                ),
            ),
            tool(
                "write_plan",
                "Write an API-authored plan. skills are responsibility/data groups: learned_policy groups have chosen priors and dense or sparse supervision; supplied_executor groups have no heuristics and executor_only segments. Sparse segments list exact decision_indices; other kinds use an empty list. Executor-only supervision bounds are null. State and visual inspection requirements are in data_contract. Choose the learned policy count from remaining learning needs; no quota for alternatives. Describe executor requirements and integration in execution_contract. No code.",
                dict(plan=PLAN),
            ),
            tool("read_plan", "Read the current plan and its content hash.", {}),
            tool(
                "check_plan",
                "Validate range coverage, inspected boundary/evidence indices and original source identity.",
                {},
            ),
            tool(
                "submit_datasets",
                "Freeze the checked plan hash, publish exact NPZ cuts plus source-media indices and heuristic Markdown, and finish. No policy code/training.",
                dict(expected_hash=TEXT),
            ),
        ]

    def done(self):
        return bool(self.j.get("frozen"))

    def execute(self, item):
        name, cid = item["name"], item["call_id"]
        previous = self.j.db.execute(
            "SELECT * FROM tools WHERE id=?", (cid,)
        ).fetchone()
        if previous:
            if previous["name"] != name or previous["args"] != item["arguments"]:
                raise ValueError("Reused tool ID has different content")
            if previous["status"] == "completed":
                return json.loads(previous["result"])
            raise RuntimeError(
                "Interrupted tool: explicit reconciliation required, no automatic replay"
            )
        self.j.charge("tool_call", self.budget.max_tool_calls, name=name, call_id=cid)
        with self.j.db:
            self.j.db.execute(
                "INSERT INTO tools VALUES(?,?,?,?,?)",
                (cid, name, item["arguments"], "started", None),
            )
        started = time.monotonic()
        try:
            args = json.loads(item["arguments"])
            definition = {s["name"]: s for s in self.schemas()}[name]
            jsonschema.validate(args, definition["parameters"])
            result = self.dispatch(name, args)
        except (ValueError, KeyError, jsonschema.ValidationError) as error:
            result = dict(error=str(error)[:6000], repair_allowed=not self.done())
        with self.j.db:
            self.j.db.execute(
                "UPDATE tools SET status='completed',result=? WHERE id=?",
                (encode(result), cid),
            )
        self.j.event("tool_cost", name=name, elapsed_seconds=time.monotonic() - started)
        return result

    def plan(self):
        version = self.j.get("plan_version")
        if not version:
            raise ValueError("Write a plan first")
        plan = read(self.j.root / "plans" / (version + ".json"))
        if object_hash(plan) != version:
            raise ValueError("Immutable plan changed")
        return version, plan

    def validate(self, plan):
        jsonschema.validate(plan, PLAN)
        seen = set()
        segment_ids = {}
        heuristic_ids = set()
        policy_ids = set()
        counts = {
            tid: np.zeros(ep["n"], dtype=np.int64)
            for tid, ep in self.sources.episodes.items()
        }
        supervised = {tid: np.zeros_like(v) for tid, v in counts.items()}
        executor = {tid: np.zeros_like(v) for tid, v in counts.items()}
        inspected = {
            tid: set(v) for tid, v in self.j.get("inspected_steps", {}).items()
        }
        images = self.j.get("inspected_images", {})
        for tid in counts:
            if not all(images.get(tid, {}).get(cam) for cam in ("third", "wrist")):
                raise ValueError(
                    "Inspect both camera views from every trajectory: " + tid
                )
        for skill in plan["skills"]:
            if skill["skill_id"] in seen:
                raise ValueError("Duplicate skill_id")
            seen.add(skill["skill_id"])
            learned = skill["component_kind"] == "learned_policy"
            if learned != bool(skill["heuristics"]):
                raise ValueError("Learned groups need a prior; supplied executor groups have no learned heuristics")
            ranges = set()
            for s in skill["segments"]:
                self.sources.interval(s)
                if s["segment_id"] in segment_ids and segment_ids[s["segment_id"]] != s:
                    raise ValueError(
                        "Reused segment_id must retain the same semantic contract"
                    )
                segment_ids[s["segment_id"]] = s
                tid = s["trajectory_id"]
                lo = s["start"]
                hi = s["stop"]
                slo = s["supervised_start"]
                shi = s["supervised_stop"]
                kind = s["supervision_kind"]
                decisions = s["decision_indices"]
                if learned != (kind != "executor_only"):
                    raise ValueError("Segment supervision kind must match group component kind")
                if kind == "executor_only":
                    if slo is not None or shi is not None or decisions or s["supervision_exclusions"]:
                        raise ValueError("Executor-only segments have null supervision bounds and no decision indices or supervision exclusions")
                    anchors = {lo, hi - 1}
                else:
                    if slo is None or shi is None or not lo <= slo < shi <= hi:
                        raise ValueError("Supervised range must be inside its materialized segment")
                    anchors = {slo, shi - 1}
                    if kind == "sparse":
                        if not decisions or len(decisions) != len(set(decisions)) or any(i < slo or i >= shi for i in decisions):
                            raise ValueError("Sparse decision indices must be nonempty, unique and inside the supervised range")
                        if not set(decisions).issubset(inspected.get(tid, set())):
                            raise ValueError("Read_steps must inspect sparse decision indices")
                    elif decisions:
                        raise ValueError("Dense supervision uses an empty decision_indices list")
                if (tid, lo, hi) in ranges:
                    raise ValueError("Duplicate identical segment within one dataset")
                ranges.add((tid, lo, hi))
                if not {lo, hi - 1}.issubset(inspected.get(tid, set())):
                    raise ValueError(
                        f"Read_steps must inspect first/last included samples: {tid} {lo} {hi - 1}"
                    )
                evidence = set(s["condition_evidence_indices"])
                boundaries = set(s["boundary_evidence_indices"])
                if not evidence or any(i < lo or i >= hi for i in evidence):
                    raise ValueError(
                        "Condition evidence must be inside its materialized segment"
                    )
                if not anchors.issubset(boundaries):
                    raise ValueError(
                        "Boundary evidence must include supervision bounds, or materialized bounds for executor-only segments"
                    )
                if not (evidence | boundaries | anchors).issubset(
                    inspected.get(tid, set())
                ):
                    raise ValueError(
                        "Read_steps must inspect all condition and boundary evidence"
                    )
                visual = set().union(*(set(v) for v in images.get(tid, {}).values()))
                if not anchors.issubset(visual):
                    raise ValueError(
                        "Read_images must inspect supervision bounds, or materialized bounds for executor-only segments"
                    )
                eligible = np.zeros_like(counts[tid], dtype=bool)
                if kind == "dense":
                    eligible[slo:shi] = True
                elif kind == "sparse":
                    eligible[decisions] = True
                else:
                    executor[tid][lo:hi] += 1
                masked = np.zeros_like(eligible)
                for exclusion in s["supervision_exclusions"]:
                    a, b = exclusion["start"], exclusion["stop"]
                    if not slo <= a < b <= shi or np.any(masked[a:b]):
                        raise ValueError(
                            "Supervision exclusions must be disjoint inside the supervised range"
                        )
                    masked[a:b] = True
                if kind == "sparse" and np.any(eligible & masked):
                    raise ValueError("Sparse decision indices cannot be excluded from supervision")
                eligible[masked] = False
                if learned and not eligible.any():
                    raise ValueError(
                        "A training segment needs at least one supervision-eligible row"
                    )
                supervised[tid] += eligible
                counts[tid][lo:hi] += 1
            for h in skill["heuristics"]:
                if h["heuristic_id"] in heuristic_ids or h["policy_id"] in policy_ids:
                    raise ValueError(
                        "Every heuristic and proposed independent policy needs a unique ID"
                    )
                heuristic_ids.add(h["heuristic_id"])
                policy_ids.add(h["policy_id"])
                for e in h["evidence"]:
                    tid = e["trajectory_id"]
                    for i in e["indices"]:
                        if i not in inspected.get(tid, set()):
                            raise ValueError(
                                f"Heuristic cites an uninspected state: {tid}/{i}"
                            )
                        if not any(t == tid and lo <= i < hi for t, lo, hi in ranges):
                            raise ValueError(
                                "Heuristic evidence must be in this dataset"
                            )
        excluded = {tid: np.zeros_like(v, dtype=bool) for tid, v in counts.items()}
        for s in plan["exclusions"]:
            self.sources.interval(s)
            tid = s["trajectory_id"]
            lo = s["start"]
            hi = s["stop"]
            if np.any(counts[tid][lo:hi]) or np.any(excluded[tid][lo:hi]):
                raise ValueError(
                    "Exclusion overlaps an assignment or another exclusion"
                )
            excluded[tid][lo:hi] = True
        stats = {}
        for tid, c in counts.items():
            missing = np.flatnonzero((c == 0) & ~excluded[tid])
            if len(missing):
                raise ValueError(
                    f"Unassigned source samples need explicit exclusions: {tid}; first={missing[:16].tolist()}, total={len(missing)}"
                )
            stats[tid] = dict(
                source_records=len(c),
                assigned_unique=int(np.sum(c > 0)),
                excluded=int(excluded[tid].sum()),
                reused_unique=int(np.sum(c > 1)),
                assignment_records=int(c.sum()),
                supervised_unique=int(np.sum(supervised[tid] > 0)),
                context_only_unique=int(np.sum((c > 0) & (supervised[tid] == 0))),
                supervision_assignment_records=int(supervised[tid].sum()),
                executor_unique=int(np.sum(executor[tid] > 0)),
                executor_assignment_records=int(executor[tid].sum()),
                context_without_executor_or_supervision_unique=int(np.sum((c > 0) & (supervised[tid] == 0) & (executor[tid] == 0))),
            )
        self.sources.verify_metadata(self.source_manifest)
        return dict(
            passed=True,
            datasets=len(plan["skills"]),
            learned_groups=sum(s["component_kind"] == "learned_policy" for s in plan["skills"]),
            supplied_executor_groups=sum(s["component_kind"] == "supplied_executor" for s in plan["skills"]),
            heuristics=sum(len(s["heuristics"]) for s in plan["skills"]),
            proposed_policies=len(policy_ids),
            segments=sum(len(s["segments"]) for s in plan["skills"]),
            coverage=stats,
            boundary_convention="[start,stop) paired records; checked start and stop-1",
            empirical_policy_performance="not implemented or trained",
            semantic_validation="Explicit condition/supervision contracts and evidence references checked structurally; semantic correctness requires review of API decisions.",
        )

    def dispatch(self, name, args):
        if self.done() and name in ("write_plan", "submit_datasets"):
            raise ValueError("Submission is immutable")
        if name == "list_demonstrations":
            return dict(demonstrations=self.sources.catalog(), contract=CONTRACT)
        if name == "read_metadata":
            return self.sources.episode(args["trajectory_id"])["meta"]
        if name == "read_motion":
            return self.sources.motion(
                args["trajectory_id"], args["start"], args["stop"], args["bins"]
            )
        if name == "read_steps":
            result = self.sources.steps(args["trajectory_id"], args["indices"])
            inspected = self.j.get("inspected_steps", {})
            tid = args["trajectory_id"]
            inspected[tid] = sorted(set(inspected.get(tid, [])) | set(args["indices"]))
            self.j.set("inspected_steps", inspected)
            return result
        if name == "read_images":
            result = []
            tid = args["trajectory_id"]
            cam = args["camera"]
            for i in args["indices"]:
                blob, meta = self.sources.image(
                    tid, cam, i, args["resolution"], args["crop"]
                )
                rel = str(
                    self.sources.media_path(tid, cam, i).relative_to(self.sources.base)
                )
                if (
                    meta["original_sha256"]
                    != self.source_manifest["media"][rel]["sha256"]
                ):
                    raise ValueError("Image changed after source freeze")
                preview_hash = __import__("hashlib").sha256(blob).hexdigest()
                path = self.j.root / "evidence" / (preview_hash + ".jpg")
                path.parent.mkdir(exist_ok=True)
                if not path.exists():
                    path.write_bytes(blob)
                meta["evidence_sha256"] = preview_hash
                self.j.event("image_evidence", **meta)
                result.extend(
                    [
                        dict(type="input_text", text=encode(meta)),
                        dict(
                            type="input_image",
                            detail="high",
                            image_url="data:image/jpeg;base64,"
                            + base64.b64encode(blob).decode(),
                        ),
                    ]
                )
            inspected = self.j.get("inspected_images", {})
            inspected.setdefault(tid, {})
            inspected[tid][cam] = sorted(
                set(inspected[tid].get(cam, [])) | set(args["indices"])
            )
            self.j.set("inspected_images", inspected)
            return result
        if name == "write_plan":
            self.j.charge("plan_version", self.budget.max_plan_versions)
            plan = args["plan"]
            jsonschema.validate(plan, PLAN)
            version = object_hash(plan)
            atomic(self.j.root / "plans" / (version + ".json"), plan)
            self.j.set("plan_version", version)
            self.j.event("plan_written", plan_hash=version)
            return dict(plan_hash=version)
        if name == "read_plan":
            version, plan = self.plan()
            return dict(plan_hash=version, plan=plan)
        if name == "check_plan":
            version, plan = self.plan()
            checks = self.validate(plan)
            self.j.set("checked_plan", version)
            return dict(plan_hash=version, **checks)
        if name == "submit_datasets":
            version, plan = self.plan()
            if (
                args["expected_hash"] != version
                or self.j.get("checked_plan") != version
            ):
                raise ValueError("Submit the exact successfully checked hash")
            checks = self.validate(plan)
            self.sources.verify_media(self.source_manifest)
            manifest = self.publish(version, plan, checks)
            self.j.set("frozen", manifest)
            return dict(
                submitted=True,
                plan_hash=version,
                checks=checks,
                manifest=str(self.output / "manifest.json"),
                policy_code_written=False,
                training_runs=0,
            )
        raise ValueError("Unknown tool")

    @staticmethod
    def markdown(skill):
        lines = [
            "# " + skill["name"],
            "",
            skill["subgoal"],
            "",
            "## Dataset rationale",
            "",
            skill["segmentation_rationale"],
            "",
            "## Component responsibility: " + skill["component_kind"],
            "",
            skill["execution_contract"],
            "",
        ]
        for s in skill["segments"]:
            lines += [
                f"### {s['segment_id']}",
                "",
                f"Source: `{s['trajectory_id']}` [{s['start']}, {s['stop']}). Supervision kind: {s['supervision_kind']}; bounds: {s['supervised_start']}, {s['supervised_stop']}; decision indices: {s['decision_indices']}.",
                "",
            ]
            for key, value in s.items():
                if key in {
                    "trajectory_id",
                    "start",
                    "stop",
                    "segment_id",
                    "supervised_start",
                    "supervised_stop",
                }:
                    continue
                rendered = (
                    value
                    if isinstance(value, str)
                    else json.dumps(value, ensure_ascii=False)
                )
                lines += [f"**{key.replace('_', ' ').capitalize()}**", "", rendered, ""]
        for i, h in enumerate(skill["heuristics"], 1):
            lines += ["", f"## Heuristic {i}", "", h["statement"], ""]
            for key, value in h.items():
                if key == "statement":
                    continue
                lines += ["### " + key.replace("_", " ").capitalize(), ""]
                if key == "evidence":
                    lines += [
                        f"- `{e['trajectory_id']}`: {e['indices']}" for e in value
                    ]
                elif isinstance(value, dict):
                    for k, v in value.items():
                        lines += [f"**{k}**", "", v, ""]
                else:
                    lines += [value, ""]
        lines += [
            "",
            "All scientific text above is unchanged Runtime API output. Rendering and slicing are developer-owned. No policy code or training exists in this stage.",
            "",
        ]
        return "\n".join(lines)

    def publish(self, version, plan, checks):
        if self.output.exists():
            raise ValueError(
                "Cut output already exists; do not overwrite a partial or frozen publication"
            )
        self.output.mkdir(parents=True)
        atomic(self.output / "plan.json", plan)
        catalog = [
            dict(
                policy_id=h["policy_id"],
                heuristic_id=h["heuristic_id"],
                dataset_group_id=g["skill_id"],
                mechanism=h["statement"],
                contract=h["policy_contract"],
                applicability=h["applicability"],
                handoff=h["handoff"],
                limitations=h["limitations"],
            )
            for g in plan["skills"]
            for h in g["heuristics"]
        ]
        atomic(self.output / "policy_catalog.json", catalog)
        execution_catalog = [
            dict(
                group_id=g["skill_id"], component_kind=g["component_kind"],
                purpose=g["subgoal"], execution_contract=g["execution_contract"],
                policy_ids=[h["policy_id"] for h in g["heuristics"]],
            ) for g in plan["skills"]
        ]
        atomic(self.output / "execution_catalog.json", execution_catalog)
        execution_lines = ["# System execution responsibilities (design only)", "", plan["calling_contract"], ""]
        for item in execution_catalog:
            execution_lines += ["## " + item["group_id"], "", item["component_kind"], "", item["purpose"], "", item["execution_contract"], ""]
        (self.output / "EXECUTION_CATALOG.md").write_text("\n".join(execution_lines))
        catalog_lines = [
            "# Proposed callable policies (not implemented or trained)",
            "",
        ]
        for item in catalog:
            catalog_lines += [
                f"## {item['policy_id']}",
                "",
                f"Dataset: `{item['dataset_group_id']}`; heuristic: `{item['heuristic_id']}`.",
                "",
                item["mechanism"],
                "",
            ]
            for key, value in item["contract"].items():
                catalog_lines += [
                    f"### {key.replace('_', ' ').capitalize()}",
                    "",
                    value,
                    "",
                ]
            catalog_lines += [
                "See the dataset heuristic document for applicability, limitations, and handoffs.",
                "",
            ]
        (self.output / "POLICY_CATALOG.md").write_text("\n".join(catalog_lines))
        datasets = []
        for skill in plan["skills"]:
            root = self.output / "datasets" / skill["skill_id"]
            root.mkdir(parents=True)
            segments = []
            for i, s in enumerate(skill["segments"]):
                folder = root / "segments" / f"{i:04d}"
                segments.append(
                    self.sources.publish_segment(s, folder, self.source_manifest)
                )
                atomic(folder / "supervision.json", s)
            dataset = dict(
                schema="real_robot.cut_dataset.v3",
                plan_hash=version,
                skill_id=skill["skill_id"],
                name=skill["name"],
                subgoal=skill["subgoal"],
                component_kind=skill["component_kind"],
                execution_contract=skill["execution_contract"],
                segments=segments,
                heuristics=skill["heuristics"],
                source_manifest=str(self.run / "source_manifest.json"),
                source_manifest_sha256=digest(self.run / "source_manifest.json"),
                sequence_semantics="Bag of independent original sequences; overlapping reuse allowed; never concatenate across cuts.",
            )
            atomic(root / "dataset.json", dataset)
            (root / "heuristic.md").write_text(self.markdown(skill))
            datasets.append(
                dict(
                    skill_id=skill["skill_id"],
                    dataset=str((root / "dataset.json").relative_to(self.output)),
                    segments=len(segments),
                    heuristics=len(skill["heuristics"]),
                )
            )
        files = {
            str(p.relative_to(self.output)): digest(p)
            for p in sorted(self.output.rglob("*"))
            if p.is_file()
        }
        manifest = dict(
            schema="real_robot.cut_manifest.v2",
            owner="Runtime API semantic plan; developer materialization",
            plan_hash=version,
            datasets=datasets,
            checks=checks,
            files=files,
            source_manifest=str(self.run / "source_manifest.json"),
            source_manifest_sha256=digest(self.run / "source_manifest.json"),
            model=self.cfg["model"],
            reasoning_effort=self.cfg["reasoning_effort"],
            policy_code_written=False,
            training_runs=0,
        )
        atomic(self.output / "manifest.json", manifest)
        return manifest
