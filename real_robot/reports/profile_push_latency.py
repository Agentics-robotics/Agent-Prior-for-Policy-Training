"""Latency-only diagnostics of unchanged policies; no robot IO or performance scoring."""

import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from real_robot.deployment.common import ROOT, PACKAGE, WEIGHTS, read, verify, wire


def inputs(start, count):
    from real_robot.data import Sources

    cfg = read(ROOT / "real_robot/configs/push_cut_v3.json")
    cfg["trajectory_ids"] = cfg["trajectory_ids"][:1]
    source = Sources(cfg)
    tid = cfg["trajectory_ids"][0]
    ep = source.episode(tid)
    keys = (
        "q",
        "dq",
        "tau_ext",
        "T_base_ee",
        "T_base_flange",
        "gripper_position",
        "gripper_width_m",
        "t",
        "recv_time",
        "img_t_wrist",
        "img_t_third",
        "img_age_wrist",
        "img_age_third",
        "sample_index",
    )
    result = []
    for index in range(start, start + count):
        obs = {key: ep["arrays"][key][index] for key in keys if key in ep["arrays"]}
        obs["metadata"] = ep["meta"]
        for camera in ("third", "wrist"):
            with Image.open(source.media_path(tid, camera, index)) as im:
                obs[camera + "_rgb"] = np.array(im.convert("RGB"))
        result.append(obs)
    return result


def call_for(inventory, name):
    chosen = inventory["instances"][0]
    u, v = chosen["center_uv"]
    # Synthetic local goal for timing only, as permitted by the API's example.
    # This is neither a labeled manipulation task nor a hardware request.
    return dict(
        call_id="latency-probe",
        scene_version="latency-probe",
        policy_id=name,
        instance_id=chosen["instance_id"],
        template=chosen["template"],
        spatial_goal=dict(
            center_uv=[u + 20, v], angle_deg=0, accept_similarity_approximation=True
        ),
        workspace_polygon=[[0, 0], [1279, 0], [1279, 719], [0, 719]],
        tolerance_px=8.0,
        tolerance_deg=8.0,
        max_duration_s=30.0,
    )


def summary(values):
    return dict(
        n=len(values),
        mean_ms=float(np.mean(values) * 1000),
        p50_ms=float(np.median(values) * 1000),
        p95_ms=float(np.percentile(values, 95) * 1000),
    )


def ipc(args, observations):
    from real_robot.deployment import PushPolicy

    output = {}
    for name in ("contour_push", "visual_push"):
        began = time.perf_counter()
        with PushPolicy(name, gpu=args.gpu) as policy:
            startup = time.perf_counter() - began
            begin = time.perf_counter()
            inventory = policy.inventory(observations[0], "latency-probe")
            inventory_time = time.perf_counter() - begin
            call = call_for(inventory, name)
            request = dict(
                operation="act",
                observation=observations[0],
                call=call,
                reset=False,
                controller_contract=None,
            )
            begin = time.perf_counter()
            payload = json.dumps(wire(request), allow_nan=False)
            encode_time = time.perf_counter() - begin
            times, states = [], []
            for obs in observations:
                began = time.perf_counter()
                result = policy.act(obs, call)
                times.append(time.perf_counter() - began)
                states.append(result["status"])
            output[name] = dict(
                startup_s=startup,
                inventory_ms=inventory_time * 1000,
                inventory_instances=len(inventory["instances"]),
                request_bytes=len(payload.encode()),
                host_encode_ms=encode_time * 1000,
                first_act_ms=times[0] * 1000,
                steady=summary(times[3:]),
                statuses=states,
                per_call_ms=[x * 1000 for x in times],
                worker_logs=str(policy.output),
            )
    return output


def stages(args, observations):
    import torch
    import cv2
    from real_robot.policy_training import public
    from real_robot.policy_training.security import lockdown

    manifest = verify()
    saved = {
        name: torch.load(WEIGHTS / item["file"], map_location="cpu", weights_only=True)
        for name, item in manifest["policies"].items()
    }
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    public.configure(None, [], output, output / "no-assets")
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    device = lockdown(
        PACKAGE / "source", output, [ROOT / "real_robot/reports"], gpu=True
    )
    module_spec = importlib.util.spec_from_file_location(
        "api_robot_policy", PACKAGE / "source/policy.py"
    )
    api = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(api)
    import invocation
    import perception

    measurements = defaultdict(list)

    def wrap(original, label, cuda=False):
        def measured(*a, **kw):
            if cuda:
                torch.cuda.synchronize()
            begin = time.perf_counter()
            result = original(*a, **kw)
            if cuda:
                torch.cuda.synchronize()
            measurements[label].append(time.perf_counter() - begin)
            return result

        return measured

    # Transparent timing wrappers; source files and function results are unchanged.
    for name in (
        "convert_observation",
        "selected_features",
        "registration",
        "history_batch",
    ):
        setattr(
            invocation,
            name,
            wrap(getattr(invocation, name), name, name == "history_batch"),
        )
    for name in ("detect", "update_tracks", "state_vector", "make_graph"):
        setattr(perception, name, wrap(getattr(perception, name), name))
    result = dict(device=device, policies={})
    for name, checkpoint in saved.items():
        spec = checkpoint["spec"]
        spec["device"] = "cuda:0"
        model = api.build_model(name, spec).to("cuda:0")
        model.load_state_dict(checkpoint["state_dict"])
        model.eval().requires_grad_(False)
        model.forward = wrap(model.forward, "network_forward", True)
        inventory = api.inventory(observations[0], "latency-probe")
        call = call_for(inventory, name)
        memory = None
        totals, states = [], []
        for index, obs in enumerate(observations):
            if index == 3:
                measurements.clear()
            torch.cuda.synchronize()
            began = time.perf_counter()
            answer = api.act(model, obs, call, memory, spec)
            torch.cuda.synchronize()
            totals.append(time.perf_counter() - began)
            states.append(answer["status"])
            memory = answer["memory"]
        result["policies"][name] = dict(
            steady_direct_act=summary(totals[3:]),
            stages={k: summary(v) for k, v in measurements.items()},
            statuses=states,
        )
        del model
        measurements.clear()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["ipc", "stages"])
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--start", type=int, default=300)
    parser.add_argument("--count", type=int, default=18)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device-isolated", action="store_true")
    args = parser.parse_args()
    if args.mode == "stages" and not args.device_isolated:
        import sys
        from appl.gpu import launch

        return launch(
            args.gpu, sys.argv[1:], module="real_robot.reports.profile_push_latency"
        )
    observations = inputs(args.start, args.count)
    value = (
        ipc(args, observations) if args.mode == "ipc" else stages(args, observations)
    )
    result = dict(
        mode=args.mode,
        gpu=args.gpu,
        sample_indices=[args.start, args.start + args.count],
        warmup_calls=3,
        robot_io=False,
        API_calls=0,
        training_updates=0,
        note="Latency only, one recorded scene and synthetic local goal; not target-machine performance or skill evaluation.",
        result=value,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
