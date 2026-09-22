"""Isolated selected-weight inference with the exact API-authored functions."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

from .common import ROOT, verify, wire, unwire


def serve(args):
    import torch
    import cv2
    import jsonschema
    from real_robot.training_pipeline import public
    from real_robot.training_pipeline.security import lockdown

    package, weights, assets, output = map(Path, (args.package,args.weights,args.assets,args.output))
    manifest = verify(package,weights,assets,args.policy)
    item = manifest["policies"][args.policy]
    checkpoint = weights / item["file"]
    saved = torch.load(checkpoint,map_location="cpu",weights_only=True)
    if (saved["schema"] != "real_robot.pipeline_inference_checkpoint.v2" or
        saved["policy_id"] != args.policy or saved["source_hashes"] != manifest["source_hashes"]):
        raise ValueError("Exported checkpoint identity mismatch")
    spec = saved["spec"]
    assert spec["policy_id"] == args.policy
    spec["device"] = "cuda:0"
    public.configure(None,[],output,assets,package / "source")
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    device = lockdown(package / "source",output,
        [checkpoint,assets,ROOT / "real_robot/deployment_pipeline_v2",ROOT / "real_robot/deployment"],gpu=True)
    definition = importlib.util.spec_from_file_location("api_robot_policy",package / "source/policy.py")
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    model = module.build_model(args.policy,spec).to("cuda:0")
    model.load_state_dict(saved["state_dict"],strict=True)
    model.eval().requires_grad_(False)
    memory = None
    print(json.dumps(dict(status="ready",policy_id=args.policy,device=device,
        checkpoint_sha256=item["sha256"],hardware_io=False)),flush=True)
    for line in sys.stdin:
        request = unwire(json.loads(line))
        if request["operation"] == "inventory":
            if not callable(getattr(module,"inventory",None)):
                raise ValueError("This API package has no inventory operation")
            # No implicit download or replacement perception backend.
            for filename in manifest["assets"]:
                if not (assets / filename).is_file():
                    raise FileNotFoundError("Download optional inventory asset: " + str(assets / filename))
            with torch.no_grad():
                answer = module.inventory(request["observation"],spec)
            answer["scene_version"] = request["scene_version"]
        elif request["operation"] == "act":
            if request.get("reset"):
                memory = None
            with torch.no_grad():
                result = module.act(model,request["observation"],request["call"],memory,spec)
            memory = result["memory"]
            answer = {k:v for k,v in result.items() if k != "memory"}
            if result["decision"] is not None:
                jsonschema.validate(result["decision"],spec["policy_config"]["decision_schema"])
                if request.get("executor_contract") is not None:
                    answer["executor_request"] = module.executor_request(result["decision"],request["executor_contract"],spec)
        else:
            raise ValueError("Unknown operation")
        answer["hardware_io"] = False
        print(json.dumps(wire(answer),allow_nan=False),flush=True)


def main():
    p = argparse.ArgumentParser()
    for name in ("policy","package","weights","assets","output"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--gpu",type=int,required=True)
    p.add_argument("--isolated",action="store_true")
    args = p.parse_args()
    if not args.isolated:
        from .gpu import launch
        return launch(args.gpu,sys.argv[1:])
    nodes = {p.name for p in Path("/dev").glob("nvidia*") if p.name.removeprefix("nvidia").isdigit()}
    if nodes != {"nvidia"+os.environ["APPL_GPU_MINOR"]}:
        raise RuntimeError("GPU isolation mismatch")
    return serve(args)


if __name__ == "__main__":
    main()
