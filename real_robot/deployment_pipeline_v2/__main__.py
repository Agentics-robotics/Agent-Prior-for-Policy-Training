"""Verify exported artifacts and load them on the receiving computer."""
import argparse
import json
import time

from appl.io import read
from .common import BUNDLES, locations, verify
from .policy import Policy


def run_example(bundle, gpu):
    package, weights, assets = locations(bundle)
    manifest = verify(package, weights, assets, require_assets=True)
    example = read(package / manifest["examples"]["interface"]["file"])
    fixture = example["fixture"]
    start = time.monotonic()
    with Policy(bundle, gpu=gpu, timeout_s=180) as policy:
        loaded = time.monotonic()
        result = policy.act(fixture["observation"], fixture["call"], reset=True,
                            executor_contract=fixture["executor_contract"])
        returned = time.monotonic()
    if result["status"] != example["expected_status"]:
        raise RuntimeError("Offline example returned an unexpected status")
    return dict(bundle=bundle, example_scope=example["scope"],
                uses_trained_checkpoint=True, hardware_io=False, result=result,
                observed_load_seconds=loaded-start, observed_geometry_call_seconds=returned-loaded,
                timing_scope="One offline example, includes IPC; excludes raw RGB perception and physical execution")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command",choices=["check", "example"])
    p.add_argument("--bundle",choices=[*BUNDLES,"all"],default="all")
    p.add_argument("--gpu",type=int,default=0)
    p.add_argument("--files-only",action="store_true")
    p.add_argument("--require-assets",action="store_true")
    args = p.parse_args()
    for bundle in BUNDLES if args.bundle == "all" else [args.bundle]:
        if args.command == "example":
            if args.files_only:
                p.error("example loads the trained model; --files-only applies to check")
            print(json.dumps(run_example(bundle, args.gpu), allow_nan=False), flush=True)
            continue
        manifest = verify(*locations(bundle),require_assets=args.require_assets)
        if not args.files_only:
            for pid in manifest["policies"]:
                with Policy(bundle,pid,gpu=args.gpu) as policy:
                    print(json.dumps(policy.ready),flush=True)
        print(json.dumps(dict(bundle=bundle,status="passed",GPU_loaded=not args.files_only,
            hardware_io=False,inventory_evaluated=False)),flush=True)


if __name__ == "__main__":
    main()
