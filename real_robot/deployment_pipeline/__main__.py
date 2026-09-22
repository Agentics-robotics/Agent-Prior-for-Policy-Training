"""Verify exported artifacts and load them on the receiving computer."""
import argparse
import json

from .common import BUNDLES, locations, verify
from .policy import Policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command",choices=["check"])
    p.add_argument("--bundle",choices=[*BUNDLES,"all"],default="all")
    p.add_argument("--gpu",type=int,default=0)
    p.add_argument("--files-only",action="store_true")
    p.add_argument("--require-assets",action="store_true")
    args = p.parse_args()
    for bundle in BUNDLES if args.bundle == "all" else [args.bundle]:
        manifest = verify(*locations(bundle),require_assets=args.require_assets)
        if not args.files_only:
            for pid in manifest["policies"]:
                with Policy(bundle,pid,gpu=args.gpu) as policy:
                    print(json.dumps(policy.ready),flush=True)
        print(json.dumps(dict(bundle=bundle,status="passed",GPU_loaded=not args.files_only,
            hardware_io=False,inventory_evaluated=False)),flush=True)


if __name__ == "__main__":
    main()
