"""Check downloadable source/weights and load the two local inference workers."""

import argparse
import json

from .common import PACKAGE, WEIGHTS, verify
from .policy import PushPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["check"])
    parser.add_argument("--weights", default=str(WEIGHTS))
    parser.add_argument("--package", default=str(PACKAGE))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--files-only", action="store_true")
    args = parser.parse_args()
    manifest = verify(args.package, args.weights)
    print(
        json.dumps(
            dict(source_and_weights_verified=True, policies=list(manifest["policies"]))
        ),
        flush=True,
    )
    if not args.files_only:
        for name in manifest["policies"]:
            with PushPolicy(
                name, gpu=args.gpu, package=args.package, weights=args.weights
            ) as model:
                print(json.dumps(model.ready), flush=True)
    print(
        json.dumps(
            dict(
                status="passed",
                GPU_loaded=not args.files_only,
                observations_evaluated=0,
                hardware_io=False,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
