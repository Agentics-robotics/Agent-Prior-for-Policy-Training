"""Deterministic coverage/numerical audit; no model selection or performance feedback."""
import argparse
import json
import numpy as np
from appl.io import ROOT, atomic, read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", type=int, default=0)
    args = parser.parse_args()
    cfg = read(ROOT / "real_robot/configs/push_training_v3.json")
    run = ROOT / cfg["run"]
    cache = run / f"package_{args.revision:02d}/prepared"
    manifest = read(cache / "result.json")
    metadata = read(cache / "metadata.json")
    arrays = {name: np.load(cache / entry["file"], mmap_mode="r", allow_pickle=False)
              for name, entry in manifest["arrays"].items()}
    nonfinite = {}
    for name, array in arrays.items():
        if array.dtype.kind not in "fc":
            continue
        nonfinite[name] = sum(int(np.count_nonzero(~np.isfinite(array[i:i + 256])))
                              for i in range(0, len(array), 256))
    direct = [name for name in arrays if name.startswith("f_")]
    assert not any(nonfinite.get(name, 0) for name in direct), nonfinite
    rows = []
    for ordinal, policy_id in enumerate(cfg["policies"]):
        eligible = arrays["policy_weight"][:, ordinal] > 0
        feature = {name: int(np.count_nonzero(arrays[name][eligible] > 0))
                   for name in ("goal_valid", "geometry_valid", "response_valid", "command_dq_valid")
                   if name in arrays}
        rows.append(dict(policy_id=policy_id, dataset_group=cfg["policy_datasets"][policy_id],
                         action_rows=int(eligible.sum()), API_valid_feature_rows=feature))
    result = dict(revision=args.revision, passed_numerical_input_check=True,
                  total_anchor_rows=manifest["num_examples"], policies=rows,
                  nonfinite_elements=nonfinite, API_segment_coverage=metadata["coverage"],
                  warning_log=str(cache / "stderr.log"),
                  interpretation="Object validity flags are API weak-label rules, not verified ground truth; not applicable to tool waypoint goals.",
                  optimizer_updates=0, performance_evaluation=False)
    atomic(run / f"preparation_review_{args.revision:02d}.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "API_segment_coverage"}))


if __name__ == "__main__":
    main()
