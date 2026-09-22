"""Read-only deterministic sampler accounting; never imports API scientific code."""
import ast

import numpy as np
import torch

from appl.io import ROOT, atomic, digest, read


EXPECTED_SAMPLER = '''
def sample_indices(policy_id,arrays,metadata,rng,step,spec):
    K=len(arrays['point']);a=(step-1)*16
    if a<K:
        return np.arange(a,min(a+16,K),dtype=np.int64)
    w=arrays['policy_weight'][:,0].astype(np.float64);w/=w.sum()
    return rng.choice(K,size=min(16,K),replace=True,p=w).astype(np.int64)
'''


def main():
    root = ROOT / "real_robot/runs/push_letters/training_v6"
    state = read(root / "workflow.json")
    assert state["status"] == "complete"
    package = root / f"package_{state['revision']:02d}"
    source = package / "source/policy.py"
    tree = ast.parse(source.read_text())
    sampler = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sample_indices")
    expected = ast.parse(EXPECTED_SAMPLER).body[0]
    assert ast.dump(sampler) == ast.dump(expected), "Sampler changed: audit must be reviewed, never guessed"
    manifest = read(package / "prepared/result.json")
    metadata = read(package / "prepared/metadata.json")
    params = read(package / "source/package.json")["policies"]["shape_push_v1"]
    result = read(package / "training/shape_push_v1/result.json")
    ckpt = package / "training/shape_push_v1/last.pt"
    assert digest(ckpt) == result["checkpoint_sha256"]
    saved = torch.load(ckpt, map_location="cpu", weights_only=True)

    def array(name):
        file = package / "prepared" / manifest["arrays"][name]["file"]
        assert digest(file) == manifest["arrays"][name]["sha256"]
        return np.load(file, allow_pickle=False)

    weights = array("policy_weight")[:, 0].astype(np.float64)
    weights /= weights.sum()
    n = len(weights)
    counts = np.zeros(n, dtype=np.int64)
    rng = np.random.default_rng(params["seed"])
    for step in range(1, result["optimizer_steps"] + 1):
        a = (step - 1) * 16
        ids = np.arange(a, min(a + 16, n)) if a < n else rng.choice(n, size=min(16, n), replace=True, p=weights)
        np.add.at(counts, ids, 1)
    assert rng.bit_generator.state == saved["numpy_sampler_state"]
    assert int(counts.sum()) == result["sample_exposures"]
    assert np.all(counts > 0)
    episodes, segments = array("example_episode"), array("example_segment")
    targets = array("target_validity") > .5
    episode_counts = {str(int(e)): int(np.sum((episodes == e) & (counts > 0))) for e in np.unique(episodes)}
    segment_counts = {str(int(s)): int(np.sum((segments == s) & (counts > 0))) for s in np.unique(segments)}
    states = saved["selected"]
    frozen_sam = [k for k in states if k.startswith("sam.")]
    trainable_tensors = {k: v for k, v in states.items() if not k.startswith("sam.")}
    assert sum(v.numel() for v in trainable_tensors.values()) == result["trainable_parameters"]
    audit = dict(date="2026-09-22", passed=True,
        method="Independent replay of AST-verified API sampler; exact saved RNG-state and exposure-count match",
        checkpoint_sha256=digest(ckpt), source_sha256=digest(source),
        unique_gradient_examples=int(np.sum(counts > 0)), per_episode=episode_counts,
        per_segment=segment_counts, total_gradient_exposures=int(counts.sum()),
        per_example_min_exposures=int(counts.min()), per_example_max_exposures=int(counts.max()),
        initial_full_sweep=True, all_valid_examples_used=True,
        channel_valid_counts=dict(contact=int(targets[:, 0].sum()), direction=int(targets[:, 1].sum()), length=int(targets[:, 2].sum())),
        contact_only_rows=int(np.sum(targets[:, 0] & ~targets[:, 1])),
        contact_direction_only_rows=int(np.sum(targets[:, 1] & ~targets[:, 2])),
        full_target_rows=int(np.all(targets, axis=1).sum()),
        API_data_audit=metadata["data_audit"], API_segment_coverage=metadata["coverage"]["segments"],
        trainable_parameters=result["trainable_parameters"], frozen_SAM_state_tensor_count=len(frozen_sam),
        frozen_SAM_state_elements=sum(states[k].numel() for k in frozen_sam),
        independent_performance_evaluation=False, robot_execution=False)
    atomic(root / "gradient_coverage_audit.json", audit)
    print(audit, flush=True)


if __name__ == "__main__":
    main()
