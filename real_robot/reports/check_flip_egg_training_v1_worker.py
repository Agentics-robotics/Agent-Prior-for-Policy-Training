"""Host-side contract test of the published isolated process, no robot I/O."""
import fcntl
from pathlib import Path

import numpy as np
from PIL import Image

from appl.io import ROOT, atomic, digest, read
from real_robot.training_pipeline.inference import PolicyProcess


def main():
    config = ROOT / "real_robot/configs/flip_egg_training_v1.json"
    cfg = read(config)
    root = ROOT / cfg["run"]
    revision = read(root / "workflow.json")["revision"]
    package = root / f"package_{revision:02d}"
    prepared = package / "prepared"
    split = np.load(prepared / "split.npy", mmap_mode="r")
    ix = int(np.flatnonzero(split == 2)[0])
    ep = int(np.load(prepared / "example_episode.npy", mmap_mode="r")[ix])
    index = int(np.load(prepared / "example_source_index.npy", mmap_mode="r")[ix])
    cut_cfg = read(ROOT / cfg["cut_config"])
    tid = cut_cfg["trajectory_ids"][ep]
    original = Path(cut_cfg["source_root"]) / tid
    with np.load(original / "obs.npz", allow_pickle=False) as z:
        rows = {k:z[k] for k in z.files}

    def rgb(i, camera):
        with Image.open(original / camera / f"{i:06d}.jpg") as im:
            return np.array(im.convert("RGB"))

    def sample(i):
        return dict(t=float(rows["t"][i]), T_base_ee=rows["T_base_ee"][i].tolist(),
            tau_ext=rows["tau_ext"][i].tolist(), gripper_width_m=float(rows["gripper_width_m"][i]),
            image_age_s=[float(rows["img_age_third"][i]),float(rows["img_age_wrist"][i])],
            third_rgb=rgb(i,"third"), wrist_rgb=rgb(i,"wrist"))

    samples = [sample(i) for i in range(index-12,index+1,3)]
    call = dict(session_id="offline_host_fixture", reference_id="original_initial_scene",
        task="turn_over_in_same_pan", tool_template="demonstrated_spatula", initial_third_rgb=rgb(0,"third"),
        reset=True, calibration_id="flip_20260919_recorded_rgb_tcp_v1", tool_retained=True,
        scene_valid=True, supported_context=True, authorized_for_proposal=True, max_predictive_spread=[10.]*6)
    # Flags/limits here are synthetic interface fixtures, not commissioned hardware facts.
    observation = dict(samples=samples,now=samples[-1]["t"],observation_id="offline_host_0")
    lock_path = ROOT / "real_robot/runs/.training_pipeline_locks/gpu_4.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        worker = PolicyProcess(config,"egg_interaction_v1",root/"host_worker_check",gpu=4,revision=revision)
        try:
            first = worker.act(observation,call,reset=True,executor_contract={})
            assert first["status"] == "active" and len(first["decision"]["velocity"]) == 6
            assert np.isfinite(first["decision"]["velocity"]).all()
            assert first["executor_request"]["operation"] == "reject"
            newer = sample(index+3)
            follow_call = {k:v for k,v in call.items() if k != "initial_third_rgb"}
            follow_call["reset"] = False
            follow_obs = dict(samples=[newer],now=newer["t"],observation_id="offline_host_1")
            second = worker.act(follow_obs,follow_call,executor_contract={})
            assert second["status"] == "active" and second["history_length"] == 5
            interrupted = worker.act(follow_obs,dict(follow_call,interrupt=True))
            assert interrupted["status"] == "interrupted" and interrupted["decision"] is None
        finally:
            worker.close()
    result = dict(passed=True, policy_id="egg_interaction_v1", original_episode=tid,
        original_index=index, cases=[first,second,interrupted], actual_separate_worker=True,
        synthetic_flags_and_limits=True, hardware_execution=False, performance_evaluation=False,
        checkpoint_sha256=digest(package/"training/egg_interaction_v1/last.pt"))
    atomic(root/"host_worker_check/result.json",result)
    print(dict(passed=True, statuses=[x["status"] for x in result["cases"]]),flush=True)


if __name__ == "__main__":
    main()
