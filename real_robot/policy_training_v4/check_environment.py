"""Developer dependency/isolation check, no demonstration or policy fitting."""
import argparse
import os
from pathlib import Path
import sys

from appl.io import ROOT, atomic


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--gpu",type=int,default=4)
    p.add_argument("--device-isolated",action="store_true")
    a=p.parse_args()
    if not a.device_isolated:
        from appl.gpu import launch
        return launch(a.gpu,sys.argv[1:],module="real_robot.policy_training_v4.check_environment")
    os.environ["CUDA_VISIBLE_DEVICES"]=os.environ["APPL_GPU_UUID"]
    out=ROOT/"real_robot/runs/push_letters/training_v4/setup/isolated_environment"
    out.mkdir(parents=True,exist_ok=True)
    source=out/"source";source.mkdir(exist_ok=True)
    (source/"fixture.py").write_text('"""TEST FIXTURE: no candidate design."""\n')
    from .security import lockdown
    device=lockdown(source,out,[],gpu=True)
    from sam2.build_sam import build_sam2
    from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
    model=build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml",ckpt_path=None,device="cuda",apply_postprocessing=False)
    atomic(out/"result.json",dict(passed=True,device=device,sam2_architecture_constructed=True,pretrained_weights_loaded=False,transformers_qwen_class_imported=True,candidate_training_updates=0,physical_robot_io=False))
    print("Isolated SAM 2 construction and Qwen class imports passed; zero candidate updates.",flush=True)


if __name__=="__main__":
    main()
