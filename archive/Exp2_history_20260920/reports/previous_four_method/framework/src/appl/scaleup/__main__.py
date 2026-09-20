"""Explicit entry point for the scale-up study."""
import argparse
import sys
from pathlib import Path
from ..io import ROOT,read


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['collect','train-naive','evaluate','freeze','check-workers'])
    p.add_argument('--task',required=True);p.add_argument('--gpu',type=int,default=6)
    p.add_argument('--seeds',type=int,nargs='+')
    p.add_argument('--scope',choices=['preflight','training'],default='training')
    p.add_argument('--condition',choices=['ID','OOD'],default='ID')
    p.add_argument('--method',choices=['naive_DP','APPL'])
    p.add_argument('--device-isolated',action='store_true')
    args=p.parse_args()
    if args.command=='freeze':
        from .protocol import freeze
        return freeze()
    if args.command=='train-naive':
        from ..prior_policies.data import configuration
        from .naive import train
        return train(configuration(ROOT/'experiments/exp2/configs/scaleup'/(args.task+'.json')),args.gpu)
    if args.command!='check-workers' and not args.seeds:p.error('Collection needs explicit seeds')
    from .protocol import config
    if args.gpu not in config(args.task)['devices']:p.error('GPU is outside the four-device scale-up allocation')
    if not args.device_isolated:
        from ..gpu import launch
        return launch(args.gpu,sys.argv[1:],module='appl.scaleup')
    if args.command=='check-workers':
        from .audit import policy_workers
        return policy_workers(args.task)
    if args.command=='evaluate':
        if len(args.seeds)!=1 or args.method is None:p.error('One explicit seed and method per fresh evaluation process')
        from .evaluate import evaluate
        return evaluate(args.task,args.method,args.condition,args.seeds[0])
    from .collect import collect
    base=ROOT/'runs/exp2/M1_scaleup'/args.task;spec=read(base/'task.json')
    output=ROOT/'data/exp2/scaleup'/args.task/'demonstrations' if args.scope=='training' else base/'preflight'/args.condition
    collect(spec,output,args.seeds,base,args.condition,args.scope)


if __name__=='__main__':main()
