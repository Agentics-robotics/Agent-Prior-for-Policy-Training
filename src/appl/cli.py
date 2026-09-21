"""Single CLI; all configuration, devices and resumption go through this module."""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from .config import load
from .io import ROOT,atomic,source_manifest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['audit','check','diagnose-dp','design','report','viewer','worker'])
    parser.add_argument('--config')
    parser.add_argument('--stage',default='D0')
    parser.add_argument('--gpu',type=int)
    parser.add_argument('--device-isolated',action='store_true')
    parser.add_argument('--request')
    parser.add_argument('--trace')
    parser.add_argument('--port',type=int,default=8086)
    parser.add_argument('--duration',type=float)
    args=parser.parse_args(); cfg=load(args.config)
    from .io import read,object_hash
    configuration_path=cfg['output']/'configuration.json'
    if configuration_path.exists():
        if object_hash(read(configuration_path))!=cfg['_hash']:raise ValueError('Existing run configuration changed; select a new run ID')
    else:atomic(configuration_path,read(cfg['_path']))
    if args.device_isolated:
        allowed=Path('/dev')/('nvidia'+os.environ['APPL_GPU_MINOR'])
        nodes={p for p in Path('/dev').glob('nvidia*') if p.name.removeprefix('nvidia').isdigit()}
        if nodes!={allowed}:raise RuntimeError('GPU worker has not inherited the exact single-device namespace')
        os.environ['CUDA_VISIBLE_DEVICES']=os.environ['APPL_GPU_UUID']
        os.environ['MUJOCO_EGL_DEVICE_ID']='0'
    if args.command in ('diagnose-dp','worker') and not args.device_isolated:
        from .gpu import launch
        default=cfg['devices']['render'] if args.command=='diagnose-dp' and args.stage=='D0' else cfg['devices']['train']
        return launch(args.gpu if args.gpu is not None else default,sys.argv[1:])
    if args.command=='audit':
        from .data import audit
        from .gpu import identity
        value=dict(data=audit(cfg),source=source_manifest(),devices=[identity(i) for i in cfg['devices']['allowed']],
            storage_free_bytes=shutil.disk_usage(cfg['output'].parent).free,configuration_sha256=cfg['_hash'])
        atomic(cfg['output']/'audit.json',value)
        from .io import archive_source
        archive_source(cfg['output'])
        print(json.dumps({k:v for k,v in value.items() if k not in ('data','source')},indent=2))
    elif args.command=='diagnose-dp':
        if args.stage=='reference':
            from .reference import check
            value=check(cfg)
        else:
            from . import diagnose
            if not hasattr(diagnose,args.stage.lower()):
                raise ValueError('Unknown or retired diagnostic stage. M0 development uses python -m appl.dp_baseline; historical D1-D4 artifacts remain unchanged.')
            value=getattr(diagnose,args.stage.lower())(cfg)
        print(json.dumps(value,indent=2))
        if value.get('success') is False:raise SystemExit(2)
    elif args.command=='check':
        import pytest
        raise SystemExit(pytest.main([str(ROOT/'tests/exp2'),'-q']))
    elif args.command=='worker':
        from .worker import run
        from .io import read
        run(cfg,read(args.request))
    elif args.command=='design':
        if args.stage=='reconstruct':
            from .surrogate import reconstruct
            reconstruct(cfg)
        else:raise ValueError('Old M1/M2 design is retired; use python -m appl.prior_policies queue')
    elif args.command=='report':
        from .report import generate
        value=generate(cfg)
        print(json.dumps(dict(formal_gates=value['formal_gates'],training_jobs=len(value['training']),episodes=len(value['episodes'])),indent=2))
    elif args.command=='viewer':
        from .viewer import serve
        serve(cfg,args.trace,port=args.port,duration=args.duration)
    else:
        raise NotImplementedError(f'{args.command} not yet implemented; never treated as a completed stage')


if __name__=='__main__':main()
