"""Shared system-wide 20k budget, per-N statistics, and full state recovery."""
import hashlib
import json
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch
from relative_dp.config import TRAIN_CONFIG
from relative_dp.dataset import WindowDataset
from relative_dp.model import make_ema, state_hash, update_ema, masked_epsilon_loss
from relative_dp.train import Trainer as BaseTrainer, runtime_setup, hardware_info, atomic_torch_save, _recover_log, _restore_milestone_checkpoint
from relative_dp.utils import ROOT, atomic_json, read_json, sha256, object_hash
from .common import R3, run_record, checkpoint_dir, implementation_path, update_run, lock, event, now
from .plugins import load_plugin


class Normalizer:
    def __init__(self, state, plugin):
        self.state, self.plugin = state, plugin
        self.mean = np.asarray(state['mean'],np.float32)
        self.std = np.asarray(state['std'],np.float32)
        self.source_ids = state['source_ids']

    @classmethod
    def fit(cls, episodes, plugin, source_ids, source_hashes):
        values=np.concatenate([plugin.observation(e['obs'][:-1]) for e in episodes]).astype(np.float64)
        mean, raw_std=values.mean(0),values.std(0)
        std=np.maximum(raw_std,.001)
        passthrough=plugin.passthrough(values.shape[-1])
        mean[passthrough]=0
        std[passthrough]=1
        state=dict(mean=mean.astype(np.float32).tolist(),std=std.astype(np.float32).tolist(),
            source_ids=source_ids,source_hashes=source_hashes,count=len(values),
            constant_dimensions=np.flatnonzero(raw_std<.001).tolist(),passthrough=passthrough,
            statistics_scope=f'current D{len(episodes)} only, obs[0:T], population std, floor .001; no OOD clipping')
        return cls(state,plugin)

    def normalize(self, raw):
        return (self.plugin.observation(raw)-self.mean)/self.std

    def normalize_history(self, raw):
        # Optional current-anchor transform of BOTH observations as one unit.
        # This is distinct from transforming each time independently.
        transform=getattr(self.plugin,'observation_history',self.plugin.observation)
        return (transform(raw)-self.mean)/self.std

    def as_dict(self):
        return self.state


def data_for_run(rec, implementation):
    manifest_path=R3/'data'/rec['task']/'manifest.json'
    manifest=read_json(manifest_path)
    if manifest.get('complete') is not True:
        raise ValueError('Training requires fully prepared, replay-validated frozen data (complete=true)')
    if len(manifest.get('train20', [])) != 20:
        raise ValueError('Frozen nested demonstration set must contain exactly 20 episodes')
    records=manifest['train20'][:rec['train_n']]
    if len(records)!=rec['train_n']:
        raise ValueError('Missing complete demonstration subset')
    episodes,hashes,ids=[],{},[]
    for record in records:
        path=ROOT/record['path']
        actual=sha256(path)
        if actual!=record['sha256']:
            raise ValueError(f'Demonstration identity changed: {path}')
        with np.load(path,allow_pickle=False) as data:
            episodes.append(dict(obs=data['obs'].copy(),actions=data['actions'].copy()))
        ids.append(record['episode_id'])
        hashes[record['episode_id']]=actual
    schema=read_json(R3/'protocol_and_task_manifests'/(rec['task']+'.json'))['observation_schema']
    plugin=load_plugin(rec['task'],schema,implementation)
    normalizer=Normalizer.fit(episodes,plugin,ids,hashes)
    ds=WindowDataset(episodes,normalizer,16)
    if hasattr(plugin,'observation_history'):
        histories=np.stack([episodes[e]['obs'][[max(0,t-1),t]] for e,t in ds.index])
        ds.observations=torch.from_numpy(normalizer.normalize_history(histories).astype(np.float32))
    raw=np.stack([episodes[e]['obs'][t] for e,t in ds.index])
    ds.actions=torch.from_numpy(plugin.action_encode(ds.actions.numpy(),raw))
    ds.raw_current=torch.from_numpy(raw.astype(np.float32))
    ds.extra_targets={key:torch.as_tensor(value) for key,value in plugin.supervision(episodes,ds.index).items()}
    for key,value in ds.extra_targets.items():
        if len(value)!=len(ds):
            raise ValueError(f'Supervision alignment {key}')
    if hasattr(plugin,'training_actions'):
        packed=plugin.training_actions(ds.actions.numpy(),{k:v.numpy() for k,v in ds.extra_targets.items()})
        ds.actions=torch.as_tensor(packed,dtype=torch.float32)
        if ds.actions.shape[:2] != ds.valid_mask.shape or ds.actions.shape[-1] < 4:
            raise ValueError('Joint diffusion target must keep native action4 as its first channels')
        if not torch.isfinite(ds.actions).all():
            raise ValueError('Nonfinite joint training targets')
    ds.manifest_hash=sha256(manifest_path)
    ds.schema=schema
    ds.plugin=plugin
    return ds


def initialized(cfg,plugin,device):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg['model_init_seed'])
        policy=plugin.build_policy(cfg)
    return policy.to(device)


class Trainer(BaseTrainer):
    def __init__(self,ds,cfg,device):
        runtime_setup()
        self.dataset=ds.to(device)
        ds.raw_current=ds.raw_current.to(device)
        ds.extra_targets={k:v.to(device) for k,v in ds.extra_targets.items()}
        self.config,self.device=dict(cfg),torch.device(device)
        self.policy=initialized(cfg,ds.plugin,device)
        self.initial_weights_hash=state_hash(self.policy.state_dict())
        self.ema=make_ema(self.policy)
        self.optimizer=torch.optim.AdamW(self.policy.parameters(),lr=cfg['learning_rate'],
            betas=tuple(cfg['betas']),weight_decay=cfg['weight_decay'],foreach=False,fused=False)
        self.loader_rng=torch.Generator(device='cpu').manual_seed(cfg['loader_seed'])
        self.diffusion_rng=torch.Generator(device='cpu').manual_seed(cfg['diffusion_seed'])
        self.step,self.elapsed_seconds,self.pairing_digest=0,0.,'0'*64
        self.forward=torch.compile(self.policy,mode=cfg['train_compile_mode'],dynamic=False) if cfg['torch_compile'] else self.policy

    def update(self):
        # The baseline takes precisely the previously validated update path.
        if not self.dataset.extra_targets and not self.config.get('custom_loss',False):
            return super().update()
        started=time.perf_counter()
        obs,actions,mask,indices=self.dataset.sample(self.config['batch_size'],self.loader_rng)
        noisy,timesteps,noise=self.policy.noise_targets(actions,self.diffusion_rng)
        digest=hashlib.sha256(bytes.fromhex(self.pairing_digest))
        for value in (indices,timesteps,noise):
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        self.pairing_digest=digest.hexdigest()
        self.optimizer.zero_grad(set_to_none=True)
        output=self.forward(noisy,timesteps,obs)
        epsilon=output['epsilon'] if isinstance(output,dict) else output
        # All candidates retain action diffusion with weight1. Joint auxiliary
        # channels, if proposed, are separately weighted in plugin.extra_loss.
        loss=masked_epsilon_loss(epsilon[...,:4],noise[...,:4],mask)
        ix=indices.to(self.device)
        batch=dict(obs=obs,actions=actions,mask=mask,noisy=noisy,timesteps=timesteps,noise=noise,
            raw_current=self.dataset.raw_current[ix],
            targets={k:v[ix] for k,v in self.dataset.extra_targets.items()},policy=self.policy)
        extra,metrics=self.dataset.plugin.extra_loss(output,batch,self.step)
        if extra is not None:
            loss=loss+extra
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(self.policy.parameters(),self.config['max_grad_norm'])
        if not torch.isfinite(grad):
            raise FloatingPointError('Nonfinite gradient')
        self.optimizer.step()
        update_ema(self.ema,self.policy,self.config['ema_decay'])
        self.step+=1
        loss_value,grad_value=loss.item(),grad.item()
        self.elapsed_seconds+=time.perf_counter()-started
        return dict(step=self.step,loss=loss_value,grad_norm=grad_value,elapsed_seconds=self.elapsed_seconds,
            updates_per_second=self.step/max(self.elapsed_seconds,1e-9),pairing_digest=self.pairing_digest,
            auxiliary={k:float(v.detach().item() if torch.is_tensor(v) else v) for k,v in metrics.items()})


def source_identity(implementation):
    import importlib
    files=[Path(__file__),Path(__file__).with_name('plugins.py'),ROOT/'pixi.lock',
        ROOT/'src/relative_dp/model.py',ROOT/'src/relative_dp/train.py',ROOT/'src/relative_dp/dataset.py',
        ROOT/'src/relative_dp/config.py',ROOT/'src/round2/learning.py']
    files.extend(sorted((ROOT/'src/relative_dp/vendor/diffusion_policy').glob('*.py')))
    module=importlib.import_module(implementation.get('plugin_module','round3.plugins'))
    files.append(Path(module.__file__))
    files.extend(ROOT/p for p in implementation.get('additional_source_files',[]))
    return {str(p.relative_to(ROOT)):sha256(p) for p in sorted(set(files))}


def train(identifier,debug_updates=None,stop_after=None,device='cuda'):
    rec=run_record(identifier)
    if debug_updates is None:
        from .proposals import validate
        proposal_validation=validate(rec['task'], freeze=False)
        if proposal_validation['status'] != 'validated':
            raise ValueError('Formal training requires saved initial task designs')
        freeze_path=R3/'design_records'/rec['task']/'initial_freeze.json'
        if not freeze_path.exists():
            raise ValueError('Formal training requires frozen initial task designs')
        if read_json(freeze_path)['proposal_sha256'] != proposal_validation['proposal_sha256']:
            raise ValueError('Initial proposal changed after freezing')
    implementation=read_json(implementation_path(rec['task'],rec['candidate_id']))
    expected_proposal=(R3/'ROUND3_SPEC.txt' if rec['candidate_id']=='B0' else
        R3/'design_records'/rec['task']/('revision.json' if rec['candidate_id']=='P4' else 'proposal.json'))
    if implementation['proposal_sha256'] != sha256(expected_proposal):
        raise ValueError('Implementation does not match saved proposal')
    schema=read_json(R3/'protocol_and_task_manifests'/(rec['task']+'.json'))['observation_schema']
    if implementation['observation_schema_sha256'] != object_hash(schema):
        raise ValueError('Implementation observation schema mismatch')
    if rec['candidate_id']=='P4' and rec['train_n'] not in (5,20):
        raise ValueError('Feedback revisions only train at N5 and N20')
    with lock('run-'+identifier,blocking=False):
        return _train(rec,implementation,debug_updates,stop_after,device)


def _train(rec,implementation,debug_updates,stop_after,device):
    identifier=rec['run_id']
    ds=data_for_run(rec,implementation)
    cfg=dict(TRAIN_CONFIG,task=rec['task'],candidate_id=rec['candidate_id'],train_n=rec['train_n'],run_id=identifier,
        obs_dim=len(ds.normalizer.mean),raw_dim=ds.raw_current.shape[-1],clip_predicted_clean_actions=False,
        thresholding=False,implementation=implementation,observation_schema=ds.schema,
        custom_loss=implementation.get('custom_loss',False))
    if hasattr(ds.plugin,'model_config'):
        overrides=ds.plugin.model_config(dict(cfg))
        forbidden={'train_updates','batch_size','microbatch_size','learning_rate','train_seed',
            'loader_seed','diffusion_seed','model_init_seed','checkpoint_updates'}
        if any(k in forbidden and v!=cfg[k] for k,v in overrides.items()):
            raise ValueError('Model hook cannot silently override shared training budget or random streams')
        cfg.update(overrides)
    if cfg['action_dim'] != ds.actions.shape[-1]:
        raise ValueError('Diffusion dimension differs from proposed training target dimension')
    if debug_updates is not None:
        cfg.update(train_updates=int(debug_updates),checkpoint_updates=[int(debug_updates)],debug=True)
    directory=(R3/'debug'/identifier if debug_updates is not None else checkpoint_dir(identifier))
    directory.mkdir(parents=True,exist_ok=True)
    sources=source_identity(implementation)
    metadata=dict(run_id=identifier,task=rec['task'],candidate_id=rec['candidate_id'],train_n=rec['train_n'],
        manifest_hash=ds.manifest_hash,code_hash=object_hash(sources),debug=debug_updates is not None,
        implementation_hash=object_hash(implementation))
    if (directory/'complete.json').exists():
        done=read_json(directory/'complete.json')
        if any(done[k]!=v for k,v in metadata.items()) or done['config_hash']!=object_hash(cfg):
            raise ValueError('Completed run identity mismatch')
        for r in done['checkpoints'].values():
            if sha256(ROOT/r['path'])!=r['sha256']:
                raise ValueError('Completed checkpoint hash mismatch')
        if debug_updates is None:
            update_run(identifier,train_status='completed',pid=None,
                train_complete_sha256=sha256(directory/'complete.json'))
        print(identifier,'validated complete',flush=True)
        return done
    trainer=Trainer(ds,cfg,device)
    # Compare against the exact same raw-input architecture without allocating a second GPU model.
    from round2.learning import Policy
    with torch.random.fork_rng(devices=[]):
        baseline=Policy(cfg['raw_dim'],dict(TRAIN_CONFIG,clip_predicted_clean_actions=False,thresholding=False))
    baseline_parameters=sum(p.numel() for p in baseline.parameters())
    del baseline
    parameter_count=sum(p.numel() for p in trainer.policy.parameters())
    if parameter_count>3*baseline_parameters:
        raise ValueError('Candidate exceeds 3x B0 parameter budget')
    latest=directory/'latest.pt'
    if latest.exists():
        restored=torch.load(latest,map_location='cpu',weights_only=False)
        trainer.restore(restored,metadata)
        _restore_milestone_checkpoint(directory,restored,cfg['checkpoint_updates'])
        _recover_log(directory/'train.jsonl',trainer.step)
        atomic_json(directory/'resume_audit.json',dict(passed=True,step=trainer.step,
            optimizer_state_entries=len(trainer.optimizer.state),pairing_digest=trainer.pairing_digest))
    atomic_json(directory/'config.json',cfg)
    atomic_json(directory/'normalizer.json',ds.normalizer.as_dict())
    atomic_json(directory/'hardware.json',hardware_info(device))
    atomic_json(directory/'identity.json',dict(**metadata,initial_weights_hash=trainer.initial_weights_hash,
        parameter_count=parameter_count,baseline_parameter_count=baseline_parameters,windows=len(ds),source_hashes=sources))
    source_dir=directory/'source_snapshot'
    for path,digest in sources.items():
        target=source_dir/path
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and sha256(target)!=digest:
            raise ValueError('Archived run source differs')
        if not target.exists():
            target.write_bytes((ROOT/path).read_bytes())
    if debug_updates is None:
        update_run(identifier,status='running',train_status='running',pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
    event('training_started',run_id=identifier,debug=debug_updates is not None,start_step=trainer.step,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
    interrupted=False
    def stop_signal(signum,frame):
        nonlocal interrupted
        interrupted=True
    previous={s:signal.signal(s,stop_signal) for s in (signal.SIGINT,signal.SIGTERM)}
    started=time.perf_counter()
    start_step=trainer.step
    try:
        with (directory/'train.jsonl').open('a',buffering=1) as log:
            while trainer.step<cfg['train_updates']:
                metric=trainer.update()
                if trainer.step%100==0 or trainer.step==1 or interrupted:
                    log.write(json.dumps(metric)+'\n')
                    print(identifier,trainer.step,round(metric['loss'],6),round(metric['updates_per_second'],2),flush=True)
                if trainer.step%1000==0 or trainer.step in cfg['checkpoint_updates'] or interrupted or trainer.step==stop_after:
                    checkpoint=trainer.checkpoint(metadata)
                    atomic_torch_save(latest,checkpoint)
                    if trainer.step in cfg['checkpoint_updates']:
                        atomic_torch_save(directory/'checkpoints'/f'step_{trainer.step:06d}.pt',checkpoint)
                    atomic_json(directory/'status.json',dict(step=trainer.step,pid=os.getpid(),running=not interrupted,updated_at=now()))
                if interrupted or trainer.step==stop_after:
                    break
    finally:
        for s,handler in previous.items():
            signal.signal(s,handler)
        with (directory/'sessions.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(start_step=start_step,end_step=trainer.step,wall_seconds=time.perf_counter()-started,
                gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),ended_at=now()))+'\n')
    if trainer.step!=cfg['train_updates']:
        if debug_updates is None:
            update_run(identifier,status='pending',train_status='pending',last_saved_step=trainer.step,pid=None)
        return dict(paused_at=trainer.step)
    checkpoints={}
    for step in cfg['checkpoint_updates']:
        path=directory/'checkpoints'/f'step_{step:06d}.pt'
        checkpoints[str(step)]=dict(path=str(path.relative_to(ROOT)),sha256=sha256(path))
    sessions=[json.loads(line) for line in (directory/'sessions.jsonl').read_text().splitlines()]
    done=dict(**metadata,step=trainer.step,config_hash=object_hash(cfg),checkpoints=checkpoints,
        elapsed_seconds=trainer.elapsed_seconds,total_session_wall_seconds=sum(s['wall_seconds'] for s in sessions),
        gpu_active_work_seconds=trainer.elapsed_seconds,gpu_time_measurement='synchronized allocated-device wall time; not exclusive GPU kernel time',
        parameter_count=parameter_count,baseline_parameter_count=baseline_parameters,
        initial_weights_hash=trainer.initial_weights_hash,pairing_digest=trainer.pairing_digest,
        samples_drawn=trainer.step*cfg['batch_size'],completed_at=now())
    atomic_json(directory/'complete.json',done)
    atomic_json(directory/'status.json',dict(step=trainer.step,running=False,completed=True,updated_at=now()))
    if debug_updates is None:
        update_run(identifier,status='pending',train_status='completed',train_complete_sha256=sha256(directory/'complete.json'),pid=None)
    event('training_completed',run_id=identifier,debug=debug_updates is not None,step=trainer.step)
    return done


class Loaded:
    def __init__(self,checkpoint_path,device='cuda'):
        runtime_setup()
        self.checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        self.config=self.checkpoint['config']
        self.device=torch.device(device)
        self.plugin=load_plugin(self.config['task'],self.config['observation_schema'],self.config['implementation'])
        self.normalizer=Normalizer(self.checkpoint['normalizer'],self.plugin)
        self.policy=initialized(self.config,self.plugin,device)
        self.policy.load_state_dict(self.checkpoint['ema'],strict=True)
        self.policy.eval()
        self.policy.enable_inference_compilation()

    def actions(self,history,generator):
        raw=np.asarray(history,np.float32)
        if raw.shape!=(2,self.config['raw_dim']):
            raise ValueError(f'Invalid observation history shape {raw.shape}')
        normalized=torch.as_tensor(self.normalizer.normalize_history(raw),device=self.device)[None]
        started=time.perf_counter()
        encoded=self.policy.predict_action(normalized,generator)[0].cpu().numpy()
        decoded=self.plugin.action_decode(encoded,raw[-1])
        diag=self.plugin.diagnostics(raw,self.policy)
        diag.update(inference_seconds=time.perf_counter()-started,
            unclipped_world_actions=decoded.tolist(),
            predicted_clipped_coordinates=int(np.sum(np.abs(decoded)>1)),predicted_coordinates=int(decoded.size))
        # ONLY world Cartesian+gripper action cube clipping; local clean samples stay unbounded.
        return np.clip(decoded,-1,1).astype(np.float32),diag
