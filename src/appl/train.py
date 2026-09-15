"""One real gradient trainer, final EMA rule, complete optimizer/RNG resumption."""
from collections import deque
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from diffusers import DDPMScheduler,DDIMScheduler

from . import baseline, data
from .public import Factory
from .io import atomic,read,digest,source_manifest,event,object_hash,archive_source


def module_at(candidate):
    if candidate is None:return baseline
    path=Path(candidate)/'policy.py'
    spec=importlib.util.spec_from_file_location('api_policy',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for name in ('build_model','condition','encode_action','decode_action','compute_loss'):
        if not callable(getattr(module,name,None)):raise ValueError('Missing candidate hook: '+name)
    return module


def tensor_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def save(path,value):
    temp=Path(path).with_suffix('.tmp');torch.save(value,temp);temp.replace(path)


def scheduler(config,cls=DDPMScheduler):
    return cls(num_train_timesteps=config['denoising_train_steps'],beta_schedule='squaredcos_cap_v2',
               clip_sample=config['clip_sample'],prediction_type='epsilon')


def deterministic_numerics():
    import os
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('Deterministic CUDA workspace must be configured by the fixed launcher')
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)


@torch.no_grad()
def sample(model,module,raw,spec,generator):
    config=spec['training']; s=scheduler(config,DDIMScheduler)
    s.set_timesteps(config['denoising_inference_steps'],device=raw.device)
    value=torch.randn((len(raw),config['horizon'],8),device=raw.device,generator=generator)
    condition=module.condition(raw,spec)
    if condition.ndim!=2 or len(condition)!=len(raw) or not torch.isfinite(condition).all():
        raise ValueError('Condition must be a finite [batch,dimension] tensor')
    for t in s.timesteps:
        predicted=model(value,t,condition)
        value=s.step(predicted,t,value,eta=0).prev_sample
    native=module.decode_action(value,spec)
    if native.shape!=value.shape or not torch.isfinite(native).all():raise ValueError('Invalid decoded native actions')
    return native


def fit(cfg,output,*,ids=None,segments=None,candidate=None,updates=None,seed=0,fixed_batch=False,lockdown=None,stop_after_checkpoint=None):
    from .gpu import verify_cuda
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    config=copy.deepcopy(cfg['training']);config['updates']=updates or config['updates']
    if (output/'result.json').exists():
        result=read(output/'result.json');previous=read(output/'request.json')
        expected=data.episodes(cfg,ids,segments)
        expected_data=[{k:e[k] for k in ('id','start','stop','source_sha256')} for e in expected]
        hashes={} if candidate is None else {p.name:digest(p) for p in Path(candidate).glob('*') if p.is_file()}
        if previous['config']!=config or previous['seed']!=seed or previous['data']!=expected_data or previous['candidate_hashes']!=hashes or previous['fixed_batch']!=fixed_batch:
            raise ValueError('Completed artifact is for different data/config/candidate/seed; use another run ID')
        if digest(result['checkpoint'])!=result['checkpoint_sha256']:raise ValueError('Completed checkpoint was modified')
        return result
    deterministic_numerics()
    torch.set_num_threads(2);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    np.random.seed(seed);random.seed(seed)
    device=verify_cuda();dev=torch.device('cuda:0')
    es=data.episodes(cfg,ids,segments); norm=data.normalizer(es,config)
    spec=dict(training=config,normalizer=norm,fields=data.SLICES,observation_dimension=47,
        skill=None if candidate is None else read(Path(candidate)/'candidate.json')['skill'],
        candidate_config={} if candidate is None else read(Path(candidate)/'candidate.json').get('config',{}))
    arrays=data.windows(es,config)
    arrays={k:torch.as_tensor(v,device=dev) for k,v in arrays.items() if k!='indices'}
    archive_source(cfg['output'])
    source=source_manifest();candidate_hashes={} if candidate is None else {p.name:digest(p) for p in Path(candidate).glob('*') if p.is_file()}
    request=dict(seed=seed,config=config,source=source,candidate_hashes=candidate_hashes,
        data=[{k:e[k] for k in ('id','start','stop','source_sha256')} for e in es],fixed_batch=fixed_batch)
    if (output/'request.json').exists() and read(output/'request.json')!=request:
        raise ValueError('Training resumption source/data/config changed; use a new run ID')
    atomic(output/'request.json',request)
    # Trusted initialization loads numerical capabilities before candidate imports.
    if lockdown is not None:lockdown()
    # Trusted CUDA warmup is infrastructure, not part of candidate initialization.
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed)
    module=module_at(candidate);factory=Factory(config)
    model=module.build_model(factory,spec).to(dev)
    if len(factory.created)!=1 or not set(id(p) for p in factory.created[0].parameters()).issubset({id(p) for p in model.parameters()}):
        raise ValueError('Candidate must retain the common diffusion backbone')
    count=sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_count=sum(p.numel() for p in factory.created[0].parameters())
    if count==0 or count>backbone_count*1.1:raise ValueError('No trainable parameters or unmatched large architecture')
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    initial_hash=tensor_hash(initial)
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    generator=torch.Generator(device=dev).manual_seed(seed)
    selection=torch.linspace(0,len(arrays['raw_obs'])-1,config['batch_size'],device=dev).long()
    noise_scheduler=scheduler(config)
    start_step=0;elapsed_before=0.;effective_targets=0
    if (output/'last.pt').exists():
        c=torch.load(output/'last.pt',map_location='cpu',weights_only=True)
        assert c['request_hash']==object_hash(request)
        model.load_state_dict(c['model']);ema.load_state_dict(c['ema']);optimizer.load_state_dict(c['optimizer'])
        generator.set_state(c['generator']);torch.set_rng_state(c['torch_rng']);torch.cuda.set_rng_state_all(c['cuda_rng'])
        random.setstate(c['python_rng']); np.random.set_state(tuple([c['numpy_rng'][0],np.asarray(c['numpy_rng'][1],dtype=np.uint32),*c['numpy_rng'][2:]]))
        start_step=c['step'];elapsed_before=c['elapsed_seconds'];initial_hash=c['initial_weights_sha256'];effective_targets=c['effective_action_targets']
        initial=c['initial_weights']
        event(output/'events.jsonl','exact_training_resume',step=start_step,checkpoint_sha256=digest(output/'last.pt'))
        if (output/'progress.json').exists():
            recorded=read(output/'progress.json')
            if recorded['step']>start_step:
                event(output/'events.jsonl','discarded_uncheckpointed_work',minimum_optimizer_steps=recorded['step']-start_step,
                    additional_unrecorded_steps_upper_bound=99,reason='Restore last exact optimizer/EMA/RNG checkpoint; lost-attempt cost is not zero')
    model.train();started=time.monotonic();losses=[];gradient_checks=[]
    for step in range(start_step+1,config['updates']+1):
        index=selection if fixed_batch else torch.randint(len(arrays['raw_obs']),(config['batch_size'],),device=dev,generator=generator)
        batch={k:v[index] for k,v in arrays.items()}
        effective_targets+=int(batch['mask'].sum())
        encoded=module.encode_action(batch['native_action'],spec)
        if encoded.shape!=batch['native_action'].shape or not torch.isfinite(encoded).all():raise ValueError('Invalid encoded actions')
        if step==start_step+1:
            inverse=module.decode_action(encoded,spec)
            if not torch.allclose(inverse,batch['native_action'],atol=1e-5,rtol=1e-5):raise ValueError('Action encoding must invert to native control')
        noise=torch.randn(encoded.shape,device=dev,generator=generator)
        timesteps=torch.randint(config['denoising_train_steps'],(len(index),),device=dev,generator=generator)
        batch.update(noise=noise,timesteps=timesteps,noisy_action=noise_scheduler.add_noise(encoded,noise,timesteps),encoded_action=encoded)
        losses_result=module.compute_loss(model,batch,spec);loss=losses_result['loss']
        if loss.ndim or not torch.isfinite(loss) or not loss.requires_grad:raise ValueError('Loss must be a differentiable finite scalar')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        if step<=10 or step==start_step+1:
            gradients=[p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):raise ValueError('Missing or nonfinite gradients')
            norm_g=float(torch.sqrt(sum(g.square().sum() for g in gradients)))
            if norm_g<=0:raise ValueError('Neural training has zero effective gradient')
            gradient_checks.append(dict(step=step,l2_norm=norm_g))
            event(output/'events.jsonl','gradient_check',step=step,l2_norm=norm_g)
        optimizer.step()
        with torch.no_grad():
            for target,param in zip(ema.parameters(),model.parameters(),strict=True):target.lerp_(param,1-config['ema_decay'])
        losses.append(float(loss.detach()))
        if step%100==0 or step==1:
            progress=dict(step=step,loss_mean=float(np.mean(losses)),elapsed_seconds=elapsed_before+time.monotonic()-started,
                          sampled_windows=step*config['batch_size'],effective_action_targets=effective_targets)
            event(output/'learning.jsonl','progress',**progress);atomic(output/'progress.json',progress);losses.clear()
        if step%config['checkpoint_every']==0 or step==config['updates']:
            rng=np.random.get_state()
            checkpoint=dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
                generator=generator.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                python_rng=random.getstate(),numpy_rng=[rng[0],rng[1].tolist(),*rng[2:]],step=step,spec=spec,
                request_hash=object_hash(request),elapsed_seconds=elapsed_before+time.monotonic()-started,
                initial_weights_sha256=initial_hash,initial_weights=initial,candidate_hashes=candidate_hashes,effective_action_targets=effective_targets)
            save(output/'last.pt',checkpoint)
            if stop_after_checkpoint==step:
                event(output/'events.jsonl','declared_checkpoint_interruption',step=step,scope='Infrastructure resume check only')
                return dict(status='interrupted_at_saved_checkpoint',step=step)
    torch.cuda.synchronize()
    final_hash=tensor_hash(model.state_dict())
    delta=sum((v.detach().cpu()-initial[k]).double().square().sum().item() for k,v in model.state_dict().items())**.5
    if final_hash==initial_hash or delta<=0:raise ValueError('Optimizer completed without changing model weights')
    ema.eval()
    example_index=selection[:min(32,len(selection))]
    predicted=sample(ema,module,arrays['raw_obs'][example_index],spec,torch.Generator(device=dev).manual_seed(913))
    target=arrays['native_action'][example_index];mask=arrays['mask'][example_index]
    joint_error=float(torch.sqrt(((predicted[:,:,:7]-target[:,:,:7]).square()*mask).sum()/(mask.sum()*7)))
    gripper_error=float(((predicted[:,:,7:]-target[:,:,7:]).abs()*mask).sum()/mask.sum())
    result=dict(success=True,optimizer_steps=config['updates'],sample_exposures=config['updates']*config['batch_size'],
        trainable_parameters=count,backbone_parameters=backbone_count,model_class=type(model).__name__,effective_action_targets=effective_targets,
        initial_weights_sha256=initial_hash,final_weights_sha256=final_hash,weight_l2_change=delta,
        finite_nonzero_gradient_checks=gradient_checks,training_elapsed_seconds=elapsed_before+time.monotonic()-started,
        device=device,spec=spec,sampling_joint_rmse=joint_error,sampling_gripper_mae=gripper_error,
        checkpoint=str(output/'last.pt'),checkpoint_sha256=digest(output/'last.pt'),neural_training_verified=True,
        checkpoint_selection=config['selection'],candidate_hashes=candidate_hashes)
    # Independent reload reproduces the same EMA sample and deterministic RNG.
    reloaded=module.build_model(Factory(config),spec).to(dev)
    reloaded.load_state_dict(torch.load(output/'last.pt',map_location='cpu',weights_only=True)['ema']);reloaded.eval()
    replay=sample(reloaded,module,arrays['raw_obs'][example_index],spec,torch.Generator(device=dev).manual_seed(913))
    result['reload_max_action_error']=float((replay-predicted).abs().max())
    if result['reload_max_action_error']>1e-6:raise ValueError('EMA checkpoint reload changes predictions')
    atomic(output/'result.json',result);return result


class LoadedPolicy:
    def __init__(self,checkpoint,candidate=None,seed=0):
        deterministic_numerics()
        c=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.spec=c['spec'];self.module=module_at(candidate);self.device=torch.device('cuda:0')
        self.model=self.module.build_model(Factory(self.spec['training']),self.spec).to(self.device)
        self.model.load_state_dict(c['ema']);self.model.eval()
        self.generator=torch.Generator(device=self.device).manual_seed(seed)
        self.history=deque(maxlen=self.spec['training']['observation_steps']);self.queue=deque()
        self.timings=[]

    def action(self,state):
        value=data.vector(state);self.history.append(value)
        while len(self.history)<self.history.maxlen:self.history.append(value)
        if not self.queue:
            started=time.perf_counter()
            prediction=sample(self.model,self.module,torch.as_tensor(np.stack(self.history),device=self.device)[None],self.spec,self.generator)[0].cpu().numpy()
            self.timings.append(time.perf_counter()-started)
            start=self.spec['training']['observation_steps']-1
            self.queue.extend(prediction[start:start+self.spec['training']['execution_steps']].tolist())
        return self.queue.popleft()
