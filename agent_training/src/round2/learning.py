"""Round1 U-Net/optimizer reused; Round2 never clips clean diffusion samples."""
import copy
import json
import time
import torch
from relative_dp.config import TRAIN_CONFIG
from relative_dp.model import DiffusionPolicy, make_ema, state_hash
from relative_dp.train import Trainer as BaseTrainer, runtime_setup, hardware_info, atomic_torch_save, _recover_log
from relative_dp.utils import ROOT, atomic_json, read_json, sha256, object_hash
from .representation import dataset, Normalizer, action_transform

RUNS = [dict(run_id=f'r2_{task}_{rep}_n20_s0', task=task, representation=rep)
        for task in ('drawer', 'door') for rep in ('world', 'frame')]


def config(run_id):
    r = next(r for r in RUNS if r['run_id'] == run_id)
    return dict(TRAIN_CONFIG, **r, clip_predicted_clean_actions=False, thresholding=False, obs_dim=41)


class Policy(DiffusionPolicy):
    @torch.no_grad()
    def predict_action(self, obs_history, generator):
        scheduler = self.inference_scheduler
        scheduler.set_timesteps(self.config['inference_steps'], device=obs_history.device)
        sample = torch.randn((len(obs_history), self.config['prediction_horizon'], 4), generator=generator,
                             device=generator.device, dtype=obs_history.dtype).to(obs_history.device)
        for timestep in scheduler.timesteps:
            if self._compiled_inference_net is None:
                epsilon = self(sample, timestep, obs_history)
            else:
                torch.compiler.cudagraph_mark_step_begin()
                epsilon = self._compiled_inference_net(sample, timestep, global_cond=obs_history.flatten(start_dim=1))
            sample = scheduler.step(epsilon, timestep, sample, eta=0., use_clipped_model_output=False,
                                    generator=generator, return_dict=True).prev_sample
        return sample


def initialized(cfg, device):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg['model_init_seed'])
        result = Policy(41, cfg)
    return result.to(device)


class Trainer(BaseTrainer):
    def __init__(self, ds, cfg, device):
        runtime_setup()
        self.dataset = ds.to(device)
        self.config, self.device = dict(cfg), torch.device(device)
        self.policy = initialized(cfg, device)
        self.initial_weights_hash = state_hash(self.policy.state_dict())
        self.ema = make_ema(self.policy)
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=cfg['learning_rate'],
            betas=tuple(cfg['betas']), weight_decay=cfg['weight_decay'], foreach=False, fused=False)
        self.loader_rng = torch.Generator(device='cpu').manual_seed(cfg['loader_seed'])
        self.diffusion_rng = torch.Generator(device='cpu').manual_seed(cfg['diffusion_seed'])
        self.step, self.elapsed_seconds, self.pairing_digest = 0, 0., '0' * 64
        self.forward = torch.compile(self.policy, mode=cfg['train_compile_mode'], dynamic=False) if cfg['torch_compile'] else self.policy


def code_hash():
    paths = list((ROOT/'src/round2').glob('*.py')) + [ROOT/'pixi.lock']
    paths += [ROOT/'src/relative_dp'/n for n in ('model.py','train.py','dataset.py','config.py')]
    return object_hash({str(p.relative_to(ROOT)):sha256(p) for p in paths if p.name not in ('report.py','cli.py','evaluate.py')})


def train(run_id, debug=False, stop_after=None):
    cfg = config(run_id)
    if debug:
        cfg.update(train_updates=200, checkpoint_updates=[200], debug=True)
    directory = ROOT / ('artifacts/round2/debug' if debug else 'runs/round2') / run_id
    directory.mkdir(parents=True, exist_ok=True)
    ds = dataset(cfg['task'], cfg['representation'])
    metadata = dict(run_id=run_id, task=cfg['task'], representation=cfg['representation'],
                    manifest_hash=ds.manifest_hash, code_hash=code_hash(), debug=debug)
    if (directory/'complete.json').exists():
        done = read_json(directory/'complete.json')
        assert all(done[k] == v for k,v in metadata.items())
        assert done['step'] == cfg['train_updates']
        for r in done['checkpoints'].values():
            assert sha256(ROOT/r['path']) == r['sha256']
        print(run_id, 'validated complete', flush=True)
        return done
    trainer = Trainer(ds, cfg, 'cuda')
    latest = directory/'latest.pt'
    if latest.exists():
        restored=torch.load(latest, map_location='cpu', weights_only=False)
        trainer.restore(restored, metadata)
        assert state_hash(trainer.policy.state_dict())==state_hash(restored['model'])
        assert state_hash(trainer.ema.state_dict())==state_hash(restored['ema'])
        assert torch.equal(trainer.loader_rng.get_state(),restored['rng']['loader'])
        assert torch.equal(trainer.diffusion_rng.get_state(),restored['rng']['diffusion'])
        atomic_json(directory/'resume_audit.json',dict(passed=True,restored_step=trainer.step,
             model_hash=state_hash(trainer.policy.state_dict()),ema_hash=state_hash(trainer.ema.state_dict()),
             pairing_digest=trainer.pairing_digest,optimizer_state_entries=len(trainer.optimizer.state)))
        _recover_log(directory/'train.jsonl', trainer.step)
    atomic_json(directory/'config.json',cfg)
    atomic_json(directory/'normalizer.json',ds.normalizer.as_dict())
    atomic_json(directory/'hardware.json',hardware_info('cuda'))
    atomic_json(directory/'identity.json',dict(**metadata,initial_weights_hash=trainer.initial_weights_hash,
        parameter_count=sum(p.numel() for p in trainer.policy.parameters()),windows=len(ds)))
    begun=time.monotonic()
    with (directory/'train.jsonl').open('a',buffering=1) as log:
        while trainer.step < cfg['train_updates']:
            metric=trainer.update()
            if trainer.step % 100 == 0 or trainer.step == 1:
                log.write(json.dumps(metric)+'\n')
                print(run_id,metric['step'],round(metric['loss'],6),round(metric['updates_per_second'],2),flush=True)
            if trainer.step % 1000 == 0 or trainer.step in cfg['checkpoint_updates'] or trainer.step==stop_after:
                ckpt=trainer.checkpoint(metadata)
                atomic_torch_save(latest,ckpt)
                if trainer.step in cfg['checkpoint_updates']:
                    atomic_torch_save(directory/'checkpoints'/f'step_{trainer.step:06d}.pt',ckpt)
            if trainer.step==stop_after and trainer.step<cfg['train_updates']:
                return dict(paused_at=trainer.step)
    checkpoints={}
    for step in cfg['checkpoint_updates']:
        p=directory/'checkpoints'/f'step_{step:06d}.pt'
        checkpoints[str(step)]=dict(path=str(p.relative_to(ROOT)),sha256=sha256(p))
    done=dict(**metadata, step=trainer.step, config_hash=object_hash(cfg), checkpoints=checkpoints,
              update_seconds=trainer.elapsed_seconds, last_session_wall_seconds=time.monotonic()-begun,
              initial_weights_hash=trainer.initial_weights_hash, pairing_digest=trainer.pairing_digest,
              parameter_count=sum(p.numel() for p in trainer.policy.parameters()))
    atomic_json(directory/'complete.json',done)
    return done


class Loaded:
    def __init__(self,path):
        runtime_setup()
        ckpt=torch.load(path,map_location='cpu',weights_only=False)
        self.cfg=ckpt['config']
        self.normalizer=Normalizer.from_dict(ckpt['normalizer'])
        self.policy=initialized(self.cfg,'cuda')
        self.policy.load_state_dict(ckpt['ema'],strict=True)
        self.policy.eval()
        self.policy.enable_inference_compilation()

    def actions(self, history, generator):
        import numpy as np
        x=torch.as_tensor(self.normalizer.normalize(history),device='cuda')[None]
        native=self.policy.predict_action(x,generator)[0].cpu().numpy()
        world=action_transform(native,np.asarray(history)[-1],self.cfg['representation'],inverse=True)
        return world


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('run_id');p.add_argument('--debug',action='store_true');p.add_argument('--stop-after',type=int)
    a=p.parse_args();train(a.run_id,a.debug,a.stop_after)
