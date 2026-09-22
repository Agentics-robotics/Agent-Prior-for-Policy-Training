import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from real_robot.training_pipeline import public
from data_tools import prepare
from online import act_impl, request_impl, cases_impl, prediction


class VisualMixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3,16,5,2,2),nn.GroupNorm(4,16),nn.SiLU(),nn.Conv2d(16,32,3,2,1),nn.GroupNorm(4,32),nn.SiLU(),nn.Conv2d(32,48,3,2,1),nn.GroupNorm(6,48),nn.SiLU(),nn.Conv2d(48,64,3,2,1),nn.GroupNorm(8,64),nn.SiLU(),nn.AdaptiveAvgPool2d((4,6)),nn.Flatten(),nn.Linear(1536,64),nn.LayerNorm(64),nn.SiLU())
        self.fuse = nn.Sequential(nn.Linear(220,128),nn.LayerNorm(128),nn.SiLU(),nn.Dropout(.05))
        self.cell = nn.GRUCell(128,128)
        self.head = nn.Linear(128,39)
        nn.init.normal_(self.head.weight,std=.005)
        nn.init.zeros_(self.head.bias)
        self.register_buffer('scales',torch.tensor([.1,.1,.1,.5,.5,.5]))
        self.register_buffer('best_score',torch.tensor(1e10))
        self.register_buffer('best_step',torch.tensor(0,dtype=torch.long))
        self.register_buffer('bad_evals',torch.tensor(0,dtype=torch.long))
        self.best_snapshot = None

    def forward(self,b):
        x, goal = b['images'], b['goal']
        if self.training:
            gain = 1 + (torch.rand(x.shape[0],1,1,1,1,1,device=x.device)-.5)*.16
            bias = (torch.rand(x.shape[0],1,1,3,1,1,device=x.device)-.5)*.06
            x = (x*gain+bias).clamp(0,1)
            goal = (goal*gain[:,0,0]+bias[:,0,0]).clamp(0,1)
        B, S, V, C, H, W = x.shape
        emb = self.encoder(x.reshape(B*S*V,C,H,W)*2-1).reshape(B,S,2,64)
        f, mask = b['features'].clone(), b['mask'].clone()
        if self.training:
            drop = (torch.rand(B,1,2,device=x.device) > .05).float()
            f[:,:,19:21] *= drop
            starts = torch.where(torch.rand(B,device=x.device)<.25,torch.randint(0,5,(B,),device=x.device),torch.zeros(B,dtype=torch.long,device=x.device))
            mask *= (torch.arange(5,device=x.device)[None,:] >= starts[:,None]).float()
            for j in range(5):
                f[starts==j,j,21:] = 0
        emb = emb * f[:,:,19:21,None]
        g = self.encoder(goal*2-1)[:,None,:].expand(B,S,64)
        inp = self.fuse(torch.cat((emb.reshape(B,S,128),g,f.clamp(-6,6)),dim=-1))
        h = torch.zeros(B,128,device=x.device)
        for j in range(S):
            nh = self.cell(inp[:,j],h)
            h = nh*mask[:,j,None]+h*(1-mask[:,j,None])
        raw = self.head(h)
        return raw[:,:3],3*torch.tanh(raw[:,3:21].reshape(B,3,6)/3),raw[:,21:].reshape(B,3,6).clamp(-2.8,.7)-.25


def prepare_data(spec):
    return prepare(spec)


def build_model(policy_id,spec):
    if policy_id != 'egg_interaction_v1':
        raise ValueError('Unknown policy')
    return VisualMixture()


def make_batch(policy_id,arrays,metadata,indices,spec):
    i = np.asarray(indices,dtype=np.int64)
    device = spec['device']
    return {'images':torch.as_tensor(arrays['frames'][arrays['sequence'][i]],device=device,dtype=torch.float32)/255.,'goal':torch.as_tensor(arrays['goals'][arrays['example_episode'][i]],device=device,dtype=torch.float32)/255.,'features':torch.as_tensor(arrays['features'][i],device=device,dtype=torch.float32),'mask':torch.as_tensor(arrays['history_mask'][i],device=device,dtype=torch.float32),'target':torch.as_tensor(arrays['target'][i],device=device,dtype=torch.float32),'target_mask':torch.as_tensor(arrays['target_mask'][i],device=device,dtype=torch.float32)}


def mixture_loss(model,b):
    logits,mu,logsd = model(b)
    y = b['target']/model.scales
    gaussian = (.5*((y[:,None,:]-mu)*torch.exp(-logsd))**2+logsd+.5*math.log(2*math.pi))*b['target_mask'][:,None,:]
    logjoint = F.log_softmax(logits,dim=-1)-gaussian.sum(-1)
    nll = -torch.logsumexp(logjoint,dim=-1).mean()
    responsibility = torch.softmax(logjoint,dim=-1).detach()
    huber = F.smooth_l1_loss(mu,y[:,None,:].expand_as(mu),reduction='none',beta=.3).mean(-1)
    endpoint = (responsibility*huber).sum(-1).mean()
    selected = mu[torch.arange(len(mu),device=mu.device),logits.argmax(-1)]*model.scales
    error = selected-b['target']
    return {'loss':nll+.2*endpoint,'nll':nll,'endpoint_huber':endpoint,'translation_mae_m_s':error[:,:3].abs().mean(),'rotation_mae_rad_s':error[:,3:].abs().mean(),'zero_translation_mae_m_s':b['target'][:,:3].abs().mean(),'zero_rotation_mae_rad_s':b['target'][:,3:].abs().mean()}


def compute_loss(model,batch,spec):
    return mixture_loss(model,batch)


def predict(model,batch,spec):
    return prediction(model,batch)


def configure_optimizer(model,spec):
    c = spec['policy_config']['config']
    return torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])


def before_update(optimizer,step,spec):
    maxsteps = spec['policy_config']['updates']
    factor = min(1.,step/100.)*(.15+.85*.5*(1+math.cos(math.pi*min(step,maxsteps)/maxsteps)))
    for p in optimizer.param_groups:
        p['lr'] = spec['policy_config']['config']['learning_rate']*factor


def sample_indices(policy_id,arrays,metadata,rng,step,spec):
    ok = (arrays['split']==0)&(arrays['policy_weight'][:,0]>0)
    episodes = np.unique(arrays['example_episode'][ok])
    batch, size = [], spec['policy_config']['batch_size']
    for j in range(size):
        ep = int(rng.choice(episodes))
        cls = 0 if j < size//10 else (2 if rng.random()<.34 else 1)
        inds = np.flatnonzero(ok&(arrays['example_episode']==ep)&(arrays['motion_class']==cls))
        if not len(inds):
            inds = np.flatnonzero(ok&(arrays['example_episode']==ep)&(arrays['motion_class']>0))
        if not len(inds):
            raise ValueError('Training episode has no moving examples')
        batch.append(int(rng.choice(inds)))
    return np.asarray(batch,dtype=np.int64)


def eval_partition(model,arrays,metadata,spec,partition):
    episodes = np.unique(arrays['example_episode'][arrays['split']==partition])
    per = []
    for ep in episodes:
        ids = np.flatnonzero((arrays['split']==partition)&(arrays['example_episode']==ep))
        selected = ids[np.unique(np.linspace(0,len(ids)-1,min(128,len(ids))).astype(int))]
        metrics, preds, truth, classes, phase = [], [], [], [], []
        for a in range(0,len(selected),24):
            ix = selected[a:a+24]
            b = make_batch('egg_interaction_v1',arrays,metadata,ix,spec)
            m = mixture_loss(model,b)
            metrics.append((len(ix),{key:float(value.cpu()) for key,value in m.items()}))
            preds.append(prediction(model,b).cpu().numpy())
            truth.append(arrays['target'][ix])
            classes.append(arrays['motion_class'][ix])
            phase.extend(np.minimum(3,((np.arange(a,a+len(ix))/len(selected))*4).astype(int)).tolist())
        p, y, cl = np.concatenate(preds), np.concatenate(truth), np.concatenate(classes)
        result = {key:sum(n*m[key] for n,m in metrics)/len(selected) for key in metrics[0][1]}
        strata = {}
        for c in [0,1,2]:
            subset = cl==c
            if subset.any():
                strata[str(c)]={'count':int(subset.sum()),'mae6':np.abs(p[subset,:6]-y[subset]).mean(0).tolist(),'zero_mae6':np.abs(y[subset]).mean(0).tolist()}
        result['strata_static_translate_rotate'] = strata
        result['progress_quartile_mae6'] = [np.abs(p[np.array(phase)==q,:6]-y[np.array(phase)==q]).mean(0).tolist() for q in range(4)]
        result['episode'] = metadata['trajectory_ids'][int(ep)]
        result['evaluated_examples'] = len(selected)
        result['within_predictive_2sd_fraction'] = float(np.mean(np.abs(p[:,:6]-y)<=2*p[:,6:]))
        per.append(result)
    keys = ['loss','nll','translation_mae_m_s','rotation_mae_rad_s','zero_translation_mae_m_s','zero_rotation_mae_rad_s']
    result = {key:float(np.mean([r[key] for r in per])) for key in keys}
    result.update({'by_episode':per,'scope':'Equal-episode average, <=128 deterministic probes per episode. Open-loop achieved-motion imitation, not rollouts.'})
    return result


def after_update(model,ema,arrays,metadata,step,metrics,spec):
    c = spec['policy_config']['config']
    if step % c['eval_every'] != 0 and step != spec['policy_config']['updates']:
        return {'stop':False,'select':None,'metrics':{}}
    val = eval_partition(model,arrays,metadata,spec,1)
    selected = val['loss'] < float(model.best_score.cpu()) - .001
    if selected:
        model.best_score.fill_(val['loss'])
        model.best_step.fill_(step)
        model.bad_evals.zero_()
        model.best_snapshot = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    else:
        model.bad_evals.add_(1)
    stop = (step >= c['minimum_updates'] and int(model.bad_evals.cpu()) >= c['patience_evaluations']) or step >= spec['policy_config']['updates']
    result = {'validation':val,'selected_step':int(model.best_step.cpu()),'selected':selected}
    if stop:
        current = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if model.best_snapshot is not None:
            model.load_state_dict(model.best_snapshot)
        result['heldout_test_selected_checkpoint'] = eval_partition(model,arrays,metadata,spec,2)
        result['train_selected_checkpoint'] = eval_partition(model,arrays,metadata,spec,0)
        model.load_state_dict(current)
        result['scientific_scope'] = 'One prescribed run; no leave-block-out retraining, physical execution or success evaluation.'
        public.write_report('training_evaluation.json',result)
    return {'stop':stop,'select':'model' if selected else None,'metrics':result}


def act(model,observation,call,memory,spec):
    return act_impl(model,observation,call,memory,spec)


def executor_request(decision,executor_contract,spec):
    return request_impl(decision,executor_contract,spec)


def calling_cases(arrays,metadata,spec):
    return cases_impl(arrays,metadata,spec)
