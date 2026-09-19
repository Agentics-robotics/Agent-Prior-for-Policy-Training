"""API-owned scientific design; developer-owned bounded tools and contract."""
import json
import os
from pathlib import Path

import numpy as np

from appl.agent import AgentLoop
from appl.io import ROOT, atomic, digest, read
from appl.prior_policies.data import shared_normalizer
from appl.prior_policies.design import DesignTools, client, schema

PROMPT = '''You are the scientific design and implementation agent for ONE full-task
prior-based Diffusion Policy. Study the task and its twelve complete training
demonstrations, choose a useful inductive bias, and implement it in a learned
diffusion model. You own the scientific choice, policy source and PRIOR.md.
The repository supplies the data, numerical interface, DDPM sampler, training
recipe and evaluator. It does not prescribe a prior or a menu of candidate designs.

This is offline design followed by autonomous learned-policy deployment. Your
single model must handle the entire task from reset through completion, including
all demonstrated transitions. No language-model agent, policy selector or external
skill schedule will run during deployment. You may learn internal structure,
representations, conditioning or differentiable auxiliary objectives appropriate
to the observed task. The final output must remain one learned action diffusion
policy and one checkpoint. There is no requirement to invent multiple heuristics.

Read INTERFACE.md, read_assignment, and actual demonstration observations before
implementing. Use read_overview to inspect a whole trajectory and read_steps to
inspect transitions at source indices of your choice. Ground your chosen bias
in training evidence and explain its expected benefit, actual gradient paths and
limitations. You see only training data; no performance results or test layouts
are available. Check failures are interface diagnostics, not rollout feedback.

Write policy.py, pipeline.json and PRIOR.md, then call check_policy. Repair your
own unsubmitted code if necessary within the fixed interface-check budget. After
a successful check, submit the exact checked package hash with submit_policy.
All source and documentation must be English. Submitted source is immutable.
'''

INTERFACE = '''# Single full-task prior Diffusion Policy

You own policy.py, pipeline.json, PRIOR.md and optional flat helper .py files.
Allowed imports: torch, numpy, math, appl.public, and your own flat modules.
Generated code executes in credential-free Landlock/seccomp workers. No filesystem,
network, shell, simulator/expert access or external action controller is available.

Required hooks:
- build_model(spec) -> torch.nn.Module, 1..64 million trainable parameters.
- model.forward(noisy_action [B,16,8], timestep scalar or [B],
  raw_history [B,2,47]) -> predicted epsilon [B,16,8].
- compute_loss(model,batch,spec) -> scalar tensor dict loss, diffusion_loss,
  prior_loss. Total and diffusion losses must be differentiable. An architectural
  prior may return zero prior_loss. Auxiliary losses must involve trainable
  predictions; an observed-state-only penalty has no learning effect.

Fixed recipe: one candidate, training seed 0, 60,000 updates, batch 128,
history 2, horizon 16, execution prefix 8, DDPM100 training and sampling,
epsilon prediction, clip_sample true, AdamW lr 1e-4 / weight decay 1e-6,
EMA .999, cosine LR with 500 warmup steps, gradient clipping at 1.
Final EMA is selected without performance-based checkpoint selection.
Interface checks train a fresh model for two updates, sample and verify EMA
reload; these updates are separate and do not initialize formal training.

spec keys: training, normalizer, fields (named [start,stop] indices),
observation_dimension=47, skill_id='full_task', candidate_config=pipeline config.
The identity field skill_id is an interface label for the whole task, not an
API trajectory cut. heuristic_index=1 is the sole candidate identity.

Raw state: qpos 0:9, qvel 9:18, tcp_pose 18:25, red_pose 25:32,
blue_pose 32:39, drawer_position 39:40, drawer_velocity 40:41,
red_goal 41:44, blue_goal 44:47. Positions are world metres, joints radians,
quaternions wxyz. Non-drawer tasks have zero drawer compatibility channels.
Two causal observations only: no images, previous action channel, episode clock,
phase labels or persistent external memory are supplied to forward. The model
is repeatedly evaluated during denoising, so forward must not advance a mutable
cross-call state on each DDPM iteration. Training and inference use this forward.

The shared observation/action normalizer fits the complete original twelve
trajectories once. In limits mode, std means HALF RANGE, not standard deviation.
Quaternion components have fixed unit bounds; constant features have unit scales.
No observation clipping or per-phase refitting. Derived features may use the
shared scales; avoid unstable division by narrow empirical subranges.

Native actions: seven absolute Panda joint targets in radians, plus one gripper
command in [-1,1], increasing toward open; per-finger target is
.025*command+.015 metres. The fixed sampler decodes normalized action samples;
environment action bounds are applied by the same evaluator as naive DP.
A relative state representation does not by itself make joint actions equivariant.
No provided IK or robot forward-kinematics solver is part of this interface.

Public helpers: DiffusionBackbone(condition_dimension,spec['training']) is an
optional conditional U-Net building block with widths 128/256/512, timestep
embedding 128, kernel 5 and groups 8. Its forward is (sample,timestep,condition).
normalize_observation(raw,spec), normalize_action(native,spec),
denormalize_action(encoded,spec), epsilon_loss(predicted_noise,noise,mask).
You may implement a different learned diffusion architecture within the budget.

batch: raw_obs [B,2,47], native_action/encoded_action/noisy_action/noise [B,16,8],
timesteps [B], mask [B,16,1], future_obs [B,16,47], future_mask [B,16,1],
alpha_bar [B]. At observation t, action slots are t-1 through t+14;
future slot j is the state AFTER that action. Padding is masked at complete
trajectory boundaries. Future observations are training labels only and must
never enter deployment forward. Reconstruct x0 as
(noisy_action-sqrt(1-alpha_bar)*predicted_epsilon)/sqrt(alpha_bar), reshaping
alpha_bar to [B,1,1]. Account for high-noise amplification in auxiliary losses.
Learned state/action consequence models are approximations, not exact physics.

pipeline.json exact keys: policy_id, skill_id, heuristic_index, prior_summary,
applicable_conditions, termination_guidance, limitations, config (JSON object).
Copy the three identity fields from the assignment. Other text fields are
nonempty English strings. PRIOR.md must explain your selected inductive bias,
training evidence with trajectory IDs/indices, implemented architecture/losses,
gradient paths, causal inputs, parameterization and limitations. HANDOFF.json
is not required: deployment always uses this same policy. Metadata termination
guidance is documentation, not a runtime controller.

The unchanged evaluator ends at the supplied simultaneous geometric task success
or its physical step limit. No extra gripper release, velocity, clearance or hold
condition is added. No test performance is available for design or selection.
'''


class FullTaskTools(DesignTools):
    def schemas(self):
        definitions = super().schemas()
        for item in definitions:
            if item['name'] == 'read_assignment':
                item['description'] = 'Read the full-task assignment, complete demonstration ranges, shared normalization and fixed training contract.'
            elif item['name'] == 'read_steps':
                item['description'] = 'Read original training states/actions at selected indices in a complete trajectory.'
        definitions.append(schema('read_overview',
            'Read evenly spaced original observations/actions from one complete training trajectory; choose denser source indices with read_steps.',
            dict(trajectory_id=dict(type='string',minLength=1),
                 frames=dict(type='integer',minimum=4,maximum=32))))
        return definitions

    def dispatch(self, name, args):
        if name == 'read_public' and args['name'] == 'INTERFACE.md':
            return dict(content=INTERFACE)
        if name == 'read_assignment':
            self.j.set('assignment_read', True)
            return dict(**self.entry, segments=read(self.entry['dataset'])['segments'],
                completion_contract=read(self.cfg['completion_contract']),
                training=self.cfg['training'], shared_normalizer=shared_normalizer(self.entry),
                task_description=read(self.folder/'task_description.json'))
        if name == 'read_overview':
            segments=read(self.entry['dataset'])['segments']
            matches=[s for s in segments if s['trajectory_id']==args['trajectory_id']]
            if len(matches)!=1: raise ValueError('Expected exactly one complete training trajectory')
            s=matches[0]
            indices=np.linspace(s['start'],s['stop'],args['frames'],dtype=int).tolist()
            return super().dispatch('read_steps',dict(trajectory_id=args['trajectory_id'],indices=indices))
        return super().dispatch(name,args)


def run(cfg, folder, gpu):
    from experiments.exp2.astra.transport import install_episode_gate
    folder=Path(folder)
    if (folder/'submission.json').exists(): return read(folder/'submission.json')
    if (folder/'design').exists(): raise RuntimeError('Existing design attempt requires explicit reconciliation')
    tools=FullTaskTools(cfg,folder,gpu)
    atomic(folder/'design/interface.json',dict(prompt=PROMPT,interface=INTERFACE,
        source_sha256=digest(__file__),policy_owner='Runtime API'))
    install_episode_gate(cfg['output'].parent,folder/'design',slots=4)
    c=client(cfg)
    original=c.respond
    def checked(request):
        if (request['model'],request.get('reasoning',{}).get('effort')) != ('gpt-6-astra','xhigh'):
            raise ValueError('Actual outgoing model/reasoning mismatch')
        return original(request)
    c.respond=checked
    try:
        AgentLoop(tools.j,tools,c,PROMPT).run(dict(policy_id=tools.entry['policy_id'],
            task='Design, implement, check and submit one full-task prior Diffusion Policy from the supplied training demonstrations.'))
        return read(folder/'submission.json')
    finally:
        tools.j.db.close()
        os.environ.pop('APPL_PRIOR_TOKEN',None)
