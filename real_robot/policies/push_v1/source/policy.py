"""Frozen cut_v3 package entry points. Training execution belongs to the framework."""
import torch
from data_pipeline import prepare,batch
from networks import ContourPolicy,VisualPolicy,loss
from invocation import act as call_policy, inventory as observe_inventory
from action_gate import decode


def prepare_data(spec):
    return prepare(spec)


def make_batch(policy_id,arrays,metadata,indices,spec):
    if policy_id not in ('contour_push','visual_push'): raise ValueError('Unknown frozen policy')
    return batch(policy_id,arrays,metadata,indices,spec)


def build_model(policy_id,spec):
    meta=spec['data_metadata']
    if policy_id=='contour_push': return ContourPolicy(meta)
    if policy_id=='visual_push': return VisualPolicy(meta)
    raise ValueError('Unknown frozen policy')


def compute_loss(model,batch,spec):
    return loss(model,batch)


def predict(model,batch,spec):
    with torch.no_grad():
        return model.decode_mean(model(batch,augment=False))


def act(model,observation,call,memory,spec):
    return call_policy(model,observation,call,memory,spec)


def decode_action(action,controller_contract,spec):
    return decode(action,controller_contract,spec)


def inventory(observation,scene_version,memory=None):
    return observe_inventory(observation,scene_version,memory)
