"""Public numerical hooks and High-level Agent entry points.
Explicit top-level definitions satisfy the executor's static hook contract.
"""
import data_pipeline
import networks
import runtime


def prepare_data(spec):
    return data_pipeline.prepare_data(spec)


def make_batch(policy_id, arrays, metadata, indices, spec):
    return data_pipeline.make_batch(policy_id, arrays, metadata, indices, spec)


def build_model(policy_id, spec):
    return networks.build_model(policy_id, spec)


def compute_loss(model, batch, spec):
    return networks.compute_loss(model, batch, spec)


def predict(model, batch, spec):
    return networks.predict(model, batch, spec)


def act(model, observation, call, memory, spec):
    return runtime.act(model, observation, call, memory, spec)


def decode_action(action, controller_contract, spec):
    return runtime.decode_action(action, controller_contract, spec)


def observe_scene(observation, scene_version, scene_memory=None):
    return runtime.observe_scene(observation, scene_version, scene_memory)


def preview_goal(template_ref, spatial_goal, workspace_polygon_uv):
    return runtime.preview_goal(template_ref, spatial_goal, workspace_polygon_uv)


def reexpress_goal(new_template, old_goal_mask):
    return runtime.reexpress_goal(new_template, old_goal_mask)
