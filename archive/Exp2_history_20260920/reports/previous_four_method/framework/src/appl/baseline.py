"""M0/M1 fixed DP mechanism: no Agent prior, no extra controller."""
from .public import normalize_observation,normalize_action,denormalize_action,epsilon_loss


def build_model(factory,spec):
    return factory(spec['observation_dimension']*spec['training']['observation_steps'])


def condition(raw_history,spec):
    return normalize_observation(raw_history,spec).flatten(start_dim=1)


def encode_action(native,spec):return normalize_action(native,spec)


def decode_action(encoded,spec):return denormalize_action(encoded,spec)


def compute_loss(model,batch,spec):
    predicted=model(batch['noisy_action'],batch['timesteps'],condition(batch['raw_obs'],spec))
    return {'loss':epsilon_loss(predicted,batch['noise'],batch['mask'])}
