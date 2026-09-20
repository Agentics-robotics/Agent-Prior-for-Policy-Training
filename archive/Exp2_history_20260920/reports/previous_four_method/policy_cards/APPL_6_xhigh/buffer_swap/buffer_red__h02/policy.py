"""Public hooks for the learned contact-belief action diffusion policy.

The complete learned model and masked losses are in implementation.py; the
explicit standard GRU cell is in recurrent.py. There are no runtime diagnostics,
scripted actions, replay paths or retained cross-invocation hidden states.
"""
from implementation import build_model, compute_loss
