"""Compatibility imports; the sole DP trainer now lives in dp_baseline."""
from .dp_baseline.train import (
    module_at, tensor_hash, save, scheduler, deterministic_numerics,
    sample, fit, LoadedPolicy,
)
