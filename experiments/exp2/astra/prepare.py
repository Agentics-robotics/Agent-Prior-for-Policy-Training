"""Prepare future API packages alongside training using existing package locks."""
import argparse

from appl.prior_policies.data import catalog
from appl.prior_policies.runner import prepare_queue
from .runner import DEVICES, NAMES, config, unchanged_framework


def main(shard):
    unchanged_framework()
    gpu = DEVICES[shard]
    offset = 0
    for name in NAMES:
        cfg = config(name)
        # Match the coordinator's global round-robin assignment exactly.
        local_shard = (shard - offset) % len(DEVICES)
        print(dict(task=name, gpu=gpu, local_shard=local_shard), flush=True)
        prepare_queue(cfg, gpu, local_shard, len(DEVICES))
        offset += len(catalog(cfg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard', type=int, required=True, choices=range(len(DEVICES)))
    main(parser.parse_args().shard)
