"""Use the fifth authorized GPU without changing the running coordinator."""
import argparse
import fcntl

from appl.io import ROOT, read
from appl.prior_policies.data import configuration
from appl.prior_policies.design import design
from appl.prior_policies.report import authorship
from appl.prior_policies.runner import train_submitted
from .runner import BASE, NAMES, entries, config, immutable_json, api_identity, unchanged_framework

RESOURCE_CONFIGS = ROOT / 'experiments/exp2/configs/astra_xhigh/resources_5gpu'


def resource_config(name):
    path = RESOURCE_CONFIGS / (name + '.json')
    original = read(config(name)['_path'])
    amended = read(path)
    if amended['devices'] != [1, 2, 3, 5, 7]:
        raise ValueError('Unexpected amended GPU pool')
    if {k:v for k,v in amended.items() if k != 'devices'} != {k:v for k,v in original.items() if k != 'devices'}:
        raise ValueError('Resource configuration changed scientific inputs')
    return configuration(path)


def main(shard):
    unchanged_framework()
    with (BASE / f'extra_gpu_{shard}.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Work from the unstarted end of the same declared portfolio. Existing
        # design/checkpoint locks arbitrate convergence with the original queues.
        for index, (name, entry) in reversed(list(enumerate(entries()))):
            if index % 2 != shard:
                continue
            cfg = resource_config(name)
            folder = cfg['output'] / 'policies' / entry['skill_id'] / f"heuristic_{entry['heuristic_index']:02d}"
            if (folder / 'failure.json').exists() or (folder / 'training/failure.json').exists():
                raise RuntimeError('Retained failure requires reconciliation: ' + str(folder))
            if (folder / 'training').exists():
                continue
            immutable_json(folder / 'assignment.json', entry)
            print(dict(stage='design',task=name,policy=entry['policy_id'],gpu=7),flush=True)
            submission = design(cfg, folder, 7)
            api_identity(folder / 'design/journal.sqlite')
            authorship(folder, submission)
            if (folder / 'training').exists():
                continue
            print(dict(stage='training',task=name,policy=entry['policy_id'],gpu=7),flush=True)
            train_submitted(cfg, entry['policy_id'], 7)
            print(dict(stage='trained',task=name,policy=entry['policy_id'],gpu=7),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard', type=int, choices=(0,1), required=True)
    main(parser.parse_args().shard)
