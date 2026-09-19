"""Train ready submissions alongside the original queues without extra updates."""
import argparse
import fcntl
import time

from appl.gpu import identity
from appl.io import read
from appl.prior_policies.report import authorship
from appl.prior_policies.runner import train_submitted
from .runner import BASE, DEVICES, config, entries, api_identity, unchanged_framework


def main(shard):
    unchanged_framework()
    gpu = DEVICES[shard]
    assigned = [(name, entry) for index, (name, entry) in enumerate(entries())
                if index % len(DEVICES) == shard]
    with (BASE / f'assist_{shard}.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            pending = 0
            for name, entry in assigned:
                cfg = config(name)
                folder = cfg['output'] / 'policies' / entry['skill_id'] / f"heuristic_{entry['heuristic_index']:02d}"
                if (folder / 'failure.json').exists() or (folder / 'training/failure.json').exists():
                    raise RuntimeError('Retained failure requires reconciliation: ' + str(folder))
                if (folder / 'training/result.json').exists():
                    continue
                pending += 1
                if not (folder / 'submission.json').exists() or (folder / 'training').exists():
                    continue
                if identity(gpu)['free_mib'] < 10000:
                    continue
                # The original training lock serializes an occasional race with
                # the main queue; train_submitted reuses a completed checkpoint.
                identity_rows = api_identity(folder / 'design/journal.sqlite', require_consumed=False)
                if any(row['status'] != 'consumed' for row in identity_rows):
                    continue
                authorship(folder, read(folder / 'submission.json'))
                print(dict(stage='train_ready_submission',task=name,policy=entry['policy_id'],gpu=gpu),flush=True)
                train_submitted(cfg, entry['policy_id'], gpu)
                print(dict(stage='trained',task=name,policy=entry['policy_id']),flush=True)
            if not pending:
                return
            time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard', type=int, required=True, choices=range(len(DEVICES)))
    main(parser.parse_args().shard)
