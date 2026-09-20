"""Prepare untouched work left after a design queue's recorded interruption."""
import argparse

from appl.io import read
from appl.prior_policies.design import design
from .runner import BASE, DEVICES, config, entries, immutable_json, api_identity, unchanged_framework


def main(shard):
    unchanged_framework()
    if read(BASE / 'preparation_overlap' / str(shard) / 'process_result.json')['returncode'] == 0:
        raise ValueError('This continuation requires a recorded interrupted preparation queue')
    for index, (name, entry) in enumerate(entries()):
        if index % len(DEVICES) != shard:
            continue
        if (BASE / 'incidents/design_recovery_authorization.json').exists():
            print(dict(status='yielded_design_slot_for_authorized_recovery'),flush=True)
            return
        cfg=config(name)
        folder=cfg['output'] / 'policies' / entry['skill_id'] / f"heuristic_{entry['heuristic_index']:02d}"
        if (folder / 'submission.json').exists():
            continue
        if (folder / 'design/journal.sqlite').exists() or (folder / 'failure.json').exists():
            print(dict(status='retained_existing_unsubmitted_session',task=name,policy=entry['policy_id']),flush=True)
            continue
        immutable_json(folder / 'assignment.json', entry)
        print(dict(stage='design_untouched_package',task=name,policy=entry['policy_id'],gpu=DEVICES[shard]),flush=True)
        design(cfg, folder, DEVICES[shard])
        api_identity(folder / 'design/journal.sqlite')
        print(dict(stage='submitted',task=name,policy=entry['policy_id']),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard',type=int,required=True,choices=range(len(DEVICES)))
    main(parser.parse_args().shard)
