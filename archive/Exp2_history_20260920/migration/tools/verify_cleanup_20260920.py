"""Verify the migration without changing any scientific artifact."""
import json
from pathlib import Path
from experiments.exp2.maintenance.cleanup_20260920 import ROOT, RECEIPTS, read, write, sha


def check_inventory(name):
    rows=read(RECEIPTS/name)
    problems=[]
    for row in rows:
        p=Path(row['path'])
        if 'link' in row:
            if not p.is_symlink() or p.readlink().as_posix()!=row['link']:
                problems.append(str(p))
        else:
            if not p.is_file():
                problems.append(str(p));continue
            s=p.stat()
            actual=dict(size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev,mode=s.st_mode)
            if any(row[k]!=v for k,v in actual.items()): problems.append(str(p))
    assert not problems, (name,problems[:20])
    return len(rows)


if __name__=='__main__':
    counts={name:check_inventory(name) for name in ['exp1_before.json','runtime_before.json','data_before.json']}
    frozen=read(RECEIPTS/'frozen_sha256.json')
    for path,h in frozen.items(): assert sha(ROOT/path)==h, path
    # Official frozen-plan checks: imports only, no model worker/API/simulation.
    from experiments.exp2.exp2_new.common import verify as verify_appl
    from experiments.exp2.binary_gripper.common import verify as verify_baselines
    from experiments.exp2.single_policy.common import verify_preparation
    verify_appl(); verify_baselines(); verify_preparation()
    selected=read(RECEIPTS/'current_validation.json')['validation']
    assert selected['passed']
    write(RECEIPTS/'validation.json',dict(passed=True,inventory_entries=counts,
        preserved_frozen_file_hashes=len(frozen),selected=selected,
        frozen_plan_checks=['exp2_new','binary_gripper','single_policy'],
        preservation_evidence='All inventoried regular files retain device, inode, size, mtime and mode through original compatibility paths. Frozen source and selected result/trace/checkpoint contents additionally match SHA-256.',
        API_calls=0,training_updates=0,simulator_episodes=0))
    print(json.dumps(read(RECEIPTS/'validation.json'),indent=2))
