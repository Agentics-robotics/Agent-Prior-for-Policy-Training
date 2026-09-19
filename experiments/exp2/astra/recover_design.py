"""Explicit, single-attempt recovery for individually authorized interruptions."""
import argparse
import fcntl
import sqlite3
import time

from appl.io import atomic, read, digest
from appl.prior_policies.design import design
from appl.prior_policies.report import authorship
from .extra_gpu import resource_config
from .runner import BASE, config, entries, api_identity, unchanged_framework

CASES = [('tray_pack', 'deliver_block__h02', 1, 0),
         ('two_block_sort', 'deliver_disengage__h01', 7, 6),
         ('unstack_sort', 'acquire_lift__h02', 8, 7)]


def main(requested=None):
    authorization = read(BASE / 'incidents/design_recovery_authorization.json')
    authorized = authorization['authorized_cases']
    available = {f'{name}/{policy}' for name,policy,_,_ in CASES}
    if not authorized or len(authorized) != len(set(authorized)) or not set(authorized) <= available:
        raise ValueError('Recovery scope must name specific recorded interruptions')
    if authorization['attempt_limit_per_case'] != 1:
        raise ValueError('Unexpected recovery limit')
    selected = authorized if requested is None else requested
    if not selected or len(selected) != len(set(selected)) or not set(selected) <= set(authorized):
        raise ValueError('Requested recovery subset must be explicitly authorized')
    unchanged_framework()
    # This failed workflow frees one design slot. Its original main queue cannot
    # issue later designs before reaching the unresolved sort case, recovered last.
    if read(BASE / 'preparation_overlap/0/process_result.json')['returncode'] == 0:
        raise ValueError('Expected the recorded interrupted preparation workflow')
    for category in ('preparation_continuation', 'preparation_continuation_after_interface_failure'):
        continuation = BASE / category / '0'
        if continuation.exists() and not (continuation / 'process_result.json').exists():
            raise ValueError('The replacement preparation workflow has not yielded its API slot yet')
    for name, policy_id, old_calls, old_tools in CASES:
        if f'{name}/{policy_id}' not in selected:
            continue
        cfg = resource_config(name)
        entry, = [e for n,e in entries() if n == name and e['policy_id'] == policy_id]
        folder = cfg['output'] / 'policies' / entry['skill_id'] / f"heuristic_{entry['heuristic_index']:02d}"
        retained = BASE / 'retained_design_attempts' / name / policy_id / 'attempt_0'
        receipt = BASE / 'incidents/design_recovery' / (name + '__' + policy_id + '.json')
        if retained.exists() or receipt.exists():
            raise ValueError('Recovery already attempted; no implicit repeat')
        if (folder / 'submission.json').exists() or list((folder / 'design/work').iterdir()):
            raise ValueError('This recovery is limited to attempts with no generated source')
        if read(folder / 'assignment.json') != entry:
            raise ValueError('The original heuristic assignment changed')
        states = api_identity(folder / 'design/journal.sqlite', require_consumed=False)
        if sum(row['calls'] for row in states) != old_calls or sum(row['calls'] for row in states if row['status'] == 'http_failed') != 1:
            raise ValueError('Original transport evidence differs from the declared case')
        db = sqlite3.connect((folder / 'design/journal.sqlite').resolve().as_uri() + '?mode=ro', uri=True)
        try:
            if db.execute('SELECT COUNT(*) FROM tools').fetchone()[0] != old_tools:
                raise ValueError('Original tool accounting differs from the declared case')
            if db.execute("SELECT COUNT(*) FROM tools WHERE status!='completed' OR name NOT IN ('read_assignment','read_public','read_steps')").fetchone()[0]:
                raise ValueError('The interrupted session contains non-read or unresolved tools')
        finally:
            db.close()
        retained.parent.mkdir(parents=True, exist_ok=True)
        # Keep the old lock while moving the complete attempt. A queued owner of
        # that inode is released only after the replacement session has finished.
        with (folder / 'design/package.lock').open('a') as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            before = {str(p.relative_to(folder)):digest(p) for p in folder.rglob('*') if p.is_file()}
            limits = dict(cfg['design'])
            limits['max_api_calls'] -= old_calls
            limits['max_tool_calls'] -= old_tools
            if min(limits['max_api_calls'], limits['max_tool_calls']) <= 0:
                raise ValueError('No declared design budget remains')
            cfg['design'] = limits
            atomic(receipt, dict(status='starting',time=time.time(),policy=policy_id,task=name,
                retained_attempt=str(retained),original_files=before,effective_remaining_limits=limits,
                prior_API_attempts=old_calls,prior_tool_calls=old_tools,
                authorization_sha256=digest(BASE / 'incidents/design_recovery_authorization.json'),
                recovery_source_sha256=digest(__file__),automatic_retry=False))
            folder.rename(retained)
            atomic(folder / 'assignment.json', entry)
            if before != {str(p.relative_to(retained)):digest(p) for p in retained.rglob('*') if p.is_file()}:
                raise ValueError('Retained interrupted attempt changed')
            print(dict(stage='authorized_fresh_design',task=name,policy=policy_id,limits=limits),flush=True)
            try:
                submission = design(cfg, folder, 7)
                api_identity(folder / 'design/journal.sqlite')
                authorship(folder, submission)
            except Exception as error:
                atomic(receipt, dict(read(receipt),status='failed',error=str(error),finished=time.time()))
                raise
            atomic(receipt, dict(read(receipt),status='submitted',version=submission['version'],finished=time.time()))
            print(dict(stage='recovery_submitted',task=name,policy=policy_id),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='append', choices=[f'{n}/{p}' for n,p,_,_ in CASES])
    main(parser.parse_args().case)
