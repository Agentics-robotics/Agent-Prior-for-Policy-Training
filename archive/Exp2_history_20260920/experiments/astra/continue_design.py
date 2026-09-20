"""One explicitly authorized continuation after the recorded reload-gate repair."""
import fcntl
import json
import shutil
import sqlite3
import time

from appl.io import atomic, read, digest, object_hash
from appl.journal import Journal
from appl.prior_policies.design import _design_locked
from appl.prior_policies.report import authorship
from .extra_gpu import resource_config
from .runner import BASE, api_identity, unchanged_framework


def main():
    incident = BASE / 'incidents/buffer_red_h02_reload'
    authorization = read(incident / 'recovery_authorization.json')
    expected = dict(authorized=True, policy='buffer_swap/buffer_red__h02',
                    additional_API_calls=8, additional_interface_checks=2,
                    max_api_calls=44, max_checks=7, max_tool_calls=100,
                    continue_same_API_history=True, formal_updates=20000)
    if any(authorization.get(k) != v for k, v in expected.items()):
        raise ValueError('Explicit bounded continuation authorization is required')
    framework = unchanged_framework()
    cfg = resource_config('buffer_swap')
    if cfg['design']['max_api_calls'] != 36 or cfg['design']['max_checks'] != 5:
        raise ValueError('Original design limits changed')
    cfg['design'] = dict(cfg['design'], max_api_calls=44, max_checks=7)
    folder = cfg['output'] / 'policies/buffer_red/heuristic_02'
    receipt = incident / 'continuation.json'
    snapshot = incident / 'before_continuation'
    if receipt.exists() or snapshot.exists() or (folder / 'submission.json').exists():
        raise ValueError('Continuation already attempted or policy already submitted')
    with (folder / 'design/package.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (folder / 'design/agent.lock').open('a') as active:
            fcntl.flock(active, fcntl.LOCK_EX | fcntl.LOCK_NB)
            journal = Journal(folder / 'design')
            try:
                before_api = [tuple(r) for r in journal.db.execute('SELECT * FROM api ORDER BY seq')]
                before_tools = [tuple(r) for r in journal.db.execute('SELECT * FROM tools ORDER BY id')]
                if len(before_api) != 36 or any(r[1] != 'consumed' for r in before_api):
                    raise ValueError('Continuation requires exactly 36 consumed original calls')
                if journal.count('interface_check') != 5 or any(r[3] != 'completed' for r in before_tools):
                    raise ValueError('Original check/tool accounting changed or a tool is unresolved')
                history = journal.get('history')
                if not history or journal.get('submitted'):
                    raise ValueError('Expected an unsubmitted, fully consumed API session')
                snapshot.mkdir()
                with sqlite3.connect(snapshot / 'journal.sqlite') as backup:
                    journal.db.backup(backup)
                shutil.copytree(folder / 'design/work', snapshot / 'work')
                atomic(snapshot / 'history.json', history)
                note = dict(role='user', content=(
                    'Framework incident update, authored by the experiment executor: '
                    'the EMA reload verification had mismatched requires_grad flags between '
                    'the EMA instance and the restored comparison model. With identical weights '
                    'this caused a numerical reload mismatch. A zero-policy-update diagnostic '
                    'verified that matching the flags removes the mismatch. The framework now '
                    'matches those flags in the reload check; the sampler, optimizer, saved '
                    'weights, deployment implementation and 1e-6 tolerance are unchanged. '
                    'Your original API outputs and current unsubmitted files have been preserved '
                    'without executor edits. The user authorized continuation of this same '
                    'session with at most 8 additional API calls and 2 additional interface '
                    'checks (44 and 7 total); the total tool-call limit remains 100. '
                    'Validate your current package using check_policy. You retain ownership '
                    'of any necessary source changes. Explicitly submit the exact successfully '
                    'checked package hash using submit_policy. No performance feedback or '
                    'additional final-training updates are available.'))
                atomic(receipt, dict(status='starting', time=time.time(), framework=framework,
                    authorization_sha256=digest(incident / 'recovery_authorization.json'),
                    source_sha256=digest(__file__), original_api_hash=object_hash(before_api),
                    original_tool_hash=object_hash(before_tools), original_history_hash=object_hash(history),
                    original_work={p.name:digest(p) for p in (folder / 'design/work').iterdir() if p.is_file()},
                    effective_limits=cfg['design'], gpu=7, appended_framework_message=note,
                    automatic_retry=False, original_api_calls=36, original_checks=5))
                journal.set('history', history + [note])
                journal.event('authorized_framework_repair_continuation',
                    authorization_sha256=digest(incident / 'recovery_authorization.json'),
                    added_API_calls=8, added_interface_checks=2, policy_output_edits=0)
            finally:
                journal.db.close()
        print(dict(stage='authorized_same_session_continuation', policy=expected['policy']), flush=True)
        try:
            submission = _design_locked(cfg, folder, 7)
            api_identity(folder / 'design/journal.sqlite')
            authorship(folder, submission)
            with sqlite3.connect((folder / 'design/journal.sqlite').resolve().as_uri() + '?mode=ro', uri=True) as db:
                after_api = db.execute('SELECT * FROM api WHERE seq<=36 ORDER BY seq').fetchall()
                after_tools = {r[0]:r for r in db.execute('SELECT * FROM tools')}
                if after_api != before_api or any(after_tools[r[0]] != r for r in before_tools):
                    raise ValueError('Original API or tool output changed during continuation')
                final_history = json.loads(db.execute("SELECT value FROM state WHERE key='history'").fetchone()[0])
                if final_history[:len(history)] != history or final_history[len(history)] != note:
                    raise ValueError('Original conversation prefix changed')
        except Exception as error:
            atomic(receipt, dict(read(receipt), status='failed', error=str(error), finished=time.time()))
            raise
        atomic(receipt, dict(read(receipt), status='submitted', version=submission['version'],
                            original_history_preserved=True, finished=time.time()))
        print(dict(stage='continuation_submitted', policy=expected['policy']), flush=True)


if __name__ == '__main__':
    main()
