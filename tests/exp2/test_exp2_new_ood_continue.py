"""Continuation scope and concurrent monetary admission; no network or physics."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from appl.io import read
from experiments.exp2.exp2_new.budget import Ledger, BudgetStop
from experiments.exp2.analysis import exp2_new_ood_continue as continuation


def test_only_unfinished_ood_cells_selected_without_mutating_plan():
    cells = [dict(index=0, condition='ID'), dict(index=1, condition='OOD'),
             dict(index=2, condition='OOD'), dict(index=3, condition='ID')]
    plan = dict(cells=cells)
    assert continuation.select_pending(plan, {1}) == [cells[2]]
    assert plan['cells'] == cells and len(cells) == 4
    with pytest.raises(ValueError, match='OOD only'):
        continuation.episode_root(dict(condition='ID'))


def test_parallel_admission_keeps_all_reservations_under_one_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(continuation, 'OUTPUT', tmp_path)
    path = tmp_path / 'budget'
    continuation.ContinuationLedger().initialize('1', 'Concurrent admission test fixture')

    def reserve(index):
        try:
            continuation.ContinuationLedger().reserve(str(index), 1000, 4096)
            return True
        except BudgetStop:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        accepted = list(pool.map(reserve, range(8)))
    value = read(path / 'ledger.json')
    assert sum(accepted) == len(value['records']) == 4
    assert sum(r['reserved_nano_usd'] for r in value['records']) <= value['cap_nano_usd']
    assert value['stopped']


def test_new_ledger_is_isolated_from_previous_stopped_round(tmp_path, monkeypatch):
    original = tmp_path / 'original'
    old = Ledger(original / 'budget'); old.initialize('78', 'Old fixture'); old.stop('Original budget stop')
    original_bytes = old.path.read_bytes()
    monkeypatch.setattr(continuation, 'OUTPUT', original / 'continuation')
    current = continuation.ContinuationLedger(); current.initialize('390', 'New authorized fixture')
    current.reserve('new/0001', 1000, 4096)
    assert old.path.read_bytes() == original_bytes
    assert read(current.path)['stopped'] is None
    assert len(read(current.path)['records']) == 1
