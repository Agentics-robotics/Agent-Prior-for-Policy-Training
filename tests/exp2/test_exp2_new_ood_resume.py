"""Resource contention must not terminate already running paid evaluations."""
from experiments.exp2.analysis.exp2_new_ood_resume import GPUS, available_devices


def test_busy_idle_slot_does_not_remove_other_active_slots():
    identities = {gpu: dict(uuid=f'GPU-test-{gpu}') for gpu in GPUS}
    active = {1: object(), 2: object()}
    before = active.copy()
    occupancy = '999, GPU-test-3, 500\n111, GPU-test-1, 4000\n'
    assert available_devices(active, occupancy, identities) == [4, 7]
    assert active == before


def test_all_slots_busy_is_a_wait_not_an_error():
    identities = {gpu: dict(uuid=f'GPU-test-{gpu}') for gpu in GPUS}
    occupancy = '\n'.join(f'999, GPU-test-{gpu}, 500' for gpu in GPUS)
    assert available_devices({}, occupancy, identities) == []
