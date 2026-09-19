"""One JSON configuration path for commands, workers and resumption."""
from pathlib import Path
from .io import ROOT, read, object_hash


def load(path=None):
    path = Path(path or ROOT/'experiments/exp2/configs/main.json').resolve()
    cfg = read(path)
    assert cfg['schema'] == 'appl.exp2.v2'
    assert set(cfg['devices']['allowed']) == {4,5,6,7}
    assert cfg['training']['execution_steps'] <= cfg['training']['horizon'] - cfg['training']['observation_steps'] + 1
    assert not set(cfg['data']['train_ids']) & set(cfg['data']['validation_ids'])
    assert cfg['training']['updates'] > 0 and cfg['training']['batch_size'] > 0
    cfg['_path'] = str(path)
    cfg['_hash'] = object_hash(read(path))
    cfg['_root'] = ROOT
    for key in ('directory',):
        cfg['data'][key] = (ROOT/cfg['data'][key]).resolve()
    cfg['output'] = (ROOT/cfg['output']).resolve()
    return cfg
