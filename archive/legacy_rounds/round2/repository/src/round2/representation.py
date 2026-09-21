import numpy as np
from scipy.spatial.transform import Rotation
from relative_dp.dataset import Normalizer as BaseNormalizer, WindowDataset
from relative_dp.utils import ROOT, sha256, read_json

PASSTHROUGH = list(range(7, 18)) + list(range(25, 36)) + [39, 40]


def matrices(obs):
    a = np.arctan2(obs[..., 39], obs[..., 40])
    flat = Rotation.from_euler('z', a.reshape(-1)).as_matrix()
    return flat.reshape(a.shape + (3, 3))


def transform(raw, representation, inverse=False):
    raw = np.asarray(raw)
    if representation == 'world':
        return raw.copy()
    assert representation == 'frame' and raw.shape[-1] == 41
    out = raw.copy()
    R = matrices(raw)
    rotate = R if inverse else np.swapaxes(R, -1, -2)
    for b in (0, 18):
        hand, handle = raw[..., b:b+3], raw[..., b+4:b+7]
        out[..., b:b+3] = np.einsum('...ij,...j->...i', rotate, hand if inverse else hand-handle)
        if inverse:
            out[..., b:b+3] += handle
        qs = raw[..., b+7:b+11]
        qm = Rotation.from_quat(qs.reshape(-1, 4)).as_matrix().reshape(qs.shape[:-1] + (3, 3))
        qout = Rotation.from_matrix((rotate @ qm).reshape(-1, 3, 3)).as_quat().reshape(qs.shape)
        # Preserve the sign relative to the algebraic quaternion product.
        yaw_q = Rotation.from_matrix(rotate.reshape(-1, 3, 3)).as_quat().reshape(qs.shape)
        v1, w1, v2, w2 = yaw_q[..., :3], yaw_q[..., 3:4], qs[..., :3], qs[..., 3:4]
        ref = np.concatenate([w1*v2+w2*v1+np.cross(v1,v2), w1*w2-np.sum(v1*v2,axis=-1,keepdims=True)],axis=-1)
        qout *= np.where(np.sum(qout*ref,axis=-1,keepdims=True)<0,-1,1)
        out[..., b+7:b+11] = qout
    goal, h = raw[..., 36:39], raw[..., 4:7]
    out[..., 36:39] = np.einsum('...ij,...j->...i', rotate, goal if inverse else goal-h)
    if inverse:
        out[..., 36:39] += h
    return out


def action_transform(actions, obs, representation, inverse=False):
    result = np.asarray(actions).copy()
    if representation == 'frame':
        R = matrices(np.asarray(obs))
        if not inverse:
            R = np.swapaxes(R, -1, -2)
        result[..., :3] = np.einsum('...ij,...tj->...ti', R, result[..., :3])
    return result


class Normalizer(BaseNormalizer):
    @classmethod
    def fit(cls, episodes, representation, source_ids, source_hashes, floor=1e-3):
        x = np.concatenate([transform(e['obs'][:-1], representation) for e in episodes]).astype(np.float64)
        mean, std = x.mean(0), x.std(0)
        constant = np.flatnonzero(std < floor).tolist()
        std = np.maximum(std, floor)
        mean[PASSTHROUGH] = 0
        std[PASSTHROUGH] = 1
        return cls(mean.astype(np.float32), std.astype(np.float32), list(source_ids), dict(source_hashes), constant, representation, len(x))

    def normalize(self, raw):
        return (transform(np.asarray(raw, np.float32), self.representation) - self.mean) / self.std

    def as_dict(self):
        state = super().as_dict()
        state['passthrough_dimensions'] = PASSTHROUGH
        state['statistics_scope'] = 'Round2 train20 obs[0:T]; positions/scalars population std floor .001; quaternions/yaw/zero placeholders passthrough; no clipping'
        return state


def dataset(task, representation):
    path = ROOT / 'data/round2/manifest.json'
    manifest = read_json(path)
    records = manifest['tasks'][task]['train20']
    assert len(records) == 20
    episodes, hashes = [], {}
    for r in records:
        p = ROOT / r['path']
        assert sha256(p) == r['sha256']
        with np.load(p, allow_pickle=False) as f:
            episodes.append(dict(obs=f['obs'].copy(), actions=f['actions'].copy()))
        hashes[r['episode_id']] = r['sha256']
    normalizer = Normalizer.fit(episodes, representation, list(hashes), hashes)
    ds = WindowDataset(episodes, normalizer)
    if representation == 'frame':
        obs = np.stack([episodes[i]['obs'][t] for i,t in ds.index])
        import torch
        ds.actions = torch.from_numpy(action_transform(ds.actions.numpy(), obs, 'frame'))
    ds.manifest_hash = sha256(path)
    return ds
