"""Read exact frozen input tensors and export bounded visualization data/figures."""

import base64
import io
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from appl.io import ROOT, read, digest
from real_robot.data import Sources


DEST = ROOT / "real_robot/reports/input_conditions_v3"
VIZ = Path("/home/users/oscar/.codex/visualizations/2026/09/21/01a0c26a-037e-71e0-a79c-45e9bde2019d")
CHANNELS = [
    ("当前物体 SDF", "Current object SDF", -1, 1, "距离 / 0.08 m；负值在内"),
    ("目标物体 SDF", "Goal object SDF", -1, 1, "距离 / 0.08 m；负值在内"),
    ("其他物体", "Other-object occupancy", 0, 1, "0–1 占据"),
    ("TCP 热图", "TCP heatmap", 0, 1, "0–1 强度"),
    ("目标流 X", "Goal correspondence X", -2, 2, "位移 / 0.20 m"),
    ("目标流 Y", "Goal correspondence Y", -2, 2, "位移 / 0.20 m"),
    ("全场景占据", "Coarse whole-scene occupancy", 0, 1, "1 m × 1 m 场景"),
    ("目标有效性", "Object-goal validity", 0, 1, "工具模型此通道为 0"),
]
GROUPS = [
    (0,7,"关节 q / 3"),(7,14,"q 有效性"),(14,21,"实测 dq"),(21,28,"dq 有效性"),
    (28,35,"tau_ext / 10"),(35,42,"tau 有效性"),(42,51,"末端旋转矩阵"),
    (51,54,"法兰−TCP 平移"),(54,60,"上一条实际动作"),(60,62,"双相机图像年龄"),
    (62,64,"夹爪反馈有效性"),(64,67,"TCP 目标误差 / 0.3 m"),
    (67,70,"TCP 目标旋转误差"),(70,76,"目标终止动作"),(76,77,"TCP 的 W 高度"),
    (77,79,"TCP−物体中心 / 0.2 m"),(79,81,"物体目标位移 / 0.2 m"),
    (81,83,"目标转角 sin / cos"),(83,84,"跟踪可信度"),(84,85,"物体目标有效"),
    (85,86,"目标形状配准得分"),(86,87,"对称角度歧义"),(87,88,"平面余量 / 0.03 m"),
    (88,89,"接触距离近似 / 0.1 m"),(89,90,"路径受阻标记"),(90,91,"几何退出标记"),
    (91,112,"零填充：未使用"),
]


def pack_maps(maps):
    """Lossless uint16 PackBits; high tag bit indicates a repeated word."""
    a = np.frombuffer(maps.tobytes(), dtype='<u2')
    chunks = []
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[j] == a[i] and j-i < 32767:
            j += 1
        if j-i >= 3:
            chunks.append(struct.pack('<HH', 32768 | (j-i), int(a[i])))
            i = j
        else:
            start = i
            i = j
            while i < len(a) and i-start < 32767:
                if i+2 < len(a) and a[i] == a[i+1] == a[i+2]:
                    break
                i += 1
            chunks.extend((struct.pack('<H', i-start), a[start:i].tobytes()))
    packed = b''.join(chunks)
    words = np.frombuffer(packed, dtype='<u2')
    out = []; i = 0
    while i < len(words):
        tag = int(words[i]); i += 1
        n = tag & 32767
        if tag & 32768:
            out.extend([int(words[i])] * n); i += 1
        else:
            out.extend(words[i:i+n]); i += n
    assert np.asarray(out, dtype='<u2').tobytes() == maps.tobytes()
    return base64.b64encode(packed).decode()


def thumbnail(path):
    with Image.open(path) as im:
        im = im.convert('RGB'); im.thumbnail((256,144))
        buff = io.BytesIO(); im.save(buff,format='JPEG',quality=48,optimize=True)
    return 'data:image/jpeg;base64,' + base64.b64encode(buff.getvalue()).decode()


def main():
    DEST.mkdir(exist_ok=True)
    VIZ.mkdir(parents=True, exist_ok=True)
    cfg = read(ROOT / "real_robot/configs/push_training_v3.json")
    package = ROOT / cfg["run"] / "package_00"
    cache = package / "prepared"
    manifest, meta = read(cache / "result.json"), read(cache / "metadata.json")
    arrays = {k: np.load(cache / v["file"], mmap_mode="r", allow_pickle=False)
              for k,v in manifest["arrays"].items()}
    sources = Sources(read(ROOT / cfg["cut_config"]))
    bundle = dict(channels=[dict(name=z,en=e,lo=l,hi=h,units=u) for z,e,l,h,u in CHANNELS],
                  groups=[dict(start=s,stop=e,label=l) for s,e,l in GROUPS],
                  offsets=[11,7,3,0], samples={})
    receipts = []
    for kind, sid in (("tool","a_l_approach_tool"),("piece","a_l_move"),("zero_goal","a_d_stage")):
        segnum, segment = next((i,s) for i,s in enumerate(meta["source_segments"]) if s["segment_id"]==sid)
        index = (segment["supervised_start"]+segment["supervised_stop"]-1)//2
        rows = np.flatnonzero((arrays["example_segment"]==segnum)&(arrays["example_source_index"]==index))
        assert len(rows)==1
        row=int(rows[0]); history=np.asarray(arrays["history"][row]); tid=segment["trajectory_id"]
        samples=[]; hashes=[]
        for hi,(frame,offset) in enumerate(zip(history,bundle["offsets"],strict=True)):
            maps=np.asarray(arrays["f_maps"][frame],dtype='<f2')
            state=np.asarray(arrays["f_state"][frame],dtype=np.float32)
            assert np.isfinite(maps).all() and np.isfinite(state).all()
            sample_index=max(segment["start"],index-offset)
            photos={}
            for camera in ('third','wrist'):
                path=sources.media_path(tid,camera,sample_index)
                photos[camera]=thumbnail(path)
                hashes.append(dict(camera=camera,index=sample_index,sha256=digest(path)))
            samples.append(dict(index=int(sample_index),cache_frame=int(frame),
                                valid=bool(arrays['history_valid'][row,hi]),
                                maps=pack_maps(maps),
                                state=[float(v) for v in state],photos=photos))
        goal_path=sources.media_path(tid,'third',segment['supervised_stop']-1)
        bundle['samples'][kind]=dict(segment=sid,trajectory=tid,index=index,anchor=row,
            frames=samples,nodes=np.asarray(arrays['f_nodes'][history[-1]],float).tolist(),
            score=np.asarray(arrays['f_score'][history[-1]],float).tolist(),
            base=np.asarray(arrays['f_base'][history[-1]],float).tolist(),
            basis=np.asarray(arrays['f_basis'][history[-1]],float).tolist(),
            target_handle=int(meta['coverage'][segnum]['target_visual_handle']),
            goal_source_index=segment['supervised_stop']-1,
            goal_photo=thumbnail(goal_path),
            object_goal_valid=bool(arrays['goal_valid'][row]),
            weak_phase=['approach','contact','reset','exit','hold'][int(arrays['phase'][row])])
        # Standalone scientific heatmaps use the exact current cached values.
        current=np.asarray(arrays['f_maps'][history[-1]],float)
        fig,axs=plt.subplots(2,4,figsize=(13.5,7),layout='constrained')
        for k,ax in enumerate(axs.flat):
            _,en,lo,up,_=CHANNELS[k]
            plot=ax.imshow(current[k],origin='upper',interpolation='nearest',vmin=lo,vmax=up,
                           cmap='RdBu_r' if lo<0 else 'viridis')
            ax.set_title(f'{k}: {en}',fontsize=10)
            ax.set_xlabel('column (cell)');ax.set_ylabel('row (cell)')
            fig.colorbar(plot,ax=ax,shrink=.65)
        fig.suptitle(f'Actual training input | {sid} | paired row {index} | t',fontsize=13)
        fig.savefig(DEST/f'{kind}_channels.png',dpi=150)
        plt.close(fig)
        receipts.append(dict(kind=kind,segment=sid,source_index=index,cache_anchor=row,
                             cache_history=history.tolist(),selection='Midpoint of specified supervised range; no performance selection',
                             RGB_sources=hashes,goal_source_index=segment['supervised_stop']-1,
                             goal_RGB_sha256=digest(goal_path)))
    encoded=json.dumps(bundle,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    (DEST/'data.json').write_text(encoded)
    receipt=dict(date='2026-09-21',kind='Exact cached neural input visualization',
                 examples=receipts,source_package=package.name,
                 cache_manifest_sha256=digest(cache/'result.json'),
                 input_map_dtype='float16; lossless little-endian uint16 PackBits; roundtrip verified for every map',
                 raw_RGB_display='256x144 JPEG thumbnail; model preprocessing used original 1280x720 RGB',
                 network_rerun=False,preprocessing_rerun=False,API_calls=0,robot_actions=0,
                 scientific_source_modified=False)
    (DEST/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    template=DEST/'template.html'
    if template.exists():
        html=template.read_text().replace('__INPUT_DATA__',encoded)
        assert len(html.encode())<1_000_000
        (VIZ/'push-v3-input-conditions.html').write_text(html)
    print(json.dumps(dict(data_bytes=len(encoded.encode()),samples=receipts),ensure_ascii=False))


if __name__=='__main__':main()
