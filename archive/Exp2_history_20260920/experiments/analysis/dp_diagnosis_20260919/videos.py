"""Make synchronized first-reset diagnostic comparisons from retained frames."""
import html
import json
import subprocess
from appl.io import read,atomic,digest
from .run import BASE,TASKS


def main():
    output=BASE/'videos';output.mkdir(exist_ok=True);records=[];sections=[]
    for task in TASKS:
        ids=list(read(BASE/'plan.json')['tasks'][task]['demonstrations']);identifier=ids[0]
        paths=[BASE/'runs'/task/identifier/'expert_replay/replay.mp4',
            BASE/'runs'/task/identifier/'frozen_DP/replay.mp4',BASE/'binary_gripper'/task/identifier/'replay.mp4']
        duration=max(float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',str(p)],text=True)) for p in paths)
        target=output/(task+'.mp4')
        if not target.exists():
            command=['ffmpeg','-v','error','-threads','1','-filter_complex_threads','1']
            for path in paths:command+=['-i',str(path)]
            filters=[]
            for i,label in enumerate(['Expert replay','Original frozen DP','Same DP + binary grip']):
                filters.append(f"[{i}:v]scale=256:256:flags=neighbor,tpad=stop_mode=clone:stop_duration=60,drawtext=text='{label}':fontsize=15:fontcolor=white:box=1:boxcolor=black@0.8:x=5:y=5[v{i}]")
            filters.append('[v0][v1][v2]hstack=inputs=3[v]')
            command+=['-filter_complex',';'.join(filters),'-map','[v]','-t',str(duration),'-c:v','libx264','-threads','1','-pix_fmt','yuv420p',str(target)]
            subprocess.run(command,check=True)
        record=dict(task=task,demonstration=identifier,path=str(target.relative_to(BASE)),sha256=digest(target),duration_seconds=duration,source_sha256=[digest(p) for p in paths])
        records.append(record)
        sections.append(f'<section><h2>{html.escape(task)} / {identifier}</h2><video controls preload="metadata" width="768" src="{record["path"]}"></video></section>')
    atomic(BASE/'videos.json',records)
    (BASE/'videos.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Frozen DP diagnosis</title><style>body{font:16px system-ui;max-width:1000px;margin:32px auto;background:#fafafa;color:#222}video{max-width:100%}section{margin:36px 0}</style><h1>DP training-reset diagnostic comparisons</h1><p>Same original demonstration reset. Columns: recorded expert action replay, frozen naive DP, and the same frozen DP with only gripper commands mapped to ±1. Playback is approximately 6× simulation speed; finished clips hold their final frame. Policy diagnostics have a 1,500-step cap. These are selected training-reset examples, not held-out performance estimates.</p>'+''.join(sections)+'</html>')
    print(records)


if __name__=='__main__':main()
