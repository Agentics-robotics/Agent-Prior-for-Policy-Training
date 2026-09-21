"""Export first-predeclared ID/OOD pairs from completed original camera frames."""
from datetime import datetime, timezone
import html

from appl.io import read, atomic, digest
from appl.scaleup.protocol import BASE, NAMES, config
from appl.scaleup.report import video


def main():
    pairs = []
    sections = []
    for task in NAMES:
        cfg = config(task)
        for condition in ('ID', 'OOD'):
            seed = cfg['evaluation']['seeds' if condition == 'ID' else 'ood_seeds'][0]
            cards = []
            records = []
            for method in ('naive_DP', 'APPL'):
                root = BASE / task / 'evaluation' / method / condition / str(seed)
                result = read(root / 'result.json')
                process = read(BASE / 'batch/jobs' / task / method / condition / str(seed) / 'process_result.json')
                assert process['returncode'] == 0 and result['completed_evaluation']
                path = video(root)
                provenance = read(root / 'replay_provenance.json')
                assert digest(path) == provenance['video_sha256']
                assert all(digest(root / name) == sha for name, sha in provenance['frames'].items())
                label = f"{method} — {result['status']} — {result['steps']} physical steps"
                relative = str(path.relative_to(BASE))
                cards.append(f'<article><h3>{html.escape(label)}</h3><video controls preload="none" src="{html.escape(relative)}"></video></article>')
                records.append(dict(method=method, result_sha256=digest(root / 'result.json'),
                    video=relative, video_sha256=provenance['video_sha256'], frames=len(provenance['frames'])))
            pairs.append(dict(task=task, condition=condition, seed=seed, methods=records))
            sections.append(f'<section data-task="{task}" data-condition="{condition}"><h2>{task.replace("_", " ")} / {condition} / seed {seed}</h2><div class="pair">'+''.join(cards)+'</div></section>')
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>First paired policy replays</title><style>
body{font-family:system-ui;margin:24px auto;max-width:1100px;padding:0 20px;background:#f3f5f7;color:#17243a}
h1{font-size:28px}h2{font-size:20px}h3{font-size:15px}p{line-height:1.5}section{margin:28px 0}section[hidden]{display:none}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}article{background:white;padding:16px;border-radius:8px}video{width:100%;image-rendering:pixelated}
select{padding:8px;margin-right:12px}@media(max-width:650px){.pair{grid-template-columns:1fr}}
</style><h1>DP and APPL: first paired layouts</h1>
<p>Each pair uses the first predeclared seed in its task and condition. Examples were not selected for success or failure. These are individual trajectories, not success-rate estimates.</p>
<p>Original 128 × 128 camera snapshots, approximately 6× physical speed. Simulation pauses for API calls; this playback does not represent wall-clock execution latency.</p>
<p><a href="REPORT.md">Full-study report</a> · <a href="replays.html">All-trial replay browser (populated after the matrix finishes)</a></p>
<label>Task <select id="task" onchange="filter()"><option value="all">All tasks</option>'''
    document += ''.join(f'<option value="{n}">{n.replace("_", " ")}</option>' for n in NAMES)
    document += '</select></label><label>Condition <select id="condition" onchange="filter()"><option value="all">ID and OOD</option><option>ID</option><option>OOD</option></select></label>'
    document += ''.join(sections)
    document += '''<script>function filter(){const t=document.getElementById('task').value,c=document.getElementById('condition').value;document.querySelectorAll('section').forEach(s=>{s.hidden=(t!=='all'&&s.dataset.task!==t)||(c!=='all'&&s.dataset.condition!==c);});}</script></html>'''
    (BASE / 'paired_examples.html').write_text(document)
    atomic(BASE / 'paired_examples_provenance.json', dict(exported_utc=datetime.now(timezone.utc).isoformat(),
        selection='First predeclared ID and OOD seed for each task; all ten pairs included.', pairs=pairs,
        source_sha256=digest(__file__), new_physical_steps=0, new_api_requests=0))
    print(dict(pairs=len(pairs), videos=2 * len(pairs), browser=str(BASE / 'paired_examples.html')), flush=True)


if __name__ == '__main__':
    main()
