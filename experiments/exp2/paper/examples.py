"""Copy recorded first-seed examples into the portable paper bundle."""
import html
from pathlib import Path
import shutil

from appl.io import atomic, digest, read
from experiments.exp2.single_policy.common import config
from .build import OUT, METHODS, NAMES, LABELS, TASK_LABELS, repo_path, table


def export(episodes):
    records=[]
    for task in NAMES:
        for condition in ('ID','OOD'):
            seed=config(task)['evaluation']['seeds' if condition=='ID' else 'ood_seeds'][0]
            for method in METHODS:
                row=next(r for r in episodes if (r['task'],r['condition'],r['seed'],r['method'])
                         ==(task,condition,seed,method))
                assert row['completed'], 'First-seed example has no complete outcome'
                root=Path(row['episode_root'])
                destination=OUT/'execution_examples'/task/condition/method
                destination.mkdir(parents=True,exist_ok=True)
                files={}
                selected=[('initial_state.json','initial_state.json'),('result.json','result.json'),
                          ('replay.mp4','replay.mp4'),('initial.png','initial.png'),('final.png','final.png')]
                if method.startswith('APPL_'):
                    selected += [('invocations.json','invocations.json'),
                                 ('api/api/0001.request.json','initial_API_request.json')]
                for relative,name in selected:
                    source=root/relative;dest=destination/name
                    shutil.copyfile(source,dest)
                    sha=digest(source);assert sha==digest(dest)
                    files[name]=dict(source=repo_path(source),sha256=sha)
                if method.startswith('APPL_'):
                    responses=sorted((root/'api/api').glob('*.response.json'))
                    last=read(responses[-1])['response']
                    # Select public tool-call objects; retain every selected object
                    # and its argument string unchanged, without private reasoning.
                    calls=[r for r in last['output'] if r['type']=='function_call']
                    atomic(destination/'last_API_function_calls.json',dict(
                        source=repo_path(responses[-1]),source_sha256=digest(responses[-1]),
                        extraction='All function_call objects in the final recorded response, unchanged.',
                        function_calls=calls))
                records.append(dict(task=task,condition=condition,seed=seed,method=method,
                    success=row['success'],steps=row['steps'],status=row['status'],
                    source=repo_path(root),bundle_folder=str(destination.relative_to(OUT)),files=files))
    assert len(records)==40
    atomic(OUT/'execution_examples/index.json',dict(
        selection='First predeclared seed in each task/condition, for every method; no outcome selection.',
        API_outputs='Original invocation records and initial requests copied byte-for-byte. Last function calls are an explicitly labeled unchanged-object extraction.',
        videos='Original recorded 128x128 camera frames, approximately 6x simulated speed; no new simulation or rerendering.',
        examples=records))
    table(OUT/'tables/execution_examples.csv',[{k:v for k,v in r.items() if k!='files'} for r in records])
    page=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Exp2 paired recorded examples</title>',
          '<style>body{font:16px system-ui;margin:24px;max-width:1400px}.row{display:flex;gap:16px;flex-wrap:wrap}article{width:300px}video{width:280px}h2{margin-top:40px}</style>',
          '<h1>Four-method paired recorded examples</h1>',
          '<p>First predeclared seed for each task and condition. Examples are not selected by outcome. '
          'Original camera frames; approximately 6x simulated speed. Counts belong to the full report.</p>']
    for task in NAMES:
        for condition in ('ID','OOD'):
            page += ['<h2>'+TASK_LABELS[task]+' / '+condition+'</h2><div class="row">']
            for method in METHODS:
                row=next(r for r in records if (r['task'],r['condition'],r['method'])==(task,condition,method))
                folder=html.escape(str(Path(row['bundle_folder']).relative_to('execution_examples')))
                page += ['<article><h3>'+LABELS[method]+'</h3>',
                         f"<p>Seed {row['seed']}; success={row['success']}; {row['steps']} steps</p>",
                         f'<video controls preload="none" src="{folder}/replay.mp4"></video>',
                         f'<p><a href="{folder}/result.json">Result</a>']
                if method.startswith('APPL_'):
                    page += [f' · <a href="{folder}/invocations.json">Exact invocations and notebook</a>']
                page += ['</p></article>']
            page += ['</div>']
    (OUT/'execution_examples/index.html').write_text('\n'.join(page)+'\n</html>\n')

