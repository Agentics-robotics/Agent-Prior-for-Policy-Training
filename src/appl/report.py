"""Central evidence tables; missing/failed stages are never promoted to success."""
from pathlib import Path
import csv
import json
import sqlite3
import time
import numpy as np
from .io import ROOT,read,atomic,digest
from .protocol import gates


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:path.write_text('');return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def api_cost(path):
    db=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True)
    events=[json.loads(row[0]) for row in db.execute("SELECT payload FROM events WHERE kind='api_cost'")]
    usage=dict(input_tokens=0,output_tokens=0,cached_input_tokens=0,reasoning_tokens=0)
    for event in events:
        u=event.get('usage') or {}
        usage['input_tokens']+=u.get('input_tokens',0);usage['output_tokens']+=u.get('output_tokens',0)
        usage['cached_input_tokens']+=u.get('input_tokens_details',{}).get('cached_tokens',0)
        usage['reasoning_tokens']+=u.get('output_tokens_details',{}).get('reasoning_tokens',0)
    row=dict(session=path.parent.name,requests=db.execute('SELECT COUNT(*) FROM api').fetchone()[0],
        tools=db.execute('SELECT COUNT(*) FROM tools').fetchone()[0],
        requests_without_receipt=db.execute('SELECT COUNT(*) FROM api WHERE response IS NULL').fetchone()[0],
        elapsed_api_seconds=sum(v['elapsed_seconds'] for v in events),**usage,
        interface_checks=db.execute("SELECT COUNT(*) FROM events WHERE kind='interface_check'").fetchone()[0],
        training_attempts=db.execute("SELECT COUNT(*) FROM events WHERE kind='training_attempt'").fetchone()[0],
        calibrations=db.execute("SELECT COUNT(*) FROM events WHERE kind='calibration'").fetchone()[0],
        price_status='Proxy billing not supplied; token/time costs recorded, dollar amount not invented')
    db.close();return row


def analyze_trace(path):
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    fields=('drawer_open','red_on_pad','blue_inside','gripper_released')
    first={field:next((r['step'] for r in rows if r.get('metrics',{}).get(field)),None) for field in fields}
    return dict(path=str(path),steps=len(rows),first_geometry_steps=first,
        maximum_red_z=max(r['state']['red_pose'][2] for r in rows),maximum_blue_z=max(r['state']['blue_pose'][2] for r in rows),
        maximum_drawer=max(r['state']['drawer_position'][0] for r in rows),final=rows[-1].get('metrics',{}),
        caveat='Geometry first events and maximum heights diagnose behavior; they are not final success predicates')


def videos(base,out):
    import cv2
    manifest=out/'video_samples.json'
    if manifest.exists():return read(manifest)
    rng=np.random.default_rng(17);selected=[]
    categories={
        'reference_success':[p for p in (base/'diagnostics/D0').glob('*/result.json') if read(p)['success']],
        'learned_failure':[p for p in (base/'diagnostics').rglob('result.json') if read(p).get('completed_evaluation') and not read(p)['success'] and list(p.parent.glob('frame_*.png'))],
        'API_capability':[base/'api_capability/native_capability/episode/result.json'],
    }
    for label,paths in categories.items():
        if not paths:continue
        path=sorted(paths)[int(rng.integers(len(paths)))];frames=sorted(path.parent.glob('frame_*.png'))
        if not frames:continue
        first=cv2.imread(str(frames[0]));video=out/(label+'.mp4')
        writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),1.,(first.shape[1],first.shape[0]))
        if not writer.isOpened():raise ValueError('Video codec failed; no successful media claim')
        for frame in frames:writer.write(cv2.imread(str(frame)))
        writer.release();decoder=cv2.VideoCapture(str(video));ok,_=decoder.read();count=int(decoder.get(cv2.CAP_PROP_FRAME_COUNT));decoder.release()
        if not ok or count!=len(frames):raise ValueError('Exported video did not decode to the expected frame count')
        selected.append(dict(category=label,source=str(path),video=str(video),sha256=digest(video),frames=count,fps=1.,
                             note='Uniformly selected saved episode; RGB sampled every 20 control steps. No interpolation or policy replay.'))
    atomic(manifest,selected);return selected


def generate(cfg):
    base=cfg['output'];out=base/'report';out.mkdir(exist_ok=True);rows=[];training=[];analysis=[]
    for path in sorted(base.rglob('result.json')):
        if path.is_relative_to(out):continue
        value=read(path);relative=str(path.relative_to(base))
        if 'optimizer_steps' in value:
            training.append(dict(path=relative,updates=value['optimizer_steps'],exposures=value['sample_exposures'],
                effective_targets=value['effective_action_targets'],parameters=value['trainable_parameters'],
                gpu=value['device']['physical_gpu'],seconds=value['training_elapsed_seconds'],
                weights_delta_l2=value['weight_l2_change'],reload_max_error=value['reload_max_action_error'],
                joint_rmse=value['sampling_joint_rmse'],gripper_mae=value['sampling_gripper_mae'],
                checkpoint_sha256=value['checkpoint_sha256'],interface_check='interface_check_' in relative,
                purpose='infrastructure_regression' if relative.startswith('checks/') else 'interface_check' if 'interface_check_' in relative else 'development_training'))
        if value.get('completed_evaluation') and (path.parent/'trace.jsonl').exists():
            final=value.get('final',{});rows.append(dict(path=relative,seed=value.get('seed'),success=value['success'],
                steps=value['steps'],status=value['status'],seconds=value['elapsed_seconds'],
                condition=value.get('condition','development'),method=value.get('method','diagnostic'),
                drawer_open=final.get('drawer_open'),red_on_pad=final.get('red_on_pad'),blue_inside=final.get('blue_inside'),
                gripper_released=final.get('gripper_released'),red_stable=final.get('red_stable'),blue_stable=final.get('blue_stable'),
                inference_seconds=value.get('local_inference_seconds'),prefix_steps=value.get('diagnostic_prefix_steps',0)))
            analysis.append(analyze_trace(path.parent/'trace.jsonl'))
    costs=[api_cost(p) for p in sorted(base.rglob('journal.sqlite'))]
    enforcement=[read(p) for p in base.rglob('enforcement.json')]
    reconstruction=read(base/'reconstruction/submission.json') if (base/'reconstruction/submission.json').exists() else None
    calibration_rows=[]
    for p in sorted((base/'reconstruction/calibration').glob('*/result.json')):
        v=read(p)
        for case in v['cases']:calibration_rows.append(dict(calibration=p.parent.name,case=case['id'],passed=case['passed'],**case['rmse_m']))
    environment_costs=[]
    for label,path,key in [('native_demonstration_replay','diagnostics/D0/result.json','episodes'),
                           ('independent_reference','reference/result.json','cases'),
                           ('API_MuJoCo_capability','api_capability/mujoco_capability/result.json','cases')]:
        p=base/path
        if p.exists():
            cases=read(p)[key];environment_costs.append(dict(scope=label,episodes=len(cases),
                control_steps=sum(c['steps'] for c in cases),elapsed_seconds=sum(c.get('elapsed_seconds',0) for c in cases)))
    for p in sorted((base/'reconstruction/calibration').glob('*/result.json')):
        v=read(p);environment_costs.append(dict(scope='MuJoCo_calibration_'+p.parent.name,episodes=len(v['cases']),control_steps=v['physical_steps']))
    environment_costs.append(dict(scope='native_learned_diagnostics',episodes=len(rows),
        control_steps=sum(r['steps']+r['prefix_steps'] for r in rows),elapsed_seconds=sum(r['seconds'] for r in rows)))
    summary=dict(updated_unix=time.time(),formal_gates=gates(cfg),training=training,episodes=rows,api=costs,
        total_optimizer_steps=sum(r['updates'] for r in training),training_gpu_seconds=sum(r['seconds'] for r in training),
        total_window_exposures=sum(r['exposures'] for r in training),formal_results_available=(base/'formal/evaluation.json').exists(),
        trusted_cuda_warmup_steps=sum(r['trusted_warmup_optimizer_steps'] for r in enforcement),
        initialization_and_lockdown_seconds=sum(r['initialization_and_lockdown_seconds'] for r in enforcement),
        reconstruction=reconstruction,calibration=calibration_rows,
        environment_costs=environment_costs,physics_substeps_per_control_step=25,
        exp1_preservation=read(ROOT/'runs/exp2/exp1_preservation_after.json') if (ROOT/'runs/exp2/exp1_preservation_after.json').exists() else None,
        exact_resume_check=read(base/'checks/exact_resume/verification.json') if (base/'checks/exact_resume/verification.json').exists() else None,
        diagnostic_and_formal_costs_must_not_be_conflated=True)
    atomic(out/'summary.json',summary);atomic(out/'trace_analysis.json',analysis)
    write_csv(out/'training.csv',training);write_csv(out/'episodes.csv',rows);write_csv(out/'api_costs.csv',costs)
    write_csv(out/'calibration.csv',calibration_rows)
    write_csv(out/'environment_costs.csv',environment_costs)
    selected_videos=videos(base,out)
    docs=ROOT/'experiments/exp2'
    table=['| 诊断 | 完成状态 | 闭环成功 |','|---|---|---|']
    for stage in ('D0','D1','D2','D2_sampling','D2_quaternion','D3/open_drawer','D3/move_red','D3/move_blue','D4','capability_M1'):
        p=base/'diagnostics'/stage/('diagnostic.json' if stage=='D1' else 'result.json')
        if not p.exists():table.append(f'| {stage} | 未完成 | — |');continue
        v=read(p);trials=v.get('results',v.get('episodes',[]))
        score=f"{sum(t['success'] for t in trials)}/{len(trials)}" if trials else ('采样检查通过' if v['success'] else '采样检查未通过')
        table.append(f"| {stage} | {'通过' if v['success'] else '未通过，结果保留'} | {score} |")
    diagnosis='''# DP 诊断

本文件的数字由 `exp2 report` 从集中 JSON 生成。D4 工程门槛在运行前固定为 16/20。
离线拟合、保存重载成功不等于闭环完成。D3 使用原始采集阶段标记，仅用于诊断；
正式 M1/M2 必须使用 API 自己确定的切分。

'''+ '\n'.join(table)+'''

已确认修复：旧候选后端没有真实神经更新，本版开放 Torch/GPU 并核验梯度、权重变化和重载；
新评估器要求完整物体占据、释放、稳定，以及技能交接时 TCP 高度超过 0.28 m。
GPU 嵌套命名空间和 XML 工具语法错误属于框架问题，见 incidents.json。

DP 控制链路：原生回放通过；动作是七个绝对关节角与一个原生归一化夹爪命令，
没有对机械臂动作额外裁成 [-1,1]。窗口按完整轨迹划分、因果对齐，并遮蔽边界填充。
训练为 DDPM epsilon，推理为同一噪声日程的 DDIM，最终 EMA 权重重载输出一致。

闭环证据：原始 D2 在第 199 步首次打开抽屉，之后抓红块时提前松爪，未成功抬升；
红块姿态偏离后，低方差四元数字段的标准化值放大。该放大是实测现象，但不能据此
断定它是唯一根因。50 步去噪 / 8 或 4 步执行段的只读诊断均失败；单位四元数范围
归一化另行训练验证，结果见表，不能隐去失败。D3 开抽屉达到约 0.30 m 后回退到
约 0.218 m；红、蓝块分别未抬升，说明单技能接触与闭环偏移已经构成障碍。
这些是有效策略失败，不作为基础设施错误清除。

当前证据不支持“DP 方法本身不行”，也不能用低 loss 宣称 ID 学会。每次训练的参数量、
关节采样误差、真实梯度步数、曝光量和耗时见 runs/exp2/rebuild_20260915/report/training.csv；
逐轨迹阶段与最大抬升高度见 trace_analysis.json。D4 未达门槛时，正式结果不启动。

恢复工程检查发现默认 CUDA 非确定计算在中断前就产生差异。固定 cuDNN deterministic、
CUBLAS workspace、禁止非确定算子后，4 步不中断和 2+2 步恢复的最终权重 hash 完全一致。
同时把可信 CUDA warmup 与候选随机初始化分离。原有未中断开发诊断按其原始源码/设置保留，
不称作采用新确定设置的正式数据；这项修复不自动消除既有闭环失败。
'''
    (docs/'DP_DIAGNOSIS.md').write_text(diagnosis)
    cap=read(base/'api_capability/submission.json') if (base/'api_capability/submission.json').exists() else None
    model=read(base/'reconstruction/submission.json') if (base/'reconstruction/submission.json').exists() else None
    actual=read(base/'api_capability/native_capability/result.json') if (base/'api_capability/native_capability/result.json').exists() else None
    body=['# Exp2 执行报告','',f"正式实验门槛：{'通过' if summary['formal_gates']['passed'] else '未通过'}。未满足项：{', '.join(summary['formal_gates']['unmet']) or '无'}。",
        '正式 M0/M1/M2 ID/OOD 矩阵尚未完成时，下面所有诊断都不能冒充正式主表。',
        '',*table,'',f"当前记录 {len(training)} 次神经训练、接口或恢复检查作业，共 {summary['total_optimizer_steps']:,} 次梯度更新、{summary['total_window_exposures']:,} 个窗口曝光，训练作业计时 {summary['training_gpu_seconds']:.1f} GPU 秒（不是 GPU kernel profiler 时间）。这不是正式候选模型数。接口/基础设施检查单列于 CSV。",
        '', '## API 实际贡献']
    if cap:
        metadata=read(base/'api_capability/versions'/cap['version']/'candidate.json')
        body.extend([f"真实 API 候选 `{metadata['policy_id']}`，版本 `{cap['version']}`。",'',metadata['prior'],'',
            'API 自选所有训练轨迹前 310 步作为开抽屉片段，加入抽屉/把手相对几何条件与轻度夹爪加权 epsilon loss。',
            '统一 DP 的条件从 94 维增至 122 维；主干宽度/深度保持一致，参数量变化单独记录。',
            '动作仍解码到原生绝对关节角，不能将观测相对化解释成关节动作平移等变。',
            f"完成 {cap['training']['optimizer_steps']} 次真实更新，权重 L2 变化 {cap['training']['weight_l2_change']:.4f}，重载最大误差 {cap['training']['reload_max_action_error']}。",
            f"原生执行 {'成功' if actual and actual['success'] else '未成功'}。同切分、同 5000 步 M1 能力对照也未成功；单一诊断不能估计 prior 的泛化收益。"])
    if model:
        body.extend(['','## MuJoCo 与场景有效性',f"API 重建版本 `{model['version']}`，固定校准通过：{model['calibration_passed']}。",
            'API 从原始 URDF 和授权示范修订指尖接触几何、摩擦与接触求解参数；开发者仅修复受控 XML 工具。',
            f"固定校准实际结果见 calibration.csv。完整 demo1000 红块位置 RMSE 从 {next(r['red'] for r in calibration_rows if r['calibration']=='000' and r['case']=='demo1000_0'):.6f} m 降到 {next(r['red'] for r in calibration_rows if r['calibration']==model['latest_calibration']['calibration_id'] and r['case']=='demo1000_0'):.6f} m。",
            '这证明已校准案例的重放一致性，不证明所有状态的仿真可信度；中途复位缺原生接触快照，物体角速度设零的限制明确保留。',
            '独立参考控制器在 ID / 位置 OOD / 状态 OOD 各两例全部通过，未增加训练数据。位置 OOD 与状态 OOD 范围见 PROTOCOL.md。'])
    p=base/'api_capability/mujoco_capability/result.json'
    if p.exists():
        v=read(p);body.extend([f"该 API 神经候选在校准后的 MuJoCo 中也实际执行了 {len(v['cases'])} 次，成功 {sum(c['success'] for c in v['cases'])} 次。这是外层能力验证，不冒充 API 完整库的一次正式反馈修订。"])
    body.extend(['','## 尚未得到的研究结论',
        '1. DP 当前失败的剩余证据支持接触误差、闭环分布偏移和阶段行为错误；训练不足/覆盖仍不能排除。',
        '2. 尚无完整 M0/M1 配对结果，不能量化分段/层级收益。',
        '3. 尚无正式 M1/M2 ID/OOD 配对结果，不能宣称 prior 提升泛化或不损害 ID。',
        '4. 真实 API 已贡献可训练的几何条件/loss 候选和通过校准的 MuJoCo 模型；候选成功率未因此自动获得保证。',
        '', '## 交付与边界',
        '唯一有效源码 src/appl，配置/文档 experiments/exp2，独立 Pixi 环境 environments/exp2；大产物在 runs/exp2 的存储盘映射。',
        'Exp1 的 85,566 个持久文件 hash 一致；原本为空的 SQLite WAL 和 SHM 是连接临时文件，主数据库 hash 不变。原入口核验 144 槽位完成、artifact issues=0。',
        '正式设计/训练/评估入口有门槛保护；第二版冻结后拒绝写代码、重训和更换成员。整个正式三技能周期尚未通过真实运行验证，不写成完成。',
        '真机驱动、硬件控制语义、标定与感知仍待配置；没有伪装成真实机器人验证。',
        '完整 API 工具调用/响应、版本、模型、预算保留；代理服务未提供美元账单，因此仅报告 token、调用、时间和训练成本。',
        '正式结果未形成时没有置信区间或效果量；配对多种子统计须等实际矩阵存在后生成。',
        '视频为固定随机种子从保存回合抽样、1 Hz 的原始 RGB 帧序列：'+', '.join('['+v['category']+'](../../runs/exp2/rebuild_20260915/report/'+Path(v['video']).name+')' for v in selected_videos)+'.',
        '', '数字来源：[summary.json](../../runs/exp2/rebuild_20260915/report/summary.json)、[训练表](../../runs/exp2/rebuild_20260915/report/training.csv)、[逐回合表](../../runs/exp2/rebuild_20260915/report/episodes.csv)、[API 成本](../../runs/exp2/rebuild_20260915/report/api_costs.csv)。'])
    (docs/'REPORT.md').write_text('\n'.join(body)+'\n')
    return summary
