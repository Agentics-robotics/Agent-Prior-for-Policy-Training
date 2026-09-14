"""Freeze the requested first16 development report from actual completed runs."""
from round3.common import R3,run_id,run_record,now
from round3.evaluate import _checked_result
from round3.report import paired_ood_delta
from relative_dp.utils import read_json,atomic_json,sha256


def main():
    identifiers=[run_id(t,c,n) for t in ('pick-place-wall','assembly')
                 for n in (5,20) for c in ('B0','P1','P2','P3')]
    missing=[r for r in identifiers if run_record(r)['dev_status']!='completed']
    if missing:
        print({'status':'pending','remaining_priority_development':missing})
        return
    out=R3/'reports';out.mkdir(exist_ok=True)
    results={r:_checked_result(R3/'dev_results'/r/'complete.json') for r in identifiers}
    trained={r:read_json(R3/'checkpoints'/r/'complete.json') for r in identifiers}
    assert all(t['step']==20000 and not t['debug'] for t in trained.values())
    rows=[]
    for rid,r in results.items():
        task,n,candidate=(r['identity'][k] for k in ('task','train_n','candidate_id'))
        baseline=results[run_id(task,'B0',n)]
        delta=paired_ood_delta(r,baseline)
        rows.append(dict(task=task,N=n,candidate=candidate,run_id=rid,
            scores={s:r['metrics'][s] for s in ('IID','C','E')},ood=r['ood_success_rate'],
            comparison=delta,parameters=r['parameter_count'],training_seconds=trained[rid]['total_session_wall_seconds'],
            development_seconds=r['last_session_wall_seconds']))
    record=dict(time=now(),stage='development_only',formal_runs=16,optimizer_updates=320000,
        scored_development_episodes=800,locked_test_episodes=0,training_seed=0,rows=rows,
        sources={f'round3/dev_results/{r}/complete.json':sha256(R3/'dev_results'/r/'complete.json') for r in identifiers})
    atomic_json(out/'first16_development.json',record)
    lines=['# Round 3 首批16次开发结果快报','',
        '已真实完成 pick-place-wall、assembly 的 B0/P1/P2/P3、N=5/20 共16次正式训练，每次20,000 updates；完成800个固定开发回合。以下均为单个训练seed的开发结果，锁定测试尚未开放。',
        '', '初始设计由实际隔离 Codex 子代理读取各任务D2与16张真实图像后保存；协调者已接触历史报告与基础设施源码，因此总体属于 Codex 辅助设计的探索实验。策略输入仍是共同数值状态。', '',
        '| 任务 | N | 候选 | IID | C | E | C/E均值 | 相对B0百分点 | 配对95%区间（百分点） |',
        '|---|---:|---|---:|---:|---:|---:|---:|---|']
    for row in rows:
        rates=[f"{row['scores'][s]['successes']}/{row['scores'][s]['n']}" for s in ('IID','C','E')]
        lo,hi=row['comparison']['paired_bootstrap_95ci_pp']
        lines.append(f"| {row['task']} | {row['N']} | {row['candidate']} | {' | '.join(rates)} | {100*row['ood']:.1f}% | {row['comparison']['delta_pp']:+.1f} | [{lo:+.1f}, {hi:+.1f}] |")
    lines+=['','区间按同一snapshot配对、在C/E内分层bootstrap；不包含训练seed或示范子集随机性。每个分布的Wilson区间、完整诊断和成本保存在JSON中。开发选优有选择偏差，不能用这份快报代替封存测试。','',
        'pick-place-wall：P1将抓取、目标、墙体关系分支编码；P2学习软阶段条件；P3组合关系编码与动作/未来手和物体位移联合扩散，并监督几何一致性。assembly：P1使用手、把手、螺母中心和目标的关系；P2在当前搬运方向坐标系中编码观测和动作；P3加入软事件条件与未来关系监督。各方法保持共同观测、相同D_N、20k和batch128预算；参数与推理计算仍有差异。','']
    for task in ('pick-place-wall','assembly'):
        for n in (5,20):
            group=[r for r in rows if r['task']==task and r['N']==n]
            best=min([r for r in group if r['candidate']!='B0'],key=lambda r:(-r['ood'],-r['scores']['IID']['success_rate'],r['parameters'],r['candidate']))
            b=next(r for r in group if r['candidate']=='B0')
            iid_delta=100*(best['scores']['IID']['success_rate']-b['scores']['IID']['success_rate'])
            lines.append(f"- {task} N{n}：按开发规则 initial-selected={best['candidate']}，OOD相对B0 {best['comparison']['delta_pp']:+.1f}个百分点，IID {iid_delta:+.1f}个百分点。"+('两者同步提升，收益可能同时包含拟合改善。' if iid_delta>0 and best['comparison']['delta_pp']>0 else ''))
    lines+=['','一次反馈决策（各自仅使用N5开发材料）：']
    for task in ('pick-place-wall','assembly'):
        path=R3/'design_records'/task/'revision.json'
        decision=read_json(path)['decision'] if path.exists() else 'pending'
        lines.append(f'- {task}: {decision}。决定与额外训练单独记录，不合并初始曲线。')
    lines+=['',f"首16个系统训练墙钟之和 {sum(r['training_seconds'] for r in rows)/3600:.2f}小时，开发评测墙钟之和 {sum(r['development_seconds'] for r in rows)/3600:.2f}小时；这些并行系统耗时之和不等于项目实际经过时间。当前用户授权GPU0–5六卡，各卡最多一个正式训练进程。",'',
        '后续继续其余主矩阵与各任务一次反馈决策；所有选择冻结后才执行锁定测试。完整动态报告：[ROUND3_REPORT.md](../ROUND3_REPORT.md)；本次不可与测试混同的数据：[first16_development.json](first16_development.json)。']
    path=out/'FIRST_DEVELOPMENT_REPORT.md';path.write_text('\n'.join(lines)+'\n')
    print({'status':'complete','report':str(path),'runs':16,'dev_episodes':800})


if __name__=='__main__':
    main()
