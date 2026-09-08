"""Validate saved human-reviewable Codex proposals; never generate mock designs."""
from .common import R3, TASKS, CANDIDATES, planned_runs, update_run, implementation_path, now, event
from relative_dp.utils import ROOT, read_json, atomic_json, sha256, object_hash

REQUIRED = ['candidate_id','knowledge_hypotheses','evidence_refs','predicted_generalization_gain',
 'applicability','failure_conditions','architecture','modules','observation_transform','action_transform',
 'losses_and_weights','augmentation','training_schedule','hierarchy_selection_and_termination',
 'chunk_transition_handling','required_train_labels','required_runtime_inputs','parameter_budget',
 'implementation_steps','confidence']

def validate(task, freeze=True):
    directory=R3/'design_records'/task
    path=directory/'proposal.json'
    if not path.exists():
        return dict(task=task,status='pending-design',evidence=str(R3/'evidence'/task))
    doc=read_json(path)
    assert doc['task']==task
    assert doc['agent_backend']=='codex_session'
    assert doc.get('created_at') and doc.get('designer') and doc.get('context_exposure')
    candidates=doc['candidates']
    assert [c['candidate_id'] for c in candidates]==['P1','P2','P3']
    for candidate in candidates:
        missing=[key for key in REQUIRED if key not in candidate]
        if missing:
            raise ValueError(f'{task}/{candidate["candidate_id"]}: missing {missing}')
        if not candidate['knowledge_hypotheses'] or not candidate['evidence_refs']:
            raise ValueError('Proposals require explicit grounded hypotheses and references')
    hashes=doc['evidence_hashes']
    if not hashes:
        raise ValueError('No evidence hashes')
    allowed=(R3/'evidence'/task).resolve()
    for name,digest in hashes.items():
        source=(ROOT/name).resolve()
        if not source.is_relative_to(allowed):
            raise ValueError('Initial designer evidence outside task D2 bundle')
        if sha256(source)!=digest:
            raise ValueError(f'Design evidence changed: {name}')
    result=dict(task=task,status='validated',proposal_sha256=sha256(path),evidence_hashes=hashes,
        candidate_ids=['P1','P2','P3'],schema_validation=True,mathematical_implementation_audit='required separately')
    if freeze:
        target=directory/'initial_freeze.json'
        if target.exists():
            old=read_json(target)
            if old['proposal_sha256']!=result['proposal_sha256']:
                raise ValueError('Initial proposal was changed after freezing; use an explicit versioned feasibility revision')
        else:
            atomic_json(target,dict(result,frozen_at=now()))
            event('initial_proposals_frozen',task=task,proposal_sha256=result['proposal_sha256'])
        for rec in planned_runs():
            if rec['task']==task and rec['status']=='pending-design':
                update_run(rec['run_id'],status='pending')
    return result

def validate_all():
    results=[validate(task) for task in TASKS]
    atomic_json(R3/'audits'/'proposal_validation.json',dict(results=results,updated_at=now()))
    return results

def baseline_config(task):
    manifest=read_json(R3/'protocol_and_task_manifests'/(task+'.json'))
    config=dict(candidate_id='B0',plugin_module='round3.plugins',plugin_class='BaselinePlugin',plugin_config={},
        proposal_sha256=sha256(R3/'ROUND3_SPEC.txt'),observation_schema_sha256=object_hash(manifest['observation_schema']),
        rationale='Unmodified Round2 state-input U-Net DP, all common raw channels, current-DN statistics; unbounded DDIM before world clipping',
        implementation_audit='baseline behavior equality to Round2 Policy with matching obs dimension',custom_loss=False)
    path=implementation_path(task,'B0')
    if path.exists():
        assert read_json(path)==config
    else:
        atomic_json(path,config)
    return config
