"""Processing contract tests only: synthetic data, scripted API, no learned prior."""
from copy import deepcopy
import json

from PIL import Image
import pytest

from appl.demonstrations import DemonstrationProcessor, Limits, process_demonstrations
from appl.io import digest, object_hash, read


@pytest.fixture
def demos(tmp_path):
    directory = tmp_path / 'input'
    directory.mkdir()
    Image.new('RGB', (2, 2), (0, 0, 0)).save(directory / 'frame.png')
    paths = []
    for identifier in ('a', 'b'):
        path = directory / (identifier + '.json')
        path.write_text(json.dumps(dict(trajectory_id=identifier, task_id='synthetic_contract_fixture',
            provenance='test_fixture', goal={'fixture_goal': True}, deployment_requirements=None,
            actions=[[i, -i] for i in range(6)], observations=[dict(timestamp=i * .1,
                state={'fixture_position': [i, i + 1, i + 2]}, images={'camera': 'frame.png'}) for i in range(7)],
            video_assets=[])))
        paths.append(path)
    return paths


@pytest.fixture
def plan():
    def segment(identifier, start, stop):
        return dict(trajectory_id=identifier, start=start, stop=stop)

    def skill(identifier, segments, index):
        return dict(skill_id=identifier, name=identifier, subgoal='Synthetic subgoal used only by a test.',
            segmentation_rationale='Scripted test boundaries; no inference or discovery claimed.', segments=segments,
            heuristics=[dict(statement='Synthetic heuristic fixture; not an experimental prior.',
                evidence=[dict(trajectory_id='a', indices=[index])],
                rationale='Tests preserve API-provided explanatory text without generating a claim.',
                applicability='Synthetic fixture only.',
                pipeline_implications='A future pipeline is unspecified; no DP recipe is imposed.',
                limitations='This is scripted test output, not a validated heuristic.',
                handoff={key: 'Synthetic interface text supplied by the test client.' for key in (
                    'entry_conditions','exit_conditions','overlap_role','successor_readiness','failure_signatures')})])

    return dict(skills=[skill('phase_alpha', [segment('a', 0, 2), segment('a', 4, 6), segment('b', 0, 3)], 0),
                        skill('phase_beta', [segment('a', 2, 4), segment('b', 3, 6)], 3)],
                exclusions=[], overlap_rationale='')


@pytest.fixture
def processor_factory(demos, tmp_path):
    created = []

    def make():
        processor = DemonstrationProcessor(demos, tmp_path / ('output_' + str(len(created))), provenance='test_fixture')
        created.append(processor)
        for identifier in ('a', 'b'):
            processor.dispatch('read_steps', dict(trajectory_id=identifier, indices=list(range(7))))
        return processor

    yield make
    for processor in created:
        processor.j.db.close()


class ScriptedClient:
    provenance = 'test_fixture'
    model = 'scripted-test-client'
    reasoning = None

    def __init__(self, plan):
        self.calls = 0
        self.plan = plan
        self.output_limits = []

    def configuration(self):
        return {'model': self.model, 'provenance': self.provenance}

    def respond(self, request):
        self.output_limits.append(request['max_output_tokens'])
        calls = [
            ('list_demonstrations', {}),
            ('read_steps', dict(trajectory_id='a', indices=list(range(7)))),
            ('read_steps', dict(trajectory_id='b', indices=list(range(7)))),
            ('read_image', dict(trajectory_id='a', index=0, camera='camera')),
            ('write_plan', dict(plan=self.plan)),
            ('check_plan', {}),
            ('submit_datasets', dict(expected_hash=object_hash(self.plan))),
        ]
        name, args = calls[self.calls]
        self.calls += 1
        return 200, dict(status='completed', output=[dict(type='function_call', name=name,
            call_id='fixture_' + str(self.calls), arguments=json.dumps(args))], usage={})


def test_api_loop_exports_portable_raw_datasets_and_heuristic_markdown(demos, plan, tmp_path):
    for number in (2, 3):
        extra = deepcopy(plan['skills'][0]['heuristics'][0])
        extra['statement'] = f'Synthetic heuristic fixture {number}; transport test only.'
        plan['skills'][0]['heuristics'].append(extra)
    original_hashes = {p: digest(p) for p in demos}
    client = ScriptedClient(plan)
    root = tmp_path / 'processed'
    result = process_demonstrations(demos, root, client=client)
    assert client.calls == 7
    assert set(client.output_limits) == {Limits().max_output_tokens}
    assert result['provenance'] == 'test_fixture'
    assert result['checks']['original_demonstrations'] == 2
    assert result['checks']['segments'] == 5
    assert result['training_updates'] == result['simulator_calls'] == 0
    dataset = read(root / 'datasets/phase_alpha/dataset.json')
    assert dataset['original_demonstration_count'] == 2 and dataset['segment_count'] == 3
    assert dataset['heuristics'] == plan['skills'][0]['heuristics']
    assert '## Heuristic 3' in (root / 'datasets/phase_alpha/heuristic.md').read_text()
    for record in result['datasets']:
        directory = (root / record['dataset']).parent
        for segment in read(root / record['dataset'])['segments']:
            output = read(directory / segment['file'])
            source = read(demos[0] if segment['trajectory_id'] == 'a' else demos[1])
            start, stop = segment['start'], segment['stop']
            assert output['actions'] == source['actions'][start:stop]
            assert len(output['observations']) == stop - start + 1
            assert output['deployment_requirements'] is None
            assert output['source']['sha256'] == original_hashes[demos[0] if segment['trajectory_id'] == 'a' else demos[1]]
            for i, obs in enumerate(output['observations']):
                assert obs['state'] == source['observations'][start + i]['state']
                assert obs['timestamp'] == source['observations'][start + i]['timestamp']
                assert digest(directory / obs['images']['camera']) == digest(demos[0].parent / 'frame.png')
        text = (root / record['heuristic']).read_text()
        assert 'Synthetic heuristic fixture' in text and 'test_fixture' in text
        assert 'future training/inference pipeline' in text
        assert '### Handoff interface' in text
        assert 'Synthetic interface text supplied by the test client.' in text
    assert {p: digest(p) for p in demos} == original_hashes
    assert process_demonstrations(demos, root, client=client) == result
    assert client.calls == 7  # A completed resume makes no API request.


def test_plan_requires_inspected_boundaries_and_grounded_evidence(processor_factory, plan):
    processor = processor_factory()
    processor.j.set('inspected_steps', {'a': [0, 2, 4, 6], 'b': [0, 3, 6]})
    with pytest.raises(ValueError, match='evidence that was not read'):
        processor._validate(plan)
    processor.j.set('inspected_steps', {'a': [0, 2, 3, 4, 6], 'b': [0, 6]})
    with pytest.raises(ValueError, match='boundary'):
        processor._validate(plan)
    processor.j.set('inspected_steps', {'a': list(range(7)), 'b': list(range(7))})
    plan['skills'][0]['heuristics'][0]['evidence'][0]['indices'] = [3]
    with pytest.raises(ValueError, match='belong to this skill'):
        processor._validate(plan)


def test_gaps_need_explicit_exclusion_and_overlap_needs_explanation(processor_factory, plan):
    processor = processor_factory()
    plan['skills'][1]['segments'][1]['start'] = 4
    with pytest.raises(ValueError, match='Unassigned actions'):
        processor._validate(plan)
    plan['exclusions'] = [dict(trajectory_id='b', start=3, stop=4, reason='Explicit synthetic exclusion.')]
    assert processor._validate(plan)['excluded_actions'] == 1
    plan['skills'][1]['segments'][0]['start'] = 1
    with pytest.raises(ValueError, match='Overlapping'):
        processor._validate(plan)
    plan['overlap_rationale'] = 'Explicit synthetic overlap for the test.'
    assert processor._validate(plan)['overlapping_actions'] == 1


def test_tool_diagnostics_and_checked_hash_are_required(processor_factory, plan):
    processor = processor_factory()
    result = processor.execute(dict(name='read_steps', call_id='invalid', arguments=json.dumps(dict(trajectory_id='hidden', indices=[0]))))
    assert 'unauthorized' in result['error']
    processor.dispatch('write_plan', dict(plan=plan))
    with pytest.raises(ValueError, match='checked plan hash'):
        processor.dispatch('submit_datasets', dict(expected_hash=object_hash(plan)))
    processor.dispatch('check_plan', {})
    changed = deepcopy(plan)
    changed['skills'][0]['name'] = 'Another test name'
    processor.dispatch('write_plan', dict(plan=changed))
    with pytest.raises(ValueError, match='checked plan hash'):
        processor.dispatch('submit_datasets', dict(expected_hash=object_hash(changed)))


def test_submitted_files_are_immutable_and_tampering_is_detected(processor_factory, plan):
    processor = processor_factory()
    processor.dispatch('write_plan', dict(plan=plan))
    processor.dispatch('check_plan', {})
    processor.dispatch('submit_datasets', dict(expected_hash=object_hash(plan)))
    with pytest.raises(ValueError, match='immutable'):
        processor.dispatch('write_plan', dict(plan=plan))
    (processor.root / 'datasets/phase_alpha/heuristic.md').write_text('tampered')
    with pytest.raises(ValueError, match='modified'):
        processor.verify_output()


@pytest.mark.parametrize('escape', ['../outside.png', 'link.png'])
def test_media_paths_cannot_escape_authorized_source(demos, tmp_path, escape):
    outside = tmp_path / 'outside.png'
    outside.write_bytes(b'not authorized')
    (demos[0].parent / 'link.png').symlink_to(outside)
    value = read(demos[0])
    value['observations'][0]['images']['camera'] = escape
    demos[0].write_text(json.dumps(value))
    with pytest.raises(ValueError, match='relative|Symlinks'):
        DemonstrationProcessor(demos, tmp_path / 'processed', provenance='test_fixture')
    assert not (tmp_path / 'processed').exists()


def test_input_changes_block_resume_and_publication(demos, processor_factory, plan):
    processor = processor_factory()
    processor.dispatch('write_plan', dict(plan=plan))
    processor.dispatch('check_plan', {})
    demos[0].write_text(demos[0].read_text() + '\n')
    with pytest.raises(ValueError, match='changed during processing'):
        processor.dispatch('submit_datasets', dict(expected_hash=object_hash(plan)))
    with pytest.raises(ValueError, match='changed; use another'):
        DemonstrationProcessor(demos, processor.root, provenance='test_fixture')
    assert not (processor.root / 'datasets').exists()


def test_unknown_api_outcome_is_not_automatically_retried(demos, plan, tmp_path):
    class InterruptedClient(ScriptedClient):
        def respond(self, request):
            self.calls += 1
            raise TimeoutError('Synthetic transport interruption')

    client = InterruptedClient(plan)
    with pytest.raises(TimeoutError):
        process_demonstrations(demos, tmp_path / 'processed', client=client)
    with pytest.raises(RuntimeError, match='no durable receipt'):
        process_demonstrations(demos, tmp_path / 'processed', client=client)
    assert client.calls == 1


def test_api_budget_does_not_silently_continue(demos, plan, tmp_path):
    client = ScriptedClient(plan)
    with pytest.raises(ValueError, match='API call budget exhausted'):
        process_demonstrations(demos, tmp_path / 'processed', client=client, limits=Limits(max_api_calls=1))
    assert client.calls == 1
    assert not (tmp_path / 'processed/datasets').exists()
