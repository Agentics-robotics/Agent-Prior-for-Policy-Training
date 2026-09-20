"""Read-only diagnosis of a retained failed check; no policy edits or updates."""
import argparse
import copy
from pathlib import Path
import sys

from appl.io import atomic, digest, read


def main(args):
    from appl.gpu import launch, verify_cuda
    if not args.device_isolated:
        launch(args.gpu, sys.argv[1:], module='experiments.exp2.astra.diagnose_reload')
    import torch
    from appl.prior_policies import data, engine
    from appl.dp_baseline.train import deterministic_numerics
    from appl.security import lockdown
    root = args.check.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    request = read(root / 'request.json')
    source = Path(read(root / 'worker_request.json')['source'])
    checkpoint = root / 'last.pt'
    receipt = dict(checkpoint_sha256=digest(checkpoint), source=str(source),
                   diagnostic_sha256=digest(__file__), policy_updates=0,
                   API_calls=0, simulator_steps=0, device=verify_cuda())
    spec, segments = data.specification(request['entry'], request['config'],
                                       read(source / 'pipeline.json'))
    raw = torch.as_tensor(data.windows(segments, request['config'])['raw_obs'][:2], device='cuda:0')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    deterministic_numerics()
    torch.set_num_threads(2)
    lockdown(source, out)
    module = engine.module_at(source)
    torch.manual_seed(0)
    first = module.build_model(spec).cuda().eval()
    first.load_state_dict(saved['ema'])
    copied = copy.deepcopy(first).requires_grad_(False)
    second = module.build_model(spec).cuda().eval()
    second.load_state_dict(saved['ema'])
    models = [('loaded_trainable', first), ('deepcopied_frozen', copied),
              ('second_loaded_trainable', second)]
    differences = {}
    state = first.state_dict()
    for name, model in models:
        differences[name] = max(float((value - state[key]).abs().max())
                                for key, value in model.state_dict().items())
    receipt['state_dict_max_errors'] = differences
    atomic(out / 'setup.json', receipt)
    prediction = []
    with torch.no_grad():
        noisy = torch.randn((2, spec['training']['horizon'], 8), device='cuda:0',
                            generator=torch.Generator(device='cuda:0').manual_seed(913))
        timestep = torch.tensor(99, device='cuda:0')
        for name, model in models:
            prediction.append((name, model(noisy, timestep, raw)))
    receipt['single_forward_max_errors'] = {
        name: float((value - prediction[0][1]).abs().max()) for name, value in prediction}
    outputs = []
    for name, model in models:
        value = engine.sample(model, raw, spec, torch.Generator(device='cuda:0').manual_seed(913))
        outputs.append((name, value))
    receipt['DDPM100_action_max_errors'] = {
        name: float((value - outputs[0][1]).abs().max()) for name, value in outputs}
    repeat = engine.sample(first, raw, spec, torch.Generator(device='cuda:0').manual_seed(913))
    receipt['same_instance_repeat_max_error'] = float((repeat - outputs[0][1]).abs().max())
    first.requires_grad_(False)
    second.requires_grad_(False)
    frozen_outputs = [engine.sample(model, raw, spec,
                      torch.Generator(device='cuda:0').manual_seed(913)) for model in (first, second)]
    receipt['same_instance_after_disabling_grad_error_vs_frozen_copy'] = float(
        (frozen_outputs[0] - outputs[1][1]).abs().max())
    receipt['second_loaded_after_disabling_grad_error_vs_frozen_copy'] = float(
        (frozen_outputs[1] - outputs[1][1]).abs().max())
    receipt['DDPM100_chunks'] = 6
    atomic(out / 'result.json', receipt)
    print(receipt, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--device-isolated', action='store_true')
    main(parser.parse_args())
