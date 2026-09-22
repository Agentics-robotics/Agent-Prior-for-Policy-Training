"""Developer-owned synthetic interface fixture, never a scientific candidate."""
import numpy as np
import torch
from real_robot.training_pipeline import public


def prepare_data(spec):
    n = len(public.records(spec["trajectory_ids"][0])["sample_index"])
    anchors = np.array([2, 2, 3], dtype=np.int64)
    x = np.array([[1.], [2.], [3.]], dtype=np.float32)
    arrays = dict(example_episode=np.zeros(3, dtype=np.int64), example_source_index=anchors,
                  example_segment=np.zeros(3, dtype=np.int64), example_variant=np.array([0, 1, 0]),
                  observation_start=anchors, observation_stop=anchors + 1,
                  target_start=anchors, target_stop=anchors + 1,
                  policy_weight=np.ones((3, len(spec["policy_ids"]))), x=x,
                  target=np.concatenate([x, x + 1], axis=1),
                  target_mask=np.array([[1., 0.], [1., 1.], [1., 1.]], dtype=np.float32))
    metadata = dict(num_examples=3, coverage="Synthetic two-event, three-variant fixture",
                    label_provenance="Synthetic targets only", data_audit="No real demonstrations used",
                    variant_definition="Synthetic alternative goal at anchor 2",
                    source_accounting=[dict(trajectory_id=spec["trajectory_ids"][0], start=0, stop=n,
                        use="context", reason="Complete synthetic source retained"),
                        dict(trajectory_id=spec["trajectory_ids"][0], start=2, stop=4,
                        use="supervision", reason="Synthetic derived training anchors")])
    return dict(arrays=arrays, metadata=metadata)


def build_model(policy_id, spec):
    return torch.nn.Linear(1, 2)


def make_batch(policy_id, arrays, metadata, indices, spec):
    return {name: torch.tensor(arrays[name][indices], dtype=torch.float32, device=spec["device"])
            for name in ("x", "target", "target_mask")}


def compute_loss(model, batch, spec):
    return dict(loss=((model(batch["x"]) - batch["target"]).square() * batch["target_mask"]).mean())


def predict(model, batch, spec):
    return model(batch["x"])


def configure_optimizer(model, spec):
    return torch.optim.SGD(model.parameters(), lr=spec["policy_config"]["config"]["lr"])


def sample_indices(policy_id, arrays, metadata, rng, step, spec):
    return np.arange(metadata["num_examples"], dtype=np.int64)


def before_update(optimizer, step, spec):
    for group in optimizer.param_groups:
        group["lr"] = spec["policy_config"]["config"]["lr"] / step


def after_update(model, ema, arrays, metadata, step, metrics, spec):
    return dict(stop=step == 2, select="model", metrics=dict(fixture=True))


def act(model, observation, call, memory, spec):
    if "x" not in observation:
        return dict(memory=None, decision=None, status="missing")
    x = torch.tensor(observation["x"], dtype=torch.float32, device=spec["device"]).reshape(1, 1)
    decision = dict(delta=model(x)[0].detach().cpu().tolist())
    return dict(memory=None, decision=decision, status="ready")


def executor_request(decision, executor_contract, spec):
    return dict(synthetic_objective=decision["delta"], hardware_io=False)


def calling_cases(arrays, metadata, spec):
    return [dict(name="valid synthetic observation", observation=dict(x=np.array([1.], dtype=np.float32)),
                 call={}, reset=True, executor_contract={}, expected_status="ready", expect_decision=True),
            dict(name="missing synthetic observation", observation={}, call={}, reset=True,
                 executor_contract={}, expected_status="missing", expect_decision=False)]
