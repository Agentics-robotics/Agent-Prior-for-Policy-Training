import torch


class WindowGRU(torch.nn.Module):
    """Standard GRU equations, explicitly unrolled without flat-weight caches.

    The same learned cell serves the two-frame observation encoder and the
    action-conditioned forecast. No hidden state persists between invocations.
    """
    def __init__(self, input_dimension, hidden_dimension):
        super().__init__()
        self.input_gates = torch.nn.Linear(input_dimension, 3*hidden_dimension)
        self.hidden_gates = torch.nn.Linear(hidden_dimension, 3*hidden_dimension)

    def forward(self, inputs, initial):
        state = initial[0]
        input_gates = self.input_gates(inputs)
        outputs = []
        for index in range(inputs.shape[1]):
            ir, iz, inn = input_gates[:, index].chunk(3, dim=-1)
            hr, hz, hn = self.hidden_gates(state).chunk(3, dim=-1)
            reset = torch.sigmoid(ir+hr)
            update = torch.sigmoid(iz+hz)
            candidate = torch.tanh(inn+reset*hn)
            state = candidate+update*(state-candidate)
            outputs.append(state)
        return torch.stack(outputs, dim=1), state.unsqueeze(0)
