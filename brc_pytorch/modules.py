import torch
import torch.nn as nn

from torch import Tensor
from typing import List, Tuple, Optional


class BRC(torch.jit.ScriptModule):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.iw = nn.Parameter(torch.Tensor(input_size, 3 * hidden_size))
        self.hw = nn.Parameter(torch.Tensor(hidden_size * 2))

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(hidden_size * 3))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            nn.init.xavier_uniform_(self.iw)
            self.hw.fill_(1.0)
            if self.bias is not None:
                self.bias.zero_()

    def init_hidden(self, batch_size: int):
        return [torch.zeros(batch_size, self.hidden_size, device=self.hw.device)]

    def get_hidden_activation(self, hidden: Tensor) -> List[Tensor]:
        return (torch.cat([hidden, hidden], dim=-1) * self.hw).chunk(2, dim=-1)

    @torch.jit.script_method
    def forward(self, input: Tensor, hidden: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
        hx, = hidden
        x = (input @ self.iw)
        if self.bias is not None:
            x += self.bias
        ia, ic, ir = x.chunk(3, dim=-1)
        ha, hc = self.get_hidden_activation(hx)

        a = 1 + torch.tanh(ia + ha)
        c = torch.sigmoid(ic + hc)

        hy = c * hx + (1 - c) * torch.tanh(ir + a * hx)

        return hy, [hy]

    def __repr__(self):
        info = {
            'input_size': str(self.input_size),
            'hidden_size': str(self.hidden_size),
            'bias': str(self.bias is not None)
        }
        pairs = map('='.join, info.items())
        return f'{self.__class__.__name__}({", ".join(pairs)})'


class NBRC(BRC):
    def __init__(self, input_dim: int, hidden_dim: int, bias: bool = True):
        super().__init__(input_dim, hidden_dim, bias)
        self.hw = nn.Parameter(torch.Tensor(hidden_dim, 2 * hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for weight in [self.iw, self.hw]:
                if weight.ndim > 1:
                    nn.init.xavier_uniform_(weight)
            self.bias.zero_()

    def get_hidden_activation(self, hidden: Tensor) -> List[Tensor]:
        ha, hc = torch.chunk(hidden @ self.hw, 2, dim=-1)
        return [ha, hc]


class LSTMCell(torch.jit.ScriptModule):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.iw = nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
        self.hw = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(4 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for weight in [self.iw, self.hw]:
                nn.init.xavier_uniform_(weight)
            if self.bias is not None:
                self.bias.zero_()

    def init_hidden(self, batch_size: int):
        return list(torch.zeros(2, batch_size, self.hidden_size, device=self.iw.device))

    @torch.jit.script_method
    def forward(self, input: Tensor, hidden: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
        hx, cx = hidden
        gates = (input @ self.iw) + (hx @ self.hw)
        if self.bias is not None:
            gates += self.bias

        ig, fg, cg, og = gates.chunk(4, 1)

        cy = torch.sigmoid(fg) * cx + torch.sigmoid(ig) * torch.tanh(cg)
        hy = torch.sigmoid(og) * torch.tanh(cy)

        return hy, [hy, cy]


class GRUCell(torch.jit.ScriptModule):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.iw = nn.Parameter(torch.Tensor(input_size, 3 * hidden_size))
        self.hw = nn.Parameter(torch.Tensor(hidden_size, 2 * hidden_size))
        self.rw = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(3 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for weight in [self.iw, self.hw, self.rw]:
                nn.init.xavier_uniform_(weight)
            if self.bias is not None:
                self.bias.zero_()

    def init_hidden(self, batch_size: int):
        return [torch.zeros(batch_size, self.hidden_size, device=self.iw.device)]

    @torch.jit.script_method
    def forward(self, input: Tensor, hidden: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
        hx, = hidden

        x = (input @ self.iw)
        if self.bias is not None:
            x += self.bias

        ig, rg, og = x.chunk(3, 1)
        ih, rh = (hx @ self.hw).chunk(2, 1)
        ig = torch.sigmoid(ig + ih)
        rg = torch.sigmoid(rg + rh)

        h_hat = torch.tanh(og + (rg * hx) @ self.rw)
        hy = (1 - ig) * hx + ig * h_hat

        return hy, [hy]


class RNNLayer(torch.jit.ScriptModule):
    def __init__(self, cell, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.cell = cell(input_size, hidden_size, bias)

    def init_hidden(self, batch_size: int):
        return self.cell.init_hidden(batch_size)

    @torch.jit.script_method
    def forward(self, input: Tensor, hidden: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(input)):
            out, hidden = self.cell(input[i], hidden)
            outputs += [out]
        return torch.stack(outputs), hidden


class StackedRNN(torch.jit.ScriptModule):
    def __init__(self,
                 cell,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 bias: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        assert num_layers > 0
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout

        self.rnns = nn.ModuleList([RNNLayer(cell, input_size if not i else hidden_size, hidden_size, bias)
                                   for i in range(num_layers)])
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

    def init_hidden(self, batch_size: int):
        return [rnn.init_hidden(batch_size) for rnn in self.rnns]

    @torch.jit.script_method
    def forward(self, input: Tensor, hidden: List[List[Tensor]]) -> Tuple[Tensor, List[List[Tensor]]]:
        out_hiddens = torch.jit.annotate(List[List[Tensor]], [])
        for i, rnn in enumerate(self.rnns):
            output, out_hidden = rnn(input, hidden[i])
            if i + 1 < self.num_layers and self.dropout is not None:
                output = self.dropout(output)
            input = output
            out_hiddens += [out_hidden]
        return output, out_hiddens


class RNNModel(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_layers: int,
                 cell,
                 bias: bool = True,
                 dropout: float = 0.0,
                 num_embeds: int = None,
                 padding_idx: Optional[int] = None,
                 sequential_output: bool = True):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.num_embeds = num_embeds
        self.sequential_output = sequential_output

        if num_embeds is not None:
            self.embedding = nn.Embedding(num_embeds, input_size, padding_idx=padding_idx)
        else:
            self.embedding = nn.Identity()

        self.encoder = StackedRNN(cell, input_size, hidden_size, num_layers, bias, dropout)
        self.decoder = nn.Linear(hidden_size, output_size)

    def init_hidden(self, batch_size: int):
        return self.encoder.init_hidden(batch_size)

    def forward(self, input, hidden):
        input = self.embedding(input)
        output, hidden = self.encoder(input, hidden)
        if not self.sequential_output:
            output = output[-1]
        output = self.decoder(output)
        return output, hidden
