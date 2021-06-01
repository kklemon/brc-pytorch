brc-pytorch
===========

PyTorch implementation of the bistable recurrent cell (BRC) from the paper _A bio-inspired bistable recurrent cell allows for
long-lasting memory_ (Vecoven et al., 2020).

Install
-------

```bash
pip install brc-pytorch
```

Usage
-----

```python
import torch
from brc_pytorch.modules import BRC, NBRC, StackedRNN

brc = StackedRNN(
    cell=BRC,  # NBRC for the neuromodulated version
    input_size=128,
    hidden_size=256,
    num_layers=3
)

# [ seq_len, batch_size, dim ]
x = torch.randn(64, 32, 128)

init_hidden = brc.init_hidden(batch_size=32)
out, hidden = brc(x, init_hidden)
```

Performance
-----------

The implementation is provided in TorchScript and makes use of the PyTorch JIT compiler.
In my not really statistically significant experiments, the implementation seems to be about half as fast as the cuDNN based reference LSTM implementation with modest batch sizes and sequence lengths which can be considered pretty solid for a non-CUDA implementation.

References
----------

```bibtex
@misc{vecoven2020bioinspired,
    title={A bio-inspired bistable recurrent cell allows for long-lasting memory},
    author={Nicolas Vecoven and Damien Ernst and Guillaume Drion},
    year={2020},
    eprint={2006.05252},
    archivePrefix={arXiv},
    primaryClass={cs.NE}
}
```