import math
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from brc_pytorch import modules, utils
from omegaconf import DictConfig


def batchify(data, batch_size):
    nbatch = data.size(0) // batch_size
    data = data.narrow(0, 0, nbatch * batch_size)
    data = data.view(batch_size, -1).t().contiguous()
    return data


def get_batch(data, idx, seq_len):
    seq_len = min(seq_len, len(data) - 1 - idx)
    src = data[idx:idx + seq_len].clone().detach()
    trg = data[idx + 1:idx + 1 + seq_len].clone().detach().view(-1)
    return src, trg


def train(model, batches, batch_size, opt, device, epoch, summary, log_interval=100):
    model.train()

    loss_sum = 0

    hidden = utils.to_device(model.init_hidden(batch_size), device=device)

    for step, batch in enumerate(batches):
        src, trg = utils.to_device(batch, device=device)
        hidden = utils.detach_hidden(hidden)

        out, hidden = model(src, hidden)
        loss = F.cross_entropy(out.view(-1, out.shape[-1]), trg.view(-1))

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.3)
        opt.step()

        loss_sum += loss.item()

        global_step = epoch * len(batches) + step
        summary.add_scalar('train_loss', loss, global_step)
        summary.add_scalar('train_ppl', math.exp(loss), global_step)
        summary.add_scalar('train_bpc', loss / math.log(2), global_step)

        if (step + 1) % log_interval == 0:
            avg_loss = loss_sum / (step + 1)
            print(f'[EPOCH {epoch:03d}][{step + 1:06d}/{len(batches):06d}] '
                  f'loss: {avg_loss:.2f}, ppl: {math.exp(avg_loss):.2f}, bpc: {avg_loss / math.log(2):.2f}')


@torch.no_grad()
def evaluate(model, batches, batch_size, device, epoch=None, summary=None, log_prefix=None):
    model.eval()

    hidden = utils.to_device(model.init_hidden(batch_size), device=device)

    loss_sum = 0
    num_steps = 0

    for batch in batches:
        src, trg = utils.to_device(batch, device=device)
        out, hidden = model(src, hidden)
        loss = F.cross_entropy(out.view(-1, out.shape[-1]), trg.view(-1))

        loss_sum += loss.item()
        num_steps += 1

    avg_loss = loss_sum / num_steps
    avg_ppl = math.exp(avg_loss)
    avg_bpc = avg_loss / math.log(2)

    if summary is not None:
        assert epoch is not None and log_prefix is not None
        summary.add_scalar(f'{log_prefix}_loss', avg_loss, epoch)
        summary.add_scalar(f'{log_prefix}_ppl', avg_ppl, epoch)
        summary.add_scalar(f'{log_prefix}_bpc', avg_bpc, epoch)

    print(f'EVAL loss: {avg_loss:.2f}, ppl: {avg_ppl:.2f}, bpc: {avg_bpc:.2f}')


@hydra.main(config_path='config', config_name='lm')
def main(cfg: DictConfig):
    device = cfg.training.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    data = torch.LongTensor(list(map(int, Path(cfg.data.path).read_bytes()))).to(device)
    test_frac, val_frac = cfg.training.test_frac, cfg.training.val_frac
    train_data, test_data, valid_data = utils.split(
        data, [1 - test_frac - val_frac, test_frac, val_frac]
    )

    train_data = batchify(train_data, cfg.training.batch_size)
    test_data = batchify(test_data, 1)
    valid_data = batchify(valid_data, 1)

    get_batches = lambda data: [
        get_batch(data, i, cfg.training.bptt) for i in range(0, train_data.size(0) - 1, cfg.training.bptt)
    ]

    train_batches = get_batches(train_data)
    test_batches = get_batches(test_data)
    valid_batches = get_batches(valid_data)

    ntokens = 256

    model = modules.RNNModel(
        input_size=cfg.model.embed_size,
        hidden_size=cfg.model.hidden_size,
        output_size=ntokens,
        num_layers=cfg.model.num_layers,
        cell=getattr(modules, cfg.model.cell),
        bias=cfg.model.bias,
        dropout=cfg.model.dropout,
        num_embeds=ntokens
    ).to(device)

    summary = SummaryWriter('.')

    print(model)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    print('TRAINING')

    for epoch in range(cfg.training.epochs):
        train(model, train_batches, cfg.training.batch_size, opt, device, epoch, summary)
        evaluate(model, valid_batches, 1, device, epoch, summary, 'val')
        # torch.save(model, 'model.pt')

    print('TESTING')
    evaluate(model, test_batches, 1, device)


if __name__ == '__main__':
    main()
