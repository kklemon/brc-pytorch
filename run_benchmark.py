import itertools
import time
import hydra
import torch
import torch.nn.functional as F
import torch.multiprocessing

from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from brc_pytorch import modules, benchmarks, utils


torch.multiprocessing.set_sharing_strategy('file_system')


def get_dataset(name: str, kwargs: dict):
    kwargs = dict(kwargs)
    if name == 'copy_first':
        dataset = benchmarks.CopyFirstInputBenchmark(**kwargs)
        return dataset, dataset
    elif name == 'denoising':
        dataset = benchmarks.DenoisingBenchmark(**kwargs)
        return dataset, dataset
    elif name == 'sequential_mnist':
        transform = transforms.ToTensor()
        data_root = kwargs.pop('data_root')
        mnist_train = datasets.MNIST(data_root, train=True, transform=transform)
        mnist_test = datasets.MNIST(data_root, train=False, transform=transform)
        return (
            benchmarks.SequentialImageClassification(mnist_train, **kwargs),
            benchmarks.SequentialImageClassification(mnist_test, **kwargs)
        )
    else:
        raise ValueError(f'Unknown dataset \'{name}\'')


@hydra.main(config_path='config', config_name='benchmark')
def main(cfg: DictConfig):
    device = cfg.training.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    train_data, test_data = get_dataset(cfg.benchmark.name, cfg.benchmark.args)

    assert train_data.meta == test_data.meta
    meta = train_data.meta

    get_batches_fn = lambda dataset: itertools.cycle(DataLoader(
        train_data,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        drop_last=True
    ))
    train_batches = get_batches_fn(train_data)
    test_batches = get_batches_fn(test_data)

    model = modules.RNNModel(
        input_size=meta.in_dim,
        hidden_size=cfg.model.hidden_size,
        output_size=meta.out_dim,
        num_layers=cfg.model.num_layers,
        cell=getattr(modules, cfg.model.cell),
        bias=cfg.model.bias,
        dropout=cfg.model.dropout,
        sequential_output=meta.sequential_output
    ).to(device)

    print(model)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    print('TRAINING')

    seen_batches = 0
    sum_loss = 0
    sum_corr = 0
    last_time = time.time()

    init_hidden = utils.to_device(model.init_hidden(cfg.training.batch_size), device=device)

    for step, (src, trg) in enumerate(itertools.islice(train_batches, cfg.training.train_steps)):
        src, trg = utils.to_batch_first(
            utils.to_device((src, trg), device)
        )

        opt.zero_grad()

        out, _ = model(src, init_hidden)
        if meta.categorical_output:
            loss = F.cross_entropy(out.flatten(end_dim=-2), trg.view(-1))
            sum_corr += ((out.argmax(-1) == trg) * 1.0).sum()
        else:
            loss = F.mse_loss(out, trg)
        loss.backward()

        opt.step()

        seen_batches += 1
        sum_loss += loss.item()

        if (step + 1) % cfg.logging.log_interval == 0:
            ms_per_batch = 1000.0 * (time.time() - last_time) / seen_batches
            metrics = {'loss': f'{sum_loss / seen_batches:.4f}'}
            if meta.categorical_output:
                metrics['acc'] = f'{100 * sum_corr / (seen_batches * cfg.training.batch_size):.2f}'
            metrics['ms per batch'] = f'{ms_per_batch:.2f}'

            print(f'[{step + 1:06d}/{cfg.training.train_steps:06d}] {", ".join(map(": ".join, metrics.items()))}')

            seen_batches = 0
            sum_loss = 0
            sum_corr = 0
            last_time = time.time()

    print('EVALUATION')

    seen_batches = 0
    sum_loss = 0
    sum_corr = 0

    for (src, trg) in itertools.islice(test_batches, cfg.training.eval_steps):
        src, trg = utils.to_batch_first(
            utils.to_device((src, trg), device)
        )

        out, _ = model(src, init_hidden)
        if meta.categorical_output:
            loss = F.cross_entropy(out.flatten(end_dim=-2), trg.view(-1))
            sum_corr += ((out.argmax(-1) == trg) * 1.0).sum()
        else:
            loss = F.mse_loss(out, trg)

        seen_batches += 1
        sum_loss += loss.item()

    metrics = {'loss': f'{sum_loss / seen_batches:.4f}'}
    if meta.categorical_output:
        metrics['acc'] = f'{100 * sum_corr / (seen_batches * cfg.training.batch_size):.2f}'

    print(', '.join(map(": ".join, metrics.items())))


if __name__ == '__main__':
    main()
