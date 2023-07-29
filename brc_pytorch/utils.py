from typing import List


def nested_apply(fn, whatever):
    if isinstance(whatever, (tuple, list)):
        return [nested_apply(fn, el) for el in whatever]
    return fn(whatever)


def to_device(whatever, device):
    return nested_apply(lambda t: t.to(device), whatever)


def to_batch_first(whatever):
    return nested_apply(lambda t: t.transpose(0, 1) if t.ndim > 2 else t, whatever)


def detach_hidden(hidden):
    return nested_apply(lambda t: t.clone().detach(), hidden)


def split(l: List, fractions: List[float]):
    assert sum(fractions) == 1
    n = len(l)
    return [l[round(n * sum(fractions[:i])):round(n * sum(fractions[:i + 1]))] for i in range(len(fractions))]
