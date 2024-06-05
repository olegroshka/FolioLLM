import torch
import torch.nn as nn
import torch.nn.functional as F


def select_scores(
        heatmap, do_sample=False, top_p=0.2, top_k=50, temperature=1,
        sorted_return=True, standardize=False
):
    if standardize:
        heatmap -= heatmap.mean()
        heatmap /= heatmap.std()

    p = torch.nn.functional.softmax(heatmap, 0)
    val, ind = p.sort(descending=True)
    val, ind = val[:top_k], ind[:top_k]
    m = int((val.cumsum(0) < val.sum() * top_p).sum(dim=-1))

    if not do_sample: return ind[:m], torch.arange(m)

    T = 10 / (temperature + 1)
    ord_stat = torch.multinomial(val ** T, m, replacement=False)

    sorted_indices = ord_stat.sort()[0]
    return ind[sorted_indices], sorted_indices
