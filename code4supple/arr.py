"""arr.py
ARR: Gumbel-Softmax K-hot mask utilities.

Functions:
  - sample_gumbel(shape, eps=1e-20, device=None)
  - gumbel_softmax_sample(logits, tau=1.0, device=None)
  - gumbel_softmax_khot(logits, K=8, tau=0.5, hard=True)

Usage:
  logits is a tensor (batch, n_features) produced by a masker head.
  Use gumbel_softmax_khot() to obtain a differentiable K-hot mask for ARR.
"""
import torch
import torch.nn.functional as F

def sample_gumbel(shape, eps=1e-20, device=None):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, tau=1.0, device=None):
    g = sample_gumbel(logits.size(), device=device)
    y = logits + g
    return F.softmax(y / tau, dim=-1)

def gumbel_softmax_khot(logits, K=8, tau=0.5, hard=True):
    device = logits.device
    y = gumbel_softmax_sample(logits, tau=tau, device=device)
    if not hard:
        return y
    # top-k selection on continuous y
    topk_idx = torch.topk(y, K, dim=-1)[1]   # (batch, K)
    hard_mask = torch.zeros_like(y)
    hard_mask.scatter_(1, topk_idx, 1.0)
    # straight-through trick: forward uses hard_mask, backward uses gradients from y
    ret = (hard_mask - y).detach() + y
    return ret

if __name__ == '__main__':
    logits = torch.randn(4, 256)
    mask = gumbel_softmax_khot(logits, K=16, tau=0.5, hard=True)
    print('mask sums', mask.sum(dim=1))
