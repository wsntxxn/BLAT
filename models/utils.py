import torch

from einops import repeat, rearrange


def generate_length_mask(lens, max_length=None):
    """
    lens: [batch_size,]
    """
    batch_size = lens.size(0)
    if max_length is None:
        max_length = max(lens)
    idxs = repeat(torch.arange(max_length), "l -> b l", b=batch_size)
    idxs = idxs.to(lens.device)
    mask = (idxs < rearrange(lens, "b -> b 1"))
    return mask


def mean_with_lens(features, lens):
    """
    features: [batch_size, time_steps, ...] 
        (assume the second dimension represents length)
    lens: [batch_size,]
    """
    if max(lens) != features.size(1):
        max_length = features.size(1)
        mask = generate_length_mask(lens, max_length)
    else:
        mask = generate_length_mask(lens)
    mask = mask.to(features.device)

    while mask.ndim < features.ndim:
        mask = rearrange(mask, "... -> ... 1")
    feature_mean = features * mask
    feature_mean = feature_mean.sum(1)
    while lens.ndim < feature_mean.ndim:
        lens = rearrange(lens, "... -> ... 1")
    feature_mean = feature_mean / lens.to(features.device)
    return feature_mean


def max_with_lens(features, lens):
    """
    features: [batch_size, time_steps, ...] 
        (assume the second dimension represents length)
    lens: [batch_size,]
    """
    if max(lens) != features.size(1):
        max_length = features.size(1)
        mask = generate_length_mask(lens, max_length)
    else:
        mask = generate_length_mask(lens)
    mask = mask.to(features.device) # [batch_size, time_steps]

    feature_max = features.clone()
    feature_max[~mask] = float("-inf")
    feature_max, _ = feature_max.max(1)
    return feature_max

