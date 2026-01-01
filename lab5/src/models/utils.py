import torch


def padding_mask(input_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """(B,L) -> (B,1,1,L) bool; True=keep"""
    return (input_ids != pad_id).unsqueeze(1).unsqueeze(2)


def masked_mean_pool(x: torch.Tensor, input_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """x: (B,L,D) -> (B,D)"""
    mask = (input_ids != pad_id).unsqueeze(-1).type_as(x)  # (B,L,1)
    x = x * mask
    denom = mask.sum(dim=1).clamp(min=1.0)                 # (B,1)
    return x.sum(dim=1) / denom