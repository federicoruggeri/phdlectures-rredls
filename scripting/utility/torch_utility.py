import torch
import torch as th


def th_fix_seed(
        seed: int
):
    th.manual_seed(seed)
    torch.use_deterministic_algorithms(True)