import functools
import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def cosine_with_warmup_step(step: int, *, warmup_steps: int, total_steps: int, num_cycles: float) -> float:
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))

    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine = math.cos(math.pi * float(num_cycles) * 2.0 * progress)

    return max(0.0, 0.5 * (1.0 + cosine))


def cosine_with_warmup(optimizer: Optimizer, warmup_steps: int, total_steps: int, num_cycles: float = 0.5) -> LambdaLR:
    scheduler = functools.partial(
        cosine_with_warmup_step,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        num_cycles=num_cycles,
    )

    return LambdaLR(optimizer, scheduler)
