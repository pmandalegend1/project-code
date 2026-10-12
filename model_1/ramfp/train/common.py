"""Shared training utilities: seeding, device, schedules, checkpoints."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def cosine_with_warmup(optimizer: torch.optim.Optimizer, total_steps: int,
                       warmup_frac: float = 0.1) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EarlyStopping:
    """Track the best score and the state dict that produced it."""

    def __init__(self, patience: int = 8, mode: str = "max"):
        self.patience, self.mode = patience, mode
        self.best = -math.inf if mode == "max" else math.inf
        self.best_state: Dict[str, torch.Tensor] | None = None
        self.bad_epochs = 0

    def improved(self, score: float) -> bool:
        return score > self.best if self.mode == "max" else score < self.best

    def step(self, score: float, model: torch.nn.Module) -> bool:
        """Returns True if training should stop."""
        if self.improved(score):
            self.best = score
            self.best_state = {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()}
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def restore(self, model: torch.nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def save_results(rows: Iterable[Dict], path: str | Path) -> None:
    import pandas as pd
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path, index=False)
    print(f"  wrote {path}")
