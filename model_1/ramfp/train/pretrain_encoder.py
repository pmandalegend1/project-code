"""
Step 07: pretrain the state encoder on next-event-type prediction.

This is the only stage of your slice with a lot of data — every event of every
match, not just the ~1,500 triples. It is where the encoder learns what
football sequences look like, and it is what makes the tiny opponent-model
training set survivable.

    python -m ramfp.train.pretrain_encoder --config configs/default.yaml \
        --sequences data/processed/sequences.parquet --out models/encoder.pt

With no --sequences it falls back to synthetic data so you can smoke-test the
whole loop before the data team ships anything.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import Config
from ..data.datasets import SequenceDataset, collate
from ..data.schema import validate_sequences
from ..models.state_encoder import PAD_ID, StateEncoder
from .common import (count_parameters, cosine_with_warmup, set_seed, to_device)


def next_event_loss(logits: torch.Tensor, batch) -> torch.Tensor:
    """
    Position t predicts the type at t+1. Positions whose target is PAD are
    ignored, which also drops the final real token (it has no successor).
    """
    target = batch["event_type"][:, 1:]
    pred = logits[:, :-1]
    return F.cross_entropy(
        pred.reshape(-1, pred.size(-1)), target.reshape(-1), ignore_index=PAD_ID)


@torch.no_grad()
def evaluate(model: StateEncoder, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    tot_loss = tot_correct = tot_n = 0.0
    for batch in loader:
        batch = to_device(batch, device)
        logits = model.pretrain_logits(batch)
        loss = next_event_loss(logits, batch)
        target = batch["event_type"][:, 1:]
        mask = target.ne(PAD_ID)
        correct = (logits[:, :-1].argmax(-1) == target) & mask
        tot_loss += loss.item() * int(mask.sum())
        tot_correct += float(correct.sum())
        tot_n += float(mask.sum())
    return tot_loss / max(tot_n, 1), tot_correct / max(tot_n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--sequences", default=None, help="parquet from step 01")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-synthetic", type=int, default=20_000)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cfg.paths.ensure()
    set_seed(cfg.seed)
    device = cfg.resolved_device()
    out_path = args.out or str(Path(cfg.paths.models) / "encoder.pt")

    if args.sequences:
        df = validate_sequences(pd.read_parquet(args.sequences))
        print(f"loaded {len(df)} real sequences from {args.sequences}")
    else:
        from ..data.synthetic import generate_sequences
        df = generate_sequences(args.n_synthetic, seed=cfg.seed, seq_len=cfg.encoder.seq_len)
        print(f"NO --sequences GIVEN: using {len(df)} synthetic sequences (smoke test only)")

    if cfg.pretrain.max_sequences:
        df = df.head(cfg.pretrain.max_sequences)

    n_val = max(1, int(0.05 * len(df)))
    train_df, val_df = df.iloc[:-n_val], df.iloc[-n_val:]
    train_ds = SequenceDataset(train_df, cfg.encoder.seq_len, mirror=cfg.train.mirror_augment)
    val_ds = SequenceDataset(val_df, cfg.encoder.seq_len, mirror=False)

    train_dl = DataLoader(train_ds, batch_size=cfg.pretrain.batch_size, shuffle=True,
                          collate_fn=collate, num_workers=cfg.train.num_workers, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.pretrain.batch_size, collate_fn=collate)

    model = StateEncoder(cfg.encoder).to(device)
    print(f"device={device}  trainable params={count_parameters(model):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.pretrain.lr,
                            weight_decay=cfg.train.weight_decay)
    total_steps = max(1, len(train_dl) * cfg.pretrain.epochs)
    sched = cosine_with_warmup(opt, total_steps, cfg.train.warmup_frac)
    use_amp = cfg.train.amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(1, cfg.pretrain.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_dl:
            batch = to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = next_event_loss(model.pretrain_logits(batch), batch)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.detach().item()
        val_loss, val_acc = evaluate(model, val_dl, device)
        print(f"epoch {epoch:>3}  train {running / max(1, len(train_dl)):.4f}  "
              f"val {val_loss:.4f}  next-event acc {val_acc:.4f}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    print(f"saved encoder -> {out_path}")


if __name__ == "__main__":
    main()
