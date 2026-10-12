"""
Step 09: train the style-conditioned opponent co-behavior model (E1).

    python -m ramfp.train.train_opponent --config configs/default.yaml \
        --triples data/processed/triples.parquet --encoder models/encoder.pt

Defaults to synthetic triples if --triples is omitted, and then also prints the
Bayes-optimal ceiling so you can tell "my model is broken" from "this problem
is noisy".

Splitting: GroupKFold by match_id. Two triples from the same match share a
game state and often the same teams, so a random split leaks and inflates
Top-1 by several points. Report mean +- std over folds x seeds; with ~1,500
examples a single split is not a result.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from ..config import Config
from ..data.datasets import TripleDataset, collate
from ..data.schema import validate_triples
from ..eval.metrics import (E1Metrics, evaluate as compute_metrics,
                            evaluate_conditional, evaluate_response)
from ..models.opponent import OpponentCoBehaviorModel
from ..models.state_encoder import StateEncoder
from .common import (EarlyStopping, cosine_with_warmup, count_parameters,
                     save_results, set_seed, to_device)


@torch.no_grad()
def predict(model: OpponentCoBehaviorModel, loader: DataLoader, device: str):
    """Run both heads and return everything the three metric views need.

    formation  -- softmax of the formation head: "given they change, to what"
    mixture    -- the planner's distribution, formation head weighted by the
                  respond head with the remaining mass on the shape they are
                  already in. This is the only one comparable to the
                  persistence baseline, because it answers the same question.
    p_change   -- the respond head alone
    """
    model.eval()
    form, mixed, p_chg, ys, chg = [], [], [], [], []
    for batch in loader:
        batch = to_device(batch, device)
        logits, respond = model(batch, return_respond=True)
        p_form = torch.softmax(logits, dim=-1)
        p_change = torch.sigmoid(respond)
        stay = torch.zeros_like(p_form)
        stay.scatter_(1, batch["opp_formation_id"].view(-1, 1).long(), 1.0)
        mix = ((1.0 - p_change).unsqueeze(-1) * stay
               + p_change.unsqueeze(-1) * p_form)
        form.append(p_form.float().cpu().numpy())
        mixed.append(mix.float().cpu().numpy())
        p_chg.append(p_change.float().cpu().numpy())
        ys.append(batch["target"].cpu().numpy())
        chg.append(batch["changed"].cpu().numpy())
    return (np.concatenate(form), np.concatenate(mixed), np.concatenate(p_chg),
            np.concatenate(ys), np.concatenate(chg).astype(bool))


@torch.no_grad()
def validation_objective(model: OpponentCoBehaviorModel, loader: DataLoader,
                         device: str, select: str) -> float:
    """Lower is better, for both options.

    "loss" is the training objective itself, evaluated on held-out matches.
    It is the default because it is the only quantity that scores both heads
    and keeps the KL calibration term in view; the conformal terminator reads
    calibrated probabilities, so selecting on something that ignores
    calibration would undercut it.

    "nll" is the formation head's NLL on the decisions where the shape really
    moved. Note this is the CONDITIONAL NLL: scored over every row instead,
    it is dominated by rows whose answer is "no change", and a model that
    learns to copy its own input would win.
    """
    model.eval()
    if select == "loss":
        total, n = 0.0, 0
        for batch in loader:
            batch = to_device(batch, device)
            logits, respond = model(batch, return_respond=True)
            out = model.loss(logits, batch["target"], respond_logit=respond,
                             respond_target=batch["changed"],
                             changed=batch["changed"].bool())
            k = int(len(batch["target"]))
            total += float(out["loss"]) * k
            n += k
        return total / max(n, 1)

    p_form, _, _, y, changed = predict(model, loader, device)
    if not changed.any():
        return float("inf")          # nothing to select on in this split
    return float(evaluate_conditional(p_form, y, changed).nll)


def train_one_fold(cfg: Config, train_df: pd.DataFrame, val_df: pd.DataFrame,
                   test_df: pd.DataFrame, encoder_path: str | None,
                   style_conditioned: bool, device: str) -> tuple[Dict, OpponentCoBehaviorModel]:
    enc = StateEncoder.load(encoder_path, map_location=device) if encoder_path else None
    if enc is not None and enc.cfg.d_context != cfg.encoder.d_context:
        raise ValueError("checkpointed encoder d_context does not match the config")

    model = OpponentCoBehaviorModel(
        cfg.encoder, cfg.opponent, encoder=enc,
        style_conditioned=style_conditioned).to(device)

    mk = lambda df, mirror, shuffle: DataLoader(   # noqa: E731
        TripleDataset(df, cfg.encoder.seq_len, mirror=mirror, knn=cfg.encoder.knn),
        batch_size=cfg.train.batch_size, shuffle=shuffle, collate_fn=collate,
        num_workers=cfg.train.num_workers)

    train_dl = mk(train_df, cfg.train.mirror_augment, True)
    val_dl = mk(val_df, False, False)
    test_dl = mk(test_df, False, False)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    sched = cosine_with_warmup(opt, max(1, len(train_dl) * cfg.train.epochs),
                               cfg.train.warmup_frac)
    # Model selection metric. Selecting on Top-1 lets the model keep sharpening
    # a confidently-wrong distribution long after accuracy has plateaued: on
    # ~750 training examples that pushed ECE from 0.05 to 0.25 in our runs.
    # NLL is the accuracy-and-calibration compromise, and ECE is what Table 3
    # reports, so select on NLL unless you have a reason not to.
    select = cfg.train.select_on
    if select not in ("loss", "nll"):
        warnings.warn(
            f"train.select_on={select!r} is a single-head metric and no "
            f"longer describes this objective; selecting on 'loss'. "
            f"Selecting on Top-1 in particular lets the model keep "
            f"sharpening a confidently-wrong distribution long after "
            f"accuracy has plateaued.")
        select = "loss"
    stopper = EarlyStopping(cfg.train.patience, mode="min")
    use_amp = cfg.train.amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for _ in range(cfg.train.epochs):
        model.train()
        for batch in train_dl:
            batch = to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, respond = model(batch, return_respond=True)
                out = model.loss(
                    logits, batch["target"],
                    respond_logit=respond,
                    respond_target=batch["changed"],
                    # The formation head sees only the decisions where the
                    # shape moved -- about 11% of rows. Trained on all of
                    # them it would learn the persistence rule instead.
                    changed=batch["changed"].bool())
            scaler.scale(out["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
        if stopper.step(validation_objective(model, val_dl, device, select),
                        model):
            break

    stopper.restore(model)
    p_form, p_mix, p_chg, y, changed = predict(model, test_dl, device)

    # Three views, because no single number describes this task.
    overall = compute_metrics(p_mix, y)                     # vs persistence
    conditional = evaluate_conditional(p_form, y, changed)  # the real choice
    response = evaluate_response(p_chg, changed)            # the rare event
    row: Dict = {}
    row.update(overall.as_row())
    row.update({f"cond_{k}": v for k, v in conditional.as_row().items()})
    row.update({f"resp_{k}": v for k, v in response.as_row().items()})
    return row, model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--triples", default=None)
    ap.add_argument("--encoder", default=None, help="pretrained encoder .pt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-synthetic", type=int, default=1500)
    ap.add_argument("--temperature", type=float, default=0.5,
                    help="synthetic difficulty; lower is easier")
    ap.add_argument("--ablate-style", action="store_true",
                    help="also train the no-style-conditioning ablation")
    ap.add_argument("--save-folds", action="store_true",
                    help="save every fold's checkpoint for the first seed as "
                         "opponent_fold{k}.pt (needed for cross-fitted planner evaluation, "
                         "where each decision must be scored by a model that never saw its match)")
    ap.add_argument("--first-seed-only", action="store_true",
                    help="train only the first seed (enough for --save-folds; ~3x faster)")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cfg.paths.ensure()
    device = cfg.resolved_device()

    if args.triples:
        df = validate_triples(pd.read_parquet(args.triples))
        ceiling = None
        print(f"loaded {len(df)} real triples from {args.triples}")
    else:
        from ..data.synthetic import generate_triples
        df, stats = generate_triples(args.n_synthetic, seed=cfg.seed,
                                     temperature=args.temperature,
                                     seq_len=cfg.encoder.seq_len)
        validate_triples(df)
        ceiling = stats
        print("NO --triples GIVEN: using synthetic data\n" + str(stats) + "\n")

    variants: List[tuple[str, bool]] = [("full (style-conditioned)", True)]
    if args.ablate_style:
        variants.append(("ablation: no style conditioning", False))

    rows: List[Dict] = []
    for name, styled in variants:
        print(f"\n=== {name} ===")
        seeds = cfg.train.seeds[:1] if args.first_seed_only else cfg.train.seeds
        for seed in seeds:
            set_seed(int(seed))
            gkf = GroupKFold(n_splits=cfg.train.n_folds)
            groups = df["match_id"].to_numpy()
            for fold, (tr_idx, te_idx) in enumerate(gkf.split(df, groups=groups)):
                tr = df.iloc[tr_idx]
                # carve a validation split out of train, still grouped by match
                val_matches = pd.unique(tr["match_id"])[: max(1, len(pd.unique(tr["match_id"])) // 8)]
                val = tr[tr["match_id"].isin(val_matches)]
                tr = tr[~tr["match_id"].isin(val_matches)]
                te = df.iloc[te_idx]
                if len(val) == 0:
                    val = te
                m, model = train_one_fold(cfg, tr, val, te, args.encoder, styled, device)
                print(f"  seed {seed} fold {fold}: "
                      f"mixture Top-1 {m['top1']:.4f} (persistence is the bar) | "
                      f"respond AUROC {m['resp_auroc']:.4f} "
                      f"AP {m['resp_average_precision']:.4f} "
                      f"(base {m['resp_base_rate']:.4f}) | "
                      f"changed-only Top-1 {m['cond_top1']:.4f} "
                      f"n={m['cond_n']}")
                rows.append({"variant": name, "seed": int(seed), "fold": fold, **m})
                # With --save-folds and no --out, leave models/opponent.pt alone:
                # calibration and conformal results were fitted on that file.
                if styled and seed == cfg.train.seeds[0] and fold == 0 and not (
                        args.save_folds and not args.out):
                    ckpt = args.out or str(Path(cfg.paths.models) / "opponent.pt")
                    model.save(ckpt)
                    print(f"    saved fold-0 checkpoint -> {ckpt}  "
                          f"({count_parameters(model):,} params)")
                if styled and seed == cfg.train.seeds[0] and args.save_folds:
                    fk = str(Path(cfg.paths.models) / f"opponent_fold{fold}.pt")
                    model.save(fk)
                    print(f"    saved fold-{fold} checkpoint -> {fk}")

    res = pd.DataFrame(rows)
    print("\n=== E1 summary (mean +- std over folds x seeds) ===")
    groups = [
        ("A. distribution over formations (mixture) -- compare to the "
         "persistence baseline, which is the only fair comparison here",
         ("top1", "top3", "ece", "nll", "macro_f1")),
        ("B. does the opponent respond at all -- the learnable question. "
         "Read average_precision against base_rate, not against 0.5",
         ("resp_auroc", "resp_average_precision", "resp_base_rate",
          "resp_recall", "resp_precision", "resp_brier",
          "resp_persistence_accuracy")),
        ("C. which formation, on the decisions where it really moved -- "
         "the only subset where this question is meaningful",
         ("cond_top1", "cond_top3", "cond_ece", "cond_nll", "cond_n")),
    ]
    for variant, g in res.groupby("variant", sort=False):
        print(f"\n{variant}")
        for title, cols in groups:
            print(f"  {title}")
            for col in cols:
                if col in g:
                    print(f"     {col:<26} {g[col].mean():.4f} "
                          f"+- {g[col].std():.4f}")
    if ceiling is not None:
        print(f"\n   reference: majority {ceiling.majority_top1:.4f} | "
              f"Bayes Top-1 {ceiling.bayes_top1:.4f} | Bayes Top-3 {ceiling.bayes_top3:.4f}")

    save_results(rows, Path(cfg.paths.results) / "table3_e1_opponent.csv")


if __name__ == "__main__":
    main()
