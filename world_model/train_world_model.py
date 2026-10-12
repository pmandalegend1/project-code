import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Batch
import yaml
from scipy.stats import pearsonr
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, ".")
from models.world_model.bipartite_gnn import BipartiteFormationGNN, MeanPoolMLP
from models.world_model.graph_builder import build_bipartite_graph

TARGETS = ["xg_delta", "possession_pct", "ppda", "shots_on_target"]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path="configs/world_model.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------

def load_data(cfg):
    windows = pd.read_parquet(cfg["stable_windows_path"])
    rms_df  = pd.read_parquet(cfg["role_matchup_stats_path"])
    sty_df  = pd.read_parquet(cfg["style_clusters_path"])

    with open(cfg["formation_vocab_path"]) as f:
        vocab_raw = json.load(f)
    formation_vocab = {int(c): idx for idx, c in enumerate(vocab_raw["vocab"])}

    return windows, rms_df, sty_df, formation_vocab


def temporal_split(windows, cfg):
    """Hold out later football from earlier football.

    Three routes, in order of preference:

    1. Explicit train_competitions / test_competitions lists in the config,
       when the (competition_id, season_id) pairs they name actually occur in
       the data.  The lists shipped with world_model.yaml enumerate five
       competitions; the corpus now holds many more, so enumeration silently
       sent nearly every window to train and left the test set empty or tiny.
       The lists are therefore honoured only when they match something.

    2. The rule those lists were written to express: for each competition,
       the most recent season is held out and every earlier season trains.
       Competitions with a single season go entirely to train, since no
       held-out season is possible for them.

    3. If no competition metadata is present at all, a chronological split by
       MATCH.  The previous fallback sorted every window in the corpus by
       window_start_sec, so train got the first 80% of minutes-within-a-match
       and test got the last 20% -- a split by phase of play, in which every
       test window is a late-game window and no match is ever unseen.  Both
       halves of that are leakage in one direction and distribution shift in
       the other.  Splitting whole matches by date, or by match_id when no
       date is available, keeps every match entirely on one side.
    """
    has_meta = "competition_id" in windows.columns and "season_id" in windows.columns
    if has_meta:
        meta = windows.dropna(subset=["competition_id", "season_id"])
        has_meta = len(meta) > 0

    if has_meta:
        pairs = set(map(tuple, meta[["competition_id", "season_id"]].drop_duplicates().values))
        train_keys = {(c["competition_id"], c["season_id"]) for c in cfg.get("train_competitions", [])}
        test_keys  = {(c["competition_id"], c["season_id"]) for c in cfg.get("test_competitions", [])}

        if test_keys & pairs:
            print(f"  split: explicit config lists ({len(test_keys & pairs)} test season(s) present)")
        else:
            if test_keys:
                print("  split: config test_competitions match no season in the data "
                      "-> using the rule they describe (hold out each competition's latest season)")
            test_keys, train_keys = set(), set()
            for cid in sorted({c for c, _ in pairs}):
                seasons = sorted(s for c, s in pairs if c == cid)
                if len(seasons) > 1:
                    test_keys.add((cid, seasons[-1]))
                    train_keys |= {(cid, s) for s in seasons[:-1]}
                else:
                    train_keys.add((cid, seasons[0]))

        key = list(zip(windows["competition_id"], windows["season_id"]))
        test_mask  = pd.Series([k in test_keys for k in key], index=windows.index)
        train_mask = pd.Series([k in train_keys for k in key], index=windows.index)
        unassigned = ~train_mask & ~test_mask
        if unassigned.any():
            print(f"  {int(unassigned.sum()):,} windows had no season metadata -> train")
        train_df = windows[train_mask | unassigned].reset_index(drop=True)
        test_df  = windows[test_mask].reset_index(drop=True)

        if len(test_df) == 0:
            print("  every competition has a single season; no season can be held out "
                  "-> falling back to a chronological split by match")
        else:
            held = sorted(test_keys)
            print(f"  held-out season(s): {held}")
            print(f"  Train: {len(train_df):,} windows | Test: {len(test_df):,} windows")
            return train_df, test_df

    order_col = "match_date" if "match_date" in windows.columns and windows["match_date"].notna().any() else "match_id"
    print(f"  split: chronological 80/20 by match, ordered on {order_col}")
    matches = (
        windows[["match_id", order_col]].drop_duplicates("match_id")
        .sort_values([order_col, "match_id"])["match_id"].tolist()
    )
    cut = int(len(matches) * 0.8)
    train_ids, test_ids = set(matches[:cut]), set(matches[cut:])
    train_df = windows[windows["match_id"].isin(train_ids)].reset_index(drop=True)
    test_df  = windows[windows["match_id"].isin(test_ids)].reset_index(drop=True)
    print(f"  matches: {len(train_ids):,} train | {len(test_ids):,} test  (no match appears in both)")

    print(f"  Train: {len(train_df):,} windows | Test: {len(test_df):,} windows")
    return train_df, test_df


def handle_nan_ppda(train_df, test_df, cfg):
    if cfg.get("drop_nan_ppda", False):
        train_df = train_df.dropna(subset=["ppda"]).reset_index(drop=True)
        test_df  = test_df.dropna(subset=["ppda"]).reset_index(drop=True)
        print(f"  After dropping NaN ppda -> train: {len(train_df):,} | test: {len(test_df):,}")
    else:
        impute_val = train_df["ppda"].mean()
        train_df["ppda"] = train_df["ppda"].fillna(impute_val)
        test_df["ppda"]  = test_df["ppda"].fillna(impute_val)
        print(f"  Imputed NaN ppda with train mean: {impute_val:.3f}")
    return train_df, test_df


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def compute_normalise_stats(train_df):
    stats = {}
    for t in TARGETS:
        mu  = float(train_df[t].mean())
        std = float(train_df[t].std())
        std = std if std > 1e-6 else 1.0
        stats[t] = {"mean": mu, "std": std}
    return stats


def normalise_df(df, stats):
    df = df.copy()
    for t in TARGETS:
        df[t] = (df[t] - stats[t]["mean"]) / stats[t]["std"]
    return df


def denormalise(arr, stats):
    result = arr.clone()
    for i, t in enumerate(TARGETS):
        result[:, i] = arr[:, i] * stats[t]["std"] + stats[t]["mean"]
    return result


# ---------------------------------------------------------------------------
# Graph dataset
# ---------------------------------------------------------------------------

def build_dataset(df, rms_df, sty_df, formation_vocab, cfg):
    graphs, targets = [], []
    vocab_set = set(formation_vocab.keys())
    skipped = 0
    skipped_oov = 0
    build_errors = {}

    for _, row in df.iterrows():
        hf = int(row["home_formation"])
        af = int(row["away_formation"])
        if hf not in vocab_set or af not in vocab_set:
            skipped += 1
            skipped_oov += 1
            continue

        ctx = {
            "minute":          row.get("minute_bucket", 45),
            "scoreline_home":  row.get("scoreline_home", 0),
            "scoreline_away":  row.get("scoreline_away", 0),
            "ppda":            row.get("ppda", None),
            "home_team_id":    row.get("home_team_id", None),
            "away_team_id":    row.get("away_team_id", None),
        }

        try:
            data = build_bipartite_graph(hf, af, ctx, rms_df, sty_df, formation_vocab)
        except Exception as e:
            # A bare counter here hid a systematic failure as ordinary
            # attrition: every window whose formation had no entry in
            # canonical_roles.json raised and was silently dropped. Record
            # the reason so a whole class of rows going missing is visible.
            skipped += 1
            key = f"{type(e).__name__}: {e}"
            build_errors[key] = build_errors.get(key, 0) + 1
            continue

        y = torch.tensor(
            [float(row[t]) for t in TARGETS], dtype=torch.float
        )
        graphs.append(data)
        targets.append(y)

    if skipped:
        pct = 100.0 * skipped / max(len(df), 1)
        print(f"  Skipped {skipped:,} of {len(df):,} rows ({pct:.1f}%): "
              f"{skipped_oov:,} out-of-vocab, {skipped - skipped_oov:,} build errors")
        for reason, count in sorted(build_errors.items(), key=lambda kv: -kv[1])[:5]:
            print(f"      {count:>6}  {reason[:110]}")
        if pct > 20:
            print("  WARNING: more than a fifth of the data was dropped; "
                  "check the formation vocabulary before reading the results.")
    return graphs, targets


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def loss_fn(pred, target, weights):
    w = torch.tensor(weights, dtype=torch.float, device=pred.device)
    sq = (pred - target) ** 2          # (B, 4)
    return (sq * w).mean()


def run_epoch(model, graphs, targets, optimizer, weights, batch_size, device, train=True):
    model.train(train)
    indices = list(range(len(graphs)))
    if train:
        random.shuffle(indices)

    total_loss = 0.0
    all_pred, all_true = [], []

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start: start + batch_size]
        # One batched forward, not one forward per graph. The old line was
        #     preds = torch.stack([model(g) for g in batch_graphs])
        # which ran batch_size separate forward passes and kept batch_size
        # separate autograd graphs alive before a single backward. That is
        # where the epoch time and most of the GPU memory went; the model's
        # forward could not take a batch at all until it was made batch-aware.
        # The outputs are identical -- see test_batched_forward_matches_loop.
        batch = Batch.from_data_list([graphs[i] for i in batch_idx]).to(device)
        batch_targets = torch.stack([targets[i] for i in batch_idx]).to(device)

        preds = model(batch).view(len(batch_idx), -1)           # (B, 4)

        loss = loss_fn(preds, batch_targets, weights)

        if train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * len(batch_idx)
        all_pred.append(preds.detach().cpu())
        all_true.append(batch_targets.detach().cpu())

    all_pred = torch.cat(all_pred, dim=0)
    all_true = torch.cat(all_true, dim=0)
    mean_loss = total_loss / len(graphs)

    per_target_mse = {}
    per_target_r   = {}
    for i, t in enumerate(TARGETS):
        mse = float(((all_pred[:, i] - all_true[:, i]) ** 2).mean())
        per_target_mse[t] = mse
        if all_pred[:, i].std() > 1e-6 and all_true[:, i].std() > 1e-6:
            r, _ = pearsonr(all_pred[:, i].numpy(), all_true[:, i].numpy())
        else:
            r = 0.0
        per_target_r[t] = float(r)

    return mean_loss, per_target_mse, per_target_r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def read_match_list(path):
    """One match_id per line; blank lines and '#' comments ignored."""
    with open(path) as f:
        return {int(x) for x in (ln.split("#")[0].strip() for ln in f) if x}


def train(config_path="configs/world_model.yaml", ablate_style=False, resume=False,
          no_graph=False, exclude_matches=None, tag=None, rms_path=None,ppda_weight=None):
    """Train the bipartite formation GNN.

    ablate_style drops the opponent style cluster from every node feature, so
    the model sees formations, roles and game state but not who it is playing.
    The style-conditioning claim is only supported if the full model beats
    this ablation by more than run-to-run noise; running it is what makes the
    claim falsifiable rather than decorative.
    """
    cfg = load_config(config_path)

    w = list(cfg["loss_weights"])
    if ppda_weight is not None:
        w[2] = float(ppda_weight)
        cfg["loss_weights"] = w
        print(f"Overriding ppda loss weight to {ppda_weight} (B10)")
    elif w[2] < 0.5 * min(w[0], w[1], w[3]):
        print(f"NOTE (B10): ppda loss weight is {w[2]}, while other targets "
              f"are {w[:2] + w[3:]}. ppda gets "
              f"{w[2] / min(w[0], w[1], w[3]):.1f}x less gradient than the "
              f"others. Pass --ppda-weight 1.0 to train a balanced variant.")
    if no_graph:
        cfg["architecture"] = "mlp"
        cfg["checkpoint_path"] = cfg["checkpoint_path"].replace(".pt", "_nograph.pt")
        cfg["log_path"] = cfg["log_path"].replace(".json", "_nograph.json")
    # Cross-fitting. The planner is evaluated on decisions from specific
    # matches; a world model that trained on windows from those same matches
    # has seen how each formation actually fared in them, and its critic
    # scores for those decisions are contaminated (this happened: 92% of the
    # fold-0 decision matches were in training). --exclude-matches removes a
    # list of matches from BOTH train and validation, so neither the weights
    # nor the early-stopping choice has seen them. --tag keeps each such run's
    # checkpoint, normalisation stats and log separate.
    if tag:
        for key in ("checkpoint_path", "log_path", "normalise_stats_path"):
            root, ext = os.path.splitext(cfg[key])
            cfg[key] = f"{root}_{tag}{ext}"
    if rms_path:
        cfg["role_matchup_stats_path"] = rms_path
    excluded = set()
    if exclude_matches:
        excluded = read_match_list(exclude_matches)
        cfg["excluded_matches_file"] = os.path.abspath(exclude_matches)
        cfg["n_excluded_matches"] = len(excluded)
        print(f"CROSS-FIT: excluding {len(excluded)} matches listed in {exclude_matches} "
              f"from train and validation")
    set_seed(cfg["seed"])

    # ppda is measured over the same window as the targets AND is target #3,
    # so feeding it as a node feature leaks the answer. Default off.
    from models.world_model import graph_builder as _gb
    _gb.USE_PPDA_FEATURE = bool(cfg.get("use_ppda_feature", False))
    if _gb.USE_PPDA_FEATURE:
        print("WARNING: use_ppda_feature is on. ppda is target #3 and is measured "
              "over the same window, so its correlation will be inflated by leakage.")
    else:
        print("ppda withheld from node features (it is target #3 measured over the "
              "same window)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["output_dir"],     exist_ok=True)

    # --- data ---
    print("\nLoading data ...")
    windows, rms_df, sty_df, formation_vocab = load_data(cfg)
    print(f"  {len(windows):,} total stable windows")
    if ablate_style:
        sty_df = None
        print("  ABLATION: style clusters withheld (every team reads as cluster 0)")

    train_df, test_df = temporal_split(windows, cfg)
    if excluded:
        n_tr, n_te = len(train_df), len(test_df)
        train_df = train_df[~train_df["match_id"].isin(excluded)].reset_index(drop=True)
        test_df = test_df[~test_df["match_id"].isin(excluded)].reset_index(drop=True)
        print(f"  CROSS-FIT: train {n_tr:,} -> {len(train_df):,} windows, "
              f"val {n_te:,} -> {len(test_df):,} windows")
        if len(train_df) == 0 or len(test_df) == 0:
            raise RuntimeError("exclusion removed every train or val window; wrong match list?")
    train_df, test_df = handle_nan_ppda(train_df, test_df, cfg)

    # normalise
    norm_stats = None
    if cfg.get("normalise_targets", True):
        norm_stats = compute_normalise_stats(train_df)
        with open(cfg["normalise_stats_path"], "w") as f:
            json.dump(norm_stats, f, indent=2)
        print(f"  Normalisation stats saved to {cfg['normalise_stats_path']}")
        train_df = normalise_df(train_df, norm_stats)
        test_df  = normalise_df(test_df,  norm_stats)

    print("\nBuilding graph datasets ...")
    t0 = time.time()
    train_graphs, train_targets = build_dataset(train_df, rms_df, sty_df, formation_vocab, cfg)
    val_graphs,   val_targets   = build_dataset(test_df,  rms_df, sty_df, formation_vocab, cfg)
    print(f"  Train: {len(train_graphs):,} graphs | Val: {len(val_graphs):,} graphs  [{time.time()-t0:.1f}s]")

    if len(train_graphs) == 0:
        raise RuntimeError("No training graphs built. Check formation vocab and stable windows.")

    # --- model ---
    arch = str(cfg.get("architecture", "gnn")).lower()
    Net = MeanPoolMLP if arch in ("mlp", "no_graph", "no-graph") else BipartiteFormationGNN
    if Net is MeanPoolMLP:
        print("ABLATION: no graph. Same node features and readout, no message passing.")
    n_formations = len(formation_vocab)
    model = Net(
        node_feat_dim=cfg["node_feat_dim"],
        node_dim=cfg["node_dim"],
        gat_heads=cfg["gnn_heads"],
        gnn_layers=cfg["gnn_layers"],
        output_dim=cfg["output_dim"],
        edge_feat_dim=cfg["edge_feat_dim"],
        n_formations=n_formations,
    ).to(device)
    print(f"Context features: {2 * n_formations + 5} linear dims "
          f"(2 x {n_formations}-way one-hot + 5 scalars)")
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")

    # --- Option A: freeze the message-passing layers, train only the readout.
    # The graph has 625k parameters and the corpus has 235 observed formation
    # changes; message-passing weights can memorise the training set long
    # before the readout learns anything. Freezing them forces the model to
    # use only the pooled node features + ctx features -- which is
    # approximately "a linear ridge on the graph readout + linear features".
    # If this version beats ridge, the graph readout carries real signal and
    # the overfitting is in the message-passing weights.
    if cfg.get("freeze_graph", False):
        for name, p in model.named_parameters():
            if "output_head" not in name and "trunk" not in name:
                p.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"FREEZE GRAPH: training {trainable:,} of {total:,} parameters")

    optimizer = AdamW(model.parameters(), lr=cfg["lr"])
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )

    weights     = cfg["loss_weights"]
    batch_size  = cfg["batch_size"]
    patience    = cfg["early_stopping_patience"]
    best_val    = float("inf")
    epochs_no_improve = 0
    log_records = []
    start_epoch = 1

    # The GPU here is shared, so a run can be killed part-way by someone else's
    # allocation rather than by anything wrong with it. Without resume, every
    # such interruption threw away the whole run and the next attempt started
    # from epoch 1.
    if resume and os.path.exists(cfg["checkpoint_path"]):
        ck = torch.load(cfg["checkpoint_path"], map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        if "optimizer_state" in ck:
            optimizer.load_state_dict(ck["optimizer_state"])
        if "scheduler_state" in ck:
            scheduler.load_state_dict(ck["scheduler_state"])
        best_val = float(ck.get("val_mse", float("inf")))
        epochs_no_improve = int(ck.get("epochs_no_improve", 0))
        log_records = list(ck.get("log_records", []))
        start_epoch = int(ck["epoch"]) + 1
        print(f"\nResuming from {cfg['checkpoint_path']}: epoch {ck['epoch']} done, "
              f"best val_mse_total={best_val:.4f}, {epochs_no_improve} epoch(s) without "
              f"improvement. Continuing at epoch {start_epoch}.")
        if "optimizer_state" not in ck:
            print("  NOTE: this checkpoint predates optimizer saving, so the AdamW "
                  "moments restart. Expect a bump in the first epoch's loss.")
    elif resume:
        print(f"\n--resume given but {cfg['checkpoint_path']} does not exist; starting fresh.")

    if start_epoch > cfg["epochs"]:
        print(f"Checkpoint is already at epoch {start_epoch - 1} of {cfg['epochs']}; nothing to do.")
        return

    print(f"\nTraining epochs {start_epoch}-{cfg['epochs']} (patience={patience}) ...\n")

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        t_ep = time.time()

        tr_loss, tr_mse, tr_r = run_epoch(
            model, train_graphs, train_targets, optimizer, weights, batch_size, device, train=True
        )
        with torch.no_grad():
            va_loss, va_mse, va_r = run_epoch(
                model, val_graphs, val_targets, optimizer, weights, batch_size, device, train=False
            )

        val_mse_total = sum(va_mse.values())
        scheduler.step(val_mse_total)
        lr_now = optimizer.param_groups[0]["lr"]

        rec = {
            "epoch":         epoch,
            "train_loss":    round(tr_loss, 6),
            "val_loss":      round(va_loss, 6),
            "val_mse_total": round(val_mse_total, 6),
            "lr":            lr_now,
        }
        if cfg.get("log_per_target_metrics", True):
            for t in TARGETS:
                rec[f"val_mse_{t}"]     = round(va_mse[t], 6)
                rec[f"val_pearson_{t}"] = round(va_r[t], 4)
                rec[f"tr_mse_{t}"]      = round(tr_mse[t], 6)
        log_records.append(rec)

        elapsed = time.time() - t_ep
        print(
            f"Epoch {epoch:03d} | tr_loss={tr_loss:.4f} | va_loss={va_loss:.4f} | "
            f"val_mse_total={val_mse_total:.4f} | lr={lr_now:.2e} | {elapsed:.1f}s"
        )
        if cfg.get("log_per_target_metrics", True):
            for t in TARGETS:
                print(f"         {t}: val_mse={va_mse[t]:.4f}  val_r={va_r[t]:.3f}")

        # checkpointing
        if val_mse_total < best_val:
            best_val = val_mse_total
            epochs_no_improve = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                # optimizer and scheduler state so a restart continues rather
                # than warm-starting from weights alone. AdamW carries running
                # moments; dropping them resets the effective learning rate and
                # can undo several epochs of progress on the first step back.
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epochs_no_improve": epochs_no_improve,
                "val_mse":     val_mse_total,
                "val_mse_per_target": va_mse,
                "val_pearson":        va_r,
                "norm_stats":  norm_stats,
                "cfg":         cfg,
                "log_records": log_records,
            }, cfg["checkpoint_path"])
            print(f"  [checkpoint] Saved best model (val_mse_total={best_val:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

        if epoch % cfg.get("save_every_n_epochs", 10) == 0:
            periodic_path = cfg["checkpoint_path"].replace(".pt", f"_ep{epoch}.pt")
            torch.save(model.state_dict(), periodic_path)

    # save log
    with open(cfg["log_path"], "w") as f:
        json.dump(log_records, f, indent=2)
    print(f"\nTraining log saved to {cfg['log_path']}")
    print(f"Best val_mse_total: {best_val:.4f}")
    print(f"Best checkpoint:    {cfg['checkpoint_path']}")

    # --- Save validation-set predictions from BOTH the GNN and ridge.
    # These are the "stacking inputs": the combiner learns on them.
    import numpy as np
    from sklearn.linear_model import Ridge

    print("\nSaving stacking inputs (GNN + ridge predictions on val set) ...")
    _, _, _ = run_epoch(
        model, val_graphs, val_targets, None, weights, batch_size, device, train=False
    )
    # Re-run to capture predictions:
    val_preds_gnn = []
    with torch.no_grad():
        for start in range(0, len(val_graphs), batch_size):
            idx = list(range(start, min(start + batch_size, len(val_graphs))))
            batch = Batch.from_data_list([val_graphs[i] for i in idx]).to(device)
            preds = model(batch).view(len(idx), -1).cpu().numpy()
            val_preds_gnn.append(preds)
    val_preds_gnn = np.concatenate(val_preds_gnn, axis=0)     # (N_val, 4)

    # Ridge predictions: build the same feature matrix the baseline uses
    from baseline_world_model import featurise, temporal_split as _ts
    sty_map = dict(zip(
        pd.read_parquet(cfg["style_clusters_path"])["team_id"].astype(float),
        pd.read_parquet(cfg["style_clusters_path"])["style_cluster"].astype(int),
    ))
    with open(cfg["formation_vocab_path"]) as f:
        vocab = json.load(f)["vocab"]

    # Recover the raw (un-normalised) val_df. The train() function normalised
    # it in-place; we need the raw version. Re-load and re-split.
    raw_windows = pd.read_parquet(cfg["stable_windows_path"])
    _, raw_test = temporal_split(raw_windows, cfg)
    if excluded:
        raw_test = raw_test[~raw_test["match_id"].isin(excluded)].reset_index(drop=True)
    # align with the same val rows
    raw_test = raw_test[raw_test["home_formation"].astype(int).isin(set(vocab))
                        & raw_test["away_formation"].astype(int).isin(set(vocab))].reset_index(drop=True)

    X_tr = featurise(train_df, vocab, sty_map)
    X_va = featurise(raw_test, vocab, sty_map)
    val_preds_ridge = np.zeros_like(val_preds_gnn)
    for j, t in enumerate(TARGETS):
        y_tr = train_df[t].astype(float).values
        y_tr = np.where(np.isfinite(y_tr), y_tr, train_df[t].mean()) if t == "ppda" else y_tr
        r = Ridge(alpha=1.0).fit(X_tr, y_tr)
        val_preds_ridge[:, j] = r.predict(X_va)

    # Reverse the z-scoring so the arrays are in original units
    for j, t in enumerate(TARGETS):
        mu, sd = norm_stats[t]["mean"], norm_stats[t]["std"]
        val_preds_gnn[:, j]   = val_preds_gnn[:, j] * sd + mu
        # ridge already predicts in original units

    np.savez(
        cfg["checkpoint_path"].replace(".pt", "_stacking_inputs.npz"),
        gnn=val_preds_gnn,
        ridge=val_preds_ridge,
        y=raw_test[TARGETS].astype(float).values,
    )
    print(f"  saved to {cfg['checkpoint_path'].replace('.pt', '_stacking_inputs.npz')}")

    return model, log_records


def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="Train the bipartite formation GNN world model.")
    ap.add_argument("--config", default="configs/world_model.yaml")
    ap.add_argument("--ablate-style", action="store_true",
                    help="withhold opponent style clusters (no-style ablation)")
    ap.add_argument("--smoke-test", action="store_true",
                    help="run the synthetic self-test instead of real training")
    ap.add_argument("--no-graph", action="store_true",
                    help="ablation: mean-pool MLP on the same features, no message passing")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the existing best checkpoint instead of starting over")
    ap.add_argument("--exclude-matches", default=None,
                    help="file of match_ids (one per line) to keep out of train AND val "
                         "(cross-fitting for clean planner evaluation)")
    ap.add_argument("--tag", default=None,
                    help="suffix for checkpoint/log/normalisation paths, e.g. excl0")
    ap.add_argument("--rms", default=None,
                    help="role-matchup stats parquet to use instead of the config's "
                         "(pass the one computed without the excluded matches)")
    ap.add_argument("--ppda-weight", type=float, default=None,
                    help="override loss weight for ppda (index 2 of loss_weights). "
                         "Default config uses 0.1; try 1.0 for the balanced variant (B10).")
    return ap.parse_args()


if __name__ == "__main__":
    def smoke_test():
        import tempfile
        print("--- train_world_model smoke test ---")

        # build minimal synthetic data
        rng   = np.random.default_rng(42)
        vocab = [4231,433,442,4141,4312,4321,352,343,532,541,4411,4132,3421,451,4222]
        n     = 80

        windows = pd.DataFrame({
            "match_id":        np.arange(n),
            "window_start_sec": rng.uniform(180, 3600, size=n),
            "home_formation":  rng.choice(vocab, size=n),
            "away_formation":  rng.choice(vocab, size=n),
            "xg_delta":        rng.normal(0, 0.15, size=n),
            "possession_pct":  rng.uniform(0.3, 0.7, size=n),
            "ppda":            rng.uniform(5, 15, size=n),
            "shots_on_target": rng.integers(0, 4, size=n).astype(float),
            "scoreline_home":  rng.integers(0, 3, size=n),
            "scoreline_away":  rng.integers(0, 3, size=n),
            "minute_bucket":   rng.integers(3, 90, size=n),
            "home_team_id":    rng.integers(0, 10, size=n),
            "away_team_id":    rng.integers(10, 20, size=n),
        })

        with open("models/world_model/positional_matchups.json") as f:
            edge_list = json.load(f)["edge_list"]
        rms_df = pd.DataFrame([
            {"attacking_role_idx": a, "defending_role_idx": d, "wr": 0.01, "xg_rate": 0.05}
            for a, d in edge_list
        ])
        sty_df = pd.DataFrame({
            "team_id":      list(range(20)),
            "style_cluster": [i % 4 for i in range(20)],
        })

        with tempfile.TemporaryDirectory() as tmp:
            data_dir  = os.path.join(tmp, "data", "processed")
            ckpt_dir  = os.path.join(tmp, "checkpoints")
            res_dir   = os.path.join(tmp, "results")
            os.makedirs(data_dir)
            os.makedirs(ckpt_dir)
            os.makedirs(res_dir)

            windows.to_parquet(os.path.join(data_dir, "stable_windows.parquet"),   index=False)
            rms_df.to_parquet( os.path.join(data_dir, "role_matchup_stats.parquet"), index=False)
            sty_df.to_parquet( os.path.join(data_dir, "team_style_clusters.parquet"), index=False)
            json.dump(
                {"vocab": vocab},
                open(os.path.join(data_dir, "formation_vocab.json"), "w")
            )

            cfg = load_config("configs/world_model.yaml")
            cfg.update({
                "stable_windows_path":      os.path.join(data_dir, "stable_windows.parquet"),
                "role_matchup_stats_path":  os.path.join(data_dir, "role_matchup_stats.parquet"),
                "style_clusters_path":      os.path.join(data_dir, "team_style_clusters.parquet"),
                "formation_vocab_path":     os.path.join(data_dir, "formation_vocab.json"),
                "checkpoint_path":          os.path.join(ckpt_dir, "world_model_best.pt"),
                "normalise_stats_path":     os.path.join(ckpt_dir, "normalise_stats.json"),
                "log_path":                 os.path.join(res_dir,  "train_log.json"),
                "epochs":                   3,
                "batch_size":               16,
                "early_stopping_patience":  5,
                "train_competitions":       [],
                "test_competitions":        [],
                "temporal_split_fallback":  "chronological_8020",
            })

            # patch load_config to return our modified cfg
            import train_world_model as twm
            _orig = twm.load_config
            twm.load_config = lambda *a, **k: cfg

            model, log = twm.train()

            twm.load_config = _orig  # restore

            assert os.path.exists(cfg["checkpoint_path"]), "Checkpoint not saved"
            assert len(log) >= 1, "No log records"
            ckpt = torch.load(cfg["checkpoint_path"], weights_only=False)
            assert "model_state" in ckpt
            assert "val_mse_per_target" in ckpt
            assert set(ckpt["val_mse_per_target"].keys()) == set(TARGETS)

            print(f"\nSmoke test: {len(log)} epochs logged")
            print(f"  Best val_mse_total = {ckpt['val_mse']:.4f}")
            print(f"  Per-target val MSE = {ckpt['val_mse_per_target']}")
            print("--- smoke test passed ---")

    _args = _cli()
    if _args.smoke_test:
        smoke_test()
    else:
        train(config_path=_args.config, ablate_style=_args.ablate_style,
              resume=_args.resume, no_graph=_args.no_graph,
              exclude_matches=_args.exclude_matches, tag=_args.tag,
              rms_path=_args.rms, ppda_weight=_args.ppda_weight)
