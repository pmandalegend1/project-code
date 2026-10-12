#!/usr/bin/env python3
"""
run_experiments.py -- trains the refactored world model and the Ridge / boosting
baselines, and runs the three evaluations with match-clustered paired bootstrap.

  E-A  learning curve   : train on a fraction of training MATCHES
  E-B  held-out pairs   : 20% of formation matchups never seen in training
  E-C  held-out forms   : 3412, 41212, 3511 never seen in training

Every experiment evaluates on the same kind of test set: windows from the most
recent season of each competition (same rule as baseline_world_model.py), and
for E-B / E-C only the test windows that use a held-out pair / formation.

Fairness notes (also printed in the report):
  * SwapSym forces xg_delta to be exactly antisymmetric, so it cannot learn the
    average home advantage (+0.03 xg). Ridge / boosting have an intercept and can.
  * ppda is NaN in some windows. Training imputes the train mean; evaluation
    ignores those windows for ppda.
  * Early stopping uses a validation set of 10% of the training matches, taken
    before any subsampling, so the validation set is the same size at every n.

Usage (from the project root):
    python run_experiments.py --exp EA --quick          # ~minutes, smoke run
    python run_experiments.py --exp all --seeds 5 --gbm # the real thing
    python run_experiments.py --selftest --assets models/world_model
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import Ridge, RidgeCV  # noqa: E402

from dense_data import (FormationBank, build_edge_stats, ctx_dim, ctx_swap_perm,  # noqa: E402
                        make_arrays)
from eval_splits import (heldout_formation_split, heldout_pair_split,  # noqa: E402
                         learning_curve_train, paired_bootstrap, _pair_key)
from world_model_refactored import build_world_model, make_ctx_swapper, mirror_roles  # noqa: E402

TARGETS = ["xg_delta", "possession_pct", "ppda", "shots_on_target"]
LOSS_W = np.array([3.0, 1.0, 0.1, 1.0], np.float32)     # from configs/world_model.yaml
CELL_W = 0.5


# ----------------------------- data helpers --------------------------------- #
def temporal_split(w):
    """Most recent season of each competition -> test (same rule as the baseline)."""
    meta = w.dropna(subset=["competition_id", "season_id"])
    pairs = set(map(tuple, meta[["competition_id", "season_id"]].drop_duplicates().values))
    test_keys, train_keys = set(), set()
    for cid in sorted({c for c, _ in pairs}):
        seasons = sorted(s for c, s in pairs if c == cid)
        if len(seasons) > 1:
            test_keys.add((cid, seasons[-1]))
            train_keys |= {(cid, s) for s in seasons[:-1]}
        else:
            train_keys.add((cid, seasons[0]))
    key = list(zip(w["competition_id"], w["season_id"]))
    mask = np.array([k in test_keys for k in key])
    return w[~mask].reset_index(drop=True), w[mask].reset_index(drop=True)


def split_val(train_df, frac=0.1, seed=0):
    ms = np.sort(train_df["match_id"].unique())
    rng = np.random.default_rng(seed)
    val_m = set(rng.choice(ms, size=max(1, int(round(frac * len(ms)))), replace=False).tolist())
    is_val = train_df["match_id"].isin(val_m)
    return train_df[~is_val].reset_index(drop=True), train_df[is_val].reset_index(drop=True)


def baseline_features(df, vocab, style_map):
    idx = {c: i for i, c in enumerate(vocab)}
    n = len(df)
    home, away = np.zeros((n, len(vocab)), np.float32), np.zeros((n, len(vocab)), np.float32)
    for r, (hf, af) in enumerate(zip(df["home_formation"], df["away_formation"])):
        if int(hf) in idx:
            home[r, idx[int(hf)]] = 1.0
        if int(af) in idx:
            away[r, idx[int(af)]] = 1.0
    ctx = np.column_stack([
        df["scoreline_home"].astype(float) - df["scoreline_away"].astype(float),
        df["minute_bucket"].astype(float),
        df["period"].astype(float) if "period" in df.columns else np.ones(n),
        df["home_team_id"].map(style_map).fillna(0).astype(float),
        df["away_team_id"].map(style_map).fillna(0).astype(float),
    ]).astype(np.float32)
    return np.hstack([home, away, ctx])


class Env:
    """Everything the fitters need that does not change between experiments."""

    def __init__(self, vocab, canon, matchups, rms, sty, device):
        self.vocab = [int(c) for c in vocab]
        self.bank = FormationBank(self.vocab, canon)
        self.edge_stats = build_edge_stats(matchups, rms)
        self.style_map = dict(zip(sty["team_id"].astype(float), sty["style_cluster"].astype(int)))
        self.n_style = int(min(8, max(2, sty["style_cluster"].max() + 1)))
        self.ctx_dim = ctx_dim(self.n_style)
        perm, sign = ctx_swap_perm(self.n_style)
        self.swapper = make_ctx_swapper(perm, sign)
        self.device = device
        self.roles_t = torch.as_tensor(self.bank.roles, device=device)
        self.adj_t = torch.as_tensor(self.bank.adj, device=device)


def prep(df, env, mean, std, ppda_fill):
    """DataFrame -> arrays (y z-scored with the SwapSym centres) + raw eval targets."""
    df = df[df["home_formation"].astype(int).isin(env.bank.index)
            & df["away_formation"].astype(int).isin(env.bank.index)].reset_index(drop=True)
    raw = df[TARGETS].astype(float).values
    y_fill = raw.copy()
    y_fill[~np.isfinite(y_fill[:, 2]), 2] = ppda_fill
    y_z = ((y_fill - mean) / std).astype(np.float32)
    arr, _ = make_arrays(df, env.bank, env.style_map, env.n_style, y_z)
    arr["y_raw"] = raw
    arr["df"] = df
    return arr


# ------------------------------- GNN ---------------------------------------- #
def _batch(env, arr, idx, mirror=False):
    hi = torch.as_tensor(arr["home_idx"][idx], device=env.device)
    ai = torch.as_tensor(arr["away_idx"][idx], device=env.device)
    hr, ar = env.roles_t[hi], env.roles_t[ai]
    if mirror:
        m = torch.rand(len(idx), device=env.device) < 0.5
        hr = torch.where(m.unsqueeze(1), mirror_roles(hr), hr)
        ar = torch.where(m.unsqueeze(1), mirror_roles(ar), ar)
    ctx = torch.as_tensor(arr["ctx"][idx], device=env.device)
    return hr, env.adj_t[hi], ar, env.adj_t[ai], ctx


@torch.no_grad()
def predict_gnn(model, env, arr, bs=2048):
    model.eval()
    out = []
    for s in range(0, len(arr["y"]), bs):
        idx = np.arange(s, min(s + bs, len(arr["y"])))
        out.append(model(*_batch(env, arr, idx)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 4))


def fit_gnn(env, tr, va, mean, std, seed, a):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_world_model(env.edge_stats, env.ctx_dim, env.swapper, mean, std,
                              poss_center=0.5, node_dim=a.node_dim, layers=a.layers, heads=4,
                              hidden=2 * a.node_dim).to(env.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    w = torch.as_tensor(LOSS_W, device=env.device)
    y_tr = torch.as_tensor(tr["y"], device=env.device)
    y_cell = torch.as_tensor(tr["y_cell"], device=env.device)
    n = len(y_tr)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(a.max_epochs):
        model.train()
        order = rng.permutation(n)
        for s in range(0, n, a.batch):
            idx = order[s:s + a.batch]
            if len(idx) < 2:
                continue
            pz = model.zscore(model(*_batch(env, tr, idx, mirror=True)))
            yi = torch.as_tensor(idx, device=env.device)
            loss = (w * (pz - y_tr[yi]) ** 2).mean() + CELL_W * (w * (pz - y_cell[yi]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        pv = (predict_gnn(model, env, va) - mean) / std
        v = float(((pv - va["y"]) ** 2).mean())
        if v < best - 1e-5:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= a.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best, ep + 1


# --------------------------- baselines -------------------------------------- #
def fit_baselines(env, tr_df, ev_df, ppda_fill, gbm):
    Xtr, Xev = (baseline_features(d, env.vocab, env.style_map) for d in (tr_df, ev_df))
    out = {"ridge": np.zeros((len(ev_df), 4))}
    if gbm:
        out["gbm"] = np.zeros((len(ev_df), 4))
    for j, t in enumerate(TARGETS):
        y = tr_df[t].astype(float).values
        y = np.where(np.isfinite(y), y, ppda_fill) if t == "ppda" else y
        alphas = np.logspace(-1, 3, 20)
        out["ridge"][:, j] = RidgeCV(alphas=alphas).fit(Xtr, y).predict(Xev)
        if gbm:
            out["gbm"][:, j] = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(Xtr, y).predict(Xev)
    return out


# --------------------------- one setting ------------------------------------ #
def run_setting(env, train_df, val_df, eval_df, seeds, a, label, subsample=None):
    """Fit GNN (+ baselines) for each seed; return aligned eval predictions."""
    preds = {"gnn": [], "ridge": [], **({"gbm": []} if a.gbm else {})}
    info = []
    ev0 = None
    for sd in seeds:
        tr_df = subsample(train_df, sd) if subsample else train_df
        ppda_fill = float(tr_df["ppda"].mean())
        mean = np.array([0.0, 0.5, ppda_fill, float(tr_df["shots_on_target"].mean())], np.float32)
        std = np.maximum(tr_df[TARGETS].astype(float).std().values, 1e-6).astype(np.float32)
        tr, va, ev = (prep(d, env, mean, std, ppda_fill) for d in (tr_df, val_df, eval_df))
        if ev0 is None:
            ev0 = ev
        t0 = time.time()
        model, vbest, epochs = fit_gnn(env, tr, va, mean, std, sd, a)
        preds["gnn"].append(predict_gnn(model, env, ev))
        base = fit_baselines(env, tr["df"], ev["df"], ppda_fill, a.gbm)
        for k, v in base.items():
            preds[k].append(v)
        info.append({"seed": sd, "n_train_windows": int(len(tr["y"])),
                     "n_train_matches": int(tr_df["match_id"].nunique()),
                     "epochs": epochs, "val_z_mse": vbest, "sec": round(time.time() - t0, 1)})
        print(f"   [{label}] seed {sd}: train {info[-1]['n_train_windows']:,} windows "
              f"({info[-1]['n_train_matches']} matches), {epochs} epochs, "
              f"val z-mse {vbest:.4f}, {info[-1]['sec']}s", flush=True)
    return {"y": ev0["y_raw"], "match": ev0["match_id"], "preds": preds, "info": info}


def summarise(res, label, a):
    rows = []
    for j, t in enumerate(TARGETS):
        y = res["y"][:, j]
        for other in [k for k in res["preds"] if k != "gnn"]:
            r = paired_bootstrap(y, [p[:, j] for p in res["preds"]["gnn"]],
                                 [p[:, j] for p in res["preds"][other]], res["match"],
                                 n_boot=a.n_boot, seed=0)
            rows.append({"setting": label, "target": t, "vs": other, **r})
    return rows


def print_rows(rows):
    print(f"\n{'setting':<14}{'target':<17}{'vs':<7}{'n_win':>8}{'mse gnn':>10}{'mse other':>11}"
          f"{'diff':>9}{'95% CI':>20}{'P(gnn<)':>9}")
    for r in rows:
        ci = f"({r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f})"
        star = " *" if r["ci95"][1] < 0 else (" (worse)" if r["ci95"][0] > 0 else "")
        print(f"{r['setting']:<14}{r['target']:<17}{r['vs']:<7}{r['n_windows']:>8,}{r['mse_a']:>10.4f}"
              f"{r['mse_b']:>11.4f}{r['diff']:>+9.4f}{ci:>20}{r['p_a_better']:>9.2f}{star}")
    print("diff = mse(gnn) - mse(other); negative is better for the GNN; * = 95% CI excludes 0 in the GNN's favour")


# ------------------------------ experiments --------------------------------- #
def experiments(env, w, a):
    train_all, test = temporal_split(w)
    seeds = list(range(a.seeds))
    out, rows = {}, []
    print(f"train {len(train_all):,} windows / {train_all.match_id.nunique():,} matches | "
          f"test {len(test):,} windows / {test.match_id.nunique():,} matches")

    if a.exp in ("EA", "all"):
        pool, val = split_val(train_all, 0.1, seed=123)
        for frac in a.fracs:
            label = f"EA n={frac:.0%}"
            print(f"\n== {label}")
            res = run_setting(env, pool, val, test, seeds, a, label,
                              subsample=lambda d, sd, f=frac: learning_curve_train(d, f, seed=1000 + sd))
            out[label] = res["info"]
            rows += summarise(res, label, a)

    if a.exp in ("EB", "all"):
        tr, ev, held = heldout_pair_split(train_all, test, 0.2, seed=0)
        pool, val = split_val(tr, 0.1, seed=123)
        label = "EB held pairs"
        print(f"\n== {label}: {len(held)} held-out matchups, eval {len(ev):,} windows "
              f"({ev.match_id.nunique()} matches), train {len(pool):,}")
        if len(ev) == 0:
            print("   no test windows use a held-out pair; skipping")
        else:
            res = run_setting(env, pool, val, ev, seeds, a, label)
            out[label] = {"held_pairs": [list(map(int, p)) for p in sorted(held)], "runs": res["info"]}
            rows += summarise(res, label, a)

    if a.exp in ("EC", "all"):
        forms = (3412, 41212, 3511)
        tr, ev = heldout_formation_split(train_all, test, forms)
        pool, val = split_val(tr, 0.1, seed=123)
        label = "EC held forms"
        print(f"\n== {label}: {forms}, eval {len(ev):,} windows ({ev.match_id.nunique()} matches), "
              f"train {len(pool):,}")
        if len(ev) == 0:
            print("   no test windows use these formations; skipping")
        else:
            res = run_setting(env, pool, val, ev, seeds, a, label)
            out[label] = {"formations": list(forms), "runs": res["info"]}
            rows += summarise(res, label, a)

    print_rows(rows)
    return out, rows


# ------------------------------- self-test ---------------------------------- #
def _synthetic(vocab, n_matches=120, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_matches):
        hf, af = rng.choice(vocab, 2)
        season = 1 if m < 90 else 2
        for k in range(10):
            sh, sa = int(rng.integers(0, 3)), int(rng.integers(0, 3))
            base = 0.02 * ((hf % 7) - (af % 7))
            rows.append(dict(match_id=m, window_start_sec=k * 300, window_end_sec=k * 300 + 300,
                             home_team_id=int(rng.integers(1, 6)), away_team_id=int(rng.integers(1, 6)),
                             home_formation=int(hf), away_formation=int(af),
                             xg_delta=base + rng.normal(0, 0.3),
                             possession_pct=float(np.clip(0.5 + 3 * base + rng.normal(0, 0.1), 0, 1)),
                             ppda=float(rng.gamma(3, 4)) if rng.random() > 0.05 else np.nan,
                             shots_on_target=float(rng.poisson(1)),
                             scoreline_home=sh, scoreline_away=sa, minute_bucket=int(k * 5 + rng.integers(0, 5)),
                             period=1 if k < 5 else 2, competition_id=1, season_id=season,
                             match_date=f"2020-0{season}-01"))
    return pd.DataFrame(rows)


def selftest(assets):
    vocab = [442, 433, 4231, 352, 3412, 41212, 3511, 343]
    canon = json.load(open(os.path.join(assets, "canonical_roles.json")))
    match = json.load(open(os.path.join(assets, "positional_matchups.json")))
    sys.path.insert(0, assets)
    sty = pd.DataFrame({"team_id": [1, 2, 3, 4, 5], "style_cluster": [0, 1, 0, 1, 0]})
    env = Env(vocab, canon, match, None, sty, torch.device("cpu"))
    w = _synthetic(vocab)
    a = argparse.Namespace(exp="all", seeds=2, fracs=[0.5], max_epochs=3, patience=2, batch=64, lr=1e-3,
                           node_dim=32, layers=2, gbm=True, n_boot=100)
    out, rows = experiments(env, w, a)
    assert rows and all(np.isfinite(r["mse_a"]) for r in rows)
    got = {r["setting"] for r in rows}
    print(f"\nsettings produced: {sorted(got)}")
    assert any(s.startswith("EA") for s in got)
    print("selftest passed")


# --------------------------------- main ------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/world_model.yaml")
    ap.add_argument("--exp", default="all", choices=["EA", "EB", "EC", "all"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--fracs", default="0.01,0.02,0.05,0.10,0.25,1.0")
    ap.add_argument("--max-epochs", type=int, default=40, dest="max_epochs")
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--node-dim", type=int, default=64, dest="node_dim")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--gbm", action="store_true", help="also run HistGradientBoosting baseline")
    ap.add_argument("--n-boot", type=int, default=1000, dest="n_boot")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quick", action="store_true", help="1 seed, 5%% data, 3 epochs: smoke run")
    ap.add_argument("--out", default="results/ramfp_experiments.json")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--assets", default="models/world_model")
    a = ap.parse_args()

    if a.selftest:
        return selftest(a.assets)

    import yaml
    cfg = yaml.safe_load(open(a.config))
    a.fracs = [float(x) for x in a.fracs.split(",")]
    if a.quick:
        a.seeds, a.fracs, a.max_epochs, a.patience, a.n_boot = 1, [0.05], 3, 2, 200
    device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if a.device == "auto" else a.device))
    print(f"device: {device}")

    w = pd.read_parquet(cfg["stable_windows_path"])
    vocab = json.load(open(cfg["formation_vocab_path"]))["vocab"]
    canon = json.load(open(cfg["canon_roles_path"]))
    match = json.load(open(cfg["matchups_path"]))
    rms = pd.read_parquet(cfg["role_matchup_stats_path"])
    sty = pd.read_parquet(cfg["style_clusters_path"])
    sys.path.insert(0, os.path.dirname(cfg["canon_roles_path"]))
    env = Env(vocab, canon, match, rms, sty, device)

    out, rows = experiments(env, w, a)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"args": {k: v for k, v in vars(a).items()}, "runs": out, "comparisons": rows},
              open(a.out, "w"), indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
