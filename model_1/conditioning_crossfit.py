"""
conditioning_crossfit.py  (v3)
"""
import json, os, sys
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FEAT_DIR = os.path.join(HERE, "results")
TRIPLES = os.path.join(HERE, "data", "processed", "triples.parquet")
OUT = os.path.join(FEAT_DIR, "conditioning_crossfit.json")

sys.path.insert(0, HERE)
from learn_ranker import fit_clogit
print("using learn_ranker.fit_clogit")


def backline_bucket(f_opp):
    s = str(int(f_opp)) if not pd.isna(f_opp) else ""
    if not s: return "other"
    d = int(s[0]) if s[0].isdigit() else 4
    return {3: "back3", 4: "back4", 5: "back5"}.get(d, "other")

def sb_fine(gd): return 0 if gd <= -2 else (1 if gd == -1 else (2 if gd == 0 else (3 if gd == 1 else 4)))
def sb_coarse(gd): return 0 if gd <= -1 else (1 if gd == 0 else 2)
def mb_fine(m): return 0 if m < 30 else (1 if m < 60 else (2 if m < 75 else 3))
def mb_coarse(m): return 0 if m < 30 else (1 if m < 60 else 2)


def smoothed_lp(tr, key_fn, marg, total, n_classes, laplace=0.5):
    cond = defaultdict(Counter)
    for _, r in tr.iterrows():
        cond[(int(r["f_src"]), key_fn(r))][int(r["f_tgt"])] += 1
    def lp(f_src, key, f_tgt):
        cell = cond.get((f_src, key), Counter())
        ctot = sum(cell.values())
        pm = (marg.get(f_tgt, 0) + laplace) / (total + laplace * n_classes)
        p = (cell.get(f_tgt, 0) + laplace * pm) / (ctot + laplace)
        return float(np.log(max(p, 1e-9)))
    return lp


def fit_and_eval(feat_cols, trk, fk):
    trk = trk.copy(); fk = fk.copy()
    trk["_is_truth"] = (trk["f_tgt"] == trk["truth"]).astype(int)
    fk["_is_truth"]  = (fk["f_tgt"]  == fk["truth"]).astype(int)

    trk_groups = []
    for _, g in trk.groupby("decision", sort=False):
        g = g.reset_index(drop=True)
        t = g.index[g["_is_truth"] == 1].tolist()
        if not t: continue
        pos = t[0]
        order = [pos] + [i for i in g.index if i != pos]
        X_mat = g.loc[order, feat_cols].fillna(0).values
        # truth is always at index 0 in this ordering
        trk_groups.append((X_mat, 0))

    if not trk_groups:
        return None
    w = fit_clogit(trk_groups)

    t1 = t3 = 0; mrr = 0.0; n = 0
    for _, g in fk.groupby("decision", sort=False):
        g = g.reset_index(drop=True)
        if not (g["_is_truth"] == 1).any(): continue
        sc = g[feat_cols].fillna(0).values @ w
        tp = int(g.index[g["_is_truth"] == 1][0])
        rank = int(np.where(np.argsort(-sc) == tp)[0][0]) + 1
        if rank == 1: t1 += 1
        if rank <= 3: t3 += 1
        mrr += 1.0 / rank; n += 1
    if n == 0: return None
    return t1/n, t3/n, mrr/n, n

def main():
    triples = pd.read_parquet(TRIPLES)
    triples["match_id"] = triples["match_id"].astype(str)
    print(f"triples: {len(triples)} rows")

    feats_all = []
    match_to_fold = {}
    for k in range(5):
        p = os.path.join(FEAT_DIR, f"ranker_features_crossfit_fold{k}.parquet")
        d = pd.read_parquet(p)
        d["match_id"] = d["match_id"].astype(str)
        d["_fold"] = k
        for m in d["match_id"].unique():
            match_to_fold[m] = k
        feats_all.append(d)
    feats = pd.concat(feats_all, ignore_index=True)
    print(f"features: {len(feats)} rows, {len(match_to_fold)} unique matches")

    triples["_fold"] = triples["match_id"].map(match_to_fold)
    triples = triples[triples["_fold"].notna()].copy()
    triples["_fold"] = triples["_fold"].astype(int)
    print(f"triples with fold: {len(triples)}")

    triples["fine_bucket"]   = [f"{sb_fine(int(gd))}_{mb_fine(int(m))}"
                                 for gd, m in zip(triples["goal_diff"], triples["minute"])]
    triples["coarse_bucket"] = [f"{sb_coarse(int(gd))}_{mb_coarse(int(m))}"
                                 for gd, m in zip(triples["goal_diff"], triples["minute"])]
    triples["backline"]      = triples["f_src"].apply(backline_bucket)

    feats = feats.merge(
        triples[["match_id", "f_src", "f_tgt", "fine_bucket", "coarse_bucket", "backline"]],
        on=["match_id", "f_src", "f_tgt"], how="left")

    variants = [
        ("state encoding z_t", "z"),
        ("fine score x minute (20 cells)", "fine"),
        ("coarse score x minute (9 cells)", "coarse"),
        ("opponent-style-conditional", "style"),
        ("opponent-backline-conditional", "backline"),
        ("team-conditional", "team"),
        ("marginal baseline", "marginal"),
    ]

    results = {}
    for label, kind in variants:
        print(f"\n=== {label} ===")
        t1s, t3s, mrrs = [], [], []
        try:
            for k in range(5):
                tr = triples[triples["_fold"] != k]
                marg = Counter(int(x) for x in tr["f_tgt"])
                total = sum(marg.values()); nc = len(marg)
                if total == 0: continue

                fk  = feats[feats["_fold"] == k].copy()
                trk = feats[feats["_fold"] != k].copy()

                if kind == "z":
                    feat_cols = [c for c in fk.columns if c.startswith("z_")]
                else:
                    if kind == "marginal":
                        fk["cond"] = fk["prior"]; trk["cond"] = trk["prior"]
                    elif kind == "style":
                        lp = smoothed_lp(tr, lambda r: int(r["opp_style"]), marg, total, nc)
                        fk["cond"]  = [lp(int(r["f_src"]), int(r["opp_style"]), int(r["f_tgt"])) for _, r in fk.iterrows()]
                        trk["cond"] = [lp(int(r["f_src"]), int(r["opp_style"]), int(r["f_tgt"])) for _, r in trk.iterrows()]
                    elif kind == "backline":
                        lp = smoothed_lp(tr, lambda r: str(r["backline"]), marg, total, nc)
                        fk["cond"]  = [lp(int(r["f_src"]), str(r["backline"]), int(r["f_tgt"])) for _, r in fk.iterrows()]
                        trk["cond"] = [lp(int(r["f_src"]), str(r["backline"]), int(r["f_tgt"])) for _, r in trk.iterrows()]
                    elif kind == "team":
                        lp = smoothed_lp(tr, lambda r: int(r["team_id"]), marg, total, nc)
                        fk["cond"]  = [lp(int(r["f_src"]), int(r["team_id"]), int(r["f_tgt"])) for _, r in fk.iterrows()]
                        trk["cond"] = [lp(int(r["f_src"]), int(r["team_id"]), int(r["f_tgt"])) for _, r in trk.iterrows()]
                    elif kind == "fine":
                        lp = smoothed_lp(tr, lambda r: str(r["fine_bucket"]), marg, total, nc)
                        fk["cond"]  = [lp(int(r["f_src"]), str(r["fine_bucket"]), int(r["f_tgt"])) for _, r in fk.iterrows()]
                        trk["cond"] = [lp(int(r["f_src"]), str(r["fine_bucket"]), int(r["f_tgt"])) for _, r in trk.iterrows()]
                    elif kind == "coarse":
                        lp = smoothed_lp(tr, lambda r: str(r["coarse_bucket"]), marg, total, nc)
                        fk["cond"]  = [lp(int(r["f_src"]), str(r["coarse_bucket"]), int(r["f_tgt"])) for _, r in fk.iterrows()]
                        trk["cond"] = [lp(int(r["f_src"]), str(r["coarse_bucket"]), int(r["f_tgt"])) for _, r in trk.iterrows()]
                    feat_cols = ["cond", "prior", "xG", "stability", "counter_resistance"]

                res = fit_and_eval(feat_cols, trk, fk)
                if res is None:
                    print(f"  fold {k}: no test rows")
                    continue
                t1, t3, mrr, n = res
                t1s.append(t1); t3s.append(t3); mrrs.append(mrr)
                print(f"  fold {k}: top-1 {t1:.4f}  top-3 {t3:.4f}  MRR {mrr:.4f}  (n={n})")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            continue

        if t1s:
            results[label] = {
                "top1": float(np.mean(t1s)),
                "top3": float(np.mean(t3s)),
                "mrr":  float(np.mean(mrrs)),
                "per_fold_top1": [float(x) for x in t1s],
            }
            print(f"  AVG: top-1 {np.mean(t1s):.4f}  top-3 {np.mean(t3s):.4f}  MRR {np.mean(mrrs):.4f}")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT}")

if __name__ == "__main__":
    main()