import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_FORMATIONS  = 15
FORMATION_EMB_DIM = 32
NODE_DIM      = 128     # must match BipartiteFormationGNN.node_dim
DEFAULT_WEIGHTS = (1/3, 1/3, 1/3)   # w1, w2, w3 (paper §5.7 equal weights)

CANON_ROLES_PATH = "models/world_model/canonical_roles.json"


# ---------------------------------------------------------------------------
# Shared formation structural similarity
# ---------------------------------------------------------------------------

def _load_formation_structural_features(canon_path=CANON_ROLES_PATH):
    """
    Build a simple structural feature vector per formation:
    [n_defenders/11, n_midfielders/11, n_attackers/11, has_cdm, has_cam]
    Used to compute ΔStruct (L2 distance between src and tgt GNN embeddings surrogate).
    """
    with open(canon_path) as f:
        d = json.load(f)

    # canonical role idx -> position group
    group = {
        0: "gk",
        1: "def", 2: "def", 3: "def", 4: "def", 5: "def",
        6: "mid", 7: "mid", 8: "att",
        9: "att", 10: "att",
    }

    features = {}
    for code_str, fm in d["formation_mappings"].items():
        code   = int(code_str)
        roles  = fm["node_assignments"]   # 11 canonical idx
        groups = [group[r] for r in roles]
        n_def  = groups.count("def") / 11
        n_mid  = groups.count("mid") / 11
        n_att  = groups.count("att") / 11
        has_cdm = float(6 in roles)
        has_cam = float(8 in roles)
        features[code] = np.array([n_def, n_mid, n_att, has_cdm, has_cam], dtype=np.float32)

    return features   # {formation_code: np.array(5,)}


_STRUCT_FEATURES = None

def _get_struct_features():
    global _STRUCT_FEATURES
    if _STRUCT_FEATURES is None:
        _STRUCT_FEATURES = _load_formation_structural_features()
    return _STRUCT_FEATURES


def structural_distance(f_src_code, f_tgt_code):
    """L2 distance between formation structural feature vectors."""
    feats = _get_struct_features()
    v_src = feats.get(int(f_src_code))
    v_tgt = feats.get(int(f_tgt_code))
    if v_src is None or v_tgt is None:
        return 1.0   # unknown formation -> maximum distance
    return float(np.linalg.norm(v_src - v_tgt))


# ---------------------------------------------------------------------------
# Critic 1: xG Critic
# ---------------------------------------------------------------------------

class XGCritic:
    """
    C_xG(τ, f_opp, st) = ŷ_xG(f_tgt, f_opp, st)
    Uses the world model output directly. Positive = attacking advantage.
    Paper §5.7 Eq. 24.
    """

    def score(self, world_model_output):
        """
        world_model_output: Tensor (4,) or (B, 4)
            [xg_delta, possession_pct, ppda, shots_on_target]
        Returns scalar float (or (B,) tensor).
        """
        if isinstance(world_model_output, torch.Tensor):
            if world_model_output.dim() == 1:
                return world_model_output[0].item()
            return world_model_output[:, 0]   # (B,)
        return float(world_model_output[0])


# ---------------------------------------------------------------------------
# Critic 2: Formation Stability Critic
# ---------------------------------------------------------------------------

class StabilityMLP(nn.Module):
    """
    Small MLP predicting transition risk from structural features.
    Input: [emb(f_src); emb(f_tgt); Δ_struct]  dim = 32+32+1 = 65
    Output: scalar logit (sigmoid → concession probability)
    Paper §5.7 Eq. 25.
    """
    def __init__(self, formation_emb_dim=FORMATION_EMB_DIM, n_formations=N_FORMATIONS):
        super().__init__()
        self.formation_emb = nn.Embedding(n_formations, formation_emb_dim)
        in_dim = formation_emb_dim * 2 + 1   # 65
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, fsrc_idx, ftgt_idx, delta_struct):
        """
        fsrc_idx, ftgt_idx: (B,) LongTensor
        delta_struct:       (B, 1) or (B,) FloatTensor
        Returns (B,) logit
        """
        e_src = self.formation_emb(fsrc_idx)     # (B, 32)
        e_tgt = self.formation_emb(ftgt_idx)     # (B, 32)
        if delta_struct.dim() == 1:
            delta_struct = delta_struct.unsqueeze(-1)    # (B, 1)
        x = torch.cat([e_src, e_tgt, delta_struct], dim=-1)   # (B, 65)
        return self.net(x).squeeze(-1)           # (B,)


class StabilityCritic:
    """
    C_stab(τ) = 1 - σ(MLP_stab([emb(f_src); emb(f_tgt); Δ_struct]))
    Higher score = lower transition risk.
    Paper §5.7 Eq. 25.
    """
    def __init__(self, mlp: StabilityMLP, formation_vocab: dict):
        """
        mlp:             trained StabilityMLP
        formation_vocab: {formation_code (int): idx (int)}
        """
        self.mlp   = mlp
        self.vocab = formation_vocab

    def score(self, fsrc_code, ftgt_code):
        """
        Returns scalar float in (0, 1). Higher = more stable.
        """
        self.mlp.eval()
        with torch.no_grad():
            fsrc_idx     = torch.tensor([self.vocab[int(fsrc_code)]], dtype=torch.long)
            ftgt_idx     = torch.tensor([self.vocab[int(ftgt_code)]], dtype=torch.long)
            delta        = torch.tensor([structural_distance(fsrc_code, ftgt_code)],
                                        dtype=torch.float)
            logit        = self.mlp(fsrc_idx, ftgt_idx, delta)
            risk         = torch.sigmoid(logit).item()
        return 1.0 - risk   # invert: 1 = stable, 0 = high risk


# ---------------------------------------------------------------------------
# Critic 3: Counter-Resistance Critic
# ---------------------------------------------------------------------------

class CounterResistanceCritic:
    """
    C_res(τ, f_opp) = ŷ_hist_xG(f_tgt, f_opp)

    Queries the world model over all training windows that involved
    structurally analogous formation pairs (structural similarity, not exact
    code equality) and returns the mean predicted xG delta.
    Paper §5.7 Eq. 26.

    Implementation note: rather than re-running GNN inference over all training
    windows at planning time, we pre-compute a lookup table of mean predicted
    xG delta per (f_tgt_canonical_group, f_opp_canonical_group) pair during
    Stage 7 of the training pipeline and load it here.

    Canonical group = structural family:
      "4back_double_pivot"  : 4231, 4222, 4132
      "4back_flat_mid"      : 442, 4141, 451, 4411
      "4back_attack_mid"    : 4312, 4321
      "4back_three_forward" : 433
      "3back"               : 352, 343, 3421
      "5back"               : 532, 541
    """

    STRUCTURAL_GROUPS = {
        4231: "4back_double_pivot",
        4222: "4back_double_pivot",
        4132: "4back_double_pivot",
        442:  "4back_flat_mid",
        4141: "4back_flat_mid",
        451:  "4back_flat_mid",
        4411: "4back_flat_mid",
        4312: "4back_attack_mid",
        4321: "4back_attack_mid",
        433:  "4back_three_forward",
        352:  "3back",
        343:  "3back",
        3421: "3back",
        532:  "5back",
        541:  "5back",
    }

    ALL_GROUPS = [
        "4back_double_pivot",
        "4back_flat_mid",
        "4back_attack_mid",
        "4back_three_forward",
        "3back",
        "5back",
    ]

    def __init__(self, lookup_table: dict):
        """
        lookup_table: {(tgt_group, opp_group): mean_xg_delta (float)}
        Built by build_counter_resistance_lookup() below.
        """
        self.lookup = lookup_table

    def score(self, ftgt_code, fopp_code):
        """
        Returns mean historical xG delta for this structural matchup.
        Falls back to 0.0 if pair not found in lookup.
        """
        tgt_grp = self.STRUCTURAL_GROUPS.get(int(ftgt_code), "unknown")
        opp_grp = self.STRUCTURAL_GROUPS.get(int(fopp_code), "unknown")
        return float(self.lookup.get((tgt_grp, opp_grp), 0.0))


def build_counter_resistance_lookup(world_model, stable_windows_df,
                                    rms_df, sty_df, formation_vocab,
                                    device=None):
    """
    Pre-compute the counter-resistance lookup table by running the trained
    world model over all stable training windows, grouping by structural
    family pair, and averaging predicted xG delta.

    Called once after world model training (Stage 7 in pipeline).
    Returns {(tgt_group, opp_group): float}
    """
    from models.world_model.graph_builder import build_bipartite_graph

    if device is None:
        device = torch.device("cpu")

    groups = CounterResistanceCritic.STRUCTURAL_GROUPS
    vocab_set = set(formation_vocab.keys())
    world_model.eval()

    accumulator = {}   # (tgt_group, opp_group) -> list of predicted xg_delta

    with torch.no_grad():
        for _, row in stable_windows_df.iterrows():
            hf = int(row["home_formation"])
            af = int(row["away_formation"])
            if hf not in vocab_set or af not in vocab_set:
                continue

            ctx = {
                "minute":         row.get("minute_bucket", 45),
                "scoreline_home": row.get("scoreline_home", 0),
                "scoreline_away": row.get("scoreline_away", 0),
                "ppda":           row.get("ppda", None),
                "home_team_id":   row.get("home_team_id", None),
                "away_team_id":   row.get("away_team_id", None),
            }
            try:
                data   = build_bipartite_graph(hf, af, ctx, rms_df, sty_df, formation_vocab)
                output = world_model(data.to(device))   # (4,)
                pred_xg = output[0].item()
            except Exception:
                continue

            tgt_grp = groups.get(hf, "unknown")
            opp_grp = groups.get(af, "unknown")
            key     = (tgt_grp, opp_grp)
            accumulator.setdefault(key, []).append(pred_xg)

    lookup = {k: float(np.mean(v)) for k, v in accumulator.items()}
    return lookup


# ---------------------------------------------------------------------------
# Aggregate scorer
# ---------------------------------------------------------------------------

class MultiCriticScorer:
    """
    Combines all three critics into a single score per transition candidate.
    Score(τ, st, f_opp) = w1*C_xG + w2*C_stab + w3*C_res   (paper Eq. 27)
    """

    def __init__(
        self,
        xg_critic:   XGCritic,
        stab_critic: StabilityCritic,
        res_critic:  CounterResistanceCritic,
        weights:     tuple = DEFAULT_WEIGHTS,
    ):
        self.xg   = xg_critic
        self.stab = stab_critic
        self.res  = res_critic
        self.w    = weights

    def score(self, fsrc_code, ftgt_code, fopp_code, world_model_output):
        """
        Returns (total_score, breakdown_dict).
        breakdown_dict keys: xg_score, stab_score, res_score
        """
        c_xg   = self.xg.score(world_model_output)
        c_stab = self.stab.score(fsrc_code, ftgt_code)
        c_res  = self.res.score(ftgt_code, fopp_code)

        total = self.w[0] * c_xg + self.w[1] * c_stab + self.w[2] * c_res

        return float(total), {
            "xg_score":   float(c_xg),
            "stab_score": float(c_stab),
            "res_score":  float(c_res),
        }

    def rank_candidates(self, candidates):
        # compute raw scores
        raw = [self.score(c["fsrc_code"], c["ftgt_code"], c["fopp_code"], c["world_model_output"])
            for c in candidates]
        # raw is [(total, {xg, stab, res}), ...]
        xg_arr = np.array([r[1]["xg_score"] for r in raw])
        stab_arr = np.array([r[1]["stab_score"] for r in raw])
        res_arr = np.array([r[1]["res_score"] for r in raw])
        # z-score within decision
        def z(a):
            s = a.std()
            return (a - a.mean()) / s if s > 1e-9 else np.zeros_like(a)
        xg_z, stab_z, res_z = z(xg_arr), z(stab_arr), z(res_arr)
        total = self.w[0] * xg_z + self.w[1] * stab_z + self.w[2] * res_z
        # reattach breakdowns
        scored = []
        for i, c in enumerate(candidates):
            scored.append({
                **c, "total_score": float(total[i]),
                "xg_score": float(xg_arr[i]),
                "stab_score": float(stab_arr[i]),
                "res_score": float(res_arr[i]),
                "xg_z": float(xg_z[i]), "stab_z": float(stab_z[i]), "res_z": float(res_z[i]),
            })
        return sorted(scored, key=lambda x: x["total_score"], reverse=True)

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    def smoke_test():
        import json
        import pandas as pd
        sys.path.insert(0, ".")
        from models.world_model.bipartite_gnn import BipartiteFormationGNN
        from models.world_model.graph_builder import build_bipartite_graph

        print("--- critics smoke test ---")

        vocab = [4231,433,442,4141,4312,4321,352,343,532,541,4411,4132,3421,451,4222]
        fvocab = {code: idx for idx, code in enumerate(vocab)}

        with open("models/world_model/positional_matchups.json") as f:
            edge_list = json.load(f)["edge_list"]
        rms_df = pd.DataFrame([
            {"attacking_role_idx": a, "defending_role_idx": d, "wr": 0.01, "xg_rate": 0.05}
            for a, d in edge_list
        ])
        sty_df = pd.DataFrame({"team_id": [1, 2], "style_cluster": [0, 2]})
        ctx = {"minute": 55, "scoreline_home": 1, "scoreline_away": 0,
               "ppda": 8.5, "home_team_id": 1, "away_team_id": 2}

        world_model = BipartiteFormationGNN()
        world_model.eval()

        # --- C_xG ---
        data = build_bipartite_graph(433, 442, ctx, rms_df, sty_df, fvocab)
        with torch.no_grad():
            wm_out = world_model(data)
        xg_critic = XGCritic()
        xg_score  = xg_critic.score(wm_out)
        assert isinstance(xg_score, float), f"xg_score not float: {type(xg_score)}"
        print(f"  C_xG(433 vs 442): {xg_score:.4f}")

        # --- C_stab ---
        stab_mlp    = StabilityMLP(n_formations=N_FORMATIONS)
        stab_critic = StabilityCritic(stab_mlp, fvocab)

        for fsrc, ftgt in [(4231, 433), (442, 532), (433, 433)]:
            s = stab_critic.score(fsrc, ftgt)
            assert 0 < s < 1, f"C_stab({fsrc}->{ftgt}) out of (0,1): {s}"
            d = structural_distance(fsrc, ftgt)
            print(f"  C_stab({fsrc}->{ftgt}): {s:.4f}  Δ_struct={d:.4f}")

        # structural ordering: same formation should have lower risk than 3->5 change
        s_same = stab_critic.score(442, 442)
        s_big  = stab_critic.score(433, 532)
        print(f"  Stability same-formation={s_same:.4f}, big-change={s_big:.4f}")

        # --- C_res lookup ---
        n_win = 40
        rng   = np.random.default_rng(42)
        windows_df = pd.DataFrame({
            "home_formation":  rng.choice(vocab, n_win),
            "away_formation":  rng.choice(vocab, n_win),
            "xg_delta":        rng.normal(0, 0.1, n_win),
            "possession_pct":  rng.uniform(0.3, 0.7, n_win),
            "ppda":            rng.uniform(5, 15, n_win),
            "shots_on_target": rng.integers(0, 4, n_win).astype(float),
            "scoreline_home":  rng.integers(0, 3, n_win),
            "scoreline_away":  rng.integers(0, 3, n_win),
            "minute_bucket":   rng.integers(3, 90, n_win),
            "home_team_id":    [1] * n_win,
            "away_team_id":    [2] * n_win,
        })

        lookup = build_counter_resistance_lookup(
            world_model, windows_df, rms_df, sty_df, fvocab
        )
        assert isinstance(lookup, dict) and len(lookup) > 0, "Empty lookup table"
        print(f"  Counter-resistance lookup: {len(lookup)} structural family pairs")

        res_critic = CounterResistanceCritic(lookup)
        for ftgt, fopp in [(4231, 442), (433, 532), (352, 541)]:
            r = res_critic.score(ftgt, fopp)
            assert isinstance(r, float), f"C_res not float: {type(r)}"
            tg = CounterResistanceCritic.STRUCTURAL_GROUPS[ftgt]
            og = CounterResistanceCritic.STRUCTURAL_GROUPS[fopp]
            print(f"  C_res({ftgt} vs {fopp}): {r:.4f}  [{tg} vs {og}]")

        # --- MultiCriticScorer ---
        scorer = MultiCriticScorer(xg_critic, stab_critic, res_critic)

        candidates = [
            {"fsrc_code": 442,  "ftgt_code": 433,  "fopp_code": 442,
             "world_model_output": wm_out, "intent": "offensive"},
            {"fsrc_code": 442,  "ftgt_code": 532,  "fopp_code": 433,
             "world_model_output": wm_out, "intent": "defensive"},
            {"fsrc_code": 442,  "ftgt_code": 4231, "fopp_code": 352,
             "world_model_output": wm_out, "intent": "pressing"},
        ]
        ranked = scorer.rank_candidates(candidates)
        assert len(ranked) == 3
        scores = [r["total_score"] for r in ranked]
        assert scores == sorted(scores, reverse=True), "Not sorted descending"
        print(f"\n  MultiCriticScorer — ranked candidates:")
        for r in ranked:
            print(f"    {r['fsrc_code']}->{r['ftgt_code']} vs {r['fopp_code']}: "
                  f"total={r['total_score']:.4f}  "
                  f"[xg={r['xg_score']:.3f} stab={r['stab_score']:.3f} res={r['res_score']:.3f}]")

        # weights sum check
        w = DEFAULT_WEIGHTS
        assert abs(sum(w) - 1.0) < 1e-6, f"Default weights don't sum to 1: {w}"
        print(f"\n  Default weights: {w}  (sum={sum(w):.1f}) ✓")

        print("--- smoke test passed ---")

    smoke_test()
