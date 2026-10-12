import json
import os
import numpy as np
import torch
from torch_geometric.data import Data
from .formation_fallback import fallback_mapping as _fallback_mapping

# Resolved against this file, not the working directory. These were plain
# relative strings, so every import worked and the first predict() from any
# other directory raised FileNotFoundError -- and it raises lazily, inside
# build_bipartite_graph, long after the import looked fine. The inference
# planner imports this package from one directory and plans from another,
# which is exactly the case that failed.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
CANON_ROLES_PATH   = os.path.join(_PKG_DIR, "canonical_roles.json")
MATCHUPS_PATH      = os.path.join(_PKG_DIR, "positional_matchups.json")

NUM_ROLES      = 11
ROLE_FEAT_DIM  = 32   # first NUM_ROLES slots are the role one-hot
STYLE_FEAT_DIM = 32   # first MAX_STYLE_CLUSTERS slots are the style one-hot
MAX_STYLE_CLUSTERS = 8   # style clustering currently selects k=2 by silhouette
NUM_NODES      = 22
NODE_FEAT_DIM  = 80   # [role_emb(32), style_emb(32), scoreline_oh(5), minute_bucket(6), ppda_bucket(4), team_flag(1)]
EDGE_FEAT_DIM  = 2    # [wr, xg_rate] for E_inter; [1.0, 0.0] for E_intra
INTRA_PLACEHOLDER = [1.0, 0.0]

# scoreline clipped to [-2, 2] -> 5 bins: -2,-1,0,1,2
SCORELINE_BINS = [-2, -1, 0, 1, 2]
# minute [0,90] -> 6 buckets of 15 min each
MINUTE_BUCKETS = 6
# ppda [0, inf] -> 4 buckets: <6, 6-9, 9-13, >=13
PPDA_BREAKS = [6.0, 9.0, 13.0]

# Set False to withhold the ppda bucket from the node features. See the note in
# _node_feature: with it on, target #3 is partly given to the model as input.
USE_PPDA_FEATURE = True

# Cross-team message passing used to run home -> away only: away nodes heard
# about the home side, home nodes never heard about the away side (the inter
# GATConv has no self-loops, so a home node got nothing from it). With this on,
# every matchup edge is added in both directions, the reverse one carrying the
# away attacker -> home defender statistics (team-lead audit, "one-way GNN").
# Default False so checkpoints trained before it load unchanged; training sets
# it from configs/world_model.yaml (bidirectional_edges) and the adapter from
# the checkpoint's own config.
BIDIRECTIONAL_INTER = False


def _load_canon(path=None):
    # read the module attribute at call time so a caller can still override it
    with open(path or CANON_ROLES_PATH) as f:
        return json.load(f)

def _load_matchups(path=None):
    with open(path or MATCHUPS_PATH) as f:
        return json.load(f)

# module-level cache so repeated calls in a training loop don't re-read disk
_CANON_CACHE    = None
_MATCHUPS_CACHE = None

def _get_canon():
    global _CANON_CACHE
    if _CANON_CACHE is None:
        _CANON_CACHE = _load_canon()
    return _CANON_CACHE

def _get_matchups():
    global _MATCHUPS_CACHE
    if _MATCHUPS_CACHE is None:
        _MATCHUPS_CACHE = _load_matchups()
    return _MATCHUPS_CACHE


def _scoreline_oh(score_diff):
    clipped = max(-2, min(2, int(round(score_diff))))
    oh = [0.0] * 5
    oh[clipped + 2] = 1.0
    return oh


def _minute_oh(minute):
    bucket = min(int(minute) // 15, MINUTE_BUCKETS - 1)
    oh = [0.0] * MINUTE_BUCKETS
    oh[bucket] = 1.0
    return oh


def _ppda_oh(ppda):
    if ppda is None or (isinstance(ppda, float) and np.isnan(ppda)):
        return [0.0, 0.0, 0.0, 1.0]   # NaN -> highest bucket (low press)
    if ppda < PPDA_BREAKS[0]:
        return [1.0, 0.0, 0.0, 0.0]
    elif ppda < PPDA_BREAKS[1]:
        return [0.0, 1.0, 0.0, 0.0]
    elif ppda < PPDA_BREAKS[2]:
        return [0.0, 0.0, 1.0, 0.0]
    else:
        return [0.0, 0.0, 0.0, 1.0]


def _node_feature(
    role_idx,        # int 0-10 canonical role index
    style_cluster,   # int 0-3
    score_diff,      # int, home - away from this team's perspective
    minute,          # float
    ppda,            # float or None
    team_flag,       # 0=home, 1=away
    role_emb_table,  # nn.Embedding or None  (None -> zeros placeholder)
    style_emb_table, # nn.Embedding or None
):
    # Role and style occupy 32 dims each of the 80-dim node feature.
    #
    # When no embedding table is passed -- which is every call made by
    # train_world_model.py -- these used to be 32 zeros apiece.  That left 64
    # of the 80 dims constant at zero and removed role identity and style
    # cluster from the graph entirely: all eleven nodes of a side became
    # identical, since the remaining dims (scoreline, minute, ppda, team flag)
    # are shared by the whole team.  A role-aware, style-conditioned world
    # model that cannot see either is not testing what it claims to test.
    #
    # The fix is a one-hot in those slots rather than a frozen random vector.
    # The GNN's first linear layer applied to a one-hot is exactly a learnable
    # embedding lookup (W @ onehot(r) is column r of W), so the model still
    # learns its own role and style representations -- it just learns them
    # through weights that receive gradients, instead of reading a table that
    # graph_builder detaches and that therefore never trains.
    if role_emb_table is not None:
        re = role_emb_table(torch.tensor(role_idx)).detach().cpu().numpy().tolist()
    else:
        re = [0.0] * ROLE_FEAT_DIM
        if role_idx is not None and 0 <= int(role_idx) < NUM_ROLES:
            re[int(role_idx)] = 1.0

    if style_emb_table is not None:
        se = style_emb_table(torch.tensor(style_cluster)).detach().cpu().numpy().tolist()
    else:
        se = [0.0] * STYLE_FEAT_DIM
        if style_cluster is not None and 0 <= int(style_cluster) < MAX_STYLE_CLUSTERS:
            se[int(style_cluster)] = 1.0

    sc_oh  = _scoreline_oh(score_diff)
    min_oh = _minute_oh(minute)
    # ppda is BOTH a node feature here and target #3 of the world model, and
    # the ppda passed in is measured over the same window as the targets. So
    # the model was being shown a 4-bucket copy of one of the answers, which
    # on this corpus correlates 0.668 with the target itself. Every other
    # feature (scoreline, minute) is measured at the window START and is a
    # legitimate predictor; this one is not.
    #
    # Switching it off keeps NODE_FEAT_DIM at 80 -- the four slots are simply
    # zero -- so a model trained either way loads into the same architecture
    # and the two runs are directly comparable.
    pd_oh  = _ppda_oh(ppda) if USE_PPDA_FEATURE else [0.0, 0.0, 0.0, 0.0]
    tf     = [float(team_flag)]

    feat = re + se + sc_oh + min_oh + pd_oh + tf
    assert len(feat) == NODE_FEAT_DIM, f"Expected {NODE_FEAT_DIM}, got {len(feat)}"
    return feat


def _formation_spec(code, canon, side):
    """Hand-specified mapping when one exists, documented fallback otherwise.

    canonical_roles.json hand-specifies 15 formations; StatsBomb records more.
    A formation with a hand entry is returned untouched, so every result
    produced before this change is reproduced bit for bit.  A formation without
    one is built from canonical_adjacency -- the fallback canonical_roles.json
    already documents ("Used as fallback in graph_builder") but that was never
    wired up, leaving build_bipartite_graph to raise on 3412, 41212 and 3511.

    We did not hand-write new entries: we tested whether the existing 15 follow
    a derivable rule and they do not (formations sharing the same first two
    lines carry different intra_edges), so extending them by rule would have
    been guesswork.  The fallback reproduces 12 of the 15 hand mappings
    exactly; the three it does not (433, 4132, 4231) are the ones whose hand
    entries are internally inconsistent, and they keep their hand entries
    regardless.  formation_fallback.fallback_report() lists every fallback use
    for the write-up.
    """
    spec = canon["formation_mappings"].get(str(code))
    if spec is not None:
        return spec
    try:
        return _fallback_mapping(code, canon)
    except Exception as exc:
        raise ValueError(f"Unknown {side} formation: {code} ({exc})") from exc


def build_bipartite_graph(
    home_formation_code,
    away_formation_code,
    game_context_dict,
    role_matchup_stats_df,
    style_clusters_df,
    formation_vocab,          # {code: idx} dict
    role_emb_table=None,      # nn.Embedding(25, 32) or None
    style_emb_table=None,     # nn.Embedding(4, 32) or None
):
    canon    = _get_canon()
    matchups = _get_matchups()

    home_code = int(home_formation_code)
    away_code = int(away_formation_code)

    home_fm = _formation_spec(home_code, canon, "home")
    away_fm = _formation_spec(away_code, canon, "away")

    home_roles = home_fm["node_assignments"]   # list of 11 canonical idx
    away_roles = away_fm["node_assignments"]
    home_intra = home_fm["intra_edges"]        # list of [i,j] in local 0-10 coords
    away_intra = away_fm["intra_edges"]

    # ---- context ----
    minute       = float(game_context_dict.get("minute", 45))
    scoreline_h  = int(game_context_dict.get("scoreline_home", 0))
    scoreline_a  = int(game_context_dict.get("scoreline_away", 0))
    score_diff   = scoreline_h - scoreline_a
    ppda         = game_context_dict.get("ppda", None)

    home_team_id = game_context_dict.get("home_team_id", None)
    away_team_id = game_context_dict.get("away_team_id", None)

    # style clusters
    def _style(team_id):
        if style_clusters_df is not None and team_id is not None:
            row = style_clusters_df[style_clusters_df["team_id"] == team_id]
            if not row.empty:
                return int(row["style_cluster"].iloc[0])
        return 0

    home_style = _style(home_team_id)
    away_style = _style(away_team_id)

    # team_flag marks the VENUE of each block, not just its slot. A swapped
    # (augmented) copy puts the away team in slot 0; it must still be flagged
    # as the away team, or home advantage gets attributed to the slot.
    home_slot_is_home = bool(game_context_dict.get("home_slot_is_home", True))
    flag_slot0 = 0 if home_slot_is_home else 1
    flag_slot1 = 1 - flag_slot0

    # ---- node features ----
    node_feats = []
    # nodes 0-10: home team
    for pos_i in range(NUM_ROLES):
        role_idx = home_roles[pos_i]
        feat = _node_feature(
            role_idx, home_style, score_diff, minute, ppda,
            team_flag=flag_slot0, role_emb_table=role_emb_table, style_emb_table=style_emb_table
        )
        node_feats.append(feat)
    # nodes 11-21: away team (score_diff flipped; ppda from away perspective)
    for pos_i in range(NUM_ROLES):
        role_idx = away_roles[pos_i]
        feat = _node_feature(
            role_idx, away_style, -score_diff, minute, ppda,
            team_flag=flag_slot1, role_emb_table=role_emb_table, style_emb_table=style_emb_table
        )
        node_feats.append(feat)

    x = torch.tensor(node_feats, dtype=torch.float)   # (22, 80)

    # ---- edge index and features ----
    src_list, dst_list, efeat_list = [], [], []

    # E_intra home (nodes 0-10, undirected -> add both directions)
    for (i, j) in home_intra:
        for s, d in [(i, j), (j, i)]:
            src_list.append(s)
            dst_list.append(d)
            efeat_list.append(INTRA_PLACEHOLDER)

    # E_intra away (nodes 11-21, offset by 11)
    for (i, j) in away_intra:
        for s, d in [(i + NUM_ROLES, j + NUM_ROLES), (j + NUM_ROLES, i + NUM_ROLES)]:
            src_list.append(s)
            dst_list.append(d)
            efeat_list.append(INTRA_PLACEHOLDER)

    # E_inter: home attacking role -> away defending role
    # build lookup from role_matchup_stats_df
    if role_matchup_stats_df is not None:
        rms = role_matchup_stats_df.set_index(
            ["attacking_role_idx", "defending_role_idx"]
        )[["wr", "xg_rate"]].to_dict("index")
    else:
        rms = {}

    inter_edge_list = matchups["edge_list"]   # [[atk_canonical, def_canonical], ...]
    for (atk_canonical, def_canonical) in inter_edge_list:
        # find which home position nodes carry atk_canonical
        # find which away position nodes carry def_canonical
        home_pos_nodes = [i for i, r in enumerate(home_roles) if r == atk_canonical]
        away_pos_nodes = [i for i, r in enumerate(away_roles)  if r == def_canonical]

        if not home_pos_nodes or not away_pos_nodes:
            continue   # role not present in this formation pair

        stats = rms.get((atk_canonical, def_canonical), {"wr": 0.0, "xg_rate": 0.0})
        ef = [float(stats["wr"]), float(stats["xg_rate"])]

        for hp in home_pos_nodes:
            for ap in away_pos_nodes:
                src_list.append(hp)
                dst_list.append(ap + NUM_ROLES)
                efeat_list.append(ef)

    if BIDIRECTIONAL_INTER:
        # the same matchups with the sides reversed: away attacker -> home
        # defender, read from the same (attacker role, defender role) table,
        # which compute_role_matchup_stats.py now fills from both directions
        for (atk_canonical, def_canonical) in inter_edge_list:
            away_atk_nodes = [i for i, r in enumerate(away_roles) if r == atk_canonical]
            home_def_nodes = [i for i, r in enumerate(home_roles) if r == def_canonical]
            if not away_atk_nodes or not home_def_nodes:
                continue
            stats = rms.get((atk_canonical, def_canonical), {"wr": 0.0, "xg_rate": 0.0})
            ef = [float(stats["wr"]), float(stats["xg_rate"])]
            for ap in away_atk_nodes:
                for hp in home_def_nodes:
                    src_list.append(ap + NUM_ROLES)
                    dst_list.append(hp)
                    efeat_list.append(ef)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor(efeat_list, dtype=torch.float)

    # ---- formation indices (used by GNN to look up embeddings) ----
    home_fmt_idx = torch.tensor(formation_vocab.get(home_code, 0), dtype=torch.long)
    away_fmt_idx = torch.tensor(formation_vocab.get(away_code, 0), dtype=torch.long)

    # ---- inter-edge mask (for attention extraction in bipartite_gnn) ----
    n_home_intra = len(home_intra) * 2
    n_away_intra = len(away_intra) * 2
    n_total      = len(src_list)
    inter_mask = torch.zeros(n_total, dtype=torch.bool)
    inter_mask[n_home_intra + n_away_intra:] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        home_formation=home_fmt_idx,
        away_formation=away_fmt_idx,
        inter_edge_mask=inter_mask,
        num_nodes=NUM_NODES,
    )
    return data


if __name__ == "__main__":
    def smoke_test():
        import pandas as pd

        print("--- graph_builder smoke test ---")

        vocab = [4231,433,442,4141,4312,4321,352,343,532,541,4411,4132,3421,451,4222]
        formation_vocab = {code: idx for idx, code in enumerate(vocab)}

        # minimal role_matchup_stats_df
        with open(MATCHUPS_PATH) as f:
            edge_list = json.load(f)["edge_list"]
        rms_df = pd.DataFrame([
            {"attacking_role_idx": a, "defending_role_idx": d, "wr": 0.01, "xg_rate": 0.05}
            for a, d in edge_list
        ])

        # minimal style_clusters_df
        sty_df = pd.DataFrame({
            "team_id":      [1, 2],
            "style_cluster": [0, 2],
        })

        ctx = {
            "minute": 55, "scoreline_home": 1, "scoreline_away": 0,
            "ppda": 8.5, "home_team_id": 1, "away_team_id": 2,
        }

        # primary test: 433 vs 442
        data = build_bipartite_graph(433, 442, ctx, rms_df, sty_df, formation_vocab)

        assert data.num_nodes == 22, f"Expected 22 nodes, got {data.num_nodes}"
        assert data.x.shape == (22, NODE_FEAT_DIM), f"x shape {data.x.shape}"
        assert data.edge_index.shape[0] == 2
        assert data.edge_attr.shape[1] == EDGE_FEAT_DIM
        assert data.edge_attr.shape[0] == data.edge_index.shape[1]
        assert data.inter_edge_mask.shape[0] == data.edge_index.shape[1]
        assert data.inter_edge_mask.any(),  "No inter edges found"
        assert (~data.inter_edge_mask).any(),"No intra edges found"
        assert not torch.isnan(data.x).any(), "NaN in node features"
        assert not torch.isnan(data.edge_attr).any(), "NaN in edge features"

        n_inter = data.inter_edge_mask.sum().item()
        n_intra = (~data.inter_edge_mask).sum().item()
        print(f"  433 vs 442: 22 nodes OK | {n_intra} intra edges | {n_inter} inter edges")

        # check all 15x15 formation pairs build without error
        errors = []
        for hf in vocab:
            for af in vocab:
                try:
                    d = build_bipartite_graph(hf, af, ctx, rms_df, sty_df, formation_vocab)
                    assert d.num_nodes == 22
                    assert not torch.isnan(d.x).any()
                except Exception as e:
                    errors.append(f"{hf}v{af}: {e}")

        assert not errors, f"Failures in all-pairs test:\n" + "\n".join(errors)
        print(f"  All 15×15 = {15*15} formation pairs built successfully")

        # team_flag check
        assert data.x[:11, -1].sum() == 0.0, "Home nodes should have team_flag=0"
        assert data.x[11:, -1].sum() == 11.0, "Away nodes should have team_flag=1"
        print("  team_flag encoding correct")

        # inter edges go from home (0-10) to away (11-21)
        inter_src = data.edge_index[0][data.inter_edge_mask]
        inter_dst = data.edge_index[1][data.inter_edge_mask]
        assert (inter_src < 11).all(),  "Inter edge source should be home node (< 11)"
        assert (inter_dst >= 11).all(), "Inter edge dest should be away node (>= 11)"
        print("  Inter-edge directionality correct")

        print("--- smoke test passed ---")

    smoke_test()
