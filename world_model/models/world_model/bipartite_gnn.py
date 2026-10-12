import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

NUM_ROLES     = 11
NUM_NODES     = 22
NODE_FEAT_DIM = 80
EDGE_FEAT_DIM = 2
NODE_DIM      = 128
READOUT_DIM   = 256   # 2 * NODE_DIM
OUTPUT_DIM    = 4     # [xg_delta, possession_pct, ppda, shots_on_target]
GNN_LAYERS    = 3
GAT_HEADS     = 4


class MeanPoolMLP(nn.Module):
    """The same inputs and the same readout, with the graph removed.

    This is the ablation the bipartite architecture has to answer. The GNN's
    case is that message passing between the two sides' role nodes captures
    something about the matchup. Everything else it has -- the 80-dim node
    features, the asymmetric home/away mean-pool readout, the output head --
    is shared with a model that does no message passing at all.

    So: mean-pool the eleven home nodes and the eleven away nodes exactly as
    `BipartiteFormationGNN` does at the end, concatenate, and put an MLP on
    top. Edges are ignored entirely. If this matches the GNN, the graph is not
    contributing and the result should say so; the decomposition in
    diagnose_world_model.py points that way, and this measures it directly.

    The constructor takes the same arguments so the two are drop-in
    interchangeable, and the hidden width is set so the parameter counts are
    comparable rather than the MLP being handicapped.
    """

    def __init__(self, node_feat_dim=NODE_FEAT_DIM, node_dim=NODE_DIM,
                 gat_heads=GAT_HEADS, gnn_layers=GNN_LAYERS,
                 output_dim=OUTPUT_DIM, edge_feat_dim=EDGE_FEAT_DIM,
                 n_formations=0):
        super().__init__()
        self.output_dim = output_dim
        self.n_formations = int(n_formations)
        # ctx_dim = one-hot home formation + one-hot away formation + 5 scalars.
        # 0 means "old checkpoint, no ctx features"; the forward path stays
        # identical to the original so old checkpoints still load.
        self.ctx_dim = (2 * self.n_formations + 5) if self.n_formations > 0 else 0
        self.input_proj = nn.Linear(node_feat_dim, node_dim)
        self.act = nn.GELU()
        hidden = node_dim * 4
        layers, d_in = [], node_dim * 2
        for _ in range(max(gnn_layers, 1)):
            layers += [nn.Linear(d_in, hidden), nn.GELU()]
            d_in = hidden
        self.trunk = nn.Sequential(*layers)
        self.output_head = nn.Linear(d_in + self.ctx_dim, output_dim)

    def forward(self, data):
        single = getattr(data, "batch", None) is None
        x = self.act(self.input_proj(data.x))
        n_graphs = x.size(0) // NUM_NODES
        idx = torch.arange(x.size(0), device=x.device)
        is_home = (idx % NUM_NODES) < NUM_ROLES

        g_home = x[idx[is_home]].view(n_graphs, NUM_ROLES, -1).mean(dim=1)
        g_away = x[idx[~is_home]].view(n_graphs, NUM_ROLES, -1).mean(dim=1)
        g = torch.cat([g_home, g_away], dim=-1)

        trunk_out = self.trunk(g)
        if self.ctx_dim > 0:
            ctx = data.ctx_features
            if single and ctx.dim() == 1:
                ctx = ctx.unsqueeze(0)
            trunk_out = torch.cat([trunk_out, ctx], dim=-1)
        out = self.output_head(trunk_out)
        return out.squeeze(0) if single and n_graphs == 1 else out

    def forward_batch(self, data_list):
        from torch_geometric.data import Batch
        return self.forward(Batch.from_data_list(data_list))


class BipartiteFormationGNN(nn.Module):
    def __init__(
        self,
        node_feat_dim=NODE_FEAT_DIM,
        node_dim=NODE_DIM,
        gat_heads=GAT_HEADS,
        gnn_layers=GNN_LAYERS,
        output_dim=OUTPUT_DIM,
        edge_feat_dim=EDGE_FEAT_DIM,
        n_formations=0,
    ):
        super().__init__()
        self.node_dim    = node_dim
        self.gnn_layers  = gnn_layers
        self.output_dim  = output_dim
        self.n_formations = int(n_formations)
        self.ctx_dim = (2 * self.n_formations + 5) if self.n_formations > 0 else 0

        # project raw node features -> node_dim
        self.input_proj = nn.Linear(node_feat_dim, node_dim)

        # per-layer GATConv modules
        # intra_home[l]: home subgraph (nodes 0-10 -> 0-10)
        # intra_away[l]: away subgraph (nodes 11-21 -> 11-21)
        # inter[l]:      home->away cross edges (nodes 0-10 -> 11-21)
        self.intra_home = nn.ModuleList()
        self.intra_away = nn.ModuleList()
        self.inter_conv = nn.ModuleList()

        for _ in range(gnn_layers):
            # concat=False -> output is out_channels (heads are averaged, not concatenated)
            # set out_channels=node_dim so every layer emits (N, node_dim)
            self.intra_home.append(GATConv(
                node_dim, node_dim,
                heads=gat_heads, concat=False,
                edge_dim=edge_feat_dim, add_self_loops=True,
            ))
            self.intra_away.append(GATConv(
                node_dim, node_dim,
                heads=gat_heads, concat=False,
                edge_dim=edge_feat_dim, add_self_loops=True,
            ))
            self.inter_conv.append(GATConv(
                node_dim, node_dim,
                heads=gat_heads, concat=False,
                edge_dim=edge_feat_dim, add_self_loops=False,
            ))

        self.act = nn.GELU()
        self.output_head = nn.Linear(READOUT_DIM + self.ctx_dim, output_dim)

    def _split_edges_batched(self, data, n_graphs):
        """Edge sets for a Batch of `n_graphs` graphs, in compact per-team coords.

        The single-graph version tests `intra_ei[0] < NUM_ROLES`, which is only
        true for the first graph once several are batched together: node ids in
        a PyG Batch are offset by 22 per graph. Here the team is read from the
        node's position WITHIN its graph, and home/away node ids are remapped
        into the compact (n_graphs * 11) spaces the intra convolutions run on.
        """
        inter_mask = data.inter_edge_mask
        intra_mask = ~inter_mask

        intra_ei = data.edge_index[:, intra_mask]
        intra_ea = data.edge_attr[intra_mask]
        inter_ei = data.edge_index[:, inter_mask]
        inter_ea = data.edge_attr[inter_mask]

        local = intra_ei % NUM_NODES
        graph = intra_ei // NUM_NODES

        # intra edges never cross teams, so the source decides the team
        home_mask = local[0] < NUM_ROLES
        away_mask = ~home_mask

        home_ei = graph[:, home_mask] * NUM_ROLES + local[:, home_mask]
        home_ea = intra_ea[home_mask]
        away_ei = graph[:, away_mask] * NUM_ROLES + (local[:, away_mask] - NUM_ROLES)
        away_ea = intra_ea[away_mask]

        return home_ei, home_ea, away_ei, away_ea, inter_ei, inter_ea

    def _split_edges(self, data):
        inter_mask = data.inter_edge_mask          # (E,) bool
        intra_mask = ~inter_mask

        intra_ei   = data.edge_index[:, intra_mask]
        intra_ea   = data.edge_attr[intra_mask]
        inter_ei   = data.edge_index[:, inter_mask]
        inter_ea   = data.edge_attr[inter_mask]

        # home intra: both src and dst < NUM_ROLES
        home_mask  = intra_ei[0] < NUM_ROLES
        home_ei    = intra_ei[:, home_mask]
        home_ea    = intra_ea[home_mask]

        # away intra: src and dst >= NUM_ROLES -> shift to local 0-10 coords
        away_mask  = intra_ei[0] >= NUM_ROLES
        away_ei    = intra_ei[:, away_mask] - NUM_ROLES
        away_ea    = intra_ea[away_mask]

        return home_ei, home_ea, away_ei, away_ea, inter_ei, inter_ea

    def forward(self, data):
        """One graph or a whole Batch, same arithmetic.

        This used to handle exactly one 22-node graph: it sliced `x[:11]` and
        `x[11:]` directly, so a batch could not be passed in. `forward_batch`
        was a Python loop over `forward`, which is not batching at all -- a
        batch of 32 ran 32 separate forward passes and held 32 separate
        autograd graphs alive before a single backward. That is where the
        training time and most of the memory went.

        Node ids in a PyG Batch of B graphs run 0..22B-1, with graph b holding
        22b..22b+21 and its home players first. Everything below works off that
        layout, so the result is identical to looping -- see
        `test_batched_forward_matches_loop`, which asserts it to 1e-5.
        """
        single = getattr(data, "batch", None) is None
        x = self.act(self.input_proj(data.x))                  # (22B, node_dim)
        n_graphs = x.size(0) // NUM_NODES

        home_ei, home_ea, away_ei, away_ea, inter_ei, inter_ea = (
            self._split_edges_batched(data, n_graphs)
        )

        idx = torch.arange(x.size(0), device=x.device)
        is_home = (idx % NUM_NODES) < NUM_ROLES
        home_sel = idx[is_home]        # ordered, so it maps onto b*11 + local
        away_sel = idx[~is_home]

        for l in range(self.gnn_layers):
            h_home = x[home_sel]       # (11B, node_dim)
            h_away = x[away_sel]

            # intra-team message passing
            h_home_new = self.act(self.intra_home[l](h_home, home_ei, home_ea))
            h_away_new = self.act(self.intra_away[l](h_away, away_ei, away_ea))

            # inter-team: home node i -> away node j. `x` is already the full
            # node matrix in graph order, which is what cat([home, away]) built
            # in the single-graph case.
            h_inter = self.act(self.inter_conv[l](x, inter_ei, inter_ea))

            x_new = torch.zeros_like(x)
            x_new = x_new.index_copy(0, home_sel, h_home_new + h_inter[home_sel])
            x_new = x_new.index_copy(0, away_sel, h_away_new + h_inter[away_sel])

            # B11: residual + LayerNorm after each GAT block. Without either,
            # the three-layer stack adds each layer's output directly onto the
            # previous activations with no identity path and no normalisation;
            # GAT attention weights are not guaranteed to keep the scale of
            # h stable across layers, and heads=4 with concat=False averages
            # four attention outputs per layer, so variance decays with depth.
            # The swap-residual of 7.55pp in the writeup shows it still trains,
            # but a residual + LN is the cheap standard fix and is worth
            # measuring against the current model.
            #
            # layer_norm with functional form (no learnable affine) so the
            # parameter count is unchanged and old checkpoints that lack
            # ln.weight / ln.bias still load. Add nn.LayerNorm modules and
            # affine=True if the ablation below shows it helps.
            x_new = x_new + x
            x_new = F.layer_norm(x_new, x_new.shape[-1:])
            x = x_new

        # asymmetric readout: mean-pool each team separately, per graph
        g_home = x[home_sel].view(n_graphs, NUM_ROLES, -1).mean(dim=1)
        g_away = x[away_sel].view(n_graphs, NUM_ROLES, -1).mean(dim=1)
        g = torch.cat([g_home, g_away], dim=-1)                # (B, READOUT_DIM)

        if self.ctx_dim > 0:
            ctx = data.ctx_features                            # (1, C) or (B, C)
            if ctx.dim() == 1:
                ctx = ctx.unsqueeze(0)
            if ctx.shape[0] != g.shape[0]:
                # single-graph input: one ctx row for one graph
                ctx = ctx.expand(g.shape[0], -1)
            g = torch.cat([g, ctx], dim=-1)

        out = self.output_head(g)                              # (B, output_dim)
        return out.squeeze(0) if single and n_graphs == 1 else out

    def forward_batch(self, data_list):
        """Kept for callers that hold a list; now a real batched forward."""
        from torch_geometric.data import Batch
        return self.forward(Batch.from_data_list(data_list))

    def get_inter_attention_weights(self, data):
        x        = self.act(self.input_proj(data.x))
        home_ei, home_ea, away_ei, away_ea, inter_ei, inter_ea = self._split_edges(data)

        attn_weights = {}   # (src_node, dst_node) -> alpha from final layer

        for l in range(self.gnn_layers):
            h_home = x[:NUM_ROLES]
            h_away = x[NUM_ROLES:]

            h_home = self.act(self.intra_home[l](h_home, home_ei, home_ea))
            h_away = self.act(self.intra_away[l](h_away, away_ei, away_ea))

            h_full = torch.cat([h_home, h_away], dim=0)

            is_last = (l == self.gnn_layers - 1)
            if is_last:
                _, (ei_out, alpha_out) = self.inter_conv[l](
                    h_full, inter_ei, inter_ea, return_attention_weights=True
                )
                # alpha_out: (E_inter, heads) -> mean over heads
                alpha_mean = alpha_out.mean(dim=1).detach().cpu()
                src_nodes  = ei_out[0].cpu()
                dst_nodes  = ei_out[1].cpu()
                for k in range(src_nodes.shape[0]):
                    key = (int(src_nodes[k]), int(dst_nodes[k]))
                    attn_weights[key] = float(alpha_mean[k])
                h_inter = self.act(self.inter_conv[l](h_full, inter_ei, inter_ea))
            else:
                h_inter = self.act(self.inter_conv[l](h_full, inter_ei, inter_ea))

            h_home_merged = h_home + h_inter[:NUM_ROLES]
            h_away_merged = h_away + h_inter[NUM_ROLES:]
            x = torch.cat([h_home_merged, h_away_merged], dim=0)

        return attn_weights


if __name__ == "__main__":
    def smoke_test():
        import json
        import pandas as pd
        import sys

        try:
            from .graph_builder import build_bipartite_graph
        except ImportError:
            sys.path.insert(0, ".")
            from models.world_model.graph_builder import build_bipartite_graph

        print("--- bipartite_gnn smoke test ---")

        vocab  = [4231,433,442,4141,4312,4321,352,343,532,541,4411,4132,3421,451,4222]
        fvocab = {code: idx for idx, code in enumerate(vocab)}

        with open("models/world_model/positional_matchups.json") as f:
            edge_list = json.load(f)["edge_list"]
        rms_df = pd.DataFrame([
            {"attacking_role_idx": a, "defending_role_idx": d, "wr": 0.01, "xg_rate": 0.05}
            for a, d in edge_list
        ])
        sty_df = pd.DataFrame({"team_id": [1, 2], "style_cluster": [0, 2]})
        ctx    = {"minute": 55, "scoreline_home": 1, "scoreline_away": 0,
                  "ppda": 8.5, "home_team_id": 1, "away_team_id": 2}

        model = BipartiteFormationGNN()
        model.eval()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {total_params:,}")

        # forward pass
        data = build_bipartite_graph(433, 442, ctx, rms_df, sty_df, fvocab)
        with torch.no_grad():
            out = model(data)
        assert out.shape == (OUTPUT_DIM,), f"Expected ({OUTPUT_DIM},), got {out.shape}"
        assert not torch.isnan(out).any(), f"NaN in output: {out}"
        print(f"  433 vs 442 forward pass: output={out.detach().numpy().round(4)}")

        # all 15x15 pairs
        errors = []
        for hf in vocab:
            for af in vocab:
                d = build_bipartite_graph(hf, af, ctx, rms_df, sty_df, fvocab)
                with torch.no_grad():
                    o = model(d)
                if o.shape != (OUTPUT_DIM,) or torch.isnan(o).any():
                    errors.append(f"{hf}v{af}: shape={o.shape} nan={torch.isnan(o).any()}")
        assert not errors, "Failures:\n" + "\n".join(errors)
        print(f"  All 15×15 pairs: no NaN, correct output shape")

        # attention weights
        data_attn = build_bipartite_graph(433, 442, ctx, rms_df, sty_df, fvocab)
        with torch.no_grad():
            attn = model.get_inter_attention_weights(data_attn)
        assert isinstance(attn, dict) and len(attn) > 0, "Empty attention dict"
        assert all(isinstance(v, float) for v in attn.values()), "Non-float attention"
        sample_key = next(iter(attn))
        src, dst = sample_key
        assert 0 <= src < NUM_ROLES, f"Attention src {src} out of home range"
        assert NUM_ROLES <= dst < NUM_NODES, f"Attention dst {dst} out of away range"
        print(f"  Attention weights: {len(attn)} inter edges, sample {sample_key}: {attn[sample_key]:.4f}")

        print("--- smoke test passed ---")

    smoke_test()
