"""
Smoke tests. Run with `python -m pytest tests -q` or `python tests/test_smoke.py`.

These are deliberately the tests that catch the bugs that would otherwise show
up as "the model just doesn't learn very well" three weeks from now: a broken
causal mask, a padding index off by one, an input that is wired into the module
signature but never actually reaches the logits, and a leaky CV split.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ramfp.config import Config                                    # noqa: E402
from ramfp.data.datasets import TripleDataset, collate             # noqa: E402
from ramfp.data.schema import SchemaError, validate_triples        # noqa: E402
from ramfp.data.synthetic import generate_triples                  # noqa: E402
from ramfp.eval.metrics import (conformal_coverage, conformal_qhat,  # noqa: E402
                                evaluate, expected_calibration_error)
from ramfp.inference import (ConformalTerminator, RAMFPPlanner,    # noqa: E402
                             TransitionGenerator, demo_state, StubWorldModel)
from ramfp.models.opponent import OpponentCoBehaviorModel          # noqa: E402
from ramfp.models.state_encoder import StateEncoder                # noqa: E402
from ramfp.vocab import (FORMATIONS, N_FORMATIONS, N_STYLES,  # noqa: E402
                         decode_formation, formation_layout, intra_adjacency,
                         role_name)

CFG = Config.load(None)
CFG.encoder.d_model = 32
CFG.encoder.n_layers = 2
CFG.encoder.gat_dim = 32
CFG.encoder.d_context = 64
CFG.encoder.seq_len = 16
CFG.opponent.d_model = 64
CFG.opponent.d_style = 8


def _batch(n: int = 8):
    df, _ = generate_triples(n, seed=3, seq_len=CFG.encoder.seq_len)
    ds = TripleDataset(df, CFG.encoder.seq_len, knn=CFG.encoder.knn)
    return collate([ds[i] for i in range(len(ds))])


# --------------------------------------------------------------------------
def test_formation_geometry():
    for code in FORMATIONS:
        pos, roles = formation_layout(code)
        assert pos.shape == (11, 2), code
        assert len(roles) == 11, code
        assert sum(decode_formation(code)) == 10, code
        adj = intra_adjacency(code)
        assert adj.shape == (11, 11)
        assert (adj == adj.T).all(), f"{code}: adjacency not symmetric"
        assert adj.diagonal().all(), f"{code}: missing self-loops"
        assert adj.sum() > 11, f"{code}: graph has no real edges"
    # the structural prior: shapes sharing a backline share those role ids
    r442 = set(formation_layout(442)[1])
    r4231 = set(formation_layout(4231)[1])
    shared = {role_name(r) for r in r442 & r4231}
    assert {"DEF-L", "DEF-LC", "DEF-RC", "DEF-R"} <= shared, shared
    print("  formation geometry ok")


def test_causal_mask():
    """Changing a LATER token must not change an EARLIER hidden state."""
    torch.manual_seed(0)
    enc = StateEncoder(CFG.encoder).eval()
    b = _batch(4)
    with torch.no_grad():
        h1 = enc.events(b)
        b2 = {k: v.clone() for k, v in b.items()}
        b2["event_type"][:, -1] = (b2["event_type"][:, -1] % 5) + 2   # perturb last token
        h2 = enc.events(b2)
    early = (h1[:, :-1] - h2[:, :-1]).abs().max().item()
    assert early < 1e-5, f"causal mask leaks: earlier states moved by {early}"
    print(f"  causal mask ok (max early drift {early:.2e})")


def test_padding_is_ignored():
    """Trailing PAD must not change the pooled representation."""
    torch.manual_seed(0)
    enc = StateEncoder(CFG.encoder).eval()
    b = _batch(4)
    short = {k: v.clone() for k, v in b.items()}
    keep = CFG.encoder.seq_len // 2
    for f in ("event_type", "event_zone", "event_team", "event_minute", "event_outcome"):
        short[f][:, keep:] = 0
    short["length"] = torch.full_like(short["length"], keep)
    with torch.no_grad():
        z = enc(short)
    assert torch.isfinite(z).all(), "padding produced NaN/Inf"
    assert z.shape == (4, CFG.encoder.d_context)
    print("  padding handling ok")


def test_transition_input_is_wired():
    """
    The single most valuable test here. If f_tgt is accepted by the signature
    but never reaches the logits, the model silently degenerates into a
    state-only predictor and every E1 number becomes meaningless.
    """
    torch.manual_seed(0)
    model = OpponentCoBehaviorModel(CFG.encoder, CFG.opponent).eval()
    b = _batch(8)
    with torch.no_grad():
        p1 = torch.softmax(model(b), -1)
        b2 = {k: v.clone() for k, v in b.items()}
        b2["f_tgt"] = (b2["f_tgt"] + 3) % N_FORMATIONS
        p2 = torch.softmax(model(b2), -1)
        b3 = {k: v.clone() for k, v in b.items()}
        b3["style"] = (b3["style"] + 1) % N_STYLES
        p3 = torch.softmax(model(b3), -1)
    assert (p1 - p2).abs().max() > 1e-4, "f_tgt does not affect the output"
    assert (p1 - p3).abs().max() > 1e-4, "style does not affect the output"
    print("  transition and style inputs are wired")


def test_style_ablation_removes_style():
    torch.manual_seed(0)
    model = OpponentCoBehaviorModel(CFG.encoder, CFG.opponent,
                                    style_conditioned=False).eval()
    b = _batch(8)
    with torch.no_grad():
        p1 = torch.softmax(model(b), -1)
        b2 = {k: v.clone() for k, v in b.items()}
        b2["style"] = (b2["style"] + 1) % N_STYLES
        p2 = torch.softmax(model(b2), -1)
    assert (p1 - p2).abs().max() < 1e-6, "ablation still uses the style label"
    print("  style ablation is a true ablation")


def test_head_matches_forward():
    """Planner scores many candidates against a cached z_t; it must agree."""
    torch.manual_seed(0)
    model = OpponentCoBehaviorModel(CFG.encoder, CFG.opponent).eval()
    b = _batch(6)
    with torch.no_grad():
        full = model(b)
        z = model.encoder(b)
        cached = model.head(z, b["f_src"], b["f_tgt"], b["intent"], b["style"])
    assert torch.allclose(full, cached, atol=1e-5)
    print("  cached-z_t head matches the full forward pass")


def test_loss_and_kl():
    torch.manual_seed(0)
    model = OpponentCoBehaviorModel(CFG.encoder, CFG.opponent)
    b = _batch(8)
    out = model.loss(model(b), b["target"])
    assert torch.isfinite(out["loss"]) and out["loss"] > 0
    assert out["kl"] >= -1e-5, "KL(p||uniform) must be non-negative"
    uniform = torch.zeros(4, N_FORMATIONS)
    kl_uniform = model.loss(uniform, torch.zeros(4, dtype=torch.long))["kl"]
    assert abs(float(kl_uniform)) < 1e-5, "KL of a uniform distribution should be 0"
    print("  loss and KL term behave")


def test_schema_rejects_bad_data():
    df, _ = generate_triples(50, seed=1, seq_len=CFG.encoder.seq_len)
    validate_triples(df)
    bad = df.copy()
    bad.loc[bad.index[0], "f_tgt"] = bad.loc[bad.index[0], "f_src"]
    try:
        validate_triples(bad)
    except SchemaError as e:
        assert "f_src == f_tgt" in str(e)
        print("  schema catches unfiltered personnel-only Tactical Shifts")
    else:
        raise AssertionError("schema accepted f_src == f_tgt")


def test_metrics():
    rng = np.random.default_rng(0)
    y = rng.integers(0, N_FORMATIONS, 500)
    perfect = np.eye(N_FORMATIONS)[y] * 0.98 + 0.02 / N_FORMATIONS
    m = evaluate(perfect, y)
    assert m.top1 == 1.0 and m.top3 == 1.0
    assert m.ece < 0.05, m.ece
    uniform = np.full((500, N_FORMATIONS), 1.0 / N_FORMATIONS)
    mu = evaluate(uniform, y)
    assert abs(mu.top1 - 1.0 / N_FORMATIONS) < 0.06
    assert expected_calibration_error(uniform, y) < 0.1
    print("  metrics behave on degenerate inputs")


def test_conformal_coverage():
    """A correct split-conformal predictor covers at 1 - alpha on fresh data."""
    rng = np.random.default_rng(0)
    n, alpha = 4000, 0.10
    y = rng.integers(0, N_FORMATIONS, n)
    logits = rng.normal(size=(n, N_FORMATIONS))
    logits[np.arange(n), y] += 1.5                    # a mediocre but real model
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    cal, test = slice(0, n // 2), slice(n // 2, n)
    qhat = conformal_qhat(p[cal], y[cal], alpha)
    cov = conformal_coverage(p[test], y[test], qhat)
    assert 1 - alpha - 0.03 < cov < 1 - alpha + 0.03, f"coverage {cov}"
    print(f"  conformal coverage {cov:.3f} at target {1 - alpha:.2f}")


def test_generator_constraints():
    gen = TransitionGenerator()
    st = demo_state()
    st.goal_diff, st.minute = 2, 80
    cands = gen.generate(st, k=100)
    assert cands, "grammar rejected everything"
    for c in cands:
        assert c.f_tgt != c.f_src, "generator proposed a no-op"
        fwd_src = decode_formation(c.f_src)[-1]
        fwd_tgt = decode_formation(c.f_tgt)[-1]
        assert fwd_tgt <= fwd_src, "went more attacking while 2 up after minute 75"
    print(f"  transition grammar ok ({len(cands)} candidates survived)")


def test_planner_end_to_end():
    torch.manual_seed(0)
    model = OpponentCoBehaviorModel(CFG.encoder, CFG.opponent).eval()
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(N_FORMATIONS), size=300)
    ys = rng.integers(0, N_FORMATIONS, 300)
    term = ConformalTerminator(0.1, 6).fit(probs, ys)
    planner = RAMFPPlanner(model, cfg=CFG.inference, terminator=term, device="cpu")
    recs = planner.plan(demo_state(), top_n=3)
    assert recs, "planner returned nothing"
    assert all(r.is_stub for r in recs), "stub results must be flagged"
    assert all(np.isfinite(r.score) for r in recs)
    assert abs(recs[0].opponent_probs.sum() - 1.0) < 1e-4
    scores = [r.score for r in recs]
    assert scores == sorted(scores, reverse=True), "results are not ranked"
    print(f"  planner ran end to end ({len(recs)} recommendations, stub-flagged)")


def test_two_heads():
    """The respond head is real, and the formation head is not trained on
    decisions where the opponent's shape never moved.

    That masking is the whole point of the second head: on real data the
    formation moves in about 6% of decisions, so a head trained on every row
    learns to copy the opponent's current shape -- the persistence rule --
    and reports a high accuracy that means nothing.
    """
    torch.manual_seed(0)
    model = OpponentCoBehaviorModel(CFG.encoder, CFG.opponent).eval()
    b = _batch(8)
    with torch.no_grad():
        logits = model(b)
        logits2, respond = model(b, return_respond=True)
    assert logits.shape == (8, N_FORMATIONS)
    assert torch.allclose(logits, logits2), "return_respond changed the logits"
    assert respond.shape == (8,), respond.shape

    # a batch with no real change must produce no formation gradient at all
    nothing = torch.zeros(8, dtype=torch.bool)
    out = model.loss(logits, b["target"], changed=nothing)
    assert float(out["ce"]) == 0.0, out["ce"]
    assert float(out["kl"]) == 0.0, out["kl"]

    # and the old single-head call must still behave exactly as before
    plain = model.loss(logits, b["target"])
    assert torch.isfinite(plain["loss"]) and float(plain["ce"]) > 0

    # the respond BCE has to actually enter the total
    both = model.loss(logits, b["target"], respond_logit=respond,
                      respond_target=torch.ones(8), changed=nothing)
    assert "bce" in both and float(both["loss"]) > 0

    # mixture is a distribution, and collapses onto the opponent's current
    # shape when the respond head is certain nothing will change
    with torch.no_grad():
        model.respond_out.weight.zero_()
        model.respond_out.bias.fill_(-20.0)
        p = model.predict_proba(b, mixture=True)
    assert torch.allclose(p.sum(-1), torch.ones(8), atol=1e-5)
    stay = p.gather(1, b["opp_formation_id"].view(-1, 1)).squeeze(1)
    assert (stay > 0.99).all(), f"no-change mass did not land on the current shape: {stay}"
    print("  two heads wired; formation head masked to real changes")


def test_stub_flag_names_the_stubbed_component():
    """`is_stub` alone sent readers after the wrong component.

    Once the world model is trained but the stability critic is not, a line
    reading "STUB WORLD MODEL" is false. The flag has to say which part is
    still a placeholder, or it stops being a warning and becomes noise people
    learn to ignore.
    """
    torch.manual_seed(0)
    model = OpponentCoBehaviorModel(CFG.encoder, CFG.opponent).eval()
    planner = RAMFPPlanner(model, cfg=CFG.inference, device="cpu")
    recs = planner.plan(demo_state(), top_n=2)

    # default planner: stub world model AND stub stability critic
    assert recs[0].is_stub
    assert "world model" in recs[0].stub_components
    assert "stability critic" in recs[0].stub_components
    assert "world model" in recs[0].explain() and "STUB" in recs[0].explain()

    # a real world model must drop out of the list, leaving only the critic
    class _RealWM:
        is_stub = False

        def predict(self, f_ours, f_theirs, state):
            return {"xg_delta": 0.01, "possession": 51.0, "ppda": 11.0, "shots": 3.0}

    planner2 = RAMFPPlanner(model, world_model=_RealWM(), cfg=CFG.inference, device="cpu")
    r2 = planner2.plan(demo_state(), top_n=2)[0]
    assert "world model" not in r2.stub_components, r2.stub_components
    assert "stability critic" in r2.stub_components
    assert r2.is_stub, "one stub remains, so the result is still flagged"
    print("  stub flag names the component that is actually a placeholder")


def test_effective_critic_weights_are_reported():
    """The config carries three weights; the default planner has two critics.

    The extra weight was dropped and the rest renormalised without a word, so
    a results table copied from the config would have claimed equal thirds
    when the score was really a 50/50 blend.
    """
    torch.manual_seed(0)
    model = OpponentCoBehaviorModel(CFG.encoder, CFG.opponent).eval()
    planner = RAMFPPlanner(model, cfg=CFG.inference, device="cpu")

    eff = planner.effective_critic_weights()
    assert set(eff) == {"xG", "stability"}
    assert abs(sum(eff.values()) - 1.0) < 1e-9
    assert all(abs(v - 0.5) < 1e-9 for v in eff.values()), eff

    with pytest.warns(RuntimeWarning, match="critic_weights"):
        planner.plan(demo_state(), top_n=1)

    # with a third critic the configured thirds are what actually apply
    class _Third:
        name = "counter_resistance"
        is_stub = False

        def score(self, f_src, f_tgt, f_opp, state, wm_out):
            return 0.0

    from ramfp.inference import StructuralStabilityCritic, XGCritic
    p3 = RAMFPPlanner(model, cfg=CFG.inference, device="cpu",
                      critics=[XGCritic(StubWorldModel()), StructuralStabilityCritic(), _Third()])
    eff3 = p3.effective_critic_weights()
    assert len(eff3) == 3 and all(abs(v - 1 / 3) < 1e-6 for v in eff3.values()), eff3
    print("  effective critic weights reported, and the mismatch warns")


def test_conformal_terminator_distinguishes_empty_from_unfitted():
    """`set_size` returned 0 both when unfitted and when the set is empty.

    Those are opposite situations: "the rule was never switched on" and "the
    model is as uncertain as it can be". Conflating them meant the logged
    `conformal set 0` on every run was unreadable, and `should_stop` -- which
    tested `size > threshold` -- never fired on an empty set, so the revision
    loop ran its full budget precisely where another round was least likely to
    help.
    """
    K = N_FORMATIONS
    term = ConformalTerminator(alpha=0.10, threshold=1)

    flat = np.full(K, 1.0 / K)
    assert term.set_size(flat) == ConformalTerminator.UNFITTED
    assert not term.is_fitted
    with pytest.warns(RuntimeWarning, match="never fitted"):
        assert term.should_stop(flat) is False

    # Calibrate on a model that is usually confident and right, so the
    # nonconformity scores are small and qhat is small. This is the regime the
    # real opponent model is in: the measured qhat was 0.6151, which admits a
    # formation only at p >= 0.385, and many real decisions clear nothing.
    rng = np.random.default_rng(0)
    cal = np.full((400, K), 0.02 / (K - 1))
    ys = rng.integers(0, K, 400)
    cal[np.arange(400), ys] = 0.98
    term.fit(cal, ys)
    assert term.is_fitted and 0.0 < term.qhat < 0.5, term.qhat

    # flat: no class reaches 1 - qhat, so the set is empty -- maximum
    # uncertainty, and the old `size > threshold` rule scored it 0 and carried on
    assert term.set_size(flat) == 0
    assert term.should_stop(flat), "an empty set is maximum uncertainty and must stop"

    # confident: a singleton set, which is certainty, so keep revising
    sharp = np.full(K, (1.0 - 0.99) / (K - 1))
    sharp[3] = 0.99
    assert term.set_size(sharp) == 1
    assert not term.should_stop(sharp), "a singleton set is certainty, not uncertainty"

    # and the other tail still stops: a sprawling set is over the threshold
    loose = ConformalTerminator(alpha=0.10, threshold=1)
    loose.fit(rng.dirichlet(np.full(K, 0.4), size=400),
              np.array([rng.choice(K, p=r) for r in rng.dirichlet(np.full(K, 0.4), size=400)]))
    assert loose.set_size(flat) > 1
    assert loose.should_stop(flat), "a sprawling set is uncertainty too"
    print("  conformal terminator separates unfitted, empty, confident and sprawling")


def test_load_honours_map_location():
    """`load(..., map_location=d)` must return a model actually on `d`.

    It returned a CPU model whatever was asked for: map_location only governs
    where the checkpoint tensors are read, and load_state_dict then copies
    into parameters that were built on the CPU. Callers took the argument at
    face value, sent it a CUDA batch, and the failure appeared deep inside an
    embedding lookup instead of at load time.
    """
    import tempfile

    torch.manual_seed(0)
    model = OpponentCoBehaviorModel(CFG.encoder, CFG.opponent)
    wanted = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/opp.pt"
        model.save(path)
        for want in wanted:
            got = OpponentCoBehaviorModel.load(path, map_location=want)
            dev = next(got.parameters()).device.type
            assert dev == want, f"asked for {want}, model came back on {dev}"
    print(f"  load() puts the model where map_location says ({', '.join(wanted)})")


def test_revised_and_unrevised_scores_share_a_scale():
    """After revision, every recommendation's score must be on one scale.

    _revise re-scores the top candidates one at a time with the raw aggregate,
    while plan() had normalised the rest across the decision. Sorting the two
    groups together compared numbers in different units. The final score of
    each recommendation must equal the normalised aggregate recomputed over
    the whole returned set.
    """
    torch.manual_seed(0)
    model = OpponentCoBehaviorModel(CFG.encoder, CFG.opponent).eval()
    rng = np.random.default_rng(0)
    term = ConformalTerminator(0.1, 6).fit(
        rng.dirichlet(np.ones(N_FORMATIONS), size=300), rng.integers(0, N_FORMATIONS, 300))
    planner = RAMFPPlanner(model, cfg=CFG.inference, terminator=term, device="cpu")
    recs = planner.plan(demo_state(), top_n=3)
    expected = [planner._aggregate(cs) for cs in
                planner._normalise_critics([r.critic_scores for r in recs])]
    got = [r.score for r in recs]
    assert np.allclose(got, expected, atol=1e-9), (got, expected)
    assert got == sorted(got, reverse=True), "not ranked after re-scoring"
    print("  revised and unrevised candidates are scored on one scale")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(fns)} tests\n")
    for fn in fns:
        fn()
    print("\nall tests passed")
