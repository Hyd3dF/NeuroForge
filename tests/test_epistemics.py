"""M3: evidence algebra (05 §2), truth maintenance (05 §3), taint (06 §2), Support (06 §4),
decision procedure (06 §6) and the taint-leak invariant (12 §4.6)."""

from __future__ import annotations

import pytest

from srm.core import tms
from srm.core.evidence import EvidenceLedger, belief, inherited_evidence, modality_factor
from srm.core.workspace import Claim, Workspace, WorkspaceFull
from srm.interface.messages import ClaimStatus, DecisionClass, EpistemicState
from srm.interface.values import Value
from srm.prediction import taint as T
from srm.prediction.decision import DecisionInputs, decide, theta_answer
from srm.prediction.support import HypothesisEval, evaluate


def ledger(plus: float = 0.0, minus: float = 0.0, root: str = "r") -> EvidenceLedger:
    led = EvidenceLedger()
    led.add(root, plus, minus)
    return led


# --- evidence algebra ---------------------------------------------------------------------------
def test_belief_formula_and_calibration(cfg) -> None:
    W = cfg.epistemics.W
    assert belief(0.9, 0.0, W) == pytest.approx(0.9 / (0.9 + W))
    # D-021: a single curated source reaches θ_answer at default stakes; a web source does not
    assert belief(0.9, 0.0, W) >= theta_answer(0.5, cfg)
    assert belief(0.5, 0.0, W) < theta_answer(0.5, cfg)
    assert belief(1.0, 0.0, W) >= theta_answer(0.5, cfg)  # two independent web sources


def test_inherited_evidence_reproduces_belief(cfg) -> None:
    W = cfg.epistemics.W
    for b in (0.1, 0.5, 0.9):
        assert belief(inherited_evidence(b, W, 1e9), 0.0, W) == pytest.approx(b)


def test_modality_factors(cfg) -> None:
    e = cfg.epistemics
    assert modality_factor("asserted", 0.0, e.modality_factors, e.hedge_slope) == 1.0
    assert modality_factor("hedged", 0.6, e.modality_factors, e.hedge_slope) == pytest.approx(0.58)
    assert modality_factor("reported", 0.0, e.modality_factors, e.hedge_slope) == 0.0


# --- workspace and truth maintenance ------------------------------------------------------------
def base_claim(ws: Workspace, cid: str, plus: float = 0.0, premises: tuple[str, ...] = (), **kw) -> Claim:
    c = Claim(cid, "s", "rel:r", Value("int", len(ws.claims)), premises=premises, **kw)
    if plus:
        c.ledger = ledger(plus, root=cid)
    ws.add_claim(c)
    if premises:
        tms.refresh(ws, cid)
    return c


def test_derived_belief_and_retraction_cascade(cfg) -> None:
    ws = Workspace(cfg)
    base_claim(ws, "a", 1.0)
    base_claim(ws, "b", 2.0)
    base_claim(ws, "c", premises=("a", "b"))
    base_claim(ws, "d", premises=("c",))
    assert ws.belief("c") == pytest.approx(min(ws.belief("a"), ws.belief("b")))
    assert ws.belief("d") == pytest.approx(ws.belief("c"))
    ws.claims["a"].ledger.add("refuter", 0.0, 5.0)
    assert tms.check_retraction(ws, "a")
    assert ws.claims["a"].status == ClaimStatus.RETRACTED
    assert ws.belief("c") == 0.0 and ws.claims["d"].status == ClaimStatus.DORMANT


def test_contradiction_without_assumptions_retracts_weaker(cfg) -> None:
    ws = Workspace(cfg)
    base_claim(ws, "strong", 2.0)
    base_claim(ws, "weak", 0.3)
    out = tms.handle_contradiction(ws, "strong", "weak")
    assert out.kind == "retracted" and out.retracted == "weak"


def test_contradiction_in_channel_kills_channel_and_records_nogood(cfg) -> None:
    ws = Workspace(cfg)
    base_claim(ws, "fact", 2.0)
    a1 = base_claim(ws, "assume1", 1.0)
    ch = ws.open_channel(a1)
    assert ch is not None and a1.state == EpistemicState.CONJECTURED and a1.tainted
    base_claim(ws, "derived", premises=("assume1", "fact"))
    assert ws.claims["derived"].channel_mask == 1 << ch
    out = tms.handle_contradiction(ws, "derived", "fact")
    assert out.kind == "channels_killed" and ch in out.killed_channels
    assert ws.claims["derived"].status == ClaimStatus.DORMANT
    assert ws.is_nogood(frozenset({a1.content_key()}))
    assert ws.claims["fact"].status != ClaimStatus.DORMANT  # global claims survive


def test_claims_in_different_channels_do_not_conflict(cfg) -> None:
    ws = Workspace(cfg)
    a = base_claim(ws, "a1", 1.0)
    b = base_claim(ws, "b1", 1.0)
    ws.open_channel(a)
    ws.open_channel(b)
    assert tms.handle_contradiction(ws, "a1", "b1").kind == "different_channels"


def test_workspace_capacity_and_eviction(cfg) -> None:
    ws = Workspace(cfg)
    for i in range(ws.K):
        base_claim(ws, f"x{i}", 0.5, relevance=float(i))
    base_claim(ws, "new", 0.5)
    assert "x0" not in ws.claims and len(ws.claims) == ws.K
    ws.protected.update(ws.claims)
    with pytest.raises(WorkspaceFull):
        base_claim(ws, "overflow", 0.5)


# --- taint ----------------------------------------------------------------------------------------
def test_taint_propagation_and_promotion_rules(cfg) -> None:
    ws = Workspace(cfg)
    base_claim(ws, "obs", 2.0)
    pred = base_claim(ws, "pred", 0.5, producer_predictive=True, tainted=True)
    d = base_claim(ws, "ded", premises=("obs", "pred"), producer_exact=True)
    d.tainted = T.derive_taint(ws, d.premises, d.producer_predictive)
    assert d.tainted  # T2: OR of premises
    with pytest.raises(T.TaintViolation):
        T.promote(ws, "pred", T.Promotion.VERIFIER, cfg)  # D-011
    with pytest.raises(T.TaintViolation):
        T.promote(ws, "pred", T.Promotion.CORROBORATION, cfg, corroborating_belief=0.5)
    with pytest.raises(T.TaintViolation):
        T.promote(ws, "ded", T.Promotion.UNTAINTED_DEDUCTION, cfg)  # premise still tainted
    T.promote(ws, "pred", T.Promotion.TEST_PASS, cfg)
    assert not pred.tainted and pred.state == EpistemicState.TESTED
    assert not d.tainted and d.state == EpistemicState.DERIVED  # cascade through exact deduction
    with pytest.raises(T.TaintViolation):
        T.check_invariant(Claim("z", "s", "r", "o", tainted=True), rendered_as_knowledge=True)


# --- Support and decision procedure -------------------------------------------------------------
def H(hid: str, plus: float = 0.0, minus: float = 0.0, **kw) -> HypothesisEval:
    return HypothesisEval(hyp_id=hid, binding=Value("int", hash(hid) % 100), ledger=ledger(plus, minus, hid), **kw)


RESIDUAL = HypothesisEval(hyp_id="residual", binding=None, is_residual=True)


def test_support_prefers_evidence_over_generator_probability(cfg) -> None:
    supported = evaluate(H("a", 0.9, grounded=True), cfg)
    popular = evaluate(H("b", 0.1, log_p_gen=10.0), cfg)  # very probable proposal, weak evidence
    assert supported.support > popular.support
    assert popular.support <= evaluate(H("c", 0.1), cfg).support + cfg.epistemics.pi_cap + 1e-9


def run(cfg, hyps, **kw):
    x = DecisionInputs(goal_keys=kw.pop("keys", ["k"]), absent=kw.pop("absent", lambda k: False),
                       hypotheses=hyps + [RESIDUAL], **kw)
    return decide(x, cfg)


def test_decision_procedure_reaches_every_state(cfg) -> None:
    assert run(cfg, [], absent=lambda k: True).cls == DecisionClass.UNKNOWN_ABSENT
    assert run(cfg, [], absent=lambda k: True, derivable=lambda k: True).cls == DecisionClass.UNKNOWN_NO_BASIS
    assert run(cfg, [], open_question=True).cls == DecisionClass.UNKNOWN_DECLARED
    known = run(cfg, [H("a", 0.9, grounded=True, knowledge_state=EpistemicState.REMEMBERED)])
    assert known.cls == DecisionClass.KNOWN and known.state == EpistemicState.REMEMBERED
    assert run(cfg, [H("a", 0.9, grounded=True, n_unresolved_contradictions=1)]).cls == DecisionClass.CONTESTED
    assert run(cfg, [H("a", 0.5, grounded=True)]).cls == DecisionClass.INSUFFICIENT
    assert run(cfg, [H("a", 0.9, grounded=True), H("b", 0.88, grounded=True)]).cls == DecisionClass.AMBIGUOUS
    pred = run(cfg, [H("a", 0.9, tainted=True, knowledge_state=EpistemicState.INHERITED)])
    assert pred.cls == DecisionClass.PREDICTION and pred.state == EpistemicState.INHERITED
    assert run(cfg, [H("a", 0.9, tainted=True, in_envelope=False)]).cls == DecisionClass.UNKNOWN_NO_BASIS
    assert run(cfg, [], budget_exhausted=True, pending_derivations=True).cls == DecisionClass.UNRESOLVED


def test_known_class_never_returns_a_tainted_hypothesis(cfg) -> None:
    """Taint-leak invariant: no combination yields KNOWN with a tainted best hypothesis."""
    import itertools

    for plus, tainted, grounded, stakes in itertools.product((0.1, 0.5, 0.9, 5.0), (True, False), (True, False), (0.0, 0.5, 1.0)):
        d = run(cfg, [H("a", plus, tainted=tainted, grounded=grounded)], stakes=stakes)
        if d.cls == DecisionClass.KNOWN:
            assert not d.best.tainted and d.best.grounded and d.best.b >= d.theta_answer


def test_stakes_raise_the_answer_threshold(cfg) -> None:
    h = H("a", 1.0, grounded=True)
    assert run(cfg, [h], stakes=0.0).cls == DecisionClass.KNOWN
    assert run(cfg, [H("a", 1.0, grounded=True)], stakes=1.0).cls == DecisionClass.INSUFFICIENT
