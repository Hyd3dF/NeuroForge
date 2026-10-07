"""Goal decision procedure (06 §6, including the D-022 corrections).

Order of evaluation:
1. UNKNOWN-ABSENT   a goal key is absent from the Sketch, not in the input and not derivable
2. UNKNOWN-DECLARED the goal matches an open question and no untainted committed answer exists
3. CONTESTED        the best hypothesis is involved in an unresolved contradiction
4. KNOWN            untainted, grounded, b ≥ θ_answer(stakes), margin ≥ δ_margin
5. AMBIGUOUS        several non-residual hypotheses within δ_margin
6. PREDICTION       tainted but in envelope with b ≥ θ_predict
7. UNKNOWN-NO-BASIS no hypotheses, or only out-of-envelope (EXTRAPOLATED) ones
8. UNRESOLVED       budget exhausted before a basis or an answer was established
9. INSUFFICIENT     relevant hypotheses exist but none qualifies (D-022 catch-all)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from srm.config.build_config import BuildConfig
from srm.interface.messages import DecisionClass, EpistemicState
from srm.prediction.support import HypothesisEval, rank


def theta_answer(stakes: float, cfg: BuildConfig) -> float:
    e = cfg.epistemics
    return e.theta_abstain + (e.theta_max - e.theta_abstain) * min(1.0, max(0.0, stakes))


@dataclass
class DecisionInputs:
    goal_keys: list[str]
    absent: Callable[[str], bool]                      # Sketch: definitely absent?
    derivable: Callable[[str], bool] = lambda key: False  # REGRESS rule lookup (06 §6.0)
    input_keys: set[str] = field(default_factory=set)  # keys supplied by the current input
    open_question: bool = False
    hypotheses: list[HypothesisEval] = field(default_factory=list)
    budget_exhausted: bool = False
    pending_derivations: bool = False
    stakes: float = 0.5


@dataclass
class Decision:
    cls: DecisionClass
    state: EpistemicState | None
    best: HypothesisEval | None
    ranked: list[HypothesisEval]
    reason: str
    theta_answer: float
    missing_keys: list[str] = field(default_factory=list)

    @property
    def answer(self):  # noqa: ANN201
        return self.best.binding if self.best is not None else None


def decide(x: DecisionInputs, cfg: BuildConfig) -> Decision:
    e = cfg.epistemics
    th = theta_answer(x.stakes, cfg)
    ranked = rank(list(x.hypotheses), cfg)

    def out(cls: DecisionClass, state: EpistemicState | None, best: HypothesisEval | None, reason: str,
            missing: Iterable[str] = ()) -> Decision:
        return Decision(cls, state, best, ranked, reason, th, list(missing))

    missing = [k for k in x.goal_keys if x.absent(k) and k not in x.input_keys and not x.derivable(k)]
    if missing:
        return out(DecisionClass.UNKNOWN_ABSENT, EpistemicState.UNKNOWN_ABSENT, None,
                   "goal keys never stored and not derivable", missing)

    real = [h for h in ranked if not h.is_residual]
    if x.open_question and not any(not h.tainted and h.b >= e.theta_commit for h in real):
        return out(DecisionClass.UNKNOWN_DECLARED, EpistemicState.UNKNOWN_DECLARED, None,
                   "matches a declared open question")

    if not real:
        if x.budget_exhausted and x.pending_derivations:
            return out(DecisionClass.UNRESOLVED, EpistemicState.UNRESOLVED, None, "budget exhausted")
        return out(DecisionClass.UNKNOWN_NO_BASIS, EpistemicState.UNKNOWN_NO_BASIS, None, "no applicable hypothesis")

    best = real[0]
    second = next((h for h in ranked if h is not best), None)
    margin = best.support - second.support if second is not None else float("inf")

    if best.n_unresolved_contradictions > 0:
        return out(DecisionClass.CONTESTED, EpistemicState.CONTESTED, best, "unresolved contradiction")

    if (not best.tainted and best.grounded and best.b >= th and margin >= e.delta_margin):
        return out(DecisionClass.KNOWN, best.knowledge_state or EpistemicState.REMEMBERED, best, "best-supported and grounded")

    close = [h for h in real[1:] if best.support - h.support < e.delta_margin]
    if close and best.b >= e.theta_predict:
        return out(DecisionClass.AMBIGUOUS, None, best, f"{len(close) + 1} hypotheses within the margin")

    if best.tainted and best.in_envelope and best.b >= e.theta_predict:
        state = best.knowledge_state if best.knowledge_state is not None else EpistemicState.PREDICTED
        return out(DecisionClass.PREDICTION, state, best, "prediction-class answer (tainted)")

    if all(not h.in_envelope for h in real):
        return out(DecisionClass.UNKNOWN_NO_BASIS, EpistemicState.UNKNOWN_NO_BASIS, None,
                   "only extrapolated hypotheses (no basis for a factual answer)")

    if x.budget_exhausted and x.pending_derivations:
        return out(DecisionClass.UNRESOLVED, EpistemicState.UNRESOLVED, best, "budget exhausted")

    return out(DecisionClass.INSUFFICIENT, EpistemicState.INSUFFICIENT, best,
               f"best belief {best.b:.2f} below θ_answer {th:.2f} or not grounded")
