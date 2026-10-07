"""Mouth (07 §5): utterance planning, taint check, deterministic articulators and templates.

The Mouth renders only what the plan contains.  Formal targets (DSL, Python, JSON, math)
use deterministic unparsers so the output equals the internal solution.  Natural language
uses per-(role, epistemic tag) templates; the neural renderer (M9) plugs in behind the same
interface and keeps the templates as fallback (D-016).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from srm.body import dsl
from srm.interface.messages import DecisionClass, EpistemicState
from srm.interface.values import Value

KNOWLEDGE_TAGS = {"OBSERVED", "REMEMBERED", "DERIVED", "TESTED"}
PREDICTION_TAGS = {"PREDICTED", "INHERITED", "ANALOGICAL", "EXTRAPOLATED", "SUGGESTED"}


class TaintLeak(RuntimeError):
    pass


@dataclass
class PlanItem:
    role: str  # answer | support | hedge | alternative | abstention | missing_info | explanation | code_block | data_block
    content: Any
    tag: str  # epistemic tag / decision class
    tainted: bool = False
    citations: list[str] = field(default_factory=list)
    format_target: str = "prose"


@dataclass
class UtterancePlan:
    items: list[PlanItem]
    format: str = "prose"
    style: dict[str, Any] = field(default_factory=dict)


@dataclass
class Rendered:
    text: str
    attributions: list[tuple[int, int, int]]  # (start, end, plan item index)
    retagged: list[int] = field(default_factory=list)


Lexicon = Callable[[str], str]


def describe_value(v: Value | None, lexicon: Lexicon) -> str:
    if v is None:
        return "?"
    if v.type == "entity_ref":
        return lexicon(str(v.value))
    if v.type == "code_ref":
        return dsl.Node.from_json(json.loads(v.value)).__str__()
    if v.is_numeric():
        num = _exact_number(v.value)
        return f"{num} {v.unit}" if v.unit else num
    return str(v.value)


def _exact_number(x: Any) -> str:
    """Literal values are copied exactly (07 §5.2 step 5): shortest round-trip representation."""
    f = float(x)
    return str(int(f)) if f.is_integer() and abs(f) < 1e15 else repr(f)


def relation_phrase(relation: str) -> str:
    return relation.split(":", 1)[-1].replace("_", " ")


class Mouth:
    def __init__(self, lexicon: Lexicon) -> None:
        self.lexicon = lexicon

    @staticmethod
    def _program_for(decision: Any, artifacts: dict[str, Any]) -> dict[str, Any] | None:
        """The rendered program must be the best-supported hypothesis, not the first artifact."""
        progs = artifacts.get("programs") or []
        best = decision.best
        if best is not None and best.binding is not None and best.binding.type == "code_ref":
            want = json.loads(best.binding.value)
            for p in progs:
                if p["json"] == want:
                    return p
        return progs[0] if progs else None

    # --- planning (07 §5.1) ---------------------------------------------------------------------------
    def plan(self, request: Any, decision: Any, record: Any, artifacts: dict[str, Any], ws: Any) -> UtterancePlan:
        cls = decision.cls
        fmt = getattr(request, "output_format", "prose")
        best = decision.best
        items: list[PlanItem] = []
        tainted = bool(best is not None and best.tainted)
        tag = (decision.state.value if decision.state is not None else cls.value)
        topic = self._topic(request)
        if cls == DecisionClass.KNOWN or cls == DecisionClass.PREDICTION:
            prog = self._program_for(decision, artifacts) if request.kind in ("dsl_task", "code_task") else None
            if prog is not None:
                items.append(PlanItem("code_block", prog["python"] if fmt == "python" else prog["dsl"], tag, tainted,
                                      format_target="python" if fmt == "python" else "dsl"))
            else:
                items.append(PlanItem("answer", {"topic": topic, "value": best.binding}, tag, tainted,
                                      citations=sorted({s["source_id"] for s in record.sources if s.get("source_id")})[:3]))
        elif cls == DecisionClass.AMBIGUOUS:
            for h in decision.ranked[:3]:
                if not h.is_residual:
                    items.append(PlanItem("alternative", {"topic": topic, "value": h.binding, "b": h.b}, "AMBIGUOUS", h.tainted))
            prog = self._program_for(decision, artifacts) if request.kind in ("dsl_task", "code_task") else None
            if prog is not None:
                items.append(PlanItem("code_block", prog["python"] if fmt == "python" else prog["dsl"], "AMBIGUOUS", tainted,
                                      format_target="python" if fmt == "python" else "dsl"))
        elif cls == DecisionClass.CONTESTED:
            for h in decision.ranked[:3]:
                if not h.is_residual:
                    items.append(PlanItem("alternative", {"topic": topic, "value": h.binding, "b": h.b}, "CONTESTED", h.tainted))
        elif cls == DecisionClass.INSUFFICIENT:
            items.append(PlanItem("hedge", {"topic": topic, "value": best.binding if best else None,
                                            "b": best.b if best else 0.0}, "INSUFFICIENT", tainted))
        elif cls == DecisionClass.UNRESOLVED:
            items.append(PlanItem("abstention", {"topic": topic}, "UNRESOLVED"))
        else:  # UNKNOWN-*
            items.append(PlanItem("abstention", {"topic": topic}, cls.value))
            if record.missing_keys:
                items.append(PlanItem("missing_info", {"keys": record.missing_keys}, cls.value))
        return UtterancePlan(items, fmt)

    def _topic(self, request: Any) -> str:
        if request.kind == "factual" and request.atoms:
            phrase = " of the ".join(relation_phrase(a.relation) for a in reversed(request.atoms))
            subject = request.atoms[0].subject
            name = subject[1:] if subject.startswith("@") else self.lexicon(subject)
            return f"the {phrase} of {name}"
        if request.kind == "math_word":
            return "the answer"
        if request.kind == "algebra":
            return f"the {request.algebra_op}ed form of {request.expr}"
        if request.kind in ("dsl_task", "code_task"):
            return "a program matching the examples"
        return "this"

    # --- rendering (07 §5.2) ---------------------------------------------------------------------------
    def render(self, plan: UtterancePlan) -> Rendered:
        parts: list[str] = []
        attributions: list[tuple[int, int, int]] = []
        retagged: list[int] = []
        pos = 0
        for i, item in enumerate(plan.items):
            if item.tainted and item.tag in KNOWLEDGE_TAGS:  # T5: hard error → re-tag as prediction
                item.tag = EpistemicState.PREDICTED.value
                retagged.append(i)
            text = self._render_item(item)
            if parts:
                text = " " + text if item.role != "code_block" else "\n" + text
            attributions.append((pos + (1 if parts else 0), pos + len(text), i))
            parts.append(text)
            pos += len(text)
        return Rendered("".join(parts), attributions, retagged)

    def _render_item(self, item: PlanItem) -> str:
        c, tag = item.content, item.tag
        if item.role == "code_block":
            return c if item.format_target != "python" else c.rstrip()
        if item.role == "answer":
            value = describe_value(c["value"], self.lexicon)
            if tag in KNOWLEDGE_TAGS:
                return f"{c['topic'][0].upper()}{c['topic'][1:]} is {value}."
            if tag == "EXTRAPOLATED":
                return f"I'm extrapolating beyond what I've verified: {c['topic']} may be {value}."
            return f"{c['topic'][0].upper()}{c['topic'][1:]} is probably {value}."
        if item.role == "alternative":
            value = describe_value(c["value"], self.lexicon)
            if tag == "CONTESTED":
                return f"Sources disagree about {c['topic']}: one says {value}."
            return f"{c['topic'][0].upper()}{c['topic'][1:]} could be {value} (belief {c['b']:.2f})."
        if item.role == "hedge":
            if c.get("value") is None:
                return f"I'm not sure about {c['topic']}."
            value = describe_value(c["value"], self.lexicon)
            return (f"I'm not sure about {c['topic']}. The best-supported option is {value}, "
                    f"but the evidence is weak (belief {c['b']:.2f}).")
        if item.role == "abstention":
            if tag == "UNRESOLVED":
                return f"I couldn't determine {c['topic']} within the budget."
            if tag == "UNKNOWN-DECLARED":
                return f"{c['topic'][0].upper()}{c['topic'][1:]} is not known (according to my sources)."
            return f"I don't know {c['topic']}."
        if item.role == "missing_info":
            keys = [self._describe_key(k) for k in c["keys"][:3]]
            return "Missing information: " + "; ".join(keys) + "."
        return str(c)

    def _describe_key(self, key: str) -> str:
        if key.startswith("er:"):
            ent, rel = key[3:].split("|", 1)
            return f"the {relation_phrase(rel)} of {self.lexicon(ent)}"
        if key.startswith("a:"):
            return f"anything about '{key[2:]}'"
        if key.startswith("e:"):
            return f"anything about {self.lexicon(key[2:])}"
        return key
