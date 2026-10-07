"""M5/M6: perception & parsing (03 Part C), Mouth (07 §5), Heart/Gate/Modulators/Subconscious (07 §1–4)."""

from __future__ import annotations

import numpy as np
import pytest

from srm.config import tiny
from srm.control import BufferItem, Gate, Heart, Modulators
from srm.data.sef import FactPattern, PatternAtom as SefAtom, QuestionRecord
from srm.interface import InterfaceLayer
from srm.interface.messages import Bid, DecisionClass, EpistemicState
from srm.interface.values import Value
from srm.memory import MemorySystem
from srm.mouth import Mouth, PlanItem, UtterancePlan
from srm.perception import Perception, chunk_text, intake, parse_dsl, parse_question
from srm.perception.parsers import parse_math, parse_python


# --- perception & parsing ----------------------------------------------------------------------------
def test_chunking_and_encoder_determinism(cfg) -> None:
    text = "The wheel rotates. " * 3 + "x" * 300
    chunks = chunk_text(text, 128)
    assert all(len(c.encode()) <= 128 for c in chunks) and chunks[0] == "The wheel rotates."
    il = InterfaceLayer(cfg)
    p1, p2 = Perception(cfg, il, MemorySystem(cfg, il)), Perception(cfg, il, MemorySystem(cfg, il))
    assert np.allclose(p1.encoder.encode("hello world"), p2.encoder.encode("hello world"))


def test_recognition_by_recall(cfg) -> None:
    il = InterfaceLayer(cfg)
    mem = MemorySystem(cfg, il)
    per = Perception(cfg, il, mem)
    first = per.process("Gears transmit torque between shafts.", remember=True)
    assert not first.chunks[0].known
    again = per.process("Gears transmit torque between shafts.")
    assert again.chunks[0].known and again.chunks[0].pointer is not None  # known content becomes a pointer
    other = per.process("Volcanoes erupt molten rock.")
    assert not other.chunks[0].known


def test_question_parser_handles_chains_and_time() -> None:
    lex = {"capital": "rel:capital", "located in": "rel:located_in", "born in": "rel:born_in", "ceo": "rel:ceo",
           "population": "rel:population"}
    atoms, q = parse_question("What is the located in of the born in of the ceo of Acme Corp?", lex)
    assert [a.relation for a in atoms] == ["rel:ceo", "rel:born_in", "rel:located_in"]
    assert atoms[0].subject == "@Acme Corp" and atoms[-1].object == "?x"
    atoms, q = parse_question("What is the population of Freedonia in 1995?", lex)
    assert q.time == (1995.0, 1996.0)
    assert parse_question("What is the flavor of Freedonia?", lex) is None


def test_math_dsl_python_parsers() -> None:
    assert parse_math("Expand (x+1)*(x-2).").algebra_op == "expand"
    assert parse_math("Ann buys 3 pens at 2 dollars each. How much does Ann pay?").kind == "math_word"
    node = parse_dsl("(map (add_n 3) (filter is_even xs))")
    assert str(node) == "(map (add_n 3) (filter is_even xs))"
    with pytest.raises(Exception):
        parse_dsl("(map xs xs)")
    summary = parse_python("def f(a):\n    return g(a)\n\ndef g(b):\n    return b\n")
    assert summary.functions == ["f", "g"] and ("f", "g") in summary.calls


def test_intake_routes_requests() -> None:
    lex = {"capital": "rel:capital"}
    assert intake("What is the capital of Freedonia?", lex).kind == "factual"
    assert intake({"examples": [{"xs": [1], "k": 0, "out": 1}], "output_type": "int"}, lex).ret_type == "int"
    q = QuestionRecord(record_id="q", source_id="s", query="?", gold_state="KNOWN",
                       pattern=FactPattern(atoms=[SefAtom(subject="@A", relation="rel:capital", object="?x")]))
    assert intake(q, lex).atoms[0].relation == "rel:capital"
    assert intake("Sing me a song", lex).kind == "unknown"


# --- Mouth ----------------------------------------------------------------------------------------------
def test_mouth_retags_tainted_knowledge_and_attributes_spans() -> None:
    mouth = Mouth(lambda s: s.split(":")[-1].title())
    plan = UtterancePlan([
        PlanItem("answer", {"topic": "the capital of Freedonia", "value": Value("entity_ref", "ent:troutio")},
                 "REMEMBERED", tainted=True),
        PlanItem("missing_info", {"keys": ["er:ent:freedonia|rel:population"]}, "UNKNOWN-ABSENT"),
    ])
    out = mouth.render(plan)
    assert out.retagged == [0] and "probably Troutio" in out.text  # T5: tainted content is never asserted
    assert "the population of Freedonia" in out.text
    for start, end, idx in out.attributions:
        assert 0 <= start < end <= len(out.text)


@pytest.mark.parametrize("tag, phrase", [
    ("REMEMBERED", "is 5"), ("INHERITED", "probably 5"), ("EXTRAPOLATED", "extrapolating"),
])
def test_mouth_tag_policy(tag: str, phrase: str) -> None:
    mouth = Mouth(str)
    text = mouth.render(UtterancePlan([PlanItem("answer", {"topic": "x", "value": Value("int", 5)}, tag)])).text
    assert phrase in text


# --- control ------------------------------------------------------------------------------------------
def test_heart_allocates_by_value_density_within_caps() -> None:
    cfg = tiny().with_overrides({"control": {"max_flops_per_beat": 1e6}})
    heart = Heart(cfg)
    bids = [Bid("cheap_valuable", {"flops": 1e5}, 1.0), Bid("expensive", {"flops": 9.5e5}, 1.0),
            Bid("worthless", {"flops": 1e3}, 0.0), Bid("bg", {"flops": 5e4}, 0.01)]
    alloc = heart.allocate(bids, remaining_flops=1e9, categories={"bg": "subconscious"})
    names = {b.process_id for b in alloc.funded}
    assert {"cheap_valuable", "bg"} <= names and "worthless" not in names and "expensive" not in names
    assert alloc.flops <= 1e6
    heart.update_credit("cheap_valuable", predicted=1.0, realized=0.0)
    assert heart.credit["cheap_valuable"] < 1.0
    assert heart.query_budget(1.0) > heart.query_budget(0.0)


def test_gate_admits_salient_and_inhibits_duplicates() -> None:
    gate = Gate()
    code = np.zeros(64, dtype=np.uint8)
    items = [BufferItem(1, "p", relevance=0.9, code=code), BufferItem(2, "p", relevance=0.8, code=code.copy()),
             BufferItem(3, "p", relevance=0.1, code=np.ones(64, dtype=np.uint8))]
    chosen = gate.admit(items, free_slots=2)
    assert [c.record_id for c in chosen] == [1, 3]


def test_modulators_respond_to_signals() -> None:
    m = Modulators()
    m.update(td_error=0.5, prediction_error=1.0, surprise_events=2, stakes=1.0, budget_fraction_left=1.0)
    assert m.da == 0.5 and m.ach > 0 and m.ne >= 2 and m.ht5 == 1.0
    assert m.gain_ach > 1.0
