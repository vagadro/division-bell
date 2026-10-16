"""Lineage tests: decide, blast radius, retract, correct, accuracy. Demo chain: f1 <- f2 <- d1 <- d2, control f3.

Only the author of a fact can supersede, correct or retract it, so in these tests the Researcher
(author of f1) is the one who retracts or corrects f1.
"""
import time

import pytest

from memory import MemoryGraphError

DAY = 24 * 3600 * 1000


@pytest.fixture
def chain(mem):
    """Researcher -> Planner -> Writer build on f1 across three sessions."""
    r, p, w = (mem.register_agent(n) for n in ("Researcher", "Planner", "Writer"))
    sr, sp, sw = (mem.start_session(a) for a in (r, p, w))
    f1 = mem.remember(r, sr, "Acme has 500 employees", "org:acme", confidence=0.8)
    mem.publish(r, f1)
    f2 = mem.remember(p, sp, "Acme is an enterprise-size account", "org:acme", derived_from=[f1])
    mem.publish(p, f2)
    f3 = mem.remember(r, sr, "Acme HQ is in Austin", "org:acme")  # control: not built on f1
    mem.publish(r, f3)
    d1 = mem.decide(p, sp, "Pitch the enterprise plan", derived_from=[f2])
    d2 = mem.decide(w, sw, "Write the pitch deck", derived_from=[d1])
    return dict(mem=mem, r=r, p=p, w=w, sr=sr, sp=sp, sw=sw, f1=f1, f2=f2, f3=f3, d1=d1, d2=d2)


def status(mem, item_id):
    return mem.g.query("MATCH (x {id:$id}) RETURN x.status", {"id": item_id}).result_set[0][0]


# --- decide ---

def test_decide_builds_lineage_and_explain_shows_it(chain):
    c = chain
    trail = c["mem"].explain(c["w"], c["d2"])
    assert [(t["id"], t["agent"], t["depth"]) for t in trail] == [
        (c["d2"], c["w"], 0), (c["d1"], c["p"], 1), (c["f2"], c["p"], 2), (c["f1"], c["r"], 3)]


def test_decide_only_on_active_items_the_agent_may_see(team):
    mem, r, sr, p, sp = team
    secret = mem.remember(r, sr, "Researcher's private note", "org:acme")
    with pytest.raises(MemoryGraphError):
        mem.decide(p, sp, "x", derived_from=[secret])  # another owner's private fact
    with pytest.raises(MemoryGraphError):
        mem.decide(p, sp, "x", derived_from=["fact:nope"])
    with pytest.raises(MemoryGraphError):
        mem.decide(r, sp, "x")  # Researcher using the Planner's session


def visibility(mem, item_id):
    return mem.g.query("MATCH (x {id:$id}) RETURN x.visibility", {"id": item_id}).result_set[0][0]


def test_a_decision_is_as_visible_as_its_sources(mem):
    r, p = (mem.register_agent(n, owner="Deepak") for n in ("Researcher", "Planner"))
    ravi = mem.register_agent("Planner", owner="Ravi")
    sr, sp = mem.start_session(r), mem.start_session(p)
    salary = mem.remember(r, sr, "Deepak's salary is $20,000", "user:deepak")      # private
    budget = mem.decide(p, sp, "Laptop budget: $1,000", derived_from=[salary])    # the user's other agent may
    assert visibility(mem, budget) == "private"
    with pytest.raises(MemoryGraphError):
        mem.explain(ravi, budget)                                                  # another user may not see it
    later = mem.decide(p, sp, "Buy in November", derived_from=[budget])
    assert visibility(mem, later) == "private"                                     # private all the way down
    with pytest.raises(MemoryGraphError):
        mem.publish(p, budget)                                                     # its source is still private
    mem.publish(r, salary)
    mem.publish(p, budget)
    assert [x["id"] for x in mem.explain(ravi, budget)][0] == budget


def test_a_decision_on_shared_sources_is_shared(chain):
    assert visibility(chain["mem"], chain["d1"]) == "shared"


# --- blast radius (dry run) ---

def test_blast_radius_lists_downstream_with_depth_and_changes_nothing(chain):
    c = chain
    radius = c["mem"].blast_radius(c["r"], c["f1"])
    assert [(x["id"], x["depth"], x["agent"]) for x in radius] == [
        (c["f2"], 1, c["p"]), (c["d1"], 2, c["p"]), (c["d2"], 3, c["w"])]
    assert c["f3"] not in [x["id"] for x in radius]
    assert all(status(c["mem"], c[k]) == "active" for k in ("f1", "f2", "f3", "d1", "d2"))


# --- retract ---

def test_retract_flags_exactly_the_downstream_items(chain):
    c = chain
    mem = c["mem"]
    out = mem.retract(c["r"], c["f1"], reason="source was a typo: it is 50")
    assert out["retracted"] == c["f1"]
    assert out["flagged"] == sorted([c["f2"], c["d1"], c["d2"]])
    assert out["hidden_flagged"] == 0
    assert out["agents"] == sorted([c["r"], c["p"], c["w"]])
    assert status(mem, c["f1"]) == "retracted"
    assert [status(mem, c[k]) for k in ("f2", "d1", "d2")] == ["flagged"] * 3
    assert status(mem, c["f3"]) == "active"  # the control fact is untouched
    row = mem.g.query("MATCH (f:Fact {id:$id}) RETURN f.retracted_by, f.retract_reason",
                      {"id": c["f1"]}).result_set[0]
    assert row == [c["r"], "source was a typo: it is 50"]


def test_flagged_facts_leave_recall_and_show_up_for_their_owners(chain):
    c = chain
    mem = c["mem"]
    mem.retract(c["r"], c["f1"])
    assert [f["id"] for f in mem.recall(c["w"], "org:acme")] == [c["f3"]]
    assert mem.flagged(c["p"]) == [
        {"id": i, "text": t, "because": [c["f1"]], "wrong": [c["f1"]], "changed": [],
         "author": c["p"], "kind": k} for i, t, k in sorted(
            [(c["d1"], "Pitch the enterprise plan", "decision"),
             (c["f2"], "Acme is an enterprise-size account", "fact")])]
    assert [f["id"] for f in mem.flagged(c["w"])] == [c["d2"]]
    assert mem.flagged(c["r"]) == []  # retracted is not flagged


def test_retract_is_a_mistake_for_the_author_only(chain):
    c = chain
    mem = c["mem"]
    mem.retract(c["r"], c["f1"])
    assert mem.records([c["r"]])[c["r"]]["mistakes"] == 1 and mem.accuracy(c["r"]) < 1.0
    assert mem.records([c["p"]])[c["p"]]["mistakes"] == 0 and mem.records([c["w"]])[c["w"]]["mistakes"] == 0


def test_retract_rules(chain):
    c = chain
    mem = c["mem"]
    mem.retract(c["r"], c["f1"])
    with pytest.raises(MemoryGraphError):
        mem.retract(c["r"], c["f1"])          # already retracted
    with pytest.raises(MemoryGraphError):
        mem.retract(c["p"], c["d1"])          # decisions are flagged, not retracted
    with pytest.raises(MemoryGraphError):
        mem.retract(c["r"], "fact:nope")
    secret = mem.remember(c["r"], c["sr"], "private", "org:acme")
    with pytest.raises(MemoryGraphError):
        mem.retract(c["p"], secret)           # cannot retract what you cannot see


def test_nothing_can_be_built_on_retracted_or_flagged_items(chain):
    c = chain
    mem = c["mem"]
    mem.retract(c["r"], c["f1"])
    with pytest.raises(MemoryGraphError):
        mem.remember(c["p"], c["sp"], "x", "org:acme", derived_from=[c["f1"]])  # retracted
    with pytest.raises(MemoryGraphError):
        mem.remember(c["p"], c["sp"], "x", "org:acme", derived_from=[c["f2"]])  # flagged
    with pytest.raises(MemoryGraphError):
        mem.decide(c["p"], c["sp"], "x", derived_from=[c["d1"]])               # flagged
    assert mem.decide(c["p"], c["sp"], "ok", derived_from=[c["f3"]])            # control is fine


def test_private_downstream_is_flagged_but_hidden_from_the_retractor(chain):
    c = chain
    mem = c["mem"]
    private = mem.remember(c["p"], c["sp"], "Planner's private idea", "org:acme", derived_from=[c["f1"]])
    out = mem.retract(c["r"], c["f1"])
    assert private not in out["flagged"] and out["hidden_flagged"] == 1
    assert status(mem, private) == "flagged"
    assert private in [f["id"] for f in mem.flagged(c["p"])]  # its owner still finds out


def test_two_retractions_both_recorded_on_a_shared_dependent(team):
    mem, r, sr, p, sp = team
    a = mem.remember(r, sr, "fact a", "org:x"); mem.publish(r, a)
    b = mem.remember(r, sr, "fact b", "org:x"); mem.publish(r, b)
    d = mem.decide(p, sp, "built on both", derived_from=[a, b])
    mem.retract(r, a)
    mem.retract(r, b)
    assert mem.flagged(p) == [{"id": d, "text": "built on both", "because": sorted([a, b]),
                               "wrong": sorted([a, b]), "changed": [], "author": p, "kind": "decision"}]


def test_handoff_recipients_are_among_the_agents_to_notify(chain):
    c = chain
    mem = c["mem"]
    editor = mem.register_agent("Editor")  # asserted nothing, but was handed f2
    mem.g.query(
        "MATCH (p:Agent {id:$p}), (e:Agent {id:$e}), (f:Fact {id:$f}) "
        "CREATE (p)-[:SENT]->(h:Handoff {id:'handoff:t', at:1}), (h)-[:TO]->(e), (h)-[:CARRIED]->(f)",
        {"p": c["p"], "e": editor, "f": c["f2"]})
    out = mem.retract(c["r"], c["f1"])
    assert editor in out["agents"]


def test_supersede_flags_downstream_as_changed_not_wrong_and_is_no_mistake(chain):
    c = chain
    mem = c["mem"]
    out = mem.supersede(c["r"], c["sr"], c["f1"], "Acme has 520 employees")  # true, then changed
    new = out["new"]
    assert out["superseded"] == c["f1"]
    assert out["flagged"] == sorted([c["f2"], c["d1"], c["d2"]])
    assert [status(mem, c[k]) for k in ("f2", "d1", "d2")] == ["flagged"] * 3
    assert status(mem, c["f1"]) == "superseded" and status(mem, c["f3"]) == "active"
    assert mem.records([c["r"]])[c["r"]]["mistakes"] == 0  # nothing was wrong
    item = {f["id"]: f for f in mem.flagged(c["p"])}[c["d1"]]
    assert item["wrong"] == [] and item["changed"] == [{"fact": c["f1"], "now": new}]


def test_confirm_brings_a_changed_item_back_but_not_a_wrong_one(chain):
    c = chain
    mem = c["mem"]
    mem.supersede(c["r"], c["sr"], c["f1"], "Acme has 520 employees")
    mem.confirm(c["p"], c["f2"])  # Planner reviewed: still an enterprise account
    assert status(mem, c["f2"]) == "active"
    assert c["f2"] in [f["id"] for f in mem.recall(c["w"], "org:acme")]
    assert status(mem, c["d1"]) == "flagged"  # each item is reviewed on its own
    assert c["f2"] not in [f["id"] for f in mem.flagged(c["p"])]


def test_confirm_rules(chain):
    c = chain
    mem = c["mem"]
    mem.supersede(c["r"], c["sr"], c["f1"], "Acme has 520 employees")
    with pytest.raises(MemoryGraphError):
        mem.confirm(c["w"], c["f2"])      # not the Planner's item
    with pytest.raises(MemoryGraphError):
        mem.confirm(c["r"], c["f3"])      # not flagged
    mem.confirm(c["p"], c["f2"])
    with pytest.raises(MemoryGraphError):
        mem.confirm(c["p"], c["f2"])      # no longer flagged


def test_a_wrong_source_cannot_be_confirmed_away(chain):
    c = chain
    mem = c["mem"]
    mem.retract(c["r"], c["f1"])
    with pytest.raises(MemoryGraphError, match="redo"):
        mem.confirm(c["p"], c["f2"])
    assert status(mem, c["f2"]) == "flagged"


def test_explain_shows_which_link_was_retracted(chain):
    c = chain
    mem = c["mem"]
    mem.retract(c["r"], c["f1"])
    trail = {t["id"]: t["status"] for t in mem.explain(c["w"], c["d2"])}
    assert trail == {c["d2"]: "flagged", c["d1"]: "flagged", c["f2"]: "flagged", c["f1"]: "retracted"}


# --- only the author can change a fact ---

def test_other_agents_cannot_supersede_correct_or_retract_a_shared_fact(chain):
    c = chain
    mem = c["mem"]
    nodes_before = mem.g.query("MATCH (n) RETURN count(n)").result_set[0][0]
    with pytest.raises(MemoryGraphError, match="only the author"):
        mem.supersede(c["p"], c["sp"], c["f1"], "Acme has 50 employees")
    with pytest.raises(MemoryGraphError, match="only the author"):
        mem.correct(c["p"], c["sp"], c["f1"], "Acme has 50 employees")
    with pytest.raises(MemoryGraphError, match="only the author"):
        mem.retract(c["p"], c["f1"])
    # nothing moved: the fact, everything built on it, and everyone's record are as before
    assert [status(mem, c[k]) for k in ("f1", "f2", "f3", "d1", "d2")] == ["active"] * 5
    assert mem.records([c["r"]])[c["r"]]["mistakes"] == 0
    assert mem.g.query("MATCH (n) RETURN count(n)").result_set[0][0] == nodes_before
    assert c["f1"] in [f["id"] for f in mem.recall(c["p"], "org:acme")]  # Planner's view unchanged


def test_a_disagreeing_agent_can_still_write_its_own_competing_fact(chain):
    c = chain
    mem = c["mem"]
    mine = mem.remember(c["p"], c["sp"], "Acme has 50 employees", "org:acme")  # private, its own memory
    assert mine in [f["id"] for f in mem.recall(c["p"], "org:acme")]
    assert mine not in [f["id"] for f in mem.recall(c["r"], "org:acme")]
    assert status(mem, c["f1"]) == "active"


# --- correct: retract + write the right fact in one step ---

def test_correct_flags_downstream_and_writes_the_right_fact(chain):
    c = chain
    mem = c["mem"]
    out = mem.correct(c["r"], c["sr"], c["f1"], "Acme has 50 employees", reason="typo: it is 50")
    new = out["new"]
    assert out["retracted"] == c["f1"]
    assert out["flagged"] == sorted([c["f2"], c["d1"], c["d2"]])
    assert status(mem, c["f1"]) == "retracted" and status(mem, new) == "active"
    assert status(mem, c["f3"]) == "active"
    assert mem.records([c["r"]])[c["r"]]["mistakes"] == 1
    # the corrected fact is shared (like the old one), about Acme, and linked by CORRECTS
    assert sorted(f["id"] for f in mem.recall(c["w"], "org:acme")) == sorted([new, c["f3"]])
    links = mem.g.query("MATCH (n:Fact {id:$n})-[:CORRECTS]->(o) RETURN o.id", {"n": new}).result_set
    assert links == [[c["f1"]]]
    assert mem.g.query("MATCH (:Fact {id:$n})-[:SUPERSEDES]->() RETURN 1", {"n": new}).result_set == []


def test_correct_is_all_or_nothing(chain):
    c = chain
    mem = c["mem"]
    nodes_before = mem.g.query("MATCH (n) RETURN count(n)").result_set[0][0]
    with pytest.raises(MemoryGraphError):
        mem.correct(c["r"], c["sp"], c["f1"], "x")  # Researcher using the Planner's session
    assert status(mem, c["f1"]) == "active" and status(mem, c["f2"]) == "active"
    assert mem.records([c["r"]])[c["r"]]["mistakes"] == 0
    assert mem.g.query("MATCH (n) RETURN count(n)").result_set[0][0] == nodes_before


def test_correct_rules(chain):
    c = chain
    mem = c["mem"]
    secret = mem.remember(c["r"], c["sr"], "private", "org:acme")
    with pytest.raises(MemoryGraphError):
        mem.correct(c["p"], c["sp"], secret, "x")           # cannot see it
    with pytest.raises(MemoryGraphError):
        mem.correct(c["p"], c["sp"], c["d1"], "x")          # a decision, not a fact
    with pytest.raises(MemoryGraphError):
        mem.correct(c["r"], c["sr"], "fact:nope", "x")
    mem.correct(c["r"], c["sr"], c["f1"], "Acme has 50 employees")
    with pytest.raises(MemoryGraphError):
        mem.correct(c["r"], c["sr"], c["f1"], "again")      # already retracted


def test_correction_is_back_dated_so_history_shows_the_right_value(team):
    mem, r, sr, p, sp = team
    now = int(time.time() * 1000)
    old = mem.remember(r, sr, "500 employees", "org:acme", valid_from=now - 10 * DAY)
    mem.publish(r, old)
    new = mem.correct(r, sr, old, "50 employees")["new"]
    assert [f["id"] for f in mem.recall(p, "org:acme", as_of=now - 7 * DAY)] == [new]


def test_a_decision_can_be_redone_on_the_corrected_fact(chain):
    c = chain
    mem = c["mem"]
    new = mem.correct(c["r"], c["sr"], c["f1"], "Acme has 50 employees")["new"]
    redo = mem.decide(c["p"], c["sp"], "Pitch the starter plan", derived_from=[new])
    assert [t["id"] for t in mem.explain(c["p"], redo)] == [redo, new]


# --- supersede after the earlier fix ---

def test_cannot_supersede_a_retracted_fact(chain):
    c = chain
    mem = c["mem"]
    mem.retract(c["r"], c["f1"])
    with pytest.raises(MemoryGraphError):
        mem.supersede(c["r"], c["sr"], c["f1"], "Acme has 50 employees")
    assert status(mem, c["f1"]) == "retracted"  # the "this was wrong" record survives


def test_cannot_supersede_a_fact_that_was_corrected(chain):
    c = chain
    mem = c["mem"]
    mem.correct(c["r"], c["sr"], c["f1"], "Acme has 50 employees")
    with pytest.raises(MemoryGraphError):
        mem.supersede(c["r"], c["sr"], c["f1"], "x")


def test_a_flagged_fact_can_still_be_superseded_by_its_author_to_redo_it(chain):
    c = chain
    mem = c["mem"]
    mem.retract(c["r"], c["f1"])
    assert status(mem, c["f2"]) == "flagged"
    new = mem.supersede(c["p"], c["sp"], c["f2"], "Acme is a small account")["new"]
    assert status(mem, c["f2"]) == "superseded" and status(mem, new) == "active"
