"""Procedures: saved how-tos that are derived from facts, flagged, confirmed and versioned like facts."""
import pytest

from memory import MemoryGraphError

STEPS = ["find the fund", "contact its portfolio team"]


@pytest.fixture
def w(mem):
    sr = mem.register_agent("Researcher", owner="Sam")
    sp = mem.register_agent("Planner", owner="Sam")
    pr = mem.register_agent("Researcher", owner="Priya")
    s_sr, s_sp, s_pr = (mem.start_session(a) for a in (sr, sp, pr))
    fact = mem.remember(sr, s_sr, "Acme is PE owned", entity="org:acme")
    mem.publish(sr, fact)
    return dict(mem=mem, sr=sr, sp=sp, pr=pr, s_sr=s_sr, s_sp=s_sp, s_pr=s_pr, fact=fact)


def status(mem, item_id):
    return mem.g.query("MATCH (x {id:$id}) RETURN x.status", {"id": item_id}).result_set[0][0]


def learn(w, **kw):
    return w["mem"].learn_procedure(w["sr"], w["s_sr"], "PE outreach", STEPS, "task:outreach",
                                    derived_from=[w["fact"]], **kw)


def test_learn_and_recall_with_visibility(w):
    mem = w["mem"]
    p = learn(w)
    assert mem.recall_procedures(w["sr"], "task:outreach") == [
        {"id": p, "name": "PE outreach", "steps": STEPS, "built_on": [w["fact"]]}]
    assert [x["id"] for x in mem.recall_procedures(w["sp"], "task:outreach")] == [p]  # same user
    assert mem.recall_procedures(w["pr"], "task:outreach") == []                       # private
    mem.publish(w["sr"], p)
    assert [x["id"] for x in mem.recall_procedures(w["pr"], "task:outreach")] == [p]


def test_bad_input_is_refused(w):
    mem = w["mem"]
    for steps in ([], [""], ["ok", "  "]):
        with pytest.raises(MemoryGraphError):
            mem.learn_procedure(w["sr"], w["s_sr"], "x", steps, "task:outreach")
    with pytest.raises(MemoryGraphError):
        mem.learn_procedure(w["sr"], w["s_sr"], " ", STEPS, "task:outreach")
    with pytest.raises(MemoryGraphError):
        mem.learn_procedure(w["sr"], w["s_pr"], "x", STEPS, "task:outreach")      # not its session
    with pytest.raises(MemoryGraphError):
        mem.learn_procedure(w["sr"], w["s_sr"], "x", STEPS, "task:outreach", derived_from=["fact:nope"])
    assert mem.g.query("MATCH (p:Procedure) RETURN count(p)").result_set[0][0] == 0


def test_publish_needs_shared_sources(w):
    mem = w["mem"]
    private_fact = mem.remember(w["sr"], w["s_sr"], "Sam's hunch", entity="org:acme")
    p = mem.learn_procedure(w["sr"], w["s_sr"], "hunch-based", STEPS, "task:outreach",
                            derived_from=[private_fact])
    with pytest.raises(MemoryGraphError):
        mem.publish(w["sr"], p)
    mem.publish(w["sr"], private_fact)
    mem.publish(w["sr"], p)


def test_nothing_can_be_built_on_a_procedure(w):
    mem = w["mem"]
    p = learn(w)
    mem.publish(w["sr"], p)
    with pytest.raises(MemoryGraphError):
        mem.remember(w["sr"], w["s_sr"], "x", entity="org:acme", derived_from=[p])
    with pytest.raises(MemoryGraphError):
        mem.decide(w["sr"], w["s_sr"], "x", derived_from=[p])


def test_retracting_the_source_flags_the_procedure_as_wrong(w):
    mem = w["mem"]
    p = learn(w)
    report = mem.retract(w["sr"], w["fact"])
    assert report["flagged"] == [p] and status(mem, p) == "flagged"
    assert mem.recall_procedures(w["sr"], "task:outreach") == []
    assert [(f["id"], f["wrong"]) for f in mem.flagged(w["sr"])] == [(p, [w["fact"]])]
    with pytest.raises(MemoryGraphError):
        mem.confirm(w["sr"], p)                                  # wrong sources: must be redone


def test_superseding_the_source_flags_changed_and_confirm_restores(w):
    mem = w["mem"]
    p = learn(w)
    new_fact = mem.supersede(w["sr"], w["s_sr"], w["fact"], "Acme is PE owned by a new fund")["new"]
    assert status(mem, p) == "flagged"
    assert mem.flagged(w["sr"])[0]["changed"] == [{"fact": w["fact"], "now": new_fact}]
    mem.confirm(w["sr"], p)
    assert status(mem, p) == "active"
    assert [x["id"] for x in mem.recall_procedures(w["sr"], "task:outreach")] == [p]


def test_revise_makes_a_new_version_and_redoes_a_flagged_one(w):
    mem = w["mem"]
    p = learn(w)
    mem.publish(w["sr"], p)
    mem.retract(w["sr"], w["fact"])
    good = mem.remember(w["sr"], w["s_sr"], "Acme is founder owned", entity="org:acme")
    mem.publish(w["sr"], good)
    p2 = mem.revise_procedure(w["sr"], w["s_sr"], p, ["call the founder"], derived_from=[good])
    assert status(mem, p) == "superseded" and status(mem, p2) == "active"
    got = mem.recall_procedures(w["pr"], "task:outreach")        # inherited: shared, about the entity
    assert got == [{"id": p2, "name": "PE outreach", "steps": ["call the founder"], "built_on": [good]}]
    assert [t["id"] for t in mem.explain(w["sr"], p2)] == [p2, good]


def test_shared_procedure_cannot_be_revised_onto_private_sources(w):
    mem = w["mem"]
    p = learn(w)
    mem.publish(w["sr"], p)
    private_fact = mem.remember(w["sr"], w["s_sr"], "hunch", entity="org:acme")
    with pytest.raises(MemoryGraphError):
        mem.revise_procedure(w["sr"], w["s_sr"], p, STEPS, derived_from=[private_fact])
    assert status(mem, p) == "active"


def test_only_the_authors_user_can_change_it(w):
    mem = w["mem"]
    p = learn(w)
    mem.publish(w["sr"], p)
    for call in (lambda: mem.revise_procedure(w["pr"], w["s_pr"], p, ["x"]),
                 lambda: mem.retire_procedure(w["pr"], p),
                 lambda: mem.publish(w["pr"], p)):
        with pytest.raises(MemoryGraphError):
            call()
    p2 = mem.revise_procedure(w["sp"], w["s_sp"], p, ["x"])      # Sam's other agent may
    assert status(mem, p2) == "active"


def test_retire_ends_it_and_is_no_mistake(w):
    mem = w["mem"]
    p = learn(w)
    mem.retire_procedure(w["sr"], p, "obsolete")
    assert status(mem, p) == "retired" and mem.records([w["sr"]])[w["sr"]]["mistakes"] == 0
    assert mem.recall_procedures(w["sr"], "task:outreach") == []
    with pytest.raises(MemoryGraphError):
        mem.retire_procedure(w["sr"], p)
    with pytest.raises(MemoryGraphError):
        mem.revise_procedure(w["sr"], w["s_sr"], p, ["x"])


def test_blast_radius_lists_the_procedure(w):
    mem = w["mem"]
    p = learn(w)
    assert [b["id"] for b in mem.blast_radius(w["sr"], w["fact"])] == [p]


def test_source_retracted_between_check_and_write(w, monkeypatch):
    mem = w["mem"]
    real = mem._require_sources

    def check_then_retract(*a, **k):
        real(*a, **k)
        mem.retract(w["sr"], w["fact"])
    monkeypatch.setattr(mem, "_require_sources", check_then_retract)
    with pytest.raises(MemoryGraphError):
        learn(w)
    monkeypatch.undo()
    assert mem.g.query("MATCH (p:Procedure) RETURN count(p)").result_set[0][0] == 0
