"""Ownership tests: Sam's agents share Sam's memory; Priya's agents can't touch it."""
import pytest

from memory import MemoryGraphError, TRUST_PENALTY


@pytest.fixture
def world(mem):
    sam_r = mem.register_agent("Researcher", owner="Sam")
    sam_p = mem.register_agent("Planner", owner="Sam")
    priya_r = mem.register_agent("Researcher", owner="Priya")
    return dict(
        mem=mem, sam_r=sam_r, sam_p=sam_p, priya_r=priya_r,
        s_sr=mem.start_session(sam_r), s_sp=mem.start_session(sam_p), s_pr=mem.start_session(priya_r))


def ids(facts):
    return sorted(f["id"] for f in facts)


def status(mem, item_id):
    return mem.g.query("MATCH (x {id:$id}) RETURN x.status", {"id": item_id}).result_set[0][0]


def test_same_role_for_two_users_are_different_agents(world):
    w = world
    assert w["sam_r"] == "agent:sam-researcher" and w["priya_r"] == "agent:priya-researcher"
    assert w["mem"].register_agent("Researcher", owner="Sam") == w["sam_r"]  # idempotent
    assert w["mem"].register_agent("Researcher") == "agent:researcher"       # no owner: owns itself


def test_a_users_agents_share_its_private_memory_other_users_do_not(world):
    w = world
    mem = w["mem"]
    note = mem.remember(w["sam_r"], w["s_sr"], "Sam's private note", "org:acme")
    assert ids(mem.recall(w["sam_p"], "org:acme")) == [note]       # Sam's other agent sees it
    assert mem.recall(w["priya_r"], "org:acme") == []              # Priya's agent does not
    with pytest.raises(MemoryGraphError):
        mem.explain(w["priya_r"], note)


def test_a_users_agent_can_build_on_its_siblings_private_note_others_cannot(world):
    w = world
    mem = w["mem"]
    note = mem.remember(w["sam_r"], w["s_sr"], "Sam's private note", "org:acme")
    built = mem.remember(w["sam_p"], w["s_sp"], "built on it", "org:acme", derived_from=[note])
    assert {f["id"]: f["built_on"] for f in mem.recall(w["sam_p"], "org:acme")}[built] == [note]
    with pytest.raises(MemoryGraphError):
        mem.remember(w["priya_r"], w["s_pr"], "built on Sam's note", "org:acme", derived_from=[note])


def test_a_users_agent_can_publish_its_siblings_note_others_cannot(world):
    w = world
    mem = w["mem"]
    note = mem.remember(w["sam_r"], w["s_sr"], "Sam's note", "org:acme")
    with pytest.raises(MemoryGraphError):
        mem.publish(w["priya_r"], note)
    mem.publish(w["sam_p"], note)
    assert ids(mem.recall(w["priya_r"], "org:acme")) == [note]  # now shared with everyone


def test_a_users_agent_can_change_its_siblings_fact_and_trust_goes_to_the_author(world):
    w = world
    mem = w["mem"]
    f = mem.remember(w["sam_r"], w["s_sr"], "Acme has 500 employees", "org:acme")
    out = mem.correct(w["sam_p"], w["s_sp"], f, "Acme has 50 employees")  # the Planner corrects it
    assert status(mem, f) == "retracted" and status(mem, out["new"]) == "active"
    assert mem.trust(w["sam_r"]) == pytest.approx(TRUST_PENALTY)  # the agent that wrote the bad fact
    assert mem.trust(w["sam_p"]) == 1.0
    g = mem.remember(w["sam_r"], w["s_sr"], "HQ is in Austin", "org:acme")
    assert mem.supersede(w["sam_p"], w["s_sp"], g, "HQ is in Dallas")["new"]


def test_another_users_agent_cannot_change_a_shared_fact_of_sam(world):
    w = world
    mem = w["mem"]
    f = mem.remember(w["sam_r"], w["s_sr"], "Acme has 500 employees", "org:acme")
    mem.publish(w["sam_r"], f)
    before = mem.g.query("MATCH (n) RETURN count(n)").result_set[0][0]
    for action in (
        lambda: mem.supersede(w["priya_r"], w["s_pr"], f, "x"),
        lambda: mem.correct(w["priya_r"], w["s_pr"], f, "x"),
        lambda: mem.retract(w["priya_r"], f),
    ):
        with pytest.raises(MemoryGraphError, match="only the author"):
            action()
    assert status(mem, f) == "active" and mem.trust(w["sam_r"]) == 1.0
    assert mem.g.query("MATCH (n) RETURN count(n)").result_set[0][0] == before
    assert f in ids(mem.recall(w["sam_p"], "org:acme"))  # Sam's memory is unchanged


def test_priyas_agent_can_still_see_and_build_on_sams_shared_fact(world):
    w = world
    mem = w["mem"]
    f = mem.remember(w["sam_r"], w["s_sr"], "Acme has 500 employees", "org:acme")
    mem.publish(w["sam_r"], f)
    mine = mem.remember(w["priya_r"], w["s_pr"], "so Acme is big", "org:acme", derived_from=[f])
    out = mem.retract(w["sam_p"], f)  # Sam's side finds out it was wrong
    assert status(mem, mine) == "flagged"
    assert out["hidden_flagged"] == 1                     # Priya's private note: Sam only sees a count
    assert [x["id"] for x in mem.flagged(w["priya_r"])] == [mine]  # Priya finds out
    assert mem.flagged(w["sam_r"]) == []                           # and Sam's agents don't see her note


def test_flagged_items_are_visible_to_all_of_the_owners_agents(world):
    w = world
    mem = w["mem"]
    f = mem.remember(w["sam_r"], w["s_sr"], "base", "org:acme")
    mem.publish(w["sam_r"], f)
    d = mem.decide(w["sam_p"], w["s_sp"], "decision on base", derived_from=[f])
    mem.supersede(w["sam_r"], w["s_sr"], f, "base v2")
    assert [x["id"] for x in mem.flagged(w["sam_r"])] == [d]   # the Researcher also sees the Planner's item
    assert mem.flagged(w["priya_r"]) == []
    mem.confirm(w["sam_r"], d)                                  # and may confirm it for Sam
    assert status(mem, d) == "active"


def test_blast_radius_shows_the_owners_private_items_but_not_other_users(world):
    w = world
    mem = w["mem"]
    f = mem.remember(w["sam_r"], w["s_sr"], "base", "org:acme")
    mem.publish(w["sam_r"], f)
    sam_private = mem.remember(w["sam_p"], w["s_sp"], "Sam's private idea", "org:acme", derived_from=[f])
    assert [x["id"] for x in mem.blast_radius(w["sam_r"], f)] == [sam_private]
    assert mem.blast_radius(w["priya_r"], f) == []
