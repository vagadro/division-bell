"""Tests for memory.py against a throwaway graph. No LLM involved, so they cost nothing."""
import time

import pytest

from memory import MemoryGraphError

DAY = 24 * 3600 * 1000


def ids(facts):
    return sorted(f["id"] for f in facts)


def count(mem, query):
    return mem.g.query(query).result_set[0][0]


def test_setup_is_idempotent(mem):
    mem.setup()  # second call must not raise


def test_register_agent_is_idempotent_and_session_needs_agent(mem):
    assert mem.register_agent("Research Bot") == "agent:research-bot"
    assert mem.register_agent("Research Bot") == "agent:research-bot"
    assert count(mem, "MATCH (a:Agent) RETURN count(a)") == 1
    with pytest.raises(MemoryGraphError):
        mem.start_session("agent:ghost")


def test_fact_is_private_until_published(team):
    mem, r, sr, p, sp = team
    fid = mem.remember(r, sr, "Acme has 50 employees", "org:acme", entity_name="Acme Corp",
                       entity_type="org", confidence=0.8)
    assert ids(mem.recall(r, "org:acme")) == [fid]
    assert mem.recall(p, "org:acme") == []
    mem.publish(r, fid)
    assert ids(mem.recall(p, "org:acme")) == [fid]


def test_only_the_author_can_publish(team):
    mem, r, sr, p, sp = team
    fid = mem.remember(r, sr, "x", "org:acme")
    with pytest.raises(MemoryGraphError):
        mem.publish(p, fid)
    with pytest.raises(MemoryGraphError):
        mem.publish(r, "fact:nope")


def test_cannot_build_on_what_you_cannot_see(team):
    mem, r, sr, p, sp = team
    secret = mem.remember(r, sr, "Researcher's private note", "org:acme")
    with pytest.raises(MemoryGraphError):
        mem.remember(p, sp, "built on it", "org:acme", derived_from=[secret])
    with pytest.raises(MemoryGraphError):
        mem.remember(p, sp, "built on nothing", "org:acme", derived_from=["fact:nope"])
    assert mem.recall(p, "org:acme") == []  # nothing was written
    mem.publish(r, secret)
    built = mem.remember(p, sp, "built on it", "org:acme", derived_from=[secret])
    assert {f["id"]: f["built_on"] for f in mem.recall(p, "org:acme")}[built] == [secret]


def test_cannot_publish_while_built_on_a_private_fact(team):
    mem, r, sr, p, sp = team
    base = mem.remember(p, sp, "base", "org:acme")
    derived = mem.remember(p, sp, "derived", "org:acme", derived_from=[base])
    with pytest.raises(MemoryGraphError):
        mem.publish(p, derived)
    mem.publish(p, base)
    mem.publish(p, derived)
    assert ids(mem.recall(r, "org:acme")) == sorted([base, derived])


def test_unknown_agent_or_foreign_session_writes_nothing(team):
    mem, r, sr, p, sp = team
    before = count(mem, "MATCH (n) RETURN count(n)")
    with pytest.raises(MemoryGraphError):
        mem.remember("agent:ghost", sr, "x", "org:acme")
    with pytest.raises(MemoryGraphError):
        mem.remember(p, sr, "x", "org:acme")  # Planner using the Researcher's session
    assert count(mem, "MATCH (n) RETURN count(n)") == before


def test_supersede_keeps_history_and_supports_as_of(team):
    mem, r, sr, p, sp = team
    now = int(time.time() * 1000)
    old = mem.remember(r, sr, "HQ is in Austin", "org:acme", valid_from=now - 10 * DAY)
    mem.publish(r, old)
    new = mem.supersede(r, sr, old, "HQ is in Dallas", at=now - 5 * DAY)["new"]
    assert ids(mem.recall(r, "org:acme")) == [new]                    # now
    assert ids(mem.recall(r, "org:acme", as_of=now - 7 * DAY)) == [old]  # before the move
    assert ids(mem.recall(r, "org:acme", as_of=now - 1 * DAY)) == [new]  # after the move
    chain = mem.g.query("MATCH (:Fact {id:$n})-[:SUPERSEDES]->(o) RETURN o.id", {"n": new}).result_set
    assert [row[0] for row in chain] == [old]
    with pytest.raises(MemoryGraphError):
        mem.supersede(r, sr, old, "again", at=now)  # old is already closed


def test_supersede_needs_a_visible_fact(team):
    mem, r, sr, p, sp = team
    secret = mem.remember(r, sr, "private", "org:acme")
    with pytest.raises(MemoryGraphError):
        mem.supersede(p, sp, secret, "hijack")


def test_new_fact_inherits_visibility_and_entity(team):
    mem, r, sr, p, sp = team
    old = mem.remember(r, sr, "HQ is in Austin", "org:acme")  # private
    new = mem.supersede(r, sr, old, "HQ is in Dallas")["new"]
    assert ids(mem.recall(r, "org:acme")) == [new]
    assert mem.recall(p, "org:acme") == []  # still private


def test_one_fact_about_several_entities(team):
    # "I work at Google Delhi" is about the user AND Google: one node, two ABOUT edges, so a move
    # leaves exactly one old (superseded) and one new (active) fact, found from either entity
    mem, r, sr, p, sp = team
    old = mem.remember(r, sr, "Sam works at Google Delhi", "user:sam",
                       also_about=["org:google", "org:google", "user:sam"])  # repeats are ignored
    assert ids(mem.recall(r, "user:sam")) == ids(mem.recall(r, "org:google")) == [old]
    new = mem.supersede(r, sr, old, "Sam works at Google Bangalore")["new"]
    assert ids(mem.recall(r, "user:sam")) == ids(mem.recall(r, "org:google")) == [new]
    rows = mem.g.query("MATCH (f:Fact) RETURN f.status ORDER BY f.status").result_set
    assert [row[0] for row in rows] == ["active", "superseded"]
    about = mem.g.query("MATCH (:Fact {id:$n})-[:ABOUT]->(e) RETURN e.id ORDER BY e.id", {"n": new}).result_set
    assert [row[0] for row in about] == ["org:google", "user:sam"]


def test_explain_walks_back_to_the_root_with_agents(team):
    mem, r, sr, p, sp = team
    f1 = mem.remember(r, sr, "root fact", "org:acme")
    mem.publish(r, f1)
    f2 = mem.remember(p, sp, "derived fact", "org:acme", derived_from=[f1])
    chain = mem.explain(p, f2)
    assert [(c["id"], c["agent"], c["depth"]) for c in chain] == [(f2, p, 0), (f1, r, 1)]


def test_explain_respects_visibility(team):
    mem, r, sr, p, sp = team
    secret = mem.remember(r, sr, "private", "org:acme")
    with pytest.raises(MemoryGraphError):
        mem.explain(p, secret)
    with pytest.raises(MemoryGraphError):
        mem.explain(p, "fact:nope")
