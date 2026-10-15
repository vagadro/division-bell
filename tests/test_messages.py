"""Conversation history: a Session is the episode, Messages hang off it, in order, private to the owner."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from memory import MemoryGraphError

N = 20


@pytest.fixture
def w(mem):
    sr = mem.register_agent("Researcher", owner="Sam")
    sp = mem.register_agent("Planner", owner="Sam")
    pr = mem.register_agent("Researcher", owner="Priya")
    return dict(mem=mem, sr=sr, sp=sp, pr=pr, s=mem.start_session(sr), s_pr=mem.start_session(pr))


def test_history_is_in_order(w):
    mem = w["mem"]
    for role, text in [("user", "who owns Acme?"), ("agent", "checking"), ("tool", "PE firm"), ("agent", "a PE firm")]:
        mem.add_message(w["sr"], w["s"], role, text)
    h = mem.history(w["sr"], w["s"])
    assert [m["seq"] for m in h] == [1, 2, 3, 4]
    assert [m["text"] for m in h] == ["who owns Acme?", "checking", "PE firm", "a PE firm"]
    assert [m["role"] for m in h] == ["user", "agent", "tool", "agent"]


def test_history_survives_a_fresh_connection(w):
    from memory import Memory
    w["mem"].add_message(w["sr"], w["s"], "user", "hello")
    again = Memory("memory_test")  # new object, as after a restart
    assert [m["text"] for m in again.history(w["sr"], w["s"])] == ["hello"]


def test_bad_writes_are_refused(w):
    mem = w["mem"]
    with pytest.raises(MemoryGraphError):
        mem.add_message(w["sr"], w["s"], "robot", "x")           # bad role
    with pytest.raises(MemoryGraphError):
        mem.add_message(w["pr"], w["s"], "user", "x")            # not Priya's session
    with pytest.raises(MemoryGraphError):
        mem.add_message(w["sr"], "sess:nope", "user", "x")       # unknown session


def test_only_the_owners_agents_can_read(w):
    mem = w["mem"]
    mem.add_message(w["sr"], w["s"], "user", "secret plan")
    assert len(mem.history(w["sp"], w["s"])) == 1                # Sam's other agent: yes
    with pytest.raises(MemoryGraphError):
        mem.history(w["pr"], w["s"])                             # Priya's agent: no


def test_parallel_appends_get_unique_ordered_numbers(w):
    mem = w["mem"]
    with ThreadPoolExecutor(max_workers=N) as pool:
        list(pool.map(lambda i: mem.add_message(w["sr"], w["s"], "user", f"m{i}"), range(N)))
    assert [m["seq"] for m in mem.history(w["sr"], w["s"])] == list(range(1, N + 1))


def test_fact_can_cite_messages_and_others_cannot_read_them(w):
    mem = w["mem"]
    m1 = mem.add_message(w["sr"], w["s"], "user", "Acme is owned by a PE firm")
    f = mem.remember(w["sr"], w["s"], "Acme is PE owned", entity="org:acme", from_messages=[m1])
    mem.publish(w["sr"], f)
    mine = mem.evidence(w["sr"], f)
    assert [(e["message"], e["text"]) for e in mine] == [(m1, "Acme is owned by a PE firm")]
    theirs = mem.evidence(w["pr"], f)                            # visible fact, unreadable message
    assert [(e["message"], e["text"]) for e in theirs] == [(m1, None)]


def test_cannot_cite_unreadable_or_unknown_messages(w):
    mem = w["mem"]
    m1 = mem.add_message(w["sr"], w["s"], "user", "private chat")
    with pytest.raises(MemoryGraphError):
        mem.remember(w["pr"], w["s_pr"], "x", entity="org:acme", from_messages=[m1])
    with pytest.raises(MemoryGraphError):
        mem.remember(w["sr"], w["s"], "x", entity="org:acme", from_messages=["msg:nope"])
    assert mem.g.query("MATCH (f:Fact) RETURN count(f)").result_set[0][0] == 0


def test_messages_are_not_lineage(w):
    """Retracting a fact never reaches messages, and messages are not in anyone's blast radius."""
    mem = w["mem"]
    m1 = mem.add_message(w["sr"], w["s"], "user", "Acme is owned by a PE firm")
    f = mem.remember(w["sr"], w["s"], "Acme is PE owned", entity="org:acme", from_messages=[m1])
    mem.publish(w["sr"], f)
    d = mem.decide(w["sr"], w["s"], "approach as a PE-owned firm", derived_from=[f])
    assert [b["id"] for b in mem.blast_radius(w["sr"], f)] == [d]
    mem.retract(w["sr"], f)
    assert mem.g.query("MATCH (m:Message) RETURN m.status").result_set == [[None]]
    assert [m["text"] for m in mem.history(w["sr"], w["s"])] == ["Acme is owned by a PE firm"]
    assert [e["message"] for e in mem.evidence(w["sr"], f)] == [m1]  # still shows where it came from


def test_evidence_follows_the_lineage(w):
    mem = w["mem"]
    m1 = mem.add_message(w["sr"], w["s"], "user", "Acme has 50 staff")
    f1 = mem.remember(w["sr"], w["s"], "50 staff", entity="org:acme", from_messages=[m1])
    f2 = mem.remember(w["sr"], w["s"], "needs 2 recruiters", entity="org:acme", derived_from=[f1])
    assert [e["message"] for e in mem.evidence(w["sr"], f2)] == [m1]
