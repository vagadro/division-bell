"""Trust in use: judged per fact first (evidence), then by the author's accuracy (a rate, not a
counter). Facts the user told an agent never count for or against it."""
import time

import pytest

from memory import ACCURACY_WARN, PRIOR_CORRECT, MemoryGraphError


def expected(claims, mistakes):
    return (claims - mistakes + PRIOR_CORRECT) / (claims + PRIOR_CORRECT)


def shared_fact(mem, agent, session, text, source="agent"):
    f = mem.remember(agent, session, text, "org:acme", source=source)
    mem.publish(agent, f)
    return f


def claims(mem, agent, session, good=0, bad=0):
    """Give the agent `good` correct claims and `bad` claims that were retracted (its mistakes)."""
    for i in range(good):
        mem.remember(agent, session, f"good {i}", "org:other")
    for i in range(bad):
        mem.retract(agent, mem.remember(agent, session, f"bad {i}", "org:bad"))


def check(mem, agent, fact, result="supported", method="web", sources=("https://example.com",)):
    return mem.verify(agent, fact, method, result, list(sources), "quoted evidence")


def row(mem, agent, fact, entity="org:acme"):
    return {f["id"]: f for f in mem.recall(agent, entity)}[fact]


# --- accuracy: a rate, not a counter ---

def test_a_new_agent_starts_at_full_accuracy(team):
    mem, r, sr, p, sp = team
    assert mem.accuracy(r) == 1.0
    assert mem.records([r])[r] == {"claims": 0, "mistakes": 0, "accuracy": 1.0}


def test_accuracy_is_a_rate_so_a_few_mistakes_among_many_claims_barely_count(team):
    mem, r, sr, p, sp = team
    w = mem.register_agent("Writer")
    sw = mem.start_session(w)
    claims(mem, r, sr, good=20, bad=3)            # 3 wrong in 23
    claims(mem, w, sw, bad=3)                     # 3 wrong in 3
    assert mem.accuracy(r) == pytest.approx(expected(23, 3)) and mem.accuracy(r) > ACCURACY_WARN
    assert mem.accuracy(w) == pytest.approx(expected(3, 3)) and mem.accuracy(w) < ACCURACY_WARN


def test_superseded_facts_are_not_mistakes(team):
    mem, r, sr, p, sp = team
    f = mem.remember(r, sr, "HQ in Austin", "org:acme")
    mem.supersede(r, sr, f, "HQ in Dallas")
    assert mem.records([r])[r]["mistakes"] == 0


# --- standing: the fact first, the author only as a fallback ---

def test_recall_returns_source_standing_and_author_accuracy(team):
    mem, r, sr, p, sp = team
    f = shared_fact(mem, r, sr, "Acme has 500 employees")
    got = row(mem, p, f)
    assert got["author"] == r and got["source"] == "agent" and got["author_accuracy"] == 1.0
    assert got["standing"] == "unverified" and got["needs_verification"] is False


def test_an_unchecked_fact_is_doubtful_only_when_its_author_is_often_wrong(team):
    mem, r, sr, p, sp = team
    f = shared_fact(mem, r, sr, "Acme has 500 employees")
    claims(mem, r, sr, bad=1)                     # 1 wrong in 2: accuracy 0.83
    assert row(mem, p, f)["standing"] == "unverified"
    claims(mem, r, sr, bad=1)                     # 2 wrong in 3: accuracy 0.71
    got = row(mem, p, f)
    assert got["standing"] == "doubtful" and got["needs_verification"] is True


def test_a_supported_check_makes_a_fact_verified_whoever_wrote_it(team):
    mem, r, sr, p, sp = team
    f = shared_fact(mem, r, sr, "Acme has 500 employees")
    claims(mem, r, sr, bad=3)
    assert row(mem, p, f)["standing"] == "doubtful"
    before = mem.accuracy(r)
    assert check(mem, p, f) == {"result": "supported"}
    got = row(mem, p, f)
    assert got["standing"] == "verified" and got["verified"] is True and got["needs_verification"] is False
    assert mem.accuracy(r) == before              # evidence is about the fact, not a score for the agent


def test_a_contradicted_fact_is_doubtful_even_from_an_accurate_author(team):
    mem, r, sr, p, sp = team
    f = shared_fact(mem, r, sr, "Acme has 500 employees")
    check(mem, p, f, result="unclear")
    assert row(mem, p, f)["standing"] == "unverified"         # unclear changes nothing
    check(mem, p, f, result="contradicted")
    assert row(mem, p, f)["standing"] == "doubtful"
    assert mem.records([r])[r]["mistakes"] == 0               # only retracting is a mistake


def test_the_users_own_words_never_count_against_the_agent(team):
    mem, r, sr, p, sp = team
    job = mem.remember(r, sr, "Deepak works at Apple", "user:deepak", source="user")
    job = mem.correct(r, sr, job, "Deepak works at Microsoft")["new"]   # the user changed their story
    job = mem.correct(r, sr, job, "Deepak works at Google")["new"]      # ...twice
    assert mem.records([r])[r] == {"claims": 0, "mistakes": 0, "accuracy": 1.0}
    got = row(mem, r, job, "user:deepak")
    assert got["source"] == "user" and got["standing"] == "user"        # the source carries over
    claims(mem, r, sr, bad=3)                                           # even an unreliable author...
    assert row(mem, r, job, "user:deepak")["needs_verification"] is False   # ...can't make it doubtful


def test_nothing_is_hidden_and_the_best_standing_comes_first(team):
    mem, r, sr, p, sp = team
    w = mem.register_agent("Writer")
    sw = mem.start_session(w)
    claims(mem, w, sw, bad=3)                                           # the Writer is often wrong
    doubtful = shared_fact(mem, w, sw, "Acme was founded in 1999")
    unverified = shared_fact(mem, r, sr, "Acme was founded in 2001")
    verified = shared_fact(mem, w, sw, "Acme is based in Austin")
    check(mem, p, verified)
    user = shared_fact(mem, r, sr, "Acme is our client", source="user")
    assert [f["id"] for f in mem.recall(p, "org:acme")] == [verified, user, unverified, doubtful]


def test_recall_as_of_also_reports_standing(team):
    mem, r, sr, p, sp = team
    f = shared_fact(mem, r, sr, "Acme has 500 employees")
    claims(mem, r, sr, bad=3)
    [got] = mem.recall(p, "org:acme", as_of=int(time.time() * 1000))
    assert got["id"] == f and got["standing"] == "doubtful"


# --- checks need evidence and a fact the checker may see ---

def test_evidence_and_valid_values_are_required(team):
    mem, r, sr, p, sp = team
    f = shared_fact(mem, r, sr, "Acme has 500 employees")
    for bad in (dict(sources=[]), dict(sources=["  "]), dict(method="peer"), dict(result="true")):
        with pytest.raises(MemoryGraphError):
            check(mem, p, f, **bad)
    with pytest.raises(MemoryGraphError):
        mem.remember(r, sr, "x", "org:acme", source="rumour")


def test_cannot_verify_what_you_cannot_see_or_inactive_facts(team):
    mem, r, sr, p, sp = team
    private = mem.remember(r, sr, "Acme secret", "org:acme")
    with pytest.raises(MemoryGraphError):
        check(mem, p, private)                     # private to another user
    check(mem, r, private)                         # its own author may check it
    gone = mem.remember(r, sr, "wrong", "org:acme")
    mem.retract(r, gone)
    with pytest.raises(MemoryGraphError):
        check(mem, p, gone)                        # retracted
    with pytest.raises(MemoryGraphError):
        check(mem, p, "fact:nope")
