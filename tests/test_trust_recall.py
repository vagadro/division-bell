"""Trust in use: recall reports each fact's author trust, sorts by it, and warns when it is low."""
import pytest

from memory import MemoryGraphError, TRUST_BOOST, TRUST_PENALTY, TRUST_VERIFY


def shared_fact(mem, agent, session, text):
    f = mem.remember(agent, session, text, "org:acme")
    mem.publish(agent, f)
    return f


def make_unreliable(mem, agent, session, mistakes):
    """Retract `mistakes` of the agent's own facts, lowering its trust."""
    for i in range(mistakes):
        mem.retract(agent, mem.remember(agent, session, f"bad {i}", "org:bad"))


def test_recall_returns_author_and_trust(team):
    mem, r, sr, p, sp = team
    f = shared_fact(mem, r, sr, "Acme has 500 employees")
    [row] = mem.recall(p, "org:acme")
    assert row["id"] == f and row["author"] == r
    assert row["trust"] == pytest.approx(1.0)
    assert row["verified"] is False and row["needs_verification"] is False


def test_most_trusted_author_comes_first_and_nothing_is_hidden(team):
    mem, r, sr, p, sp = team
    mem_w = mem.register_agent("Writer")
    sw = mem.start_session(mem_w)
    make_unreliable(mem, r, sr, 1)                       # Researcher: 0.9
    low = shared_fact(mem, r, sr, "Acme was founded in 1999")
    high = shared_fact(mem, mem_w, sw, "Acme was founded in 2001")  # Writer: 1.0
    assert [x["id"] for x in mem.recall(p, "org:acme")] == [high, low]


def test_warning_starts_below_threshold(team):
    mem, r, sr, p, sp = team
    make_unreliable(mem, r, sr, 2)                       # 0.81: still above 0.75
    f = shared_fact(mem, r, sr, "Acme has 500 employees")
    assert mem.trust(r) == pytest.approx(TRUST_PENALTY ** 2) and mem.trust(r) > TRUST_VERIFY
    assert mem.recall(p, "org:acme")[0]["needs_verification"] is False
    make_unreliable(mem, r, sr, 1)                       # 0.729: below 0.75
    [row] = mem.recall(p, "org:acme")
    assert row["id"] == f and row["needs_verification"] is True


def test_warning_applies_to_old_facts_too(team):
    mem, r, sr, p, sp = team
    old = shared_fact(mem, r, sr, "Acme has 500 employees")   # written while trust was 1.0
    make_unreliable(mem, r, sr, 3)
    assert mem.recall(p, "org:acme")[0]["id"] == old
    assert mem.recall(p, "org:acme")[0]["needs_verification"] is True


def check(mem, agent, fact, result="supported", method="web", sources=("https://example.com",)):
    return mem.verify(agent, fact, method, result, list(sources), "quoted evidence")


def test_a_supported_check_clears_the_warning(team):
    mem, r, sr, p, sp = team
    f = shared_fact(mem, r, sr, "Acme has 500 employees")
    make_unreliable(mem, r, sr, 3)
    assert mem.recall(p, "org:acme")[0]["needs_verification"] is True
    out = check(mem, p, f)
    assert out["trust_raised"] is True and out["trust"] == pytest.approx(TRUST_PENALTY ** 3 * TRUST_BOOST)
    [row] = mem.recall(p, "org:acme")
    assert row["verified"] is True and row["needs_verification"] is False


def test_contradicted_or_unclear_changes_nothing(team):
    mem, r, sr, p, sp = team
    f = shared_fact(mem, r, sr, "Acme has 500 employees")
    make_unreliable(mem, r, sr, 3)
    before = mem.trust(r)
    for result in ("contradicted", "unclear"):
        assert check(mem, p, f, result=result)["trust_raised"] is False
    assert mem.trust(r) == pytest.approx(before)
    [row] = mem.recall(p, "org:acme")
    assert row["verified"] is False and row["needs_verification"] is True


def test_only_the_first_supported_check_raises_trust(team):
    mem, r, sr, p, sp = team
    f = shared_fact(mem, r, sr, "Acme has 500 employees")
    make_unreliable(mem, r, sr, 1)
    assert check(mem, p, f)["trust_raised"] is True
    after_first = mem.trust(r)
    assert check(mem, p, f, method="doc", sources=["handbook.pdf"])["trust_raised"] is False
    assert mem.trust(r) == pytest.approx(after_first)


def test_trust_is_capped_at_one(team):
    mem, r, sr, p, sp = team
    f = shared_fact(mem, r, sr, "Acme has 500 employees")
    assert check(mem, p, f)["trust"] == pytest.approx(1.0)


def test_evidence_and_valid_values_are_required(team):
    mem, r, sr, p, sp = team
    f = shared_fact(mem, r, sr, "Acme has 500 employees")
    for bad in (dict(sources=[]), dict(sources=["  "]), dict(method="peer"), dict(result="true")):
        with pytest.raises(MemoryGraphError):
            check(mem, p, f, **bad)
    assert mem.trust(r) == pytest.approx(1.0)


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


def test_recall_as_of_also_reports_trust(team):
    mem, r, sr, p, sp = team
    f = shared_fact(mem, r, sr, "Acme has 500 employees")
    make_unreliable(mem, r, sr, 3)
    import time
    [row] = mem.recall(p, "org:acme", as_of=int(time.time() * 1000))
    assert row["id"] == f and row["needs_verification"] is True
