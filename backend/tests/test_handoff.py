"""Handoffs: pass items between agents without sharing anything or leaking private memory."""
import pytest

from memory import MemoryGraphError


@pytest.fixture
def w(mem):
    """Researcher -> Planner (Sam's agents) and Priya's Planner; one shared fact, one private."""
    sr = mem.register_agent("Researcher", owner="Sam")
    sp = mem.register_agent("Planner", owner="Sam")
    pp = mem.register_agent("Planner", owner="Priya")
    s = mem.start_session(sr)
    shared = mem.remember(sr, s, "Acme has 50 staff", entity="org:acme")
    mem.publish(sr, shared)
    private = mem.remember(sr, s, "Acme may be for sale", entity="org:acme")
    return dict(mem=mem, sr=sr, sp=sp, pp=pp, s=s, shared=shared, private=private,
                sp_s=mem.start_session(sp))


def count(mem, label):
    return mem.g.query(f"MATCH (n:{label}) RETURN count(n)").result_set[0][0]


def test_handoff_appears_in_recipients_inbox(w):
    mem = w["mem"]
    h = mem.handoff(w["sr"], w["pp"], [w["shared"]], note="please plan hiring")
    box = mem.inbox(w["pp"])
    assert [b["id"] for b in box] == [h]
    assert box[0]["sender"] == w["sr"] and box[0]["note"] == "please plan hiring"
    assert [i["id"] for i in box[0]["items"]] == [w["shared"]]
    assert mem.inbox(w["sr"]) == []  # the sender's inbox is separate


def test_same_user_may_hand_over_a_private_item_and_it_stays_private(w):
    mem = w["mem"]
    mem.handoff(w["sr"], w["sp"], [w["private"]])
    assert [i["id"] for i in mem.inbox(w["sp"])[0]["items"]] == [w["private"]]
    assert mem.g.query("MATCH (f:Fact {id:$id}) RETURN f.visibility",
                       {"id": w["private"]}).result_set[0][0] == "private"


def test_private_item_to_another_user_is_refused_until_published(w):
    mem = w["mem"]
    with pytest.raises(MemoryGraphError, match="publish them first"):
        mem.handoff(w["sr"], w["pp"], [w["private"]])
    assert count(mem, "Handoff") == 0
    mem.publish(w["sr"], w["private"])
    mem.handoff(w["sr"], w["pp"], [w["private"]])
    assert count(mem, "Handoff") == 1


def test_sender_cannot_hand_over_what_it_cannot_see(w):
    mem = w["mem"]
    with pytest.raises(MemoryGraphError):
        mem.handoff(w["pp"], w["sp"], [w["private"]])  # Priya's agent can't see Sam's private fact


@pytest.mark.parametrize("case", ["empty", "self", "unknown_recipient", "unknown_item"])
def test_bad_requests(w, case):
    mem = w["mem"]
    args = {
        "empty": (w["sr"], w["sp"], []),
        "self": (w["sr"], w["sr"], [w["shared"]]),
        "unknown_recipient": (w["sr"], "agent:nobody", [w["shared"]]),
        "unknown_item": (w["sr"], w["sp"], ["fact:nope"]),
    }[case]
    with pytest.raises(MemoryGraphError):
        mem.handoff(*args)
    assert count(mem, "Handoff") == 0


def test_retracted_item_cannot_be_handed_off(w):
    mem = w["mem"]
    mem.retract(w["sr"], w["shared"])
    with pytest.raises(MemoryGraphError):
        mem.handoff(w["sr"], w["pp"], [w["shared"]])


def test_retract_reports_the_recipient_and_inbox_shows_status(w):
    mem = w["mem"]
    mem.handoff(w["sr"], w["pp"], [w["shared"]])
    report = mem.retract(w["sr"], w["shared"])
    assert w["pp"] in report["agents"]
    assert mem.inbox(w["pp"])[0]["items"][0]["status"] == "retracted"


def test_item_retracted_between_check_and_write_is_not_handed_off(w, monkeypatch):
    mem = w["mem"]
    real, calls = mem._require_agent, []

    def check_then_retract(*a, **k):  # the checks pass, then the item is retracted before the write
        out = real(*a, **k)
        calls.append(a)
        if len(calls) == 2:
            mem.retract(w["sr"], w["shared"])
        return out
    monkeypatch.setattr(mem, "_require_agent", check_then_retract)
    with pytest.raises(MemoryGraphError):
        mem.handoff(w["sr"], w["pp"], [w["shared"]])
    monkeypatch.undo()
    assert count(mem, "Handoff") == 0


def test_a_flagged_item_can_be_handed_to_a_teammate_to_redo(mem):
    r, p = (mem.register_agent(n, owner="Deepak") for n in ("Researcher", "Planner"))
    sr = mem.start_session(r, team=[p])
    fact = mem.remember(r, sr, "Acme HQ is in Austin", "org:acme")
    plan = mem.decide(p, sr, "Visit Austin", derived_from=[fact])
    mem.correct(r, sr, fact, "Acme HQ is in Dallas")
    assert mem.handoff(r, p, [plan], note="redo: HQ is Dallas")         # flagged: a redo request
    with pytest.raises(MemoryGraphError):
        mem.handoff(r, p, [fact])                                       # retracted: never
