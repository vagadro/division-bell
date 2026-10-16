"""Automatic upkeep: copies are caught on save, web checks share public facts and dispute other
users' facts, and flagged work is put in front of the agents at the start of a turn."""
import pytest

from llm import _attention
from memory import MemoryGraphError
from tools import AgentTools

URL = "https://example.com/source"


@pytest.fixture
def deepak(mem):
    t = AgentTools.open(mem, "Researcher", owner="Deepak")
    t.web_urls = {URL}  # as if this turn's web search returned it
    return t


@pytest.fixture
def priya(mem):
    t = AgentTools.open(mem, "Researcher", owner="Priya")
    t.web_urls = {URL}
    return t


def ids(facts):
    return [f["id"] for f in facts]


# --- copies are caught when saving ---

def test_a_copy_under_another_entity_is_caught(deepak):
    first = deepak.call("remember", {"text": "Deepak works with Google at the New Delhi office",
                                     "entity": "user:deepak"})["id"]
    out = deepak.call("remember", {"text": "Deepak's workplace is the Google New Delhi office",
                                   "entity": "org:google"})
    assert out["saved"] is False and ids(out["similar"]) == [first] and out["similar"][0]["yours"]
    unrelated = deepak.call("remember", {"text": "Google India was founded in 2003", "entity": "org:google"})
    assert "id" in unrelated


def test_a_newer_number_is_caught_and_distinct_from_lets_a_different_claim_through(deepak):
    old = deepak.call("save_web_fact", {"text": "Alphabet, Google's parent company, had 190,820 employees "
                                        "as of December 2025", "entity": "org:google", "sources": [URL]})["id"]
    out = deepak.call("save_web_fact", {"text": "Alphabet, Google's parent company, reported 198,933 "
                                        "employees as of June 2026", "entity": "org:alphabet", "sources": [URL]})
    assert out["saved"] is False and ids(out["similar"]) == [old]
    assert len(deepak.memory.recall(deepak.agent, "org:google")) == 1   # nothing shared, nothing saved
    again = deepak.call("save_web_fact", {"text": "Alphabet, Google's parent company, reported 198,933 "
                                          "employees as of June 2026", "entity": "org:alphabet",
                                          "sources": [URL], "distinct_from": [old]})
    assert again["shared"] is True


def test_another_users_private_facts_are_never_shown_as_similar(deepak, priya):
    deepak.call("remember", {"text": "Acme pays its interns 40,000 a month", "entity": "org:acme"})
    out = priya.call("remember", {"text": "Acme pays its interns 40,000 a month", "entity": "org:acme"})
    assert "id" in out  # Priya can't see Deepak's private note, so it is no copy for her


# --- web checks: share public facts, dispute other users' facts ---

def test_a_supported_outside_world_fact_is_shared_but_facts_about_the_user_never(deepak, priya):
    acme = deepak.call("remember", {"text": "Acme has 500 employees", "entity": "org:acme"})["id"]
    job = deepak.call("remember", {"text": "Deepak works at Acme", "entity": "user:deepak",
                                   "also_about": ["org:acme"]})
    job = job["id"] if "id" in job else None
    assert job, "a claim about the user is not a copy of a claim about Acme's size"
    out = deepak.call("check_fact", {"fact_id": acme, "result": "supported", "sources": [URL]})
    assert out["shared"] is True
    assert ids(priya.memory.recall(priya.agent, "org:acme")) == [acme]
    assert priya.memory.recall(priya.agent, "org:acme")[0]["standing"] == "verified"
    out = deepak.call("check_fact", {"fact_id": job, "result": "supported", "sources": [URL]})
    assert out["shared"] is False and priya.memory.recall(priya.agent, "user:deepak") == []


def test_check_fact_needs_urls_from_this_turns_search(deepak):
    f = deepak.call("remember", {"text": "Acme has 500 employees", "entity": "org:acme"})["id"]
    out = deepak.call("check_fact", {"fact_id": f, "result": "supported", "sources": ["https://made.up"]})
    assert "error" in out and deepak.memory.fact_info(f)["visibility"] == "private"


def test_another_users_contradiction_disputes_a_verified_fact_until_its_owner_acts(deepak, priya):
    f = deepak.call("save_web_fact", {"text": "Alphabet has 190,820 employees", "entity": "org:alphabet",
                                      "sources": [URL]})["id"]
    mem = deepak.memory
    assert mem.disputes(deepak.agent) == []
    out = priya.call("check_fact", {"fact_id": f, "result": "contradicted", "sources": [URL],
                                    "note": "198,933 as of June 2026"})
    assert "another user's" in out["next"]
    assert mem.recall(priya.agent, "org:alphabet")[0]["standing"] == "doubtful"  # the latest check wins
    [d] = mem.disputes(deepak.agent)
    assert d["id"] == f and d["by"] == priya.agent and d["note"] == "198,933 as of June 2026"
    assert mem.disputes(priya.agent) == []          # only the owner's agents are asked to act
    assert "disputed fact " + f in _attention(mem, deepak.agent)
    new = deepak.call("supersede", {"old_id": f, "text": "Alphabet has 198,933 employees", "sources": [URL]})["new"]
    assert mem.disputes(deepak.agent) == [] and mem.recall(priya.agent, "org:alphabet")[0]["id"] == new


def test_a_later_supported_check_clears_a_dispute(deepak, priya):
    f = deepak.call("save_web_fact", {"text": "Acme is based in Berlin", "entity": "org:acme",
                                      "sources": [URL]})["id"]
    priya.call("check_fact", {"fact_id": f, "result": "contradicted", "sources": [URL]})
    deepak.call("check_fact", {"fact_id": f, "result": "supported", "sources": [URL]})
    assert deepak.memory.disputes(deepak.agent) == []
    assert deepak.memory.recall(priya.agent, "org:acme")[0]["standing"] == "verified"


# --- flagged work: shown at the start of a turn, closed when redone ---

def test_flagged_work_is_listed_for_the_team_and_a_redo_closes_it(mem):
    r = AgentTools.open(mem, "Researcher", owner="Deepak")
    p = AgentTools(mem, mem.register_agent("Planner", owner="Deepak"), r.session)
    mem.join_session(p.agent, r.session)
    assert _attention(mem, r.agent) == ""             # nothing to do: nothing is added to the prompt
    hq = r.call("remember", {"text": "Acme HQ is in Austin", "entity": "org:acme"})["id"]
    plan = p.call("decide", {"text": "Visit Austin", "derived_from": [hq]})["id"]
    new = r.call("correct", {"old_id": hq, "text": "Acme HQ is in Dallas", "reason": "misread"})["new"]
    text = _attention(mem, r.agent)
    assert f"flagged decision {plan} by {p.agent}" in text and f"source wrong: {hq}" in text
    with pytest.raises(MemoryGraphError):
        mem.decide(p.agent, r.session, "Visit Dallas", derived_from=[new], replaces=[new])  # not a flagged decision
    redo = p.call("decide", {"text": "Visit Dallas", "derived_from": [new], "replaces": [plan]})["id"]
    assert mem.flagged(p.agent) == [] and _attention(mem, r.agent) == ""
    rows = mem.g.query("MATCH (n {id:$n})-[:SUPERSEDES]->(o) RETURN o.id, o.status", {"n": redo}).result_set
    assert rows == [[plan, "superseded"]]


def test_a_redone_draft_keeps_its_lineage(mem):
    w = AgentTools.open(mem, "Writer", owner="Deepak")
    fact = w.call("remember", {"text": "Acme HQ is in Austin", "entity": "org:acme"})["id"]
    d1 = w.call("decide", {"text": "Visit Austin", "derived_from": [fact]})["id"]
    draft = w.call("remember", {"text": "Dear team, we fly to Austin", "entity": "doc:trip",
                                "derived_from": [d1], "source": "agent"})["id"]
    d2 = w.call("decide", {"text": "Visit Austin in May", "derived_from": [fact]})["id"]
    new = w.call("supersede", {"old_id": draft, "text": "Dear team, we fly to Austin in May",
                               "derived_from": [d2]})["new"]
    assert [x["id"] for x in mem.explain(w.agent, new)] == [new, d2, fact]


def test_needs_attention_goes_in_front_of_the_message_but_is_not_saved(deepak, priya):
    from llm import MockLLM, chat_turn
    f = deepak.call("save_web_fact", {"text": "Infosys was founded in 1991", "entity": "org:infosys",
                                      "sources": [URL]})["id"]
    priya.call("check_fact", {"fact_id": f, "result": "contradicted", "sources": [URL], "note": "1981"})
    client = MockLLM([MockLLM.reply("Hi!")])
    chat_turn(deepak, "researcher", "hi", client=client, owner="Deepak")
    sent = client.requests[0]["messages"][-1]["content"]
    assert sent.startswith("[Needs attention") and f"disputed fact {f}" in sent and sent.endswith("[Message]\nhi")
    assert "Needs attention" not in client.requests[0]["system"][1]["text"]
    saved = [m["text"] for m in deepak.memory.history(deepak.agent, deepak.session) if m["role"] == "user"]
    assert saved == ["hi"]                       # the chat keeps the user's own words


def test_the_actions_line_lists_changes_and_failures_not_reads():
    import json
    from llm import _actions_line
    record = json.dumps({"stop": "end_turn", "web_search": True, "calls": [
        {"tool": "recall", "args": {"entity": "org:acme"}, "result": '{"facts": []}', "ok": True},
        {"tool": "remember", "args": {"text": "x" * 60, "entity": "org:acme"},
         "result": '{"saved": false, "similar": [{"id": "fact:aa"}]}', "ok": True},
        {"tool": "supersede", "args": {"old_id": "fact:aa", "text": "y"},
         "result": '{"superseded": "fact:aa", "new": "fact:bb"}', "ok": True},
        {"tool": "check_fact", "args": {"fact_id": "fact:cc", "result": "contradicted", "sources": ["u"]},
         "result": '{"error": "none of these URLs"}', "ok": False},
        {"tool": "handoff", "args": {"recipient": "agent:deepak-planner", "item_ids": ["fact:bb"]},
         "result": '{"handoff": "handoff:dd"}', "ok": True}]})
    assert _actions_line(record) == (
        '(actions: web_search; remember "' + "x" * 40 + '…" -> not saved: a similar fact exists; '
        "supersede fact:aa -> fact:bb; check_fact fact:cc contradicted -> failed; "
        "handoff agent:deepak-planner -> handoff:dd)")
    assert _actions_line(json.dumps({"calls": [{"tool": "recall", "args": {}, "result": "{}"}]})) == ""
    assert _actions_line("not json") == ""
