"""Chat loop tests: the real chat_turn() loop against MockLLM (scripted replies, $0).

Covers what a turn saves (messages, the fact -> message link), the loop's limits (round cap,
spend cap, pause_turn), the web fact guard, read_document, and the source facts are saved with.
"""
import pytest

from llm import LLMBudgetError, MockLLM, Spend, chat_turn
from tools import AgentTools

URL = "https://example.com/india"


@pytest.fixture
def deepak(mem, tmp_path):
    return AgentTools.open(mem, "Researcher", owner="Deepak", docs_dir=tmp_path)


@pytest.fixture
def ravi(mem):
    return AgentTools.open(mem, "Researcher", owner="Ravi")


def turn(tools, script, text="hi", role="researcher", **kw):
    client = MockLLM(script)
    return chat_turn(tools, role, text, client=client, owner="Deepak", **kw), client


def tool_results(request):
    """The tool_result blocks sent back to the model in a request."""
    return [b for m in request["messages"] if isinstance(m["content"], list)
            for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_result"]


# --- what a turn saves ---

def test_a_plain_answer_saves_both_messages(deepak):
    out, client = turn(deepak, [MockLLM.reply("Hello!")])
    assert out["reply"] == "Hello!" and out["stop"] == "end_turn" and out["cost"] > 0
    history = deepak.memory.history(deepak.agent, deepak.session)
    assert [(m["role"], m["text"]) for m in history if m["role"] != "tool"] == [("user", "hi"), ("agent", "Hello!")]
    assert history[0]["id"] == out["message"] and history[-1]["role"] == "tool"   # the turn's record


def test_the_system_prompt_is_cached_and_names_the_user_after_the_cache(deepak):
    _, client = turn(deepak, [MockLLM.reply("ok")])
    fixed, per_user = client.requests[0]["system"]
    assert fixed["cache_control"] == {"type": "ephemeral"} and "Deepak" not in fixed["text"]
    assert "cache_control" not in per_user
    assert "Deepak" in per_user["text"] and "agent:deepak-planner" in per_user["text"]
    assert "user:deepak" in per_user["text"]  # the user's entity id is given, never guessed


def test_earlier_messages_of_the_chat_are_sent_nothing_else(deepak):
    turn(deepak, [MockLLM.reply("Hello!")])
    _, client = turn(deepak, [MockLLM.reply("Again!")], text="again")
    sent = client.requests[0]["messages"]
    assert [(m["role"], m["content"]) for m in sent[:3]] == [
        ("user", "hi"), ("assistant", "Hello!"), ("user", "again")]


def test_a_fact_saved_in_a_turn_is_linked_to_the_user_message(deepak, ravi):
    script = [MockLLM.tool_call("remember", {"text": "Deepak's salary is $20,000", "entity": "user:deepak"}),
              MockLLM.reply("Noted.")]
    out, _ = turn(deepak, script, text="my salary is 20000$")
    fact = out["tool_calls"][0]["result"]["id"]
    [ev] = deepak.memory.evidence(deepak.agent, fact)
    assert ev["message"] == out["message"] and ev["text"] == "my salary is 20000$"
    assert ravi.call("recall", {"entity": "user:deepak"})["facts"] == []   # private: another user can't see it
    assert deepak.current_message is None                                   # cleared after the turn


# --- the loop ---

def test_all_tool_results_go_back_and_errors_are_marked(deepak):
    script = [MockLLM.tool_call("publish", {"item_id": "fact:nope"}), MockLLM.reply("That failed.")]
    out, client = turn(deepak, script)
    [result] = tool_results(client.requests[1])
    assert result["is_error"] is True and "error" in out["tool_calls"][0]["result"]
    assert out["reply"] == "That failed."


def test_a_model_that_never_stops_is_cut_off(deepak):
    script = [MockLLM.tool_call("recall", {"entity": "org:acme"}) for _ in range(6)]
    out, client = turn(deepak, script)
    assert out["stop"] == "round_cap" and len(client.requests) == 6
    assert "ran out of steps" in out["reply"]
    assert [m for m in deepak.memory.history(deepak.agent, deepak.session) if m["role"] == "agent"][-1]["text"] == out["reply"]


def test_the_spend_cap_stops_the_next_call(deepak):
    spend = Spend(cap=0.0001)   # one mock reply costs $0.00015
    script = [MockLLM.tool_call("recall", {"entity": "org:acme"}), MockLLM.reply("never sent")]
    with pytest.raises(LLMBudgetError):
        turn(deepak, script, spend=spend)
    with pytest.raises(LLMBudgetError):   # and the next turn is refused before any call
        turn(deepak, [MockLLM.reply("x")], spend=spend)


def test_spend_prices_tokens_cache_and_searches():
    from types import SimpleNamespace as NS
    usage = NS(input_tokens=1_000_000, output_tokens=1_000_000, cache_creation_input_tokens=0,
               cache_read_input_tokens=1_000_000, server_tool_use=NS(web_search_requests=2))
    assert Spend().add("claude-haiku-5-5", usage) == pytest.approx(0.10 + 0.50 + 0.01 + 0.02)


def test_a_paused_search_is_resumed(deepak):
    out, client = turn(deepak, [MockLLM.web_search([URL]), MockLLM.reply("New Delhi.")])
    assert len(client.requests) == 2 and out["reply"] == "New Delhi." and out["stop"] == "end_turn"


def test_a_refusal_gets_a_plain_reply(deepak):
    out, _ = turn(deepak, [MockLLM.reply("", stop_reason="refusal")])
    assert out["stop"] == "refusal" and out["reply"]


def test_only_the_researcher_gets_web_search(mem, deepak):
    planner = AgentTools.open(mem, "Planner", owner="Deepak")
    _, r = turn(deepak, [MockLLM.reply("ok")])
    _, p = turn(planner, [MockLLM.reply("ok")], role="planner")
    assert r.requests[0]["tools"][0]["name"] == "web_search"
    assert "web_search" not in [t["name"] for t in p.requests[0]["tools"]]


# --- web facts ---

def test_a_web_fact_must_cite_a_url_the_search_returned(deepak, ravi):
    fact = {"text": "The capital of India is New Delhi", "entity": "country:india"}
    script = [MockLLM.web_search([URL]),
              MockLLM.tool_call("save_web_fact", {**fact, "sources": ["https://made-up.example"]}),
              MockLLM.tool_call("save_web_fact", {**fact, "sources": [URL, "https://made-up.example"]}),
              MockLLM.reply("New Delhi.")]
    out, _ = turn(deepak, script, text="capital of India?")
    refused, saved = out["tool_calls"][0]["result"], out["tool_calls"][1]["result"]
    assert "error" in refused
    assert saved["shared"] and saved["sources"] == [URL] and saved["ignored"] == ["https://made-up.example"]
    [seen] = ravi.call("recall", {"entity": "country:india"})["facts"]   # shared with other users
    assert seen["id"] == saved["id"] and seen["verified"] is True


def test_search_urls_do_not_carry_over_to_the_next_turn(deepak):
    turn(deepak, [MockLLM.web_search([URL]), MockLLM.reply("ok")])
    script = [MockLLM.tool_call("save_web_fact", {"text": "x", "entity": "country:india", "sources": [URL]}),
              MockLLM.reply("ok")]
    out, _ = turn(deepak, script)
    assert "error" in out["tool_calls"][0]["result"]


# --- documents ---

def test_read_document_returns_labelled_passages(deepak, tmp_path):
    (tmp_path / "notes.txt").write_text("Budget for 2026 is 5 lakh.")
    out = deepak.call("read_document", {"document": "notes.txt"})
    assert out["complete"] and "5 lakh" in out["passages"][0]["text"] and out["passages"][0]["label"]


def test_read_document_stays_inside_the_documents_folder(deepak):
    assert "error" in deepak.call("read_document", {"document": "../secret.txt"})


# --- sources: web facts are verified claims, user facts are the user's word ---

def test_facts_carry_their_source(deepak):
    script = [MockLLM.web_search([URL]),
              MockLLM.tool_call("save_web_fact", {"text": "The capital of India is New Delhi",
                                                  "entity": "country:india", "sources": [URL]}),
              MockLLM.tool_call("remember", {"text": "Deepak lives in Pune", "entity": "user:deepak"}),
              MockLLM.reply("New Delhi.")]
    turn(deepak, script)
    [web] = deepak.memory.recall(deepak.agent, "country:india")
    [mine] = deepak.memory.recall(deepak.agent, "user:deepak")
    assert (web["source"], web["standing"]) == ("web", "verified")
    assert (mine["source"], mine["standing"]) == ("user", "user")         # the remember tool's default
    assert deepak.memory.records([deepak.agent])[deepak.agent]["claims"] == 1   # only the web fact


# --- what the agent did is recorded for the page, not said in the reply ---

def test_what_the_agent_did_is_recorded_and_replayed_as_one_line(deepak):
    import json
    script = [MockLLM.tool_call("remember", {"text": "Deepak works at Apple", "entity": "user:deepak"}),
              MockLLM.reply("Nice to meet you, Deepak!")]
    turn(deepak, script, text="I work at Apple")
    history = deepak.memory.history(deepak.agent, deepak.session)
    assert [m["role"] for m in history] == ["user", "agent", "tool"]        # the reply, then the record
    record = json.loads(history[2]["text"])
    assert record["stop"] == "end_turn" and record["web_search"] is False
    [call] = record["calls"]
    assert call["tool"] == "remember" and call["args"]["entity"] == "user:deepak" and call["ok"]
    assert '"id": "fact:' in call["result"]
    _, client = turn(deepak, [MockLLM.reply("ok")], text="again")
    sent = [m["content"] for m in client.requests[0]["messages"]]
    fact = json.loads(call["result"])["id"]
    # not the raw record: one line saying what really happened, so the agent never doubts it
    assert sent == ["I work at Apple",
                    f'Nice to meet you, Deepak!\n(actions: remember "Deepak works at Apple" -> {fact})', "again"]
    other = deepak.memory.start_session(deepak.agent)
    found = deepak.memory.search_history(deepak.agent, "apple", exclude_session=other)
    assert {m["role"] for m in found} == {"user"}                         # nor found by search
