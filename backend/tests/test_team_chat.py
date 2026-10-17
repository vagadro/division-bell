"""Team chat tests: one chat (session) shared by a user's agents, with MockLLM ($0)."""
import pytest

from llm import MockLLM, Spend, chat_turn, team_turn
from memory import MemoryGraphError
from tools import AgentTools


@pytest.fixture
def team(mem):
    """Deepak's three agents in one chat."""
    r, p, w = (mem.register_agent(n, owner="Deepak") for n in ("Researcher", "Planner", "Writer"))
    chat = mem.start_session(r, team=[p, w])
    return mem, chat, {a.rsplit("-", 1)[-1]: AgentTools(mem, a, chat) for a in (r, p, w)}


# One session shared by the user's agents

def test_the_users_agents_share_one_chat(team):
    mem, chat, agents = team
    mem.add_message(agents["researcher"].agent, chat, "user", "plan my laptop budget")
    mem.add_message(agents["planner"].agent, chat, "agent", "Budget: $1,000")
    history = mem.history(agents["writer"].agent, chat)   # the Writer reads the same chat
    assert [(m["role"], m["agent"]) for m in history] == [
        ("user", agents["researcher"].agent), ("agent", agents["planner"].agent)]


def test_another_users_agent_cannot_join_or_read_the_chat(team):
    mem, chat, _ = team
    ravi = mem.register_agent("Researcher", owner="Ravi")
    with pytest.raises(MemoryGraphError):
        mem.join_session(ravi, chat)
    with pytest.raises(MemoryGraphError):
        mem.history(ravi, chat)


def test_an_agent_sees_its_teammates_replies_labelled(team):
    _, _, agents = team
    chat_turn(agents["researcher"], "researcher", "hi", client=MockLLM([MockLLM.reply("Found it.")]))
    client = MockLLM([MockLLM.reply("Planned.")])
    chat_turn(agents["planner"], "planner", "now plan", client=client)
    sent = [(m["role"], m["content"]) for m in client.requests[0]["messages"]]
    assert sent == [("user", "hi"), ("assistant", "[researcher] Found it."), ("user", "now plan")]


# Automatic handoffs (team_turn)

@pytest.fixture
def fact(team):
    mem, chat, agents = team
    return mem.remember(agents["researcher"].agent, chat, "Deepak's salary is $20,000", "user:deepak")


def handoff(to, items, note):
    return MockLLM.tool_call("handoff", {"recipient": to.agent, "item_ids": items, "note": note})


def test_a_simple_question_stays_with_the_researcher(team):
    _, _, agents = team
    out = team_turn(agents, "hi", client=MockLLM([MockLLM.reply("Hello!")]))
    assert [s["role"] for s in out["steps"]] == ["researcher"] and out["reply"] == "Hello!"


def test_handoffs_run_the_next_agent_at_once(team, fact):
    mem, chat, agents = team
    script = [handoff(agents["planner"], [fact], "plan a laptop budget"), MockLLM.reply("Passed on."),
              MockLLM.tool_call("decide", {"text": "Laptop budget: $1,000", "derived_from": [fact]}),
              handoff(agents["writer"], [fact], "write it up"), MockLLM.reply("Decided."),
              MockLLM.reply("Draft: your laptop budget is $1,000.")]
    client = MockLLM(script)
    out = team_turn(agents, "plan my laptop budget", client=client)
    assert [s["role"] for s in out["steps"]] == ["researcher", "planner", "writer"]
    assert out["reply"] == "Draft: your laptop budget is $1,000." and out["stop"] == "done"
    assert not client.script                                            # every scripted reply was used
    decision = out["steps"][1]["tool_calls"][0]["result"]
    assert decision["id"].startswith("dec:")                            # built on the private salary: allowed
    writer_task = client.requests[-1]["messages"][-1]["content"]
    assert writer_task.startswith("Handoff from the planner: write it up") and fact in writer_task
    notes = [m for m in mem.history(agents["writer"].agent, chat) if m["text"].startswith("Handoff")]
    assert [m["agent"] for m in notes] == [agents["researcher"].agent, agents["planner"].agent]


def test_a_team_that_keeps_passing_work_on_is_stopped(team, fact):
    _, _, agents = team
    r, p = agents["researcher"], agents["planner"]
    script = [handoff(p, [fact], "you"), MockLLM.reply("."), handoff(r, [fact], "no, you"), MockLLM.reply("."),
              handoff(p, [fact], "you"), MockLLM.reply(".")]
    out = team_turn(agents, "go", client=MockLLM(script), max_hops=2)
    assert out["stop"] == "hop_cap" and len(out["steps"]) == 3


def test_a_handoff_to_another_users_agent_is_stored_not_run(team, fact):
    mem, _, agents = team
    ravi = mem.register_agent("Planner", owner="Ravi")
    mem.publish(agents["researcher"].agent, fact)                      # a handoff to another user needs a shared item
    script = [MockLLM.tool_call("handoff", {"recipient": ravi, "item_ids": [fact], "note": "fyi"}),
              MockLLM.reply("Sent to Ravi.")]
    out = team_turn(agents, "send it to Ravi", client=MockLLM(script))
    assert len(out["steps"]) == 1 and out["steps"][0]["tool_calls"][0]["result"].get("handoff")
    assert mem.inbox(ravi)[0]["note"] == "fyi"


def test_the_spend_cap_covers_the_whole_chain(team, fact):
    _, _, agents = team
    script = [handoff(agents["planner"], [fact], "plan"), MockLLM.reply("Passed on."), MockLLM.reply("never")]
    out = team_turn(agents, "plan", client=MockLLM(script), spend=Spend(cap=0.0002))
    assert out["stop"] == "spend_cap" and len(out["steps"]) == 1 and "spending cap" in out["reply"]


# Search_history: earlier chats, full-text index

@pytest.fixture
def earlier_chat(team):
    """Deepak said three things in an earlier chat; Ravi talked about salaries too."""
    mem, _, agents = team
    r = agents["researcher"].agent
    old = mem.start_session(r)
    for text in ("My salary is 20000 dollars", "The Acme meeting went well", "salary talks are next week"):
        mem.add_message(r, old, "user", text)
    ravi = mem.register_agent("Researcher", owner="Ravi")
    mem.add_message(ravi, mem.start_session(ravi), "user", "My salary is secret")
    return old


def texts(out):
    return [m["text"] for m in out["messages"]]


def test_search_finds_earlier_chats_best_match_first(team, earlier_chat):
    _, _, agents = team
    out = agents["planner"].call("search_history", {"query": "salaries"})   # word forms match
    assert sorted(texts(out)) == ["My salary is 20000 dollars", "salary talks are next week"]
    assert all(m["session"] == earlier_chat for m in out["messages"])
    best = agents["planner"].call("search_history", {"query": "salary dollars"})
    assert texts(best)[0] == "My salary is 20000 dollars"                  # has all the words: first


def test_any_of_the_words_may_match(team, earlier_chat):
    _, _, agents = team
    assert texts(agents["planner"].call("search_history", {"query": "earn"})) == []   # words, not meaning
    assert "My salary is 20000 dollars" in texts(agents["planner"].call("search_history", {"query": "earn salary"}))


def test_search_never_returns_another_users_messages(team, earlier_chat):
    _, _, agents = team
    assert "My salary is secret" not in texts(agents["researcher"].call("search_history", {"query": "salary"}))


def test_search_leaves_out_the_current_chat(team, earlier_chat):
    mem, chat, agents = team
    mem.add_message(agents["researcher"].agent, chat, "user", "Acme again")
    found = agents["researcher"].call("search_history", {"query": "acme"})["messages"]
    assert [m["session"] for m in found] == [earlier_chat]


def test_odd_characters_do_not_break_the_search(team, earlier_chat):
    _, _, agents = team
    assert texts(agents["planner"].call("search_history", {"query": "salary)( \"x"}))   # cleaned to salary|x
    assert "error" in agents["planner"].call("search_history", {"query": "!!!"})
