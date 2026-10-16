"""Read views for the UI: agents, chats, and the graph panel's snapshot (filtered per user)."""
import pytest


@pytest.fixture
def org(mem):
    """Deepak (two agents, one chat) and Ravi share an org. Deepak has a private salary fact
    from a message and a private decision on it; Ravi has published one fact."""
    r, p = (mem.register_agent(n, owner="Deepak") for n in ("Researcher", "Planner"))
    ravi = mem.register_agent("Researcher", owner="Ravi")
    chat = mem.start_session(r, team=[p])
    msg = mem.add_message(r, chat, "user", "my salary is 20000$ and I want a laptop budget")
    salary = mem.remember(r, chat, "Deepak's salary is $20,000", "user:deepak", from_messages=[msg])
    budget = mem.decide(p, chat, "Laptop budget: $1,000", derived_from=[salary])
    rs = mem.start_session(ravi)
    india = mem.remember(ravi, rs, "The capital of India is New Delhi", "country:india")
    mem.publish(ravi, india)
    mem.verify(r, india, "web", "supported", ["https://example.com/india"])
    return dict(mem=mem, r=r, p=p, ravi=ravi, chat=chat, msg=msg, salary=salary, budget=budget,
                india=india, ravi_chat=rs)


def test_agents_lists_everyone_in_the_org(org):
    listed = {a["id"]: a for a in org["mem"].agents()}
    assert set(listed) == {org["r"], org["p"], org["ravi"]}
    assert listed[org["ravi"]]["owner"] == "user:ravi" and listed[org["r"]]["accuracy"] == 1.0


def test_chats_are_the_users_own_newest_first_titled_by_the_first_message(org):
    mem = org["mem"]
    later = mem.start_session(org["p"])
    mem.add_message(org["p"], later, "user", "x" * 100)
    chats = mem.chats(org["r"])                                   # the Researcher sees the Planner's chat too
    assert [c["id"] for c in chats] == [later, org["chat"]]
    assert chats[0]["title"] == "x" * 59 + "…"
    assert chats[1]["title"].startswith("my salary") and chats[1]["messages"] == 1
    assert org["ravi_chat"] not in [c["id"] for c in chats]


def test_the_graph_shows_the_users_private_items_and_shared_ones(org):
    view = org["mem"].graph(org["r"])
    nodes = {n["id"]: n for n in view["nodes"]}
    assert {org["salary"], org["budget"], org["india"], org["msg"], "user:deepak", org["ravi"]} <= set(nodes)
    assert nodes[org["budget"]]["visibility"] == "private" and nodes[org["india"]]["visibility"] == "shared"
    edges = {(e["source"], e["type"], e["target"]) for e in view["edges"]}
    assert {(org["budget"], "DERIVED_FROM", org["salary"]), (org["salary"], "EXTRACTED_FROM", org["msg"]),
            (org["salary"], "ABOUT", "user:deepak"), (org["r"], "ASSERTED", org["salary"])} <= edges
    [check] = [e for e in view["edges"] if e["type"] == "VERIFIED"]
    assert (check["source"], check["target"], check["result"]) == (org["r"], org["india"], "supported")


def test_another_user_never_sees_private_items_or_messages(org):
    view = org["mem"].graph(org["ravi"])
    ids = {n["id"] for n in view["nodes"]}
    assert org["india"] in ids
    assert not ids & {org["salary"], org["budget"], org["msg"], "user:deepak"}
    assert all(e["source"] in ids and e["target"] in ids for e in view["edges"])   # no edge to a hidden node
    assert "salary" not in str(view).lower()


def test_status_changes_show_in_the_graph(org):
    mem = org["mem"]
    mem.retract(org["r"], org["salary"], reason="wrong number")
    nodes = {n["id"]: n for n in mem.graph(org["r"])["nodes"]}
    assert nodes[org["salary"]]["status"] == "retracted" and nodes[org["budget"]]["status"] == "flagged"
