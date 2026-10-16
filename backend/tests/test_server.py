"""HTTP API tests: FastAPI's TestClient calls the app in-process, with MockLLM ($0) and test graphs."""
import pytest
from fastapi.testclient import TestClient

from llm import MockLLM
from memory import Tenants
from server import ORGS, create_app


@pytest.fixture
def tenants():
    t = Tenants(prefix="test_srv_")
    yield t
    for org in ORGS:
        t.drop(org.lower())


def make(tenants, script=(), **kw):
    client = MockLLM(list(script))
    return TestClient(create_app(tenants=tenants, client=client, mode="mock", **kw)), client


def who(user="Deepak", org="DemoOrg1"):
    return {"X-User": user, "X-Org": org}


def test_entering_creates_the_users_three_agents(tenants):
    api, _ = make(tenants)
    out = api.post("/api/enter", headers=who()).json()
    assert out["agents"] == {"researcher": "agent:deepak-researcher", "planner": "agent:deepak-planner",
                             "writer": "agent:deepak-writer"}
    assert out["chats"] == []


@pytest.mark.parametrize("headers", [who(org="Acme"), who(user=""), who(user="../x"), {}])
def test_bad_identity_headers_are_refused(tenants, headers):
    api, _ = make(tenants)
    assert api.post("/api/enter", headers=headers).status_code == 400


def test_a_chat_round_trip(tenants):
    api, client = make(tenants, [MockLLM.reply("New Delhi.")])
    chat = api.post("/api/chats", headers=who()).json()["id"]
    out = api.post(f"/api/chats/{chat}/messages", json={"text": "capital of India?"}, headers=who()).json()
    assert out["reply"] == "New Delhi." and [s["role"] for s in out["steps"]] == ["researcher"]
    history = api.get(f"/api/chats/{chat}", headers=who()).json()["messages"]
    assert [m["text"] for m in history if m["role"] != "tool"] == ["capital of India?", "New Delhi."]
    [listed] = api.get("/api/chats", headers=who()).json()["chats"]
    assert listed["id"] == chat and listed["title"] == "capital of India?"
    assert "Deepak" in client.requests[0]["system"][1]["text"]


def test_another_user_can_neither_read_nor_write_a_chat(tenants):
    api, _ = make(tenants, [MockLLM.reply("never")])
    chat = api.post("/api/chats", headers=who()).json()["id"]
    ravi = who(user="Ravi")
    assert api.get(f"/api/chats/{chat}", headers=ravi).status_code == 400
    assert api.post(f"/api/chats/{chat}/messages", json={"text": "hi"}, headers=ravi).status_code == 400


def seed(api, tenants, user="Deepak", org="DemoOrg1"):
    """The user's memory and agents, to write test data the way the agents would (not the page)."""
    agents = api.post("/api/enter", headers=who(user=user, org=org)).json()["agents"]
    mem = tenants.get(org.lower())
    return mem, agents, mem.start_session(agents["researcher"], team=[agents["planner"]])


def test_the_page_can_only_read(tenants):
    api, _ = make(tenants)
    mem, a, chat = seed(api, tenants)
    fact = mem.remember(a["researcher"], chat, "Acme has 500 employees", "org:acme")
    mem.publish(a["researcher"], fact)
    dec = mem.decide(a["planner"], chat, "Pitch enterprise", derived_from=[fact])
    radius = api.post("/api/tools/blast_radius", headers=who(), json={"args": {"item_id": fact}}).json()
    assert [x["id"] for x in radius["affected"]] == [dec]
    chain = api.post("/api/tools/explain", headers=who(), json={"args": {"item_id": dec}}).json()["chain"]
    assert [c["id"] for c in chain] == [dec, fact]
    # nothing that changes memory can be sent from the page, not even by the fact's own user
    for name, args in [("retract", {"fact_id": fact}), ("publish", {"item_id": fact}),
                       ("correct", {"old_id": fact, "text": "Acme has 50 employees"}),
                       ("remember", {"text": "x", "entity": "org:acme"}), ("confirm", {"item_id": dec})]:
        assert api.post(f"/api/tools/{name}", headers=who(), json={"args": args}).status_code == 403
    assert mem.recall(a["researcher"], "org:acme")[0]["id"] == fact      # unchanged
    mem.correct(a["researcher"], chat, fact, "Acme has 50 employees")     # an agent can
    flagged = api.get("/api/flagged", headers=who()).json()["flagged"]
    assert [i["id"] for i in flagged["planner"]] == [dec]


def test_the_graph_view_is_filtered_per_user(tenants):
    api, _ = make(tenants)
    mem, a, chat = seed(api, tenants)
    mem.remember(a["researcher"], chat, "my salary is $20,000", "user:deepak")
    mine = {n["id"] for n in api.get("/api/graph", headers=who()).json()["nodes"]}
    ravis = api.get("/api/graph", headers=who(user="Ravi")).json()
    assert "user:deepak" in mine and "salary" not in str(ravis)


def test_orgs_are_separate_graphs(tenants):
    api, _ = make(tenants)
    mem, a, chat = seed(api, tenants)
    mem.publish(a["researcher"], mem.remember(a["researcher"], chat, "x", "org:acme"))
    other_org = api.get("/api/graph", headers=who(user="Ravi", org="DemoOrg2")).json()
    assert other_org["nodes"] == []


def test_the_spend_cap_answers_429(tenants):
    api, _ = make(tenants, [MockLLM.reply("x")], spend_cap=0)
    chat = api.post("/api/chats", headers=who()).json()["id"]
    assert api.post(f"/api/chats/{chat}/messages", json={"text": "hi"}, headers=who()).status_code == 429
    assert api.get("/api/status").json()["mode"] == "mock"
