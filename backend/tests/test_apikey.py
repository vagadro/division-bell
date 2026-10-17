"""The visitor's own Anthropic API key: sent per request, never stored, spend counted per key."""
import httpx
import pytest
from fastapi.testclient import TestClient

import anthropic
import server
from llm import MockLLM
from memory import Tenants
from server import ORGS, create_app

KEY_A, KEY_B = "sk-ant-test-key-AAAA1111", "sk-ant-test-key-BBBB2222"


@pytest.fixture
def tenants():
    t = Tenants(prefix="test_key_")
    yield t
    for org in ORGS:
        t.drop(org.lower())


@pytest.fixture(autouse=True)
def no_server_key(monkeypatch, tenants):
    # Depends on tenants because opening the database loads .env, which would put a real key back.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture
def seen(monkeypatch):
    """Replaces anthropic.Anthropic with a factory that records the key it was given and hands
    back a scripted client, so no request leaves the machine."""
    made = []

    def factory(api_key=None, **kwargs):
        client = MockLLM([MockLLM.reply("Noted.")] * 5)
        client.models = type("M", (), {"list": staticmethod(lambda **kw: [])})()
        made.append((api_key, client))
        return client

    monkeypatch.setattr(server.anthropic, "Anthropic", factory)
    return made


def real_app(tenants):
    """The app as shipped: haiku mode and no injected client, so every request needs a key."""
    return TestClient(create_app(tenants=tenants, mode="haiku"))


def chat_of(api):
    h = {"X-User": "Deepak", "X-Org": "DemoOrg1"}
    return h, api.post("/api/chats", headers=h).json()["id"]


def test_the_server_starts_without_any_key(tenants):
    assert real_app(tenants).get("/api/orgs").status_code == 200


def test_a_message_without_a_key_asks_for_one(tenants):
    api = real_app(tenants)
    h, chat = chat_of(api)
    res = api.post(f"/api/chats/{chat}/messages", json={"text": "hello"}, headers=h)
    assert res.status_code == 401 and "API key" in res.json()["detail"]


def test_the_key_from_the_page_is_what_the_client_is_built_with(tenants, seen):
    api = real_app(tenants)
    h, chat = chat_of(api)
    res = api.post(f"/api/chats/{chat}/messages", json={"text": "hello"}, headers={**h, "X-Api-Key": KEY_A})
    assert res.status_code == 200 and res.json()["reply"] == "Noted."
    assert [k for k, _ in seen] == [KEY_A]


def test_the_server_fallback_key_is_used_when_the_page_sends_none(tenants, seen, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY_B)
    api = real_app(tenants)
    h, chat = chat_of(api)
    assert api.post(f"/api/chats/{chat}/messages", json={"text": "hello"}, headers=h).status_code == 200
    assert [k for k, _ in seen] == [KEY_B]
    assert api.get("/api/status").json()["server_key"] is True


def test_a_key_from_the_page_wins_over_the_server_key(tenants, seen, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY_B)
    api = real_app(tenants)
    h, chat = chat_of(api)
    api.post(f"/api/chats/{chat}/messages", json={"text": "hello"}, headers={**h, "X-Api-Key": KEY_A})
    assert [k for k, _ in seen] == [KEY_A]


def test_spend_is_counted_per_key(tenants, seen):
    api = real_app(tenants)
    h, chat = chat_of(api)
    api.post(f"/api/chats/{chat}/messages", json={"text": "hello"}, headers={**h, "X-Api-Key": KEY_A})
    a = api.get("/api/status", headers={"X-Api-Key": KEY_A}).json()
    b = api.get("/api/status", headers={"X-Api-Key": KEY_B}).json()
    assert a["calls"] == 1 and b["calls"] == 0


def test_the_key_never_appears_in_a_response(tenants, seen):
    api = real_app(tenants)
    h, chat = chat_of(api)
    bodies = [
        api.post(f"/api/chats/{chat}/messages", json={"text": "hello"}, headers={**h, "X-Api-Key": KEY_A}).text,
        api.get("/api/status", headers={"X-Api-Key": KEY_A}).text,
        api.post("/api/key/check", headers={"X-Api-Key": KEY_A}).text,
        api.get(f"/api/chats/{chat}", headers=h).text,
        api.get("/api/graph", headers=h).text,
    ]
    assert not any(KEY_A in b or "AAAA1111" in b for b in bodies)


def test_key_check_accepts_a_working_key(tenants, seen):
    assert real_app(tenants).post("/api/key/check", headers={"X-Api-Key": KEY_A}).json() == {"ok": True}


def test_key_check_without_a_key_asks_for_one(tenants):
    assert real_app(tenants).post("/api/key/check").status_code == 401


def rejected():
    request = httpx.Request("GET", "https://api.anthropic.com/v1/models")
    return anthropic.AuthenticationError("invalid x-api-key", response=httpx.Response(401, request=request), body=None)


def test_a_rejected_key_gives_a_plain_message_not_the_key(tenants, monkeypatch):
    def factory(api_key=None, **kwargs):
        def boom(**kw):
            raise rejected()
        client = MockLLM([])
        client.models = type("M", (), {"list": staticmethod(boom)})()
        return client

    monkeypatch.setattr(server.anthropic, "Anthropic", factory)
    res = real_app(tenants).post("/api/key/check", headers={"X-Api-Key": KEY_A})
    assert res.status_code == 401 and "rejected" in res.json()["detail"] and KEY_A not in res.text
