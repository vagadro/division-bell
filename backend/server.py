"""HTTP API for the demo page. A thin layer: every route calls team_turn(), AgentTools or a
Memory read view, so the memory rules apply to the page exactly as to agents and tests.

Who is calling comes from two headers, X-User (a name) and X-Org (DemoOrg1-3). Nothing is
authenticated: anyone who sends the same name and org is that user. The server binds to
127.0.0.1 only. Each user gets three agents (Researcher, Planner, Writer) on first visit.

Mode: MEMORY_LLM=haiku (default): the real agents, spend-capped. Whoever uses the page pastes
their own Anthropic API key; the page sends it in the X-Api-Key header with each message. The
server builds a client for that request only and never stores or logs the key. An
ANTHROPIC_API_KEY in the environment is an optional fallback for requests that send none.
MEMORY_LLM=mock is a scripted stand-in that exists for the tests only.
Run (from backend/): uv run uvicorn server:app --host 127.0.0.1 --port 8000
"""
import hashlib
import os
import re
from pathlib import Path

import anthropic
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import llm
from documents import DocumentError
from memory import MemoryGraphError, Tenants
from tools import AgentTools

ORGS = ("DemoOrg1", "DemoOrg2", "DemoOrg3")
ROLES = ("researcher", "planner", "writer")
STATIC = Path(__file__).resolve().parent.parent / "frontend"  # the page, served at /static
# The page may only look: memory changes (publish, correct, retract, confirm...) are made by the
# agents in the chat, never by a button, as they would be in a real organisation.
READ_TOOLS = ("recall", "explain", "blast_radius", "flagged", "inbox", "search_history", "recall_procedures")
NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9 ]{0,30}")


class Text(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class ToolCall(BaseModel):
    role: str = "researcher"      # which of the caller's own agents runs the tool
    chat: str | None = None       # the chat it runs in (needed by tools that write)
    args: dict = {}


def create_app(tenants=None, client=None, mode=None, spend_cap=llm.SPEND_CAP, docs_dir=None):
    """Build the app. Tests pass their own tenants (a test prefix) and a scripted client."""
    mode = mode or os.environ.get("MEMORY_LLM", "haiku")
    if mode not in ("mock", "haiku"):
        raise ValueError(f"MEMORY_LLM must be 'mock' or 'haiku', not {mode!r}")
    state = {"tenants": tenants, "client": client, "spend": llm.Spend(cap=spend_cap), "spends": {}}
    docs_dir = docs_dir or os.environ.get("MEMORY_DOCS_DIR")
    app = FastAPI(title="GraphHacks shared agent memory")

    def tenants_():
        if state["tenants"] is None:
            state["tenants"] = Tenants()
        return state["tenants"]

    def key_for(header_key):
        """The Anthropic key a request uses: the one the page sent, else the server's own, else None."""
        return (header_key or "").strip() or os.environ.get("ANTHROPIC_API_KEY") or None

    def scripted():
        """True when a test injected a client or the mock mode is on: no real key is involved."""
        return state["client"] is not None or mode == "mock"

    def spend_for(key):
        """The spend counter for a key, found by a hash of it so the key itself is not kept."""
        if scripted():
            return state["spend"]
        tag = hashlib.sha256((key or "").encode()).hexdigest()[:16]
        return state["spends"].setdefault(tag, llm.Spend(cap=spend_cap))

    def client_for(header_key):
        """A client for this request. Without any key the page is told to ask for one."""
        if scripted():
            if state["client"] is None:
                state["client"] = llm.EchoLLM()
            return state["client"]
        key = key_for(header_key)
        if not key:
            raise HTTPException(401, "Add your Anthropic API key (bottom left of the page) to talk to the agents.")
        return anthropic.Anthropic(api_key=key)

    def caller(user, org):
        """(memory, owner name, {role: agent id}) for the request's headers. Creates the agents."""
        if org not in ORGS:
            raise HTTPException(400, f"X-Org must be one of {list(ORGS)}")
        if not user or not NAME_RE.fullmatch(user.strip()):
            raise HTTPException(400, "X-User must be a name: letters, digits and spaces, up to 31")
        mem = tenants_().get(org.lower())
        owner = user.strip()
        return mem, owner, {r: mem.register_agent(r.title(), owner=owner) for r in ROLES}

    def team(mem, agents, chat):
        """The user's three agents as tools on one chat. Joining is a no-op if already in."""
        for agent in agents.values():
            mem.join_session(agent, chat)
        return {r: AgentTools(mem, a, chat, docs_dir=docs_dir) for r, a in agents.items()}

    def guarded(fn):
        """Run fn; memory rule errors become 400s, the spend cap a 429."""
        try:
            return fn()
        except (MemoryGraphError, DocumentError) as e:
            raise HTTPException(400, str(e))
        except llm.LLMBudgetError as e:
            raise HTTPException(429, str(e))
        except anthropic.AuthenticationError:
            raise HTTPException(401, "Anthropic rejected this API key. Check it and paste it again.")
        except anthropic.PermissionDeniedError:
            raise HTTPException(403, "This API key is not allowed to use the model.")
        except anthropic.RateLimitError:
            raise HTTPException(429, "Anthropic is rate limiting this key. Wait a moment and try again.")
        except anthropic.APIConnectionError:
            raise HTTPException(502, "Could not reach the Anthropic API.")
        except anthropic.APIStatusError as e:
            raise HTTPException(502, f"The Anthropic API returned an error ({e.status_code}).")

    # Who am I

    @app.get("/api/orgs")
    def orgs():
        return {"orgs": list(ORGS)}

    @app.post("/api/enter")
    def enter(x_user: str = Header(None), x_org: str = Header(None)):
        """First visit (or return): make sure the user's agents exist; list their chats."""
        mem, owner, agents = caller(x_user, x_org)
        return {"user": owner, "org": x_org, "agents": agents, "chats": mem.chats(agents["researcher"])}

    @app.get("/api/status")
    def status(x_api_key: str = Header(None)):
        """Spend is counted per API key. `server_key` tells the page whether the server has its
        own key, so it knows whether to ask the visitor for one."""
        s = spend_for(key_for(x_api_key))
        return {"mode": mode, "spend": round(s.total, 6), "cap": s.cap, "calls": s.calls,
                "server_key": bool(os.environ.get("ANTHROPIC_API_KEY")) or scripted()}

    @app.post("/api/key/check")
    def key_check(x_api_key: str = Header(None)):
        """Does Anthropic accept this key? Uses the free model-list call, so it costs nothing."""
        client = client_for(x_api_key)
        if not scripted():
            guarded(lambda: client.models.list(limit=1))
        return {"ok": True}

    # Chats

    @app.get("/api/chats")
    def chats(x_user: str = Header(None), x_org: str = Header(None)):
        mem, _, agents = caller(x_user, x_org)
        return {"chats": mem.chats(agents["researcher"])}

    @app.post("/api/chats")
    def new_chat(x_user: str = Header(None), x_org: str = Header(None)):
        mem, _, agents = caller(x_user, x_org)
        chat = mem.start_session(agents["researcher"], team=[agents["planner"], agents["writer"]])
        return {"id": chat}

    @app.get("/api/chats/{chat}")
    def history(chat: str, x_user: str = Header(None), x_org: str = Header(None)):
        mem, _, agents = caller(x_user, x_org)
        return guarded(lambda: {"id": chat, "messages": mem.history(agents["researcher"], chat)})

    @app.post("/api/chats/{chat}/messages")
    def send(chat: str, body: Text, x_user: str = Header(None), x_org: str = Header(None),
             x_api_key: str = Header(None)):
        """One user message through the team. Returns the reply and every agent's steps."""
        mem, owner, agents = caller(x_user, x_org)
        client = client_for(x_api_key)
        return guarded(lambda: llm.team_turn(team(mem, agents, chat), body.text, client=client,
                                             spend=spend_for(key_for(x_api_key)), owner=owner))

    # The panels

    @app.get("/api/agents")
    def agents_(x_user: str = Header(None), x_org: str = Header(None)):
        mem, _, agents = caller(x_user, x_org)
        return {"agents": mem.agents(), "mine": agents}

    @app.get("/api/graph")
    def graph(x_user: str = Header(None), x_org: str = Header(None)):
        mem, _, agents = caller(x_user, x_org)
        return mem.graph(agents["researcher"])

    @app.get("/api/flagged")
    def flagged(x_user: str = Header(None), x_org: str = Header(None)):
        mem, _, agents = caller(x_user, x_org)
        return {"flagged": {r: mem.flagged(a) for r, a in agents.items()}}

    @app.post("/api/tools/{name}")
    def tool(name: str, body: ToolCall, x_user: str = Header(None), x_org: str = Header(None)):
        """A read-only button on the page (Explain, Blast radius...). It runs as one of the caller's
        own agents, so it sees what that agent may see. Tools that change memory are refused."""
        mem, _, agents = caller(x_user, x_org)
        if body.role not in agents:
            raise HTTPException(400, f"role must be one of {list(ROLES)}")
        if name not in READ_TOOLS:
            raise HTTPException(403, f"{name!r} is not available on the page: the agents change "
                                     f"memory themselves, from the chat")
        chat = body.chat or next((c["id"] for c in mem.chats(agents["researcher"])), None) \
            or mem.start_session(agents["researcher"], team=[agents["planner"], agents["writer"]])
        tools = guarded(lambda: team(mem, agents, chat))[body.role]
        out = tools.call(name, body.args)
        if "error" in out:
            raise HTTPException(400, out["error"])
        return out

    # The page

    if STATIC.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def page():
        index = STATIC / "index.html"
        if not index.is_file():
            raise HTTPException(404, "the page is missing (frontend/index.html)")
        return FileResponse(index)

    return app


app = create_app()
