"""HTTP API for the demo page. A thin layer: every route calls team_turn(), AgentTools or a
Memory read view, so the memory rules apply to the page exactly as to agents and tests.

Who is calling comes from two headers, X-User (a name) and X-Org (DemoOrg1-3). Nothing is
authenticated: anyone who sends the same name and org is that user. The server binds to
127.0.0.1 only. Each user gets three agents (Researcher, Planner, Writer) on first visit.

Mode: MEMORY_LLM=mock (default, EchoLLM, $0) or MEMORY_LLM=haiku (the real API, spend-capped).
Run: uv run uvicorn server:app --host 127.0.0.1 --port 8000
"""
import os
import re
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import llm
from documents import DocumentError
from memory import MemoryGraphError, Tenants
from tools import TOOL_NAMES, AgentTools

ORGS = ("DemoOrg1", "DemoOrg2", "DemoOrg3")
ROLES = ("researcher", "planner", "writer")
STATIC = Path(__file__).resolve().parent / "static"
NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9 ]{0,30}")


class Text(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class ToolCall(BaseModel):
    role: str = "researcher"      # which of the caller's own agents runs the tool
    chat: str | None = None       # the chat it runs in (needed by tools that write)
    args: dict = {}


def create_app(tenants=None, client=None, mode=None, spend_cap=llm.SPEND_CAP, docs_dir=None):
    """Build the app. Tests pass their own tenants (a test prefix) and a scripted client."""
    mode = mode or os.environ.get("MEMORY_LLM", "mock")
    if mode not in ("mock", "haiku"):
        raise ValueError(f"MEMORY_LLM must be 'mock' or 'haiku', not {mode!r}")
    state = {"tenants": tenants, "client": client, "spend": llm.Spend(cap=spend_cap)}
    docs_dir = docs_dir or os.environ.get("MEMORY_DOCS_DIR")
    app = FastAPI(title="GraphHacks shared agent memory")

    def tenants_():
        if state["tenants"] is None:
            state["tenants"] = Tenants()
        return state["tenants"]

    def llm_client():
        if state["client"] is None:
            if mode == "mock":
                state["client"] = llm.EchoLLM()
            else:
                import anthropic
                state["client"] = anthropic.Anthropic()
        return state["client"]

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

    # --- who am I ---

    @app.get("/api/orgs")
    def orgs():
        return {"orgs": list(ORGS)}

    @app.post("/api/enter")
    def enter(x_user: str = Header(None), x_org: str = Header(None)):
        """First visit (or return): make sure the user's agents exist; list their chats."""
        mem, owner, agents = caller(x_user, x_org)
        return {"user": owner, "org": x_org, "agents": agents, "chats": mem.chats(agents["researcher"])}

    @app.get("/api/status")
    def status():
        s = state["spend"]
        return {"mode": mode, "spend": round(s.total, 6), "cap": s.cap, "calls": s.calls}

    # --- chats ---

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
    def send(chat: str, body: Text, x_user: str = Header(None), x_org: str = Header(None)):
        """One user message through the team. Returns the reply and every agent's steps."""
        mem, owner, agents = caller(x_user, x_org)
        return guarded(lambda: llm.team_turn(team(mem, agents, chat), body.text, client=llm_client(),
                                             spend=state["spend"], owner=owner))

    # --- the panels ---

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
        """A button on the page (Retract, Correct, Confirm, Publish, Blast radius, Explain...).
        It runs as one of the caller's own agents, so the same rules apply as for the model."""
        mem, _, agents = caller(x_user, x_org)
        if body.role not in agents:
            raise HTTPException(400, f"role must be one of {list(ROLES)}")
        if name not in TOOL_NAMES:
            raise HTTPException(404, f"unknown tool {name!r}")
        chat = body.chat or next((c["id"] for c in mem.chats(agents["researcher"])), None) \
            or mem.start_session(agents["researcher"], team=[agents["planner"], agents["writer"]])
        tools = guarded(lambda: team(mem, agents, chat))[body.role]
        out = tools.call(name, body.args)
        if "error" in out:
            raise HTTPException(400, out["error"])
        return out

    # --- the page ---

    if STATIC.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def page():
        index = STATIC / "index.html"
        if not index.is_file():
            raise HTTPException(404, "the page is not built yet (static/index.html)")
        return FileResponse(index)

    return app


app = create_app()
