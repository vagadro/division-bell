"""The tools an LLM agent can use on the shared memory. Defined once, used twice:

  - the demo agents call them through the Anthropic API (anthropic_tools() + call())
  - mcp_server.py registers the same methods with an MCP server (Claude Code / Desktop)

Who is calling is fixed when AgentTools is built (agent + session), never typed by the model,
so a model cannot act as someone else. Errors come back as {"error": "..."} so the model can
read them and adapt, instead of crashing the tool.

Each public method below is one tool: its docstring is the description the model sees, and the
type hints (with Field descriptions) become the argument schema. The method names are listed in
TOOL_NAMES; setup and admin calls (register_agent, start_session, drop) are deliberately not tools.
"""
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from pydantic import Field, TypeAdapter, ValidationError, validate_call

import documents
import llm
from documents import DocumentError
from memory import MemoryGraphError

TOOL_NAMES = (
    "remember", "recall", "publish", "supersede", "correct", "retract", "confirm",
    "decide", "explain", "blast_radius", "flagged",
    "handoff", "inbox", "learn_procedure", "recall_procedures", "verify_fact",
)

Text = Annotated[str, Field(description="plain text")]
FactId = Annotated[str, Field(description="id of a fact, like fact:3")]
ItemId = Annotated[str, Field(description="id of a fact, decision or procedure")]
Entity = Annotated[str, Field(description="stable id of the thing it is about, like org:acme")]
Sources = Annotated[list[str], Field(description="ids of the shared facts/decisions this is built on")]


class AgentTools:
    def __init__(self, memory, agent, session, checker=None, docs_dir=None):
        self.memory, self.agent, self.session = memory, agent, session
        self.checker = checker      # built on first use, so no API key is needed until then
        self.docs_dir = Path(docs_dir).resolve() if docs_dir else None

    @classmethod
    def open(cls, memory, name, owner=None, checker=None, docs_dir=None):
        """Register the agent (if new) and start its session."""
        agent = memory.register_agent(name, owner=owner)
        return cls(memory, agent, memory.start_session(agent), checker, docs_dir)

    # --- facts ---

    def remember(self, text: Text, entity: Entity,
                 entity_name: Annotated[str | None, Field(description="readable name of the entity")] = None,
                 confidence: Annotated[float, Field(ge=0, le=1, description="how sure you are, 0 to 1")] = 0.5,
                 derived_from: Sources = ()) -> dict:
        """Write down a fact as a private note. Only you (and your user's other agents) see it
        until you publish it. If it relies on other facts, pass their ids in derived_from so a
        later correction can find it."""
        fact = self.memory.remember(self.agent, self.session, text, entity, entity_name=entity_name,
                                    confidence=confidence, derived_from=list(derived_from))
        return {"id": fact}

    def recall(self, entity: Entity,
               as_of: Annotated[int | None, Field(description="epoch ms; omit for what is true now")] = None) -> dict:
        """What is currently known about an entity. Each fact shows its author and the author's
        trust, most trusted first. needs_verification means the author has a poor record and
        no outside evidence yet: check it before relying on it."""
        return {"facts": self.memory.recall(self.agent, entity, as_of=as_of)}

    def publish(self, fact_id: FactId) -> dict:
        """Share one of your private facts with the whole team. Its sources must be shared first."""
        self.memory.publish(self.agent, fact_id)
        return {"published": fact_id}

    def supersede(self, old_id: FactId, text: Text,
                  confidence: Annotated[float, Field(ge=0, le=1)] = 0.5) -> dict:
        """A fact was true and then CHANGED (not wrong). Replace it and keep the old one as history.
        Things built on it are flagged as 'changed' and can be confirmed. Only the author's user can.
        Returns the new fact's id as 'new', and what was flagged."""
        return self.memory.supersede(self.agent, self.session, old_id, text, confidence=confidence)

    def correct(self, old_id: FactId, text: Text,
                reason: Annotated[str, Field(description="why the old fact was wrong")] = "") -> dict:
        """One of your facts was WRONG. Retract it and write the right one in one step. Everything
        built on the wrong fact is flagged to be redone. Returns the new fact's id as 'new', and
        what was flagged."""
        return self.memory.correct(self.agent, self.session, old_id, text, reason=reason)

    def retract(self, fact_id: FactId,
                reason: Annotated[str, Field(description="why it was wrong")] = "") -> dict:
        """One of your facts was WRONG and has no replacement. Flags everything built on it.
        This lowers the trust record of the agent that wrote it. Use supersede if it merely changed."""
        return self.memory.retract(self.agent, fact_id, reason=reason)

    def confirm(self, item_id: ItemId) -> dict:
        """You reviewed an item that was flagged only because a source CHANGED, and it is still fine:
        make it active again. Items flagged as wrong must be redone instead."""
        self.memory.confirm(self.agent, item_id)
        return {"confirmed": item_id}

    # --- lineage ---

    def decide(self, text: Text, derived_from: Sources) -> dict:
        """Record a decision built on shared facts or decisions, so it can be traced and flagged
        if one of them turns out wrong."""
        return {"id": self.memory.decide(self.agent, self.session, text, derived_from=list(derived_from))}

    def explain(self, item_id: ItemId) -> dict:
        """The chain of facts and agents behind an item, nearest first, with each link's status."""
        return {"chain": self.memory.explain(self.agent, item_id)}

    def blast_radius(self, item_id: ItemId) -> dict:
        """Dry run: everything built on an item, and who wrote it. Changes nothing."""
        return {"affected": self.memory.blast_radius(self.agent, item_id)}

    def flagged(self) -> dict:
        """Your items that need attention. Each lists 'wrong' (sources that were wrong: redo the
        item) and 'changed' (sources that changed, with the fact now replacing them: review it,
        then confirm it or redo it)."""
        return {"items": self.memory.flagged(self.agent)}

    # --- coordination ---

    def handoff(self, recipient: Annotated[str, Field(description="agent id, like agent:sam-planner")],
                item_ids: Annotated[list[str], Field(min_length=1)],
                note: Annotated[str, Field(description="what the recipient should do with them")] = "") -> dict:
        """Pass active facts or decisions to another agent. It never shares anything: publish a
        private item first if the recipient belongs to another user."""
        return {"handoff": self.memory.handoff(self.agent, recipient, list(item_ids), note=note)}

    def inbox(self) -> dict:
        """What other agents handed to you, newest first, with each item's current status."""
        return {"handoffs": self.memory.inbox(self.agent)}

    def learn_procedure(self, name: Annotated[str, Field(description="short name of the how-to")],
                        steps: Annotated[list[str], Field(min_length=1, description="ordered steps")],
                        entity: Entity, derived_from: Sources = ()) -> dict:
        """Save a how-to you worked out, as a private procedure about an entity or task type."""
        return {"id": self.memory.learn_procedure(self.agent, self.session, name, list(steps), entity,
                                                  derived_from=list(derived_from))}

    def recall_procedures(self, entity: Entity) -> dict:
        """Saved how-tos about an entity or task type that you may see."""
        return {"procedures": self.memory.recall_procedures(self.agent, entity)}

    # --- evidence ---

    def verify_fact(self, fact_id: FactId,
                    method: Annotated[str, Field(pattern="^(web|doc)$",
                                                 description="'web' searches the web; 'doc' reads a document")] = "web",
                    document: Annotated[str | None, Field(
                        description="file name inside the documents folder; needed for method 'doc'")] = None) -> dict:
        """Check a fact against outside evidence, not against another agent's opinion. Only a
        'supported' result clears a low-trust warning and raises the author's trust. Web checks
        cost money and are capped, so use them only when it matters."""
        doc = None
        if method == "doc":
            doc = documents.extract(self._document_path(document))
        if self.checker is None:
            self.checker = llm.HaikuChecker()
        out = llm.check_and_record(self.memory, self.agent, fact_id, method, self.checker, document=doc)
        out["verdict"] = asdict(out["verdict"])
        return out

    def _document_path(self, name):
        if not self.docs_dir:
            raise DocumentError("no documents folder is configured, so 'doc' checks are unavailable")
        if not name:
            raise DocumentError("method 'doc' needs a document file name")
        path = (self.docs_dir / name).resolve()
        if not path.is_relative_to(self.docs_dir):
            raise DocumentError(f"{name!r} is outside the documents folder")
        return path

    # --- the two ways to use the list ---

    def call(self, name, args=None):
        """Run one tool by name. Always returns a dict; failures become {"error": ...}."""
        if name not in TOOL_NAMES:
            return {"error": f"unknown tool {name!r}"}
        try:
            result = validate_call(getattr(self, name))(**(args or {}))
        except ValidationError as e:
            return {"error": "bad arguments: " + "; ".join(
                f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())}
        except (MemoryGraphError, DocumentError, llm.LLMBudgetError) as e:
            return {"error": str(e)}
        return result if isinstance(result, dict) else {"result": result}

    def anthropic_tools(self):
        """The tool list in the Anthropic API format. The last tool carries a cache marker, so the
        whole list is cached between calls."""
        tools = [{"name": n, "description": _description(getattr(self, n)),
                  "input_schema": _schema(getattr(self, n))} for n in TOOL_NAMES]
        tools[-1]["cache_control"] = {"type": "ephemeral"}
        return tools


def _description(method):
    return " ".join(method.__doc__.split())


def _schema(method):
    schema = TypeAdapter(method).json_schema()

    def clean(node):  # drop the auto-generated titles: they only add tokens
        if isinstance(node, dict):
            return {k: clean(v) for k, v in node.items() if k not in ("title", "additionalProperties")}
        if isinstance(node, list):
            return [clean(v) for v in node]
        return node
    return clean(schema)
