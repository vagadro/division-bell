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
    "save_web_fact", "check_fact", "read_document", "search_history",
)
READ_DOC_CHARS = 12_000  # most document text one read_document call returns

Text = Annotated[str, Field(description="plain text")]
FactId = Annotated[str, Field(description="id of a fact, like fact:3")]
ItemId = Annotated[str, Field(description="id of a fact, decision or procedure")]
Entity = Annotated[str, Field(description="stable id of the thing it is about, like org:acme")]
AlsoAbout = Annotated[list[str], Field(description=(
    "other entity ids the same fact is about, like [\"org:google\"] for a fact about the user's "
    "job. One fact is found from each of them; never save a copy under another entity"))]
Sources = Annotated[list[str], Field(description="ids of the shared facts/decisions this is built on")]
DistinctFrom = Annotated[list[str], Field(description=(
    "ids of similar saved facts you looked at and found to be about something else; "
    "only after a save answered saved: false"))]
SIMILAR_HINT = ("Not saved: similar facts exist. Read them. Same claim: don't save it again. Yours and "
                "newer (it changed): supersede it. Yours and wrong: correct it. Another user's and "
                "your search shows it is wrong or out of date: check_fact it as contradicted, then "
                "save yours with distinct_from. A different claim: save again with distinct_from.")


class AgentTools:
    def __init__(self, memory, agent, session, checker=None, docs_dir=None):
        self.memory, self.agent, self.session = memory, agent, session
        self.owner = memory.owner_of(agent)
        # facts about the user live under this entity and are never shared automatically
        self.user_entity = self.owner if self.owner.startswith("user:") else None
        self.checker = checker      # built on first use, so no API key is needed until then
        self.docs_dir = Path(docs_dir).resolve() if docs_dir else None
        # Set by llm.chat_turn for the length of one user message, never by the model:
        self.current_message = None  # facts saved now are linked to this message
        self.web_urls = set()        # URLs the web search returned in this turn

    @classmethod
    def open(cls, memory, name, owner=None, checker=None, docs_dir=None):
        """Register the agent (if new) and start its session."""
        agent = memory.register_agent(name, owner=owner)
        return cls(memory, agent, memory.start_session(agent), checker, docs_dir)

    # Facts

    def remember(self, text: Text, entity: Entity,
                 entity_name: Annotated[str | None, Field(description="readable name of the entity")] = None,
                 confidence: Annotated[float, Field(ge=0, le=1, description="how sure you are, 0 to 1")] = 0.5,
                 derived_from: Sources = (),
                 source: Annotated[str, Field(pattern="^(user|document|agent)$", description=(
                     "'user' if the user told you, 'document' if you read it in a document, "
                     "'agent' if you worked it out yourself"))] = "user",
                 also_about: AlsoAbout = (), distinct_from: DistinctFrom = ()) -> dict:
        """Write down a fact as a private note. Only you (and your user's other agents) see it
        until you publish it. If it relies on other facts, pass their ids in derived_from so a
        later correction can find it. Say honestly where it came from: if the user later
        changes something they told you, that is not counted as your mistake. If a similar fact is
        already saved, nothing is saved and the similar facts come back (saved: false)."""
        similar = self._similar(text, [entity, *also_about], distinct_from)
        if similar:
            return similar
        fact = self.memory.remember(self.agent, self.session, text, entity, entity_name=entity_name,
                                    confidence=confidence, derived_from=list(derived_from),
                                    from_messages=[self.current_message] if self.current_message else (),
                                    source=source, also_about=list(also_about))
        return {"id": fact}

    def _similar(self, text, entities, distinct_from):
        """The not-saved answer if facts like this one exist (and were not ruled out), else None.
        Drafts (doc:... entities) are not checked: a draft repeats the facts it is written from."""
        if any(e.startswith("doc:") for e in entities):
            return None
        found = [f for f in self.memory.similar_facts(self.agent, text, entities)
                 if f["id"] not in set(distinct_from)]
        return {"saved": False, "similar": found, "what_to_do": SIMILAR_HINT} if found else None

    def save_web_fact(self, text: Text, entity: Entity,
                      sources: Annotated[list[str], Field(
                          min_length=1, description="URLs from this turn's web search results")],
                      entity_name: Annotated[str | None, Field(description="readable name of the entity")] = None,
                      also_about: AlsoAbout = (), distinct_from: DistinctFrom = ()) -> dict:
        """Save a fact you just found with web_search. It is shared with the whole team at once and
        recorded as checked against those web pages. Cite only URLs the search returned; run
        web_search first. If a similar fact is already saved, nothing is saved and the similar
        facts come back (saved: false)."""
        cited = list(dict.fromkeys(sources))
        found = [u for u in cited if u in self.web_urls]
        if not found:
            raise MemoryGraphError("none of these URLs came back from a web search in this turn; "
                                   "search first, then cite URLs from the results")
        saved = self.remember(text, entity, entity_name=entity_name, confidence=0.9, source="web",
                              also_about=also_about, distinct_from=distinct_from)
        if "id" not in saved:
            return saved
        fact = saved["id"]
        # three steps, not one query: if one fails, the fact is still there as a private note
        self.memory.publish(self.agent, fact)
        self.memory.verify(self.agent, fact, "web", "supported", found)
        return {"id": fact, "shared": True, "sources": found,
                "ignored": [u for u in cited if u not in found]}

    def check_fact(self, fact_id: FactId,
                   result: Annotated[str, Field(pattern="^(supported|contradicted)$", description=(
                       "'supported' if the search results back the fact, 'contradicted' if they "
                       "show it is wrong or out of date"))],
                   sources: Annotated[list[str], Field(
                       min_length=1, description="URLs from this turn's web_search results")],
                   note: Annotated[str, Field(description=(
                       "what the sources say, in one sentence (for contradicted: the right or newer value)"))] = "") -> dict:
        """Record what your web_search showed about a saved fact (anyone's). Supported: the fact
        counts as verified, and a private fact that is not about the user is then shared with the
        team (it is public knowledge). Contradicted: the fact counts as doubtful for everyone; if
        it is your user's, fix it now with supersede or correct; if it is another user's, its
        owner's agents are told to look at it."""
        cited = list(dict.fromkeys(sources))
        found = [u for u in cited if u in self.web_urls]
        if not found:
            raise MemoryGraphError("none of these URLs came back from a web search in this turn; "
                                   "search first, then cite URLs from the results")
        self.memory.verify(self.agent, fact_id, "web", result, found, snippet=note)
        info = self.memory.fact_info(fact_id)
        mine = info["owner"] == self.owner
        out = {"recorded": result, "fact": fact_id, "sources": found}
        if result == "supported" and mine and info["visibility"] == "private":
            if self.user_entity in info["entities"]:
                out["shared"] = False  # about the user: stays private whatever the web says
            else:
                try:
                    self.memory.publish(self.agent, fact_id)
                    out["shared"] = True
                except MemoryGraphError as e:  # built on private items: stays private
                    out["shared"], out["why_not_shared"] = False, str(e)
        elif result == "contradicted":
            out["next"] = ("yours: fix it now with supersede (it changed) or correct (it was wrong)" if mine
                           else "another user's: its owner's agents will see this; save your own value "
                                "with save_web_fact and distinct_from")
        return out

    def recall(self, entity: Entity,
               as_of: Annotated[int | None, Field(description="epoch ms; omit for what is true now")] = None) -> dict:
        """What is currently known about an entity. Each fact has a standing: verified (outside
        evidence supports it), user (the user said it), unverified (no evidence, reliable author)
        or doubtful (evidence contradicts it, or its author is often wrong). needs_verification
        means doubtful: check it before relying on it. Best standing first."""
        return {"facts": self.memory.recall(self.agent, entity, as_of=as_of)}

    def publish(self, item_id: ItemId) -> dict:
        """Share one of your private facts or decisions with every user in the organisation. Its
        sources must be shared first. Only do this when the user asks you to share something."""
        self.memory.publish(self.agent, item_id)
        return {"published": item_id}

    def supersede(self, old_id: FactId, text: Text,
                  confidence: Annotated[float, Field(ge=0, le=1)] = 0.5,
                  sources: Annotated[list[str], Field(description=(
                      "URLs from this turn's web_search that back the new value, if it came "
                      "from the web; the new fact is then recorded as checked"))] = (),
                  derived_from: Sources = ()) -> dict:
        """A fact was true and then CHANGED (not wrong). Replace it and keep the old one as history.
        Things built on it are flagged as 'changed' and can be confirmed. Only the author's user can.
        Returns the new fact's id as 'new', and what was flagged."""
        cited = list(dict.fromkeys(sources))
        found = [u for u in cited if u in self.web_urls]
        if cited and not found:  # checked first, so a bad citation changes nothing
            raise MemoryGraphError("none of these URLs came back from a web search in this turn; "
                                   "search first, then cite URLs from the results")
        report = self.memory.supersede(self.agent, self.session, old_id, text, confidence=confidence,
                                       derived_from=list(derived_from))
        if found:
            self.memory.verify(self.agent, report["new"], "web", "supported", found)
            report["sources"] = found
        return report

    def correct(self, old_id: FactId, text: Text,
                reason: Annotated[str, Field(description="why the old fact was wrong")] = "") -> dict:
        """One of your facts was WRONG. Retract it and write the right one in one step. Everything
        built on the wrong fact is flagged to be redone. Returns the new fact's id as 'new', and
        what was flagged."""
        return self.memory.correct(self.agent, self.session, old_id, text, reason=reason)

    def retract(self, fact_id: FactId,
                reason: Annotated[str, Field(description="why it was wrong")] = "") -> dict:
        """One of your facts was WRONG and has no replacement. Flags everything built on it.
        It counts as a mistake for the agent that wrote it, unless the user was the source. Use
        supersede if it merely changed."""
        return self.memory.retract(self.agent, fact_id, reason=reason)

    def confirm(self, item_id: ItemId) -> dict:
        """You reviewed an item that was flagged only because a source CHANGED, and it is still fine:
        make it active again. Items flagged as wrong must be redone instead."""
        self.memory.confirm(self.agent, item_id)
        return {"confirmed": item_id}

    # Lineage

    def decide(self, text: Text, derived_from: Sources,
               replaces: Annotated[list[str], Field(description=(
                   "ids of flagged decisions this one redoes; they are closed as superseded"))] = ()) -> dict:
        """Record a decision built on facts or decisions, so it can be traced and flagged if one of
        them turns out wrong. It is private to your user (publish shares it). To redo a flagged
        decision, pass it in replaces."""
        return {"id": self.memory.decide(self.agent, self.session, text, derived_from=list(derived_from),
                                         replaces=list(replaces))}

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

    # Coordination

    def handoff(self, recipient: Annotated[str, Field(description="agent id, like agent:sam-planner")],
                item_ids: Annotated[list[str], Field(min_length=1)],
                note: Annotated[str, Field(description="what the recipient should do with them")] = "") -> dict:
        """Pass facts or decisions to another agent. Flagged ones may be passed too, to ask a
        teammate to redo or review them. It never shares anything: publish a private item first
        if the recipient belongs to another user."""
        return {"handoff": self.memory.handoff(self.agent, recipient, list(item_ids), note=note)}

    def search_history(self, query: Annotated[str, Field(
                           description="words to look for, including similar words, like: salary income pay earn")]) -> dict:
        """Search what your user said in earlier chats (this chat is already in front of you).
        Any of the words may match; messages with all of them come first. Matches words, not
        meaning, so include similar words. Use it for things that were said but may not have
        been saved as facts; recall saved facts first."""
        return {"messages": self.memory.search_history(self.agent, query, exclude_session=self.session)}

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

    # Evidence

    def verify_fact(self, fact_id: FactId,
                    method: Annotated[str, Field(pattern="^(web|doc)$",
                                                 description="'web' searches the web; 'doc' reads a document")] = "web",
                    document: Annotated[str | None, Field(
                        description="file name inside the documents folder; needed for method 'doc'")] = None) -> dict:
        """Check a fact against outside evidence, not against another agent's opinion. Only a
        'supported' result makes the fact verified; 'contradicted' makes it doubtful. Web checks
        cost money and are capped, so use them only when it matters."""
        doc = None
        if method == "doc":
            doc = documents.extract(self._document_path(document))
        if self.checker is None:
            self.checker = llm.HaikuChecker()
        out = llm.check_and_record(self.memory, self.agent, fact_id, method, self.checker, document=doc)
        out["verdict"] = asdict(out["verdict"])
        return out

    def read_document(self, document: Annotated[str, Field(
                          description="file name inside the documents folder")],
                      query: Annotated[str, Field(
                          description="what you are looking for; picks the passages of a long document")] = "") -> dict:
        """Read an uploaded document. Each passage says where it came from (page, sheet, row).
        Facts you take from it are the user's: save them with remember, which keeps them private."""
        doc = documents.extract(self._document_path(document))
        passages = documents.select_passages(doc, query, max_chars=READ_DOC_CHARS)
        if not passages:
            return {"document": doc.name, "passages": [],
                    "note": "the document is long and no passage matched the query; try other words"}
        return {"document": doc.name, "passages": [{"label": p.label, "text": p.text} for p in passages],
                "complete": len(passages) == len(doc.passages)}

    def _document_path(self, name):
        if not self.docs_dir:
            raise DocumentError("no documents folder is configured, so 'doc' checks are unavailable")
        if not name:
            raise DocumentError("method 'doc' needs a document file name")
        path = (self.docs_dir / name).resolve()
        if not path.is_relative_to(self.docs_dir):
            raise DocumentError(f"{name!r} is outside the documents folder")
        return path

    # The two ways to use the list

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
