"""Shared agent memory on FalkorDB. Wraps the Cypher in queries.py and enforces the rules.

Owners: an agent can belong to a user (`register_agent(name, owner="Sam")`). A user's agents share
that user's private memory and may change each other's facts; agents of other users may not.
An agent without an owner owns itself.

Visibility: a fact is `private` (only its owner's agents see it) until it is published.
An agent may only build on (derive from) active things it can see, and may only publish a fact
once everything it was built on is shared, so shared lineage never points at hidden items.
A decision is as visible as its sources: private if any source is private, else shared.

Changes: only the author's owner can supersede, correct or retract a fact, so one user's memory
never changes because of another user's edit. Items built on a changed fact are flagged
("changed") and items built on a wrong fact are flagged ("wrong"), so their owners find out.
"""
import os
import re
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from falkordb import FalkorDB
from redis.exceptions import ResponseError

import queries

INDEXED_LABELS = ("Fact", "Decision", "Agent", "Entity", "Handoff", "Message", "Procedure")
MESSAGE_ROLES = ("user", "agent", "tool")
# Trust is judged per fact first: a fact with a supported evidence check is trusted whoever wrote
# it. Only a fact with no evidence falls back on its author's accuracy, which is a rate, not a
# counter: (claims - mistakes + PRIOR_CORRECT) / (claims + PRIOR_CORRECT). A new agent counts as
# if it had PRIOR_CORRECT correct claims (so it starts at 1.0), and 3 mistakes in 100 claims
# barely move it (0.97) while 3 in 3 do (0.57).
PRIOR_CORRECT = 4
ACCURACY_WARN = 0.75  # an unchecked fact by an author below this accuracy needs verification
FACT_SOURCES = ("user", "web", "document", "agent")  # "user" facts never count for or against an agent
STANDING_ORDER = {"verified": 0, "user": 1, "unverified": 2, "doubtful": 3}
VERIFY_METHODS = ("web", "doc")  # evidence must come from outside the agents: a web search or a document
VERIFY_RESULTS = ("supported", "contradicted", "unclear")


class MemoryGraphError(Exception):
    """A memory rule was broken (unknown agent, hidden source, not the owner, ...)."""


def _now():
    return int(time.time() * 1000)


def _new_id(prefix):
    return f"{prefix}:{uuid.uuid4().hex[:8]}"


def _slug(text):
    return text.strip().lower().replace(" ", "-")


def _connect():
    load_dotenv(Path(__file__).resolve().parent / ".env")
    return FalkorDB(
        host=os.environ["FALKORDB_HOST"],
        port=int(os.environ["FALKORDB_PORT"]),
        username=os.environ["FALKORDB_USERNAME"],
        password=os.environ["FALKORDB_PASSWORD"],
    )


class Memory:
    def __init__(self, graph="memory", db=None):
        if db is None:
            db = _connect()
        self.db = db
        self.name = graph
        self.g = db.select_graph(graph)

    # --- admin ---

    def setup(self):
        """Create the id indexes and the full-text index on message text. Safe to call repeatedly."""
        statements = [f"CREATE INDEX FOR (n:{label}) ON (n.id)" for label in INDEXED_LABELS]
        statements.append(queries.CREATE_MESSAGE_TEXT_INDEX)
        for statement in statements:
            try:
                self.g.query(statement)
            except ResponseError as e:
                if "already indexed" not in str(e).lower():
                    raise

    def drop(self):
        """Delete the whole graph (used by tests)."""
        if self.name in self.db.list_graphs():
            self.g.delete()

    # --- agents and sessions ---

    def register_agent(self, name, owner=None):
        """Create (or find) an agent. With an owner, the id includes it (agent:sam-researcher), so
        two users can each have a "Researcher". Safe to call repeatedly."""
        if owner:
            agent_id, owner_id = f"agent:{_slug(owner)}-{_slug(name)}", f"user:{_slug(owner)}"
        else:
            agent_id = f"agent:{_slug(name)}"
            owner_id = agent_id  # an agent without a user owns itself
        self.g.query(
            "MERGE (a:Agent {id:$id}) ON CREATE SET a.name = $name, a.owner = $owner",
            {"id": agent_id, "name": name, "owner": owner_id},
        )
        return agent_id

    def start_session(self, agent, team=()):
        """Start a session (one chat). `team`: other agents of the same owner that share it, so
        a user's agents all read and write one conversation. Returns the session id."""
        session_id = _new_id("sess")
        res = self.g.query(
            "MATCH (a:Agent {id:$agent}) CREATE (a)-[:RAN]->(:Session {id:$s, started_at:$now})",
            {"agent": agent, "s": session_id, "now": _now()},
        )
        if res.nodes_created == 0:
            raise MemoryGraphError(f"unknown agent {agent!r}")
        for other in team:
            self.join_session(other, session_id)
        return session_id

    def join_session(self, agent, session):
        """Let another agent of the same owner into a session. Agents of other users can't join."""
        if not self.g.query(queries.JOIN_SESSION, {"agent": agent, "session": session}).result_set:
            raise MemoryGraphError(f"{agent} can't join {session}: unknown, or it belongs to another user")

    def records(self, agents):
        """{agent id: {"claims", "mistakes", "accuracy"}} for these agents (unknown ids left out)."""
        rows = self.g.query(queries.AGENT_RECORDS, {"ids": list(dict.fromkeys(agents))}).result_set
        # FalkorDB's sum() returns a float, so both counts are made whole numbers here
        counts = {r[0]: (int(r[1]), int(r[2])) for r in rows}
        return {a: {"claims": c, "mistakes": m, "accuracy": (c - m + PRIOR_CORRECT) / (c + PRIOR_CORRECT)}
                for a, (c, m) in counts.items()}

    def accuracy(self, agent):
        """The agent's accuracy (1.0 for a new agent). Raises for an unknown agent."""
        return self._require_agent(agent)["accuracy"]

    def _require_agent(self, agent):
        record = self.records([agent]).get(agent)
        if record is None:
            raise MemoryGraphError(f"unknown agent {agent!r}")
        return record

    # --- visibility and source checks ---

    def _require_visible(self, agent, ids):
        """Raise unless every id is a fact/decision this agent may see."""
        ids = list(dict.fromkeys(ids))
        if not ids:
            return
        rows = self.g.query(queries.VISIBLE_ITEMS, {"me": agent, "ids": ids}).result_set
        found = {r[0] for r in rows}
        missing = [i for i in ids if i not in found]
        if missing:
            raise MemoryGraphError(f"not found, or not visible to {agent}: {missing}")

    def _require_sources(self, agent, ids):
        """Raise unless every id is an active fact/decision this agent may build on."""
        ids = list(dict.fromkeys(ids))
        if not ids:
            return
        rows = self.g.query(queries.USABLE_SOURCES, {"me": agent, "ids": ids}).result_set
        found = {r[0] for r in rows}
        missing = [i for i in ids if i not in found]
        if missing:
            why = "unknown, not visible to the agent, or no longer active (retracted, flagged, superseded)"
            raise MemoryGraphError(f"cannot build on {missing}: {why}")

    def _require_may_change(self, agent, item_id):
        """Raise unless this agent, or another agent of the same owner, wrote the item."""
        if not self.g.query(queries.MAY_CHANGE, {"me": agent, "id": item_id}).result_set:
            raise MemoryGraphError(
                f"only the author (or the author's other agents) can change {item_id}; "
                f"{agent} can write its own fact instead")

    # --- write ---

    def remember(self, agent, session, text, entity, entity_name=None, entity_type="thing",
                 confidence=0.5, derived_from=(), valid_from=None, from_messages=(), source="agent"):
        """Write a private fact. Returns its id. `from_messages` records which conversation
        messages it was taken from (evidence only: a wrong message can't exist, so it is not lineage).
        `source`: user (the user said it), web, document, or agent (the agent's own claim). Only
        non-user facts count toward the agent's accuracy."""
        if source not in FACT_SOURCES:
            raise MemoryGraphError(f"source must be one of {FACT_SOURCES}, not {source!r}")
        derived_from = list(dict.fromkeys(derived_from))
        from_messages = self._require_messages(agent, from_messages)
        self._require_sources(agent, derived_from)
        fact_id = _new_id("fact")
        now = _now()
        res = self.g.query(queries.REMEMBER, {
            "agent": agent, "session": session, "entity": entity,
            "entity_name": entity_name or entity, "entity_type": entity_type,
            "id": fact_id, "text": text, "conf": confidence,
            "valid_from": valid_from if valid_from is not None else now, "now": now,
            "derived_from": derived_from, "from_messages": from_messages, "source": source,
        })
        if res.nodes_created == 0:
            raise MemoryGraphError(f"agent {agent!r} did not run session {session!r} (or one is unknown), "
                f"or a source stopped being active while writing")
        return fact_id

    def decide(self, agent, session, text, derived_from=()):
        """Record a decision built on active items this agent may see. Returns its id. The decision
        is private (only its owner's agents see it) if any source is private, else shared."""
        derived_from = list(dict.fromkeys(derived_from))
        self._require_sources(agent, derived_from)
        decision_id = _new_id("dec")
        res = self.g.query(queries.DECIDE, {
            "agent": agent, "session": session, "id": decision_id, "text": text,
            "now": _now(), "derived_from": derived_from,
        })
        if res.nodes_created == 0:
            raise MemoryGraphError(f"agent {agent!r} did not run session {session!r} (or one is unknown), "
                f"or a source stopped being active while writing")
        return decision_id

    def publish(self, agent, fact_id):
        """Share a private fact, decision or procedure with everyone. Only its owner's agents, and
        only once its sources are shared."""
        rows = self.g.query(queries.PUBLISH_CHECK, {"agent": agent, "id": fact_id}).result_set
        if not rows:
            raise MemoryGraphError(f"{agent} has no active item {fact_id} of its own to publish")
        blockers = rows[0][1]
        if blockers:
            raise MemoryGraphError(f"cannot publish {fact_id}: built on private {sorted(blockers)}; publish those first")
        self.g.query(queries.PUBLISH, {"agent": agent, "id": fact_id})

    # --- conversation history (episodic memory) ---

    def _require_messages(self, agent, ids):
        """Return the distinct message ids, raising unless this agent may read every one."""
        ids = list(dict.fromkeys(ids))
        if ids:
            rows = self.g.query(queries.READABLE_MESSAGES, {"me": agent, "ids": ids}).result_set
            missing = [i for i in ids if i not in {r[0] for r in rows}]
            if missing:
                raise MemoryGraphError(f"messages not found, or not readable by {agent}: {missing}")
        return ids

    def add_message(self, agent, session, role, text):
        """Append a message to a session's conversation. Returns its id. Messages are private to
        the owning user's agents and are never shared or retracted."""
        if role not in MESSAGE_ROLES:
            raise MemoryGraphError(f"bad role {role!r}: use one of {MESSAGE_ROLES}")
        message_id = _new_id("msg")
        res = self.g.query(queries.ADD_MESSAGE, {
            "agent": agent, "session": session, "id": message_id, "role": role,
            "text": text, "now": _now()})
        if not res.result_set:
            raise MemoryGraphError(f"agent {agent!r} did not run session {session!r} (or one is unknown)")
        return message_id

    def history(self, agent, session):
        """The conversation of a session, in order. Readable by the owning user's agents only."""
        if not self.g.query(queries.SESSION_ACCESS, {"me": agent, "session": session}).result_set:
            raise MemoryGraphError(f"session {session!r} not found, or not readable by {agent}")
        rows = self.g.query(queries.HISTORY, {"session": session}).result_set
        return [{"id": r[0], "role": r[1], "text": r[2], "seq": r[3], "at": r[4], "agent": r[5]}
                for r in rows]

    def search_history(self, agent, query, limit=5, exclude_session=None):
        """Search the text of every chat this agent's user had (never another user's), best match
        first. Any of the words may match, and messages with all of them rank first; word forms
        match ("salaries" finds "salary"); "sal*" is a prefix. Words, not meaning: "earn" won't
        find "salary", so pass similar words too. Other characters are dropped, so a query can't
        break the search syntax."""
        terms = re.findall(r"\w+\*?", query or "")
        if not terms:
            raise MemoryGraphError("search_history needs at least one word")
        # "(a b)|a|b": a plain OR scores a message with one word the same as one with all of them,
        # so the all-words group is added to lift full matches to the top
        words = "|".join(terms) if len(terms) == 1 else f"({' '.join(terms)})|" + "|".join(terms)
        try:
            rows = self.g.query(queries.SEARCH_HISTORY, {
                "me": agent, "q": words, "limit": int(limit), "exclude": exclude_session}).result_set
        except ResponseError as e:
            raise MemoryGraphError(f"could not search for {words!r}: {e}") from e
        return [{"id": r[0], "role": r[1], "agent": r[2], "text": r[3], "at": r[4], "session": r[5],
                 "score": r[6]} for r in rows]

    # --- read views for the UI ---

    def agents(self):
        """Every agent in this org, with its owner and record (claims, mistakes, accuracy)."""
        rows = self.g.query(queries.LIST_AGENTS).result_set
        records = self.records([r[0] for r in rows]) if rows else {}
        return [{"id": r[0], "name": r[1], "owner": r[2], **records[r[0]]} for r in rows]

    def chats(self, agent, title_chars=60):
        """This agent's user's chats, most recent first. The title is the first user message."""
        rows = self.g.query(queries.LIST_CHATS, {"me": agent}).result_set
        short = lambda t: t if t is None or len(t) <= title_chars else t[:title_chars - 1].rstrip() + "…"
        return [{"id": r[0], "started_at": r[1], "title": short(r[2]), "messages": r[3], "last_at": r[4]}
                for r in rows]

    def graph(self, agent, limit=500):
        """What the graph panel draws for this agent: {"nodes", "edges"}. Only items it may see
        (shared, or its user's own), their entities, authors and verifiers, and the messages
        they came from if it may read them. Edges join only those nodes, so none leads to a
        hidden node."""
        nodes = {}
        for r in self.g.query(queries.GRAPH_ITEMS, {"me": agent, "limit": int(limit)}).result_set:
            nodes[r[0]] = {"id": r[0], "label": r[1], "text": r[2], "status": r[3], "visibility": r[4],
                           "author": r[5], "recorded_at": r[6]}
        items = list(nodes)
        neighbours = self.g.query(queries.GRAPH_NEIGHBOURS, {"ids": items}).result_set if items else []
        messages = [r[0] for r in neighbours if r[1] == "Message"]
        readable = {r[0] for r in self.g.query(
            queries.READABLE_MESSAGES, {"me": agent, "ids": messages}).result_set} if messages else set()
        for node_id, label, text, role in neighbours:
            if label != "Message" or node_id in readable:
                nodes.setdefault(node_id, {"id": node_id, "label": label, "text": text, "role": role})
        agent_info = {a["id"]: a for a in self.agents()}
        for node in list(nodes.values()):
            for a in {node.get("author"), node["id"] if node["label"] == "Agent" else None} - {None}:
                info = agent_info.get(a, {})
                nodes[a] = {"id": a, "label": "Agent", "text": info.get("name"),
                            "owner": info.get("owner"), "accuracy": info.get("accuracy"),
                            "claims": info.get("claims"), "mistakes": info.get("mistakes")}
        ids = list(nodes)
        edges = [{"source": r[0], "type": r[1], "target": r[2], **({"result": r[3]} if r[3] else {})}
                 for r in self.g.query(queries.GRAPH_EDGES, {"ids": ids}).result_set] if ids else []
        return {"nodes": list(nodes.values()), "edges": edges}

    def evidence(self, agent, item_id):
        """The conversation messages behind an item or anything it was built on. The text is
        shown only for messages this agent may read; others appear with text None."""
        self._require_visible(agent, [item_id])
        rows = self.g.query(queries.EVIDENCE, {"id": item_id}).result_set
        readable = {r[0] for r in self.g.query(
            queries.READABLE_MESSAGES, {"me": agent, "ids": [r[1] for r in rows]}).result_set} if rows else set()
        return [{"item": r[0], "message": r[1], "role": r[2], "seq": r[4], "session": r[5],
                 "text": r[3] if r[1] in readable else None} for r in rows]

    # --- procedures (procedural memory) ---

    @staticmethod
    def _check_steps(name, steps):
        steps = list(steps)
        if not steps or not all(isinstance(s, str) and s.strip() for s in steps):
            raise MemoryGraphError("a procedure needs at least one step, each a non-empty string")
        if name is not None and not name.strip():
            raise MemoryGraphError("a procedure needs a name")
        return steps

    def learn_procedure(self, agent, session, name, steps, entity, entity_name=None,
                        entity_type="thing", derived_from=()):
        """Save a private how-to: a name and ordered steps, about an entity (e.g. a task type).
        May be derived from facts/decisions, so it is flagged when they are retracted or changed.
        Nothing can be built on a procedure. Returns its id; share it with publish()."""
        steps = self._check_steps(name, steps)
        derived_from = list(dict.fromkeys(derived_from))
        self._require_sources(agent, derived_from)
        procedure_id = _new_id("proc")
        res = self.g.query(queries.LEARN_PROCEDURE, {
            "agent": agent, "session": session, "entity": entity,
            "entity_name": entity_name or entity, "entity_type": entity_type,
            "id": procedure_id, "name": name, "steps": steps, "now": _now(),
            "derived_from": derived_from})
        if not res.result_set:
            raise MemoryGraphError(
                f"agent {agent!r} did not run session {session!r} (or one is unknown), "
                f"or a source stopped being active while writing")
        return procedure_id

    def revise_procedure(self, agent, session, old_id, steps, name=None, derived_from=()):
        """Replace a procedure with a new version (the old one stays as history). Also the way to
        redo a flagged procedure. Only the author's owner can do it. The new version keeps the
        old one's visibility; a shared procedure can only be built on shared sources.
        Returns the new id."""
        steps = self._check_steps(name, steps)
        derived_from = list(dict.fromkeys(derived_from))
        self._require_visible(agent, [old_id])
        self._require_may_change(agent, old_id)
        self._require_sources(agent, derived_from)
        new_id = _new_id("proc")
        res = self.g.query(queries.REVISE_PROCEDURE, {
            "old_id": old_id, "new_id": new_id, "name": name, "steps": steps, "agent": agent,
            "session": session, "now": _now(), "derived_from": derived_from})
        if not res.result_set:
            raise MemoryGraphError(
                f"cannot revise {old_id}: not an active/flagged procedure, {agent!r} did not run "
                f"{session!r}, or a source is not active (a shared procedure needs shared sources)")
        return new_id

    def retire_procedure(self, agent, procedure_id, reason=""):
        """The procedure is obsolete or wrong. Only the author's owner. Not counted as a mistake."""
        self._require_visible(agent, [procedure_id])
        self._require_may_change(agent, procedure_id)
        res = self.g.query(queries.RETIRE_PROCEDURE, {
            "id": procedure_id, "me": agent, "reason": reason, "now": _now()})
        if not res.result_set:
            raise MemoryGraphError(f"cannot retire {procedure_id}: not an active/flagged procedure")

    def recall_procedures(self, agent, entity):
        """Active procedures about an entity that this agent may see (flagged ones are left out)."""
        rows = self.g.query(queries.RECALL_PROCEDURES, {"entity": entity, "me": agent}).result_set
        return [{"id": r[0], "name": r[1], "steps": r[2], "built_on": sorted(r[3])} for r in rows]

    def handoff(self, sender, recipient, item_ids, note=""):
        """Pass active facts/decisions from one agent to another. Returns the handoff id.

        Every item must be something both agents may see: shared, or private to a user who owns
        both agents. A handoff never shares anything, so to hand a private item to another user's
        agent, publish it first. If an item is later retracted, the recipient is among the agents
        reported by retract(). The recipient reads its handoffs with inbox().
        """
        item_ids = list(dict.fromkeys(item_ids))
        if not item_ids:
            raise MemoryGraphError("a handoff must carry at least one item")
        if sender == recipient:
            raise MemoryGraphError("an agent cannot hand off to itself")
        self._require_agent(sender), self._require_agent(recipient)  # raises if either is unknown
        self._require_sources(sender, item_ids)
        for who in (sender, recipient):
            rows = self.g.query(queries.VISIBLE_ITEMS, {"me": who, "ids": item_ids}).result_set
            hidden = [i for i in item_ids if i not in {r[0] for r in rows}]
            if hidden:
                raise MemoryGraphError(
                    f"cannot hand off {hidden}: private, and {who} may not see them "
                    f"(publish them first)")
        handoff_id = _new_id("handoff")
        res = self.g.query(queries.HANDOFF, {
            "sender": sender, "recipient": recipient, "ids": item_ids,
            "id": handoff_id, "note": note, "now": _now()})
        if res.nodes_created == 0:
            raise MemoryGraphError("handoff failed: an item stopped being active or visible while writing")
        return handoff_id

    def inbox(self, agent):
        """What was handed to this agent, newest first, with each item's current status."""
        self._require_agent(agent)  # raises if unknown
        rows = self.g.query(queries.INBOX, {"me": agent}).result_set
        return [{"id": r[0], "sender": r[1], "note": r[2], "at": r[3], "items": r[4]} for r in rows]

    def supersede(self, agent, session, old_id, text, confidence=0.5, at=None):
        """The fact was true and then changed: replace it, keeping the old one as history.

        Only the author's owner can do this. Items built on the old fact are flagged "changed" so
        their owners can review (confirm) or redo them; the author's accuracy is untouched, since
        nothing was wrong. Returns {"superseded", "new", "flagged", "hidden_flagged", "agents"}.
        """
        self._require_visible(agent, [old_id])
        self._require_may_change(agent, old_id)
        new_id = _new_id("fact")
        now = _now()
        res = self.g.query(queries.SUPERSEDE, {
            "old_id": old_id, "new_id": new_id, "text": text, "conf": confidence,
            "agent": agent, "session": session, "at": at if at is not None else now, "now": now,
        })
        if not res.result_set:
            raise MemoryGraphError(
                f"cannot supersede {old_id}: not an active fact, or {agent!r} did not run {session!r}")
        report = self._report(agent, old_id, res.result_set[0][2], "superseded")
        report["new"] = new_id
        return report

    def retract(self, agent, fact_id, reason=""):
        """Mark one of your facts wrong and flag everything built on it ("wrong").

        Only the author's owner can retract. It counts as a mistake in the author's accuracy (a
        reliability record, not a punishment), unless the user was the fact's source. Returns {"retracted", "flagged", "hidden_flagged", "agents"}:
        `flagged` lists the flagged items this agent may see, `hidden_flagged` counts private items
        of other users (their owners still see them in flagged()), `agents` are the ones to notify.
        """
        self._require_visible(agent, [fact_id])
        self._require_may_change(agent, fact_id)
        res = self.g.query(queries.RETRACT, {
            "id": fact_id, "by": agent, "reason": reason, "now": _now()})
        if not res.result_set:
            raise MemoryGraphError(f"cannot retract {fact_id}: not a fact, or already retracted/superseded")
        return self._report(agent, fact_id, res.result_set[0][1], "retracted")

    def correct(self, agent, session, old_id, text, confidence=0.5, reason=""):
        """One of your facts was wrong: retract it and write the right one, in a single step.

        Same effects as retract() (only the author's owner can do it; flags everything built on
        the old fact as "wrong"; a mistake for the author unless the user was the source), plus a
        new fact that keeps the old one's visibility, source and entity links and is valid from the same time (the mistake
        was in the record, not in the world). Returns retract()'s report plus {"new": id}.
        Use supersede() instead when the fact was true and then changed.
        """
        self._require_visible(agent, [old_id])
        self._require_may_change(agent, old_id)
        new_id = _new_id("fact")
        res = self.g.query(queries.CORRECT, {
            "old_id": old_id, "new_id": new_id, "text": text, "conf": confidence,
            "agent": agent, "session": session, "reason": reason,
            "now": _now()})
        if not res.result_set:
            raise MemoryGraphError(
                f"cannot correct {old_id}: not a fact, already retracted/superseded, "
                f"or {agent!r} did not run {session!r}")
        report = self._report(agent, old_id, res.result_set[0][2], "retracted")
        report["new"] = new_id
        return report

    def confirm(self, agent, item_id):
        """You reviewed a flagged item and it is still fine, so make it active again.

        Only for items flagged because a source CHANGED. If a source was WRONG the item has to be
        redone instead (supersede it, or write a new fact or decision).
        """
        self._require_visible(agent, [item_id])
        res = self.g.query(queries.CONFIRM, {"me": agent, "id": item_id, "now": _now()})
        if not res.result_set:
            raise MemoryGraphError(
                f"cannot confirm {item_id}: not flagged, not yours, or built on something that was wrong "
                f"(redo it instead)")

    def _report(self, agent, item_id, flagged_all, label):
        """Build the result of a change: what this agent may see, a hidden count, who to notify."""
        visible = []
        if flagged_all:
            rows = self.g.query(queries.VISIBLE_ITEMS, {"me": agent, "ids": flagged_all}).result_set
            visible = sorted(r[0] for r in rows)
        authors, recipients = self.g.query(
            queries.AFFECTED_AGENTS, {"ids": [item_id] + visible}).result_set[0]
        return {
            label: item_id,
            "flagged": visible,
            "hidden_flagged": len(flagged_all) - len(visible),
            "agents": sorted(set(authors) | set(recipients)),
        }

    # --- read ---

    def recall(self, agent, entity, as_of=None):
        """Active facts about an entity this agent may see, each with the ids it was built on.

        Each fact has a `standing`, judged on the fact first:
          verified    a check (web or document) supported it, whoever wrote it
          user        the user said it: their word, not an agent's claim
          unverified  no evidence yet, and its author's accuracy is fine
          doubtful    a check contradicted it, or no evidence and its author's accuracy is low
        needs_verification is true for doubtful facts. Nothing is hidden: facts come sorted by
        standing, then by the author's accuracy. Retracted and flagged facts are left out.
        as_of=None gives what is current now; a timestamp (ms) gives what was true then.
        """
        if as_of is None:
            res = self.g.query(queries.RECALL, {"entity": entity, "me": agent})
        else:
            res = self.g.query(queries.RECALL_AS_OF, {"entity": entity, "me": agent, "t": as_of})
        rows = res.result_set
        records = self.records({r[3] for r in rows}) if rows else {}
        facts = []
        for fid, text, built_on, author, source, supported, contradicted in rows:
            accuracy = records[author]["accuracy"]
            if supported:
                standing = "verified"
            elif source == "user" and not contradicted:
                standing = "user"
            elif contradicted or accuracy < ACCURACY_WARN:
                standing = "doubtful"
            else:
                standing = "unverified"
            facts.append({"id": fid, "text": text, "built_on": sorted(built_on), "author": author,
                          "source": source, "author_accuracy": accuracy, "standing": standing,
                          "verified": supported, "needs_verification": standing == "doubtful"})
        return sorted(facts, key=lambda f: (STANDING_ORDER[f["standing"]], -f["author_accuracy"], f["id"]))

    def verify(self, agent, fact_id, method, result, sources, snippet=""):
        """Record the outcome of an evidence check on a fact. Returns {"result"}.

        This only records. The check itself (a web search or a read of an uploaded document)
        is done by the checker in llm.py, which supplies `result`, `sources` and `snippet`.
        Evidence is required: `sources` (URLs or document names) must not be empty. Any agent
        that may see the fact can record a check; it does not edit the fact. Evidence is about the
        fact: 'supported' makes it verified in recall, 'contradicted' makes it doubtful, and
        neither changes the author's accuracy (only a retraction does). Retracting is still the
        owner's decision.
        """
        if method not in VERIFY_METHODS:
            raise MemoryGraphError(f"method must be one of {VERIFY_METHODS}, not {method!r}")
        if result not in VERIFY_RESULTS:
            raise MemoryGraphError(f"result must be one of {VERIFY_RESULTS}, not {result!r}")
        sources = [s for s in (sources or []) if s and s.strip()]
        if not sources:
            raise MemoryGraphError("a check needs evidence: at least one source (URL or document name)")
        rows = self.g.query(queries.VERIFY, {
            "me": agent, "id": fact_id, "method": method, "result": result,
            "sources": sources, "snippet": snippet, "now": _now()}).result_set
        if not rows:
            raise MemoryGraphError(
                f"cannot verify {fact_id}: not found, not visible to {agent}, or not active")
        return {"result": result}

    def explain(self, agent, item_id):
        """The chain of facts and agents behind an item, nearest first, with each link's status."""
        self._require_visible(agent, [item_id])
        rows = self.g.query(queries.PROVENANCE, {"id": item_id}).result_set
        return [{"id": r[0], "text": r[1], "agent": r[2], "depth": r[3], "status": r[4]} for r in rows]

    def blast_radius(self, agent, item_id):
        """Dry run: everything built on an item (that this agent may see). Changes nothing."""
        self._require_visible(agent, [item_id])
        rows = self.g.query(queries.BLAST_RADIUS, {"id": item_id, "me": agent}).result_set
        return [{"id": r[0], "text": r[1], "status": r[2], "agent": r[3], "depth": r[4]} for r in rows]

    def flagged(self, agent):
        """Flagged items of yours (and your owner's other agents) that need review or redoing.

        Each has `wrong` (source facts that were wrong: redo it) and `changed` (source facts that
        were superseded, with the fact that replaced them: review it, then confirm or redo).
        """
        rows = self.g.query(queries.FLAGGED, {"me": agent}).result_set
        changed_ids = sorted({c for r in rows for c in r[4]})
        now_map = {}
        if changed_ids:
            now_map = dict(self.g.query(queries.SUPERSEDED_BY, {"ids": changed_ids}).result_set)
        return [{
            "id": r[0], "text": r[1], "because": sorted(r[2] or []),
            "wrong": sorted(r[3]),
            "changed": [{"fact": c, "now": now_map.get(c)} for c in sorted(r[4])],
        } for r in rows]


class Tenants:
    """One FalkorDB graph per tenant (organisation). Nothing crosses the graph boundary: ids,
    agents and facts of one tenant do not exist in another, so isolation needs no filtering.

    tenants.get("acme") -> a Memory on graph "tenant_acme" (created and indexed on first use).
    """
    NAME_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,62}")

    def __init__(self, db=None, prefix="tenant_"):
        self.db = db if db is not None else _connect()
        self.prefix = prefix
        self._open = {}

    def _graph_name(self, tenant):
        # strict names, so a tenant name can never address some other graph
        if not isinstance(tenant, str) or not self.NAME_RE.fullmatch(tenant):
            raise MemoryGraphError(f"bad tenant name {tenant!r}: use lowercase letters, digits and '-'")
        return self.prefix + tenant

    def get(self, tenant):
        name = self._graph_name(tenant)
        if tenant not in self._open:
            mem = Memory(name, db=self.db)
            mem.setup()
            self._open[tenant] = mem
        return self._open[tenant]

    def list(self):
        return sorted(g[len(self.prefix):] for g in self.db.list_graphs() if g.startswith(self.prefix))

    def drop(self, tenant):
        """Delete one tenant's whole graph. Other tenants are untouched."""
        self.get(tenant).drop()
        self._open.pop(tenant, None)
