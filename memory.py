"""Shared agent memory on FalkorDB. Wraps the Cypher in queries.py and enforces the rules.

Owners: an agent can belong to a user (`register_agent(name, owner="Sam")`). A user's agents share
that user's private memory and may change each other's facts; agents of other users may not.
An agent without an owner owns itself.

Visibility: a fact is `private` (only its owner's agents see it) until it is published.
An agent may only build on (derive from) active things it can see, and may only publish a fact
once everything it was built on is shared, so shared lineage never points at hidden items.
Decisions are team-visible, so they may only be built on shared items.

Changes: only the author's owner can supersede, correct or retract a fact, so one user's memory
never changes because of another user's edit. Items built on a changed fact are flagged
("changed") and items built on a wrong fact are flagged ("wrong"), so their owners find out.
"""
import os
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from falkordb import FalkorDB
from redis.exceptions import ResponseError

import queries

INDEXED_LABELS = ("Fact", "Decision", "Agent", "Entity", "Handoff")
TRUST_PENALTY = 0.8  # an author's trust counter is multiplied by this when its fact is retracted


class MemoryGraphError(Exception):
    """A memory rule was broken (unknown agent, hidden source, not the owner, ...)."""


def _now():
    return int(time.time() * 1000)


def _new_id(prefix):
    return f"{prefix}:{uuid.uuid4().hex[:8]}"


def _slug(text):
    return text.strip().lower().replace(" ", "-")


class Memory:
    def __init__(self, graph="memory", db=None):
        if db is None:
            load_dotenv(Path(__file__).resolve().parent / ".env")
            db = FalkorDB(
                host=os.environ["FALKORDB_HOST"],
                port=int(os.environ["FALKORDB_PORT"]),
                username=os.environ["FALKORDB_USERNAME"],
                password=os.environ["FALKORDB_PASSWORD"],
            )
        self.db = db
        self.name = graph
        self.g = db.select_graph(graph)

    # --- admin ---

    def setup(self):
        """Create the id indexes. Safe to call repeatedly."""
        for label in INDEXED_LABELS:
            try:
                self.g.query(f"CREATE INDEX FOR (n:{label}) ON (n.id)")
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
            "MERGE (a:Agent {id:$id}) ON CREATE SET a.name = $name, a.trust = 1.0, a.owner = $owner",
            {"id": agent_id, "name": name, "owner": owner_id},
        )
        return agent_id

    def start_session(self, agent):
        session_id = _new_id("sess")
        res = self.g.query(
            "MATCH (a:Agent {id:$agent}) CREATE (a)-[:RAN]->(:Session {id:$s, started_at:$now})",
            {"agent": agent, "s": session_id, "now": _now()},
        )
        if res.nodes_created == 0:
            raise MemoryGraphError(f"unknown agent {agent!r}")
        return session_id

    def trust(self, agent):
        rows = self.g.query("MATCH (a:Agent {id:$id}) RETURN a.trust", {"id": agent}).result_set
        if not rows:
            raise MemoryGraphError(f"unknown agent {agent!r}")
        return rows[0][0]

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

    def _require_sources(self, agent, ids, shared_only):
        """Raise unless every id is an active fact/decision this agent may build on."""
        ids = list(dict.fromkeys(ids))
        if not ids:
            return
        rows = self.g.query(
            queries.USABLE_SOURCES, {"me": agent, "ids": ids, "shared_only": shared_only}).result_set
        found = {r[0] for r in rows}
        missing = [i for i in ids if i not in found]
        if missing:
            why = "unknown, not visible to the agent, or no longer active (retracted, flagged, superseded)"
            if shared_only:
                why += "; decisions may only be built on shared items"
            raise MemoryGraphError(f"cannot build on {missing}: {why}")

    def _require_may_change(self, agent, item_id):
        """Raise unless this agent, or another agent of the same owner, wrote the item."""
        if not self.g.query(queries.MAY_CHANGE, {"me": agent, "id": item_id}).result_set:
            raise MemoryGraphError(
                f"only the author (or the author's other agents) can change {item_id}; "
                f"{agent} can write its own fact instead")

    # --- write ---

    def remember(self, agent, session, text, entity, entity_name=None, entity_type="thing",
                 confidence=0.5, derived_from=(), valid_from=None):
        """Write a private fact. Returns its id."""
        derived_from = list(derived_from)
        self._require_sources(agent, derived_from, shared_only=False)
        fact_id = _new_id("fact")
        now = _now()
        res = self.g.query(queries.REMEMBER, {
            "agent": agent, "session": session, "entity": entity,
            "entity_name": entity_name or entity, "entity_type": entity_type,
            "id": fact_id, "text": text, "conf": confidence,
            "valid_from": valid_from if valid_from is not None else now, "now": now,
            "derived_from": derived_from,
        })
        if res.nodes_created == 0:
            raise MemoryGraphError(f"agent {agent!r} did not run session {session!r} (or one is unknown)")
        return fact_id

    def decide(self, agent, session, text, derived_from=()):
        """Record a decision built on shared, active items. Returns its id."""
        derived_from = list(derived_from)
        self._require_sources(agent, derived_from, shared_only=True)
        decision_id = _new_id("dec")
        res = self.g.query(queries.DECIDE, {
            "agent": agent, "session": session, "id": decision_id, "text": text,
            "now": _now(), "derived_from": derived_from,
        })
        if res.nodes_created == 0:
            raise MemoryGraphError(f"agent {agent!r} did not run session {session!r} (or one is unknown)")
        return decision_id

    def publish(self, agent, fact_id):
        """Share a private fact with everyone. Only its owner's agents, and only once its sources are shared."""
        rows = self.g.query(queries.PUBLISH_CHECK, {"agent": agent, "id": fact_id}).result_set
        if not rows:
            raise MemoryGraphError(f"{agent} has no active fact {fact_id} of its own to publish")
        blockers = rows[0][1]
        if blockers:
            raise MemoryGraphError(f"cannot publish {fact_id}: built on private {sorted(blockers)}; publish those first")
        self.g.query(queries.PUBLISH, {"agent": agent, "id": fact_id})

    def supersede(self, agent, session, old_id, text, confidence=0.5, at=None):
        """The fact was true and then changed: replace it, keeping the old one as history.

        Only the author's owner can do this. Items built on the old fact are flagged "changed" so
        their owners can review (confirm) or redo them; the author's trust is untouched, since
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

        Only the author's owner can retract. The author's trust counter is lowered (a reliability
        record, not a punishment). Returns {"retracted", "flagged", "hidden_flagged", "agents"}:
        `flagged` lists the flagged items this agent may see, `hidden_flagged` counts private items
        of other users (their owners still see them in flagged()), `agents` are the ones to notify.
        """
        self._require_visible(agent, [fact_id])
        self._require_may_change(agent, fact_id)
        res = self.g.query(queries.RETRACT, {
            "id": fact_id, "by": agent, "reason": reason, "factor": TRUST_PENALTY, "now": _now()})
        if not res.result_set:
            raise MemoryGraphError(f"cannot retract {fact_id}: not a fact, or already retracted/superseded")
        return self._report(agent, fact_id, res.result_set[0][1], "retracted")

    def correct(self, agent, session, old_id, text, confidence=0.5, reason=""):
        """One of your facts was wrong: retract it and write the right one, in a single step.

        Same effects as retract() (only the author's owner can do it; flags everything built on
        the old fact as "wrong"; lowers the author's trust counter), plus a new fact that keeps
        the old one's visibility and entity links and is valid from the same time (the mistake
        was in the record, not in the world). Returns retract()'s report plus {"new": id}.
        Use supersede() instead when the fact was true and then changed.
        """
        self._require_visible(agent, [old_id])
        self._require_may_change(agent, old_id)
        new_id = _new_id("fact")
        res = self.g.query(queries.CORRECT, {
            "old_id": old_id, "new_id": new_id, "text": text, "conf": confidence,
            "agent": agent, "session": session, "reason": reason,
            "factor": TRUST_PENALTY, "now": _now()})
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

        Retracted and flagged facts are left out. as_of=None gives what is current now; a
        timestamp (ms) gives what was true then.
        """
        if as_of is None:
            res = self.g.query(queries.RECALL, {"entity": entity, "me": agent})
        else:
            res = self.g.query(queries.RECALL_AS_OF, {"entity": entity, "me": agent, "t": as_of})
        return [{"id": r[0], "text": r[1], "built_on": sorted(r[2])} for r in res.result_set]

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
