"""Core Cypher for the shared memory graph. memory.py wraps these.

Conventions: DERIVED_FROM points from the derived item to its source.
valid_from/valid_to = when a fact was true in the world (no valid_to = still true).
recorded_at = when the system wrote it down. All times are epoch milliseconds.

Ownership: an agent may have an `owner` (its user, e.g. 'user:sam'); an agent without one owns
itself. "Same owner" (author.id = me.id OR author.owner = me.owner) is the single rule for who
may see a private fact and who may change, retract, publish or confirm a fact.
"""

# remember: an agent writes a fact. Starts private. Optionally derived from other facts/decisions.
# Params: agent, session, entity, entity_name, entity_type, id, text, conf, valid_from, now,
#         derived_from (list of ids, may be empty)
# Writes nothing if the agent did not run the session (the caller should check the stats).
REMEMBER = """
MATCH (a:Agent {id:$agent})-[:RAN]->(s:Session {id:$session})
MERGE (e:Entity {id:$entity})
  ON CREATE SET e.name = $entity_name, e.type = $entity_type
CREATE (f:Fact {id:$id, text:$text, visibility:'private', status:'active',
                confidence:$conf, valid_from:$valid_from, recorded_at:$now})
CREATE (a)-[:ASSERTED]->(f), (f)-[:ABOUT]->(e), (f)-[:IN_SESSION]->(s)
WITH f
UNWIND $derived_from AS src_id
MATCH (src {id:src_id})
CREATE (f)-[:DERIVED_FROM]->(src)
"""

# supersede: the old fact was true but changed. Close it, link the new one, and flag every
# still-active downstream item as "changed" (flag_changed) so its owner can review or redo it.
# One atomic query. The new fact inherits the old fact's visibility, ABOUT links. Only the
# author (or an agent of the same owner) can do this. Returns no row, and writes nothing, if the
# old fact is not active/flagged, the agent may not change it, or it did not run the session.
# Params: old_id, new_id, text, conf, agent, session, at, now
SUPERSEDE = """
MATCH (old:Fact {id:$old_id}) WHERE old.valid_to IS NULL AND old.status IN ['active', 'flagged']
MATCH (a:Agent {id:$agent})-[:RAN]->(s:Session {id:$session})
MATCH (author:Agent)-[:ASSERTED]->(old) WHERE author.id = a.id OR author.owner = a.owner
SET old.valid_to = $at, old.status = 'superseded'
CREATE (new:Fact {id:$new_id, text:$text, visibility:old.visibility, status:'active',
                  confidence:$conf, valid_from:$at, recorded_at:$now})
CREATE (new)-[:SUPERSEDES]->(old), (a)-[:ASSERTED]->(new), (new)-[:IN_SESSION]->(s)
WITH old, new
MATCH (old)-[:ABOUT]->(e)
CREATE (new)-[:ABOUT]->(e)
WITH DISTINCT old, new
OPTIONAL MATCH (old)<-[:DERIVED_FROM*1..]-(x) WHERE x.status IN ['active', 'flagged']
WITH old, new, collect(DISTINCT x) AS affected
FOREACH (n IN affected | SET n.status = 'flagged',
                             n.flagged_by = coalesce(n.flagged_by, []) + [old.id],
                             n.flag_changed = coalesce(n.flag_changed, []) + [old.id])
RETURN old.id AS id, new.id AS new_id, [n IN affected | n.id] AS flagged
"""

# recall: current facts about an entity that this agent may see (shared, or its owner's own
# private ones), each with the ids of what it was built on, up to 3 hops back.
# Only status 'active' is returned, so superseded, retracted and flagged facts are left out.
# Params: entity, me
RECALL = """
MATCH (me:Agent {id:$me})
MATCH (author:Agent)-[:ASSERTED]->(f:Fact)-[:ABOUT]->(:Entity {id:$entity})
WHERE f.status = 'active' AND f.valid_to IS NULL
  AND (f.visibility = 'shared' OR author.id = me.id OR author.owner = me.owner)
OPTIONAL MATCH (f)-[:DERIVED_FROM*1..3]->(src)
RETURN f.id AS id, f.text AS text, collect(DISTINCT src.id) AS built_on
ORDER BY id
"""

# provenance (explain): the chain behind an item, nearest first, with the agent who asserted
# each link. *0.. includes the item itself at depth 0.
# Params: id
PROVENANCE = """
MATCH p = (x {id:$id})-[:DERIVED_FROM*0..]->(src)
MATCH (a:Agent)-[:ASSERTED]->(src)
RETURN src.id AS id, src.text AS text, a.id AS agent, length(p) AS depth, src.status AS status
ORDER BY depth
"""

# recall as of a past time: facts that were true at $t (superseded ones count, retracted do not).
# Params: entity, me, t
RECALL_AS_OF = """
MATCH (me:Agent {id:$me})
MATCH (author:Agent)-[:ASSERTED]->(f:Fact)-[:ABOUT]->(:Entity {id:$entity})
WHERE f.status IN ['active', 'superseded']
  AND f.valid_from <= $t AND (f.valid_to IS NULL OR f.valid_to > $t)
  AND (f.visibility = 'shared' OR author.id = me.id OR author.owner = me.owner)
OPTIONAL MATCH (f)-[:DERIVED_FROM*1..3]->(src)
RETURN f.id AS id, f.text AS text, collect(DISTINCT src.id) AS built_on
ORDER BY id
"""

# Which of these ids are facts/decisions this agent may see (shared, or its owner's own)?
# Decisions have no visibility property, so they count as visible.
# Params: me, ids
VISIBLE_ITEMS = """
MATCH (me:Agent {id:$me})
MATCH (x) WHERE x.id IN $ids AND labels(x)[0] IN ['Fact', 'Decision']
OPTIONAL MATCH (author:Agent)-[:ASSERTED]->(x)
WITH me, x, author
WHERE x.visibility IS NULL OR x.visibility = 'shared'
   OR author.id = me.id OR author.owner = me.owner
RETURN DISTINCT x.id
"""

# publish step 1: may this agent publish this active fact (it, or an agent of its owner, wrote it),
# and which private ancestors does it have? No row means not found, not allowed, or not active.
# Params: agent, id
PUBLISH_CHECK = """
MATCH (me:Agent {id:$agent})
MATCH (author:Agent)-[:ASSERTED]->(f:Fact {id:$id})
WHERE f.status = 'active' AND (author.id = me.id OR author.owner = me.owner)
OPTIONAL MATCH (f)-[:DERIVED_FROM*1..]->(src) WHERE src.visibility = 'private'
RETURN f.id, collect(DISTINCT src.id) AS private_ancestors
"""

# publish step 2: make it visible to everyone. Params: agent, id
PUBLISH = """
MATCH (me:Agent {id:$agent})
MATCH (author:Agent)-[:ASSERTED]->(f:Fact {id:$id})
WHERE f.status = 'active' AND (author.id = me.id OR author.owner = me.owner)
SET f.visibility = 'shared'
RETURN f.id
"""

# --- lineage ---

# Which of these ids can this agent build on? They must be active facts/decisions it can see.
# shared_only=true (used by decisions) also rules out private items, even its owner's own.
# Params: me, ids, shared_only
USABLE_SOURCES = """
MATCH (me:Agent {id:$me})
MATCH (x) WHERE x.id IN $ids AND labels(x)[0] IN ['Fact', 'Decision'] AND x.status = 'active'
OPTIONAL MATCH (author:Agent)-[:ASSERTED]->(x)
WITH me, x, author
WHERE x.visibility IS NULL OR x.visibility = 'shared'
   OR (NOT $shared_only AND (author.id = me.id OR author.owner = me.owner))
RETURN DISTINCT x.id
"""

# decide: an agent records a decision built on shared items. Decisions have no visibility
# property (they are team-visible). Writes nothing if the agent did not run the session.
# Params: agent, session, id, text, now, derived_from (list of ids, may be empty)
DECIDE = """
MATCH (a:Agent {id:$agent})-[:RAN]->(s:Session {id:$session})
CREATE (d:Decision {id:$id, text:$text, status:'active', created_at:$now, recorded_at:$now})
CREATE (a)-[:ASSERTED]->(d), (d)-[:IN_SESSION]->(s)
WITH d
UNWIND $derived_from AS src_id
MATCH (src {id:src_id})
CREATE (d)-[:DERIVED_FROM]->(src)
"""

# blast radius preview: everything built on an item, directly or indirectly, with the nearest
# distance. Only items this agent may see. Changes nothing. Params: id, me
BLAST_RADIUS = """
MATCH (me:Agent {id:$me})
MATCH p = (s {id:$id})<-[:DERIVED_FROM*1..]-(x)
MATCH (a:Agent)-[:ASSERTED]->(x)
WHERE x.visibility IS NULL OR x.visibility = 'shared' OR a.id = me.id OR a.owner = me.owner
RETURN x.id AS id, x.text AS text, x.status AS status, a.id AS agent, min(length(p)) AS depth
ORDER BY depth, id
"""

# retract: the fact was wrong. One atomic query that (1) marks it retracted, (2) lowers the
# author's trust counter by $factor, and (3) flags every still-active downstream item as
# "wrong" (flag_wrong), recording which retraction flagged it. $by must be the author or an
# agent of the author's owner. Returns no row if the fact is not an active/flagged fact that
# $by may change. Params: id, by, reason, factor, now
RETRACT = """
MATCH (f:Fact {id:$id}) WHERE f.status IN ['active', 'flagged']
MATCH (me:Agent {id:$by})
MATCH (asserter:Agent)-[:ASSERTED]->(f) WHERE asserter.id = me.id OR asserter.owner = me.owner
SET f.status = 'retracted', f.retracted_at = $now, f.retracted_by = $by,
    f.retract_reason = $reason, asserter.trust = asserter.trust * $factor
WITH f
OPTIONAL MATCH (f)<-[:DERIVED_FROM*1..]-(x) WHERE x.status IN ['active', 'flagged']
WITH f, collect(DISTINCT x) AS affected
FOREACH (n IN affected | SET n.status = 'flagged',
                             n.flagged_by = coalesce(n.flagged_by, []) + [f.id],
                             n.flag_wrong = coalesce(n.flag_wrong, []) + [f.id])
RETURN f.id AS id, [n IN affected | n.id] AS flagged
"""

# correct: the old fact was wrong and here is the right one. One atomic query that does what
# RETRACT does (retract, flag downstream as wrong, trust counter) and also writes the corrected
# fact. The new fact copies the old one's visibility, ABOUT links and valid_from (it was a
# mistake in the record, so the right value was true all along), and a CORRECTS edge points
# from it to the old fact. Only the author or an agent of the same owner can do this. Returns
# no row, and writes nothing, if the old fact is not an active/flagged fact, the agent may not
# change it, or it did not run the session.
# Params: old_id, new_id, text, conf, agent, session, reason, factor, now
CORRECT = """
MATCH (f:Fact {id:$old_id}) WHERE f.status IN ['active', 'flagged']
MATCH (a:Agent {id:$agent})-[:RAN]->(s:Session {id:$session})
MATCH (asserter:Agent)-[:ASSERTED]->(f) WHERE asserter.id = a.id OR asserter.owner = a.owner
SET f.status = 'retracted', f.retracted_at = $now, f.retracted_by = $agent,
    f.retract_reason = $reason, asserter.trust = asserter.trust * $factor
CREATE (new:Fact {id:$new_id, text:$text, visibility:f.visibility, status:'active',
                  confidence:$conf, valid_from:f.valid_from, recorded_at:$now})
CREATE (new)-[:CORRECTS]->(f), (a)-[:ASSERTED]->(new), (new)-[:IN_SESSION]->(s)
WITH f, new
MATCH (f)-[:ABOUT]->(e)
CREATE (new)-[:ABOUT]->(e)
WITH DISTINCT f, new
OPTIONAL MATCH (f)<-[:DERIVED_FROM*1..]-(x) WHERE x.status IN ['active', 'flagged']
WITH f, new, collect(DISTINCT x) AS affected
FOREACH (n IN affected | SET n.status = 'flagged',
                             n.flagged_by = coalesce(n.flagged_by, []) + [f.id],
                             n.flag_wrong = coalesce(n.flag_wrong, []) + [f.id])
RETURN f.id AS id, new.id AS new_id, [n IN affected | n.id] AS flagged
"""

# Agents to notify about these items: who asserted them, and who received them in a handoff.
# Params: ids
AFFECTED_AGENTS = """
MATCH (x) WHERE x.id IN $ids
OPTIONAL MATCH (a:Agent)-[:ASSERTED]->(x)
OPTIONAL MATCH (x)<-[:CARRIED]-(:Handoff)-[:TO]->(r:Agent)
RETURN collect(DISTINCT a.id) AS authors, collect(DISTINCT r.id) AS recipients
"""

# What of mine (or my owner's other agents') is flagged and needs review or redoing?
# `wrong` = sources that were wrong, `changed` = sources that were superseded. Params: me
FLAGGED = """
MATCH (me:Agent {id:$me})
MATCH (author:Agent)-[:ASSERTED]->(x)
WHERE x.status = 'flagged' AND (author.id = me.id OR author.owner = me.owner)
RETURN x.id AS id, x.text AS text, x.flagged_by AS because,
       coalesce(x.flag_wrong, []) AS wrong, coalesce(x.flag_changed, []) AS changed
ORDER BY id
"""

# Which fact replaced each of these superseded facts? Params: ids
SUPERSEDED_BY = """
MATCH (n:Fact)-[:SUPERSEDES]->(o) WHERE o.id IN $ids
RETURN o.id, n.id
"""

# confirm: the owner reviewed a flagged item and says it is still fine. Only allowed when it was
# flagged because sources CHANGED; if any source was WRONG it has to be redone instead.
# Params: me, id, now
CONFIRM = """
MATCH (me:Agent {id:$me})
MATCH (author:Agent)-[:ASSERTED]->(x {id:$id})
WHERE x.status = 'flagged' AND size(coalesce(x.flag_wrong, [])) = 0
  AND (author.id = me.id OR author.owner = me.owner)
SET x.status = 'active', x.confirmed_at = $now, x.confirmed_by = $me,
    x.flagged_by = [], x.flag_changed = []
RETURN x.id
"""

# May this agent change this item (it wrote it, or an agent of the same owner did)? Params: me, id
MAY_CHANGE = """
MATCH (me:Agent {id:$me})
MATCH (author:Agent)-[:ASSERTED]->(x {id:$id})
WHERE author.id = me.id OR author.owner = me.owner
RETURN x.id
"""
