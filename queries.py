"""Core Cypher for the shared memory graph. memory.py (Step 2) wraps these.

Conventions: DERIVED_FROM points from the derived item to its source.
valid_from/valid_to = when a fact was true in the world (no valid_to = still true).
recorded_at = when the system wrote it down. All times are epoch milliseconds.
"""

# remember: an agent writes a fact. Starts private. Optionally derived from other facts/decisions.
# Params: agent, session, entity, entity_name, entity_type, id, text, conf, valid_from, now,
#         derived_from (list of ids, may be empty)
# Writes nothing if the agent or session does not exist (the caller should check the stats).
REMEMBER = """
MATCH (a:Agent {id:$agent}), (s:Session {id:$session})
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

# supersede: the old fact was true but changed. Close it and link the new one.
# One query, so there is never a moment with two current facts. The new fact inherits the
# old fact's visibility and ABOUT links. Creates nothing if the old fact is already closed.
# Params: old_id, new_id, text, conf, agent, session, at, now
SUPERSEDE = """
MATCH (old:Fact {id:$old_id}) WHERE old.valid_to IS NULL
MATCH (a:Agent {id:$agent}), (s:Session {id:$session})
SET old.valid_to = $at, old.status = 'superseded'
CREATE (new:Fact {id:$new_id, text:$text, visibility:old.visibility, status:'active',
                  confidence:$conf, valid_from:$at, recorded_at:$now})
CREATE (new)-[:SUPERSEDES]->(old), (a)-[:ASSERTED]->(new), (new)-[:IN_SESSION]->(s)
WITH new, old
MATCH (old)-[:ABOUT]->(e)
CREATE (new)-[:ABOUT]->(e)
"""

# recall: current facts about an entity that this agent may see (shared, or its own private),
# each with the ids of what it was built on, up to 3 hops back.
# Only status 'active' is returned, so superseded, retracted and flagged facts are left out.
# Params: entity, me
RECALL = """
MATCH (f:Fact)-[:ABOUT]->(:Entity {id:$entity})
WHERE f.status = 'active' AND f.valid_to IS NULL
  AND (f.visibility = 'shared' OR (:Agent {id:$me})-[:ASSERTED]->(f))
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
RETURN src.id AS id, src.text AS text, a.id AS agent, length(p) AS depth
ORDER BY depth
"""
