"""Core Cypher for the shared memory graph. memory.py wraps these.

Conventions: DERIVED_FROM points from the derived item to its source.
valid_from/valid_to = when a fact was true in the world (no valid_to = still true).
recorded_at = when the system wrote it down. All times are epoch milliseconds.

Ownership: an agent may have an `owner` (its user, e.g. 'user:sam'); an agent without one owns
itself. "Same owner" (author.id = me.id OR author.owner = me.owner) is the single rule for who
may see a private fact and who may change, retract, publish or confirm a fact.
"""

# remember: an agent writes a fact. Starts private. Optionally derived from other facts/decisions.
# `source` says where it came from: user (the user said it), web, document, or agent (the agent's
# own claim). Only non-user facts count toward the author's accuracy.
# Params: agent, session, entity, entity_name, entity_type, id, text, conf, valid_from, now, source,
#         derived_from (list of distinct ids, may be empty), from_messages (message ids, may be empty;
#         EXTRACTED_FROM edges, which blast radius never follows), also_about (more entity ids
#         the same fact is about: one node with several ABOUT edges, never a copy per entity)
# Check and write are ONE query: if any source is not an active fact/decision (e.g. it was
# retracted a moment ago) nothing is written, so a retract can never slip in between.
# Writes nothing, too, if the agent did not run the session (the caller should check the stats).
REMEMBER = """
MATCH (a:Agent {id:$agent})-[:RAN]->(s:Session {id:$session})
OPTIONAL MATCH (src) WHERE src.id IN $derived_from AND labels(src)[0] IN ['Fact', 'Decision']
                       AND src.status = 'active'
WITH a, s, collect(DISTINCT src) AS sources
WHERE size(sources) = size($derived_from)
MERGE (e:Entity {id:$entity})
  ON CREATE SET e.name = $entity_name, e.type = $entity_type
CREATE (f:Fact {id:$id, text:$text, visibility:'private', status:'active', source:$source,
                confidence:$conf, valid_from:$valid_from, recorded_at:$now})
CREATE (a)-[:ASSERTED]->(f), (f)-[:ABOUT]->(e), (f)-[:IN_SESSION]->(s)
WITH f, sources
FOREACH (x IN $also_about | MERGE (o:Entity {id:x}) ON CREATE SET o.name = x, o.type = 'thing'
                            CREATE (f)-[:ABOUT]->(o))
WITH f, sources
OPTIONAL MATCH (m:Message) WHERE m.id IN $from_messages
WITH f, sources, collect(m) AS msgs
FOREACH (m IN msgs | CREATE (f)-[:EXTRACTED_FROM]->(m))
WITH f, sources
UNWIND sources AS src
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
CREATE (new:Fact {id:$new_id, text:$text, visibility:old.visibility, status:'active', source:old.source,
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
# Also returns the author, the fact's source, and when a check last supported or contradicted it
# (the caller adds the author's accuracy and the fact's standing, and sorts; the latest check wins).
# Params: entity, me
RECALL = """
MATCH (me:Agent {id:$me})
MATCH (author:Agent)-[:ASSERTED]->(f:Fact)-[:ABOUT]->(:Entity {id:$entity})
WHERE f.status = 'active' AND f.valid_to IS NULL
  AND (f.visibility = 'shared' OR author.id = me.id OR author.owner = me.owner)
OPTIONAL MATCH (f)-[:DERIVED_FROM*1..3]->(src)
OPTIONAL MATCH (:Agent)-[v:VERIFIED {result:'supported'}]->(f)
OPTIONAL MATCH (:Agent)-[x:VERIFIED {result:'contradicted'}]->(f)
RETURN f.id AS id, f.text AS text, collect(DISTINCT src.id) AS built_on, author.id AS author,
       coalesce(f.source, 'agent') AS source, max(v.at) AS supported_at, max(x.at) AS contradicted_at
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
OPTIONAL MATCH (:Agent)-[v:VERIFIED {result:'supported'}]->(f)
OPTIONAL MATCH (:Agent)-[x:VERIFIED {result:'contradicted'}]->(f)
RETURN f.id AS id, f.text AS text, collect(DISTINCT src.id) AS built_on, author.id AS author,
       coalesce(f.source, 'agent') AS source, max(v.at) AS supported_at, max(x.at) AS contradicted_at
"""

# Which of these ids are facts/decisions/procedures this agent may see (shared, or its owner's own)?
# An item with no visibility property (a decision from before decisions had one) counts as shared.
# Params: me, ids
VISIBLE_ITEMS = """
MATCH (me:Agent {id:$me})
MATCH (x) WHERE x.id IN $ids AND labels(x)[0] IN ['Fact', 'Decision', 'Procedure']
OPTIONAL MATCH (author:Agent)-[:ASSERTED]->(x)
WITH me, x, author
WHERE x.visibility IS NULL OR x.visibility = 'shared'
   OR author.id = me.id OR author.owner = me.owner
RETURN DISTINCT x.id
"""

# publish step 1: may this agent publish this active fact/decision/procedure (it, or an agent of its owner, wrote it),
# and which private ancestors does it have? No row means not found, not allowed, or not active.
# Params: agent, id
PUBLISH_CHECK = """
MATCH (me:Agent {id:$agent})
MATCH (author:Agent)-[:ASSERTED]->(f {id:$id})
WHERE labels(f)[0] IN ['Fact', 'Decision', 'Procedure'] AND f.status = 'active' AND (author.id = me.id OR author.owner = me.owner)
OPTIONAL MATCH (f)-[:DERIVED_FROM*1..]->(src) WHERE src.visibility = 'private'
RETURN f.id, collect(DISTINCT src.id) AS private_ancestors
"""

# publish step 2: make it visible to everyone. Params: agent, id
PUBLISH = """
MATCH (me:Agent {id:$agent})
MATCH (author:Agent)-[:ASSERTED]->(f {id:$id})
WHERE labels(f)[0] IN ['Fact', 'Decision', 'Procedure'] AND f.status = 'active' AND (author.id = me.id OR author.owner = me.owner)
SET f.visibility = 'shared'
RETURN f.id
"""

# Lineage

# Which of these ids can this agent build on? They must be active facts/decisions it can see.
# Params: me, ids
USABLE_SOURCES = """
MATCH (me:Agent {id:$me})
MATCH (x) WHERE x.id IN $ids AND labels(x)[0] IN ['Fact', 'Decision'] AND x.status = 'active'
OPTIONAL MATCH (author:Agent)-[:ASSERTED]->(x)
WITH me, x, author
WHERE x.visibility IS NULL OR x.visibility = 'shared'
   OR author.id = me.id OR author.owner = me.owner
RETURN DISTINCT x.id
"""

# decide: an agent records a decision built on items it may see. Like a fact, it starts private
# (only its owner's agents see it): a plan made for one user is theirs, even when it is built on
# shared facts. It is shared only by publish. Decisions written before visibility existed have no
# visibility property and count as shared.
# Writes nothing if the agent did not run the session.
# Params: agent, session, id, text, now, derived_from (list of ids, may be empty)
DECIDE = """
MATCH (a:Agent {id:$agent})-[:RAN]->(s:Session {id:$session})
OPTIONAL MATCH (src) WHERE src.id IN $derived_from AND labels(src)[0] IN ['Fact', 'Decision']
                       AND src.status = 'active'
WITH a, s, collect(DISTINCT src) AS sources
WHERE size(sources) = size($derived_from)
CREATE (d:Decision {id:$id, text:$text, status:'active', created_at:$now, recorded_at:$now,
                    visibility:'private'})
CREATE (a)-[:ASSERTED]->(d), (d)-[:IN_SESSION]->(s)
WITH d, sources
UNWIND sources AS src
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

# retract: the fact was wrong. One atomic query that (1) marks it retracted (which counts as a
# mistake in its author's accuracy, unless the user was the source) and (2) flags every
# still-active downstream item as "wrong" (flag_wrong), recording which retraction flagged it.
# $by must be the author or an agent of the author's owner. Returns no row if the fact is not an
# active/flagged fact that $by may change. Params: id, by, reason, now
RETRACT = """
MATCH (f:Fact {id:$id}) WHERE f.status IN ['active', 'flagged']
MATCH (me:Agent {id:$by})
MATCH (asserter:Agent)-[:ASSERTED]->(f) WHERE asserter.id = me.id OR asserter.owner = me.owner
SET f.status = 'retracted', f.retracted_at = $now, f.retracted_by = $by,
    f.retract_reason = $reason
WITH f
OPTIONAL MATCH (f)<-[:DERIVED_FROM*1..]-(x) WHERE x.status IN ['active', 'flagged']
WITH f, collect(DISTINCT x) AS affected
FOREACH (n IN affected | SET n.status = 'flagged',
                             n.flagged_by = coalesce(n.flagged_by, []) + [f.id],
                             n.flag_wrong = coalesce(n.flag_wrong, []) + [f.id])
RETURN f.id AS id, [n IN affected | n.id] AS flagged
"""

# correct: the old fact was wrong and here is the right one. One atomic query that does what
# RETRACT does (retract, flag downstream as wrong) and also writes the corrected
# fact. The new fact copies the old one's visibility, source, ABOUT links and valid_from (it was a
# mistake in the record, so the right value was true all along), and a CORRECTS edge points
# from it to the old fact. Only the author or an agent of the same owner can do this. Returns
# no row, and writes nothing, if the old fact is not an active/flagged fact, the agent may not
# change it, or it did not run the session.
# Params: old_id, new_id, text, conf, agent, session, reason, now
CORRECT = """
MATCH (f:Fact {id:$old_id}) WHERE f.status IN ['active', 'flagged']
MATCH (a:Agent {id:$agent})-[:RAN]->(s:Session {id:$session})
MATCH (asserter:Agent)-[:ASSERTED]->(f) WHERE asserter.id = a.id OR asserter.owner = a.owner
SET f.status = 'retracted', f.retracted_at = $now, f.retracted_by = $agent,
    f.retract_reason = $reason
CREATE (new:Fact {id:$new_id, text:$text, visibility:f.visibility, status:'active', source:f.source,
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
       coalesce(x.flag_wrong, []) AS wrong, coalesce(x.flag_changed, []) AS changed,
       author.id AS author, labels(x)[0] AS kind
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

# Handoffs

# handoff: the sender passes items to the recipient. One atomic query, so an item retracted a
# moment ago can't slip through. Every item must be an active or flagged (a redo request)
# fact/decision that BOTH agents may
# see: shared, or private to a user who owns both agents. Nothing is published or shared by
# handing off. Writes nothing if either agent is unknown, they are the same agent, or any item
# fails the test. Params: sender, recipient, ids (distinct, not empty), id, note, now
HANDOFF = """
MATCH (s:Agent {id:$sender}), (r:Agent {id:$recipient}) WHERE s.id <> r.id
OPTIONAL MATCH (author:Agent)-[:ASSERTED]->(x)
  WHERE x.id IN $ids AND labels(x)[0] IN ['Fact', 'Decision'] AND x.status IN ['active', 'flagged']
    AND (x.visibility IS NULL OR x.visibility = 'shared'
         OR (author.owner = s.owner AND author.owner = r.owner))
WITH s, r, collect(DISTINCT x) AS items
WHERE size(items) = size($ids)
CREATE (s)-[:SENT]->(h:Handoff {id:$id, at:$now, note:$note})-[:TO]->(r)
WITH h, items
UNWIND items AS x
CREATE (h)-[:CARRIED]->(x)
"""

# What was handed to this agent, newest first, with the current status of each item. Params: me
INBOX = """
MATCH (s:Agent)-[:SENT]->(h:Handoff)-[:TO]->(:Agent {id:$me})
MATCH (h)-[:CARRIED]->(x)
RETURN h.id AS id, s.id AS sender, h.note AS note, h.at AS at,
       collect({id: x.id, text: x.text, status: x.status}) AS items
ORDER BY at DESC, id
"""

# Conversation history (episodic memory)
# A Session is the episode; Message nodes hang off it, in order (seq). Messages are a record of
# what was said, so they are never wrong or retracted: facts link to them with EXTRACTED_FROM,
# not DERIVED_FROM, and blast radius never follows that edge. Only the owning user's agents can
# read them.

# Append a message. The counter on the session hands out seq inside the write, so parallel
# appends get unique, ordered numbers. Writes nothing if the agent did not run the session.
# Params: agent, session, id, role, text, now
ADD_MESSAGE = """
MATCH (:Agent {id:$agent})-[:RAN]->(s:Session {id:$session})
SET s.msg_count = coalesce(s.msg_count, 0) + 1
WITH s, s.msg_count AS seq
CREATE (s)-[:HAS_MESSAGE]->(:Message {id:$id, role:$role, text:$text, seq:seq, at:$now, agent:$agent})
RETURN seq
"""

# join: another agent of the same owner joins a session, so one chat is shared by a user's
# agents. Returns no row if the session is unknown or none of its agents has the same owner.
# Params: agent, session
JOIN_SESSION = """
MATCH (a:Agent {id:$agent})
MATCH (b:Agent)-[:RAN]->(s:Session {id:$session}) WHERE b.owner = a.owner
WITH a, s LIMIT 1
MERGE (a)-[:RAN]->(s)
RETURN s.id
"""

# May this agent read this session (it, or an agent of the same owner, ran it)? Params: me, session
SESSION_ACCESS = """
MATCH (me:Agent {id:$me})
MATCH (a:Agent)-[:RAN]->(s:Session {id:$session}) WHERE a.id = me.id OR a.owner = me.owner
RETURN s.id
"""

# Messages of a session in order. Params: session
HISTORY = """
MATCH (:Session {id:$session})-[:HAS_MESSAGE]->(m:Message)
RETURN m.id, m.role, m.text, m.seq, m.at, m.agent
ORDER BY m.seq
"""

# Full-text index on message text (FalkorDB runs it on RediSearch): built as messages are
# written, so a search is an index lookup, not a scan of every message.
CREATE_MESSAGE_TEXT_INDEX = "CALL db.idx.fulltext.createNodeIndex('Message', 'text')"
# The same for fact text, so a new fact can be compared with similar ones under any entity.
CREATE_FACT_TEXT_INDEX = "CALL db.idx.fulltext.createNodeIndex('Fact', 'text')"

# search past chats: messages matching $q, ranked by the index's relevance score, only from
# sessions run by this agent's user, optionally leaving out one session (the current chat).
# The index covers the whole graph (one org), so the owner filter is what keeps users apart.
# Params: me, q, limit, exclude
SEARCH_HISTORY = """
CALL db.idx.fulltext.queryNodes('Message', $q) YIELD node, score
MATCH (s:Session)-[:HAS_MESSAGE]->(node)
WHERE ($exclude IS NULL OR s.id <> $exclude) AND node.role IN ['user', 'agent']
MATCH (me:Agent {id:$me})
MATCH (a:Agent)-[:RAN]->(s) WHERE a.owner = me.owner
WITH DISTINCT node, score, s
RETURN node.id, node.role, node.agent, node.text, node.at, s.id, score
ORDER BY score DESC, node.at DESC
LIMIT $limit
"""

# Read views for the UI (change nothing)

# Every agent in this org's graph. Agents are not private. Params: none
LIST_AGENTS = """
MATCH (a:Agent)
RETURN a.id, a.name, a.owner
ORDER BY a.owner, a.id
"""

# An agent's record: its claims (facts it asserted that did not come from the user) and its
# mistakes (those claims later retracted, by retract or correct). Superseded facts are not
# mistakes (they were true, then changed), and facts the user told it never count: if the user
# changes their story, the agent was not wrong. Counted live from the graph, so there is no
# stored number to drift or race. Params: ids
AGENT_RECORDS = """
MATCH (a:Agent) WHERE a.id IN $ids
OPTIONAL MATCH (a)-[:ASSERTED]->(c:Fact) WHERE coalesce(c.source, 'agent') <> 'user'
RETURN a.id, count(c) AS claims, sum(CASE WHEN c.status = 'retracted' THEN 1 ELSE 0 END) AS mistakes
"""

# This agent's user's chats (sessions any of its agents ran), most recent activity first, with
# the first user message as the title. Params: me
LIST_CHATS = """
MATCH (me:Agent {id:$me})
MATCH (a:Agent)-[:RAN]->(s:Session) WHERE a.owner = me.owner
WITH DISTINCT s
OPTIONAL MATCH (s)-[:HAS_MESSAGE]->(m:Message)
WITH s, m ORDER BY m.seq
WITH s, collect(m) AS ms
RETURN s.id, s.started_at, [x IN ms WHERE x.role = 'user' | x.text][0] AS title, size(ms),
       CASE WHEN size(ms) > 0 THEN ms[size(ms)-1].at ELSE s.started_at END AS last
ORDER BY last DESC
"""

# graph view step 1: the facts, decisions and procedures this agent may see (shared, or its
# user's own), with their author. Same rule as VISIBLE_ITEMS. Params: me, limit
GRAPH_ITEMS = """
MATCH (me:Agent {id:$me})
MATCH (author:Agent)-[:ASSERTED]->(x) WHERE labels(x)[0] IN ['Fact', 'Decision', 'Procedure']
  AND (x.visibility IS NULL OR x.visibility = 'shared' OR author.id = me.id OR author.owner = me.owner)
RETURN x.id, labels(x)[0], coalesce(x.text, x.name), x.status, coalesce(x.visibility, 'shared'),
       author.id, x.recorded_at
ORDER BY x.recorded_at DESC
LIMIT $limit
"""

# graph view step 2: what those items point to: entities, verifying agents, and the messages they
# were taken from (the caller keeps only messages this agent may read). Params: ids
GRAPH_NEIGHBOURS = """
MATCH (x)-[r:ABOUT|EXTRACTED_FROM]->(n) WHERE x.id IN $ids
RETURN DISTINCT n.id AS id, labels(n)[0] AS label, coalesce(n.name, n.text) AS text, n.role AS role
UNION
MATCH (v:Agent)-[:VERIFIED]->(x) WHERE x.id IN $ids
RETURN DISTINCT v.id AS id, 'Agent' AS label, v.name AS text, null AS role
"""

# graph view step 3: the edges between nodes already chosen, so no edge leads to a hidden node.
# Params: ids
GRAPH_EDGES = """
MATCH (a)-[r]->(b) WHERE a.id IN $ids AND b.id IN $ids
RETURN a.id, type(r), b.id, r.result
"""

# Which of these message ids can this agent read? Params: me, ids
READABLE_MESSAGES = """
MATCH (me:Agent {id:$me})
MATCH (a:Agent)-[:RAN]->(:Session)-[:HAS_MESSAGE]->(m:Message) WHERE m.id IN $ids
  AND (a.id = me.id OR a.owner = me.owner)
RETURN DISTINCT m.id
"""

# Messages behind an item or anything it was derived from. Text is returned for everyone; the
# caller blanks it for messages the agent may not read. Params: id
EVIDENCE = """
MATCH (x {id:$id})-[:DERIVED_FROM*0..]->(src)-[:EXTRACTED_FROM]->(m:Message)<-[:HAS_MESSAGE]-(s:Session)
RETURN src.id, m.id, m.role, m.text, m.seq, s.id
ORDER BY src.id, m.seq
"""

# Procedures (procedural memory)
# A Procedure is a saved how-to: a name and an ordered list of steps (a property, not Step nodes),
# ABOUT an entity (like a task or an organisation) so it can be found. It may be derived from
# facts/decisions, so retracting or superseding a source flags it (RETRACT/SUPERSEDE follow
# DERIVED_FROM with no label filter); confirm works as for facts. It is a leaf: nothing can be
# derived from a procedure (USABLE_SOURCES only allows facts and decisions). Private until
# publish (PUBLISH accepts it). A new version SUPERSEDES the old one; retire ends it.

# learn: write a private procedure. Source check and write are one query. Returns no row if the
# agent did not run the session or a source is not an active fact/decision.
# Params: agent, session, entity, entity_name, entity_type, id, name, steps, now, derived_from
LEARN_PROCEDURE = """
MATCH (a:Agent {id:$agent})-[:RAN]->(s:Session {id:$session})
OPTIONAL MATCH (src) WHERE src.id IN $derived_from AND labels(src)[0] IN ['Fact', 'Decision']
                       AND src.status = 'active'
WITH a, s, collect(DISTINCT src) AS sources
WHERE size(sources) = size($derived_from)
MERGE (e:Entity {id:$entity})
  ON CREATE SET e.name = $entity_name, e.type = $entity_type
CREATE (p:Procedure {id:$id, name:$name, steps:$steps, visibility:'private', status:'active',
                     recorded_at:$now})
CREATE (a)-[:ASSERTED]->(p), (p)-[:ABOUT]->(e), (p)-[:IN_SESSION]->(s)
WITH p, sources
FOREACH (src IN sources | CREATE (p)-[:DERIVED_FROM]->(src))
RETURN p.id
"""

# revise: replace an active/flagged procedure with a new version (same visibility and ABOUT
# links, its own sources). If the old one is shared, the new sources must be shared too. Only the
# author's owner can do it. Returns no row, and writes nothing, otherwise.
# Params: old_id, new_id, name (null keeps the old name), steps, agent, session, now, derived_from
REVISE_PROCEDURE = """
MATCH (old:Procedure {id:$old_id}) WHERE old.status IN ['active', 'flagged']
MATCH (a:Agent {id:$agent})-[:RAN]->(s:Session {id:$session})
MATCH (author:Agent)-[:ASSERTED]->(old) WHERE author.id = a.id OR author.owner = a.owner
OPTIONAL MATCH (src) WHERE src.id IN $derived_from AND labels(src)[0] IN ['Fact', 'Decision']
                       AND src.status = 'active'
                       AND (old.visibility <> 'shared' OR src.visibility IS NULL OR src.visibility = 'shared')
WITH old, a, s, collect(DISTINCT src) AS sources
WHERE size(sources) = size($derived_from)
SET old.status = 'superseded'
CREATE (new:Procedure {id:$new_id, name:coalesce($name, old.name), steps:$steps,
                       visibility:old.visibility, status:'active', recorded_at:$now})
CREATE (new)-[:SUPERSEDES]->(old), (a)-[:ASSERTED]->(new), (new)-[:IN_SESSION]->(s)
WITH old, new, sources
MATCH (old)-[:ABOUT]->(e)
CREATE (new)-[:ABOUT]->(e)
WITH DISTINCT new, sources
FOREACH (src IN sources | CREATE (new)-[:DERIVED_FROM]->(src))
RETURN new.id
"""

# retire: the procedure is obsolete or wrong. A leaf, so nothing downstream to flag, and no accuracy
# change. Only the author's owner. Params: id, me, reason, now
RETIRE_PROCEDURE = """
MATCH (p:Procedure {id:$id}) WHERE p.status IN ['active', 'flagged']
MATCH (me:Agent {id:$me})
MATCH (author:Agent)-[:ASSERTED]->(p) WHERE author.id = me.id OR author.owner = me.owner
SET p.status = 'retired', p.retired_at = $now, p.retired_by = $me, p.retire_reason = $reason
RETURN p.id
"""

# Active procedures about an entity that this agent may see, with the ids they were built on.
# Flagged, superseded and retired ones are left out. Params: entity, me
RECALL_PROCEDURES = """
MATCH (me:Agent {id:$me})
MATCH (author:Agent)-[:ASSERTED]->(p:Procedure)-[:ABOUT]->(:Entity {id:$entity})
WHERE p.status = 'active'
  AND (p.visibility = 'shared' OR author.id = me.id OR author.owner = me.owner)
OPTIONAL MATCH (p)-[:DERIVED_FROM]->(src)
RETURN p.id, p.name, p.steps, collect(DISTINCT src.id)
ORDER BY p.id
"""

# Verification

# verify: record the outcome of an evidence check (web or doc) on an active fact this agent may
# see. Writes a VERIFIED edge for every outcome and changes nothing else: evidence is about the
# fact, so a supported check makes that fact trusted, but it is not a score for the agent.
# Checks on one fact get strictly increasing times (one ms after the last if the clock has not
# moved), so "the latest check wins" always has an answer.
# Params: me, id, method, result, sources, snippet, now
VERIFY = """
MATCH (me:Agent {id:$me})
MATCH (author:Agent)-[:ASSERTED]->(f:Fact {id:$id})
WHERE f.status = 'active'
  AND (f.visibility = 'shared' OR author.id = me.id OR author.owner = me.owner)
OPTIONAL MATCH (:Agent)-[old:VERIFIED]->(f)
WITH me, f, max(old.at) AS last
CREATE (me)-[:VERIFIED {method:$method, result:$result, sources:$sources, snippet:$snippet,
                        at: CASE WHEN last IS NULL OR $now > last THEN $now ELSE last + 1 END}]->(f)
RETURN f.id
"""


# Automatic upkeep

# Candidates for "is this new fact a copy of one we have?": active facts this agent may see,
# (1) about any of the new fact's entities, or (2) matching its words in the full-text index under
# any entity (that is how a copy saved under another entity is caught). The caller keeps only those
# with enough words in common. `yours` = the agent's user may change it. Params: me, entities, q, limit
SIMILAR_BY_ENTITY = """
MATCH (me:Agent {id:$me})
MATCH (author:Agent)-[:ASSERTED]->(f:Fact)-[:ABOUT]->(e:Entity)
WHERE e.id IN $entities AND f.status = 'active'
  AND (f.visibility = 'shared' OR author.id = me.id OR author.owner = me.owner)
RETURN DISTINCT f.id, f.text, author.id, author.owner = me.owner
"""
SIMILAR_BY_TEXT = """
CALL db.idx.fulltext.queryNodes('Fact', $q) YIELD node, score
WITH node AS f, score WHERE f.status = 'active'
MATCH (me:Agent {id:$me})
MATCH (author:Agent)-[:ASSERTED]->(f)
WHERE f.visibility = 'shared' OR author.id = me.id OR author.owner = me.owner
RETURN f.id, f.text, author.id, author.owner = me.owner
ORDER BY score DESC
LIMIT $limit
"""

# Facts of my user's agents that another user's evidence check contradicts, and that no check has
# supported since: my agents should look at them (supersede, correct, or check them again).
# Another user can never edit my facts, so this is how they tell me. Params: me
DISPUTES = """
MATCH (me:Agent {id:$me})
MATCH (author:Agent)-[:ASSERTED]->(f:Fact {status:'active'})
WHERE author.id = me.id OR author.owner = me.owner
MATCH (by:Agent)-[x:VERIFIED {result:'contradicted'}]->(f) WHERE by.owner <> me.owner
OPTIONAL MATCH (:Agent)-[v:VERIFIED {result:'supported'}]->(f)
WITH f, author, by, x, max(v.at) AS supported_at
WHERE supported_at IS NULL OR supported_at < x.at
RETURN f.id, f.text, author.id, by.id, x.snippet, x.sources, x.at
ORDER BY x.at DESC
"""

# What the automatic sharing rule needs to know about a fact. Params: id
FACT_INFO = """
MATCH (author:Agent)-[:ASSERTED]->(f:Fact {id:$id})
OPTIONAL MATCH (f)-[:ABOUT]->(e:Entity)
RETURN f.visibility, f.status, coalesce(f.source, 'agent'), author.owner, collect(e.id)
"""

# A new decision redoes flagged ones: they become superseded (history, out of every flagged list)
# and the new one points to them. Only flagged decisions of the same user. Params: new, old, me
REPLACE_DECISIONS = """
MATCH (me:Agent {id:$me})
MATCH (new:Decision {id:$new})
MATCH (author:Agent)-[:ASSERTED]->(old:Decision)
WHERE old.id IN $old AND old.status = 'flagged' AND (author.id = me.id OR author.owner = me.owner)
SET old.status = 'superseded', old.superseded_at = timestamp()
CREATE (new)-[:SUPERSEDES]->(old)
RETURN old.id
"""

# Link a fact that was just written (by supersede) to the items it is built on. The caller has
# checked them with USABLE_SOURCES; only active ones are linked. Params: id, ids
LINK_SOURCES = """
MATCH (n:Fact {id:$id})
MATCH (src) WHERE src.id IN $ids AND labels(src)[0] IN ['Fact', 'Decision'] AND src.status = 'active'
CREATE (n)-[:DERIVED_FROM]->(src)
"""
