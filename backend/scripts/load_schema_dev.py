"""Rebuild the `schema_dev` graph: indexes + demo-story seed, then verify with queries."""
import os
import time
from dotenv import load_dotenv
from falkordb import FalkorDB

load_dotenv()

db = FalkorDB(
    host=os.environ["FALKORDB_HOST"],
    port=int(os.environ["FALKORDB_PORT"]),
    username=os.environ["FALKORDB_USERNAME"],
    password=os.environ["FALKORDB_PASSWORD"],
)

GRAPH = "schema_dev"
if GRAPH in db.list_graphs():
    db.select_graph(GRAPH).delete()  # start clean so reruns don't duplicate
g = db.select_graph(GRAPH)

for label in ("Fact", "Decision", "Agent", "Entity", "Handoff"):
    g.query(f"CREATE INDEX FOR (n:{label}) ON (n.id)")

DAY = 24 * 3600 * 1000
now = int(time.time() * 1000)
P = {"now": now, "long_ago": now - 90 * DAY, "moved_at": now - 30 * DAY}

# Lineage chain: f1 <- f2 <- d1 <- d2 (DERIVED_FROM points derived -> source).
# f3 is an unrelated control; f4 supersedes f3 (HQ moved 30 days ago, recorded today).
# valid_from/valid_to = when it was true in the world; recorded_at = when we wrote it down.
g.query(
    """
    CREATE (res:Agent {id:'agent:researcher', name:'Researcher', trust:1.0}),
           (pla:Agent {id:'agent:planner',    name:'Planner',    trust:1.0}),
           (wri:Agent {id:'agent:writer',     name:'Writer',     trust:1.0}),
           (s1:Session {id:'sess:1', started_at:$now}),
           (s2:Session {id:'sess:2', started_at:$now}),
           (s3:Session {id:'sess:3', started_at:$now}),
           (s4:Session {id:'sess:4', started_at:$now}),
           (acme:Entity {id:'org:acme', name:'Acme Corp', type:'org'}),
           (f1:Fact {id:'fact:1', text:'Acme has 500 employees', visibility:'shared', status:'active', confidence:0.8, valid_from:$long_ago, recorded_at:$now}),
           (f2:Fact {id:'fact:2', text:'Acme is an enterprise-size account', visibility:'shared', status:'active', confidence:0.7, valid_from:$long_ago, recorded_at:$now}),
           (f3:Fact {id:'fact:3', text:'Acme HQ is in Austin', visibility:'shared', status:'superseded', confidence:0.9, valid_from:$long_ago, valid_to:$moved_at, recorded_at:$now}),
           (f4:Fact {id:'fact:4', text:'Acme HQ is in Dallas', visibility:'shared', status:'active', confidence:0.9, valid_from:$moved_at, recorded_at:$now}),
           (d1:Decision {id:'dec:1', text:'Pitch the enterprise plan to Acme', status:'active', created_at:$now, recorded_at:$now}),
           (d2:Decision {id:'dec:2', text:'Write the enterprise pitch deck', status:'active', created_at:$now, recorded_at:$now}),
           (h1:Handoff {id:'handoff:1', at:$now}),
           (h2:Handoff {id:'handoff:2', at:$now}),
           (res)-[:RAN]->(s1), (pla)-[:RAN]->(s2), (wri)-[:RAN]->(s3), (res)-[:RAN]->(s4),
           (res)-[:ASSERTED]->(f1), (res)-[:ASSERTED]->(f3), (res)-[:ASSERTED]->(f4),
           (pla)-[:ASSERTED]->(f2), (pla)-[:ASSERTED]->(d1), (wri)-[:ASSERTED]->(d2),
           (f1)-[:ABOUT]->(acme), (f2)-[:ABOUT]->(acme), (f3)-[:ABOUT]->(acme), (f4)-[:ABOUT]->(acme),
           (f1)-[:IN_SESSION]->(s1), (f3)-[:IN_SESSION]->(s1), (f2)-[:IN_SESSION]->(s2),
           (d1)-[:IN_SESSION]->(s2), (d2)-[:IN_SESSION]->(s3), (f4)-[:IN_SESSION]->(s4),
           (f2)-[:DERIVED_FROM]->(f1), (d1)-[:DERIVED_FROM]->(f2), (d2)-[:DERIVED_FROM]->(d1),
           (f4)-[:SUPERSEDES]->(f3),
           (res)-[:SENT]->(h1), (h1)-[:TO]->(pla), (h1)-[:CARRIED]->(f1), (h1)-[:CARRIED]->(f3),
           (pla)-[:SENT]->(h2), (h2)-[:TO]->(wri), (h2)-[:CARRIED]->(f2), (h2)-[:CARRIED]->(d1)
    """,
    P,
)

nodes = dict(g.query("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY label").result_set)
edges = dict(g.query("MATCH ()-[r]->() RETURN type(r) AS edge, count(*) AS n ORDER BY edge").result_set)
print("nodes:", nodes, "total", sum(nodes.values()))
print("edges:", edges, "total", sum(edges.values()))

assert nodes == {"Agent": 3, "Decision": 2, "Entity": 1, "Fact": 4, "Handoff": 2, "Session": 4}, nodes
assert edges == {
    "ABOUT": 4, "ASSERTED": 6, "CARRIED": 4, "DERIVED_FROM": 3, "IN_SESSION": 6,
    "RAN": 4, "SENT": 2, "SUPERSEDES": 1, "TO": 2,
}, edges


def ids(query, params=None):
    return sorted(row[0] for row in g.query(query, params).result_set)


# 1. Blast radius of fact:1 = everything built on it, directly or indirectly (f3/f4 not included)
blast = ids("MATCH (:Fact {id:'fact:1'})<-[:DERIVED_FROM*1..]-(x) RETURN DISTINCT x.id")
print("blast radius of fact:1:", blast)
assert blast == ["dec:1", "dec:2", "fact:2"], blast

# 2. Which agents were handed fact:1 directly
received = ids("MATCH (:Fact {id:'fact:1'})<-[:CARRIED]-(:Handoff)-[:TO]->(a:Agent) RETURN a.id")
print("agents handed fact:1:", received)
assert received == ["agent:planner"], received

# 3. Current facts about Acme (no valid_to = still valid)
current = ids("MATCH (f:Fact)-[:ABOUT]->(:Entity {id:'org:acme'}) WHERE f.valid_to IS NULL RETURN f.id")
print("current facts about Acme:", current)
assert current == ["fact:1", "fact:2", "fact:4"], current

# 4. As-of query: what was true 45 days ago
AS_OF = """
MATCH (f:Fact)-[:ABOUT]->(:Entity {id:'org:acme'})
WHERE f.valid_from <= $t AND (f.valid_to IS NULL OR f.valid_to > $t)
RETURN f.id
"""
past = ids(AS_OF, {"t": now - 45 * DAY})
print("true 45 days ago:", past)
assert past == ["fact:1", "fact:2", "fact:3"], past

print("OK: counts and query checks match the expected schema")
