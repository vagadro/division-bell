"""Test the four core queries against a freshly rebuilt `schema_dev` graph."""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from falkordb import FalkorDB

import queries

load_dotenv(ROOT / ".env")

# Reset the graph to the known seed first.
subprocess.run([sys.executable, str(ROOT / "scripts" / "load_schema_dev.py")],
               check=True, cwd=ROOT, capture_output=True)

db = FalkorDB(
    host=os.environ["FALKORDB_HOST"],
    port=int(os.environ["FALKORDB_PORT"]),
    username=os.environ["FALKORDB_USERNAME"],
    password=os.environ["FALKORDB_PASSWORD"],
)
g = db.select_graph("schema_dev")
now = int(time.time() * 1000)


def rows(query, params):
    return g.query(query, params).result_set


def recall(me):
    # collect() has no guaranteed order, so sort built_on before comparing
    return {r[0]: sorted(r[2]) for r in rows(queries.RECALL, {"entity": "org:acme", "me": me})}


# --- recall (baseline): current Acme facts, with lineage attached ---
base = recall("agent:researcher")
print("recall baseline:", base)
assert sorted(base) == ["fact:1", "fact:2", "fact:4"], base
assert base["fact:1"] == [] and base["fact:2"] == ["fact:1"], base

# --- remember: Planner writes a private fact derived from fact:2 ---
res = g.query(queries.REMEMBER, {
    "agent": "agent:planner", "session": "sess:2", "entity": "org:acme",
    "entity_name": "Acme Corp", "entity_type": "org", "id": "fact:5",
    "text": "Acme likely needs SSO", "conf": 0.6, "valid_from": now, "now": now,
    "derived_from": ["fact:2"],
})
print("remember: nodes", res.nodes_created, "edges", res.relationships_created)
assert res.nodes_created == 1 and res.relationships_created == 4  # ASSERTED, ABOUT, IN_SESSION, DERIVED_FROM

# private means only the author sees it; lineage reaches back 2 hops
assert "fact:5" not in recall("agent:researcher")
planner = recall("agent:planner")
print("recall as planner:", planner)
assert planner["fact:5"] == ["fact:1", "fact:2"], planner

# remember with an unknown agent writes nothing
bad = g.query(queries.REMEMBER, {
    "agent": "agent:ghost", "session": "sess:2", "entity": "org:acme",
    "entity_name": "Acme Corp", "entity_type": "org", "id": "fact:x",
    "text": "x", "conf": 0.1, "valid_from": now, "now": now, "derived_from": [],
})
assert bad.nodes_created == 0

# --- supersede: HQ moves again (fact:4 Dallas -> fact:6 Houston) ---
sup = {"old_id": "fact:4", "new_id": "fact:6", "text": "Acme HQ is in Houston", "conf": 0.9,
       "agent": "agent:researcher", "session": "sess:4", "at": now, "now": now}
res = g.query(queries.SUPERSEDE, sup)
print("supersede: nodes", res.nodes_created, "edges", res.relationships_created)
assert res.nodes_created == 1 and res.relationships_created == 4  # SUPERSEDES, ASSERTED, IN_SESSION, ABOUT

now_facts = recall("agent:researcher")
print("recall after supersede:", sorted(now_facts))
assert sorted(now_facts) == ["fact:1", "fact:2", "fact:6"], now_facts
chain = sorted(r[0] for r in rows(
    "MATCH (:Fact {id:'fact:6'})-[:SUPERSEDES*]->(o) RETURN o.id", {}))
assert chain == ["fact:3", "fact:4"], chain  # history kept: Houston > Dallas > Austin
old = rows("MATCH (f:Fact {id:'fact:4'}) RETURN f.status, f.valid_to", {})[0]
assert old[0] == "superseded" and old[1] == now, old

# superseding an already-closed fact must create nothing
again = g.query(queries.SUPERSEDE, {**sup, "new_id": "fact:7"})
assert again.nodes_created == 0

# --- provenance: why does the Writer's deck (dec:2) exist? ---
prov = rows(queries.PROVENANCE, {"id": "dec:2"})
print("provenance of dec:2:", [(r[0], r[2], r[3]) for r in prov])
assert [(r[0], r[2], r[3]) for r in prov] == [
    ("dec:2", "agent:writer", 0),
    ("dec:1", "agent:planner", 1),
    ("fact:2", "agent:planner", 2),
    ("fact:1", "agent:researcher", 3),
], prov

print("OK: all four core queries behave as expected")
