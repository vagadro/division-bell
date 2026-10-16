"""The demo story, scripted: Researcher -> Planner -> Writer build on a fact, the fact is wrong,
one correction flags everything downstream, and only those parts are redone.

Every step goes through AgentTools.call(), the same door an LLM agent uses. The web check is a
MockChecker, so the run costs $0. The graph stays in tenant "story" afterwards for inspection
(FalkorDB Browser, graph tenant_story). Run: uv run python scripts/demo_story.py
"""
import json
import sys

sys.path.insert(0, ".")
from llm import MockChecker, Verdict
from memory import Tenants
from tools import AgentTools

tenants = Tenants()
tenants.drop("story")
mem = tenants.get("story")

checker = MockChecker({"Acme's HQ is in Berlin": Verdict(
    "supported", ["https://example.com/acme-about"], "Acme Corp, headquartered in Berlin")})


def show(title, data):
    print(f"\n--- {title}")
    print(json.dumps(data, indent=2, default=str) if not isinstance(data, str) else data)


def step(who, tool, **args):
    out = who.call(tool, args)
    if "error" in out:
        sys.exit(f"{tool} failed for {who.agent}: {out['error']}")
    return out


def ids(items):
    return sorted(i["id"] for i in items)


# Priya's Researcher, and Sam's Planner and Writer (two users, so sharing is explicit)
researcher = AgentTools.open(mem, "Researcher", owner="Priya", checker=checker)
planner = AgentTools.open(mem, "Planner", owner="Sam")
writer = AgentTools.open(mem, "Writer", owner="Sam")

# History: the Researcher has been wrong twice before (trust 1.0 -> 0.81, still no warning)
for i in (1, 2):
    old = step(researcher, "remember", text=f"Old claim {i}", entity="org:old")["id"]
    step(researcher, "retract", fact_id=old, reason="was wrong")

# Session 1 - Researcher: one wrong fact (really 50 employees) and one good one
f1 = step(researcher, "remember", text="Acme Corp has 500 employees", entity="org:acme",
          entity_name="Acme Corp", confidence=0.9)["id"]
f3 = step(researcher, "remember", text="Acme's HQ is in Berlin", entity="org:acme")["id"]
step(researcher, "publish", fact_id=f1)
step(researcher, "publish", fact_id=f3)

# Planner (different user) builds on the shared facts
seen = step(planner, "recall", entity="org:acme")["facts"]
show("Planner recalls org:acme (author trust shown, no warning yet)",
     [(f["id"], f["text"], round(f["trust"], 3), f["needs_verification"]) for f in seen])
d1 = step(planner, "decide", text="Pitch the enterprise plan to Acme", derived_from=[f1])["id"]
f2 = step(planner, "remember", text="Acme needs 20 onboarding seats", entity="org:acme",
          derived_from=[f1])["id"]
d2 = step(planner, "decide", text="Plan a Berlin on-site kickoff for the enterprise plan",
          derived_from=[d1, f3])["id"]
step(planner, "handoff", recipient=writer.agent, item_ids=[d1, d2, f2], note="Draft the proposal")

# Session 2 - Writer reads its inbox and writes a draft that rests on the Planner's decision
inbox = step(writer, "inbox")["handoffs"]
show("Writer's inbox", [(h["id"] if "id" in h else h) for h in inbox][:1])
f4 = step(writer, "remember", text="Draft: the enterprise plan fits Acme's 500 staff",
          entity="doc:proposal", derived_from=[d1])["id"]

# Dry run, then the correction (the Researcher finds out Acme has 50 employees)
show("Blast radius of the wrong fact (dry run)", [(a["id"], a["agent"], a["depth"])
                                                  for a in step(researcher, "blast_radius", item_id=f1)["affected"]])
fix = step(researcher, "correct", old_id=f1, text="Acme Corp has 50 employees",
           reason="HR page says 50, not 500")
show("correct() result", fix)
f1b = fix["new"]

# Sam's agents see what is flagged; the control fact is untouched
flagged = step(planner, "flagged")["items"]
show("Planner's flagged items", [(i["id"], "wrong because " + ",".join(i["wrong"])) for i in flagged])
show("Writer's flagged items", [(i["id"], "wrong because " + ",".join(i["wrong"]))
                                for i in step(writer, "flagged")["items"]])
assert {d1, d2, f2, f4} <= {i["id"] for i in flagged} | {x["id"] for x in step(writer, "flagged")["items"]}
control = [f for f in step(planner, "recall", entity="org:acme")["facts"] if f["id"] == f3]
show("Control fact f3 still active", [(f["id"], f["text"], round(f["trust"], 3), f["needs_verification"])
                                      for f in control])

# Redo only the flagged parts, on the corrected fact
d1b = step(planner, "decide", text="Pitch the starter plan to Acme", derived_from=[f1b])["id"]
d2b = step(planner, "decide", text="Plan a Berlin on-site kickoff for the starter plan",
           derived_from=[d1b, f3])["id"]
f4b = step(writer, "remember", text="Draft: the starter plan fits Acme's 50 staff",
           entity="doc:proposal", derived_from=[d1b])["id"]
show("Redone", {"decisions": [d1b, d2b], "draft fact": f4b})

# Trust in use: three mistakes now, so the Researcher's remaining fact carries a warning...
after = step(planner, "recall", entity="org:acme")["facts"]
show("Planner recalls org:acme after the correction",
     [(f["id"], f["text"], round(f["trust"], 3), f["needs_verification"]) for f in after])
# ...until outside evidence supports it (mock web check)
check = step(researcher, "verify_fact", fact_id=f3, method="web")
show("verify_fact(f3) via mock web check", {"result": check["verdict"]["result"],
                                            "sources": check["verdict"]["sources"],
                                            "trust": round(check["trust"], 3)})
final = step(planner, "recall", entity="org:acme")["facts"]
show("Planner recalls org:acme at the end",
     [(f["id"], f["text"], round(f["trust"], 3), f["needs_verification"], f["verified"]) for f in final])
print("\nGraph kept in tenant 'story' (FalkorDB graph: tenant_story).")
