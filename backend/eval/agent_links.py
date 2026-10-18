"""Eval B: do real agents record what they built on?

Ten hand-written situations are run through the real Researcher, Planner and Writer (Claude Haiku)
in two chat turns each: first the user states some facts, then asks for a plan and a draft.
Afterwards the graph is checked, with no model involved:

  truth     an item (a decision or a draft the agents wrote) relied on a stated fact if its text
            contains that fact's value (a hand-written pattern per fact). This is a lower bound:
            an item can rely on a fact without repeating its value.
  recorded  the item is in blast_radius(fact), that is, a DERIVED_FROM path leads back to it.

Results go to eval/results/agent_links.md; the raw runs to eval/data/agent_runs.json, so grading
can be redone for free:  uv run python eval/agent_links.py --regrade

Run from backend/:  uv run python eval/agent_links.py [--only N]   (about $0.3 on Haiku 5.5)
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from memory import Memory  # noqa: E402
from llm import Spend, team_turn  # noqa: E402
from tools import AgentTools  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE / "data" / "agent_runs.json"
RESULTS = HERE / "results" / "agent_links.md"
SPEND_CAP = 1.00

# Each situation: who, what they tell the team, what they ask for, and the facts with the pattern
# that finds each one's value in a later text (numbers are compared with commas removed).
SCENARIOS = [
    {"name": "bakery", "owner": "Hannah",
     "tell": "I run a small bakery in Leeds. We sell about 220 sourdough loaves a week, flour has gone up to 1.15 pounds a kilo, and our oven only bakes 30 loaves per batch.",
     "ask": "Plan our weekly baking schedule from that, and write a short note to our regular customers about the flour price rise.",
     "facts": {"weekly sales": r"\b220\b", "flour price": r"\b1\.15\b", "oven batch size": r"\b30[- ](?:loaves|loaf)|batch(?:es)? of 30|30 per batch"}},
    {"name": "trek", "owner": "Tomas",
     "tell": "I'm planning a 12 day trek in Nepal in April with 3 friends. We each have 2,400 dollars to spend and the highest point of our route is 5,364 metres.",
     "ask": "Plan what we should book first and how to split our money, then draft an email to the trekking agency about permits and guides.",
     "facts": {"trip length": r"\b12[- ]day", "budget each": r"\b2400\b", "highest point": r"\b5364\b"}},
    {"name": "consulting", "owner": "Ines",
     "tell": "I freelance as a data consultant. I charge 95 euros an hour, I have 14 billable hours free each week, and my biggest client Orlin pays about 40 days late.",
     "ask": "Plan what monthly income I can count on from this, and write a polite but firm payment reminder to Orlin.",
     "facts": {"hourly rate": r"\b95\b", "free hours": r"\b14\b", "late payment": r"\b40\b"}},
    {"name": "moving", "owner": "Pedro",
     "tell": "We are moving to a two bedroom flat in Lisbon. It has no washing machine, the van we can rent holds 15 cubic metres, and the movers charge 85 euros an hour.",
     "ask": "Plan the move, including what to do about the washing machine, then write a message to our new landlord.",
     "facts": {"washing machine": r"washing machine|washer", "van size": r"\b15\b", "mover rate": r"\b85\b"}},
    {"name": "podcast", "owner": "Zainab",
     "tell": "My podcast gets about 1,800 listens an episode. A sponsor pays 22 dollars per thousand listens, and I publish every Tuesday.",
     "ask": "Plan a price for a four episode sponsor package for a new sponsor, then write the pitch email.",
     "facts": {"listens": r"\b1800\b", "sponsor rate": r"\b22\b", "publishing day": r"tuesday"}},
    {"name": "sciencefair", "owner": "Mr Okafor",
     "tell": "Our school science fair is on 9 March in the gym. 38 student projects are registered, and each one needs a table 1.2 metres wide.",
     "ask": "Plan the layout of the gym, then write the newsletter item for parents.",
     "facts": {"date": r"9 march|march 9", "projects": r"\b38\b", "table width": r"\b1\.2\b"}},
    {"name": "release", "owner": "Ravi",
     "tell": "Our mobile app version 4.2 ships on 3 September. The crash rate is 0.8 percent of sessions, and the payment SDK we use is deprecated at the end of October.",
     "ask": "Plan what has to be done before we ship, then write the release notes.",
     "facts": {"ship date": r"3 september|september 3", "crash rate": r"\b0\.8\b", "sdk deadline": r"end of october|october"}},
    {"name": "garden", "owner": "Marguerite",
     "tell": "Our community garden has 24 plots and a waiting list of 61 people. The water bill is 310 pounds a year.",
     "ask": "Plan the yearly fee per plot and how to handle the waiting list, then write the notice for the gate.",
     "facts": {"plots": r"\b24\b", "waiting list": r"\b61\b", "water bill": r"\b310\b"}},
    {"name": "grant", "owner": "Dr Lindqvist",
     "tell": "Our lab grant is 180,000 dollars over three years. The postdoc costs 72,000 a year with benefits, and the mass spectrometer service contract is 9,500 a year.",
     "ask": "Plan how to spend what is left of the budget, then write a short update email to our programme officer.",
     "facts": {"grant total": r"\b180000\b", "postdoc cost": r"\b72000\b", "service contract": r"\b9500\b"}},
    {"name": "tour", "owner": "Kasper",
     "tell": "Our band has a 9 city tour in November. The van costs about 0.31 euros a kilometre in fuel, and the longest drive is 640 km from Brno to Hamburg.",
     "ask": "Plan the driving and the fuel budget, then write an email to our booking agent about the long drive.",
     "facts": {"cities": r"\b9[- ]cit", "fuel cost": r"\b0\.31\b", "longest drive": r"\b640\b"}},
]


def norm(text):
    return re.sub(r"(?<=\d),(?=\d{3})", "", text or "").lower()


def run_scenario(sc, client, spend):
    """Run one situation on a fresh graph. Returns what the graph holds afterwards."""
    mem = Memory(f"eval_links_{sc['name']}")
    mem.drop()
    mem.setup()
    r, p, w = (mem.register_agent(n, owner=sc["owner"]) for n in ("Researcher", "Planner", "Writer"))
    chat = mem.start_session(r, team=[p, w])
    team = {a.rsplit("-", 1)[-1]: AgentTools(mem, a, chat) for a in (r, p, w)}
    steps = []
    for text in (sc["tell"], sc["ask"]):
        out = team_turn(team, text, client=client, spend=spend, owner=sc["owner"])
        steps += [{"role": s["role"], "reply": s["reply"][:300], "stop": s["stop"], "calls": [{"tool": c["tool"], "args": c["args"], "ok": "error" not in c["result"]}
                                                for c in s["tool_calls"]]} for s in out["steps"]]
    rows = mem.g.query("MATCH (n) WHERE n:Fact OR n:Decision RETURN n.id, labels(n)[0], n.text, n.source, n.status").result_set
    items = [{"id": i, "kind": k, "text": t, "source": s, "status": st} for i, k, t, s, st in rows]
    edges = mem.g.query("MATCH (a)-[:DERIVED_FROM]->(b) RETURN a.id, b.id").result_set
    blast = {i["id"]: [x["id"] for x in mem.blast_radius(r, i["id"])] for i in items}
    mem.drop()
    return {"name": sc["name"], "items": items, "edges": [list(e) for e in edges], "blast": blast, "steps": steps}


def grade(sc, run):
    """Per scenario: truth pairs, recorded pairs, and the per-item picture."""
    items = {i["id"]: i for i in run["items"]}
    downstream = [i for i in run["items"] if i["kind"] == "Decision" or i["source"] == "agent"]
    has_link = {a for a, _ in run["edges"]}
    out = {"items": len(downstream), "no_source": [], "pairs": 0, "found": 0, "missed": [],
           "flagged": 0, "flagged_unmentioned": 0, "facts_found": 0, "facts_total": len(sc["facts"])}
    for i in downstream:
        if i["id"] not in has_link:
            out["no_source"].append(i["text"][:80])
    for label, pattern in sc["facts"].items():
        roots = [i for i in run["items"] if i["kind"] == "Fact" and i["source"] != "agent"
                 and re.search(pattern, norm(i["text"]))]
        if not roots:
            continue
        out["facts_found"] += 1
        truth = {i["id"] for i in downstream if re.search(pattern, norm(i["text"]))}
        flagged = {x for r in roots for x in run["blast"][r["id"]]} & {i["id"] for i in downstream}
        out["pairs"] += len(truth)
        out["found"] += len(truth & flagged)
        out["flagged"] += len(flagged)
        out["flagged_unmentioned"] += len(flagged - truth)
        for t in sorted(truth - flagged):
            out["missed"].append(f"{label}: {items[t]['text'][:90]}")
    return out


def report(graded, runs, cost):
    tot = lambda k: sum(g[k] for g in graded)
    pairs, found = tot("pairs"), tot("found")
    n_items, n_orphans = tot("items"), sum(len(g["no_source"]) for g in graded)
    flagged, unmentioned = tot("flagged"), tot("flagged_unmentioned")
    lines = [
        "# Do real agents record what they built on? (eval B)", "",
        f"10 hand-written situations, run through the real Researcher, Planner and Writer (Claude Haiku 5.5) in two chat turns each: "
        f"the user states three facts, then asks for a plan and a draft. Cost ${cost:.2f}. "
        "Nothing was told to the agents about this test. A first pilot on one situation used an open-ended request that needed a number the user never gave, so the Researcher asked for it and wrote no plan; the requests were then reworded to be answerable from the stated facts (the pilot is not counted).", "",
        "**Truth.** A decision or draft relied on a stated fact if its text contains that fact's value. This is a lower bound: an item can rely on a fact without repeating the value.",
        "**Recorded.** The item is in blast_radius(fact): a DERIVED_FROM path leads back to the fact.", "",
        f"- Decisions and drafts written by the agents: **{n_items}**. Of these, **{n_orphans}** recorded no source at all.",
        f"- Situations where a decision was written: **{sum(1 for r in runs if any(i['kind'] == 'Decision' for i in r['items']))} of {len(runs)}**; where a draft was written: **{sum(1 for r in runs if any(i['source'] == 'agent' and i['kind'] == 'Fact' for i in r['items']))} of {len(runs)}**.",
        f"- Stated facts the Researcher saved: **{tot('facts_found')} of {tot('facts_total')}**.",
        f"- Dependencies that show in the text: **{pairs}**. Recorded: **{found}** (**{found / pairs:.0%}**), missed: {pairs - found} ({(pairs - found) / pairs:.0%}).",
        f"- Items flagged that do not repeat the fact's value: {unmentioned} of {flagged} ({unmentioned / flagged:.0%}). "
        "These may be real dependencies stated in other words, or links that are too generous.", "",
        "## Per situation", "", "| Situation | Facts saved | Decisions and drafts | Dependencies in text | Recorded | Missed |", "|---|---|---|---|---|---|"]
    for sc, g in zip(SCENARIOS, graded):
        lines.append(f"| {sc['name']} | {g['facts_found']}/{g['facts_total']} | {g['items']} | {g['pairs']} | {g['found']} | {g['pairs'] - g['found']} |")
    miss = [m for g in graded for m in g["missed"]]
    orph = [m for g in graded for m in g["no_source"]]
    if miss:
        lines += ["", "## Missed dependencies (value appears, no path recorded)", ""] + [f"- {m}" for m in miss]
    if orph:
        lines += ["", "## Items that recorded no source", ""] + [f"- {m}" for m in orph]
    lines += ["", "## Limits", "",
              "- Ten short situations written for this test, one model (Haiku 5.5), one run each. Real agents face longer and messier work.",
              "- The truth rule is mechanical. It misses dependencies that are not repeated in the text, and it counts a repeated number as reliance even when it is a coincidence.",
              "- Each situation has one user and a fresh graph, so this measures recording, not recall across many sessions."]
    return "\n".join(lines) + "\n"


def main():
    only = int(sys.argv[sys.argv.index("--only") + 1]) if "--only" in sys.argv else None
    chosen = SCENARIOS[:only] if only else SCENARIOS
    if "--regrade" in sys.argv:
        runs = json.loads(RUNS.read_text())
        cost = runs["cost"]
        runs = runs["runs"]
    else:
        import anthropic
        client, spend, runs = anthropic.Anthropic(), Spend(cap=SPEND_CAP), []
        for sc in chosen:
            before = spend.total
            runs.append(run_scenario(sc, client, spend))
            print(f"{sc['name']}: ${spend.total - before:.3f} (total ${spend.total:.3f})", flush=True)
        cost = spend.total
        RUNS.parent.mkdir(exist_ok=True)
        RUNS.write_text(json.dumps({"cost": cost, "runs": runs}, indent=1, default=str))
    graded = [grade(sc, run) for sc, run in zip(chosen, runs)]
    text = report(graded, runs, cost)
    if only is None or "--regrade" in sys.argv:
        RESULTS.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
