"""Templated lineage scenarios for the rule checks (lifecycle.py) and the load tests (scale.py).

The wording comes from f-string templates, so only the shape of the graph matters here; the accuracy
evaluation (accuracy.py) uses the natural-language dataset instead.

Each scenario is one fictional company with three root facts (headcount, headquarters, funding),
items built on them (derived facts, Planner decisions, a procedure, Writer drafts) and look-alikes:
facts and decisions about the same company that share words with the roots but are NOT built on
them. Every non-root item is kept with probability KEEP (fixed seed) if all its sources were kept,
so chains, branches and depths differ between companies.

The answer key (what depends on what) is computed here in plain Python from the template links,
never read back from the database, so the graph is graded against something it did not produce.
"""
import random

from memory import _slug

SEED = 7
KEEP = 0.75

COMPANIES = [
    "Acme", "Borealis", "Cobalt Labs", "Driftwood", "Emberline", "Fennec", "Granite Works", "Halcyon",
    "Ironleaf", "Juniper Systems", "Kestrel", "Lumen Bay", "Marlow", "Nimbus Freight", "Orchid Health",
    "Pinecrest", "Quarry Analytics", "Redwater", "Saltmarsh", "Tidewell", "Umbra Games", "Vantage Rail",
    "Willowby", "Xenon Grid", "Yarrow Foods", "Zephyr Air", "Alder Finance", "Brightfold", "Cinder Robotics",
    "Dunmore Retail",
]
CITIES = ["Berlin", "Lisbon", "Toronto", "Austin", "Pune", "Osaka", "Dublin", "Nairobi", "Lyon", "Denver"]
OUTSIDERS = ["Northwind", "Contoso", "Globex", "Initech", "Umbrella", "Hooli"]  # never a scenario company

# key: (kind, author role, text template, source keys). kind is fact, decision or procedure.
# Procedures have "name | step; step; step" text. Roots have no sources.
ROOTS = {
    "R1": ("fact", "researcher", "{co} has {n} employees", ()),
    "R2": ("fact", "researcher", "{co} is headquartered in {city}", ()),
    "R3": ("fact", "researcher", "{co} raised ${money}M in its last funding round", ()),
}
DERIVED = {
    "A1": ("fact", "researcher", "{co} counts as a mid-sized company", ("R1",)),
    "A2": ("decision", "planner", "Pitch the mid-market plan to {co}", ("A1",)),
    "A3": ("procedure", "planner",
           "Mid-market pitch | Open with the team-size case study; Show seat pricing; Offer a 30-day pilot",
           ("A2",)),
    "A4": ("decision", "planner", "Book an offsite venue for {n} people", ("R1",)),
    "A5": ("fact", "writer", "Draft: catering budget of EUR {budget}k", ("A4",)),
    "A6": ("decision", "planner", "Ask finance to approve EUR {budget}k", ("A5",)),
    "B1": ("decision", "planner", "Hold the kickoff meeting in {city}", ("R2",)),
    "B2": ("fact", "writer", "Draft: offsite invite for {city}, venue for {n} guests", ("B1", "A4")),
    "B3": ("decision", "planner", "Send the invite to the team leads", ("B2",)),
    "B4": ("fact", "researcher", "The team travels by train, two nights in hotels", ("R2",)),
    "C1": ("fact", "researcher", "{co} has budget for a premium plan", ("R3",)),
    "C2": ("decision", "planner", "Offer {co} the premium tier at the mid-market price", ("C1", "A2")),
    "C3": ("fact", "writer", "Draft: proposal email with premium pricing and a pilot", ("C2",)),
}
LOOKALIKES = {  # same company or same wording, no link to any root
    "X1": ("fact", "researcher", "{co} has {offices} offices", ()),
    "X2": ("decision", "planner", "Ship one welcome kit to each {co} office", ("X1",)),
    "X3": ("fact", "researcher", "{co} was founded in {year}", ()),
    "X4": ("fact", "researcher", "{other} has {n} employees", ()),
    "X5": ("fact", "researcher", "{co} employees use Slack every day", ()),
    "X6": ("decision", "planner", "Invite {co} to the {city2} user meetup", ("X3",)),
}


def build(seed=SEED, keep=KEEP):
    """Return a list of scenarios. Each is {"company", "items": {key: item}} where an item is
    {"kind", "role", "text", "sources": [keys]}, in an order where sources come first."""
    rng = random.Random(seed)
    scenarios = []
    for i, co in enumerate(COMPANIES):
        n = rng.choice([40, 120, 250, 500, 800, 1500])
        city, city2 = rng.sample(CITIES, 2)
        values = dict(co=co, n=n, city=city, city2=city2, money=rng.choice([5, 12, 30, 80]),
                      budget=round(n * 0.08), offices=rng.randint(2, 9),
                      year=rng.randint(1985, 2020), other=OUTSIDERS[i % len(OUTSIDERS)])
        items = {}
        for group in (ROOTS, DERIVED, LOOKALIKES):
            for key, (kind, role, text, sources) in group.items():
                if not all(s in items for s in sources):
                    continue
                if group is not ROOTS and sources and rng.random() > keep:
                    continue
                items[key] = {"kind": kind, "role": role, "text": text.format(**values),
                              "sources": list(sources)}
        scenarios.append({"company": co, "items": items})
    return scenarios


def descendants(items, key):
    """{key: depth} of every item built on `key`, directly (1) or through others (shortest path)."""
    children = {}
    for k, item in items.items():
        for s in item["sources"]:
            children.setdefault(s, []).append(k)
    out, frontier, depth = {}, [key], 0
    while frontier:
        depth += 1
        nxt = []
        for k in frontier:
            for c in children.get(k, []):
                if c not in out:
                    out[c] = depth
                    nxt.append(c)
        frontier = nxt
    return out


def queries(scenarios):
    """The wrong-fact questions: every fact that something was built on. Yields
    (scenario index, fact key, {descendant key: depth})."""
    for i, sc in enumerate(scenarios):
        for key, item in sc["items"].items():
            if item["kind"] != "fact":
                continue
            truth = descendants(sc["items"], key)
            if truth:
                yield i, key, truth


if __name__ == "__main__":
    scs = build()
    qs = list(queries(scs))
    n_items = sum(len(s["items"]) for s in scs)
    print(f"{len(scs)} scenarios, {n_items} items, {len(qs)} queries")
    for k, item in scs[0]["items"].items():
        print(f"  {k:3} {item['kind']:9} {item['text']}  <- {item['sources']}")


def load(mem, scs, forget=0.0, seed=0):
    """Write every scenario through the Memory API. With `forget`, each source link is left out
    with that probability (an agent that did not record what it built on). Returns
    (agents by role, per scenario {key: node id})."""
    rng = random.Random(seed)
    agents = {role: mem.register_agent(role.title(), owner="Eval") for role in ("researcher", "planner", "writer")}
    sessions = {role: mem.start_session(a) for role, a in agents.items()}
    ids = []
    for sc in scs:
        entity = f"org:{_slug(sc['company'])}"
        m = {}
        for key, item in sc["items"].items():
            srcs = [m[s] for s in item["sources"] if not (forget and rng.random() < forget)]
            a, s = agents[item["role"]], sessions[item["role"]]
            if item["kind"] == "fact":
                m[key] = mem.remember(a, s, item["text"], entity, entity_name=sc["company"],
                                      entity_type="org", derived_from=srcs, source="agent")
            elif item["kind"] == "decision":
                m[key] = mem.decide(a, s, item["text"], derived_from=srcs)
            else:
                name, steps = item["text"].split(" | ")
                m[key] = mem.learn_procedure(a, s, name, steps.split("; "), "topic:pitch",
                                             derived_from=srcs)
        ids.append(m)
    return agents, ids
