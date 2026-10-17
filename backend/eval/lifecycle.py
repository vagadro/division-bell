"""Lifecycle evaluation: the real write path (supersede, retract, confirm, correct) on every
templated company, plus a sharing story between two users.

Every check is counted as passed or failed and written to eval/results/lifecycle.md. A failed
check prints what was expected and what came back. Uses the templated scenarios from synthetic.py: these
are rules about statuses and permissions, so only the shape of the graph matters.

Run from backend/:  uv run python eval/lifecycle.py   (cost $0, no LLM; graph eval_lifecycle kept)
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from memory import Memory, MemoryGraphError, _slug  # noqa: E402
import synthetic as scenarios  # noqa: E402
from synthetic import load  # noqa: E402

GRAPH = "eval_lifecycle"
RESULTS = Path(__file__).resolve().parent / "results" / "lifecycle.md"

results = defaultdict(lambda: [0, 0])  # check name -> [passed, total]
failures = []


def check(name, ok, detail=""):
    results[name][1] += 1
    if ok:
        results[name][0] += 1
    elif len(failures) < 20:
        failures.append(f"{name}: {detail}")


def refused(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except MemoryGraphError:
        return True
    return False


def status(mem, ids):
    rows = mem.g.query("MATCH (n) WHERE n.id IN $ids RETURN n.id, n.status", {"ids": list(ids)}).result_set
    return dict(rows)


def company_story(mem, agents, sessions, sc, m, outsider):
    """Supersede HQ, retract headcount, confirm, recall, correct funding, for one company."""
    items = sc["items"]
    me, sess = agents["researcher"], sessions["researcher"]
    entity = f"org:{_slug(sc['company'])}"
    below = {k: {m[x] for x in scenarios.descendants(items, k)} for k in ("R1", "R2", "R3")}
    mistakes = lambda: mem.records([me])[me]["mistakes"]  # noqa: E731

    # 1. supersede: HQ moved. Items built on it are flagged "changed"; no mistake counted.
    before = mistakes()
    rep = mem.supersede(me, sess, m["R2"], f"{sc['company']} moved its headquarters to Reykjavik")
    new_hq = rep["new"]
    check("supersede flags exactly the items built on the fact", set(rep["flagged"]) == below["R2"],
          f"{sc['company']}: got {sorted(rep['flagged'])}, want {sorted(below['R2'])}")
    flagged = {f["id"]: f for f in mem.flagged(me)}
    if below["R2"]:
        check("supersede: flagged() says 'changed' and names the new fact",
              all(any(c["fact"] == m["R2"] and c["now"] == new_hq for c in flagged[i]["changed"])
                  and not flagged[i]["wrong"] for i in below["R2"]), sc["company"])
    check("supersede does not count as a mistake", mistakes() == before, f"{before} -> {mistakes()}")
    texts = {f["text"] for f in mem.recall(me, entity)}
    check("recall shows the new value, not the old one",
          any("Reykjavik" in t for t in texts) and items["R2"]["text"] not in texts, sc["company"])

    # 2. retract: headcount was wrong. Items built on it are flagged "wrong"; one mistake counted.
    before = mistakes()
    rep = mem.retract(me, m["R1"], reason="headcount was wrong")
    check("retract flags exactly the items built on the fact", set(rep["flagged"]) == below["R1"],
          f"{sc['company']}: got {sorted(rep['flagged'])}, want {sorted(below['R1'])}")
    check("retract counts one mistake for the author", mistakes() == before + 1, f"{before} -> {mistakes()}")
    flagged = {f["id"]: f for f in mem.flagged(me)}
    if below["R1"]:
        check("retract: flagged() says 'wrong' for every item built on it",
              all(m["R1"] in flagged[i]["wrong"] for i in below["R1"]), sc["company"])
    both = below["R1"] & below["R2"]
    if both:
        check("an item built on a changed AND a wrong fact keeps both reasons",
              all(flagged[i]["wrong"] and flagged[i]["changed"] for i in both), sc["company"])

    # 3. rules: nothing new on a retracted or flagged item; no second retract; no supersede of it.
    check("cannot build a fact on a retracted fact",
          refused(mem.remember, me, sess, "built on a wrong fact", entity, derived_from=[m["R1"]]))
    if below["R1"]:
        check("cannot build a decision on a flagged item",
              refused(mem.decide, agents["planner"], sessions["planner"], "plan on flagged",
                      derived_from=[sorted(below["R1"])[0]]))
    check("cannot retract a fact twice", refused(mem.retract, me, m["R1"]))
    check("cannot supersede a retracted fact", refused(mem.supersede, me, sess, m["R1"], "new text"))
    check("another user cannot retract the fact", refused(mem.retract, outsider, m["R3"]))

    # 4. confirm: allowed when only a source changed, refused when any source was wrong.
    for i in below["R2"] - below["R1"]:
        check("confirm works on an item flagged only 'changed'", not refused(mem.confirm, me, i))
    for i in below["R1"]:
        check("confirm is refused on an item flagged 'wrong'", refused(mem.confirm, me, i))
    st = status(mem, set(m.values()))
    if below["R1"] or below["R2"]:
        check("after confirm: changed-only items active, wrong items still flagged",
              all(st[i] == "active" for i in below["R2"] - below["R1"])
              and all(st[i] == "flagged" for i in below["R1"]), sc["company"])

    # 5. recall leaves out retracted and flagged facts, and shows confirmed ones again.
    recalled = {f["id"] for f in mem.recall(me, entity)}
    facts_of = {m[k] for k, it in items.items() if it["kind"] == "fact"}
    hidden = ({m["R1"], m["R2"]} | below["R1"]) & facts_of
    check("recall leaves out retracted, superseded and flagged facts", not (recalled & hidden),
          f"{sc['company']}: {sorted(recalled & hidden)}")
    if (below["R2"] - below["R1"]) & facts_of:
        check("recall shows confirmed facts again", (below["R2"] - below["R1"]) & facts_of <= recalled, sc["company"])

    # 6. correct: retract funding and write the right value in one step.
    before = mistakes()
    rep = mem.correct(me, sess, m["R3"], f"{sc['company']} raised $1M in its last funding round", reason="typo")
    new = rep["new"]
    check("correct flags exactly the items built on the fact", set(rep["flagged"]) == below["R3"],
          f"{sc['company']}: got {sorted(rep['flagged'])}, want {sorted(below['R3'])}")
    row = mem.g.query("MATCH (n:Fact {id:$n})-[:CORRECTS]->(o:Fact {id:$o}), (n)-[:ABOUT]->(e:Entity {id:$e}) "
                      "RETURN n.status, o.status", {"n": new, "o": m["R3"], "e": entity}).result_set
    check("correct: new fact active, about the same entity, CORRECTS the old one (now retracted)",
          row == [["active", "retracted"]], f"{sc['company']}: {row}")
    check("correct counts one mistake for the author", mistakes() == before + 1, f"{before} -> {mistakes()}")


def cross_user_story(mem, i):
    """Alice shares a fact; Bob builds two private items on it and gets it in a handoff; Alice retracts."""
    alice = mem.register_agent("Researcher", owner=f"Alice{i}")
    bob_r = mem.register_agent("Researcher", owner=f"Bob{i}")
    bob_p = mem.register_agent("Planner", owner=f"Bob{i}")
    sa, sbr, sbp = mem.start_session(alice), mem.start_session(bob_r), mem.start_session(bob_p)
    fact = mem.remember(alice, sa, f"Client{i} renews its contract in March", f"org:client{i}", source="web")
    secret = mem.remember(alice, sa, f"Alice{i} thinks Client{i} is unhappy", f"org:client{i}")
    mem.publish(alice, fact)
    plan = mem.decide(bob_p, sbp, f"Prepare the Client{i} renewal deck in February", derived_from=[fact])
    draft = mem.remember(bob_r, sbr, f"Draft: Client{i} renewal deck outline", f"doc:deck{i}", derived_from=[plan])
    mem.handoff(alice, bob_r, [fact], note="for your renewal plan")

    check("other user cannot see a private fact", secret not in {f["id"] for f in mem.recall(bob_r, f"org:client{i}")})
    check("other user cannot build on a private fact",
          refused(mem.decide, bob_p, sbp, "plan on a secret", derived_from=[secret]))
    check("other user cannot retract or supersede a shared fact",
          refused(mem.retract, bob_r, fact) and refused(mem.supersede, bob_r, sbr, fact, "x"))
    rep = mem.retract(alice, fact, reason="renewal is in June")
    check("retract: other user's private items are counted, not shown",
          rep["flagged"] == [] and rep["hidden_flagged"] == 2, f"{rep}")
    # The report names only agents whose items the retractor may see (and handoff recipients), so
    # Alice does not learn who built private work on her fact; Bob's planner finds out by pulling.
    check("retract: the handoff recipient is listed to notify", bob_r in rep["agents"], f"{rep['agents']}")
    check("retract: the report does not name the author of a hidden item", bob_p not in rep["agents"],
          f"{rep['agents']}")
    check("retract: the hidden item's author finds it in its own flagged()",
          plan in {f["id"] for f in mem.flagged(bob_p)})
    bob_sees = {f["id"]: f for f in mem.flagged(bob_r)}
    check("other user's flagged() shows both items as 'wrong'",
          {plan, draft} <= set(bob_sees) and all(fact in bob_sees[x]["wrong"] for x in (plan, draft)), f"{bob_sees}")


def main():
    scs = scenarios.build()
    mem = Memory(GRAPH)
    mem.drop()
    mem.setup()
    agents, ids = load(mem, scs)
    sessions = {}
    for role, a in agents.items():
        sessions[role] = mem.start_session(a)
    outsider = mem.register_agent("Researcher", owner="Outsider")
    for sc, m in zip(scs, ids):
        company_story(mem, agents, sessions, sc, m, outsider)
    for i in range(len(scs)):
        cross_user_story(mem, i)

    passed = sum(p for p, _ in results.values())
    total = sum(t for _, t in results.values())
    lines = ["# Lifecycle checks", "",
             f"{len(results)} different rules, each checked on up to {len(scs)} templated companies (the shapes "
             f"differ, the wording does not matter here): supersede, retract, confirm, recall and correct, and a sharing "
             f"story between two users. Checks with nothing to check in a company are skipped, not counted. "
             f"**{passed} of {total} evaluations passed.** Tenant isolation is covered by tests/test_tenancy.py.", "",
             "| Check | Passed |", "|---|---|"]
    lines += [f"| {name} | {p}/{t} |" for name, (p, t) in results.items()]
    if failures:
        lines += ["", "## Failures (first 20)", ""] + [f"- {f}" for f in failures]
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
