"""Blast-radius accuracy on the natural-language dataset (data/scenarios.json).

Question: a fact turns out to be wrong; which saved items were built on it? The answer key is the
dependency structure of the scenarios (known by construction, audited by reading the text). The memory
is loaded through the real Memory API into FalkorDB, and every method answers every question:

  graph                  Memory.blast_radius: one variable-length DERIVED_FROM traversal
  graph, 1 hop           only items built directly on the fact (ablation)
  flag every later item  the floor: everything in the same scenario written after the wrong fact
  vector, all memory     fastembed bge-small vectors in a FalkorDB vector index, whole memory
  vector, later items    the same, only items of the same scenario written after the fact
  keyword, later items   shared content words (Jaccard), same candidates
  LLM reads the notes    Haiku gets every note of the scenario (no links) and says which rely on
                         the wrong one; 3 independent runs, cached in data/reader_cache.json

Similarity methods flag items above a cutoff chosen leave-one-scenario-out: each scenario is scored
with the cutoff that is best on the other 15 scenarios, never on its own answers.
Uncertainty: bootstrap over scenarios (the unit that varies), 95% percentile interval.
Finally, a simulated sensitivity run in which agents forget or invent links; the rates are assumptions.

Run from backend/:  uv run python eval/accuracy.py   (graph eval_real; LLM cost about 15 cents)
"""
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import numpy as np
from fastembed import TextEmbedding

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm import MODEL, Spend  # noqa: E402
from memory import Memory, _content_words  # noqa: E402
import dataset  # noqa: E402

GRAPH = "eval_real"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DIM = 384
CUTS = np.round(np.arange(0.0, 1.0, 0.01), 2)
READER_RUNS = 3
READER_CACHE = Path(__file__).resolve().parent / "data" / "reader_cache.json"
READER_CAP = 1.00
BOOTSTRAPS = 2000
SENS_RATES = (0.10, 0.25)
SENS_SEEDS = 5
RESULTS = Path(__file__).resolve().parent / "results" / "accuracy.md"

NAMES = {
    "graph": "Graph (blast_radius)",
    "hop1": "Graph, 1 hop only",
    "all": "Flag every later item in the scenario",
    "vec_all": "Vector similarity, whole memory",
    "vec": "Vector similarity, later items of the scenario",
    "kw": "Keyword overlap, later items of the scenario",
    "reader": f"LLM reads all the notes (Haiku, mean of {READER_RUNS} runs)",
}
ORDER = ["graph", "hop1", "all", "vec_all", "vec", "kw", "reader"]
SHORT = {"graph": "Graph", "hop1": "1 hop", "all": "Flag all", "vec_all": "Vector (all)", "vec": "Vector (later)",
         "kw": "Keyword", "reader": "LLM reader"}


# Loading the memory

def load_graph(mem, scs, sources_of=None):
    """Write every scenario through the Memory API, in order. `sources_of(scenario, item)` may
    return other source ids than the design says (used by the sensitivity run).
    Returns (agent id by role, {scenario id: {item id: node id}})."""
    agents = {role: mem.register_agent(role.title(), owner="Eval") for role in dataset.ROLES}
    sessions = {role: mem.start_session(a) for role, a in agents.items()}
    ids = {}
    for sc in scs:
        m = ids.setdefault(sc["id"], {})
        for it in sc["items"]:
            srcs = [m[s] for s in (sources_of(sc, it) if sources_of else it["sources"])]
            a, s = agents[it["role"]], sessions[it["role"]]
            entity = f"{sc['id']}:{it['about']}" if it["about"] else None
            if it["kind"] in ("fact", "draft"):
                nid = mem.remember(a, s, dataset.item_text(it), entity, entity_name=it["about"],
                                   entity_type="topic", derived_from=srcs, source="agent")
            elif it["kind"] == "decision":
                nid = mem.decide(a, s, dataset.item_text(it), derived_from=srcs)
            else:
                nid = mem.learn_procedure(a, s, it["text"]["name"], it["text"]["steps"], entity,
                                          entity_name=it["about"], entity_type="topic", derived_from=srcs)
            m[it["id"]] = nid
    return agents, ids


def embed(mem, scs, ids):
    """Vectors on every item node, in a FalkorDB vector index. Returns {node id: vector}."""
    nodes = [(ids[sc["id"]][it["id"]], dataset.item_text(it)) for sc in scs for it in sc["items"]]
    vecs = list(TextEmbedding(EMBED_MODEL).embed([t for _, t in nodes]))
    rows = [{"id": n, "v": [float(x) for x in v]} for (n, _), v in zip(nodes, vecs)]
    # Index first: one created over existing nodes fills in the background, and a query sent
    # before it is done fails with "Invalid arguments for procedure".
    mem.g.query(f"CREATE VECTOR INDEX FOR (n:Item) ON (n.embedding) "
                f"OPTIONS {{dimension:{DIM}, similarityFunction:'cosine'}}")
    mem.g.query("UNWIND $rows AS r MATCH (n {id:r.id}) SET n:Item, n.embedding = vecf32(r.v)", {"rows": rows})
    return {r["id"]: r["v"] for r in rows}


def similarities(mem, vector, k):
    """{node id: cosine similarity} of the k nearest items (the index returns 1 - similarity)."""
    rows = mem.g.query("CALL db.idx.vector.queryNodes('Item', 'embedding', $k, vecf32($v)) "
                       "YIELD node, score RETURN node.id, score", {"v": vector, "k": k}).result_set
    return {nid: 1.0 - score for nid, score in rows}


# Metrics

def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def counts(pred, truth):
    tp = len(pred & truth)
    return np.array([tp, len(pred) - tp, len(truth) - tp])


def cutoff_counts(sims, truth):
    """Counts (tp, fp, fn) for every cutoff in CUTS: flag items whose score is >= the cutoff."""
    nodes = list(sims)
    s = np.array([sims[n] for n in nodes])
    t = np.array([n in truth for n in nodes])
    flagged = s[None, :] >= CUTS[:, None]
    tp = (flagged & t).sum(1)
    return np.stack([tp, flagged.sum(1) - tp, len(truth) - tp], axis=1)


def best_cutoff(per_question, skip=None):
    """Index into CUTS with the best pooled F1 over the questions of all scenarios but `skip`.
    Ties go to the higher cutoff, so 'flag everything' is never picked on a tie."""
    tot = sum(q["cc"] for q in per_question if q["sid"] != skip)
    f1s = [prf(*row)[2] for row in tot]
    return max(range(len(CUTS)), key=lambda i: (round(f1s[i], 9), i))


# The LLM reader

READER_SYSTEM = """You audit the shared notebook of a team of AI agents (a researcher, a planner and a writer). You are given every entry, oldest first. One entry has just been found to be wrong. Decide which other entries were written because of it, relying on it directly or through other entries, so that they may now be wrong too.

Do NOT include entries that merely cover the same topic, use similar numbers, or belong to the same situation without relying on the wrong entry. An entry can only rely on entries written before it.
Reply with JSON only: {"relies_on_it": [entry numbers]}"""


def reader_prompt(sc, root):
    lines = ["Notebook entries, oldest first:"]
    for n, it in enumerate(sc["items"], 1):
        lines.append(f"[{n}] ({it['role']}) " + " ".join(dataset.item_text(it).split()))
    k = [it["id"] for it in sc["items"]].index(root) + 1
    lines += ["", f"Entry [{k}] has just been found to be wrong. Which other entries rely on it?"]
    return "\n".join(lines)


def ask_reader(client, spend, lock, sc, root):
    with lock:
        spend.check()
    resp = client.messages.create(model=MODEL, max_tokens=8000, output_config={"effort": "medium"},
                                  system=READER_SYSTEM, messages=[{"role": "user", "content": reader_prompt(sc, root)}])
    with lock:
        spend.add(MODEL, resp.usage)
    text = "".join(b.text for b in resp.content if b.type == "text")
    a, b = text.find("{"), text.rfind("}")
    try:
        picked = json.loads(text[a:b + 1])["relies_on_it"]
    except (ValueError, KeyError, TypeError):
        return None
    order = [it["id"] for it in sc["items"]]
    return [order[n - 1] for n in picked if isinstance(n, int) and 1 <= n <= len(order) and order[n - 1] != root]


def reader_answers(scs, qs):
    """{(scenario id, root id, run): [item ids]} for every question, cached on disk."""
    cache = json.loads(READER_CACHE.read_text()) if READER_CACHE.exists() else {}
    by_id = {sc["id"]: sc for sc in scs}
    todo = [(q["sid"], q["root"], r) for q in qs for r in range(READER_RUNS)
            if f"{q['sid']}|{q['root']}|{r}" not in cache]
    if todo:
        client, spend, lock = anthropic.Anthropic(), Spend(cap=READER_CAP), threading.Lock()

        def run(job):
            sid, root, r = job
            for _ in range(3):
                got = ask_reader(client, spend, lock, by_id[sid], root)
                if got is not None:
                    return f"{sid}|{root}|{r}", got
            raise RuntimeError(f"reader gave no usable answer for {sid}/{root}")

        with ThreadPoolExecutor(max_workers=8) as pool:
            for key, got in pool.map(run, todo):
                cache[key] = got
        READER_CACHE.write_text(json.dumps(cache, indent=0))
        print(f"reader: {len(todo)} calls, ${spend.total:.3f}")
    return cache


# The questions and every method's answer

def build_questions(scs, ids, agents, mem, vectors):
    qs = []
    pos = {}
    for sc in scs:
        pos[sc["id"]] = {it["id"]: n for n, it in enumerate(sc["items"])}
    n_all = len(vectors)
    for sc in scs:
        sid, items = sc["id"], {it["id"]: it for it in sc["items"]}
        m = ids[sid]
        back = {nid: iid for iid, nid in m.items()}
        for root, truth in dataset.questions(sc):
            later = [i for i in items if pos[sid][i] > pos[sid][root]]
            assert set(truth) <= set(later), (sid, root)
            found = {back[r["id"]]: r["depth"] for r in mem.blast_radius(agents["researcher"], m[root])}
            rwords = _content_words(dataset.item_text(items[root]))
            sims_all = {n: s for n, s in similarities(mem, vectors[m[root]], n_all).items() if n != m[root]}
            qs.append({
                "sid": sid, "root": root, "m": m, "back": back,
                "truth": {m[i]: d for i, d in truth.items()},
                "later": [m[i] for i in later],
                "low": {m[i] for i in truth if len(rwords & _content_words(dataset.item_text(items[i]))) <= 1},
                "graph": {m[i] for i in found},
                "hop1": {m[i] for i, d in found.items() if d == 1},
                "all": {m[i] for i in later},
                "sims": {
                    "vec_all": sims_all,
                    "vec": {n: sims_all[n] for n in (m[i] for i in later)},
                    "kw": {m[i]: len(rwords & _content_words(dataset.item_text(items[i])))
                              / len(rwords | _content_words(dataset.item_text(items[i]))) for i in later},
                },
            })
    return qs


def add_predictions(qs, scs):
    """Fill q['pred'][run] for every method run. Similarity methods use leave-one-scenario-out cutoffs."""
    cutoffs = {}
    for key in ("vec_all", "vec", "kw"):
        for q in qs:
            q["cc"] = cutoff_counts(q["sims"][key], set(q["truth"]))
        cutoffs[key] = {sc["id"]: best_cutoff(qs, skip=sc["id"]) for sc in scs}
        cutoffs[key]["*"] = best_cutoff(qs)  # the oracle cutoff, for the footnote
        for q in qs:
            q.setdefault("pred", {})[key] = {n for n, s in q["sims"][key].items() if s >= CUTS[cutoffs[key][q["sid"]]]}
            q.setdefault("oracle", {})[key] = counts({n for n, s in q["sims"][key].items() if s >= CUTS[cutoffs[key]["*"]]},
                                                     set(q["truth"]))
    by_id = {sc["id"]: sc for sc in scs}
    answers = reader_answers(scs, qs)
    for q in qs:
        for key in ("graph", "hop1", "all"):
            q.setdefault("pred", {})[key] = q[key]
        for r in range(READER_RUNS):
            q["pred"][f"reader{r}"] = {q["m"][i] for i in answers[f"{q['sid']}|{q['root']}|{r}"]}
    return cutoffs


def runs_of(key):
    return [f"reader{r}" for r in range(READER_RUNS)] if key == "reader" else [key]


def scenario_counts(qs, scs, key):
    """{scenario id: [tp, fp, fn]} summed over its questions, for each run of a method (a list per run)."""
    out = []
    for run in runs_of(key):
        d = {sc["id"]: np.zeros(3, dtype=int) for sc in scs}
        for q in qs:
            d[q["sid"]] += counts(q["pred"][run], set(q["truth"]))
        out.append(d)
    return out


def pooled(per_run, sids):
    """Mean over runs of (precision, recall, F1) pooled over the scenarios in `sids` (repeats allowed)."""
    rows = [prf(*sum(d[s] for s in sids)) for d in per_run]
    return tuple(float(np.mean([r[i] for r in rows])) for i in range(3))


def bootstrap(per_run, sids, seed=1):
    rng = np.random.default_rng(seed)
    draws = np.array([pooled(per_run, [sids[i] for i in rng.integers(0, len(sids), len(sids))])
                      for _ in range(BOOTSTRAPS)])
    return np.percentile(draws, [2.5, 97.5], axis=0)  # rows: low/high; columns: P, R, F1


def recall_by(qs, key, groups):
    """Recall of a method on the dependents in each group (a function node -> label), mean over runs."""
    out = {}
    for run in runs_of(key):
        hit, tot = {}, {}
        for q in qs:
            for n, d in q["truth"].items():
                g = groups(q, n, d)
                tot[g] = tot.get(g, 0) + 1
                hit[g] = hit.get(g, 0) + (n in q["pred"][run])
        for g in tot:
            out.setdefault(g, []).append(hit[g] / tot[g])
    return {g: float(np.mean(v)) for g, v in out.items()}


# Sensitivity: agents that forget or invent links (the rates are assumptions)

def sensitivity(scs, qs):
    rows = []
    for mode in ("forget", "invent"):
        for rate in SENS_RATES:
            tot = np.zeros(3)
            for seed in range(SENS_SEEDS):
                rng = random.Random(seed * 1000 + int(rate * 100))

                def sources_of(sc, it, mode=mode, rate=rate, rng=rng):
                    srcs = list(it["sources"])
                    if mode == "forget":
                        return [s for s in srcs if rng.random() >= rate]
                    if rng.random() < rate:
                        earlier = [e for e in sc["items"][:[x["id"] for x in sc["items"]].index(it["id"])]
                                   if e["kind"] != "procedure" and e["id"] not in srcs]
                        same = [e for e in earlier if e["about"] and e["about"] == it["about"]]
                        pick = rng.choice(same or earlier) if earlier else None
                        if pick:
                            srcs.append(pick["id"])
                    return srcs

                mem = Memory(f"{GRAPH}_sens")
                mem.drop()
                mem.setup()
                agents, ids = load_graph(mem, scs, sources_of)
                for q in qs:
                    m = ids[q["sid"]]
                    back = {nid: iid for iid, nid in m.items()}
                    got = {back[r["id"]] for r in mem.blast_radius(agents["researcher"], m[q["root"]])}
                    truth = {q["back"][n] for n in q["truth"]}
                    tot += counts(got, truth)
                mem.drop()
            rows.append((mode, rate, *prf(*tot)))
            print(f"sensitivity {mode} {rate:.0%}: P {rows[-1][2]:.2f} R {rows[-1][3]:.2f}")
    return rows


# Report

def fmt(point, ci, i):
    if abs(ci[1][i] - ci[0][i]) < 1e-9:  # no spread over scenarios: a point, not an interval
        return f"{point[i]:.2f}"
    return f"{point[i]:.2f} ({ci[0][i]:.2f}-{ci[1][i]:.2f})"


def examples(qs, scs, key, run, n=3):
    """A few concrete failures of a method: false alarms and misses, one per scenario. For a
    similarity method the worst cases come first (most similar false alarm, least similar miss)."""
    by_id = {sc["id"]: {it["id"]: it for it in sc["items"]} for sc in scs}
    fa, miss = [], []
    for q in qs:
        items = by_id[q["sid"]]
        score = lambda node: q["sims"].get(key, {}).get(node, 0.0)  # noqa: E731
        for node in q["pred"][run] - set(q["truth"]):
            fa.append((score(node), q["sid"], items[q["root"]], items[q["back"][node]]))
        for node in set(q["truth"]) - q["pred"][run]:
            miss.append((score(node), q["sid"], items[q["root"]], items[q["back"][node]]))
    fa.sort(key=lambda x: -x[0])
    miss.sort(key=lambda x: x[0])
    clip = lambda it: " ".join(dataset.item_text(it).split())[:150]  # noqa: E731
    out = []
    for title, rows in (("flagged although it did not rely on the wrong fact", fa),
                        ("missed although it relied on the wrong fact", miss)):
        out.append(f"*{title}:*")
        seen = set()
        for _, sid, root, other in rows:
            if sid not in seen and len(seen) < n:
                seen.add(sid)
                out.append(f"- wrong: \"{clip(root)}\" -> \"{clip(other)}\"")
    return out


def main():
    scs = dataset.load()
    sids = [sc["id"] for sc in scs]
    mem = Memory(GRAPH)
    mem.drop()
    mem.setup()
    agents, ids = load_graph(mem, scs)
    vectors = embed(mem, scs, ids)
    qs = build_questions(scs, ids, agents, mem, vectors)
    cutoffs = add_predictions(qs, scs)
    print(f"{len(scs)} scenarios, {len(vectors)} items, {len(qs)} questions")

    per = {k: scenario_counts(qs, scs, k) for k in ORDER}
    point = {k: pooled(per[k], sids) for k in ORDER}
    ci = {k: bootstrap(per[k], sids) for k in ORDER}
    fp_per_tp = {}
    for k in ORDER:
        tot = [sum(d.values()) for d in per[k]]  # one [tp, fp, fn] per run
        fp_per_tp[k] = float(np.mean([t[1] for t in tot]) / max(1e-9, np.mean([t[0] for t in tot])))

    n_dep = sum(len(q["truth"]) for q in qs)
    depth_n = {d: sum(1 for q in qs for x in q["truth"].values() if x == d) for d in (1, 2, 3)}
    depth_n[3] += sum(1 for q in qs for x in q["truth"].values() if x > 3)
    n_low = sum(len(q["low"]) for q in qs)
    n_items = sum(len(sc["items"]) for sc in scs)
    n_decoys = sum(1 for sc in scs for it in sc["items"] if dataset.decoy(it))

    lines = [
        "# Blast-radius accuracy",
        "",
        f"**Data.** {len(scs)} scenarios from {len(set(sc['domain'] for sc in scs))} domains, {n_items} items "
        f"(facts, decisions, drafts, procedures; {n_decoys} are decoys on the same topic that nothing relied on). "
        f"The wording was written by a language model for hand-designed scenarios. "
        f"{len(qs)} questions (\"this fact is wrong, what relied on it?\") with {n_dep} items that truly relied on it: "
        f"{depth_n[1]} at 1 hop, {depth_n[2]} at 2, {depth_n[3]} at 3 or more. "
        f"{n_low} of them ({n_low / n_dep:.0%}) share at most one content word with the wrong fact.",
        "",
        "**Reading the table.** Precision = of the items flagged, the share that really relied on the wrong fact. "
        "Recall = of the items that relied on it, the share flagged. F1 combines them. In brackets: 95% interval "
        "from resampling the 16 scenarios. \"Innocent per real\" = false alarms for every item correctly flagged.",
        "",
        "| Method | Precision | Recall | F1 | Innocent per real |",
        "|---|---|---|---|---|",
    ]
    for k in ORDER:
        lines.append(f"| {NAMES[k]} | {fmt(point[k], ci[k], 0)} | {fmt(point[k], ci[k], 1)} | "
                     f"{fmt(point[k], ci[k], 2)} | {fp_per_tp[k]:.1f} |")

    oracle = {k: prf(*sum(q["oracle"][k] for q in qs)) for k in ("vec_all", "vec", "kw")}
    runs = [pooled([r], sids) for r in per["reader"]]
    spread = lambda i: f"{min(r[i] for r in runs):.2f}-{max(r[i] for r in runs):.2f}"  # noqa: E731
    lines += [
        "",
        f"The graph has no interval because its answers equal the key on every scenario. The {READER_RUNS} reader runs "
        f"alone scored F1 {spread(2)} (precision {spread(0)}, recall {spread(1)}). "
        "Similarity cutoffs are chosen leave-one-scenario-out (each scenario is scored with the cutoff that is best on "
        "the other 15). Even the best single cutoff, picked with the answers in view, only reaches F1 "
        f"{oracle['vec_all'][2]:.2f} (vector, whole memory), {oracle['vec'][2]:.2f} (vector, later items) and "
        f"{oracle['kw'][2]:.2f} (keyword, later items).",
        "",
        "## Per scenario (F1)",
        "",
        "| Scenario | Questions | " + " | ".join(SHORT[k] for k in ORDER) + " |",
        "|---|---|" + "---|" * len(ORDER),
    ]
    for sc in scs:
        n = sum(1 for q in qs if q["sid"] == sc["id"])
        cells = [f"{pooled(per[k], [sc['id']])[2]:.2f}" for k in ORDER]
        lines.append(f"| {sc['id']} ({sc['domain']}) | {n} | " + " | ".join(cells) + " |")

    depth = lambda q, n, d: min(d, 3)  # noqa: E731
    overlap = lambda q, n, d: "low" if n in q["low"] else "high"  # noqa: E731
    lines += ["", "## Recall by how far down the chain, and by shared wording", "",
              "| Method | 1 hop | 2 hops | 3+ hops | Shares at most 1 word with the wrong fact | Shares more |",
              "|---|---|---|---|---|---|"]
    for k in ORDER:
        d, o = recall_by(qs, k, depth), recall_by(qs, k, overlap)
        lines.append(f"| {NAMES[k]} | {d[1]:.2f} | {d[2]:.2f} | {d[3]:.2f} | {o['low']:.2f} | {o['high']:.2f} |")

    lines += ["", "## What the other methods get wrong", "", "**Vector similarity, later items of the scenario**", ""]
    lines += examples(qs, scs, "vec", "vec")
    lines += ["", "**LLM reads all the notes (run 1)**", ""]
    lines += examples(qs, scs, "reader", "reader0")

    sens = sensitivity(scs, qs)
    lines += [
        "", "## If agents record the links badly (simulation)", "",
        "The graph can only follow links that were recorded. These rates are assumptions, not measurements of "
        f"real agents. Each row averages {SENS_SEEDS} random draws over the same {len(qs)} questions. "
        "*Forget* drops that share of the recorded links. *Invent* adds, with that probability per item, one link to an "
        "earlier item the item did not rely on (preferring the same topic, so decoys get cited).",
        "",
        "| Links | Rate | Precision | Recall | F1 |", "|---|---|---|---|---|",
    ]
    for mode, rate, p, r, f1 in sens:
        lines.append(f"| {mode} | {rate:.0%} | {p:.2f} | {r:.2f} | {f1:.2f} |")
    lines += [
        "",
        "## Limits",
        "",
        "- The wording is model-written for 16 hand-designed situations, not real agent traffic, and the LLM reader is the same "
        "model family that wrote it. The dependency key is known by construction, and a different author could disagree on a few borderline items.",
        "- The graph answers from recorded links, so its score on perfect links is a correctness check of the traversal. "
        "A plain list of parent ids walked in code would give the same answer; the evidence here is that similarity and "
        "reading cannot recover dependency, and that the traversal is exact.",
        "- The key counts every item downstream of a wrong fact as relying on it, however many steps away, which is what the "
        "traversal returns. A person might call a distant item (a procedure three steps down) unaffected, so read the graph's "
        "flags as 'review these', a safe superset, not 'these are certainly wrong'.",
        "- The key was audited once after the first run, by reading where the reader consistently disagreed with it. One flaw "
        "was found and fixed, which removed one of the reader's apparent false alarms, so the fix went against the graph. "
        "Two borderline cases remain, both between sibling drafts of the same story.",
        "- 16 scenarios is a small sample (see the intervals). The LLM reader is one model (Haiku 5.5) at medium effort; a "
        "bigger model would probably score higher. Its cost grows with the size of the memory it must read (each question "
        "sent the whole notebook here, about 15 notes); the graph query does not. That scaling was not measured here.",
        "",
    ]
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
