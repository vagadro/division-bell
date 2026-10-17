"""Scale and concurrency evaluation for blast radius and retract.

  size        the templated scenarios copied x1, x10, x100; blast_radius and retract timed
  shapes      a long chain, a wide fan-out, and a diamond lattice (paths double per layer)
  concurrency retracts and writes in 20 threads at once, then invariants: nothing active is
              built on a retracted fact, every retract counted once, safe writes all succeeded

Results go to eval/results/scale.md. Times are wall-clock from Python to local Docker and back.

Run from backend/:  uv run python eval/scale.py   (cost $0, no LLM; takes a few minutes)
"""
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from redis.exceptions import ResponseError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from memory import Memory, MemoryGraphError, _now  # noqa: E402
import queries  # noqa: E402
import synthetic as scenarios  # noqa: E402
from synthetic import load  # noqa: E402

GRAPH = "eval_scale"
COPIES = (1, 10, 100)
SAMPLES = 100
TIMEOUT_MS = 20000
THREADS = 20
ROUNDS = 5
RESULTS = Path(__file__).resolve().parent / "results" / "scale.md"


def copies(n):
    """n copies of the 30 companies, each copy with its own random shapes and company names."""
    out = []
    for j in range(n):
        for sc in scenarios.build(seed=scenarios.SEED + j):
            out.append({"company": f"{sc['company']} {j}", "items": sc["items"]})
    return out


def fresh(name=GRAPH):
    mem = Memory(name)
    mem.drop()
    mem.setup()
    return mem


def timed(fn, *args):
    start = time.perf_counter()
    out = fn(*args)
    return (time.perf_counter() - start) * 1000, out


def ms_summary(xs):
    xs = sorted(xs)
    return f"{statistics.median(xs):.1f} / {xs[int(len(xs) * 0.95) - 1]:.1f} / {xs[-1]:.1f}"


def size_rows():
    rows = []
    for n in COPIES:
        scs = copies(n)
        mem = fresh()
        start = time.perf_counter()
        agents, ids = load(mem, scs)
        load_s = time.perf_counter() - start
        items = mem.g.query("MATCH (n) WHERE n:Fact OR n:Decision OR n:Procedure RETURN count(n)").result_set[0][0]
        me = agents["researcher"]
        picks = random.Random(n).sample(range(len(scs)), min(SAMPLES, len(scs)))
        br, rt, ok = [], [], 0
        for i in picks:
            truth = {ids[i][k] for k in scenarios.descendants(scs[i]["items"], "R1")}
            t, found = timed(mem.blast_radius, me, ids[i]["R1"])
            br.append(t)
            t, rep = timed(mem.retract, me, ids[i]["R1"])
            rt.append(t)
            ok += {r["id"] for r in found} == truth == set(rep["flagged"])
        rows.append((items, len(picks), load_s, br, rt, ok))
        print(f"size x{n}: {items} items loaded in {load_s:.0f}s")
    return rows


def chain(mem, me, sess, length):
    ids = [mem.remember(me, sess, "link 0", "topic:chain")]
    for k in range(1, length + 1):
        ids.append(mem.remember(me, sess, f"link {k}", "topic:chain", derived_from=[ids[-1]]))
    return ids


def fan(mem, me, sess, width):
    root = mem.remember(me, sess, "fan root", "topic:fan")
    for k in range(width):
        mem.remember(me, sess, f"leaf {k}", "topic:fan", derived_from=[root])
    return root


def lattice(mem, me, sess, layers):
    """Two items per layer, each built on both items of the layer below: 2^layers paths, 2*layers items."""
    root = mem.remember(me, sess, "lattice root", "topic:lattice")
    below = [root]
    for k in range(layers):
        below = [mem.remember(me, sess, f"layer {k} item {j}", "topic:lattice", derived_from=below) for j in (0, 1)]
    return root


def query_ms(mem, q, params):
    """Run a raw query with a timeout. Returns (ms, rows) or (None, error text)."""
    start = time.perf_counter()
    try:
        rows = mem.g.query(q, params, timeout=TIMEOUT_MS).result_set
    except ResponseError as e:
        return None, str(e)
    return (time.perf_counter() - start) * 1000, rows


def shape_rows():
    rows = []
    shapes = [("chain", chain, n, n) for n in (10, 50, 200)]
    shapes += [("fan-out", fan, 1000, 1000)]
    shapes += [("diamond lattice", lattice, n, 2 * n) for n in (5, 10, 15, 20)]
    for name, build, size, expected in shapes:
        mem = fresh(f"{GRAPH}_shape")
        me = mem.register_agent("Researcher", owner="Eval")
        sess = mem.start_session(me)
        out = build(mem, me, sess, size)
        root = out[0] if isinstance(out, list) else out
        t_br, br = query_ms(mem, queries.BLAST_RADIUS, {"id": root, "me": me})
        t_rt, rt = query_ms(mem, queries.RETRACT, {"id": root, "by": me, "reason": "", "now": _now()})
        found = len(br) if t_br is not None else None
        flagged = len(rt[0][1]) if t_rt is not None else None
        rows.append((name, size, expected, t_br, found, t_rt, flagged, br if t_br is None else rt if t_rt is None else ""))
        print(f"{name} {size}: blast {t_br} ms ({found}), retract {t_rt} ms ({flagged})")
        mem.drop()
    return rows


def concurrency_round(seed):
    scs = scenarios.build()
    mem = fresh(f"{GRAPH}_conc")
    agents, ids = load(mem, scs)
    me, planner = agents["researcher"], agents["planner"]
    psess = mem.start_session(planner)
    tasks = [("retract", i) for i in range(len(scs))]
    tasks += [("risky", i) for i in range(len(scs))] + [("safe", i) for i in range(len(scs))]
    random.Random(seed).shuffle(tasks)

    def run(task):
        kind, i = task
        m = ids[i]
        try:
            if kind == "retract":
                mem.retract(me, m["R1"])
            elif kind == "risky":  # built on the headcount chain that is being retracted right now
                src = m.get("A4") or m.get("A1") or m["R1"]
                mem.decide(planner, psess, "plan on headcount", derived_from=[src])
            else:  # built on the funding fact, which nobody retracts
                mem.decide(planner, psess, "plan on funding", derived_from=[m["R3"]])
            return kind, True
        except MemoryGraphError:
            return kind, False

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        out = list(pool.map(run, tasks))
    q = mem.g.query
    leaked = q("MATCH (x)-[:DERIVED_FROM*1..]->(f:Fact {status:'retracted'}) WHERE x.status = 'active' "
               "RETURN count(DISTINCT x)").result_set[0][0]
    retracted = q("MATCH (f:Fact {status:'retracted'}) RETURN count(f)").result_set[0][0]
    mistakes = mem.records([me])[me]["mistakes"]
    safe_ok = sum(ok for kind, ok in out if kind == "safe")
    risky_ok = sum(ok for kind, ok in out if kind == "risky")
    retract_ok = sum(ok for kind, ok in out if kind == "retract")
    mem.drop()
    return len(scs), leaked, retracted, mistakes, retract_ok, safe_ok, risky_ok


def main():
    sizes = size_rows()
    shapes = shape_rows()
    rounds = [concurrency_round(seed) for seed in range(ROUNDS)]
    Memory(GRAPH).drop()

    lines = ["# Scale and concurrency", "",
             "Times in ms, wall-clock from Python to FalkorDB in local Docker and back. The graphs are templated "
             "(only their size and shape matter here). Blast radius gets slower as the whole graph grows, "
             "even for a small affected set: the query plan starts from every agent's items instead of the one "
             "wrong fact, so cost follows graph size. Fine at demo scale; not a constant-time lookup.", "",
             "## Graph size", "",
             "Blast radius and retract of a company's headcount fact, on random companies.", "",
             "| Items in graph | Queries | Load time | Blast radius median / p95 / max | Retract median / p95 / max | Exact answers |",
             "|---|---|---|---|---|---|"]
    for items, n, load_s, br, rt, ok in sizes:
        lines.append(f"| {items:,} | {n} | {load_s:.0f} s | {ms_summary(br)} | {ms_summary(rt)} | {ok}/{n} |")
    lines += ["", "## Shapes", "",
              "| Shape | Size | Items below the root | Blast radius ms | Found | Retract ms | Flagged |",
              "|---|---|---|---|---|---|---|"]
    for name, size, expected, t_br, found, t_rt, flagged, err in shapes:
        fmt = lambda t: f"{t:.1f}" if t is not None else f"timeout ({TIMEOUT_MS // 1000} s)"  # noqa: E731
        lines.append(f"| {name} | {size} | {expected} | {fmt(t_br)} | {found if found is not None else '-'} | "
                     f"{fmt(t_rt)} | {flagged if flagged is not None else '-'} |")
    lines += ["", "Size: chain = links, fan-out = items on the root, lattice = layers (2 items each, "
              "2^layers paths).", "", "## Concurrency", "",
              f"{THREADS} threads, each round: one retract per company, plus a decision built on that company's "
              "headcount chain (racing the retract) and one built on its funding fact (never retracted).", "",
              "| Round | Companies | Retracts ok | Mistakes counted | Safe writes ok | Racing writes accepted | "
              "Active items built on a retracted fact |", "|---|---|---|---|---|---|---|"]
    for k, (n, leaked, retracted, mistakes, retract_ok, safe_ok, risky_ok) in enumerate(rounds, 1):
        lines.append(f"| {k} | {n} | {retract_ok}/{n} | {mistakes} | {safe_ok}/{n} | {risky_ok}/{n} | {leaked} |")
    lines += ["", "A racing write is accepted if it landed before the retract (then the retract flags it) and "
              "refused after. The last column must be 0.", ""]
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
