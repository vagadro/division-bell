from concurrent.futures import ThreadPoolExecutor

import pytest

from memory import MemoryGraphError, TRUST_PENALTY

N = 20


def run_parallel(fn, args_list):
    """Run fn(*args) in N threads at once. Returns (results, errors); errors are MemoryGraphErrors."""
    def call(args):
        try:
            return fn(*args), None
        except MemoryGraphError as e:
            return None, e
    with ThreadPoolExecutor(max_workers=N) as pool:
        out = list(pool.map(call, args_list))
    return [r for r, e in out if e is None], [e for r, e in out if e is not None]


def test_parallel_register_agent_makes_one_node(mem):
    run_parallel(mem.register_agent, [("Researcher",)] * N)
    assert mem.g.query("MATCH (a:Agent) RETURN count(a)").result_set[0][0] == 1


def test_parallel_remember_loses_nothing(mem):
    agents = [mem.register_agent(f"W{i}") for i in range(N)]
    sessions = [mem.start_session(a) for a in agents]
    ids, errors = run_parallel(
        lambda a, s, i: mem.remember(a, s, f"fact {i}", entity="org:acme"),
        [(a, s, i) for i, (a, s) in enumerate(zip(agents, sessions))])
    assert not errors and len(set(ids)) == N
    assert mem.g.query("MATCH (f:Fact) RETURN count(f)").result_set[0][0] == N
    assert mem.g.query("MATCH (e:Entity {id:'org:acme'}) RETURN count(e)").result_set[0][0] == 1


def test_parallel_retract_same_fact_lowers_trust_once(team):
    mem, r, rs, p, ps = team
    f = mem.remember(r, rs, "wrong", entity="org:acme")
    mem.publish(r, f)
    d = mem.decide(p, ps, "built on it", derived_from=[f])
    done, errors = run_parallel(mem.retract, [(r, f)] * N)
    assert len(done) == 1 and len(errors) == N - 1
    assert mem.trust(r) == pytest.approx(TRUST_PENALTY)
    assert [x["id"] for x in mem.flagged(p)] == [d]


def test_parallel_derive_while_retract_leaves_nothing_active_on_bad_source(team):
    """No fact or decision may end up active while built on a retracted fact."""
    mem, r, rs, p, ps = team
    f = mem.remember(r, rs, "wrong", entity="org:acme")
    mem.publish(r, f)
    jobs = [("retract", r, f)] + [("decide", p, ps, f)] * N

    def job(kind, *a):
        if kind == "retract":
            return mem.retract(*a)
        return mem.decide(a[0], a[1], "x", derived_from=[a[2]])
    run_parallel(job, jobs)
    bad = mem.g.query(
        "MATCH (x)-[:DERIVED_FROM]->(f:Fact {status:'retracted'}) "
        "WHERE x.status = 'active' RETURN x.id").result_set
    assert bad == []


def test_parallel_publish_and_retract_end_consistent(team):
    mem, r, rs, p, ps = team
    f = mem.remember(r, rs, "maybe wrong", entity="org:acme")
    run_parallel(lambda kind: mem.publish(r, f) if kind == "p" else mem.retract(r, f),
                 [("p",), ("r",)] * (N // 2))
    status = mem.g.query("MATCH (f:Fact {id:$id}) RETURN f.status", {"id": f}).result_set[0][0]
    assert status == "retracted"


def test_reads_during_writes_do_not_fail(team):
    mem, r, rs, p, ps = team
    f = mem.remember(r, rs, "seed", entity="org:acme")
    mem.publish(r, f)

    def job(i):
        if i % 2:
            return mem.remember(r, rs, f"fact {i}", entity="org:acme")
        return mem.recall(p, "org:acme")
    results, errors = run_parallel(job, [(i,) for i in range(N * 2)])
    assert not errors and len(results) == N * 2


def test_source_retracted_between_check_and_write(team, monkeypatch):
    """Force the worst timing: the source is retracted after decide() checked it, before it wrote."""
    mem, r, rs, p, ps = team
    f = mem.remember(r, rs, "wrong", entity="org:acme")
    mem.publish(r, f)
    real = mem._require_sources

    def check_then_retract(*a, **k):
        real(*a, **k)          # the check passes: f is still active
        mem.retract(r, f)      # ...then someone retracts it before the write
    monkeypatch.setattr(mem, "_require_sources", check_then_retract)
    try:
        mem.decide(p, ps, "built on a source that just died", derived_from=[f])
    except MemoryGraphError:
        pass  # refusing is fine
    monkeypatch.undo()
    bad = mem.g.query(
        "MATCH (x:Decision)-[:DERIVED_FROM]->(f:Fact {status:'retracted'}) "
        "WHERE x.status = 'active' RETURN x.id").result_set
    assert bad == []
