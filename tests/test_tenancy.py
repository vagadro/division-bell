import pytest

from memory import MemoryGraphError, Tenants


@pytest.fixture
def tenants():
    t = Tenants(prefix="tenant_test_")
    for name in t.list():
        t.drop(name)
    yield t
    for name in t.list():
        t.drop(name)


def _fact(mem, text="Acme has 50 staff"):
    a = mem.register_agent("Researcher")
    s = mem.start_session(a)
    f = mem.remember(a, s, text, entity="org:acme")
    mem.publish(a, f)
    return a, s, f


def test_get_creates_and_lists(tenants):
    tenants.get("acme")
    tenants.get("globex")
    assert tenants.list() == ["acme", "globex"]
    assert tenants.get("acme") is tenants.get("acme")


@pytest.mark.parametrize("bad", ["", "Acme", "a b", "a;b", "../x", "tenant_x!", None])
def test_bad_names_rejected(tenants, bad):
    with pytest.raises(MemoryGraphError):
        tenants.get(bad)


def test_recall_does_not_cross(tenants):
    a, b = tenants.get("acme"), tenants.get("globex")
    ag, _, _ = _fact(a, "Acme has 50 staff")
    bg, _, _ = _fact(b, "Globex is private equity owned")
    assert [r["text"] for r in a.recall(ag, "org:acme")] == ["Acme has 50 staff"]
    assert [r["text"] for r in b.recall(bg, "org:acme")] == ["Globex is private equity owned"]


def test_retract_in_one_tenant_flags_nothing_in_other(tenants):
    a, b = tenants.get("acme"), tenants.get("globex")
    ag, sa, fa = _fact(a)
    bg, sb, fb = _fact(b)
    da = a.decide(ag, sa, "Hire", derived_from=[fa])
    db_ = b.decide(bg, sb, "Hire", derived_from=[fb])
    a.retract(ag, fa)
    assert [x["id"] for x in a.flagged(ag)] == [da]
    assert b.flagged(bg) == []
    assert b.trust(bg) == 1.0 and a.trust(ag) < 1.0
    assert db_  # still active in globex


def test_foreign_id_unknown_in_other_tenant(tenants):
    a, b = tenants.get("acme"), tenants.get("globex")
    _, _, fa = _fact(a)
    bg = b.register_agent("Researcher")  # same agent id as in acme
    bs = b.start_session(bg)
    with pytest.raises(MemoryGraphError):
        b.remember(bg, bs, "built on acme fact", entity="org:acme", derived_from=[fa])
    with pytest.raises(MemoryGraphError):
        b.retract(bg, fa)


def test_drop_one_tenant_keeps_other(tenants):
    a, b = tenants.get("acme"), tenants.get("globex")
    _fact(a)
    bg, _, _ = _fact(b)
    tenants.drop("acme")
    assert tenants.list() == ["globex"]
    assert len(b.recall(bg, "org:acme")) == 1
