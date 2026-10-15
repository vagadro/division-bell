import pytest

from memory import Memory


@pytest.fixture
def mem():
    m = Memory("memory_test")
    m.drop()
    m.setup()
    yield m
    m.drop()


@pytest.fixture
def team(mem):
    """Researcher and Planner, each with a session."""
    r = mem.register_agent("Researcher")
    p = mem.register_agent("Planner")
    return mem, r, mem.start_session(r), p, mem.start_session(p)
