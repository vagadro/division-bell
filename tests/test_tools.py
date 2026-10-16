"""Tool list tests: the tools run as one fixed agent, errors come back as data, schemas are valid."""
import asyncio

import pytest

import tools
from llm import MockChecker, Verdict
from mcp_server import build_server
from tools import TOOL_NAMES, AgentTools


@pytest.fixture
def sam(mem):
    return AgentTools.open(mem, "Researcher", owner="Sam", checker=MockChecker())


@pytest.fixture
def priya(mem):
    return AgentTools.open(mem, "Planner", owner="Priya", checker=MockChecker())


def test_a_fact_travels_through_the_tools(sam, priya):
    fid = sam.call("remember", {"text": "Acme has 500 employees", "entity": "org:acme"})["id"]
    assert priya.call("recall", {"entity": "org:acme"})["facts"] == []          # still private
    sam.call("publish", {"item_id": fid})
    [fact] = priya.call("recall", {"entity": "org:acme"})["facts"]
    assert fact["id"] == fid and fact["author"] == sam.agent and fact["needs_verification"] is False


def test_the_caller_is_fixed_by_the_config_not_the_arguments(sam, priya):
    fid = sam.call("remember", {"text": "x", "entity": "org:acme"})["id"]
    sam.call("publish", {"item_id": fid})
    out = priya.call("retract", {"fact_id": fid, "agent": sam.agent})   # an "agent" argument is not accepted
    assert "error" in out


def test_supersede_with_web_sources_keeps_the_new_value_verified(sam, priya):
    old = sam.call("remember", {"text": "Acme has 500 employees", "entity": "org:acme", "source": "agent"})["id"]
    sam.call("publish", {"item_id": old})
    sam.web_urls = {"https://acme.example/about"}  # what this turn's web search returned
    bad = sam.call("supersede", {"old_id": old, "text": "Acme has 600 employees",
                                 "sources": ["https://made.up/page"]})
    assert "error" in bad and [f["id"] for f in priya.call("recall", {"entity": "org:acme"})["facts"]] == [old]
    out = sam.call("supersede", {"old_id": old, "text": "Acme has 600 employees",
                                 "sources": ["https://acme.example/about", "https://made.up/page"]})
    assert out["sources"] == ["https://acme.example/about"]
    [fact] = priya.call("recall", {"entity": "org:acme"})["facts"]
    assert fact["id"] == out["new"] and fact["standing"] == "verified"


def test_lineage_through_the_tools(sam):
    f1 = sam.call("remember", {"text": "Acme has 500 employees", "entity": "org:acme"})["id"]
    sam.call("publish", {"item_id": f1})
    d1 = sam.call("decide", {"text": "Target Acme", "derived_from": [f1]})["id"]
    assert [a["id"] for a in sam.call("blast_radius", {"item_id": f1})["affected"]] == [d1]
    sam.call("retract", {"fact_id": f1, "reason": "wrong"})
    [item] = sam.call("flagged")["items"]
    assert item["id"] == d1 and item["wrong"] == [f1]
    assert sam.call("explain", {"item_id": d1})["chain"][0]["id"] == d1


def test_errors_come_back_as_data(sam, priya):
    fid = sam.call("remember", {"text": "x", "entity": "org:acme"})["id"]
    sam.call("publish", {"item_id": fid})
    assert "only the author" in priya.call("retract", {"fact_id": fid})["error"]
    assert "unknown tool" in sam.call("drop")["error"]
    assert "bad arguments" in sam.call("remember", {"text": "x"})["error"]            # entity missing
    assert "bad arguments" in sam.call("remember", {"text": "x", "entity": "e", "confidence": 3})["error"]


def test_handoff_and_inbox(mem, sam):
    planner = AgentTools.open(mem, "Planner", owner="Sam")
    fid = sam.call("remember", {"text": "x", "entity": "org:acme"})["id"]
    sam.call("handoff", {"recipient": planner.agent, "item_ids": [fid], "note": "plan it"})
    assert planner.call("inbox")["handoffs"]


def test_procedures(sam):
    pid = sam.call("learn_procedure", {"name": "Onboard", "steps": ["call", "email"], "entity": "task:onboard"})["id"]
    assert [p["id"] for p in sam.call("recall_procedures", {"entity": "task:onboard"})["procedures"]] == [pid]


def test_verify_fact_records_a_supported_check(mem):
    checker = MockChecker({"Acme has 500 employees": Verdict("supported", ["https://e.com"], "q")})
    t = AgentTools.open(mem, "Researcher", checker=checker)
    fid = t.call("remember", {"text": "Acme has 500 employees", "entity": "org:acme"})["id"]
    out = t.call("verify_fact", {"fact_id": fid, "method": "web"})
    assert out["recorded"] and out["verdict"]["result"] == "supported"
    assert t.call("recall", {"entity": "org:acme"})["facts"][0]["verified"] is True


def test_verify_fact_documents_stay_inside_the_documents_folder(mem, tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "hr.txt").write_text("Acme has 500 employees.")
    (tmp_path / "secret.txt").write_text("Acme has 500 employees.")
    checker = MockChecker({"Acme has 500 employees": Verdict("supported", ["hr.txt para 1"], "q")})
    t = AgentTools.open(mem, "Researcher", checker=checker, docs_dir=tmp_path / "docs")
    fid = t.call("remember", {"text": "Acme has 500 employees", "entity": "org:acme"})["id"]
    assert t.call("verify_fact", {"fact_id": fid, "method": "doc", "document": "hr.txt"})["recorded"]
    for bad in ("../secret.txt", str(tmp_path / "secret.txt"), None):
        assert "error" in t.call("verify_fact", {"fact_id": fid, "method": "doc", "document": bad})
    no_folder = AgentTools.open(mem, "Writer", checker=checker)
    assert "no documents folder" in no_folder.call("verify_fact", {"fact_id": fid, "method": "doc", "document": "hr.txt"})["error"]


def test_anthropic_tool_list(sam):
    listed = sam.anthropic_tools()
    assert [t["name"] for t in listed] == list(TOOL_NAMES)
    for t in listed:
        assert t["description"] and t["input_schema"]["type"] == "object"
        assert "title" not in t["input_schema"] and "self" not in t["input_schema"]["properties"]
        assert "agent" not in t["input_schema"]["properties"] and "session" not in t["input_schema"]["properties"]
    assert listed[-1]["cache_control"] == {"type": "ephemeral"}
    assert sam.anthropic_tools()[0]["input_schema"]["required"] == ["text", "entity"]


def test_mcp_server_lists_the_same_tools_and_runs_one(sam):
    server = build_server(sam)

    async def go():
        listed = await server.list_tools()
        result = await server.call_tool("remember", {"text": "via mcp", "entity": "org:acme"})
        return listed, result

    listed, result = asyncio.run(go())
    assert sorted(t.name for t in listed) == sorted(TOOL_NAMES)
    assert "fact:" in str(result)
    assert sam.call("recall", {"entity": "org:acme"})["facts"][0]["text"] == "via mcp"


def test_mcp_errors_carry_the_reason(sam):
    result = asyncio.run(build_server(sam).call_tool("retract", {"fact_id": "fact:nope"}))
    assert "not found, or not visible" in str(result)


def test_correct_and_supersede_return_the_new_id_and_what_was_flagged(sam):
    f1 = sam.call("remember", {"text": "Acme has 500 employees", "entity": "org:acme"})["id"]
    sam.call("publish", {"item_id": f1})
    d1 = sam.call("decide", {"text": "Target Acme", "derived_from": [f1]})["id"]
    fixed = sam.call("correct", {"old_id": f1, "text": "Acme has 50 employees", "reason": "HR page"})
    assert fixed["retracted"] == f1 and fixed["new"].startswith("fact:") and d1 in fixed["flagged"]
    changed = sam.call("supersede", {"old_id": fixed["new"], "text": "Acme has 60 employees"})
    assert changed["superseded"] == fixed["new"] and changed["new"].startswith("fact:")
