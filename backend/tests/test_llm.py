"""Checker tests. HaikuChecker runs against a fake client, so nothing here calls the API."""
import json
from types import SimpleNamespace as NS

import pytest

from documents import Document, Passage
from llm import HaikuChecker, LLMBudgetError, MockChecker, Verdict, check_and_record

URL = "https://example.com/acme"


class FakeClient:
    """Stands in for anthropic.Anthropic(): returns one canned response per call."""

    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def answer(result="supported", sources=(URL,), snippet="Acme has 500 employees"):
    return json.dumps({"result": result, "sources": list(sources), "snippet": snippet})


def web_response(text, searches=1, urls=(URL,)):
    blocks = [NS(type="web_search_tool_result", content=[NS(url=u) for u in urls]),
              NS(type="text", text=text)]
    usage = NS(server_tool_use=NS(web_search_requests=searches))
    return NS(content=blocks if searches else blocks[1:], usage=usage)


def text_response(text):
    return NS(content=[NS(type="text", text=text)], usage=NS(server_tool_use=None))


def doc():
    return Document("hr.pdf", [Passage("hr.pdf p.1", "As of 2025, Acme has  500\nemployees worldwide."),
                               Passage("hr.pdf p.2", "Unrelated text about parking.")])


# Web guards

def test_web_supported_with_real_sources():
    c = HaikuChecker(FakeClient(web_response(answer())))
    v = c.check("Acme has 500 employees", "web")
    assert (v.result, v.sources, v.snippet) == ("supported", [URL], "Acme has 500 employees")


def test_web_answer_without_a_real_search_is_unclear():
    c = HaikuChecker(FakeClient(web_response(answer(), searches=0)))
    assert c.check("Acme has 500 employees", "web").result == "unclear"


def test_web_invented_sources_are_unclear():
    c = HaikuChecker(FakeClient(web_response(answer(sources=["https://made-up.example"]))))
    v = c.check("Acme has 500 employees", "web")
    assert v.result == "unclear" and v.sources == []


def test_web_keeps_only_sources_that_were_in_the_results():
    c = HaikuChecker(FakeClient(web_response(answer(sources=[URL, "https://made-up.example"]))))
    assert c.check("x", "web").sources == [URL]


def test_web_cap_stops_further_checks():
    client = FakeClient(web_response(answer()), web_response(answer()))
    c = HaikuChecker(client, max_web_checks=1)
    c.check("x", "web")
    with pytest.raises(LLMBudgetError):
        c.check("y", "web")
    assert len(client.calls) == 1


# Doc guards

def test_doc_supported_when_snippet_is_in_the_document():
    c = HaikuChecker(FakeClient(text_response(answer(sources=["hr.pdf p.1"], snippet="acme has 500 employees"))))
    v = c.check("Acme has 500 employees", "doc", doc())
    assert v.result == "supported" and v.sources == ["hr.pdf p.1"]    # case and spacing ignored


def test_doc_fabricated_snippet_is_unclear():
    c = HaikuChecker(FakeClient(text_response(answer(snippet="Acme has 900 employees"))))
    assert c.check("Acme has 900 employees", "doc", doc()).result == "unclear"


def test_doc_source_comes_from_the_document_not_the_model():
    c = HaikuChecker(FakeClient(text_response(answer(sources=["wrong label"], snippet="Acme has 500 employees"))))
    assert c.check("Acme has 500 employees", "doc", doc()).sources == ["hr.pdf p.1"]


def test_doc_text_goes_to_the_model_as_data_with_a_cacheable_system_prompt():
    client = FakeClient(text_response(answer(result="unclear")))
    HaikuChecker(client).check("Acme has 500 employees", "doc", doc())
    call = client.calls[0]
    assert "<document>" in call["messages"][0]["content"] and "[hr.pdf p.1]" in call["messages"][0]["content"]
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"} and "tools" not in call


def test_doc_with_no_related_passage_makes_no_api_call(monkeypatch):
    import llm
    monkeypatch.setattr(llm, "select_passages", lambda d, claim: [])
    client = FakeClient()
    assert HaikuChecker(client).check("x", "doc", doc()).result == "unclear" and client.calls == []


# Parsing

@pytest.mark.parametrize("text", ["not json", "", "{}", '{"result": "true"}', '{"result": 5}', "[1]"])
def test_malformed_answers_are_unclear(text):
    c = HaikuChecker(FakeClient(web_response(text)))
    assert c.check("x", "web").result == "unclear"


def test_json_wrapped_in_prose_is_still_read():
    c = HaikuChecker(FakeClient(web_response("Here you go: " + answer() + " Hope that helps.")))
    assert c.check("x", "web").result == "supported"


def test_bad_arguments():
    with pytest.raises(ValueError):
        HaikuChecker(FakeClient()).check("x", "peer")
    with pytest.raises(ValueError):
        HaikuChecker(FakeClient()).check("x", "doc")


# Glue: check_and_record

def fact_and_checker(team, verdict):
    mem, r, sr, p, sp = team
    f = mem.remember(r, sr, "Acme has 500 employees", "org:acme")
    mem.publish(r, f)
    return mem, r, p, f, MockChecker({"Acme has 500 employees": verdict})


def test_check_and_record_stores_a_supported_verdict(team):
    mem, r, p, f, checker = fact_and_checker(team, Verdict("supported", [URL], "Acme has 500 employees"))
    before = mem.accuracy(r)
    out = check_and_record(mem, p, f, "web", checker)
    assert out["recorded"] and out["result"] == "supported"
    assert mem.accuracy(r) == before                                # evidence is about the fact, not the agent
    assert mem.recall(p, "org:acme")[0]["verified"] is True
    assert checker.calls[0][:2] == ("Acme has 500 employees", "web")


def test_check_and_record_does_not_store_an_unclear_verdict_without_evidence(team):
    mem, r, p, f, checker = fact_and_checker(team, Verdict("unclear", note="nothing found"))
    out = check_and_record(mem, p, f, "web", checker)
    assert out["recorded"] is False and out["verdict"].note == "nothing found"
    assert mem.recall(p, "org:acme")[0]["verified"] is False


def test_check_and_record_refuses_invisible_facts(team):
    from memory import MemoryGraphError
    mem, r, sr, p, sp = team
    private = mem.remember(r, sr, "secret", "org:acme")
    with pytest.raises(MemoryGraphError):
        check_and_record(mem, p, private, "web", MockChecker())
