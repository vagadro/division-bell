"""The frozen evaluation dataset (data/scenarios.json) and the checks every consumer relies on."""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data" / "scenarios.json"
KINDS = ("fact", "decision", "draft", "procedure")
ROLES = ("researcher", "planner", "writer")


def item_text(item):
    """The wording of an item as one string (a procedure is its name and steps)."""
    t = item["text"]
    if isinstance(t, dict):
        return f"{t['name']}: " + "; ".join(t["steps"])
    return t


def decoy(item):
    return item["id"].startswith("x")


def descendants(items, key):
    """{item id: depth} of every item built on `key`, directly (1) or through others (shortest path).
    `items` is a list of dataset items (dicts with id and sources)."""
    children = {}
    for it in items:
        for s in it["sources"]:
            children.setdefault(s, []).append(it["id"])
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


def questions(scenario):
    """Wrong-fact questions of a scenario: every fact or draft that something was built on.
    Yields (item id, {dependent id: depth})."""
    for it in scenario["items"]:
        if it["kind"] in ("fact", "draft"):
            truth = descendants(scenario["items"], it["id"])
            if truth:
                yield it["id"], truth


def load():
    return json.loads(DATA.read_text())["scenarios"]
