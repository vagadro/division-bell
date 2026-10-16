import os
from dotenv import load_dotenv
from falkordb import FalkorDB

load_dotenv()

db = FalkorDB(
    host=os.environ["FALKORDB_HOST"],
    port=int(os.environ["FALKORDB_PORT"]),
    username=os.environ["FALKORDB_USERNAME"],
    password=os.environ["FALKORDB_PASSWORD"],
)

g = db.select_graph("hello")

g.query(
    """
    MERGE (a:Agent {name: $agent})
    MERGE (f:Fact {text: $fact})
    MERGE (a)-[:ASSERTED]->(f)
    """,
    {"agent": "Researcher", "fact": "FalkorDB is a graph database"},
)

result = g.query("MATCH (a:Agent)-[:ASSERTED]->(f:Fact) RETURN a.name, f.text")
for name, text in result.result_set:
    print(f"{name} asserted: {text}")

print("graphs:", db.list_graphs())
