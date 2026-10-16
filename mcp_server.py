"""Local MCP server (stdio) that gives Claude Code / Claude Desktop the shared-memory tools.

Who it acts as is set by environment variables, never by the model:
  MEMORY_TENANT    which organisation's graph (default "demo")
  MEMORY_AGENT     the agent's role name, e.g. "Researcher"
  MEMORY_OWNER     the user the agent belongs to, e.g. "Priya" (optional)
  MEMORY_DOCS_DIR  the only folder verify_fact may read documents from (optional)
FalkorDB credentials and ANTHROPIC_API_KEY come from .env. Identity is not authenticated.

Run:  uv run python mcp_server.py
"""
import functools
import os

from mcp.server.mcpserver import MCPServer

from memory import Tenants
from tools import TOOL_NAMES, AgentTools

INSTRUCTIONS = (
    "Shared memory for a team of agents. Write facts with remember, share them with publish, "
    "build decisions on shared facts, and when a fact turns out wrong use retract or correct: "
    "everything built on it is flagged. Trust warnings on recalled facts mean: verify before relying.")


def build_server(agent_tools):
    """An MCP server whose tools are the AgentTools methods. Schemas come from the same type hints
    that anthropic_tools() uses, so the two lists cannot drift apart."""
    server = MCPServer("shared-memory", instructions=INSTRUCTIONS)
    for name in TOOL_NAMES:
        server.add_tool(_through_call(agent_tools, name), name=name)
    return server


def _through_call(agent_tools, name):
    """The tool's own signature (so the schema is unchanged), but run via call(), so failures
    reach the model as {"error": "<reason>"} instead of MCP's generic "Error executing tool"."""
    @functools.wraps(getattr(agent_tools, name))
    def run(**kwargs):
        return agent_tools.call(name, kwargs)
    return run


def main():
    memory = Tenants().get(os.environ.get("MEMORY_TENANT", "demo"))
    agent_tools = AgentTools.open(
        memory, os.environ.get("MEMORY_AGENT", "Assistant"), owner=os.environ.get("MEMORY_OWNER"),
        docs_dir=os.environ.get("MEMORY_DOCS_DIR"))
    build_server(agent_tools).run("stdio")


if __name__ == "__main__":
    main()
