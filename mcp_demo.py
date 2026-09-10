"""End-to-end demonstration of Marginalia's MCP server and Strands MCP client.

Launches mcp_server.py as a subprocess over stdio transport, discovers exposed tools
via the Model Context Protocol, attaches them to a Strands Agent, and executes a real query.
"""

import os
from pathlib import Path
import sys
from dotenv import load_dotenv
from mcp import StdioServerParameters, stdio_client
from strands import Agent
from strands.models.openai import OpenAIModel
from strands.tools.mcp import MCPClient


def main() -> None:
    # 1. Load environment variables
    base_dir = Path(__file__).resolve().parent
    load_dotenv(base_dir / ".env")
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is not set in .env")

    # 2. Configure stdio transport for the FastMCP server subprocess
    server_script = base_dir / "mcp_server.py"
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_script)],
        cwd=str(base_dir),
    )

    print("=" * 80)
    print("INITIALIZING STRANDS MCP CLIENT (STDIO TRANSPORT)")
    print(f"Target Server: {server_script}")
    print("=" * 80)

    # 3. Connect to FastMCP server via Strands MCPClient
    client = MCPClient(lambda: stdio_client(server_params))

    with client:
        # Discover tools dynamically over MCP protocol
        discovered_tools = client.list_tools_sync()

        print(f"\n[MCP Discovery] Found {len(discovered_tools)} tool(s) over protocol:")
        for idx, tool in enumerate(discovered_tools, 1):
            name = getattr(tool, "tool_name", str(tool))
            desc = (
                tool.tool_spec.get("description", "").strip()
                if hasattr(tool, "tool_spec")
                else ""
            )
            print(f"\n  {idx}. Tool Name: {name}")
            print(f"     Description: {desc}")

        # 4. Initialize model and agent with MCP-discovered tools
        model = OpenAIModel(
            client_args={
                "api_key": groq_api_key,
                "base_url": "https://api.groq.com/openai/v1",
            },
            model_id="openai/gpt-oss-120b",
        )

        agent = Agent(
            model=model,
            tools=list(discovered_tools),
        )

        prompt = (
            "Use the search_papers tool to find 3 recent papers about 'LLM agents' "
            "published since 2026-08-01, then list their titles."
        )

        print("\n" + "=" * 80)
        print("AGENT PROMPT:")
        print(f"\"{prompt}\"")
        print("=" * 80)
        print("\nExecuting agent invocation (inspect server stdio logs below)...")

        # 5. Execute agent call
        response = agent(prompt)

        print("\n" + "=" * 80)
        print("AGENT RESPONSE:")
        print("=" * 80)
        print(str(response))
        print("=" * 80)


if __name__ == "__main__":
    main()
