"""FastMCP server exposing arXiv paper retrieval tool for Marginalia."""

from mcp.server.fastmcp import FastMCP
from tools.fetch_papers import fetch_new_papers

mcp = FastMCP("marginalia-arxiv")


@mcp.tool()
def search_papers(
    topic_query: str, since_date: str, max_results: int = 10
) -> list[dict]:
    """Search arXiv for recent papers matching a topic query, filtered by
    publication date. topic_query: natural-language phrase or a
    structured arXiv query (with field prefixes / AND/OR). since_date:
    'YYYY-MM-DD', only papers published on or after this date. Returns a
    list of paper metadata dicts."""
    return fetch_new_papers(topic_query, since_date, max_results)


if __name__ == "__main__":
    mcp.run()
