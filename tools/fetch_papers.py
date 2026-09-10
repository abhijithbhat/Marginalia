"""arXiv paper retrieval tool for the Marginalia research digest agent."""

from datetime import date, datetime
import pprint
import feedparser
import requests
from strands import tool


@tool
def fetch_new_papers(
    topic_query: str, since_date: str, max_results: int = 10
) -> list[dict]:
    """Fetch new research papers from arXiv matching a topic query and filter by submission date.

    Args:
        topic_query (str): The research topic or keywords to search for on arXiv
            (e.g., "large language model agents").
        since_date (str): The cutoff date in "YYYY-MM-DD" format (e.g., "2026-08-01").
            Only papers published on or after this date will be included.
        max_results (int, optional): The maximum number of papers to retrieve from
            the arXiv API. Defaults to 10.

    Returns:
        list[dict]: A list of dictionaries representing matching papers. Each dictionary
            contains the following keys:
            - "arxiv_id" (str): The arXiv paper identifier (e.g., "2609.05416v1").
            - "title" (str): The cleaned title of the paper.
            - "authors" (list[str]): A list of author names.
            - "abstract" (str): The abstract/summary text of the paper.
            - "published_date" (str): The publication timestamp (e.g., "2026-09-04T17:59:35Z").
            - "pdf_url" (str): The direct URL to the PDF version of the paper.

        Returns an empty list if the request fails or no matching papers are found.
    """
    try:
        ARXIV_PREFIXES = (
            "au:",
            "ti:",
            "abs:",
            "cat:",
            "co:",
            "jr:",
            "rn:",
            "id:",
            "all:",
        )
        BOOLEAN_OPERATORS = (" AND ", " OR ", " ANDNOT ")

        has_prefix_or_boolean = any(
            prefix in topic_query for prefix in ARXIV_PREFIXES
        ) or any(op in topic_query for op in BOOLEAN_OPERATORS)

        if not has_prefix_or_boolean and " " in topic_query:
            search_query_value = f'all:"{topic_query}"'
        else:
            search_query_value = topic_query

        params = {
            "search_query": search_query_value,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": max_results,
        }
        api_url = "http://export.arxiv.org/api/query"
        response = requests.get(api_url, params=params, timeout=30)
        if response.status_code != 200:
            return []

        feed = feedparser.parse(response.content)
        if not feed.entries:
            return []

        # Parse the cutoff date threshold
        try:
            since_dt = datetime.strptime(since_date.strip(), "%Y-%m-%d").date()
        except Exception:
            since_dt = None

        papers: list[dict] = []
        for entry in feed.entries:
            # Extract published date for comparison
            entry_pub_date = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                entry_pub_date = date(
                    entry.published_parsed.tm_year,
                    entry.published_parsed.tm_mon,
                    entry.published_parsed.tm_mday,
                )
            elif hasattr(entry, "published") and entry.published:
                try:
                    entry_pub_date = datetime.strptime(
                        entry.published[:10], "%Y-%m-%d"
                    ).date()
                except Exception:
                    pass

            # Filter out entries published before since_date
            if since_dt and entry_pub_date and entry_pub_date < since_dt:
                continue

            raw_id = entry.get("id", "")
            arxiv_id = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id
            title = " ".join(entry.get("title", "").split())
            authors = [
                a.get("name", "").strip()
                for a in entry.get("authors", [])
                if a.get("name")
            ]
            abstract = entry.get("summary", "").strip()
            published_date = entry.get("published", "").strip()

            pdf_url = ""
            for link in entry.get("links", []):
                if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                    pdf_url = link.get("href", "")
                    break
            if not pdf_url and arxiv_id:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            papers.append(
                {
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "published_date": published_date,
                    "pdf_url": pdf_url,
                }
            )

        return papers
    except Exception:
        return []


if __name__ == "__main__":
    print("--- Naive phrase, now auto-quoted ---")
    results_naive = fetch_new_papers("large language model agents", "2026-08-01", 5)
    print(f"Retrieved {len(results_naive)} paper(s):")
    for idx, paper in enumerate(results_naive, 1):
        print(f"  {idx}. {paper['title']} ({paper['published_date']})")
    print("\nFull result details:")
    pprint.pprint(results_naive, indent=2)

    print("\n--- Pre-structured query, passed through untouched ---")
    results_structured = fetch_new_papers(
        'cat:cs.AI AND abs:"language model" AND abs:agent', "2026-08-01", 5
    )
    print(f"Retrieved {len(results_structured)} paper(s):")
    for idx, paper in enumerate(results_structured, 1):
        print(f"  {idx}. {paper['title']} ({paper['published_date']})")
    print("\nFull result details:")
    pprint.pprint(results_structured, indent=2)

