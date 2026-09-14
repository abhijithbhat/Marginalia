"""End-to-end research digest pipeline for Marginalia.

Coordinates:
1. Deterministic arXiv retrieval covering full profile keywords (deduplicated by arxiv_id)
2. Strands Agent reasoning over candidate papers against corpus themes (single turn, no tool calls)
3. Deterministic relevance scoring (70% keyword, 30% recency)
4. Vector-based claim verification against prior researcher notes in Qdrant Cloud
5. State persistence (last_run.json and seen_ids.json) to prevent duplicate surfacing across digests
"""

from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import re
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from strands import Agent
from strands.models.openai import OpenAIModel
from strands.types.exceptions import MaxTokensReachedException

from scoring import load_profile, score_relevance
from tools.fetch_papers import fetch_new_papers
from verification import skeptic_review, verify_claim

logger = logging.getLogger(__name__)


def get_since_date(default_days: int = 7) -> str:
    """Read last_run_date from last_run.json, defaulting to 7 days ago if not found.

    Args:
        default_days: Number of days in the past to default to (default: 7).

    Returns:
        str: Date formatted as "YYYY-MM-DD".
    """
    last_run_file = Path(__file__).resolve().parent / "last_run.json"
    if last_run_file.exists():
        try:
            with open(last_run_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                date_val = data.get("last_run_date")
                if date_val:
                    return date_val
        except Exception as e:
            logger.warning("Could not read last_run.json: %s", e)

    fallback_date = datetime.now(timezone.utc) - timedelta(days=default_days)
    return fallback_date.strftime("%Y-%m-%d")


def parse_agent_json(raw_text: str) -> list[dict]:
    """Parse JSON array from agent response text with fallback extraction.

    Degrades gracefully to an empty list on failure.

    Args:
        raw_text: Raw string output from the agent.

    Returns:
        list[dict]: Parsed list of claim objects.
    """
    if not raw_text:
        return []

    cleaned = raw_text.strip()

    # 1. Direct JSON parse
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # 2. Extract from markdown code block ```json [...] ```
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, list):
                return data
        except Exception:
            pass

    # 3. Extract bracketed array [...]
    match = re.search(r"(\[.*\])", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, list):
                return data
        except Exception:
            pass

    # 4. Fallback: extract individual JSON objects {...}
    obj_matches = re.findall(r"\{[^{}]*\"arxiv_id\"[^{}]*\}", cleaned)
    if obj_matches:
        fallback_list = []
        for obj_str in obj_matches:
            try:
                obj = json.loads(obj_str)
                if isinstance(obj, dict) and "arxiv_id" in obj:
                    fallback_list.append(obj)
            except Exception:
                pass
        if fallback_list:
            return fallback_list

    # 5. Fallback: extract from numbered bullet points if model formatted as text list
    bullet_matches = re.findall(
        r"(\d{4}\.\d{4,5}(?:v\d+)?)\s*[\u2013\u2014\-:]\s*(?:connection:\s*)?([^\n]+)",
        cleaned,
    )
    if bullet_matches:
        return [
            {"arxiv_id": aid, "connection_claim": claim.strip()}
            for aid, claim in bullet_matches
        ]

    logger.warning("Failed to parse agent response as JSON. Raw response: %s", raw_text)
    return []


def run_digest(since_date: str) -> list[dict]:
    """Run the complete research digest pipeline.

    Args:
        since_date: Cutoff date in "YYYY-MM-DD" format.

    Returns:
        list[dict]: Filtered and verified papers with full metadata, scores, and claims.
    """
    base_dir = Path(__file__).resolve().parent
    env_path = base_dir / ".env"
    load_dotenv(dotenv_path=env_path)

    groq_api_key = os.getenv("GROQ_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is not set in .env")
    if not qdrant_url or not qdrant_api_key:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in .env")

    # Load topic profile
    profile = load_profile()
    keywords = profile.get("keywords", [])

    # Core themes established in the researcher's corpus
    corpus_themes = [
        "Multi-agent systems coordination and error propagation",
        "Tool use and interface contracts / docstrings",
        "Language model agents, loops, and deterministic safety checks",
        "Retrieval augmented generation (RAG) and claim verification (RegBrain)",
        "Agent pipeline orchestration and modular handoffs",
        "Model Context Protocol (MCP) tool standard",
    ]
    themes_readable = "; ".join(corpus_themes)

    # --- Step 1 (Deterministic retrieval): query arXiv for each keyword directly ---
    print(f"Fetching recent papers across {len(keywords)} profile keywords since {since_date}...")
    deduped_candidates: dict[str, dict] = {}
    for kw in keywords:
        papers = fetch_new_papers(topic_query=kw, since_date=since_date, max_results=5)
        for p in papers:
            aid = p.get("arxiv_id")
            if aid and aid not in deduped_candidates:
                deduped_candidates[aid] = p

    print(f"Retrieved {len(deduped_candidates)} unique candidate papers.")

    if not deduped_candidates:
        return []

    # Lookup map for matching arXiv IDs (with or without version suffix)
    id_lookup: dict[str, dict] = {}
    for aid, p in deduped_candidates.items():
        id_lookup[aid] = p
        id_lookup[aid.lower()] = p
        if "v" in aid:
            base_aid = aid.split("v")[0]
            id_lookup[base_aid] = p
            id_lookup[base_aid.lower()] = p

    # --- Step 2 (Agent reasoning, no tools): prompt agent with compact candidates list ---
    candidate_entries = []
    for aid, p in deduped_candidates.items():
        title = p.get("title", "").strip()
        abstract = p.get("abstract", "").strip()
        if len(abstract) > 300:
            abstract = abstract[:300] + "..."
        candidate_entries.append(
            f"- arXiv ID: {aid}\n  Title: {title}\n  Abstract: {abstract}"
        )

    candidates_text = "\n\n".join(candidate_entries)

    model = OpenAIModel(
        client_args={
            "api_key": groq_api_key,
            "base_url": "https://api.groq.com/openai/v1",
        },
        model_id="openai/gpt-oss-120b",
        params={"max_tokens": 8192},
    )

    system_prompt = (
        "You are an autonomous research-digest assistant. Evaluate the provided candidate papers "
        "against the established research themes. "
        "Output ONLY a raw JSON array of objects with 'arxiv_id' and 'connection_claim'. "
        "Start your response immediately with '[' and end with ']'. "
        "DO NOT write introductory text, bullet points, preliminary analysis, or drafting. "
        "Omit any paper that does not have a genuine connection. Do NOT invent connections."
    )

    # Agent with NO tools attached
    agent = Agent(
        model=model,
        system_prompt=system_prompt,
    )

    prompt = (
        f"Candidate papers published on or after {since_date}:\n\n"
        f"{candidates_text}\n\n"
        f"Established research themes:\n"
        f"{themes_readable}\n\n"
        f"For each paper with a genuine connection to one of these themes, write one concise sentence explaining how the paper addresses or examines that specific theme (e.g., 'Examines [topic], relating directly to our note on [theme] that [key principle]'). Be conservative and precise: do not claim the note validates the paper's empirical results. "
        f"Output ONLY a JSON array starting with '[' and ending with ']'. "
        f"Do NOT write any thinking, notes, or explanations outside the JSON:\n"
        f'[{{"arxiv_id": "...", "connection_claim": "..."}}, ...]'
    )

    print("Submitting candidate papers to Strands Agent for connection reasoning...")
    try:
        agent_result = agent(prompt)
        raw_output = str(agent_result)
    except MaxTokensReachedException as e:
        logger.warning("Agent reached max tokens limit: %s", e)
        raw_output = ""
        if hasattr(agent, "messages") and agent.messages:
            last_msg = agent.messages[-1]
            if isinstance(last_msg, dict):
                content = last_msg.get("content", [])
                if isinstance(content, list):
                    raw_output = "".join(
                        c.get("text", "") for c in content if isinstance(c, dict)
                    )
                elif isinstance(content, str):
                    raw_output = content
            else:
                raw_output = str(last_msg)
    except Exception as e:
        logger.error("Agent reasoning call failed: %s", e)
        raw_output = ""

    parsed_items = parse_agent_json(raw_output)
    print(f"Agent identified connections for {len(parsed_items)} paper(s).")

    # Connect to Qdrant Cloud for claim verification
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    # --- Steps 3 onward: score, verify, and filter candidates ---
    candidates_survived: list[dict] = []
    for item in parsed_items:
        item_id = item.get("arxiv_id", "").strip()
        if not item_id:
            continue

        paper = id_lookup.get(item_id) or id_lookup.get(item_id.lower())
        if not paper and "v" in item_id:
            paper = id_lookup.get(item_id.split("v")[0])

        if not paper:
            continue

        claim_text = item.get("connection_claim", "").strip()

        # Unconditional deterministic scoring
        score = score_relevance(paper, profile)

        # Unconditional semantic verification
        verified = verify_claim(
            claim_text=claim_text,
            qdrant_client=qdrant_client,
            collection_name="marginalia_notes",
            threshold=0.5,
        )

        # Adversarial verification tier (only runs on claims that passed verify_claim)
        if verified.get("verified"):
            skeptic = skeptic_review(
                claim_text=claim_text,
                supporting_excerpt=verified.get("supporting_excerpt") or "",
                groq_api_key=groq_api_key,
            )
            best_score = verified.get("best_match_score", 0.0)
            if best_score >= 0.6 and not skeptic.get("has_objection"):
                confidence_tier = "strongly_verified"
                skeptic_objection = None
            elif best_score >= 0.5:
                confidence_tier = "verified_flagged"
                skeptic_objection = skeptic.get("objection_text")
            else:
                confidence_tier = "rejected"
                skeptic_objection = None
        else:
            confidence_tier = "rejected"
            skeptic_objection = None

        # Filter gate: total_score >= 0.3 AND keyword_score > 0
        if score["total_score"] >= 0.3 and score["keyword_score"] > 0:
            candidates_survived.append(
                {
                    **paper,
                    "score": score,
                    "claim": claim_text,
                    "claim_verified": verified,
                    "confidence_tier": confidence_tier,
                    "skeptic_objection": skeptic_objection,
                }
            )

    # Sort survivors by total_score descending
    candidates_survived.sort(key=lambda p: p["score"]["total_score"], reverse=True)

    # --- Deduplication against seen_ids.json ---
    seen_ids_file = base_dir / "seen_ids.json"
    seen_ids_list: list[str] = []
    if seen_ids_file.exists():
        try:
            with open(seen_ids_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    seen_ids_list = data
        except Exception as e:
            logger.warning("Could not read seen_ids.json: %s", e)

    seen_ids_set = set(seen_ids_list)
    final_survivors: list[dict] = []
    new_surfaced_ids: list[str] = []

    for p in candidates_survived:
        aid = p.get("arxiv_id", "")
        base_aid = aid.split("v")[0] if "v" in aid else aid
        if aid not in seen_ids_set and base_aid not in seen_ids_set:
            final_survivors.append(p)
            new_surfaced_ids.append(aid)

    # Update seen_ids.json with newly surfaced IDs
    seen_ids_list.extend(new_surfaced_ids)
    with open(seen_ids_file, "w", encoding="utf-8") as f:
        json.dump(seen_ids_list, f, indent=2)

    # Write today's date back to last_run.json after successful run
    last_run_file = base_dir / "last_run.json"
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with open(last_run_file, "w", encoding="utf-8") as f:
        json.dump({"last_run_date": today_str}, f, indent=2)

    return final_survivors


if __name__ == "__main__":
    since_date = get_since_date(default_days=7)
    print(f"Initializing Marginalia Digest Pipeline (since: {since_date})...\n")

    results = run_digest(since_date)

    # Write full result list to digest.json at project root (pretty-printed)
    digest_file = Path(__file__).resolve().parent / "digest.json"
    if results:
        with open(digest_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Digest saved to {digest_file.name}")
    else:
        print(f"No new papers surfaced in this run. Preserving existing {digest_file.name}")

    print("\n" + "=" * 80)
    print(f"DIGEST PIPELINE RESULTS ({len(results)} paper(s) surfaced)")
    print("=" * 80)

    for idx, paper in enumerate(results, 1):
        print(f"\n{idx}. Title: {paper.get('title')}")
        print(f"   arXiv ID: {paper.get('arxiv_id')}")
        print(f"   Published Date: {paper.get('published_date')}")
        print(f"   PDF URL: {paper.get('pdf_url')}")
        print(f"   Authors: {', '.join(paper.get('authors', []))}")
        print(f"   Score Breakdown:")
        print(f"     - Total Score: {paper['score']['total_score']}")
        print(f"     - Keyword Score: {paper['score']['keyword_score']}")
        print(f"     - Recency Score: {paper['score']['recency_score']}")
        print(f"     - Matched Keywords: {paper['score']['matched_keywords']}")
        print(f"     - Days Since Published: {paper['score']['days_since_published']}")
        print(f"   Agent's Connection Claim:")
        print(f"     \"{paper.get('claim')}\"")

        verified_info = paper.get("claim_verified", {})
        tier = paper.get("confidence_tier", "rejected").upper()
        status = "VERIFIED (PASS)" if verified_info.get("verified") else "REJECTED (FAIL)"
        print(f"   Claim Verification Status: {status} | Tier: {tier}")
        print(f"     - Match Score: {verified_info.get('best_match_score')}")
        print(f"     - Matched Note Source: {verified_info.get('matched_source')}")
        if paper.get("skeptic_objection"):
            print(f"     - Skeptic Objection: \"{paper.get('skeptic_objection')}\"")
        if verified_info.get("supporting_excerpt"):
            print(f"     - Supporting Excerpt: \"{verified_info.get('supporting_excerpt')}\"")
        else:
            print(f"     - Supporting Excerpt: None")

    print("\n" + "=" * 80)
