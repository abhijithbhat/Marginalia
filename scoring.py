"""Deterministic relevance scoring for research papers.

This module is not an agent tool. It is an un-corruptible safety-critical gate
called directly by orchestration code to rank and filter papers based on a
transparent rubric: 70% keyword overlap and 30% recency decay.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import re


def load_profile(path: str = "profile.json") -> dict:
    """Read and return topic profile settings from a JSON file.

    Args:
        path: Path to the profile JSON file (defaults to "profile.json").

    Returns:
        dict: Parsed profile configuration containing keywords and recency window.
    """
    file_path = Path(path)
    if not file_path.is_absolute() and not file_path.exists():
        candidate = Path(__file__).resolve().parent / path
        if candidate.exists():
            file_path = candidate

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def score_relevance(paper: dict, profile: dict) -> dict:
    """Compute deterministic relevance score for a paper against a topic profile.

    This function does NOT call any LLM. It calculates:
    1. Keyword score (70%): Title substring matches count 2x, abstract matches 1x.
    2. Recency score (30%): Linear decay over recency_window_days.

    Args:
        paper: Dictionary with keys 'title', 'abstract', 'published_date', etc.
        profile: Dictionary with keys 'keywords' and 'recency_window_days'.

    Returns:
        dict: Scoring breakdown containing:
            - "total_score" (float): 0.7 * keyword_score + 0.3 * recency_score (rounded to 3 decimals)
            - "keyword_score" (float): Normalized keyword match score (0.0 - 1.0)
            - "recency_score" (float): Recency decay score (0.0 - 1.0)
            - "matched_keywords" (list[str]): List of keywords present in title/abstract
            - "days_since_published" (float): Number of days elapsed since publication
    """
    keywords = profile.get("keywords", [])
    recency_window_days = float(profile.get("recency_window_days", 30))

    title = paper.get("title", "")
    abstract = paper.get("abstract", "")

    # --- 1. Keyword Score ---
    matched_weights = 0
    matched_keywords: list[str] = []

    for kw in keywords:
        kw_clean = kw.strip()
        variant = kw_clean[:-1] if kw_clean.lower().endswith("s") else kw_clean + "s"
        variants = (kw_clean, variant)

        in_title = any(
            bool(re.search(r"\b" + re.escape(v) + r"\b", title, re.IGNORECASE))
            for v in variants
        )
        in_abstract = any(
            bool(re.search(r"\b" + re.escape(v) + r"\b", abstract, re.IGNORECASE))
            for v in variants
        )

        if in_title or in_abstract:
            matched_keywords.append(kw)
        if in_title:
            matched_weights += 2
        if in_abstract:
            matched_weights += 1

    max_possible_weight = len(keywords) * 2
    if max_possible_weight > 0:
        keyword_score = min(1.0, max(0.0, matched_weights / max_possible_weight))
    else:
        keyword_score = 0.0

    # --- 2. Recency Score ---
    pub_date_str = paper.get("published_date", "")
    try:
        # Python 3.11+ handles '...Z' suffix natively via datetime.fromisoformat
        pub_dt = datetime.fromisoformat(pub_date_str)
        now_dt = datetime.now(timezone.utc)
        days_elapsed = (now_dt - pub_dt).total_seconds() / 86400.0
        days_elapsed = max(0.0, days_elapsed)
    except Exception:
        days_elapsed = recency_window_days

    recency_score = max(0.0, 1.0 - (days_elapsed / recency_window_days))

    # --- 3. Total Score ---
    total_score = round(0.7 * keyword_score + 0.3 * recency_score, 3)

    return {
        "total_score": total_score,
        "keyword_score": round(keyword_score, 3),
        "recency_score": round(recency_score, 3),
        "matched_keywords": matched_keywords,
        "days_since_published": round(days_elapsed, 1),
    }


if __name__ == "__main__":
    from tools.fetch_papers import fetch_new_papers

    profile = load_profile()
    papers = fetch_new_papers("large language model agents", "2026-08-01", 8)

    scored_papers = []
    for paper in papers:
        breakdown = score_relevance(paper, profile)
        scored_papers.append((paper, breakdown))

    # Sort descending by total_score
    scored_papers.sort(key=lambda item: item[1]["total_score"], reverse=True)

    print("=" * 80)
    print(f"SCORED PAPERS ({len(scored_papers)} total) — SORTED BY TOTAL SCORE DESCENDING")
    print("=" * 80)

    surpassed_threshold = 0
    below_threshold = 0

    for idx, (paper, breakdown) in enumerate(scored_papers, 1):
        is_above = breakdown["total_score"] >= 0.3 and breakdown["keyword_score"] > 0
        if is_above:
            surpassed_threshold += 1
        else:
            below_threshold += 1

        status_tag = (
            "SURFACE (>= 0.3 & kw > 0)" if is_above else "FILTER OUT (< 0.3 or kw == 0)"
        )
        print(f"\n{idx}. [{status_tag}] Total Score: {breakdown['total_score']}")
        print(f"   Title: {paper.get('title')}")
        print(
            f"   arXiv ID: {paper.get('arxiv_id')} | Published: {paper.get('published_date')} ({breakdown['days_since_published']} days ago)"
        )
        print(
            f"   Keyword Score: {breakdown['keyword_score']} | Recency Score: {breakdown['recency_score']}"
        )
        print(
            f"   Matched Keywords: {breakdown['matched_keywords'] if breakdown['matched_keywords'] else 'None'}"
        )

    print("\n" + "=" * 80)
    print("Threshold evaluation (bar: total_score >= 0.3 AND keyword_score > 0):")
    print(f"  Surfaced: {surpassed_threshold} papers")
    print(f"  Fell below: {below_threshold} papers")
    print("=" * 80)
