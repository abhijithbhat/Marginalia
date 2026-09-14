"""Marginalia Digest Review Dashboard.

A single-file Flask web application for reviewing research digest candidates,
inspecting deterministic scores and verification claims, making review decisions,
and expanding the persistent note corpus and Qdrant vector index upon approval.
"""

from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template_string, request, url_for
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# Setup logging and directories
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DIGEST_FILE = BASE_DIR / "digest.json"
DECISIONS_FILE = BASE_DIR / "decisions.json"
CORPUS_DIR = BASE_DIR / "corpus"
CORPUS_DIR.mkdir(exist_ok=True)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "marginalia_notes"

# Initialize Qdrant Client
qdrant_client = None
if QDRANT_URL and QDRANT_API_KEY:
    try:
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    except Exception as e:
        logger.warning("Could not initialize QdrantClient: %s", e)

# Lazy-loaded embedding model for incremental upserts (keeps worker boot instant and within 512MB RAM)
_embedding_model = None


def get_embedding_model():
    """Load SentenceTransformer on first approval to keep worker memory low."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


app = Flask(__name__)


def load_digest() -> list[dict]:
    """Load candidate papers from digest.json."""
    if not DIGEST_FILE.exists():
        return []
    try:
        with open(DIGEST_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        logger.error("Failed to read digest.json: %s", e)
        return []


def load_decisions() -> dict[str, dict]:
    """Load existing decisions mapped by arxiv_id."""
    if not DECISIONS_FILE.exists():
        return {}
    try:
        with open(DECISIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return {item["arxiv_id"]: item for item in data if "arxiv_id" in item}
    except Exception as e:
        logger.error("Failed to read decisions.json: %s", e)
    return {}


def save_decision_record(arxiv_id: str, decision: str) -> dict:
    """Append decision to decisions.json and return the saved record."""
    records: list[dict] = []
    if DECISIONS_FILE.exists():
        try:
            with open(DECISIONS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    records = data
        except Exception:
            records = []

    # Update or append
    now_str = datetime.now(timezone.utc).isoformat()
    updated = False
    for rec in records:
        if rec.get("arxiv_id") == arxiv_id:
            rec["decision"] = decision
            rec["timestamp"] = now_str
            updated = True
            break

    if not updated:
        new_record = {
            "arxiv_id": arxiv_id,
            "decision": decision,
            "timestamp": now_str,
        }
        records.append(new_record)
    else:
        new_record = {"arxiv_id": arxiv_id, "decision": decision, "timestamp": now_str}

    with open(DECISIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    return new_record


def index_approved_paper(paper: dict) -> Path:
    """Save approved paper to corpus/ and incrementally upsert vector to Qdrant."""
    arxiv_id = paper.get("arxiv_id", "unknown")
    safe_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", arxiv_id)
    file_path = CORPUS_DIR / f"approved_{safe_id}.md"

    # 1. Write structured markdown note to corpus/
    title = paper.get("title", "Untitled")
    authors = ", ".join(paper.get("authors", []))
    published_date = paper.get("published_date", "")
    abstract = paper.get("abstract", "").strip()
    claim = paper.get("claim", "").strip()
    pdf_url = paper.get("pdf_url", "")

    content = (
        f"# {title}\n\n"
        f"**arXiv ID**: {arxiv_id}\n\n"
        f"**Published Date**: {published_date}\n\n"
        f"**Authors**: {authors}\n\n"
        f"**PDF**: {pdf_url}\n\n"
        f"## Abstract\n{abstract}\n\n"
        f"## Established Connection Claim\n{claim}\n"
    )
    file_path.write_text(content, encoding="utf-8")
    logger.info("Saved approved paper to %s", file_path)

    # 2. Incremental vector upsert to Qdrant Cloud without rebuilding
    if qdrant_client:
        try:
            vector = get_embedding_model().encode(content).tolist()

            source_filename = file_path.name
            point_id = int(hashlib.sha256(source_filename.encode()).hexdigest()[:16], 16)

            point = PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "source": source_filename,
                    "text": content,
                },
            )
            qdrant_client.upsert(collection_name=COLLECTION_NAME, points=[point])
            logger.info(
                "Upserted vector point id=%d for %s into %s",
                point_id,
                source_filename,
                COLLECTION_NAME,
            )
        except Exception as e:
            logger.error("Failed to upsert to Qdrant: %s", e)

    return file_path


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Marginalia — Research Digest Review</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300..800;1,300..800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #070a12;
      --bg-card: rgba(17, 24, 39, 0.75);
      --bg-card-hover: rgba(26, 36, 56, 0.85);
      --border-subtle: rgba(255, 255, 255, 0.08);
      --border-active: rgba(99, 102, 241, 0.4);
      --text-main: #f3f4f6;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      
      --verified-color: #10b981;
      --verified-bg: rgba(16, 185, 129, 0.1);
      --verified-border: rgba(16, 185, 129, 0.3);
      
      --rejected-color: #f43f5e;
      --rejected-bg: rgba(244, 63, 94, 0.08);
      --rejected-border: rgba(244, 63, 94, 0.28);
      
      --accent: #6366f1;
      --accent-glow: rgba(99, 102, 241, 0.2);
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      background-color: var(--bg-dark);
      background-image: 
        radial-gradient(circle at 15% 10%, rgba(99, 102, 241, 0.12) 0%, transparent 40%),
        radial-gradient(circle at 85% 60%, rgba(16, 185, 129, 0.07) 0%, transparent 45%);
      background-attachment: fixed;
      color: var(--text-main);
      font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
      line-height: 1.5;
      padding: 2.5rem 1.5rem 5rem;
      min-height: 100vh;
    }

    .container {
      max-width: 1180px;
      margin: 0 auto;
    }

    /* Header */
    header {
      margin-bottom: 2.5rem;
      border-bottom: 1px solid var(--border-subtle);
      padding-bottom: 1.75rem;
    }

    .header-top {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      flex-wrap: wrap;
      gap: 1.5rem;
      margin-bottom: 1.5rem;
    }

    .brand h1 {
      font-size: 2rem;
      font-weight: 800;
      letter-spacing: -0.03em;
      background: linear-gradient(135deg, #ffffff 40%, #a5b4fc 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      display: flex;
      align-items: center;
      gap: 0.6rem;
    }

    .brand p {
      color: var(--text-muted);
      font-size: 0.95rem;
      margin-top: 0.35rem;
    }

    .qdrant-pill {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--border-subtle);
      padding: 0.4rem 0.9rem;
      border-radius: 9999px;
      font-size: 0.8rem;
      color: var(--text-muted);
      font-family: 'JetBrains Mono', monospace;
    }

    .qdrant-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--verified-color);
      box-shadow: 0 0 10px var(--verified-color);
    }

    /* Stats Grid */
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 1rem;
      margin-bottom: 2rem;
    }

    .stat-card {
      background: var(--bg-card);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-subtle);
      border-radius: 1rem;
      padding: 1.15rem 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 0.3rem;
      transition: transform 0.2s, border-color 0.2s;
    }

    .stat-card:hover {
      border-color: var(--border-active);
      transform: translateY(-2px);
    }

    .stat-label {
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-dim);
      font-weight: 600;
    }

    .stat-value {
      font-size: 1.8rem;
      font-weight: 800;
      font-family: 'JetBrains Mono', monospace;
      color: #fff;
    }

    .stat-value.verified { color: var(--verified-color); }
    .stat-value.rejected { color: var(--rejected-color); }
    .stat-value.approved { color: #38bdf8; }

    /* Controls: Filter & Search */
    .controls-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 1rem;
      flex-wrap: wrap;
      margin-bottom: 2rem;
      background: rgba(17, 24, 39, 0.5);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      padding: 0.6rem 0.8rem;
    }

    .filter-tabs {
      display: flex;
      gap: 0.35rem;
      flex-wrap: wrap;
    }

    .tab-btn {
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-size: 0.85rem;
      font-weight: 600;
      padding: 0.45rem 0.9rem;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.15s ease;
    }

    .tab-btn:hover {
      color: #fff;
      background: rgba(255, 255, 255, 0.05);
    }

    .tab-btn.active {
      background: var(--accent);
      color: #fff;
      box-shadow: 0 0 12px var(--accent-glow);
    }

    .search-box {
      flex: 1;
      max-width: 320px;
      min-width: 200px;
      position: relative;
    }

    .search-box input {
      width: 100%;
      background: rgba(0, 0, 0, 0.3);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 0.5rem 0.9rem;
      color: #fff;
      font-size: 0.85rem;
      outline: none;
      transition: border-color 0.2s;
    }

    .search-box input:focus {
      border-color: var(--accent);
    }

    /* Paper List */
    .paper-list {
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
    }

    .paper-card {
      background: var(--bg-card);
      backdrop-filter: blur(16px);
      border: 1px solid var(--border-subtle);
      border-radius: 1.25rem;
      padding: 1.75rem;
      transition: all 0.25s ease;
      position: relative;
      overflow: hidden;
    }

    .paper-card:hover {
      background: var(--bg-card-hover);
      border-color: rgba(255, 255, 255, 0.14);
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }

    .paper-card.decided-approve {
      border-left: 4px solid var(--verified-color);
    }

    .paper-card.decided-skip {
      border-left: 4px solid var(--text-dim);
      opacity: 0.72;
    }

    .paper-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 1.25rem;
      margin-bottom: 1rem;
    }

    .title-area {
      flex: 1;
    }

    .title-area h2 {
      font-size: 1.25rem;
      font-weight: 700;
      line-height: 1.4;
      margin-bottom: 0.45rem;
    }

    .title-area h2 a {
      color: #fff;
      text-decoration: none;
      transition: color 0.15s;
    }

    .title-area h2 a:hover {
      color: #a5b4fc;
      text-decoration: underline;
    }

    .paper-meta {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.75rem;
      font-size: 0.82rem;
      color: var(--text-muted);
    }

    .arxiv-badge {
      font-family: 'JetBrains Mono', monospace;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--border-subtle);
      padding: 0.15rem 0.5rem;
      border-radius: 6px;
      color: #cbd5e1;
    }

    /* Score Badge Container */
    .score-badge {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      background: rgba(0, 0, 0, 0.4);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      padding: 0.6rem 1rem;
      min-width: 85px;
      text-align: center;
    }

    .score-val {
      font-family: 'JetBrains Mono', monospace;
      font-size: 1.35rem;
      font-weight: 800;
      color: #a5b4fc;
    }

    .score-label {
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-dim);
      font-weight: 600;
    }

    /* Score Breakdown Row */
    .score-breakdown-row {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 1.25rem;
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid rgba(255, 255, 255, 0.04);
      border-radius: 10px;
      padding: 0.65rem 1rem;
      margin: 1rem 0;
      font-size: 0.82rem;
    }

    .metric-item {
      display: flex;
      align-items: center;
      gap: 0.45rem;
    }

    .metric-name {
      color: var(--text-dim);
    }

    .metric-val {
      font-family: 'JetBrains Mono', monospace;
      color: #e2e8f0;
      font-weight: 600;
    }

    .keyword-pill {
      background: rgba(99, 102, 241, 0.15);
      border: 1px solid rgba(99, 102, 241, 0.35);
      color: #c7d2fe;
      padding: 0.15rem 0.5rem;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 500;
    }

    /* Abstract details */
    .abstract-text {
      font-size: 0.88rem;
      color: #94a3b8;
      line-height: 1.6;
      margin: 0.75rem 0 1.25rem;
    }

    /* Agent Connection Claim Callout */
    .claim-box {
      background: rgba(255, 255, 255, 0.02);
      border-left: 3px solid #818cf8;
      border-radius: 0 8px 8px 0;
      padding: 0.85rem 1.15rem;
      margin-bottom: 1.25rem;
    }

    .claim-tag {
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #818cf8;
      font-weight: 700;
      margin-bottom: 0.3rem;
      display: flex;
      align-items: center;
      gap: 0.35rem;
    }

    .claim-quote {
      font-size: 0.92rem;
      font-style: italic;
      color: #f1f5f9;
      line-height: 1.45;
    }

    /* Verification Status Banner */
    .verification-block {
      border-radius: 10px;
      padding: 0.95rem 1.2rem;
      margin-bottom: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 0.45rem;
    }

    .verification-block.verified {
      background: var(--verified-bg);
      border: 1px solid var(--verified-border);
    }

    .verification-block.rejected {
      background: var(--rejected-bg);
      border: 1px solid var(--rejected-border);
    }

    .v-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 0.5rem;
    }

    .v-badge {
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      font-size: 0.78rem;
      font-weight: 800;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      padding: 0.25rem 0.65rem;
      border-radius: 6px;
    }

    .verified .v-badge {
      background: rgba(16, 185, 129, 0.2);
      color: var(--verified-color);
      border: 1px solid rgba(16, 185, 129, 0.4);
    }

    .rejected .v-badge {
      background: rgba(244, 63, 94, 0.2);
      color: var(--rejected-color);
      border: 1px solid rgba(244, 63, 94, 0.4);
    }

    .v-score-meta {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.8rem;
      color: var(--text-muted);
    }

    .v-excerpt {
      font-size: 0.84rem;
      color: #cbd5e1;
      background: rgba(0, 0, 0, 0.25);
      padding: 0.55rem 0.8rem;
      border-radius: 6px;
      line-height: 1.5;
    }

    .v-source-file {
      font-family: 'JetBrains Mono', monospace;
      color: #34d399;
      font-size: 0.78rem;
    }

    .v-rejection-note {
      font-size: 0.8rem;
      color: #fda4af;
      line-height: 1.4;
    }

    /* Actions Bar */
    .card-actions {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-top: 1rem;
      border-top: 1px solid rgba(255, 255, 255, 0.06);
    }

    .decision-status {
      font-size: 0.82rem;
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 0.45rem;
    }

    .status-badge-approved {
      color: var(--verified-color);
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.35);
      padding: 0.3rem 0.7rem;
      border-radius: 9999px;
    }

    .status-badge-skipped {
      color: #94a3b8;
      background: rgba(148, 163, 184, 0.12);
      border: 1px solid rgba(148, 163, 184, 0.25);
      padding: 0.3rem 0.7rem;
      border-radius: 9999px;
    }

    .status-badge-pending {
      color: #fbbf24;
      background: rgba(251, 191, 36, 0.1);
      border: 1px solid rgba(251, 191, 36, 0.25);
      padding: 0.3rem 0.7rem;
      border-radius: 9999px;
    }

    .btn-group {
      display: flex;
      gap: 0.65rem;
    }

    .btn {
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
      font-family: inherit;
      font-size: 0.85rem;
      font-weight: 700;
      padding: 0.55rem 1.15rem;
      border-radius: 8px;
      border: none;
      cursor: pointer;
      transition: all 0.15s ease;
    }

    .btn-approve {
      background: linear-gradient(135deg, #10b981 0%, #059669 100%);
      color: #fff;
      box-shadow: 0 4px 14px rgba(16, 185, 129, 0.25);
    }

    .btn-approve:hover {
      background: linear-gradient(135deg, #34d399 0%, #10b981 100%);
      box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
      transform: translateY(-1px);
    }

    .btn-skip {
      background: rgba(255, 255, 255, 0.06);
      color: var(--text-muted);
      border: 1px solid var(--border-subtle);
    }

    .btn-skip:hover {
      background: rgba(255, 255, 255, 0.12);
      color: #fff;
      border-color: rgba(255, 255, 255, 0.2);
    }

    .btn:disabled {
      opacity: 0.4;
      cursor: not-allowed;
      transform: none !important;
    }

    /* Toast */
    #toast {
      position: fixed;
      bottom: 2rem;
      right: 2rem;
      background: rgba(15, 23, 42, 0.95);
      border: 1px solid var(--accent);
      color: #fff;
      padding: 0.9rem 1.4rem;
      border-radius: 12px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
      font-size: 0.88rem;
      display: none;
      align-items: center;
      gap: 0.6rem;
      z-index: 1000;
      animation: slideUp 0.2s ease-out;
    }

    @keyframes slideUp {
      from { transform: translateY(20px); opacity: 0; }
      to { transform: translateY(0); opacity: 1; }
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="header-top">
        <div class="brand">
          <h1>Marginalia <span>• Review</span></h1>
          <p>Curated arXiv digest candidates evaluated through deterministic scoring and vector verification.</p>
        </div>
        <div class="qdrant-pill">
          <span class="qdrant-dot"></span>
          Qdrant Collection: marginalia_notes ({{ qdrant_points }} points)
        </div>
      </div>

      <!-- Stats Grid -->
      <div class="stats-grid">
        <div class="stat-card">
          <span class="stat-label">Total Surfaced</span>
          <span class="stat-value" id="stat-total">{{ stats.total }}</span>
        </div>
        <div class="stat-card">
          <span class="stat-label">Verified Claims</span>
          <span class="stat-value verified" id="stat-verified">{{ stats.verified }}</span>
        </div>
        <div class="stat-card">
          <span class="stat-label">Rejected Claims</span>
          <span class="stat-value rejected" id="stat-rejected">{{ stats.rejected }}</span>
        </div>
        <div class="stat-card">
          <span class="stat-label">Approved to Corpus</span>
          <span class="stat-value approved" id="stat-approved">{{ stats.approved }}</span>
        </div>
        <div class="stat-card">
          <span class="stat-label">Skipped</span>
          <span class="stat-value" id="stat-skipped">{{ stats.skipped }}</span>
        </div>
      </div>

      <!-- Filter and Search Controls -->
      <div class="controls-bar">
        <div class="filter-tabs">
          <button class="tab-btn active" data-filter="all">All ({{ stats.total }})</button>
          <button class="tab-btn" data-filter="pending">Pending ({{ stats.pending }})</button>
          <button class="tab-btn" data-filter="verified">Verified ({{ stats.verified }})</button>
          <button class="tab-btn" data-filter="rejected">Rejected ({{ stats.rejected }})</button>
          <button class="tab-btn" data-filter="approved">Approved ({{ stats.approved }})</button>
          <button class="tab-btn" data-filter="skipped">Skipped ({{ stats.skipped }})</button>
        </div>
        <div class="search-box">
          <input type="text" id="searchInput" placeholder="Filter by title, author, keyword...">
        </div>
      </div>
    </header>

    <!-- Paper Cards List -->
    <main class="paper-list">
      {% for paper in papers %}
      {% set aid = paper.arxiv_id %}
      {% set dec = decisions.get(aid) %}
      {% set is_verified = paper.claim_verified and paper.claim_verified.verified %}
      <article 
        class="paper-card {% if dec %}decided-{{ dec.decision }}{% endif %}" 
        id="card-{{ aid | replace('.', '-') }}"
        data-aid="{{ aid }}"
        data-verified="{{ 'true' if is_verified else 'false' }}"
        data-decision="{{ dec.decision if dec else 'pending' }}"
      >
        <div class="paper-header">
          <div class="title-area">
            <h2>
              <a href="{{ paper.pdf_url }}" target="_blank" rel="noopener noreferrer">
                {{ paper.title }}
              </a>
            </h2>
            <div class="paper-meta">
              <span class="arxiv-badge">{{ aid }}</span>
              <span>Published {{ paper.score.days_since_published }}d ago ({{ paper.published_date[:10] }})</span>
              <span>•</span>
              <span>{{ paper.authors | join(', ') }}</span>
            </div>
          </div>
          <div class="score-badge" title="Deterministic Score: 70% Keyword, 30% Recency">
            <span class="score-val">{{ paper.score.total_score }}</span>
            <span class="score-label">Total Score</span>
          </div>
        </div>

        <!-- Score Breakdown Metric Grid -->
        <div class="score-breakdown-row">
          <div class="metric-item">
            <span class="metric-name">Keyword Match:</span>
            <span class="metric-val">{{ paper.score.keyword_score }}</span>
          </div>
          <div class="metric-item">
            <span class="metric-name">Recency Decay:</span>
            <span class="metric-val">{{ paper.score.recency_score }}</span>
          </div>
          <div class="metric-item" style="margin-left: auto; display: flex; gap: 0.35rem; align-items: center;">
            <span class="metric-name">Matched:</span>
            {% for kw in paper.score.matched_keywords %}
              <span class="keyword-pill">{{ kw }}</span>
            {% else %}
              <span style="color: var(--text-dim);">None</span>
            {% endfor %}
          </div>
        </div>

        <!-- Abstract -->
        <p class="abstract-text">
          {{ paper.abstract }}
        </p>

        <!-- Agent Claim Callout -->
        <div class="claim-box">
          <div class="claim-tag">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/></svg>
            Agent Connection Claim
          </div>
          <p class="claim-quote">"{{ paper.claim }}"</p>
        </div>

        <!-- Distinct Verification Treatment: Honest Display -->
        {% if is_verified %}
        <div class="verification-block verified">
          <div class="v-header">
            <span class="v-badge">✓ Verified by Corpus</span>
            <span class="v-score-meta">Cosine Similarity: {{ paper.claim_verified.best_match_score }} (Threshold: 0.50)</span>
          </div>
          <div class="v-excerpt">
            <div style="font-size: 0.72rem; text-transform: uppercase; color: #34d399; margin-bottom: 0.2rem;">
              Grounding Note: <span class="v-source-file">{{ paper.claim_verified.matched_source }}</span>
            </div>
            "{{ paper.claim_verified.supporting_excerpt }}"
          </div>
        </div>
        {% else %}
        <div class="verification-block rejected">
          <div class="v-header">
            <span class="v-badge">✕ Unverified Claim (Rejected)</span>
            <span class="v-score-meta">Cosine Similarity: {{ paper.claim_verified.best_match_score if paper.claim_verified else '0.0000' }} (Threshold: 0.50)</span>
          </div>
          <p class="v-rejection-note">
            The model asserted a link to our themes, but vector similarity against existing notes fell below the 0.50 threshold. 
            Kept fully visible for auditability.
          </p>
        </div>
        {% endif %}

        <!-- Decision Footer -->
        <div class="card-actions">
          <div class="decision-status" id="status-text-{{ aid | replace('.', '-') }}">
            {% if dec and dec.decision == 'approve' %}
              <span class="status-badge-approved">✓ Approved into Corpus</span>
            {% elif dec and dec.decision == 'skip' %}
              <span class="status-badge-skipped">↷ Skipped</span>
            {% else %}
              <span class="status-badge-pending">⏳ Pending Review</span>
            {% endif %}
          </div>
          <div class="btn-group">
            <button 
              type="button" 
              class="btn btn-approve" 
              onclick="submitDecision('{{ aid }}', 'approve')"
              title="Approve paper, save to corpus/, and index into Qdrant"
            >
              Approve
            </button>
            <button 
              type="button" 
              class="btn btn-skip" 
              onclick="submitDecision('{{ aid }}', 'skip')"
              title="Skip paper without adding to corpus"
            >
              Skip
            </button>
          </div>
        </div>
      </article>
      {% endfor %}
    </main>
  </div>

  <div id="toast"></div>

  <script>
    // Tab filtering
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const filter = btn.dataset.filter;
        applyFilters(filter, document.getElementById('searchInput').value.toLowerCase());
      });
    });

    // Search input filtering
    document.getElementById('searchInput').addEventListener('input', (e) => {
      const activeFilter = document.querySelector('.tab-btn.active').dataset.filter;
      applyFilters(activeFilter, e.target.value.toLowerCase());
    });

    function applyFilters(filter, query) {
      document.querySelectorAll('.paper-card').forEach(card => {
        const isVerified = card.dataset.verified === 'true';
        const decision = card.dataset.decision; // 'approve', 'skip', or 'pending'
        const textContent = card.innerText.toLowerCase();

        let matchesTab = true;
        if (filter === 'verified') matchesTab = isVerified;
        else if (filter === 'rejected') matchesTab = !isVerified;
        else if (filter === 'approved') matchesTab = (decision === 'approve');
        else if (filter === 'skipped') matchesTab = (decision === 'skip');
        else if (filter === 'pending') matchesTab = (decision === 'pending');

        const matchesQuery = !query || textContent.includes(query);

        if (matchesTab && matchesQuery) {
          card.style.display = 'block';
        } else {
          card.style.display = 'none';
        }
      });
    }

    // Submit review decision via AJAX
    async function submitDecision(arxivId, decision) {
      const cardId = 'card-' + arxivId.replace(/[.]/g, '-');
      const card = document.getElementById(cardId);
      const statusEl = document.getElementById('status-text-' + arxivId.replace(/[.]/g, '-'));

      if (!card) {
        console.error('submitDecision: card not found for', arxivId);
        return;
      }

      // Disable buttons to indicate progress
      const btns = card.querySelectorAll('.btn');
      btns.forEach(b => { b.disabled = true; });

      try {
        const response = await fetch('/decide', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ arxiv_id: arxivId, decision: decision })
        });
        const result = await response.json();

        if (result.status === 'success') {
          // Update card attributes and styling
          card.dataset.decision = decision;
          card.classList.remove('decided-approve', 'decided-skip');
          card.classList.add('decided-' + decision);

          if (statusEl) {
            if (decision === 'approve') {
              statusEl.innerHTML = '<span class="status-badge-approved">✓ Approved into Corpus</span>';
            } else {
              statusEl.innerHTML = '<span class="status-badge-skipped">↷ Skipped</span>';
            }
          }

          if (decision === 'approve') {
            showToast('✓ Paper approved! Saved to corpus/ and indexed into Qdrant.');
          } else {
            showToast('↷ Paper marked as skipped in decisions.json.');
          }

          // Refresh live stats
          updateStatCounters();
        } else {
          showToast('Error recording decision: ' + (result.message || 'Unknown'));
          btns.forEach(b => { b.disabled = false; });
        }
      } catch (err) {
        showToast('Network error while saving decision: ' + err.message);
        btns.forEach(b => { b.disabled = false; });
      }
    }

    function updateStatCounters() {
      let approved = 0, skipped = 0, pending = 0;
      document.querySelectorAll('.paper-card').forEach(card => {
        const d = card.dataset.decision;
        if (d === 'approve') approved++;
        else if (d === 'skip') skipped++;
        else pending++;
      });
      const elApp = document.getElementById('stat-approved');
      const elSkip = document.getElementById('stat-skipped');
      if (elApp) elApp.innerText = approved;
      if (elSkip) elSkip.innerText = skipped;
    }

    function showToast(msg) {
      const toast = document.getElementById('toast');
      toast.innerText = msg;
      toast.style.display = 'flex';
      setTimeout(() => { toast.style.display = 'none'; }, 3500);
    }
  </script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    """Render the dashboard page with all digest papers and existing decisions."""
    papers = load_digest()
    decisions = load_decisions()

    verified_count = sum(
        1
        for p in papers
        if p.get("claim_verified") and p.get("claim_verified", {}).get("verified")
    )
    rejected_count = len(papers) - verified_count
    approved_count = sum(1 for d in decisions.values() if d.get("decision") == "approve")
    skipped_count = sum(1 for d in decisions.values() if d.get("decision") == "skip")
    pending_count = len(papers) - (approved_count + skipped_count)

    stats = {
        "total": len(papers),
        "verified": verified_count,
        "rejected": rejected_count,
        "approved": approved_count,
        "skipped": skipped_count,
        "pending": max(0, pending_count),
    }

    qdrant_points = 6
    if qdrant_client:
        try:
            info = qdrant_client.get_collection(COLLECTION_NAME)
            qdrant_points = info.points_count
        except Exception:
            pass

    return render_template_string(
        DASHBOARD_HTML,
        papers=papers,
        decisions=decisions,
        stats=stats,
        qdrant_points=qdrant_points,
    )


@app.route("/decide", methods=["POST"])
def decide():
    """Record a review decision for a paper and expand corpus/Qdrant on approval."""
    data = request.get_json(silent=True) or request.form
    arxiv_id = data.get("arxiv_id", "").strip()
    decision = data.get("decision", "").strip().lower()

    if not arxiv_id or decision not in ("approve", "skip"):
        return jsonify({"status": "error", "message": "Invalid arxiv_id or decision"}), 400

    # 1. Save to decisions.json
    record = save_decision_record(arxiv_id, decision)

    indexed_file = None
    if decision == "approve":
        # 2. Find paper in digest.json to write to corpus/ and index to Qdrant
        papers = load_digest()
        target_paper = next((p for p in papers if p.get("arxiv_id") == arxiv_id), None)
        if not target_paper:
            # Try without version suffix
            target_paper = next(
                (p for p in papers if p.get("arxiv_id", "").split("v")[0] == arxiv_id.split("v")[0]),
                None,
            )

        if target_paper:
            indexed_file = index_approved_paper(target_paper)
        else:
            logger.warning("Paper %s not found in digest.json to create corpus file", arxiv_id)

    # If it was a form post, redirect back to /
    if not request.is_json:
        return redirect(url_for("index"))

    return jsonify(
        {
            "status": "success",
            "decision": decision,
            "arxiv_id": arxiv_id,
            "timestamp": record.get("timestamp"),
            "corpus_file": indexed_file.name if indexed_file else None,
        }
    )


import socket


def find_available_port(preferred_port: int = 5000) -> int:
    """Find the preferred port or fall back to the next available port."""
    for p in (preferred_port, 5001, 5002, 8080):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", p))
                return p
            except OSError:
                continue
    return preferred_port


if __name__ == "__main__":
    preferred = int(os.getenv("PORT", 5000))
    port = find_available_port(preferred)
    if port != preferred:
        print(
            f"\n[Notice] Port {preferred} is in use (macOS AirPlay Receiver). "
            f"Starting Review Dashboard on http://localhost:{port} (http://127.0.0.1:{port})\n"
        )
    else:
        print(f"\nStarting Review Dashboard on http://localhost:{port}\n")

    app.run(host="0.0.0.0", port=port, debug=False)
