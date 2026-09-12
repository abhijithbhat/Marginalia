"""Verification layer for Marginalia.

Grounds generated research claims against the researcher's own prior notes
stored in Qdrant Cloud using semantic similarity embeddings.

This is NOT an agent tool — it is a deterministic verification gate called
directly by orchestration code.
"""

import hashlib
import os
from pathlib import Path
import re
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from strands import Agent
from strands.models.openai import OpenAIModel

# Load embedding model once at module import time
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def build_corpus_index(
    qdrant_url: str,
    qdrant_api_key: str,
    collection_name: str = "marginalia_notes",
) -> None:
    """Index prior research notes from the corpus/ folder into Qdrant Cloud.

    Creates the collection if it does not already exist (vector size 384, Cosine distance)
    and upserts embeddings with source metadata and text content.

    Args:
        qdrant_url: The Qdrant Cloud cluster endpoint URL.
        qdrant_api_key: The API key for the Qdrant Cloud cluster.
        collection_name: Target collection name (defaults to "marginalia_notes").
    """
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    # Create collection if it doesn't already exist
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    # Locate corpus directory
    corpus_dir = Path(__file__).resolve().parent / "corpus"
    if not corpus_dir.exists():
        corpus_dir = Path("corpus")

    note_files = sorted(list(corpus_dir.glob("*.md")))
    points: list[PointStruct] = []

    for file_path in note_files:
        content = file_path.read_text(encoding="utf-8").strip()
        vector = embedding_model.encode(content).tolist()
        source_filename = file_path.name
        point_id = int(hashlib.sha256(source_filename.encode()).hexdigest()[:16], 16)
        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "source": source_filename,
                    "text": content,
                },
            )
        )

    if points:
        client.upsert(collection_name=collection_name, points=points)


def verify_claim(
    claim_text: str,
    qdrant_client: QdrantClient,
    collection_name: str = "marginalia_notes",
    threshold: float = 0.5,
) -> dict:
    """Verify whether a candidate claim is supported by the researcher's indexed notes.

    Performs semantic cosine similarity search between the claim and the corpus notes.
    No generative model calls are made.

    Args:
        claim_text: The sentence or claim to verify against the corpus.
        qdrant_client: Connected QdrantClient instance.
        collection_name: Qdrant collection containing corpus notes (default: "marginalia_notes").
        threshold: Minimum cosine similarity required to verify (default: 0.5).

    Returns:
        dict: Verification breakdown containing:
            - "verified" (bool): True if best_match_score >= threshold, else False.
            - "best_match_score" (float): Highest cosine similarity score.
            - "supporting_excerpt" (str or None): Excerpt text if verified, else None.
            - "matched_source" (str or None): Filename of matching note if verified, else None.
    """
    query_vector = embedding_model.encode(claim_text).tolist()

    search_result = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=1,
        with_payload=True,
    )

    if not search_result.points:
        return {
            "verified": False,
            "best_match_score": 0.0,
            "supporting_excerpt": None,
            "matched_source": None,
        }

    best_point = search_result.points[0]
    best_score = float(best_point.score)
    payload = best_point.payload or {}
    is_verified = best_score >= threshold

    return {
        "verified": is_verified,
        "best_match_score": round(best_score, 4),
        "supporting_excerpt": payload.get("text") if is_verified else None,
        "matched_source": payload.get("source") if is_verified else None,
    }


def skeptic_review(
    claim_text: str,
    supporting_excerpt: str,
    groq_api_key: str | None = None,
) -> dict:
    """Evaluate a claimed connection against its supporting excerpt using an adversarial Skeptic Agent.

    Args:
        claim_text: The claimed connection.
        supporting_excerpt: The ground-truth excerpt from the researcher's note corpus.
        groq_api_key: Optional Groq API key (defaults to GROQ_API_KEY environment variable).

    Returns:
        dict: {"has_objection": bool, "objection_text": str or None}
    """
    api_key = groq_api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        return {"has_objection": False, "objection_text": None}

    model = OpenAIModel(
        client_args={
            "api_key": api_key,
            "base_url": "https://api.groq.com/openai/v1",
        },
        model_id="openai/gpt-oss-120b",
        params={"max_tokens": 1024},
    )

    system_prompt = (
        "You are a skeptical reviewer. Given a claimed connection and the specific excerpt "
        "it's supposedly grounded in, find the single strongest reason the claim overstates "
        "or misreads that connection. If there genuinely isn't a good objection, say exactly "
        "'NO OBJECTION' and nothing else. Otherwise, state the objection in one sentence — "
        "nothing else, no preamble."
    )

    agent = Agent(
        model=model,
        system_prompt=system_prompt,
    )

    prompt = f"Claim: {claim_text}\n\nSupporting excerpt: {supporting_excerpt}"

    try:
        response = agent(prompt)
        raw_text = str(response).strip()
    except Exception:
        return {"has_objection": False, "objection_text": None}

    # Strip thinking tags if generated
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    cleaned = cleaned.strip('"\'').strip()

    norm = cleaned.rstrip(".").strip().upper()
    if norm == "NO OBJECTION":
        return {"has_objection": False, "objection_text": None}
    else:
        return {"has_objection": True, "objection_text": cleaned}


if __name__ == "__main__":
    import os
    import pprint

    # Load environment variables
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path)

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url or not qdrant_api_key:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in .env")

    print(f"Connecting to Qdrant Cloud at: {qdrant_url[:35]}...")
    build_corpus_index(qdrant_url, qdrant_api_key, collection_name="marginalia_notes")
    print("Corpus index built and synced successfully.\n")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    test_claims = [
        "This paper's approach relies on precise tool docstrings to reduce incorrect function calls, echoing lessons from tool-use design.",
        "This paper's verification step ensures generated claims are grounded in retrieved evidence, similar to the self-verifying RAG lesson from RegBrain.",
        "This paper proves P=NP using a novel quantum error-correction scheme.",
    ]

    print("=" * 80)
    print("VERIFYING TEST CLAIMS (Threshold: 0.5)")
    print("=" * 80)

    for idx, claim in enumerate(test_claims, 1):
        result = verify_claim(claim, client, collection_name="marginalia_notes", threshold=0.5)
        status = "VERIFIED (PASS)" if result["verified"] else "REJECTED (FAIL)"
        print(f"\nClaim {idx}: \"{claim}\"")
        print(f"  Status: {status}")
        print(f"  Best Match Score: {result['best_match_score']}")
        print(f"  Matched Source: {result['matched_source']}")
        if result["supporting_excerpt"]:
            print(f"  Supporting Excerpt: \"{result['supporting_excerpt'][:120]}...\"")
        else:
            print("  Supporting Excerpt: None")

    print("\n" + "=" * 80)
