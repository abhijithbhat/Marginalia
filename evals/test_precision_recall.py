"""Synthetic benchmark evaluating Precision, Recall, and Semantic Verification gates."""

import os
from pathlib import Path
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Ensure parent directory is in path for module imports
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from scoring import load_profile, score_relevance
from verification import verify_claim

# 10 Synthetic Papers: 5 genuinely on-topic, 5 off-topic decoys
BENCHMARK_PAPERS = [
    # --- 5 On-Topic Papers ---
    {
        "arxiv_id": "synth_01", "is_target": True,
        "title": "Hierarchical Multi-Agent Systems for Collaborative Problem Solving",
        "abstract": "We propose a hierarchical multi-agent framework where a manager agent coordinates domain agents with shared memory.",
        "published_date": "2026-09-08T00:00:00Z",
        "claim": "Explores hierarchical multi-agent systems where a manager delegates to specialists, addressing our notes on multi-agent coordination.",
    },
    {
        "arxiv_id": "synth_02", "is_target": True,
        "title": "Self-Verifying RAG: Grounding Factuality in Document Retrieval",
        "abstract": "Retrieval augmented generation (RAG) often hallucinates. We introduce a verification pass to ground generated claims in retrieved evidence.",
        "published_date": "2026-09-07T00:00:00Z",
        "claim": "Proposes a post-retrieval verification pass checking whether claims are grounded in evidence, mirroring our RegBrain lesson on RAG.",
    },
    {
        "arxiv_id": "synth_03", "is_target": True,
        "title": "Contract-Driven Tool Use in LLM Agents",
        "abstract": "LLM agents often emit invalid arguments during function calling. We prove that rigorous tool docstrings prevent execution failure.",
        "published_date": "2026-09-06T00:00:00Z",
        "claim": "Highlights that a tool docstring is the critical interface contract for function calling, validating our takeaway on tool-use design.",
    },
    {
        "arxiv_id": "synth_04", "is_target": True,
        "title": "Standardizing Tool Interoperability with Model Context Protocol",
        "abstract": "We present a unified runtime using Model Context Protocol (MCP) allowing tools to be shared across agents without rewrites.",
        "published_date": "2026-09-05T00:00:00Z",
        "claim": "Demonstrates how the Model Context Protocol standardizes tool interfaces across different agent frameworks instead of rewriting them.",
    },
    {
        "arxiv_id": "synth_05", "is_target": True,
        "title": "Dynamic Agent Orchestration Graphs for Language Model Agents",
        "abstract": "Language model agents require dynamic orchestration graphs to manage execution flow and prevent cascading errors.",
        "published_date": "2026-09-04T00:00:00Z",
        "claim": "Investigates dynamic agent orchestration and stopping error propagation across language model agents, connecting with our notes.",
    },
    # --- 5 Off-Topic Decoys ---
    {
        "arxiv_id": "decoy_01", "is_target": False,
        "title": "Spectroscopic Analysis of Exoplanet Atmospheric Composition",
        "abstract": "High-resolution transmission spectroscopy reveals carbon monoxide and methane signatures in hot Jupiter HD 209458b.",
        "published_date": "2026-09-08T00:00:00Z",
        "claim": "Measures molecular absorption bands and atmospheric transmission in exoplanet spectra.",
    },
    {
        "arxiv_id": "decoy_02", "is_target": False,
        "title": "Fault-Tolerant Surface Codes for Superconducting Qubits",
        "abstract": "We implement surface code error correction on a 72-qubit processor, suppressing logical phase-flip errors.",
        "published_date": "2026-09-07T00:00:00Z",
        "claim": "Implements quantum surface codes to protect superconducting qubits from environmental decoherence.",
    },
    {
        "arxiv_id": "decoy_03", "is_target": False,
        "title": "Topical Retinoid Formulations for Chronic Dermatitis",
        "abstract": "A randomized trial evaluating the efficacy and cutaneous tolerance of micro-encapsulated tretinoin cream in dermatitis patients.",
        "published_date": "2026-09-06T00:00:00Z",
        "claim": "Evaluates dermatological safety profiles and clinical remission outcomes for topical retinoid treatments.",
    },
    {
        "arxiv_id": "decoy_04", "is_target": False,
        "title": "Seismic Resilience of Reinforced Concrete Bridge Columns",
        "abstract": "Shake table experiments analyze dynamic structural response and plastic hinge deformation in concrete pier columns.",
        "published_date": "2026-09-05T00:00:00Z",
        "claim": "Models nonlinear seismic shear resistance and cyclic plastic degradation in reinforced infrastructure.",
    },
    {
        "arxiv_id": "decoy_05", "is_target": False,
        "title": "CRISPR-Cas9 Editing of Phytoene Desaturase in Arabidopsis",
        "abstract": "Targeted knockout of the PDS gene leads to distinct albino phenotypes, confirming high mutagenic efficiency of dual gRNAs.",
        "published_date": "2026-09-04T00:00:00Z",
        "claim": "Uses CRISPR mutagenesis to generate albino mutants and study chlorophyll biosynthesis pathways.",
    },
]


def run_benchmark():
    load_dotenv(BASE_DIR / ".env")
    profile = load_profile(str(BASE_DIR / "profile.json"))
    qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

    tp = fp = fn = tn = 0
    on_topic_verified = 0
    decoys_caught_by_verify = 0

    print("\n" + "=" * 78)
    print("MARGINALIA SYNTHETIC BENCHMARK: SCORE GATE & VERIFICATION AUDIT")
    print("=" * 78)

    for paper in BENCHMARK_PAPERS:
        score = score_relevance(paper, profile)
        passed_score_gate = (score["total_score"] >= 0.3 and score["keyword_score"] > 0)
        verify_res = verify_claim(paper["claim"], qdrant_client)

        if paper["is_target"]:
            if passed_score_gate:
                tp += 1
            else:
                fn += 1
            if verify_res["verified"]:
                on_topic_verified += 1
        else:
            if passed_score_gate:
                fp += 1
            else:
                tn += 1
            if not verify_res["verified"]:
                decoys_caught_by_verify += 1

        label = "TARGET" if paper["is_target"] else "DECOY "
        gate_str = "PASS" if passed_score_gate else "FAIL"
        ver_str = "VERIFIED" if verify_res["verified"] else "REJECTED"
        print(f"[{label}] {paper['arxiv_id']} | Score: {score['total_score']} ({gate_str}) | Verify: {ver_str} ({verify_res['best_match_score']:.3f}) | {paper['title'][:40]}...")

    precision = (tp / (tp + fp)) * 100 if (tp + fp) else 0.0
    recall = (tp / (tp + fn)) * 100 if (tp + fn) else 0.0

    print("-" * 78)
    print(f"Deterministic Score Gate:  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"Score Gate Precision:      {precision:.1f}%")
    print(f"Score Gate Recall:         {recall:.1f}%")
    print(f"Semantic Verification Gate: {on_topic_verified}/5 on-topic verified, {decoys_caught_by_verify}/5 decoys rejected")
    print("=" * 78)
    print(f'README SENTENCE: "On a 10-paper synthetic benchmark, the pipeline achieves {precision:.1f}% precision, {recall:.1f}% recall on the score gate, with the semantic verification gate rejecting {decoys_caught_by_verify}/5 ({decoys_caught_by_verify/5*100:.0f}%) decoys even if slipped past scoring."\n')


if __name__ == "__main__":
    run_benchmark()
