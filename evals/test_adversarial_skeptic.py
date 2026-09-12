"""Synthetic evaluation benchmark demonstrating all three verification tiers.

This eval harness proves the multi-tier adversarial verification logic deterministically:
1. strongly_verified: High vector match (>= 0.60) + Skeptic Agent returns 'NO OBJECTION'.
2. verified_flagged: Passes base vector gate (>= 0.50) + Skeptic Agent catches overstatement.
3. rejected: Fails base vector gate (< 0.50) + Skeptic call is bypassed.

This runs independently of live pipeline output (digest.json is NEVER touched).
"""

import os
from pathlib import Path
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Ensure parent directory is in path for module imports
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from verification import skeptic_review, verify_claim

SYNTHETIC_EVAL_CASES = [
    {
        "name": "Tightly Grounded Tool Contract (Target: strongly_verified)",
        "expected_tier": "strongly_verified",
        "claim": "This paper emphasizes that a tool docstring serves as an interface contract that dictates function calling reliability, echoing lessons from tool-use design.",
    },
    {
        "name": "Overreaching Multi-Agent Claim (Target: verified_flagged)",
        "expected_tier": "verified_flagged",
        "claim": "This paper completely solves multi-agent coordination by proving that hierarchical orchestration eliminates all agent errors and hallucinations.",
    },
    {
        "name": "Off-Topic Astrophysics Decoy (Target: rejected)",
        "expected_tier": "rejected",
        "claim": "This paper measures molecular absorption bands and atmospheric transmission in exoplanets.",
    },
]


def run_skeptic_eval():
    load_dotenv(BASE_DIR / ".env")
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not url or not api_key or not groq_api_key:
        print("Error: Missing credentials in .env")
        sys.exit(1)

    client = QdrantClient(url=url, api_key=api_key)

    print("\n" + "=" * 80)
    print("MARGINALIA ADVERSARIAL SKEPTIC TIER EVALUATION (SYNTHETIC HARNESS)")
    print("=" * 80)

    results = []

    for idx, case in enumerate(SYNTHETIC_EVAL_CASES, 1):
        print(f"\n--- Case {idx}: {case['name']} ---")
        claim = case["claim"]
        print(f"Claim: \"{claim}\"")

        # 1. Base semantic vector verification
        ver = verify_claim(claim, client, collection_name="marginalia_notes", threshold=0.5)
        score = ver["best_match_score"]
        is_verified = ver["verified"]
        matched_source = ver["matched_source"]
        print(f"Base Gate: {'PASS' if is_verified else 'FAIL'} | Vector Score: {score:.4f} | Source: {matched_source}")

        # 2. Adversarial Skeptic Tier
        if is_verified:
            skeptic = skeptic_review(claim, ver["supporting_excerpt"] or "", groq_api_key=groq_api_key)
            has_objection = skeptic["has_objection"]
            objection = skeptic["objection_text"]

            if score >= 0.6 and not has_objection:
                tier = "strongly_verified"
            elif score >= 0.5:
                tier = "verified_flagged"
            else:
                tier = "rejected"
        else:
            skeptic = {"has_objection": False, "objection_text": None}
            has_objection = False
            objection = None
            tier = "rejected"
            print("Skeptic Call: BYPASSED (base vector gate failed)")

        print(f"Assigned Tier: {tier.upper()} (Expected: {case['expected_tier'].upper()})")
        if objection:
            print(f"Skeptic Objection: \"{objection}\"")
        elif tier == "strongly_verified":
            print("Skeptic Output: NO OBJECTION (Claim fully substantiated by corpus note)")

        assert tier == case["expected_tier"], f"Tier mismatch for case {case['name']}: got {tier}, expected {case['expected_tier']}"
        results.append((case["name"], tier, score, objection))

    print("\n" + "=" * 80)
    print("SUMMARY OF ADVERSARIAL TIER EVALUATION:")
    for name, tier, score, obj in results:
        status_icon = "🟢" if tier == "strongly_verified" else ("🟡" if tier == "verified_flagged" else "🔴")
        print(f"  {status_icon} [{tier.upper():<18}] Score: {score:.4f} | {name}")
    print("=" * 80)
    print("ALL 3 TIERS DETERMINISTICALLY DEMONSTRATED IN EVAL HARNESS!\n")


if __name__ == "__main__":
    run_skeptic_eval()
