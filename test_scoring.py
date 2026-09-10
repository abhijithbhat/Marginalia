"""Unit tests for deterministic scoring and word-boundary keyword matching."""

import unittest
from scoring import score_relevance


class TestScoringKeywordMatching(unittest.TestCase):
    def setUp(self):
        self.profile = {
            "keywords": [
                "multi-agent systems",
                "tool use",
                "function calling",
                "language model agents",
                "LLM agents",
                "LLM-based agents",
                "retrieval augmented generation",
                "RAG",
                "agent orchestration",
                "model context protocol",
                "MCP",
            ],
            "recency_window_days": 30,
        }

    def test_word_boundary_avoids_false_positives(self):
        """Confirm that words containing 'rag' as a substring (paragraph, storage, coverage)

        do NOT trigger a match for the keyword 'RAG'.
        """
        paper = {
            "title": "Storage Optimization for Large Paragraph Processing",
            "abstract": (
                "In this work, we address storage and coverage in document pipelines. "
                "Each paragraph is indexed without external knowledge."
            ),
            "published_date": "2026-09-01T00:00:00Z",
        }
        result = score_relevance(paper, self.profile)
        self.assertEqual(result["keyword_score"], 0.0)
        self.assertEqual(result["matched_keywords"], [])
        self.assertNotIn("RAG", result["matched_keywords"])

    def test_rag_positive_matches(self):
        """Confirm that 'RAG' and its plural variant 'RAGs' match correctly."""
        # Title match
        paper_title = {
            "title": "Benchmarking RAG for Code Generation",
            "abstract": "We explore retrieval systems.",
            "published_date": "2026-09-01T00:00:00Z",
        }
        res_title = score_relevance(paper_title, self.profile)
        self.assertIn("RAG", res_title["matched_keywords"])
        self.assertGreater(res_title["keyword_score"], 0.0)

        # Abstract match with plural
        paper_plural = {
            "title": "Document QA Systems",
            "abstract": "We compare various RAGs and their retriever components.",
            "published_date": "2026-09-01T00:00:00Z",
        }
        res_plural = score_relevance(paper_plural, self.profile)
        self.assertIn("RAG", res_plural["matched_keywords"])
        self.assertGreater(res_plural["keyword_score"], 0.0)

    def test_singular_plural_variants(self):
        """Confirm singular and plural variants match seamlessly across terms."""
        # Keyword in profile is 'LLM agents' -> text has singular 'LLM agent'
        paper = {
            "title": "Evaluating a Single LLM Agent in Complex Environments",
            "abstract": "Our study investigates how an autonomous agent performs.",
            "published_date": "2026-09-01T00:00:00Z",
        }
        result = score_relevance(paper, self.profile)
        self.assertIn("LLM agents", result["matched_keywords"])
        self.assertGreater(result["keyword_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
