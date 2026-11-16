"""Tests for environment configuration handling."""

import os
import unittest
from unittest.mock import patch

from env_utils import get_env_value


class TestGetEnvValue(unittest.TestCase):
    def test_strips_surrounding_whitespace(self):
        with patch.dict(os.environ, {"QDRANT_URL": " https://example.com\n"}):
            self.assertEqual(get_env_value("QDRANT_URL"), "https://example.com")

    def test_returns_none_for_missing_value(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(get_env_value("QDRANT_URL"))


if __name__ == "__main__":
    unittest.main()
