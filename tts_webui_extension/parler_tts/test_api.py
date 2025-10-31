#!/usr/bin/env python3
"""
Unit tests for Parler TTS API functions.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the extension directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))

from api import get_tokenizer


class TestParlerTTSAPI(unittest.TestCase):
    """Test cases for Parler TTS API functions."""

    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_get_tokenizer(self, mock_tokenizer_from_pretrained):
        """Test get_tokenizer function."""
        mock_tokenizer = MagicMock()
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer

        result = get_tokenizer("test-model")

        mock_tokenizer_from_pretrained.assert_called_once_with("test-model", cache_dir=os.path.join("data", "models", "parler_tts", "cache"))
        self.assertEqual(result, mock_tokenizer)


if __name__ == "__main__":
    unittest.main()
