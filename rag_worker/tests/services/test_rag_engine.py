"""
Unit tests for RAGEngine.
"""
import pytest
import json
from unittest.mock import MagicMock, patch
from llama_index.core.base.response.schema import Response
from worker.services.rag_engine import RAGEngine


class TestRAGEngine:
    """Tests for RAGEngine."""

    @patch('worker.services.rag_engine.RouterService')
    def test_init(self, mock_router_service):
        """Test RAGEngine initialization."""
        mock_router = MagicMock()
        mock_router_service.return_value = mock_router

        engine = RAGEngine()

        assert engine.router == mock_router
        mock_router_service.assert_called_once()

    @patch('worker.services.rag_engine.RouterService')
    def test_run_once(self, mock_router_service):
        """Test run_once method."""
        mock_router = MagicMock()
        mock_response = Response(response="Test response")
        mock_router.route.return_value = mock_response
        mock_router_service.return_value = mock_router

        engine = RAGEngine()
        result = engine.run_once("test query", "code")

        assert result == mock_response
        mock_router.route.assert_called_once_with("test query", force_level="code")

    @patch('worker.services.rag_engine.RouterService')
    @patch('worker.services.rag_engine.Settings')
    def test_propose_refinement_valid_json(self, mock_settings, mock_router_service):
        """Test propose_refinement with valid JSON response."""
        mock_router = MagicMock()
        mock_router_service.return_value = mock_router

        mock_llm = MagicMock()
        mock_completion = MagicMock()
        mock_completion.text = '{"query": "refined query", "force_level": "package"}'
        mock_llm.complete.return_value = mock_completion
        mock_settings.llm = mock_llm

        engine = RAGEngine()
        result = engine.propose_refinement("original query")

        assert result == {"query": "refined query", "force_level": "package"}
        mock_llm.complete.assert_called_once()

    @patch('worker.services.rag_engine.RouterService')
    @patch('worker.services.rag_engine.Settings')
    def test_propose_refinement_json_with_regex(self, mock_settings, mock_router_service):
        """Test propose_refinement with JSON embedded in text."""
        mock_router = MagicMock()
        mock_router_service.return_value = mock_router

        mock_llm = MagicMock()
        mock_completion = MagicMock()
        mock_completion.text = 'Here is the JSON: {"query": "extracted query", "force_level": "project"} and more text'
        mock_llm.complete.return_value = mock_completion
        mock_settings.llm = mock_llm

        engine = RAGEngine()
        result = engine.propose_refinement("original query")

        assert result == {"query": "extracted query", "force_level": "project"}

    @patch('worker.services.rag_engine.RouterService')
    @patch('worker.services.rag_engine.Settings')
    def test_propose_refinement_invalid_force_level(self, mock_settings, mock_router_service):
        """Test propose_refinement with invalid force_level."""
        mock_router = MagicMock()
        mock_router_service.return_value = mock_router

        mock_llm = MagicMock()
        mock_completion = MagicMock()
        mock_completion.text = '{"query": "refined query", "force_level": "invalid_level"}'
        mock_llm.complete.return_value = mock_completion
        mock_settings.llm = mock_llm

        engine = RAGEngine()
        result = engine.propose_refinement("original query")

        # Invalid force_level should be set to None
        assert result == {"query": "refined query", "force_level": None}

    @patch('worker.services.rag_engine.RouterService')
    @patch('worker.services.rag_engine.Settings')
    def test_propose_refinement_missing_query(self, mock_settings, mock_router_service):
        """Test propose_refinement with missing query field."""
        mock_router = MagicMock()
        mock_router_service.return_value = mock_router

        mock_llm = MagicMock()
        mock_completion = MagicMock()
        mock_completion.text = '{"force_level": "code"}'
        mock_llm.complete.return_value = mock_completion
        mock_settings.llm = mock_llm

        engine = RAGEngine()
        result = engine.propose_refinement("original query")

        # Missing query should fall back to original
        assert result == {"query": "original query", "force_level": "code"}

    @patch('worker.services.rag_engine.RouterService')
    @patch('worker.services.rag_engine.Settings')
    def test_propose_refinement_valid_force_levels(self, mock_settings, mock_router_service):
        """Test propose_refinement with all valid force_level values."""
        mock_router = MagicMock()
        mock_router_service.return_value = mock_router

        mock_llm = MagicMock()
        mock_settings.llm = mock_llm

        engine = RAGEngine()

        valid_levels = ["code", "package", "project", "none", None]

        for level in valid_levels:
            mock_completion = MagicMock()
            mock_completion.text = json.dumps({"query": "test", "force_level": level})
            mock_llm.complete.return_value = mock_completion

            result = engine.propose_refinement("test query")

            assert result["force_level"] == level

    @patch('worker.services.rag_engine.RouterService')
    @patch('worker.services.rag_engine.Settings')
    def test_propose_refinement_json_parse_error(self, mock_settings, mock_router_service):
        """Test propose_refinement with JSON parse error."""
        mock_router = MagicMock()
        mock_router_service.return_value = mock_router

        mock_llm = MagicMock()
        mock_completion = MagicMock()
        mock_completion.text = "This is not JSON at all"
        mock_llm.complete.return_value = mock_completion
        mock_settings.llm = mock_llm

        engine = RAGEngine()
        result = engine.propose_refinement("original query")

        # Should fall back to original query with no force_level
        assert result == {"query": "original query", "force_level": None}

    @patch('worker.services.rag_engine.RouterService')
    @patch('worker.services.rag_engine.Settings')
    def test_propose_refinement_prompt_construction(self, mock_settings, mock_router_service):
        """Test that propose_refinement constructs the prompt correctly."""
        mock_router = MagicMock()
        mock_router_service.return_value = mock_router

        mock_llm = MagicMock()
        mock_completion = MagicMock()
        mock_completion.text = '{"query": "test", "force_level": null}'
        mock_llm.complete.return_value = mock_completion
        mock_settings.llm = mock_llm

        engine = RAGEngine()
        engine.propose_refinement("test question")

        # Verify the prompt contains expected elements
        call_args = mock_llm.complete.call_args[0][0]
        assert "Improve retrieval for a codebase" in call_args
        assert "code|package|project|none" in call_args
        assert "test question" in call_args
        assert "JSON" in call_args
