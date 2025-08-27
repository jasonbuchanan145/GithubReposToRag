"""
Unit tests for RouterService.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from llama_index.core.base.response.schema import Response

from worker.services.router_service import RouterService, DirectLLMQueryEngine


class TestDirectLLMQueryEngine:
    """Tests for DirectLLMQueryEngine."""

    @patch('worker.services.router_service.Settings')
    def test_init_default_callback_manager(self, mock_settings):
        """Test initialization with default callback manager."""
        mock_settings.callback_manager = MagicMock()

        engine = DirectLLMQueryEngine()

        assert engine.callback_manager == mock_settings.callback_manager

    @patch('worker.services.router_service.Settings')
    def test_init_custom_callback_manager(self, mock_settings):
        """Test initialization with custom callback manager."""
        custom_manager = MagicMock()

        engine = DirectLLMQueryEngine(callback_manager=custom_manager)

        assert engine.callback_manager == custom_manager

    def test_get_prompt_modules(self):
        """Test _get_prompt_modules returns empty dict."""
        engine = DirectLLMQueryEngine(callback_manager=MagicMock())

        result = engine._get_prompt_modules()

        assert result == {}

    @patch('worker.services.router_service.Settings')
    def test_query_with_string(self, mock_settings):
        """Test query method with string input."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = MagicMock(text="Test response")
        mock_settings.llm = mock_llm

        engine = DirectLLMQueryEngine()
        response = engine.query("Test query")

        assert isinstance(response, Response)
        assert response.response == "Test response"
        mock_llm.complete.assert_called_once_with("Test query")

    @patch('worker.services.router_service.Settings')
    def test_query_internal_with_query_bundle(self, mock_settings):
        """Test _query method with query bundle object."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = MagicMock(text="Bundle response")
        mock_settings.llm = mock_llm

        engine = DirectLLMQueryEngine()

        # Mock query bundle
        query_bundle = MagicMock()
        query_bundle.query_str = "Bundle query"

        response = engine._query(query_bundle)

        assert isinstance(response, Response)
        assert response.response == "Bundle response"
        mock_llm.complete.assert_called_once_with("Bundle query")

    @patch('worker.services.router_service.Settings')
    def test_query_internal_with_string(self, mock_settings):
        """Test _query method with string input."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = MagicMock(text="String response")
        mock_settings.llm = mock_llm

        engine = DirectLLMQueryEngine()
        response = engine._query("Direct string")

        assert isinstance(response, Response)
        assert response.response == "String response"
        mock_llm.complete.assert_called_once_with("Direct string")


    @pytest.mark.asyncio
    @patch('worker.services.router_service.Settings')
    async def test_aquery_with_acomplete(self, mock_settings):
        mock_llm = MagicMock()
        mock_llm.acomplete = AsyncMock(return_value=SimpleNamespace(text="Async response"))
        mock_settings.llm = mock_llm

        engine = DirectLLMQueryEngine()
        resp = await engine.aquery("Async query")

        assert isinstance(resp, Response)
        assert resp.response == "Async response"


class TestRouterService:
    """Tests for RouterService."""

    @patch('worker.services.router_service.vector_store_for_table')
    @patch('worker.services.router_service.Settings')
    @patch('worker.services.router_service.QwenLLM')
    @patch('worker.services.router_service.HuggingFaceEmbedding')
    @patch('worker.services.router_service.VectorStoreIndex')
    @patch('worker.services.router_service.VectorIndexRetriever')
    @patch('worker.services.router_service.TreeSummarize')
    @patch('worker.services.router_service.RetrieverQueryEngine')
    @patch('worker.services.router_service.LLMSingleSelector')
    @patch('worker.services.router_service.RouterQueryEngine')
    def test_init(self, mock_router_engine, mock_selector, mock_retriever_engine,
                  mock_tree_summarize, mock_retriever, mock_index,
                  mock_embed_model, mock_qwen, mock_settings, mock_vector_store):
        """Test RouterService initialization."""
        # Setup mocks
        mock_vector_store.return_value = MagicMock()
        mock_index.from_vector_store.return_value = MagicMock()
        mock_retriever.return_value = MagicMock()
        mock_tree_summarize.return_value = MagicMock()
        mock_retriever_engine.return_value = MagicMock()
        mock_selector.from_defaults.return_value = MagicMock()
        mock_router_engine.return_value = MagicMock()

        service = RouterService()

        # Verify Settings were configured
        assert mock_settings.llm is not None
        assert mock_settings.embed_model is not None

        # Verify engines were created for each level
        assert len(service._engines) == 3
        assert "code" in service._engines
        assert "package" in service._engines
        assert "project" in service._engines

        # Verify direct engine exists
        assert service._direct_engine is not None

        # Verify router was created
        assert service._router is not None

    @patch('worker.services.router_service.vector_store_for_table')
    @patch('worker.services.router_service.Settings')
    @patch('worker.services.router_service.QwenLLM')
    @patch('worker.services.router_service.HuggingFaceEmbedding')
    @patch('worker.services.router_service.VectorStoreIndex')
    @patch('worker.services.router_service.VectorIndexRetriever')
    @patch('worker.services.router_service.TreeSummarize')
    @patch('worker.services.router_service.RetrieverQueryEngine')
    @patch('worker.services.router_service.LLMSingleSelector')
    @patch('worker.services.router_service.RouterQueryEngine')
    def test_route_without_force_level(self, mock_router_engine, mock_selector,
                                      mock_retriever_engine, mock_tree_summarize,
                                      mock_retriever, mock_index, mock_embed_model,
                                      mock_qwen, mock_settings, mock_vector_store):
        """Test routing without forced level."""
        # Setup mocks
        mock_vector_store.return_value = MagicMock()
        mock_index.from_vector_store.return_value = MagicMock()
        mock_retriever.return_value = MagicMock()
        mock_tree_summarize.return_value = MagicMock()
        mock_retriever_engine.return_value = MagicMock()
        mock_selector.from_defaults.return_value = MagicMock()

        mock_router = MagicMock()
        mock_response = Response(response="Router response")
        mock_router.query.return_value = mock_response
        mock_router_engine.return_value = mock_router

        service = RouterService()
        result = service.route("Test query")

        assert result == mock_response
        mock_router.query.assert_called_once_with("Test query")

    @patch('worker.services.router_service.vector_store_for_table')
    @patch('worker.services.router_service.Settings')
    @patch('worker.services.router_service.QwenLLM')
    @patch('worker.services.router_service.HuggingFaceEmbedding')
    @patch('worker.services.router_service.VectorStoreIndex')
    @patch('worker.services.router_service.VectorIndexRetriever')
    @patch('worker.services.router_service.TreeSummarize')
    @patch('worker.services.router_service.RetrieverQueryEngine')
    @patch('worker.services.router_service.LLMSingleSelector')
    @patch('worker.services.router_service.RouterQueryEngine')
    def test_route_with_force_level_none(self, mock_router_engine, mock_selector,
                                        mock_retriever_engine, mock_tree_summarize,
                                        mock_retriever, mock_index, mock_embed_model,
                                        mock_qwen, mock_settings, mock_vector_store):
        """Test routing with force_level='none'."""
        # Setup mocks
        mock_vector_store.return_value = MagicMock()
        mock_index.from_vector_store.return_value = MagicMock()
        mock_retriever.return_value = MagicMock()
        mock_tree_summarize.return_value = MagicMock()
        mock_retriever_engine.return_value = MagicMock()
        mock_selector.from_defaults.return_value = MagicMock()
        mock_router_engine.return_value = MagicMock()

        service = RouterService()

        # Mock the direct engine query method
        mock_direct_response = Response(response="Direct response")
        service._direct_engine.query = MagicMock(return_value=mock_direct_response)

        result = service.route("Test query", force_level="none")

        assert result == mock_direct_response
        service._direct_engine.query.assert_called_once_with("Test query")

    @patch('worker.services.router_service.vector_store_for_table')
    @patch('worker.services.router_service.Settings')
    @patch('worker.services.router_service.QwenLLM')
    @patch('worker.services.router_service.HuggingFaceEmbedding')
    @patch('worker.services.router_service.VectorStoreIndex')
    @patch('worker.services.router_service.VectorIndexRetriever')
    @patch('worker.services.router_service.TreeSummarize')
    @patch('worker.services.router_service.RetrieverQueryEngine')
    @patch('worker.services.router_service.LLMSingleSelector')
    @patch('worker.services.router_service.RouterQueryEngine')
    def test_route_with_valid_force_level(self, mock_router_engine, mock_selector,
                                         mock_retriever_engine, mock_tree_summarize,
                                         mock_retriever, mock_index, mock_embed_model,
                                         mock_qwen, mock_settings, mock_vector_store):
        """Test routing with valid force_level."""
        # Setup mocks
        mock_vector_store.return_value = MagicMock()
        mock_index.from_vector_store.return_value = MagicMock()
        mock_retriever.return_value = MagicMock()
        mock_tree_summarize.return_value = MagicMock()

        mock_engine = MagicMock()
        mock_response = Response(response="Code level response")
        mock_engine.query.return_value = mock_response
        mock_retriever_engine.return_value = mock_engine

        mock_selector.from_defaults.return_value = MagicMock()
        mock_router_engine.return_value = MagicMock()

        service = RouterService()
        result = service.route("Test query", force_level="code")

        assert result == mock_response
        mock_engine.query.assert_called_once_with("Test query")

    @patch('worker.services.router_service.vector_store_for_table')
    @patch('worker.services.router_service.Settings')
    @patch('worker.services.router_service.QwenLLM')
    @patch('worker.services.router_service.HuggingFaceEmbedding')
    @patch('worker.services.router_service.VectorStoreIndex')
    @patch('worker.services.router_service.VectorIndexRetriever')
    @patch('worker.services.router_service.TreeSummarize')
    @patch('worker.services.router_service.RetrieverQueryEngine')
    @patch('worker.services.router_service.LLMSingleSelector')
    @patch('worker.services.router_service.RouterQueryEngine')
    def test_route_with_invalid_force_level(self, mock_router_engine, mock_selector,
                                           mock_retriever_engine, mock_tree_summarize,
                                           mock_retriever, mock_index, mock_embed_model,
                                           mock_qwen, mock_settings, mock_vector_store):
        """Test routing with invalid force_level raises error."""
        # Setup mocks
        mock_vector_store.return_value = MagicMock()
        mock_index.from_vector_store.return_value = MagicMock()
        mock_retriever.return_value = MagicMock()
        mock_tree_summarize.return_value = MagicMock()
        mock_retriever_engine.return_value = MagicMock()
        mock_selector.from_defaults.return_value = MagicMock()
        mock_router_engine.return_value = MagicMock()

        service = RouterService()

        with pytest.raises(ValueError, match="Unknown force_level: invalid"):
            service.route("Test query", force_level="invalid")
