"""
Pytest configuration and fixtures for rag_worker tests.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock
from llama_index.core.base.response.schema import Response
from llama_index.core.schema import NodeWithScore, TextNode


@pytest.fixture
def mock_settings(mocker):
    """Mock LlamaIndex Settings."""
    mock_settings = mocker.patch('worker.services.router_service.Settings')
    mock_llm = MagicMock()
    mock_llm.complete.return_value = MagicMock(text="Mocked LLM response")
    mock_llm.acomplete = AsyncMock(return_value=MagicMock(text="Mocked async LLM response"))
    mock_settings.llm = mock_llm
    mock_settings.embed_model = MagicMock()
    mock_settings.callback_manager = MagicMock()
    return mock_settings


@pytest.fixture
def mock_vector_store(mocker):
    """Mock vector store creation."""
    return mocker.patch('worker.services.router_service.vector_store_for_table')


@pytest.fixture
def mock_response():
    """Create a mock Response object with source nodes."""
    response = Response(response="Test response")

    # Create mock source nodes
    node1 = TextNode(text="Source text 1", metadata={"file": "test1.py", "line": 10})
    node2 = TextNode(text="Source text 2", metadata={"file": "test2.py", "line": 20})

    source_node1 = NodeWithScore(node=node1, score=0.9)
    source_node2 = NodeWithScore(node=node2, score=0.8)

    response.source_nodes = [source_node1, source_node2]
    return response


@pytest.fixture
def mock_cassandra_session(mocker):
    """Mock Cassandra session."""
    return mocker.patch('worker.services.cassandra_store.get_cassandra_session')


@pytest.fixture
def mock_requests(mocker):
    """Mock requests library for HTTP calls."""
    return mocker.patch('worker.services.qwen_llm.requests')
