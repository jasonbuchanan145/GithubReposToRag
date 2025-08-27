"""
Unit tests for worker module.
"""
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from llama_index.core.base.response.schema import Response
from llama_index.core.schema import NodeWithScore, TextNode

from worker.worker import (
    run_rag_job, _extract_sources, _should_retry
)


class TestWorkerHelperFunctions:
    """Tests for worker helper functions."""

    def test_extract_sources_with_nodes(self):
        """Test _extract_sources with valid source nodes."""
        response = Response(response="Test response")

        # Create mock nodes
        node1 = TextNode(text="This is a long text that should be truncated because it exceeds 300 characters. " * 5)
        node2 = TextNode(text="Short text")
        node1.metadata = {"file": "test1.py", "line": 10}
        node2.metadata = {"file": "test2.py", "line": 20}

        source1 = NodeWithScore(node=node1, score=0.9)
        source2 = NodeWithScore(node=node2, score=0.8)
        response.source_nodes = [source1, source2]

        sources = _extract_sources(response)

        assert len(sources) == 2
        assert sources[0]["score"] == 0.9
        assert sources[0]["metadata"] == {"file": "test1.py", "line": 10}
        assert len(sources[0]["text"]) <= 303  # 300 chars + "..."
        assert sources[0]["text"].endswith("...")

        assert sources[1]["score"] == 0.8
        assert sources[1]["metadata"] == {"file": "test2.py", "line": 20}
        assert sources[1]["text"] == "Short text"

    def test_extract_sources_without_nodes(self):
        """Test _extract_sources with no source nodes."""
        response = Response(response="Test response")
        response.source_nodes = None

        sources = _extract_sources(response)

        assert sources == []

    def test_extract_sources_empty_nodes(self):
        """Test _extract_sources with empty source nodes."""
        response = Response(response="Test response")
        response.source_nodes = []

        sources = _extract_sources(response)

        assert sources == []

    def test_extract_sources_exception_handling(self):
        """Test _extract_sources handles exceptions gracefully."""
        response = MagicMock()
        response.source_nodes = [MagicMock()]
        response.source_nodes[0].node.text = None  # This might cause an issue
        response.source_nodes[0].node.metadata = None
        response.source_nodes[0].score = None

        # Should not raise exception, should return empty list
        sources = _extract_sources(response)

        # Even with exceptions, should try to extract what it can
        assert isinstance(sources, list)

    def test_should_retry_with_force_none(self):
        """Test _should_retry with force_level='none'."""
        assert not _should_retry(0, 1, "none")
        assert not _should_retry(10, 1, "none")

    def test_should_retry_max_attempts(self):
        """Test _should_retry at max attempts."""
        with patch('worker.worker.MAX_RAG_ATTEMPTS', 3):
            assert not _should_retry(0, 3, "code")
            assert not _should_retry(0, 4, "code")

    def test_should_retry_sufficient_sources(self):
        """Test _should_retry with sufficient sources."""
        with patch('worker.worker.MIN_SOURCE_NODES', 2):
            assert not _should_retry(2, 1, "code")
            assert not _should_retry(3, 1, "code")

    def test_should_retry_insufficient_sources(self):
        """Test _should_retry with insufficient sources."""
        with patch('worker.worker.MAX_RAG_ATTEMPTS', 3), \
             patch('worker.worker.MIN_SOURCE_NODES', 2):
            assert _should_retry(1, 1, "code")
            assert _should_retry(0, 2, "package")


class TestRunRagJob:
    """Tests for run_rag_job function."""

    @pytest.mark.asyncio
    async def test_run_rag_job_success_no_retry(self):
        """Test successful job without retry."""
        job_id = "test-job-123"
        req = {"query": "test query", "force_level": "code"}

        # Mock response with sufficient sources
        mock_response = Response(response="Test answer")
        node = TextNode(text="source text")
        node.metadata = {"file": "test.py"}
        source_node = NodeWithScore(node=node, score=0.9)
        mock_response.source_nodes = [source_node]

        with patch('worker.worker.bus') as mock_bus, \
             patch('worker.worker.flags') as mock_flags, \
             patch('worker.worker.engine') as mock_engine, \
             patch('worker.worker._should_retry', return_value=False):

            mock_bus.emit = AsyncMock()
            mock_flags.is_cancelled = AsyncMock(return_value=False)
            mock_engine.run_once.return_value = mock_response

            await run_rag_job(None, job_id, req)

            # Verify engine was called
            mock_engine.run_once.assert_called_once_with("test query", "code")

            # Verify final emission
            final_call = None
            for call in mock_bus.emit.call_args_list:
                if call[0][1] == "final":
                    final_call = call
                    break

            assert final_call is not None
            assert "Test answer" in final_call[0][2]["answer"]
            assert final_call[0][2]["sources"] is not None
            assert len(final_call[0][2]["sources"]) == 1

    @pytest.mark.asyncio
    async def test_run_rag_job_cancelled(self):
        """Test job cancellation."""
        job_id = "test-job-cancelled"
        req = {"query": "test query"}

        with patch('worker.worker.bus') as mock_bus, \
             patch('worker.worker.flags') as mock_flags:

            mock_bus.emit = AsyncMock()
            mock_flags.is_cancelled = AsyncMock(return_value=True)

            await run_rag_job(None, job_id, req)

            # Verify cancellation was handled
            final_call = None
            for call in mock_bus.emit.call_args_list:
                if call[0][1] == "final":
                    final_call = call
                    break

            assert final_call is not None
            assert final_call[0][2]["cancelled"] is True

    @pytest.mark.asyncio
    async def test_run_rag_job_with_retry(self):
        """Test job with retry mechanism."""
        job_id = "test-job-retry"
        req = {"query": "initial query"}

        # Mock first response with insufficient sources
        mock_response1 = Response(response="First answer")
        mock_response1.source_nodes = []

        # Mock second response with sufficient sources
        mock_response2 = Response(response="Second answer")
        node = TextNode(text="source text")
        source_node = NodeWithScore(node=node, score=0.9)
        mock_response2.source_nodes = [source_node]

        with patch('worker.worker.bus') as mock_bus, \
             patch('worker.worker.flags') as mock_flags, \
             patch('worker.worker.engine') as mock_engine, \
             patch('worker.worker._should_retry', side_effect=[True, False]), \
             patch('worker.worker.MAX_RAG_ATTEMPTS', 3):

            mock_bus.emit = AsyncMock()
            mock_flags.is_cancelled = AsyncMock(return_value=False)
            mock_engine.run_once.side_effect = [mock_response1, mock_response2]
            mock_engine.propose_refinement.return_value = {
                "query": "refined query",
                "force_level": "package"
            }

            await run_rag_job(None, job_id, req)

            # Verify engine was called twice
            assert mock_engine.run_once.call_count == 2
            mock_engine.propose_refinement.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_rag_job_exception_handling(self):
        """Test job exception handling."""
        job_id = "test-job-error"
        req = {"query": "test query"}

        with patch('worker.worker.bus') as mock_bus, \
             patch('worker.worker.flags') as mock_flags, \
             patch('worker.worker.engine') as mock_engine:

            mock_bus.emit = AsyncMock()
            mock_flags.is_cancelled = AsyncMock(return_value=False)
            mock_engine.run_once.side_effect = Exception("Test error")

            await run_rag_job(None, job_id, req)

            # Verify error was handled
            error_call = None
            final_call = None
            for call in mock_bus.emit.call_args_list:
                if call[0][1] == "error":
                    error_call = call
                elif call[0][1] == "final":
                    final_call = call

            assert error_call is not None
            assert "Test error" in error_call[0][2]["message"]
            assert final_call is not None
            assert final_call[0][2]["error"] is True

    @pytest.mark.asyncio
    async def test_run_rag_job_empty_query(self):
        """Test job with empty query."""
        job_id = "test-job-empty"
        req = {"query": "   "}  # Whitespace only

        with patch('worker.worker.bus') as mock_bus, \
             patch('worker.worker.flags') as mock_flags, \
             patch('worker.worker.engine') as mock_engine:

            mock_bus.emit = AsyncMock()
            mock_flags.is_cancelled = AsyncMock(return_value=False)
            mock_engine.run_once.return_value = Response(response="Empty query response")

            await run_rag_job(None, job_id, req)

            # Should still process but with empty string
            mock_engine.run_once.assert_called()
            call_args = mock_engine.run_once.call_args[0]
            assert call_args[0] == ""  # Stripped empty query

    @pytest.mark.asyncio
    async def test_run_rag_job_progress_emissions(self):
        """Test that job emits proper progress events."""
        job_id = "test-job-progress"
        req = {"query": "test query", "force_level": "code"}

        mock_response = Response(response="Test answer")
        mock_response.source_nodes = []

        with patch('worker.worker.bus') as mock_bus, \
             patch('worker.worker.flags') as mock_flags, \
             patch('worker.worker.engine') as mock_engine, \
             patch('worker.worker._should_retry', return_value=False):

            mock_bus.emit = AsyncMock()
            mock_flags.is_cancelled = AsyncMock(return_value=False)
            mock_engine.run_once.return_value = mock_response

            await run_rag_job(None, job_id, req)

            # Check that all expected events were emitted
            emitted_events = [call[0][1] for call in mock_bus.emit.call_args_list]
            assert "started" in emitted_events
            assert "iteration" in emitted_events
            assert "retrieval" in emitted_events
            assert "final" in emitted_events


class TestWorkerSettings:
    """Tests for WorkerSettings class."""

    def test_worker_settings_configuration(self):
        """Test WorkerSettings has correct configuration."""
        from worker.worker import WorkerSettings

        # Test that the settings exist and have expected values
        assert hasattr(WorkerSettings, 'redis_settings')
        assert hasattr(WorkerSettings, 'functions')
        assert hasattr(WorkerSettings, 'max_jobs')
        assert hasattr(WorkerSettings, 'job_timeout')
        assert hasattr(WorkerSettings, 'keep_result')

        # Test that run_rag_job is in functions
        assert run_rag_job in WorkerSettings.functions
