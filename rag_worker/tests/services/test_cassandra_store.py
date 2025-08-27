"""
Unit tests for cassandra_store.
"""
import pytest
from unittest.mock import MagicMock, patch
from worker.services.cassandra_store import get_cassandra_session, vector_store_for_table


class TestCassandraStore:
    """Tests for Cassandra store functions."""

    def test_get_cassandra_session_cached(self):
        """Test that session is cached after first creation."""
        # Reset the global session
        import worker.services.cassandra_store as cs
        cs._session = None

        with patch('worker.services.cassandra_store.Cluster') as mock_cluster, \
             patch('worker.services.cassandra_store.PlainTextAuthProvider') as mock_auth:

            mock_session = MagicMock()
            mock_cluster_instance = MagicMock()
            mock_cluster_instance.connect.return_value = mock_session
            mock_cluster.return_value = mock_cluster_instance

            # First call should create session
            session1 = get_cassandra_session()
            assert session1 == mock_session

            # Second call should return cached session
            session2 = get_cassandra_session()
            assert session2 == mock_session
            assert session1 is session2

            # Cluster should only be created once
            mock_cluster.assert_called_once()
            mock_cluster_instance.connect.assert_called_once()

    @patch('worker.services.cassandra_store.get_cassandra_session')
    @patch('worker.services.cassandra_store.CassandraVectorStore')
    def test_vector_store_for_table(self, mock_vector_store_class, mock_get_session):
        """Test vector store creation for a table."""
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        mock_vector_store = MagicMock()
        mock_vector_store_class.return_value = mock_vector_store

        result = vector_store_for_table("test_table")

        assert result == mock_vector_store
        mock_get_session.assert_called_once()
        mock_vector_store_class.assert_called_once_with(
            session=mock_session,
            table="test_table",
            embedding_dimension=384,  # From config
            keyspace="vector_store"   # From config
        )

    @patch('worker.services.cassandra_store.Cluster')
    @patch('worker.services.cassandra_store.PlainTextAuthProvider')
    def test_get_cassandra_session_creates_auth_provider(self, mock_auth, mock_cluster):
        """Test that authentication provider is created correctly."""
        # Reset the global session
        import worker.services.cassandra_store as cs
        cs._session = None

        mock_session = MagicMock()
        mock_cluster_instance = MagicMock()
        mock_cluster_instance.connect.return_value = mock_session
        mock_cluster.return_value = mock_cluster_instance

        mock_auth_provider = MagicMock()
        mock_auth.return_value = mock_auth_provider

        session = get_cassandra_session()

        # Verify auth provider was created with correct credentials
        mock_auth.assert_called_once_with(
            username="cassandra",  # From config
            password="testyMcTesterson"  # From config
        )

        # Verify cluster was created with auth provider
        mock_cluster.assert_called_once_with(
            ["rag-demo-cassandra"],  # From config
            port=9042,  # From config
            auth_provider=mock_auth_provider
        )

        # Verify session connection
        mock_cluster_instance.connect.assert_called_once_with("vector_store")  # From config
        assert session == mock_session

    def teardown_method(self):
        """Reset global session after each test."""
        import worker.services.cassandra_store as cs
        cs._session = None
