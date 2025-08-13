import requests
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi_health import health
# LlamaIndex imports
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.cassandra import CassandraVectorStore

import psutil
import time
from datetime import datetime

# Singleton variable for application start time
_app_start_time = None

def register_health_endpoints(app: FastAPI, 
                             CASSANDRA_USERNAME: str,
                             CASSANDRA_PASSWORD: str, 
                             CASSANDRA_HOST: str, 
                             CASSANDRA_PORT: int, 
                             CASSANDRA_KEYSPACE: str, 
                             QWEN_ENDPOINT: str, 
                             global_index):
    """Register health endpoints on the provided FastAPI app"""

    @app.get("/actuator/health")
    async def detailed_health():
        """Comprehensive health check similar to Spring Boot Actuator"""
        start_time = time.time()

        health_checks = {
            "status": "UP",
            "components": {},
            "details": {
                "application": {
                    "name": "RAG API Service",
                    "version": "1.0.0",
                    "uptime": time.time() - _get_app_start_time(),
                    "timestamp": datetime.utcnow().isoformat()
                },
                "system": {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_usage": psutil.disk_usage('/').percent
                }
            }
        }

        # Cassandra Health
        try:
            auth_provider = PlainTextAuthProvider(username=CASSANDRA_USERNAME, password=CASSANDRA_PASSWORD)
            cluster = Cluster([CASSANDRA_HOST], port=CASSANDRA_PORT, auth_provider=auth_provider)
            session = cluster.connect(CASSANDRA_KEYSPACE)

            # Check table count
            result = session.execute(f"SELECT COUNT(*) FROM {CASSANDRA_KEYSPACE}.embeddings")
            count = result.one()[0] if result else 0

            cluster.shutdown()

            health_checks["components"]["cassandra"] = {
                "status": "UP",
                "details": {
                    "host": CASSANDRA_HOST,
                    "port": CASSANDRA_PORT,
                    "keyspace": CASSANDRA_KEYSPACE,
                    "embeddings_count": count
                }
            }
        except Exception as e:
            health_checks["components"]["cassandra"] = {
                "status": "DOWN",
                "details": {"error": str(e)}
            }
            health_checks["status"] = "DOWN"

        # Qwen Service Health
        try:
            response = requests.get(f"{QWEN_ENDPOINT}/health", timeout=5)
            health_checks["components"]["qwen"] = {
                "status": "UP" if response.status_code == 200 else "DOWN",
                "details": {
                    "endpoint": QWEN_ENDPOINT,
                    "response_time_ms": response.elapsed.total_seconds() * 1000
                }
            }
        except Exception as e:
            health_checks["components"]["qwen"] = {
                "status": "DOWN",
                "details": {"error": str(e)}
            }
            health_checks["status"] = "DOWN"

        # Vector Index Health
        try:
            if global_index is not None:
                # Try a simple retrieval to test the index
                test_retriever = global_index.as_retriever(similarity_top_k=1)
                test_results = test_retriever.retrieve("health check")

                health_checks["components"]["vector_index"] = {
                    "status": "UP",
                    "details": {
                        "initialized": True,
                        "test_results_count": len(test_results)
                    }
                }
            else:
                health_checks["components"]["vector_index"] = {
                    "status": "DOWN",
                    "details": {"initialized": False}
                }
                health_checks["status"] = "DOWN"
        except Exception as e:
            health_checks["components"]["vector_index"] = {
                "status": "DOWN",
                "details": {"error": str(e)}
            }
            health_checks["status"] = "DOWN"

        # Add response time
        health_checks["details"]["response_time_ms"] = (time.time() - start_time) * 1000

        return health_checks


def _get_app_start_time():
    """Get or initialize the application start time singleton"""
    global _app_start_time
    if _app_start_time is None:
        _app_start_time = time.time()
    return _app_start_time
