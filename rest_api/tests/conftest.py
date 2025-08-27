
import sys
import types
import asyncio
from fastapi import FastAPI
import pytest

from rest_api.src.app.controllers.jobs_controller import router


# ---- Create stub rag_shared package before importing the router ----
@pytest.fixture(scope="session", autouse=True)
def stub_rag_shared():
    # Create package and submodules
    rag_shared = types.ModuleType("rag_shared")
    # config
    config = types.ModuleType("rag_shared.config")
    config.REDIS_URL = "redis://localhost:6379/0"
    # arbitrary config fields used by health/main (not used in router tests)
    config.LOG_LEVEL = "INFO"
    config.CASSANDRA_USERNAME = "u"
    config.CASSANDRA_PASSWORD = "p"
    config.CASSANDRA_HOST = "localhost"
    config.CASSANDRA_PORT = 9042
    config.CASSANDRA_KEYSPACE = "ks"
    config.QWEN_ENDPOINT = "http://qwen"
    # models with a minimal Pydantic v2 BaseModel
    import pydantic
    class QueryRequest(pydantic.BaseModel):
        q: str = "hello"
    models = types.ModuleType("rag_shared.models")
    models.QueryRequest = QueryRequest
    # bus stubs (will be monkeypatched per-test)
    bus_mod = types.ModuleType("rag_shared.bus")
    class _Bus:
        def __init__(self, url): self.url=url
        async def stream(self, job_id):
            yield b"data: 100\\n\\n"
    class _Flags:
        def __init__(self, url): self.url=url
        async def cancel(self, job_id): return None
    bus_mod.ProgressBus = _Bus
    bus_mod.CancelFlags = _Flags
    # attach submodules
    sys.modules["rag_shared"] = rag_shared
    sys.modules["rag_shared.config"] = config
    sys.modules["rag_shared.models"] = models
    sys.modules["rag_shared.bus"] = bus_mod
    # Stub arq connections module
    arq_connections = types.ModuleType("arq.connections")
    class _DummyPool:
        def __init__(self): self.calls=[]
        async def enqueue_job(self, *args, **kwargs):
            self.calls.append((args, kwargs))
    async def create_pool(settings):
        return _DummyPool()
    class RedisSettings:
        @classmethod
        def from_dsn(cls, dsn): return cls()
    arq_connections.create_pool = create_pool
    arq_connections.RedisSettings = RedisSettings
    sys.modules["arq.connections"] = arq_connections
    yield
    # no teardown needed

@pytest.fixture()
def app_with_jobs_router(stub_rag_shared):
    app = FastAPI()
    app.include_router(router)
    return app
