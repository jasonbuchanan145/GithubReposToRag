import json
import types

from fastapi import FastAPI
from fastapi.testclient import TestClient

# Import the health registration helpers
from rest_api.src.app.health import register_health_endpoints, _format_uptime
from rest_api.src.app import health as health_mod

class FakeElapsed:
    def total_seconds(self):
        return 0.123

class FakeResponse:
    status_code = 200
    elapsed = FakeElapsed()

def build_app_with_health(monkeypatch, cassandra_ok=True, qwen_ok=True, index_initialized=True):

    app = FastAPI()

    # ---- Patch Cassandra Cluster ----


    if cassandra_ok:
        class FakeResult:
            def one(self):
                return [42]  # COUNT(*)
        class FakeSession:
            def execute(self, q):
                return FakeResult()
        class FakeCluster:
            def __init__(self, *a, **k): pass
            def connect(self, keyspace):
                return FakeSession()
            def shutdown(self):
                return None
        monkeypatch.setattr(health_mod, "Cluster", FakeCluster, raising=True)
        class FakeAuth:
            def __init__(self, username, password): pass
        monkeypatch.setattr(health_mod, "PlainTextAuthProvider", FakeAuth, raising=True)
    else:
        class ExplodingCluster:
            def __init__(self,*a,**k): 
                raise RuntimeError("no cassandra")
        monkeypatch.setattr(health_mod, "Cluster", ExplodingCluster, raising=True)

    # ---- Patch requests to QWEN endpoint ----
    if qwen_ok:
        monkeypatch.setattr(health_mod, "requests", types.SimpleNamespace(get=lambda url, timeout=1.0: FakeResponse()))
    else:
        class ReqErr:
            def get(self, *a, **k):
                raise RuntimeError("qwen down")
        monkeypatch.setattr(health_mod, "requests", ReqErr())

    # ---- Provide a fake vector index ----
    class FakeRetriever:
        def retrieve(self, q): 
            return [ {"doc": "x"} ] if index_initialized else []
    class FakeIndex:
        def as_retriever(self, similarity_top_k=1):
            return FakeRetriever()
    global_index = FakeIndex() if index_initialized else None

    # Register route
    register_health_endpoints(
        app,
        CASSANDRA_USERNAME="u",
        CASSANDRA_PASSWORD="p",
        CASSANDRA_HOST="localhost",
        CASSANDRA_PORT=9042,
        CASSANDRA_KEYSPACE="ks",
        QWEN_ENDPOINT="http://qwen/health",
        global_index=global_index,
    )

    return app

def test_health_all_up(monkeypatch):
    app = build_app_with_health(monkeypatch, cassandra_ok=True, qwen_ok=True, index_initialized=True)
    client = TestClient(app)
    r = client.get("/health")
    print(json.dumps(r.json(), indent=2))
    assert r.status_code == 200
    j = r.json()

    assert j["status"] == "UP"
    # Components present and UP
    comps = j["components"]
    assert comps["cassandra"]["status"] == "UP"
    assert comps["qwen"]["status"] == "UP"
    assert comps["vector_index"]["status"] == "UP"
    assert comps["vector_index"]["details"]["initialized"] is True
    # Has application/system details
    assert "application" in j["details"]
    assert "system" in j["details"]
    assert isinstance(j["details"]["application"]["uptime_human_readable"], str)

def test_health_cassandra_down(monkeypatch):
    app = build_app_with_health(monkeypatch, cassandra_ok=False, qwen_ok=True, index_initialized=True)
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 503  # degraded -> service unavailable
    j = r.json()
    assert j["status"] == "DOWN"
    assert j["components"]["cassandra"]["status"] == "DOWN"

def test_health_qwen_down_sets_overall_down(monkeypatch):
    app = build_app_with_health(monkeypatch, cassandra_ok=True, qwen_ok=False, index_initialized=True)
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 503
    j = r.json()
    assert j["status"] == "DOWN"
    assert j["components"]["qwen"]["status"] == "DOWN"

def test_health_vector_index_not_initialized(monkeypatch):
    app = build_app_with_health(monkeypatch, cassandra_ok=True, qwen_ok=True, index_initialized=False)
    client = TestClient(app)
    r = client.get("/health")
    # vector index missing shouldn't fail health
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "UP"
    assert j["components"]["vector_index"]["details"]["initialized"] is False

def test_format_uptime_helper():
    assert _format_uptime(0.4).endswith("seconds")
    assert "1 minute" in _format_uptime(60*1 + 2)
    assert "1 hour" in _format_uptime(3600 + 5)
