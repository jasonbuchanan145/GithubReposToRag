
import asyncio
import rest_api.src.app.controllers.jobs_controller as jc
from fastapi.testclient import TestClient

def test_create_job_enqueues_and_returns_id(app_with_jobs_router, monkeypatch):
    # Capture the created pool so we can inspect calls
    created = {}
    class DummyPool:
        def __init__(self): self.calls=[]
        async def enqueue_job(self, *args, **kwargs):
            self.calls.append((args, kwargs))
    async def fake_create_pool(_): 
        created['pool']=DummyPool()
        return created['pool']

    monkeypatch.setattr(jc, "create_pool", fake_create_pool)

    client = TestClient(app_with_jobs_router)
    payload = {"query": "who was first?"}
    r = client.post("/rag/jobs", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "job_id" in data and isinstance(data["job_id"], str) and len(data["job_id"])>0

    # ensure job was enqueued once with expected args
    pool = created['pool']
    assert len(pool.calls)==1
    (args, kwargs) = pool.calls[0]
    # first arg is function name
    assert args[0] == "run_rag_job"
    # second arg is job_id that should match response
    assert args[1] == data["job_id"]
    # third arg is the serialized model dict
    assert isinstance(args[2], dict)
    assert args[2]["query"] == payload["query"]

def test_job_events_streams_sse(app_with_jobs_router, monkeypatch):
    # Make the bus stream yield two chunks
    async def agen():
        yield b"data: 10\\n\\n"
        yield b"data: 20\\n\\n"
    class DummyBus:
        def __init__(self, *_): pass
        async def stream(self, job_id): 
            async for x in agen():
                yield x
    monkeypatch.setattr(jc, "bus", DummyBus(None))

    client = TestClient(app_with_jobs_router)
    with client.stream("GET", "/rag/jobs/abc/events") as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = b"".join(list(r.iter_raw()))
        assert b"data: 10" in body and b"data: 20" in body

def test_cancel_job_calls_flag_and_returns_status(app_with_jobs_router, monkeypatch):
    called = {}
    class DummyFlags:
        def __init__(self, *_): pass
        async def cancel(self, job_id):
            called['job_id'] = job_id
    monkeypatch.setattr(jc, "flags", DummyFlags(None))

    client = TestClient(app_with_jobs_router)
    r = client.post("/rag/jobs/xyz/cancel")
    assert r.status_code == 200
    assert r.json() == {"status": "cancelling", "job_id": "xyz"}
    assert called['job_id'] == "xyz"
