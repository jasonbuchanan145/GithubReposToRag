from fastapi import APIRouter
from starlette.responses import StreamingResponse
from uuid import uuid4
from rag_shared.bus import ProgressBus, CancelFlags
from rag_shared.config import REDIS_URL
from rag_shared.models import QueryRequest
from arq.connections import create_pool, RedisSettings

router = APIRouter()
bus = ProgressBus(REDIS_URL)
flags = CancelFlags(REDIS_URL)

def _redis_settings(): return RedisSettings.from_dsn(REDIS_URL)

@router.post("/rag/jobs")
async def create_job(req: QueryRequest):
    job_id = uuid4().hex
    pool = await create_pool(_redis_settings())
    await pool.enqueue_job("run_rag_job", job_id, req.model_dump())
    return {"job_id": job_id}

@router.get("/rag/jobs/{job_id}/events")
async def job_events(job_id: str):
    async def gen():
        async for chunk in bus.stream(job_id):
            yield chunk
    return StreamingResponse(gen(), media_type="text/event-stream")

@router.post("/rag/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    await flags.cancel(job_id)
    return {"status": "cancelling", "job_id": job_id}
