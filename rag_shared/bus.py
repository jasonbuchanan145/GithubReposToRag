import json, asyncio
from typing import AsyncIterator, Dict
import redis.asyncio as redis

_CHAN = "job:{id}:events"
_FLAG = "job:{id}:cancel"

class ProgressBus:
    def __init__(self, url: str): self.url = url
    async def _conn(self): return await redis.from_url(self.url, decode_responses=True)

    async def emit(self, job_id: str, event: str, data: Dict):
        r = await self._conn()
        await r.publish(_CHAN.format(id=job_id), json.dumps({"event": event, "data": data}, ensure_ascii=False))
        await r.close()

    async def stream(self, job_id: str) -> AsyncIterator[str]:
        r = await self._conn()
        ps = r.pubsub()
        await ps.subscribe(_CHAN.format(id=job_id))
        try:
            while True:
                msg = await ps.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if msg and msg["type"] == "message":
                    yield f"data: {msg['data']}\n\n"
                yield f": ping\n\n"
                await asyncio.sleep(1.0)
        finally:
            await ps.unsubscribe(_CHAN.format(id=job_id))
            await ps.close(); await r.close()

class CancelFlags:
    def __init__(self, url: str): self.url = url
    async def is_cancelled(self, job_id: str) -> bool:
        r = await redis.from_url(self.url, decode_responses=True)
        v = await r.get(_FLAG.format(id=job_id)); await r.close()
        return v is not None
    async def cancel(self, job_id: str):
        r = await redis.from_url(self.url, decode_responses=True)
        await r.set(_FLAG.format(id=job_id), "1", ex=3600); await r.close()
