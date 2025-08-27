
from typing import Dict, List, Any, Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    repo_name: Optional[str] = None


class RAGResponse(BaseModel):
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None
