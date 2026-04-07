from __future__ import annotations
from pydantic import BaseModel, Field
import httpx
from cmap_agent.config.settings import settings

class WebSearchArgs(BaseModel):
    query: str = Field(..., description="Web search query")
    limit: int = Field(5, ge=1, le=10, description="Max results")

def web_search(args: WebSearchArgs, ctx: dict) -> dict:
    if not settings.TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY is not set; web.search is unavailable.")
    url = "https://api.tavily.com/search"
    payload = {"api_key": settings.TAVILY_API_KEY, "query": args.query, "max_results": args.limit}
    with httpx.Client(timeout=30) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
    results=[]
    for item in data.get("results", [])[: args.limit]:
        results.append({
            "title": item.get("title"),
            "url": item.get("url"),
            "content": item.get("content") or item.get("snippet"),
            "score": item.get("score"),
        })
    return {"results": results}
