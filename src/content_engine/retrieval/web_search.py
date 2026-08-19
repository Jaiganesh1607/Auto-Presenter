import asyncio
import os
import structlog
from typing import List, Dict
import yaml
from pathlib import Path
from duckduckgo_search import DDGS
from .cache import get_from_cache, save_to_cache

logger = structlog.get_logger(__name__)

class SearchTimeoutError(Exception):
    pass

# Load config
config_path = Path(__file__).parent.parent.parent.parent / "configs" / "retrieval.yaml"
try:
    with open(config_path, "r") as f:
        CONFIG = yaml.safe_load(f)
except Exception:
    CONFIG = {"web_search_provider": "wikipedia", "web_search_timeout": 10}

def _do_duckduckgo_search(query: str) -> List[Dict]:
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5, backend="html"):
                results.append({
                    "url": r.get("href"),
                    "title": r.get("title"),
                    "snippet": r.get("body")
                })
    except Exception as e:
        logger.warning("duckduckgo_failed_or_ratelimited", error=str(e))
    return results

def _do_wikipedia_search(query: str) -> List[Dict]:
    import wikipedia
    # Wikipedia API strictly requires a custom User-Agent to avoid rate limiting and 403 errors
    wikipedia.set_user_agent("AutoPresenterBot/1.0 (https://github.com/jaiganesh/auto-presenter)")
    results = []
    try:
        search_results = wikipedia.search(query, results=2)
        for title in search_results:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                results.append({
                    "url": page.url,
                    "title": page.title,
                    "snippet": page.summary[:1000]
                })
            except Exception:
                continue
    except Exception as e:
        logger.warning("wikipedia_search_failed", error=str(e))
    return results

def _do_tavily_search(query: str) -> List[Dict]:
    from tavily import TavilyClient
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        logger.warning("tavily_api_key_missing_falling_back_to_ddg")
        return _do_duckduckgo_search(query)
        
    client = TavilyClient(api_key=api_key)
    response = client.search(query, max_results=5)
    results = []
    for r in response.get("results", []):
        results.append({
            "url": r.get("url"),
            "title": r.get("title"),
            "snippet": r.get("content")
        })
    return results

async def search_web(query: str) -> List[Dict]:
    """Perform a web search with timeout and caching."""
    cached = await get_from_cache(query, "web")
    if cached is not None:
        return cached

    provider = CONFIG.get("web_search_provider", "wikipedia")
    timeout = CONFIG.get("web_search_timeout", 10)
    
    logger.info("web_search_started", query=query, provider=provider)
    
    try:
        if provider == "tavily":
            results = await asyncio.wait_for(
                asyncio.to_thread(_do_tavily_search, query),
                timeout=timeout
            )
        elif provider == "wikipedia":
            results = await asyncio.wait_for(
                asyncio.to_thread(_do_wikipedia_search, query),
                timeout=timeout
            )
        else:
            results = await asyncio.wait_for(
                asyncio.to_thread(_do_duckduckgo_search, query),
                timeout=timeout
            )
            
        await save_to_cache(query, "web", results)
        return results
    except asyncio.TimeoutError:
        logger.error("web_search_timeout", query=query)
        raise SearchTimeoutError(f"Web search timed out after {timeout} seconds")
    except Exception as e:
        logger.error("web_search_failed", query=query, error=str(e))
        # Return empty list on failure instead of crashing
        return []
