import asyncio
import structlog
from typing import List, Dict
import yaml
from pathlib import Path
import arxiv
from .cache import get_from_cache, save_to_cache
from .web_search import SearchTimeoutError

logger = structlog.get_logger(__name__)

# Load config
config_path = Path(__file__).parent.parent.parent.parent / "configs" / "retrieval.yaml"
try:
    with open(config_path, "r") as f:
        CONFIG = yaml.safe_load(f)
except Exception:
    CONFIG = {"academic_search_timeout": 10}

def _do_arxiv_search(query: str) -> List[Dict]:
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=3,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    results = []
    for result in client.results(search):
        results.append({
            "url": result.entry_id,
            "title": result.title,
            "snippet": result.summary
        })
    return results

_consecutive_failures = 0
_CIRCUIT_BREAKER_LIMIT = 3
_circuit_open = False

def reset_academic_circuit_breaker():
    global _consecutive_failures, _circuit_open
    _consecutive_failures = 0
    _circuit_open = False

async def search_academic(query: str) -> List[Dict]:
    """Perform academic search (arXiv) with timeout and caching."""
    global _consecutive_failures, _circuit_open
    
    if _circuit_open:
        logger.warning("academic_search_skipped", reason="circuit_breaker_open")
        raise SearchTimeoutError("Circuit breaker open: academic search disabled due to repeated failures")
        
    cached = await get_from_cache(query, "academic")
    if cached is not None:
        return cached

    timeout = CONFIG.get("academic_search_timeout", 10)
    
    logger.info("academic_search_started", query=query)
    
    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(_do_arxiv_search, query),
            timeout=timeout
        )
            
        await save_to_cache(query, "academic", results)
        _consecutive_failures = 0  # reset on success
        return results
    except asyncio.TimeoutError:
        _consecutive_failures += 1
        if _consecutive_failures >= _CIRCUIT_BREAKER_LIMIT:
            _circuit_open = True
            logger.error("academic_search_circuit_breaker_tripped", failures=_consecutive_failures)
        else:
            logger.error("academic_search_timeout", query=query, failures=_consecutive_failures)
            
        raise SearchTimeoutError(f"Academic search timed out after {timeout} seconds")
    except Exception as e:
        _consecutive_failures += 1
        if _consecutive_failures >= _CIRCUIT_BREAKER_LIMIT:
            _circuit_open = True
            logger.error("academic_search_circuit_breaker_tripped", failures=_consecutive_failures)
        else:
            logger.error("academic_search_failed", query=query, error=str(e))
        return []
