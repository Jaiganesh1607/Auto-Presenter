import asyncio
import structlog
from typing import List, Dict
import yaml
from pathlib import Path

from src.content_engine.retrieval.perspective_planner import Perspective
from src.content_engine.retrieval.web_search import search_web
from src.content_engine.retrieval.academic_search import search_academic
from src.content_engine.retrieval.fetcher import fetch_page_content
from src.content_engine.retrieval.rerank import rerank_chunks, RankedChunk

logger = structlog.get_logger(__name__)

config_path = Path(__file__).parent.parent.parent.parent / "configs" / "retrieval.yaml"
try:
    with open(config_path, "r") as f:
        CONFIG = yaml.safe_load(f)
except Exception:
    CONFIG = {"concurrency_limit": 10}

async def _process_query(query: str) -> List[dict]:
    """Process a single query: search web + academic, fetch URLs, chunk texts."""
    
    # 1. Search concurrently
    search_tasks = [
        search_web(query),
        search_academic(query)
    ]
    
    search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
    
    all_links = []
    for res in search_results:
        if isinstance(res, list):
            all_links.extend(res)
        elif isinstance(res, Exception):
            logger.warning("search_task_failed", error=str(res))
            
    import urllib.parse
    
    def _normalize_url(url: str) -> str:
        parsed = urllib.parse.urlparse(url)
        # Lowercase domain, strip trailing slashes, remove fragments
        return urllib.parse.urlunparse((parsed.scheme, parsed.netloc.lower(), parsed.path.rstrip('/'), parsed.params, parsed.query, ''))
        
    # Remove duplicates based on normalized URL and limit to top 1 per query to prevent CPU bottleneck
    unique_links_dict = {}
    for link in all_links:
        norm_url = _normalize_url(link["url"])
        if norm_url not in unique_links_dict:
            link["url"] = norm_url # Store normalized version
            unique_links_dict[norm_url] = link
            
    unique_links = list(unique_links_dict.values())[:1]
    
    # 2. Fetch contents concurrently
    fetch_tasks = [fetch_page_content(link["url"]) for link in unique_links]
    fetched_contents = await asyncio.gather(*fetch_tasks, return_exceptions=True)
    
    chunks = []
    for link, content in zip(unique_links, fetched_contents):
        if isinstance(content, Exception):
            logger.warning("fetch_content_failed", url=link["url"], error=str(content))
            continue
            
        if not content:
            continue
            
        import textwrap
        paragraphs = []
        for p in content.split("\n\n"):
            p = p.strip()
            if len(p) > 100:
                # Wrap at 1500 chars (approx 350 tokens) to guarantee no cross-encoder truncation
                wrapped = textwrap.wrap(p, width=1500, break_long_words=False, replace_whitespace=False)
                paragraphs.extend(wrapped)
                
        for p in paragraphs:
            chunks.append({
                "url": link["url"],
                "text": p
            })
            
    return chunks

async def parallel_retrieve(perspectives: List[Perspective], target_query: str) -> List[RankedChunk]:
    """Execute all queries from all perspectives concurrently, then rerank against the main topic."""
    
    all_queries = set()
    
    for p in perspectives:
        for q in p.queries:
            all_queries.add(q)
            
    logger.info("parallel_retrieval_started", num_queries=len(all_queries))
    
    # Process all queries concurrently with a semaphore
    limit = CONFIG.get("concurrency_limit", 10)
    sem = asyncio.Semaphore(limit)
    
    async def bounded_process(query: str):
        async with sem:
            return await _process_query(query)
            
    tasks = [bounded_process(q) for q in all_queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    all_chunks = []
    successful = 0
    exceptions = {}
    for res in results:
        if isinstance(res, list):
            all_chunks.extend(res)
            successful += 1
        elif isinstance(res, Exception):
            ex_type = type(res).__name__
            exceptions[ex_type] = exceptions.get(ex_type, 0) + 1
            logger.warning("query_processing_failed", error=str(res))
            
    logger.info("retrieval_diagnostics", 
                total_queries=len(all_queries), 
                successful_queries=successful, 
                exceptions=exceptions, 
                chunks_retrieved=len(all_chunks))
            
    # Deduplicate chunks to avoid reranking identical texts
    unique_chunks = {}
    for c in all_chunks:
        unique_chunks[c["text"]] = c
        
    logger.info("parallel_retrieval_finished", num_unique_chunks=len(unique_chunks))
    
    # 3. Rerank against the original topic/target_query
    ranked = await rerank_chunks(target_query, list(unique_chunks.values()))
    
    return ranked
