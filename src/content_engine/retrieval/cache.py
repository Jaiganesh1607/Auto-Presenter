import asyncio
import hashlib
import json
import time
from pathlib import Path
import aiofiles
import structlog
import yaml

logger = structlog.get_logger(__name__)

# Load config
config_path = Path(__file__).parent.parent.parent.parent / "configs" / "retrieval.yaml"
try:
    with open(config_path, "r") as f:
        CONFIG = yaml.safe_load(f)
except Exception:
    CONFIG = {"cache_ttl_seconds": 86400}

CACHE_DIR = Path(".cache/retrieval")

def _compute_hash(query: str) -> str:
    normalized = query.lower().strip()
    return hashlib.sha256(normalized.encode()).hexdigest()

async def get_from_cache(query: str, source: str) -> list[dict] | None:
    """Retrieve search results from disk cache if not expired."""
    query_hash = _compute_hash(query)
    cache_file = CACHE_DIR / f"{source}_{query_hash}.json"
    
    if not cache_file.exists():
        return None
        
    try:
        async with aiofiles.open(cache_file, "r", encoding="utf-8") as f:
            data = json.loads(await f.read())
            
        # Check TTL
        if time.time() - data.get("timestamp", 0) > CONFIG.get("cache_ttl_seconds", 86400):
            return None
            
        logger.info("retrieval_cache_hit", query=query, source=source)
        return data.get("results")
    except Exception as e:
        logger.warning("retrieval_cache_read_failed", error=str(e))
        return None

async def save_to_cache(query: str, source: str, results: list[dict]) -> None:
    """Save search results to disk cache."""
    query_hash = _compute_hash(query)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{source}_{query_hash}.json"
    
    data = {
        "query": query,
        "source": source,
        "timestamp": time.time(),
        "results": results
    }
    
    try:
        async with aiofiles.open(cache_file, "w", encoding="utf-8") as f:
            await f.write(json.dumps(data))
    except Exception as e:
        logger.warning("retrieval_cache_write_failed", error=str(e))
