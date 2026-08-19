import asyncio
import structlog
import yaml
from pathlib import Path
import httpx
from trafilatura import fetch_url, extract

logger = structlog.get_logger(__name__)

config_path = Path(__file__).parent.parent.parent.parent / "configs" / "retrieval.yaml"
try:
    with open(config_path, "r") as f:
        CONFIG = yaml.safe_load(f)
except Exception:
    CONFIG = {"fetcher_timeout": 15}

def _clean_bibliography(text: str) -> str:
    cutoff_headers = [
        "\nReferences\n", "\nBibliography\n", "\nSee also\n", "\nFurther reading\n", "\nExternal links\n",
        "\nNotes\n", "References\n", "Bibliography\n"
    ]
    
    # We want the first occurrence, but only if it seems like a section header (e.g., has newlines around it)
    earliest_idx = len(text)
    for header in cutoff_headers:
        idx = text.find(header)
        if idx != -1 and idx > len(text) * 0.4:
            earliest_idx = min(earliest_idx, idx)
            
    if earliest_idx < len(text):
        return text[:earliest_idx]
    return text

def _fetch_and_extract_sync(url: str) -> str:
    # trafilatura fetch_url uses urllib, it is blocking
    downloaded = fetch_url(url)
    if downloaded is None:
        return ""
        
    text = extract(downloaded, include_links=False, include_images=False, include_tables=True)
    if not text:
        return ""
        
    return _clean_bibliography(text)

async def fetch_page_content(url: str, max_retries: int = 2) -> str:
    """Fetch URL and extract main content using trafilatura."""
    timeout = CONFIG.get("fetcher_timeout", 20)  # Increased default timeout to 20s
    
    logger.info("fetch_started", url=url)
    
    for attempt in range(max_retries):
        try:
            content = await asyncio.wait_for(
                asyncio.to_thread(_fetch_and_extract_sync, url),
                timeout=timeout
            )
            
            if not content:
                logger.info("fetch_success_zero_content", url=url)
                
            return content
        except asyncio.TimeoutError as e:
            if attempt < max_retries - 1:
                logger.warning("fetch_timeout_retrying", url=url, attempt=attempt+1)
                await asyncio.sleep(1.5)
            else:
                logger.warning("fetch_timeout_final", url=url)
                raise e
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning("fetch_failed_retrying", url=url, attempt=attempt+1, error=str(e))
                await asyncio.sleep(1.5)
            else:
                logger.warning("fetch_failed_final", url=url, error=str(e))
                raise e
