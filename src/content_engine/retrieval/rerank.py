import asyncio
import structlog
import yaml
from pathlib import Path
from pydantic import BaseModel
from typing import List

logger = structlog.get_logger(__name__)

config_path = Path(__file__).parent.parent.parent.parent / "configs" / "retrieval.yaml"
try:
    with open(config_path, "r") as f:
        CONFIG = yaml.safe_load(f)
except Exception:
    CONFIG = {"rerank_top_k": 8}

class RankedChunk(BaseModel):
    text: str
    url: str
    relevance_score: float

# Load model globally to avoid reloading overhead
_encoder = None

def _get_encoder():
    global _encoder
    if _encoder is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Using a small, fast model suitable for reranking
            _encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        except ImportError:
            logger.error("sentence_transformers_not_installed")
            raise
    return _encoder

def get_domain_authority_score(url: str) -> float:
    from urllib.parse import urlparse
    try:
        domain = urlparse(url).netloc.lower()
        path = urlparse(url).path.lower()
    except Exception:
        return 1.0
        
    # Deprioritize listicles and SEO farms
    if "best" in path or "top" in path or "examples" in path:
        return 0.8
        
    # Boost high authority (docs, edu, gov, wikipedia)
    allowlist_high = [".edu", ".gov", "arxiv.org", "wikipedia.org", "docs.", ".readthedocs.io"]
    for pattern in allowlist_high:
        if pattern in domain:
            return 1.2
            
    # Boost mid-tier tech publishers
    allowlist_mid = ["stackoverflow.com", "github.com", "towardsdatascience.com", "infoq.com", "martinfowler.com"]
    for pattern in allowlist_mid:
        if pattern in domain:
            return 1.1
            
    return 1.0

def _rerank_sync(query: str, chunks: List[dict]) -> List[RankedChunk]:
    if not chunks:
        return []
        
    encoder = _get_encoder()
    
    # Extract just the texts
    texts = [c["text"] for c in chunks]
    
    # Compute embeddings
    # query gets a special prefix for BGE models
    query_embedding = encoder.encode(f"Represent this sentence for searching relevant passages: {query}", normalize_embeddings=True)
    chunk_embeddings = encoder.encode(texts, normalize_embeddings=True)
    
    # Calculate cosine similarity (since embeddings are normalized, dot product is cosine similarity)
    # Use standard python/numpy
    scores = (query_embedding @ chunk_embeddings.T).tolist()
    
    ranked_results = []
    for chunk, score in zip(chunks, scores):
        authority_weight = get_domain_authority_score(chunk["url"])
        ranked_results.append(RankedChunk(
            text=chunk["text"],
            url=chunk["url"],
            relevance_score=float(score) * authority_weight
        ))
        
    # Sort by relevance
    ranked_results.sort(key=lambda x: x.relevance_score, reverse=True)
    
    top_k = CONFIG.get("rerank_top_k", 8)
    return ranked_results[:top_k]

async def rerank_chunks(query: str, chunks: List[dict]) -> List[RankedChunk]:
    """Rerank text chunks based on semantic similarity to the query."""
    if not chunks:
        return []
        
    try:
        # Run in thread pool since sentence-transformers is CPU bound
        ranked = await asyncio.to_thread(_rerank_sync, query, chunks)
        return ranked
    except Exception as e:
        logger.error("reranking_failed", error=str(e))
        # Fallback to returning them unranked or empty
        return []
