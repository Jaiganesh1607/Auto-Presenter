import json
import asyncio
import structlog
from pydantic import BaseModel
from typing import List
from src.content_engine.retrieval.rerank import RankedChunk
from src.content_engine.llm.client import LLMClient
from src.content_engine.llm.schemas import ChatRequest, ChatMessage

logger = structlog.get_logger(__name__)

class Claim(BaseModel):
    text: str
    source_id: str
    source_chunk_text: str

class FlaggedClaim(Claim):
    entailment_score: float
    flag_reason: str

async def extract_claims(chunk: RankedChunk, llm_client: LLMClient) -> List[Claim]:
    """Extract atomic factual claims from a single text chunk using an LLM."""
    
    prompt = f"""You are a strict fact extractor. Read the following text snippet and extract independent, atomic factual claims.

RULES:
1. Extract a MAXIMUM of 3 most important factual claims. Do not extract trivial details.
2. Each claim must be a single, complete sentence that makes sense out of context.
3. DO NOT add any information, context, or editorializing not explicitly present in the source text.
4. CRITICAL: Skip and reject claims that are vague meta-commentary, promotional filler, or lack concrete technical substance (e.g. "X is essential", "understanding Y is vital"). Every claim MUST contain a concrete noun, number, named technology, or specific mechanism.
5. CRITICAL: Skip and reject claims that use first-person pronouns (e.g. "we", "our", "I", "my") to refer to the source's own company, product, or team. These are marketing statements, not general facts.
6. Output strictly as JSON in the format: {{"claims": ["fact 1", "fact 2"]}}

Source text:
{chunk.text}
"""
    
    request = ChatRequest(
        model=llm_client.provider.model_name,
        messages=[ChatMessage(role="user", content=prompt)],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    
    try:
        response = await llm_client.complete(request)
        import json_repair
        data = json_repair.repair_json(response.content, return_objects=True)
        if not isinstance(data, dict):
            data = {}
        raw_claims = data.get("claims", [])
        
        hedge_phrases = [
            "is essential", "cannot be overstated", "plays a crucial role", 
            "is vital for", "important to note", "deep understanding",
            "is a must", "game changer", "highly recommended",
            "it is important", "crucial to understand", "underlying technologies"
        ]
        
        claims = []
        for text in raw_claims:
            text_lower = text.lower()
            if any(hedge in text_lower for hedge in hedge_phrases):
                logger.debug("claim_filtered_as_hedge", text=text)
                continue
                
            # Filter first-person promotional claims
            if any(word in text_lower.split() for word in ["we", "our", "my", "i"]):
                logger.debug("claim_filtered_as_promotional", text=text)
                continue
                
            claims.append(Claim(
                text=text,
                source_id=chunk.url,
                source_chunk_text=chunk.text
            ))
        return claims
    except Exception as e:
        logger.error("claim_extraction_failed", url=chunk.url, error=str(e))
        return []

async def batch_extract_claims(chunks: List[RankedChunk], llm_client: LLMClient, concurrency: int = 10) -> List[Claim]:
    """Extract claims from multiple chunks concurrently with a limit."""
    sem = asyncio.Semaphore(concurrency)
    
    async def bounded_extract(chunk):
        async with sem:
            return await extract_claims(chunk, llm_client)
            
    tasks = [bounded_extract(c) for c in chunks]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    all_claims = []
    for res in results:
        if isinstance(res, list):
            all_claims.extend(res)
        elif isinstance(res, Exception):
            logger.warning("batch_extract_task_failed", error=str(res))
            
    return all_claims
