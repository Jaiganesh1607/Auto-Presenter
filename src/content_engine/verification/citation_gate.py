from typing import List, Tuple
import structlog
from src.content_engine.claims.extractor import Claim

logger = structlog.get_logger(__name__)

def check_citation_presence(claims: List[Claim]) -> List[Claim]:
    """
    Returns the subset of claims that fail the basic structural citation check
    (missing or whitespace-only source_id or source_chunk_text).
    """
    flagged = []
    for claim in claims:
        if not claim.source_id or not claim.source_id.strip():
            flagged.append(claim)
            continue
            
        if not claim.source_chunk_text or not claim.source_chunk_text.strip():
            flagged.append(claim)
            continue
            
    if flagged:
        logger.info("citation_gate_flagged", count=len(flagged))
        
    return flagged
