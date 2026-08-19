import structlog
from typing import List
from src.content_engine.claims.extractor import Claim
from src.content_engine.retrieval.rerank import get_domain_authority_score

logger = structlog.get_logger(__name__)

def dedup_claims(claims: List[Claim]) -> List[Claim]:
    if not claims:
        return []
        
    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    except ImportError:
        logger.warning("sentence_transformers_missing_for_dedup")
        return claims
        
    import numpy as np
    
    texts = [c.text for c in claims]
    embeds = encoder.encode(texts, normalize_embeddings=True)
    
    similarity = np.dot(embeds, embeds.T)
    
    keep_indices = set(range(len(claims)))
    
    for i in range(len(claims)):
        if i not in keep_indices:
            continue
        for j in range(i + 1, len(claims)):
            if j not in keep_indices:
                continue
                
            if similarity[i, j] > 0.90:
                # They are near duplicates. Decide which to keep.
                score_i = get_domain_authority_score(claims[i].source_id)
                score_j = get_domain_authority_score(claims[j].source_id)
                
                if score_i > score_j:
                    keep_indices.remove(j)
                elif score_j > score_i:
                    keep_indices.remove(i)
                    break # i is removed, move to next i
                else:
                    # Prefer shorter/more concise
                    if len(claims[i].text) <= len(claims[j].text):
                        keep_indices.remove(j)
                    else:
                        keep_indices.remove(i)
                        break
                        
    deduped = [claims[idx] for idx in sorted(list(keep_indices))]
    if len(deduped) < len(claims):
        logger.info("claims_deduped", original=len(claims), deduped=len(deduped))
    return deduped
