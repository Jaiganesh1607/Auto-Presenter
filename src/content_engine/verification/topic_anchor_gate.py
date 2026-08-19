import structlog
from typing import List
from src.content_engine.claims.extractor import Claim, FlaggedClaim

logger = structlog.get_logger(__name__)

_encoder = None

def _get_encoder():
    global _encoder
    if _encoder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        except ImportError:
            logger.error("sentence_transformers_missing")
            raise
    return _encoder

def check_topic_anchor(claims: List[Claim], root_topic: str, min_score: float = 0.3) -> List[FlaggedClaim]:
    """
    Catch gross topic drift by checking if the claim is somewhat related to the original root topic.
    """
    if not claims or not root_topic:
        return []
        
    try:
        encoder = _get_encoder()
        import numpy as np
        
        topic_embed = encoder.encode([root_topic], normalize_embeddings=True)
        claim_embeds = encoder.encode([c.text for c in claims], normalize_embeddings=True)
        
        similarity = np.dot(claim_embeds, topic_embed.T).flatten()
        
        flagged = []
        for i, claim in enumerate(claims):
            score = float(similarity[i])
            if score < min_score:
                flagged.append(FlaggedClaim(
                    text=claim.text,
                    source_id=claim.source_id,
                    source_chunk_text=claim.source_chunk_text,
                    entailment_score=score,  # Store similarity here for debugging
                    flag_reason=f"Topic drift: similarity to '{root_topic}' is {score:.2f} < {min_score}"
                ))
        return flagged
    except Exception as e:
        logger.error("topic_anchor_scoring_failed", error=str(e))
        return []

def check_named_entity_drift(claim_text: str, primary_subject: str, section_id: str) -> bool:
    """
    Returns True if the claim is valid (no drift), False if it drifted to a different named entity.
    """
    if not primary_subject:
        return True
        
    # Allow comparisons
    if "comparison" in section_id.lower() or "vs" in section_id.lower():
        return True
        
    # Check if primary subject (case insensitive) is in claim
    if primary_subject.lower() in claim_text.lower():
        return True
        
    # If the subject is entirely missing from the claim, it's drifting (e.g. talking about Kubernetes entirely)
    return False
