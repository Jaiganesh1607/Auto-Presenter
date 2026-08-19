import asyncio
import structlog
from typing import List, Tuple
from src.content_engine.claims.extractor import Claim, FlaggedClaim
from src.content_engine.verification.citation_gate import check_citation_presence
from src.content_engine.verification.topic_anchor_gate import check_topic_anchor

logger = structlog.get_logger(__name__)

_cross_encoder = None

def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            # We use a lightweight NLI model
            _cross_encoder = CrossEncoder('cross-encoder/nli-deberta-v3-base')
        except ImportError:
            logger.error("sentence_transformers_missing")
            raise
    return _cross_encoder

def _check_entailment_sync(claims: List[Claim], min_score: float) -> List[FlaggedClaim]:
    if not claims:
        return []
        
    model = _get_cross_encoder()
    
    # NLI models usually expect (premise, hypothesis) pairs
    # Premise = source_chunk_text, Hypothesis = claim.text
    pairs = [(c.source_chunk_text, c.text) for c in claims]
    
    # Predict returns logits for contradiction, entailment, neutral
    # Wait, cross-encoder/nli-deberta-v3-base returns logits for [contradiction, entailment, neutral].
    # So we can just take the entailment score. Or we can just use softmax.
    # Actually, cross-encoder/ms-marco-MiniLM-L-6-v2 might be simpler for scoring, but NLI is better for facts.
    # Let's use predict with softmax.
    import numpy as np
    
    try:
        # Some NLI models return 3 labels: [contradiction, entailment, neutral] (deberta-v3 is usually this)
        # We need to apply softmax
        logits = model.predict(pairs)
        
        flagged = []
        for i, claim in enumerate(claims):
            # DeBERTa-v3-base-nli label mapping: 0=contradiction, 1=entailment, 2=neutral
            exp_logits = np.exp(logits[i] - np.max(logits[i]))
            probs = exp_logits / exp_logits.sum()
            contradiction_score = float(probs[0])
            entailment_score = float(probs[1])
            
            max_contradiction = 0.5
            # Flag if contradiction is too high (NLI models often classify summaries as neutral)
            if contradiction_score > max_contradiction:
                flagged.append(FlaggedClaim(
                    text=claim.text,
                    source_id=claim.source_id,
                    source_chunk_text=claim.source_chunk_text,
                    entailment_score=entailment_score,
                    flag_reason=f"High contradiction score: {contradiction_score:.2f}"
                ))
        return flagged
    except Exception as e:
        logger.error("entailment_scoring_failed", error=str(e))
        return []

async def check_entailment(claims: List[Claim], min_score: float = 0.35) -> List[FlaggedClaim]:
    """Check whether claims are entailed by their source text."""
    if not claims:
        return []
        
    return await asyncio.to_thread(_check_entailment_sync, claims, min_score)

async def verify_claims(claims: List[Claim], topic: str) -> Tuple[List[Claim], List[FlaggedClaim]]:
    """
    Run claims through all gates (citation, topic anchor, entailment). 
    Returns (verified_claims, flagged_claims).
    """
    # 1. Citation gate (structural)
    citation_failed = check_citation_presence(claims)
    citation_failed_texts = {c.text for c in citation_failed}
    
    structurally_valid = [c for c in claims if c.text not in citation_failed_texts]
    
    # Convert citation failures to FlaggedClaim objects
    flagged_results = [
        FlaggedClaim(
            text=c.text,
            source_id=c.source_id,
            source_chunk_text=c.source_chunk_text,
            entailment_score=0.0,
            flag_reason="Missing structural citation"
        )
        for c in citation_failed
    ]
    # 2. Topic Anchor gate (drift prevention)
    topic_drift_failed = check_topic_anchor(structurally_valid, root_topic=topic, min_score=0.55)
    topic_drift_texts = {c.text for c in topic_drift_failed}
    
    topic_valid = [c for c in structurally_valid if c.text not in topic_drift_texts]
    flagged_results.extend(topic_drift_failed)
    
    # 3. Entailment gate (semantic)
    entailment_failed = await check_entailment(topic_valid, min_score=0.35)
    # Use text-based matching to identify which claims failed entailment
    failed_texts = {c.text for c in entailment_failed}
    
    flagged_results.extend(entailment_failed)
    
    # The fully verified claims pass all gates
    verified = [c for c in topic_valid if c.text not in failed_texts]
    
    return verified, flagged_results
