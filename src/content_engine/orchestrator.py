import time
import structlog
from pathlib import Path
from typing import List, Optional

from src.content_engine.llm.client import LLMClient
from src.content_engine.ingestion.parser import parse_document
from src.content_engine.retrieval.genre_classifier import classify_genre
from src.content_engine.outline.templates import GENRE_TEMPLATES
from src.content_engine.retrieval.perspective_planner import generate_perspectives, Perspective
from src.content_engine.retrieval.orchestrator import parallel_retrieve
from src.content_engine.claims.extractor import batch_extract_claims
from src.content_engine.verification.entailment_gate import verify_claims
from src.content_engine.manuscript.synthesizer import synthesize_manuscript, Manuscript

logger = structlog.get_logger(__name__)

async def run_content_engine(topic: str, uploaded_files: Optional[List[Path]], llm_client: LLMClient, fast_mode: bool = True) -> Manuscript:
    """Run the end-to-end content engine pipeline."""
    
    logger.info("pipeline_started", topic=topic)
    start_time = time.time()
    
    try:
        # 1. Ingestion (Optional)
        ingested_chunks = []
        if uploaded_files:
            stage_start = time.time()
            for path in uploaded_files:
                doc = await parse_document(path)
                if doc.markdown_text:
                    ingested_chunks.append({
                        "url": f"file://{path.name}",
                        "text": doc.markdown_text
                    })
            logger.info("stage_ingestion_complete", elapsed=time.time() - stage_start)
            
        # 2. Genre Classification
        stage_start = time.time()
        genre = await classify_genre(topic, llm_client)
        outline = GENRE_TEMPLATES.get(genre, GENRE_TEMPLATES["explainer"])
        logger.info("stage_genre_classification_complete", genre=genre, elapsed=time.time() - stage_start)
        
        # 3. Perspective Generation
        stage_start = time.time()
        perspectives = await generate_perspectives(topic, genre, llm_client)
        logger.info("stage_perspective_generation_complete", perspectives_count=len(perspectives), elapsed=time.time() - stage_start)
        
        # 4. Parallel Retrieval & Reranking
        stage_start = time.time()
        ranked_chunks = await parallel_retrieve(perspectives, topic)
        
        # Inject ingested local docs as top priority chunks
        from src.content_engine.retrieval.rerank import RankedChunk
        for chunk in ingested_chunks:
            ranked_chunks.insert(0, RankedChunk(text=chunk["text"], url=chunk["url"], relevance_score=1.0))
            
        logger.info("stage_parallel_retrieval_complete", chunks_count=len(ranked_chunks), elapsed=time.time() - stage_start)
        
        # 5. Claim Extraction
        stage_start = time.time()
        chunk_limit = 5 if fast_mode else 10
        raw_claims = await batch_extract_claims(ranked_chunks[:chunk_limit], llm_client)
        logger.info("stage_claim_extraction_complete", claims_count=len(raw_claims), elapsed=time.time() - stage_start)
        
        # 6. Verification Gates
        stage_start = time.time()
        verified_claims, flagged_claims = await verify_claims(raw_claims, topic)
        logger.info("stage_verification_complete", verified=len(verified_claims), flagged=len(flagged_claims), elapsed=time.time() - stage_start)
        
        # 6.5 Deduplicate Claims (MOVED TO END)
        
        # 6.6 Extract Primary Subject for Entity Drift Check
        stage_start = time.time()
        subject_prompt = f"What is the single primary named subject (proper noun) of this topic: '{topic}'? Return ONLY the noun string, no punctuation or explanation."
        try:
            from src.content_engine.llm.schemas import ChatRequest, ChatMessage
            sub_req = ChatRequest(model=llm_client.provider.model_name, messages=[ChatMessage(role="user", content=subject_prompt)], temperature=0.1)
            sub_res = await llm_client.complete(sub_req)
            primary_subject = sub_res.content.strip().strip("'\"").strip()
            if len(primary_subject.split()) > 3:
                primary_subject = topic.split()[-1]
        except Exception:
            primary_subject = topic.split()[-1]
        logger.info("stage_primary_subject_extracted", subject=primary_subject, elapsed=time.time() - stage_start)

        # 7. Section Assignment (Simple similarity via embedding)
        stage_start = time.time()
        section_claims = {spec.id: [] for spec in outline}
        if raw_claims:
            try:
                from sentence_transformers import SentenceTransformer
                encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
                section_texts = [f"{spec.display_title}: {spec.scope_description}" for spec in outline]
                section_embeds = encoder.encode(section_texts, normalize_embeddings=True)
                
                # DIAGNOSTIC: Embed all raw claims to see where they would have gone
                raw_claim_embeds = encoder.encode([c.text for c in raw_claims], normalize_embeddings=True)
                import numpy as np
                raw_similarity = np.dot(raw_claim_embeds, section_embeds.T)
                raw_best_sections = np.argmax(raw_similarity, axis=1)
                raw_best_scores = np.max(raw_similarity, axis=1)
                
                # Build diagnostic stats per section
                diagnostic_stats = {spec.id: {"extracted": 0, "citation_failed": 0, "entailment_failed": 0, "dedup_failed": 0, "relevance_failed": 0, "survived": 0} for spec in outline}
                
                flagged_reasons = {c.text: c.flag_reason for c in flagged_claims}
                verified_texts = {c.text for c in verified_claims}
                
                from src.content_engine.verification.topic_anchor_gate import check_named_entity_drift
                
                for idx, c in enumerate(raw_claims):
                    sec_idx = raw_best_sections[idx]
                    sec_id = outline[sec_idx].id
                    score = float(raw_best_scores[idx])
                    
                    diagnostic_stats[sec_id]["extracted"] += 1
                    
                    if c.text in flagged_reasons:
                        reason = flagged_reasons[c.text]
                        if "citation" in reason.lower():
                            diagnostic_stats[sec_id]["citation_failed"] += 1
                        else:
                            diagnostic_stats[sec_id]["entailment_failed"] += 1
                    elif not check_named_entity_drift(c.text, primary_subject, sec_id):
                        # Use relevance_failed for now since it's an assignment-stage rejection
                        diagnostic_stats[sec_id]["relevance_failed"] += 1
                        logger.debug("named_entity_drift_caught", claim=c.text, subject=primary_subject, section=sec_id)
                    elif c.text not in verified_texts:
                        diagnostic_stats[sec_id]["dedup_failed"] += 1
                    elif score < 0.65:
                        diagnostic_stats[sec_id]["relevance_failed"] += 1
                    else:
                        diagnostic_stats[sec_id]["survived"] += 1
                        section_claims[sec_id].append(c)
                
                for spec in outline:
                    stats = diagnostic_stats[spec.id]
                    logger.info("first_pass_diagnostics", 
                                section=spec.id, 
                                extracted=stats["extracted"], 
                                citation_drop=stats["citation_failed"],
                                entailment_drop=stats["entailment_failed"],
                                dedup_drop=stats["dedup_failed"],
                                relevance_drop=stats["relevance_failed"],
                                survived=stats["survived"])
                                
            except Exception as e:
                logger.error("section_assignment_failed", error=str(e))
                if verified_claims:
                    section_claims[outline[0].id] = verified_claims
                
        # Retry logic for sections with < 3 claims (Disabled in fast_mode to save massive amounts of time)
        if not fast_mode:
            for spec in outline:
                if len(section_claims[spec.id]) < 3:
                    logger.info("insufficient_claims_triggering_retry", section=spec.id, count=len(section_claims[spec.id]))
                    
                    # Ask the LLM for broadened queries
                    retry_prompt = f"""We are researching '{topic}' for the section '{spec.display_title}'.
The scope of this section is: {spec.scope_description}
Our previous specific searches yielded too few facts. Generate exactly 2 broadened, simplified search queries to find general but factual information for this section.
CRITICAL: Disambiguate generic terms against the root topic '{topic}'. Do not leave terms ambiguous (e.g. if the topic is Docker, write 'software containerization history', not just 'containerization history').
Do not use overly complex or narrow terms. Keep them broad but relevant.
Return strictly a JSON object: {{"queries": ["query 1", "query 2"]}}"""

                    from src.content_engine.llm.schemas import ChatRequest, ChatMessage
                    import json_repair
                    try:
                        retry_req = ChatRequest(model=llm_client.provider.model_name, messages=[ChatMessage(role="user", content=retry_prompt)], temperature=0.5, response_format={"type": "json_object"})
                        retry_res = await llm_client.complete(retry_req)
                        retry_data = json_repair.repair_json(retry_res.content, return_objects=True)
                        broadened_queries = retry_data.get("queries", [f"{topic} {spec.display_title} overview"])
                    except Exception as e:
                        logger.error("retry_query_generation_failed", error=str(e))
                        broadened_queries = [f"{topic} overview", f"{topic} {spec.display_title}"]
                    
                    retry_perspectives = [Perspective(role="Broadener", queries=broadened_queries)]
                    retry_chunks = await parallel_retrieve(retry_perspectives, topic)
                    
                    # Fetch only from new chunks not in original pool
                    existing_urls = set(c.source_id for c in section_claims[spec.id]) if verified_claims else set()
                    new_chunks = [c for c in retry_chunks if c.url not in existing_urls][:5]
                    
                    if new_chunks:
                        retry_raw_claims = await batch_extract_claims(new_chunks, llm_client)
                        retry_verified, _ = await verify_claims(retry_raw_claims, topic)
                        
                        # Apply entity drift check to retry claims
                        safe_retry_claims = []
                        for c in retry_verified:
                            if check_named_entity_drift(c.text, primary_subject, spec.id):
                                safe_retry_claims.append(c)
                            else:
                                logger.debug("named_entity_drift_caught_in_retry", claim=c.text, subject=primary_subject, section=spec.id)
                                
                        section_claims[spec.id].extend(safe_retry_claims)
                        logger.info("retry_complete", section=spec.id, new_claims_added=len(safe_retry_claims), total_now=len(section_claims[spec.id]))
                    
        # 7.5 Global Deduplication (Across all sections and retries)
        from src.content_engine.claims.dedup import dedup_claims
        all_claims = []
        for claims_list in section_claims.values():
            all_claims.extend(claims_list)
            
        if all_claims:
            deduped_all = dedup_claims(all_claims)
            deduped_texts = {c.text for c in deduped_all}
            
            new_section_claims = {spec.id: [] for spec in outline}
            assigned_texts = set()
            for spec in outline:
                for c in section_claims[spec.id]:
                    if c.text in deduped_texts and c.text not in assigned_texts:
                        new_section_claims[spec.id].append(c)
                        assigned_texts.add(c.text)
            section_claims = new_section_claims
            
        logger.info("stage_assignment_complete", elapsed=time.time() - stage_start)
        
        # 8. Manuscript Synthesis
        stage_start = time.time()
        manuscript = await synthesize_manuscript(outline, section_claims, llm_client)
        logger.info("stage_synthesis_complete", elapsed=time.time() - stage_start)
        
        logger.info("pipeline_finished", total_elapsed=time.time() - start_time)
        return manuscript
        
    except Exception as e:
        logger.error("pipeline_failed", error=str(e), exc_info=True)
        # Return partial/empty manuscript on catastrophic failure
        from src.content_engine.manuscript.synthesizer import Manuscript, Section
        return Manuscript(sections=[
            Section(section_id="error", title="Pipeline Error", body_markdown=f"An error occurred: {str(e)}", source_manifest={})
        ])
