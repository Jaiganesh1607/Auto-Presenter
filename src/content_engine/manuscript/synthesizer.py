import json
import structlog
import asyncio
from pydantic import BaseModel
from typing import List, Dict, Optional
from src.content_engine.outline.templates import SectionSpec
from src.content_engine.claims.extractor import Claim
from src.content_engine.llm.client import LLMClient
from src.content_engine.llm.schemas import ChatRequest, ChatMessage
from src.content_engine.verification.citation_gate import check_citation_presence

logger = structlog.get_logger(__name__)

class Section(BaseModel):
    section_id: str
    title: str
    body_markdown: str
    source_manifest: Dict[str, str] # e.g. "[S1]" -> "url or document path"
    flags: List[str] = []

class Manuscript(BaseModel):
    sections: List[Section]

async def _synthesize_section(spec: SectionSpec, claims: List[Claim], llm_client: LLMClient) -> Section:
    if not claims:
        logger.warning("section_insufficient_coverage_no_claims", section=spec.id)
        return Section(
            section_id=spec.id,
            title=spec.display_title,
            body_markdown="",
            source_manifest={},
            flags=["insufficient_coverage"]
        )

    # Group claims by unique source_id to prevent deduplication failures
    unique_sources = []
    source_to_claims = {}
    if claims:
        for claim in claims:
            if claim.source_id not in source_to_claims:
                source_to_claims[claim.source_id] = []
                unique_sources.append(claim.source_id)
            source_to_claims[claim.source_id].append(claim.text)
            
    claims_context = ""
    source_manifest = {}
    
    if unique_sources:
        for i, source_id in enumerate(unique_sources, 1):
            tag = f"[S{i}]"
            source_manifest[tag] = source_id
            claims_context += f"Source {tag}:\n"
            for text in source_to_claims[source_id]:
                claims_context += f"- {text}\n"
            claims_context += "\n"
        
    prompt = f"""You are an expert technical writer. Write the comprehensive section "{spec.display_title}" for an article.
    
Section Goal: {spec.scope_description}

Below are the ONLY verified facts you can use:
{claims_context if claims_context else "(No specific retrieved facts available.)"}

Instructions:
1. Before writing, analyze the provided facts and logically group them by sub-theme if they span different aspects of the topic.
2. Write each sub-theme as its own distinct, short paragraph rather than blending unrelated claims into one long paragraph using generic transition words (e.g. "Furthermore", "Additionally").
3. If the claims do not share a strong common thread to form a single continuous narrative, do NOT force false continuity. Write clearly separated sub-groups based strictly on what facts are available.
4. Incorporate the provided verified facts precisely. For ANY claim you make, you MUST append its exact source tag (e.g. [S1]).
5. CRITICAL: Do NOT fabricate content. Never invent generic filler to stitch paragraphs together.
6. CRITICAL: Do NOT use markdown headers (e.g. ##, ###) inside the body text. Output flowing paragraphs only. Do not invent sub-sections.
7. Do not hallucinate URLs or source tags.

Output strictly as a JSON object with a single key "body_markdown" containing the paragraph text.
"""
    
    request = ChatRequest(
        model=llm_client.provider.model_name,
        messages=[ChatMessage(role="user", content=prompt)],
        temperature=0.2,
        response_format={"type": "json_object"}
    )
    
    try:
        response = await llm_client.complete(request)
        import json_repair
        data = json_repair.repair_json(response.content, return_objects=True)
        if not isinstance(data, dict):
            data = {}
        body_markdown = data.get("body_markdown", "")
        
        # We could run citation_gate against pseudo-claims extracted from body_markdown to verify tags aren't dropped,
        # but a simple regex or string check is enough to confirm tags exist in the output.
        flags = []
        
        # 1. Empty body check
        if not body_markdown.strip():
            flags.append("generation_failed")
            flags.append("insufficient_coverage")
            logger.warning("section_empty_body", section=spec.id)
            return Section(
                section_id=spec.id,
                title=spec.display_title,
                body_markdown="",
                source_manifest={},
                flags=flags
            )
            
        # 1b. Header rejection/stripping
        import re
        if re.search(r'^#{1,6}\s+', body_markdown, flags=re.MULTILINE):
            logger.warning("section_contains_headers", section=spec.id)
            body_markdown = re.sub(r'^#{1,6}\s+(.*)$', r'\1', body_markdown, flags=re.MULTILINE)
            flags.append("stripped_headers")

        # 2. Citation check
        for tag in source_manifest.keys():
            if tag not in body_markdown:
                logger.debug("citation_tag_dropped", tag=tag, section=spec.id)
                
        has_any_tag = any(tag in body_markdown for tag in source_manifest.keys())
        if not has_any_tag:
            # If there are no tags, it means the content is fabricated/unsupported.
            flags.append("insufficient_coverage")
            logger.warning("section_insufficient_coverage_no_tags", section=spec.id)
            
        return Section(
            section_id=spec.id,
            title=spec.display_title,
            body_markdown=body_markdown,
            source_manifest=source_manifest,
            flags=flags
        )
    except Exception as e:
        logger.error("synthesize_section_failed", section=spec.id, error=str(e))
        return Section(
            section_id=spec.id,
            title=spec.display_title,
            body_markdown=f"Failed to generate section: {str(e)}",
            source_manifest={},
            flags=["generation_failed"]
        )

async def synthesize_manuscript(outline: List[SectionSpec], verified_claims: Dict[str, List[Claim]], llm_client: LLMClient) -> Manuscript:
    """Synthesize the final manuscript by generating text for each section using verified claims."""
    
    # Process sections concurrently
    tasks = []
    for spec in outline:
        claims = verified_claims.get(spec.id, [])
        tasks.append(_synthesize_section(spec, claims, llm_client))
        
    sections = await asyncio.gather(*tasks)
    
    sections_list = list(sections)
    
    # FIX 2: Hard invariant check
    for sec in sections_list:
        if sec.source_manifest and not sec.body_markdown.strip():
            logger.error("invariant_violation: empty body with sources", section=sec.section_id)
            sec.flags.append("generation_failed")
            sec.source_manifest = {}
            if "insufficient_coverage" not in sec.flags:
                sec.flags.append("insufficient_coverage")
                
    return Manuscript(sections=sections_list)
