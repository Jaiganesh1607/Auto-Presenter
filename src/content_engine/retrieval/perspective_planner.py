import json
import structlog
from pydantic import BaseModel
from src.content_engine.llm.client import LLMClient
from src.content_engine.llm.schemas import ChatRequest, ChatMessage

logger = structlog.get_logger(__name__)

class Perspective(BaseModel):
    role: str
    queries: list[str]

class PerspectivesResponse(BaseModel):
    perspectives: list[Perspective]

from src.content_engine.outline.templates import GENRE_TEMPLATES

async def generate_perspectives(topic: str, genre: str, llm_client: LLMClient) -> list[Perspective]:
    """Generate distinct research perspectives and search queries for a topic."""
    
    outline = GENRE_TEMPLATES.get(genre, GENRE_TEMPLATES["explainer"])
    outline_text = "\n".join([f"- {s.display_title}: {s.scope_description}" for s in outline])
    
    prompt = f"""You are a research planner preparing to investigate the topic "{topic}" for a "{genre}" article.
The article will be structured with the following sections and scopes:
{outline_text}

You must adopt exactly 3 distinct expert roles or perspectives that would approach this topic differently.
For EACH perspective, generate exactly 3 highly specific search queries that this expert would use to find relevant information. Ensure that collectively, your queries provide coverage for ALL the sections listed above (especially the problem/limitations it solves).

CRITICAL CONSTRAINTS:
1. You must generate exactly 3 perspectives, with exactly 3 queries each.
2. The perspectives must be completely non-overlapping in what they would search for.
3. Every search query must be distinct. No identical queries across different perspectives.
4. Queries MUST be highly specific and technical. Do not just use the general topic name. Include specific mechanisms, proper nouns, and technical terms relevant to the section scope. Think like an expert typing a precise search into a search engine.
5. DISAMBIGUATE GENERIC TERMS: Explicitly tie queries back to the root topic "{topic}". For example, if the topic is "Docker", a query like "history of containerization" is BAD because it might match shipping containers. A GOOD query is "history of software containerization Docker".
6. Return the result strictly as a JSON object with a single "perspectives" array containing objects with "role" and "queries" (a list of strings).

Example of BAD generic queries:
["Docker architecture", "Docker core vocabulary", "What is the problem with old systems", "TCP/IP vs HTTP/2"]

Example of GOOD specific queries for those same concepts:
["Docker containers vs virtual machines kernel isolation namespaces cgroups", "Dockerfile image container registry build process definition", "dependency hell before containers", "why do networks need layering", "TCP/IP compared to HTTP/2 performance tradeoffs"]

Example output format:
{{
  "perspectives": [
    {{
      "role": "Systems Architect",
      "queries": ["Docker containers vs virtual machines kernel isolation namespaces cgroups"]
    }},
    {{
      "role": "DevOps Engineer",
      "queries": ["Dockerfile image container registry build process definition"]
    }}
  ]
}}
"""
    
    request = ChatRequest(
        model=llm_client.provider.model_name,
        messages=[ChatMessage(role="user", content=prompt)],
        temperature=0.7, # slightly higher to encourage varied queries
        response_format={"type": "json_object"}
    )
    
    try:
        response = await llm_client.complete(request)
        import json_repair
        data = json_repair.repair_json(response.content, return_objects=True)
        if not isinstance(data, dict):
            data = {}
        raw_perspectives = data.get("perspectives", [])
        
        results = []
        for p in raw_perspectives:
            results.append(Perspective(
                role=p.get("role", "General Researcher"),
                queries=p.get("queries", [])
            ))
            
        # Ensure we always return at least something if LLM hallucinated the structure
        if len(results) == 0:
            logger.warning("perspective_planner_empty_response", fallback="using_default")
            return [Perspective(role="Default Researcher", queries=[topic])]
            
        return results
    except Exception as e:
        logger.error("perspective_planner_failed", error=str(e))
        # Fallback to single perspective to prevent crashing pipeline
        return [Perspective(role="Fallback Researcher", queries=[topic])]
